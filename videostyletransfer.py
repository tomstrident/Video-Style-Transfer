

import os
import cv2
import time
import imageio

import numpy as np
import tensorflow as tf
import opticalflow as of

from scipy.io import loadmat
#from diffusion_inpainting import image_diffusion
from scipy.ndimage.interpolation import map_coordinates

import network

#source activate ...
#export PYTHONPATH=/home/tom/Downloads/flownet2-master/python:$PYTHONPATH
#export PYTHONPATH=/home/tom/Documents/PythonProjects/VST/flownet2-master/python:$PYTHONPATH
#echo $PYTHONPATH

class VideoStyleTransferModule:
  
  def __init__(self, 
               content, style, 
               alpha=1, beta=0.1, gamma=200,
               frame_size=None,#[120, 160], #None
               precalc_flow=None,
               num_iters=50, pyramid_layers=4, opt_type='scipy',
               inpaint=None,
               model_path='Models/imagenet-vgg-verydeep-19.mat'):
    
    self.imgnet_mean = np.array([123.68, 116.779, 103.939])
    self.num_layers = 36
    self.frame_size = frame_size
    self.external_flow = False
    
    self.content_raw = []
    self.content_images = []
    self.style_image = []
    
    if self.external_flow:
      self.server = network.network()
      self.server.host_setup()
      self.server.host_connect()
    
    #self.tf_def_sess = tf.Session()
    
    if type(content) is str:
      reader = imageio.get_reader(content)
      for im in reader:
        self.content_raw.append(im)
        self.content_images.append(self.prep_image(im))
    elif type(content) is list: 
      for im in content:
        self.content_raw.append(im)
        self.content_images.append(self.prep_image(im))
    else:
      raise ValueError('videostyletransfer: content input ivalid')
    
    if type(style) is str:
      style_image = imageio.imread(style)
      self.style_image = self.prep_image(style_image)
    elif type(style) is np.ndarray: 
      self.style_image =  self.prep_image(style)
    else:
      raise ValueError('videostyletransfer: style input ivalid')
    
    self.output_dir = os.getcwd() + '/output'
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
    
    self.image_shape = self.content_images[0].shape
    self.weight_shape = self.image_shape[:3] + (1,)

    self.input_image = np.empty(self.image_shape)

    # optical flow (PWC Net)
    #pwc_model_path = os.getcwd() + '/PWC_Net/model_3000epoch/model_3007.ckpt'
    #self.pwc_net = pwc.PWC(self.image_shape, pwc_model_path)

    # temporal
    
    self.J = [1, 2, 4]
    self.c_long = [np.zeros(self.weight_shape) for _ in self.J]
    self.x_w = [np.zeros(self.image_shape) for _ in self.J]
    
    self.past_frames = []
    self.past_styled = []
    self.flow_forward = []
    self.flow_backward = []

    self.warped_image = []
    self.weight_image = []

    # image pyramid
    self.pyramid_layers = pyramid_layers#5
    self.p_shapes = self.pyramid_shapes(self.image_shape[1:3])
    self.p_style = self.image_pyramid(self.style_image)
    
    n_f = len(self.J)
    self.w = [np.zeros((n_f,) + p_shape + (3,)) for p_shape in self.p_shapes]
    self.c = [np.zeros((n_f,) + p_shape + (1,)) for p_shape in self.p_shapes]
    
    print('pyramid shapes:', self.p_shapes)
    
    self.alpha = [alpha for _ in range(self.pyramid_layers)]
    self.beta = [beta for i in range(self.pyramid_layers)]
    self.gamma = [gamma for i in range(self.pyramid_layers)]
    
    self.num_iters = [num_iters for i in range(self.pyramid_layers)] #*(i + 1)
    self.opt_type = opt_type
    self.inpaint = inpaint
    
    # style transfer
    self.tf_x_ph = []
    self.tf_x_va = []

    self.tf_p_ph = []
    self.tf_a_ph = []
    
    self.tf_w_ph = []
    self.tf_c_ph = []
    
    self.L_content = []
    self.L_style = []
    self.L_temporal = []
    self.L_total = []
    
    # optimizer
    self.tf_options = []
    self.tf_optimizer = []
    self.tf_initializer = []
    
    self.vgg_data = self.load_network_data(model_path)
    
    # graph construction: for every pyramid layer we construct a separate graph
    self.tf_graphs = [tf.Graph() for _ in range(self.pyramid_layers)]
    self.tf_config = tf.ConfigProto()#allow_soft_placement=True, 
    self.tf_config.gpu_options.visible_device_list= '0'
    
    for i, g in enumerate(self.tf_graphs):
      with g.as_default():
        self.tf_init_graph(tf_shape=self.p_shapes[i], 
                           alpha=self.alpha[i], beta=self.beta[i], gamma=self.gamma[i], 
                           iters=self.num_iters[i])
          
    self.tf_sess = [tf.Session(graph=g, config=self.tf_config) for g in self.tf_graphs]

  # Basic Style Transfer Functions ============================================
  
  def prep_image(self, image):
      
    image = image - self.imgnet_mean
    image = image[...,::-1] #BGR
    image = np.reshape(image, ((1,) + image.shape))
    
    return np.float32(image)
  
  def postp_image(self, image):
    
    image = image[0,:,:,::-1] #RGB
    image = image + self.imgnet_mean
    image = np.clip(image, 0, 255).astype('uint8')
    
    return image
    
  def load_network_data(self, model_path):

    layers = []
    weights = []
    biases = []
    
    if not os.path.exists(model_path):
      raise ValueError('videostyletransfer: model path ivalid or no model in \
                       folder')
      
    data = loadmat(model_path)
    network_data = data['layers']
    
    for i in range(self.num_layers):
      name = network_data[0][i][0][0][0][0]
      if name[:4] == 'conv':
        w = network_data[0][i][0][0][2][0][0]
        b = network_data[0][i][0][0][2][0][1]
        weights.append(w.transpose((1, 0, 2, 3)))
        biases.append(b.reshape(-1))
      layers.append(name)
    
    return [layers, weights, biases] 
  
  def vgg(self, vgg_data, input_image):
    
    # input image: [batch, height, width, channels]
    # weights: [height, width, channels_in, channels_o]
    
    layers = vgg_data[0]
    weights = vgg_data[1]
    biases = vgg_data[2]
    
    idx = 0
    net = {}
    node = input_image
      
    for layer in layers:
      name = layer[:4]
      if name == 'conv':
        w = weights[idx]
        b = biases[idx]
        idx += 1
        node = tf.nn.bias_add(tf.nn.conv2d(node, tf.constant(w), 
                                           strides=(1, 1, 1, 1), 
                                           padding='SAME'), b)
      elif name == 'relu':
        node = tf.nn.relu(node)
      elif name == 'pool':
        node = tf.nn.max_pool(node, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), 
                              padding='SAME')
      net[layer] = node
  
    return net
    
  def gram_matrix(self, mat):
    
    b, h, w, c = mat.get_shape()#batch size 2 mit content und style
    F = tf.reshape(mat, (h*w, c))
    #F = F - tf.reduce_mean(F, axis=0, keepdims=True)
    G = tf.matmul(tf.transpose(F), F) / int(h*w)
    
    return G
  
  # Video Style Transfer Functions ============================================
  
  def warp(self, A, flow):
    
    assert A.shape[-3:-1] == flow.shape[-3:-1], "dimension error: input and \
                                                 flow size do not match"
    h, w = flow.shape[:2]
    c = A.shape[-1]
    
    x = (flow[...,0] + np.arange(w)).ravel()
    y = (flow[...,1] + np.arange(h)[:,np.newaxis]).ravel()
    
    warped = np.empty((h, w, c))
    
    for i in range(c):
      R = A[...,i].reshape((h, w))
      warped[...,i] = map_coordinates(R, [y, x]).reshape((h, w))
    
    return warped.reshape(A.shape)
  
  def mid_term_weights(self, w_warp, w_back):
    
    weights = np.ones(w_warp.shape[:2])
    
    norm_wb = np.linalg.norm(w_warp + w_back, axis=2)**2.0
    norm_w = np.linalg.norm(w_warp, axis=2)**2.0
    norm_b = np.linalg.norm(w_back, axis=2)**2.0
    
    disoccluded_regions = norm_wb > 0.01*(norm_w + norm_b) + 0.5
    
    norm_u = np.linalg.norm(np.gradient(w_back[...,0]), axis=0)**2.0
    norm_v = np.linalg.norm(np.gradient(w_back[...,1]), axis=0)**2.0
    
    motion_boundaries = norm_u + norm_v > 0.01*norm_b + 0.002
    
    weights[np.where(disoccluded_regions)] = 0
    weights[np.where(motion_boundaries)] = 0
    
    return weights
  
  def long_term_weights(self, frame, past_frames):
    
    mid_term = []
    warp_flows = []
    
    for i, past_frame in enumerate(past_frames):
      
      #forward_flow, backward_flow = of.optical_flow(frame, past_frame)
      forward_flow, backward_flow = of.optical_flow_pre(frame, past_frame)
      #forward_flow = self.flow_forward[i]
      #backward_flow = self.flow_backward[i]

      '''
      f1 = self.postp_image(self.content_images[frame])
      f2 = self.postp_image(self.content_images[past_frame])
      
      imgs = [f2, f1]
      
      img_height = f1.shape[0]
      img_width = f1.shape[1]
      
      img_shape = (img_height, img_width, 3)
      flo_shape = (img_height, img_width, 2)
      
      self.server.send_images(imgs)
      rev_imgs = self.server.recv_images(flo_shape, np.float32)
      
      forward_flow = rev_imgs[0]
      
      imgs = [f1, f2]
      self.server.send_images(imgs)
      rev_imgs = self.server.recv_images(flo_shape, np.float32)
      backward_flow = rev_imgs[0]
      '''
      
      warped_flow = self.warp(forward_flow, backward_flow)
      warp_flows.append(backward_flow)
      
      new_c = self.mid_term_weights(warped_flow, backward_flow)
      long_term = new_c.copy()

      for c in mid_term:
        long_term -= c
        
      long_term[long_term < 0] = 0

      mid_term.append(new_c)
      self.c_long[i] = long_term.reshape((1,) + long_term.shape + (1,))
    
    return warp_flows

  # Tensorflow Functions ======================================================

  def tf_init_graph(self, tf_shape, alpha, beta, gamma, iters):
    ph_shape = (1,) + tf_shape + (3,)
    ph_shape_c = (1,) + tf_shape + (1,)
    print(ph_shape, 'alpha', alpha, 'beta', beta, 'gamma', gamma, 'iters', iters)
    
    self.tf_x_ph.append(tf.placeholder(tf.float32, shape=ph_shape))
    self.tf_x_va.append(tf.Variable(self.tf_x_ph[-1], 
                                    trainable=True, 
                                    dtype=tf.float32))

    self.tf_p_ph.append(tf.placeholder(tf.float32, shape=ph_shape))
    self.tf_a_ph.append(tf.placeholder(tf.float32, shape=ph_shape))
    
    self.tf_w_ph.append(tf.placeholder(tf.float32, shape=ph_shape))# for _ in self.J]
    self.tf_c_ph.append(tf.placeholder(tf.float32, shape=ph_shape_c))# for _ in self.J]
    
    content_layer = ['relu4_2']
    style_layer = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    
    # content layer
    P = self.vgg(self.vgg_data, self.tf_p_ph[-1])
    P = [P[l] for l in content_layer]
    
    # style layer
    A = self.vgg(self.vgg_data, self.tf_a_ph[-1])
    A = [self.gram_matrix(A[l]) for l in style_layer]
    
    # input layer
    X = self.vgg(self.vgg_data, self.tf_x_va[-1])
    F = [X[l] for l in (content_layer)]
    G = [self.gram_matrix(X[l]) for l in (style_layer)]
    
    content_weights = [1e0]
    #style_weights = [0.2/n**2 for n in [64,128,256,512,512]]
    style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
    
    self.L_content.append(0.0)
    self.L_style.append(0.0)
    self.L_temporal.append(tf.constant(0.0))
    
    # content loss
    for l, n in enumerate(content_layer):
      self.L_content[-1] += content_weights[l]*tf.reduce_mean((F[l] - P[l])**2.0)
    
    # style loss
    for l, n in enumerate(style_layer):
      self.L_style[-1] += style_weights[l]*tf.reduce_mean((G[l] - A[l])**2.0)
      
    # temporal loss
    #for l, n in enumerate(self.J):
    #  self.L_temporal[-1] += tf.reduce_mean(tf.multiply(self.tf_cl[i], (self.x - self.tf_warp[i])**2.0))
    self.L_temporal[-1] = tf.reduce_mean(tf.multiply(self.tf_c_ph[-1], (self.tf_x_va[-1] - self.tf_w_ph[-1])**2.0))
      
    # total loss
    self.L_total.append(alpha*self.L_content[-1] + beta*self.L_style[-1] + gamma*self.L_temporal[-1])
    
    # optimizer
    self.tf_options.append({'maxiter': iters, 'disp': False})#, 'ftol': 0.0001
    
    #self.test_grad.append(tf.gradient(self.L_total[-1], self.tf_x_va[-1]))
    
    if self.opt_type is 'scipy':
      opt = tf.contrib.opt.ScipyOptimizerInterface(self.L_total[-1], 
                                                   method='L-BFGS-B', 
                                                   options=self.tf_options[-1])
    elif self.opt_type is 'tf':
      opt = tf.train.MomentumOptimizer(1e2, 0.9, use_nesterov=True
                                       ).minimize(self.L_total[-1])
    self.tf_optimizer.append(opt)
    self.tf_initializer.append(tf.global_variables_initializer())
    
  def run(self, layer):
    
    def callback(L, L_c, L_s, L_t, it=[0]):
      print('Iteration: %4d' % it[0], 
            'Total: %12g, Content: %12g, Style: %12g, Temporal: %12g' % (L, L_c, L_s, L_t))
      it[0] += 1
      
    self.tf_sess[layer].run(self.tf_initializer[layer], 
                            {self.tf_x_ph[layer]: self.input_image})
    
    #g_test = self.tf_sess[layer].run(self.grad_test[layer])
    #print(g_test.shape)
    #imageio.imwrite('grad_test.png', self.postp_image(g_test))
    
    if self.opt_type is 'scipy':
      self.tf_optimizer[layer].minimize(self.tf_sess[layer], 
                       feed_dict={self.tf_p_ph[layer]:self.content_image, 
                                  self.tf_a_ph[layer]:self.style_image,
                                  self.tf_w_ph[layer]:self.warped_image, 
                                  self.tf_c_ph[layer]:self.weight_image},
                       fetches=[self.L_total[layer], 
                                self.alpha[layer]*self.L_content[layer], 
                                self.beta[layer]*self.L_style[layer],
                                self.gamma[layer]*self.L_temporal[layer]], 
                       loss_callback=callback)
      
    elif self.opt_type is 'tf':
      for i in range(self.num_iters[layer]):
        loss, _ = self.tf_sess[layer].run([self.L_total[layer], 
                                          self.tf_optimizer[layer]], 
                                          feed_dict={self.tf_p_ph[layer]:self.content_image,
                                                     self.tf_a_ph[layer]:self.style_image})
        print(i, loss)

    return self.tf_sess[layer].run(self.tf_x_va[layer])

  # Image Pyramid Functions ===================================================
  
  def pyramid_shapes(self, base_shape):
    
    shapes = []
    for n in range(self.pyramid_layers):
      shapes.append(tuple((np.array(base_shape)/(2**n)).astype(np.int32)))
      
    return shapes

  def image_pyramid(self, tensor):
    
    pyramid = []
    for shape in self.p_shapes:
      pyramid.append(self.resize_bicubic(tensor, shape))
      
    return pyramid

  def resize_bicubic(self, tensor, size):
    
    resized = cv2.resize(tensor[0], size[::-1], interpolation=cv2.INTER_CUBIC)

    if len(resized.shape) < 3:
      resized = resized.reshape(resized.shape + (1,))
    
    return resized.reshape((1,) + resized.shape)

  def resize_same(self, c_long, c_shape):
    
    nor = []
    for c in c_long:
      nor.append(self.resize_bicubic(c, c_shape))

    return nor

  def resize_whole(self, cont, inp):
    
    for p, shape in enumerate(self.p_shapes):
      for i, ip in enumerate(inp):
        cont[p][i] = self.resize_bicubic(ip, shape)
        
    return cont
    
  def queue_flows(self, idx):
    
    flow_queue = [self.content_raw[idx + 1]]
    print('main idx: ', idx + 1)
  
    for j in self.J:
        if (idx + 1) >= j:
          print('idx: ', idx + 1 - j)
          flow_queue.append(self.content_raw[idx + 1 - j])
    
    self.server.send_images(flow_queue)
  
  def recv_flows(self, past_frames):
    
    num_flows = len(past_frames)
    flo_shape = (2*num_flows,) + self.image_shape[1:3] + (2,)
    print(flo_shape)
    rev_flows = self.server.recv_images(flo_shape, np.float32)
    
    return rev_flows[:num_flows], rev_flows[num_flows:]
    
    
  # Main ======================================================================

  def optimize_images(self, cshape=None, multipass=True, init=False):
    
    print('optimize_start')
    
    styled_images = []
    past_frames = []
    past_styled = []
    calc_times = []

    frame_indices = np.arange(len(self.content_images)) + 1
    
    for i, f_idx in enumerate(frame_indices):
      
      time_start = time.time()
      
      self.content_image = self.content_images[f_idx - 1]
      
      if i + 1 < len(frame_indices) and self.external_flow:
        self.queue_flows(i)
        
      flows = self.long_term_weights(f_idx, past_frames)

      for j, x_p in enumerate(past_styled):
        self.x_w[j] = self.warp(x_p, flows[j])
        #imageio.imsave((os.getcwd() + '/output/warp_%04d' % f_idx) + '_' + str(j) + '.png', self.postp_image(self.x_w[j]))
      
      self.p_content = self.image_pyramid(self.content_image)
      #I_rand = self.prep_image(np.random.normal(0, 128, self.p_content[-1].shape[1:]))
      self.input_image = self.p_content[-1]#I_rand#
      #if i == 0:
      #  self.input_image = self.p_content[-1]#I_rand#
      #else:
      #  self.input_image = self.x_w[0]
      
      self.w = self.resize_whole(self.w, self.x_w)
      self.c = self.resize_whole(self.c, self.c_long)
      
      for k, p_layer in enumerate(reversed(range(self.pyramid_layers))):
        
        p_shape = self.p_shapes[p_layer]
        
        wrp = self.w[p_layer]
        nor = self.c[p_layer]
        
        inv = 1.0 - np.sum(nor, axis=0) #shape?
        self.input_image = self.resize_bicubic(self.input_image, p_shape)*inv
        
        for l, w in enumerate(wrp):
          self.input_image += w*nor[l]
          
        self.content_image = self.p_content[p_layer]
        self.style_image = self.p_style[p_layer]
        self.warped_image = self.input_image.copy()
        self.weight_image = 1.0 - np.reshape(inv, (1,) + inv.shape)
        
        styled = self.run(layer=p_layer)
        self.input_image = styled.copy()
        #imageio.imsave((os.getcwd() + '/output/pyr_%04d' % f_idx) + '_' + str(k) + '.png', self.postp_image(styled))
      
      styled_images.append(styled)
      imageio.imsave((self.output_dir + '/styled_%04d.png' % f_idx), self.postp_image(styled)) #am ende erst?
      
      past_frames = []
      past_styled = []
      
      for j in self.J:
        if (i + 1) >= j and not init:
          past_frames.append(frame_indices[i + 1 - j])
          past_styled.append(styled_images[i + 1 - j])
          print('past:', frame_indices[i + 1 - j], 'styled:', i + 1 - j)
      
      if i + 1 < len(frame_indices) and self.external_flow:
        self.flow_forward, self.flow_backward = self.recv_flows(past_frames)
      
      time_end = time.time()
      print('style calculation complete: ', time_end - time_start)
      calc_times.append(time_end - time_start)
    
    print(calc_times)
    print(np.sum(np.array(calc_times)))
    return styled_images


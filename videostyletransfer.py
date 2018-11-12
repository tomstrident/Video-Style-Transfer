

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
from threading import Barrier, Event, Thread

import network

#source activate ...
#export PYTHONPATH=/home/tom/Downloads/flownet2-master/python:$PYTHONPATH
#export PYTHONPATH=/home/tom/Documents/PythonProjects/VST/flownet2-master/python:$PYTHONPATH
#echo $PYTHONPATH

class VideoStyleTransferModule:
  
  def __init__(self, 
               content, style, flow_path,
               alpha=1, beta=1, gamma=200,
               external_flow=False, show_info=False,
               num_iters=3000, pyramid_layers=4, opt_type='scipy',
               inpaint=None, eps=1e1, reduce_layers=True,
               model_path='Models/imagenet-vgg-verydeep-19.mat'):
    
    self.imgnet_mean = np.array([123.68, 116.779, 103.939])
    self.num_layers = 36
    self.flow_path = flow_path
    self.external_flow = external_flow
    self.show_info = show_info
    
    self.content_raw = []
    self.content_images = []
    self.style_image = []
    
    if self.external_flow:
      self.server = network.network()
      self.server.host_setup()
      self.server.host_connect()
    
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
    self.flow_shape = self.image_shape[1:3] + (2,)

    self.input_image = np.empty(self.image_shape)

    # temporal
    self.J = [1, 2, 4]
    self.c_mid = [np.zeros(self.weight_shape) for _ in self.J]
    self.c_long = [np.zeros(self.weight_shape) for _ in self.J]
    self.x_w = [np.zeros(self.image_shape) for _ in self.J]
    
    self.forward_flows = [np.zeros(self.flow_shape) for _ in self.J]
    self.backward_flows = [np.zeros(self.flow_shape) for _ in self.J]
    self.warped_flows = [np.zeros(self.flow_shape) for _ in self.J]

    self.past_frames = []
    self.past_styled = []
    self.flow_forward = []
    self.flow_backward = []

    self.warped_image = []
    self.weight_image = []

    # image pyramid
    self.pyramid_layers = pyramid_layers
    self.p_shapes = self.pyramid_shapes(self.image_shape[1:3])
    self.p_style = self.image_pyramid(self.style_image)
    
    n_f = len(self.J)
    self.w = [np.zeros((n_f,) + p_shape + (3,)) for p_shape in self.p_shapes]
    self.c = [np.zeros((n_f,) + p_shape + (1,)) for p_shape in self.p_shapes]
    
    print('pyramid shapes:', self.p_shapes)
    
    self.alpha = [alpha for _ in range(self.pyramid_layers)]
    self.beta = [beta for i in range(self.pyramid_layers)]
    self.gamma = [gamma for i in range(self.pyramid_layers)]
    
    self.num_iters = [num_iters for i in range(self.pyramid_layers)]
    #self.num_iters = [4, 14, 34]
    #self.num_iters = [2, 7, 17]
    self.opt_type = opt_type
    self.inpaint = inpaint
    
    # multithreading
    self.t_num = 2*len(self.J)
    self.t_stop = Event()
    self.t_bar = Barrier(self.t_num + 1)
    self.threads = []
    
    for tid in range(self.t_num):
      t_arg = (tid % 3, self.t_stop, self.t_bar)
      if tid < 3:
        self.threads.append(Thread(target=self.t_warp_flows, args=t_arg))
      else:
        self.threads.append(Thread(target=self.t_warp_styled, args=t_arg))
      self.threads[-1].start()

    # style transfer
    self.content_layer = ['relu4_2']
    
    if reduce_layers:
      self.style_layer = ['relu3_1']
    else:
      self.style_layer = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

    self.tf_x_ph = []
    self.tf_p_ph = []
    self.tf_a_ph = []
    self.tf_w_ph = []
    self.tf_c_ph = []

    self.tf_x_va = []
    self.tf_p_va = []
    self.tf_a_va = []
    self.tf_w_va = []
    self.tf_c_va = []
    
    self.L_content = []
    self.L_style = []
    self.L_temporal = []
    self.L_total = []
    
    # optimizer
    #self.eps = [1e1, 1e0, 1e-1, 1e-2]
    self.eps = [eps for _ in range(self.pyramid_layers)]
    self.tf_options = []
    self.tf_optimizer = []
    self.tf_initializer = []
    
    self.vgg_data = self.load_network_data(model_path)
    
    # stats
    self.ipo = [[] for _ in range(self.pyramid_layers)] # iterations per optimization
    self.tpo = [[] for _ in range(self.pyramid_layers)] # time per optimization
    self.ips = [[] for _ in range(self.pyramid_layers)] # iterations per second
    self.tpt = [] # total process time

    # graph construction: for every pyramid layer we construct a separate graph
    self.tf_graphs = [tf.Graph() for _ in range(self.pyramid_layers)]
    self.tf_config = tf.ConfigProto()#allow_soft_placement=True, 
    self.tf_config.gpu_options.visible_device_list= '0'
    
    for i, g in enumerate(self.tf_graphs):
      with g.as_default():
        self.tf_init_graph(tf_shape=self.p_shapes[i], 
                           alpha=self.alpha[i], beta=self.beta[i], gamma=self.gamma[i], 
                           iters=self.num_iters[i], eps=self.eps[i])
          
    self.tf_sess = [tf.Session(graph=g, config=self.tf_config) for g in self.tf_graphs]

  def __del__(self):
    print('del called')
    #self.exit_threads()

  #def __exit__(self, exc_type, exc_value, traceback):
  #  print('exit called')
  #  self.exit_threads()
    
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
    
    b, h, w, c = mat.get_shape()
    F = tf.reshape(mat, (h*w, c))
    G = tf.matmul(tf.transpose(F), F) / int(h*w)
    
    return G
  
  # Video Style Transfer Functions ============================================
  
  def warp(self, A, flow, shape):
    
    assert A.shape[-3:-1] == flow.shape[-3:-1], "dimension error: input and \
                                                 flow size do not match"
    h, w = flow.shape[:2]
    x = (flow[...,0] + np.arange(w)).astype(A.dtype)
    y = (flow[...,1] + np.arange(h)[:,np.newaxis]).astype(A.dtype)

    return cv2.remap(A, x, y, cv2.INTER_LINEAR).reshape(shape)

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
    
    return weights.reshape(self.weight_shape)

  def acquire_optical_flows(self, frame, past_frames):
    
    h, w = self.flow_shape[0:2]

    for i, past_frame in enumerate(past_frames):
      
      forward_path = self.flow_path + 'frame_%04d-%04d.flo' % (past_frame, frame)
      backward_path = self.flow_path + 'frame_%04d-%04d.flo' % (frame, past_frame)
      
      forward_flow = of.read_flow_file(forward_path)
      backward_flow = of.read_flow_file(backward_path)
      
      self.forward_flows[i] = forward_flow[:h,:w,:]
      self.backward_flows[i] = backward_flow[:h,:w,:]
      
    return self.forward_flows, self.backward_flows

  def warp_optical_flows(self, forward_flows, backward_flows):
    
    for i, _ in enumerate(self.past_styled):
      self.warped_flows[i] = self.warp(forward_flows[i], backward_flows[i])
      
    return self.warped_flows

  def warp_past_styled(self, past_styled, backward_flows):

    for j, _ in enumerate(self.past_styled):
      self.x_w[j] = self.warp(self.past_styled[j], self.backward_flows[j])
      #imageio.imsave((os.getcwd() + '/output/warp_%04d' % f_idx) + '_' + str(j) + '.png', self.postp_image(self.x_w[j]))
      
    return self.x_w
  
  def compute_c_mid(self, warped_flows, backward_flows):
  
    for i, _ in enumerate(self.past_styled):
      self.c_mid[i] = self.mid_term_weights(warped_flows[i], backward_flows[i])
      
    return self.c_mid
    
  def compute_c_long(self, c_mid):
  
    c_temp = []
  
    for i, _ in enumerate(self.past_styled):
      self.c_long[i] = self.c_mid[i].copy()
      
      for c in c_temp:
        self.c_long[i] -= c
        
      self.c_long[i] = np.maximum(self.c_long[i], 0)
      c_temp.append(self.c_mid[i])
      
    return self.c_long
  
  def compute_past_weights(self, warped_flows, backward_flows):
  
    mid_term = []
    
    for i, _ in enumerate(self.past_styled):
      new_c = self.mid_term_weights(warped_flows[i], backward_flows[i])
      long_term = new_c.copy()

      for c in mid_term:
        long_term -= c
        
      long_term = np.maximum(long_term, 0)

      mid_term.append(new_c)
      self.c_long[i] = long_term.reshape((1,) + long_term.shape + (1,))
      
    return self.c_long

  # Tensorflow Functions ======================================================

  def tf_init_graph(self, tf_shape, alpha, beta, gamma, iters, eps):

    ph_shape = (1,) + tf_shape + (3,)
    ph_shape_c = (1,) + tf_shape + (1,)

    print(ph_shape, 'alpha', alpha, 'beta', beta, 'gamma', gamma, 'iters', iters, 'eps', eps)
    
    self.tf_x_ph.append(tf.placeholder(tf.float32, shape=ph_shape))
    self.tf_p_ph.append(tf.placeholder(tf.float32, shape=ph_shape))
    self.tf_a_ph.append(tf.placeholder(tf.float32, shape=ph_shape))
    self.tf_w_ph.append(tf.placeholder(tf.float32, shape=ph_shape))
    self.tf_c_ph.append(tf.placeholder(tf.float32, shape=ph_shape_c))
    
    self.tf_x_va.append(tf.Variable(self.tf_x_ph[-1], trainable=True, dtype=tf.float32))
    self.tf_p_va.append(tf.Variable(self.tf_p_ph[-1], trainable=False, dtype=tf.float32))
    self.tf_a_va.append(tf.Variable(self.tf_a_ph[-1], trainable=False, dtype=tf.float32))
    self.tf_w_va.append(tf.Variable(self.tf_w_ph[-1], trainable=False, dtype=tf.float32))
    self.tf_c_va.append(tf.Variable(self.tf_c_ph[-1], trainable=False, dtype=tf.float32))
        
    # content layer
    P = self.vgg(self.vgg_data, self.tf_p_va[-1])
    P = [P[l] for l in self.content_layer]
    
    # style layer
    A = self.vgg(self.vgg_data, self.tf_a_va[-1])
    A = [self.gram_matrix(A[l]) for l in self.style_layer]
    
    # input layer
    X = self.vgg(self.vgg_data, self.tf_x_va[-1])
    F = [X[l] for l in (self.content_layer)]
    G = [self.gram_matrix(X[l]) for l in (self.style_layer)]
    
    content_weights = [1e0]
    style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
    
    self.L_content.append(0.0)
    self.L_style.append(0.0)
    self.L_temporal.append(tf.constant(0.0))
    
    # content loss
    for l, n in enumerate(self.content_layer):
      self.L_content[-1] += content_weights[l]*tf.reduce_mean((F[l] - P[l])**2.0)
    
    # style loss
    for l, n in enumerate(self.style_layer):
      self.L_style[-1] += style_weights[l]*tf.reduce_mean((G[l] - A[l])**2.0)
      
    # temporal loss
    #for l, n in enumerate(self.J):
    #  self.L_temporal[-1] += tf.reduce_mean(tf.multiply(self.tf_cl[i], (self.x - self.tf_warp[i])**2.0))
    self.L_temporal[-1] = tf.reduce_mean(tf.multiply(self.tf_c_va[-1], (self.tf_x_va[-1] - self.tf_w_va[-1])**2.0))
      
    # total loss
    self.L_total.append(alpha*self.L_content[-1] + beta*self.L_style[-1] + gamma*self.L_temporal[-1])
    
    # optimizer
    self.tf_options.append({'maxiter': iters, 'disp': False, 'ftol': 0.0001})#
    
    if self.opt_type is 'scipy':
      opt = tf.contrib.opt.ScipyOptimizerInterface(self.L_total[-1], 
                                                   method='L-BFGS-B', 
                                                   options=self.tf_options[-1])
    elif self.opt_type is 'tf':
      #opt = tf.train.MomentumOptimizer(1e-1, 0.9, use_nesterov=True).minimize(self.L_total[-1])
      #elif self.opt_type is 'adam':
      opt = tf.train.AdamOptimizer(1e1).minimize(self.L_total[-1])

    self.tf_optimizer.append(opt)
    self.tf_initializer.append(tf.global_variables_initializer())
    
  def run(self, layer, frame_iter):
    
    def callback(L, L_c, L_s, L_t, it=[0]):
      print('Iteration: %4d' % it[0], 
            'Total: %12g, Content: %12g, Style: %12g, Temporal: %12g' % (L, L_c, L_s, L_t))
      it[0] += 1
    
    #init
    i = 0
    self.tf_sess[layer].run(self.tf_initializer[layer], 
                            {self.tf_x_ph[layer]:self.input_image,
                             self.tf_p_ph[layer]:self.content_image, 
                             self.tf_a_ph[layer]:self.style_image,
                             self.tf_w_ph[layer]:self.warped_image, 
                             self.tf_c_ph[layer]:self.weight_image})
    o1 = time.time()
    iter_count = 0
    
    if self.opt_type is 'scipy':
      iter_count = self.num_iters[layer]
      self.tf_optimizer[layer].minimize(self.tf_sess[layer], 
                       fetches=[self.L_total[layer], 
                                self.alpha[layer]*self.L_content[layer], 
                                self.beta[layer]*self.L_style[layer],
                                self.gamma[layer]*self.L_temporal[layer]], 
                       loss_callback=callback)
      
    elif self.opt_type is 'tf':
      
      #while inter_time - start_time < 1.0:
      for i in range(self.num_iters[layer]):
        
        loss, _ = self.tf_sess[layer].run([self.L_total[layer], 
                                          self.tf_optimizer[layer]])
        print('Iteration:', i, 'Loss:', loss)
        iter_count += 1

    o2 = time.time()
    
    print('optimization time', o2 - o1, 'iterations', iter_count)
    
    self.tpo[layer].append(o2 - o1)
    self.ipo[layer].append(iter_count)
    self.ips[layer].append(iter_count/(o2 - o1))

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

  # Multithreading ============================================================

  def t_warp_flows(self, tid, ctl, bar):
    print('start flow thread', tid)
    while True:
      bar.wait()
      
      if ctl.is_set():
        break
      
      if len(self.past_styled) > tid:
        #print('go', tid)
        self.warped_flows[tid] = self.warp(self.forward_flows[tid], self.backward_flows[tid], self.flow_shape)
        self.c_mid[tid] = self.mid_term_weights(self.warped_flows[tid], self.backward_flows[tid])
        bar.wait()
    print('end flow thread', tid)
    
  def t_warp_styled(self, tid, ctl, bar):
    print('start style thread', tid)
    while True:
      bar.wait()
      
      if ctl.is_set():
        break
      
      if len(self.past_styled) > tid:
        #print('go', tid)
        self.x_w[tid] = self.warp(self.past_styled[tid][0], self.backward_flows[tid], self.image_shape)
        #imageio.imsave((os.getcwd() + '/output/warp_%04d' % 1) + '_' + str(tid) + '.png', self.postp_image(self.x_w[tid]))
        for p, shape in enumerate(self.p_shapes):
          self.w[p][tid] = self.resize_bicubic(self.x_w[tid], shape)
          
        bar.wait()
    print('end style thread', tid)
    
  def exit_threads(self):
    print('exit_threads called')
    self.t_stop.set()
    self.t_bar.wait()
    for t in self.threads:
      t.join()
    
  # Main ======================================================================

  def psnr(self, img1, img2):
    mse = np.mean((self.postp_image(img1) - self.postp_image(img2))**2)
    
    if mse == 0:
      return 100.0
      
    return 20*np.log10(255.0/np.sqrt(mse))

  def optimize_images(self, cshape=None, multipass=True, init=False):
    
    styled_images = []
    past_frames = []
    ref = []
    #psnrs = []

    frame_indices = np.arange(len(self.content_images)) + 1
    
    for i, f_idx in enumerate(frame_indices):
      
      #ref = imageio.imread(os.getcwd() + ('/ref/styled_%04d.png' % f_idx))
      #ref = self.prep_image(ref)

      time_start = time.time()
      
      self.content_image = self.content_images[f_idx - 1]
      self.p_content = self.image_pyramid(self.content_image)
      self.input_image = self.p_content[-1]
      
      if i + 1 < len(frame_indices) and self.external_flow:
        self.queue_flows(i)
      else:
        self.acquire_optical_flows(f_idx, past_frames)
      
      t1 = time.time()
      self.t_bar.wait()
      self.t_bar.wait()
      
      t2 = time.time()
      print('t21', t2 - t1)
      
      self.c_long = self.compute_c_long(self.c_mid)
      self.c = self.resize_whole(self.c, self.c_long)
      
      t3 = time.time()
      print('t32', t3 - t2)
      print('t31', t3 - t1)

      for k, p_layer in enumerate(reversed(range(self.pyramid_layers))):
        
        p_shape = self.p_shapes[p_layer]
        
        wrp = self.w[p_layer]
        nor = self.c[p_layer]

        inv = 1.0 - np.sum(nor, axis=0)
        self.input_image = self.resize_bicubic(self.input_image, p_shape)*inv

        for l, w in enumerate(wrp):
          self.input_image += w*nor[l]

        self.content_image = self.p_content[p_layer]
        self.style_image = self.p_style[p_layer]
        self.warped_image = self.input_image.copy()
        self.weight_image = 1.0 - np.reshape(inv, (1,) + inv.shape)
        
        s1 = time.time()
        styled = self.run(layer=p_layer, frame_iter=i)
        s2 = time.time()
        print('style time', s2 - s1)
        self.input_image = styled.copy()
        #imageio.imsave((os.getcwd() + '/output/pyr_%04d' % f_idx) + '_' + str(k) + '.png', self.postp_image(styled))
      
      styled_images.append(styled)

      t4 = time.time()
      print('t43', t4 - t3)

      imageio.imsave((self.output_dir + '/styled_%04d.png' % f_idx), self.postp_image(styled))
      
      past_frames = []
      self.past_styled = []
      
      for j in self.J:
        if (i + 1) >= j and not init:
          past_frames.append(frame_indices[i + 1 - j])
          self.past_styled.append(styled_images[i + 1 - j])
          print('past:', frame_indices[i + 1 - j], 'styled:', i + 1 - j)
      
      if i + 1 < len(frame_indices) and self.external_flow:
        self.flow_forward, self.flow_backward = self.recv_flows(past_frames)
      
      time_end = time.time()

      #psnrs.append(self.psnr(ref, styled))

      print('t4 end', time_end - t4)
      print('style calculation complete: ', time_end - time_start)
      self.tpt.append(time_end - time_start)
    
    if self.show_info:
      for i in range(self.pyramid_layers):
        print('layer', str(i))
        print('time per optimization', self.tpo[i], 'tpo avg', np.mean(np.array(self.tpo[i])))
        print('iters per optimization', self.ipo[i], 'ipo avg', np.mean(np.array(self.ipo[i])))
        print('iters per second', self.ips[i], 'ips avg', np.mean(np.array(self.ips[i])))

      print('total process time', self.tpt, 'tps sum', np.sum(np.array(self.tpt)))
      #print('psnrs', psnrs)

    self.exit_threads()

    return [self.postp_image(s) for s in styled_images]


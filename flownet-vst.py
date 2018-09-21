#!/usr/bin/env python2.7

from __future__ import print_function

import os
import time
import caffe
import tempfile
import numpy as np
from math import ceil

import network
from flow_utils import vis_flow
import imageio

#python run-flownet.py /home/schrotter/ICG-Projects/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5 /home/schrotter/ICG-Projects/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template frame_0001.png frame_0002.png c_test.flo

#source activate tensorflow
#export PYTHONPATH=/home/tom/Downloads/flownet2-master/python:$PYTHONPATH
#export PYTHONPATH=/home/tom/Documents/PythonProjects/VST/flownet2-master/python:$PYTHONPATH
#export PYTHONPATH=/home/schrotter/ICG-Projects/flownet2/python:$PYTHONPATH
#echo $PYTHONPATH
#python /home/tom/Documents/PythonProjects/VST/webcam.py
#/home/tom/Documents/PythonProjects/VST/video_style_transfer.py

path_flownet = os.getcwd() + '/flownet2/models/FlowNet2/'
path_caffemodel = path_flownet + 'FlowNet2_weights.caffemodel.h5'
path_deployproto = path_flownet + 'FlowNet2_deploy.prototxt.template'

#path_flownet = os.getcwd() + '/flownet2/models/FlowNet2-CS/'
#path_caffemodel = path_flownet + 'FlowNet2-CS_weights.caffemodel'
#path_deployproto = path_flownet + 'FlowNet2-CS_deploy.prototxt.template'

image_width = 512#256#160#512#128#1024
image_height = 384#192#120#384#96#436

img_shape = (image_height, image_width, 3)
flo_shape = (image_height, image_width, 2)
test_image = np.zeros(img_shape)

num_blobs = 2
gpu_id = 1 #0

if(not os.path.exists(path_caffemodel)): 
  raise BaseException('caffemodel does not exist: '+path_caffemodel)
  
if(not os.path.exists(path_deployproto)): 
  raise BaseException('deploy-proto does not exist: '+path_deployproto)

vars = {}
vars['TARGET_WIDTH'] = image_width
vars['TARGET_HEIGHT'] = image_height

divisor = 64.
vars['ADAPTED_WIDTH'] = int(ceil(image_width/divisor) * divisor)
vars['ADAPTED_HEIGHT'] = int(ceil(image_height/divisor) * divisor)

vars['SCALE_WIDTH'] = image_width / float(vars['ADAPTED_WIDTH']);
vars['SCALE_HEIGHT'] = image_height / float(vars['ADAPTED_HEIGHT']);

tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

proto = open(path_deployproto).readlines()
for line in proto:
  for key, value in vars.items():
    tag = "$%s$" % key
    line = line.replace(tag, str(value))

  tmp.write(line)

tmp.flush()

caffe.set_logging_disabled()
caffe.set_device(gpu_id)
caffe.set_mode_gpu()

net = caffe.Net(tmp.name, path_caffemodel, caffe.TEST)

print('loading complete')

# =============================================================================

def load_input(img0, img1):

  input_data = []
  
  #img0 = scipy.misc.imread('c1.png')
  #img1 = scipy.misc.imread('c2.png')
  
  if len(img0.shape) < 3: 
    input_data.append(img0[np.newaxis, np.newaxis, :, :])
  else:                   
    input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    
  if len(img1.shape) < 3: 
    input_data.append(img1[np.newaxis, np.newaxis, :, :])
  else:                   
    input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
  
  in_dict = {}
  for blob_idx in range(num_blobs):
    in_dict[net.inputs[blob_idx]] = input_data[blob_idx]
    
  return in_dict

#load_input(test_image, test_image) #pre allocation

def optical_flow_calculation(im1, im2):
  time_start = time.time()
  print(im1.shape, im2.shape)
  input_dict = load_input(im1, im2)
  print('in dict')
  #
  # There is some non-deterministic nan-bug in caffe
  # it seems to be a race-condition 
  #
  
  i = 1
  while i <= 5:
    i += 1
  
    net.forward(**input_dict)
  
    containsNaN = False
    for name in net.blobs:
      blob = net.blobs[name]
      has_nan = np.isnan(blob.data[...]).any()
  
      if has_nan:
        print('blob %s contains nan' % name)
        containsNaN = True
  
    if not containsNaN:
      print('Succeeded.')
      break
    else:
      print('**************** FOUND NANs, RETRYING ****************')
  
  flow = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0).astype(np.float32)
  time_end = time.time()
  print('flow calculation complete', flow.shape, time_end - time_start)
  
  return flow
  
# main ========================================================================

optical_flow_calculation(test_image, test_image) #pre allocation
J = [2, 3, 3, 4] #num of images to be transmitted
client = network.network()

try:
  client.connect()
  running = True
  counter = 0
  
  while running:
    in_idx = J[counter]
    if counter < 3:
      counter = counter + 1
    
    in_img_shape = (in_idx,) + img_shape
    imgs = client.recv_images(in_img_shape, np.uint8)
    
    img_queue = []
    flow_list = []
    
    #forward flow
    for img in imgs[1:]:
      img_queue.append([img, imgs[0]])
    
    #backward_flow
    for img in imgs[1:]:
      img_queue.append([imgs[0], img])
    
    for img_pair in img_queue:
      im1, im2 = img_pair
      flow = optical_flow_calculation(im1, im2)
      flow_list.append(flow)
    
    #flow_img = vis_flow(flow)
    #imageio.imwrite('flownet2test.png', flow_img)
    #imageio.imwrite('flownet2im1.png', imgs[0])
    #imageio.imwrite('flownet2im2.png', imgs[1])
    print('wait for transmit')
    client.send_images(flow_list)

except Exception as e:
  print(e)
  
finally:
  client.disconnect()


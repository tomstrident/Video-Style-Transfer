

import subprocess
import numpy as np
from matplotlib.colors import hsv_to_rgb

#flow_path = 'D:/Eigene Dateien/Dokumente/PythonProjects/VideoStyleTransfer/DeepFlow2/test_video2/'#temple_2
#flow_path = '/home/schrotter/projects/Ruder/flows/temple_2/'
#flow_path = '/home/schrotter/projects/Ruder/flows/alley_1/'
flow_path = '/home/schrotter/projects/Ruder/flows/market_6/'

tag_float = 202021.25

def read_flow_file(path):
  
  flow_data = []
    
  with open(path, 'r') as fid:
    
    tag = np.fromfile(fid, np.float32, count=1)
    w = np.fromfile(fid, np.int32, count=1)[0]
    h = np.fromfile(fid, np.int32, count=1)[0]
    
    if(tag != tag_float):
      print('error')
    
    data = np.fromfile(fid, np.float32, count=2*h*w)
    flow_data = np.resize(data, (h, w, 2))

  return flow_data

def visualize_flow(flow):
  
  h, w = flow.shape[:2]
  hsv = np.empty([h, w, 3])
  
  def cart2polar(x, y):
    rho = np.sqrt(x**2.0 + y**2.0)
    phi = np.arctan2(y, x)
    return [rho, phi]
  
  rho, phi = cart2polar(flow[...,0], flow[...,1])
  
  def normalize(mat):
    mat_min = np.min(mat)
    mat_max = np.max(mat)
    return ((mat - mat_min) / (mat_max - mat_min))
  
  hsv[...,0] = normalize(phi)
  hsv[...,1] = 1.0
  hsv[...,2] = normalize(rho)
  
  return hsv_to_rgb(hsv)

def optical_flow_pre(image_A, image_B):

  forward_path = flow_path + 'frame_%04d-%04d.flo' % (image_B, image_A)
  backward_path = flow_path + 'frame_%04d-%04d.flo' % (image_A, image_B)
  
  forward_flow = read_flow_file(forward_path)
  backward_flow = read_flow_file(backward_path)
  
  return forward_flow[:384,:512,:], backward_flow[:384,:512,:]
  #return forward_flow[:384,:512,:], backward_flow[:384,:512,:]
  #return forward_flow[:384,:512,:], backward_flow[:384,:512,:]
  #return forward_flow, backward_flow
'''
def optical_flow2(image_A, image_B):
  
  f1 = postp_image(content_images[image_A - 1])
  f2 = postp_image(content_images[image_B - 1])
  
  imageio.imwrite('/home/tom/Documents/PythonProjects/VST/f1.png', f1)
  imageio.imwrite('/home/tom/Documents/PythonProjects/VST/f2.png', f2)
  
  subprocess.call(["/home/tom/anaconda3/envs/python27/bin/python", "/home/tom/Downloads/flownet2-master/scripts/run-flownet.py", "/home/tom/Downloads/flownet2-master_cpu/models/FlowNet2/FlowNet2_weights.caffemodel.h5", "/home/tom/Downloads/flownet2-master_cpu/models/FlowNet2/FlowNet2_deploy.prototxt.template", "/home/tom/Documents/PythonProjects/VST/f1.png", "/home/tom/Documents/PythonProjects/VST/f2.png", "/home/tom/Documents/PythonProjects/VST/f.flo"], stdout=subprocess.PIPE)
  forward_flow = self.read_flow_file('/home/tom/Documents/PythonProjects/VST/f.flo')
  
  imageio.imwrite('/home/tom/Documents/PythonProjects/VST/f2.png', f1)
  imageio.imwrite('/home/tom/Documents/PythonProjects/VST/f1.png', f2)
  
  subprocess.call(["/home/tom/anaconda3/envs/python27/bin/python", "/home/tom/Downloads/flownet2-master/scripts/run-flownet.py", "/home/tom/Downloads/flownet2-master_cpu/models/FlowNet2/FlowNet2_weights.caffemodel.h5", "/home/tom/Downloads/flownet2-master_cpu/models/FlowNet2/FlowNet2_deploy.prototxt.template", "/home/tom/Documents/PythonProjects/VST/f2.png", "/home/tom/Documents/PythonProjects/VST/f1.png", "/home/tom/Documents/PythonProjects/VST/f.flo"], stdout=subprocess.PIPE)
  backward_flow = self.read_flow_file('/home/tom/Documents/PythonProjects/VST/f.flo')
  
  return forward_flow, backward_flow
'''
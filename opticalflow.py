
import numpy as np
from matplotlib.colors import hsv_to_rgb

def read_flow_file(path):
  
  flow_data = []
    
  with open(path, 'r') as fid:
    
    tag = np.fromfile(fid, np.float32, count=1)
    w = np.fromfile(fid, np.int32, count=1)[0]
    h = np.fromfile(fid, np.int32, count=1)[0]
    
    if(tag != 202021.25):
      print('error: no flow file')
    
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

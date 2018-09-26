

from threading import Thread
import socket
import struct
import time

import numpy as np

class network(Thread):
  
  def __init__(self):
    Thread.__init__(self)
    self.ADDRESS = ("localhost", 12801)
    self.s = socket.socket()
    
    self.sc = None
    self.net = self.s
  
  def host_setup(self):
    self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.s.bind(self.ADDRESS)
    self.s.listen(1)
  
  def host_connect(self):
    print("Wait for connection")
    self.sc, self.info = self.s.accept()
    print("VST client connected:", self.info)
    
    self.net = self.sc
  
  def connect(self, address=("localhost", 12801)):
    self.s.connect(address)
    print('connected')
  
  def disconnect(self):
    self.s.close()
    
    if self.sc is not None:
      self.sc.close()
    print('disconnected')
  
  def recv_images(self, img_shape, data_type):
    data = self.recv_data()
    images = np.frombuffer(data, dtype=data_type)
    
    return images.reshape(img_shape)
  
  def send_images(self, images):
    images = np.array(images)
    data = images.ravel().tobytes()
    self.send_data(data)
  
  def recv_data(self):
    
    len_str = self.net.recv(4)
    
    while len(len_str) == 0:
      time.sleep(0) #yield
      
    print('unpack len', len(len_str))
    size = struct.unpack('!i', len_str)[0]
    print('size:', size)
    img_str = b''

    while size > 0:
      if size >= 4096:
        data = self.net.recv(4096)
      else:
        data = self.net.recv(size)

      if not data:
        break
        
      size -= len(data)
      img_str += data

    print('len:', len(img_str))
    #image = img_str.decode('utf-8')
      
    return img_str
  
  def send_data(self, data):
    #img_str = img_str.encode('utf-8')
    len_str = struct.pack('!i', len(data))
  
    self.net.send(len_str)
    print('pack len:', len(data))
    self.net.sendall(data)
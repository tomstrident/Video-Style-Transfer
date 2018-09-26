#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


num_frames = 50
filename = 'temple_2'
file_dir = os.getcwd() + '/MPI-Sintel-complete/training/final/'
out_dir = os.getcwd() + '/output-flow-files/'

if not os.path.exists(out_dir + filename):
  os.mkdir(out_dir + filename)

with open(filename + '.txt', 'w') as text_file:
  
  for i in range(1, num_frames + 1):

    first_frame = file_dir + filename + ('/frame_%04d.png' % i) + ' '
    second_frame = file_dir + filename + ('/frame_%04d.png' % (i + 1)) + ' '
    flow_file = out_dir + filename + ('/frame_%04d-%04d.flo' % (i, i)) + '\n'
  
    text_file.write(first_frame + second_frame + flow_file)
    
    if i > 1:
      first_frame = file_dir + filename + ('/frame_%04d.png' % (i - 1)) + ' '
      second_frame = file_dir + filename + ('/frame_%04d.png' % (i + 1)) + ' '
      flow_file = out_dir + filename + ('/frame_%04d-%04d.flo' % i) + '\n'
  
      text_file.write(first_frame + second_frame + flow_file)
      
    if i > 3:
      first_frame = file_dir + filename + ('/frame_%04d.png' % (i - 3)) + ' '
      second_frame = file_dir + filename + ('/frame_%04d.png' % (i + 1)) + ' '
      flow_file = out_dir + filename + ('/frame_%04d-%04d.flo' % i) + '\n'
  
      text_file.write(first_frame + second_frame + flow_file)
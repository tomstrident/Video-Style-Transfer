#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

num_frames = 5
frame_range = range(1, num_frames + 1)

filename = 'temple_2'
in_dir = os.getcwd() + '/input/'
out_dir = os.getcwd() + '/flow/'

deepflow_path = '/home/tom/Downloads/DeepFlow_release2.0/deepflow2 '
deepmatching_path = '/home/tom/Downloads/deepmatching_1.2.1_c++/deepmatching '

if not os.path.exists(out_dir):
  os.mkdir(out_dir)

if not os.path.exists(out_dir + filename):
  os.mkdir(out_dir + filename)

with open(filename + '.sh', 'w') as sh_script:
  
  sh_script.write('#!/bin/bash\n')
  sh_script.write('echo "started generating flows ..."\n')
  
  for i in frame_range:
    
    sh_script.write('echo "forward frame ' + str(i) + '/' + str(num_frames) + '"\n')
    first_frame = in_dir + filename + ('/frame_%04d.png ' % i)
    
    if i - 1 > 0:
      second_frame = in_dir + filename + ('/frame_%04d.png ' % (i - 1))
      forward_flow = out_dir + filename + ('/frame_%04d-%04d.flo' % (i - 1, i))
      cmd_forward = deepmatching_path + second_frame + first_frame + '| ' + deepflow_path + second_frame + first_frame + forward_flow + ' -match -sintel\n'
      sh_script.write(cmd_forward)
    if i - 2 > 0:
      second_frame = in_dir + filename + ('/frame_%04d.png ' % (i - 2))
      forward_flow = out_dir + filename + ('/frame_%04d-%04d.flo' % (i - 2, i))
      cmd_forward = deepmatching_path + second_frame + first_frame + '| ' + deepflow_path + second_frame + first_frame + forward_flow + ' -match -sintel\n'
      sh_script.write(cmd_forward)
    if i - 4 > 0:
      second_frame = in_dir + filename + ('/frame_%04d.png ' % (i - 4))
      forward_flow = out_dir + filename + ('/frame_%04d-%04d.flo' % (i - 4, i))
      cmd_forward = deepmatching_path + second_frame + first_frame + '| ' + deepflow_path + second_frame + first_frame + forward_flow + ' -match -sintel\n'
      sh_script.write(cmd_forward)
    
  for i in reversed(frame_range):
    
    sh_script.write('echo "backward frame ' + str(i) + '/' + str(num_frames) + '"\n')
    first_frame = in_dir + filename + ('/frame_%04d.png ' % i)
    
    if i + 1 < num_frames + 1:
      second_frame = in_dir + filename + ('/frame_%04d.png ' % (i + 1))
      backward_flow = out_dir + filename + ('/frame_%04d-%04d.flo' % (i + 1, i))
      cmd_backward = deepmatching_path + second_frame + first_frame + '| ' + deepflow_path + second_frame + first_frame + backward_flow + ' -match -sintel\n'
      sh_script.write(cmd_backward)
    if i + 2 < num_frames + 1:
      second_frame = in_dir + filename + ('/frame_%04d.png ' % (i + 2))
      backward_flow = out_dir + filename + ('/frame_%04d-%04d.flo' % (i + 2, i))
      cmd_backward = deepmatching_path + second_frame + first_frame + '| ' + deepflow_path + second_frame + first_frame + backward_flow + ' -match -sintel\n'
      sh_script.write(cmd_backward)
    if i + 4 < num_frames + 1:
      second_frame = in_dir + filename + ('/frame_%04d.png ' % (i + 4))
      backward_flow = out_dir + filename + ('/frame_%04d-%04d.flo' % (i + 4, i))
      cmd_backward = deepmatching_path + second_frame + first_frame + '| ' + deepflow_path + second_frame + first_frame + backward_flow + ' -match -sintel\n'
      sh_script.write(cmd_backward)
  


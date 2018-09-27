#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import imageio

video_id = 'test_video2'

output_dir = os.getcwd() + '/input/' + video_id + '/'

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

reader = imageio.get_reader(os.getcwd() + '/input/' + video_id + '.mp4')

for i, im in enumerate(reader):
  imageio.imsave(output_dir + '/frame_%04d.png' % (i + 1), im)
  
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import imageio
import scipy.misc

import numpy as np
import videostyletransfer as vst

content_path = os.getcwd() + '/input/temple_2/'
style_path = os.getcwd() + '/style-images/starry_night.jpg'
flow_path = os.getcwd() + '/flow/temple_2/'

height = 384#192#96#384#436
width = 512#256#128#512#1024

num_frames = 5

content = []
for i in range(1, num_frames + 1):
  content_image = imageio.imread(content_path + ('frame_%04d.png' % i))
  content.append(content_image[:height,:width,:])

style = imageio.imread(style_path)
style = scipy.misc.imresize(style, [height, width])
style = np.array(style)

vst_module = vst.VideoStyleTransferModule(content, style, flow_path)
styled_frames = vst_module.optimize_images()

writer = imageio.get_writer(os.getcwd() + '/output/temple_2.mp4', fps=30)

for f in styled_frames:
    writer.append_data(f)
    
writer.close()

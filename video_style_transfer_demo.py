#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import imageio
import scipy.misc

import numpy as np
import videostyletransfer as vst

video_id = 'temple_2'

content_path = os.getcwd() + '/input/' + video_id + '/'
style_path = os.getcwd() + '/style-images/starry_night.jpg'
flow_path = os.getcwd() + '/flow/' + video_id + '/'

height = 384#192#96#384#436
width = 512#256#128#512#1024

num_frames = 5
fps = 30

content = []
for i in range(1, num_frames + 1):
  content_image = imageio.imread(content_path + ('frame_%04d.png' % i))
  content.append(content_image[:height,:width,:])

style = imageio.imread(style_path)
style = scipy.misc.imresize(style, [height, width])
style = np.array(style)

vst_module = vst.VideoStyleTransferModule(content, style, flow_path)
styled_frames = vst_module.optimize_images()

vid_id = os.getcwd() + '/output/' + video_id + '.mp4'
writer = imageio.get_writer(vid_id, fps=fps)

for f in styled_frames:
  writer.append_data(f)
    
writer.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import imageio

video_id = 'temple_2'
frame_folder = os.getcwd() + '/output/' + video_id + '/'

num_frames = 30
frames = []

for i in range(num_frames):
  frames.append(imageio.imread((frame_folder + 'styled_%04d.png') % (i + 1)))

writer = imageio.get_writer(os.getcwd() + '/output/' + video_id + '.mp4', fps=30)

for f in frames:
    writer.append_data(f)

writer.close()
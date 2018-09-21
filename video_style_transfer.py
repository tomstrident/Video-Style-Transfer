

import os
import imageio
import scipy.misc

import numpy as np
import videostyletransfer as vst

#style_path = os.getcwd() + '/style-images/starry_night.jpg'
#style_path = os.getcwd() + '/style-images/the_scream.jpg'
style_path = os.getcwd() + '/style-images/the_shipwreck_of_the_minotaur.jpg'

#content_path = 'D:/Eigene Dateien/Dokumente/PythonProjects/VideoStyleTransfer/final/temple_2' #os.getcwd() + '/final/temple_2'
#content_path = 'D:/Eigene Dateien/Dokumente/PythonProjects/VST/test_videos/test_video2'
#content_path = '/home/tom/Documents/PythonProjects/VST/test_videos/test_video2'
#content_path = '/home/schrotter/ICG-Projects/test_videos/test_video2'
#content_path = '/home/schrotter/ICG-Projects/test_videos/temple_2'
content_path = '/home/schrotter/projects/Ruder/videos/market_6'
#content_path = '/home/schrotter/projects/Ruder/videos/alley_1'

height = 384#192#96#384#436
width = 512#256#128#512#1024

content = []
for i in range(1, 31):
  content_image = imageio.imread(content_path + ('/frame_%04d.png' % i))
  #content.append(content_image[:height,:width,:])
  content.append(scipy.misc.imresize(content_image, [height, width]))
  #content.append(content_image)

style = imageio.imread(style_path)
style = scipy.misc.imresize(style, [height, width])
style = np.array(style)

vst_module = vst.VideoStyleTransferModule(content, style)
vst_module.optimize_images()



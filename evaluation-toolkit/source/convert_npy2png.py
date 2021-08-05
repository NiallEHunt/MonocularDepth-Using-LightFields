# This file was modified as part of this research project.

import os
import glob
from toolkit.utils import file_io

image_path = 'C:/Users/Niall/College/Year_5/Dissertation/Results/depth_maps/AiF/DenseDepth'
paths = glob.glob(os.path.join(image_path, '*.npy'))

for path in paths:
    depth_map = file_io.read_file(path)
    depth_map = depth_map[:, :, 0]
    scene_name = path.split('\\')[-1].split('.')[0]
    file_io.write_file(depth_map, "{}/{}.png".format(image_path, scene_name))

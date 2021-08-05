# This file was modified as part of this research project.

# -*- coding: utf-8 -*-

############################################################################
#  This file is part of the 4D Light Field Benchmark.                      #
#                                                                          #
#  This work is licensed under the Creative Commons                        #
#  Attribution-NonCommercial-ShareAlike 4.0 International License.         #
#  To view a copy of this license,                                         #
#  visit http://creativecommons.org/licenses/by-nc-sa/4.0/.                #
#                                                                          #
#  Authors: Katrin Honauer & Ole Johannsen                                 #
#  Contact: contact@lightfield-analysis.net                                #
#  Website: www.lightfield-analysis.net                                    #
#                                                                          #
#  The 4D Light Field Benchmark was jointly created by the University of   #
#  Konstanz and the HCI at Heidelberg University. If you use any part of   #
#  the benchmark, please cite our paper "A dataset and evaluation          #
#  methodology for depth estimation on 4D light fields". Thanks!           #
#                                                                          #
#  @inproceedings{honauer2016benchmark,                                    #
#    title={A dataset and evaluation methodology for depth estimation on   #
#           4D light fields},                                              #
#    author={Honauer, Katrin and Johannsen, Ole and Kondermann, Daniel     #
#            and Goldluecke, Bastian},                                     #
#    booktitle={Asian Conference on Computer Vision},                      #
#    year={2016},                                                          #
#    organization={Springer}                                               #
#    }                                                                     #
#                                                                          #
############################################################################

import os
import glob
import argparse
from toolkit.scenes import PhotorealisticScene
from toolkit.utils import file_io


MIN = 0.
MAX = 255.


def main(args):
    image_path = args.image_path
    config_path = args.config_path
    pfm_path = args.pfm_path

    if not os.path.exists(pfm_path):
        os.makedirs(pfm_path)

    if os.path.isfile(image_path):
        # Only testing on a single image
        paths = [image_path]
        configs = [config_path]
    elif os.path.isdir(image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(image_path, '*.npy'))
        configs = glob.glob(os.path.join(config_path, '*.cfg'))
    else:
        raise Exception("Can not find args.image_path: {}".format(image_path))

    paths = sorted(paths)
    configs = sorted(configs)

    for idx, image_path in enumerate(paths):
        print('----------------------------------------------------------------')
        print('PNG: {} PFM: {}'.format(image_path.split('\\')[-1], configs[idx].split('\\')[-1]))
        scene = PhotorealisticScene("demo", path_to_config=configs[idx])

        disp_map = file_io.read_file(image_path)
        print(disp_map.shape)
        # log.info("Input range: [%0.1f, %0.1f]" % (np.min(disp_map), np.max(disp_map)))

        # scale from [MIN, MAX] to [disp_min, disp_max]
        disp_map = (scene.disp_max - scene.disp_min) * (disp_map - MIN) / (MAX - MIN) + scene.disp_min
        # log.info("Output range: [%0.1f, %0.1f]" % (np.min(disp_map), np.max(disp_map)))

        scene_name = image_path.split('\\')[-1].split('.')[0]
        file_io.write_file(disp_map, "{}/{}.pfm".format(pfm_path, scene_name))
        print('----------------------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PNGs to PFMs')
    parser.add_argument('--image_path', type=str, help='path to input image or folder of input images', required=True)
    parser.add_argument('--config_path', type=str, help='path to the LF configs', required=True)
    parser.add_argument('--pfm_path', type=str, help='The path to save PFMs in', required=True)
    args = parser.parse_args()
    main(args)

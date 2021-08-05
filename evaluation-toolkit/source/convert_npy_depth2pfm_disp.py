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


def main(npy_path, config_path, output_path):
    npy_path = npy_path
    config_path = config_path
    disp_map_path = output_path

    if not os.path.exists(disp_map_path):
        os.makedirs(disp_map_path)

    if os.path.isfile(npy_path):
        # Only testing on a single image
        paths = [npy_path]
        configs = [config_path]
    elif os.path.isdir(npy_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(npy_path, '*.npy'))
        configs = glob.glob(os.path.join(config_path, '*.cfg'))
    else:
        raise Exception("Can not find args.npy_path: {}".format(npy_path))

    paths = sorted(paths)
    configs = sorted(configs)

    for idx, image_path in enumerate(paths):
        scene_name = image_path.split('\\')[-1].split('.')[0]

        print('----------------------------------------------------------------')
        print('{} depth -> disp'.format(scene_name))

        scene = PhotorealisticScene("demo", path_to_config=configs[idx])
        depth_map = file_io.read_file(image_path)
        disp_map = scene.depth2disp(depth_map)
        min_ = disp_map.min()
        max_ = disp_map.max()
        disp_map = (scene.disp_max - scene.disp_min) * (disp_map - min_) / (max_ - min_) + scene.disp_min
        file_io.write_file(disp_map, "{}/{}.pfm".format(disp_map_path, scene_name))

        print('----------------------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert npy depth maps to PFM disp maps')
    parser.add_argument('--npy_path', type=str, help='path to input npy depth maps (predictions of the models)',
                        required=True)
    parser.add_argument('--config_path', type=str, help='path to the LF configs', required=True)
    parser.add_argument('--output_path', type=str, help='The path to save PFM disp maps in', required=True)
    args = parser.parse_args()
    main(args.npy_path, args.config_path, args.output_path)

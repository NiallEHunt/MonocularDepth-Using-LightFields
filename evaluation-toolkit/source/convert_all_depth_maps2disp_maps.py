# This file was modified as part of this research project.

import os
from glob import glob
from shutil import copyfile
import convert_npy_depth2pfm_disp as cnv

depth_map_path = 'C:/Users/Niall/College/Year_5/Dissertation/Results/depth_maps/'
output_path = 'C:/Users/Niall/College/Year_5/Dissertation/Results/disp_maps/'
config_path = 'C:/Users/Niall/College/Year_5/Dissertation/Data/HCI_benchmark/configs_initial_tests'

tests = ['AiF/', 'refocused_0.3/', 'refocused_0.7/']

models = [
    'defocus-net/epoch_final/focus_dist_0.1/',
    'defocus-net/epoch_final/focus_dist_0.3/',
    'defocus-net/epoch_final/focus_dist_0.7/',
    'defocus-net/epoch_final/focus_dist_0.15/',
    'defocus-net/epoch_final/focus_dist_1.5/',
    'DenseDepth/',
    'GDN-Pytorch/',
    'monodepth2/'
]

# Move runtimes to evaluation folder
# for test in tests:
#     for model in models:
#         in_path = depth_map_path + test + model
#         if model.startswith('defocus-net'):
#             out_path = output_path + test + 'algo_results/defocus-net_' + model.split('_')[-1] + 'runtimes/'
#         else:
#             out_path = output_path + test + 'algo_results/' + model + 'runtimes/'
#
#         txt_files = glob(in_path + '*.txt')
#         for file in txt_files:
#             # print(out_path + file.split('\\')[-1])
#             copyfile(file, out_path + file.split('\\')[-1])

# Convert depth maps from npy to disp maps
for test in tests:
    for model in models:
        in_path = depth_map_path + test + model
        if model.startswith('defocus-net'):
            out_path = output_path + test + 'algo_results/defocus-net_' + model.split('_')[-1] + 'disp_maps'
        else:
            out_path = output_path + test + 'algo_results/' + model + 'disp_maps'

        if not os.path.exists(out_path):
            os.makedirs(out_path)
            os.makedirs('/'.join(out_path.split('/')[:-1]) + '/runtimes')

        cnv.main(in_path, config_path, out_path)

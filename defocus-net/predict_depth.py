# This file was modified as part of this research project.

import os
import glob
import argparse
import time
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image

from src.arch.dofNet_arch1 import AENet

import torch


def predict_depth(args):
    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    device = torch.device('cpu')
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Model name and path
    model_path = 'outputs/models/a01_d06_t01/'
    model_prefix = 'a01_d06_t01_'
    model_name = model_path + model_prefix
    model_name += 'final.pth' if args.final_model else f'ep{args.epoch}.pth'

    # Load model
    model = AENet(in_dim=3, out_dim=1, num_filter=16, flag_step2=True)

    state_dict = torch.load(model_name, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for _, image_path in enumerate(paths):
            # Load image and preprocess
            input_image = Image.open(image_path)
            original_width, original_height = input_image.size
            input_image = input_image.resize((256, 256), Image.LANCZOS)
            mats_input = np.zeros((256, 256, 0))

            img_all = np.array(input_image)
            mat_all = img_all.copy() / 255
            mats_input = np.concatenate((mats_input, mat_all), axis=2)
            mats_input = mats_input.transpose((2, 0, 1))
            mats_input = mats_input[np.newaxis, :, :, :]
            x = torch.from_numpy(mats_input)
            x = x.float().to(device)

            if args.multiple_focus:
                focus_dists = [0.1, .15, .3, 0.7, 1.5]
            else:
                focus_dists = [1]
            for t in focus_dists:
                focus_distance = t / 1.5
                x2_fcs = torch.ones([x.shape[0], 1, x.shape[2], x.shape[3]])
                x2_fcs[:, 0:1, :, :] = x2_fcs[:, 0:1, :, :] * focus_distance

                # Prediction
                t0 = time.time()
                outputs = model(x, inp=3, k=1, flag_step2=True, x2=x2_fcs)
                t1 = time.time()
                total_t = t1-t0
                print(f'Time taken for {os.path.splitext(os.path.basename(image_path))[0]} = {total_t}')

                out_depth = outputs[0]
                depth_map = torch.nn.functional.interpolate(out_depth, (original_height, original_width), mode='bilinear', align_corners=False)
                depth_map_np = depth_map.squeeze().cpu().numpy()

                vmax = np.percentile(depth_map_np, 95)
                normalizer = mpl.colors.Normalize(vmin=depth_map_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='gray')
                colormapped_im = (mapper.to_rgba(depth_map_np)[:, :, :3] * 255).astype(np.uint8)
                im = Image.fromarray(colormapped_im)

                output_name = os.path.splitext(os.path.basename(image_path))[0]
                if args.final_model:
                    output_path = os.path.join(args.output_path, 'epoch_final')
                else:
                    output_path = os.path.join(args.output_path, f'epoch_{args.epoch}')
                if args.multiple_focus:
                    output_path = os.path.join(output_path, f'focus_dist_{t}')

                os.makedirs(output_path, exist_ok=True)
                output_path_name = os.path.join(output_path, f'{output_name}.png')
                im.save(output_path_name)
                output_path_name = os.path.join(output_path, f'{output_name}.npy')
                np.save(output_path_name, depth_map_np)
                with open(f'{output_path}/{output_name}.txt', 'w') as f:
                    f.write(str(total_t))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict depth with defocus-net')
    parser.add_argument('--image_path', type=str, help='path to input image or folder of input images', required=True)
    parser.add_argument('--output_path', type=str, help='path to save output', required=True)
    parser.add_argument('--epoch', type=int, help='The epoch checkpoint model to run')
    parser.add_argument('--final_model', type=bool, help='If the final model should be used')
    parser.add_argument('--multiple_focus', type=bool, help='If all focus distances should be used')
    parser.add_argument('--ext', type=str, help='the extension of images to search for', default='png')
    args = parser.parse_args()

    if not args.final_model and not args.epoch:
        parser.error('Either --final_model or --epoch needs to be set')

    predict_depth(args)

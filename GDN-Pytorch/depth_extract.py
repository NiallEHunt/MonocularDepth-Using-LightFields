# This file was modified as part of this research project. It was copied and modified from the original
# `depth_extract.py` provided by the original authors of GDN-Pytorch

from AE_model_unet import *
import os
import torch.backends.cudnn as cudnn
import time
from path import Path
from imageio import imread
import scipy.misc
from torch.autograd import Variable
import collections
import argparse


def load_as_float(path):
    return imread(path).astype(np.float32)


class Resize(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    'nearest' or 'bilinear'
    """

    def __init__(self, interpolation='bilinear'):
        self.interpolation = interpolation

    def __call__(self, img, size, img_type='rgb'):
        assert isinstance(size, int) or isinstance(size, float) or \
               (isinstance(size, collections.Iterable) and len(size) == 2)
        if img_type == 'rgb':
            if img.ndim == 3:
                return scipy.misc.imresize(img, size, self.interpolation)
            elif img.ndim == 2:
                img = scipy.misc.imresize(img, size, self.interpolation)
                img_tmp = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
                img_tmp[:, :, 0] = img[:, :]
                img = img_tmp
                return img
        elif img_type == 'depth':
            if img.ndim == 2:
                img = scipy.misc.imresize(img, size, self.interpolation, 'F')
            elif img.ndim == 3:
                img = scipy.misc.imresize(img[:, :, 0], size, self.interpolation, 'F')
            img_tmp = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
            img_tmp[:, :, 0] = img[:, :]
            img = img_tmp
            return img
        else:
            RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))


def predict_depth(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    upsampling = nn.functional.interpolate
    resize = Resize()

    ae = AutoEncoder()
    ae = ae.cuda()
    ae = nn.DataParallel(ae)
    ae.load_state_dict(torch.load(args.model_dir))
    ae = ae.eval()

    cudnn.benchmark = True
    torch.cuda.synchronize()
    sum = 0.0

    scene = Path(args.input_path)
    png_list = (scene.files("*.png"))
    jpg_list = (scene.files("*.jpg"))
    img_list = sorted(png_list + jpg_list)
    sample = []
    filenames = []
    org_H_list = []
    org_W_list = []
    for filename in img_list:
        img = load_as_float(filename)
        org_H_list.append(img.shape[0])
        org_W_list.append(img.shape[1])
        img = resize(img, (128, 416), 'rgb')
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)
        img = img / 255
        img = (img - 0.5) / 0.5
        # img = upsampling(img, (128,416),mode='bilinear', align_corners = False)
        img = Variable(img)
        sample.append(img)
        filenames.append(filename.split('\\')[-1])

    print("sample len: ", len(sample))
    '''
    for i in range(100):
         torch.cuda.synchronize()
         start = time.time()
         result = ae(img,istrain=False)
         tmp = time.time() - start
         print(tmp)
         sum += tmp
         print("----",i," th iter")
    '''
    i = 0
    os.makedirs(args.output_path, exist_ok=True)
    result_dir = args.output_path + '/'
    k = 0
    t = 0
    img_ = None
    for tens in sample:
        filename = filenames[i]
        org_H = org_H_list[i]
        org_W = org_W_list[i]
        torch.cuda.synchronize()

        start = time.time()
        img = ae(tens, istrain=False)
        tmp = time.time() - start

        if i > 0:
            sum += tmp

        img = upsampling(img, (128, 416), mode='bilinear', align_corners=False)
        npy_array_to_save = upsampling(img, (org_H, org_W), mode='bilinear', align_corners=False)
        npy_array_to_save = npy_array_to_save.cpu().detach().numpy()
        img = img[0].cpu().detach().numpy()

        if img.shape[0] == 3:
            img_ = np.empty([128, 416, 3])
            img_[:, :, 0] = img[0, :, :]
            img_[:, :, 1] = img[1, :, :]
            img_[:, :, 2] = img[2, :, :]
        elif img.shape[0] == 1:
            numpy_arr_ = np.empty([org_H, org_W])
            numpy_arr_[:, :] = npy_array_to_save[0, :, :]
            img_ = np.empty([128, 416])
            img_[:, :] = img[0, :, :]

        img_ = resize(img_, (org_H, org_W), 'rgb')

        if img_.shape[2] == 1:
            img_ = img_[:, :, 0]

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        print(result_dir + filename)
        print(f"final shape {img_.shape}")
        np.save(result_dir + filename.split('.')[0] + '.npy', numpy_arr_)
        scipy.misc.imsave(result_dir + filename, img_)

        with open(f'{result_dir}/{filename.split(".")[0]}.txt', 'w') as f:
            f.write(str(tmp))

        i = i + 1
        print("----", i, " th iter")

    print("Depth estimation demo is finished")
    print("Avg time: ", sum / len(sample))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrained Depth AutoEncoder',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str, default="..\\models\\GDN_RtoD_pretrained.pkl")
    parser.add_argument('--gpu_num', type=str, default="0")
    parser.add_argument('--output_path', type=str, help='Output folder for depth images. Default: ./output',
                        default='./output')
    parser.add_argument('--input_path', type=str, help='Path to input folder of jpg or png images', required=True)
    args = parser.parse_args()

    predict_depth(args)

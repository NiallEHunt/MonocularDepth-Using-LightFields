import random
import importlib
from os import listdir, makedirs
from os.path import isfile, join, isdir

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import numpy as np
import OpenEXR
from PIL import Image

from azureml.core import Run

run = Run.get_context()

TRAIN_PARAMS = {
    'ARCH_NUM': 1,
    'EPOCHS_NUM': 101,
    'EPOCH_START': 0,
    'FILTER_NUM': 16,
    'RANDOM_LEN_INPUT': True,

    'TRAINING_MODE': 2,

    'LEARNING_RATE': 0.0001,
    'FLAG_GPU': True,

    'MODEL1_LOAD': False,
    'MODEL1_ARCH_NUM': 1,
    'MODEL1_NAME': 'd02_t01',
    'MODEL1_EPOCH': 1000,
    'MODEL1_LOSS_WEIGHT': 1.,

    'MODEL2_TRAIN_STEP': True,
}

DATA_PARAMS = {
    'DATA_PATH': 'data/',
    'DATA_SET': 'fs_',
    'DATA_NUM': 6,

    'FLAG_SHUFFLE': True,

    'INP_IMG_NUM': 5,
    'FLAG_IO_DATA': {
        'INP_RGB': True,
        'INP_COC': False,
        'INP_DIST': True,

        'OUT_COC': True,
        'OUT_DEPTH': True,
    },
    'TRAIN_SPLIT': 0.8,
    'DATASET_SHUFFLE': True,
    'WORKERS_NUM': 4,
    'BATCH_SIZE': 16,
    'DATA_RATIO_STRATEGY': 0,
    'FOCUS_DIST': [0.1, .15, .3, 0.7, 1.5],
    'F_NUMBER': 1.,
    'MAX_DPT': 3.,
}

OUTPUT_PARAMS = {
    'EXP_NUM': 1,
    'MODEL_PATH': 'outputs/models/',
}


def set_output_folders():
    model_name = 'a' + str(TRAIN_PARAMS['ARCH_NUM']).zfill(2) + '_d' + str(DATA_PARAMS['DATA_NUM']).zfill(
        2) + '_t' + str(
        OUTPUT_PARAMS['EXP_NUM']).zfill(2)
    models_dir = OUTPUT_PARAMS['MODEL_PATH'] + model_name + '/'

    makedirs(models_dir, exist_ok=True)

    return models_dir, model_name


def set_comp_device():
    device_comp = torch.device("cpu")
    if TRAIN_PARAMS['FLAG_GPU']:
        device_comp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device_comp


def _abs_val(x):
    if isinstance(x, np.ndarray) or isinstance(x, float) or isinstance(x, int):
        return np.abs(x)
    else:
        return x.abs()


# reading depth files
def read_dpt(img_dpt_path):
    # pt = Imath.PixelType(Imath.PixelType.HALF)  # FLOAT HALF
    dpt_img = OpenEXR.InputFile(img_dpt_path)
    dw = dpt_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (r, g, b) = dpt_img.channels("RGB")
    dpt = np.fromstring(r, dtype=np.float16)
    dpt.shape = (size[1], size[0])
    return dpt


# to calculate circle of confusion
class CameraLens:
    def __init__(self, focal_length, sensor_size_full=(0, 0), resolution=(1, 1), aperture_diameter=None, f_number=None,
                 depth_scale=1):
        self.focal_length = focal_length
        self.depth_scale = depth_scale
        self.sensor_size_full = sensor_size_full

        if aperture_diameter is not None:
            self.aperture_diameter = aperture_diameter
            self.f_number = (focal_length / aperture_diameter) if aperture_diameter != 0 else 0
        else:
            self.f_number = f_number
            self.aperture_diameter = focal_length / f_number

        if self.sensor_size_full is not None:
            self.resolution = resolution
            self.aspect_ratio = resolution[0] / resolution[1]
            self.sensor_size = [self.sensor_size_full[0], self.sensor_size_full[0] / self.aspect_ratio]
        else:
            self.resolution = None
            self.aspect_ratio = None
            self.sensor_size = None
            self.fov = None
            self.focal_length_pixel = None

    def _get_indep_fac(self, focus_distance):
        return (self.aperture_diameter * self.focal_length) / (focus_distance - self.focal_length)

    def get_coc(self, focus_distance, depth):
        if isinstance(focus_distance, torch.Tensor):
            for _ in range(len(depth.shape) - len(focus_distance.shape)):
                focus_distance = focus_distance.unsqueeze(-1)

        return (_abs_val(depth - focus_distance) / depth) * self._get_indep_fac(focus_distance)


class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""

    def __init__(self, root_dir, transform_fnc=None, flag_shuffle=False, img_num=1, data_ratio=0,
                 flag_inputs=None, flag_outputs=None, focus_dist=None, f_number=0.1, max_dpt=3.):
        flag_inputs = [False, False] if flag_inputs is None else flag_inputs
        flag_outputs = [False, False] if flag_outputs is None else flag_outputs
        focus_dist = [0.1, .15, .3, 0.7, 1.5] if focus_dist is None else focus_dist

        self.root_dir = root_dir
        self.transform_fnc = transform_fnc
        self.flag_shuffle = flag_shuffle

        self.flag_rgb = flag_inputs[0]
        self.flag_coc = flag_inputs[1]

        self.img_num = img_num
        self.data_ratio = data_ratio

        self.flag_out_coc = flag_outputs[0]
        self.flag_out_depth = flag_outputs[1]

        self.focus_dist = focus_dist

        # Load and sort all images
        self.imglist_all = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "All.tif"]
        self.imglist_dpt = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "Dpt.exr"]

        print("Total number of samples", len(self.imglist_dpt), "  Total number of seqs",
              len(self.imglist_dpt) / img_num)

        self.imglist_all.sort()
        self.imglist_dpt.sort()

        self.camera = CameraLens(2.9 * 1e-3, f_number=f_number)
        self.max_dpt = max_dpt

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, idx):
        # Read and process an image
        idx_dpt = int(idx)
        img_dpt = read_dpt(self.root_dir + self.imglist_dpt[idx_dpt])
        img_dpt = np.clip(img_dpt, 0., self.max_dpt)
        mat_dpt = img_dpt / self.max_dpt

        mat_dpt = mat_dpt.copy()[:, :, np.newaxis]

        ind = idx * self.img_num

        num_list = list(range(self.img_num))
        if self.data_ratio == 1:
            num_list = [0, 1, 2, 3, 4]
        if self.flag_shuffle:
            random.shuffle(num_list)

        # add RGB, CoC, Depth inputs
        mats_input = np.zeros((256, 256, 0))
        mats_output = np.zeros((256, 256, 0))

        for i in range(self.img_num):
            if self.flag_rgb:
                im = Image.open(self.root_dir + self.imglist_all[ind + num_list[i]])
                img_all = np.array(im)
                mat_all = img_all.copy() / 255.
                mats_input = np.concatenate((mats_input, mat_all), axis=2)

            if self.flag_coc or self.flag_out_coc:
                img_msk = self.camera.get_coc(self.focus_dist[i], img_dpt)
                img_msk = np.clip(img_msk, 0, 1.0e-4) / 1.0e-4
                mat_msk = img_msk.copy()[:, :, np.newaxis]
                if self.flag_coc:
                    mats_input = np.concatenate((mats_input, mat_msk), axis=2)
                if self.flag_out_coc:
                    mats_output = np.concatenate((mats_output, mat_msk), axis=2)

        if self.flag_out_depth:
            mats_output = np.concatenate((mats_output, mat_dpt), axis=2)

        sample = {'input': mats_input, 'output': mats_output}

        if self.transform_fnc:
            sample = self.transform_fnc(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        mats_input, mats_output = sample['input'], sample['output']

        mats_input = mats_input.transpose((2, 0, 1))
        mats_output = mats_output.transpose((2, 0, 1))
        return {'input': torch.from_numpy(mats_input),
                'output': torch.from_numpy(mats_output), }


def load_data():
    # data_dir = DATA_PARAMS['DATA_PATH'] + DATA_PARAMS['DATA_SET'] + str(DATA_PARAMS['DATA_NUM']) + '\\'
    data_dir = DATA_PARAMS['DATA_PATH'] + '/'
    img_dataset = ImageDataset(root_dir=data_dir,
                               transform_fnc=transforms.Compose([ToTensor()]),
                               flag_shuffle=DATA_PARAMS['FLAG_SHUFFLE'],
                               img_num=DATA_PARAMS['INP_IMG_NUM'],
                               data_ratio=DATA_PARAMS['DATA_RATIO_STRATEGY'],
                               flag_inputs=[DATA_PARAMS['FLAG_IO_DATA']['INP_RGB'],
                                            DATA_PARAMS['FLAG_IO_DATA']['INP_COC']],
                               flag_outputs=[DATA_PARAMS['FLAG_IO_DATA']['OUT_COC'],
                                             DATA_PARAMS['FLAG_IO_DATA']['OUT_DEPTH']],
                               focus_dist=DATA_PARAMS['FOCUS_DIST'],
                               f_number=DATA_PARAMS['F_NUMBER'],
                               max_dpt=DATA_PARAMS['MAX_DPT'])

    indices = list(range(len(img_dataset)))
    split = int(len(img_dataset) * DATA_PARAMS['TRAIN_SPLIT'])

    indices_train = indices[:split]
    indices_valid = indices[split:]

    dataset_train = torch.utils.data.Subset(img_dataset, indices_train)
    dataset_valid = torch.utils.data.Subset(img_dataset, indices_valid)

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=DATA_PARAMS['WORKERS_NUM'],
                                               batch_size=DATA_PARAMS['BATCH_SIZE'],
                                               shuffle=DATA_PARAMS['DATASET_SHUFFLE'])
    loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, num_workers=1, batch_size=1, shuffle=False)

    total_steps = int(len(dataset_train) / DATA_PARAMS['BATCH_SIZE'])
    print("Total number of steps per epoch:", total_steps)
    print("Total number of training sample:", len(dataset_train))

    return [loader_train, loader_valid], total_steps


def weights_init(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)


def load_model(model_dir, model_name):
    arch = importlib.import_module('arch.dofNet_arch' + str(TRAIN_PARAMS['ARCH_NUM']))

    ch_inp_num = 0
    if DATA_PARAMS['FLAG_IO_DATA']['INP_RGB']:
        ch_inp_num += 3
    if DATA_PARAMS['FLAG_IO_DATA']['INP_COC']:
        ch_inp_num += 1

    ch_out_num = 0

    if DATA_PARAMS['FLAG_IO_DATA']['OUT_DEPTH']:
        ch_out_num += 1
    ch_out_num_all = ch_out_num
    if DATA_PARAMS['FLAG_IO_DATA']['OUT_COC']:
        ch_out_num_all = ch_out_num + 1 * DATA_PARAMS['INP_IMG_NUM']
        ch_out_num += 1

    total_ch_inp = ch_inp_num * DATA_PARAMS['INP_IMG_NUM']
    if TRAIN_PARAMS['ARCH_NUM'] > 0:
        total_ch_inp = ch_inp_num

        flag_step2 = False
        if TRAIN_PARAMS['TRAINING_MODE'] == 2:
            flag_step2 = True
        model = arch.AENet(total_ch_inp, 1, TRAIN_PARAMS['FILTER_NUM'], flag_step2=flag_step2)
    else:
        model = arch.AENet(total_ch_inp, ch_out_num_all, TRAIN_PARAMS['FILTER_NUM'])
    model.apply(weights_init)

    params = list(model.parameters())
    print("model.parameters()", len(params))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable params/Total number:",
          str(pytorch_total_params_train) + "/" + str(pytorch_total_params))

    if TRAIN_PARAMS['EPOCH_START'] > 0:
        model.load_state_dict(torch.load(model_dir + model_name + '_ep' + str(TRAIN_PARAMS['EPOCH_START']) + '.pth'))
        print("Model loaded:", model_name, " epoch:", str(TRAIN_PARAMS['EPOCH_START']))

    return model, ch_inp_num, ch_out_num


def forward_pass(X, model_info, stacknum=1, additional_input=None):
    # to train with random number of inputs
    if TRAIN_PARAMS['RANDOM_LEN_INPUT'] == 1 and stacknum < DATA_PARAMS['INP_IMG_NUM']:
        X[:, model_info['inp_ch_num'] * stacknum:, :, :] = torch.zeros(
            [X.shape[0], (DATA_PARAMS['INP_IMG_NUM'] - stacknum) * model_info['inp_ch_num'], X.shape[2], X.shape[3]])

    flag_step2 = True if TRAIN_PARAMS['TRAINING_MODE'] == 2 else False

    outputs = model_info['model'](X, model_info['inp_ch_num'], stacknum, flag_step2=flag_step2, x2=additional_input)

    return (outputs[1], outputs[0]) if TRAIN_PARAMS['TRAINING_MODE'] == 2 else (outputs, outputs)


def train_model(loaders, model_info):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model_info['model_params'], lr=TRAIN_PARAMS['LEARNING_RATE'])

    focus_dists = DATA_PARAMS['FOCUS_DIST']

    # Training
    print("Total number of epochs:", TRAIN_PARAMS['EPOCHS_NUM'])
    for e_iter in range(TRAIN_PARAMS['EPOCHS_NUM'] - TRAIN_PARAMS['EPOCH_START']):
        epoch_iter = e_iter + TRAIN_PARAMS['EPOCH_START']
        loss_sum, iter_count = 0, 0

        for st_iter, sample_batch in enumerate(loaders[0]):

            # Setting up input and output data
            X = sample_batch['input'].float().to(model_info['device_comp'])
            Y = sample_batch['output'].float().to(model_info['device_comp'])
            optimizer.zero_grad()

            gt_step1, gt_step2 = None, None

            if TRAIN_PARAMS['TRAINING_MODE'] == 2:
                gt_step1 = Y[:, :-1, :, :]
                gt_step2 = Y[:, -1:, :, :]

            stacknum = DATA_PARAMS['INP_IMG_NUM']
            if TRAIN_PARAMS['RANDOM_LEN_INPUT'] > 0:
                stacknum = np.random.randint(1, DATA_PARAMS['INP_IMG_NUM'])
            Y = Y[:, :stacknum, :, :]
            gt_step1 = gt_step1[:, :stacknum, :, :]

            # Focus distance maps
            X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
            for t in range(stacknum):
                if DATA_PARAMS['FLAG_IO_DATA']['INP_DIST']:
                    focus_distance = focus_dists[t] / focus_dists[-1]
                    X2_fcs[:, t:(t + 1), :, :] = X2_fcs[:, t:(t + 1), :, :] * (focus_distance)
            X2_fcs = X2_fcs.float().to(model_info['device_comp'])

            # Forward and compute loss
            output_step1, output_step2 = forward_pass(X, model_info, stacknum=stacknum, additional_input=X2_fcs)

            loss = None

            if TRAIN_PARAMS['TRAINING_MODE'] == 2:
                loss_step1, loss_step2 = 0, 0
                if DATA_PARAMS['FLAG_IO_DATA']['OUT_COC']:
                    loss_step1 = criterion(output_step1, gt_step1)
                if DATA_PARAMS['FLAG_IO_DATA']['OUT_DEPTH']:
                    loss_step2 = criterion(output_step2, gt_step2)
                loss = loss_step1 * TRAIN_PARAMS['MODEL1_LOSS_WEIGHT'] + loss_step2
            elif TRAIN_PARAMS['TRAINING_MODE'] == 1:
                loss = criterion(output_step1, Y)

            loss.backward()
            optimizer.step()

            # Training log
            loss_sum += loss.item()
            iter_count += 1.

            if (st_iter + 1) % 5 == 0:
                run.log('loss', loss_sum / iter_count)
                print(model_info['model_name'], 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch_iter + 1, TRAIN_PARAMS['EPOCHS_NUM'], st_iter + 1, model_info['total_steps'],
                              loss_sum / iter_count))

        # Save model
        if (epoch_iter + 1) % 100 == 0:
            torch.save(model_info['model'].state_dict(),
                       model_info['model_dir'] + model_info['model_name'] + '_ep' + str(epoch_iter + 1) + '.pth')

    torch.save(model_info['model'].state_dict(),
               model_info['model_dir'] + model_info['model_name'] + '_final.pth')


def run_exp(data_path=None, epochs=None):
    if data_path is not None:
        DATA_PARAMS['DATA_PATH'] = data_path
    if epochs is not None:
        TRAIN_PARAMS['EPOCHS_NUM'] = epochs

    # Initial preparations
    model_dir, model_name = set_output_folders()
    device_comp = set_comp_device()

    # Training initializations
    loaders, total_steps = load_data()

    model, inp_ch_num, out_ch_num = load_model(model_dir, model_name)
    model = model.to(device=device_comp)
    model_params = model.parameters()

    model_info = {'model': model,
                  'model_dir': model_dir,
                  'model_name': model_name,
                  'total_steps': total_steps,
                  'inp_ch_num': inp_ch_num,
                  'out_ch_num': out_ch_num,
                  'device_comp': device_comp,
                  'model_params': model_params,
                  }
    print("inp_ch_num", inp_ch_num, "   out_ch_num", out_ch_num)

    # Run training
    train_model(loaders=loaders, model_info=model_info)


if __name__ == '__main__':
    run_exp()

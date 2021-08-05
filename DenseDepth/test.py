# This file was modified as part of this research project. It was copied and modified from the original `test.py`
# provided by the original authors of DenseDepth

import os
import glob
import argparse
import matplotlib
import numpy as np
import time

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, save_images
from matplotlib import pyplot as plt
from skimage.transform import resize
from PIL import Image

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--output', default='outputs', type=str, help='Output path.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs, names = load_images( glob.glob(args.input) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
tic = time.time()
outputs = predict(model, inputs)
toc = time.time()
avg_time = (toc-tic)/len(names)

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
os.makedirs(args.output, exist_ok=True)
for idx, image in enumerate(outputs):
  out_image = resize(image, (512, 512), order=1, preserve_range=True, mode='reflect', anti_aliasing=True)
  
  np.save(f"{args.output}/{names[idx]}.npy", out_image)

  save_images(f"{args.output}/{names[idx]}.png", out_image)

  with open(f"{args.output}/{names[idx]}.txt", 'w') as f:
    f.write(str(avg_time))

# viz = display_images(outputs.copy(), inputs.copy())
# plt.figure(figsize=(10,5))
# plt.imshow(viz)
# plt.savefig('test.png')
# plt.show()

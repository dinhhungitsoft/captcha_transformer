import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import random

# SEED = 1234

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
N_EPOCHS = 400
LEARNING_RATE =3e-4
CLIP = 1
# SRC = utils.SRC
# TRG = utils.TRG
INPUT_DIM = 145
OUTPUT_DIM = 22
HID_DIM = 64
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 4
DEC_HEADS = 4
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

DATA_DIR = "/home/hung/learn/pytorch/captcha_trainsformer/data/captcha_images_v2"
BATCH_SIZE = 16
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
NUM_WORKERS = 4
EPOCH = 200
DEVICE = "cpu"
# NUM_CHARS = 21
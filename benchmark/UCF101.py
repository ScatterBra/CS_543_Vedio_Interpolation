import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from benchmark.pytorch_msssim import ssim_matlab
from rife.model.RIFE import Model

parser = argparse.ArgumentParser(description='Evaluate RIFE on UCF101')
parser.add_argument('--model', required=True, help='Path to the model weights file')
parser.add_argument('--data', required=True, help='Path to the ucf101_interp_ours directory')
args = parser.parse_args()

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_model(args.model)
model.eval()
model.device()

path = args.data
dirs = os.listdir(path)
psnr_list = []
ssim_list = []
print(len(dirs))
for d in dirs:
    img0 = os.path.join(path, d, 'frame_00.png')
    img1 = os.path.join(path, d, 'frame_02.png')
    gt = os.path.join(path, d, 'frame_01_gt.png')
    img0 = (torch.tensor(cv2.imread(img0).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    img1 = (torch.tensor(cv2.imread(img1).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    gt = (torch.tensor(cv2.imread(gt).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    pred = model.inference(img0, img1)[0]
    ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    out = pred.detach().cpu().numpy().transpose(1, 2, 0)
    out = np.round(out * 255) / 255.
    gt = gt[0].cpu().numpy().transpose(1, 2, 0)
    psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))

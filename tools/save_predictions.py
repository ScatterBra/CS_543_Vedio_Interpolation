"""保存 RIFE 在 UCF101 和 Middlebury 上的预测帧和对比图。

不修改官方 benchmark 脚本。所用的模型、输入、补边和取整方式都与
benchmark/UCF101.py、benchmark/MiddleBury_Other.py 相同，
所以 metrics.csv 里各样本分数的平均值就等于 benchmark 的结果。

用法：
    python save_predictions.py --rife ~/work/ECCV2022-RIFE                 # 两个数据集都跑
    python save_predictions.py --rife ~/work/ECCV2022-RIFE --only ucf101   # 只跑 UCF101
    python save_predictions.py --rife ~/work/ECCV2022-RIFE --out ./outputs # 改输出目录

--rife 是官方 RIFE 仓库的根目录，模型（train_log/）和数据要按教程放在里面。
不写 --rife 时，默认用当前所在目录。--out 默认是 <RIFE 仓库>/outputs。

输出（<out>/<数据集>/ 下）：
  <样本>/pred.png      预测出的中间帧
  <样本>/compare.png   第一行：输入帧 0 | 真值 | RIFE 预测 | 输入帧 1
                       第二行：样本编号和分数 | 真值 | RIFE 预测 | 误差热力图（|预测-真值| x10）
  metrics.csv          每个样本的分数，误差最大的排在最前
"""
import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np
import torch

parser = argparse.ArgumentParser(description='保存 RIFE 预测结果和对比图')
parser.add_argument('--rife', default='.', help='官方 RIFE 仓库根目录（默认当前目录）')
parser.add_argument('--out', default=None, help='输出目录（默认 <RIFE 仓库>/outputs）')
parser.add_argument('--only', choices=['ucf101', 'middlebury'], help='只跑一个数据集')
args = parser.parse_args()

REPO = os.path.abspath(os.path.expanduser(args.rife))
if not os.path.isfile(os.path.join(REPO, 'model', 'RIFE.py')):
    sys.exit(f'找不到 {REPO}/model/RIFE.py，请用 --rife 指向官方 RIFE 仓库的根目录')
if not os.path.isfile(os.path.join(REPO, 'train_log', 'flownet.pkl')):
    sys.exit(f'找不到 {REPO}/train_log/flownet.pkl，请先按教程下载论文模型')
args.out = os.path.abspath(os.path.expanduser(args.out)) if args.out else os.path.join(REPO, 'outputs')
os.chdir(REPO)  # 和 benchmark 一样，数据用相对路径读取
sys.path.insert(0, REPO)
from model.RIFE import Model  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.load_model('train_log')
model.eval()
model.device()
torch.set_grad_enabled(False)


def to_t(img):
    return torch.tensor(img.transpose(2, 0, 1) / 255.).to(device).float().unsqueeze(0)


def label(img, text):
    # cv2.putText 不支持中文，图上的标签用英文
    img = img.copy()
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 9 * len(text) + 10), 20), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def compare(f0, gt, pred, f1, title):
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(2)
    err_vis = cv2.applyColorMap(np.clip(err * 10, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    blank = np.zeros_like(gt)
    row1 = np.hstack([label(f0, 'frame 0 (input)'), label(gt, 'ground truth'),
                      label(pred, 'RIFE prediction'), label(f1, 'frame 1 (input)')])
    row2 = np.hstack([label(blank, title), label(gt, 'ground truth'),
                      label(pred, 'RIFE prediction'), label(err_vis, '|pred-gt| x10')])
    return np.vstack([row1, row2])


def run_ucf(out_root):
    path = 'UCF101/ucf101_interp_ours/'
    out_dir = os.path.join(out_root, 'ucf101')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for d in sorted(os.listdir(path), key=int):
        f0 = cv2.imread(path + d + '/frame_00.png')
        f1 = cv2.imread(path + d + '/frame_02.png')
        gt = cv2.imread(path + d + '/frame_01_gt.png')
        pred = model.inference(to_t(f0), to_t(f1))[0]
        pred8 = np.round(pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        mse = (((gt / 255.) - (pred8 / 255.)) ** 2).mean()  # 与 benchmark 相同：H×W×3 个值一起平均
        psnr = -10 * math.log10(mse)
        sd = os.path.join(out_dir, d)
        os.makedirs(sd, exist_ok=True)
        cv2.imwrite(os.path.join(sd, 'pred.png'), pred8)
        cv2.imwrite(os.path.join(sd, 'compare.png'), compare(f0, gt, pred8, f1, f'#{d} PSNR {psnr:.2f}'))
        rows.append((d, psnr))
    rows.sort(key=lambda r: r[1])
    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'psnr_db'])
        w.writerows((r[0], f'{r[1]:.3f}') for r in rows)
    print(f'UCF101：{len(rows)} 组，平均 PSNR {np.mean([r[1] for r in rows]):.3f}')


def run_middlebury(out_root):
    names = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper',
             'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
    out_dir = os.path.join(out_root, 'middlebury')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for n in names:
        f0 = cv2.imread(f'other-data/{n}/frame10.png')
        f1 = cv2.imread(f'other-data/{n}/frame11.png')
        gt = cv2.imread(f'other-gt-interp/{n}/frame10i11.png')
        h, w = f0.shape[:2]
        imgs = torch.zeros([1, 6, 480, 640]).to(device)  # 与 benchmark 相同：补零到 640×480
        imgs[:, :3, :h, :w] = to_t(f0)
        imgs[:, 3:, :h, :w] = to_t(f1)
        pred = model.inference(imgs[:, :3], imgs[:, 3:])[0]
        pred8 = np.round(pred.cpu().numpy().transpose(1, 2, 0)[:h, :w] * 255).astype(np.uint8)
        ie = np.abs(pred8.astype(np.float64) - gt).mean()
        sd = os.path.join(out_dir, n)
        os.makedirs(sd, exist_ok=True)
        cv2.imwrite(os.path.join(sd, 'pred.png'), pred8)
        cv2.imwrite(os.path.join(sd, 'compare.png'), compare(f0, gt, pred8, f1, f'{n} IE {ie:.3f}'))
        rows.append((n, ie))
    rows.sort(key=lambda r: -r[1])
    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'ie'])
        w.writerows((r[0], f'{r[1]:.3f}') for r in rows)
    print(f'Middlebury：{len(rows)} 个场景，平均 IE {np.mean([r[1] for r in rows]):.3f}')


if args.only in (None, 'middlebury'):
    run_middlebury(args.out)
if args.only in (None, 'ucf101'):
    run_ucf(args.out)

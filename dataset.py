import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, data_root, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 448
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
           
    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = 0.5
        return img0, gt, img1, timestep
    
        # RIFEm with Vimeo-Septuplet
        # imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
        # ind = [0, 1, 2, 3, 4, 5, 6]
        # random.shuffle(ind)
        # ind = ind[:3]
        # ind.sort()
        # img0 = cv2.imread(imgpaths[ind[0]])
        # gt = cv2.imread(imgpaths[ind[1]])
        # img1 = cv2.imread(imgpaths[ind[2]])        
        # timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
            
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        if self.dataset_name == 'train':
            img0, gt, img1 = self.crop(img0, gt, img1, 224, 224)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            # random rotation
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep


class GameDataset(Dataset):
    def __init__(self, dataset_name, data_root, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = 720
        self.w = 1280
        self.data_root = data_root
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        val_fn = os.path.join(self.data_root, 'tri_vallist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r', encoding='utf-8') as f:
            self.trainlist = f.read().splitlines()
        with open(val_fn, 'r', encoding='utf-8') as f:
            self.vallist = f.read().splitlines()
        with open(test_fn, 'r', encoding='utf-8') as f:
            self.testlist = f.read().splitlines()
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        elif self.dataset_name == 'validation':
            self.meta_data = self.vallist
        else:
            raise ValueError(f'Unknown dataset split: {self.dataset_name}')
        self.meta_data = [line.strip() for line in self.meta_data if line.strip()]
        if not self.meta_data:
            raise ValueError(f'No {self.dataset_name} samples listed in {self.data_root}')

    def crop(self, img0, gt, img1, h, w, flow, invdepth0, invdepth2):
        ih, iw, _ = img0.shape
        if ih < h or iw < w:
            raise ValueError(f'The training crop {h}x{w} exceeds the image size {ih}x{iw}')
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        flow = flow[x:x+h, y:y+w, :]
        invdepth0 = invdepth0[x:x+h, y:y+w]
        invdepth2 = invdepth2[x:x+h, y:y+w]
        return img0, gt, img1, flow, invdepth0, invdepth2

    def getimg(self, index):
        imgpath = os.path.join(self.data_root, self.meta_data[index])
        imgpaths = [os.path.join(imgpath, name) for name in ('I0.png', 'I1_gt.png', 'I2.png')]
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        if img0 is None or img1 is None or gt is None:
            raise ValueError(f'Could not read the image triplet in {imgpath}')

        with np.load(os.path.join(imgpath, 'flow_I2_to_I0.npz')) as f:
            flow = f['flow'].astype(np.float32)
        with np.load(os.path.join(imgpath, 'invdepth_I0.npz')) as f:
            invdepth0 = f['invdepth'].astype(np.float32)
        with np.load(os.path.join(imgpath, 'invdepth_I2.npz')) as f:
            invdepth2 = f['invdepth'].astype(np.float32)
        ih, iw = img0.shape[:2]
        if (img1.shape != img0.shape or gt.shape != img0.shape
                or flow.shape != (ih, iw, 2)
                or invdepth0.shape != (ih, iw) or invdepth2.shape != (ih, iw)):
            raise ValueError(f'Image and buffer dimensions do not match in {imgpath}')
        timestep = 0.5
        return img0, gt, img1, timestep, flow, invdepth0, invdepth2

    def __getitem__(self, index):
        img0, gt, img1, timestep, flow, invdepth0, invdepth2 = self.getimg(index)
        if self.dataset_name == 'train':
            img0, gt, img1, flow, invdepth0, invdepth2 = self.crop(
                img0, gt, img1, 224, 224, flow, invdepth0, invdepth2
            )
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                flow = flow[::-1]
                flow[..., 1] *= -1
                invdepth0 = invdepth0[::-1]
                invdepth2 = invdepth2[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                flow = flow[:, ::-1]
                flow[..., 0] *= -1
                invdepth0 = invdepth0[:, ::-1]
                invdepth2 = invdepth2[:, ::-1]
            # Keep endpoint order because the dataset only contains I2-to-I0 flow.

            p = random.uniform(0, 1)
            rotation = None
            if p < 0.25:
                rotation = cv2.ROTATE_90_CLOCKWISE
            elif p < 0.5:
                rotation = cv2.ROTATE_180
            elif p < 0.75:
                rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            if rotation is not None:
                img0 = cv2.rotate(img0, rotation)
                gt = cv2.rotate(gt, rotation)
                img1 = cv2.rotate(img1, rotation)
                flow = cv2.rotate(flow, rotation)
                invdepth0 = cv2.rotate(invdepth0, rotation)
                invdepth2 = cv2.rotate(invdepth2, rotation)
                if rotation == cv2.ROTATE_90_CLOCKWISE:
                    flow = flow[..., ::-1]
                    flow[..., 0] *= -1
                elif rotation == cv2.ROTATE_180:
                    flow *= -1
                else:
                    flow = flow[..., ::-1]
                    flow[..., 1] *= -1

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return {
            'data': torch.cat((img0, img1, gt), 0),
            'timestep': timestep,
            'flow_I2_to_I0': torch.from_numpy(flow.copy()).permute(2, 0, 1),
            'invdepth_I0': torch.from_numpy(invdepth0.copy()).unsqueeze(0),
            'invdepth_I2': torch.from_numpy(invdepth2.copy()).unsqueeze(0),
        }

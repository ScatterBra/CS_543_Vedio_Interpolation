# Load one sample group and check the motion-vector convention: warp I0 to I2's grid with flow_I2_to_I0 and compare.
# Usage: python check_sample.py <group folder>      (needs numpy and Pillow only)
#   e.g. python check_sample.py CloverSweep_0445
import sys, json, numpy as np
from PIL import Image

g = sys.argv[1] if len(sys.argv) > 1 else 'CloverSweep_0445'
rgb = lambda n: np.asarray(Image.open(f'{g}/{n}.png'), np.float32) / 255            # (720, 1280, 3), 0..1
I0, I1, I2 = rgb('I0'), rgb('I1_gt'), rgb('I2')
flow = np.load(f'{g}/flow_I2_to_I0.npz')['flow'].astype(np.float32)                # (720, 1280, 2) px, [..., 0] = x, [..., 1] = y
inv0 = np.load(f'{g}/invdepth_I0.npz')['invdepth'].astype(np.float32)              # (720, 1280) 1/m, sky ~ 0
inv2 = np.load(f'{g}/invdepth_I2.npz')['invdepth'].astype(np.float32)
H, W = inv2.shape; ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)


def sample(img, x, y):  # bilinear, clamped at the border
    x = np.clip(x, 0, W - 1); y = np.clip(y, 0, H - 1); x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1); wx = x - x0; wy = y - y0
    if img.ndim == 3: wx = wx[..., None]; wy = wy[..., None]
    return (img[y0, x0] * (1 - wx) + img[y0, x1] * wx) * (1 - wy) + (img[y1, x0] * (1 - wx) + img[y1, x1] * wx) * wy


qx, qy = xs + flow[..., 0], ys + flow[..., 1]                                       # where each I2 pixel was in I0
inside = (qx >= 0) & (qx <= W - 1) & (qy >= 0) & (qy <= H - 1)
I2_from_I0 = sample(I0, qx, qy)
# visible in I0 too = the I0 surface found there has about the same inverse depth (within 15 %: the camera moves
# in 1/30 s, so depth does change a little, more for near objects and fast forward motion)
vis = inside & (np.abs(sample(inv0, qx, qy) - inv2) <= 0.15 * np.maximum(inv2, 1e-4))
psnr = lambda a, b, m: -10 * np.log10(((a - b) ** 2)[m].mean())
print(json.load(open(f'{g}/meta.json', encoding='utf-8'))['why'])
print(f'no warp, I0 vs I2:                    {psnr(I0, I2, np.ones((H, W), bool)):.2f} dB')
print(f'I0 warped by the flow vs I2, all px:  {psnr(I2_from_I0, I2, inside):.2f} dB  ({100 * inside.mean():.1f}% of pixels land inside I0)')
print(f'same, only px also visible in I0:     {psnr(I2_from_I0, I2, vis):.2f} dB  ({100 * vis.mean():.1f}% of pixels)')
print(f'flow magnitude median / max:          {np.median(np.hypot(*flow.transpose(2, 0, 1))):.1f} / {np.hypot(*flow.transpose(2, 0, 1)).max():.1f} px')

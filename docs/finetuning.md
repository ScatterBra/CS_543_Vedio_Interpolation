# 使用游戏数据微调 RIFE

## 数据

在 `--data` 指定的根目录下放置 `tri_trainlist.txt`、`tri_vallist.txt` 和 `tri_testlist.txt`，分别用于训练、验证和测试。每行写一组样本相对于根目录的路径，例如 `CloverSweep_0445`。

## 单卡微调

```bash
cd /workspace
python3 -m torch.distributed.launch --nproc_per_node=1 train.py \
  --world_size=1 --game_data --data sample_data \
  --pretrained /rife/train_log/flownet.pkl --batch_size 1 --epoch 5 \
  --output outputs/finetuning/model.pkl
```

- 正式训练时，将 `--data sample_data` 换成完整数据集的根目录，并保留 `--game_data`。
- `--batch_size` 是每张 GPU 的训练批大小，按显存调整。`--epoch` 是训练轮数，按实验设置调整。
- `--pretrained` 指向已有的 RIFE 权重。`--output` 指定新权重的保存路径，换实验时可以换一个输出目录。
- 单卡时 `--nproc_per_node` 和 `--world_size` 都是 `1`。单机多卡时，两者都设为使用的 GPU 数。

## TensorBoard

```bash
tensorboard --logdir outputs/finetuning/logs
```

如果更换了 `--output` 的父目录，把 `--logdir` 改为新目录下的 `logs`。在浏览器打开终端显示的地址。

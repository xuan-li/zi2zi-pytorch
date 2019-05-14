from data import DatasetFromObj
from torch.utils.data import DataLoader
from model import Zi2ZiModel

import torch
dataset = DatasetFromObj('dataset/val.obj')
dataloader = DataLoader(dataset, batch_size=20,
                        shuffle=True, num_workers=1)
batch = next(iter(dataloader))
model = Zi2ZiModel(embedding_num=41, save_dir='checkpoints')
model.set_input(batch[0], batch[2], batch[1])
model.print_networks(True)
model.load_networks(0)



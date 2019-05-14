from data import DatasetFromObj
from torch.utils.data import DataLoader
from model import UNetGenerator
from model import Discriminator
import torch
dataset = DatasetFromObj('dataset/val.obj')
dataloader = DataLoader(dataset, batch_size=20,
                        shuffle=True, num_workers=1)
batch = next(iter(dataloader))
G = UNetGenerator(is_training=False)

fake, enc = G(torch.randn(20,128), batch[2])
print(fake.shape)

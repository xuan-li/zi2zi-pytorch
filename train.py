from data import DatasetFromObj
from torch.utils.data import DataLoader
from model import UNetGenerator
from model import Discriminator
dataset = DatasetFromObj('dataset/val.obj')
dataloader = DataLoader(dataset, batch_size=20,
                        shuffle=True, num_workers=1)
batch = next(iter(dataloader))
G = UNetGenerator(embedding_num=41)
D = Discriminator(6,embedding_num=41)
print(batch[0].max())
enc = G.encoder(batch[1])
print(enc.shape)
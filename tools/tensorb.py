from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from tabulate import tabulate
import torchvision.utils as tu

c=[[i*j for i in range (20)] for j in range(10)]
b=torch.rand(120,40,3)*255
a = SummaryWriter(log_dir= "tb_test")
for i in range(10):
    a.add_text("tester",tabulate(c),i)
    a.add_image("lalalalala", (b**i), i, dataformats="HWC")
print(str(tabulate(c)))
a.close()

import numpy as np
import cv2
from debug.convcrf import GaussCRF, get_default_conf
import torch

# output = np.ones((2,300,300))*0.5

output = np.zeros((2,300,300))
for x in range(300):
    for y in range(300):
        r=((x-150)**2+(y-150)**2)**0.5
        if(r<55):
            output[0][x][y]=1
            output[1][x][y]=0

        elif(r>=55 and r<75):
            output[0][x][y]=1-0.5*(r-55)/(75-55)
            output[1][x][y]=1-output[0][x][y]
        else:
            output[0][x][y]=max(0.5-0.8*(r-75)/(125),0)
            output[1][x][y]=1-output[0][x][y]
output/=0.5
output = torch.Tensor(output).cuda()
output = output.unsqueeze(0)
# output = -torch.log(output)


img_path="debug/ok.png"
raw_image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32).transpose(2, 0, 1)
raw_image = torch.from_numpy(raw_image).to("cuda")
raw_image = raw_image.unsqueeze(dim=0)
print(raw_image.shape)
# output shape is [1,21,w,h]
num_classes=output.shape[1]
crf = GaussCRF(conf=get_default_conf(), shape=raw_image.shape[2:], nclasses=num_classes, use_gpu=True)
crf = crf.to("cuda")
output = crf.forward(output, raw_image)

import matplotlib.pyplot as plt
plt.imshow(output[0][0].cpu().numpy(),cmap="nipy_spectral")        
plt.savefig("debug/test_crf_fg.jpg")
plt.imshow(output[0][1].cpu().numpy(),cmap="nipy_spectral")        
plt.savefig("debug/test_crf_bg.jpg")

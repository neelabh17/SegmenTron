import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
prob = np.zeros((2,300,300))

for x in range(300):
    for y in range(300):
        r=((x-150)**2+(y-150)**2)**0.5
        if(r<55):
            prob[0][x][y]=1
            prob[1][x][y]=0

        elif(r>=55 and r<75):
            prob[0][x][y]=1-0.5*(r-55)/(75-55)
            prob[1][x][y]=1-prob[0][x][y]
        else:
            prob[0][x][y]=max(0.5-0.8*(r-75)/(125),0)
            prob[1][x][y]=1-prob[0][x][y]


# prob=np.ones((2,300,300))

# for x in range(150-150,150+150):
#     for y in range(150-150,150+150):
#         prob[0][x][y]=0
#         prob[1][x][y]=1
# d=90
# for x in range(150-d,150+d):
#     for y in range(150-d,150+d):
#         prob[0][x][y]=0.5
#         prob[1][x][y]=0.5
# d=80
# for x in range(150-d,150+d):
#     for y in range(150-d,150+d):
#         prob[0][x][y]=0.75
#         prob[1][x][y]=0.25
# d=50
# for x in range(150-d,150+d):
#     for y in range(150-d,150+d):
#         prob[0][x][y]=1
#         prob[1][x][y]=0




plt.imshow(torch.Tensor(prob).softmax(dim=0)[0].numpy(),cmap="nipy_spectral")        
plt.savefig("debug/fg.jpg")
plt.imshow(torch.Tensor(prob).softmax(dim=0)[1].numpy(),cmap="nipy_spectral")        
plt.savefig("debug/bg.jpg")
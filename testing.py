from calibration_library.metrics import CCELoss as perimageCCE
from calibration_library.cce_loss import CCELoss

a = CCELoss(19,10)
b = perimageCCE(19,10)

import torch
img = torch.rand(4,19,769,769).cuda()
target = torch.randint(0,20,(4,769,769)).cuda()
a.forward(img,target)

with torch.no_grad():
    for output, target in zip(img,target.detach()):
        #older ece requires softmax and size output=[class,w,h] target=[w,h]
        b.update(output.softmax(dim=0), target)

b.get_overall_CCELoss()


img = torch.rand(4,19,769,769).cuda()
target = torch.randint(0,20,(4,769,769)).cuda()
a.forward(img,target)

with torch.no_grad():
    for output, target in zip(img,target.detach()):
        #older ece requires softmax and size output=[class,w,h] target=[w,h]
        b.update(output.softmax(dim=0), target)

(b.get_overall_CCELoss())

a.get_overall_CCELoss()
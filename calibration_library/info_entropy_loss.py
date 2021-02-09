import torch
import numpy as np
import torch.nn.functional as F

# @neelabh17 implementation


class InfoEntropyLoss(torch.nn.Module):
    
    def forward(self , output):
        '''
        output = [batch, n_Class, h , w] np array: The complete logit vector of an image 
        
        return the entropy calculated across dim 1 and the mean it out across all batch and pixels
        '''

        b = F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
        b = -1.0 * b.sum(1)
        return b.mean()


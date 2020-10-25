import pickle
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
import pickle 
# from torch.utils.tensorboard import SummaryWriter
from calibration_library import metrics, visualization
import os
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15,name="randomtest"):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confidences, predictions , labels):
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            print(f"Currently performing evaluation for bin {[bin_lower,bin_upper]}")
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece



ece_folder="eceData"
postfix="Foggy_Calibrated_DLV3Plus"
include_bg=True
n_bins=10

saveDir=os.path.join(ece_folder,postfix)

file=open(os.path.join(saveDir,"conf.pickle"),"rb")
conf=pickle.load(file)
file.close()
print("Confidences Sucessfully Loaded")

file=open(os.path.join(saveDir,"obj.pickle"),"rb")
obj=pickle.load(file)
file.close()
print("Preditions Sucessfully Loaded")


file=open(os.path.join(saveDir,"gt.pickle"),"rb")
gt=pickle.load(file)
print("GT Sucessfully Loaded")
file.close()

print("--------------------------------\n")

print(gt.shape)


if(include_bg):
    sel=(gt!=255) 
else:
    sel=(gt!=255) * (gt!=0)

# print(sel[0].shape)

gt=gt[sel].view(-1,1)
print(gt.shape)
conf=conf[sel].view(-1,1)
obj=obj[sel].view(-1,1)
# exit()

print(f"Starting with {n_bins} bins")
# neelLoss=ECELoss(n_bins,"DeepLabV3withbg")
# neelLoss=ECELoss(n_bins,postfix)
# loss=neelLoss.forward(conf,obj,gt)
# print(f"ECE Loss={loss}")

gt=gt.numpy()
conf=conf.numpy()
obj=obj.numpy()

print("Implementing official metric library")
ece_criterion = metrics.ECELoss()
eceLoss=ece_criterion.loss(conf,obj,gt,n_bins,False)
print('ECE with probabilties %f' % (eceLoss))

file=open(os.path.join(saveDir,"Results.txt"),"a")
file.write(f"{postfix}_bin={n_bins}_incBG={str(include_bg)}\t\t\t ECE Loss: {eceLoss}\n")


plot_folder=os.path.join(saveDir,"plots")
makedirs(plot_folder)
conf_hist = visualization.ConfidenceHistogram()
plt_test = conf_hist.plot(conf,obj,gt,title="Confidence Histogram")
plt_test.savefig(os.path.join(plot_folder,f'conf_histogram_bin={n_bins}_incBG={str(include_bg)}.png'),bbox_inches='tight')
#plt_test.show()

rel_diagram = visualization.ReliabilityDiagram()
plt_test_2 = rel_diagram.plot(conf,obj,gt,title="Reliability Diagram")
plt_test_2.savefig(os.path.join(plot_folder,f'rel_diagram_bin={n_bins}_incBG={str(include_bg)}.png'),bbox_inches='tight')
#plt_test_2.show()
import numpy as np
from scipy.special import softmax
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib
sys.path.append("calibration_library")


class CELoss(object):

    def compute_bin_boundaries(self, probabilities = np.array([])):

        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]


    def get_probabilities(self, conf, obj, labels, logits=False):
        #If not probabilities apply softmax!
        # if logits:
        #     self.probabilities = softmax(output, axis=1)
        # else:
        #     self.probabilities = conf

        self.labels = labels
        self.confidences = conf
        self.predictions = obj
        self.accuracies = np.equal(self.predictions,labels)

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        #self.acc_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)


    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = self.acc_matrix[:,index]


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])
    
    def return_iterative_compute_bins(self,conf, obj, labels ):
        bin_total = np.zeros(self.n_bins)
        bin_total_correct = np.zeros(self.n_bins)
        bin_conf_total = np.zeros(self.n_bins)

        labels = labels
        confidences = conf
        predictions = obj
        accuracies = np.equal(predictions,labels) 

        
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            # self.bin_prop[i] = np.mean(in_bin)
            bin_total[i] = np.sum(in_bin)

            if bin_total[i].item() > 0:
                bin_total_correct[i] = np.sum(accuracies[in_bin])
                bin_conf_total[i] = np.sum(confidences[in_bin])

        return bin_total, bin_total_correct, bin_conf_total
    
    def update_cumulative_bins(self, bin_total,bin_total_correct, bin_conf_total):
        '''
        bin_total = n x n_bins
        bin_total_correct = n x n_bins
        bin_conf_total = n x n_bins

        '''
        
        
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)
        
        # Summing up all
        bin_total=np.sum(bin_total, axis=0)
        bin_total_correct=np.sum(bin_total_correct, axis=0)
        bin_conf_total=np.sum(bin_conf_total, axis=0)

        # calculating original metrics
        total_predictions=np.sum(bin_total)
        self.bin_prop = bin_total/total_predictions
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            if bin_total[i].item() > 0:
            
                self.bin_acc[i] = bin_total_correct[i]/bin_total[i]
                self.bin_conf[i] = bin_conf_total[i]/bin_total[i]

        self.bin_score = np.abs(self.bin_conf - self.bin_acc)

class MaxProbCELoss(CELoss):
    def loss(self, conf,obj, labels, n_bins = 10, logits = False):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(conf,obj, labels, logits)
        super().compute_bins()

    def make_bins(self,n_bins):
        self.n_bins=n_bins
        super().compute_bin_boundaries()

    def get_collective_bins(self,conf,obj, labels):
        return super().return_iterative_compute_bins(conf,obj, labels)

    def get_interative_loss(self, bin_total,bin_total_correct, bin_conf_total):
        super().update_cumulative_bins(bin_total,bin_total_correct, bin_conf_total)

class IterativeECELoss(MaxProbCELoss):

    def make_bins(self,n_bins=15):
        super().make_bins(n_bins)

    def get_collective_bins(self,conf,obj, labels):
        return super().get_collective_bins(conf,obj, labels)

    def get_interative_loss(self, bin_total,bin_total_correct, bin_conf_total):
        super().get_interative_loss(bin_total,bin_total_correct, bin_conf_total)
        print(self.bin_prop)
        # print(self.bin_acc)
        # print(self.bin_conf)

        print("-----------------------------------------")
        
        
        return np.dot(self.bin_prop,self.bin_score)


#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):

    def loss(self, conf,obj, labels, n_bins = 10, logits = False):
        super().loss(conf,obj, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_score)

class MCELoss(MaxProbCELoss):
    
    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.max(self.bin_score)

#https://arxiv.org/abs/1905.11001
#Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_conf * np.maximum(self.bin_conf-self.bin_acc,np.zeros(self.n_bins)))


#https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        sce = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            sce += np.dot(self.bin_prop,self.bin_score)

        return sce/self.n_class

class TACELoss(CELoss):

    def loss(self, output, labels, threshold = 0.01, n_bins = 15, logits = True):
        tace = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().get_probabilities(output, labels, logits)
        self.probabilities[self.probabilities < threshold] = 0
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bin_boundaries(self.probabilities[:,i]) 
            super().compute_bins(i)
            tace += np.dot(self.bin_prop,self.bin_score)

        return tace/self.n_class

#create TACELoss with threshold fixed at 0
class ACELoss(TACELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        return super().loss(output, labels, 0.0 , n_bins, logits)


# @neelabh implementation

class CCELoss():

    def createBins(self):

        #uniform bin spacing
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.avg_bin = torch.Tensor((self.bin_lowers + self.bin_uppers)/2).cuda()


    def update(self, output, target):
        '''
        create an three array of [n_class, n_bins]

        -> Number of prediciton array for that specification
        -> Number of correct prediction for that class
        -> Percentge of correct 
        '''
        # print(self.idealMap.shape, target.shape)
        assert self.idealMap.shape[1:] == target.shape 

        no_pred = torch.Tensor(np.array([]).reshape(output.shape[0],-1)).cuda()
        no_acc = torch.Tensor(np.array([]).reshape(output.shape[0],-1)).cuda()
        conf_sum = torch.Tensor(np.array([]).reshape(output.shape[0],-1)).cuda()
        # print(output.shape)

        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):

            mask = (output> bin_lower) * (output<= bin_upper)
            
            output_clone = output.clone()
            output_clone[ ~ mask] = 0
            # import pdb; pdb.set_trace()
            conf_sum_in_bin = torch.sum(output_clone, dim = (1,2)).reshape(-1,1)
            # print(conf_sum.shape, conf_sum_in_bin.shape)
            conf_sum = torch.cat((conf_sum , conf_sum_in_bin ), dim =1)

            no_pred_in_bin = torch.sum( mask , dim =(1,2) ).reshape(-1,1)
            # print(no_pred.shape, no_pred_in_bin.shape)
            no_pred = torch.cat((no_pred , no_pred_in_bin ), dim =1)
            
            no_acc_in_bin = torch.sum((mask) * (self.idealMap == target), dim = (1,2)).reshape(-1,1)
            no_acc = torch.cat((no_acc , no_acc_in_bin ), dim =1)

        self.no_pred_tot+=no_pred
        self.no_acc_tot+=no_acc
        self.conf_sum_tot+= conf_sum

        # print(torch.sum(self.no_pred_tot), target.shape[0]*target.shape[1])

        # self.perc = (no_acc)/(no_pred + 1e-13)
        
    def createIdealMap(self):
        base = (np.array([ i for i in range(self.n_clases)]))
        # self.idealMap=torch.Tensor(np.tile(base , (self.output.shape[2], self.output.shape[1], 1)).T).cuda()
        
        # Toy dataset
        # self.idealMap=torch.Tensor(np.tile(base , (400, 300, 1)).T).cuda()

        # Cityscapes
        # self.idealMap=torch.Tensor(np.tile(base , (2048, 1024, 1)).T).cuda()
        
        # cityscapes training
        self.idealMap=torch.Tensor(np.tile(base , (769, 769, 1)).T).cuda()
        
        # For zurich
        # self.idealMap=torch.Tensor(np.tile(base , (1920, 1080, 1)).T).cuda()

        # print(self.idealMap.shape, self.output.shape)
        # assert self.idealMap.shape == self.output.shape
        # creates a floor (like in a biulding) of class values

    def get_perc_table(self, classes):
        self.perc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        self.perc *= 100
        
        from tabulate import tabulate
        x= list(self.perc.cpu().numpy())

        for i in range(len(x)):
            x[i]=list(x[i])
            x[i]=[classes[i]]+list(x[i])
        print(tabulate(x, headers = ["Classes"]+[ "{:0.2f} - {:0.2f}".format(self.bin_lowers[i] * 100, self.bin_uppers[i] * 100) for i in range( len(self.bin_lowers))]))
        
        return self.perc

    def get_perc_table_img(self, classes):
        self.perc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        self.perc *= 100
        avg_conf *= 100


        # Plotting for table
        fig, ax = plt.subplots(figsize=(6,8))
        bin_str_label=[ "{} - {}".format(int(self.bin_lowers[i] * 100), int(self.bin_uppers[i] * 100)) for i in range( len(self.bin_lowers))]
        im, cbar = heatmap(self.perc.cpu().numpy(), classes, bin_str_label, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", size=7)

        fig.tight_layout()
        # plt.show()
        plt.savefig("temp_files/buffer_image_table.jpg")
        import cv2
        img_table = cv2.imread("temp_files/buffer_image_table.jpg")
        # print(img_table.shape)
        
        
        # Plotting for dif map
        fig, ax = plt.subplots(figsize=(6,8))
        bin_str_label=[ "{} - {}".format(int(self.bin_lowers[i] * 100), int(self.bin_uppers[i] * 100)) for i in range( len(self.bin_lowers))]
        im, cbar = heatmap(torch.abs(self.perc-avg_conf).cpu().numpy(), classes, bin_str_label, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", size=7)

        fig.tight_layout()
        # plt.show()
        plt.savefig("temp_files/buffer_image_dif.jpg")
        import cv2
        img_dif = cv2.imread("temp_files/buffer_image_dif.jpg")
        # print(img_dif.shape)
        return img_table, img_dif

    def get_count_table_img(self, classes):
        self.perc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        self.perc *= 100
        avg_conf *= 100


        # Plotting for table
        fig, ax = plt.subplots(figsize=(6,8))
        bin_str_label=[ "{} - {}".format(int(self.bin_lowers[i] * 100), int(self.bin_uppers[i] * 100)) for i in range( len(self.bin_lowers))]
        im, cbar = heatmap(100*(self.no_pred_tot/(torch.sum(self.no_pred_tot))).cpu().numpy(), classes, bin_str_label, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", size=7)

        fig.tight_layout()
        # plt.show()
        plt.savefig("temp_files/buffer_image_table.jpg")
        import cv2
        img_table = cv2.imread("temp_files/buffer_image_table.jpg")
        # print(img_table.shape)
        
        
        # Plotting for dif map not required in this case
        fig, ax = plt.subplots(figsize=(6,8))
        bin_str_label=[ "{} - {}".format(int(self.bin_lowers[i] * 100), int(self.bin_uppers[i] * 100)) for i in range( len(self.bin_lowers))]
        im, cbar = heatmap(torch.abs(self.perc-avg_conf).cpu().numpy(), classes, bin_str_label, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", size=7)

        fig.tight_layout()
        # plt.show()
        plt.savefig("temp_files/buffer_image_dif.jpg")
        import cv2
        img_dif = cv2.imread("temp_files/buffer_image_dif.jpg")
        # print(img_dif.shape)
        return img_table, img_dif



    def get_overall_CCELoss(self):
        avg_acc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)


        # overall_eceLoss = torch.sum(((avg_acc - avg_conf)**2))    
        
        # Correct implementation
        overall_eceLoss = torch.sum(((avg_acc - avg_conf)**2) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))    

        print("Overall CCE Loss (Metrics) = ", overall_eceLoss)

        return overall_eceLoss

        
    def get_classVise_CCELoss(self, classes):
        avg_acc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        # print(avg_acc.shape)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        # print(avg_conf.shape)

        x = torch.sum(torch.abs(avg_acc-avg_conf) * self.no_pred_tot, dim = 1) / torch.sum(self.no_pred_tot, dim = 1)
        x = x.reshape(-1,1)

        # print(x.shape)

        x=list(x)
        from tabulate import tabulate
        for i in range(len(x)):
            x[i]=list(x[i])
            x[i]=[classes[i]]+list(x[i])
        print(tabulate(x, headers = ["Classes", "ECELoss"]))

    


    def __init__(self, n_classes, n_bins = 10):
        '''
        output = [n_Class, h , w] np array: The complete probability vector of an image
        target = [h , w] np array: The GT for the image
        n_bins = [h , w] np array: Number of bins for the Calibration division

        '''
        self.n_clases = n_classes
        self.n_bins = n_bins

        self.no_pred_tot = torch.zeros(n_classes, n_bins).cuda()
        self.no_acc_tot = torch.zeros(n_classes, n_bins).cuda()
        self.conf_sum_tot = torch.zeros(n_classes, n_bins).cuda()

        self.createBins()
        
        self.createIdealMap()
        
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels,fontsize=8)
    ax.set_yticklabels(row_labels,fontsize=8)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



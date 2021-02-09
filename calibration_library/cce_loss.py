import torch
import numpy as np


# @neelabh17 implementation


class CCELoss(torch.nn.Module):
    def __init__(self, n_classes, n_bins = 10):
        '''
        output = [n_Class, h , w] np array: The complete probability vector of an image
        target = [h , w] np array: The GT for the image
        n_bins = [h , w] np array: Number of bins for the Calibration division

        '''
        super(CCELoss,self).__init__()
        self.n_classes = n_classes
        self.n_bins = n_bins


        self.createBins()
        self.createIdealMap()

        

    def forward(self , output, target):
        '''
        output = [batch, n_Class, h , w] np array: The complete logit vector of an image 

        target = [batch, h , w] np array: The GT for the image

        create an three array of [n_class, n_bins]
        -> Number of prediciton array for that specification
        -> Number of correct prediction for that class
        -> Percentge of correct 
        '''

        output = torch.softmax(output, dim=1)
        self.no_pred_tot = torch.zeros(self.n_classes, self.n_bins).cuda()
        self.no_acc_tot = torch.zeros(self.n_classes, self.n_bins).cuda()
        self.conf_sum_tot = torch.zeros(self.n_classes, self.n_bins).cuda()

        # print(self.idealMap.shape, target.shape)
        assert self.idealMap.shape[1:] == target.shape[1:]
        #  Making batch x numclass x 0 vector for appendage further
        no_pred = torch.Tensor(np.array([]).reshape(output.shape[0], output.shape[1],-1)).cuda()
        # print(no_pred.shape)
        # print(output.shape)
        no_acc = torch.Tensor(np.array([]).reshape(output.shape[0], output.shape[1],-1)).cuda()
        conf_sum = torch.Tensor(np.array([]).reshape(output.shape[0], output.shape[1],-1)).cuda()
        # print(output.shape)

        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):

            mask = (output> bin_lower) * (output <= bin_upper)
            
            output_clone = output.clone()
            output_clone[ ~ mask] = 0
            # import pdb; pdb.set_trace()
            conf_sum_in_bin = torch.sum(output_clone, dim = (2,3))
            # shape = [batch, classes]
            # print(conf_sum_in_bin.shape)
            conf_sum_in_bin = conf_sum_in_bin.unsqueeze(2)
            # shape = [batch, classes, 1]

            # print(conf_sum.shape, conf_sum_in_bin.shape)
            conf_sum = torch.cat((conf_sum , conf_sum_in_bin ), dim =2)

            
            no_pred_in_bin = torch.sum( mask , dim =(2,3) ).unsqueeze(2)
            # shape = [batch, classes, 1]
            # print(no_pred.shape, no_pred_in_bin.shape)
            no_pred = torch.cat((no_pred , no_pred_in_bin ), dim =2)
            
            no_acc_in_bin = torch.sum((mask) * (self.idealMap == target.unsqueeze(1)), dim = (2,3)).unsqueeze(2)
            # shape = [batch, classes, 1]
            no_acc = torch.cat((no_acc , no_acc_in_bin ), dim =2)

        self.no_pred_tot = no_pred
        self.no_acc_tot = no_acc
        self.conf_sum_tot =  conf_sum

        # reducing to one dimension, summing across all batches
        self.no_pred_tot = torch.sum(self.no_pred_tot, dim = 0)
        self.no_acc_tot = torch.sum(self.no_acc_tot, dim = 0)
        self.conf_sum_tot = torch.sum(self.conf_sum_tot, dim = 0)

        avg_acc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        # overall_cceLoss = torch.sum(torch.abs(avg_acc - avg_conf) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))
        # overall_cceLoss = torch.sum(((avg_acc - avg_conf)**2))

        # Correct implementation
        overall_cceLoss = torch.sum(((avg_acc - avg_conf)**2) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))

        return overall_cceLoss

    def createBins(self):

        #uniform bin spacing
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.avg_bin = torch.Tensor((self.bin_lowers + self.bin_uppers)/2).cuda()
        
    def createIdealMap(self):
        '''
        creates a floor (like in a biulding) of class values
        
        '''


        base = (np.array([ i for i in range(self.n_classes)]))
        # self.idealMap=torch.Tensor(np.tile(base , (self.output.shape[2], self.output.shape[1], 1)).T).cuda()
        
        # Toy dataset
        # self.idealMap=torch.Tensor(np.tile(base , (400,300, 1)).T).cuda()

        # Cityscapes
        self.idealMap=torch.Tensor(np.tile(base , (2048, 1024, 1)).T).cuda()
        
        # For zurich
        # self.idealMap=torch.Tensor(np.tile(base , (1920, 1080, 1)).T).cuda()

        # print(self.idealMap.shape, self.output.shape)
        # assert self.idealMap.shape == self.output.shape

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

    def get_overall_CCELoss(self):
        avg_acc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        overall_eceLoss = torch.sum(torch.abs(avg_acc - avg_conf) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))    

        print("Overall ECE Loss = ", overall_eceLoss)

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
    


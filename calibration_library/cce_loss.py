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

        print(self.idealMap.shape, target.shape)
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
        self.idealMap=torch.Tensor(np.tile(base , (400,300, 1)).T).cuda()

        # Cityscapes
        # self.idealMap=torch.Tensor(np.tile(base , (2048, 1024, 1)).T).cuda()
        
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

    


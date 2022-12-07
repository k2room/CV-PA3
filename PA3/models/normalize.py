import torch
import torch.nn as nn
import torch.nn.functional as F

class RAIN(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        '''Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        '''
        super(RAIN, self).__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps

    def forward(self, x, mask):

        # fill the blank
        mask = F.interpolate(mask.detach(), size=x.size()[2:])
        mf, stdf = self.get_foreground_mean_std(x*mask, mask) # mean and std for foreground
        mb, stdb = self.get_foreground_mean_std(x*(1-mask), 1-mask) # mean and std for background

        normf = ((stdb * ((x-mf) / stdf) + mb) * (self.foreground_gamma[None,:,None,None]+1) + self.foreground_beta[None,:,None,None]) * mask # foreground
        normb = ((x-mb) / stdb * (self.background_gamma[None,:,None,None]+1) + self.background_beta[None,:,None,None]) * (1-mask) # background
        
        return normf + normb

    def get_foreground_mean_std(self, region, mask):

        # fill the blank
        m = torch.sum(region, dim=[2,3])/torch.sum(mask, dim=[2,3]) # mean
        m = m[:,:,None,None]
        std = torch.sum((region+(1-mask)*m-m)**2, dim=[2,3])/torch.sum(mask, dim=[2,3])
        std = variance[:,:,None,None]
        std = torch.sqrt(var+self.eps)

        return m, std
import numpy as np


class TimeAveragedDist(object):

    def __init__(self, explr_idx=[0], log_std=0., capacity=100):

        self.log_std   = log_std
        self.capacity  = capacity
        self.explr_idx = explr_idx 


    def log_prob(self, x, s, grad=False):
        """ Calculates the log probability of the time averaged distribution  
        """
        if grad:
            return -(x[self.explr_idx]-s)/np.exp(self.log_std)
        return -0.5 * self.log_std - 0.5 * np.sum((x[:, self.explr_idx]-s)/np.exp(self.log_std),1)

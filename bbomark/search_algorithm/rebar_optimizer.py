from logging import log
import numpy as np
import torch

from bbomark.core import AbstractOptimizer
from bbomark.configspace.space import Configurations
from bbomark.configspace.feature_space import FeatureSpace_discrete_all_oneHot

class SNG(AbstractOptimizer, FeatureSpace_discrete_all_oneHot):
    """discrete version implementation

    Args:
        AbstractOptimizer ([type]): [description]
        FeatureSpace_discrete_all_oneHot ([type]): [description]
    """    
    

    def __init__(self, config_spaces, lam=1, discrete_degree=10):
        """[summary]

        Args:
            config_spaces ([type]): [description]
            lam (int, optional): update required sample num. Defaults to 1.
            discrete_degree (int, optional): 把连续空间均分为离散点的粒度. Defaults to 10.
        """        

        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_discrete_all_oneHot.__init__(self, discrete_degree)

        self.lam = lam
        self.discrete_degree = discrete_degree

        self.feature_dim = self.dense_dimension
        self.theta_num = self.sparse_dimension
        # self.max_dim = (self.categories).max()
        self.logits = [np.zeros(cat_num) for cat_num in self.categories]
        
        self.evaluate_func = None

        self.buffer_x = []
        self.buffer_y = []
    
    def _get_z_given_b_theta(self, logits, b, v):
        """计算 P(z|b)

        Args:
            logits ([float]): [description]
            b ([int]): one-hot
            v ([float]): [0, 1]

        Returns:
            [type]: [description]
        """        
        theta = torch.softmax(logits)
        v_b_log = torch.log(torch.sum(v*b)) # log(v[b.argmax()])
        z_tilde = -torch.log(-v_b_log)*b - \
            torch.log(-v_b_log/theta - v_b_log)*(1-b)
        return z_tilde
    
    def _get_z_given_theta(self, logits, u):
        """sample categories var z (reparameterized trick)

        Args:
            logits ([float]): [description]
            u ([float]): [0, 1]

        Returns:
            [type]: [description]
        """        
        return logits - torch.log( -torch.log(u) )
    
    def estimate_gradient(self, f_b, b, z, z_tilde, temp):
        

    def update(self, x, y):
        """
        sample from p(z|\theta) use u
        sample from p(z|b, \theta) use v
        Args:
            x ([type]): [description]
            y ([type]): [description]
        """
        tmp = torch.exp(log_tmp)
        for i, cat_num in enumerate(self.categories):
            # 设有 N 个分类变量，第i个分类变量具有cat_num的类别数
            u = torch.rand(self.lam, cat_num)
            v = torch.rand(self.lam, cat_num)
            # theta = self.theta[i]
            logits = self.logits[i]
            z = self._get_z_given_theta(logits)
            ones = torch.sparse.torch.eye()
            b = ones.index_select(0, torch.argmax(z))
            f_b = self.evaluate_func(b)
            z_tilde = self._get_z_given_b_theta(logits, b, v)
            f_z_tilde = self.evaluate_func(torch.softmax(z_tilde))
            f_z = self.evaluate_func(torch.softmax(z))
            self.estimate_gradient(f_b, b, z, z_tilde, f_z_tilde, temp)

    def observe(self, features, y):
        self.buffer_x.extend(features)
        self.buffer_y.extend(y)
        if len(self.buffer_y) < self.lam:
            return
        self.update(np.asarray(self.buffer_x), np.asarray(self.buffer_y))
        # 清空
        self.buffer_x = []
        self.buffer_y = []

    def suggest(self, n_suggestions):
        assert n_suggestions == 1, "noly support one suggest now"
        features = [self.sampling() for _ in range(n_suggestions)]
        x_guess = [None] * n_suggestions
        for ii, xx in enumerate(features): # 因为feature并不是横排成一个向量，需要ravel()
            x_array = self.feature_to_array(xx.ravel(), self.sparse_dimension)
            dict_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
            # dict_unwarped = Configurations.array_to_dictUnwarped(self.space, np.argmax(xx,axis=-1) / (self.categories-1))
            x_guess[ii] = dict_unwarped
        return x_guess, features
    
    def sampling(self):
        return 

opt_class = SNG
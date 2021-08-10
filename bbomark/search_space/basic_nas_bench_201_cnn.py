import numpy as np
import torch.nn as nn

from bbomark.core import TestFunction
from bbomark.utils.cell_based import NASBench201CNN

class Model(TestFunction):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        super().__init__()
        self.testProblem = NASBench201CNN()
        
    def evaluate(self, params):
        self.testProblem(params)

import sys
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torchinfo
import torch

# Get all available accelerators such as CUDA, MPS, MTIA, or XPUd
def test():
    num_accelerators = torch.accelerator.device_count() 

    
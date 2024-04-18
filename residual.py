import numpy as np
import torch
import time

"""
This is just an example to illustrate the inner workings of the "residual network"
concept outlined in "Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning"
by Oguntula et al.
"""

# Define the function f, modeled by a neural network
class ResidualNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualNetwork, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
# if __name__ == "main":
#     input_dim = 5
#     model = ResidualNetwork(input)
import numpy as np
import torch

class Preprocessor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Preprocessor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 16)
        self.h0 = torch.nn.Linear(16, 16)
        self.out = torch.nn.Linear(16, output_dim)
    
    def forward(self, x):
        out0 = self.linear(x)
        out1 = self.h0(out0)
        return self.out(out1)

# if __name__ == "__main__":
#     # Define the dimensions
#     in_dim = 4  # dimension of vector x
#     out_dim = 16  # dimension of output vector

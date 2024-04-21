import numpy as np
import torch
import time

"""
This is just an example to illustrate the inner workings of the process 
outlined in "Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning"
by Oguntula et al.
"""


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 16)
        self.h0 = torch.nn.Linear(16, 16)
        self.out = torch.nn.Linear(16, output_dim)

    def forward(self, x):
        out0 = self.linear(x)
        out1 = self.h0(out0)
        return self.out(out1)

def test():
    dim_x = 8
    num_agents = 3
    dim_b = 8
    res_out_dim = 16
    out_dim = 8

    # residual (encoder/compressor)
    res_net = MLP(dim_x, res_out_dim)
    # belief
    belief_net = MLP(dim_x, num_agents * dim_b)
    # actor - processes residual and beliefs and outputs actions
    actor_net = MLP(num_agents * dim_b + res_out_dim, out_dim)

    # sample input vector x
    x = torch.randn(dim_x)
    # obtain residual
    z = res_net(x)
    # 1st order belief prediction
    beliefs = belief_net(x)
    
    belief_matrix = torch.zeros((num_agents, dim_b))
    for i in range(num_agents):
        belief_matrix[i] = beliefs[i * dim_b : (i + 1) * dim_b]

    print(f"belief_matrix: \n {belief_matrix}")


def belief_test():
    # Define the dimensions
    dim_x = 5  # dimension of vector x
    dim_b = 10  # dimension of belief vector
    K = 12  # total number of agents

    # Initialize the neural network
    model = MLP(dim_x, K * dim_b)

    # Sample input vector x
    x = torch.randn(dim_x)
    print(f"x: \n {x}")

    # Compute the belief matrix B
    b = torch.randn(dim_b)  # Assuming b is the agent's own belief vector
    print(f"b: \n {x}")

    belief_matrix = torch.zeros((K, dim_b))
    print(f"belief_matrix: \n {belief_matrix}")

    output = model(x)
    print(f"output: \n {output}")

    time.sleep(1)
    for i in range(K):
        belief_matrix[i] = b + output[i * dim_b : (i + 1) * dim_b]
        print(f"belief_matrix: \n {belief_matrix}")

    # Print the belief matrix B
    print(belief_matrix)

    print(output.shape)
    assert output.shape == torch.Size([120])
    print(belief_matrix.shape)
    assert output.shape == torch.Size([12, 10])


if __name__ == "__main__":
    test()

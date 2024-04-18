import numpy as np
import torch
import time

"""
This is just an example to illustrate the inner workings of the "belief network"
concept outlined in "Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning"
by Oguntula et al.
"""

# Define the function f, modeled by a neural network
class BeliefNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BeliefNetwork, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    # Define the dimensions
    dim_x = 5  # dimension of vector x
    dim_b = 10  # dimension of belief vector
    K = 12  # total number of agents

    # Initialize the neural network
    model = BeliefNetwork(dim_x, K * dim_b)

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
    print(belief_matrix.shape)

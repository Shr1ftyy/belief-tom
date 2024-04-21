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


class AgentNet(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        belief_dim: int = 8,
        res_out_dim: int = 16,
        out_dim: int = 8,
        num_agents: int = 1,
    ):
        super(AgentNet, self).__init__()
        self.x_dim = input_dim
        self.b_dim = belief_dim
        self.res_out_dim = res_out_dim
        self.out_dim = out_dim
        self.num_agents = num_agents

        # residual (encoder/compressor)
        self.res_net = MLP(self.x_dim, res_out_dim)
        # belief
        self.belief_net = MLP(self.x_dim, self.num_agents * self.b_dim)
        # actor - processes residual and beliefs and outputs actions
        self.actor_net = MLP(self.num_agents * self.b_dim + res_out_dim, out_dim)

    def forward(self, x):
        # obtain residual
        z = self.res_net(x)
        # 1st order belief prediction
        b = self.belief_net(x)

        B = torch.zeros((self.num_agents, self.b_dim))
        for i in range(self.num_agents):
            B[i] = b[i * self.b_dim : (i + 1) * self.b_dim]

        # print(f"belief_matrix: \n {B}")
        flattened = B.flatten()
        final = torch.cat((z, flattened))  # concatenate residual and beliefs
        # print(f"flattened belief_matrix: \n {flattened}")
        # print(f"flattened shape: \n {flattened.shape}")

        return self.actor_net(final)


if __name__ == "__main__":
    device = "mps"  # TODO: set to desired device
    torch.set_default_device(device=device)
    agent = AgentNet()
    agent.to(device=device)
    print(agent)
    # sample input vector x
    x = torch.randn(agent.x_dim)
    # predict
    pred = agent(x)
    print(pred)

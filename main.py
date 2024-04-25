import numpy as np
import torch
import cv2
import argparse
from loguru import logger
from omegaconf import OmegaConf

from pettingzoo.mpe import simple_adversary_v3

import time
import sys
import os

"""
This is an example to illustrate the inner workings of the process 
outlined in "Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning"
by Oguntula et al.
"""

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(MLP, self).__init__()
        self.device = device
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, output_dim)
        )

    def forward(self, x):
        x = self.flatten(x).to(device=self.device)
        out = self.linear_relu_stack(x)
        return out


class AgentNet(torch.nn.Module):
    def __init__(
        self,
        device: str,
        input_dim: int = 8,
        belief_dim: int = 8,
        res_out_dim: int = 16,
        out_dim: int = 5,
        num_agents: int = 1,
    ):
        super(AgentNet, self).__init__()
        self.device = device
        self.x_dim = input_dim
        self.b_dim = belief_dim
        self.res_out_dim = res_out_dim
        self.out_dim = out_dim
        self.num_agents = num_agents

        # residual (encoder/compressor)
        self.res_net = MLP(self.x_dim, res_out_dim, self.device)
        # belief
        self.belief_net = MLP(self.x_dim, self.num_agents * self.b_dim, self.device)
        # actor - processes residual and beliefs and outputs actions
        self.actor_net = MLP(self.num_agents * self.b_dim + res_out_dim, out_dim, self.device)

    def forward(self, x):
        # obtain residual
        z = self.res_net(x)
        # 1st order belief prediction
        b = self.belief_net(x)

        # belief matrix isn't really need for computations here
        # B = torch.zeros((self.num_agents, self.b_dim))
        # for i in range(self.num_agents):
        #     B[i] = b[i * self.b_dim : (i + 1) * self.b_dim]

        # print(f"belief_matrix: \n {B}")
        final = torch.cat((z, b), dim=1)  # concatenate residual and beliefs
        # print(f"flattened belief_matrix: \n {flattened}")
        # print(f"flattened shape: \n {flattened.shape}")

        return self.actor_net(final)


def get_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["info", "debug", "trace"],
        required=True,
        default="info",
    )
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    # check arguments
    if os.path.isfile(args.config):
        pass
    elif os.path.isdir(args.config):
        print(logger.error(f"{args.config} is a directory. Please enter a file"))
        logger.error("closing...")
        exit()
    else:
        logger.error(f"{args.config} does not exist.")
        logger.error("closing...")
        exit()

    return args


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="run demo")
    args = get_args(parser)

    # set up debugging
    logger.remove()
    logger.disable("DEBUG")
    logger.add(
        sys.stderr, level=args.logging_level.upper()
    )  # Add a new logger with appropriate level

    # load config and initialize environment
    config = OmegaConf.load(args.config)
    # Simple Adversary (Physical Deception)
    env = simple_adversary_v3.env(
        N=config.env.num_agents_landmarks,
        max_cycles=config.env.max_cycles,
        render_mode="rgb_array",
    )
    env.reset()

    # see https://pettingzoo.farama.org/environments/mpe/simple_adversary for more info on this
    # TODO: cleanup?
    good_input_dim = 2 + 2 + (2 + 2) * (config.env.num_agents_landmarks) - 2
    adversary_input_dim = (2 + 2) * (config.env.num_agents_landmarks) 
    device = config.model.device  # TODO: set to desired device
    logger.debug(f"setting default device as: {device}")
    torch.set_default_device(device=device)

    models = {}
    for a in env.agents:
        name = str(a)
        # TODO: "stricter" checking?
        input_dim = adversary_input_dim if "adversary" in name else good_input_dim
        # TODO: which agents are we predicting beliefs for when performing 2nd order belief prediction? -> this will affect the num_agents parameter
        # for now, we assume that is for all agents for both good agents and the adversary.
        model = AgentNet(device=device, input_dim=input_dim, belief_dim=input_dim, res_out_dim=config.model.res_out_dim, out_dim=config.model.out_dim, num_agents=env.num_agents)
        models[name] = model
        models[name].to(device=device)
        logger.debug(f"----==== {name} ====----")
        logger.debug("----==== Architecture ====----")
        logger.debug(model)


    prev_obs = {}
    for agent in env.agent_iter():
        key = cv2.waitKey(config.env.keyDelay) & 0xFF  # Wait for a key press for 100 milliseconds
        observation, reward, termination, truncation, info = env.last()
        assert(observation.shape[0] == models[str(agent)].x_dim)

        if termination or truncation:
            action = None
        else:
            # TODO: use policy
            action_space = env.action_space(agent)
            policy = models[str(agent)]
            obs = torch.Tensor(observation).view(1, -1)
            logger.trace(f"obs: {obs}")
            outs = policy(obs)
            action = torch.argmax(outs).item()

        env.step(action)
        rdr = env.render()

        screen = cv2.cvtColor(rdr, cv2.COLOR_BGR2RGB)
        cv2.imshow("i believe", screen)
        # Check if the 'q' key is pressed
        if key == ord("q"):
            break  # Break the loop if 'q' is pressed
        elif key == ord("s"):
           cv2.imwrite("preview.png", screen)

        prev_obs[str(agent)] = observation

    cv2.destroyAllWindows()
    env.close()

'''
Just here for personal reference - Syeam
Observation arrays:
N=2
    10, agent_0: [0.55255926, -1.0281525, 0.55255926, -1.0281525, -0.78694487, -0.7872531, 0.5758478, -1.525509, -0.5624235, -0.21160883]
    8, adversary_0: [-0.02328854, 0.49735662, -1.3627926, 0.7382559, -0.5758478, 1.525509, -1.1382713, 1.3139002]
N=3
    14, agent_0: [0.78709215, 1.935311, 0.67603475, 1.3587927, 0.78709215, 1.935311, -0.20407286, 1.4376235, 1.5138645, 1.6963395, 1.5660207, -0.22972785, 0.20548968, 0.98830765]
    12, adversary_0: [-0.83782977, -0.33754683, -0.72677237, 0.23897147, -1.7179374, -0.25871605, 0.052156176, -1.9260674, -1.5138645, -1.6963395, -1.3083749, -0.70803183]
N=4
    18, agent_0: [2.0770037, -1.6093378, 1.453365, -1.086062, 2.0770037, -1.6093378, 1.4042718, -0.7966016, 1.1837791, -1.1081372, 0.41359973, -2.4731004, 1.4526888, -0.73592144, 1.9404204, -0.6026803, 1.3408152, -2.4923494]
    16, adversary_0: [1.0397652, 1.3870385, 1.6634039, 0.8637627, 0.9906721, 1.6764989, 0.7701794, 1.3649632, -0.41359973, 2.4731004, 1.0390891, 1.737179, 1.5268207, 1.8704201, 0.92721546, -0.019249031]
'''

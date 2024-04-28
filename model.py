from typing import Any, Dict, List
from gym.spaces import Discrete


from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

import marl

torch, nn = try_import_torch()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256, 256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out


class ToMModel(TorchModelV2, nn.Module):
    """The policy architecture"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super(ToMModel, self).__init__(
            obs_space, action_space, None, model_config, name
        )

        in_size = int(np.product(obs_space.shape))

        # residual network
        self.res_net = SlimFC(
            in_size=in_size,
            out_size=num_outputs,
            activation_fn=model_config["activation"],
        )

        # belief predictor
        self.belief_net = SlimFC(
            in_size=in_size,
            out_size=model_config["num_agents"] * model_config["belief_dim"],
            activation_fn=model_config["activation"],
        )

        # actor
        self.actor_net = SlimFC(
            in_size=model_config["num_agents"] * model_config["belief_dim"]
            + model_config["res_out_dim"],
            out_size=num_outputs,
            activation_fn=model_config["activation"],
        )

    def forward(
        self, input_dict: Dict[str, Any], state: List[Any], seq_lens: Any
    ) -> (TensorType, List[TensorType]):  # type: ignore
        # TODO: fix the ignoring above
        # obtain residual
        x = input_dict["obs"].flatten()
        z = self.res_net(x)
        # belief prediction
        b = self.belief_net(x)
        # concatenate residual and beliefs
        final = torch.cat((z, b), dim=0)  

        return self.actor_net(final), []


if __name__ == "__main__":
    import ray
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus, num_workers=args.num_workers)
    model = ToMModel(
        obs_space=torch.zeros(5, device="cuda:0"),
        action_space=torch.zeros(5, device="cuda:0"),
        name="ToMModel",
        num_outputs=5,
        model_config={"res_out_dim": 12, "num_agents": 3, "belief_dim": 3, "activation": "relu"},
    )
    model.to(device="cuda:0")

    print(model)
    obs = torch.rand(5, device="cuda:0")
    input_dict = SampleBatch()
    input_dict["obs"] = obs
    out, _ = model(input_dict=input_dict)
    print(f"out: {out}")
    print(model.last_output())

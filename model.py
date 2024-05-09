from typing import Any, Dict, List
from gym.spaces import Discrete


from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from marllib.marl.models.zoo.mlp.base_mlp import BaseMLP
from marllib.marl.models.zoo.mlp.cc_mlp import CentralizedCriticMLP
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

import marl
import numpy as np

torch, nn = try_import_torch()

class ToMModel(CentralizedCriticMLP):
    """The policy architecture"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(ToMModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        in_size = int(np.product(obs_space.shape))

        # residual network
        # self.res_net = SlimFC(
        #     in_size=in_size,
        #     out_size=self.custom_model_config["res_out_dim"],
        #     activation_fn=self.custom_model_config["activation"],
        # )
        # residual network is self.encoder created by `BaseMLP`

        # variational network - the network q_theta(z|b) that approximates the conditional
        # distribution p_sigma(z|b) (see paper for more technical info on this)
        # But it's basically used as a way to ensure that the belief and residual vectors
        # do NOT convey the same information
        self.variational_net = SlimFC(
            in_size=self.custom_config["num_agents"] * self.custom_config["belief_dim"],
            out_size=self.custom_config["res_out_dim"],
            # TODO: should we set initializer=norm_c_initializer(SMTHG_ELSE)? what are it's benefits & drawbacks?
            initializer=normc_initializer(0.01),
            activation_fn=self.custom_config["activation"],
        )

        # belief predictor
        self.belief_net = SlimFC(
            in_size=in_size,
            out_size=self.custom_config["num_agents"]
            * self.custom_config["belief_dim"],
            # TODO: should we set initializer=norm_c_initializer(SMTHG_ELSE)? what are it's benefits & drawbacks?
            initializer=normc_initializer(0.01),
            activation_fn=self.custom_config["activation"],
        )

        # actor
        self.actor_net = SlimFC(
            in_size=self.custom_config["num_agents"]
            * self.custom_config["belief_dim"]
            + self.custom_config["res_out_dim"],
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=self.custom_config["activation"],
        )

    def forward(
        self, input_dict: Dict[str, Any], state: List[Any], seq_lens: Any
    ) -> (TensorType, List[TensorType]):  # type: ignore
        # obtain residual
        x = input_dict["obs"]["obs"].float()
        self.inputs = x
        self.residual = self.p_encoder(x)
        # belief prediction
        self.beliefs = self.belief_net(x)
        # concatenate residual and beliefs
        final = torch.cat((self.residual, self.beliefs), dim=1)

        # TODO: should this be the final output or the residual?
        self._features = self.residual

        return self.actor_net(final), []

    def get_beliefs(self):
        assert self.beliefs is not None, "must call forward() first"
        return self.beliefs

    def get_residual(self):
        assert self.residual is not None, "must call forward() first"
        return self.residual


# if __name__ == "__main__":
#     import ray
#     import argparse
#     import numpy as np

#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     # ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus, num_workers=args.num_workers)
#     model = ToMModel(
#         obs_space=torch.zeros(5, device="cuda:0"),
#         action_space=torch.zeros(5, device="cuda:0"),
#         name="ToMModel",
#         num_outputs=5,
#         model_config={
#             # TODO: apparantly we shouldn't be doing it like this according to the log messages?
#             "custom_model_config": {
#                 "res_out_dim": 12,
#                 "num_agents": 3,
#                 "belief_dim": 3,
#                 "activation": "relu",
#             }
#         },
#     )
#     model.to(device="cuda:0")

#     print(model)
#     obs = torch.rand(5, device="cuda:0")
#     input_dict = SampleBatch()
#     input_dict["obs"]["obs"] = obs
#     out, _ = model(input_dict=input_dict)
#     print(f"out: {out}")
#     print(model.last_output())

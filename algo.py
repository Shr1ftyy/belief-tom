"""
Add new algorithm to marllib - with custom policy architecture
"""

# import argparse
# import os
import threading
import time
from typing import Dict, List, Union

from ray.rllib.utils import force_list, NullContextManager
from ray import tune
from ray.tune import CLIReporter
from ray.tune.utils import merge_dicts

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog

from ray.rllib.utils.typing import ModelGradients
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from marllib.marl.algos.utils.centralized_critic import (
    centralized_critic_postprocessing,
)

from ray.rllib.policy import Policy

from marllib.marl.algos.core.CC.mappo import (
    MAPPOTorchPolicy,
    MAPPOTrainer,
    central_critic_ppo_loss,
)

# from marllib.marl.algos.utils.centralized_critic import (
#     CentralizedValueMixin,
#     centralized_critic_postprocessing,
# )
from marllib.marl.algos.utils.setup_utils import AlgVar

from marllib import marl
import json
from model import ToMModel
import numpy as np

torch, nn = try_import_torch()

# TODO: is there a cleaner way to do this?
TOMAPPO_CONFIG = PPO_CONFIG
TOMAPPO_CONFIG["model"]["fcnet_activation"] = "relu"


# Variational loss L_q
def L_q(b, z, variational_net):
    q_z_given_b = variational_net(b)
    log_q_z_given_b = -nn.functional.mse_loss(q_z_given_b, z, reduction="none")
    return -log_q_z_given_b.mean()


# Residual loss L_residual
def L_residual(b, z, variational_net):
    joint_log_prob = torch.log(variational_net(b) + 1e-8).mean()
    marginal_b_log_prob = torch.log(
        variational_net(b).mean(0, keepdim=True) + 1e-8
    ).mean()
    marginal_z_log_prob = torch.log(z.mean(0, keepdim=True) + 1e-8).mean()

    loss_residual = joint_log_prob - (marginal_b_log_prob + marginal_z_log_prob)
    return loss_residual


def tom_policy_loss(policy, model, dist_class, train_batch):
    ppo_loss = central_critic_ppo_loss(policy, model, dist_class, train_batch)
    # try:
    #     # TODO: why is this empty even though the preprocess function returns actual values
    #     # also how the heck does the learn loaded on batch function or whatever work?
    #     if train_batch["other_agents_rewards"].shape[0] > 0:
    #         print("")
    # except: # noqa
    #     pass
    # TODO (IMMEDIATE): figure out how to do grad update on belief network with belief loss
    # belief_loss = F.mse_loss(model.get_beliefs(), train_batch["rewards"])  # TODO: belief loss
    # interesting, never knew torch.tensor (small t) was a thing (???) lol
    belief_loss = torch.tensor(0.0, requires_grad=True)
    # TODO: are the residual loss and variational loss define correctly?
    # NOTE: below I assume that beliefs, residuals, etc, have already been generated after ppo_loss()
    # initiates a forward pass through the networks, otherwise we should do some forward passes again
    # for each network we're obtaining losses for
    residual_loss = L_residual(
        model.get_beliefs(), model.get_residual(), model.variational_net
    )
    variational_loss = L_q(
        model.get_beliefs(), model.get_residual(), model.variational_net
    )
    # residual_loss = 0
    total_loss = (
        policy.config["model"]["custom_model_config"]["alpha"] * ppo_loss
        + policy.config["model"]["custom_model_config"]["beta"] * belief_loss
        + policy.config["model"]["custom_model_config"]["gamma"] * residual_loss
    )

    # TODO: should we be taking a mean?
    num_agents = model.custom_config["num_agents"]
    mean_residual_loss = residual_loss / num_agents
    mean_belief_loss = belief_loss / num_agents
    mean_variational_loss = variational_loss / num_agents

    model.tower_stats["mean_residual_loss"] = mean_residual_loss
    model.tower_stats["mean_belief_loss"] = mean_belief_loss
    model.tower_stats["mean_variational_loss"] = mean_variational_loss

    # TODO: what losses should be returned here? All three, or just one? Should we use custom_loss()???
    return total_loss, belief_loss, variational_loss
    # return total_loss


def tommappo_stats(
    policy: Policy, train_batch: SampleBatch
) -> Dict[str, torch.TensorType]:  # type: ignore
    """Stats function for our custom policy. Returns a dict with important KL and loss stats.

    Args:
        policy (Policy): The Policy to generate stats for.
        train_batch (SampleBatch): The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    return {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": torch.mean(torch.stack(policy.get_tower_stats("total_loss"))),
        "policy_loss": torch.mean(
            torch.stack(policy.get_tower_stats("mean_policy_loss"))
        ),
        "vf_loss": torch.mean(torch.stack(policy.get_tower_stats("mean_vf_loss"))),
        "vf_explained_var": torch.mean(
            torch.stack(policy.get_tower_stats("vf_explained_var"))
        ),
        "kl": torch.mean(torch.stack(policy.get_tower_stats("mean_kl_loss"))),
        "entropy": torch.mean(torch.stack(policy.get_tower_stats("mean_entropy"))),
        "entropy_coeff": policy.entropy_coeff,
        "residual_loss": torch.mean(
            torch.stack(policy.get_tower_stats("mean_residual_loss"))
        ),
        "belief_loss": torch.mean(
            torch.stack(policy.get_tower_stats("mean_belief_loss"))
        ),
        "variational_loss": torch.mean(
            torch.stack(policy.get_tower_stats("mean_variational_loss"))
        ),
    }


def tomappo_preprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    train_batch = centralized_critic_postprocessing(
        policy=policy,
        sample_batch=sample_batch,
        other_agent_batches=other_agent_batches,
        episode=episode,
    )

    custom_config = policy.config["model"]["custom_model_config"]
    opp_action_in_cc = custom_config["opp_action_in_cc"]
    global_state_flag = custom_config["global_state_flag"]

    opponent_batch = None

    if (not opp_action_in_cc and global_state_flag) or other_agent_batches is None:
        pass
    else:
        # need other agent batches
        n_agents = custom_config["num_agents"]

        opponent_agents_num = n_agents - 1
        opponent_batch_list = list(other_agent_batches.values())
        raw_opponent_batch = [
            opponent_batch_list[i][1] for i in range(opponent_agents_num)
        ]
        opponent_batch = []
        for one_opponent_batch in raw_opponent_batch:
            if len(one_opponent_batch) == len(sample_batch):
                pass
            else:
                if len(one_opponent_batch) > len(sample_batch):
                    one_opponent_batch = one_opponent_batch.slice(0, len(sample_batch))
                else:  # len(one_opponent_batch) < len(sample_batch):
                    length_dif = len(sample_batch) - len(one_opponent_batch)
                    one_opponent_batch = one_opponent_batch.concat(
                        one_opponent_batch.slice(
                            len(one_opponent_batch) - length_dif,
                            len(one_opponent_batch),
                        )
                    )
            opponent_batch.append(one_opponent_batch)

    # train_batch["kj"]
    if opponent_batch is not None:
        train_batch["other_agent_rewards"] = np.array([b["rewards"] for b in opponent_batch])
    else:
        train_batch["other_agent_rewards"] = None

    return train_batch


def run_tommappo(model_class, config_dict, common_config, env_dict, stop, restore):
    ModelCatalog.register_custom_model(model_name="ToMModel", model_class=model_class)

    _param = AlgVar(config_dict)

    train_batch_size = _param["batch_episode"] * env_dict["episode_limit"]

    if "fixed_batch_timesteps" in config_dict:
        train_batch_size = config_dict["fixed_batch_timesteps"]
    episode_limit = env_dict["episode_limit"]

    batch_mode = _param["batch_mode"]
    lr = _param["lr"]

    config = {
        "train_batch_size": train_batch_size,
        "batch_mode": batch_mode,
        "lr": lr if restore is None else 1e-10,
        "model": {
            "custom_model": "ToMModel",
            "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }

    config.update(common_config)

    algorithm = config_dict["algorithm"]
    map_name = config_dict["env_args"]["map_name"]
    # arch = config_dict["model_arch_args"]["core_arch"]
    arch = "tommodel"
    RUNNING_NAME = "_".join([algorithm, arch, map_name])

    if restore is not None:
        with open(restore["params_path"], "r") as JSON:
            raw_config = json.load(JSON)
            raw_config = raw_config["model"]["custom_model_config"]["model_arch_args"]
            check_config = config["model"]["custom_model_config"]["model_arch_args"]
            if check_config != raw_config:
                raise ValueError(
                    "is not using the params required by the checkpoint model"
                )
        model_path = restore["model_path"]
    else:
        model_path = None

    available_local_dir = "algo_results"

    config["framework"] = "torch"

    results = tune.run(
        ToMMAPPOTrainer,
        name=RUNNING_NAME,
        checkpoint_at_end=config_dict["checkpoint_end"],
        checkpoint_freq=config_dict["checkpoint_freq"],
        restore=model_path,
        stop=stop,
        config=config,
        verbose=1,
        progress_reporter=CLIReporter(),
        local_dir=(
            available_local_dir
            if config_dict["local_dir"] == ""
            else config_dict["local_dir"]
        ),
    )

    return results


def custom_multi_gpu_parallel_grad_calc(self, sample_batches):
    """A modified version of ray.rllib.policy.TorchPolicy._multi_gpu_parallel_grad_calc created to support
    to seperate the training of the belief and variational network from the rest of the policy
    """

    assert len(self.model_gpu_towers) == len(sample_batches)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(shard_idx, model, sample_batch, device):
        torch.set_grad_enabled(grad_enabled)
        try:
            with (
                NullContextManager()
                if device.type == "cpu"
                else torch.cuda.device(device)
            ):
                loss_out = force_list(
                    self._loss(self, model, self.dist_class, sample_batch)
                )

                # Call Model's custom-loss with Policy loss outputs and
                # train_batch.
                loss_out = model.custom_loss(loss_out, sample_batch)

                assert len(loss_out) == len(self._optimizers)

                # Loop through all optimizers.
                grad_info = {"allreduce_latency": 0.0}

                parameters = list(model.parameters())
                all_grads = [None for _ in range(len(parameters))]
                for opt_idx, opt in enumerate(self._optimizers):
                    # Erase gradients in all vars of the tower that this
                    # optimizer would affect.
                    param_indices = self.multi_gpu_param_groups[opt_idx]
                    for param_idx, param in enumerate(parameters):
                        if param_idx in param_indices and param.grad is not None:
                            param.grad.data.zero_()
                    # Recompute gradients of loss over all variables.
                    loss_out[opt_idx].backward(retain_graph=True)
                    grad_info.update(self.extra_grad_process(opt, loss_out[opt_idx]))

                    grads = []
                    # Note that return values are just references;
                    # Calling zero_grad would modify the values.
                    for param_idx, param in enumerate(parameters):
                        if param_idx in param_indices:
                            if param.grad is not None:
                                grads.append(param.grad)
                            all_grads[param_idx] = param.grad

                    if self.distributed_world_size:
                        start = time.time()
                        if torch.cuda.is_available():
                            # Sadly, allreduce_coalesced does not work with
                            # CUDA yet.
                            for g in grads:
                                torch.distributed.all_reduce(
                                    g, op=torch.distributed.ReduceOp.SUM
                                )
                        else:
                            torch.distributed.all_reduce_coalesced(
                                grads, op=torch.distributed.ReduceOp.SUM
                            )

                        for param_group in opt.param_groups:
                            for p in param_group["params"]:
                                if p.grad is not None:
                                    p.grad /= self.distributed_world_size

                        grad_info["allreduce_latency"] += time.time() - start

            with lock:
                results[shard_idx] = (all_grads, grad_info)
        except Exception as e:
            with lock:
                results[shard_idx] = (
                    ValueError(
                        e.args[0]
                        + "\n"
                        + "In tower {} on device {}".format(shard_idx, device)
                    ),
                    e,
                )

    # Single device (GPU) or fake-GPU case (serialize for better
    # debugging).
    if len(self.devices) == 1 or self.config["_fake_gpus"]:
        for shard_idx, (model, sample_batch, device) in enumerate(
            zip(self.model_gpu_towers, sample_batches, self.devices)
        ):
            _worker(shard_idx, model, sample_batch, device)
            # Raise errors right away for better debugging.
            last_result = results[len(results) - 1]
            if isinstance(last_result[0], ValueError):
                raise last_result[0] from last_result[1]
    # Multi device (GPU) case: Parallelize via threads.
    else:
        threads = [
            threading.Thread(
                target=_worker, args=(shard_idx, model, sample_batch, device)
            )
            for shard_idx, (model, sample_batch, device) in enumerate(
                zip(self.model_gpu_towers, sample_batches, self.devices)
            )
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # Gather all threads' outputs and return.
    outputs = []
    for shard_idx in range(len(sample_batches)):
        output = results[shard_idx]
        if isinstance(output[0], Exception):
            raise output[0] from output[1]
        outputs.append(results[shard_idx])
    return outputs


# # TODO: update this???
def optimizer(
    self, config
) -> Union[List["torch.optim.Optimizer"], "torch.optim.Optimizer"]:  # type: ignore
    """Customizes the pytorch optimizers to be used by params

    Returns:
        Union[List[torch.optim.Optimizer], torch.optim.Optimizer]:
            The local PyTorch optimizer(s) to use for this Policy.
    """
    # params = filter(lambda p: p.requires_grad, self.model.parameters())
    # TODO: is there a smarter way to go about this? actually, i think using the config var here somehow might do the trick!!!
    model_params = set(self.model.parameters())
    belief_params = set(self.model.belief_net.parameters())
    var_params = set(self.model.variational_net.parameters())
    optims = []
    # TODO: is it ok to just convert to list and be off with it?
    params = [
        list(
            model_params - belief_params - var_params
        ),  # model params (excluding variational and belief net params)
        list(belief_params),  # belief net params
        list(var_params),  # variational net params
    ]
    if config is not None:
        for p in params:
            optims.append(torch.optim.Adam(p, lr=config["lr"]))
    else:
        for p in params:
            optims.append(torch.optim.Adam(p, lr=config["lr"]))

    return optims


def compute_gradients(self, postprocessed_batch: SampleBatch) -> ModelGradients:

    assert len(self.devices) == 1

    # If not done yet, see whether we have to zero-pad this batch.
    if not postprocessed_batch.zero_padded:
        pad_batch_to_sequences_of_same_size(
            batch=postprocessed_batch,
            max_seq_len=self.max_seq_len,
            shuffle=False,
            batch_divisibility_req=self.batch_divisibility_req,
            view_requirements=self.view_requirements,
        )

    postprocessed_batch.is_training = True
    self._lazy_tensor_dict(postprocessed_batch, device=self.devices[0])

    # # Freeze the belief network - is this what I should do?
    # self.model.belief_net.requires_grad = False

    # Do the (maybe parallelized) gradient calculation step.
    # tower_outputs = self._multi_gpu_parallel_grad_calc([postprocessed_batch])
    tower_outputs = custom_multi_gpu_parallel_grad_calc(self, [postprocessed_batch])

    all_grads, grad_info = tower_outputs[0]

    grad_info["allreduce_latency"] /= len(self._optimizers)
    grad_info.update(self.extra_grad_info(postprocessed_batch))

    fetches = self.extra_compute_grad_fetches()

    return all_grads, dict(fetches, **{LEARNER_STATS_KEY: grad_info})


def get_policy_class_tomappo(config_):
    if config_["framework"] == "torch":
        return ToMMAPPOPolicy


# <class 'ray.rllib.policy.torch_policy_template.MyTorchPolicy'>
ToMMAPPOPolicy = MAPPOTorchPolicy.with_updates(
    name="ToMMAPPOPolicy",
    get_default_config=lambda: TOMAPPO_CONFIG,
    optimizer_fn=optimizer,
    stats_fn=tommappo_stats,
    postprocess_fn=tomappo_preprocessing,
    loss_fn=tom_policy_loss,
    compute_gradients_fn=compute_gradients,
)

# <class 'ray.rllib.agents.trainer_template.MyCustomTrainer'>
ToMMAPPOTrainer = MAPPOTrainer.with_updates(
    name="ToMMAPPOTrainer",
    default_policy=ToMMAPPOPolicy,
    get_policy_class=get_policy_class_tomappo,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    env = marl.make_env(environment_name="mpe", map_name="simple_adversary")
    marl.algos.register_algo(algo_name="tommappo", style="cc", script=run_tommappo)

    # parser.add_argument('--num_cpus', type=int)
    # parser.add_argument('--num_workers', type=int)
    # parser.add_argument('--num_gpus', type=int)
    # parser.add_argument('--stop_iters', type=int)
    # ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus)

    tommappo = marl.algos.tommappo(hyperparam_source="common")

    model = (
        ToMModel,
        {
            "res_out_dim": 8,
            "num_agents": 2,
            "belief_dim": 1,
            "activation": "relu",
            "model_arch_args": {"fc_layer": 2, "out_dim_fc_0": 128, "out_dim_fc_1": 8},
            "fcnet_activation": "relu",
            "alpha": 0.5,
            "beta": 0.25,
            "gamma": 0.25,
        },
    )
    print(model)
    tommappo.fit(
        env,
        model,
        num_gpus=0,
        num_workers=1,
        local_mode=True,
        stop={"timesteps_total": 10000000},
        checkpoint_freq=100,
        share_policy="group",
    )

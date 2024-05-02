"""
Add new algorithm to marllib - with custom policy architecture
"""

# import argparse
# import os

import ray
import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.utils import merge_dicts
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog

from marllib.marl.algos.core.CC.mappo import (
    MAPPOTorchPolicy,
    MAPPOTrainer,
    get_policy_class_mappo,
    central_critic_ppo_loss,
)
from marllib.marl.algos.utils.centralized_critic import (
    CentralizedValueMixin,
    centralized_critic_postprocessing,
)
from marllib.marl.algos.utils.setup_utils import AlgVar

from marllib import marl
import json
from model import ToMModel

# parser = argparse.ArgumentParser()
# parser.add_argument("--stop-iters", type=int, default=200)
# parser.add_argument("--num-cpus", type=int, default=0)

# def mutual_information_loss(policy, model, train_batch):
#     belief = model.belief_network(train_batch[SampleBatch.CUR_OBS].float())
#     residual = model.residual_network(train_batch[SampleBatch.CUR_OBS].float())

#     # Compute belief and residual distributions
#     belief_dist = policy.belief_distribution(belief)
#     residual_dist = policy.residual_distribution(residual)

#     # Compute the joint distribution P(B, Z)
#     joint_dist = belief_dist.unsqueeze(1) * residual_dist.unsqueeze(0)

#     # Compute the product of marginal distributions P(B) * P(Z)
#     marginal_belief = belief_dist.unsqueeze(1).sum(dim=0)
#     marginal_residual = residual_dist.unsqueeze(0).sum(dim=0)
#     product_marginals = marginal_belief.unsqueeze(1) * marginal_residual.unsqueeze(0)

#     # Compute KL-divergence between the joint distribution and product of marginals
#     kl_divergence = F.kl_div(joint_dist.log(), product_marginals, reduction='batchmean')

#     return kl_divergence

# def variational_loss(policy, model, train_batch):
#     belief = model.belief_network(train_batch[SampleBatch.CUR_OBS].float())
#     residual = model.residual_network(train_batch[SampleBatch.CUR_OBS].float())

#     # Compute log likelihood of residual given belief
#     log_likelihood_belief_residual = model.residual_network(belief, train_batch[SampleBatch.CUR_OBS].float()).log_prob(residual)
#     # Compute log likelihood of residual given sampled beliefs
#     log_likelihood_residual = model.residual_network(belief.unsqueeze(0), train_batch[SampleBatch.CUR_OBS].float()).log_prob(residual.unsqueeze(0)).mean(dim=0)

#     # Compute contrastive log-ratio upper bound loss
#     loss_q = -log_likelihood_belief_residual
#     loss_residual = log_likelihood_belief_residual - log_likelihood_residual

#     return loss_q, loss_residual


def tom_policy_loss(policy, model, dist_class, train_batch):
    ppo_loss = central_critic_ppo_loss(policy, model, dist_class, train_batch)
    belief_loss = 0  # TODO: belief loss
    # residual_loss, _ = TODO: residual loss ... variational_loss(policy, model, train_batch)
    residual_loss = 0
    total_loss = (
        policy.config["custom_model_config"]["alpha"] * ppo_loss
        + policy.config["custom_model_config"]["beta"] * belief_loss
        + policy.config["custom_model_config_"]["gamma"] * residual_loss
    )
    return total_loss


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


def get_policy_class_tomappo(config_):
    if config_["framework"] == "torch":
        return MAPPOTorchPolicy


# <class 'ray.rllib.policy.torch_policy_template.MyTorchPolicy'>
ToMMAPPOPolicy = MAPPOTorchPolicy.with_updates(
    name="ToMMAPPOPolicy",
    loss_fn=tom_policy_loss,
)

# <class 'ray.rllib.agents.trainer_template.MyCustomTrainer'>
ToMMAPPOTrainer = MAPPOTrainer.with_updates(
    name="ToMMAPPOTrainer",
    default_policy=ToMMAPPOPolicy,
    get_policy_class=get_policy_class_tomappo,
)


if __name__ == "__main__":
    import ray
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

    model = marl.build_model(env, tommappo, {"core_arch": "mlp"})
    # model = (ToMModel, {"res_out_dim": 8, "num_agents": 2, "belief_dim": 1, "activation": "relu", "model_arch_args": {"fc_layer": 1, "out_dim_fc_0": }})
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
            "gamma": 0.25
        },
    )
    print(model)
    tommappo.fit(
        env,
        model,
        num_gpus=0,
        stop={"timesteps_total": 10000000},
        checkpoint_freq=100,
        share_policy="group",
    )

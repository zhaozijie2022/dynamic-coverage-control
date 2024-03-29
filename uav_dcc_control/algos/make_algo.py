import importlib
import utils.pytorch_utils as ptu


def make_algo(cfg):
    if "maddpg" in cfg.algo_file:
        algo_file = importlib.import_module("algos." + cfg.algo_file)
        Agent = getattr(algo_file, cfg.algo_class)
        return [Agent(
            n_agents=cfg.n_agents,
            agent_id=i,

            obs_dim_n=cfg.obs_dim_n,
            action_dim_n=cfg.action_dim_n,
            context_dim=cfg.context_dim,

            actor_layers=cfg.actor_layers,
            critic_layers=cfg.critic_layers,

            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            gamma=cfg.gamma,
            tau=cfg.tau,
        ).to(ptu.device) for i in range(cfg.n_agents)]
    elif "mappo" in cfg.algo_file:
        from algos.mappo import MAPPOTrainer, MAPPOPolicy
        policy = []
        for agent_id in range(cfg.n_agents):
            share_observation_space = (
                cfg.share_observation_space[agent_id]
                if cfg.use_centralized_V
                else cfg.observation_space[agent_id]
            )
            # policy network
            po = MAPPOPolicy(
                cfg,
                cfg.observation_space[agent_id],
                share_observation_space,
                cfg.action_space[agent_id],
            )
            policy.append(po)
        return [MAPPOTrainer(
            cfg=cfg,
            policy=policy[i],
            agent_id=i,
        ) for i in range(cfg.n_agents)]










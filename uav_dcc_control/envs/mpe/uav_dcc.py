import envs.mpe.multiagent.scenarios as scenarios
import numpy as np
from envs.mpe.multiagent.environment import MultiAgentEnv
from gym.spaces import Box


class DCEnv:
    def __init__(
            self,
            scenario,
            num_agents=4,
            num_pois=20,
            max_ep_len=100,
            r_cover=0.2,
            r_comm=0.4,
            comm_r_scale=0.95,
            comm_force_scale=0.0,
            **kwargs):
        # assert scenario == "coverage"
        self.n_agents = num_agents
        scenario = scenarios.load(scenario + ".py").Scenario(
            num_agents,
            num_pois,
            r_cover,
            r_comm,
            comm_r_scale,
            comm_force_scale,
            **kwargs
        )
        # create world
        world = scenario.make_world()
        # create multiagent environment
        self.env = MultiAgentEnv(world=world,
                                 reset_callback=scenario.reset_world,
                                 reward_callback=scenario.reward,
                                 observation_callback=scenario.observation,
                                 done_callback=scenario.done)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        share_obs_dim = sum([self.env.observation_space[i].shape[0] for i in range(self.n_agents)])
        self.share_observation_space = [Box(low=np.array([-np.inf] * share_obs_dim, dtype=np.float32),
                                            high=np.array([np.inf] * share_obs_dim, dtype=np.float32),
                                            dtype=np.float32) for _ in range(self.n_agents)]
        self.max_ep_len = max_ep_len

    def step(self, actions):
        obs_n, reward_n, done_n, info_n = self.env.step(actions)
        info_n["coverage_rate"] = self.env.world.coverage_rate
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode=mode)

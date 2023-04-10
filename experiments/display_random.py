import numpy as np
import time
from maddpg.lib4cover import make_env
from maddpg.common.distributions import make_pdtype

scen_name = "coverage_3"
max_ep_len = 30

if __name__ == '__main__':
    env = make_env(scen_name, r_cover=0.2, r_comm=0.4, comm_r_scale=0.9, comm_force_scale=0.5)

    obs_n = env.reset()
    episode_step = 0
    while True:
        action_n = []
        for i in range(4):
            # tmp = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            tmp = np.random.uniform(0, 1, size=5)
            tmp = np.exp(tmp) / np.sum(np.exp(tmp))
            action_n.append(tmp)

        if not env.world.connect_:
            action_r = env.world.revise_action(action_n)
            new_obs_n, rew_n, done_n = env.step(action_r)
            obs_n = new_obs_n
            print("action_r: ", action_r)
        else:
            new_obs_n, rew_n, done_n = env.step(action_n)
            obs_n = new_obs_n
            print("action_n: ", action_n)
        # time.sleep(0.05)
        # env.render()

        episode_step += 1

        if all(done_n) or (episode_step > max_ep_len):
            obs_n = env.reset()
            episode_step = 0
            break




















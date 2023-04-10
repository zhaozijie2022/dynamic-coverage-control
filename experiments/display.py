import os
import time
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from maddpg.lib4cover import make_env, get_trainers, parse_args
import maddpg.common.tf_util as U
import imageio
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

scen_name = "coverage2"
load_path = "./coverage3/#done_policy/"

is_save_gif = True
save_gif_path = "./coverage3/#gif/coverage_3_doasdne.gif"

max_ep_len = 80  # episode最多有多少状态转换
num_eps = 5
good_p, adv_p = "maddpg", "maddpg"
lr, gma = 1e-2, 0.95


if __name__ == '__main__':
    arglist = parse_args(scen_name, max_ep_len, num_eps, lr, gma, num_units=128)
    with U.single_threaded_session():  # 单个CPU的session
        env = make_env(scen_name, r_cover=0.25, r_comm=0.5, comm_r_scale=0.9, comm_force_scale=50.0)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, obs_shape_n, arglist)
        U.initialize()

        print('Loading previous state...')
        U.load_state(load_path)

        obs_n = env.reset()
        episode_step = 0
        connect_step = 0
        frames = []

        while True:
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

            # if not env.world.connect_:
            #     action_r = env.world.revise_action(action_n)
            #     new_obs_n, rew_n, done_n = env.step(action_r)
            #     obs_n = new_obs_n
            # else:
            #     new_obs_n, rew_n, done_n = env.step(action_n)
            #     obs_n = new_obs_n
            if is_save_gif:
                frames.append(env.render(mode='rgb_array')[0])
            else:
                time.sleep(0.05)
                env.render()
            new_obs_n, rew_n, done_n = env.step(action_n)
            obs_n = new_obs_n
            episode_step += 1
            if env.world.connect:
                connect_step += 1
            else:
                frames = []
                obs_n = env.reset()
                episode_step = 0
                connect_step = 0



            if all(done_n) or (episode_step > arglist.max_episode_len):
                if is_save_gif:
                    if episode_step >= 40 or connect_step <= 0.99 * episode_step:
                        frames = []
                        obs_n = env.reset()
                        episode_step = 0
                        connect_step = 0
                        continue

                    env.close()
                    frames = [frames[0]] * 5 + frames
                    frames += [frames[-1]] * 5
                    imageio.mimsave(save_gif_path, frames, 'GIF', duration=0.1)
                    break
                else:
                    obs_n = env.reset()
                    episode_step = 0
                    connect_step = 0




















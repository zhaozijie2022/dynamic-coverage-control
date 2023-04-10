# 无连通保持下的无人机动态覆盖控制

import os
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 只打印warning信息
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # 只使用CPU, 使用GPU反而慢

import maddpg.common.tf_util as U
from maddpg.lib4cover import make_env, get_trainers, create_dir, parse_args

scenario_name = "coverage1"

# load_policy_path = "./train_data/" + os.listdir("./train_data")[-1] + "/policy/"
# load_buffer_path = "./train_data/" + os.listdir("./train_data")[-1] + "/buffers/"

load_policy_path = None
load_buffer_path = None

max_episode_len = 80  # episode最多有多少状态转换
num_eps = 20000  # 执行多少episode
lr, gamma = 10e-2, 0.95
batch_size = 1024

display_rate = 100
save_rate = 1000
plot_rate = 5000


if __name__ == '__main__':
    arglist = parse_args(scenario_name, max_episode_len, num_eps, lr, gamma, batch_size, num_units=128)
    save_policy_path, save_plots_path, save_buffer_path = create_dir(scenario_name)
    with U.single_threaded_session():
        env = make_env(scenario_name, r_cover=0.2, r_comm=0.4, comm_r_scale=-1, comm_force_scale=-1)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, obs_shape_n, arglist, buffer_path=load_buffer_path)
        U.initialize()

        if load_policy_path:
            U.load_state(load_policy_path)
            print("Load previous state, done!")

        episode_rewards = [0]  # 所有episode的rewards
        episode_rps = []  # reward per step
        episode_coverage = []  # 所有episode结束时的覆盖率
        done_steps = []  # 所有episode, 执行完任务所需的steps

        train_episodes = 0  # 现在经过了多少episodes
        episode_step = 0  # 当前episode, 经历了多少step
        train_step = 0  # 本次训练, 当前经过了多少step
        t_start = time.time()

        saver = tf.train.Saver()
        print('Starting iterations...')

        obs_n = env.reset()
        while True:
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

            new_obs_n, rew_n, done_n = env.step(action_n)
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])
            obs_n = new_obs_n

            # 记录数据
            episode_step += 1
            episode_rewards[-1] += sum(rew_n)

            # 记录数据
            terminal = (episode_step >= arglist.max_episode_len) or all(done_n)
            if terminal:
                episode_rps.append(episode_rewards[-1]/episode_step)
                episode_rewards.append(0)
                episode_coverage.append(env.world.coverage_rate)
                done_steps.append(episode_step)

                episode_step = 0
                train_episodes += 1
                obs_n = env.reset()

            train_step += 1

            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # 展示成果
            if terminal and (train_episodes % display_rate == 0):
                print("Eps: %d, Sps:%d, RpS: %.2f, C: %.3f, MSp: %.2f, T: %.2f"
                      % (train_episodes,
                         train_step,
                         float(np.mean(episode_rps[-display_rate:])),
                         float(np.mean(episode_coverage[-display_rate:])),
                         float(np.mean(done_steps[-display_rate:])),
                         time.time() - t_start,
                         )
                      )
                t_start = time.time()

            # 保存模型
            if terminal and (train_episodes % save_rate == 0):
                U.save_state(save_policy_path, saver=saver)
                print("model has been saved")

            # 画轨迹图
            if terminal and (train_episodes % plot_rate == 0):
                avg_cov, avg_rps = [], []
                plot_step = 500
                for i in range(0, train_episodes, plot_step):
                    avg_rps.append(np.mean(episode_rps[i:i+plot_step]))
                    avg_cov.append(np.mean(episode_coverage[i:i+plot_step]))

                plt.figure()
                plt.plot(avg_rps)
                plt.title("Rewards per Step")
                plt.show()

                plt.figure()
                plt.plot(avg_cov, label="Coverage", color="red")
                plt.title("Coverage & Connectivity")
                plt.ylim([0, 1])
                plt.show()

            # 保存rewards, coverage rate, done steps 和 buffers
            if train_episodes >= arglist.num_episodes:

                rps_file_name = save_plots_path + 'rewards.pkl'
                with open(rps_file_name, 'wb') as fp:
                    pickle.dump(episode_rps, fp)

                cover_file_name = save_plots_path + "coverage_rate.pkl"
                with open(cover_file_name, 'wb') as fp:
                    pickle.dump(episode_coverage, fp)

                steps_file_name = save_plots_path + "done_steps.pkl"
                with open(steps_file_name, 'wb') as fp:
                    pickle.dump(done_steps, fp)

                for trainer in trainers:
                    buffer_file_name = save_buffer_path + "buffer_" + trainer.name + ".pkl"
                    with open(buffer_file_name, "wb") as fp:
                        pickle.dump(trainer.replay_buffer, fp)
                print('...Finished total of %d episodes.' % len(episode_rewards))
                break

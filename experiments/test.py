import os
import numpy as np
import tensorflow as tf
from maddpg.lib4cover import make_env, get_trainers, parse_args
import maddpg.common.tf_util as U
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

scenario_name = "coverage1"
# load_path = "./coverage_2/coverage_2_02_13_10_07/policy/"
load_path = "./" + scenario_name + "/#done_policy/"

max_ep_len = 80  # episode最多有多少状态转换
num_eps = 50
lr, gma = 1e-2, 0.95


if __name__ == "__main__":
    arglist = parse_args(scenario_name, max_ep_len, num_eps, lr, gma)
    tf_config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        # log_device_placement=True,
        # allow_soft_placement=True
    )
    with tf.compat.v1.Session(config=tf_config):  # 单个CPU的session
        # 创建环境
        env = make_env(scenario_name, r_cover=0.25, r_comm=0.5, comm_r_scale=0.9, comm_force_scale=0.0)
        # 创建trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, obs_shape_n, arglist)
        U.initialize()

        U.load_state(load_path)
        print("Load previous state, done!")

        obs_n = env.reset()
        episode_step = 0
        num_ep = 0

        step_rewards = []  # 单次episode, 每个step获得的奖励
        episode_rewards = []  # 所有episode的rewards
        done_steps = []  # 所有episode, 执行完任务所需的steps
        episode_coverage = []  # 所有episode结束时的覆盖率
        episode_connectivity = [0]  # 所有episode结束时的连接率
        episode_collide = []  # 所有episode结束时的碰撞次数

        while True:
            episode_connectivity[-1] += env.world.connect
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

            if not env.world.connect_:
                action_r = env.world.revise_action(action_n)
                new_obs_n, rew_n, done_n = env.step(action_r)
                obs_n = new_obs_n
            else:
                new_obs_n, rew_n, done_n = env.step(action_n)
                obs_n = new_obs_n

            # new_obs_n, rew_n, done_n = env.step(action_n)
            # obs_n = new_obs_n
            episode_step += 1
            terminal = (episode_step >= arglist.max_episode_len) or all(done_n)

            step_rewards.append(rew_n[0])

            if terminal:
                num_ep += 1
                episode_connectivity[-1] /= episode_step
                print("Eps %d, RpS: %.2f, Cov: %.2f, Conn: %.2f, Sps: %d" %
                      (num_ep, float(np.mean(step_rewards)), env.world.coverage_rate, episode_connectivity[-1], episode_step))
                episode_rewards.append(sum(step_rewards)/episode_step)
                step_rewards = []
                episode_coverage.append(env.world.coverage_rate)
                done_steps.append(episode_step)
                episode_connectivity.append(0)
                episode_step = 0
                obs_n = env.reset()

            if num_ep >= num_eps:
                print("episodes: %d, mean rps: %.2f, mean coverage: %.3f, mean connectivity: %.3f mean steps: %.2f"
                      % (num_eps,
                         float(np.mean(episode_rewards)),
                         float(np.mean(episode_coverage)),
                         float(np.mean(episode_connectivity)),
                         float(np.mean(done_steps))
                         ))
                break




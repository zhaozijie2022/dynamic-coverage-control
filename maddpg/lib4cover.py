import tensorflow as tf
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import pickle
import time
import os
import argparse


def mlp_model(input, num_outputs, scope, reuse=False, num_units=128):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, r_cover=0.2, r_comm=0.4, comm_r_scale=0.9, comm_force_scale=0.0):
    """环境部分"""
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    # 使用在"./multiagent/scenarios/scenario_name.py"中定义的Scenario类来实例对象
    scenario = scenarios.load(scenario_name + ".py").Scenario(r_cover, r_comm, comm_r_scale, comm_force_scale)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        done_callback=scenario.done)
    return env


def get_trainers(env, obs_shape_n, arglist, buffer_path=None):
    """算法部分: 为每个agent创建trainer并添加到trainers_list中"""
    trainers = []
    model = mlp_model  # Actor 和 Critic的网络结构都是mlp
    trainer = MADDPGAgentTrainer
    # 使用MADDPGAgentTrainer定义初始化trainer
    # env.n为agent的个数
    if buffer_path:
        buffers = []
        for i in range(env.n):
            with open(os.path.join(buffer_path, "buffer_" + "agent_%d" % i + ".pkl"), "rb") as fp:
                buffer = pickle.load(fp)
                buffers.append(buffer)
    else:
        buffers = [None] * env.n
    for i in range(env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            replay_buffer=buffers[i],
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def create_dir(scenario_name):
    """构造目录 ./scenario_name/train_data/time_struct/(plots, policy, buffer)"""
    scenario_path = "./train_data/"
    if not os.path.exists(scenario_path):
        os.mkdir(scenario_path)

    tm_struct = time.localtime(time.time())
    experiment_name = "%02d_%02d_%02d_%02d" % \
                      (tm_struct[1], tm_struct[2], tm_struct[3], tm_struct[4])
    experiment_path = os.path.join(scenario_path, experiment_name)
    if os.path.exists(experiment_path):
        os.remove(experiment_path)
    else:
        os.mkdir(experiment_path)

    save_paths = list()
    save_paths.append(experiment_path + "/policy/")
    save_paths.append(experiment_path + "/plots/")
    save_paths.append(experiment_path + "/buffers/")
    for save_path in save_paths:
        os.mkdir(save_path)
    return save_paths[0], save_paths[1], save_paths[2]


def parse_args(scen_name, max_ep_len, num_eps, llrr=1e-2, gma=0.95, batch_size=1024, num_units=128):
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    args = parser.parse_args()

    args.scenario_name = scen_name
    args.max_episode_len = max_ep_len
    args.num_episodes = num_eps
    args.good_policy, args.adv_policy = "maddpg", "maddpg"
    args.lr = llrr
    args.gamma = gma
    args.batch_size = batch_size
    args.num_units = num_units
    return args

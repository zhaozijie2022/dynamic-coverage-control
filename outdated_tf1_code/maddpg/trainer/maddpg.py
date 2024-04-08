import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r * (1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    """参数说明: make_obs_ph_n -- 列表, 其元素为每个agent的观察的placeholder
                act_space_n -- 列表, 其元素为各个agent的action_space
                p_index -- agent的编号, 第i个agent
                p_func, q_func -- Actor网络和Critic网络的网络模型
                local_q_func -- 使用DDPG算法还是MADDPG算法
                grad_norm_clipping -- 梯度修剪"""
    # policy_train
    with tf.variable_scope(scope, reuse=reuse):
        # 获得各个agent的action_space的概率分布类型, make_pdtype 在 distributions.py 中
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # act_pdtype_n -- 列表, 存储各个agent的动作空间的概率分布类型  shape==[n]

        # 创建obs和act的placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        # obs_ph_n, act_ph_n -- 列表, 其元素为各个agent的obs和act的placeholder  shape==[n]

        p_input = obs_ph_n[p_index]  # 第i个policy网络的输入是第i个的观察, shape==[batch_size, obs_shape_n[i]]
        # 使用p_func的网络结构构造p网络, 存储网络参数并初始化
        p = p_func(p_input, num_outputs=int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        # num_outputs=第i个agent的action_space的维数
        # p_func=mlp_model (in train.py), p为多层感知机的输出, shape==[batch_size, num_outputs]
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # U.absolute_scope_name获取作用域的全名
        # U.scope_vars输入为一个变量作用域, 输出是该作用域的所有变量

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)  # 根据p初始化一个概率分布对象  p→logits
        act_sample = act_pd.sample()  # 采样, shape==[batch_size, act_space_n[i].shape]
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        # p_reg为flat的平方均值, 反应该参数的模的大小, 加入损失函数是一种正则化

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        # 观察和动作为critic网络的输入
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        # q_input -- tensor, critic网络的输入,shape = []
        if local_q_func:  # ddpg算法只能看到自己对应agent的obs和act
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3
        # shape = (), 在优化policy时, 损失函数为Q值, 即尽可能地采取高分值的动作, 第二项为正则化
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)
        # U.minimize_and_clip 最小化目标函数并修剪梯度, 返回的是对参数p_func_vars的梯度

        # Create callable functions
        # 创建可调用的函数, 向这些函数传递函数可以将placeholder转换为data
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        # 同步target网络

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}
        # act -- 将inputs=[obs_ph_n[p_index]]输入到p_func中, 得到网络输出p, 即描述动作概率分布的参数, 再根据此参数采样得到动作

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        # TD-target

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))
        # TD-error

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, replay_buffer=None, local_q_func = False):
        # 父类在__init__.py中定义, 其定义了所有的函数但是没写任何内容, 所有的函数都进行了重定义
        self.name = name  # name = "agent_%d" % i
        self.n = len(obs_shape_n)  # self.n agent的个数
        self.agent_index = agent_index  # 第i个agent
        self.args = args
        # 根据obs_shape_n提供的obs形状创建placeholders
        obs_ph_n = []  # obs_ph_n为一个列表, 其元素为每个agent的观察的placeholder
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())


        # 调用p_train和q_train
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,  # 梯度修剪
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = ReplayBuffer(int(1e6))  # 该类在replay_buffer.py中定义
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        # s, a, r, s', is_done
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        # replay_buffer 不满时不更新; 按次数更新(t%100 -- 每100次更新1次)
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        # 随机抽取batch_size个样本
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1 # 采样了多少次, 采样了1次
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            # 将obs输入到target_p网络, 生成target_act
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            #
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
            # 这里的target_q表示该batch_size所有样本的总TD-target
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

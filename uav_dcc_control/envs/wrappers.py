from gym.envs.registration import load
import gym
import numpy as np
from gym import Env
from gym import spaces
import os
from typing import List, Tuple, Dict, Any
# from multi_goals.half_cheetah_dir import HalfCheetahDirEnvMulti
from abc import ABC, abstractmethod
from utils.util import tile_images
from multiprocessing import Process, Pipe
from gym.spaces.box import Box


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self): pass

    @abstractmethod
    def step_async(self, actions): pass

    @abstractmethod
    def step_wait(self): pass

    def close_extras(self): pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task(data)
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        train_envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        self.n_agents = len(observation_space)
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews).reshape(self.n_envs, self.n_agents, 1), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def meta_reset_task(self, task_idxes):
        assert len(task_idxes) == self.n_envs
        for remote, task_idx in zip(self.remotes, task_idxes):
            remote.send(('reset_task', task_idx))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self, task_idx):
        for remote in self.remotes:
            remote.send(('reset_task', task_idx))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)


class DummyVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.n_agents = env.n_agents
        self.n_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews.reshape(self.n_envs, self.n_agents, 1), dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]  # [env_num, agent_num, obs_dim]
        return np.array(obs)

    def reset_task(self, task_idx):
        for env in self.envs:
            env.reset_task(task_idx)

    def meta_reset_task(self, task_idxes):
        assert len(task_idxes) == self.n_envs
        for env, task_idx in zip(self.envs, task_idxes):
            env.reset_task(task_idx)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError



# class EnvWrapper(gym.Wrapper):
#     # 对应corro的VariBadWrapper, 保留bamdp的设定
#     def __init__(self, env, episodes_per_task, **kwargs):
#         # env的类型应该是xxxEnvMulti, 例如HalfCheetahDirEnvMulti
#         # mujoco_multi已经进行了action的normalize, [-1, 1] -> [lb, ub]
#         super().__init__(env)
#         env.reward_range = env.env.reward_range
#         env.metadata = env.env.metadata
#         self.n_agents = env.n_agents
#         self.n_tasks = env.num_tasks
#
#         # region bamdp的设定
#         # add_done_info, 要不要在obs中加入done信息
#         if episodes_per_task > 1:
#             self.add_done_info = True
#         else:
#             self.add_done_info = False
#
#         if self.add_done_info:
#             # obs的最后一维加入done信息
#             for i in range(self.env.n_agents):
#                 if isinstance(self.env.observation_space[i], spaces.Box):
#                     if len(self.env.observation_space[i].shape) > 1:
#                         raise ValueError
#                     self.env.observation_space[i] = spaces.Box(low=np.array([*self.env.observation_space[i].low, 0]),
#                                                                high=np.array([*self.env.observation_space[i].high, 1]),
#                                                                dtype=np.float32)
#                 else:
#                     raise NotImplementedError
#
#         # calculate horizon length H^+
#         self.episodes_per_task = episodes_per_task
#         # counts the number of episodes
#         self.episode_count = 0
#         # count timesteps in BAMDP
#         self.step_count_bamdp = 0.0
#         # the horizon in the BAMDP is the one in the MDP times the number of episodes per task,
#         # and if we train a policy that maximises the return over all episodes
#         # we add transitions to the reset start in-between episodes
#         self.horizon_bamdp = self.episodes_per_task * self.env.max_episode_steps
#         # add dummy timesteps in-between episodes for resetting the MDP
#         self.horizon_bamdp += self.episodes_per_task - 1
#         # this tells us if we have reached the horizon in the underlying MDP
#         self.done_mdp = True
#         # endregion
#
#     def reset(self, task_idx=None):
#         # reset task
#         self.env.reset_task(task_idx)
#         self.episode_count = 0
#         self.step_count_bamdp = 0
#
#         # normal reset
#         obs_n = self.env.reset()
#
#         # TODO marl不满足这个条件, 但没用到bamdp, 暂时不改
#         if self.add_done_info:
#             for i in range(self.env.n_agents):
#                 obs_n[i] = np.concatenate((obs_n[i], [0.0]))
#
#         self.done_mdp = False
#
#         return obs_n
#
#     def reset_mdp(self):
#         obs_n = self.env.reset()
#         # if self.add_timestep:
#         #     state = np.concatenate((state, [self.step_count_bamdp / self.horizon_bamdp]))
#         if self.add_done_info:
#             for i in range(self.env.n_agents):
#                 obs_n[i] = np.concatenate((obs_n[i], [0.0]))
#         self.done_mdp = False
#         return obs_n
#
#     def step(self, actions: List[np.ndarray]):
#         obs_n, reward_n, done, info = self.env.step(actions)
#         if done:
#             self.done_mdp = done
#             info['done_mdp'] = self.done_mdp
#
#         if self.add_done_info:
#             for i in range(self.env.n_agents):
#                 obs_n[i] = np.concatenate((obs_n[i], [float(self.done_mdp)]))
#
#         self.step_count_bamdp += 1
#
#         done_bamdp = False
#         if self.done_mdp:  # done_mdp always False
#             self.episode_count += 1
#             if self.episode_count >= self.episodes_per_task:
#                 done_bamdp = True
#
#         if self.done_mdp and not done_bamdp:
#             info['state_state'] = self.reset_mdp()
#
#         return obs_n, reward_n, done, info

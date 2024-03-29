# Programed by Z.Zhao
# 不考虑连通保持的覆盖控制场景

import numpy as np
from envs.mpe.multiagent.CoverageWorld import CoverageWorld
from envs.mpe.multiagent.core import Agent, Landmark
from envs.mpe.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, num_agents=4, num_pois=20, r_cover=0.25, r_comm=0.5, comm_r_scale=0.9, comm_force_scale=0.5):
        # agents的数量, 起飞位置, poi的数量和起飞位置
        self.num_agents = num_agents
        self.num_pois = num_pois
        self.pos_pois = np.random.uniform(-1, 1, (num_pois, 2))

        self.r_cover = r_cover
        self.r_comm = r_comm
        self.size = 0.02
        self.m_energy = 5.0

        self.rew_cover = 75.0
        self.rew_done = 1500.0
        self.rew_unconnect = -0.0
        self.rew_out = -100

        self.comm_r_scale = comm_r_scale  # r_comm * comm_force_scale = 计算聚合力时的通信半径
        self.comm_force_scale = comm_force_scale  # 连通保持聚合力的倍数

    def make_world(self):
        world = CoverageWorld()
        world.bb = 1.2
        world.boundary = [np.array([world.bb, 0]), np.array([-world.bb, 0]),
                          np.array([0, world.bb]), np.array([0, -world.bb])]

        world.collaborative = True
        num_agents = 4
        num_landmarks = 20

        world.agents = [Agent() for _ in range(num_agents)]  # 代表UAV, size为覆盖面积
        world.landmarks = [Landmark() for _ in range(num_landmarks)]

        for i, agent in enumerate(world.agents):
            agent.name = "agent_%d" % i
            agent.collide = False
            agent.silent = True
            agent.size = self.size
            agent.r_cover = self.r_cover
            agent.r_comm = self.r_comm
            agent.max_speed = 0.5
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "poi_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.size
            landmark.m_energy = self.m_energy

        self.reset_world(world)
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.05, 0.15, 0.05])
            agent.cover_color = np.array([0.05, 0.25, 0.05])
            agent.comm_color = np.array([0.05, 0.35, 0.05])
            agent.state.p_pos = np.zeros(world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            # landmark.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            landmark.state.p_pos = self.pos_pois[i, :]
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.energy = 0.0
            landmark.done, landmark.just = False, False

    def reward(self, agent, world):
        rew = 0.0
        for i, poi in enumerate(world.landmarks):
            if not poi.done:
                dists = [np.linalg.norm(ag.state.p_pos - poi.state.p_pos) for ag in world.agents]
                rew -= min(dists)
                # 距离poi最近的uav, 二者之间的距离作为负奖励, 该poi的energy_to_cover为乘数
            elif poi.just:
                rew += self.rew_cover
                poi.just = False
        if all([poi.done for poi in world.landmarks]):
            rew += self.rew_done
        for i, agent in enumerate(world.agents):
            abs_pos = np.abs(agent.state.p_pos)
            rew += np.sum(abs_pos[abs_pos > 1] - 1) * self.rew_out
            if (abs_pos > 1.5).any():
                rew += self.rew_out
        return rew

    def observation(self, agent, world):
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        pos_pois = []
        for poi in world.landmarks:
            pos_pois.append(poi.state.p_pos - agent.state.p_pos)
            pos_pois.append([poi.energy, poi.m_energy, poi.done])
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + pos_pois)

    def done(self, agent, world):
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            if (abs_pos > 1.5).any():
                return True
        return all([poi.done for poi in world.landmarks])















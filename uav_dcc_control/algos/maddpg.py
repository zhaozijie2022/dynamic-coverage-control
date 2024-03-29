import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import utils.pytorch_utils as ptu
from torchkit.networks import Mlp
from typing import List


class MADDPGActor(Mlp):
    def forward(self, inputs, **kwargs):
        output = super().forward(inputs, **kwargs)
        return torch.tanh(output)
        # return output


class MADDPG(nn.Module):
    # 每个agent对应一个MADDPG
    def __init__(self, n_agents: int, agent_id: int,
                 obs_dim_n: List[int], action_dim_n: List[int], context_dim: int,
                 actor_layers: list, critic_layers: list,
                 actor_lr=5e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.01,
                 clip_grad_value=None):
        super().__init__()

        self.n_agents = n_agents
        self.agent_id = agent_id

        self.obs_dim_n = obs_dim_n
        self.action_dim_n = action_dim_n
        self.obs_dim = obs_dim_n[agent_id]
        self.action_dim = action_dim_n[agent_id]
        self.context_dim = context_dim

        self.gamma = gamma
        self.tau = tau
        self.clip_grad_value = clip_grad_value

        self.policy = MADDPGActor(input_size=self.obs_dim + context_dim, output_size=self.action_dim,
                                  hidden_sizes=actor_layers)
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

        self.qf = Mlp(input_size=sum(obs_dim_n) + sum(action_dim_n) + context_dim, output_size=1,
                      hidden_sizes=critic_layers)
        self.qf_optim = Adam(self.qf.parameters(), lr=critic_lr)

        self.policy_target = copy.deepcopy(self.policy)
        self.qf_target = copy.deepcopy(self.qf)

        self.train_step = 0
        # 每个agent维持一个hidden_state, 用于执行, 在每个episode开始时清零

    def select_action(self, obs: np.ndarray, context: torch.tensor):
        # 此函数只用于执行, 不用于训练, obs维度为(obs_dim), context: (context_dim,)
        obs = torch.FloatTensor(obs).to(ptu.device)
        x = torch.cat([obs, context], dim=-1)
        action = self.policy(x)
        return action.detach().cpu().numpy()

    def update(self, obs_n, action_n, reward_n, next_obs_n, done_n, context, **kwargs):
        # xxx_n: List[tensor(batch_size, xxx_dim), n_agents]
        # context: tensor(batch_size, context_dim), 由(context_dim, )广播而来

        self.train_step += 1
        agents = kwargs["agents"]

        with torch.no_grad():
            # 计算next_action以计算target_q
            next_action_n = [torch.zeros_like(action, device=ptu.device) for action in action_n]
            for j, agent_j in enumerate(agents):
                input_policy = torch.cat([obs_n[j], context], dim=-1)
                next_action_n[j] = agent_j.policy_target(input_policy)

            # 计算 target q
            input_q_ma = torch.cat([torch.cat([next_obs_n[i], next_action_n[i]], dim=-1)
                                    for i in range(self.n_agents)], dim=-1)
            input_q = torch.cat([input_q_ma, context], dim=-1)
            # (batch_size, (obs_dim + action_dim) * n_agents + context_dim)
            next_q_target = self.qf_target(input_q)
            q_target = (reward_n[self.agent_id] +
                        (1 - done_n[self.agent_id]) * self.gamma * next_q_target).detach()

        input_q_ma = torch.cat([torch.cat([obs_n[i], action_n[i]], dim=-1)
                                for i in range(self.n_agents)], dim=-1)

        input_q_ae = input_q_ma.detach().clone()  # 当ae需要使用rl_loss时, 准备一个input_q_ae用于计算rl_loss
        target_q_ae = q_target.detach().clone()

        input_q = torch.cat([input_q_ma, context], dim=-1)
        q_pred = self.qf(input_q)
        qf_loss = F.mse_loss(q_pred, q_target)

        input_policy = torch.cat([obs_n[self.agent_id], context], dim=-1)
        action_n[self.agent_id] = self.policy(input_policy)

        input_q_ma = torch.cat([torch.cat([obs_n[i], action_n[i]], dim=-1)
                                for i in range(self.n_agents)], dim=-1)
        input_q = torch.cat([input_q_ma, context], dim=-1)

        policy_loss = - self.qf(input_q).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.policy)
        self.policy_optim.step()

        self.qf_optim.zero_grad()
        qf_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.qf)
        self.qf_optim.step()

        self.soft_target_update()
        agent_loss = {"agent_%d_qf_loss" % self.agent_id: qf_loss.item(),
                      "agent_%d_policy_loss" % self.agent_id: policy_loss.item()}

        return agent_loss, input_q_ae, target_q_ae

        # return {"agent_%d_qf_loss" % self.agent_id: qf_loss.item(),
        #         "agent_%d_policy_loss" % self.agent_id: policy_loss.item()},

    def soft_target_update(self):
        ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)
        ptu.soft_update_from_to(self.qf, self.qf_target, self.tau)

    def hard_target_update(self):
        ptu.copy_model_params_from_to(self.policy, self.policy_target)
        ptu.copy_model_params_from_to(self.qf, self.qf_target)

    def _clip_grads(self, net):
        for p in net.parameters():
            p.grad.data.clamp_(-self.clip_grad_value, self.clip_grad_value)

    def save_model(self, save_path):
        save_path = os.path.join(save_path, "agent_%d" % self.agent_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model_dict = {
            "policy": self.policy.state_dict(),
            "qf": self.qf.state_dict(),
        }
        for key in model_dict.keys():
            tmp_save_path = os.path.join(save_path, key + ".pth")
            torch.save(model_dict[key], tmp_save_path)

    def load_model(self, load_path):
        load_path = os.path.join(load_path, "agent_%d" % self.agent_id)
        model_dict = {
            "policy": self.policy,
            "qf": self.qf,
        }
        for key in model_dict.keys():
            tmp_load_path = os.path.join(load_path, key + ".pth")
            model_dict[key].load_state_dict(torch.load(tmp_load_path))
        self.hard_target_update()


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# Dynamic Coverage Control of UAV Swarm under Connectivity Preservation

Based on MARL (Multi-Agent Reinforcement Learning), 
this project provides a dynamic coverage control algorithm for UAV swarm.
Our task is to plan the flight route of the UAV swarm
so that all discrete PoIs (Points of Interest) can be covered for a certain period of time.  
基于多智能体强化学习, 本项目提供了一个面向无人机集群的动态覆盖控制方法.
我们的任务是规划无人机集群的轨迹, 使得在一段时间内, 所有PoIs都能被覆盖住.


Considering the particularity of UAV swarm control, 
this project focuses on analyzing how to maintain the communication connectivity of the swarm during execution.  
进阶要求是无人机在执行任务时连通保持(在后续给出定义)

## 1.Dynamic Point Coverage Environment  动态点覆盖控制的问题定义
There are $N$ UVAs, $x_i^t$ and $M$ PoIs, $p_j$.
Each PoI has certain detection requirements, 
and each UAV can provide detection capabilities within a certain range of space.
We use the concept of energy and power to describe, 
the UAV swarm can provide power for point $p_j$, 
the power value changes with the movement of the UAV swarm.
Over time, the obtained by PoI $p_j$ will accumulate as energy.
When the energy obtained by $p_j$ is greater than required, 
the task is considered to be completed.  
有$N$个无人机，分别为$x_i^t$，以及$M$个PoIs，分别为$p_j$。
每个PoI都有特定的检测需求，而每个无人机都能在一定范围内提供检测能力。
我们使用能量和功率的概念来描述，无人机群体可以为PoI $p_j$提供功率，该功率值随着无人机群体的移动而变化。
随着时间的推移，点$p_j$获得的功率将累积为能量。
当$p_j$获得的能量大于所需能量时，任务被视为已完成。

Problem can be formulated as follow,

$$
P_{i,j}^t(x^t, p_j) = M_p \exp [-(x_i^t-p_j)^2 / r^2]  , \|x_i^t-p_j\| \< r 
$$

, where $M_p$ is denoted by the peak power of the sensor, $r$ is the detection radius of the sensor.

$$
E^t_{N,j} = \int_0^t \sum_{i=1}^N Pow^\tau_{i,j} d\tau
$$

,where $E^r_j$ is denoted by the required energy of $p_j$. 
When $\forall j, [E_{N,j}^t\geq E^r_j]$, 
the task is considered to be completed.

### Reinforcement Environment
We build a dynamic coverage environment based on [Multiagent-Particle-Envs](https://github.com/openai/multiagent-particle-envs).
Class `CoverageWorld` inherits from `multiagent.core.world`, in whose `step()`, the power and energy are calculated, and the PoI state is updated.
`multiagent/scenarios/coverage.py` describes the dynamic coverage scenario.
`multiagent/render.py` has been modified to display in real time the current power obtained by PoIs and the communication between the UAVs.
Some other changes, such as adding connectivity maintaining constraints, revising action according to constraints, will be mentioned later.  
我们基于Multiagent-Particle-Envs构建了一个动态覆盖环境。
CoverageWorld类继承自multiagent.core.world，在其step()方法中计算功率和能量，并更新了PoI的状态(通过判断已获得功率和所需功率, 判定是否已经完成覆盖)。
scenarios中的文件描述了动态覆盖场景, 其中coverage1.py是不考虑连通保持的版本, coverage2.py是考虑连通保持的版本. 
multiagent/render.py被修改以实时显示PoIs获得的当前功率以及无人机之间的通信连通情况。
其他一些更改，比如添加保持连接性的约束条件，根据约束条件修改动作，稍后会提到。


## 2. Dynamic Control based MARL
The agent's observations include its own position and velocity, 
as well as the relative positions of other agents and PoIs.
The actions of the agent include forward, backward, left, and right, and keeping still.
As a purely cooperative scenario, the rewards of all agents are the same and are set as follows,   
Agent的观测包括自身的位置和速度，以及其他agent和PoIs的相对位置, 当前能量, 所需能量, 是否完成覆盖的标志位。
Agent的动作包括前进、后退、向左、向右和保持静止。
作为一个纯粹的合作场景，所有agent的奖励都相同，并设置如下：

$$
R^t = R_{task}(\|M_d^t-M_d^{t-1}\|) + R_{done}^t + R_{dist}\sum_{j\in M_d^t} \min_i \\|x_i^t-p_j\\|_2
$$

,where $M_d^t$ is the set of done PoIs at time t, the first term represents a one-time reward for completing the coverage of a single poI;
$R_{done}^t$ is one-time reward for task completion, which is equal to 0 only when the task is completed;
The third item is the sum of distance of undone PoIs and its nearest agent.
The third term of the reward is crucial because it solves the problem of reward sparsity.   
其中，$M_d^t$是在时间$t$完成的PoIs的集合，第一项表示完成单个PoI覆盖的一次性奖励；
$R_{done}^t$是任务完成的一次性奖励，仅当任务完成时等于0；
第三项是未完成的PoIs及其最近代理之间距离的总和。
奖励的第三项非常关键，因为它作为引导项使得奖励更加密集。


The trained resulted is displayed as follow, (2 and 3 is under connectivity preservation)

<div style="text-align: center;">
  <img src="https://github.com/zhaozijie2022/images/blob/master/dynamic-coverage-control/cov1.gif" width="250px">
  <img src="https://github.com/zhaozijie2022/images/blob/master/dynamic-coverage-control/cov2.gif" width="250px">
  <img src="https://github.com/zhaozijie2022/images/blob/master/dynamic-coverage-control/cov3.gif" width="250px">
</div>


<div style="text-align: center;">
  <img src="https://github.com/zhaozijie2022/images/blob/master/dynamic-coverage-control/cc.png" width="250px">
  <img src="https://github.com/zhaozijie2022/images/blob/master/dynamic-coverage-control/stp.png" width="250px">
  <img src="https://github.com/zhaozijie2022/images/blob/master/dynamic-coverage-control/trajectory3.png" width="250px">
</div>

## 3. MARL Code
MAPPO-based code in uav_dcc_control
基于MAPPO算法的torch的代码在uav_dcc_control中, 目前实现了场景1(无连通保持约束下的覆盖)和场景2(规则约束下的连通保持覆盖). 

场景3(基于动作矫正器的连通保持覆盖)在基于tensorflow的代码中, 我已经看不懂了, 能看懂tf1的可以试着看一下

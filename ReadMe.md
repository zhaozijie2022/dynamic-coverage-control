# Dynamic Coverage Control of UAV Swarm under Connectivity Preservation

Based on MARL (Multi-Agent Reinforcement Learning), 
this project provides a dynamic coverage control algorithm for UAV swarm.
Our task is to plan the flight route of the UAV swarm
so that all discrete PoIs (Points of Interest) can be covered for a certain period of time.
基于强化学习算法为无人机集群导航, 我们的任务时规划无人机的路线, 使得所有的PoI都得到覆盖.

Considering the particularity of UAV swarm control, 
this project focuses on analyzing how to maintain the communication connectivity of the swarm during execution.
我们的目标是在执行覆盖任务的同时保持连通

## 1.Dynamic Point Coverage Environment
### Problem Formulation
There are $N$ UVAs, $x_i^t$ and $M$ PoIs, $p_j$.
Each PoI has certain detection requirements, 
and each UAV can provide detection capabilities within a certain range of space.
We use the concept of energy and power to describe, 
the UAV swarm can provide power for point $p_j$, 
the power value changes with the movement of the UAV swarm.
Over time, the obtained by PoI $p_j$ will accumulate as energy.
When the energy obtained by $p_j$ is greater than required, 
the task is considered to be completed.

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


## 2. Dynamic Control based MADDPG
The agent's observations include its own position and velocity, 
as well as the relative positions of other agents and PoIs.
The actions of the agent include forward, backward, left, and right, and keeping still.
As a purely cooperative scenario, the rewards of all agents are the same and are set as follows,

$$
R^t = R_{task}(\|M_d^t-M_d^{t-1}\|) + R_{done}^t + R_{dist}\sum_{j\in M_d^t} \min_i \\|x_i^t-p_j\\|_2
$$

,where $M_d^t$ is the set of done PoIs at time t, the first term represents a one-time reward for completing the coverage of a single poI;
$R_{done}^t$ is one-time reward for task completion, which is equal to 0 only when the task is completed;
The third item is the sum of distance of undone PoIs and its nearest agent.
The third term of the reward is crucial because it solves the problem of reward sparsity.

The trained resulted is displayed as follow, (2 and 3 is unde connectivity preservation)

<img src="https://github.com/zhaozijie2022/images/blob/master/dynamic-coverage-control/cov1.gif" width="250px"> <img src="https://github.com/zhaozijie2022/images/blob/master/dynamic-coverage-control/cov2.gif" width="250px"> <img src="https://github.com/zhaozijie2022/images/blob/master/dynamic-coverage-control/cov3.gif" width="250px">

## To be Continued

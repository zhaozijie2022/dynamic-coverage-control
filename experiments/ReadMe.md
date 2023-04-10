# Minitest-1 无连通保持/无避碰场景下的无人机动态覆盖控制
以OpenAI于2017年给出的MPE环境和MADDPG原始代码为基础, 修改如下
## ./multiagent/scenarios/coveragr_1
具有4个UAVs和20个PoIs的覆盖任务场景

1. make_world与reset_world \
UAV使用multiagent.core中的Agent类, PoI使用multiagent.core中的Landmark类, CoverWorld继承multiagent.core中的World类.
对CoverWorld, 重写step函数, 增加更新每个poi的energy的函数: 对未完成覆盖的poi, 遍历agents, 
检查该poi是否位于agent覆盖范围, 在则新增energy. 若此轮新增后poi完成覆盖, 则将poi.done和poi.just置1.
\
p.s. poi的颜色随energy的升高而变化直到完成覆盖任务


UAV设置为不可碰撞, 增加属性r_cover和r_comm以及对应的颜色cover_color和comm_color.

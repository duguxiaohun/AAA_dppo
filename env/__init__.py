import gym

# Town10 原始场景（两个名字指向同一实现）
gym.register(
    id="Carla-v0",
    entry_point="env.carla_env:InterSection",
)
gym.register(
    id="CarlaTown10Cross-v0",
    entry_point="env.carla_env:InterSection",
)

# 兼容别名（便于在 YAML 中按 carlaenv10 命名切换）
gym.register(
    id="carlaenv10-v0",
    entry_point="env.carla_env:InterSection",
)

# Town05 十字路口左转场景
gym.register(
    id="CarlaTown05Cross-v0",
    entry_point="env.carla_env_town05:InterSection",
)


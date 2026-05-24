import gym

# Town10 十字路口场景
gym.register(
    id="CarlaTown10Cross-v0",
    entry_point="env.carla_env_town10:InterSection",
)

# Town05 十字路口左转场景
gym.register(
    id="CarlaTown05Cross-v0",
    entry_point="env.carla_env_town05:InterSection",
)

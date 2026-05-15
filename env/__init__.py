import gym

# Town10 原始场景
gym.register(
    id="Carla-v0",
    entry_point="env.carla_env:InterSection",
)

# Town05 十字路口左转场景
gym.register(
    id="CarlaTown05Cross-v0",
    entry_point="env.carla_env_town05:InterSection",
)

# 兜底别名
try:
    gym.register(
        id="Carla",
        entry_point="env.carla_env:InterSection",
    )
except gym.error.RegistrationError:
    pass


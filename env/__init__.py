import gym
from env.carla_env import InterSection

# 主注册
gym.register(
    id="Carla-v0",
    entry_point="env.carla_env:InterSection",
)



# 兜底：有地方传 "Carla" 也能跑
try:
    gym.register(
        id="Carla",
        entry_point="env.carla_env:InterSection",
    )
except gym.error.RegistrationError:
    pass



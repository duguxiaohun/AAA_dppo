import gym


def _register_or_replace(env_id, entry_point):
    registry = gym.envs.registration.registry
    if hasattr(registry, "env_specs") and env_id in registry.env_specs:
        del registry.env_specs[env_id]
    elif env_id in registry:
        del registry[env_id]

    gym.register(id=env_id, entry_point=entry_point)


def register_local_carla_envs():
    _register_or_replace(
        "CarlaTown10Cross-v0",
        "script.carla_dataset_collection.envs.carla_env_town10:InterSection",
    )
    _register_or_replace(
        "CarlaTown05Cross-v0",
        "script.carla_dataset_collection.envs.carla_env_town05:InterSection",
    )
    _register_or_replace(
        "CarlaTown03Cross-v0",
        "script.carla_dataset_collection.envs.carla_env_town03:InterSection",
    )


register_local_carla_envs()

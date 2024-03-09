import gymnasium as gym


def create_env(env_name, seed=42):
    env = gym.make(env_name, render_mode="rgb_array")
    action_space = env.action_space
    action_count = action_space.n if isinstance(action_space, gym.spaces.Discrete) else -1
    obs, info = env.reset(seed=seed)
    return env, obs, info, action_count

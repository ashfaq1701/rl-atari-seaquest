import tensorflow as tf
from src.env import create_env
from src.models import get_model
from src.training import play_multiple_episodes, discount_and_normalize_rewards


def run_main_loop(
        n_iterations,
        n_episodes,
        n_max_steps,
        discount_factor,
        model_type,
        env_name,
        env_seed=None,
        model_seed=None):
    env, obs, _, action_count = create_env(env_name, env_seed)
    model = get_model(model_type, obs.shape, action_count, model_seed)

    loss_fn = tf.keras.losses.sparse_categorical_crossentropy
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)

    mean_reward_per_iteration = []

    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn)
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

        total_rewards = sum(map(sum, all_rewards))
        mean_reward = total_rewards / n_episodes
        print(f"Iteration {iteration + 1} / {n_iterations}: Mean reward: {mean_reward:.1f}")
        mean_reward_per_iteration.append(mean_reward)

        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                 for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)

        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    return mean_reward_per_iteration, model

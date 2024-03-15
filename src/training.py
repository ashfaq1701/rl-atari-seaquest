import numpy as np
import tensorflow as tf
from gymnasium import Env


# Reference: 9.	A. Geron, “Hands on machine learning with Scikit-Learn and Tensorflow,” O’Reilly: 1044-1099
def play_one_step(env: Env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        probas = model(obs[np.newaxis])
        action = tf.constant([[np.random.choice(range(probas.shape[-1]), p=probas.numpy()[-1])]])
        loss = tf.reduce_mean(loss_fn(action, probas))

    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, truncated, _ = env.step(action[0, 0].numpy())
    return obs, reward, done, truncated, grads


# Reference: 9.	A. Geron, “Hands on machine learning with Scikit-Learn and Tensorflow,” O’Reilly: 1044-1099
def play_multiple_episodes(env: Env, iteration_idx, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs, info, = env.reset(seed=iteration_idx * 1000 + episode * 10)
        for step in range(n_max_steps):
            obs, reward, done, truncated, grads = play_one_step(
                env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)

            if done or truncated:
                break

        all_rewards.append(current_rewards)
        all_grads.append(current_grads)

    return all_rewards, all_grads


# Reference: 9.	A. Geron, “Hands on machine learning with Scikit-Learn and Tensorflow,” O’Reilly: 1044-1099
def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted


# Reference: 9.	A. Geron, “Hands on machine learning with Scikit-Learn and Tensorflow,” O’Reilly: 1044-1099
def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std() + np.finfo(float).eps

    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]



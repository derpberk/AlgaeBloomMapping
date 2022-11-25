from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from monoagent_environment_class import SingleAgentEnvironment
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np


vectorized = True

# Creamos el escenario e imponemos las condiciones de entrenamiento #
nav_map = np.ones((50, 50))

env = SingleAgentEnvironment(detection_radius=3,
                             navigation_map=nav_map,
                             initial_position=np.array([20, 20]),
                             total_distance=300,
                             movement_distance=3,
                             max_colisions=15)

if vectorized:
	env_make = lambda: SingleAgentEnvironment(detection_radius=3,
	                                          navigation_map=nav_map,
	                                          initial_position=np.array([20, 20]),
	                                          total_distance=300,
	                                          movement_distance=3,
	                                          max_colisions=15)

	venv = DummyVecEnv([env_make])
	env = VecFrameStack(venv, n_stack=3, channels_order="first")

model = DQN.load('TrainedModels/dqn_algae_blooms_stacked.zip', env=env)

mean_reward, std_reward = evaluate_policy(model,
                                          model.get_env(),
                                          n_eval_episodes=10,
                                          render=True,
                                          deterministic=True)

print(f"Reward: {mean_reward} +- {1.96 * std_reward} [95% conf.]")
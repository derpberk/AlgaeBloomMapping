from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from monoagent_environment_class import SingleAgentEnvironment
from custom_cnn_policy import CustomCNN
import numpy as np
import torch

# Creamos el escenario e imponemos las condiciones de entrenamiento #
nav_map = np.ones((50,50))

env = SingleAgentEnvironment(detection_radius=2, 
                                    navigation_map=nav_map, 
                                    initial_position=np.array([20,20]), 
                                    total_distance=200, 
                                    movement_distance=3, 
                                    max_colisions=15)

eval_Env = SingleAgentEnvironment(detection_radius=2, 
                                    navigation_map=nav_map, 
                                    initial_position=np.array([20,20]), 
                                    total_distance=200, 
                                    movement_distance=3, 
                                    max_colisions=1)


# Creamos el diccionario para describir cómo va a ser la política #
policy_kwargs = dict(
    features_extractor_class = CustomCNN,
    features_extractor_kwargs = dict(features_dim=128),
    net_arch = [128, 128, 128],
    activation_fn = torch.nn.ReLU,
)

# Creamos el model de entrenamiento con sus hiperparámetros #
model = DQN("CnnPolicy", 
            env = env, 
            policy_kwargs = policy_kwargs,
            learning_rate = 1e-4,
            buffer_size = 1_000,
            learning_starts = 200,
            batch_size = 32,
            tau  = 1.0,
            gamma = 0.99,
            target_update_interval = 100,
            exploration_fraction = 0.1,
            exploration_initial_eps  = 1.0,
            exploration_final_eps = 0.05,
            tensorboard_log = './Logs',
            verbose=1)


model.learn(total_timesteps = 100,
            log_interval = 4,
            eval_env = eval_Env,
            eval_freq = 10,
            n_eval_episodes = 10,
            progress_bar=True)

model.save("dqn_algae_blooms")

mean_reward, std_reward = evaluate_policy(model, 
                                            model.get_env(), 
                                            n_eval_episodes=10, 
                                            render=True,
                                            deterministic=False)

print(f"Reward: {mean_reward} +- {1.96*std_reward} [95% conf.]")

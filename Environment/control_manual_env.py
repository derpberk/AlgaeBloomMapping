from monoagent_environment_class import SingleAgentEnvironment
import numpy as np
from pynput import keyboard
import time

def read_key():

    with keyboard.Events() as events:
        # Block for as much as possible
        event = events.get(1e6)

        if event.key == keyboard.KeyCode.from_char('x'):
            a = 0
        elif event.key == keyboard.KeyCode.from_char('c'):
            a = 1
        elif event.key == keyboard.KeyCode.from_char('d'):
            a = 2
        elif event.key == keyboard.KeyCode.from_char('e'):
            a = 3
        elif event.key == keyboard.KeyCode.from_char('w'):
            a = 4
        elif event.key == keyboard.KeyCode.from_char('q'):
            a = 5
        elif event.key == keyboard.KeyCode.from_char('a'):
            a = 6
        elif event.key == keyboard.KeyCode.from_char('z'):
            a = 7
        else:
            return None

        time.sleep(0.1)

        return a

nav_map = np.ones((50,50))
env = SingleAgentEnvironment(detection_radius=2, 
                                navigation_map=nav_map, 
                                initial_position=np.array([20,20]), 
                                total_distance=1000, 
                                movement_distance=3, 
                                max_colisions=50)


_ = env.reset()
env.render()
done = False

while not done:

    action = read_key()

    if action is not None:

        _, r, done, _ = env.step(action)

        env.render()

        print("Reward: ", r)
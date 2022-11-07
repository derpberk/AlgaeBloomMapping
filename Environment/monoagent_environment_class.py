import gym
import numpy as np
from vehicle_class import Vehicle
from algae_bloom_generation import algae_bloom, algae_colormap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SingleAgentEnvironment(gym.Env):
    """ Escenario para un solo agente """

    def __init__(self, navigation_map, initial_position, total_distance, movement_distance, max_colisions, detection_radius) -> None:
        """ Inicializamos el escenario. """
        super(gym.Env).__init__()

        # Creamos el objeto tipo Vehicle #
        self.my_vehicle = Vehicle(initial_position = initial_position,
                                obstacle_map = navigation_map,
                                max_travel_distance = total_distance,
                                movement_length = movement_distance) 
        
        # Guardamos el mapa de navegación para dibujarlo luego
        self.navigation_map = navigation_map
        self.max_colisions = max_colisions
        self.colisions = 0
        self.detection_mask = None
        self.detection_radius = detection_radius
        self.event_map = None
        self.idleness_map = None
        self.fig = None
        self.algae_bloom_gt = algae_bloom(self.navigation_map.shape)

        # Creamos atributos del escenario GYM:
        # 1. El action space -> Cuántas acciones tenemos.
        self.action_space = gym.spaces.Discrete(8)
        # 2. El obs spcae -> Cómo es el estado.
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,*navigation_map.shape))

    def compute_detection_mask(self, pos):

        pos = pos.astype(int)

        detection_mask = np.zeros_like(self.navigation_map)

        h_bound = np.clip((pos[0] - self.detection_radius, pos[0] + self.detection_radius +1), 0, self.navigation_map.shape[0])
        v_bound = np.clip((pos[1] - self.detection_radius, pos[1] + self.detection_radius +1), 0, self.navigation_map.shape[1])

        detection_mask[h_bound[0] : h_bound[1],
                       v_bound[0] : v_bound[1]] = 1.0

        return detection_mask

    def reset(self):
        """ Reseteamos todas las variables """

        # Reseteamos el agente #
        self.my_vehicle.reset_agent()
        # Reseteamos la mascara de deeccion
        self.detection_mask = self.compute_detection_mask(self.my_vehicle.position)
        # Reseteamos el mapa del modelo y el de idleness #
        self.idleness_map = self.navigation_map.copy()
        self.previous_model = np.zeros_like(self.navigation_map)
        self.model = np.zeros_like(self.navigation_map)
        # Reseteamos el mapa de algas #
        self.algae_bloom_gt.reset()
        # Actualizamos el modelo y las matrices #
        self.update_maps()
        # Reseteamos el estado
        self.state = self.generate_states()
        
        # Devolvemos el estado inicial s0
        return self.state

    def update_maps(self):
        """ Actualizamos el mapa de idleness y el mapa de eventos (aleatorios) """

        # Aumentamos cada posición del IDLENESS MAP en 0.01 
        self.idleness_map = np.clip(self.idleness_map + 0.01, 0.0, 1.0) * self.navigation_map
        # Ponemos el idleness en la zona actual del agente a 0
        self.idleness_map = self.idleness_map * (1.0 - self.detection_mask)
        # Actualizamos nuestro modelo con el DETECTION MASK #
        self.previous_model = self.model.copy()
        self.model = self.model * (1.0 - self.detection_mask) + self.algae_bloom_gt.algae_map * self.detection_mask
        # Actualizamos el mapa de las algas #
        self.algae_map = self.algae_bloom_gt.step()

    def generate_states(self):
        """ Generamos el estado como un conjunto de imagenes """

        state = np.zeros(shape=(3, *self.navigation_map.shape))
        # Mapa de obstáculos #
        # state[0] = self.navigation_map.copy()
        # Mapa de idleness #
        state[0] = self.idleness_map.copy()
        # Mapa de posición #
        state[1] = self.detection_mask * 0.5
        state[1,self.my_vehicle.position[0], self.my_vehicle.position[1]] = 1.0
        # El modelo de algas #
        state[2] = self.model

        return state


    def step(self, action: int):
        """ Procesamos la acción y devolvemos un estado, una recompensa, un done y otra info. """

        # Movemos el vehículo #
        result = self.my_vehicle.move(action)

        # Calculamos la nueva máscara de deteccion #
        self.detection_mask = self.compute_detection_mask(self.my_vehicle.position)

        # Calculamos cuántos eventos hemos detectado #
        informacion_detectada = np.sum(np.abs(self.model - self.previous_model))
        idleness_cubierto = np.sum(self.idleness_map * self.detection_mask)/(np.pi * self.detection_radius**2)

        # Calculamos la recompensa en función de lo que ha ocurrido
        if result == "COLLISION":
            # Tenemos colision #
            self.colisions += 1
            done = self.colisions >= self.max_colisions
            reward = -1.0

        else:
            # Si no hay colisión, calculamos la recompensa 
            reward = informacion_detectada + idleness_cubierto
            done = result == "DISTANCE"

        # Obtenemos el nuevo estado #
        self.state = self.generate_states()

        # Actualizamos el mapa de eventos #
        self.update_maps()

        # Devolvemos (s,r,d,info)
        return self.state, reward, done, {}


    def render(self, mode = 'human'):
        """ Pintamos el escenario """

        if self.fig is None:

            self.fig, self.axs = plt.subplots(1,4, figsize=(10,4))

            #self.d0 = self.axs[0].imshow(self.state[0], cmap = 'gray')
            self.d1 = self.axs[0].imshow(self.state[0], cmap = 'gray')
            self.d2 = self.axs[1].imshow(self.state[1], cmap = 'jet')
            back = np.zeros_like(self.navigation_map) + 0.15
            back[::2, 1::2] = 0.3
            back[1::2, ::2] = 0.3
            self.axs[2].imshow(back, cmap = 'gray', zorder = 1, vmin=0, vmax=1)
            self.d3 = self.axs[2].imshow(self.state[2], vmin=0.0, vmax=1.0, cmap = algae_colormap, alpha= 1.0 - self.idleness_map, zorder=10)
            self.d4 = self.axs[3].imshow(self.algae_map, vmin=0.0, vmax=1.0, cmap = algae_colormap)


        else:

            #self.d0.set_data(self.state[0])
            self.d1.set_data(self.state[0])
            self.d2.set_data(self.state[1])
            self.d3.set_data(self.state[2])
            self.d3.set_alpha(1.0 - self.idleness_map)
            self.d4.set_data(self.algae_map)


        self.fig.canvas.draw()
        plt.draw()
        plt.pause(0.01)



if __name__ == "__main__":

    nav_map = np.ones((50,50))
    env = SingleAgentEnvironment(detection_radius=3, 
                                    navigation_map=nav_map, 
                                    initial_position=np.array([10,10]), 
                                    total_distance=500, 
                                    movement_distance=3, 
                                    max_colisions=15)


    _ = env.reset()
    env.render()
    done = False
    r = -1
    t = 0


    while not done:
        t += 1
        if r < 0 or t % 3 == 0:
            action = env.action_space.sample()

        _, r, done, _ = env.step(action)

        env.render()

        print("Reward: ", r)

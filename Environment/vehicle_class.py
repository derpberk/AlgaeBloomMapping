import numpy as np


class Vehicle:
    """ La clase vehículo define cómo se comporta un vehículo 
    en el espacio bidimesional: procesa su movimiento, almacena su posición, ... """

    def __init__(self, 
                initial_position: np.ndarray, 
                obstacle_map: np.ndarray,
                max_travel_distance: int,
                movement_length: int) -> None:

        """ Este es el constructor del vehículo. Define cómo se crea un objeto tipo Vehicle. """

        # Creamos los atributos del vehículo #
        self.initial_position = initial_position  # Su posición inicial
        self.position = initial_position  # Su posición actual
        self.max_travel_distance = max_travel_distance  # El número máximo de distancia que recorre el vehículo
        self.movement_length = movement_length  # Cuanto nos movemos por cada movimiento
        self.obstacle_map = obstacle_map  # Mapa de obstaculos
        self.distance = 0  # Distancia recorrida por el agente


    def move(self, action: int):
        """ El vehiculo recibe una acción discret [0,1,2,3,4,5,6,7] y se mueve, si puede. """

        assert 8 > action >= 0, "La accion solicitada es imposible!"

        movement_angle = action / 8.0 * 2*np.pi  # Calculamos el ángulo que nos movemos de 0 a 2pi

        # Computamos el vector de desplazamiento (un vector de enteros) #
        movement_vector  = (self.movement_length * np.array([np.cos(movement_angle), np.sin(movement_angle)])).astype(int)

        # Antes de movernos al sitio, comprobamos que se puede visitar #
        next_position = self.position + movement_vector

        # También comprobamos que la distancia máxima no se ha superado
        if self.distance > self.max_travel_distance:
            # El vehículo ya no se puede mover: está sin batería
            return "DISTANCE"
        elif any(next_position < 0) or any(next_position >= self.obstacle_map.shape):
            # Hay un obstáculo ahí!
            return "COLLISION"
        elif self.obstacle_map[next_position[0], next_position[1]] == 0.0:
            # FUERA DEL MAPA!
            return "COLLISION"
        else:
            # Se puede mover ahí! #
            # Acumulamos la distancia recorrida #
            self.distance += np.linalg.norm(next_position - self.position)
            # Actualizamos la posicion
            self.position = next_position.copy()
            # Devolvemos True para indicar que el movimiento tuvo éxito
            return "OK"


    def reset_agent(self):
        """Este método pone al agente en su posición inicial y resetea la distancia """

        # Reseteamos la posición inicial #
        self.position = self.initial_position.copy()
        # Reseteamos la distancia recorrida #
        self.distance = 0.0


import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.colors

algae_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue","darkcyan", "darkgreen", "forestgreen"])
fuelspill_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue", "olive", "saddlebrown", "indigo"])


class algae_bloom:

    def __init__(self, shape: tuple) -> None:
        """ Generador de ground truths de algas con dinámica """

        # Creamos un mapa vacio #
        self.map = np.zeros(shape)
        self.particles = None
        self.starting_point = None
        x, y = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]))
        self.visitable_positions = np.column_stack((x.flatten(), y.flatten()))
        self.fig = None
        self.dt = 0.2
        self.current_field_fn = np.vectorize(self.current_field, signature="(n) -> (n)")


    def reset(self):

        starting_point = np.array((np.random.randint(self.map.shape[0]/4, 3*self.map.shape[0]/4), np.random.randint(self.map.shape[1]/2, 2* self.map.shape[1]/3)))
        self.particles = np.random.multivariate_normal(starting_point, np.array([[7.0, 0.0],[0.0, 7.0]]),size=(100,))
        
        starting_point = np.array((np.random.randint(self.map.shape[0]/4, 3*self.map.shape[0]/4), np.random.randint(self.map.shape[1]/2, 2* self.map.shape[1]/3)))
        self.particles = np.vstack(( self.particles, np.random.multivariate_normal(starting_point, np.array([[3.0, 0.0],[0.0, 3.0]]),size=(100,))))

        in_bound_particles = np.array([particle for particle in self.particles if self.is_inside(particle)])
        self.map[in_bound_particles[:,0].astype(int), in_bound_particles[:, 1].astype(int)] = 1.0

        self.algae_map = gaussian_filter(self.map, 0.8)
        
        return self.algae_map
        
    def current_field(self, position):

        #u = - np.sin(2 * np.pi * (position[0] - self.map.shape[0] // 2) / self.map.shape[0]) + np.cos(2 * np.pi * (position[1] - self.map.shape[1] // 2) / self.map.shape[1])
        #v = np.cos(2 * np.pi * (position[0] - self.map.shape[0] // 2) / self.map.shape[0]) - np.sin(2 * np.pi * (position[1] - self.map.shape[1] // 2) / self.map.shape[1])
        u = -(position[1] - self.map.shape[1] / 2) / np.linalg.norm(position - np.array(self.map.shape)/2 + 1e-6)
        v = (position[0] - self.map.shape[0] / 2) / np.linalg.norm(position - np.array(self.map.shape)/2 + 1e-6)

        return np.array((u,v))

    def is_inside(self, particle):

        
        particle = particle.astype(int)
        if particle[0] >= 0 and particle[0] < self.map.shape[0] and  particle[1] >= 0 and particle[1] < self.map.shape[1]:
            return True
        else:
            return False

    def step(self):

        self.map[:,:] = 0.0
        
        random_movement = np.random.rand(len(self.particles),2)
        current_movement = self.current_field_fn(self.particles)

        self.particles = self.particles + self.dt * (0.2 * random_movement + current_movement)
        
        in_bound_particles = np.array([particle for particle in self.particles if self.is_inside(particle)])

        self.map[in_bound_particles[:,0].astype(int), in_bound_particles[:, 1].astype(int)] = 1.0

        self.algae_map = gaussian_filter(self.map, 0.8)

        return self.algae_map

    def render(self):
        
        f_map = gaussian_filter(self.map, 0.8)

        if self.fig is None:
            current = self.current_field_fn(self.visitable_positions)

            self.fig, self.ax = plt.subplots(1,1)
            #self.ax.quiver(self.visitable_positions[::6,1], self.visitable_positions[::6,0], current[::6,1], -current[::6,0], color='black', alpha = 0.25)
            self.d = self.ax.imshow(f_map, cmap = algae_colormap, vmin=0.0, vmax = 1.0)
        else:
            self.d.set_data(f_map)

        self.fig.canvas.draw()
        plt.pause(0.01)
    



        

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1)

    gt = algae_bloom((50,50))

    m = gt.reset()
    gt.render()

    for _ in range(1000):

        m = gt.step()
        gt.render()

    


        
        
        
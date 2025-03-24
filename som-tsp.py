import math
import random

class SOM_TSP_Matrix:
    def __init__(self, adjacency_matrix, num_neurons=None, learning_rate=0.8, 
                 decay_rate=0.999, neighborhood_size=None, max_iter=1000):
        """
        Initialize the SOM-TSP solver with City 1 (index 0) as fixed start/end point.
        
        Parameters:
        - adjacency_matrix: The graph adjacency matrix (City 1 is first row/column)
        - num_neurons: Number of neurons in the SOM
        - learning_rate: Initial learning rate
        - decay_rate: Decay rate for learning parameters
        - neighborhood_size: Initial neighborhood size
        - max_iter: Maximum iterations
        """
        self.matrix = adjacency_matrix
        self.n_cities = len(adjacency_matrix)
        
        # Convert matrix to coordinate representation for SOM
        self.cities = self._matrix_to_coordinates()
        
        # Set default parameters
        if num_neurons is None:
            num_neurons = 2 * self.n_cities
        if neighborhood_size is None:
            neighborhood_size = num_neurons // 2
            
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.neighborhood_size = neighborhood_size
        self.max_iter = max_iter
        
        # Initialize neurons with City 1 at fixed position
        self.neurons = self._initialize_neurons()

    def _matrix_to_coordinates(self):
        """Convert adjacency matrix to 2D coordinates with City 1 at origin."""
        # Start with random positions except City 1 at (0,0)
        coords = [(0, 0)]  # City 1 at origin
        coords.extend([(random.random(), random.random()) for _ in range(self.n_cities-1)])
        
        # Run spring model iterations
        for _ in range(100):
            new_coords = [(0, 0)]  # Keep City 1 fixed
            for i in range(1, self.n_cities):  # Only update other cities
                fx, fy = 0.0, 0.0
                for j in range(self.n_cities):
                    if i == j or self.matrix[i][j] == 0:
                        continue
                    
                    dx = coords[j][0] - coords[i][0]
                    dy = coords[j][1] - coords[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    desired_dist = 1.0 / (self.matrix[i][j] + 0.1)
                    
                    if dist > 0:
                        force = (dist - desired_dist) / dist
                        fx += force * dx
                        fy += force * dy
                
                new_x = coords[i][0] + 0.1 * fx
                new_y = coords[i][1] + 0.1 * fy
                new_coords.append((new_x, new_y))
            coords = new_coords
        
        return coords
    
    def _initialize_neurons(self):
        """Initialize neurons with City 1's neuron at fixed position."""
        # Calculate centroid of other cities
        if self.n_cities > 1:
            sum_x = sum(c[0] for c in self.cities[1:])
            sum_y = sum(c[1] for c in self.cities[1:])
            centroid = (sum_x / (self.n_cities-1), sum_y / (self.n_cities-1))
        else:
            centroid = (0, 0)
        
        # Calculate radius from City 1 to farthest city
        max_dist = max(math.sqrt(c[0]**2 + c[1]**2) for c in self.cities)
        
        # Place City 1's neuron at fixed position (top of circle)
        fixed_angle = math.pi/2  # 90 degrees (top position)
        neurons = [
            (centroid[0] + max_dist * math.cos(fixed_angle),
             centroid[1] + max_dist * math.sin(fixed_angle))
        ]
        
        # Place remaining neurons in a circle
        for i in range(1, self.num_neurons):
            angle = 2 * math.pi * i / self.num_neurons + fixed_angle
            x = centroid[0] + max_dist * math.cos(angle)
            y = centroid[1] + max_dist * math.sin(angle)
            neurons.append((x, y))
            
        return neurons
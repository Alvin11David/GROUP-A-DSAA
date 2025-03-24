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
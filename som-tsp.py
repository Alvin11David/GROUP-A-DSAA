import numpy as np
from math import inf
from collections import deque
import random

class StrictAdjacencySOM_TSP:
    def __init__(self, adjacency_matrix, n_iterations=10000, learning_rate=0.8):
        self.adj_matrix = np.array(adjacency_matrix)
        self.n_cities = len(adjacency_matrix)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        
        # Convert adjacency matrix to coordinates
        self.city_coords = self._adjacency_to_coords()
        
        # Initialize SOM parameters
        self.n_neurons = self.n_cities * 3  # More neurons for better coverage
        self.neurons = self._initialize_neurons()
    
    def _adjacency_to_coords(self):
        """Convert adjacency matrix to 2D coordinates using MDS"""
        coords = np.random.rand(self.n_cities, 2)
        for _ in range(100):
            for i in range(self.n_cities):
                for j in range(i+1, self.n_cities):
                    if self.adj_matrix[i,j] != inf:
                        target_dist = self.adj_matrix[i,j] / np.max(self.adj_matrix[self.adj_matrix != inf])
                        current_dist = np.linalg.norm(coords[i] - coords[j])
                        if current_dist > 0:
                            force = (target_dist - current_dist) / current_dist
                            coords[i] -= 0.05 * force * (coords[j] - coords[i])
                            coords[j] += 0.05 * force * (coords[j] - coords[i])
        return coords
    
    def _initialize_neurons(self):
        """Initialize neurons in a circle around city center"""
        center = np.mean(self.city_coords, axis=0)
        radius = 0.5 * np.max(np.ptp(self.city_coords, axis=0))
        angles = np.linspace(0, 2*np.pi, self.n_neurons, endpoint=False)
        return np.array([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])
    
    def train(self):
        """Train the SOM"""
        for iteration in range(self.n_iterations):
            lr = self.learning_rate * (1 - iteration/self.n_iterations)
            radius = max(1, self.n_neurons/2 * np.exp(-iteration/self.n_iterations))
            
            city_idx = random.randint(0, self.n_cities-1)
            winner = np.argmin(np.linalg.norm(self.neurons - self.city_coords[city_idx], axis=1))
            
            # Update neurons
            for i, neuron in enumerate(self.neurons):
                dist_to_winner = min(abs(i - winner), self.n_neurons - abs(i - winner))
                influence = np.exp(-(dist_to_winner**2)/(2*(radius**2))) * lr
                self.neurons[i] += influence * (self.city_coords[city_idx] - neuron)
    
    def get_valid_route(self):
        """Generate route that strictly follows adjacency rules using BFS"""
        # Start with city 0 (1 in display)
        queue = deque()
        queue.append(([0], set([0])))
        
        best_route = None
        best_distance = inf
        
        while queue:
            current_route, visited = queue.popleft()
            
            # Complete the cycle if all cities visited
            if len(visited) == self.n_cities:
                if self.adj_matrix[current_route[-1], 0] != inf:
                    final_route = current_route + [0]
                    distance = sum(self.adj_matrix[final_route[i], final_route[i+1]] 
                                for i in range(len(final_route)-1))
                    if distance < best_distance:
                        best_distance = distance
                        best_route = final_route
                continue
            
            # Explore all valid adjacent cities
            last_city = current_route[-1]
            for next_city in range(self.n_cities):
                if (next_city not in visited and 
                    self.adj_matrix[last_city, next_city] != inf):
                    new_visited = set(visited)
                    new_visited.add(next_city)
                    queue.append((current_route + [next_city], new_visited))
        
        return best_route
    
    def solve(self):
        """Run the complete solution process"""
        self.train()
        route = self.get_valid_route()
        if route:
            distance = sum(self.adj_matrix[route[i], route[i+1]] 
                       for i in range(len(route)-1))
            return route, distance
        return None, inf

# Adjacency matrix
adjacency_matrix = [
    [0, 12, 10, inf, inf, inf, 12],  # City 1 (index 0)
    [12, 0, 8, 12, inf, inf, inf],   # City 2 (index 1)
    [10, 8, 0, 11, 3, inf, 9],      # City 3 (index 2)
    [inf, 12, 11, 0, 11, 10, inf],  # City 4 (index 3)
    [inf, inf, 3, 11, 0, 6, 7],     # City 5 (index 4)
    [inf, inf, inf, 10, 6, 0, 9],   # City 6 (index 5)
    [12, inf, 9, inf, 7, 9, 0]      # City 7 (index 6)
]

# Solve with strict adjacency constraints
solver = StrictAdjacencySOM_TSP(adjacency_matrix)
route, distance = solver.solve()

if route:
    print("Valid Route Found:")
    print("Path:", " -> ".join(str(c+1) for c in route))
    print(f"Total Distance: {distance}")
else:
    print("No valid route found that satisfies all constraints")
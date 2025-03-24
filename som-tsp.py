import random
import math

# Given adjacency matrix (Graph)
graph = [
    [0, 12, 10, 0, 0, 0, 12],  # City 1
    [12, 0, 8, 12, 0, 0, 0],   # City 2
    [10, 8, 0, 11, 3, 0, 9],   # City 3
    [0, 12, 11, 0, 11, 10, 0], # City 4
    [0, 0, 3, 11, 0, 6, 7],    # City 5
    [0, 0, 0, 10, 6, 0, 9],    # City 6
    [12, 0, 9, 0, 7, 9, 0]     # City 7
]

# Number of cities
n_cities = len(graph)

# Generate random (x, y) coordinates for cities
random.seed(42)  # Fix seed for reproducibility
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n_cities)]

# Initialize random neurons (nodes in the SOM)
n_neurons = 20  # More neurons than cities for better learning
neurons = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n_neurons)]

# Hyperparameters
learning_rate = 0.8
sigma = 10
iterations = 5000


def euclidean_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def find_winner(city):
    """Find the neuron closest to the given city (Best Matching Unit - BMU)."""
    min_dist = float("inf")
    best_neuron = 0
    for i, neuron in enumerate(neurons):
        dist = euclidean_distance(city, neuron)
        if dist < min_dist:
            min_dist = dist
            best_neuron = i
    return best_neuron


def update_weights(bmu_index, city, iteration):
    """Adjust weights of neurons using a Gaussian neighborhood function."""
    t = iteration / iterations  # Normalize time
    lr = learning_rate * (1 - t)  # Decay learning rate
    sigma_t = sigma * (1 - t)  # Decay sigma

    for i in range(n_neurons):
        distance_to_bmu = min(abs(i - bmu_index), n_neurons - abs(i - bmu_index))
        influence = math.exp(-distance_to_bmu**2 / (2 * sigma_t**2))
        neurons[i] = (
            neurons[i][0] + lr * influence * (city[0] - neurons[i][0]),
            neurons[i][1] + lr * influence * (city[1] - neurons[i][1])
        )


def train():
    """Train the SOM network to approximate the TSP path."""
    for iteration in range(iterations):
        city = cities[random.randint(0, n_cities - 1)]  # Random city selection
        bmu_index = find_winner(city)  # Find Best Matching Unit (BMU)
        update_weights(bmu_index, city, iteration)  # Update neurons


def get_tsp_path():
    """Get the order of cities based on neuron proximity and ensure it starts and ends at City 1."""
    city_to_neuron = [(i, find_winner(cities[i])) for i in range(n_cities)]
    city_to_neuron.sort(key=lambda x: x[1])  # Sort by neuron index
    path = [c[0] + 1 for c in city_to_neuron]  # Convert to 1-based index

    # Ensure the path starts and ends at City 1
    if path[0] != 1:
        path.remove(1)
        path.insert(0, 1)
    path.append(1)  # Add City 1 at the end to complete the cycle

    return path


def calculate_total_distance(path):
    """Calculate the total distance of the TSP path using the adjacency matrix."""
    total_distance = 0
    for i in range(len(path) - 1):  # Include the return trip to City 1
        city1 = path[i] - 1  # Convert to 0-based index
        city2 = path[i + 1] - 1  # Next city
        distance = graph[city1][city2]

        if distance == 0:  # If no direct connection exists, find nearest neighbor
            neighbors = [graph[city1][j] for j in range(n_cities) if graph[city1][j] > 0]
            if neighbors:
                distance = min(neighbors)  # Use shortest possible adjacent path

        total_distance += distance
    return total_distance


# Train the SOM and get the optimal TSP path
train()
tsp_path = get_tsp_path()
total_distance = calculate_total_distance(tsp_path)

# Print the results
print("Optimal TSP Path (City Index Order):", tsp_path)
print("Total Distance of the Path:", total_distance)

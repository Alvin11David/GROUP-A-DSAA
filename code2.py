from itertools import permutations

def tsp_dynamic(graph):
    n = len(graph)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from city 1 (index 0)

    for mask in range(1 << n):  # Iterate over all subsets of cities
        for i in range(n):  # Last visited city
            if mask & (1 << i):  # Ensure i is in the subset
                for j in range(n):  # Next city to visit
                    if not (mask & (1 << j)) and graph[i][j] > 0:  
                        new_mask = mask | (1 << j)
                        dp[new_mask][j] = min(dp[new_mask][j], dp[mask][i] + graph[i][j])

    # Find the shortest path that returns to the start city (1)
    final_mask = (1 << n) - 1
    return min(dp[final_mask][i] + graph[i][0] for i in range(1, n) if graph[i][0] > 0)

# Graph represented as an adjacency matrix
graph = [
    [0, float('inf'), 10, float('inf'), float('inf'), float('inf'), 12],  # City 1
    [float('inf'), 0, 8, 12, float('inf'), float('inf'), float('inf')],   # City 2
    [10, 8, 0, float('inf'), 3, float('inf'), float('inf')],              # City 3
    [float('inf'), 12, float('inf'), 0, 11, 10, float('inf')],            # City 4
    [float('inf'), float('inf'), 3, 11, 0, 6, 7],                         # City 5
    [float('inf'), float('inf'), float('inf'), 10, 6, 0, 9],              # City 6
    [12, float('inf'), float('inf'), float('inf'), 7, 9, 0]               # City 7
]

print("Minimum TSP Cost:", tsp_dynamic(graph))
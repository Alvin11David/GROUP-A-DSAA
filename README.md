CSC 1204 Data Structures and Algorithms Project: Traveling Salesman Problem Using Classical and SOM-Based Methods
TABLE OF CONTENT
1. OVERVIEW
2. PROBLEM STATEMENT
3. IMPLEMENTATION DETAILS
4. SETUP AND INSTALLATION
5. CONTRIBUTORS
6. LICENSE

OVERVIEW
This repository contains an implementation of the Traveling Salesman Problem (TSP) using both a classical algorithm and a Self-Organizing Map (SOM) approach. The project is part of the CSC 1204: Data Structures and Algorithms practical assignment at Makerere University

PROBLEM STATEMENT
The Traveling Salesman Problem (TSP) requires finding the shortest possible route that visits each city exactly once and returns to the starting city. The dataset consists of a 7-city graph with labeled distances.

IMPLEMENTATION DETAILS
Implementation Details
1. Classical TSP Solution
Implemented using Dynamic Programming (Held-Karp algorithm) for optimality.

Provides an exact solution with O(n² * 2ⁿ) time complexity.

Outputs the optimal route and total travel distance.

2. Self-Organizing Map (SOM) Approach
Uses an unsupervised neural network to approximate the TSP solution.

Key features:

Neurons represent cities, trained to follow the shortest path.

Winner-takes-all rule updates the neurons iteratively.

Uses decaying learning rate and neighborhood function.

Results in a near-optimal solution with faster execution for larger instances.

3. Comparison and Analysis
Route Comparison: Evaluates the effectiveness of both methods.

Computational Complexity: Discusses the trade-off between accuracy and performance.

Use Cases: Identifies scenarios where heuristic approaches (SOM) are preferable over exact algorithms.

SETUP AND INSTALLATION
Ensure you have installed Python.
To run or test the implementations locally, follow these steps:
CLONE the repository in your terminal:
git clone git@github.com:Alvin11David/GROUP-A-DSAA.git
NAVIGATE to the project directory:
cd GROUP-A-DSAA

CONTRIBUTORS
1. Waluube Alvin David          24/U/11805/PS
2. Ageno Elizabeth              24/U/25850/PS
3. Akello Lilian                24/U/03142/PS
4. Namuli Martina Daniella      23/U/14837/EVE
5. Okwir Francis                24/U/10703/EVE

LICENSE
This project is under the MIT license.
https://github.com/Alvin11David/GROUP-A-DSAA/blob/dynamicprogramming/LICENSE




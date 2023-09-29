import csv
import sys
import time
from queue import PriorityQueue

# Initialize an empty dictionary to hold the graph
graph = {}

n = len(sys.argv)

while n != 3:
    print("ERROR: Not enough or too many input arguments.\n")
    break


def getDriving(node_idx, child_idx):
    with open('driving.csv', newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')

        # Iterate over each row in the CSV file
        for row_index, row in enumerate(reader):
            # Check if this is the row we want
            if row_index == node_idx:  # Assuming 0-based indexing, row 2 would be index 1
                # Retrieve the cell value in column 3 (assuming 1-based indexing)
                cell_value = row[child_idx]  # Assuming 0-based indexing, column 3 would be index 2
                return float(cell_value)
                break


def getStraightLine(node_idx, child_idx):
    with open('straightline.csv', newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')

        # Iterate over each row in the CSV file
        for row_index, row in enumerate(reader):
            # Check if this is the row we want
            if row_index == node_idx:  # Assuming 0-based indexing, row 2 would be index 1
                # Retrieve the cell value in column 3 (assuming 1-based indexing)
                cell_value = row[child_idx]  # Assuming 0-based indexing, column 3 would be index 2
                return float(cell_value)
                break


def getStateNames():
    with open('driving.csv') as f:
        reader = csv.reader(f, delimiter=',')
        list_of_column_names = []
        for row in reader:
            list_of_column_names = [row]
            break

        return list_of_column_names[0]


def greedy_best_first_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start))  # Add the start node to the priority queue with priority 0
    came_from = {start: None}  # Keep track of the path from the start node to each node in the graph
    cost_so_far = {start: 0}  # Keep track of the cost to reach each node from the start node
    expanded = 0
    start = time.time()
    while not frontier.empty():
        expanded += 1
        current_cost, current_node = frontier.get()  # Get the node with the lowest priority (i.e., the node with the lowest estimated cost)
        if current_node == goal:  # If we've reached the goal node, return the path to it
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()

            end = time.time()
            print("\nGreedy Best First:")
            print_results(path, calculateTotalDistance(path), end - start, expanded)
            return path

        for next_node, weight in graph[current_node].items():  # Explore the neighbors of the current node
            new_cost = cost_so_far[current_node] + weight
            if next_node not in cost_so_far or new_cost < cost_so_far[
                next_node]:  # If a better path to this node is found, update the priority queue and the cost_so_far dictionary
                priority = weight  # Use the weight of the edge as the heuristic (i.e., the estimated cost)
                came_from[next_node] = current_node
                frontier.put((priority, next_node))
                cost_so_far[next_node] = new_cost

    return None  # If we've exhausted all nodes without finding the goal, return None


def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start))  # Add the start node to the priority queue with priority 0
    came_from = {start: None}  # Keep track of the path from the start node to each node in the graph
    cost_so_far = {start: 0}  # Keep track of the cost to reach each node from the start node
    start = time.time()
    expanded = 0
    while not frontier.empty():
        current_cost, current_node = frontier.get()  # Get the node with the lowest priority (i.e., the node with the lowest estimated cost)
        expanded += 1
        if current_node == goal:  # If we've reached the goal node, return the path to it
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()

            end = time.time()
            print("\nA* search:")
            print_results(path, calculateTotalDistance(path), end - start, expanded)
            return path

        for next_node, weight in graph[current_node].items():  # Explore the neighbors of the current node
            heuristic = heurisitic_func(came_from, current_node, next_node, goal)
            new_cost = heuristic
            if next_node not in cost_so_far or new_cost < cost_so_far[
                next_node]:  # If a better path to this node is found, update the priority queue and the cost_so_far dictionary
                priority = heuristic  # Use the weight of the edge as the heuristic (i.e., the estimated cost)
                came_from[next_node] = current_node
                frontier.put((priority, next_node))
                cost_so_far[next_node] = new_cost

    return None  # If we've exhausted all nodes without finding the goal, return None


def heurisitic_func(curr_path, current_node, next_node, goal):
    heuristic = getStraightLine(getStateNames().index(next_node), getStateNames().index(goal))

    path = []
    while current_node is not None:
        path.append(current_node)
        current_node = curr_path[current_node]
    path.reverse()

    for i in range(len(path)):
        if i == len(path) - 1:
            heuristic += getDriving(getStateNames().index(path[i]), getStateNames().index(next_node))
        else:
            heuristic += getDriving(getStateNames().index(path[i]), getStateNames().index(path[i + 1]))

    return heuristic


def calculateTotalDistance(path):
    totalDistance = 0
    for i in range(len(path) - 1):
        curr_idx = getStateNames().index(path[i])
        next_idx = getStateNames().index(path[i + 1])
        totalDistance += getDriving(curr_idx, next_idx)

    return totalDistance


def loadGraph(goal):
    # Read the CSV file
    with open('driving.csv', 'r') as f:
        reader = csv.reader(f)
        row_idx = 1;
        # Skip the header row if there is one
        next(reader, None)
        # Loop over each row in the CSV file
        for row in reader:
            # Get the node label from the first column
            node = row[0]
            # Initialize an empty dictionary to hold the edges for this node
            edges = {}
            # Loop over the remaining columns in the row, which contain the weights
            for i in range(1, len(row)):
                # If the weight is non-zero, add an edge to the graph
                if float(row[i]) > 0:
                    # Get the weight from the CSV file
                    weight = getStraightLine(i, getStateNames().index(goal))
                    # print(goal + " to " + str(getStateNames()[row_idx]) + " " + str(weight))
                    # Add the edge to the dictionary
                    edges[getStateNames()[i]] = weight
            # Add the edges for this node to the graph dictionary
            graph[node] = edges
            row_idx += 1

    return (graph)


def print_results(path, totalDistance, time, expanded):
    print("\nSolution path: ", path,
          "\nNumber of states on a path: ", len(path),
          "\nNumber of Expanded nodes: ", expanded,
          "\nPath cost: ", totalDistance, " miles",
          "\nExecution Time: ", str(time), " seconds \n")


start = sys.argv[1]
goal = sys.argv[2]

if (start in getStateNames()) and (goal in getStateNames()):
    solutionGraph = loadGraph(goal)
    print("Beluonwu, Pearl Chidera, A20444304 solution:")
    print("Initial state: ", start, "\nGoal state: ", goal)

    greedy = greedy_best_first_search(solutionGraph, start, goal)
    a_star = a_star_search(solutionGraph, start, goal)
else:
    print("Solution path: FAILURE: NO PATH FOUND",
          "\nNumber of states on a path: 0",
          "\nPath cost: 0",
          "\nExecution time: 0")

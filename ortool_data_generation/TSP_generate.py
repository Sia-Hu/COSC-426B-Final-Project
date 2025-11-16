import pandas as pd
import numpy as np
import random
import string
import argparse
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_distance_matrix(xs, ys):
    """Create distance matrix from coordinates."""
    n = len(xs)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                d = int(round(np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)))
                row.append(d)
        matrix.append(row)
    return matrix

def solve_tsp_ortools(distance_matrix):
    """Solve TSP using Google OR-Tools."""
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # Add the return to start
        
        total_distance = solution.ObjectiveValue()
        return route, total_distance
    else:
        return None, None

def compute_distances_dict(xs, ys):
    """Compute all pairwise distances and return as dictionary (symmetric, no duplicates)."""
    labels = list(string.ascii_uppercase[:len(xs)])
    distances = {}
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):  # Only compute upper triangle
            d = int(round(np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)))
            distances[f"{labels[i]}{labels[j]}"] = d
    return labels, distances

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate TSP dataset with specified number of nodes')
parser.add_argument('--num_cities', type=int, default=4, 
                    help='Number of cities/nodes in TSP (default: 4)')
parser.add_argument('--num_instances', type=int, default=1000,
                    help='Total number of instances to generate (default: 1000)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Generate random TSP instances
np.random.seed(args.seed)
random.seed(args.seed)

num_instances = args.num_instances
num_cities = args.num_cities

print(f"Generating {num_instances} TSP instances with {num_cities} cities...")

# Validate num_cities
if num_cities > 26:
    raise ValueError("Number of cities cannot exceed 26 (limited by alphabet)")
if num_cities < 3:
    raise ValueError("Number of cities must be at least 3")

# In-context prompt template
prompt_template = """You will visit {n} cities named A to {last_city}. You must visit each city once and return to the starting city A at the end. I will provide you with the distance between each pair of cities. Note that the distance from city X to city Y is the same as from Y to X. Your task is to find the visiting order for the stations that minimizes the total distance you will travel to finish the journey. The answer format should be connecting the cities in order with -.

Distances: {distances}

Optimal Route: {route}"""

# Generate instances
instances = []
for i in range(num_instances):
    # Generate random coordinates for cities (0-100 range)
    xs = np.random.uniform(0, 100, num_cities).tolist()
    ys = np.random.uniform(0, 100, num_cities).tolist()
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(xs, ys)
    
    # Solve with OR-Tools
    route_indices, total_distance = solve_tsp_ortools(distance_matrix)
    
    if route_indices is not None:
        instances.append({
            'xs': xs,
            'ys': ys,
            'route_indices': route_indices,
            'total_distance': total_distance
        })

print(f"Generated {len(instances)} valid TSP instances")

# Split data: 80% train, 10% val, 10% test
train_size = int(0.8 * len(instances))
val_size = int(0.1 * len(instances))
test_size = len(instances) - train_size - val_size

train_instances = instances[:train_size]
val_instances = instances[train_size:train_size + val_size]
test_instances = instances[train_size + val_size:]

print(f"Train: {len(train_instances)}, Val: {len(val_instances)}, Test: {len(test_instances)}")

# Create training set
train_texts = []
for inst in train_instances:
    labels, distances = compute_distances_dict(inst['xs'], inst['ys'])
    route_letters = [labels[idx] for idx in inst['route_indices']]
    route_str = "-".join(route_letters)
    
    dist_str = ", ".join([f"{k} is {v}" for k, v in distances.items()])
    last_city = labels[-1]
    
    text = prompt_template.format(
        n=num_cities,
        last_city=last_city,
        distances=dist_str,
        route=route_str
    )
    train_texts.append(text)

train_df = pd.DataFrame(train_texts, columns=["text"])
train_df.to_csv(f"TSP_{num_cities}nodes_train.tsv", sep="\t", index=False)
print(f"Saved training set: {len(train_texts)} instances")

# Create validation set
val_texts = []
for inst in val_instances:
    labels, distances = compute_distances_dict(inst['xs'], inst['ys'])
    route_letters = [labels[idx] for idx in inst['route_indices']]
    route_str = "-".join(route_letters)
    
    dist_str = ", ".join([f"{k} is {v}" for k, v in distances.items()])
    last_city = labels[-1]
    
    text = prompt_template.format(
        n=num_cities,
        last_city=last_city,
        distances=dist_str,
        route=route_str
    )
    val_texts.append(text)

val_df = pd.DataFrame(val_texts, columns=["text"])
val_df.to_csv(f"TSP_{num_cities}nodes_val.tsv", sep="\t", index=False)
print(f"Saved validation set: {len(val_texts)} instances")

# Create test set with expected/unexpected pairs
test_rows = []
sentid = 1
pairid = 1

for inst in test_instances:
    labels, distances = compute_distances_dict(inst['xs'], inst['ys'])
    route_letters = [labels[idx] for idx in inst['route_indices']]
    route_str = "-".join(route_letters)
    
    dist_str = ", ".join([f"{k} is {v}" for k, v in distances.items()])
    last_city = labels[-1]
    
    # Expected text
    expected_text = prompt_template.format(
        n=num_cities,
        last_city=last_city,
        distances=dist_str,
        route=route_str
    )
    
    # Unexpected text - keep A at start and end, shuffle middle
    if route_letters[0] == 'A' and route_letters[-1] == 'A':
        middle_cities = route_letters[1:-1]
        shuffled_middle = middle_cities.copy()
        random.shuffle(shuffled_middle)
        # Ensure it's different
        while shuffled_middle == middle_cities and len(middle_cities) > 1:
            random.shuffle(shuffled_middle)
        wrong_route = ['A'] + shuffled_middle + ['A']
    else:
        # Fallback: just swap two cities
        wrong_route = route_letters.copy()
        if len(route_letters) > 2:
            i, j = random.sample(range(len(route_letters)-1), 2)
            wrong_route[i], wrong_route[j] = wrong_route[j], wrong_route[i]
    
    wrong_route_str = "-".join(wrong_route)
    unexpected_text = prompt_template.format(
        n=num_cities,
        last_city=last_city,
        distances=dist_str,
        route=wrong_route_str
    )
    
    # Add expected
    test_rows.append([sentid, pairid, "expected", expected_text])
    sentid += 1
    
    # Add unexpected
    test_rows.append([sentid, pairid, "unexpected", unexpected_text])
    sentid += 1
    pairid += 1

test_df = pd.DataFrame(test_rows, columns=["sentid", "pairid", "comparison", "sentence"])
test_df.to_csv(f"TSP_{num_cities}nodes_test.tsv", sep="\t", index=False)
print(f"Saved test set: {len(test_rows)} rows ({pairid-1} pairs)")

# Create subset with first 2 pairs
subset_df = test_df[test_df["pairid"] <= 2].copy()
subset_df.to_csv(f"TSP_{num_cities}nodes_test_subset.tsv", sep="\t", index=False)
print(f"Saved subset test set: {len(subset_df)} rows (2 pairs)")

print("\nSample from training set:")
print(train_texts[0][:300] + "...")
print(f"\nDataset generation complete for {num_cities}-node TSP!")
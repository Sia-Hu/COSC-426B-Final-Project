import pandas as pd
import numpy as np
import random
import string
import argparse
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# -------------------------------------------------------
# Distance Matrix
# -------------------------------------------------------
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

# -------------------------------------------------------
# Solve TSP using OR-Tools
# -------------------------------------------------------
def solve_tsp_ortools(distance_matrix):
    """Solve TSP using Google OR-Tools."""
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        return route, solution.ObjectiveValue()

    return None, None

# -------------------------------------------------------
# Compute Distances for Dictionary Used in Prompt
# -------------------------------------------------------
def compute_distances_dict(xs, ys):
    labels = list(string.ascii_uppercase[:len(xs)])
    distances = {}
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):  
            d = int(round(np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)))
            distances[f"{labels[i]}{labels[j]}"] = d
    return labels, distances

# -------------------------------------------------------
# Argument Parser
# -------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate TSP dataset with minimal pairs')
parser.add_argument('--num_cities', type=int, default=4,
                    help='Number of cities/nodes in TSP (default: 4)')
parser.add_argument('--num_instances', type=int, default=10000,
                    help='Total number of instances to generate (default: 1000)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

num_instances = args.num_instances
num_cities = args.num_cities

print(f"Generating {num_instances} TSP instances with {num_cities} cities...")

if num_cities > 26:
    raise ValueError("Number of cities cannot exceed 26.")
if num_cities < 3:
    raise ValueError("Number of cities must be at least 3.")

# -------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------
prompt_template = """You are a math expert. Below is an instruction that describes a task. Write a response that appropriately completes the request by reasoning step by step.

Instruction: You will visit {n} cities named A to {last_city}. You must visit each city once and return to the starting city A at the end. I will provide you with the distance between each pair of cities. Note that the distance from city X to city Y is the same as from Y to X.

To help you think through the problem, follow these reasoning steps:
1. Review the Cities: Note the set of cities A through {last_city}.
2. Review the Distances: Look at the provided distances between each pair of cities. These distances determine which connections might be shorter or longer.
3. Plan a Route: Consider a route that visits each city exactly once and returns to A. Think through the ordering step by step.
4. Validate the Route: Ensure the route is a full cycle, visits all cities exactly once, and starts and ends at A.

Your task is to think through these steps and then provide the final route that minimizes the total travel distance. 
The answer format should be connecting the cities in order with -. 

Distances: {distances}.

Response: {route}."""

# -------------------------------------------------------
# Generate Instances
# -------------------------------------------------------
instances = []
for _ in range(num_instances):
    xs = np.random.uniform(0, 100, num_cities).tolist()
    ys = np.random.uniform(0, 100, num_cities).tolist()

    distance_matrix = create_distance_matrix(xs, ys)
    route_indices, total_distance = solve_tsp_ortools(distance_matrix)

    if route_indices is not None:
        instances.append({
            'xs': xs,
            'ys': ys,
            'route_indices': route_indices,
            'total_distance': total_distance
        })

print(f"Generated {len(instances)} valid instances.")

# -------------------------------------------------------
# Train/Val/Test Split
# -------------------------------------------------------
train_size = int(0.8 * len(instances))
val_size = int(0.1 * len(instances))
test_size = len(instances) - train_size - val_size

train_instances = instances[:train_size]
val_instances = instances[train_size:train_size + val_size]
test_instances = instances[train_size + val_size:]

print(f"Train: {len(train_instances)}, Val: {len(val_instances)}, Test: {len(test_instances)}")

# -------------------------------------------------------
# Build Training Set
# -------------------------------------------------------
train_texts = []
for inst in train_instances:
    labels, distances = compute_distances_dict(inst['xs'], inst['ys'])
    route_letters = [labels[idx] for idx in inst['route_indices']]
    route_str = "-".join(route_letters)
    dist_str = ", ".join([f"{k} is {v}" for k, v in distances.items()])

    text = prompt_template.format(
        n=num_cities,
        last_city=labels[-1],
        distances=dist_str,
        route=route_str
    )
    train_texts.append(text)

# train_df = pd.DataFrame(train_texts, columns=["text"])
# train_df.to_csv(f"TSP_{num_cities}nodes_train.tsv", sep="\t", index=False)
# print(f"Saved training set.")

# -------------------------------------------------------
# Build Validation Set
# -------------------------------------------------------
val_texts = []
for inst in val_instances:
    labels, distances = compute_distances_dict(inst['xs'], inst['ys'])
    route_letters = [labels[idx] for idx in inst['route_indices']]
    route_str = "-".join(route_letters)
    dist_str = ", ".join([f"{k} is {v}" for k, v in distances.items()])

    text = prompt_template.format(
        n=num_cities,
        last_city=labels[-1],
        distances=dist_str,
        route=route_str
    )
    val_texts.append(text)

# val_df = pd.DataFrame(val_texts, columns=["text"])
# val_df.to_csv(f"TSP_{num_cities}nodes_val.tsv", sep="\t", index=False)
# print(f"Saved validation set.")

# -------------------------------------------------------
# Build Test Set with 3 Minimal Pair Types + NEW nearest-city type
# -------------------------------------------------------
test_rows = []
pairid = 1
sentid = 1

for inst in test_instances:
    labels, distances = compute_distances_dict(inst['xs'], inst['ys'])
    dist_str = ", ".join([f"{k} is {v}" for k, v in distances.items()])
    route_letters = [labels[idx] for idx in inst['route_indices']]
    route_str = "-".join(route_letters)

    # Recompute distance matrix for nearest-city computation
    distance_matrix = create_distance_matrix(inst['xs'], inst['ys'])  # NEW

    expected_text_standard = prompt_template.format(
        n=num_cities,
        last_city=labels[-1],
        distances=dist_str,
        route=route_str
    )

    expected_text_invalid = prompt_template.format(
        n=num_cities,
        last_city=labels[-1],
        distances=dist_str,
        route=route_str
    )

    # --------------------------------------
    # RANDOM WRONG ROUTE
    # --------------------------------------
    if route_letters[0] == 'A' and route_letters[-1] == 'A':
        middle = route_letters[1:-1]
        shuffled = middle.copy()
        random.shuffle(shuffled)
        while shuffled == middle and len(middle) > 1:
            random.shuffle(shuffled)
        wrong_random = ['A'] + shuffled + ['A']
    else:
        wrong_random = route_letters.copy()
        if len(wrong_random) > 2:
            i, j = random.sample(range(len(wrong_random)), 2)
            wrong_random[i], wrong_random[j] = wrong_random[j], wrong_random[i]

    unexpected_random = prompt_template.format(
        n=num_cities,
        last_city=labels[-1],
        distances=dist_str,
        route="-".join(wrong_random)
    )

    # --------------------------------------
    # SUBOPTIMAL WRONG ROUTE
    # --------------------------------------
    wrong_suboptimal = route_letters.copy()
    if len(wrong_suboptimal) > 3:
        i, j = random.sample(range(1, len(wrong_suboptimal)-1), 2)
        wrong_suboptimal[i], wrong_suboptimal[j] = wrong_suboptimal[j], wrong_suboptimal[i]

    unexpected_suboptimal = prompt_template.format(
        n=num_cities,
        last_city=labels[-1],
        distances=dist_str,
        route="-".join(wrong_suboptimal)
    )

    # --------------------------------------
    # INVALID WRONG ROUTE
    # --------------------------------------
    invalid_type = random.choice(["duplicate", "missing", "no_return", "duplicate_replace"])

    if invalid_type == "duplicate":
        # Insert a duplicated station → length increases
        wrong_invalid = route_letters.copy()
        idx = random.randint(1, len(labels) - 1)
        wrong_invalid.insert(1, labels[idx])

    elif invalid_type == "missing" and len(route_letters) > 3:
        # Remove one city → length decreases
        wrong_invalid = route_letters.copy()
        wrong_invalid.pop(2)

    elif invalid_type == "no_return":
        # Remove last city (A-return) → length decreases
        wrong_invalid = route_letters[:-1]

    elif invalid_type == "duplicate_replace":
        # NEW: same length, one duplicate, one removed
        wrong_invalid = route_letters.copy()

        # Choose a station to duplicate (not A)
        dup_idx = random.randint(1, len(labels) - 1)
        dup_city = labels[dup_idx]

        # Choose a station to remove (cannot be the same as duplicate)
        removable = [c for c in wrong_invalid if c != dup_city]
        remove_city = random.choice(removable)

        # Remove one city
        wrong_invalid.remove(remove_city)

        # Insert duplicated city at a random position (not first, to avoid trivial A)
        insert_pos = random.randint(1, len(wrong_invalid))
        wrong_invalid.insert(insert_pos, dup_city)

    unexpected_invalid = prompt_template.format(
        n=num_cities,
        last_city=labels[-1],
        distances=dist_str,
        route="-".join(wrong_invalid)
    )

    # ------------------------------------------------
    # Add minimal pairs — with INVALID using template_invalid for expected
    # ------------------------------------------------
    # RANDOM PAIR
    test_rows.append([sentid, pairid, "expected", expected_text_standard])
    test_rows.append([sentid + 1, pairid, "unexpected", unexpected_random])
    sentid += 2
    pairid += 1

    # SUBOPTIMAL PAIR
    test_rows.append([sentid, pairid, "expected", expected_text_standard])
    test_rows.append([sentid + 1, pairid, "unexpected", unexpected_suboptimal])
    sentid += 2
    pairid += 1

    # INVALID PAIR
    test_rows.append([sentid, pairid, "expected", expected_text_invalid])
    test_rows.append([sentid + 1, pairid, "unexpected", unexpected_invalid])
    sentid += 2
    pairid += 1


# -------------------------------------------------------
# Save test sets
# -------------------------------------------------------
test_df = pd.DataFrame(test_rows, columns=["sentid", "pairid", "comparison", "sentence"])
test_df.to_csv(f"TSP_{num_cities}nodes_test.tsv", sep="\t", index=False)
print("Saved combined test set.")

# -------------------------------------------------------
# Save test sets separately with independent numbering
# -------------------------------------------------------

def save_test_file(df_subset, filename):
    n_pairs = len(df_subset) // 2
    sentid = 1
    pairid = 1
    rows = []
    for i in range(0, len(df_subset), 2):
        expected_row = df_subset.iloc[i]
        unexpected_row = df_subset.iloc[i+1]

        rows.append([sentid, pairid, "expected", expected_row['sentence']])
        rows.append([sentid + 1, pairid, "unexpected", unexpected_row['sentence']])
        sentid += 2
        pairid += 1

    df_new = pd.DataFrame(rows, columns=["sentid", "pairid", "comparison", "sentence"])
    df_new.to_csv(filename, sep="\t", index=False)
    print(f"Saved {filename} with {pairid-1} pairs and {sentid-1} sentences.")

# Extract sets by pair type
random_pairs = test_df[test_df["pairid"] % 3 == 1].reset_index(drop=True)
suboptimal_pairs = test_df[test_df["pairid"] % 3 == 2].reset_index(drop=True)
invalid_pairs = test_df[test_df["pairid"] % 3 == 0].reset_index(drop=True)

# Save each with independent pairid and sentid
save_test_file(random_pairs, f"TSP_{num_cities}nodes_test_random_COT.tsv")
save_test_file(suboptimal_pairs, f"TSP_{num_cities}nodes_test_suboptimal_COT.tsv")
save_test_file(invalid_pairs, f"TSP_{num_cities}nodes_test_invalid_COT.tsv")

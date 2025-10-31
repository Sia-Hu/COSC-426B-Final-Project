# Probing Compositional Abstraction in Language Models through the Traveling Salesperson Problem
Code and dataset for Sia and Christine's final project for Fall 2025: COSC 426B Natural Language Processing.

## Repository Structure

```
COSC-426B-Final-Project/
├── Minimal Working Examples/  # For proposal
│   ├── data/
        ├── 4 nodes/ 
        ├── 15 nodes/                     
│   ├── predictions/                   
│   ├── results/                   
│   ├── config files/                   
│   └── turing submission/          
└── README.md                 # This file
```

## Data Preprocessing

This section outlines the complete pipeline for transforming raw Traveling Salesperson Problem (TSP) instances into the minimal pair `.tsv` format required by the NLPscholar framework.

### Overview

The primary goal is to convert TSP data into a textual, minimal-pair format. Each pair consists of two entries:

* **`expected`**: A text prompt describing the TSP instance (cities, distances) followed by the correct **optimal route**.
* **`unexpected`**: The *same* text prompt, but followed by an incorrect route.

---

### 1. Data Sources

We use two primary sources for generating TSP instances:

* **Primary (Small-Scale TSPs):**
    * **Description:** We generate synthetic TSP instances with a small number of cities (e.g., 4, 8, and 16).
    * **Generation:** Pairwise distances are randomly generated.
    * **Optimal Solver:** The optimal tour for each instance is computed using **Google OR-Tools**.

* **Backup (15-City TSPs):**
    * **Description:** We use the [data from Kaggle](https://www.kaggle.com/datasets/stephanhocke/15k-tsps-with-optimal-tours).
    * **Generation:** Each instance provides the 2D coordinates for 15 cities.
    * **Optimal Solver:** The dataset provides pre-computed optimal routes calculated using **Gurobi 9.0**.

---

### 2. Processing Pipeline

1.  **Instance Ingestion:** Load a raw TSP instance, which includes the data (a distance matrix) and the known optimal route.
2.  **Distance Calculation:** If the instance provides coordinates, compute the pairwise Euclidean distance matrix. 
3.  **Text Prompt Generation:** Create a standardized text prompt that describes the problem. This prompt includes:
    * The task description
    * The pairwise distances formatted as text
4.  **"Expected" Sample Creation (Optimal):**
    * Take the text prompt from Step 3
    * Append the known **optimal route**
5.  **"Unexpected" Sample Creation:**
    * Generate a suboptimal route. The current method for this is **randomly shuffling the middle nodes** of the optimal route (i.e., keeping the start/end city the same but altering the path in between)
    * Take the *same* text prompt from Step 3
    * Append this new route
---

### 3. Example:

```tsv
1	1	expected	"You will visit 4 cities named A to D. You must visit each city once and return to the starting city A at the end. I will provide you with the distance between each pair of cities. Note that the distance from city X to city Y is the same as from Y to X. Your task is to find the visiting order for the stations that minimizes the total distance you will travel to finish the journey. The answer format should be connecting the cities in order with -. Distances: AB is 80, AC is 97, AD is 30, BC is 71, BD is 56, CD is 94. Optimal Route: A-D-B-C-A"
2	1	unexpected	"You will visit 4 cities named A to D. You must visit each city once and return to the starting city A at the end. I will provide you with the distance between each pair of cities. Note that the distance from city X to city Y is the same as from Y to X. Your task is to find the visiting order for the stations that minimizes the total distance you will travel to finish the journey. The answer format should be connecting the cities in order with -. Distances: AB is 80, AC is 97, AD is 30, BC is 71, BD is 56, CD is 94. Optimal Route: A-B-D-C-A"
```
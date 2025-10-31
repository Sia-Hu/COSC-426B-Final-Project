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

We used 2 main ways to generate the dataset:

- **Small number of nodes**: We generate smaller TSP instances using random distances and optimal tours from Google OR-Tools. We also perform text transformations into the same minimal pair format with the sense of in context learning.
- **15 nodes**: Raw data could be found at https://www.kaggle.com/datasets/stephanhocke/15k-tsps-with-optimal-tours where each instance includes 15 nodes with coordinates and an optimal route computed using Gurobi 9.0. 
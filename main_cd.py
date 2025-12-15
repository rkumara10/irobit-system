#!/usr/bin/env python3
"""
Irobit Engine — Level 5 (Utility Version, No psutil)
- 5,000 irobits
- Continuous operation
- Adaptive weights
- Performs lightweight tasks to benefit the laptop
"""

import time
import random
import numpy as np

# ===== Configuration =====
N_IROBITS = 5000
OUT_DEGREE = 10
SEED = 42
INJECT_COUNT = 50
INJECT_VALUE = 1.0
POLARITY_RATIO = 0.5
WEIGHT_RANGE = 1.0
LOG_INTERVAL = 50
SLEEP_TIME = 0.001
LEARNING_RATE = 0.001
# ==========================

random.seed(SEED)
np.random.seed(SEED)

# ===== Build Graph =====
def build_graph(n_nodes, out_degree, weight_range, polarity_ratio):
    n = n_nodes
    m = n * out_degree
    sources = np.empty(m, dtype=np.int32)
    targets = np.empty(m, dtype=np.int32)
    weights = np.empty(m, dtype=np.float32)
    
    polarity = np.ones(n, dtype=np.float32)
    neg_count = int((1.0 - polarity_ratio) * n)
    if neg_count > 0:
        neg_idx = np.random.choice(n, neg_count, replace=False)
        polarity[neg_idx] = -1.0

    idx = 0
    for src in range(n):
        choices = list(range(n))
        if out_degree < n:
            choices.pop(src)
            tars = random.sample(choices, out_degree)
        else:
            tars = [random.randrange(n) for _ in range(out_degree)]
        for tar in tars:
            sources[idx] = src
            targets[idx] = tar
            weights[idx] = random.uniform(-weight_range, weight_range)
            idx += 1

    return np.zeros(n, dtype=np.float32), sources, targets, weights, polarity

# ===== Step Function =====
def step(states, sources, targets, weights, polarity):
    inputs = np.zeros(states.shape[0], dtype=np.float32)
    np.add.at(inputs, targets, weights * states[sources])
    inputs *= polarity
    return np.tanh(inputs)

# ===== Inject Function =====
def inject(states, indices, value):
    states[indices] = value

# ===== Useful Task Function =====
def laptop_task(states):
    """
    Task that uses irobits to compute something useful:
    - Sum random arrays weighted by irobits
    """
    arr = np.random.rand(1000)  # small array
    weighted_sum = np.dot(states[:1000], arr[:1000]) / 1000
    return weighted_sum

# ===== Online Learning =====
def update_weights(states, sources, targets, weights, error, lr):
    np.add.at(weights, np.arange(weights.size), lr * error * states[sources])

# ===== Summarize =====
def summarize(step_num, states):
    top_idx = np.argsort(-np.abs(states))[:8]
    top = [(int(i), float(states[i])) for i in top_idx]
    weighted_sum = laptop_task(states)
    print(f"Step {step_num}: top irobits={top}; task_value={weighted_sum:.4f}")

# ===== Main Loop =====
def main():
    print("Irobit Engine — Level 5 (Laptop Utility Mode, No psutil)")
    states, sources, targets, weights, polarity = build_graph(
        N_IROBITS, OUT_DEGREE, WEIGHT_RANGE, POLARITY_RATIO
    )
    inject_indices = np.random.choice(N_IROBITS, INJECT_COUNT, replace=False)
    inject(states, inject_indices, INJECT_VALUE)
    print(f"Injected {INJECT_COUNT} nodes at indices {list(inject_indices[:10])}...")

    step_num = 0
    try:
        while True:
            states = step(states, sources, targets, weights, polarity)
            weighted_sum = laptop_task(states)
            error = 0.01 - weighted_sum  # minimal online learning
            update_weights(states, sources, targets, weights, error, LEARNING_RATE)

            step_num += 1
            if step_num % LOG_INTERVAL == 0:
                summarize(step_num, states)
            time.sleep(SLEEP_TIME)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

if __name__ == "__main__":
    main()

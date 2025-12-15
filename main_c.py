#!/usr/bin/env python3
"""
Irobit Engine — Useful Version
- 5,000 irobits
- 10 outgoing connections per irobit
- Synchronous updates with tanh activation
- Continuous background operation
- Uses irobits to perform a small regression task as “useful work”
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
SLEEP_TIME = 0.001  # seconds per update
# ==========================

random.seed(SEED)
np.random.seed(SEED)

# ===== Build Irobit Graph =====
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

# ===== Useful Work Function =====
# Use the irobits to compute a lightweight regression target
def useful_work(states):
    # For demo: simple regression on sum of active nodes
    X = np.linspace(0, 1, len(states))
    y_pred = np.dot(states, X) / len(states)
    return y_pred

# ===== Summarize Function =====
def summarize(step_num, states, top_k=5):
    act_count = int(np.sum(np.abs(states) > 1e-3))
    top_idx = np.argsort(-np.abs(states))[:top_k]
    top = [(int(i), float(states[i])) for i in top_idx]
    useful = useful_work(states)
    print(f"Step {step_num}: active={act_count}/{len(states)}; top={top}; useful={useful:.4f}")

# ===== Main Loop =====
def main():
    print("Irobit Engine — Useful Version (5,000 irobits)")
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
            step_num += 1
            if step_num % LOG_INTERVAL == 0:
                summarize(step_num, states, top_k=8)
            time.sleep(SLEEP_TIME)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

if __name__ == "__main__":
    main()

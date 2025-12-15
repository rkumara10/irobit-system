#!/usr/bin/env python3
"""
Irobit Engine — Version 2
- 5000 irobits
- 10 outgoing random connections per irobit
- positive / negative polarity per irobit
- synchronous update with tanh activation
- NumPy vectorized for speed, with pure-Python fallback

Run:
    python irobits_v2.py
"""

import sys
import time
import math
import random

# Try to import numpy; if not available, fallback
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

# ====== Configuration ======
N_IROBITS = 5000        # number of irobits requested
OUT_DEGREE = 10         # outgoing connections per irobit
SEED = 42               # reproducible randomness
STEPS = 200             # number of synchronous update steps
INJECT_COUNT = 50       # how many irobits to inject initial signal
INJECT_VALUE = 1.0      # initial injection amplitude
POLARITY_RATIO = 0.5    # fraction of positive irobits (rest negative)
WEIGHT_RANGE = 1.0      # weights sampled uniform in [-WEIGHT_RANGE, WEIGHT_RANGE]
# ===========================

random.seed(SEED)

def build_graph_numpy(n_nodes, out_degree, weight_range, polarity_ratio):
    """
    Build adjacency as vectors for NumPy:
    returns: states (n,), sources (m,), targets (m,), weights (m,), polarity (n,)
    """
    n = n_nodes
    m = n * out_degree
    sources = np.empty(m, dtype=np.int32)
    targets = np.empty(m, dtype=np.int32)
    weights = np.empty(m, dtype=np.float32)

    # Assign polarity +1 (positive) or -1 (negative)
    polarity = np.ones(n, dtype=np.float32)
    # sample which nodes are negative
    neg_count = int((1.0 - polarity_ratio) * n)
    if neg_count > 0:
        neg_idx = np.random.choice(n, neg_count, replace=False)
        polarity[neg_idx] = -1.0

    idx = 0
    for src in range(n):
        # pick out_degree distinct targets (allow self-target occasionally if desired; here avoid self)
        # If n is small and out_degree large, allow replacement; here n >> out_degree so sample without replacement.
        choices = list(range(n))
        if out_degree < n:
            # avoid self by removing
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

def step_numpy(states, sources, targets, weights, polarity):
    """
    One synchronous step using vectorized accumulation.
    states: (n,)
    returns new_states (n,)
    """
    n = states.shape[0]
    inputs = np.zeros(n, dtype=np.float32)

    # accumulate contributions inputs[target] += weight * states[source]
    # vectorized: inputs[targets] += weights * states[sources]
    np.add.at(inputs, targets, weights * states[sources])

    # apply polarity: positive nodes accept input, negative nodes invert input
    inputs = inputs * polarity

    # activation
    new_states = np.tanh(inputs)
    return new_states

# Pure-python fallback
def build_graph_py(n_nodes, out_degree, weight_range, polarity_ratio):
    nodes = [{'state':0.0, 'outs':[], 'polarity':1.0} for _ in range(n_nodes)]
    # set polarity
    neg_count = int((1.0 - polarity_ratio) * n_nodes)
    neg_indices = set(random.sample(range(n_nodes), neg_count)) if neg_count>0 else set()
    for i in range(n_nodes):
        nodes[i]['polarity'] = -1.0 if i in neg_indices else 1.0

    for src in range(n_nodes):
        choices = list(range(n_nodes))
        if out_degree < n_nodes:
            choices.pop(src)
            tars = random.sample(choices, out_degree)
        else:
            tars = [random.randrange(n_nodes) for _ in range(out_degree)]
        for tar in tars:
            w = random.uniform(-weight_range, weight_range)
            nodes[src]['outs'].append((tar, w))
    return nodes

def step_py(nodes):
    n = len(nodes)
    inputs = [0.0]*n
    for src_idx, node in enumerate(nodes):
        s = node['state']
        for tar, w in node['outs']:
            inputs[tar] += w * s
    # apply polarity and activation
    new_states = [0.0]*n
    for i,node in enumerate(nodes):
        inp = inputs[i] * node['polarity']
        new_states[i] = math.tanh(inp)
    # update
    for i,node in enumerate(nodes):
        node['state'] = new_states[i]
    return [node['state'] for node in nodes]

def inject_numpy(states, indices, value):
    states[indices] = value

def inject_py(nodes, indices, value):
    for i in indices:
        nodes[i]['state'] = value

def summarize(step_num, states, top_k=5):
    if HAS_NUMPY:
        n = states.size
        # count active nodes (abs(state) > threshold)
        act_count = int(np.sum(np.abs(states) > 1e-3))
        top_idx = np.argsort(-np.abs(states))[:top_k]
        top = [(int(i), float(states[i])) for i in top_idx]
        print(f"Step {step_num}: active={act_count}/{n}; top={top}")
    else:
        n = len(states)
        act_count = sum(1 for s in states if abs(s)>1e-3)
        top = sorted(list(enumerate(states)), key=lambda x: -abs(x[1]))[:top_k]
        print(f"Step {step_num}: active={act_count}/{n}; top={top}")

def main():
    print("Irobit v2 simulator — 5,000 irobits, 10 links each")
    print("NumPy available:", HAS_NUMPY)
    start_time = time.time()

    if HAS_NUMPY:
        # initialize RNG for numpy
        np.random.seed(SEED)
        states, sources, targets, weights, polarity = build_graph_numpy(
            N_IROBITS, OUT_DEGREE, WEIGHT_RANGE, POLARITY_RATIO
        )

        # initial injection: pick INJECT_COUNT random nodes and set state
        inject_indices = np.random.choice(N_IROBITS, INJECT_COUNT, replace=False)
        inject_numpy(states, inject_indices, INJECT_VALUE)
        print(f"Injected {INJECT_COUNT} nodes at indices {list(inject_indices[:10])}...")

        # run steps
        for t in range(STEPS):
            states = step_numpy(states, sources, targets, weights, polarity)
            if t % 10 == 0 or t < 5 or t == STEPS-1:
                summarize(t, states, top_k=8)

    else:
        # pure python fallback
        nodes = build_graph_py(N_IROBITS, OUT_DEGREE, WEIGHT_RANGE, POLARITY_RATIO)
        inject_indices = random.sample(range(N_IROBITS), INJECT_COUNT)
        inject_py(nodes, inject_indices, INJECT_VALUE)
        print(f"(py) Injected {INJECT_COUNT} nodes at indices {inject_indices[:10]}...")

        for t in range(STEPS):
            states = step_py(nodes)
            if t % 10 == 0 or t < 5 or t == STEPS-1:
                summarize(t, states, top_k=8)

    elapsed = time.time() - start_time
    print(f"Simulation finished in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

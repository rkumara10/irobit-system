#!/usr/bin/env python3
"""
50k iRobits simulator — NumPy vectorized, continuous
- 50,000 iRobits
- 10 outgoing connections each
- each connection stores: target, weight, user data, control flag, activity type (0=NORMAL,1=CPU,2=GPU)
- runs until Ctrl+C
"""

import numpy as np
import time
import sys

# ===== CONFIG =====
N_IROBITS = 50_000
OUT_DEGREE = 10
STEPS_PER_CYCLE = 5         # how many update iterations per "cycle" before optional logging
INJECT_COUNT = 100
INJECT_VALUE = 1.0
POLARITY_RATIO = 0.5
SEED = 42
LOG_INTERVAL = 50           # cycles
SLEEP_SEC = 0.001           # small sleep to keep CPU healthy
CPU_MULT = 1.2
GPU_MULT = 1.5
# ==================

rng = np.random.default_rng(SEED)

# compact activity encoding: 0=NORMAL,1=CPU,2=GPU
activity_choices = [0,1,2]
activity_weights = [0.85, 0.10, 0.05]

# create storage for states, polarities
states = np.zeros(N_IROBITS, dtype=np.float32)
polarity = np.ones(N_IROBITS, dtype=np.float32)
neg_count = int((1.0 - POLARITY_RATIO) * N_IROBITS)
if neg_count > 0:
    neg_idx = rng.choice(N_IROBITS, neg_count, replace=False)
    polarity[neg_idx] = -1.0

# Create connections arrays (flattened)
m = N_IROBITS * OUT_DEGREE
# targets: int64 to be safe
targets = rng.integers(0, N_IROBITS, size=m, dtype=np.int64)
weights = rng.uniform(-1.0, 1.0, size=m).astype(np.float32)
data_field = rng.uniform(-1.0, 1.0, size=m).astype(np.float32)   # user data per connection
control_flag = rng.choice(a=[0,1], size=m, p=[0.8,0.2]).astype(np.uint8)
activity = rng.choice(a=activity_choices, size=m, p=activity_weights).astype(np.uint8)

# helper to index per-source: sources are implicit 0..N_IROBITS-1 repeated OUT_DEGREE times
sources = np.repeat(np.arange(N_IROBITS, dtype=np.int64), OUT_DEGREE)

# injection
inject_idx = rng.choice(N_IROBITS, size=INJECT_COUNT, replace=False)
states[inject_idx] = INJECT_VALUE
print(f"Injected {INJECT_COUNT} iRobits (example indices): {inject_idx[:10].tolist()}")

# Precompute multipliers array for activity types
# We'll use activity_mul[activity] to scale; store as float32
activity_mul = np.array([1.0, CPU_MULT, GPU_MULT], dtype=np.float32)

cycle = 0
try:
    while True:
        # perform STEPS_PER_CYCLE synchronous steps
        for step in range(STEPS_PER_CYCLE):
            # accumulate contributions into inputs
            inputs = np.zeros(N_IROBITS, dtype=np.float32)

            # contribution per connection = states[source] * weights * data * activity_mul
            # vectorized compute: get states for each source
            src_states = states[sources]                    # shape (m,)
            contribs = src_states * weights * data_field * activity_mul[activity]

            # scatter add into inputs at target indices
            np.add.at(inputs, targets, contribs)

            # apply polarity and activation
            states = np.tanh(inputs * polarity)

        cycle += 1
        if cycle % LOG_INTERVAL == 0:
            # report top active iRobits
            top_k = 8
            idx = np.argpartition(-np.abs(states), top_k)[:top_k]
            idx = idx[np.argsort(-np.abs(states[idx]))]
            top = [(int(i), float(states[i])) for i in idx]
            active_count = int(np.sum(np.abs(states) > 1e-3))
            print(f"[Cycle {cycle}] active={active_count}/{N_IROBITS}; top={top}")

        # small sleep to avoid pegging CPU if you want
        time.sleep(SLEEP_SEC)

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
    sys.exit(0)

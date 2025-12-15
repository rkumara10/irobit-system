#!/usr/bin/env python3
"""
WCompute Irobit v2.7 Simulator — 500,000 nodes
Fixed propagation bug and safe indexing
"""

import numpy as np
import time

# =======================
# CONFIGURATION
# =======================
NUM_IROBITS = 500_000
CONNECTIONS_PER_IRO = 10
STEPS = 200
DECAY = 0.96
PROP_GAIN = 0.35
NOISE = 0.002
SEED_COUNT = 2000
SEED_LOW, SEED_HIGH = 0.6, 1.0
RNG_SEED = 12345

# =======================
# INITIALIZATION
# =======================
rng = np.random.default_rng(RNG_SEED)

print(f"Initializing {NUM_IROBITS} irobits...")
t0 = time.time()

# activations for each irobits
activations = np.zeros(NUM_IROBITS, dtype=np.float32)

# each irobits connects to CONNECTIONS_PER_IRO other irobits
targets = rng.integers(0, NUM_IROBITS, size=(NUM_IROBITS, CONNECTIONS_PER_IRO), dtype=np.int32)

# safety check
assert np.all(targets >= 0) and np.all(targets < NUM_IROBITS), "Target indices out of range!"

# seed nodes (random initial activations)
seed_nodes = rng.choice(NUM_IROBITS, size=SEED_COUNT, replace=False)
activations[seed_nodes] = rng.uniform(SEED_LOW, SEED_HIGH, size=SEED_COUNT).astype(np.float32)
print(f"Injected {SEED_COUNT} seed nodes (example indices): {seed_nodes[:10].tolist()}")

# buffer for new activations
new_values = np.zeros(NUM_IROBITS, dtype=np.float32)

# =======================
# SIMULATION LOOP
# =======================
for step in range(STEPS):
    new_values.fill(0.0)

    # propagate activations
    contribs = activations[:, None] * PROP_GAIN           # shape: (NUM_IROBITS, CONNECTIONS_PER_IRO)
    flat_targets = targets.flatten().astype(np.intp)     # safe integer type for np.add.at
    flat_contribs = contribs.flatten().astype(np.float32)
    np.add.at(new_values, flat_targets, flat_contribs)

    # decay + noise
    noise = rng.uniform(-NOISE, NOISE, size=NUM_IROBITS).astype(np.float32)
    activations = new_values * DECAY + noise
    np.clip(activations, -1.0, 1.0, out=activations)

    # diagnostics
    if step % 10 == 0 or step < 5 or step == STEPS - 1:
        active_count = int(np.sum(np.abs(activations) > 0.01))
        top_k = 8
        top_idx = np.argpartition(-np.abs(activations), top_k)[:top_k]
        top_sorted = top_idx[np.argsort(-np.abs(activations[top_idx]))]
        top = [(int(i), float(activations[i])) for i in top_sorted]
        print(f"Step {step}: active={active_count}/{NUM_IROBITS}; top={top}")

t1 = time.time()
print(f"Simulation finished in {t1 - t0:.2f} seconds")

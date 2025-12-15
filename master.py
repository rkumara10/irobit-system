import numpy as np
import random
import time

# ===== Configuration =====
N_IROBITS = 10**12   # 1 trillion iRobits
OUT_DEGREE = 10
CHUNK_SIZE = 10**6       # simulate in chunks
POLARITY_RATIO = 0.5
STEPS = 10               # steps per chunk update
ACTIVITY_TYPES = ['NORMAL', 'CPU', 'GPU']
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Connection dtype
dtype_connection = np.dtype([
    ('target', np.int64),
    ('weight', np.float32),
    ('data', np.float32),    # now stores user data for all
    ('control', np.bool_),
    ('activity', 'U4')       # 'CPU', 'GPU', 'NORMAL'
])

# ===== Build Chunk =====
def build_chunk(chunk_size):
    n = chunk_size
    connections = np.zeros((n, OUT_DEGREE), dtype=dtype_connection)
    polarity = np.ones(n, dtype=np.float32)

    neg_count = int((1.0-POLARITY_RATIO)*n)
    if neg_count>0:
        neg_indices = np.random.choice(n, neg_count, replace=False)
        polarity[neg_indices] = -1.0

    for i in range(n):
        for j in range(OUT_DEGREE):
            target = random.randint(0, N_IROBITS-1)
            weight = random.uniform(-1.0, 1.0)
            # User-provided data stored in all connections
            data = random.uniform(-1.0, 1.0)  # can replace with user input
            control = random.choice([True, False])
            activity = random.choices(ACTIVITY_TYPES, weights=[0.85,0.1,0.05])[0]
            connections[i,j] = (target, weight, data, control, activity)
    return connections, polarity, np.zeros(n, dtype=np.float32)

# ===== Step Simulation =====
def step_chunk(states, connections, polarity):
    n = states.size
    inputs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = states[i]
        for j in range(connections.shape[1]):
            c = connections[i,j]
            # Use stored data in computation
            factor = c.data  # factor based on stored user data
            if c.activity == 'CPU':
                inputs[i] += s * c.weight * 1.2 * factor
            elif c.activity == 'GPU':
                inputs[i] += s * c.weight * 1.5 * factor
            else:
                inputs[i] += s * c.weight * factor
    new_states = np.tanh(inputs * polarity)
    return new_states

# ===== Main Loop =====
def main():
    chunk_idx = 0
    while True:
        print(f"Initializing chunk {chunk_idx}")
        conns, pol, states = build_chunk(CHUNK_SIZE)

        # Inject initial values
        inject_indices = np.random.choice(CHUNK_SIZE, min(50, CHUNK_SIZE), replace=False)
        states[inject_indices] = 1.0

        for step in range(STEPS):
            states = step_chunk(states, conns, pol)
            if step % 2 == 0:
                top_idx = np.argsort(-np.abs(states))[:5]
                print(f"Chunk {chunk_idx} Step {step} top: {[(i, states[i]) for i in top_idx]}")
        
        chunk_idx += 1

if __name__ == "__main__":
    main()

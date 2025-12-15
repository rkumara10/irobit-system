/* iro_50k.c
   50k iRobits simulator — flat arrays, continuous loop
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define N_IROBITS 50000
#define OUT_DEGREE 10
#define M (N_IROBITS * OUT_DEGREE)
#define STEPS_PER_CYCLE 5
#define INJECT_COUNT 100
#define INJECT_VALUE 1.0f
#define POLARITY_RATIO 0.5f
#define LOG_INTERVAL 50
#define CPU_MULT 1.2f
#define GPU_MULT 1.5f
#define SLEEP_MS 1

/* utility: sleep in ms */
#ifdef _WIN32
#include <windows.h>
static void msleep(int ms) { Sleep(ms); }
#else
#include <unistd.h>
static void msleep(int ms) { usleep(ms * 1000); }
#endif

/* fast tanh approximation: use standard tanhf */
static inline float tanh_f(float x) { return tanhf(x); }

int main(void) {
    srand((unsigned)time(NULL));

    /* states and polarity */
    float *states = (float*)calloc(N_IROBITS, sizeof(float));
    float *polarity = (float*)malloc(N_IROBITS * sizeof(float));
    if (!states || !polarity) { fprintf(stderr,"alloc fail\n"); return 1; }

    /* connections: flat arrays length M */
    int64_t *targets = (int64_t*)malloc(M * sizeof(int64_t));
    float *weights = (float*)malloc(M * sizeof(float));
    float *data_field = (float*)malloc(M * sizeof(float));
    unsigned char *control_flag = (unsigned char*)malloc(M * sizeof(unsigned char));
    unsigned char *activity = (unsigned char*)malloc(M * sizeof(unsigned char)); /* 0=NORMAL,1=CPU,2=GPU */
    int64_t *sources = (int64_t*)malloc(M * sizeof(int64_t)); /* repeated source indices */

    if (!targets || !weights || !data_field || !control_flag || !activity || !sources) {
        fprintf(stderr,"connection alloc fail\n"); return 1;
    }

    /* initialize polarity */
    int neg_count = (int)((1.0f - POLARITY_RATIO) * N_IROBITS);
    for (int i = 0; i < N_IROBITS; ++i) polarity[i] = 1.0f;
    for (int i = 0; i < neg_count; ++i) {
        int idx = rand() % N_IROBITS;
        polarity[idx] = -1.0f;
    }

    /* initialize connections */
    for (int i = 0; i < N_IROBITS; ++i) {
        for (int j = 0; j < OUT_DEGREE; ++j) {
            int64_t idx = (int64_t)i * OUT_DEGREE + j;
            sources[idx] = i;
            targets[idx] = rand() % N_IROBITS;
            weights[idx] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            data_field[idx] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            control_flag[idx] = (rand() % 5) == 0 ? 1 : 0; /* ~20% control */
            int r = rand() % 100;
            if (r < 85) activity[idx] = 0;
            else if (r < 95) activity[idx] = 1;
            else activity[idx] = 2;
        }
    }

    /* initial injection */
    for (int k = 0; k < INJECT_COUNT; ++k) {
        int id = rand() % N_IROBITS;
        states[id] = INJECT_VALUE;
    }
    printf("Injected %d iRobits\n", INJECT_COUNT);

    int cycle = 0;
    float *inputs = (float*)malloc(N_IROBITS * sizeof(float));

    while (1) {
        for (int step = 0; step < STEPS_PER_CYCLE; ++step) {
            /* zero inputs */
            memset(inputs, 0, N_IROBITS * sizeof(float));

            /* vectorized-like loop over connections */
            for (int64_t e = 0; e < M; ++e) {
                int64_t src = sources[e];
                float s = states[src];
                float w = weights[e];
                float d = data_field[e];
                unsigned char act = activity[e];
                float mul = 1.0f;
                if (act == 1) mul = CPU_MULT;
                else if (act == 2) mul = GPU_MULT;
                /* contribution to target */
                int64_t tar = targets[e];
                inputs[tar] += s * w * d * mul;
            }

            /* apply polarity and activation */
            for (int i = 0; i < N_IROBITS; ++i) {
                float val = inputs[i] * polarity[i];
                states[i] = tanh_f(val);
            }
        }

        cycle++;
        if (cycle % LOG_INTERVAL == 0) {
            /* find top K active */
            int K = 8;
            /* simple selection: find top K by absolute value */
            float topvals[8]; int topidx[8];
            for (int t = 0; t < K; ++t) { topvals[t] = 0.0f; topidx[t] = -1; }
            for (int i = 0; i < N_IROBITS; ++i) {
                float a = fabsf(states[i]);
                for (int t = 0; t < K; ++t) {
                    if (a > topvals[t]) {
                        /* shift */
                        for (int s = K-1; s > t; --s) { topvals[s] = topvals[s-1]; topidx[s] = topidx[s-1]; }
                        topvals[t] = a; topidx[t] = i;
                        break;
                    }
                }
            }
            int active_count = 0;
            for (int i = 0; i < N_IROBITS; ++i) if (fabsf(states[i]) > 1e-3f) active_count++;
            printf("[cycle %d] active=%d/%d top=", cycle, active_count, N_IROBITS);
            for (int t = 0; t < K; ++t) if (topidx[t] >= 0) printf("(%d,%.4f) ", topidx[t], states[topidx[t]]);
            printf("\n");
        }

        msleep(SLEEP_MS);
    }

    /* never reached but tidy */
    free(states); free(polarity); free(targets); free(weights);
    free(data_field); free(control_flag); free(activity); free(sources); free(inputs);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define OUT_DEGREE 10
#define CHUNK_SIZE 1000000

typedef enum {NORMAL, CPU, GPU} ActivityType;

typedef struct {
    int64_t target;
    float weight;
    float data;       // stores user-provided data
    bool control;
    ActivityType activity;
} Connection;

typedef struct {
    float state;
    float polarity;
    Connection outs[OUT_DEGREE];
} Irobit;

float rand_float() {
    return ((float)rand()/RAND_MAX)*2.0f - 1.0f;
}

void build_chunk(Irobit *chunk, int chunk_size, int64_t total_irobits) {
    for(int i=0; i<chunk_size; i++) {
        chunk[i].state = 0.0f;
        chunk[i].polarity = (rand()%2)?1.0f:-1.0f;
        for(int j=0;j<OUT_DEGREE;j++){
            chunk[i].outs[j].target = rand()%total_irobits;
            chunk[i].outs[j].weight = rand_float();
            chunk[i].outs[j].data = rand_float();  // store user data
            chunk[i].outs[j].control = rand()%2;
            int r = rand()%100;
            if(r<85) chunk[i].outs[j].activity=NORMAL;
            else if(r<95) chunk[i].outs[j].activity=CPU;
            else chunk[i].outs[j].activity=GPU;
        }
    }
}

float tanh_f(float x){
    return (expf(x)-expf(-x))/(expf(x)+expf(-x));
}

void step_chunk(Irobit *chunk, int chunk_size){
    float *inputs = (float*)calloc(chunk_size,sizeof(float));
    for(int i=0;i<chunk_size;i++){
        float s = chunk[i].state;
        for(int j=0;j<OUT_DEGREE;j++){
            Connection c = chunk[i].outs[j];
            float factor = c.data; // user data affects propagation
            if(c.activity==CPU) inputs[i] += s*c.weight*1.2f*factor;
            else if(c.activity==GPU) inputs[i] += s*c.weight*1.5f*factor;
            else inputs[i] += s*c.weight*factor;
        }
    }
    for(int i=0;i<chunk_size;i++){
        chunk[i].state = tanh_f(inputs[i]*chunk[i].polarity);
    }
    free(inputs);
}

int main(){
    srand(time(NULL));
    int64_t total_irobits = 1000000000000LL;
    int chunk_idx = 0;

    while(1){
        Irobit *chunk = malloc(sizeof(Irobit)*CHUNK_SIZE);
        build_chunk(chunk, CHUNK_SIZE, total_irobits);

        for(int i=0;i<50 && i<CHUNK_SIZE;i++) chunk[i].state = 1.0f;

        for(int step=0; step<10; step++){
            step_chunk(chunk, CHUNK_SIZE);
        }

        free(chunk);
        chunk_idx++;
        printf("Processed chunk %d\n", chunk_idx);
    }
    return 0;
}

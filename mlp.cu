// mnist_resmlp.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif


typedef struct {
    double data_loading;

    double fwd_proj;
    double fwd_blocks;
    double fwd_head;
    double fwd_softmax;

    double loss_ce;

    double bwd_softmax_grad;
    double bwd_head;
    double bwd_blocks;
    double bwd_proj;

    double weight_updates;

    double total_time;
} TimingStats;

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

#define D_MODEL 512
#define N_BLOCKS 4

#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 10
#define LEARNING_RATE 0.01f

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (!file) { fprintf(stderr, "Error opening file: %s\n", filename); exit(1); }
    size_t read_size = fread(data, sizeof(float), size, file);
    if ((int)read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (!file) { fprintf(stderr, "Error opening file: %s\n", filename); exit(1); }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if ((int)read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) data[i] = (data[i] - mean) / std;
}

void initialize_weights(float *w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        float r = (float)rand() / (float)RAND_MAX;    
        w[i] = (2.0f * r - 1.0f) * scale;
    }
}

void initialize_bias(float *b, int size) {
    for (int i = 0; i < size; i++) b[i] = 0.0f;
}


__global__ void matmul_a_b_kernel(const float *A, const float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) sum += A[row * n + i] * B[i * k + col];
        C[row * k + col] = sum;
    }
}

__global__ void matmul_a_bt_kernel(const float *A, const float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) sum += A[row * n + i] * B[col * n + i];
        C[row * k + col] = sum;
    }
}

__global__ void matmul_at_b_kernel(const float *A, const float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < n && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i) sum += A[i * n + row] * B[i * k + col];
        C[row * k + col] = sum;
    }
}

__global__ void bias_forward_kernel(float *x, const float *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;
    if (b < batch_size && i < size) x[idx] += bias[i];
}

__global__ void relu_forward_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) x[idx] = fmaxf(0.0f, x[idx]);
}

__global__ void relu_backward_kernel(float *grad, const float *x_post_relu, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad[idx] *= (x_post_relu[idx] > 0.0f ? 1.0f : 0.0f);
}

__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    if (b < batch_size) {
        float max_val = x[b * size];
        for (int i = 1; i < size; i++) max_val = fmaxf(max_val, x[b * size + i]);

        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }
        for (int i = 0; i < size; i++) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

__global__ void zero_grad_kernel(float *grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad[idx] = 0.0f;
}

__global__ void compute_output_gradients_kernel(float *grad_output, const float *output, const int *labels, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        for (int i = 0; i < OUTPUT_SIZE; ++i) grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
        for (int i = 0; i < OUTPUT_SIZE; ++i) grad_output[b * OUTPUT_SIZE + i] /= (float)batch_size;
    }
}

__global__ void bias_backward_kernel(float *grad_bias, const float *grad, int batch_size, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) sum += grad[b * size + i];
        grad_bias[i] = sum;
    }
}

__global__ void weight_update_kernel(float *weights, const float *grad_weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) weights[idx] -= LEARNING_RATE * grad_weights[idx];
}

__global__ void residual_add_kernel(float *x, const float *y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) x[idx] += y[idx];
}



float cross_entropy_loss(const float *output, const int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / (float)batch_size;
}


typedef struct {
    float *W0, *b0, *dW0, *db0;

    float *W1[N_BLOCKS], *b1[N_BLOCKS], *dW1[N_BLOCKS], *db1[N_BLOCKS];
    float *W2[N_BLOCKS], *b2[N_BLOCKS], *dW2[N_BLOCKS], *db2[N_BLOCKS];

    float *Wh, *bh, *dWh, *dbh;
} ResMLP;

static void gpu_alloc(float **p, size_t bytes) { CUDA_CHECK(cudaMalloc((void**)p, bytes)); }

void initialize_resmlp(ResMLP *m) {
    gpu_alloc(&m->W0,  INPUT_SIZE * D_MODEL * sizeof(float));
    gpu_alloc(&m->b0,  D_MODEL * sizeof(float));
    gpu_alloc(&m->dW0, INPUT_SIZE * D_MODEL * sizeof(float));
    gpu_alloc(&m->db0, D_MODEL * sizeof(float));

    for (int l = 0; l < N_BLOCKS; l++) {
        gpu_alloc(&m->W1[l],  D_MODEL * D_MODEL * sizeof(float));
        gpu_alloc(&m->b1[l],  D_MODEL * sizeof(float));
        gpu_alloc(&m->dW1[l], D_MODEL * D_MODEL * sizeof(float));
        gpu_alloc(&m->db1[l], D_MODEL * sizeof(float));

        gpu_alloc(&m->W2[l],  D_MODEL * D_MODEL * sizeof(float));
        gpu_alloc(&m->b2[l],  D_MODEL * sizeof(float));
        gpu_alloc(&m->dW2[l], D_MODEL * D_MODEL * sizeof(float));
        gpu_alloc(&m->db2[l], D_MODEL * sizeof(float));
    }

    gpu_alloc(&m->Wh,  D_MODEL * OUTPUT_SIZE * sizeof(float));
    gpu_alloc(&m->bh,  OUTPUT_SIZE * sizeof(float));
    gpu_alloc(&m->dWh, D_MODEL * OUTPUT_SIZE * sizeof(float));
    gpu_alloc(&m->dbh, OUTPUT_SIZE * sizeof(float));

    float *hW0  = (float*)malloc(INPUT_SIZE * D_MODEL * sizeof(float));
    float *hb0  = (float*)malloc(D_MODEL * sizeof(float));
    float *hWh  = (float*)malloc(D_MODEL * OUTPUT_SIZE * sizeof(float));
    float *hbh  = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(hW0, INPUT_SIZE, D_MODEL);
    initialize_bias(hb0, D_MODEL);
    initialize_weights(hWh, D_MODEL, OUTPUT_SIZE);
    initialize_bias(hbh, OUTPUT_SIZE);

    CUDA_CHECK(cudaMemcpy(m->W0, hW0, INPUT_SIZE * D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m->b0, hb0, D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m->Wh, hWh, D_MODEL * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m->bh, hbh, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    free(hW0); free(hb0); free(hWh); free(hbh);

    for (int l = 0; l < N_BLOCKS; l++) {
        float *hW1 = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
        float *hb1 = (float*)malloc(D_MODEL * sizeof(float));
        float *hW2 = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
        float *hb2 = (float*)malloc(D_MODEL * sizeof(float));

        initialize_weights(hW1, D_MODEL, D_MODEL);
        initialize_bias(hb1, D_MODEL);
        initialize_weights(hW2, D_MODEL, D_MODEL);
        initialize_bias(hb2, D_MODEL);

        CUDA_CHECK(cudaMemcpy(m->W1[l], hW1, D_MODEL * D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m->b1[l], hb1, D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m->W2[l], hW2, D_MODEL * D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m->b2[l], hb2, D_MODEL * sizeof(float), cudaMemcpyHostToDevice));

        free(hW1); free(hb1); free(hW2); free(hb2);
    }
}

void free_resmlp(ResMLP *m) {
    CUDA_CHECK(cudaFree(m->W0));  CUDA_CHECK(cudaFree(m->b0));
    CUDA_CHECK(cudaFree(m->dW0)); CUDA_CHECK(cudaFree(m->db0));

    for (int l = 0; l < N_BLOCKS; l++) {
        CUDA_CHECK(cudaFree(m->W1[l]));  CUDA_CHECK(cudaFree(m->b1[l]));
        CUDA_CHECK(cudaFree(m->dW1[l])); CUDA_CHECK(cudaFree(m->db1[l]));
        CUDA_CHECK(cudaFree(m->W2[l]));  CUDA_CHECK(cudaFree(m->b2[l]));
        CUDA_CHECK(cudaFree(m->dW2[l])); CUDA_CHECK(cudaFree(m->db2[l]));
    }

    CUDA_CHECK(cudaFree(m->Wh));  CUDA_CHECK(cudaFree(m->bh));
    CUDA_CHECK(cudaFree(m->dWh)); CUDA_CHECK(cudaFree(m->dbh));
}


typedef struct {
    float *x[N_BLOCKS + 1];
    float *z1[N_BLOCKS];
    float *a1[N_BLOCKS];
    float *z2[N_BLOCKS];
    float *x_pre_relu[N_BLOCKS]; 
    float *logits;              
    float *probs;              
} Activations;

void alloc_activations(Activations *act, int B) {
    for (int i = 0; i < N_BLOCKS + 1; i++) gpu_alloc(&act->x[i], B * D_MODEL * sizeof(float));
    for (int l = 0; l < N_BLOCKS; l++) {
        gpu_alloc(&act->z1[l], B * D_MODEL * sizeof(float));
        gpu_alloc(&act->a1[l], B * D_MODEL * sizeof(float));
        gpu_alloc(&act->z2[l], B * D_MODEL * sizeof(float));
        gpu_alloc(&act->x_pre_relu[l], B * D_MODEL * sizeof(float));
    }
    gpu_alloc(&act->logits, B * OUTPUT_SIZE * sizeof(float));
    gpu_alloc(&act->probs,  B * OUTPUT_SIZE * sizeof(float));
}

void free_activations(Activations *act) {
    for (int i = 0; i < N_BLOCKS + 1; i++) CUDA_CHECK(cudaFree(act->x[i]));
    for (int l = 0; l < N_BLOCKS; l++) {
        CUDA_CHECK(cudaFree(act->z1[l]));
        CUDA_CHECK(cudaFree(act->a1[l]));
        CUDA_CHECK(cudaFree(act->z2[l]));
        CUDA_CHECK(cudaFree(act->x_pre_relu[l]));
    }
    CUDA_CHECK(cudaFree(act->logits));
    CUDA_CHECK(cudaFree(act->probs));
}

void forward_resmlp_timed(
    ResMLP *m,
    const float *d_input, 
    Activations *act,
    int B,
    TimingStats *stats
) {
    struct timespec start, end;
    dim3 block(32, 32);

    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_proj((D_MODEL + block.x - 1) / block.x, (B + block.y - 1) / block.y);
    matmul_a_b_kernel<<<grid_proj, block>>>(d_input, m->W0, act->x[0], B, INPUT_SIZE, D_MODEL);
    bias_forward_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(act->x[0], m->b0, B, D_MODEL);
    relu_forward_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(act->x[0], B * D_MODEL);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_proj += get_time_diff(start, end);

    // blocks
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int l = 0; l < N_BLOCKS; l++) {
        // z1 = x @ W1 + b1
        dim3 grid1((D_MODEL + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        matmul_a_b_kernel<<<grid1, block>>>(act->x[l], m->W1[l], act->z1[l], B, D_MODEL, D_MODEL);
        bias_forward_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(act->z1[l], m->b1[l], B, D_MODEL);

        // a1 = relu(z1) (copy z1->a1 then relu in-place)
        CUDA_CHECK(cudaMemcpy(act->a1[l], act->z1[l], B * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice));
        relu_forward_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(act->a1[l], B * D_MODEL);

        // z2 = a1 @ W2 + b2
        dim3 grid2((D_MODEL + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        matmul_a_b_kernel<<<grid2, block>>>(act->a1[l], m->W2[l], act->z2[l], B, D_MODEL, D_MODEL);
        bias_forward_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(act->z2[l], m->b2[l], B, D_MODEL);

        // x_pre_relu = x + z2
        CUDA_CHECK(cudaMemcpy(act->x_pre_relu[l], act->x[l], B * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice));
        residual_add_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(act->x_pre_relu[l], act->z2[l], B * D_MODEL);

        // x[l+1] = relu(x_pre_relu)
        CUDA_CHECK(cudaMemcpy(act->x[l+1], act->x_pre_relu[l], B * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice));
        relu_forward_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(act->x[l+1], B * D_MODEL);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_blocks += get_time_diff(start, end);

    // head logits: (B,D) @ (D,10) -> (B,10), then bias, then softmax
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_h((OUTPUT_SIZE + block.x - 1) / block.x, (B + block.y - 1) / block.y);
    matmul_a_b_kernel<<<grid_h, block>>>(act->x[N_BLOCKS], m->Wh, act->logits, B, D_MODEL, OUTPUT_SIZE);
    bias_forward_kernel<<<(B * OUTPUT_SIZE + 255) / 256, 256>>>(act->logits, m->bh, B, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_head += get_time_diff(start, end);

    // probs = softmax(logits)
    clock_gettime(CLOCK_MONOTONIC, &start);
    CUDA_CHECK(cudaMemcpy(act->probs, act->logits, B * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
    softmax_kernel<<<B, 1>>>(act->probs, B, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_softmax += get_time_diff(start, end);
}

void backward_resmlp_timed(
    ResMLP *m,
    const float *d_input,      // (B,784)
    const int *d_labels,       // (B,)
    Activations *act,
    int B,
    TimingStats *stats
) {
    struct timespec start, end;
    dim3 block(32, 32);

    zero_grad_kernel<<<(INPUT_SIZE * D_MODEL + 255) / 256, 256>>>(m->dW0, INPUT_SIZE * D_MODEL);
    zero_grad_kernel<<<(D_MODEL + 255) / 256, 256>>>(m->db0, D_MODEL);

    for (int l = 0; l < N_BLOCKS; l++) {
        zero_grad_kernel<<<(D_MODEL * D_MODEL + 255) / 256, 256>>>(m->dW1[l], D_MODEL * D_MODEL);
        zero_grad_kernel<<<(D_MODEL + 255) / 256, 256>>>(m->db1[l], D_MODEL);
        zero_grad_kernel<<<(D_MODEL * D_MODEL + 255) / 256, 256>>>(m->dW2[l], D_MODEL * D_MODEL);
        zero_grad_kernel<<<(D_MODEL + 255) / 256, 256>>>(m->db2[l], D_MODEL);
    }
    zero_grad_kernel<<<(D_MODEL * OUTPUT_SIZE + 255) / 256, 256>>>(m->dWh, D_MODEL * OUTPUT_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(m->dbh, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_grad_out = NULL;   // (B,10)
    float *d_grad_x = NULL;     // (B,D) running gradient
    float *d_tmp = NULL;        // (B,D) temp
    float *d_tmp2 = NULL;       // (B,D) temp
    CUDA_CHECK(cudaMalloc(&d_grad_out, B * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_x,   B * D_MODEL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp,      B * D_MODEL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp2,     B * D_MODEL * sizeof(float)));

    clock_gettime(CLOCK_MONOTONIC, &start);
    compute_output_gradients_kernel<<<(B + 255) / 256, 256>>>(d_grad_out, act->probs, d_labels, B);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_softmax_grad += get_time_diff(start, end);

 
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_dWh((OUTPUT_SIZE + block.x - 1) / block.x, (D_MODEL + block.y - 1) / block.y);
    matmul_at_b_kernel<<<grid_dWh, block>>>(act->x[N_BLOCKS], d_grad_out, m->dWh, B, D_MODEL, OUTPUT_SIZE);
    bias_backward_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(m->dbh, d_grad_out, B, OUTPUT_SIZE);

    dim3 grid_dX((D_MODEL + block.x - 1) / block.x, (B + block.y - 1) / block.y);
    matmul_a_bt_kernel<<<grid_dX, block>>>(d_grad_out, m->Wh, d_grad_x, B, OUTPUT_SIZE, D_MODEL);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_head += get_time_diff(start, end);


    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int l = N_BLOCKS - 1; l >= 0; l--) {
        relu_backward_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(d_grad_x, act->x[l+1], B * D_MODEL);

        CUDA_CHECK(cudaMemcpy(d_tmp,  d_grad_x, B * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice)); // grad_z2
        CUDA_CHECK(cudaMemcpy(d_tmp2, d_grad_x, B * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice)); // residual path to x

        // z2 = a1 @ W2 + b2
        // dW2 = a1.T @ grad_z2
        dim3 grid_dW2((D_MODEL + block.x - 1) / block.x, (D_MODEL + block.y - 1) / block.y);
        matmul_at_b_kernel<<<grid_dW2, block>>>(act->a1[l], d_tmp, m->dW2[l], B, D_MODEL, D_MODEL);
        bias_backward_kernel<<<(D_MODEL + 255) / 256, 256>>>(m->db2[l], d_tmp, B, D_MODEL);

        // grad_a1 = grad_z2 @ W2.T  (into d_grad_x temporarily)
        dim3 grid_ga1((D_MODEL + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        matmul_a_bt_kernel<<<grid_ga1, block>>>(d_tmp, m->W2[l], d_grad_x, B, D_MODEL, D_MODEL);

        // a1 = relu(z1)
        relu_backward_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(d_grad_x, act->a1[l], B * D_MODEL); // uses post-relu a1

        // z1 = x[l] @ W1 + b1
        // dW1 = x[l].T @ grad_z1
        dim3 grid_dW1((D_MODEL + block.x - 1) / block.x, (D_MODEL + block.y - 1) / block.y);
        matmul_at_b_kernel<<<grid_dW1, block>>>(act->x[l], d_grad_x, m->dW1[l], B, D_MODEL, D_MODEL);
        bias_backward_kernel<<<(D_MODEL + 255) / 256, 256>>>(m->db1[l], d_grad_x, B, D_MODEL);

        // grad_x_main = grad_z1 @ W1.T  (into d_tmp)
        dim3 grid_gx((D_MODEL + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        matmul_a_bt_kernel<<<grid_gx, block>>>(d_grad_x, m->W1[l], d_tmp, B, D_MODEL, D_MODEL);

        // total grad_x = grad_x_main + grad_x_residual
        CUDA_CHECK(cudaMemcpy(d_grad_x, d_tmp2, B * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice)); // start with residual
        residual_add_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(d_grad_x, d_tmp, B * D_MODEL);          // add main
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_blocks += get_time_diff(start, end);

    // x[0] = relu( X @ W0 + b0 )
    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_backward_kernel<<<(B * D_MODEL + 255) / 256, 256>>>(d_grad_x, act->x[0], B * D_MODEL);

    // dW0 = X.T @ grad_x0   => (784,D)
    dim3 grid_dW0((D_MODEL + block.x - 1) / block.x, (INPUT_SIZE + block.y - 1) / block.y);
    matmul_at_b_kernel<<<grid_dW0, block>>>(d_input, d_grad_x, m->dW0, B, INPUT_SIZE, D_MODEL);
    bias_backward_kernel<<<(D_MODEL + 255) / 256, 256>>>(m->db0, d_grad_x, B, D_MODEL);

    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_proj += get_time_diff(start, end);

    CUDA_CHECK(cudaFree(d_grad_out));
    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_tmp2));
}

// Update weights
void update_resmlp_timed(ResMLP *m, TimingStats *stats) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    weight_update_kernel<<<(INPUT_SIZE * D_MODEL + 255) / 256, 256>>>(m->W0, m->dW0, INPUT_SIZE * D_MODEL);
    weight_update_kernel<<<(D_MODEL + 255) / 256, 256>>>(m->b0, m->db0, D_MODEL);

    for (int l = 0; l < N_BLOCKS; l++) {
        weight_update_kernel<<<(D_MODEL * D_MODEL + 255) / 256, 256>>>(m->W1[l], m->dW1[l], D_MODEL * D_MODEL);
        weight_update_kernel<<<(D_MODEL + 255) / 256, 256>>>(m->b1[l], m->db1[l], D_MODEL);

        weight_update_kernel<<<(D_MODEL * D_MODEL + 255) / 256, 256>>>(m->W2[l], m->dW2[l], D_MODEL * D_MODEL);
        weight_update_kernel<<<(D_MODEL + 255) / 256, 256>>>(m->b2[l], m->db2[l], D_MODEL);
    }

    weight_update_kernel<<<(D_MODEL * OUTPUT_SIZE + 255) / 256, 256>>>(m->Wh, m->dWh, D_MODEL * OUTPUT_SIZE);
    weight_update_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(m->bh, m->dbh, OUTPUT_SIZE);

    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->weight_updates += get_time_diff(start, end);
}


void train_resmlp(ResMLP *m, const float *X_train, const int *y_train) {
    float *d_x = NULL;
    int *d_y = NULL;
    CUDA_CHECK(cudaMalloc(&d_x, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, BATCH_SIZE * sizeof(int)));

    Activations act;
    alloc_activations(&act, BATCH_SIZE);

    float *h_probs = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    TimingStats stats;
    memset(&stats, 0, sizeof(stats));

    struct timespec total_start, total_end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;

            // copy batch to GPU
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaMemcpy(d_x, X_train + start_idx * INPUT_SIZE,
                                  BATCH_SIZE * INPUT_SIZE * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_y, y_train + start_idx,
                                  BATCH_SIZE * sizeof(int),
                                  cudaMemcpyHostToDevice));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.data_loading += get_time_diff(step_start, step_end);

            // forward
            forward_resmlp_timed(m, d_x, &act, BATCH_SIZE, &stats);

            // loss on CPU (copy probs back)
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaMemcpy(h_probs, act.probs, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            float loss = cross_entropy_loss(h_probs, y_train + start_idx, BATCH_SIZE);
            epoch_loss += loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.loss_ce += get_time_diff(step_start, step_end);

            // backward
            backward_resmlp_timed(m, d_x, d_y, &act, BATCH_SIZE, &stats);

            // update
            update_resmlp_timed(m, &stats);
        }

        printf("Epoch %d loss: %.4f\n", epoch, epoch_loss / (float)num_batches);
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);

    printf("\n=== CUDA RESIDUAL-MLP TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);

    printf("  Data loading:   %6.3fs (%5.1f%%)\n", stats.data_loading, 100.0 * stats.data_loading / stats.total_time);

    double fwd = stats.fwd_proj + stats.fwd_blocks + stats.fwd_head + stats.fwd_softmax;
    printf("  Forward total:  %6.3fs (%5.1f%%)\n", fwd, 100.0 * fwd / stats.total_time);
    printf("    proj:         %6.3fs\n", stats.fwd_proj);
    printf("    blocks:       %6.3fs\n", stats.fwd_blocks);
    printf("    head:         %6.3fs\n", stats.fwd_head);
    printf("    softmax:      %6.3fs\n", stats.fwd_softmax);

    printf("  Loss (CE CPU):  %6.3fs (%5.1f%%)\n", stats.loss_ce, 100.0 * stats.loss_ce / stats.total_time);

    double bwd = stats.bwd_softmax_grad + stats.bwd_head + stats.bwd_blocks + stats.bwd_proj;
    printf("  Backward total: %6.3fs (%5.1f%%)\n", bwd, 100.0 * bwd / stats.total_time);
    printf("    softmax grad: %6.3fs\n", stats.bwd_softmax_grad);
    printf("    head:         %6.3fs\n", stats.bwd_head);
    printf("    blocks:       %6.3fs\n", stats.bwd_blocks);
    printf("    proj:         %6.3fs\n", stats.bwd_proj);

    printf("  Updates:        %6.3fs (%5.1f%%)\n", stats.weight_updates, 100.0 * stats.weight_updates / stats.total_time);

    // cleanup
    free(h_probs);
    free_activations(&act);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}


int main() {
    srand((unsigned)time(NULL));

    ResMLP model;
    initialize_resmlp(&model);

    float *X_train = (float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int   *y_train = (int*)malloc(TRAIN_SIZE * sizeof(int));
    float *X_test  = (float*)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int   *y_test  = (int*)malloc(TEST_SIZE * sizeof(int));

    load_data("./data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("./data/y_train.bin", y_train, TRAIN_SIZE);

    load_data("./data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    normalize_data(X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("./data/y_test.bin", y_test, TEST_SIZE);

    train_resmlp(&model, X_train, y_train);

    free_resmlp(&model);

    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}

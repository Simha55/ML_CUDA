


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

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

static double get_time_diff(struct timespec start, struct timespec end) {
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
#define RES_SCALE 1.0f


static void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (!file) { fprintf(stderr, "Error opening file: %s\n", filename); exit(1); }
    size_t read_size = fread(data, sizeof(float), (size_t)size, file);
    if ((int)read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

static void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (!file) { fprintf(stderr, "Error opening file: %s\n", filename); exit(1); }
    size_t read_size = fread(labels, sizeof(int), (size_t)size, file);
    if ((int)read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

static void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std  = 0.3081f;
    for (int i = 0; i < size; i++) data[i] = (data[i] - mean) / std;
}


static void initialize_weights(float *w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        float r = (float)rand() / (float)RAND_MAX; // [0,1]
        w[i] = (2.0f * r - 1.0f) * scale;
    }
}

static void initialize_bias(float *b, int size) {
    for (int i = 0; i < size; i++) b[i] = 0.0f;
}



static void matmul_rm(const float *A, const float *B, float *C, int m, int n, int k) {

    for (int i = 0; i < m; i++) {
        const float *Ai = A + (size_t)i * n;
        float *Ci = C + (size_t)i * k;
        // init row
        for (int j = 0; j < k; j++) Ci[j] = 0.0f;

        for (int t = 0; t < n; t++) {
            float a = Ai[t];
            const float *Bt = B + (size_t)t * k;
            for (int j = 0; j < k; j++) {
                Ci[j] += a * Bt[j];
            }
        }
    }
}

static void add_bias_rm(float *X, const float *b, int m, int n) {
    for (int i = 0; i < m; i++) {
        float *Xi = X + (size_t)i * n;
        for (int j = 0; j < n; j++) Xi[j] += b[j];
    }
}

static void relu_inplace(float *X, int size) {
    for (int i = 0; i < size; i++) X[i] = (X[i] > 0.0f) ? X[i] : 0.0f;
}

static void relu_backward_inplace(float *dX, const float *X_post_relu, int size) {
    for (int i = 0; i < size; i++) {
        dX[i] *= (X_post_relu[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

static void residual_add_scaled(float *X, const float *Y, float alpha, int size) {
    for (int i = 0; i < size; i++) X[i] += alpha * Y[i];
}

static void softmax_rowwise(float *X, int B, int C) {
    for (int b = 0; b < B; b++) {
        float *row = X + (size_t)b * C;
        float maxv = row[0];
        for (int i = 1; i < C; i++) if (row[i] > maxv) maxv = row[i];

        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            float v = expf(row[i] - maxv);
            row[i] = v;
            sum += v;
        }
        if (!(sum > 0.0f) || !isfinite(sum)) {
            float u = 1.0f / (float)C;
            for (int i = 0; i < C; i++) row[i] = u;
            continue;
        }
        for (int i = 0; i < C; i++) {
            float p = row[i] / sum;
            row[i] = fmaxf(p, 1e-7f);
        }
    }
}

static float cross_entropy_loss_cpu(const float *probs, const int *labels, int B) {
    float total = 0.0f;
    for (int b = 0; b < B; b++) {
        int y = labels[b];
        float p = probs[(size_t)b * OUTPUT_SIZE + y];
        total -= logf(fmaxf(p, 1e-7f));
    }
    return total / (float)B;
}

static void softmax_ce_backward(float *dlogits, const float *probs, const int *labels, int B) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < OUTPUT_SIZE; c++) {
            dlogits[(size_t)b * OUTPUT_SIZE + c] = probs[(size_t)b * OUTPUT_SIZE + c];
        }
        dlogits[(size_t)b * OUTPUT_SIZE + labels[b]] -= 1.0f;
        float invB = 1.0f / (float)B;
        for (int c = 0; c < OUTPUT_SIZE; c++) dlogits[(size_t)b * OUTPUT_SIZE + c] *= invB;
    }
}


static void matmul_AT_B_rm(const float *A, const float *B, float *C, int m, int n, int k) {

    for (int i = 0; i < n * k; i++) C[i] = 0.0f;

    for (int t = 0; t < m; t++) {
        const float *At = A + (size_t)t * n;
        const float *Bt = B + (size_t)t * k;
        for (int i = 0; i < n; i++) {
            float a = At[i];
            float *Ci = C + (size_t)i * k;
            for (int j = 0; j < k; j++) {
                Ci[j] += a * Bt[j];
            }
        }
    }
}


static void matmul_A_BT_rm(const float *A, const float *B, float *C, int m, int n, int k) {

    for (int i = 0; i < m; i++) {
        const float *Ai = A + (size_t)i * n;
        float *Ci = C + (size_t)i * k;
        for (int j = 0; j < k; j++) {
            const float *Bj = B + (size_t)j * n;
            float sum = 0.0f;
            for (int t = 0; t < n; t++) sum += Ai[t] * Bj[t];
            Ci[j] = sum;
        }
    }
}

static void bias_backward_cpu(float *db, const float *dY, int B, int out) {
    for (int j = 0; j < out; j++) db[j] = 0.0f;
    for (int b = 0; b < B; b++) {
        const float *row = dY + (size_t)b * out;
        for (int j = 0; j < out; j++) db[j] += row[j];
    }
}

static void sgd_update(float *W, const float *dW, int size, float lr) {
    for (int i = 0; i < size; i++) W[i] -= lr * dW[i];
}

typedef struct {
    float *W0, *b0, *dW0, *db0; 

    float *W1[N_BLOCKS], *b1[N_BLOCKS], *dW1[N_BLOCKS], *db1[N_BLOCKS]; 
    float *W2[N_BLOCKS], *b2[N_BLOCKS], *dW2[N_BLOCKS], *db2[N_BLOCKS];

    // head: D -> 10
    float *Wh, *bh, *dWh, *dbh; 
} ResMLP;

static void initialize_resmlp_cpu(ResMLP *m) {
    m->W0  = (float*)malloc(INPUT_SIZE * D_MODEL * sizeof(float));
    m->b0  = (float*)malloc(D_MODEL * sizeof(float));
    m->dW0 = (float*)malloc(INPUT_SIZE * D_MODEL * sizeof(float));
    m->db0 = (float*)malloc(D_MODEL * sizeof(float));

    for (int l = 0; l < N_BLOCKS; l++) {
        m->W1[l]  = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
        m->b1[l]  = (float*)malloc(D_MODEL * sizeof(float));
        m->dW1[l] = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
        m->db1[l] = (float*)malloc(D_MODEL * sizeof(float));

        m->W2[l]  = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
        m->b2[l]  = (float*)malloc(D_MODEL * sizeof(float));
        m->dW2[l] = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
        m->db2[l] = (float*)malloc(D_MODEL * sizeof(float));
    }

    m->Wh  = (float*)malloc(D_MODEL * OUTPUT_SIZE * sizeof(float));
    m->bh  = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    m->dWh = (float*)malloc(D_MODEL * OUTPUT_SIZE * sizeof(float));
    m->dbh = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(m->W0, INPUT_SIZE, D_MODEL);
    initialize_bias(m->b0, D_MODEL);

    for (int l = 0; l < N_BLOCKS; l++) {
        initialize_weights(m->W1[l], D_MODEL, D_MODEL);
        initialize_bias(m->b1[l], D_MODEL);
        initialize_weights(m->W2[l], D_MODEL, D_MODEL);
        initialize_bias(m->b2[l], D_MODEL);
    }

    initialize_weights(m->Wh, D_MODEL, OUTPUT_SIZE);
    initialize_bias(m->bh, OUTPUT_SIZE);
}

static void free_resmlp_cpu(ResMLP *m) {
    free(m->W0); free(m->b0); free(m->dW0); free(m->db0);
    for (int l = 0; l < N_BLOCKS; l++) {
        free(m->W1[l]); free(m->b1[l]); free(m->dW1[l]); free(m->db1[l]);
        free(m->W2[l]); free(m->b2[l]); free(m->dW2[l]); free(m->db2[l]);
    }
    free(m->Wh); free(m->bh); free(m->dWh); free(m->dbh);
}


typedef struct {
    float *x[N_BLOCKS + 1];       // (B,D)
    float *z1[N_BLOCKS];          // (B,D)
    float *a1[N_BLOCKS];          // (B,D)
    float *z2[N_BLOCKS];          // (B,D)
    float *x_pre_relu[N_BLOCKS];  // (B,D)
    float *logits;                // (B,10)
    float *probs;                 // (B,10)
} Activations;

static void alloc_activations_cpu(Activations *a, int B) {
    for (int i = 0; i < N_BLOCKS + 1; i++) a->x[i] = (float*)malloc((size_t)B * D_MODEL * sizeof(float));
    for (int l = 0; l < N_BLOCKS; l++) {
        a->z1[l] = (float*)malloc((size_t)B * D_MODEL * sizeof(float));
        a->a1[l] = (float*)malloc((size_t)B * D_MODEL * sizeof(float));
        a->z2[l] = (float*)malloc((size_t)B * D_MODEL * sizeof(float));
        a->x_pre_relu[l] = (float*)malloc((size_t)B * D_MODEL * sizeof(float));
    }
    a->logits = (float*)malloc((size_t)B * OUTPUT_SIZE * sizeof(float));
    a->probs  = (float*)malloc((size_t)B * OUTPUT_SIZE * sizeof(float));
}

static void free_activations_cpu(Activations *a) {
    for (int i = 0; i < N_BLOCKS + 1; i++) free(a->x[i]);
    for (int l = 0; l < N_BLOCKS; l++) {
        free(a->z1[l]); free(a->a1[l]); free(a->z2[l]); free(a->x_pre_relu[l]);
    }
    free(a->logits);
    free(a->probs);
}


static void forward_resmlp_timed_cpu(
    ResMLP *m,
    const float *input_batch, // (B,784)
    Activations *a,
    int B,
    TimingStats *stats
) {
    struct timespec s,e;

    // proj: x0 = relu( X @ W0 + b0 )
    clock_gettime(CLOCK_MONOTONIC, &s);
    matmul_rm(input_batch, m->W0, a->x[0], B, INPUT_SIZE, D_MODEL);
    add_bias_rm(a->x[0], m->b0, B, D_MODEL);
    relu_inplace(a->x[0], B * D_MODEL);
    clock_gettime(CLOCK_MONOTONIC, &e);
    stats->fwd_proj += get_time_diff(s,e);

    // blocks
    clock_gettime(CLOCK_MONOTONIC, &s);
    for (int l = 0; l < N_BLOCKS; l++) {
        // z1 = x @ W1 + b1
        matmul_rm(a->x[l], m->W1[l], a->z1[l], B, D_MODEL, D_MODEL);
        add_bias_rm(a->z1[l], m->b1[l], B, D_MODEL);

        // a1 = relu(z1)
        memcpy(a->a1[l], a->z1[l], (size_t)B * D_MODEL * sizeof(float));
        relu_inplace(a->a1[l], B * D_MODEL);

        // z2 = a1 @ W2 + b2
        matmul_rm(a->a1[l], m->W2[l], a->z2[l], B, D_MODEL, D_MODEL);
        add_bias_rm(a->z2[l], m->b2[l], B, D_MODEL);

        // x_pre_relu = x + RES_SCALE*z2
        memcpy(a->x_pre_relu[l], a->x[l], (size_t)B * D_MODEL * sizeof(float));
        residual_add_scaled(a->x_pre_relu[l], a->z2[l], RES_SCALE, B * D_MODEL);

        // x[l+1] = relu(x_pre_relu)
        memcpy(a->x[l+1], a->x_pre_relu[l], (size_t)B * D_MODEL * sizeof(float));
        relu_inplace(a->x[l+1], B * D_MODEL);
    }
    clock_gettime(CLOCK_MONOTONIC, &e);
    stats->fwd_blocks += get_time_diff(s,e);

    // head: logits = x_last @ Wh + bh
    clock_gettime(CLOCK_MONOTONIC, &s);
    matmul_rm(a->x[N_BLOCKS], m->Wh, a->logits, B, D_MODEL, OUTPUT_SIZE);
    add_bias_rm(a->logits, m->bh, B, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &e);
    stats->fwd_head += get_time_diff(s,e);

    // softmax: probs = softmax(logits)
    clock_gettime(CLOCK_MONOTONIC, &s);
    memcpy(a->probs, a->logits, (size_t)B * OUTPUT_SIZE * sizeof(float));
    softmax_rowwise(a->probs, B, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &e);
    stats->fwd_softmax += get_time_diff(s,e);
}

static void backward_resmlp_timed_cpu(
    ResMLP *m,
    const float *input_batch, // (B,784)
    const int *labels_batch,  // (B,)
    Activations *a,
    int B,
    TimingStats *stats,
    // temp buffers
    float *dlogits,   // (B,10)
    float *dX,        // (B,D) running grad
    float *tmp,       // (B,D)
    float *tmp2       // (B,D)
) {
    struct timespec s,e;

    // zero grads
    memset(m->dW0, 0, INPUT_SIZE * D_MODEL * sizeof(float));
    memset(m->db0, 0, D_MODEL * sizeof(float));
    for (int l = 0; l < N_BLOCKS; l++) {
        memset(m->dW1[l], 0, D_MODEL * D_MODEL * sizeof(float));
        memset(m->db1[l], 0, D_MODEL * sizeof(float));
        memset(m->dW2[l], 0, D_MODEL * D_MODEL * sizeof(float));
        memset(m->db2[l], 0, D_MODEL * sizeof(float));
    }
    memset(m->dWh, 0, D_MODEL * OUTPUT_SIZE * sizeof(float));
    memset(m->dbh, 0, OUTPUT_SIZE * sizeof(float));

    // softmax grad: dlogits = (probs - onehot)/B
    clock_gettime(CLOCK_MONOTONIC, &s);
    softmax_ce_backward(dlogits, a->probs, labels_batch, B);
    clock_gettime(CLOCK_MONOTONIC, &e);
    stats->bwd_softmax_grad += get_time_diff(s,e);

    // head grads
    clock_gettime(CLOCK_MONOTONIC, &s);
    // dWh = x_last^T @ dlogits
    matmul_AT_B_rm(a->x[N_BLOCKS], dlogits, m->dWh, B, D_MODEL, OUTPUT_SIZE);
    bias_backward_cpu(m->dbh, dlogits, B, OUTPUT_SIZE);

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < D_MODEL; i++) {
            float sum = 0.0f;
            const float *Wi = m->Wh + (size_t)i * OUTPUT_SIZE;
            const float *dl = dlogits + (size_t)b * OUTPUT_SIZE;
            for (int c = 0; c < OUTPUT_SIZE; c++) sum += dl[c] * Wi[c];
            dX[(size_t)b * D_MODEL + i] = sum;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &e);
    stats->bwd_head += get_time_diff(s,e);

    // blocks backward
    clock_gettime(CLOCK_MONOTONIC, &s);
    for (int l = N_BLOCKS - 1; l >= 0; l--) {
        // back through output ReLU: x[l+1] is post-ReLU
        relu_backward_inplace(dX, a->x[l+1], B * D_MODEL);

        // split grad at xsum = x + RES_SCALE*z2
        // grad_z2 = dX * RES_SCALE
        // grad_x_res = dX
        memcpy(tmp, dX, (size_t)B * D_MODEL * sizeof(float));   // tmp = grad_z2 (will scale)
        if (RES_SCALE != 1.0f) {
            for (int i = 0; i < B * D_MODEL; i++) tmp[i] *= RES_SCALE;
        }
        memcpy(tmp2, dX, (size_t)B * D_MODEL * sizeof(float));  // tmp2 = residual grad to x[l]

        // z2 = a1 @ W2 + b2
        matmul_AT_B_rm(a->a1[l], tmp, m->dW2[l], B, D_MODEL, D_MODEL);
        bias_backward_cpu(m->db2[l], tmp, B, D_MODEL);

        // grad_a1 = tmp @ W2^T
        // grad_a1[b,i] = sum_j tmp[b,j] * W2[i,j]
        for (int b = 0; b < B; b++) {
            const float *gz2 = tmp + (size_t)b * D_MODEL;
            float *ga1 = dX + (size_t)b * D_MODEL; // reuse dX for grad_a1
            for (int i = 0; i < D_MODEL; i++) {
                const float *Wi = m->W2[l] + (size_t)i * D_MODEL;
                float sum = 0.0f;
                for (int j = 0; j < D_MODEL; j++) sum += gz2[j] * Wi[j];
                ga1[i] = sum;
            }
        }

        // a1 = relu(z1): derivative uses post-relu a1
        relu_backward_inplace(dX, a->a1[l], B * D_MODEL);

        // z1 = x[l] @ W1 + b1
        matmul_AT_B_rm(a->x[l], dX, m->dW1[l], B, D_MODEL, D_MODEL);
        bias_backward_cpu(m->db1[l], dX, B, D_MODEL);

        // grad_x_main = dX @ W1^T into tmp
        for (int b = 0; b < B; b++) {
            const float *gz1 = dX + (size_t)b * D_MODEL;
            float *gx = tmp + (size_t)b * D_MODEL;
            for (int i = 0; i < D_MODEL; i++) {
                const float *Wi = m->W1[l] + (size_t)i * D_MODEL;
                float sum = 0.0f;
                for (int j = 0; j < D_MODEL; j++) sum += gz1[j] * Wi[j];
                gx[i] = sum;
            }
        }

        // total grad into x[l] = grad_x_residual + grad_x_main
        // dX = tmp2 + tmp
        for (int i = 0; i < B * D_MODEL; i++) dX[i] = tmp2[i] + tmp[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &e);
    stats->bwd_blocks += get_time_diff(s,e);

    // proj backward
    clock_gettime(CLOCK_MONOTONIC, &s);
    // x[0] is post-ReLU
    relu_backward_inplace(dX, a->x[0], B * D_MODEL);

    // dW0 = X^T @ dX
    matmul_AT_B_rm(input_batch, dX, m->dW0, B, INPUT_SIZE, D_MODEL);
    bias_backward_cpu(m->db0, dX, B, D_MODEL);
    clock_gettime(CLOCK_MONOTONIC, &e);
    stats->bwd_proj += get_time_diff(s,e);
}

static void update_resmlp_timed_cpu(ResMLP *m, TimingStats *stats) {
    struct timespec s,e;
    clock_gettime(CLOCK_MONOTONIC, &s);

    sgd_update(m->W0, m->dW0, INPUT_SIZE * D_MODEL, LEARNING_RATE);
    sgd_update(m->b0, m->db0, D_MODEL, LEARNING_RATE);

    for (int l = 0; l < N_BLOCKS; l++) {
        sgd_update(m->W1[l], m->dW1[l], D_MODEL * D_MODEL, LEARNING_RATE);
        sgd_update(m->b1[l], m->db1[l], D_MODEL, LEARNING_RATE);

        sgd_update(m->W2[l], m->dW2[l], D_MODEL * D_MODEL, LEARNING_RATE);
        sgd_update(m->b2[l], m->db2[l], D_MODEL, LEARNING_RATE);
    }

    sgd_update(m->Wh, m->dWh, D_MODEL * OUTPUT_SIZE, LEARNING_RATE);
    sgd_update(m->bh, m->dbh, OUTPUT_SIZE, LEARNING_RATE);

    clock_gettime(CLOCK_MONOTONIC, &e);
    stats->weight_updates += get_time_diff(s,e);
}

static void train_resmlp_cpu(ResMLP *m, const float *X_train, const int *y_train) {
    TimingStats stats;
    memset(&stats, 0, sizeof(stats));

    Activations a;
    alloc_activations_cpu(&a, BATCH_SIZE);

    float *dlogits = (float*)malloc((size_t)BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    float *dX      = (float*)malloc((size_t)BATCH_SIZE * D_MODEL * sizeof(float));
    float *tmp     = (float*)malloc((size_t)BATCH_SIZE * D_MODEL * sizeof(float));
    float *tmp2    = (float*)malloc((size_t)BATCH_SIZE * D_MODEL * sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    struct timespec total_start, total_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;

            // "data_loading" on CPU means: selecting pointers / copying to a contiguous buffer if you want.
            // To match your CUDA stat meaning (H2D copy), we will time a memcpy into a batch buffer.
            struct timespec s,e;
            clock_gettime(CLOCK_MONOTONIC, &s);

            // Make contiguous batch buffers to mimic "load batch" cost
            static float batch_x[BATCH_SIZE * INPUT_SIZE];
            static int   batch_y[BATCH_SIZE];

            memcpy(batch_x, X_train + (size_t)start_idx * INPUT_SIZE, (size_t)BATCH_SIZE * INPUT_SIZE * sizeof(float));
            memcpy(batch_y, y_train + start_idx, (size_t)BATCH_SIZE * sizeof(int));

            clock_gettime(CLOCK_MONOTONIC, &e);
            stats.data_loading += get_time_diff(s,e);

            // forward
            forward_resmlp_timed_cpu(m, batch_x, &a, BATCH_SIZE, &stats);

            // loss (CPU CE)
            clock_gettime(CLOCK_MONOTONIC, &s);
            float loss = cross_entropy_loss_cpu(a.probs, batch_y, BATCH_SIZE);
            epoch_loss += loss;
            clock_gettime(CLOCK_MONOTONIC, &e);
            stats.loss_ce += get_time_diff(s,e);

            // backward
            backward_resmlp_timed_cpu(m, batch_x, batch_y, &a, BATCH_SIZE, &stats, dlogits, dX, tmp, tmp2);

            // update
            update_resmlp_timed_cpu(m, &stats);
        }

        printf("Epoch %d loss: %.4f\n", epoch, epoch_loss / (float)num_batches);
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);

    // Print breakdown (same formatting style as your CUDA output)
    printf("\n=== CPU RESIDUAL-MLP TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);

    printf("  Data loading:    %6.3fs (%5.1f%%)\n",
           stats.data_loading, 100.0 * stats.data_loading / stats.total_time);

    double fwd = stats.fwd_proj + stats.fwd_blocks + stats.fwd_head + stats.fwd_softmax;
    printf("  Forward total:   %6.3fs (%5.1f%%)\n",
           fwd, 100.0 * fwd / stats.total_time);
    printf("    proj:          %6.3fs\n", stats.fwd_proj);
    printf("    blocks:        %6.3fs\n", stats.fwd_blocks);
    printf("    head:          %6.3fs\n", stats.fwd_head);
    printf("    softmax:       %6.3fs\n", stats.fwd_softmax);

    printf("  Loss (CE CPU):   %6.3fs (%5.1f%%)\n",
           stats.loss_ce, 100.0 * stats.loss_ce / stats.total_time);

    double bwd = stats.bwd_softmax_grad + stats.bwd_head + stats.bwd_blocks + stats.bwd_proj;
    printf("  Backward total:  %6.3fs (%5.1f%%)\n",
           bwd, 100.0 * bwd / stats.total_time);
    printf("    softmax grad:  %6.3fs\n", stats.bwd_softmax_grad);
    printf("    head:          %6.3fs\n", stats.bwd_head);
    printf("    blocks:        %6.3fs\n", stats.bwd_blocks);
    printf("    proj:          %6.3fs\n", stats.bwd_proj);

    printf("  Updates:         %6.3fs (%5.1f%%)\n",
           stats.weight_updates, 100.0 * stats.weight_updates / stats.total_time);

    free(dlogits);
    free(dX);
    free(tmp);
    free(tmp2);
    free_activations_cpu(&a);
}


int main() {
    srand((unsigned)time(NULL));

    ResMLP model;
    initialize_resmlp_cpu(&model);

    float *X_train = (float*)malloc((size_t)TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int   *y_train = (int*)malloc((size_t)TRAIN_SIZE * sizeof(int));
    float *X_test  = (float*)malloc((size_t)TEST_SIZE * INPUT_SIZE * sizeof(float));
    int   *y_test  = (int*)malloc((size_t)TEST_SIZE * sizeof(int));

    load_data("./data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("./data/y_train.bin", y_train, TRAIN_SIZE);

    load_data("./data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    normalize_data(X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("./data/y_test.bin", y_test, TEST_SIZE);

    train_resmlp_cpu(&model, X_train, y_train);

    free_resmlp_cpu(&model);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}


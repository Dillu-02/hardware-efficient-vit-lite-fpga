// FULL CODE WITH SOFTMAX NORMALISATION INCLUDED
// ERROR-FREE AND NO SLACK. ALL LOOP CONSTRAINTS WERE SATISFIED
// 1 input + 12 weights + 1 Wo + 1 output ports
// port pragmas are added, REMOVED Wo array partitioning
// limited resource utilization observed

#include <hls_math.h>
#include "mhsa.h"

#define TILE_SIZE 16
#define K_TILE 16

#define TILE_I 16
#define TILE_J 16
#define TILE_K 16

void tiled_qkv_projection(
    data_t input_buf[NUM_PATCHES][EMBEDDING_DIM],
    data_t weight[EMBEDDING_DIM][HEAD_DIM],
    data_t output_buf[NUM_PATCHES][HEAD_DIM]
) {
#pragma HLS INLINE off

    // Initialize output
    for (int i = 0; i < NUM_PATCHES; i++) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES
        for (int j = 0; j < HEAD_DIM; j++) {
#pragma HLS loop_tripcount min=1 max=HEAD_DIM
#pragma HLS PIPELINE II=1
            output_buf[i][j] = 0;
        }
    }

    // Tiled GEMM
    for (int it = 0; it < NUM_PATCHES; it += TILE_I) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES/TILE_I
        for (int jt = 0; jt < HEAD_DIM; jt += TILE_J) {
#pragma HLS loop_tripcount min=1 max=HEAD_DIM/TILE_J
            for (int kt = 0; kt < EMBEDDING_DIM; kt += TILE_K) {
#pragma HLS loop_tripcount min=1 max=EMBEDDING_DIM/TILE_K
                for (int i = it; i < it + TILE_I && i < NUM_PATCHES; i++) {
#pragma HLS loop_tripcount min=1 max=TILE_I
                    for (int j = jt; j < jt + TILE_J && j < HEAD_DIM; j++) {
#pragma HLS loop_tripcount min=1 max=TILE_J
#pragma HLS PIPELINE II=1
                        for (int k = kt; k < kt + TILE_K && k < EMBEDDING_DIM; k++) {
#pragma HLS loop_tripcount min=1 max=TILE_K
                            output_buf[i][j] += input_buf[i][k] * weight[k][j];
                        }
                    }
                }
            }
        }
    }
}


void process_one_head(
    data_t input_buf[NUM_PATCHES][EMBEDDING_DIM],
    data_t Wq[EMBEDDING_DIM][HEAD_DIM],
    data_t Wk[EMBEDDING_DIM][HEAD_DIM],
    data_t Wv[EMBEDDING_DIM][HEAD_DIM],
    data_t Z_concat[NUM_PATCHES][NUM_HEADS * HEAD_DIM],
    int head_index
)
{
#pragma HLS INLINE off

    data_t Q[NUM_PATCHES][HEAD_DIM];
    data_t K[NUM_PATCHES][HEAD_DIM];
    data_t V[NUM_PATCHES][HEAD_DIM];
    data_t scores[NUM_PATCHES][NUM_PATCHES];
    data_t softmax[NUM_PATCHES][NUM_PATCHES];
    data_t Z[NUM_PATCHES][HEAD_DIM];

#pragma HLS ARRAY_PARTITION variable=Q block factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=K block factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=V block factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=Z block factor=16 dim=2

    // === Step 1: Q, K, V projection using tiled function ===
    tiled_qkv_projection(input_buf, Wq, Q);
    tiled_qkv_projection(input_buf, Wk, K);
    tiled_qkv_projection(input_buf, Wv, V);

    // === Step 2: Attention Scores (QÂ·K^T) ===
    for (int ti = 0; ti < NUM_PATCHES; ti += TILE_SIZE) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES/TILE_SIZE
        for (int tj = 0; tj < NUM_PATCHES; tj += TILE_SIZE) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES/TILE_SIZE
            for (int i = ti; i < ti + TILE_SIZE; i++) {
#pragma HLS loop_tripcount min=1 max=TILE_SIZE
                for (int j = tj; j < tj + TILE_SIZE; j++) {
#pragma HLS loop_tripcount min=1 max=TILE_SIZE
#pragma HLS PIPELINE II=2
                    data_t sum = 0;
                    for (int k = 0; k < HEAD_DIM; k++) {
#pragma HLS loop_tripcount min=1 max=HEAD_DIM
                        sum += Q[i][k] * K[j][k];
                    }
                    scores[i][j] = sum / sqrtf((float)HEAD_DIM);
                }
            }
        }
    }



    // === Step 3: Row-wise Softmax (Optimized) ===
    for (int i = 0; i < NUM_PATCHES; i++) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES
        data_t max_val = scores[i][0];

        // Find row max
        for (int j = 1; j < NUM_PATCHES; j++) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES
#pragma HLS PIPELINE II=1
            if (scores[i][j] > max_val)
                max_val = scores[i][j];
        }

        // Compute exp and accumulate sum
        data_t temp_exp[NUM_PATCHES];
#pragma HLS ARRAY_PARTITION variable=temp_exp complete
        data_t sum_exp = 0;

        for (int j = 0; j < NUM_PATCHES; j++) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES
#pragma HLS PIPELINE II=3
            data_t val = scores[i][j] - max_val;
            data_t exp_val = hls::expf(val);
            temp_exp[j] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize
        for (int j = 0; j < NUM_PATCHES; j++) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES
#pragma HLS PIPELINE II=1
            softmax[i][j] = temp_exp[j] / sum_exp;
        }
    }




    // === Step 4: Weighted Sum Z = softmax Ã— V ===
    for (int i = 0; i < NUM_PATCHES; i++) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES
        for (int j = 0; j < HEAD_DIM; j++) {
#pragma HLS loop_tripcount min=1 max=HEAD_DIM
            Z[i][j] = 0;
            for (int kt = 0; kt < NUM_PATCHES; kt += K_TILE) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES/K_TILE
                for (int k = kt; k < kt + K_TILE && k < NUM_PATCHES; k++) {
#pragma HLS loop_tripcount min=1 max=K_TILE
#pragma HLS PIPELINE II=3
                    Z[i][j] += softmax[i][k] * V[k][j];
                }
            }
        }
    }

    // === Step 5: Write Z to concat buffer ===
    for (int i = 0; i < NUM_PATCHES; i++) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES
        for (int j = 0; j < HEAD_DIM; j++) {
#pragma HLS loop_tripcount min=1 max=EMBEDDING_DIM
#pragma HLS PIPELINE II=1
            Z_concat[i][head_index * HEAD_DIM + j] = Z[i][j];
        }
    }
}


// =========================== TOP FUNCTION ===========================
void multi_head_attention(
    data_t input[NUM_PATCHES][EMBEDDING_DIM],
    data_t Wq_head1[EMBEDDING_DIM][HEAD_DIM],
    data_t Wk_head1[EMBEDDING_DIM][HEAD_DIM],
    data_t Wv_head1[EMBEDDING_DIM][HEAD_DIM],

    data_t Wq_head2[EMBEDDING_DIM][HEAD_DIM],
    data_t Wk_head2[EMBEDDING_DIM][HEAD_DIM],
    data_t Wv_head2[EMBEDDING_DIM][HEAD_DIM],

    data_t Wq_head3[EMBEDDING_DIM][HEAD_DIM],
    data_t Wk_head3[EMBEDDING_DIM][HEAD_DIM],
    data_t Wv_head3[EMBEDDING_DIM][HEAD_DIM],

    data_t Wq_head4[EMBEDDING_DIM][HEAD_DIM],
    data_t Wk_head4[EMBEDDING_DIM][HEAD_DIM],
    data_t Wv_head4[EMBEDDING_DIM][HEAD_DIM],

    data_t Wo[NUM_HEADS * HEAD_DIM][EMBEDDING_DIM],
    data_t output[NUM_PATCHES][EMBEDDING_DIM]
)
{

#pragma HLS INTERFACE m_axi port=input    offset=slave bundle=in
#pragma HLS INTERFACE m_axi port=output   offset=slave bundle=out
#pragma HLS INTERFACE m_axi port=Wo       offset=slave bundle=wo

#pragma HLS INTERFACE m_axi port=Wq_head1 offset=slave bundle=wq1
#pragma HLS INTERFACE m_axi port=Wk_head1 offset=slave bundle=wk1
#pragma HLS INTERFACE m_axi port=Wv_head1 offset=slave bundle=wv1

#pragma HLS INTERFACE m_axi port=Wq_head2 offset=slave bundle=wq2
#pragma HLS INTERFACE m_axi port=Wk_head2 offset=slave bundle=wk2
#pragma HLS INTERFACE m_axi port=Wv_head2 offset=slave bundle=wv2

#pragma HLS INTERFACE m_axi port=Wq_head3 offset=slave bundle=wq3
#pragma HLS INTERFACE m_axi port=Wk_head3 offset=slave bundle=wk3
#pragma HLS INTERFACE m_axi port=Wv_head3 offset=slave bundle=wv3

#pragma HLS INTERFACE m_axi port=Wq_head4 offset=slave bundle=wq4
#pragma HLS INTERFACE m_axi port=Wk_head4 offset=slave bundle=wk4
#pragma HLS INTERFACE m_axi port=Wv_head4 offset=slave bundle=wv4

#pragma HLS INTERFACE s_axilite port=return 

    // === STEP 1: BUFFER INPUT ===
    data_t input_buf[NUM_PATCHES][EMBEDDING_DIM];
    load_input: for (int i = 0; i < NUM_PATCHES; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
#pragma HLS PIPELINE II=1
            input_buf[i][j] = input[i][j];
        }
    }

    // === STEP 2: PROCESS ALL HEADS ===
    data_t Z_concat[NUM_PATCHES][NUM_HEADS * HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=Z_concat block factor=16 dim=2

    process_one_head(input_buf, Wq_head1, Wk_head1, Wv_head1, Z_concat, 0);
    process_one_head(input_buf, Wq_head2, Wk_head2, Wv_head2, Z_concat, 1);
    process_one_head(input_buf, Wq_head3, Wk_head3, Wv_head3, Z_concat, 2);
    process_one_head(input_buf, Wq_head4, Wk_head4, Wv_head4, Z_concat, 3);

// #pragma HLS ARRAY_PARTITION variable=Wo block factor=16 dim=1

    // === STEP 3: FINAL OUTPUT = Z Ã— Wo ===
    for (int i = 0; i < NUM_PATCHES; i++) {
#pragma HLS loop_tripcount min=1 max=NUM_PATCHES
        for (int j = 0; j < EMBEDDING_DIM; j++) {
#pragma HLS loop_tripcount min=1 max=EMBEDDING_DIM
            output[i][j] = 0;
            for (int kt = 0; kt < NUM_HEADS * HEAD_DIM; kt += K_TILE) {
#pragma HLS loop_tripcount min=1 max=(NUM_HEADS*HEAD_DIM)/K_TILE
                for (int k = kt; k < kt + K_TILE && k < NUM_HEADS * HEAD_DIM; k++) {
#pragma HLS loop_tripcount min=1 max=K_TILE
#pragma HLS PIPELINE II=3
                    output[i][j] += Z_concat[i][k] * Wo[k][j];
                }
            }
        }
    }
}

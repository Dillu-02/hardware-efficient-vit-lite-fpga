#ifndef MHSA_H
#define MHSA_H

// === MHSA Configuration Constants ===
#define PATCH_SIZE              16
#define IMAGE_SIZE              224
#define NUM_PATCHES             196    // (224/16)^2 = 14x14 = 196 patches
#define NUM_HEADS               4
#define HEAD_DIM                64
#define EMBEDDING_DIM           64     // Output of MHSA layer
#define WEIGHT_PLUS_BIAS_DIM    (EMBEDDING_DIM + 1)  // Extra row for bias per projection

typedef float data_t;

// === Top-Level MHSA Function Declaration ===
void multi_head_attention(
    data_t input[NUM_PATCHES][EMBEDDING_DIM],                 // Input patch embeddings (196, 64)

    // Weight + Bias for Q projection of each head (65, 64)
    data_t Wq_head1[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],
    data_t Wq_head2[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],
    data_t Wq_head3[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],
    data_t Wq_head4[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],

    // Weight + Bias for K projection of each head (65, 64)
    data_t Wk_head1[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],
    data_t Wk_head2[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],
    data_t Wk_head3[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],
    data_t Wk_head4[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],

    // Weight + Bias for V projection of each head (65, 64)
    data_t Wv_head1[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],
    data_t Wv_head2[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],
    data_t Wv_head3[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],
    data_t Wv_head4[WEIGHT_PLUS_BIAS_DIM][HEAD_DIM],

    // Output projection weight + bias (256 + 1, 64)
    data_t Wo[NUM_HEADS * HEAD_DIM + 1][EMBEDDING_DIM],        // Includes 1 extra row for output bias

    // Final output after MHSA (196, 64)
    data_t output[NUM_PATCHES][EMBEDDING_DIM]
);

#endif // MHSA_H

#include "policy.h"
#include "arm_math.h"
#include <string.h>
#include <math.h>

// Network architecture constants
#define LAYER_0_ROWS 64
#define LAYER_0_COLS 21
#define LAYER_1_ROWS 64  
#define LAYER_1_COLS 64
#define LAYER_2_ROWS 4
#define LAYER_2_COLS 64

// Pre-allocated matrices for CMSIS operations
static arm_matrix_instance_f32 weight_matrices[NUM_LAYERS];
static arm_matrix_instance_f32 input_matrix;
static arm_matrix_instance_f32 layer0_output_matrix;
static arm_matrix_instance_f32 layer1_output_matrix;
static arm_matrix_instance_f32 layer2_output_matrix;

// Working buffers for intermediate results
static float layer0_output[LAYER_0_ROWS];
static float layer1_output[LAYER_1_ROWS];
static float layer2_output[LAYER_2_ROWS];
static float input_buffer[INPUT_SIZE];

// Matrix data storage (will be loaded during init from policy.h)
static float weight_data_0[LAYER_0_ROWS * LAYER_0_COLS];
static float weight_data_1[LAYER_1_ROWS * LAYER_1_COLS];
static float weight_data_2[LAYER_2_ROWS * LAYER_2_COLS];

static float bias_data_0[LAYER_0_ROWS];
static float bias_data_1[LAYER_1_ROWS];
static float bias_data_2[LAYER_2_ROWS];

// Observation normalization (you'll need to fill these from your training)
static const float obs_mean[INPUT_SIZE] = {
    // Fill with your normalization means - example values
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

static const float obs_std[INPUT_SIZE] = {
    // Fill with your normalization stds - example values  
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

// Fast activation functions using CMSIS
static inline void relu_inplace(float* data, uint16_t size) {
    for (uint16_t i = 0; i < size; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

static inline void tanh_inplace(float* data, uint16_t size) {
    // Calculate tanh using standard math library
    // CMSIS doesn't have a direct tanh function, so we use the standard approach
    for (uint16_t i = 0; i < size; i++) {
        data[i] = tanhf(data[i]);
    }
}

// Optimized dense layer using pre-allocated CMSIS matrices
static void dense_layer_cmsis_fast(arm_matrix_instance_f32* weights, 
                                   const float* bias,
                                   arm_matrix_instance_f32* input_mat,
                                   arm_matrix_instance_f32* output_mat) {
    
    // Perform matrix multiplication: output = weights * input
    arm_mat_mult_f32(weights, input_mat, output_mat);
    
    // Add bias using vectorized addition
    arm_add_f32(output_mat->pData, bias, output_mat->pData, output_mat->numRows);
}

void policyLoadWeights(void) {
    // Copy weights from policy.h static const arrays to optimized storage
    memcpy(weight_data_0, weights_0, sizeof(weight_data_0));
    memcpy(weight_data_1, weights_1, sizeof(weight_data_1));  
    memcpy(weight_data_2, weights_2, sizeof(weight_data_2));
    
    memcpy(bias_data_0, biases_0, sizeof(bias_data_0));
    memcpy(bias_data_1, biases_1, sizeof(bias_data_1));
    memcpy(bias_data_2, biases_2, sizeof(bias_data_2));
    
    // Initialize CMSIS matrix structures for weights
    arm_mat_init_f32(&weight_matrices[0], LAYER_0_ROWS, LAYER_0_COLS, weight_data_0);
    arm_mat_init_f32(&weight_matrices[1], LAYER_1_ROWS, LAYER_1_COLS, weight_data_1);
    arm_mat_init_f32(&weight_matrices[2], LAYER_2_ROWS, LAYER_2_COLS, weight_data_2);
    
    // Initialize matrix structures for inputs and outputs (column vectors)
    arm_mat_init_f32(&input_matrix, INPUT_SIZE, 1, input_buffer);
    arm_mat_init_f32(&layer0_output_matrix, LAYER_0_ROWS, 1, layer0_output);
    arm_mat_init_f32(&layer1_output_matrix, LAYER_1_ROWS, 1, layer1_output);
    arm_mat_init_f32(&layer2_output_matrix, LAYER_2_ROWS, 1, layer2_output);
}

void policyForward(const float* observations, float* actions) {
    // Normalize observations directly into input_buffer
    for (uint16_t i = 0; i < INPUT_SIZE; i++) {
        input_buffer[i] = (observations[i] - obs_mean[i]) / obs_std[i];
    }
    
    // Layer 0: input -> hidden1 (21 -> 64)
    dense_layer_cmsis_fast(&weight_matrices[0], bias_data_0, 
                          &input_matrix, &layer0_output_matrix);
    tanh_inplace(layer0_output, LAYER_0_ROWS);
    
    // Layer 1: hidden1 -> hidden2 (64 -> 64)  
    dense_layer_cmsis_fast(&weight_matrices[1], bias_data_1,
                          &layer0_output_matrix, &layer1_output_matrix);
    tanh_inplace(layer1_output, LAYER_1_ROWS);
    
    // Layer 2: hidden2 -> output (64 -> 4)
    dense_layer_cmsis_fast(&weight_matrices[2], bias_data_2,
                          &layer1_output_matrix, &layer2_output_matrix);
    
    // Copy result to actions (no activation on output layer)
    memcpy(actions, layer2_output, OUTPUT_SIZE * sizeof(float));
}

// Alternative simplified version if you want to bypass normalization
void policyForwardRaw(const float* observations, float* actions) {
    // Copy observations directly to input buffer
    memcpy(input_buffer, observations, INPUT_SIZE * sizeof(float));
    
    // Layer 0: input -> hidden1 (21 -> 64)
    dense_layer_cmsis_fast(&weight_matrices[0], bias_data_0, 
                          &input_matrix, &layer0_output_matrix);
    tanh_inplace(layer0_output, LAYER_0_ROWS);
    
    // Layer 1: hidden1 -> hidden2 (64 -> 64)  
    dense_layer_cmsis_fast(&weight_matrices[1], bias_data_1,
                          &layer0_output_matrix, &layer1_output_matrix);
    tanh_inplace(layer1_output, LAYER_1_ROWS);
    
    // Layer 2: hidden2 -> output (64 -> 4)
    dense_layer_cmsis_fast(&weight_matrices[2], bias_data_2,
                          &layer1_output_matrix, &layer2_output_matrix);
    
    // Copy result to actions (no final activation)
    memcpy(actions, layer2_output, OUTPUT_SIZE * sizeof(float));
}

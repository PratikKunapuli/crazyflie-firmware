
#include "stabilizer_types.h"

#include "attitude_controller.h"
#include "position_controller.h"
#include "controller_rl.h"
#include "policy.h"

#include "param.h"
#include "log.h"
#include "debug.h"
#include "math3d.h"
#include "arm_math.h"
#include "usec_time.h"


#define ATTITUDE_UPDATE_DT    (float)(1.0f/ATTITUDE_RATE)

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
// static float weight_data_0[LAYER_0_ROWS * LAYER_0_COLS];
// static float weight_data_1[LAYER_1_ROWS * LAYER_1_COLS];
// static float weight_data_2[LAYER_2_ROWS * LAYER_2_COLS];

// static float bias_data_0[LAYER_0_ROWS];
// static float bias_data_1[LAYER_1_ROWS];
// static float bias_data_2[LAYER_2_ROWS];

static attitude_t attitudeDesired;
static attitude_t rateDesired;
static float actuatorThrust;

static float cmd_thrust;
static float cmd_roll;
static float cmd_pitch;
static float cmd_yaw;
static float r_roll;
static float r_pitch;
static float r_yaw;
static float accelz;
static float obs_time;
static float inference_time;
static float in_to_out_latency;

// RL policy variables
#define OBS_SIZE INPUT_SIZE  // Use the size from policy.h
#define ACTION_SIZE OUTPUT_SIZE
static float observations[OBS_SIZE];
static float actions[ACTION_SIZE];
static float previous_action[ACTION_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f}; // Initialize to zero
static uint64_t first_call_time = 0;

// static float body_rate_scalar_xy = 720.0f; // Scale for XY body rates
// static float body_rate_scalar_z = 720.0f;  // Scale for Z body
static float body_rate_scalar_xy = 360.0f; // Scale for XY body rates
static float body_rate_scalar_z = 90.0f;  // Scale for Z body

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
    // memcpy(weight_data_0, weights_0, sizeof(weight_data_0));
    // memcpy(weight_data_1, weights_1, sizeof(weight_data_1));  
    // memcpy(weight_data_2, weights_2, sizeof(weight_data_2));
    
    // memcpy(bias_data_0, biases_0, sizeof(bias_data_0));
    // memcpy(bias_data_1, biases_1, sizeof(bias_data_1));
    // memcpy(bias_data_2, biases_2, sizeof(bias_data_2));
    
    // Initialize CMSIS matrix structures for weights
    arm_mat_init_f32(&weight_matrices[0], LAYER_0_ROWS, LAYER_0_COLS, (float*) weights_0);
    arm_mat_init_f32(&weight_matrices[1], LAYER_1_ROWS, LAYER_1_COLS, (float*) weights_1);
    arm_mat_init_f32(&weight_matrices[2], LAYER_2_ROWS, LAYER_2_COLS, (float*) weights_2);

    // Initialize matrix structures for inputs and outputs (column vectors)
    arm_mat_init_f32(&input_matrix, INPUT_SIZE, 1, input_buffer);
    arm_mat_init_f32(&layer0_output_matrix, LAYER_0_ROWS, 1, layer0_output);
    arm_mat_init_f32(&layer1_output_matrix, LAYER_1_ROWS, 1, layer1_output);
    arm_mat_init_f32(&layer2_output_matrix, LAYER_2_ROWS, 1, layer2_output);
}

// Alternative simplified version if you want to bypass normalization
void policyForwardRaw(const float* observations, float* actions) {
    // Copy observations directly to input buffer
    memcpy(input_buffer, observations, INPUT_SIZE * sizeof(float));
    
    // Layer 0: input -> hidden1 (21 -> 64)
    dense_layer_cmsis_fast(&weight_matrices[0], biases_0, 
                          &input_matrix, &layer0_output_matrix);
    tanh_inplace(layer0_output, LAYER_0_ROWS);

    // Layer 1: hidden1 -> hidden2 (64 -> 64)
    dense_layer_cmsis_fast(&weight_matrices[1], biases_1,
                          &layer0_output_matrix, &layer1_output_matrix);
    tanh_inplace(layer1_output, LAYER_1_ROWS);
    
    // Layer 2: hidden2 -> output (64 -> 4)
    dense_layer_cmsis_fast(&weight_matrices[2], biases_2,
                          &layer1_output_matrix, &layer2_output_matrix);
    
    // Copy result to actions (no final activation)
    memcpy(actions, layer2_output, OUTPUT_SIZE * sizeof(float));
}

void controllerRlInit(void)
{
  attitudeControllerInit(ATTITUDE_UPDATE_DT);
  positionControllerInit();

  // Read the weights from policy.h and load them into matricies for fast inference. 
  policyLoadWeights();

  DEBUG_PRINT("Controller RL initialized\n");
}

bool controllerRlTest(void)
{
  bool pass = true;

  pass &= attitudeControllerTest();

  return pass;
}

// Prepare observations for the neural network
static void prepare_observations(const sensorData_t *sensors, const state_t *state, 
                                const setpoint_t *setpoint, const float* previous_action) {
    // Note: You'll need to adjust this based on your training observation space
    // This is a template - modify according to your actual observation definition    
    int idx = 0;

    // Grab current state data into vectors
    struct vec pos_w = mkvec(state->position.x, state->position.y, state->position.z);
    struct vec vel_w = mkvec(state->velocity.x, state->velocity.y, state->velocity.z);
    struct quat q = mkquat(state->attitudeQuaternion.x, state->attitudeQuaternion.y, state->attitudeQuaternion.z, state->attitudeQuaternion.w);
    struct quat q_inv = qinv(q);
    // struct mat33 R = quat2rotmat(q);

    // Make goal data into vectors
    struct vec goal_pos = mkvec(setpoint->position.x, setpoint->position.y, setpoint->position.z);
    struct vec goal_vel = mkvec(0.0f, 0.0f, 0.0f); // Assuming no velocity setpoint for now
    struct quat q_des = rpy2quat(mkvec(0.0f, 0.0f, radians(setpoint->attitude.yaw)));  // Convert yaw to radians
    // struct mat33 R_des = quat2rotmat(q_des);
    
    // Position errors (3)
    struct vec pos_b = qvrot(q_inv, vsub(goal_pos, pos_w));
    observations[idx++] = pos_b.x; // Position error in body frame
    observations[idx++] = pos_b.y;
    observations[idx++] = pos_b.z;

    // Orientation Errors, as rotation matrix (9)
    struct quat ori_b_quat = qqmul(q_inv, q_des); // Quaternion representing the orientation error
    struct mat33 ori_b = quat2rotmat(ori_b_quat);
    
    observations[idx++] = ori_b.m[0][0]; // Fill the observation matrix with the rotation error matrix
    observations[idx++] = ori_b.m[0][1];
    observations[idx++] = ori_b.m[0][2];
    observations[idx++] = ori_b.m[1][0];
    observations[idx++] = ori_b.m[1][1];
    observations[idx++] = ori_b.m[1][2];
    observations[idx++] = ori_b.m[2][0];
    observations[idx++] = ori_b.m[2][1];
    observations[idx++] = ori_b.m[2][2];

    // Gravity Vector in Body Frame (3)
    struct vec gravity = mkvec(0.0f, 0.0f, -1.0f);
    struct vec gravity_body = qvrot(q_inv, gravity);
    observations[idx++] = gravity_body.x;
    observations[idx++] = gravity_body.y;
    observations[idx++] = gravity_body.z;
    
    // Linear Velocity in Body Frame (3)
    struct vec vel_b = qvrot(q_inv, vsub(goal_vel, vel_w));
    observations[idx++] = vel_b.x;
    observations[idx++] = vel_b.y;
    observations[idx++] = vel_b.z;

    // Angular Rates (already in body frame) (3)
    struct vec omega_b = mkvec(radians(sensors->gyro.x), -radians(sensors->gyro.y), radians(sensors->gyro.z));
    observations[idx++] = omega_b.x;
    observations[idx++] = omega_b.y;
    observations[idx++] = omega_b.z;

    // Add more observations as needed to reach INPUT_SIZE (21)
    // Example: previous actions, thrust command, etc.
    if (USE_PREVIOUS_ACTION) {
        for (int i = 0; i < ACTION_SIZE; i++) {
            observations[idx++] = previous_action[i]; // Previous actions
        }
    }

    for (int i = idx; i < OBS_SIZE; i++) {
        observations[i] = 0.0f;  // Pad with zeros if needed
    }
}

void controllerRl(control_t *control, const setpoint_t *setpoint,
                                         const sensorData_t *sensors,
                                         const state_t *state,
                                         const stabilizerStep_t stabilizerStep)
{
  if (first_call_time == 0) {
    first_call_time = usecTimestamp();
  }

  control->controlMode = controlModeLegacy;

  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, stabilizerStep)) {
    return;
  }

  // This is where we will call the RL controller and get the desired CTBR command
  if (RATE_DO_EXECUTE(POSITION_RATE, stabilizerStep)) {
    // Prepare observations for the neural network
    uint64_t pre_obs_timestamp = usecTimestamp();
    prepare_observations(sensors, state, setpoint, previous_action);
    uint64_t post_obs_timestamp = usecTimestamp();
    // Forward pass through the RL policy to get actions
    policyForwardRaw(observations, actions);
    
    uint64_t post_inference_timestamp = usecTimestamp();

    obs_time = (post_obs_timestamp - pre_obs_timestamp) * 0.001f; // Convert to milliseconds
    inference_time = (post_inference_timestamp - post_obs_timestamp) * 0.001f; // Convert to milliseconds

    // Actions are from [-1, 1]. Clip them and then scale to the control range
    for (int i = 0; i < ACTION_SIZE; i++) {
      actions[i] = clamp(actions[i], -1.0f, 1.0f); // Scale to [0, 1.0]
    }

    // Store the previous action for the next iteration
    for (int i = 0; i < ACTION_SIZE; i++) {
      previous_action[i] = actions[i];
    }

    // Overwrite actions based on time to get a step response in PITCH. 0.5s for -1.0, then 0.5s for +1.0, then back to 0.0
    // uint64_t current_time = usecTimestamp();
    // float elapsed_time = (current_time - first_call_time) * 1e-6f; // Convert to seconds
    // if (elapsed_time < 0.3f) {
    //     actions[0] = 0.25f; // Thrust
    //     actions[1] = -1.0f; // Roll rate
    //     actions[2] = 0.0f; // Pitch rate
    //     actions[3] = 0.0f; // Yaw rate
    // } else if (elapsed_time < 0.6f) {
    //     actions[0] = 0.25f; // Thrust
    //     actions[1] = 1.0f; // Roll rate
    //     actions[2] = 0.0f; // Pitch rate
    //     actions[3] = 0.0f; // Yaw rate
    // } else if (elapsed_time < 0.9f) {
    //     actions[0] = 0.25f; // Thrust
    //     actions[1] = 0.0f; // Roll rate
    //     actions[2] = 1.0f; // Pitch rate
    //     actions[3] = 0.0f; // Yaw rate
    // } else if (elapsed_time < 1.2f) {
    //     actions[0] = 0.25f; // Thrust
    //     actions[1] = 0.0f; // Roll rate
    //     actions[2] = -1.0f; // Pitch rate
    //     actions[3] = 0.0f; // Yaw rate
    // } else {
    //     actions[0] = 0.0f; // Thrust
    //     actions[1] = 0.0f; // Roll rate
    //     actions[2] = 0.0f; // Pitch rate
    //     actions[3] = 0.0f; // Yaw rate
    // }

    actuatorThrust = ((actions[0]+ 1.0f) / 2.0f) * UINT16_MAX; // Scale to thrust range
    control->thrust = actuatorThrust;
    // Set desired attitude rates based on actions
    rateDesired.roll = actions[1] * body_rate_scalar_xy;  // Assuming first action is roll rate in deg/s
    rateDesired.pitch = actions[2] * -body_rate_scalar_xy; // Assuming second action is pitch rate in deg/s NEGATED
    rateDesired.yaw = actions[3] * body_rate_scalar_z;  // Assuming third action is yaw rate in deg/s

    uint64_t final_timestamp = usecTimestamp();
    in_to_out_latency = (final_timestamp - pre_obs_timestamp) * 0.001f; // Convert to milliseconds
  }

  if (RATE_DO_EXECUTE(ATTITUDE_RATE, stabilizerStep)) {
    // TODO: Investigate possibility to subtract gyro drift.
    attitudeControllerCorrectRatePID(sensors->gyro.x, -sensors->gyro.y, sensors->gyro.z,
                             rateDesired.roll, rateDesired.pitch, rateDesired.yaw);

    attitudeControllerGetActuatorOutput(&control->roll,
                                        &control->pitch,
                                        &control->yaw);

    control->yaw = -control->yaw;

    cmd_thrust = control->thrust;
    cmd_roll = control->roll;
    cmd_pitch = control->pitch;
    cmd_yaw = control->yaw;
    r_roll = radians(sensors->gyro.x);
    r_pitch = -radians(sensors->gyro.y);
    r_yaw = radians(sensors->gyro.z);
    accelz = sensors->acc.z;
  }

  control->thrust = actuatorThrust;

  if (control->thrust == 0)
  {
    control->thrust = 0;
    control->roll = 0;
    control->pitch = 0;
    control->yaw = 0;

    cmd_thrust = control->thrust;
    cmd_roll = control->roll;
    cmd_pitch = control->pitch;
    cmd_yaw = control->yaw;

    attitudeControllerResetAllPID(state->attitude.roll, state->attitude.pitch, state->attitude.yaw);
    positionControllerResetAllPID(state->position.x, state->position.y, state->position.z);

    // Reset the calculated YAW angle for rate control
    attitudeDesired.yaw = state->attitude.yaw;
  }
}

PARAM_GROUP_START(ctrlRL)
/**
 * @brief Scaling factor for XY body rates
 */
PARAM_ADD(PARAM_FLOAT | PARAM_PERSISTENT_NOT_STORED, scale_xy, &body_rate_scalar_xy)
/**
 * @brief Scaling factor for Z body rates
 */
PARAM_ADD(PARAM_FLOAT | PARAM_PERSISTENT_NOT_STORED, scale_z, &body_rate_scalar_z)

PARAM_GROUP_STOP(ctrlRL)

/**
 * Logging variables for the command and reference signals for the
 * altitude PID controller
 */
LOG_GROUP_START(ctrlRL)
/**
 * @brief Thrust command
 */
LOG_ADD(LOG_FLOAT, cmd_thrust, &actuatorThrust)
/**
 * @brief Roll command
 */
LOG_ADD(LOG_FLOAT, cmd_roll, &rateDesired.roll)
/**
 * @brief Pitch command
 */
LOG_ADD(LOG_FLOAT, cmd_pitch, &rateDesired.pitch)
/**
 * @brief yaw command
 */
LOG_ADD(LOG_FLOAT, cmd_yaw, &rateDesired.yaw)

LOG_ADD(LOG_FLOAT, control_thrust, &cmd_thrust)
LOG_ADD(LOG_FLOAT, control_roll, &cmd_roll)
LOG_ADD(LOG_FLOAT, control_pitch, &cmd_pitch)
LOG_ADD(LOG_FLOAT, control_yaw, &cmd_yaw)

LOG_ADD(LOG_FLOAT, obs_0, &observations[0])
LOG_ADD(LOG_FLOAT, obs_1, &observations[1])
LOG_ADD(LOG_FLOAT, obs_2, &observations[2])
LOG_ADD(LOG_FLOAT, obs_3, &observations[3])
LOG_ADD(LOG_FLOAT, obs_4, &observations[4])
LOG_ADD(LOG_FLOAT, obs_5, &observations[5])
LOG_ADD(LOG_FLOAT, obs_6, &observations[6])
LOG_ADD(LOG_FLOAT, obs_7, &observations[7])
LOG_ADD(LOG_FLOAT, obs_8, &observations[8])
LOG_ADD(LOG_FLOAT, obs_9, &observations[9])
LOG_ADD(LOG_FLOAT, obs_10, &observations[10])
LOG_ADD(LOG_FLOAT, obs_11, &observations[11])
LOG_ADD(LOG_FLOAT, obs_12, &observations[12])
LOG_ADD(LOG_FLOAT, obs_13, &observations[13])
LOG_ADD(LOG_FLOAT, obs_14, &observations[14])
LOG_ADD(LOG_FLOAT, obs_15, &observations[15])
LOG_ADD(LOG_FLOAT, obs_16, &observations[16])
LOG_ADD(LOG_FLOAT, obs_17, &observations[17])
LOG_ADD(LOG_FLOAT, obs_18, &observations[18])
LOG_ADD(LOG_FLOAT, obs_19, &observations[19])
LOG_ADD(LOG_FLOAT, obs_20, &observations[20])

LOG_ADD(LOG_FLOAT, action_0, &actions[0]) // Thrust
LOG_ADD(LOG_FLOAT, action_1, &actions[1]) // Roll rate
LOG_ADD(LOG_FLOAT, action_2, &actions[2]) // Pitch rate
LOG_ADD(LOG_FLOAT, action_3, &actions[3]) // Yaw rate 

LOG_ADD(LOG_FLOAT, obs_time, &obs_time) // Time taken to prepare observations
LOG_ADD(LOG_FLOAT, inference_time, &inference_time) // Time taken for inference
LOG_ADD(LOG_FLOAT, in_out_ms, &in_to_out_latency) // Latency from input to output

LOG_GROUP_STOP(ctrlRL)

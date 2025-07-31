
#include "stabilizer_types.h"

#include "attitude_controller.h"
#include "position_controller.h"
#include "controller_rl.h"
#include "policy.h"

#include "debug.h"
#include "log.h"
#include "param.h"
#include "math3d.h"
#include "arm_math.h"

#define ATTITUDE_UPDATE_DT    (float)(1.0f/ATTITUDE_RATE)

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

// RL policy variables
#define OBS_SIZE INPUT_SIZE  // Use the size from policy.h
#define ACTION_SIZE OUTPUT_SIZE
static float observations[OBS_SIZE];
static float actions[ACTION_SIZE];

static float body_rate_scalar_xy = 10.0f; // Scale for XY body rates
static float body_rate_scalar_z = 10.0f;  // Scale for Z body

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
                                const setpoint_t *setpoint) {
    // Note: You'll need to adjust this based on your training observation space
    // This is a template - modify according to your actual observation definition
    
    int idx = 0;
    
    // Position errors (3)
    observations[idx++] = setpoint->position.x - state->position.x;
    observations[idx++] = setpoint->position.y - state->position.y;
    observations[idx++] = setpoint->position.z - state->position.z;
    
    // Orientation Errors, as rotation matrix (9)
    struct quat q = mkquat(state->attitudeQuaternion.x, state->attitudeQuaternion.y, state->attitudeQuaternion.z, state->attitudeQuaternion.w);
    struct mat33 R = quat2rotmat(q);
    struct quat q_des = mkquat(setpoint->attitudeQuaternion.x, setpoint->attitudeQuaternion.y, setpoint->attitudeQuaternion.z, setpoint->attitudeQuaternion.w);
    struct mat33 R_des = quat2rotmat(q_des);
    struct mat33 eRM = msub(mmul(mtranspose(R_des), R), mmul(mtranspose(R), R_des)); // rotation error matrix
    observations[idx++] = eRM.m[0][0];
    observations[idx++] = eRM.m[0][1];
    observations[idx++] = eRM.m[0][2];
    observations[idx++] = eRM.m[1][0];
    observations[idx++] = eRM.m[1][1];
    observations[idx++] = eRM.m[1][2];
    observations[idx++] = eRM.m[2][0];
    observations[idx++] = eRM.m[2][1];
    observations[idx++] = eRM.m[2][2];

    // Gravity Vector in Body Frame (3)
    struct vec gravity = mkvec(0.0f, 0.0f, -1.0f);
    struct vec gravity_body = mvmul(R, gravity);
    observations[idx++] = gravity_body.x;
    observations[idx++] = gravity_body.y;
    observations[idx++] = gravity_body.z;
    
    // Linear Velocity in Body Frame (3)
    struct vec velocity_body = mvmul(R, mkvec(state->velocity.x, state->velocity.y, state->velocity.z));
    observations[idx++] = velocity_body.x;
    observations[idx++] = velocity_body.y;
    observations[idx++] = velocity_body.z;
    
    // Angular Rates (already in body frame) (3)
    observations[idx++] = sensors->gyro.x;
    observations[idx++] = sensors->gyro.y;
    observations[idx++] = sensors->gyro.z;

    // Add more observations as needed to reach INPUT_SIZE (21)
    // Example: previous actions, thrust command, etc.
    for (int i = idx; i < OBS_SIZE; i++) {
        observations[i] = 0.0f;  // Pad with zeros if needed
    }
}

void controllerRl(control_t *control, const setpoint_t *setpoint,
                                         const sensorData_t *sensors,
                                         const state_t *state,
                                         const stabilizerStep_t stabilizerStep)
{
  control->controlMode = controlModeLegacy;

  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, stabilizerStep)) {
    return;
  }

  // This is where we weill call the RL controller and get the desired CTBR command
  if (RATE_DO_EXECUTE(POSITION_RATE, stabilizerStep)) {
    // Prepare observations for the neural network
    prepare_observations(sensors, state, setpoint);

    // Forward pass through the RL policy to get actions
    policyForwardRaw(observations, actions);

    // Actions are from [-1, 1]. Clip them and then scale to the control range
    for (int i = 0; i < ACTION_SIZE; i++) {
      actions[i] = (clamp(actions[i], -1.0f, 1.0f) + 1.0f) / 2.0f; // Scale to [0, 1.0]
    }

    actuatorThrust = actions[0] * UINT16_MAX; // Scale to thrust range
    rateDesired.roll = actions[1] * body_rate_scalar_xy;  // Assuming first action is roll rate
    rateDesired.pitch = actions[2] * body_rate_scalar_xy; // Assuming second action is pitch rate
    rateDesired.yaw = actions[3] * body_rate_scalar_z;  // Assuming third action is yaw rate
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

// PARAM_GROUP_START(ctrlRL)
// PARAM_ADD(PARAM_FLOAT, body_rate_scalar_xy, &body_rate_scalar_xy)
// PARAM_ADD(PARAM_FLOAT, body_rate_scalar_z, &body_rate_scalar_z)
// PARAM_GROUP_STOP(ctrlRL)

/**
 * Logging variables for the command and reference signals for the
 * altitude PID controller
 */
LOG_GROUP_START(ctrlRL)
/**
 * @brief Thrust command
 */
LOG_ADD(LOG_FLOAT, cmd_thrust, &cmd_thrust)
/**
 * @brief Roll command
 */
LOG_ADD(LOG_FLOAT, cmd_roll, &cmd_roll)
/**
 * @brief Pitch command
 */
LOG_ADD(LOG_FLOAT, cmd_pitch, &cmd_pitch)
/**
 * @brief yaw command
 */
LOG_ADD(LOG_FLOAT, cmd_yaw, &cmd_yaw)
/**
 * @brief Gyro roll measurement in radians
 */
LOG_ADD(LOG_FLOAT, r_roll, &r_roll)
/**
 * @brief Gyro pitch measurement in radians
 */
LOG_ADD(LOG_FLOAT, r_pitch, &r_pitch)
/**
 * @brief Yaw  measurement in radians
 */
LOG_ADD(LOG_FLOAT, r_yaw, &r_yaw)
/**
 * @brief Acceleration in the zaxis in G-force
 */
LOG_ADD(LOG_FLOAT, accelz, &accelz)
/**
 * @brief Thrust command without (tilt)compensation
 */
LOG_ADD(LOG_FLOAT, actuatorThrust, &actuatorThrust)
/**
 * @brief Desired roll setpoint
 */
LOG_ADD(LOG_FLOAT, roll,      &attitudeDesired.roll)
/**
 * @brief Desired pitch setpoint
 */
LOG_ADD(LOG_FLOAT, pitch,     &attitudeDesired.pitch)
/**
 * @brief Desired yaw setpoint
 */
LOG_ADD(LOG_FLOAT, yaw,       &attitudeDesired.yaw)
/**
 * @brief Desired roll rate setpoint
 */
LOG_ADD(LOG_FLOAT, rollRate,  &rateDesired.roll)
/**
 * @brief Desired pitch rate setpoint
 */
LOG_ADD(LOG_FLOAT, pitchRate, &rateDesired.pitch)
/**
 * @brief Desired yaw rate setpoint
 */
LOG_ADD(LOG_FLOAT, yawRate,   &rateDesired.yaw)
LOG_GROUP_STOP(ctrlRL)

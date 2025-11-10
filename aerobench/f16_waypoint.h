/* F16 Waypoint Environment - Header-only C implementation
 * Single-waypoint continuous task with physics-based F-16 simulation
 * Integrates f16_model.h and raylib_renderer.h - all C, no Python needed
 */

#ifndef F16_WAYPOINT_H
#define F16_WAYPOINT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "f16_model.h"
#include "raylib_renderer.h"

// State indices
#define VT 0
#define ALPHA 1
#define BETA 2
#define PHI 3
#define THETA 4
#define PSI 5
#define P 6
#define Q 7
#define R 8
#define POSN 9
#define POSE 10
#define ALT 11
#define POW 12
#define NUM_STATE_VARS 13

// Physics bounds - alpha (angle of attack) and velocity limits
#define ALPHA_MIN -2.0f
#define ALPHA_MAX 2.0f
#define VELOCITY_MIN 200.0f
#define VELOCITY_MAX 3000.0f

// World bounds - defines the playable box (east/north ±bounds, altitude 0 to max)
#define WORLD_BOUNDS_E 18000.0f
#define WORLD_BOUNDS_N 24000.0f
#define WORLD_BOUNDS_ALT 15000.0f

// Waypoint generation parameters
#define WAYPOINT_DISTANCE_MIN 12000.0f  // Min distance from jet when generating waypoint
#define WAYPOINT_DISTANCE_MAX 13000.0f  // Max distance from jet when generating waypoint
#define WAYPOINT_ALT_MIN 2000.0f        // Min altitude for waypoint generation
#define WAYPOINT_ALT_MAX 4000.0f        // Max altitude for waypoint generation
#define WAYPOINT_WORLDBOUND_MARGIN 3000.0f  // Min distance waypoints must keep from side walls and ceiling (not floor)
#define WAYPOINT_CAPTURE_RADIUS 500.0f  // Distance to capture waypoint

#define PI 3.14159265358979323846f

// Required Log struct for PufferLib
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

// F16 Waypoint environment struct
typedef struct {
    // State (16 element: 13 base + 3 integrators for LQR)
    double state[16];
    double waypoint[3];  // [east, north, altitude]
    double time;
    double step_size;
    double time_limit;
    
    // Control reference (high-level autopilot commands)
    double u_ref[4];  // [Nz, ps, Ny_r, throttle]
    double u_ref_prev[4];  // Previous control reference for smoothness penalty
    
    // Extended state outputs
    double xd[16];
    double u_deg[7];  // [throttle, elevator, aileron, rudder, Nz_ref, ps_ref, Ny_r_ref]
    double Nz, ps, Ny_r;
    
    // Episode tracking
    int tick;
    unsigned int seed;
    
    // Reward (computed using comprehensive reward function)
    double reward;
    
    // Renderer state
    RenderState render_state;
} F16Waypoint;

// Utility: Random double in range [min, max]
static inline double randd(double min, double max) {
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

// Utility: Wrap angle to [-pi, pi]
static inline double wrap_to_pi(double angle) {
    double result = fmod(angle, 2.0 * PI);
    if (result > PI) result -= 2.0 * PI;
    if (result < -PI) result += 2.0 * PI;
    return result;
}

// Utility: Gaussian reward function
static inline double gaussian_reward(double x, double sigma) {
    return exp(-0.5 * (x / sigma) * (x / sigma));
}

// Utility: Cartesian to spherical coordinates
static void cart2sph(double x, double y, double z, double* az, double* elev, double* r) {
    double h = sqrt(x * x + y * y);
    *r = sqrt(h * h + z * z);
    *elev = atan2(z, h);
    *az = atan2(y, x);
}

// Get waypoint observation in spherical coordinates (no scaling)
static void get_waypoint_obs_noscale_sph(const double* state, const double* waypoint, 
                                         double* delta_az, double* delta_elev, double* r) {
    // Delta waypoint vector
    double delta_e = waypoint[0] - state[POSE];
    double delta_n = waypoint[1] - state[POSN];
    double delta_alt = waypoint[2] - state[ALT];
    
    // Convert to spherical coordinates
    double az, elev, range;
    cart2sph(delta_e, delta_n, delta_alt, &az, &elev, &range);
    
    // Calculate azimuth error (heading error)
    double psi = state[PSI];
    *delta_az = wrap_to_pi(PI / 2.0 - az) - wrap_to_pi(psi);
    *delta_az = wrap_to_pi(*delta_az);
    
    // Zero out azimuth if directly above/below (within 10 ft horizontal distance)
    double horiz_dist = sqrt(delta_e * delta_e + delta_n * delta_n);
    if (horiz_dist <= 10.0) {
        *delta_az = 0.0;
    }
    
    // Calculate elevation error (pitch error)
    double theta = state[THETA];
    *delta_elev = wrap_to_pi(elev) - wrap_to_pi(theta);
    *delta_elev = wrap_to_pi(*delta_elev);
    
    *r = range;
}

// Compute waypoint reward using spherical coordinates
static double get_waypoint_reward_sph(const double* state, const double* waypoint) {
    double delta_az, delta_elev, r;
    get_waypoint_obs_noscale_sph(state, waypoint, &delta_az, &delta_elev, &r);
    
    // Gaussian rewards for alignment and proximity
    double rew_az = gaussian_reward(delta_az, 30.0 / 180.0 * PI);    // 30 degrees std
    double rew_el = gaussian_reward(delta_elev, 30.0 / 180.0 * PI);  // 30 degrees std
    double rew_rr = gaussian_reward(r, 1000.0);                       // 1000 ft std
    
    return rew_rr + 0.1 * (rew_az + rew_el);
}

// Compute comprehensive reward (matching JAX implementation)
static double compute_reward(F16Waypoint* env, const double u_ref[4], const double u_ref_prev[4]) {
    double reward = 0.0;
    
    // 1. Waypoint proximity reward (spherical coordinates)
    double reward_waypoint = 2.0 * get_waypoint_reward_sph(env->state, env->waypoint);
    
    // Scale by velocity Gaussian (encourage 500 ft/s cruise speed)
    double velocity_error = env->state[VT] - 500.0;
    reward_waypoint *= gaussian_reward(velocity_error, 100.0);
    
    // 2. Style/smoothness penalties
    double phi = env->state[PHI];
    double p = env->state[P];
    double q = env->state[Q];
    double r = env->state[R];
    
    double reward_roll = -5.0e-3 * (phi * phi);
    double reward_rollrate = -5.0e-3 * (p * p);
    double reward_pitchrate = -5.0e-3 * (q * q);
    double reward_yawrate = -5.0e-3 * (r * r);
    
    // 3. Velocity regulation reward
    double reward_velocity = 2.0e-2 * gaussian_reward(velocity_error, 50.0);
    
    // 4. Action magnitude penalty (L2 norm of actions)
    double action_mag_sq = 0.0;
    for (int i = 0; i < 3; i++) {  // Only first 3 actions (Nz, ps, throttle)
        action_mag_sq += u_ref[i] * u_ref[i];
    }
    double reward_actionmag = -1.0e-2 * (action_mag_sq / 3.0);
    
    // 5. Action smoothness penalty (difference from previous action)
    double action_delta_sq = 0.0;
    for (int i = 0; i < 3; i++) {
        double delta = u_ref_prev[i] - u_ref[i];
        action_delta_sq += delta * delta;
    }
    double reward_actiondlt = -1.0e-1 * (action_delta_sq / 3.0);
    
    // 6. Alive bonus
    double reward_alive = 1.0e-2;
    
    // Total reward
    reward = reward_waypoint + reward_roll + reward_rollrate + reward_pitchrate + 
             reward_yawrate + reward_velocity + reward_actionmag + reward_actiondlt + 
             reward_alive;
    
    // Clip reward to minimum of -0.2
    if (reward < -0.2) {
        reward = -0.2;
    }
    
    return reward;
}

// Generate initial state within world bounds
static void generate_initial_state(double* state) {
    memset(state, 0, 16 * sizeof(double));
    state[VT] = 540.0;
    state[ALPHA] = 0.03706505;//deg2rad(2.1215);
    state[THETA] = 0.03706505;//deg2rad(2.1215);
    state[ALT] = 1500.0;
    state[POW] = 9.0;
    
    // Randomize starting position within world bounds
    state[POSE] = randd(-WORLD_BOUNDS_E * 0.5, WORLD_BOUNDS_E * 0.5);
    state[POSN] = randd(-WORLD_BOUNDS_N * 0.5, WORLD_BOUNDS_N * 0.5);
}

// Generate random waypoint within world bounds with margin
static void generate_waypoint(F16Waypoint* env) {
    // Ensure waypoint generation is always feasible
    static int asserts_checked = 0;
    if (!asserts_checked) {
        // Verify waypoint altitude range fits within world bounds with margin from ceiling
        if (!(WAYPOINT_ALT_MIN >= 0 && WAYPOINT_ALT_MAX <= (WORLD_BOUNDS_ALT - WAYPOINT_WORLDBOUND_MARGIN))) {
            fprintf(stderr, "ASSERT FAILED: Waypoint altitude range [%.1f, %.1f] must fit within [0, %.1f] (world bound %.1f - margin %.1f)\n",
                    WAYPOINT_ALT_MIN, WAYPOINT_ALT_MAX, WORLD_BOUNDS_ALT - WAYPOINT_WORLDBOUND_MARGIN, 
                    WORLD_BOUNDS_ALT, WAYPOINT_WORLDBOUND_MARGIN);
            exit(1);
        }
        // Verify world bounds are positive and large enough for margins
        if (!(WORLD_BOUNDS_E > WAYPOINT_WORLDBOUND_MARGIN && 
              WORLD_BOUNDS_N > WAYPOINT_WORLDBOUND_MARGIN && 
              WORLD_BOUNDS_ALT > WAYPOINT_WORLDBOUND_MARGIN)) {
            fprintf(stderr, "ASSERT FAILED: World bounds (E=%.1f, N=%.1f, ALT=%.1f) must be > margin (%.1f)\n",
                    WORLD_BOUNDS_E, WORLD_BOUNDS_N, WORLD_BOUNDS_ALT, WAYPOINT_WORLDBOUND_MARGIN);
            exit(1);
        }
        // Verify waypoint distance constraints are achievable within world bounds with margins
        double max_distance_from_center = sqrt(
            (WORLD_BOUNDS_E - WAYPOINT_WORLDBOUND_MARGIN) * (WORLD_BOUNDS_E - WAYPOINT_WORLDBOUND_MARGIN) + 
            (WORLD_BOUNDS_N - WAYPOINT_WORLDBOUND_MARGIN) * (WORLD_BOUNDS_N - WAYPOINT_WORLDBOUND_MARGIN)
        );
        if (!(WAYPOINT_DISTANCE_MAX <= max_distance_from_center)) {
            fprintf(stderr, "ASSERT FAILED: WAYPOINT_DISTANCE_MAX (%.1f) must be <= diagonal of bounded area (%.1f)\n",
                    WAYPOINT_DISTANCE_MAX, max_distance_from_center);
            exit(1);
        }
        asserts_checked = 1;
    }
    
    double pos_e = env->state[POSE];
    double pos_n = env->state[POSN];
    
    // Generate waypoint at desired distance from current position
    double distance = randd(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX);
    double angle = randd(0.0, 2.0 * PI);
    
    double target_e = pos_e + distance * cos(angle);
    double target_n = pos_n + distance * sin(angle);
    
    // Clamp to world bounds with margin (not applied to floor at altitude 0)
    env->waypoint[0] = fmax(-WORLD_BOUNDS_E + WAYPOINT_WORLDBOUND_MARGIN, 
                            fmin(WORLD_BOUNDS_E - WAYPOINT_WORLDBOUND_MARGIN, target_e));
    env->waypoint[1] = fmax(-WORLD_BOUNDS_N + WAYPOINT_WORLDBOUND_MARGIN, 
                            fmin(WORLD_BOUNDS_N - WAYPOINT_WORLDBOUND_MARGIN, target_n));
    env->waypoint[2] = randd(WAYPOINT_ALT_MIN, WAYPOINT_ALT_MAX);
}

// Check physics bounds and world bounds
static int check_physics_violation(const double* state) {
    // Physics bounds
    if (state[ALPHA] <= ALPHA_MIN || state[ALPHA] >= ALPHA_MAX) return 1;
    if (state[VT] < VELOCITY_MIN || state[VT] > VELOCITY_MAX) return 1;
    
    // World bounds
    if (state[POSE] < -WORLD_BOUNDS_E || state[POSE] > WORLD_BOUNDS_E) return 1;
    if (state[POSN] < -WORLD_BOUNDS_N || state[POSN] > WORLD_BOUNDS_N) return 1;
    if (state[ALT] < 0 || state[ALT] > WORLD_BOUNDS_ALT) return 1;
    
    return 0;
}

// Check waypoint capture
static int check_waypoint_captured(F16Waypoint* env) {
    double dx = env->state[POSE] - env->waypoint[0];
    double dy = env->state[POSN] - env->waypoint[1];
    double dz = env->state[ALT] - env->waypoint[2];
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    return distance < WAYPOINT_CAPTURE_RADIUS;
}

// Euler integration step using actual F16 model
static void euler_step(F16Waypoint* env) {
    // Call the controlled F-16 model from f16_model.h
    controlled_f16(env->state, env->u_ref, env->xd, env->u_deg, 
                   &env->Nz, &env->ps, &env->Ny_r);
    
    // Integrate using simple Euler method
    for (int i = 0; i < 16; i++) {
        env->state[i] += env->xd[i] * env->step_size;
    }
    env->time += env->step_size;
}

// Update render state from environment state
static void update_render_state(F16Waypoint* env) {
    env->render_state.time_sec = (float)env->time;
    env->render_state.speed_fps = (float)env->state[VT];
    env->render_state.alpha_rad = (float)env->state[ALPHA];
    env->render_state.beta_rad = (float)env->state[BETA];
    env->render_state.phi_rad = (float)env->state[PHI];
    env->render_state.theta_rad = (float)env->state[THETA];
    env->render_state.psi_rad = (float)env->state[PSI];
    env->render_state.pos_e = (float)env->state[POSE];
    env->render_state.pos_n = (float)env->state[POSN];
    env->render_state.altitude = (float)env->state[ALT];
    env->render_state.nz_g = (float)env->Nz;
    env->render_state.ps_rad_s = (float)env->ps;
    env->render_state.waypoint_e = (float)env->waypoint[0];
    env->render_state.waypoint_n = (float)env->waypoint[1];
    env->render_state.waypoint_alt = (float)env->waypoint[2];
    env->render_state.waypoint_radius = (float)WAYPOINT_CAPTURE_RADIUS;
    env->render_state.world_bounds_e_min = -WORLD_BOUNDS_E;
    env->render_state.world_bounds_e_max = WORLD_BOUNDS_E;
    env->render_state.world_bounds_n_min = -WORLD_BOUNDS_N;
    env->render_state.world_bounds_n_max = WORLD_BOUNDS_N;
    env->render_state.world_bounds_alt_max = WORLD_BOUNDS_ALT;
    env->render_state.reward = (float)env->reward;
}

// Required: Reset environment
void f16_waypoint_reset(F16Waypoint* env, int keep_position) {
    if (!keep_position) {
        // Full reset to initial state
        generate_initial_state(env->state);
    }
    // else keep current state (waypoint captured successfully)
    
    env->time = 0.0;
    env->tick = 0;
    env->reward = 0.0;
    
    // Generate new waypoint
    generate_waypoint(env);
    
    // Initialize control reference to zero
    memset(env->u_ref, 0, 4 * sizeof(double));
    memset(env->u_ref_prev, 0, 4 * sizeof(double));
    
    // Update render state
    update_render_state(env);
}

// Required: Step environment
// Returns: 0=continue, 1=terminated (success), 2=terminated (physics violation), 3=truncated (time limit)
int f16_waypoint_step(F16Waypoint* env, const double u_ref[4]) {
    env->tick += 1;
    
    // Compute reward using comprehensive reward function
    env->reward = compute_reward(env, u_ref, env->u_ref_prev);
    
    // Store previous action for next step
    memcpy(env->u_ref_prev, env->u_ref, 4 * sizeof(double));
    
    // Copy control reference
    memcpy(env->u_ref, u_ref, 4 * sizeof(double));
    
    // Perform integration step using actual F16 model
    euler_step(env);
    
    // Check physics violation
    if (check_physics_violation(env->state)) {
        return 2;  // Physics violation
    }
    
    // Check time limit
    if (env->time >= env->time_limit) {
        return 3;  // Truncated
    }
    
    // Check waypoint capture
    if (check_waypoint_captured(env)) {
        return 1;  // Success
    }
    
    // Update render state
    update_render_state(env);
    
    return 0;  // Continue
}

// Required: Render environment
void f16_waypoint_render(F16Waypoint* env) {
    raylib_renderer_render(&env->render_state);
}

// Required: Check if window should close
int f16_waypoint_should_close(void) {
    return raylib_renderer_should_close() ? 1 : 0;
}

// Required: Close environment
void f16_waypoint_close(void) {
    raylib_renderer_close();
}

// Clear trail only (called when episode ends but continuing)
void f16_waypoint_clear_trail(void) {
    raylib_renderer_clear_trail();
}

#endif // F16_WAYPOINT_H

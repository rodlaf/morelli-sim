/* F16 Waypoint Environment - Header-only C implementation
 * Single-waypoint continuous task with physics-based F-16 simulation
 * Integrates f16_model.h and raylib_renderer.h - all C, no Python needed
 */

#ifndef F16_WAYPOINT_H
#define F16_WAYPOINT_H

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

// Physics bounds
#define ALPHA_MIN -2.0f
#define ALPHA_MAX 2.0f
#define VELOCITY_MIN 200.0f
#define VELOCITY_MAX 3000.0f
#define ALTITUDE_MIN -10000.0f
#define ALTITUDE_MAX 100000.0f

// World bounds
#define WORLD_BOUNDS_E 20000.0f
#define WORLD_BOUNDS_N 20000.0f

// Waypoint parameters
#define WAYPOINT_DISTANCE_MIN 12000.0f
#define WAYPOINT_DISTANCE_MAX 13000.0f
#define WAYPOINT_ALT_MIN 2000.0f
#define WAYPOINT_ALT_MAX 4000.0f
#define WAYPOINT_CAPTURE_RADIUS 500.0f

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
    
    // Extended state outputs
    double xd[16];
    double u_deg[7];  // [throttle, elevator, aileron, rudder, Nz_ref, ps_ref, Ny_r_ref]
    double Nz, ps, Ny_r;
    
    // Episode tracking
    int tick;
    unsigned int seed;
    
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

// Generate initial state
static void generate_initial_state(double* state) {
    memset(state, 0, 16 * sizeof(double));
    state[VT] = 540.0;
    state[ALPHA] = 0.03706505;//deg2rad(2.1215);
    state[THETA] = 0.03706505;//deg2rad(2.1215);
    state[ALT] = 1500.0;
    state[POW] = 9.0;
}

// Generate random waypoint from current position
static void generate_waypoint(F16Waypoint* env) {
    double pos_e = env->state[POSE];
    double pos_n = env->state[POSN];
    
    double distance = randd(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX);
    double angle = randd(0.0, 2.0 * PI);
    
    env->waypoint[0] = pos_e + distance * cos(angle);
    env->waypoint[1] = pos_n + distance * sin(angle);
    env->waypoint[2] = randd(WAYPOINT_ALT_MIN, WAYPOINT_ALT_MAX);
}

// Check physics bounds
static int check_physics_violation(const double* state) {
    if (state[ALPHA] <= ALPHA_MIN || state[ALPHA] >= ALPHA_MAX) return 1;
    if (state[VT] < VELOCITY_MIN || state[VT] > VELOCITY_MAX) return 1;
    if (state[ALT] <= ALTITUDE_MIN || state[ALT] >= ALTITUDE_MAX) return 1;
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
    env->render_state.world_bounds_alt_max = ALTITUDE_MAX;
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
    
    // Generate new waypoint
    generate_waypoint(env);
    
    // Initialize control reference to zero
    memset(env->u_ref, 0, 4 * sizeof(double));
    
    // Update render state
    update_render_state(env);
}

// Required: Step environment
// Returns: 0=continue, 1=terminated (success), 2=terminated (physics violation), 3=truncated (time limit)
int f16_waypoint_step(F16Waypoint* env, const double u_ref[4]) {
    env->tick += 1;
    
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

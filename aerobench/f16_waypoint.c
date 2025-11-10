/* F-16 Waypoint Demo - Standalone C version
 * Compile: gcc -o f16_waypoint f16_waypoint.c $(pkg-config --cflags --libs raylib) -lm -O3
 * Run: ./f16_waypoint
 */

#include "f16_waypoint.h"
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

/* Get wall-clock time in seconds */
static double get_wall_time(void) {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

/* Autopilot configuration */
typedef struct {
    double waypoint[3];
    
    /* Gains */
    double cfg_k_vt;
    double cfg_airspeed;
    double cfg_k_alt;
    double cfg_k_h_dot;
    double cfg_k_prop_psi;
    double cfg_k_der_psi;
    double cfg_k_prop_phi;
    double cfg_k_der_phi;
    double cfg_max_bank_deg;
    double cfg_max_nz_cmd;
    double cfg_min_nz_cmd;
} Autopilot;

/* Initialize autopilot with waypoint */
void autopilot_init(Autopilot* ap, const double waypoint[3]) {
    ap->waypoint[0] = waypoint[0];
    ap->waypoint[1] = waypoint[1];
    ap->waypoint[2] = waypoint[2];
    
    ap->cfg_k_vt = 0.25;
    ap->cfg_airspeed = 550.0;
    ap->cfg_k_alt = 0.005;
    ap->cfg_k_h_dot = 0.02;
    ap->cfg_k_prop_psi = 5.0;
    ap->cfg_k_der_psi = 0.5;
    ap->cfg_k_prop_phi = 0.75;
    ap->cfg_k_der_phi = 0.5;
    ap->cfg_max_bank_deg = 65.0;
    ap->cfg_max_nz_cmd = 4.0;
    ap->cfg_min_nz_cmd = -1.0;
}

/* Update autopilot waypoint */
void autopilot_update_waypoint(Autopilot* ap, const double waypoint[3]) {
    ap->waypoint[0] = waypoint[0];
    ap->waypoint[1] = waypoint[1];
    ap->waypoint[2] = waypoint[2];
}

/* Get heading to waypoint */
static double get_waypoint_heading(Autopilot* ap, const double* state) {
    double e_pos = state[POSE];
    double n_pos = state[POSN];
    double delta_e = ap->waypoint[0] - e_pos;
    double delta_n = ap->waypoint[1] - n_pos;
    return wrap_to_pi(PI/2.0 - atan2(delta_n, delta_e));
}

/* Get path angle gamma */
static double get_path_angle(const double* state) {
    double alpha = state[ALPHA];
    double beta = state[BETA];
    double phi = state[PHI];
    double theta = state[THETA];
    
    return asin((cos(alpha)*sin(theta) - 
                 sin(alpha)*cos(theta)*cos(phi))*cos(beta) - 
                (cos(theta)*sin(phi))*sin(beta));
}

/* Track altitude wings level */
static double track_altitude_wings_level(Autopilot* ap, const double* state) {
    double h_cmd = ap->waypoint[2];
    double vt = state[VT];
    double h = state[ALT];
    double h_error = h_cmd - h;
    double gamma = get_path_angle(state);
    double h_dot = vt * sin(gamma);
    return ap->cfg_k_alt * h_error - ap->cfg_k_h_dot * h_dot;
}

/* Get Nz for level turn */
static double get_nz_for_level_turn(const double* state) {
    double phi = state[PHI];
    if (fabs(phi) > 1e-6) {
        return 1.0 / cos(phi) - 1.0;
    }
    return 0.0;
}

/* Track altitude */
static double track_altitude(Autopilot* ap, const double* state) {
    double h_cmd = ap->waypoint[2];
    double h = state[ALT];
    double phi = state[PHI];
    double h_error = h_cmd - h;
    double nz_alt = track_altitude_wings_level(ap, state);
    double nz_roll = get_nz_for_level_turn(state);
    
    if (h_error > 0) {
        return nz_alt + nz_roll;
    } else if (fabs(phi) < 15.0 * (PI / 180.0)) {
        return nz_alt + nz_roll;
    } else {
        return fmax(0.0, nz_alt + nz_roll);
    }
}

/* Get action from autopilot */
void autopilot_get_action(Autopilot* ap, const double* state, double u_ref[4]) {
    /* Get desired heading to waypoint */
    double psi_cmd = get_waypoint_heading(ap, state);
    
    /* PD Control on heading using roll */
    double psi = wrap_to_pi(state[PSI]);
    double r = state[R];
    double psi_err = wrap_to_pi(psi_cmd - psi);
    double phi_cmd = psi_err * ap->cfg_k_prop_psi - r * ap->cfg_k_der_psi;
    
    /* Bound bank angle */
    double max_bank_rad = ap->cfg_max_bank_deg * (PI / 180.0);
    phi_cmd = clamp(phi_cmd, -max_bank_rad, max_bank_rad);
    
    /* PD control on roll angle */
    double phi = state[PHI];
    double p = state[P];
    double ps_cmd = (phi_cmd - phi) * ap->cfg_k_prop_phi - p * ap->cfg_k_der_phi;
    
    /* Track altitude */
    double nz_cmd = track_altitude(ap, state);
    nz_cmd = clamp(nz_cmd, ap->cfg_min_nz_cmd, ap->cfg_max_nz_cmd);
    
    /* Track airspeed */
    double throttle = ap->cfg_k_vt * (ap->cfg_airspeed - state[VT]);
    
    /* Set control */
    u_ref[0] = nz_cmd;
    u_ref[1] = ps_cmd;
    u_ref[2] = 0.0;  /* Ny_r */
    u_ref[3] = throttle;
}

/* Main simulation loop */
int main(int argc, char** argv) {
    printf("F-16 Waypoint Demo (Standalone C)\n");
    printf("==================================\n\n");
    
    /* Create environment */
    F16Waypoint env;
    env.step_size = 1.0 / 30.0;
    env.time_limit = 100.0;
    env.seed = (unsigned int)time(NULL);
    srand(env.seed);
    
    /* Reset environment */
    f16_waypoint_reset(&env, 0);
    
    /* Create autopilot */
    Autopilot autopilot;
    autopilot_init(&autopilot, env.waypoint);
    
    printf("Starting simulation...\n");
    printf("Waypoint: E=%.1f N=%.1f Alt=%.1f\n\n",
           env.waypoint[0], env.waypoint[1], env.waypoint[2]);
    printf("Press ESC to exit\n\n");
    
    /* Timing variables */
    double playback_speed = 1;
    int render_fps = 60;
    double target_frame_time = 1.0 / render_fps;
    double last_render_time = get_wall_time();
    double last_step_time = get_wall_time();
    int waypoint_count = 1;
    
    /* Main loop */
    f16_waypoint_render(&env);
    
    while (!f16_waypoint_should_close()) {
        double current_time = get_wall_time();
        
        /* Step simulation at playback speed (uncapped) */
        double real_time_since_step = current_time - last_step_time;
        double target_sim_time = env.time + (real_time_since_step * playback_speed);
        
        /* Run simulation steps to catch up */
        while (env.time < target_sim_time) {
            double u_ref[4];
            autopilot_get_action(&autopilot, env.state, u_ref);
            
            int result = f16_waypoint_step(&env, u_ref);
            
            if (result != 0) {
                /* Episode ended */
                int is_success = (result == 1);
                
                f16_waypoint_clear_trail();
                f16_waypoint_reset(&env, is_success);
                autopilot_update_waypoint(&autopilot, env.waypoint);
                
                last_step_time = get_wall_time();
                
                if (is_success) {
                    waypoint_count++;
                    printf("Waypoint %d reached!\n", waypoint_count - 1);
                    printf("New waypoint %d: E=%.1f N=%.1f Alt=%.1f\n\n",
                           waypoint_count, env.waypoint[0], env.waypoint[1], env.waypoint[2]);
                } else {
                    printf("Physics violation! Resetting...\n\n");
                    waypoint_count = 1;
                }
                break;
            }
        }
        
        last_step_time = current_time;
        
        /* Render at fixed FPS (frame skipping happens automatically) */
        double time_since_render = current_time - last_render_time;
        if (time_since_render >= target_frame_time) {
            f16_waypoint_render(&env);
            last_render_time = current_time;
        }
        
        /* Small sleep to avoid busy waiting */
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 1000000;  /* 1ms */
        nanosleep(&ts, NULL);
    }
    
    f16_waypoint_close();
    printf("\nSimulation closed\n");
    
    return 0;
}

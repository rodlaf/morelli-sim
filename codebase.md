# Codebase

## Structure

```
- .describeignore
- .gitignore
- README.md
- code/
  - aerobench/
    - examples/
      - anim3d/
        - run_all.sh
        - run_u_turn_anim3d.py
      - straight_and_level/
        - run.py
      - waypoint/
        - run_u_turn.py
        - run_waypoint.py
        - waypoint_autopilot.py
    - highlevel/
      - autopilot.py
      - controlled_f16.py
    - lowlevel/
      - adc.py
      - cl.py
      - cm.py
      - cn.py
      - cx.py
      - cy.py
      - cz.py
      - dampp.py
      - dlda.py
      - dldr.py
      - dnda.py
      - dndr.py
      - low_level_controller.py
      - morellif16.py
      - pdot.py
      - rtau.py
      - subf16_model.py
      - tgear.py
      - thrust.py
    - run_f16_sim.py
    - util.py
    - visualize/
      - anim.py
      - anim3d.py
      - bak_matplotlib.mlpstyle
      - plot.py
- requirements.txt
```

## Files

### .describeignore

```
.git/
dist/
codebase.md

*.gif
*.mat
*.onnx

LICENSE
.pylintrc
```

### .gitignore

```
*~
*.pyc
TAGS
*.png
*.gif
*.so
*.mp4
*Debug/
\#*\#
.\#*
*_flymake.py
*.mypy_cache
__pycache__/
*.pkl
.venv/
```

### README.md

```
﻿
Objective: C++ Morelli Sim

---

<p align="center"> <img src="anim3d.gif"/> </p>

Note: This is the v2 branch of the code, which is now a python3 project and includes more modularity and general simulation capabilities. For the original benchmark paper version see the v1 branch.

# AeroBenchVVPython Overview
This project contains a python version of models and controllers that test automated aircraft maneuvers by performing simulations. The hope is to provide a benchmark to motivate better verification and analysis methods, working beyond models based on Dubins car dynamics, towards the sorts of models used in aerospace engineering. Roughly speaking, the dynamics are nonlinear, have about 10-20 dimensions (continuous state variables), and hybrid in the sense of discontinuous ODEs, but not with jumps in the state. 

This is a python port of the original matlab version, which can can see for
more information: https://github.com/pheidlauf/AeroBenchVV

# Citation

For citation purposes, please use: "Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers", P. Heidlauf, A. Collins, M. Bolender, S. Bak, 5th International Workshop on Applied Verification for Continuous and Hybrid Systems (ARCH 2018)

# Required Libraries 

The following Python libraries are required (can be installed using `sudo pip install <library>`):

`numpy` - for matrix operations

`scipy` - for simulation / numerical integration (RK45) and trim condition optimization 

`matplotlib` - for animation / plotting (requires `ffmpeg` for .mp4 output or `imagemagick` for .gif)

`slycot` - for control design (not needed for simulation)

`control` - for control design (not needed for simulation)

### Animation isuses
Use matplotlib version 3.1.1 if you get errors like: 
"art3d.py", line 175, in set_3d_properties
    zs = np.broadcast_to(zs, xs.shape)
AttributeError: 'list' object has no attribute 'shape'

### Release Documentation
Distribution A: Approved for Public Release (88ABW-2020-2188) (changes in this version)

Distribution A: Approved for Public Release (88ABW-2017-6379) (v1)

```

### code/aerobench/examples/anim3d/run_all.sh

```
#!/bin/bash

time python3 run_combined_anim3d.py combined.gif
time python3 run_combined_anim3d.py combined.mp4

time python3 run_GCAS_anim3d.py gcas.mp4
time python3 run_GCAS_anim3d.py gcas.gif

time python3 run_u_turn_anim3d.py u_turn.mp4
time python3 run_u_turn_anim3d.py u_turn.gif

```

### code/aerobench/examples/anim3d/run_u_turn_anim3d.py

```
'''
Stanley Bak

plots 3d animation for 'u_turn' scenario 
'''

import sys

import numpy as np
from numpy import deg2rad

import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import anim3d, plot
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot

def simulate(filename):
    'simulate the system, returning waypoints, res'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1500        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = 0           # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 150 # simulation time

    # make waypoint list
    waypoints = [[-5000, -7500, alt],
                 [-15000, -7500, alt-500],
                 [-15000, 5000, alt-200]]

    ap = WaypointAutopilot(waypoints, stdout=True)

    step = 1/30
    extended_states = True
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')

    print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    if filename.endswith('.mp4'):
        skip_override = 4
    elif filename.endswith('.gif'):
        skip_override = 15
    else:
        skip_override = 30

    anim_lines = []
    modes = res['modes']
    modes = modes[0::skip_override]

    def init_extra(ax):
        'initialize plot extra shapes'

        l1, = ax.plot([], [], [], 'bo', ms=8, lw=0, zorder=50)
        anim_lines.append(l1)

        l2, = ax.plot([], [], [], 'lime', marker='o', ms=8, lw=0, zorder=50)
        anim_lines.append(l2)

        return anim_lines

    def update_extra(frame):
        'update plot extra shapes'

        mode_names = ['Waypoint 1', 'Waypoint 2', 'Waypoint 3']

        done_xs = []
        done_ys = []
        done_zs = []

        blue_xs = []
        blue_ys = []
        blue_zs = []

        for i, mode_name in enumerate(mode_names):
            if modes[frame] == mode_name:
                blue_xs.append(waypoints[i][0])
                blue_ys.append(waypoints[i][1])
                blue_zs.append(waypoints[i][2])
                break

            done_xs.append(waypoints[i][0])
            done_ys.append(waypoints[i][1])
            done_zs.append(waypoints[i][2])

        anim_lines[0].set_data(blue_xs, blue_ys)
        anim_lines[0].set_3d_properties(blue_zs)

        anim_lines[1].set_data(done_xs, done_ys)
        anim_lines[1].set_3d_properties(done_zs)

    return res, init_extra, update_extra, skip_override, waypoints

def main():
    'main function'

    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")

    res, init_extra, update_extra, skip_override, waypoints = simulate(filename)

    plot.plot_single(res, 'alt', title='Altitude (ft)')
    alt_filename = 'waypoint_altitude.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

    plot.plot_overhead(res, waypoints=waypoints)
    overhead_filename = 'waypoint_overhead.png'
    plt.savefig(overhead_filename)
    print(f"Made {overhead_filename}")
    plt.close()
        
    anim3d.make_anim(res, filename, f16_scale=70, viewsize=5000, viewsize_z=4000, trail_pts=np.inf,
                     elev=27, azim=-107, skip_frames=skip_override,
                     chase=True, fixed_floor=True, init_extra=init_extra, update_extra=update_extra)

if __name__ == '__main__':
    main()

```

### code/aerobench/examples/straight_and_level/run.py

```
'''
Stanley Bak

Straight and level flight example
'''

from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim
from aerobench.highlevel.autopilot import Autopilot
from aerobench.util import StateIndex

from aerobench.visualize import plot

class StraightAndLevelAutopilot(Autopilot):
    '''Simple Autopilot with proportional control'''

    def __init__(self, init):
        self.alt_setpoint = init[StateIndex.ALT]
        self.vel_setpoint = init[StateIndex.VEL]

        Autopilot.__init__(self, 'init_mode')

    def get_u_ref(self, _t, x_f16):
        '''get the reference inputs signals'''

        airspeed = x_f16[0]   # Vt            (ft/sec)
        alpha = x_f16[1]      # AoA           (rad)
        theta = x_f16[4]      # Pitch angle   (rad)
        gamma = theta - alpha # Path angle    (rad)
        h = x_f16[11]         # Altitude      (feet)

        # Proportional Control
        k_alt = 0.01
        h_error = self.alt_setpoint - h
        Nz = k_alt * h_error # Allows stacking of cmds

        # (Psuedo) Derivative control using path angle
        k_gamma = 15
        Nz = Nz - k_gamma*gamma

        # try to maintain a fixed airspeed near start point
        K_vt = 0.5
        throttle = -K_vt * (airspeed - self.vel_setpoint)

        Nz = min(max(-1, Nz), 6)

        return Nz, 0, 0, throttle

def main():
    'main function'

    ### Initial Conditions ###
    power = 7.6 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(1.8) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 3600        # altitude (ft)
    vt = 550          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0.03         # Pitch angle from nose level (rad)
    psi = 0       # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 50 # simulation time
    
    ap = StraightAndLevelAutopilot(init)

    res = run_f16_sim(init, tmax, ap)

    print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")

    plot.plot_overhead(res)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")
    
    plot.plot_single(res, 'alt', title='Altitude (ft)')
    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_single(res, 'vt', title='Velocity (ft/sec)')
    filename = 'vel.png'
    plt.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    main()

```

### code/aerobench/examples/waypoint/run_u_turn.py

```
'''
Stanley Bak

should match 'u_turn' scenario from matlab version
'''

from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim

from aerobench.visualize import plot

from waypoint_autopilot import WaypointAutopilot

def main():
    'main function'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1500        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = 0           # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 150 # simulation time

    # make waypoint list
    waypoints = [[-5000, -7500, alt],
                 [-15000, -7500, alt],
                 [-20000, 0, alt+500]]

    ap = WaypointAutopilot(waypoints, stdout=True)

    step = 1/30
    extended_states = True
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')
    
    print(f"Simulation Completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    plot.plot_single(res, 'alt', title='Altitude (ft)')
    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_overhead(res, waypoints=waypoints)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_attitude(res)
    filename = 'attitude.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot inner loop controls + references
    plot.plot_inner_loop(res)
    filename = 'inner_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot outer loop controls + references
    plot.plot_outer_loop(res)
    filename = 'outer_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    main()

```

### code/aerobench/examples/waypoint/run_waypoint.py

```
'''
Stanley Bak

should match 'waypoint' scenario from matlab version
'''

import math

from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim

from aerobench.visualize import plot

from .waypoint_autopilot import WaypointAutopilot

def main():
    'main function'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 3800        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = math.pi/8   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 75 # simulation time

    # make waypoint list
    e_pt = 1000
    n_pt = 3000
    h_pt = 4000

    waypoints = [[e_pt, n_pt, h_pt],
                 [e_pt + 2000, n_pt + 5000, h_pt - 100],
                 [e_pt - 2000, n_pt + 15000, h_pt - 250],
                 [e_pt - 500, n_pt + 25000, 300]]

    ap = WaypointAutopilot(waypoints, stdout=True)

    step = 1/30
    extended_states = True
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')

    print(f"Simulation Completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    plot.plot_single(res, 'alt', title='Altitude (ft)')
    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_overhead(res, waypoints=waypoints)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_attitude(res)
    filename = 'attitude.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot inner loop controls + references
    plot.plot_inner_loop(res)
    filename = 'inner_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot outer loop controls + references
    plot.plot_outer_loop(res)
    filename = 'outer_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    main()

```

### code/aerobench/examples/waypoint/waypoint_autopilot.py

```
'''waypoint autopilot

ported from matlab v2
'''

from math import pi, atan2, sqrt, sin, cos, asin

import numpy as np

from aerobench.highlevel.autopilot import Autopilot
from aerobench.util import StateIndex
from aerobench.lowlevel.low_level_controller import LowLevelController

class WaypointAutopilot(Autopilot):
    '''waypoint follower autopilot'''

    def __init__(self, waypoints, gain_str='old', stdout=False):
        'waypoints is a list of 3-tuples'

        self.stdout = stdout
        self.waypoints = waypoints
        self.waypoint_index = 0

        # waypoint config
        self.cfg_slant_range_threshold = 250

        # default control when not waypoint tracking
        self.cfg_u_ol_default = (0, 0, 0, 0.3)

        # control config
        # Gains for speed control
        self.cfg_k_vt = 0.25
        self.cfg_airspeed = 550

        # Gains for altitude tracking
        self.cfg_k_alt = 0.005
        self.cfg_k_h_dot = 0.02

        # Gains for heading tracking
        self.cfg_k_prop_psi = 5
        self.cfg_k_der_psi = 0.5

        # Gains for roll tracking
        self.cfg_k_prop_phi = 0.75
        self.cfg_k_der_phi = 0.5
        self.cfg_max_bank_deg = 65 # maximum bank angle setpoint
        # v2 was 0.5, 0.9

        # Ranges for Nz
        self.cfg_max_nz_cmd = 4
        self.cfg_min_nz_cmd = -1

        self.done_time = 0.0

        llc = LowLevelController(gain_str=gain_str)

        Autopilot.__init__(self, 'Waypoint 1', llc=llc)

    def log(self, s):
        'print to terminal if stdout is true'

        if self.stdout:
            print(s)

    def get_u_ref(self, _t, x_f16):
        '''get the reference input signals'''

        if self.mode != "Done":
            psi_cmd = self.get_waypoint_data(x_f16)[0]

            # Get desired roll angle given desired heading
            phi_cmd = self.get_phi_to_track_heading(x_f16, psi_cmd)
            ps_cmd = self.track_roll_angle(x_f16, phi_cmd)

            nz_cmd = self.track_altitude(x_f16)
            throttle = self.track_airspeed(x_f16)
        else:
           # Waypoint Following complete: fly level.
            throttle = self.track_airspeed(x_f16)
            ps_cmd = self.track_roll_angle(x_f16, 0)
            nz_cmd = self.track_altitude_wings_level(x_f16)

        # trim to limits
        nz_cmd = max(self.cfg_min_nz_cmd, min(self.cfg_max_nz_cmd, nz_cmd))
        #throttle = max(min(throttle, 1), 0) # do not limit throttle as negative throttle is valid as it gets passed through high/low level controllers

        # Create reference vector
        rv = [nz_cmd, ps_cmd, 0, throttle]

        return rv

    def track_altitude(self, x_f16):
        'get nz to track altitude, taking turning into account'

        h_cmd = self.waypoints[self.waypoint_index][2]

        h = x_f16[StateIndex.ALT]
        phi = x_f16[StateIndex.PHI]

        # Calculate altitude error (positive => below target alt)
        h_error = h_cmd - h
        nz_alt = self.track_altitude_wings_level(x_f16)
        nz_roll = get_nz_for_level_turn_ol(x_f16)

        if h_error > 0:
            # Ascend wings level or banked
            nz = nz_alt + nz_roll
        elif abs(phi) < np.deg2rad(15):
            # Descend wings (close enough to) level
            nz = nz_alt + nz_roll
        else:
            # Descend in bank (no negative Gs)
            nz = max(0, nz_alt + nz_roll)

        return nz

    def get_phi_to_track_heading(self, x_f16, psi_cmd):
        'get phi from psi_cmd'

        # PD Control on heading angle using phi_cmd as control

        # Pull out important variables for ease of use
        psi = wrap_to_pi(x_f16[StateIndex.PSI])
        r = x_f16[StateIndex.R]

        # Calculate PD control
        psi_err = wrap_to_pi(psi_cmd - psi)

        phi_cmd = psi_err * self.cfg_k_prop_psi - r * self.cfg_k_der_psi

        # Bound to acceptable bank angles:
        max_bank_rad = np.deg2rad(self.cfg_max_bank_deg)

        phi_cmd = min(max(phi_cmd, -max_bank_rad), max_bank_rad)

        return phi_cmd

    def track_roll_angle(self, x_f16, phi_cmd):
        'get roll angle command (ps_cmd)'

        # PD control on roll angle using stability roll rate

        # Pull out important variables for ease of use
        phi = x_f16[StateIndex.PHI]
        p = x_f16[StateIndex.P]

        # Calculate PD control
        ps = (phi_cmd-phi) * self.cfg_k_prop_phi - p * self.cfg_k_der_phi

        return ps

    def track_airspeed(self, x_f16):
        'get throttle command'

        vt_cmd = self.cfg_airspeed

        # Proportional control on airspeed using throttle
        throttle = self.cfg_k_vt * (vt_cmd - x_f16[StateIndex.VT])

        return throttle

    def track_altitude_wings_level(self, x_f16):
        'get nz to track altitude'

        i = self.waypoint_index if self.waypoint_index < len(self.waypoints) else -1

        h_cmd = self.waypoints[i][2]

        vt = x_f16[StateIndex.VT]
        h = x_f16[StateIndex.ALT]

        # Proportional-Derivative Control
        h_error = h_cmd - h
        gamma = get_path_angle(x_f16)
        h_dot = vt * sin(gamma) # Calculated, not differentiated

        # Calculate Nz command
        nz = self.cfg_k_alt*h_error - self.cfg_k_h_dot*h_dot

        return nz

    def is_finished(self, t, x_f16):
        'is the maneuver done?'

        rv = self.waypoint_index >= len(self.waypoints) and self.done_time + 5.0 < t

        return rv

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete state based on the current aircraft state. Returns True iff the discrete state
        has changed.
        '''

        if self.waypoint_index < len(self.waypoints):
            slant_range = self.get_waypoint_data(x_f16)[-1]

            if slant_range < self.cfg_slant_range_threshold:
                self.waypoint_index += 1

                if self.waypoint_index >= len(self.waypoints):
                    self.done_time = t

        premode = self.mode

        if self.waypoint_index >= len(self.waypoints):
            self.mode = 'Done'
        else:
            self.mode = f'Waypoint {self.waypoint_index + 1}'

        rv = premode != self.mode

        if rv:
            self.log(f"Waypoint transition {premode} -> {self.mode} at time {t}")

        return rv

    def get_waypoint_data(self, x_f16):
        '''returns current waypoint data tuple based on the current waypoint:

        (heading, inclination, horiz_range, vert_range, slant_range)

        heading = heading to tgt, equivalent to psi (rad)
        inclination = polar angle to tgt, equivalent to theta (rad)
        horiz_range = horizontal range to tgt (ft)
        vert_range = vertical range to tgt (ft)
        slant_range = total range to tgt (ft)
        '''

        waypoint = self.waypoints[self.waypoint_index]

        e_pos = x_f16[StateIndex.POSE]
        n_pos = x_f16[StateIndex.POSN]
        alt = x_f16[StateIndex.ALT]

        delta = [waypoint[i] - [e_pos, n_pos, alt][i] for i in range(3)]

        _, inclination, slant_range = cart2sph(delta)

        heading = wrap_to_pi(pi/2 - atan2(delta[1], delta[0]))

        horiz_range = np.linalg.norm(delta[0:2])
        vert_range = np.linalg.norm(delta[2])

        return heading, inclination, horiz_range, vert_range, slant_range

def get_nz_for_level_turn_ol(x_f16):
    'get nz to do a level turn'

    # Pull g's to maintain altitude during bank based on trig

    # Calculate theta
    phi = x_f16[StateIndex.PHI]

    if abs(phi): # if cos(phi) ~= 0, basically
        nz = 1 / cos(phi) - 1 # Keeps plane at altitude
    else:
        nz = 0

    return nz

def get_path_angle(x_f16):
    'get the path angle gamma'

    alpha = x_f16[StateIndex.ALPHA]       # AoA           (rad)
    beta = x_f16[StateIndex.BETA]         # Sideslip      (rad)
    phi = x_f16[StateIndex.PHI]           # Roll anle     (rad)
    theta = x_f16[StateIndex.THETA]       # Pitch angle   (rad)

    gamma = asin((cos(alpha)*sin(theta)- \
        sin(alpha)*cos(theta)*cos(phi))*cos(beta) - \
        (cos(theta)*sin(phi))*sin(beta))

    return gamma

def wrap_to_pi(psi_rad):
    '''handle angle wrapping

    returns equivelent angle in range [-pi, pi]
    '''

    rv = psi_rad % (2 * pi)

    if rv > pi:
        rv -= 2 * pi

    return rv

def cart2sph(pt3d):
    '''
    Cartesian to spherical coordinates

    returns az, elev, r
    '''

    x, y, z = pt3d

    h = sqrt(x*x + y*y)
    r = sqrt(h*h + z*z)

    elev = atan2(z, h)
    az = atan2(y, x)

    return az, elev, r

if __name__ == '__main__':
    print("Autopulot script not meant to be run directly.")

```

### code/aerobench/highlevel/autopilot.py

```
'''
Stanley Bak
Autopilot State-Machine Logic

There is a high-level advance_discrete_state() function, which checks if we should change the current discrete state,
and a get_u_ref(f16_state) function, which gets the reference inputs at the current discrete state.
'''

import abc
from math import pi

import numpy as np
from numpy import deg2rad

from aerobench.lowlevel.low_level_controller import LowLevelController
from aerobench.util import Freezable

class Autopilot(Freezable):
    '''A container object for the hybrid automaton logic for a particular autopilot instance'''

    def __init__(self, init_mode, llc=None):

        assert isinstance(init_mode, str), 'init_mode should be a string'

        if llc is None:
            # use default
            llc = LowLevelController()

        self.llc = llc
        self.xequil = llc.xequil
        self.uequil = llc.uequil
        
        self.mode = init_mode # discrete state, this should be overwritten by subclasses

        self.freeze_attrs()

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete mode based on the current aircraft state. Returns True iff the discrete mode
        has changed. It's also suggested to update self.mode to the current mode name.
        '''

        return False

    def is_finished(self, t, x_f16):
        '''
        returns True if the simulation should stop (for example, after maneuver completes)

        this is called after advance_discrete_state
        '''

        return False

    @abc.abstractmethod
    def get_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals. Override this one
        in subclasses.

        returns four values per aircraft: Nz, ps, Ny_r, throttle
        '''

        return

    def get_checked_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals and check them against ctrl limits
        '''

        rv = np.array(self.get_u_ref(t, x_f16), dtype=float)

        assert rv.size % 4 == 0, "get_u_ref should return Nz, ps, Ny_r, throttle for each aircraft"

        for i in range(rv.size //4):
            Nz, _ps, _Ny_r, _throttle = rv[4*i:4*(i+1)]

            l, u = self.llc.ctrlLimits.NzMin, self.llc.ctrlLimits.NzMax
            assert l <= Nz <= u, f"autopilot commanded invalid Nz ({Nz}). Not in range [{l}, {u}]"

        return rv

class FixedSpeedAutopilot(Autopilot):
    '''Simple Autopilot that gives a fixed speed command using proportional control'''

    def __init__(self, setpoint, p_gain):
        self.setpoint = setpoint
        self.p_gain = p_gain

        init_mode = 'tracking speed'
        Autopilot.__init__(self, init_mode)

    def get_u_ref(self, t, x_f16):
        '''for the current discrete state, get the reference inputs signals'''

        x_dif = self.setpoint - x_f16[0]

        return 0, 0, 0, self.p_gain * x_dif

```

### code/aerobench/highlevel/controlled_f16.py

```
'''
Stanley Bak
Python Version of F-16 GCAS
ODE derivative code (controlled F16)
'''

from math import sin, cos

import numpy as np
from numpy import deg2rad

from aerobench.lowlevel.subf16_model import subf16_model
from aerobench.lowlevel.low_level_controller import LowLevelController
from aerobench.util import StateIndex

def controlled_f16(t, x_f16, u_ref, llc, f16_model='morelli', v2_integrators=False):
    'returns the LQR-controlled F-16 state derivatives and more'

    assert isinstance(x_f16, np.ndarray)
    assert isinstance(llc, LowLevelController)
    assert u_ref.size == 4

    assert f16_model in ['stevens', 'morelli'], 'Unknown F16_model: {}'.format(f16_model)

    x_ctrl, u_deg = llc.get_u_deg(u_ref, x_f16)

    # Note: Control vector (u) for subF16 is in units of degrees
    xd_model, Nz, Ny, _, _ = subf16_model(x_f16[0:13], u_deg, f16_model)

    if v2_integrators:
        # integrators from matlab v2 model
        ps = xd_model[6] * cos(xd_model[1]) + xd_model[8] * sin(xd_model[1])

        Ny_r = Ny + xd_model[8]
    else:
        # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
        ps = x_ctrl[4] * cos(x_ctrl[0]) + x_ctrl[5] * sin(x_ctrl[0])

        # Calculate (side force + yaw rate) term
        Ny_r = Ny + x_ctrl[5]

    xd = np.zeros((x_f16.shape[0],))
    xd[:len(xd_model)] = xd_model

    # integrators from low-level controller
    start = len(xd_model)
    end = start + llc.get_num_integrators()
    int_der = llc.get_integrator_derivatives(t, x_f16, u_ref, Nz, ps, Ny_r)
    xd[start:end] = int_der

    # Convert all degree values to radians for output
    u_rad = np.zeros((7,)) # throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref

    u_rad[0] = u_deg[0] # throttle

    for i in range(1, 4):
        u_rad[i] = deg2rad(u_deg[i])

    u_rad[4:7] = u_ref[0:3] # inner-loop commands are 4-7

    return xd, u_rad, Nz, ps, Ny_r

```

### code/aerobench/lowlevel/adc.py

```
'''
Stanley Bak
adc.py for F-16 model
'''

from math import sqrt

def adc(vt, alt):
    '''converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure (qbar)

    See pages 63-65 of Stevens & Lewis, "Aircraft Control and Simulation", 2nd edition
    '''

    # vt = freestream air speed

    ro = 2.377e-3
    tfac = 1 - .703e-5 * alt

    if alt >= 35000: # in stratosphere
        t = 390
    else:
        t = 519 * tfac # 3 rankine per atmosphere (3 rankine per 1000 ft)

    # rho = freestream mass density
    rho = ro * tfac**4.14

    # a = speed of sound at the ambient conditions
    # speed of sound in a fluid is the sqrt of the quotient of the modulus of elasticity over the mass density
    a = sqrt(1.4 * 1716.3 * t)

    # amach = mach number
    amach = vt / a

    # qbar = dynamic pressure
    qbar = .5 * rho * vt * vt

    return amach, qbar

```

### code/aerobench/lowlevel/cl.py

```
'''
Stanley Bak
F-16 GCAS in Python
cl function
'''

import numpy as np

from aerobench.util import fix, sign

def cl(alpha, beta):
    'cl function'

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [-.001, -.004, -.008, -.012, -.016, -.022, -.022, -.021, -.015, -.008, -.013, -.015], \
    [-.003, -.009, -.017, -.024, -.030, -.041, -.045, -.040, -.016, -.002, -.010, -.019], \
    [-.001, -.010, -.020, -.030, -.039, -.054, -.057, -.054, -.023, -.006, -.014, -.027], \
    [.000, -.010, -.022, -.034, -.047, -.060, -.069, -.067, -.033, -.036, -.035, -.035], \
    [.007, -.010, -.023, -.034, -.049, -.063, -.081, -.079, -.060, -.058, -.062, -.059], \
    [.009, -.011, -.023, -.037, -.050, -.068, -.089, -.088, -.091, -.076, -.077, -.076]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .2 * abs(beta)
    m = fix(s)
    if m == 0:
        m = 1

    if m >= 6:
        m = 5

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 1
    n = n + 1
    t = a[k-1, m-1]
    u = a[k-1, n-1]
    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)
    dum = v + (w - v) * abs(db)

    return dum * sign(beta)

```

### code/aerobench/lowlevel/cm.py

```
'''
Stanley Bak
F-16 GCAS Python
'''

import numpy as np
from aerobench.util import fix, sign

def cm(alpha, el):
    'cm function'

    a = np.array([[.205, .168, .186, .196, .213, .251, .245, .238, .252, .231, .198, .192], \
        [.081, .077, .107, .110, .110, .141, .127, .119, .133, .108, .081, .093], \
        [-.046, -.020, -.009, -.005, -.006, .010, .006, -.001, .014, .000, -.013, .032], \
        [-.174, -.145, -.121, -.127, -.129, -.102, -.097, -.113, -.087, -.084, -.069, -.006], \
        [-.259, -.202, -.184, -.193, -.199, -.150, -.160, -.167, -.104, -.076, -.041, -.005]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = el / 12
    m = fix(s)

    if m <= -2:
        m = -1

    if m >= 2:
        m = 1

    de = s - m
    n = m + fix(1.1 * sign(de))
    k = k + 3
    l = l + 3
    m = m + 3
    n = n + 3
    t = a[k-1, m-1]
    u = a[k-1, n-1]
    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)

    return v + (w - v) * abs(de)


```

### code/aerobench/lowlevel/cn.py

```
'''
Stanley Bak
F16 GCAS in Python
cn function
'''

import numpy as np
from aerobench.util import fix, sign

def cn(alpha, beta):
    'cn function'

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [.018, .019, .018, .019, .019, .018, .013, .007, .004, -.014, -.017, -.033], \
        [.038, .042, .042, .042, .043, .039, .030, .017, .004, -.035, -.047, -.057], \
        [.056, .057, .059, .058, .058, .053, .032, .012, .002, -.046, -.071, -.073], \
        [.064, .077, .076, .074, .073, .057, .029, .007, .012, -.034, -.065, -.041], \
        [.074, .086, .093, .089, .080, .062, .049, .022, .028, -.012, -.002, -.013], \
        [.079, .090, .106, .106, .096, .080, .068, .030, .064, .015, .011, -.001]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .2 * abs(beta)
    m = fix(s)

    if m == 0:
        m = 1

    if m >= 6:
        m = 5

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 1
    n = n + 1
    t = a[k-1, m-1]
    u = a[k-1, n-1]

    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)
    dum = v + (w - v) * abs(db)

    return dum * sign(beta)

```

### code/aerobench/lowlevel/cx.py

```
'''
Stanley Bak
Python F-16 GCAS
cx
'''

import numpy as np

from aerobench.util import fix, sign

def cx(alpha, el):
    'cx definition'

    a = np.array([[-.099, -.081, -.081, -.063, -.025, .044, .097, .113, .145, .167, .174, .166], \
        [-.048, -.038, -.040, -.021, .016, .083, .127, .137, .162, .177, .179, .167], \
        [-.022, -.020, -.021, -.004, .032, .094, .128, .130, .154, .161, .155, .138], \
        [-.040, -.038, -.039, -.025, .006, .062, .087, .085, .100, .110, .104, .091], \
        [-.083, -.073, -.076, -.072, -.046, .012, .024, .025, .043, .053, .047, .040]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = el / 12
    m = fix(s)
    if m <= -2:
        m = -1

    if m >= 2:
        m = 1

    de = s - m
    n = m + fix(1.1 * sign(de))
    k = k + 3
    l = l + 3
    m = m + 3
    n = n + 3
    t = a[k-1, m-1]
    u = a[k-1, n-1]
    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)
    cxx = v + (w - v) * abs(de)

    return cxx

```

### code/aerobench/lowlevel/cy.py

```
'''
Stanley Bak
Python F-16 GCAS
'''

def cy(beta, ail, rdr):
    'cy function'

    return -.02 * beta + .021 * (ail / 20) + .086 * (rdr / 30)

```

### code/aerobench/lowlevel/cz.py

```
'''
Stanley Bak
Python F-16 GCAS
Cz function
'''

import numpy as np
from aerobench.util import fix, sign

def cz(alpha, beta, el):
    'cz function'

    a = np.array([.770, .241, -.100, -.415, -.731, -1.053, -1.355, -1.646, -1.917, -2.120, -2.248, -2.229], \
        dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    l = l + 3
    k = k + 3
    s = a[k-1] + abs(da) * (a[l-1] - a[k-1])

    return s * (1 - (beta / 57.3)**2) - .19 * (el / 25)

```

### code/aerobench/lowlevel/dampp.py

```
'''
Stanley Bak
F16 GCAS in Python
dampp function
'''

import numpy as np
from aerobench.util import fix, sign

def dampp(alpha):
    'dampp functon'

    a = np.array([[-.267, -.110, .308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.50, 1.49, 1.83, 1.21], \
        [.882, .852, .876, .958, .962, .974, .819, .483, .590, 1.21, -.493, -1.04], \
        [-.108, -.108, -.188, .110, .258, .226, .344, .362, .611, .529, .298, -2.27], \
        [-8.80, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2, -29.0, -29.8, -38.3, -35.3], \
        [-.126, -.026, .063, .113, .208, .230, .319, .437, .680, .100, .447, -.330], \
        [-.360, -.359, -.443, -.420, -.383, -.375, -.329, -.294, -.230, -.210, -.120, -.100], \
        [-7.21, -.540, -5.23, -5.26, -6.11, -6.64, -5.69, -6.00, -6.20, -6.40, -6.60, -6.00], \
        [-.380, -.363, -.378, -.386, -.370, -.453, -.550, -.582, -.595, -.637, -1.02, -.840], \
        [.061, .052, .052, -.012, -.013, -.024, .050, .150, .130, .158, .240, .150]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    k = k + 3
    l = l + 3

    d = np.zeros((9,))

    for i in range(9):
        d[i] = a[k-1, i] + abs(da) * (a[l-1, i] - a[k-1, i])

    return d

```

### code/aerobench/lowlevel/dlda.py

```
'''
Stanley Bak
F-16 GCAS in Python
dlda function
'''

import numpy as np
from aerobench.util import fix, sign

def dlda(alpha, beta):
    'dlda function'

    a = np.array([[-.041, -.052, -.053, -.056, -.050, -.056, -.082, -.059, -.042, -.038, -.027, -.017], \
    [-.041, -.053, -.053, -.053, -.050, -.051, -.066, -.043, -.038, -.027, -.023, -.016], \
    [-.042, -.053, -.052, -.051, -.049, -.049, -.043, -.035, -.026, -.016, -.018, -.014], \
    [-.040, -.052, -.051, -.052, -.048, -.048, -.042, -.037, -.031, -.026, -.017, -.012], \
    [-.043, -.049, -.048, -.049, -.043, -.042, -.042, -.036, -.025, -.021, -.016, -.011], \
    [-.044, -.048, -.048, -.047, -.042, -.041, -.020, -.028, -.013, -.014, -.011, -.010], \
    [-.043, -.049, -.047, -.045, -.042, -.037, -.003, -.013, -.010, -.003, -.007, -.008]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .1 * beta
    m = fix(s)
    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k-1, m-1]
    u = a[k-1, n-1]
    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)

    return v + (w - v) * abs(db)

```

### code/aerobench/lowlevel/dldr.py

```
'''
Stanley Bak
Python GCAS F - 16
dldr function
'''

import numpy as np
from aerobench.util import sign, fix

def dldr(alpha, beta):
    'dldr function'

    a = np.array([[.005, .017, .014, .010, -.005, .009, .019, .005, -.000, -.005, -.011, .008], \
    [.007, .016, .014, .014, .013, .009, .012, .005, .000, .004, .009, .007], \
    [.013, .013, .011, .012, .011, .009, .008, .005, -.002, .005, .003, .005], \
    [.018, .015, .015, .014, .014, .014, .014, .015, .013, .011, .006, .001], \
    [.015, .014, .013, .013, .012, .011, .011, .010, .008, .008, .007, .003], \
    [.021, .011, .010, .011, .010, .009, .008, .010, .006, .005, .000, .001], \
    [.023, .010, .011, .011, .011, .010, .008, .010, .006, .014, .020, .000]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .1 * beta
    m = fix(s)

    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k-1, m-1]
    u = a[k-1, n-1]

    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)

    return v + (w - v) * abs(db)

```

### code/aerobench/lowlevel/dnda.py

```
'''
Stanley Bak
F16 GCAS in Python
dnda function
'''

import numpy as np
from aerobench.util import fix, sign

def dnda(alpha, beta):
    'dnda function'

    a = np.array([[.001, -.027, -.017, -.013, -.012, -.016, .001, .017, .011, .017, .008, .016], \
        [.002, -.014, -.016, -.016, -.014, -.019, -.021, .002, .012, .016, .015, .011], \
        [-.006, -.008, -.006, -.006, -.005, -.008, -.005, .007, .004, .007, .006, .006], \
        [-.011, -.011, -.010, -.009, -.008, -.006, .000, .004, .007, .010, .004, .010], \
        [-.015, -.015, -.014, -.012, -.011, -.008, -.002, .002, .006, .012, .011, .011], \
        [-.024, -.010, -.004, -.002, -.001, .003, .014, .006, -.001, .004, .004, .006], \
        [-.022, .002, -.003, -.005, -.003, -.001, -.009, -.009, -.001, .003, -.002, .001]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .1 * beta
    m = fix(s)
    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k-1, m-1]
    u = a[k-1, n-1]
    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)

    return v + (w - v) * abs(db)

```

### code/aerobench/lowlevel/dndr.py

```
'''
Stanley Bak
F16 GCAS in Python
dndr function
'''

import numpy as np
from aerobench.util import fix, sign

def dndr(alpha, beta):
    'dndr function'

    a = np.array([[-.018, -.052, -.052, -.052, -.054, -.049, -.059, -.051, -.030, -.037, -.026, -.013], \
        [-.028, -.051, -.043, -.046, -.045, -.049, -.057, -.052, -.030, -.033, -.030, -.008], \
        [-.037, -.041, -.038, -.040, -.040, -.038, -.037, -.030, -.027, -.024, -.019, -.013], \
        [-.048, -.045, -.045, -.045, -.044, -.045, -.047, -.048, -.049, -.045, -.033, -.016], \
        [-.043, -.044, -.041, -.041, -.040, -.038, -.034, -.035, -.035, -.029, -.022, -.009], \
        [-.052, -.034, -.036, -.036, -.035, -.028, -.024, -.023, -.020, -.016, -.010, -.014], \
        [-.062, -.034, -.027, -.028, -.027, -.027, -.023, -.023, -.019, -.009, -.025, -.010]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .1 * beta
    m = fix(s)
    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k-1, m-1]
    u = a[k-1, n-1]
    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)
    return v + (w - v) * abs(db)

```

### code/aerobench/lowlevel/low_level_controller.py

```
'''
Stanley Bak
Low-level flight controller
'''

import numpy as np
from aerobench.util import Freezable

class CtrlLimits(Freezable):
    'Control Limits'

    def __init__(self):
        self.ThrottleMax = 1 # Afterburner on for throttle > 0.7
        self.ThrottleMin = 0
        self.ElevatorMaxDeg = 25
        self.ElevatorMinDeg = -25
        self.AileronMaxDeg = 21.5
        self.AileronMinDeg = -21.5
        self.RudderMaxDeg = 30
        self.RudderMinDeg = -30
        
        self.NzMax = 6
        self.NzMin = -1

        self.freeze_attrs()

class LowLevelController(Freezable):
    '''low level flight controller
    '''

    old_k_long = np.array([[-156.8801506723475, -31.037008068526642, -38.72983346216317]], dtype=float)
    old_k_lat = np.array([[37.84483, -25.40956, -6.82876, -332.88343, -17.15997],
                          [-23.91233, 5.69968, -21.63431, 64.49490, -88.36203]], dtype=float)

    old_xequil = np.array([502.0, 0.0389, 0.0, 0.0, 0.0389, 0.0, 0.0, 0.0, \
                        0.0, 0.0, 0.0, 1000.0, 9.0567], dtype=float).transpose()
    old_uequil = np.array([0.1395, -0.7496, 0.0, 0.0], dtype=float).transpose()

    def __init__(self, gain_str='old'):
        # Hard coded LQR gain matrix from matlab version

        assert gain_str == 'old'

        # Longitudinal Gains
        K_long = LowLevelController.old_k_long
        K_lat = LowLevelController.old_k_lat

        self.K_lqr = np.zeros((3, 8))
        self.K_lqr[:1, :3] = K_long
        self.K_lqr[1:, 3:] = K_lat

        # equilibrium points from BuildLqrControllers.py
        self.xequil = LowLevelController.old_xequil
        self.uequil = LowLevelController.old_uequil

        self.ctrlLimits = CtrlLimits()

        self.model_str = 'morelli'

        self.freeze_attrs()

    def get_u_deg(self, u_ref, f16_state):
        'get the reference commands for the control surfaces'

        # Calculate perturbation from trim state
        x_delta = f16_state.copy()
        x_delta[:len(self.xequil)] -= self.xequil

        ## Implement LQR Feedback Control
        # Reorder states to match controller:
        # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
        x_ctrl = np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=float)

        # Initialize control vectors
        u_deg = np.zeros((4,)) # throt, ele, ail, rud

        # Calculate control using LQR gains
        u_deg[1:4] = np.dot(-self.K_lqr, x_ctrl) # Full Control

        # Set throttle as directed from output of getOuterLoopCtrl(...)
        u_deg[0] = u_ref[3]

        # Add in equilibrium control
        u_deg[0:4] += self.uequil

        ## Limit controls to saturation limits
        ctrlLimits = self.ctrlLimits

        # Limit throttle from 0 to 1
        u_deg[0] = max(min(u_deg[0], ctrlLimits.ThrottleMax), ctrlLimits.ThrottleMin)

        # Limit elevator from -25 to 25 deg
        u_deg[1] = max(min(u_deg[1], ctrlLimits.ElevatorMaxDeg), ctrlLimits.ElevatorMinDeg)

        # Limit aileron from -21.5 to 21.5 deg
        u_deg[2] = max(min(u_deg[2], ctrlLimits.AileronMaxDeg), ctrlLimits.AileronMinDeg)

        # Limit rudder from -30 to 30 deg
        u_deg[3] = max(min(u_deg[3], ctrlLimits.RudderMaxDeg), ctrlLimits.RudderMinDeg)

        return x_ctrl, u_deg

    def get_num_integrators(self):
        'get the number of integrators in the low-level controller'

        return 3

    def get_integrator_derivatives(self, t, x_f16, u_ref, Nz, ps, Ny_r):
        'get the derivatives of the integrators in the low-level controller'

        return [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]

```

### code/aerobench/lowlevel/morellif16.py

```
'''
Stanley Bak
F16 GCAS in Python

Morelli dynamics (Polynomial interpolation)
'''

def Morellif16(alpha, beta, de, da, dr, p, q, r, cbar, b, V, xcg, xcgref):
    'desc'

    #alpha=max(-10*pi/180,min(45*pi/180,alpha)) # bounds alpha between -10 deg and 45 deg
    #beta = max( - 30 * pi / 180, min(30 * pi / 180, beta)) #bounds beta between -30 deg and 30 deg
    #de = max( - 25 * pi / 180, min(25 * pi / 180, de)) #bounds elevator deflection between -25 deg and 25 deg
    #da = max( - 21.5 * pi / 180, min(21.5 * pi / 180, da)) #bounds aileron deflection between -21.5 deg and 21.5 deg
    #dr = max( - 30 * pi / 180, min(30 * pi / 180, dr)) #bounds rudder deflection between -30 deg and 30 deg

    # xcgref = 0.35
    #reference longitudinal cg position in Morelli f16 model


    phat = p * b / (2 * V)
    qhat = q * cbar / (2 * V)
    rhat = r * b / (2 * V)
    ##
    a0 = -1.943367e-2
    a1 = 2.136104e-1
    a2 = -2.903457e-1
    a3 = -3.348641e-3
    a4 = -2.060504e-1
    a5 = 6.988016e-1
    a6 = -9.035381e-1

    b0 = 4.833383e-1
    b1 = 8.644627
    b2 = 1.131098e1
    b3 = -7.422961e1
    b4 = 6.075776e1

    c0 = -1.145916
    c1 = 6.016057e-2
    c2 = 1.642479e-1

    d0 = -1.006733e-1
    d1 = 8.679799e-1
    d2 = 4.260586
    d3 = -6.923267

    e0 = 8.071648e-1
    e1 = 1.189633e-1
    e2 = 4.177702
    e3 = -9.162236

    f0 = -1.378278e-1
    f1 = -4.211369
    f2 = 4.775187
    f3 = -1.026225e1
    f4 = 8.399763
    f5 = -4.354000e-1

    g0 = -3.054956e1
    g1 = -4.132305e1
    g2 = 3.292788e2
    g3 = -6.848038e2
    g4 = 4.080244e2

    h0 = -1.05853e-1
    h1 = -5.776677e-1
    h2 = -1.672435e-2
    h3 = 1.357256e-1
    h4 = 2.172952e-1
    h5 = 3.464156
    h6 = -2.835451
    h7 = -1.098104

    i0 = -4.126806e-1
    i1 = -1.189974e-1
    i2 = 1.247721
    i3 = -7.391132e-1

    j0 = 6.250437e-2
    j1 = 6.067723e-1
    j2 = -1.101964
    j3 = 9.100087
    j4 = -1.192672e1

    k0 = -1.463144e-1
    k1 = -4.07391e-2
    k2 = 3.253159e-2
    k3 = 4.851209e-1
    k4 = 2.978850e-1
    k5 = -3.746393e-1
    k6 = -3.213068e-1

    l0 = 2.635729e-2
    l1 = -2.192910e-2
    l2 = -3.152901e-3
    l3 = -5.817803e-2
    l4 = 4.516159e-1
    l5 = -4.928702e-1
    l6 = -1.579864e-2

    m0 = -2.029370e-2
    m1 = 4.660702e-2
    m2 = -6.012308e-1
    m3 = -8.062977e-2
    m4 = 8.320429e-2
    m5 = 5.018538e-1
    m6 = 6.378864e-1
    m7 = 4.226356e-1

    n0 = -5.19153
    n1 = -3.554716
    n2 = -3.598636e1
    n3 = 2.247355e2
    n4 = -4.120991e2
    n5 = 2.411750e2

    o0 = 2.993363e-1
    o1 = 6.594004e-2
    o2 = -2.003125e-1
    o3 = -6.233977e-2
    o4 = -2.107885
    o5 = 2.141420
    o6 = 8.476901e-1

    p0 = 2.677652e-2
    p1 = -3.298246e-1
    p2 = 1.926178e-1
    p3 = 4.013325
    p4 = -4.404302

    q0 = -3.698756e-1
    q1 = -1.167551e-1
    q2 = -7.641297e-1

    r0 = -3.348717e-2
    r1 = 4.276655e-2
    r2 = 6.573646e-3
    r3 = 3.535831e-1
    r4 = -1.373308
    r5 = 1.237582
    r6 = 2.302543e-1
    r7 = -2.512876e-1
    r8 = 1.588105e-1
    r9 = -5.199526e-1

    s0 = -8.115894e-2
    s1 = -1.156580e-2
    s2 = 2.514167e-2
    s3 = 2.038748e-1
    s4 = -3.337476e-1
    s5 = 1.004297e-1

    ##
    Cx0 = a0 + a1 * alpha + a2 * de**2 + a3 * de + a4 * alpha * de + a5 * alpha**2 + a6 * alpha**3
    Cxq = b0 + b1 * alpha + b2 * alpha**2 + b3 * alpha**3 + b4 * alpha**4
    Cy0 = c0 * beta + c1 * da + c2 * dr
    Cyp = d0 + d1 * alpha + d2 * alpha**2 + d3 * alpha**3
    Cyr = e0 + e1 * alpha + e2 * alpha**2 + e3 * alpha**3
    Cz0 = (f0 + f1 * alpha + f2 * alpha**2 + f3 * alpha**3 + f4 * alpha**4) * (1 - beta**2) + f5 * de
    Czq = g0 + g1 * alpha + g2 * alpha**2 + g3 * alpha**3 + g4 * alpha**4
    Cl0 = h0 * beta + h1 * alpha * beta + h2 * alpha**2 * beta + h3 * beta**2 + h4 * alpha * beta**2 + h5 * \
        alpha**3 * beta + h6 * alpha**4 * beta + h7 * alpha**2 * beta**2
    Clp = i0 + i1 * alpha + i2 * alpha**2 + i3 * alpha**3
    Clr = j0 + j1 * alpha + j2 * alpha**2 + j3 * alpha**3 + j4 * alpha**4
    Clda = k0 + k1 * alpha + k2 * beta + k3 * alpha**2 + k4 * alpha * beta + k5 * alpha**2 * beta + k6 * alpha**3
    Cldr = l0 + l1 * alpha + l2 * beta + l3 * alpha * beta + l4 * alpha**2 * beta + l5 * alpha**3 * beta + l6 * beta**2
    Cm0 = m0 + m1 * alpha + m2 * de + m3 * alpha * de + m4 * de**2 + m5 * alpha**2 * de + m6 * de**3 + m7 * \
        alpha * de**2


    Cmq = n0 + n1 * alpha + n2 * alpha**2 + n3 * alpha**3 + n4 * alpha**4 + n5 * alpha**5
    Cn0 = o0 * beta + o1 * alpha * beta + o2 * beta**2 + o3 * alpha * beta**2 + o4 * alpha**2 * beta + o5 * \
        alpha**2 * beta**2 + o6 * alpha**3 * beta
    Cnp = p0 + p1 * alpha + p2 * alpha**2 + p3 * alpha**3 + p4 * alpha**4
    Cnr = q0 + q1 * alpha + q2 * alpha**2
    Cnda = r0 + r1 * alpha + r2 * beta + r3 * alpha * beta + r4 * alpha**2 * beta + r5 * alpha**3 * beta + r6 * \
        alpha**2 + r7 * alpha**3 + r8 * beta**3 + r9 * alpha * beta**3
    Cndr = s0 + s1 * alpha + s2 * beta + s3 * alpha * beta + s4 * alpha**2 * beta + s5 * alpha**2
    ##

    Cx = Cx0 + Cxq * qhat
    Cy = Cy0 + Cyp * phat + Cyr * rhat
    Cz = Cz0 + Czq * qhat
    Cl = Cl0 + Clp * phat + Clr * rhat + Clda * da + Cldr * dr
    Cm = Cm0 + Cmq * qhat + Cz * (xcgref - xcg)
    Cn = Cn0 + Cnp * phat + Cnr * rhat + Cnda * da + Cndr * dr - Cy * (xcgref - xcg) * (cbar / b)

    return Cx, Cy, Cz, Cl, Cm, Cn

```

### code/aerobench/lowlevel/pdot.py

```
'''
Stanley Bak
Python F-16
power derivative (pdot)
'''

from aerobench.lowlevel.rtau import rtau

def pdot(p3, p1):
    'pdot function'

    if p1 >= 50:
        if p3 >= 50:
            t = 5
            p2 = p1
        else:
            p2 = 60
            t = rtau(p2 - p3)
    else:
        if p3 >= 50:
            t = 5
            p2 = 40
        else:
            p2 = p1
            t = rtau(p2 - p3)

    pd = t * (p2 - p3)

    return pd

```

### code/aerobench/lowlevel/rtau.py

```
'''
Stanley Bak
Python F-16

Rtau function
'''

def rtau(dp):
    'rtau function'

    if dp <= 25:
        rt = 1.0
    elif dp >= 50:
        rt = .1
    else:
        rt = 1.9 - .036 * dp

    return rt

```

### code/aerobench/lowlevel/subf16_model.py

```
'''
Stanley Bak
Python F-16 subf16
outputs aircraft state vector deriative
'''

#         x[0] = air speed, VT    (ft/sec)
#         x[1] = angle of attack, alpha  (rad)
#         x[2] = angle of sideslip, beta (rad)
#         x[3] = roll angle, phi  (rad)
#         x[4] = pitch angle, theta  (rad)
#         x[5] = yaw angle, psi  (rad)
#         x[6] = roll rate, P  (rad/sec)
#         x[7] = pitch rate, Q  (rad/sec)
#         x[8] = yaw rate, R  (rad/sec)
#         x[9] = northward horizontal displacement, pn  (feet)
#         x[10] = eastward horizontal displacement, pe  (feet)
#         x[11] = altitude, h  (feet)
#         x[12] = engine thrust dynamics lag state, pow
#
#         u[0] = throttle command  0.0 < u(1) < 1.0
#         u[1] = elevator command in degrees
#         u[2] = aileron command in degrees
#         u[3] = rudder command in degrees
#

from math import sin, cos, pi

from aerobench.lowlevel.adc import adc
from aerobench.lowlevel.tgear import tgear
from aerobench.lowlevel.pdot import pdot
from aerobench.lowlevel.thrust import thrust
from aerobench.lowlevel.cx import cx
from aerobench.lowlevel.cy import cy
from aerobench.lowlevel.cz import cz
from aerobench.lowlevel.cl import cl
from aerobench.lowlevel.dlda import dlda
from aerobench.lowlevel.dldr import dldr
from aerobench.lowlevel.cm import cm
from aerobench.lowlevel.cn import cn
from aerobench.lowlevel.dnda import dnda
from aerobench.lowlevel.dndr import dndr
from aerobench.lowlevel.dampp import dampp

from aerobench.lowlevel.morellif16 import Morellif16

def subf16_model(x, u, model, adjust_cy=True):
    '''output aircraft state vector derivative for a given input

    The reference for the model is Appendix A of Stevens & Lewis
    '''

    assert model in ['stevens', 'morelli']
    assert len(x) == 13
    assert len(u) == 4

    xcg = 0.35

    thtlc, el, ail, rdr = u

    s = 300
    b = 30
    cbar = 11.32
    rm = 1.57e-3
    xcgr = .35
    he = 160.0
    c1 = -.770
    c2 = .02755
    c3 = 1.055e-4
    c4 = 1.642e-6
    c5 = .9604
    c6 = 1.759e-2
    c7 = 1.792e-5
    c8 = -.7336
    c9 = 1.587e-5
    rtod = 57.29578
    g = 32.17

    xd = x.copy()
    vt = x[0]
    alpha = x[1]*rtod
    beta = x[2]*rtod
    phi = x[3]
    theta = x[4]
    psi = x[5]
    p = x[6]
    q = x[7]
    r = x[8]
    alt = x[11]
    power = x[12]

    # air data computer and engine model
    amach, qbar = adc(vt, alt)
    cpow = tgear(thtlc)

    xd[12] = pdot(power, cpow)

    t = thrust(power, alt, amach)
    dail = ail/20
    drdr = rdr/30

    # component build up

    if model == 'stevens':
        # stevens & lewis (look up table version)
        cxt = cx(alpha, el)
        cyt = cy(beta, ail, rdr)
        czt = cz(alpha, beta, el)

        clt = cl(alpha, beta) + dlda(alpha, beta) * dail + dldr(alpha, beta) * drdr
        cmt = cm(alpha, el)
        cnt = cn(alpha, beta) + dnda(alpha, beta) * dail + dndr(alpha, beta) * drdr
    else:
        # morelli model (polynomial version)
        cxt, cyt, czt, clt, cmt, cnt = Morellif16(alpha*pi/180, beta*pi/180, el*pi/180, ail*pi/180, rdr*pi/180, \
                                                  p, q, r, cbar, b, vt, xcg, xcgr)

    # add damping derivatives

    tvt = .5 / vt
    b2v = b * tvt
    cq = cbar * q * tvt

    # get ready for state equations
    d = dampp(alpha)
    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]
    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6] + czt * (xcgr-xcg)
    cnt = cnt + b2v * (d[7] * r + d[8] * p)-cyt * (xcgr-xcg) * cbar/b
    cbta = cos(x[2])
    u = vt * cos(x[1]) * cbta
    v = vt * sin(x[2])
    w = vt * sin(x[1]) * cbta
    sth = sin(theta)
    cth = cos(theta)
    sph = sin(phi)
    cph = cos(phi)
    spsi = sin(psi)
    cpsi = cos(psi)
    qs = qbar * s
    qsb = qs * b
    rmqs = rm * qs
    gcth = g * cth
    qsph = q * sph
    ay = rmqs * cyt
    az = rmqs * czt

    # force equations
    udot = r * v-q * w-g * sth + rm * (qs * cxt + t)
    vdot = p * w-r * u + gcth * sph + ay
    wdot = q * u-p * v + gcth * cph + az
    dum = (u * u + w * w)

    xd[0] = (u * udot + v * vdot + w * wdot)/vt
    xd[1] = (u * wdot-w * udot)/dum
    xd[2] = (vt * vdot-v * xd[0]) * cbta/dum

    # kinematics
    xd[3] = p + (sth/cth) * (qsph + r * cph)
    xd[4] = q * cph-r * sph
    xd[5] = (qsph + r * cph)/cth

    # moments
    xd[6] = (c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt)

    xd[7] = (c5 * p-c7 * he) * r + c6 * (r * r-p * p) + qs * cbar * c7 * cmt
    xd[8] = (c8 * p-c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt)

    # navigation
    t1 = sph * cpsi
    t2 = cph * sth
    t3 = sph * spsi
    s1 = cth * cpsi
    s2 = cth * spsi
    s3 = t1 * sth-cph * spsi
    s4 = t3 * sth + cph * cpsi
    s5 = sph * cth
    s6 = t2 * cpsi + t3
    s7 = t2 * spsi-t1
    s8 = cph * cth
    xd[9] = u * s1 + v * s3 + w * s6 # north speed
    xd[10] = u * s2 + v * s4 + w * s7 # east speed
    xd[11] = u * sth-v * s5-w * s8 # vertical speed

    # outputs

    xa = 15.0                  # sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
    az = az-xa * xd[7]         # moves normal accel in front of c.g.

    ####################################
    ###### peter additions below ######
    if adjust_cy:
        ay = ay+xa*xd[8]           # moves side accel in front of c.g.

    # For extraction of Nz
    Nz = (-az / g) - 1 # zeroed at 1 g, positive g = pulling up
    Ny = ay / g

    return xd, Nz, Ny, az, ay

```

### code/aerobench/lowlevel/tgear.py

```
'''
Stanley Bak
Python F-16 GCAS
'''

def tgear(thtl):
    'tgear function'

    if thtl <= .77:
        tg = 64.94 * thtl
    else:
        tg = 217.38 * thtl - 117.38

    return tg

```

### code/aerobench/lowlevel/thrust.py

```
'''
Stanle Bak
Python F-16
Thrust function
'''

import numpy as np

from aerobench.util import fix

def thrust(power, alt, rmach):
    'thrust lookup-table version'

    a = np.array([[1060, 670, 880, 1140, 1500, 1860], \
        [635, 425, 690, 1010, 1330, 1700], \
        [60, 25, 345, 755, 1130, 1525], \
        [-1020, -170, -300, 350, 910, 1360], \
        [-2700, -1900, -1300, -247, 600, 1100], \
        [-3600, -1400, -595, -342, -200, 700]], dtype=float).T

    b = np.array([[12680, 9150, 6200, 3950, 2450, 1400], \
        [12680, 9150, 6313, 4040, 2470, 1400], \
        [12610, 9312, 6610, 4290, 2600, 1560], \
        [12640, 9839, 7090, 4660, 2840, 1660], \
        [12390, 10176, 7750, 5320, 3250, 1930], \
        [11680, 9848, 8050, 6100, 3800, 2310]], dtype=float).T

    c = np.array([[20000, 15000, 10800, 7000, 4000, 2500], \
        [21420, 15700, 11225, 7323, 4435, 2600], \
        [22700, 16860, 12250, 8154, 5000, 2835], \
        [24240, 18910, 13760, 9285, 5700, 3215], \
        [26070, 21075, 15975, 11115, 6860, 3950], \
        [28886, 23319, 18300, 13484, 8642, 5057]], dtype=float).T

    if alt < 0:
        alt = 0.01 # uh, why not 0?

    h = .0001 * alt

    i = fix(h)

    if i >= 5:
        i = 4

    dh = h - i
    rm = 5 * rmach
    m = fix(rm)

    if m >= 5:
        m = 4
    elif m <= 0:
        m = 0

    dm = rm - m
    cdh = 1 - dh

    # do not increment these, since python is 0-indexed while matlab is 1-indexed
    #i = i + 1
    #m = m + 1

    s = b[i, m] * cdh + b[i + 1, m] * dh
    t = b[i, m + 1] * cdh + b[i + 1, m + 1] * dh
    tmil = s + (t - s) * dm

    if power < 50:
        s = a[i, m] * cdh + a[i + 1, m] * dh
        t = a[i, m + 1] * cdh + a[i + 1, m + 1] * dh
        tidl = s + (t - s) * dm
        thrst = tidl + (tmil - tidl) * power * .02
    else:
        s = c[i, m] * cdh + c[i + 1, m] * dh
        t = c[i, m + 1] * cdh + c[i + 1, m + 1] * dh
        tmax = s + (t - s) * dm
        thrst = tmil + (tmax - tmil) * (power - 50) * .02

    return thrst

```

### code/aerobench/run_f16_sim.py

```
'''
Stanley Bak
run_f16_sim python version
'''

import time

import numpy as np
from scipy.integrate import RK45

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler, StateIndex, print_state, Freezable

class F16SimState(Freezable):
    '''object containing simulation state

    With this interface you can run partial simulations, rather than having to simulate for the entire time bound

    if you just want a single run with a fixed time, it may be easier to use the run_f16_sim function
    '''

    def __init__(self, initial_state, ap, step=1/30, extended_states=False,
                integrator_str='rk45', v2_integrators=False, print_errors=True, keep_intermediate_states=True,
                 custom_stop_func=None):

        self.model_str = model_str = ap.llc.model_str
        self.v2_integrators = v2_integrators
        initial_state = np.array(initial_state, dtype=float)

        self.keep_intermediate_states = keep_intermediate_states
        self.custom_stop_func = custom_stop_func

        self.step = step
        self.ap = ap
        self.print_errors = print_errors

        llc = ap.llc

        num_vars = len(get_state_names()) + llc.get_num_integrators()

        if initial_state.size < num_vars:
            # append integral error states to state vector
            x0 = np.zeros(num_vars)
            x0[:initial_state.shape[0]] = initial_state
        else:
            x0 = initial_state

        assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"
        self.x0 = x0
        self.ap = ap

        self.times = None
        self.states = None
        self.modes = None

        self.extended_states = extended_states

        if self.extended_states:
            self.xd_list = None
            self.u_list = None
            self.Nz_list = None
            self.ps_list = None
            self.Ny_r_list = None

        self.cur_sim_time = 0
        self.total_sim_time = 0

        self.der_func = make_der_func(ap, model_str, v2_integrators)

        if integrator_str == 'rk45':
            integrator_class = RK45
            self.integrator_kwargs = {}
        else:
            assert integrator_str == 'euler'
            integrator_class = Euler
            self.integrator_kwargs = {'step': step}

        self.integrator_class = integrator_class
        self.integrator = None

        self.freeze_attrs()

    def init_simulation(self):
        'initial simulation (upon first call to simulate_to)'

        assert self.integrator is None

        self.times = [0]
        self.states = [self.x0]

        # mode can change at time 0
        self.ap.advance_discrete_mode(self.times[-1], self.states[-1])

        self.modes = [self.ap.mode]

        if self.extended_states:
            xd, u, Nz, ps, Ny_r = get_extended_states(self.ap, self.times[-1], self.states[-1],
                                                      self.model_str, self.v2_integrators)

            self.xd_list = [xd]
            self.u_list = [u]
            self.Nz_list = [Nz]
            self.ps_list = [ps]
            self.Ny_r_list = [Ny_r]
        
        # note: fixed_step argument is unused by rk45, used with euler
        self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1], np.inf,
                                                **self.integrator_kwargs)

    def simulate_to(self, tmax, tol=1e-7, update_mode_at_start=False):
        '''simulate up to the passed in time

        this adds states to self.times, self.states, self.modes, and the other extended state lists if applicable 
        '''

        # underflow errors were occuring if I don't do this
        oldsettings = np.geterr()
        np.seterr(all='raise', under='ignore')

        start = time.perf_counter()

        ap = self.ap
        step = self.step

        if self.integrator is None:
            self.init_simulation()
        elif update_mode_at_start:
            mode_changed = ap.advance_discrete_mode(self.times[-1], self.states[-1])
            self.modes[-1] = ap.mode # overwrite last mode

            if mode_changed:
                # re-initialize the integration class on discrete mode switches
                self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1], np.inf, \
                                                        **self.integrator_kwargs)

        assert tmax >= self.cur_sim_time
        self.cur_sim_time = tmax

        assert self.integrator.status == 'running', \
            f"integrator status was {self.integrator.status} in call to simulate_to()"

        assert len(self.modes) == len(self.times), f"modes len was {len(self.modes)}, times len was {len(self.times)}"
        assert len(self.states) == len(self.times)

        while True:
            if not self.keep_intermediate_states and len(self.times) > 1:
                # drop all except last state
                self.times = [self.times[-1]]
                self.states = [self.states[-1]]
                self.modes = [self.modes[-1]]

                if self.extended_states:
                    self.xd_list = [self.xd_list[-1]]
                    self.u_list = [self.u_list[-1]]
                    self.Nz_list = [self.Nz_list[-1]]
                    self.ps_list = [self.ps_list[-1]]
                    self.Ny_r_list = [self.Ny_r_list[-1]]

            next_step_time = self.times[-1] + step

            if abs(self.times[-1] - tmax) > tol and next_step_time > tmax:
                # use a small last step
                next_step_time = tmax

            if next_step_time >= tmax + tol:
                # don't do any more steps
                break

            # goal for rest of the loop: do one more step

            while next_step_time >= self.integrator.t + tol:
                # keep advancing integrator until it goes past the next step time
                assert self.integrator.status == 'running'
                
                self.integrator.step()

                if self.integrator.status != 'running':
                    break

            if self.integrator.status != 'running':
                break

            # get the state at next_step_time
            self.times.append(next_step_time)

            if abs(self.integrator.t - next_step_time) < tol:
                self.states.append(self.integrator.x)
            else:
                dense_output = self.integrator.dense_output()
                self.states.append(dense_output(next_step_time))

            # re-run dynamics function at current state to get non-state variables
            if self.extended_states:
                xd, u, Nz, ps, Ny_r = get_extended_states(ap, self.times[-1], self.states[-1],
                                                          self.model_str, self.v2_integrators)

                self.xd_list.append(xd)
                self.u_list.append(u)

                self.Nz_list.append(Nz)
                self.ps_list.append(ps)
                self.Ny_r_list.append(Ny_r)

            mode_changed = ap.advance_discrete_mode(self.times[-1], self.states[-1])
            self.modes.append(ap.mode)

            stop_func = self.custom_stop_func if self.custom_stop_func is not None else ap.is_finished

            if stop_func(self.times[-1], self.states[-1]):
                # this both causes the outer loop to exit and sets res['status'] appropriately
                self.integrator.status = 'autopilot finished'
                break

            if mode_changed:
                # re-initialize the integration class on discrete mode switches
                self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1], np.inf,
                                                        **self.integrator_kwargs)

        if self.integrator.status == 'failed' and self.print_errors:
            print(f'Warning: integrator status was "{self.integrator.status}"')
        elif self.integrator.status != 'autopilot finished':
            assert abs(self.times[-1] - tmax) < tol, f"tmax was {tmax}, self.times[-1] was {self.times[-1]}"

        self.total_sim_time += time.perf_counter() - start
        np.seterr(**oldsettings)

def run_f16_sim(initial_state, tmax, ap, step=1/30, extended_states=False,
                integrator_str='rk45', v2_integrators=False, print_errors=True,
                custom_stop_func=None):
    '''Simulates and analyzes autonomous F-16 maneuvers

    if multiple aircraft are to be simulated at the same time,
    initial_state should be the concatenated full (including integrators) initial state.

    returns a dict with the following keys:

    'status': integration status, should be 'finished' if no errors, or 'autopilot finished'
    'times': time history
    'states': state history at each time step
    'modes': mode history at each time step

    if extended_states was True, result also includes:
    'xd_list' - derivative at each time step
    'ps_list' - ps at each time step
    'Nz_list' - Nz at each time step
    'Ny_r_list' - Ny_r at each time step
    'u_list' - input at each time step, input is 7-tuple: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    These are tuples if multiple aircraft are used
    '''

    fss = F16SimState(initial_state, ap, step, extended_states,
                integrator_str, v2_integrators, print_errors, custom_stop_func=custom_stop_func)

    fss.simulate_to(tmax)

    # extract states

    res = {}
    res['status'] = fss.integrator.status
    res['times'] = fss.times
    res['states'] = np.array(fss.states, dtype=float)
    res['modes'] = fss.modes

    if extended_states:
        res['xd_list'] = fss.xd_list
        res['ps_list'] = fss.ps_list
        res['Nz_list'] = fss.Nz_list
        res['Ny_r_list'] = fss.Ny_r_list
        res['u_list'] = fss.u_list

    res['runtime'] = fss.total_sim_time

    return res

class SimModelError(RuntimeError):
    'simulation state went outside of what the model is capable of simulating'

def make_der_func(ap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]

            #print(f".called der_func(aircraft={i}, t={t}, state={full_state}")

            alpha = state[StateIndex.ALPHA]
            if not -2 < alpha < 2:
                raise SimModelError(f"alpha ({alpha}) out of bounds")

            vel = state[StateIndex.VEL]
            # even going lower than 300 is probably not a good idea
            if not 200 <= vel <= 3000:
                raise SimModelError(f"velocity ({vel}) out of bounds")

            alt = state[StateIndex.ALT]
            if not -10000 < alt < 100000:
                raise SimModelError(f"altitude ({alt}) out of bounds")

            u_ref = u_refs[4*i:4*(i+1)]

            xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv

    return der_func

def get_extended_states(ap, t, full_state, model_str, v2_integrators):
    '''get xd, u, Nz, ps, Ny_r at the current time / state

    returns tuples if more than one aircraft
    '''

    llc = ap.llc
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = full_state.size // num_vars

    xd_tup = []
    u_tup = []
    Nz_tup = []
    ps_tup = []
    Ny_r_tup = []

    u_refs = ap.get_checked_u_ref(t, full_state)

    for i in range(num_aircraft):
        state = full_state[num_vars*i:num_vars*(i+1)]
        u_ref = u_refs[4*i:4*(i+1)]

        xd, u, Nz, ps, Ny_r = controlled_f16(t, state, u_ref, llc, model_str, v2_integrators)

        xd_tup.append(xd)
        u_tup.append(u)
        Nz_tup.append(Nz)
        ps_tup.append(ps)
        Ny_r_tup.append(Ny_r)

    if num_aircraft == 1:
        rv_xd = xd_tup[0]
        rv_u = u_tup[0]
        rv_Nz = Nz_tup[0]
        rv_ps = ps_tup[0]
        rv_Ny_r = Ny_r_tup[0]
    else:
        rv_xd = tuple(xd_tup)
        rv_u = tuple(u_tup)
        rv_Nz = tuple(Nz_tup)
        rv_ps = tuple(ps_tup)
        rv_Ny_r = tuple(Ny_r_tup)

    return rv_xd, rv_u, rv_Nz, rv_ps, rv_Ny_r

```

### code/aerobench/util.py

```
'''
Utilities for F-16 GCAS
'''

import os
from math import floor, ceil

import numpy as np

class StateIndex:
    'list of static state indices'

    VT = 0
    VEL = 0 # alias
    
    ALPHA = 1
    BETA = 2
    PHI = 3 # roll angle
    THETA = 4 # pitch angle
    PSI = 5 # yaw angle
    
    P = 6
    Q = 7
    R = 8
    
    POSN = 9
    POS_N = 9
    
    POSE = 10
    POS_E = 10
    
    ALT = 11
    H = 11
    
    POW = 12

def print_state(state):
    'print the state to stdout'

    for l, s in zip(get_state_names(), state):
        print(f"{l}: {s}")

class Freezable():
    'a class where you can freeze the fields (prevent new fields from being created)'

    _frozen = False

    def freeze_attrs(self):
        'prevents any new attributes from being created in the object'
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError("{} does not contain attribute '{}' (object was frozen)".format(self, key))

        object.__setattr__(self, key, value)

class Euler(Freezable):
    '''fixed step euler integration

    loosely based on scipy.integrate.RK45
    '''

    def __init__(self, der_func, tstart, ystart, tend, step=0, time_tol=1e-9):
        assert step > 0, "arg step > 0 required in Euler integrator"
        assert tend > tstart

        self.der_func = der_func # signature (t, x)
        self.tstep = step
        self.t = tstart
        self.y = ystart.copy()
        self.yprev = None
        self.tprev = None
        self.tend = tend

        self.status = 'running'

        self.time_tol = time_tol

        self.freeze_attrs()

    def step(self):
        'take one step'

        if self.status == 'running':
            self.yprev = self.y.copy()
            self.tprev = self.t
            yd = self.der_func(self.t, self.y)

            self.t += self.tstep

            if self.t + self.time_tol >= self.tend:
                self.t = self.tend

            dt = self.t - self.tprev
            self.y += dt * yd

            if self.t == self.tend:
                self.status = 'finished'

    def dense_output(self):
        'return a function of time'

        assert self.tprev is not None

        dy = self.y - self.yprev
        dt = self.t - self.tprev

        dydt = dy / dt

        def fun(t):
            'return state at time t (linear interpolation)'

            deltat = t - self.tprev

            return self.yprev + dydt * deltat

        return fun

def get_state_names():
    'returns a list of state variable names'

    return ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'P', 'Q', 'R', 'pos_n', 'pos_e', 'alt', 'pow']

def printmat(mat, main_label, row_label_str, col_label_str):
    'print a matrix'

    if isinstance(row_label_str, list) and len(row_label_str) == 0:
        row_label_str = None

    assert isinstance(main_label, str)
    assert row_label_str is None or isinstance(row_label_str, str)
    assert isinstance(col_label_str, str)

    mat = np.array(mat)
    if len(mat.shape) == 1:
        mat.shape = (1, mat.shape[0]) # one-row matrix

    print("{main_label} =")

    row_labels = None if row_label_str is None else row_label_str.split(' ')
    col_labels = col_label_str.split(' ')

    width = 7

    width = max(width, max([len(l) for l in col_labels]))

    if row_labels is not None:
        width = max(width, max([len(l) for l in row_labels]))

    width += 1

    # add blank space for row labels
    if row_labels is not None:
        print("{: <{}}".format('', width), end='')

    # print col lables
    for col_label in col_labels:
        if len(col_label) > width:
            col_label = col_label[:width]

        print("{: >{}}".format(col_label, width), end='')

    print('')

    if row_labels is not None:
        assert len(row_labels) == mat.shape[0], \
            "row labels (len={}) expected one element for each row of the matrix ({})".format( \
            len(row_labels), mat.shape[0])

    for r in range(mat.shape[0]):
        row = mat[r]

        if row_labels is not None:
            label = row_labels[r]

            if len(label) > width:
                label = label[:width]

            print("{:<{}}".format(label, width), end='')

        for num in row:
            #print("{:#<{}}".format(num, width), end='')
            print("{:{}.{}g}".format(num, width, width-3), end='')

        print('')


def fix(ele):
    'round towards zero'

    assert isinstance(ele, float)

    if ele > 0:
        rv = int(floor(ele))
    else:
        rv = int(ceil(ele))

    return rv

def sign(ele):
    'sign of a number'

    if ele < 0:
        rv = -1
    elif ele == 0:
        rv = 0
    else:
        rv = 1

    return rv

def extract_single_result(res, index, llc):
    'extract a res object for a sinlge aircraft from a multi-aircraft simulation'

    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = res['states'][0].size // num_vars

    if num_aircraft == 1:
        assert index == 0
        rv = res
    else:
        rv = {}
        rv['status'] = res['status']
        rv['times'] = res['times']
        rv['modes'] = res['modes']

        full_states = res['states']
        rv['states'] = full_states[:, num_vars*index:num_vars*(index+1)]

        if 'xd_list' in res:
            # extended states
            key_list = ['xd_list', 'ps_list', 'Nz_list', 'Ny_r_list', 'u_list']

            for key in key_list:
                rv[key] = [tup[index] for tup in res[key]]

    return rv


class SafetyLimits(Freezable):
    'a class for holding a set of safety limits.'

    def __init__(self, **kwargs):
        self.altitude = kwargs['altitude'] if 'altitude' in kwargs and kwargs['altitude'] is not None else None
        self.Nz = kwargs['Nz'] if 'Nz' in kwargs and kwargs['Nz'] is not None else None
        self.v = kwargs['v'] if 'v' in kwargs and kwargs['v'] is not None else None
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs and kwargs['alpha'] is not None else None

        self.psMaxAccelDeg = kwargs['psMaxAccelDeg'] if 'psMaxAccelDeg' in kwargs and kwargs['psMaxAccelDeg'] is not None else None
        self.betaMaxDeg = kwargs['betaMaxDeg'] if 'betaMaxDeg' in kwargs and kwargs['betaMaxDeg'] is not None else None

        self.freeze_attrs()

class SafetyLimitsVerifier(Freezable):
    'given some limits (in a SafetyLimits) and optional low-level controller (in a LowLevelController), verify whether the simulation results are safe.'

    def __init__(self, safety_limits, llc=None):
        self.safety_limits = safety_limits
        self.llc = llc

    def verify(self, results):
        # Determine the number of state variables per tick of the simulation.
        if self.llc is not None:
            num_state_vars = len(get_state_names()) + \
                             self.llc.get_num_integrators()
        else:
            num_state_vars = len(get_state_names())
        # Check whether the results are sane.
        assert (results['states'].size % num_state_vars) == 0, \
            "Wrong number of state variables."

        # Go through each tick of the simulation and determine whether
        # the object(s) was (were) in a safe state.
        for i in range(results['states'].size // num_state_vars):
            _vt, alpha, beta, _phi, \
                _theta, _psi, _p, _q, \
                _r, _pos_n, _pos_e, alt, \
                _, _, _, _ = results['states'][i]
            nz = results['Nz_list'][i]
            ps = results['ps_list'][i]

            if self.safety_limits.altitude is not None:
                assert self.safety_limits.altitude[0] <= alt <= self.safety_limits.altitude[1], "Altitude ({}) is not within the specified limits ({}, {}).".format(alt, self.safety_limits.altitude[0], self.safety_limits.altitude[1])

            if self.safety_limits.Nz is not None:
                assert self.safety_limits.Nz[0] <= nz <= self.safety_limits.Nz[1], "Nz ({}) is not within the specified limits ({}, {}).".format(nz, self.safety_limits.Nz[0], self.safety_limits.Nz[1])

            if self.safety_limits.alpha is not None:
                assert self.safety_limits.alpha[0] <= alpha <= self.safety_limits.alpha[1], "alpha ({}) is not within the specified limits ({}, {}).".format(nz, self.safety_limits.alpha[0], self.safety_limits.alpha[1])

            if self.safety_limits.psMaxAccelDeg is not None:
                assert ps <= self.safety_limits.psMaxAccelDeg, "Ps is not less than the specified max."

            if self.safety_limits.betaMaxDeg is not None:
                assert beta <= self.safety_limits.betaMaxDeg, "Beta is not less than the specified max."

def get_script_path(script_filename):
    '''get the path this script

    Pass __file__ as the argument
    '''
    
    return os.path.dirname(os.path.realpath(script_filename))


```

### code/aerobench/visualize/anim.py

```
'''
2d animation utilities for aerobench
'''

import math
import time
import os
import traceback

from scipy import ndimage

from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from aerobench.visualize import plot
from aerobench.util import StateIndex, get_script_path, get_state_names


def make_anim(res, llc, filename, skip_frames=None, figsize=(7, 5), show_closest=True, print_frame=True,
              init_extra=None, update_extra=None):
    '''
    make a 2d animiation

    params can be a list for back-to-back animations
    '''

    plot.init_plot()
    start = time.time()

    if not isinstance(res, list):
        res = [res]

    if not isinstance(skip_frames, list):
        skip_frames = [skip_frames]

    if not isinstance(init_extra, list):
        init_extra = [init_extra]

    if not isinstance(update_extra, list):
        update_extra = [update_extra]

    #####
    # fill in defaults
    for i, skip in enumerate(skip_frames):
        if skip is not None:
            continue

        if filename == '': # plot to the screen
            skip_frames[i] = 5
        elif filename.endswith('.gif'):
            skip_frames[i] = 2
        else:
            skip_frames[i] = 1 # plot every frame

    if filename == '':
        filename = None

    ##
    all_times = []
    all_states = []
    all_modes = []

    for r, skip in zip(res, skip_frames):
        t = r['times']
        s = r['states']
        m = r['modes']

        t = t[0::skip]
        s = s[0::skip]
        m = m[0::skip]

        all_times.append(t)
        all_states.append(s)
        all_modes.append(m)
            
    ##

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ##

    num_planes_list = []
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    max_num_planes = 0

    for states in all_states:
        s0 = states[0]
        assert(s0.size % num_vars) == 0
        
        num_planes = s0.size // num_vars
        num_planes_list.append(num_planes)
        max_num_planes = max(max_num_planes, num_planes)

    ax.set_xlabel('X [ft]', fontsize=14)
    ax.set_ylabel('Y [ft]', fontsize=14)

    states = all_states[0]
    axis_limits = plot.set_axis_limits(ax, num_vars, states)

    parent = get_script_path(__file__)
    plane_path = os.path.join(parent, 'airplane.png')
    img = plt.imread(plane_path)

    planes = []
    plane_trails = []
    plane_previews = []
    plane_size_factor = 0.05 # percent of axis limits

    for i in range(max_num_planes):
        size = (axis_limits[1] - axis_limits[0]) * plane_size_factor
        box = Bbox.from_bounds(0, 0, size, size)
        tbox = TransformedBbox(box, ax.transData)
        box_image = BboxImage(tbox, zorder=2)

        #theta_deg = (theta - math.pi / 2) / math.pi * 180 # original image is facing up, not right
        theta_deg = 0
        img_rotated = ndimage.rotate(img, theta_deg, order=1)

        box_image.set_data(img_rotated)
        ax.add_artist(box_image)

        planes.append(box_image)

        trail, = ax.plot([], [], lw=1, zorder=2)
        plane_trails.append(trail)

        preview, = ax.plot([], [], ':', color=trail.get_color(), lw=1, zorder=1)
        plane_previews.append(preview)

    # text
    fontsize = 13
    cur_min_text = ax.text(0.47, 0.96, "", transform=ax.transAxes, fontsize=fontsize, color='b',
                           horizontalalignment='center')
    
    acc_min_text = ax.text(0.97, 0.96, "", transform=ax.transAxes, fontsize=fontsize, color='r',
                           horizontalalignment='right')
    
    time_text = ax.text(0.03, 0.96, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='left')

    l1, = ax.plot([], [], 'b:', lw=2, zorder=4)
    l2, = ax.plot([], [], 'r:', lw=2, zorder=5)
    min_lines = [l1, l2]

    if not show_closest:
        l1.set_visible(False)
        l2.set_visible(False)

    extra_lines = []

    for func in init_extra:
        if func is not None:
            extra_lines.append(func(ax))
        else:
            extra_lines.append([])

    first_frames = []
    frames = 0

    for t in all_times:
        first_frames.append(frames)
        frames += len(t)

    dist_data = [0, [[0], [0]]] # cur_global_min, cur_global_min_line (list of xs, ys)

    def anim_func(global_frame):
        'updates for the animation frame'

        index = 0
        first_frame = False

        for i, f in enumerate(first_frames):
            if global_frame >= f:
                index = i

                if global_frame == f:
                    first_frame = True
                    break

        frame = global_frame - first_frames[index]
        states = all_states[index]
        times = all_times[index]
        modes = all_modes[index]

        if print_frame:
            print(f"Frame: {global_frame}/{frames} - Index {index} frame {frame}/{len(times)}")

        time_text.set_text(f"Time: {round(times[frame], 1)} sec")

        mode = modes[frame]

        state = states[frame]
        num_planes = num_planes_list[index]
        
        cur_min = np.inf
        min_line = [[], []] # list of [xs, ys]
        
        for p1 in range(num_planes):
            s1 = state[p1*num_vars:(p1+1)*num_vars]
            
            x1 = s1[StateIndex.POS_E]
            y1 = s1[StateIndex.POS_N]
            
            for p2 in range(p1+1, num_planes):
                s2 = state[p2*num_vars:(p2+1)*num_vars]

                x2 = s2[StateIndex.POS_E]
                y2 = s2[StateIndex.POS_N]

                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)

                if dist < cur_min:
                    cur_min = dist
                    min_line = [[x1, x2], [y1, y2]]

        cur_min_text.set_text(f"Cur Dist: {round(cur_min)} ft")

        if first_frame or cur_min < dist_data[0]:
            dist_data[0] = cur_min
            dist_data[1] = min_line

        acc_min_text.set_text(f"Min Dist: {round(dist_data[0])} ft")

        min_lines[0].set_data(min_line[0], min_line[1])
        min_lines[1].set_data(dist_data[1][0], dist_data[1][1])

        if first_frame:
            axis_limits[:] = plot.set_axis_limits(ax, num_vars, states)

            # initialize dist_stats (index 0: cur_min, index 1: total_min)
            for i, lines in enumerate(extra_lines):
                for line in lines:
                    line.set_visible(i == index)

            for i, (p, t) in enumerate(zip(planes, plane_trails)):
                is_vis = i < num_planes_list[index]

                p.set_visible(is_vis)
                t.set_visible(is_vis)

        # do trail
        for i in range(num_planes):
            # do trails
            offset = i * num_vars
            pos_xs = [pt[offset + StateIndex.POS_E] for pt in states]
            pos_ys = [pt[offset + StateIndex.POS_N] for pt in states]

            plane_trails[i].set_data(pos_xs[:frame], pos_ys[:frame])
            plane_previews[i].set_data(pos_xs, pos_ys)

            # do plane images
            s = state[i*num_vars:(i+1)*num_vars]
            psi = s[StateIndex.PSI]
            x = s[StateIndex.POS_E]
            y = s[StateIndex.POS_N]

            theta_deg = -psi * 180 / math.pi
            original_size = list(img.shape)
            img_rotated = ndimage.rotate(img, theta_deg, order=1)
            rotated_size = list(img_rotated.shape)
            ratios = [r / o for r, o in zip(rotated_size, original_size)]
            planes[i].set_data(img_rotated)

            size = (axis_limits[1] - axis_limits[0]) * plane_size_factor
            width = size * ratios[0]
            height = size * ratios[1]
            box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
            tbox = TransformedBbox(box, ax.transData)
            planes[i].bbox = tbox

        if update_extra[index] is not None:
            update_extra[index](frame, times[frame], state, mode)

    plt.tight_layout()

    interval = 30

    if filename is not None and filename.endswith('.gif'):
        interval = 60

    anim_obj = animation.FuncAnimation(fig, anim_func, frames, interval=interval, \
        blit=False, repeat=True)

    if filename is not None:

        if filename.endswith('.gif'):
            print("\nSaving animation to '{}' using 'imagemagick'...".format(filename))
            anim_obj.save(filename, dpi=60, writer='imagemagick') # dpi was 80
            print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
        else:
            fps = 40
            codec = 'libx264'

            print("\nSaving '{}' at {:.2f} fps using ffmpeg with codec '{}'.".format(filename, fps, codec))

            # if this fails do: 'sudo apt-get install ffmpeg'
            try:
                extra_args = []

                if codec is not None:
                    extra_args += ['-vcodec', str(codec)]

                anim_obj.save(filename, fps=fps, extra_args=extra_args)
                print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
            except AttributeError:
                traceback.print_exc()
                print("\nSaving video file failed! Is ffmpeg installed? Can you run 'ffmpeg' in the terminal?")
    else:
        plt.show()

```

### code/aerobench/visualize/anim3d.py

```
'''
3d plotting utilities for aerobench
'''

import math
import time
import os
import traceback

from scipy.io import loadmat

import numpy as np
from numpy import rad2deg

# these imports are needed for 3d plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from aerobench.visualize import plot
from aerobench.util import StateIndex, get_script_path

def make_anim(res, filename, viewsize=1000, viewsize_z=1000, f16_scale=30, trail_pts=60,
              elev=30, azim=45, skip_frames=None, chase=False, fixed_floor=False,
              init_extra=None, update_extra=None):
    '''
    make a 3d plot of the GCAS maneuver.

    see examples/anim3d folder for examples on usage
    '''

    plot.init_plot()
    start = time.time()

    if not isinstance(res, list):
        res = [res]

    if not isinstance(viewsize, list):
        viewsize = [viewsize]

    if not isinstance(viewsize_z, list):
        viewsize_z = [viewsize_z]

    if not isinstance(f16_scale, list):
        f16_scale = [f16_scale]

    if not isinstance(trail_pts, list):
        trail_pts = [trail_pts]

    if not isinstance(elev, list):
        elev = [elev]

    if not isinstance(azim, list):
        azim = [azim]

    if not isinstance(skip_frames, list):
        skip_frames = [skip_frames]

    if not isinstance(chase, list):
        chase = [chase]

    if not isinstance(fixed_floor, list):
        fixed_floor = [fixed_floor]

    if not isinstance(init_extra, list):
        init_extra = [init_extra]

    if not isinstance(update_extra, list):
        update_extra = [update_extra]

    #####
    # fill in defaults
    if filename == '':
        full_plot = False
    else:
        full_plot = True

    for i, skip in enumerate(skip_frames):
        if skip is not None:
            continue

        if filename == '': # plot to the screen
            skip_frames[i] = 5
        elif filename.endswith('.gif'):
            skip_frames[i] = 2
        else:
            skip_frames[i] = 1 # plot every frame

    if filename == '':
        filename = None

    ##
    all_times = []
    all_states = []
    all_modes = []
    all_ps_list = []
    all_Nz_list = []

    for r, skip in zip(res, skip_frames):
        t = r['times']
        s = r['states']
        m = r['modes']
        ps = r['ps_list']
        Nz = r['Nz_list']

        t = t[0::skip]
        s = s[0::skip]
        m = m[0::skip]
        ps = ps[0::skip]
        Nz = Nz[0::skip]

        all_times.append(t)
        all_states.append(s)
        all_modes.append(m)
        all_ps_list.append(ps)
        all_Nz_list.append(Nz)
            
    ##

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    ##

    parent = get_script_path(__file__)
    plane_point_data = os.path.join(parent, 'f-16.mat')

    data = loadmat(plane_point_data)
    f16_pts = data['V']
    f16_faces = data['F']

    plane_polys = Poly3DCollection([], color=None if full_plot else 'k')
    ax.add_collection3d(plane_polys)

    ax.set_xlabel('X [ft]', fontsize=14)
    ax.set_ylabel('Y [ft]', fontsize=14)
    ax.set_zlabel('Altitude [ft]', fontsize=14)

    # text
    fontsize = 14
    time_text = ax.text2D(0.05, 0.97, "", transform=ax.transAxes, fontsize=fontsize)
    mode_text = ax.text2D(0.95, 0.97, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    alt_text = ax.text2D(0.05, 0.93, "", transform=ax.transAxes, fontsize=fontsize)
    v_text = ax.text2D(0.95, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    alpha_text = ax.text2D(0.05, 0.89, "", transform=ax.transAxes, fontsize=fontsize)
    beta_text = ax.text2D(0.95, 0.89, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    nz_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, fontsize=fontsize)
    ps_text = ax.text2D(0.95, 0.85, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    ang_text = ax.text2D(0.5, 0.81, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')

    trail_line, = ax.plot([], [], [], color='r', lw=2, zorder=50)

    extra_lines = []

    for func in init_extra:
        if func is not None:
            extra_lines.append(func(ax))
        else:
            extra_lines.append([])

    first_frames = []
    frames = 0

    for t in all_times:
        first_frames.append(frames)
        frames += len(t)

    def anim_func(global_frame):
        'updates for the animation frame'

        index = 0
        first_frame = False

        for i, f in enumerate(first_frames):
            if global_frame >= f:
                index = i

                if global_frame == f:
                    first_frame = True
                    break

        frame = global_frame - first_frames[index]
        states = all_states[index]
        times = all_times[index]
        modes = all_modes[index]
        Nz_list = all_Nz_list[index]
        ps_list = all_ps_list[index]

        print(f"Frame: {global_frame}/{frames} - Index {index} frame {frame}/{len(times)}")

        speed = states[frame][0]
        alpha = states[frame][1]
        beta = states[frame][2]
        alt = states[frame][11]

        phi = states[frame][StateIndex.PHI]
        theta = states[frame][StateIndex.THETA]
        psi = states[frame][StateIndex.PSI]

        dx = states[frame][StateIndex.POS_E]
        dy = states[frame][StateIndex.POS_N]
        dz = states[frame][StateIndex.ALT]

        if first_frame:
            ax.view_init(elev[index], azim[index])

            for i, lines in enumerate(extra_lines):
                for line in lines:
                    line.set_visible(i == index)

        time_text.set_text('t = {:.2f} sec'.format(times[frame]))

        if chase[index]:
            ax.view_init(elev[index], rad2deg(-psi) - 90.0)

        colors = ['red', 'blue', 'green', 'magenta']

        mode_names = []

        for mode in modes:
            if not mode in mode_names:
                mode_names.append(mode)

        mode = modes[frame]
        mode_index = modes.index(mode)
        col = colors[mode_index % len(colors)]
        mode_text.set_color(col)
        mode_text.set_text('Mode: {}'.format(mode.capitalize()))

        alt_text.set_text('h = {:.2f} ft'.format(alt))
        v_text.set_text('V = {:.2f} ft/sec'.format(speed))

        alpha_text.set_text('$\\alpha$ = {:.2f} deg'.format(rad2deg(alpha)))
        beta_text.set_text('$\\beta$ = {:.2f} deg'.format(rad2deg(beta)))

        nz_text.set_text('$N_z$ = {:.2f} g'.format(Nz_list[frame]))
        ps_text.set_text('$p_s$ = {:.2f} deg/sec'.format(rad2deg(ps_list[frame])))

        ang_text.set_text('[$\\phi$, $\\theta$, $\\psi$] = [{:.2f}, {:.2f}, {:.2f}] deg'.format(\
            rad2deg(phi), rad2deg(theta), rad2deg(psi)))

        s = f16_scale[index]
        s = 25 if s is None else s
        pts = scale3d(f16_pts, [-s, s, s])

        pts = rotate3d(pts, theta, psi - math.pi/2, -phi)

        size = viewsize[index]
        size = 1000 if size is None else size
        minx = dx - size
        maxx = dx + size
        miny = dy - size
        maxy = dy + size

        vz = viewsize_z[index]
        vz = 1000 if vz is None else vz

        if fixed_floor[index]:
            minz = 0
            maxz = vz
        else:
            minz = dz - vz
            maxz = dz + vz

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])
        ax.set_zlim([minz, maxz])

        verts = []
        fc = []
        ec = []
        count = 0

        # draw ground
        if minz <= 0 <= maxz:
            z = 0
            verts.append([(minx, miny, z), (maxx, miny, z), (maxx, maxy, z), (minx, maxy, z)])
            fc.append('0.8')
            ec.append('0.8')

        # draw f16
        for face in f16_faces:
            face_pts = []

            count = count + 1

            if not full_plot and count % 10 != 0:
                continue

            for findex in face:
                face_pts.append((pts[findex-1][0] + dx, \
                    pts[findex-1][1] + dy, \
                    pts[findex-1][2] + dz))

            verts.append(face_pts)
            fc.append('0.2')
            ec.append('0.2')

        plane_polys.set_verts(verts)
        plane_polys.set_facecolor(fc)
        plane_polys.set_edgecolor(ec)

        # do trail
        t = trail_pts[index]
        t = 200 if t is None else t
        trail_len = t // skip_frames[index]
        start_index = max(0, frame-trail_len)

        pos_xs = [pt[StateIndex.POS_E] for pt in states]
        pos_ys = [pt[StateIndex.POS_N] for pt in states]
        pos_zs = [pt[StateIndex.ALT] for pt in states]

        trail_line.set_data(np.asarray(pos_xs[start_index:frame]), np.asarray(pos_ys[start_index:frame]))
        trail_line.set_3d_properties(np.asarray(pos_zs[start_index:frame]))

        if update_extra[index] is not None:
            update_extra[index](frame)

    plt.tight_layout()

    interval = 30

    if filename is not None and filename.endswith('.gif'):
        interval = 60

    anim_obj = animation.FuncAnimation(fig, anim_func, frames, interval=interval, \
        blit=False, repeat=True)

    if filename is not None:

        if filename.endswith('.gif'):
            print("\nSaving animation to '{}' using 'imagemagick'...".format(filename))
            anim_obj.save(filename, dpi=60, writer='imagemagick') # dpi was 80
            print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
        else:
            fps = 40
            codec = 'libx264'

            print("\nSaving '{}' at {:.2f} fps using ffmpeg with codec '{}'.".format(filename, fps, codec))

            # if this fails do: 'sudo apt-get install ffmpeg'
            try:
                extra_args = []

                if codec is not None:
                    extra_args += ['-vcodec', str(codec)]

                anim_obj.save(filename, fps=fps, extra_args=extra_args)
                print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
            except AttributeError:
                traceback.print_exc()
                print("\nSaving video file failed! Is ffmpeg installed? Can you run 'ffmpeg' in the terminal?")
    else:
        plt.show()

def scale3d(pts, scale_list):
    'scale a 3d ndarray of points, and return the new ndarray'

    assert len(scale_list) == 3

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        for d in range(3):
            rv[i, d] = scale_list[d] * pts[i, d]

    return rv

def rotate3d(pts, theta, psi, phi):
    'rotates an ndarray of 3d points, returns new list'

    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    sinPsi = math.sin(psi)
    cosPsi = math.cos(psi)
    sinPhi = math.sin(phi)
    cosPhi = math.cos(phi)

    transform_matrix = np.array([ \
        [cosPsi * cosTheta, -sinPsi * cosTheta, sinTheta], \
        [cosPsi * sinTheta * sinPhi + sinPsi * cosPhi, \
        -sinPsi * sinTheta * sinPhi + cosPsi * cosPhi, \
        -cosTheta * sinPhi], \
        [-cosPsi * sinTheta * cosPhi + sinPsi * sinPhi, \
        sinPsi * sinTheta * cosPhi + cosPsi * sinPhi, \
        cosTheta * cosPhi]], dtype=float)

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        rv[i] = np.dot(pts[i], transform_matrix)

    return rv

```

### code/aerobench/visualize/bak_matplotlib.mlpstyle

```
font.family : serif
xtick.labelsize : 14
ytick.labelsize : 14

axes.labelsize : 20
axes.titlesize : 28
    
path.simplify : False 

```

### code/aerobench/visualize/plot.py

```
'''
Stanley Bak
Python code for F-16 animation video output
'''

import math
import os

from scipy import ndimage

import numpy as np

from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

import matplotlib
import matplotlib.pyplot as plt

from aerobench.util import get_state_names, StateIndex, get_script_path

def init_plot():
    'initialize plotting style'

    matplotlib.use('TkAgg') # set backend

    parent = get_script_path(__file__)
    p = os.path.join(parent, 'bak_matplotlib.mlpstyle')

    plt.style.use(['bmh', p])

def set_axis_limits(ax, num_vars, states, zoom_factor=1.2):
    '''set axis limits

    returns [xmin, xmax, ymin, ymax]
    '''

    assert len(states) > 0

    minx, miny = np.inf, np.inf
    maxx, maxy = -np.inf, -np.inf

    for state in states:
        start = 0
        end = num_vars

        while start < state.size:
            s = state[start:end]
            start = end
            end += num_vars

            x = s[StateIndex.POS_E]
            y = s[StateIndex.POS_N]

            minx = min(minx, x)
            maxx = max(maxx, x)
            miny = min(miny, y)
            maxy = max(maxy, y)

    dx = maxx - minx
    dy = maxy - miny

    if dx < dy:
        midx = (maxx + minx) / 2

        minx = midx - dy/2
        maxx = midx + dy/2
        dx = dy
    elif dy < dx:
        midy = (maxy + miny) / 2

        miny = midy - dx/2
        maxy = midy + dx/2
        dy = dx

    print(f"zoom_factor: {zoom_factor}, x range before: {minx, maxx}")
    # adjust zoom
    midx = (maxx + minx) / 2
    midy = (maxy + miny) / 2

    minx = midx - zoom_factor * dx/2
    maxx = midx + zoom_factor * dx/2

    miny = midy - zoom_factor * dy/2
    maxy = midy + zoom_factor * dy/2

    print(f"x range after: {minx, maxx}")

    # add buffer
    xs = [minx, maxx]
    ys = [miny, maxy]

    ax.set_xlim(xs)
    ax.set_ylim(ys)

    return xs + ys

def plot_overhead(run_sim_result, waypoints=None, llc=None, figsize=(7, 5), plot_frame=0, plane_size_factor=0.05,
                  zoom_factor=1.2, axis_limits=None, aircraft_red_mask=None):
    '''altitude over time plot from run_f16_sum result object

    note: call plt.show() afterwards to have plot show up

    plane_size_factor is percent of axis limits

    returns axis object
    '''

    init_plot()

    res = run_sim_result
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 1, 1)

    full_states = res['states']

    if llc is not None:
        num_vars = len(get_state_names()) + llc.get_num_integrators()
        num_aircraft = full_states[0, :].size // num_vars
    else:
        num_vars = full_states[0, :].size
        num_aircraft = 1

    if axis_limits is not None:
        ax.set_xlim(axis_limits[:2])
        ax.set_ylim(axis_limits[2:])
    else:
        axis_limits = set_axis_limits(ax, num_vars, res['states'], zoom_factor)

    parent = get_script_path(__file__)
    plane_path = os.path.join(parent, 'airplane.png')
    black_img = plt.imread(plane_path)

    red_img = black_img.copy()
    red_img[:, :, 0] = 1
    red_img[:, :, 1:3] = 0

    planes = []

    # init planes
    for i in range(num_aircraft):
        size = (axis_limits[1] - axis_limits[0]) * plane_size_factor
        box = Bbox.from_bounds(0, 0, size, size)
        tbox = TransformedBbox(box, ax.transData)
        box_image = BboxImage(tbox, zorder=2)

        #theta_deg = (theta - math.pi / 2) / math.pi * 180 # original image is facing up, not right
        theta_deg = 0
        i = black_img if aircraft_red_mask is None or not aircraft_red_mask[i] else red_img
        img_rotated = ndimage.rotate(i, theta_deg, order=1)

        box_image.set_data(img_rotated)
        ax.add_artist(box_image)

        planes.append(box_image)

    for i in range(num_aircraft):
        states = full_states[:, i*num_vars:(i+1)*num_vars]

        ys = states[:, StateIndex.POSN] # 9: n/s position (ft)
        xs = states[:, StateIndex.POSE] # 10: e/w position (ft)

        if plot_frame == 0:
            ax.plot(xs, ys, '-')
        else:
            line = ax.plot(xs, ys, ':', zorder=1)[0]
            ax.plot(xs[:plot_frame+1], ys[:plot_frame+1], '-', color=line.get_color(), zorder=2)

        #label = 'Start' if i == 0 else None
        #ax.plot([xs[0]], [ys[1]], 'k*', ms=8, label=label)
        # plot aircraft image
        psi = states[plot_frame, StateIndex.PSI]
        x = states[plot_frame, StateIndex.POS_E]
        y = states[plot_frame, StateIndex.POS_N]
        theta_deg = -psi * 180 / math.pi

        img = black_img if aircraft_red_mask is None or not aircraft_red_mask[i] else red_img
        
        original_size = list(img.shape)
        img_rotated = ndimage.rotate(img, theta_deg, order=1)
        rotated_size = list(img_rotated.shape)
        ratios = [r / o for r, o in zip(rotated_size, original_size)]
        planes[i].set_data(img_rotated)

        size = (axis_limits[1] - axis_limits[0]) * plane_size_factor
        width = size * ratios[0]
        height = size * ratios[1]
        box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
        tbox = TransformedBbox(box, ax.transData)
        planes[i].bbox = tbox

    if waypoints is not None:
        xs = [wp[0] for wp in waypoints]
        ys = [wp[1] for wp in waypoints]

        ax.plot(xs, ys, 'ro', label='Waypoints')

    ax.set_ylabel('North / South Position (ft)')
    ax.set_xlabel('East / West Position (ft)')

    ax.set_title('Overhead Plot')

    return ax

def plot_attitude(run_sim_result, title='Attitude History', skip_yaw=True, figsize=(7, 5), ax=None):
    'plot a single variable over time'

    make_ax = ax is None
    res = run_sim_result

    if make_ax:
        init_plot()

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 1, 1)
        ax.ticklabel_format(useOffset=False)

    times = res['times']
    states = res['states']

    indices = [StateIndex.PHI, StateIndex.THETA, StateIndex.PSI, StateIndex.P, StateIndex.Q, StateIndex.R]
    labels = ['Roll (Phi)', 'Pitch (Theta)', 'Yaw (Psi)', 'Roll Rate (P)', 'Pitch Rate (Q)', 'Yaw Rate (R)']
    colors = ['r-', 'g-', 'b-', 'r--', 'g--', 'b--']

    rad_to_deg_factor = 180 / math.pi

    for index, label, color in zip(indices, labels, colors):
        if skip_yaw and index == StateIndex.PSI:
            continue

        ys = states[:, index] # 11: altitude (ft)

        ax.plot(times, ys * rad_to_deg_factor, color, label=label)

    ax.set_ylabel('Attitudes & Rates (deg, deg/s)')
    ax.set_xlabel('Time (sec)')

    if title is not None:
        ax.set_title(title)

    ax.legend()

    if make_ax:
        plt.tight_layout()

def plot_outer_loop(run_sim_result, title='Outer Loop Controls', ax=None):
    'plot a single variable over time'

    res = run_sim_result
    assert 'u_list' in res, "Simulation must be run with extended_states=True"
    make_ax = ax is None

    if make_ax:
        init_plot()

        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_subplot(1, 1, 1)
        ax.ticklabel_format(useOffset=False)

    times = res['times']
    u_list = res['u_list']
    ps_list = res['ps_list']
    nz_list = res['Nz_list']
    ny_r_list = res['Ny_r_list']

    # u is: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    # u_ref is: Nz, ps, Ny + r, throttle
    ys_list = []

    ys_list.append(nz_list)
    ys_list.append([u[4] for u in u_list])

    ys_list.append(ps_list)
    ys_list.append([u[5] for u in u_list])

    ys_list.append(ny_r_list)
    ys_list.append([u[6] for u in u_list])

    # throttle reference is not included... although it's just a small offset so probably less important
    ys_list.append([u[0] for u in u_list])

    labels = ['N_z', 'N_z,ref', 'P_s', 'P_s,ref', 'N_yr', 'N_yr,ref', 'Throttle']
    colors = ['r', 'r', 'lime', 'lime', 'b', 'b', 'c']

    for i, (ys, label, color) in enumerate(zip(ys_list, labels, colors)):
        lt = '-' if i % 2 == 0 else ':'
        lw = 1 if i % 2 == 0 else 3

        ax.plot(times, ys, lt, lw=lw, color=color, label=label)

    ax.set_ylabel('Autopilot (deg & percent)')
    ax.set_xlabel('Time (sec)')

    if title is not None:
        ax.set_title(title)

    ax.legend()

    if make_ax:
        plt.tight_layout()

def plot_inner_loop(run_sim_result, title='Inner Loop Controls', ax=None):
    'plot inner loop controls over time'

    res = run_sim_result
    assert 'u_list' in res, "Simulation must be run with extended_states=True"

    make_ax = ax is None

    if make_ax:
        init_plot()

        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_subplot(1, 1, 1)
        ax.ticklabel_format(useOffset=False)

    times = res['times']
    u_list = res['u_list']

    # u is throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    ys_list = []

    rad_to_deg_factor = 180 / math.pi

    for i in range(4):
        factor = 1.0 if i == 0 else rad_to_deg_factor
        ys_list.append([u[i] * factor for u in u_list])

    labels = ['Throttle', 'Elevator', 'Aileron', 'Rudder']
    colors = ['b-', 'r-', '#FFA500', 'm-']

    for ys, label, color in zip(ys_list, labels, colors):
        ax.plot(times, ys, color, label=label)

    ax.set_ylabel('Controls (deg & percent)')
    ax.set_xlabel('Time (sec)')

    if title is not None:
        ax.set_title(title)

    ax.legend()

    if make_ax:
        plt.tight_layout()

def plot_single(run_sim_result, state_name, title=None, ax=None):
    'plot a single variable over time'

    make_ax = ax is None
    res = run_sim_result

    if ax is None:
        init_plot()

        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_subplot(1, 1, 1)
        ax.ticklabel_format(useOffset=False)

    times = res['times']
    states = res['states']

    index = get_state_names().index(state_name)
    ys = states[:, index] # 11: altitude (ft)

    ax.plot(times, ys, '-')

    ax.set_ylabel(state_name)
    ax.set_xlabel('Time')

    if title is not None:
        ax.set_title(title)

    if make_ax:
        plt.tight_layout()

def plot2d(filename, times, plot_data_list):
    '''plot state variables in 2d

    plot data list of is a list of (values_list, var_data),
    where values_list is an 2-d array, the first is time step, the second is a state vector
    and each var_data is a list of tuples: (state_index, label)
    '''

    num_plots = sum([len(var_data) for _, var_data in plot_data_list])

    fig = plt.figure(figsize=(7, 5))

    for plot_index in range(num_plots):
        ax = fig.add_subplot(num_plots, 1, plot_index + 1)
        ax.tick_params(axis='both', which='major', labelsize=16)

        sum_plots = 0
        states = None
        state_var_data = None

        for values_list, var_data in plot_data_list:
            if plot_index < sum_plots + len(var_data):
                states = values_list
                state_var_data = var_data
                break

            sum_plots += len(var_data)

        state_index, label = state_var_data[plot_index - sum_plots]

        if state_index == 0 and isinstance(states[0], float): # state is just a single number
            ys = states
        else:
            ys = [state[state_index] for state in states]

        ax.plot(times, ys, '-')

        ax.set_ylabel(label, fontsize=16)

        # last one gets an x axis label
        if plot_index == num_plots - 1:
            ax.set_xlabel('Time', fontsize=16)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

```

### requirements.txt

```
numpy
scipy
matplotlib
slycot
control
```


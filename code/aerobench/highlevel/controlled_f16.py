'''
Stanley Bak
Python Version of F-16 GCAS

Low-level flight controller

and

ODE derivative code (controlled F16)
'''

from math import sin, cos

import numpy as np
from numpy import deg2rad
from enum import Enum

from aerobench.lowlevel.subf16_model import subf16_model

class CtrlLimits(Enum):
    'Control Limits'

    ThrottleMax = 1 # Afterburner on for throttle > 0.7
    ThrottleMin = 0
    ElevatorMaxDeg = 25
    ElevatorMinDeg = -25
    AileronMaxDeg = 21.5
    AileronMinDeg = -21.5
    RudderMaxDeg = 30
    RudderMinDeg = -30
    
    NzMax = 6
    NzMin = -1

# Longitudinal Gains
K_long = np.array([[-156.8801506723475, -31.037008068526642, -38.72983346216317]], dtype=float)
K_lat = np.array([[37.84483, -25.40956, -6.82876, -332.88343, -17.15997],
                        [-23.91233, 5.69968, -21.63431, 64.49490, -88.36203]], dtype=float)

xequil = np.array([502.0, 0.0389, 0.0, 0.0, 0.0389, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 1000.0, 9.0567], dtype=float).transpose()
uequil = np.array([0.1395, -0.7496, 0.0, 0.0], dtype=float).transpose()

K_lqr = np.zeros((3, 8))
K_lqr[:1, :3] = K_long
K_lqr[1:, 3:] = K_lat

def get_u_deg(u_ref, f16_state):
    'get the reference commands for the control surfaces'

    # Calculate perturbation from trim state
    x_delta = f16_state.copy()
    x_delta[:len(xequil)] -= xequil

    ## Implement LQR Feedback Control
    # Reorder states to match controller:
    # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
    x_ctrl = np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=float)

    # Initialize control vectors
    u_deg = np.zeros((4,)) # throt, ele, ail, rud

    # Calculate control using LQR gains
    u_deg[1:4] = np.dot(-K_lqr, x_ctrl) # Full Control

    # Set throttle as directed from output of getOuterLoopCtrl(...)
    u_deg[0] = u_ref[3]

    # Add in equilibrium control
    u_deg[0:4] += uequil

    ## Limit controls to saturation limits

    # Limit throttle from 0 to 1
    u_deg[0] = max(min(u_deg[0], CtrlLimits.ThrottleMax.value), CtrlLimits.ThrottleMin.value)

    # Limit elevator from -25 to 25 deg
    u_deg[1] = max(min(u_deg[1], CtrlLimits.ElevatorMaxDeg.value), CtrlLimits.ElevatorMinDeg.value)

    # Limit aileron from -21.5 to 21.5 deg
    u_deg[2] = max(min(u_deg[2], CtrlLimits.AileronMaxDeg.value), CtrlLimits.AileronMinDeg.value)

    # Limit rudder from -30 to 30 deg
    u_deg[3] = max(min(u_deg[3], CtrlLimits.RudderMaxDeg.value), CtrlLimits.RudderMinDeg.value)

    return x_ctrl, u_deg

def get_integrator_derivatives(u_ref, Nz, ps, Ny_r):
    'get the derivatives of the integrators in the low-level controller'

    return [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]


def controlled_f16(x_f16, u_ref):
    'returns the LQR-controlled F-16 state derivatives and more'

    assert isinstance(x_f16, np.ndarray)
    assert u_ref.size == 4

    x_ctrl, u_deg = get_u_deg(u_ref, x_f16)

    # Note: Control vector (u) for subF16 is in units of degrees
    xd_model, Nz, Ny, _, _ = subf16_model(x_f16[0:13], u_deg)

    # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
    ps = x_ctrl[4] * cos(x_ctrl[0]) + x_ctrl[5] * sin(x_ctrl[0])

    # Calculate (side force + yaw rate) term
    Ny_r = Ny + x_ctrl[5]

    xd = np.zeros((x_f16.shape[0],))
    xd[:len(xd_model)] = xd_model

    # integrators from low-level controller
    start = len(xd_model)
    end = start + 3 # three integrators: Nz, ps, Ny_r
    int_der = get_integrator_derivatives(u_ref, Nz, ps, Ny_r)
    xd[start:end] = int_der

    # Convert all degree values to radians for output
    u_rad = np.zeros((7,)) # throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref

    u_rad[0] = u_deg[0] # throttle

    for i in range(1, 4):
        u_rad[i] = deg2rad(u_deg[i])

    u_rad[4:7] = u_ref[0:3] # inner-loop commands are 4-7

    return xd, u_rad, Nz, ps, Ny_r

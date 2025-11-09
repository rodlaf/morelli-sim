'''
Utilities for F-16 simulation
'''

from math import floor, ceil
import numpy as np


class StateIndex:
    'State variable indices'

    VT = 0
    VEL = 0  # alias
    
    ALPHA = 1
    BETA = 2
    PHI = 3  # roll angle
    THETA = 4  # pitch angle
    PSI = 5  # yaw angle
    
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


class Freezable:
    'Prevents new attributes from being created after freeze_attrs() is called'

    _frozen = False

    def freeze_attrs(self):
        'Prevents any new attributes from being created in the object'
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError(f"{self} does not contain attribute '{key}' (object was frozen)")

        object.__setattr__(self, key, value)


class Euler(Freezable):
    '''Fixed step Euler integration (based on scipy.integrate.RK45 interface)'''

    def __init__(self, der_func, tstart, ystart, tend, step=0, time_tol=1e-9):
        assert step > 0, "arg step > 0 required in Euler integrator"
        assert tend > tstart

        self.der_func = der_func  # signature (t, x)
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
        'Take one integration step'

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
        'Return a function for linear interpolation between steps'

        assert self.tprev is not None

        dy = self.y - self.yprev
        dt = self.t - self.tprev
        dydt = dy / dt

        def fun(t):
            'Return state at time t (linear interpolation)'
            deltat = t - self.tprev
            return self.yprev + dydt * deltat

        return fun


def get_state_names():
    'Returns list of state variable names'
    return ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'P', 'Q', 'R', 'pos_n', 'pos_e', 'alt', 'pow']



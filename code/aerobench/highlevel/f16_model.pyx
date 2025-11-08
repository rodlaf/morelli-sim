# cython: language_level=3
import numpy as np
cimport numpy as np

cdef extern from "f16_model.h":
    void controlled_f16(const double *x_f16, const double *u_ref,
                        double *xd, double *u_rad,
                        double *Nz_out, double *ps_out, double *Ny_r_out)


def controlled_f16_wrapper(np.ndarray[np.float64_t, ndim=1] x_f16,
                            np.ndarray[np.float64_t, ndim=1] u_ref):
    """Call the C controlled_f16 implementation and return numpy outputs."""
    if x_f16.size != 16:
        raise ValueError("x_f16 must have length 16")
    if u_ref.size != 4:
        raise ValueError("u_ref must have length 4")

    cdef np.ndarray[np.float64_t, ndim=1] x_arr = np.ascontiguousarray(x_f16, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] u_arr = np.ascontiguousarray(u_ref, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] xd_arr = np.zeros(16, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] u_rad_arr = np.zeros(7, dtype=np.float64)

    cdef double[::1] x_view = x_arr
    cdef double[::1] u_view = u_arr
    cdef double[::1] xd_view = xd_arr
    cdef double[::1] u_rad_view = u_rad_arr

    cdef double Nz
    cdef double ps
    cdef double Ny_r

    controlled_f16(&x_view[0], &u_view[0], &xd_view[0], &u_rad_view[0], &Nz, &ps, &Ny_r)

    return xd_arr, u_rad_arr, Nz, ps, Ny_r

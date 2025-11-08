cdef extern from "f16_model.h":
    void controlled_f16(const double *x_f16, const double *u_ref,
                        double *xd, double *u_rad,
                        double *Nz_out, double *ps_out, double *Ny_r_out)

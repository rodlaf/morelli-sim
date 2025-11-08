import numpy as np

try:
    from . import f16_model as _f16_model
except ImportError as exc:  # pragma: no cover - build-time error path
    raise ImportError(
        "Cython extension aerobench.highlevel.f16_model not built; "
        "run `python setup.py build_ext --inplace` first."
    ) from exc


def controlled_f16(x_f16: np.ndarray, u_ref: np.ndarray):
    'returns the LQR-controlled F-16 state derivatives and more'

    if not isinstance(x_f16, np.ndarray):
        raise TypeError("x_f16 must be a numpy array")
    if not isinstance(u_ref, np.ndarray):
        raise TypeError("u_ref must be a numpy array")
    if x_f16.size != 16:
        raise ValueError("x_f16 must have length 16")
    if u_ref.size != 4:
        raise ValueError("u_ref must have length 4")

    x_vec = np.ascontiguousarray(x_f16, dtype=np.float64)
    u_vec = np.ascontiguousarray(u_ref, dtype=np.float64)

    xd, u_rad, Nz, ps, Ny_r = _f16_model.controlled_f16_wrapper(x_vec, u_vec)
    return xd, u_rad, Nz, ps, Ny_r

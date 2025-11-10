# AeroBenchVVPython - F-16 Waypoint Environment

Fast C-based F-16 simulation with continuous waypoint reaching task.

## Build

Install dependencies:
```bash
pip install numpy scipy cython
brew install raylib  # macOS
```

Build C extensions:
```bash
python setup.py build_ext --inplace
```

## Run

```bash
python -m aerobench.demo
```

Controls: ESC to exit, window auto-resets on physics violations.

## Citation

"Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers", P. Heidlauf, A. Collins, M. Bolender, S. Bak, ARCH 2018

Distribution A: Approved for Public Release (88ABW-2020-2188)

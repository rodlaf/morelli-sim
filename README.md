# AeroBenchVVPython - F-16 Waypoint Environment

Fast C-based F-16 simulation with continuous waypoint reaching task.

## Build

Install dependencies:
```bash
pip install numpy scipy cython
brew install raylib pkg-config  # macOS
```

Build C extensions:
```bash
python setup.py build_ext --inplace
```

## Run

Python demo (with autopilot):
```bash
python -m aerobench.demo
```

Standalone C demo (no Python required):
```bash
cd aerobench
gcc -o f16_waypoint f16_waypoint.c $(pkg-config --cflags --libs raylib) -lm -O3
./f16_waypoint
```

Speed test:
```bash
python -m aerobench.f16_waypoint
```

Controls: ESC to exit, left-click+drag to rotate camera, scroll to zoom

## Citation

"Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers", P. Heidlauf, A. Collins, M. Bolender, S. Bak, ARCH 2018

Distribution A: Approved for Public Release (88ABW-2020-2188)

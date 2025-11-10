from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
import subprocess
import os

# Get raylib library path
def get_raylib_config():
    """Get raylib include and library paths from pkg-config."""
    try:
        # Try pkg-config first
        cflags = subprocess.check_output(['pkg-config', '--cflags', 'raylib']).decode().strip().split()
        libs = subprocess.check_output(['pkg-config', '--libs', 'raylib']).decode().strip().split()
        
        include_dirs = [flag[2:] for flag in cflags if flag.startswith('-I')]
        library_dirs = [flag[2:] for flag in libs if flag.startswith('-L')]
        libraries = [flag[2:] for flag in libs if flag.startswith('-l')]
        
        return include_dirs, library_dirs, libraries
    except:
        # Fallback to pyray's raylib
        import pyray
        pyray_path = os.path.dirname(pyray.__file__)
        
        # Common locations for raylib when installed with pyray
        possible_includes = [
            os.path.join(pyray_path, 'include'),
            '/usr/local/include',
            '/opt/homebrew/include',  # macOS ARM
            '/usr/include',
        ]
        
        possible_lib_dirs = [
            os.path.join(pyray_path),
            '/usr/local/lib',
            '/opt/homebrew/lib',  # macOS ARM
            '/usr/lib',
        ]
        
        include_dirs = [d for d in possible_includes if os.path.exists(d)]
        library_dirs = [d for d in possible_lib_dirs if os.path.exists(d)]
        
        return include_dirs, library_dirs, ['raylib']

raylib_includes, raylib_lib_dirs, raylib_libs = get_raylib_config()

# Single extension - f16_waypoint_cy which includes everything
# It uses f16_waypoint.h which includes f16_model.h and raylib_renderer.h
extensions = [
    Extension(
        name="aerobench.f16_waypoint_cy",
        sources=["aerobench/f16_waypoint_cy.pyx"],
        include_dirs=["aerobench", np.get_include()] + raylib_includes,
        library_dirs=raylib_lib_dirs,
        libraries=raylib_libs,
        language="c",
        extra_compile_args=["-std=c99"],  # Ensure C99 support
    )
]

setup(
    name="aerobench-f16",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)

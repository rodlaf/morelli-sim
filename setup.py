from setuptools import Extension, setup
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
        # Fallback to common raylib locations
        possible_includes = [
            '/usr/local/include',
            '/opt/homebrew/include',  # macOS ARM
            '/usr/include',
        ]
        
        possible_lib_dirs = [
            '/usr/local/lib',
            '/opt/homebrew/lib',  # macOS ARM
            '/usr/lib',
        ]
        
        include_dirs = [d for d in possible_includes if os.path.exists(d)]
        library_dirs = [d for d in possible_lib_dirs if os.path.exists(d)]
        
        return include_dirs, library_dirs, ['raylib']

raylib_includes, raylib_lib_dirs, raylib_libs = get_raylib_config()

# PufferLib C binding for vectorized training
extensions = [
    Extension(
        name="aerobench.binding",
        sources=["aerobench/binding.c"],
        include_dirs=[
            ".",  # For env_binding.h in project root
            "aerobench",
            np.get_include(),
        ] + raylib_includes,
        library_dirs=raylib_lib_dirs,
        libraries=raylib_libs + ['m'],  # Add math library
        language="c",
        extra_compile_args=["-std=c11", "-O3"],  # C11 for better compatibility
    )
]

setup(
    name="aerobench-f16",
    ext_modules=extensions,
    zip_safe=False,
)

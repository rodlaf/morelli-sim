from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="code.aerobench.highlevel.f16_model",
        sources=["code/aerobench/highlevel/f16_model.pyx"],
        include_dirs=["code/aerobench/highlevel", np.get_include()],
        language="c",
    )
]

setup(
    name="aerobench-f16-model",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)

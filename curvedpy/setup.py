from setuptools import setup, find_packages

VERSION = '0.0.2' 
DESCRIPTION = 'General Relativistic Geodesic Integrator'
LONG_DESCRIPTION = '...'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="curvedpy", 
        version=VERSION,
        author="B.L. de Vries",
        author_email="<bldevries@protonmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["numpy", "scipy", "sympy"], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'relativistic', 'ray', 'tracer', 'blackhole', 'astronomy', 'physics', 'numerical'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
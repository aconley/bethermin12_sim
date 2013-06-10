from distutils.core import setup

import sys
major, minor1, minor2, release, serial = sys.version_info

if (major < 3) and (minor1 < 7):
    raise SystemExit("bethermin12_sim requires at least python 2.7")

setup(
    name="bethermin12_sim",
    version="0.2.1",
    author="Alexander Conley",
    author_email="alexander.conley@colorado.edu",
    packages=["bethermin12_sim"],
    package_data={'bethermin12_sim':['resources/*fits']},
    license="GPL",
    description="Simulations of Bethermin et al. 2012 model",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    requires = ['numpy (>1.5.0)', 'scipy (>0.8.0)', 
                'astropy (>0.2.0)']
)


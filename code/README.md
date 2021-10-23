The code of optical flow process.
Some optical flow functions are implemented by CPP and Python.

# 1. CPP
Tested on GCC 8.3.

# 2. Python

`main.py` include the run and comparison code for BMVC 2021.

- `python -m pip uninstall panoopticalflow --yes`: uninstall the original package;
- `python -m build`: to build the `whl` package;
- `pip ./dist/panoopticalflow-1.0.0-py3-none-any.whl`: install build package.

The package provide following CLI script:

- `python -m panoopticalflow.visuazliation --task vis`: to visualize current folder `flo`, `pfm` and `dpt` files and save `jpg` image.

# 3. Matlab

# 4. Blender
Blender project file and script.
# Virtual 3D model of AIMotion-Lab and simulation with MuJoCo engine

The old documentation can be accessed [here](https://github.com/AIMotionLab-SZTAKI/Mujoco-Simulator/wiki).

## Installation
1. Create and activate a virtual environment

2. Install libraries with pip
```
$ pip install -e .
```
3. Navigate to scripts

```
$ cd scripts
```

4. Run one of the scripts like below
```
$ python3 testing.py
```

5. Generate docs (they will be generated in docs/build/html)
```
cd docs
make clean
make html
```
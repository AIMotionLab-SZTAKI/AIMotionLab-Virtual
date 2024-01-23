from setuptools import setup
from setuptools import find_packages

setup(name='aiml_virtual',
      packages=find_packages(),
      py_modules=["xml_generator"],
      version='1.0.0',
      install_requires=[
        #'absl-py==1.2.0',
        #'cycler==0.11.0',
        #'fonttools==4.37.3',
        'glfw==2.5.5',
        'matplotlib==3.6.0',
        #'gym==0.21',
        #'stable_baselines3==1.5.0',
        'mujoco==2.3',
        'numpy==1.23.3',
        #'packaging==21.3',
        #'Pillow==9.2.0',
        'PyOpenGL==3.1.6',
        #'pyparsing==3.0.9',
        #'python-dateutil==2.8.2',
        'scipy==1.10.1',
        #'six==1.16.0',
        #'torch==1.11.0',
        #'tensorboard==2.9.0',
        #'imageio==2.22.4',
        'opencv-python==4.9.0.80',
        'motioncapture==1.0a1; python_version<"3.10"',
        'motioncapture==1.0a2; python_version>"3.9"',
        'sympy==1.10.1',
        'mosek==9.3.21',
        'control==0.9.2',
        'cvxopt',  # @ https://github.com/AIMotionLab-SZTAKI/cvxopt/raw/mosek_handler/dist/cvxopt-0%2Buntagged.55.gc611b51.dirty-cp38-cp38-linux_x86_64.whl'
        'ffmpeg-python==0.2.0',
        'win-precise-time==1.4.2 ; platform_system=="Windows"',
        'windows-curses==2.3.1 ; platform_system=="Windows"',
        'numpy-stl==3.1.1',
        'casadi==3.6.4'
        ]
      )

from setuptools import setup

setup(name='aimotionlabvirtual',
      version='1.0.0',
      py_modules=['gui', 'util', 'classes'],
      #py_modules=['gui', 'util', 'classes', '../crazyflie-mujoco/ctrl'],
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
        #'opencv-python==4.6.0.66',
        'motioncapture==1.0a1; python_version<"3.10"',
        'motioncapture==1.0a2; python_version>"3.9"',
        'sympy==1.10.1',
        'mosek==9.3.21',
        'control==0.9.2',
        'cvxopt',  # @ https://github.com/AIMotionLab-SZTAKI/cvxopt/raw/mosek_handler/dist/cvxopt-0%2Buntagged.55.gc611b51.dirty-cp38-cp38-linux_x86_64.whl'
        'ffmpeg-python==0.2.0'
        ]
      )

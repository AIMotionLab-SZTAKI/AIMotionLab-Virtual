from setuptools import setup

setup(name='mujoco-simulation',
      version='1.0.0',
      py_modules=['gui', 'util', 'classes'],
      install_requires=[
        'absl-py==1.2.0',
        'contourpy==1.0.5',
        'cycler==0.11.0',
        'fonttools==4.37.3',
        'glfw==2.5.5',
        'kiwisolver==1.4.4',
        'matplotlib==3.6.0',
        'mujoco==2.3',
        'numpy==1.23.3',
        'packaging==21.3',
        'Pillow==9.2.0',
        'PyOpenGL==3.1.6',
        'pyparsing==3.0.9',
        'python-dateutil==2.8.2',
        'scipy==1.9.1',
        'six==1.16.0',
        #'imageio==2.22.4',
        'opencv-python==4.6.0.66',
        'motioncapture==1.0a1'
        ]
      )

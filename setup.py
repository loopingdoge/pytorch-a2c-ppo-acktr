from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='pytorch-a2c-ppo-acktr',
      packages=[package for package in find_packages()
                if package.startswith('pytorch-a2c-ppo-acktr')],
      install_requires=[
          'gym[mujoco,atari,classic_control,robotics]',
          'torch',
          'gym-retro',
          'baselines'
      ],
      description='A2C, PPO, ACKTR algorithms implemented in Pytorch',
      author='Hehe',
      url='https://github.com/loopingdoge/pytorch-a2c-ppo-acktr',
      author_email='wow@loopingdoge.io',
      version='0.1')
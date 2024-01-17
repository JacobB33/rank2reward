from setuptools import setup, find_packages
import os


def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


requirements = read_requirements_file("requirements.txt")
requirements.append("r3m @ git+https://github.com/facebookresearch/r3m.git")
setup(
    name="rank2reward",
    version="0.0.1",
    description="Rank2Reward, a inverse reinforcement learning toolkit",
    packages=find_packages(),
    install_requires=requirements
)

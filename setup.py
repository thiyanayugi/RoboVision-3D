"""Setup configuration for RoboVision-3D."""

from setuptools import setup, find_packages

setup(
    name="robovision-3d",
    version="1.0.0",
    author="Thiyanayugi Mariraj",
    description="Computer vision and robotics system for indoor mapping and 3D reconstruction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thiyanayugi/RoboVision-3D",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "ultralytics>=8.0",
        "torch>=2.0",
        "open3d>=0.17.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "scikit-learn>=1.3.0",
    ],
)

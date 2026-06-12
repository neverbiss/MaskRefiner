from setuptools import setup, find_packages

setup(
    name="maskrefiner",
    version="1.0.0",
    description="Universal mask refinement for Segment Anything Models",
    author="neverbiss",
    author_email="100753803+neverbiss@users.noreply.github.com",
    url="https://github.com/neverbiss/MaskRefiner",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.7",
        "torchvision>=0.8",
        "numpy>=1.19",
        "opencv-python>=4.5",
        "FastGeodis",
        "Pillow>=8.0",
        "scipy>=1.5",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "pytest",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)

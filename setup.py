from setuptools import setup, find_packages

setup(
    name="astrophoto-scripts",
    version="1.0.0",
    description="CLI pipeline for multi-exposure astrophotography, RAW/DNG, and landscape stacking.",
    author="T-Dynamos & Antigravity",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "tifffile>=2023.7.10",
        "rawpy>=0.20.0",
        "onnxruntime>=1.15.0",
    ],
    entry_points={
        "console_scripts": [
            "astrophoto=astrophoto.cli:main",
        ],
    },
    python_requires=">=3.9",
)

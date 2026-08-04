from setuptools import setup, find_packages

setup(
    name="arcfit",
    version="0.1.0",
    description="A package for fitting arcs in 2D data",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyFAI",
        "Pillow",
        "fabio",
    ],
    python_requires=">=3.6",
)


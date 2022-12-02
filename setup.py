from setuptools import setup

setup(
    name="guided_diffusion",
    py_modules=["guided_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)

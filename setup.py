import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_named_dims",
    version="0.1.0",
    author="Piotr Migdał",
    author_email="pmigdal@gmail.com",
    description="PyTorch tensor dimension names for all nn.Modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stared/pytorch-named-dims",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['torch>=1.4.0']
)

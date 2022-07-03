import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpquant",
    version="0.1.5",
    description="Genetic Programming for Quant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="UePG",
    author_email="hanlin.warng@gmail.com",
    url="https://github.com/UePG-21/gpquant",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

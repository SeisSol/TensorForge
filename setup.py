import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
        'numpy',
        'scipy']

setuptools.setup(
    name="gemmforge",
    version="0.0.1",
    license="MIT",
    author="Ravil Dorozhinskii",
    author_email="ravil.aviva.com@gmail.com",
    description="GPU-GEMM generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=["gemmforge"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=install_requires,
)
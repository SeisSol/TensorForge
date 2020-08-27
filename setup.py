import setuptools

with open("gemmforge/VERSION", "r") as version_file:
    current_version = version_file.read().strip()

 
with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = ['numpy']

setuptools.setup(
    name="gemmforge",
    version=current_version,
    license="MIT",
    author="Ravil Dorozhinskii",
    author_email="ravil.aviva.com@gmail.com",
    description="GPU-GEMM generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["gemmforge", 
              "gemmforge.matrix", 
              "gemmforge.loaders",
              "gemmforge.constructs",
              "gemmforge.initializers"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/ravil-mobile/gemmforge/wiki",
    python_requires='>=3.5',
    install_requires=install_requires,
    include_package_data=True,
)

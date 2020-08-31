## GPU-GEMM generator for the Discontinuous Galerkin method

## Installation
#### For users
```console
pip install gemmforge
```

#### For developers
```console
git clone https://github.com/ravil-mobile/gemmforge.git gemmforge
cd gemmforge
pip install -e .
```

## Usage
```python
from gemmforge import DenseMatrix, GemmGenerator, GenerationError
from gemmforge import arch

arch = arch.produce("nvidia", "sm_60")

mat_a = DenseMatrix(num_rows=56,
                    num_cols=9,
                    addressing="strided",
                    bbox=[0, 0, 55, 8],
                    transpose=False)

mat_b = DenseMatrix(num_rows=9,
                    num_cols=9,
                    addressing="strided",
                    bbox=[0, 0, 8, 8],
                    transpose=False)


mat_c = DenseMatrix(num_rows=56,
                    num_cols=9,
                    bbox=[0, 0, 55, 8],
                    addressing="strided",
                    transpose=False)
try:
    gen = GemmGenerator(arch, "float")
    gen.generate(mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)
    print(gen.get_kernel())
    print(gen.get_launcher())
    print(gen.get_launcher_header())

except GenerationError as err:
    print("ERROR: {}".format(err))
    raise err
```

## GPU-GEMM generator for the Discontinuous Galerkin method

## Installation
#### For users
```console
pip3 install gemmforge
```

#### For developers
```console
git clone https://github.com/ravil-mobile/gemmforge.git gemmforge
cd gemmforge
pip3 install -e .
```


## Usage
```python
from gemmforge import DenseMatrix, GenerationError, GemmGenerator
from gemmforge.vm import vm_factory


mat_a = DenseMatrix(num_rows=56,
                    num_cols=9,
                    addressing="strided",
                    bbox=[0, 0, 56, 9])

mat_b = DenseMatrix(num_rows=9,
                    num_cols=9,
                    addressing="strided",
                    bbox=[0, 0, 9, 9])


mat_c = DenseMatrix(num_rows=56,
                    num_cols=9,
                    bbox=[0, 0, 56, 9],
                    addressing="strided")

try:
    vm = vm_factory(arch="sm_60", backend="cuda", fp_type="float")
    gen = GemmGenerator(vm)

    gen.set(False, False, mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)
    gen.generate()
    print(gen.get_kernel())
    print(gen.get_launcher())
    print(gen.get_launcher_header())

except GenerationError as err:
    print(f'ERROR: {err}')
    raise err
```

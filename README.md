[![Build Status](http://vmbungartz10.informatik.tu-muenchen.de/seissol/buildStatus/icon?job=gemmforge)](http://vmbungartz10.informatik.tu-muenchen.de/seissol/view/Forge/job/gemmforge/)
[![Build Status](http://vmbungartz10.informatik.tu-muenchen.de/seissol/buildStatus/icon?job=gemmforge-install&subject=pip)](http://vmbungartz10.informatik.tu-muenchen.de/seissol/view/Forge/job/gemmforge-install/)
[![Build Status](http://vmbungartz10.informatik.tu-muenchen.de/seissol/buildStatus/icon?job=gemmforge-linters&subject=Code+Style)](http://vmbungartz10.informatik.tu-muenchen.de/seissol/view/Forge/job/gemmforge-linters/)


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
    vm = vm_factory(name="nvidia", sub_name="sm_60", fp_type="float")
    gen = GemmGenerator(vm)

    gen.set(mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)
    gen.generate()
    print(gen.get_kernel())
    print(gen.get_launcher())
    print(gen.get_launcher_header())

except GenerationError as err:
    print(f'ERROR: {err}')
    raise err
```

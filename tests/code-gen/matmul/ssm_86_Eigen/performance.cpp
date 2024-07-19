#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "kernel.h"
#include "tensor.h"
#include "Stopwatch.h"
#include "Util.h"
using namespace tensorforge;
void trashTheCache(double* trash, int size);
int main(int argc, char** argv) {
  int _fixedReps = (argc >= 2) ? atoi(argv[1]) : -1;
  int _reps, _error;
  Stopwatch _sw;
  double _time, _nzflops, _flops;
  printf("kernel,repetitions,time,numnzflop,numflop,nzgflops,gflops\n");
  {
    real* C__;
    _error = posix_memalign(reinterpret_cast<void**>(&C__), ALIGNMENT, tensor::C::size()*sizeof(real));
    real* A__;
    _error = posix_memalign(reinterpret_cast<void**>(&A__), ALIGNMENT, tensor::A::size()*sizeof(real));
    real* B__;
    _error = posix_memalign(reinterpret_cast<void**>(&B__), ALIGNMENT, tensor::B::size()*sizeof(real));
    fillWithStuff(C__, tensor::C::size());
    fillWithStuff(A__, tensor::A::size());
    fillWithStuff(B__, tensor::B::size());
    _reps = _fixedReps;
    if (_reps < 0) {
      _reps = ceil(40000000000.0/kernel::matmulAB::HardwareFlops);
    }
    kernel::matmulAB _kernel_matmulAB;
    _kernel_matmulAB.C = C__;
    _kernel_matmulAB.A = A__;
    _kernel_matmulAB.B = B__;
    _kernel_matmulAB.execute();
    _sw.start();
    for (int i = 0; i < _reps; ++i) {
      _kernel_matmulAB.execute();
    }
    _time = _sw.stop();
    _nzflops = static_cast<double>(kernel::matmulAB::NonZeroFlops) * _reps / _time / 1.0e9;
    _flops = static_cast<double>(kernel::matmulAB::HardwareFlops) * _reps / _time / 1.0e9;
    printf("matmulAB,%u,%lf,%lu,%lu,%lf,%lf\n", _reps, _time, kernel::matmulAB::NonZeroFlops, kernel::matmulAB::HardwareFlops, _nzflops, _flops);
    free(C__);
    free(A__);
    free(B__);
  }
  {
    real* C__;
    _error = posix_memalign(reinterpret_cast<void**>(&C__), ALIGNMENT, tensor::C::size()*sizeof(real));
    real* A__;
    _error = posix_memalign(reinterpret_cast<void**>(&A__), ALIGNMENT, tensor::A::size()*sizeof(real));
    real* B__;
    _error = posix_memalign(reinterpret_cast<void**>(&B__), ALIGNMENT, tensor::B::size()*sizeof(real));
    fillWithStuff(C__, tensor::C::size());
    fillWithStuff(A__, tensor::A::size());
    fillWithStuff(B__, tensor::B::size());
    _reps = _fixedReps;
    if (_reps < 0) {
      _reps = ceil(40000000000.0/kernel::matmulATB::HardwareFlops);
    }
    kernel::matmulATB _kernel_matmulATB;
    _kernel_matmulATB.C = C__;
    _kernel_matmulATB.A = A__;
    _kernel_matmulATB.B = B__;
    _kernel_matmulATB.execute();
    _sw.start();
    for (int i = 0; i < _reps; ++i) {
      _kernel_matmulATB.execute();
    }
    _time = _sw.stop();
    _nzflops = static_cast<double>(kernel::matmulATB::NonZeroFlops) * _reps / _time / 1.0e9;
    _flops = static_cast<double>(kernel::matmulATB::HardwareFlops) * _reps / _time / 1.0e9;
    printf("matmulATB,%u,%lf,%lu,%lu,%lf,%lf\n", _reps, _time, kernel::matmulATB::NonZeroFlops, kernel::matmulATB::HardwareFlops, _nzflops, _flops);
    free(C__);
    free(A__);
    free(B__);
  }
  {
    real* C__;
    _error = posix_memalign(reinterpret_cast<void**>(&C__), ALIGNMENT, tensor::C::size()*sizeof(real));
    real* A__;
    _error = posix_memalign(reinterpret_cast<void**>(&A__), ALIGNMENT, tensor::A::size()*sizeof(real));
    real* B__;
    _error = posix_memalign(reinterpret_cast<void**>(&B__), ALIGNMENT, tensor::B::size()*sizeof(real));
    fillWithStuff(C__, tensor::C::size());
    fillWithStuff(A__, tensor::A::size());
    fillWithStuff(B__, tensor::B::size());
    _reps = _fixedReps;
    if (_reps < 0) {
      _reps = ceil(40000000000.0/kernel::matmulABT::HardwareFlops);
    }
    kernel::matmulABT _kernel_matmulABT;
    _kernel_matmulABT.C = C__;
    _kernel_matmulABT.A = A__;
    _kernel_matmulABT.B = B__;
    _kernel_matmulABT.execute();
    _sw.start();
    for (int i = 0; i < _reps; ++i) {
      _kernel_matmulABT.execute();
    }
    _time = _sw.stop();
    _nzflops = static_cast<double>(kernel::matmulABT::NonZeroFlops) * _reps / _time / 1.0e9;
    _flops = static_cast<double>(kernel::matmulABT::HardwareFlops) * _reps / _time / 1.0e9;
    printf("matmulABT,%u,%lf,%lu,%lu,%lf,%lf\n", _reps, _time, kernel::matmulABT::NonZeroFlops, kernel::matmulABT::HardwareFlops, _nzflops, _flops);
    free(C__);
    free(A__);
    free(B__);
  }
  {
    real* C__;
    _error = posix_memalign(reinterpret_cast<void**>(&C__), ALIGNMENT, tensor::C::size()*sizeof(real));
    real* B__;
    _error = posix_memalign(reinterpret_cast<void**>(&B__), ALIGNMENT, tensor::B::size()*sizeof(real));
    real* A__;
    _error = posix_memalign(reinterpret_cast<void**>(&A__), ALIGNMENT, tensor::A::size()*sizeof(real));
    fillWithStuff(C__, tensor::C::size());
    fillWithStuff(B__, tensor::B::size());
    fillWithStuff(A__, tensor::A::size());
    _reps = _fixedReps;
    if (_reps < 0) {
      _reps = ceil(40000000000.0/kernel::matmulATBT::HardwareFlops);
    }
    kernel::matmulATBT _kernel_matmulATBT;
    _kernel_matmulATBT.C = C__;
    _kernel_matmulATBT.B = B__;
    _kernel_matmulATBT.A = A__;
    _kernel_matmulATBT.execute();
    _sw.start();
    for (int i = 0; i < _reps; ++i) {
      _kernel_matmulATBT.execute();
    }
    _time = _sw.stop();
    _nzflops = static_cast<double>(kernel::matmulATBT::NonZeroFlops) * _reps / _time / 1.0e9;
    _flops = static_cast<double>(kernel::matmulATBT::HardwareFlops) * _reps / _time / 1.0e9;
    printf("matmulATBT,%u,%lf,%lu,%lu,%lf,%lf\n", _reps, _time, kernel::matmulATBT::NonZeroFlops, kernel::matmulATBT::HardwareFlops, _nzflops, _flops);
    free(C__);
    free(B__);
    free(A__);
  }
  return 0;
}

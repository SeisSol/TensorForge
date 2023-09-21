#ifndef KERNELFORGE_TYPEDEF_H
#define KERNELFORGE_TYPEDEF_H

#include <vector>

#if REAL_SIZE == 8
typedef double real;
#elif REAL_SIZE == 4
typedef float real;
#else
#  error REAL_SIZE not supported.
#endif

using PackedData = std::vector<std::vector<real>>;

#endif // KERNELFORGE_TYPEDEF_H

#ifndef SIMPLE_BANDWIDTH_TEST_KERNEL_H
#define SIMPLE_BANDWIDTH_TEST_KERNEL_H

#include <stdio.h>

void copyData(float *To, float *From, size_t size, size_t blocks, size_t threads, void* stream);

#endif //SIMPLE_BANDWIDTH_TEST_KERNEL_H

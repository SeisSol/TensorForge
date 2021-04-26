#ifndef SIMPLE_BANDWIDTH_TEST_KERNEL_H
#define SIMPLE_BANDWIDTH_TEST_KERNEL_H

#include <stdio.h>

void copyData(float *To, float *From, size_t size, long blocks, long threads, void* stream);

#endif //SIMPLE_BANDWIDTH_TEST_KERNEL_H

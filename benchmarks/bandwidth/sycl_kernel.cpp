#include "kernel.h"
#include <CL/sycl.hpp>

void copyData(float *To, float *From, size_t size, long blocks, long threads, void *stream) {
    ((cl::sycl::queue *)stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(cl::sycl::nd_range<1> {blocks*threads, threads}, [=](cl::sycl::nd_item<1> item) {
            if (item.get_global_id(0) < size) {
                To[item.get_global_id(0)] = From[item.get_global_id(0)];
            }
        });
    });
}
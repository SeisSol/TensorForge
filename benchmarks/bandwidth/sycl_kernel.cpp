#include "kernel.h"
#include <CL/sycl.hpp>
#include <device.h>

using namespace device;

void copyData(float *To, float *From, size_t size, size_t blocks, size_t threads, void *stream) {
    auto device = &DeviceInstance::getInstance();
    long workItemSize = device->api->getMaxThreadBlockSize();

    ((cl::sycl::queue *)stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(cl::sycl::nd_range<1> {blocks*threads, workItemSize}, [=](cl::sycl::nd_item<1> item) {
            if (item.get_global_id(0) < size) {
                To[item.get_global_id(0)] = From[item.get_global_id(0)];
            }
        });
    });
}
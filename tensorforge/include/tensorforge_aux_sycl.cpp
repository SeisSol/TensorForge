#include <iostream>
#include <sycl/sycl.hpp>

namespace tensorforge {
void checkErr(const std::string &File, int Line) {}

void syncDevice(void *stream) {
  if (stream == nullptr) {
    throw std::invalid_argument("cant sync device without queue!");
  }

  ((sycl::queue *)stream)->wait();
}
} // namespace tensorforge

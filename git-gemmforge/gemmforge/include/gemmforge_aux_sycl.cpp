#include <iostream>
#include <CL/sycl.hpp>

namespace gemmforge {
    void checkErr(const std::string &File, int Line) {

    }

  void synchDevice(void *stream) {
      if(stream == nullptr)
        throw std::invalid_argument("cant sync device without queue!");

      ((cl::sycl::queue *)stream)->wait();
  }
}



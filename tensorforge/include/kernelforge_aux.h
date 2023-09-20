#ifndef KERNELFORGE_INTERNALS_H
#define KERNELFORGE_INTERNALS_H

#include <string>

#define CHECK_ERR kernelforge::checkErr(__FILE__,__LINE__)
namespace kernelforge {
  void checkErr(const std::string &file, int line);
  void synchDevice(void *stream = nullptr);
}

#endif  // KERNELFORGE_INTERNALS_H
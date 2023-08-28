#ifndef GEMMFORGE_INTERNALS_H
#define GEMMFORGE_INTERNALS_H

#include <string>

#define CHECK_ERR gemmforge::checkErr(__FILE__,__LINE__)
namespace gemmforge {
  void checkErr(const std::string &file, int line);
  void synchDevice(void *stream = nullptr);
}

#endif  // GEMMFORGE_INTERNALS_H
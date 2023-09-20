#ifndef TENSORFORGE_INTERNALS_H
#define TENSORFORGE_INTERNALS_H

#include <string>

#define CHECK_ERR tensorforge::checkErr(__FILE__,__LINE__)
namespace tensorforge {
  void checkErr(const std::string &file, int line);
  void synchDevice(void *stream = nullptr);
}

#endif  // TENSORFORGE_INTERNALS_H
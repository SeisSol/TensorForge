
#ifndef GEMMGEN_INTERNALS_H
#define GEMMGEN_INTERNALS_H

#include <string>

#define CHECK_ERR gemmgen::checkErr(__FILE__,__LINE__)
namespace gemmgen {
  void checkErr(const std::string &file, int line);
  void synchDevice();
}

#endif  // GEMMGEN_INTERNALS_H
#include <iostream>
#include <omp.h>

namespace tensorforge
{
  std::string PrevFile = "";
  int PrevLine = 0;

  void checkErr(const std::string &File, int Line)
  {
#ifndef NDEBUG

#endif
  }

  void synchDevice(void *stream)
  {
    auto *realstream = static_cast<int *>(stream);
#pragma omp taskwait depend(inout : realstream[0])
    checkErr(__FILE__, __LINE__);
  }
}

// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#include "tensorforge_aux.h"

#include <hip/hip_runtime.h>
#include <iostream>

namespace tensorforge {
std::string PrevFile = "";
int PrevLine = 0;

void checkErr(const std::string &File, int Line) {
#ifndef NDEBUG
  hipError_t Error = hipGetLastError();
  if (Error != hipSuccess) {
    std::cout << std::endl
              << File << ", line " << Line << ": " << hipGetErrorString(Error)
              << " (" << Error << ")" << std::endl;

    if (PrevLine > 0)
      std::cout << "Previous HIP call:" << std::endl
                << PrevFile << ", line " << PrevLine << std::endl;
    throw;
  }
  PrevFile = File;
  PrevLine = Line;
#endif
}

void syncDevice(void *stream) {
  CHECK_RES(hipDeviceSynchronize());
  checkErr(__FILE__, __LINE__);
}
} // namespace tensorforge

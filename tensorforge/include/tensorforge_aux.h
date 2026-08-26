// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_AUX_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_AUX_H_

#include <array>
#include <string>

#include "tensorforge_device/base.h"

#define CHECK_ERR tensorforge::checkErr(__FILE__, __LINE__)

#if 0
#define CHECK_RES(call) tensorforge::checkRes(__FILE__, __LINE__, (call))
#else
#define CHECK_RES(call) (void)(call);
#endif

namespace tensorforge {

void checkErr(const std::string &file, int line);
void syncDevice(void *stream = nullptr);

template <typename... Args>
std::array<void *, sizeof...(Args)> argsPtrs(Args &...args) {
  return std::array<void *, sizeof...(Args)>{&args...};
}

} // namespace tensorforge

#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_AUX_H_

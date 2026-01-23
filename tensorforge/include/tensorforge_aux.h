#ifndef TENSORFORGE_INTERNALS_H
#define TENSORFORGE_INTERNALS_H

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

#endif // TENSORFORGE_INTERNALS_H

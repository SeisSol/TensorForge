#ifndef TENSORFORGE_INTERNALS_H
#define TENSORFORGE_INTERNALS_H

#include <string>

#define CHECK_ERR tensorforge::checkErr(__FILE__, __LINE__)

#if 0
#define CHECK_RES(call) tensorforge::checkRes(__FILE__, __LINE__, (call))
#else
#define CHECK_RES(call) (void)(call);
#endif

namespace tensorforge {

void checkErr(const std::string &file, int line);
void synchDevice(void *stream = nullptr);

} // namespace tensorforge

#endif // TENSORFORGE_INTERNALS_H

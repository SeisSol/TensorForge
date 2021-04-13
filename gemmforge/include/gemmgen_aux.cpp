#include <iostream>
#include <hip/hip_runtime.h>

namespace gemmgen {
    std::string PrevFile = "";
    int PrevLine = 0;

    void checkErr(const std::string &File, int Line) {
      hipError_t Error = hipGetLastError();
      if (Error != hipSuccess) {
        std::cout << std::endl << File 
                  << ", line " << Line
                  << ": " << hipGetErrorString(Error) 
                  << " (" << Error << ")" 
                  << std::endl;
                  
        if (PrevLine > 0)
          std::cout << "Previous HIP call:" << std::endl
                    << PrevFile << ", line " << PrevLine << std::endl;
        throw;
      }
      PrevFile = File;
      PrevLine = Line;
    }

  void synchDevice() {
    hipDeviceSynchronize();
    checkErr(__FILE__, __LINE__);
  }
}



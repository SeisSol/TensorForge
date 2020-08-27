#include <iostream>
#include <cuda_runtime.h>

namespace gemmgen {
    std::string PrevFile = "";
    int PrevLine = 0;

    void checkErr(const std::string &File, int Line) {
      cudaError_t Error = cudaGetLastError();
      if (Error != cudaSuccess) {
        std::cout << std::endl << File 
                  << ", line " << Line
                  << ": " << cudaGetErrorString(Error) 
                  << " (" << Error << ")" 
                  << std::endl;
                  
        if (PrevLine > 0)
          std::cout << "Previous CUDA call:" << std::endl
                    << PrevFile << ", line " << PrevLine << std::endl;
        throw;
      }
      PrevFile = File;
      PrevLine = Line;
    }

  void synchDevice() {
    cudaDeviceSynchronize();
    checkErr(__FILE__, __LINE__);
  }
}



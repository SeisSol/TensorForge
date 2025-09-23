from ..kernel import SingleSourceKernel

class CudaKernel(SingleSourceKernel):
    def kernellaunch(self):
        if self.gridsync:
            return f'cudaLaunchCooperativeKernel(kernel_{self.name}, grid, block, {argsref}, {self.shmem}, stream);'
        else:
            return f'kernel_{self.name}<<<grid, block, {self.shmem}, stream>>>({argspass});'

    def source(self):
        return f"""
__global__ void kernel_{self.name}({args}) {{
{self.code}
}}
"""

    def persistent(self):
        return f"""
static int launchsize_{self.name}() {{
    static int initialized = 0;
    if (initialized == 0) {{
        cudaFuncSetAttribute(kernel_{self.name}, cudaFuncAttributeMaxDynamicSharedMemorySize, {self.shmem});
        CHECK_ERR;
        initialized = true;
    }}
    return initialized;
}}
"""

    def cpp(self):

        return f"""
static void init_{self.name}() {{
    static bool initialized = false;
    if (!initialized) {{
        cudaFuncSetAttribute(kernel_{self.name}, cudaFuncAttributeMaxDynamicSharedMemorySize, {self.shmem});
        CHECK_ERR;
        initialized = true;
    }}
}}

void launch_{self.name}(void* streamPtr, size_t numElements, {args}) {{
    init_{self.name}();
    auto stream = static_cast<cudaStream_t>(streamPtr);
    dim3 block({self.num_threads}, 1, 1);
    dim3 grid({self.numElements}, 1, 1);
    {self.kernellaunch()}
}}
"""

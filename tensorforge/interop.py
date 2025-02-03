def get_cmake_path():
    import os
    mydir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(mydir, 'share', 'cmake')

def print_cmake_path():
    print(get_cmake_path())

def get_routine_generator(yateto):
    import tensorforge.codegen.gpukernel as gk
    return gk.GpuKernelRoutineGenerator

def get_version():
    import os
    mydir = os.path.dirname(os.path.realpath(__file__))
    with file(os.path.join(mydir, 'VERSION')):
        return file.read()

def print_version():
    print(get_version())

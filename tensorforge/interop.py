def get_cmake_path():
    import os
    mydir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(mydir, 'share', 'cmake')

def print_cmake_path():
    print(get_cmake_path(), end='')

def get_routine_generator(yateto):
    import tensorforge.frontend.yateto as fe
    return fe.YatetoFrontend

def get_version():
    import os
    mydir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(mydir, 'VERSION')) as file:
        return file.read()

def print_version():
    print(get_version(), end='')

def use_fusedgemm_cost():
    return True

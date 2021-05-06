from .stub_initializer import StubInitializer
from .initializer import ExactInitializer
from gemmforge.vm import VM


def initializer_factory(vm: VM, init_value, matrix):
    if init_value == 1.0:
        return StubInitializer(vm)
    else:
        return ExactInitializer(vm, init_value, matrix)

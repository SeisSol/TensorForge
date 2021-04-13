from .stub_initializer import StubInitializer
from .abstract_initializer import ExactInitializer


def initializer_factory(init_value, matrix, arch, precision):
    if init_value == 1.0:
        return StubInitializer(arch, precision)
    else:
        return ExactInitializer(init_value, matrix, arch, precision)

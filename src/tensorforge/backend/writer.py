# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: Carsten Uphoff
# SPDX-FileContributor: David Schneller

import sys

from io import StringIO

class VarAlloc:
    def __init__(self):
        self.counter = -1

    def alloc(self, prefix='v'):
        self.counter += 1
        return f'{prefix}{self.counter}'

    def next_index(self):
        """Hand out a raw index, so an IRBuilder can share this counter."""
        self.counter += 1
        return self.counter

class Block:
  def __init__(self, writer, argument, foot=''):
    self.writer = writer
    self.argument = argument
    self.foot = foot

  def __enter__(self):
    space = ' ' if self.argument else ''
    self.writer.speculate(self.argument + space + '{')
    self.writer.indent += 1

  def __exit__(self, type, value, traceback):
    self.writer.indent -= 1
    self.writer.speculateClear('}' + self.foot)

  def __call__(self, line):
    self.writer(line)


class MultiBlock:
  def __init__(self, writer, arguments, foot=None):
    self.writer = writer
    self.arguments = arguments
    if foot is None:
      self.foot = [''] * len(self.arguments)
    else:
      self.foot = foot

  def __enter__(self):
    for arg in self.arguments:
      self.writer(arg + ' {')
      self.writer.indent += 1

  def __exit__(self, type, value, traceback):
    # Blocks are closed in reverse order, thus reverse footer
    for arg, foot in zip(self.arguments, reversed(self.foot)):
      self.writer.indent -= 1
      self.writer('}' + foot)


class Writer:
  def __init__(self, stream=sys.stdout, factor=2):
    self.stream = StringIO()
    self.indent = 0
    self.factor = 2
    self.alloc = VarAlloc()
    self.stack = []

  def varalloc(self, prefix='v'):
    # TODO: maybe move out?
    return self.alloc.alloc(prefix)

  def get_src(self):
    return self.stream.getvalue()

  def __enter__(self):
    self.out = open(self.stream, 'w+') if isinstance(self.stream, str) else self.stream
    return self

  def __exit__(self, type, value, traceback):
    if self.out is not sys.stdout:
      self.stream.close()
    self.out = None

  def __call__(self, code):
    white_spaces = (' ' * self.factor) * self.indent
    for substack in self.stack:
      for deferred in substack:
        self.stream.write(deferred)
    self.stack = []
    for line in code.splitlines():
      self.stream.write(white_spaces + line + '\n')

  def speculate(self, code):
    white_spaces = (' ' * self.factor) * self.indent
    substack = []
    for line in code.splitlines():
      substack += [white_spaces + line + '\n']
    self.stack += [substack]

  def speculateClear(self, code):
    if len(self.stack) > 0:
      self.stack.pop()
    else:
      self(code)

  def new_line(self):
    self.__call__('')

  def Emptyline(self):
    self.stream.write('\n')

  def Block(self, text):
    return Block(self, text)

  def Scope(self):
    return Block(self, '')

  def If(self, expression):
    return Block(self, 'if ({})'.format(expression))

  def For(self, argument, unroll=False):
    # `unroll` is `True` for the bare pragma or a count for `#pragma unroll N`;
    # see `pir.build._unroll_pragma`, which both writers share.
    from tensorforge.backend.pir.build import _unroll_pragma
    return Block(self, '{}for ({})'.format(_unroll_pragma(unroll), argument))

  def While(self, argument):
    return Block(self, 'while ({})'.format(argument))

  def AnonymousScope(self):
    return Block(self, '')

  def Expression(self, expression):
    return self.__call__("{};".format(expression))

  def Function(self, name, arguments='', returnType='void', const=False):
    return Block(self, '{} {}({}){}'.format(returnType, name, arguments, ' const' if const else ''))

  def Comment(self, text):
    return self.__call__("// {}".format(text))

  def Pragma(self, name):
    return self.__call__("#pragma {}".format(name))

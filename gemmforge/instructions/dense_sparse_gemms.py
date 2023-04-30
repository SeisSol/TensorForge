from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.symbol_table import SymbolType, Symbol, DataView, InverseSymbolTable
from gemmforge.basic_types import GeneralLexicon, DataFlowDirection, RegMemObject
from gemmforge.exceptions import InternalError
from abc import abstractmethod


class ShrMemBasedDenseSparseGemm(AbstractInstruction):
  def __init__(self, **kwargs):
    raise Exception("TODO")

  def gen_code(self, writer):
    raise Exception("TODO!")

  def _get_inner_loop(self, writer, op1_value):
    raise Exception("TODO")

  def __str__(self) -> str:
    raise Exception("TODO")


class RegisterOnlyDenseSparseGemm(AbstractInstruction):
  def __init__(self, **kwargs):
    raise Exception("TODO!")

  def gen_code(self, writer):
    raise Exception("TODO!")

  def __str__(self) -> str:
    raise Exception("TODO!")
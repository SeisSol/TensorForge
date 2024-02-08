from kernelforge import DenseMatrix
from itertools import product
import functools
from copy import deepcopy


class LoaderError(Exception):
  pass


class TestLoader:
  def __init__(self, test_spec):
    self._test_spec = test_spec
    self._param_iterator = None
    self._analyze()

  def __iter__(self):
    return self

  def __next__(self):
    test_params = next(self._param_iterator)
    spec = deepcopy(self._test_spec)
    for param in test_params:
      _set_value(spec, param, test_params[param])

    return (self._produce_matrix(spec["matrix_a"]),
            self._produce_matrix(spec["matrix_c"]),
            spec["alpha"],
            spec["beta"],
            spec["num_elements"],
            self._gen_test_name(test_params))

  def _produce_matrix(self, matrix_spec):
    return DenseMatrix(num_rows=matrix_spec["rows"],
                       num_cols=matrix_spec["cols"],
                       addressing=matrix_spec["addressing"],
                       bbox=matrix_spec["bbox"])

  def is_param(self, param):
    if isinstance(param, str):
      if param.find('param') != -1:
        return True
      else:
        return False
    else:
      return False

  def _analyze(self):
    flatten_spec = {}
    _build_flatten_table(flatten_spec, self._test_spec)

    params = {}
    for item in flatten_spec:
      if (self.is_param(flatten_spec[item])):
        params[item] = self._test_spec[flatten_spec[item]]

    self._param_iterator = (dict(zip(params, x)) for x in product(*params.values()))

  def _gen_test_name(self, params):
    param_to_str = []
    for item in params:
      item_str = "_".join(item)
      value_str = params[item]
      if isinstance(params[item], float):
        value_str = str(params[item]).replace('.', '_')
      if isinstance(params[item], list):
        value_str = [str(item) for item in params[item]]
        value_str = "_".join(value_str)
      param_to_str.append("{}_{}".format(item_str, value_str))

    return "{}_{}".format(self._test_spec['test_base_name'], "_".join(param_to_str))


def _build_flatten_table(flatten_table, original_table, combo_key=()):
  if isinstance(original_table, dict):
    for key in original_table:
      _build_flatten_table(flatten_table, original_table[key], (*combo_key, key))
  else:
    flatten_table[combo_key] = original_table


def _set_value(table, combo_key, value):
  if len(combo_key) == 1:
    table[combo_key[0]] = value
  else:
    _set_value(table[combo_key[0]], combo_key[1:], value)


def _get_value(table, combo_key):
  return functools.reduce(dict.get, combo_key, table)

from .basic_types import GeneralLexicon


def get_extra_offset_name(base_name: str):
  return f'{base_name}{GeneralLexicon.EXTRA_OFFSET}'

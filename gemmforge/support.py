import os
import shutil


def copy_includes_and_aux(destination, name_as=None):
  if os.path.isdir(destination):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    top_dir, _ = os.path.split(current_dir)

    src = os.path.join(top_dir, "include")
    trg = name_as if name_as else "include"
    trg = os.path.join(destination, trg)

    if os.path.exists(trg):
      raise FileExistsError(f'a directory with a name ({name_as}) is exists in {destination}')

    shutil.copytree(src, trg)

  else:
    raise NotADirectoryError(f'there is not such dir as {destination}')


def print_cmake_path():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  print(os.path.join(current_dir, "share/cmake"), end='')


def get_version():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(current_dir, 'VERSION')) as version_file:
    version = version_file.read().strip()
    print(version, end='')
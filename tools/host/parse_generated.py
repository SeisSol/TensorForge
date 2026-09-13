# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Read a generated kernel's descriptor list back out of its comments.

The generator writes the list it was given above the code it produced, which
makes a shipped `gpulike_subroutine.cpp` a record of what the frontend asked
for -- enough to run the analysis on a real kernel without a frontend, a
matrices directory or a GPU.

Read rather than trusted: what comes back is shapes, boxes, addressing and
index maps, which is what the structural questions are asked in terms of.
Values, sparsity and alignment are not in the comment and are not invented
here, so a merge this reports has still to be confirmed against the real
descriptors.  For telling one repeated step from six different ones that is
enough, because steps that differ do so in their shapes and their operands.
"""

import argparse
import re
import sys

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

ADDRESSING = {
    'pointer_based': Addressing.PTR_BASED,
    'strided': Addressing.STRIDED,
    'none': Addressing.NONE,
    'scalar': Addressing.SCALAR,
}

OPERAND = re.compile(
    r'(?P<name>\w+) '
    r'(?P<shape>[\d×]*)\((?P<storage>[\d×]*)\) '
    r'(?P<box>[{}\d.×]*) *'
    r'(?P<addressing>pointer_based|strided|none|scalar)'
    r'\((?P<effective>[{}\d.×]*)\)'
    r'\[(?P<target>[-\d, ]*)\]')


def _dims(text):
    return [int(d) for d in text.split('×')] if text else []


def _box(text, rank):
    if not text:
        return BoundingBox([0], [1])
    parts = re.findall(r'\{(\d+)\.\.(\d+)\}', text)
    if not parts:
        return BoundingBox([0] * rank, [1] * rank)
    return BoundingBox([int(lo) for lo, _ in parts],
                       [int(hi) for _, hi in parts])


def _target(text):
    return [int(t) for t in text.split(',')] if text.strip() else []


def parse_operand(match, pool):
    shape = _dims(match['shape']) or [1]
    box = _box(match['box'], len(shape))
    name = match['name']
    key = (name, tuple(shape))
    if key not in pool:
        pool[key] = Tensor(shape, ADDRESSING[match['addressing']], box,
                           alias=name, is_tmp=name.startswith('t'),
                           datatype=Datatype.F32)
    return SubTensor(pool[key], box), _target(match['target'])


#: The kernel's metadata as one JSON line (`Generator.kernel_info`).
META = re.compile(r'//\s*tensorforge-meta:\s*(\{.*\})\s*$', re.M)


def _meta_view(row, pool, tensors):
    """One operand of an operation: the tensor from `operands` where the
    kernel lists it there, and from the view itself for a temporary."""
    shape = row['shape'] or [1]
    name = row['name']
    if name in tensors:
        tbox = BoundingBox(*tensors[name]['bbox'])
    else:
        lower = [l + o for l, o in zip(row['bbox'][0], row['offset'])]
        upper = [u + o for u, o in zip(row['bbox'][1], row['offset'])]
        tbox = BoundingBox(lower, upper)
    key = (name, tuple(shape))
    if key not in pool:
        pool[key] = Tensor(shape, ADDRESSING[row['addressing']], tbox,
                           alias=name, is_tmp=bool(row.get('is_tmp')),
                           datatype=Datatype.F32)
    return SubTensor(pool[key], BoundingBox(*row['bbox']))


def parse_meta(text, pool=None):
    """The multilinear operations of a kernel carrying a `tensorforge-meta`
    line, or None for one that does not (generated before it existed)."""
    import json
    found = META.search(text)
    if found is None:
        return None
    pool = {} if pool is None else pool
    meta = json.loads(found.group(1))
    tensors = {o['name']: o for o in meta.get('operands', [])}
    descrs = []
    for row in meta.get('operations', []):
        if row.get('kind') != 'multilinear':
            continue
        descrs.append(MultilinearDescr(
            _meta_view(row['dest'], pool, tensors),
            [_meta_view(o, pool, tensors) for o in row['ops']],
            row['target'], row['permute'], row['add']))
    return descrs


def parse_kernel(text, pool=None):
    """Every operation stated in one kernel's comment block, in order.

    From the `tensorforge-meta` line where the kernel has one, and from the
    descriptor lines older kernels carry otherwise."""
    pool = {} if pool is None else pool
    meta = parse_meta(text, pool)
    if meta is not None:
        return meta
    descrs = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('//') or ' = ' not in line and ' += ' not in line:
            continue
        body = line[2:].strip()
        add = ' += ' in body
        left, right = body.split(' += ' if add else ' = ', 1)

        dest = OPERAND.match(left.strip())
        if dest is None:
            continue
        dest_view, dest_target = parse_operand(dest, pool)

        # The operands are separated by the same multiplication sign that
        # sits inside every shape and every box, so they are found rather
        # than split apart.
        ops, targets = [], []
        for found in OPERAND.finditer(right):
            view, target = parse_operand(found, pool)
            ops.append(view)
            targets.append(target)
        if not ops:
            continue

        descrs.append(MultilinearDescr(
            dest_view, ops, targets,
            [list(range(len(t))) for t in targets],
            add))
    return descrs


def split_kernels(path):
    """`{name: comment block}` for every kernel in a generated file."""
    out, name, buffer = {}, None, []
    start = re.compile(r'\s*kernel_(\w+)\(')
    with open(path, errors='replace') as handle:
        for line in handle:
            found = start.match(line)
            if found:
                if name is not None:
                    out[name] = '\n'.join(buffer)
                name, buffer = found.group(1), []
            elif name is not None and line.lstrip().startswith('//'):
                buffer.append(line)
    if name is not None:
        out[name] = '\n'.join(buffer)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', help='a generated gpulike_subroutine.cpp')
    ap.add_argument('--kernel', help='one kernel by name; default is the largest')
    args = ap.parse_args()

    blocks = split_kernels(args.source)
    if not blocks:
        sys.exit('no kernels found')

    if args.kernel:
        name = args.kernel
    else:
        name = max(blocks, key=lambda k: len(parse_kernel(blocks[k])))

    descrs = parse_kernel(blocks[name])
    print(f'kernel_{name}: {len(descrs)} operation(s)')
    for i, descr in enumerate(descrs):
        ops = ' x '.join(o.tensor.alias for o in descr.ops)
        print(f'  {i:3d}  {descr.dest.tensor.alias:>4} '
              f'{"+=" if descr.add else " ="} {ops}')


if __name__ == '__main__':
    main()

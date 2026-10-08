# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""The target a kernel is generated for: a device, and the language it is
written in for that device.

Three parts, and they answer different questions:

* `hw`, the device's row of `hw_descr_db.yml`: the wave, the register files,
  shared memory, the instruction cache.  What the silicon has.
* `lexic`, the spelling: how a statement the generator has decided on is
  written in CUDA, HIP, SYCL, ESIMD or OpenMP.  It answers no question about
  *what* to emit.
* the target itself, for every question whose answer decides what is
  emitted -- whether there is a packed FMA, an asynchronous copy, a prefetch,
  a nontemporal access or a native atomic, how far a cross-lane exchange
  reaches, whether a rendezvous narrower than the block exists.  Those turn
  on the device and the language together, so neither part can answer them
  alone: HIP compiles for NVIDIA as well, one Intel device runs two
  lowerings, and an intrinsic that exists on the silicon is no use where the
  language has no way to name it.

The capabilities below are asked once per question and computed from `hw`
and `backend` when asked, so a test may stand a part in for a row the table
does not have.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

from tensorforge.backend.placement import VendorPolicy
from tensorforge.common.basic_types import Datatype
from tensorforge.common.vm.hw_descr import HwDecription, hw_descr_factory
from tensorforge.common.vm.lexic import (EXPLICIT_SIMD_BACKENDS, Lexic,
                                         lexic_factory)
from tensorforge.common.vm.lexic.sycl_lexic import smallest_sub_group

#: Names a backend is also known by.
_ALIASES = {'hipsycl': 'acpp', 'dpcpp': 'oneapi'}

#: The SYCL lowerings: AdaptiveCpp, DPC++ under SPMD, and DPC++ with an
#: explicit vector per work-item.
SYCL_BACKENDS = ('acpp', 'oneapi', 'esimd')

#: The sub-group sizes an SPMD kernel may require on Xe: SIMD16 and SIMD32.
XE_SUB_GROUPS = (16, 32)

#: The types `__ldcg` and `__stcg` are declared over, as the CUDA lowering
#: spells them.
#:
#: A list and not a rule, because the underlying set is a list too -- CUDA
#: declares the pair one overload at a time -- and because the answer depends
#: on the *spelling* a datatype gets there, not on the datatype.  `tf32` is
#: `uint32_t` on that target, so it takes the `unsigned int` overload and
#: belongs here; on Intel the same member is a class type and would not.
#:
#: Absent, each for its own reason.  `F128` has no overload at any
#: architecture, and a hinted access to one is the compile error this set
#: exists to prevent.  `BOOL` has none either, and `const bool*` converts to
#: no other pointer type, so it would fail the same way the day something
#: loads one.  `F16` and `BF16` are spelled `half` and `bfloat16`, which
#: nothing in `include/` declares for CUDA -- so a kernel carrying them fails
#: earlier than this, and claiming an overload for a type that has no
#: declaration would be a guess.  When that spelling arrives and resolves to
#: `__half`/`__nv_bfloat16`, `cuda_fp16.hpp` and `cuda_bf16.hpp` do declare
#: the pair, and this is the line that changes.
CUDA_CACHE_HINT_TYPES = frozenset({
    Datatype.F32,
    Datatype.F64,
    Datatype.I8,
    Datatype.I16,
    Datatype.I32,
    Datatype.I64,
    Datatype.U32,
    Datatype.TF32,
})

#: Intel parts whose LSC has the prefetch the ESIMD API lowers to.
#:
#: A list because the API's own documentation is one -- "DG2, PVC only" --
#: and there is no feature query to derive it from.  An earlier Xe part
#: compiles the call and has no instruction under it.
ESIMD_PREFETCH_ARCHS = frozenset({'pvc', 'dg2'})

#: AMD parts with `global_load_lds`, the direct global -> LDS load that an
#: asynchronous copy lowers to there.
AMD_GLOBAL_LOAD_LDS = frozenset({'gfx90a', 'gfx940', 'gfx941', 'gfx942',
                                 'gfx950'})


@dataclass(frozen=True)
class Preferences:
    """What was measured to be the better choice where every choice is
    correct, per vendor (`PREFERENCES`) and adjusted for the lowering
    (`Target.prefs`).

    Each field names what reads it, and the reader says what the choice
    trades; the rows say which way a vendor went and on what evidence.  A
    measurement of one kernel on one device overrides these
    (`generators.preferences`).
    """

    #: Threads a block holds where it stages nothing in shared memory
    #: (`RegmaxBlockPolicy.get_num_mults_per_block`).
    unstaged_block_threads: int = 256
    #: Whether a lead width above one keeps the multiplications a block holds
    #: -- the block gets smaller -- rather than its threads
    #: (`RegmaxBlockPolicy`).
    widening_keeps_mults: bool = True
    #: Where operands are read from and results kept (`backend.placement`).
    #: A vendor without a row reads everything in place, which is correct and
    #: slow.
    placement: VendorPolicy = field(default_factory=VendorPolicy)
    #: The defaults of the options of the same names (`common.options`).
    preload_globals: bool = False
    split_predicated_load: bool = False
    inline_constants: int = 64
    argument_constants: bool = False
    #: What the autotuner turns (`generators.tuning`): the matrix path,
    tune_matrix_path: bool = False
    #: staging the batch-constant operands,
    tune_preload_globals: bool = False
    #: lane counts down to the vector unit rather than to `lanes.MIN_LANES`,
    lanes_down_to_wave: bool = False
    #: and a lead width of two only where a lane holds one pair per column.
    one_pair_per_lane: bool = False
    #: How the autotuner judges registers: against the fitted allocation
    #: rather than the modeled bytes (`tuning._over_budget`),
    budget_by_fit: bool = False
    #: and by the waves the allocation leaves a SIMD
    #: (`tuning._register_blocks`).
    rank_by_register_waves: bool = False
    #: Registers per lane the target's compiler allocates as a function of
    #: the modeled bytes, fitted over builds that did not spill:
    #: `(intercept, slope)` on bytes / 4 (`tuning.register_estimate`).
    register_fit: Optional[Tuple[float, float]] = None


#: One row per vendor.
PREFERENCES = {
    'nvidia': Preferences(
        # Four warps, one per scheduler of an SM (four since Volta), so that
        # an SM holds several independent blocks rather than one large one.
        # Through the generator on sm_120: `local_flux` -6.0 %, `chain_three`,
        # `square_notrans` and `wide_cascade` within the noise (+0.1 to
        # +1.5 %).
        unstaged_block_threads=128,
        placement=VendorPolicy(preload_operands_into_registers=True,
                               keep_results_in_registers=True),
        # A 32-bit immediate fits the instruction: a chain of four 9x9
        # operators on sm_120 had 768 instructions instead of 888 by value.
        inline_constants=4096,
        argument_constants=True,
        tune_matrix_path=True,
        # ptxas sm_100a, 52 builds, residuals within 40.
        register_fit=(51.0, 1.05)),
    'amd': Preferences(
        # The smaller block is measured to lose: `local_flux` at lead width
        # two on gfx1150 took 242 ns an element at 128 threads (four mults)
        # against 153 at 256 (eight), and 256 was the fastest arrangement of
        # the kernel at either width.  The halved block NVIDIA prefers made
        # `chain_three` 3.6 % and `wide_cascade` 4.1 % slower there.
        widening_keeps_mults=False,
        placement=VendorPolicy(preload_operands_into_registers=True,
                               keep_results_in_registers=True,
                               atomic_accumulation=True,
                               broadcast_without_staging=True),
        preload_globals=True,
        # hipcc spilled every gfx942 build that needed two pairs -- 16 lanes
        # at 35 and 56 rows, 32 at 80 and 120: 512 registers and 2 to 8 KB of
        # scratch, where the model saw 200 -- and none that needed one.
        one_pair_per_lane=True,
        budget_by_fit=True,
        rank_by_register_waves=True,
        # hipcc gfx942, 29 builds, residuals within 38 -- no intercept worth
        # the name.
        register_fit=(0.0, 1.26)),
    'intel': Preferences(
        # The sub-group broadcast is cheap enough here that an operand whose
        # lane axis is not where a reader expects it can be read in place.
        placement=VendorPolicy(preload_operands_into_registers=True,
                               keep_results_in_registers=True,
                               broadcast_without_staging=True),
        split_predicated_load=True,
        tune_preload_globals=True,
        lanes_down_to_wave=True),
}


def _explicit_simd(prefs: Preferences) -> Preferences:
    """A row as the explicit-vector lowering changes it.

    The broadcast without staging needs a value whose distribution over the
    lanes is known; the SPMD lowering carries that in the index expression,
    and the explicit-vector one carries it in the type -- where a value read
    once and used across the whole vector has no distribution to give.  So
    the answer is not available there, whatever the hardware could do.

    Every work-item holds its own copy of an operator, so staging it once per
    block is 1.71x over the twenty elastic kernels on pvc -- the default
    rather than a knob, and the lane counts are the vector's width, which the
    tuner leaves alone (`tuning.simple_space`).
    """
    return replace(prefs,
                   placement=replace(prefs.placement,
                                     broadcast_without_staging=False),
                   preload_globals=True, tune_preload_globals=False,
                   lanes_down_to_wave=False)


class Target:
    """One device and one lowering for it."""

    def __init__(self, arch: str, backend: str):
        backend = _ALIASES.get(backend, backend)
        #: The lowering, as asked: `esimd` stays apart from `oneapi`, whose
        #: device it runs on.
        self.backend: str = backend
        #: Whether the lowering is an explicit vector per work-item.
        self.explicit_simd: bool = backend in EXPLICIT_SIMD_BACKENDS
        #: What the device has.
        self.hw: HwDecription = hw_descr_factory(arch, backend)
        #: How a decided statement is written.
        self.lexic: Lexic = lexic_factory(
            backend=backend, underlying_hardware=self.hw.vendor,
            sub_groups=self.pinned_sub_groups)

    def headers(self) -> List[str]:
        """The headers a translation unit with this target's kernels
        includes."""
        return ['tensorforge_aux.h'] + self.lexic.get_headers()

    @property
    def prefs(self) -> Preferences:
        """This target's row of `PREFERENCES`, with what the lowering cannot
        express removed (`_explicit_simd`)."""
        prefs = PREFERENCES.get(self.hw.vendor, Preferences())
        return _explicit_simd(prefs) if self.explicit_simd else prefs

    @property
    def sycl(self) -> bool:
        return self.backend in SYCL_BACKENDS

    @property
    def relaxed_vectors(self) -> bool:
        """Whether a wide access can be written so that it is legal at element
        alignment (`Lexic.get_fptype(relaxed=True)`).  Not under SYCL's SPMD
        lowering: `sycl::vec` brings its own alignment, so the width of an
        access there has to come from an alignment its address proves.  The
        explicit vector reads a `simd` at element alignment."""
        return not self.sycl or self.explicit_simd

    # -- arithmetic --------------------------------------------------------- #

    def packed_fma_width(self, datatype: Datatype) -> int:
        """Elements one FMA instruction covers: 2 where one instruction does
        two FMAs, so that a lead width of two halves the arithmetic instead of
        only regrouping it, and 1 elsewhere.

        NVIDIA from sm_100 to sm_11x, FP32 only (`FFMA2`, through
        `__ffma2_rn`; `cuda.h` forms the pairs on exactly these).  sm_120
        declares the intrinsic and lowers it to two FFMA, and no NVIDIA part
        has a packed FP64 FMA.  AMD as LLVM records it
        (`primitives.amd.select.packed_fma_lanes`): `v_pk_fma_f32` on CDNA2
        and later and on gfx1250/gfx1251, `v_pk_fma_f64` on gfx1251 alone, and
        none on RDNA 3 and 3.5.  A width of two elsewhere is two scalar FMAs
        and the padding the pair costs.
        """
        if self.hw.vendor == 'nvidia':
            level = self.hw.sm_level()
            return 2 if (datatype == Datatype.F32 and level is not None
                         and 100 <= level < 120) else 1
        if self.hw.vendor == 'amd' and self.hw.gfx_level() is not None:
            from tensorforge.backend.instructions.compute.primitives.amd \
                import select
            return select.packed_fma_lanes_at(datatype, self.hw.gfx_level())
        return 1

    def lead_vectors(self) -> bool:
        """Whether the lowering spells arithmetic on a lane's vector of lead
        elements, which a lead width above one needs.

        CUDA and HIP can: `VectorT`/`VectorRelaxedT` carry arithmetic, and the
        naturally-aligned and element-aligned spellings convert to each other
        -- on HIP as GNU vector types, on CUDA through the operators and
        conversion `cuda.h` gives its `VectorStruct`.

        SYCL cannot, and for two separate reasons.  `sycl::vec` has no
        element-aligned twin, so a relaxed cast has nowhere to go; and it does
        not define `operator*` between two `vec`s the way a GNU vector does,
        so the product does not compile even where the cast would.  The ESIMD
        emitter is further out still -- its whole model puts the lane axis in
        the type, so a per-lane width is a second axis it has no spelling for
        yet.
        """
        return self.backend in ('cuda', 'hip')

    # -- cross-lane --------------------------------------------------------- #

    def sync_mult(self, num_threads: int) -> bool:
        """Whether a rendezvous of exactly the `num_threads` threads of one
        multiplication exists, narrower than the whole block.

        Asked before the thread-block policy sizes a block, because the answer
        decides how many multiplications may share one.  A target that says
        False gets one multiplication per block whenever a multiplication is
        wider than a wave: the block barrier is then the multiplication's own
        barrier, and the loop around it is block-uniform, so there is a legal
        spelling.  Say True without a spelling in `Lexic.sync_mult` and the
        policy packs several multiplications into a block that cannot
        separate them.

        `num_threads` is part of the question rather than decoration on it.
        Every sub-block rendezvous counts participants, and every one of them
        counts in a unit -- threads, waves, sub-groups -- that a width not
        divisible by the wave cannot express.

        CUDA: below a wave the multiplication is a run of lanes inside one,
        and `__syncwarp` takes the mask of exactly those lanes.  Above it the
        multiplication is a whole number of waves and `barrier.sync id, count`
        meets exactly `count` threads -- but the count must be a multiple of
        the warp size, so a width that leaves a partial wave has no spelling
        and falls back to the group.  That covers the widths `MultLayout`
        interleaves as well: their group is driven in lockstep and the block
        is sized to it, so the block barrier is the group's
        (`SyncThreads.participants`).

        HIP: within a wave free, since the lanes of a wave are in lockstep
        and the rendezvous has already happened -- the reason `sync_simd` is
        None there.  Across waves it needs a barrier object of its own: GFX12
        splits `s_barrier` into signal and wait but leaves only the workgroup
        barrier visible to the shader, and the objects a shader may assign,
        1 through 16, arrive at GFX12.5.  Even there the answer is False: the
        objects would need a prologue -- `s_barrier_init` with the expected
        count, a workgroup barrier so that nobody joins before the init lands,
        and one `s_barrier_join` per wave, all before the first use -- and the
        generator emits none.

        SYCL under ESIMD: one work-item *is* the vector, so a multiplication
        of any width is held by a single work-item, executed in order, with no
        second party to wait for.  The rendezvous costs nothing and is exact.

        SYCL under SPMD has no mask.  `sycl::group_barrier` takes a group
        object, the narrowest one is the sub-group, and a multiplication
        occupying *part* of a sub-group cannot be met on its own -- the named
        barriers Xe has are reachable from ESIMD and exactly there they are
        not needed.  But the sub-group is not the hardware wave where the
        kernel states it (`pinned_sub_groups`) and states it as the
        multiplication's own width: there the sub-group *is* the
        multiplication and its barrier meets exactly the threads that have to
        meet.  Asking the wave instead would cost every 32-lane kernel on a
        16-wide PVC its block: the cap would fall to one multiplication per
        work-group -- 32 work-items where the thread budget allows 256 -- and
        `elastic-o6s:derivative` runs that way at 30.2 ns an element against
        12.7 at sixteen lanes, where the multiplication happens to equal the
        wave and no cap applies.

        The other lowerings have no sub-block rendezvous.
        """
        wave = self.hw.vec_unit_length
        if self.backend == 'cuda':
            if num_threads < wave:
                return wave % num_threads == 0
            return num_threads % wave == 0
        if self.backend == 'hip':
            if num_threads <= wave:
                return wave % num_threads == 0
            return False
        if self.sycl:
            if self.explicit_simd:
                return True
            sizes = self.pinned_sub_groups
            return (sizes is not None
                    and smallest_sub_group(sizes, num_threads) == num_threads)
        return False

    def exchange_reach(self, num_threads: int) -> int:
        """How many lanes one cross-lane exchange reaches -- a shuffle, a
        broadcast, a reduction -- for a multiplication of `num_threads`.

        The wave by default.  Under ESIMD the multiplication is one
        work-item's vector, so an exchange reaches all of it.  Under SPMD with
        the sub-group stated, the one the multiplication is put in -- 32 lanes
        for a 32-lane multiplication on a device whose vector unit is 16.
        """
        if self.sycl:
            if self.explicit_simd:
                return num_threads
            sizes = self.pinned_sub_groups
            if sizes is not None:
                size = smallest_sub_group(sizes, num_threads)
                if size is not None:
                    return size
        return self.hw.vec_unit_length

    @property
    def pinned_sub_groups(self) -> Optional[Tuple[int, ...]]:
        """The sub-group sizes a kernel states, or None where the device
        picks its own.

        DPC++ under SPMD on an Intel device states it
        (`reqd_sub_group_size`), and states the multiplication's own width
        where one of these fits: the lane search puts 32 lanes on a
        multiplication -- the ceiling is deliberately not the 16-wide vector
        unit (`lanes.deduce`) -- so at a fixed 16 a multiplication would span
        two sub-groups.  Elsewhere -- AdaptiveCpp, a plug-in -- the device
        decides.
        """
        if (self.hw.vendor == 'intel' and self.backend == 'oneapi'):
            return XE_SUB_GROUPS
        return None

    def sub_group_width(self, lanes: int) -> Optional[int]:
        """The sub-group a SYCL kernel of `lanes` lanes is laid out against,
        or None outside SYCL.

        The smallest size that holds one multiplication whole where there is
        one, and the widest otherwise -- the multiplication then spans
        several sub-groups, and its broadcast sources have to be replicated
        into each (`Temporaries._lead_axes`).
        """
        if not self.sycl:
            return None
        if not lanes:
            return XE_SUB_GROUPS[-1]
        return (smallest_sub_group(XE_SUB_GROUPS, lanes)
                or next((s for s in XE_SUB_GROUPS if s >= lanes),
                        XE_SUB_GROUPS[-1]))

    def folds_broadcast(self) -> bool:
        """Whether a broadcast of one lane's element is an operand, not a
        value.

        On Intel it is a region: `r20.3<0;1,0>` reads one element of a
        register and spreads it over the instruction's lanes, so the broadcast
        occupies no register of its own -- the order-6 derivative's simd32
        build has 8033 such regions and not one message that is not a spill.
        `sycl::group_broadcast` of a fixed lane becomes exactly that.  Not
        under an explicit vector, where a broadcast is a `simd` operation that
        produces a vector of its own, and not where the exchange is an
        instruction (`__shfl_sync` writes a register, and an AMD DPP chain
        writes one per step).

        Read by `pir.pressure` through `_record_pressure`: counted as values,
        the broadcasts would be 1944 of the 3400 bytes a lane that the order-6
        derivative is judged by, which is 57 % of a figure compared against a
        register file.
        """
        return self.sycl and not self.explicit_simd

    def lanes_share_register_file(self) -> bool:
        """Whether the lanes of a sub-group share one thread's register file.

        Intel under SPMD: one hardware thread holds the whole sub-group, so
        what has to fit the file is a lane's footprint times the sub-group
        (`tuning._over_budget`).  Under the explicit vector the work-item is
        the thread and its footprint is already the whole vector.
        """
        return self.hw.vendor == 'intel' and not self.explicit_simd

    # -- memory ------------------------------------------------------------- #

    def nontemporal(self, datatype: Datatype, length: int = 1) -> bool:
        """Whether a nontemporal access of `length` x `datatype` can be
        spelled here.

        What a caller needs to know is whether the hint *exists* for this
        type, not whether something could be written: a target without one
        emits the access plainly, which costs a cache policy and nothing else.

        CUDA: `__ldcg`/`__stcg` are an overload set and not a generic
        (`CUDA_CACHE_HINT_TYPES`).  A type outside it does not get a slower
        load, it gets `no instance of overloaded function "__ldcg" matches the
        argument list` -- at every architecture, since the overload set is a
        property of the header and not of the target.  `length > 1` is
        refused: a wide value is spelled `tensorforge::VectorT<T, N>`, a
        struct of `cuda.h`, and the overloads are declared over `floatN` --
        same size, same alignment, no conversion between them.

        HIP on AMD: `__builtin_nontemporal_load` and `_store` take any scalar
        or vector operand, so there is no type to turn away.  HIP compiles for
        NVIDIA as well, where neither builtin is declared.

        Nothing else spells one.
        """
        if self.backend == 'cuda':
            return length == 1 and datatype in CUDA_CACHE_HINT_TYPES
        if self.backend == 'hip':
            return self.hw.vendor == 'amd'
        return False

    def native_atomic(self, op, datatype: Datatype, length: int = 1) -> bool:
        """Whether an atomic update of `length` x `datatype` is one
        instruction.

        Not "can it be spelled": every target can spell it, and one that has
        no instruction gets a compare-and-swap loop -- slower than the
        read-modify-write the atomic was chosen to replace.  So the honest
        answer to give the placement policy is about the instruction
        (`backend.atomics`), and a target without one says False and is
        accumulated into normally.  Addition only: `op` is None for it.

        No under the explicit-SIMD lowering, whatever the hardware can do.
        `atomic_ref` binds one reference to one element, and under ESIMD the
        value a store carries is a `simd<T, N>` with a mask beside it -- there
        is no scalar to bind.  The instruction there is
        `esimd::atomic_update<atomic_op::fadd>`, which is a different emitter;
        until it exists, refusing is what keeps an ESIMD kernel from being
        handed an `atomic_ref<simd<float, 16>>` that does not compile.
        """
        if self.explicit_simd:
            return False
        from tensorforge.backend import atomics
        return op is None and atomics.native_add(self, datatype, length)

    def async_copy_path(self) -> bool:
        """Whether the device has an asynchronous global -> shared path:
        `cp.async` from sm_80 on, and `global_load_lds` on CDNA2 and CDNA3."""
        if self.hw.vendor == 'nvidia':
            level = self.hw.sm_level()
            return level is not None and level >= 80
        if self.hw.vendor == 'amd':
            return str(self.hw.model) in AMD_GLOBAL_LOAD_LDS
        return False

    def copy_async_sizes(self) -> Tuple[int, ...]:
        """Bytes per thread an asynchronous copy moves at once, or () where
        there is no such copy here.

        CUDA's `__pipeline_memcpy_async` takes 4, 8 and 16.  HIP on AMD
        lowers to `global_load_lds`, which takes 1, 2 and 4 on gfx90a and
        gfx94x (gfx950 adds 12 and 16); on NVIDIA HIP spells the CUDA one.
        Every copy of a body has to be one of these, or the whole body goes
        synchronous (`pir.emit`).
        """
        if not self.async_copy_path():
            return ()
        if self.backend == 'cuda':
            return (4, 8, 16)
        if self.backend == 'hip':
            return (1, 2, 4) if self.hw.vendor == 'amd' else (4, 8, 16)
        return ()

    def one_wait_counter(self) -> bool:
        """Whether loads into registers and asynchronous copies retire
        through one counter, so that a wait for either counts both: AMD's
        `vmcnt`.  NVIDIA scoreboards register loads in hardware and counts
        copies in groups of its own."""
        return self.hw.vendor == 'amd'

    def async_staging(self) -> bool:
        """Whether the staging transfers into shared memory are issued as
        asynchronous copies where they can be.

        NVIDIA only.  HIP spells one for CDNA2/3 (`copy_async_sizes`), which
        the emitter would lower; no staging transfer asks for it there.
        """
        return self.hw.vendor == 'nvidia'

    def prefetch(self) -> bool:
        """Whether a data prefetch reaches an instruction here.

        A hint and nothing else: it moves no value and releases no token, and
        the emitter drops it on a target that has none -- so a False costs a
        cache policy and not a kernel.

        CUDA: `prefetch.global.L1` and `.L2`, PTX ISA 2.0, sm_50 and up.
        Everything the table has a row for is above that, so the check states
        what the helpers in `cuda.h` are allowed to assume rather than a gate
        anything is expected to fail; the inline PTX there carries no
        architecture guard of its own, and a target below the line would
        reach `ptxas` and fail there.

        HIP: gfx12 and up, where `hasPrefetch` in LLVM's subtarget is
        `GFX12Insts`.  `__builtin_prefetch` is accepted at every AMD target
        and selects no instruction below gfx12, so a False there does not
        avert an error, it avoids carrying a statement that reaches nothing.
        HIP on NVIDIA would be an NVPTX question and not this one.

        SYCL: `multi_ptr::prefetch` is core SYCL 2020, declared for every
        global-space pointer, and an implementation with nothing behind it
        still has to accept the call -- so under SPMD the answer does not
        consult the device at all.  Under ESIMD `esimd::prefetch` is a thin
        wrapper over an LSC message that two Intel generations have
        (`ESIMD_PREFETCH_ARCHS`).

        OpenCL: `prefetch` is an OpenCL C 1.0 builtin, so every conforming
        device has it, and the specification says only that it does not
        change what the kernel computes.
        """
        if self.backend == 'cuda':
            level = self.hw.sm_level()
            return level is not None and level >= 50
        if self.backend == 'hip':
            if self.hw.vendor != 'amd':
                return False
            level = self.hw.gfx_level()
            return level is not None and level >= 0x1200
        if self.sycl:
            if self.explicit_simd:
                return str(self.hw.model) in ESIMD_PREFETCH_ARCHS
            return True
        return self.backend == 'opencl'

    def prefetch_line_bytes(self) -> int:
        """How much one prefetch statement covers, in bytes.

        A cache line where the instruction names one address: a longer run is
        that many statements apart.  `prefetch.global.L2` asks for the
        128-byte line holding the address, and the AMD parts HIP prefetches on
        have the same line.  ESIMD asks with one message for a run of up to 31
        lines -- a gather, one lane per line, and one lane spare for a run
        that does not start on a line (`prefetchHinted` in `isycl.h`).
        """
        if self.backend in ('cuda', 'hip'):
            return 128
        if self.explicit_simd:
            return 31 * 64
        return 64

    # -- launch ------------------------------------------------------------- #

    def bounds_grid(self) -> bool:
        """Whether a grid-stride launch is sized to the device or to the
        batch.

        True: the launcher asks how many blocks the device holds and the grid
        is the smaller of that and the batch, so each block loops over its
        share.  False: the grid covers the batch in one round,
        `ceil(elements / mults_per_block)` blocks, and every block runs its
        loop once.  A cooperative launch is bounded either way -- it has to be
        resident at once.

        False under SYCL.  On a Data Center GPU Max 1550 a work-group that
        loops over more than one round of the batch is slow, whatever the
        occupancy: the SeisSol elastic kernels at a batch of 262144 ran as
        slowly at 2 rounds as at 32, and twice as fast (geomean, up to 7x) at
        1, with VTune counting the same occupancy either way.  Bounded by
        `max_compute_units` they looped over one round per XVE.
        """
        return not self.sycl

    def launch_control(self) -> bool:
        """Whether `clusterlaunchcontrol` exists here: sm_100 and above.

        The instruction is PTX ISA 8.6, and the CCCL wrappers gate it on
        `__CUDA_ARCH__ >= 1000`, so a lower target does not fail to compile --
        it fails to *link*, against a stub named
        `__cuda_ptx_clusterlaunchcontrol_try_cancel_is_not_supported_before_SM_100__`.
        Which is a decent error to read and a bad one to reach from a switch,
        so the question is answered here.  Verified on an sm_120 part: the
        plain target assembles it, no `sm_120a` needed.
        """
        if self.hw.vendor != 'nvidia':
            return False
        model = str(self.hw.model)
        if not model.startswith('sm_') or not model[3:].isdigit():
            return False
        return int(model[3:]) >= 100

    def min_blocks_bound(self) -> bool:
        """Whether a kernel's launch bounds may name a minimum of resident
        blocks (`Options.min_blocks_per_sm`), which `ptxas` sizes the
        registers for."""
        return self.hw.vendor == 'nvidia'

    def grid_barrier(self) -> bool:
        """Whether a barrier across the whole grid can be spelled: not under
        SYCL, whose lexic has no `sync_grid`."""
        return not self.sycl

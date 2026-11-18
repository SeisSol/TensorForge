// === base name ===
kernel_26a3f00c01401c73

// === header ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
inline constexpr tensorforge::LaunchInfo launch_info_kernel_26a3f00c01401c73 = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_26a3f00c01401c73(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_26a3f00c01401c73(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
tensorforge::LaunchConfig launch_config_kernel_26a3f00c01401c73(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (static_cast<size_t>((numElements0 + 1) / 2) + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 4352 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_26a3f00c01401c73(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_26a3f00c01401c73(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_26a3f00c01401c73(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_26a3f00c01401c73_k;
inline void kernel_kernel_26a3f00c01401c73(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_26a3f00c01401c73_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4352 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: stack_elements=2
        // launch: 16 lanes x 16 per block = block 1x16x1, 17408 B shared, occupancy grid
        // operands:
        //   m0 16×8×2(16×8×2) {0..16}×{0..8}×{0..2} indexed(stride=136, offset=4)
        //   m1 16×16(16×16) {0..16}×{0..16} none ordered
        //   m2 16×8×2(16×8×2) {0..16}×{0..8}×{0..2} indexed_pointer(slots=3, slot=2)
        // operations:
        //   m0[i,j,l] = m1[i,k] × m2[k,j,l]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4352}],"shared_bytes":17408,"shared_elements":4352,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"indexed(stride=136, offset=4)","alias":"C","bbox":[[0,0,0],[16,8,2]],"name":"m0","ordered":false,"parts":1,"shape":[16,8,2],"stack":2,"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":true,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed_pointer(slots=3, slot=2)","alias":"B","bbox":[[0,0,0],[16,8,2]],"name":"m2","ordered":false,"parts":1,"shape":[16,8,2],"stack":2,"variant":false}],"operations":[{"add":false,"dest":{"addressing":"indexed(stride=136, offset=4)","bbox":[[0,0,0],[16,8,2]],"is_tmp":false,"name":"m0","offset":[0,0,0],"shape":[16,8,2]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"indexed_pointer(slots=3, slot=2)","bbox":[[0,0,0],[16,8,2]],"is_tmp":false,"name":"m2","offset":[0,0,0],"shape":[16,8,2]}],"permute":[[0,1],[0,1,2]],"target":[[0,-1],[-1,1,2]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < ((numElements0 + 1) / 2)) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : (flags0[2 * v3_batchId1] || (2 * v3_batchId1 + 1 < numElements0 && flags0[2 * v3_batchId1 + 1]));
          size_t v682_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v684_flagSecond = (v682_flagSecondAhead < ((numElements0 + 1) / 2)) ? v682_flagSecondAhead : v3_batchId1;
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : (flags0[2 * v684_flagSecond] || (2 * v684_flagSecond + 1 < numElements0 && flags0[2 * v684_flagSecond + 1]));
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const bool peel_batchIdStack0_0 = 2 * v3_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 0]));
          const bool peel_batchIdStack0_1 = 2 * v3_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 1]));
          const float *const __restrict__ peel_glb_m2_s0 = &m2[static_cast<size_t>(elementIndices[(peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]) * 3 + 2][0 + m2_extraOffset];
          const float *const __restrict__ peel_glb_m2_s1 = &m2[static_cast<size_t>(elementIndices[(peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]) * 3 + 2][0 + m2_extraOffset];
          if (allowed_peel) {
            // s0 = load{g>s}(glb_m2[0, 1, 2])
            tensorforge::intel_esimd::simd<float, 64> v698_ld;
            v698_ld.copy_from(peel_glb_m2_s0 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v698_ld);
            tensorforge::intel_esimd::simd<float, 64> v699_ld;
            v699_ld.copy_from(peel_glb_m2_s0 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v699_ld);
            tensorforge::intel_esimd::simd<float, 64> v700_ld;
            v700_ld.copy_from(peel_glb_m2_s1 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 128 + 4 * 0 + 0), v700_ld);
            tensorforge::intel_esimd::simd<float, 64> v701_ld;
            v701_ld.copy_from(peel_glb_m2_s1 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 128 + 4 * 0 + 64), v701_ld);
          }
          uint32_t v686_flagWord(flagWordFirst);
          uint32_t v687_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < ((numElements0 + 1) / 2); v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < ((numElements0 + 1) / 2)) ? v10_ahead1 : v9_batchId0;
            const bool wrap_batchIdStack0_0 = 2 * v12_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v12_batchId1 + 0]));
            const bool wrap_batchIdStack0_1 = 2 * v12_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v12_batchId1 + 1]));
            const float *const __restrict__ wrap_glb_m2_s0 = &m2[static_cast<size_t>(elementIndices[(wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : (wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : 2 * v12_batchId1))]) * 3 + 2][0 + m2_extraOffset];
            const float *const __restrict__ wrap_glb_m2_s1 = &m2[static_cast<size_t>(elementIndices[(wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : (wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : 2 * v12_batchId1))]) * 3 + 2][0 + m2_extraOffset];
            size_t v708_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v710_flagAhead = (v708_flagAheadAhead < ((numElements0 + 1) / 2)) ? v708_flagAheadAhead : v12_batchId1;
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : (flags0[2 * v710_flagAhead] || (2 * v710_flagAhead + 1 < numElements0 && flags0[2 * v710_flagAhead + 1]));
            const bool batchIdStack0_0 = 2 * v9_batchId0 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v9_batchId0 + 0]));
            const bool batchIdStack0_1 = 2 * v9_batchId0 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v9_batchId0 + 1]));
            const bool allowed = static_cast<bool>(v686_flagWord);
            if (allowed) {
              float *const glb_m0_s0 = &m0[static_cast<size_t>(elementIndices[(batchIdStack0_0 ? 2 * v9_batchId0 + 0 : (batchIdStack0_0 ? 2 * v9_batchId0 + 0 : batchIdStack0_1 ? 2 * v9_batchId0 + 1 : 2 * v9_batchId0))]) * 136 + 0 + 4 + m0_extraOffset];
              float *const glb_m0_s1 = &m0[static_cast<size_t>(elementIndices[(batchIdStack0_1 ? 2 * v9_batchId0 + 1 : (batchIdStack0_0 ? 2 * v9_batchId0 + 0 : batchIdStack0_1 ? 2 * v9_batchId0 + 1 : 2 * v9_batchId0))]) * 136 + 0 + 4 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 8), (0, 2)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v32_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v36_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v40_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v44_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v48_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v52_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v56_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v60_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v68_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v72_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v76_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v84_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v88_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v92_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v93_acc{};
              tensorforge::intel_esimd::simd<float, 16> v95_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v93_acc += ((static_cast<float>(v95_data[0])) * v32_data);
              v93_acc += ((static_cast<float>(v95_data[1])) * v36_data);
              v93_acc += ((static_cast<float>(v95_data[2])) * v40_data);
              v93_acc += ((static_cast<float>(v95_data[3])) * v44_data);
              v93_acc += ((static_cast<float>(v95_data[4])) * v48_data);
              v93_acc += ((static_cast<float>(v95_data[5])) * v52_data);
              v93_acc += ((static_cast<float>(v95_data[6])) * v56_data);
              v93_acc += ((static_cast<float>(v95_data[7])) * v60_data);
              v93_acc += ((static_cast<float>(v95_data[8])) * v64_data);
              v93_acc += ((static_cast<float>(v95_data[9])) * v68_data);
              v93_acc += ((static_cast<float>(v95_data[10])) * v72_data);
              v93_acc += ((static_cast<float>(v95_data[11])) * v76_data);
              v93_acc += ((static_cast<float>(v95_data[12])) * v80_data);
              v93_acc += ((static_cast<float>(v95_data[13])) * v84_data);
              v93_acc += ((static_cast<float>(v95_data[14])) * v88_data);
              v93_acc += ((static_cast<float>(v95_data[15])) * v92_data);
              ir0.template select<16, 1>(0) = v93_acc;
              tensorforge::intel_esimd::simd<float, 16> v128_acc{};
              tensorforge::intel_esimd::simd<float, 16> v131_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v128_acc += ((static_cast<float>(v131_data[0])) * v32_data);
              v128_acc += ((static_cast<float>(v131_data[1])) * v36_data);
              v128_acc += ((static_cast<float>(v131_data[2])) * v40_data);
              v128_acc += ((static_cast<float>(v131_data[3])) * v44_data);
              v128_acc += ((static_cast<float>(v131_data[4])) * v48_data);
              v128_acc += ((static_cast<float>(v131_data[5])) * v52_data);
              v128_acc += ((static_cast<float>(v131_data[6])) * v56_data);
              v128_acc += ((static_cast<float>(v131_data[7])) * v60_data);
              v128_acc += ((static_cast<float>(v131_data[8])) * v64_data);
              v128_acc += ((static_cast<float>(v131_data[9])) * v68_data);
              v128_acc += ((static_cast<float>(v131_data[10])) * v72_data);
              v128_acc += ((static_cast<float>(v131_data[11])) * v76_data);
              v128_acc += ((static_cast<float>(v131_data[12])) * v80_data);
              v128_acc += ((static_cast<float>(v131_data[13])) * v84_data);
              v128_acc += ((static_cast<float>(v131_data[14])) * v88_data);
              v128_acc += ((static_cast<float>(v131_data[15])) * v92_data);
              ir0.template select<16, 1>(16) = v128_acc;
              tensorforge::intel_esimd::simd<float, 16> v164_acc{};
              tensorforge::intel_esimd::simd<float, 16> v167_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v164_acc += ((static_cast<float>(v167_data[0])) * v32_data);
              v164_acc += ((static_cast<float>(v167_data[1])) * v36_data);
              v164_acc += ((static_cast<float>(v167_data[2])) * v40_data);
              v164_acc += ((static_cast<float>(v167_data[3])) * v44_data);
              v164_acc += ((static_cast<float>(v167_data[4])) * v48_data);
              v164_acc += ((static_cast<float>(v167_data[5])) * v52_data);
              v164_acc += ((static_cast<float>(v167_data[6])) * v56_data);
              v164_acc += ((static_cast<float>(v167_data[7])) * v60_data);
              v164_acc += ((static_cast<float>(v167_data[8])) * v64_data);
              v164_acc += ((static_cast<float>(v167_data[9])) * v68_data);
              v164_acc += ((static_cast<float>(v167_data[10])) * v72_data);
              v164_acc += ((static_cast<float>(v167_data[11])) * v76_data);
              v164_acc += ((static_cast<float>(v167_data[12])) * v80_data);
              v164_acc += ((static_cast<float>(v167_data[13])) * v84_data);
              v164_acc += ((static_cast<float>(v167_data[14])) * v88_data);
              v164_acc += ((static_cast<float>(v167_data[15])) * v92_data);
              ir0.template select<16, 1>(32) = v164_acc;
              tensorforge::intel_esimd::simd<float, 16> v200_acc{};
              tensorforge::intel_esimd::simd<float, 16> v203_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v200_acc += ((static_cast<float>(v203_data[0])) * v32_data);
              v200_acc += ((static_cast<float>(v203_data[1])) * v36_data);
              v200_acc += ((static_cast<float>(v203_data[2])) * v40_data);
              v200_acc += ((static_cast<float>(v203_data[3])) * v44_data);
              v200_acc += ((static_cast<float>(v203_data[4])) * v48_data);
              v200_acc += ((static_cast<float>(v203_data[5])) * v52_data);
              v200_acc += ((static_cast<float>(v203_data[6])) * v56_data);
              v200_acc += ((static_cast<float>(v203_data[7])) * v60_data);
              v200_acc += ((static_cast<float>(v203_data[8])) * v64_data);
              v200_acc += ((static_cast<float>(v203_data[9])) * v68_data);
              v200_acc += ((static_cast<float>(v203_data[10])) * v72_data);
              v200_acc += ((static_cast<float>(v203_data[11])) * v76_data);
              v200_acc += ((static_cast<float>(v203_data[12])) * v80_data);
              v200_acc += ((static_cast<float>(v203_data[13])) * v84_data);
              v200_acc += ((static_cast<float>(v203_data[14])) * v88_data);
              v200_acc += ((static_cast<float>(v203_data[15])) * v92_data);
              ir0.template select<16, 1>(48) = v200_acc;
              tensorforge::intel_esimd::simd<float, 16> v236_acc{};
              tensorforge::intel_esimd::simd<float, 16> v239_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v236_acc += ((static_cast<float>(v239_data[0])) * v32_data);
              v236_acc += ((static_cast<float>(v239_data[1])) * v36_data);
              v236_acc += ((static_cast<float>(v239_data[2])) * v40_data);
              v236_acc += ((static_cast<float>(v239_data[3])) * v44_data);
              v236_acc += ((static_cast<float>(v239_data[4])) * v48_data);
              v236_acc += ((static_cast<float>(v239_data[5])) * v52_data);
              v236_acc += ((static_cast<float>(v239_data[6])) * v56_data);
              v236_acc += ((static_cast<float>(v239_data[7])) * v60_data);
              v236_acc += ((static_cast<float>(v239_data[8])) * v64_data);
              v236_acc += ((static_cast<float>(v239_data[9])) * v68_data);
              v236_acc += ((static_cast<float>(v239_data[10])) * v72_data);
              v236_acc += ((static_cast<float>(v239_data[11])) * v76_data);
              v236_acc += ((static_cast<float>(v239_data[12])) * v80_data);
              v236_acc += ((static_cast<float>(v239_data[13])) * v84_data);
              v236_acc += ((static_cast<float>(v239_data[14])) * v88_data);
              v236_acc += ((static_cast<float>(v239_data[15])) * v92_data);
              ir0.template select<16, 1>(64) = v236_acc;
              tensorforge::intel_esimd::simd<float, 16> v272_acc{};
              tensorforge::intel_esimd::simd<float, 16> v275_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v272_acc += ((static_cast<float>(v275_data[0])) * v32_data);
              v272_acc += ((static_cast<float>(v275_data[1])) * v36_data);
              v272_acc += ((static_cast<float>(v275_data[2])) * v40_data);
              v272_acc += ((static_cast<float>(v275_data[3])) * v44_data);
              v272_acc += ((static_cast<float>(v275_data[4])) * v48_data);
              v272_acc += ((static_cast<float>(v275_data[5])) * v52_data);
              v272_acc += ((static_cast<float>(v275_data[6])) * v56_data);
              v272_acc += ((static_cast<float>(v275_data[7])) * v60_data);
              v272_acc += ((static_cast<float>(v275_data[8])) * v64_data);
              v272_acc += ((static_cast<float>(v275_data[9])) * v68_data);
              v272_acc += ((static_cast<float>(v275_data[10])) * v72_data);
              v272_acc += ((static_cast<float>(v275_data[11])) * v76_data);
              v272_acc += ((static_cast<float>(v275_data[12])) * v80_data);
              v272_acc += ((static_cast<float>(v275_data[13])) * v84_data);
              v272_acc += ((static_cast<float>(v275_data[14])) * v88_data);
              v272_acc += ((static_cast<float>(v275_data[15])) * v92_data);
              ir0.template select<16, 1>(80) = v272_acc;
              tensorforge::intel_esimd::simd<float, 16> v308_acc{};
              tensorforge::intel_esimd::simd<float, 16> v311_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v308_acc += ((static_cast<float>(v311_data[0])) * v32_data);
              v308_acc += ((static_cast<float>(v311_data[1])) * v36_data);
              v308_acc += ((static_cast<float>(v311_data[2])) * v40_data);
              v308_acc += ((static_cast<float>(v311_data[3])) * v44_data);
              v308_acc += ((static_cast<float>(v311_data[4])) * v48_data);
              v308_acc += ((static_cast<float>(v311_data[5])) * v52_data);
              v308_acc += ((static_cast<float>(v311_data[6])) * v56_data);
              v308_acc += ((static_cast<float>(v311_data[7])) * v60_data);
              v308_acc += ((static_cast<float>(v311_data[8])) * v64_data);
              v308_acc += ((static_cast<float>(v311_data[9])) * v68_data);
              v308_acc += ((static_cast<float>(v311_data[10])) * v72_data);
              v308_acc += ((static_cast<float>(v311_data[11])) * v76_data);
              v308_acc += ((static_cast<float>(v311_data[12])) * v80_data);
              v308_acc += ((static_cast<float>(v311_data[13])) * v84_data);
              v308_acc += ((static_cast<float>(v311_data[14])) * v88_data);
              v308_acc += ((static_cast<float>(v311_data[15])) * v92_data);
              ir0.template select<16, 1>(96) = v308_acc;
              tensorforge::intel_esimd::simd<float, 16> v344_acc{};
              tensorforge::intel_esimd::simd<float, 16> v347_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v344_acc += ((static_cast<float>(v347_data[0])) * v32_data);
              v344_acc += ((static_cast<float>(v347_data[1])) * v36_data);
              v344_acc += ((static_cast<float>(v347_data[2])) * v40_data);
              v344_acc += ((static_cast<float>(v347_data[3])) * v44_data);
              v344_acc += ((static_cast<float>(v347_data[4])) * v48_data);
              v344_acc += ((static_cast<float>(v347_data[5])) * v52_data);
              v344_acc += ((static_cast<float>(v347_data[6])) * v56_data);
              v344_acc += ((static_cast<float>(v347_data[7])) * v60_data);
              v344_acc += ((static_cast<float>(v347_data[8])) * v64_data);
              v344_acc += ((static_cast<float>(v347_data[9])) * v68_data);
              v344_acc += ((static_cast<float>(v347_data[10])) * v72_data);
              v344_acc += ((static_cast<float>(v347_data[11])) * v76_data);
              v344_acc += ((static_cast<float>(v347_data[12])) * v80_data);
              v344_acc += ((static_cast<float>(v347_data[13])) * v84_data);
              v344_acc += ((static_cast<float>(v347_data[14])) * v88_data);
              v344_acc += ((static_cast<float>(v347_data[15])) * v92_data);
              ir0.template select<16, 1>(112) = v344_acc;
              tensorforge::intel_esimd::simd<float, 16> v380_acc{};
              tensorforge::intel_esimd::simd<float, 16> v382_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v380_acc += ((static_cast<float>(v382_data[0])) * v32_data);
              v380_acc += ((static_cast<float>(v382_data[1])) * v36_data);
              v380_acc += ((static_cast<float>(v382_data[2])) * v40_data);
              v380_acc += ((static_cast<float>(v382_data[3])) * v44_data);
              v380_acc += ((static_cast<float>(v382_data[4])) * v48_data);
              v380_acc += ((static_cast<float>(v382_data[5])) * v52_data);
              v380_acc += ((static_cast<float>(v382_data[6])) * v56_data);
              v380_acc += ((static_cast<float>(v382_data[7])) * v60_data);
              v380_acc += ((static_cast<float>(v382_data[8])) * v64_data);
              v380_acc += ((static_cast<float>(v382_data[9])) * v68_data);
              v380_acc += ((static_cast<float>(v382_data[10])) * v72_data);
              v380_acc += ((static_cast<float>(v382_data[11])) * v76_data);
              v380_acc += ((static_cast<float>(v382_data[12])) * v80_data);
              v380_acc += ((static_cast<float>(v382_data[13])) * v84_data);
              v380_acc += ((static_cast<float>(v382_data[14])) * v88_data);
              v380_acc += ((static_cast<float>(v382_data[15])) * v92_data);
              ir0.template select<16, 1>(128) = v380_acc;
              tensorforge::intel_esimd::simd<float, 16> v415_acc{};
              tensorforge::intel_esimd::simd<float, 16> v417_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v415_acc += ((static_cast<float>(v417_data[0])) * v32_data);
              v415_acc += ((static_cast<float>(v417_data[1])) * v36_data);
              v415_acc += ((static_cast<float>(v417_data[2])) * v40_data);
              v415_acc += ((static_cast<float>(v417_data[3])) * v44_data);
              v415_acc += ((static_cast<float>(v417_data[4])) * v48_data);
              v415_acc += ((static_cast<float>(v417_data[5])) * v52_data);
              v415_acc += ((static_cast<float>(v417_data[6])) * v56_data);
              v415_acc += ((static_cast<float>(v417_data[7])) * v60_data);
              v415_acc += ((static_cast<float>(v417_data[8])) * v64_data);
              v415_acc += ((static_cast<float>(v417_data[9])) * v68_data);
              v415_acc += ((static_cast<float>(v417_data[10])) * v72_data);
              v415_acc += ((static_cast<float>(v417_data[11])) * v76_data);
              v415_acc += ((static_cast<float>(v417_data[12])) * v80_data);
              v415_acc += ((static_cast<float>(v417_data[13])) * v84_data);
              v415_acc += ((static_cast<float>(v417_data[14])) * v88_data);
              v415_acc += ((static_cast<float>(v417_data[15])) * v92_data);
              ir0.template select<16, 1>(144) = v415_acc;
              tensorforge::intel_esimd::simd<float, 16> v450_acc{};
              tensorforge::intel_esimd::simd<float, 16> v452_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v450_acc += ((static_cast<float>(v452_data[0])) * v32_data);
              v450_acc += ((static_cast<float>(v452_data[1])) * v36_data);
              v450_acc += ((static_cast<float>(v452_data[2])) * v40_data);
              v450_acc += ((static_cast<float>(v452_data[3])) * v44_data);
              v450_acc += ((static_cast<float>(v452_data[4])) * v48_data);
              v450_acc += ((static_cast<float>(v452_data[5])) * v52_data);
              v450_acc += ((static_cast<float>(v452_data[6])) * v56_data);
              v450_acc += ((static_cast<float>(v452_data[7])) * v60_data);
              v450_acc += ((static_cast<float>(v452_data[8])) * v64_data);
              v450_acc += ((static_cast<float>(v452_data[9])) * v68_data);
              v450_acc += ((static_cast<float>(v452_data[10])) * v72_data);
              v450_acc += ((static_cast<float>(v452_data[11])) * v76_data);
              v450_acc += ((static_cast<float>(v452_data[12])) * v80_data);
              v450_acc += ((static_cast<float>(v452_data[13])) * v84_data);
              v450_acc += ((static_cast<float>(v452_data[14])) * v88_data);
              v450_acc += ((static_cast<float>(v452_data[15])) * v92_data);
              ir0.template select<16, 1>(160) = v450_acc;
              tensorforge::intel_esimd::simd<float, 16> v485_acc{};
              tensorforge::intel_esimd::simd<float, 16> v487_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v485_acc += ((static_cast<float>(v487_data[0])) * v32_data);
              v485_acc += ((static_cast<float>(v487_data[1])) * v36_data);
              v485_acc += ((static_cast<float>(v487_data[2])) * v40_data);
              v485_acc += ((static_cast<float>(v487_data[3])) * v44_data);
              v485_acc += ((static_cast<float>(v487_data[4])) * v48_data);
              v485_acc += ((static_cast<float>(v487_data[5])) * v52_data);
              v485_acc += ((static_cast<float>(v487_data[6])) * v56_data);
              v485_acc += ((static_cast<float>(v487_data[7])) * v60_data);
              v485_acc += ((static_cast<float>(v487_data[8])) * v64_data);
              v485_acc += ((static_cast<float>(v487_data[9])) * v68_data);
              v485_acc += ((static_cast<float>(v487_data[10])) * v72_data);
              v485_acc += ((static_cast<float>(v487_data[11])) * v76_data);
              v485_acc += ((static_cast<float>(v487_data[12])) * v80_data);
              v485_acc += ((static_cast<float>(v487_data[13])) * v84_data);
              v485_acc += ((static_cast<float>(v487_data[14])) * v88_data);
              v485_acc += ((static_cast<float>(v487_data[15])) * v92_data);
              ir0.template select<16, 1>(176) = v485_acc;
              tensorforge::intel_esimd::simd<float, 16> v520_acc{};
              tensorforge::intel_esimd::simd<float, 16> v522_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v520_acc += ((static_cast<float>(v522_data[0])) * v32_data);
              v520_acc += ((static_cast<float>(v522_data[1])) * v36_data);
              v520_acc += ((static_cast<float>(v522_data[2])) * v40_data);
              v520_acc += ((static_cast<float>(v522_data[3])) * v44_data);
              v520_acc += ((static_cast<float>(v522_data[4])) * v48_data);
              v520_acc += ((static_cast<float>(v522_data[5])) * v52_data);
              v520_acc += ((static_cast<float>(v522_data[6])) * v56_data);
              v520_acc += ((static_cast<float>(v522_data[7])) * v60_data);
              v520_acc += ((static_cast<float>(v522_data[8])) * v64_data);
              v520_acc += ((static_cast<float>(v522_data[9])) * v68_data);
              v520_acc += ((static_cast<float>(v522_data[10])) * v72_data);
              v520_acc += ((static_cast<float>(v522_data[11])) * v76_data);
              v520_acc += ((static_cast<float>(v522_data[12])) * v80_data);
              v520_acc += ((static_cast<float>(v522_data[13])) * v84_data);
              v520_acc += ((static_cast<float>(v522_data[14])) * v88_data);
              v520_acc += ((static_cast<float>(v522_data[15])) * v92_data);
              ir0.template select<16, 1>(192) = v520_acc;
              tensorforge::intel_esimd::simd<float, 16> v555_acc{};
              tensorforge::intel_esimd::simd<float, 16> v557_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v555_acc += ((static_cast<float>(v557_data[0])) * v32_data);
              v555_acc += ((static_cast<float>(v557_data[1])) * v36_data);
              v555_acc += ((static_cast<float>(v557_data[2])) * v40_data);
              v555_acc += ((static_cast<float>(v557_data[3])) * v44_data);
              v555_acc += ((static_cast<float>(v557_data[4])) * v48_data);
              v555_acc += ((static_cast<float>(v557_data[5])) * v52_data);
              v555_acc += ((static_cast<float>(v557_data[6])) * v56_data);
              v555_acc += ((static_cast<float>(v557_data[7])) * v60_data);
              v555_acc += ((static_cast<float>(v557_data[8])) * v64_data);
              v555_acc += ((static_cast<float>(v557_data[9])) * v68_data);
              v555_acc += ((static_cast<float>(v557_data[10])) * v72_data);
              v555_acc += ((static_cast<float>(v557_data[11])) * v76_data);
              v555_acc += ((static_cast<float>(v557_data[12])) * v80_data);
              v555_acc += ((static_cast<float>(v557_data[13])) * v84_data);
              v555_acc += ((static_cast<float>(v557_data[14])) * v88_data);
              v555_acc += ((static_cast<float>(v557_data[15])) * v92_data);
              ir0.template select<16, 1>(208) = v555_acc;
              tensorforge::intel_esimd::simd<float, 16> v590_acc{};
              tensorforge::intel_esimd::simd<float, 16> v592_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v590_acc += ((static_cast<float>(v592_data[0])) * v32_data);
              v590_acc += ((static_cast<float>(v592_data[1])) * v36_data);
              v590_acc += ((static_cast<float>(v592_data[2])) * v40_data);
              v590_acc += ((static_cast<float>(v592_data[3])) * v44_data);
              v590_acc += ((static_cast<float>(v592_data[4])) * v48_data);
              v590_acc += ((static_cast<float>(v592_data[5])) * v52_data);
              v590_acc += ((static_cast<float>(v592_data[6])) * v56_data);
              v590_acc += ((static_cast<float>(v592_data[7])) * v60_data);
              v590_acc += ((static_cast<float>(v592_data[8])) * v64_data);
              v590_acc += ((static_cast<float>(v592_data[9])) * v68_data);
              v590_acc += ((static_cast<float>(v592_data[10])) * v72_data);
              v590_acc += ((static_cast<float>(v592_data[11])) * v76_data);
              v590_acc += ((static_cast<float>(v592_data[12])) * v80_data);
              v590_acc += ((static_cast<float>(v592_data[13])) * v84_data);
              v590_acc += ((static_cast<float>(v592_data[14])) * v88_data);
              v590_acc += ((static_cast<float>(v592_data[15])) * v92_data);
              ir0.template select<16, 1>(224) = v590_acc;
              tensorforge::intel_esimd::simd<float, 16> v625_acc{};
              tensorforge::intel_esimd::simd<float, 16> v627_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v625_acc += ((static_cast<float>(v627_data[0])) * v32_data);
              v625_acc += ((static_cast<float>(v627_data[1])) * v36_data);
              v625_acc += ((static_cast<float>(v627_data[2])) * v40_data);
              v625_acc += ((static_cast<float>(v627_data[3])) * v44_data);
              v625_acc += ((static_cast<float>(v627_data[4])) * v48_data);
              v625_acc += ((static_cast<float>(v627_data[5])) * v52_data);
              v625_acc += ((static_cast<float>(v627_data[6])) * v56_data);
              v625_acc += ((static_cast<float>(v627_data[7])) * v60_data);
              v625_acc += ((static_cast<float>(v627_data[8])) * v64_data);
              v625_acc += ((static_cast<float>(v627_data[9])) * v68_data);
              v625_acc += ((static_cast<float>(v627_data[10])) * v72_data);
              v625_acc += ((static_cast<float>(v627_data[11])) * v76_data);
              v625_acc += ((static_cast<float>(v627_data[12])) * v80_data);
              v625_acc += ((static_cast<float>(v627_data[13])) * v84_data);
              v625_acc += ((static_cast<float>(v627_data[14])) * v88_data);
              v625_acc += ((static_cast<float>(v627_data[15])) * v92_data);
              ir0.template select<16, 1>(240) = v625_acc;
              // r0 = ir0
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v669_i0 = 0; v669_i0 < 1; ++v669_i0) {
                int32_t v672_a = v669_i0 * 16;
                #pragma unroll
                for (int32_t v670_i1 = 0; v670_i1 < 8; ++v670_i1) {
                  int32_t v673_a = v670_i1 * 16;
                  int32_t v675_a = v672_a + v673_a;
                  #pragma unroll
                  for (int32_t v671_i2 = 0; v671_i2 < 2; ++v671_i2) {
                    tensorforge::intel_esimd::simd<float, 16> v677_data(ir0.template select<16, 1>((v673_a + (v671_i2 * 128))));
                    if ((v671_i2 == 0 ? batchIdStack0_0 : batchIdStack0_1)) {
                      float *const glb_m0_se0 = v671_i2 == 0 ? glb_m0_s0 : glb_m0_s1;
                      v677_data.copy_to(glb_m0_se0 + (v675_a));
                    }
                  }
                }
              }
            }
            const bool allowed_next = static_cast<bool>(v687_flagWordNext);
            if (allowed_next) {
              // s0 = load{g>s}(glb_m2[0, 1, 2])
              tensorforge::intel_esimd::simd<float, 64> v702_ld;
              v702_ld.copy_from(wrap_glb_m2_s0 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v702_ld);
              tensorforge::intel_esimd::simd<float, 64> v703_ld;
              v703_ld.copy_from(wrap_glb_m2_s0 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v703_ld);
              tensorforge::intel_esimd::simd<float, 64> v704_ld;
              v704_ld.copy_from(wrap_glb_m2_s1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 128 + 4 * 0 + 0), v704_ld);
              tensorforge::intel_esimd::simd<float, 64> v705_ld;
              v705_ld.copy_from(wrap_glb_m2_s1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 128 + 4 * 0 + 64), v705_ld);
            }
            v686_flagWord = v687_flagWordNext;
            v687_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


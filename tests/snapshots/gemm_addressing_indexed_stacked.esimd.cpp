// === base name ===
kernel_fcf1c0e72843484d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_fcf1c0e72843484d = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_fcf1c0e72843484d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_fcf1c0e72843484d(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_fcf1c0e72843484d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_fcf1c0e72843484d(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_fcf1c0e72843484d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_fcf1c0e72843484d(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_fcf1c0e72843484d_k;
inline void kernel_kernel_fcf1c0e72843484d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_fcf1c0e72843484d_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
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
          size_t v684_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v686_flagSecond = (v684_flagSecondAhead < ((numElements0 + 1) / 2)) ? v684_flagSecondAhead : v3_batchId1;
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : (flags0[2 * v686_flagSecond] || (2 * v686_flagSecond + 1 < numElements0 && flags0[2 * v686_flagSecond + 1]));
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const bool peel_batchIdStack0_0 = 2 * v3_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 0]));
          const bool peel_batchIdStack0_1 = 2 * v3_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 1]));
          const size_t peel_elementIndex_s0 = static_cast<size_t>(elementIndices[(peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]);
          const size_t peel_elementIndex_s1 = static_cast<size_t>(elementIndices[(peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]);
          const float *const __restrict__ peel_glb_m2_s0 = &m2[peel_elementIndex_s0 * 3 + 2][0 + m2_extraOffset];
          const float *const __restrict__ peel_glb_m2_s1 = &m2[peel_elementIndex_s1 * 3 + 2][0 + m2_extraOffset];
          if (allowed_peel) {
            // s0 = load{g>s}(glb_m2[0, 1, 2])
            tensorforge::intel_esimd::simd<float, 64> v704_ld;
            v704_ld.copy_from(peel_glb_m2_s0 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v704_ld);
            tensorforge::intel_esimd::simd<float, 64> v705_ld;
            v705_ld.copy_from(peel_glb_m2_s0 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v705_ld);
            tensorforge::intel_esimd::simd<float, 64> v706_ld;
            v706_ld.copy_from(peel_glb_m2_s1 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 128 + 4 * 0 + 0), v706_ld);
            tensorforge::intel_esimd::simd<float, 64> v707_ld;
            v707_ld.copy_from(peel_glb_m2_s1 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 128 + 4 * 0 + 64), v707_ld);
          }
          uint32_t v688_flagWord(flagWordFirst);
          uint32_t v689_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < ((numElements0 + 1) / 2); v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < ((numElements0 + 1) / 2)) ? v10_ahead1 : v9_batchId0;
            const bool wrap_batchIdStack0_0 = 2 * v12_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v12_batchId1 + 0]));
            const bool wrap_batchIdStack0_1 = 2 * v12_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v12_batchId1 + 1]));
            const size_t wrap_elementIndex_s0 = static_cast<size_t>(elementIndices[(wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : (wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : 2 * v12_batchId1))]);
            const size_t wrap_elementIndex_s1 = static_cast<size_t>(elementIndices[(wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : (wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : 2 * v12_batchId1))]);
            const float *const __restrict__ wrap_glb_m2_s0 = &m2[wrap_elementIndex_s0 * 3 + 2][0 + m2_extraOffset];
            const float *const __restrict__ wrap_glb_m2_s1 = &m2[wrap_elementIndex_s1 * 3 + 2][0 + m2_extraOffset];
            size_t v714_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v716_flagAhead = (v714_flagAheadAhead < ((numElements0 + 1) / 2)) ? v714_flagAheadAhead : v12_batchId1;
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : (flags0[2 * v716_flagAhead] || (2 * v716_flagAhead + 1 < numElements0 && flags0[2 * v716_flagAhead + 1]));
            const bool batchIdStack0_0 = 2 * v9_batchId0 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v9_batchId0 + 0]));
            const bool batchIdStack0_1 = 2 * v9_batchId0 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v9_batchId0 + 1]));
            const bool allowed = static_cast<bool>(v688_flagWord);
            if (allowed) {
              const size_t elementIndex_s0 = static_cast<size_t>(elementIndices[(batchIdStack0_0 ? 2 * v9_batchId0 + 0 : (batchIdStack0_0 ? 2 * v9_batchId0 + 0 : batchIdStack0_1 ? 2 * v9_batchId0 + 1 : 2 * v9_batchId0))]);
              const size_t elementIndex_s1 = static_cast<size_t>(elementIndices[(batchIdStack0_1 ? 2 * v9_batchId0 + 1 : (batchIdStack0_0 ? 2 * v9_batchId0 + 0 : batchIdStack0_1 ? 2 * v9_batchId0 + 1 : 2 * v9_batchId0))]);
              float *const glb_m0_s0 = &m0[elementIndex_s0 * 136 + 0 + 4 + m0_extraOffset];
              float *const glb_m0_s1 = &m0[elementIndex_s1 * 136 + 0 + 4 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 8), (0, 2)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v34_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v38_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v42_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v46_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v54_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v58_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v62_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v70_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v74_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v78_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v82_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v86_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v90_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v94_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v95_acc{};
              tensorforge::intel_esimd::simd<float, 16> v97_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v95_acc += ((static_cast<float>(v97_data[0])) * v34_data);
              v95_acc += ((static_cast<float>(v97_data[1])) * v38_data);
              v95_acc += ((static_cast<float>(v97_data[2])) * v42_data);
              v95_acc += ((static_cast<float>(v97_data[3])) * v46_data);
              v95_acc += ((static_cast<float>(v97_data[4])) * v50_data);
              v95_acc += ((static_cast<float>(v97_data[5])) * v54_data);
              v95_acc += ((static_cast<float>(v97_data[6])) * v58_data);
              v95_acc += ((static_cast<float>(v97_data[7])) * v62_data);
              v95_acc += ((static_cast<float>(v97_data[8])) * v66_data);
              v95_acc += ((static_cast<float>(v97_data[9])) * v70_data);
              v95_acc += ((static_cast<float>(v97_data[10])) * v74_data);
              v95_acc += ((static_cast<float>(v97_data[11])) * v78_data);
              v95_acc += ((static_cast<float>(v97_data[12])) * v82_data);
              v95_acc += ((static_cast<float>(v97_data[13])) * v86_data);
              v95_acc += ((static_cast<float>(v97_data[14])) * v90_data);
              v95_acc += ((static_cast<float>(v97_data[15])) * v94_data);
              ir0.template select<16, 1>(0) = v95_acc;
              tensorforge::intel_esimd::simd<float, 16> v130_acc{};
              tensorforge::intel_esimd::simd<float, 16> v133_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v130_acc += ((static_cast<float>(v133_data[0])) * v34_data);
              v130_acc += ((static_cast<float>(v133_data[1])) * v38_data);
              v130_acc += ((static_cast<float>(v133_data[2])) * v42_data);
              v130_acc += ((static_cast<float>(v133_data[3])) * v46_data);
              v130_acc += ((static_cast<float>(v133_data[4])) * v50_data);
              v130_acc += ((static_cast<float>(v133_data[5])) * v54_data);
              v130_acc += ((static_cast<float>(v133_data[6])) * v58_data);
              v130_acc += ((static_cast<float>(v133_data[7])) * v62_data);
              v130_acc += ((static_cast<float>(v133_data[8])) * v66_data);
              v130_acc += ((static_cast<float>(v133_data[9])) * v70_data);
              v130_acc += ((static_cast<float>(v133_data[10])) * v74_data);
              v130_acc += ((static_cast<float>(v133_data[11])) * v78_data);
              v130_acc += ((static_cast<float>(v133_data[12])) * v82_data);
              v130_acc += ((static_cast<float>(v133_data[13])) * v86_data);
              v130_acc += ((static_cast<float>(v133_data[14])) * v90_data);
              v130_acc += ((static_cast<float>(v133_data[15])) * v94_data);
              ir0.template select<16, 1>(16) = v130_acc;
              tensorforge::intel_esimd::simd<float, 16> v166_acc{};
              tensorforge::intel_esimd::simd<float, 16> v169_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v166_acc += ((static_cast<float>(v169_data[0])) * v34_data);
              v166_acc += ((static_cast<float>(v169_data[1])) * v38_data);
              v166_acc += ((static_cast<float>(v169_data[2])) * v42_data);
              v166_acc += ((static_cast<float>(v169_data[3])) * v46_data);
              v166_acc += ((static_cast<float>(v169_data[4])) * v50_data);
              v166_acc += ((static_cast<float>(v169_data[5])) * v54_data);
              v166_acc += ((static_cast<float>(v169_data[6])) * v58_data);
              v166_acc += ((static_cast<float>(v169_data[7])) * v62_data);
              v166_acc += ((static_cast<float>(v169_data[8])) * v66_data);
              v166_acc += ((static_cast<float>(v169_data[9])) * v70_data);
              v166_acc += ((static_cast<float>(v169_data[10])) * v74_data);
              v166_acc += ((static_cast<float>(v169_data[11])) * v78_data);
              v166_acc += ((static_cast<float>(v169_data[12])) * v82_data);
              v166_acc += ((static_cast<float>(v169_data[13])) * v86_data);
              v166_acc += ((static_cast<float>(v169_data[14])) * v90_data);
              v166_acc += ((static_cast<float>(v169_data[15])) * v94_data);
              ir0.template select<16, 1>(32) = v166_acc;
              tensorforge::intel_esimd::simd<float, 16> v202_acc{};
              tensorforge::intel_esimd::simd<float, 16> v205_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v202_acc += ((static_cast<float>(v205_data[0])) * v34_data);
              v202_acc += ((static_cast<float>(v205_data[1])) * v38_data);
              v202_acc += ((static_cast<float>(v205_data[2])) * v42_data);
              v202_acc += ((static_cast<float>(v205_data[3])) * v46_data);
              v202_acc += ((static_cast<float>(v205_data[4])) * v50_data);
              v202_acc += ((static_cast<float>(v205_data[5])) * v54_data);
              v202_acc += ((static_cast<float>(v205_data[6])) * v58_data);
              v202_acc += ((static_cast<float>(v205_data[7])) * v62_data);
              v202_acc += ((static_cast<float>(v205_data[8])) * v66_data);
              v202_acc += ((static_cast<float>(v205_data[9])) * v70_data);
              v202_acc += ((static_cast<float>(v205_data[10])) * v74_data);
              v202_acc += ((static_cast<float>(v205_data[11])) * v78_data);
              v202_acc += ((static_cast<float>(v205_data[12])) * v82_data);
              v202_acc += ((static_cast<float>(v205_data[13])) * v86_data);
              v202_acc += ((static_cast<float>(v205_data[14])) * v90_data);
              v202_acc += ((static_cast<float>(v205_data[15])) * v94_data);
              ir0.template select<16, 1>(48) = v202_acc;
              tensorforge::intel_esimd::simd<float, 16> v238_acc{};
              tensorforge::intel_esimd::simd<float, 16> v241_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v238_acc += ((static_cast<float>(v241_data[0])) * v34_data);
              v238_acc += ((static_cast<float>(v241_data[1])) * v38_data);
              v238_acc += ((static_cast<float>(v241_data[2])) * v42_data);
              v238_acc += ((static_cast<float>(v241_data[3])) * v46_data);
              v238_acc += ((static_cast<float>(v241_data[4])) * v50_data);
              v238_acc += ((static_cast<float>(v241_data[5])) * v54_data);
              v238_acc += ((static_cast<float>(v241_data[6])) * v58_data);
              v238_acc += ((static_cast<float>(v241_data[7])) * v62_data);
              v238_acc += ((static_cast<float>(v241_data[8])) * v66_data);
              v238_acc += ((static_cast<float>(v241_data[9])) * v70_data);
              v238_acc += ((static_cast<float>(v241_data[10])) * v74_data);
              v238_acc += ((static_cast<float>(v241_data[11])) * v78_data);
              v238_acc += ((static_cast<float>(v241_data[12])) * v82_data);
              v238_acc += ((static_cast<float>(v241_data[13])) * v86_data);
              v238_acc += ((static_cast<float>(v241_data[14])) * v90_data);
              v238_acc += ((static_cast<float>(v241_data[15])) * v94_data);
              ir0.template select<16, 1>(64) = v238_acc;
              tensorforge::intel_esimd::simd<float, 16> v274_acc{};
              tensorforge::intel_esimd::simd<float, 16> v277_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v274_acc += ((static_cast<float>(v277_data[0])) * v34_data);
              v274_acc += ((static_cast<float>(v277_data[1])) * v38_data);
              v274_acc += ((static_cast<float>(v277_data[2])) * v42_data);
              v274_acc += ((static_cast<float>(v277_data[3])) * v46_data);
              v274_acc += ((static_cast<float>(v277_data[4])) * v50_data);
              v274_acc += ((static_cast<float>(v277_data[5])) * v54_data);
              v274_acc += ((static_cast<float>(v277_data[6])) * v58_data);
              v274_acc += ((static_cast<float>(v277_data[7])) * v62_data);
              v274_acc += ((static_cast<float>(v277_data[8])) * v66_data);
              v274_acc += ((static_cast<float>(v277_data[9])) * v70_data);
              v274_acc += ((static_cast<float>(v277_data[10])) * v74_data);
              v274_acc += ((static_cast<float>(v277_data[11])) * v78_data);
              v274_acc += ((static_cast<float>(v277_data[12])) * v82_data);
              v274_acc += ((static_cast<float>(v277_data[13])) * v86_data);
              v274_acc += ((static_cast<float>(v277_data[14])) * v90_data);
              v274_acc += ((static_cast<float>(v277_data[15])) * v94_data);
              ir0.template select<16, 1>(80) = v274_acc;
              tensorforge::intel_esimd::simd<float, 16> v310_acc{};
              tensorforge::intel_esimd::simd<float, 16> v313_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v310_acc += ((static_cast<float>(v313_data[0])) * v34_data);
              v310_acc += ((static_cast<float>(v313_data[1])) * v38_data);
              v310_acc += ((static_cast<float>(v313_data[2])) * v42_data);
              v310_acc += ((static_cast<float>(v313_data[3])) * v46_data);
              v310_acc += ((static_cast<float>(v313_data[4])) * v50_data);
              v310_acc += ((static_cast<float>(v313_data[5])) * v54_data);
              v310_acc += ((static_cast<float>(v313_data[6])) * v58_data);
              v310_acc += ((static_cast<float>(v313_data[7])) * v62_data);
              v310_acc += ((static_cast<float>(v313_data[8])) * v66_data);
              v310_acc += ((static_cast<float>(v313_data[9])) * v70_data);
              v310_acc += ((static_cast<float>(v313_data[10])) * v74_data);
              v310_acc += ((static_cast<float>(v313_data[11])) * v78_data);
              v310_acc += ((static_cast<float>(v313_data[12])) * v82_data);
              v310_acc += ((static_cast<float>(v313_data[13])) * v86_data);
              v310_acc += ((static_cast<float>(v313_data[14])) * v90_data);
              v310_acc += ((static_cast<float>(v313_data[15])) * v94_data);
              ir0.template select<16, 1>(96) = v310_acc;
              tensorforge::intel_esimd::simd<float, 16> v346_acc{};
              tensorforge::intel_esimd::simd<float, 16> v349_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v346_acc += ((static_cast<float>(v349_data[0])) * v34_data);
              v346_acc += ((static_cast<float>(v349_data[1])) * v38_data);
              v346_acc += ((static_cast<float>(v349_data[2])) * v42_data);
              v346_acc += ((static_cast<float>(v349_data[3])) * v46_data);
              v346_acc += ((static_cast<float>(v349_data[4])) * v50_data);
              v346_acc += ((static_cast<float>(v349_data[5])) * v54_data);
              v346_acc += ((static_cast<float>(v349_data[6])) * v58_data);
              v346_acc += ((static_cast<float>(v349_data[7])) * v62_data);
              v346_acc += ((static_cast<float>(v349_data[8])) * v66_data);
              v346_acc += ((static_cast<float>(v349_data[9])) * v70_data);
              v346_acc += ((static_cast<float>(v349_data[10])) * v74_data);
              v346_acc += ((static_cast<float>(v349_data[11])) * v78_data);
              v346_acc += ((static_cast<float>(v349_data[12])) * v82_data);
              v346_acc += ((static_cast<float>(v349_data[13])) * v86_data);
              v346_acc += ((static_cast<float>(v349_data[14])) * v90_data);
              v346_acc += ((static_cast<float>(v349_data[15])) * v94_data);
              ir0.template select<16, 1>(112) = v346_acc;
              tensorforge::intel_esimd::simd<float, 16> v382_acc{};
              tensorforge::intel_esimd::simd<float, 16> v384_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v382_acc += ((static_cast<float>(v384_data[0])) * v34_data);
              v382_acc += ((static_cast<float>(v384_data[1])) * v38_data);
              v382_acc += ((static_cast<float>(v384_data[2])) * v42_data);
              v382_acc += ((static_cast<float>(v384_data[3])) * v46_data);
              v382_acc += ((static_cast<float>(v384_data[4])) * v50_data);
              v382_acc += ((static_cast<float>(v384_data[5])) * v54_data);
              v382_acc += ((static_cast<float>(v384_data[6])) * v58_data);
              v382_acc += ((static_cast<float>(v384_data[7])) * v62_data);
              v382_acc += ((static_cast<float>(v384_data[8])) * v66_data);
              v382_acc += ((static_cast<float>(v384_data[9])) * v70_data);
              v382_acc += ((static_cast<float>(v384_data[10])) * v74_data);
              v382_acc += ((static_cast<float>(v384_data[11])) * v78_data);
              v382_acc += ((static_cast<float>(v384_data[12])) * v82_data);
              v382_acc += ((static_cast<float>(v384_data[13])) * v86_data);
              v382_acc += ((static_cast<float>(v384_data[14])) * v90_data);
              v382_acc += ((static_cast<float>(v384_data[15])) * v94_data);
              ir0.template select<16, 1>(128) = v382_acc;
              tensorforge::intel_esimd::simd<float, 16> v417_acc{};
              tensorforge::intel_esimd::simd<float, 16> v419_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v417_acc += ((static_cast<float>(v419_data[0])) * v34_data);
              v417_acc += ((static_cast<float>(v419_data[1])) * v38_data);
              v417_acc += ((static_cast<float>(v419_data[2])) * v42_data);
              v417_acc += ((static_cast<float>(v419_data[3])) * v46_data);
              v417_acc += ((static_cast<float>(v419_data[4])) * v50_data);
              v417_acc += ((static_cast<float>(v419_data[5])) * v54_data);
              v417_acc += ((static_cast<float>(v419_data[6])) * v58_data);
              v417_acc += ((static_cast<float>(v419_data[7])) * v62_data);
              v417_acc += ((static_cast<float>(v419_data[8])) * v66_data);
              v417_acc += ((static_cast<float>(v419_data[9])) * v70_data);
              v417_acc += ((static_cast<float>(v419_data[10])) * v74_data);
              v417_acc += ((static_cast<float>(v419_data[11])) * v78_data);
              v417_acc += ((static_cast<float>(v419_data[12])) * v82_data);
              v417_acc += ((static_cast<float>(v419_data[13])) * v86_data);
              v417_acc += ((static_cast<float>(v419_data[14])) * v90_data);
              v417_acc += ((static_cast<float>(v419_data[15])) * v94_data);
              ir0.template select<16, 1>(144) = v417_acc;
              tensorforge::intel_esimd::simd<float, 16> v452_acc{};
              tensorforge::intel_esimd::simd<float, 16> v454_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v452_acc += ((static_cast<float>(v454_data[0])) * v34_data);
              v452_acc += ((static_cast<float>(v454_data[1])) * v38_data);
              v452_acc += ((static_cast<float>(v454_data[2])) * v42_data);
              v452_acc += ((static_cast<float>(v454_data[3])) * v46_data);
              v452_acc += ((static_cast<float>(v454_data[4])) * v50_data);
              v452_acc += ((static_cast<float>(v454_data[5])) * v54_data);
              v452_acc += ((static_cast<float>(v454_data[6])) * v58_data);
              v452_acc += ((static_cast<float>(v454_data[7])) * v62_data);
              v452_acc += ((static_cast<float>(v454_data[8])) * v66_data);
              v452_acc += ((static_cast<float>(v454_data[9])) * v70_data);
              v452_acc += ((static_cast<float>(v454_data[10])) * v74_data);
              v452_acc += ((static_cast<float>(v454_data[11])) * v78_data);
              v452_acc += ((static_cast<float>(v454_data[12])) * v82_data);
              v452_acc += ((static_cast<float>(v454_data[13])) * v86_data);
              v452_acc += ((static_cast<float>(v454_data[14])) * v90_data);
              v452_acc += ((static_cast<float>(v454_data[15])) * v94_data);
              ir0.template select<16, 1>(160) = v452_acc;
              tensorforge::intel_esimd::simd<float, 16> v487_acc{};
              tensorforge::intel_esimd::simd<float, 16> v489_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v487_acc += ((static_cast<float>(v489_data[0])) * v34_data);
              v487_acc += ((static_cast<float>(v489_data[1])) * v38_data);
              v487_acc += ((static_cast<float>(v489_data[2])) * v42_data);
              v487_acc += ((static_cast<float>(v489_data[3])) * v46_data);
              v487_acc += ((static_cast<float>(v489_data[4])) * v50_data);
              v487_acc += ((static_cast<float>(v489_data[5])) * v54_data);
              v487_acc += ((static_cast<float>(v489_data[6])) * v58_data);
              v487_acc += ((static_cast<float>(v489_data[7])) * v62_data);
              v487_acc += ((static_cast<float>(v489_data[8])) * v66_data);
              v487_acc += ((static_cast<float>(v489_data[9])) * v70_data);
              v487_acc += ((static_cast<float>(v489_data[10])) * v74_data);
              v487_acc += ((static_cast<float>(v489_data[11])) * v78_data);
              v487_acc += ((static_cast<float>(v489_data[12])) * v82_data);
              v487_acc += ((static_cast<float>(v489_data[13])) * v86_data);
              v487_acc += ((static_cast<float>(v489_data[14])) * v90_data);
              v487_acc += ((static_cast<float>(v489_data[15])) * v94_data);
              ir0.template select<16, 1>(176) = v487_acc;
              tensorforge::intel_esimd::simd<float, 16> v522_acc{};
              tensorforge::intel_esimd::simd<float, 16> v524_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v522_acc += ((static_cast<float>(v524_data[0])) * v34_data);
              v522_acc += ((static_cast<float>(v524_data[1])) * v38_data);
              v522_acc += ((static_cast<float>(v524_data[2])) * v42_data);
              v522_acc += ((static_cast<float>(v524_data[3])) * v46_data);
              v522_acc += ((static_cast<float>(v524_data[4])) * v50_data);
              v522_acc += ((static_cast<float>(v524_data[5])) * v54_data);
              v522_acc += ((static_cast<float>(v524_data[6])) * v58_data);
              v522_acc += ((static_cast<float>(v524_data[7])) * v62_data);
              v522_acc += ((static_cast<float>(v524_data[8])) * v66_data);
              v522_acc += ((static_cast<float>(v524_data[9])) * v70_data);
              v522_acc += ((static_cast<float>(v524_data[10])) * v74_data);
              v522_acc += ((static_cast<float>(v524_data[11])) * v78_data);
              v522_acc += ((static_cast<float>(v524_data[12])) * v82_data);
              v522_acc += ((static_cast<float>(v524_data[13])) * v86_data);
              v522_acc += ((static_cast<float>(v524_data[14])) * v90_data);
              v522_acc += ((static_cast<float>(v524_data[15])) * v94_data);
              ir0.template select<16, 1>(192) = v522_acc;
              tensorforge::intel_esimd::simd<float, 16> v557_acc{};
              tensorforge::intel_esimd::simd<float, 16> v559_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v557_acc += ((static_cast<float>(v559_data[0])) * v34_data);
              v557_acc += ((static_cast<float>(v559_data[1])) * v38_data);
              v557_acc += ((static_cast<float>(v559_data[2])) * v42_data);
              v557_acc += ((static_cast<float>(v559_data[3])) * v46_data);
              v557_acc += ((static_cast<float>(v559_data[4])) * v50_data);
              v557_acc += ((static_cast<float>(v559_data[5])) * v54_data);
              v557_acc += ((static_cast<float>(v559_data[6])) * v58_data);
              v557_acc += ((static_cast<float>(v559_data[7])) * v62_data);
              v557_acc += ((static_cast<float>(v559_data[8])) * v66_data);
              v557_acc += ((static_cast<float>(v559_data[9])) * v70_data);
              v557_acc += ((static_cast<float>(v559_data[10])) * v74_data);
              v557_acc += ((static_cast<float>(v559_data[11])) * v78_data);
              v557_acc += ((static_cast<float>(v559_data[12])) * v82_data);
              v557_acc += ((static_cast<float>(v559_data[13])) * v86_data);
              v557_acc += ((static_cast<float>(v559_data[14])) * v90_data);
              v557_acc += ((static_cast<float>(v559_data[15])) * v94_data);
              ir0.template select<16, 1>(208) = v557_acc;
              tensorforge::intel_esimd::simd<float, 16> v592_acc{};
              tensorforge::intel_esimd::simd<float, 16> v594_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v592_acc += ((static_cast<float>(v594_data[0])) * v34_data);
              v592_acc += ((static_cast<float>(v594_data[1])) * v38_data);
              v592_acc += ((static_cast<float>(v594_data[2])) * v42_data);
              v592_acc += ((static_cast<float>(v594_data[3])) * v46_data);
              v592_acc += ((static_cast<float>(v594_data[4])) * v50_data);
              v592_acc += ((static_cast<float>(v594_data[5])) * v54_data);
              v592_acc += ((static_cast<float>(v594_data[6])) * v58_data);
              v592_acc += ((static_cast<float>(v594_data[7])) * v62_data);
              v592_acc += ((static_cast<float>(v594_data[8])) * v66_data);
              v592_acc += ((static_cast<float>(v594_data[9])) * v70_data);
              v592_acc += ((static_cast<float>(v594_data[10])) * v74_data);
              v592_acc += ((static_cast<float>(v594_data[11])) * v78_data);
              v592_acc += ((static_cast<float>(v594_data[12])) * v82_data);
              v592_acc += ((static_cast<float>(v594_data[13])) * v86_data);
              v592_acc += ((static_cast<float>(v594_data[14])) * v90_data);
              v592_acc += ((static_cast<float>(v594_data[15])) * v94_data);
              ir0.template select<16, 1>(224) = v592_acc;
              tensorforge::intel_esimd::simd<float, 16> v627_acc{};
              tensorforge::intel_esimd::simd<float, 16> v629_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v627_acc += ((static_cast<float>(v629_data[0])) * v34_data);
              v627_acc += ((static_cast<float>(v629_data[1])) * v38_data);
              v627_acc += ((static_cast<float>(v629_data[2])) * v42_data);
              v627_acc += ((static_cast<float>(v629_data[3])) * v46_data);
              v627_acc += ((static_cast<float>(v629_data[4])) * v50_data);
              v627_acc += ((static_cast<float>(v629_data[5])) * v54_data);
              v627_acc += ((static_cast<float>(v629_data[6])) * v58_data);
              v627_acc += ((static_cast<float>(v629_data[7])) * v62_data);
              v627_acc += ((static_cast<float>(v629_data[8])) * v66_data);
              v627_acc += ((static_cast<float>(v629_data[9])) * v70_data);
              v627_acc += ((static_cast<float>(v629_data[10])) * v74_data);
              v627_acc += ((static_cast<float>(v629_data[11])) * v78_data);
              v627_acc += ((static_cast<float>(v629_data[12])) * v82_data);
              v627_acc += ((static_cast<float>(v629_data[13])) * v86_data);
              v627_acc += ((static_cast<float>(v629_data[14])) * v90_data);
              v627_acc += ((static_cast<float>(v629_data[15])) * v94_data);
              ir0.template select<16, 1>(240) = v627_acc;
              // r0 = ir0
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v671_i0 = 0; v671_i0 < 1; ++v671_i0) {
                int32_t v674_a = v671_i0 * 16;
                #pragma unroll
                for (int32_t v672_i1 = 0; v672_i1 < 8; ++v672_i1) {
                  int32_t v675_a = v672_i1 * 16;
                  int32_t v677_a = v674_a + v675_a;
                  #pragma unroll
                  for (int32_t v673_i2 = 0; v673_i2 < 2; ++v673_i2) {
                    tensorforge::intel_esimd::simd<float, 16> v679_data(ir0.template select<16, 1>((v675_a + (v673_i2 * 128))));
                    if ((v673_i2 == 0 ? batchIdStack0_0 : batchIdStack0_1)) {
                      float *const glb_m0_se0 = v673_i2 == 0 ? glb_m0_s0 : glb_m0_s1;
                      v679_data.copy_to(glb_m0_se0 + (v677_a));
                    }
                  }
                }
              }
            }
            const bool allowed_next = static_cast<bool>(v689_flagWordNext);
            if (allowed_next) {
              // s0 = load{g>s}(glb_m2[0, 1, 2])
              tensorforge::intel_esimd::simd<float, 64> v708_ld;
              v708_ld.copy_from(wrap_glb_m2_s0 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v708_ld);
              tensorforge::intel_esimd::simd<float, 64> v709_ld;
              v709_ld.copy_from(wrap_glb_m2_s0 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v709_ld);
              tensorforge::intel_esimd::simd<float, 64> v710_ld;
              v710_ld.copy_from(wrap_glb_m2_s1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 128 + 4 * 0 + 0), v710_ld);
              tensorforge::intel_esimd::simd<float, 64> v711_ld;
              v711_ld.copy_from(wrap_glb_m2_s1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 128 + 4 * 0 + 64), v711_ld);
            }
            v688_flagWord = v689_flagWordNext;
            v689_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


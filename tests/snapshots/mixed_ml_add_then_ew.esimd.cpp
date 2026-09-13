// === base name ===
kernel_269ce28aec87e769

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_269ce28aec87e769 = {{1, 16, 1}, 16, 16, 1, 16, 5120, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_269ce28aec87e769(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_269ce28aec87e769(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_269ce28aec87e769(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 1280 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_269ce28aec87e769(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_269ce28aec87e769(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_269ce28aec87e769(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_269ce28aec87e769(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1280 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 5120 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        //   m4 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t0[i,j] += m2[i,k] × m3[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1280}],"shared_bytes":5120,"shared_elements":1280,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (80 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const float *const __restrict__ pf_glb_m0 = &m0[v10_batchId1 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ pf_glb_m1 = &m1[v10_batchId1 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v10_batchId1 * 64 + 0 + m2_extraOffset];
            const float *const __restrict__ pf_glb_m3 = &m3[v10_batchId1 * 64 + 0 + m3_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 64 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 128> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                tensorforge::intel_esimd::simd<float, 8> v30_data;
                v30_data.copy_from(glb_m0 + ((v25_i1 * 8)));
                r0.template select<8, 1>((v25_i1 * 16)) = v30_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v33_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 128> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
                tensorforge::intel_esimd::simd<float, 8> v40_data;
                v40_data.copy_from(glb_m2 + ((v35_i1 * 8)));
                r2.template select<8, 1>((v35_i1 * 16)) = v40_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v52_acc{};
              tensorforge::intel_esimd::simd<float, 16> v56_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v52_acc += ((static_cast<float>(v56_data[0])) * v44_data);
              v52_acc += ((static_cast<float>(v56_data[1])) * v45_data);
              v52_acc += ((static_cast<float>(v56_data[2])) * v46_data);
              v52_acc += ((static_cast<float>(v56_data[3])) * v47_data);
              v52_acc += ((static_cast<float>(v56_data[4])) * v48_data);
              v52_acc += ((static_cast<float>(v56_data[5])) * v49_data);
              v52_acc += ((static_cast<float>(v56_data[6])) * v50_data);
              v52_acc += ((static_cast<float>(v56_data[7])) * v51_data);
              r1.template select<16, 1>(0) = v52_acc;
              tensorforge::intel_esimd::simd<float, 16> v73_acc{};
              tensorforge::intel_esimd::simd<float, 16> v75_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              v73_acc += ((static_cast<float>(v75_data[0])) * v44_data);
              v73_acc += ((static_cast<float>(v75_data[1])) * v45_data);
              v73_acc += ((static_cast<float>(v75_data[2])) * v46_data);
              v73_acc += ((static_cast<float>(v75_data[3])) * v47_data);
              v73_acc += ((static_cast<float>(v75_data[4])) * v48_data);
              v73_acc += ((static_cast<float>(v75_data[5])) * v49_data);
              v73_acc += ((static_cast<float>(v75_data[6])) * v50_data);
              v73_acc += ((static_cast<float>(v75_data[7])) * v51_data);
              r1.template select<16, 1>(16) = v73_acc;
              tensorforge::intel_esimd::simd<float, 16> v92_acc{};
              tensorforge::intel_esimd::simd<float, 16> v94_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v92_acc += ((static_cast<float>(v94_data[0])) * v44_data);
              v92_acc += ((static_cast<float>(v94_data[1])) * v45_data);
              v92_acc += ((static_cast<float>(v94_data[2])) * v46_data);
              v92_acc += ((static_cast<float>(v94_data[3])) * v47_data);
              v92_acc += ((static_cast<float>(v94_data[4])) * v48_data);
              v92_acc += ((static_cast<float>(v94_data[5])) * v49_data);
              v92_acc += ((static_cast<float>(v94_data[6])) * v50_data);
              v92_acc += ((static_cast<float>(v94_data[7])) * v51_data);
              r1.template select<16, 1>(32) = v92_acc;
              tensorforge::intel_esimd::simd<float, 16> v111_acc{};
              tensorforge::intel_esimd::simd<float, 16> v113_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v111_acc += ((static_cast<float>(v113_data[0])) * v44_data);
              v111_acc += ((static_cast<float>(v113_data[1])) * v45_data);
              v111_acc += ((static_cast<float>(v113_data[2])) * v46_data);
              v111_acc += ((static_cast<float>(v113_data[3])) * v47_data);
              v111_acc += ((static_cast<float>(v113_data[4])) * v48_data);
              v111_acc += ((static_cast<float>(v113_data[5])) * v49_data);
              v111_acc += ((static_cast<float>(v113_data[6])) * v50_data);
              v111_acc += ((static_cast<float>(v113_data[7])) * v51_data);
              r1.template select<16, 1>(48) = v111_acc;
              tensorforge::intel_esimd::simd<float, 16> v130_acc{};
              tensorforge::intel_esimd::simd<float, 16> v132_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v130_acc += ((static_cast<float>(v132_data[0])) * v44_data);
              v130_acc += ((static_cast<float>(v132_data[1])) * v45_data);
              v130_acc += ((static_cast<float>(v132_data[2])) * v46_data);
              v130_acc += ((static_cast<float>(v132_data[3])) * v47_data);
              v130_acc += ((static_cast<float>(v132_data[4])) * v48_data);
              v130_acc += ((static_cast<float>(v132_data[5])) * v49_data);
              v130_acc += ((static_cast<float>(v132_data[6])) * v50_data);
              v130_acc += ((static_cast<float>(v132_data[7])) * v51_data);
              r1.template select<16, 1>(64) = v130_acc;
              tensorforge::intel_esimd::simd<float, 16> v149_acc{};
              tensorforge::intel_esimd::simd<float, 16> v151_data = tensorforge::slmLoad<float, 16>(s0 + (40_i32));
              v149_acc += ((static_cast<float>(v151_data[0])) * v44_data);
              v149_acc += ((static_cast<float>(v151_data[1])) * v45_data);
              v149_acc += ((static_cast<float>(v151_data[2])) * v46_data);
              v149_acc += ((static_cast<float>(v151_data[3])) * v47_data);
              v149_acc += ((static_cast<float>(v151_data[4])) * v48_data);
              v149_acc += ((static_cast<float>(v151_data[5])) * v49_data);
              v149_acc += ((static_cast<float>(v151_data[6])) * v50_data);
              v149_acc += ((static_cast<float>(v151_data[7])) * v51_data);
              r1.template select<16, 1>(80) = v149_acc;
              tensorforge::intel_esimd::simd<float, 16> v168_acc{};
              tensorforge::intel_esimd::simd<float, 16> v170_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v168_acc += ((static_cast<float>(v170_data[0])) * v44_data);
              v168_acc += ((static_cast<float>(v170_data[1])) * v45_data);
              v168_acc += ((static_cast<float>(v170_data[2])) * v46_data);
              v168_acc += ((static_cast<float>(v170_data[3])) * v47_data);
              v168_acc += ((static_cast<float>(v170_data[4])) * v48_data);
              v168_acc += ((static_cast<float>(v170_data[5])) * v49_data);
              v168_acc += ((static_cast<float>(v170_data[6])) * v50_data);
              v168_acc += ((static_cast<float>(v170_data[7])) * v51_data);
              r1.template select<16, 1>(96) = v168_acc;
              tensorforge::intel_esimd::simd<float, 16> v187_acc{};
              tensorforge::intel_esimd::simd<float, 16> v189_data = tensorforge::slmLoad<float, 16>(s0 + (56_i32));
              v187_acc += ((static_cast<float>(v189_data[0])) * v44_data);
              v187_acc += ((static_cast<float>(v189_data[1])) * v45_data);
              v187_acc += ((static_cast<float>(v189_data[2])) * v46_data);
              v187_acc += ((static_cast<float>(v189_data[3])) * v47_data);
              v187_acc += ((static_cast<float>(v189_data[4])) * v48_data);
              v187_acc += ((static_cast<float>(v189_data[5])) * v49_data);
              v187_acc += ((static_cast<float>(v189_data[6])) * v50_data);
              v187_acc += ((static_cast<float>(v189_data[7])) * v51_data);
              r1.template select<16, 1>(112) = v187_acc;
              // s2 = load{g>s}(glb_m3[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v206_ld;
              v206_ld.copy_from(glb_m3 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 0), v206_ld);
              // wait(r2 = load{g>r}(glb_m2););
              // wait(s2 = load{g>s}(glb_m3[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r3(0.0f);
              // r3 = +(r2 * s2) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 128> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v209_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v210_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v211_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v212_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v213_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v214_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v215_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v216_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v217_acc{};
              tensorforge::intel_esimd::simd<float, 16> v221_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v217_acc += ((static_cast<float>(v221_data[0])) * v209_data);
              v217_acc += ((static_cast<float>(v221_data[1])) * v210_data);
              v217_acc += ((static_cast<float>(v221_data[2])) * v211_data);
              v217_acc += ((static_cast<float>(v221_data[3])) * v212_data);
              v217_acc += ((static_cast<float>(v221_data[4])) * v213_data);
              v217_acc += ((static_cast<float>(v221_data[5])) * v214_data);
              v217_acc += ((static_cast<float>(v221_data[6])) * v215_data);
              v217_acc += ((static_cast<float>(v221_data[7])) * v216_data);
              ir3.template select<16, 1>(0) = v217_acc;
              tensorforge::intel_esimd::simd<float, 16> v238_acc{};
              tensorforge::intel_esimd::simd<float, 16> v240_data = tensorforge::slmLoad<float, 16>(s2 + (8_i32));
              v238_acc += ((static_cast<float>(v240_data[0])) * v209_data);
              v238_acc += ((static_cast<float>(v240_data[1])) * v210_data);
              v238_acc += ((static_cast<float>(v240_data[2])) * v211_data);
              v238_acc += ((static_cast<float>(v240_data[3])) * v212_data);
              v238_acc += ((static_cast<float>(v240_data[4])) * v213_data);
              v238_acc += ((static_cast<float>(v240_data[5])) * v214_data);
              v238_acc += ((static_cast<float>(v240_data[6])) * v215_data);
              v238_acc += ((static_cast<float>(v240_data[7])) * v216_data);
              ir3.template select<16, 1>(16) = v238_acc;
              tensorforge::intel_esimd::simd<float, 16> v257_acc{};
              tensorforge::intel_esimd::simd<float, 16> v259_data = tensorforge::slmLoad<float, 16>(s2 + (16_i32));
              v257_acc += ((static_cast<float>(v259_data[0])) * v209_data);
              v257_acc += ((static_cast<float>(v259_data[1])) * v210_data);
              v257_acc += ((static_cast<float>(v259_data[2])) * v211_data);
              v257_acc += ((static_cast<float>(v259_data[3])) * v212_data);
              v257_acc += ((static_cast<float>(v259_data[4])) * v213_data);
              v257_acc += ((static_cast<float>(v259_data[5])) * v214_data);
              v257_acc += ((static_cast<float>(v259_data[6])) * v215_data);
              v257_acc += ((static_cast<float>(v259_data[7])) * v216_data);
              ir3.template select<16, 1>(32) = v257_acc;
              tensorforge::intel_esimd::simd<float, 16> v276_acc{};
              tensorforge::intel_esimd::simd<float, 16> v278_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v276_acc += ((static_cast<float>(v278_data[0])) * v209_data);
              v276_acc += ((static_cast<float>(v278_data[1])) * v210_data);
              v276_acc += ((static_cast<float>(v278_data[2])) * v211_data);
              v276_acc += ((static_cast<float>(v278_data[3])) * v212_data);
              v276_acc += ((static_cast<float>(v278_data[4])) * v213_data);
              v276_acc += ((static_cast<float>(v278_data[5])) * v214_data);
              v276_acc += ((static_cast<float>(v278_data[6])) * v215_data);
              v276_acc += ((static_cast<float>(v278_data[7])) * v216_data);
              ir3.template select<16, 1>(48) = v276_acc;
              tensorforge::intel_esimd::simd<float, 16> v295_acc{};
              tensorforge::intel_esimd::simd<float, 16> v297_data = tensorforge::slmLoad<float, 16>(s2 + (32_i32));
              v295_acc += ((static_cast<float>(v297_data[0])) * v209_data);
              v295_acc += ((static_cast<float>(v297_data[1])) * v210_data);
              v295_acc += ((static_cast<float>(v297_data[2])) * v211_data);
              v295_acc += ((static_cast<float>(v297_data[3])) * v212_data);
              v295_acc += ((static_cast<float>(v297_data[4])) * v213_data);
              v295_acc += ((static_cast<float>(v297_data[5])) * v214_data);
              v295_acc += ((static_cast<float>(v297_data[6])) * v215_data);
              v295_acc += ((static_cast<float>(v297_data[7])) * v216_data);
              ir3.template select<16, 1>(64) = v295_acc;
              tensorforge::intel_esimd::simd<float, 16> v314_acc{};
              tensorforge::intel_esimd::simd<float, 16> v316_data = tensorforge::slmLoad<float, 16>(s2 + (40_i32));
              v314_acc += ((static_cast<float>(v316_data[0])) * v209_data);
              v314_acc += ((static_cast<float>(v316_data[1])) * v210_data);
              v314_acc += ((static_cast<float>(v316_data[2])) * v211_data);
              v314_acc += ((static_cast<float>(v316_data[3])) * v212_data);
              v314_acc += ((static_cast<float>(v316_data[4])) * v213_data);
              v314_acc += ((static_cast<float>(v316_data[5])) * v214_data);
              v314_acc += ((static_cast<float>(v316_data[6])) * v215_data);
              v314_acc += ((static_cast<float>(v316_data[7])) * v216_data);
              ir3.template select<16, 1>(80) = v314_acc;
              tensorforge::intel_esimd::simd<float, 16> v333_acc{};
              tensorforge::intel_esimd::simd<float, 16> v335_data = tensorforge::slmLoad<float, 16>(s2 + (48_i32));
              v333_acc += ((static_cast<float>(v335_data[0])) * v209_data);
              v333_acc += ((static_cast<float>(v335_data[1])) * v210_data);
              v333_acc += ((static_cast<float>(v335_data[2])) * v211_data);
              v333_acc += ((static_cast<float>(v335_data[3])) * v212_data);
              v333_acc += ((static_cast<float>(v335_data[4])) * v213_data);
              v333_acc += ((static_cast<float>(v335_data[5])) * v214_data);
              v333_acc += ((static_cast<float>(v335_data[6])) * v215_data);
              v333_acc += ((static_cast<float>(v335_data[7])) * v216_data);
              ir3.template select<16, 1>(96) = v333_acc;
              tensorforge::intel_esimd::simd<float, 16> v352_acc{};
              tensorforge::intel_esimd::simd<float, 16> v354_data = tensorforge::slmLoad<float, 16>(s2 + (56_i32));
              v352_acc += ((static_cast<float>(v354_data[0])) * v209_data);
              v352_acc += ((static_cast<float>(v354_data[1])) * v210_data);
              v352_acc += ((static_cast<float>(v354_data[2])) * v211_data);
              v352_acc += ((static_cast<float>(v354_data[3])) * v212_data);
              v352_acc += ((static_cast<float>(v354_data[4])) * v213_data);
              v352_acc += ((static_cast<float>(v354_data[5])) * v214_data);
              v352_acc += ((static_cast<float>(v354_data[6])) * v215_data);
              v352_acc += ((static_cast<float>(v354_data[7])) * v216_data);
              ir3.template select<16, 1>(112) = v352_acc;
              #pragma unroll
              for (int32_t v371_n1 = 0; v371_n1 < 8; ++v371_n1) {
                int32_t v372_a = v371_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v374_data(ir3.template select<8, 1>(v372_a));
                tensorforge::intel_esimd::simd<float, 8> v375_data(r1.template select<8, 1>(v372_a));
                r3.template select<8, 1>(v372_a) = (v375_data + v374_data);
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v377_i1 = 0; v377_i1 < 8; ++v377_i1) {
                tensorforge::intel_esimd::simd<float, 8> v380_data(r3.template select<8, 1>((v377_i1 * 16)));
                tensorforge::slmStore<float, 8>(s1 + ((v377_i1 * 8)), v380_data);
              }
              // glb_m4 = abs(s1)
              #pragma unroll
              for (int32_t v385_k1 = 0; v385_k1 < 8; ++v385_k1) {
                int32_t v388_a = v385_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v390_data = tensorforge::slmLoad<float, 8>(s1 + (v388_a));
                (tensorforge::intel_esimd::abs(v390_data)).copy_to(glb_m4 + (v388_a));
              }
            }
            tensorforge::prefetchRunsL2<256, 256, 256, 256>(&pf_glb_m0[0], &pf_glb_m1[0], &pf_glb_m2[0], &pf_glb_m3[0]);
          }
        }
      }
    });
  });
}


// === base name ===
kernel_c685841ebfcb088d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c685841ebfcb088d = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c685841ebfcb088d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c685841ebfcb088d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c685841ebfcb088d(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_c685841ebfcb088d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c685841ebfcb088d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c685841ebfcb088d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c685841ebfcb088d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..12}×{0..6} = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (144);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const float *const __restrict__ pf_glb_m0 = &m0[v9_batchId1 * 144 + 0 + m0_extraOffset];
            const float *const __restrict__ pf_glb_m1 = &m1[v9_batchId1 * 144 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m3 = &m3[v9_batchId1 * 144 + 0 + m3_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v22_i1 = 0; v22_i1 < 12; ++v22_i1) {
                tensorforge::intel_esimd::simd<float, 12> v27_data;
                v27_data.copy_from(glb_m0 + ((v22_i1 * 12)));
                r0.template select<12, 1>((v22_i1 * 16)) = v27_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v30_ld;
              v30_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v30_ld);
              tensorforge::intel_esimd::simd<float, 64> v31_ld;
              v31_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v31_ld);
              tensorforge::intel_esimd::simd<float, 16> v32_ld;
              v32_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v32_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v34_i1 = 0; v34_i1 < 12; ++v34_i1) {
                tensorforge::intel_esimd::simd<float, 12> v39_data;
                v39_data.copy_from(glb_m3 + ((v34_i1 * 12)));
                r2.template select<12, 1>((v34_i1 * 16)) = v39_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 96> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v55_acc{};
              tensorforge::intel_esimd::simd<float, 16> v59_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v55_acc += ((static_cast<float>(v59_data[0])) * v43_data);
              v55_acc += ((static_cast<float>(v59_data[1])) * v44_data);
              v55_acc += ((static_cast<float>(v59_data[2])) * v45_data);
              v55_acc += ((static_cast<float>(v59_data[3])) * v46_data);
              v55_acc += ((static_cast<float>(v59_data[4])) * v47_data);
              v55_acc += ((static_cast<float>(v59_data[5])) * v48_data);
              v55_acc += ((static_cast<float>(v59_data[6])) * v49_data);
              v55_acc += ((static_cast<float>(v59_data[7])) * v50_data);
              v55_acc += ((static_cast<float>(v59_data[8])) * v51_data);
              v55_acc += ((static_cast<float>(v59_data[9])) * v52_data);
              v55_acc += ((static_cast<float>(v59_data[10])) * v53_data);
              v55_acc += ((static_cast<float>(v59_data[11])) * v54_data);
              r1.template select<16, 1>(0) = v55_acc;
              tensorforge::intel_esimd::simd<float, 16> v84_acc{};
              tensorforge::intel_esimd::simd<float, 16> v86_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v84_acc += ((static_cast<float>(v86_data[0])) * v43_data);
              v84_acc += ((static_cast<float>(v86_data[1])) * v44_data);
              v84_acc += ((static_cast<float>(v86_data[2])) * v45_data);
              v84_acc += ((static_cast<float>(v86_data[3])) * v46_data);
              v84_acc += ((static_cast<float>(v86_data[4])) * v47_data);
              v84_acc += ((static_cast<float>(v86_data[5])) * v48_data);
              v84_acc += ((static_cast<float>(v86_data[6])) * v49_data);
              v84_acc += ((static_cast<float>(v86_data[7])) * v50_data);
              v84_acc += ((static_cast<float>(v86_data[8])) * v51_data);
              v84_acc += ((static_cast<float>(v86_data[9])) * v52_data);
              v84_acc += ((static_cast<float>(v86_data[10])) * v53_data);
              v84_acc += ((static_cast<float>(v86_data[11])) * v54_data);
              r1.template select<16, 1>(16) = v84_acc;
              tensorforge::intel_esimd::simd<float, 16> v111_acc{};
              tensorforge::intel_esimd::simd<float, 16> v113_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v111_acc += ((static_cast<float>(v113_data[0])) * v43_data);
              v111_acc += ((static_cast<float>(v113_data[1])) * v44_data);
              v111_acc += ((static_cast<float>(v113_data[2])) * v45_data);
              v111_acc += ((static_cast<float>(v113_data[3])) * v46_data);
              v111_acc += ((static_cast<float>(v113_data[4])) * v47_data);
              v111_acc += ((static_cast<float>(v113_data[5])) * v48_data);
              v111_acc += ((static_cast<float>(v113_data[6])) * v49_data);
              v111_acc += ((static_cast<float>(v113_data[7])) * v50_data);
              v111_acc += ((static_cast<float>(v113_data[8])) * v51_data);
              v111_acc += ((static_cast<float>(v113_data[9])) * v52_data);
              v111_acc += ((static_cast<float>(v113_data[10])) * v53_data);
              v111_acc += ((static_cast<float>(v113_data[11])) * v54_data);
              r1.template select<16, 1>(32) = v111_acc;
              tensorforge::intel_esimd::simd<float, 16> v138_acc{};
              tensorforge::intel_esimd::simd<float, 16> v140_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v138_acc += ((static_cast<float>(v140_data[0])) * v43_data);
              v138_acc += ((static_cast<float>(v140_data[1])) * v44_data);
              v138_acc += ((static_cast<float>(v140_data[2])) * v45_data);
              v138_acc += ((static_cast<float>(v140_data[3])) * v46_data);
              v138_acc += ((static_cast<float>(v140_data[4])) * v47_data);
              v138_acc += ((static_cast<float>(v140_data[5])) * v48_data);
              v138_acc += ((static_cast<float>(v140_data[6])) * v49_data);
              v138_acc += ((static_cast<float>(v140_data[7])) * v50_data);
              v138_acc += ((static_cast<float>(v140_data[8])) * v51_data);
              v138_acc += ((static_cast<float>(v140_data[9])) * v52_data);
              v138_acc += ((static_cast<float>(v140_data[10])) * v53_data);
              v138_acc += ((static_cast<float>(v140_data[11])) * v54_data);
              r1.template select<16, 1>(48) = v138_acc;
              tensorforge::intel_esimd::simd<float, 16> v165_acc{};
              tensorforge::intel_esimd::simd<float, 16> v167_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v165_acc += ((static_cast<float>(v167_data[0])) * v43_data);
              v165_acc += ((static_cast<float>(v167_data[1])) * v44_data);
              v165_acc += ((static_cast<float>(v167_data[2])) * v45_data);
              v165_acc += ((static_cast<float>(v167_data[3])) * v46_data);
              v165_acc += ((static_cast<float>(v167_data[4])) * v47_data);
              v165_acc += ((static_cast<float>(v167_data[5])) * v48_data);
              v165_acc += ((static_cast<float>(v167_data[6])) * v49_data);
              v165_acc += ((static_cast<float>(v167_data[7])) * v50_data);
              v165_acc += ((static_cast<float>(v167_data[8])) * v51_data);
              v165_acc += ((static_cast<float>(v167_data[9])) * v52_data);
              v165_acc += ((static_cast<float>(v167_data[10])) * v53_data);
              v165_acc += ((static_cast<float>(v167_data[11])) * v54_data);
              r1.template select<16, 1>(64) = v165_acc;
              tensorforge::intel_esimd::simd<float, 16> v192_acc{};
              tensorforge::intel_esimd::simd<float, 16> v194_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v192_acc += ((static_cast<float>(v194_data[0])) * v43_data);
              v192_acc += ((static_cast<float>(v194_data[1])) * v44_data);
              v192_acc += ((static_cast<float>(v194_data[2])) * v45_data);
              v192_acc += ((static_cast<float>(v194_data[3])) * v46_data);
              v192_acc += ((static_cast<float>(v194_data[4])) * v47_data);
              v192_acc += ((static_cast<float>(v194_data[5])) * v48_data);
              v192_acc += ((static_cast<float>(v194_data[6])) * v49_data);
              v192_acc += ((static_cast<float>(v194_data[7])) * v50_data);
              v192_acc += ((static_cast<float>(v194_data[8])) * v51_data);
              v192_acc += ((static_cast<float>(v194_data[9])) * v52_data);
              v192_acc += ((static_cast<float>(v194_data[10])) * v53_data);
              v192_acc += ((static_cast<float>(v194_data[11])) * v54_data);
              r1.template select<16, 1>(80) = v192_acc;
              // s1 = store{r>s, clear}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v219_z1 = 6; v219_z1 < 12; ++v219_z1) {
                s1[(v219_z1 * 12)] = 0.0f;
              }
              #pragma unroll
              for (int32_t v225_i1 = 0; v225_i1 < 6; ++v225_i1) {
                tensorforge::intel_esimd::simd<float, 12> v228_data(r1.template select<12, 1>((v225_i1 * 16)));
                tensorforge::slmStore<float, 12>(s1 + ((v225_i1 * 12)), v228_data);
              }
              // wait(r2 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v235_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v236_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v237_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v238_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v239_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v240_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v241_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v242_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v243_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v244_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v245_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v246_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v247_acc{};
              tensorforge::intel_esimd::simd<float, 16> v251_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v247_acc += ((static_cast<float>(v251_data[0])) * v235_data);
              v247_acc += ((static_cast<float>(v251_data[1])) * v236_data);
              v247_acc += ((static_cast<float>(v251_data[2])) * v237_data);
              v247_acc += ((static_cast<float>(v251_data[3])) * v238_data);
              v247_acc += ((static_cast<float>(v251_data[4])) * v239_data);
              v247_acc += ((static_cast<float>(v251_data[5])) * v240_data);
              v247_acc += ((static_cast<float>(v251_data[6])) * v241_data);
              v247_acc += ((static_cast<float>(v251_data[7])) * v242_data);
              v247_acc += ((static_cast<float>(v251_data[8])) * v243_data);
              v247_acc += ((static_cast<float>(v251_data[9])) * v244_data);
              v247_acc += ((static_cast<float>(v251_data[10])) * v245_data);
              v247_acc += ((static_cast<float>(v251_data[11])) * v246_data);
              ir3.template select<16, 1>(0) = v247_acc;
              tensorforge::intel_esimd::simd<float, 16> v276_acc{};
              tensorforge::intel_esimd::simd<float, 16> v278_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v276_acc += ((static_cast<float>(v278_data[0])) * v235_data);
              v276_acc += ((static_cast<float>(v278_data[1])) * v236_data);
              v276_acc += ((static_cast<float>(v278_data[2])) * v237_data);
              v276_acc += ((static_cast<float>(v278_data[3])) * v238_data);
              v276_acc += ((static_cast<float>(v278_data[4])) * v239_data);
              v276_acc += ((static_cast<float>(v278_data[5])) * v240_data);
              v276_acc += ((static_cast<float>(v278_data[6])) * v241_data);
              v276_acc += ((static_cast<float>(v278_data[7])) * v242_data);
              v276_acc += ((static_cast<float>(v278_data[8])) * v243_data);
              v276_acc += ((static_cast<float>(v278_data[9])) * v244_data);
              v276_acc += ((static_cast<float>(v278_data[10])) * v245_data);
              v276_acc += ((static_cast<float>(v278_data[11])) * v246_data);
              ir3.template select<16, 1>(16) = v276_acc;
              tensorforge::intel_esimd::simd<float, 16> v303_acc{};
              tensorforge::intel_esimd::simd<float, 16> v305_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v303_acc += ((static_cast<float>(v305_data[0])) * v235_data);
              v303_acc += ((static_cast<float>(v305_data[1])) * v236_data);
              v303_acc += ((static_cast<float>(v305_data[2])) * v237_data);
              v303_acc += ((static_cast<float>(v305_data[3])) * v238_data);
              v303_acc += ((static_cast<float>(v305_data[4])) * v239_data);
              v303_acc += ((static_cast<float>(v305_data[5])) * v240_data);
              v303_acc += ((static_cast<float>(v305_data[6])) * v241_data);
              v303_acc += ((static_cast<float>(v305_data[7])) * v242_data);
              v303_acc += ((static_cast<float>(v305_data[8])) * v243_data);
              v303_acc += ((static_cast<float>(v305_data[9])) * v244_data);
              v303_acc += ((static_cast<float>(v305_data[10])) * v245_data);
              v303_acc += ((static_cast<float>(v305_data[11])) * v246_data);
              ir3.template select<16, 1>(32) = v303_acc;
              tensorforge::intel_esimd::simd<float, 16> v330_acc{};
              tensorforge::intel_esimd::simd<float, 16> v332_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v330_acc += ((static_cast<float>(v332_data[0])) * v235_data);
              v330_acc += ((static_cast<float>(v332_data[1])) * v236_data);
              v330_acc += ((static_cast<float>(v332_data[2])) * v237_data);
              v330_acc += ((static_cast<float>(v332_data[3])) * v238_data);
              v330_acc += ((static_cast<float>(v332_data[4])) * v239_data);
              v330_acc += ((static_cast<float>(v332_data[5])) * v240_data);
              v330_acc += ((static_cast<float>(v332_data[6])) * v241_data);
              v330_acc += ((static_cast<float>(v332_data[7])) * v242_data);
              v330_acc += ((static_cast<float>(v332_data[8])) * v243_data);
              v330_acc += ((static_cast<float>(v332_data[9])) * v244_data);
              v330_acc += ((static_cast<float>(v332_data[10])) * v245_data);
              v330_acc += ((static_cast<float>(v332_data[11])) * v246_data);
              ir3.template select<16, 1>(48) = v330_acc;
              tensorforge::intel_esimd::simd<float, 16> v357_acc{};
              tensorforge::intel_esimd::simd<float, 16> v359_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v357_acc += ((static_cast<float>(v359_data[0])) * v235_data);
              v357_acc += ((static_cast<float>(v359_data[1])) * v236_data);
              v357_acc += ((static_cast<float>(v359_data[2])) * v237_data);
              v357_acc += ((static_cast<float>(v359_data[3])) * v238_data);
              v357_acc += ((static_cast<float>(v359_data[4])) * v239_data);
              v357_acc += ((static_cast<float>(v359_data[5])) * v240_data);
              v357_acc += ((static_cast<float>(v359_data[6])) * v241_data);
              v357_acc += ((static_cast<float>(v359_data[7])) * v242_data);
              v357_acc += ((static_cast<float>(v359_data[8])) * v243_data);
              v357_acc += ((static_cast<float>(v359_data[9])) * v244_data);
              v357_acc += ((static_cast<float>(v359_data[10])) * v245_data);
              v357_acc += ((static_cast<float>(v359_data[11])) * v246_data);
              ir3.template select<16, 1>(64) = v357_acc;
              tensorforge::intel_esimd::simd<float, 16> v384_acc{};
              tensorforge::intel_esimd::simd<float, 16> v386_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v384_acc += ((static_cast<float>(v386_data[0])) * v235_data);
              v384_acc += ((static_cast<float>(v386_data[1])) * v236_data);
              v384_acc += ((static_cast<float>(v386_data[2])) * v237_data);
              v384_acc += ((static_cast<float>(v386_data[3])) * v238_data);
              v384_acc += ((static_cast<float>(v386_data[4])) * v239_data);
              v384_acc += ((static_cast<float>(v386_data[5])) * v240_data);
              v384_acc += ((static_cast<float>(v386_data[6])) * v241_data);
              v384_acc += ((static_cast<float>(v386_data[7])) * v242_data);
              v384_acc += ((static_cast<float>(v386_data[8])) * v243_data);
              v384_acc += ((static_cast<float>(v386_data[9])) * v244_data);
              v384_acc += ((static_cast<float>(v386_data[10])) * v245_data);
              v384_acc += ((static_cast<float>(v386_data[11])) * v246_data);
              ir3.template select<16, 1>(80) = v384_acc;
              tensorforge::intel_esimd::simd<float, 16> v411_acc{};
              tensorforge::intel_esimd::simd<float, 16> v413_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v411_acc += ((static_cast<float>(v413_data[0])) * v235_data);
              v411_acc += ((static_cast<float>(v413_data[1])) * v236_data);
              v411_acc += ((static_cast<float>(v413_data[2])) * v237_data);
              v411_acc += ((static_cast<float>(v413_data[3])) * v238_data);
              v411_acc += ((static_cast<float>(v413_data[4])) * v239_data);
              v411_acc += ((static_cast<float>(v413_data[5])) * v240_data);
              v411_acc += ((static_cast<float>(v413_data[6])) * v241_data);
              v411_acc += ((static_cast<float>(v413_data[7])) * v242_data);
              v411_acc += ((static_cast<float>(v413_data[8])) * v243_data);
              v411_acc += ((static_cast<float>(v413_data[9])) * v244_data);
              v411_acc += ((static_cast<float>(v413_data[10])) * v245_data);
              v411_acc += ((static_cast<float>(v413_data[11])) * v246_data);
              ir3.template select<16, 1>(96) = v411_acc;
              tensorforge::intel_esimd::simd<float, 16> v438_acc{};
              tensorforge::intel_esimd::simd<float, 16> v440_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v438_acc += ((static_cast<float>(v440_data[0])) * v235_data);
              v438_acc += ((static_cast<float>(v440_data[1])) * v236_data);
              v438_acc += ((static_cast<float>(v440_data[2])) * v237_data);
              v438_acc += ((static_cast<float>(v440_data[3])) * v238_data);
              v438_acc += ((static_cast<float>(v440_data[4])) * v239_data);
              v438_acc += ((static_cast<float>(v440_data[5])) * v240_data);
              v438_acc += ((static_cast<float>(v440_data[6])) * v241_data);
              v438_acc += ((static_cast<float>(v440_data[7])) * v242_data);
              v438_acc += ((static_cast<float>(v440_data[8])) * v243_data);
              v438_acc += ((static_cast<float>(v440_data[9])) * v244_data);
              v438_acc += ((static_cast<float>(v440_data[10])) * v245_data);
              v438_acc += ((static_cast<float>(v440_data[11])) * v246_data);
              ir3.template select<16, 1>(112) = v438_acc;
              tensorforge::intel_esimd::simd<float, 16> v465_acc{};
              tensorforge::intel_esimd::simd<float, 16> v467_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v465_acc += ((static_cast<float>(v467_data[0])) * v235_data);
              v465_acc += ((static_cast<float>(v467_data[1])) * v236_data);
              v465_acc += ((static_cast<float>(v467_data[2])) * v237_data);
              v465_acc += ((static_cast<float>(v467_data[3])) * v238_data);
              v465_acc += ((static_cast<float>(v467_data[4])) * v239_data);
              v465_acc += ((static_cast<float>(v467_data[5])) * v240_data);
              v465_acc += ((static_cast<float>(v467_data[6])) * v241_data);
              v465_acc += ((static_cast<float>(v467_data[7])) * v242_data);
              v465_acc += ((static_cast<float>(v467_data[8])) * v243_data);
              v465_acc += ((static_cast<float>(v467_data[9])) * v244_data);
              v465_acc += ((static_cast<float>(v467_data[10])) * v245_data);
              v465_acc += ((static_cast<float>(v467_data[11])) * v246_data);
              ir3.template select<16, 1>(128) = v465_acc;
              tensorforge::intel_esimd::simd<float, 16> v492_acc{};
              tensorforge::intel_esimd::simd<float, 16> v494_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v492_acc += ((static_cast<float>(v494_data[0])) * v235_data);
              v492_acc += ((static_cast<float>(v494_data[1])) * v236_data);
              v492_acc += ((static_cast<float>(v494_data[2])) * v237_data);
              v492_acc += ((static_cast<float>(v494_data[3])) * v238_data);
              v492_acc += ((static_cast<float>(v494_data[4])) * v239_data);
              v492_acc += ((static_cast<float>(v494_data[5])) * v240_data);
              v492_acc += ((static_cast<float>(v494_data[6])) * v241_data);
              v492_acc += ((static_cast<float>(v494_data[7])) * v242_data);
              v492_acc += ((static_cast<float>(v494_data[8])) * v243_data);
              v492_acc += ((static_cast<float>(v494_data[9])) * v244_data);
              v492_acc += ((static_cast<float>(v494_data[10])) * v245_data);
              v492_acc += ((static_cast<float>(v494_data[11])) * v246_data);
              ir3.template select<16, 1>(144) = v492_acc;
              tensorforge::intel_esimd::simd<float, 16> v519_acc{};
              tensorforge::intel_esimd::simd<float, 16> v521_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v519_acc += ((static_cast<float>(v521_data[0])) * v235_data);
              v519_acc += ((static_cast<float>(v521_data[1])) * v236_data);
              v519_acc += ((static_cast<float>(v521_data[2])) * v237_data);
              v519_acc += ((static_cast<float>(v521_data[3])) * v238_data);
              v519_acc += ((static_cast<float>(v521_data[4])) * v239_data);
              v519_acc += ((static_cast<float>(v521_data[5])) * v240_data);
              v519_acc += ((static_cast<float>(v521_data[6])) * v241_data);
              v519_acc += ((static_cast<float>(v521_data[7])) * v242_data);
              v519_acc += ((static_cast<float>(v521_data[8])) * v243_data);
              v519_acc += ((static_cast<float>(v521_data[9])) * v244_data);
              v519_acc += ((static_cast<float>(v521_data[10])) * v245_data);
              v519_acc += ((static_cast<float>(v521_data[11])) * v246_data);
              ir3.template select<16, 1>(160) = v519_acc;
              tensorforge::intel_esimd::simd<float, 16> v546_acc{};
              tensorforge::intel_esimd::simd<float, 16> v548_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v546_acc += ((static_cast<float>(v548_data[0])) * v235_data);
              v546_acc += ((static_cast<float>(v548_data[1])) * v236_data);
              v546_acc += ((static_cast<float>(v548_data[2])) * v237_data);
              v546_acc += ((static_cast<float>(v548_data[3])) * v238_data);
              v546_acc += ((static_cast<float>(v548_data[4])) * v239_data);
              v546_acc += ((static_cast<float>(v548_data[5])) * v240_data);
              v546_acc += ((static_cast<float>(v548_data[6])) * v241_data);
              v546_acc += ((static_cast<float>(v548_data[7])) * v242_data);
              v546_acc += ((static_cast<float>(v548_data[8])) * v243_data);
              v546_acc += ((static_cast<float>(v548_data[9])) * v244_data);
              v546_acc += ((static_cast<float>(v548_data[10])) * v245_data);
              v546_acc += ((static_cast<float>(v548_data[11])) * v246_data);
              ir3.template select<16, 1>(176) = v546_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v573_n1 = 0; v573_n1 < 12; ++v573_n1) {
                int32_t v574_a = v573_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v576_data(ir3.template select<12, 1>(v574_a));
                r3.template select<12, 1>(v574_a) = v576_data;
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v577_i1 = 0; v577_i1 < 12; ++v577_i1) {
                tensorforge::intel_esimd::simd<float, 12> v580_data(r3.template select<12, 1>((v577_i1 * 16)));
                v580_data.copy_to(glb_m2 + ((v577_i1 * 12)));
              }
            }
            tensorforge::prefetchRunsL2<576, 576, 576>(&pf_glb_m0[0], &pf_glb_m1[0], &pf_glb_m3[0]);
          }
        }
      }
    });
  });
}


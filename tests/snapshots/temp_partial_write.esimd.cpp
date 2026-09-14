// === base name ===
kernel_f9dbd654680499dd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f9dbd654680499dd = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f9dbd654680499dd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f9dbd654680499dd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f9dbd654680499dd(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_f9dbd654680499dd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f9dbd654680499dd(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f9dbd654680499dd(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f9dbd654680499dd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 12; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 12> v24_data;
                v24_data.copy_from(glb_m0 + ((v19_i1 * 12)));
                r0.template select<12, 1>((v19_i1 * 16)) = v24_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v27_ld;
              v27_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v27_ld);
              tensorforge::intel_esimd::simd<float, 64> v28_ld;
              v28_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v28_ld);
              tensorforge::intel_esimd::simd<float, 16> v29_ld;
              v29_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v29_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v31_i1 = 0; v31_i1 < 12; ++v31_i1) {
                tensorforge::intel_esimd::simd<float, 12> v36_data;
                v36_data.copy_from(glb_m3 + ((v31_i1 * 12)));
                r2.template select<12, 1>((v31_i1 * 16)) = v36_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 96> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v52_acc{};
              tensorforge::intel_esimd::simd<float, 16> v56_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v52_acc += ((static_cast<float>(v56_data[0])) * v40_data);
              v52_acc += ((static_cast<float>(v56_data[1])) * v41_data);
              v52_acc += ((static_cast<float>(v56_data[2])) * v42_data);
              v52_acc += ((static_cast<float>(v56_data[3])) * v43_data);
              v52_acc += ((static_cast<float>(v56_data[4])) * v44_data);
              v52_acc += ((static_cast<float>(v56_data[5])) * v45_data);
              v52_acc += ((static_cast<float>(v56_data[6])) * v46_data);
              v52_acc += ((static_cast<float>(v56_data[7])) * v47_data);
              v52_acc += ((static_cast<float>(v56_data[8])) * v48_data);
              v52_acc += ((static_cast<float>(v56_data[9])) * v49_data);
              v52_acc += ((static_cast<float>(v56_data[10])) * v50_data);
              v52_acc += ((static_cast<float>(v56_data[11])) * v51_data);
              r1.template select<16, 1>(0) = v52_acc;
              tensorforge::intel_esimd::simd<float, 16> v81_acc{};
              tensorforge::intel_esimd::simd<float, 16> v83_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v81_acc += ((static_cast<float>(v83_data[0])) * v40_data);
              v81_acc += ((static_cast<float>(v83_data[1])) * v41_data);
              v81_acc += ((static_cast<float>(v83_data[2])) * v42_data);
              v81_acc += ((static_cast<float>(v83_data[3])) * v43_data);
              v81_acc += ((static_cast<float>(v83_data[4])) * v44_data);
              v81_acc += ((static_cast<float>(v83_data[5])) * v45_data);
              v81_acc += ((static_cast<float>(v83_data[6])) * v46_data);
              v81_acc += ((static_cast<float>(v83_data[7])) * v47_data);
              v81_acc += ((static_cast<float>(v83_data[8])) * v48_data);
              v81_acc += ((static_cast<float>(v83_data[9])) * v49_data);
              v81_acc += ((static_cast<float>(v83_data[10])) * v50_data);
              v81_acc += ((static_cast<float>(v83_data[11])) * v51_data);
              r1.template select<16, 1>(16) = v81_acc;
              tensorforge::intel_esimd::simd<float, 16> v108_acc{};
              tensorforge::intel_esimd::simd<float, 16> v110_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v108_acc += ((static_cast<float>(v110_data[0])) * v40_data);
              v108_acc += ((static_cast<float>(v110_data[1])) * v41_data);
              v108_acc += ((static_cast<float>(v110_data[2])) * v42_data);
              v108_acc += ((static_cast<float>(v110_data[3])) * v43_data);
              v108_acc += ((static_cast<float>(v110_data[4])) * v44_data);
              v108_acc += ((static_cast<float>(v110_data[5])) * v45_data);
              v108_acc += ((static_cast<float>(v110_data[6])) * v46_data);
              v108_acc += ((static_cast<float>(v110_data[7])) * v47_data);
              v108_acc += ((static_cast<float>(v110_data[8])) * v48_data);
              v108_acc += ((static_cast<float>(v110_data[9])) * v49_data);
              v108_acc += ((static_cast<float>(v110_data[10])) * v50_data);
              v108_acc += ((static_cast<float>(v110_data[11])) * v51_data);
              r1.template select<16, 1>(32) = v108_acc;
              tensorforge::intel_esimd::simd<float, 16> v135_acc{};
              tensorforge::intel_esimd::simd<float, 16> v137_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v135_acc += ((static_cast<float>(v137_data[0])) * v40_data);
              v135_acc += ((static_cast<float>(v137_data[1])) * v41_data);
              v135_acc += ((static_cast<float>(v137_data[2])) * v42_data);
              v135_acc += ((static_cast<float>(v137_data[3])) * v43_data);
              v135_acc += ((static_cast<float>(v137_data[4])) * v44_data);
              v135_acc += ((static_cast<float>(v137_data[5])) * v45_data);
              v135_acc += ((static_cast<float>(v137_data[6])) * v46_data);
              v135_acc += ((static_cast<float>(v137_data[7])) * v47_data);
              v135_acc += ((static_cast<float>(v137_data[8])) * v48_data);
              v135_acc += ((static_cast<float>(v137_data[9])) * v49_data);
              v135_acc += ((static_cast<float>(v137_data[10])) * v50_data);
              v135_acc += ((static_cast<float>(v137_data[11])) * v51_data);
              r1.template select<16, 1>(48) = v135_acc;
              tensorforge::intel_esimd::simd<float, 16> v162_acc{};
              tensorforge::intel_esimd::simd<float, 16> v164_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v162_acc += ((static_cast<float>(v164_data[0])) * v40_data);
              v162_acc += ((static_cast<float>(v164_data[1])) * v41_data);
              v162_acc += ((static_cast<float>(v164_data[2])) * v42_data);
              v162_acc += ((static_cast<float>(v164_data[3])) * v43_data);
              v162_acc += ((static_cast<float>(v164_data[4])) * v44_data);
              v162_acc += ((static_cast<float>(v164_data[5])) * v45_data);
              v162_acc += ((static_cast<float>(v164_data[6])) * v46_data);
              v162_acc += ((static_cast<float>(v164_data[7])) * v47_data);
              v162_acc += ((static_cast<float>(v164_data[8])) * v48_data);
              v162_acc += ((static_cast<float>(v164_data[9])) * v49_data);
              v162_acc += ((static_cast<float>(v164_data[10])) * v50_data);
              v162_acc += ((static_cast<float>(v164_data[11])) * v51_data);
              r1.template select<16, 1>(64) = v162_acc;
              tensorforge::intel_esimd::simd<float, 16> v189_acc{};
              tensorforge::intel_esimd::simd<float, 16> v191_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v189_acc += ((static_cast<float>(v191_data[0])) * v40_data);
              v189_acc += ((static_cast<float>(v191_data[1])) * v41_data);
              v189_acc += ((static_cast<float>(v191_data[2])) * v42_data);
              v189_acc += ((static_cast<float>(v191_data[3])) * v43_data);
              v189_acc += ((static_cast<float>(v191_data[4])) * v44_data);
              v189_acc += ((static_cast<float>(v191_data[5])) * v45_data);
              v189_acc += ((static_cast<float>(v191_data[6])) * v46_data);
              v189_acc += ((static_cast<float>(v191_data[7])) * v47_data);
              v189_acc += ((static_cast<float>(v191_data[8])) * v48_data);
              v189_acc += ((static_cast<float>(v191_data[9])) * v49_data);
              v189_acc += ((static_cast<float>(v191_data[10])) * v50_data);
              v189_acc += ((static_cast<float>(v191_data[11])) * v51_data);
              r1.template select<16, 1>(80) = v189_acc;
              // s1 = store{r>s, clear}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v216_z1 = 6; v216_z1 < 12; ++v216_z1) {
                s1[(v216_z1 * 12)] = 0.0f;
              }
              #pragma unroll
              for (int32_t v222_i1 = 0; v222_i1 < 6; ++v222_i1) {
                tensorforge::intel_esimd::simd<float, 12> v225_data(r1.template select<12, 1>((v222_i1 * 16)));
                tensorforge::slmStore<float, 12>(s1 + ((v222_i1 * 12)), v225_data);
              }
              // wait(r2 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v232_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v233_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v234_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v235_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v236_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v237_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v238_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v239_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v240_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v241_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v242_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v243_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v244_acc{};
              tensorforge::intel_esimd::simd<float, 16> v248_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v244_acc += ((static_cast<float>(v248_data[0])) * v232_data);
              v244_acc += ((static_cast<float>(v248_data[1])) * v233_data);
              v244_acc += ((static_cast<float>(v248_data[2])) * v234_data);
              v244_acc += ((static_cast<float>(v248_data[3])) * v235_data);
              v244_acc += ((static_cast<float>(v248_data[4])) * v236_data);
              v244_acc += ((static_cast<float>(v248_data[5])) * v237_data);
              v244_acc += ((static_cast<float>(v248_data[6])) * v238_data);
              v244_acc += ((static_cast<float>(v248_data[7])) * v239_data);
              v244_acc += ((static_cast<float>(v248_data[8])) * v240_data);
              v244_acc += ((static_cast<float>(v248_data[9])) * v241_data);
              v244_acc += ((static_cast<float>(v248_data[10])) * v242_data);
              v244_acc += ((static_cast<float>(v248_data[11])) * v243_data);
              ir3.template select<16, 1>(0) = v244_acc;
              tensorforge::intel_esimd::simd<float, 16> v273_acc{};
              tensorforge::intel_esimd::simd<float, 16> v275_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v273_acc += ((static_cast<float>(v275_data[0])) * v232_data);
              v273_acc += ((static_cast<float>(v275_data[1])) * v233_data);
              v273_acc += ((static_cast<float>(v275_data[2])) * v234_data);
              v273_acc += ((static_cast<float>(v275_data[3])) * v235_data);
              v273_acc += ((static_cast<float>(v275_data[4])) * v236_data);
              v273_acc += ((static_cast<float>(v275_data[5])) * v237_data);
              v273_acc += ((static_cast<float>(v275_data[6])) * v238_data);
              v273_acc += ((static_cast<float>(v275_data[7])) * v239_data);
              v273_acc += ((static_cast<float>(v275_data[8])) * v240_data);
              v273_acc += ((static_cast<float>(v275_data[9])) * v241_data);
              v273_acc += ((static_cast<float>(v275_data[10])) * v242_data);
              v273_acc += ((static_cast<float>(v275_data[11])) * v243_data);
              ir3.template select<16, 1>(16) = v273_acc;
              tensorforge::intel_esimd::simd<float, 16> v300_acc{};
              tensorforge::intel_esimd::simd<float, 16> v302_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v300_acc += ((static_cast<float>(v302_data[0])) * v232_data);
              v300_acc += ((static_cast<float>(v302_data[1])) * v233_data);
              v300_acc += ((static_cast<float>(v302_data[2])) * v234_data);
              v300_acc += ((static_cast<float>(v302_data[3])) * v235_data);
              v300_acc += ((static_cast<float>(v302_data[4])) * v236_data);
              v300_acc += ((static_cast<float>(v302_data[5])) * v237_data);
              v300_acc += ((static_cast<float>(v302_data[6])) * v238_data);
              v300_acc += ((static_cast<float>(v302_data[7])) * v239_data);
              v300_acc += ((static_cast<float>(v302_data[8])) * v240_data);
              v300_acc += ((static_cast<float>(v302_data[9])) * v241_data);
              v300_acc += ((static_cast<float>(v302_data[10])) * v242_data);
              v300_acc += ((static_cast<float>(v302_data[11])) * v243_data);
              ir3.template select<16, 1>(32) = v300_acc;
              tensorforge::intel_esimd::simd<float, 16> v327_acc{};
              tensorforge::intel_esimd::simd<float, 16> v329_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v327_acc += ((static_cast<float>(v329_data[0])) * v232_data);
              v327_acc += ((static_cast<float>(v329_data[1])) * v233_data);
              v327_acc += ((static_cast<float>(v329_data[2])) * v234_data);
              v327_acc += ((static_cast<float>(v329_data[3])) * v235_data);
              v327_acc += ((static_cast<float>(v329_data[4])) * v236_data);
              v327_acc += ((static_cast<float>(v329_data[5])) * v237_data);
              v327_acc += ((static_cast<float>(v329_data[6])) * v238_data);
              v327_acc += ((static_cast<float>(v329_data[7])) * v239_data);
              v327_acc += ((static_cast<float>(v329_data[8])) * v240_data);
              v327_acc += ((static_cast<float>(v329_data[9])) * v241_data);
              v327_acc += ((static_cast<float>(v329_data[10])) * v242_data);
              v327_acc += ((static_cast<float>(v329_data[11])) * v243_data);
              ir3.template select<16, 1>(48) = v327_acc;
              tensorforge::intel_esimd::simd<float, 16> v354_acc{};
              tensorforge::intel_esimd::simd<float, 16> v356_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v354_acc += ((static_cast<float>(v356_data[0])) * v232_data);
              v354_acc += ((static_cast<float>(v356_data[1])) * v233_data);
              v354_acc += ((static_cast<float>(v356_data[2])) * v234_data);
              v354_acc += ((static_cast<float>(v356_data[3])) * v235_data);
              v354_acc += ((static_cast<float>(v356_data[4])) * v236_data);
              v354_acc += ((static_cast<float>(v356_data[5])) * v237_data);
              v354_acc += ((static_cast<float>(v356_data[6])) * v238_data);
              v354_acc += ((static_cast<float>(v356_data[7])) * v239_data);
              v354_acc += ((static_cast<float>(v356_data[8])) * v240_data);
              v354_acc += ((static_cast<float>(v356_data[9])) * v241_data);
              v354_acc += ((static_cast<float>(v356_data[10])) * v242_data);
              v354_acc += ((static_cast<float>(v356_data[11])) * v243_data);
              ir3.template select<16, 1>(64) = v354_acc;
              tensorforge::intel_esimd::simd<float, 16> v381_acc{};
              tensorforge::intel_esimd::simd<float, 16> v383_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v381_acc += ((static_cast<float>(v383_data[0])) * v232_data);
              v381_acc += ((static_cast<float>(v383_data[1])) * v233_data);
              v381_acc += ((static_cast<float>(v383_data[2])) * v234_data);
              v381_acc += ((static_cast<float>(v383_data[3])) * v235_data);
              v381_acc += ((static_cast<float>(v383_data[4])) * v236_data);
              v381_acc += ((static_cast<float>(v383_data[5])) * v237_data);
              v381_acc += ((static_cast<float>(v383_data[6])) * v238_data);
              v381_acc += ((static_cast<float>(v383_data[7])) * v239_data);
              v381_acc += ((static_cast<float>(v383_data[8])) * v240_data);
              v381_acc += ((static_cast<float>(v383_data[9])) * v241_data);
              v381_acc += ((static_cast<float>(v383_data[10])) * v242_data);
              v381_acc += ((static_cast<float>(v383_data[11])) * v243_data);
              ir3.template select<16, 1>(80) = v381_acc;
              tensorforge::intel_esimd::simd<float, 16> v408_acc{};
              tensorforge::intel_esimd::simd<float, 16> v410_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v408_acc += ((static_cast<float>(v410_data[0])) * v232_data);
              v408_acc += ((static_cast<float>(v410_data[1])) * v233_data);
              v408_acc += ((static_cast<float>(v410_data[2])) * v234_data);
              v408_acc += ((static_cast<float>(v410_data[3])) * v235_data);
              v408_acc += ((static_cast<float>(v410_data[4])) * v236_data);
              v408_acc += ((static_cast<float>(v410_data[5])) * v237_data);
              v408_acc += ((static_cast<float>(v410_data[6])) * v238_data);
              v408_acc += ((static_cast<float>(v410_data[7])) * v239_data);
              v408_acc += ((static_cast<float>(v410_data[8])) * v240_data);
              v408_acc += ((static_cast<float>(v410_data[9])) * v241_data);
              v408_acc += ((static_cast<float>(v410_data[10])) * v242_data);
              v408_acc += ((static_cast<float>(v410_data[11])) * v243_data);
              ir3.template select<16, 1>(96) = v408_acc;
              tensorforge::intel_esimd::simd<float, 16> v435_acc{};
              tensorforge::intel_esimd::simd<float, 16> v437_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v435_acc += ((static_cast<float>(v437_data[0])) * v232_data);
              v435_acc += ((static_cast<float>(v437_data[1])) * v233_data);
              v435_acc += ((static_cast<float>(v437_data[2])) * v234_data);
              v435_acc += ((static_cast<float>(v437_data[3])) * v235_data);
              v435_acc += ((static_cast<float>(v437_data[4])) * v236_data);
              v435_acc += ((static_cast<float>(v437_data[5])) * v237_data);
              v435_acc += ((static_cast<float>(v437_data[6])) * v238_data);
              v435_acc += ((static_cast<float>(v437_data[7])) * v239_data);
              v435_acc += ((static_cast<float>(v437_data[8])) * v240_data);
              v435_acc += ((static_cast<float>(v437_data[9])) * v241_data);
              v435_acc += ((static_cast<float>(v437_data[10])) * v242_data);
              v435_acc += ((static_cast<float>(v437_data[11])) * v243_data);
              ir3.template select<16, 1>(112) = v435_acc;
              tensorforge::intel_esimd::simd<float, 16> v462_acc{};
              tensorforge::intel_esimd::simd<float, 16> v464_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v462_acc += ((static_cast<float>(v464_data[0])) * v232_data);
              v462_acc += ((static_cast<float>(v464_data[1])) * v233_data);
              v462_acc += ((static_cast<float>(v464_data[2])) * v234_data);
              v462_acc += ((static_cast<float>(v464_data[3])) * v235_data);
              v462_acc += ((static_cast<float>(v464_data[4])) * v236_data);
              v462_acc += ((static_cast<float>(v464_data[5])) * v237_data);
              v462_acc += ((static_cast<float>(v464_data[6])) * v238_data);
              v462_acc += ((static_cast<float>(v464_data[7])) * v239_data);
              v462_acc += ((static_cast<float>(v464_data[8])) * v240_data);
              v462_acc += ((static_cast<float>(v464_data[9])) * v241_data);
              v462_acc += ((static_cast<float>(v464_data[10])) * v242_data);
              v462_acc += ((static_cast<float>(v464_data[11])) * v243_data);
              ir3.template select<16, 1>(128) = v462_acc;
              tensorforge::intel_esimd::simd<float, 16> v489_acc{};
              tensorforge::intel_esimd::simd<float, 16> v491_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v489_acc += ((static_cast<float>(v491_data[0])) * v232_data);
              v489_acc += ((static_cast<float>(v491_data[1])) * v233_data);
              v489_acc += ((static_cast<float>(v491_data[2])) * v234_data);
              v489_acc += ((static_cast<float>(v491_data[3])) * v235_data);
              v489_acc += ((static_cast<float>(v491_data[4])) * v236_data);
              v489_acc += ((static_cast<float>(v491_data[5])) * v237_data);
              v489_acc += ((static_cast<float>(v491_data[6])) * v238_data);
              v489_acc += ((static_cast<float>(v491_data[7])) * v239_data);
              v489_acc += ((static_cast<float>(v491_data[8])) * v240_data);
              v489_acc += ((static_cast<float>(v491_data[9])) * v241_data);
              v489_acc += ((static_cast<float>(v491_data[10])) * v242_data);
              v489_acc += ((static_cast<float>(v491_data[11])) * v243_data);
              ir3.template select<16, 1>(144) = v489_acc;
              tensorforge::intel_esimd::simd<float, 16> v516_acc{};
              tensorforge::intel_esimd::simd<float, 16> v518_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v516_acc += ((static_cast<float>(v518_data[0])) * v232_data);
              v516_acc += ((static_cast<float>(v518_data[1])) * v233_data);
              v516_acc += ((static_cast<float>(v518_data[2])) * v234_data);
              v516_acc += ((static_cast<float>(v518_data[3])) * v235_data);
              v516_acc += ((static_cast<float>(v518_data[4])) * v236_data);
              v516_acc += ((static_cast<float>(v518_data[5])) * v237_data);
              v516_acc += ((static_cast<float>(v518_data[6])) * v238_data);
              v516_acc += ((static_cast<float>(v518_data[7])) * v239_data);
              v516_acc += ((static_cast<float>(v518_data[8])) * v240_data);
              v516_acc += ((static_cast<float>(v518_data[9])) * v241_data);
              v516_acc += ((static_cast<float>(v518_data[10])) * v242_data);
              v516_acc += ((static_cast<float>(v518_data[11])) * v243_data);
              ir3.template select<16, 1>(160) = v516_acc;
              tensorforge::intel_esimd::simd<float, 16> v543_acc{};
              tensorforge::intel_esimd::simd<float, 16> v545_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v543_acc += ((static_cast<float>(v545_data[0])) * v232_data);
              v543_acc += ((static_cast<float>(v545_data[1])) * v233_data);
              v543_acc += ((static_cast<float>(v545_data[2])) * v234_data);
              v543_acc += ((static_cast<float>(v545_data[3])) * v235_data);
              v543_acc += ((static_cast<float>(v545_data[4])) * v236_data);
              v543_acc += ((static_cast<float>(v545_data[5])) * v237_data);
              v543_acc += ((static_cast<float>(v545_data[6])) * v238_data);
              v543_acc += ((static_cast<float>(v545_data[7])) * v239_data);
              v543_acc += ((static_cast<float>(v545_data[8])) * v240_data);
              v543_acc += ((static_cast<float>(v545_data[9])) * v241_data);
              v543_acc += ((static_cast<float>(v545_data[10])) * v242_data);
              v543_acc += ((static_cast<float>(v545_data[11])) * v243_data);
              ir3.template select<16, 1>(176) = v543_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v570_n1 = 0; v570_n1 < 12; ++v570_n1) {
                int32_t v571_a = v570_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v573_data(ir3.template select<12, 1>(v571_a));
                r3.template select<12, 1>(v571_a) = v573_data;
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v574_i1 = 0; v574_i1 < 12; ++v574_i1) {
                tensorforge::intel_esimd::simd<float, 12> v577_data(r3.template select<12, 1>((v574_i1 * 16)));
                v577_data.copy_to(glb_m2 + ((v574_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


// === base name ===
kernel_e7888dfe55c13aa8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e7888dfe55c13aa8 = {{1, 16, 1}, 16, 9, 1, 16, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e7888dfe55c13aa8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e7888dfe55c13aa8(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e7888dfe55c13aa8(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1792 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e7888dfe55c13aa8(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e7888dfe55c13aa8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e7888dfe55c13aa8(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e7888dfe55c13aa8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1792 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (9 active) x 16 per block = block 1x16x1, 7168 B shared, occupancy grid
        // operands:
        //   m0 9×9(9×9) {0..9}×{0..9} strided
        //   m1 9×9(9×9) {0..9}×{0..9} strided
        //   m2 9×9(9×9) {0..9}×{0..9} strided
        //   m3 ()  scalar
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j] × m3[]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":9,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1792}],"shared_bytes":7168,"shared_elements":1792,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[0,1],[0,1],[]],"target":[[0,-1],[-1,1],[]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (112 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 81 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 81 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 81 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 9> v24_data;
                v24_data.copy_from(glb_m1 + ((v19_i1 * 9)));
                r0.template select<9, 1>((v19_i1 * 16)) = v24_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v27_ld);
              tensorforge::intel_esimd::simd<float, 16> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 64), v28_ld);
              float v29_ld = glb_m2[0 + 0 + 1 * 0 + 80];
              s0[0 + 0 + 1 * 0 + 80] = v29_ld;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v41_acc{};
              tensorforge::intel_esimd::simd<float, 16> v45_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v41_acc += ((static_cast<float>(v45_data[0])) * v32_data);
              v41_acc += ((static_cast<float>(v45_data[1])) * v33_data);
              v41_acc += ((static_cast<float>(v45_data[2])) * v34_data);
              v41_acc += ((static_cast<float>(v45_data[3])) * v35_data);
              v41_acc += ((static_cast<float>(v45_data[4])) * v36_data);
              v41_acc += ((static_cast<float>(v45_data[5])) * v37_data);
              v41_acc += ((static_cast<float>(v45_data[6])) * v38_data);
              v41_acc += ((static_cast<float>(v45_data[7])) * v39_data);
              v41_acc += ((static_cast<float>(v45_data[8])) * v40_data);
              ir1.template select<16, 1>(0) = v41_acc;
              tensorforge::intel_esimd::simd<float, 16> v64_acc{};
              tensorforge::intel_esimd::simd<float, 16> v66_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              v64_acc += ((static_cast<float>(v66_data[0])) * v32_data);
              v64_acc += ((static_cast<float>(v66_data[1])) * v33_data);
              v64_acc += ((static_cast<float>(v66_data[2])) * v34_data);
              v64_acc += ((static_cast<float>(v66_data[3])) * v35_data);
              v64_acc += ((static_cast<float>(v66_data[4])) * v36_data);
              v64_acc += ((static_cast<float>(v66_data[5])) * v37_data);
              v64_acc += ((static_cast<float>(v66_data[6])) * v38_data);
              v64_acc += ((static_cast<float>(v66_data[7])) * v39_data);
              v64_acc += ((static_cast<float>(v66_data[8])) * v40_data);
              ir1.template select<16, 1>(16) = v64_acc;
              tensorforge::intel_esimd::simd<float, 16> v85_acc{};
              tensorforge::intel_esimd::simd<float, 16> v87_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              v85_acc += ((static_cast<float>(v87_data[0])) * v32_data);
              v85_acc += ((static_cast<float>(v87_data[1])) * v33_data);
              v85_acc += ((static_cast<float>(v87_data[2])) * v34_data);
              v85_acc += ((static_cast<float>(v87_data[3])) * v35_data);
              v85_acc += ((static_cast<float>(v87_data[4])) * v36_data);
              v85_acc += ((static_cast<float>(v87_data[5])) * v37_data);
              v85_acc += ((static_cast<float>(v87_data[6])) * v38_data);
              v85_acc += ((static_cast<float>(v87_data[7])) * v39_data);
              v85_acc += ((static_cast<float>(v87_data[8])) * v40_data);
              ir1.template select<16, 1>(32) = v85_acc;
              tensorforge::intel_esimd::simd<float, 16> v106_acc{};
              tensorforge::intel_esimd::simd<float, 16> v108_data = tensorforge::slmLoad<float, 16>(s0 + (27_i32));
              v106_acc += ((static_cast<float>(v108_data[0])) * v32_data);
              v106_acc += ((static_cast<float>(v108_data[1])) * v33_data);
              v106_acc += ((static_cast<float>(v108_data[2])) * v34_data);
              v106_acc += ((static_cast<float>(v108_data[3])) * v35_data);
              v106_acc += ((static_cast<float>(v108_data[4])) * v36_data);
              v106_acc += ((static_cast<float>(v108_data[5])) * v37_data);
              v106_acc += ((static_cast<float>(v108_data[6])) * v38_data);
              v106_acc += ((static_cast<float>(v108_data[7])) * v39_data);
              v106_acc += ((static_cast<float>(v108_data[8])) * v40_data);
              ir1.template select<16, 1>(48) = v106_acc;
              tensorforge::intel_esimd::simd<float, 16> v127_acc{};
              tensorforge::intel_esimd::simd<float, 16> v129_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v127_acc += ((static_cast<float>(v129_data[0])) * v32_data);
              v127_acc += ((static_cast<float>(v129_data[1])) * v33_data);
              v127_acc += ((static_cast<float>(v129_data[2])) * v34_data);
              v127_acc += ((static_cast<float>(v129_data[3])) * v35_data);
              v127_acc += ((static_cast<float>(v129_data[4])) * v36_data);
              v127_acc += ((static_cast<float>(v129_data[5])) * v37_data);
              v127_acc += ((static_cast<float>(v129_data[6])) * v38_data);
              v127_acc += ((static_cast<float>(v129_data[7])) * v39_data);
              v127_acc += ((static_cast<float>(v129_data[8])) * v40_data);
              ir1.template select<16, 1>(64) = v127_acc;
              tensorforge::intel_esimd::simd<float, 16> v148_acc{};
              tensorforge::intel_esimd::simd<float, 16> v150_data = tensorforge::slmLoad<float, 16>(s0 + (45_i32));
              v148_acc += ((static_cast<float>(v150_data[0])) * v32_data);
              v148_acc += ((static_cast<float>(v150_data[1])) * v33_data);
              v148_acc += ((static_cast<float>(v150_data[2])) * v34_data);
              v148_acc += ((static_cast<float>(v150_data[3])) * v35_data);
              v148_acc += ((static_cast<float>(v150_data[4])) * v36_data);
              v148_acc += ((static_cast<float>(v150_data[5])) * v37_data);
              v148_acc += ((static_cast<float>(v150_data[6])) * v38_data);
              v148_acc += ((static_cast<float>(v150_data[7])) * v39_data);
              v148_acc += ((static_cast<float>(v150_data[8])) * v40_data);
              ir1.template select<16, 1>(80) = v148_acc;
              tensorforge::intel_esimd::simd<float, 16> v169_acc{};
              tensorforge::intel_esimd::simd<float, 16> v171_data = tensorforge::slmLoad<float, 16>(s0 + (54_i32));
              v169_acc += ((static_cast<float>(v171_data[0])) * v32_data);
              v169_acc += ((static_cast<float>(v171_data[1])) * v33_data);
              v169_acc += ((static_cast<float>(v171_data[2])) * v34_data);
              v169_acc += ((static_cast<float>(v171_data[3])) * v35_data);
              v169_acc += ((static_cast<float>(v171_data[4])) * v36_data);
              v169_acc += ((static_cast<float>(v171_data[5])) * v37_data);
              v169_acc += ((static_cast<float>(v171_data[6])) * v38_data);
              v169_acc += ((static_cast<float>(v171_data[7])) * v39_data);
              v169_acc += ((static_cast<float>(v171_data[8])) * v40_data);
              ir1.template select<16, 1>(96) = v169_acc;
              tensorforge::intel_esimd::simd<float, 16> v190_acc{};
              tensorforge::intel_esimd::simd<float, 16> v192_data = tensorforge::slmLoad<float, 16>(s0 + (63_i32));
              v190_acc += ((static_cast<float>(v192_data[0])) * v32_data);
              v190_acc += ((static_cast<float>(v192_data[1])) * v33_data);
              v190_acc += ((static_cast<float>(v192_data[2])) * v34_data);
              v190_acc += ((static_cast<float>(v192_data[3])) * v35_data);
              v190_acc += ((static_cast<float>(v192_data[4])) * v36_data);
              v190_acc += ((static_cast<float>(v192_data[5])) * v37_data);
              v190_acc += ((static_cast<float>(v192_data[6])) * v38_data);
              v190_acc += ((static_cast<float>(v192_data[7])) * v39_data);
              v190_acc += ((static_cast<float>(v192_data[8])) * v40_data);
              ir1.template select<16, 1>(112) = v190_acc;
              tensorforge::intel_esimd::simd<float, 16> v211_acc{};
              tensorforge::intel_esimd::simd<float, 16> v213_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v211_acc += ((static_cast<float>(v213_data[0])) * v32_data);
              v211_acc += ((static_cast<float>(v213_data[1])) * v33_data);
              v211_acc += ((static_cast<float>(v213_data[2])) * v34_data);
              v211_acc += ((static_cast<float>(v213_data[3])) * v35_data);
              v211_acc += ((static_cast<float>(v213_data[4])) * v36_data);
              v211_acc += ((static_cast<float>(v213_data[5])) * v37_data);
              v211_acc += ((static_cast<float>(v213_data[6])) * v38_data);
              v211_acc += ((static_cast<float>(v213_data[7])) * v39_data);
              v211_acc += ((static_cast<float>(v213_data[8])) * v40_data);
              ir1.template select<16, 1>(128) = v211_acc;
              #pragma unroll
              for (int32_t v233_n1 = 0; v233_n1 < 9; ++v233_n1) {
                int32_t v234_a = v233_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v236_data(ir1.template select<9, 1>(v234_a));
                r1.template select<9, 1>(v234_a) = (v236_data * 13.0f);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v238_i1 = 0; v238_i1 < 9; ++v238_i1) {
                tensorforge::intel_esimd::simd<float, 9> v241_data(r1.template select<9, 1>((v238_i1 * 16)));
                v241_data.copy_to(glb_m0 + ((v238_i1 * 9)));
              }
            }
            tensorforge::prefetchRunsL2<324, 324>(&pf_glb_m1[0], &pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


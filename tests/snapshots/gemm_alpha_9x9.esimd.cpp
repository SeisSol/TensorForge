// === base name ===
kernel_1e8e705750f29ff5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1e8e705750f29ff5 = {{1, 16, 1}, 16, 9, 1, 16, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1e8e705750f29ff5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1e8e705750f29ff5(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1e8e705750f29ff5(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1792 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_1e8e705750f29ff5(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1e8e705750f29ff5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1e8e705750f29ff5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1e8e705750f29ff5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 81 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v17_i1 = 0; v17_i1 < 9; ++v17_i1) {
                tensorforge::intel_esimd::simd<float, 9> v22_data;
                v22_data.copy_from(glb_m1 + ((v17_i1 * 9)));
                r0.template select<9, 1>((v17_i1 * 16)) = v22_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v25_ld;
              v25_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v25_ld);
              tensorforge::intel_esimd::simd<float, 16> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 64), v26_ld);
              float v27_ld = glb_m2[0 + 0 + 1 * 0 + 80];
              s0[0 + 0 + 1 * 0 + 80] = v27_ld;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 9), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v30_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v31_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v39_acc{};
              tensorforge::intel_esimd::simd<float, 16> v43_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v39_acc += ((static_cast<float>(v43_data[0])) * v30_data);
              v39_acc += ((static_cast<float>(v43_data[1])) * v31_data);
              v39_acc += ((static_cast<float>(v43_data[2])) * v32_data);
              v39_acc += ((static_cast<float>(v43_data[3])) * v33_data);
              v39_acc += ((static_cast<float>(v43_data[4])) * v34_data);
              v39_acc += ((static_cast<float>(v43_data[5])) * v35_data);
              v39_acc += ((static_cast<float>(v43_data[6])) * v36_data);
              v39_acc += ((static_cast<float>(v43_data[7])) * v37_data);
              v39_acc += ((static_cast<float>(v43_data[8])) * v38_data);
              ir1.template select<16, 1>(0) = v39_acc;
              tensorforge::intel_esimd::simd<float, 16> v62_acc{};
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              v62_acc += ((static_cast<float>(v64_data[0])) * v30_data);
              v62_acc += ((static_cast<float>(v64_data[1])) * v31_data);
              v62_acc += ((static_cast<float>(v64_data[2])) * v32_data);
              v62_acc += ((static_cast<float>(v64_data[3])) * v33_data);
              v62_acc += ((static_cast<float>(v64_data[4])) * v34_data);
              v62_acc += ((static_cast<float>(v64_data[5])) * v35_data);
              v62_acc += ((static_cast<float>(v64_data[6])) * v36_data);
              v62_acc += ((static_cast<float>(v64_data[7])) * v37_data);
              v62_acc += ((static_cast<float>(v64_data[8])) * v38_data);
              ir1.template select<16, 1>(16) = v62_acc;
              tensorforge::intel_esimd::simd<float, 16> v83_acc{};
              tensorforge::intel_esimd::simd<float, 16> v85_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              v83_acc += ((static_cast<float>(v85_data[0])) * v30_data);
              v83_acc += ((static_cast<float>(v85_data[1])) * v31_data);
              v83_acc += ((static_cast<float>(v85_data[2])) * v32_data);
              v83_acc += ((static_cast<float>(v85_data[3])) * v33_data);
              v83_acc += ((static_cast<float>(v85_data[4])) * v34_data);
              v83_acc += ((static_cast<float>(v85_data[5])) * v35_data);
              v83_acc += ((static_cast<float>(v85_data[6])) * v36_data);
              v83_acc += ((static_cast<float>(v85_data[7])) * v37_data);
              v83_acc += ((static_cast<float>(v85_data[8])) * v38_data);
              ir1.template select<16, 1>(32) = v83_acc;
              tensorforge::intel_esimd::simd<float, 16> v104_acc{};
              tensorforge::intel_esimd::simd<float, 16> v106_data = tensorforge::slmLoad<float, 16>(s0 + (27_i32));
              v104_acc += ((static_cast<float>(v106_data[0])) * v30_data);
              v104_acc += ((static_cast<float>(v106_data[1])) * v31_data);
              v104_acc += ((static_cast<float>(v106_data[2])) * v32_data);
              v104_acc += ((static_cast<float>(v106_data[3])) * v33_data);
              v104_acc += ((static_cast<float>(v106_data[4])) * v34_data);
              v104_acc += ((static_cast<float>(v106_data[5])) * v35_data);
              v104_acc += ((static_cast<float>(v106_data[6])) * v36_data);
              v104_acc += ((static_cast<float>(v106_data[7])) * v37_data);
              v104_acc += ((static_cast<float>(v106_data[8])) * v38_data);
              ir1.template select<16, 1>(48) = v104_acc;
              tensorforge::intel_esimd::simd<float, 16> v125_acc{};
              tensorforge::intel_esimd::simd<float, 16> v127_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v125_acc += ((static_cast<float>(v127_data[0])) * v30_data);
              v125_acc += ((static_cast<float>(v127_data[1])) * v31_data);
              v125_acc += ((static_cast<float>(v127_data[2])) * v32_data);
              v125_acc += ((static_cast<float>(v127_data[3])) * v33_data);
              v125_acc += ((static_cast<float>(v127_data[4])) * v34_data);
              v125_acc += ((static_cast<float>(v127_data[5])) * v35_data);
              v125_acc += ((static_cast<float>(v127_data[6])) * v36_data);
              v125_acc += ((static_cast<float>(v127_data[7])) * v37_data);
              v125_acc += ((static_cast<float>(v127_data[8])) * v38_data);
              ir1.template select<16, 1>(64) = v125_acc;
              tensorforge::intel_esimd::simd<float, 16> v146_acc{};
              tensorforge::intel_esimd::simd<float, 16> v148_data = tensorforge::slmLoad<float, 16>(s0 + (45_i32));
              v146_acc += ((static_cast<float>(v148_data[0])) * v30_data);
              v146_acc += ((static_cast<float>(v148_data[1])) * v31_data);
              v146_acc += ((static_cast<float>(v148_data[2])) * v32_data);
              v146_acc += ((static_cast<float>(v148_data[3])) * v33_data);
              v146_acc += ((static_cast<float>(v148_data[4])) * v34_data);
              v146_acc += ((static_cast<float>(v148_data[5])) * v35_data);
              v146_acc += ((static_cast<float>(v148_data[6])) * v36_data);
              v146_acc += ((static_cast<float>(v148_data[7])) * v37_data);
              v146_acc += ((static_cast<float>(v148_data[8])) * v38_data);
              ir1.template select<16, 1>(80) = v146_acc;
              tensorforge::intel_esimd::simd<float, 16> v167_acc{};
              tensorforge::intel_esimd::simd<float, 16> v169_data = tensorforge::slmLoad<float, 16>(s0 + (54_i32));
              v167_acc += ((static_cast<float>(v169_data[0])) * v30_data);
              v167_acc += ((static_cast<float>(v169_data[1])) * v31_data);
              v167_acc += ((static_cast<float>(v169_data[2])) * v32_data);
              v167_acc += ((static_cast<float>(v169_data[3])) * v33_data);
              v167_acc += ((static_cast<float>(v169_data[4])) * v34_data);
              v167_acc += ((static_cast<float>(v169_data[5])) * v35_data);
              v167_acc += ((static_cast<float>(v169_data[6])) * v36_data);
              v167_acc += ((static_cast<float>(v169_data[7])) * v37_data);
              v167_acc += ((static_cast<float>(v169_data[8])) * v38_data);
              ir1.template select<16, 1>(96) = v167_acc;
              tensorforge::intel_esimd::simd<float, 16> v188_acc{};
              tensorforge::intel_esimd::simd<float, 16> v190_data = tensorforge::slmLoad<float, 16>(s0 + (63_i32));
              v188_acc += ((static_cast<float>(v190_data[0])) * v30_data);
              v188_acc += ((static_cast<float>(v190_data[1])) * v31_data);
              v188_acc += ((static_cast<float>(v190_data[2])) * v32_data);
              v188_acc += ((static_cast<float>(v190_data[3])) * v33_data);
              v188_acc += ((static_cast<float>(v190_data[4])) * v34_data);
              v188_acc += ((static_cast<float>(v190_data[5])) * v35_data);
              v188_acc += ((static_cast<float>(v190_data[6])) * v36_data);
              v188_acc += ((static_cast<float>(v190_data[7])) * v37_data);
              v188_acc += ((static_cast<float>(v190_data[8])) * v38_data);
              ir1.template select<16, 1>(112) = v188_acc;
              tensorforge::intel_esimd::simd<float, 16> v209_acc{};
              tensorforge::intel_esimd::simd<float, 16> v211_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v209_acc += ((static_cast<float>(v211_data[0])) * v30_data);
              v209_acc += ((static_cast<float>(v211_data[1])) * v31_data);
              v209_acc += ((static_cast<float>(v211_data[2])) * v32_data);
              v209_acc += ((static_cast<float>(v211_data[3])) * v33_data);
              v209_acc += ((static_cast<float>(v211_data[4])) * v34_data);
              v209_acc += ((static_cast<float>(v211_data[5])) * v35_data);
              v209_acc += ((static_cast<float>(v211_data[6])) * v36_data);
              v209_acc += ((static_cast<float>(v211_data[7])) * v37_data);
              v209_acc += ((static_cast<float>(v211_data[8])) * v38_data);
              ir1.template select<16, 1>(128) = v209_acc;
              // r1 = ir1 * glb_m3
              #pragma unroll
              for (int32_t v231_n1 = 0; v231_n1 < 9; ++v231_n1) {
                int32_t v232_a = v231_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v234_data(ir1.template select<9, 1>(v232_a));
                r1.template select<9, 1>(v232_a) = (v234_data * 13.0f);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v236_i1 = 0; v236_i1 < 9; ++v236_i1) {
                tensorforge::intel_esimd::simd<float, 9> v239_data(r1.template select<9, 1>((v236_i1 * 16)));
                v239_data.copy_to(glb_m0 + ((v236_i1 * 9)));
              }
            }
          }
        }
      }
    });
  });
}


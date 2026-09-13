// === base name ===
kernel_17d1c5caa8b234c7

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_17d1c5caa8b234c7 = {{1, 16, 1}, 16, 16, 1, 16, 5120, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_17d1c5caa8b234c7(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_17d1c5caa8b234c7(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_17d1c5caa8b234c7(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_17d1c5caa8b234c7(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_17d1c5caa8b234c7(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_17d1c5caa8b234c7(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_17d1c5caa8b234c7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   C = abs(M)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1280}],"shared_bytes":5120,"shared_elements":1280,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"M","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (80 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 64 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v5_batchId0 * 64 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 128> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i1 = 0; v20_i1 < 8; ++v20_i1) {
                tensorforge::intel_esimd::simd<float, 8> v25_data;
                v25_data.copy_from(glb_m1 + ((v20_i1 * 8)));
                r0.template select<8, 1>((v20_i1 * 16)) = v25_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v28_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v31_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v39_acc{};
              tensorforge::intel_esimd::simd<float, 16> v43_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v39_acc += ((static_cast<float>(v43_data[0])) * v31_data);
              v39_acc += ((static_cast<float>(v43_data[1])) * v32_data);
              v39_acc += ((static_cast<float>(v43_data[2])) * v33_data);
              v39_acc += ((static_cast<float>(v43_data[3])) * v34_data);
              v39_acc += ((static_cast<float>(v43_data[4])) * v35_data);
              v39_acc += ((static_cast<float>(v43_data[5])) * v36_data);
              v39_acc += ((static_cast<float>(v43_data[6])) * v37_data);
              v39_acc += ((static_cast<float>(v43_data[7])) * v38_data);
              ir1.template select<16, 1>(0) = v39_acc;
              tensorforge::intel_esimd::simd<float, 16> v60_acc{};
              tensorforge::intel_esimd::simd<float, 16> v62_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              v60_acc += ((static_cast<float>(v62_data[0])) * v31_data);
              v60_acc += ((static_cast<float>(v62_data[1])) * v32_data);
              v60_acc += ((static_cast<float>(v62_data[2])) * v33_data);
              v60_acc += ((static_cast<float>(v62_data[3])) * v34_data);
              v60_acc += ((static_cast<float>(v62_data[4])) * v35_data);
              v60_acc += ((static_cast<float>(v62_data[5])) * v36_data);
              v60_acc += ((static_cast<float>(v62_data[6])) * v37_data);
              v60_acc += ((static_cast<float>(v62_data[7])) * v38_data);
              ir1.template select<16, 1>(16) = v60_acc;
              tensorforge::intel_esimd::simd<float, 16> v79_acc{};
              tensorforge::intel_esimd::simd<float, 16> v81_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v79_acc += ((static_cast<float>(v81_data[0])) * v31_data);
              v79_acc += ((static_cast<float>(v81_data[1])) * v32_data);
              v79_acc += ((static_cast<float>(v81_data[2])) * v33_data);
              v79_acc += ((static_cast<float>(v81_data[3])) * v34_data);
              v79_acc += ((static_cast<float>(v81_data[4])) * v35_data);
              v79_acc += ((static_cast<float>(v81_data[5])) * v36_data);
              v79_acc += ((static_cast<float>(v81_data[6])) * v37_data);
              v79_acc += ((static_cast<float>(v81_data[7])) * v38_data);
              ir1.template select<16, 1>(32) = v79_acc;
              tensorforge::intel_esimd::simd<float, 16> v98_acc{};
              tensorforge::intel_esimd::simd<float, 16> v100_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v98_acc += ((static_cast<float>(v100_data[0])) * v31_data);
              v98_acc += ((static_cast<float>(v100_data[1])) * v32_data);
              v98_acc += ((static_cast<float>(v100_data[2])) * v33_data);
              v98_acc += ((static_cast<float>(v100_data[3])) * v34_data);
              v98_acc += ((static_cast<float>(v100_data[4])) * v35_data);
              v98_acc += ((static_cast<float>(v100_data[5])) * v36_data);
              v98_acc += ((static_cast<float>(v100_data[6])) * v37_data);
              v98_acc += ((static_cast<float>(v100_data[7])) * v38_data);
              ir1.template select<16, 1>(48) = v98_acc;
              tensorforge::intel_esimd::simd<float, 16> v117_acc{};
              tensorforge::intel_esimd::simd<float, 16> v119_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v117_acc += ((static_cast<float>(v119_data[0])) * v31_data);
              v117_acc += ((static_cast<float>(v119_data[1])) * v32_data);
              v117_acc += ((static_cast<float>(v119_data[2])) * v33_data);
              v117_acc += ((static_cast<float>(v119_data[3])) * v34_data);
              v117_acc += ((static_cast<float>(v119_data[4])) * v35_data);
              v117_acc += ((static_cast<float>(v119_data[5])) * v36_data);
              v117_acc += ((static_cast<float>(v119_data[6])) * v37_data);
              v117_acc += ((static_cast<float>(v119_data[7])) * v38_data);
              ir1.template select<16, 1>(64) = v117_acc;
              tensorforge::intel_esimd::simd<float, 16> v136_acc{};
              tensorforge::intel_esimd::simd<float, 16> v138_data = tensorforge::slmLoad<float, 16>(s0 + (40_i32));
              v136_acc += ((static_cast<float>(v138_data[0])) * v31_data);
              v136_acc += ((static_cast<float>(v138_data[1])) * v32_data);
              v136_acc += ((static_cast<float>(v138_data[2])) * v33_data);
              v136_acc += ((static_cast<float>(v138_data[3])) * v34_data);
              v136_acc += ((static_cast<float>(v138_data[4])) * v35_data);
              v136_acc += ((static_cast<float>(v138_data[5])) * v36_data);
              v136_acc += ((static_cast<float>(v138_data[6])) * v37_data);
              v136_acc += ((static_cast<float>(v138_data[7])) * v38_data);
              ir1.template select<16, 1>(80) = v136_acc;
              tensorforge::intel_esimd::simd<float, 16> v155_acc{};
              tensorforge::intel_esimd::simd<float, 16> v157_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v155_acc += ((static_cast<float>(v157_data[0])) * v31_data);
              v155_acc += ((static_cast<float>(v157_data[1])) * v32_data);
              v155_acc += ((static_cast<float>(v157_data[2])) * v33_data);
              v155_acc += ((static_cast<float>(v157_data[3])) * v34_data);
              v155_acc += ((static_cast<float>(v157_data[4])) * v35_data);
              v155_acc += ((static_cast<float>(v157_data[5])) * v36_data);
              v155_acc += ((static_cast<float>(v157_data[6])) * v37_data);
              v155_acc += ((static_cast<float>(v157_data[7])) * v38_data);
              ir1.template select<16, 1>(96) = v155_acc;
              tensorforge::intel_esimd::simd<float, 16> v174_acc{};
              tensorforge::intel_esimd::simd<float, 16> v176_data = tensorforge::slmLoad<float, 16>(s0 + (56_i32));
              v174_acc += ((static_cast<float>(v176_data[0])) * v31_data);
              v174_acc += ((static_cast<float>(v176_data[1])) * v32_data);
              v174_acc += ((static_cast<float>(v176_data[2])) * v33_data);
              v174_acc += ((static_cast<float>(v176_data[3])) * v34_data);
              v174_acc += ((static_cast<float>(v176_data[4])) * v35_data);
              v174_acc += ((static_cast<float>(v176_data[5])) * v36_data);
              v174_acc += ((static_cast<float>(v176_data[6])) * v37_data);
              v174_acc += ((static_cast<float>(v176_data[7])) * v38_data);
              ir1.template select<16, 1>(112) = v174_acc;
              #pragma unroll
              for (int32_t v193_n1 = 0; v193_n1 < 8; ++v193_n1) {
                int32_t v194_a = v193_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v196_data(ir1.template select<8, 1>(v194_a));
                r1.template select<8, 1>(v194_a) = v196_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v197_i1 = 0; v197_i1 < 8; ++v197_i1) {
                tensorforge::intel_esimd::simd<float, 8> v200_data(r1.template select<8, 1>((v197_i1 * 16)));
                v200_data.copy_to(glb_m0 + ((v197_i1 * 8)));
              }
              // glb_m3 = abs(glb_m0)
              #pragma unroll
              for (int32_t v205_k1 = 0; v205_k1 < 8; ++v205_k1) {
                int32_t v208_a = v205_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v210_data;
                v210_data.copy_from(glb_m0 + (v208_a));
                (tensorforge::intel_esimd::abs(v210_data)).copy_to(glb_m3 + (v208_a));
              }
            }
            tensorforge::prefetchRunsL2<256, 256>(&pf_glb_m1[0], &pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


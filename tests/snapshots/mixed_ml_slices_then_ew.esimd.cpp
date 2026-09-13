// === base name ===
kernel_b5c2c5b25a7b8c6f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b5c2c5b25a7b8c6f = {{1, 16, 1}, 16, 16, 1, 16, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b5c2c5b25a7b8c6f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b5c2c5b25a7b8c6f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b5c2c5b25a7b8c6f(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_b5c2c5b25a7b8c6f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b5c2c5b25a7b8c6f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b5c2c5b25a7b8c6f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b5c2c5b25a7b8c6f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1792 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 7168 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×4(8×4) {0..8}×{0..4} strided
        //   m2 8×4(8×4) {0..8}×{0..4} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j]@{0..8}×{0..4} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{0..8}×{4..8} = m0[i,k] × m2[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1792}],"shared_bytes":7168,"shared_elements":1792,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (112 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const float *const __restrict__ pf_glb_m0 = &m0[v10_batchId1 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ pf_glb_m1 = &m1[v10_batchId1 * 32 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v10_batchId1 * 32 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 128> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 8; ++v23_i1) {
                tensorforge::intel_esimd::simd<float, 8> v28_data;
                v28_data.copy_from(glb_m0 + ((v23_i1 * 8)));
                r0.template select<8, 1>((v23_i1 * 16)) = v28_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v31_ld;
              v31_ld.copy_from(glb_m1 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 0), v31_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 2 * 0 + 0), v32_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v42_acc{};
              tensorforge::intel_esimd::simd<float, 16> v46_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v42_acc += ((static_cast<float>(v46_data[0])) * v34_data);
              v42_acc += ((static_cast<float>(v46_data[1])) * v35_data);
              v42_acc += ((static_cast<float>(v46_data[2])) * v36_data);
              v42_acc += ((static_cast<float>(v46_data[3])) * v37_data);
              v42_acc += ((static_cast<float>(v46_data[4])) * v38_data);
              v42_acc += ((static_cast<float>(v46_data[5])) * v39_data);
              v42_acc += ((static_cast<float>(v46_data[6])) * v40_data);
              v42_acc += ((static_cast<float>(v46_data[7])) * v41_data);
              r1.template select<16, 1>(0) = v42_acc;
              tensorforge::intel_esimd::simd<float, 16> v63_acc{};
              tensorforge::intel_esimd::simd<float, 16> v65_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              v63_acc += ((static_cast<float>(v65_data[0])) * v34_data);
              v63_acc += ((static_cast<float>(v65_data[1])) * v35_data);
              v63_acc += ((static_cast<float>(v65_data[2])) * v36_data);
              v63_acc += ((static_cast<float>(v65_data[3])) * v37_data);
              v63_acc += ((static_cast<float>(v65_data[4])) * v38_data);
              v63_acc += ((static_cast<float>(v65_data[5])) * v39_data);
              v63_acc += ((static_cast<float>(v65_data[6])) * v40_data);
              v63_acc += ((static_cast<float>(v65_data[7])) * v41_data);
              r1.template select<16, 1>(16) = v63_acc;
              tensorforge::intel_esimd::simd<float, 16> v82_acc{};
              tensorforge::intel_esimd::simd<float, 16> v84_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v82_acc += ((static_cast<float>(v84_data[0])) * v34_data);
              v82_acc += ((static_cast<float>(v84_data[1])) * v35_data);
              v82_acc += ((static_cast<float>(v84_data[2])) * v36_data);
              v82_acc += ((static_cast<float>(v84_data[3])) * v37_data);
              v82_acc += ((static_cast<float>(v84_data[4])) * v38_data);
              v82_acc += ((static_cast<float>(v84_data[5])) * v39_data);
              v82_acc += ((static_cast<float>(v84_data[6])) * v40_data);
              v82_acc += ((static_cast<float>(v84_data[7])) * v41_data);
              r1.template select<16, 1>(32) = v82_acc;
              tensorforge::intel_esimd::simd<float, 16> v101_acc{};
              tensorforge::intel_esimd::simd<float, 16> v103_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v101_acc += ((static_cast<float>(v103_data[0])) * v34_data);
              v101_acc += ((static_cast<float>(v103_data[1])) * v35_data);
              v101_acc += ((static_cast<float>(v103_data[2])) * v36_data);
              v101_acc += ((static_cast<float>(v103_data[3])) * v37_data);
              v101_acc += ((static_cast<float>(v103_data[4])) * v38_data);
              v101_acc += ((static_cast<float>(v103_data[5])) * v39_data);
              v101_acc += ((static_cast<float>(v103_data[6])) * v40_data);
              v101_acc += ((static_cast<float>(v103_data[7])) * v41_data);
              r1.template select<16, 1>(48) = v101_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v120_i1 = 0; v120_i1 < 4; ++v120_i1) {
                tensorforge::intel_esimd::simd<float, 8> v123_data(r1.template select<8, 1>((v120_i1 * 16)));
                tensorforge::slmStore<float, 8>(s1 + ((v120_i1 * 8)), v123_data);
              }
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r2(0.0f);
              // r2 = +(r0 * s2) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 64> ir2(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v138_acc{};
              tensorforge::intel_esimd::simd<float, 16> v142_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v138_acc += ((static_cast<float>(v142_data[0])) * v34_data);
              v138_acc += ((static_cast<float>(v142_data[1])) * v35_data);
              v138_acc += ((static_cast<float>(v142_data[2])) * v36_data);
              v138_acc += ((static_cast<float>(v142_data[3])) * v37_data);
              v138_acc += ((static_cast<float>(v142_data[4])) * v38_data);
              v138_acc += ((static_cast<float>(v142_data[5])) * v39_data);
              v138_acc += ((static_cast<float>(v142_data[6])) * v40_data);
              v138_acc += ((static_cast<float>(v142_data[7])) * v41_data);
              ir2.template select<16, 1>(0) = v138_acc;
              tensorforge::intel_esimd::simd<float, 16> v159_acc{};
              tensorforge::intel_esimd::simd<float, 16> v161_data = tensorforge::slmLoad<float, 16>(s2 + (8_i32));
              v159_acc += ((static_cast<float>(v161_data[0])) * v34_data);
              v159_acc += ((static_cast<float>(v161_data[1])) * v35_data);
              v159_acc += ((static_cast<float>(v161_data[2])) * v36_data);
              v159_acc += ((static_cast<float>(v161_data[3])) * v37_data);
              v159_acc += ((static_cast<float>(v161_data[4])) * v38_data);
              v159_acc += ((static_cast<float>(v161_data[5])) * v39_data);
              v159_acc += ((static_cast<float>(v161_data[6])) * v40_data);
              v159_acc += ((static_cast<float>(v161_data[7])) * v41_data);
              ir2.template select<16, 1>(16) = v159_acc;
              tensorforge::intel_esimd::simd<float, 16> v178_acc{};
              tensorforge::intel_esimd::simd<float, 16> v180_data = tensorforge::slmLoad<float, 16>(s2 + (16_i32));
              v178_acc += ((static_cast<float>(v180_data[0])) * v34_data);
              v178_acc += ((static_cast<float>(v180_data[1])) * v35_data);
              v178_acc += ((static_cast<float>(v180_data[2])) * v36_data);
              v178_acc += ((static_cast<float>(v180_data[3])) * v37_data);
              v178_acc += ((static_cast<float>(v180_data[4])) * v38_data);
              v178_acc += ((static_cast<float>(v180_data[5])) * v39_data);
              v178_acc += ((static_cast<float>(v180_data[6])) * v40_data);
              v178_acc += ((static_cast<float>(v180_data[7])) * v41_data);
              ir2.template select<16, 1>(32) = v178_acc;
              tensorforge::intel_esimd::simd<float, 16> v197_acc{};
              tensorforge::intel_esimd::simd<float, 16> v199_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v197_acc += ((static_cast<float>(v199_data[0])) * v34_data);
              v197_acc += ((static_cast<float>(v199_data[1])) * v35_data);
              v197_acc += ((static_cast<float>(v199_data[2])) * v36_data);
              v197_acc += ((static_cast<float>(v199_data[3])) * v37_data);
              v197_acc += ((static_cast<float>(v199_data[4])) * v38_data);
              v197_acc += ((static_cast<float>(v199_data[5])) * v39_data);
              v197_acc += ((static_cast<float>(v199_data[6])) * v40_data);
              v197_acc += ((static_cast<float>(v199_data[7])) * v41_data);
              ir2.template select<16, 1>(48) = v197_acc;
              #pragma unroll
              for (int32_t v216_n1 = 0; v216_n1 < 4; ++v216_n1) {
                int32_t v217_a = v216_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v219_data(ir2.template select<8, 1>(v217_a));
                r2.template select<8, 1>(v217_a) = v219_data;
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v220_i1 = 0; v220_i1 < 4; ++v220_i1) {
                tensorforge::intel_esimd::simd<float, 8> v223_data(r2.template select<8, 1>((v220_i1 * 16)));
                tensorforge::slmStore<float, 8>(s1 + (((v220_i1 + 4) * 8)), v223_data);
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v229_k1 = 0; v229_k1 < 8; ++v229_k1) {
                int32_t v232_a = v229_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v234_data = tensorforge::slmLoad<float, 8>(s1 + (v232_a));
                (tensorforge::intel_esimd::abs(v234_data)).copy_to(glb_m3 + (v232_a));
              }
            }
            tensorforge::prefetchRunsL2<256, 128, 128>(&pf_glb_m0[0], &pf_glb_m1[0], &pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


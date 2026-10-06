// === base name ===
kernel_062268710e3736b5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_062268710e3736b5 = {{1, 128, 1}, 2, 2, 1, 128, 12288, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_062268710e3736b5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_062268710e3736b5(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_062268710e3736b5(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 128, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 128 - 1) / 128;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 128;
  config.block[2] = 1;
  config.sharedMemBytes = 768 * sizeof(__float128);
  config.cooperative = false;
  return config;
}
void launcher_kernel_062268710e3736b5(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_062268710e3736b5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_062268710e3736b5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_062268710e3736b5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, __float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<768 * sizeof(__float128)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 2 lanes x 128 per block = block 1x128x1, 12288 B shared, occupancy grid
        // operands:
        //   m0 2×2(2×2) {0..2}×{0..2} strided
        //   m1 2×2(2×2) {0..2}×{0..2} strided
        //   m2 2×2(2×2) {0..2}×{0..2} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"__float128","launch":{"active_threads":2,"block":[1,128,1],"cooperative":false,"lead_width":1,"mults_per_block":128,"persistent":true,"sections":[{"barrier":false,"mults_per_block":128,"shared_elements":768}],"shared_bytes":12288,"shared_elements":768,"threads_per_mult":2},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[2,2]],"name":"m0","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[2,2]],"name":"m1","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[2,2]],"name":"m2","ordered":false,"parts":1,"shape":[2,2],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[2,2]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[2,2]},{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[2,2]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<__float128> totalShrMem = tensorforge::SlmPtr<__float128>(0);
          tensorforge::SlmPtr<__float128> localShrMem0 = totalShrMem + (6 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<__float128> tempShrMem = localShrMem0 + (4);
          tensorforge::SlmPtr<__float128> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              __float128 *const __restrict__ glb_m0 = &m0[v11_batchId0 * 4 + 0 + m0_extraOffset];
              const __float128 *const __restrict__ glb_m1 = &m1[v11_batchId0 * 4 + 0 + m1_extraOffset];
              const __float128 *const __restrict__ glb_m2 = &m2[v11_batchId0 * 4 + 0 + m2_extraOffset];
              __float128 r0[4]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                int32_t v25_lead = v23_i0 * 2;
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 2; ++v24_i1) {
                  int32_t v28_a = v25_lead + (v24_i1 * 2);
                  tensorforge::intel_esimd::simd<__float128, 2> v29_data;
                  v29_data.copy_from(glb_m1 + (v28_a));
                  v29_data.copy_to(r0 + (v28_a));
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<__float128, 2> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<__float128, 2>(s0 + (0 + 0 + 1 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<__float128, 2> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 2));
              tensorforge::slmStore<__float128, 2>(s0 + (0 + 0 + 1 * 0 + 2), v32_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              __float128 r1[4]{};
              // ir1 = +(r0 * s0)
              // [(0, 2), (0, 2)] [(0, 2)]
              __float128 ir1[4]{};
              tensorforge::intel_esimd::simd<__float128, 2> v35_data;
              v35_data.copy_from(r0 + (0));
              __float128 v36_data = s0[0];
              tensorforge::intel_esimd::simd<__float128, 2> v38_data;
              v38_data.copy_from(ir1 + (0));
              (v38_data + (v35_data * v36_data)).copy_to(ir1 + (0));
              __float128 v41_data = s0[2];
              tensorforge::intel_esimd::simd<__float128, 2> v43_data;
              v43_data.copy_from(ir1 + (2));
              (v43_data + (v35_data * v41_data)).copy_to(ir1 + (2));
              tensorforge::intel_esimd::simd<__float128, 2> v45_data;
              v45_data.copy_from(r0 + (2));
              __float128 v46_data = s0[1];
              tensorforge::intel_esimd::simd<__float128, 2> v48_data;
              v48_data.copy_from(ir1 + (0));
              (v48_data + (v45_data * v46_data)).copy_to(ir1 + (0));
              __float128 v51_data = s0[3];
              tensorforge::intel_esimd::simd<__float128, 2> v53_data;
              v53_data.copy_from(ir1 + (2));
              (v53_data + (v45_data * v51_data)).copy_to(ir1 + (2));
              // r1 = ir1
              #pragma unroll
              for (int32_t v55_n0 = 0; v55_n0 < 1; ++v55_n0) {
                int32_t v57_a = v55_n0 * 2;
                #pragma unroll
                for (int32_t v56_n1 = 0; v56_n1 < 2; ++v56_n1) {
                  int32_t v59_a = v57_a + (v56_n1 * 2);
                  tensorforge::intel_esimd::simd<__float128, 2> v60_data;
                  v60_data.copy_from(ir1 + (v59_a));
                  v60_data.copy_to(r1 + (v59_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v61_i0 = 0; v61_i0 < 1; ++v61_i0) {
                int32_t v63_a = v61_i0 * 2;
                #pragma unroll
                for (int32_t v62_i1 = 0; v62_i1 < 2; ++v62_i1) {
                  int32_t v65_a = v63_a + (v62_i1 * 2);
                  tensorforge::intel_esimd::simd<__float128, 2> v66_data;
                  v66_data.copy_from(r1 + (v65_a));
                  v66_data.copy_to(glb_m0 + (v65_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


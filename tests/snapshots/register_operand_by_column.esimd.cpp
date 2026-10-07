// === base name ===
kernel_40bc25900c936516

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_40bc25900c936516 = {{1, 8, 1}, 32, 21, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_40bc25900c936516(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_40bc25900c936516(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_40bc25900c936516(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 8 - 1) / 8;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 704 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_40bc25900c936516(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_40bc25900c936516(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_40bc25900c936516(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_40bc25900c936516(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<704 * sizeof(double)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (21 active) x 8 per block = block 1x8x1, 5632 B shared, occupancy grid
        // operands:
        //   m0 32×3(32×3) {0..32}×{0..3} pointer_based
        //   m1 32×3(32×3) {0..32}×{0..3} pointer_based
        //   m2 9×9(9×9) {0..9}×{0..9} pointer_based
        // operations:
        //   m0[i,j]@{0..21}×{0..3} = m1[i,k]@{0..21}×{0..3} × m2[j,k]@{6..9}×{6..9}
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":21,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":704}],"shared_bytes":5632,"shared_elements":704,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,3]],"name":"m0","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"M0","bbox":[[0,0],[32,3]],"name":"m1","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"T","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[21,3]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[21,3]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,3]},{"addressing":"pointer_based","bbox":[[0,0],[3,3]],"is_tmp":false,"name":"m2","offset":[6,6],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (88 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v8_batchId0][0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v8_batchId0][0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v8_batchId0][0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 96> r0(0.0);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i1 = 0; v20_i1 < 3; ++v20_i1) {
                int32_t v23_a = v20_i1 * 32;
                tensorforge::intel_esimd::simd<double, 21> v25_data;
                v25_data.copy_from(glb_m1 + (v23_a));
                r0.template select<21, 1>(v23_a) = v25_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<double, 64> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 64>(s0 + (0 + 0 + 2 * 0 + 0), v27_ld);
              tensorforge::intel_esimd::simd<double, 17> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<double, 17>(s0 + (0 + 0 + 1 * 0 + 64), v28_ld);
              tensorforge::intel_esimd::simd<double, 96> r1(0.0);
              // ir1 = +(r0 * s0)
              // [(0, 21), (0, 3)] [(0, 3)]
              tensorforge::intel_esimd::simd<double, 96> ir1(0.0);
              tensorforge::intel_esimd::simd<double, 32> v31_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<double, 28> s0_w0 = tensorforge::slmLoad<double, 28>(s0 + 60);
              double v32_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 32> v34_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v34_data + (v31_data * v32_data));
              double v37_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 32> v39_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v39_data + (v31_data * v37_data));
              double v42_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 32> v44_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v44_data + (v31_data * v42_data));
              tensorforge::intel_esimd::simd<double, 32> v46_data(r0.template select<32, 1>(32));
              double v47_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 32> v49_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v49_data + (v46_data * v47_data));
              double v52_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 32> v54_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v54_data + (v46_data * v52_data));
              double v57_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 32> v59_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v59_data + (v46_data * v57_data));
              tensorforge::intel_esimd::simd<double, 32> v61_data(r0.template select<32, 1>(64));
              double v62_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 32> v64_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v64_data + (v61_data * v62_data));
              double v67_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 32> v69_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v69_data + (v61_data * v67_data));
              double v72_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 32> v74_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v74_data + (v61_data * v72_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v76_n1 = 0; v76_n1 < 3; ++v76_n1) {
                int32_t v77_a = v76_n1 * 32;
                tensorforge::intel_esimd::simd<double, 21> v79_data(ir1.template select<21, 1>(v77_a));
                r1.template select<21, 1>(v77_a) = v79_data;
              }
              // glb_m0 = store{r>g}(r1);
              tensorforge::intel_esimd::simd_mask<32> v85_g = (tensorforge::intel_esimd::simd<int32_t, 32>(0, 1)) < 21;
              #pragma unroll
              for (int32_t v80_i1 = 0; v80_i1 < 3; ++v80_i1) {
                int32_t v81_a = v80_i1 * 32;
                tensorforge::intel_esimd::simd<double, 32> v83_data(r1.template select<32, 1>(v81_a));
                tensorforge::intel_esimd::simd<double, 32> v86_pad(0.0);
                v86_pad.merge(tensorforge::intel_esimd::simd<double, 32>(v83_data), v85_g);
                v86_pad.copy_to(glb_m0 + (v81_a));
              }
            }
          }
        }
      }
    });
  });
}


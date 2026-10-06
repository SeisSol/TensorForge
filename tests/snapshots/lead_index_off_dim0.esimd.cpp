// === base name ===
kernel_e184c6f26cb4cd43

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e184c6f26cb4cd43 = {{1, 8, 1}, 32, 20, 1, 8, 512, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e184c6f26cb4cd43(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e184c6f26cb4cd43(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e184c6f26cb4cd43(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 128 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e184c6f26cb4cd43(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e184c6f26cb4cd43(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e184c6f26cb4cd43(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e184c6f26cb4cd43(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<128 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (20 active) x 8 per block = block 1x8x1, 512 B shared, occupancy grid
        // operands:
        //   m0 20×9(20×9) {0..20}×{0..9} strided
        //   m1 1×20(1×20) {0..1}×{0..20} strided
        //   m2 1×9(1×9) {0..1}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[k,i] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":128}],"shared_bytes":512,"shared_elements":128,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[1,20]],"name":"m1","ordered":false,"parts":1,"shape":[1,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[1,9]],"name":"m2","ordered":false,"parts":1,"shape":[1,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[1,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[1,20]},{"addressing":"strided","bbox":[[0,0],[1,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[1,9]}],"permute":[[0,1],[0,1]],"target":[[-1,0],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (16 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (16);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 180 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 20 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 9 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 32> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                tensorforge::intel_esimd::simd<float, 20> v27_data;
                v27_data.copy_from(glb_m1 + (v23_i0));
                r0.template select<20, 1>(v23_i0) = v27_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 9> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 0), v29_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 288> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 20), (0, 9)] [(0, 1)]
              tensorforge::intel_esimd::simd<float, 288> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v32_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> s0_w0 = tensorforge::slmLoad<float, 16>(s0 + 0);
              float v33_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v35_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v35_data + (v32_data * v33_data));
              float v38_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v40_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v40_data + (v32_data * v38_data));
              float v43_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v45_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v45_data + (v32_data * v43_data));
              float v48_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v50_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v50_data + (v32_data * v48_data));
              float v53_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v55_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v55_data + (v32_data * v53_data));
              float v58_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v60_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v60_data + (v32_data * v58_data));
              float v63_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v65_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v65_data + (v32_data * v63_data));
              float v68_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v70_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v70_data + (v32_data * v68_data));
              float v73_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v75_data(ir1.template select<32, 1>(256));
              ir1.template select<32, 1>(256) = (v75_data + (v32_data * v73_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v77_n1 = 0; v77_n1 < 9; ++v77_n1) {
                int32_t v78_a = v77_n1 * 32;
                tensorforge::intel_esimd::simd<float, 20> v80_data(ir1.template select<20, 1>(v78_a));
                r1.template select<20, 1>(v78_a) = v80_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v81_i1 = 0; v81_i1 < 9; ++v81_i1) {
                tensorforge::intel_esimd::simd<float, 20> v84_data(r1.template select<20, 1>((v81_i1 * 32)));
                v84_data.copy_to(glb_m0 + ((v81_i1 * 20)));
              }
            }
          }
        }
      }
    });
  });
}


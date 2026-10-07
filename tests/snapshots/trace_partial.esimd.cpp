// === base name ===
kernel_07420e126385d082

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_07420e126385d082 = {{1, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_07420e126385d082(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_07420e126385d082(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_07420e126385d082(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_07420e126385d082(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_07420e126385d082(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_07420e126385d082(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_07420e126385d082(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<256 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16(16) {0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i] = m1[i,k]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"OUT","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (16 * item.get_local_id(1) + 0);
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 256 + 0 + m1_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v20_lead = v18_i0 * 16;
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
                  int32_t v23_a = v20_lead + (v19_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v24_data;
                  v24_data.copy_from(glb_m1 + (v23_a));
                  r0.template select<16, 1>(v23_a) = v24_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 16> r1(0.0f);
              // ir1 = +(r0)
              // [(0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 16> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v28_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v29_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v29_data + v28_data);
              tensorforge::intel_esimd::simd<float, 16> v31_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v32_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v32_data + v31_data);
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v35_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v35_data + v34_data);
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v38_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v38_data + v37_data);
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v41_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v41_data + v40_data);
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v44_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v44_data + v43_data);
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v47_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v47_data + v46_data);
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v50_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v50_data + v49_data);
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v53_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v53_data + v52_data);
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v56_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v56_data + v55_data);
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v59_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v59_data + v58_data);
              tensorforge::intel_esimd::simd<float, 16> v61_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v62_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v62_data + v61_data);
              tensorforge::intel_esimd::simd<float, 16> v64_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v65_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v65_data + v64_data);
              tensorforge::intel_esimd::simd<float, 16> v67_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v68_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v68_data + v67_data);
              tensorforge::intel_esimd::simd<float, 16> v70_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v71_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v71_data + v70_data);
              tensorforge::intel_esimd::simd<float, 16> v73_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v74_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v74_data + v73_data);
              // r1 = ir1
              #pragma unroll
              for (int32_t v76_n0 = 0; v76_n0 < 1; ++v76_n0) {
                int32_t v77_a = v76_n0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v78_data(ir1.template select<16, 1>(v77_a));
                r1.template select<16, 1>(v77_a) = v78_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v79_i0 = 0; v79_i0 < 1; ++v79_i0) {
                int32_t v80_a = v79_i0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v81_data(r1.template select<16, 1>(v80_a));
                v81_data.copy_to(glb_m0 + (v80_a));
              }
            }
          }
        }
      }
    });
  });
}


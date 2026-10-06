// === base name ===
kernel_208513480c71a751

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_208513480c71a751 = {{1, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_208513480c71a751(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_208513480c71a751(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_208513480c71a751(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_208513480c71a751(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_208513480c71a751(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_208513480c71a751(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_208513480c71a751(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<256 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 24×6(24×6) {0..24}×{0..6} strided
        //   m1 6(6) {0..6} strided
        // operations:
        //   TMP = +(A, dims=[0])
        //   m1[i] = t0[i]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[24,6]],"name":"m0","ordered":false,"parts":1,"shape":[24,6],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[6]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,6]}],"permute":[[0,1]],"target":[[-1,0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[6]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (16 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (0);
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 144 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 6 + 0 + m1_extraOffset];
              float r0[16]{};
              // r0 = +(glb_m0, dims=[0])
              tensorforge::intel_esimd::simd_mask<16> v28_own = (tensorforge::intel_esimd::simd<int32_t, 16>(0, 1)) < 8;
              #pragma unroll
              for (int32_t v21_k1 = 0; v21_k1 < 6; ++v21_k1) {
                int32_t v25_a = v21_k1 * 24;
                tensorforge::intel_esimd::simd<float, 16> v27_data;
                v27_data.copy_from(glb_m0 + (v25_a));
                tensorforge::intel_esimd::simd<float, 16> v32_data(0.0f);
                tensorforge::intel_esimd::simd<float, 8> v32_data_part;
                v32_data_part.copy_from(glb_m0 + ((16_i32 + v25_a)));
                v32_data.template select<8, 1>(0) = v32_data_part;
                tensorforge::intel_esimd::simd<float, 16> v34_own(0.0f);
                v34_own.merge(tensorforge::intel_esimd::simd<float, 16>(v32_data), v28_own);
                r0[v21_k1] = (tensorforge::intel_esimd::reduce<float>((v27_data + v34_own), std::plus<>()));
              }
              float r1[16]{};
              // ir1 = +(r0)
              // [(0, 6)] []
              float ir1[16]{};
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(ir1 + (0));
              (v40_data + v39_data).copy_to(ir1 + (0));
              // r1 = ir1
              tensorforge::intel_esimd::simd<float, 6> v42_data;
              v42_data.copy_from(ir1 + (0));
              v42_data.copy_to(r1 + (0));
              // glb_m1 = store{r>g}(r1);
              tensorforge::intel_esimd::simd<float, 6> v43_data;
              v43_data.copy_from(r1 + (0));
              v43_data.copy_to(glb_m1 + (0_i32));
            }
          }
        }
      }
    });
  });
}


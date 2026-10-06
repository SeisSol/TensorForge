// === base name ===
kernel_d1a23fce592db4a4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d1a23fce592db4a4 = {{1, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d1a23fce592db4a4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d1a23fce592db4a4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d1a23fce592db4a4(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 512 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_d1a23fce592db4a4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d1a23fce592db4a4(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d1a23fce592db4a4(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d1a23fce592db4a4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<512 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 16(16) {0..16} strided
        //   m1 24×16(24×6) {0..24}×{0..6} strided
        //   m2 16(16) {0..16} strided
        // operations:
        //   t0[i] = m0[i]
        //   TMP = +(A, dims=[0])
        //   m2[i] = t0[i]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"X","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[24,6]],"name":"m1","ordered":false,"parts":1,"shape":[24,16],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[16]],"name":"m2","ordered":false,"parts":1,"shape":[16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,16]}],"permute":[[0,1]],"target":[[-1,0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m2","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[16]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (32 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (16);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 16 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                int32_t v24_lead = v23_i0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v26_data;
                v26_data.copy_from(glb_m0 + (v24_lead));
                v26_data.copy_to(r0 + (v24_lead));
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[16]{};
              // r1 = +(r0) + None
              // [(0, 16)] []
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r1 + (0));
              (v29_data + v28_data).copy_to(r1 + (0));
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
                int32_t v32_a = v31_i0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v33_data;
                v33_data.copy_from(r1 + (v32_a));
                tensorforge::slmStore<float, 16>(s0 + (v32_a), v33_data);
              }
              float r2[16]{};
              // r2 = +(glb_m1, dims=[0])
              tensorforge::intel_esimd::simd_mask<16> v43_own = (tensorforge::intel_esimd::simd<int32_t, 16>(0, 1)) < 8;
              #pragma unroll
              for (int32_t v36_k1 = 0; v36_k1 < 6; ++v36_k1) {
                int32_t v40_a = v36_k1 * 24;
                tensorforge::intel_esimd::simd<float, 16> v42_data;
                v42_data.copy_from(glb_m1 + (v40_a));
                tensorforge::intel_esimd::simd<float, 16> v47_data(0.0f);
                tensorforge::intel_esimd::simd<float, 8> v47_data_part;
                v47_data_part.copy_from(glb_m1 + ((16_i32 + v40_a)));
                v47_data.template select<8, 1>(0) = v47_data_part;
                tensorforge::intel_esimd::simd<float, 16> v49_own(0.0f);
                v49_own.merge(tensorforge::intel_esimd::simd<float, 16>(v47_data), v43_own);
                r2[v36_k1] = (tensorforge::intel_esimd::reduce<float>((v42_data + v49_own), std::plus<>()));
              }
              // s0 = store{r>s, clear}(localShrMem0, r2);
              s0[6_i32] = 0.0f;
              tensorforge::intel_esimd::simd<float, 6> v56_data;
              v56_data.copy_from(r2 + (0));
              tensorforge::slmStore<float, 6>(s0 + (0_i32), v56_data);
              float r3[16]{};
              // ir3 = +(s0)
              // [(0, 16)] []
              float ir3[16]{};
              tensorforge::intel_esimd::simd<float, 16> v63_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data;
              v64_data.copy_from(ir3 + (0));
              (v64_data + v63_data).copy_to(ir3 + (0));
              // r3 = ir3
              #pragma unroll
              for (int32_t v66_n0 = 0; v66_n0 < 1; ++v66_n0) {
                int32_t v67_a = v66_n0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v68_data;
                v68_data.copy_from(ir3 + (v67_a));
                v68_data.copy_to(r3 + (v67_a));
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v69_i0 = 0; v69_i0 < 1; ++v69_i0) {
                int32_t v70_a = v69_i0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v71_data;
                v71_data.copy_from(r3 + (v70_a));
                v71_data.copy_to(glb_m2 + (v70_a));
              }
            }
          }
        }
      }
    });
  });
}


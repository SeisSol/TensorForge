// === base name ===
kernel_8315a68840278ec8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8315a68840278ec8 = {{1, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8315a68840278ec8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8315a68840278ec8(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8315a68840278ec8(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_8315a68840278ec8(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8315a68840278ec8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8315a68840278ec8(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8315a68840278ec8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<256 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   TMP = +(A, dims=[1])
        //   m1[i,j] = t0[i] × m2[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[8]],"is_tmp":true,"name":"t0","offset":[0],"shape":[8]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"target":[[0,-1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0],[8]],"is_tmp":true,"name":"t0","offset":[0],"shape":[8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0],[0,1]],"target":[[0],[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (16 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (0);
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 64 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
                tensorforge::intel_esimd::simd<float, 8> v21_data;
                v21_data.copy_from(glb_m2 + ((v16_i1 * 8)));
                r1.template select<8, 1>((v16_i1 * 16)) = v21_data;
              }
              tensorforge::intel_esimd::simd<float, 16> r0(0.0f);
              // r0 = +(glb_m0, dims=[1])
              tensorforge::intel_esimd::simd<float, 8> v26_acc0(0.0f);
              #pragma unroll
              for (int32_t v25_r1 = 0; v25_r1 < 8; ++v25_r1) {
                tensorforge::intel_esimd::simd<float, 8> v31_data;
                v31_data.copy_from(glb_m0 + ((v25_r1 * 8)));
                v26_acc0 = (v26_acc0 + v31_data);
              }
              r0.template select<8, 1>(0) = v26_acc0;
              // wait(r1 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 128> r2(0.0f);
              // ir2 = +(r0 * r1)
              // [(0, 8), (0, 8)] []
              tensorforge::intel_esimd::simd<float, 128> ir2(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v39_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v39_data + (v36_data * v37_data));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v44_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v44_data + (v36_data * v42_data));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v49_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v49_data + (v36_data * v47_data));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v54_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v54_data + (v36_data * v52_data));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r1.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v59_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v59_data + (v36_data * v57_data));
              tensorforge::intel_esimd::simd<float, 16> v62_data(r1.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v64_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v64_data + (v36_data * v62_data));
              tensorforge::intel_esimd::simd<float, 16> v67_data(r1.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v69_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v69_data + (v36_data * v67_data));
              tensorforge::intel_esimd::simd<float, 16> v72_data(r1.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v74_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v74_data + (v36_data * v72_data));
              // r2 = ir2
              #pragma unroll
              for (int32_t v76_n1 = 0; v76_n1 < 8; ++v76_n1) {
                int32_t v77_a = v76_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v79_data(ir2.template select<8, 1>(v77_a));
                r2.template select<8, 1>(v77_a) = v79_data;
              }
              // glb_m1 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v80_i1 = 0; v80_i1 < 8; ++v80_i1) {
                tensorforge::intel_esimd::simd<float, 8> v83_data(r2.template select<8, 1>((v80_i1 * 16)));
                v83_data.copy_to(glb_m1 + ((v80_i1 * 8)));
              }
            }
          }
        }
      }
    });
  });
}


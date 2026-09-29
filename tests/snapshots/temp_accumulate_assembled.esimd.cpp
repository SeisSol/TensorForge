// === base name ===
kernel_37013ed8d897daaa

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_37013ed8d897daaa = {{1, 16, 1}, 16, 12, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_37013ed8d897daaa(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_37013ed8d897daaa(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_37013ed8d897daaa(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_37013ed8d897daaa(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_37013ed8d897daaa(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_37013ed8d897daaa(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_37013ed8d897daaa(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<512 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 6(6) {0..6} strided
        //   m1 6(6) {0..6} strided
        //   m2 12(12) {0..12} strided
        //   m3 12(12) {0..12} strided
        // operations:
        //   t0[i]@{0..6} = m0[i]
        //   t0[i]@{6..12} = m1[i]
        //   t0[i] += m2[i]
        //   m3[i] = t0[i]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"a","bbox":[[0],[6]],"name":"m0","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"strided","alias":"b","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"strided","alias":"w","bbox":[[0],[12]],"name":"m2","ordered":false,"parts":1,"shape":[12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0],[12]],"name":"m3","ordered":false,"parts":1,"shape":[12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[6]],"is_tmp":false,"name":"m0","offset":[0],"shape":[6]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[6],"shape":[12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]}],"permute":[[0]],"target":[[0]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0],[12]],"is_tmp":true,"name":"t0","offset":[0],"shape":[12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[12]],"is_tmp":false,"name":"m2","offset":[0],"shape":[12]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[12]],"is_tmp":false,"name":"m3","offset":[0],"shape":[12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0],[12]],"is_tmp":true,"name":"t0","offset":[0],"shape":[12]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (32 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (16);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 6 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 6 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 12 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v5_batchId0 * 12 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 16> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              tensorforge::intel_esimd::simd<float, 6> v20_data;
              v20_data.copy_from(glb_m0 + (0_i32));
              r0.template select<6, 1>(0) = v20_data;
              tensorforge::intel_esimd::simd<float, 16> r2(0.0f);
              // r2 = load{g>r}(glb_m1);
              tensorforge::intel_esimd::simd<float, 6> v22_data;
              v22_data.copy_from(glb_m1 + (0_i32));
              r2.template select<6, 1>(0) = v22_data;
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 16> r1(0.0f);
              // r1 = +(r0) + None
              // [(0, 6)] []
              tensorforge::intel_esimd::simd<float, 16> v24_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v25_data(r1.template select<16, 1>(0));
              r1.template select<16, 1>(0) = (v25_data + v24_data);
              // s0 = store{r>s}(localShrMem0, r1);
              tensorforge::intel_esimd::simd<float, 6> v27_data(r1.template select<6, 1>(0));
              tensorforge::slmStore<float, 6>(s0 + (0_i32), v27_data);
              tensorforge::intel_esimd::simd<float, 16> r4(0.0f);
              // r4 = load{g>r}(glb_m2);
              tensorforge::intel_esimd::simd<float, 12> v31_data;
              v31_data.copy_from(glb_m2 + (0_i32));
              r4.template select<12, 1>(0) = v31_data;
              // wait(r2 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 16> r3(0.0f);
              // ir3 = +(r2)
              // [(0, 6)] []
              tensorforge::intel_esimd::simd<float, 16> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v34_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v35_data(ir3.template select<16, 1>(0));
              ir3.template select<16, 1>(0) = (v35_data + v34_data);
              // r3 = ir3
              tensorforge::intel_esimd::simd<float, 6> v37_data(ir3.template select<6, 1>(0));
              r3.template select<6, 1>(0) = v37_data;
              // s0 = store{r>s}(localShrMem0, r3);
              tensorforge::intel_esimd::simd<float, 6> v38_data(r3.template select<6, 1>(0));
              tensorforge::slmStore<float, 6>(s0 + (6_i32), v38_data);
              // wait(r4 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 16> r5(0.0f);
              // ir5 = +(r4)
              // [(0, 12)] []
              tensorforge::intel_esimd::simd<float, 16> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v42_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v43_data(ir5.template select<16, 1>(0));
              ir5.template select<16, 1>(0) = (v43_data + v42_data);
              // r5 = ir5 + s0
              tensorforge::intel_esimd::simd<float, 12> v45_data(ir5.template select<12, 1>(0));
              tensorforge::intel_esimd::simd<float, 12> v46_data = tensorforge::slmLoad<float, 12>(s0 + (0_i32));
              r5.template select<12, 1>(0) = (v46_data + v45_data);
              // s0 = store{r>s}(localShrMem0, r5);
              tensorforge::intel_esimd::simd<float, 12> v48_data(r5.template select<12, 1>(0));
              tensorforge::slmStore<float, 12>(s0 + (0_i32), v48_data);
              tensorforge::intel_esimd::simd<float, 16> r6(0.0f);
              // ir6 = +(s0)
              // [(0, 12)] []
              tensorforge::intel_esimd::simd<float, 16> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v53_data(0.0f);
              v53_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v54_data(ir6.template select<16, 1>(0));
              ir6.template select<16, 1>(0) = (v54_data + v53_data);
              // r6 = ir6
              tensorforge::intel_esimd::simd<float, 12> v56_data(ir6.template select<12, 1>(0));
              r6.template select<12, 1>(0) = v56_data;
              // glb_m3 = store{r>g}(r6);
              tensorforge::intel_esimd::simd<float, 12> v57_data(r6.template select<12, 1>(0));
              v57_data.copy_to(glb_m3 + (0_i32));
            }
          }
        }
      }
    });
  });
}


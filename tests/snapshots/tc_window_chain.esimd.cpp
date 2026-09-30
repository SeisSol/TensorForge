// === base name ===
kernel_f8749045806e8889

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f8749045806e8889 = {{1, 32, 1}, 16, 16, 1, 32, 24576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f8749045806e8889(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f8749045806e8889(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f8749045806e8889(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 32, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 32 - 1) / 32;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 32;
  config.block[2] = 1;
  config.sharedMemBytes = 6144 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f8749045806e8889(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f8749045806e8889(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f8749045806e8889(stream, grid, block, m0, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f8749045806e8889(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<6144 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 32 per block = block 1x32x1, 24576 B shared, occupancy grid
        // operands:
        //   m0 16×20(16×17) {0..16}×{1..18} none
        //   m1 20×9(17×9) {1..18}×{0..9} strided
        //   m2 16×9(16×9) {0..16}×{0..9} strided
        //   m3 16×20(16×15) {0..16}×{1..16} none
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]@{1..16}×{0..9}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":6144}],"shared_bytes":24576,"shared_elements":6144,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"none","alias":"A1","bbox":[[0,1],[16,18]],"name":"m0","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"pointer_based","bbox":[[1,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 512);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (160);
          const float *const __restrict__ ptr_glb_m0 = &m0[0];
          tensorforge::SlmPtr<float> glb_m0 = totalShrMem + (0);
          // glb_m0 = load{g>s}(ptr_glb_m0[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v5_ld;
            v5_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v5_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v6_ld;
            v6_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v6_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v7_ld;
            v7_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v7_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v8_ld;
            v8_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v8_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v9_ld;
            v9_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v9_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v10_ld;
            v10_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v10_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v11_ld;
            v11_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v11_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v12_ld;
            v12_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v13_ld;
            v13_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v14_ld;
            v14_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 16> v15_ld;
            v15_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 16> v16_ld;
            v16_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<float, 16> v17_ld;
            v17_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<float, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<float, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 15) {
            tensorforge::intel_esimd::simd<float, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          if (item.get_local_id(1) == 16) {
            tensorforge::intel_esimd::simd<float, 16> v21_ld;
            v21_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v21_ld);
          }
          const float *const __restrict__ ptr_glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> glb_m3 = totalShrMem + (272);
          // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v24_ld;
            v24_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v24_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v25_ld;
            v25_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v25_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v26_ld;
            v26_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v26_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v27_ld;
            v27_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v27_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v28_ld;
            v28_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v28_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v29_ld;
            v29_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v29_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v30_ld;
            v30_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v30_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v31_ld;
            v31_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v31_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v32_ld;
            v32_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v32_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v33_ld;
            v33_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v33_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 16> v34_ld;
            v34_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v34_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 16> v35_ld;
            v35_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v35_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<float, 16> v36_ld;
            v36_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v36_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<float, 16> v37_ld;
            v37_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v37_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<float, 16> v38_ld;
            v38_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v38_ld);
          }
          // wait(glb_m0 = load{g>s}(ptr_glb_m0[0, 1]));
          // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v42_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v42_batchId0 < numElements0; v42_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v43_ahead1 = v42_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v45_batchId1 = (v43_ahead1 < numElements0) ? v43_ahead1 : v42_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v42_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m1 = &m1[v42_batchId0 * 153 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v42_batchId0 * 144 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v52_ld;
              v52_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v52_ld);
              tensorforge::intel_esimd::simd<float, 64> v53_ld;
              v53_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v53_ld);
              tensorforge::intel_esimd::simd<float, 16> v54_ld;
              v54_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v54_ld);
              tensorforge::intel_esimd::simd<float, 9> v55_ld;
              v55_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v55_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // r0 = +(glb_m0 * s0) + None
              // [(0, 16), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run0 = tensorforge::slmLoad<float, 64>(glb_m0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data(glb_m0_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v62_data(glb_m0_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v64_data(glb_m0_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v66_data(glb_m0_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run1 = tensorforge::slmLoad<float, 64>(glb_m0 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_data(glb_m0_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v70_data(glb_m0_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v72_data(glb_m0_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v74_data(glb_m0_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run2 = tensorforge::slmLoad<float, 64>(glb_m0 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_data(glb_m0_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v78_data(glb_m0_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v80_data(glb_m0_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v82_data(glb_m0_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run3 = tensorforge::slmLoad<float, 64>(glb_m0 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v84_data(glb_m0_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v86_data(glb_m0_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v88_data(glb_m0_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v90_data(glb_m0_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v92_data = tensorforge::slmLoad<float, 16>(glb_m0 + (256_i32));
              tensorforge::intel_esimd::simd<float, 16> v93_acc{};
              tensorforge::intel_esimd::simd<float, 16> v96_data(0.0f);
              v96_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v93_acc += ((static_cast<float>(v96_data[1])) * v60_data);
              v93_acc += ((static_cast<float>(v96_data[2])) * v62_data);
              v93_acc += ((static_cast<float>(v96_data[3])) * v64_data);
              v93_acc += ((static_cast<float>(v96_data[4])) * v66_data);
              v93_acc += ((static_cast<float>(v96_data[5])) * v68_data);
              v93_acc += ((static_cast<float>(v96_data[6])) * v70_data);
              v93_acc += ((static_cast<float>(v96_data[7])) * v72_data);
              v93_acc += ((static_cast<float>(v96_data[8])) * v74_data);
              v93_acc += ((static_cast<float>(v96_data[9])) * v76_data);
              v93_acc += ((static_cast<float>(v96_data[10])) * v78_data);
              v93_acc += ((static_cast<float>(v96_data[11])) * v80_data);
              v93_acc += ((static_cast<float>(v96_data[12])) * v82_data);
              v93_acc += ((static_cast<float>(v96_data[13])) * v84_data);
              v93_acc += ((static_cast<float>(v96_data[14])) * v86_data);
              v93_acc += ((static_cast<float>(v96_data[15])) * v88_data);
              tensorforge::intel_esimd::simd<float, 16> v132_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v93_acc += ((static_cast<float>(v132_data[0])) * v90_data);
              v93_acc += ((static_cast<float>(v132_data[1])) * v92_data);
              r0.template select<16, 1>(0) = v93_acc;
              tensorforge::intel_esimd::simd<float, 16> v137_acc{};
              tensorforge::intel_esimd::simd<float, 16> v139_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v137_acc += ((static_cast<float>(v139_data[1])) * v60_data);
              v137_acc += ((static_cast<float>(v139_data[2])) * v62_data);
              v137_acc += ((static_cast<float>(v139_data[3])) * v64_data);
              v137_acc += ((static_cast<float>(v139_data[4])) * v66_data);
              v137_acc += ((static_cast<float>(v139_data[5])) * v68_data);
              v137_acc += ((static_cast<float>(v139_data[6])) * v70_data);
              v137_acc += ((static_cast<float>(v139_data[7])) * v72_data);
              v137_acc += ((static_cast<float>(v139_data[8])) * v74_data);
              v137_acc += ((static_cast<float>(v139_data[9])) * v76_data);
              v137_acc += ((static_cast<float>(v139_data[10])) * v78_data);
              v137_acc += ((static_cast<float>(v139_data[11])) * v80_data);
              v137_acc += ((static_cast<float>(v139_data[12])) * v82_data);
              v137_acc += ((static_cast<float>(v139_data[13])) * v84_data);
              v137_acc += ((static_cast<float>(v139_data[14])) * v86_data);
              v137_acc += ((static_cast<float>(v139_data[15])) * v88_data);
              tensorforge::intel_esimd::simd<float, 16> v172_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v137_acc += ((static_cast<float>(v172_data[0])) * v90_data);
              v137_acc += ((static_cast<float>(v172_data[1])) * v92_data);
              r0.template select<16, 1>(16) = v137_acc;
              tensorforge::intel_esimd::simd<float, 16> v177_acc{};
              tensorforge::intel_esimd::simd<float, 16> v179_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v177_acc += ((static_cast<float>(v179_data[1])) * v60_data);
              v177_acc += ((static_cast<float>(v179_data[2])) * v62_data);
              v177_acc += ((static_cast<float>(v179_data[3])) * v64_data);
              v177_acc += ((static_cast<float>(v179_data[4])) * v66_data);
              v177_acc += ((static_cast<float>(v179_data[5])) * v68_data);
              v177_acc += ((static_cast<float>(v179_data[6])) * v70_data);
              v177_acc += ((static_cast<float>(v179_data[7])) * v72_data);
              v177_acc += ((static_cast<float>(v179_data[8])) * v74_data);
              v177_acc += ((static_cast<float>(v179_data[9])) * v76_data);
              v177_acc += ((static_cast<float>(v179_data[10])) * v78_data);
              v177_acc += ((static_cast<float>(v179_data[11])) * v80_data);
              v177_acc += ((static_cast<float>(v179_data[12])) * v82_data);
              v177_acc += ((static_cast<float>(v179_data[13])) * v84_data);
              v177_acc += ((static_cast<float>(v179_data[14])) * v86_data);
              v177_acc += ((static_cast<float>(v179_data[15])) * v88_data);
              tensorforge::intel_esimd::simd<float, 16> v212_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v177_acc += ((static_cast<float>(v212_data[0])) * v90_data);
              v177_acc += ((static_cast<float>(v212_data[1])) * v92_data);
              r0.template select<16, 1>(32) = v177_acc;
              tensorforge::intel_esimd::simd<float, 16> v217_acc{};
              tensorforge::intel_esimd::simd<float, 16> v219_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v217_acc += ((static_cast<float>(v219_data[1])) * v60_data);
              v217_acc += ((static_cast<float>(v219_data[2])) * v62_data);
              v217_acc += ((static_cast<float>(v219_data[3])) * v64_data);
              v217_acc += ((static_cast<float>(v219_data[4])) * v66_data);
              v217_acc += ((static_cast<float>(v219_data[5])) * v68_data);
              v217_acc += ((static_cast<float>(v219_data[6])) * v70_data);
              v217_acc += ((static_cast<float>(v219_data[7])) * v72_data);
              v217_acc += ((static_cast<float>(v219_data[8])) * v74_data);
              v217_acc += ((static_cast<float>(v219_data[9])) * v76_data);
              v217_acc += ((static_cast<float>(v219_data[10])) * v78_data);
              v217_acc += ((static_cast<float>(v219_data[11])) * v80_data);
              v217_acc += ((static_cast<float>(v219_data[12])) * v82_data);
              v217_acc += ((static_cast<float>(v219_data[13])) * v84_data);
              v217_acc += ((static_cast<float>(v219_data[14])) * v86_data);
              v217_acc += ((static_cast<float>(v219_data[15])) * v88_data);
              tensorforge::intel_esimd::simd<float, 16> v252_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v217_acc += ((static_cast<float>(v252_data[0])) * v90_data);
              v217_acc += ((static_cast<float>(v252_data[1])) * v92_data);
              r0.template select<16, 1>(48) = v217_acc;
              tensorforge::intel_esimd::simd<float, 16> v257_acc{};
              tensorforge::intel_esimd::simd<float, 16> v259_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v257_acc += ((static_cast<float>(v259_data[1])) * v60_data);
              v257_acc += ((static_cast<float>(v259_data[2])) * v62_data);
              v257_acc += ((static_cast<float>(v259_data[3])) * v64_data);
              v257_acc += ((static_cast<float>(v259_data[4])) * v66_data);
              v257_acc += ((static_cast<float>(v259_data[5])) * v68_data);
              v257_acc += ((static_cast<float>(v259_data[6])) * v70_data);
              v257_acc += ((static_cast<float>(v259_data[7])) * v72_data);
              v257_acc += ((static_cast<float>(v259_data[8])) * v74_data);
              v257_acc += ((static_cast<float>(v259_data[9])) * v76_data);
              v257_acc += ((static_cast<float>(v259_data[10])) * v78_data);
              v257_acc += ((static_cast<float>(v259_data[11])) * v80_data);
              v257_acc += ((static_cast<float>(v259_data[12])) * v82_data);
              v257_acc += ((static_cast<float>(v259_data[13])) * v84_data);
              v257_acc += ((static_cast<float>(v259_data[14])) * v86_data);
              v257_acc += ((static_cast<float>(v259_data[15])) * v88_data);
              tensorforge::intel_esimd::simd<float, 16> v292_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v257_acc += ((static_cast<float>(v292_data[0])) * v90_data);
              v257_acc += ((static_cast<float>(v292_data[1])) * v92_data);
              r0.template select<16, 1>(64) = v257_acc;
              tensorforge::intel_esimd::simd<float, 16> v297_acc{};
              tensorforge::intel_esimd::simd<float, 16> v299_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v297_acc += ((static_cast<float>(v299_data[1])) * v60_data);
              v297_acc += ((static_cast<float>(v299_data[2])) * v62_data);
              v297_acc += ((static_cast<float>(v299_data[3])) * v64_data);
              v297_acc += ((static_cast<float>(v299_data[4])) * v66_data);
              v297_acc += ((static_cast<float>(v299_data[5])) * v68_data);
              v297_acc += ((static_cast<float>(v299_data[6])) * v70_data);
              v297_acc += ((static_cast<float>(v299_data[7])) * v72_data);
              v297_acc += ((static_cast<float>(v299_data[8])) * v74_data);
              v297_acc += ((static_cast<float>(v299_data[9])) * v76_data);
              v297_acc += ((static_cast<float>(v299_data[10])) * v78_data);
              v297_acc += ((static_cast<float>(v299_data[11])) * v80_data);
              v297_acc += ((static_cast<float>(v299_data[12])) * v82_data);
              v297_acc += ((static_cast<float>(v299_data[13])) * v84_data);
              v297_acc += ((static_cast<float>(v299_data[14])) * v86_data);
              v297_acc += ((static_cast<float>(v299_data[15])) * v88_data);
              tensorforge::intel_esimd::simd<float, 16> v332_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v297_acc += ((static_cast<float>(v332_data[0])) * v90_data);
              v297_acc += ((static_cast<float>(v332_data[1])) * v92_data);
              r0.template select<16, 1>(80) = v297_acc;
              tensorforge::intel_esimd::simd<float, 16> v337_acc{};
              tensorforge::intel_esimd::simd<float, 16> v339_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v337_acc += ((static_cast<float>(v339_data[1])) * v60_data);
              v337_acc += ((static_cast<float>(v339_data[2])) * v62_data);
              v337_acc += ((static_cast<float>(v339_data[3])) * v64_data);
              v337_acc += ((static_cast<float>(v339_data[4])) * v66_data);
              v337_acc += ((static_cast<float>(v339_data[5])) * v68_data);
              v337_acc += ((static_cast<float>(v339_data[6])) * v70_data);
              v337_acc += ((static_cast<float>(v339_data[7])) * v72_data);
              v337_acc += ((static_cast<float>(v339_data[8])) * v74_data);
              v337_acc += ((static_cast<float>(v339_data[9])) * v76_data);
              v337_acc += ((static_cast<float>(v339_data[10])) * v78_data);
              v337_acc += ((static_cast<float>(v339_data[11])) * v80_data);
              v337_acc += ((static_cast<float>(v339_data[12])) * v82_data);
              v337_acc += ((static_cast<float>(v339_data[13])) * v84_data);
              v337_acc += ((static_cast<float>(v339_data[14])) * v86_data);
              v337_acc += ((static_cast<float>(v339_data[15])) * v88_data);
              tensorforge::intel_esimd::simd<float, 16> v372_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v337_acc += ((static_cast<float>(v372_data[0])) * v90_data);
              v337_acc += ((static_cast<float>(v372_data[1])) * v92_data);
              r0.template select<16, 1>(96) = v337_acc;
              tensorforge::intel_esimd::simd<float, 16> v377_acc{};
              tensorforge::intel_esimd::simd<float, 16> v379_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v377_acc += ((static_cast<float>(v379_data[1])) * v60_data);
              v377_acc += ((static_cast<float>(v379_data[2])) * v62_data);
              v377_acc += ((static_cast<float>(v379_data[3])) * v64_data);
              v377_acc += ((static_cast<float>(v379_data[4])) * v66_data);
              v377_acc += ((static_cast<float>(v379_data[5])) * v68_data);
              v377_acc += ((static_cast<float>(v379_data[6])) * v70_data);
              v377_acc += ((static_cast<float>(v379_data[7])) * v72_data);
              v377_acc += ((static_cast<float>(v379_data[8])) * v74_data);
              v377_acc += ((static_cast<float>(v379_data[9])) * v76_data);
              v377_acc += ((static_cast<float>(v379_data[10])) * v78_data);
              v377_acc += ((static_cast<float>(v379_data[11])) * v80_data);
              v377_acc += ((static_cast<float>(v379_data[12])) * v82_data);
              v377_acc += ((static_cast<float>(v379_data[13])) * v84_data);
              v377_acc += ((static_cast<float>(v379_data[14])) * v86_data);
              v377_acc += ((static_cast<float>(v379_data[15])) * v88_data);
              tensorforge::intel_esimd::simd<float, 16> v412_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v377_acc += ((static_cast<float>(v412_data[0])) * v90_data);
              v377_acc += ((static_cast<float>(v412_data[1])) * v92_data);
              r0.template select<16, 1>(112) = v377_acc;
              tensorforge::intel_esimd::simd<float, 16> v417_acc{};
              tensorforge::intel_esimd::simd<float, 16> v419_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v417_acc += ((static_cast<float>(v419_data[1])) * v60_data);
              v417_acc += ((static_cast<float>(v419_data[2])) * v62_data);
              v417_acc += ((static_cast<float>(v419_data[3])) * v64_data);
              v417_acc += ((static_cast<float>(v419_data[4])) * v66_data);
              v417_acc += ((static_cast<float>(v419_data[5])) * v68_data);
              v417_acc += ((static_cast<float>(v419_data[6])) * v70_data);
              v417_acc += ((static_cast<float>(v419_data[7])) * v72_data);
              v417_acc += ((static_cast<float>(v419_data[8])) * v74_data);
              v417_acc += ((static_cast<float>(v419_data[9])) * v76_data);
              v417_acc += ((static_cast<float>(v419_data[10])) * v78_data);
              v417_acc += ((static_cast<float>(v419_data[11])) * v80_data);
              v417_acc += ((static_cast<float>(v419_data[12])) * v82_data);
              v417_acc += ((static_cast<float>(v419_data[13])) * v84_data);
              v417_acc += ((static_cast<float>(v419_data[14])) * v86_data);
              v417_acc += ((static_cast<float>(v419_data[15])) * v88_data);
              tensorforge::intel_esimd::simd<float, 16> v452_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v417_acc += ((static_cast<float>(v452_data[0])) * v90_data);
              v417_acc += ((static_cast<float>(v452_data[1])) * v92_data);
              r0.template select<16, 1>(128) = v417_acc;
              // s1 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v457_i0 = 0; v457_i0 < 1; ++v457_i0) {
                int32_t v459_a = v457_i0 * 16;
                #pragma unroll
                for (int32_t v458_i1 = 0; v458_i1 < 9; ++v458_i1) {
                  int32_t v461_a = v459_a + (v458_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v462_data(r0.template select<16, 1>(v461_a));
                  tensorforge::slmStore<float, 16>(s1 + (v461_a), v462_data);
                }
              }
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 16), (0, 9)] [(1, 16)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run4 = tensorforge::slmLoad<float, 64>(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v470_data(glb_m3_run4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v472_data(glb_m3_run4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v474_data(glb_m3_run4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v476_data(glb_m3_run4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run5 = tensorforge::slmLoad<float, 64>(glb_m3 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v478_data(glb_m3_run5.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v480_data(glb_m3_run5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v482_data(glb_m3_run5.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v484_data(glb_m3_run5.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run6 = tensorforge::slmLoad<float, 64>(glb_m3 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v486_data(glb_m3_run6.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v488_data(glb_m3_run6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v490_data(glb_m3_run6.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v492_data(glb_m3_run6.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 48> glb_m3_run7 = tensorforge::slmLoad<float, 48>(glb_m3 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v494_data(glb_m3_run7.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v496_data(glb_m3_run7.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v498_data(glb_m3_run7.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v499_acc{};
              tensorforge::intel_esimd::simd<float, 16> v500_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v499_acc += ((static_cast<float>(v500_data[1])) * v470_data);
              v499_acc += ((static_cast<float>(v500_data[2])) * v472_data);
              v499_acc += ((static_cast<float>(v500_data[3])) * v474_data);
              v499_acc += ((static_cast<float>(v500_data[4])) * v476_data);
              v499_acc += ((static_cast<float>(v500_data[5])) * v478_data);
              v499_acc += ((static_cast<float>(v500_data[6])) * v480_data);
              v499_acc += ((static_cast<float>(v500_data[7])) * v482_data);
              v499_acc += ((static_cast<float>(v500_data[8])) * v484_data);
              v499_acc += ((static_cast<float>(v500_data[9])) * v486_data);
              v499_acc += ((static_cast<float>(v500_data[10])) * v488_data);
              v499_acc += ((static_cast<float>(v500_data[11])) * v490_data);
              v499_acc += ((static_cast<float>(v500_data[12])) * v492_data);
              v499_acc += ((static_cast<float>(v500_data[13])) * v494_data);
              v499_acc += ((static_cast<float>(v500_data[14])) * v496_data);
              v499_acc += ((static_cast<float>(v500_data[15])) * v498_data);
              ir1.template select<16, 1>(0) = v499_acc;
              tensorforge::intel_esimd::simd<float, 16> v532_acc{};
              tensorforge::intel_esimd::simd<float, 16> v533_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v532_acc += ((static_cast<float>(v533_data[1])) * v470_data);
              v532_acc += ((static_cast<float>(v533_data[2])) * v472_data);
              v532_acc += ((static_cast<float>(v533_data[3])) * v474_data);
              v532_acc += ((static_cast<float>(v533_data[4])) * v476_data);
              v532_acc += ((static_cast<float>(v533_data[5])) * v478_data);
              v532_acc += ((static_cast<float>(v533_data[6])) * v480_data);
              v532_acc += ((static_cast<float>(v533_data[7])) * v482_data);
              v532_acc += ((static_cast<float>(v533_data[8])) * v484_data);
              v532_acc += ((static_cast<float>(v533_data[9])) * v486_data);
              v532_acc += ((static_cast<float>(v533_data[10])) * v488_data);
              v532_acc += ((static_cast<float>(v533_data[11])) * v490_data);
              v532_acc += ((static_cast<float>(v533_data[12])) * v492_data);
              v532_acc += ((static_cast<float>(v533_data[13])) * v494_data);
              v532_acc += ((static_cast<float>(v533_data[14])) * v496_data);
              v532_acc += ((static_cast<float>(v533_data[15])) * v498_data);
              ir1.template select<16, 1>(16) = v532_acc;
              tensorforge::intel_esimd::simd<float, 16> v565_acc{};
              tensorforge::intel_esimd::simd<float, 16> v566_data = tensorforge::slmLoad<float, 16>(s1 + (32_i32));
              v565_acc += ((static_cast<float>(v566_data[1])) * v470_data);
              v565_acc += ((static_cast<float>(v566_data[2])) * v472_data);
              v565_acc += ((static_cast<float>(v566_data[3])) * v474_data);
              v565_acc += ((static_cast<float>(v566_data[4])) * v476_data);
              v565_acc += ((static_cast<float>(v566_data[5])) * v478_data);
              v565_acc += ((static_cast<float>(v566_data[6])) * v480_data);
              v565_acc += ((static_cast<float>(v566_data[7])) * v482_data);
              v565_acc += ((static_cast<float>(v566_data[8])) * v484_data);
              v565_acc += ((static_cast<float>(v566_data[9])) * v486_data);
              v565_acc += ((static_cast<float>(v566_data[10])) * v488_data);
              v565_acc += ((static_cast<float>(v566_data[11])) * v490_data);
              v565_acc += ((static_cast<float>(v566_data[12])) * v492_data);
              v565_acc += ((static_cast<float>(v566_data[13])) * v494_data);
              v565_acc += ((static_cast<float>(v566_data[14])) * v496_data);
              v565_acc += ((static_cast<float>(v566_data[15])) * v498_data);
              ir1.template select<16, 1>(32) = v565_acc;
              tensorforge::intel_esimd::simd<float, 16> v598_acc{};
              tensorforge::intel_esimd::simd<float, 16> v599_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v598_acc += ((static_cast<float>(v599_data[1])) * v470_data);
              v598_acc += ((static_cast<float>(v599_data[2])) * v472_data);
              v598_acc += ((static_cast<float>(v599_data[3])) * v474_data);
              v598_acc += ((static_cast<float>(v599_data[4])) * v476_data);
              v598_acc += ((static_cast<float>(v599_data[5])) * v478_data);
              v598_acc += ((static_cast<float>(v599_data[6])) * v480_data);
              v598_acc += ((static_cast<float>(v599_data[7])) * v482_data);
              v598_acc += ((static_cast<float>(v599_data[8])) * v484_data);
              v598_acc += ((static_cast<float>(v599_data[9])) * v486_data);
              v598_acc += ((static_cast<float>(v599_data[10])) * v488_data);
              v598_acc += ((static_cast<float>(v599_data[11])) * v490_data);
              v598_acc += ((static_cast<float>(v599_data[12])) * v492_data);
              v598_acc += ((static_cast<float>(v599_data[13])) * v494_data);
              v598_acc += ((static_cast<float>(v599_data[14])) * v496_data);
              v598_acc += ((static_cast<float>(v599_data[15])) * v498_data);
              ir1.template select<16, 1>(48) = v598_acc;
              tensorforge::intel_esimd::simd<float, 16> v631_acc{};
              tensorforge::intel_esimd::simd<float, 16> v632_data = tensorforge::slmLoad<float, 16>(s1 + (64_i32));
              v631_acc += ((static_cast<float>(v632_data[1])) * v470_data);
              v631_acc += ((static_cast<float>(v632_data[2])) * v472_data);
              v631_acc += ((static_cast<float>(v632_data[3])) * v474_data);
              v631_acc += ((static_cast<float>(v632_data[4])) * v476_data);
              v631_acc += ((static_cast<float>(v632_data[5])) * v478_data);
              v631_acc += ((static_cast<float>(v632_data[6])) * v480_data);
              v631_acc += ((static_cast<float>(v632_data[7])) * v482_data);
              v631_acc += ((static_cast<float>(v632_data[8])) * v484_data);
              v631_acc += ((static_cast<float>(v632_data[9])) * v486_data);
              v631_acc += ((static_cast<float>(v632_data[10])) * v488_data);
              v631_acc += ((static_cast<float>(v632_data[11])) * v490_data);
              v631_acc += ((static_cast<float>(v632_data[12])) * v492_data);
              v631_acc += ((static_cast<float>(v632_data[13])) * v494_data);
              v631_acc += ((static_cast<float>(v632_data[14])) * v496_data);
              v631_acc += ((static_cast<float>(v632_data[15])) * v498_data);
              ir1.template select<16, 1>(64) = v631_acc;
              tensorforge::intel_esimd::simd<float, 16> v664_acc{};
              tensorforge::intel_esimd::simd<float, 16> v665_data = tensorforge::slmLoad<float, 16>(s1 + (80_i32));
              v664_acc += ((static_cast<float>(v665_data[1])) * v470_data);
              v664_acc += ((static_cast<float>(v665_data[2])) * v472_data);
              v664_acc += ((static_cast<float>(v665_data[3])) * v474_data);
              v664_acc += ((static_cast<float>(v665_data[4])) * v476_data);
              v664_acc += ((static_cast<float>(v665_data[5])) * v478_data);
              v664_acc += ((static_cast<float>(v665_data[6])) * v480_data);
              v664_acc += ((static_cast<float>(v665_data[7])) * v482_data);
              v664_acc += ((static_cast<float>(v665_data[8])) * v484_data);
              v664_acc += ((static_cast<float>(v665_data[9])) * v486_data);
              v664_acc += ((static_cast<float>(v665_data[10])) * v488_data);
              v664_acc += ((static_cast<float>(v665_data[11])) * v490_data);
              v664_acc += ((static_cast<float>(v665_data[12])) * v492_data);
              v664_acc += ((static_cast<float>(v665_data[13])) * v494_data);
              v664_acc += ((static_cast<float>(v665_data[14])) * v496_data);
              v664_acc += ((static_cast<float>(v665_data[15])) * v498_data);
              ir1.template select<16, 1>(80) = v664_acc;
              tensorforge::intel_esimd::simd<float, 16> v697_acc{};
              tensorforge::intel_esimd::simd<float, 16> v698_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v697_acc += ((static_cast<float>(v698_data[1])) * v470_data);
              v697_acc += ((static_cast<float>(v698_data[2])) * v472_data);
              v697_acc += ((static_cast<float>(v698_data[3])) * v474_data);
              v697_acc += ((static_cast<float>(v698_data[4])) * v476_data);
              v697_acc += ((static_cast<float>(v698_data[5])) * v478_data);
              v697_acc += ((static_cast<float>(v698_data[6])) * v480_data);
              v697_acc += ((static_cast<float>(v698_data[7])) * v482_data);
              v697_acc += ((static_cast<float>(v698_data[8])) * v484_data);
              v697_acc += ((static_cast<float>(v698_data[9])) * v486_data);
              v697_acc += ((static_cast<float>(v698_data[10])) * v488_data);
              v697_acc += ((static_cast<float>(v698_data[11])) * v490_data);
              v697_acc += ((static_cast<float>(v698_data[12])) * v492_data);
              v697_acc += ((static_cast<float>(v698_data[13])) * v494_data);
              v697_acc += ((static_cast<float>(v698_data[14])) * v496_data);
              v697_acc += ((static_cast<float>(v698_data[15])) * v498_data);
              ir1.template select<16, 1>(96) = v697_acc;
              tensorforge::intel_esimd::simd<float, 16> v730_acc{};
              tensorforge::intel_esimd::simd<float, 16> v731_data = tensorforge::slmLoad<float, 16>(s1 + (112_i32));
              v730_acc += ((static_cast<float>(v731_data[1])) * v470_data);
              v730_acc += ((static_cast<float>(v731_data[2])) * v472_data);
              v730_acc += ((static_cast<float>(v731_data[3])) * v474_data);
              v730_acc += ((static_cast<float>(v731_data[4])) * v476_data);
              v730_acc += ((static_cast<float>(v731_data[5])) * v478_data);
              v730_acc += ((static_cast<float>(v731_data[6])) * v480_data);
              v730_acc += ((static_cast<float>(v731_data[7])) * v482_data);
              v730_acc += ((static_cast<float>(v731_data[8])) * v484_data);
              v730_acc += ((static_cast<float>(v731_data[9])) * v486_data);
              v730_acc += ((static_cast<float>(v731_data[10])) * v488_data);
              v730_acc += ((static_cast<float>(v731_data[11])) * v490_data);
              v730_acc += ((static_cast<float>(v731_data[12])) * v492_data);
              v730_acc += ((static_cast<float>(v731_data[13])) * v494_data);
              v730_acc += ((static_cast<float>(v731_data[14])) * v496_data);
              v730_acc += ((static_cast<float>(v731_data[15])) * v498_data);
              ir1.template select<16, 1>(112) = v730_acc;
              tensorforge::intel_esimd::simd<float, 16> v763_acc{};
              tensorforge::intel_esimd::simd<float, 16> v764_data = tensorforge::slmLoad<float, 16>(s1 + (128_i32));
              v763_acc += ((static_cast<float>(v764_data[1])) * v470_data);
              v763_acc += ((static_cast<float>(v764_data[2])) * v472_data);
              v763_acc += ((static_cast<float>(v764_data[3])) * v474_data);
              v763_acc += ((static_cast<float>(v764_data[4])) * v476_data);
              v763_acc += ((static_cast<float>(v764_data[5])) * v478_data);
              v763_acc += ((static_cast<float>(v764_data[6])) * v480_data);
              v763_acc += ((static_cast<float>(v764_data[7])) * v482_data);
              v763_acc += ((static_cast<float>(v764_data[8])) * v484_data);
              v763_acc += ((static_cast<float>(v764_data[9])) * v486_data);
              v763_acc += ((static_cast<float>(v764_data[10])) * v488_data);
              v763_acc += ((static_cast<float>(v764_data[11])) * v490_data);
              v763_acc += ((static_cast<float>(v764_data[12])) * v492_data);
              v763_acc += ((static_cast<float>(v764_data[13])) * v494_data);
              v763_acc += ((static_cast<float>(v764_data[14])) * v496_data);
              v763_acc += ((static_cast<float>(v764_data[15])) * v498_data);
              ir1.template select<16, 1>(128) = v763_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v796_n0 = 0; v796_n0 < 1; ++v796_n0) {
                int32_t v798_a = v796_n0 * 16;
                #pragma unroll
                for (int32_t v797_n1 = 0; v797_n1 < 9; ++v797_n1) {
                  int32_t v800_a = v798_a + (v797_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v801_data(ir1.template select<16, 1>(v800_a));
                  r1.template select<16, 1>(v800_a) = v801_data;
                }
              }
              // glb_m2 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v802_i0 = 0; v802_i0 < 1; ++v802_i0) {
                int32_t v804_a = v802_i0 * 16;
                #pragma unroll
                for (int32_t v803_i1 = 0; v803_i1 < 9; ++v803_i1) {
                  int32_t v806_a = v804_a + (v803_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v807_data(r1.template select<16, 1>(v806_a));
                  v807_data.copy_to(glb_m2 + (v806_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


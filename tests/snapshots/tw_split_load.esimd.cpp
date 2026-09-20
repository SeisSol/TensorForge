// === base name ===
kernel_3890388aa1efb102

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3890388aa1efb102 = {{1, 32, 1}, 16, 10, 1, 32, 44416, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3890388aa1efb102(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3890388aa1efb102(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3890388aa1efb102(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 11104 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_3890388aa1efb102(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3890388aa1efb102(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3890388aa1efb102(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3890388aa1efb102(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<11104 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (10 active) x 32 per block = block 1x32x1, 44416 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 10×17(10×17) {0..10}×{0..17} none
        //   m2 17×9(17×9) {0..17}×{0..9} strided
        //   m3 10×17(10×17) {0..10}×{0..17} none
        //   m4 17×9(17×9) {0..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":11104}],"shared_bytes":44416,"shared_elements":11104,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,0],[10,17]],"name":"m1","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[17,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,0],[10,17]],"name":"m3","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[17,9]],"name":"m4","ordered":false,"parts":1,"shape":[17,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (336 * item.get_local_id(1) + 352);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (320);
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v5_ld;
            v5_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v5_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v6_ld;
            v6_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v6_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v7_ld;
            v7_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v7_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v8_ld;
            v8_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v8_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v9_ld;
            v9_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v9_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v10_ld;
            v10_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v10_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v11_ld;
            v11_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v11_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 10> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 10>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          const float *const __restrict__ ptr_glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> glb_m3 = totalShrMem + (176);
          // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v21_ld;
            v21_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v21_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v22_ld;
            v22_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v22_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v23_ld;
            v23_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v23_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v24_ld;
            v24_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v24_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v25_ld;
            v25_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v25_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v26_ld;
            v26_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v26_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v27_ld;
            v27_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v27_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 10> v28_ld;
            v28_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 10>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v28_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (160);
          for (size_t v32_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v32_batchId0 < numElements0; v32_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v33_ahead1 = v32_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v35_batchId1 = (v33_ahead1 < numElements0) ? v33_ahead1 : v32_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v32_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v32_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v32_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v32_batchId0 * 153 + 0 + m4_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v43_ld;
              v43_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v43_ld);
              tensorforge::intel_esimd::simd<float, 64> v44_ld;
              v44_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v44_ld);
              tensorforge::intel_esimd::simd<float, 16> v45_ld;
              v45_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v45_ld);
              tensorforge::intel_esimd::simd<float, 9> v46_ld;
              v46_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v46_ld);
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v47_ld;
              v47_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v47_ld);
              tensorforge::intel_esimd::simd<float, 64> v48_ld;
              v48_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 64), v48_ld);
              tensorforge::intel_esimd::simd<float, 16> v49_ld;
              v49_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s1 + (0 + 0 + 1 * 0 + 128), v49_ld);
              tensorforge::intel_esimd::simd<float, 9> v50_ld;
              v50_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s1 + (0 + 0 + 1 * 0 + 144), v50_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v56_data = tensorforge::slmLoad<float, 16>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(glb_m1 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data = tensorforge::slmLoad<float, 16>(glb_m1 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data = tensorforge::slmLoad<float, 16>(glb_m1 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data = tensorforge::slmLoad<float, 16>(glb_m1 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_data = tensorforge::slmLoad<float, 16>(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data = tensorforge::slmLoad<float, 16>(glb_m1 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_data = tensorforge::slmLoad<float, 16>(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v74_data = tensorforge::slmLoad<float, 16>(glb_m1 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_data = tensorforge::slmLoad<float, 16>(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v78_data = tensorforge::slmLoad<float, 16>(glb_m1 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data = tensorforge::slmLoad<float, 16>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v82_data = tensorforge::slmLoad<float, 16>(glb_m1 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v84_data = tensorforge::slmLoad<float, 16>(glb_m1 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v86_data = tensorforge::slmLoad<float, 16>(glb_m1 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v88_data = tensorforge::slmLoad<float, 16>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v89_acc{};
              tensorforge::intel_esimd::simd<float, 16> v90_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v89_acc += ((static_cast<float>(v90_data[0])) * v56_data);
              v89_acc += ((static_cast<float>(v90_data[1])) * v58_data);
              v89_acc += ((static_cast<float>(v90_data[2])) * v60_data);
              v89_acc += ((static_cast<float>(v90_data[3])) * v62_data);
              v89_acc += ((static_cast<float>(v90_data[4])) * v64_data);
              v89_acc += ((static_cast<float>(v90_data[5])) * v66_data);
              v89_acc += ((static_cast<float>(v90_data[6])) * v68_data);
              v89_acc += ((static_cast<float>(v90_data[7])) * v70_data);
              v89_acc += ((static_cast<float>(v90_data[8])) * v72_data);
              v89_acc += ((static_cast<float>(v90_data[9])) * v74_data);
              v89_acc += ((static_cast<float>(v90_data[10])) * v76_data);
              v89_acc += ((static_cast<float>(v90_data[11])) * v78_data);
              v89_acc += ((static_cast<float>(v90_data[12])) * v80_data);
              v89_acc += ((static_cast<float>(v90_data[13])) * v82_data);
              v89_acc += ((static_cast<float>(v90_data[14])) * v84_data);
              v89_acc += ((static_cast<float>(v90_data[15])) * v86_data);
              tensorforge::intel_esimd::simd<float, 16> v126_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v89_acc += ((static_cast<float>(v126_data[0])) * v88_data);
              ir0.template select<16, 1>(0) = v89_acc;
              tensorforge::intel_esimd::simd<float, 16> v129_acc{};
              tensorforge::intel_esimd::simd<float, 16> v131_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              v129_acc += ((static_cast<float>(v131_data[0])) * v56_data);
              v129_acc += ((static_cast<float>(v131_data[1])) * v58_data);
              v129_acc += ((static_cast<float>(v131_data[2])) * v60_data);
              v129_acc += ((static_cast<float>(v131_data[3])) * v62_data);
              v129_acc += ((static_cast<float>(v131_data[4])) * v64_data);
              v129_acc += ((static_cast<float>(v131_data[5])) * v66_data);
              v129_acc += ((static_cast<float>(v131_data[6])) * v68_data);
              v129_acc += ((static_cast<float>(v131_data[7])) * v70_data);
              v129_acc += ((static_cast<float>(v131_data[8])) * v72_data);
              v129_acc += ((static_cast<float>(v131_data[9])) * v74_data);
              v129_acc += ((static_cast<float>(v131_data[10])) * v76_data);
              v129_acc += ((static_cast<float>(v131_data[11])) * v78_data);
              v129_acc += ((static_cast<float>(v131_data[12])) * v80_data);
              v129_acc += ((static_cast<float>(v131_data[13])) * v82_data);
              v129_acc += ((static_cast<float>(v131_data[14])) * v84_data);
              v129_acc += ((static_cast<float>(v131_data[15])) * v86_data);
              tensorforge::intel_esimd::simd<float, 16> v165_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v129_acc += ((static_cast<float>(v165_data[0])) * v88_data);
              ir0.template select<16, 1>(16) = v129_acc;
              tensorforge::intel_esimd::simd<float, 16> v168_acc{};
              tensorforge::intel_esimd::simd<float, 16> v170_data = tensorforge::slmLoad<float, 16>(s0 + (34_i32));
              v168_acc += ((static_cast<float>(v170_data[0])) * v56_data);
              v168_acc += ((static_cast<float>(v170_data[1])) * v58_data);
              v168_acc += ((static_cast<float>(v170_data[2])) * v60_data);
              v168_acc += ((static_cast<float>(v170_data[3])) * v62_data);
              v168_acc += ((static_cast<float>(v170_data[4])) * v64_data);
              v168_acc += ((static_cast<float>(v170_data[5])) * v66_data);
              v168_acc += ((static_cast<float>(v170_data[6])) * v68_data);
              v168_acc += ((static_cast<float>(v170_data[7])) * v70_data);
              v168_acc += ((static_cast<float>(v170_data[8])) * v72_data);
              v168_acc += ((static_cast<float>(v170_data[9])) * v74_data);
              v168_acc += ((static_cast<float>(v170_data[10])) * v76_data);
              v168_acc += ((static_cast<float>(v170_data[11])) * v78_data);
              v168_acc += ((static_cast<float>(v170_data[12])) * v80_data);
              v168_acc += ((static_cast<float>(v170_data[13])) * v82_data);
              v168_acc += ((static_cast<float>(v170_data[14])) * v84_data);
              v168_acc += ((static_cast<float>(v170_data[15])) * v86_data);
              tensorforge::intel_esimd::simd<float, 16> v204_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v168_acc += ((static_cast<float>(v204_data[0])) * v88_data);
              ir0.template select<16, 1>(32) = v168_acc;
              tensorforge::intel_esimd::simd<float, 16> v207_acc{};
              tensorforge::intel_esimd::simd<float, 16> v209_data = tensorforge::slmLoad<float, 16>(s0 + (51_i32));
              v207_acc += ((static_cast<float>(v209_data[0])) * v56_data);
              v207_acc += ((static_cast<float>(v209_data[1])) * v58_data);
              v207_acc += ((static_cast<float>(v209_data[2])) * v60_data);
              v207_acc += ((static_cast<float>(v209_data[3])) * v62_data);
              v207_acc += ((static_cast<float>(v209_data[4])) * v64_data);
              v207_acc += ((static_cast<float>(v209_data[5])) * v66_data);
              v207_acc += ((static_cast<float>(v209_data[6])) * v68_data);
              v207_acc += ((static_cast<float>(v209_data[7])) * v70_data);
              v207_acc += ((static_cast<float>(v209_data[8])) * v72_data);
              v207_acc += ((static_cast<float>(v209_data[9])) * v74_data);
              v207_acc += ((static_cast<float>(v209_data[10])) * v76_data);
              v207_acc += ((static_cast<float>(v209_data[11])) * v78_data);
              v207_acc += ((static_cast<float>(v209_data[12])) * v80_data);
              v207_acc += ((static_cast<float>(v209_data[13])) * v82_data);
              v207_acc += ((static_cast<float>(v209_data[14])) * v84_data);
              v207_acc += ((static_cast<float>(v209_data[15])) * v86_data);
              tensorforge::intel_esimd::simd<float, 16> v243_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v207_acc += ((static_cast<float>(v243_data[0])) * v88_data);
              ir0.template select<16, 1>(48) = v207_acc;
              tensorforge::intel_esimd::simd<float, 16> v246_acc{};
              tensorforge::intel_esimd::simd<float, 16> v248_data = tensorforge::slmLoad<float, 16>(s0 + (68_i32));
              v246_acc += ((static_cast<float>(v248_data[0])) * v56_data);
              v246_acc += ((static_cast<float>(v248_data[1])) * v58_data);
              v246_acc += ((static_cast<float>(v248_data[2])) * v60_data);
              v246_acc += ((static_cast<float>(v248_data[3])) * v62_data);
              v246_acc += ((static_cast<float>(v248_data[4])) * v64_data);
              v246_acc += ((static_cast<float>(v248_data[5])) * v66_data);
              v246_acc += ((static_cast<float>(v248_data[6])) * v68_data);
              v246_acc += ((static_cast<float>(v248_data[7])) * v70_data);
              v246_acc += ((static_cast<float>(v248_data[8])) * v72_data);
              v246_acc += ((static_cast<float>(v248_data[9])) * v74_data);
              v246_acc += ((static_cast<float>(v248_data[10])) * v76_data);
              v246_acc += ((static_cast<float>(v248_data[11])) * v78_data);
              v246_acc += ((static_cast<float>(v248_data[12])) * v80_data);
              v246_acc += ((static_cast<float>(v248_data[13])) * v82_data);
              v246_acc += ((static_cast<float>(v248_data[14])) * v84_data);
              v246_acc += ((static_cast<float>(v248_data[15])) * v86_data);
              tensorforge::intel_esimd::simd<float, 16> v282_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v246_acc += ((static_cast<float>(v282_data[0])) * v88_data);
              ir0.template select<16, 1>(64) = v246_acc;
              tensorforge::intel_esimd::simd<float, 16> v285_acc{};
              tensorforge::intel_esimd::simd<float, 16> v287_data = tensorforge::slmLoad<float, 16>(s0 + (85_i32));
              v285_acc += ((static_cast<float>(v287_data[0])) * v56_data);
              v285_acc += ((static_cast<float>(v287_data[1])) * v58_data);
              v285_acc += ((static_cast<float>(v287_data[2])) * v60_data);
              v285_acc += ((static_cast<float>(v287_data[3])) * v62_data);
              v285_acc += ((static_cast<float>(v287_data[4])) * v64_data);
              v285_acc += ((static_cast<float>(v287_data[5])) * v66_data);
              v285_acc += ((static_cast<float>(v287_data[6])) * v68_data);
              v285_acc += ((static_cast<float>(v287_data[7])) * v70_data);
              v285_acc += ((static_cast<float>(v287_data[8])) * v72_data);
              v285_acc += ((static_cast<float>(v287_data[9])) * v74_data);
              v285_acc += ((static_cast<float>(v287_data[10])) * v76_data);
              v285_acc += ((static_cast<float>(v287_data[11])) * v78_data);
              v285_acc += ((static_cast<float>(v287_data[12])) * v80_data);
              v285_acc += ((static_cast<float>(v287_data[13])) * v82_data);
              v285_acc += ((static_cast<float>(v287_data[14])) * v84_data);
              v285_acc += ((static_cast<float>(v287_data[15])) * v86_data);
              tensorforge::intel_esimd::simd<float, 16> v321_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v285_acc += ((static_cast<float>(v321_data[0])) * v88_data);
              ir0.template select<16, 1>(80) = v285_acc;
              tensorforge::intel_esimd::simd<float, 16> v324_acc{};
              tensorforge::intel_esimd::simd<float, 16> v326_data = tensorforge::slmLoad<float, 16>(s0 + (102_i32));
              v324_acc += ((static_cast<float>(v326_data[0])) * v56_data);
              v324_acc += ((static_cast<float>(v326_data[1])) * v58_data);
              v324_acc += ((static_cast<float>(v326_data[2])) * v60_data);
              v324_acc += ((static_cast<float>(v326_data[3])) * v62_data);
              v324_acc += ((static_cast<float>(v326_data[4])) * v64_data);
              v324_acc += ((static_cast<float>(v326_data[5])) * v66_data);
              v324_acc += ((static_cast<float>(v326_data[6])) * v68_data);
              v324_acc += ((static_cast<float>(v326_data[7])) * v70_data);
              v324_acc += ((static_cast<float>(v326_data[8])) * v72_data);
              v324_acc += ((static_cast<float>(v326_data[9])) * v74_data);
              v324_acc += ((static_cast<float>(v326_data[10])) * v76_data);
              v324_acc += ((static_cast<float>(v326_data[11])) * v78_data);
              v324_acc += ((static_cast<float>(v326_data[12])) * v80_data);
              v324_acc += ((static_cast<float>(v326_data[13])) * v82_data);
              v324_acc += ((static_cast<float>(v326_data[14])) * v84_data);
              v324_acc += ((static_cast<float>(v326_data[15])) * v86_data);
              tensorforge::intel_esimd::simd<float, 16> v360_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v324_acc += ((static_cast<float>(v360_data[0])) * v88_data);
              ir0.template select<16, 1>(96) = v324_acc;
              tensorforge::intel_esimd::simd<float, 16> v363_acc{};
              tensorforge::intel_esimd::simd<float, 16> v365_data = tensorforge::slmLoad<float, 16>(s0 + (119_i32));
              v363_acc += ((static_cast<float>(v365_data[0])) * v56_data);
              v363_acc += ((static_cast<float>(v365_data[1])) * v58_data);
              v363_acc += ((static_cast<float>(v365_data[2])) * v60_data);
              v363_acc += ((static_cast<float>(v365_data[3])) * v62_data);
              v363_acc += ((static_cast<float>(v365_data[4])) * v64_data);
              v363_acc += ((static_cast<float>(v365_data[5])) * v66_data);
              v363_acc += ((static_cast<float>(v365_data[6])) * v68_data);
              v363_acc += ((static_cast<float>(v365_data[7])) * v70_data);
              v363_acc += ((static_cast<float>(v365_data[8])) * v72_data);
              v363_acc += ((static_cast<float>(v365_data[9])) * v74_data);
              v363_acc += ((static_cast<float>(v365_data[10])) * v76_data);
              v363_acc += ((static_cast<float>(v365_data[11])) * v78_data);
              v363_acc += ((static_cast<float>(v365_data[12])) * v80_data);
              v363_acc += ((static_cast<float>(v365_data[13])) * v82_data);
              v363_acc += ((static_cast<float>(v365_data[14])) * v84_data);
              v363_acc += ((static_cast<float>(v365_data[15])) * v86_data);
              tensorforge::intel_esimd::simd<float, 16> v399_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v363_acc += ((static_cast<float>(v399_data[0])) * v88_data);
              ir0.template select<16, 1>(112) = v363_acc;
              tensorforge::intel_esimd::simd<float, 16> v402_acc{};
              tensorforge::intel_esimd::simd<float, 16> v404_data = tensorforge::slmLoad<float, 16>(s0 + (136_i32));
              v402_acc += ((static_cast<float>(v404_data[0])) * v56_data);
              v402_acc += ((static_cast<float>(v404_data[1])) * v58_data);
              v402_acc += ((static_cast<float>(v404_data[2])) * v60_data);
              v402_acc += ((static_cast<float>(v404_data[3])) * v62_data);
              v402_acc += ((static_cast<float>(v404_data[4])) * v64_data);
              v402_acc += ((static_cast<float>(v404_data[5])) * v66_data);
              v402_acc += ((static_cast<float>(v404_data[6])) * v68_data);
              v402_acc += ((static_cast<float>(v404_data[7])) * v70_data);
              v402_acc += ((static_cast<float>(v404_data[8])) * v72_data);
              v402_acc += ((static_cast<float>(v404_data[9])) * v74_data);
              v402_acc += ((static_cast<float>(v404_data[10])) * v76_data);
              v402_acc += ((static_cast<float>(v404_data[11])) * v78_data);
              v402_acc += ((static_cast<float>(v404_data[12])) * v80_data);
              v402_acc += ((static_cast<float>(v404_data[13])) * v82_data);
              v402_acc += ((static_cast<float>(v404_data[14])) * v84_data);
              v402_acc += ((static_cast<float>(v404_data[15])) * v86_data);
              tensorforge::intel_esimd::simd<float, 16> v438_data = tensorforge::slmLoad<float, 16>(s0 + (152_i32));
              v402_acc += ((static_cast<float>(v438_data[0])) * v88_data);
              ir0.template select<16, 1>(128) = v402_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v441_n1 = 0; v441_n1 < 9; ++v441_n1) {
                int32_t v442_a = v441_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v444_data(ir0.template select<10, 1>(v442_a));
                r0.template select<10, 1>(v442_a) = v444_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 10), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v450_data = tensorforge::slmLoad<float, 16>(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v452_data = tensorforge::slmLoad<float, 16>(glb_m3 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v454_data = tensorforge::slmLoad<float, 16>(glb_m3 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v456_data = tensorforge::slmLoad<float, 16>(glb_m3 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v458_data = tensorforge::slmLoad<float, 16>(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v460_data = tensorforge::slmLoad<float, 16>(glb_m3 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v462_data = tensorforge::slmLoad<float, 16>(glb_m3 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v464_data = tensorforge::slmLoad<float, 16>(glb_m3 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v466_data = tensorforge::slmLoad<float, 16>(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v468_data = tensorforge::slmLoad<float, 16>(glb_m3 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v470_data = tensorforge::slmLoad<float, 16>(glb_m3 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v472_data = tensorforge::slmLoad<float, 16>(glb_m3 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v474_data = tensorforge::slmLoad<float, 16>(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v476_data = tensorforge::slmLoad<float, 16>(glb_m3 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v478_data = tensorforge::slmLoad<float, 16>(glb_m3 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v480_data = tensorforge::slmLoad<float, 16>(glb_m3 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v482_data = tensorforge::slmLoad<float, 16>(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v483_acc{};
              tensorforge::intel_esimd::simd<float, 16> v484_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v483_acc += ((static_cast<float>(v484_data[0])) * v450_data);
              v483_acc += ((static_cast<float>(v484_data[1])) * v452_data);
              v483_acc += ((static_cast<float>(v484_data[2])) * v454_data);
              v483_acc += ((static_cast<float>(v484_data[3])) * v456_data);
              v483_acc += ((static_cast<float>(v484_data[4])) * v458_data);
              v483_acc += ((static_cast<float>(v484_data[5])) * v460_data);
              v483_acc += ((static_cast<float>(v484_data[6])) * v462_data);
              v483_acc += ((static_cast<float>(v484_data[7])) * v464_data);
              v483_acc += ((static_cast<float>(v484_data[8])) * v466_data);
              v483_acc += ((static_cast<float>(v484_data[9])) * v468_data);
              v483_acc += ((static_cast<float>(v484_data[10])) * v470_data);
              v483_acc += ((static_cast<float>(v484_data[11])) * v472_data);
              v483_acc += ((static_cast<float>(v484_data[12])) * v474_data);
              v483_acc += ((static_cast<float>(v484_data[13])) * v476_data);
              v483_acc += ((static_cast<float>(v484_data[14])) * v478_data);
              v483_acc += ((static_cast<float>(v484_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v520_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v483_acc += ((static_cast<float>(v520_data[0])) * v482_data);
              ir1.template select<16, 1>(0) = v483_acc;
              tensorforge::intel_esimd::simd<float, 16> v523_acc{};
              tensorforge::intel_esimd::simd<float, 16> v525_data = tensorforge::slmLoad<float, 16>(s1 + (17_i32));
              v523_acc += ((static_cast<float>(v525_data[0])) * v450_data);
              v523_acc += ((static_cast<float>(v525_data[1])) * v452_data);
              v523_acc += ((static_cast<float>(v525_data[2])) * v454_data);
              v523_acc += ((static_cast<float>(v525_data[3])) * v456_data);
              v523_acc += ((static_cast<float>(v525_data[4])) * v458_data);
              v523_acc += ((static_cast<float>(v525_data[5])) * v460_data);
              v523_acc += ((static_cast<float>(v525_data[6])) * v462_data);
              v523_acc += ((static_cast<float>(v525_data[7])) * v464_data);
              v523_acc += ((static_cast<float>(v525_data[8])) * v466_data);
              v523_acc += ((static_cast<float>(v525_data[9])) * v468_data);
              v523_acc += ((static_cast<float>(v525_data[10])) * v470_data);
              v523_acc += ((static_cast<float>(v525_data[11])) * v472_data);
              v523_acc += ((static_cast<float>(v525_data[12])) * v474_data);
              v523_acc += ((static_cast<float>(v525_data[13])) * v476_data);
              v523_acc += ((static_cast<float>(v525_data[14])) * v478_data);
              v523_acc += ((static_cast<float>(v525_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v559_data = tensorforge::slmLoad<float, 16>(s1 + (33_i32));
              v523_acc += ((static_cast<float>(v559_data[0])) * v482_data);
              ir1.template select<16, 1>(16) = v523_acc;
              tensorforge::intel_esimd::simd<float, 16> v562_acc{};
              tensorforge::intel_esimd::simd<float, 16> v564_data = tensorforge::slmLoad<float, 16>(s1 + (34_i32));
              v562_acc += ((static_cast<float>(v564_data[0])) * v450_data);
              v562_acc += ((static_cast<float>(v564_data[1])) * v452_data);
              v562_acc += ((static_cast<float>(v564_data[2])) * v454_data);
              v562_acc += ((static_cast<float>(v564_data[3])) * v456_data);
              v562_acc += ((static_cast<float>(v564_data[4])) * v458_data);
              v562_acc += ((static_cast<float>(v564_data[5])) * v460_data);
              v562_acc += ((static_cast<float>(v564_data[6])) * v462_data);
              v562_acc += ((static_cast<float>(v564_data[7])) * v464_data);
              v562_acc += ((static_cast<float>(v564_data[8])) * v466_data);
              v562_acc += ((static_cast<float>(v564_data[9])) * v468_data);
              v562_acc += ((static_cast<float>(v564_data[10])) * v470_data);
              v562_acc += ((static_cast<float>(v564_data[11])) * v472_data);
              v562_acc += ((static_cast<float>(v564_data[12])) * v474_data);
              v562_acc += ((static_cast<float>(v564_data[13])) * v476_data);
              v562_acc += ((static_cast<float>(v564_data[14])) * v478_data);
              v562_acc += ((static_cast<float>(v564_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v598_data = tensorforge::slmLoad<float, 16>(s1 + (50_i32));
              v562_acc += ((static_cast<float>(v598_data[0])) * v482_data);
              ir1.template select<16, 1>(32) = v562_acc;
              tensorforge::intel_esimd::simd<float, 16> v601_acc{};
              tensorforge::intel_esimd::simd<float, 16> v603_data = tensorforge::slmLoad<float, 16>(s1 + (51_i32));
              v601_acc += ((static_cast<float>(v603_data[0])) * v450_data);
              v601_acc += ((static_cast<float>(v603_data[1])) * v452_data);
              v601_acc += ((static_cast<float>(v603_data[2])) * v454_data);
              v601_acc += ((static_cast<float>(v603_data[3])) * v456_data);
              v601_acc += ((static_cast<float>(v603_data[4])) * v458_data);
              v601_acc += ((static_cast<float>(v603_data[5])) * v460_data);
              v601_acc += ((static_cast<float>(v603_data[6])) * v462_data);
              v601_acc += ((static_cast<float>(v603_data[7])) * v464_data);
              v601_acc += ((static_cast<float>(v603_data[8])) * v466_data);
              v601_acc += ((static_cast<float>(v603_data[9])) * v468_data);
              v601_acc += ((static_cast<float>(v603_data[10])) * v470_data);
              v601_acc += ((static_cast<float>(v603_data[11])) * v472_data);
              v601_acc += ((static_cast<float>(v603_data[12])) * v474_data);
              v601_acc += ((static_cast<float>(v603_data[13])) * v476_data);
              v601_acc += ((static_cast<float>(v603_data[14])) * v478_data);
              v601_acc += ((static_cast<float>(v603_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v637_data = tensorforge::slmLoad<float, 16>(s1 + (67_i32));
              v601_acc += ((static_cast<float>(v637_data[0])) * v482_data);
              ir1.template select<16, 1>(48) = v601_acc;
              tensorforge::intel_esimd::simd<float, 16> v640_acc{};
              tensorforge::intel_esimd::simd<float, 16> v642_data = tensorforge::slmLoad<float, 16>(s1 + (68_i32));
              v640_acc += ((static_cast<float>(v642_data[0])) * v450_data);
              v640_acc += ((static_cast<float>(v642_data[1])) * v452_data);
              v640_acc += ((static_cast<float>(v642_data[2])) * v454_data);
              v640_acc += ((static_cast<float>(v642_data[3])) * v456_data);
              v640_acc += ((static_cast<float>(v642_data[4])) * v458_data);
              v640_acc += ((static_cast<float>(v642_data[5])) * v460_data);
              v640_acc += ((static_cast<float>(v642_data[6])) * v462_data);
              v640_acc += ((static_cast<float>(v642_data[7])) * v464_data);
              v640_acc += ((static_cast<float>(v642_data[8])) * v466_data);
              v640_acc += ((static_cast<float>(v642_data[9])) * v468_data);
              v640_acc += ((static_cast<float>(v642_data[10])) * v470_data);
              v640_acc += ((static_cast<float>(v642_data[11])) * v472_data);
              v640_acc += ((static_cast<float>(v642_data[12])) * v474_data);
              v640_acc += ((static_cast<float>(v642_data[13])) * v476_data);
              v640_acc += ((static_cast<float>(v642_data[14])) * v478_data);
              v640_acc += ((static_cast<float>(v642_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v676_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v640_acc += ((static_cast<float>(v676_data[0])) * v482_data);
              ir1.template select<16, 1>(64) = v640_acc;
              tensorforge::intel_esimd::simd<float, 16> v679_acc{};
              tensorforge::intel_esimd::simd<float, 16> v681_data = tensorforge::slmLoad<float, 16>(s1 + (85_i32));
              v679_acc += ((static_cast<float>(v681_data[0])) * v450_data);
              v679_acc += ((static_cast<float>(v681_data[1])) * v452_data);
              v679_acc += ((static_cast<float>(v681_data[2])) * v454_data);
              v679_acc += ((static_cast<float>(v681_data[3])) * v456_data);
              v679_acc += ((static_cast<float>(v681_data[4])) * v458_data);
              v679_acc += ((static_cast<float>(v681_data[5])) * v460_data);
              v679_acc += ((static_cast<float>(v681_data[6])) * v462_data);
              v679_acc += ((static_cast<float>(v681_data[7])) * v464_data);
              v679_acc += ((static_cast<float>(v681_data[8])) * v466_data);
              v679_acc += ((static_cast<float>(v681_data[9])) * v468_data);
              v679_acc += ((static_cast<float>(v681_data[10])) * v470_data);
              v679_acc += ((static_cast<float>(v681_data[11])) * v472_data);
              v679_acc += ((static_cast<float>(v681_data[12])) * v474_data);
              v679_acc += ((static_cast<float>(v681_data[13])) * v476_data);
              v679_acc += ((static_cast<float>(v681_data[14])) * v478_data);
              v679_acc += ((static_cast<float>(v681_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v715_data = tensorforge::slmLoad<float, 16>(s1 + (101_i32));
              v679_acc += ((static_cast<float>(v715_data[0])) * v482_data);
              ir1.template select<16, 1>(80) = v679_acc;
              tensorforge::intel_esimd::simd<float, 16> v718_acc{};
              tensorforge::intel_esimd::simd<float, 16> v720_data = tensorforge::slmLoad<float, 16>(s1 + (102_i32));
              v718_acc += ((static_cast<float>(v720_data[0])) * v450_data);
              v718_acc += ((static_cast<float>(v720_data[1])) * v452_data);
              v718_acc += ((static_cast<float>(v720_data[2])) * v454_data);
              v718_acc += ((static_cast<float>(v720_data[3])) * v456_data);
              v718_acc += ((static_cast<float>(v720_data[4])) * v458_data);
              v718_acc += ((static_cast<float>(v720_data[5])) * v460_data);
              v718_acc += ((static_cast<float>(v720_data[6])) * v462_data);
              v718_acc += ((static_cast<float>(v720_data[7])) * v464_data);
              v718_acc += ((static_cast<float>(v720_data[8])) * v466_data);
              v718_acc += ((static_cast<float>(v720_data[9])) * v468_data);
              v718_acc += ((static_cast<float>(v720_data[10])) * v470_data);
              v718_acc += ((static_cast<float>(v720_data[11])) * v472_data);
              v718_acc += ((static_cast<float>(v720_data[12])) * v474_data);
              v718_acc += ((static_cast<float>(v720_data[13])) * v476_data);
              v718_acc += ((static_cast<float>(v720_data[14])) * v478_data);
              v718_acc += ((static_cast<float>(v720_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v754_data = tensorforge::slmLoad<float, 16>(s1 + (118_i32));
              v718_acc += ((static_cast<float>(v754_data[0])) * v482_data);
              ir1.template select<16, 1>(96) = v718_acc;
              tensorforge::intel_esimd::simd<float, 16> v757_acc{};
              tensorforge::intel_esimd::simd<float, 16> v759_data = tensorforge::slmLoad<float, 16>(s1 + (119_i32));
              v757_acc += ((static_cast<float>(v759_data[0])) * v450_data);
              v757_acc += ((static_cast<float>(v759_data[1])) * v452_data);
              v757_acc += ((static_cast<float>(v759_data[2])) * v454_data);
              v757_acc += ((static_cast<float>(v759_data[3])) * v456_data);
              v757_acc += ((static_cast<float>(v759_data[4])) * v458_data);
              v757_acc += ((static_cast<float>(v759_data[5])) * v460_data);
              v757_acc += ((static_cast<float>(v759_data[6])) * v462_data);
              v757_acc += ((static_cast<float>(v759_data[7])) * v464_data);
              v757_acc += ((static_cast<float>(v759_data[8])) * v466_data);
              v757_acc += ((static_cast<float>(v759_data[9])) * v468_data);
              v757_acc += ((static_cast<float>(v759_data[10])) * v470_data);
              v757_acc += ((static_cast<float>(v759_data[11])) * v472_data);
              v757_acc += ((static_cast<float>(v759_data[12])) * v474_data);
              v757_acc += ((static_cast<float>(v759_data[13])) * v476_data);
              v757_acc += ((static_cast<float>(v759_data[14])) * v478_data);
              v757_acc += ((static_cast<float>(v759_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v793_data = tensorforge::slmLoad<float, 16>(s1 + (135_i32));
              v757_acc += ((static_cast<float>(v793_data[0])) * v482_data);
              ir1.template select<16, 1>(112) = v757_acc;
              tensorforge::intel_esimd::simd<float, 16> v796_acc{};
              tensorforge::intel_esimd::simd<float, 16> v798_data = tensorforge::slmLoad<float, 16>(s1 + (136_i32));
              v796_acc += ((static_cast<float>(v798_data[0])) * v450_data);
              v796_acc += ((static_cast<float>(v798_data[1])) * v452_data);
              v796_acc += ((static_cast<float>(v798_data[2])) * v454_data);
              v796_acc += ((static_cast<float>(v798_data[3])) * v456_data);
              v796_acc += ((static_cast<float>(v798_data[4])) * v458_data);
              v796_acc += ((static_cast<float>(v798_data[5])) * v460_data);
              v796_acc += ((static_cast<float>(v798_data[6])) * v462_data);
              v796_acc += ((static_cast<float>(v798_data[7])) * v464_data);
              v796_acc += ((static_cast<float>(v798_data[8])) * v466_data);
              v796_acc += ((static_cast<float>(v798_data[9])) * v468_data);
              v796_acc += ((static_cast<float>(v798_data[10])) * v470_data);
              v796_acc += ((static_cast<float>(v798_data[11])) * v472_data);
              v796_acc += ((static_cast<float>(v798_data[12])) * v474_data);
              v796_acc += ((static_cast<float>(v798_data[13])) * v476_data);
              v796_acc += ((static_cast<float>(v798_data[14])) * v478_data);
              v796_acc += ((static_cast<float>(v798_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v832_data = tensorforge::slmLoad<float, 16>(s1 + (152_i32));
              v796_acc += ((static_cast<float>(v832_data[0])) * v482_data);
              ir1.template select<16, 1>(128) = v796_acc;
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v835_n1 = 0; v835_n1 < 9; ++v835_n1) {
                int32_t v836_a = v835_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v838_data(ir1.template select<10, 1>(v836_a));
                tensorforge::intel_esimd::simd<float, 10> v839_data(r0.template select<10, 1>(v836_a));
                r1.template select<10, 1>(v836_a) = (v839_data + v838_data);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v841_i1 = 0; v841_i1 < 9; ++v841_i1) {
                tensorforge::intel_esimd::simd<float, 10> v844_data(r1.template select<10, 1>((v841_i1 * 16)));
                v844_data.copy_to(glb_m0 + ((v841_i1 * 10)));
              }
            }
          }
        }
      }
    });
  });
}


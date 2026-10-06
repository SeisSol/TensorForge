// === base name ===
kernel_443f6db1a65ea7f8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_443f6db1a65ea7f8 = {{1, 32, 1}, 16, 10, 1, 32, 44416, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_443f6db1a65ea7f8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_443f6db1a65ea7f8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_443f6db1a65ea7f8(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_443f6db1a65ea7f8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_443f6db1a65ea7f8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_443f6db1a65ea7f8(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_443f6db1a65ea7f8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<11104 * sizeof(float)>(); {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":11104}],"shared_bytes":44416,"shared_elements":11104,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,0],[10,17]],"name":"m1","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[17,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,0],[10,17]],"name":"m3","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[17,9]],"name":"m4","ordered":false,"parts":1,"shape":[17,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (336 * item.get_local_id(1) + 352);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (320);
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v16_ld;
            v16_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v17_ld;
            v17_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v21_ld;
            v21_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v21_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 10> v22_ld;
            v22_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 10>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v22_ld);
          }
          const float *const __restrict__ ptr_glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> glb_m3 = totalShrMem + (176);
          // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v25_ld;
            v25_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v25_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v26_ld;
            v26_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v26_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v27_ld;
            v27_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v27_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v28_ld;
            v28_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v28_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v29_ld;
            v29_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v29_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v30_ld;
            v30_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v30_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v31_ld;
            v31_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v31_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v32_ld;
            v32_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v32_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v33_ld;
            v33_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v33_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v34_ld;
            v34_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v34_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 10> v35_ld;
            v35_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 10>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v35_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (160);
          for (size_t v38_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v38_batchId0 < numElements0; v38_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v39_ahead1 = v38_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v41_batchId1 = (v39_ahead1 < numElements0) ? v39_ahead1 : v38_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v38_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v38_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v38_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v38_batchId0 * 153 + 0 + m4_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v49_ld;
              v49_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v49_ld);
              tensorforge::intel_esimd::simd<float, 64> v50_ld;
              v50_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v50_ld);
              tensorforge::intel_esimd::simd<float, 16> v51_ld;
              v51_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v51_ld);
              tensorforge::intel_esimd::simd<float, 9> v52_ld;
              v52_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v52_ld);
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v53_ld;
              v53_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v53_ld);
              tensorforge::intel_esimd::simd<float, 64> v54_ld;
              v54_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 64), v54_ld);
              tensorforge::intel_esimd::simd<float, 16> v55_ld;
              v55_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s1 + (0 + 0 + 1 * 0 + 128), v55_ld);
              tensorforge::intel_esimd::simd<float, 9> v56_ld;
              v56_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s1 + (0 + 0 + 1 * 0 + 144), v56_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v62_data = tensorforge::slmLoad<float, 16>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(glb_m1 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data = tensorforge::slmLoad<float, 16>(glb_m1 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_data = tensorforge::slmLoad<float, 16>(glb_m1 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data = tensorforge::slmLoad<float, 16>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_data = tensorforge::slmLoad<float, 16>(glb_m1 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v74_data = tensorforge::slmLoad<float, 16>(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_data = tensorforge::slmLoad<float, 16>(glb_m1 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v78_data = tensorforge::slmLoad<float, 16>(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data = tensorforge::slmLoad<float, 16>(glb_m1 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v82_data = tensorforge::slmLoad<float, 16>(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v84_data = tensorforge::slmLoad<float, 16>(glb_m1 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v86_data = tensorforge::slmLoad<float, 16>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v88_data = tensorforge::slmLoad<float, 16>(glb_m1 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v90_data = tensorforge::slmLoad<float, 16>(glb_m1 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v92_data = tensorforge::slmLoad<float, 16>(glb_m1 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v94_data = tensorforge::slmLoad<float, 16>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v95_acc{};
              tensorforge::intel_esimd::simd<float, 16> v96_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v95_acc += ((static_cast<float>(v96_data[0])) * v62_data);
              v95_acc += ((static_cast<float>(v96_data[1])) * v64_data);
              v95_acc += ((static_cast<float>(v96_data[2])) * v66_data);
              v95_acc += ((static_cast<float>(v96_data[3])) * v68_data);
              v95_acc += ((static_cast<float>(v96_data[4])) * v70_data);
              v95_acc += ((static_cast<float>(v96_data[5])) * v72_data);
              v95_acc += ((static_cast<float>(v96_data[6])) * v74_data);
              v95_acc += ((static_cast<float>(v96_data[7])) * v76_data);
              v95_acc += ((static_cast<float>(v96_data[8])) * v78_data);
              v95_acc += ((static_cast<float>(v96_data[9])) * v80_data);
              v95_acc += ((static_cast<float>(v96_data[10])) * v82_data);
              v95_acc += ((static_cast<float>(v96_data[11])) * v84_data);
              v95_acc += ((static_cast<float>(v96_data[12])) * v86_data);
              v95_acc += ((static_cast<float>(v96_data[13])) * v88_data);
              v95_acc += ((static_cast<float>(v96_data[14])) * v90_data);
              v95_acc += ((static_cast<float>(v96_data[15])) * v92_data);
              tensorforge::intel_esimd::simd<float, 16> v132_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v95_acc += ((static_cast<float>(v132_data[0])) * v94_data);
              ir0.template select<16, 1>(0) = v95_acc;
              tensorforge::intel_esimd::simd<float, 16> v135_acc{};
              tensorforge::intel_esimd::simd<float, 16> v137_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              v135_acc += ((static_cast<float>(v137_data[0])) * v62_data);
              v135_acc += ((static_cast<float>(v137_data[1])) * v64_data);
              v135_acc += ((static_cast<float>(v137_data[2])) * v66_data);
              v135_acc += ((static_cast<float>(v137_data[3])) * v68_data);
              v135_acc += ((static_cast<float>(v137_data[4])) * v70_data);
              v135_acc += ((static_cast<float>(v137_data[5])) * v72_data);
              v135_acc += ((static_cast<float>(v137_data[6])) * v74_data);
              v135_acc += ((static_cast<float>(v137_data[7])) * v76_data);
              v135_acc += ((static_cast<float>(v137_data[8])) * v78_data);
              v135_acc += ((static_cast<float>(v137_data[9])) * v80_data);
              v135_acc += ((static_cast<float>(v137_data[10])) * v82_data);
              v135_acc += ((static_cast<float>(v137_data[11])) * v84_data);
              v135_acc += ((static_cast<float>(v137_data[12])) * v86_data);
              v135_acc += ((static_cast<float>(v137_data[13])) * v88_data);
              v135_acc += ((static_cast<float>(v137_data[14])) * v90_data);
              v135_acc += ((static_cast<float>(v137_data[15])) * v92_data);
              tensorforge::intel_esimd::simd<float, 16> v171_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v135_acc += ((static_cast<float>(v171_data[0])) * v94_data);
              ir0.template select<16, 1>(16) = v135_acc;
              tensorforge::intel_esimd::simd<float, 16> v174_acc{};
              tensorforge::intel_esimd::simd<float, 16> v176_data = tensorforge::slmLoad<float, 16>(s0 + (34_i32));
              v174_acc += ((static_cast<float>(v176_data[0])) * v62_data);
              v174_acc += ((static_cast<float>(v176_data[1])) * v64_data);
              v174_acc += ((static_cast<float>(v176_data[2])) * v66_data);
              v174_acc += ((static_cast<float>(v176_data[3])) * v68_data);
              v174_acc += ((static_cast<float>(v176_data[4])) * v70_data);
              v174_acc += ((static_cast<float>(v176_data[5])) * v72_data);
              v174_acc += ((static_cast<float>(v176_data[6])) * v74_data);
              v174_acc += ((static_cast<float>(v176_data[7])) * v76_data);
              v174_acc += ((static_cast<float>(v176_data[8])) * v78_data);
              v174_acc += ((static_cast<float>(v176_data[9])) * v80_data);
              v174_acc += ((static_cast<float>(v176_data[10])) * v82_data);
              v174_acc += ((static_cast<float>(v176_data[11])) * v84_data);
              v174_acc += ((static_cast<float>(v176_data[12])) * v86_data);
              v174_acc += ((static_cast<float>(v176_data[13])) * v88_data);
              v174_acc += ((static_cast<float>(v176_data[14])) * v90_data);
              v174_acc += ((static_cast<float>(v176_data[15])) * v92_data);
              tensorforge::intel_esimd::simd<float, 16> v210_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v174_acc += ((static_cast<float>(v210_data[0])) * v94_data);
              ir0.template select<16, 1>(32) = v174_acc;
              tensorforge::intel_esimd::simd<float, 16> v213_acc{};
              tensorforge::intel_esimd::simd<float, 16> v215_data = tensorforge::slmLoad<float, 16>(s0 + (51_i32));
              v213_acc += ((static_cast<float>(v215_data[0])) * v62_data);
              v213_acc += ((static_cast<float>(v215_data[1])) * v64_data);
              v213_acc += ((static_cast<float>(v215_data[2])) * v66_data);
              v213_acc += ((static_cast<float>(v215_data[3])) * v68_data);
              v213_acc += ((static_cast<float>(v215_data[4])) * v70_data);
              v213_acc += ((static_cast<float>(v215_data[5])) * v72_data);
              v213_acc += ((static_cast<float>(v215_data[6])) * v74_data);
              v213_acc += ((static_cast<float>(v215_data[7])) * v76_data);
              v213_acc += ((static_cast<float>(v215_data[8])) * v78_data);
              v213_acc += ((static_cast<float>(v215_data[9])) * v80_data);
              v213_acc += ((static_cast<float>(v215_data[10])) * v82_data);
              v213_acc += ((static_cast<float>(v215_data[11])) * v84_data);
              v213_acc += ((static_cast<float>(v215_data[12])) * v86_data);
              v213_acc += ((static_cast<float>(v215_data[13])) * v88_data);
              v213_acc += ((static_cast<float>(v215_data[14])) * v90_data);
              v213_acc += ((static_cast<float>(v215_data[15])) * v92_data);
              tensorforge::intel_esimd::simd<float, 16> v249_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v213_acc += ((static_cast<float>(v249_data[0])) * v94_data);
              ir0.template select<16, 1>(48) = v213_acc;
              tensorforge::intel_esimd::simd<float, 16> v252_acc{};
              tensorforge::intel_esimd::simd<float, 16> v254_data = tensorforge::slmLoad<float, 16>(s0 + (68_i32));
              v252_acc += ((static_cast<float>(v254_data[0])) * v62_data);
              v252_acc += ((static_cast<float>(v254_data[1])) * v64_data);
              v252_acc += ((static_cast<float>(v254_data[2])) * v66_data);
              v252_acc += ((static_cast<float>(v254_data[3])) * v68_data);
              v252_acc += ((static_cast<float>(v254_data[4])) * v70_data);
              v252_acc += ((static_cast<float>(v254_data[5])) * v72_data);
              v252_acc += ((static_cast<float>(v254_data[6])) * v74_data);
              v252_acc += ((static_cast<float>(v254_data[7])) * v76_data);
              v252_acc += ((static_cast<float>(v254_data[8])) * v78_data);
              v252_acc += ((static_cast<float>(v254_data[9])) * v80_data);
              v252_acc += ((static_cast<float>(v254_data[10])) * v82_data);
              v252_acc += ((static_cast<float>(v254_data[11])) * v84_data);
              v252_acc += ((static_cast<float>(v254_data[12])) * v86_data);
              v252_acc += ((static_cast<float>(v254_data[13])) * v88_data);
              v252_acc += ((static_cast<float>(v254_data[14])) * v90_data);
              v252_acc += ((static_cast<float>(v254_data[15])) * v92_data);
              tensorforge::intel_esimd::simd<float, 16> v288_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v252_acc += ((static_cast<float>(v288_data[0])) * v94_data);
              ir0.template select<16, 1>(64) = v252_acc;
              tensorforge::intel_esimd::simd<float, 16> v291_acc{};
              tensorforge::intel_esimd::simd<float, 16> v293_data = tensorforge::slmLoad<float, 16>(s0 + (85_i32));
              v291_acc += ((static_cast<float>(v293_data[0])) * v62_data);
              v291_acc += ((static_cast<float>(v293_data[1])) * v64_data);
              v291_acc += ((static_cast<float>(v293_data[2])) * v66_data);
              v291_acc += ((static_cast<float>(v293_data[3])) * v68_data);
              v291_acc += ((static_cast<float>(v293_data[4])) * v70_data);
              v291_acc += ((static_cast<float>(v293_data[5])) * v72_data);
              v291_acc += ((static_cast<float>(v293_data[6])) * v74_data);
              v291_acc += ((static_cast<float>(v293_data[7])) * v76_data);
              v291_acc += ((static_cast<float>(v293_data[8])) * v78_data);
              v291_acc += ((static_cast<float>(v293_data[9])) * v80_data);
              v291_acc += ((static_cast<float>(v293_data[10])) * v82_data);
              v291_acc += ((static_cast<float>(v293_data[11])) * v84_data);
              v291_acc += ((static_cast<float>(v293_data[12])) * v86_data);
              v291_acc += ((static_cast<float>(v293_data[13])) * v88_data);
              v291_acc += ((static_cast<float>(v293_data[14])) * v90_data);
              v291_acc += ((static_cast<float>(v293_data[15])) * v92_data);
              tensorforge::intel_esimd::simd<float, 16> v327_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v291_acc += ((static_cast<float>(v327_data[0])) * v94_data);
              ir0.template select<16, 1>(80) = v291_acc;
              tensorforge::intel_esimd::simd<float, 16> v330_acc{};
              tensorforge::intel_esimd::simd<float, 16> v332_data = tensorforge::slmLoad<float, 16>(s0 + (102_i32));
              v330_acc += ((static_cast<float>(v332_data[0])) * v62_data);
              v330_acc += ((static_cast<float>(v332_data[1])) * v64_data);
              v330_acc += ((static_cast<float>(v332_data[2])) * v66_data);
              v330_acc += ((static_cast<float>(v332_data[3])) * v68_data);
              v330_acc += ((static_cast<float>(v332_data[4])) * v70_data);
              v330_acc += ((static_cast<float>(v332_data[5])) * v72_data);
              v330_acc += ((static_cast<float>(v332_data[6])) * v74_data);
              v330_acc += ((static_cast<float>(v332_data[7])) * v76_data);
              v330_acc += ((static_cast<float>(v332_data[8])) * v78_data);
              v330_acc += ((static_cast<float>(v332_data[9])) * v80_data);
              v330_acc += ((static_cast<float>(v332_data[10])) * v82_data);
              v330_acc += ((static_cast<float>(v332_data[11])) * v84_data);
              v330_acc += ((static_cast<float>(v332_data[12])) * v86_data);
              v330_acc += ((static_cast<float>(v332_data[13])) * v88_data);
              v330_acc += ((static_cast<float>(v332_data[14])) * v90_data);
              v330_acc += ((static_cast<float>(v332_data[15])) * v92_data);
              tensorforge::intel_esimd::simd<float, 16> v366_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v330_acc += ((static_cast<float>(v366_data[0])) * v94_data);
              ir0.template select<16, 1>(96) = v330_acc;
              tensorforge::intel_esimd::simd<float, 16> v369_acc{};
              tensorforge::intel_esimd::simd<float, 16> v371_data = tensorforge::slmLoad<float, 16>(s0 + (119_i32));
              v369_acc += ((static_cast<float>(v371_data[0])) * v62_data);
              v369_acc += ((static_cast<float>(v371_data[1])) * v64_data);
              v369_acc += ((static_cast<float>(v371_data[2])) * v66_data);
              v369_acc += ((static_cast<float>(v371_data[3])) * v68_data);
              v369_acc += ((static_cast<float>(v371_data[4])) * v70_data);
              v369_acc += ((static_cast<float>(v371_data[5])) * v72_data);
              v369_acc += ((static_cast<float>(v371_data[6])) * v74_data);
              v369_acc += ((static_cast<float>(v371_data[7])) * v76_data);
              v369_acc += ((static_cast<float>(v371_data[8])) * v78_data);
              v369_acc += ((static_cast<float>(v371_data[9])) * v80_data);
              v369_acc += ((static_cast<float>(v371_data[10])) * v82_data);
              v369_acc += ((static_cast<float>(v371_data[11])) * v84_data);
              v369_acc += ((static_cast<float>(v371_data[12])) * v86_data);
              v369_acc += ((static_cast<float>(v371_data[13])) * v88_data);
              v369_acc += ((static_cast<float>(v371_data[14])) * v90_data);
              v369_acc += ((static_cast<float>(v371_data[15])) * v92_data);
              tensorforge::intel_esimd::simd<float, 16> v405_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v369_acc += ((static_cast<float>(v405_data[0])) * v94_data);
              ir0.template select<16, 1>(112) = v369_acc;
              tensorforge::intel_esimd::simd<float, 16> v408_acc{};
              tensorforge::intel_esimd::simd<float, 16> v410_data = tensorforge::slmLoad<float, 16>(s0 + (136_i32));
              v408_acc += ((static_cast<float>(v410_data[0])) * v62_data);
              v408_acc += ((static_cast<float>(v410_data[1])) * v64_data);
              v408_acc += ((static_cast<float>(v410_data[2])) * v66_data);
              v408_acc += ((static_cast<float>(v410_data[3])) * v68_data);
              v408_acc += ((static_cast<float>(v410_data[4])) * v70_data);
              v408_acc += ((static_cast<float>(v410_data[5])) * v72_data);
              v408_acc += ((static_cast<float>(v410_data[6])) * v74_data);
              v408_acc += ((static_cast<float>(v410_data[7])) * v76_data);
              v408_acc += ((static_cast<float>(v410_data[8])) * v78_data);
              v408_acc += ((static_cast<float>(v410_data[9])) * v80_data);
              v408_acc += ((static_cast<float>(v410_data[10])) * v82_data);
              v408_acc += ((static_cast<float>(v410_data[11])) * v84_data);
              v408_acc += ((static_cast<float>(v410_data[12])) * v86_data);
              v408_acc += ((static_cast<float>(v410_data[13])) * v88_data);
              v408_acc += ((static_cast<float>(v410_data[14])) * v90_data);
              v408_acc += ((static_cast<float>(v410_data[15])) * v92_data);
              tensorforge::intel_esimd::simd<float, 16> v444_data = tensorforge::slmLoad<float, 16>(s0 + (152_i32));
              v408_acc += ((static_cast<float>(v444_data[0])) * v94_data);
              ir0.template select<16, 1>(128) = v408_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v447_n1 = 0; v447_n1 < 9; ++v447_n1) {
                int32_t v448_a = v447_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v450_data(ir0.template select<10, 1>(v448_a));
                r0.template select<10, 1>(v448_a) = v450_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 10), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v456_data = tensorforge::slmLoad<float, 16>(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v458_data = tensorforge::slmLoad<float, 16>(glb_m3 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v460_data = tensorforge::slmLoad<float, 16>(glb_m3 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v462_data = tensorforge::slmLoad<float, 16>(glb_m3 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v464_data = tensorforge::slmLoad<float, 16>(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v466_data = tensorforge::slmLoad<float, 16>(glb_m3 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v468_data = tensorforge::slmLoad<float, 16>(glb_m3 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v470_data = tensorforge::slmLoad<float, 16>(glb_m3 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v472_data = tensorforge::slmLoad<float, 16>(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v474_data = tensorforge::slmLoad<float, 16>(glb_m3 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v476_data = tensorforge::slmLoad<float, 16>(glb_m3 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v478_data = tensorforge::slmLoad<float, 16>(glb_m3 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v480_data = tensorforge::slmLoad<float, 16>(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v482_data = tensorforge::slmLoad<float, 16>(glb_m3 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v484_data = tensorforge::slmLoad<float, 16>(glb_m3 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v486_data = tensorforge::slmLoad<float, 16>(glb_m3 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v488_data = tensorforge::slmLoad<float, 16>(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v489_acc{};
              tensorforge::intel_esimd::simd<float, 16> v490_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v489_acc += ((static_cast<float>(v490_data[0])) * v456_data);
              v489_acc += ((static_cast<float>(v490_data[1])) * v458_data);
              v489_acc += ((static_cast<float>(v490_data[2])) * v460_data);
              v489_acc += ((static_cast<float>(v490_data[3])) * v462_data);
              v489_acc += ((static_cast<float>(v490_data[4])) * v464_data);
              v489_acc += ((static_cast<float>(v490_data[5])) * v466_data);
              v489_acc += ((static_cast<float>(v490_data[6])) * v468_data);
              v489_acc += ((static_cast<float>(v490_data[7])) * v470_data);
              v489_acc += ((static_cast<float>(v490_data[8])) * v472_data);
              v489_acc += ((static_cast<float>(v490_data[9])) * v474_data);
              v489_acc += ((static_cast<float>(v490_data[10])) * v476_data);
              v489_acc += ((static_cast<float>(v490_data[11])) * v478_data);
              v489_acc += ((static_cast<float>(v490_data[12])) * v480_data);
              v489_acc += ((static_cast<float>(v490_data[13])) * v482_data);
              v489_acc += ((static_cast<float>(v490_data[14])) * v484_data);
              v489_acc += ((static_cast<float>(v490_data[15])) * v486_data);
              tensorforge::intel_esimd::simd<float, 16> v526_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v489_acc += ((static_cast<float>(v526_data[0])) * v488_data);
              ir1.template select<16, 1>(0) = v489_acc;
              tensorforge::intel_esimd::simd<float, 16> v529_acc{};
              tensorforge::intel_esimd::simd<float, 16> v531_data = tensorforge::slmLoad<float, 16>(s1 + (17_i32));
              v529_acc += ((static_cast<float>(v531_data[0])) * v456_data);
              v529_acc += ((static_cast<float>(v531_data[1])) * v458_data);
              v529_acc += ((static_cast<float>(v531_data[2])) * v460_data);
              v529_acc += ((static_cast<float>(v531_data[3])) * v462_data);
              v529_acc += ((static_cast<float>(v531_data[4])) * v464_data);
              v529_acc += ((static_cast<float>(v531_data[5])) * v466_data);
              v529_acc += ((static_cast<float>(v531_data[6])) * v468_data);
              v529_acc += ((static_cast<float>(v531_data[7])) * v470_data);
              v529_acc += ((static_cast<float>(v531_data[8])) * v472_data);
              v529_acc += ((static_cast<float>(v531_data[9])) * v474_data);
              v529_acc += ((static_cast<float>(v531_data[10])) * v476_data);
              v529_acc += ((static_cast<float>(v531_data[11])) * v478_data);
              v529_acc += ((static_cast<float>(v531_data[12])) * v480_data);
              v529_acc += ((static_cast<float>(v531_data[13])) * v482_data);
              v529_acc += ((static_cast<float>(v531_data[14])) * v484_data);
              v529_acc += ((static_cast<float>(v531_data[15])) * v486_data);
              tensorforge::intel_esimd::simd<float, 16> v565_data = tensorforge::slmLoad<float, 16>(s1 + (33_i32));
              v529_acc += ((static_cast<float>(v565_data[0])) * v488_data);
              ir1.template select<16, 1>(16) = v529_acc;
              tensorforge::intel_esimd::simd<float, 16> v568_acc{};
              tensorforge::intel_esimd::simd<float, 16> v570_data = tensorforge::slmLoad<float, 16>(s1 + (34_i32));
              v568_acc += ((static_cast<float>(v570_data[0])) * v456_data);
              v568_acc += ((static_cast<float>(v570_data[1])) * v458_data);
              v568_acc += ((static_cast<float>(v570_data[2])) * v460_data);
              v568_acc += ((static_cast<float>(v570_data[3])) * v462_data);
              v568_acc += ((static_cast<float>(v570_data[4])) * v464_data);
              v568_acc += ((static_cast<float>(v570_data[5])) * v466_data);
              v568_acc += ((static_cast<float>(v570_data[6])) * v468_data);
              v568_acc += ((static_cast<float>(v570_data[7])) * v470_data);
              v568_acc += ((static_cast<float>(v570_data[8])) * v472_data);
              v568_acc += ((static_cast<float>(v570_data[9])) * v474_data);
              v568_acc += ((static_cast<float>(v570_data[10])) * v476_data);
              v568_acc += ((static_cast<float>(v570_data[11])) * v478_data);
              v568_acc += ((static_cast<float>(v570_data[12])) * v480_data);
              v568_acc += ((static_cast<float>(v570_data[13])) * v482_data);
              v568_acc += ((static_cast<float>(v570_data[14])) * v484_data);
              v568_acc += ((static_cast<float>(v570_data[15])) * v486_data);
              tensorforge::intel_esimd::simd<float, 16> v604_data = tensorforge::slmLoad<float, 16>(s1 + (50_i32));
              v568_acc += ((static_cast<float>(v604_data[0])) * v488_data);
              ir1.template select<16, 1>(32) = v568_acc;
              tensorforge::intel_esimd::simd<float, 16> v607_acc{};
              tensorforge::intel_esimd::simd<float, 16> v609_data = tensorforge::slmLoad<float, 16>(s1 + (51_i32));
              v607_acc += ((static_cast<float>(v609_data[0])) * v456_data);
              v607_acc += ((static_cast<float>(v609_data[1])) * v458_data);
              v607_acc += ((static_cast<float>(v609_data[2])) * v460_data);
              v607_acc += ((static_cast<float>(v609_data[3])) * v462_data);
              v607_acc += ((static_cast<float>(v609_data[4])) * v464_data);
              v607_acc += ((static_cast<float>(v609_data[5])) * v466_data);
              v607_acc += ((static_cast<float>(v609_data[6])) * v468_data);
              v607_acc += ((static_cast<float>(v609_data[7])) * v470_data);
              v607_acc += ((static_cast<float>(v609_data[8])) * v472_data);
              v607_acc += ((static_cast<float>(v609_data[9])) * v474_data);
              v607_acc += ((static_cast<float>(v609_data[10])) * v476_data);
              v607_acc += ((static_cast<float>(v609_data[11])) * v478_data);
              v607_acc += ((static_cast<float>(v609_data[12])) * v480_data);
              v607_acc += ((static_cast<float>(v609_data[13])) * v482_data);
              v607_acc += ((static_cast<float>(v609_data[14])) * v484_data);
              v607_acc += ((static_cast<float>(v609_data[15])) * v486_data);
              tensorforge::intel_esimd::simd<float, 16> v643_data = tensorforge::slmLoad<float, 16>(s1 + (67_i32));
              v607_acc += ((static_cast<float>(v643_data[0])) * v488_data);
              ir1.template select<16, 1>(48) = v607_acc;
              tensorforge::intel_esimd::simd<float, 16> v646_acc{};
              tensorforge::intel_esimd::simd<float, 16> v648_data = tensorforge::slmLoad<float, 16>(s1 + (68_i32));
              v646_acc += ((static_cast<float>(v648_data[0])) * v456_data);
              v646_acc += ((static_cast<float>(v648_data[1])) * v458_data);
              v646_acc += ((static_cast<float>(v648_data[2])) * v460_data);
              v646_acc += ((static_cast<float>(v648_data[3])) * v462_data);
              v646_acc += ((static_cast<float>(v648_data[4])) * v464_data);
              v646_acc += ((static_cast<float>(v648_data[5])) * v466_data);
              v646_acc += ((static_cast<float>(v648_data[6])) * v468_data);
              v646_acc += ((static_cast<float>(v648_data[7])) * v470_data);
              v646_acc += ((static_cast<float>(v648_data[8])) * v472_data);
              v646_acc += ((static_cast<float>(v648_data[9])) * v474_data);
              v646_acc += ((static_cast<float>(v648_data[10])) * v476_data);
              v646_acc += ((static_cast<float>(v648_data[11])) * v478_data);
              v646_acc += ((static_cast<float>(v648_data[12])) * v480_data);
              v646_acc += ((static_cast<float>(v648_data[13])) * v482_data);
              v646_acc += ((static_cast<float>(v648_data[14])) * v484_data);
              v646_acc += ((static_cast<float>(v648_data[15])) * v486_data);
              tensorforge::intel_esimd::simd<float, 16> v682_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v646_acc += ((static_cast<float>(v682_data[0])) * v488_data);
              ir1.template select<16, 1>(64) = v646_acc;
              tensorforge::intel_esimd::simd<float, 16> v685_acc{};
              tensorforge::intel_esimd::simd<float, 16> v687_data = tensorforge::slmLoad<float, 16>(s1 + (85_i32));
              v685_acc += ((static_cast<float>(v687_data[0])) * v456_data);
              v685_acc += ((static_cast<float>(v687_data[1])) * v458_data);
              v685_acc += ((static_cast<float>(v687_data[2])) * v460_data);
              v685_acc += ((static_cast<float>(v687_data[3])) * v462_data);
              v685_acc += ((static_cast<float>(v687_data[4])) * v464_data);
              v685_acc += ((static_cast<float>(v687_data[5])) * v466_data);
              v685_acc += ((static_cast<float>(v687_data[6])) * v468_data);
              v685_acc += ((static_cast<float>(v687_data[7])) * v470_data);
              v685_acc += ((static_cast<float>(v687_data[8])) * v472_data);
              v685_acc += ((static_cast<float>(v687_data[9])) * v474_data);
              v685_acc += ((static_cast<float>(v687_data[10])) * v476_data);
              v685_acc += ((static_cast<float>(v687_data[11])) * v478_data);
              v685_acc += ((static_cast<float>(v687_data[12])) * v480_data);
              v685_acc += ((static_cast<float>(v687_data[13])) * v482_data);
              v685_acc += ((static_cast<float>(v687_data[14])) * v484_data);
              v685_acc += ((static_cast<float>(v687_data[15])) * v486_data);
              tensorforge::intel_esimd::simd<float, 16> v721_data = tensorforge::slmLoad<float, 16>(s1 + (101_i32));
              v685_acc += ((static_cast<float>(v721_data[0])) * v488_data);
              ir1.template select<16, 1>(80) = v685_acc;
              tensorforge::intel_esimd::simd<float, 16> v724_acc{};
              tensorforge::intel_esimd::simd<float, 16> v726_data = tensorforge::slmLoad<float, 16>(s1 + (102_i32));
              v724_acc += ((static_cast<float>(v726_data[0])) * v456_data);
              v724_acc += ((static_cast<float>(v726_data[1])) * v458_data);
              v724_acc += ((static_cast<float>(v726_data[2])) * v460_data);
              v724_acc += ((static_cast<float>(v726_data[3])) * v462_data);
              v724_acc += ((static_cast<float>(v726_data[4])) * v464_data);
              v724_acc += ((static_cast<float>(v726_data[5])) * v466_data);
              v724_acc += ((static_cast<float>(v726_data[6])) * v468_data);
              v724_acc += ((static_cast<float>(v726_data[7])) * v470_data);
              v724_acc += ((static_cast<float>(v726_data[8])) * v472_data);
              v724_acc += ((static_cast<float>(v726_data[9])) * v474_data);
              v724_acc += ((static_cast<float>(v726_data[10])) * v476_data);
              v724_acc += ((static_cast<float>(v726_data[11])) * v478_data);
              v724_acc += ((static_cast<float>(v726_data[12])) * v480_data);
              v724_acc += ((static_cast<float>(v726_data[13])) * v482_data);
              v724_acc += ((static_cast<float>(v726_data[14])) * v484_data);
              v724_acc += ((static_cast<float>(v726_data[15])) * v486_data);
              tensorforge::intel_esimd::simd<float, 16> v760_data = tensorforge::slmLoad<float, 16>(s1 + (118_i32));
              v724_acc += ((static_cast<float>(v760_data[0])) * v488_data);
              ir1.template select<16, 1>(96) = v724_acc;
              tensorforge::intel_esimd::simd<float, 16> v763_acc{};
              tensorforge::intel_esimd::simd<float, 16> v765_data = tensorforge::slmLoad<float, 16>(s1 + (119_i32));
              v763_acc += ((static_cast<float>(v765_data[0])) * v456_data);
              v763_acc += ((static_cast<float>(v765_data[1])) * v458_data);
              v763_acc += ((static_cast<float>(v765_data[2])) * v460_data);
              v763_acc += ((static_cast<float>(v765_data[3])) * v462_data);
              v763_acc += ((static_cast<float>(v765_data[4])) * v464_data);
              v763_acc += ((static_cast<float>(v765_data[5])) * v466_data);
              v763_acc += ((static_cast<float>(v765_data[6])) * v468_data);
              v763_acc += ((static_cast<float>(v765_data[7])) * v470_data);
              v763_acc += ((static_cast<float>(v765_data[8])) * v472_data);
              v763_acc += ((static_cast<float>(v765_data[9])) * v474_data);
              v763_acc += ((static_cast<float>(v765_data[10])) * v476_data);
              v763_acc += ((static_cast<float>(v765_data[11])) * v478_data);
              v763_acc += ((static_cast<float>(v765_data[12])) * v480_data);
              v763_acc += ((static_cast<float>(v765_data[13])) * v482_data);
              v763_acc += ((static_cast<float>(v765_data[14])) * v484_data);
              v763_acc += ((static_cast<float>(v765_data[15])) * v486_data);
              tensorforge::intel_esimd::simd<float, 16> v799_data = tensorforge::slmLoad<float, 16>(s1 + (135_i32));
              v763_acc += ((static_cast<float>(v799_data[0])) * v488_data);
              ir1.template select<16, 1>(112) = v763_acc;
              tensorforge::intel_esimd::simd<float, 16> v802_acc{};
              tensorforge::intel_esimd::simd<float, 16> v804_data = tensorforge::slmLoad<float, 16>(s1 + (136_i32));
              v802_acc += ((static_cast<float>(v804_data[0])) * v456_data);
              v802_acc += ((static_cast<float>(v804_data[1])) * v458_data);
              v802_acc += ((static_cast<float>(v804_data[2])) * v460_data);
              v802_acc += ((static_cast<float>(v804_data[3])) * v462_data);
              v802_acc += ((static_cast<float>(v804_data[4])) * v464_data);
              v802_acc += ((static_cast<float>(v804_data[5])) * v466_data);
              v802_acc += ((static_cast<float>(v804_data[6])) * v468_data);
              v802_acc += ((static_cast<float>(v804_data[7])) * v470_data);
              v802_acc += ((static_cast<float>(v804_data[8])) * v472_data);
              v802_acc += ((static_cast<float>(v804_data[9])) * v474_data);
              v802_acc += ((static_cast<float>(v804_data[10])) * v476_data);
              v802_acc += ((static_cast<float>(v804_data[11])) * v478_data);
              v802_acc += ((static_cast<float>(v804_data[12])) * v480_data);
              v802_acc += ((static_cast<float>(v804_data[13])) * v482_data);
              v802_acc += ((static_cast<float>(v804_data[14])) * v484_data);
              v802_acc += ((static_cast<float>(v804_data[15])) * v486_data);
              tensorforge::intel_esimd::simd<float, 16> v838_data = tensorforge::slmLoad<float, 16>(s1 + (152_i32));
              v802_acc += ((static_cast<float>(v838_data[0])) * v488_data);
              ir1.template select<16, 1>(128) = v802_acc;
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v841_n1 = 0; v841_n1 < 9; ++v841_n1) {
                int32_t v842_a = v841_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v844_data(ir1.template select<10, 1>(v842_a));
                tensorforge::intel_esimd::simd<float, 10> v845_data(r0.template select<10, 1>(v842_a));
                r1.template select<10, 1>(v842_a) = (v845_data + v844_data);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v847_i1 = 0; v847_i1 < 9; ++v847_i1) {
                tensorforge::intel_esimd::simd<float, 10> v850_data(r1.template select<10, 1>((v847_i1 * 16)));
                v850_data.copy_to(glb_m0 + ((v847_i1 * 10)));
              }
            }
          }
        }
      }
    });
  });
}


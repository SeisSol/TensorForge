// === base name ===
kernel_13a3c07b64f5e839

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_13a3c07b64f5e839 = {{1, 32, 1}, 16, 10, 1, 32, 46528, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_13a3c07b64f5e839(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_13a3c07b64f5e839(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_13a3c07b64f5e839(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 11632 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_13a3c07b64f5e839(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_13a3c07b64f5e839(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_13a3c07b64f5e839(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_13a3c07b64f5e839(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<11632 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (10 active) x 32 per block = block 1x32x1, 46528 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 16×20(10×17) {0..10}×{1..18} none
        //   m2 20×9(17×9) {1..18}×{0..9} strided
        //   m3 16×20(10×18) {0..10}×{1..19} none
        //   m4 20×9(18×9) {1..19}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":11632}],"shared_bytes":46528,"shared_elements":11632,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[10,19]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[1,0],[19,9]],"name":"m4","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,19]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[19,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (352 * item.get_local_id(1) + 368);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (336);
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
            tensorforge::intel_esimd::simd<float, 16> v35_ld;
            v35_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v35_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 4> v36_ld;
            v36_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 4>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v36_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (160);
          for (size_t v39_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v39_batchId0 < numElements0; v39_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v40_ahead1 = v39_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v42_batchId1 = (v40_ahead1 < numElements0) ? v40_ahead1 : v39_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v39_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v39_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v39_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v39_batchId0 * 162 + 0 + m4_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v50_ld;
              v50_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v50_ld);
              tensorforge::intel_esimd::simd<float, 64> v51_ld;
              v51_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v51_ld);
              tensorforge::intel_esimd::simd<float, 16> v52_ld;
              v52_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v52_ld);
              tensorforge::intel_esimd::simd<float, 9> v53_ld;
              v53_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v53_ld);
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v54_ld;
              v54_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v54_ld);
              tensorforge::intel_esimd::simd<float, 64> v55_ld;
              v55_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 64), v55_ld);
              tensorforge::intel_esimd::simd<float, 32> v56_ld;
              v56_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 128));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 2 * 0 + 128), v56_ld);
              tensorforge::intel_esimd::simd<float, 2> v57_ld;
              v57_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 2>(s1 + (0 + 0 + 1 * 0 + 160), v57_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v63_data = tensorforge::slmLoad<float, 16>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data = tensorforge::slmLoad<float, 16>(glb_m1 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data = tensorforge::slmLoad<float, 16>(glb_m1 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v69_data = tensorforge::slmLoad<float, 16>(glb_m1 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data = tensorforge::slmLoad<float, 16>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v73_data = tensorforge::slmLoad<float, 16>(glb_m1 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data = tensorforge::slmLoad<float, 16>(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v77_data = tensorforge::slmLoad<float, 16>(glb_m1 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v79_data = tensorforge::slmLoad<float, 16>(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v81_data = tensorforge::slmLoad<float, 16>(glb_m1 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v83_data = tensorforge::slmLoad<float, 16>(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v85_data = tensorforge::slmLoad<float, 16>(glb_m1 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v87_data = tensorforge::slmLoad<float, 16>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v89_data = tensorforge::slmLoad<float, 16>(glb_m1 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v91_data = tensorforge::slmLoad<float, 16>(glb_m1 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v93_data = tensorforge::slmLoad<float, 16>(glb_m1 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v95_data = tensorforge::slmLoad<float, 16>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v96_acc{};
              tensorforge::intel_esimd::simd<float, 16> v99_data(0.0f);
              v99_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v96_acc += ((static_cast<float>(v99_data[1])) * v63_data);
              v96_acc += ((static_cast<float>(v99_data[2])) * v65_data);
              v96_acc += ((static_cast<float>(v99_data[3])) * v67_data);
              v96_acc += ((static_cast<float>(v99_data[4])) * v69_data);
              v96_acc += ((static_cast<float>(v99_data[5])) * v71_data);
              v96_acc += ((static_cast<float>(v99_data[6])) * v73_data);
              v96_acc += ((static_cast<float>(v99_data[7])) * v75_data);
              v96_acc += ((static_cast<float>(v99_data[8])) * v77_data);
              v96_acc += ((static_cast<float>(v99_data[9])) * v79_data);
              v96_acc += ((static_cast<float>(v99_data[10])) * v81_data);
              v96_acc += ((static_cast<float>(v99_data[11])) * v83_data);
              v96_acc += ((static_cast<float>(v99_data[12])) * v85_data);
              v96_acc += ((static_cast<float>(v99_data[13])) * v87_data);
              v96_acc += ((static_cast<float>(v99_data[14])) * v89_data);
              v96_acc += ((static_cast<float>(v99_data[15])) * v91_data);
              tensorforge::intel_esimd::simd<float, 16> v135_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v96_acc += ((static_cast<float>(v135_data[0])) * v93_data);
              v96_acc += ((static_cast<float>(v135_data[1])) * v95_data);
              ir0.template select<16, 1>(0) = v96_acc;
              tensorforge::intel_esimd::simd<float, 16> v140_acc{};
              tensorforge::intel_esimd::simd<float, 16> v142_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v140_acc += ((static_cast<float>(v142_data[1])) * v63_data);
              v140_acc += ((static_cast<float>(v142_data[2])) * v65_data);
              v140_acc += ((static_cast<float>(v142_data[3])) * v67_data);
              v140_acc += ((static_cast<float>(v142_data[4])) * v69_data);
              v140_acc += ((static_cast<float>(v142_data[5])) * v71_data);
              v140_acc += ((static_cast<float>(v142_data[6])) * v73_data);
              v140_acc += ((static_cast<float>(v142_data[7])) * v75_data);
              v140_acc += ((static_cast<float>(v142_data[8])) * v77_data);
              v140_acc += ((static_cast<float>(v142_data[9])) * v79_data);
              v140_acc += ((static_cast<float>(v142_data[10])) * v81_data);
              v140_acc += ((static_cast<float>(v142_data[11])) * v83_data);
              v140_acc += ((static_cast<float>(v142_data[12])) * v85_data);
              v140_acc += ((static_cast<float>(v142_data[13])) * v87_data);
              v140_acc += ((static_cast<float>(v142_data[14])) * v89_data);
              v140_acc += ((static_cast<float>(v142_data[15])) * v91_data);
              tensorforge::intel_esimd::simd<float, 16> v175_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v140_acc += ((static_cast<float>(v175_data[0])) * v93_data);
              v140_acc += ((static_cast<float>(v175_data[1])) * v95_data);
              ir0.template select<16, 1>(16) = v140_acc;
              tensorforge::intel_esimd::simd<float, 16> v180_acc{};
              tensorforge::intel_esimd::simd<float, 16> v182_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v180_acc += ((static_cast<float>(v182_data[1])) * v63_data);
              v180_acc += ((static_cast<float>(v182_data[2])) * v65_data);
              v180_acc += ((static_cast<float>(v182_data[3])) * v67_data);
              v180_acc += ((static_cast<float>(v182_data[4])) * v69_data);
              v180_acc += ((static_cast<float>(v182_data[5])) * v71_data);
              v180_acc += ((static_cast<float>(v182_data[6])) * v73_data);
              v180_acc += ((static_cast<float>(v182_data[7])) * v75_data);
              v180_acc += ((static_cast<float>(v182_data[8])) * v77_data);
              v180_acc += ((static_cast<float>(v182_data[9])) * v79_data);
              v180_acc += ((static_cast<float>(v182_data[10])) * v81_data);
              v180_acc += ((static_cast<float>(v182_data[11])) * v83_data);
              v180_acc += ((static_cast<float>(v182_data[12])) * v85_data);
              v180_acc += ((static_cast<float>(v182_data[13])) * v87_data);
              v180_acc += ((static_cast<float>(v182_data[14])) * v89_data);
              v180_acc += ((static_cast<float>(v182_data[15])) * v91_data);
              tensorforge::intel_esimd::simd<float, 16> v215_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v180_acc += ((static_cast<float>(v215_data[0])) * v93_data);
              v180_acc += ((static_cast<float>(v215_data[1])) * v95_data);
              ir0.template select<16, 1>(32) = v180_acc;
              tensorforge::intel_esimd::simd<float, 16> v220_acc{};
              tensorforge::intel_esimd::simd<float, 16> v222_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v220_acc += ((static_cast<float>(v222_data[1])) * v63_data);
              v220_acc += ((static_cast<float>(v222_data[2])) * v65_data);
              v220_acc += ((static_cast<float>(v222_data[3])) * v67_data);
              v220_acc += ((static_cast<float>(v222_data[4])) * v69_data);
              v220_acc += ((static_cast<float>(v222_data[5])) * v71_data);
              v220_acc += ((static_cast<float>(v222_data[6])) * v73_data);
              v220_acc += ((static_cast<float>(v222_data[7])) * v75_data);
              v220_acc += ((static_cast<float>(v222_data[8])) * v77_data);
              v220_acc += ((static_cast<float>(v222_data[9])) * v79_data);
              v220_acc += ((static_cast<float>(v222_data[10])) * v81_data);
              v220_acc += ((static_cast<float>(v222_data[11])) * v83_data);
              v220_acc += ((static_cast<float>(v222_data[12])) * v85_data);
              v220_acc += ((static_cast<float>(v222_data[13])) * v87_data);
              v220_acc += ((static_cast<float>(v222_data[14])) * v89_data);
              v220_acc += ((static_cast<float>(v222_data[15])) * v91_data);
              tensorforge::intel_esimd::simd<float, 16> v255_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v220_acc += ((static_cast<float>(v255_data[0])) * v93_data);
              v220_acc += ((static_cast<float>(v255_data[1])) * v95_data);
              ir0.template select<16, 1>(48) = v220_acc;
              tensorforge::intel_esimd::simd<float, 16> v260_acc{};
              tensorforge::intel_esimd::simd<float, 16> v262_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v260_acc += ((static_cast<float>(v262_data[1])) * v63_data);
              v260_acc += ((static_cast<float>(v262_data[2])) * v65_data);
              v260_acc += ((static_cast<float>(v262_data[3])) * v67_data);
              v260_acc += ((static_cast<float>(v262_data[4])) * v69_data);
              v260_acc += ((static_cast<float>(v262_data[5])) * v71_data);
              v260_acc += ((static_cast<float>(v262_data[6])) * v73_data);
              v260_acc += ((static_cast<float>(v262_data[7])) * v75_data);
              v260_acc += ((static_cast<float>(v262_data[8])) * v77_data);
              v260_acc += ((static_cast<float>(v262_data[9])) * v79_data);
              v260_acc += ((static_cast<float>(v262_data[10])) * v81_data);
              v260_acc += ((static_cast<float>(v262_data[11])) * v83_data);
              v260_acc += ((static_cast<float>(v262_data[12])) * v85_data);
              v260_acc += ((static_cast<float>(v262_data[13])) * v87_data);
              v260_acc += ((static_cast<float>(v262_data[14])) * v89_data);
              v260_acc += ((static_cast<float>(v262_data[15])) * v91_data);
              tensorforge::intel_esimd::simd<float, 16> v295_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v260_acc += ((static_cast<float>(v295_data[0])) * v93_data);
              v260_acc += ((static_cast<float>(v295_data[1])) * v95_data);
              ir0.template select<16, 1>(64) = v260_acc;
              tensorforge::intel_esimd::simd<float, 16> v300_acc{};
              tensorforge::intel_esimd::simd<float, 16> v302_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v300_acc += ((static_cast<float>(v302_data[1])) * v63_data);
              v300_acc += ((static_cast<float>(v302_data[2])) * v65_data);
              v300_acc += ((static_cast<float>(v302_data[3])) * v67_data);
              v300_acc += ((static_cast<float>(v302_data[4])) * v69_data);
              v300_acc += ((static_cast<float>(v302_data[5])) * v71_data);
              v300_acc += ((static_cast<float>(v302_data[6])) * v73_data);
              v300_acc += ((static_cast<float>(v302_data[7])) * v75_data);
              v300_acc += ((static_cast<float>(v302_data[8])) * v77_data);
              v300_acc += ((static_cast<float>(v302_data[9])) * v79_data);
              v300_acc += ((static_cast<float>(v302_data[10])) * v81_data);
              v300_acc += ((static_cast<float>(v302_data[11])) * v83_data);
              v300_acc += ((static_cast<float>(v302_data[12])) * v85_data);
              v300_acc += ((static_cast<float>(v302_data[13])) * v87_data);
              v300_acc += ((static_cast<float>(v302_data[14])) * v89_data);
              v300_acc += ((static_cast<float>(v302_data[15])) * v91_data);
              tensorforge::intel_esimd::simd<float, 16> v335_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v300_acc += ((static_cast<float>(v335_data[0])) * v93_data);
              v300_acc += ((static_cast<float>(v335_data[1])) * v95_data);
              ir0.template select<16, 1>(80) = v300_acc;
              tensorforge::intel_esimd::simd<float, 16> v340_acc{};
              tensorforge::intel_esimd::simd<float, 16> v342_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v340_acc += ((static_cast<float>(v342_data[1])) * v63_data);
              v340_acc += ((static_cast<float>(v342_data[2])) * v65_data);
              v340_acc += ((static_cast<float>(v342_data[3])) * v67_data);
              v340_acc += ((static_cast<float>(v342_data[4])) * v69_data);
              v340_acc += ((static_cast<float>(v342_data[5])) * v71_data);
              v340_acc += ((static_cast<float>(v342_data[6])) * v73_data);
              v340_acc += ((static_cast<float>(v342_data[7])) * v75_data);
              v340_acc += ((static_cast<float>(v342_data[8])) * v77_data);
              v340_acc += ((static_cast<float>(v342_data[9])) * v79_data);
              v340_acc += ((static_cast<float>(v342_data[10])) * v81_data);
              v340_acc += ((static_cast<float>(v342_data[11])) * v83_data);
              v340_acc += ((static_cast<float>(v342_data[12])) * v85_data);
              v340_acc += ((static_cast<float>(v342_data[13])) * v87_data);
              v340_acc += ((static_cast<float>(v342_data[14])) * v89_data);
              v340_acc += ((static_cast<float>(v342_data[15])) * v91_data);
              tensorforge::intel_esimd::simd<float, 16> v375_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v340_acc += ((static_cast<float>(v375_data[0])) * v93_data);
              v340_acc += ((static_cast<float>(v375_data[1])) * v95_data);
              ir0.template select<16, 1>(96) = v340_acc;
              tensorforge::intel_esimd::simd<float, 16> v380_acc{};
              tensorforge::intel_esimd::simd<float, 16> v382_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v380_acc += ((static_cast<float>(v382_data[1])) * v63_data);
              v380_acc += ((static_cast<float>(v382_data[2])) * v65_data);
              v380_acc += ((static_cast<float>(v382_data[3])) * v67_data);
              v380_acc += ((static_cast<float>(v382_data[4])) * v69_data);
              v380_acc += ((static_cast<float>(v382_data[5])) * v71_data);
              v380_acc += ((static_cast<float>(v382_data[6])) * v73_data);
              v380_acc += ((static_cast<float>(v382_data[7])) * v75_data);
              v380_acc += ((static_cast<float>(v382_data[8])) * v77_data);
              v380_acc += ((static_cast<float>(v382_data[9])) * v79_data);
              v380_acc += ((static_cast<float>(v382_data[10])) * v81_data);
              v380_acc += ((static_cast<float>(v382_data[11])) * v83_data);
              v380_acc += ((static_cast<float>(v382_data[12])) * v85_data);
              v380_acc += ((static_cast<float>(v382_data[13])) * v87_data);
              v380_acc += ((static_cast<float>(v382_data[14])) * v89_data);
              v380_acc += ((static_cast<float>(v382_data[15])) * v91_data);
              tensorforge::intel_esimd::simd<float, 16> v415_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v380_acc += ((static_cast<float>(v415_data[0])) * v93_data);
              v380_acc += ((static_cast<float>(v415_data[1])) * v95_data);
              ir0.template select<16, 1>(112) = v380_acc;
              tensorforge::intel_esimd::simd<float, 16> v420_acc{};
              tensorforge::intel_esimd::simd<float, 16> v422_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v420_acc += ((static_cast<float>(v422_data[1])) * v63_data);
              v420_acc += ((static_cast<float>(v422_data[2])) * v65_data);
              v420_acc += ((static_cast<float>(v422_data[3])) * v67_data);
              v420_acc += ((static_cast<float>(v422_data[4])) * v69_data);
              v420_acc += ((static_cast<float>(v422_data[5])) * v71_data);
              v420_acc += ((static_cast<float>(v422_data[6])) * v73_data);
              v420_acc += ((static_cast<float>(v422_data[7])) * v75_data);
              v420_acc += ((static_cast<float>(v422_data[8])) * v77_data);
              v420_acc += ((static_cast<float>(v422_data[9])) * v79_data);
              v420_acc += ((static_cast<float>(v422_data[10])) * v81_data);
              v420_acc += ((static_cast<float>(v422_data[11])) * v83_data);
              v420_acc += ((static_cast<float>(v422_data[12])) * v85_data);
              v420_acc += ((static_cast<float>(v422_data[13])) * v87_data);
              v420_acc += ((static_cast<float>(v422_data[14])) * v89_data);
              v420_acc += ((static_cast<float>(v422_data[15])) * v91_data);
              tensorforge::intel_esimd::simd<float, 16> v455_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v420_acc += ((static_cast<float>(v455_data[0])) * v93_data);
              v420_acc += ((static_cast<float>(v455_data[1])) * v95_data);
              ir0.template select<16, 1>(128) = v420_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v460_n1 = 0; v460_n1 < 9; ++v460_n1) {
                int32_t v461_a = v460_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v463_data(ir0.template select<10, 1>(v461_a));
                r0.template select<10, 1>(v461_a) = v463_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 10), (0, 9)] [(1, 19)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v469_data = tensorforge::slmLoad<float, 16>(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v471_data = tensorforge::slmLoad<float, 16>(glb_m3 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v473_data = tensorforge::slmLoad<float, 16>(glb_m3 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v475_data = tensorforge::slmLoad<float, 16>(glb_m3 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v477_data = tensorforge::slmLoad<float, 16>(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v479_data = tensorforge::slmLoad<float, 16>(glb_m3 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v481_data = tensorforge::slmLoad<float, 16>(glb_m3 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v483_data = tensorforge::slmLoad<float, 16>(glb_m3 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v485_data = tensorforge::slmLoad<float, 16>(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v487_data = tensorforge::slmLoad<float, 16>(glb_m3 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v489_data = tensorforge::slmLoad<float, 16>(glb_m3 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v491_data = tensorforge::slmLoad<float, 16>(glb_m3 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v493_data = tensorforge::slmLoad<float, 16>(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v495_data = tensorforge::slmLoad<float, 16>(glb_m3 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v497_data = tensorforge::slmLoad<float, 16>(glb_m3 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v499_data = tensorforge::slmLoad<float, 16>(glb_m3 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v501_data = tensorforge::slmLoad<float, 16>(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v503_data = tensorforge::slmLoad<float, 16>(glb_m3 + (170_i32));
              tensorforge::intel_esimd::simd<float, 16> v504_acc{};
              tensorforge::intel_esimd::simd<float, 16> v507_data(0.0f);
              v507_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s1 + (-1_i32)) + 1);
              v504_acc += ((static_cast<float>(v507_data[1])) * v469_data);
              v504_acc += ((static_cast<float>(v507_data[2])) * v471_data);
              v504_acc += ((static_cast<float>(v507_data[3])) * v473_data);
              v504_acc += ((static_cast<float>(v507_data[4])) * v475_data);
              v504_acc += ((static_cast<float>(v507_data[5])) * v477_data);
              v504_acc += ((static_cast<float>(v507_data[6])) * v479_data);
              v504_acc += ((static_cast<float>(v507_data[7])) * v481_data);
              v504_acc += ((static_cast<float>(v507_data[8])) * v483_data);
              v504_acc += ((static_cast<float>(v507_data[9])) * v485_data);
              v504_acc += ((static_cast<float>(v507_data[10])) * v487_data);
              v504_acc += ((static_cast<float>(v507_data[11])) * v489_data);
              v504_acc += ((static_cast<float>(v507_data[12])) * v491_data);
              v504_acc += ((static_cast<float>(v507_data[13])) * v493_data);
              v504_acc += ((static_cast<float>(v507_data[14])) * v495_data);
              v504_acc += ((static_cast<float>(v507_data[15])) * v497_data);
              tensorforge::intel_esimd::simd<float, 16> v543_data = tensorforge::slmLoad<float, 16>(s1 + (15_i32));
              v504_acc += ((static_cast<float>(v543_data[0])) * v499_data);
              v504_acc += ((static_cast<float>(v543_data[1])) * v501_data);
              v504_acc += ((static_cast<float>(v543_data[2])) * v503_data);
              ir1.template select<16, 1>(0) = v504_acc;
              tensorforge::intel_esimd::simd<float, 16> v550_acc{};
              tensorforge::intel_esimd::simd<float, 16> v552_data = tensorforge::slmLoad<float, 16>(s1 + (17_i32));
              v550_acc += ((static_cast<float>(v552_data[1])) * v469_data);
              v550_acc += ((static_cast<float>(v552_data[2])) * v471_data);
              v550_acc += ((static_cast<float>(v552_data[3])) * v473_data);
              v550_acc += ((static_cast<float>(v552_data[4])) * v475_data);
              v550_acc += ((static_cast<float>(v552_data[5])) * v477_data);
              v550_acc += ((static_cast<float>(v552_data[6])) * v479_data);
              v550_acc += ((static_cast<float>(v552_data[7])) * v481_data);
              v550_acc += ((static_cast<float>(v552_data[8])) * v483_data);
              v550_acc += ((static_cast<float>(v552_data[9])) * v485_data);
              v550_acc += ((static_cast<float>(v552_data[10])) * v487_data);
              v550_acc += ((static_cast<float>(v552_data[11])) * v489_data);
              v550_acc += ((static_cast<float>(v552_data[12])) * v491_data);
              v550_acc += ((static_cast<float>(v552_data[13])) * v493_data);
              v550_acc += ((static_cast<float>(v552_data[14])) * v495_data);
              v550_acc += ((static_cast<float>(v552_data[15])) * v497_data);
              tensorforge::intel_esimd::simd<float, 16> v585_data = tensorforge::slmLoad<float, 16>(s1 + (33_i32));
              v550_acc += ((static_cast<float>(v585_data[0])) * v499_data);
              v550_acc += ((static_cast<float>(v585_data[1])) * v501_data);
              v550_acc += ((static_cast<float>(v585_data[2])) * v503_data);
              ir1.template select<16, 1>(16) = v550_acc;
              tensorforge::intel_esimd::simd<float, 16> v592_acc{};
              tensorforge::intel_esimd::simd<float, 16> v594_data = tensorforge::slmLoad<float, 16>(s1 + (35_i32));
              v592_acc += ((static_cast<float>(v594_data[1])) * v469_data);
              v592_acc += ((static_cast<float>(v594_data[2])) * v471_data);
              v592_acc += ((static_cast<float>(v594_data[3])) * v473_data);
              v592_acc += ((static_cast<float>(v594_data[4])) * v475_data);
              v592_acc += ((static_cast<float>(v594_data[5])) * v477_data);
              v592_acc += ((static_cast<float>(v594_data[6])) * v479_data);
              v592_acc += ((static_cast<float>(v594_data[7])) * v481_data);
              v592_acc += ((static_cast<float>(v594_data[8])) * v483_data);
              v592_acc += ((static_cast<float>(v594_data[9])) * v485_data);
              v592_acc += ((static_cast<float>(v594_data[10])) * v487_data);
              v592_acc += ((static_cast<float>(v594_data[11])) * v489_data);
              v592_acc += ((static_cast<float>(v594_data[12])) * v491_data);
              v592_acc += ((static_cast<float>(v594_data[13])) * v493_data);
              v592_acc += ((static_cast<float>(v594_data[14])) * v495_data);
              v592_acc += ((static_cast<float>(v594_data[15])) * v497_data);
              tensorforge::intel_esimd::simd<float, 16> v627_data = tensorforge::slmLoad<float, 16>(s1 + (51_i32));
              v592_acc += ((static_cast<float>(v627_data[0])) * v499_data);
              v592_acc += ((static_cast<float>(v627_data[1])) * v501_data);
              v592_acc += ((static_cast<float>(v627_data[2])) * v503_data);
              ir1.template select<16, 1>(32) = v592_acc;
              tensorforge::intel_esimd::simd<float, 16> v634_acc{};
              tensorforge::intel_esimd::simd<float, 16> v636_data = tensorforge::slmLoad<float, 16>(s1 + (53_i32));
              v634_acc += ((static_cast<float>(v636_data[1])) * v469_data);
              v634_acc += ((static_cast<float>(v636_data[2])) * v471_data);
              v634_acc += ((static_cast<float>(v636_data[3])) * v473_data);
              v634_acc += ((static_cast<float>(v636_data[4])) * v475_data);
              v634_acc += ((static_cast<float>(v636_data[5])) * v477_data);
              v634_acc += ((static_cast<float>(v636_data[6])) * v479_data);
              v634_acc += ((static_cast<float>(v636_data[7])) * v481_data);
              v634_acc += ((static_cast<float>(v636_data[8])) * v483_data);
              v634_acc += ((static_cast<float>(v636_data[9])) * v485_data);
              v634_acc += ((static_cast<float>(v636_data[10])) * v487_data);
              v634_acc += ((static_cast<float>(v636_data[11])) * v489_data);
              v634_acc += ((static_cast<float>(v636_data[12])) * v491_data);
              v634_acc += ((static_cast<float>(v636_data[13])) * v493_data);
              v634_acc += ((static_cast<float>(v636_data[14])) * v495_data);
              v634_acc += ((static_cast<float>(v636_data[15])) * v497_data);
              tensorforge::intel_esimd::simd<float, 16> v669_data = tensorforge::slmLoad<float, 16>(s1 + (69_i32));
              v634_acc += ((static_cast<float>(v669_data[0])) * v499_data);
              v634_acc += ((static_cast<float>(v669_data[1])) * v501_data);
              v634_acc += ((static_cast<float>(v669_data[2])) * v503_data);
              ir1.template select<16, 1>(48) = v634_acc;
              tensorforge::intel_esimd::simd<float, 16> v676_acc{};
              tensorforge::intel_esimd::simd<float, 16> v678_data = tensorforge::slmLoad<float, 16>(s1 + (71_i32));
              v676_acc += ((static_cast<float>(v678_data[1])) * v469_data);
              v676_acc += ((static_cast<float>(v678_data[2])) * v471_data);
              v676_acc += ((static_cast<float>(v678_data[3])) * v473_data);
              v676_acc += ((static_cast<float>(v678_data[4])) * v475_data);
              v676_acc += ((static_cast<float>(v678_data[5])) * v477_data);
              v676_acc += ((static_cast<float>(v678_data[6])) * v479_data);
              v676_acc += ((static_cast<float>(v678_data[7])) * v481_data);
              v676_acc += ((static_cast<float>(v678_data[8])) * v483_data);
              v676_acc += ((static_cast<float>(v678_data[9])) * v485_data);
              v676_acc += ((static_cast<float>(v678_data[10])) * v487_data);
              v676_acc += ((static_cast<float>(v678_data[11])) * v489_data);
              v676_acc += ((static_cast<float>(v678_data[12])) * v491_data);
              v676_acc += ((static_cast<float>(v678_data[13])) * v493_data);
              v676_acc += ((static_cast<float>(v678_data[14])) * v495_data);
              v676_acc += ((static_cast<float>(v678_data[15])) * v497_data);
              tensorforge::intel_esimd::simd<float, 16> v711_data = tensorforge::slmLoad<float, 16>(s1 + (87_i32));
              v676_acc += ((static_cast<float>(v711_data[0])) * v499_data);
              v676_acc += ((static_cast<float>(v711_data[1])) * v501_data);
              v676_acc += ((static_cast<float>(v711_data[2])) * v503_data);
              ir1.template select<16, 1>(64) = v676_acc;
              tensorforge::intel_esimd::simd<float, 16> v718_acc{};
              tensorforge::intel_esimd::simd<float, 16> v720_data = tensorforge::slmLoad<float, 16>(s1 + (89_i32));
              v718_acc += ((static_cast<float>(v720_data[1])) * v469_data);
              v718_acc += ((static_cast<float>(v720_data[2])) * v471_data);
              v718_acc += ((static_cast<float>(v720_data[3])) * v473_data);
              v718_acc += ((static_cast<float>(v720_data[4])) * v475_data);
              v718_acc += ((static_cast<float>(v720_data[5])) * v477_data);
              v718_acc += ((static_cast<float>(v720_data[6])) * v479_data);
              v718_acc += ((static_cast<float>(v720_data[7])) * v481_data);
              v718_acc += ((static_cast<float>(v720_data[8])) * v483_data);
              v718_acc += ((static_cast<float>(v720_data[9])) * v485_data);
              v718_acc += ((static_cast<float>(v720_data[10])) * v487_data);
              v718_acc += ((static_cast<float>(v720_data[11])) * v489_data);
              v718_acc += ((static_cast<float>(v720_data[12])) * v491_data);
              v718_acc += ((static_cast<float>(v720_data[13])) * v493_data);
              v718_acc += ((static_cast<float>(v720_data[14])) * v495_data);
              v718_acc += ((static_cast<float>(v720_data[15])) * v497_data);
              tensorforge::intel_esimd::simd<float, 16> v753_data = tensorforge::slmLoad<float, 16>(s1 + (105_i32));
              v718_acc += ((static_cast<float>(v753_data[0])) * v499_data);
              v718_acc += ((static_cast<float>(v753_data[1])) * v501_data);
              v718_acc += ((static_cast<float>(v753_data[2])) * v503_data);
              ir1.template select<16, 1>(80) = v718_acc;
              tensorforge::intel_esimd::simd<float, 16> v760_acc{};
              tensorforge::intel_esimd::simd<float, 16> v762_data = tensorforge::slmLoad<float, 16>(s1 + (107_i32));
              v760_acc += ((static_cast<float>(v762_data[1])) * v469_data);
              v760_acc += ((static_cast<float>(v762_data[2])) * v471_data);
              v760_acc += ((static_cast<float>(v762_data[3])) * v473_data);
              v760_acc += ((static_cast<float>(v762_data[4])) * v475_data);
              v760_acc += ((static_cast<float>(v762_data[5])) * v477_data);
              v760_acc += ((static_cast<float>(v762_data[6])) * v479_data);
              v760_acc += ((static_cast<float>(v762_data[7])) * v481_data);
              v760_acc += ((static_cast<float>(v762_data[8])) * v483_data);
              v760_acc += ((static_cast<float>(v762_data[9])) * v485_data);
              v760_acc += ((static_cast<float>(v762_data[10])) * v487_data);
              v760_acc += ((static_cast<float>(v762_data[11])) * v489_data);
              v760_acc += ((static_cast<float>(v762_data[12])) * v491_data);
              v760_acc += ((static_cast<float>(v762_data[13])) * v493_data);
              v760_acc += ((static_cast<float>(v762_data[14])) * v495_data);
              v760_acc += ((static_cast<float>(v762_data[15])) * v497_data);
              tensorforge::intel_esimd::simd<float, 16> v795_data = tensorforge::slmLoad<float, 16>(s1 + (123_i32));
              v760_acc += ((static_cast<float>(v795_data[0])) * v499_data);
              v760_acc += ((static_cast<float>(v795_data[1])) * v501_data);
              v760_acc += ((static_cast<float>(v795_data[2])) * v503_data);
              ir1.template select<16, 1>(96) = v760_acc;
              tensorforge::intel_esimd::simd<float, 16> v802_acc{};
              tensorforge::intel_esimd::simd<float, 16> v804_data = tensorforge::slmLoad<float, 16>(s1 + (125_i32));
              v802_acc += ((static_cast<float>(v804_data[1])) * v469_data);
              v802_acc += ((static_cast<float>(v804_data[2])) * v471_data);
              v802_acc += ((static_cast<float>(v804_data[3])) * v473_data);
              v802_acc += ((static_cast<float>(v804_data[4])) * v475_data);
              v802_acc += ((static_cast<float>(v804_data[5])) * v477_data);
              v802_acc += ((static_cast<float>(v804_data[6])) * v479_data);
              v802_acc += ((static_cast<float>(v804_data[7])) * v481_data);
              v802_acc += ((static_cast<float>(v804_data[8])) * v483_data);
              v802_acc += ((static_cast<float>(v804_data[9])) * v485_data);
              v802_acc += ((static_cast<float>(v804_data[10])) * v487_data);
              v802_acc += ((static_cast<float>(v804_data[11])) * v489_data);
              v802_acc += ((static_cast<float>(v804_data[12])) * v491_data);
              v802_acc += ((static_cast<float>(v804_data[13])) * v493_data);
              v802_acc += ((static_cast<float>(v804_data[14])) * v495_data);
              v802_acc += ((static_cast<float>(v804_data[15])) * v497_data);
              tensorforge::intel_esimd::simd<float, 16> v837_data = tensorforge::slmLoad<float, 16>(s1 + (141_i32));
              v802_acc += ((static_cast<float>(v837_data[0])) * v499_data);
              v802_acc += ((static_cast<float>(v837_data[1])) * v501_data);
              v802_acc += ((static_cast<float>(v837_data[2])) * v503_data);
              ir1.template select<16, 1>(112) = v802_acc;
              tensorforge::intel_esimd::simd<float, 16> v844_acc{};
              tensorforge::intel_esimd::simd<float, 16> v846_data = tensorforge::slmLoad<float, 16>(s1 + (143_i32));
              v844_acc += ((static_cast<float>(v846_data[1])) * v469_data);
              v844_acc += ((static_cast<float>(v846_data[2])) * v471_data);
              v844_acc += ((static_cast<float>(v846_data[3])) * v473_data);
              v844_acc += ((static_cast<float>(v846_data[4])) * v475_data);
              v844_acc += ((static_cast<float>(v846_data[5])) * v477_data);
              v844_acc += ((static_cast<float>(v846_data[6])) * v479_data);
              v844_acc += ((static_cast<float>(v846_data[7])) * v481_data);
              v844_acc += ((static_cast<float>(v846_data[8])) * v483_data);
              v844_acc += ((static_cast<float>(v846_data[9])) * v485_data);
              v844_acc += ((static_cast<float>(v846_data[10])) * v487_data);
              v844_acc += ((static_cast<float>(v846_data[11])) * v489_data);
              v844_acc += ((static_cast<float>(v846_data[12])) * v491_data);
              v844_acc += ((static_cast<float>(v846_data[13])) * v493_data);
              v844_acc += ((static_cast<float>(v846_data[14])) * v495_data);
              v844_acc += ((static_cast<float>(v846_data[15])) * v497_data);
              tensorforge::intel_esimd::simd<float, 16> v879_data = tensorforge::slmLoad<float, 16>(s1 + (159_i32));
              v844_acc += ((static_cast<float>(v879_data[0])) * v499_data);
              v844_acc += ((static_cast<float>(v879_data[1])) * v501_data);
              v844_acc += ((static_cast<float>(v879_data[2])) * v503_data);
              ir1.template select<16, 1>(128) = v844_acc;
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v886_n1 = 0; v886_n1 < 9; ++v886_n1) {
                int32_t v887_a = v886_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v889_data(ir1.template select<10, 1>(v887_a));
                tensorforge::intel_esimd::simd<float, 10> v890_data(r0.template select<10, 1>(v887_a));
                r1.template select<10, 1>(v887_a) = (v890_data + v889_data);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v892_i1 = 0; v892_i1 < 9; ++v892_i1) {
                tensorforge::intel_esimd::simd<float, 10> v895_data(r1.template select<10, 1>((v892_i1 * 16)));
                v895_data.copy_to(glb_m0 + ((v892_i1 * 10)));
              }
            }
          }
        }
      }
    });
  });
}


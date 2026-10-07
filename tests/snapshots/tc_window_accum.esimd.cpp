// === base name ===
kernel_ed651cc06983b666

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ed651cc06983b666 = {{1, 32, 1}, 16, 10, 1, 32, 46528, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ed651cc06983b666(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ed651cc06983b666(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ed651cc06983b666(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_ed651cc06983b666(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ed651cc06983b666(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ed651cc06983b666(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ed651cc06983b666(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v9_ld;
            v9_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v9_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v10_ld;
            v10_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v10_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v11_ld;
            v11_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v11_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v16_ld;
            v16_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v17_ld;
            v17_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 10> v19_ld;
            v19_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 10>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          const float *const __restrict__ ptr_glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> glb_m3 = totalShrMem + (176);
          // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v22_ld;
            v22_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v22_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v23_ld;
            v23_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v23_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v24_ld;
            v24_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v24_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v25_ld;
            v25_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v25_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v26_ld;
            v26_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v26_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v27_ld;
            v27_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v27_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v28_ld;
            v28_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v28_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v29_ld;
            v29_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v29_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v30_ld;
            v30_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v30_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v31_ld;
            v31_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v31_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 16> v32_ld;
            v32_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v32_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 4> v33_ld;
            v33_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 4>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v33_ld);
          }
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (176);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v36_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v36_batchId0 < numElements0; v36_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v37_ahead1 = v36_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v39_batchId1 = (v37_ahead1 < numElements0) ? v37_ahead1 : v36_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v36_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v36_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v36_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v36_batchId0 * 162 + 0 + m4_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v47_ld;
              v47_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v47_ld);
              tensorforge::intel_esimd::simd<float, 64> v48_ld;
              v48_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v48_ld);
              tensorforge::intel_esimd::simd<float, 16> v49_ld;
              v49_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v49_ld);
              tensorforge::intel_esimd::simd<float, 9> v50_ld;
              v50_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v50_ld);
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v457_ld;
              v457_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v457_ld);
              tensorforge::intel_esimd::simd<float, 64> v458_ld;
              v458_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 64), v458_ld);
              tensorforge::intel_esimd::simd<float, 32> v459_ld;
              v459_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 128));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 2 * 0 + 128), v459_ld);
              tensorforge::intel_esimd::simd<float, 2> v460_ld;
              v460_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 2>(s1 + (0 + 0 + 1 * 0 + 160), v460_ld);
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(1, 18)]
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
              tensorforge::intel_esimd::simd<float, 16> v92_data(0.0f);
              v92_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v89_acc += ((static_cast<float>(v92_data[1])) * v56_data);
              v89_acc += ((static_cast<float>(v92_data[2])) * v58_data);
              v89_acc += ((static_cast<float>(v92_data[3])) * v60_data);
              v89_acc += ((static_cast<float>(v92_data[4])) * v62_data);
              v89_acc += ((static_cast<float>(v92_data[5])) * v64_data);
              v89_acc += ((static_cast<float>(v92_data[6])) * v66_data);
              v89_acc += ((static_cast<float>(v92_data[7])) * v68_data);
              v89_acc += ((static_cast<float>(v92_data[8])) * v70_data);
              v89_acc += ((static_cast<float>(v92_data[9])) * v72_data);
              v89_acc += ((static_cast<float>(v92_data[10])) * v74_data);
              v89_acc += ((static_cast<float>(v92_data[11])) * v76_data);
              v89_acc += ((static_cast<float>(v92_data[12])) * v78_data);
              v89_acc += ((static_cast<float>(v92_data[13])) * v80_data);
              v89_acc += ((static_cast<float>(v92_data[14])) * v82_data);
              v89_acc += ((static_cast<float>(v92_data[15])) * v84_data);
              tensorforge::intel_esimd::simd<float, 16> v128_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v89_acc += ((static_cast<float>(v128_data[0])) * v86_data);
              v89_acc += ((static_cast<float>(v128_data[1])) * v88_data);
              ir0.template select<16, 1>(0) = v89_acc;
              tensorforge::intel_esimd::simd<float, 16> v133_acc{};
              tensorforge::intel_esimd::simd<float, 16> v135_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v133_acc += ((static_cast<float>(v135_data[1])) * v56_data);
              v133_acc += ((static_cast<float>(v135_data[2])) * v58_data);
              v133_acc += ((static_cast<float>(v135_data[3])) * v60_data);
              v133_acc += ((static_cast<float>(v135_data[4])) * v62_data);
              v133_acc += ((static_cast<float>(v135_data[5])) * v64_data);
              v133_acc += ((static_cast<float>(v135_data[6])) * v66_data);
              v133_acc += ((static_cast<float>(v135_data[7])) * v68_data);
              v133_acc += ((static_cast<float>(v135_data[8])) * v70_data);
              v133_acc += ((static_cast<float>(v135_data[9])) * v72_data);
              v133_acc += ((static_cast<float>(v135_data[10])) * v74_data);
              v133_acc += ((static_cast<float>(v135_data[11])) * v76_data);
              v133_acc += ((static_cast<float>(v135_data[12])) * v78_data);
              v133_acc += ((static_cast<float>(v135_data[13])) * v80_data);
              v133_acc += ((static_cast<float>(v135_data[14])) * v82_data);
              v133_acc += ((static_cast<float>(v135_data[15])) * v84_data);
              tensorforge::intel_esimd::simd<float, 16> v168_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v133_acc += ((static_cast<float>(v168_data[0])) * v86_data);
              v133_acc += ((static_cast<float>(v168_data[1])) * v88_data);
              ir0.template select<16, 1>(16) = v133_acc;
              tensorforge::intel_esimd::simd<float, 16> v173_acc{};
              tensorforge::intel_esimd::simd<float, 16> v175_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v173_acc += ((static_cast<float>(v175_data[1])) * v56_data);
              v173_acc += ((static_cast<float>(v175_data[2])) * v58_data);
              v173_acc += ((static_cast<float>(v175_data[3])) * v60_data);
              v173_acc += ((static_cast<float>(v175_data[4])) * v62_data);
              v173_acc += ((static_cast<float>(v175_data[5])) * v64_data);
              v173_acc += ((static_cast<float>(v175_data[6])) * v66_data);
              v173_acc += ((static_cast<float>(v175_data[7])) * v68_data);
              v173_acc += ((static_cast<float>(v175_data[8])) * v70_data);
              v173_acc += ((static_cast<float>(v175_data[9])) * v72_data);
              v173_acc += ((static_cast<float>(v175_data[10])) * v74_data);
              v173_acc += ((static_cast<float>(v175_data[11])) * v76_data);
              v173_acc += ((static_cast<float>(v175_data[12])) * v78_data);
              v173_acc += ((static_cast<float>(v175_data[13])) * v80_data);
              v173_acc += ((static_cast<float>(v175_data[14])) * v82_data);
              v173_acc += ((static_cast<float>(v175_data[15])) * v84_data);
              tensorforge::intel_esimd::simd<float, 16> v208_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v173_acc += ((static_cast<float>(v208_data[0])) * v86_data);
              v173_acc += ((static_cast<float>(v208_data[1])) * v88_data);
              ir0.template select<16, 1>(32) = v173_acc;
              tensorforge::intel_esimd::simd<float, 16> v213_acc{};
              tensorforge::intel_esimd::simd<float, 16> v215_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v213_acc += ((static_cast<float>(v215_data[1])) * v56_data);
              v213_acc += ((static_cast<float>(v215_data[2])) * v58_data);
              v213_acc += ((static_cast<float>(v215_data[3])) * v60_data);
              v213_acc += ((static_cast<float>(v215_data[4])) * v62_data);
              v213_acc += ((static_cast<float>(v215_data[5])) * v64_data);
              v213_acc += ((static_cast<float>(v215_data[6])) * v66_data);
              v213_acc += ((static_cast<float>(v215_data[7])) * v68_data);
              v213_acc += ((static_cast<float>(v215_data[8])) * v70_data);
              v213_acc += ((static_cast<float>(v215_data[9])) * v72_data);
              v213_acc += ((static_cast<float>(v215_data[10])) * v74_data);
              v213_acc += ((static_cast<float>(v215_data[11])) * v76_data);
              v213_acc += ((static_cast<float>(v215_data[12])) * v78_data);
              v213_acc += ((static_cast<float>(v215_data[13])) * v80_data);
              v213_acc += ((static_cast<float>(v215_data[14])) * v82_data);
              v213_acc += ((static_cast<float>(v215_data[15])) * v84_data);
              tensorforge::intel_esimd::simd<float, 16> v248_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v213_acc += ((static_cast<float>(v248_data[0])) * v86_data);
              v213_acc += ((static_cast<float>(v248_data[1])) * v88_data);
              ir0.template select<16, 1>(48) = v213_acc;
              tensorforge::intel_esimd::simd<float, 16> v253_acc{};
              tensorforge::intel_esimd::simd<float, 16> v255_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v253_acc += ((static_cast<float>(v255_data[1])) * v56_data);
              v253_acc += ((static_cast<float>(v255_data[2])) * v58_data);
              v253_acc += ((static_cast<float>(v255_data[3])) * v60_data);
              v253_acc += ((static_cast<float>(v255_data[4])) * v62_data);
              v253_acc += ((static_cast<float>(v255_data[5])) * v64_data);
              v253_acc += ((static_cast<float>(v255_data[6])) * v66_data);
              v253_acc += ((static_cast<float>(v255_data[7])) * v68_data);
              v253_acc += ((static_cast<float>(v255_data[8])) * v70_data);
              v253_acc += ((static_cast<float>(v255_data[9])) * v72_data);
              v253_acc += ((static_cast<float>(v255_data[10])) * v74_data);
              v253_acc += ((static_cast<float>(v255_data[11])) * v76_data);
              v253_acc += ((static_cast<float>(v255_data[12])) * v78_data);
              v253_acc += ((static_cast<float>(v255_data[13])) * v80_data);
              v253_acc += ((static_cast<float>(v255_data[14])) * v82_data);
              v253_acc += ((static_cast<float>(v255_data[15])) * v84_data);
              tensorforge::intel_esimd::simd<float, 16> v288_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v253_acc += ((static_cast<float>(v288_data[0])) * v86_data);
              v253_acc += ((static_cast<float>(v288_data[1])) * v88_data);
              ir0.template select<16, 1>(64) = v253_acc;
              tensorforge::intel_esimd::simd<float, 16> v293_acc{};
              tensorforge::intel_esimd::simd<float, 16> v295_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v293_acc += ((static_cast<float>(v295_data[1])) * v56_data);
              v293_acc += ((static_cast<float>(v295_data[2])) * v58_data);
              v293_acc += ((static_cast<float>(v295_data[3])) * v60_data);
              v293_acc += ((static_cast<float>(v295_data[4])) * v62_data);
              v293_acc += ((static_cast<float>(v295_data[5])) * v64_data);
              v293_acc += ((static_cast<float>(v295_data[6])) * v66_data);
              v293_acc += ((static_cast<float>(v295_data[7])) * v68_data);
              v293_acc += ((static_cast<float>(v295_data[8])) * v70_data);
              v293_acc += ((static_cast<float>(v295_data[9])) * v72_data);
              v293_acc += ((static_cast<float>(v295_data[10])) * v74_data);
              v293_acc += ((static_cast<float>(v295_data[11])) * v76_data);
              v293_acc += ((static_cast<float>(v295_data[12])) * v78_data);
              v293_acc += ((static_cast<float>(v295_data[13])) * v80_data);
              v293_acc += ((static_cast<float>(v295_data[14])) * v82_data);
              v293_acc += ((static_cast<float>(v295_data[15])) * v84_data);
              tensorforge::intel_esimd::simd<float, 16> v328_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v293_acc += ((static_cast<float>(v328_data[0])) * v86_data);
              v293_acc += ((static_cast<float>(v328_data[1])) * v88_data);
              ir0.template select<16, 1>(80) = v293_acc;
              tensorforge::intel_esimd::simd<float, 16> v333_acc{};
              tensorforge::intel_esimd::simd<float, 16> v335_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v333_acc += ((static_cast<float>(v335_data[1])) * v56_data);
              v333_acc += ((static_cast<float>(v335_data[2])) * v58_data);
              v333_acc += ((static_cast<float>(v335_data[3])) * v60_data);
              v333_acc += ((static_cast<float>(v335_data[4])) * v62_data);
              v333_acc += ((static_cast<float>(v335_data[5])) * v64_data);
              v333_acc += ((static_cast<float>(v335_data[6])) * v66_data);
              v333_acc += ((static_cast<float>(v335_data[7])) * v68_data);
              v333_acc += ((static_cast<float>(v335_data[8])) * v70_data);
              v333_acc += ((static_cast<float>(v335_data[9])) * v72_data);
              v333_acc += ((static_cast<float>(v335_data[10])) * v74_data);
              v333_acc += ((static_cast<float>(v335_data[11])) * v76_data);
              v333_acc += ((static_cast<float>(v335_data[12])) * v78_data);
              v333_acc += ((static_cast<float>(v335_data[13])) * v80_data);
              v333_acc += ((static_cast<float>(v335_data[14])) * v82_data);
              v333_acc += ((static_cast<float>(v335_data[15])) * v84_data);
              tensorforge::intel_esimd::simd<float, 16> v368_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v333_acc += ((static_cast<float>(v368_data[0])) * v86_data);
              v333_acc += ((static_cast<float>(v368_data[1])) * v88_data);
              ir0.template select<16, 1>(96) = v333_acc;
              tensorforge::intel_esimd::simd<float, 16> v373_acc{};
              tensorforge::intel_esimd::simd<float, 16> v375_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v373_acc += ((static_cast<float>(v375_data[1])) * v56_data);
              v373_acc += ((static_cast<float>(v375_data[2])) * v58_data);
              v373_acc += ((static_cast<float>(v375_data[3])) * v60_data);
              v373_acc += ((static_cast<float>(v375_data[4])) * v62_data);
              v373_acc += ((static_cast<float>(v375_data[5])) * v64_data);
              v373_acc += ((static_cast<float>(v375_data[6])) * v66_data);
              v373_acc += ((static_cast<float>(v375_data[7])) * v68_data);
              v373_acc += ((static_cast<float>(v375_data[8])) * v70_data);
              v373_acc += ((static_cast<float>(v375_data[9])) * v72_data);
              v373_acc += ((static_cast<float>(v375_data[10])) * v74_data);
              v373_acc += ((static_cast<float>(v375_data[11])) * v76_data);
              v373_acc += ((static_cast<float>(v375_data[12])) * v78_data);
              v373_acc += ((static_cast<float>(v375_data[13])) * v80_data);
              v373_acc += ((static_cast<float>(v375_data[14])) * v82_data);
              v373_acc += ((static_cast<float>(v375_data[15])) * v84_data);
              tensorforge::intel_esimd::simd<float, 16> v408_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v373_acc += ((static_cast<float>(v408_data[0])) * v86_data);
              v373_acc += ((static_cast<float>(v408_data[1])) * v88_data);
              ir0.template select<16, 1>(112) = v373_acc;
              tensorforge::intel_esimd::simd<float, 16> v413_acc{};
              tensorforge::intel_esimd::simd<float, 16> v415_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v413_acc += ((static_cast<float>(v415_data[1])) * v56_data);
              v413_acc += ((static_cast<float>(v415_data[2])) * v58_data);
              v413_acc += ((static_cast<float>(v415_data[3])) * v60_data);
              v413_acc += ((static_cast<float>(v415_data[4])) * v62_data);
              v413_acc += ((static_cast<float>(v415_data[5])) * v64_data);
              v413_acc += ((static_cast<float>(v415_data[6])) * v66_data);
              v413_acc += ((static_cast<float>(v415_data[7])) * v68_data);
              v413_acc += ((static_cast<float>(v415_data[8])) * v70_data);
              v413_acc += ((static_cast<float>(v415_data[9])) * v72_data);
              v413_acc += ((static_cast<float>(v415_data[10])) * v74_data);
              v413_acc += ((static_cast<float>(v415_data[11])) * v76_data);
              v413_acc += ((static_cast<float>(v415_data[12])) * v78_data);
              v413_acc += ((static_cast<float>(v415_data[13])) * v80_data);
              v413_acc += ((static_cast<float>(v415_data[14])) * v82_data);
              v413_acc += ((static_cast<float>(v415_data[15])) * v84_data);
              tensorforge::intel_esimd::simd<float, 16> v448_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v413_acc += ((static_cast<float>(v448_data[0])) * v86_data);
              v413_acc += ((static_cast<float>(v448_data[1])) * v88_data);
              ir0.template select<16, 1>(128) = v413_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v453_n1 = 0; v453_n1 < 9; ++v453_n1) {
                int32_t v454_a = v453_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v456_data(ir0.template select<10, 1>(v454_a));
                r0.template select<10, 1>(v454_a) = v456_data;
              }
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 10), (0, 9)] [(1, 19)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v466_data = tensorforge::slmLoad<float, 16>(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v468_data = tensorforge::slmLoad<float, 16>(glb_m3 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v470_data = tensorforge::slmLoad<float, 16>(glb_m3 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v472_data = tensorforge::slmLoad<float, 16>(glb_m3 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v474_data = tensorforge::slmLoad<float, 16>(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v476_data = tensorforge::slmLoad<float, 16>(glb_m3 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v478_data = tensorforge::slmLoad<float, 16>(glb_m3 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v480_data = tensorforge::slmLoad<float, 16>(glb_m3 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v482_data = tensorforge::slmLoad<float, 16>(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v484_data = tensorforge::slmLoad<float, 16>(glb_m3 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v486_data = tensorforge::slmLoad<float, 16>(glb_m3 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v488_data = tensorforge::slmLoad<float, 16>(glb_m3 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v490_data = tensorforge::slmLoad<float, 16>(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v492_data = tensorforge::slmLoad<float, 16>(glb_m3 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v494_data = tensorforge::slmLoad<float, 16>(glb_m3 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v496_data = tensorforge::slmLoad<float, 16>(glb_m3 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v498_data = tensorforge::slmLoad<float, 16>(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v500_data = tensorforge::slmLoad<float, 16>(glb_m3 + (170_i32));
              tensorforge::intel_esimd::simd<float, 16> v501_acc{};
              tensorforge::intel_esimd::simd<float, 16> v504_data(0.0f);
              v504_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s1 + (-1_i32)) + 1);
              v501_acc += ((static_cast<float>(v504_data[1])) * v466_data);
              v501_acc += ((static_cast<float>(v504_data[2])) * v468_data);
              v501_acc += ((static_cast<float>(v504_data[3])) * v470_data);
              v501_acc += ((static_cast<float>(v504_data[4])) * v472_data);
              v501_acc += ((static_cast<float>(v504_data[5])) * v474_data);
              v501_acc += ((static_cast<float>(v504_data[6])) * v476_data);
              v501_acc += ((static_cast<float>(v504_data[7])) * v478_data);
              v501_acc += ((static_cast<float>(v504_data[8])) * v480_data);
              v501_acc += ((static_cast<float>(v504_data[9])) * v482_data);
              v501_acc += ((static_cast<float>(v504_data[10])) * v484_data);
              v501_acc += ((static_cast<float>(v504_data[11])) * v486_data);
              v501_acc += ((static_cast<float>(v504_data[12])) * v488_data);
              v501_acc += ((static_cast<float>(v504_data[13])) * v490_data);
              v501_acc += ((static_cast<float>(v504_data[14])) * v492_data);
              v501_acc += ((static_cast<float>(v504_data[15])) * v494_data);
              tensorforge::intel_esimd::simd<float, 16> v540_data = tensorforge::slmLoad<float, 16>(s1 + (15_i32));
              v501_acc += ((static_cast<float>(v540_data[0])) * v496_data);
              v501_acc += ((static_cast<float>(v540_data[1])) * v498_data);
              v501_acc += ((static_cast<float>(v540_data[2])) * v500_data);
              ir1.template select<16, 1>(0) = v501_acc;
              tensorforge::intel_esimd::simd<float, 16> v547_acc{};
              tensorforge::intel_esimd::simd<float, 16> v549_data = tensorforge::slmLoad<float, 16>(s1 + (17_i32));
              v547_acc += ((static_cast<float>(v549_data[1])) * v466_data);
              v547_acc += ((static_cast<float>(v549_data[2])) * v468_data);
              v547_acc += ((static_cast<float>(v549_data[3])) * v470_data);
              v547_acc += ((static_cast<float>(v549_data[4])) * v472_data);
              v547_acc += ((static_cast<float>(v549_data[5])) * v474_data);
              v547_acc += ((static_cast<float>(v549_data[6])) * v476_data);
              v547_acc += ((static_cast<float>(v549_data[7])) * v478_data);
              v547_acc += ((static_cast<float>(v549_data[8])) * v480_data);
              v547_acc += ((static_cast<float>(v549_data[9])) * v482_data);
              v547_acc += ((static_cast<float>(v549_data[10])) * v484_data);
              v547_acc += ((static_cast<float>(v549_data[11])) * v486_data);
              v547_acc += ((static_cast<float>(v549_data[12])) * v488_data);
              v547_acc += ((static_cast<float>(v549_data[13])) * v490_data);
              v547_acc += ((static_cast<float>(v549_data[14])) * v492_data);
              v547_acc += ((static_cast<float>(v549_data[15])) * v494_data);
              tensorforge::intel_esimd::simd<float, 16> v582_data = tensorforge::slmLoad<float, 16>(s1 + (33_i32));
              v547_acc += ((static_cast<float>(v582_data[0])) * v496_data);
              v547_acc += ((static_cast<float>(v582_data[1])) * v498_data);
              v547_acc += ((static_cast<float>(v582_data[2])) * v500_data);
              ir1.template select<16, 1>(16) = v547_acc;
              tensorforge::intel_esimd::simd<float, 16> v589_acc{};
              tensorforge::intel_esimd::simd<float, 16> v591_data = tensorforge::slmLoad<float, 16>(s1 + (35_i32));
              v589_acc += ((static_cast<float>(v591_data[1])) * v466_data);
              v589_acc += ((static_cast<float>(v591_data[2])) * v468_data);
              v589_acc += ((static_cast<float>(v591_data[3])) * v470_data);
              v589_acc += ((static_cast<float>(v591_data[4])) * v472_data);
              v589_acc += ((static_cast<float>(v591_data[5])) * v474_data);
              v589_acc += ((static_cast<float>(v591_data[6])) * v476_data);
              v589_acc += ((static_cast<float>(v591_data[7])) * v478_data);
              v589_acc += ((static_cast<float>(v591_data[8])) * v480_data);
              v589_acc += ((static_cast<float>(v591_data[9])) * v482_data);
              v589_acc += ((static_cast<float>(v591_data[10])) * v484_data);
              v589_acc += ((static_cast<float>(v591_data[11])) * v486_data);
              v589_acc += ((static_cast<float>(v591_data[12])) * v488_data);
              v589_acc += ((static_cast<float>(v591_data[13])) * v490_data);
              v589_acc += ((static_cast<float>(v591_data[14])) * v492_data);
              v589_acc += ((static_cast<float>(v591_data[15])) * v494_data);
              tensorforge::intel_esimd::simd<float, 16> v624_data = tensorforge::slmLoad<float, 16>(s1 + (51_i32));
              v589_acc += ((static_cast<float>(v624_data[0])) * v496_data);
              v589_acc += ((static_cast<float>(v624_data[1])) * v498_data);
              v589_acc += ((static_cast<float>(v624_data[2])) * v500_data);
              ir1.template select<16, 1>(32) = v589_acc;
              tensorforge::intel_esimd::simd<float, 16> v631_acc{};
              tensorforge::intel_esimd::simd<float, 16> v633_data = tensorforge::slmLoad<float, 16>(s1 + (53_i32));
              v631_acc += ((static_cast<float>(v633_data[1])) * v466_data);
              v631_acc += ((static_cast<float>(v633_data[2])) * v468_data);
              v631_acc += ((static_cast<float>(v633_data[3])) * v470_data);
              v631_acc += ((static_cast<float>(v633_data[4])) * v472_data);
              v631_acc += ((static_cast<float>(v633_data[5])) * v474_data);
              v631_acc += ((static_cast<float>(v633_data[6])) * v476_data);
              v631_acc += ((static_cast<float>(v633_data[7])) * v478_data);
              v631_acc += ((static_cast<float>(v633_data[8])) * v480_data);
              v631_acc += ((static_cast<float>(v633_data[9])) * v482_data);
              v631_acc += ((static_cast<float>(v633_data[10])) * v484_data);
              v631_acc += ((static_cast<float>(v633_data[11])) * v486_data);
              v631_acc += ((static_cast<float>(v633_data[12])) * v488_data);
              v631_acc += ((static_cast<float>(v633_data[13])) * v490_data);
              v631_acc += ((static_cast<float>(v633_data[14])) * v492_data);
              v631_acc += ((static_cast<float>(v633_data[15])) * v494_data);
              tensorforge::intel_esimd::simd<float, 16> v666_data = tensorforge::slmLoad<float, 16>(s1 + (69_i32));
              v631_acc += ((static_cast<float>(v666_data[0])) * v496_data);
              v631_acc += ((static_cast<float>(v666_data[1])) * v498_data);
              v631_acc += ((static_cast<float>(v666_data[2])) * v500_data);
              ir1.template select<16, 1>(48) = v631_acc;
              tensorforge::intel_esimd::simd<float, 16> v673_acc{};
              tensorforge::intel_esimd::simd<float, 16> v675_data = tensorforge::slmLoad<float, 16>(s1 + (71_i32));
              v673_acc += ((static_cast<float>(v675_data[1])) * v466_data);
              v673_acc += ((static_cast<float>(v675_data[2])) * v468_data);
              v673_acc += ((static_cast<float>(v675_data[3])) * v470_data);
              v673_acc += ((static_cast<float>(v675_data[4])) * v472_data);
              v673_acc += ((static_cast<float>(v675_data[5])) * v474_data);
              v673_acc += ((static_cast<float>(v675_data[6])) * v476_data);
              v673_acc += ((static_cast<float>(v675_data[7])) * v478_data);
              v673_acc += ((static_cast<float>(v675_data[8])) * v480_data);
              v673_acc += ((static_cast<float>(v675_data[9])) * v482_data);
              v673_acc += ((static_cast<float>(v675_data[10])) * v484_data);
              v673_acc += ((static_cast<float>(v675_data[11])) * v486_data);
              v673_acc += ((static_cast<float>(v675_data[12])) * v488_data);
              v673_acc += ((static_cast<float>(v675_data[13])) * v490_data);
              v673_acc += ((static_cast<float>(v675_data[14])) * v492_data);
              v673_acc += ((static_cast<float>(v675_data[15])) * v494_data);
              tensorforge::intel_esimd::simd<float, 16> v708_data = tensorforge::slmLoad<float, 16>(s1 + (87_i32));
              v673_acc += ((static_cast<float>(v708_data[0])) * v496_data);
              v673_acc += ((static_cast<float>(v708_data[1])) * v498_data);
              v673_acc += ((static_cast<float>(v708_data[2])) * v500_data);
              ir1.template select<16, 1>(64) = v673_acc;
              tensorforge::intel_esimd::simd<float, 16> v715_acc{};
              tensorforge::intel_esimd::simd<float, 16> v717_data = tensorforge::slmLoad<float, 16>(s1 + (89_i32));
              v715_acc += ((static_cast<float>(v717_data[1])) * v466_data);
              v715_acc += ((static_cast<float>(v717_data[2])) * v468_data);
              v715_acc += ((static_cast<float>(v717_data[3])) * v470_data);
              v715_acc += ((static_cast<float>(v717_data[4])) * v472_data);
              v715_acc += ((static_cast<float>(v717_data[5])) * v474_data);
              v715_acc += ((static_cast<float>(v717_data[6])) * v476_data);
              v715_acc += ((static_cast<float>(v717_data[7])) * v478_data);
              v715_acc += ((static_cast<float>(v717_data[8])) * v480_data);
              v715_acc += ((static_cast<float>(v717_data[9])) * v482_data);
              v715_acc += ((static_cast<float>(v717_data[10])) * v484_data);
              v715_acc += ((static_cast<float>(v717_data[11])) * v486_data);
              v715_acc += ((static_cast<float>(v717_data[12])) * v488_data);
              v715_acc += ((static_cast<float>(v717_data[13])) * v490_data);
              v715_acc += ((static_cast<float>(v717_data[14])) * v492_data);
              v715_acc += ((static_cast<float>(v717_data[15])) * v494_data);
              tensorforge::intel_esimd::simd<float, 16> v750_data = tensorforge::slmLoad<float, 16>(s1 + (105_i32));
              v715_acc += ((static_cast<float>(v750_data[0])) * v496_data);
              v715_acc += ((static_cast<float>(v750_data[1])) * v498_data);
              v715_acc += ((static_cast<float>(v750_data[2])) * v500_data);
              ir1.template select<16, 1>(80) = v715_acc;
              tensorforge::intel_esimd::simd<float, 16> v757_acc{};
              tensorforge::intel_esimd::simd<float, 16> v759_data = tensorforge::slmLoad<float, 16>(s1 + (107_i32));
              v757_acc += ((static_cast<float>(v759_data[1])) * v466_data);
              v757_acc += ((static_cast<float>(v759_data[2])) * v468_data);
              v757_acc += ((static_cast<float>(v759_data[3])) * v470_data);
              v757_acc += ((static_cast<float>(v759_data[4])) * v472_data);
              v757_acc += ((static_cast<float>(v759_data[5])) * v474_data);
              v757_acc += ((static_cast<float>(v759_data[6])) * v476_data);
              v757_acc += ((static_cast<float>(v759_data[7])) * v478_data);
              v757_acc += ((static_cast<float>(v759_data[8])) * v480_data);
              v757_acc += ((static_cast<float>(v759_data[9])) * v482_data);
              v757_acc += ((static_cast<float>(v759_data[10])) * v484_data);
              v757_acc += ((static_cast<float>(v759_data[11])) * v486_data);
              v757_acc += ((static_cast<float>(v759_data[12])) * v488_data);
              v757_acc += ((static_cast<float>(v759_data[13])) * v490_data);
              v757_acc += ((static_cast<float>(v759_data[14])) * v492_data);
              v757_acc += ((static_cast<float>(v759_data[15])) * v494_data);
              tensorforge::intel_esimd::simd<float, 16> v792_data = tensorforge::slmLoad<float, 16>(s1 + (123_i32));
              v757_acc += ((static_cast<float>(v792_data[0])) * v496_data);
              v757_acc += ((static_cast<float>(v792_data[1])) * v498_data);
              v757_acc += ((static_cast<float>(v792_data[2])) * v500_data);
              ir1.template select<16, 1>(96) = v757_acc;
              tensorforge::intel_esimd::simd<float, 16> v799_acc{};
              tensorforge::intel_esimd::simd<float, 16> v801_data = tensorforge::slmLoad<float, 16>(s1 + (125_i32));
              v799_acc += ((static_cast<float>(v801_data[1])) * v466_data);
              v799_acc += ((static_cast<float>(v801_data[2])) * v468_data);
              v799_acc += ((static_cast<float>(v801_data[3])) * v470_data);
              v799_acc += ((static_cast<float>(v801_data[4])) * v472_data);
              v799_acc += ((static_cast<float>(v801_data[5])) * v474_data);
              v799_acc += ((static_cast<float>(v801_data[6])) * v476_data);
              v799_acc += ((static_cast<float>(v801_data[7])) * v478_data);
              v799_acc += ((static_cast<float>(v801_data[8])) * v480_data);
              v799_acc += ((static_cast<float>(v801_data[9])) * v482_data);
              v799_acc += ((static_cast<float>(v801_data[10])) * v484_data);
              v799_acc += ((static_cast<float>(v801_data[11])) * v486_data);
              v799_acc += ((static_cast<float>(v801_data[12])) * v488_data);
              v799_acc += ((static_cast<float>(v801_data[13])) * v490_data);
              v799_acc += ((static_cast<float>(v801_data[14])) * v492_data);
              v799_acc += ((static_cast<float>(v801_data[15])) * v494_data);
              tensorforge::intel_esimd::simd<float, 16> v834_data = tensorforge::slmLoad<float, 16>(s1 + (141_i32));
              v799_acc += ((static_cast<float>(v834_data[0])) * v496_data);
              v799_acc += ((static_cast<float>(v834_data[1])) * v498_data);
              v799_acc += ((static_cast<float>(v834_data[2])) * v500_data);
              ir1.template select<16, 1>(112) = v799_acc;
              tensorforge::intel_esimd::simd<float, 16> v841_acc{};
              tensorforge::intel_esimd::simd<float, 16> v843_data = tensorforge::slmLoad<float, 16>(s1 + (143_i32));
              v841_acc += ((static_cast<float>(v843_data[1])) * v466_data);
              v841_acc += ((static_cast<float>(v843_data[2])) * v468_data);
              v841_acc += ((static_cast<float>(v843_data[3])) * v470_data);
              v841_acc += ((static_cast<float>(v843_data[4])) * v472_data);
              v841_acc += ((static_cast<float>(v843_data[5])) * v474_data);
              v841_acc += ((static_cast<float>(v843_data[6])) * v476_data);
              v841_acc += ((static_cast<float>(v843_data[7])) * v478_data);
              v841_acc += ((static_cast<float>(v843_data[8])) * v480_data);
              v841_acc += ((static_cast<float>(v843_data[9])) * v482_data);
              v841_acc += ((static_cast<float>(v843_data[10])) * v484_data);
              v841_acc += ((static_cast<float>(v843_data[11])) * v486_data);
              v841_acc += ((static_cast<float>(v843_data[12])) * v488_data);
              v841_acc += ((static_cast<float>(v843_data[13])) * v490_data);
              v841_acc += ((static_cast<float>(v843_data[14])) * v492_data);
              v841_acc += ((static_cast<float>(v843_data[15])) * v494_data);
              tensorforge::intel_esimd::simd<float, 16> v876_data = tensorforge::slmLoad<float, 16>(s1 + (159_i32));
              v841_acc += ((static_cast<float>(v876_data[0])) * v496_data);
              v841_acc += ((static_cast<float>(v876_data[1])) * v498_data);
              v841_acc += ((static_cast<float>(v876_data[2])) * v500_data);
              ir1.template select<16, 1>(128) = v841_acc;
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v883_n1 = 0; v883_n1 < 9; ++v883_n1) {
                int32_t v884_a = v883_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v886_data(ir1.template select<10, 1>(v884_a));
                tensorforge::intel_esimd::simd<float, 10> v887_data(r0.template select<10, 1>(v884_a));
                r1.template select<10, 1>(v884_a) = (v887_data + v886_data);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v889_i1 = 0; v889_i1 < 9; ++v889_i1) {
                tensorforge::intel_esimd::simd<float, 10> v892_data(r1.template select<10, 1>((v889_i1 * 16)));
                v892_data.copy_to(glb_m0 + ((v889_i1 * 10)));
              }
            }
          }
        }
      }
    });
  });
}


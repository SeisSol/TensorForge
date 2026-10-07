// === base name ===
kernel_04194a9bbdd9955d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_04194a9bbdd9955d = {{1, 32, 1}, 16, 10, 1, 32, 44416, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_04194a9bbdd9955d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_04194a9bbdd9955d(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_04194a9bbdd9955d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_04194a9bbdd9955d(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_04194a9bbdd9955d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_04194a9bbdd9955d(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_04194a9bbdd9955d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
            tensorforge::intel_esimd::simd<float, 10> v32_ld;
            v32_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 10>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v32_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (160);
          for (size_t v35_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v35_batchId0 < numElements0; v35_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v36_ahead1 = v35_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v38_batchId1 = (v36_ahead1 < numElements0) ? v36_ahead1 : v35_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v35_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v35_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v35_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v35_batchId0 * 153 + 0 + m4_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v46_ld;
              v46_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v46_ld);
              tensorforge::intel_esimd::simd<float, 64> v47_ld;
              v47_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v47_ld);
              tensorforge::intel_esimd::simd<float, 16> v48_ld;
              v48_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v48_ld);
              tensorforge::intel_esimd::simd<float, 9> v49_ld;
              v49_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v49_ld);
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v50_ld;
              v50_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v50_ld);
              tensorforge::intel_esimd::simd<float, 64> v51_ld;
              v51_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 64), v51_ld);
              tensorforge::intel_esimd::simd<float, 16> v52_ld;
              v52_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s1 + (0 + 0 + 1 * 0 + 128), v52_ld);
              tensorforge::intel_esimd::simd<float, 9> v53_ld;
              v53_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s1 + (0 + 0 + 1 * 0 + 144), v53_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v59_data = tensorforge::slmLoad<float, 16>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v61_data = tensorforge::slmLoad<float, 16>(glb_m1 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data = tensorforge::slmLoad<float, 16>(glb_m1 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data = tensorforge::slmLoad<float, 16>(glb_m1 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data = tensorforge::slmLoad<float, 16>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v69_data = tensorforge::slmLoad<float, 16>(glb_m1 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data = tensorforge::slmLoad<float, 16>(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v73_data = tensorforge::slmLoad<float, 16>(glb_m1 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data = tensorforge::slmLoad<float, 16>(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v77_data = tensorforge::slmLoad<float, 16>(glb_m1 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v79_data = tensorforge::slmLoad<float, 16>(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v81_data = tensorforge::slmLoad<float, 16>(glb_m1 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v83_data = tensorforge::slmLoad<float, 16>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v85_data = tensorforge::slmLoad<float, 16>(glb_m1 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v87_data = tensorforge::slmLoad<float, 16>(glb_m1 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v89_data = tensorforge::slmLoad<float, 16>(glb_m1 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v91_data = tensorforge::slmLoad<float, 16>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v92_acc{};
              tensorforge::intel_esimd::simd<float, 16> v93_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v92_acc += ((static_cast<float>(v93_data[0])) * v59_data);
              v92_acc += ((static_cast<float>(v93_data[1])) * v61_data);
              v92_acc += ((static_cast<float>(v93_data[2])) * v63_data);
              v92_acc += ((static_cast<float>(v93_data[3])) * v65_data);
              v92_acc += ((static_cast<float>(v93_data[4])) * v67_data);
              v92_acc += ((static_cast<float>(v93_data[5])) * v69_data);
              v92_acc += ((static_cast<float>(v93_data[6])) * v71_data);
              v92_acc += ((static_cast<float>(v93_data[7])) * v73_data);
              v92_acc += ((static_cast<float>(v93_data[8])) * v75_data);
              v92_acc += ((static_cast<float>(v93_data[9])) * v77_data);
              v92_acc += ((static_cast<float>(v93_data[10])) * v79_data);
              v92_acc += ((static_cast<float>(v93_data[11])) * v81_data);
              v92_acc += ((static_cast<float>(v93_data[12])) * v83_data);
              v92_acc += ((static_cast<float>(v93_data[13])) * v85_data);
              v92_acc += ((static_cast<float>(v93_data[14])) * v87_data);
              v92_acc += ((static_cast<float>(v93_data[15])) * v89_data);
              tensorforge::intel_esimd::simd<float, 16> v129_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v92_acc += ((static_cast<float>(v129_data[0])) * v91_data);
              ir0.template select<16, 1>(0) = v92_acc;
              tensorforge::intel_esimd::simd<float, 16> v132_acc{};
              tensorforge::intel_esimd::simd<float, 16> v134_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              v132_acc += ((static_cast<float>(v134_data[0])) * v59_data);
              v132_acc += ((static_cast<float>(v134_data[1])) * v61_data);
              v132_acc += ((static_cast<float>(v134_data[2])) * v63_data);
              v132_acc += ((static_cast<float>(v134_data[3])) * v65_data);
              v132_acc += ((static_cast<float>(v134_data[4])) * v67_data);
              v132_acc += ((static_cast<float>(v134_data[5])) * v69_data);
              v132_acc += ((static_cast<float>(v134_data[6])) * v71_data);
              v132_acc += ((static_cast<float>(v134_data[7])) * v73_data);
              v132_acc += ((static_cast<float>(v134_data[8])) * v75_data);
              v132_acc += ((static_cast<float>(v134_data[9])) * v77_data);
              v132_acc += ((static_cast<float>(v134_data[10])) * v79_data);
              v132_acc += ((static_cast<float>(v134_data[11])) * v81_data);
              v132_acc += ((static_cast<float>(v134_data[12])) * v83_data);
              v132_acc += ((static_cast<float>(v134_data[13])) * v85_data);
              v132_acc += ((static_cast<float>(v134_data[14])) * v87_data);
              v132_acc += ((static_cast<float>(v134_data[15])) * v89_data);
              tensorforge::intel_esimd::simd<float, 16> v168_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v132_acc += ((static_cast<float>(v168_data[0])) * v91_data);
              ir0.template select<16, 1>(16) = v132_acc;
              tensorforge::intel_esimd::simd<float, 16> v171_acc{};
              tensorforge::intel_esimd::simd<float, 16> v173_data = tensorforge::slmLoad<float, 16>(s0 + (34_i32));
              v171_acc += ((static_cast<float>(v173_data[0])) * v59_data);
              v171_acc += ((static_cast<float>(v173_data[1])) * v61_data);
              v171_acc += ((static_cast<float>(v173_data[2])) * v63_data);
              v171_acc += ((static_cast<float>(v173_data[3])) * v65_data);
              v171_acc += ((static_cast<float>(v173_data[4])) * v67_data);
              v171_acc += ((static_cast<float>(v173_data[5])) * v69_data);
              v171_acc += ((static_cast<float>(v173_data[6])) * v71_data);
              v171_acc += ((static_cast<float>(v173_data[7])) * v73_data);
              v171_acc += ((static_cast<float>(v173_data[8])) * v75_data);
              v171_acc += ((static_cast<float>(v173_data[9])) * v77_data);
              v171_acc += ((static_cast<float>(v173_data[10])) * v79_data);
              v171_acc += ((static_cast<float>(v173_data[11])) * v81_data);
              v171_acc += ((static_cast<float>(v173_data[12])) * v83_data);
              v171_acc += ((static_cast<float>(v173_data[13])) * v85_data);
              v171_acc += ((static_cast<float>(v173_data[14])) * v87_data);
              v171_acc += ((static_cast<float>(v173_data[15])) * v89_data);
              tensorforge::intel_esimd::simd<float, 16> v207_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v171_acc += ((static_cast<float>(v207_data[0])) * v91_data);
              ir0.template select<16, 1>(32) = v171_acc;
              tensorforge::intel_esimd::simd<float, 16> v210_acc{};
              tensorforge::intel_esimd::simd<float, 16> v212_data = tensorforge::slmLoad<float, 16>(s0 + (51_i32));
              v210_acc += ((static_cast<float>(v212_data[0])) * v59_data);
              v210_acc += ((static_cast<float>(v212_data[1])) * v61_data);
              v210_acc += ((static_cast<float>(v212_data[2])) * v63_data);
              v210_acc += ((static_cast<float>(v212_data[3])) * v65_data);
              v210_acc += ((static_cast<float>(v212_data[4])) * v67_data);
              v210_acc += ((static_cast<float>(v212_data[5])) * v69_data);
              v210_acc += ((static_cast<float>(v212_data[6])) * v71_data);
              v210_acc += ((static_cast<float>(v212_data[7])) * v73_data);
              v210_acc += ((static_cast<float>(v212_data[8])) * v75_data);
              v210_acc += ((static_cast<float>(v212_data[9])) * v77_data);
              v210_acc += ((static_cast<float>(v212_data[10])) * v79_data);
              v210_acc += ((static_cast<float>(v212_data[11])) * v81_data);
              v210_acc += ((static_cast<float>(v212_data[12])) * v83_data);
              v210_acc += ((static_cast<float>(v212_data[13])) * v85_data);
              v210_acc += ((static_cast<float>(v212_data[14])) * v87_data);
              v210_acc += ((static_cast<float>(v212_data[15])) * v89_data);
              tensorforge::intel_esimd::simd<float, 16> v246_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v210_acc += ((static_cast<float>(v246_data[0])) * v91_data);
              ir0.template select<16, 1>(48) = v210_acc;
              tensorforge::intel_esimd::simd<float, 16> v249_acc{};
              tensorforge::intel_esimd::simd<float, 16> v251_data = tensorforge::slmLoad<float, 16>(s0 + (68_i32));
              v249_acc += ((static_cast<float>(v251_data[0])) * v59_data);
              v249_acc += ((static_cast<float>(v251_data[1])) * v61_data);
              v249_acc += ((static_cast<float>(v251_data[2])) * v63_data);
              v249_acc += ((static_cast<float>(v251_data[3])) * v65_data);
              v249_acc += ((static_cast<float>(v251_data[4])) * v67_data);
              v249_acc += ((static_cast<float>(v251_data[5])) * v69_data);
              v249_acc += ((static_cast<float>(v251_data[6])) * v71_data);
              v249_acc += ((static_cast<float>(v251_data[7])) * v73_data);
              v249_acc += ((static_cast<float>(v251_data[8])) * v75_data);
              v249_acc += ((static_cast<float>(v251_data[9])) * v77_data);
              v249_acc += ((static_cast<float>(v251_data[10])) * v79_data);
              v249_acc += ((static_cast<float>(v251_data[11])) * v81_data);
              v249_acc += ((static_cast<float>(v251_data[12])) * v83_data);
              v249_acc += ((static_cast<float>(v251_data[13])) * v85_data);
              v249_acc += ((static_cast<float>(v251_data[14])) * v87_data);
              v249_acc += ((static_cast<float>(v251_data[15])) * v89_data);
              tensorforge::intel_esimd::simd<float, 16> v285_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v249_acc += ((static_cast<float>(v285_data[0])) * v91_data);
              ir0.template select<16, 1>(64) = v249_acc;
              tensorforge::intel_esimd::simd<float, 16> v288_acc{};
              tensorforge::intel_esimd::simd<float, 16> v290_data = tensorforge::slmLoad<float, 16>(s0 + (85_i32));
              v288_acc += ((static_cast<float>(v290_data[0])) * v59_data);
              v288_acc += ((static_cast<float>(v290_data[1])) * v61_data);
              v288_acc += ((static_cast<float>(v290_data[2])) * v63_data);
              v288_acc += ((static_cast<float>(v290_data[3])) * v65_data);
              v288_acc += ((static_cast<float>(v290_data[4])) * v67_data);
              v288_acc += ((static_cast<float>(v290_data[5])) * v69_data);
              v288_acc += ((static_cast<float>(v290_data[6])) * v71_data);
              v288_acc += ((static_cast<float>(v290_data[7])) * v73_data);
              v288_acc += ((static_cast<float>(v290_data[8])) * v75_data);
              v288_acc += ((static_cast<float>(v290_data[9])) * v77_data);
              v288_acc += ((static_cast<float>(v290_data[10])) * v79_data);
              v288_acc += ((static_cast<float>(v290_data[11])) * v81_data);
              v288_acc += ((static_cast<float>(v290_data[12])) * v83_data);
              v288_acc += ((static_cast<float>(v290_data[13])) * v85_data);
              v288_acc += ((static_cast<float>(v290_data[14])) * v87_data);
              v288_acc += ((static_cast<float>(v290_data[15])) * v89_data);
              tensorforge::intel_esimd::simd<float, 16> v324_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v288_acc += ((static_cast<float>(v324_data[0])) * v91_data);
              ir0.template select<16, 1>(80) = v288_acc;
              tensorforge::intel_esimd::simd<float, 16> v327_acc{};
              tensorforge::intel_esimd::simd<float, 16> v329_data = tensorforge::slmLoad<float, 16>(s0 + (102_i32));
              v327_acc += ((static_cast<float>(v329_data[0])) * v59_data);
              v327_acc += ((static_cast<float>(v329_data[1])) * v61_data);
              v327_acc += ((static_cast<float>(v329_data[2])) * v63_data);
              v327_acc += ((static_cast<float>(v329_data[3])) * v65_data);
              v327_acc += ((static_cast<float>(v329_data[4])) * v67_data);
              v327_acc += ((static_cast<float>(v329_data[5])) * v69_data);
              v327_acc += ((static_cast<float>(v329_data[6])) * v71_data);
              v327_acc += ((static_cast<float>(v329_data[7])) * v73_data);
              v327_acc += ((static_cast<float>(v329_data[8])) * v75_data);
              v327_acc += ((static_cast<float>(v329_data[9])) * v77_data);
              v327_acc += ((static_cast<float>(v329_data[10])) * v79_data);
              v327_acc += ((static_cast<float>(v329_data[11])) * v81_data);
              v327_acc += ((static_cast<float>(v329_data[12])) * v83_data);
              v327_acc += ((static_cast<float>(v329_data[13])) * v85_data);
              v327_acc += ((static_cast<float>(v329_data[14])) * v87_data);
              v327_acc += ((static_cast<float>(v329_data[15])) * v89_data);
              tensorforge::intel_esimd::simd<float, 16> v363_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v327_acc += ((static_cast<float>(v363_data[0])) * v91_data);
              ir0.template select<16, 1>(96) = v327_acc;
              tensorforge::intel_esimd::simd<float, 16> v366_acc{};
              tensorforge::intel_esimd::simd<float, 16> v368_data = tensorforge::slmLoad<float, 16>(s0 + (119_i32));
              v366_acc += ((static_cast<float>(v368_data[0])) * v59_data);
              v366_acc += ((static_cast<float>(v368_data[1])) * v61_data);
              v366_acc += ((static_cast<float>(v368_data[2])) * v63_data);
              v366_acc += ((static_cast<float>(v368_data[3])) * v65_data);
              v366_acc += ((static_cast<float>(v368_data[4])) * v67_data);
              v366_acc += ((static_cast<float>(v368_data[5])) * v69_data);
              v366_acc += ((static_cast<float>(v368_data[6])) * v71_data);
              v366_acc += ((static_cast<float>(v368_data[7])) * v73_data);
              v366_acc += ((static_cast<float>(v368_data[8])) * v75_data);
              v366_acc += ((static_cast<float>(v368_data[9])) * v77_data);
              v366_acc += ((static_cast<float>(v368_data[10])) * v79_data);
              v366_acc += ((static_cast<float>(v368_data[11])) * v81_data);
              v366_acc += ((static_cast<float>(v368_data[12])) * v83_data);
              v366_acc += ((static_cast<float>(v368_data[13])) * v85_data);
              v366_acc += ((static_cast<float>(v368_data[14])) * v87_data);
              v366_acc += ((static_cast<float>(v368_data[15])) * v89_data);
              tensorforge::intel_esimd::simd<float, 16> v402_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v366_acc += ((static_cast<float>(v402_data[0])) * v91_data);
              ir0.template select<16, 1>(112) = v366_acc;
              tensorforge::intel_esimd::simd<float, 16> v405_acc{};
              tensorforge::intel_esimd::simd<float, 16> v407_data = tensorforge::slmLoad<float, 16>(s0 + (136_i32));
              v405_acc += ((static_cast<float>(v407_data[0])) * v59_data);
              v405_acc += ((static_cast<float>(v407_data[1])) * v61_data);
              v405_acc += ((static_cast<float>(v407_data[2])) * v63_data);
              v405_acc += ((static_cast<float>(v407_data[3])) * v65_data);
              v405_acc += ((static_cast<float>(v407_data[4])) * v67_data);
              v405_acc += ((static_cast<float>(v407_data[5])) * v69_data);
              v405_acc += ((static_cast<float>(v407_data[6])) * v71_data);
              v405_acc += ((static_cast<float>(v407_data[7])) * v73_data);
              v405_acc += ((static_cast<float>(v407_data[8])) * v75_data);
              v405_acc += ((static_cast<float>(v407_data[9])) * v77_data);
              v405_acc += ((static_cast<float>(v407_data[10])) * v79_data);
              v405_acc += ((static_cast<float>(v407_data[11])) * v81_data);
              v405_acc += ((static_cast<float>(v407_data[12])) * v83_data);
              v405_acc += ((static_cast<float>(v407_data[13])) * v85_data);
              v405_acc += ((static_cast<float>(v407_data[14])) * v87_data);
              v405_acc += ((static_cast<float>(v407_data[15])) * v89_data);
              tensorforge::intel_esimd::simd<float, 16> v441_data = tensorforge::slmLoad<float, 16>(s0 + (152_i32));
              v405_acc += ((static_cast<float>(v441_data[0])) * v91_data);
              ir0.template select<16, 1>(128) = v405_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v444_n1 = 0; v444_n1 < 9; ++v444_n1) {
                int32_t v445_a = v444_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v447_data(ir0.template select<10, 1>(v445_a));
                r0.template select<10, 1>(v445_a) = v447_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 10), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v453_data = tensorforge::slmLoad<float, 16>(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v455_data = tensorforge::slmLoad<float, 16>(glb_m3 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v457_data = tensorforge::slmLoad<float, 16>(glb_m3 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v459_data = tensorforge::slmLoad<float, 16>(glb_m3 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v461_data = tensorforge::slmLoad<float, 16>(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v463_data = tensorforge::slmLoad<float, 16>(glb_m3 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v465_data = tensorforge::slmLoad<float, 16>(glb_m3 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v467_data = tensorforge::slmLoad<float, 16>(glb_m3 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v469_data = tensorforge::slmLoad<float, 16>(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v471_data = tensorforge::slmLoad<float, 16>(glb_m3 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v473_data = tensorforge::slmLoad<float, 16>(glb_m3 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v475_data = tensorforge::slmLoad<float, 16>(glb_m3 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v477_data = tensorforge::slmLoad<float, 16>(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v479_data = tensorforge::slmLoad<float, 16>(glb_m3 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v481_data = tensorforge::slmLoad<float, 16>(glb_m3 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v483_data = tensorforge::slmLoad<float, 16>(glb_m3 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v485_data = tensorforge::slmLoad<float, 16>(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v486_acc{};
              tensorforge::intel_esimd::simd<float, 16> v487_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v486_acc += ((static_cast<float>(v487_data[0])) * v453_data);
              v486_acc += ((static_cast<float>(v487_data[1])) * v455_data);
              v486_acc += ((static_cast<float>(v487_data[2])) * v457_data);
              v486_acc += ((static_cast<float>(v487_data[3])) * v459_data);
              v486_acc += ((static_cast<float>(v487_data[4])) * v461_data);
              v486_acc += ((static_cast<float>(v487_data[5])) * v463_data);
              v486_acc += ((static_cast<float>(v487_data[6])) * v465_data);
              v486_acc += ((static_cast<float>(v487_data[7])) * v467_data);
              v486_acc += ((static_cast<float>(v487_data[8])) * v469_data);
              v486_acc += ((static_cast<float>(v487_data[9])) * v471_data);
              v486_acc += ((static_cast<float>(v487_data[10])) * v473_data);
              v486_acc += ((static_cast<float>(v487_data[11])) * v475_data);
              v486_acc += ((static_cast<float>(v487_data[12])) * v477_data);
              v486_acc += ((static_cast<float>(v487_data[13])) * v479_data);
              v486_acc += ((static_cast<float>(v487_data[14])) * v481_data);
              v486_acc += ((static_cast<float>(v487_data[15])) * v483_data);
              tensorforge::intel_esimd::simd<float, 16> v523_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v486_acc += ((static_cast<float>(v523_data[0])) * v485_data);
              ir1.template select<16, 1>(0) = v486_acc;
              tensorforge::intel_esimd::simd<float, 16> v526_acc{};
              tensorforge::intel_esimd::simd<float, 16> v528_data = tensorforge::slmLoad<float, 16>(s1 + (17_i32));
              v526_acc += ((static_cast<float>(v528_data[0])) * v453_data);
              v526_acc += ((static_cast<float>(v528_data[1])) * v455_data);
              v526_acc += ((static_cast<float>(v528_data[2])) * v457_data);
              v526_acc += ((static_cast<float>(v528_data[3])) * v459_data);
              v526_acc += ((static_cast<float>(v528_data[4])) * v461_data);
              v526_acc += ((static_cast<float>(v528_data[5])) * v463_data);
              v526_acc += ((static_cast<float>(v528_data[6])) * v465_data);
              v526_acc += ((static_cast<float>(v528_data[7])) * v467_data);
              v526_acc += ((static_cast<float>(v528_data[8])) * v469_data);
              v526_acc += ((static_cast<float>(v528_data[9])) * v471_data);
              v526_acc += ((static_cast<float>(v528_data[10])) * v473_data);
              v526_acc += ((static_cast<float>(v528_data[11])) * v475_data);
              v526_acc += ((static_cast<float>(v528_data[12])) * v477_data);
              v526_acc += ((static_cast<float>(v528_data[13])) * v479_data);
              v526_acc += ((static_cast<float>(v528_data[14])) * v481_data);
              v526_acc += ((static_cast<float>(v528_data[15])) * v483_data);
              tensorforge::intel_esimd::simd<float, 16> v562_data = tensorforge::slmLoad<float, 16>(s1 + (33_i32));
              v526_acc += ((static_cast<float>(v562_data[0])) * v485_data);
              ir1.template select<16, 1>(16) = v526_acc;
              tensorforge::intel_esimd::simd<float, 16> v565_acc{};
              tensorforge::intel_esimd::simd<float, 16> v567_data = tensorforge::slmLoad<float, 16>(s1 + (34_i32));
              v565_acc += ((static_cast<float>(v567_data[0])) * v453_data);
              v565_acc += ((static_cast<float>(v567_data[1])) * v455_data);
              v565_acc += ((static_cast<float>(v567_data[2])) * v457_data);
              v565_acc += ((static_cast<float>(v567_data[3])) * v459_data);
              v565_acc += ((static_cast<float>(v567_data[4])) * v461_data);
              v565_acc += ((static_cast<float>(v567_data[5])) * v463_data);
              v565_acc += ((static_cast<float>(v567_data[6])) * v465_data);
              v565_acc += ((static_cast<float>(v567_data[7])) * v467_data);
              v565_acc += ((static_cast<float>(v567_data[8])) * v469_data);
              v565_acc += ((static_cast<float>(v567_data[9])) * v471_data);
              v565_acc += ((static_cast<float>(v567_data[10])) * v473_data);
              v565_acc += ((static_cast<float>(v567_data[11])) * v475_data);
              v565_acc += ((static_cast<float>(v567_data[12])) * v477_data);
              v565_acc += ((static_cast<float>(v567_data[13])) * v479_data);
              v565_acc += ((static_cast<float>(v567_data[14])) * v481_data);
              v565_acc += ((static_cast<float>(v567_data[15])) * v483_data);
              tensorforge::intel_esimd::simd<float, 16> v601_data = tensorforge::slmLoad<float, 16>(s1 + (50_i32));
              v565_acc += ((static_cast<float>(v601_data[0])) * v485_data);
              ir1.template select<16, 1>(32) = v565_acc;
              tensorforge::intel_esimd::simd<float, 16> v604_acc{};
              tensorforge::intel_esimd::simd<float, 16> v606_data = tensorforge::slmLoad<float, 16>(s1 + (51_i32));
              v604_acc += ((static_cast<float>(v606_data[0])) * v453_data);
              v604_acc += ((static_cast<float>(v606_data[1])) * v455_data);
              v604_acc += ((static_cast<float>(v606_data[2])) * v457_data);
              v604_acc += ((static_cast<float>(v606_data[3])) * v459_data);
              v604_acc += ((static_cast<float>(v606_data[4])) * v461_data);
              v604_acc += ((static_cast<float>(v606_data[5])) * v463_data);
              v604_acc += ((static_cast<float>(v606_data[6])) * v465_data);
              v604_acc += ((static_cast<float>(v606_data[7])) * v467_data);
              v604_acc += ((static_cast<float>(v606_data[8])) * v469_data);
              v604_acc += ((static_cast<float>(v606_data[9])) * v471_data);
              v604_acc += ((static_cast<float>(v606_data[10])) * v473_data);
              v604_acc += ((static_cast<float>(v606_data[11])) * v475_data);
              v604_acc += ((static_cast<float>(v606_data[12])) * v477_data);
              v604_acc += ((static_cast<float>(v606_data[13])) * v479_data);
              v604_acc += ((static_cast<float>(v606_data[14])) * v481_data);
              v604_acc += ((static_cast<float>(v606_data[15])) * v483_data);
              tensorforge::intel_esimd::simd<float, 16> v640_data = tensorforge::slmLoad<float, 16>(s1 + (67_i32));
              v604_acc += ((static_cast<float>(v640_data[0])) * v485_data);
              ir1.template select<16, 1>(48) = v604_acc;
              tensorforge::intel_esimd::simd<float, 16> v643_acc{};
              tensorforge::intel_esimd::simd<float, 16> v645_data = tensorforge::slmLoad<float, 16>(s1 + (68_i32));
              v643_acc += ((static_cast<float>(v645_data[0])) * v453_data);
              v643_acc += ((static_cast<float>(v645_data[1])) * v455_data);
              v643_acc += ((static_cast<float>(v645_data[2])) * v457_data);
              v643_acc += ((static_cast<float>(v645_data[3])) * v459_data);
              v643_acc += ((static_cast<float>(v645_data[4])) * v461_data);
              v643_acc += ((static_cast<float>(v645_data[5])) * v463_data);
              v643_acc += ((static_cast<float>(v645_data[6])) * v465_data);
              v643_acc += ((static_cast<float>(v645_data[7])) * v467_data);
              v643_acc += ((static_cast<float>(v645_data[8])) * v469_data);
              v643_acc += ((static_cast<float>(v645_data[9])) * v471_data);
              v643_acc += ((static_cast<float>(v645_data[10])) * v473_data);
              v643_acc += ((static_cast<float>(v645_data[11])) * v475_data);
              v643_acc += ((static_cast<float>(v645_data[12])) * v477_data);
              v643_acc += ((static_cast<float>(v645_data[13])) * v479_data);
              v643_acc += ((static_cast<float>(v645_data[14])) * v481_data);
              v643_acc += ((static_cast<float>(v645_data[15])) * v483_data);
              tensorforge::intel_esimd::simd<float, 16> v679_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v643_acc += ((static_cast<float>(v679_data[0])) * v485_data);
              ir1.template select<16, 1>(64) = v643_acc;
              tensorforge::intel_esimd::simd<float, 16> v682_acc{};
              tensorforge::intel_esimd::simd<float, 16> v684_data = tensorforge::slmLoad<float, 16>(s1 + (85_i32));
              v682_acc += ((static_cast<float>(v684_data[0])) * v453_data);
              v682_acc += ((static_cast<float>(v684_data[1])) * v455_data);
              v682_acc += ((static_cast<float>(v684_data[2])) * v457_data);
              v682_acc += ((static_cast<float>(v684_data[3])) * v459_data);
              v682_acc += ((static_cast<float>(v684_data[4])) * v461_data);
              v682_acc += ((static_cast<float>(v684_data[5])) * v463_data);
              v682_acc += ((static_cast<float>(v684_data[6])) * v465_data);
              v682_acc += ((static_cast<float>(v684_data[7])) * v467_data);
              v682_acc += ((static_cast<float>(v684_data[8])) * v469_data);
              v682_acc += ((static_cast<float>(v684_data[9])) * v471_data);
              v682_acc += ((static_cast<float>(v684_data[10])) * v473_data);
              v682_acc += ((static_cast<float>(v684_data[11])) * v475_data);
              v682_acc += ((static_cast<float>(v684_data[12])) * v477_data);
              v682_acc += ((static_cast<float>(v684_data[13])) * v479_data);
              v682_acc += ((static_cast<float>(v684_data[14])) * v481_data);
              v682_acc += ((static_cast<float>(v684_data[15])) * v483_data);
              tensorforge::intel_esimd::simd<float, 16> v718_data = tensorforge::slmLoad<float, 16>(s1 + (101_i32));
              v682_acc += ((static_cast<float>(v718_data[0])) * v485_data);
              ir1.template select<16, 1>(80) = v682_acc;
              tensorforge::intel_esimd::simd<float, 16> v721_acc{};
              tensorforge::intel_esimd::simd<float, 16> v723_data = tensorforge::slmLoad<float, 16>(s1 + (102_i32));
              v721_acc += ((static_cast<float>(v723_data[0])) * v453_data);
              v721_acc += ((static_cast<float>(v723_data[1])) * v455_data);
              v721_acc += ((static_cast<float>(v723_data[2])) * v457_data);
              v721_acc += ((static_cast<float>(v723_data[3])) * v459_data);
              v721_acc += ((static_cast<float>(v723_data[4])) * v461_data);
              v721_acc += ((static_cast<float>(v723_data[5])) * v463_data);
              v721_acc += ((static_cast<float>(v723_data[6])) * v465_data);
              v721_acc += ((static_cast<float>(v723_data[7])) * v467_data);
              v721_acc += ((static_cast<float>(v723_data[8])) * v469_data);
              v721_acc += ((static_cast<float>(v723_data[9])) * v471_data);
              v721_acc += ((static_cast<float>(v723_data[10])) * v473_data);
              v721_acc += ((static_cast<float>(v723_data[11])) * v475_data);
              v721_acc += ((static_cast<float>(v723_data[12])) * v477_data);
              v721_acc += ((static_cast<float>(v723_data[13])) * v479_data);
              v721_acc += ((static_cast<float>(v723_data[14])) * v481_data);
              v721_acc += ((static_cast<float>(v723_data[15])) * v483_data);
              tensorforge::intel_esimd::simd<float, 16> v757_data = tensorforge::slmLoad<float, 16>(s1 + (118_i32));
              v721_acc += ((static_cast<float>(v757_data[0])) * v485_data);
              ir1.template select<16, 1>(96) = v721_acc;
              tensorforge::intel_esimd::simd<float, 16> v760_acc{};
              tensorforge::intel_esimd::simd<float, 16> v762_data = tensorforge::slmLoad<float, 16>(s1 + (119_i32));
              v760_acc += ((static_cast<float>(v762_data[0])) * v453_data);
              v760_acc += ((static_cast<float>(v762_data[1])) * v455_data);
              v760_acc += ((static_cast<float>(v762_data[2])) * v457_data);
              v760_acc += ((static_cast<float>(v762_data[3])) * v459_data);
              v760_acc += ((static_cast<float>(v762_data[4])) * v461_data);
              v760_acc += ((static_cast<float>(v762_data[5])) * v463_data);
              v760_acc += ((static_cast<float>(v762_data[6])) * v465_data);
              v760_acc += ((static_cast<float>(v762_data[7])) * v467_data);
              v760_acc += ((static_cast<float>(v762_data[8])) * v469_data);
              v760_acc += ((static_cast<float>(v762_data[9])) * v471_data);
              v760_acc += ((static_cast<float>(v762_data[10])) * v473_data);
              v760_acc += ((static_cast<float>(v762_data[11])) * v475_data);
              v760_acc += ((static_cast<float>(v762_data[12])) * v477_data);
              v760_acc += ((static_cast<float>(v762_data[13])) * v479_data);
              v760_acc += ((static_cast<float>(v762_data[14])) * v481_data);
              v760_acc += ((static_cast<float>(v762_data[15])) * v483_data);
              tensorforge::intel_esimd::simd<float, 16> v796_data = tensorforge::slmLoad<float, 16>(s1 + (135_i32));
              v760_acc += ((static_cast<float>(v796_data[0])) * v485_data);
              ir1.template select<16, 1>(112) = v760_acc;
              tensorforge::intel_esimd::simd<float, 16> v799_acc{};
              tensorforge::intel_esimd::simd<float, 16> v801_data = tensorforge::slmLoad<float, 16>(s1 + (136_i32));
              v799_acc += ((static_cast<float>(v801_data[0])) * v453_data);
              v799_acc += ((static_cast<float>(v801_data[1])) * v455_data);
              v799_acc += ((static_cast<float>(v801_data[2])) * v457_data);
              v799_acc += ((static_cast<float>(v801_data[3])) * v459_data);
              v799_acc += ((static_cast<float>(v801_data[4])) * v461_data);
              v799_acc += ((static_cast<float>(v801_data[5])) * v463_data);
              v799_acc += ((static_cast<float>(v801_data[6])) * v465_data);
              v799_acc += ((static_cast<float>(v801_data[7])) * v467_data);
              v799_acc += ((static_cast<float>(v801_data[8])) * v469_data);
              v799_acc += ((static_cast<float>(v801_data[9])) * v471_data);
              v799_acc += ((static_cast<float>(v801_data[10])) * v473_data);
              v799_acc += ((static_cast<float>(v801_data[11])) * v475_data);
              v799_acc += ((static_cast<float>(v801_data[12])) * v477_data);
              v799_acc += ((static_cast<float>(v801_data[13])) * v479_data);
              v799_acc += ((static_cast<float>(v801_data[14])) * v481_data);
              v799_acc += ((static_cast<float>(v801_data[15])) * v483_data);
              tensorforge::intel_esimd::simd<float, 16> v835_data = tensorforge::slmLoad<float, 16>(s1 + (152_i32));
              v799_acc += ((static_cast<float>(v835_data[0])) * v485_data);
              ir1.template select<16, 1>(128) = v799_acc;
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v838_n1 = 0; v838_n1 < 9; ++v838_n1) {
                int32_t v839_a = v838_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v841_data(ir1.template select<10, 1>(v839_a));
                tensorforge::intel_esimd::simd<float, 10> v842_data(r0.template select<10, 1>(v839_a));
                r1.template select<10, 1>(v839_a) = (v842_data + v841_data);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v844_i1 = 0; v844_i1 < 9; ++v844_i1) {
                tensorforge::intel_esimd::simd<float, 10> v847_data(r1.template select<10, 1>((v844_i1 * 16)));
                v847_data.copy_to(glb_m0 + ((v844_i1 * 10)));
              }
            }
          }
        }
      }
    });
  });
}


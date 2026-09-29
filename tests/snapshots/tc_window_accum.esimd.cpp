// === base name ===
kernel_70f83df4796cf6f3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_70f83df4796cf6f3 = {{1, 32, 1}, 16, 10, 1, 32, 46528, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_70f83df4796cf6f3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_70f83df4796cf6f3(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_70f83df4796cf6f3(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_70f83df4796cf6f3(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_70f83df4796cf6f3(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_70f83df4796cf6f3(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_70f83df4796cf6f3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<11632 * sizeof(float)>(); {
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":11632}],"shared_bytes":46528,"shared_elements":11632,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[10,19]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[1,0],[19,9]],"name":"m4","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,19]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[19,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (352 * item.get_local_id(1) + 368);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (336);
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
            tensorforge::intel_esimd::simd<float, 16> v28_ld;
            v28_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v28_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 4> v29_ld;
            v29_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 4>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v29_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (160);
          for (size_t v33_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v33_batchId0 < numElements0; v33_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v34_ahead1 = v33_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v36_batchId1 = (v34_ahead1 < numElements0) ? v34_ahead1 : v33_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v33_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v33_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v33_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v33_batchId0 * 162 + 0 + m4_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v44_ld;
              v44_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v44_ld);
              tensorforge::intel_esimd::simd<float, 64> v45_ld;
              v45_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v45_ld);
              tensorforge::intel_esimd::simd<float, 16> v46_ld;
              v46_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v46_ld);
              tensorforge::intel_esimd::simd<float, 9> v47_ld;
              v47_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v47_ld);
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v48_ld;
              v48_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v48_ld);
              tensorforge::intel_esimd::simd<float, 64> v49_ld;
              v49_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 64), v49_ld);
              tensorforge::intel_esimd::simd<float, 32> v50_ld;
              v50_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 128));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 2 * 0 + 128), v50_ld);
              tensorforge::intel_esimd::simd<float, 2> v51_ld;
              v51_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 2>(s1 + (0 + 0 + 1 * 0 + 160), v51_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v57_data = tensorforge::slmLoad<float, 16>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data = tensorforge::slmLoad<float, 16>(glb_m1 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v61_data = tensorforge::slmLoad<float, 16>(glb_m1 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data = tensorforge::slmLoad<float, 16>(glb_m1 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data = tensorforge::slmLoad<float, 16>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data = tensorforge::slmLoad<float, 16>(glb_m1 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v69_data = tensorforge::slmLoad<float, 16>(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data = tensorforge::slmLoad<float, 16>(glb_m1 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v73_data = tensorforge::slmLoad<float, 16>(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data = tensorforge::slmLoad<float, 16>(glb_m1 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v77_data = tensorforge::slmLoad<float, 16>(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v79_data = tensorforge::slmLoad<float, 16>(glb_m1 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v81_data = tensorforge::slmLoad<float, 16>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v83_data = tensorforge::slmLoad<float, 16>(glb_m1 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v85_data = tensorforge::slmLoad<float, 16>(glb_m1 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v87_data = tensorforge::slmLoad<float, 16>(glb_m1 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v89_data = tensorforge::slmLoad<float, 16>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v90_acc{};
              tensorforge::intel_esimd::simd<float, 16> v93_data(0.0f);
              v93_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v90_acc += ((static_cast<float>(v93_data[1])) * v57_data);
              v90_acc += ((static_cast<float>(v93_data[2])) * v59_data);
              v90_acc += ((static_cast<float>(v93_data[3])) * v61_data);
              v90_acc += ((static_cast<float>(v93_data[4])) * v63_data);
              v90_acc += ((static_cast<float>(v93_data[5])) * v65_data);
              v90_acc += ((static_cast<float>(v93_data[6])) * v67_data);
              v90_acc += ((static_cast<float>(v93_data[7])) * v69_data);
              v90_acc += ((static_cast<float>(v93_data[8])) * v71_data);
              v90_acc += ((static_cast<float>(v93_data[9])) * v73_data);
              v90_acc += ((static_cast<float>(v93_data[10])) * v75_data);
              v90_acc += ((static_cast<float>(v93_data[11])) * v77_data);
              v90_acc += ((static_cast<float>(v93_data[12])) * v79_data);
              v90_acc += ((static_cast<float>(v93_data[13])) * v81_data);
              v90_acc += ((static_cast<float>(v93_data[14])) * v83_data);
              v90_acc += ((static_cast<float>(v93_data[15])) * v85_data);
              tensorforge::intel_esimd::simd<float, 16> v129_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v90_acc += ((static_cast<float>(v129_data[0])) * v87_data);
              v90_acc += ((static_cast<float>(v129_data[1])) * v89_data);
              ir0.template select<16, 1>(0) = v90_acc;
              tensorforge::intel_esimd::simd<float, 16> v134_acc{};
              tensorforge::intel_esimd::simd<float, 16> v136_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v134_acc += ((static_cast<float>(v136_data[1])) * v57_data);
              v134_acc += ((static_cast<float>(v136_data[2])) * v59_data);
              v134_acc += ((static_cast<float>(v136_data[3])) * v61_data);
              v134_acc += ((static_cast<float>(v136_data[4])) * v63_data);
              v134_acc += ((static_cast<float>(v136_data[5])) * v65_data);
              v134_acc += ((static_cast<float>(v136_data[6])) * v67_data);
              v134_acc += ((static_cast<float>(v136_data[7])) * v69_data);
              v134_acc += ((static_cast<float>(v136_data[8])) * v71_data);
              v134_acc += ((static_cast<float>(v136_data[9])) * v73_data);
              v134_acc += ((static_cast<float>(v136_data[10])) * v75_data);
              v134_acc += ((static_cast<float>(v136_data[11])) * v77_data);
              v134_acc += ((static_cast<float>(v136_data[12])) * v79_data);
              v134_acc += ((static_cast<float>(v136_data[13])) * v81_data);
              v134_acc += ((static_cast<float>(v136_data[14])) * v83_data);
              v134_acc += ((static_cast<float>(v136_data[15])) * v85_data);
              tensorforge::intel_esimd::simd<float, 16> v169_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v134_acc += ((static_cast<float>(v169_data[0])) * v87_data);
              v134_acc += ((static_cast<float>(v169_data[1])) * v89_data);
              ir0.template select<16, 1>(16) = v134_acc;
              tensorforge::intel_esimd::simd<float, 16> v174_acc{};
              tensorforge::intel_esimd::simd<float, 16> v176_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v174_acc += ((static_cast<float>(v176_data[1])) * v57_data);
              v174_acc += ((static_cast<float>(v176_data[2])) * v59_data);
              v174_acc += ((static_cast<float>(v176_data[3])) * v61_data);
              v174_acc += ((static_cast<float>(v176_data[4])) * v63_data);
              v174_acc += ((static_cast<float>(v176_data[5])) * v65_data);
              v174_acc += ((static_cast<float>(v176_data[6])) * v67_data);
              v174_acc += ((static_cast<float>(v176_data[7])) * v69_data);
              v174_acc += ((static_cast<float>(v176_data[8])) * v71_data);
              v174_acc += ((static_cast<float>(v176_data[9])) * v73_data);
              v174_acc += ((static_cast<float>(v176_data[10])) * v75_data);
              v174_acc += ((static_cast<float>(v176_data[11])) * v77_data);
              v174_acc += ((static_cast<float>(v176_data[12])) * v79_data);
              v174_acc += ((static_cast<float>(v176_data[13])) * v81_data);
              v174_acc += ((static_cast<float>(v176_data[14])) * v83_data);
              v174_acc += ((static_cast<float>(v176_data[15])) * v85_data);
              tensorforge::intel_esimd::simd<float, 16> v209_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v174_acc += ((static_cast<float>(v209_data[0])) * v87_data);
              v174_acc += ((static_cast<float>(v209_data[1])) * v89_data);
              ir0.template select<16, 1>(32) = v174_acc;
              tensorforge::intel_esimd::simd<float, 16> v214_acc{};
              tensorforge::intel_esimd::simd<float, 16> v216_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v214_acc += ((static_cast<float>(v216_data[1])) * v57_data);
              v214_acc += ((static_cast<float>(v216_data[2])) * v59_data);
              v214_acc += ((static_cast<float>(v216_data[3])) * v61_data);
              v214_acc += ((static_cast<float>(v216_data[4])) * v63_data);
              v214_acc += ((static_cast<float>(v216_data[5])) * v65_data);
              v214_acc += ((static_cast<float>(v216_data[6])) * v67_data);
              v214_acc += ((static_cast<float>(v216_data[7])) * v69_data);
              v214_acc += ((static_cast<float>(v216_data[8])) * v71_data);
              v214_acc += ((static_cast<float>(v216_data[9])) * v73_data);
              v214_acc += ((static_cast<float>(v216_data[10])) * v75_data);
              v214_acc += ((static_cast<float>(v216_data[11])) * v77_data);
              v214_acc += ((static_cast<float>(v216_data[12])) * v79_data);
              v214_acc += ((static_cast<float>(v216_data[13])) * v81_data);
              v214_acc += ((static_cast<float>(v216_data[14])) * v83_data);
              v214_acc += ((static_cast<float>(v216_data[15])) * v85_data);
              tensorforge::intel_esimd::simd<float, 16> v249_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v214_acc += ((static_cast<float>(v249_data[0])) * v87_data);
              v214_acc += ((static_cast<float>(v249_data[1])) * v89_data);
              ir0.template select<16, 1>(48) = v214_acc;
              tensorforge::intel_esimd::simd<float, 16> v254_acc{};
              tensorforge::intel_esimd::simd<float, 16> v256_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v254_acc += ((static_cast<float>(v256_data[1])) * v57_data);
              v254_acc += ((static_cast<float>(v256_data[2])) * v59_data);
              v254_acc += ((static_cast<float>(v256_data[3])) * v61_data);
              v254_acc += ((static_cast<float>(v256_data[4])) * v63_data);
              v254_acc += ((static_cast<float>(v256_data[5])) * v65_data);
              v254_acc += ((static_cast<float>(v256_data[6])) * v67_data);
              v254_acc += ((static_cast<float>(v256_data[7])) * v69_data);
              v254_acc += ((static_cast<float>(v256_data[8])) * v71_data);
              v254_acc += ((static_cast<float>(v256_data[9])) * v73_data);
              v254_acc += ((static_cast<float>(v256_data[10])) * v75_data);
              v254_acc += ((static_cast<float>(v256_data[11])) * v77_data);
              v254_acc += ((static_cast<float>(v256_data[12])) * v79_data);
              v254_acc += ((static_cast<float>(v256_data[13])) * v81_data);
              v254_acc += ((static_cast<float>(v256_data[14])) * v83_data);
              v254_acc += ((static_cast<float>(v256_data[15])) * v85_data);
              tensorforge::intel_esimd::simd<float, 16> v289_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v254_acc += ((static_cast<float>(v289_data[0])) * v87_data);
              v254_acc += ((static_cast<float>(v289_data[1])) * v89_data);
              ir0.template select<16, 1>(64) = v254_acc;
              tensorforge::intel_esimd::simd<float, 16> v294_acc{};
              tensorforge::intel_esimd::simd<float, 16> v296_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v294_acc += ((static_cast<float>(v296_data[1])) * v57_data);
              v294_acc += ((static_cast<float>(v296_data[2])) * v59_data);
              v294_acc += ((static_cast<float>(v296_data[3])) * v61_data);
              v294_acc += ((static_cast<float>(v296_data[4])) * v63_data);
              v294_acc += ((static_cast<float>(v296_data[5])) * v65_data);
              v294_acc += ((static_cast<float>(v296_data[6])) * v67_data);
              v294_acc += ((static_cast<float>(v296_data[7])) * v69_data);
              v294_acc += ((static_cast<float>(v296_data[8])) * v71_data);
              v294_acc += ((static_cast<float>(v296_data[9])) * v73_data);
              v294_acc += ((static_cast<float>(v296_data[10])) * v75_data);
              v294_acc += ((static_cast<float>(v296_data[11])) * v77_data);
              v294_acc += ((static_cast<float>(v296_data[12])) * v79_data);
              v294_acc += ((static_cast<float>(v296_data[13])) * v81_data);
              v294_acc += ((static_cast<float>(v296_data[14])) * v83_data);
              v294_acc += ((static_cast<float>(v296_data[15])) * v85_data);
              tensorforge::intel_esimd::simd<float, 16> v329_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v294_acc += ((static_cast<float>(v329_data[0])) * v87_data);
              v294_acc += ((static_cast<float>(v329_data[1])) * v89_data);
              ir0.template select<16, 1>(80) = v294_acc;
              tensorforge::intel_esimd::simd<float, 16> v334_acc{};
              tensorforge::intel_esimd::simd<float, 16> v336_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v334_acc += ((static_cast<float>(v336_data[1])) * v57_data);
              v334_acc += ((static_cast<float>(v336_data[2])) * v59_data);
              v334_acc += ((static_cast<float>(v336_data[3])) * v61_data);
              v334_acc += ((static_cast<float>(v336_data[4])) * v63_data);
              v334_acc += ((static_cast<float>(v336_data[5])) * v65_data);
              v334_acc += ((static_cast<float>(v336_data[6])) * v67_data);
              v334_acc += ((static_cast<float>(v336_data[7])) * v69_data);
              v334_acc += ((static_cast<float>(v336_data[8])) * v71_data);
              v334_acc += ((static_cast<float>(v336_data[9])) * v73_data);
              v334_acc += ((static_cast<float>(v336_data[10])) * v75_data);
              v334_acc += ((static_cast<float>(v336_data[11])) * v77_data);
              v334_acc += ((static_cast<float>(v336_data[12])) * v79_data);
              v334_acc += ((static_cast<float>(v336_data[13])) * v81_data);
              v334_acc += ((static_cast<float>(v336_data[14])) * v83_data);
              v334_acc += ((static_cast<float>(v336_data[15])) * v85_data);
              tensorforge::intel_esimd::simd<float, 16> v369_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v334_acc += ((static_cast<float>(v369_data[0])) * v87_data);
              v334_acc += ((static_cast<float>(v369_data[1])) * v89_data);
              ir0.template select<16, 1>(96) = v334_acc;
              tensorforge::intel_esimd::simd<float, 16> v374_acc{};
              tensorforge::intel_esimd::simd<float, 16> v376_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v374_acc += ((static_cast<float>(v376_data[1])) * v57_data);
              v374_acc += ((static_cast<float>(v376_data[2])) * v59_data);
              v374_acc += ((static_cast<float>(v376_data[3])) * v61_data);
              v374_acc += ((static_cast<float>(v376_data[4])) * v63_data);
              v374_acc += ((static_cast<float>(v376_data[5])) * v65_data);
              v374_acc += ((static_cast<float>(v376_data[6])) * v67_data);
              v374_acc += ((static_cast<float>(v376_data[7])) * v69_data);
              v374_acc += ((static_cast<float>(v376_data[8])) * v71_data);
              v374_acc += ((static_cast<float>(v376_data[9])) * v73_data);
              v374_acc += ((static_cast<float>(v376_data[10])) * v75_data);
              v374_acc += ((static_cast<float>(v376_data[11])) * v77_data);
              v374_acc += ((static_cast<float>(v376_data[12])) * v79_data);
              v374_acc += ((static_cast<float>(v376_data[13])) * v81_data);
              v374_acc += ((static_cast<float>(v376_data[14])) * v83_data);
              v374_acc += ((static_cast<float>(v376_data[15])) * v85_data);
              tensorforge::intel_esimd::simd<float, 16> v409_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v374_acc += ((static_cast<float>(v409_data[0])) * v87_data);
              v374_acc += ((static_cast<float>(v409_data[1])) * v89_data);
              ir0.template select<16, 1>(112) = v374_acc;
              tensorforge::intel_esimd::simd<float, 16> v414_acc{};
              tensorforge::intel_esimd::simd<float, 16> v416_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v414_acc += ((static_cast<float>(v416_data[1])) * v57_data);
              v414_acc += ((static_cast<float>(v416_data[2])) * v59_data);
              v414_acc += ((static_cast<float>(v416_data[3])) * v61_data);
              v414_acc += ((static_cast<float>(v416_data[4])) * v63_data);
              v414_acc += ((static_cast<float>(v416_data[5])) * v65_data);
              v414_acc += ((static_cast<float>(v416_data[6])) * v67_data);
              v414_acc += ((static_cast<float>(v416_data[7])) * v69_data);
              v414_acc += ((static_cast<float>(v416_data[8])) * v71_data);
              v414_acc += ((static_cast<float>(v416_data[9])) * v73_data);
              v414_acc += ((static_cast<float>(v416_data[10])) * v75_data);
              v414_acc += ((static_cast<float>(v416_data[11])) * v77_data);
              v414_acc += ((static_cast<float>(v416_data[12])) * v79_data);
              v414_acc += ((static_cast<float>(v416_data[13])) * v81_data);
              v414_acc += ((static_cast<float>(v416_data[14])) * v83_data);
              v414_acc += ((static_cast<float>(v416_data[15])) * v85_data);
              tensorforge::intel_esimd::simd<float, 16> v449_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v414_acc += ((static_cast<float>(v449_data[0])) * v87_data);
              v414_acc += ((static_cast<float>(v449_data[1])) * v89_data);
              ir0.template select<16, 1>(128) = v414_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v454_n1 = 0; v454_n1 < 9; ++v454_n1) {
                int32_t v455_a = v454_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v457_data(ir0.template select<10, 1>(v455_a));
                r0.template select<10, 1>(v455_a) = v457_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 10), (0, 9)] [(1, 19)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v463_data = tensorforge::slmLoad<float, 16>(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v465_data = tensorforge::slmLoad<float, 16>(glb_m3 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v467_data = tensorforge::slmLoad<float, 16>(glb_m3 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v469_data = tensorforge::slmLoad<float, 16>(glb_m3 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v471_data = tensorforge::slmLoad<float, 16>(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v473_data = tensorforge::slmLoad<float, 16>(glb_m3 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v475_data = tensorforge::slmLoad<float, 16>(glb_m3 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v477_data = tensorforge::slmLoad<float, 16>(glb_m3 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v479_data = tensorforge::slmLoad<float, 16>(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v481_data = tensorforge::slmLoad<float, 16>(glb_m3 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v483_data = tensorforge::slmLoad<float, 16>(glb_m3 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v485_data = tensorforge::slmLoad<float, 16>(glb_m3 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v487_data = tensorforge::slmLoad<float, 16>(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v489_data = tensorforge::slmLoad<float, 16>(glb_m3 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v491_data = tensorforge::slmLoad<float, 16>(glb_m3 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v493_data = tensorforge::slmLoad<float, 16>(glb_m3 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v495_data = tensorforge::slmLoad<float, 16>(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v497_data = tensorforge::slmLoad<float, 16>(glb_m3 + (170_i32));
              tensorforge::intel_esimd::simd<float, 16> v498_acc{};
              tensorforge::intel_esimd::simd<float, 16> v501_data(0.0f);
              v501_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s1 + (-1_i32)) + 1);
              v498_acc += ((static_cast<float>(v501_data[1])) * v463_data);
              v498_acc += ((static_cast<float>(v501_data[2])) * v465_data);
              v498_acc += ((static_cast<float>(v501_data[3])) * v467_data);
              v498_acc += ((static_cast<float>(v501_data[4])) * v469_data);
              v498_acc += ((static_cast<float>(v501_data[5])) * v471_data);
              v498_acc += ((static_cast<float>(v501_data[6])) * v473_data);
              v498_acc += ((static_cast<float>(v501_data[7])) * v475_data);
              v498_acc += ((static_cast<float>(v501_data[8])) * v477_data);
              v498_acc += ((static_cast<float>(v501_data[9])) * v479_data);
              v498_acc += ((static_cast<float>(v501_data[10])) * v481_data);
              v498_acc += ((static_cast<float>(v501_data[11])) * v483_data);
              v498_acc += ((static_cast<float>(v501_data[12])) * v485_data);
              v498_acc += ((static_cast<float>(v501_data[13])) * v487_data);
              v498_acc += ((static_cast<float>(v501_data[14])) * v489_data);
              v498_acc += ((static_cast<float>(v501_data[15])) * v491_data);
              tensorforge::intel_esimd::simd<float, 16> v537_data = tensorforge::slmLoad<float, 16>(s1 + (15_i32));
              v498_acc += ((static_cast<float>(v537_data[0])) * v493_data);
              v498_acc += ((static_cast<float>(v537_data[1])) * v495_data);
              v498_acc += ((static_cast<float>(v537_data[2])) * v497_data);
              ir1.template select<16, 1>(0) = v498_acc;
              tensorforge::intel_esimd::simd<float, 16> v544_acc{};
              tensorforge::intel_esimd::simd<float, 16> v546_data = tensorforge::slmLoad<float, 16>(s1 + (17_i32));
              v544_acc += ((static_cast<float>(v546_data[1])) * v463_data);
              v544_acc += ((static_cast<float>(v546_data[2])) * v465_data);
              v544_acc += ((static_cast<float>(v546_data[3])) * v467_data);
              v544_acc += ((static_cast<float>(v546_data[4])) * v469_data);
              v544_acc += ((static_cast<float>(v546_data[5])) * v471_data);
              v544_acc += ((static_cast<float>(v546_data[6])) * v473_data);
              v544_acc += ((static_cast<float>(v546_data[7])) * v475_data);
              v544_acc += ((static_cast<float>(v546_data[8])) * v477_data);
              v544_acc += ((static_cast<float>(v546_data[9])) * v479_data);
              v544_acc += ((static_cast<float>(v546_data[10])) * v481_data);
              v544_acc += ((static_cast<float>(v546_data[11])) * v483_data);
              v544_acc += ((static_cast<float>(v546_data[12])) * v485_data);
              v544_acc += ((static_cast<float>(v546_data[13])) * v487_data);
              v544_acc += ((static_cast<float>(v546_data[14])) * v489_data);
              v544_acc += ((static_cast<float>(v546_data[15])) * v491_data);
              tensorforge::intel_esimd::simd<float, 16> v579_data = tensorforge::slmLoad<float, 16>(s1 + (33_i32));
              v544_acc += ((static_cast<float>(v579_data[0])) * v493_data);
              v544_acc += ((static_cast<float>(v579_data[1])) * v495_data);
              v544_acc += ((static_cast<float>(v579_data[2])) * v497_data);
              ir1.template select<16, 1>(16) = v544_acc;
              tensorforge::intel_esimd::simd<float, 16> v586_acc{};
              tensorforge::intel_esimd::simd<float, 16> v588_data = tensorforge::slmLoad<float, 16>(s1 + (35_i32));
              v586_acc += ((static_cast<float>(v588_data[1])) * v463_data);
              v586_acc += ((static_cast<float>(v588_data[2])) * v465_data);
              v586_acc += ((static_cast<float>(v588_data[3])) * v467_data);
              v586_acc += ((static_cast<float>(v588_data[4])) * v469_data);
              v586_acc += ((static_cast<float>(v588_data[5])) * v471_data);
              v586_acc += ((static_cast<float>(v588_data[6])) * v473_data);
              v586_acc += ((static_cast<float>(v588_data[7])) * v475_data);
              v586_acc += ((static_cast<float>(v588_data[8])) * v477_data);
              v586_acc += ((static_cast<float>(v588_data[9])) * v479_data);
              v586_acc += ((static_cast<float>(v588_data[10])) * v481_data);
              v586_acc += ((static_cast<float>(v588_data[11])) * v483_data);
              v586_acc += ((static_cast<float>(v588_data[12])) * v485_data);
              v586_acc += ((static_cast<float>(v588_data[13])) * v487_data);
              v586_acc += ((static_cast<float>(v588_data[14])) * v489_data);
              v586_acc += ((static_cast<float>(v588_data[15])) * v491_data);
              tensorforge::intel_esimd::simd<float, 16> v621_data = tensorforge::slmLoad<float, 16>(s1 + (51_i32));
              v586_acc += ((static_cast<float>(v621_data[0])) * v493_data);
              v586_acc += ((static_cast<float>(v621_data[1])) * v495_data);
              v586_acc += ((static_cast<float>(v621_data[2])) * v497_data);
              ir1.template select<16, 1>(32) = v586_acc;
              tensorforge::intel_esimd::simd<float, 16> v628_acc{};
              tensorforge::intel_esimd::simd<float, 16> v630_data = tensorforge::slmLoad<float, 16>(s1 + (53_i32));
              v628_acc += ((static_cast<float>(v630_data[1])) * v463_data);
              v628_acc += ((static_cast<float>(v630_data[2])) * v465_data);
              v628_acc += ((static_cast<float>(v630_data[3])) * v467_data);
              v628_acc += ((static_cast<float>(v630_data[4])) * v469_data);
              v628_acc += ((static_cast<float>(v630_data[5])) * v471_data);
              v628_acc += ((static_cast<float>(v630_data[6])) * v473_data);
              v628_acc += ((static_cast<float>(v630_data[7])) * v475_data);
              v628_acc += ((static_cast<float>(v630_data[8])) * v477_data);
              v628_acc += ((static_cast<float>(v630_data[9])) * v479_data);
              v628_acc += ((static_cast<float>(v630_data[10])) * v481_data);
              v628_acc += ((static_cast<float>(v630_data[11])) * v483_data);
              v628_acc += ((static_cast<float>(v630_data[12])) * v485_data);
              v628_acc += ((static_cast<float>(v630_data[13])) * v487_data);
              v628_acc += ((static_cast<float>(v630_data[14])) * v489_data);
              v628_acc += ((static_cast<float>(v630_data[15])) * v491_data);
              tensorforge::intel_esimd::simd<float, 16> v663_data = tensorforge::slmLoad<float, 16>(s1 + (69_i32));
              v628_acc += ((static_cast<float>(v663_data[0])) * v493_data);
              v628_acc += ((static_cast<float>(v663_data[1])) * v495_data);
              v628_acc += ((static_cast<float>(v663_data[2])) * v497_data);
              ir1.template select<16, 1>(48) = v628_acc;
              tensorforge::intel_esimd::simd<float, 16> v670_acc{};
              tensorforge::intel_esimd::simd<float, 16> v672_data = tensorforge::slmLoad<float, 16>(s1 + (71_i32));
              v670_acc += ((static_cast<float>(v672_data[1])) * v463_data);
              v670_acc += ((static_cast<float>(v672_data[2])) * v465_data);
              v670_acc += ((static_cast<float>(v672_data[3])) * v467_data);
              v670_acc += ((static_cast<float>(v672_data[4])) * v469_data);
              v670_acc += ((static_cast<float>(v672_data[5])) * v471_data);
              v670_acc += ((static_cast<float>(v672_data[6])) * v473_data);
              v670_acc += ((static_cast<float>(v672_data[7])) * v475_data);
              v670_acc += ((static_cast<float>(v672_data[8])) * v477_data);
              v670_acc += ((static_cast<float>(v672_data[9])) * v479_data);
              v670_acc += ((static_cast<float>(v672_data[10])) * v481_data);
              v670_acc += ((static_cast<float>(v672_data[11])) * v483_data);
              v670_acc += ((static_cast<float>(v672_data[12])) * v485_data);
              v670_acc += ((static_cast<float>(v672_data[13])) * v487_data);
              v670_acc += ((static_cast<float>(v672_data[14])) * v489_data);
              v670_acc += ((static_cast<float>(v672_data[15])) * v491_data);
              tensorforge::intel_esimd::simd<float, 16> v705_data = tensorforge::slmLoad<float, 16>(s1 + (87_i32));
              v670_acc += ((static_cast<float>(v705_data[0])) * v493_data);
              v670_acc += ((static_cast<float>(v705_data[1])) * v495_data);
              v670_acc += ((static_cast<float>(v705_data[2])) * v497_data);
              ir1.template select<16, 1>(64) = v670_acc;
              tensorforge::intel_esimd::simd<float, 16> v712_acc{};
              tensorforge::intel_esimd::simd<float, 16> v714_data = tensorforge::slmLoad<float, 16>(s1 + (89_i32));
              v712_acc += ((static_cast<float>(v714_data[1])) * v463_data);
              v712_acc += ((static_cast<float>(v714_data[2])) * v465_data);
              v712_acc += ((static_cast<float>(v714_data[3])) * v467_data);
              v712_acc += ((static_cast<float>(v714_data[4])) * v469_data);
              v712_acc += ((static_cast<float>(v714_data[5])) * v471_data);
              v712_acc += ((static_cast<float>(v714_data[6])) * v473_data);
              v712_acc += ((static_cast<float>(v714_data[7])) * v475_data);
              v712_acc += ((static_cast<float>(v714_data[8])) * v477_data);
              v712_acc += ((static_cast<float>(v714_data[9])) * v479_data);
              v712_acc += ((static_cast<float>(v714_data[10])) * v481_data);
              v712_acc += ((static_cast<float>(v714_data[11])) * v483_data);
              v712_acc += ((static_cast<float>(v714_data[12])) * v485_data);
              v712_acc += ((static_cast<float>(v714_data[13])) * v487_data);
              v712_acc += ((static_cast<float>(v714_data[14])) * v489_data);
              v712_acc += ((static_cast<float>(v714_data[15])) * v491_data);
              tensorforge::intel_esimd::simd<float, 16> v747_data = tensorforge::slmLoad<float, 16>(s1 + (105_i32));
              v712_acc += ((static_cast<float>(v747_data[0])) * v493_data);
              v712_acc += ((static_cast<float>(v747_data[1])) * v495_data);
              v712_acc += ((static_cast<float>(v747_data[2])) * v497_data);
              ir1.template select<16, 1>(80) = v712_acc;
              tensorforge::intel_esimd::simd<float, 16> v754_acc{};
              tensorforge::intel_esimd::simd<float, 16> v756_data = tensorforge::slmLoad<float, 16>(s1 + (107_i32));
              v754_acc += ((static_cast<float>(v756_data[1])) * v463_data);
              v754_acc += ((static_cast<float>(v756_data[2])) * v465_data);
              v754_acc += ((static_cast<float>(v756_data[3])) * v467_data);
              v754_acc += ((static_cast<float>(v756_data[4])) * v469_data);
              v754_acc += ((static_cast<float>(v756_data[5])) * v471_data);
              v754_acc += ((static_cast<float>(v756_data[6])) * v473_data);
              v754_acc += ((static_cast<float>(v756_data[7])) * v475_data);
              v754_acc += ((static_cast<float>(v756_data[8])) * v477_data);
              v754_acc += ((static_cast<float>(v756_data[9])) * v479_data);
              v754_acc += ((static_cast<float>(v756_data[10])) * v481_data);
              v754_acc += ((static_cast<float>(v756_data[11])) * v483_data);
              v754_acc += ((static_cast<float>(v756_data[12])) * v485_data);
              v754_acc += ((static_cast<float>(v756_data[13])) * v487_data);
              v754_acc += ((static_cast<float>(v756_data[14])) * v489_data);
              v754_acc += ((static_cast<float>(v756_data[15])) * v491_data);
              tensorforge::intel_esimd::simd<float, 16> v789_data = tensorforge::slmLoad<float, 16>(s1 + (123_i32));
              v754_acc += ((static_cast<float>(v789_data[0])) * v493_data);
              v754_acc += ((static_cast<float>(v789_data[1])) * v495_data);
              v754_acc += ((static_cast<float>(v789_data[2])) * v497_data);
              ir1.template select<16, 1>(96) = v754_acc;
              tensorforge::intel_esimd::simd<float, 16> v796_acc{};
              tensorforge::intel_esimd::simd<float, 16> v798_data = tensorforge::slmLoad<float, 16>(s1 + (125_i32));
              v796_acc += ((static_cast<float>(v798_data[1])) * v463_data);
              v796_acc += ((static_cast<float>(v798_data[2])) * v465_data);
              v796_acc += ((static_cast<float>(v798_data[3])) * v467_data);
              v796_acc += ((static_cast<float>(v798_data[4])) * v469_data);
              v796_acc += ((static_cast<float>(v798_data[5])) * v471_data);
              v796_acc += ((static_cast<float>(v798_data[6])) * v473_data);
              v796_acc += ((static_cast<float>(v798_data[7])) * v475_data);
              v796_acc += ((static_cast<float>(v798_data[8])) * v477_data);
              v796_acc += ((static_cast<float>(v798_data[9])) * v479_data);
              v796_acc += ((static_cast<float>(v798_data[10])) * v481_data);
              v796_acc += ((static_cast<float>(v798_data[11])) * v483_data);
              v796_acc += ((static_cast<float>(v798_data[12])) * v485_data);
              v796_acc += ((static_cast<float>(v798_data[13])) * v487_data);
              v796_acc += ((static_cast<float>(v798_data[14])) * v489_data);
              v796_acc += ((static_cast<float>(v798_data[15])) * v491_data);
              tensorforge::intel_esimd::simd<float, 16> v831_data = tensorforge::slmLoad<float, 16>(s1 + (141_i32));
              v796_acc += ((static_cast<float>(v831_data[0])) * v493_data);
              v796_acc += ((static_cast<float>(v831_data[1])) * v495_data);
              v796_acc += ((static_cast<float>(v831_data[2])) * v497_data);
              ir1.template select<16, 1>(112) = v796_acc;
              tensorforge::intel_esimd::simd<float, 16> v838_acc{};
              tensorforge::intel_esimd::simd<float, 16> v840_data = tensorforge::slmLoad<float, 16>(s1 + (143_i32));
              v838_acc += ((static_cast<float>(v840_data[1])) * v463_data);
              v838_acc += ((static_cast<float>(v840_data[2])) * v465_data);
              v838_acc += ((static_cast<float>(v840_data[3])) * v467_data);
              v838_acc += ((static_cast<float>(v840_data[4])) * v469_data);
              v838_acc += ((static_cast<float>(v840_data[5])) * v471_data);
              v838_acc += ((static_cast<float>(v840_data[6])) * v473_data);
              v838_acc += ((static_cast<float>(v840_data[7])) * v475_data);
              v838_acc += ((static_cast<float>(v840_data[8])) * v477_data);
              v838_acc += ((static_cast<float>(v840_data[9])) * v479_data);
              v838_acc += ((static_cast<float>(v840_data[10])) * v481_data);
              v838_acc += ((static_cast<float>(v840_data[11])) * v483_data);
              v838_acc += ((static_cast<float>(v840_data[12])) * v485_data);
              v838_acc += ((static_cast<float>(v840_data[13])) * v487_data);
              v838_acc += ((static_cast<float>(v840_data[14])) * v489_data);
              v838_acc += ((static_cast<float>(v840_data[15])) * v491_data);
              tensorforge::intel_esimd::simd<float, 16> v873_data = tensorforge::slmLoad<float, 16>(s1 + (159_i32));
              v838_acc += ((static_cast<float>(v873_data[0])) * v493_data);
              v838_acc += ((static_cast<float>(v873_data[1])) * v495_data);
              v838_acc += ((static_cast<float>(v873_data[2])) * v497_data);
              ir1.template select<16, 1>(128) = v838_acc;
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v880_n1 = 0; v880_n1 < 9; ++v880_n1) {
                int32_t v881_a = v880_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v883_data(ir1.template select<10, 1>(v881_a));
                tensorforge::intel_esimd::simd<float, 10> v884_data(r0.template select<10, 1>(v881_a));
                r1.template select<10, 1>(v881_a) = (v884_data + v883_data);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v886_i1 = 0; v886_i1 < 9; ++v886_i1) {
                tensorforge::intel_esimd::simd<float, 10> v889_data(r1.template select<10, 1>((v886_i1 * 16)));
                v889_data.copy_to(glb_m0 + ((v886_i1 * 10)));
              }
            }
          }
        }
      }
    });
  });
}


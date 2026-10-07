// === base name ===
kernel_1cd9f16b7283acd1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1cd9f16b7283acd1 = {{1, 32, 1}, 16, 16, 1, 32, 24576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1cd9f16b7283acd1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1cd9f16b7283acd1(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1cd9f16b7283acd1(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_1cd9f16b7283acd1(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1cd9f16b7283acd1(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1cd9f16b7283acd1(stream, grid, block, m0, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1cd9f16b7283acd1(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 512);
          const float *const __restrict__ ptr_glb_m0 = &m0[0];
          tensorforge::SlmPtr<float> glb_m0 = totalShrMem + (0);
          // glb_m0 = load{g>s}(ptr_glb_m0[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v9_ld;
            v9_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v9_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v10_ld;
            v10_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v10_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v11_ld;
            v11_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v11_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v12_ld;
            v12_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v13_ld;
            v13_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v14_ld;
            v14_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v15_ld;
            v15_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v16_ld;
            v16_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v17_ld;
            v17_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<float, 16> v21_ld;
            v21_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v21_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<float, 16> v22_ld;
            v22_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v22_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<float, 16> v23_ld;
            v23_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v23_ld);
          }
          if (item.get_local_id(1) == 15) {
            tensorforge::intel_esimd::simd<float, 16> v24_ld;
            v24_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v24_ld);
          }
          if (item.get_local_id(1) == 16) {
            tensorforge::intel_esimd::simd<float, 16> v25_ld;
            v25_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v25_ld);
          }
          const float *const __restrict__ ptr_glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> glb_m3 = totalShrMem + (272);
          // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v28_ld;
            v28_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v28_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v29_ld;
            v29_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v29_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v30_ld;
            v30_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v30_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v31_ld;
            v31_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v31_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v32_ld;
            v32_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v32_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v33_ld;
            v33_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v33_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v34_ld;
            v34_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v34_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v35_ld;
            v35_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v35_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v36_ld;
            v36_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v36_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v37_ld;
            v37_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v37_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 16> v38_ld;
            v38_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v38_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 16> v39_ld;
            v39_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v39_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<float, 16> v40_ld;
            v40_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v40_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<float, 16> v41_ld;
            v41_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v41_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<float, 16> v42_ld;
            v42_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v42_ld);
          }
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v45_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v45_batchId0 < numElements0; v45_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v46_ahead1 = v45_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v48_batchId1 = (v46_ahead1 < numElements0) ? v46_ahead1 : v45_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v45_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m1 = &m1[v45_batchId0 * 153 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v45_batchId0 * 144 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v55_ld;
              v55_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v55_ld);
              tensorforge::intel_esimd::simd<float, 64> v56_ld;
              v56_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v56_ld);
              tensorforge::intel_esimd::simd<float, 16> v57_ld;
              v57_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v57_ld);
              tensorforge::intel_esimd::simd<float, 9> v58_ld;
              v58_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v58_ld);
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // r0 = +(glb_m0 * s0) + None
              // [(0, 16), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run0 = tensorforge::slmLoad<float, 64>(glb_m0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data(glb_m0_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v65_data(glb_m0_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v67_data(glb_m0_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v69_data(glb_m0_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run1 = tensorforge::slmLoad<float, 64>(glb_m0 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data(glb_m0_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v73_data(glb_m0_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v75_data(glb_m0_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v77_data(glb_m0_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run2 = tensorforge::slmLoad<float, 64>(glb_m0 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v79_data(glb_m0_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v81_data(glb_m0_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v83_data(glb_m0_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v85_data(glb_m0_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run3 = tensorforge::slmLoad<float, 64>(glb_m0 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v87_data(glb_m0_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v89_data(glb_m0_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v91_data(glb_m0_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v93_data(glb_m0_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v95_data = tensorforge::slmLoad<float, 16>(glb_m0 + (256_i32));
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
              r0.template select<16, 1>(0) = v96_acc;
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
              r0.template select<16, 1>(16) = v140_acc;
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
              r0.template select<16, 1>(32) = v180_acc;
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
              r0.template select<16, 1>(48) = v220_acc;
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
              r0.template select<16, 1>(64) = v260_acc;
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
              r0.template select<16, 1>(80) = v300_acc;
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
              r0.template select<16, 1>(96) = v340_acc;
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
              r0.template select<16, 1>(112) = v380_acc;
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
              r0.template select<16, 1>(128) = v420_acc;
              // s1 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v460_i0 = 0; v460_i0 < 1; ++v460_i0) {
                int32_t v462_a = v460_i0 * 16;
                #pragma unroll
                for (int32_t v461_i1 = 0; v461_i1 < 9; ++v461_i1) {
                  int32_t v464_a = v462_a + (v461_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v465_data(r0.template select<16, 1>(v464_a));
                  tensorforge::slmStore<float, 16>(s1 + (v464_a), v465_data);
                }
              }
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 16), (0, 9)] [(1, 16)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run4 = tensorforge::slmLoad<float, 64>(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v473_data(glb_m3_run4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v475_data(glb_m3_run4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v477_data(glb_m3_run4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v479_data(glb_m3_run4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run5 = tensorforge::slmLoad<float, 64>(glb_m3 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v481_data(glb_m3_run5.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v483_data(glb_m3_run5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v485_data(glb_m3_run5.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v487_data(glb_m3_run5.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run6 = tensorforge::slmLoad<float, 64>(glb_m3 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v489_data(glb_m3_run6.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v491_data(glb_m3_run6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v493_data(glb_m3_run6.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v495_data(glb_m3_run6.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 48> glb_m3_run7 = tensorforge::slmLoad<float, 48>(glb_m3 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v497_data(glb_m3_run7.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v499_data(glb_m3_run7.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v501_data(glb_m3_run7.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v502_acc{};
              tensorforge::intel_esimd::simd<float, 16> v503_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v502_acc += ((static_cast<float>(v503_data[1])) * v473_data);
              v502_acc += ((static_cast<float>(v503_data[2])) * v475_data);
              v502_acc += ((static_cast<float>(v503_data[3])) * v477_data);
              v502_acc += ((static_cast<float>(v503_data[4])) * v479_data);
              v502_acc += ((static_cast<float>(v503_data[5])) * v481_data);
              v502_acc += ((static_cast<float>(v503_data[6])) * v483_data);
              v502_acc += ((static_cast<float>(v503_data[7])) * v485_data);
              v502_acc += ((static_cast<float>(v503_data[8])) * v487_data);
              v502_acc += ((static_cast<float>(v503_data[9])) * v489_data);
              v502_acc += ((static_cast<float>(v503_data[10])) * v491_data);
              v502_acc += ((static_cast<float>(v503_data[11])) * v493_data);
              v502_acc += ((static_cast<float>(v503_data[12])) * v495_data);
              v502_acc += ((static_cast<float>(v503_data[13])) * v497_data);
              v502_acc += ((static_cast<float>(v503_data[14])) * v499_data);
              v502_acc += ((static_cast<float>(v503_data[15])) * v501_data);
              ir1.template select<16, 1>(0) = v502_acc;
              tensorforge::intel_esimd::simd<float, 16> v535_acc{};
              tensorforge::intel_esimd::simd<float, 16> v536_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v535_acc += ((static_cast<float>(v536_data[1])) * v473_data);
              v535_acc += ((static_cast<float>(v536_data[2])) * v475_data);
              v535_acc += ((static_cast<float>(v536_data[3])) * v477_data);
              v535_acc += ((static_cast<float>(v536_data[4])) * v479_data);
              v535_acc += ((static_cast<float>(v536_data[5])) * v481_data);
              v535_acc += ((static_cast<float>(v536_data[6])) * v483_data);
              v535_acc += ((static_cast<float>(v536_data[7])) * v485_data);
              v535_acc += ((static_cast<float>(v536_data[8])) * v487_data);
              v535_acc += ((static_cast<float>(v536_data[9])) * v489_data);
              v535_acc += ((static_cast<float>(v536_data[10])) * v491_data);
              v535_acc += ((static_cast<float>(v536_data[11])) * v493_data);
              v535_acc += ((static_cast<float>(v536_data[12])) * v495_data);
              v535_acc += ((static_cast<float>(v536_data[13])) * v497_data);
              v535_acc += ((static_cast<float>(v536_data[14])) * v499_data);
              v535_acc += ((static_cast<float>(v536_data[15])) * v501_data);
              ir1.template select<16, 1>(16) = v535_acc;
              tensorforge::intel_esimd::simd<float, 16> v568_acc{};
              tensorforge::intel_esimd::simd<float, 16> v569_data = tensorforge::slmLoad<float, 16>(s1 + (32_i32));
              v568_acc += ((static_cast<float>(v569_data[1])) * v473_data);
              v568_acc += ((static_cast<float>(v569_data[2])) * v475_data);
              v568_acc += ((static_cast<float>(v569_data[3])) * v477_data);
              v568_acc += ((static_cast<float>(v569_data[4])) * v479_data);
              v568_acc += ((static_cast<float>(v569_data[5])) * v481_data);
              v568_acc += ((static_cast<float>(v569_data[6])) * v483_data);
              v568_acc += ((static_cast<float>(v569_data[7])) * v485_data);
              v568_acc += ((static_cast<float>(v569_data[8])) * v487_data);
              v568_acc += ((static_cast<float>(v569_data[9])) * v489_data);
              v568_acc += ((static_cast<float>(v569_data[10])) * v491_data);
              v568_acc += ((static_cast<float>(v569_data[11])) * v493_data);
              v568_acc += ((static_cast<float>(v569_data[12])) * v495_data);
              v568_acc += ((static_cast<float>(v569_data[13])) * v497_data);
              v568_acc += ((static_cast<float>(v569_data[14])) * v499_data);
              v568_acc += ((static_cast<float>(v569_data[15])) * v501_data);
              ir1.template select<16, 1>(32) = v568_acc;
              tensorforge::intel_esimd::simd<float, 16> v601_acc{};
              tensorforge::intel_esimd::simd<float, 16> v602_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v601_acc += ((static_cast<float>(v602_data[1])) * v473_data);
              v601_acc += ((static_cast<float>(v602_data[2])) * v475_data);
              v601_acc += ((static_cast<float>(v602_data[3])) * v477_data);
              v601_acc += ((static_cast<float>(v602_data[4])) * v479_data);
              v601_acc += ((static_cast<float>(v602_data[5])) * v481_data);
              v601_acc += ((static_cast<float>(v602_data[6])) * v483_data);
              v601_acc += ((static_cast<float>(v602_data[7])) * v485_data);
              v601_acc += ((static_cast<float>(v602_data[8])) * v487_data);
              v601_acc += ((static_cast<float>(v602_data[9])) * v489_data);
              v601_acc += ((static_cast<float>(v602_data[10])) * v491_data);
              v601_acc += ((static_cast<float>(v602_data[11])) * v493_data);
              v601_acc += ((static_cast<float>(v602_data[12])) * v495_data);
              v601_acc += ((static_cast<float>(v602_data[13])) * v497_data);
              v601_acc += ((static_cast<float>(v602_data[14])) * v499_data);
              v601_acc += ((static_cast<float>(v602_data[15])) * v501_data);
              ir1.template select<16, 1>(48) = v601_acc;
              tensorforge::intel_esimd::simd<float, 16> v634_acc{};
              tensorforge::intel_esimd::simd<float, 16> v635_data = tensorforge::slmLoad<float, 16>(s1 + (64_i32));
              v634_acc += ((static_cast<float>(v635_data[1])) * v473_data);
              v634_acc += ((static_cast<float>(v635_data[2])) * v475_data);
              v634_acc += ((static_cast<float>(v635_data[3])) * v477_data);
              v634_acc += ((static_cast<float>(v635_data[4])) * v479_data);
              v634_acc += ((static_cast<float>(v635_data[5])) * v481_data);
              v634_acc += ((static_cast<float>(v635_data[6])) * v483_data);
              v634_acc += ((static_cast<float>(v635_data[7])) * v485_data);
              v634_acc += ((static_cast<float>(v635_data[8])) * v487_data);
              v634_acc += ((static_cast<float>(v635_data[9])) * v489_data);
              v634_acc += ((static_cast<float>(v635_data[10])) * v491_data);
              v634_acc += ((static_cast<float>(v635_data[11])) * v493_data);
              v634_acc += ((static_cast<float>(v635_data[12])) * v495_data);
              v634_acc += ((static_cast<float>(v635_data[13])) * v497_data);
              v634_acc += ((static_cast<float>(v635_data[14])) * v499_data);
              v634_acc += ((static_cast<float>(v635_data[15])) * v501_data);
              ir1.template select<16, 1>(64) = v634_acc;
              tensorforge::intel_esimd::simd<float, 16> v667_acc{};
              tensorforge::intel_esimd::simd<float, 16> v668_data = tensorforge::slmLoad<float, 16>(s1 + (80_i32));
              v667_acc += ((static_cast<float>(v668_data[1])) * v473_data);
              v667_acc += ((static_cast<float>(v668_data[2])) * v475_data);
              v667_acc += ((static_cast<float>(v668_data[3])) * v477_data);
              v667_acc += ((static_cast<float>(v668_data[4])) * v479_data);
              v667_acc += ((static_cast<float>(v668_data[5])) * v481_data);
              v667_acc += ((static_cast<float>(v668_data[6])) * v483_data);
              v667_acc += ((static_cast<float>(v668_data[7])) * v485_data);
              v667_acc += ((static_cast<float>(v668_data[8])) * v487_data);
              v667_acc += ((static_cast<float>(v668_data[9])) * v489_data);
              v667_acc += ((static_cast<float>(v668_data[10])) * v491_data);
              v667_acc += ((static_cast<float>(v668_data[11])) * v493_data);
              v667_acc += ((static_cast<float>(v668_data[12])) * v495_data);
              v667_acc += ((static_cast<float>(v668_data[13])) * v497_data);
              v667_acc += ((static_cast<float>(v668_data[14])) * v499_data);
              v667_acc += ((static_cast<float>(v668_data[15])) * v501_data);
              ir1.template select<16, 1>(80) = v667_acc;
              tensorforge::intel_esimd::simd<float, 16> v700_acc{};
              tensorforge::intel_esimd::simd<float, 16> v701_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v700_acc += ((static_cast<float>(v701_data[1])) * v473_data);
              v700_acc += ((static_cast<float>(v701_data[2])) * v475_data);
              v700_acc += ((static_cast<float>(v701_data[3])) * v477_data);
              v700_acc += ((static_cast<float>(v701_data[4])) * v479_data);
              v700_acc += ((static_cast<float>(v701_data[5])) * v481_data);
              v700_acc += ((static_cast<float>(v701_data[6])) * v483_data);
              v700_acc += ((static_cast<float>(v701_data[7])) * v485_data);
              v700_acc += ((static_cast<float>(v701_data[8])) * v487_data);
              v700_acc += ((static_cast<float>(v701_data[9])) * v489_data);
              v700_acc += ((static_cast<float>(v701_data[10])) * v491_data);
              v700_acc += ((static_cast<float>(v701_data[11])) * v493_data);
              v700_acc += ((static_cast<float>(v701_data[12])) * v495_data);
              v700_acc += ((static_cast<float>(v701_data[13])) * v497_data);
              v700_acc += ((static_cast<float>(v701_data[14])) * v499_data);
              v700_acc += ((static_cast<float>(v701_data[15])) * v501_data);
              ir1.template select<16, 1>(96) = v700_acc;
              tensorforge::intel_esimd::simd<float, 16> v733_acc{};
              tensorforge::intel_esimd::simd<float, 16> v734_data = tensorforge::slmLoad<float, 16>(s1 + (112_i32));
              v733_acc += ((static_cast<float>(v734_data[1])) * v473_data);
              v733_acc += ((static_cast<float>(v734_data[2])) * v475_data);
              v733_acc += ((static_cast<float>(v734_data[3])) * v477_data);
              v733_acc += ((static_cast<float>(v734_data[4])) * v479_data);
              v733_acc += ((static_cast<float>(v734_data[5])) * v481_data);
              v733_acc += ((static_cast<float>(v734_data[6])) * v483_data);
              v733_acc += ((static_cast<float>(v734_data[7])) * v485_data);
              v733_acc += ((static_cast<float>(v734_data[8])) * v487_data);
              v733_acc += ((static_cast<float>(v734_data[9])) * v489_data);
              v733_acc += ((static_cast<float>(v734_data[10])) * v491_data);
              v733_acc += ((static_cast<float>(v734_data[11])) * v493_data);
              v733_acc += ((static_cast<float>(v734_data[12])) * v495_data);
              v733_acc += ((static_cast<float>(v734_data[13])) * v497_data);
              v733_acc += ((static_cast<float>(v734_data[14])) * v499_data);
              v733_acc += ((static_cast<float>(v734_data[15])) * v501_data);
              ir1.template select<16, 1>(112) = v733_acc;
              tensorforge::intel_esimd::simd<float, 16> v766_acc{};
              tensorforge::intel_esimd::simd<float, 16> v767_data = tensorforge::slmLoad<float, 16>(s1 + (128_i32));
              v766_acc += ((static_cast<float>(v767_data[1])) * v473_data);
              v766_acc += ((static_cast<float>(v767_data[2])) * v475_data);
              v766_acc += ((static_cast<float>(v767_data[3])) * v477_data);
              v766_acc += ((static_cast<float>(v767_data[4])) * v479_data);
              v766_acc += ((static_cast<float>(v767_data[5])) * v481_data);
              v766_acc += ((static_cast<float>(v767_data[6])) * v483_data);
              v766_acc += ((static_cast<float>(v767_data[7])) * v485_data);
              v766_acc += ((static_cast<float>(v767_data[8])) * v487_data);
              v766_acc += ((static_cast<float>(v767_data[9])) * v489_data);
              v766_acc += ((static_cast<float>(v767_data[10])) * v491_data);
              v766_acc += ((static_cast<float>(v767_data[11])) * v493_data);
              v766_acc += ((static_cast<float>(v767_data[12])) * v495_data);
              v766_acc += ((static_cast<float>(v767_data[13])) * v497_data);
              v766_acc += ((static_cast<float>(v767_data[14])) * v499_data);
              v766_acc += ((static_cast<float>(v767_data[15])) * v501_data);
              ir1.template select<16, 1>(128) = v766_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v799_n0 = 0; v799_n0 < 1; ++v799_n0) {
                int32_t v801_a = v799_n0 * 16;
                #pragma unroll
                for (int32_t v800_n1 = 0; v800_n1 < 9; ++v800_n1) {
                  int32_t v803_a = v801_a + (v800_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v804_data(ir1.template select<16, 1>(v803_a));
                  r1.template select<16, 1>(v803_a) = v804_data;
                }
              }
              // glb_m2 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v805_i0 = 0; v805_i0 < 1; ++v805_i0) {
                int32_t v807_a = v805_i0 * 16;
                #pragma unroll
                for (int32_t v806_i1 = 0; v806_i1 < 9; ++v806_i1) {
                  int32_t v809_a = v807_a + (v806_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v810_data(r1.template select<16, 1>(v809_a));
                  v810_data.copy_to(glb_m2 + (v809_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


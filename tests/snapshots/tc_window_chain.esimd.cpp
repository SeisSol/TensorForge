// === base name ===
kernel_2295eabad0bc8d60

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2295eabad0bc8d60 = {{1, 32, 1}, 16, 16, 1, 32, 24576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2295eabad0bc8d60(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2295eabad0bc8d60(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2295eabad0bc8d60(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_2295eabad0bc8d60(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2295eabad0bc8d60(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2295eabad0bc8d60(stream, grid, block, m0, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2295eabad0bc8d60(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (160);
          const float *const __restrict__ ptr_glb_m0 = &m0[0];
          tensorforge::SlmPtr<float> glb_m0 = totalShrMem + (0);
          // glb_m0 = load{g>s}(ptr_glb_m0[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v12_ld;
            v12_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v13_ld;
            v13_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v14_ld;
            v14_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v15_ld;
            v15_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v16_ld;
            v16_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v17_ld;
            v17_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v21_ld;
            v21_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v21_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 16> v22_ld;
            v22_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v22_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 16> v23_ld;
            v23_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v23_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<float, 16> v24_ld;
            v24_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v24_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<float, 16> v25_ld;
            v25_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v25_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<float, 16> v26_ld;
            v26_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v26_ld);
          }
          if (item.get_local_id(1) == 15) {
            tensorforge::intel_esimd::simd<float, 16> v27_ld;
            v27_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v27_ld);
          }
          if (item.get_local_id(1) == 16) {
            tensorforge::intel_esimd::simd<float, 16> v28_ld;
            v28_ld.copy_from(ptr_glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m0 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v28_ld);
          }
          const float *const __restrict__ ptr_glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> glb_m3 = totalShrMem + (272);
          // glb_m3 = load{g>s}(ptr_glb_m3[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 16> v31_ld;
            v31_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v31_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 16> v32_ld;
            v32_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v32_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 16> v33_ld;
            v33_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v33_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 16> v34_ld;
            v34_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v34_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 16> v35_ld;
            v35_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v35_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 16> v36_ld;
            v36_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v36_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 16> v37_ld;
            v37_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v37_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 16> v38_ld;
            v38_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v38_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 16> v39_ld;
            v39_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v39_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 16> v40_ld;
            v40_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v40_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<float, 16> v41_ld;
            v41_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v41_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 16> v42_ld;
            v42_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v42_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<float, 16> v43_ld;
            v43_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v43_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<float, 16> v44_ld;
            v44_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v44_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<float, 16> v45_ld;
            v45_ld.copy_from(ptr_glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m3 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v45_ld);
          }
          // wait(glb_m0 = load{g>s}(ptr_glb_m0[0, 1]));
          // wait(glb_m3 = load{g>s}(ptr_glb_m3[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v48_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v48_batchId0 < numElements0; v48_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v49_ahead1 = v48_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v51_batchId1 = (v49_ahead1 < numElements0) ? v49_ahead1 : v48_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v48_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m1 = &m1[v48_batchId0 * 153 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v48_batchId0 * 144 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v58_ld;
              v58_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v58_ld);
              tensorforge::intel_esimd::simd<float, 64> v59_ld;
              v59_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v59_ld);
              tensorforge::intel_esimd::simd<float, 16> v60_ld;
              v60_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v60_ld);
              tensorforge::intel_esimd::simd<float, 9> v61_ld;
              v61_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v61_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // r0 = +(glb_m0 * s0) + None
              // [(0, 16), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run0 = tensorforge::slmLoad<float, 64>(glb_m0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data(glb_m0_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v68_data(glb_m0_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v70_data(glb_m0_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v72_data(glb_m0_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run1 = tensorforge::slmLoad<float, 64>(glb_m0 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v74_data(glb_m0_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v76_data(glb_m0_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v78_data(glb_m0_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v80_data(glb_m0_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run2 = tensorforge::slmLoad<float, 64>(glb_m0 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v82_data(glb_m0_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v84_data(glb_m0_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v86_data(glb_m0_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v88_data(glb_m0_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run3 = tensorforge::slmLoad<float, 64>(glb_m0 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v90_data(glb_m0_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v92_data(glb_m0_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v94_data(glb_m0_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v96_data(glb_m0_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v98_data = tensorforge::slmLoad<float, 16>(glb_m0 + (256_i32));
              tensorforge::intel_esimd::simd<float, 16> v99_acc{};
              tensorforge::intel_esimd::simd<float, 16> v102_data(0.0f);
              v102_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v99_acc += ((static_cast<float>(v102_data[1])) * v66_data);
              v99_acc += ((static_cast<float>(v102_data[2])) * v68_data);
              v99_acc += ((static_cast<float>(v102_data[3])) * v70_data);
              v99_acc += ((static_cast<float>(v102_data[4])) * v72_data);
              v99_acc += ((static_cast<float>(v102_data[5])) * v74_data);
              v99_acc += ((static_cast<float>(v102_data[6])) * v76_data);
              v99_acc += ((static_cast<float>(v102_data[7])) * v78_data);
              v99_acc += ((static_cast<float>(v102_data[8])) * v80_data);
              v99_acc += ((static_cast<float>(v102_data[9])) * v82_data);
              v99_acc += ((static_cast<float>(v102_data[10])) * v84_data);
              v99_acc += ((static_cast<float>(v102_data[11])) * v86_data);
              v99_acc += ((static_cast<float>(v102_data[12])) * v88_data);
              v99_acc += ((static_cast<float>(v102_data[13])) * v90_data);
              v99_acc += ((static_cast<float>(v102_data[14])) * v92_data);
              v99_acc += ((static_cast<float>(v102_data[15])) * v94_data);
              tensorforge::intel_esimd::simd<float, 16> v138_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v99_acc += ((static_cast<float>(v138_data[0])) * v96_data);
              v99_acc += ((static_cast<float>(v138_data[1])) * v98_data);
              r0.template select<16, 1>(0) = v99_acc;
              tensorforge::intel_esimd::simd<float, 16> v143_acc{};
              tensorforge::intel_esimd::simd<float, 16> v145_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v143_acc += ((static_cast<float>(v145_data[1])) * v66_data);
              v143_acc += ((static_cast<float>(v145_data[2])) * v68_data);
              v143_acc += ((static_cast<float>(v145_data[3])) * v70_data);
              v143_acc += ((static_cast<float>(v145_data[4])) * v72_data);
              v143_acc += ((static_cast<float>(v145_data[5])) * v74_data);
              v143_acc += ((static_cast<float>(v145_data[6])) * v76_data);
              v143_acc += ((static_cast<float>(v145_data[7])) * v78_data);
              v143_acc += ((static_cast<float>(v145_data[8])) * v80_data);
              v143_acc += ((static_cast<float>(v145_data[9])) * v82_data);
              v143_acc += ((static_cast<float>(v145_data[10])) * v84_data);
              v143_acc += ((static_cast<float>(v145_data[11])) * v86_data);
              v143_acc += ((static_cast<float>(v145_data[12])) * v88_data);
              v143_acc += ((static_cast<float>(v145_data[13])) * v90_data);
              v143_acc += ((static_cast<float>(v145_data[14])) * v92_data);
              v143_acc += ((static_cast<float>(v145_data[15])) * v94_data);
              tensorforge::intel_esimd::simd<float, 16> v178_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v143_acc += ((static_cast<float>(v178_data[0])) * v96_data);
              v143_acc += ((static_cast<float>(v178_data[1])) * v98_data);
              r0.template select<16, 1>(16) = v143_acc;
              tensorforge::intel_esimd::simd<float, 16> v183_acc{};
              tensorforge::intel_esimd::simd<float, 16> v185_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v183_acc += ((static_cast<float>(v185_data[1])) * v66_data);
              v183_acc += ((static_cast<float>(v185_data[2])) * v68_data);
              v183_acc += ((static_cast<float>(v185_data[3])) * v70_data);
              v183_acc += ((static_cast<float>(v185_data[4])) * v72_data);
              v183_acc += ((static_cast<float>(v185_data[5])) * v74_data);
              v183_acc += ((static_cast<float>(v185_data[6])) * v76_data);
              v183_acc += ((static_cast<float>(v185_data[7])) * v78_data);
              v183_acc += ((static_cast<float>(v185_data[8])) * v80_data);
              v183_acc += ((static_cast<float>(v185_data[9])) * v82_data);
              v183_acc += ((static_cast<float>(v185_data[10])) * v84_data);
              v183_acc += ((static_cast<float>(v185_data[11])) * v86_data);
              v183_acc += ((static_cast<float>(v185_data[12])) * v88_data);
              v183_acc += ((static_cast<float>(v185_data[13])) * v90_data);
              v183_acc += ((static_cast<float>(v185_data[14])) * v92_data);
              v183_acc += ((static_cast<float>(v185_data[15])) * v94_data);
              tensorforge::intel_esimd::simd<float, 16> v218_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v183_acc += ((static_cast<float>(v218_data[0])) * v96_data);
              v183_acc += ((static_cast<float>(v218_data[1])) * v98_data);
              r0.template select<16, 1>(32) = v183_acc;
              tensorforge::intel_esimd::simd<float, 16> v223_acc{};
              tensorforge::intel_esimd::simd<float, 16> v225_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v223_acc += ((static_cast<float>(v225_data[1])) * v66_data);
              v223_acc += ((static_cast<float>(v225_data[2])) * v68_data);
              v223_acc += ((static_cast<float>(v225_data[3])) * v70_data);
              v223_acc += ((static_cast<float>(v225_data[4])) * v72_data);
              v223_acc += ((static_cast<float>(v225_data[5])) * v74_data);
              v223_acc += ((static_cast<float>(v225_data[6])) * v76_data);
              v223_acc += ((static_cast<float>(v225_data[7])) * v78_data);
              v223_acc += ((static_cast<float>(v225_data[8])) * v80_data);
              v223_acc += ((static_cast<float>(v225_data[9])) * v82_data);
              v223_acc += ((static_cast<float>(v225_data[10])) * v84_data);
              v223_acc += ((static_cast<float>(v225_data[11])) * v86_data);
              v223_acc += ((static_cast<float>(v225_data[12])) * v88_data);
              v223_acc += ((static_cast<float>(v225_data[13])) * v90_data);
              v223_acc += ((static_cast<float>(v225_data[14])) * v92_data);
              v223_acc += ((static_cast<float>(v225_data[15])) * v94_data);
              tensorforge::intel_esimd::simd<float, 16> v258_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v223_acc += ((static_cast<float>(v258_data[0])) * v96_data);
              v223_acc += ((static_cast<float>(v258_data[1])) * v98_data);
              r0.template select<16, 1>(48) = v223_acc;
              tensorforge::intel_esimd::simd<float, 16> v263_acc{};
              tensorforge::intel_esimd::simd<float, 16> v265_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v263_acc += ((static_cast<float>(v265_data[1])) * v66_data);
              v263_acc += ((static_cast<float>(v265_data[2])) * v68_data);
              v263_acc += ((static_cast<float>(v265_data[3])) * v70_data);
              v263_acc += ((static_cast<float>(v265_data[4])) * v72_data);
              v263_acc += ((static_cast<float>(v265_data[5])) * v74_data);
              v263_acc += ((static_cast<float>(v265_data[6])) * v76_data);
              v263_acc += ((static_cast<float>(v265_data[7])) * v78_data);
              v263_acc += ((static_cast<float>(v265_data[8])) * v80_data);
              v263_acc += ((static_cast<float>(v265_data[9])) * v82_data);
              v263_acc += ((static_cast<float>(v265_data[10])) * v84_data);
              v263_acc += ((static_cast<float>(v265_data[11])) * v86_data);
              v263_acc += ((static_cast<float>(v265_data[12])) * v88_data);
              v263_acc += ((static_cast<float>(v265_data[13])) * v90_data);
              v263_acc += ((static_cast<float>(v265_data[14])) * v92_data);
              v263_acc += ((static_cast<float>(v265_data[15])) * v94_data);
              tensorforge::intel_esimd::simd<float, 16> v298_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v263_acc += ((static_cast<float>(v298_data[0])) * v96_data);
              v263_acc += ((static_cast<float>(v298_data[1])) * v98_data);
              r0.template select<16, 1>(64) = v263_acc;
              tensorforge::intel_esimd::simd<float, 16> v303_acc{};
              tensorforge::intel_esimd::simd<float, 16> v305_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v303_acc += ((static_cast<float>(v305_data[1])) * v66_data);
              v303_acc += ((static_cast<float>(v305_data[2])) * v68_data);
              v303_acc += ((static_cast<float>(v305_data[3])) * v70_data);
              v303_acc += ((static_cast<float>(v305_data[4])) * v72_data);
              v303_acc += ((static_cast<float>(v305_data[5])) * v74_data);
              v303_acc += ((static_cast<float>(v305_data[6])) * v76_data);
              v303_acc += ((static_cast<float>(v305_data[7])) * v78_data);
              v303_acc += ((static_cast<float>(v305_data[8])) * v80_data);
              v303_acc += ((static_cast<float>(v305_data[9])) * v82_data);
              v303_acc += ((static_cast<float>(v305_data[10])) * v84_data);
              v303_acc += ((static_cast<float>(v305_data[11])) * v86_data);
              v303_acc += ((static_cast<float>(v305_data[12])) * v88_data);
              v303_acc += ((static_cast<float>(v305_data[13])) * v90_data);
              v303_acc += ((static_cast<float>(v305_data[14])) * v92_data);
              v303_acc += ((static_cast<float>(v305_data[15])) * v94_data);
              tensorforge::intel_esimd::simd<float, 16> v338_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v303_acc += ((static_cast<float>(v338_data[0])) * v96_data);
              v303_acc += ((static_cast<float>(v338_data[1])) * v98_data);
              r0.template select<16, 1>(80) = v303_acc;
              tensorforge::intel_esimd::simd<float, 16> v343_acc{};
              tensorforge::intel_esimd::simd<float, 16> v345_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v343_acc += ((static_cast<float>(v345_data[1])) * v66_data);
              v343_acc += ((static_cast<float>(v345_data[2])) * v68_data);
              v343_acc += ((static_cast<float>(v345_data[3])) * v70_data);
              v343_acc += ((static_cast<float>(v345_data[4])) * v72_data);
              v343_acc += ((static_cast<float>(v345_data[5])) * v74_data);
              v343_acc += ((static_cast<float>(v345_data[6])) * v76_data);
              v343_acc += ((static_cast<float>(v345_data[7])) * v78_data);
              v343_acc += ((static_cast<float>(v345_data[8])) * v80_data);
              v343_acc += ((static_cast<float>(v345_data[9])) * v82_data);
              v343_acc += ((static_cast<float>(v345_data[10])) * v84_data);
              v343_acc += ((static_cast<float>(v345_data[11])) * v86_data);
              v343_acc += ((static_cast<float>(v345_data[12])) * v88_data);
              v343_acc += ((static_cast<float>(v345_data[13])) * v90_data);
              v343_acc += ((static_cast<float>(v345_data[14])) * v92_data);
              v343_acc += ((static_cast<float>(v345_data[15])) * v94_data);
              tensorforge::intel_esimd::simd<float, 16> v378_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v343_acc += ((static_cast<float>(v378_data[0])) * v96_data);
              v343_acc += ((static_cast<float>(v378_data[1])) * v98_data);
              r0.template select<16, 1>(96) = v343_acc;
              tensorforge::intel_esimd::simd<float, 16> v383_acc{};
              tensorforge::intel_esimd::simd<float, 16> v385_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v383_acc += ((static_cast<float>(v385_data[1])) * v66_data);
              v383_acc += ((static_cast<float>(v385_data[2])) * v68_data);
              v383_acc += ((static_cast<float>(v385_data[3])) * v70_data);
              v383_acc += ((static_cast<float>(v385_data[4])) * v72_data);
              v383_acc += ((static_cast<float>(v385_data[5])) * v74_data);
              v383_acc += ((static_cast<float>(v385_data[6])) * v76_data);
              v383_acc += ((static_cast<float>(v385_data[7])) * v78_data);
              v383_acc += ((static_cast<float>(v385_data[8])) * v80_data);
              v383_acc += ((static_cast<float>(v385_data[9])) * v82_data);
              v383_acc += ((static_cast<float>(v385_data[10])) * v84_data);
              v383_acc += ((static_cast<float>(v385_data[11])) * v86_data);
              v383_acc += ((static_cast<float>(v385_data[12])) * v88_data);
              v383_acc += ((static_cast<float>(v385_data[13])) * v90_data);
              v383_acc += ((static_cast<float>(v385_data[14])) * v92_data);
              v383_acc += ((static_cast<float>(v385_data[15])) * v94_data);
              tensorforge::intel_esimd::simd<float, 16> v418_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v383_acc += ((static_cast<float>(v418_data[0])) * v96_data);
              v383_acc += ((static_cast<float>(v418_data[1])) * v98_data);
              r0.template select<16, 1>(112) = v383_acc;
              tensorforge::intel_esimd::simd<float, 16> v423_acc{};
              tensorforge::intel_esimd::simd<float, 16> v425_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v423_acc += ((static_cast<float>(v425_data[1])) * v66_data);
              v423_acc += ((static_cast<float>(v425_data[2])) * v68_data);
              v423_acc += ((static_cast<float>(v425_data[3])) * v70_data);
              v423_acc += ((static_cast<float>(v425_data[4])) * v72_data);
              v423_acc += ((static_cast<float>(v425_data[5])) * v74_data);
              v423_acc += ((static_cast<float>(v425_data[6])) * v76_data);
              v423_acc += ((static_cast<float>(v425_data[7])) * v78_data);
              v423_acc += ((static_cast<float>(v425_data[8])) * v80_data);
              v423_acc += ((static_cast<float>(v425_data[9])) * v82_data);
              v423_acc += ((static_cast<float>(v425_data[10])) * v84_data);
              v423_acc += ((static_cast<float>(v425_data[11])) * v86_data);
              v423_acc += ((static_cast<float>(v425_data[12])) * v88_data);
              v423_acc += ((static_cast<float>(v425_data[13])) * v90_data);
              v423_acc += ((static_cast<float>(v425_data[14])) * v92_data);
              v423_acc += ((static_cast<float>(v425_data[15])) * v94_data);
              tensorforge::intel_esimd::simd<float, 16> v458_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v423_acc += ((static_cast<float>(v458_data[0])) * v96_data);
              v423_acc += ((static_cast<float>(v458_data[1])) * v98_data);
              r0.template select<16, 1>(128) = v423_acc;
              // s1 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v463_i0 = 0; v463_i0 < 1; ++v463_i0) {
                int32_t v465_a = v463_i0 * 16;
                #pragma unroll
                for (int32_t v464_i1 = 0; v464_i1 < 9; ++v464_i1) {
                  int32_t v467_a = v465_a + (v464_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v468_data(r0.template select<16, 1>(v467_a));
                  tensorforge::slmStore<float, 16>(s1 + (v467_a), v468_data);
                }
              }
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 16), (0, 9)] [(1, 16)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run4 = tensorforge::slmLoad<float, 64>(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v476_data(glb_m3_run4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v478_data(glb_m3_run4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v480_data(glb_m3_run4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v482_data(glb_m3_run4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run5 = tensorforge::slmLoad<float, 64>(glb_m3 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v484_data(glb_m3_run5.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v486_data(glb_m3_run5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v488_data(glb_m3_run5.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v490_data(glb_m3_run5.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run6 = tensorforge::slmLoad<float, 64>(glb_m3 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v492_data(glb_m3_run6.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v494_data(glb_m3_run6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v496_data(glb_m3_run6.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v498_data(glb_m3_run6.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 48> glb_m3_run7 = tensorforge::slmLoad<float, 48>(glb_m3 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v500_data(glb_m3_run7.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v502_data(glb_m3_run7.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v504_data(glb_m3_run7.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v505_acc{};
              tensorforge::intel_esimd::simd<float, 16> v506_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v505_acc += ((static_cast<float>(v506_data[1])) * v476_data);
              v505_acc += ((static_cast<float>(v506_data[2])) * v478_data);
              v505_acc += ((static_cast<float>(v506_data[3])) * v480_data);
              v505_acc += ((static_cast<float>(v506_data[4])) * v482_data);
              v505_acc += ((static_cast<float>(v506_data[5])) * v484_data);
              v505_acc += ((static_cast<float>(v506_data[6])) * v486_data);
              v505_acc += ((static_cast<float>(v506_data[7])) * v488_data);
              v505_acc += ((static_cast<float>(v506_data[8])) * v490_data);
              v505_acc += ((static_cast<float>(v506_data[9])) * v492_data);
              v505_acc += ((static_cast<float>(v506_data[10])) * v494_data);
              v505_acc += ((static_cast<float>(v506_data[11])) * v496_data);
              v505_acc += ((static_cast<float>(v506_data[12])) * v498_data);
              v505_acc += ((static_cast<float>(v506_data[13])) * v500_data);
              v505_acc += ((static_cast<float>(v506_data[14])) * v502_data);
              v505_acc += ((static_cast<float>(v506_data[15])) * v504_data);
              ir1.template select<16, 1>(0) = v505_acc;
              tensorforge::intel_esimd::simd<float, 16> v538_acc{};
              tensorforge::intel_esimd::simd<float, 16> v539_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v538_acc += ((static_cast<float>(v539_data[1])) * v476_data);
              v538_acc += ((static_cast<float>(v539_data[2])) * v478_data);
              v538_acc += ((static_cast<float>(v539_data[3])) * v480_data);
              v538_acc += ((static_cast<float>(v539_data[4])) * v482_data);
              v538_acc += ((static_cast<float>(v539_data[5])) * v484_data);
              v538_acc += ((static_cast<float>(v539_data[6])) * v486_data);
              v538_acc += ((static_cast<float>(v539_data[7])) * v488_data);
              v538_acc += ((static_cast<float>(v539_data[8])) * v490_data);
              v538_acc += ((static_cast<float>(v539_data[9])) * v492_data);
              v538_acc += ((static_cast<float>(v539_data[10])) * v494_data);
              v538_acc += ((static_cast<float>(v539_data[11])) * v496_data);
              v538_acc += ((static_cast<float>(v539_data[12])) * v498_data);
              v538_acc += ((static_cast<float>(v539_data[13])) * v500_data);
              v538_acc += ((static_cast<float>(v539_data[14])) * v502_data);
              v538_acc += ((static_cast<float>(v539_data[15])) * v504_data);
              ir1.template select<16, 1>(16) = v538_acc;
              tensorforge::intel_esimd::simd<float, 16> v571_acc{};
              tensorforge::intel_esimd::simd<float, 16> v572_data = tensorforge::slmLoad<float, 16>(s1 + (32_i32));
              v571_acc += ((static_cast<float>(v572_data[1])) * v476_data);
              v571_acc += ((static_cast<float>(v572_data[2])) * v478_data);
              v571_acc += ((static_cast<float>(v572_data[3])) * v480_data);
              v571_acc += ((static_cast<float>(v572_data[4])) * v482_data);
              v571_acc += ((static_cast<float>(v572_data[5])) * v484_data);
              v571_acc += ((static_cast<float>(v572_data[6])) * v486_data);
              v571_acc += ((static_cast<float>(v572_data[7])) * v488_data);
              v571_acc += ((static_cast<float>(v572_data[8])) * v490_data);
              v571_acc += ((static_cast<float>(v572_data[9])) * v492_data);
              v571_acc += ((static_cast<float>(v572_data[10])) * v494_data);
              v571_acc += ((static_cast<float>(v572_data[11])) * v496_data);
              v571_acc += ((static_cast<float>(v572_data[12])) * v498_data);
              v571_acc += ((static_cast<float>(v572_data[13])) * v500_data);
              v571_acc += ((static_cast<float>(v572_data[14])) * v502_data);
              v571_acc += ((static_cast<float>(v572_data[15])) * v504_data);
              ir1.template select<16, 1>(32) = v571_acc;
              tensorforge::intel_esimd::simd<float, 16> v604_acc{};
              tensorforge::intel_esimd::simd<float, 16> v605_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v604_acc += ((static_cast<float>(v605_data[1])) * v476_data);
              v604_acc += ((static_cast<float>(v605_data[2])) * v478_data);
              v604_acc += ((static_cast<float>(v605_data[3])) * v480_data);
              v604_acc += ((static_cast<float>(v605_data[4])) * v482_data);
              v604_acc += ((static_cast<float>(v605_data[5])) * v484_data);
              v604_acc += ((static_cast<float>(v605_data[6])) * v486_data);
              v604_acc += ((static_cast<float>(v605_data[7])) * v488_data);
              v604_acc += ((static_cast<float>(v605_data[8])) * v490_data);
              v604_acc += ((static_cast<float>(v605_data[9])) * v492_data);
              v604_acc += ((static_cast<float>(v605_data[10])) * v494_data);
              v604_acc += ((static_cast<float>(v605_data[11])) * v496_data);
              v604_acc += ((static_cast<float>(v605_data[12])) * v498_data);
              v604_acc += ((static_cast<float>(v605_data[13])) * v500_data);
              v604_acc += ((static_cast<float>(v605_data[14])) * v502_data);
              v604_acc += ((static_cast<float>(v605_data[15])) * v504_data);
              ir1.template select<16, 1>(48) = v604_acc;
              tensorforge::intel_esimd::simd<float, 16> v637_acc{};
              tensorforge::intel_esimd::simd<float, 16> v638_data = tensorforge::slmLoad<float, 16>(s1 + (64_i32));
              v637_acc += ((static_cast<float>(v638_data[1])) * v476_data);
              v637_acc += ((static_cast<float>(v638_data[2])) * v478_data);
              v637_acc += ((static_cast<float>(v638_data[3])) * v480_data);
              v637_acc += ((static_cast<float>(v638_data[4])) * v482_data);
              v637_acc += ((static_cast<float>(v638_data[5])) * v484_data);
              v637_acc += ((static_cast<float>(v638_data[6])) * v486_data);
              v637_acc += ((static_cast<float>(v638_data[7])) * v488_data);
              v637_acc += ((static_cast<float>(v638_data[8])) * v490_data);
              v637_acc += ((static_cast<float>(v638_data[9])) * v492_data);
              v637_acc += ((static_cast<float>(v638_data[10])) * v494_data);
              v637_acc += ((static_cast<float>(v638_data[11])) * v496_data);
              v637_acc += ((static_cast<float>(v638_data[12])) * v498_data);
              v637_acc += ((static_cast<float>(v638_data[13])) * v500_data);
              v637_acc += ((static_cast<float>(v638_data[14])) * v502_data);
              v637_acc += ((static_cast<float>(v638_data[15])) * v504_data);
              ir1.template select<16, 1>(64) = v637_acc;
              tensorforge::intel_esimd::simd<float, 16> v670_acc{};
              tensorforge::intel_esimd::simd<float, 16> v671_data = tensorforge::slmLoad<float, 16>(s1 + (80_i32));
              v670_acc += ((static_cast<float>(v671_data[1])) * v476_data);
              v670_acc += ((static_cast<float>(v671_data[2])) * v478_data);
              v670_acc += ((static_cast<float>(v671_data[3])) * v480_data);
              v670_acc += ((static_cast<float>(v671_data[4])) * v482_data);
              v670_acc += ((static_cast<float>(v671_data[5])) * v484_data);
              v670_acc += ((static_cast<float>(v671_data[6])) * v486_data);
              v670_acc += ((static_cast<float>(v671_data[7])) * v488_data);
              v670_acc += ((static_cast<float>(v671_data[8])) * v490_data);
              v670_acc += ((static_cast<float>(v671_data[9])) * v492_data);
              v670_acc += ((static_cast<float>(v671_data[10])) * v494_data);
              v670_acc += ((static_cast<float>(v671_data[11])) * v496_data);
              v670_acc += ((static_cast<float>(v671_data[12])) * v498_data);
              v670_acc += ((static_cast<float>(v671_data[13])) * v500_data);
              v670_acc += ((static_cast<float>(v671_data[14])) * v502_data);
              v670_acc += ((static_cast<float>(v671_data[15])) * v504_data);
              ir1.template select<16, 1>(80) = v670_acc;
              tensorforge::intel_esimd::simd<float, 16> v703_acc{};
              tensorforge::intel_esimd::simd<float, 16> v704_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v703_acc += ((static_cast<float>(v704_data[1])) * v476_data);
              v703_acc += ((static_cast<float>(v704_data[2])) * v478_data);
              v703_acc += ((static_cast<float>(v704_data[3])) * v480_data);
              v703_acc += ((static_cast<float>(v704_data[4])) * v482_data);
              v703_acc += ((static_cast<float>(v704_data[5])) * v484_data);
              v703_acc += ((static_cast<float>(v704_data[6])) * v486_data);
              v703_acc += ((static_cast<float>(v704_data[7])) * v488_data);
              v703_acc += ((static_cast<float>(v704_data[8])) * v490_data);
              v703_acc += ((static_cast<float>(v704_data[9])) * v492_data);
              v703_acc += ((static_cast<float>(v704_data[10])) * v494_data);
              v703_acc += ((static_cast<float>(v704_data[11])) * v496_data);
              v703_acc += ((static_cast<float>(v704_data[12])) * v498_data);
              v703_acc += ((static_cast<float>(v704_data[13])) * v500_data);
              v703_acc += ((static_cast<float>(v704_data[14])) * v502_data);
              v703_acc += ((static_cast<float>(v704_data[15])) * v504_data);
              ir1.template select<16, 1>(96) = v703_acc;
              tensorforge::intel_esimd::simd<float, 16> v736_acc{};
              tensorforge::intel_esimd::simd<float, 16> v737_data = tensorforge::slmLoad<float, 16>(s1 + (112_i32));
              v736_acc += ((static_cast<float>(v737_data[1])) * v476_data);
              v736_acc += ((static_cast<float>(v737_data[2])) * v478_data);
              v736_acc += ((static_cast<float>(v737_data[3])) * v480_data);
              v736_acc += ((static_cast<float>(v737_data[4])) * v482_data);
              v736_acc += ((static_cast<float>(v737_data[5])) * v484_data);
              v736_acc += ((static_cast<float>(v737_data[6])) * v486_data);
              v736_acc += ((static_cast<float>(v737_data[7])) * v488_data);
              v736_acc += ((static_cast<float>(v737_data[8])) * v490_data);
              v736_acc += ((static_cast<float>(v737_data[9])) * v492_data);
              v736_acc += ((static_cast<float>(v737_data[10])) * v494_data);
              v736_acc += ((static_cast<float>(v737_data[11])) * v496_data);
              v736_acc += ((static_cast<float>(v737_data[12])) * v498_data);
              v736_acc += ((static_cast<float>(v737_data[13])) * v500_data);
              v736_acc += ((static_cast<float>(v737_data[14])) * v502_data);
              v736_acc += ((static_cast<float>(v737_data[15])) * v504_data);
              ir1.template select<16, 1>(112) = v736_acc;
              tensorforge::intel_esimd::simd<float, 16> v769_acc{};
              tensorforge::intel_esimd::simd<float, 16> v770_data = tensorforge::slmLoad<float, 16>(s1 + (128_i32));
              v769_acc += ((static_cast<float>(v770_data[1])) * v476_data);
              v769_acc += ((static_cast<float>(v770_data[2])) * v478_data);
              v769_acc += ((static_cast<float>(v770_data[3])) * v480_data);
              v769_acc += ((static_cast<float>(v770_data[4])) * v482_data);
              v769_acc += ((static_cast<float>(v770_data[5])) * v484_data);
              v769_acc += ((static_cast<float>(v770_data[6])) * v486_data);
              v769_acc += ((static_cast<float>(v770_data[7])) * v488_data);
              v769_acc += ((static_cast<float>(v770_data[8])) * v490_data);
              v769_acc += ((static_cast<float>(v770_data[9])) * v492_data);
              v769_acc += ((static_cast<float>(v770_data[10])) * v494_data);
              v769_acc += ((static_cast<float>(v770_data[11])) * v496_data);
              v769_acc += ((static_cast<float>(v770_data[12])) * v498_data);
              v769_acc += ((static_cast<float>(v770_data[13])) * v500_data);
              v769_acc += ((static_cast<float>(v770_data[14])) * v502_data);
              v769_acc += ((static_cast<float>(v770_data[15])) * v504_data);
              ir1.template select<16, 1>(128) = v769_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v802_n0 = 0; v802_n0 < 1; ++v802_n0) {
                int32_t v804_a = v802_n0 * 16;
                #pragma unroll
                for (int32_t v803_n1 = 0; v803_n1 < 9; ++v803_n1) {
                  int32_t v806_a = v804_a + (v803_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v807_data(ir1.template select<16, 1>(v806_a));
                  r1.template select<16, 1>(v806_a) = v807_data;
                }
              }
              // glb_m2 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v808_i0 = 0; v808_i0 < 1; ++v808_i0) {
                int32_t v810_a = v808_i0 * 16;
                #pragma unroll
                for (int32_t v809_i1 = 0; v809_i1 < 9; ++v809_i1) {
                  int32_t v812_a = v810_a + (v809_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v813_data(r1.template select<16, 1>(v812_a));
                  v813_data.copy_to(glb_m2 + (v812_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


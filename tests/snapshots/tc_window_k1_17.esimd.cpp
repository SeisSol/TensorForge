// === base name ===
kernel_f31db27b93ec6a19

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f31db27b93ec6a19 = {{1, 32, 1}, 16, 16, 1, 32, 23616, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f31db27b93ec6a19(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f31db27b93ec6a19(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f31db27b93ec6a19(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 5904 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f31db27b93ec6a19(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f31db27b93ec6a19(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f31db27b93ec6a19(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f31db27b93ec6a19(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5904 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 32 per block = block 1x32x1, 23616 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×17) {0..16}×{1..18} none
        //   m2 20×9(17×9) {1..18}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":5904}],"shared_bytes":23616,"shared_elements":5904,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 272);
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
            tensorforge::intel_esimd::simd<float, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<float, 16> v21_ld;
            v21_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v21_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<float, 16> v22_ld;
            v22_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v22_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<float, 16> v23_ld;
            v23_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v23_ld);
          }
          if (item.get_local_id(1) == 15) {
            tensorforge::intel_esimd::simd<float, 16> v24_ld;
            v24_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v24_ld);
          }
          if (item.get_local_id(1) == 16) {
            tensorforge::intel_esimd::simd<float, 16> v25_ld;
            v25_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v25_ld);
          }
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v27_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v27_batchId0 < numElements0; v27_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v28_ahead1 = v27_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v30_batchId1 = (v28_ahead1 < numElements0) ? v28_ahead1 : v27_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v27_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v27_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v27_batchId0 * 153 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v37_ld;
              v37_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v37_ld);
              tensorforge::intel_esimd::simd<float, 64> v38_ld;
              v38_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v38_ld);
              tensorforge::intel_esimd::simd<float, 16> v39_ld;
              v39_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v39_ld);
              tensorforge::intel_esimd::simd<float, 9> v40_ld;
              v40_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v40_ld);
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v46_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v48_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v50_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v52_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v54_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v56_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v58_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v60_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v64_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v66_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v68_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v72_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v74_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v76_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v78_data = tensorforge::slmLoad<float, 16>(glb_m1 + (256_i32));
              tensorforge::intel_esimd::simd<float, 16> v79_acc{};
              tensorforge::intel_esimd::simd<float, 16> v82_data(0.0f);
              v82_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v79_acc += ((static_cast<float>(v82_data[1])) * v46_data);
              v79_acc += ((static_cast<float>(v82_data[2])) * v48_data);
              v79_acc += ((static_cast<float>(v82_data[3])) * v50_data);
              v79_acc += ((static_cast<float>(v82_data[4])) * v52_data);
              v79_acc += ((static_cast<float>(v82_data[5])) * v54_data);
              v79_acc += ((static_cast<float>(v82_data[6])) * v56_data);
              v79_acc += ((static_cast<float>(v82_data[7])) * v58_data);
              v79_acc += ((static_cast<float>(v82_data[8])) * v60_data);
              v79_acc += ((static_cast<float>(v82_data[9])) * v62_data);
              v79_acc += ((static_cast<float>(v82_data[10])) * v64_data);
              v79_acc += ((static_cast<float>(v82_data[11])) * v66_data);
              v79_acc += ((static_cast<float>(v82_data[12])) * v68_data);
              v79_acc += ((static_cast<float>(v82_data[13])) * v70_data);
              v79_acc += ((static_cast<float>(v82_data[14])) * v72_data);
              v79_acc += ((static_cast<float>(v82_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v118_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v79_acc += ((static_cast<float>(v118_data[0])) * v76_data);
              v79_acc += ((static_cast<float>(v118_data[1])) * v78_data);
              ir0.template select<16, 1>(0) = v79_acc;
              tensorforge::intel_esimd::simd<float, 16> v123_acc{};
              tensorforge::intel_esimd::simd<float, 16> v125_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v123_acc += ((static_cast<float>(v125_data[1])) * v46_data);
              v123_acc += ((static_cast<float>(v125_data[2])) * v48_data);
              v123_acc += ((static_cast<float>(v125_data[3])) * v50_data);
              v123_acc += ((static_cast<float>(v125_data[4])) * v52_data);
              v123_acc += ((static_cast<float>(v125_data[5])) * v54_data);
              v123_acc += ((static_cast<float>(v125_data[6])) * v56_data);
              v123_acc += ((static_cast<float>(v125_data[7])) * v58_data);
              v123_acc += ((static_cast<float>(v125_data[8])) * v60_data);
              v123_acc += ((static_cast<float>(v125_data[9])) * v62_data);
              v123_acc += ((static_cast<float>(v125_data[10])) * v64_data);
              v123_acc += ((static_cast<float>(v125_data[11])) * v66_data);
              v123_acc += ((static_cast<float>(v125_data[12])) * v68_data);
              v123_acc += ((static_cast<float>(v125_data[13])) * v70_data);
              v123_acc += ((static_cast<float>(v125_data[14])) * v72_data);
              v123_acc += ((static_cast<float>(v125_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v158_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v123_acc += ((static_cast<float>(v158_data[0])) * v76_data);
              v123_acc += ((static_cast<float>(v158_data[1])) * v78_data);
              ir0.template select<16, 1>(16) = v123_acc;
              tensorforge::intel_esimd::simd<float, 16> v163_acc{};
              tensorforge::intel_esimd::simd<float, 16> v165_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v163_acc += ((static_cast<float>(v165_data[1])) * v46_data);
              v163_acc += ((static_cast<float>(v165_data[2])) * v48_data);
              v163_acc += ((static_cast<float>(v165_data[3])) * v50_data);
              v163_acc += ((static_cast<float>(v165_data[4])) * v52_data);
              v163_acc += ((static_cast<float>(v165_data[5])) * v54_data);
              v163_acc += ((static_cast<float>(v165_data[6])) * v56_data);
              v163_acc += ((static_cast<float>(v165_data[7])) * v58_data);
              v163_acc += ((static_cast<float>(v165_data[8])) * v60_data);
              v163_acc += ((static_cast<float>(v165_data[9])) * v62_data);
              v163_acc += ((static_cast<float>(v165_data[10])) * v64_data);
              v163_acc += ((static_cast<float>(v165_data[11])) * v66_data);
              v163_acc += ((static_cast<float>(v165_data[12])) * v68_data);
              v163_acc += ((static_cast<float>(v165_data[13])) * v70_data);
              v163_acc += ((static_cast<float>(v165_data[14])) * v72_data);
              v163_acc += ((static_cast<float>(v165_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v198_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v163_acc += ((static_cast<float>(v198_data[0])) * v76_data);
              v163_acc += ((static_cast<float>(v198_data[1])) * v78_data);
              ir0.template select<16, 1>(32) = v163_acc;
              tensorforge::intel_esimd::simd<float, 16> v203_acc{};
              tensorforge::intel_esimd::simd<float, 16> v205_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v203_acc += ((static_cast<float>(v205_data[1])) * v46_data);
              v203_acc += ((static_cast<float>(v205_data[2])) * v48_data);
              v203_acc += ((static_cast<float>(v205_data[3])) * v50_data);
              v203_acc += ((static_cast<float>(v205_data[4])) * v52_data);
              v203_acc += ((static_cast<float>(v205_data[5])) * v54_data);
              v203_acc += ((static_cast<float>(v205_data[6])) * v56_data);
              v203_acc += ((static_cast<float>(v205_data[7])) * v58_data);
              v203_acc += ((static_cast<float>(v205_data[8])) * v60_data);
              v203_acc += ((static_cast<float>(v205_data[9])) * v62_data);
              v203_acc += ((static_cast<float>(v205_data[10])) * v64_data);
              v203_acc += ((static_cast<float>(v205_data[11])) * v66_data);
              v203_acc += ((static_cast<float>(v205_data[12])) * v68_data);
              v203_acc += ((static_cast<float>(v205_data[13])) * v70_data);
              v203_acc += ((static_cast<float>(v205_data[14])) * v72_data);
              v203_acc += ((static_cast<float>(v205_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v238_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v203_acc += ((static_cast<float>(v238_data[0])) * v76_data);
              v203_acc += ((static_cast<float>(v238_data[1])) * v78_data);
              ir0.template select<16, 1>(48) = v203_acc;
              tensorforge::intel_esimd::simd<float, 16> v243_acc{};
              tensorforge::intel_esimd::simd<float, 16> v245_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v243_acc += ((static_cast<float>(v245_data[1])) * v46_data);
              v243_acc += ((static_cast<float>(v245_data[2])) * v48_data);
              v243_acc += ((static_cast<float>(v245_data[3])) * v50_data);
              v243_acc += ((static_cast<float>(v245_data[4])) * v52_data);
              v243_acc += ((static_cast<float>(v245_data[5])) * v54_data);
              v243_acc += ((static_cast<float>(v245_data[6])) * v56_data);
              v243_acc += ((static_cast<float>(v245_data[7])) * v58_data);
              v243_acc += ((static_cast<float>(v245_data[8])) * v60_data);
              v243_acc += ((static_cast<float>(v245_data[9])) * v62_data);
              v243_acc += ((static_cast<float>(v245_data[10])) * v64_data);
              v243_acc += ((static_cast<float>(v245_data[11])) * v66_data);
              v243_acc += ((static_cast<float>(v245_data[12])) * v68_data);
              v243_acc += ((static_cast<float>(v245_data[13])) * v70_data);
              v243_acc += ((static_cast<float>(v245_data[14])) * v72_data);
              v243_acc += ((static_cast<float>(v245_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v278_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v243_acc += ((static_cast<float>(v278_data[0])) * v76_data);
              v243_acc += ((static_cast<float>(v278_data[1])) * v78_data);
              ir0.template select<16, 1>(64) = v243_acc;
              tensorforge::intel_esimd::simd<float, 16> v283_acc{};
              tensorforge::intel_esimd::simd<float, 16> v285_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v283_acc += ((static_cast<float>(v285_data[1])) * v46_data);
              v283_acc += ((static_cast<float>(v285_data[2])) * v48_data);
              v283_acc += ((static_cast<float>(v285_data[3])) * v50_data);
              v283_acc += ((static_cast<float>(v285_data[4])) * v52_data);
              v283_acc += ((static_cast<float>(v285_data[5])) * v54_data);
              v283_acc += ((static_cast<float>(v285_data[6])) * v56_data);
              v283_acc += ((static_cast<float>(v285_data[7])) * v58_data);
              v283_acc += ((static_cast<float>(v285_data[8])) * v60_data);
              v283_acc += ((static_cast<float>(v285_data[9])) * v62_data);
              v283_acc += ((static_cast<float>(v285_data[10])) * v64_data);
              v283_acc += ((static_cast<float>(v285_data[11])) * v66_data);
              v283_acc += ((static_cast<float>(v285_data[12])) * v68_data);
              v283_acc += ((static_cast<float>(v285_data[13])) * v70_data);
              v283_acc += ((static_cast<float>(v285_data[14])) * v72_data);
              v283_acc += ((static_cast<float>(v285_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v318_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v283_acc += ((static_cast<float>(v318_data[0])) * v76_data);
              v283_acc += ((static_cast<float>(v318_data[1])) * v78_data);
              ir0.template select<16, 1>(80) = v283_acc;
              tensorforge::intel_esimd::simd<float, 16> v323_acc{};
              tensorforge::intel_esimd::simd<float, 16> v325_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v323_acc += ((static_cast<float>(v325_data[1])) * v46_data);
              v323_acc += ((static_cast<float>(v325_data[2])) * v48_data);
              v323_acc += ((static_cast<float>(v325_data[3])) * v50_data);
              v323_acc += ((static_cast<float>(v325_data[4])) * v52_data);
              v323_acc += ((static_cast<float>(v325_data[5])) * v54_data);
              v323_acc += ((static_cast<float>(v325_data[6])) * v56_data);
              v323_acc += ((static_cast<float>(v325_data[7])) * v58_data);
              v323_acc += ((static_cast<float>(v325_data[8])) * v60_data);
              v323_acc += ((static_cast<float>(v325_data[9])) * v62_data);
              v323_acc += ((static_cast<float>(v325_data[10])) * v64_data);
              v323_acc += ((static_cast<float>(v325_data[11])) * v66_data);
              v323_acc += ((static_cast<float>(v325_data[12])) * v68_data);
              v323_acc += ((static_cast<float>(v325_data[13])) * v70_data);
              v323_acc += ((static_cast<float>(v325_data[14])) * v72_data);
              v323_acc += ((static_cast<float>(v325_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v358_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v323_acc += ((static_cast<float>(v358_data[0])) * v76_data);
              v323_acc += ((static_cast<float>(v358_data[1])) * v78_data);
              ir0.template select<16, 1>(96) = v323_acc;
              tensorforge::intel_esimd::simd<float, 16> v363_acc{};
              tensorforge::intel_esimd::simd<float, 16> v365_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v363_acc += ((static_cast<float>(v365_data[1])) * v46_data);
              v363_acc += ((static_cast<float>(v365_data[2])) * v48_data);
              v363_acc += ((static_cast<float>(v365_data[3])) * v50_data);
              v363_acc += ((static_cast<float>(v365_data[4])) * v52_data);
              v363_acc += ((static_cast<float>(v365_data[5])) * v54_data);
              v363_acc += ((static_cast<float>(v365_data[6])) * v56_data);
              v363_acc += ((static_cast<float>(v365_data[7])) * v58_data);
              v363_acc += ((static_cast<float>(v365_data[8])) * v60_data);
              v363_acc += ((static_cast<float>(v365_data[9])) * v62_data);
              v363_acc += ((static_cast<float>(v365_data[10])) * v64_data);
              v363_acc += ((static_cast<float>(v365_data[11])) * v66_data);
              v363_acc += ((static_cast<float>(v365_data[12])) * v68_data);
              v363_acc += ((static_cast<float>(v365_data[13])) * v70_data);
              v363_acc += ((static_cast<float>(v365_data[14])) * v72_data);
              v363_acc += ((static_cast<float>(v365_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v398_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v363_acc += ((static_cast<float>(v398_data[0])) * v76_data);
              v363_acc += ((static_cast<float>(v398_data[1])) * v78_data);
              ir0.template select<16, 1>(112) = v363_acc;
              tensorforge::intel_esimd::simd<float, 16> v403_acc{};
              tensorforge::intel_esimd::simd<float, 16> v405_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v403_acc += ((static_cast<float>(v405_data[1])) * v46_data);
              v403_acc += ((static_cast<float>(v405_data[2])) * v48_data);
              v403_acc += ((static_cast<float>(v405_data[3])) * v50_data);
              v403_acc += ((static_cast<float>(v405_data[4])) * v52_data);
              v403_acc += ((static_cast<float>(v405_data[5])) * v54_data);
              v403_acc += ((static_cast<float>(v405_data[6])) * v56_data);
              v403_acc += ((static_cast<float>(v405_data[7])) * v58_data);
              v403_acc += ((static_cast<float>(v405_data[8])) * v60_data);
              v403_acc += ((static_cast<float>(v405_data[9])) * v62_data);
              v403_acc += ((static_cast<float>(v405_data[10])) * v64_data);
              v403_acc += ((static_cast<float>(v405_data[11])) * v66_data);
              v403_acc += ((static_cast<float>(v405_data[12])) * v68_data);
              v403_acc += ((static_cast<float>(v405_data[13])) * v70_data);
              v403_acc += ((static_cast<float>(v405_data[14])) * v72_data);
              v403_acc += ((static_cast<float>(v405_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v438_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v403_acc += ((static_cast<float>(v438_data[0])) * v76_data);
              v403_acc += ((static_cast<float>(v438_data[1])) * v78_data);
              ir0.template select<16, 1>(128) = v403_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v443_n0 = 0; v443_n0 < 1; ++v443_n0) {
                int32_t v445_a = v443_n0 * 16;
                #pragma unroll
                for (int32_t v444_n1 = 0; v444_n1 < 9; ++v444_n1) {
                  int32_t v447_a = v445_a + (v444_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v448_data(ir0.template select<16, 1>(v447_a));
                  r0.template select<16, 1>(v447_a) = v448_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v449_i0 = 0; v449_i0 < 1; ++v449_i0) {
                int32_t v451_a = v449_i0 * 16;
                #pragma unroll
                for (int32_t v450_i1 = 0; v450_i1 < 9; ++v450_i1) {
                  int32_t v453_a = v451_a + (v450_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v454_data(r0.template select<16, 1>(v453_a));
                  v454_data.copy_to(glb_m0 + (v453_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


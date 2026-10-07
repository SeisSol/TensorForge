// === base name ===
kernel_7ffeeb57b15135ef

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7ffeeb57b15135ef = {{1, 32, 1}, 16, 10, 1, 32, 23232, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7ffeeb57b15135ef(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7ffeeb57b15135ef(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7ffeeb57b15135ef(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 5808 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7ffeeb57b15135ef(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7ffeeb57b15135ef(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7ffeeb57b15135ef(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7ffeeb57b15135ef(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5808 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (10 active) x 32 per block = block 1x32x1, 23232 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 16×20(10×17) {0..10}×{1..18} none
        //   m2 20×9(17×9) {1..18}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":5808}],"shared_bytes":23232,"shared_elements":5808,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 176);
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
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v21_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v21_batchId0 < numElements0; v21_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v22_ahead1 = v21_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v24_batchId1 = (v22_ahead1 < numElements0) ? v22_ahead1 : v21_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v21_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v21_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v21_batchId0 * 153 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v32_ld);
              tensorforge::intel_esimd::simd<float, 16> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v33_ld);
              tensorforge::intel_esimd::simd<float, 9> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v34_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v40_data = tensorforge::slmLoad<float, 16>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v42_data = tensorforge::slmLoad<float, 16>(glb_m1 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v44_data = tensorforge::slmLoad<float, 16>(glb_m1 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v46_data = tensorforge::slmLoad<float, 16>(glb_m1 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v48_data = tensorforge::slmLoad<float, 16>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data = tensorforge::slmLoad<float, 16>(glb_m1 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v52_data = tensorforge::slmLoad<float, 16>(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v54_data = tensorforge::slmLoad<float, 16>(glb_m1 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data = tensorforge::slmLoad<float, 16>(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(glb_m1 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data = tensorforge::slmLoad<float, 16>(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data = tensorforge::slmLoad<float, 16>(glb_m1 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data = tensorforge::slmLoad<float, 16>(glb_m1 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_data = tensorforge::slmLoad<float, 16>(glb_m1 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data = tensorforge::slmLoad<float, 16>(glb_m1 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_data = tensorforge::slmLoad<float, 16>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v73_acc{};
              tensorforge::intel_esimd::simd<float, 16> v76_data(0.0f);
              v76_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v73_acc += ((static_cast<float>(v76_data[1])) * v40_data);
              v73_acc += ((static_cast<float>(v76_data[2])) * v42_data);
              v73_acc += ((static_cast<float>(v76_data[3])) * v44_data);
              v73_acc += ((static_cast<float>(v76_data[4])) * v46_data);
              v73_acc += ((static_cast<float>(v76_data[5])) * v48_data);
              v73_acc += ((static_cast<float>(v76_data[6])) * v50_data);
              v73_acc += ((static_cast<float>(v76_data[7])) * v52_data);
              v73_acc += ((static_cast<float>(v76_data[8])) * v54_data);
              v73_acc += ((static_cast<float>(v76_data[9])) * v56_data);
              v73_acc += ((static_cast<float>(v76_data[10])) * v58_data);
              v73_acc += ((static_cast<float>(v76_data[11])) * v60_data);
              v73_acc += ((static_cast<float>(v76_data[12])) * v62_data);
              v73_acc += ((static_cast<float>(v76_data[13])) * v64_data);
              v73_acc += ((static_cast<float>(v76_data[14])) * v66_data);
              v73_acc += ((static_cast<float>(v76_data[15])) * v68_data);
              tensorforge::intel_esimd::simd<float, 16> v112_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v73_acc += ((static_cast<float>(v112_data[0])) * v70_data);
              v73_acc += ((static_cast<float>(v112_data[1])) * v72_data);
              ir0.template select<16, 1>(0) = v73_acc;
              tensorforge::intel_esimd::simd<float, 16> v117_acc{};
              tensorforge::intel_esimd::simd<float, 16> v119_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v117_acc += ((static_cast<float>(v119_data[1])) * v40_data);
              v117_acc += ((static_cast<float>(v119_data[2])) * v42_data);
              v117_acc += ((static_cast<float>(v119_data[3])) * v44_data);
              v117_acc += ((static_cast<float>(v119_data[4])) * v46_data);
              v117_acc += ((static_cast<float>(v119_data[5])) * v48_data);
              v117_acc += ((static_cast<float>(v119_data[6])) * v50_data);
              v117_acc += ((static_cast<float>(v119_data[7])) * v52_data);
              v117_acc += ((static_cast<float>(v119_data[8])) * v54_data);
              v117_acc += ((static_cast<float>(v119_data[9])) * v56_data);
              v117_acc += ((static_cast<float>(v119_data[10])) * v58_data);
              v117_acc += ((static_cast<float>(v119_data[11])) * v60_data);
              v117_acc += ((static_cast<float>(v119_data[12])) * v62_data);
              v117_acc += ((static_cast<float>(v119_data[13])) * v64_data);
              v117_acc += ((static_cast<float>(v119_data[14])) * v66_data);
              v117_acc += ((static_cast<float>(v119_data[15])) * v68_data);
              tensorforge::intel_esimd::simd<float, 16> v152_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v117_acc += ((static_cast<float>(v152_data[0])) * v70_data);
              v117_acc += ((static_cast<float>(v152_data[1])) * v72_data);
              ir0.template select<16, 1>(16) = v117_acc;
              tensorforge::intel_esimd::simd<float, 16> v157_acc{};
              tensorforge::intel_esimd::simd<float, 16> v159_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v157_acc += ((static_cast<float>(v159_data[1])) * v40_data);
              v157_acc += ((static_cast<float>(v159_data[2])) * v42_data);
              v157_acc += ((static_cast<float>(v159_data[3])) * v44_data);
              v157_acc += ((static_cast<float>(v159_data[4])) * v46_data);
              v157_acc += ((static_cast<float>(v159_data[5])) * v48_data);
              v157_acc += ((static_cast<float>(v159_data[6])) * v50_data);
              v157_acc += ((static_cast<float>(v159_data[7])) * v52_data);
              v157_acc += ((static_cast<float>(v159_data[8])) * v54_data);
              v157_acc += ((static_cast<float>(v159_data[9])) * v56_data);
              v157_acc += ((static_cast<float>(v159_data[10])) * v58_data);
              v157_acc += ((static_cast<float>(v159_data[11])) * v60_data);
              v157_acc += ((static_cast<float>(v159_data[12])) * v62_data);
              v157_acc += ((static_cast<float>(v159_data[13])) * v64_data);
              v157_acc += ((static_cast<float>(v159_data[14])) * v66_data);
              v157_acc += ((static_cast<float>(v159_data[15])) * v68_data);
              tensorforge::intel_esimd::simd<float, 16> v192_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v157_acc += ((static_cast<float>(v192_data[0])) * v70_data);
              v157_acc += ((static_cast<float>(v192_data[1])) * v72_data);
              ir0.template select<16, 1>(32) = v157_acc;
              tensorforge::intel_esimd::simd<float, 16> v197_acc{};
              tensorforge::intel_esimd::simd<float, 16> v199_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v197_acc += ((static_cast<float>(v199_data[1])) * v40_data);
              v197_acc += ((static_cast<float>(v199_data[2])) * v42_data);
              v197_acc += ((static_cast<float>(v199_data[3])) * v44_data);
              v197_acc += ((static_cast<float>(v199_data[4])) * v46_data);
              v197_acc += ((static_cast<float>(v199_data[5])) * v48_data);
              v197_acc += ((static_cast<float>(v199_data[6])) * v50_data);
              v197_acc += ((static_cast<float>(v199_data[7])) * v52_data);
              v197_acc += ((static_cast<float>(v199_data[8])) * v54_data);
              v197_acc += ((static_cast<float>(v199_data[9])) * v56_data);
              v197_acc += ((static_cast<float>(v199_data[10])) * v58_data);
              v197_acc += ((static_cast<float>(v199_data[11])) * v60_data);
              v197_acc += ((static_cast<float>(v199_data[12])) * v62_data);
              v197_acc += ((static_cast<float>(v199_data[13])) * v64_data);
              v197_acc += ((static_cast<float>(v199_data[14])) * v66_data);
              v197_acc += ((static_cast<float>(v199_data[15])) * v68_data);
              tensorforge::intel_esimd::simd<float, 16> v232_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v197_acc += ((static_cast<float>(v232_data[0])) * v70_data);
              v197_acc += ((static_cast<float>(v232_data[1])) * v72_data);
              ir0.template select<16, 1>(48) = v197_acc;
              tensorforge::intel_esimd::simd<float, 16> v237_acc{};
              tensorforge::intel_esimd::simd<float, 16> v239_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v237_acc += ((static_cast<float>(v239_data[1])) * v40_data);
              v237_acc += ((static_cast<float>(v239_data[2])) * v42_data);
              v237_acc += ((static_cast<float>(v239_data[3])) * v44_data);
              v237_acc += ((static_cast<float>(v239_data[4])) * v46_data);
              v237_acc += ((static_cast<float>(v239_data[5])) * v48_data);
              v237_acc += ((static_cast<float>(v239_data[6])) * v50_data);
              v237_acc += ((static_cast<float>(v239_data[7])) * v52_data);
              v237_acc += ((static_cast<float>(v239_data[8])) * v54_data);
              v237_acc += ((static_cast<float>(v239_data[9])) * v56_data);
              v237_acc += ((static_cast<float>(v239_data[10])) * v58_data);
              v237_acc += ((static_cast<float>(v239_data[11])) * v60_data);
              v237_acc += ((static_cast<float>(v239_data[12])) * v62_data);
              v237_acc += ((static_cast<float>(v239_data[13])) * v64_data);
              v237_acc += ((static_cast<float>(v239_data[14])) * v66_data);
              v237_acc += ((static_cast<float>(v239_data[15])) * v68_data);
              tensorforge::intel_esimd::simd<float, 16> v272_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v237_acc += ((static_cast<float>(v272_data[0])) * v70_data);
              v237_acc += ((static_cast<float>(v272_data[1])) * v72_data);
              ir0.template select<16, 1>(64) = v237_acc;
              tensorforge::intel_esimd::simd<float, 16> v277_acc{};
              tensorforge::intel_esimd::simd<float, 16> v279_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v277_acc += ((static_cast<float>(v279_data[1])) * v40_data);
              v277_acc += ((static_cast<float>(v279_data[2])) * v42_data);
              v277_acc += ((static_cast<float>(v279_data[3])) * v44_data);
              v277_acc += ((static_cast<float>(v279_data[4])) * v46_data);
              v277_acc += ((static_cast<float>(v279_data[5])) * v48_data);
              v277_acc += ((static_cast<float>(v279_data[6])) * v50_data);
              v277_acc += ((static_cast<float>(v279_data[7])) * v52_data);
              v277_acc += ((static_cast<float>(v279_data[8])) * v54_data);
              v277_acc += ((static_cast<float>(v279_data[9])) * v56_data);
              v277_acc += ((static_cast<float>(v279_data[10])) * v58_data);
              v277_acc += ((static_cast<float>(v279_data[11])) * v60_data);
              v277_acc += ((static_cast<float>(v279_data[12])) * v62_data);
              v277_acc += ((static_cast<float>(v279_data[13])) * v64_data);
              v277_acc += ((static_cast<float>(v279_data[14])) * v66_data);
              v277_acc += ((static_cast<float>(v279_data[15])) * v68_data);
              tensorforge::intel_esimd::simd<float, 16> v312_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v277_acc += ((static_cast<float>(v312_data[0])) * v70_data);
              v277_acc += ((static_cast<float>(v312_data[1])) * v72_data);
              ir0.template select<16, 1>(80) = v277_acc;
              tensorforge::intel_esimd::simd<float, 16> v317_acc{};
              tensorforge::intel_esimd::simd<float, 16> v319_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v317_acc += ((static_cast<float>(v319_data[1])) * v40_data);
              v317_acc += ((static_cast<float>(v319_data[2])) * v42_data);
              v317_acc += ((static_cast<float>(v319_data[3])) * v44_data);
              v317_acc += ((static_cast<float>(v319_data[4])) * v46_data);
              v317_acc += ((static_cast<float>(v319_data[5])) * v48_data);
              v317_acc += ((static_cast<float>(v319_data[6])) * v50_data);
              v317_acc += ((static_cast<float>(v319_data[7])) * v52_data);
              v317_acc += ((static_cast<float>(v319_data[8])) * v54_data);
              v317_acc += ((static_cast<float>(v319_data[9])) * v56_data);
              v317_acc += ((static_cast<float>(v319_data[10])) * v58_data);
              v317_acc += ((static_cast<float>(v319_data[11])) * v60_data);
              v317_acc += ((static_cast<float>(v319_data[12])) * v62_data);
              v317_acc += ((static_cast<float>(v319_data[13])) * v64_data);
              v317_acc += ((static_cast<float>(v319_data[14])) * v66_data);
              v317_acc += ((static_cast<float>(v319_data[15])) * v68_data);
              tensorforge::intel_esimd::simd<float, 16> v352_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v317_acc += ((static_cast<float>(v352_data[0])) * v70_data);
              v317_acc += ((static_cast<float>(v352_data[1])) * v72_data);
              ir0.template select<16, 1>(96) = v317_acc;
              tensorforge::intel_esimd::simd<float, 16> v357_acc{};
              tensorforge::intel_esimd::simd<float, 16> v359_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v357_acc += ((static_cast<float>(v359_data[1])) * v40_data);
              v357_acc += ((static_cast<float>(v359_data[2])) * v42_data);
              v357_acc += ((static_cast<float>(v359_data[3])) * v44_data);
              v357_acc += ((static_cast<float>(v359_data[4])) * v46_data);
              v357_acc += ((static_cast<float>(v359_data[5])) * v48_data);
              v357_acc += ((static_cast<float>(v359_data[6])) * v50_data);
              v357_acc += ((static_cast<float>(v359_data[7])) * v52_data);
              v357_acc += ((static_cast<float>(v359_data[8])) * v54_data);
              v357_acc += ((static_cast<float>(v359_data[9])) * v56_data);
              v357_acc += ((static_cast<float>(v359_data[10])) * v58_data);
              v357_acc += ((static_cast<float>(v359_data[11])) * v60_data);
              v357_acc += ((static_cast<float>(v359_data[12])) * v62_data);
              v357_acc += ((static_cast<float>(v359_data[13])) * v64_data);
              v357_acc += ((static_cast<float>(v359_data[14])) * v66_data);
              v357_acc += ((static_cast<float>(v359_data[15])) * v68_data);
              tensorforge::intel_esimd::simd<float, 16> v392_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v357_acc += ((static_cast<float>(v392_data[0])) * v70_data);
              v357_acc += ((static_cast<float>(v392_data[1])) * v72_data);
              ir0.template select<16, 1>(112) = v357_acc;
              tensorforge::intel_esimd::simd<float, 16> v397_acc{};
              tensorforge::intel_esimd::simd<float, 16> v399_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v397_acc += ((static_cast<float>(v399_data[1])) * v40_data);
              v397_acc += ((static_cast<float>(v399_data[2])) * v42_data);
              v397_acc += ((static_cast<float>(v399_data[3])) * v44_data);
              v397_acc += ((static_cast<float>(v399_data[4])) * v46_data);
              v397_acc += ((static_cast<float>(v399_data[5])) * v48_data);
              v397_acc += ((static_cast<float>(v399_data[6])) * v50_data);
              v397_acc += ((static_cast<float>(v399_data[7])) * v52_data);
              v397_acc += ((static_cast<float>(v399_data[8])) * v54_data);
              v397_acc += ((static_cast<float>(v399_data[9])) * v56_data);
              v397_acc += ((static_cast<float>(v399_data[10])) * v58_data);
              v397_acc += ((static_cast<float>(v399_data[11])) * v60_data);
              v397_acc += ((static_cast<float>(v399_data[12])) * v62_data);
              v397_acc += ((static_cast<float>(v399_data[13])) * v64_data);
              v397_acc += ((static_cast<float>(v399_data[14])) * v66_data);
              v397_acc += ((static_cast<float>(v399_data[15])) * v68_data);
              tensorforge::intel_esimd::simd<float, 16> v432_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v397_acc += ((static_cast<float>(v432_data[0])) * v70_data);
              v397_acc += ((static_cast<float>(v432_data[1])) * v72_data);
              ir0.template select<16, 1>(128) = v397_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v437_n1 = 0; v437_n1 < 9; ++v437_n1) {
                int32_t v438_a = v437_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v440_data(ir0.template select<10, 1>(v438_a));
                r0.template select<10, 1>(v438_a) = v440_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v441_i1 = 0; v441_i1 < 9; ++v441_i1) {
                tensorforge::intel_esimd::simd<float, 10> v444_data(r0.template select<10, 1>((v441_i1 * 16)));
                v444_data.copy_to(glb_m0 + ((v441_i1 * 10)));
              }
            }
          }
        }
      }
    });
  });
}


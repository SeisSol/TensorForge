// === base name ===
kernel_64720e64ff5b9da9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_64720e64ff5b9da9 = {{1, 32, 1}, 16, 16, 1, 32, 21504, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_64720e64ff5b9da9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_64720e64ff5b9da9(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_64720e64ff5b9da9(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 5376 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_64720e64ff5b9da9(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_64720e64ff5b9da9(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_64720e64ff5b9da9(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_64720e64ff5b9da9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5376 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 32 per block = block 1x32x1, 21504 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×16) {0..16}×{1..17} none
        //   m2 20×9(16×9) {1..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":5376}],"shared_bytes":21504,"shared_elements":5376,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 256);
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
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v26_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v26_batchId0 < numElements0; v26_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v27_ahead1 = v26_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v29_batchId1 = (v27_ahead1 < numElements0) ? v27_ahead1 : v26_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v26_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v26_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v26_batchId0 * 144 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v36_ld;
              v36_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v36_ld);
              tensorforge::intel_esimd::simd<float, 64> v37_ld;
              v37_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v37_ld);
              tensorforge::intel_esimd::simd<float, 16> v38_ld;
              v38_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v38_ld);
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 9)] [(1, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v44_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v46_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v48_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v50_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v52_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v54_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v56_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v58_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v62_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v64_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v66_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v70_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v72_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v74_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v75_acc{};
              tensorforge::intel_esimd::simd<float, 16> v78_data(0.0f);
              v78_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v75_acc += ((static_cast<float>(v78_data[1])) * v44_data);
              v75_acc += ((static_cast<float>(v78_data[2])) * v46_data);
              v75_acc += ((static_cast<float>(v78_data[3])) * v48_data);
              v75_acc += ((static_cast<float>(v78_data[4])) * v50_data);
              v75_acc += ((static_cast<float>(v78_data[5])) * v52_data);
              v75_acc += ((static_cast<float>(v78_data[6])) * v54_data);
              v75_acc += ((static_cast<float>(v78_data[7])) * v56_data);
              v75_acc += ((static_cast<float>(v78_data[8])) * v58_data);
              v75_acc += ((static_cast<float>(v78_data[9])) * v60_data);
              v75_acc += ((static_cast<float>(v78_data[10])) * v62_data);
              v75_acc += ((static_cast<float>(v78_data[11])) * v64_data);
              v75_acc += ((static_cast<float>(v78_data[12])) * v66_data);
              v75_acc += ((static_cast<float>(v78_data[13])) * v68_data);
              v75_acc += ((static_cast<float>(v78_data[14])) * v70_data);
              v75_acc += ((static_cast<float>(v78_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v114_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v75_acc += ((static_cast<float>(v114_data[0])) * v74_data);
              ir0.template select<16, 1>(0) = v75_acc;
              tensorforge::intel_esimd::simd<float, 16> v117_acc{};
              v117_acc += ((static_cast<float>(v114_data[1])) * v44_data);
              v117_acc += ((static_cast<float>(v114_data[2])) * v46_data);
              v117_acc += ((static_cast<float>(v114_data[3])) * v48_data);
              v117_acc += ((static_cast<float>(v114_data[4])) * v50_data);
              v117_acc += ((static_cast<float>(v114_data[5])) * v52_data);
              v117_acc += ((static_cast<float>(v114_data[6])) * v54_data);
              v117_acc += ((static_cast<float>(v114_data[7])) * v56_data);
              v117_acc += ((static_cast<float>(v114_data[8])) * v58_data);
              v117_acc += ((static_cast<float>(v114_data[9])) * v60_data);
              v117_acc += ((static_cast<float>(v114_data[10])) * v62_data);
              v117_acc += ((static_cast<float>(v114_data[11])) * v64_data);
              v117_acc += ((static_cast<float>(v114_data[12])) * v66_data);
              v117_acc += ((static_cast<float>(v114_data[13])) * v68_data);
              v117_acc += ((static_cast<float>(v114_data[14])) * v70_data);
              v117_acc += ((static_cast<float>(v114_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v152_data = tensorforge::slmLoad<float, 16>(s0 + (31_i32));
              v117_acc += ((static_cast<float>(v152_data[0])) * v74_data);
              ir0.template select<16, 1>(16) = v117_acc;
              tensorforge::intel_esimd::simd<float, 16> v155_acc{};
              v155_acc += ((static_cast<float>(v152_data[1])) * v44_data);
              v155_acc += ((static_cast<float>(v152_data[2])) * v46_data);
              v155_acc += ((static_cast<float>(v152_data[3])) * v48_data);
              v155_acc += ((static_cast<float>(v152_data[4])) * v50_data);
              v155_acc += ((static_cast<float>(v152_data[5])) * v52_data);
              v155_acc += ((static_cast<float>(v152_data[6])) * v54_data);
              v155_acc += ((static_cast<float>(v152_data[7])) * v56_data);
              v155_acc += ((static_cast<float>(v152_data[8])) * v58_data);
              v155_acc += ((static_cast<float>(v152_data[9])) * v60_data);
              v155_acc += ((static_cast<float>(v152_data[10])) * v62_data);
              v155_acc += ((static_cast<float>(v152_data[11])) * v64_data);
              v155_acc += ((static_cast<float>(v152_data[12])) * v66_data);
              v155_acc += ((static_cast<float>(v152_data[13])) * v68_data);
              v155_acc += ((static_cast<float>(v152_data[14])) * v70_data);
              v155_acc += ((static_cast<float>(v152_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v190_data = tensorforge::slmLoad<float, 16>(s0 + (47_i32));
              v155_acc += ((static_cast<float>(v190_data[0])) * v74_data);
              ir0.template select<16, 1>(32) = v155_acc;
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              v193_acc += ((static_cast<float>(v190_data[1])) * v44_data);
              v193_acc += ((static_cast<float>(v190_data[2])) * v46_data);
              v193_acc += ((static_cast<float>(v190_data[3])) * v48_data);
              v193_acc += ((static_cast<float>(v190_data[4])) * v50_data);
              v193_acc += ((static_cast<float>(v190_data[5])) * v52_data);
              v193_acc += ((static_cast<float>(v190_data[6])) * v54_data);
              v193_acc += ((static_cast<float>(v190_data[7])) * v56_data);
              v193_acc += ((static_cast<float>(v190_data[8])) * v58_data);
              v193_acc += ((static_cast<float>(v190_data[9])) * v60_data);
              v193_acc += ((static_cast<float>(v190_data[10])) * v62_data);
              v193_acc += ((static_cast<float>(v190_data[11])) * v64_data);
              v193_acc += ((static_cast<float>(v190_data[12])) * v66_data);
              v193_acc += ((static_cast<float>(v190_data[13])) * v68_data);
              v193_acc += ((static_cast<float>(v190_data[14])) * v70_data);
              v193_acc += ((static_cast<float>(v190_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v228_data = tensorforge::slmLoad<float, 16>(s0 + (63_i32));
              v193_acc += ((static_cast<float>(v228_data[0])) * v74_data);
              ir0.template select<16, 1>(48) = v193_acc;
              tensorforge::intel_esimd::simd<float, 16> v231_acc{};
              v231_acc += ((static_cast<float>(v228_data[1])) * v44_data);
              v231_acc += ((static_cast<float>(v228_data[2])) * v46_data);
              v231_acc += ((static_cast<float>(v228_data[3])) * v48_data);
              v231_acc += ((static_cast<float>(v228_data[4])) * v50_data);
              v231_acc += ((static_cast<float>(v228_data[5])) * v52_data);
              v231_acc += ((static_cast<float>(v228_data[6])) * v54_data);
              v231_acc += ((static_cast<float>(v228_data[7])) * v56_data);
              v231_acc += ((static_cast<float>(v228_data[8])) * v58_data);
              v231_acc += ((static_cast<float>(v228_data[9])) * v60_data);
              v231_acc += ((static_cast<float>(v228_data[10])) * v62_data);
              v231_acc += ((static_cast<float>(v228_data[11])) * v64_data);
              v231_acc += ((static_cast<float>(v228_data[12])) * v66_data);
              v231_acc += ((static_cast<float>(v228_data[13])) * v68_data);
              v231_acc += ((static_cast<float>(v228_data[14])) * v70_data);
              v231_acc += ((static_cast<float>(v228_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v266_data = tensorforge::slmLoad<float, 16>(s0 + (79_i32));
              v231_acc += ((static_cast<float>(v266_data[0])) * v74_data);
              ir0.template select<16, 1>(64) = v231_acc;
              tensorforge::intel_esimd::simd<float, 16> v269_acc{};
              v269_acc += ((static_cast<float>(v266_data[1])) * v44_data);
              v269_acc += ((static_cast<float>(v266_data[2])) * v46_data);
              v269_acc += ((static_cast<float>(v266_data[3])) * v48_data);
              v269_acc += ((static_cast<float>(v266_data[4])) * v50_data);
              v269_acc += ((static_cast<float>(v266_data[5])) * v52_data);
              v269_acc += ((static_cast<float>(v266_data[6])) * v54_data);
              v269_acc += ((static_cast<float>(v266_data[7])) * v56_data);
              v269_acc += ((static_cast<float>(v266_data[8])) * v58_data);
              v269_acc += ((static_cast<float>(v266_data[9])) * v60_data);
              v269_acc += ((static_cast<float>(v266_data[10])) * v62_data);
              v269_acc += ((static_cast<float>(v266_data[11])) * v64_data);
              v269_acc += ((static_cast<float>(v266_data[12])) * v66_data);
              v269_acc += ((static_cast<float>(v266_data[13])) * v68_data);
              v269_acc += ((static_cast<float>(v266_data[14])) * v70_data);
              v269_acc += ((static_cast<float>(v266_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v304_data = tensorforge::slmLoad<float, 16>(s0 + (95_i32));
              v269_acc += ((static_cast<float>(v304_data[0])) * v74_data);
              ir0.template select<16, 1>(80) = v269_acc;
              tensorforge::intel_esimd::simd<float, 16> v307_acc{};
              v307_acc += ((static_cast<float>(v304_data[1])) * v44_data);
              v307_acc += ((static_cast<float>(v304_data[2])) * v46_data);
              v307_acc += ((static_cast<float>(v304_data[3])) * v48_data);
              v307_acc += ((static_cast<float>(v304_data[4])) * v50_data);
              v307_acc += ((static_cast<float>(v304_data[5])) * v52_data);
              v307_acc += ((static_cast<float>(v304_data[6])) * v54_data);
              v307_acc += ((static_cast<float>(v304_data[7])) * v56_data);
              v307_acc += ((static_cast<float>(v304_data[8])) * v58_data);
              v307_acc += ((static_cast<float>(v304_data[9])) * v60_data);
              v307_acc += ((static_cast<float>(v304_data[10])) * v62_data);
              v307_acc += ((static_cast<float>(v304_data[11])) * v64_data);
              v307_acc += ((static_cast<float>(v304_data[12])) * v66_data);
              v307_acc += ((static_cast<float>(v304_data[13])) * v68_data);
              v307_acc += ((static_cast<float>(v304_data[14])) * v70_data);
              v307_acc += ((static_cast<float>(v304_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v342_data = tensorforge::slmLoad<float, 16>(s0 + (111_i32));
              v307_acc += ((static_cast<float>(v342_data[0])) * v74_data);
              ir0.template select<16, 1>(96) = v307_acc;
              tensorforge::intel_esimd::simd<float, 16> v345_acc{};
              v345_acc += ((static_cast<float>(v342_data[1])) * v44_data);
              v345_acc += ((static_cast<float>(v342_data[2])) * v46_data);
              v345_acc += ((static_cast<float>(v342_data[3])) * v48_data);
              v345_acc += ((static_cast<float>(v342_data[4])) * v50_data);
              v345_acc += ((static_cast<float>(v342_data[5])) * v52_data);
              v345_acc += ((static_cast<float>(v342_data[6])) * v54_data);
              v345_acc += ((static_cast<float>(v342_data[7])) * v56_data);
              v345_acc += ((static_cast<float>(v342_data[8])) * v58_data);
              v345_acc += ((static_cast<float>(v342_data[9])) * v60_data);
              v345_acc += ((static_cast<float>(v342_data[10])) * v62_data);
              v345_acc += ((static_cast<float>(v342_data[11])) * v64_data);
              v345_acc += ((static_cast<float>(v342_data[12])) * v66_data);
              v345_acc += ((static_cast<float>(v342_data[13])) * v68_data);
              v345_acc += ((static_cast<float>(v342_data[14])) * v70_data);
              v345_acc += ((static_cast<float>(v342_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v380_data = tensorforge::slmLoad<float, 16>(s0 + (127_i32));
              v345_acc += ((static_cast<float>(v380_data[0])) * v74_data);
              ir0.template select<16, 1>(112) = v345_acc;
              tensorforge::intel_esimd::simd<float, 16> v383_acc{};
              v383_acc += ((static_cast<float>(v380_data[1])) * v44_data);
              v383_acc += ((static_cast<float>(v380_data[2])) * v46_data);
              v383_acc += ((static_cast<float>(v380_data[3])) * v48_data);
              v383_acc += ((static_cast<float>(v380_data[4])) * v50_data);
              v383_acc += ((static_cast<float>(v380_data[5])) * v52_data);
              v383_acc += ((static_cast<float>(v380_data[6])) * v54_data);
              v383_acc += ((static_cast<float>(v380_data[7])) * v56_data);
              v383_acc += ((static_cast<float>(v380_data[8])) * v58_data);
              v383_acc += ((static_cast<float>(v380_data[9])) * v60_data);
              v383_acc += ((static_cast<float>(v380_data[10])) * v62_data);
              v383_acc += ((static_cast<float>(v380_data[11])) * v64_data);
              v383_acc += ((static_cast<float>(v380_data[12])) * v66_data);
              v383_acc += ((static_cast<float>(v380_data[13])) * v68_data);
              v383_acc += ((static_cast<float>(v380_data[14])) * v70_data);
              v383_acc += ((static_cast<float>(v380_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v418_data = tensorforge::slmLoad<float, 16>(s0 + (143_i32));
              v383_acc += ((static_cast<float>(v418_data[0])) * v74_data);
              ir0.template select<16, 1>(128) = v383_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v421_n0 = 0; v421_n0 < 1; ++v421_n0) {
                int32_t v423_a = v421_n0 * 16;
                #pragma unroll
                for (int32_t v422_n1 = 0; v422_n1 < 9; ++v422_n1) {
                  int32_t v425_a = v423_a + (v422_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v426_data(ir0.template select<16, 1>(v425_a));
                  r0.template select<16, 1>(v425_a) = v426_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v427_i0 = 0; v427_i0 < 1; ++v427_i0) {
                int32_t v429_a = v427_i0 * 16;
                #pragma unroll
                for (int32_t v428_i1 = 0; v428_i1 < 9; ++v428_i1) {
                  int32_t v431_a = v429_a + (v428_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v432_data(r0.template select<16, 1>(v431_a));
                  v432_data.copy_to(glb_m0 + (v431_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


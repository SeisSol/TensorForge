// === base name ===
kernel_4b2e033353751299

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4b2e033353751299 = {{1, 32, 1}, 16, 16, 1, 32, 21504, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4b2e033353751299(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4b2e033353751299(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4b2e033353751299(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4b2e033353751299(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4b2e033353751299(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4b2e033353751299(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4b2e033353751299(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (144);
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
            tensorforge::intel_esimd::simd<float, 16> v22_ld;
            v22_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v22_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 16> v23_ld;
            v23_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v23_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<float, 16> v24_ld;
            v24_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v24_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<float, 16> v25_ld;
            v25_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v25_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<float, 16> v26_ld;
            v26_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v26_ld);
          }
          if (item.get_local_id(1) == 15) {
            tensorforge::intel_esimd::simd<float, 16> v27_ld;
            v27_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v27_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v29_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v29_batchId0 < numElements0; v29_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v30_ahead1 = v29_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v32_batchId1 = (v30_ahead1 < numElements0) ? v30_ahead1 : v29_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v29_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v29_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v29_batchId0 * 144 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v39_ld;
              v39_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v39_ld);
              tensorforge::intel_esimd::simd<float, 64> v40_ld;
              v40_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v40_ld);
              tensorforge::intel_esimd::simd<float, 16> v41_ld;
              v41_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v41_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 9)] [(1, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v49_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v51_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v53_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v57_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v59_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v61_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v65_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v67_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v69_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v73_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v75_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v77_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v78_acc{};
              tensorforge::intel_esimd::simd<float, 16> v81_data(0.0f);
              v81_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v78_acc += ((static_cast<float>(v81_data[1])) * v47_data);
              v78_acc += ((static_cast<float>(v81_data[2])) * v49_data);
              v78_acc += ((static_cast<float>(v81_data[3])) * v51_data);
              v78_acc += ((static_cast<float>(v81_data[4])) * v53_data);
              v78_acc += ((static_cast<float>(v81_data[5])) * v55_data);
              v78_acc += ((static_cast<float>(v81_data[6])) * v57_data);
              v78_acc += ((static_cast<float>(v81_data[7])) * v59_data);
              v78_acc += ((static_cast<float>(v81_data[8])) * v61_data);
              v78_acc += ((static_cast<float>(v81_data[9])) * v63_data);
              v78_acc += ((static_cast<float>(v81_data[10])) * v65_data);
              v78_acc += ((static_cast<float>(v81_data[11])) * v67_data);
              v78_acc += ((static_cast<float>(v81_data[12])) * v69_data);
              v78_acc += ((static_cast<float>(v81_data[13])) * v71_data);
              v78_acc += ((static_cast<float>(v81_data[14])) * v73_data);
              v78_acc += ((static_cast<float>(v81_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v117_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v78_acc += ((static_cast<float>(v117_data[0])) * v77_data);
              ir0.template select<16, 1>(0) = v78_acc;
              tensorforge::intel_esimd::simd<float, 16> v120_acc{};
              v120_acc += ((static_cast<float>(v117_data[1])) * v47_data);
              v120_acc += ((static_cast<float>(v117_data[2])) * v49_data);
              v120_acc += ((static_cast<float>(v117_data[3])) * v51_data);
              v120_acc += ((static_cast<float>(v117_data[4])) * v53_data);
              v120_acc += ((static_cast<float>(v117_data[5])) * v55_data);
              v120_acc += ((static_cast<float>(v117_data[6])) * v57_data);
              v120_acc += ((static_cast<float>(v117_data[7])) * v59_data);
              v120_acc += ((static_cast<float>(v117_data[8])) * v61_data);
              v120_acc += ((static_cast<float>(v117_data[9])) * v63_data);
              v120_acc += ((static_cast<float>(v117_data[10])) * v65_data);
              v120_acc += ((static_cast<float>(v117_data[11])) * v67_data);
              v120_acc += ((static_cast<float>(v117_data[12])) * v69_data);
              v120_acc += ((static_cast<float>(v117_data[13])) * v71_data);
              v120_acc += ((static_cast<float>(v117_data[14])) * v73_data);
              v120_acc += ((static_cast<float>(v117_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v155_data = tensorforge::slmLoad<float, 16>(s0 + (31_i32));
              v120_acc += ((static_cast<float>(v155_data[0])) * v77_data);
              ir0.template select<16, 1>(16) = v120_acc;
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              v158_acc += ((static_cast<float>(v155_data[1])) * v47_data);
              v158_acc += ((static_cast<float>(v155_data[2])) * v49_data);
              v158_acc += ((static_cast<float>(v155_data[3])) * v51_data);
              v158_acc += ((static_cast<float>(v155_data[4])) * v53_data);
              v158_acc += ((static_cast<float>(v155_data[5])) * v55_data);
              v158_acc += ((static_cast<float>(v155_data[6])) * v57_data);
              v158_acc += ((static_cast<float>(v155_data[7])) * v59_data);
              v158_acc += ((static_cast<float>(v155_data[8])) * v61_data);
              v158_acc += ((static_cast<float>(v155_data[9])) * v63_data);
              v158_acc += ((static_cast<float>(v155_data[10])) * v65_data);
              v158_acc += ((static_cast<float>(v155_data[11])) * v67_data);
              v158_acc += ((static_cast<float>(v155_data[12])) * v69_data);
              v158_acc += ((static_cast<float>(v155_data[13])) * v71_data);
              v158_acc += ((static_cast<float>(v155_data[14])) * v73_data);
              v158_acc += ((static_cast<float>(v155_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v193_data = tensorforge::slmLoad<float, 16>(s0 + (47_i32));
              v158_acc += ((static_cast<float>(v193_data[0])) * v77_data);
              ir0.template select<16, 1>(32) = v158_acc;
              tensorforge::intel_esimd::simd<float, 16> v196_acc{};
              v196_acc += ((static_cast<float>(v193_data[1])) * v47_data);
              v196_acc += ((static_cast<float>(v193_data[2])) * v49_data);
              v196_acc += ((static_cast<float>(v193_data[3])) * v51_data);
              v196_acc += ((static_cast<float>(v193_data[4])) * v53_data);
              v196_acc += ((static_cast<float>(v193_data[5])) * v55_data);
              v196_acc += ((static_cast<float>(v193_data[6])) * v57_data);
              v196_acc += ((static_cast<float>(v193_data[7])) * v59_data);
              v196_acc += ((static_cast<float>(v193_data[8])) * v61_data);
              v196_acc += ((static_cast<float>(v193_data[9])) * v63_data);
              v196_acc += ((static_cast<float>(v193_data[10])) * v65_data);
              v196_acc += ((static_cast<float>(v193_data[11])) * v67_data);
              v196_acc += ((static_cast<float>(v193_data[12])) * v69_data);
              v196_acc += ((static_cast<float>(v193_data[13])) * v71_data);
              v196_acc += ((static_cast<float>(v193_data[14])) * v73_data);
              v196_acc += ((static_cast<float>(v193_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v231_data = tensorforge::slmLoad<float, 16>(s0 + (63_i32));
              v196_acc += ((static_cast<float>(v231_data[0])) * v77_data);
              ir0.template select<16, 1>(48) = v196_acc;
              tensorforge::intel_esimd::simd<float, 16> v234_acc{};
              v234_acc += ((static_cast<float>(v231_data[1])) * v47_data);
              v234_acc += ((static_cast<float>(v231_data[2])) * v49_data);
              v234_acc += ((static_cast<float>(v231_data[3])) * v51_data);
              v234_acc += ((static_cast<float>(v231_data[4])) * v53_data);
              v234_acc += ((static_cast<float>(v231_data[5])) * v55_data);
              v234_acc += ((static_cast<float>(v231_data[6])) * v57_data);
              v234_acc += ((static_cast<float>(v231_data[7])) * v59_data);
              v234_acc += ((static_cast<float>(v231_data[8])) * v61_data);
              v234_acc += ((static_cast<float>(v231_data[9])) * v63_data);
              v234_acc += ((static_cast<float>(v231_data[10])) * v65_data);
              v234_acc += ((static_cast<float>(v231_data[11])) * v67_data);
              v234_acc += ((static_cast<float>(v231_data[12])) * v69_data);
              v234_acc += ((static_cast<float>(v231_data[13])) * v71_data);
              v234_acc += ((static_cast<float>(v231_data[14])) * v73_data);
              v234_acc += ((static_cast<float>(v231_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v269_data = tensorforge::slmLoad<float, 16>(s0 + (79_i32));
              v234_acc += ((static_cast<float>(v269_data[0])) * v77_data);
              ir0.template select<16, 1>(64) = v234_acc;
              tensorforge::intel_esimd::simd<float, 16> v272_acc{};
              v272_acc += ((static_cast<float>(v269_data[1])) * v47_data);
              v272_acc += ((static_cast<float>(v269_data[2])) * v49_data);
              v272_acc += ((static_cast<float>(v269_data[3])) * v51_data);
              v272_acc += ((static_cast<float>(v269_data[4])) * v53_data);
              v272_acc += ((static_cast<float>(v269_data[5])) * v55_data);
              v272_acc += ((static_cast<float>(v269_data[6])) * v57_data);
              v272_acc += ((static_cast<float>(v269_data[7])) * v59_data);
              v272_acc += ((static_cast<float>(v269_data[8])) * v61_data);
              v272_acc += ((static_cast<float>(v269_data[9])) * v63_data);
              v272_acc += ((static_cast<float>(v269_data[10])) * v65_data);
              v272_acc += ((static_cast<float>(v269_data[11])) * v67_data);
              v272_acc += ((static_cast<float>(v269_data[12])) * v69_data);
              v272_acc += ((static_cast<float>(v269_data[13])) * v71_data);
              v272_acc += ((static_cast<float>(v269_data[14])) * v73_data);
              v272_acc += ((static_cast<float>(v269_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v307_data = tensorforge::slmLoad<float, 16>(s0 + (95_i32));
              v272_acc += ((static_cast<float>(v307_data[0])) * v77_data);
              ir0.template select<16, 1>(80) = v272_acc;
              tensorforge::intel_esimd::simd<float, 16> v310_acc{};
              v310_acc += ((static_cast<float>(v307_data[1])) * v47_data);
              v310_acc += ((static_cast<float>(v307_data[2])) * v49_data);
              v310_acc += ((static_cast<float>(v307_data[3])) * v51_data);
              v310_acc += ((static_cast<float>(v307_data[4])) * v53_data);
              v310_acc += ((static_cast<float>(v307_data[5])) * v55_data);
              v310_acc += ((static_cast<float>(v307_data[6])) * v57_data);
              v310_acc += ((static_cast<float>(v307_data[7])) * v59_data);
              v310_acc += ((static_cast<float>(v307_data[8])) * v61_data);
              v310_acc += ((static_cast<float>(v307_data[9])) * v63_data);
              v310_acc += ((static_cast<float>(v307_data[10])) * v65_data);
              v310_acc += ((static_cast<float>(v307_data[11])) * v67_data);
              v310_acc += ((static_cast<float>(v307_data[12])) * v69_data);
              v310_acc += ((static_cast<float>(v307_data[13])) * v71_data);
              v310_acc += ((static_cast<float>(v307_data[14])) * v73_data);
              v310_acc += ((static_cast<float>(v307_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v345_data = tensorforge::slmLoad<float, 16>(s0 + (111_i32));
              v310_acc += ((static_cast<float>(v345_data[0])) * v77_data);
              ir0.template select<16, 1>(96) = v310_acc;
              tensorforge::intel_esimd::simd<float, 16> v348_acc{};
              v348_acc += ((static_cast<float>(v345_data[1])) * v47_data);
              v348_acc += ((static_cast<float>(v345_data[2])) * v49_data);
              v348_acc += ((static_cast<float>(v345_data[3])) * v51_data);
              v348_acc += ((static_cast<float>(v345_data[4])) * v53_data);
              v348_acc += ((static_cast<float>(v345_data[5])) * v55_data);
              v348_acc += ((static_cast<float>(v345_data[6])) * v57_data);
              v348_acc += ((static_cast<float>(v345_data[7])) * v59_data);
              v348_acc += ((static_cast<float>(v345_data[8])) * v61_data);
              v348_acc += ((static_cast<float>(v345_data[9])) * v63_data);
              v348_acc += ((static_cast<float>(v345_data[10])) * v65_data);
              v348_acc += ((static_cast<float>(v345_data[11])) * v67_data);
              v348_acc += ((static_cast<float>(v345_data[12])) * v69_data);
              v348_acc += ((static_cast<float>(v345_data[13])) * v71_data);
              v348_acc += ((static_cast<float>(v345_data[14])) * v73_data);
              v348_acc += ((static_cast<float>(v345_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v383_data = tensorforge::slmLoad<float, 16>(s0 + (127_i32));
              v348_acc += ((static_cast<float>(v383_data[0])) * v77_data);
              ir0.template select<16, 1>(112) = v348_acc;
              tensorforge::intel_esimd::simd<float, 16> v386_acc{};
              v386_acc += ((static_cast<float>(v383_data[1])) * v47_data);
              v386_acc += ((static_cast<float>(v383_data[2])) * v49_data);
              v386_acc += ((static_cast<float>(v383_data[3])) * v51_data);
              v386_acc += ((static_cast<float>(v383_data[4])) * v53_data);
              v386_acc += ((static_cast<float>(v383_data[5])) * v55_data);
              v386_acc += ((static_cast<float>(v383_data[6])) * v57_data);
              v386_acc += ((static_cast<float>(v383_data[7])) * v59_data);
              v386_acc += ((static_cast<float>(v383_data[8])) * v61_data);
              v386_acc += ((static_cast<float>(v383_data[9])) * v63_data);
              v386_acc += ((static_cast<float>(v383_data[10])) * v65_data);
              v386_acc += ((static_cast<float>(v383_data[11])) * v67_data);
              v386_acc += ((static_cast<float>(v383_data[12])) * v69_data);
              v386_acc += ((static_cast<float>(v383_data[13])) * v71_data);
              v386_acc += ((static_cast<float>(v383_data[14])) * v73_data);
              v386_acc += ((static_cast<float>(v383_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v421_data = tensorforge::slmLoad<float, 16>(s0 + (143_i32));
              v386_acc += ((static_cast<float>(v421_data[0])) * v77_data);
              ir0.template select<16, 1>(128) = v386_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v424_n0 = 0; v424_n0 < 1; ++v424_n0) {
                int32_t v426_a = v424_n0 * 16;
                #pragma unroll
                for (int32_t v425_n1 = 0; v425_n1 < 9; ++v425_n1) {
                  int32_t v428_a = v426_a + (v425_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v429_data(ir0.template select<16, 1>(v428_a));
                  r0.template select<16, 1>(v428_a) = v429_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v430_i0 = 0; v430_i0 < 1; ++v430_i0) {
                int32_t v432_a = v430_i0 * 16;
                #pragma unroll
                for (int32_t v431_i1 = 0; v431_i1 < 9; ++v431_i1) {
                  int32_t v434_a = v432_a + (v431_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v435_data(r0.template select<16, 1>(v434_a));
                  v435_data.copy_to(glb_m0 + (v434_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


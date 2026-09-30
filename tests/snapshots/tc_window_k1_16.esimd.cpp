// === base name ===
kernel_402fc5709080ab23

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_402fc5709080ab23 = {{1, 32, 1}, 16, 16, 1, 32, 21504, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_402fc5709080ab23(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_402fc5709080ab23(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_402fc5709080ab23(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_402fc5709080ab23(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_402fc5709080ab23(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_402fc5709080ab23(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_402fc5709080ab23(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 256);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (144);
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
            tensorforge::intel_esimd::simd<float, 16> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<float, 16> v16_ld;
            v16_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<float, 16> v17_ld;
            v17_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<float, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<float, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 15) {
            tensorforge::intel_esimd::simd<float, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v23_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v23_batchId0 < numElements0; v23_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v24_ahead1 = v23_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v26_batchId1 = (v24_ahead1 < numElements0) ? v24_ahead1 : v23_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v23_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v23_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v23_batchId0 * 144 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v33_ld);
              tensorforge::intel_esimd::simd<float, 64> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v34_ld);
              tensorforge::intel_esimd::simd<float, 16> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v35_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 9)] [(1, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v41_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v43_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v45_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v47_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v49_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v51_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v53_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v55_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v57_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v59_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v61_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v63_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v67_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v69_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v71_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v72_acc{};
              tensorforge::intel_esimd::simd<float, 16> v75_data(0.0f);
              v75_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v72_acc += ((static_cast<float>(v75_data[1])) * v41_data);
              v72_acc += ((static_cast<float>(v75_data[2])) * v43_data);
              v72_acc += ((static_cast<float>(v75_data[3])) * v45_data);
              v72_acc += ((static_cast<float>(v75_data[4])) * v47_data);
              v72_acc += ((static_cast<float>(v75_data[5])) * v49_data);
              v72_acc += ((static_cast<float>(v75_data[6])) * v51_data);
              v72_acc += ((static_cast<float>(v75_data[7])) * v53_data);
              v72_acc += ((static_cast<float>(v75_data[8])) * v55_data);
              v72_acc += ((static_cast<float>(v75_data[9])) * v57_data);
              v72_acc += ((static_cast<float>(v75_data[10])) * v59_data);
              v72_acc += ((static_cast<float>(v75_data[11])) * v61_data);
              v72_acc += ((static_cast<float>(v75_data[12])) * v63_data);
              v72_acc += ((static_cast<float>(v75_data[13])) * v65_data);
              v72_acc += ((static_cast<float>(v75_data[14])) * v67_data);
              v72_acc += ((static_cast<float>(v75_data[15])) * v69_data);
              tensorforge::intel_esimd::simd<float, 16> v111_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v72_acc += ((static_cast<float>(v111_data[0])) * v71_data);
              ir0.template select<16, 1>(0) = v72_acc;
              tensorforge::intel_esimd::simd<float, 16> v114_acc{};
              v114_acc += ((static_cast<float>(v111_data[1])) * v41_data);
              v114_acc += ((static_cast<float>(v111_data[2])) * v43_data);
              v114_acc += ((static_cast<float>(v111_data[3])) * v45_data);
              v114_acc += ((static_cast<float>(v111_data[4])) * v47_data);
              v114_acc += ((static_cast<float>(v111_data[5])) * v49_data);
              v114_acc += ((static_cast<float>(v111_data[6])) * v51_data);
              v114_acc += ((static_cast<float>(v111_data[7])) * v53_data);
              v114_acc += ((static_cast<float>(v111_data[8])) * v55_data);
              v114_acc += ((static_cast<float>(v111_data[9])) * v57_data);
              v114_acc += ((static_cast<float>(v111_data[10])) * v59_data);
              v114_acc += ((static_cast<float>(v111_data[11])) * v61_data);
              v114_acc += ((static_cast<float>(v111_data[12])) * v63_data);
              v114_acc += ((static_cast<float>(v111_data[13])) * v65_data);
              v114_acc += ((static_cast<float>(v111_data[14])) * v67_data);
              v114_acc += ((static_cast<float>(v111_data[15])) * v69_data);
              tensorforge::intel_esimd::simd<float, 16> v149_data = tensorforge::slmLoad<float, 16>(s0 + (31_i32));
              v114_acc += ((static_cast<float>(v149_data[0])) * v71_data);
              ir0.template select<16, 1>(16) = v114_acc;
              tensorforge::intel_esimd::simd<float, 16> v152_acc{};
              v152_acc += ((static_cast<float>(v149_data[1])) * v41_data);
              v152_acc += ((static_cast<float>(v149_data[2])) * v43_data);
              v152_acc += ((static_cast<float>(v149_data[3])) * v45_data);
              v152_acc += ((static_cast<float>(v149_data[4])) * v47_data);
              v152_acc += ((static_cast<float>(v149_data[5])) * v49_data);
              v152_acc += ((static_cast<float>(v149_data[6])) * v51_data);
              v152_acc += ((static_cast<float>(v149_data[7])) * v53_data);
              v152_acc += ((static_cast<float>(v149_data[8])) * v55_data);
              v152_acc += ((static_cast<float>(v149_data[9])) * v57_data);
              v152_acc += ((static_cast<float>(v149_data[10])) * v59_data);
              v152_acc += ((static_cast<float>(v149_data[11])) * v61_data);
              v152_acc += ((static_cast<float>(v149_data[12])) * v63_data);
              v152_acc += ((static_cast<float>(v149_data[13])) * v65_data);
              v152_acc += ((static_cast<float>(v149_data[14])) * v67_data);
              v152_acc += ((static_cast<float>(v149_data[15])) * v69_data);
              tensorforge::intel_esimd::simd<float, 16> v187_data = tensorforge::slmLoad<float, 16>(s0 + (47_i32));
              v152_acc += ((static_cast<float>(v187_data[0])) * v71_data);
              ir0.template select<16, 1>(32) = v152_acc;
              tensorforge::intel_esimd::simd<float, 16> v190_acc{};
              v190_acc += ((static_cast<float>(v187_data[1])) * v41_data);
              v190_acc += ((static_cast<float>(v187_data[2])) * v43_data);
              v190_acc += ((static_cast<float>(v187_data[3])) * v45_data);
              v190_acc += ((static_cast<float>(v187_data[4])) * v47_data);
              v190_acc += ((static_cast<float>(v187_data[5])) * v49_data);
              v190_acc += ((static_cast<float>(v187_data[6])) * v51_data);
              v190_acc += ((static_cast<float>(v187_data[7])) * v53_data);
              v190_acc += ((static_cast<float>(v187_data[8])) * v55_data);
              v190_acc += ((static_cast<float>(v187_data[9])) * v57_data);
              v190_acc += ((static_cast<float>(v187_data[10])) * v59_data);
              v190_acc += ((static_cast<float>(v187_data[11])) * v61_data);
              v190_acc += ((static_cast<float>(v187_data[12])) * v63_data);
              v190_acc += ((static_cast<float>(v187_data[13])) * v65_data);
              v190_acc += ((static_cast<float>(v187_data[14])) * v67_data);
              v190_acc += ((static_cast<float>(v187_data[15])) * v69_data);
              tensorforge::intel_esimd::simd<float, 16> v225_data = tensorforge::slmLoad<float, 16>(s0 + (63_i32));
              v190_acc += ((static_cast<float>(v225_data[0])) * v71_data);
              ir0.template select<16, 1>(48) = v190_acc;
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              v228_acc += ((static_cast<float>(v225_data[1])) * v41_data);
              v228_acc += ((static_cast<float>(v225_data[2])) * v43_data);
              v228_acc += ((static_cast<float>(v225_data[3])) * v45_data);
              v228_acc += ((static_cast<float>(v225_data[4])) * v47_data);
              v228_acc += ((static_cast<float>(v225_data[5])) * v49_data);
              v228_acc += ((static_cast<float>(v225_data[6])) * v51_data);
              v228_acc += ((static_cast<float>(v225_data[7])) * v53_data);
              v228_acc += ((static_cast<float>(v225_data[8])) * v55_data);
              v228_acc += ((static_cast<float>(v225_data[9])) * v57_data);
              v228_acc += ((static_cast<float>(v225_data[10])) * v59_data);
              v228_acc += ((static_cast<float>(v225_data[11])) * v61_data);
              v228_acc += ((static_cast<float>(v225_data[12])) * v63_data);
              v228_acc += ((static_cast<float>(v225_data[13])) * v65_data);
              v228_acc += ((static_cast<float>(v225_data[14])) * v67_data);
              v228_acc += ((static_cast<float>(v225_data[15])) * v69_data);
              tensorforge::intel_esimd::simd<float, 16> v263_data = tensorforge::slmLoad<float, 16>(s0 + (79_i32));
              v228_acc += ((static_cast<float>(v263_data[0])) * v71_data);
              ir0.template select<16, 1>(64) = v228_acc;
              tensorforge::intel_esimd::simd<float, 16> v266_acc{};
              v266_acc += ((static_cast<float>(v263_data[1])) * v41_data);
              v266_acc += ((static_cast<float>(v263_data[2])) * v43_data);
              v266_acc += ((static_cast<float>(v263_data[3])) * v45_data);
              v266_acc += ((static_cast<float>(v263_data[4])) * v47_data);
              v266_acc += ((static_cast<float>(v263_data[5])) * v49_data);
              v266_acc += ((static_cast<float>(v263_data[6])) * v51_data);
              v266_acc += ((static_cast<float>(v263_data[7])) * v53_data);
              v266_acc += ((static_cast<float>(v263_data[8])) * v55_data);
              v266_acc += ((static_cast<float>(v263_data[9])) * v57_data);
              v266_acc += ((static_cast<float>(v263_data[10])) * v59_data);
              v266_acc += ((static_cast<float>(v263_data[11])) * v61_data);
              v266_acc += ((static_cast<float>(v263_data[12])) * v63_data);
              v266_acc += ((static_cast<float>(v263_data[13])) * v65_data);
              v266_acc += ((static_cast<float>(v263_data[14])) * v67_data);
              v266_acc += ((static_cast<float>(v263_data[15])) * v69_data);
              tensorforge::intel_esimd::simd<float, 16> v301_data = tensorforge::slmLoad<float, 16>(s0 + (95_i32));
              v266_acc += ((static_cast<float>(v301_data[0])) * v71_data);
              ir0.template select<16, 1>(80) = v266_acc;
              tensorforge::intel_esimd::simd<float, 16> v304_acc{};
              v304_acc += ((static_cast<float>(v301_data[1])) * v41_data);
              v304_acc += ((static_cast<float>(v301_data[2])) * v43_data);
              v304_acc += ((static_cast<float>(v301_data[3])) * v45_data);
              v304_acc += ((static_cast<float>(v301_data[4])) * v47_data);
              v304_acc += ((static_cast<float>(v301_data[5])) * v49_data);
              v304_acc += ((static_cast<float>(v301_data[6])) * v51_data);
              v304_acc += ((static_cast<float>(v301_data[7])) * v53_data);
              v304_acc += ((static_cast<float>(v301_data[8])) * v55_data);
              v304_acc += ((static_cast<float>(v301_data[9])) * v57_data);
              v304_acc += ((static_cast<float>(v301_data[10])) * v59_data);
              v304_acc += ((static_cast<float>(v301_data[11])) * v61_data);
              v304_acc += ((static_cast<float>(v301_data[12])) * v63_data);
              v304_acc += ((static_cast<float>(v301_data[13])) * v65_data);
              v304_acc += ((static_cast<float>(v301_data[14])) * v67_data);
              v304_acc += ((static_cast<float>(v301_data[15])) * v69_data);
              tensorforge::intel_esimd::simd<float, 16> v339_data = tensorforge::slmLoad<float, 16>(s0 + (111_i32));
              v304_acc += ((static_cast<float>(v339_data[0])) * v71_data);
              ir0.template select<16, 1>(96) = v304_acc;
              tensorforge::intel_esimd::simd<float, 16> v342_acc{};
              v342_acc += ((static_cast<float>(v339_data[1])) * v41_data);
              v342_acc += ((static_cast<float>(v339_data[2])) * v43_data);
              v342_acc += ((static_cast<float>(v339_data[3])) * v45_data);
              v342_acc += ((static_cast<float>(v339_data[4])) * v47_data);
              v342_acc += ((static_cast<float>(v339_data[5])) * v49_data);
              v342_acc += ((static_cast<float>(v339_data[6])) * v51_data);
              v342_acc += ((static_cast<float>(v339_data[7])) * v53_data);
              v342_acc += ((static_cast<float>(v339_data[8])) * v55_data);
              v342_acc += ((static_cast<float>(v339_data[9])) * v57_data);
              v342_acc += ((static_cast<float>(v339_data[10])) * v59_data);
              v342_acc += ((static_cast<float>(v339_data[11])) * v61_data);
              v342_acc += ((static_cast<float>(v339_data[12])) * v63_data);
              v342_acc += ((static_cast<float>(v339_data[13])) * v65_data);
              v342_acc += ((static_cast<float>(v339_data[14])) * v67_data);
              v342_acc += ((static_cast<float>(v339_data[15])) * v69_data);
              tensorforge::intel_esimd::simd<float, 16> v377_data = tensorforge::slmLoad<float, 16>(s0 + (127_i32));
              v342_acc += ((static_cast<float>(v377_data[0])) * v71_data);
              ir0.template select<16, 1>(112) = v342_acc;
              tensorforge::intel_esimd::simd<float, 16> v380_acc{};
              v380_acc += ((static_cast<float>(v377_data[1])) * v41_data);
              v380_acc += ((static_cast<float>(v377_data[2])) * v43_data);
              v380_acc += ((static_cast<float>(v377_data[3])) * v45_data);
              v380_acc += ((static_cast<float>(v377_data[4])) * v47_data);
              v380_acc += ((static_cast<float>(v377_data[5])) * v49_data);
              v380_acc += ((static_cast<float>(v377_data[6])) * v51_data);
              v380_acc += ((static_cast<float>(v377_data[7])) * v53_data);
              v380_acc += ((static_cast<float>(v377_data[8])) * v55_data);
              v380_acc += ((static_cast<float>(v377_data[9])) * v57_data);
              v380_acc += ((static_cast<float>(v377_data[10])) * v59_data);
              v380_acc += ((static_cast<float>(v377_data[11])) * v61_data);
              v380_acc += ((static_cast<float>(v377_data[12])) * v63_data);
              v380_acc += ((static_cast<float>(v377_data[13])) * v65_data);
              v380_acc += ((static_cast<float>(v377_data[14])) * v67_data);
              v380_acc += ((static_cast<float>(v377_data[15])) * v69_data);
              tensorforge::intel_esimd::simd<float, 16> v415_data = tensorforge::slmLoad<float, 16>(s0 + (143_i32));
              v380_acc += ((static_cast<float>(v415_data[0])) * v71_data);
              ir0.template select<16, 1>(128) = v380_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v418_n0 = 0; v418_n0 < 1; ++v418_n0) {
                int32_t v420_a = v418_n0 * 16;
                #pragma unroll
                for (int32_t v419_n1 = 0; v419_n1 < 9; ++v419_n1) {
                  int32_t v422_a = v420_a + (v419_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v423_data(ir0.template select<16, 1>(v422_a));
                  r0.template select<16, 1>(v422_a) = v423_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v424_i0 = 0; v424_i0 < 1; ++v424_i0) {
                int32_t v426_a = v424_i0 * 16;
                #pragma unroll
                for (int32_t v425_i1 = 0; v425_i1 < 9; ++v425_i1) {
                  int32_t v428_a = v426_a + (v425_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v429_data(r0.template select<16, 1>(v428_a));
                  v429_data.copy_to(glb_m0 + (v428_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


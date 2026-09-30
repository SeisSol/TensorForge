// === base name ===
kernel_d0a58d81f604b183

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d0a58d81f604b183 = {{1, 32, 1}, 16, 10, 1, 32, 23232, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d0a58d81f604b183(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d0a58d81f604b183(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d0a58d81f604b183(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_d0a58d81f604b183(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d0a58d81f604b183(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d0a58d81f604b183(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d0a58d81f604b183(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 176);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (160);
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
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v18_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v18_batchId0 < numElements0; v18_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v19_ahead1 = v18_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v21_batchId1 = (v19_ahead1 < numElements0) ? v19_ahead1 : v18_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v18_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v18_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v18_batchId0 * 153 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v28_ld);
              tensorforge::intel_esimd::simd<float, 64> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v29_ld);
              tensorforge::intel_esimd::simd<float, 16> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v30_ld);
              tensorforge::intel_esimd::simd<float, 9> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v31_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v37_data = tensorforge::slmLoad<float, 16>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v39_data = tensorforge::slmLoad<float, 16>(glb_m1 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v41_data = tensorforge::slmLoad<float, 16>(glb_m1 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v43_data = tensorforge::slmLoad<float, 16>(glb_m1 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v45_data = tensorforge::slmLoad<float, 16>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data = tensorforge::slmLoad<float, 16>(glb_m1 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v49_data = tensorforge::slmLoad<float, 16>(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data = tensorforge::slmLoad<float, 16>(glb_m1 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v53_data = tensorforge::slmLoad<float, 16>(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data = tensorforge::slmLoad<float, 16>(glb_m1 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v57_data = tensorforge::slmLoad<float, 16>(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data = tensorforge::slmLoad<float, 16>(glb_m1 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v61_data = tensorforge::slmLoad<float, 16>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data = tensorforge::slmLoad<float, 16>(glb_m1 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data = tensorforge::slmLoad<float, 16>(glb_m1 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data = tensorforge::slmLoad<float, 16>(glb_m1 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v69_data = tensorforge::slmLoad<float, 16>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_acc{};
              tensorforge::intel_esimd::simd<float, 16> v73_data(0.0f);
              v73_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v70_acc += ((static_cast<float>(v73_data[1])) * v37_data);
              v70_acc += ((static_cast<float>(v73_data[2])) * v39_data);
              v70_acc += ((static_cast<float>(v73_data[3])) * v41_data);
              v70_acc += ((static_cast<float>(v73_data[4])) * v43_data);
              v70_acc += ((static_cast<float>(v73_data[5])) * v45_data);
              v70_acc += ((static_cast<float>(v73_data[6])) * v47_data);
              v70_acc += ((static_cast<float>(v73_data[7])) * v49_data);
              v70_acc += ((static_cast<float>(v73_data[8])) * v51_data);
              v70_acc += ((static_cast<float>(v73_data[9])) * v53_data);
              v70_acc += ((static_cast<float>(v73_data[10])) * v55_data);
              v70_acc += ((static_cast<float>(v73_data[11])) * v57_data);
              v70_acc += ((static_cast<float>(v73_data[12])) * v59_data);
              v70_acc += ((static_cast<float>(v73_data[13])) * v61_data);
              v70_acc += ((static_cast<float>(v73_data[14])) * v63_data);
              v70_acc += ((static_cast<float>(v73_data[15])) * v65_data);
              tensorforge::intel_esimd::simd<float, 16> v109_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v70_acc += ((static_cast<float>(v109_data[0])) * v67_data);
              v70_acc += ((static_cast<float>(v109_data[1])) * v69_data);
              ir0.template select<16, 1>(0) = v70_acc;
              tensorforge::intel_esimd::simd<float, 16> v114_acc{};
              tensorforge::intel_esimd::simd<float, 16> v116_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v114_acc += ((static_cast<float>(v116_data[1])) * v37_data);
              v114_acc += ((static_cast<float>(v116_data[2])) * v39_data);
              v114_acc += ((static_cast<float>(v116_data[3])) * v41_data);
              v114_acc += ((static_cast<float>(v116_data[4])) * v43_data);
              v114_acc += ((static_cast<float>(v116_data[5])) * v45_data);
              v114_acc += ((static_cast<float>(v116_data[6])) * v47_data);
              v114_acc += ((static_cast<float>(v116_data[7])) * v49_data);
              v114_acc += ((static_cast<float>(v116_data[8])) * v51_data);
              v114_acc += ((static_cast<float>(v116_data[9])) * v53_data);
              v114_acc += ((static_cast<float>(v116_data[10])) * v55_data);
              v114_acc += ((static_cast<float>(v116_data[11])) * v57_data);
              v114_acc += ((static_cast<float>(v116_data[12])) * v59_data);
              v114_acc += ((static_cast<float>(v116_data[13])) * v61_data);
              v114_acc += ((static_cast<float>(v116_data[14])) * v63_data);
              v114_acc += ((static_cast<float>(v116_data[15])) * v65_data);
              tensorforge::intel_esimd::simd<float, 16> v149_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v114_acc += ((static_cast<float>(v149_data[0])) * v67_data);
              v114_acc += ((static_cast<float>(v149_data[1])) * v69_data);
              ir0.template select<16, 1>(16) = v114_acc;
              tensorforge::intel_esimd::simd<float, 16> v154_acc{};
              tensorforge::intel_esimd::simd<float, 16> v156_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v154_acc += ((static_cast<float>(v156_data[1])) * v37_data);
              v154_acc += ((static_cast<float>(v156_data[2])) * v39_data);
              v154_acc += ((static_cast<float>(v156_data[3])) * v41_data);
              v154_acc += ((static_cast<float>(v156_data[4])) * v43_data);
              v154_acc += ((static_cast<float>(v156_data[5])) * v45_data);
              v154_acc += ((static_cast<float>(v156_data[6])) * v47_data);
              v154_acc += ((static_cast<float>(v156_data[7])) * v49_data);
              v154_acc += ((static_cast<float>(v156_data[8])) * v51_data);
              v154_acc += ((static_cast<float>(v156_data[9])) * v53_data);
              v154_acc += ((static_cast<float>(v156_data[10])) * v55_data);
              v154_acc += ((static_cast<float>(v156_data[11])) * v57_data);
              v154_acc += ((static_cast<float>(v156_data[12])) * v59_data);
              v154_acc += ((static_cast<float>(v156_data[13])) * v61_data);
              v154_acc += ((static_cast<float>(v156_data[14])) * v63_data);
              v154_acc += ((static_cast<float>(v156_data[15])) * v65_data);
              tensorforge::intel_esimd::simd<float, 16> v189_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v154_acc += ((static_cast<float>(v189_data[0])) * v67_data);
              v154_acc += ((static_cast<float>(v189_data[1])) * v69_data);
              ir0.template select<16, 1>(32) = v154_acc;
              tensorforge::intel_esimd::simd<float, 16> v194_acc{};
              tensorforge::intel_esimd::simd<float, 16> v196_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v194_acc += ((static_cast<float>(v196_data[1])) * v37_data);
              v194_acc += ((static_cast<float>(v196_data[2])) * v39_data);
              v194_acc += ((static_cast<float>(v196_data[3])) * v41_data);
              v194_acc += ((static_cast<float>(v196_data[4])) * v43_data);
              v194_acc += ((static_cast<float>(v196_data[5])) * v45_data);
              v194_acc += ((static_cast<float>(v196_data[6])) * v47_data);
              v194_acc += ((static_cast<float>(v196_data[7])) * v49_data);
              v194_acc += ((static_cast<float>(v196_data[8])) * v51_data);
              v194_acc += ((static_cast<float>(v196_data[9])) * v53_data);
              v194_acc += ((static_cast<float>(v196_data[10])) * v55_data);
              v194_acc += ((static_cast<float>(v196_data[11])) * v57_data);
              v194_acc += ((static_cast<float>(v196_data[12])) * v59_data);
              v194_acc += ((static_cast<float>(v196_data[13])) * v61_data);
              v194_acc += ((static_cast<float>(v196_data[14])) * v63_data);
              v194_acc += ((static_cast<float>(v196_data[15])) * v65_data);
              tensorforge::intel_esimd::simd<float, 16> v229_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v194_acc += ((static_cast<float>(v229_data[0])) * v67_data);
              v194_acc += ((static_cast<float>(v229_data[1])) * v69_data);
              ir0.template select<16, 1>(48) = v194_acc;
              tensorforge::intel_esimd::simd<float, 16> v234_acc{};
              tensorforge::intel_esimd::simd<float, 16> v236_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v234_acc += ((static_cast<float>(v236_data[1])) * v37_data);
              v234_acc += ((static_cast<float>(v236_data[2])) * v39_data);
              v234_acc += ((static_cast<float>(v236_data[3])) * v41_data);
              v234_acc += ((static_cast<float>(v236_data[4])) * v43_data);
              v234_acc += ((static_cast<float>(v236_data[5])) * v45_data);
              v234_acc += ((static_cast<float>(v236_data[6])) * v47_data);
              v234_acc += ((static_cast<float>(v236_data[7])) * v49_data);
              v234_acc += ((static_cast<float>(v236_data[8])) * v51_data);
              v234_acc += ((static_cast<float>(v236_data[9])) * v53_data);
              v234_acc += ((static_cast<float>(v236_data[10])) * v55_data);
              v234_acc += ((static_cast<float>(v236_data[11])) * v57_data);
              v234_acc += ((static_cast<float>(v236_data[12])) * v59_data);
              v234_acc += ((static_cast<float>(v236_data[13])) * v61_data);
              v234_acc += ((static_cast<float>(v236_data[14])) * v63_data);
              v234_acc += ((static_cast<float>(v236_data[15])) * v65_data);
              tensorforge::intel_esimd::simd<float, 16> v269_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v234_acc += ((static_cast<float>(v269_data[0])) * v67_data);
              v234_acc += ((static_cast<float>(v269_data[1])) * v69_data);
              ir0.template select<16, 1>(64) = v234_acc;
              tensorforge::intel_esimd::simd<float, 16> v274_acc{};
              tensorforge::intel_esimd::simd<float, 16> v276_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v274_acc += ((static_cast<float>(v276_data[1])) * v37_data);
              v274_acc += ((static_cast<float>(v276_data[2])) * v39_data);
              v274_acc += ((static_cast<float>(v276_data[3])) * v41_data);
              v274_acc += ((static_cast<float>(v276_data[4])) * v43_data);
              v274_acc += ((static_cast<float>(v276_data[5])) * v45_data);
              v274_acc += ((static_cast<float>(v276_data[6])) * v47_data);
              v274_acc += ((static_cast<float>(v276_data[7])) * v49_data);
              v274_acc += ((static_cast<float>(v276_data[8])) * v51_data);
              v274_acc += ((static_cast<float>(v276_data[9])) * v53_data);
              v274_acc += ((static_cast<float>(v276_data[10])) * v55_data);
              v274_acc += ((static_cast<float>(v276_data[11])) * v57_data);
              v274_acc += ((static_cast<float>(v276_data[12])) * v59_data);
              v274_acc += ((static_cast<float>(v276_data[13])) * v61_data);
              v274_acc += ((static_cast<float>(v276_data[14])) * v63_data);
              v274_acc += ((static_cast<float>(v276_data[15])) * v65_data);
              tensorforge::intel_esimd::simd<float, 16> v309_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v274_acc += ((static_cast<float>(v309_data[0])) * v67_data);
              v274_acc += ((static_cast<float>(v309_data[1])) * v69_data);
              ir0.template select<16, 1>(80) = v274_acc;
              tensorforge::intel_esimd::simd<float, 16> v314_acc{};
              tensorforge::intel_esimd::simd<float, 16> v316_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v314_acc += ((static_cast<float>(v316_data[1])) * v37_data);
              v314_acc += ((static_cast<float>(v316_data[2])) * v39_data);
              v314_acc += ((static_cast<float>(v316_data[3])) * v41_data);
              v314_acc += ((static_cast<float>(v316_data[4])) * v43_data);
              v314_acc += ((static_cast<float>(v316_data[5])) * v45_data);
              v314_acc += ((static_cast<float>(v316_data[6])) * v47_data);
              v314_acc += ((static_cast<float>(v316_data[7])) * v49_data);
              v314_acc += ((static_cast<float>(v316_data[8])) * v51_data);
              v314_acc += ((static_cast<float>(v316_data[9])) * v53_data);
              v314_acc += ((static_cast<float>(v316_data[10])) * v55_data);
              v314_acc += ((static_cast<float>(v316_data[11])) * v57_data);
              v314_acc += ((static_cast<float>(v316_data[12])) * v59_data);
              v314_acc += ((static_cast<float>(v316_data[13])) * v61_data);
              v314_acc += ((static_cast<float>(v316_data[14])) * v63_data);
              v314_acc += ((static_cast<float>(v316_data[15])) * v65_data);
              tensorforge::intel_esimd::simd<float, 16> v349_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v314_acc += ((static_cast<float>(v349_data[0])) * v67_data);
              v314_acc += ((static_cast<float>(v349_data[1])) * v69_data);
              ir0.template select<16, 1>(96) = v314_acc;
              tensorforge::intel_esimd::simd<float, 16> v354_acc{};
              tensorforge::intel_esimd::simd<float, 16> v356_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v354_acc += ((static_cast<float>(v356_data[1])) * v37_data);
              v354_acc += ((static_cast<float>(v356_data[2])) * v39_data);
              v354_acc += ((static_cast<float>(v356_data[3])) * v41_data);
              v354_acc += ((static_cast<float>(v356_data[4])) * v43_data);
              v354_acc += ((static_cast<float>(v356_data[5])) * v45_data);
              v354_acc += ((static_cast<float>(v356_data[6])) * v47_data);
              v354_acc += ((static_cast<float>(v356_data[7])) * v49_data);
              v354_acc += ((static_cast<float>(v356_data[8])) * v51_data);
              v354_acc += ((static_cast<float>(v356_data[9])) * v53_data);
              v354_acc += ((static_cast<float>(v356_data[10])) * v55_data);
              v354_acc += ((static_cast<float>(v356_data[11])) * v57_data);
              v354_acc += ((static_cast<float>(v356_data[12])) * v59_data);
              v354_acc += ((static_cast<float>(v356_data[13])) * v61_data);
              v354_acc += ((static_cast<float>(v356_data[14])) * v63_data);
              v354_acc += ((static_cast<float>(v356_data[15])) * v65_data);
              tensorforge::intel_esimd::simd<float, 16> v389_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v354_acc += ((static_cast<float>(v389_data[0])) * v67_data);
              v354_acc += ((static_cast<float>(v389_data[1])) * v69_data);
              ir0.template select<16, 1>(112) = v354_acc;
              tensorforge::intel_esimd::simd<float, 16> v394_acc{};
              tensorforge::intel_esimd::simd<float, 16> v396_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v394_acc += ((static_cast<float>(v396_data[1])) * v37_data);
              v394_acc += ((static_cast<float>(v396_data[2])) * v39_data);
              v394_acc += ((static_cast<float>(v396_data[3])) * v41_data);
              v394_acc += ((static_cast<float>(v396_data[4])) * v43_data);
              v394_acc += ((static_cast<float>(v396_data[5])) * v45_data);
              v394_acc += ((static_cast<float>(v396_data[6])) * v47_data);
              v394_acc += ((static_cast<float>(v396_data[7])) * v49_data);
              v394_acc += ((static_cast<float>(v396_data[8])) * v51_data);
              v394_acc += ((static_cast<float>(v396_data[9])) * v53_data);
              v394_acc += ((static_cast<float>(v396_data[10])) * v55_data);
              v394_acc += ((static_cast<float>(v396_data[11])) * v57_data);
              v394_acc += ((static_cast<float>(v396_data[12])) * v59_data);
              v394_acc += ((static_cast<float>(v396_data[13])) * v61_data);
              v394_acc += ((static_cast<float>(v396_data[14])) * v63_data);
              v394_acc += ((static_cast<float>(v396_data[15])) * v65_data);
              tensorforge::intel_esimd::simd<float, 16> v429_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v394_acc += ((static_cast<float>(v429_data[0])) * v67_data);
              v394_acc += ((static_cast<float>(v429_data[1])) * v69_data);
              ir0.template select<16, 1>(128) = v394_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v434_n1 = 0; v434_n1 < 9; ++v434_n1) {
                int32_t v435_a = v434_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v437_data(ir0.template select<10, 1>(v435_a));
                r0.template select<10, 1>(v435_a) = v437_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v438_i1 = 0; v438_i1 < 9; ++v438_i1) {
                tensorforge::intel_esimd::simd<float, 10> v441_data(r0.template select<10, 1>((v438_i1 * 16)));
                v441_data.copy_to(glb_m0 + ((v438_i1 * 10)));
              }
            }
          }
        }
      }
    });
  });
}


// === base name ===
kernel_1d9fb35d12101f98

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1d9fb35d12101f98 = {{1, 32, 1}, 16, 16, 1, 32, 23616, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1d9fb35d12101f98(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1d9fb35d12101f98(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1d9fb35d12101f98(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_1d9fb35d12101f98(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1d9fb35d12101f98(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1d9fb35d12101f98(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1d9fb35d12101f98(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5904 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 32 per block = block 1x32x1, 23616 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×17) {0..16}×{0..17} none
        //   m2 20×9(17×9) {0..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":5904}],"shared_bytes":23616,"shared_elements":5904,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 272);
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
          if (item.get_local_id(1) == 16) {
            tensorforge::intel_esimd::simd<float, 16> v21_ld;
            v21_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v21_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v24_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v24_batchId0 < numElements0; v24_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v25_ahead1 = v24_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v27_batchId1 = (v25_ahead1 < numElements0) ? v25_ahead1 : v24_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v24_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v24_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v24_batchId0 * 153 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v34_ld);
              tensorforge::intel_esimd::simd<float, 64> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v35_ld);
              tensorforge::intel_esimd::simd<float, 16> v36_ld;
              v36_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v36_ld);
              tensorforge::intel_esimd::simd<float, 9> v37_ld;
              v37_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v37_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v43_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v45_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v47_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v49_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v53_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v55_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v57_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v61_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v63_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v65_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v69_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v71_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v73_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v75_data = tensorforge::slmLoad<float, 16>(glb_m1 + (256_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_acc{};
              tensorforge::intel_esimd::simd<float, 16> v77_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v76_acc += ((static_cast<float>(v77_data[0])) * v43_data);
              v76_acc += ((static_cast<float>(v77_data[1])) * v45_data);
              v76_acc += ((static_cast<float>(v77_data[2])) * v47_data);
              v76_acc += ((static_cast<float>(v77_data[3])) * v49_data);
              v76_acc += ((static_cast<float>(v77_data[4])) * v51_data);
              v76_acc += ((static_cast<float>(v77_data[5])) * v53_data);
              v76_acc += ((static_cast<float>(v77_data[6])) * v55_data);
              v76_acc += ((static_cast<float>(v77_data[7])) * v57_data);
              v76_acc += ((static_cast<float>(v77_data[8])) * v59_data);
              v76_acc += ((static_cast<float>(v77_data[9])) * v61_data);
              v76_acc += ((static_cast<float>(v77_data[10])) * v63_data);
              v76_acc += ((static_cast<float>(v77_data[11])) * v65_data);
              v76_acc += ((static_cast<float>(v77_data[12])) * v67_data);
              v76_acc += ((static_cast<float>(v77_data[13])) * v69_data);
              v76_acc += ((static_cast<float>(v77_data[14])) * v71_data);
              v76_acc += ((static_cast<float>(v77_data[15])) * v73_data);
              tensorforge::intel_esimd::simd<float, 16> v113_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v76_acc += ((static_cast<float>(v113_data[0])) * v75_data);
              ir0.template select<16, 1>(0) = v76_acc;
              tensorforge::intel_esimd::simd<float, 16> v116_acc{};
              tensorforge::intel_esimd::simd<float, 16> v118_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              v116_acc += ((static_cast<float>(v118_data[0])) * v43_data);
              v116_acc += ((static_cast<float>(v118_data[1])) * v45_data);
              v116_acc += ((static_cast<float>(v118_data[2])) * v47_data);
              v116_acc += ((static_cast<float>(v118_data[3])) * v49_data);
              v116_acc += ((static_cast<float>(v118_data[4])) * v51_data);
              v116_acc += ((static_cast<float>(v118_data[5])) * v53_data);
              v116_acc += ((static_cast<float>(v118_data[6])) * v55_data);
              v116_acc += ((static_cast<float>(v118_data[7])) * v57_data);
              v116_acc += ((static_cast<float>(v118_data[8])) * v59_data);
              v116_acc += ((static_cast<float>(v118_data[9])) * v61_data);
              v116_acc += ((static_cast<float>(v118_data[10])) * v63_data);
              v116_acc += ((static_cast<float>(v118_data[11])) * v65_data);
              v116_acc += ((static_cast<float>(v118_data[12])) * v67_data);
              v116_acc += ((static_cast<float>(v118_data[13])) * v69_data);
              v116_acc += ((static_cast<float>(v118_data[14])) * v71_data);
              v116_acc += ((static_cast<float>(v118_data[15])) * v73_data);
              tensorforge::intel_esimd::simd<float, 16> v152_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v116_acc += ((static_cast<float>(v152_data[0])) * v75_data);
              ir0.template select<16, 1>(16) = v116_acc;
              tensorforge::intel_esimd::simd<float, 16> v155_acc{};
              tensorforge::intel_esimd::simd<float, 16> v157_data = tensorforge::slmLoad<float, 16>(s0 + (34_i32));
              v155_acc += ((static_cast<float>(v157_data[0])) * v43_data);
              v155_acc += ((static_cast<float>(v157_data[1])) * v45_data);
              v155_acc += ((static_cast<float>(v157_data[2])) * v47_data);
              v155_acc += ((static_cast<float>(v157_data[3])) * v49_data);
              v155_acc += ((static_cast<float>(v157_data[4])) * v51_data);
              v155_acc += ((static_cast<float>(v157_data[5])) * v53_data);
              v155_acc += ((static_cast<float>(v157_data[6])) * v55_data);
              v155_acc += ((static_cast<float>(v157_data[7])) * v57_data);
              v155_acc += ((static_cast<float>(v157_data[8])) * v59_data);
              v155_acc += ((static_cast<float>(v157_data[9])) * v61_data);
              v155_acc += ((static_cast<float>(v157_data[10])) * v63_data);
              v155_acc += ((static_cast<float>(v157_data[11])) * v65_data);
              v155_acc += ((static_cast<float>(v157_data[12])) * v67_data);
              v155_acc += ((static_cast<float>(v157_data[13])) * v69_data);
              v155_acc += ((static_cast<float>(v157_data[14])) * v71_data);
              v155_acc += ((static_cast<float>(v157_data[15])) * v73_data);
              tensorforge::intel_esimd::simd<float, 16> v191_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v155_acc += ((static_cast<float>(v191_data[0])) * v75_data);
              ir0.template select<16, 1>(32) = v155_acc;
              tensorforge::intel_esimd::simd<float, 16> v194_acc{};
              tensorforge::intel_esimd::simd<float, 16> v196_data = tensorforge::slmLoad<float, 16>(s0 + (51_i32));
              v194_acc += ((static_cast<float>(v196_data[0])) * v43_data);
              v194_acc += ((static_cast<float>(v196_data[1])) * v45_data);
              v194_acc += ((static_cast<float>(v196_data[2])) * v47_data);
              v194_acc += ((static_cast<float>(v196_data[3])) * v49_data);
              v194_acc += ((static_cast<float>(v196_data[4])) * v51_data);
              v194_acc += ((static_cast<float>(v196_data[5])) * v53_data);
              v194_acc += ((static_cast<float>(v196_data[6])) * v55_data);
              v194_acc += ((static_cast<float>(v196_data[7])) * v57_data);
              v194_acc += ((static_cast<float>(v196_data[8])) * v59_data);
              v194_acc += ((static_cast<float>(v196_data[9])) * v61_data);
              v194_acc += ((static_cast<float>(v196_data[10])) * v63_data);
              v194_acc += ((static_cast<float>(v196_data[11])) * v65_data);
              v194_acc += ((static_cast<float>(v196_data[12])) * v67_data);
              v194_acc += ((static_cast<float>(v196_data[13])) * v69_data);
              v194_acc += ((static_cast<float>(v196_data[14])) * v71_data);
              v194_acc += ((static_cast<float>(v196_data[15])) * v73_data);
              tensorforge::intel_esimd::simd<float, 16> v230_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v194_acc += ((static_cast<float>(v230_data[0])) * v75_data);
              ir0.template select<16, 1>(48) = v194_acc;
              tensorforge::intel_esimd::simd<float, 16> v233_acc{};
              tensorforge::intel_esimd::simd<float, 16> v235_data = tensorforge::slmLoad<float, 16>(s0 + (68_i32));
              v233_acc += ((static_cast<float>(v235_data[0])) * v43_data);
              v233_acc += ((static_cast<float>(v235_data[1])) * v45_data);
              v233_acc += ((static_cast<float>(v235_data[2])) * v47_data);
              v233_acc += ((static_cast<float>(v235_data[3])) * v49_data);
              v233_acc += ((static_cast<float>(v235_data[4])) * v51_data);
              v233_acc += ((static_cast<float>(v235_data[5])) * v53_data);
              v233_acc += ((static_cast<float>(v235_data[6])) * v55_data);
              v233_acc += ((static_cast<float>(v235_data[7])) * v57_data);
              v233_acc += ((static_cast<float>(v235_data[8])) * v59_data);
              v233_acc += ((static_cast<float>(v235_data[9])) * v61_data);
              v233_acc += ((static_cast<float>(v235_data[10])) * v63_data);
              v233_acc += ((static_cast<float>(v235_data[11])) * v65_data);
              v233_acc += ((static_cast<float>(v235_data[12])) * v67_data);
              v233_acc += ((static_cast<float>(v235_data[13])) * v69_data);
              v233_acc += ((static_cast<float>(v235_data[14])) * v71_data);
              v233_acc += ((static_cast<float>(v235_data[15])) * v73_data);
              tensorforge::intel_esimd::simd<float, 16> v269_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v233_acc += ((static_cast<float>(v269_data[0])) * v75_data);
              ir0.template select<16, 1>(64) = v233_acc;
              tensorforge::intel_esimd::simd<float, 16> v272_acc{};
              tensorforge::intel_esimd::simd<float, 16> v274_data = tensorforge::slmLoad<float, 16>(s0 + (85_i32));
              v272_acc += ((static_cast<float>(v274_data[0])) * v43_data);
              v272_acc += ((static_cast<float>(v274_data[1])) * v45_data);
              v272_acc += ((static_cast<float>(v274_data[2])) * v47_data);
              v272_acc += ((static_cast<float>(v274_data[3])) * v49_data);
              v272_acc += ((static_cast<float>(v274_data[4])) * v51_data);
              v272_acc += ((static_cast<float>(v274_data[5])) * v53_data);
              v272_acc += ((static_cast<float>(v274_data[6])) * v55_data);
              v272_acc += ((static_cast<float>(v274_data[7])) * v57_data);
              v272_acc += ((static_cast<float>(v274_data[8])) * v59_data);
              v272_acc += ((static_cast<float>(v274_data[9])) * v61_data);
              v272_acc += ((static_cast<float>(v274_data[10])) * v63_data);
              v272_acc += ((static_cast<float>(v274_data[11])) * v65_data);
              v272_acc += ((static_cast<float>(v274_data[12])) * v67_data);
              v272_acc += ((static_cast<float>(v274_data[13])) * v69_data);
              v272_acc += ((static_cast<float>(v274_data[14])) * v71_data);
              v272_acc += ((static_cast<float>(v274_data[15])) * v73_data);
              tensorforge::intel_esimd::simd<float, 16> v308_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v272_acc += ((static_cast<float>(v308_data[0])) * v75_data);
              ir0.template select<16, 1>(80) = v272_acc;
              tensorforge::intel_esimd::simd<float, 16> v311_acc{};
              tensorforge::intel_esimd::simd<float, 16> v313_data = tensorforge::slmLoad<float, 16>(s0 + (102_i32));
              v311_acc += ((static_cast<float>(v313_data[0])) * v43_data);
              v311_acc += ((static_cast<float>(v313_data[1])) * v45_data);
              v311_acc += ((static_cast<float>(v313_data[2])) * v47_data);
              v311_acc += ((static_cast<float>(v313_data[3])) * v49_data);
              v311_acc += ((static_cast<float>(v313_data[4])) * v51_data);
              v311_acc += ((static_cast<float>(v313_data[5])) * v53_data);
              v311_acc += ((static_cast<float>(v313_data[6])) * v55_data);
              v311_acc += ((static_cast<float>(v313_data[7])) * v57_data);
              v311_acc += ((static_cast<float>(v313_data[8])) * v59_data);
              v311_acc += ((static_cast<float>(v313_data[9])) * v61_data);
              v311_acc += ((static_cast<float>(v313_data[10])) * v63_data);
              v311_acc += ((static_cast<float>(v313_data[11])) * v65_data);
              v311_acc += ((static_cast<float>(v313_data[12])) * v67_data);
              v311_acc += ((static_cast<float>(v313_data[13])) * v69_data);
              v311_acc += ((static_cast<float>(v313_data[14])) * v71_data);
              v311_acc += ((static_cast<float>(v313_data[15])) * v73_data);
              tensorforge::intel_esimd::simd<float, 16> v347_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v311_acc += ((static_cast<float>(v347_data[0])) * v75_data);
              ir0.template select<16, 1>(96) = v311_acc;
              tensorforge::intel_esimd::simd<float, 16> v350_acc{};
              tensorforge::intel_esimd::simd<float, 16> v352_data = tensorforge::slmLoad<float, 16>(s0 + (119_i32));
              v350_acc += ((static_cast<float>(v352_data[0])) * v43_data);
              v350_acc += ((static_cast<float>(v352_data[1])) * v45_data);
              v350_acc += ((static_cast<float>(v352_data[2])) * v47_data);
              v350_acc += ((static_cast<float>(v352_data[3])) * v49_data);
              v350_acc += ((static_cast<float>(v352_data[4])) * v51_data);
              v350_acc += ((static_cast<float>(v352_data[5])) * v53_data);
              v350_acc += ((static_cast<float>(v352_data[6])) * v55_data);
              v350_acc += ((static_cast<float>(v352_data[7])) * v57_data);
              v350_acc += ((static_cast<float>(v352_data[8])) * v59_data);
              v350_acc += ((static_cast<float>(v352_data[9])) * v61_data);
              v350_acc += ((static_cast<float>(v352_data[10])) * v63_data);
              v350_acc += ((static_cast<float>(v352_data[11])) * v65_data);
              v350_acc += ((static_cast<float>(v352_data[12])) * v67_data);
              v350_acc += ((static_cast<float>(v352_data[13])) * v69_data);
              v350_acc += ((static_cast<float>(v352_data[14])) * v71_data);
              v350_acc += ((static_cast<float>(v352_data[15])) * v73_data);
              tensorforge::intel_esimd::simd<float, 16> v386_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v350_acc += ((static_cast<float>(v386_data[0])) * v75_data);
              ir0.template select<16, 1>(112) = v350_acc;
              tensorforge::intel_esimd::simd<float, 16> v389_acc{};
              tensorforge::intel_esimd::simd<float, 16> v391_data = tensorforge::slmLoad<float, 16>(s0 + (136_i32));
              v389_acc += ((static_cast<float>(v391_data[0])) * v43_data);
              v389_acc += ((static_cast<float>(v391_data[1])) * v45_data);
              v389_acc += ((static_cast<float>(v391_data[2])) * v47_data);
              v389_acc += ((static_cast<float>(v391_data[3])) * v49_data);
              v389_acc += ((static_cast<float>(v391_data[4])) * v51_data);
              v389_acc += ((static_cast<float>(v391_data[5])) * v53_data);
              v389_acc += ((static_cast<float>(v391_data[6])) * v55_data);
              v389_acc += ((static_cast<float>(v391_data[7])) * v57_data);
              v389_acc += ((static_cast<float>(v391_data[8])) * v59_data);
              v389_acc += ((static_cast<float>(v391_data[9])) * v61_data);
              v389_acc += ((static_cast<float>(v391_data[10])) * v63_data);
              v389_acc += ((static_cast<float>(v391_data[11])) * v65_data);
              v389_acc += ((static_cast<float>(v391_data[12])) * v67_data);
              v389_acc += ((static_cast<float>(v391_data[13])) * v69_data);
              v389_acc += ((static_cast<float>(v391_data[14])) * v71_data);
              v389_acc += ((static_cast<float>(v391_data[15])) * v73_data);
              tensorforge::intel_esimd::simd<float, 16> v425_data = tensorforge::slmLoad<float, 16>(s0 + (152_i32));
              v389_acc += ((static_cast<float>(v425_data[0])) * v75_data);
              ir0.template select<16, 1>(128) = v389_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v428_n0 = 0; v428_n0 < 1; ++v428_n0) {
                int32_t v430_a = v428_n0 * 16;
                #pragma unroll
                for (int32_t v429_n1 = 0; v429_n1 < 9; ++v429_n1) {
                  int32_t v432_a = v430_a + (v429_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v433_data(ir0.template select<16, 1>(v432_a));
                  r0.template select<16, 1>(v432_a) = v433_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v434_i0 = 0; v434_i0 < 1; ++v434_i0) {
                int32_t v436_a = v434_i0 * 16;
                #pragma unroll
                for (int32_t v435_i1 = 0; v435_i1 < 9; ++v435_i1) {
                  int32_t v438_a = v436_a + (v435_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v439_data(r0.template select<16, 1>(v438_a));
                  v439_data.copy_to(glb_m0 + (v438_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


// === base name ===
kernel_a47608e9967753af

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a47608e9967753af = {{1, 32, 1}, 16, 16, 1, 32, 35840, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a47608e9967753af(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a47608e9967753af(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a47608e9967753af(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 8960 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a47608e9967753af(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a47608e9967753af(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a47608e9967753af(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a47608e9967753af(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<8960 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 32 per block = block 1x32x1, 35840 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} none
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":8960}],"shared_bytes":35840,"shared_elements":8960,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 256);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (256);
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
              float *const __restrict__ glb_m0 = &m0[v23_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v23_batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v33_ld);
              tensorforge::intel_esimd::simd<float, 64> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v34_ld);
              tensorforge::intel_esimd::simd<float, 64> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 128));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v35_ld);
              tensorforge::intel_esimd::simd<float, 64> v36_ld;
              v36_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 192));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v36_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v42_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v44_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v46_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v48_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v52_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v54_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v56_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v58_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v60_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v62_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v64_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v68_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v70_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v72_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v73_acc{};
              tensorforge::intel_esimd::simd<float, 16> v74_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v73_acc += ((static_cast<float>(v74_data[0])) * v42_data);
              v73_acc += ((static_cast<float>(v74_data[1])) * v44_data);
              v73_acc += ((static_cast<float>(v74_data[2])) * v46_data);
              v73_acc += ((static_cast<float>(v74_data[3])) * v48_data);
              v73_acc += ((static_cast<float>(v74_data[4])) * v50_data);
              v73_acc += ((static_cast<float>(v74_data[5])) * v52_data);
              v73_acc += ((static_cast<float>(v74_data[6])) * v54_data);
              v73_acc += ((static_cast<float>(v74_data[7])) * v56_data);
              v73_acc += ((static_cast<float>(v74_data[8])) * v58_data);
              v73_acc += ((static_cast<float>(v74_data[9])) * v60_data);
              v73_acc += ((static_cast<float>(v74_data[10])) * v62_data);
              v73_acc += ((static_cast<float>(v74_data[11])) * v64_data);
              v73_acc += ((static_cast<float>(v74_data[12])) * v66_data);
              v73_acc += ((static_cast<float>(v74_data[13])) * v68_data);
              v73_acc += ((static_cast<float>(v74_data[14])) * v70_data);
              v73_acc += ((static_cast<float>(v74_data[15])) * v72_data);
              ir0.template select<16, 1>(0) = v73_acc;
              tensorforge::intel_esimd::simd<float, 16> v107_acc{};
              tensorforge::intel_esimd::simd<float, 16> v108_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v107_acc += ((static_cast<float>(v108_data[0])) * v42_data);
              v107_acc += ((static_cast<float>(v108_data[1])) * v44_data);
              v107_acc += ((static_cast<float>(v108_data[2])) * v46_data);
              v107_acc += ((static_cast<float>(v108_data[3])) * v48_data);
              v107_acc += ((static_cast<float>(v108_data[4])) * v50_data);
              v107_acc += ((static_cast<float>(v108_data[5])) * v52_data);
              v107_acc += ((static_cast<float>(v108_data[6])) * v54_data);
              v107_acc += ((static_cast<float>(v108_data[7])) * v56_data);
              v107_acc += ((static_cast<float>(v108_data[8])) * v58_data);
              v107_acc += ((static_cast<float>(v108_data[9])) * v60_data);
              v107_acc += ((static_cast<float>(v108_data[10])) * v62_data);
              v107_acc += ((static_cast<float>(v108_data[11])) * v64_data);
              v107_acc += ((static_cast<float>(v108_data[12])) * v66_data);
              v107_acc += ((static_cast<float>(v108_data[13])) * v68_data);
              v107_acc += ((static_cast<float>(v108_data[14])) * v70_data);
              v107_acc += ((static_cast<float>(v108_data[15])) * v72_data);
              ir0.template select<16, 1>(16) = v107_acc;
              tensorforge::intel_esimd::simd<float, 16> v141_acc{};
              tensorforge::intel_esimd::simd<float, 16> v142_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v141_acc += ((static_cast<float>(v142_data[0])) * v42_data);
              v141_acc += ((static_cast<float>(v142_data[1])) * v44_data);
              v141_acc += ((static_cast<float>(v142_data[2])) * v46_data);
              v141_acc += ((static_cast<float>(v142_data[3])) * v48_data);
              v141_acc += ((static_cast<float>(v142_data[4])) * v50_data);
              v141_acc += ((static_cast<float>(v142_data[5])) * v52_data);
              v141_acc += ((static_cast<float>(v142_data[6])) * v54_data);
              v141_acc += ((static_cast<float>(v142_data[7])) * v56_data);
              v141_acc += ((static_cast<float>(v142_data[8])) * v58_data);
              v141_acc += ((static_cast<float>(v142_data[9])) * v60_data);
              v141_acc += ((static_cast<float>(v142_data[10])) * v62_data);
              v141_acc += ((static_cast<float>(v142_data[11])) * v64_data);
              v141_acc += ((static_cast<float>(v142_data[12])) * v66_data);
              v141_acc += ((static_cast<float>(v142_data[13])) * v68_data);
              v141_acc += ((static_cast<float>(v142_data[14])) * v70_data);
              v141_acc += ((static_cast<float>(v142_data[15])) * v72_data);
              ir0.template select<16, 1>(32) = v141_acc;
              tensorforge::intel_esimd::simd<float, 16> v175_acc{};
              tensorforge::intel_esimd::simd<float, 16> v176_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v175_acc += ((static_cast<float>(v176_data[0])) * v42_data);
              v175_acc += ((static_cast<float>(v176_data[1])) * v44_data);
              v175_acc += ((static_cast<float>(v176_data[2])) * v46_data);
              v175_acc += ((static_cast<float>(v176_data[3])) * v48_data);
              v175_acc += ((static_cast<float>(v176_data[4])) * v50_data);
              v175_acc += ((static_cast<float>(v176_data[5])) * v52_data);
              v175_acc += ((static_cast<float>(v176_data[6])) * v54_data);
              v175_acc += ((static_cast<float>(v176_data[7])) * v56_data);
              v175_acc += ((static_cast<float>(v176_data[8])) * v58_data);
              v175_acc += ((static_cast<float>(v176_data[9])) * v60_data);
              v175_acc += ((static_cast<float>(v176_data[10])) * v62_data);
              v175_acc += ((static_cast<float>(v176_data[11])) * v64_data);
              v175_acc += ((static_cast<float>(v176_data[12])) * v66_data);
              v175_acc += ((static_cast<float>(v176_data[13])) * v68_data);
              v175_acc += ((static_cast<float>(v176_data[14])) * v70_data);
              v175_acc += ((static_cast<float>(v176_data[15])) * v72_data);
              ir0.template select<16, 1>(48) = v175_acc;
              tensorforge::intel_esimd::simd<float, 16> v209_acc{};
              tensorforge::intel_esimd::simd<float, 16> v210_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v209_acc += ((static_cast<float>(v210_data[0])) * v42_data);
              v209_acc += ((static_cast<float>(v210_data[1])) * v44_data);
              v209_acc += ((static_cast<float>(v210_data[2])) * v46_data);
              v209_acc += ((static_cast<float>(v210_data[3])) * v48_data);
              v209_acc += ((static_cast<float>(v210_data[4])) * v50_data);
              v209_acc += ((static_cast<float>(v210_data[5])) * v52_data);
              v209_acc += ((static_cast<float>(v210_data[6])) * v54_data);
              v209_acc += ((static_cast<float>(v210_data[7])) * v56_data);
              v209_acc += ((static_cast<float>(v210_data[8])) * v58_data);
              v209_acc += ((static_cast<float>(v210_data[9])) * v60_data);
              v209_acc += ((static_cast<float>(v210_data[10])) * v62_data);
              v209_acc += ((static_cast<float>(v210_data[11])) * v64_data);
              v209_acc += ((static_cast<float>(v210_data[12])) * v66_data);
              v209_acc += ((static_cast<float>(v210_data[13])) * v68_data);
              v209_acc += ((static_cast<float>(v210_data[14])) * v70_data);
              v209_acc += ((static_cast<float>(v210_data[15])) * v72_data);
              ir0.template select<16, 1>(64) = v209_acc;
              tensorforge::intel_esimd::simd<float, 16> v243_acc{};
              tensorforge::intel_esimd::simd<float, 16> v244_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v243_acc += ((static_cast<float>(v244_data[0])) * v42_data);
              v243_acc += ((static_cast<float>(v244_data[1])) * v44_data);
              v243_acc += ((static_cast<float>(v244_data[2])) * v46_data);
              v243_acc += ((static_cast<float>(v244_data[3])) * v48_data);
              v243_acc += ((static_cast<float>(v244_data[4])) * v50_data);
              v243_acc += ((static_cast<float>(v244_data[5])) * v52_data);
              v243_acc += ((static_cast<float>(v244_data[6])) * v54_data);
              v243_acc += ((static_cast<float>(v244_data[7])) * v56_data);
              v243_acc += ((static_cast<float>(v244_data[8])) * v58_data);
              v243_acc += ((static_cast<float>(v244_data[9])) * v60_data);
              v243_acc += ((static_cast<float>(v244_data[10])) * v62_data);
              v243_acc += ((static_cast<float>(v244_data[11])) * v64_data);
              v243_acc += ((static_cast<float>(v244_data[12])) * v66_data);
              v243_acc += ((static_cast<float>(v244_data[13])) * v68_data);
              v243_acc += ((static_cast<float>(v244_data[14])) * v70_data);
              v243_acc += ((static_cast<float>(v244_data[15])) * v72_data);
              ir0.template select<16, 1>(80) = v243_acc;
              tensorforge::intel_esimd::simd<float, 16> v277_acc{};
              tensorforge::intel_esimd::simd<float, 16> v278_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v277_acc += ((static_cast<float>(v278_data[0])) * v42_data);
              v277_acc += ((static_cast<float>(v278_data[1])) * v44_data);
              v277_acc += ((static_cast<float>(v278_data[2])) * v46_data);
              v277_acc += ((static_cast<float>(v278_data[3])) * v48_data);
              v277_acc += ((static_cast<float>(v278_data[4])) * v50_data);
              v277_acc += ((static_cast<float>(v278_data[5])) * v52_data);
              v277_acc += ((static_cast<float>(v278_data[6])) * v54_data);
              v277_acc += ((static_cast<float>(v278_data[7])) * v56_data);
              v277_acc += ((static_cast<float>(v278_data[8])) * v58_data);
              v277_acc += ((static_cast<float>(v278_data[9])) * v60_data);
              v277_acc += ((static_cast<float>(v278_data[10])) * v62_data);
              v277_acc += ((static_cast<float>(v278_data[11])) * v64_data);
              v277_acc += ((static_cast<float>(v278_data[12])) * v66_data);
              v277_acc += ((static_cast<float>(v278_data[13])) * v68_data);
              v277_acc += ((static_cast<float>(v278_data[14])) * v70_data);
              v277_acc += ((static_cast<float>(v278_data[15])) * v72_data);
              ir0.template select<16, 1>(96) = v277_acc;
              tensorforge::intel_esimd::simd<float, 16> v311_acc{};
              tensorforge::intel_esimd::simd<float, 16> v312_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v311_acc += ((static_cast<float>(v312_data[0])) * v42_data);
              v311_acc += ((static_cast<float>(v312_data[1])) * v44_data);
              v311_acc += ((static_cast<float>(v312_data[2])) * v46_data);
              v311_acc += ((static_cast<float>(v312_data[3])) * v48_data);
              v311_acc += ((static_cast<float>(v312_data[4])) * v50_data);
              v311_acc += ((static_cast<float>(v312_data[5])) * v52_data);
              v311_acc += ((static_cast<float>(v312_data[6])) * v54_data);
              v311_acc += ((static_cast<float>(v312_data[7])) * v56_data);
              v311_acc += ((static_cast<float>(v312_data[8])) * v58_data);
              v311_acc += ((static_cast<float>(v312_data[9])) * v60_data);
              v311_acc += ((static_cast<float>(v312_data[10])) * v62_data);
              v311_acc += ((static_cast<float>(v312_data[11])) * v64_data);
              v311_acc += ((static_cast<float>(v312_data[12])) * v66_data);
              v311_acc += ((static_cast<float>(v312_data[13])) * v68_data);
              v311_acc += ((static_cast<float>(v312_data[14])) * v70_data);
              v311_acc += ((static_cast<float>(v312_data[15])) * v72_data);
              ir0.template select<16, 1>(112) = v311_acc;
              tensorforge::intel_esimd::simd<float, 16> v345_acc{};
              tensorforge::intel_esimd::simd<float, 16> v346_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v345_acc += ((static_cast<float>(v346_data[0])) * v42_data);
              v345_acc += ((static_cast<float>(v346_data[1])) * v44_data);
              v345_acc += ((static_cast<float>(v346_data[2])) * v46_data);
              v345_acc += ((static_cast<float>(v346_data[3])) * v48_data);
              v345_acc += ((static_cast<float>(v346_data[4])) * v50_data);
              v345_acc += ((static_cast<float>(v346_data[5])) * v52_data);
              v345_acc += ((static_cast<float>(v346_data[6])) * v54_data);
              v345_acc += ((static_cast<float>(v346_data[7])) * v56_data);
              v345_acc += ((static_cast<float>(v346_data[8])) * v58_data);
              v345_acc += ((static_cast<float>(v346_data[9])) * v60_data);
              v345_acc += ((static_cast<float>(v346_data[10])) * v62_data);
              v345_acc += ((static_cast<float>(v346_data[11])) * v64_data);
              v345_acc += ((static_cast<float>(v346_data[12])) * v66_data);
              v345_acc += ((static_cast<float>(v346_data[13])) * v68_data);
              v345_acc += ((static_cast<float>(v346_data[14])) * v70_data);
              v345_acc += ((static_cast<float>(v346_data[15])) * v72_data);
              ir0.template select<16, 1>(128) = v345_acc;
              tensorforge::intel_esimd::simd<float, 16> v379_acc{};
              tensorforge::intel_esimd::simd<float, 16> v380_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v379_acc += ((static_cast<float>(v380_data[0])) * v42_data);
              v379_acc += ((static_cast<float>(v380_data[1])) * v44_data);
              v379_acc += ((static_cast<float>(v380_data[2])) * v46_data);
              v379_acc += ((static_cast<float>(v380_data[3])) * v48_data);
              v379_acc += ((static_cast<float>(v380_data[4])) * v50_data);
              v379_acc += ((static_cast<float>(v380_data[5])) * v52_data);
              v379_acc += ((static_cast<float>(v380_data[6])) * v54_data);
              v379_acc += ((static_cast<float>(v380_data[7])) * v56_data);
              v379_acc += ((static_cast<float>(v380_data[8])) * v58_data);
              v379_acc += ((static_cast<float>(v380_data[9])) * v60_data);
              v379_acc += ((static_cast<float>(v380_data[10])) * v62_data);
              v379_acc += ((static_cast<float>(v380_data[11])) * v64_data);
              v379_acc += ((static_cast<float>(v380_data[12])) * v66_data);
              v379_acc += ((static_cast<float>(v380_data[13])) * v68_data);
              v379_acc += ((static_cast<float>(v380_data[14])) * v70_data);
              v379_acc += ((static_cast<float>(v380_data[15])) * v72_data);
              ir0.template select<16, 1>(144) = v379_acc;
              tensorforge::intel_esimd::simd<float, 16> v413_acc{};
              tensorforge::intel_esimd::simd<float, 16> v414_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v413_acc += ((static_cast<float>(v414_data[0])) * v42_data);
              v413_acc += ((static_cast<float>(v414_data[1])) * v44_data);
              v413_acc += ((static_cast<float>(v414_data[2])) * v46_data);
              v413_acc += ((static_cast<float>(v414_data[3])) * v48_data);
              v413_acc += ((static_cast<float>(v414_data[4])) * v50_data);
              v413_acc += ((static_cast<float>(v414_data[5])) * v52_data);
              v413_acc += ((static_cast<float>(v414_data[6])) * v54_data);
              v413_acc += ((static_cast<float>(v414_data[7])) * v56_data);
              v413_acc += ((static_cast<float>(v414_data[8])) * v58_data);
              v413_acc += ((static_cast<float>(v414_data[9])) * v60_data);
              v413_acc += ((static_cast<float>(v414_data[10])) * v62_data);
              v413_acc += ((static_cast<float>(v414_data[11])) * v64_data);
              v413_acc += ((static_cast<float>(v414_data[12])) * v66_data);
              v413_acc += ((static_cast<float>(v414_data[13])) * v68_data);
              v413_acc += ((static_cast<float>(v414_data[14])) * v70_data);
              v413_acc += ((static_cast<float>(v414_data[15])) * v72_data);
              ir0.template select<16, 1>(160) = v413_acc;
              tensorforge::intel_esimd::simd<float, 16> v447_acc{};
              tensorforge::intel_esimd::simd<float, 16> v448_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v447_acc += ((static_cast<float>(v448_data[0])) * v42_data);
              v447_acc += ((static_cast<float>(v448_data[1])) * v44_data);
              v447_acc += ((static_cast<float>(v448_data[2])) * v46_data);
              v447_acc += ((static_cast<float>(v448_data[3])) * v48_data);
              v447_acc += ((static_cast<float>(v448_data[4])) * v50_data);
              v447_acc += ((static_cast<float>(v448_data[5])) * v52_data);
              v447_acc += ((static_cast<float>(v448_data[6])) * v54_data);
              v447_acc += ((static_cast<float>(v448_data[7])) * v56_data);
              v447_acc += ((static_cast<float>(v448_data[8])) * v58_data);
              v447_acc += ((static_cast<float>(v448_data[9])) * v60_data);
              v447_acc += ((static_cast<float>(v448_data[10])) * v62_data);
              v447_acc += ((static_cast<float>(v448_data[11])) * v64_data);
              v447_acc += ((static_cast<float>(v448_data[12])) * v66_data);
              v447_acc += ((static_cast<float>(v448_data[13])) * v68_data);
              v447_acc += ((static_cast<float>(v448_data[14])) * v70_data);
              v447_acc += ((static_cast<float>(v448_data[15])) * v72_data);
              ir0.template select<16, 1>(176) = v447_acc;
              tensorforge::intel_esimd::simd<float, 16> v481_acc{};
              tensorforge::intel_esimd::simd<float, 16> v482_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v481_acc += ((static_cast<float>(v482_data[0])) * v42_data);
              v481_acc += ((static_cast<float>(v482_data[1])) * v44_data);
              v481_acc += ((static_cast<float>(v482_data[2])) * v46_data);
              v481_acc += ((static_cast<float>(v482_data[3])) * v48_data);
              v481_acc += ((static_cast<float>(v482_data[4])) * v50_data);
              v481_acc += ((static_cast<float>(v482_data[5])) * v52_data);
              v481_acc += ((static_cast<float>(v482_data[6])) * v54_data);
              v481_acc += ((static_cast<float>(v482_data[7])) * v56_data);
              v481_acc += ((static_cast<float>(v482_data[8])) * v58_data);
              v481_acc += ((static_cast<float>(v482_data[9])) * v60_data);
              v481_acc += ((static_cast<float>(v482_data[10])) * v62_data);
              v481_acc += ((static_cast<float>(v482_data[11])) * v64_data);
              v481_acc += ((static_cast<float>(v482_data[12])) * v66_data);
              v481_acc += ((static_cast<float>(v482_data[13])) * v68_data);
              v481_acc += ((static_cast<float>(v482_data[14])) * v70_data);
              v481_acc += ((static_cast<float>(v482_data[15])) * v72_data);
              ir0.template select<16, 1>(192) = v481_acc;
              tensorforge::intel_esimd::simd<float, 16> v515_acc{};
              tensorforge::intel_esimd::simd<float, 16> v516_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v515_acc += ((static_cast<float>(v516_data[0])) * v42_data);
              v515_acc += ((static_cast<float>(v516_data[1])) * v44_data);
              v515_acc += ((static_cast<float>(v516_data[2])) * v46_data);
              v515_acc += ((static_cast<float>(v516_data[3])) * v48_data);
              v515_acc += ((static_cast<float>(v516_data[4])) * v50_data);
              v515_acc += ((static_cast<float>(v516_data[5])) * v52_data);
              v515_acc += ((static_cast<float>(v516_data[6])) * v54_data);
              v515_acc += ((static_cast<float>(v516_data[7])) * v56_data);
              v515_acc += ((static_cast<float>(v516_data[8])) * v58_data);
              v515_acc += ((static_cast<float>(v516_data[9])) * v60_data);
              v515_acc += ((static_cast<float>(v516_data[10])) * v62_data);
              v515_acc += ((static_cast<float>(v516_data[11])) * v64_data);
              v515_acc += ((static_cast<float>(v516_data[12])) * v66_data);
              v515_acc += ((static_cast<float>(v516_data[13])) * v68_data);
              v515_acc += ((static_cast<float>(v516_data[14])) * v70_data);
              v515_acc += ((static_cast<float>(v516_data[15])) * v72_data);
              ir0.template select<16, 1>(208) = v515_acc;
              tensorforge::intel_esimd::simd<float, 16> v549_acc{};
              tensorforge::intel_esimd::simd<float, 16> v550_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v549_acc += ((static_cast<float>(v550_data[0])) * v42_data);
              v549_acc += ((static_cast<float>(v550_data[1])) * v44_data);
              v549_acc += ((static_cast<float>(v550_data[2])) * v46_data);
              v549_acc += ((static_cast<float>(v550_data[3])) * v48_data);
              v549_acc += ((static_cast<float>(v550_data[4])) * v50_data);
              v549_acc += ((static_cast<float>(v550_data[5])) * v52_data);
              v549_acc += ((static_cast<float>(v550_data[6])) * v54_data);
              v549_acc += ((static_cast<float>(v550_data[7])) * v56_data);
              v549_acc += ((static_cast<float>(v550_data[8])) * v58_data);
              v549_acc += ((static_cast<float>(v550_data[9])) * v60_data);
              v549_acc += ((static_cast<float>(v550_data[10])) * v62_data);
              v549_acc += ((static_cast<float>(v550_data[11])) * v64_data);
              v549_acc += ((static_cast<float>(v550_data[12])) * v66_data);
              v549_acc += ((static_cast<float>(v550_data[13])) * v68_data);
              v549_acc += ((static_cast<float>(v550_data[14])) * v70_data);
              v549_acc += ((static_cast<float>(v550_data[15])) * v72_data);
              ir0.template select<16, 1>(224) = v549_acc;
              tensorforge::intel_esimd::simd<float, 16> v583_acc{};
              tensorforge::intel_esimd::simd<float, 16> v584_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v583_acc += ((static_cast<float>(v584_data[0])) * v42_data);
              v583_acc += ((static_cast<float>(v584_data[1])) * v44_data);
              v583_acc += ((static_cast<float>(v584_data[2])) * v46_data);
              v583_acc += ((static_cast<float>(v584_data[3])) * v48_data);
              v583_acc += ((static_cast<float>(v584_data[4])) * v50_data);
              v583_acc += ((static_cast<float>(v584_data[5])) * v52_data);
              v583_acc += ((static_cast<float>(v584_data[6])) * v54_data);
              v583_acc += ((static_cast<float>(v584_data[7])) * v56_data);
              v583_acc += ((static_cast<float>(v584_data[8])) * v58_data);
              v583_acc += ((static_cast<float>(v584_data[9])) * v60_data);
              v583_acc += ((static_cast<float>(v584_data[10])) * v62_data);
              v583_acc += ((static_cast<float>(v584_data[11])) * v64_data);
              v583_acc += ((static_cast<float>(v584_data[12])) * v66_data);
              v583_acc += ((static_cast<float>(v584_data[13])) * v68_data);
              v583_acc += ((static_cast<float>(v584_data[14])) * v70_data);
              v583_acc += ((static_cast<float>(v584_data[15])) * v72_data);
              ir0.template select<16, 1>(240) = v583_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v617_n0 = 0; v617_n0 < 1; ++v617_n0) {
                int32_t v619_a = v617_n0 * 16;
                #pragma unroll
                for (int32_t v618_n1 = 0; v618_n1 < 16; ++v618_n1) {
                  int32_t v621_a = v619_a + (v618_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v622_data(ir0.template select<16, 1>(v621_a));
                  r0.template select<16, 1>(v621_a) = v622_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v623_i0 = 0; v623_i0 < 1; ++v623_i0) {
                int32_t v625_a = v623_i0 * 16;
                #pragma unroll
                for (int32_t v624_i1 = 0; v624_i1 < 16; ++v624_i1) {
                  int32_t v627_a = v625_a + (v624_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v628_data(r0.template select<16, 1>(v627_a));
                  v628_data.copy_to(glb_m0 + (v627_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


// === base name ===
kernel_3d6c8602fe70b5e0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3d6c8602fe70b5e0 = {{1, 32, 1}, 16, 16, 1, 32, 35840, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3d6c8602fe70b5e0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3d6c8602fe70b5e0(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3d6c8602fe70b5e0(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_3d6c8602fe70b5e0(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3d6c8602fe70b5e0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3d6c8602fe70b5e0(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3d6c8602fe70b5e0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<8960 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 32 per block = block 1x32x1, 35840 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} none
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":8960}],"shared_bytes":35840,"shared_elements":8960,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 256);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (256);
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
              float *const __restrict__ glb_m0 = &m0[v29_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v29_batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v39_ld;
              v39_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v39_ld);
              tensorforge::intel_esimd::simd<float, 64> v40_ld;
              v40_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v40_ld);
              tensorforge::intel_esimd::simd<float, 64> v41_ld;
              v41_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 128));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v41_ld);
              tensorforge::intel_esimd::simd<float, 64> v42_ld;
              v42_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 192));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v42_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v48_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v50_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v52_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v54_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v58_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v60_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v62_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v66_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v68_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v70_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v74_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v76_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v78_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v79_acc{};
              tensorforge::intel_esimd::simd<float, 16> v80_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v79_acc += ((static_cast<float>(v80_data[0])) * v48_data);
              v79_acc += ((static_cast<float>(v80_data[1])) * v50_data);
              v79_acc += ((static_cast<float>(v80_data[2])) * v52_data);
              v79_acc += ((static_cast<float>(v80_data[3])) * v54_data);
              v79_acc += ((static_cast<float>(v80_data[4])) * v56_data);
              v79_acc += ((static_cast<float>(v80_data[5])) * v58_data);
              v79_acc += ((static_cast<float>(v80_data[6])) * v60_data);
              v79_acc += ((static_cast<float>(v80_data[7])) * v62_data);
              v79_acc += ((static_cast<float>(v80_data[8])) * v64_data);
              v79_acc += ((static_cast<float>(v80_data[9])) * v66_data);
              v79_acc += ((static_cast<float>(v80_data[10])) * v68_data);
              v79_acc += ((static_cast<float>(v80_data[11])) * v70_data);
              v79_acc += ((static_cast<float>(v80_data[12])) * v72_data);
              v79_acc += ((static_cast<float>(v80_data[13])) * v74_data);
              v79_acc += ((static_cast<float>(v80_data[14])) * v76_data);
              v79_acc += ((static_cast<float>(v80_data[15])) * v78_data);
              ir0.template select<16, 1>(0) = v79_acc;
              tensorforge::intel_esimd::simd<float, 16> v113_acc{};
              tensorforge::intel_esimd::simd<float, 16> v114_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v113_acc += ((static_cast<float>(v114_data[0])) * v48_data);
              v113_acc += ((static_cast<float>(v114_data[1])) * v50_data);
              v113_acc += ((static_cast<float>(v114_data[2])) * v52_data);
              v113_acc += ((static_cast<float>(v114_data[3])) * v54_data);
              v113_acc += ((static_cast<float>(v114_data[4])) * v56_data);
              v113_acc += ((static_cast<float>(v114_data[5])) * v58_data);
              v113_acc += ((static_cast<float>(v114_data[6])) * v60_data);
              v113_acc += ((static_cast<float>(v114_data[7])) * v62_data);
              v113_acc += ((static_cast<float>(v114_data[8])) * v64_data);
              v113_acc += ((static_cast<float>(v114_data[9])) * v66_data);
              v113_acc += ((static_cast<float>(v114_data[10])) * v68_data);
              v113_acc += ((static_cast<float>(v114_data[11])) * v70_data);
              v113_acc += ((static_cast<float>(v114_data[12])) * v72_data);
              v113_acc += ((static_cast<float>(v114_data[13])) * v74_data);
              v113_acc += ((static_cast<float>(v114_data[14])) * v76_data);
              v113_acc += ((static_cast<float>(v114_data[15])) * v78_data);
              ir0.template select<16, 1>(16) = v113_acc;
              tensorforge::intel_esimd::simd<float, 16> v147_acc{};
              tensorforge::intel_esimd::simd<float, 16> v148_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v147_acc += ((static_cast<float>(v148_data[0])) * v48_data);
              v147_acc += ((static_cast<float>(v148_data[1])) * v50_data);
              v147_acc += ((static_cast<float>(v148_data[2])) * v52_data);
              v147_acc += ((static_cast<float>(v148_data[3])) * v54_data);
              v147_acc += ((static_cast<float>(v148_data[4])) * v56_data);
              v147_acc += ((static_cast<float>(v148_data[5])) * v58_data);
              v147_acc += ((static_cast<float>(v148_data[6])) * v60_data);
              v147_acc += ((static_cast<float>(v148_data[7])) * v62_data);
              v147_acc += ((static_cast<float>(v148_data[8])) * v64_data);
              v147_acc += ((static_cast<float>(v148_data[9])) * v66_data);
              v147_acc += ((static_cast<float>(v148_data[10])) * v68_data);
              v147_acc += ((static_cast<float>(v148_data[11])) * v70_data);
              v147_acc += ((static_cast<float>(v148_data[12])) * v72_data);
              v147_acc += ((static_cast<float>(v148_data[13])) * v74_data);
              v147_acc += ((static_cast<float>(v148_data[14])) * v76_data);
              v147_acc += ((static_cast<float>(v148_data[15])) * v78_data);
              ir0.template select<16, 1>(32) = v147_acc;
              tensorforge::intel_esimd::simd<float, 16> v181_acc{};
              tensorforge::intel_esimd::simd<float, 16> v182_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v181_acc += ((static_cast<float>(v182_data[0])) * v48_data);
              v181_acc += ((static_cast<float>(v182_data[1])) * v50_data);
              v181_acc += ((static_cast<float>(v182_data[2])) * v52_data);
              v181_acc += ((static_cast<float>(v182_data[3])) * v54_data);
              v181_acc += ((static_cast<float>(v182_data[4])) * v56_data);
              v181_acc += ((static_cast<float>(v182_data[5])) * v58_data);
              v181_acc += ((static_cast<float>(v182_data[6])) * v60_data);
              v181_acc += ((static_cast<float>(v182_data[7])) * v62_data);
              v181_acc += ((static_cast<float>(v182_data[8])) * v64_data);
              v181_acc += ((static_cast<float>(v182_data[9])) * v66_data);
              v181_acc += ((static_cast<float>(v182_data[10])) * v68_data);
              v181_acc += ((static_cast<float>(v182_data[11])) * v70_data);
              v181_acc += ((static_cast<float>(v182_data[12])) * v72_data);
              v181_acc += ((static_cast<float>(v182_data[13])) * v74_data);
              v181_acc += ((static_cast<float>(v182_data[14])) * v76_data);
              v181_acc += ((static_cast<float>(v182_data[15])) * v78_data);
              ir0.template select<16, 1>(48) = v181_acc;
              tensorforge::intel_esimd::simd<float, 16> v215_acc{};
              tensorforge::intel_esimd::simd<float, 16> v216_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v215_acc += ((static_cast<float>(v216_data[0])) * v48_data);
              v215_acc += ((static_cast<float>(v216_data[1])) * v50_data);
              v215_acc += ((static_cast<float>(v216_data[2])) * v52_data);
              v215_acc += ((static_cast<float>(v216_data[3])) * v54_data);
              v215_acc += ((static_cast<float>(v216_data[4])) * v56_data);
              v215_acc += ((static_cast<float>(v216_data[5])) * v58_data);
              v215_acc += ((static_cast<float>(v216_data[6])) * v60_data);
              v215_acc += ((static_cast<float>(v216_data[7])) * v62_data);
              v215_acc += ((static_cast<float>(v216_data[8])) * v64_data);
              v215_acc += ((static_cast<float>(v216_data[9])) * v66_data);
              v215_acc += ((static_cast<float>(v216_data[10])) * v68_data);
              v215_acc += ((static_cast<float>(v216_data[11])) * v70_data);
              v215_acc += ((static_cast<float>(v216_data[12])) * v72_data);
              v215_acc += ((static_cast<float>(v216_data[13])) * v74_data);
              v215_acc += ((static_cast<float>(v216_data[14])) * v76_data);
              v215_acc += ((static_cast<float>(v216_data[15])) * v78_data);
              ir0.template select<16, 1>(64) = v215_acc;
              tensorforge::intel_esimd::simd<float, 16> v249_acc{};
              tensorforge::intel_esimd::simd<float, 16> v250_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v249_acc += ((static_cast<float>(v250_data[0])) * v48_data);
              v249_acc += ((static_cast<float>(v250_data[1])) * v50_data);
              v249_acc += ((static_cast<float>(v250_data[2])) * v52_data);
              v249_acc += ((static_cast<float>(v250_data[3])) * v54_data);
              v249_acc += ((static_cast<float>(v250_data[4])) * v56_data);
              v249_acc += ((static_cast<float>(v250_data[5])) * v58_data);
              v249_acc += ((static_cast<float>(v250_data[6])) * v60_data);
              v249_acc += ((static_cast<float>(v250_data[7])) * v62_data);
              v249_acc += ((static_cast<float>(v250_data[8])) * v64_data);
              v249_acc += ((static_cast<float>(v250_data[9])) * v66_data);
              v249_acc += ((static_cast<float>(v250_data[10])) * v68_data);
              v249_acc += ((static_cast<float>(v250_data[11])) * v70_data);
              v249_acc += ((static_cast<float>(v250_data[12])) * v72_data);
              v249_acc += ((static_cast<float>(v250_data[13])) * v74_data);
              v249_acc += ((static_cast<float>(v250_data[14])) * v76_data);
              v249_acc += ((static_cast<float>(v250_data[15])) * v78_data);
              ir0.template select<16, 1>(80) = v249_acc;
              tensorforge::intel_esimd::simd<float, 16> v283_acc{};
              tensorforge::intel_esimd::simd<float, 16> v284_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v283_acc += ((static_cast<float>(v284_data[0])) * v48_data);
              v283_acc += ((static_cast<float>(v284_data[1])) * v50_data);
              v283_acc += ((static_cast<float>(v284_data[2])) * v52_data);
              v283_acc += ((static_cast<float>(v284_data[3])) * v54_data);
              v283_acc += ((static_cast<float>(v284_data[4])) * v56_data);
              v283_acc += ((static_cast<float>(v284_data[5])) * v58_data);
              v283_acc += ((static_cast<float>(v284_data[6])) * v60_data);
              v283_acc += ((static_cast<float>(v284_data[7])) * v62_data);
              v283_acc += ((static_cast<float>(v284_data[8])) * v64_data);
              v283_acc += ((static_cast<float>(v284_data[9])) * v66_data);
              v283_acc += ((static_cast<float>(v284_data[10])) * v68_data);
              v283_acc += ((static_cast<float>(v284_data[11])) * v70_data);
              v283_acc += ((static_cast<float>(v284_data[12])) * v72_data);
              v283_acc += ((static_cast<float>(v284_data[13])) * v74_data);
              v283_acc += ((static_cast<float>(v284_data[14])) * v76_data);
              v283_acc += ((static_cast<float>(v284_data[15])) * v78_data);
              ir0.template select<16, 1>(96) = v283_acc;
              tensorforge::intel_esimd::simd<float, 16> v317_acc{};
              tensorforge::intel_esimd::simd<float, 16> v318_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v317_acc += ((static_cast<float>(v318_data[0])) * v48_data);
              v317_acc += ((static_cast<float>(v318_data[1])) * v50_data);
              v317_acc += ((static_cast<float>(v318_data[2])) * v52_data);
              v317_acc += ((static_cast<float>(v318_data[3])) * v54_data);
              v317_acc += ((static_cast<float>(v318_data[4])) * v56_data);
              v317_acc += ((static_cast<float>(v318_data[5])) * v58_data);
              v317_acc += ((static_cast<float>(v318_data[6])) * v60_data);
              v317_acc += ((static_cast<float>(v318_data[7])) * v62_data);
              v317_acc += ((static_cast<float>(v318_data[8])) * v64_data);
              v317_acc += ((static_cast<float>(v318_data[9])) * v66_data);
              v317_acc += ((static_cast<float>(v318_data[10])) * v68_data);
              v317_acc += ((static_cast<float>(v318_data[11])) * v70_data);
              v317_acc += ((static_cast<float>(v318_data[12])) * v72_data);
              v317_acc += ((static_cast<float>(v318_data[13])) * v74_data);
              v317_acc += ((static_cast<float>(v318_data[14])) * v76_data);
              v317_acc += ((static_cast<float>(v318_data[15])) * v78_data);
              ir0.template select<16, 1>(112) = v317_acc;
              tensorforge::intel_esimd::simd<float, 16> v351_acc{};
              tensorforge::intel_esimd::simd<float, 16> v352_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v351_acc += ((static_cast<float>(v352_data[0])) * v48_data);
              v351_acc += ((static_cast<float>(v352_data[1])) * v50_data);
              v351_acc += ((static_cast<float>(v352_data[2])) * v52_data);
              v351_acc += ((static_cast<float>(v352_data[3])) * v54_data);
              v351_acc += ((static_cast<float>(v352_data[4])) * v56_data);
              v351_acc += ((static_cast<float>(v352_data[5])) * v58_data);
              v351_acc += ((static_cast<float>(v352_data[6])) * v60_data);
              v351_acc += ((static_cast<float>(v352_data[7])) * v62_data);
              v351_acc += ((static_cast<float>(v352_data[8])) * v64_data);
              v351_acc += ((static_cast<float>(v352_data[9])) * v66_data);
              v351_acc += ((static_cast<float>(v352_data[10])) * v68_data);
              v351_acc += ((static_cast<float>(v352_data[11])) * v70_data);
              v351_acc += ((static_cast<float>(v352_data[12])) * v72_data);
              v351_acc += ((static_cast<float>(v352_data[13])) * v74_data);
              v351_acc += ((static_cast<float>(v352_data[14])) * v76_data);
              v351_acc += ((static_cast<float>(v352_data[15])) * v78_data);
              ir0.template select<16, 1>(128) = v351_acc;
              tensorforge::intel_esimd::simd<float, 16> v385_acc{};
              tensorforge::intel_esimd::simd<float, 16> v386_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v385_acc += ((static_cast<float>(v386_data[0])) * v48_data);
              v385_acc += ((static_cast<float>(v386_data[1])) * v50_data);
              v385_acc += ((static_cast<float>(v386_data[2])) * v52_data);
              v385_acc += ((static_cast<float>(v386_data[3])) * v54_data);
              v385_acc += ((static_cast<float>(v386_data[4])) * v56_data);
              v385_acc += ((static_cast<float>(v386_data[5])) * v58_data);
              v385_acc += ((static_cast<float>(v386_data[6])) * v60_data);
              v385_acc += ((static_cast<float>(v386_data[7])) * v62_data);
              v385_acc += ((static_cast<float>(v386_data[8])) * v64_data);
              v385_acc += ((static_cast<float>(v386_data[9])) * v66_data);
              v385_acc += ((static_cast<float>(v386_data[10])) * v68_data);
              v385_acc += ((static_cast<float>(v386_data[11])) * v70_data);
              v385_acc += ((static_cast<float>(v386_data[12])) * v72_data);
              v385_acc += ((static_cast<float>(v386_data[13])) * v74_data);
              v385_acc += ((static_cast<float>(v386_data[14])) * v76_data);
              v385_acc += ((static_cast<float>(v386_data[15])) * v78_data);
              ir0.template select<16, 1>(144) = v385_acc;
              tensorforge::intel_esimd::simd<float, 16> v419_acc{};
              tensorforge::intel_esimd::simd<float, 16> v420_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v419_acc += ((static_cast<float>(v420_data[0])) * v48_data);
              v419_acc += ((static_cast<float>(v420_data[1])) * v50_data);
              v419_acc += ((static_cast<float>(v420_data[2])) * v52_data);
              v419_acc += ((static_cast<float>(v420_data[3])) * v54_data);
              v419_acc += ((static_cast<float>(v420_data[4])) * v56_data);
              v419_acc += ((static_cast<float>(v420_data[5])) * v58_data);
              v419_acc += ((static_cast<float>(v420_data[6])) * v60_data);
              v419_acc += ((static_cast<float>(v420_data[7])) * v62_data);
              v419_acc += ((static_cast<float>(v420_data[8])) * v64_data);
              v419_acc += ((static_cast<float>(v420_data[9])) * v66_data);
              v419_acc += ((static_cast<float>(v420_data[10])) * v68_data);
              v419_acc += ((static_cast<float>(v420_data[11])) * v70_data);
              v419_acc += ((static_cast<float>(v420_data[12])) * v72_data);
              v419_acc += ((static_cast<float>(v420_data[13])) * v74_data);
              v419_acc += ((static_cast<float>(v420_data[14])) * v76_data);
              v419_acc += ((static_cast<float>(v420_data[15])) * v78_data);
              ir0.template select<16, 1>(160) = v419_acc;
              tensorforge::intel_esimd::simd<float, 16> v453_acc{};
              tensorforge::intel_esimd::simd<float, 16> v454_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v453_acc += ((static_cast<float>(v454_data[0])) * v48_data);
              v453_acc += ((static_cast<float>(v454_data[1])) * v50_data);
              v453_acc += ((static_cast<float>(v454_data[2])) * v52_data);
              v453_acc += ((static_cast<float>(v454_data[3])) * v54_data);
              v453_acc += ((static_cast<float>(v454_data[4])) * v56_data);
              v453_acc += ((static_cast<float>(v454_data[5])) * v58_data);
              v453_acc += ((static_cast<float>(v454_data[6])) * v60_data);
              v453_acc += ((static_cast<float>(v454_data[7])) * v62_data);
              v453_acc += ((static_cast<float>(v454_data[8])) * v64_data);
              v453_acc += ((static_cast<float>(v454_data[9])) * v66_data);
              v453_acc += ((static_cast<float>(v454_data[10])) * v68_data);
              v453_acc += ((static_cast<float>(v454_data[11])) * v70_data);
              v453_acc += ((static_cast<float>(v454_data[12])) * v72_data);
              v453_acc += ((static_cast<float>(v454_data[13])) * v74_data);
              v453_acc += ((static_cast<float>(v454_data[14])) * v76_data);
              v453_acc += ((static_cast<float>(v454_data[15])) * v78_data);
              ir0.template select<16, 1>(176) = v453_acc;
              tensorforge::intel_esimd::simd<float, 16> v487_acc{};
              tensorforge::intel_esimd::simd<float, 16> v488_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v487_acc += ((static_cast<float>(v488_data[0])) * v48_data);
              v487_acc += ((static_cast<float>(v488_data[1])) * v50_data);
              v487_acc += ((static_cast<float>(v488_data[2])) * v52_data);
              v487_acc += ((static_cast<float>(v488_data[3])) * v54_data);
              v487_acc += ((static_cast<float>(v488_data[4])) * v56_data);
              v487_acc += ((static_cast<float>(v488_data[5])) * v58_data);
              v487_acc += ((static_cast<float>(v488_data[6])) * v60_data);
              v487_acc += ((static_cast<float>(v488_data[7])) * v62_data);
              v487_acc += ((static_cast<float>(v488_data[8])) * v64_data);
              v487_acc += ((static_cast<float>(v488_data[9])) * v66_data);
              v487_acc += ((static_cast<float>(v488_data[10])) * v68_data);
              v487_acc += ((static_cast<float>(v488_data[11])) * v70_data);
              v487_acc += ((static_cast<float>(v488_data[12])) * v72_data);
              v487_acc += ((static_cast<float>(v488_data[13])) * v74_data);
              v487_acc += ((static_cast<float>(v488_data[14])) * v76_data);
              v487_acc += ((static_cast<float>(v488_data[15])) * v78_data);
              ir0.template select<16, 1>(192) = v487_acc;
              tensorforge::intel_esimd::simd<float, 16> v521_acc{};
              tensorforge::intel_esimd::simd<float, 16> v522_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v521_acc += ((static_cast<float>(v522_data[0])) * v48_data);
              v521_acc += ((static_cast<float>(v522_data[1])) * v50_data);
              v521_acc += ((static_cast<float>(v522_data[2])) * v52_data);
              v521_acc += ((static_cast<float>(v522_data[3])) * v54_data);
              v521_acc += ((static_cast<float>(v522_data[4])) * v56_data);
              v521_acc += ((static_cast<float>(v522_data[5])) * v58_data);
              v521_acc += ((static_cast<float>(v522_data[6])) * v60_data);
              v521_acc += ((static_cast<float>(v522_data[7])) * v62_data);
              v521_acc += ((static_cast<float>(v522_data[8])) * v64_data);
              v521_acc += ((static_cast<float>(v522_data[9])) * v66_data);
              v521_acc += ((static_cast<float>(v522_data[10])) * v68_data);
              v521_acc += ((static_cast<float>(v522_data[11])) * v70_data);
              v521_acc += ((static_cast<float>(v522_data[12])) * v72_data);
              v521_acc += ((static_cast<float>(v522_data[13])) * v74_data);
              v521_acc += ((static_cast<float>(v522_data[14])) * v76_data);
              v521_acc += ((static_cast<float>(v522_data[15])) * v78_data);
              ir0.template select<16, 1>(208) = v521_acc;
              tensorforge::intel_esimd::simd<float, 16> v555_acc{};
              tensorforge::intel_esimd::simd<float, 16> v556_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v555_acc += ((static_cast<float>(v556_data[0])) * v48_data);
              v555_acc += ((static_cast<float>(v556_data[1])) * v50_data);
              v555_acc += ((static_cast<float>(v556_data[2])) * v52_data);
              v555_acc += ((static_cast<float>(v556_data[3])) * v54_data);
              v555_acc += ((static_cast<float>(v556_data[4])) * v56_data);
              v555_acc += ((static_cast<float>(v556_data[5])) * v58_data);
              v555_acc += ((static_cast<float>(v556_data[6])) * v60_data);
              v555_acc += ((static_cast<float>(v556_data[7])) * v62_data);
              v555_acc += ((static_cast<float>(v556_data[8])) * v64_data);
              v555_acc += ((static_cast<float>(v556_data[9])) * v66_data);
              v555_acc += ((static_cast<float>(v556_data[10])) * v68_data);
              v555_acc += ((static_cast<float>(v556_data[11])) * v70_data);
              v555_acc += ((static_cast<float>(v556_data[12])) * v72_data);
              v555_acc += ((static_cast<float>(v556_data[13])) * v74_data);
              v555_acc += ((static_cast<float>(v556_data[14])) * v76_data);
              v555_acc += ((static_cast<float>(v556_data[15])) * v78_data);
              ir0.template select<16, 1>(224) = v555_acc;
              tensorforge::intel_esimd::simd<float, 16> v589_acc{};
              tensorforge::intel_esimd::simd<float, 16> v590_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v589_acc += ((static_cast<float>(v590_data[0])) * v48_data);
              v589_acc += ((static_cast<float>(v590_data[1])) * v50_data);
              v589_acc += ((static_cast<float>(v590_data[2])) * v52_data);
              v589_acc += ((static_cast<float>(v590_data[3])) * v54_data);
              v589_acc += ((static_cast<float>(v590_data[4])) * v56_data);
              v589_acc += ((static_cast<float>(v590_data[5])) * v58_data);
              v589_acc += ((static_cast<float>(v590_data[6])) * v60_data);
              v589_acc += ((static_cast<float>(v590_data[7])) * v62_data);
              v589_acc += ((static_cast<float>(v590_data[8])) * v64_data);
              v589_acc += ((static_cast<float>(v590_data[9])) * v66_data);
              v589_acc += ((static_cast<float>(v590_data[10])) * v68_data);
              v589_acc += ((static_cast<float>(v590_data[11])) * v70_data);
              v589_acc += ((static_cast<float>(v590_data[12])) * v72_data);
              v589_acc += ((static_cast<float>(v590_data[13])) * v74_data);
              v589_acc += ((static_cast<float>(v590_data[14])) * v76_data);
              v589_acc += ((static_cast<float>(v590_data[15])) * v78_data);
              ir0.template select<16, 1>(240) = v589_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v623_n0 = 0; v623_n0 < 1; ++v623_n0) {
                int32_t v625_a = v623_n0 * 16;
                #pragma unroll
                for (int32_t v624_n1 = 0; v624_n1 < 16; ++v624_n1) {
                  int32_t v627_a = v625_a + (v624_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v628_data(ir0.template select<16, 1>(v627_a));
                  r0.template select<16, 1>(v627_a) = v628_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v629_i0 = 0; v629_i0 < 1; ++v629_i0) {
                int32_t v631_a = v629_i0 * 16;
                #pragma unroll
                for (int32_t v630_i1 = 0; v630_i1 < 16; ++v630_i1) {
                  int32_t v633_a = v631_a + (v630_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v634_data(r0.template select<16, 1>(v633_a));
                  v634_data.copy_to(glb_m0 + (v633_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


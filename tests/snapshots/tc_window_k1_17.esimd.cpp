// === base name ===
kernel_d3fa2bebc395a6f0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d3fa2bebc395a6f0 = {{1, 32, 1}, 16, 16, 1, 32, 23616, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d3fa2bebc395a6f0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d3fa2bebc395a6f0(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d3fa2bebc395a6f0(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_d3fa2bebc395a6f0(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d3fa2bebc395a6f0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d3fa2bebc395a6f0(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d3fa2bebc395a6f0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (160);
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
          if (item.get_local_id(1) == 16) {
            tensorforge::intel_esimd::simd<float, 16> v28_ld;
            v28_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<float, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v28_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v30_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v30_batchId0 < numElements0; v30_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v31_ahead1 = v30_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v33_batchId1 = (v31_ahead1 < numElements0) ? v31_ahead1 : v30_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v30_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v30_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v30_batchId0 * 153 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v40_ld;
              v40_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v40_ld);
              tensorforge::intel_esimd::simd<float, 64> v41_ld;
              v41_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v41_ld);
              tensorforge::intel_esimd::simd<float, 16> v42_ld;
              v42_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v42_ld);
              tensorforge::intel_esimd::simd<float, 9> v43_ld;
              v43_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v43_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v49_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v51_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v53_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v55_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v57_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v59_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v61_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v63_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v67_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v69_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v71_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v73_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v75_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v77_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v79_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v81_data = tensorforge::slmLoad<float, 16>(glb_m1 + (256_i32));
              tensorforge::intel_esimd::simd<float, 16> v82_acc{};
              tensorforge::intel_esimd::simd<float, 16> v85_data(0.0f);
              v85_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v82_acc += ((static_cast<float>(v85_data[1])) * v49_data);
              v82_acc += ((static_cast<float>(v85_data[2])) * v51_data);
              v82_acc += ((static_cast<float>(v85_data[3])) * v53_data);
              v82_acc += ((static_cast<float>(v85_data[4])) * v55_data);
              v82_acc += ((static_cast<float>(v85_data[5])) * v57_data);
              v82_acc += ((static_cast<float>(v85_data[6])) * v59_data);
              v82_acc += ((static_cast<float>(v85_data[7])) * v61_data);
              v82_acc += ((static_cast<float>(v85_data[8])) * v63_data);
              v82_acc += ((static_cast<float>(v85_data[9])) * v65_data);
              v82_acc += ((static_cast<float>(v85_data[10])) * v67_data);
              v82_acc += ((static_cast<float>(v85_data[11])) * v69_data);
              v82_acc += ((static_cast<float>(v85_data[12])) * v71_data);
              v82_acc += ((static_cast<float>(v85_data[13])) * v73_data);
              v82_acc += ((static_cast<float>(v85_data[14])) * v75_data);
              v82_acc += ((static_cast<float>(v85_data[15])) * v77_data);
              tensorforge::intel_esimd::simd<float, 16> v121_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v82_acc += ((static_cast<float>(v121_data[0])) * v79_data);
              v82_acc += ((static_cast<float>(v121_data[1])) * v81_data);
              ir0.template select<16, 1>(0) = v82_acc;
              tensorforge::intel_esimd::simd<float, 16> v126_acc{};
              tensorforge::intel_esimd::simd<float, 16> v128_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v126_acc += ((static_cast<float>(v128_data[1])) * v49_data);
              v126_acc += ((static_cast<float>(v128_data[2])) * v51_data);
              v126_acc += ((static_cast<float>(v128_data[3])) * v53_data);
              v126_acc += ((static_cast<float>(v128_data[4])) * v55_data);
              v126_acc += ((static_cast<float>(v128_data[5])) * v57_data);
              v126_acc += ((static_cast<float>(v128_data[6])) * v59_data);
              v126_acc += ((static_cast<float>(v128_data[7])) * v61_data);
              v126_acc += ((static_cast<float>(v128_data[8])) * v63_data);
              v126_acc += ((static_cast<float>(v128_data[9])) * v65_data);
              v126_acc += ((static_cast<float>(v128_data[10])) * v67_data);
              v126_acc += ((static_cast<float>(v128_data[11])) * v69_data);
              v126_acc += ((static_cast<float>(v128_data[12])) * v71_data);
              v126_acc += ((static_cast<float>(v128_data[13])) * v73_data);
              v126_acc += ((static_cast<float>(v128_data[14])) * v75_data);
              v126_acc += ((static_cast<float>(v128_data[15])) * v77_data);
              tensorforge::intel_esimd::simd<float, 16> v161_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v126_acc += ((static_cast<float>(v161_data[0])) * v79_data);
              v126_acc += ((static_cast<float>(v161_data[1])) * v81_data);
              ir0.template select<16, 1>(16) = v126_acc;
              tensorforge::intel_esimd::simd<float, 16> v166_acc{};
              tensorforge::intel_esimd::simd<float, 16> v168_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v166_acc += ((static_cast<float>(v168_data[1])) * v49_data);
              v166_acc += ((static_cast<float>(v168_data[2])) * v51_data);
              v166_acc += ((static_cast<float>(v168_data[3])) * v53_data);
              v166_acc += ((static_cast<float>(v168_data[4])) * v55_data);
              v166_acc += ((static_cast<float>(v168_data[5])) * v57_data);
              v166_acc += ((static_cast<float>(v168_data[6])) * v59_data);
              v166_acc += ((static_cast<float>(v168_data[7])) * v61_data);
              v166_acc += ((static_cast<float>(v168_data[8])) * v63_data);
              v166_acc += ((static_cast<float>(v168_data[9])) * v65_data);
              v166_acc += ((static_cast<float>(v168_data[10])) * v67_data);
              v166_acc += ((static_cast<float>(v168_data[11])) * v69_data);
              v166_acc += ((static_cast<float>(v168_data[12])) * v71_data);
              v166_acc += ((static_cast<float>(v168_data[13])) * v73_data);
              v166_acc += ((static_cast<float>(v168_data[14])) * v75_data);
              v166_acc += ((static_cast<float>(v168_data[15])) * v77_data);
              tensorforge::intel_esimd::simd<float, 16> v201_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v166_acc += ((static_cast<float>(v201_data[0])) * v79_data);
              v166_acc += ((static_cast<float>(v201_data[1])) * v81_data);
              ir0.template select<16, 1>(32) = v166_acc;
              tensorforge::intel_esimd::simd<float, 16> v206_acc{};
              tensorforge::intel_esimd::simd<float, 16> v208_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v206_acc += ((static_cast<float>(v208_data[1])) * v49_data);
              v206_acc += ((static_cast<float>(v208_data[2])) * v51_data);
              v206_acc += ((static_cast<float>(v208_data[3])) * v53_data);
              v206_acc += ((static_cast<float>(v208_data[4])) * v55_data);
              v206_acc += ((static_cast<float>(v208_data[5])) * v57_data);
              v206_acc += ((static_cast<float>(v208_data[6])) * v59_data);
              v206_acc += ((static_cast<float>(v208_data[7])) * v61_data);
              v206_acc += ((static_cast<float>(v208_data[8])) * v63_data);
              v206_acc += ((static_cast<float>(v208_data[9])) * v65_data);
              v206_acc += ((static_cast<float>(v208_data[10])) * v67_data);
              v206_acc += ((static_cast<float>(v208_data[11])) * v69_data);
              v206_acc += ((static_cast<float>(v208_data[12])) * v71_data);
              v206_acc += ((static_cast<float>(v208_data[13])) * v73_data);
              v206_acc += ((static_cast<float>(v208_data[14])) * v75_data);
              v206_acc += ((static_cast<float>(v208_data[15])) * v77_data);
              tensorforge::intel_esimd::simd<float, 16> v241_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v206_acc += ((static_cast<float>(v241_data[0])) * v79_data);
              v206_acc += ((static_cast<float>(v241_data[1])) * v81_data);
              ir0.template select<16, 1>(48) = v206_acc;
              tensorforge::intel_esimd::simd<float, 16> v246_acc{};
              tensorforge::intel_esimd::simd<float, 16> v248_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v246_acc += ((static_cast<float>(v248_data[1])) * v49_data);
              v246_acc += ((static_cast<float>(v248_data[2])) * v51_data);
              v246_acc += ((static_cast<float>(v248_data[3])) * v53_data);
              v246_acc += ((static_cast<float>(v248_data[4])) * v55_data);
              v246_acc += ((static_cast<float>(v248_data[5])) * v57_data);
              v246_acc += ((static_cast<float>(v248_data[6])) * v59_data);
              v246_acc += ((static_cast<float>(v248_data[7])) * v61_data);
              v246_acc += ((static_cast<float>(v248_data[8])) * v63_data);
              v246_acc += ((static_cast<float>(v248_data[9])) * v65_data);
              v246_acc += ((static_cast<float>(v248_data[10])) * v67_data);
              v246_acc += ((static_cast<float>(v248_data[11])) * v69_data);
              v246_acc += ((static_cast<float>(v248_data[12])) * v71_data);
              v246_acc += ((static_cast<float>(v248_data[13])) * v73_data);
              v246_acc += ((static_cast<float>(v248_data[14])) * v75_data);
              v246_acc += ((static_cast<float>(v248_data[15])) * v77_data);
              tensorforge::intel_esimd::simd<float, 16> v281_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v246_acc += ((static_cast<float>(v281_data[0])) * v79_data);
              v246_acc += ((static_cast<float>(v281_data[1])) * v81_data);
              ir0.template select<16, 1>(64) = v246_acc;
              tensorforge::intel_esimd::simd<float, 16> v286_acc{};
              tensorforge::intel_esimd::simd<float, 16> v288_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v286_acc += ((static_cast<float>(v288_data[1])) * v49_data);
              v286_acc += ((static_cast<float>(v288_data[2])) * v51_data);
              v286_acc += ((static_cast<float>(v288_data[3])) * v53_data);
              v286_acc += ((static_cast<float>(v288_data[4])) * v55_data);
              v286_acc += ((static_cast<float>(v288_data[5])) * v57_data);
              v286_acc += ((static_cast<float>(v288_data[6])) * v59_data);
              v286_acc += ((static_cast<float>(v288_data[7])) * v61_data);
              v286_acc += ((static_cast<float>(v288_data[8])) * v63_data);
              v286_acc += ((static_cast<float>(v288_data[9])) * v65_data);
              v286_acc += ((static_cast<float>(v288_data[10])) * v67_data);
              v286_acc += ((static_cast<float>(v288_data[11])) * v69_data);
              v286_acc += ((static_cast<float>(v288_data[12])) * v71_data);
              v286_acc += ((static_cast<float>(v288_data[13])) * v73_data);
              v286_acc += ((static_cast<float>(v288_data[14])) * v75_data);
              v286_acc += ((static_cast<float>(v288_data[15])) * v77_data);
              tensorforge::intel_esimd::simd<float, 16> v321_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v286_acc += ((static_cast<float>(v321_data[0])) * v79_data);
              v286_acc += ((static_cast<float>(v321_data[1])) * v81_data);
              ir0.template select<16, 1>(80) = v286_acc;
              tensorforge::intel_esimd::simd<float, 16> v326_acc{};
              tensorforge::intel_esimd::simd<float, 16> v328_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v326_acc += ((static_cast<float>(v328_data[1])) * v49_data);
              v326_acc += ((static_cast<float>(v328_data[2])) * v51_data);
              v326_acc += ((static_cast<float>(v328_data[3])) * v53_data);
              v326_acc += ((static_cast<float>(v328_data[4])) * v55_data);
              v326_acc += ((static_cast<float>(v328_data[5])) * v57_data);
              v326_acc += ((static_cast<float>(v328_data[6])) * v59_data);
              v326_acc += ((static_cast<float>(v328_data[7])) * v61_data);
              v326_acc += ((static_cast<float>(v328_data[8])) * v63_data);
              v326_acc += ((static_cast<float>(v328_data[9])) * v65_data);
              v326_acc += ((static_cast<float>(v328_data[10])) * v67_data);
              v326_acc += ((static_cast<float>(v328_data[11])) * v69_data);
              v326_acc += ((static_cast<float>(v328_data[12])) * v71_data);
              v326_acc += ((static_cast<float>(v328_data[13])) * v73_data);
              v326_acc += ((static_cast<float>(v328_data[14])) * v75_data);
              v326_acc += ((static_cast<float>(v328_data[15])) * v77_data);
              tensorforge::intel_esimd::simd<float, 16> v361_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v326_acc += ((static_cast<float>(v361_data[0])) * v79_data);
              v326_acc += ((static_cast<float>(v361_data[1])) * v81_data);
              ir0.template select<16, 1>(96) = v326_acc;
              tensorforge::intel_esimd::simd<float, 16> v366_acc{};
              tensorforge::intel_esimd::simd<float, 16> v368_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v366_acc += ((static_cast<float>(v368_data[1])) * v49_data);
              v366_acc += ((static_cast<float>(v368_data[2])) * v51_data);
              v366_acc += ((static_cast<float>(v368_data[3])) * v53_data);
              v366_acc += ((static_cast<float>(v368_data[4])) * v55_data);
              v366_acc += ((static_cast<float>(v368_data[5])) * v57_data);
              v366_acc += ((static_cast<float>(v368_data[6])) * v59_data);
              v366_acc += ((static_cast<float>(v368_data[7])) * v61_data);
              v366_acc += ((static_cast<float>(v368_data[8])) * v63_data);
              v366_acc += ((static_cast<float>(v368_data[9])) * v65_data);
              v366_acc += ((static_cast<float>(v368_data[10])) * v67_data);
              v366_acc += ((static_cast<float>(v368_data[11])) * v69_data);
              v366_acc += ((static_cast<float>(v368_data[12])) * v71_data);
              v366_acc += ((static_cast<float>(v368_data[13])) * v73_data);
              v366_acc += ((static_cast<float>(v368_data[14])) * v75_data);
              v366_acc += ((static_cast<float>(v368_data[15])) * v77_data);
              tensorforge::intel_esimd::simd<float, 16> v401_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v366_acc += ((static_cast<float>(v401_data[0])) * v79_data);
              v366_acc += ((static_cast<float>(v401_data[1])) * v81_data);
              ir0.template select<16, 1>(112) = v366_acc;
              tensorforge::intel_esimd::simd<float, 16> v406_acc{};
              tensorforge::intel_esimd::simd<float, 16> v408_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v406_acc += ((static_cast<float>(v408_data[1])) * v49_data);
              v406_acc += ((static_cast<float>(v408_data[2])) * v51_data);
              v406_acc += ((static_cast<float>(v408_data[3])) * v53_data);
              v406_acc += ((static_cast<float>(v408_data[4])) * v55_data);
              v406_acc += ((static_cast<float>(v408_data[5])) * v57_data);
              v406_acc += ((static_cast<float>(v408_data[6])) * v59_data);
              v406_acc += ((static_cast<float>(v408_data[7])) * v61_data);
              v406_acc += ((static_cast<float>(v408_data[8])) * v63_data);
              v406_acc += ((static_cast<float>(v408_data[9])) * v65_data);
              v406_acc += ((static_cast<float>(v408_data[10])) * v67_data);
              v406_acc += ((static_cast<float>(v408_data[11])) * v69_data);
              v406_acc += ((static_cast<float>(v408_data[12])) * v71_data);
              v406_acc += ((static_cast<float>(v408_data[13])) * v73_data);
              v406_acc += ((static_cast<float>(v408_data[14])) * v75_data);
              v406_acc += ((static_cast<float>(v408_data[15])) * v77_data);
              tensorforge::intel_esimd::simd<float, 16> v441_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v406_acc += ((static_cast<float>(v441_data[0])) * v79_data);
              v406_acc += ((static_cast<float>(v441_data[1])) * v81_data);
              ir0.template select<16, 1>(128) = v406_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v446_n0 = 0; v446_n0 < 1; ++v446_n0) {
                int32_t v448_a = v446_n0 * 16;
                #pragma unroll
                for (int32_t v447_n1 = 0; v447_n1 < 9; ++v447_n1) {
                  int32_t v450_a = v448_a + (v447_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v451_data(ir0.template select<16, 1>(v450_a));
                  r0.template select<16, 1>(v450_a) = v451_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v452_i0 = 0; v452_i0 < 1; ++v452_i0) {
                int32_t v454_a = v452_i0 * 16;
                #pragma unroll
                for (int32_t v453_i1 = 0; v453_i1 < 9; ++v453_i1) {
                  int32_t v456_a = v454_a + (v453_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v457_data(r0.template select<16, 1>(v456_a));
                  v457_data.copy_to(glb_m0 + (v456_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


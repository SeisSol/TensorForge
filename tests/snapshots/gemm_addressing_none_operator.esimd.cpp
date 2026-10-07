// === base name ===
kernel_86e9021473611844

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_86e9021473611844 = {{1, 32, 1}, 16, 16, 1, 32, 35840, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_86e9021473611844(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_86e9021473611844(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_86e9021473611844(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_86e9021473611844(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_86e9021473611844(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_86e9021473611844(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_86e9021473611844(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v26_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v26_batchId0 < numElements0; v26_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v27_ahead1 = v26_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v29_batchId1 = (v27_ahead1 < numElements0) ? v27_ahead1 : v26_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v26_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v26_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v26_batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v36_ld;
              v36_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v36_ld);
              tensorforge::intel_esimd::simd<float, 64> v37_ld;
              v37_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v37_ld);
              tensorforge::intel_esimd::simd<float, 64> v38_ld;
              v38_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 128));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v38_ld);
              tensorforge::intel_esimd::simd<float, 64> v39_ld;
              v39_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 192));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v39_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v45_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v47_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v49_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v51_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v53_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v55_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v57_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v59_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v61_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v63_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v65_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v67_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v69_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v71_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v73_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v75_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v76_acc{};
              tensorforge::intel_esimd::simd<float, 16> v77_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v76_acc += ((static_cast<float>(v77_data[0])) * v45_data);
              v76_acc += ((static_cast<float>(v77_data[1])) * v47_data);
              v76_acc += ((static_cast<float>(v77_data[2])) * v49_data);
              v76_acc += ((static_cast<float>(v77_data[3])) * v51_data);
              v76_acc += ((static_cast<float>(v77_data[4])) * v53_data);
              v76_acc += ((static_cast<float>(v77_data[5])) * v55_data);
              v76_acc += ((static_cast<float>(v77_data[6])) * v57_data);
              v76_acc += ((static_cast<float>(v77_data[7])) * v59_data);
              v76_acc += ((static_cast<float>(v77_data[8])) * v61_data);
              v76_acc += ((static_cast<float>(v77_data[9])) * v63_data);
              v76_acc += ((static_cast<float>(v77_data[10])) * v65_data);
              v76_acc += ((static_cast<float>(v77_data[11])) * v67_data);
              v76_acc += ((static_cast<float>(v77_data[12])) * v69_data);
              v76_acc += ((static_cast<float>(v77_data[13])) * v71_data);
              v76_acc += ((static_cast<float>(v77_data[14])) * v73_data);
              v76_acc += ((static_cast<float>(v77_data[15])) * v75_data);
              ir0.template select<16, 1>(0) = v76_acc;
              tensorforge::intel_esimd::simd<float, 16> v110_acc{};
              tensorforge::intel_esimd::simd<float, 16> v111_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v110_acc += ((static_cast<float>(v111_data[0])) * v45_data);
              v110_acc += ((static_cast<float>(v111_data[1])) * v47_data);
              v110_acc += ((static_cast<float>(v111_data[2])) * v49_data);
              v110_acc += ((static_cast<float>(v111_data[3])) * v51_data);
              v110_acc += ((static_cast<float>(v111_data[4])) * v53_data);
              v110_acc += ((static_cast<float>(v111_data[5])) * v55_data);
              v110_acc += ((static_cast<float>(v111_data[6])) * v57_data);
              v110_acc += ((static_cast<float>(v111_data[7])) * v59_data);
              v110_acc += ((static_cast<float>(v111_data[8])) * v61_data);
              v110_acc += ((static_cast<float>(v111_data[9])) * v63_data);
              v110_acc += ((static_cast<float>(v111_data[10])) * v65_data);
              v110_acc += ((static_cast<float>(v111_data[11])) * v67_data);
              v110_acc += ((static_cast<float>(v111_data[12])) * v69_data);
              v110_acc += ((static_cast<float>(v111_data[13])) * v71_data);
              v110_acc += ((static_cast<float>(v111_data[14])) * v73_data);
              v110_acc += ((static_cast<float>(v111_data[15])) * v75_data);
              ir0.template select<16, 1>(16) = v110_acc;
              tensorforge::intel_esimd::simd<float, 16> v144_acc{};
              tensorforge::intel_esimd::simd<float, 16> v145_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v144_acc += ((static_cast<float>(v145_data[0])) * v45_data);
              v144_acc += ((static_cast<float>(v145_data[1])) * v47_data);
              v144_acc += ((static_cast<float>(v145_data[2])) * v49_data);
              v144_acc += ((static_cast<float>(v145_data[3])) * v51_data);
              v144_acc += ((static_cast<float>(v145_data[4])) * v53_data);
              v144_acc += ((static_cast<float>(v145_data[5])) * v55_data);
              v144_acc += ((static_cast<float>(v145_data[6])) * v57_data);
              v144_acc += ((static_cast<float>(v145_data[7])) * v59_data);
              v144_acc += ((static_cast<float>(v145_data[8])) * v61_data);
              v144_acc += ((static_cast<float>(v145_data[9])) * v63_data);
              v144_acc += ((static_cast<float>(v145_data[10])) * v65_data);
              v144_acc += ((static_cast<float>(v145_data[11])) * v67_data);
              v144_acc += ((static_cast<float>(v145_data[12])) * v69_data);
              v144_acc += ((static_cast<float>(v145_data[13])) * v71_data);
              v144_acc += ((static_cast<float>(v145_data[14])) * v73_data);
              v144_acc += ((static_cast<float>(v145_data[15])) * v75_data);
              ir0.template select<16, 1>(32) = v144_acc;
              tensorforge::intel_esimd::simd<float, 16> v178_acc{};
              tensorforge::intel_esimd::simd<float, 16> v179_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v178_acc += ((static_cast<float>(v179_data[0])) * v45_data);
              v178_acc += ((static_cast<float>(v179_data[1])) * v47_data);
              v178_acc += ((static_cast<float>(v179_data[2])) * v49_data);
              v178_acc += ((static_cast<float>(v179_data[3])) * v51_data);
              v178_acc += ((static_cast<float>(v179_data[4])) * v53_data);
              v178_acc += ((static_cast<float>(v179_data[5])) * v55_data);
              v178_acc += ((static_cast<float>(v179_data[6])) * v57_data);
              v178_acc += ((static_cast<float>(v179_data[7])) * v59_data);
              v178_acc += ((static_cast<float>(v179_data[8])) * v61_data);
              v178_acc += ((static_cast<float>(v179_data[9])) * v63_data);
              v178_acc += ((static_cast<float>(v179_data[10])) * v65_data);
              v178_acc += ((static_cast<float>(v179_data[11])) * v67_data);
              v178_acc += ((static_cast<float>(v179_data[12])) * v69_data);
              v178_acc += ((static_cast<float>(v179_data[13])) * v71_data);
              v178_acc += ((static_cast<float>(v179_data[14])) * v73_data);
              v178_acc += ((static_cast<float>(v179_data[15])) * v75_data);
              ir0.template select<16, 1>(48) = v178_acc;
              tensorforge::intel_esimd::simd<float, 16> v212_acc{};
              tensorforge::intel_esimd::simd<float, 16> v213_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v212_acc += ((static_cast<float>(v213_data[0])) * v45_data);
              v212_acc += ((static_cast<float>(v213_data[1])) * v47_data);
              v212_acc += ((static_cast<float>(v213_data[2])) * v49_data);
              v212_acc += ((static_cast<float>(v213_data[3])) * v51_data);
              v212_acc += ((static_cast<float>(v213_data[4])) * v53_data);
              v212_acc += ((static_cast<float>(v213_data[5])) * v55_data);
              v212_acc += ((static_cast<float>(v213_data[6])) * v57_data);
              v212_acc += ((static_cast<float>(v213_data[7])) * v59_data);
              v212_acc += ((static_cast<float>(v213_data[8])) * v61_data);
              v212_acc += ((static_cast<float>(v213_data[9])) * v63_data);
              v212_acc += ((static_cast<float>(v213_data[10])) * v65_data);
              v212_acc += ((static_cast<float>(v213_data[11])) * v67_data);
              v212_acc += ((static_cast<float>(v213_data[12])) * v69_data);
              v212_acc += ((static_cast<float>(v213_data[13])) * v71_data);
              v212_acc += ((static_cast<float>(v213_data[14])) * v73_data);
              v212_acc += ((static_cast<float>(v213_data[15])) * v75_data);
              ir0.template select<16, 1>(64) = v212_acc;
              tensorforge::intel_esimd::simd<float, 16> v246_acc{};
              tensorforge::intel_esimd::simd<float, 16> v247_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v246_acc += ((static_cast<float>(v247_data[0])) * v45_data);
              v246_acc += ((static_cast<float>(v247_data[1])) * v47_data);
              v246_acc += ((static_cast<float>(v247_data[2])) * v49_data);
              v246_acc += ((static_cast<float>(v247_data[3])) * v51_data);
              v246_acc += ((static_cast<float>(v247_data[4])) * v53_data);
              v246_acc += ((static_cast<float>(v247_data[5])) * v55_data);
              v246_acc += ((static_cast<float>(v247_data[6])) * v57_data);
              v246_acc += ((static_cast<float>(v247_data[7])) * v59_data);
              v246_acc += ((static_cast<float>(v247_data[8])) * v61_data);
              v246_acc += ((static_cast<float>(v247_data[9])) * v63_data);
              v246_acc += ((static_cast<float>(v247_data[10])) * v65_data);
              v246_acc += ((static_cast<float>(v247_data[11])) * v67_data);
              v246_acc += ((static_cast<float>(v247_data[12])) * v69_data);
              v246_acc += ((static_cast<float>(v247_data[13])) * v71_data);
              v246_acc += ((static_cast<float>(v247_data[14])) * v73_data);
              v246_acc += ((static_cast<float>(v247_data[15])) * v75_data);
              ir0.template select<16, 1>(80) = v246_acc;
              tensorforge::intel_esimd::simd<float, 16> v280_acc{};
              tensorforge::intel_esimd::simd<float, 16> v281_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v280_acc += ((static_cast<float>(v281_data[0])) * v45_data);
              v280_acc += ((static_cast<float>(v281_data[1])) * v47_data);
              v280_acc += ((static_cast<float>(v281_data[2])) * v49_data);
              v280_acc += ((static_cast<float>(v281_data[3])) * v51_data);
              v280_acc += ((static_cast<float>(v281_data[4])) * v53_data);
              v280_acc += ((static_cast<float>(v281_data[5])) * v55_data);
              v280_acc += ((static_cast<float>(v281_data[6])) * v57_data);
              v280_acc += ((static_cast<float>(v281_data[7])) * v59_data);
              v280_acc += ((static_cast<float>(v281_data[8])) * v61_data);
              v280_acc += ((static_cast<float>(v281_data[9])) * v63_data);
              v280_acc += ((static_cast<float>(v281_data[10])) * v65_data);
              v280_acc += ((static_cast<float>(v281_data[11])) * v67_data);
              v280_acc += ((static_cast<float>(v281_data[12])) * v69_data);
              v280_acc += ((static_cast<float>(v281_data[13])) * v71_data);
              v280_acc += ((static_cast<float>(v281_data[14])) * v73_data);
              v280_acc += ((static_cast<float>(v281_data[15])) * v75_data);
              ir0.template select<16, 1>(96) = v280_acc;
              tensorforge::intel_esimd::simd<float, 16> v314_acc{};
              tensorforge::intel_esimd::simd<float, 16> v315_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v314_acc += ((static_cast<float>(v315_data[0])) * v45_data);
              v314_acc += ((static_cast<float>(v315_data[1])) * v47_data);
              v314_acc += ((static_cast<float>(v315_data[2])) * v49_data);
              v314_acc += ((static_cast<float>(v315_data[3])) * v51_data);
              v314_acc += ((static_cast<float>(v315_data[4])) * v53_data);
              v314_acc += ((static_cast<float>(v315_data[5])) * v55_data);
              v314_acc += ((static_cast<float>(v315_data[6])) * v57_data);
              v314_acc += ((static_cast<float>(v315_data[7])) * v59_data);
              v314_acc += ((static_cast<float>(v315_data[8])) * v61_data);
              v314_acc += ((static_cast<float>(v315_data[9])) * v63_data);
              v314_acc += ((static_cast<float>(v315_data[10])) * v65_data);
              v314_acc += ((static_cast<float>(v315_data[11])) * v67_data);
              v314_acc += ((static_cast<float>(v315_data[12])) * v69_data);
              v314_acc += ((static_cast<float>(v315_data[13])) * v71_data);
              v314_acc += ((static_cast<float>(v315_data[14])) * v73_data);
              v314_acc += ((static_cast<float>(v315_data[15])) * v75_data);
              ir0.template select<16, 1>(112) = v314_acc;
              tensorforge::intel_esimd::simd<float, 16> v348_acc{};
              tensorforge::intel_esimd::simd<float, 16> v349_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v348_acc += ((static_cast<float>(v349_data[0])) * v45_data);
              v348_acc += ((static_cast<float>(v349_data[1])) * v47_data);
              v348_acc += ((static_cast<float>(v349_data[2])) * v49_data);
              v348_acc += ((static_cast<float>(v349_data[3])) * v51_data);
              v348_acc += ((static_cast<float>(v349_data[4])) * v53_data);
              v348_acc += ((static_cast<float>(v349_data[5])) * v55_data);
              v348_acc += ((static_cast<float>(v349_data[6])) * v57_data);
              v348_acc += ((static_cast<float>(v349_data[7])) * v59_data);
              v348_acc += ((static_cast<float>(v349_data[8])) * v61_data);
              v348_acc += ((static_cast<float>(v349_data[9])) * v63_data);
              v348_acc += ((static_cast<float>(v349_data[10])) * v65_data);
              v348_acc += ((static_cast<float>(v349_data[11])) * v67_data);
              v348_acc += ((static_cast<float>(v349_data[12])) * v69_data);
              v348_acc += ((static_cast<float>(v349_data[13])) * v71_data);
              v348_acc += ((static_cast<float>(v349_data[14])) * v73_data);
              v348_acc += ((static_cast<float>(v349_data[15])) * v75_data);
              ir0.template select<16, 1>(128) = v348_acc;
              tensorforge::intel_esimd::simd<float, 16> v382_acc{};
              tensorforge::intel_esimd::simd<float, 16> v383_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v382_acc += ((static_cast<float>(v383_data[0])) * v45_data);
              v382_acc += ((static_cast<float>(v383_data[1])) * v47_data);
              v382_acc += ((static_cast<float>(v383_data[2])) * v49_data);
              v382_acc += ((static_cast<float>(v383_data[3])) * v51_data);
              v382_acc += ((static_cast<float>(v383_data[4])) * v53_data);
              v382_acc += ((static_cast<float>(v383_data[5])) * v55_data);
              v382_acc += ((static_cast<float>(v383_data[6])) * v57_data);
              v382_acc += ((static_cast<float>(v383_data[7])) * v59_data);
              v382_acc += ((static_cast<float>(v383_data[8])) * v61_data);
              v382_acc += ((static_cast<float>(v383_data[9])) * v63_data);
              v382_acc += ((static_cast<float>(v383_data[10])) * v65_data);
              v382_acc += ((static_cast<float>(v383_data[11])) * v67_data);
              v382_acc += ((static_cast<float>(v383_data[12])) * v69_data);
              v382_acc += ((static_cast<float>(v383_data[13])) * v71_data);
              v382_acc += ((static_cast<float>(v383_data[14])) * v73_data);
              v382_acc += ((static_cast<float>(v383_data[15])) * v75_data);
              ir0.template select<16, 1>(144) = v382_acc;
              tensorforge::intel_esimd::simd<float, 16> v416_acc{};
              tensorforge::intel_esimd::simd<float, 16> v417_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v416_acc += ((static_cast<float>(v417_data[0])) * v45_data);
              v416_acc += ((static_cast<float>(v417_data[1])) * v47_data);
              v416_acc += ((static_cast<float>(v417_data[2])) * v49_data);
              v416_acc += ((static_cast<float>(v417_data[3])) * v51_data);
              v416_acc += ((static_cast<float>(v417_data[4])) * v53_data);
              v416_acc += ((static_cast<float>(v417_data[5])) * v55_data);
              v416_acc += ((static_cast<float>(v417_data[6])) * v57_data);
              v416_acc += ((static_cast<float>(v417_data[7])) * v59_data);
              v416_acc += ((static_cast<float>(v417_data[8])) * v61_data);
              v416_acc += ((static_cast<float>(v417_data[9])) * v63_data);
              v416_acc += ((static_cast<float>(v417_data[10])) * v65_data);
              v416_acc += ((static_cast<float>(v417_data[11])) * v67_data);
              v416_acc += ((static_cast<float>(v417_data[12])) * v69_data);
              v416_acc += ((static_cast<float>(v417_data[13])) * v71_data);
              v416_acc += ((static_cast<float>(v417_data[14])) * v73_data);
              v416_acc += ((static_cast<float>(v417_data[15])) * v75_data);
              ir0.template select<16, 1>(160) = v416_acc;
              tensorforge::intel_esimd::simd<float, 16> v450_acc{};
              tensorforge::intel_esimd::simd<float, 16> v451_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v450_acc += ((static_cast<float>(v451_data[0])) * v45_data);
              v450_acc += ((static_cast<float>(v451_data[1])) * v47_data);
              v450_acc += ((static_cast<float>(v451_data[2])) * v49_data);
              v450_acc += ((static_cast<float>(v451_data[3])) * v51_data);
              v450_acc += ((static_cast<float>(v451_data[4])) * v53_data);
              v450_acc += ((static_cast<float>(v451_data[5])) * v55_data);
              v450_acc += ((static_cast<float>(v451_data[6])) * v57_data);
              v450_acc += ((static_cast<float>(v451_data[7])) * v59_data);
              v450_acc += ((static_cast<float>(v451_data[8])) * v61_data);
              v450_acc += ((static_cast<float>(v451_data[9])) * v63_data);
              v450_acc += ((static_cast<float>(v451_data[10])) * v65_data);
              v450_acc += ((static_cast<float>(v451_data[11])) * v67_data);
              v450_acc += ((static_cast<float>(v451_data[12])) * v69_data);
              v450_acc += ((static_cast<float>(v451_data[13])) * v71_data);
              v450_acc += ((static_cast<float>(v451_data[14])) * v73_data);
              v450_acc += ((static_cast<float>(v451_data[15])) * v75_data);
              ir0.template select<16, 1>(176) = v450_acc;
              tensorforge::intel_esimd::simd<float, 16> v484_acc{};
              tensorforge::intel_esimd::simd<float, 16> v485_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v484_acc += ((static_cast<float>(v485_data[0])) * v45_data);
              v484_acc += ((static_cast<float>(v485_data[1])) * v47_data);
              v484_acc += ((static_cast<float>(v485_data[2])) * v49_data);
              v484_acc += ((static_cast<float>(v485_data[3])) * v51_data);
              v484_acc += ((static_cast<float>(v485_data[4])) * v53_data);
              v484_acc += ((static_cast<float>(v485_data[5])) * v55_data);
              v484_acc += ((static_cast<float>(v485_data[6])) * v57_data);
              v484_acc += ((static_cast<float>(v485_data[7])) * v59_data);
              v484_acc += ((static_cast<float>(v485_data[8])) * v61_data);
              v484_acc += ((static_cast<float>(v485_data[9])) * v63_data);
              v484_acc += ((static_cast<float>(v485_data[10])) * v65_data);
              v484_acc += ((static_cast<float>(v485_data[11])) * v67_data);
              v484_acc += ((static_cast<float>(v485_data[12])) * v69_data);
              v484_acc += ((static_cast<float>(v485_data[13])) * v71_data);
              v484_acc += ((static_cast<float>(v485_data[14])) * v73_data);
              v484_acc += ((static_cast<float>(v485_data[15])) * v75_data);
              ir0.template select<16, 1>(192) = v484_acc;
              tensorforge::intel_esimd::simd<float, 16> v518_acc{};
              tensorforge::intel_esimd::simd<float, 16> v519_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v518_acc += ((static_cast<float>(v519_data[0])) * v45_data);
              v518_acc += ((static_cast<float>(v519_data[1])) * v47_data);
              v518_acc += ((static_cast<float>(v519_data[2])) * v49_data);
              v518_acc += ((static_cast<float>(v519_data[3])) * v51_data);
              v518_acc += ((static_cast<float>(v519_data[4])) * v53_data);
              v518_acc += ((static_cast<float>(v519_data[5])) * v55_data);
              v518_acc += ((static_cast<float>(v519_data[6])) * v57_data);
              v518_acc += ((static_cast<float>(v519_data[7])) * v59_data);
              v518_acc += ((static_cast<float>(v519_data[8])) * v61_data);
              v518_acc += ((static_cast<float>(v519_data[9])) * v63_data);
              v518_acc += ((static_cast<float>(v519_data[10])) * v65_data);
              v518_acc += ((static_cast<float>(v519_data[11])) * v67_data);
              v518_acc += ((static_cast<float>(v519_data[12])) * v69_data);
              v518_acc += ((static_cast<float>(v519_data[13])) * v71_data);
              v518_acc += ((static_cast<float>(v519_data[14])) * v73_data);
              v518_acc += ((static_cast<float>(v519_data[15])) * v75_data);
              ir0.template select<16, 1>(208) = v518_acc;
              tensorforge::intel_esimd::simd<float, 16> v552_acc{};
              tensorforge::intel_esimd::simd<float, 16> v553_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v552_acc += ((static_cast<float>(v553_data[0])) * v45_data);
              v552_acc += ((static_cast<float>(v553_data[1])) * v47_data);
              v552_acc += ((static_cast<float>(v553_data[2])) * v49_data);
              v552_acc += ((static_cast<float>(v553_data[3])) * v51_data);
              v552_acc += ((static_cast<float>(v553_data[4])) * v53_data);
              v552_acc += ((static_cast<float>(v553_data[5])) * v55_data);
              v552_acc += ((static_cast<float>(v553_data[6])) * v57_data);
              v552_acc += ((static_cast<float>(v553_data[7])) * v59_data);
              v552_acc += ((static_cast<float>(v553_data[8])) * v61_data);
              v552_acc += ((static_cast<float>(v553_data[9])) * v63_data);
              v552_acc += ((static_cast<float>(v553_data[10])) * v65_data);
              v552_acc += ((static_cast<float>(v553_data[11])) * v67_data);
              v552_acc += ((static_cast<float>(v553_data[12])) * v69_data);
              v552_acc += ((static_cast<float>(v553_data[13])) * v71_data);
              v552_acc += ((static_cast<float>(v553_data[14])) * v73_data);
              v552_acc += ((static_cast<float>(v553_data[15])) * v75_data);
              ir0.template select<16, 1>(224) = v552_acc;
              tensorforge::intel_esimd::simd<float, 16> v586_acc{};
              tensorforge::intel_esimd::simd<float, 16> v587_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v586_acc += ((static_cast<float>(v587_data[0])) * v45_data);
              v586_acc += ((static_cast<float>(v587_data[1])) * v47_data);
              v586_acc += ((static_cast<float>(v587_data[2])) * v49_data);
              v586_acc += ((static_cast<float>(v587_data[3])) * v51_data);
              v586_acc += ((static_cast<float>(v587_data[4])) * v53_data);
              v586_acc += ((static_cast<float>(v587_data[5])) * v55_data);
              v586_acc += ((static_cast<float>(v587_data[6])) * v57_data);
              v586_acc += ((static_cast<float>(v587_data[7])) * v59_data);
              v586_acc += ((static_cast<float>(v587_data[8])) * v61_data);
              v586_acc += ((static_cast<float>(v587_data[9])) * v63_data);
              v586_acc += ((static_cast<float>(v587_data[10])) * v65_data);
              v586_acc += ((static_cast<float>(v587_data[11])) * v67_data);
              v586_acc += ((static_cast<float>(v587_data[12])) * v69_data);
              v586_acc += ((static_cast<float>(v587_data[13])) * v71_data);
              v586_acc += ((static_cast<float>(v587_data[14])) * v73_data);
              v586_acc += ((static_cast<float>(v587_data[15])) * v75_data);
              ir0.template select<16, 1>(240) = v586_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v620_n0 = 0; v620_n0 < 1; ++v620_n0) {
                int32_t v622_a = v620_n0 * 16;
                #pragma unroll
                for (int32_t v621_n1 = 0; v621_n1 < 16; ++v621_n1) {
                  int32_t v624_a = v622_a + (v621_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v625_data(ir0.template select<16, 1>(v624_a));
                  r0.template select<16, 1>(v624_a) = v625_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v626_i0 = 0; v626_i0 < 1; ++v626_i0) {
                int32_t v628_a = v626_i0 * 16;
                #pragma unroll
                for (int32_t v627_i1 = 0; v627_i1 < 16; ++v627_i1) {
                  int32_t v630_a = v628_a + (v627_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v631_data(r0.template select<16, 1>(v630_a));
                  v631_data.copy_to(glb_m0 + (v630_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


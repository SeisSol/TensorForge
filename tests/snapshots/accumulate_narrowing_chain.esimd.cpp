// === base name ===
kernel_09a3f9ddcf83350b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_09a3f9ddcf83350b = {{1, 8, 1}, 32, 20, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_09a3f9ddcf83350b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_09a3f9ddcf83350b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_09a3f9ddcf83350b(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_09a3f9ddcf83350b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_09a3f9ddcf83350b(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_09a3f9ddcf83350b(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_09a3f9ddcf83350b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      // generated with TensorForge. Version: 0.0.1
      // options: default
      // launch: 32 lanes (20 active) x 8 per block = block 1x8x1, 0 B shared, occupancy grid
      // operands:
      //   m0 20×9(20×9) {0..20}×{0..9} strided
      //   m1 20×9(20×9) {0..20}×{0..9} strided
      //   m2 10×9(10×9) {0..10}×{0..9} strided
      //   m3 4×9(4×9) {0..4}×{0..9} strided
      //   m4 1×9(1×9) {0..1}×{0..9} strided
      // operations:
      //   m0[i,j] = m1[i,j]
      //   m0[i,j] += m2[i,j]
      //   m0[i,j] += m3[i,j]
      //   m0[i,j] += m4[i,j]
      // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"F0","bbox":[[0,0],[10,9]],"name":"m2","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"strided","alias":"F1","bbox":[[0,0],[4,9]],"name":"m3","ordered":false,"parts":1,"shape":[4,9],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[1,9]],"name":"m4","ordered":false,"parts":1,"shape":[1,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[10,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[4,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[4,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[1,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[1,9]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
      {
        const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
        const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
        const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
        for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
          size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
          const float *const __restrict__ pf_glb_m1 = &m1[v4_batchId1 * 180 + 0 + m1_extraOffset];
          const float *const __restrict__ pf_glb_m2 = &m2[v4_batchId1 * 90 + 0 + m2_extraOffset];
          const float *const __restrict__ pf_glb_m3 = &m3[v4_batchId1 * 36 + 0 + m3_extraOffset];
          const float *const __restrict__ pf_glb_m4 = &m4[v4_batchId1 * 9 + 0 + m4_extraOffset];
          const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
          if (allowed) {
            float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 180 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 180 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v1_batchId0 * 90 + 0 + m2_extraOffset];
            const float *const __restrict__ glb_m3 = &m3[v1_batchId0 * 36 + 0 + m3_extraOffset];
            const float *const __restrict__ glb_m4 = &m4[v1_batchId0 * 9 + 0 + m4_extraOffset];
            tensorforge::intel_esimd::simd<float, 288> r0(0.0f);
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
              tensorforge::intel_esimd::simd<float, 20> v24_data;
              v24_data.copy_from(glb_m1 + ((v19_i1 * 20)));
              r0.template select<20, 1>((v19_i1 * 32)) = v24_data;
            }
            tensorforge::intel_esimd::simd<float, 288> r2(0.0f);
            // r2 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 9; ++v28_i1) {
              tensorforge::intel_esimd::simd<float, 10> v33_data;
              v33_data.copy_from(glb_m2 + ((v28_i1 * 10)));
              r2.template select<10, 1>((v28_i1 * 32)) = v33_data;
            }
            // wait(r0 = load{g>r}(glb_m1););
            tensorforge::intel_esimd::simd<float, 288> r1(0.0f);
            // ir1 = +(r0)
            // [(0, 20), (0, 9)] []
            tensorforge::intel_esimd::simd<float, 288> ir1(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v38_data(r0.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v39_data(ir1.template select<32, 1>(0));
            ir1.template select<32, 1>(0) = (v39_data + v38_data);
            tensorforge::intel_esimd::simd<float, 32> v41_data(r0.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v42_data(ir1.template select<32, 1>(32));
            ir1.template select<32, 1>(32) = (v42_data + v41_data);
            tensorforge::intel_esimd::simd<float, 32> v44_data(r0.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v45_data(ir1.template select<32, 1>(64));
            ir1.template select<32, 1>(64) = (v45_data + v44_data);
            tensorforge::intel_esimd::simd<float, 32> v47_data(r0.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v48_data(ir1.template select<32, 1>(96));
            ir1.template select<32, 1>(96) = (v48_data + v47_data);
            tensorforge::intel_esimd::simd<float, 32> v50_data(r0.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v51_data(ir1.template select<32, 1>(128));
            ir1.template select<32, 1>(128) = (v51_data + v50_data);
            tensorforge::intel_esimd::simd<float, 32> v53_data(r0.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v54_data(ir1.template select<32, 1>(160));
            ir1.template select<32, 1>(160) = (v54_data + v53_data);
            tensorforge::intel_esimd::simd<float, 32> v56_data(r0.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v57_data(ir1.template select<32, 1>(192));
            ir1.template select<32, 1>(192) = (v57_data + v56_data);
            tensorforge::intel_esimd::simd<float, 32> v59_data(r0.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v60_data(ir1.template select<32, 1>(224));
            ir1.template select<32, 1>(224) = (v60_data + v59_data);
            tensorforge::intel_esimd::simd<float, 32> v62_data(r0.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v63_data(ir1.template select<32, 1>(256));
            ir1.template select<32, 1>(256) = (v63_data + v62_data);
            // r1 = ir1
            #pragma unroll
            for (int32_t v65_n1 = 0; v65_n1 < 9; ++v65_n1) {
              int32_t v66_a = v65_n1 * 32;
              tensorforge::intel_esimd::simd<float, 20> v68_data(ir1.template select<20, 1>(v66_a));
              r1.template select<20, 1>(v66_a) = v68_data;
            }
            tensorforge::intel_esimd::simd<float, 288> r4(0.0f);
            // r4 = load{g>r}(glb_m3);
            #pragma unroll
            for (int32_t v70_i1 = 0; v70_i1 < 9; ++v70_i1) {
              tensorforge::intel_esimd::simd<float, 4> v75_data;
              v75_data.copy_from(glb_m3 + ((v70_i1 * 4)));
              r4.template select<4, 1>((v70_i1 * 32)) = v75_data;
            }
            // wait(r2 = load{g>r}(glb_m2););
            tensorforge::intel_esimd::simd<float, 288> r3(0.0f);
            // ir3 = +(r2)
            // [(0, 10), (0, 9)] []
            tensorforge::intel_esimd::simd<float, 288> ir3(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v80_data(r2.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v81_data(ir3.template select<32, 1>(0));
            ir3.template select<32, 1>(0) = (v81_data + v80_data);
            tensorforge::intel_esimd::simd<float, 32> v83_data(r2.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v84_data(ir3.template select<32, 1>(32));
            ir3.template select<32, 1>(32) = (v84_data + v83_data);
            tensorforge::intel_esimd::simd<float, 32> v86_data(r2.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v87_data(ir3.template select<32, 1>(64));
            ir3.template select<32, 1>(64) = (v87_data + v86_data);
            tensorforge::intel_esimd::simd<float, 32> v89_data(r2.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v90_data(ir3.template select<32, 1>(96));
            ir3.template select<32, 1>(96) = (v90_data + v89_data);
            tensorforge::intel_esimd::simd<float, 32> v92_data(r2.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v93_data(ir3.template select<32, 1>(128));
            ir3.template select<32, 1>(128) = (v93_data + v92_data);
            tensorforge::intel_esimd::simd<float, 32> v95_data(r2.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v96_data(ir3.template select<32, 1>(160));
            ir3.template select<32, 1>(160) = (v96_data + v95_data);
            tensorforge::intel_esimd::simd<float, 32> v98_data(r2.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v99_data(ir3.template select<32, 1>(192));
            ir3.template select<32, 1>(192) = (v99_data + v98_data);
            tensorforge::intel_esimd::simd<float, 32> v101_data(r2.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v102_data(ir3.template select<32, 1>(224));
            ir3.template select<32, 1>(224) = (v102_data + v101_data);
            tensorforge::intel_esimd::simd<float, 32> v104_data(r2.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v105_data(ir3.template select<32, 1>(256));
            ir3.template select<32, 1>(256) = (v105_data + v104_data);
            // r3 = ir3 + r1
            #pragma unroll
            for (int32_t v107_n1 = 0; v107_n1 < 9; ++v107_n1) {
              int32_t v108_a = v107_n1 * 32;
              tensorforge::intel_esimd::simd<float, 20> v110_data(ir3.template select<20, 1>(v108_a));
              tensorforge::intel_esimd::simd<float, 20> v111_data(r1.template select<20, 1>(v108_a));
              r3.template select<20, 1>(v108_a) = (v111_data + v110_data);
            }
            tensorforge::intel_esimd::simd<float, 288> r6(0.0f);
            // r6 = load{g>r}(glb_m4);
            #pragma unroll
            for (int32_t v114_i1 = 0; v114_i1 < 9; ++v114_i1) {
              float v116_data = glb_m4[v114_i1];
              r6[(v114_i1 * 32)] = v116_data;
            }
            // wait(r4 = load{g>r}(glb_m3););
            tensorforge::intel_esimd::simd<float, 288> r5(0.0f);
            // ir5 = +(r4)
            // [(0, 4), (0, 9)] []
            tensorforge::intel_esimd::simd<float, 288> ir5(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v121_data(r4.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v122_data(ir5.template select<32, 1>(0));
            ir5.template select<32, 1>(0) = (v122_data + v121_data);
            tensorforge::intel_esimd::simd<float, 32> v124_data(r4.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v125_data(ir5.template select<32, 1>(32));
            ir5.template select<32, 1>(32) = (v125_data + v124_data);
            tensorforge::intel_esimd::simd<float, 32> v127_data(r4.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v128_data(ir5.template select<32, 1>(64));
            ir5.template select<32, 1>(64) = (v128_data + v127_data);
            tensorforge::intel_esimd::simd<float, 32> v130_data(r4.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v131_data(ir5.template select<32, 1>(96));
            ir5.template select<32, 1>(96) = (v131_data + v130_data);
            tensorforge::intel_esimd::simd<float, 32> v133_data(r4.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v134_data(ir5.template select<32, 1>(128));
            ir5.template select<32, 1>(128) = (v134_data + v133_data);
            tensorforge::intel_esimd::simd<float, 32> v136_data(r4.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v137_data(ir5.template select<32, 1>(160));
            ir5.template select<32, 1>(160) = (v137_data + v136_data);
            tensorforge::intel_esimd::simd<float, 32> v139_data(r4.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v140_data(ir5.template select<32, 1>(192));
            ir5.template select<32, 1>(192) = (v140_data + v139_data);
            tensorforge::intel_esimd::simd<float, 32> v142_data(r4.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v143_data(ir5.template select<32, 1>(224));
            ir5.template select<32, 1>(224) = (v143_data + v142_data);
            tensorforge::intel_esimd::simd<float, 32> v145_data(r4.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v146_data(ir5.template select<32, 1>(256));
            ir5.template select<32, 1>(256) = (v146_data + v145_data);
            // r5 = ir5 + r3
            #pragma unroll
            for (int32_t v148_n1 = 0; v148_n1 < 9; ++v148_n1) {
              int32_t v149_a = v148_n1 * 32;
              tensorforge::intel_esimd::simd<float, 20> v151_data(ir5.template select<20, 1>(v149_a));
              tensorforge::intel_esimd::simd<float, 20> v152_data(r3.template select<20, 1>(v149_a));
              r5.template select<20, 1>(v149_a) = (v152_data + v151_data);
            }
            // wait(r6 = load{g>r}(glb_m4););
            tensorforge::intel_esimd::simd<float, 288> r7(0.0f);
            // ir7 = +(r6)
            // [(0, 1), (0, 9)] []
            tensorforge::intel_esimd::simd<float, 288> ir7(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v156_data(r6.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v157_data(ir7.template select<32, 1>(0));
            ir7.template select<32, 1>(0) = (v157_data + v156_data);
            tensorforge::intel_esimd::simd<float, 32> v159_data(r6.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v160_data(ir7.template select<32, 1>(32));
            ir7.template select<32, 1>(32) = (v160_data + v159_data);
            tensorforge::intel_esimd::simd<float, 32> v162_data(r6.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v163_data(ir7.template select<32, 1>(64));
            ir7.template select<32, 1>(64) = (v163_data + v162_data);
            tensorforge::intel_esimd::simd<float, 32> v165_data(r6.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v166_data(ir7.template select<32, 1>(96));
            ir7.template select<32, 1>(96) = (v166_data + v165_data);
            tensorforge::intel_esimd::simd<float, 32> v168_data(r6.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v169_data(ir7.template select<32, 1>(128));
            ir7.template select<32, 1>(128) = (v169_data + v168_data);
            tensorforge::intel_esimd::simd<float, 32> v171_data(r6.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v172_data(ir7.template select<32, 1>(160));
            ir7.template select<32, 1>(160) = (v172_data + v171_data);
            tensorforge::intel_esimd::simd<float, 32> v174_data(r6.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v175_data(ir7.template select<32, 1>(192));
            ir7.template select<32, 1>(192) = (v175_data + v174_data);
            tensorforge::intel_esimd::simd<float, 32> v177_data(r6.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v178_data(ir7.template select<32, 1>(224));
            ir7.template select<32, 1>(224) = (v178_data + v177_data);
            tensorforge::intel_esimd::simd<float, 32> v180_data(r6.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v181_data(ir7.template select<32, 1>(256));
            ir7.template select<32, 1>(256) = (v181_data + v180_data);
            // r7 = ir7 + r5
            #pragma unroll
            for (int32_t v183_n1 = 0; v183_n1 < 9; ++v183_n1) {
              int32_t v184_a = v183_n1 * 32;
              tensorforge::intel_esimd::simd<float, 20> v186_data(ir7.template select<20, 1>(v184_a));
              tensorforge::intel_esimd::simd<float, 20> v187_data(r5.template select<20, 1>(v184_a));
              r7.template select<20, 1>(v184_a) = (v187_data + v186_data);
            }
            // glb_m0 = store{r>g}(r7);
            #pragma unroll
            for (int32_t v189_i1 = 0; v189_i1 < 9; ++v189_i1) {
              tensorforge::intel_esimd::simd<float, 20> v192_data(r7.template select<20, 1>((v189_i1 * 32)));
              v192_data.copy_to(glb_m0 + ((v189_i1 * 20)));
            }
          }
          tensorforge::prefetchRunsL2<720, 360, 144, 36>(&pf_glb_m1[0], &pf_glb_m2[0], &pf_glb_m3[0], &pf_glb_m4[0]);
        }
      }
    });
  });
}


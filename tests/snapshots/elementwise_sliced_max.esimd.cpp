// === base name ===
kernel_a0efa9b331ff376c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a0efa9b331ff376c = {{1, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a0efa9b331ff376c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a0efa9b331ff376c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a0efa9b331ff376c(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 8 - 1) / 8;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a0efa9b331ff376c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a0efa9b331ff376c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a0efa9b331ff376c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a0efa9b331ff376c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      using namespace tensorforge::literals;
      // generated with TensorForge. Version: 0.0.1
      // options: default
      // launch: 32 lanes (64 active) x 8 per block = block 1x8x1, 0 B shared, occupancy grid
      // operands:
      //   m0 64×19(64×19) {0..64}×{0..19} strided
      //   m1 64×19(64×19) {0..64}×{0..19} strided
      // operations:
      //   m0[i,j]@{0..64}×{0..17} += m1[i,j]@{0..64}×{0..17}
      //   t = max(IA, I)
      //   m0[i,j]@{0..64}×{17..19} = t0[i,j]
      // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"IA","bbox":[[0,0],[64,19]],"name":"m0","ordered":false,"parts":1,"shape":[64,19],"variant":false},{"addressing":"strided","alias":"I","bbox":[[0,0],[64,19]],"name":"m1","ordered":false,"parts":1,"shape":[64,19],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[64,17]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,19]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,19]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[64,2]},"kind":"elementwise","op":"MAX","ops":[{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":false,"name":"m0","offset":[0,17],"shape":[64,19]},{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":false,"name":"m1","offset":[0,17],"shape":[64,19]}],"permute":[[0,1],[0,1]],"scalars":[],"target":[[0,1],[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":false,"name":"m0","offset":[0,17],"shape":[64,19]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[64,2]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
      {
        const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
        const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
        const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
        for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
          size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
          const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
          if (allowed) {
            float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 1216 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 1216 + 0 + m1_extraOffset];
            tensorforge::intel_esimd::simd<float, 1216> r0(0.0f);
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v12_i0 = 0; v12_i0 < 2; ++v12_i0) {
              int32_t v14_lead = v12_i0 * 32;
              #pragma unroll
              for (int32_t v13_i1 = 0; v13_i1 < 19; ++v13_i1) {
                int32_t v17_a = v14_lead + (v13_i1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v18_data;
                v18_data.copy_from(glb_m1 + (v17_a));
                r0.template select<32, 1>(v17_a) = v18_data;
              }
            }
            tensorforge::intel_esimd::simd<float, 1088> r1(0.0f);
            // r1 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v21_i0 = 0; v21_i0 < 2; ++v21_i0) {
              int32_t v23_lead = v21_i0 * 32;
              #pragma unroll
              for (int32_t v22_i1 = 0; v22_i1 < 17; ++v22_i1) {
                int32_t v26_a = v23_lead + (v22_i1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v27_data;
                v27_data.copy_from(glb_m0 + (v26_a));
                r1.template select<32, 1>(v26_a) = v27_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m1););
            // wait(r1 = load{g>r}(glb_m0););
            tensorforge::intel_esimd::simd<float, 1088> r2(0.0f);
            // ir2 = +(r0)
            // [(0, 64), (0, 17)] []
            tensorforge::intel_esimd::simd<float, 1088> ir2(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v31_data(r0.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v32_data(ir2.template select<32, 1>(0));
            ir2.template select<32, 1>(0) = (v32_data + v31_data);
            tensorforge::intel_esimd::simd<float, 32> v34_data(r0.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v35_data(ir2.template select<32, 1>(64));
            ir2.template select<32, 1>(64) = (v35_data + v34_data);
            tensorforge::intel_esimd::simd<float, 32> v37_data(r0.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v38_data(ir2.template select<32, 1>(128));
            ir2.template select<32, 1>(128) = (v38_data + v37_data);
            tensorforge::intel_esimd::simd<float, 32> v40_data(r0.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v41_data(ir2.template select<32, 1>(192));
            ir2.template select<32, 1>(192) = (v41_data + v40_data);
            tensorforge::intel_esimd::simd<float, 32> v43_data(r0.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v44_data(ir2.template select<32, 1>(256));
            ir2.template select<32, 1>(256) = (v44_data + v43_data);
            tensorforge::intel_esimd::simd<float, 32> v46_data(r0.template select<32, 1>(320));
            tensorforge::intel_esimd::simd<float, 32> v47_data(ir2.template select<32, 1>(320));
            ir2.template select<32, 1>(320) = (v47_data + v46_data);
            tensorforge::intel_esimd::simd<float, 32> v49_data(r0.template select<32, 1>(384));
            tensorforge::intel_esimd::simd<float, 32> v50_data(ir2.template select<32, 1>(384));
            ir2.template select<32, 1>(384) = (v50_data + v49_data);
            tensorforge::intel_esimd::simd<float, 32> v52_data(r0.template select<32, 1>(448));
            tensorforge::intel_esimd::simd<float, 32> v53_data(ir2.template select<32, 1>(448));
            ir2.template select<32, 1>(448) = (v53_data + v52_data);
            tensorforge::intel_esimd::simd<float, 32> v55_data(r0.template select<32, 1>(512));
            tensorforge::intel_esimd::simd<float, 32> v56_data(ir2.template select<32, 1>(512));
            ir2.template select<32, 1>(512) = (v56_data + v55_data);
            tensorforge::intel_esimd::simd<float, 32> v58_data(r0.template select<32, 1>(576));
            tensorforge::intel_esimd::simd<float, 32> v59_data(ir2.template select<32, 1>(576));
            ir2.template select<32, 1>(576) = (v59_data + v58_data);
            tensorforge::intel_esimd::simd<float, 32> v61_data(r0.template select<32, 1>(640));
            tensorforge::intel_esimd::simd<float, 32> v62_data(ir2.template select<32, 1>(640));
            ir2.template select<32, 1>(640) = (v62_data + v61_data);
            tensorforge::intel_esimd::simd<float, 32> v64_data(r0.template select<32, 1>(704));
            tensorforge::intel_esimd::simd<float, 32> v65_data(ir2.template select<32, 1>(704));
            ir2.template select<32, 1>(704) = (v65_data + v64_data);
            tensorforge::intel_esimd::simd<float, 32> v67_data(r0.template select<32, 1>(768));
            tensorforge::intel_esimd::simd<float, 32> v68_data(ir2.template select<32, 1>(768));
            ir2.template select<32, 1>(768) = (v68_data + v67_data);
            tensorforge::intel_esimd::simd<float, 32> v70_data(r0.template select<32, 1>(832));
            tensorforge::intel_esimd::simd<float, 32> v71_data(ir2.template select<32, 1>(832));
            ir2.template select<32, 1>(832) = (v71_data + v70_data);
            tensorforge::intel_esimd::simd<float, 32> v73_data(r0.template select<32, 1>(896));
            tensorforge::intel_esimd::simd<float, 32> v74_data(ir2.template select<32, 1>(896));
            ir2.template select<32, 1>(896) = (v74_data + v73_data);
            tensorforge::intel_esimd::simd<float, 32> v76_data(r0.template select<32, 1>(960));
            tensorforge::intel_esimd::simd<float, 32> v77_data(ir2.template select<32, 1>(960));
            ir2.template select<32, 1>(960) = (v77_data + v76_data);
            tensorforge::intel_esimd::simd<float, 32> v79_data(r0.template select<32, 1>(1024));
            tensorforge::intel_esimd::simd<float, 32> v80_data(ir2.template select<32, 1>(1024));
            ir2.template select<32, 1>(1024) = (v80_data + v79_data);
            tensorforge::intel_esimd::simd<float, 32> v82_data(r0.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v83_data(ir2.template select<32, 1>(32));
            ir2.template select<32, 1>(32) = (v83_data + v82_data);
            tensorforge::intel_esimd::simd<float, 32> v85_data(r0.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v86_data(ir2.template select<32, 1>(96));
            ir2.template select<32, 1>(96) = (v86_data + v85_data);
            tensorforge::intel_esimd::simd<float, 32> v88_data(r0.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v89_data(ir2.template select<32, 1>(160));
            ir2.template select<32, 1>(160) = (v89_data + v88_data);
            tensorforge::intel_esimd::simd<float, 32> v91_data(r0.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v92_data(ir2.template select<32, 1>(224));
            ir2.template select<32, 1>(224) = (v92_data + v91_data);
            tensorforge::intel_esimd::simd<float, 32> v94_data(r0.template select<32, 1>(288));
            tensorforge::intel_esimd::simd<float, 32> v95_data(ir2.template select<32, 1>(288));
            ir2.template select<32, 1>(288) = (v95_data + v94_data);
            tensorforge::intel_esimd::simd<float, 32> v97_data(r0.template select<32, 1>(352));
            tensorforge::intel_esimd::simd<float, 32> v98_data(ir2.template select<32, 1>(352));
            ir2.template select<32, 1>(352) = (v98_data + v97_data);
            tensorforge::intel_esimd::simd<float, 32> v100_data(r0.template select<32, 1>(416));
            tensorforge::intel_esimd::simd<float, 32> v101_data(ir2.template select<32, 1>(416));
            ir2.template select<32, 1>(416) = (v101_data + v100_data);
            tensorforge::intel_esimd::simd<float, 32> v103_data(r0.template select<32, 1>(480));
            tensorforge::intel_esimd::simd<float, 32> v104_data(ir2.template select<32, 1>(480));
            ir2.template select<32, 1>(480) = (v104_data + v103_data);
            tensorforge::intel_esimd::simd<float, 32> v106_data(r0.template select<32, 1>(544));
            tensorforge::intel_esimd::simd<float, 32> v107_data(ir2.template select<32, 1>(544));
            ir2.template select<32, 1>(544) = (v107_data + v106_data);
            tensorforge::intel_esimd::simd<float, 32> v109_data(r0.template select<32, 1>(608));
            tensorforge::intel_esimd::simd<float, 32> v110_data(ir2.template select<32, 1>(608));
            ir2.template select<32, 1>(608) = (v110_data + v109_data);
            tensorforge::intel_esimd::simd<float, 32> v112_data(r0.template select<32, 1>(672));
            tensorforge::intel_esimd::simd<float, 32> v113_data(ir2.template select<32, 1>(672));
            ir2.template select<32, 1>(672) = (v113_data + v112_data);
            tensorforge::intel_esimd::simd<float, 32> v115_data(r0.template select<32, 1>(736));
            tensorforge::intel_esimd::simd<float, 32> v116_data(ir2.template select<32, 1>(736));
            ir2.template select<32, 1>(736) = (v116_data + v115_data);
            tensorforge::intel_esimd::simd<float, 32> v118_data(r0.template select<32, 1>(800));
            tensorforge::intel_esimd::simd<float, 32> v119_data(ir2.template select<32, 1>(800));
            ir2.template select<32, 1>(800) = (v119_data + v118_data);
            tensorforge::intel_esimd::simd<float, 32> v121_data(r0.template select<32, 1>(864));
            tensorforge::intel_esimd::simd<float, 32> v122_data(ir2.template select<32, 1>(864));
            ir2.template select<32, 1>(864) = (v122_data + v121_data);
            tensorforge::intel_esimd::simd<float, 32> v124_data(r0.template select<32, 1>(928));
            tensorforge::intel_esimd::simd<float, 32> v125_data(ir2.template select<32, 1>(928));
            ir2.template select<32, 1>(928) = (v125_data + v124_data);
            tensorforge::intel_esimd::simd<float, 32> v127_data(r0.template select<32, 1>(992));
            tensorforge::intel_esimd::simd<float, 32> v128_data(ir2.template select<32, 1>(992));
            ir2.template select<32, 1>(992) = (v128_data + v127_data);
            tensorforge::intel_esimd::simd<float, 32> v130_data(r0.template select<32, 1>(1056));
            tensorforge::intel_esimd::simd<float, 32> v131_data(ir2.template select<32, 1>(1056));
            ir2.template select<32, 1>(1056) = (v131_data + v130_data);
            // r2 = ir2 + r1
            #pragma unroll
            for (int32_t v133_n0 = 0; v133_n0 < 2; ++v133_n0) {
              int32_t v135_a = v133_n0 * 32;
              #pragma unroll
              for (int32_t v134_n1 = 0; v134_n1 < 17; ++v134_n1) {
                int32_t v137_a = v135_a + (v134_n1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v138_data(ir2.template select<32, 1>(v137_a));
                tensorforge::intel_esimd::simd<float, 32> v139_data(r1.template select<32, 1>(v137_a));
                r2.template select<32, 1>(v137_a) = (v139_data + v138_data);
              }
            }
            // glb_m0 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v141_i0 = 0; v141_i0 < 2; ++v141_i0) {
              int32_t v143_a = v141_i0 * 32;
              #pragma unroll
              for (int32_t v142_i1 = 0; v142_i1 < 17; ++v142_i1) {
                int32_t v145_a = v143_a + (v142_i1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v146_data(r2.template select<32, 1>(v145_a));
                v146_data.copy_to(glb_m0 + (v145_a));
              }
            }
            tensorforge::intel_esimd::simd<float, 128> r3(0.0f);
            // r3 = max(glb_m0, glb_m1)
            #pragma unroll
            for (int32_t v150_k0 = 0; v150_k0 < 2; ++v150_k0) {
              int32_t v152_lead = v150_k0 * 32;
              #pragma unroll
              for (int32_t v151_k1 = 0; v151_k1 < 2; ++v151_k1) {
                int32_t v156_a = v152_lead + ((v151_k1 + 17) * 64);
                tensorforge::intel_esimd::simd<float, 32> v157_data;
                v157_data.copy_from(glb_m0 + (v156_a));
                tensorforge::intel_esimd::simd<float, 32> v158_data;
                v158_data.copy_from(glb_m1 + (v156_a));
                r3.template select<32, 1>((v152_lead + (v151_k1 * 64))) = (tensorforge::intel_esimd::max(v157_data, v158_data));
              }
            }
            tensorforge::intel_esimd::simd<float, 128> r4(0.0f);
            // ir4 = +(r3)
            // [(0, 64), (0, 2)] []
            tensorforge::intel_esimd::simd<float, 128> ir4(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v164_data(r3.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v165_data(ir4.template select<32, 1>(0));
            ir4.template select<32, 1>(0) = (v165_data + v164_data);
            tensorforge::intel_esimd::simd<float, 32> v167_data(r3.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v168_data(ir4.template select<32, 1>(64));
            ir4.template select<32, 1>(64) = (v168_data + v167_data);
            tensorforge::intel_esimd::simd<float, 32> v170_data(r3.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v171_data(ir4.template select<32, 1>(32));
            ir4.template select<32, 1>(32) = (v171_data + v170_data);
            tensorforge::intel_esimd::simd<float, 32> v173_data(r3.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v174_data(ir4.template select<32, 1>(96));
            ir4.template select<32, 1>(96) = (v174_data + v173_data);
            // r4 = ir4
            #pragma unroll
            for (int32_t v176_n0 = 0; v176_n0 < 2; ++v176_n0) {
              int32_t v178_a = v176_n0 * 32;
              #pragma unroll
              for (int32_t v177_n1 = 0; v177_n1 < 2; ++v177_n1) {
                int32_t v180_a = v178_a + (v177_n1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v181_data(ir4.template select<32, 1>(v180_a));
                r4.template select<32, 1>(v180_a) = v181_data;
              }
            }
            // glb_m0 = store{r>g}(r4);
            #pragma unroll
            for (int32_t v182_i0 = 0; v182_i0 < 2; ++v182_i0) {
              int32_t v184_a = v182_i0 * 32;
              #pragma unroll
              for (int32_t v183_i1 = 0; v183_i1 < 2; ++v183_i1) {
                tensorforge::intel_esimd::simd<float, 32> v187_data(r4.template select<32, 1>((v184_a + (v183_i1 * 64))));
                v187_data.copy_to(glb_m0 + ((v184_a + ((v183_i1 + 17) * 64))));
              }
            }
          }
        }
      }
    });
  });
}


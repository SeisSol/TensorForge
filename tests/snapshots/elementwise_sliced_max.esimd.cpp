// === base name ===
kernel_ffceafdb6e1f3bfe

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ffceafdb6e1f3bfe = {{1, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ffceafdb6e1f3bfe(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ffceafdb6e1f3bfe(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ffceafdb6e1f3bfe(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_ffceafdb6e1f3bfe(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ffceafdb6e1f3bfe(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ffceafdb6e1f3bfe(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ffceafdb6e1f3bfe(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
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
        for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
          size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
          const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
          if (allowed) {
            float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 1216 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 1216 + 0 + m1_extraOffset];
            tensorforge::intel_esimd::simd<float, 1216> r0(0.0f);
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v18_i0 = 0; v18_i0 < 2; ++v18_i0) {
              int32_t v20_lead = v18_i0 * 32;
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 19; ++v19_i1) {
                int32_t v23_a = v20_lead + (v19_i1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v24_data;
                v24_data.copy_from(glb_m1 + (v23_a));
                r0.template select<32, 1>(v23_a) = v24_data;
              }
            }
            tensorforge::intel_esimd::simd<float, 1088> r1(0.0f);
            // r1 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v27_i0 = 0; v27_i0 < 2; ++v27_i0) {
              int32_t v29_lead = v27_i0 * 32;
              #pragma unroll
              for (int32_t v28_i1 = 0; v28_i1 < 17; ++v28_i1) {
                int32_t v32_a = v29_lead + (v28_i1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v33_data;
                v33_data.copy_from(glb_m0 + (v32_a));
                r1.template select<32, 1>(v32_a) = v33_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m1););
            // wait(r1 = load{g>r}(glb_m0););
            tensorforge::intel_esimd::simd<float, 1088> r2(0.0f);
            // ir2 = +(r0)
            // [(0, 64), (0, 17)] []
            tensorforge::intel_esimd::simd<float, 1088> ir2(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v37_data(r0.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v38_data(ir2.template select<32, 1>(0));
            ir2.template select<32, 1>(0) = (v38_data + v37_data);
            tensorforge::intel_esimd::simd<float, 32> v40_data(r0.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v41_data(ir2.template select<32, 1>(64));
            ir2.template select<32, 1>(64) = (v41_data + v40_data);
            tensorforge::intel_esimd::simd<float, 32> v43_data(r0.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v44_data(ir2.template select<32, 1>(128));
            ir2.template select<32, 1>(128) = (v44_data + v43_data);
            tensorforge::intel_esimd::simd<float, 32> v46_data(r0.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v47_data(ir2.template select<32, 1>(192));
            ir2.template select<32, 1>(192) = (v47_data + v46_data);
            tensorforge::intel_esimd::simd<float, 32> v49_data(r0.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v50_data(ir2.template select<32, 1>(256));
            ir2.template select<32, 1>(256) = (v50_data + v49_data);
            tensorforge::intel_esimd::simd<float, 32> v52_data(r0.template select<32, 1>(320));
            tensorforge::intel_esimd::simd<float, 32> v53_data(ir2.template select<32, 1>(320));
            ir2.template select<32, 1>(320) = (v53_data + v52_data);
            tensorforge::intel_esimd::simd<float, 32> v55_data(r0.template select<32, 1>(384));
            tensorforge::intel_esimd::simd<float, 32> v56_data(ir2.template select<32, 1>(384));
            ir2.template select<32, 1>(384) = (v56_data + v55_data);
            tensorforge::intel_esimd::simd<float, 32> v58_data(r0.template select<32, 1>(448));
            tensorforge::intel_esimd::simd<float, 32> v59_data(ir2.template select<32, 1>(448));
            ir2.template select<32, 1>(448) = (v59_data + v58_data);
            tensorforge::intel_esimd::simd<float, 32> v61_data(r0.template select<32, 1>(512));
            tensorforge::intel_esimd::simd<float, 32> v62_data(ir2.template select<32, 1>(512));
            ir2.template select<32, 1>(512) = (v62_data + v61_data);
            tensorforge::intel_esimd::simd<float, 32> v64_data(r0.template select<32, 1>(576));
            tensorforge::intel_esimd::simd<float, 32> v65_data(ir2.template select<32, 1>(576));
            ir2.template select<32, 1>(576) = (v65_data + v64_data);
            tensorforge::intel_esimd::simd<float, 32> v67_data(r0.template select<32, 1>(640));
            tensorforge::intel_esimd::simd<float, 32> v68_data(ir2.template select<32, 1>(640));
            ir2.template select<32, 1>(640) = (v68_data + v67_data);
            tensorforge::intel_esimd::simd<float, 32> v70_data(r0.template select<32, 1>(704));
            tensorforge::intel_esimd::simd<float, 32> v71_data(ir2.template select<32, 1>(704));
            ir2.template select<32, 1>(704) = (v71_data + v70_data);
            tensorforge::intel_esimd::simd<float, 32> v73_data(r0.template select<32, 1>(768));
            tensorforge::intel_esimd::simd<float, 32> v74_data(ir2.template select<32, 1>(768));
            ir2.template select<32, 1>(768) = (v74_data + v73_data);
            tensorforge::intel_esimd::simd<float, 32> v76_data(r0.template select<32, 1>(832));
            tensorforge::intel_esimd::simd<float, 32> v77_data(ir2.template select<32, 1>(832));
            ir2.template select<32, 1>(832) = (v77_data + v76_data);
            tensorforge::intel_esimd::simd<float, 32> v79_data(r0.template select<32, 1>(896));
            tensorforge::intel_esimd::simd<float, 32> v80_data(ir2.template select<32, 1>(896));
            ir2.template select<32, 1>(896) = (v80_data + v79_data);
            tensorforge::intel_esimd::simd<float, 32> v82_data(r0.template select<32, 1>(960));
            tensorforge::intel_esimd::simd<float, 32> v83_data(ir2.template select<32, 1>(960));
            ir2.template select<32, 1>(960) = (v83_data + v82_data);
            tensorforge::intel_esimd::simd<float, 32> v85_data(r0.template select<32, 1>(1024));
            tensorforge::intel_esimd::simd<float, 32> v86_data(ir2.template select<32, 1>(1024));
            ir2.template select<32, 1>(1024) = (v86_data + v85_data);
            tensorforge::intel_esimd::simd<float, 32> v88_data(r0.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v89_data(ir2.template select<32, 1>(32));
            ir2.template select<32, 1>(32) = (v89_data + v88_data);
            tensorforge::intel_esimd::simd<float, 32> v91_data(r0.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v92_data(ir2.template select<32, 1>(96));
            ir2.template select<32, 1>(96) = (v92_data + v91_data);
            tensorforge::intel_esimd::simd<float, 32> v94_data(r0.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v95_data(ir2.template select<32, 1>(160));
            ir2.template select<32, 1>(160) = (v95_data + v94_data);
            tensorforge::intel_esimd::simd<float, 32> v97_data(r0.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v98_data(ir2.template select<32, 1>(224));
            ir2.template select<32, 1>(224) = (v98_data + v97_data);
            tensorforge::intel_esimd::simd<float, 32> v100_data(r0.template select<32, 1>(288));
            tensorforge::intel_esimd::simd<float, 32> v101_data(ir2.template select<32, 1>(288));
            ir2.template select<32, 1>(288) = (v101_data + v100_data);
            tensorforge::intel_esimd::simd<float, 32> v103_data(r0.template select<32, 1>(352));
            tensorforge::intel_esimd::simd<float, 32> v104_data(ir2.template select<32, 1>(352));
            ir2.template select<32, 1>(352) = (v104_data + v103_data);
            tensorforge::intel_esimd::simd<float, 32> v106_data(r0.template select<32, 1>(416));
            tensorforge::intel_esimd::simd<float, 32> v107_data(ir2.template select<32, 1>(416));
            ir2.template select<32, 1>(416) = (v107_data + v106_data);
            tensorforge::intel_esimd::simd<float, 32> v109_data(r0.template select<32, 1>(480));
            tensorforge::intel_esimd::simd<float, 32> v110_data(ir2.template select<32, 1>(480));
            ir2.template select<32, 1>(480) = (v110_data + v109_data);
            tensorforge::intel_esimd::simd<float, 32> v112_data(r0.template select<32, 1>(544));
            tensorforge::intel_esimd::simd<float, 32> v113_data(ir2.template select<32, 1>(544));
            ir2.template select<32, 1>(544) = (v113_data + v112_data);
            tensorforge::intel_esimd::simd<float, 32> v115_data(r0.template select<32, 1>(608));
            tensorforge::intel_esimd::simd<float, 32> v116_data(ir2.template select<32, 1>(608));
            ir2.template select<32, 1>(608) = (v116_data + v115_data);
            tensorforge::intel_esimd::simd<float, 32> v118_data(r0.template select<32, 1>(672));
            tensorforge::intel_esimd::simd<float, 32> v119_data(ir2.template select<32, 1>(672));
            ir2.template select<32, 1>(672) = (v119_data + v118_data);
            tensorforge::intel_esimd::simd<float, 32> v121_data(r0.template select<32, 1>(736));
            tensorforge::intel_esimd::simd<float, 32> v122_data(ir2.template select<32, 1>(736));
            ir2.template select<32, 1>(736) = (v122_data + v121_data);
            tensorforge::intel_esimd::simd<float, 32> v124_data(r0.template select<32, 1>(800));
            tensorforge::intel_esimd::simd<float, 32> v125_data(ir2.template select<32, 1>(800));
            ir2.template select<32, 1>(800) = (v125_data + v124_data);
            tensorforge::intel_esimd::simd<float, 32> v127_data(r0.template select<32, 1>(864));
            tensorforge::intel_esimd::simd<float, 32> v128_data(ir2.template select<32, 1>(864));
            ir2.template select<32, 1>(864) = (v128_data + v127_data);
            tensorforge::intel_esimd::simd<float, 32> v130_data(r0.template select<32, 1>(928));
            tensorforge::intel_esimd::simd<float, 32> v131_data(ir2.template select<32, 1>(928));
            ir2.template select<32, 1>(928) = (v131_data + v130_data);
            tensorforge::intel_esimd::simd<float, 32> v133_data(r0.template select<32, 1>(992));
            tensorforge::intel_esimd::simd<float, 32> v134_data(ir2.template select<32, 1>(992));
            ir2.template select<32, 1>(992) = (v134_data + v133_data);
            tensorforge::intel_esimd::simd<float, 32> v136_data(r0.template select<32, 1>(1056));
            tensorforge::intel_esimd::simd<float, 32> v137_data(ir2.template select<32, 1>(1056));
            ir2.template select<32, 1>(1056) = (v137_data + v136_data);
            // r2 = ir2 + r1
            #pragma unroll
            for (int32_t v139_n0 = 0; v139_n0 < 2; ++v139_n0) {
              int32_t v141_a = v139_n0 * 32;
              #pragma unroll
              for (int32_t v140_n1 = 0; v140_n1 < 17; ++v140_n1) {
                int32_t v143_a = v141_a + (v140_n1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v144_data(ir2.template select<32, 1>(v143_a));
                tensorforge::intel_esimd::simd<float, 32> v145_data(r1.template select<32, 1>(v143_a));
                r2.template select<32, 1>(v143_a) = (v145_data + v144_data);
              }
            }
            // glb_m0 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v147_i0 = 0; v147_i0 < 2; ++v147_i0) {
              int32_t v149_a = v147_i0 * 32;
              #pragma unroll
              for (int32_t v148_i1 = 0; v148_i1 < 17; ++v148_i1) {
                int32_t v151_a = v149_a + (v148_i1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v152_data(r2.template select<32, 1>(v151_a));
                v152_data.copy_to(glb_m0 + (v151_a));
              }
            }
            tensorforge::intel_esimd::simd<float, 128> r3(0.0f);
            // r3 = max(glb_m0, glb_m1)
            #pragma unroll
            for (int32_t v156_k0 = 0; v156_k0 < 2; ++v156_k0) {
              int32_t v158_lead = v156_k0 * 32;
              #pragma unroll
              for (int32_t v157_k1 = 0; v157_k1 < 2; ++v157_k1) {
                int32_t v162_a = v158_lead + ((v157_k1 + 17) * 64);
                tensorforge::intel_esimd::simd<float, 32> v163_data;
                v163_data.copy_from(glb_m0 + (v162_a));
                tensorforge::intel_esimd::simd<float, 32> v164_data;
                v164_data.copy_from(glb_m1 + (v162_a));
                r3.template select<32, 1>((v158_lead + (v157_k1 * 64))) = (tensorforge::intel_esimd::max(v163_data, v164_data));
              }
            }
            tensorforge::intel_esimd::simd<float, 128> r4(0.0f);
            // ir4 = +(r3)
            // [(0, 64), (0, 2)] []
            tensorforge::intel_esimd::simd<float, 128> ir4(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v170_data(r3.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v171_data(ir4.template select<32, 1>(0));
            ir4.template select<32, 1>(0) = (v171_data + v170_data);
            tensorforge::intel_esimd::simd<float, 32> v173_data(r3.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v174_data(ir4.template select<32, 1>(64));
            ir4.template select<32, 1>(64) = (v174_data + v173_data);
            tensorforge::intel_esimd::simd<float, 32> v176_data(r3.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v177_data(ir4.template select<32, 1>(32));
            ir4.template select<32, 1>(32) = (v177_data + v176_data);
            tensorforge::intel_esimd::simd<float, 32> v179_data(r3.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v180_data(ir4.template select<32, 1>(96));
            ir4.template select<32, 1>(96) = (v180_data + v179_data);
            // r4 = ir4
            #pragma unroll
            for (int32_t v182_n0 = 0; v182_n0 < 2; ++v182_n0) {
              int32_t v184_a = v182_n0 * 32;
              #pragma unroll
              for (int32_t v183_n1 = 0; v183_n1 < 2; ++v183_n1) {
                int32_t v186_a = v184_a + (v183_n1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v187_data(ir4.template select<32, 1>(v186_a));
                r4.template select<32, 1>(v186_a) = v187_data;
              }
            }
            // glb_m0 = store{r>g}(r4);
            #pragma unroll
            for (int32_t v188_i0 = 0; v188_i0 < 2; ++v188_i0) {
              int32_t v190_a = v188_i0 * 32;
              #pragma unroll
              for (int32_t v189_i1 = 0; v189_i1 < 2; ++v189_i1) {
                tensorforge::intel_esimd::simd<float, 32> v193_data(r4.template select<32, 1>((v190_a + (v189_i1 * 64))));
                v193_data.copy_to(glb_m0 + ((v190_a + ((v189_i1 + 17) * 64))));
              }
            }
          }
        }
      }
    });
  });
}


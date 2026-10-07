// === base name ===
kernel_ad52b05e9fec1afd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ad52b05e9fec1afd = {{8, 2, 1}, 8, 8, 1, 2, 576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ad52b05e9fec1afd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ad52b05e9fec1afd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ad52b05e9fec1afd(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (8, 2, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 2 - 1) / 2;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 8;
  config.block[1] = 2;
  config.block[2] = 1;
  config.sharedMemBytes = 144 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ad52b05e9fec1afd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ad52b05e9fec1afd(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ad52b05e9fec1afd(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ad52b05e9fec1afd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (144, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 2 per block = block 8x2x1, 576 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×4(8×4) {0..8}×{0..4} strided
        //   m2 8×4(8×4) {0..8}×{0..4} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j]@{0..8}×{0..4} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{0..8}×{4..8} = m0[i,k] × m2[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":144}],"shared_bytes":576,"shared_elements":144,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[72 * item.get_local_id(1) + 0];
          float * __restrict__ s0 = &localShrMem0[0];
          size_t v9_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v27_lead = item.get_local_id(2) % 8;
          for (size_t v10_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v10_batchIdGroup0 < numElements0; v10_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_row = v10_batchIdGroup0 + v9_batchIdLane0;
            const bool batchIdActive0 = v11_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v11_row]));
            size_t v13_batchId0 = batchIdActive0 ? v11_row : v10_batchIdGroup0;
            size_t v14_ahead1 = v13_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v16_batchId1 = (v14_ahead1 < numElements0) ? v14_ahead1 : v13_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v13_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v13_batchId0 * 32 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v13_batchId0 * 32 + 0 + m2_extraOffset];
            float *const __restrict__ glb_m3 = &m3[v13_batchId0 * 64 + 0 + m3_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
              int32_t v31_lead = v27_lead + (v28_i0 * 8);
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
                float v34_data = glb_m0[(v31_lead + (v29_i1 * 8))];
                r0[(v28_i0 + v29_i1)] = v34_data;
              }
            }
            float r1[4]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v37_i0 = 0; v37_i0 < 1; ++v37_i0) {
              int32_t v40_lead = v27_lead + (v37_i0 * 8);
              #pragma unroll
              for (int32_t v38_i1 = 0; v38_i1 < 4; ++v38_i1) {
                float v43_data = glb_m1[(v40_lead + (v38_i1 * 8))];
                r1[(v37_i0 + v38_i1)] = v43_data;
              }
            }
            float r3[4]{};
            // r3 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v250_i0 = 0; v250_i0 < 1; ++v250_i0) {
              int32_t v253_lead = v27_lead + (v250_i0 * 8);
              #pragma unroll
              for (int32_t v251_i1 = 0; v251_i1 < 4; ++v251_i1) {
                float v256_data = glb_m2[(v253_lead + (v251_i1 * 8))];
                r3[(v250_i0 + v251_i1)] = v256_data;
              }
            }
            float r2[4]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 4)] [(0, 8)]
            float v46_data = r0[0];
            float v47_data = r1[0];
            float v50_data = r2[0];
            r2[0] = (v50_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v53_data = r1[1];
            float v56_data = r2[1];
            r2[1] = (v56_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v59_data = r1[2];
            float v62_data = r2[2];
            r2[2] = (v62_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v65_data = r1[3];
            float v68_data = r2[3];
            r2[3] = (v68_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v70_data = r0[1];
            float v74_data = r2[0];
            r2[0] = (v74_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v80_data = r2[1];
            r2[1] = (v80_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v86_data = r2[2];
            r2[2] = (v86_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v92_data = r2[3];
            r2[3] = (v92_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v94_data = r0[2];
            float v98_data = r2[0];
            r2[0] = (v98_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v104_data = r2[1];
            r2[1] = (v104_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v110_data = r2[2];
            r2[2] = (v110_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v116_data = r2[3];
            r2[3] = (v116_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v118_data = r0[3];
            float v122_data = r2[0];
            r2[0] = (v122_data + (v118_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v128_data = r2[1];
            r2[1] = (v128_data + (v118_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v134_data = r2[2];
            r2[2] = (v134_data + (v118_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v140_data = r2[3];
            r2[3] = (v140_data + (v118_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v142_data = r0[4];
            float v146_data = r2[0];
            r2[0] = (v146_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v152_data = r2[1];
            r2[1] = (v152_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v158_data = r2[2];
            r2[2] = (v158_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v164_data = r2[3];
            r2[3] = (v164_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v166_data = r0[5];
            float v170_data = r2[0];
            r2[0] = (v170_data + (v166_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v176_data = r2[1];
            r2[1] = (v176_data + (v166_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v182_data = r2[2];
            r2[2] = (v182_data + (v166_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v188_data = r2[3];
            r2[3] = (v188_data + (v166_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v190_data = r0[6];
            float v194_data = r2[0];
            r2[0] = (v194_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v200_data = r2[1];
            r2[1] = (v200_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v206_data = r2[2];
            r2[2] = (v206_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v212_data = r2[3];
            r2[3] = (v212_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v214_data = r0[7];
            float v218_data = r2[0];
            r2[0] = (v218_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v224_data = r2[1];
            r2[1] = (v224_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v230_data = r2[2];
            r2[2] = (v230_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v236_data = r2[3];
            r2[3] = (v236_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // s0 = store{r>s}(localShrMem0, r2);
            #pragma unroll
            for (int32_t v238_i0 = 0; v238_i0 < 1; ++v238_i0) {
              int32_t v243_lead = v27_lead + (v238_i0 * 8);
              #pragma unroll
              for (int32_t v239_i1 = 0; v239_i1 < 4; ++v239_i1) {
                float v241_data = r2[(v238_i0 + v239_i1)];
                int32_t v245_a = v243_lead + (v239_i1 * 8);
                s0[(v245_a ^ ((v245_a >> 5) & 31))] = v241_data;
              }
            }
            float r4[4]{};
            // ir4 = +(r0 * r3)
            // [(0, 8), (0, 4)] [(0, 8)]
            float ir4[4]{};
            float v261_data = r3[0];
            float v264_data = ir4[0];
            ir4[0] = (v264_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v267_data = r3[1];
            float v270_data = ir4[1];
            ir4[1] = (v270_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v273_data = r3[2];
            float v276_data = ir4[2];
            ir4[2] = (v276_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v279_data = r3[3];
            float v282_data = ir4[3];
            ir4[3] = (v282_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v288_data = ir4[0];
            ir4[0] = (v288_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v294_data = ir4[1];
            ir4[1] = (v294_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v300_data = ir4[2];
            ir4[2] = (v300_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v306_data = ir4[3];
            ir4[3] = (v306_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v312_data = ir4[0];
            ir4[0] = (v312_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v318_data = ir4[1];
            ir4[1] = (v318_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v324_data = ir4[2];
            ir4[2] = (v324_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v330_data = ir4[3];
            ir4[3] = (v330_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v336_data = ir4[0];
            ir4[0] = (v336_data + (v118_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v342_data = ir4[1];
            ir4[1] = (v342_data + (v118_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v348_data = ir4[2];
            ir4[2] = (v348_data + (v118_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v354_data = ir4[3];
            ir4[3] = (v354_data + (v118_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v360_data = ir4[0];
            ir4[0] = (v360_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v366_data = ir4[1];
            ir4[1] = (v366_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v372_data = ir4[2];
            ir4[2] = (v372_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v378_data = ir4[3];
            ir4[3] = (v378_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v384_data = ir4[0];
            ir4[0] = (v384_data + (v166_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v390_data = ir4[1];
            ir4[1] = (v390_data + (v166_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v396_data = ir4[2];
            ir4[2] = (v396_data + (v166_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v402_data = ir4[3];
            ir4[3] = (v402_data + (v166_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v408_data = ir4[0];
            ir4[0] = (v408_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v414_data = ir4[1];
            ir4[1] = (v414_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v420_data = ir4[2];
            ir4[2] = (v420_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v426_data = ir4[3];
            ir4[3] = (v426_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v432_data = ir4[0];
            ir4[0] = (v432_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v438_data = ir4[1];
            ir4[1] = (v438_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v444_data = ir4[2];
            ir4[2] = (v444_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v450_data = ir4[3];
            ir4[3] = (v450_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r4 = ir4
            #pragma unroll
            for (int32_t v452_n0 = 0; v452_n0 < 1; ++v452_n0) {
              #pragma unroll
              for (int32_t v453_n1 = 0; v453_n1 < 4; ++v453_n1) {
                int32_t v454_a = v452_n0 + v453_n1;
                float v455_data = ir4[v454_a];
                r4[v454_a] = v455_data;
              }
            }
            // s0 = store{r>s}(localShrMem0, r4);
            #pragma unroll
            for (int32_t v456_i0 = 0; v456_i0 < 1; ++v456_i0) {
              int32_t v461_lead = v27_lead + (v456_i0 * 8);
              #pragma unroll
              for (int32_t v457_i1 = 0; v457_i1 < 4; ++v457_i1) {
                float v459_data = r4[(v456_i0 + v457_i1)];
                int32_t v464_a = v461_lead + ((v457_i1 + 4) * 8);
                s0[(v464_a ^ ((v464_a >> 5) & 31))] = v459_data;
              }
            }
            // glb_m3 = abs(s0)
            item.barrier();
            #pragma unroll
            for (int32_t v468_k0 = 0; v468_k0 < 1; ++v468_k0) {
              int32_t v471_lead = v27_lead + (v468_k0 * 8);
              #pragma unroll
              for (int32_t v469_k1 = 0; v469_k1 < 8; ++v469_k1) {
                int32_t v473_a = v471_lead + (v469_k1 * 8);
                float v477_data = s0[(v473_a ^ ((v473_a >> 5) & 31))];
                float v478_e = sycl::fabs(v477_data);
                if (batchIdActive0) {
                  glb_m3[v473_a] = v478_e;
                }
              }
            }
            item.barrier();
          }
        }
      });
    }
  });
}


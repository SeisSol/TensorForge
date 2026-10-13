// === base name ===
kernel_723b888b327d5f4e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_723b888b327d5f4e = {{8, 2, 1}, 8, 8, 1, 2, 576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_723b888b327d5f4e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_723b888b327d5f4e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_723b888b327d5f4e(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_723b888b327d5f4e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_723b888b327d5f4e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_723b888b327d5f4e(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_723b888b327d5f4e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
          size_t v12_batchIdGroup0First = ((item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) < numElements0) ? (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) : 0u;
          int32_t v30_lead = item.get_local_id(2) % 8;
          float r0[8]{};
          size_t v485_row = v12_batchIdGroup0First + v9_batchIdLane0;
          const bool peel_batchIdActive0 = v485_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v485_row]));
          const float *const __restrict__ peel_glb_m0 = &m0[(peel_batchIdActive0 ? v485_row : v12_batchIdGroup0First) * 64 + 0 + m0_extraOffset];
          // r0 = load{g>r}(peel_glb_m0);
          #pragma unroll
          for (int32_t v493_i0 = 0; v493_i0 < 1; ++v493_i0) {
            int32_t v495_lead = v30_lead + (v493_i0 * 8);
            #pragma unroll
            for (int32_t v496_i1 = 0; v496_i1 < 8; ++v496_i1) {
              float v499_data = peel_glb_m0[(v495_lead + (v496_i1 * 8))];
              r0[(v493_i0 + v496_i1)] = v499_data;
            }
          }
          for (size_t v13_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v13_batchIdGroup0 < numElements0; v13_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v482_groupNextAhead = v13_batchIdGroup0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v484_groupNext = (v482_groupNextAhead < numElements0) ? v482_groupNextAhead : v13_batchIdGroup0;
            size_t v489_row = v484_groupNext + v9_batchIdLane0;
            const bool wrap_batchIdActive0 = v489_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v489_row]));
            const float *const __restrict__ wrap_glb_m0 = &m0[(wrap_batchIdActive0 ? v489_row : v484_groupNext) * 64 + 0 + m0_extraOffset];
            size_t v14_row = v13_batchIdGroup0 + v9_batchIdLane0;
            const bool batchIdActive0 = v14_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v14_row]));
            size_t v16_batchId0 = batchIdActive0 ? v14_row : v13_batchIdGroup0;
            size_t v17_ahead1 = v16_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v19_batchId1 = (v17_ahead1 < numElements0) ? v17_ahead1 : v16_batchId0;
            const float *const __restrict__ glb_m1 = &m1[v16_batchId0 * 32 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v16_batchId0 * 32 + 0 + m2_extraOffset];
            float *const __restrict__ glb_m3 = &m3[v16_batchId0 * 64 + 0 + m3_extraOffset];
            float r1[4]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v40_i0 = 0; v40_i0 < 1; ++v40_i0) {
              int32_t v43_lead = v30_lead + (v40_i0 * 8);
              #pragma unroll
              for (int32_t v41_i1 = 0; v41_i1 < 4; ++v41_i1) {
                float v46_data = glb_m1[(v43_lead + (v41_i1 * 8))];
                r1[(v40_i0 + v41_i1)] = v46_data;
              }
            }
            float r3[4]{};
            // r3 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v253_i0 = 0; v253_i0 < 1; ++v253_i0) {
              int32_t v256_lead = v30_lead + (v253_i0 * 8);
              #pragma unroll
              for (int32_t v254_i1 = 0; v254_i1 < 4; ++v254_i1) {
                float v259_data = glb_m2[(v256_lead + (v254_i1 * 8))];
                r3[(v253_i0 + v254_i1)] = v259_data;
              }
            }
            float r2[4]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 4)] [(0, 8)]
            float v49_data = r0[0];
            float v50_data = r1[0];
            float v53_data = r2[0];
            r2[0] = (v53_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v56_data = r1[1];
            float v59_data = r2[1];
            r2[1] = (v59_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v62_data = r1[2];
            float v65_data = r2[2];
            r2[2] = (v65_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v68_data = r1[3];
            float v71_data = r2[3];
            r2[3] = (v71_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v73_data = r0[1];
            float v77_data = r2[0];
            r2[0] = (v77_data + (v73_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v83_data = r2[1];
            r2[1] = (v83_data + (v73_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v89_data = r2[2];
            r2[2] = (v89_data + (v73_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v95_data = r2[3];
            r2[3] = (v95_data + (v73_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v97_data = r0[2];
            float v101_data = r2[0];
            r2[0] = (v101_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v107_data = r2[1];
            r2[1] = (v107_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v113_data = r2[2];
            r2[2] = (v113_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v119_data = r2[3];
            r2[3] = (v119_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v121_data = r0[3];
            float v125_data = r2[0];
            r2[0] = (v125_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v131_data = r2[1];
            r2[1] = (v131_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v137_data = r2[2];
            r2[2] = (v137_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v143_data = r2[3];
            r2[3] = (v143_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v145_data = r0[4];
            float v149_data = r2[0];
            r2[0] = (v149_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v155_data = r2[1];
            r2[1] = (v155_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v161_data = r2[2];
            r2[2] = (v161_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v167_data = r2[3];
            r2[3] = (v167_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v169_data = r0[5];
            float v173_data = r2[0];
            r2[0] = (v173_data + (v169_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v179_data = r2[1];
            r2[1] = (v179_data + (v169_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v185_data = r2[2];
            r2[2] = (v185_data + (v169_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v191_data = r2[3];
            r2[3] = (v191_data + (v169_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v193_data = r0[6];
            float v197_data = r2[0];
            r2[0] = (v197_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v203_data = r2[1];
            r2[1] = (v203_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v209_data = r2[2];
            r2[2] = (v209_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v215_data = r2[3];
            r2[3] = (v215_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v217_data = r0[7];
            // r0 = load{g>r}(wrap_glb_m0);
            #pragma unroll
            for (int32_t v501_i0 = 0; v501_i0 < 1; ++v501_i0) {
              int32_t v503_lead = v30_lead + (v501_i0 * 8);
              #pragma unroll
              for (int32_t v504_i1 = 0; v504_i1 < 8; ++v504_i1) {
                float v507_data = wrap_glb_m0[(v503_lead + (v504_i1 * 8))];
                r0[(v501_i0 + v504_i1)] = v507_data;
              }
            }
            float v221_data = r2[0];
            r2[0] = (v221_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v227_data = r2[1];
            r2[1] = (v227_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v233_data = r2[2];
            r2[2] = (v233_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v239_data = r2[3];
            r2[3] = (v239_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // s0 = store{r>s}(localShrMem0, r2);
            #pragma unroll
            for (int32_t v241_i0 = 0; v241_i0 < 1; ++v241_i0) {
              int32_t v246_lead = v30_lead + (v241_i0 * 8);
              #pragma unroll
              for (int32_t v242_i1 = 0; v242_i1 < 4; ++v242_i1) {
                float v244_data = r2[(v241_i0 + v242_i1)];
                int32_t v248_a = v246_lead + (v242_i1 * 8);
                s0[(v248_a ^ ((v248_a >> 5) & 31))] = v244_data;
              }
            }
            float r4[4]{};
            // ir4 = +(r0 * r3)
            // [(0, 8), (0, 4)] [(0, 8)]
            float ir4[4]{};
            float v264_data = r3[0];
            float v267_data = ir4[0];
            ir4[0] = (v267_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v270_data = r3[1];
            float v273_data = ir4[1];
            ir4[1] = (v273_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v276_data = r3[2];
            float v279_data = ir4[2];
            ir4[2] = (v279_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v282_data = r3[3];
            float v285_data = ir4[3];
            ir4[3] = (v285_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v291_data = ir4[0];
            ir4[0] = (v291_data + (v73_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v297_data = ir4[1];
            ir4[1] = (v297_data + (v73_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v303_data = ir4[2];
            ir4[2] = (v303_data + (v73_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v309_data = ir4[3];
            ir4[3] = (v309_data + (v73_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v315_data = ir4[0];
            ir4[0] = (v315_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v321_data = ir4[1];
            ir4[1] = (v321_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v327_data = ir4[2];
            ir4[2] = (v327_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v333_data = ir4[3];
            ir4[3] = (v333_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v339_data = ir4[0];
            ir4[0] = (v339_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v345_data = ir4[1];
            ir4[1] = (v345_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v351_data = ir4[2];
            ir4[2] = (v351_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v357_data = ir4[3];
            ir4[3] = (v357_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v363_data = ir4[0];
            ir4[0] = (v363_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v369_data = ir4[1];
            ir4[1] = (v369_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v375_data = ir4[2];
            ir4[2] = (v375_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v381_data = ir4[3];
            ir4[3] = (v381_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v387_data = ir4[0];
            ir4[0] = (v387_data + (v169_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v393_data = ir4[1];
            ir4[1] = (v393_data + (v169_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v399_data = ir4[2];
            ir4[2] = (v399_data + (v169_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v405_data = ir4[3];
            ir4[3] = (v405_data + (v169_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v411_data = ir4[0];
            ir4[0] = (v411_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v417_data = ir4[1];
            ir4[1] = (v417_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v423_data = ir4[2];
            ir4[2] = (v423_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v429_data = ir4[3];
            ir4[3] = (v429_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v435_data = ir4[0];
            ir4[0] = (v435_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v441_data = ir4[1];
            ir4[1] = (v441_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v447_data = ir4[2];
            ir4[2] = (v447_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v453_data = ir4[3];
            ir4[3] = (v453_data + (v217_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r4 = ir4
            #pragma unroll
            for (int32_t v455_n0 = 0; v455_n0 < 1; ++v455_n0) {
              #pragma unroll
              for (int32_t v456_n1 = 0; v456_n1 < 4; ++v456_n1) {
                int32_t v457_a = v455_n0 + v456_n1;
                float v458_data = ir4[v457_a];
                r4[v457_a] = v458_data;
              }
            }
            // s0 = store{r>s}(localShrMem0, r4);
            #pragma unroll
            for (int32_t v459_i0 = 0; v459_i0 < 1; ++v459_i0) {
              int32_t v464_lead = v30_lead + (v459_i0 * 8);
              #pragma unroll
              for (int32_t v460_i1 = 0; v460_i1 < 4; ++v460_i1) {
                float v462_data = r4[(v459_i0 + v460_i1)];
                int32_t v467_a = v464_lead + ((v460_i1 + 4) * 8);
                s0[(v467_a ^ ((v467_a >> 5) & 31))] = v462_data;
              }
            }
            // glb_m3 = abs(s0)
            item.barrier();
            #pragma unroll
            for (int32_t v471_k0 = 0; v471_k0 < 1; ++v471_k0) {
              int32_t v474_lead = v30_lead + (v471_k0 * 8);
              #pragma unroll
              for (int32_t v472_k1 = 0; v472_k1 < 8; ++v472_k1) {
                int32_t v476_a = v474_lead + (v472_k1 * 8);
                float v480_data = s0[(v476_a ^ ((v476_a >> 5) & 31))];
                float v481_e = sycl::fabs(v480_data);
                if (batchIdActive0) {
                  glb_m3[v476_a] = v481_e;
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


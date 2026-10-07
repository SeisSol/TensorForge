// === base name ===
kernel_1681f906b66f27af

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1681f906b66f27af = {{8, 2, 1}, 8, 8, 1, 2, 576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1681f906b66f27af(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1681f906b66f27af(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1681f906b66f27af(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_1681f906b66f27af(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1681f906b66f27af(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1681f906b66f27af(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1681f906b66f27af(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
            // wait(r0 = load{g>r}(glb_m0););
            float r3[4]{};
            // r3 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v46_i0 = 0; v46_i0 < 1; ++v46_i0) {
              int32_t v49_lead = v27_lead + (v46_i0 * 8);
              #pragma unroll
              for (int32_t v47_i1 = 0; v47_i1 < 4; ++v47_i1) {
                float v52_data = glb_m2[(v49_lead + (v47_i1 * 8))];
                r3[(v46_i0 + v47_i1)] = v52_data;
              }
            }
            // wait(r1 = load{g>r}(glb_m1););
            float r2[4]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 4)] [(0, 8)]
            float v55_data = r0[0];
            float v56_data = r1[0];
            float v59_data = r2[0];
            r2[0] = (v59_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v62_data = r1[1];
            float v65_data = r2[1];
            r2[1] = (v65_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v68_data = r1[2];
            float v71_data = r2[2];
            r2[2] = (v71_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v74_data = r1[3];
            float v77_data = r2[3];
            r2[3] = (v77_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v79_data = r0[1];
            float v83_data = r2[0];
            r2[0] = (v83_data + (v79_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v89_data = r2[1];
            r2[1] = (v89_data + (v79_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v95_data = r2[2];
            r2[2] = (v95_data + (v79_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v101_data = r2[3];
            r2[3] = (v101_data + (v79_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v103_data = r0[2];
            float v107_data = r2[0];
            r2[0] = (v107_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v113_data = r2[1];
            r2[1] = (v113_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v119_data = r2[2];
            r2[2] = (v119_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v125_data = r2[3];
            r2[3] = (v125_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v127_data = r0[3];
            float v131_data = r2[0];
            r2[0] = (v131_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v137_data = r2[1];
            r2[1] = (v137_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v143_data = r2[2];
            r2[2] = (v143_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v149_data = r2[3];
            r2[3] = (v149_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v151_data = r0[4];
            float v155_data = r2[0];
            r2[0] = (v155_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v161_data = r2[1];
            r2[1] = (v161_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v167_data = r2[2];
            r2[2] = (v167_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v173_data = r2[3];
            r2[3] = (v173_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v175_data = r0[5];
            float v179_data = r2[0];
            r2[0] = (v179_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v185_data = r2[1];
            r2[1] = (v185_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v191_data = r2[2];
            r2[2] = (v191_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v197_data = r2[3];
            r2[3] = (v197_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v199_data = r0[6];
            float v203_data = r2[0];
            r2[0] = (v203_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v209_data = r2[1];
            r2[1] = (v209_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v215_data = r2[2];
            r2[2] = (v215_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v221_data = r2[3];
            r2[3] = (v221_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v223_data = r0[7];
            float v227_data = r2[0];
            r2[0] = (v227_data + (v223_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v233_data = r2[1];
            r2[1] = (v233_data + (v223_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v239_data = r2[2];
            r2[2] = (v239_data + (v223_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v245_data = r2[3];
            r2[3] = (v245_data + (v223_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // s0 = store{r>s}(localShrMem0, r2);
            #pragma unroll
            for (int32_t v247_i0 = 0; v247_i0 < 1; ++v247_i0) {
              int32_t v252_lead = v27_lead + (v247_i0 * 8);
              #pragma unroll
              for (int32_t v248_i1 = 0; v248_i1 < 4; ++v248_i1) {
                float v250_data = r2[(v247_i0 + v248_i1)];
                int32_t v254_a = v252_lead + (v248_i1 * 8);
                s0[(v254_a ^ ((v254_a >> 5) & 31))] = v250_data;
              }
            }
            // wait(r3 = load{g>r}(glb_m2););
            float r4[4]{};
            // ir4 = +(r0 * r3)
            // [(0, 8), (0, 4)] [(0, 8)]
            float ir4[4]{};
            float v261_data = r3[0];
            float v264_data = ir4[0];
            ir4[0] = (v264_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v267_data = r3[1];
            float v270_data = ir4[1];
            ir4[1] = (v270_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v273_data = r3[2];
            float v276_data = ir4[2];
            ir4[2] = (v276_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v279_data = r3[3];
            float v282_data = ir4[3];
            ir4[3] = (v282_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v288_data = ir4[0];
            ir4[0] = (v288_data + (v79_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v294_data = ir4[1];
            ir4[1] = (v294_data + (v79_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v300_data = ir4[2];
            ir4[2] = (v300_data + (v79_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v306_data = ir4[3];
            ir4[3] = (v306_data + (v79_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v312_data = ir4[0];
            ir4[0] = (v312_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v318_data = ir4[1];
            ir4[1] = (v318_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v324_data = ir4[2];
            ir4[2] = (v324_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v330_data = ir4[3];
            ir4[3] = (v330_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v336_data = ir4[0];
            ir4[0] = (v336_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v342_data = ir4[1];
            ir4[1] = (v342_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v348_data = ir4[2];
            ir4[2] = (v348_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v354_data = ir4[3];
            ir4[3] = (v354_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v360_data = ir4[0];
            ir4[0] = (v360_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v366_data = ir4[1];
            ir4[1] = (v366_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v372_data = ir4[2];
            ir4[2] = (v372_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v378_data = ir4[3];
            ir4[3] = (v378_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v384_data = ir4[0];
            ir4[0] = (v384_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v390_data = ir4[1];
            ir4[1] = (v390_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v396_data = ir4[2];
            ir4[2] = (v396_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v402_data = ir4[3];
            ir4[3] = (v402_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v408_data = ir4[0];
            ir4[0] = (v408_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v414_data = ir4[1];
            ir4[1] = (v414_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v420_data = ir4[2];
            ir4[2] = (v420_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v426_data = ir4[3];
            ir4[3] = (v426_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v432_data = ir4[0];
            ir4[0] = (v432_data + (v223_data * (sycl::select_from_group(item.get_sub_group(), v261_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v438_data = ir4[1];
            ir4[1] = (v438_data + (v223_data * (sycl::select_from_group(item.get_sub_group(), v267_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v444_data = ir4[2];
            ir4[2] = (v444_data + (v223_data * (sycl::select_from_group(item.get_sub_group(), v273_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v450_data = ir4[3];
            ir4[3] = (v450_data + (v223_data * (sycl::select_from_group(item.get_sub_group(), v279_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
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


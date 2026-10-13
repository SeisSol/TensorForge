// === base name ===
kernel_0e5398682c6098ca

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0e5398682c6098ca = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0e5398682c6098ca(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0e5398682c6098ca(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0e5398682c6098ca(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 16 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0e5398682c6098ca(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0e5398682c6098ca(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0e5398682c6098ca(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0e5398682c6098ca(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (16, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 2 per block = block 8x2x1, 64 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   C = abs(M)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"M","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          size_t v8_batchIdLane0 = item.get_local_id(1) % 2;
          size_t v11_batchIdGroup0First = ((item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) < numElements0) ? (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) : 0u;
          int32_t v29_lead = item.get_local_id(2) % 8;
          float r0[8]{};
          size_t v456_row = v11_batchIdGroup0First + v8_batchIdLane0;
          const bool peel_batchIdActive0 = v456_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v456_row]));
          const float *const __restrict__ peel_glb_m1 = &m1[(peel_batchIdActive0 ? v456_row : v11_batchIdGroup0First) * 64 + 0 + m1_extraOffset];
          // r0 = load{g>r}(peel_glb_m1);
          #pragma unroll
          for (int32_t v464_i0 = 0; v464_i0 < 1; ++v464_i0) {
            int32_t v466_lead = v29_lead + (v464_i0 * 8);
            #pragma unroll
            for (int32_t v467_i1 = 0; v467_i1 < 8; ++v467_i1) {
              float v470_data = peel_glb_m1[(v466_lead + (v467_i1 * 8))];
              r0[(v464_i0 + v467_i1)] = v470_data;
            }
          }
          for (size_t v12_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v453_groupNextAhead = v12_batchIdGroup0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v455_groupNext = (v453_groupNextAhead < numElements0) ? v453_groupNextAhead : v12_batchIdGroup0;
            size_t v460_row = v455_groupNext + v8_batchIdLane0;
            const bool wrap_batchIdActive0 = v460_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v460_row]));
            const float *const __restrict__ wrap_glb_m1 = &m1[(wrap_batchIdActive0 ? v460_row : v455_groupNext) * 64 + 0 + m1_extraOffset];
            size_t v13_row = v12_batchIdGroup0 + v8_batchIdLane0;
            const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
            size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
            size_t v16_ahead1 = v15_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
            float *const __restrict__ glb_m0 = &m0[v15_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v15_batchId0 * 64 + 0 + m2_extraOffset];
            float *const __restrict__ glb_m3 = &m3[v15_batchId0 * 64 + 0 + m3_extraOffset];
            float r1[8]{};
            // r1 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v39_i0 = 0; v39_i0 < 1; ++v39_i0) {
              int32_t v42_lead = v29_lead + (v39_i0 * 8);
              #pragma unroll
              for (int32_t v40_i1 = 0; v40_i1 < 8; ++v40_i1) {
                float v45_data = glb_m2[(v42_lead + (v40_i1 * 8))];
                r1[(v39_i0 + v40_i1)] = v45_data;
              }
            }
            float r2[8]{};
            // ir2 = +(r0 * r1)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir2[8]{};
            float v49_data = r0[0];
            float v50_data = r1[0];
            float v53_data = ir2[0];
            ir2[0] = (v53_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v56_data = r1[1];
            float v59_data = ir2[1];
            ir2[1] = (v59_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v62_data = r1[2];
            float v65_data = ir2[2];
            ir2[2] = (v65_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v68_data = r1[3];
            float v71_data = ir2[3];
            ir2[3] = (v71_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v74_data = r1[4];
            float v77_data = ir2[4];
            ir2[4] = (v77_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v80_data = r1[5];
            float v83_data = ir2[5];
            ir2[5] = (v83_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v86_data = r1[6];
            float v89_data = ir2[6];
            ir2[6] = (v89_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v92_data = r1[7];
            float v95_data = ir2[7];
            ir2[7] = (v95_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v97_data = r0[1];
            float v101_data = ir2[0];
            ir2[0] = (v101_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v107_data = ir2[1];
            ir2[1] = (v107_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v113_data = ir2[2];
            ir2[2] = (v113_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v119_data = ir2[3];
            ir2[3] = (v119_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v125_data = ir2[4];
            ir2[4] = (v125_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v131_data = ir2[5];
            ir2[5] = (v131_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v137_data = ir2[6];
            ir2[6] = (v137_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v143_data = ir2[7];
            ir2[7] = (v143_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v145_data = r0[2];
            float v149_data = ir2[0];
            ir2[0] = (v149_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v155_data = ir2[1];
            ir2[1] = (v155_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v161_data = ir2[2];
            ir2[2] = (v161_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v167_data = ir2[3];
            ir2[3] = (v167_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v173_data = ir2[4];
            ir2[4] = (v173_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v179_data = ir2[5];
            ir2[5] = (v179_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v185_data = ir2[6];
            ir2[6] = (v185_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v191_data = ir2[7];
            ir2[7] = (v191_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v193_data = r0[3];
            float v197_data = ir2[0];
            ir2[0] = (v197_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v203_data = ir2[1];
            ir2[1] = (v203_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v209_data = ir2[2];
            ir2[2] = (v209_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v215_data = ir2[3];
            ir2[3] = (v215_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v221_data = ir2[4];
            ir2[4] = (v221_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v227_data = ir2[5];
            ir2[5] = (v227_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v233_data = ir2[6];
            ir2[6] = (v233_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v239_data = ir2[7];
            ir2[7] = (v239_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v241_data = r0[4];
            float v245_data = ir2[0];
            ir2[0] = (v245_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v251_data = ir2[1];
            ir2[1] = (v251_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v257_data = ir2[2];
            ir2[2] = (v257_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v263_data = ir2[3];
            ir2[3] = (v263_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v269_data = ir2[4];
            ir2[4] = (v269_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v275_data = ir2[5];
            ir2[5] = (v275_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v281_data = ir2[6];
            ir2[6] = (v281_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v287_data = ir2[7];
            ir2[7] = (v287_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v289_data = r0[5];
            float v293_data = ir2[0];
            ir2[0] = (v293_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v299_data = ir2[1];
            ir2[1] = (v299_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v305_data = ir2[2];
            ir2[2] = (v305_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v311_data = ir2[3];
            ir2[3] = (v311_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v317_data = ir2[4];
            ir2[4] = (v317_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v323_data = ir2[5];
            ir2[5] = (v323_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v329_data = ir2[6];
            ir2[6] = (v329_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v335_data = ir2[7];
            ir2[7] = (v335_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v337_data = r0[6];
            float v341_data = ir2[0];
            ir2[0] = (v341_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v347_data = ir2[1];
            ir2[1] = (v347_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v353_data = ir2[2];
            ir2[2] = (v353_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v359_data = ir2[3];
            ir2[3] = (v359_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v365_data = ir2[4];
            ir2[4] = (v365_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v371_data = ir2[5];
            ir2[5] = (v371_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v377_data = ir2[6];
            ir2[6] = (v377_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v383_data = ir2[7];
            ir2[7] = (v383_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v385_data = r0[7];
            // r0 = load{g>r}(wrap_glb_m1);
            #pragma unroll
            for (int32_t v472_i0 = 0; v472_i0 < 1; ++v472_i0) {
              int32_t v474_lead = v29_lead + (v472_i0 * 8);
              #pragma unroll
              for (int32_t v475_i1 = 0; v475_i1 < 8; ++v475_i1) {
                float v478_data = wrap_glb_m1[(v474_lead + (v475_i1 * 8))];
                r0[(v472_i0 + v475_i1)] = v478_data;
              }
            }
            float v389_data = ir2[0];
            ir2[0] = (v389_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v395_data = ir2[1];
            ir2[1] = (v395_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v401_data = ir2[2];
            ir2[2] = (v401_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v407_data = ir2[3];
            ir2[3] = (v407_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v413_data = ir2[4];
            ir2[4] = (v413_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v419_data = ir2[5];
            ir2[5] = (v419_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v425_data = ir2[6];
            ir2[6] = (v425_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v431_data = ir2[7];
            ir2[7] = (v431_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r2 = ir2
            #pragma unroll
            for (int32_t v433_n0 = 0; v433_n0 < 1; ++v433_n0) {
              #pragma unroll
              for (int32_t v434_n1 = 0; v434_n1 < 8; ++v434_n1) {
                int32_t v435_a = v433_n0 + v434_n1;
                float v436_data = ir2[v435_a];
                r2[v435_a] = v436_data;
              }
            }
            // glb_m0 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v437_i0 = 0; v437_i0 < 1; ++v437_i0) {
              #pragma unroll
              for (int32_t v438_i1 = 0; v438_i1 < 8; ++v438_i1) {
                float v440_data = r2[(v437_i0 + v438_i1)];
                if (batchIdActive0) {
                  glb_m0[((v29_lead + (v437_i0 * 8)) + (v438_i1 * 8))] = v440_data;
                }
              }
            }
            // glb_m3 = abs(glb_m0)
            #pragma unroll
            for (int32_t v445_k0 = 0; v445_k0 < 1; ++v445_k0) {
              int32_t v448_lead = v29_lead + (v445_k0 * 8);
              #pragma unroll
              for (int32_t v446_k1 = 0; v446_k1 < 8; ++v446_k1) {
                int32_t v450_a = v448_lead + (v446_k1 * 8);
                float v451_data = glb_m0[v450_a];
                float v452_e = sycl::fabs(v451_data);
                if (batchIdActive0) {
                  glb_m3[v450_a] = v452_e;
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


// === base name ===
kernel_354b8104dc1ff2e6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_354b8104dc1ff2e6 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_354b8104dc1ff2e6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_354b8104dc1ff2e6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_354b8104dc1ff2e6(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_354b8104dc1ff2e6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_354b8104dc1ff2e6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_354b8104dc1ff2e6(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_354b8104dc1ff2e6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×16(12×16) {0..12}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v21_lead = item.get_local_id(2) % 16;
              bool v22_g = v21_lead < 12;
              if (v22_g) {
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                  float v28_data = glb_m1[(v21_lead + (v23_i1 * 12))];
                  r0[v23_i1] = v28_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
                int32_t v34_lead = v21_lead + (v31_i0 * 16);
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
                  float v37_data = glb_m2[(v34_lead + (v32_i1 * 16))];
                  r1[(v31_i0 + v32_i1)] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = load{g>r}(glb_m0);
              if (v22_g) {
                #pragma unroll
                for (int32_t v40_i1 = 0; v40_i1 < 8; ++v40_i1) {
                  float v45_data = glb_m0[(v21_lead + (v40_i1 * 12))];
                  r2[v40_i1] = v45_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              // wait(r2 = load{g>r}(glb_m0););
              float r3[8]{};
              // ir3 = +(r0 * r1)
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir3[8]{};
              float v49_data = r0[0];
              float v50_data = r1[0];
              float v53_data = ir3[0];
              ir3[0] = (v53_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v56_data = r1[1];
              float v59_data = ir3[1];
              ir3[1] = (v59_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v62_data = r1[2];
              float v65_data = ir3[2];
              ir3[2] = (v65_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v68_data = r1[3];
              float v71_data = ir3[3];
              ir3[3] = (v71_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v74_data = r1[4];
              float v77_data = ir3[4];
              ir3[4] = (v77_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v80_data = r1[5];
              float v83_data = ir3[5];
              ir3[5] = (v83_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v86_data = r1[6];
              float v89_data = ir3[6];
              ir3[6] = (v89_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v92_data = r1[7];
              float v95_data = ir3[7];
              ir3[7] = (v95_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v97_data = r0[1];
              float v101_data = ir3[0];
              ir3[0] = (v101_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v107_data = ir3[1];
              ir3[1] = (v107_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v113_data = ir3[2];
              ir3[2] = (v113_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v119_data = ir3[3];
              ir3[3] = (v119_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v125_data = ir3[4];
              ir3[4] = (v125_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v131_data = ir3[5];
              ir3[5] = (v131_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v137_data = ir3[6];
              ir3[6] = (v137_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v143_data = ir3[7];
              ir3[7] = (v143_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v145_data = r0[2];
              float v149_data = ir3[0];
              ir3[0] = (v149_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v155_data = ir3[1];
              ir3[1] = (v155_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v161_data = ir3[2];
              ir3[2] = (v161_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v167_data = ir3[3];
              ir3[3] = (v167_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v173_data = ir3[4];
              ir3[4] = (v173_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v179_data = ir3[5];
              ir3[5] = (v179_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v185_data = ir3[6];
              ir3[6] = (v185_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v191_data = ir3[7];
              ir3[7] = (v191_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v193_data = r0[3];
              float v197_data = ir3[0];
              ir3[0] = (v197_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v203_data = ir3[1];
              ir3[1] = (v203_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v209_data = ir3[2];
              ir3[2] = (v209_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v215_data = ir3[3];
              ir3[3] = (v215_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v221_data = ir3[4];
              ir3[4] = (v221_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v227_data = ir3[5];
              ir3[5] = (v227_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v233_data = ir3[6];
              ir3[6] = (v233_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v239_data = ir3[7];
              ir3[7] = (v239_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v241_data = r0[4];
              float v245_data = ir3[0];
              ir3[0] = (v245_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v251_data = ir3[1];
              ir3[1] = (v251_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v257_data = ir3[2];
              ir3[2] = (v257_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v263_data = ir3[3];
              ir3[3] = (v263_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v269_data = ir3[4];
              ir3[4] = (v269_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v275_data = ir3[5];
              ir3[5] = (v275_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v281_data = ir3[6];
              ir3[6] = (v281_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v287_data = ir3[7];
              ir3[7] = (v287_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v289_data = r0[5];
              float v293_data = ir3[0];
              ir3[0] = (v293_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v299_data = ir3[1];
              ir3[1] = (v299_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v305_data = ir3[2];
              ir3[2] = (v305_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v311_data = ir3[3];
              ir3[3] = (v311_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v317_data = ir3[4];
              ir3[4] = (v317_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v323_data = ir3[5];
              ir3[5] = (v323_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v329_data = ir3[6];
              ir3[6] = (v329_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v335_data = ir3[7];
              ir3[7] = (v335_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v337_data = r0[6];
              float v341_data = ir3[0];
              ir3[0] = (v341_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v347_data = ir3[1];
              ir3[1] = (v347_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v353_data = ir3[2];
              ir3[2] = (v353_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v359_data = ir3[3];
              ir3[3] = (v359_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v365_data = ir3[4];
              ir3[4] = (v365_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v371_data = ir3[5];
              ir3[5] = (v371_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v377_data = ir3[6];
              ir3[6] = (v377_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v383_data = ir3[7];
              ir3[7] = (v383_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v385_data = r0[7];
              float v389_data = ir3[0];
              ir3[0] = (v389_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v395_data = ir3[1];
              ir3[1] = (v395_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v401_data = ir3[2];
              ir3[2] = (v401_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v407_data = ir3[3];
              ir3[3] = (v407_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v413_data = ir3[4];
              ir3[4] = (v413_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v419_data = ir3[5];
              ir3[5] = (v419_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v425_data = ir3[6];
              ir3[6] = (v425_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v431_data = ir3[7];
              ir3[7] = (v431_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v433_data = r0[8];
              float v437_data = ir3[0];
              ir3[0] = (v437_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v443_data = ir3[1];
              ir3[1] = (v443_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v449_data = ir3[2];
              ir3[2] = (v449_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v455_data = ir3[3];
              ir3[3] = (v455_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v461_data = ir3[4];
              ir3[4] = (v461_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v467_data = ir3[5];
              ir3[5] = (v467_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v473_data = ir3[6];
              ir3[6] = (v473_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v479_data = ir3[7];
              ir3[7] = (v479_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v481_data = r0[9];
              float v485_data = ir3[0];
              ir3[0] = (v485_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v491_data = ir3[1];
              ir3[1] = (v491_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v497_data = ir3[2];
              ir3[2] = (v497_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v503_data = ir3[3];
              ir3[3] = (v503_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v509_data = ir3[4];
              ir3[4] = (v509_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v515_data = ir3[5];
              ir3[5] = (v515_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v521_data = ir3[6];
              ir3[6] = (v521_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v527_data = ir3[7];
              ir3[7] = (v527_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v529_data = r0[10];
              float v533_data = ir3[0];
              ir3[0] = (v533_data + (v529_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v539_data = ir3[1];
              ir3[1] = (v539_data + (v529_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v545_data = ir3[2];
              ir3[2] = (v545_data + (v529_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v551_data = ir3[3];
              ir3[3] = (v551_data + (v529_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v557_data = ir3[4];
              ir3[4] = (v557_data + (v529_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v563_data = ir3[5];
              ir3[5] = (v563_data + (v529_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v569_data = ir3[6];
              ir3[6] = (v569_data + (v529_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v575_data = ir3[7];
              ir3[7] = (v575_data + (v529_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v577_data = r0[11];
              float v581_data = ir3[0];
              ir3[0] = (v581_data + (v577_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v587_data = ir3[1];
              ir3[1] = (v587_data + (v577_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v593_data = ir3[2];
              ir3[2] = (v593_data + (v577_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v599_data = ir3[3];
              ir3[3] = (v599_data + (v577_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v605_data = ir3[4];
              ir3[4] = (v605_data + (v577_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v611_data = ir3[5];
              ir3[5] = (v611_data + (v577_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v617_data = ir3[6];
              ir3[6] = (v617_data + (v577_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v623_data = ir3[7];
              ir3[7] = (v623_data + (v577_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v625_data = r0[12];
              float v629_data = ir3[0];
              ir3[0] = (v629_data + (v625_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v635_data = ir3[1];
              ir3[1] = (v635_data + (v625_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v641_data = ir3[2];
              ir3[2] = (v641_data + (v625_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v647_data = ir3[3];
              ir3[3] = (v647_data + (v625_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v653_data = ir3[4];
              ir3[4] = (v653_data + (v625_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v659_data = ir3[5];
              ir3[5] = (v659_data + (v625_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v665_data = ir3[6];
              ir3[6] = (v665_data + (v625_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v671_data = ir3[7];
              ir3[7] = (v671_data + (v625_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v673_data = r0[13];
              float v677_data = ir3[0];
              ir3[0] = (v677_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v683_data = ir3[1];
              ir3[1] = (v683_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v689_data = ir3[2];
              ir3[2] = (v689_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v695_data = ir3[3];
              ir3[3] = (v695_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v701_data = ir3[4];
              ir3[4] = (v701_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v707_data = ir3[5];
              ir3[5] = (v707_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v713_data = ir3[6];
              ir3[6] = (v713_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v719_data = ir3[7];
              ir3[7] = (v719_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v721_data = r0[14];
              float v725_data = ir3[0];
              ir3[0] = (v725_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v731_data = ir3[1];
              ir3[1] = (v731_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v737_data = ir3[2];
              ir3[2] = (v737_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v743_data = ir3[3];
              ir3[3] = (v743_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v749_data = ir3[4];
              ir3[4] = (v749_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v755_data = ir3[5];
              ir3[5] = (v755_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v761_data = ir3[6];
              ir3[6] = (v761_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v767_data = ir3[7];
              ir3[7] = (v767_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v769_data = r0[15];
              float v773_data = ir3[0];
              ir3[0] = (v773_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v779_data = ir3[1];
              ir3[1] = (v779_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v785_data = ir3[2];
              ir3[2] = (v785_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v791_data = ir3[3];
              ir3[3] = (v791_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v797_data = ir3[4];
              ir3[4] = (v797_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v803_data = ir3[5];
              ir3[5] = (v803_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v809_data = ir3[6];
              ir3[6] = (v809_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v815_data = ir3[7];
              ir3[7] = (v815_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r3 = ir3 + r2
              if (v22_g) {
                #pragma unroll
                for (int32_t v817_n1 = 0; v817_n1 < 8; ++v817_n1) {
                  float v819_data = ir3[v817_n1];
                  float v820_data = r2[v817_n1];
                  r3[v817_n1] = (v820_data + v819_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v22_g) {
                #pragma unroll
                for (int32_t v822_i1 = 0; v822_i1 < 8; ++v822_i1) {
                  float v824_data = r3[v822_i1];
                  glb_m0[(v21_lead + (v822_i1 * 12))] = v824_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}


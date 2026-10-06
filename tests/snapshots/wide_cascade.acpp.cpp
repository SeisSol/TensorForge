// === base name ===
kernel_d8583de5693fbf3d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d8583de5693fbf3d = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d8583de5693fbf3d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d8583de5693fbf3d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d8583de5693fbf3d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_d8583de5693fbf3d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d8583de5693fbf3d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d8583de5693fbf3d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d8583de5693fbf3d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×11(16×11) {0..16}×{0..11} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×11(16×11) {0..16}×{0..11} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,11]],"name":"m0","ordered":false,"parts":1,"shape":[16,11],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,11]],"name":"m2","ordered":false,"parts":1,"shape":[16,11],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,11]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,11]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 176 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 176 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v23_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v27_lead = v23_lead + (v24_i0 * 16);
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 16; ++v25_i1) {
                  float v30_data = glb_m1[(v27_lead + (v25_i1 * 16))];
                  r0[(v24_i0 + v25_i1)] = v30_data;
                }
              }
              float r1[11]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
                int32_t v36_lead = v23_lead + (v33_i0 * 16);
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 11; ++v34_i1) {
                  float v39_data = glb_m2[(v36_lead + (v34_i1 * 16))];
                  r1[(v33_i0 + v34_i1)] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[11]{};
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 11)] [(0, 16)]
              float ir2[11]{};
              float v43_data = r0[0];
              float v44_data = r1[0];
              float v47_data = ir2[0];
              ir2[0] = (v47_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v50_data = r1[1];
              float v53_data = ir2[1];
              ir2[1] = (v53_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v56_data = r1[2];
              float v59_data = ir2[2];
              ir2[2] = (v59_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v62_data = r1[3];
              float v65_data = ir2[3];
              ir2[3] = (v65_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v68_data = r1[4];
              float v71_data = ir2[4];
              ir2[4] = (v71_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v74_data = r1[5];
              float v77_data = ir2[5];
              ir2[5] = (v77_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v80_data = r1[6];
              float v83_data = ir2[6];
              ir2[6] = (v83_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v86_data = r1[7];
              float v89_data = ir2[7];
              ir2[7] = (v89_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v92_data = r1[8];
              float v95_data = ir2[8];
              ir2[8] = (v95_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v98_data = r1[9];
              float v101_data = ir2[9];
              ir2[9] = (v101_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v104_data = r1[10];
              float v107_data = ir2[10];
              ir2[10] = (v107_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v109_data = r0[1];
              float v113_data = ir2[0];
              ir2[0] = (v113_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v119_data = ir2[1];
              ir2[1] = (v119_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v125_data = ir2[2];
              ir2[2] = (v125_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v131_data = ir2[3];
              ir2[3] = (v131_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v137_data = ir2[4];
              ir2[4] = (v137_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v143_data = ir2[5];
              ir2[5] = (v143_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v149_data = ir2[6];
              ir2[6] = (v149_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v155_data = ir2[7];
              ir2[7] = (v155_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v161_data = ir2[8];
              ir2[8] = (v161_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v167_data = ir2[9];
              ir2[9] = (v167_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v173_data = ir2[10];
              ir2[10] = (v173_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v175_data = r0[2];
              float v179_data = ir2[0];
              ir2[0] = (v179_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v185_data = ir2[1];
              ir2[1] = (v185_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v191_data = ir2[2];
              ir2[2] = (v191_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v197_data = ir2[3];
              ir2[3] = (v197_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v203_data = ir2[4];
              ir2[4] = (v203_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v209_data = ir2[5];
              ir2[5] = (v209_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v215_data = ir2[6];
              ir2[6] = (v215_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v221_data = ir2[7];
              ir2[7] = (v221_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v227_data = ir2[8];
              ir2[8] = (v227_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v233_data = ir2[9];
              ir2[9] = (v233_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v239_data = ir2[10];
              ir2[10] = (v239_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v241_data = r0[3];
              float v245_data = ir2[0];
              ir2[0] = (v245_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v251_data = ir2[1];
              ir2[1] = (v251_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v257_data = ir2[2];
              ir2[2] = (v257_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v263_data = ir2[3];
              ir2[3] = (v263_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v269_data = ir2[4];
              ir2[4] = (v269_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v275_data = ir2[5];
              ir2[5] = (v275_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v281_data = ir2[6];
              ir2[6] = (v281_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v287_data = ir2[7];
              ir2[7] = (v287_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v293_data = ir2[8];
              ir2[8] = (v293_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v299_data = ir2[9];
              ir2[9] = (v299_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v305_data = ir2[10];
              ir2[10] = (v305_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v307_data = r0[4];
              float v311_data = ir2[0];
              ir2[0] = (v311_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v317_data = ir2[1];
              ir2[1] = (v317_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v323_data = ir2[2];
              ir2[2] = (v323_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v329_data = ir2[3];
              ir2[3] = (v329_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v335_data = ir2[4];
              ir2[4] = (v335_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v341_data = ir2[5];
              ir2[5] = (v341_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v347_data = ir2[6];
              ir2[6] = (v347_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v353_data = ir2[7];
              ir2[7] = (v353_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v359_data = ir2[8];
              ir2[8] = (v359_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v365_data = ir2[9];
              ir2[9] = (v365_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v371_data = ir2[10];
              ir2[10] = (v371_data + (v307_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v373_data = r0[5];
              float v377_data = ir2[0];
              ir2[0] = (v377_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v383_data = ir2[1];
              ir2[1] = (v383_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v389_data = ir2[2];
              ir2[2] = (v389_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v395_data = ir2[3];
              ir2[3] = (v395_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v401_data = ir2[4];
              ir2[4] = (v401_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v407_data = ir2[5];
              ir2[5] = (v407_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v413_data = ir2[6];
              ir2[6] = (v413_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v419_data = ir2[7];
              ir2[7] = (v419_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v425_data = ir2[8];
              ir2[8] = (v425_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v431_data = ir2[9];
              ir2[9] = (v431_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v437_data = ir2[10];
              ir2[10] = (v437_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v439_data = r0[6];
              float v443_data = ir2[0];
              ir2[0] = (v443_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v449_data = ir2[1];
              ir2[1] = (v449_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v455_data = ir2[2];
              ir2[2] = (v455_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v461_data = ir2[3];
              ir2[3] = (v461_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v467_data = ir2[4];
              ir2[4] = (v467_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v473_data = ir2[5];
              ir2[5] = (v473_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v479_data = ir2[6];
              ir2[6] = (v479_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v485_data = ir2[7];
              ir2[7] = (v485_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v491_data = ir2[8];
              ir2[8] = (v491_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v497_data = ir2[9];
              ir2[9] = (v497_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v503_data = ir2[10];
              ir2[10] = (v503_data + (v439_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v505_data = r0[7];
              float v509_data = ir2[0];
              ir2[0] = (v509_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v515_data = ir2[1];
              ir2[1] = (v515_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v521_data = ir2[2];
              ir2[2] = (v521_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v527_data = ir2[3];
              ir2[3] = (v527_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v533_data = ir2[4];
              ir2[4] = (v533_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v539_data = ir2[5];
              ir2[5] = (v539_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v545_data = ir2[6];
              ir2[6] = (v545_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v551_data = ir2[7];
              ir2[7] = (v551_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v557_data = ir2[8];
              ir2[8] = (v557_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v563_data = ir2[9];
              ir2[9] = (v563_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v569_data = ir2[10];
              ir2[10] = (v569_data + (v505_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v571_data = r0[8];
              float v575_data = ir2[0];
              ir2[0] = (v575_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v581_data = ir2[1];
              ir2[1] = (v581_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v587_data = ir2[2];
              ir2[2] = (v587_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v593_data = ir2[3];
              ir2[3] = (v593_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v599_data = ir2[4];
              ir2[4] = (v599_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v605_data = ir2[5];
              ir2[5] = (v605_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v611_data = ir2[6];
              ir2[6] = (v611_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v617_data = ir2[7];
              ir2[7] = (v617_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v623_data = ir2[8];
              ir2[8] = (v623_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v629_data = ir2[9];
              ir2[9] = (v629_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v635_data = ir2[10];
              ir2[10] = (v635_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v637_data = r0[9];
              float v641_data = ir2[0];
              ir2[0] = (v641_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v647_data = ir2[1];
              ir2[1] = (v647_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v653_data = ir2[2];
              ir2[2] = (v653_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v659_data = ir2[3];
              ir2[3] = (v659_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v665_data = ir2[4];
              ir2[4] = (v665_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v671_data = ir2[5];
              ir2[5] = (v671_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v677_data = ir2[6];
              ir2[6] = (v677_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v683_data = ir2[7];
              ir2[7] = (v683_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v689_data = ir2[8];
              ir2[8] = (v689_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v695_data = ir2[9];
              ir2[9] = (v695_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v701_data = ir2[10];
              ir2[10] = (v701_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v703_data = r0[10];
              float v707_data = ir2[0];
              ir2[0] = (v707_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v713_data = ir2[1];
              ir2[1] = (v713_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v719_data = ir2[2];
              ir2[2] = (v719_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v725_data = ir2[3];
              ir2[3] = (v725_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v731_data = ir2[4];
              ir2[4] = (v731_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v737_data = ir2[5];
              ir2[5] = (v737_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v743_data = ir2[6];
              ir2[6] = (v743_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v749_data = ir2[7];
              ir2[7] = (v749_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v755_data = ir2[8];
              ir2[8] = (v755_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v761_data = ir2[9];
              ir2[9] = (v761_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v767_data = ir2[10];
              ir2[10] = (v767_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v769_data = r0[11];
              float v773_data = ir2[0];
              ir2[0] = (v773_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v779_data = ir2[1];
              ir2[1] = (v779_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v785_data = ir2[2];
              ir2[2] = (v785_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v791_data = ir2[3];
              ir2[3] = (v791_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v797_data = ir2[4];
              ir2[4] = (v797_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v803_data = ir2[5];
              ir2[5] = (v803_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v809_data = ir2[6];
              ir2[6] = (v809_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v815_data = ir2[7];
              ir2[7] = (v815_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v821_data = ir2[8];
              ir2[8] = (v821_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v827_data = ir2[9];
              ir2[9] = (v827_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v833_data = ir2[10];
              ir2[10] = (v833_data + (v769_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v835_data = r0[12];
              float v839_data = ir2[0];
              ir2[0] = (v839_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v845_data = ir2[1];
              ir2[1] = (v845_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v851_data = ir2[2];
              ir2[2] = (v851_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v857_data = ir2[3];
              ir2[3] = (v857_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v863_data = ir2[4];
              ir2[4] = (v863_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v869_data = ir2[5];
              ir2[5] = (v869_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v875_data = ir2[6];
              ir2[6] = (v875_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v881_data = ir2[7];
              ir2[7] = (v881_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v887_data = ir2[8];
              ir2[8] = (v887_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v893_data = ir2[9];
              ir2[9] = (v893_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v899_data = ir2[10];
              ir2[10] = (v899_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v901_data = r0[13];
              float v905_data = ir2[0];
              ir2[0] = (v905_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v911_data = ir2[1];
              ir2[1] = (v911_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v917_data = ir2[2];
              ir2[2] = (v917_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v923_data = ir2[3];
              ir2[3] = (v923_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v929_data = ir2[4];
              ir2[4] = (v929_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v935_data = ir2[5];
              ir2[5] = (v935_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v941_data = ir2[6];
              ir2[6] = (v941_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v947_data = ir2[7];
              ir2[7] = (v947_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v953_data = ir2[8];
              ir2[8] = (v953_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v959_data = ir2[9];
              ir2[9] = (v959_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v965_data = ir2[10];
              ir2[10] = (v965_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v967_data = r0[14];
              float v971_data = ir2[0];
              ir2[0] = (v971_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v977_data = ir2[1];
              ir2[1] = (v977_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v983_data = ir2[2];
              ir2[2] = (v983_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v989_data = ir2[3];
              ir2[3] = (v989_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v995_data = ir2[4];
              ir2[4] = (v995_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1001_data = ir2[5];
              ir2[5] = (v1001_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1007_data = ir2[6];
              ir2[6] = (v1007_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1013_data = ir2[7];
              ir2[7] = (v1013_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1019_data = ir2[8];
              ir2[8] = (v1019_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1025_data = ir2[9];
              ir2[9] = (v1025_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1031_data = ir2[10];
              ir2[10] = (v1031_data + (v967_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1033_data = r0[15];
              float v1037_data = ir2[0];
              ir2[0] = (v1037_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1043_data = ir2[1];
              ir2[1] = (v1043_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1049_data = ir2[2];
              ir2[2] = (v1049_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1055_data = ir2[3];
              ir2[3] = (v1055_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1061_data = ir2[4];
              ir2[4] = (v1061_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1067_data = ir2[5];
              ir2[5] = (v1067_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1073_data = ir2[6];
              ir2[6] = (v1073_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1079_data = ir2[7];
              ir2[7] = (v1079_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1085_data = ir2[8];
              ir2[8] = (v1085_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1091_data = ir2[9];
              ir2[9] = (v1091_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1097_data = ir2[10];
              ir2[10] = (v1097_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v1099_n0 = 0; v1099_n0 < 1; ++v1099_n0) {
                #pragma unroll
                for (int32_t v1100_n1 = 0; v1100_n1 < 11; ++v1100_n1) {
                  int32_t v1101_a = v1099_n0 + v1100_n1;
                  float v1102_data = ir2[v1101_a];
                  r2[v1101_a] = v1102_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1103_i0 = 0; v1103_i0 < 1; ++v1103_i0) {
                int32_t v1108_lead = v23_lead + (v1103_i0 * 16);
                #pragma unroll
                for (int32_t v1104_i1 = 0; v1104_i1 < 11; ++v1104_i1) {
                  float v1106_data = r2[(v1103_i0 + v1104_i1)];
                  glb_m0[(v1108_lead + (v1104_i1 * 16))] = v1106_data;
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


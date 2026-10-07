// === base name ===
kernel_60289bb15fb176a6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_60289bb15fb176a6 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_60289bb15fb176a6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_60289bb15fb176a6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_60289bb15fb176a6(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_60289bb15fb176a6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_60289bb15fb176a6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_60289bb15fb176a6(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_60289bb15fb176a6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×8(12×8) {4..16}×{0..8} strided
        //   m1 16×16(12×16) {4..16}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[4,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[4,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[4,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
                int32_t v27_a = (v21_lead + 4) - 4;
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                  float v30_data = glb_m1[(v27_a + (v23_i1 * 12))];
                  r0[v23_i1] = v30_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
                int32_t v36_lead = v21_lead + (v33_i0 * 16);
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
                  float v39_data = glb_m2[(v36_lead + (v34_i1 * 16))];
                  r1[(v33_i0 + v34_i1)] = v39_data;
                }
              }
              float r2[8]{};
              // ir2 = +(r0 * r1)
              // [(16, 28), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v44_data = r1[0];
              float v45_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v50_data = r1[1];
              float v51_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v56_data = r1[2];
              float v57_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v62_data = r1[3];
              float v63_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v68_data = r1[4];
              float v69_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v74_data = r1[5];
              float v75_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v80_data = r1[6];
              float v81_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v86_data = r1[7];
              float v87_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              if (v22_g) {
                float v43_data = r0[0];
                float v47_data = ir2[0];
                ir2[0] = (v47_data + (v43_data * v45_bc));
                float v53_data = ir2[1];
                ir2[1] = (v53_data + (v43_data * v51_bc));
                float v59_data = ir2[2];
                ir2[2] = (v59_data + (v43_data * v57_bc));
                float v65_data = ir2[3];
                ir2[3] = (v65_data + (v43_data * v63_bc));
                float v71_data = ir2[4];
                ir2[4] = (v71_data + (v43_data * v69_bc));
                float v77_data = ir2[5];
                ir2[5] = (v77_data + (v43_data * v75_bc));
                float v83_data = ir2[6];
                ir2[6] = (v83_data + (v43_data * v81_bc));
                float v89_data = ir2[7];
                ir2[7] = (v89_data + (v43_data * v87_bc));
              }
              float v93_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v99_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v105_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v111_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v117_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v123_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v129_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v135_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              if (v22_g) {
                float v91_data = r0[1];
                float v95_data = ir2[0];
                ir2[0] = (v95_data + (v91_data * v93_bc));
                float v101_data = ir2[1];
                ir2[1] = (v101_data + (v91_data * v99_bc));
                float v107_data = ir2[2];
                ir2[2] = (v107_data + (v91_data * v105_bc));
                float v113_data = ir2[3];
                ir2[3] = (v113_data + (v91_data * v111_bc));
                float v119_data = ir2[4];
                ir2[4] = (v119_data + (v91_data * v117_bc));
                float v125_data = ir2[5];
                ir2[5] = (v125_data + (v91_data * v123_bc));
                float v131_data = ir2[6];
                ir2[6] = (v131_data + (v91_data * v129_bc));
                float v137_data = ir2[7];
                ir2[7] = (v137_data + (v91_data * v135_bc));
              }
              float v141_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v147_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v153_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v159_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v165_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v171_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v177_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v183_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              if (v22_g) {
                float v139_data = r0[2];
                float v143_data = ir2[0];
                ir2[0] = (v143_data + (v139_data * v141_bc));
                float v149_data = ir2[1];
                ir2[1] = (v149_data + (v139_data * v147_bc));
                float v155_data = ir2[2];
                ir2[2] = (v155_data + (v139_data * v153_bc));
                float v161_data = ir2[3];
                ir2[3] = (v161_data + (v139_data * v159_bc));
                float v167_data = ir2[4];
                ir2[4] = (v167_data + (v139_data * v165_bc));
                float v173_data = ir2[5];
                ir2[5] = (v173_data + (v139_data * v171_bc));
                float v179_data = ir2[6];
                ir2[6] = (v179_data + (v139_data * v177_bc));
                float v185_data = ir2[7];
                ir2[7] = (v185_data + (v139_data * v183_bc));
              }
              float v189_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v195_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v201_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v207_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v213_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v219_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v225_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v231_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              if (v22_g) {
                float v187_data = r0[3];
                float v191_data = ir2[0];
                ir2[0] = (v191_data + (v187_data * v189_bc));
                float v197_data = ir2[1];
                ir2[1] = (v197_data + (v187_data * v195_bc));
                float v203_data = ir2[2];
                ir2[2] = (v203_data + (v187_data * v201_bc));
                float v209_data = ir2[3];
                ir2[3] = (v209_data + (v187_data * v207_bc));
                float v215_data = ir2[4];
                ir2[4] = (v215_data + (v187_data * v213_bc));
                float v221_data = ir2[5];
                ir2[5] = (v221_data + (v187_data * v219_bc));
                float v227_data = ir2[6];
                ir2[6] = (v227_data + (v187_data * v225_bc));
                float v233_data = ir2[7];
                ir2[7] = (v233_data + (v187_data * v231_bc));
              }
              float v237_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v243_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v249_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v255_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v261_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v267_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v273_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v279_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              if (v22_g) {
                float v235_data = r0[4];
                float v239_data = ir2[0];
                ir2[0] = (v239_data + (v235_data * v237_bc));
                float v245_data = ir2[1];
                ir2[1] = (v245_data + (v235_data * v243_bc));
                float v251_data = ir2[2];
                ir2[2] = (v251_data + (v235_data * v249_bc));
                float v257_data = ir2[3];
                ir2[3] = (v257_data + (v235_data * v255_bc));
                float v263_data = ir2[4];
                ir2[4] = (v263_data + (v235_data * v261_bc));
                float v269_data = ir2[5];
                ir2[5] = (v269_data + (v235_data * v267_bc));
                float v275_data = ir2[6];
                ir2[6] = (v275_data + (v235_data * v273_bc));
                float v281_data = ir2[7];
                ir2[7] = (v281_data + (v235_data * v279_bc));
              }
              float v285_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v291_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v297_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v303_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v309_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v315_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v321_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v327_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              if (v22_g) {
                float v283_data = r0[5];
                float v287_data = ir2[0];
                ir2[0] = (v287_data + (v283_data * v285_bc));
                float v293_data = ir2[1];
                ir2[1] = (v293_data + (v283_data * v291_bc));
                float v299_data = ir2[2];
                ir2[2] = (v299_data + (v283_data * v297_bc));
                float v305_data = ir2[3];
                ir2[3] = (v305_data + (v283_data * v303_bc));
                float v311_data = ir2[4];
                ir2[4] = (v311_data + (v283_data * v309_bc));
                float v317_data = ir2[5];
                ir2[5] = (v317_data + (v283_data * v315_bc));
                float v323_data = ir2[6];
                ir2[6] = (v323_data + (v283_data * v321_bc));
                float v329_data = ir2[7];
                ir2[7] = (v329_data + (v283_data * v327_bc));
              }
              float v333_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v339_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v345_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v351_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v357_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v363_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v369_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v375_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              if (v22_g) {
                float v331_data = r0[6];
                float v335_data = ir2[0];
                ir2[0] = (v335_data + (v331_data * v333_bc));
                float v341_data = ir2[1];
                ir2[1] = (v341_data + (v331_data * v339_bc));
                float v347_data = ir2[2];
                ir2[2] = (v347_data + (v331_data * v345_bc));
                float v353_data = ir2[3];
                ir2[3] = (v353_data + (v331_data * v351_bc));
                float v359_data = ir2[4];
                ir2[4] = (v359_data + (v331_data * v357_bc));
                float v365_data = ir2[5];
                ir2[5] = (v365_data + (v331_data * v363_bc));
                float v371_data = ir2[6];
                ir2[6] = (v371_data + (v331_data * v369_bc));
                float v377_data = ir2[7];
                ir2[7] = (v377_data + (v331_data * v375_bc));
              }
              float v381_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v387_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v393_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v399_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v405_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v411_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v417_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v423_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              if (v22_g) {
                float v379_data = r0[7];
                float v383_data = ir2[0];
                ir2[0] = (v383_data + (v379_data * v381_bc));
                float v389_data = ir2[1];
                ir2[1] = (v389_data + (v379_data * v387_bc));
                float v395_data = ir2[2];
                ir2[2] = (v395_data + (v379_data * v393_bc));
                float v401_data = ir2[3];
                ir2[3] = (v401_data + (v379_data * v399_bc));
                float v407_data = ir2[4];
                ir2[4] = (v407_data + (v379_data * v405_bc));
                float v413_data = ir2[5];
                ir2[5] = (v413_data + (v379_data * v411_bc));
                float v419_data = ir2[6];
                ir2[6] = (v419_data + (v379_data * v417_bc));
                float v425_data = ir2[7];
                ir2[7] = (v425_data + (v379_data * v423_bc));
              }
              float v429_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v435_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v441_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v447_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v453_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v459_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v465_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v471_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              if (v22_g) {
                float v427_data = r0[8];
                float v431_data = ir2[0];
                ir2[0] = (v431_data + (v427_data * v429_bc));
                float v437_data = ir2[1];
                ir2[1] = (v437_data + (v427_data * v435_bc));
                float v443_data = ir2[2];
                ir2[2] = (v443_data + (v427_data * v441_bc));
                float v449_data = ir2[3];
                ir2[3] = (v449_data + (v427_data * v447_bc));
                float v455_data = ir2[4];
                ir2[4] = (v455_data + (v427_data * v453_bc));
                float v461_data = ir2[5];
                ir2[5] = (v461_data + (v427_data * v459_bc));
                float v467_data = ir2[6];
                ir2[6] = (v467_data + (v427_data * v465_bc));
                float v473_data = ir2[7];
                ir2[7] = (v473_data + (v427_data * v471_bc));
              }
              float v477_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v483_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v489_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v495_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v501_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v507_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v513_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v519_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              if (v22_g) {
                float v475_data = r0[9];
                float v479_data = ir2[0];
                ir2[0] = (v479_data + (v475_data * v477_bc));
                float v485_data = ir2[1];
                ir2[1] = (v485_data + (v475_data * v483_bc));
                float v491_data = ir2[2];
                ir2[2] = (v491_data + (v475_data * v489_bc));
                float v497_data = ir2[3];
                ir2[3] = (v497_data + (v475_data * v495_bc));
                float v503_data = ir2[4];
                ir2[4] = (v503_data + (v475_data * v501_bc));
                float v509_data = ir2[5];
                ir2[5] = (v509_data + (v475_data * v507_bc));
                float v515_data = ir2[6];
                ir2[6] = (v515_data + (v475_data * v513_bc));
                float v521_data = ir2[7];
                ir2[7] = (v521_data + (v475_data * v519_bc));
              }
              float v525_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v531_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v537_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v543_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v549_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v555_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v561_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v567_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              if (v22_g) {
                float v523_data = r0[10];
                float v527_data = ir2[0];
                ir2[0] = (v527_data + (v523_data * v525_bc));
                float v533_data = ir2[1];
                ir2[1] = (v533_data + (v523_data * v531_bc));
                float v539_data = ir2[2];
                ir2[2] = (v539_data + (v523_data * v537_bc));
                float v545_data = ir2[3];
                ir2[3] = (v545_data + (v523_data * v543_bc));
                float v551_data = ir2[4];
                ir2[4] = (v551_data + (v523_data * v549_bc));
                float v557_data = ir2[5];
                ir2[5] = (v557_data + (v523_data * v555_bc));
                float v563_data = ir2[6];
                ir2[6] = (v563_data + (v523_data * v561_bc));
                float v569_data = ir2[7];
                ir2[7] = (v569_data + (v523_data * v567_bc));
              }
              float v573_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v579_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v585_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v591_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v597_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v603_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v609_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v615_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              if (v22_g) {
                float v571_data = r0[11];
                float v575_data = ir2[0];
                ir2[0] = (v575_data + (v571_data * v573_bc));
                float v581_data = ir2[1];
                ir2[1] = (v581_data + (v571_data * v579_bc));
                float v587_data = ir2[2];
                ir2[2] = (v587_data + (v571_data * v585_bc));
                float v593_data = ir2[3];
                ir2[3] = (v593_data + (v571_data * v591_bc));
                float v599_data = ir2[4];
                ir2[4] = (v599_data + (v571_data * v597_bc));
                float v605_data = ir2[5];
                ir2[5] = (v605_data + (v571_data * v603_bc));
                float v611_data = ir2[6];
                ir2[6] = (v611_data + (v571_data * v609_bc));
                float v617_data = ir2[7];
                ir2[7] = (v617_data + (v571_data * v615_bc));
              }
              float v621_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v627_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v633_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v639_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v645_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v651_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v657_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v663_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              if (v22_g) {
                float v619_data = r0[12];
                float v623_data = ir2[0];
                ir2[0] = (v623_data + (v619_data * v621_bc));
                float v629_data = ir2[1];
                ir2[1] = (v629_data + (v619_data * v627_bc));
                float v635_data = ir2[2];
                ir2[2] = (v635_data + (v619_data * v633_bc));
                float v641_data = ir2[3];
                ir2[3] = (v641_data + (v619_data * v639_bc));
                float v647_data = ir2[4];
                ir2[4] = (v647_data + (v619_data * v645_bc));
                float v653_data = ir2[5];
                ir2[5] = (v653_data + (v619_data * v651_bc));
                float v659_data = ir2[6];
                ir2[6] = (v659_data + (v619_data * v657_bc));
                float v665_data = ir2[7];
                ir2[7] = (v665_data + (v619_data * v663_bc));
              }
              float v669_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v675_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v681_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v687_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v693_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v699_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v705_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v711_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              if (v22_g) {
                float v667_data = r0[13];
                float v671_data = ir2[0];
                ir2[0] = (v671_data + (v667_data * v669_bc));
                float v677_data = ir2[1];
                ir2[1] = (v677_data + (v667_data * v675_bc));
                float v683_data = ir2[2];
                ir2[2] = (v683_data + (v667_data * v681_bc));
                float v689_data = ir2[3];
                ir2[3] = (v689_data + (v667_data * v687_bc));
                float v695_data = ir2[4];
                ir2[4] = (v695_data + (v667_data * v693_bc));
                float v701_data = ir2[5];
                ir2[5] = (v701_data + (v667_data * v699_bc));
                float v707_data = ir2[6];
                ir2[6] = (v707_data + (v667_data * v705_bc));
                float v713_data = ir2[7];
                ir2[7] = (v713_data + (v667_data * v711_bc));
              }
              float v717_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v723_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v729_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v735_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v741_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v747_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v753_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v759_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              if (v22_g) {
                float v715_data = r0[14];
                float v719_data = ir2[0];
                ir2[0] = (v719_data + (v715_data * v717_bc));
                float v725_data = ir2[1];
                ir2[1] = (v725_data + (v715_data * v723_bc));
                float v731_data = ir2[2];
                ir2[2] = (v731_data + (v715_data * v729_bc));
                float v737_data = ir2[3];
                ir2[3] = (v737_data + (v715_data * v735_bc));
                float v743_data = ir2[4];
                ir2[4] = (v743_data + (v715_data * v741_bc));
                float v749_data = ir2[5];
                ir2[5] = (v749_data + (v715_data * v747_bc));
                float v755_data = ir2[6];
                ir2[6] = (v755_data + (v715_data * v753_bc));
                float v761_data = ir2[7];
                ir2[7] = (v761_data + (v715_data * v759_bc));
              }
              float v765_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v771_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v777_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v783_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v789_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v795_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v801_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v807_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              if (v22_g) {
                float v763_data = r0[15];
                float v767_data = ir2[0];
                ir2[0] = (v767_data + (v763_data * v765_bc));
                float v773_data = ir2[1];
                ir2[1] = (v773_data + (v763_data * v771_bc));
                float v779_data = ir2[2];
                ir2[2] = (v779_data + (v763_data * v777_bc));
                float v785_data = ir2[3];
                ir2[3] = (v785_data + (v763_data * v783_bc));
                float v791_data = ir2[4];
                ir2[4] = (v791_data + (v763_data * v789_bc));
                float v797_data = ir2[5];
                ir2[5] = (v797_data + (v763_data * v795_bc));
                float v803_data = ir2[6];
                ir2[6] = (v803_data + (v763_data * v801_bc));
                float v809_data = ir2[7];
                ir2[7] = (v809_data + (v763_data * v807_bc));
              }
              // r2 = ir2
              if (v22_g) {
                #pragma unroll
                for (int32_t v811_n1 = 0; v811_n1 < 8; ++v811_n1) {
                  float v813_data = ir2[v811_n1];
                  r2[v811_n1] = v813_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v22_g) {
                int32_t v820_a = ((v21_lead + 16_i32) + -12) - 4;
                #pragma unroll
                for (int32_t v814_i1 = 0; v814_i1 < 8; ++v814_i1) {
                  float v816_data = r2[v814_i1];
                  glb_m0[(v820_a + (v814_i1 * 12))] = v816_data;
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


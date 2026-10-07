// === base name ===
kernel_2112cc6b84b0181f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2112cc6b84b0181f = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2112cc6b84b0181f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2112cc6b84b0181f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2112cc6b84b0181f(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_2112cc6b84b0181f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2112cc6b84b0181f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2112cc6b84b0181f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2112cc6b84b0181f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 256 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v21_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
                int32_t v25_lead = v21_lead + (v22_i0 * 16);
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                  float v28_data = glb_m1[(v25_lead + (v23_i1 * 16))];
                  r0[(v22_i0 + v23_i1)] = v28_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
                int32_t v34_lead = v21_lead + (v31_i0 * 16);
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 16; ++v32_i1) {
                  float v37_data = glb_m2[(v34_lead + (v32_i1 * 16))];
                  r1[(v31_i0 + v32_i1)] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v41_data = r0[0];
              float v42_data = r1[0];
              float v45_data = ir2[0];
              ir2[0] = (v45_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v48_data = r1[1];
              float v51_data = ir2[1];
              ir2[1] = (v51_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v54_data = r1[2];
              float v57_data = ir2[2];
              ir2[2] = (v57_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v60_data = r1[3];
              float v63_data = ir2[3];
              ir2[3] = (v63_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v66_data = r1[4];
              float v69_data = ir2[4];
              ir2[4] = (v69_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v72_data = r1[5];
              float v75_data = ir2[5];
              ir2[5] = (v75_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v78_data = r1[6];
              float v81_data = ir2[6];
              ir2[6] = (v81_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v84_data = r1[7];
              float v87_data = ir2[7];
              ir2[7] = (v87_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v90_data = r1[8];
              float v93_data = ir2[8];
              ir2[8] = (v93_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v96_data = r1[9];
              float v99_data = ir2[9];
              ir2[9] = (v99_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v102_data = r1[10];
              float v105_data = ir2[10];
              ir2[10] = (v105_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v108_data = r1[11];
              float v111_data = ir2[11];
              ir2[11] = (v111_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v114_data = r1[12];
              float v117_data = ir2[12];
              ir2[12] = (v117_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v120_data = r1[13];
              float v123_data = ir2[13];
              ir2[13] = (v123_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v126_data = r1[14];
              float v129_data = ir2[14];
              ir2[14] = (v129_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v132_data = r1[15];
              float v135_data = ir2[15];
              ir2[15] = (v135_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v137_data = r0[1];
              float v141_data = ir2[0];
              ir2[0] = (v141_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v147_data = ir2[1];
              ir2[1] = (v147_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v153_data = ir2[2];
              ir2[2] = (v153_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v159_data = ir2[3];
              ir2[3] = (v159_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v165_data = ir2[4];
              ir2[4] = (v165_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v171_data = ir2[5];
              ir2[5] = (v171_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v177_data = ir2[6];
              ir2[6] = (v177_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v183_data = ir2[7];
              ir2[7] = (v183_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v189_data = ir2[8];
              ir2[8] = (v189_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v195_data = ir2[9];
              ir2[9] = (v195_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v201_data = ir2[10];
              ir2[10] = (v201_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v207_data = ir2[11];
              ir2[11] = (v207_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v213_data = ir2[12];
              ir2[12] = (v213_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v219_data = ir2[13];
              ir2[13] = (v219_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v225_data = ir2[14];
              ir2[14] = (v225_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v231_data = ir2[15];
              ir2[15] = (v231_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v233_data = r0[2];
              float v237_data = ir2[0];
              ir2[0] = (v237_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v243_data = ir2[1];
              ir2[1] = (v243_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v249_data = ir2[2];
              ir2[2] = (v249_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v255_data = ir2[3];
              ir2[3] = (v255_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v261_data = ir2[4];
              ir2[4] = (v261_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v267_data = ir2[5];
              ir2[5] = (v267_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v273_data = ir2[6];
              ir2[6] = (v273_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v279_data = ir2[7];
              ir2[7] = (v279_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v285_data = ir2[8];
              ir2[8] = (v285_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v291_data = ir2[9];
              ir2[9] = (v291_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v297_data = ir2[10];
              ir2[10] = (v297_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v303_data = ir2[11];
              ir2[11] = (v303_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v309_data = ir2[12];
              ir2[12] = (v309_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v315_data = ir2[13];
              ir2[13] = (v315_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v321_data = ir2[14];
              ir2[14] = (v321_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v327_data = ir2[15];
              ir2[15] = (v327_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v329_data = r0[3];
              float v333_data = ir2[0];
              ir2[0] = (v333_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v339_data = ir2[1];
              ir2[1] = (v339_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v345_data = ir2[2];
              ir2[2] = (v345_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v351_data = ir2[3];
              ir2[3] = (v351_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v357_data = ir2[4];
              ir2[4] = (v357_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v363_data = ir2[5];
              ir2[5] = (v363_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v369_data = ir2[6];
              ir2[6] = (v369_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v375_data = ir2[7];
              ir2[7] = (v375_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v381_data = ir2[8];
              ir2[8] = (v381_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v387_data = ir2[9];
              ir2[9] = (v387_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v393_data = ir2[10];
              ir2[10] = (v393_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v399_data = ir2[11];
              ir2[11] = (v399_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v405_data = ir2[12];
              ir2[12] = (v405_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v411_data = ir2[13];
              ir2[13] = (v411_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v417_data = ir2[14];
              ir2[14] = (v417_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v423_data = ir2[15];
              ir2[15] = (v423_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v425_data = r0[4];
              float v429_data = ir2[0];
              ir2[0] = (v429_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v435_data = ir2[1];
              ir2[1] = (v435_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v441_data = ir2[2];
              ir2[2] = (v441_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v447_data = ir2[3];
              ir2[3] = (v447_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v453_data = ir2[4];
              ir2[4] = (v453_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v459_data = ir2[5];
              ir2[5] = (v459_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v465_data = ir2[6];
              ir2[6] = (v465_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v471_data = ir2[7];
              ir2[7] = (v471_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v477_data = ir2[8];
              ir2[8] = (v477_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v483_data = ir2[9];
              ir2[9] = (v483_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v489_data = ir2[10];
              ir2[10] = (v489_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v495_data = ir2[11];
              ir2[11] = (v495_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v501_data = ir2[12];
              ir2[12] = (v501_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v507_data = ir2[13];
              ir2[13] = (v507_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v513_data = ir2[14];
              ir2[14] = (v513_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v519_data = ir2[15];
              ir2[15] = (v519_data + (v425_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v521_data = r0[5];
              float v525_data = ir2[0];
              ir2[0] = (v525_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v531_data = ir2[1];
              ir2[1] = (v531_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v537_data = ir2[2];
              ir2[2] = (v537_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v543_data = ir2[3];
              ir2[3] = (v543_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v549_data = ir2[4];
              ir2[4] = (v549_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v555_data = ir2[5];
              ir2[5] = (v555_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v561_data = ir2[6];
              ir2[6] = (v561_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v567_data = ir2[7];
              ir2[7] = (v567_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v573_data = ir2[8];
              ir2[8] = (v573_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v579_data = ir2[9];
              ir2[9] = (v579_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v585_data = ir2[10];
              ir2[10] = (v585_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v591_data = ir2[11];
              ir2[11] = (v591_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v597_data = ir2[12];
              ir2[12] = (v597_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v603_data = ir2[13];
              ir2[13] = (v603_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v609_data = ir2[14];
              ir2[14] = (v609_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v615_data = ir2[15];
              ir2[15] = (v615_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v617_data = r0[6];
              float v621_data = ir2[0];
              ir2[0] = (v621_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v627_data = ir2[1];
              ir2[1] = (v627_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v633_data = ir2[2];
              ir2[2] = (v633_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v639_data = ir2[3];
              ir2[3] = (v639_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v645_data = ir2[4];
              ir2[4] = (v645_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v651_data = ir2[5];
              ir2[5] = (v651_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v657_data = ir2[6];
              ir2[6] = (v657_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v663_data = ir2[7];
              ir2[7] = (v663_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v669_data = ir2[8];
              ir2[8] = (v669_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v675_data = ir2[9];
              ir2[9] = (v675_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v681_data = ir2[10];
              ir2[10] = (v681_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v687_data = ir2[11];
              ir2[11] = (v687_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v693_data = ir2[12];
              ir2[12] = (v693_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v699_data = ir2[13];
              ir2[13] = (v699_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v705_data = ir2[14];
              ir2[14] = (v705_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v711_data = ir2[15];
              ir2[15] = (v711_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v713_data = r0[7];
              float v717_data = ir2[0];
              ir2[0] = (v717_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v723_data = ir2[1];
              ir2[1] = (v723_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v729_data = ir2[2];
              ir2[2] = (v729_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v735_data = ir2[3];
              ir2[3] = (v735_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v741_data = ir2[4];
              ir2[4] = (v741_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v747_data = ir2[5];
              ir2[5] = (v747_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v753_data = ir2[6];
              ir2[6] = (v753_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v759_data = ir2[7];
              ir2[7] = (v759_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v765_data = ir2[8];
              ir2[8] = (v765_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v771_data = ir2[9];
              ir2[9] = (v771_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v777_data = ir2[10];
              ir2[10] = (v777_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v783_data = ir2[11];
              ir2[11] = (v783_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v789_data = ir2[12];
              ir2[12] = (v789_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v795_data = ir2[13];
              ir2[13] = (v795_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v801_data = ir2[14];
              ir2[14] = (v801_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v807_data = ir2[15];
              ir2[15] = (v807_data + (v713_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v809_data = r0[8];
              float v813_data = ir2[0];
              ir2[0] = (v813_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v819_data = ir2[1];
              ir2[1] = (v819_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v825_data = ir2[2];
              ir2[2] = (v825_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v831_data = ir2[3];
              ir2[3] = (v831_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v837_data = ir2[4];
              ir2[4] = (v837_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v843_data = ir2[5];
              ir2[5] = (v843_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v849_data = ir2[6];
              ir2[6] = (v849_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v855_data = ir2[7];
              ir2[7] = (v855_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v861_data = ir2[8];
              ir2[8] = (v861_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v867_data = ir2[9];
              ir2[9] = (v867_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v873_data = ir2[10];
              ir2[10] = (v873_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v879_data = ir2[11];
              ir2[11] = (v879_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v885_data = ir2[12];
              ir2[12] = (v885_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v891_data = ir2[13];
              ir2[13] = (v891_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v897_data = ir2[14];
              ir2[14] = (v897_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v903_data = ir2[15];
              ir2[15] = (v903_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v905_data = r0[9];
              float v909_data = ir2[0];
              ir2[0] = (v909_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v915_data = ir2[1];
              ir2[1] = (v915_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v921_data = ir2[2];
              ir2[2] = (v921_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v927_data = ir2[3];
              ir2[3] = (v927_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v933_data = ir2[4];
              ir2[4] = (v933_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v939_data = ir2[5];
              ir2[5] = (v939_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v945_data = ir2[6];
              ir2[6] = (v945_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v951_data = ir2[7];
              ir2[7] = (v951_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v957_data = ir2[8];
              ir2[8] = (v957_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v963_data = ir2[9];
              ir2[9] = (v963_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v969_data = ir2[10];
              ir2[10] = (v969_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v975_data = ir2[11];
              ir2[11] = (v975_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v981_data = ir2[12];
              ir2[12] = (v981_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v987_data = ir2[13];
              ir2[13] = (v987_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v993_data = ir2[14];
              ir2[14] = (v993_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v999_data = ir2[15];
              ir2[15] = (v999_data + (v905_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1001_data = r0[10];
              float v1005_data = ir2[0];
              ir2[0] = (v1005_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1011_data = ir2[1];
              ir2[1] = (v1011_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1017_data = ir2[2];
              ir2[2] = (v1017_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1023_data = ir2[3];
              ir2[3] = (v1023_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1029_data = ir2[4];
              ir2[4] = (v1029_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1035_data = ir2[5];
              ir2[5] = (v1035_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1041_data = ir2[6];
              ir2[6] = (v1041_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1047_data = ir2[7];
              ir2[7] = (v1047_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1053_data = ir2[8];
              ir2[8] = (v1053_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1059_data = ir2[9];
              ir2[9] = (v1059_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1065_data = ir2[10];
              ir2[10] = (v1065_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1071_data = ir2[11];
              ir2[11] = (v1071_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1077_data = ir2[12];
              ir2[12] = (v1077_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1083_data = ir2[13];
              ir2[13] = (v1083_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1089_data = ir2[14];
              ir2[14] = (v1089_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1095_data = ir2[15];
              ir2[15] = (v1095_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1097_data = r0[11];
              float v1101_data = ir2[0];
              ir2[0] = (v1101_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1107_data = ir2[1];
              ir2[1] = (v1107_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1113_data = ir2[2];
              ir2[2] = (v1113_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1119_data = ir2[3];
              ir2[3] = (v1119_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1125_data = ir2[4];
              ir2[4] = (v1125_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1131_data = ir2[5];
              ir2[5] = (v1131_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1137_data = ir2[6];
              ir2[6] = (v1137_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1143_data = ir2[7];
              ir2[7] = (v1143_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1149_data = ir2[8];
              ir2[8] = (v1149_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1155_data = ir2[9];
              ir2[9] = (v1155_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1161_data = ir2[10];
              ir2[10] = (v1161_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1167_data = ir2[11];
              ir2[11] = (v1167_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1173_data = ir2[12];
              ir2[12] = (v1173_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1179_data = ir2[13];
              ir2[13] = (v1179_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1185_data = ir2[14];
              ir2[14] = (v1185_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1191_data = ir2[15];
              ir2[15] = (v1191_data + (v1097_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1193_data = r0[12];
              float v1197_data = ir2[0];
              ir2[0] = (v1197_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1203_data = ir2[1];
              ir2[1] = (v1203_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1209_data = ir2[2];
              ir2[2] = (v1209_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1215_data = ir2[3];
              ir2[3] = (v1215_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1221_data = ir2[4];
              ir2[4] = (v1221_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1227_data = ir2[5];
              ir2[5] = (v1227_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1233_data = ir2[6];
              ir2[6] = (v1233_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1239_data = ir2[7];
              ir2[7] = (v1239_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1245_data = ir2[8];
              ir2[8] = (v1245_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1251_data = ir2[9];
              ir2[9] = (v1251_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1257_data = ir2[10];
              ir2[10] = (v1257_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1263_data = ir2[11];
              ir2[11] = (v1263_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1269_data = ir2[12];
              ir2[12] = (v1269_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1275_data = ir2[13];
              ir2[13] = (v1275_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1281_data = ir2[14];
              ir2[14] = (v1281_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1287_data = ir2[15];
              ir2[15] = (v1287_data + (v1193_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1289_data = r0[13];
              float v1293_data = ir2[0];
              ir2[0] = (v1293_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1299_data = ir2[1];
              ir2[1] = (v1299_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1305_data = ir2[2];
              ir2[2] = (v1305_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1311_data = ir2[3];
              ir2[3] = (v1311_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1317_data = ir2[4];
              ir2[4] = (v1317_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1323_data = ir2[5];
              ir2[5] = (v1323_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1329_data = ir2[6];
              ir2[6] = (v1329_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1335_data = ir2[7];
              ir2[7] = (v1335_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1341_data = ir2[8];
              ir2[8] = (v1341_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1347_data = ir2[9];
              ir2[9] = (v1347_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1353_data = ir2[10];
              ir2[10] = (v1353_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1359_data = ir2[11];
              ir2[11] = (v1359_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1365_data = ir2[12];
              ir2[12] = (v1365_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1371_data = ir2[13];
              ir2[13] = (v1371_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1377_data = ir2[14];
              ir2[14] = (v1377_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1383_data = ir2[15];
              ir2[15] = (v1383_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1385_data = r0[14];
              float v1389_data = ir2[0];
              ir2[0] = (v1389_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1395_data = ir2[1];
              ir2[1] = (v1395_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1401_data = ir2[2];
              ir2[2] = (v1401_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1407_data = ir2[3];
              ir2[3] = (v1407_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1413_data = ir2[4];
              ir2[4] = (v1413_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1419_data = ir2[5];
              ir2[5] = (v1419_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1425_data = ir2[6];
              ir2[6] = (v1425_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1431_data = ir2[7];
              ir2[7] = (v1431_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1437_data = ir2[8];
              ir2[8] = (v1437_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1443_data = ir2[9];
              ir2[9] = (v1443_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1449_data = ir2[10];
              ir2[10] = (v1449_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1455_data = ir2[11];
              ir2[11] = (v1455_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1461_data = ir2[12];
              ir2[12] = (v1461_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1467_data = ir2[13];
              ir2[13] = (v1467_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1473_data = ir2[14];
              ir2[14] = (v1473_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1479_data = ir2[15];
              ir2[15] = (v1479_data + (v1385_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1481_data = r0[15];
              float v1485_data = ir2[0];
              ir2[0] = (v1485_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1491_data = ir2[1];
              ir2[1] = (v1491_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1497_data = ir2[2];
              ir2[2] = (v1497_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1503_data = ir2[3];
              ir2[3] = (v1503_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1509_data = ir2[4];
              ir2[4] = (v1509_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1515_data = ir2[5];
              ir2[5] = (v1515_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1521_data = ir2[6];
              ir2[6] = (v1521_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1527_data = ir2[7];
              ir2[7] = (v1527_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1533_data = ir2[8];
              ir2[8] = (v1533_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1539_data = ir2[9];
              ir2[9] = (v1539_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1545_data = ir2[10];
              ir2[10] = (v1545_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1551_data = ir2[11];
              ir2[11] = (v1551_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1557_data = ir2[12];
              ir2[12] = (v1557_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1563_data = ir2[13];
              ir2[13] = (v1563_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1569_data = ir2[14];
              ir2[14] = (v1569_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1575_data = ir2[15];
              ir2[15] = (v1575_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v1577_n0 = 0; v1577_n0 < 1; ++v1577_n0) {
                #pragma unroll
                for (int32_t v1578_n1 = 0; v1578_n1 < 16; ++v1578_n1) {
                  int32_t v1579_a = v1577_n0 + v1578_n1;
                  float v1580_data = ir2[v1579_a];
                  r2[v1579_a] = v1580_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1581_i0 = 0; v1581_i0 < 1; ++v1581_i0) {
                int32_t v1586_lead = v21_lead + (v1581_i0 * 16);
                #pragma unroll
                for (int32_t v1582_i1 = 0; v1582_i1 < 16; ++v1582_i1) {
                  float v1584_data = r2[(v1581_i0 + v1582_i1)];
                  glb_m0[(v1586_lead + (v1582_i1 * 16))] = v1584_data;
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


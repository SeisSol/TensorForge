// === base name ===
kernel_aca4ad06504cf59d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_aca4ad06504cf59d = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_aca4ad06504cf59d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_aca4ad06504cf59d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_aca4ad06504cf59d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_aca4ad06504cf59d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_aca4ad06504cf59d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_aca4ad06504cf59d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_aca4ad06504cf59d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(4×12) {4..8}×{0..12} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   D = abs(N)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 48 + 0 + m3_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v22_lead = item.get_local_id(2) % 16;
              bool v23_g = v22_lead < 12;
              if (v23_g) {
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                  float v29_data = glb_m1[(v22_lead + (v24_i1 * 12))];
                  r0[v24_i1] = v29_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m2);
              if (v23_g) {
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
                  float v37_data = glb_m2[(v22_lead + (v32_i1 * 12))];
                  r1[v32_i1] = v37_data;
                }
              }
              float r2[12]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir2[12]{};
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
              float v113_data = r0[1];
              float v117_data = ir2[0];
              ir2[0] = (v117_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v123_data = ir2[1];
              ir2[1] = (v123_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v129_data = ir2[2];
              ir2[2] = (v129_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v135_data = ir2[3];
              ir2[3] = (v135_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v141_data = ir2[4];
              ir2[4] = (v141_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v147_data = ir2[5];
              ir2[5] = (v147_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v153_data = ir2[6];
              ir2[6] = (v153_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v159_data = ir2[7];
              ir2[7] = (v159_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v165_data = ir2[8];
              ir2[8] = (v165_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v171_data = ir2[9];
              ir2[9] = (v171_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v177_data = ir2[10];
              ir2[10] = (v177_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v183_data = ir2[11];
              ir2[11] = (v183_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v185_data = r0[2];
              float v189_data = ir2[0];
              ir2[0] = (v189_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v195_data = ir2[1];
              ir2[1] = (v195_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v201_data = ir2[2];
              ir2[2] = (v201_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v207_data = ir2[3];
              ir2[3] = (v207_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v213_data = ir2[4];
              ir2[4] = (v213_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v219_data = ir2[5];
              ir2[5] = (v219_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v225_data = ir2[6];
              ir2[6] = (v225_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v231_data = ir2[7];
              ir2[7] = (v231_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v237_data = ir2[8];
              ir2[8] = (v237_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v243_data = ir2[9];
              ir2[9] = (v243_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v249_data = ir2[10];
              ir2[10] = (v249_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v255_data = ir2[11];
              ir2[11] = (v255_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v257_data = r0[3];
              float v261_data = ir2[0];
              ir2[0] = (v261_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v267_data = ir2[1];
              ir2[1] = (v267_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v273_data = ir2[2];
              ir2[2] = (v273_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v279_data = ir2[3];
              ir2[3] = (v279_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v285_data = ir2[4];
              ir2[4] = (v285_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v291_data = ir2[5];
              ir2[5] = (v291_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v297_data = ir2[6];
              ir2[6] = (v297_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v303_data = ir2[7];
              ir2[7] = (v303_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v309_data = ir2[8];
              ir2[8] = (v309_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v315_data = ir2[9];
              ir2[9] = (v315_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v321_data = ir2[10];
              ir2[10] = (v321_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v327_data = ir2[11];
              ir2[11] = (v327_data + (v257_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v329_data = r0[4];
              float v333_data = ir2[0];
              ir2[0] = (v333_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v339_data = ir2[1];
              ir2[1] = (v339_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v345_data = ir2[2];
              ir2[2] = (v345_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v351_data = ir2[3];
              ir2[3] = (v351_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v357_data = ir2[4];
              ir2[4] = (v357_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v363_data = ir2[5];
              ir2[5] = (v363_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v369_data = ir2[6];
              ir2[6] = (v369_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v375_data = ir2[7];
              ir2[7] = (v375_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v381_data = ir2[8];
              ir2[8] = (v381_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v387_data = ir2[9];
              ir2[9] = (v387_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v393_data = ir2[10];
              ir2[10] = (v393_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v399_data = ir2[11];
              ir2[11] = (v399_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v401_data = r0[5];
              float v405_data = ir2[0];
              ir2[0] = (v405_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v411_data = ir2[1];
              ir2[1] = (v411_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v417_data = ir2[2];
              ir2[2] = (v417_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v423_data = ir2[3];
              ir2[3] = (v423_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v429_data = ir2[4];
              ir2[4] = (v429_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v435_data = ir2[5];
              ir2[5] = (v435_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v441_data = ir2[6];
              ir2[6] = (v441_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v447_data = ir2[7];
              ir2[7] = (v447_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v453_data = ir2[8];
              ir2[8] = (v453_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v459_data = ir2[9];
              ir2[9] = (v459_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v465_data = ir2[10];
              ir2[10] = (v465_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v471_data = ir2[11];
              ir2[11] = (v471_data + (v401_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v473_data = r0[6];
              float v477_data = ir2[0];
              ir2[0] = (v477_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v483_data = ir2[1];
              ir2[1] = (v483_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v489_data = ir2[2];
              ir2[2] = (v489_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v495_data = ir2[3];
              ir2[3] = (v495_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v501_data = ir2[4];
              ir2[4] = (v501_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v507_data = ir2[5];
              ir2[5] = (v507_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v513_data = ir2[6];
              ir2[6] = (v513_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v519_data = ir2[7];
              ir2[7] = (v519_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v525_data = ir2[8];
              ir2[8] = (v525_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v531_data = ir2[9];
              ir2[9] = (v531_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v537_data = ir2[10];
              ir2[10] = (v537_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v543_data = ir2[11];
              ir2[11] = (v543_data + (v473_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v545_data = r0[7];
              float v549_data = ir2[0];
              ir2[0] = (v549_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v555_data = ir2[1];
              ir2[1] = (v555_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v561_data = ir2[2];
              ir2[2] = (v561_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v567_data = ir2[3];
              ir2[3] = (v567_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v573_data = ir2[4];
              ir2[4] = (v573_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v579_data = ir2[5];
              ir2[5] = (v579_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v585_data = ir2[6];
              ir2[6] = (v585_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v591_data = ir2[7];
              ir2[7] = (v591_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v597_data = ir2[8];
              ir2[8] = (v597_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v603_data = ir2[9];
              ir2[9] = (v603_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v609_data = ir2[10];
              ir2[10] = (v609_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v615_data = ir2[11];
              ir2[11] = (v615_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v617_data = r0[8];
              float v621_data = ir2[0];
              ir2[0] = (v621_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v627_data = ir2[1];
              ir2[1] = (v627_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v633_data = ir2[2];
              ir2[2] = (v633_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v639_data = ir2[3];
              ir2[3] = (v639_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v645_data = ir2[4];
              ir2[4] = (v645_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v651_data = ir2[5];
              ir2[5] = (v651_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v657_data = ir2[6];
              ir2[6] = (v657_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v663_data = ir2[7];
              ir2[7] = (v663_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v669_data = ir2[8];
              ir2[8] = (v669_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v675_data = ir2[9];
              ir2[9] = (v675_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v681_data = ir2[10];
              ir2[10] = (v681_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v687_data = ir2[11];
              ir2[11] = (v687_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v689_data = r0[9];
              float v693_data = ir2[0];
              ir2[0] = (v693_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v699_data = ir2[1];
              ir2[1] = (v699_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v705_data = ir2[2];
              ir2[2] = (v705_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v711_data = ir2[3];
              ir2[3] = (v711_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v717_data = ir2[4];
              ir2[4] = (v717_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v723_data = ir2[5];
              ir2[5] = (v723_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v729_data = ir2[6];
              ir2[6] = (v729_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v735_data = ir2[7];
              ir2[7] = (v735_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v741_data = ir2[8];
              ir2[8] = (v741_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v747_data = ir2[9];
              ir2[9] = (v747_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v753_data = ir2[10];
              ir2[10] = (v753_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v759_data = ir2[11];
              ir2[11] = (v759_data + (v689_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v761_data = r0[10];
              float v765_data = ir2[0];
              ir2[0] = (v765_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v771_data = ir2[1];
              ir2[1] = (v771_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v777_data = ir2[2];
              ir2[2] = (v777_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v783_data = ir2[3];
              ir2[3] = (v783_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v789_data = ir2[4];
              ir2[4] = (v789_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v795_data = ir2[5];
              ir2[5] = (v795_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v801_data = ir2[6];
              ir2[6] = (v801_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v807_data = ir2[7];
              ir2[7] = (v807_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v813_data = ir2[8];
              ir2[8] = (v813_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v819_data = ir2[9];
              ir2[9] = (v819_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v825_data = ir2[10];
              ir2[10] = (v825_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v831_data = ir2[11];
              ir2[11] = (v831_data + (v761_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v833_data = r0[11];
              float v837_data = ir2[0];
              ir2[0] = (v837_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v843_data = ir2[1];
              ir2[1] = (v843_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v849_data = ir2[2];
              ir2[2] = (v849_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v855_data = ir2[3];
              ir2[3] = (v855_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v861_data = ir2[4];
              ir2[4] = (v861_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v867_data = ir2[5];
              ir2[5] = (v867_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v873_data = ir2[6];
              ir2[6] = (v873_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v879_data = ir2[7];
              ir2[7] = (v879_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v885_data = ir2[8];
              ir2[8] = (v885_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v891_data = ir2[9];
              ir2[9] = (v891_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v897_data = ir2[10];
              ir2[10] = (v897_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v903_data = ir2[11];
              ir2[11] = (v903_data + (v833_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r2 = ir2
              if (v23_g) {
                #pragma unroll
                for (int32_t v905_n1 = 0; v905_n1 < 12; ++v905_n1) {
                  float v907_data = ir2[v905_n1];
                  r2[v905_n1] = v907_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v23_g) {
                #pragma unroll
                for (int32_t v908_i1 = 0; v908_i1 < 12; ++v908_i1) {
                  float v910_data = r2[v908_i1];
                  glb_m0[(v22_lead + (v908_i1 * 12))] = v910_data;
                }
              }
              float r3[12]{};
              // r3 = abs(glb_m3)
              bool v916_g = v22_lead < 4;
              if (v916_g) {
                int32_t v921_a = (v22_lead + 4) - 4;
                #pragma unroll
                for (int32_t v917_k1 = 0; v917_k1 < 12; ++v917_k1) {
                  float v924_data = glb_m3[(v921_a + (v917_k1 * 4))];
                  r3[v917_k1] = (sycl::fabs(v924_data));
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v916_g) {
                int32_t v933_off = v22_lead + 4;
                #pragma unroll
                for (int32_t v928_i1 = 0; v928_i1 < 12; ++v928_i1) {
                  float v930_data = r3[v928_i1];
                  glb_m0[(v933_off + (v928_i1 * 12))] = v930_data;
                }
              }
              if (v22_lead >= 12) {
                int32_t v941_off = (v22_lead + -16_i32) + 4;
                #pragma unroll
                for (int32_t v937_z1 = 0; v937_z1 < 12; ++v937_z1) {
                  glb_m0[(v941_off + (v937_z1 * 12))] = 0.0f;
                }
              }
              if ((v22_lead >= 4) && (v22_lead < 8)) {
                int32_t v951_off = v22_lead + 4;
                #pragma unroll
                for (int32_t v947_z1 = 0; v947_z1 < 12; ++v947_z1) {
                  glb_m0[(v951_off + (v947_z1 * 12))] = 0.0f;
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


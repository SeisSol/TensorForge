// === base name ===
kernel_f35e389fbfa60467

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f35e389fbfa60467 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f35e389fbfa60467(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f35e389fbfa60467(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f35e389fbfa60467(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_f35e389fbfa60467(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f35e389fbfa60467(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f35e389fbfa60467(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f35e389fbfa60467(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×6) {0..12}×{0..6} strided
        //   m1 32×32(6×6) {0..6}×{0..6} strided
        //   m2 32×32(12×6) {0..12}×{0..6} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[6,6]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,6]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[6,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 144 + 0 + m3_extraOffset];
              float r0[6]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v22_lead = item.get_local_id(2) % 16;
              bool v23_g = v22_lead < 12;
              if (v23_g) {
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 6; ++v24_i1) {
                  float v29_data = glb_m0[(v22_lead + (v24_i1 * 12))];
                  r0[v24_i1] = v29_data;
                }
              }
              float r1[6]{};
              // r1 = load{g>r}(glb_m1);
              if (v22_lead < 6) {
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 6; ++v33_i1) {
                  float v38_data = glb_m1[(v22_lead + (v33_i1 * 6))];
                  r1[v33_i1] = v38_data;
                }
              }
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v23_g) {
                #pragma unroll
                for (int32_t v258_i1 = 0; v258_i1 < 12; ++v258_i1) {
                  float v263_data = glb_m3[(v22_lead + (v258_i1 * 12))];
                  r3[v258_i1] = v263_data;
                }
              }
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              float v41_data = r0[0];
              float v42_data = r1[0];
              float v45_data = r2[0];
              r2[0] = (v45_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v48_data = r1[1];
              float v51_data = r2[1];
              r2[1] = (v51_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v54_data = r1[2];
              float v57_data = r2[2];
              r2[2] = (v57_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v60_data = r1[3];
              float v63_data = r2[3];
              r2[3] = (v63_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v66_data = r1[4];
              float v69_data = r2[4];
              r2[4] = (v69_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v72_data = r1[5];
              float v75_data = r2[5];
              r2[5] = (v75_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v77_data = r0[1];
              float v81_data = r2[0];
              r2[0] = (v81_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v87_data = r2[1];
              r2[1] = (v87_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v93_data = r2[2];
              r2[2] = (v93_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v99_data = r2[3];
              r2[3] = (v99_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v105_data = r2[4];
              r2[4] = (v105_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v111_data = r2[5];
              r2[5] = (v111_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v113_data = r0[2];
              float v117_data = r2[0];
              r2[0] = (v117_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v123_data = r2[1];
              r2[1] = (v123_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v129_data = r2[2];
              r2[2] = (v129_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v135_data = r2[3];
              r2[3] = (v135_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v141_data = r2[4];
              r2[4] = (v141_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v147_data = r2[5];
              r2[5] = (v147_data + (v113_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v149_data = r0[3];
              float v153_data = r2[0];
              r2[0] = (v153_data + (v149_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v159_data = r2[1];
              r2[1] = (v159_data + (v149_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v165_data = r2[2];
              r2[2] = (v165_data + (v149_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v171_data = r2[3];
              r2[3] = (v171_data + (v149_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v177_data = r2[4];
              r2[4] = (v177_data + (v149_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v183_data = r2[5];
              r2[5] = (v183_data + (v149_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v185_data = r0[4];
              float v189_data = r2[0];
              r2[0] = (v189_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v195_data = r2[1];
              r2[1] = (v195_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v201_data = r2[2];
              r2[2] = (v201_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v207_data = r2[3];
              r2[3] = (v207_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v213_data = r2[4];
              r2[4] = (v213_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v219_data = r2[5];
              r2[5] = (v219_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v221_data = r0[5];
              float v225_data = r2[0];
              r2[0] = (v225_data + (v221_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v231_data = r2[1];
              r2[1] = (v231_data + (v221_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v237_data = r2[2];
              r2[2] = (v237_data + (v221_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v243_data = r2[3];
              r2[3] = (v243_data + (v221_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v249_data = r2[4];
              r2[4] = (v249_data + (v221_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v255_data = r2[5];
              r2[5] = (v255_data + (v221_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float r4[6]{};
              // ir4 = +(r3 * r2)
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir4[6]{};
              float v267_data = r3[0];
              float v268_data = r2[0];
              float v271_data = ir4[0];
              ir4[0] = (v271_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v274_data = r2[1];
              float v277_data = ir4[1];
              ir4[1] = (v277_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v280_data = r2[2];
              float v283_data = ir4[2];
              ir4[2] = (v283_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v286_data = r2[3];
              float v289_data = ir4[3];
              ir4[3] = (v289_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v292_data = r2[4];
              float v295_data = ir4[4];
              ir4[4] = (v295_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v298_data = r2[5];
              float v301_data = ir4[5];
              ir4[5] = (v301_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v303_data = r3[1];
              float v307_data = ir4[0];
              ir4[0] = (v307_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v313_data = ir4[1];
              ir4[1] = (v313_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v319_data = ir4[2];
              ir4[2] = (v319_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v325_data = ir4[3];
              ir4[3] = (v325_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v331_data = ir4[4];
              ir4[4] = (v331_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v337_data = ir4[5];
              ir4[5] = (v337_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v339_data = r3[2];
              float v343_data = ir4[0];
              ir4[0] = (v343_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v349_data = ir4[1];
              ir4[1] = (v349_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v355_data = ir4[2];
              ir4[2] = (v355_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v361_data = ir4[3];
              ir4[3] = (v361_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v367_data = ir4[4];
              ir4[4] = (v367_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v373_data = ir4[5];
              ir4[5] = (v373_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v375_data = r3[3];
              float v379_data = ir4[0];
              ir4[0] = (v379_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v385_data = ir4[1];
              ir4[1] = (v385_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v391_data = ir4[2];
              ir4[2] = (v391_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v397_data = ir4[3];
              ir4[3] = (v397_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v403_data = ir4[4];
              ir4[4] = (v403_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v409_data = ir4[5];
              ir4[5] = (v409_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v411_data = r3[4];
              float v415_data = ir4[0];
              ir4[0] = (v415_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v421_data = ir4[1];
              ir4[1] = (v421_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v427_data = ir4[2];
              ir4[2] = (v427_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v433_data = ir4[3];
              ir4[3] = (v433_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v439_data = ir4[4];
              ir4[4] = (v439_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v445_data = ir4[5];
              ir4[5] = (v445_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v447_data = r3[5];
              float v451_data = ir4[0];
              ir4[0] = (v451_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v457_data = ir4[1];
              ir4[1] = (v457_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v463_data = ir4[2];
              ir4[2] = (v463_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v469_data = ir4[3];
              ir4[3] = (v469_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v475_data = ir4[4];
              ir4[4] = (v475_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v481_data = ir4[5];
              ir4[5] = (v481_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v483_data = r3[6];
              float v487_data = ir4[0];
              ir4[0] = (v487_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v493_data = ir4[1];
              ir4[1] = (v493_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v499_data = ir4[2];
              ir4[2] = (v499_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v505_data = ir4[3];
              ir4[3] = (v505_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v511_data = ir4[4];
              ir4[4] = (v511_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v517_data = ir4[5];
              ir4[5] = (v517_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v519_data = r3[7];
              float v523_data = ir4[0];
              ir4[0] = (v523_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v529_data = ir4[1];
              ir4[1] = (v529_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v535_data = ir4[2];
              ir4[2] = (v535_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v541_data = ir4[3];
              ir4[3] = (v541_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v547_data = ir4[4];
              ir4[4] = (v547_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v553_data = ir4[5];
              ir4[5] = (v553_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v555_data = r3[8];
              float v559_data = ir4[0];
              ir4[0] = (v559_data + (v555_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v565_data = ir4[1];
              ir4[1] = (v565_data + (v555_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v571_data = ir4[2];
              ir4[2] = (v571_data + (v555_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v577_data = ir4[3];
              ir4[3] = (v577_data + (v555_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v583_data = ir4[4];
              ir4[4] = (v583_data + (v555_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v589_data = ir4[5];
              ir4[5] = (v589_data + (v555_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v591_data = r3[9];
              float v595_data = ir4[0];
              ir4[0] = (v595_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v601_data = ir4[1];
              ir4[1] = (v601_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v607_data = ir4[2];
              ir4[2] = (v607_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v613_data = ir4[3];
              ir4[3] = (v613_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v619_data = ir4[4];
              ir4[4] = (v619_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v625_data = ir4[5];
              ir4[5] = (v625_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v627_data = r3[10];
              float v631_data = ir4[0];
              ir4[0] = (v631_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v637_data = ir4[1];
              ir4[1] = (v637_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v643_data = ir4[2];
              ir4[2] = (v643_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v649_data = ir4[3];
              ir4[3] = (v649_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v655_data = ir4[4];
              ir4[4] = (v655_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v661_data = ir4[5];
              ir4[5] = (v661_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v663_data = r3[11];
              float v667_data = ir4[0];
              ir4[0] = (v667_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v673_data = ir4[1];
              ir4[1] = (v673_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v274_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v679_data = ir4[2];
              ir4[2] = (v679_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v280_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v685_data = ir4[3];
              ir4[3] = (v685_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v286_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v691_data = ir4[4];
              ir4[4] = (v691_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v292_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v697_data = ir4[5];
              ir4[5] = (v697_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v298_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r4 = ir4
              if (v23_g) {
                #pragma unroll
                for (int32_t v699_n1 = 0; v699_n1 < 6; ++v699_n1) {
                  float v701_data = ir4[v699_n1];
                  r4[v699_n1] = v701_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v23_g) {
                #pragma unroll
                for (int32_t v702_i1 = 0; v702_i1 < 6; ++v702_i1) {
                  float v704_data = r4[v702_i1];
                  glb_m2[(v22_lead + (v702_i1 * 12))] = v704_data;
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


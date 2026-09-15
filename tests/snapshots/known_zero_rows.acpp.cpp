// === base name ===
kernel_28299604e3ae8d9e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_28299604e3ae8d9e = {{32, 1, 1}, 32, 40, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_28299604e3ae8d9e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_28299604e3ae8d9e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_28299604e3ae8d9e(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_28299604e3ae8d9e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_28299604e3ae8d9e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_28299604e3ae8d9e(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_28299604e3ae8d9e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (40 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 240 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 48 + 0 + m2_extraOffset];
              float r0[6]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v15_lead = item.get_local_id(2) % 32;
              bool v16_g = v15_lead < 8;
              if (v16_g) {
                #pragma unroll
                for (int32_t v17_i1 = 0; v17_i1 < 6; ++v17_i1) {
                  float v22_data = glb_m2[(v15_lead + (v17_i1 * 8))];
                  r0[v17_i1] = v22_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[12]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 40), (0, 6)] [(0, 8)]
              float ir1[12]{};
              int32_t v27_lead = v15_lead + 32_i32;
              float v29_data = v16_g ? (glb_m1[v27_lead]) : (0.0f);
              float v30_data = r0[0];
              float v33_data = ir1[1];
              ir1[1] = (v33_data + (v29_data * (sycl::select_from_group(item.get_sub_group(), v30_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v36_data = r0[1];
              float v39_data = ir1[3];
              ir1[3] = (v39_data + (v29_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v42_data = r0[2];
              float v45_data = ir1[5];
              ir1[5] = (v45_data + (v29_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v48_data = r0[3];
              float v51_data = ir1[7];
              ir1[7] = (v51_data + (v29_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v54_data = r0[4];
              float v57_data = ir1[9];
              ir1[9] = (v57_data + (v29_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v60_data = r0[5];
              float v63_data = ir1[11];
              ir1[11] = (v63_data + (v29_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v68_data = glb_m1[(v15_lead + 40)];
              float v70_bc = sycl::select_from_group(item.get_sub_group(), v30_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v72_data = ir1[0];
              ir1[0] = (v72_data + (v68_data * v70_bc));
              float v76_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v78_data = ir1[2];
              ir1[2] = (v78_data + (v68_data * v76_bc));
              float v82_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v84_data = ir1[4];
              ir1[4] = (v84_data + (v68_data * v82_bc));
              float v88_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v90_data = ir1[6];
              ir1[6] = (v90_data + (v68_data * v88_bc));
              float v94_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v96_data = ir1[8];
              ir1[8] = (v96_data + (v68_data * v94_bc));
              float v100_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v102_data = ir1[10];
              ir1[10] = (v102_data + (v68_data * v100_bc));
              float v105_data = v16_g ? (glb_m1[(v27_lead + 40)]) : (0.0f);
              float v109_data = ir1[1];
              ir1[1] = (v109_data + (v105_data * v70_bc));
              float v115_data = ir1[3];
              ir1[3] = (v115_data + (v105_data * v76_bc));
              float v121_data = ir1[5];
              ir1[5] = (v121_data + (v105_data * v82_bc));
              float v127_data = ir1[7];
              ir1[7] = (v127_data + (v105_data * v88_bc));
              float v133_data = ir1[9];
              ir1[9] = (v133_data + (v105_data * v94_bc));
              float v139_data = ir1[11];
              ir1[11] = (v139_data + (v105_data * v100_bc));
              float v142_data = v16_g ? (glb_m1[(v27_lead + 80)]) : (0.0f);
              float v146_data = ir1[1];
              ir1[1] = (v146_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v30_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v152_data = ir1[3];
              ir1[3] = (v152_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v158_data = ir1[5];
              ir1[5] = (v158_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v164_data = ir1[7];
              ir1[7] = (v164_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v170_data = ir1[9];
              ir1[9] = (v170_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v176_data = ir1[11];
              ir1[11] = (v176_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v179_data = glb_m1[(v15_lead + 120)];
              float v181_bc = sycl::select_from_group(item.get_sub_group(), v30_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v183_data = ir1[0];
              ir1[0] = (v183_data + (v179_data * v181_bc));
              float v187_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v189_data = ir1[2];
              ir1[2] = (v189_data + (v179_data * v187_bc));
              float v193_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v195_data = ir1[4];
              ir1[4] = (v195_data + (v179_data * v193_bc));
              float v199_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v201_data = ir1[6];
              ir1[6] = (v201_data + (v179_data * v199_bc));
              float v205_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v207_data = ir1[8];
              ir1[8] = (v207_data + (v179_data * v205_bc));
              float v211_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v213_data = ir1[10];
              ir1[10] = (v213_data + (v179_data * v211_bc));
              float v216_data = v16_g ? (glb_m1[(v27_lead + 120)]) : (0.0f);
              float v220_data = ir1[1];
              ir1[1] = (v220_data + (v216_data * v181_bc));
              float v226_data = ir1[3];
              ir1[3] = (v226_data + (v216_data * v187_bc));
              float v232_data = ir1[5];
              ir1[5] = (v232_data + (v216_data * v193_bc));
              float v238_data = ir1[7];
              ir1[7] = (v238_data + (v216_data * v199_bc));
              float v244_data = ir1[9];
              ir1[9] = (v244_data + (v216_data * v205_bc));
              float v250_data = ir1[11];
              ir1[11] = (v250_data + (v216_data * v211_bc));
              float v253_data = glb_m1[(v15_lead + 160)];
              float v255_bc = sycl::select_from_group(item.get_sub_group(), v30_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v257_data = ir1[0];
              ir1[0] = (v257_data + (v253_data * v255_bc));
              float v261_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v263_data = ir1[2];
              ir1[2] = (v263_data + (v253_data * v261_bc));
              float v267_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v269_data = ir1[4];
              ir1[4] = (v269_data + (v253_data * v267_bc));
              float v273_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v275_data = ir1[6];
              ir1[6] = (v275_data + (v253_data * v273_bc));
              float v279_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v281_data = ir1[8];
              ir1[8] = (v281_data + (v253_data * v279_bc));
              float v285_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v287_data = ir1[10];
              ir1[10] = (v287_data + (v253_data * v285_bc));
              float v290_data = v16_g ? (glb_m1[(v27_lead + 160)]) : (0.0f);
              float v294_data = ir1[1];
              ir1[1] = (v294_data + (v290_data * v255_bc));
              float v300_data = ir1[3];
              ir1[3] = (v300_data + (v290_data * v261_bc));
              float v306_data = ir1[5];
              ir1[5] = (v306_data + (v290_data * v267_bc));
              float v312_data = ir1[7];
              ir1[7] = (v312_data + (v290_data * v273_bc));
              float v318_data = ir1[9];
              ir1[9] = (v318_data + (v290_data * v279_bc));
              float v324_data = ir1[11];
              ir1[11] = (v324_data + (v290_data * v285_bc));
              float v327_data = v16_g ? (glb_m1[(v27_lead + 200)]) : (0.0f);
              float v331_data = ir1[1];
              ir1[1] = (v331_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v30_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v337_data = ir1[3];
              ir1[3] = (v337_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v343_data = ir1[5];
              ir1[5] = (v343_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v349_data = ir1[7];
              ir1[7] = (v349_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v355_data = ir1[9];
              ir1[9] = (v355_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v361_data = ir1[11];
              ir1[11] = (v361_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v364_data = glb_m1[(v15_lead + 240)];
              float v366_bc = sycl::select_from_group(item.get_sub_group(), v30_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v368_data = ir1[0];
              ir1[0] = (v368_data + (v364_data * v366_bc));
              float v372_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v374_data = ir1[2];
              ir1[2] = (v374_data + (v364_data * v372_bc));
              float v378_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v380_data = ir1[4];
              ir1[4] = (v380_data + (v364_data * v378_bc));
              float v384_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v386_data = ir1[6];
              ir1[6] = (v386_data + (v364_data * v384_bc));
              float v390_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v392_data = ir1[8];
              ir1[8] = (v392_data + (v364_data * v390_bc));
              float v396_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v398_data = ir1[10];
              ir1[10] = (v398_data + (v364_data * v396_bc));
              float v401_data = v16_g ? (glb_m1[(v27_lead + 240)]) : (0.0f);
              float v405_data = ir1[1];
              ir1[1] = (v405_data + (v401_data * v366_bc));
              float v411_data = ir1[3];
              ir1[3] = (v411_data + (v401_data * v372_bc));
              float v417_data = ir1[5];
              ir1[5] = (v417_data + (v401_data * v378_bc));
              float v423_data = ir1[7];
              ir1[7] = (v423_data + (v401_data * v384_bc));
              float v429_data = ir1[9];
              ir1[9] = (v429_data + (v401_data * v390_bc));
              float v435_data = ir1[11];
              ir1[11] = (v435_data + (v401_data * v396_bc));
              float v438_data = glb_m1[(v15_lead + 280)];
              float v440_bc = sycl::select_from_group(item.get_sub_group(), v30_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v442_data = ir1[0];
              ir1[0] = (v442_data + (v438_data * v440_bc));
              float v446_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v448_data = ir1[2];
              ir1[2] = (v448_data + (v438_data * v446_bc));
              float v452_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v454_data = ir1[4];
              ir1[4] = (v454_data + (v438_data * v452_bc));
              float v458_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v460_data = ir1[6];
              ir1[6] = (v460_data + (v438_data * v458_bc));
              float v464_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v466_data = ir1[8];
              ir1[8] = (v466_data + (v438_data * v464_bc));
              float v470_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v472_data = ir1[10];
              ir1[10] = (v472_data + (v438_data * v470_bc));
              float v475_data = v16_g ? (glb_m1[(v27_lead + 280)]) : (0.0f);
              float v479_data = ir1[1];
              ir1[1] = (v479_data + (v475_data * v440_bc));
              float v485_data = ir1[3];
              ir1[3] = (v485_data + (v475_data * v446_bc));
              float v491_data = ir1[5];
              ir1[5] = (v491_data + (v475_data * v452_bc));
              float v497_data = ir1[7];
              ir1[7] = (v497_data + (v475_data * v458_bc));
              float v503_data = ir1[9];
              ir1[9] = (v503_data + (v475_data * v464_bc));
              float v509_data = ir1[11];
              ir1[11] = (v509_data + (v475_data * v470_bc));
              // r1 = ir1
              #pragma unroll
              for (int32_t v511_n0 = 0; v511_n0 < 1; ++v511_n0) {
                #pragma unroll
                for (int32_t v512_n1 = 0; v512_n1 < 6; ++v512_n1) {
                  int32_t v514_a = v511_n0 + (v512_n1 * 2);
                  float v515_data = ir1[v514_a];
                  r1[v514_a] = v515_data;
                }
              }
              if (v16_g) {
                #pragma unroll
                for (int32_t v516_n1 = 0; v516_n1 < 6; ++v516_n1) {
                  int32_t v518_a = 1 + (v516_n1 * 2);
                  float v519_data = ir1[v518_a];
                  r1[v518_a] = v519_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v520_i0 = 0; v520_i0 < 1; ++v520_i0) {
                int32_t v526_lead = v15_lead + (v520_i0 * 32);
                #pragma unroll
                for (int32_t v521_i1 = 0; v521_i1 < 6; ++v521_i1) {
                  float v524_data = r1[(v520_i0 + (v521_i1 * 2))];
                  glb_m0[(v526_lead + (v521_i1 * 40))] = v524_data;
                }
              }
              if (v16_g) {
                #pragma unroll
                for (int32_t v529_i1 = 0; v529_i1 < 6; ++v529_i1) {
                  float v532_data = r1[(1 + (v529_i1 * 2))];
                  glb_m0[(v27_lead + (v529_i1 * 40))] = v532_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}


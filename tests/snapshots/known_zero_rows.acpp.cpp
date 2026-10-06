// === base name ===
kernel_87dff0909fca5891

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_87dff0909fca5891 = {{32, 1, 1}, 32, 40, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_87dff0909fca5891(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_87dff0909fca5891(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_87dff0909fca5891(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 1 - 1) / 1;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_87dff0909fca5891(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_87dff0909fca5891(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_87dff0909fca5891(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_87dff0909fca5891(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (40 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 240 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 48 + 0 + m2_extraOffset];
              float r0[6]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v21_lead = item.get_local_id(2) % 32;
              bool v22_g = v21_lead < 8;
              if (v22_g) {
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 6; ++v23_i1) {
                  float v28_data = glb_m2[(v21_lead + (v23_i1 * 8))];
                  r0[v23_i1] = v28_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[12]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 40), (0, 6)] [(0, 8)]
              float ir1[12]{};
              int32_t v33_lead = v21_lead + 32_i32;
              float v35_data_pre = glb_m1[v22_g ? (v33_lead) : (0)];
              float v35_data = v22_g ? (v35_data_pre) : (0.0f);
              float v36_data = r0[0];
              float v39_data = ir1[1];
              ir1[1] = (v39_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v42_data = r0[1];
              float v45_data = ir1[3];
              ir1[3] = (v45_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v48_data = r0[2];
              float v51_data = ir1[5];
              ir1[5] = (v51_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v54_data = r0[3];
              float v57_data = ir1[7];
              ir1[7] = (v57_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v60_data = r0[4];
              float v63_data = ir1[9];
              ir1[9] = (v63_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v66_data = r0[5];
              float v69_data = ir1[11];
              ir1[11] = (v69_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v74_data = glb_m1[(v21_lead + 40)];
              float v76_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v78_data = ir1[0];
              ir1[0] = (v78_data + (v74_data * v76_bc));
              float v82_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v84_data = ir1[2];
              ir1[2] = (v84_data + (v74_data * v82_bc));
              float v88_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v90_data = ir1[4];
              ir1[4] = (v90_data + (v74_data * v88_bc));
              float v94_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v96_data = ir1[6];
              ir1[6] = (v96_data + (v74_data * v94_bc));
              float v100_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v102_data = ir1[8];
              ir1[8] = (v102_data + (v74_data * v100_bc));
              float v106_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v108_data = ir1[10];
              ir1[10] = (v108_data + (v74_data * v106_bc));
              float v111_data_pre = glb_m1[v22_g ? ((v33_lead + 40)) : (0)];
              float v111_data = v22_g ? (v111_data_pre) : (0.0f);
              float v115_data = ir1[1];
              ir1[1] = (v115_data + (v111_data * v76_bc));
              float v121_data = ir1[3];
              ir1[3] = (v121_data + (v111_data * v82_bc));
              float v127_data = ir1[5];
              ir1[5] = (v127_data + (v111_data * v88_bc));
              float v133_data = ir1[7];
              ir1[7] = (v133_data + (v111_data * v94_bc));
              float v139_data = ir1[9];
              ir1[9] = (v139_data + (v111_data * v100_bc));
              float v145_data = ir1[11];
              ir1[11] = (v145_data + (v111_data * v106_bc));
              float v148_data_pre = glb_m1[v22_g ? ((v33_lead + 80)) : (0)];
              float v148_data = v22_g ? (v148_data_pre) : (0.0f);
              float v152_data = ir1[1];
              ir1[1] = (v152_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v158_data = ir1[3];
              ir1[3] = (v158_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v164_data = ir1[5];
              ir1[5] = (v164_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v170_data = ir1[7];
              ir1[7] = (v170_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v176_data = ir1[9];
              ir1[9] = (v176_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v182_data = ir1[11];
              ir1[11] = (v182_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v185_data = glb_m1[(v21_lead + 120)];
              float v187_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v189_data = ir1[0];
              ir1[0] = (v189_data + (v185_data * v187_bc));
              float v193_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v195_data = ir1[2];
              ir1[2] = (v195_data + (v185_data * v193_bc));
              float v199_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v201_data = ir1[4];
              ir1[4] = (v201_data + (v185_data * v199_bc));
              float v205_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v207_data = ir1[6];
              ir1[6] = (v207_data + (v185_data * v205_bc));
              float v211_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v213_data = ir1[8];
              ir1[8] = (v213_data + (v185_data * v211_bc));
              float v217_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v219_data = ir1[10];
              ir1[10] = (v219_data + (v185_data * v217_bc));
              float v222_data_pre = glb_m1[v22_g ? ((v33_lead + 120)) : (0)];
              float v222_data = v22_g ? (v222_data_pre) : (0.0f);
              float v226_data = ir1[1];
              ir1[1] = (v226_data + (v222_data * v187_bc));
              float v232_data = ir1[3];
              ir1[3] = (v232_data + (v222_data * v193_bc));
              float v238_data = ir1[5];
              ir1[5] = (v238_data + (v222_data * v199_bc));
              float v244_data = ir1[7];
              ir1[7] = (v244_data + (v222_data * v205_bc));
              float v250_data = ir1[9];
              ir1[9] = (v250_data + (v222_data * v211_bc));
              float v256_data = ir1[11];
              ir1[11] = (v256_data + (v222_data * v217_bc));
              float v259_data = glb_m1[(v21_lead + 160)];
              float v261_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v263_data = ir1[0];
              ir1[0] = (v263_data + (v259_data * v261_bc));
              float v267_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v269_data = ir1[2];
              ir1[2] = (v269_data + (v259_data * v267_bc));
              float v273_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v275_data = ir1[4];
              ir1[4] = (v275_data + (v259_data * v273_bc));
              float v279_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v281_data = ir1[6];
              ir1[6] = (v281_data + (v259_data * v279_bc));
              float v285_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v287_data = ir1[8];
              ir1[8] = (v287_data + (v259_data * v285_bc));
              float v291_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v293_data = ir1[10];
              ir1[10] = (v293_data + (v259_data * v291_bc));
              float v296_data_pre = glb_m1[v22_g ? ((v33_lead + 160)) : (0)];
              float v296_data = v22_g ? (v296_data_pre) : (0.0f);
              float v300_data = ir1[1];
              ir1[1] = (v300_data + (v296_data * v261_bc));
              float v306_data = ir1[3];
              ir1[3] = (v306_data + (v296_data * v267_bc));
              float v312_data = ir1[5];
              ir1[5] = (v312_data + (v296_data * v273_bc));
              float v318_data = ir1[7];
              ir1[7] = (v318_data + (v296_data * v279_bc));
              float v324_data = ir1[9];
              ir1[9] = (v324_data + (v296_data * v285_bc));
              float v330_data = ir1[11];
              ir1[11] = (v330_data + (v296_data * v291_bc));
              float v333_data_pre = glb_m1[v22_g ? ((v33_lead + 200)) : (0)];
              float v333_data = v22_g ? (v333_data_pre) : (0.0f);
              float v337_data = ir1[1];
              ir1[1] = (v337_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v343_data = ir1[3];
              ir1[3] = (v343_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v349_data = ir1[5];
              ir1[5] = (v349_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v355_data = ir1[7];
              ir1[7] = (v355_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v361_data = ir1[9];
              ir1[9] = (v361_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v367_data = ir1[11];
              ir1[11] = (v367_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v370_data = glb_m1[(v21_lead + 240)];
              float v372_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v374_data = ir1[0];
              ir1[0] = (v374_data + (v370_data * v372_bc));
              float v378_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v380_data = ir1[2];
              ir1[2] = (v380_data + (v370_data * v378_bc));
              float v384_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v386_data = ir1[4];
              ir1[4] = (v386_data + (v370_data * v384_bc));
              float v390_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v392_data = ir1[6];
              ir1[6] = (v392_data + (v370_data * v390_bc));
              float v396_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v398_data = ir1[8];
              ir1[8] = (v398_data + (v370_data * v396_bc));
              float v402_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v404_data = ir1[10];
              ir1[10] = (v404_data + (v370_data * v402_bc));
              float v407_data_pre = glb_m1[v22_g ? ((v33_lead + 240)) : (0)];
              float v407_data = v22_g ? (v407_data_pre) : (0.0f);
              float v411_data = ir1[1];
              ir1[1] = (v411_data + (v407_data * v372_bc));
              float v417_data = ir1[3];
              ir1[3] = (v417_data + (v407_data * v378_bc));
              float v423_data = ir1[5];
              ir1[5] = (v423_data + (v407_data * v384_bc));
              float v429_data = ir1[7];
              ir1[7] = (v429_data + (v407_data * v390_bc));
              float v435_data = ir1[9];
              ir1[9] = (v435_data + (v407_data * v396_bc));
              float v441_data = ir1[11];
              ir1[11] = (v441_data + (v407_data * v402_bc));
              float v444_data = glb_m1[(v21_lead + 280)];
              float v446_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v448_data = ir1[0];
              ir1[0] = (v448_data + (v444_data * v446_bc));
              float v452_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v454_data = ir1[2];
              ir1[2] = (v454_data + (v444_data * v452_bc));
              float v458_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v460_data = ir1[4];
              ir1[4] = (v460_data + (v444_data * v458_bc));
              float v464_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v466_data = ir1[6];
              ir1[6] = (v466_data + (v444_data * v464_bc));
              float v470_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v472_data = ir1[8];
              ir1[8] = (v472_data + (v444_data * v470_bc));
              float v476_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v478_data = ir1[10];
              ir1[10] = (v478_data + (v444_data * v476_bc));
              float v481_data_pre = glb_m1[v22_g ? ((v33_lead + 280)) : (0)];
              float v481_data = v22_g ? (v481_data_pre) : (0.0f);
              float v485_data = ir1[1];
              ir1[1] = (v485_data + (v481_data * v446_bc));
              float v491_data = ir1[3];
              ir1[3] = (v491_data + (v481_data * v452_bc));
              float v497_data = ir1[5];
              ir1[5] = (v497_data + (v481_data * v458_bc));
              float v503_data = ir1[7];
              ir1[7] = (v503_data + (v481_data * v464_bc));
              float v509_data = ir1[9];
              ir1[9] = (v509_data + (v481_data * v470_bc));
              float v515_data = ir1[11];
              ir1[11] = (v515_data + (v481_data * v476_bc));
              // r1 = ir1
              #pragma unroll
              for (int32_t v517_n0 = 0; v517_n0 < 1; ++v517_n0) {
                #pragma unroll
                for (int32_t v518_n1 = 0; v518_n1 < 6; ++v518_n1) {
                  int32_t v520_a = v517_n0 + (v518_n1 * 2);
                  float v521_data = ir1[v520_a];
                  r1[v520_a] = v521_data;
                }
              }
              if (v22_g) {
                #pragma unroll
                for (int32_t v522_n1 = 0; v522_n1 < 6; ++v522_n1) {
                  int32_t v524_a = 1 + (v522_n1 * 2);
                  float v525_data = ir1[v524_a];
                  r1[v524_a] = v525_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v526_i0 = 0; v526_i0 < 1; ++v526_i0) {
                int32_t v532_lead = v21_lead + (v526_i0 * 32);
                #pragma unroll
                for (int32_t v527_i1 = 0; v527_i1 < 6; ++v527_i1) {
                  float v530_data = r1[(v526_i0 + (v527_i1 * 2))];
                  glb_m0[(v532_lead + (v527_i1 * 40))] = v530_data;
                }
              }
              if (v22_g) {
                #pragma unroll
                for (int32_t v535_i1 = 0; v535_i1 < 6; ++v535_i1) {
                  float v538_data = r1[(1 + (v535_i1 * 2))];
                  glb_m0[(v33_lead + (v535_i1 * 40))] = v538_data;
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


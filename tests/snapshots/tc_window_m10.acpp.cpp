// === base name ===
kernel_61d17d146179dbb6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_61d17d146179dbb6 = {{16, 16, 1}, 16, 10, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_61d17d146179dbb6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_61d17d146179dbb6(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_61d17d146179dbb6(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_61d17d146179dbb6(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_61d17d146179dbb6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_61d17d146179dbb6(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_61d17d146179dbb6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (10 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 16×20(10×17) {0..10}×{1..18} none
        //   m2 20×9(17×9) {1..18}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 153 + 0 + m2_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v17_lead = item.get_local_id(2) % 16;
              if (v17_lead >= 1) {
                int32_t v22_a = v17_lead - 1;
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
                  float v25_data = glb_m2[(v22_a + (v19_i1 * 17))];
                  r0[(v19_i1 * 2)] = v25_data;
                }
              }
              if (v17_lead < 2) {
                int32_t v32_a = (v17_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
                  float v35_data = glb_m2[(v32_a + (v29_i1 * 17))];
                  r0[(1 + (v29_i1 * 2))] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 10), (0, 9)] [(1, 18)]
              float ir1[9]{};
              bool v43_g = v17_lead < 10;
              float v44_data = v43_g ? (glb_m1[v17_lead]) : (0.0f);
              float v45_data = r0[0];
              float v48_data = ir1[0];
              ir1[0] = (v48_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v51_data = r0[2];
              float v54_data = ir1[1];
              ir1[1] = (v54_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v57_data = r0[4];
              float v60_data = ir1[2];
              ir1[2] = (v60_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v63_data = r0[6];
              float v66_data = ir1[3];
              ir1[3] = (v66_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v69_data = r0[8];
              float v72_data = ir1[4];
              ir1[4] = (v72_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v75_data = r0[10];
              float v78_data = ir1[5];
              ir1[5] = (v78_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v81_data = r0[12];
              float v84_data = ir1[6];
              ir1[6] = (v84_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v87_data = r0[14];
              float v90_data = ir1[7];
              ir1[7] = (v90_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v93_data = r0[16];
              float v96_data = ir1[8];
              ir1[8] = (v96_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v99_data = v43_g ? (glb_m1[(v17_lead + 10)]) : (0.0f);
              float v103_data = ir1[0];
              ir1[0] = (v103_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v109_data = ir1[1];
              ir1[1] = (v109_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v115_data = ir1[2];
              ir1[2] = (v115_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v121_data = ir1[3];
              ir1[3] = (v121_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v127_data = ir1[4];
              ir1[4] = (v127_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v133_data = ir1[5];
              ir1[5] = (v133_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v139_data = ir1[6];
              ir1[6] = (v139_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v145_data = ir1[7];
              ir1[7] = (v145_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v151_data = ir1[8];
              ir1[8] = (v151_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v154_data = v43_g ? (glb_m1[(v17_lead + 20)]) : (0.0f);
              float v158_data = ir1[0];
              ir1[0] = (v158_data + (v154_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v164_data = ir1[1];
              ir1[1] = (v164_data + (v154_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v170_data = ir1[2];
              ir1[2] = (v170_data + (v154_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v176_data = ir1[3];
              ir1[3] = (v176_data + (v154_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v182_data = ir1[4];
              ir1[4] = (v182_data + (v154_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v188_data = ir1[5];
              ir1[5] = (v188_data + (v154_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v194_data = ir1[6];
              ir1[6] = (v194_data + (v154_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v200_data = ir1[7];
              ir1[7] = (v200_data + (v154_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v206_data = ir1[8];
              ir1[8] = (v206_data + (v154_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v209_data = v43_g ? (glb_m1[(v17_lead + 30)]) : (0.0f);
              float v213_data = ir1[0];
              ir1[0] = (v213_data + (v209_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v219_data = ir1[1];
              ir1[1] = (v219_data + (v209_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v225_data = ir1[2];
              ir1[2] = (v225_data + (v209_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v231_data = ir1[3];
              ir1[3] = (v231_data + (v209_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v237_data = ir1[4];
              ir1[4] = (v237_data + (v209_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v243_data = ir1[5];
              ir1[5] = (v243_data + (v209_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v249_data = ir1[6];
              ir1[6] = (v249_data + (v209_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v255_data = ir1[7];
              ir1[7] = (v255_data + (v209_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v261_data = ir1[8];
              ir1[8] = (v261_data + (v209_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v264_data = v43_g ? (glb_m1[(v17_lead + 40)]) : (0.0f);
              float v268_data = ir1[0];
              ir1[0] = (v268_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v274_data = ir1[1];
              ir1[1] = (v274_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v280_data = ir1[2];
              ir1[2] = (v280_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v286_data = ir1[3];
              ir1[3] = (v286_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v292_data = ir1[4];
              ir1[4] = (v292_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v298_data = ir1[5];
              ir1[5] = (v298_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v304_data = ir1[6];
              ir1[6] = (v304_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v310_data = ir1[7];
              ir1[7] = (v310_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v316_data = ir1[8];
              ir1[8] = (v316_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v319_data = v43_g ? (glb_m1[(v17_lead + 50)]) : (0.0f);
              float v323_data = ir1[0];
              ir1[0] = (v323_data + (v319_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v329_data = ir1[1];
              ir1[1] = (v329_data + (v319_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v335_data = ir1[2];
              ir1[2] = (v335_data + (v319_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v341_data = ir1[3];
              ir1[3] = (v341_data + (v319_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v347_data = ir1[4];
              ir1[4] = (v347_data + (v319_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v353_data = ir1[5];
              ir1[5] = (v353_data + (v319_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v359_data = ir1[6];
              ir1[6] = (v359_data + (v319_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v365_data = ir1[7];
              ir1[7] = (v365_data + (v319_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v371_data = ir1[8];
              ir1[8] = (v371_data + (v319_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v374_data = v43_g ? (glb_m1[(v17_lead + 60)]) : (0.0f);
              float v378_data = ir1[0];
              ir1[0] = (v378_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v384_data = ir1[1];
              ir1[1] = (v384_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v390_data = ir1[2];
              ir1[2] = (v390_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v396_data = ir1[3];
              ir1[3] = (v396_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v402_data = ir1[4];
              ir1[4] = (v402_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v408_data = ir1[5];
              ir1[5] = (v408_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v414_data = ir1[6];
              ir1[6] = (v414_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v420_data = ir1[7];
              ir1[7] = (v420_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v426_data = ir1[8];
              ir1[8] = (v426_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v429_data = v43_g ? (glb_m1[(v17_lead + 70)]) : (0.0f);
              float v433_data = ir1[0];
              ir1[0] = (v433_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v439_data = ir1[1];
              ir1[1] = (v439_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v445_data = ir1[2];
              ir1[2] = (v445_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v451_data = ir1[3];
              ir1[3] = (v451_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v457_data = ir1[4];
              ir1[4] = (v457_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v463_data = ir1[5];
              ir1[5] = (v463_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v469_data = ir1[6];
              ir1[6] = (v469_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v475_data = ir1[7];
              ir1[7] = (v475_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v481_data = ir1[8];
              ir1[8] = (v481_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v484_data = v43_g ? (glb_m1[(v17_lead + 80)]) : (0.0f);
              float v488_data = ir1[0];
              ir1[0] = (v488_data + (v484_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v494_data = ir1[1];
              ir1[1] = (v494_data + (v484_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v500_data = ir1[2];
              ir1[2] = (v500_data + (v484_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v506_data = ir1[3];
              ir1[3] = (v506_data + (v484_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v512_data = ir1[4];
              ir1[4] = (v512_data + (v484_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v518_data = ir1[5];
              ir1[5] = (v518_data + (v484_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v524_data = ir1[6];
              ir1[6] = (v524_data + (v484_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v530_data = ir1[7];
              ir1[7] = (v530_data + (v484_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v536_data = ir1[8];
              ir1[8] = (v536_data + (v484_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v539_data = v43_g ? (glb_m1[(v17_lead + 90)]) : (0.0f);
              float v543_data = ir1[0];
              ir1[0] = (v543_data + (v539_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v549_data = ir1[1];
              ir1[1] = (v549_data + (v539_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v555_data = ir1[2];
              ir1[2] = (v555_data + (v539_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v561_data = ir1[3];
              ir1[3] = (v561_data + (v539_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v567_data = ir1[4];
              ir1[4] = (v567_data + (v539_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v573_data = ir1[5];
              ir1[5] = (v573_data + (v539_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v579_data = ir1[6];
              ir1[6] = (v579_data + (v539_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v585_data = ir1[7];
              ir1[7] = (v585_data + (v539_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v591_data = ir1[8];
              ir1[8] = (v591_data + (v539_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v594_data = v43_g ? (glb_m1[(v17_lead + 100)]) : (0.0f);
              float v598_data = ir1[0];
              ir1[0] = (v598_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v604_data = ir1[1];
              ir1[1] = (v604_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v610_data = ir1[2];
              ir1[2] = (v610_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v616_data = ir1[3];
              ir1[3] = (v616_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v622_data = ir1[4];
              ir1[4] = (v622_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v628_data = ir1[5];
              ir1[5] = (v628_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v634_data = ir1[6];
              ir1[6] = (v634_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v640_data = ir1[7];
              ir1[7] = (v640_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v646_data = ir1[8];
              ir1[8] = (v646_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v649_data = v43_g ? (glb_m1[(v17_lead + 110)]) : (0.0f);
              float v653_data = ir1[0];
              ir1[0] = (v653_data + (v649_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v659_data = ir1[1];
              ir1[1] = (v659_data + (v649_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v665_data = ir1[2];
              ir1[2] = (v665_data + (v649_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v671_data = ir1[3];
              ir1[3] = (v671_data + (v649_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v677_data = ir1[4];
              ir1[4] = (v677_data + (v649_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v683_data = ir1[5];
              ir1[5] = (v683_data + (v649_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v689_data = ir1[6];
              ir1[6] = (v689_data + (v649_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v695_data = ir1[7];
              ir1[7] = (v695_data + (v649_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v701_data = ir1[8];
              ir1[8] = (v701_data + (v649_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v704_data = v43_g ? (glb_m1[(v17_lead + 120)]) : (0.0f);
              float v708_data = ir1[0];
              ir1[0] = (v708_data + (v704_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v714_data = ir1[1];
              ir1[1] = (v714_data + (v704_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v720_data = ir1[2];
              ir1[2] = (v720_data + (v704_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v726_data = ir1[3];
              ir1[3] = (v726_data + (v704_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v732_data = ir1[4];
              ir1[4] = (v732_data + (v704_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v738_data = ir1[5];
              ir1[5] = (v738_data + (v704_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v744_data = ir1[6];
              ir1[6] = (v744_data + (v704_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v750_data = ir1[7];
              ir1[7] = (v750_data + (v704_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v756_data = ir1[8];
              ir1[8] = (v756_data + (v704_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v759_data = v43_g ? (glb_m1[(v17_lead + 130)]) : (0.0f);
              float v763_data = ir1[0];
              ir1[0] = (v763_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v769_data = ir1[1];
              ir1[1] = (v769_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v775_data = ir1[2];
              ir1[2] = (v775_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v781_data = ir1[3];
              ir1[3] = (v781_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v787_data = ir1[4];
              ir1[4] = (v787_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v793_data = ir1[5];
              ir1[5] = (v793_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v799_data = ir1[6];
              ir1[6] = (v799_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v805_data = ir1[7];
              ir1[7] = (v805_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v811_data = ir1[8];
              ir1[8] = (v811_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v814_data = v43_g ? (glb_m1[(v17_lead + 140)]) : (0.0f);
              float v818_data = ir1[0];
              ir1[0] = (v818_data + (v814_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v824_data = ir1[1];
              ir1[1] = (v824_data + (v814_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v830_data = ir1[2];
              ir1[2] = (v830_data + (v814_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v836_data = ir1[3];
              ir1[3] = (v836_data + (v814_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v842_data = ir1[4];
              ir1[4] = (v842_data + (v814_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v848_data = ir1[5];
              ir1[5] = (v848_data + (v814_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v854_data = ir1[6];
              ir1[6] = (v854_data + (v814_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v860_data = ir1[7];
              ir1[7] = (v860_data + (v814_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v866_data = ir1[8];
              ir1[8] = (v866_data + (v814_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v869_data = v43_g ? (glb_m1[(v17_lead + 150)]) : (0.0f);
              float v870_data = r0[1];
              float v873_data = ir1[0];
              ir1[0] = (v873_data + (v869_data * (sycl::select_from_group(item.get_sub_group(), v870_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v876_data = r0[3];
              float v879_data = ir1[1];
              ir1[1] = (v879_data + (v869_data * (sycl::select_from_group(item.get_sub_group(), v876_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v882_data = r0[5];
              float v885_data = ir1[2];
              ir1[2] = (v885_data + (v869_data * (sycl::select_from_group(item.get_sub_group(), v882_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v888_data = r0[7];
              float v891_data = ir1[3];
              ir1[3] = (v891_data + (v869_data * (sycl::select_from_group(item.get_sub_group(), v888_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v894_data = r0[9];
              float v897_data = ir1[4];
              ir1[4] = (v897_data + (v869_data * (sycl::select_from_group(item.get_sub_group(), v894_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v900_data = r0[11];
              float v903_data = ir1[5];
              ir1[5] = (v903_data + (v869_data * (sycl::select_from_group(item.get_sub_group(), v900_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v906_data = r0[13];
              float v909_data = ir1[6];
              ir1[6] = (v909_data + (v869_data * (sycl::select_from_group(item.get_sub_group(), v906_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v912_data = r0[15];
              float v915_data = ir1[7];
              ir1[7] = (v915_data + (v869_data * (sycl::select_from_group(item.get_sub_group(), v912_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v918_data = r0[17];
              float v921_data = ir1[8];
              ir1[8] = (v921_data + (v869_data * (sycl::select_from_group(item.get_sub_group(), v918_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v924_data = v43_g ? (glb_m1[(v17_lead + 160)]) : (0.0f);
              float v928_data = ir1[0];
              ir1[0] = (v928_data + (v924_data * (sycl::select_from_group(item.get_sub_group(), v870_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v934_data = ir1[1];
              ir1[1] = (v934_data + (v924_data * (sycl::select_from_group(item.get_sub_group(), v876_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v940_data = ir1[2];
              ir1[2] = (v940_data + (v924_data * (sycl::select_from_group(item.get_sub_group(), v882_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v946_data = ir1[3];
              ir1[3] = (v946_data + (v924_data * (sycl::select_from_group(item.get_sub_group(), v888_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v952_data = ir1[4];
              ir1[4] = (v952_data + (v924_data * (sycl::select_from_group(item.get_sub_group(), v894_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v958_data = ir1[5];
              ir1[5] = (v958_data + (v924_data * (sycl::select_from_group(item.get_sub_group(), v900_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v964_data = ir1[6];
              ir1[6] = (v964_data + (v924_data * (sycl::select_from_group(item.get_sub_group(), v906_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v970_data = ir1[7];
              ir1[7] = (v970_data + (v924_data * (sycl::select_from_group(item.get_sub_group(), v912_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v976_data = ir1[8];
              ir1[8] = (v976_data + (v924_data * (sycl::select_from_group(item.get_sub_group(), v918_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              // r1 = ir1
              if (v43_g) {
                #pragma unroll
                for (int32_t v979_n1 = 0; v979_n1 < 9; ++v979_n1) {
                  float v981_data = ir1[v979_n1];
                  r1[v979_n1] = v981_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              if (v43_g) {
                #pragma unroll
                for (int32_t v983_i1 = 0; v983_i1 < 9; ++v983_i1) {
                  float v985_data = r1[v983_i1];
                  glb_m0[(v17_lead + (v983_i1 * 10))] = v985_data;
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


// === base name ===
kernel_acdae7c56bbfe2b0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_acdae7c56bbfe2b0 = {{16, 16, 1}, 16, 10, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_acdae7c56bbfe2b0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_acdae7c56bbfe2b0(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_acdae7c56bbfe2b0(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_acdae7c56bbfe2b0(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_acdae7c56bbfe2b0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_acdae7c56bbfe2b0(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_acdae7c56bbfe2b0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (10 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 16×20(10×17) {0..10}×{1..18} none
        //   m2 20×9(17×9) {1..18}×{0..9} strided
        //   m3 16×20(10×18) {0..10}×{1..19} none
        //   m4 20×9(18×9) {1..19}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[10,19]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[1,0],[19,9]],"name":"m4","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,19]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[19,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          const float *const __restrict__ glb_m1 = &m1[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 162 + 0 + m4_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v23_lead = item.get_local_id(2) % 16;
              bool v24_g = v23_lead >= 1;
              if (v24_g) {
                int32_t v28_a = v23_lead - 1;
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
                  float v31_data = glb_m2[(v28_a + (v25_i1 * 17))];
                  r0[(v25_i1 * 2)] = v31_data;
                }
              }
              if (v23_lead < 2) {
                int32_t v38_a = (v23_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 9; ++v35_i1) {
                  float v41_data = glb_m2[(v38_a + (v35_i1 * 17))];
                  r0[(1 + (v35_i1 * 2))] = v41_data;
                }
              }
              float r2[18]{};
              // r2 = load{g>r}(glb_m4);
              if (v24_g) {
                int32_t v992_a = v23_lead - 1;
                #pragma unroll
                for (int32_t v989_i1 = 0; v989_i1 < 9; ++v989_i1) {
                  float v995_data = glb_m4[(v992_a + (v989_i1 * 18))];
                  r2[(v989_i1 * 2)] = v995_data;
                }
              }
              if (v23_lead < 3) {
                int32_t v1002_a = (v23_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v999_i1 = 0; v999_i1 < 9; ++v999_i1) {
                  float v1005_data = glb_m4[(v1002_a + (v999_i1 * 18))];
                  r2[(1 + (v999_i1 * 2))] = v1005_data;
                }
              }
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 10), (0, 9)] [(1, 18)]
              float ir1[9]{};
              bool v49_g = v23_lead < 10;
              float v50_data_pre = glb_m1[v49_g ? (v23_lead) : (0)];
              float v50_data = v49_g ? (v50_data_pre) : (0.0f);
              float v51_data = r0[0];
              float v54_data = ir1[0];
              ir1[0] = (v54_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v57_data = r0[2];
              float v60_data = ir1[1];
              ir1[1] = (v60_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v63_data = r0[4];
              float v66_data = ir1[2];
              ir1[2] = (v66_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v69_data = r0[6];
              float v72_data = ir1[3];
              ir1[3] = (v72_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v75_data = r0[8];
              float v78_data = ir1[4];
              ir1[4] = (v78_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v81_data = r0[10];
              float v84_data = ir1[5];
              ir1[5] = (v84_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v87_data = r0[12];
              float v90_data = ir1[6];
              ir1[6] = (v90_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v93_data = r0[14];
              float v96_data = ir1[7];
              ir1[7] = (v96_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v99_data = r0[16];
              float v102_data = ir1[8];
              ir1[8] = (v102_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v104_a = v23_lead + 10;
              float v105_data_pre = glb_m1[v49_g ? (v104_a) : (0)];
              float v105_data = v49_g ? (v105_data_pre) : (0.0f);
              float v109_data = ir1[0];
              ir1[0] = (v109_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v115_data = ir1[1];
              ir1[1] = (v115_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v121_data = ir1[2];
              ir1[2] = (v121_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v127_data = ir1[3];
              ir1[3] = (v127_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v133_data = ir1[4];
              ir1[4] = (v133_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v139_data = ir1[5];
              ir1[5] = (v139_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v145_data = ir1[6];
              ir1[6] = (v145_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v151_data = ir1[7];
              ir1[7] = (v151_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v157_data = ir1[8];
              ir1[8] = (v157_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v159_a = v23_lead + 20;
              float v160_data_pre = glb_m1[v49_g ? (v159_a) : (0)];
              float v160_data = v49_g ? (v160_data_pre) : (0.0f);
              float v164_data = ir1[0];
              ir1[0] = (v164_data + (v160_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v170_data = ir1[1];
              ir1[1] = (v170_data + (v160_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v176_data = ir1[2];
              ir1[2] = (v176_data + (v160_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v182_data = ir1[3];
              ir1[3] = (v182_data + (v160_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v188_data = ir1[4];
              ir1[4] = (v188_data + (v160_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v194_data = ir1[5];
              ir1[5] = (v194_data + (v160_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v200_data = ir1[6];
              ir1[6] = (v200_data + (v160_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v206_data = ir1[7];
              ir1[7] = (v206_data + (v160_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v212_data = ir1[8];
              ir1[8] = (v212_data + (v160_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v214_a = v23_lead + 30;
              float v215_data_pre = glb_m1[v49_g ? (v214_a) : (0)];
              float v215_data = v49_g ? (v215_data_pre) : (0.0f);
              float v219_data = ir1[0];
              ir1[0] = (v219_data + (v215_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v225_data = ir1[1];
              ir1[1] = (v225_data + (v215_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v231_data = ir1[2];
              ir1[2] = (v231_data + (v215_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v237_data = ir1[3];
              ir1[3] = (v237_data + (v215_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v243_data = ir1[4];
              ir1[4] = (v243_data + (v215_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v249_data = ir1[5];
              ir1[5] = (v249_data + (v215_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v255_data = ir1[6];
              ir1[6] = (v255_data + (v215_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v261_data = ir1[7];
              ir1[7] = (v261_data + (v215_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v267_data = ir1[8];
              ir1[8] = (v267_data + (v215_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v269_a = v23_lead + 40;
              float v270_data_pre = glb_m1[v49_g ? (v269_a) : (0)];
              float v270_data = v49_g ? (v270_data_pre) : (0.0f);
              float v274_data = ir1[0];
              ir1[0] = (v274_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v280_data = ir1[1];
              ir1[1] = (v280_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v286_data = ir1[2];
              ir1[2] = (v286_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v292_data = ir1[3];
              ir1[3] = (v292_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v298_data = ir1[4];
              ir1[4] = (v298_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v304_data = ir1[5];
              ir1[5] = (v304_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v310_data = ir1[6];
              ir1[6] = (v310_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v316_data = ir1[7];
              ir1[7] = (v316_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v322_data = ir1[8];
              ir1[8] = (v322_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v324_a = v23_lead + 50;
              float v325_data_pre = glb_m1[v49_g ? (v324_a) : (0)];
              float v325_data = v49_g ? (v325_data_pre) : (0.0f);
              float v329_data = ir1[0];
              ir1[0] = (v329_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v335_data = ir1[1];
              ir1[1] = (v335_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v341_data = ir1[2];
              ir1[2] = (v341_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v347_data = ir1[3];
              ir1[3] = (v347_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v353_data = ir1[4];
              ir1[4] = (v353_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v359_data = ir1[5];
              ir1[5] = (v359_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v365_data = ir1[6];
              ir1[6] = (v365_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v371_data = ir1[7];
              ir1[7] = (v371_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v377_data = ir1[8];
              ir1[8] = (v377_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v379_a = v23_lead + 60;
              float v380_data_pre = glb_m1[v49_g ? (v379_a) : (0)];
              float v380_data = v49_g ? (v380_data_pre) : (0.0f);
              float v384_data = ir1[0];
              ir1[0] = (v384_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v390_data = ir1[1];
              ir1[1] = (v390_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v396_data = ir1[2];
              ir1[2] = (v396_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v402_data = ir1[3];
              ir1[3] = (v402_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v408_data = ir1[4];
              ir1[4] = (v408_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v414_data = ir1[5];
              ir1[5] = (v414_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v420_data = ir1[6];
              ir1[6] = (v420_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v426_data = ir1[7];
              ir1[7] = (v426_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v432_data = ir1[8];
              ir1[8] = (v432_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v434_a = v23_lead + 70;
              float v435_data_pre = glb_m1[v49_g ? (v434_a) : (0)];
              float v435_data = v49_g ? (v435_data_pre) : (0.0f);
              float v439_data = ir1[0];
              ir1[0] = (v439_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v445_data = ir1[1];
              ir1[1] = (v445_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v451_data = ir1[2];
              ir1[2] = (v451_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v457_data = ir1[3];
              ir1[3] = (v457_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v463_data = ir1[4];
              ir1[4] = (v463_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v469_data = ir1[5];
              ir1[5] = (v469_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v475_data = ir1[6];
              ir1[6] = (v475_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v481_data = ir1[7];
              ir1[7] = (v481_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v487_data = ir1[8];
              ir1[8] = (v487_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v489_a = v23_lead + 80;
              float v490_data_pre = glb_m1[v49_g ? (v489_a) : (0)];
              float v490_data = v49_g ? (v490_data_pre) : (0.0f);
              float v494_data = ir1[0];
              ir1[0] = (v494_data + (v490_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v500_data = ir1[1];
              ir1[1] = (v500_data + (v490_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v506_data = ir1[2];
              ir1[2] = (v506_data + (v490_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v512_data = ir1[3];
              ir1[3] = (v512_data + (v490_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v518_data = ir1[4];
              ir1[4] = (v518_data + (v490_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v524_data = ir1[5];
              ir1[5] = (v524_data + (v490_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v530_data = ir1[6];
              ir1[6] = (v530_data + (v490_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v536_data = ir1[7];
              ir1[7] = (v536_data + (v490_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v542_data = ir1[8];
              ir1[8] = (v542_data + (v490_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v544_a = v23_lead + 90;
              float v545_data_pre = glb_m1[v49_g ? (v544_a) : (0)];
              float v545_data = v49_g ? (v545_data_pre) : (0.0f);
              float v549_data = ir1[0];
              ir1[0] = (v549_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v555_data = ir1[1];
              ir1[1] = (v555_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v561_data = ir1[2];
              ir1[2] = (v561_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v567_data = ir1[3];
              ir1[3] = (v567_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v573_data = ir1[4];
              ir1[4] = (v573_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v579_data = ir1[5];
              ir1[5] = (v579_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v585_data = ir1[6];
              ir1[6] = (v585_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v591_data = ir1[7];
              ir1[7] = (v591_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v597_data = ir1[8];
              ir1[8] = (v597_data + (v545_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v599_a = v23_lead + 100;
              float v600_data_pre = glb_m1[v49_g ? (v599_a) : (0)];
              float v600_data = v49_g ? (v600_data_pre) : (0.0f);
              float v604_data = ir1[0];
              ir1[0] = (v604_data + (v600_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v610_data = ir1[1];
              ir1[1] = (v610_data + (v600_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v616_data = ir1[2];
              ir1[2] = (v616_data + (v600_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v622_data = ir1[3];
              ir1[3] = (v622_data + (v600_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v628_data = ir1[4];
              ir1[4] = (v628_data + (v600_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v634_data = ir1[5];
              ir1[5] = (v634_data + (v600_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v640_data = ir1[6];
              ir1[6] = (v640_data + (v600_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v646_data = ir1[7];
              ir1[7] = (v646_data + (v600_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v652_data = ir1[8];
              ir1[8] = (v652_data + (v600_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v654_a = v23_lead + 110;
              float v655_data_pre = glb_m1[v49_g ? (v654_a) : (0)];
              float v655_data = v49_g ? (v655_data_pre) : (0.0f);
              float v659_data = ir1[0];
              ir1[0] = (v659_data + (v655_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v665_data = ir1[1];
              ir1[1] = (v665_data + (v655_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v671_data = ir1[2];
              ir1[2] = (v671_data + (v655_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v677_data = ir1[3];
              ir1[3] = (v677_data + (v655_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v683_data = ir1[4];
              ir1[4] = (v683_data + (v655_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v689_data = ir1[5];
              ir1[5] = (v689_data + (v655_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v695_data = ir1[6];
              ir1[6] = (v695_data + (v655_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v701_data = ir1[7];
              ir1[7] = (v701_data + (v655_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v707_data = ir1[8];
              ir1[8] = (v707_data + (v655_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v709_a = v23_lead + 120;
              float v710_data_pre = glb_m1[v49_g ? (v709_a) : (0)];
              float v710_data = v49_g ? (v710_data_pre) : (0.0f);
              float v714_data = ir1[0];
              ir1[0] = (v714_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v720_data = ir1[1];
              ir1[1] = (v720_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v726_data = ir1[2];
              ir1[2] = (v726_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v732_data = ir1[3];
              ir1[3] = (v732_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v738_data = ir1[4];
              ir1[4] = (v738_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v744_data = ir1[5];
              ir1[5] = (v744_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v750_data = ir1[6];
              ir1[6] = (v750_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v756_data = ir1[7];
              ir1[7] = (v756_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v762_data = ir1[8];
              ir1[8] = (v762_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v764_a = v23_lead + 130;
              float v765_data_pre = glb_m1[v49_g ? (v764_a) : (0)];
              float v765_data = v49_g ? (v765_data_pre) : (0.0f);
              float v769_data = ir1[0];
              ir1[0] = (v769_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v775_data = ir1[1];
              ir1[1] = (v775_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v781_data = ir1[2];
              ir1[2] = (v781_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v787_data = ir1[3];
              ir1[3] = (v787_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v793_data = ir1[4];
              ir1[4] = (v793_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v799_data = ir1[5];
              ir1[5] = (v799_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v805_data = ir1[6];
              ir1[6] = (v805_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v811_data = ir1[7];
              ir1[7] = (v811_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v817_data = ir1[8];
              ir1[8] = (v817_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v819_a = v23_lead + 140;
              float v820_data_pre = glb_m1[v49_g ? (v819_a) : (0)];
              float v820_data = v49_g ? (v820_data_pre) : (0.0f);
              float v824_data = ir1[0];
              ir1[0] = (v824_data + (v820_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v830_data = ir1[1];
              ir1[1] = (v830_data + (v820_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v836_data = ir1[2];
              ir1[2] = (v836_data + (v820_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v842_data = ir1[3];
              ir1[3] = (v842_data + (v820_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v848_data = ir1[4];
              ir1[4] = (v848_data + (v820_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v854_data = ir1[5];
              ir1[5] = (v854_data + (v820_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v860_data = ir1[6];
              ir1[6] = (v860_data + (v820_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v866_data = ir1[7];
              ir1[7] = (v866_data + (v820_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v872_data = ir1[8];
              ir1[8] = (v872_data + (v820_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              int32_t v874_a = v23_lead + 150;
              float v875_data_pre = glb_m1[v49_g ? (v874_a) : (0)];
              float v875_data = v49_g ? (v875_data_pre) : (0.0f);
              float v876_data = r0[1];
              float v879_data = ir1[0];
              ir1[0] = (v879_data + (v875_data * (sycl::select_from_group(item.get_sub_group(), v876_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v882_data = r0[3];
              float v885_data = ir1[1];
              ir1[1] = (v885_data + (v875_data * (sycl::select_from_group(item.get_sub_group(), v882_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v888_data = r0[5];
              float v891_data = ir1[2];
              ir1[2] = (v891_data + (v875_data * (sycl::select_from_group(item.get_sub_group(), v888_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v894_data = r0[7];
              float v897_data = ir1[3];
              ir1[3] = (v897_data + (v875_data * (sycl::select_from_group(item.get_sub_group(), v894_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v900_data = r0[9];
              float v903_data = ir1[4];
              ir1[4] = (v903_data + (v875_data * (sycl::select_from_group(item.get_sub_group(), v900_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v906_data = r0[11];
              float v909_data = ir1[5];
              ir1[5] = (v909_data + (v875_data * (sycl::select_from_group(item.get_sub_group(), v906_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v912_data = r0[13];
              float v915_data = ir1[6];
              ir1[6] = (v915_data + (v875_data * (sycl::select_from_group(item.get_sub_group(), v912_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v918_data = r0[15];
              float v921_data = ir1[7];
              ir1[7] = (v921_data + (v875_data * (sycl::select_from_group(item.get_sub_group(), v918_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v924_data = r0[17];
              float v927_data = ir1[8];
              ir1[8] = (v927_data + (v875_data * (sycl::select_from_group(item.get_sub_group(), v924_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              int32_t v929_a = v23_lead + 160;
              float v930_data_pre = glb_m1[v49_g ? (v929_a) : (0)];
              float v930_data = v49_g ? (v930_data_pre) : (0.0f);
              float v934_data = ir1[0];
              ir1[0] = (v934_data + (v930_data * (sycl::select_from_group(item.get_sub_group(), v876_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v940_data = ir1[1];
              ir1[1] = (v940_data + (v930_data * (sycl::select_from_group(item.get_sub_group(), v882_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v946_data = ir1[2];
              ir1[2] = (v946_data + (v930_data * (sycl::select_from_group(item.get_sub_group(), v888_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v952_data = ir1[3];
              ir1[3] = (v952_data + (v930_data * (sycl::select_from_group(item.get_sub_group(), v894_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v958_data = ir1[4];
              ir1[4] = (v958_data + (v930_data * (sycl::select_from_group(item.get_sub_group(), v900_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v964_data = ir1[5];
              ir1[5] = (v964_data + (v930_data * (sycl::select_from_group(item.get_sub_group(), v906_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v970_data = ir1[6];
              ir1[6] = (v970_data + (v930_data * (sycl::select_from_group(item.get_sub_group(), v912_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v976_data = ir1[7];
              ir1[7] = (v976_data + (v930_data * (sycl::select_from_group(item.get_sub_group(), v918_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v982_data = ir1[8];
              ir1[8] = (v982_data + (v930_data * (sycl::select_from_group(item.get_sub_group(), v924_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              // r1 = ir1
              if (v49_g) {
                #pragma unroll
                for (int32_t v985_n1 = 0; v985_n1 < 9; ++v985_n1) {
                  float v987_data = ir1[v985_n1];
                  r1[v985_n1] = v987_data;
                }
              }
              float r3[9]{};
              // ir3 = +(glb_m3 * r2)
              // [(0, 10), (0, 9)] [(1, 19)]
              float ir3[9]{};
              float v1014_data_pre = glb_m3[v49_g ? (v23_lead) : (0)];
              float v1014_data = v49_g ? (v1014_data_pre) : (0.0f);
              float v1015_data = r2[0];
              float v1018_data = ir3[0];
              ir3[0] = (v1018_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1021_data = r2[2];
              float v1024_data = ir3[1];
              ir3[1] = (v1024_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1027_data = r2[4];
              float v1030_data = ir3[2];
              ir3[2] = (v1030_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1033_data = r2[6];
              float v1036_data = ir3[3];
              ir3[3] = (v1036_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1039_data = r2[8];
              float v1042_data = ir3[4];
              ir3[4] = (v1042_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1045_data = r2[10];
              float v1048_data = ir3[5];
              ir3[5] = (v1048_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1051_data = r2[12];
              float v1054_data = ir3[6];
              ir3[6] = (v1054_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1057_data = r2[14];
              float v1060_data = ir3[7];
              ir3[7] = (v1060_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1063_data = r2[16];
              float v1066_data = ir3[8];
              ir3[8] = (v1066_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1069_data_pre = glb_m3[v49_g ? (v104_a) : (0)];
              float v1069_data = v49_g ? (v1069_data_pre) : (0.0f);
              float v1073_data = ir3[0];
              ir3[0] = (v1073_data + (v1069_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1079_data = ir3[1];
              ir3[1] = (v1079_data + (v1069_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1085_data = ir3[2];
              ir3[2] = (v1085_data + (v1069_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1091_data = ir3[3];
              ir3[3] = (v1091_data + (v1069_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1097_data = ir3[4];
              ir3[4] = (v1097_data + (v1069_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1103_data = ir3[5];
              ir3[5] = (v1103_data + (v1069_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1109_data = ir3[6];
              ir3[6] = (v1109_data + (v1069_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1115_data = ir3[7];
              ir3[7] = (v1115_data + (v1069_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1121_data = ir3[8];
              ir3[8] = (v1121_data + (v1069_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1124_data_pre = glb_m3[v49_g ? (v159_a) : (0)];
              float v1124_data = v49_g ? (v1124_data_pre) : (0.0f);
              float v1128_data = ir3[0];
              ir3[0] = (v1128_data + (v1124_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1134_data = ir3[1];
              ir3[1] = (v1134_data + (v1124_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1140_data = ir3[2];
              ir3[2] = (v1140_data + (v1124_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1146_data = ir3[3];
              ir3[3] = (v1146_data + (v1124_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1152_data = ir3[4];
              ir3[4] = (v1152_data + (v1124_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1158_data = ir3[5];
              ir3[5] = (v1158_data + (v1124_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1164_data = ir3[6];
              ir3[6] = (v1164_data + (v1124_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1170_data = ir3[7];
              ir3[7] = (v1170_data + (v1124_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1176_data = ir3[8];
              ir3[8] = (v1176_data + (v1124_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1179_data_pre = glb_m3[v49_g ? (v214_a) : (0)];
              float v1179_data = v49_g ? (v1179_data_pre) : (0.0f);
              float v1183_data = ir3[0];
              ir3[0] = (v1183_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1189_data = ir3[1];
              ir3[1] = (v1189_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1195_data = ir3[2];
              ir3[2] = (v1195_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1201_data = ir3[3];
              ir3[3] = (v1201_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1207_data = ir3[4];
              ir3[4] = (v1207_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1213_data = ir3[5];
              ir3[5] = (v1213_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1219_data = ir3[6];
              ir3[6] = (v1219_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1225_data = ir3[7];
              ir3[7] = (v1225_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1231_data = ir3[8];
              ir3[8] = (v1231_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1234_data_pre = glb_m3[v49_g ? (v269_a) : (0)];
              float v1234_data = v49_g ? (v1234_data_pre) : (0.0f);
              float v1238_data = ir3[0];
              ir3[0] = (v1238_data + (v1234_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1244_data = ir3[1];
              ir3[1] = (v1244_data + (v1234_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1250_data = ir3[2];
              ir3[2] = (v1250_data + (v1234_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1256_data = ir3[3];
              ir3[3] = (v1256_data + (v1234_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1262_data = ir3[4];
              ir3[4] = (v1262_data + (v1234_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1268_data = ir3[5];
              ir3[5] = (v1268_data + (v1234_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1274_data = ir3[6];
              ir3[6] = (v1274_data + (v1234_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1280_data = ir3[7];
              ir3[7] = (v1280_data + (v1234_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1286_data = ir3[8];
              ir3[8] = (v1286_data + (v1234_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1289_data_pre = glb_m3[v49_g ? (v324_a) : (0)];
              float v1289_data = v49_g ? (v1289_data_pre) : (0.0f);
              float v1293_data = ir3[0];
              ir3[0] = (v1293_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1299_data = ir3[1];
              ir3[1] = (v1299_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1305_data = ir3[2];
              ir3[2] = (v1305_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1311_data = ir3[3];
              ir3[3] = (v1311_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1317_data = ir3[4];
              ir3[4] = (v1317_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1323_data = ir3[5];
              ir3[5] = (v1323_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1329_data = ir3[6];
              ir3[6] = (v1329_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1335_data = ir3[7];
              ir3[7] = (v1335_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1341_data = ir3[8];
              ir3[8] = (v1341_data + (v1289_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1344_data_pre = glb_m3[v49_g ? (v379_a) : (0)];
              float v1344_data = v49_g ? (v1344_data_pre) : (0.0f);
              float v1348_data = ir3[0];
              ir3[0] = (v1348_data + (v1344_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1354_data = ir3[1];
              ir3[1] = (v1354_data + (v1344_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1360_data = ir3[2];
              ir3[2] = (v1360_data + (v1344_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1366_data = ir3[3];
              ir3[3] = (v1366_data + (v1344_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1372_data = ir3[4];
              ir3[4] = (v1372_data + (v1344_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1378_data = ir3[5];
              ir3[5] = (v1378_data + (v1344_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1384_data = ir3[6];
              ir3[6] = (v1384_data + (v1344_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1390_data = ir3[7];
              ir3[7] = (v1390_data + (v1344_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1396_data = ir3[8];
              ir3[8] = (v1396_data + (v1344_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1399_data_pre = glb_m3[v49_g ? (v434_a) : (0)];
              float v1399_data = v49_g ? (v1399_data_pre) : (0.0f);
              float v1403_data = ir3[0];
              ir3[0] = (v1403_data + (v1399_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1409_data = ir3[1];
              ir3[1] = (v1409_data + (v1399_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1415_data = ir3[2];
              ir3[2] = (v1415_data + (v1399_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1421_data = ir3[3];
              ir3[3] = (v1421_data + (v1399_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1427_data = ir3[4];
              ir3[4] = (v1427_data + (v1399_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1433_data = ir3[5];
              ir3[5] = (v1433_data + (v1399_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1439_data = ir3[6];
              ir3[6] = (v1439_data + (v1399_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1445_data = ir3[7];
              ir3[7] = (v1445_data + (v1399_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1451_data = ir3[8];
              ir3[8] = (v1451_data + (v1399_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1454_data_pre = glb_m3[v49_g ? (v489_a) : (0)];
              float v1454_data = v49_g ? (v1454_data_pre) : (0.0f);
              float v1458_data = ir3[0];
              ir3[0] = (v1458_data + (v1454_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1464_data = ir3[1];
              ir3[1] = (v1464_data + (v1454_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1470_data = ir3[2];
              ir3[2] = (v1470_data + (v1454_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1476_data = ir3[3];
              ir3[3] = (v1476_data + (v1454_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1482_data = ir3[4];
              ir3[4] = (v1482_data + (v1454_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1488_data = ir3[5];
              ir3[5] = (v1488_data + (v1454_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1494_data = ir3[6];
              ir3[6] = (v1494_data + (v1454_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1500_data = ir3[7];
              ir3[7] = (v1500_data + (v1454_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1506_data = ir3[8];
              ir3[8] = (v1506_data + (v1454_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1509_data_pre = glb_m3[v49_g ? (v544_a) : (0)];
              float v1509_data = v49_g ? (v1509_data_pre) : (0.0f);
              float v1513_data = ir3[0];
              ir3[0] = (v1513_data + (v1509_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1519_data = ir3[1];
              ir3[1] = (v1519_data + (v1509_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1525_data = ir3[2];
              ir3[2] = (v1525_data + (v1509_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1531_data = ir3[3];
              ir3[3] = (v1531_data + (v1509_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1537_data = ir3[4];
              ir3[4] = (v1537_data + (v1509_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1543_data = ir3[5];
              ir3[5] = (v1543_data + (v1509_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1549_data = ir3[6];
              ir3[6] = (v1549_data + (v1509_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1555_data = ir3[7];
              ir3[7] = (v1555_data + (v1509_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1561_data = ir3[8];
              ir3[8] = (v1561_data + (v1509_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1564_data_pre = glb_m3[v49_g ? (v599_a) : (0)];
              float v1564_data = v49_g ? (v1564_data_pre) : (0.0f);
              float v1568_data = ir3[0];
              ir3[0] = (v1568_data + (v1564_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1574_data = ir3[1];
              ir3[1] = (v1574_data + (v1564_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1580_data = ir3[2];
              ir3[2] = (v1580_data + (v1564_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1586_data = ir3[3];
              ir3[3] = (v1586_data + (v1564_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1592_data = ir3[4];
              ir3[4] = (v1592_data + (v1564_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1598_data = ir3[5];
              ir3[5] = (v1598_data + (v1564_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1604_data = ir3[6];
              ir3[6] = (v1604_data + (v1564_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1610_data = ir3[7];
              ir3[7] = (v1610_data + (v1564_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1616_data = ir3[8];
              ir3[8] = (v1616_data + (v1564_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1619_data_pre = glb_m3[v49_g ? (v654_a) : (0)];
              float v1619_data = v49_g ? (v1619_data_pre) : (0.0f);
              float v1623_data = ir3[0];
              ir3[0] = (v1623_data + (v1619_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1629_data = ir3[1];
              ir3[1] = (v1629_data + (v1619_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1635_data = ir3[2];
              ir3[2] = (v1635_data + (v1619_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1641_data = ir3[3];
              ir3[3] = (v1641_data + (v1619_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1647_data = ir3[4];
              ir3[4] = (v1647_data + (v1619_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1653_data = ir3[5];
              ir3[5] = (v1653_data + (v1619_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1659_data = ir3[6];
              ir3[6] = (v1659_data + (v1619_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1665_data = ir3[7];
              ir3[7] = (v1665_data + (v1619_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1671_data = ir3[8];
              ir3[8] = (v1671_data + (v1619_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1674_data_pre = glb_m3[v49_g ? (v709_a) : (0)];
              float v1674_data = v49_g ? (v1674_data_pre) : (0.0f);
              float v1678_data = ir3[0];
              ir3[0] = (v1678_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1684_data = ir3[1];
              ir3[1] = (v1684_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1690_data = ir3[2];
              ir3[2] = (v1690_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1696_data = ir3[3];
              ir3[3] = (v1696_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1702_data = ir3[4];
              ir3[4] = (v1702_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1708_data = ir3[5];
              ir3[5] = (v1708_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1714_data = ir3[6];
              ir3[6] = (v1714_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1720_data = ir3[7];
              ir3[7] = (v1720_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1726_data = ir3[8];
              ir3[8] = (v1726_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1729_data_pre = glb_m3[v49_g ? (v764_a) : (0)];
              float v1729_data = v49_g ? (v1729_data_pre) : (0.0f);
              float v1733_data = ir3[0];
              ir3[0] = (v1733_data + (v1729_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1739_data = ir3[1];
              ir3[1] = (v1739_data + (v1729_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1745_data = ir3[2];
              ir3[2] = (v1745_data + (v1729_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1751_data = ir3[3];
              ir3[3] = (v1751_data + (v1729_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1757_data = ir3[4];
              ir3[4] = (v1757_data + (v1729_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1763_data = ir3[5];
              ir3[5] = (v1763_data + (v1729_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1769_data = ir3[6];
              ir3[6] = (v1769_data + (v1729_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1775_data = ir3[7];
              ir3[7] = (v1775_data + (v1729_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1781_data = ir3[8];
              ir3[8] = (v1781_data + (v1729_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1784_data_pre = glb_m3[v49_g ? (v819_a) : (0)];
              float v1784_data = v49_g ? (v1784_data_pre) : (0.0f);
              float v1788_data = ir3[0];
              ir3[0] = (v1788_data + (v1784_data * (sycl::select_from_group(item.get_sub_group(), v1015_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1794_data = ir3[1];
              ir3[1] = (v1794_data + (v1784_data * (sycl::select_from_group(item.get_sub_group(), v1021_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1800_data = ir3[2];
              ir3[2] = (v1800_data + (v1784_data * (sycl::select_from_group(item.get_sub_group(), v1027_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1806_data = ir3[3];
              ir3[3] = (v1806_data + (v1784_data * (sycl::select_from_group(item.get_sub_group(), v1033_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1812_data = ir3[4];
              ir3[4] = (v1812_data + (v1784_data * (sycl::select_from_group(item.get_sub_group(), v1039_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1818_data = ir3[5];
              ir3[5] = (v1818_data + (v1784_data * (sycl::select_from_group(item.get_sub_group(), v1045_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1824_data = ir3[6];
              ir3[6] = (v1824_data + (v1784_data * (sycl::select_from_group(item.get_sub_group(), v1051_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1830_data = ir3[7];
              ir3[7] = (v1830_data + (v1784_data * (sycl::select_from_group(item.get_sub_group(), v1057_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1836_data = ir3[8];
              ir3[8] = (v1836_data + (v1784_data * (sycl::select_from_group(item.get_sub_group(), v1063_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1839_data_pre = glb_m3[v49_g ? (v874_a) : (0)];
              float v1839_data = v49_g ? (v1839_data_pre) : (0.0f);
              float v1840_data = r2[1];
              float v1843_data = ir3[0];
              ir3[0] = (v1843_data + (v1839_data * (sycl::select_from_group(item.get_sub_group(), v1840_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1846_data = r2[3];
              float v1849_data = ir3[1];
              ir3[1] = (v1849_data + (v1839_data * (sycl::select_from_group(item.get_sub_group(), v1846_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1852_data = r2[5];
              float v1855_data = ir3[2];
              ir3[2] = (v1855_data + (v1839_data * (sycl::select_from_group(item.get_sub_group(), v1852_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1858_data = r2[7];
              float v1861_data = ir3[3];
              ir3[3] = (v1861_data + (v1839_data * (sycl::select_from_group(item.get_sub_group(), v1858_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1864_data = r2[9];
              float v1867_data = ir3[4];
              ir3[4] = (v1867_data + (v1839_data * (sycl::select_from_group(item.get_sub_group(), v1864_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1870_data = r2[11];
              float v1873_data = ir3[5];
              ir3[5] = (v1873_data + (v1839_data * (sycl::select_from_group(item.get_sub_group(), v1870_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1876_data = r2[13];
              float v1879_data = ir3[6];
              ir3[6] = (v1879_data + (v1839_data * (sycl::select_from_group(item.get_sub_group(), v1876_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1882_data = r2[15];
              float v1885_data = ir3[7];
              ir3[7] = (v1885_data + (v1839_data * (sycl::select_from_group(item.get_sub_group(), v1882_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1888_data = r2[17];
              float v1891_data = ir3[8];
              ir3[8] = (v1891_data + (v1839_data * (sycl::select_from_group(item.get_sub_group(), v1888_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1894_data_pre = glb_m3[v49_g ? (v929_a) : (0)];
              float v1894_data = v49_g ? (v1894_data_pre) : (0.0f);
              float v1898_data = ir3[0];
              ir3[0] = (v1898_data + (v1894_data * (sycl::select_from_group(item.get_sub_group(), v1840_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1904_data = ir3[1];
              ir3[1] = (v1904_data + (v1894_data * (sycl::select_from_group(item.get_sub_group(), v1846_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1910_data = ir3[2];
              ir3[2] = (v1910_data + (v1894_data * (sycl::select_from_group(item.get_sub_group(), v1852_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1916_data = ir3[3];
              ir3[3] = (v1916_data + (v1894_data * (sycl::select_from_group(item.get_sub_group(), v1858_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1922_data = ir3[4];
              ir3[4] = (v1922_data + (v1894_data * (sycl::select_from_group(item.get_sub_group(), v1864_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1928_data = ir3[5];
              ir3[5] = (v1928_data + (v1894_data * (sycl::select_from_group(item.get_sub_group(), v1870_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1934_data = ir3[6];
              ir3[6] = (v1934_data + (v1894_data * (sycl::select_from_group(item.get_sub_group(), v1876_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1940_data = ir3[7];
              ir3[7] = (v1940_data + (v1894_data * (sycl::select_from_group(item.get_sub_group(), v1882_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1946_data = ir3[8];
              ir3[8] = (v1946_data + (v1894_data * (sycl::select_from_group(item.get_sub_group(), v1888_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1949_data_pre = glb_m3[v49_g ? ((v23_lead + 170)) : (0)];
              float v1949_data = v49_g ? (v1949_data_pre) : (0.0f);
              float v1953_data = ir3[0];
              ir3[0] = (v1953_data + (v1949_data * (sycl::select_from_group(item.get_sub_group(), v1840_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1959_data = ir3[1];
              ir3[1] = (v1959_data + (v1949_data * (sycl::select_from_group(item.get_sub_group(), v1846_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1965_data = ir3[2];
              ir3[2] = (v1965_data + (v1949_data * (sycl::select_from_group(item.get_sub_group(), v1852_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1971_data = ir3[3];
              ir3[3] = (v1971_data + (v1949_data * (sycl::select_from_group(item.get_sub_group(), v1858_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1977_data = ir3[4];
              ir3[4] = (v1977_data + (v1949_data * (sycl::select_from_group(item.get_sub_group(), v1864_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1983_data = ir3[5];
              ir3[5] = (v1983_data + (v1949_data * (sycl::select_from_group(item.get_sub_group(), v1870_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1989_data = ir3[6];
              ir3[6] = (v1989_data + (v1949_data * (sycl::select_from_group(item.get_sub_group(), v1876_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1995_data = ir3[7];
              ir3[7] = (v1995_data + (v1949_data * (sycl::select_from_group(item.get_sub_group(), v1882_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v2001_data = ir3[8];
              ir3[8] = (v2001_data + (v1949_data * (sycl::select_from_group(item.get_sub_group(), v1888_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              // r3 = ir3 + r1
              if (v49_g) {
                #pragma unroll
                for (int32_t v2004_n1 = 0; v2004_n1 < 9; ++v2004_n1) {
                  float v2006_data = ir3[v2004_n1];
                  float v2007_data = r1[v2004_n1];
                  r3[v2004_n1] = (v2007_data + v2006_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v49_g) {
                #pragma unroll
                for (int32_t v2010_i1 = 0; v2010_i1 < 9; ++v2010_i1) {
                  float v2012_data = r3[v2010_i1];
                  glb_m0[(v23_lead + (v2010_i1 * 10))] = v2012_data;
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


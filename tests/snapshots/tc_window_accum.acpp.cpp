// === base name ===
kernel_d5e759b27e9f97fc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d5e759b27e9f97fc = {{16, 16, 1}, 16, 10, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d5e759b27e9f97fc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d5e759b27e9f97fc(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d5e759b27e9f97fc(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_d5e759b27e9f97fc(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d5e759b27e9f97fc(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d5e759b27e9f97fc(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d5e759b27e9f97fc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
                int32_t v48_a = v23_lead - 1;
                #pragma unroll
                for (int32_t v45_i1 = 0; v45_i1 < 9; ++v45_i1) {
                  float v51_data = glb_m4[(v48_a + (v45_i1 * 18))];
                  r2[(v45_i1 * 2)] = v51_data;
                }
              }
              if (v23_lead < 3) {
                int32_t v58_a = (v23_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v55_i1 = 0; v55_i1 < 9; ++v55_i1) {
                  float v61_data = glb_m4[(v58_a + (v55_i1 * 18))];
                  r2[(1 + (v55_i1 * 2))] = v61_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 10), (0, 9)] [(1, 18)]
              float ir1[9]{};
              bool v69_g = v23_lead < 10;
              float v70_data_pre = glb_m1[v69_g ? (v23_lead) : (0)];
              float v70_data = v69_g ? (v70_data_pre) : (0.0f);
              float v71_data = r0[0];
              float v74_data = ir1[0];
              ir1[0] = (v74_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v77_data = r0[2];
              float v80_data = ir1[1];
              ir1[1] = (v80_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v83_data = r0[4];
              float v86_data = ir1[2];
              ir1[2] = (v86_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v89_data = r0[6];
              float v92_data = ir1[3];
              ir1[3] = (v92_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v95_data = r0[8];
              float v98_data = ir1[4];
              ir1[4] = (v98_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v101_data = r0[10];
              float v104_data = ir1[5];
              ir1[5] = (v104_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v107_data = r0[12];
              float v110_data = ir1[6];
              ir1[6] = (v110_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v113_data = r0[14];
              float v116_data = ir1[7];
              ir1[7] = (v116_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v119_data = r0[16];
              float v122_data = ir1[8];
              ir1[8] = (v122_data + (v70_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v124_a = v23_lead + 10;
              float v125_data_pre = glb_m1[v69_g ? (v124_a) : (0)];
              float v125_data = v69_g ? (v125_data_pre) : (0.0f);
              float v129_data = ir1[0];
              ir1[0] = (v129_data + (v125_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v135_data = ir1[1];
              ir1[1] = (v135_data + (v125_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v141_data = ir1[2];
              ir1[2] = (v141_data + (v125_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v147_data = ir1[3];
              ir1[3] = (v147_data + (v125_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v153_data = ir1[4];
              ir1[4] = (v153_data + (v125_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v159_data = ir1[5];
              ir1[5] = (v159_data + (v125_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v165_data = ir1[6];
              ir1[6] = (v165_data + (v125_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v171_data = ir1[7];
              ir1[7] = (v171_data + (v125_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v177_data = ir1[8];
              ir1[8] = (v177_data + (v125_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v179_a = v23_lead + 20;
              float v180_data_pre = glb_m1[v69_g ? (v179_a) : (0)];
              float v180_data = v69_g ? (v180_data_pre) : (0.0f);
              float v184_data = ir1[0];
              ir1[0] = (v184_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v190_data = ir1[1];
              ir1[1] = (v190_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v196_data = ir1[2];
              ir1[2] = (v196_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v202_data = ir1[3];
              ir1[3] = (v202_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v208_data = ir1[4];
              ir1[4] = (v208_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v214_data = ir1[5];
              ir1[5] = (v214_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v220_data = ir1[6];
              ir1[6] = (v220_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v226_data = ir1[7];
              ir1[7] = (v226_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v232_data = ir1[8];
              ir1[8] = (v232_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v234_a = v23_lead + 30;
              float v235_data_pre = glb_m1[v69_g ? (v234_a) : (0)];
              float v235_data = v69_g ? (v235_data_pre) : (0.0f);
              float v239_data = ir1[0];
              ir1[0] = (v239_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v245_data = ir1[1];
              ir1[1] = (v245_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v251_data = ir1[2];
              ir1[2] = (v251_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v257_data = ir1[3];
              ir1[3] = (v257_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v263_data = ir1[4];
              ir1[4] = (v263_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v269_data = ir1[5];
              ir1[5] = (v269_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v275_data = ir1[6];
              ir1[6] = (v275_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v281_data = ir1[7];
              ir1[7] = (v281_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v287_data = ir1[8];
              ir1[8] = (v287_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v289_a = v23_lead + 40;
              float v290_data_pre = glb_m1[v69_g ? (v289_a) : (0)];
              float v290_data = v69_g ? (v290_data_pre) : (0.0f);
              float v294_data = ir1[0];
              ir1[0] = (v294_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v300_data = ir1[1];
              ir1[1] = (v300_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v306_data = ir1[2];
              ir1[2] = (v306_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v312_data = ir1[3];
              ir1[3] = (v312_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v318_data = ir1[4];
              ir1[4] = (v318_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v324_data = ir1[5];
              ir1[5] = (v324_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v330_data = ir1[6];
              ir1[6] = (v330_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v336_data = ir1[7];
              ir1[7] = (v336_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v342_data = ir1[8];
              ir1[8] = (v342_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v344_a = v23_lead + 50;
              float v345_data_pre = glb_m1[v69_g ? (v344_a) : (0)];
              float v345_data = v69_g ? (v345_data_pre) : (0.0f);
              float v349_data = ir1[0];
              ir1[0] = (v349_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v355_data = ir1[1];
              ir1[1] = (v355_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v361_data = ir1[2];
              ir1[2] = (v361_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v367_data = ir1[3];
              ir1[3] = (v367_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v373_data = ir1[4];
              ir1[4] = (v373_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v379_data = ir1[5];
              ir1[5] = (v379_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v385_data = ir1[6];
              ir1[6] = (v385_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v391_data = ir1[7];
              ir1[7] = (v391_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v397_data = ir1[8];
              ir1[8] = (v397_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v399_a = v23_lead + 60;
              float v400_data_pre = glb_m1[v69_g ? (v399_a) : (0)];
              float v400_data = v69_g ? (v400_data_pre) : (0.0f);
              float v404_data = ir1[0];
              ir1[0] = (v404_data + (v400_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v410_data = ir1[1];
              ir1[1] = (v410_data + (v400_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v416_data = ir1[2];
              ir1[2] = (v416_data + (v400_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v422_data = ir1[3];
              ir1[3] = (v422_data + (v400_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v428_data = ir1[4];
              ir1[4] = (v428_data + (v400_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v434_data = ir1[5];
              ir1[5] = (v434_data + (v400_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v440_data = ir1[6];
              ir1[6] = (v440_data + (v400_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v446_data = ir1[7];
              ir1[7] = (v446_data + (v400_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v452_data = ir1[8];
              ir1[8] = (v452_data + (v400_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v454_a = v23_lead + 70;
              float v455_data_pre = glb_m1[v69_g ? (v454_a) : (0)];
              float v455_data = v69_g ? (v455_data_pre) : (0.0f);
              float v459_data = ir1[0];
              ir1[0] = (v459_data + (v455_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v465_data = ir1[1];
              ir1[1] = (v465_data + (v455_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v471_data = ir1[2];
              ir1[2] = (v471_data + (v455_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v477_data = ir1[3];
              ir1[3] = (v477_data + (v455_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v483_data = ir1[4];
              ir1[4] = (v483_data + (v455_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v489_data = ir1[5];
              ir1[5] = (v489_data + (v455_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v495_data = ir1[6];
              ir1[6] = (v495_data + (v455_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v501_data = ir1[7];
              ir1[7] = (v501_data + (v455_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v507_data = ir1[8];
              ir1[8] = (v507_data + (v455_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v509_a = v23_lead + 80;
              float v510_data_pre = glb_m1[v69_g ? (v509_a) : (0)];
              float v510_data = v69_g ? (v510_data_pre) : (0.0f);
              float v514_data = ir1[0];
              ir1[0] = (v514_data + (v510_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v520_data = ir1[1];
              ir1[1] = (v520_data + (v510_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v526_data = ir1[2];
              ir1[2] = (v526_data + (v510_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v532_data = ir1[3];
              ir1[3] = (v532_data + (v510_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v538_data = ir1[4];
              ir1[4] = (v538_data + (v510_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v544_data = ir1[5];
              ir1[5] = (v544_data + (v510_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v550_data = ir1[6];
              ir1[6] = (v550_data + (v510_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v556_data = ir1[7];
              ir1[7] = (v556_data + (v510_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v562_data = ir1[8];
              ir1[8] = (v562_data + (v510_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v564_a = v23_lead + 90;
              float v565_data_pre = glb_m1[v69_g ? (v564_a) : (0)];
              float v565_data = v69_g ? (v565_data_pre) : (0.0f);
              float v569_data = ir1[0];
              ir1[0] = (v569_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v575_data = ir1[1];
              ir1[1] = (v575_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v581_data = ir1[2];
              ir1[2] = (v581_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v587_data = ir1[3];
              ir1[3] = (v587_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v593_data = ir1[4];
              ir1[4] = (v593_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v599_data = ir1[5];
              ir1[5] = (v599_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v605_data = ir1[6];
              ir1[6] = (v605_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v611_data = ir1[7];
              ir1[7] = (v611_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v617_data = ir1[8];
              ir1[8] = (v617_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v619_a = v23_lead + 100;
              float v620_data_pre = glb_m1[v69_g ? (v619_a) : (0)];
              float v620_data = v69_g ? (v620_data_pre) : (0.0f);
              float v624_data = ir1[0];
              ir1[0] = (v624_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v630_data = ir1[1];
              ir1[1] = (v630_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v636_data = ir1[2];
              ir1[2] = (v636_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v642_data = ir1[3];
              ir1[3] = (v642_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v648_data = ir1[4];
              ir1[4] = (v648_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v654_data = ir1[5];
              ir1[5] = (v654_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v660_data = ir1[6];
              ir1[6] = (v660_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v666_data = ir1[7];
              ir1[7] = (v666_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v672_data = ir1[8];
              ir1[8] = (v672_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v674_a = v23_lead + 110;
              float v675_data_pre = glb_m1[v69_g ? (v674_a) : (0)];
              float v675_data = v69_g ? (v675_data_pre) : (0.0f);
              float v679_data = ir1[0];
              ir1[0] = (v679_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v685_data = ir1[1];
              ir1[1] = (v685_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v691_data = ir1[2];
              ir1[2] = (v691_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v697_data = ir1[3];
              ir1[3] = (v697_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v703_data = ir1[4];
              ir1[4] = (v703_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v709_data = ir1[5];
              ir1[5] = (v709_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v715_data = ir1[6];
              ir1[6] = (v715_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v721_data = ir1[7];
              ir1[7] = (v721_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v727_data = ir1[8];
              ir1[8] = (v727_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v729_a = v23_lead + 120;
              float v730_data_pre = glb_m1[v69_g ? (v729_a) : (0)];
              float v730_data = v69_g ? (v730_data_pre) : (0.0f);
              float v734_data = ir1[0];
              ir1[0] = (v734_data + (v730_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v740_data = ir1[1];
              ir1[1] = (v740_data + (v730_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v746_data = ir1[2];
              ir1[2] = (v746_data + (v730_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v752_data = ir1[3];
              ir1[3] = (v752_data + (v730_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v758_data = ir1[4];
              ir1[4] = (v758_data + (v730_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v764_data = ir1[5];
              ir1[5] = (v764_data + (v730_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v770_data = ir1[6];
              ir1[6] = (v770_data + (v730_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v776_data = ir1[7];
              ir1[7] = (v776_data + (v730_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v782_data = ir1[8];
              ir1[8] = (v782_data + (v730_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v784_a = v23_lead + 130;
              float v785_data_pre = glb_m1[v69_g ? (v784_a) : (0)];
              float v785_data = v69_g ? (v785_data_pre) : (0.0f);
              float v789_data = ir1[0];
              ir1[0] = (v789_data + (v785_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v795_data = ir1[1];
              ir1[1] = (v795_data + (v785_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v801_data = ir1[2];
              ir1[2] = (v801_data + (v785_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v807_data = ir1[3];
              ir1[3] = (v807_data + (v785_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v813_data = ir1[4];
              ir1[4] = (v813_data + (v785_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v819_data = ir1[5];
              ir1[5] = (v819_data + (v785_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v825_data = ir1[6];
              ir1[6] = (v825_data + (v785_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v831_data = ir1[7];
              ir1[7] = (v831_data + (v785_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v837_data = ir1[8];
              ir1[8] = (v837_data + (v785_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v839_a = v23_lead + 140;
              float v840_data_pre = glb_m1[v69_g ? (v839_a) : (0)];
              float v840_data = v69_g ? (v840_data_pre) : (0.0f);
              float v844_data = ir1[0];
              ir1[0] = (v844_data + (v840_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v850_data = ir1[1];
              ir1[1] = (v850_data + (v840_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v856_data = ir1[2];
              ir1[2] = (v856_data + (v840_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v862_data = ir1[3];
              ir1[3] = (v862_data + (v840_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v868_data = ir1[4];
              ir1[4] = (v868_data + (v840_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v874_data = ir1[5];
              ir1[5] = (v874_data + (v840_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v880_data = ir1[6];
              ir1[6] = (v880_data + (v840_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v886_data = ir1[7];
              ir1[7] = (v886_data + (v840_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v892_data = ir1[8];
              ir1[8] = (v892_data + (v840_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              int32_t v894_a = v23_lead + 150;
              float v895_data_pre = glb_m1[v69_g ? (v894_a) : (0)];
              float v895_data = v69_g ? (v895_data_pre) : (0.0f);
              float v896_data = r0[1];
              float v899_data = ir1[0];
              ir1[0] = (v899_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v896_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v902_data = r0[3];
              float v905_data = ir1[1];
              ir1[1] = (v905_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v902_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v908_data = r0[5];
              float v911_data = ir1[2];
              ir1[2] = (v911_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v908_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v914_data = r0[7];
              float v917_data = ir1[3];
              ir1[3] = (v917_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v914_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v920_data = r0[9];
              float v923_data = ir1[4];
              ir1[4] = (v923_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v920_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v926_data = r0[11];
              float v929_data = ir1[5];
              ir1[5] = (v929_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v926_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v932_data = r0[13];
              float v935_data = ir1[6];
              ir1[6] = (v935_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v932_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v938_data = r0[15];
              float v941_data = ir1[7];
              ir1[7] = (v941_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v938_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v944_data = r0[17];
              float v947_data = ir1[8];
              ir1[8] = (v947_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v944_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              int32_t v949_a = v23_lead + 160;
              float v950_data_pre = glb_m1[v69_g ? (v949_a) : (0)];
              float v950_data = v69_g ? (v950_data_pre) : (0.0f);
              float v954_data = ir1[0];
              ir1[0] = (v954_data + (v950_data * (sycl::select_from_group(item.get_sub_group(), v896_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v960_data = ir1[1];
              ir1[1] = (v960_data + (v950_data * (sycl::select_from_group(item.get_sub_group(), v902_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v966_data = ir1[2];
              ir1[2] = (v966_data + (v950_data * (sycl::select_from_group(item.get_sub_group(), v908_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v972_data = ir1[3];
              ir1[3] = (v972_data + (v950_data * (sycl::select_from_group(item.get_sub_group(), v914_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v978_data = ir1[4];
              ir1[4] = (v978_data + (v950_data * (sycl::select_from_group(item.get_sub_group(), v920_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v984_data = ir1[5];
              ir1[5] = (v984_data + (v950_data * (sycl::select_from_group(item.get_sub_group(), v926_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v990_data = ir1[6];
              ir1[6] = (v990_data + (v950_data * (sycl::select_from_group(item.get_sub_group(), v932_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v996_data = ir1[7];
              ir1[7] = (v996_data + (v950_data * (sycl::select_from_group(item.get_sub_group(), v938_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1002_data = ir1[8];
              ir1[8] = (v1002_data + (v950_data * (sycl::select_from_group(item.get_sub_group(), v944_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              // r1 = ir1
              if (v69_g) {
                #pragma unroll
                for (int32_t v1005_n1 = 0; v1005_n1 < 9; ++v1005_n1) {
                  float v1007_data = ir1[v1005_n1];
                  r1[v1005_n1] = v1007_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m4););
              float r3[9]{};
              // ir3 = +(glb_m3 * r2)
              // [(0, 10), (0, 9)] [(1, 19)]
              float ir3[9]{};
              float v1014_data_pre = glb_m3[v69_g ? (v23_lead) : (0)];
              float v1014_data = v69_g ? (v1014_data_pre) : (0.0f);
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
              float v1069_data_pre = glb_m3[v69_g ? (v124_a) : (0)];
              float v1069_data = v69_g ? (v1069_data_pre) : (0.0f);
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
              float v1124_data_pre = glb_m3[v69_g ? (v179_a) : (0)];
              float v1124_data = v69_g ? (v1124_data_pre) : (0.0f);
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
              float v1179_data_pre = glb_m3[v69_g ? (v234_a) : (0)];
              float v1179_data = v69_g ? (v1179_data_pre) : (0.0f);
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
              float v1234_data_pre = glb_m3[v69_g ? (v289_a) : (0)];
              float v1234_data = v69_g ? (v1234_data_pre) : (0.0f);
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
              float v1289_data_pre = glb_m3[v69_g ? (v344_a) : (0)];
              float v1289_data = v69_g ? (v1289_data_pre) : (0.0f);
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
              float v1344_data_pre = glb_m3[v69_g ? (v399_a) : (0)];
              float v1344_data = v69_g ? (v1344_data_pre) : (0.0f);
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
              float v1399_data_pre = glb_m3[v69_g ? (v454_a) : (0)];
              float v1399_data = v69_g ? (v1399_data_pre) : (0.0f);
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
              float v1454_data_pre = glb_m3[v69_g ? (v509_a) : (0)];
              float v1454_data = v69_g ? (v1454_data_pre) : (0.0f);
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
              float v1509_data_pre = glb_m3[v69_g ? (v564_a) : (0)];
              float v1509_data = v69_g ? (v1509_data_pre) : (0.0f);
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
              float v1564_data_pre = glb_m3[v69_g ? (v619_a) : (0)];
              float v1564_data = v69_g ? (v1564_data_pre) : (0.0f);
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
              float v1619_data_pre = glb_m3[v69_g ? (v674_a) : (0)];
              float v1619_data = v69_g ? (v1619_data_pre) : (0.0f);
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
              float v1674_data_pre = glb_m3[v69_g ? (v729_a) : (0)];
              float v1674_data = v69_g ? (v1674_data_pre) : (0.0f);
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
              float v1729_data_pre = glb_m3[v69_g ? (v784_a) : (0)];
              float v1729_data = v69_g ? (v1729_data_pre) : (0.0f);
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
              float v1784_data_pre = glb_m3[v69_g ? (v839_a) : (0)];
              float v1784_data = v69_g ? (v1784_data_pre) : (0.0f);
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
              float v1839_data_pre = glb_m3[v69_g ? (v894_a) : (0)];
              float v1839_data = v69_g ? (v1839_data_pre) : (0.0f);
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
              float v1894_data_pre = glb_m3[v69_g ? (v949_a) : (0)];
              float v1894_data = v69_g ? (v1894_data_pre) : (0.0f);
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
              float v1949_data_pre = glb_m3[v69_g ? ((v23_lead + 170)) : (0)];
              float v1949_data = v69_g ? (v1949_data_pre) : (0.0f);
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
              if (v69_g) {
                #pragma unroll
                for (int32_t v2004_n1 = 0; v2004_n1 < 9; ++v2004_n1) {
                  float v2006_data = ir3[v2004_n1];
                  float v2007_data = r1[v2004_n1];
                  r3[v2004_n1] = (v2007_data + v2006_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v69_g) {
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


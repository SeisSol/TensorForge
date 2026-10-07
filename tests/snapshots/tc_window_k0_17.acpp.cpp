// === base name ===
kernel_28d4ade0f09ee80e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_28d4ade0f09ee80e = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_28d4ade0f09ee80e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_28d4ade0f09ee80e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_28d4ade0f09ee80e(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_28d4ade0f09ee80e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_28d4ade0f09ee80e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_28d4ade0f09ee80e(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_28d4ade0f09ee80e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×17) {0..16}×{0..17} none
        //   m2 20×9(17×9) {0..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 153 + 0 + m2_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v21_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
                int32_t v25_lead = v21_lead + (v22_i0 * 16);
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 9; ++v23_i1) {
                  float v28_data = glb_m2[(v25_lead + (v23_i1 * 17))];
                  r0[(v22_i0 + (v23_i1 * 2))] = v28_data;
                }
              }
              if (v21_lead < 1) {
                int32_t v34_lead = v21_lead + 16_i32;
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 9; ++v32_i1) {
                  float v37_data = glb_m2[(v34_lead + (v32_i1 * 17))];
                  r0[(1 + (v32_i1 * 2))] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 16), (0, 9)] [(0, 17)]
              float ir1[9]{};
              float v45_data = glb_m1[v21_lead];
              float v46_data = r0[0];
              float v49_data = ir1[0];
              ir1[0] = (v49_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v52_data = r0[2];
              float v55_data = ir1[1];
              ir1[1] = (v55_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v58_data = r0[4];
              float v61_data = ir1[2];
              ir1[2] = (v61_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v64_data = r0[6];
              float v67_data = ir1[3];
              ir1[3] = (v67_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v70_data = r0[8];
              float v73_data = ir1[4];
              ir1[4] = (v73_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v76_data = r0[10];
              float v79_data = ir1[5];
              ir1[5] = (v79_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v82_data = r0[12];
              float v85_data = ir1[6];
              ir1[6] = (v85_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v88_data = r0[14];
              float v91_data = ir1[7];
              ir1[7] = (v91_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v94_data = r0[16];
              float v97_data = ir1[8];
              ir1[8] = (v97_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v100_data = glb_m1[(v21_lead + 16)];
              float v104_data = ir1[0];
              ir1[0] = (v104_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v110_data = ir1[1];
              ir1[1] = (v110_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v116_data = ir1[2];
              ir1[2] = (v116_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v122_data = ir1[3];
              ir1[3] = (v122_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v128_data = ir1[4];
              ir1[4] = (v128_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v134_data = ir1[5];
              ir1[5] = (v134_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v140_data = ir1[6];
              ir1[6] = (v140_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v146_data = ir1[7];
              ir1[7] = (v146_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v152_data = ir1[8];
              ir1[8] = (v152_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v155_data = glb_m1[(v21_lead + 32)];
              float v159_data = ir1[0];
              ir1[0] = (v159_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v165_data = ir1[1];
              ir1[1] = (v165_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v171_data = ir1[2];
              ir1[2] = (v171_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v177_data = ir1[3];
              ir1[3] = (v177_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v183_data = ir1[4];
              ir1[4] = (v183_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v189_data = ir1[5];
              ir1[5] = (v189_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v195_data = ir1[6];
              ir1[6] = (v195_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v201_data = ir1[7];
              ir1[7] = (v201_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v207_data = ir1[8];
              ir1[8] = (v207_data + (v155_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v210_data = glb_m1[(v21_lead + 48)];
              float v214_data = ir1[0];
              ir1[0] = (v214_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v220_data = ir1[1];
              ir1[1] = (v220_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v226_data = ir1[2];
              ir1[2] = (v226_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v232_data = ir1[3];
              ir1[3] = (v232_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v238_data = ir1[4];
              ir1[4] = (v238_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v244_data = ir1[5];
              ir1[5] = (v244_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v250_data = ir1[6];
              ir1[6] = (v250_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v256_data = ir1[7];
              ir1[7] = (v256_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v262_data = ir1[8];
              ir1[8] = (v262_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v265_data = glb_m1[(v21_lead + 64)];
              float v269_data = ir1[0];
              ir1[0] = (v269_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v275_data = ir1[1];
              ir1[1] = (v275_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v281_data = ir1[2];
              ir1[2] = (v281_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v287_data = ir1[3];
              ir1[3] = (v287_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v293_data = ir1[4];
              ir1[4] = (v293_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v299_data = ir1[5];
              ir1[5] = (v299_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v305_data = ir1[6];
              ir1[6] = (v305_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v311_data = ir1[7];
              ir1[7] = (v311_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v317_data = ir1[8];
              ir1[8] = (v317_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v320_data = glb_m1[(v21_lead + 80)];
              float v324_data = ir1[0];
              ir1[0] = (v324_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v330_data = ir1[1];
              ir1[1] = (v330_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v336_data = ir1[2];
              ir1[2] = (v336_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v342_data = ir1[3];
              ir1[3] = (v342_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v348_data = ir1[4];
              ir1[4] = (v348_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v354_data = ir1[5];
              ir1[5] = (v354_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v360_data = ir1[6];
              ir1[6] = (v360_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v366_data = ir1[7];
              ir1[7] = (v366_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v372_data = ir1[8];
              ir1[8] = (v372_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v375_data = glb_m1[(v21_lead + 96)];
              float v379_data = ir1[0];
              ir1[0] = (v379_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v385_data = ir1[1];
              ir1[1] = (v385_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v391_data = ir1[2];
              ir1[2] = (v391_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v397_data = ir1[3];
              ir1[3] = (v397_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v403_data = ir1[4];
              ir1[4] = (v403_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v409_data = ir1[5];
              ir1[5] = (v409_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v415_data = ir1[6];
              ir1[6] = (v415_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v421_data = ir1[7];
              ir1[7] = (v421_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v427_data = ir1[8];
              ir1[8] = (v427_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v430_data = glb_m1[(v21_lead + 112)];
              float v434_data = ir1[0];
              ir1[0] = (v434_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v440_data = ir1[1];
              ir1[1] = (v440_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v446_data = ir1[2];
              ir1[2] = (v446_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v452_data = ir1[3];
              ir1[3] = (v452_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v458_data = ir1[4];
              ir1[4] = (v458_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v464_data = ir1[5];
              ir1[5] = (v464_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v470_data = ir1[6];
              ir1[6] = (v470_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v476_data = ir1[7];
              ir1[7] = (v476_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v482_data = ir1[8];
              ir1[8] = (v482_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v485_data = glb_m1[(v21_lead + 128)];
              float v489_data = ir1[0];
              ir1[0] = (v489_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v495_data = ir1[1];
              ir1[1] = (v495_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v501_data = ir1[2];
              ir1[2] = (v501_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v507_data = ir1[3];
              ir1[3] = (v507_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v513_data = ir1[4];
              ir1[4] = (v513_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v519_data = ir1[5];
              ir1[5] = (v519_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v525_data = ir1[6];
              ir1[6] = (v525_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v531_data = ir1[7];
              ir1[7] = (v531_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v537_data = ir1[8];
              ir1[8] = (v537_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v540_data = glb_m1[(v21_lead + 144)];
              float v544_data = ir1[0];
              ir1[0] = (v544_data + (v540_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v550_data = ir1[1];
              ir1[1] = (v550_data + (v540_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v556_data = ir1[2];
              ir1[2] = (v556_data + (v540_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v562_data = ir1[3];
              ir1[3] = (v562_data + (v540_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v568_data = ir1[4];
              ir1[4] = (v568_data + (v540_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v574_data = ir1[5];
              ir1[5] = (v574_data + (v540_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v580_data = ir1[6];
              ir1[6] = (v580_data + (v540_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v586_data = ir1[7];
              ir1[7] = (v586_data + (v540_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v592_data = ir1[8];
              ir1[8] = (v592_data + (v540_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v595_data = glb_m1[(v21_lead + 160)];
              float v599_data = ir1[0];
              ir1[0] = (v599_data + (v595_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v605_data = ir1[1];
              ir1[1] = (v605_data + (v595_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v611_data = ir1[2];
              ir1[2] = (v611_data + (v595_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v617_data = ir1[3];
              ir1[3] = (v617_data + (v595_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v623_data = ir1[4];
              ir1[4] = (v623_data + (v595_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v629_data = ir1[5];
              ir1[5] = (v629_data + (v595_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v635_data = ir1[6];
              ir1[6] = (v635_data + (v595_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v641_data = ir1[7];
              ir1[7] = (v641_data + (v595_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v647_data = ir1[8];
              ir1[8] = (v647_data + (v595_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v650_data = glb_m1[(v21_lead + 176)];
              float v654_data = ir1[0];
              ir1[0] = (v654_data + (v650_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v660_data = ir1[1];
              ir1[1] = (v660_data + (v650_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v666_data = ir1[2];
              ir1[2] = (v666_data + (v650_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v672_data = ir1[3];
              ir1[3] = (v672_data + (v650_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v678_data = ir1[4];
              ir1[4] = (v678_data + (v650_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v684_data = ir1[5];
              ir1[5] = (v684_data + (v650_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v690_data = ir1[6];
              ir1[6] = (v690_data + (v650_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v696_data = ir1[7];
              ir1[7] = (v696_data + (v650_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v702_data = ir1[8];
              ir1[8] = (v702_data + (v650_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v705_data = glb_m1[(v21_lead + 192)];
              float v709_data = ir1[0];
              ir1[0] = (v709_data + (v705_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v715_data = ir1[1];
              ir1[1] = (v715_data + (v705_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v721_data = ir1[2];
              ir1[2] = (v721_data + (v705_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v727_data = ir1[3];
              ir1[3] = (v727_data + (v705_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v733_data = ir1[4];
              ir1[4] = (v733_data + (v705_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v739_data = ir1[5];
              ir1[5] = (v739_data + (v705_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v745_data = ir1[6];
              ir1[6] = (v745_data + (v705_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v751_data = ir1[7];
              ir1[7] = (v751_data + (v705_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v757_data = ir1[8];
              ir1[8] = (v757_data + (v705_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v760_data = glb_m1[(v21_lead + 208)];
              float v764_data = ir1[0];
              ir1[0] = (v764_data + (v760_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v770_data = ir1[1];
              ir1[1] = (v770_data + (v760_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v776_data = ir1[2];
              ir1[2] = (v776_data + (v760_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v782_data = ir1[3];
              ir1[3] = (v782_data + (v760_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v788_data = ir1[4];
              ir1[4] = (v788_data + (v760_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v794_data = ir1[5];
              ir1[5] = (v794_data + (v760_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v800_data = ir1[6];
              ir1[6] = (v800_data + (v760_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v806_data = ir1[7];
              ir1[7] = (v806_data + (v760_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v812_data = ir1[8];
              ir1[8] = (v812_data + (v760_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v815_data = glb_m1[(v21_lead + 224)];
              float v819_data = ir1[0];
              ir1[0] = (v819_data + (v815_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v825_data = ir1[1];
              ir1[1] = (v825_data + (v815_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v831_data = ir1[2];
              ir1[2] = (v831_data + (v815_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v837_data = ir1[3];
              ir1[3] = (v837_data + (v815_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v843_data = ir1[4];
              ir1[4] = (v843_data + (v815_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v849_data = ir1[5];
              ir1[5] = (v849_data + (v815_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v855_data = ir1[6];
              ir1[6] = (v855_data + (v815_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v861_data = ir1[7];
              ir1[7] = (v861_data + (v815_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v867_data = ir1[8];
              ir1[8] = (v867_data + (v815_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v870_data = glb_m1[(v21_lead + 240)];
              float v874_data = ir1[0];
              ir1[0] = (v874_data + (v870_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v880_data = ir1[1];
              ir1[1] = (v880_data + (v870_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v886_data = ir1[2];
              ir1[2] = (v886_data + (v870_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v892_data = ir1[3];
              ir1[3] = (v892_data + (v870_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v898_data = ir1[4];
              ir1[4] = (v898_data + (v870_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v904_data = ir1[5];
              ir1[5] = (v904_data + (v870_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v910_data = ir1[6];
              ir1[6] = (v910_data + (v870_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v916_data = ir1[7];
              ir1[7] = (v916_data + (v870_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v922_data = ir1[8];
              ir1[8] = (v922_data + (v870_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v925_data = glb_m1[(v21_lead + 256)];
              float v926_data = r0[1];
              float v929_data = ir1[0];
              ir1[0] = (v929_data + (v925_data * (sycl::select_from_group(item.get_sub_group(), v926_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v932_data = r0[3];
              float v935_data = ir1[1];
              ir1[1] = (v935_data + (v925_data * (sycl::select_from_group(item.get_sub_group(), v932_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v938_data = r0[5];
              float v941_data = ir1[2];
              ir1[2] = (v941_data + (v925_data * (sycl::select_from_group(item.get_sub_group(), v938_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v944_data = r0[7];
              float v947_data = ir1[3];
              ir1[3] = (v947_data + (v925_data * (sycl::select_from_group(item.get_sub_group(), v944_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v950_data = r0[9];
              float v953_data = ir1[4];
              ir1[4] = (v953_data + (v925_data * (sycl::select_from_group(item.get_sub_group(), v950_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v956_data = r0[11];
              float v959_data = ir1[5];
              ir1[5] = (v959_data + (v925_data * (sycl::select_from_group(item.get_sub_group(), v956_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v962_data = r0[13];
              float v965_data = ir1[6];
              ir1[6] = (v965_data + (v925_data * (sycl::select_from_group(item.get_sub_group(), v962_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v968_data = r0[15];
              float v971_data = ir1[7];
              ir1[7] = (v971_data + (v925_data * (sycl::select_from_group(item.get_sub_group(), v968_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v974_data = r0[17];
              float v977_data = ir1[8];
              ir1[8] = (v977_data + (v925_data * (sycl::select_from_group(item.get_sub_group(), v974_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              // r1 = ir1
              #pragma unroll
              for (int32_t v979_n0 = 0; v979_n0 < 1; ++v979_n0) {
                #pragma unroll
                for (int32_t v980_n1 = 0; v980_n1 < 9; ++v980_n1) {
                  int32_t v981_a = v979_n0 + v980_n1;
                  float v982_data = ir1[v981_a];
                  r1[v981_a] = v982_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v983_i0 = 0; v983_i0 < 1; ++v983_i0) {
                int32_t v988_lead = v21_lead + (v983_i0 * 16);
                #pragma unroll
                for (int32_t v984_i1 = 0; v984_i1 < 9; ++v984_i1) {
                  float v986_data = r1[(v983_i0 + v984_i1)];
                  glb_m0[(v988_lead + (v984_i1 * 16))] = v986_data;
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


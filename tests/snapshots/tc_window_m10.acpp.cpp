// === base name ===
kernel_b97bafa7cc264134

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b97bafa7cc264134 = {{16, 16, 1}, 16, 10, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b97bafa7cc264134(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b97bafa7cc264134(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b97bafa7cc264134(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_b97bafa7cc264134(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b97bafa7cc264134(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b97bafa7cc264134(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b97bafa7cc264134(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 153 + 0 + m2_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v21_lead = item.get_local_id(2) % 16;
              if (v21_lead >= 1) {
                int32_t v26_a = v21_lead - 1;
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 9; ++v23_i1) {
                  float v29_data = glb_m2[(v26_a + (v23_i1 * 17))];
                  r0[(v23_i1 * 2)] = v29_data;
                }
              }
              if (v21_lead < 2) {
                int32_t v36_a = (v21_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
                  float v39_data = glb_m2[(v36_a + (v33_i1 * 17))];
                  r0[(1 + (v33_i1 * 2))] = v39_data;
                }
              }
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 10), (0, 9)] [(1, 18)]
              float ir1[9]{};
              bool v47_g = v21_lead < 10;
              float v48_data_pre = glb_m1[v47_g ? (v21_lead) : (0)];
              float v48_data = v47_g ? (v48_data_pre) : (0.0f);
              float v49_data = r0[0];
              float v52_data = ir1[0];
              ir1[0] = (v52_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v55_data = r0[2];
              float v58_data = ir1[1];
              ir1[1] = (v58_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v61_data = r0[4];
              float v64_data = ir1[2];
              ir1[2] = (v64_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v67_data = r0[6];
              float v70_data = ir1[3];
              ir1[3] = (v70_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v73_data = r0[8];
              float v76_data = ir1[4];
              ir1[4] = (v76_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v79_data = r0[10];
              float v82_data = ir1[5];
              ir1[5] = (v82_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v85_data = r0[12];
              float v88_data = ir1[6];
              ir1[6] = (v88_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v91_data = r0[14];
              float v94_data = ir1[7];
              ir1[7] = (v94_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v97_data = r0[16];
              float v100_data = ir1[8];
              ir1[8] = (v100_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v103_data_pre = glb_m1[v47_g ? ((v21_lead + 10)) : (0)];
              float v103_data = v47_g ? (v103_data_pre) : (0.0f);
              float v107_data = ir1[0];
              ir1[0] = (v107_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v113_data = ir1[1];
              ir1[1] = (v113_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v119_data = ir1[2];
              ir1[2] = (v119_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v125_data = ir1[3];
              ir1[3] = (v125_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v131_data = ir1[4];
              ir1[4] = (v131_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v137_data = ir1[5];
              ir1[5] = (v137_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v143_data = ir1[6];
              ir1[6] = (v143_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v149_data = ir1[7];
              ir1[7] = (v149_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v155_data = ir1[8];
              ir1[8] = (v155_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v158_data_pre = glb_m1[v47_g ? ((v21_lead + 20)) : (0)];
              float v158_data = v47_g ? (v158_data_pre) : (0.0f);
              float v162_data = ir1[0];
              ir1[0] = (v162_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v168_data = ir1[1];
              ir1[1] = (v168_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v174_data = ir1[2];
              ir1[2] = (v174_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v180_data = ir1[3];
              ir1[3] = (v180_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v186_data = ir1[4];
              ir1[4] = (v186_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v192_data = ir1[5];
              ir1[5] = (v192_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v198_data = ir1[6];
              ir1[6] = (v198_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v204_data = ir1[7];
              ir1[7] = (v204_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v210_data = ir1[8];
              ir1[8] = (v210_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v213_data_pre = glb_m1[v47_g ? ((v21_lead + 30)) : (0)];
              float v213_data = v47_g ? (v213_data_pre) : (0.0f);
              float v217_data = ir1[0];
              ir1[0] = (v217_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v223_data = ir1[1];
              ir1[1] = (v223_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v229_data = ir1[2];
              ir1[2] = (v229_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v235_data = ir1[3];
              ir1[3] = (v235_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v241_data = ir1[4];
              ir1[4] = (v241_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v247_data = ir1[5];
              ir1[5] = (v247_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v253_data = ir1[6];
              ir1[6] = (v253_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v259_data = ir1[7];
              ir1[7] = (v259_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v265_data = ir1[8];
              ir1[8] = (v265_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v268_data_pre = glb_m1[v47_g ? ((v21_lead + 40)) : (0)];
              float v268_data = v47_g ? (v268_data_pre) : (0.0f);
              float v272_data = ir1[0];
              ir1[0] = (v272_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v278_data = ir1[1];
              ir1[1] = (v278_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v284_data = ir1[2];
              ir1[2] = (v284_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v290_data = ir1[3];
              ir1[3] = (v290_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v296_data = ir1[4];
              ir1[4] = (v296_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v302_data = ir1[5];
              ir1[5] = (v302_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v308_data = ir1[6];
              ir1[6] = (v308_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v314_data = ir1[7];
              ir1[7] = (v314_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v320_data = ir1[8];
              ir1[8] = (v320_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v323_data_pre = glb_m1[v47_g ? ((v21_lead + 50)) : (0)];
              float v323_data = v47_g ? (v323_data_pre) : (0.0f);
              float v327_data = ir1[0];
              ir1[0] = (v327_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v333_data = ir1[1];
              ir1[1] = (v333_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v339_data = ir1[2];
              ir1[2] = (v339_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v345_data = ir1[3];
              ir1[3] = (v345_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v351_data = ir1[4];
              ir1[4] = (v351_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v357_data = ir1[5];
              ir1[5] = (v357_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v363_data = ir1[6];
              ir1[6] = (v363_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v369_data = ir1[7];
              ir1[7] = (v369_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v375_data = ir1[8];
              ir1[8] = (v375_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v378_data_pre = glb_m1[v47_g ? ((v21_lead + 60)) : (0)];
              float v378_data = v47_g ? (v378_data_pre) : (0.0f);
              float v382_data = ir1[0];
              ir1[0] = (v382_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v388_data = ir1[1];
              ir1[1] = (v388_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v394_data = ir1[2];
              ir1[2] = (v394_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v400_data = ir1[3];
              ir1[3] = (v400_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v406_data = ir1[4];
              ir1[4] = (v406_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v412_data = ir1[5];
              ir1[5] = (v412_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v418_data = ir1[6];
              ir1[6] = (v418_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v424_data = ir1[7];
              ir1[7] = (v424_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v430_data = ir1[8];
              ir1[8] = (v430_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v433_data_pre = glb_m1[v47_g ? ((v21_lead + 70)) : (0)];
              float v433_data = v47_g ? (v433_data_pre) : (0.0f);
              float v437_data = ir1[0];
              ir1[0] = (v437_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v443_data = ir1[1];
              ir1[1] = (v443_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v449_data = ir1[2];
              ir1[2] = (v449_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v455_data = ir1[3];
              ir1[3] = (v455_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v461_data = ir1[4];
              ir1[4] = (v461_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v467_data = ir1[5];
              ir1[5] = (v467_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v473_data = ir1[6];
              ir1[6] = (v473_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v479_data = ir1[7];
              ir1[7] = (v479_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v485_data = ir1[8];
              ir1[8] = (v485_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v488_data_pre = glb_m1[v47_g ? ((v21_lead + 80)) : (0)];
              float v488_data = v47_g ? (v488_data_pre) : (0.0f);
              float v492_data = ir1[0];
              ir1[0] = (v492_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v498_data = ir1[1];
              ir1[1] = (v498_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v504_data = ir1[2];
              ir1[2] = (v504_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v510_data = ir1[3];
              ir1[3] = (v510_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v516_data = ir1[4];
              ir1[4] = (v516_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v522_data = ir1[5];
              ir1[5] = (v522_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v528_data = ir1[6];
              ir1[6] = (v528_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v534_data = ir1[7];
              ir1[7] = (v534_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v540_data = ir1[8];
              ir1[8] = (v540_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v543_data_pre = glb_m1[v47_g ? ((v21_lead + 90)) : (0)];
              float v543_data = v47_g ? (v543_data_pre) : (0.0f);
              float v547_data = ir1[0];
              ir1[0] = (v547_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v553_data = ir1[1];
              ir1[1] = (v553_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v559_data = ir1[2];
              ir1[2] = (v559_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v565_data = ir1[3];
              ir1[3] = (v565_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v571_data = ir1[4];
              ir1[4] = (v571_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v577_data = ir1[5];
              ir1[5] = (v577_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v583_data = ir1[6];
              ir1[6] = (v583_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v589_data = ir1[7];
              ir1[7] = (v589_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v595_data = ir1[8];
              ir1[8] = (v595_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v598_data_pre = glb_m1[v47_g ? ((v21_lead + 100)) : (0)];
              float v598_data = v47_g ? (v598_data_pre) : (0.0f);
              float v602_data = ir1[0];
              ir1[0] = (v602_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v608_data = ir1[1];
              ir1[1] = (v608_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v614_data = ir1[2];
              ir1[2] = (v614_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v620_data = ir1[3];
              ir1[3] = (v620_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v626_data = ir1[4];
              ir1[4] = (v626_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v632_data = ir1[5];
              ir1[5] = (v632_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v638_data = ir1[6];
              ir1[6] = (v638_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v644_data = ir1[7];
              ir1[7] = (v644_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v650_data = ir1[8];
              ir1[8] = (v650_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v653_data_pre = glb_m1[v47_g ? ((v21_lead + 110)) : (0)];
              float v653_data = v47_g ? (v653_data_pre) : (0.0f);
              float v657_data = ir1[0];
              ir1[0] = (v657_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v663_data = ir1[1];
              ir1[1] = (v663_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v669_data = ir1[2];
              ir1[2] = (v669_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v675_data = ir1[3];
              ir1[3] = (v675_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v681_data = ir1[4];
              ir1[4] = (v681_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v687_data = ir1[5];
              ir1[5] = (v687_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v693_data = ir1[6];
              ir1[6] = (v693_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v699_data = ir1[7];
              ir1[7] = (v699_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v705_data = ir1[8];
              ir1[8] = (v705_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v708_data_pre = glb_m1[v47_g ? ((v21_lead + 120)) : (0)];
              float v708_data = v47_g ? (v708_data_pre) : (0.0f);
              float v712_data = ir1[0];
              ir1[0] = (v712_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v718_data = ir1[1];
              ir1[1] = (v718_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v724_data = ir1[2];
              ir1[2] = (v724_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v730_data = ir1[3];
              ir1[3] = (v730_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v736_data = ir1[4];
              ir1[4] = (v736_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v742_data = ir1[5];
              ir1[5] = (v742_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v748_data = ir1[6];
              ir1[6] = (v748_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v754_data = ir1[7];
              ir1[7] = (v754_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v760_data = ir1[8];
              ir1[8] = (v760_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v763_data_pre = glb_m1[v47_g ? ((v21_lead + 130)) : (0)];
              float v763_data = v47_g ? (v763_data_pre) : (0.0f);
              float v767_data = ir1[0];
              ir1[0] = (v767_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v773_data = ir1[1];
              ir1[1] = (v773_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v779_data = ir1[2];
              ir1[2] = (v779_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v785_data = ir1[3];
              ir1[3] = (v785_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v791_data = ir1[4];
              ir1[4] = (v791_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v797_data = ir1[5];
              ir1[5] = (v797_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v803_data = ir1[6];
              ir1[6] = (v803_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v809_data = ir1[7];
              ir1[7] = (v809_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v815_data = ir1[8];
              ir1[8] = (v815_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v818_data_pre = glb_m1[v47_g ? ((v21_lead + 140)) : (0)];
              float v818_data = v47_g ? (v818_data_pre) : (0.0f);
              float v822_data = ir1[0];
              ir1[0] = (v822_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v828_data = ir1[1];
              ir1[1] = (v828_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v834_data = ir1[2];
              ir1[2] = (v834_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v840_data = ir1[3];
              ir1[3] = (v840_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v846_data = ir1[4];
              ir1[4] = (v846_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v852_data = ir1[5];
              ir1[5] = (v852_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v858_data = ir1[6];
              ir1[6] = (v858_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v864_data = ir1[7];
              ir1[7] = (v864_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v870_data = ir1[8];
              ir1[8] = (v870_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v873_data_pre = glb_m1[v47_g ? ((v21_lead + 150)) : (0)];
              float v873_data = v47_g ? (v873_data_pre) : (0.0f);
              float v874_data = r0[1];
              float v877_data = ir1[0];
              ir1[0] = (v877_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v880_data = r0[3];
              float v883_data = ir1[1];
              ir1[1] = (v883_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v886_data = r0[5];
              float v889_data = ir1[2];
              ir1[2] = (v889_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v892_data = r0[7];
              float v895_data = ir1[3];
              ir1[3] = (v895_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v892_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v898_data = r0[9];
              float v901_data = ir1[4];
              ir1[4] = (v901_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v898_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v904_data = r0[11];
              float v907_data = ir1[5];
              ir1[5] = (v907_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v904_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v910_data = r0[13];
              float v913_data = ir1[6];
              ir1[6] = (v913_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v910_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v916_data = r0[15];
              float v919_data = ir1[7];
              ir1[7] = (v919_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v916_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v922_data = r0[17];
              float v925_data = ir1[8];
              ir1[8] = (v925_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v922_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v928_data_pre = glb_m1[v47_g ? ((v21_lead + 160)) : (0)];
              float v928_data = v47_g ? (v928_data_pre) : (0.0f);
              float v932_data = ir1[0];
              ir1[0] = (v932_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v938_data = ir1[1];
              ir1[1] = (v938_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v944_data = ir1[2];
              ir1[2] = (v944_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v950_data = ir1[3];
              ir1[3] = (v950_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v892_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v956_data = ir1[4];
              ir1[4] = (v956_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v898_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v962_data = ir1[5];
              ir1[5] = (v962_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v904_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v968_data = ir1[6];
              ir1[6] = (v968_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v910_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v974_data = ir1[7];
              ir1[7] = (v974_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v916_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v980_data = ir1[8];
              ir1[8] = (v980_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v922_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              // r1 = ir1
              if (v47_g) {
                #pragma unroll
                for (int32_t v983_n1 = 0; v983_n1 < 9; ++v983_n1) {
                  float v985_data = ir1[v983_n1];
                  r1[v983_n1] = v985_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              if (v47_g) {
                #pragma unroll
                for (int32_t v987_i1 = 0; v987_i1 < 9; ++v987_i1) {
                  float v989_data = r1[v987_i1];
                  glb_m0[(v21_lead + (v987_i1 * 10))] = v989_data;
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


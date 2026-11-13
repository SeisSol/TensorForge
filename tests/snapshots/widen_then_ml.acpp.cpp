// === base name ===
kernel_394d93e2fda2c0e4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_394d93e2fda2c0e4 = {{32, 1, 1}, 32, 20, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_394d93e2fda2c0e4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_394d93e2fda2c0e4(const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_394d93e2fda2c0e4(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_394d93e2fda2c0e4(const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_394d93e2fda2c0e4(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_394d93e2fda2c0e4(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_394d93e2fda2c0e4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (20 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 20×9(20×9) {0..20}×{0..9} strided
        //   m2 20×10(20×10) {0..20}×{0..10} none
        // operations:
        //   wide = copy(I)
        //   m1[i,j] = m2[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"I","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"strided","alias":"canonical","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"embed","bbox":[[0,0],[20,10]],"name":"m2","ordered":false,"parts":1,"shape":[20,10],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"elementwise","op":"COPY","ops":[{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,10]},{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const double *const __restrict__ glb_m2 = &m2[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 90 + 0 + m0_extraOffset];
              double *const __restrict__ glb_m1 = &m1[v9_batchId0 * 180 + 0 + m1_extraOffset];
              double r0[9]{};
              // r0 = copy(glb_m0)
              int32_t v22_lead = item.get_local_id(2) % 32;
              bool v23_g = v22_lead < 10;
              #pragma unroll
              for (int32_t v24_k1 = 0; v24_k1 < 9; ++v24_k1) {
                float v29_data_pre = glb_m0[v23_g ? ((v22_lead + (v24_k1 * 10))) : (0)];
                float v29_data = v23_g ? (v29_data_pre) : (0.0f);
                if (v23_g) {
                  r0[v24_k1] = v29_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r0);
              double r1[9]{};
              // r1 = load{g>r}(s0);
              double r2[9]{};
              // ir2 = +(glb_m2 * r1)
              // [(0, 20), (0, 9)] [(0, 10)]
              double ir2[9]{};
              bool v56_g = v22_lead < 20;
              double v57_data_pre = glb_m2[v56_g ? (v22_lead) : (0)];
              double v57_data = v56_g ? (v57_data_pre) : (0.0);
              double v58_data = r0[0];
              double v59_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              double v64_data = r0[1];
              double v65_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              double v70_data = r0[2];
              double v71_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              double v76_data = r0[3];
              double v77_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              double v82_data = r0[4];
              double v83_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              double v88_data = r0[5];
              double v89_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              double v94_data = r0[6];
              double v95_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              double v100_data = r0[7];
              double v101_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              double v106_data = r0[8];
              double v107_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              double v112_data_pre = glb_m2[v56_g ? ((v22_lead + 20)) : (0)];
              double v112_data = v56_g ? (v112_data_pre) : (0.0);
              double v114_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              double v120_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              double v126_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              double v132_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              double v138_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              double v144_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              double v150_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              double v156_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              double v162_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              double v167_data_pre = glb_m2[v56_g ? ((v22_lead + 40)) : (0)];
              double v167_data = v56_g ? (v167_data_pre) : (0.0);
              double v169_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              double v175_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              double v181_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              double v187_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              double v193_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              double v199_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              double v205_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              double v211_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              double v217_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              double v222_data_pre = glb_m2[v56_g ? ((v22_lead + 60)) : (0)];
              double v222_data = v56_g ? (v222_data_pre) : (0.0);
              double v224_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              double v230_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              double v236_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              double v242_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              double v248_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              double v254_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              double v260_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              double v266_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              double v272_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              double v277_data_pre = glb_m2[v56_g ? ((v22_lead + 80)) : (0)];
              double v277_data = v56_g ? (v277_data_pre) : (0.0);
              double v279_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              double v285_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              double v291_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              double v297_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              double v303_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              double v309_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              double v315_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              double v321_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              double v327_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              double v332_data_pre = glb_m2[v56_g ? ((v22_lead + 100)) : (0)];
              double v332_data = v56_g ? (v332_data_pre) : (0.0);
              double v334_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              double v340_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              double v346_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              double v352_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              double v358_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              double v364_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              double v370_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              double v376_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              double v382_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              double v387_data_pre = glb_m2[v56_g ? ((v22_lead + 120)) : (0)];
              double v387_data = v56_g ? (v387_data_pre) : (0.0);
              double v389_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              double v395_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              double v401_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              double v407_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              double v413_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              double v419_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              double v425_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              double v431_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              double v437_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              double v442_data_pre = glb_m2[v56_g ? ((v22_lead + 140)) : (0)];
              double v442_data = v56_g ? (v442_data_pre) : (0.0);
              double v444_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              double v450_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              double v456_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              double v462_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              double v468_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              double v474_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              double v480_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              double v486_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              double v492_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              double v497_data_pre = glb_m2[v56_g ? ((v22_lead + 160)) : (0)];
              double v497_data = v56_g ? (v497_data_pre) : (0.0);
              double v499_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              double v505_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              double v508_p = ((((((((v57_data * v65_bc) + (v112_data * v120_bc)) + (v167_data * v175_bc)) + (v222_data * v230_bc)) + (v277_data * v285_bc)) + (v332_data * v340_bc)) + (v387_data * v395_bc)) + (v442_data * v450_bc)) + (v497_data * v505_bc);
              double v511_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              double v514_p = ((((((((v57_data * v71_bc) + (v112_data * v126_bc)) + (v167_data * v181_bc)) + (v222_data * v236_bc)) + (v277_data * v291_bc)) + (v332_data * v346_bc)) + (v387_data * v401_bc)) + (v442_data * v456_bc)) + (v497_data * v511_bc);
              double v517_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              double v520_p = ((((((((v57_data * v77_bc) + (v112_data * v132_bc)) + (v167_data * v187_bc)) + (v222_data * v242_bc)) + (v277_data * v297_bc)) + (v332_data * v352_bc)) + (v387_data * v407_bc)) + (v442_data * v462_bc)) + (v497_data * v517_bc);
              double v523_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              double v526_p = ((((((((v57_data * v83_bc) + (v112_data * v138_bc)) + (v167_data * v193_bc)) + (v222_data * v248_bc)) + (v277_data * v303_bc)) + (v332_data * v358_bc)) + (v387_data * v413_bc)) + (v442_data * v468_bc)) + (v497_data * v523_bc);
              double v529_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              double v532_p = ((((((((v57_data * v89_bc) + (v112_data * v144_bc)) + (v167_data * v199_bc)) + (v222_data * v254_bc)) + (v277_data * v309_bc)) + (v332_data * v364_bc)) + (v387_data * v419_bc)) + (v442_data * v474_bc)) + (v497_data * v529_bc);
              double v535_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              double v538_p = ((((((((v57_data * v95_bc) + (v112_data * v150_bc)) + (v167_data * v205_bc)) + (v222_data * v260_bc)) + (v277_data * v315_bc)) + (v332_data * v370_bc)) + (v387_data * v425_bc)) + (v442_data * v480_bc)) + (v497_data * v535_bc);
              double v541_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              double v544_p = ((((((((v57_data * v101_bc) + (v112_data * v156_bc)) + (v167_data * v211_bc)) + (v222_data * v266_bc)) + (v277_data * v321_bc)) + (v332_data * v376_bc)) + (v387_data * v431_bc)) + (v442_data * v486_bc)) + (v497_data * v541_bc);
              double v547_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              double v550_p = ((((((((v57_data * v107_bc) + (v112_data * v162_bc)) + (v167_data * v217_bc)) + (v222_data * v272_bc)) + (v277_data * v327_bc)) + (v332_data * v382_bc)) + (v387_data * v437_bc)) + (v442_data * v492_bc)) + (v497_data * v547_bc);
              double v552_data_pre = glb_m2[v56_g ? ((v22_lead + 180)) : (0)];
              double v552_data = v56_g ? (v552_data_pre) : (0.0);
              double v554_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              ir2[0] = ((((((((((v57_data * v59_bc) + (v112_data * v114_bc)) + (v167_data * v169_bc)) + (v222_data * v224_bc)) + (v277_data * v279_bc)) + (v332_data * v334_bc)) + (v387_data * v389_bc)) + (v442_data * v444_bc)) + (v497_data * v499_bc)) + (v552_data * v554_bc));
              double v560_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              ir2[1] = (v508_p + (v552_data * v560_bc));
              double v566_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              ir2[2] = (v514_p + (v552_data * v566_bc));
              double v572_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              ir2[3] = (v520_p + (v552_data * v572_bc));
              double v578_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              ir2[4] = (v526_p + (v552_data * v578_bc));
              double v584_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              ir2[5] = (v532_p + (v552_data * v584_bc));
              double v590_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              ir2[6] = (v538_p + (v552_data * v590_bc));
              double v596_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              ir2[7] = (v544_p + (v552_data * v596_bc));
              double v602_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              ir2[8] = (v550_p + (v552_data * v602_bc));
              // r2 = ir2
              // glb_m1 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v611_i1 = 0; v611_i1 < 9; ++v611_i1) {
                double v613_data_pre = ir2[v611_i1];
                double v613_data = v56_g ? (v613_data_pre) : (0.0);
                if (v56_g) {
                  glb_m1[(v22_lead + (v611_i1 * 20))] = v613_data;
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


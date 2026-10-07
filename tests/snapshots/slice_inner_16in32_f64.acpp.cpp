// === base name ===
kernel_acb82be000da5af6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_acb82be000da5af6 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_acb82be000da5af6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_acb82be000da5af6(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_acb82be000da5af6(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_acb82be000da5af6(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_acb82be000da5af6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_acb82be000da5af6(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_acb82be000da5af6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 16×8(16×8) {0..16}×{0..8} strided
        //   m1 32×32(32×32) {0..32}×{0..32} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{8..24}×{8..24} × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[8,8],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v7_batchId0 * 128 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v7_batchId0 * 1024 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v7_batchId0 * 128 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v21_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
                int32_t v26_off = (v21_lead + (v22_i0 * 16)) + 8;
                #pragma unroll
                for (int32_t v23_i1 = 8; v23_i1 < 24; ++v23_i1) {
                  double v29_data = glb_m1[(v26_off + (v23_i1 * 32))];
                  r0[(v22_i0 + (v23_i1 - 8))] = v29_data;
                }
              }
              double r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
                int32_t v36_lead = v21_lead + (v33_i0 * 16);
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
                  double v39_data = glb_m2[(v36_lead + (v34_i1 * 16))];
                  r1[(v33_i0 + v34_i1)] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              double r2[8]{};
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 8)] [(0, 16)]
              double ir2[8]{};
              double v43_data = r0[0];
              double v44_data = r1[0];
              double v47_data = ir2[0];
              ir2[0] = (v47_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v50_data = r1[1];
              double v53_data = ir2[1];
              ir2[1] = (v53_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v56_data = r1[2];
              double v59_data = ir2[2];
              ir2[2] = (v59_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v62_data = r1[3];
              double v65_data = ir2[3];
              ir2[3] = (v65_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v68_data = r1[4];
              double v71_data = ir2[4];
              ir2[4] = (v71_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v74_data = r1[5];
              double v77_data = ir2[5];
              ir2[5] = (v77_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v80_data = r1[6];
              double v83_data = ir2[6];
              ir2[6] = (v83_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v86_data = r1[7];
              double v89_data = ir2[7];
              ir2[7] = (v89_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v91_data = r0[1];
              double v95_data = ir2[0];
              ir2[0] = (v95_data + (v91_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v101_data = ir2[1];
              ir2[1] = (v101_data + (v91_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v107_data = ir2[2];
              ir2[2] = (v107_data + (v91_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v113_data = ir2[3];
              ir2[3] = (v113_data + (v91_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v119_data = ir2[4];
              ir2[4] = (v119_data + (v91_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v125_data = ir2[5];
              ir2[5] = (v125_data + (v91_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v131_data = ir2[6];
              ir2[6] = (v131_data + (v91_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v137_data = ir2[7];
              ir2[7] = (v137_data + (v91_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v139_data = r0[2];
              double v143_data = ir2[0];
              ir2[0] = (v143_data + (v139_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v149_data = ir2[1];
              ir2[1] = (v149_data + (v139_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v155_data = ir2[2];
              ir2[2] = (v155_data + (v139_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v161_data = ir2[3];
              ir2[3] = (v161_data + (v139_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v167_data = ir2[4];
              ir2[4] = (v167_data + (v139_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v173_data = ir2[5];
              ir2[5] = (v173_data + (v139_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v179_data = ir2[6];
              ir2[6] = (v179_data + (v139_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v185_data = ir2[7];
              ir2[7] = (v185_data + (v139_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v187_data = r0[3];
              double v191_data = ir2[0];
              ir2[0] = (v191_data + (v187_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v197_data = ir2[1];
              ir2[1] = (v197_data + (v187_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v203_data = ir2[2];
              ir2[2] = (v203_data + (v187_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v209_data = ir2[3];
              ir2[3] = (v209_data + (v187_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v215_data = ir2[4];
              ir2[4] = (v215_data + (v187_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v221_data = ir2[5];
              ir2[5] = (v221_data + (v187_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v227_data = ir2[6];
              ir2[6] = (v227_data + (v187_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v233_data = ir2[7];
              ir2[7] = (v233_data + (v187_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v235_data = r0[4];
              double v239_data = ir2[0];
              ir2[0] = (v239_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v245_data = ir2[1];
              ir2[1] = (v245_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v251_data = ir2[2];
              ir2[2] = (v251_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v257_data = ir2[3];
              ir2[3] = (v257_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v263_data = ir2[4];
              ir2[4] = (v263_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v269_data = ir2[5];
              ir2[5] = (v269_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v275_data = ir2[6];
              ir2[6] = (v275_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v281_data = ir2[7];
              ir2[7] = (v281_data + (v235_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v283_data = r0[5];
              double v287_data = ir2[0];
              ir2[0] = (v287_data + (v283_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v293_data = ir2[1];
              ir2[1] = (v293_data + (v283_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v299_data = ir2[2];
              ir2[2] = (v299_data + (v283_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v305_data = ir2[3];
              ir2[3] = (v305_data + (v283_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v311_data = ir2[4];
              ir2[4] = (v311_data + (v283_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v317_data = ir2[5];
              ir2[5] = (v317_data + (v283_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v323_data = ir2[6];
              ir2[6] = (v323_data + (v283_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v329_data = ir2[7];
              ir2[7] = (v329_data + (v283_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v331_data = r0[6];
              double v335_data = ir2[0];
              ir2[0] = (v335_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v341_data = ir2[1];
              ir2[1] = (v341_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v347_data = ir2[2];
              ir2[2] = (v347_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v353_data = ir2[3];
              ir2[3] = (v353_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v359_data = ir2[4];
              ir2[4] = (v359_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v365_data = ir2[5];
              ir2[5] = (v365_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v371_data = ir2[6];
              ir2[6] = (v371_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v377_data = ir2[7];
              ir2[7] = (v377_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v379_data = r0[7];
              double v383_data = ir2[0];
              ir2[0] = (v383_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v389_data = ir2[1];
              ir2[1] = (v389_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v395_data = ir2[2];
              ir2[2] = (v395_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v401_data = ir2[3];
              ir2[3] = (v401_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v407_data = ir2[4];
              ir2[4] = (v407_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v413_data = ir2[5];
              ir2[5] = (v413_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v419_data = ir2[6];
              ir2[6] = (v419_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v425_data = ir2[7];
              ir2[7] = (v425_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v427_data = r0[8];
              double v431_data = ir2[0];
              ir2[0] = (v431_data + (v427_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v437_data = ir2[1];
              ir2[1] = (v437_data + (v427_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v443_data = ir2[2];
              ir2[2] = (v443_data + (v427_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v449_data = ir2[3];
              ir2[3] = (v449_data + (v427_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v455_data = ir2[4];
              ir2[4] = (v455_data + (v427_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v461_data = ir2[5];
              ir2[5] = (v461_data + (v427_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v467_data = ir2[6];
              ir2[6] = (v467_data + (v427_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v473_data = ir2[7];
              ir2[7] = (v473_data + (v427_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v475_data = r0[9];
              double v479_data = ir2[0];
              ir2[0] = (v479_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v485_data = ir2[1];
              ir2[1] = (v485_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v491_data = ir2[2];
              ir2[2] = (v491_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v497_data = ir2[3];
              ir2[3] = (v497_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v503_data = ir2[4];
              ir2[4] = (v503_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v509_data = ir2[5];
              ir2[5] = (v509_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v515_data = ir2[6];
              ir2[6] = (v515_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v521_data = ir2[7];
              ir2[7] = (v521_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v523_data = r0[10];
              double v527_data = ir2[0];
              ir2[0] = (v527_data + (v523_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v533_data = ir2[1];
              ir2[1] = (v533_data + (v523_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v539_data = ir2[2];
              ir2[2] = (v539_data + (v523_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v545_data = ir2[3];
              ir2[3] = (v545_data + (v523_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v551_data = ir2[4];
              ir2[4] = (v551_data + (v523_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v557_data = ir2[5];
              ir2[5] = (v557_data + (v523_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v563_data = ir2[6];
              ir2[6] = (v563_data + (v523_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v569_data = ir2[7];
              ir2[7] = (v569_data + (v523_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v571_data = r0[11];
              double v575_data = ir2[0];
              ir2[0] = (v575_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v581_data = ir2[1];
              ir2[1] = (v581_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v587_data = ir2[2];
              ir2[2] = (v587_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v593_data = ir2[3];
              ir2[3] = (v593_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v599_data = ir2[4];
              ir2[4] = (v599_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v605_data = ir2[5];
              ir2[5] = (v605_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v611_data = ir2[6];
              ir2[6] = (v611_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v617_data = ir2[7];
              ir2[7] = (v617_data + (v571_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v619_data = r0[12];
              double v623_data = ir2[0];
              ir2[0] = (v623_data + (v619_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v629_data = ir2[1];
              ir2[1] = (v629_data + (v619_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v635_data = ir2[2];
              ir2[2] = (v635_data + (v619_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v641_data = ir2[3];
              ir2[3] = (v641_data + (v619_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v647_data = ir2[4];
              ir2[4] = (v647_data + (v619_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v653_data = ir2[5];
              ir2[5] = (v653_data + (v619_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v659_data = ir2[6];
              ir2[6] = (v659_data + (v619_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v665_data = ir2[7];
              ir2[7] = (v665_data + (v619_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v667_data = r0[13];
              double v671_data = ir2[0];
              ir2[0] = (v671_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v677_data = ir2[1];
              ir2[1] = (v677_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v683_data = ir2[2];
              ir2[2] = (v683_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v689_data = ir2[3];
              ir2[3] = (v689_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v695_data = ir2[4];
              ir2[4] = (v695_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v701_data = ir2[5];
              ir2[5] = (v701_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v707_data = ir2[6];
              ir2[6] = (v707_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v713_data = ir2[7];
              ir2[7] = (v713_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v715_data = r0[14];
              double v719_data = ir2[0];
              ir2[0] = (v719_data + (v715_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v725_data = ir2[1];
              ir2[1] = (v725_data + (v715_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v731_data = ir2[2];
              ir2[2] = (v731_data + (v715_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v737_data = ir2[3];
              ir2[3] = (v737_data + (v715_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v743_data = ir2[4];
              ir2[4] = (v743_data + (v715_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v749_data = ir2[5];
              ir2[5] = (v749_data + (v715_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v755_data = ir2[6];
              ir2[6] = (v755_data + (v715_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v761_data = ir2[7];
              ir2[7] = (v761_data + (v715_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v763_data = r0[15];
              double v767_data = ir2[0];
              ir2[0] = (v767_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v773_data = ir2[1];
              ir2[1] = (v773_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v779_data = ir2[2];
              ir2[2] = (v779_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v785_data = ir2[3];
              ir2[3] = (v785_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v791_data = ir2[4];
              ir2[4] = (v791_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v797_data = ir2[5];
              ir2[5] = (v797_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v803_data = ir2[6];
              ir2[6] = (v803_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v809_data = ir2[7];
              ir2[7] = (v809_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v811_n0 = 0; v811_n0 < 1; ++v811_n0) {
                #pragma unroll
                for (int32_t v812_n1 = 0; v812_n1 < 8; ++v812_n1) {
                  int32_t v813_a = v811_n0 + v812_n1;
                  double v814_data = ir2[v813_a];
                  r2[v813_a] = v814_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v815_i0 = 0; v815_i0 < 1; ++v815_i0) {
                int32_t v820_lead = v21_lead + (v815_i0 * 16);
                #pragma unroll
                for (int32_t v816_i1 = 0; v816_i1 < 8; ++v816_i1) {
                  double v818_data = r2[(v815_i0 + v816_i1)];
                  glb_m0[(v820_lead + (v816_i1 * 16))] = v818_data;
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


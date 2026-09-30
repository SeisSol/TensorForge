// === base name ===
kernel_2e3d0690ef6e65fb

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2e3d0690ef6e65fb = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2e3d0690ef6e65fb(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2e3d0690ef6e65fb(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2e3d0690ef6e65fb(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_2e3d0690ef6e65fb(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2e3d0690ef6e65fb(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2e3d0690ef6e65fb(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2e3d0690ef6e65fb(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(4×12) {4..8}×{0..12} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   D = abs(N)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v3_batchId0 * 48 + 0 + m3_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v18_lead = item.get_local_id(2) % 16;
              bool v19_g = v18_lead < 12;
              if (v19_g) {
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 12; ++v20_i1) {
                  float v25_data = glb_m1[(v18_lead + (v20_i1 * 12))];
                  r0[v20_i1] = v25_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m2);
              if (v19_g) {
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
                  float v33_data = glb_m2[(v18_lead + (v28_i1 * 12))];
                  r1[v28_i1] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[12]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir2[12]{};
              float v37_data = r0[0];
              float v38_data = r1[0];
              float v41_data = ir2[0];
              ir2[0] = (v41_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v44_data = r1[1];
              float v47_data = ir2[1];
              ir2[1] = (v47_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v50_data = r1[2];
              float v53_data = ir2[2];
              ir2[2] = (v53_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v56_data = r1[3];
              float v59_data = ir2[3];
              ir2[3] = (v59_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v62_data = r1[4];
              float v65_data = ir2[4];
              ir2[4] = (v65_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v68_data = r1[5];
              float v71_data = ir2[5];
              ir2[5] = (v71_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v74_data = r1[6];
              float v77_data = ir2[6];
              ir2[6] = (v77_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v80_data = r1[7];
              float v83_data = ir2[7];
              ir2[7] = (v83_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v86_data = r1[8];
              float v89_data = ir2[8];
              ir2[8] = (v89_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v92_data = r1[9];
              float v95_data = ir2[9];
              ir2[9] = (v95_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v98_data = r1[10];
              float v101_data = ir2[10];
              ir2[10] = (v101_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v104_data = r1[11];
              float v107_data = ir2[11];
              ir2[11] = (v107_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v109_data = r0[1];
              float v113_data = ir2[0];
              ir2[0] = (v113_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v119_data = ir2[1];
              ir2[1] = (v119_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v125_data = ir2[2];
              ir2[2] = (v125_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v131_data = ir2[3];
              ir2[3] = (v131_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v137_data = ir2[4];
              ir2[4] = (v137_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v143_data = ir2[5];
              ir2[5] = (v143_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v149_data = ir2[6];
              ir2[6] = (v149_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v155_data = ir2[7];
              ir2[7] = (v155_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v161_data = ir2[8];
              ir2[8] = (v161_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v167_data = ir2[9];
              ir2[9] = (v167_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v173_data = ir2[10];
              ir2[10] = (v173_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v179_data = ir2[11];
              ir2[11] = (v179_data + (v109_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v181_data = r0[2];
              float v185_data = ir2[0];
              ir2[0] = (v185_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v191_data = ir2[1];
              ir2[1] = (v191_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v197_data = ir2[2];
              ir2[2] = (v197_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v203_data = ir2[3];
              ir2[3] = (v203_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v209_data = ir2[4];
              ir2[4] = (v209_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v215_data = ir2[5];
              ir2[5] = (v215_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v221_data = ir2[6];
              ir2[6] = (v221_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v227_data = ir2[7];
              ir2[7] = (v227_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v233_data = ir2[8];
              ir2[8] = (v233_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v239_data = ir2[9];
              ir2[9] = (v239_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v245_data = ir2[10];
              ir2[10] = (v245_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v251_data = ir2[11];
              ir2[11] = (v251_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v253_data = r0[3];
              float v257_data = ir2[0];
              ir2[0] = (v257_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v263_data = ir2[1];
              ir2[1] = (v263_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v269_data = ir2[2];
              ir2[2] = (v269_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v275_data = ir2[3];
              ir2[3] = (v275_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v281_data = ir2[4];
              ir2[4] = (v281_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v287_data = ir2[5];
              ir2[5] = (v287_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v293_data = ir2[6];
              ir2[6] = (v293_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v299_data = ir2[7];
              ir2[7] = (v299_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v305_data = ir2[8];
              ir2[8] = (v305_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v311_data = ir2[9];
              ir2[9] = (v311_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v317_data = ir2[10];
              ir2[10] = (v317_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v323_data = ir2[11];
              ir2[11] = (v323_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v325_data = r0[4];
              float v329_data = ir2[0];
              ir2[0] = (v329_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v335_data = ir2[1];
              ir2[1] = (v335_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v341_data = ir2[2];
              ir2[2] = (v341_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v347_data = ir2[3];
              ir2[3] = (v347_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v353_data = ir2[4];
              ir2[4] = (v353_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v359_data = ir2[5];
              ir2[5] = (v359_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v365_data = ir2[6];
              ir2[6] = (v365_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v371_data = ir2[7];
              ir2[7] = (v371_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v377_data = ir2[8];
              ir2[8] = (v377_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v383_data = ir2[9];
              ir2[9] = (v383_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v389_data = ir2[10];
              ir2[10] = (v389_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v395_data = ir2[11];
              ir2[11] = (v395_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v397_data = r0[5];
              float v401_data = ir2[0];
              ir2[0] = (v401_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v407_data = ir2[1];
              ir2[1] = (v407_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v413_data = ir2[2];
              ir2[2] = (v413_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v419_data = ir2[3];
              ir2[3] = (v419_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v425_data = ir2[4];
              ir2[4] = (v425_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v431_data = ir2[5];
              ir2[5] = (v431_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v437_data = ir2[6];
              ir2[6] = (v437_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v443_data = ir2[7];
              ir2[7] = (v443_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v449_data = ir2[8];
              ir2[8] = (v449_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v455_data = ir2[9];
              ir2[9] = (v455_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v461_data = ir2[10];
              ir2[10] = (v461_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v467_data = ir2[11];
              ir2[11] = (v467_data + (v397_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v469_data = r0[6];
              float v473_data = ir2[0];
              ir2[0] = (v473_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v479_data = ir2[1];
              ir2[1] = (v479_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v485_data = ir2[2];
              ir2[2] = (v485_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v491_data = ir2[3];
              ir2[3] = (v491_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v497_data = ir2[4];
              ir2[4] = (v497_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v503_data = ir2[5];
              ir2[5] = (v503_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v509_data = ir2[6];
              ir2[6] = (v509_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v515_data = ir2[7];
              ir2[7] = (v515_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v521_data = ir2[8];
              ir2[8] = (v521_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v527_data = ir2[9];
              ir2[9] = (v527_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v533_data = ir2[10];
              ir2[10] = (v533_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v539_data = ir2[11];
              ir2[11] = (v539_data + (v469_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v541_data = r0[7];
              float v545_data = ir2[0];
              ir2[0] = (v545_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v551_data = ir2[1];
              ir2[1] = (v551_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v557_data = ir2[2];
              ir2[2] = (v557_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v563_data = ir2[3];
              ir2[3] = (v563_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v569_data = ir2[4];
              ir2[4] = (v569_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v575_data = ir2[5];
              ir2[5] = (v575_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v581_data = ir2[6];
              ir2[6] = (v581_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v587_data = ir2[7];
              ir2[7] = (v587_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v593_data = ir2[8];
              ir2[8] = (v593_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v599_data = ir2[9];
              ir2[9] = (v599_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v605_data = ir2[10];
              ir2[10] = (v605_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v611_data = ir2[11];
              ir2[11] = (v611_data + (v541_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v613_data = r0[8];
              float v617_data = ir2[0];
              ir2[0] = (v617_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v623_data = ir2[1];
              ir2[1] = (v623_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v629_data = ir2[2];
              ir2[2] = (v629_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v635_data = ir2[3];
              ir2[3] = (v635_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v641_data = ir2[4];
              ir2[4] = (v641_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v647_data = ir2[5];
              ir2[5] = (v647_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v653_data = ir2[6];
              ir2[6] = (v653_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v659_data = ir2[7];
              ir2[7] = (v659_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v665_data = ir2[8];
              ir2[8] = (v665_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v671_data = ir2[9];
              ir2[9] = (v671_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v677_data = ir2[10];
              ir2[10] = (v677_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v683_data = ir2[11];
              ir2[11] = (v683_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v685_data = r0[9];
              float v689_data = ir2[0];
              ir2[0] = (v689_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v695_data = ir2[1];
              ir2[1] = (v695_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v701_data = ir2[2];
              ir2[2] = (v701_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v707_data = ir2[3];
              ir2[3] = (v707_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v713_data = ir2[4];
              ir2[4] = (v713_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v719_data = ir2[5];
              ir2[5] = (v719_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v725_data = ir2[6];
              ir2[6] = (v725_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v731_data = ir2[7];
              ir2[7] = (v731_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v737_data = ir2[8];
              ir2[8] = (v737_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v743_data = ir2[9];
              ir2[9] = (v743_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v749_data = ir2[10];
              ir2[10] = (v749_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v755_data = ir2[11];
              ir2[11] = (v755_data + (v685_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v757_data = r0[10];
              float v761_data = ir2[0];
              ir2[0] = (v761_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v767_data = ir2[1];
              ir2[1] = (v767_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v773_data = ir2[2];
              ir2[2] = (v773_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v779_data = ir2[3];
              ir2[3] = (v779_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v785_data = ir2[4];
              ir2[4] = (v785_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v791_data = ir2[5];
              ir2[5] = (v791_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v797_data = ir2[6];
              ir2[6] = (v797_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v803_data = ir2[7];
              ir2[7] = (v803_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v809_data = ir2[8];
              ir2[8] = (v809_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v815_data = ir2[9];
              ir2[9] = (v815_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v821_data = ir2[10];
              ir2[10] = (v821_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v827_data = ir2[11];
              ir2[11] = (v827_data + (v757_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v829_data = r0[11];
              float v833_data = ir2[0];
              ir2[0] = (v833_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v839_data = ir2[1];
              ir2[1] = (v839_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v845_data = ir2[2];
              ir2[2] = (v845_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v851_data = ir2[3];
              ir2[3] = (v851_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v857_data = ir2[4];
              ir2[4] = (v857_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v863_data = ir2[5];
              ir2[5] = (v863_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v869_data = ir2[6];
              ir2[6] = (v869_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v875_data = ir2[7];
              ir2[7] = (v875_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v881_data = ir2[8];
              ir2[8] = (v881_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v887_data = ir2[9];
              ir2[9] = (v887_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v893_data = ir2[10];
              ir2[10] = (v893_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v899_data = ir2[11];
              ir2[11] = (v899_data + (v829_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r2 = ir2
              if (v19_g) {
                #pragma unroll
                for (int32_t v901_n1 = 0; v901_n1 < 12; ++v901_n1) {
                  float v903_data = ir2[v901_n1];
                  r2[v901_n1] = v903_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v19_g) {
                #pragma unroll
                for (int32_t v904_i1 = 0; v904_i1 < 12; ++v904_i1) {
                  float v906_data = r2[v904_i1];
                  glb_m0[(v18_lead + (v904_i1 * 12))] = v906_data;
                }
              }
              float r3[12]{};
              // r3 = abs(glb_m3)
              bool v912_g = v18_lead < 4;
              if (v912_g) {
                int32_t v917_a = (v18_lead + 4) - 4;
                #pragma unroll
                for (int32_t v913_k1 = 0; v913_k1 < 12; ++v913_k1) {
                  float v920_data = glb_m3[(v917_a + (v913_k1 * 4))];
                  r3[v913_k1] = (sycl::fabs(v920_data));
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v912_g) {
                int32_t v929_off = v18_lead + 4;
                #pragma unroll
                for (int32_t v924_i1 = 0; v924_i1 < 12; ++v924_i1) {
                  float v926_data = r3[v924_i1];
                  glb_m0[(v929_off + (v924_i1 * 12))] = v926_data;
                }
              }
              if (v18_lead >= 12) {
                int32_t v937_off = (v18_lead + -16_i32) + 4;
                #pragma unroll
                for (int32_t v933_z1 = 0; v933_z1 < 12; ++v933_z1) {
                  glb_m0[(v937_off + (v933_z1 * 12))] = 0.0f;
                }
              }
              if ((v18_lead >= 4) && (v18_lead < 8)) {
                int32_t v947_off = v18_lead + 4;
                #pragma unroll
                for (int32_t v943_z1 = 0; v943_z1 < 12; ++v943_z1) {
                  glb_m0[(v947_off + (v943_z1 * 12))] = 0.0f;
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


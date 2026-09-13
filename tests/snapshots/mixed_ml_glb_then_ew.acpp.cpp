// === base name ===
kernel_a7b91c84b9313b9d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a7b91c84b9313b9d = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a7b91c84b9313b9d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a7b91c84b9313b9d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a7b91c84b9313b9d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_a7b91c84b9313b9d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a7b91c84b9313b9d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a7b91c84b9313b9d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a7b91c84b9313b9d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   C = abs(M)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"M","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
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
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v3_batchId0 * 64 + 0 + m3_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v18_lead = item.get_local_id(2) % 16;
              bool v19_g = v18_lead < 8;
              if (v19_g) {
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 8; ++v20_i1) {
                  float v25_data = glb_m1[(v18_lead + (v20_i1 * 8))];
                  r0[v20_i1] = v25_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              if (v19_g) {
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
                  float v33_data = glb_m2[(v18_lead + (v28_i1 * 8))];
                  r1[v28_i1] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir2[8]{};
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
              float v85_data = r0[1];
              float v89_data = ir2[0];
              ir2[0] = (v89_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v95_data = ir2[1];
              ir2[1] = (v95_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v101_data = ir2[2];
              ir2[2] = (v101_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v107_data = ir2[3];
              ir2[3] = (v107_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v113_data = ir2[4];
              ir2[4] = (v113_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v119_data = ir2[5];
              ir2[5] = (v119_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v125_data = ir2[6];
              ir2[6] = (v125_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v131_data = ir2[7];
              ir2[7] = (v131_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v133_data = r0[2];
              float v137_data = ir2[0];
              ir2[0] = (v137_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v143_data = ir2[1];
              ir2[1] = (v143_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v149_data = ir2[2];
              ir2[2] = (v149_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v155_data = ir2[3];
              ir2[3] = (v155_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v161_data = ir2[4];
              ir2[4] = (v161_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v167_data = ir2[5];
              ir2[5] = (v167_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v173_data = ir2[6];
              ir2[6] = (v173_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v179_data = ir2[7];
              ir2[7] = (v179_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v181_data = r0[3];
              float v185_data = ir2[0];
              ir2[0] = (v185_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v191_data = ir2[1];
              ir2[1] = (v191_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v197_data = ir2[2];
              ir2[2] = (v197_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v203_data = ir2[3];
              ir2[3] = (v203_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v209_data = ir2[4];
              ir2[4] = (v209_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v215_data = ir2[5];
              ir2[5] = (v215_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v221_data = ir2[6];
              ir2[6] = (v221_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v227_data = ir2[7];
              ir2[7] = (v227_data + (v181_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v229_data = r0[4];
              float v233_data = ir2[0];
              ir2[0] = (v233_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v239_data = ir2[1];
              ir2[1] = (v239_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v245_data = ir2[2];
              ir2[2] = (v245_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v251_data = ir2[3];
              ir2[3] = (v251_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v257_data = ir2[4];
              ir2[4] = (v257_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v263_data = ir2[5];
              ir2[5] = (v263_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v269_data = ir2[6];
              ir2[6] = (v269_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v275_data = ir2[7];
              ir2[7] = (v275_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v277_data = r0[5];
              float v281_data = ir2[0];
              ir2[0] = (v281_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v287_data = ir2[1];
              ir2[1] = (v287_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v293_data = ir2[2];
              ir2[2] = (v293_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v299_data = ir2[3];
              ir2[3] = (v299_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v305_data = ir2[4];
              ir2[4] = (v305_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v311_data = ir2[5];
              ir2[5] = (v311_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v317_data = ir2[6];
              ir2[6] = (v317_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v323_data = ir2[7];
              ir2[7] = (v323_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v325_data = r0[6];
              float v329_data = ir2[0];
              ir2[0] = (v329_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v335_data = ir2[1];
              ir2[1] = (v335_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v341_data = ir2[2];
              ir2[2] = (v341_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v347_data = ir2[3];
              ir2[3] = (v347_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v353_data = ir2[4];
              ir2[4] = (v353_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v359_data = ir2[5];
              ir2[5] = (v359_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v365_data = ir2[6];
              ir2[6] = (v365_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v371_data = ir2[7];
              ir2[7] = (v371_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v373_data = r0[7];
              float v377_data = ir2[0];
              ir2[0] = (v377_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v383_data = ir2[1];
              ir2[1] = (v383_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v389_data = ir2[2];
              ir2[2] = (v389_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v395_data = ir2[3];
              ir2[3] = (v395_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v401_data = ir2[4];
              ir2[4] = (v401_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v407_data = ir2[5];
              ir2[5] = (v407_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v413_data = ir2[6];
              ir2[6] = (v413_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v419_data = ir2[7];
              ir2[7] = (v419_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              if (v19_g) {
                #pragma unroll
                for (int32_t v421_n1 = 0; v421_n1 < 8; ++v421_n1) {
                  float v423_data = ir2[v421_n1];
                  r2[v421_n1] = v423_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v19_g) {
                #pragma unroll
                for (int32_t v424_i1 = 0; v424_i1 < 8; ++v424_i1) {
                  float v426_data = r2[v424_i1];
                  glb_m0[(v18_lead + (v424_i1 * 8))] = v426_data;
                }
              }
              // glb_m3 = abs(glb_m0)
              if (v19_g) {
                #pragma unroll
                for (int32_t v431_k1 = 0; v431_k1 < 8; ++v431_k1) {
                  int32_t v435_a = v18_lead + (v431_k1 * 8);
                  float v436_data = glb_m0[v435_a];
                  glb_m3[v435_a] = (sycl::fabs(v436_data));
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


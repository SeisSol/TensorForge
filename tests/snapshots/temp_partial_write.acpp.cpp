// === base name ===
kernel_d1964e1f85cba76f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d1964e1f85cba76f = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d1964e1f85cba76f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d1964e1f85cba76f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d1964e1f85cba76f(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_d1964e1f85cba76f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d1964e1f85cba76f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d1964e1f85cba76f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d1964e1f85cba76f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..12}×{0..6} = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 144 + 0 + m3_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v23_lead = item.get_local_id(2) % 16;
              bool v24_g = v23_lead < 12;
              if (v24_g) {
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
                  float v30_data = glb_m0[(v23_lead + (v25_i1 * 12))];
                  r0[v25_i1] = v30_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              if (v24_g) {
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
                  float v38_data = glb_m1[(v23_lead + (v33_i1 * 12))];
                  r1[v33_i1] = v38_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v24_g) {
                #pragma unroll
                for (int32_t v41_i1 = 0; v41_i1 < 12; ++v41_i1) {
                  float v46_data = glb_m3[(v23_lead + (v41_i1 * 12))];
                  r3[v41_i1] = v46_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float v49_data = r0[0];
              float v50_data = r1[0];
              float v53_data = r2[0];
              r2[0] = (v53_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v56_data = r1[1];
              float v59_data = r2[1];
              r2[1] = (v59_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v62_data = r1[2];
              float v65_data = r2[2];
              r2[2] = (v65_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v68_data = r1[3];
              float v71_data = r2[3];
              r2[3] = (v71_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v74_data = r1[4];
              float v77_data = r2[4];
              r2[4] = (v77_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v80_data = r1[5];
              float v83_data = r2[5];
              r2[5] = (v83_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v85_data = r0[1];
              float v89_data = r2[0];
              r2[0] = (v89_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v95_data = r2[1];
              r2[1] = (v95_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v101_data = r2[2];
              r2[2] = (v101_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v107_data = r2[3];
              r2[3] = (v107_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v113_data = r2[4];
              r2[4] = (v113_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v119_data = r2[5];
              r2[5] = (v119_data + (v85_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v121_data = r0[2];
              float v125_data = r2[0];
              r2[0] = (v125_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v131_data = r2[1];
              r2[1] = (v131_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v137_data = r2[2];
              r2[2] = (v137_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v143_data = r2[3];
              r2[3] = (v143_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v149_data = r2[4];
              r2[4] = (v149_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v155_data = r2[5];
              r2[5] = (v155_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v157_data = r0[3];
              float v161_data = r2[0];
              r2[0] = (v161_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v167_data = r2[1];
              r2[1] = (v167_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v173_data = r2[2];
              r2[2] = (v173_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v179_data = r2[3];
              r2[3] = (v179_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v185_data = r2[4];
              r2[4] = (v185_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v191_data = r2[5];
              r2[5] = (v191_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v193_data = r0[4];
              float v197_data = r2[0];
              r2[0] = (v197_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v203_data = r2[1];
              r2[1] = (v203_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v209_data = r2[2];
              r2[2] = (v209_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v215_data = r2[3];
              r2[3] = (v215_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v221_data = r2[4];
              r2[4] = (v221_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v227_data = r2[5];
              r2[5] = (v227_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v229_data = r0[5];
              float v233_data = r2[0];
              r2[0] = (v233_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v239_data = r2[1];
              r2[1] = (v239_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v245_data = r2[2];
              r2[2] = (v245_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v251_data = r2[3];
              r2[3] = (v251_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v257_data = r2[4];
              r2[4] = (v257_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v263_data = r2[5];
              r2[5] = (v263_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v265_data = r0[6];
              float v269_data = r2[0];
              r2[0] = (v269_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v275_data = r2[1];
              r2[1] = (v275_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v281_data = r2[2];
              r2[2] = (v281_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v287_data = r2[3];
              r2[3] = (v287_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v293_data = r2[4];
              r2[4] = (v293_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v299_data = r2[5];
              r2[5] = (v299_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v301_data = r0[7];
              float v305_data = r2[0];
              r2[0] = (v305_data + (v301_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v311_data = r2[1];
              r2[1] = (v311_data + (v301_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v317_data = r2[2];
              r2[2] = (v317_data + (v301_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v323_data = r2[3];
              r2[3] = (v323_data + (v301_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v329_data = r2[4];
              r2[4] = (v329_data + (v301_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v335_data = r2[5];
              r2[5] = (v335_data + (v301_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v337_data = r0[8];
              float v341_data = r2[0];
              r2[0] = (v341_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v347_data = r2[1];
              r2[1] = (v347_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v353_data = r2[2];
              r2[2] = (v353_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v359_data = r2[3];
              r2[3] = (v359_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v365_data = r2[4];
              r2[4] = (v365_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v371_data = r2[5];
              r2[5] = (v371_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v373_data = r0[9];
              float v377_data = r2[0];
              r2[0] = (v377_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v383_data = r2[1];
              r2[1] = (v383_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v389_data = r2[2];
              r2[2] = (v389_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v395_data = r2[3];
              r2[3] = (v395_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v401_data = r2[4];
              r2[4] = (v401_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v407_data = r2[5];
              r2[5] = (v407_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v409_data = r0[10];
              float v413_data = r2[0];
              r2[0] = (v413_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v419_data = r2[1];
              r2[1] = (v419_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v425_data = r2[2];
              r2[2] = (v425_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v431_data = r2[3];
              r2[3] = (v431_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v437_data = r2[4];
              r2[4] = (v437_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v443_data = r2[5];
              r2[5] = (v443_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v445_data = r0[11];
              float v449_data = r2[0];
              r2[0] = (v449_data + (v445_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v455_data = r2[1];
              r2[1] = (v455_data + (v445_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v461_data = r2[2];
              r2[2] = (v461_data + (v445_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v467_data = r2[3];
              r2[3] = (v467_data + (v445_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v473_data = r2[4];
              r2[4] = (v473_data + (v445_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v479_data = r2[5];
              r2[5] = (v479_data + (v445_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // s0 = store{r>s, clear}(localShrMem0, r2);
              if (v24_g) {
                #pragma unroll
                for (int32_t v481_z1 = 6; v481_z1 < 12; ++v481_z1) {
                  int32_t v486_a = v23_lead + (v481_z1 * 12);
                  s0[(v486_a ^ ((v486_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v24_g) {
                #pragma unroll
                for (int32_t v490_i1 = 0; v490_i1 < 6; ++v490_i1) {
                  float v492_data = r2[v490_i1];
                  int32_t v496_a = v23_lead + (v490_i1 * 12);
                  s0[(v496_a ^ ((v496_a >> 4) & 15))] = v492_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r4[12]{};
              // ir4 = +(r3 * s0)
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v502_data = r3[0];
              sycl::group_barrier(item.get_sub_group());
              float v503_data = s0[0];
              float v505_data = ir4[0];
              ir4[0] = (v505_data + (v502_data * v503_data));
              float v508_data = s0[12];
              float v510_data = ir4[1];
              ir4[1] = (v510_data + (v502_data * v508_data));
              float v513_data = s0[25];
              float v515_data = ir4[2];
              ir4[2] = (v515_data + (v502_data * v513_data));
              float v518_data = s0[38];
              float v520_data = ir4[3];
              ir4[3] = (v520_data + (v502_data * v518_data));
              float v523_data = s0[51];
              float v525_data = ir4[4];
              ir4[4] = (v525_data + (v502_data * v523_data));
              float v528_data = s0[63];
              float v530_data = ir4[5];
              ir4[5] = (v530_data + (v502_data * v528_data));
              float v533_data = s0[76];
              float v535_data = ir4[6];
              ir4[6] = (v535_data + (v502_data * v533_data));
              float v538_data = s0[81];
              float v540_data = ir4[7];
              ir4[7] = (v540_data + (v502_data * v538_data));
              float v543_data = s0[102];
              float v545_data = ir4[8];
              ir4[8] = (v545_data + (v502_data * v543_data));
              float v548_data = s0[106];
              float v550_data = ir4[9];
              ir4[9] = (v550_data + (v502_data * v548_data));
              float v553_data = s0[127];
              float v555_data = ir4[10];
              ir4[10] = (v555_data + (v502_data * v553_data));
              float v558_data = s0[140];
              float v560_data = ir4[11];
              ir4[11] = (v560_data + (v502_data * v558_data));
              float v562_data = r3[1];
              float v563_data = s0[1];
              float v565_data = ir4[0];
              ir4[0] = (v565_data + (v562_data * v563_data));
              float v568_data = s0[13];
              float v570_data = ir4[1];
              ir4[1] = (v570_data + (v562_data * v568_data));
              float v573_data = s0[24];
              float v575_data = ir4[2];
              ir4[2] = (v575_data + (v562_data * v573_data));
              float v578_data = s0[39];
              float v580_data = ir4[3];
              ir4[3] = (v580_data + (v562_data * v578_data));
              float v583_data = s0[50];
              float v585_data = ir4[4];
              ir4[4] = (v585_data + (v562_data * v583_data));
              float v588_data = s0[62];
              float v590_data = ir4[5];
              ir4[5] = (v590_data + (v562_data * v588_data));
              float v593_data = s0[77];
              float v595_data = ir4[6];
              ir4[6] = (v595_data + (v562_data * v593_data));
              float v598_data = s0[80];
              float v600_data = ir4[7];
              ir4[7] = (v600_data + (v562_data * v598_data));
              float v603_data = s0[103];
              float v605_data = ir4[8];
              ir4[8] = (v605_data + (v562_data * v603_data));
              float v608_data = s0[107];
              float v610_data = ir4[9];
              ir4[9] = (v610_data + (v562_data * v608_data));
              float v613_data = s0[126];
              float v615_data = ir4[10];
              ir4[10] = (v615_data + (v562_data * v613_data));
              float v618_data = s0[141];
              float v620_data = ir4[11];
              ir4[11] = (v620_data + (v562_data * v618_data));
              float v622_data = r3[2];
              float v623_data = s0[2];
              float v625_data = ir4[0];
              ir4[0] = (v625_data + (v622_data * v623_data));
              float v628_data = s0[14];
              float v630_data = ir4[1];
              ir4[1] = (v630_data + (v622_data * v628_data));
              float v633_data = s0[27];
              float v635_data = ir4[2];
              ir4[2] = (v635_data + (v622_data * v633_data));
              float v638_data = s0[36];
              float v640_data = ir4[3];
              ir4[3] = (v640_data + (v622_data * v638_data));
              float v643_data = s0[49];
              float v645_data = ir4[4];
              ir4[4] = (v645_data + (v622_data * v643_data));
              float v648_data = s0[61];
              float v650_data = ir4[5];
              ir4[5] = (v650_data + (v622_data * v648_data));
              float v653_data = s0[78];
              float v655_data = ir4[6];
              ir4[6] = (v655_data + (v622_data * v653_data));
              float v658_data = s0[83];
              float v660_data = ir4[7];
              ir4[7] = (v660_data + (v622_data * v658_data));
              float v663_data = s0[100];
              float v665_data = ir4[8];
              ir4[8] = (v665_data + (v622_data * v663_data));
              float v668_data = s0[104];
              float v670_data = ir4[9];
              ir4[9] = (v670_data + (v622_data * v668_data));
              float v673_data = s0[125];
              float v675_data = ir4[10];
              ir4[10] = (v675_data + (v622_data * v673_data));
              float v678_data = s0[142];
              float v680_data = ir4[11];
              ir4[11] = (v680_data + (v622_data * v678_data));
              float v682_data = r3[3];
              float v683_data = s0[3];
              float v685_data = ir4[0];
              ir4[0] = (v685_data + (v682_data * v683_data));
              float v688_data = s0[15];
              float v690_data = ir4[1];
              ir4[1] = (v690_data + (v682_data * v688_data));
              float v693_data = s0[26];
              float v695_data = ir4[2];
              ir4[2] = (v695_data + (v682_data * v693_data));
              float v698_data = s0[37];
              float v700_data = ir4[3];
              ir4[3] = (v700_data + (v682_data * v698_data));
              float v703_data = s0[48];
              float v705_data = ir4[4];
              ir4[4] = (v705_data + (v682_data * v703_data));
              float v708_data = s0[60];
              float v710_data = ir4[5];
              ir4[5] = (v710_data + (v682_data * v708_data));
              float v713_data = s0[79];
              float v715_data = ir4[6];
              ir4[6] = (v715_data + (v682_data * v713_data));
              float v718_data = s0[82];
              float v720_data = ir4[7];
              ir4[7] = (v720_data + (v682_data * v718_data));
              float v723_data = s0[101];
              float v725_data = ir4[8];
              ir4[8] = (v725_data + (v682_data * v723_data));
              float v728_data = s0[105];
              float v730_data = ir4[9];
              ir4[9] = (v730_data + (v682_data * v728_data));
              float v733_data = s0[124];
              float v735_data = ir4[10];
              ir4[10] = (v735_data + (v682_data * v733_data));
              float v738_data = s0[143];
              float v740_data = ir4[11];
              ir4[11] = (v740_data + (v682_data * v738_data));
              float v742_data = r3[4];
              float v743_data = s0[4];
              float v745_data = ir4[0];
              ir4[0] = (v745_data + (v742_data * v743_data));
              float v748_data = s0[17];
              float v750_data = ir4[1];
              ir4[1] = (v750_data + (v742_data * v748_data));
              float v753_data = s0[29];
              float v755_data = ir4[2];
              ir4[2] = (v755_data + (v742_data * v753_data));
              float v758_data = s0[42];
              float v760_data = ir4[3];
              ir4[3] = (v760_data + (v742_data * v758_data));
              float v763_data = s0[55];
              float v765_data = ir4[4];
              ir4[4] = (v765_data + (v742_data * v763_data));
              float v768_data = s0[68];
              float v770_data = ir4[5];
              ir4[5] = (v770_data + (v742_data * v768_data));
              float v773_data = s0[72];
              float v775_data = ir4[6];
              ir4[6] = (v775_data + (v742_data * v773_data));
              float v778_data = s0[93];
              float v780_data = ir4[7];
              ir4[7] = (v780_data + (v742_data * v778_data));
              float v783_data = s0[98];
              float v785_data = ir4[8];
              ir4[8] = (v785_data + (v742_data * v783_data));
              float v788_data = s0[119];
              float v790_data = ir4[9];
              ir4[9] = (v790_data + (v742_data * v788_data));
              float v793_data = s0[123];
              float v795_data = ir4[10];
              ir4[10] = (v795_data + (v742_data * v793_data));
              float v798_data = s0[128];
              float v800_data = ir4[11];
              ir4[11] = (v800_data + (v742_data * v798_data));
              float v802_data = r3[5];
              float v803_data = s0[5];
              float v805_data = ir4[0];
              ir4[0] = (v805_data + (v802_data * v803_data));
              float v808_data = s0[16];
              float v810_data = ir4[1];
              ir4[1] = (v810_data + (v802_data * v808_data));
              float v813_data = s0[28];
              float v815_data = ir4[2];
              ir4[2] = (v815_data + (v802_data * v813_data));
              float v818_data = s0[43];
              float v820_data = ir4[3];
              ir4[3] = (v820_data + (v802_data * v818_data));
              float v823_data = s0[54];
              float v825_data = ir4[4];
              ir4[4] = (v825_data + (v802_data * v823_data));
              float v828_data = s0[69];
              float v830_data = ir4[5];
              ir4[5] = (v830_data + (v802_data * v828_data));
              float v833_data = s0[73];
              float v835_data = ir4[6];
              ir4[6] = (v835_data + (v802_data * v833_data));
              float v838_data = s0[92];
              float v840_data = ir4[7];
              ir4[7] = (v840_data + (v802_data * v838_data));
              float v843_data = s0[99];
              float v845_data = ir4[8];
              ir4[8] = (v845_data + (v802_data * v843_data));
              float v848_data = s0[118];
              float v850_data = ir4[9];
              ir4[9] = (v850_data + (v802_data * v848_data));
              float v853_data = s0[122];
              float v855_data = ir4[10];
              ir4[10] = (v855_data + (v802_data * v853_data));
              float v858_data = s0[129];
              float v860_data = ir4[11];
              ir4[11] = (v860_data + (v802_data * v858_data));
              float v862_data = r3[6];
              float v863_data = s0[6];
              float v865_data = ir4[0];
              ir4[0] = (v865_data + (v862_data * v863_data));
              float v868_data = s0[19];
              float v870_data = ir4[1];
              ir4[1] = (v870_data + (v862_data * v868_data));
              float v873_data = s0[31];
              float v875_data = ir4[2];
              ir4[2] = (v875_data + (v862_data * v873_data));
              float v878_data = s0[40];
              float v880_data = ir4[3];
              ir4[3] = (v880_data + (v862_data * v878_data));
              float v883_data = s0[53];
              float v885_data = ir4[4];
              ir4[4] = (v885_data + (v862_data * v883_data));
              float v888_data = s0[70];
              float v890_data = ir4[5];
              ir4[5] = (v890_data + (v862_data * v888_data));
              float v893_data = s0[74];
              float v895_data = ir4[6];
              ir4[6] = (v895_data + (v862_data * v893_data));
              float v898_data = s0[95];
              float v900_data = ir4[7];
              ir4[7] = (v900_data + (v862_data * v898_data));
              float v903_data = s0[96];
              float v905_data = ir4[8];
              ir4[8] = (v905_data + (v862_data * v903_data));
              float v908_data = s0[117];
              float v910_data = ir4[9];
              ir4[9] = (v910_data + (v862_data * v908_data));
              float v913_data = s0[121];
              float v915_data = ir4[10];
              ir4[10] = (v915_data + (v862_data * v913_data));
              float v918_data = s0[130];
              float v920_data = ir4[11];
              ir4[11] = (v920_data + (v862_data * v918_data));
              float v922_data = r3[7];
              float v923_data = s0[7];
              float v925_data = ir4[0];
              ir4[0] = (v925_data + (v922_data * v923_data));
              float v928_data = s0[18];
              float v930_data = ir4[1];
              ir4[1] = (v930_data + (v922_data * v928_data));
              float v933_data = s0[30];
              float v935_data = ir4[2];
              ir4[2] = (v935_data + (v922_data * v933_data));
              float v938_data = s0[41];
              float v940_data = ir4[3];
              ir4[3] = (v940_data + (v922_data * v938_data));
              float v943_data = s0[52];
              float v945_data = ir4[4];
              ir4[4] = (v945_data + (v922_data * v943_data));
              float v948_data = s0[71];
              float v950_data = ir4[5];
              ir4[5] = (v950_data + (v922_data * v948_data));
              float v953_data = s0[75];
              float v955_data = ir4[6];
              ir4[6] = (v955_data + (v922_data * v953_data));
              float v958_data = s0[94];
              float v960_data = ir4[7];
              ir4[7] = (v960_data + (v922_data * v958_data));
              float v963_data = s0[97];
              float v965_data = ir4[8];
              ir4[8] = (v965_data + (v922_data * v963_data));
              float v968_data = s0[116];
              float v970_data = ir4[9];
              ir4[9] = (v970_data + (v922_data * v968_data));
              float v973_data = s0[120];
              float v975_data = ir4[10];
              ir4[10] = (v975_data + (v922_data * v973_data));
              float v978_data = s0[131];
              float v980_data = ir4[11];
              ir4[11] = (v980_data + (v922_data * v978_data));
              float v982_data = r3[8];
              float v983_data = s0[8];
              float v985_data = ir4[0];
              ir4[0] = (v985_data + (v982_data * v983_data));
              float v988_data = s0[21];
              float v990_data = ir4[1];
              ir4[1] = (v990_data + (v982_data * v988_data));
              float v993_data = s0[34];
              float v995_data = ir4[2];
              ir4[2] = (v995_data + (v982_data * v993_data));
              float v998_data = s0[46];
              float v1000_data = ir4[3];
              ir4[3] = (v1000_data + (v982_data * v998_data));
              float v1003_data = s0[59];
              float v1005_data = ir4[4];
              ir4[4] = (v1005_data + (v982_data * v1003_data));
              float v1008_data = s0[64];
              float v1010_data = ir4[5];
              ir4[5] = (v1010_data + (v982_data * v1008_data));
              float v1013_data = s0[85];
              float v1015_data = ir4[6];
              ir4[6] = (v1015_data + (v982_data * v1013_data));
              float v1018_data = s0[89];
              float v1020_data = ir4[7];
              ir4[7] = (v1020_data + (v982_data * v1018_data));
              float v1023_data = s0[110];
              float v1025_data = ir4[8];
              ir4[8] = (v1025_data + (v982_data * v1023_data));
              float v1028_data = s0[115];
              float v1030_data = ir4[9];
              ir4[9] = (v1030_data + (v982_data * v1028_data));
              float v1033_data = s0[136];
              float v1035_data = ir4[10];
              ir4[10] = (v1035_data + (v982_data * v1033_data));
              float v1038_data = s0[132];
              float v1040_data = ir4[11];
              ir4[11] = (v1040_data + (v982_data * v1038_data));
              float v1042_data = r3[9];
              float v1043_data = s0[9];
              float v1045_data = ir4[0];
              ir4[0] = (v1045_data + (v1042_data * v1043_data));
              float v1048_data = s0[20];
              float v1050_data = ir4[1];
              ir4[1] = (v1050_data + (v1042_data * v1048_data));
              float v1053_data = s0[35];
              float v1055_data = ir4[2];
              ir4[2] = (v1055_data + (v1042_data * v1053_data));
              float v1058_data = s0[47];
              float v1060_data = ir4[3];
              ir4[3] = (v1060_data + (v1042_data * v1058_data));
              float v1063_data = s0[58];
              float v1065_data = ir4[4];
              ir4[4] = (v1065_data + (v1042_data * v1063_data));
              float v1068_data = s0[65];
              float v1070_data = ir4[5];
              ir4[5] = (v1070_data + (v1042_data * v1068_data));
              float v1073_data = s0[84];
              float v1075_data = ir4[6];
              ir4[6] = (v1075_data + (v1042_data * v1073_data));
              float v1078_data = s0[88];
              float v1080_data = ir4[7];
              ir4[7] = (v1080_data + (v1042_data * v1078_data));
              float v1083_data = s0[111];
              float v1085_data = ir4[8];
              ir4[8] = (v1085_data + (v1042_data * v1083_data));
              float v1088_data = s0[114];
              float v1090_data = ir4[9];
              ir4[9] = (v1090_data + (v1042_data * v1088_data));
              float v1093_data = s0[137];
              float v1095_data = ir4[10];
              ir4[10] = (v1095_data + (v1042_data * v1093_data));
              float v1098_data = s0[133];
              float v1100_data = ir4[11];
              ir4[11] = (v1100_data + (v1042_data * v1098_data));
              float v1102_data = r3[10];
              float v1103_data = s0[10];
              float v1105_data = ir4[0];
              ir4[0] = (v1105_data + (v1102_data * v1103_data));
              float v1108_data = s0[23];
              float v1110_data = ir4[1];
              ir4[1] = (v1110_data + (v1102_data * v1108_data));
              float v1113_data = s0[32];
              float v1115_data = ir4[2];
              ir4[2] = (v1115_data + (v1102_data * v1113_data));
              float v1118_data = s0[44];
              float v1120_data = ir4[3];
              ir4[3] = (v1120_data + (v1102_data * v1118_data));
              float v1123_data = s0[57];
              float v1125_data = ir4[4];
              ir4[4] = (v1125_data + (v1102_data * v1123_data));
              float v1128_data = s0[66];
              float v1130_data = ir4[5];
              ir4[5] = (v1130_data + (v1102_data * v1128_data));
              float v1133_data = s0[87];
              float v1135_data = ir4[6];
              ir4[6] = (v1135_data + (v1102_data * v1133_data));
              float v1138_data = s0[91];
              float v1140_data = ir4[7];
              ir4[7] = (v1140_data + (v1102_data * v1138_data));
              float v1143_data = s0[108];
              float v1145_data = ir4[8];
              ir4[8] = (v1145_data + (v1102_data * v1143_data));
              float v1148_data = s0[113];
              float v1150_data = ir4[9];
              ir4[9] = (v1150_data + (v1102_data * v1148_data));
              float v1153_data = s0[138];
              float v1155_data = ir4[10];
              ir4[10] = (v1155_data + (v1102_data * v1153_data));
              float v1158_data = s0[134];
              float v1160_data = ir4[11];
              ir4[11] = (v1160_data + (v1102_data * v1158_data));
              float v1162_data = r3[11];
              float v1163_data = s0[11];
              float v1165_data = ir4[0];
              ir4[0] = (v1165_data + (v1162_data * v1163_data));
              float v1168_data = s0[22];
              float v1170_data = ir4[1];
              ir4[1] = (v1170_data + (v1162_data * v1168_data));
              float v1173_data = s0[33];
              float v1175_data = ir4[2];
              ir4[2] = (v1175_data + (v1162_data * v1173_data));
              float v1178_data = s0[45];
              float v1180_data = ir4[3];
              ir4[3] = (v1180_data + (v1162_data * v1178_data));
              float v1183_data = s0[56];
              float v1185_data = ir4[4];
              ir4[4] = (v1185_data + (v1162_data * v1183_data));
              float v1188_data = s0[67];
              float v1190_data = ir4[5];
              ir4[5] = (v1190_data + (v1162_data * v1188_data));
              float v1193_data = s0[86];
              float v1195_data = ir4[6];
              ir4[6] = (v1195_data + (v1162_data * v1193_data));
              float v1198_data = s0[90];
              float v1200_data = ir4[7];
              ir4[7] = (v1200_data + (v1162_data * v1198_data));
              float v1203_data = s0[109];
              float v1205_data = ir4[8];
              ir4[8] = (v1205_data + (v1162_data * v1203_data));
              float v1208_data = s0[112];
              float v1210_data = ir4[9];
              ir4[9] = (v1210_data + (v1162_data * v1208_data));
              float v1213_data = s0[139];
              float v1215_data = ir4[10];
              ir4[10] = (v1215_data + (v1162_data * v1213_data));
              float v1218_data = s0[135];
              float v1220_data = ir4[11];
              ir4[11] = (v1220_data + (v1162_data * v1218_data));
              // r4 = ir4
              if (v24_g) {
                #pragma unroll
                for (int32_t v1222_n1 = 0; v1222_n1 < 12; ++v1222_n1) {
                  float v1224_data = ir4[v1222_n1];
                  r4[v1222_n1] = v1224_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v24_g) {
                #pragma unroll
                for (int32_t v1225_i1 = 0; v1225_i1 < 12; ++v1225_i1) {
                  float v1227_data = r4[v1225_i1];
                  glb_m2[(v23_lead + (v1225_i1 * 12))] = v1227_data;
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


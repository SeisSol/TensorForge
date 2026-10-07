// === base name ===
kernel_21ea6dce82c8dab8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_21ea6dce82c8dab8 = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_21ea6dce82c8dab8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_21ea6dce82c8dab8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_21ea6dce82c8dab8(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (8, 2, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 2 - 1) / 2;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 8;
  config.block[1] = 2;
  config.block[2] = 1;
  config.sharedMemBytes = 16 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_21ea6dce82c8dab8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_21ea6dce82c8dab8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_21ea6dce82c8dab8(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_21ea6dce82c8dab8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (16, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 2 per block = block 8x2x1, 64 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        //   m4 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t0[i,j] += m2[i,k] × m3[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          size_t v8_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v27_lead = item.get_local_id(2) % 8;
          for (size_t v9_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v9_batchIdGroup0 < numElements0; v9_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_row = v9_batchIdGroup0 + v8_batchIdLane0;
            const bool batchIdActive0 = v10_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v10_row]));
            size_t v12_batchId0 = batchIdActive0 ? v10_row : v9_batchIdGroup0;
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 64 + 0 + m2_extraOffset];
            const float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 64 + 0 + m3_extraOffset];
            float *const __restrict__ glb_m4 = &m4[v12_batchId0 * 64 + 0 + m4_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
              int32_t v31_lead = v27_lead + (v28_i0 * 8);
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
                float v34_data = glb_m0[(v31_lead + (v29_i1 * 8))];
                r0[(v28_i0 + v29_i1)] = v34_data;
              }
            }
            float r1[8]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v37_i0 = 0; v37_i0 < 1; ++v37_i0) {
              int32_t v40_lead = v27_lead + (v37_i0 * 8);
              #pragma unroll
              for (int32_t v38_i1 = 0; v38_i1 < 8; ++v38_i1) {
                float v43_data = glb_m1[(v40_lead + (v38_i1 * 8))];
                r1[(v37_i0 + v38_i1)] = v43_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            float r3[8]{};
            // r3 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v46_i0 = 0; v46_i0 < 1; ++v46_i0) {
              int32_t v49_lead = v27_lead + (v46_i0 * 8);
              #pragma unroll
              for (int32_t v47_i1 = 0; v47_i1 < 8; ++v47_i1) {
                float v52_data = glb_m2[(v49_lead + (v47_i1 * 8))];
                r3[(v46_i0 + v47_i1)] = v52_data;
              }
            }
            // wait(r1 = load{g>r}(glb_m1););
            float r2[8]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 8)] [(0, 8)]
            float v55_data = r0[0];
            float v56_data = r1[0];
            float v59_data = r2[0];
            r2[0] = (v59_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v62_data = r1[1];
            float v65_data = r2[1];
            r2[1] = (v65_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v68_data = r1[2];
            float v71_data = r2[2];
            r2[2] = (v71_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v74_data = r1[3];
            float v77_data = r2[3];
            r2[3] = (v77_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v80_data = r1[4];
            float v83_data = r2[4];
            r2[4] = (v83_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v86_data = r1[5];
            float v89_data = r2[5];
            r2[5] = (v89_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v92_data = r1[6];
            float v95_data = r2[6];
            r2[6] = (v95_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v98_data = r1[7];
            float v101_data = r2[7];
            r2[7] = (v101_data + (v55_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v103_data = r0[1];
            float v107_data = r2[0];
            r2[0] = (v107_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v113_data = r2[1];
            r2[1] = (v113_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v119_data = r2[2];
            r2[2] = (v119_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v125_data = r2[3];
            r2[3] = (v125_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v131_data = r2[4];
            r2[4] = (v131_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v137_data = r2[5];
            r2[5] = (v137_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v143_data = r2[6];
            r2[6] = (v143_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v149_data = r2[7];
            r2[7] = (v149_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v151_data = r0[2];
            float v155_data = r2[0];
            r2[0] = (v155_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v161_data = r2[1];
            r2[1] = (v161_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v167_data = r2[2];
            r2[2] = (v167_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v173_data = r2[3];
            r2[3] = (v173_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v179_data = r2[4];
            r2[4] = (v179_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v185_data = r2[5];
            r2[5] = (v185_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v191_data = r2[6];
            r2[6] = (v191_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v197_data = r2[7];
            r2[7] = (v197_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v199_data = r0[3];
            float v203_data = r2[0];
            r2[0] = (v203_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v209_data = r2[1];
            r2[1] = (v209_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v215_data = r2[2];
            r2[2] = (v215_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v221_data = r2[3];
            r2[3] = (v221_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v227_data = r2[4];
            r2[4] = (v227_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v233_data = r2[5];
            r2[5] = (v233_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v239_data = r2[6];
            r2[6] = (v239_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v245_data = r2[7];
            r2[7] = (v245_data + (v199_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v247_data = r0[4];
            float v251_data = r2[0];
            r2[0] = (v251_data + (v247_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v257_data = r2[1];
            r2[1] = (v257_data + (v247_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v263_data = r2[2];
            r2[2] = (v263_data + (v247_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v269_data = r2[3];
            r2[3] = (v269_data + (v247_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v275_data = r2[4];
            r2[4] = (v275_data + (v247_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v281_data = r2[5];
            r2[5] = (v281_data + (v247_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v287_data = r2[6];
            r2[6] = (v287_data + (v247_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v293_data = r2[7];
            r2[7] = (v293_data + (v247_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v295_data = r0[5];
            float v299_data = r2[0];
            r2[0] = (v299_data + (v295_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v305_data = r2[1];
            r2[1] = (v305_data + (v295_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v311_data = r2[2];
            r2[2] = (v311_data + (v295_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v317_data = r2[3];
            r2[3] = (v317_data + (v295_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v323_data = r2[4];
            r2[4] = (v323_data + (v295_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v329_data = r2[5];
            r2[5] = (v329_data + (v295_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v335_data = r2[6];
            r2[6] = (v335_data + (v295_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v341_data = r2[7];
            r2[7] = (v341_data + (v295_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v343_data = r0[6];
            float v347_data = r2[0];
            r2[0] = (v347_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v353_data = r2[1];
            r2[1] = (v353_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v359_data = r2[2];
            r2[2] = (v359_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v365_data = r2[3];
            r2[3] = (v365_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v371_data = r2[4];
            r2[4] = (v371_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v377_data = r2[5];
            r2[5] = (v377_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v383_data = r2[6];
            r2[6] = (v383_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v389_data = r2[7];
            r2[7] = (v389_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v391_data = r0[7];
            float v395_data = r2[0];
            r2[0] = (v395_data + (v391_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v401_data = r2[1];
            r2[1] = (v401_data + (v391_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v407_data = r2[2];
            r2[2] = (v407_data + (v391_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v413_data = r2[3];
            r2[3] = (v413_data + (v391_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v419_data = r2[4];
            r2[4] = (v419_data + (v391_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v425_data = r2[5];
            r2[5] = (v425_data + (v391_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v431_data = r2[6];
            r2[6] = (v431_data + (v391_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v437_data = r2[7];
            r2[7] = (v437_data + (v391_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float r4[8]{};
            // r4 = load{g>r}(glb_m3);
            #pragma unroll
            for (int32_t v440_i0 = 0; v440_i0 < 1; ++v440_i0) {
              int32_t v443_lead = v27_lead + (v440_i0 * 8);
              #pragma unroll
              for (int32_t v441_i1 = 0; v441_i1 < 8; ++v441_i1) {
                float v446_data = glb_m3[(v443_lead + (v441_i1 * 8))];
                r4[(v440_i0 + v441_i1)] = v446_data;
              }
            }
            // wait(r3 = load{g>r}(glb_m2););
            // wait(r4 = load{g>r}(glb_m3););
            float r5[8]{};
            // ir5 = +(r3 * r4)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir5[8]{};
            float v450_data = r3[0];
            float v451_data = r4[0];
            float v454_data = ir5[0];
            ir5[0] = (v454_data + (v450_data * (sycl::select_from_group(item.get_sub_group(), v451_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v457_data = r4[1];
            float v460_data = ir5[1];
            ir5[1] = (v460_data + (v450_data * (sycl::select_from_group(item.get_sub_group(), v457_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v463_data = r4[2];
            float v466_data = ir5[2];
            ir5[2] = (v466_data + (v450_data * (sycl::select_from_group(item.get_sub_group(), v463_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v469_data = r4[3];
            float v472_data = ir5[3];
            ir5[3] = (v472_data + (v450_data * (sycl::select_from_group(item.get_sub_group(), v469_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v475_data = r4[4];
            float v478_data = ir5[4];
            ir5[4] = (v478_data + (v450_data * (sycl::select_from_group(item.get_sub_group(), v475_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v481_data = r4[5];
            float v484_data = ir5[5];
            ir5[5] = (v484_data + (v450_data * (sycl::select_from_group(item.get_sub_group(), v481_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v487_data = r4[6];
            float v490_data = ir5[6];
            ir5[6] = (v490_data + (v450_data * (sycl::select_from_group(item.get_sub_group(), v487_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v493_data = r4[7];
            float v496_data = ir5[7];
            ir5[7] = (v496_data + (v450_data * (sycl::select_from_group(item.get_sub_group(), v493_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v498_data = r3[1];
            float v502_data = ir5[0];
            ir5[0] = (v502_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v451_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v508_data = ir5[1];
            ir5[1] = (v508_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v457_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v514_data = ir5[2];
            ir5[2] = (v514_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v463_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v520_data = ir5[3];
            ir5[3] = (v520_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v469_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v526_data = ir5[4];
            ir5[4] = (v526_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v475_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v532_data = ir5[5];
            ir5[5] = (v532_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v481_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v538_data = ir5[6];
            ir5[6] = (v538_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v487_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v544_data = ir5[7];
            ir5[7] = (v544_data + (v498_data * (sycl::select_from_group(item.get_sub_group(), v493_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v546_data = r3[2];
            float v550_data = ir5[0];
            ir5[0] = (v550_data + (v546_data * (sycl::select_from_group(item.get_sub_group(), v451_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v556_data = ir5[1];
            ir5[1] = (v556_data + (v546_data * (sycl::select_from_group(item.get_sub_group(), v457_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v562_data = ir5[2];
            ir5[2] = (v562_data + (v546_data * (sycl::select_from_group(item.get_sub_group(), v463_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v568_data = ir5[3];
            ir5[3] = (v568_data + (v546_data * (sycl::select_from_group(item.get_sub_group(), v469_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v574_data = ir5[4];
            ir5[4] = (v574_data + (v546_data * (sycl::select_from_group(item.get_sub_group(), v475_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v580_data = ir5[5];
            ir5[5] = (v580_data + (v546_data * (sycl::select_from_group(item.get_sub_group(), v481_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v586_data = ir5[6];
            ir5[6] = (v586_data + (v546_data * (sycl::select_from_group(item.get_sub_group(), v487_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v592_data = ir5[7];
            ir5[7] = (v592_data + (v546_data * (sycl::select_from_group(item.get_sub_group(), v493_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v594_data = r3[3];
            float v598_data = ir5[0];
            ir5[0] = (v598_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v451_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v604_data = ir5[1];
            ir5[1] = (v604_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v457_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v610_data = ir5[2];
            ir5[2] = (v610_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v463_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v616_data = ir5[3];
            ir5[3] = (v616_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v469_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v622_data = ir5[4];
            ir5[4] = (v622_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v475_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v628_data = ir5[5];
            ir5[5] = (v628_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v481_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v634_data = ir5[6];
            ir5[6] = (v634_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v487_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v640_data = ir5[7];
            ir5[7] = (v640_data + (v594_data * (sycl::select_from_group(item.get_sub_group(), v493_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v642_data = r3[4];
            float v646_data = ir5[0];
            ir5[0] = (v646_data + (v642_data * (sycl::select_from_group(item.get_sub_group(), v451_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v652_data = ir5[1];
            ir5[1] = (v652_data + (v642_data * (sycl::select_from_group(item.get_sub_group(), v457_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v658_data = ir5[2];
            ir5[2] = (v658_data + (v642_data * (sycl::select_from_group(item.get_sub_group(), v463_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v664_data = ir5[3];
            ir5[3] = (v664_data + (v642_data * (sycl::select_from_group(item.get_sub_group(), v469_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v670_data = ir5[4];
            ir5[4] = (v670_data + (v642_data * (sycl::select_from_group(item.get_sub_group(), v475_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v676_data = ir5[5];
            ir5[5] = (v676_data + (v642_data * (sycl::select_from_group(item.get_sub_group(), v481_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v682_data = ir5[6];
            ir5[6] = (v682_data + (v642_data * (sycl::select_from_group(item.get_sub_group(), v487_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v688_data = ir5[7];
            ir5[7] = (v688_data + (v642_data * (sycl::select_from_group(item.get_sub_group(), v493_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v690_data = r3[5];
            float v694_data = ir5[0];
            ir5[0] = (v694_data + (v690_data * (sycl::select_from_group(item.get_sub_group(), v451_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v700_data = ir5[1];
            ir5[1] = (v700_data + (v690_data * (sycl::select_from_group(item.get_sub_group(), v457_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v706_data = ir5[2];
            ir5[2] = (v706_data + (v690_data * (sycl::select_from_group(item.get_sub_group(), v463_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v712_data = ir5[3];
            ir5[3] = (v712_data + (v690_data * (sycl::select_from_group(item.get_sub_group(), v469_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v718_data = ir5[4];
            ir5[4] = (v718_data + (v690_data * (sycl::select_from_group(item.get_sub_group(), v475_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v724_data = ir5[5];
            ir5[5] = (v724_data + (v690_data * (sycl::select_from_group(item.get_sub_group(), v481_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v730_data = ir5[6];
            ir5[6] = (v730_data + (v690_data * (sycl::select_from_group(item.get_sub_group(), v487_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v736_data = ir5[7];
            ir5[7] = (v736_data + (v690_data * (sycl::select_from_group(item.get_sub_group(), v493_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v738_data = r3[6];
            float v742_data = ir5[0];
            ir5[0] = (v742_data + (v738_data * (sycl::select_from_group(item.get_sub_group(), v451_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v748_data = ir5[1];
            ir5[1] = (v748_data + (v738_data * (sycl::select_from_group(item.get_sub_group(), v457_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v754_data = ir5[2];
            ir5[2] = (v754_data + (v738_data * (sycl::select_from_group(item.get_sub_group(), v463_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v760_data = ir5[3];
            ir5[3] = (v760_data + (v738_data * (sycl::select_from_group(item.get_sub_group(), v469_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v766_data = ir5[4];
            ir5[4] = (v766_data + (v738_data * (sycl::select_from_group(item.get_sub_group(), v475_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v772_data = ir5[5];
            ir5[5] = (v772_data + (v738_data * (sycl::select_from_group(item.get_sub_group(), v481_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v778_data = ir5[6];
            ir5[6] = (v778_data + (v738_data * (sycl::select_from_group(item.get_sub_group(), v487_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v784_data = ir5[7];
            ir5[7] = (v784_data + (v738_data * (sycl::select_from_group(item.get_sub_group(), v493_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v786_data = r3[7];
            float v790_data = ir5[0];
            ir5[0] = (v790_data + (v786_data * (sycl::select_from_group(item.get_sub_group(), v451_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v796_data = ir5[1];
            ir5[1] = (v796_data + (v786_data * (sycl::select_from_group(item.get_sub_group(), v457_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v802_data = ir5[2];
            ir5[2] = (v802_data + (v786_data * (sycl::select_from_group(item.get_sub_group(), v463_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v808_data = ir5[3];
            ir5[3] = (v808_data + (v786_data * (sycl::select_from_group(item.get_sub_group(), v469_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v814_data = ir5[4];
            ir5[4] = (v814_data + (v786_data * (sycl::select_from_group(item.get_sub_group(), v475_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v820_data = ir5[5];
            ir5[5] = (v820_data + (v786_data * (sycl::select_from_group(item.get_sub_group(), v481_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v826_data = ir5[6];
            ir5[6] = (v826_data + (v786_data * (sycl::select_from_group(item.get_sub_group(), v487_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v832_data = ir5[7];
            ir5[7] = (v832_data + (v786_data * (sycl::select_from_group(item.get_sub_group(), v493_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r5 = ir5 + r2
            #pragma unroll
            for (int32_t v834_n0 = 0; v834_n0 < 1; ++v834_n0) {
              #pragma unroll
              for (int32_t v835_n1 = 0; v835_n1 < 8; ++v835_n1) {
                int32_t v836_a = v834_n0 + v835_n1;
                float v837_data = ir5[v836_a];
                float v838_data = r2[v836_a];
                r5[v836_a] = (v838_data + v837_data);
              }
            }
            // glb_m4 = abs(r5)
            #pragma unroll
            for (int32_t v840_k0 = 0; v840_k0 < 1; ++v840_k0) {
              #pragma unroll
              for (int32_t v841_k1 = 0; v841_k1 < 8; ++v841_k1) {
                float v843_data = r5[(v840_k0 + v841_k1)];
                float v844_e = sycl::fabs(v843_data);
                if (batchIdActive0) {
                  glb_m4[((v27_lead + (v840_k0 * 8)) + (v841_k1 * 8))] = v844_e;
                }
              }
            }
            item.barrier();
          }
        }
      });
    }
  });
}


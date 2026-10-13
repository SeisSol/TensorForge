// === base name ===
kernel_71e112f81afeb035

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_71e112f81afeb035 = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_71e112f81afeb035(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_71e112f81afeb035(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_71e112f81afeb035(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_71e112f81afeb035(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_71e112f81afeb035(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_71e112f81afeb035(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_71e112f81afeb035(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
          size_t v11_batchIdGroup0First = ((item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) < numElements0) ? (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) : 0u;
          int32_t v30_lead = item.get_local_id(2) % 8;
          float r0[8]{};
          size_t v855_row = v11_batchIdGroup0First + v8_batchIdLane0;
          const bool peel_batchIdActive0 = v855_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v855_row]));
          const float *const __restrict__ peel_glb_m0 = &m0[(peel_batchIdActive0 ? v855_row : v11_batchIdGroup0First) * 64 + 0 + m0_extraOffset];
          // r0 = load{g>r}(peel_glb_m0);
          #pragma unroll
          for (int32_t v863_i0 = 0; v863_i0 < 1; ++v863_i0) {
            int32_t v865_lead = v30_lead + (v863_i0 * 8);
            #pragma unroll
            for (int32_t v866_i1 = 0; v866_i1 < 8; ++v866_i1) {
              float v869_data = peel_glb_m0[(v865_lead + (v866_i1 * 8))];
              r0[(v863_i0 + v866_i1)] = v869_data;
            }
          }
          for (size_t v12_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v852_groupNextAhead = v12_batchIdGroup0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v854_groupNext = (v852_groupNextAhead < numElements0) ? v852_groupNextAhead : v12_batchIdGroup0;
            size_t v859_row = v854_groupNext + v8_batchIdLane0;
            const bool wrap_batchIdActive0 = v859_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v859_row]));
            const float *const __restrict__ wrap_glb_m0 = &m0[(wrap_batchIdActive0 ? v859_row : v854_groupNext) * 64 + 0 + m0_extraOffset];
            size_t v13_row = v12_batchIdGroup0 + v8_batchIdLane0;
            const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
            size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
            size_t v16_ahead1 = v15_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
            const float *const __restrict__ glb_m1 = &m1[v15_batchId0 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v15_batchId0 * 64 + 0 + m2_extraOffset];
            const float *const __restrict__ glb_m3 = &m3[v15_batchId0 * 64 + 0 + m3_extraOffset];
            float *const __restrict__ glb_m4 = &m4[v15_batchId0 * 64 + 0 + m4_extraOffset];
            float r1[8]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v40_i0 = 0; v40_i0 < 1; ++v40_i0) {
              int32_t v43_lead = v30_lead + (v40_i0 * 8);
              #pragma unroll
              for (int32_t v41_i1 = 0; v41_i1 < 8; ++v41_i1) {
                float v46_data = glb_m1[(v43_lead + (v41_i1 * 8))];
                r1[(v40_i0 + v41_i1)] = v46_data;
              }
            }
            float r3[8]{};
            // r3 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v434_i0 = 0; v434_i0 < 1; ++v434_i0) {
              int32_t v437_lead = v30_lead + (v434_i0 * 8);
              #pragma unroll
              for (int32_t v435_i1 = 0; v435_i1 < 8; ++v435_i1) {
                float v440_data = glb_m2[(v437_lead + (v435_i1 * 8))];
                r3[(v434_i0 + v435_i1)] = v440_data;
              }
            }
            float r2[8]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 8)] [(0, 8)]
            float v49_data = r0[0];
            float v50_data = r1[0];
            float v53_data = r2[0];
            r2[0] = (v53_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v56_data = r1[1];
            float v59_data = r2[1];
            r2[1] = (v59_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v62_data = r1[2];
            float v65_data = r2[2];
            r2[2] = (v65_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v68_data = r1[3];
            float v71_data = r2[3];
            r2[3] = (v71_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v74_data = r1[4];
            float v77_data = r2[4];
            r2[4] = (v77_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v80_data = r1[5];
            float v83_data = r2[5];
            r2[5] = (v83_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v86_data = r1[6];
            float v89_data = r2[6];
            r2[6] = (v89_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v92_data = r1[7];
            float v95_data = r2[7];
            r2[7] = (v95_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v97_data = r0[1];
            float v101_data = r2[0];
            r2[0] = (v101_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v107_data = r2[1];
            r2[1] = (v107_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v113_data = r2[2];
            r2[2] = (v113_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v119_data = r2[3];
            r2[3] = (v119_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v125_data = r2[4];
            r2[4] = (v125_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v131_data = r2[5];
            r2[5] = (v131_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v137_data = r2[6];
            r2[6] = (v137_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v143_data = r2[7];
            r2[7] = (v143_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v145_data = r0[2];
            float v149_data = r2[0];
            r2[0] = (v149_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v155_data = r2[1];
            r2[1] = (v155_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v161_data = r2[2];
            r2[2] = (v161_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v167_data = r2[3];
            r2[3] = (v167_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v173_data = r2[4];
            r2[4] = (v173_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v179_data = r2[5];
            r2[5] = (v179_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v185_data = r2[6];
            r2[6] = (v185_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v191_data = r2[7];
            r2[7] = (v191_data + (v145_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v193_data = r0[3];
            float v197_data = r2[0];
            r2[0] = (v197_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v203_data = r2[1];
            r2[1] = (v203_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v209_data = r2[2];
            r2[2] = (v209_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v215_data = r2[3];
            r2[3] = (v215_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v221_data = r2[4];
            r2[4] = (v221_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v227_data = r2[5];
            r2[5] = (v227_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v233_data = r2[6];
            r2[6] = (v233_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v239_data = r2[7];
            r2[7] = (v239_data + (v193_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v241_data = r0[4];
            float v245_data = r2[0];
            r2[0] = (v245_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v251_data = r2[1];
            r2[1] = (v251_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v257_data = r2[2];
            r2[2] = (v257_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v263_data = r2[3];
            r2[3] = (v263_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v269_data = r2[4];
            r2[4] = (v269_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v275_data = r2[5];
            r2[5] = (v275_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v281_data = r2[6];
            r2[6] = (v281_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v287_data = r2[7];
            r2[7] = (v287_data + (v241_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v289_data = r0[5];
            float v293_data = r2[0];
            r2[0] = (v293_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v299_data = r2[1];
            r2[1] = (v299_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v305_data = r2[2];
            r2[2] = (v305_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v311_data = r2[3];
            r2[3] = (v311_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v317_data = r2[4];
            r2[4] = (v317_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v323_data = r2[5];
            r2[5] = (v323_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v329_data = r2[6];
            r2[6] = (v329_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v335_data = r2[7];
            r2[7] = (v335_data + (v289_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v337_data = r0[6];
            float v341_data = r2[0];
            r2[0] = (v341_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v347_data = r2[1];
            r2[1] = (v347_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v353_data = r2[2];
            r2[2] = (v353_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v359_data = r2[3];
            r2[3] = (v359_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v365_data = r2[4];
            r2[4] = (v365_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v371_data = r2[5];
            r2[5] = (v371_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v377_data = r2[6];
            r2[6] = (v377_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v383_data = r2[7];
            r2[7] = (v383_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v385_data = r0[7];
            float v389_data = r2[0];
            r2[0] = (v389_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v395_data = r2[1];
            r2[1] = (v395_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v401_data = r2[2];
            r2[2] = (v401_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v407_data = r2[3];
            r2[3] = (v407_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v413_data = r2[4];
            r2[4] = (v413_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v419_data = r2[5];
            r2[5] = (v419_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v425_data = r2[6];
            r2[6] = (v425_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v431_data = r2[7];
            r2[7] = (v431_data + (v385_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float r4[8]{};
            // r4 = load{g>r}(glb_m3);
            #pragma unroll
            for (int32_t v443_i0 = 0; v443_i0 < 1; ++v443_i0) {
              int32_t v446_lead = v30_lead + (v443_i0 * 8);
              #pragma unroll
              for (int32_t v444_i1 = 0; v444_i1 < 8; ++v444_i1) {
                float v449_data = glb_m3[(v446_lead + (v444_i1 * 8))];
                r4[(v443_i0 + v444_i1)] = v449_data;
              }
            }
            float r5[8]{};
            // ir5 = +(r3 * r4)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir5[8]{};
            float v453_data = r3[0];
            float v454_data = r4[0];
            float v457_data = ir5[0];
            ir5[0] = (v457_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v460_data = r4[1];
            float v463_data = ir5[1];
            ir5[1] = (v463_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v466_data = r4[2];
            float v469_data = ir5[2];
            ir5[2] = (v469_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v472_data = r4[3];
            float v475_data = ir5[3];
            ir5[3] = (v475_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v478_data = r4[4];
            float v481_data = ir5[4];
            ir5[4] = (v481_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v484_data = r4[5];
            float v487_data = ir5[5];
            ir5[5] = (v487_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v490_data = r4[6];
            float v493_data = ir5[6];
            ir5[6] = (v493_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v496_data = r4[7];
            float v498_p = v453_data * (sycl::select_from_group(item.get_sub_group(), v496_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)));
            float v499_data = ir5[7];
            // r0 = load{g>r}(wrap_glb_m0);
            #pragma unroll
            for (int32_t v871_i0 = 0; v871_i0 < 1; ++v871_i0) {
              int32_t v873_lead = v30_lead + (v871_i0 * 8);
              #pragma unroll
              for (int32_t v874_i1 = 0; v874_i1 < 8; ++v874_i1) {
                float v877_data = wrap_glb_m0[(v873_lead + (v874_i1 * 8))];
                r0[(v871_i0 + v874_i1)] = v877_data;
              }
            }
            ir5[7] = (v499_data + v498_p);
            float v501_data = r3[1];
            float v505_data = ir5[0];
            ir5[0] = (v505_data + (v501_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v511_data = ir5[1];
            ir5[1] = (v511_data + (v501_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v517_data = ir5[2];
            ir5[2] = (v517_data + (v501_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v523_data = ir5[3];
            ir5[3] = (v523_data + (v501_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v529_data = ir5[4];
            ir5[4] = (v529_data + (v501_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v535_data = ir5[5];
            ir5[5] = (v535_data + (v501_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v541_data = ir5[6];
            ir5[6] = (v541_data + (v501_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v547_data = ir5[7];
            ir5[7] = (v547_data + (v501_data * (sycl::select_from_group(item.get_sub_group(), v496_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v549_data = r3[2];
            float v553_data = ir5[0];
            ir5[0] = (v553_data + (v549_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v559_data = ir5[1];
            ir5[1] = (v559_data + (v549_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v565_data = ir5[2];
            ir5[2] = (v565_data + (v549_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v571_data = ir5[3];
            ir5[3] = (v571_data + (v549_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v577_data = ir5[4];
            ir5[4] = (v577_data + (v549_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v583_data = ir5[5];
            ir5[5] = (v583_data + (v549_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v589_data = ir5[6];
            ir5[6] = (v589_data + (v549_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v595_data = ir5[7];
            ir5[7] = (v595_data + (v549_data * (sycl::select_from_group(item.get_sub_group(), v496_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v597_data = r3[3];
            float v601_data = ir5[0];
            ir5[0] = (v601_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v607_data = ir5[1];
            ir5[1] = (v607_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v613_data = ir5[2];
            ir5[2] = (v613_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v619_data = ir5[3];
            ir5[3] = (v619_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v625_data = ir5[4];
            ir5[4] = (v625_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v631_data = ir5[5];
            ir5[5] = (v631_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v637_data = ir5[6];
            ir5[6] = (v637_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v643_data = ir5[7];
            ir5[7] = (v643_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v496_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v645_data = r3[4];
            float v649_data = ir5[0];
            ir5[0] = (v649_data + (v645_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v655_data = ir5[1];
            ir5[1] = (v655_data + (v645_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v661_data = ir5[2];
            ir5[2] = (v661_data + (v645_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v667_data = ir5[3];
            ir5[3] = (v667_data + (v645_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v673_data = ir5[4];
            ir5[4] = (v673_data + (v645_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v679_data = ir5[5];
            ir5[5] = (v679_data + (v645_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v685_data = ir5[6];
            ir5[6] = (v685_data + (v645_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v691_data = ir5[7];
            ir5[7] = (v691_data + (v645_data * (sycl::select_from_group(item.get_sub_group(), v496_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v693_data = r3[5];
            float v697_data = ir5[0];
            ir5[0] = (v697_data + (v693_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v703_data = ir5[1];
            ir5[1] = (v703_data + (v693_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v709_data = ir5[2];
            ir5[2] = (v709_data + (v693_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v715_data = ir5[3];
            ir5[3] = (v715_data + (v693_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v721_data = ir5[4];
            ir5[4] = (v721_data + (v693_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v727_data = ir5[5];
            ir5[5] = (v727_data + (v693_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v733_data = ir5[6];
            ir5[6] = (v733_data + (v693_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v739_data = ir5[7];
            ir5[7] = (v739_data + (v693_data * (sycl::select_from_group(item.get_sub_group(), v496_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v741_data = r3[6];
            float v745_data = ir5[0];
            ir5[0] = (v745_data + (v741_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v751_data = ir5[1];
            ir5[1] = (v751_data + (v741_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v757_data = ir5[2];
            ir5[2] = (v757_data + (v741_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v763_data = ir5[3];
            ir5[3] = (v763_data + (v741_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v769_data = ir5[4];
            ir5[4] = (v769_data + (v741_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v775_data = ir5[5];
            ir5[5] = (v775_data + (v741_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v781_data = ir5[6];
            ir5[6] = (v781_data + (v741_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v787_data = ir5[7];
            ir5[7] = (v787_data + (v741_data * (sycl::select_from_group(item.get_sub_group(), v496_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v789_data = r3[7];
            float v793_data = ir5[0];
            ir5[0] = (v793_data + (v789_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v799_data = ir5[1];
            ir5[1] = (v799_data + (v789_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v805_data = ir5[2];
            ir5[2] = (v805_data + (v789_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v811_data = ir5[3];
            ir5[3] = (v811_data + (v789_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v817_data = ir5[4];
            ir5[4] = (v817_data + (v789_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v823_data = ir5[5];
            ir5[5] = (v823_data + (v789_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v829_data = ir5[6];
            ir5[6] = (v829_data + (v789_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v835_data = ir5[7];
            ir5[7] = (v835_data + (v789_data * (sycl::select_from_group(item.get_sub_group(), v496_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r5 = ir5 + r2
            #pragma unroll
            for (int32_t v837_n0 = 0; v837_n0 < 1; ++v837_n0) {
              #pragma unroll
              for (int32_t v838_n1 = 0; v838_n1 < 8; ++v838_n1) {
                int32_t v839_a = v837_n0 + v838_n1;
                float v840_data = ir5[v839_a];
                float v841_data = r2[v839_a];
                r5[v839_a] = (v841_data + v840_data);
              }
            }
            // glb_m4 = abs(r5)
            #pragma unroll
            for (int32_t v843_k0 = 0; v843_k0 < 1; ++v843_k0) {
              #pragma unroll
              for (int32_t v844_k1 = 0; v844_k1 < 8; ++v844_k1) {
                float v846_data = r5[(v843_k0 + v844_k1)];
                float v847_e = sycl::fabs(v846_data);
                if (batchIdActive0) {
                  glb_m4[((v30_lead + (v843_k0 * 8)) + (v844_k1 * 8))] = v847_e;
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


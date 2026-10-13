// === base name ===
kernel_1a02079ac67d0239

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1a02079ac67d0239 = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1a02079ac67d0239(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1a02079ac67d0239(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1a02079ac67d0239(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_1a02079ac67d0239(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1a02079ac67d0239(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1a02079ac67d0239(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1a02079ac67d0239(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
        // operations:
        //   TMP = abs(A)
        //   m1[i,j] = t0[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          size_t v8_batchIdLane0 = item.get_local_id(1) % 2;
          size_t v11_batchIdGroup0First = ((item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) < numElements0) ? (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) : 0u;
          int32_t v41_lead = item.get_local_id(2) % 8;
          float r1[8]{};
          size_t v451_row = v11_batchIdGroup0First + v8_batchIdLane0;
          const bool peel_batchIdActive0 = v451_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v451_row]));
          const float *const __restrict__ peel_glb_m2 = &m2[(peel_batchIdActive0 ? v451_row : v11_batchIdGroup0First) * 64 + 0 + m2_extraOffset];
          // r1 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v459_i0 = 0; v459_i0 < 1; ++v459_i0) {
            int32_t v461_lead = v41_lead + (v459_i0 * 8);
            #pragma unroll
            for (int32_t v462_i1 = 0; v462_i1 < 8; ++v462_i1) {
              float v465_data = peel_glb_m2[(v461_lead + (v462_i1 * 8))];
              r1[(v459_i0 + v462_i1)] = v465_data;
            }
          }
          for (size_t v12_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v448_groupNextAhead = v12_batchIdGroup0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v450_groupNext = (v448_groupNextAhead < numElements0) ? v448_groupNextAhead : v12_batchIdGroup0;
            size_t v455_row = v450_groupNext + v8_batchIdLane0;
            const bool wrap_batchIdActive0 = v455_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v455_row]));
            const float *const __restrict__ wrap_glb_m2 = &m2[(wrap_batchIdActive0 ? v455_row : v450_groupNext) * 64 + 0 + m2_extraOffset];
            size_t v13_row = v12_batchIdGroup0 + v8_batchIdLane0;
            const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
            size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
            size_t v16_ahead1 = v15_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v15_batchId0 * 64 + 0 + m0_extraOffset];
            float *const __restrict__ glb_m1 = &m1[v15_batchId0 * 64 + 0 + m1_extraOffset];
            float r0[8]{};
            // r0 = abs(glb_m0)
            #pragma unroll
            for (int32_t v29_k0 = 0; v29_k0 < 1; ++v29_k0) {
              int32_t v32_lead = v41_lead + (v29_k0 * 8);
              #pragma unroll
              for (int32_t v30_k1 = 0; v30_k1 < 8; ++v30_k1) {
                float v35_data = glb_m0[(v32_lead + (v30_k1 * 8))];
                r0[(v29_k0 + v30_k1)] = (sycl::fabs(v35_data));
              }
            }
            float r2[8]{};
            // ir2 = +(r0 * r1)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir2[8]{};
            float v52_data = r0[0];
            float v53_data = r1[0];
            float v56_data = ir2[0];
            ir2[0] = (v56_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v59_data = r1[1];
            float v62_data = ir2[1];
            ir2[1] = (v62_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v65_data = r1[2];
            float v68_data = ir2[2];
            ir2[2] = (v68_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v71_data = r1[3];
            float v74_data = ir2[3];
            ir2[3] = (v74_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v77_data = r1[4];
            float v80_data = ir2[4];
            ir2[4] = (v80_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v83_data = r1[5];
            float v86_data = ir2[5];
            ir2[5] = (v86_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v89_data = r1[6];
            float v92_data = ir2[6];
            ir2[6] = (v92_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v95_data = r1[7];
            // r1 = load{g>r}(wrap_glb_m2);
            #pragma unroll
            for (int32_t v467_i0 = 0; v467_i0 < 1; ++v467_i0) {
              int32_t v469_lead = v41_lead + (v467_i0 * 8);
              #pragma unroll
              for (int32_t v470_i1 = 0; v470_i1 < 8; ++v470_i1) {
                float v473_data = wrap_glb_m2[(v469_lead + (v470_i1 * 8))];
                r1[(v467_i0 + v470_i1)] = v473_data;
              }
            }
            float v98_data = ir2[7];
            ir2[7] = (v98_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v100_data = r0[1];
            float v104_data = ir2[0];
            ir2[0] = (v104_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v110_data = ir2[1];
            ir2[1] = (v110_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v116_data = ir2[2];
            ir2[2] = (v116_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v122_data = ir2[3];
            ir2[3] = (v122_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v128_data = ir2[4];
            ir2[4] = (v128_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v134_data = ir2[5];
            ir2[5] = (v134_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v140_data = ir2[6];
            ir2[6] = (v140_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v146_data = ir2[7];
            ir2[7] = (v146_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v148_data = r0[2];
            float v152_data = ir2[0];
            ir2[0] = (v152_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v158_data = ir2[1];
            ir2[1] = (v158_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v164_data = ir2[2];
            ir2[2] = (v164_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v170_data = ir2[3];
            ir2[3] = (v170_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v176_data = ir2[4];
            ir2[4] = (v176_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v182_data = ir2[5];
            ir2[5] = (v182_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v188_data = ir2[6];
            ir2[6] = (v188_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v194_data = ir2[7];
            ir2[7] = (v194_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v196_data = r0[3];
            float v200_data = ir2[0];
            ir2[0] = (v200_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v206_data = ir2[1];
            ir2[1] = (v206_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v212_data = ir2[2];
            ir2[2] = (v212_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v218_data = ir2[3];
            ir2[3] = (v218_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v224_data = ir2[4];
            ir2[4] = (v224_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v230_data = ir2[5];
            ir2[5] = (v230_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v236_data = ir2[6];
            ir2[6] = (v236_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v242_data = ir2[7];
            ir2[7] = (v242_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v244_data = r0[4];
            float v248_data = ir2[0];
            ir2[0] = (v248_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v254_data = ir2[1];
            ir2[1] = (v254_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v260_data = ir2[2];
            ir2[2] = (v260_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v266_data = ir2[3];
            ir2[3] = (v266_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v272_data = ir2[4];
            ir2[4] = (v272_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v278_data = ir2[5];
            ir2[5] = (v278_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v284_data = ir2[6];
            ir2[6] = (v284_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v290_data = ir2[7];
            ir2[7] = (v290_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v292_data = r0[5];
            float v296_data = ir2[0];
            ir2[0] = (v296_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v302_data = ir2[1];
            ir2[1] = (v302_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v308_data = ir2[2];
            ir2[2] = (v308_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v314_data = ir2[3];
            ir2[3] = (v314_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v320_data = ir2[4];
            ir2[4] = (v320_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v326_data = ir2[5];
            ir2[5] = (v326_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v332_data = ir2[6];
            ir2[6] = (v332_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v338_data = ir2[7];
            ir2[7] = (v338_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v340_data = r0[6];
            float v344_data = ir2[0];
            ir2[0] = (v344_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v350_data = ir2[1];
            ir2[1] = (v350_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v356_data = ir2[2];
            ir2[2] = (v356_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v362_data = ir2[3];
            ir2[3] = (v362_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v368_data = ir2[4];
            ir2[4] = (v368_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v374_data = ir2[5];
            ir2[5] = (v374_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v380_data = ir2[6];
            ir2[6] = (v380_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v386_data = ir2[7];
            ir2[7] = (v386_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v388_data = r0[7];
            float v392_data = ir2[0];
            ir2[0] = (v392_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v398_data = ir2[1];
            ir2[1] = (v398_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v404_data = ir2[2];
            ir2[2] = (v404_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v410_data = ir2[3];
            ir2[3] = (v410_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v416_data = ir2[4];
            ir2[4] = (v416_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v422_data = ir2[5];
            ir2[5] = (v422_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v428_data = ir2[6];
            ir2[6] = (v428_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v434_data = ir2[7];
            ir2[7] = (v434_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r2 = ir2
            #pragma unroll
            for (int32_t v436_n0 = 0; v436_n0 < 1; ++v436_n0) {
              #pragma unroll
              for (int32_t v437_n1 = 0; v437_n1 < 8; ++v437_n1) {
                int32_t v438_a = v436_n0 + v437_n1;
                float v439_data = ir2[v438_a];
                r2[v438_a] = v439_data;
              }
            }
            // glb_m1 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v440_i0 = 0; v440_i0 < 1; ++v440_i0) {
              #pragma unroll
              for (int32_t v441_i1 = 0; v441_i1 < 8; ++v441_i1) {
                float v443_data = r2[(v440_i0 + v441_i1)];
                if (batchIdActive0) {
                  glb_m1[((v41_lead + (v440_i0 * 8)) + (v441_i1 * 8))] = v443_data;
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


// === base name ===
kernel_2514a43511dfc46c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2514a43511dfc46c = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2514a43511dfc46c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2514a43511dfc46c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2514a43511dfc46c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_2514a43511dfc46c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2514a43511dfc46c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2514a43511dfc46c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2514a43511dfc46c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 12×7(12×7) {0..12}×{0..7} strided
        //   m1 12×20(12×20) {0..12}×{0..20} indexed(stride=runtime)
        //   m2 20×7(20×7) {0..20}×{0..7} indexed
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,7]],"name":"m0","ordered":false,"parts":1,"shape":[12,7],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"indexed","alias":"B","bbox":[[0,0],[20,7]],"name":"m2","ordered":false,"parts":1,"shape":[20,7],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,7]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,7]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"indexed","bbox":[[0,0],[20,7]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,7]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          float r0[20]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v904_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v904_flagSecondAhead < numElements0) ? v904_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const float *const __restrict__ peel_glb_m1 = &m1[static_cast<size_t>(elementIndices[v3_batchId1]) * m1_stride + 0 + m1_extraOffset];
          int32_t v914_lead = item.get_local_id(2) % 16;
          bool v915_g = v914_lead < 12;
          bool v932_g = allowed_peel && v915_g;
          // r0 = load{g>r}(peel_glb_m1);
          #pragma unroll
          for (int32_t v924_i1 = 0; v924_i1 < 20; ++v924_i1) {
            float v927_data_pre = peel_glb_m1[v932_g ? ((v914_lead + (v924_i1 * 12))) : (0)];
            float v927_data = v932_g ? (v927_data_pre) : (0.0f);
            r0[v924_i1] = v927_data;
          }
          uint32_t v908_flagWord = flagWordFirst;
          uint32_t v909_flagWordNext = flagWordSecond;
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const float *const __restrict__ wrap_glb_m1 = &m1[static_cast<size_t>(elementIndices[v10_batchId1]) * m1_stride + 0 + m1_extraOffset];
            size_t v937_flagAheadAhead = v10_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v937_flagAheadAhead < numElements0) ? v937_flagAheadAhead : v10_batchId1)];
            const bool allowed = static_cast<bool>(v908_flagWord);
            const bool allowed_next = static_cast<bool>(v909_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 84 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[static_cast<size_t>(elementIndices[v7_batchId0]) * 140 + 0 + m2_extraOffset];
              float r1[14]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
                int32_t v34_lead = v914_lead + (v31_i0 * 16);
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 7; ++v32_i1) {
                  float v37_data = glb_m2[(v34_lead + (v32_i1 * 20))];
                  r1[(v31_i0 + (v32_i1 * 2))] = v37_data;
                }
              }
              int32_t v42_lead = v914_lead + 16_i32;
              bool v45_g = v914_lead < 4;
              #pragma unroll
              for (int32_t v40_i1 = 0; v40_i1 < 7; ++v40_i1) {
                float v46_data_pre = glb_m2[v45_g ? ((v42_lead + (v40_i1 * 20))) : (0)];
                float v46_data = v45_g ? (v46_data_pre) : (0.0f);
                r1[(1 + (v40_i1 * 2))] = v46_data;
              }
              float r2[7]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 7)] [(0, 20)]
              float ir2[7]{};
              float v51_data = r0[0];
              float v52_data = r1[0];
              float v53_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v58_data = r1[2];
              float v59_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v64_data = r1[4];
              float v65_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v70_data = r1[6];
              float v71_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v76_data = r1[8];
              float v77_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v82_data = r1[10];
              float v83_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v88_data = r1[12];
              float v89_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v93_data = r0[1];
              float v95_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v101_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v107_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v113_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v119_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v125_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v131_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v135_data = r0[2];
              float v137_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v143_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v149_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v155_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v161_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v167_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v173_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v177_data = r0[3];
              float v179_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v185_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v191_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v197_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v203_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v209_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v215_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v219_data = r0[4];
              float v221_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v227_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v233_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v239_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v245_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v251_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v257_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v261_data = r0[5];
              float v263_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v269_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v275_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v281_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v287_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v293_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v299_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v303_data = r0[6];
              float v305_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v311_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v317_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v323_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v329_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v335_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v341_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v345_data = r0[7];
              float v347_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v353_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v359_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v365_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v371_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v377_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v383_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v387_data = r0[8];
              float v389_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v395_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v401_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v407_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v413_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v419_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v425_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v429_data = r0[9];
              float v431_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v437_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v443_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v449_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v455_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v461_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v467_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v471_data = r0[10];
              float v473_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v479_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v485_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v491_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v497_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v503_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v509_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v513_data = r0[11];
              float v515_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v521_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v527_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v533_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v539_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v545_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v551_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v555_data = r0[12];
              float v557_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v563_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v569_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v575_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v581_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v587_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v593_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v597_data = r0[13];
              float v599_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v605_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v611_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v617_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v623_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v629_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v635_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v639_data = r0[14];
              float v641_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v647_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v653_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v659_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v665_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v671_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v677_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v681_data = r0[15];
              float v683_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v689_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v695_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v701_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v707_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v713_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v719_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v723_data = r0[16];
              float v724_data = r1[1];
              float v725_bc = sycl::select_from_group(item.get_sub_group(), v724_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v730_data = r1[3];
              float v731_bc = sycl::select_from_group(item.get_sub_group(), v730_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v736_data = r1[5];
              float v737_bc = sycl::select_from_group(item.get_sub_group(), v736_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v742_data = r1[7];
              float v743_bc = sycl::select_from_group(item.get_sub_group(), v742_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v748_data = r1[9];
              float v749_bc = sycl::select_from_group(item.get_sub_group(), v748_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v754_data = r1[11];
              float v755_bc = sycl::select_from_group(item.get_sub_group(), v754_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v760_data = r1[13];
              float v761_bc = sycl::select_from_group(item.get_sub_group(), v760_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v765_data = r0[17];
              float v767_bc = sycl::select_from_group(item.get_sub_group(), v724_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v773_bc = sycl::select_from_group(item.get_sub_group(), v730_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v779_bc = sycl::select_from_group(item.get_sub_group(), v736_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v785_bc = sycl::select_from_group(item.get_sub_group(), v742_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v791_bc = sycl::select_from_group(item.get_sub_group(), v748_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v797_bc = sycl::select_from_group(item.get_sub_group(), v754_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v803_bc = sycl::select_from_group(item.get_sub_group(), v760_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v807_data = r0[18];
              float v809_bc = sycl::select_from_group(item.get_sub_group(), v724_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v812_p = ((((((((((((((((((v51_data * v53_bc) + (v93_data * v95_bc)) + (v135_data * v137_bc)) + (v177_data * v179_bc)) + (v219_data * v221_bc)) + (v261_data * v263_bc)) + (v303_data * v305_bc)) + (v345_data * v347_bc)) + (v387_data * v389_bc)) + (v429_data * v431_bc)) + (v471_data * v473_bc)) + (v513_data * v515_bc)) + (v555_data * v557_bc)) + (v597_data * v599_bc)) + (v639_data * v641_bc)) + (v681_data * v683_bc)) + (v723_data * v725_bc)) + (v765_data * v767_bc)) + (v807_data * v809_bc);
              float v815_bc = sycl::select_from_group(item.get_sub_group(), v730_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v818_p = ((((((((((((((((((v51_data * v59_bc) + (v93_data * v101_bc)) + (v135_data * v143_bc)) + (v177_data * v185_bc)) + (v219_data * v227_bc)) + (v261_data * v269_bc)) + (v303_data * v311_bc)) + (v345_data * v353_bc)) + (v387_data * v395_bc)) + (v429_data * v437_bc)) + (v471_data * v479_bc)) + (v513_data * v521_bc)) + (v555_data * v563_bc)) + (v597_data * v605_bc)) + (v639_data * v647_bc)) + (v681_data * v689_bc)) + (v723_data * v731_bc)) + (v765_data * v773_bc)) + (v807_data * v815_bc);
              float v821_bc = sycl::select_from_group(item.get_sub_group(), v736_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v824_p = ((((((((((((((((((v51_data * v65_bc) + (v93_data * v107_bc)) + (v135_data * v149_bc)) + (v177_data * v191_bc)) + (v219_data * v233_bc)) + (v261_data * v275_bc)) + (v303_data * v317_bc)) + (v345_data * v359_bc)) + (v387_data * v401_bc)) + (v429_data * v443_bc)) + (v471_data * v485_bc)) + (v513_data * v527_bc)) + (v555_data * v569_bc)) + (v597_data * v611_bc)) + (v639_data * v653_bc)) + (v681_data * v695_bc)) + (v723_data * v737_bc)) + (v765_data * v779_bc)) + (v807_data * v821_bc);
              float v827_bc = sycl::select_from_group(item.get_sub_group(), v742_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v830_p = ((((((((((((((((((v51_data * v71_bc) + (v93_data * v113_bc)) + (v135_data * v155_bc)) + (v177_data * v197_bc)) + (v219_data * v239_bc)) + (v261_data * v281_bc)) + (v303_data * v323_bc)) + (v345_data * v365_bc)) + (v387_data * v407_bc)) + (v429_data * v449_bc)) + (v471_data * v491_bc)) + (v513_data * v533_bc)) + (v555_data * v575_bc)) + (v597_data * v617_bc)) + (v639_data * v659_bc)) + (v681_data * v701_bc)) + (v723_data * v743_bc)) + (v765_data * v785_bc)) + (v807_data * v827_bc);
              float v833_bc = sycl::select_from_group(item.get_sub_group(), v748_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v836_p = ((((((((((((((((((v51_data * v77_bc) + (v93_data * v119_bc)) + (v135_data * v161_bc)) + (v177_data * v203_bc)) + (v219_data * v245_bc)) + (v261_data * v287_bc)) + (v303_data * v329_bc)) + (v345_data * v371_bc)) + (v387_data * v413_bc)) + (v429_data * v455_bc)) + (v471_data * v497_bc)) + (v513_data * v539_bc)) + (v555_data * v581_bc)) + (v597_data * v623_bc)) + (v639_data * v665_bc)) + (v681_data * v707_bc)) + (v723_data * v749_bc)) + (v765_data * v791_bc)) + (v807_data * v833_bc);
              float v839_bc = sycl::select_from_group(item.get_sub_group(), v754_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v842_p = ((((((((((((((((((v51_data * v83_bc) + (v93_data * v125_bc)) + (v135_data * v167_bc)) + (v177_data * v209_bc)) + (v219_data * v251_bc)) + (v261_data * v293_bc)) + (v303_data * v335_bc)) + (v345_data * v377_bc)) + (v387_data * v419_bc)) + (v429_data * v461_bc)) + (v471_data * v503_bc)) + (v513_data * v545_bc)) + (v555_data * v587_bc)) + (v597_data * v629_bc)) + (v639_data * v671_bc)) + (v681_data * v713_bc)) + (v723_data * v755_bc)) + (v765_data * v797_bc)) + (v807_data * v839_bc);
              float v845_bc = sycl::select_from_group(item.get_sub_group(), v760_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v848_p = ((((((((((((((((((v51_data * v89_bc) + (v93_data * v131_bc)) + (v135_data * v173_bc)) + (v177_data * v215_bc)) + (v219_data * v257_bc)) + (v261_data * v299_bc)) + (v303_data * v341_bc)) + (v345_data * v383_bc)) + (v387_data * v425_bc)) + (v429_data * v467_bc)) + (v471_data * v509_bc)) + (v513_data * v551_bc)) + (v555_data * v593_bc)) + (v597_data * v635_bc)) + (v639_data * v677_bc)) + (v681_data * v719_bc)) + (v723_data * v761_bc)) + (v765_data * v803_bc)) + (v807_data * v845_bc);
              float v849_data = r0[19];
              bool v933_g = allowed_next && v915_g;
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v928_i1 = 0; v928_i1 < 20; ++v928_i1) {
                float v931_data_pre = wrap_glb_m1[v933_g ? ((v914_lead + (v928_i1 * 12))) : (0)];
                float v931_data = v933_g ? (v931_data_pre) : (0.0f);
                r0[v928_i1] = v931_data;
              }
              float v851_bc = sycl::select_from_group(item.get_sub_group(), v724_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[0] = (v812_p + (v849_data * v851_bc));
              float v857_bc = sycl::select_from_group(item.get_sub_group(), v730_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[1] = (v818_p + (v849_data * v857_bc));
              float v863_bc = sycl::select_from_group(item.get_sub_group(), v736_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[2] = (v824_p + (v849_data * v863_bc));
              float v869_bc = sycl::select_from_group(item.get_sub_group(), v742_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[3] = (v830_p + (v849_data * v869_bc));
              float v875_bc = sycl::select_from_group(item.get_sub_group(), v748_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[4] = (v836_p + (v849_data * v875_bc));
              float v881_bc = sycl::select_from_group(item.get_sub_group(), v754_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[5] = (v842_p + (v849_data * v881_bc));
              float v887_bc = sycl::select_from_group(item.get_sub_group(), v760_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[6] = (v848_p + (v849_data * v887_bc));
              // r2 = ir2
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v896_i1 = 0; v896_i1 < 7; ++v896_i1) {
                float v898_data = ir2[v896_i1];
                if (v915_g) {
                  glb_m0[(v914_lead + (v896_i1 * 12))] = v898_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              bool v934_g = allowed_next && v915_g;
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v920_i1 = 0; v920_i1 < 20; ++v920_i1) {
                float v923_data_pre = wrap_glb_m1[v934_g ? ((v914_lead + (v920_i1 * 12))) : (0)];
                float v923_data = v934_g ? (v923_data_pre) : (0.0f);
                r0[v920_i1] = v923_data;
              }
            }
            v908_flagWord = v909_flagWordNext;
            v909_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


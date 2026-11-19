// === base name ===
kernel_fd1551bde4c7b897

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_fd1551bde4c7b897 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_fd1551bde4c7b897(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_fd1551bde4c7b897(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_fd1551bde4c7b897(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_fd1551bde4c7b897(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_fd1551bde4c7b897(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_fd1551bde4c7b897(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_fd1551bde4c7b897(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
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
          size_t v905_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v905_flagSecondAhead < numElements0) ? v905_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
          const float *const __restrict__ peel_glb_m1 = &m1[peel_elementIndex * m1_stride + 0 + m1_extraOffset];
          int32_t v916_lead = item.get_local_id(2) % 16;
          bool v917_g = v916_lead < 12;
          bool v935_g = allowed_peel && v917_g;
          // r0 = load{g>r}(peel_glb_m1);
          #pragma unroll
          for (int32_t v927_i1 = 0; v927_i1 < 20; ++v927_i1) {
            float v930_data_pre = peel_glb_m1[v935_g ? ((v916_lead + (v927_i1 * 12))) : (0)];
            float v930_data = v935_g ? (v930_data_pre) : (0.0f);
            r0[v927_i1] = v930_data;
          }
          uint32_t v909_flagWord = flagWordFirst;
          uint32_t v910_flagWordNext = flagWordSecond;
          size_t v945_elementIndex = peel_elementIndex;
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v10_batchId1]);
            const float *const __restrict__ wrap_glb_m1 = &m1[wrap_elementIndex * m1_stride + 0 + m1_extraOffset];
            size_t v940_flagAheadAhead = v10_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v940_flagAheadAhead < numElements0) ? v940_flagAheadAhead : v10_batchId1)];
            const bool allowed = static_cast<bool>(v909_flagWord);
            const bool allowed_next = static_cast<bool>(v910_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 84 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v945_elementIndex * 140 + 0 + m2_extraOffset];
              float r1[14]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v35_lead = v916_lead + (v32_i0 * 16);
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 7; ++v33_i1) {
                  float v38_data = glb_m2[(v35_lead + (v33_i1 * 20))];
                  r1[(v32_i0 + (v33_i1 * 2))] = v38_data;
                }
              }
              int32_t v43_lead = v916_lead + 16_i32;
              bool v46_g = v916_lead < 4;
              #pragma unroll
              for (int32_t v41_i1 = 0; v41_i1 < 7; ++v41_i1) {
                float v47_data_pre = glb_m2[v46_g ? ((v43_lead + (v41_i1 * 20))) : (0)];
                float v47_data = v46_g ? (v47_data_pre) : (0.0f);
                r1[(1 + (v41_i1 * 2))] = v47_data;
              }
              float r2[7]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 7)] [(0, 20)]
              float ir2[7]{};
              float v52_data = r0[0];
              float v53_data = r1[0];
              float v54_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v59_data = r1[2];
              float v60_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v65_data = r1[4];
              float v66_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v71_data = r1[6];
              float v72_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v77_data = r1[8];
              float v78_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v83_data = r1[10];
              float v84_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v89_data = r1[12];
              float v90_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v94_data = r0[1];
              float v96_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v102_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v108_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v114_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v120_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v126_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v132_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v136_data = r0[2];
              float v138_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v144_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v150_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v156_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v162_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v168_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v174_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v178_data = r0[3];
              float v180_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v186_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v192_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v198_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v204_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v210_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v216_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v220_data = r0[4];
              float v222_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v228_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v234_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v240_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v246_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v252_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v258_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v262_data = r0[5];
              float v264_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v270_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v276_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v282_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v288_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v294_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v300_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v304_data = r0[6];
              float v306_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v312_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v318_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v324_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v330_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v336_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v342_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v346_data = r0[7];
              float v348_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v354_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v360_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v366_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v372_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v378_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v384_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v388_data = r0[8];
              float v390_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v396_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v402_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v408_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v414_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v420_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v426_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v430_data = r0[9];
              float v432_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v438_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v444_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v450_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v456_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v462_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v468_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v472_data = r0[10];
              float v474_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v480_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v486_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v492_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v498_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v504_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v510_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v514_data = r0[11];
              float v516_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v522_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v528_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v534_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v540_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v546_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v552_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v556_data = r0[12];
              float v558_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v564_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v570_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v576_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v582_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v588_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v594_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v598_data = r0[13];
              float v600_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v606_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v612_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v618_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v624_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v630_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v636_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v640_data = r0[14];
              float v642_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v648_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v654_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v660_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v666_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v672_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v678_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v682_data = r0[15];
              float v684_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v690_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v696_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v702_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v708_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v714_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v720_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v724_data = r0[16];
              float v725_data = r1[1];
              float v726_bc = sycl::select_from_group(item.get_sub_group(), v725_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v731_data = r1[3];
              float v732_bc = sycl::select_from_group(item.get_sub_group(), v731_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v737_data = r1[5];
              float v738_bc = sycl::select_from_group(item.get_sub_group(), v737_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v743_data = r1[7];
              float v744_bc = sycl::select_from_group(item.get_sub_group(), v743_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v749_data = r1[9];
              float v750_bc = sycl::select_from_group(item.get_sub_group(), v749_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v755_data = r1[11];
              float v756_bc = sycl::select_from_group(item.get_sub_group(), v755_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v761_data = r1[13];
              float v762_bc = sycl::select_from_group(item.get_sub_group(), v761_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v766_data = r0[17];
              float v768_bc = sycl::select_from_group(item.get_sub_group(), v725_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v774_bc = sycl::select_from_group(item.get_sub_group(), v731_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v780_bc = sycl::select_from_group(item.get_sub_group(), v737_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v786_bc = sycl::select_from_group(item.get_sub_group(), v743_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v792_bc = sycl::select_from_group(item.get_sub_group(), v749_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v798_bc = sycl::select_from_group(item.get_sub_group(), v755_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v804_bc = sycl::select_from_group(item.get_sub_group(), v761_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v808_data = r0[18];
              float v810_bc = sycl::select_from_group(item.get_sub_group(), v725_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v813_p = ((((((((((((((((((v52_data * v54_bc) + (v94_data * v96_bc)) + (v136_data * v138_bc)) + (v178_data * v180_bc)) + (v220_data * v222_bc)) + (v262_data * v264_bc)) + (v304_data * v306_bc)) + (v346_data * v348_bc)) + (v388_data * v390_bc)) + (v430_data * v432_bc)) + (v472_data * v474_bc)) + (v514_data * v516_bc)) + (v556_data * v558_bc)) + (v598_data * v600_bc)) + (v640_data * v642_bc)) + (v682_data * v684_bc)) + (v724_data * v726_bc)) + (v766_data * v768_bc)) + (v808_data * v810_bc);
              float v816_bc = sycl::select_from_group(item.get_sub_group(), v731_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v819_p = ((((((((((((((((((v52_data * v60_bc) + (v94_data * v102_bc)) + (v136_data * v144_bc)) + (v178_data * v186_bc)) + (v220_data * v228_bc)) + (v262_data * v270_bc)) + (v304_data * v312_bc)) + (v346_data * v354_bc)) + (v388_data * v396_bc)) + (v430_data * v438_bc)) + (v472_data * v480_bc)) + (v514_data * v522_bc)) + (v556_data * v564_bc)) + (v598_data * v606_bc)) + (v640_data * v648_bc)) + (v682_data * v690_bc)) + (v724_data * v732_bc)) + (v766_data * v774_bc)) + (v808_data * v816_bc);
              float v822_bc = sycl::select_from_group(item.get_sub_group(), v737_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v825_p = ((((((((((((((((((v52_data * v66_bc) + (v94_data * v108_bc)) + (v136_data * v150_bc)) + (v178_data * v192_bc)) + (v220_data * v234_bc)) + (v262_data * v276_bc)) + (v304_data * v318_bc)) + (v346_data * v360_bc)) + (v388_data * v402_bc)) + (v430_data * v444_bc)) + (v472_data * v486_bc)) + (v514_data * v528_bc)) + (v556_data * v570_bc)) + (v598_data * v612_bc)) + (v640_data * v654_bc)) + (v682_data * v696_bc)) + (v724_data * v738_bc)) + (v766_data * v780_bc)) + (v808_data * v822_bc);
              float v828_bc = sycl::select_from_group(item.get_sub_group(), v743_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v831_p = ((((((((((((((((((v52_data * v72_bc) + (v94_data * v114_bc)) + (v136_data * v156_bc)) + (v178_data * v198_bc)) + (v220_data * v240_bc)) + (v262_data * v282_bc)) + (v304_data * v324_bc)) + (v346_data * v366_bc)) + (v388_data * v408_bc)) + (v430_data * v450_bc)) + (v472_data * v492_bc)) + (v514_data * v534_bc)) + (v556_data * v576_bc)) + (v598_data * v618_bc)) + (v640_data * v660_bc)) + (v682_data * v702_bc)) + (v724_data * v744_bc)) + (v766_data * v786_bc)) + (v808_data * v828_bc);
              float v834_bc = sycl::select_from_group(item.get_sub_group(), v749_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v837_p = ((((((((((((((((((v52_data * v78_bc) + (v94_data * v120_bc)) + (v136_data * v162_bc)) + (v178_data * v204_bc)) + (v220_data * v246_bc)) + (v262_data * v288_bc)) + (v304_data * v330_bc)) + (v346_data * v372_bc)) + (v388_data * v414_bc)) + (v430_data * v456_bc)) + (v472_data * v498_bc)) + (v514_data * v540_bc)) + (v556_data * v582_bc)) + (v598_data * v624_bc)) + (v640_data * v666_bc)) + (v682_data * v708_bc)) + (v724_data * v750_bc)) + (v766_data * v792_bc)) + (v808_data * v834_bc);
              float v840_bc = sycl::select_from_group(item.get_sub_group(), v755_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v843_p = ((((((((((((((((((v52_data * v84_bc) + (v94_data * v126_bc)) + (v136_data * v168_bc)) + (v178_data * v210_bc)) + (v220_data * v252_bc)) + (v262_data * v294_bc)) + (v304_data * v336_bc)) + (v346_data * v378_bc)) + (v388_data * v420_bc)) + (v430_data * v462_bc)) + (v472_data * v504_bc)) + (v514_data * v546_bc)) + (v556_data * v588_bc)) + (v598_data * v630_bc)) + (v640_data * v672_bc)) + (v682_data * v714_bc)) + (v724_data * v756_bc)) + (v766_data * v798_bc)) + (v808_data * v840_bc);
              float v846_bc = sycl::select_from_group(item.get_sub_group(), v761_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v849_p = ((((((((((((((((((v52_data * v90_bc) + (v94_data * v132_bc)) + (v136_data * v174_bc)) + (v178_data * v216_bc)) + (v220_data * v258_bc)) + (v262_data * v300_bc)) + (v304_data * v342_bc)) + (v346_data * v384_bc)) + (v388_data * v426_bc)) + (v430_data * v468_bc)) + (v472_data * v510_bc)) + (v514_data * v552_bc)) + (v556_data * v594_bc)) + (v598_data * v636_bc)) + (v640_data * v678_bc)) + (v682_data * v720_bc)) + (v724_data * v762_bc)) + (v766_data * v804_bc)) + (v808_data * v846_bc);
              float v850_data = r0[19];
              bool v936_g = allowed_next && v917_g;
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v931_i1 = 0; v931_i1 < 20; ++v931_i1) {
                float v934_data_pre = wrap_glb_m1[v936_g ? ((v916_lead + (v931_i1 * 12))) : (0)];
                float v934_data = v936_g ? (v934_data_pre) : (0.0f);
                r0[v931_i1] = v934_data;
              }
              float v852_bc = sycl::select_from_group(item.get_sub_group(), v725_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[0] = (v813_p + (v850_data * v852_bc));
              float v858_bc = sycl::select_from_group(item.get_sub_group(), v731_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[1] = (v819_p + (v850_data * v858_bc));
              float v864_bc = sycl::select_from_group(item.get_sub_group(), v737_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[2] = (v825_p + (v850_data * v864_bc));
              float v870_bc = sycl::select_from_group(item.get_sub_group(), v743_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[3] = (v831_p + (v850_data * v870_bc));
              float v876_bc = sycl::select_from_group(item.get_sub_group(), v749_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[4] = (v837_p + (v850_data * v876_bc));
              float v882_bc = sycl::select_from_group(item.get_sub_group(), v755_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[5] = (v843_p + (v850_data * v882_bc));
              float v888_bc = sycl::select_from_group(item.get_sub_group(), v761_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir2[6] = (v849_p + (v850_data * v888_bc));
              // r2 = ir2
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v897_i1 = 0; v897_i1 < 7; ++v897_i1) {
                float v899_data = ir2[v897_i1];
                if (v917_g) {
                  glb_m0[(v916_lead + (v897_i1 * 12))] = v899_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              bool v937_g = allowed_next && v917_g;
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v923_i1 = 0; v923_i1 < 20; ++v923_i1) {
                float v926_data_pre = wrap_glb_m1[v937_g ? ((v916_lead + (v923_i1 * 12))) : (0)];
                float v926_data = v937_g ? (v926_data_pre) : (0.0f);
                r0[v923_i1] = v926_data;
              }
            }
            v909_flagWord = v910_flagWordNext;
            v910_flagWordNext = flagWordAhead;
            v945_elementIndex = wrap_elementIndex;
          }
        }
      });
    }
  });
}


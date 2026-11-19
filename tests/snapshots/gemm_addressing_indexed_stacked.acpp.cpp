// === base name ===
kernel_41c92d343b8ce9b9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_41c92d343b8ce9b9 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_41c92d343b8ce9b9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_41c92d343b8ce9b9(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_41c92d343b8ce9b9(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (static_cast<size_t>((numElements0 + 1) / 2) + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_41c92d343b8ce9b9(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_41c92d343b8ce9b9(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_41c92d343b8ce9b9(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_41c92d343b8ce9b9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: stack_elements=2
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×8×2(16×8×2) {0..16}×{0..8}×{0..2} indexed(stride=136, offset=4)
        //   m1 16×16(16×16) {0..16}×{0..16} none
        //   m2 16×8×2(16×8×2) {0..16}×{0..8}×{0..2} indexed_pointer(slots=3, slot=2)
        // operations:
        //   m0[i,j,l] = m1[i,k] × m2[k,j,l]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"indexed(stride=136, offset=4)","alias":"C","bbox":[[0,0,0],[16,8,2]],"name":"m0","ordered":false,"parts":1,"shape":[16,8,2],"stack":2,"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed_pointer(slots=3, slot=2)","alias":"B","bbox":[[0,0,0],[16,8,2]],"name":"m2","ordered":false,"parts":1,"shape":[16,8,2],"stack":2,"variant":false}],"operations":[{"add":false,"dest":{"addressing":"indexed(stride=136, offset=4)","bbox":[[0,0,0],[16,8,2]],"is_tmp":false,"name":"m0","offset":[0,0,0],"shape":[16,8,2]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"indexed_pointer(slots=3, slot=2)","bbox":[[0,0,0],[16,8,2]],"is_tmp":false,"name":"m2","offset":[0,0,0],"shape":[16,8,2]}],"permute":[[0,1],[0,1,2]],"target":[[0,-1],[-1,1,2]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < ((numElements0 + 1) / 2)) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          float r0[16]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : (flags0[2 * v3_batchId1] || (2 * v3_batchId1 + 1 < numElements0 && flags0[2 * v3_batchId1 + 1]));
          size_t v1616_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v1618_flagSecond = (v1616_flagSecondAhead < ((numElements0 + 1) / 2)) ? v1616_flagSecondAhead : v3_batchId1;
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : (flags0[2 * v1618_flagSecond] || (2 * v1618_flagSecond + 1 < numElements0 && flags0[2 * v1618_flagSecond + 1]));
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const bool peel_batchIdStack0_0 = 2 * v3_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 0]));
          const bool peel_batchIdStack0_1 = 2 * v3_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 1]));
          const size_t peel_elementIndex_s0 = static_cast<size_t>(elementIndices[(peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]);
          const size_t peel_elementIndex_s1 = static_cast<size_t>(elementIndices[(peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]);
          const float *const __restrict__ peel_glb_m2_s0 = &m2[peel_elementIndex_s0 * 3 + 2][0 + m2_extraOffset];
          const float *const __restrict__ peel_glb_m2_s1 = &m2[peel_elementIndex_s1 * 3 + 2][0 + m2_extraOffset];
          int32_t v1631_lead = item.get_local_id(2) % 16;
          if (allowed_peel) {
            // r0 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v1652_i0 = 0; v1652_i0 < 1; ++v1652_i0) {
              int32_t v1654_lead = v1631_lead + (v1652_i0 * 16);
              #pragma unroll
              for (int32_t v1655_i1 = 0; v1655_i1 < 8; ++v1655_i1) {
                int32_t v1657_a = v1654_lead + (v1655_i1 * 16);
                int32_t v1658_a = v1652_i0 + v1655_i1;
                #pragma unroll
                for (int32_t v1659_i2 = 0; v1659_i2 < 2; ++v1659_i2) {
                  const float *const __restrict__ glb_m2_se0 = v1659_i2 == 0 ? peel_glb_m2_s0 : peel_glb_m2_s1;
                  float v1661_data = glb_m2_se0[v1657_a];
                  r0[(v1658_a + (v1659_i2 * 8))] = v1661_data;
                }
              }
            }
          }
          uint32_t v1620_flagWord = flagWordFirst;
          uint32_t v1621_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < ((numElements0 + 1) / 2); v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < ((numElements0 + 1) / 2)) ? v9_ahead1 : v8_batchId0;
            const bool wrap_batchIdStack0_0 = 2 * v11_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v11_batchId1 + 0]));
            const bool wrap_batchIdStack0_1 = 2 * v11_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v11_batchId1 + 1]));
            const size_t wrap_elementIndex_s0 = static_cast<size_t>(elementIndices[(wrap_batchIdStack0_0 ? 2 * v11_batchId1 + 0 : (wrap_batchIdStack0_0 ? 2 * v11_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v11_batchId1 + 1 : 2 * v11_batchId1))]);
            const size_t wrap_elementIndex_s1 = static_cast<size_t>(elementIndices[(wrap_batchIdStack0_1 ? 2 * v11_batchId1 + 1 : (wrap_batchIdStack0_0 ? 2 * v11_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v11_batchId1 + 1 : 2 * v11_batchId1))]);
            const float *const __restrict__ wrap_glb_m2_s0 = &m2[wrap_elementIndex_s0 * 3 + 2][0 + m2_extraOffset];
            const float *const __restrict__ wrap_glb_m2_s1 = &m2[wrap_elementIndex_s1 * 3 + 2][0 + m2_extraOffset];
            size_t v1678_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v1680_flagAhead = (v1678_flagAheadAhead < ((numElements0 + 1) / 2)) ? v1678_flagAheadAhead : v11_batchId1;
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : (flags0[2 * v1680_flagAhead] || (2 * v1680_flagAhead + 1 < numElements0 && flags0[2 * v1680_flagAhead + 1]));
            const bool batchIdStack0_0 = 2 * v8_batchId0 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v8_batchId0 + 0]));
            const bool batchIdStack0_1 = 2 * v8_batchId0 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v8_batchId0 + 1]));
            const bool allowed = static_cast<bool>(v1620_flagWord);
            const bool allowed_next = static_cast<bool>(v1621_flagWordNext);
            if (allowed) {
              const size_t elementIndex_s0 = static_cast<size_t>(elementIndices[(batchIdStack0_0 ? 2 * v8_batchId0 + 0 : (batchIdStack0_0 ? 2 * v8_batchId0 + 0 : batchIdStack0_1 ? 2 * v8_batchId0 + 1 : 2 * v8_batchId0))]);
              const size_t elementIndex_s1 = static_cast<size_t>(elementIndices[(batchIdStack0_1 ? 2 * v8_batchId0 + 1 : (batchIdStack0_0 ? 2 * v8_batchId0 + 0 : batchIdStack0_1 ? 2 * v8_batchId0 + 1 : 2 * v8_batchId0))]);
              float *const glb_m0_s0 = &m0[elementIndex_s0 * 136 + 0 + 4 + m0_extraOffset];
              float *const glb_m0_s1 = &m0[elementIndex_s1 * 136 + 0 + 4 + m0_extraOffset];
              float r1[16]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 16), (0, 8), (0, 2)] [(0, 16)]
              float ir1[16]{};
              float v45_data = glb_m1[v1631_lead];
              float v46_data = r0[0];
              float v47_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v52_data = r0[8];
              float v53_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v58_data = r0[1];
              float v59_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v64_data = r0[9];
              float v65_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v70_data = r0[2];
              float v71_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v76_data = r0[10];
              float v77_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v82_data = r0[3];
              float v83_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v88_data = r0[11];
              float v89_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v94_data = r0[4];
              float v95_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v100_data = r0[12];
              float v101_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v106_data = r0[5];
              float v107_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v112_data = r0[13];
              float v113_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v118_data = r0[6];
              float v119_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v124_data = r0[14];
              float v125_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v130_data = r0[7];
              float v131_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v136_data = r0[15];
              float v137_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v142_data = glb_m1[(v1631_lead + 16)];
              float v144_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v150_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v156_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v162_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v168_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v174_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v180_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v186_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v192_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v198_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v204_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v210_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v216_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v222_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v228_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v234_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v239_data = glb_m1[(v1631_lead + 32)];
              float v241_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v247_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v253_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v259_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v265_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v271_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v277_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v283_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v289_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v295_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v301_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v307_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v313_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v319_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v325_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v331_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v336_data = glb_m1[(v1631_lead + 48)];
              float v338_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v344_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v350_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v356_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v362_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v368_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v374_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v380_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v386_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v392_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v398_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v404_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v410_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v416_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v422_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v428_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v433_data = glb_m1[(v1631_lead + 64)];
              float v435_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v441_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v447_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v453_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v459_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v465_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v471_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v477_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v483_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v489_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v495_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v501_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v507_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v513_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v519_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v525_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v530_data = glb_m1[(v1631_lead + 80)];
              float v532_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v538_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v544_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v550_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v556_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v562_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v568_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v574_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v580_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v586_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v592_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v598_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v604_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v610_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v616_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v622_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v627_data = glb_m1[(v1631_lead + 96)];
              float v629_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v635_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v641_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v647_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v653_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v659_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v665_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v671_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v677_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v683_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v689_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v695_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v701_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v707_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v713_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v719_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v724_data = glb_m1[(v1631_lead + 112)];
              float v726_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v732_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v738_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v744_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v750_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v756_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v762_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v768_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v774_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v780_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v786_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v792_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v798_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v804_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v810_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v816_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v821_data = glb_m1[(v1631_lead + 128)];
              float v823_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v829_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v835_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v841_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v847_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v853_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v859_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v865_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v871_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v877_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v883_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v889_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v895_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v901_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v907_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v913_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v918_data = glb_m1[(v1631_lead + 144)];
              float v920_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v926_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v932_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v938_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v944_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v950_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v956_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v962_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v968_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v974_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v980_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v986_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v992_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v998_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v1004_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v1010_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v1015_data = glb_m1[(v1631_lead + 160)];
              float v1017_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1023_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1029_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1035_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1041_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1047_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1053_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1059_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1065_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1071_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1077_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1083_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1089_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1095_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1101_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1107_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1112_data = glb_m1[(v1631_lead + 176)];
              float v1114_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1120_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1126_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1132_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1138_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1144_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1150_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1156_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1162_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1168_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1174_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1180_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1186_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1192_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1198_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1204_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1209_data = glb_m1[(v1631_lead + 192)];
              float v1211_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1217_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1220_p = ((((((((((((v45_data * v53_bc) + (v142_data * v150_bc)) + (v239_data * v247_bc)) + (v336_data * v344_bc)) + (v433_data * v441_bc)) + (v530_data * v538_bc)) + (v627_data * v635_bc)) + (v724_data * v732_bc)) + (v821_data * v829_bc)) + (v918_data * v926_bc)) + (v1015_data * v1023_bc)) + (v1112_data * v1120_bc)) + (v1209_data * v1217_bc);
              float v1223_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1226_p = ((((((((((((v45_data * v59_bc) + (v142_data * v156_bc)) + (v239_data * v253_bc)) + (v336_data * v350_bc)) + (v433_data * v447_bc)) + (v530_data * v544_bc)) + (v627_data * v641_bc)) + (v724_data * v738_bc)) + (v821_data * v835_bc)) + (v918_data * v932_bc)) + (v1015_data * v1029_bc)) + (v1112_data * v1126_bc)) + (v1209_data * v1223_bc);
              float v1229_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1232_p = ((((((((((((v45_data * v65_bc) + (v142_data * v162_bc)) + (v239_data * v259_bc)) + (v336_data * v356_bc)) + (v433_data * v453_bc)) + (v530_data * v550_bc)) + (v627_data * v647_bc)) + (v724_data * v744_bc)) + (v821_data * v841_bc)) + (v918_data * v938_bc)) + (v1015_data * v1035_bc)) + (v1112_data * v1132_bc)) + (v1209_data * v1229_bc);
              float v1235_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1238_p = ((((((((((((v45_data * v71_bc) + (v142_data * v168_bc)) + (v239_data * v265_bc)) + (v336_data * v362_bc)) + (v433_data * v459_bc)) + (v530_data * v556_bc)) + (v627_data * v653_bc)) + (v724_data * v750_bc)) + (v821_data * v847_bc)) + (v918_data * v944_bc)) + (v1015_data * v1041_bc)) + (v1112_data * v1138_bc)) + (v1209_data * v1235_bc);
              float v1241_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1244_p = ((((((((((((v45_data * v77_bc) + (v142_data * v174_bc)) + (v239_data * v271_bc)) + (v336_data * v368_bc)) + (v433_data * v465_bc)) + (v530_data * v562_bc)) + (v627_data * v659_bc)) + (v724_data * v756_bc)) + (v821_data * v853_bc)) + (v918_data * v950_bc)) + (v1015_data * v1047_bc)) + (v1112_data * v1144_bc)) + (v1209_data * v1241_bc);
              float v1247_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1250_p = ((((((((((((v45_data * v83_bc) + (v142_data * v180_bc)) + (v239_data * v277_bc)) + (v336_data * v374_bc)) + (v433_data * v471_bc)) + (v530_data * v568_bc)) + (v627_data * v665_bc)) + (v724_data * v762_bc)) + (v821_data * v859_bc)) + (v918_data * v956_bc)) + (v1015_data * v1053_bc)) + (v1112_data * v1150_bc)) + (v1209_data * v1247_bc);
              float v1253_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1256_p = ((((((((((((v45_data * v89_bc) + (v142_data * v186_bc)) + (v239_data * v283_bc)) + (v336_data * v380_bc)) + (v433_data * v477_bc)) + (v530_data * v574_bc)) + (v627_data * v671_bc)) + (v724_data * v768_bc)) + (v821_data * v865_bc)) + (v918_data * v962_bc)) + (v1015_data * v1059_bc)) + (v1112_data * v1156_bc)) + (v1209_data * v1253_bc);
              float v1259_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1262_p = ((((((((((((v45_data * v95_bc) + (v142_data * v192_bc)) + (v239_data * v289_bc)) + (v336_data * v386_bc)) + (v433_data * v483_bc)) + (v530_data * v580_bc)) + (v627_data * v677_bc)) + (v724_data * v774_bc)) + (v821_data * v871_bc)) + (v918_data * v968_bc)) + (v1015_data * v1065_bc)) + (v1112_data * v1162_bc)) + (v1209_data * v1259_bc);
              float v1265_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1268_p = ((((((((((((v45_data * v101_bc) + (v142_data * v198_bc)) + (v239_data * v295_bc)) + (v336_data * v392_bc)) + (v433_data * v489_bc)) + (v530_data * v586_bc)) + (v627_data * v683_bc)) + (v724_data * v780_bc)) + (v821_data * v877_bc)) + (v918_data * v974_bc)) + (v1015_data * v1071_bc)) + (v1112_data * v1168_bc)) + (v1209_data * v1265_bc);
              float v1271_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1274_p = ((((((((((((v45_data * v107_bc) + (v142_data * v204_bc)) + (v239_data * v301_bc)) + (v336_data * v398_bc)) + (v433_data * v495_bc)) + (v530_data * v592_bc)) + (v627_data * v689_bc)) + (v724_data * v786_bc)) + (v821_data * v883_bc)) + (v918_data * v980_bc)) + (v1015_data * v1077_bc)) + (v1112_data * v1174_bc)) + (v1209_data * v1271_bc);
              float v1277_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1280_p = ((((((((((((v45_data * v113_bc) + (v142_data * v210_bc)) + (v239_data * v307_bc)) + (v336_data * v404_bc)) + (v433_data * v501_bc)) + (v530_data * v598_bc)) + (v627_data * v695_bc)) + (v724_data * v792_bc)) + (v821_data * v889_bc)) + (v918_data * v986_bc)) + (v1015_data * v1083_bc)) + (v1112_data * v1180_bc)) + (v1209_data * v1277_bc);
              float v1283_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1286_p = ((((((((((((v45_data * v119_bc) + (v142_data * v216_bc)) + (v239_data * v313_bc)) + (v336_data * v410_bc)) + (v433_data * v507_bc)) + (v530_data * v604_bc)) + (v627_data * v701_bc)) + (v724_data * v798_bc)) + (v821_data * v895_bc)) + (v918_data * v992_bc)) + (v1015_data * v1089_bc)) + (v1112_data * v1186_bc)) + (v1209_data * v1283_bc);
              float v1289_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1292_p = ((((((((((((v45_data * v125_bc) + (v142_data * v222_bc)) + (v239_data * v319_bc)) + (v336_data * v416_bc)) + (v433_data * v513_bc)) + (v530_data * v610_bc)) + (v627_data * v707_bc)) + (v724_data * v804_bc)) + (v821_data * v901_bc)) + (v918_data * v998_bc)) + (v1015_data * v1095_bc)) + (v1112_data * v1192_bc)) + (v1209_data * v1289_bc);
              float v1295_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1298_p = ((((((((((((v45_data * v131_bc) + (v142_data * v228_bc)) + (v239_data * v325_bc)) + (v336_data * v422_bc)) + (v433_data * v519_bc)) + (v530_data * v616_bc)) + (v627_data * v713_bc)) + (v724_data * v810_bc)) + (v821_data * v907_bc)) + (v918_data * v1004_bc)) + (v1015_data * v1101_bc)) + (v1112_data * v1198_bc)) + (v1209_data * v1295_bc);
              float v1301_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1304_p = ((((((((((((v45_data * v137_bc) + (v142_data * v234_bc)) + (v239_data * v331_bc)) + (v336_data * v428_bc)) + (v433_data * v525_bc)) + (v530_data * v622_bc)) + (v627_data * v719_bc)) + (v724_data * v816_bc)) + (v821_data * v913_bc)) + (v918_data * v1010_bc)) + (v1015_data * v1107_bc)) + (v1112_data * v1204_bc)) + (v1209_data * v1301_bc);
              float v1306_data = glb_m1[(v1631_lead + 208)];
              float v1308_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1311_p = (((((((((((((v45_data * v47_bc) + (v142_data * v144_bc)) + (v239_data * v241_bc)) + (v336_data * v338_bc)) + (v433_data * v435_bc)) + (v530_data * v532_bc)) + (v627_data * v629_bc)) + (v724_data * v726_bc)) + (v821_data * v823_bc)) + (v918_data * v920_bc)) + (v1015_data * v1017_bc)) + (v1112_data * v1114_bc)) + (v1209_data * v1211_bc)) + (v1306_data * v1308_bc);
              float v1314_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              if (allowed_next) {
                // r0 = load{g>r}(glb_m2);
                #pragma unroll
                for (int32_t v1664_i0 = 0; v1664_i0 < 1; ++v1664_i0) {
                  int32_t v1666_lead = v1631_lead + (v1664_i0 * 16);
                  #pragma unroll
                  for (int32_t v1667_i1 = 0; v1667_i1 < 8; ++v1667_i1) {
                    int32_t v1669_a = v1666_lead + (v1667_i1 * 16);
                    int32_t v1670_a = v1664_i0 + v1667_i1;
                    #pragma unroll
                    for (int32_t v1671_i2 = 0; v1671_i2 < 2; ++v1671_i2) {
                      const float *const __restrict__ glb_m2_se0 = v1671_i2 == 0 ? wrap_glb_m2_s0 : wrap_glb_m2_s1;
                      float v1673_data = glb_m2_se0[v1669_a];
                      r0[(v1670_a + (v1671_i2 * 8))] = v1673_data;
                    }
                  }
                }
              }
              float v1320_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1326_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1332_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1338_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1344_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1350_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1356_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1362_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1368_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1374_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1380_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1386_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1392_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1398_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1403_data = glb_m1[(v1631_lead + 224)];
              float v1405_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1411_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1414_p = (v1220_p + (v1306_data * v1314_bc)) + (v1403_data * v1411_bc);
              float v1417_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1420_p = (v1226_p + (v1306_data * v1320_bc)) + (v1403_data * v1417_bc);
              float v1423_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1426_p = (v1232_p + (v1306_data * v1326_bc)) + (v1403_data * v1423_bc);
              float v1429_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1432_p = (v1238_p + (v1306_data * v1332_bc)) + (v1403_data * v1429_bc);
              float v1435_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1438_p = (v1244_p + (v1306_data * v1338_bc)) + (v1403_data * v1435_bc);
              float v1441_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1444_p = (v1250_p + (v1306_data * v1344_bc)) + (v1403_data * v1441_bc);
              float v1447_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1450_p = (v1256_p + (v1306_data * v1350_bc)) + (v1403_data * v1447_bc);
              float v1453_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1456_p = (v1262_p + (v1306_data * v1356_bc)) + (v1403_data * v1453_bc);
              float v1459_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1462_p = (v1268_p + (v1306_data * v1362_bc)) + (v1403_data * v1459_bc);
              float v1465_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1468_p = (v1274_p + (v1306_data * v1368_bc)) + (v1403_data * v1465_bc);
              float v1471_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1474_p = (v1280_p + (v1306_data * v1374_bc)) + (v1403_data * v1471_bc);
              float v1477_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1480_p = (v1286_p + (v1306_data * v1380_bc)) + (v1403_data * v1477_bc);
              float v1483_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1486_p = (v1292_p + (v1306_data * v1386_bc)) + (v1403_data * v1483_bc);
              float v1489_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1492_p = (v1298_p + (v1306_data * v1392_bc)) + (v1403_data * v1489_bc);
              float v1495_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1498_p = (v1304_p + (v1306_data * v1398_bc)) + (v1403_data * v1495_bc);
              float v1500_data = glb_m1[(v1631_lead + 240)];
              float v1502_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[0] = ((v1311_p + (v1403_data * v1405_bc)) + (v1500_data * v1502_bc));
              float v1508_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[8] = (v1414_p + (v1500_data * v1508_bc));
              float v1514_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[1] = (v1420_p + (v1500_data * v1514_bc));
              float v1520_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[9] = (v1426_p + (v1500_data * v1520_bc));
              float v1526_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[2] = (v1432_p + (v1500_data * v1526_bc));
              float v1532_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[10] = (v1438_p + (v1500_data * v1532_bc));
              float v1538_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[3] = (v1444_p + (v1500_data * v1538_bc));
              float v1544_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[11] = (v1450_p + (v1500_data * v1544_bc));
              float v1550_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[4] = (v1456_p + (v1500_data * v1550_bc));
              float v1556_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[12] = (v1462_p + (v1500_data * v1556_bc));
              float v1562_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[5] = (v1468_p + (v1500_data * v1562_bc));
              float v1568_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[13] = (v1474_p + (v1500_data * v1568_bc));
              float v1574_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[6] = (v1480_p + (v1500_data * v1574_bc));
              float v1580_bc = sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[14] = (v1486_p + (v1500_data * v1580_bc));
              float v1586_bc = sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[7] = (v1492_p + (v1500_data * v1586_bc));
              float v1592_bc = sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[15] = (v1498_p + (v1500_data * v1592_bc));
              // r1 = ir1
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v1603_i0 = 0; v1603_i0 < 1; ++v1603_i0) {
                #pragma unroll
                for (int32_t v1604_i1 = 0; v1604_i1 < 8; ++v1604_i1) {
                  #pragma unroll
                  for (int32_t v1605_i2 = 0; v1605_i2 < 2; ++v1605_i2) {
                    float v1609_data = ir1[(v1604_i1 + (v1605_i2 * 8))];
                    if ((v1605_i2 == 0 ? batchIdStack0_0 : batchIdStack0_1)) {
                      float *const glb_m0_se0 = v1605_i2 == 0 ? glb_m0_s0 : glb_m0_s1;
                      glb_m0_se0[((v1631_lead + (v1603_i0 * 16)) + (v1604_i1 * 16))] = v1609_data;
                    }
                  }
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              if (allowed_next) {
                // r0 = load{g>r}(glb_m2);
                #pragma unroll
                for (int32_t v1640_i0 = 0; v1640_i0 < 1; ++v1640_i0) {
                  int32_t v1642_lead = v1631_lead + (v1640_i0 * 16);
                  #pragma unroll
                  for (int32_t v1643_i1 = 0; v1643_i1 < 8; ++v1643_i1) {
                    int32_t v1645_a = v1642_lead + (v1643_i1 * 16);
                    int32_t v1646_a = v1640_i0 + v1643_i1;
                    #pragma unroll
                    for (int32_t v1647_i2 = 0; v1647_i2 < 2; ++v1647_i2) {
                      const float *const __restrict__ glb_m2_se0 = v1647_i2 == 0 ? wrap_glb_m2_s0 : wrap_glb_m2_s1;
                      float v1649_data = glb_m2_se0[v1645_a];
                      r0[(v1646_a + (v1647_i2 * 8))] = v1649_data;
                    }
                  }
                }
              }
            }
            v1620_flagWord = v1621_flagWordNext;
            v1621_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


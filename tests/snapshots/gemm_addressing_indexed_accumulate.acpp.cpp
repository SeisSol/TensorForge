// === base name ===
kernel_3c48323419c82eec

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3c48323419c82eec = {{32, 1, 1}, 32, 20, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3c48323419c82eec(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3c48323419c82eec(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3c48323419c82eec(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 1 - 1) / 1;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_3c48323419c82eec(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3c48323419c82eec(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3c48323419c82eec(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3c48323419c82eec(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (20 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 20×9(20×9) {0..20}×{0..9} indexed_pointer(slots=2, slot=1)
        //   m1 20×9(20×9) {0..20}×{0..9} none
        //   m2 9×9(9×9) {0..9}×{0..9} indexed(offset=12)
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"indexed_pointer(slots=2, slot=1)","alias":"C","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"indexed(offset=12)","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"indexed_pointer(slots=2, slot=1)","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]},{"addressing":"indexed(offset=12)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          float r0[9]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v555_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v555_flagSecondAhead < numElements0) ? v555_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
          const float *const __restrict__ peel_glb_m2 = &m2[peel_elementIndex * 81 + 0 + 12 + m2_extraOffset];
          int32_t v566_lead = item.get_local_id(2) % 32;
          bool v567_g = v566_lead < 9;
          bool v585_g = allowed_peel && v567_g;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v577_i1 = 0; v577_i1 < 9; ++v577_i1) {
            float v580_data_pre = peel_glb_m2[v585_g ? ((v566_lead + (v577_i1 * 9))) : (0)];
            float v580_data = v585_g ? (v580_data_pre) : (0.0f);
            r0[v577_i1] = v580_data;
          }
          uint32_t v559_flagWord = flagWordFirst;
          uint32_t v560_flagWordNext = flagWordSecond;
          size_t v595_elementIndex = peel_elementIndex;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v11_batchId1]);
            const float *const __restrict__ wrap_glb_m2 = &m2[wrap_elementIndex * 81 + 0 + 12 + m2_extraOffset];
            size_t v590_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v590_flagAheadAhead < numElements0) ? v590_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v559_flagWord);
            const bool allowed_next = static_cast<bool>(v560_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v595_elementIndex * 2 + 1][0 + m0_extraOffset];
              float r1[9]{};
              // r1 = load{g>r}(glb_m0);
              bool v537_g = v566_lead < 20;
              #pragma unroll
              for (int32_t v532_i1 = 0; v532_i1 < 9; ++v532_i1) {
                float v538_data_pre = glb_m0[v537_g ? ((v566_lead + (v532_i1 * 20))) : (0)];
                float v538_data = v537_g ? (v538_data_pre) : (0.0f);
                r1[v532_i1] = v538_data;
              }
              float r2[9]{};
              // ir2 = +(glb_m1 * r0)
              // [(0, 20), (0, 9)] [(0, 9)]
              float ir2[9]{};
              float v37_data_pre = glb_m1[v537_g ? (v566_lead) : (0)];
              float v37_data = v537_g ? (v37_data_pre) : (0.0f);
              float v38_data = r0[0];
              float v39_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v40_p = v37_data * v39_bc;
              float v44_data = r0[1];
              float v45_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v46_p = v37_data * v45_bc;
              float v50_data = r0[2];
              float v51_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v52_p = v37_data * v51_bc;
              float v56_data = r0[3];
              float v57_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v58_p = v37_data * v57_bc;
              float v62_data = r0[4];
              float v63_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v64_p = v37_data * v63_bc;
              float v68_data = r0[5];
              float v69_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v70_p = v37_data * v69_bc;
              float v74_data = r0[6];
              float v75_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v76_p = v37_data * v75_bc;
              float v80_data = r0[7];
              float v81_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v82_p = v37_data * v81_bc;
              float v86_data = r0[8];
              float v87_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v88_p = v37_data * v87_bc;
              float v92_data_pre = glb_m1[v537_g ? ((v566_lead + 20)) : (0)];
              float v92_data = v537_g ? (v92_data_pre) : (0.0f);
              bool v586_g = allowed_next && v567_g;
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v581_i1 = 0; v581_i1 < 9; ++v581_i1) {
                float v584_data_pre = wrap_glb_m2[v586_g ? ((v566_lead + (v581_i1 * 9))) : (0)];
                float v584_data = v586_g ? (v584_data_pre) : (0.0f);
                r0[v581_i1] = v584_data;
              }
              float v94_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v100_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v106_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v112_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v118_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v124_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v130_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v136_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v142_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v147_data_pre = glb_m1[v537_g ? ((v566_lead + 40)) : (0)];
              float v147_data = v537_g ? (v147_data_pre) : (0.0f);
              float v149_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v155_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v161_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v167_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v173_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v179_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v185_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v191_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v197_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v202_data_pre = glb_m1[v537_g ? ((v566_lead + 60)) : (0)];
              float v202_data = v537_g ? (v202_data_pre) : (0.0f);
              float v204_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v210_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v216_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v222_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v228_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v234_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v240_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v246_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v252_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v257_data_pre = glb_m1[v537_g ? ((v566_lead + 80)) : (0)];
              float v257_data = v537_g ? (v257_data_pre) : (0.0f);
              float v259_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v265_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v271_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v277_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v283_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v289_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v295_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v301_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v307_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v312_data_pre = glb_m1[v537_g ? ((v566_lead + 100)) : (0)];
              float v312_data = v537_g ? (v312_data_pre) : (0.0f);
              float v314_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v320_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v326_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v332_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v338_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v344_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v350_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v356_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v362_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v367_data_pre = glb_m1[v537_g ? ((v566_lead + 120)) : (0)];
              float v367_data = v537_g ? (v367_data_pre) : (0.0f);
              float v369_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v375_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v381_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v387_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v393_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v399_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v405_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v411_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v417_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v422_data_pre = glb_m1[v537_g ? ((v566_lead + 140)) : (0)];
              float v422_data = v537_g ? (v422_data_pre) : (0.0f);
              float v424_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v430_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v433_p = ((((((v46_p + (v92_data * v100_bc)) + (v147_data * v155_bc)) + (v202_data * v210_bc)) + (v257_data * v265_bc)) + (v312_data * v320_bc)) + (v367_data * v375_bc)) + (v422_data * v430_bc);
              float v436_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v439_p = ((((((v52_p + (v92_data * v106_bc)) + (v147_data * v161_bc)) + (v202_data * v216_bc)) + (v257_data * v271_bc)) + (v312_data * v326_bc)) + (v367_data * v381_bc)) + (v422_data * v436_bc);
              float v442_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v445_p = ((((((v58_p + (v92_data * v112_bc)) + (v147_data * v167_bc)) + (v202_data * v222_bc)) + (v257_data * v277_bc)) + (v312_data * v332_bc)) + (v367_data * v387_bc)) + (v422_data * v442_bc);
              float v448_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v451_p = ((((((v64_p + (v92_data * v118_bc)) + (v147_data * v173_bc)) + (v202_data * v228_bc)) + (v257_data * v283_bc)) + (v312_data * v338_bc)) + (v367_data * v393_bc)) + (v422_data * v448_bc);
              float v454_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v457_p = ((((((v70_p + (v92_data * v124_bc)) + (v147_data * v179_bc)) + (v202_data * v234_bc)) + (v257_data * v289_bc)) + (v312_data * v344_bc)) + (v367_data * v399_bc)) + (v422_data * v454_bc);
              float v460_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v463_p = ((((((v76_p + (v92_data * v130_bc)) + (v147_data * v185_bc)) + (v202_data * v240_bc)) + (v257_data * v295_bc)) + (v312_data * v350_bc)) + (v367_data * v405_bc)) + (v422_data * v460_bc);
              float v466_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v469_p = ((((((v82_p + (v92_data * v136_bc)) + (v147_data * v191_bc)) + (v202_data * v246_bc)) + (v257_data * v301_bc)) + (v312_data * v356_bc)) + (v367_data * v411_bc)) + (v422_data * v466_bc);
              float v472_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v475_p = ((((((v88_p + (v92_data * v142_bc)) + (v147_data * v197_bc)) + (v202_data * v252_bc)) + (v257_data * v307_bc)) + (v312_data * v362_bc)) + (v367_data * v417_bc)) + (v422_data * v472_bc);
              float v477_data_pre = glb_m1[v537_g ? ((v566_lead + 160)) : (0)];
              float v477_data = v537_g ? (v477_data_pre) : (0.0f);
              float v479_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[0] = ((((((((v40_p + (v92_data * v94_bc)) + (v147_data * v149_bc)) + (v202_data * v204_bc)) + (v257_data * v259_bc)) + (v312_data * v314_bc)) + (v367_data * v369_bc)) + (v422_data * v424_bc)) + (v477_data * v479_bc));
              float v485_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[1] = (v433_p + (v477_data * v485_bc));
              float v491_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[2] = (v439_p + (v477_data * v491_bc));
              float v497_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[3] = (v445_p + (v477_data * v497_bc));
              float v503_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[4] = (v451_p + (v477_data * v503_bc));
              float v509_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[5] = (v457_p + (v477_data * v509_bc));
              float v515_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[6] = (v463_p + (v477_data * v515_bc));
              float v521_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[7] = (v469_p + (v477_data * v521_bc));
              float v527_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[8] = (v475_p + (v477_data * v527_bc));
              // r2 = ir2 + r1
              #pragma unroll
              for (int32_t v541_n1 = 0; v541_n1 < 9; ++v541_n1) {
                float v543_data_pre = ir2[v541_n1];
                float v543_data = v537_g ? (v543_data_pre) : (0.0f);
                float v544_data_pre = r1[v541_n1];
                float v544_data = v537_g ? (v544_data_pre) : (0.0f);
                if (v537_g) {
                  r2[v541_n1] = (v544_data + v543_data);
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v547_i1 = 0; v547_i1 < 9; ++v547_i1) {
                float v549_data_pre = r2[v547_i1];
                float v549_data = v537_g ? (v549_data_pre) : (0.0f);
                if (v537_g) {
                  glb_m0[(v566_lead + (v547_i1 * 20))] = v549_data;
                }
              }
              item.barrier();
            }
            else {
              bool v587_g = allowed_next && v567_g;
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v573_i1 = 0; v573_i1 < 9; ++v573_i1) {
                float v576_data_pre = wrap_glb_m2[v587_g ? ((v566_lead + (v573_i1 * 9))) : (0)];
                float v576_data = v587_g ? (v576_data_pre) : (0.0f);
                r0[v573_i1] = v576_data;
              }
            }
            v559_flagWord = v560_flagWordNext;
            v560_flagWordNext = flagWordAhead;
            v595_elementIndex = wrap_elementIndex;
          }
        }
      });
    }
  });
}


// === base name ===
kernel_5efd66bc02ddf038

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5efd66bc02ddf038 = {{32, 1, 1}, 32, 40, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5efd66bc02ddf038(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5efd66bc02ddf038(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5efd66bc02ddf038(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_5efd66bc02ddf038(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5efd66bc02ddf038(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5efd66bc02ddf038(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5efd66bc02ddf038(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (40 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none ordered
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":true,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          float r0[6]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v616_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v616_flagSecondAhead < numElements0) ? v616_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          int32_t v624_lead = item.get_local_id(2) % 32;
          bool v625_g = v624_lead < 8;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v634_i1 = 0; v634_i1 < 6; ++v634_i1) {
            float v637_data_pre = peel_glb_m2[v625_g ? ((v624_lead + (v634_i1 * 8))) : (0)];
            float v637_data = v625_g ? (v637_data_pre) : (0.0f);
            r0[v634_i1] = v637_data;
          }
          uint32_t v620_flagWord = flagWordFirst;
          uint32_t v621_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v644_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v644_flagAheadAhead < numElements0) ? v644_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v620_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 240 + 0 + m0_extraOffset];
              float r1[12]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 40), (0, 6)] [(0, 8)]
              float ir1[12]{};
              int32_t v32_a = v624_lead * 2;
              sycl::vec<float, 2> v34_data = *(sycl::vec<float, 2>*)&glb_m1[v32_a];
              float v35_p = v34_data[1];
              float v36_data = r0[0];
              ir1[1] = (v35_p * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v43_data = r0[1];
              ir1[3] = (v35_p * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v50_data = r0[2];
              ir1[5] = (v35_p * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v57_data = r0[3];
              ir1[7] = (v35_p * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v64_data = r0[4];
              ir1[9] = (v35_p * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v71_data = r0[5];
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v638_i1 = 0; v638_i1 < 6; ++v638_i1) {
                float v641_data_pre = wrap_glb_m2[v625_g ? ((v624_lead + (v638_i1 * 8))) : (0)];
                float v641_data = v625_g ? (v641_data_pre) : (0.0f);
                r0[v638_i1] = v641_data;
              }
              ir1[11] = (v35_p * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              sycl::vec<float, 2> v77_data = *(sycl::vec<float, 2>*)&glb_m1[(64 + v32_a)];
              float v78_p = v77_data[0];
              float v80_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[0] = (v78_p * v80_bc);
              float v87_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[2] = (v78_p * v87_bc);
              float v94_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[4] = (v78_p * v94_bc);
              float v101_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[6] = (v78_p * v101_bc);
              float v108_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[8] = (v78_p * v108_bc);
              float v115_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[10] = (v78_p * v115_bc);
              float v120_p = v77_data[1];
              float v124_data = ir1[1];
              ir1[1] = (v124_data + (v120_p * v80_bc));
              float v131_data = ir1[3];
              ir1[3] = (v131_data + (v120_p * v87_bc));
              float v138_data = ir1[5];
              ir1[5] = (v138_data + (v120_p * v94_bc));
              float v145_data = ir1[7];
              ir1[7] = (v145_data + (v120_p * v101_bc));
              float v152_data = ir1[9];
              ir1[9] = (v152_data + (v120_p * v108_bc));
              float v159_data = ir1[11];
              ir1[11] = (v159_data + (v120_p * v115_bc));
              sycl::vec<float, 2> v162_data = *(sycl::vec<float, 2>*)&glb_m1[(128 + v32_a)];
              float v163_p = v162_data[1];
              float v167_data = ir1[1];
              ir1[1] = (v167_data + (v163_p * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v174_data = ir1[3];
              ir1[3] = (v174_data + (v163_p * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v181_data = ir1[5];
              ir1[5] = (v181_data + (v163_p * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v188_data = ir1[7];
              ir1[7] = (v188_data + (v163_p * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v195_data = ir1[9];
              ir1[9] = (v195_data + (v163_p * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v202_data = ir1[11];
              ir1[11] = (v202_data + (v163_p * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              sycl::vec<float, 2> v205_data = *(sycl::vec<float, 2>*)&glb_m1[(192 + v32_a)];
              float v206_p = v205_data[0];
              float v208_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v210_data = ir1[0];
              ir1[0] = (v210_data + (v206_p * v208_bc));
              float v215_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v217_data = ir1[2];
              ir1[2] = (v217_data + (v206_p * v215_bc));
              float v222_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v224_data = ir1[4];
              ir1[4] = (v224_data + (v206_p * v222_bc));
              float v229_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v231_data = ir1[6];
              ir1[6] = (v231_data + (v206_p * v229_bc));
              float v236_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v238_data = ir1[8];
              ir1[8] = (v238_data + (v206_p * v236_bc));
              float v243_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v245_data = ir1[10];
              ir1[10] = (v245_data + (v206_p * v243_bc));
              float v248_p = v205_data[1];
              float v252_data = ir1[1];
              ir1[1] = (v252_data + (v248_p * v208_bc));
              float v259_data = ir1[3];
              ir1[3] = (v259_data + (v248_p * v215_bc));
              float v266_data = ir1[5];
              ir1[5] = (v266_data + (v248_p * v222_bc));
              float v273_data = ir1[7];
              ir1[7] = (v273_data + (v248_p * v229_bc));
              float v280_data = ir1[9];
              ir1[9] = (v280_data + (v248_p * v236_bc));
              float v287_data = ir1[11];
              ir1[11] = (v287_data + (v248_p * v243_bc));
              sycl::vec<float, 2> v290_data = *(sycl::vec<float, 2>*)&glb_m1[(256 + v32_a)];
              float v291_p = v290_data[0];
              float v293_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v295_data = ir1[0];
              ir1[0] = (v295_data + (v291_p * v293_bc));
              float v300_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v302_data = ir1[2];
              ir1[2] = (v302_data + (v291_p * v300_bc));
              float v307_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v309_data = ir1[4];
              ir1[4] = (v309_data + (v291_p * v307_bc));
              float v314_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v316_data = ir1[6];
              ir1[6] = (v316_data + (v291_p * v314_bc));
              float v321_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v323_data = ir1[8];
              ir1[8] = (v323_data + (v291_p * v321_bc));
              float v328_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v330_data = ir1[10];
              ir1[10] = (v330_data + (v291_p * v328_bc));
              float v333_p = v290_data[1];
              float v337_data = ir1[1];
              ir1[1] = (v337_data + (v333_p * v293_bc));
              float v344_data = ir1[3];
              ir1[3] = (v344_data + (v333_p * v300_bc));
              float v351_data = ir1[5];
              ir1[5] = (v351_data + (v333_p * v307_bc));
              float v358_data = ir1[7];
              ir1[7] = (v358_data + (v333_p * v314_bc));
              float v365_data = ir1[9];
              ir1[9] = (v365_data + (v333_p * v321_bc));
              float v372_data = ir1[11];
              ir1[11] = (v372_data + (v333_p * v328_bc));
              sycl::vec<float, 2> v375_data = *(sycl::vec<float, 2>*)&glb_m1[(320 + v32_a)];
              float v376_p = v375_data[1];
              float v380_data = ir1[1];
              ir1[1] = (v380_data + (v376_p * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v387_data = ir1[3];
              ir1[3] = (v387_data + (v376_p * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v394_data = ir1[5];
              ir1[5] = (v394_data + (v376_p * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v401_data = ir1[7];
              ir1[7] = (v401_data + (v376_p * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v408_data = ir1[9];
              ir1[9] = (v408_data + (v376_p * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v415_data = ir1[11];
              ir1[11] = (v415_data + (v376_p * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              sycl::vec<float, 2> v418_data = *(sycl::vec<float, 2>*)&glb_m1[(384 + v32_a)];
              float v419_p = v418_data[0];
              float v421_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v423_data = ir1[0];
              ir1[0] = (v423_data + (v419_p * v421_bc));
              float v428_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v430_data = ir1[2];
              ir1[2] = (v430_data + (v419_p * v428_bc));
              float v435_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v437_data = ir1[4];
              ir1[4] = (v437_data + (v419_p * v435_bc));
              float v442_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v444_data = ir1[6];
              ir1[6] = (v444_data + (v419_p * v442_bc));
              float v449_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v451_data = ir1[8];
              ir1[8] = (v451_data + (v419_p * v449_bc));
              float v456_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v458_data = ir1[10];
              ir1[10] = (v458_data + (v419_p * v456_bc));
              float v461_p = v418_data[1];
              float v465_data = ir1[1];
              ir1[1] = (v465_data + (v461_p * v421_bc));
              float v472_data = ir1[3];
              ir1[3] = (v472_data + (v461_p * v428_bc));
              float v479_data = ir1[5];
              ir1[5] = (v479_data + (v461_p * v435_bc));
              float v486_data = ir1[7];
              ir1[7] = (v486_data + (v461_p * v442_bc));
              float v493_data = ir1[9];
              ir1[9] = (v493_data + (v461_p * v449_bc));
              float v500_data = ir1[11];
              ir1[11] = (v500_data + (v461_p * v456_bc));
              sycl::vec<float, 2> v503_data = *(sycl::vec<float, 2>*)&glb_m1[(448 + v32_a)];
              float v504_p = v503_data[0];
              float v506_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v508_data = ir1[0];
              ir1[0] = (v508_data + (v504_p * v506_bc));
              float v513_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v515_data = ir1[2];
              ir1[2] = (v515_data + (v504_p * v513_bc));
              float v520_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v522_data = ir1[4];
              ir1[4] = (v522_data + (v504_p * v520_bc));
              float v527_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v529_data = ir1[6];
              ir1[6] = (v529_data + (v504_p * v527_bc));
              float v534_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v536_data = ir1[8];
              ir1[8] = (v536_data + (v504_p * v534_bc));
              float v541_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v543_data = ir1[10];
              ir1[10] = (v543_data + (v504_p * v541_bc));
              float v546_p = v503_data[1];
              float v550_data = ir1[1];
              ir1[1] = (v550_data + (v546_p * v506_bc));
              float v557_data = ir1[3];
              ir1[3] = (v557_data + (v546_p * v513_bc));
              float v564_data = ir1[5];
              ir1[5] = (v564_data + (v546_p * v520_bc));
              float v571_data = ir1[7];
              ir1[7] = (v571_data + (v546_p * v527_bc));
              float v578_data = ir1[9];
              ir1[9] = (v578_data + (v546_p * v534_bc));
              float v585_data = ir1[11];
              ir1[11] = (v585_data + (v546_p * v541_bc));
              // r1 = ir1
              #pragma unroll
              for (int32_t v587_n0 = 0; v587_n0 < 1; ++v587_n0) {
                #pragma unroll
                for (int32_t v588_n1 = 0; v588_n1 < 6; ++v588_n1) {
                  int32_t v590_a = v587_n0 + (v588_n1 * 2);
                  float v591_data = ir1[v590_a];
                  r1[v590_a] = v591_data;
                }
              }
              if (v625_g) {
                #pragma unroll
                for (int32_t v593_n1 = 0; v593_n1 < 6; ++v593_n1) {
                  int32_t v595_a = 1 + (v593_n1 * 2);
                  float v596_data = ir1[v595_a];
                  r1[v595_a] = v596_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v597_i0 = 0; v597_i0 < 1; ++v597_i0) {
                int32_t v603_lead = v624_lead + (v597_i0 * 32);
                #pragma unroll
                for (int32_t v598_i1 = 0; v598_i1 < 6; ++v598_i1) {
                  float v601_data = r1[(v597_i0 + (v598_i1 * 2))];
                  glb_m0[(v603_lead + (v598_i1 * 40))] = v601_data;
                }
              }
              if (v625_g) {
                int32_t v612_lead = v624_lead + 32_i32;
                #pragma unroll
                for (int32_t v607_i1 = 0; v607_i1 < 6; ++v607_i1) {
                  float v610_data = r1[(1 + (v607_i1 * 2))];
                  glb_m0[(v612_lead + (v607_i1 * 40))] = v610_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v630_i1 = 0; v630_i1 < 6; ++v630_i1) {
                float v633_data_pre = wrap_glb_m2[v625_g ? ((v624_lead + (v630_i1 * 8))) : (0)];
                float v633_data = v625_g ? (v633_data_pre) : (0.0f);
                r0[v630_i1] = v633_data;
              }
            }
            v620_flagWord = v621_flagWordNext;
            v621_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


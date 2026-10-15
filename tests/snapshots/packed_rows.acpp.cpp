// === base name ===
kernel_a3ab90d4fe1a3c1b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a3ab90d4fe1a3c1b = {{32, 1, 1}, 32, 40, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a3ab90d4fe1a3c1b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a3ab90d4fe1a3c1b(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a3ab90d4fe1a3c1b(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_a3ab90d4fe1a3c1b(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a3ab90d4fe1a3c1b(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a3ab90d4fe1a3c1b(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a3ab90d4fe1a3c1b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: pack_operands=rows
        // launch: 32 lanes (40 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none ordered rows-packed
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":true,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          float r0[6]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v370_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v370_flagSecondAhead < numElements0) ? v370_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          int32_t v378_lead = item.get_local_id(2) % 32;
          bool v379_g = v378_lead < 8;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v388_i1 = 0; v388_i1 < 6; ++v388_i1) {
            float v391_data_pre = peel_glb_m2[v379_g ? ((v378_lead + (v388_i1 * 8))) : (0)];
            float v391_data = v379_g ? (v391_data_pre) : (0.0f);
            r0[v388_i1] = v391_data;
          }
          uint32_t v374_flagWord = flagWordFirst;
          uint32_t v375_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v398_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v398_flagAheadAhead < numElements0) ? v398_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v374_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 240 + 0 + m0_extraOffset];
              float r1[12]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 40), (0, 6)] [(0, 8)]
              float ir1[12]{};
              bool v38_g = (v378_lead >= 5) && (v378_lead < 29);
              float v39_data_pre = glb_m1[v38_g ? ((v378_lead + -5)) : (0)];
              float v39_data = v38_g ? (v39_data_pre) : (0.0f);
              float v40_data = r0[0];
              ir1[0] = (v39_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v46_data = r0[1];
              ir1[2] = (v39_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v52_data = r0[2];
              ir1[4] = (v39_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v58_data = r0[3];
              ir1[6] = (v39_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v64_data = r0[4];
              ir1[8] = (v39_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v70_data = r0[5];
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v392_i1 = 0; v392_i1 < 6; ++v392_i1) {
                float v395_data_pre = wrap_glb_m2[v379_g ? ((v378_lead + (v392_i1 * 8))) : (0)];
                float v395_data = v379_g ? (v395_data_pre) : (0.0f);
                r0[v392_i1] = v395_data;
              }
              ir1[10] = (v39_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v77_data_pre = glb_m1[v38_g ? ((v378_lead + 19)) : (0)];
              float v77_data = v38_g ? (v77_data_pre) : (0.0f);
              float v81_data = ir1[0];
              ir1[0] = (v81_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v87_data = ir1[2];
              ir1[2] = (v87_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v93_data = ir1[4];
              ir1[4] = (v93_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v99_data = ir1[6];
              ir1[6] = (v99_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v105_data = ir1[8];
              ir1[8] = (v105_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v111_data = ir1[10];
              ir1[10] = (v111_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v115_data_pre = glb_m1[v38_g ? ((v378_lead + 43)) : (0)];
              float v115_data = v38_g ? (v115_data_pre) : (0.0f);
              float v119_data = ir1[0];
              ir1[0] = (v119_data + (v115_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v125_data = ir1[2];
              ir1[2] = (v125_data + (v115_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v131_data = ir1[4];
              ir1[4] = (v131_data + (v115_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v137_data = ir1[6];
              ir1[6] = (v137_data + (v115_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v143_data = ir1[8];
              ir1[8] = (v143_data + (v115_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v149_data = ir1[10];
              ir1[10] = (v149_data + (v115_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v153_data_pre = glb_m1[v38_g ? ((v378_lead + 67)) : (0)];
              float v153_data = v38_g ? (v153_data_pre) : (0.0f);
              float v157_data = ir1[0];
              ir1[0] = (v157_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v163_data = ir1[2];
              ir1[2] = (v163_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v169_data = ir1[4];
              ir1[4] = (v169_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v175_data = ir1[6];
              ir1[6] = (v175_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v181_data = ir1[8];
              ir1[8] = (v181_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v187_data = ir1[10];
              ir1[10] = (v187_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v191_data_pre = glb_m1[v38_g ? ((v378_lead + 91)) : (0)];
              float v191_data = v38_g ? (v191_data_pre) : (0.0f);
              float v195_data = ir1[0];
              ir1[0] = (v195_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v201_data = ir1[2];
              ir1[2] = (v201_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v207_data = ir1[4];
              ir1[4] = (v207_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v213_data = ir1[6];
              ir1[6] = (v213_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v219_data = ir1[8];
              ir1[8] = (v219_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v225_data = ir1[10];
              ir1[10] = (v225_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v229_data_pre = glb_m1[v38_g ? ((v378_lead + 115)) : (0)];
              float v229_data = v38_g ? (v229_data_pre) : (0.0f);
              float v233_data = ir1[0];
              ir1[0] = (v233_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v239_data = ir1[2];
              ir1[2] = (v239_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v245_data = ir1[4];
              ir1[4] = (v245_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v251_data = ir1[6];
              ir1[6] = (v251_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v257_data = ir1[8];
              ir1[8] = (v257_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v263_data = ir1[10];
              ir1[10] = (v263_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v267_data_pre = glb_m1[v38_g ? ((v378_lead + 139)) : (0)];
              float v267_data = v38_g ? (v267_data_pre) : (0.0f);
              float v271_data = ir1[0];
              ir1[0] = (v271_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v277_data = ir1[2];
              ir1[2] = (v277_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v283_data = ir1[4];
              ir1[4] = (v283_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v289_data = ir1[6];
              ir1[6] = (v289_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v295_data = ir1[8];
              ir1[8] = (v295_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v301_data = ir1[10];
              ir1[10] = (v301_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v305_data_pre = glb_m1[v38_g ? ((v378_lead + 163)) : (0)];
              float v305_data = v38_g ? (v305_data_pre) : (0.0f);
              float v309_data = ir1[0];
              ir1[0] = (v309_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v315_data = ir1[2];
              ir1[2] = (v315_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v321_data = ir1[4];
              ir1[4] = (v321_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v327_data = ir1[6];
              ir1[6] = (v327_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v333_data = ir1[8];
              ir1[8] = (v333_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v339_data = ir1[10];
              ir1[10] = (v339_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              // r1 = ir1
              #pragma unroll
              for (int32_t v341_n0 = 0; v341_n0 < 1; ++v341_n0) {
                #pragma unroll
                for (int32_t v342_n1 = 0; v342_n1 < 6; ++v342_n1) {
                  int32_t v344_a = v341_n0 + (v342_n1 * 2);
                  float v345_data = ir1[v344_a];
                  r1[v344_a] = v345_data;
                }
              }
              if (v379_g) {
                #pragma unroll
                for (int32_t v347_n1 = 0; v347_n1 < 6; ++v347_n1) {
                  int32_t v349_a = 1 + (v347_n1 * 2);
                  float v350_data = ir1[v349_a];
                  r1[v349_a] = v350_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v351_i0 = 0; v351_i0 < 1; ++v351_i0) {
                int32_t v357_lead = v378_lead + (v351_i0 * 32);
                #pragma unroll
                for (int32_t v352_i1 = 0; v352_i1 < 6; ++v352_i1) {
                  float v355_data = r1[(v351_i0 + (v352_i1 * 2))];
                  glb_m0[(v357_lead + (v352_i1 * 40))] = v355_data;
                }
              }
              if (v379_g) {
                int32_t v366_lead = v378_lead + 32_i32;
                #pragma unroll
                for (int32_t v361_i1 = 0; v361_i1 < 6; ++v361_i1) {
                  float v364_data = r1[(1 + (v361_i1 * 2))];
                  glb_m0[(v366_lead + (v361_i1 * 40))] = v364_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v384_i1 = 0; v384_i1 < 6; ++v384_i1) {
                float v387_data_pre = wrap_glb_m2[v379_g ? ((v378_lead + (v384_i1 * 8))) : (0)];
                float v387_data = v379_g ? (v387_data_pre) : (0.0f);
                r0[v384_i1] = v387_data;
              }
            }
            v374_flagWord = v375_flagWordNext;
            v375_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


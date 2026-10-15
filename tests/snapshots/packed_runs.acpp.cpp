// === base name ===
kernel_424892e3d1b4f198

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_424892e3d1b4f198 = {{32, 1, 1}, 32, 40, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_424892e3d1b4f198(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_424892e3d1b4f198(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_424892e3d1b4f198(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_424892e3d1b4f198(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_424892e3d1b4f198(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_424892e3d1b4f198(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_424892e3d1b4f198(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: pack_operands=runs
        // launch: 32 lanes (40 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none ordered runs-packed
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":true,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          float r0[6]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v461_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v461_flagSecondAhead < numElements0) ? v461_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          int32_t v469_lead = item.get_local_id(2) % 32;
          bool v470_g = v469_lead < 8;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v479_i1 = 0; v479_i1 < 6; ++v479_i1) {
            float v482_data_pre = peel_glb_m2[v470_g ? ((v469_lead + (v479_i1 * 8))) : (0)];
            float v482_data = v470_g ? (v482_data_pre) : (0.0f);
            r0[v479_i1] = v482_data;
          }
          uint32_t v465_flagWord = flagWordFirst;
          uint32_t v466_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v489_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v489_flagAheadAhead < numElements0) ? v489_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v465_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 240 + 0 + m0_extraOffset];
              float r1[12]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 40), (0, 6)] [(0, 8)]
              float ir1[12]{};
              float v36_data_pre = glb_m1[v470_g ? (v469_lead) : (0)];
              float v36_data = v470_g ? (v36_data_pre) : (0.0f);
              float v37_data = r0[0];
              ir1[0] = (v36_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v43_data = r0[1];
              ir1[2] = (v36_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v49_data = r0[2];
              ir1[4] = (v36_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v55_data = r0[3];
              ir1[6] = (v36_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v61_data = r0[4];
              ir1[8] = (v36_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v67_data = r0[5];
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v483_i1 = 0; v483_i1 < 6; ++v483_i1) {
                float v486_data_pre = wrap_glb_m2[v470_g ? ((v469_lead + (v483_i1 * 8))) : (0)];
                float v486_data = v470_g ? (v486_data_pre) : (0.0f);
                r0[v483_i1] = v486_data;
              }
              ir1[10] = (v36_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v77_data_pre = glb_m1[((v469_lead >= 3) && (v469_lead < 21)) ? ((v469_lead + 5)) : (0)];
              float v77_data = ((v469_lead >= 3) && (v469_lead < 21)) ? (v77_data_pre) : (0.0f);
              float v81_data = ir1[0];
              ir1[0] = (v81_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v87_data = ir1[2];
              ir1[2] = (v87_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v93_data = ir1[4];
              ir1[4] = (v93_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v99_data = ir1[6];
              ir1[6] = (v99_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v105_data = ir1[8];
              ir1[8] = (v105_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v111_data = ir1[10];
              ir1[10] = (v111_data + (v77_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v116_data_pre = glb_m1[(v469_lead >= 28) ? ((v469_lead + -2)) : (0)];
              float v116_data = (v469_lead >= 28) ? (v116_data_pre) : (0.0f);
              float v118_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v120_data = ir1[0];
              ir1[0] = (v120_data + (v116_data * v118_bc));
              float v124_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v126_data = ir1[2];
              ir1[2] = (v126_data + (v116_data * v124_bc));
              float v130_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v132_data = ir1[4];
              ir1[4] = (v132_data + (v116_data * v130_bc));
              float v136_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v138_data = ir1[6];
              ir1[6] = (v138_data + (v116_data * v136_bc));
              float v142_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v144_data = ir1[8];
              ir1[8] = (v144_data + (v116_data * v142_bc));
              float v148_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v150_data = ir1[10];
              ir1[10] = (v150_data + (v116_data * v148_bc));
              int32_t v153_lead = v469_lead + 32_i32;
              float v157_data_pre = glb_m1[(v469_lead < 4) ? ((v153_lead + -2)) : (0)];
              float v157_data = (v469_lead < 4) ? (v157_data_pre) : (0.0f);
              ir1[1] = (v157_data * v118_bc);
              ir1[3] = (v157_data * v124_bc);
              ir1[5] = (v157_data * v130_bc);
              ir1[7] = (v157_data * v136_bc);
              ir1[9] = (v157_data * v142_bc);
              ir1[11] = (v157_data * v148_bc);
              float v197_data_pre = glb_m1[((v469_lead >= 1) && v470_g) ? ((v153_lead + 1)) : (0)];
              float v197_data = ((v469_lead >= 1) && v470_g) ? (v197_data_pre) : (0.0f);
              float v201_data = ir1[1];
              ir1[1] = (v201_data + (v197_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v207_data = ir1[3];
              ir1[3] = (v207_data + (v197_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v213_data = ir1[5];
              ir1[5] = (v213_data + (v197_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v219_data = ir1[7];
              ir1[7] = (v219_data + (v197_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v225_data = ir1[9];
              ir1[9] = (v225_data + (v197_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v231_data = ir1[11];
              ir1[11] = (v231_data + (v197_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v238_data_pre = glb_m1[((v469_lead >= 17) && (v469_lead < 18)) ? ((v469_lead + 24)) : (0)];
              float v238_data = ((v469_lead >= 17) && (v469_lead < 18)) ? (v238_data_pre) : (0.0f);
              float v242_data = ir1[0];
              ir1[0] = (v242_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v248_data = ir1[2];
              ir1[2] = (v248_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v254_data = ir1[4];
              ir1[4] = (v254_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v260_data = ir1[6];
              ir1[6] = (v260_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v266_data = ir1[8];
              ir1[8] = (v266_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v272_data = ir1[10];
              ir1[10] = (v272_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v277_data_pre = glb_m1[(v469_lead >= 9) ? ((v469_lead + 33)) : (0)];
              float v277_data = (v469_lead >= 9) ? (v277_data_pre) : (0.0f);
              float v281_data = ir1[0];
              ir1[0] = (v281_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v287_data = ir1[2];
              ir1[2] = (v287_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v293_data = ir1[4];
              ir1[4] = (v293_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v299_data = ir1[6];
              ir1[6] = (v299_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v305_data = ir1[8];
              ir1[8] = (v305_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v311_data = ir1[10];
              ir1[10] = (v311_data + (v277_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v316_data_pre = glb_m1[(v469_lead >= 31) ? ((v469_lead + 34)) : (0)];
              float v316_data = (v469_lead >= 31) ? (v316_data_pre) : (0.0f);
              float v318_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v320_data = ir1[0];
              ir1[0] = (v320_data + (v316_data * v318_bc));
              float v324_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v326_data = ir1[2];
              ir1[2] = (v326_data + (v316_data * v324_bc));
              float v330_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v332_data = ir1[4];
              ir1[4] = (v332_data + (v316_data * v330_bc));
              float v336_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v338_data = ir1[6];
              ir1[6] = (v338_data + (v316_data * v336_bc));
              float v342_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v344_data = ir1[8];
              ir1[8] = (v344_data + (v316_data * v342_bc));
              float v348_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v350_data = ir1[10];
              ir1[10] = (v350_data + (v316_data * v348_bc));
              float v355_data_pre = glb_m1[(v469_lead < 1) ? ((v153_lead + 34)) : (0)];
              float v355_data = (v469_lead < 1) ? (v355_data_pre) : (0.0f);
              float v359_data = ir1[1];
              ir1[1] = (v359_data + (v355_data * v318_bc));
              float v365_data = ir1[3];
              ir1[3] = (v365_data + (v355_data * v324_bc));
              float v371_data = ir1[5];
              ir1[5] = (v371_data + (v355_data * v330_bc));
              float v377_data = ir1[7];
              ir1[7] = (v377_data + (v355_data * v336_bc));
              float v383_data = ir1[9];
              ir1[9] = (v383_data + (v355_data * v342_bc));
              float v389_data = ir1[11];
              ir1[11] = (v389_data + (v355_data * v348_bc));
              float v396_data_pre = glb_m1[((v469_lead >= 12) && (v469_lead < 30)) ? ((v469_lead + 55)) : (0)];
              float v396_data = ((v469_lead >= 12) && (v469_lead < 30)) ? (v396_data_pre) : (0.0f);
              float v400_data = ir1[0];
              ir1[0] = (v400_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v406_data = ir1[2];
              ir1[2] = (v406_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v412_data = ir1[4];
              ir1[4] = (v412_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v418_data = ir1[6];
              ir1[6] = (v418_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v424_data = ir1[8];
              ir1[8] = (v424_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v430_data = ir1[10];
              ir1[10] = (v430_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              // r1 = ir1
              #pragma unroll
              for (int32_t v432_n0 = 0; v432_n0 < 1; ++v432_n0) {
                #pragma unroll
                for (int32_t v433_n1 = 0; v433_n1 < 6; ++v433_n1) {
                  int32_t v435_a = v432_n0 + (v433_n1 * 2);
                  float v436_data = ir1[v435_a];
                  r1[v435_a] = v436_data;
                }
              }
              if (v470_g) {
                #pragma unroll
                for (int32_t v438_n1 = 0; v438_n1 < 6; ++v438_n1) {
                  int32_t v440_a = 1 + (v438_n1 * 2);
                  float v441_data = ir1[v440_a];
                  r1[v440_a] = v441_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v442_i0 = 0; v442_i0 < 1; ++v442_i0) {
                int32_t v448_lead = v469_lead + (v442_i0 * 32);
                #pragma unroll
                for (int32_t v443_i1 = 0; v443_i1 < 6; ++v443_i1) {
                  float v446_data = r1[(v442_i0 + (v443_i1 * 2))];
                  glb_m0[(v448_lead + (v443_i1 * 40))] = v446_data;
                }
              }
              if (v470_g) {
                #pragma unroll
                for (int32_t v452_i1 = 0; v452_i1 < 6; ++v452_i1) {
                  float v455_data = r1[(1 + (v452_i1 * 2))];
                  glb_m0[(v153_lead + (v452_i1 * 40))] = v455_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v475_i1 = 0; v475_i1 < 6; ++v475_i1) {
                float v478_data_pre = wrap_glb_m2[v470_g ? ((v469_lead + (v475_i1 * 8))) : (0)];
                float v478_data = v470_g ? (v478_data_pre) : (0.0f);
                r0[v475_i1] = v478_data;
              }
            }
            v465_flagWord = v466_flagWordNext;
            v466_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


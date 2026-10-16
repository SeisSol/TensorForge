// === base name ===
kernel_9acce683c9c727d7

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9acce683c9c727d7 = {{32, 1, 1}, 32, 40, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9acce683c9c727d7(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9acce683c9c727d7(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9acce683c9c727d7(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_9acce683c9c727d7(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9acce683c9c727d7(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_9acce683c9c727d7(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_9acce683c9c727d7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          size_t v561_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v561_flagSecondAhead < numElements0) ? v561_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          int32_t v569_lead = item.get_local_id(2) % 32;
          bool v570_g = v569_lead < 8;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v579_i1 = 0; v579_i1 < 6; ++v579_i1) {
            float v582_data_pre = peel_glb_m2[v570_g ? ((v569_lead + (v579_i1 * 8))) : (0)];
            float v582_data = v570_g ? (v582_data_pre) : (0.0f);
            r0[v579_i1] = v582_data;
          }
          uint32_t v565_flagWord = flagWordFirst;
          uint32_t v566_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v589_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v589_flagAheadAhead < numElements0) ? v589_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v565_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 240 + 0 + m0_extraOffset];
              float r1[12]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 40), (0, 6)] [(0, 8)]
              float ir1[12]{};
              int32_t v33_lead = v569_lead + 32_i32;
              float v37_data_pre = glb_m1[v570_g ? ((v33_lead + -32)) : (0)];
              float v37_data = v570_g ? (v37_data_pre) : (0.0f);
              float v38_data = r0[0];
              ir1[1] = (v37_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v44_data = r0[1];
              ir1[3] = (v37_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v50_data = r0[2];
              ir1[5] = (v37_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v56_data = r0[3];
              ir1[7] = (v37_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v62_data = r0[4];
              ir1[9] = (v37_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v68_data = r0[5];
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v583_i1 = 0; v583_i1 < 6; ++v583_i1) {
                float v586_data_pre = wrap_glb_m2[v570_g ? ((v569_lead + (v583_i1 * 8))) : (0)];
                float v586_data = v570_g ? (v586_data_pre) : (0.0f);
                r0[v583_i1] = v586_data;
              }
              ir1[11] = (v37_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v77_data = glb_m1[(v569_lead + 8)];
              float v79_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[0] = (v77_data * v79_bc);
              float v85_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[2] = (v77_data * v85_bc);
              float v91_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[4] = (v77_data * v91_bc);
              float v97_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[6] = (v77_data * v97_bc);
              float v103_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[8] = (v77_data * v103_bc);
              float v109_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              ir1[10] = (v77_data * v109_bc);
              float v115_data_pre = glb_m1[v570_g ? ((v33_lead + 8)) : (0)];
              float v115_data = v570_g ? (v115_data_pre) : (0.0f);
              float v119_data = ir1[1];
              ir1[1] = (v119_data + (v115_data * v79_bc));
              float v125_data = ir1[3];
              ir1[3] = (v125_data + (v115_data * v85_bc));
              float v131_data = ir1[5];
              ir1[5] = (v131_data + (v115_data * v91_bc));
              float v137_data = ir1[7];
              ir1[7] = (v137_data + (v115_data * v97_bc));
              float v143_data = ir1[9];
              ir1[9] = (v143_data + (v115_data * v103_bc));
              float v149_data = ir1[11];
              ir1[11] = (v149_data + (v115_data * v109_bc));
              float v153_data_pre = glb_m1[v570_g ? ((v33_lead + 16)) : (0)];
              float v153_data = v570_g ? (v153_data_pre) : (0.0f);
              float v157_data = ir1[1];
              ir1[1] = (v157_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v163_data = ir1[3];
              ir1[3] = (v163_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v169_data = ir1[5];
              ir1[5] = (v169_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v175_data = ir1[7];
              ir1[7] = (v175_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v181_data = ir1[9];
              ir1[9] = (v181_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v187_data = ir1[11];
              ir1[11] = (v187_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v192_data_pre = glb_m1[(v569_lead >= 17) ? ((v569_lead + 39)) : (0)];
              float v192_data = (v569_lead >= 17) ? (v192_data_pre) : (0.0f);
              float v194_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v196_data = ir1[0];
              ir1[0] = (v196_data + (v192_data * v194_bc));
              float v200_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v202_data = ir1[2];
              ir1[2] = (v202_data + (v192_data * v200_bc));
              float v206_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v208_data = ir1[4];
              ir1[4] = (v208_data + (v192_data * v206_bc));
              float v212_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v214_data = ir1[6];
              ir1[6] = (v214_data + (v192_data * v212_bc));
              float v218_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v220_data = ir1[8];
              ir1[8] = (v220_data + (v192_data * v218_bc));
              float v224_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v226_data = ir1[10];
              ir1[10] = (v226_data + (v192_data * v224_bc));
              float v230_data_pre = glb_m1[v570_g ? ((v33_lead + 39)) : (0)];
              float v230_data = v570_g ? (v230_data_pre) : (0.0f);
              float v234_data = ir1[1];
              ir1[1] = (v234_data + (v230_data * v194_bc));
              float v240_data = ir1[3];
              ir1[3] = (v240_data + (v230_data * v200_bc));
              float v246_data = ir1[5];
              ir1[5] = (v246_data + (v230_data * v206_bc));
              float v252_data = ir1[7];
              ir1[7] = (v252_data + (v230_data * v212_bc));
              float v258_data = ir1[9];
              ir1[9] = (v258_data + (v230_data * v218_bc));
              float v264_data = ir1[11];
              ir1[11] = (v264_data + (v230_data * v224_bc));
              float v268_data = glb_m1[(v569_lead + 79)];
              float v270_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v272_data = ir1[0];
              ir1[0] = (v272_data + (v268_data * v270_bc));
              float v276_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v278_data = ir1[2];
              ir1[2] = (v278_data + (v268_data * v276_bc));
              float v282_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v284_data = ir1[4];
              ir1[4] = (v284_data + (v268_data * v282_bc));
              float v288_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v290_data = ir1[6];
              ir1[6] = (v290_data + (v268_data * v288_bc));
              float v294_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v296_data = ir1[8];
              ir1[8] = (v296_data + (v268_data * v294_bc));
              float v300_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v302_data = ir1[10];
              ir1[10] = (v302_data + (v268_data * v300_bc));
              float v306_data_pre = glb_m1[v570_g ? ((v33_lead + 79)) : (0)];
              float v306_data = v570_g ? (v306_data_pre) : (0.0f);
              float v310_data = ir1[1];
              ir1[1] = (v310_data + (v306_data * v270_bc));
              float v316_data = ir1[3];
              ir1[3] = (v316_data + (v306_data * v276_bc));
              float v322_data = ir1[5];
              ir1[5] = (v322_data + (v306_data * v282_bc));
              float v328_data = ir1[7];
              ir1[7] = (v328_data + (v306_data * v288_bc));
              float v334_data = ir1[9];
              ir1[9] = (v334_data + (v306_data * v294_bc));
              float v340_data = ir1[11];
              ir1[11] = (v340_data + (v306_data * v300_bc));
              float v344_data_pre = glb_m1[v570_g ? ((v33_lead + 87)) : (0)];
              float v344_data = v570_g ? (v344_data_pre) : (0.0f);
              float v348_data = ir1[1];
              ir1[1] = (v348_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v354_data = ir1[3];
              ir1[3] = (v354_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v360_data = ir1[5];
              ir1[5] = (v360_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v366_data = ir1[7];
              ir1[7] = (v366_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v372_data = ir1[9];
              ir1[9] = (v372_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v378_data = ir1[11];
              ir1[11] = (v378_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v382_data = glb_m1[(v569_lead + 127)];
              float v384_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v386_data = ir1[0];
              ir1[0] = (v386_data + (v382_data * v384_bc));
              float v390_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v392_data = ir1[2];
              ir1[2] = (v392_data + (v382_data * v390_bc));
              float v396_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v398_data = ir1[4];
              ir1[4] = (v398_data + (v382_data * v396_bc));
              float v402_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v404_data = ir1[6];
              ir1[6] = (v404_data + (v382_data * v402_bc));
              float v408_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v410_data = ir1[8];
              ir1[8] = (v410_data + (v382_data * v408_bc));
              float v414_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v416_data = ir1[10];
              ir1[10] = (v416_data + (v382_data * v414_bc));
              float v420_data_pre = glb_m1[v570_g ? ((v33_lead + 127)) : (0)];
              float v420_data = v570_g ? (v420_data_pre) : (0.0f);
              float v424_data = ir1[1];
              ir1[1] = (v424_data + (v420_data * v384_bc));
              float v430_data = ir1[3];
              ir1[3] = (v430_data + (v420_data * v390_bc));
              float v436_data = ir1[5];
              ir1[5] = (v436_data + (v420_data * v396_bc));
              float v442_data = ir1[7];
              ir1[7] = (v442_data + (v420_data * v402_bc));
              float v448_data = ir1[9];
              ir1[9] = (v448_data + (v420_data * v408_bc));
              float v454_data = ir1[11];
              ir1[11] = (v454_data + (v420_data * v414_bc));
              float v458_data = glb_m1[(v569_lead + 167)];
              float v460_bc = sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v462_data = ir1[0];
              ir1[0] = (v462_data + (v458_data * v460_bc));
              float v466_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v468_data = ir1[2];
              ir1[2] = (v468_data + (v458_data * v466_bc));
              float v472_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v474_data = ir1[4];
              ir1[4] = (v474_data + (v458_data * v472_bc));
              float v478_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v480_data = ir1[6];
              ir1[6] = (v480_data + (v458_data * v478_bc));
              float v484_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v486_data = ir1[8];
              ir1[8] = (v486_data + (v458_data * v484_bc));
              float v490_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v492_data = ir1[10];
              ir1[10] = (v492_data + (v458_data * v490_bc));
              float v496_data_pre = glb_m1[v570_g ? ((v33_lead + 167)) : (0)];
              float v496_data = v570_g ? (v496_data_pre) : (0.0f);
              float v500_data = ir1[1];
              ir1[1] = (v500_data + (v496_data * v460_bc));
              float v506_data = ir1[3];
              ir1[3] = (v506_data + (v496_data * v466_bc));
              float v512_data = ir1[5];
              ir1[5] = (v512_data + (v496_data * v472_bc));
              float v518_data = ir1[7];
              ir1[7] = (v518_data + (v496_data * v478_bc));
              float v524_data = ir1[9];
              ir1[9] = (v524_data + (v496_data * v484_bc));
              float v530_data = ir1[11];
              ir1[11] = (v530_data + (v496_data * v490_bc));
              // r1 = ir1
              #pragma unroll
              for (int32_t v532_n0 = 0; v532_n0 < 1; ++v532_n0) {
                #pragma unroll
                for (int32_t v533_n1 = 0; v533_n1 < 6; ++v533_n1) {
                  int32_t v535_a = v532_n0 + (v533_n1 * 2);
                  float v536_data = ir1[v535_a];
                  r1[v535_a] = v536_data;
                }
              }
              if (v570_g) {
                #pragma unroll
                for (int32_t v538_n1 = 0; v538_n1 < 6; ++v538_n1) {
                  int32_t v540_a = 1 + (v538_n1 * 2);
                  float v541_data = ir1[v540_a];
                  r1[v540_a] = v541_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v542_i0 = 0; v542_i0 < 1; ++v542_i0) {
                int32_t v548_lead = v569_lead + (v542_i0 * 32);
                #pragma unroll
                for (int32_t v543_i1 = 0; v543_i1 < 6; ++v543_i1) {
                  float v546_data = r1[(v542_i0 + (v543_i1 * 2))];
                  glb_m0[(v548_lead + (v543_i1 * 40))] = v546_data;
                }
              }
              if (v570_g) {
                #pragma unroll
                for (int32_t v552_i1 = 0; v552_i1 < 6; ++v552_i1) {
                  float v555_data = r1[(1 + (v552_i1 * 2))];
                  glb_m0[(v33_lead + (v552_i1 * 40))] = v555_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v575_i1 = 0; v575_i1 < 6; ++v575_i1) {
                float v578_data_pre = wrap_glb_m2[v570_g ? ((v569_lead + (v575_i1 * 8))) : (0)];
                float v578_data = v570_g ? (v578_data_pre) : (0.0f);
                r0[v575_i1] = v578_data;
              }
            }
            v565_flagWord = v566_flagWordNext;
            v566_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


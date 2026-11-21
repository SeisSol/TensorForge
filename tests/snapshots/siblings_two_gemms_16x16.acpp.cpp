// === base name ===
kernel_b391a94f5e67b3bf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b391a94f5e67b3bf = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 2};
tensorforge::LaunchConfig launch_config_kernel_b391a94f5e67b3bf(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_b391a94f5e67b3bf(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b391a94f5e67b3bf(size_t numElements0, size_t numElements1, void* streamPtr) {
  (void)numElements0;
  (void)numElements1;
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
void launcher_kernel_b391a94f5e67b3bf(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b391a94f5e67b3bf(numElements0, numElements1, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b391a94f5e67b3bf(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, numElements1, flags0, flags1);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b391a94f5e67b3bf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        //   m3 16×16(16×16) {0..16}×{0..16} strided
        //   m4 16×16(16×16) {0..16}×{0..16} strided
        //   m5 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   siblings
        //   m3[i,j] = m4[i,k] × m5[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0},{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"F","bbox":[[0,0],[16,16]],"name":"m5","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"siblings"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          float r0[16]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v1593_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1593_flagSecondAhead < numElements0) ? v1593_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m1 = &m1[v3_batchId1 * 256 + 0 + m1_extraOffset];
          int32_t v1601_lead = item.get_local_id(2) % 32;
          bool v1602_g = v1601_lead < 16;
          // r0 = load{g>r}(peel_glb_m1);
          #pragma unroll
          for (int32_t v1611_i1 = 0; v1611_i1 < 16; ++v1611_i1) {
            float v1614_data_pre = peel_glb_m1[v1602_g ? ((v1601_lead + (v1611_i1 * 16))) : (0)];
            float v1614_data = v1602_g ? (v1614_data_pre) : (0.0f);
            r0[v1611_i1] = v1614_data;
          }
          uint32_t v1597_flagWord = flagWordFirst;
          uint32_t v1598_flagWordNext = flagWordSecond;
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const float *const __restrict__ wrap_glb_m1 = &m1[v10_batchId1 * 256 + 0 + m1_extraOffset];
            size_t v1621_flagAheadAhead = v10_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1621_flagAheadAhead < numElements0) ? v1621_flagAheadAhead : v10_batchId1)];
            const bool allowed = static_cast<bool>(v1597_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 256 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 256 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 256 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v7_batchId0 * 256 + 0 + m5_extraOffset];
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v34_i1 = 0; v34_i1 < 16; ++v34_i1) {
                float v40_data_pre = glb_m2[v1602_g ? ((v1601_lead + (v34_i1 * 16))) : (0)];
                float v40_data = v1602_g ? (v40_data_pre) : (0.0f);
                r1[v34_i1] = v40_data;
              }
              float r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v44_data = r0[0];
              float v45_data = r1[0];
              float v46_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v51_data = r1[1];
              float v52_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v57_data = r1[2];
              float v58_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v63_data = r1[3];
              float v64_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v69_data = r1[4];
              float v70_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v75_data = r1[5];
              float v76_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v81_data = r1[6];
              float v82_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v87_data = r1[7];
              float v88_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v93_data = r1[8];
              float v94_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v99_data = r1[9];
              float v100_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v105_data = r1[10];
              float v106_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v111_data = r1[11];
              float v112_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v117_data = r1[12];
              float v118_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v123_data = r1[13];
              float v124_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v129_data = r1[14];
              float v130_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v135_data = r1[15];
              float v136_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v140_data = r0[1];
              float v142_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v148_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v154_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v160_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v166_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v172_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v178_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v184_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v190_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v196_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v202_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v208_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v214_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v220_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v226_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v232_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v236_data = r0[2];
              float v238_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v244_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v250_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v256_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v262_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v268_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v274_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v280_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v286_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v292_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v298_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v304_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v310_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v316_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v322_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v328_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v332_data = r0[3];
              float v334_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v340_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v346_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v352_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v358_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v364_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v370_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v376_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v382_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v388_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v394_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v400_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v406_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v412_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v418_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v424_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v428_data = r0[4];
              float v430_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v436_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v442_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v448_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v454_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v460_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v466_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v472_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v478_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v484_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v490_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v496_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v502_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v508_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v514_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v520_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v524_data = r0[5];
              float v526_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v532_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v538_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v544_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v550_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v556_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v562_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v568_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v574_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v580_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v586_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v592_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v598_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v604_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v610_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v616_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v620_data = r0[6];
              float v622_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v628_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v634_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v640_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v646_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v652_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v658_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v664_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v670_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v676_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v682_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v688_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v694_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v700_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v706_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v712_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v716_data = r0[7];
              float v718_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v724_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v730_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v736_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v742_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v748_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v754_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v760_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v766_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v772_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v778_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v784_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v790_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v796_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v802_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v808_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v812_data = r0[8];
              float v814_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v820_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v826_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v832_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v838_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v844_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v850_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v856_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v862_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v868_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v874_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v880_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v886_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v892_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v898_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v904_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v908_data = r0[9];
              float v910_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v916_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v922_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v928_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v934_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v940_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v946_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v952_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v958_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v964_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v970_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v976_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v982_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v988_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v994_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v1000_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v1004_data = r0[10];
              float v1006_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1012_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1018_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1024_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1030_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1036_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1042_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1048_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1054_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1060_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1066_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1072_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1078_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1084_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1090_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1096_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v1100_data = r0[11];
              float v1102_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1108_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1114_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1120_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1126_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1132_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1138_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1144_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1150_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1156_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1162_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1168_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1174_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1180_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1186_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1192_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1196_data = r0[12];
              float v1198_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1204_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1210_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1216_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1222_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1228_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1234_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1240_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1246_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1252_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1258_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1264_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1270_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1276_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1282_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1288_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1292_data = r0[13];
              float v1294_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1300_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1306_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1312_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1318_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1324_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1330_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1336_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1342_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1348_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1354_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1360_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1366_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1372_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1378_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1384_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1388_data = r0[14];
              float v1390_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1393_p = ((((((((((((((v44_data * v46_bc) + (v140_data * v142_bc)) + (v236_data * v238_bc)) + (v332_data * v334_bc)) + (v428_data * v430_bc)) + (v524_data * v526_bc)) + (v620_data * v622_bc)) + (v716_data * v718_bc)) + (v812_data * v814_bc)) + (v908_data * v910_bc)) + (v1004_data * v1006_bc)) + (v1100_data * v1102_bc)) + (v1196_data * v1198_bc)) + (v1292_data * v1294_bc)) + (v1388_data * v1390_bc);
              float v1396_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1399_p = ((((((((((((((v44_data * v52_bc) + (v140_data * v148_bc)) + (v236_data * v244_bc)) + (v332_data * v340_bc)) + (v428_data * v436_bc)) + (v524_data * v532_bc)) + (v620_data * v628_bc)) + (v716_data * v724_bc)) + (v812_data * v820_bc)) + (v908_data * v916_bc)) + (v1004_data * v1012_bc)) + (v1100_data * v1108_bc)) + (v1196_data * v1204_bc)) + (v1292_data * v1300_bc)) + (v1388_data * v1396_bc);
              float v1402_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1405_p = ((((((((((((((v44_data * v58_bc) + (v140_data * v154_bc)) + (v236_data * v250_bc)) + (v332_data * v346_bc)) + (v428_data * v442_bc)) + (v524_data * v538_bc)) + (v620_data * v634_bc)) + (v716_data * v730_bc)) + (v812_data * v826_bc)) + (v908_data * v922_bc)) + (v1004_data * v1018_bc)) + (v1100_data * v1114_bc)) + (v1196_data * v1210_bc)) + (v1292_data * v1306_bc)) + (v1388_data * v1402_bc);
              float v1408_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1411_p = ((((((((((((((v44_data * v64_bc) + (v140_data * v160_bc)) + (v236_data * v256_bc)) + (v332_data * v352_bc)) + (v428_data * v448_bc)) + (v524_data * v544_bc)) + (v620_data * v640_bc)) + (v716_data * v736_bc)) + (v812_data * v832_bc)) + (v908_data * v928_bc)) + (v1004_data * v1024_bc)) + (v1100_data * v1120_bc)) + (v1196_data * v1216_bc)) + (v1292_data * v1312_bc)) + (v1388_data * v1408_bc);
              float v1414_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1417_p = ((((((((((((((v44_data * v70_bc) + (v140_data * v166_bc)) + (v236_data * v262_bc)) + (v332_data * v358_bc)) + (v428_data * v454_bc)) + (v524_data * v550_bc)) + (v620_data * v646_bc)) + (v716_data * v742_bc)) + (v812_data * v838_bc)) + (v908_data * v934_bc)) + (v1004_data * v1030_bc)) + (v1100_data * v1126_bc)) + (v1196_data * v1222_bc)) + (v1292_data * v1318_bc)) + (v1388_data * v1414_bc);
              float v1420_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1423_p = ((((((((((((((v44_data * v76_bc) + (v140_data * v172_bc)) + (v236_data * v268_bc)) + (v332_data * v364_bc)) + (v428_data * v460_bc)) + (v524_data * v556_bc)) + (v620_data * v652_bc)) + (v716_data * v748_bc)) + (v812_data * v844_bc)) + (v908_data * v940_bc)) + (v1004_data * v1036_bc)) + (v1100_data * v1132_bc)) + (v1196_data * v1228_bc)) + (v1292_data * v1324_bc)) + (v1388_data * v1420_bc);
              float v1426_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1429_p = ((((((((((((((v44_data * v82_bc) + (v140_data * v178_bc)) + (v236_data * v274_bc)) + (v332_data * v370_bc)) + (v428_data * v466_bc)) + (v524_data * v562_bc)) + (v620_data * v658_bc)) + (v716_data * v754_bc)) + (v812_data * v850_bc)) + (v908_data * v946_bc)) + (v1004_data * v1042_bc)) + (v1100_data * v1138_bc)) + (v1196_data * v1234_bc)) + (v1292_data * v1330_bc)) + (v1388_data * v1426_bc);
              float v1432_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1435_p = ((((((((((((((v44_data * v88_bc) + (v140_data * v184_bc)) + (v236_data * v280_bc)) + (v332_data * v376_bc)) + (v428_data * v472_bc)) + (v524_data * v568_bc)) + (v620_data * v664_bc)) + (v716_data * v760_bc)) + (v812_data * v856_bc)) + (v908_data * v952_bc)) + (v1004_data * v1048_bc)) + (v1100_data * v1144_bc)) + (v1196_data * v1240_bc)) + (v1292_data * v1336_bc)) + (v1388_data * v1432_bc);
              float v1438_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1441_p = ((((((((((((((v44_data * v94_bc) + (v140_data * v190_bc)) + (v236_data * v286_bc)) + (v332_data * v382_bc)) + (v428_data * v478_bc)) + (v524_data * v574_bc)) + (v620_data * v670_bc)) + (v716_data * v766_bc)) + (v812_data * v862_bc)) + (v908_data * v958_bc)) + (v1004_data * v1054_bc)) + (v1100_data * v1150_bc)) + (v1196_data * v1246_bc)) + (v1292_data * v1342_bc)) + (v1388_data * v1438_bc);
              float v1444_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1447_p = ((((((((((((((v44_data * v100_bc) + (v140_data * v196_bc)) + (v236_data * v292_bc)) + (v332_data * v388_bc)) + (v428_data * v484_bc)) + (v524_data * v580_bc)) + (v620_data * v676_bc)) + (v716_data * v772_bc)) + (v812_data * v868_bc)) + (v908_data * v964_bc)) + (v1004_data * v1060_bc)) + (v1100_data * v1156_bc)) + (v1196_data * v1252_bc)) + (v1292_data * v1348_bc)) + (v1388_data * v1444_bc);
              float v1450_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1453_p = ((((((((((((((v44_data * v106_bc) + (v140_data * v202_bc)) + (v236_data * v298_bc)) + (v332_data * v394_bc)) + (v428_data * v490_bc)) + (v524_data * v586_bc)) + (v620_data * v682_bc)) + (v716_data * v778_bc)) + (v812_data * v874_bc)) + (v908_data * v970_bc)) + (v1004_data * v1066_bc)) + (v1100_data * v1162_bc)) + (v1196_data * v1258_bc)) + (v1292_data * v1354_bc)) + (v1388_data * v1450_bc);
              float v1456_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1459_p = ((((((((((((((v44_data * v112_bc) + (v140_data * v208_bc)) + (v236_data * v304_bc)) + (v332_data * v400_bc)) + (v428_data * v496_bc)) + (v524_data * v592_bc)) + (v620_data * v688_bc)) + (v716_data * v784_bc)) + (v812_data * v880_bc)) + (v908_data * v976_bc)) + (v1004_data * v1072_bc)) + (v1100_data * v1168_bc)) + (v1196_data * v1264_bc)) + (v1292_data * v1360_bc)) + (v1388_data * v1456_bc);
              float v1462_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1465_p = ((((((((((((((v44_data * v118_bc) + (v140_data * v214_bc)) + (v236_data * v310_bc)) + (v332_data * v406_bc)) + (v428_data * v502_bc)) + (v524_data * v598_bc)) + (v620_data * v694_bc)) + (v716_data * v790_bc)) + (v812_data * v886_bc)) + (v908_data * v982_bc)) + (v1004_data * v1078_bc)) + (v1100_data * v1174_bc)) + (v1196_data * v1270_bc)) + (v1292_data * v1366_bc)) + (v1388_data * v1462_bc);
              float v1468_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1471_p = ((((((((((((((v44_data * v124_bc) + (v140_data * v220_bc)) + (v236_data * v316_bc)) + (v332_data * v412_bc)) + (v428_data * v508_bc)) + (v524_data * v604_bc)) + (v620_data * v700_bc)) + (v716_data * v796_bc)) + (v812_data * v892_bc)) + (v908_data * v988_bc)) + (v1004_data * v1084_bc)) + (v1100_data * v1180_bc)) + (v1196_data * v1276_bc)) + (v1292_data * v1372_bc)) + (v1388_data * v1468_bc);
              float v1474_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1477_p = ((((((((((((((v44_data * v130_bc) + (v140_data * v226_bc)) + (v236_data * v322_bc)) + (v332_data * v418_bc)) + (v428_data * v514_bc)) + (v524_data * v610_bc)) + (v620_data * v706_bc)) + (v716_data * v802_bc)) + (v812_data * v898_bc)) + (v908_data * v994_bc)) + (v1004_data * v1090_bc)) + (v1100_data * v1186_bc)) + (v1196_data * v1282_bc)) + (v1292_data * v1378_bc)) + (v1388_data * v1474_bc);
              float v1480_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1483_p = ((((((((((((((v44_data * v136_bc) + (v140_data * v232_bc)) + (v236_data * v328_bc)) + (v332_data * v424_bc)) + (v428_data * v520_bc)) + (v524_data * v616_bc)) + (v620_data * v712_bc)) + (v716_data * v808_bc)) + (v812_data * v904_bc)) + (v908_data * v1000_bc)) + (v1004_data * v1096_bc)) + (v1100_data * v1192_bc)) + (v1196_data * v1288_bc)) + (v1292_data * v1384_bc)) + (v1388_data * v1480_bc);
              float v1484_data = r0[15];
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v1615_i1 = 0; v1615_i1 < 16; ++v1615_i1) {
                float v1618_data_pre = wrap_glb_m1[v1602_g ? ((v1601_lead + (v1615_i1 * 16))) : (0)];
                float v1618_data = v1602_g ? (v1618_data_pre) : (0.0f);
                r0[v1615_i1] = v1618_data;
              }
              float v1486_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[0] = (v1393_p + (v1484_data * v1486_bc));
              float v1492_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[1] = (v1399_p + (v1484_data * v1492_bc));
              float v1498_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[2] = (v1405_p + (v1484_data * v1498_bc));
              float v1504_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[3] = (v1411_p + (v1484_data * v1504_bc));
              float v1510_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[4] = (v1417_p + (v1484_data * v1510_bc));
              float v1516_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[5] = (v1423_p + (v1484_data * v1516_bc));
              float v1522_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[6] = (v1429_p + (v1484_data * v1522_bc));
              float v1528_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[7] = (v1435_p + (v1484_data * v1528_bc));
              float v1534_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[8] = (v1441_p + (v1484_data * v1534_bc));
              float v1540_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[9] = (v1447_p + (v1484_data * v1540_bc));
              float v1546_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[10] = (v1453_p + (v1484_data * v1546_bc));
              float v1552_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[11] = (v1459_p + (v1484_data * v1552_bc));
              float v1558_bc = sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[12] = (v1465_p + (v1484_data * v1558_bc));
              float v1564_bc = sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[13] = (v1471_p + (v1484_data * v1564_bc));
              float v1570_bc = sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[14] = (v1477_p + (v1484_data * v1570_bc));
              float v1576_bc = sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[15] = (v1483_p + (v1484_data * v1576_bc));
              // r2 = ir2
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1585_i1 = 0; v1585_i1 < 16; ++v1585_i1) {
                float v1587_data = ir2[v1585_i1];
                if (v1602_g) {
                  glb_m0[(v1601_lead + (v1585_i1 * 16))] = v1587_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v1607_i1 = 0; v1607_i1 < 16; ++v1607_i1) {
                float v1610_data_pre = wrap_glb_m1[v1602_g ? ((v1601_lead + (v1607_i1 * 16))) : (0)];
                float v1610_data = v1602_g ? (v1610_data_pre) : (0.0f);
                r0[v1607_i1] = v1610_data;
              }
            }
            v1597_flagWord = v1598_flagWordNext;
            v1598_flagWordNext = flagWordAhead;
          }
        }
        {
          size_t v1629_batchId1 = (((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) + numElements0) % (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements1) ? ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) + numElements0) % (item.get_group_range(2) * item.get_group().get_local_range(1)) : 0u;
          item.barrier();
          float r0[16]{};
          const uint32_t flagWordFirst = flags1 == nullptr ? 1 : flags1[v1629_batchId1];
          size_t v3219_flagSecondAhead = v1629_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags1 == nullptr ? 1 : flags1[((v3219_flagSecondAhead < numElements1) ? v3219_flagSecondAhead : v1629_batchId1)];
          const float *const __restrict__ peel_glb_m4 = &m4[v1629_batchId1 * 256 + 0 + m4_extraOffset];
          int32_t v3227_lead = item.get_local_id(2) % 32;
          bool v3228_g = v3227_lead < 16;
          // r0 = load{g>r}(peel_glb_m4);
          #pragma unroll
          for (int32_t v3237_i1 = 0; v3237_i1 < 16; ++v3237_i1) {
            float v3240_data_pre = peel_glb_m4[v3228_g ? ((v3227_lead + (v3237_i1 * 16))) : (0)];
            float v3240_data = v3228_g ? (v3240_data_pre) : (0.0f);
            r0[v3237_i1] = v3240_data;
          }
          uint32_t v3223_flagWord = flagWordFirst;
          uint32_t v3224_flagWordNext = flagWordSecond;
          for (size_t v1633_batchId0 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) + numElements0) % (item.get_group_range(2) * item.get_group().get_local_range(1)); v1633_batchId0 < numElements1; v1633_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v1634_ahead1 = v1633_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v1636_batchId1 = (v1634_ahead1 < numElements1) ? v1634_ahead1 : v1633_batchId0;
            const float *const __restrict__ wrap_glb_m4 = &m4[v1636_batchId1 * 256 + 0 + m4_extraOffset];
            size_t v3247_flagAheadAhead = v1636_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags1 == nullptr ? 1 : flags1[((v3247_flagAheadAhead < numElements1) ? v3247_flagAheadAhead : v1636_batchId1)];
            const bool allowed = static_cast<bool>(v3223_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v1633_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1633_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v1633_batchId0 * 256 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v1633_batchId0 * 256 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v1633_batchId0 * 256 + 0 + m5_extraOffset];
              float r1[16]{};
              // r1 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v1660_i1 = 0; v1660_i1 < 16; ++v1660_i1) {
                float v1666_data_pre = glb_m5[v3228_g ? ((v3227_lead + (v1660_i1 * 16))) : (0)];
                float v1666_data = v3228_g ? (v1666_data_pre) : (0.0f);
                r1[v1660_i1] = v1666_data;
              }
              float r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v1670_data = r0[0];
              float v1671_data = r1[0];
              float v1672_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1677_data = r1[1];
              float v1678_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1683_data = r1[2];
              float v1684_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1689_data = r1[3];
              float v1690_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1695_data = r1[4];
              float v1696_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1701_data = r1[5];
              float v1702_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1707_data = r1[6];
              float v1708_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1713_data = r1[7];
              float v1714_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1719_data = r1[8];
              float v1720_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1725_data = r1[9];
              float v1726_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1731_data = r1[10];
              float v1732_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1737_data = r1[11];
              float v1738_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1743_data = r1[12];
              float v1744_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1749_data = r1[13];
              float v1750_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1755_data = r1[14];
              float v1756_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1761_data = r1[15];
              float v1762_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v1766_data = r0[1];
              float v1768_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1774_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1780_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1786_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1792_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1798_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1804_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1810_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1816_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1822_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1828_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1834_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1840_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1846_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1852_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1858_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v1862_data = r0[2];
              float v1864_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1870_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1876_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1882_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1888_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1894_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1900_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1906_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1912_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1918_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1924_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1930_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1936_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1942_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1948_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1954_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v1958_data = r0[3];
              float v1960_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v1966_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v1972_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v1978_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v1984_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v1990_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v1996_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2002_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2008_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2014_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2020_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2026_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2032_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2038_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2044_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2050_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v2054_data = r0[4];
              float v2056_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2062_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2068_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2074_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2080_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2086_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2092_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2098_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2104_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2110_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2116_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2122_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2128_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2134_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2140_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2146_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v2150_data = r0[5];
              float v2152_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2158_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2164_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2170_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2176_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2182_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2188_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2194_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2200_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2206_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2212_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2218_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2224_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2230_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2236_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2242_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v2246_data = r0[6];
              float v2248_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2254_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2260_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2266_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2272_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2278_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2284_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2290_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2296_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2302_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2308_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2314_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2320_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2326_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2332_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2338_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v2342_data = r0[7];
              float v2344_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2350_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2356_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2362_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2368_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2374_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2380_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2386_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2392_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2398_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2404_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2410_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2416_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2422_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2428_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2434_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v2438_data = r0[8];
              float v2440_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2446_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2452_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2458_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2464_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2470_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2476_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2482_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2488_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2494_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2500_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2506_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2512_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2518_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2524_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2530_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v2534_data = r0[9];
              float v2536_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2542_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2548_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2554_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2560_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2566_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2572_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2578_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2584_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2590_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2596_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2602_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2608_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2614_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2620_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2626_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v2630_data = r0[10];
              float v2632_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2638_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2644_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2650_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2656_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2662_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2668_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2674_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2680_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2686_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2692_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2698_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2704_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2710_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2716_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2722_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v2726_data = r0[11];
              float v2728_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2734_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2740_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2746_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2752_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2758_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2764_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2770_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2776_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2782_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2788_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2794_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2800_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2806_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2812_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2818_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v2822_data = r0[12];
              float v2824_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2830_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2836_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2842_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2848_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2854_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2860_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2866_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2872_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2878_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2884_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2890_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2896_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2902_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2908_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2914_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v2918_data = r0[13];
              float v2920_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2926_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2932_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2938_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2944_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2950_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2956_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2962_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2968_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2974_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2980_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2986_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2992_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v2998_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v3004_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v3010_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v3014_data = r0[14];
              float v3016_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3019_p = ((((((((((((((v1670_data * v1672_bc) + (v1766_data * v1768_bc)) + (v1862_data * v1864_bc)) + (v1958_data * v1960_bc)) + (v2054_data * v2056_bc)) + (v2150_data * v2152_bc)) + (v2246_data * v2248_bc)) + (v2342_data * v2344_bc)) + (v2438_data * v2440_bc)) + (v2534_data * v2536_bc)) + (v2630_data * v2632_bc)) + (v2726_data * v2728_bc)) + (v2822_data * v2824_bc)) + (v2918_data * v2920_bc)) + (v3014_data * v3016_bc);
              float v3022_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3025_p = ((((((((((((((v1670_data * v1678_bc) + (v1766_data * v1774_bc)) + (v1862_data * v1870_bc)) + (v1958_data * v1966_bc)) + (v2054_data * v2062_bc)) + (v2150_data * v2158_bc)) + (v2246_data * v2254_bc)) + (v2342_data * v2350_bc)) + (v2438_data * v2446_bc)) + (v2534_data * v2542_bc)) + (v2630_data * v2638_bc)) + (v2726_data * v2734_bc)) + (v2822_data * v2830_bc)) + (v2918_data * v2926_bc)) + (v3014_data * v3022_bc);
              float v3028_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3031_p = ((((((((((((((v1670_data * v1684_bc) + (v1766_data * v1780_bc)) + (v1862_data * v1876_bc)) + (v1958_data * v1972_bc)) + (v2054_data * v2068_bc)) + (v2150_data * v2164_bc)) + (v2246_data * v2260_bc)) + (v2342_data * v2356_bc)) + (v2438_data * v2452_bc)) + (v2534_data * v2548_bc)) + (v2630_data * v2644_bc)) + (v2726_data * v2740_bc)) + (v2822_data * v2836_bc)) + (v2918_data * v2932_bc)) + (v3014_data * v3028_bc);
              float v3034_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3037_p = ((((((((((((((v1670_data * v1690_bc) + (v1766_data * v1786_bc)) + (v1862_data * v1882_bc)) + (v1958_data * v1978_bc)) + (v2054_data * v2074_bc)) + (v2150_data * v2170_bc)) + (v2246_data * v2266_bc)) + (v2342_data * v2362_bc)) + (v2438_data * v2458_bc)) + (v2534_data * v2554_bc)) + (v2630_data * v2650_bc)) + (v2726_data * v2746_bc)) + (v2822_data * v2842_bc)) + (v2918_data * v2938_bc)) + (v3014_data * v3034_bc);
              float v3040_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3043_p = ((((((((((((((v1670_data * v1696_bc) + (v1766_data * v1792_bc)) + (v1862_data * v1888_bc)) + (v1958_data * v1984_bc)) + (v2054_data * v2080_bc)) + (v2150_data * v2176_bc)) + (v2246_data * v2272_bc)) + (v2342_data * v2368_bc)) + (v2438_data * v2464_bc)) + (v2534_data * v2560_bc)) + (v2630_data * v2656_bc)) + (v2726_data * v2752_bc)) + (v2822_data * v2848_bc)) + (v2918_data * v2944_bc)) + (v3014_data * v3040_bc);
              float v3046_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3049_p = ((((((((((((((v1670_data * v1702_bc) + (v1766_data * v1798_bc)) + (v1862_data * v1894_bc)) + (v1958_data * v1990_bc)) + (v2054_data * v2086_bc)) + (v2150_data * v2182_bc)) + (v2246_data * v2278_bc)) + (v2342_data * v2374_bc)) + (v2438_data * v2470_bc)) + (v2534_data * v2566_bc)) + (v2630_data * v2662_bc)) + (v2726_data * v2758_bc)) + (v2822_data * v2854_bc)) + (v2918_data * v2950_bc)) + (v3014_data * v3046_bc);
              float v3052_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3055_p = ((((((((((((((v1670_data * v1708_bc) + (v1766_data * v1804_bc)) + (v1862_data * v1900_bc)) + (v1958_data * v1996_bc)) + (v2054_data * v2092_bc)) + (v2150_data * v2188_bc)) + (v2246_data * v2284_bc)) + (v2342_data * v2380_bc)) + (v2438_data * v2476_bc)) + (v2534_data * v2572_bc)) + (v2630_data * v2668_bc)) + (v2726_data * v2764_bc)) + (v2822_data * v2860_bc)) + (v2918_data * v2956_bc)) + (v3014_data * v3052_bc);
              float v3058_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3061_p = ((((((((((((((v1670_data * v1714_bc) + (v1766_data * v1810_bc)) + (v1862_data * v1906_bc)) + (v1958_data * v2002_bc)) + (v2054_data * v2098_bc)) + (v2150_data * v2194_bc)) + (v2246_data * v2290_bc)) + (v2342_data * v2386_bc)) + (v2438_data * v2482_bc)) + (v2534_data * v2578_bc)) + (v2630_data * v2674_bc)) + (v2726_data * v2770_bc)) + (v2822_data * v2866_bc)) + (v2918_data * v2962_bc)) + (v3014_data * v3058_bc);
              float v3064_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3067_p = ((((((((((((((v1670_data * v1720_bc) + (v1766_data * v1816_bc)) + (v1862_data * v1912_bc)) + (v1958_data * v2008_bc)) + (v2054_data * v2104_bc)) + (v2150_data * v2200_bc)) + (v2246_data * v2296_bc)) + (v2342_data * v2392_bc)) + (v2438_data * v2488_bc)) + (v2534_data * v2584_bc)) + (v2630_data * v2680_bc)) + (v2726_data * v2776_bc)) + (v2822_data * v2872_bc)) + (v2918_data * v2968_bc)) + (v3014_data * v3064_bc);
              float v3070_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3073_p = ((((((((((((((v1670_data * v1726_bc) + (v1766_data * v1822_bc)) + (v1862_data * v1918_bc)) + (v1958_data * v2014_bc)) + (v2054_data * v2110_bc)) + (v2150_data * v2206_bc)) + (v2246_data * v2302_bc)) + (v2342_data * v2398_bc)) + (v2438_data * v2494_bc)) + (v2534_data * v2590_bc)) + (v2630_data * v2686_bc)) + (v2726_data * v2782_bc)) + (v2822_data * v2878_bc)) + (v2918_data * v2974_bc)) + (v3014_data * v3070_bc);
              float v3076_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3079_p = ((((((((((((((v1670_data * v1732_bc) + (v1766_data * v1828_bc)) + (v1862_data * v1924_bc)) + (v1958_data * v2020_bc)) + (v2054_data * v2116_bc)) + (v2150_data * v2212_bc)) + (v2246_data * v2308_bc)) + (v2342_data * v2404_bc)) + (v2438_data * v2500_bc)) + (v2534_data * v2596_bc)) + (v2630_data * v2692_bc)) + (v2726_data * v2788_bc)) + (v2822_data * v2884_bc)) + (v2918_data * v2980_bc)) + (v3014_data * v3076_bc);
              float v3082_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3085_p = ((((((((((((((v1670_data * v1738_bc) + (v1766_data * v1834_bc)) + (v1862_data * v1930_bc)) + (v1958_data * v2026_bc)) + (v2054_data * v2122_bc)) + (v2150_data * v2218_bc)) + (v2246_data * v2314_bc)) + (v2342_data * v2410_bc)) + (v2438_data * v2506_bc)) + (v2534_data * v2602_bc)) + (v2630_data * v2698_bc)) + (v2726_data * v2794_bc)) + (v2822_data * v2890_bc)) + (v2918_data * v2986_bc)) + (v3014_data * v3082_bc);
              float v3088_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3091_p = ((((((((((((((v1670_data * v1744_bc) + (v1766_data * v1840_bc)) + (v1862_data * v1936_bc)) + (v1958_data * v2032_bc)) + (v2054_data * v2128_bc)) + (v2150_data * v2224_bc)) + (v2246_data * v2320_bc)) + (v2342_data * v2416_bc)) + (v2438_data * v2512_bc)) + (v2534_data * v2608_bc)) + (v2630_data * v2704_bc)) + (v2726_data * v2800_bc)) + (v2822_data * v2896_bc)) + (v2918_data * v2992_bc)) + (v3014_data * v3088_bc);
              float v3094_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3097_p = ((((((((((((((v1670_data * v1750_bc) + (v1766_data * v1846_bc)) + (v1862_data * v1942_bc)) + (v1958_data * v2038_bc)) + (v2054_data * v2134_bc)) + (v2150_data * v2230_bc)) + (v2246_data * v2326_bc)) + (v2342_data * v2422_bc)) + (v2438_data * v2518_bc)) + (v2534_data * v2614_bc)) + (v2630_data * v2710_bc)) + (v2726_data * v2806_bc)) + (v2822_data * v2902_bc)) + (v2918_data * v2998_bc)) + (v3014_data * v3094_bc);
              float v3100_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3103_p = ((((((((((((((v1670_data * v1756_bc) + (v1766_data * v1852_bc)) + (v1862_data * v1948_bc)) + (v1958_data * v2044_bc)) + (v2054_data * v2140_bc)) + (v2150_data * v2236_bc)) + (v2246_data * v2332_bc)) + (v2342_data * v2428_bc)) + (v2438_data * v2524_bc)) + (v2534_data * v2620_bc)) + (v2630_data * v2716_bc)) + (v2726_data * v2812_bc)) + (v2822_data * v2908_bc)) + (v2918_data * v3004_bc)) + (v3014_data * v3100_bc);
              float v3106_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v3109_p = ((((((((((((((v1670_data * v1762_bc) + (v1766_data * v1858_bc)) + (v1862_data * v1954_bc)) + (v1958_data * v2050_bc)) + (v2054_data * v2146_bc)) + (v2150_data * v2242_bc)) + (v2246_data * v2338_bc)) + (v2342_data * v2434_bc)) + (v2438_data * v2530_bc)) + (v2534_data * v2626_bc)) + (v2630_data * v2722_bc)) + (v2726_data * v2818_bc)) + (v2822_data * v2914_bc)) + (v2918_data * v3010_bc)) + (v3014_data * v3106_bc);
              float v3110_data = r0[15];
              // r0 = load{g>r}(wrap_glb_m4);
              #pragma unroll
              for (int32_t v3241_i1 = 0; v3241_i1 < 16; ++v3241_i1) {
                float v3244_data_pre = wrap_glb_m4[v3228_g ? ((v3227_lead + (v3241_i1 * 16))) : (0)];
                float v3244_data = v3228_g ? (v3244_data_pre) : (0.0f);
                r0[v3241_i1] = v3244_data;
              }
              float v3112_bc = sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[0] = (v3019_p + (v3110_data * v3112_bc));
              float v3118_bc = sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[1] = (v3025_p + (v3110_data * v3118_bc));
              float v3124_bc = sycl::select_from_group(item.get_sub_group(), v1683_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[2] = (v3031_p + (v3110_data * v3124_bc));
              float v3130_bc = sycl::select_from_group(item.get_sub_group(), v1689_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[3] = (v3037_p + (v3110_data * v3130_bc));
              float v3136_bc = sycl::select_from_group(item.get_sub_group(), v1695_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[4] = (v3043_p + (v3110_data * v3136_bc));
              float v3142_bc = sycl::select_from_group(item.get_sub_group(), v1701_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[5] = (v3049_p + (v3110_data * v3142_bc));
              float v3148_bc = sycl::select_from_group(item.get_sub_group(), v1707_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[6] = (v3055_p + (v3110_data * v3148_bc));
              float v3154_bc = sycl::select_from_group(item.get_sub_group(), v1713_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[7] = (v3061_p + (v3110_data * v3154_bc));
              float v3160_bc = sycl::select_from_group(item.get_sub_group(), v1719_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[8] = (v3067_p + (v3110_data * v3160_bc));
              float v3166_bc = sycl::select_from_group(item.get_sub_group(), v1725_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[9] = (v3073_p + (v3110_data * v3166_bc));
              float v3172_bc = sycl::select_from_group(item.get_sub_group(), v1731_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[10] = (v3079_p + (v3110_data * v3172_bc));
              float v3178_bc = sycl::select_from_group(item.get_sub_group(), v1737_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[11] = (v3085_p + (v3110_data * v3178_bc));
              float v3184_bc = sycl::select_from_group(item.get_sub_group(), v1743_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[12] = (v3091_p + (v3110_data * v3184_bc));
              float v3190_bc = sycl::select_from_group(item.get_sub_group(), v1749_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[13] = (v3097_p + (v3110_data * v3190_bc));
              float v3196_bc = sycl::select_from_group(item.get_sub_group(), v1755_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[14] = (v3103_p + (v3110_data * v3196_bc));
              float v3202_bc = sycl::select_from_group(item.get_sub_group(), v1761_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              ir2[15] = (v3109_p + (v3110_data * v3202_bc));
              // r2 = ir2
              // glb_m3 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v3211_i1 = 0; v3211_i1 < 16; ++v3211_i1) {
                float v3213_data = ir2[v3211_i1];
                if (v3228_g) {
                  glb_m3[(v3227_lead + (v3211_i1 * 16))] = v3213_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m4);
              #pragma unroll
              for (int32_t v3233_i1 = 0; v3233_i1 < 16; ++v3233_i1) {
                float v3236_data_pre = wrap_glb_m4[v3228_g ? ((v3227_lead + (v3233_i1 * 16))) : (0)];
                float v3236_data = v3228_g ? (v3236_data_pre) : (0.0f);
                r0[v3233_i1] = v3236_data;
              }
            }
            v3223_flagWord = v3224_flagWordNext;
            v3224_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


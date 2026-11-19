// === base name ===
kernel_ce4a93b81dffe6e0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ce4a93b81dffe6e0 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ce4a93b81dffe6e0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ce4a93b81dffe6e0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ce4a93b81dffe6e0(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_ce4a93b81dffe6e0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ce4a93b81dffe6e0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ce4a93b81dffe6e0(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2, m2_extraOffset, m2_slot, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ce4a93b81dffe6e0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided(stride=300, offset=8)
        //   m1 16×16(16×16) {0..16}×{0..16} indexed(stride=runtime)
        //   m2 16×16(16×16) {0..16}×{0..16} indexed_pointer(offset=4, slots=4, slot=runtime)
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided(stride=300, offset=8)","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided(stride=300, offset=8)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          float r0[16]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v1591_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1591_flagSecondAhead < numElements0) ? v1591_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
          const float *const __restrict__ peel_glb_m1 = &m1[peel_elementIndex * m1_stride + 0 + m1_extraOffset];
          int32_t v1602_lead = item.get_local_id(2) % 16;
          // r0 = load{g>r}(peel_glb_m1);
          #pragma unroll
          for (int32_t v1615_i0 = 0; v1615_i0 < 1; ++v1615_i0) {
            int32_t v1617_lead = v1602_lead + (v1615_i0 * 16);
            #pragma unroll
            for (int32_t v1618_i1 = 0; v1618_i1 < 16; ++v1618_i1) {
              float v1621_data_pre = peel_glb_m1[allowed_peel ? ((v1617_lead + (v1618_i1 * 16))) : (0)];
              float v1621_data = allowed_peel ? (v1621_data_pre) : (0.0f);
              r0[(v1615_i0 + v1618_i1)] = v1621_data;
            }
          }
          uint32_t v1595_flagWord = flagWordFirst;
          uint32_t v1596_flagWordNext = flagWordSecond;
          size_t v1638_elementIndex = peel_elementIndex;
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v10_batchId1]);
            const float *const __restrict__ wrap_glb_m1 = &m1[wrap_elementIndex * m1_stride + 0 + m1_extraOffset];
            size_t v1633_flagAheadAhead = v10_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1633_flagAheadAhead < numElements0) ? v1633_flagAheadAhead : v10_batchId1)];
            const bool allowed = static_cast<bool>(v1595_flagWord);
            const bool allowed_next = static_cast<bool>(v1596_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 300 + 0 + 8 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v1638_elementIndex * 4 + m2_slot][0 + 4 + m2_extraOffset];
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v35_lead = v1602_lead + (v32_i0 * 16);
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 16; ++v33_i1) {
                  float v38_data = glb_m2[(v35_lead + (v33_i1 * 16))];
                  r1[(v32_i0 + v33_i1)] = v38_data;
                }
              }
              float r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v42_data = r0[0];
              float v43_data = r1[0];
              float v44_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v49_data = r1[1];
              float v50_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v55_data = r1[2];
              float v56_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v61_data = r1[3];
              float v62_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v67_data = r1[4];
              float v68_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v73_data = r1[5];
              float v74_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v79_data = r1[6];
              float v80_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v85_data = r1[7];
              float v86_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v91_data = r1[8];
              float v92_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v97_data = r1[9];
              float v98_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v103_data = r1[10];
              float v104_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v109_data = r1[11];
              float v110_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v115_data = r1[12];
              float v116_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v121_data = r1[13];
              float v122_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v127_data = r1[14];
              float v128_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v133_data = r1[15];
              float v134_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v138_data = r0[1];
              float v140_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v146_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v152_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v158_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v164_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v170_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v176_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v182_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v188_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v194_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v200_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v206_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v212_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v218_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v224_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v230_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v234_data = r0[2];
              float v236_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v242_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v248_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v254_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v260_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v266_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v272_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v278_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v284_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v290_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v296_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v302_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v308_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v314_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v320_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v326_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v330_data = r0[3];
              float v332_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v338_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v344_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v350_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v356_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v362_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v368_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v374_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v380_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v386_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v392_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v398_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v404_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v410_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v416_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v422_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v426_data = r0[4];
              float v428_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v434_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v440_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v446_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v452_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v458_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v464_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v470_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v476_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v482_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v488_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v494_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v500_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v506_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v512_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v518_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v522_data = r0[5];
              float v524_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v530_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v536_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v542_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v548_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v554_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v560_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v566_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v572_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v578_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v584_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v590_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v596_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v602_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v608_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v614_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v618_data = r0[6];
              float v620_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v626_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v632_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v638_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v644_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v650_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v656_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v662_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v668_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v674_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v680_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v686_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v692_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v698_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v704_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v710_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v714_data = r0[7];
              float v716_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v722_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v728_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v734_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v740_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v746_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v752_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v758_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v764_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v770_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v776_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v782_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v788_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v794_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v800_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v806_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v810_data = r0[8];
              float v812_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v818_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v824_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v830_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v836_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v842_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v848_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v854_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v860_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v866_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v872_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v878_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v884_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v890_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v896_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v902_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v906_data = r0[9];
              float v908_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v914_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v920_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v926_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v932_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v938_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v944_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v950_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v956_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v962_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v968_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v974_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v980_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v986_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v992_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v998_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v1002_data = r0[10];
              float v1004_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1010_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1016_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1022_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1028_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1034_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1040_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1046_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1052_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1058_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1064_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1070_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1076_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1082_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1088_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1094_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1098_data = r0[11];
              float v1100_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1106_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1112_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1118_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1124_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1130_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1136_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1142_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1148_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1154_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1160_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1166_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1172_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1178_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1184_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1190_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1194_data = r0[12];
              float v1196_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1202_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1208_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1214_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1220_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1226_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1232_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1238_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1244_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1250_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1256_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1262_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1268_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1274_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1280_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1286_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1290_data = r0[13];
              float v1292_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1298_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1304_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1310_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1316_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1322_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1328_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1334_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1340_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1346_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1352_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1358_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1364_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1370_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1376_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1382_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1386_data = r0[14];
              float v1388_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1391_p = ((((((((((((((v42_data * v44_bc) + (v138_data * v140_bc)) + (v234_data * v236_bc)) + (v330_data * v332_bc)) + (v426_data * v428_bc)) + (v522_data * v524_bc)) + (v618_data * v620_bc)) + (v714_data * v716_bc)) + (v810_data * v812_bc)) + (v906_data * v908_bc)) + (v1002_data * v1004_bc)) + (v1098_data * v1100_bc)) + (v1194_data * v1196_bc)) + (v1290_data * v1292_bc)) + (v1386_data * v1388_bc);
              float v1394_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1397_p = ((((((((((((((v42_data * v50_bc) + (v138_data * v146_bc)) + (v234_data * v242_bc)) + (v330_data * v338_bc)) + (v426_data * v434_bc)) + (v522_data * v530_bc)) + (v618_data * v626_bc)) + (v714_data * v722_bc)) + (v810_data * v818_bc)) + (v906_data * v914_bc)) + (v1002_data * v1010_bc)) + (v1098_data * v1106_bc)) + (v1194_data * v1202_bc)) + (v1290_data * v1298_bc)) + (v1386_data * v1394_bc);
              float v1400_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1403_p = ((((((((((((((v42_data * v56_bc) + (v138_data * v152_bc)) + (v234_data * v248_bc)) + (v330_data * v344_bc)) + (v426_data * v440_bc)) + (v522_data * v536_bc)) + (v618_data * v632_bc)) + (v714_data * v728_bc)) + (v810_data * v824_bc)) + (v906_data * v920_bc)) + (v1002_data * v1016_bc)) + (v1098_data * v1112_bc)) + (v1194_data * v1208_bc)) + (v1290_data * v1304_bc)) + (v1386_data * v1400_bc);
              float v1406_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1409_p = ((((((((((((((v42_data * v62_bc) + (v138_data * v158_bc)) + (v234_data * v254_bc)) + (v330_data * v350_bc)) + (v426_data * v446_bc)) + (v522_data * v542_bc)) + (v618_data * v638_bc)) + (v714_data * v734_bc)) + (v810_data * v830_bc)) + (v906_data * v926_bc)) + (v1002_data * v1022_bc)) + (v1098_data * v1118_bc)) + (v1194_data * v1214_bc)) + (v1290_data * v1310_bc)) + (v1386_data * v1406_bc);
              float v1412_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1415_p = ((((((((((((((v42_data * v68_bc) + (v138_data * v164_bc)) + (v234_data * v260_bc)) + (v330_data * v356_bc)) + (v426_data * v452_bc)) + (v522_data * v548_bc)) + (v618_data * v644_bc)) + (v714_data * v740_bc)) + (v810_data * v836_bc)) + (v906_data * v932_bc)) + (v1002_data * v1028_bc)) + (v1098_data * v1124_bc)) + (v1194_data * v1220_bc)) + (v1290_data * v1316_bc)) + (v1386_data * v1412_bc);
              float v1418_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1421_p = ((((((((((((((v42_data * v74_bc) + (v138_data * v170_bc)) + (v234_data * v266_bc)) + (v330_data * v362_bc)) + (v426_data * v458_bc)) + (v522_data * v554_bc)) + (v618_data * v650_bc)) + (v714_data * v746_bc)) + (v810_data * v842_bc)) + (v906_data * v938_bc)) + (v1002_data * v1034_bc)) + (v1098_data * v1130_bc)) + (v1194_data * v1226_bc)) + (v1290_data * v1322_bc)) + (v1386_data * v1418_bc);
              float v1424_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1427_p = ((((((((((((((v42_data * v80_bc) + (v138_data * v176_bc)) + (v234_data * v272_bc)) + (v330_data * v368_bc)) + (v426_data * v464_bc)) + (v522_data * v560_bc)) + (v618_data * v656_bc)) + (v714_data * v752_bc)) + (v810_data * v848_bc)) + (v906_data * v944_bc)) + (v1002_data * v1040_bc)) + (v1098_data * v1136_bc)) + (v1194_data * v1232_bc)) + (v1290_data * v1328_bc)) + (v1386_data * v1424_bc);
              float v1430_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1433_p = ((((((((((((((v42_data * v86_bc) + (v138_data * v182_bc)) + (v234_data * v278_bc)) + (v330_data * v374_bc)) + (v426_data * v470_bc)) + (v522_data * v566_bc)) + (v618_data * v662_bc)) + (v714_data * v758_bc)) + (v810_data * v854_bc)) + (v906_data * v950_bc)) + (v1002_data * v1046_bc)) + (v1098_data * v1142_bc)) + (v1194_data * v1238_bc)) + (v1290_data * v1334_bc)) + (v1386_data * v1430_bc);
              float v1436_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1439_p = ((((((((((((((v42_data * v92_bc) + (v138_data * v188_bc)) + (v234_data * v284_bc)) + (v330_data * v380_bc)) + (v426_data * v476_bc)) + (v522_data * v572_bc)) + (v618_data * v668_bc)) + (v714_data * v764_bc)) + (v810_data * v860_bc)) + (v906_data * v956_bc)) + (v1002_data * v1052_bc)) + (v1098_data * v1148_bc)) + (v1194_data * v1244_bc)) + (v1290_data * v1340_bc)) + (v1386_data * v1436_bc);
              float v1442_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1445_p = ((((((((((((((v42_data * v98_bc) + (v138_data * v194_bc)) + (v234_data * v290_bc)) + (v330_data * v386_bc)) + (v426_data * v482_bc)) + (v522_data * v578_bc)) + (v618_data * v674_bc)) + (v714_data * v770_bc)) + (v810_data * v866_bc)) + (v906_data * v962_bc)) + (v1002_data * v1058_bc)) + (v1098_data * v1154_bc)) + (v1194_data * v1250_bc)) + (v1290_data * v1346_bc)) + (v1386_data * v1442_bc);
              float v1448_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1451_p = ((((((((((((((v42_data * v104_bc) + (v138_data * v200_bc)) + (v234_data * v296_bc)) + (v330_data * v392_bc)) + (v426_data * v488_bc)) + (v522_data * v584_bc)) + (v618_data * v680_bc)) + (v714_data * v776_bc)) + (v810_data * v872_bc)) + (v906_data * v968_bc)) + (v1002_data * v1064_bc)) + (v1098_data * v1160_bc)) + (v1194_data * v1256_bc)) + (v1290_data * v1352_bc)) + (v1386_data * v1448_bc);
              float v1454_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1457_p = ((((((((((((((v42_data * v110_bc) + (v138_data * v206_bc)) + (v234_data * v302_bc)) + (v330_data * v398_bc)) + (v426_data * v494_bc)) + (v522_data * v590_bc)) + (v618_data * v686_bc)) + (v714_data * v782_bc)) + (v810_data * v878_bc)) + (v906_data * v974_bc)) + (v1002_data * v1070_bc)) + (v1098_data * v1166_bc)) + (v1194_data * v1262_bc)) + (v1290_data * v1358_bc)) + (v1386_data * v1454_bc);
              float v1460_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1463_p = ((((((((((((((v42_data * v116_bc) + (v138_data * v212_bc)) + (v234_data * v308_bc)) + (v330_data * v404_bc)) + (v426_data * v500_bc)) + (v522_data * v596_bc)) + (v618_data * v692_bc)) + (v714_data * v788_bc)) + (v810_data * v884_bc)) + (v906_data * v980_bc)) + (v1002_data * v1076_bc)) + (v1098_data * v1172_bc)) + (v1194_data * v1268_bc)) + (v1290_data * v1364_bc)) + (v1386_data * v1460_bc);
              float v1466_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1469_p = ((((((((((((((v42_data * v122_bc) + (v138_data * v218_bc)) + (v234_data * v314_bc)) + (v330_data * v410_bc)) + (v426_data * v506_bc)) + (v522_data * v602_bc)) + (v618_data * v698_bc)) + (v714_data * v794_bc)) + (v810_data * v890_bc)) + (v906_data * v986_bc)) + (v1002_data * v1082_bc)) + (v1098_data * v1178_bc)) + (v1194_data * v1274_bc)) + (v1290_data * v1370_bc)) + (v1386_data * v1466_bc);
              float v1472_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1475_p = ((((((((((((((v42_data * v128_bc) + (v138_data * v224_bc)) + (v234_data * v320_bc)) + (v330_data * v416_bc)) + (v426_data * v512_bc)) + (v522_data * v608_bc)) + (v618_data * v704_bc)) + (v714_data * v800_bc)) + (v810_data * v896_bc)) + (v906_data * v992_bc)) + (v1002_data * v1088_bc)) + (v1098_data * v1184_bc)) + (v1194_data * v1280_bc)) + (v1290_data * v1376_bc)) + (v1386_data * v1472_bc);
              float v1478_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1481_p = ((((((((((((((v42_data * v134_bc) + (v138_data * v230_bc)) + (v234_data * v326_bc)) + (v330_data * v422_bc)) + (v426_data * v518_bc)) + (v522_data * v614_bc)) + (v618_data * v710_bc)) + (v714_data * v806_bc)) + (v810_data * v902_bc)) + (v906_data * v998_bc)) + (v1002_data * v1094_bc)) + (v1098_data * v1190_bc)) + (v1194_data * v1286_bc)) + (v1290_data * v1382_bc)) + (v1386_data * v1478_bc);
              float v1482_data = r0[15];
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v1623_i0 = 0; v1623_i0 < 1; ++v1623_i0) {
                int32_t v1625_lead = v1602_lead + (v1623_i0 * 16);
                #pragma unroll
                for (int32_t v1626_i1 = 0; v1626_i1 < 16; ++v1626_i1) {
                  float v1629_data_pre = wrap_glb_m1[allowed_next ? ((v1625_lead + (v1626_i1 * 16))) : (0)];
                  float v1629_data = allowed_next ? (v1629_data_pre) : (0.0f);
                  r0[(v1623_i0 + v1626_i1)] = v1629_data;
                }
              }
              float v1484_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[0] = (v1391_p + (v1482_data * v1484_bc));
              float v1490_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[1] = (v1397_p + (v1482_data * v1490_bc));
              float v1496_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[2] = (v1403_p + (v1482_data * v1496_bc));
              float v1502_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[3] = (v1409_p + (v1482_data * v1502_bc));
              float v1508_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[4] = (v1415_p + (v1482_data * v1508_bc));
              float v1514_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[5] = (v1421_p + (v1482_data * v1514_bc));
              float v1520_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[6] = (v1427_p + (v1482_data * v1520_bc));
              float v1526_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[7] = (v1433_p + (v1482_data * v1526_bc));
              float v1532_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[8] = (v1439_p + (v1482_data * v1532_bc));
              float v1538_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[9] = (v1445_p + (v1482_data * v1538_bc));
              float v1544_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[10] = (v1451_p + (v1482_data * v1544_bc));
              float v1550_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[11] = (v1457_p + (v1482_data * v1550_bc));
              float v1556_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[12] = (v1463_p + (v1482_data * v1556_bc));
              float v1562_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[13] = (v1469_p + (v1482_data * v1562_bc));
              float v1568_bc = sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[14] = (v1475_p + (v1482_data * v1568_bc));
              float v1574_bc = sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[15] = (v1481_p + (v1482_data * v1574_bc));
              // r2 = ir2
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1582_i0 = 0; v1582_i0 < 1; ++v1582_i0) {
                int32_t v1587_lead = v1602_lead + (v1582_i0 * 16);
                #pragma unroll
                for (int32_t v1583_i1 = 0; v1583_i1 < 16; ++v1583_i1) {
                  float v1585_data = ir2[v1583_i1];
                  glb_m0[(v1587_lead + (v1583_i1 * 16))] = v1585_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v1607_i0 = 0; v1607_i0 < 1; ++v1607_i0) {
                int32_t v1609_lead = v1602_lead + (v1607_i0 * 16);
                #pragma unroll
                for (int32_t v1610_i1 = 0; v1610_i1 < 16; ++v1610_i1) {
                  float v1613_data_pre = wrap_glb_m1[allowed_next ? ((v1609_lead + (v1610_i1 * 16))) : (0)];
                  float v1613_data = allowed_next ? (v1613_data_pre) : (0.0f);
                  r0[(v1607_i0 + v1610_i1)] = v1613_data;
                }
              }
            }
            v1595_flagWord = v1596_flagWordNext;
            v1596_flagWordNext = flagWordAhead;
            v1638_elementIndex = wrap_elementIndex;
          }
        }
      });
    }
  });
}


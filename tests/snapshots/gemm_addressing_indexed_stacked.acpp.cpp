// === base name ===
kernel_4d241a88ad45b765

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4d241a88ad45b765 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4d241a88ad45b765(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4d241a88ad45b765(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4d241a88ad45b765(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4d241a88ad45b765(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4d241a88ad45b765(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4d241a88ad45b765(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4d241a88ad45b765(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
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
          size_t v1614_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v1616_flagSecond = (v1614_flagSecondAhead < ((numElements0 + 1) / 2)) ? v1614_flagSecondAhead : v3_batchId1;
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : (flags0[2 * v1616_flagSecond] || (2 * v1616_flagSecond + 1 < numElements0 && flags0[2 * v1616_flagSecond + 1]));
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const bool peel_batchIdStack0_0 = 2 * v3_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 0]));
          const bool peel_batchIdStack0_1 = 2 * v3_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 1]));
          const float *const __restrict__ peel_glb_m2_s0 = &m2[static_cast<size_t>(elementIndices[(peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]) * 3 + 2][0 + m2_extraOffset];
          const float *const __restrict__ peel_glb_m2_s1 = &m2[static_cast<size_t>(elementIndices[(peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]) * 3 + 2][0 + m2_extraOffset];
          int32_t v1627_lead = item.get_local_id(2) % 16;
          if (allowed_peel) {
            // r0 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v1646_i0 = 0; v1646_i0 < 1; ++v1646_i0) {
              int32_t v1648_lead = v1627_lead + (v1646_i0 * 16);
              #pragma unroll
              for (int32_t v1649_i1 = 0; v1649_i1 < 8; ++v1649_i1) {
                int32_t v1651_a = v1648_lead + (v1649_i1 * 16);
                int32_t v1652_a = v1646_i0 + v1649_i1;
                #pragma unroll
                for (int32_t v1653_i2 = 0; v1653_i2 < 2; ++v1653_i2) {
                  const float *const __restrict__ glb_m2_se0 = v1653_i2 == 0 ? peel_glb_m2_s0 : peel_glb_m2_s1;
                  float v1655_data = glb_m2_se0[v1651_a];
                  r0[(v1652_a + (v1653_i2 * 8))] = v1655_data;
                }
              }
            }
          }
          uint32_t v1618_flagWord = flagWordFirst;
          uint32_t v1619_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < ((numElements0 + 1) / 2); v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < ((numElements0 + 1) / 2)) ? v9_ahead1 : v8_batchId0;
            const bool wrap_batchIdStack0_0 = 2 * v11_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v11_batchId1 + 0]));
            const bool wrap_batchIdStack0_1 = 2 * v11_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v11_batchId1 + 1]));
            const float *const __restrict__ wrap_glb_m2_s0 = &m2[static_cast<size_t>(elementIndices[(wrap_batchIdStack0_0 ? 2 * v11_batchId1 + 0 : (wrap_batchIdStack0_0 ? 2 * v11_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v11_batchId1 + 1 : 2 * v11_batchId1))]) * 3 + 2][0 + m2_extraOffset];
            const float *const __restrict__ wrap_glb_m2_s1 = &m2[static_cast<size_t>(elementIndices[(wrap_batchIdStack0_1 ? 2 * v11_batchId1 + 1 : (wrap_batchIdStack0_0 ? 2 * v11_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v11_batchId1 + 1 : 2 * v11_batchId1))]) * 3 + 2][0 + m2_extraOffset];
            size_t v1672_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v1674_flagAhead = (v1672_flagAheadAhead < ((numElements0 + 1) / 2)) ? v1672_flagAheadAhead : v11_batchId1;
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : (flags0[2 * v1674_flagAhead] || (2 * v1674_flagAhead + 1 < numElements0 && flags0[2 * v1674_flagAhead + 1]));
            const bool batchIdStack0_0 = 2 * v8_batchId0 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v8_batchId0 + 0]));
            const bool batchIdStack0_1 = 2 * v8_batchId0 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v8_batchId0 + 1]));
            const bool allowed = static_cast<bool>(v1618_flagWord);
            const bool allowed_next = static_cast<bool>(v1619_flagWordNext);
            if (allowed) {
              float *const glb_m0_s0 = &m0[static_cast<size_t>(elementIndices[(batchIdStack0_0 ? 2 * v8_batchId0 + 0 : (batchIdStack0_0 ? 2 * v8_batchId0 + 0 : batchIdStack0_1 ? 2 * v8_batchId0 + 1 : 2 * v8_batchId0))]) * 136 + 0 + 4 + m0_extraOffset];
              float *const glb_m0_s1 = &m0[static_cast<size_t>(elementIndices[(batchIdStack0_1 ? 2 * v8_batchId0 + 1 : (batchIdStack0_0 ? 2 * v8_batchId0 + 0 : batchIdStack0_1 ? 2 * v8_batchId0 + 1 : 2 * v8_batchId0))]) * 136 + 0 + 4 + m0_extraOffset];
              float r1[16]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 16), (0, 8), (0, 2)] [(0, 16)]
              float ir1[16]{};
              float v43_data = glb_m1[v1627_lead];
              float v44_data = r0[0];
              float v45_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v50_data = r0[8];
              float v51_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v56_data = r0[1];
              float v57_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v62_data = r0[9];
              float v63_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v68_data = r0[2];
              float v69_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v74_data = r0[10];
              float v75_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v80_data = r0[3];
              float v81_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v86_data = r0[11];
              float v87_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v92_data = r0[4];
              float v93_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v98_data = r0[12];
              float v99_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v104_data = r0[5];
              float v105_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v110_data = r0[13];
              float v111_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v116_data = r0[6];
              float v117_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v122_data = r0[14];
              float v123_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v128_data = r0[7];
              float v129_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v134_data = r0[15];
              float v135_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v140_data = glb_m1[(v1627_lead + 16)];
              float v142_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v148_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v154_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v160_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v166_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v172_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v178_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v184_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v190_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v196_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v202_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v208_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v214_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v220_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v226_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v232_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v237_data = glb_m1[(v1627_lead + 32)];
              float v239_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v245_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v251_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v257_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v263_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v269_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v275_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v281_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v287_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v293_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v299_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v305_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v311_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v317_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v323_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v329_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v334_data = glb_m1[(v1627_lead + 48)];
              float v336_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v342_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v348_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v354_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v360_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v366_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v372_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v378_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v384_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v390_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v396_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v402_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v408_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v414_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v420_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v426_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v431_data = glb_m1[(v1627_lead + 64)];
              float v433_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v439_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v445_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v451_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v457_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v463_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v469_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v475_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v481_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v487_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v493_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v499_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v505_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v511_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v517_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v523_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v528_data = glb_m1[(v1627_lead + 80)];
              float v530_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v536_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v542_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v548_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v554_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v560_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v566_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v572_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v578_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v584_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v590_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v596_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v602_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v608_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v614_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v620_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v625_data = glb_m1[(v1627_lead + 96)];
              float v627_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v633_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v639_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v645_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v651_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v657_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v663_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v669_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v675_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v681_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v687_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v693_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v699_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v705_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v711_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v717_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v722_data = glb_m1[(v1627_lead + 112)];
              float v724_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v730_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v736_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v742_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v748_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v754_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v760_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v766_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v772_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v778_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v784_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v790_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v796_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v802_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v808_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v814_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v819_data = glb_m1[(v1627_lead + 128)];
              float v821_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v827_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v833_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v839_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v845_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v851_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v857_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v863_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v869_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v875_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v881_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v887_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v893_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v899_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v905_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v911_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v916_data = glb_m1[(v1627_lead + 144)];
              float v918_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v924_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v930_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v936_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v942_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v948_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v954_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v960_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v966_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v972_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v978_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v984_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v990_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v996_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v1002_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v1008_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v1013_data = glb_m1[(v1627_lead + 160)];
              float v1015_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1021_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1027_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1033_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1039_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1045_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1051_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1057_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1063_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1069_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1075_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1081_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1087_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1093_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1099_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1105_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1110_data = glb_m1[(v1627_lead + 176)];
              float v1112_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1118_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1124_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1130_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1136_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1142_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1148_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1154_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1160_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1166_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1172_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1178_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1184_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1190_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1196_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1202_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1207_data = glb_m1[(v1627_lead + 192)];
              float v1209_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1212_p = ((((((((((((v43_data * v45_bc) + (v140_data * v142_bc)) + (v237_data * v239_bc)) + (v334_data * v336_bc)) + (v431_data * v433_bc)) + (v528_data * v530_bc)) + (v625_data * v627_bc)) + (v722_data * v724_bc)) + (v819_data * v821_bc)) + (v916_data * v918_bc)) + (v1013_data * v1015_bc)) + (v1110_data * v1112_bc)) + (v1207_data * v1209_bc);
              float v1215_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1218_p = ((((((((((((v43_data * v51_bc) + (v140_data * v148_bc)) + (v237_data * v245_bc)) + (v334_data * v342_bc)) + (v431_data * v439_bc)) + (v528_data * v536_bc)) + (v625_data * v633_bc)) + (v722_data * v730_bc)) + (v819_data * v827_bc)) + (v916_data * v924_bc)) + (v1013_data * v1021_bc)) + (v1110_data * v1118_bc)) + (v1207_data * v1215_bc);
              float v1221_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1224_p = ((((((((((((v43_data * v57_bc) + (v140_data * v154_bc)) + (v237_data * v251_bc)) + (v334_data * v348_bc)) + (v431_data * v445_bc)) + (v528_data * v542_bc)) + (v625_data * v639_bc)) + (v722_data * v736_bc)) + (v819_data * v833_bc)) + (v916_data * v930_bc)) + (v1013_data * v1027_bc)) + (v1110_data * v1124_bc)) + (v1207_data * v1221_bc);
              float v1227_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1230_p = ((((((((((((v43_data * v63_bc) + (v140_data * v160_bc)) + (v237_data * v257_bc)) + (v334_data * v354_bc)) + (v431_data * v451_bc)) + (v528_data * v548_bc)) + (v625_data * v645_bc)) + (v722_data * v742_bc)) + (v819_data * v839_bc)) + (v916_data * v936_bc)) + (v1013_data * v1033_bc)) + (v1110_data * v1130_bc)) + (v1207_data * v1227_bc);
              float v1233_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1236_p = ((((((((((((v43_data * v69_bc) + (v140_data * v166_bc)) + (v237_data * v263_bc)) + (v334_data * v360_bc)) + (v431_data * v457_bc)) + (v528_data * v554_bc)) + (v625_data * v651_bc)) + (v722_data * v748_bc)) + (v819_data * v845_bc)) + (v916_data * v942_bc)) + (v1013_data * v1039_bc)) + (v1110_data * v1136_bc)) + (v1207_data * v1233_bc);
              float v1239_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1242_p = ((((((((((((v43_data * v75_bc) + (v140_data * v172_bc)) + (v237_data * v269_bc)) + (v334_data * v366_bc)) + (v431_data * v463_bc)) + (v528_data * v560_bc)) + (v625_data * v657_bc)) + (v722_data * v754_bc)) + (v819_data * v851_bc)) + (v916_data * v948_bc)) + (v1013_data * v1045_bc)) + (v1110_data * v1142_bc)) + (v1207_data * v1239_bc);
              float v1245_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1248_p = ((((((((((((v43_data * v81_bc) + (v140_data * v178_bc)) + (v237_data * v275_bc)) + (v334_data * v372_bc)) + (v431_data * v469_bc)) + (v528_data * v566_bc)) + (v625_data * v663_bc)) + (v722_data * v760_bc)) + (v819_data * v857_bc)) + (v916_data * v954_bc)) + (v1013_data * v1051_bc)) + (v1110_data * v1148_bc)) + (v1207_data * v1245_bc);
              float v1251_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1254_p = ((((((((((((v43_data * v87_bc) + (v140_data * v184_bc)) + (v237_data * v281_bc)) + (v334_data * v378_bc)) + (v431_data * v475_bc)) + (v528_data * v572_bc)) + (v625_data * v669_bc)) + (v722_data * v766_bc)) + (v819_data * v863_bc)) + (v916_data * v960_bc)) + (v1013_data * v1057_bc)) + (v1110_data * v1154_bc)) + (v1207_data * v1251_bc);
              float v1257_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1260_p = ((((((((((((v43_data * v93_bc) + (v140_data * v190_bc)) + (v237_data * v287_bc)) + (v334_data * v384_bc)) + (v431_data * v481_bc)) + (v528_data * v578_bc)) + (v625_data * v675_bc)) + (v722_data * v772_bc)) + (v819_data * v869_bc)) + (v916_data * v966_bc)) + (v1013_data * v1063_bc)) + (v1110_data * v1160_bc)) + (v1207_data * v1257_bc);
              float v1263_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1266_p = ((((((((((((v43_data * v99_bc) + (v140_data * v196_bc)) + (v237_data * v293_bc)) + (v334_data * v390_bc)) + (v431_data * v487_bc)) + (v528_data * v584_bc)) + (v625_data * v681_bc)) + (v722_data * v778_bc)) + (v819_data * v875_bc)) + (v916_data * v972_bc)) + (v1013_data * v1069_bc)) + (v1110_data * v1166_bc)) + (v1207_data * v1263_bc);
              float v1269_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1272_p = ((((((((((((v43_data * v105_bc) + (v140_data * v202_bc)) + (v237_data * v299_bc)) + (v334_data * v396_bc)) + (v431_data * v493_bc)) + (v528_data * v590_bc)) + (v625_data * v687_bc)) + (v722_data * v784_bc)) + (v819_data * v881_bc)) + (v916_data * v978_bc)) + (v1013_data * v1075_bc)) + (v1110_data * v1172_bc)) + (v1207_data * v1269_bc);
              float v1275_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1278_p = ((((((((((((v43_data * v111_bc) + (v140_data * v208_bc)) + (v237_data * v305_bc)) + (v334_data * v402_bc)) + (v431_data * v499_bc)) + (v528_data * v596_bc)) + (v625_data * v693_bc)) + (v722_data * v790_bc)) + (v819_data * v887_bc)) + (v916_data * v984_bc)) + (v1013_data * v1081_bc)) + (v1110_data * v1178_bc)) + (v1207_data * v1275_bc);
              float v1281_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1284_p = ((((((((((((v43_data * v117_bc) + (v140_data * v214_bc)) + (v237_data * v311_bc)) + (v334_data * v408_bc)) + (v431_data * v505_bc)) + (v528_data * v602_bc)) + (v625_data * v699_bc)) + (v722_data * v796_bc)) + (v819_data * v893_bc)) + (v916_data * v990_bc)) + (v1013_data * v1087_bc)) + (v1110_data * v1184_bc)) + (v1207_data * v1281_bc);
              float v1287_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1290_p = ((((((((((((v43_data * v123_bc) + (v140_data * v220_bc)) + (v237_data * v317_bc)) + (v334_data * v414_bc)) + (v431_data * v511_bc)) + (v528_data * v608_bc)) + (v625_data * v705_bc)) + (v722_data * v802_bc)) + (v819_data * v899_bc)) + (v916_data * v996_bc)) + (v1013_data * v1093_bc)) + (v1110_data * v1190_bc)) + (v1207_data * v1287_bc);
              float v1293_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1296_p = ((((((((((((v43_data * v129_bc) + (v140_data * v226_bc)) + (v237_data * v323_bc)) + (v334_data * v420_bc)) + (v431_data * v517_bc)) + (v528_data * v614_bc)) + (v625_data * v711_bc)) + (v722_data * v808_bc)) + (v819_data * v905_bc)) + (v916_data * v1002_bc)) + (v1013_data * v1099_bc)) + (v1110_data * v1196_bc)) + (v1207_data * v1293_bc);
              float v1299_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1302_p = ((((((((((((v43_data * v135_bc) + (v140_data * v232_bc)) + (v237_data * v329_bc)) + (v334_data * v426_bc)) + (v431_data * v523_bc)) + (v528_data * v620_bc)) + (v625_data * v717_bc)) + (v722_data * v814_bc)) + (v819_data * v911_bc)) + (v916_data * v1008_bc)) + (v1013_data * v1105_bc)) + (v1110_data * v1202_bc)) + (v1207_data * v1299_bc);
              float v1304_data = glb_m1[(v1627_lead + 208)];
              float v1306_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1307_p = v1304_data * v1306_bc;
              if (allowed_next) {
                // r0 = load{g>r}(glb_m2);
                #pragma unroll
                for (int32_t v1658_i0 = 0; v1658_i0 < 1; ++v1658_i0) {
                  int32_t v1660_lead = v1627_lead + (v1658_i0 * 16);
                  #pragma unroll
                  for (int32_t v1661_i1 = 0; v1661_i1 < 8; ++v1661_i1) {
                    int32_t v1663_a = v1660_lead + (v1661_i1 * 16);
                    int32_t v1664_a = v1658_i0 + v1661_i1;
                    #pragma unroll
                    for (int32_t v1665_i2 = 0; v1665_i2 < 2; ++v1665_i2) {
                      const float *const __restrict__ glb_m2_se0 = v1665_i2 == 0 ? wrap_glb_m2_s0 : wrap_glb_m2_s1;
                      float v1667_data = glb_m2_se0[v1663_a];
                      r0[(v1664_a + (v1665_i2 * 8))] = v1667_data;
                    }
                  }
                }
              }
              float v1312_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1318_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1324_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1330_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1336_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1342_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1348_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1354_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1360_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1366_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1372_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1378_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1384_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1390_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1396_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1401_data = glb_m1[(v1627_lead + 224)];
              float v1403_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1409_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1412_p = (v1218_p + (v1304_data * v1312_bc)) + (v1401_data * v1409_bc);
              float v1415_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1418_p = (v1224_p + (v1304_data * v1318_bc)) + (v1401_data * v1415_bc);
              float v1421_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1424_p = (v1230_p + (v1304_data * v1324_bc)) + (v1401_data * v1421_bc);
              float v1427_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1430_p = (v1236_p + (v1304_data * v1330_bc)) + (v1401_data * v1427_bc);
              float v1433_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1436_p = (v1242_p + (v1304_data * v1336_bc)) + (v1401_data * v1433_bc);
              float v1439_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1442_p = (v1248_p + (v1304_data * v1342_bc)) + (v1401_data * v1439_bc);
              float v1445_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1448_p = (v1254_p + (v1304_data * v1348_bc)) + (v1401_data * v1445_bc);
              float v1451_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1454_p = (v1260_p + (v1304_data * v1354_bc)) + (v1401_data * v1451_bc);
              float v1457_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1460_p = (v1266_p + (v1304_data * v1360_bc)) + (v1401_data * v1457_bc);
              float v1463_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1466_p = (v1272_p + (v1304_data * v1366_bc)) + (v1401_data * v1463_bc);
              float v1469_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1472_p = (v1278_p + (v1304_data * v1372_bc)) + (v1401_data * v1469_bc);
              float v1475_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1478_p = (v1284_p + (v1304_data * v1378_bc)) + (v1401_data * v1475_bc);
              float v1481_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1484_p = (v1290_p + (v1304_data * v1384_bc)) + (v1401_data * v1481_bc);
              float v1487_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1490_p = (v1296_p + (v1304_data * v1390_bc)) + (v1401_data * v1487_bc);
              float v1493_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1496_p = (v1302_p + (v1304_data * v1396_bc)) + (v1401_data * v1493_bc);
              float v1498_data = glb_m1[(v1627_lead + 240)];
              float v1500_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[0] = (((v1212_p + v1307_p) + (v1401_data * v1403_bc)) + (v1498_data * v1500_bc));
              float v1506_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[8] = (v1412_p + (v1498_data * v1506_bc));
              float v1512_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[1] = (v1418_p + (v1498_data * v1512_bc));
              float v1518_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[9] = (v1424_p + (v1498_data * v1518_bc));
              float v1524_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[2] = (v1430_p + (v1498_data * v1524_bc));
              float v1530_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[10] = (v1436_p + (v1498_data * v1530_bc));
              float v1536_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[3] = (v1442_p + (v1498_data * v1536_bc));
              float v1542_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[11] = (v1448_p + (v1498_data * v1542_bc));
              float v1548_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[4] = (v1454_p + (v1498_data * v1548_bc));
              float v1554_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[12] = (v1460_p + (v1498_data * v1554_bc));
              float v1560_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[5] = (v1466_p + (v1498_data * v1560_bc));
              float v1566_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[13] = (v1472_p + (v1498_data * v1566_bc));
              float v1572_bc = sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[6] = (v1478_p + (v1498_data * v1572_bc));
              float v1578_bc = sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[14] = (v1484_p + (v1498_data * v1578_bc));
              float v1584_bc = sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[7] = (v1490_p + (v1498_data * v1584_bc));
              float v1590_bc = sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir1[15] = (v1496_p + (v1498_data * v1590_bc));
              // r1 = ir1
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v1601_i0 = 0; v1601_i0 < 1; ++v1601_i0) {
                #pragma unroll
                for (int32_t v1602_i1 = 0; v1602_i1 < 8; ++v1602_i1) {
                  #pragma unroll
                  for (int32_t v1603_i2 = 0; v1603_i2 < 2; ++v1603_i2) {
                    float v1607_data = ir1[(v1602_i1 + (v1603_i2 * 8))];
                    if ((v1603_i2 == 0 ? batchIdStack0_0 : batchIdStack0_1)) {
                      float *const glb_m0_se0 = v1603_i2 == 0 ? glb_m0_s0 : glb_m0_s1;
                      glb_m0_se0[((v1627_lead + (v1601_i0 * 16)) + (v1602_i1 * 16))] = v1607_data;
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
                for (int32_t v1634_i0 = 0; v1634_i0 < 1; ++v1634_i0) {
                  int32_t v1636_lead = v1627_lead + (v1634_i0 * 16);
                  #pragma unroll
                  for (int32_t v1637_i1 = 0; v1637_i1 < 8; ++v1637_i1) {
                    int32_t v1639_a = v1636_lead + (v1637_i1 * 16);
                    int32_t v1640_a = v1634_i0 + v1637_i1;
                    #pragma unroll
                    for (int32_t v1641_i2 = 0; v1641_i2 < 2; ++v1641_i2) {
                      const float *const __restrict__ glb_m2_se0 = v1641_i2 == 0 ? wrap_glb_m2_s0 : wrap_glb_m2_s1;
                      float v1643_data = glb_m2_se0[v1639_a];
                      r0[(v1640_a + (v1641_i2 * 8))] = v1643_data;
                    }
                  }
                }
              }
            }
            v1618_flagWord = v1619_flagWordNext;
            v1619_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


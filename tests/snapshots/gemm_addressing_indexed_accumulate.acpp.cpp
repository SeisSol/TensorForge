// === base name ===
kernel_7a28c9834a0e8fb2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7a28c9834a0e8fb2 = {{32, 1, 1}, 32, 20, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7a28c9834a0e8fb2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7a28c9834a0e8fb2(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7a28c9834a0e8fb2(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_7a28c9834a0e8fb2(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7a28c9834a0e8fb2(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7a28c9834a0e8fb2(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7a28c9834a0e8fb2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
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
          size_t v554_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v554_flagSecondAhead < numElements0) ? v554_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const float *const __restrict__ peel_glb_m2 = &m2[static_cast<size_t>(elementIndices[v3_batchId1]) * 81 + 0 + 12 + m2_extraOffset];
          int32_t v564_lead = item.get_local_id(2) % 32;
          bool v565_g = v564_lead < 9;
          bool v582_g = allowed_peel && v565_g;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v574_i1 = 0; v574_i1 < 9; ++v574_i1) {
            float v577_data_pre = peel_glb_m2[v582_g ? ((v564_lead + (v574_i1 * 9))) : (0)];
            float v577_data = v582_g ? (v577_data_pre) : (0.0f);
            r0[v574_i1] = v577_data;
          }
          uint32_t v558_flagWord = flagWordFirst;
          uint32_t v559_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[static_cast<size_t>(elementIndices[v11_batchId1]) * 81 + 0 + 12 + m2_extraOffset];
            size_t v587_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v587_flagAheadAhead < numElements0) ? v587_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v558_flagWord);
            const bool allowed_next = static_cast<bool>(v559_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[static_cast<size_t>(elementIndices[v8_batchId0]) * 2 + 1][0 + m0_extraOffset];
              float r1[9]{};
              // r1 = load{g>r}(glb_m0);
              bool v536_g = v564_lead < 20;
              #pragma unroll
              for (int32_t v531_i1 = 0; v531_i1 < 9; ++v531_i1) {
                float v537_data_pre = glb_m0[v536_g ? ((v564_lead + (v531_i1 * 20))) : (0)];
                float v537_data = v536_g ? (v537_data_pre) : (0.0f);
                r1[v531_i1] = v537_data;
              }
              float r2[9]{};
              // ir2 = +(glb_m1 * r0)
              // [(0, 20), (0, 9)] [(0, 9)]
              float ir2[9]{};
              float v36_data_pre = glb_m1[v536_g ? (v564_lead) : (0)];
              float v36_data = v536_g ? (v36_data_pre) : (0.0f);
              float v37_data = r0[0];
              float v38_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v39_p = v36_data * v38_bc;
              float v43_data = r0[1];
              float v44_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v45_p = v36_data * v44_bc;
              float v49_data = r0[2];
              float v50_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v51_p = v36_data * v50_bc;
              float v55_data = r0[3];
              float v56_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v57_p = v36_data * v56_bc;
              float v61_data = r0[4];
              float v62_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v63_p = v36_data * v62_bc;
              float v67_data = r0[5];
              float v68_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v69_p = v36_data * v68_bc;
              float v73_data = r0[6];
              float v74_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v75_p = v36_data * v74_bc;
              float v79_data = r0[7];
              float v80_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v81_p = v36_data * v80_bc;
              float v85_data = r0[8];
              float v86_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v87_p = v36_data * v86_bc;
              bool v583_g = allowed_next && v565_g;
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v578_i1 = 0; v578_i1 < 9; ++v578_i1) {
                float v581_data_pre = wrap_glb_m2[v583_g ? ((v564_lead + (v578_i1 * 9))) : (0)];
                float v581_data = v583_g ? (v581_data_pre) : (0.0f);
                r0[v578_i1] = v581_data;
              }
              float v91_data_pre = glb_m1[v536_g ? ((v564_lead + 20)) : (0)];
              float v91_data = v536_g ? (v91_data_pre) : (0.0f);
              float v93_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v99_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v105_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v111_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v117_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v123_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v129_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v135_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v141_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v146_data_pre = glb_m1[v536_g ? ((v564_lead + 40)) : (0)];
              float v146_data = v536_g ? (v146_data_pre) : (0.0f);
              float v148_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v154_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v160_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v166_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v172_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v178_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v184_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v190_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v196_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v201_data_pre = glb_m1[v536_g ? ((v564_lead + 60)) : (0)];
              float v201_data = v536_g ? (v201_data_pre) : (0.0f);
              float v203_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v209_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v215_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v221_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v227_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v233_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v239_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v245_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v251_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v256_data_pre = glb_m1[v536_g ? ((v564_lead + 80)) : (0)];
              float v256_data = v536_g ? (v256_data_pre) : (0.0f);
              float v258_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v264_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v270_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v276_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v282_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v288_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v294_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v300_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v306_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v311_data_pre = glb_m1[v536_g ? ((v564_lead + 100)) : (0)];
              float v311_data = v536_g ? (v311_data_pre) : (0.0f);
              float v313_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v319_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v325_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v331_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v337_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v343_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v349_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v355_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v361_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v366_data_pre = glb_m1[v536_g ? ((v564_lead + 120)) : (0)];
              float v366_data = v536_g ? (v366_data_pre) : (0.0f);
              float v368_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v374_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v380_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v386_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v392_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v398_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v404_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v410_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v416_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v421_data_pre = glb_m1[v536_g ? ((v564_lead + 140)) : (0)];
              float v421_data = v536_g ? (v421_data_pre) : (0.0f);
              float v423_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v429_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v432_p = ((((((v45_p + (v91_data * v99_bc)) + (v146_data * v154_bc)) + (v201_data * v209_bc)) + (v256_data * v264_bc)) + (v311_data * v319_bc)) + (v366_data * v374_bc)) + (v421_data * v429_bc);
              float v435_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v438_p = ((((((v51_p + (v91_data * v105_bc)) + (v146_data * v160_bc)) + (v201_data * v215_bc)) + (v256_data * v270_bc)) + (v311_data * v325_bc)) + (v366_data * v380_bc)) + (v421_data * v435_bc);
              float v441_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v444_p = ((((((v57_p + (v91_data * v111_bc)) + (v146_data * v166_bc)) + (v201_data * v221_bc)) + (v256_data * v276_bc)) + (v311_data * v331_bc)) + (v366_data * v386_bc)) + (v421_data * v441_bc);
              float v447_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v450_p = ((((((v63_p + (v91_data * v117_bc)) + (v146_data * v172_bc)) + (v201_data * v227_bc)) + (v256_data * v282_bc)) + (v311_data * v337_bc)) + (v366_data * v392_bc)) + (v421_data * v447_bc);
              float v453_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v456_p = ((((((v69_p + (v91_data * v123_bc)) + (v146_data * v178_bc)) + (v201_data * v233_bc)) + (v256_data * v288_bc)) + (v311_data * v343_bc)) + (v366_data * v398_bc)) + (v421_data * v453_bc);
              float v459_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v462_p = ((((((v75_p + (v91_data * v129_bc)) + (v146_data * v184_bc)) + (v201_data * v239_bc)) + (v256_data * v294_bc)) + (v311_data * v349_bc)) + (v366_data * v404_bc)) + (v421_data * v459_bc);
              float v465_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v468_p = ((((((v81_p + (v91_data * v135_bc)) + (v146_data * v190_bc)) + (v201_data * v245_bc)) + (v256_data * v300_bc)) + (v311_data * v355_bc)) + (v366_data * v410_bc)) + (v421_data * v465_bc);
              float v471_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v474_p = ((((((v87_p + (v91_data * v141_bc)) + (v146_data * v196_bc)) + (v201_data * v251_bc)) + (v256_data * v306_bc)) + (v311_data * v361_bc)) + (v366_data * v416_bc)) + (v421_data * v471_bc);
              float v476_data_pre = glb_m1[v536_g ? ((v564_lead + 160)) : (0)];
              float v476_data = v536_g ? (v476_data_pre) : (0.0f);
              float v478_bc = sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[0] = ((((((((v39_p + (v91_data * v93_bc)) + (v146_data * v148_bc)) + (v201_data * v203_bc)) + (v256_data * v258_bc)) + (v311_data * v313_bc)) + (v366_data * v368_bc)) + (v421_data * v423_bc)) + (v476_data * v478_bc));
              float v484_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[1] = (v432_p + (v476_data * v484_bc));
              float v490_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[2] = (v438_p + (v476_data * v490_bc));
              float v496_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[3] = (v444_p + (v476_data * v496_bc));
              float v502_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[4] = (v450_p + (v476_data * v502_bc));
              float v508_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[5] = (v456_p + (v476_data * v508_bc));
              float v514_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[6] = (v462_p + (v476_data * v514_bc));
              float v520_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[7] = (v468_p + (v476_data * v520_bc));
              float v526_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              ir2[8] = (v474_p + (v476_data * v526_bc));
              // r2 = ir2 + r1
              #pragma unroll
              for (int32_t v540_n1 = 0; v540_n1 < 9; ++v540_n1) {
                float v542_data_pre = ir2[v540_n1];
                float v542_data = v536_g ? (v542_data_pre) : (0.0f);
                float v543_data_pre = r1[v540_n1];
                float v543_data = v536_g ? (v543_data_pre) : (0.0f);
                if (v536_g) {
                  r2[v540_n1] = (v543_data + v542_data);
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v546_i1 = 0; v546_i1 < 9; ++v546_i1) {
                float v548_data_pre = r2[v546_i1];
                float v548_data = v536_g ? (v548_data_pre) : (0.0f);
                if (v536_g) {
                  glb_m0[(v564_lead + (v546_i1 * 20))] = v548_data;
                }
              }
              item.barrier();
            }
            else {
              bool v584_g = allowed_next && v565_g;
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v570_i1 = 0; v570_i1 < 9; ++v570_i1) {
                float v573_data_pre = wrap_glb_m2[v584_g ? ((v564_lead + (v570_i1 * 9))) : (0)];
                float v573_data = v584_g ? (v573_data_pre) : (0.0f);
                r0[v570_i1] = v573_data;
              }
            }
            v558_flagWord = v559_flagWordNext;
            v559_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


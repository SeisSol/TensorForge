// === base name ===
kernel_75923508de941612

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_75923508de941612 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_75923508de941612(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_75923508de941612(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_75923508de941612(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_75923508de941612(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_75923508de941612(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_75923508de941612(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_75923508de941612(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: k_roll=4,pack_operands=bits
        // launch: 32 lanes x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 32×6(32×6) {0..32}×{0..6} strided
        //   m1 32×48(32×48) {0..32}×{0..48} none ordered bits-packed
        //   m2 48×6(48×6) {0..48}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[32,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[32,48]],"name":"m1","ordered":true,"parts":1,"shape":[32,48],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[48,6]],"name":"m2","ordered":false,"parts":1,"shape":[48,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[32,48]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,48]},{"addressing":"strided","bbox":[[0,0],[48,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[48,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          float r0[12]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v2070_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v2070_flagSecondAhead < numElements0) ? v2070_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 288 + 0 + m2_extraOffset];
          int32_t v2078_lead = item.get_local_id(2) % 32;
          int32_t v2080_lead = v2078_lead + 32_i32;
          bool v2081_g = v2078_lead < 16;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v2103_i0 = 0; v2103_i0 < 1; ++v2103_i0) {
            int32_t v2105_lead = v2078_lead + (v2103_i0 * 32);
            #pragma unroll
            for (int32_t v2106_i1 = 0; v2106_i1 < 6; ++v2106_i1) {
              float v2109_data = peel_glb_m2[(v2105_lead + (v2106_i1 * 48))];
              r0[(v2103_i0 + (v2106_i1 * 2))] = v2109_data;
            }
          }
          #pragma unroll
          for (int32_t v2112_i1 = 0; v2112_i1 < 6; ++v2112_i1) {
            float v2115_data_pre = peel_glb_m2[v2081_g ? ((v2080_lead + (v2112_i1 * 48))) : (0)];
            float v2115_data = v2081_g ? (v2115_data_pre) : (0.0f);
            r0[(1 + (v2112_i1 * 2))] = v2115_data;
          }
          uint32_t v2074_flagWord = flagWordFirst;
          uint32_t v2075_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 288 + 0 + m2_extraOffset];
            size_t v2135_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v2135_flagAheadAhead < numElements0) ? v2135_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v2074_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 192 + 0 + m0_extraOffset];
              float r1[6]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 32), (0, 6)] [(0, 48)]
              float ir1[6]{};
              int32_t v43_bit = (1 << v2078_lead) - 1;
              float v50_data_pre = glb_m1[(((3067833782 >> v2078_lead) & 1) != 0) ? ((sycl::popcount((3067833782 & v43_bit)))) : (0)];
              float v50_data = (((3067833782 >> v2078_lead) & 1) != 0) ? (v50_data_pre) : (0.0f);
              float v51_data = r0[0];
              ir1[0] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v57_data = r0[2];
              ir1[1] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v63_data = r0[4];
              ir1[2] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v69_data = r0[6];
              ir1[3] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v75_data = r0[8];
              ir1[4] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v81_data = r0[10];
              ir1[5] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              int32_t v87_rank = sycl::popcount((460175040 & v43_bit));
              bool v90_g = ((460175040 >> v2078_lead) & 1) != 0;
              float v93_data_pre = glb_m1[v90_g ? ((v87_rank + 21)) : (0)];
              float v93_data = v90_g ? (v93_data_pre) : (0.0f);
              float v97_data = ir1[0];
              ir1[0] = (v97_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v103_data = ir1[1];
              ir1[1] = (v103_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v109_data = ir1[2];
              ir1[2] = (v109_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v115_data = ir1[3];
              ir1[3] = (v115_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v121_data = ir1[4];
              ir1[4] = (v121_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v127_data = ir1[5];
              ir1[5] = (v127_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v136_data_pre = glb_m1[(((28760940 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((28760940 & v43_bit))) + 37)) : (0)];
              float v136_data = (((28760940 >> v2078_lead) & 1) != 0) ? (v136_data_pre) : (0.0f);
              float v140_data = ir1[0];
              ir1[0] = (v140_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v146_data = ir1[1];
              ir1[1] = (v146_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v152_data = ir1[2];
              ir1[2] = (v152_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v158_data = ir1[3];
              ir1[3] = (v158_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v164_data = ir1[4];
              ir1[4] = (v164_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v170_data = ir1[5];
              ir1[5] = (v170_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              int32_t v173_rank = sycl::popcount((920350080 & v43_bit));
              bool v176_g = ((920350080 >> v2078_lead) & 1) != 0;
              float v179_data_pre = glb_m1[v176_g ? ((v173_rank + 53)) : (0)];
              float v179_data = v176_g ? (v179_data_pre) : (0.0f);
              float v183_data = ir1[0];
              ir1[0] = (v183_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v189_data = ir1[1];
              ir1[1] = (v189_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v195_data = ir1[2];
              ir1[2] = (v195_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v201_data = ir1[3];
              ir1[3] = (v201_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v207_data = ir1[4];
              ir1[4] = (v207_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v213_data = ir1[5];
              ir1[5] = (v213_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v222_data_pre = glb_m1[(((57521880 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((57521880 & v43_bit))) + 69)) : (0)];
              float v222_data = (((57521880 >> v2078_lead) & 1) != 0) ? (v222_data_pre) : (0.0f);
              float v226_data = ir1[0];
              ir1[0] = (v226_data + (v222_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v232_data = ir1[1];
              ir1[1] = (v232_data + (v222_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v238_data = ir1[2];
              ir1[2] = (v238_data + (v222_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v244_data = ir1[3];
              ir1[3] = (v244_data + (v222_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v250_data = ir1[4];
              ir1[4] = (v250_data + (v222_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v256_data = ir1[5];
              ir1[5] = (v256_data + (v222_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              int32_t v259_rank = sycl::popcount((1840700160 & v43_bit));
              bool v262_g = ((1840700160 >> v2078_lead) & 1) != 0;
              float v265_data_pre = glb_m1[v262_g ? ((v259_rank + 85)) : (0)];
              float v265_data = v262_g ? (v265_data_pre) : (0.0f);
              float v269_data = ir1[0];
              ir1[0] = (v269_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v275_data = ir1[1];
              ir1[1] = (v275_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v281_data = ir1[2];
              ir1[2] = (v281_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v287_data = ir1[3];
              ir1[3] = (v287_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v293_data = ir1[4];
              ir1[4] = (v293_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v299_data = ir1[5];
              ir1[5] = (v299_data + (v265_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v308_data_pre = glb_m1[(((115043760 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((115043760 & v43_bit))) + 101)) : (0)];
              float v308_data = (((115043760 >> v2078_lead) & 1) != 0) ? (v308_data_pre) : (0.0f);
              float v312_data = ir1[0];
              ir1[0] = (v312_data + (v308_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v318_data = ir1[1];
              ir1[1] = (v318_data + (v308_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v324_data = ir1[2];
              ir1[2] = (v324_data + (v308_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v330_data = ir1[3];
              ir1[3] = (v330_data + (v308_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v336_data = ir1[4];
              ir1[4] = (v336_data + (v308_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v342_data = ir1[5];
              ir1[5] = (v342_data + (v308_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v351_data_pre = glb_m1[(((3681400320 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((3681400320 & v43_bit))) + 117)) : (0)];
              float v351_data = (((3681400320 >> v2078_lead) & 1) != 0) ? (v351_data_pre) : (0.0f);
              float v355_data = ir1[0];
              ir1[0] = (v355_data + (v351_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v361_data = ir1[1];
              ir1[1] = (v361_data + (v351_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v367_data = ir1[2];
              ir1[2] = (v367_data + (v351_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v373_data = ir1[3];
              ir1[3] = (v373_data + (v351_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v379_data = ir1[4];
              ir1[4] = (v379_data + (v351_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v385_data = ir1[5];
              ir1[5] = (v385_data + (v351_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v394_data_pre = glb_m1[(((230087520 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((230087520 & v43_bit))) + 133)) : (0)];
              float v394_data = (((230087520 >> v2078_lead) & 1) != 0) ? (v394_data_pre) : (0.0f);
              float v398_data = ir1[0];
              ir1[0] = (v398_data + (v394_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v404_data = ir1[1];
              ir1[1] = (v404_data + (v394_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v410_data = ir1[2];
              ir1[2] = (v410_data + (v394_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v416_data = ir1[3];
              ir1[3] = (v416_data + (v394_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v422_data = ir1[4];
              ir1[4] = (v422_data + (v394_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v428_data = ir1[5];
              ir1[5] = (v428_data + (v394_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v437_data_pre = glb_m1[(((47934902 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((47934902 & v43_bit))) + 149)) : (0)];
              float v437_data = (((47934902 >> v2078_lead) & 1) != 0) ? (v437_data_pre) : (0.0f);
              float v441_data = ir1[0];
              ir1[0] = (v441_data + (v437_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v447_data = ir1[1];
              ir1[1] = (v447_data + (v437_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v453_data = ir1[2];
              ir1[2] = (v453_data + (v437_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v459_data = ir1[3];
              ir1[3] = (v459_data + (v437_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v465_data = ir1[4];
              ir1[4] = (v465_data + (v437_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v471_data = ir1[5];
              ir1[5] = (v471_data + (v437_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v475_data_pre = glb_m1[v90_g ? ((v87_rank + 166)) : (0)];
              float v475_data = v90_g ? (v475_data_pre) : (0.0f);
              float v479_data = ir1[0];
              ir1[0] = (v479_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v485_data = ir1[1];
              ir1[1] = (v485_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v491_data = ir1[2];
              ir1[2] = (v491_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v497_data = ir1[3];
              ir1[3] = (v497_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v503_data = ir1[4];
              ir1[4] = (v503_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v509_data = ir1[5];
              ir1[5] = (v509_data + (v475_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v518_data_pre = glb_m1[(((95869804 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((95869804 & v43_bit))) + 182)) : (0)];
              float v518_data = (((95869804 >> v2078_lead) & 1) != 0) ? (v518_data_pre) : (0.0f);
              float v522_data = ir1[0];
              ir1[0] = (v522_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v528_data = ir1[1];
              ir1[1] = (v528_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v534_data = ir1[2];
              ir1[2] = (v534_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v540_data = ir1[3];
              ir1[3] = (v540_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v546_data = ir1[4];
              ir1[4] = (v546_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v552_data = ir1[5];
              ir1[5] = (v552_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v556_data_pre = glb_m1[v176_g ? ((v173_rank + 199)) : (0)];
              float v556_data = v176_g ? (v556_data_pre) : (0.0f);
              float v560_data = ir1[0];
              ir1[0] = (v560_data + (v556_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v566_data = ir1[1];
              ir1[1] = (v566_data + (v556_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v572_data = ir1[2];
              ir1[2] = (v572_data + (v556_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v578_data = ir1[3];
              ir1[3] = (v578_data + (v556_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v584_data = ir1[4];
              ir1[4] = (v584_data + (v556_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v590_data = ir1[5];
              ir1[5] = (v590_data + (v556_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v599_data_pre = glb_m1[(((191739608 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((191739608 & v43_bit))) + 215)) : (0)];
              float v599_data = (((191739608 >> v2078_lead) & 1) != 0) ? (v599_data_pre) : (0.0f);
              float v603_data = ir1[0];
              ir1[0] = (v603_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v609_data = ir1[1];
              ir1[1] = (v609_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v615_data = ir1[2];
              ir1[2] = (v615_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v621_data = ir1[3];
              ir1[3] = (v621_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v627_data = ir1[4];
              ir1[4] = (v627_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v633_data = ir1[5];
              ir1[5] = (v633_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v637_data_pre = glb_m1[v262_g ? ((v259_rank + 232)) : (0)];
              float v637_data = v262_g ? (v637_data_pre) : (0.0f);
              float v641_data = ir1[0];
              ir1[0] = (v641_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v647_data = ir1[1];
              ir1[1] = (v647_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v653_data = ir1[2];
              ir1[2] = (v653_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v659_data = ir1[3];
              ir1[3] = (v659_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v665_data = ir1[4];
              ir1[4] = (v665_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v671_data = ir1[5];
              ir1[5] = (v671_data + (v637_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v680_data_pre = glb_m1[(((383479216 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((383479216 & v43_bit))) + 248)) : (0)];
              float v680_data = (((383479216 >> v2078_lead) & 1) != 0) ? (v680_data_pre) : (0.0f);
              float v684_data = ir1[0];
              ir1[0] = (v684_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v690_data = ir1[1];
              ir1[1] = (v690_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v696_data = ir1[2];
              ir1[2] = (v696_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v702_data = ir1[3];
              ir1[3] = (v702_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v708_data = ir1[4];
              ir1[4] = (v708_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v714_data = ir1[5];
              ir1[5] = (v714_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              int32_t v717_rank = sycl::popcount((57521664 & v43_bit));
              bool v720_g = ((57521664 >> v2078_lead) & 1) != 0;
              float v723_data_pre = glb_m1[v720_g ? ((v717_rank + 265)) : (0)];
              float v723_data = v720_g ? (v723_data_pre) : (0.0f);
              float v727_data = ir1[0];
              ir1[0] = (v727_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v733_data = ir1[1];
              ir1[1] = (v733_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v739_data = ir1[2];
              ir1[2] = (v739_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v745_data = ir1[3];
              ir1[3] = (v745_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v751_data = ir1[4];
              ir1[4] = (v751_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v757_data = ir1[5];
              ir1[5] = (v757_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v766_data_pre = glb_m1[(((766958432 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((766958432 & v43_bit))) + 277)) : (0)];
              float v766_data = (((766958432 >> v2078_lead) & 1) != 0) ? (v766_data_pre) : (0.0f);
              float v770_data = ir1[0];
              ir1[0] = (v770_data + (v766_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v776_data = ir1[1];
              ir1[1] = (v776_data + (v766_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v782_data = ir1[2];
              ir1[2] = (v782_data + (v766_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v788_data = ir1[3];
              ir1[3] = (v788_data + (v766_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v794_data = ir1[4];
              ir1[4] = (v794_data + (v766_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v800_data = ir1[5];
              ir1[5] = (v800_data + (v766_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              int32_t v803_rank = sycl::popcount((115043766 & v43_bit));
              bool v806_g = ((115043766 >> v2078_lead) & 1) != 0;
              float v809_data_pre = glb_m1[v806_g ? ((v803_rank + 294)) : (0)];
              float v809_data = v806_g ? (v809_data_pre) : (0.0f);
              float v813_data = ir1[0];
              ir1[0] = (v813_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v819_data = ir1[1];
              ir1[1] = (v819_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v825_data = ir1[2];
              ir1[2] = (v825_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v831_data = ir1[3];
              ir1[3] = (v831_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v837_data = ir1[4];
              ir1[4] = (v837_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v843_data = ir1[5];
              ir1[5] = (v843_data + (v809_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v852_data_pre = glb_m1[(((1533916864 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((1533916864 & v43_bit))) + 312)) : (0)];
              float v852_data = (((1533916864 >> v2078_lead) & 1) != 0) ? (v852_data_pre) : (0.0f);
              float v856_data = ir1[0];
              ir1[0] = (v856_data + (v852_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v862_data = ir1[1];
              ir1[1] = (v862_data + (v852_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v868_data = ir1[2];
              ir1[2] = (v868_data + (v852_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v874_data = ir1[3];
              ir1[3] = (v874_data + (v852_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v880_data = ir1[4];
              ir1[4] = (v880_data + (v852_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v886_data = ir1[5];
              ir1[5] = (v886_data + (v852_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              int32_t v889_rank = sycl::popcount((230087532 & v43_bit));
              bool v892_g = ((230087532 >> v2078_lead) & 1) != 0;
              float v895_data_pre = glb_m1[v892_g ? ((v889_rank + 329)) : (0)];
              float v895_data = v892_g ? (v895_data_pre) : (0.0f);
              float v899_data = ir1[0];
              ir1[0] = (v899_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v905_data = ir1[1];
              ir1[1] = (v905_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v911_data = ir1[2];
              ir1[2] = (v911_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v917_data = ir1[3];
              ir1[3] = (v917_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v923_data = ir1[4];
              ir1[4] = (v923_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v929_data = ir1[5];
              ir1[5] = (v929_data + (v895_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v938_data_pre = glb_m1[(((3067833728 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((3067833728 & v43_bit))) + 347)) : (0)];
              float v938_data = (((3067833728 >> v2078_lead) & 1) != 0) ? (v938_data_pre) : (0.0f);
              float v942_data = ir1[0];
              ir1[0] = (v942_data + (v938_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v948_data = ir1[1];
              ir1[1] = (v948_data + (v938_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v954_data = ir1[2];
              ir1[2] = (v954_data + (v938_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v960_data = ir1[3];
              ir1[3] = (v960_data + (v938_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v966_data = ir1[4];
              ir1[4] = (v966_data + (v938_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v972_data = ir1[5];
              ir1[5] = (v972_data + (v938_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              int32_t v975_rank = sycl::popcount((460175064 & v43_bit));
              bool v978_g = ((460175064 >> v2078_lead) & 1) != 0;
              float v981_data_pre = glb_m1[v978_g ? ((v975_rank + 364)) : (0)];
              float v981_data = v978_g ? (v981_data_pre) : (0.0f);
              float v985_data = ir1[0];
              ir1[0] = (v985_data + (v981_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v991_data = ir1[1];
              ir1[1] = (v991_data + (v981_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v997_data = ir1[2];
              ir1[2] = (v997_data + (v981_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1003_data = ir1[3];
              ir1[3] = (v1003_data + (v981_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1009_data = ir1[4];
              ir1[4] = (v1009_data + (v981_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1015_data = ir1[5];
              ir1[5] = (v1015_data + (v981_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1024_data_pre = glb_m1[(((28760832 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((28760832 & v43_bit))) + 382)) : (0)];
              float v1024_data = (((28760832 >> v2078_lead) & 1) != 0) ? (v1024_data_pre) : (0.0f);
              float v1028_data = ir1[0];
              ir1[0] = (v1028_data + (v1024_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1034_data = ir1[1];
              ir1[1] = (v1034_data + (v1024_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1040_data = ir1[2];
              ir1[2] = (v1040_data + (v1024_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1046_data = ir1[3];
              ir1[3] = (v1046_data + (v1024_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1052_data = ir1[4];
              ir1[4] = (v1052_data + (v1024_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1058_data = ir1[5];
              ir1[5] = (v1058_data + (v1024_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              int32_t v1061_rank = sycl::popcount((920350128 & v43_bit));
              bool v1064_g = ((920350128 >> v2078_lead) & 1) != 0;
              float v1067_data_pre = glb_m1[v1064_g ? ((v1061_rank + 394)) : (0)];
              float v1067_data = v1064_g ? (v1067_data_pre) : (0.0f);
              float v1071_data = ir1[0];
              ir1[0] = (v1071_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (24)))));
              float v1077_data = ir1[1];
              ir1[1] = (v1077_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (24)))));
              float v1083_data = ir1[2];
              ir1[2] = (v1083_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (24)))));
              float v1089_data = ir1[3];
              ir1[3] = (v1089_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (24)))));
              float v1095_data = ir1[4];
              ir1[4] = (v1095_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (24)))));
              float v1101_data = ir1[5];
              ir1[5] = (v1101_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (24)))));
              float v1105_data_pre = glb_m1[v720_g ? ((v717_rank + 412)) : (0)];
              float v1105_data = v720_g ? (v1105_data_pre) : (0.0f);
              float v1109_data = ir1[0];
              ir1[0] = (v1109_data + (v1105_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (25)))));
              float v1115_data = ir1[1];
              ir1[1] = (v1115_data + (v1105_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (25)))));
              float v1121_data = ir1[2];
              ir1[2] = (v1121_data + (v1105_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (25)))));
              float v1127_data = ir1[3];
              ir1[3] = (v1127_data + (v1105_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (25)))));
              float v1133_data = ir1[4];
              ir1[4] = (v1133_data + (v1105_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (25)))));
              float v1139_data = ir1[5];
              ir1[5] = (v1139_data + (v1105_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (25)))));
              int32_t v1142_rank = sycl::popcount((1840700256 & v43_bit));
              bool v1145_g = ((1840700256 >> v2078_lead) & 1) != 0;
              float v1148_data_pre = glb_m1[v1145_g ? ((v1142_rank + 424)) : (0)];
              float v1148_data = v1145_g ? (v1148_data_pre) : (0.0f);
              float v1152_data = ir1[0];
              ir1[0] = (v1152_data + (v1148_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (26)))));
              float v1158_data = ir1[1];
              ir1[1] = (v1158_data + (v1148_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (26)))));
              float v1164_data = ir1[2];
              ir1[2] = (v1164_data + (v1148_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (26)))));
              float v1170_data = ir1[3];
              ir1[3] = (v1170_data + (v1148_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (26)))));
              float v1176_data = ir1[4];
              ir1[4] = (v1176_data + (v1148_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (26)))));
              float v1182_data = ir1[5];
              ir1[5] = (v1182_data + (v1148_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (26)))));
              float v1186_data_pre = glb_m1[v806_g ? ((v803_rank + 442)) : (0)];
              float v1186_data = v806_g ? (v1186_data_pre) : (0.0f);
              float v1190_data = ir1[0];
              ir1[0] = (v1190_data + (v1186_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (27)))));
              float v1196_data = ir1[1];
              ir1[1] = (v1196_data + (v1186_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (27)))));
              float v1202_data = ir1[2];
              ir1[2] = (v1202_data + (v1186_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (27)))));
              float v1208_data = ir1[3];
              ir1[3] = (v1208_data + (v1186_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (27)))));
              float v1214_data = ir1[4];
              ir1[4] = (v1214_data + (v1186_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (27)))));
              float v1220_data = ir1[5];
              ir1[5] = (v1220_data + (v1186_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (27)))));
              float v1229_data_pre = glb_m1[(((3681400512 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((3681400512 & v43_bit))) + 460)) : (0)];
              float v1229_data = (((3681400512 >> v2078_lead) & 1) != 0) ? (v1229_data_pre) : (0.0f);
              float v1233_data = ir1[0];
              ir1[0] = (v1233_data + (v1229_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (28)))));
              float v1239_data = ir1[1];
              ir1[1] = (v1239_data + (v1229_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (28)))));
              float v1245_data = ir1[2];
              ir1[2] = (v1245_data + (v1229_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (28)))));
              float v1251_data = ir1[3];
              ir1[3] = (v1251_data + (v1229_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (28)))));
              float v1257_data = ir1[4];
              ir1[4] = (v1257_data + (v1229_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (28)))));
              float v1263_data = ir1[5];
              ir1[5] = (v1263_data + (v1229_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (28)))));
              float v1267_data_pre = glb_m1[v892_g ? ((v889_rank + 478)) : (0)];
              float v1267_data = v892_g ? (v1267_data_pre) : (0.0f);
              float v1271_data = ir1[0];
              ir1[0] = (v1271_data + (v1267_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (29)))));
              float v1277_data = ir1[1];
              ir1[1] = (v1277_data + (v1267_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (29)))));
              float v1283_data = ir1[2];
              ir1[2] = (v1283_data + (v1267_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (29)))));
              float v1289_data = ir1[3];
              ir1[3] = (v1289_data + (v1267_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (29)))));
              float v1295_data = ir1[4];
              ir1[4] = (v1295_data + (v1267_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (29)))));
              float v1301_data = ir1[5];
              ir1[5] = (v1301_data + (v1267_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (29)))));
              float v1310_data_pre = glb_m1[(((47934848 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((47934848 & v43_bit))) + 496)) : (0)];
              float v1310_data = (((47934848 >> v2078_lead) & 1) != 0) ? (v1310_data_pre) : (0.0f);
              float v1314_data = ir1[0];
              ir1[0] = (v1314_data + (v1310_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (30)))));
              float v1320_data = ir1[1];
              ir1[1] = (v1320_data + (v1310_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (30)))));
              float v1326_data = ir1[2];
              ir1[2] = (v1326_data + (v1310_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (30)))));
              float v1332_data = ir1[3];
              ir1[3] = (v1332_data + (v1310_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (30)))));
              float v1338_data = ir1[4];
              ir1[4] = (v1338_data + (v1310_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (30)))));
              float v1344_data = ir1[5];
              ir1[5] = (v1344_data + (v1310_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (30)))));
              float v1348_data_pre = glb_m1[v978_g ? ((v975_rank + 509)) : (0)];
              float v1348_data = v978_g ? (v1348_data_pre) : (0.0f);
              float v1352_data = ir1[0];
              ir1[0] = (v1352_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (31)))));
              float v1358_data = ir1[1];
              ir1[1] = (v1358_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (31)))));
              float v1364_data = ir1[2];
              ir1[2] = (v1364_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (31)))));
              float v1370_data = ir1[3];
              ir1[3] = (v1370_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (31)))));
              float v1376_data = ir1[4];
              ir1[4] = (v1376_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (31)))));
              float v1382_data = ir1[5];
              ir1[5] = (v1382_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (31)))));
              float v1391_data_pre = glb_m1[(((95869696 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((95869696 & v43_bit))) + 527)) : (0)];
              float v1391_data = (((95869696 >> v2078_lead) & 1) != 0) ? (v1391_data_pre) : (0.0f);
              float v1392_data = r0[1];
              float v1395_data = ir1[0];
              ir1[0] = (v1395_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1398_data = r0[3];
              float v1401_data = ir1[1];
              ir1[1] = (v1401_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1404_data = r0[5];
              float v1407_data = ir1[2];
              ir1[2] = (v1407_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1410_data = r0[7];
              float v1413_data = ir1[3];
              ir1[3] = (v1413_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1416_data = r0[9];
              float v1419_data = ir1[4];
              ir1[4] = (v1419_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1422_data = r0[11];
              float v1425_data = ir1[5];
              ir1[5] = (v1425_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1429_data_pre = glb_m1[v1064_g ? ((v1061_rank + 540)) : (0)];
              float v1429_data = v1064_g ? (v1429_data_pre) : (0.0f);
              float v1433_data = ir1[0];
              ir1[0] = (v1433_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1439_data = ir1[1];
              ir1[1] = (v1439_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1445_data = ir1[2];
              ir1[2] = (v1445_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1451_data = ir1[3];
              ir1[3] = (v1451_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1457_data = ir1[4];
              ir1[4] = (v1457_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1463_data = ir1[5];
              ir1[5] = (v1463_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1472_data_pre = glb_m1[(((191739392 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((191739392 & v43_bit))) + 558)) : (0)];
              float v1472_data = (((191739392 >> v2078_lead) & 1) != 0) ? (v1472_data_pre) : (0.0f);
              float v1476_data = ir1[0];
              ir1[0] = (v1476_data + (v1472_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1482_data = ir1[1];
              ir1[1] = (v1482_data + (v1472_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1488_data = ir1[2];
              ir1[2] = (v1488_data + (v1472_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1494_data = ir1[3];
              ir1[3] = (v1494_data + (v1472_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1500_data = ir1[4];
              ir1[4] = (v1500_data + (v1472_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1506_data = ir1[5];
              ir1[5] = (v1506_data + (v1472_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1510_data_pre = glb_m1[v1145_g ? ((v1142_rank + 571)) : (0)];
              float v1510_data = v1145_g ? (v1510_data_pre) : (0.0f);
              float v1514_data = ir1[0];
              ir1[0] = (v1514_data + (v1510_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1520_data = ir1[1];
              ir1[1] = (v1520_data + (v1510_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1526_data = ir1[2];
              ir1[2] = (v1526_data + (v1510_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1532_data = ir1[3];
              ir1[3] = (v1532_data + (v1510_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1538_data = ir1[4];
              ir1[4] = (v1538_data + (v1510_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1544_data = ir1[5];
              ir1[5] = (v1544_data + (v1510_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1553_data_pre = glb_m1[(((383479222 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((383479222 & v43_bit))) + 589)) : (0)];
              float v1553_data = (((383479222 >> v2078_lead) & 1) != 0) ? (v1553_data_pre) : (0.0f);
              float v1556_p = v1553_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)));
              float v1557_data = ir1[0];
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v2118_i0 = 0; v2118_i0 < 1; ++v2118_i0) {
                int32_t v2120_lead = v2078_lead + (v2118_i0 * 32);
                #pragma unroll
                for (int32_t v2121_i1 = 0; v2121_i1 < 6; ++v2121_i1) {
                  float v2124_data = wrap_glb_m2[(v2120_lead + (v2121_i1 * 48))];
                  r0[(v2118_i0 + (v2121_i1 * 2))] = v2124_data;
                }
              }
              #pragma unroll
              for (int32_t v2127_i1 = 0; v2127_i1 < 6; ++v2127_i1) {
                float v2130_data_pre = wrap_glb_m2[v2081_g ? ((v2080_lead + (v2127_i1 * 48))) : (0)];
                float v2130_data = v2081_g ? (v2130_data_pre) : (0.0f);
                r0[(1 + (v2127_i1 * 2))] = v2130_data;
              }
              ir1[0] = (v1557_data + v1556_p);
              float v1563_data = ir1[1];
              ir1[1] = (v1563_data + (v1553_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1569_data = ir1[2];
              ir1[2] = (v1569_data + (v1553_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1575_data = ir1[3];
              ir1[3] = (v1575_data + (v1553_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1581_data = ir1[4];
              ir1[4] = (v1581_data + (v1553_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1587_data = ir1[5];
              ir1[5] = (v1587_data + (v1553_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              int32_t v1590_rank = sycl::popcount((57521856 & v43_bit));
              bool v1593_g = ((57521856 >> v2078_lead) & 1) != 0;
              float v1596_data_pre = glb_m1[v1593_g ? ((v1590_rank + 608)) : (0)];
              float v1596_data = v1593_g ? (v1596_data_pre) : (0.0f);
              float v1600_data = ir1[0];
              ir1[0] = (v1600_data + (v1596_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1606_data = ir1[1];
              ir1[1] = (v1606_data + (v1596_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1612_data = ir1[2];
              ir1[2] = (v1612_data + (v1596_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1618_data = ir1[3];
              ir1[3] = (v1618_data + (v1596_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1624_data = ir1[4];
              ir1[4] = (v1624_data + (v1596_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1630_data = ir1[5];
              ir1[5] = (v1630_data + (v1596_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1639_data_pre = glb_m1[(((766958444 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((766958444 & v43_bit))) + 622)) : (0)];
              float v1639_data = (((766958444 >> v2078_lead) & 1) != 0) ? (v1639_data_pre) : (0.0f);
              float v1643_data = ir1[0];
              ir1[0] = (v1643_data + (v1639_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1649_data = ir1[1];
              ir1[1] = (v1649_data + (v1639_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1655_data = ir1[2];
              ir1[2] = (v1655_data + (v1639_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1661_data = ir1[3];
              ir1[3] = (v1661_data + (v1639_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1667_data = ir1[4];
              ir1[4] = (v1667_data + (v1639_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1673_data = ir1[5];
              ir1[5] = (v1673_data + (v1639_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1682_data_pre = glb_m1[(((115043712 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((115043712 & v43_bit))) + 641)) : (0)];
              float v1682_data = (((115043712 >> v2078_lead) & 1) != 0) ? (v1682_data_pre) : (0.0f);
              float v1686_data = ir1[0];
              ir1[0] = (v1686_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1692_data = ir1[1];
              ir1[1] = (v1692_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1698_data = ir1[2];
              ir1[2] = (v1698_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1704_data = ir1[3];
              ir1[3] = (v1704_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1710_data = ir1[4];
              ir1[4] = (v1710_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1716_data = ir1[5];
              ir1[5] = (v1716_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1725_data_pre = glb_m1[(((1533916888 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((1533916888 & v43_bit))) + 655)) : (0)];
              float v1725_data = (((1533916888 >> v2078_lead) & 1) != 0) ? (v1725_data_pre) : (0.0f);
              float v1729_data = ir1[0];
              ir1[0] = (v1729_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1735_data = ir1[1];
              ir1[1] = (v1735_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1741_data = ir1[2];
              ir1[2] = (v1741_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1747_data = ir1[3];
              ir1[3] = (v1747_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1753_data = ir1[4];
              ir1[4] = (v1753_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1759_data = ir1[5];
              ir1[5] = (v1759_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1768_data_pre = glb_m1[(((230087424 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((230087424 & v43_bit))) + 674)) : (0)];
              float v1768_data = (((230087424 >> v2078_lead) & 1) != 0) ? (v1768_data_pre) : (0.0f);
              float v1772_data = ir1[0];
              ir1[0] = (v1772_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1778_data = ir1[1];
              ir1[1] = (v1778_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1784_data = ir1[2];
              ir1[2] = (v1784_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1790_data = ir1[3];
              ir1[3] = (v1790_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1796_data = ir1[4];
              ir1[4] = (v1796_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1802_data = ir1[5];
              ir1[5] = (v1802_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1811_data_pre = glb_m1[(((3067833776 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((3067833776 & v43_bit))) + 688)) : (0)];
              float v1811_data = (((3067833776 >> v2078_lead) & 1) != 0) ? (v1811_data_pre) : (0.0f);
              float v1815_data = ir1[0];
              ir1[0] = (v1815_data + (v1811_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1821_data = ir1[1];
              ir1[1] = (v1821_data + (v1811_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1827_data = ir1[2];
              ir1[2] = (v1827_data + (v1811_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1833_data = ir1[3];
              ir1[3] = (v1833_data + (v1811_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1839_data = ir1[4];
              ir1[4] = (v1839_data + (v1811_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1845_data = ir1[5];
              ir1[5] = (v1845_data + (v1811_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1854_data_pre = glb_m1[(((460174848 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((460174848 & v43_bit))) + 707)) : (0)];
              float v1854_data = (((460174848 >> v2078_lead) & 1) != 0) ? (v1854_data_pre) : (0.0f);
              float v1858_data = ir1[0];
              ir1[0] = (v1858_data + (v1854_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1864_data = ir1[1];
              ir1[1] = (v1864_data + (v1854_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1870_data = ir1[2];
              ir1[2] = (v1870_data + (v1854_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1876_data = ir1[3];
              ir1[3] = (v1876_data + (v1854_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1882_data = ir1[4];
              ir1[4] = (v1882_data + (v1854_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1888_data = ir1[5];
              ir1[5] = (v1888_data + (v1854_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1897_data_pre = glb_m1[(((28760928 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((28760928 & v43_bit))) + 721)) : (0)];
              float v1897_data = (((28760928 >> v2078_lead) & 1) != 0) ? (v1897_data_pre) : (0.0f);
              float v1901_data = ir1[0];
              ir1[0] = (v1901_data + (v1897_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1907_data = ir1[1];
              ir1[1] = (v1907_data + (v1897_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1913_data = ir1[2];
              ir1[2] = (v1913_data + (v1897_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1919_data = ir1[3];
              ir1[3] = (v1919_data + (v1897_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1925_data = ir1[4];
              ir1[4] = (v1925_data + (v1897_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1931_data = ir1[5];
              ir1[5] = (v1931_data + (v1897_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1940_data_pre = glb_m1[(((920350134 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((920350134 & v43_bit))) + 735)) : (0)];
              float v1940_data = (((920350134 >> v2078_lead) & 1) != 0) ? (v1940_data_pre) : (0.0f);
              float v1944_data = ir1[0];
              ir1[0] = (v1944_data + (v1940_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v1950_data = ir1[1];
              ir1[1] = (v1950_data + (v1940_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v1956_data = ir1[2];
              ir1[2] = (v1956_data + (v1940_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v1962_data = ir1[3];
              ir1[3] = (v1962_data + (v1940_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v1968_data = ir1[4];
              ir1[4] = (v1968_data + (v1940_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v1974_data = ir1[5];
              ir1[5] = (v1974_data + (v1940_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v1978_data_pre = glb_m1[v1593_g ? ((v1590_rank + 755)) : (0)];
              float v1978_data = v1593_g ? (v1978_data_pre) : (0.0f);
              float v1982_data = ir1[0];
              ir1[0] = (v1982_data + (v1978_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v1988_data = ir1[1];
              ir1[1] = (v1988_data + (v1978_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v1994_data = ir1[2];
              ir1[2] = (v1994_data + (v1978_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v2000_data = ir1[3];
              ir1[3] = (v2000_data + (v1978_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v2006_data = ir1[4];
              ir1[4] = (v2006_data + (v1978_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v2012_data = ir1[5];
              ir1[5] = (v2012_data + (v1978_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v2021_data_pre = glb_m1[(((1840700268 >> v2078_lead) & 1) != 0) ? (((sycl::popcount((1840700268 & v43_bit))) + 769)) : (0)];
              float v2021_data = (((1840700268 >> v2078_lead) & 1) != 0) ? (v2021_data_pre) : (0.0f);
              float v2025_data = ir1[0];
              ir1[0] = (v2025_data + (v2021_data * (sycl::select_from_group(item.get_sub_group(), v1392_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v2031_data = ir1[1];
              ir1[1] = (v2031_data + (v2021_data * (sycl::select_from_group(item.get_sub_group(), v1398_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v2037_data = ir1[2];
              ir1[2] = (v2037_data + (v2021_data * (sycl::select_from_group(item.get_sub_group(), v1404_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v2043_data = ir1[3];
              ir1[3] = (v2043_data + (v2021_data * (sycl::select_from_group(item.get_sub_group(), v1410_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v2049_data = ir1[4];
              ir1[4] = (v2049_data + (v2021_data * (sycl::select_from_group(item.get_sub_group(), v1416_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v2055_data = ir1[5];
              ir1[5] = (v2055_data + (v2021_data * (sycl::select_from_group(item.get_sub_group(), v1422_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              // r1 = ir1
              #pragma unroll
              for (int32_t v2057_n0 = 0; v2057_n0 < 1; ++v2057_n0) {
                #pragma unroll
                for (int32_t v2058_n1 = 0; v2058_n1 < 6; ++v2058_n1) {
                  int32_t v2059_a = v2057_n0 + v2058_n1;
                  float v2060_data = ir1[v2059_a];
                  r1[v2059_a] = v2060_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v2061_i0 = 0; v2061_i0 < 1; ++v2061_i0) {
                int32_t v2066_lead = v2078_lead + (v2061_i0 * 32);
                #pragma unroll
                for (int32_t v2062_i1 = 0; v2062_i1 < 6; ++v2062_i1) {
                  float v2064_data = r1[(v2061_i0 + v2062_i1)];
                  glb_m0[(v2066_lead + (v2062_i1 * 32))] = v2064_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v2088_i0 = 0; v2088_i0 < 1; ++v2088_i0) {
                int32_t v2090_lead = v2078_lead + (v2088_i0 * 32);
                #pragma unroll
                for (int32_t v2091_i1 = 0; v2091_i1 < 6; ++v2091_i1) {
                  float v2094_data = wrap_glb_m2[(v2090_lead + (v2091_i1 * 48))];
                  r0[(v2088_i0 + (v2091_i1 * 2))] = v2094_data;
                }
              }
              #pragma unroll
              for (int32_t v2097_i1 = 0; v2097_i1 < 6; ++v2097_i1) {
                float v2100_data_pre = wrap_glb_m2[v2081_g ? ((v2080_lead + (v2097_i1 * 48))) : (0)];
                float v2100_data = v2081_g ? (v2100_data_pre) : (0.0f);
                r0[(1 + (v2097_i1 * 2))] = v2100_data;
              }
            }
            v2074_flagWord = v2075_flagWordNext;
            v2075_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


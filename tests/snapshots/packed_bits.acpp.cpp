// === base name ===
kernel_d21dda8545818e5a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d21dda8545818e5a = {{32, 1, 1}, 32, 24, 1, 1, 192, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d21dda8545818e5a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d21dda8545818e5a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d21dda8545818e5a(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 48 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_d21dda8545818e5a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d21dda8545818e5a(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d21dda8545818e5a(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d21dda8545818e5a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (48, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: pack_operands=bits,preload_globals=1
        // launch: 32 lanes (24 active) x 1 per block = block 32x1x1, 192 B shared, occupancy grid
        // operands:
        //   m0 24×6(24×6) {0..24}×{0..6} strided
        //   m1 24×7(24×7) {0..24}×{0..7} none ordered bits-packed
        //   m2 7×6(7×6) {0..7}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":48}],"shared_bytes":192,"shared_elements":48,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[24,6]],"name":"m0","ordered":false,"parts":1,"shape":[24,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[24,7]],"name":"m1","ordered":true,"parts":1,"shape":[24,7],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[7,6]],"name":"m2","ordered":false,"parts":1,"shape":[7,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[24,7]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,7]},{"addressing":"strided","bbox":[[0,0],[7,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[7,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[0 * item.get_local_id(1) + 48];
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          float * __restrict__ glb_m1 = &totalShrMem[0];
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          float v9_ld = ptr_glb_m1[0 + 0 + 1 * (item.get_local_id(2) + item.get_local_id(1) * item.get_group().get_local_range(2)) + 0];
          glb_m1[0 + 0 + 1 * (item.get_local_id(2) + item.get_local_id(1) * item.get_group().get_local_range(2)) + 0] = v9_ld;
          if ((item.get_local_id(2) + (item.get_local_id(1) * 32)) < 15) {
            float v15_ld = ptr_glb_m1[0 + 0 + 1 * (item.get_local_id(2) + item.get_local_id(1) * item.get_group().get_local_range(2)) + 32];
            glb_m1[0 + 0 + 1 * (item.get_local_id(2) + item.get_local_id(1) * item.get_group().get_local_range(2)) + 32] = v15_ld;
          }
          item.barrier();
          float r0[6]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v363_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v363_flagSecondAhead < numElements0) ? v363_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 42 + 0 + m2_extraOffset];
          int32_t v371_lead = item.get_local_id(2) % 32;
          bool v372_g = v371_lead < 7;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v381_i1 = 0; v381_i1 < 6; ++v381_i1) {
            float v384_data_pre = peel_glb_m2[v372_g ? ((v371_lead + (v381_i1 * 7))) : (0)];
            float v384_data = v372_g ? (v384_data_pre) : (0.0f);
            r0[v381_i1] = v384_data;
          }
          uint32_t v367_flagWord = flagWordFirst;
          uint32_t v368_flagWordNext = flagWordSecond;
          for (size_t v16_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v16_batchId0 < numElements0; v16_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v17_ahead1 = v16_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v19_batchId1 = (v17_ahead1 < numElements0) ? v17_ahead1 : v16_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v19_batchId1 * 42 + 0 + m2_extraOffset];
            size_t v391_flagAheadAhead = v19_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v391_flagAheadAhead < numElements0) ? v391_flagAheadAhead : v19_batchId1)];
            const bool allowed = static_cast<bool>(v367_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v16_batchId0 * 144 + 0 + m0_extraOffset];
              float r1[6]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 24), (0, 6)] [(0, 7)]
              float ir1[6]{};
              int32_t v41_bit = (1 << v371_lead) - 1;
              bool v48_g = v371_lead < 24;
              float v50_data_pre = glb_m1[(v48_g && (((8390151 >> v371_lead) & 1) != 0)) ? ((sycl::popcount((8390151 & v41_bit)))) : (0)];
              float v50_data = (v48_g && (((8390151 >> v371_lead) & 1) != 0)) ? (v50_data_pre) : (0.0f);
              float v51_data = r0[0];
              ir1[0] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v57_data = r0[1];
              ir1[1] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v63_data = r0[2];
              ir1[2] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v69_data = r0[3];
              ir1[3] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v75_data = r0[4];
              ir1[4] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v81_data = r0[5];
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v385_i1 = 0; v385_i1 < 6; ++v385_i1) {
                float v388_data_pre = wrap_glb_m2[v372_g ? ((v371_lead + (v385_i1 * 7))) : (0)];
                float v388_data = v372_g ? (v388_data_pre) : (0.0f);
                r0[v385_i1] = v388_data;
              }
              ir1[5] = (v50_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v94_data_pre = glb_m1[(v48_g && (((16 >> v371_lead) & 1) != 0)) ? (((sycl::popcount((16 & v41_bit))) + 6)) : (0)];
              float v94_data = (v48_g && (((16 >> v371_lead) & 1) != 0)) ? (v94_data_pre) : (0.0f);
              float v98_data = ir1[0];
              ir1[0] = (v98_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v104_data = ir1[1];
              ir1[1] = (v104_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v110_data = ir1[2];
              ir1[2] = (v110_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v116_data = ir1[3];
              ir1[3] = (v116_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v122_data = ir1[4];
              ir1[4] = (v122_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v128_data = ir1[5];
              ir1[5] = (v128_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v138_data_pre = glb_m1[(v48_g && (((16777215 >> v371_lead) & 1) != 0)) ? (((sycl::popcount((16777215 & v41_bit))) + 7)) : (0)];
              float v138_data = (v48_g && (((16777215 >> v371_lead) & 1) != 0)) ? (v138_data_pre) : (0.0f);
              float v142_data = ir1[0];
              ir1[0] = (v142_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v148_data = ir1[1];
              ir1[1] = (v148_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v154_data = ir1[2];
              ir1[2] = (v154_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v160_data = ir1[3];
              ir1[3] = (v160_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v166_data = ir1[4];
              ir1[4] = (v166_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v172_data = ir1[5];
              ir1[5] = (v172_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v182_data_pre = glb_m1[(v48_g && (((8388609 >> v371_lead) & 1) != 0)) ? (((sycl::popcount((8388609 & v41_bit))) + 31)) : (0)];
              float v182_data = (v48_g && (((8388609 >> v371_lead) & 1) != 0)) ? (v182_data_pre) : (0.0f);
              float v186_data = ir1[0];
              ir1[0] = (v186_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v192_data = ir1[1];
              ir1[1] = (v192_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v198_data = ir1[2];
              ir1[2] = (v198_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v204_data = ir1[3];
              ir1[3] = (v204_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v210_data = ir1[4];
              ir1[4] = (v210_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v216_data = ir1[5];
              ir1[5] = (v216_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v226_data_pre = glb_m1[(v48_g && (((665768 >> v371_lead) & 1) != 0)) ? (((sycl::popcount((665768 & v41_bit))) + 33)) : (0)];
              float v226_data = (v48_g && (((665768 >> v371_lead) & 1) != 0)) ? (v226_data_pre) : (0.0f);
              float v230_data = ir1[0];
              ir1[0] = (v230_data + (v226_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v236_data = ir1[1];
              ir1[1] = (v236_data + (v226_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v242_data = ir1[2];
              ir1[2] = (v242_data + (v226_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v248_data = ir1[3];
              ir1[3] = (v248_data + (v226_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v254_data = ir1[4];
              ir1[4] = (v254_data + (v226_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v260_data = ir1[5];
              ir1[5] = (v260_data + (v226_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v270_data_pre = glb_m1[(v48_g && (((126976 >> v371_lead) & 1) != 0)) ? (((sycl::popcount((126976 & v41_bit))) + 40)) : (0)];
              float v270_data = (v48_g && (((126976 >> v371_lead) & 1) != 0)) ? (v270_data_pre) : (0.0f);
              float v274_data = ir1[0];
              ir1[0] = (v274_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v280_data = ir1[1];
              ir1[1] = (v280_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v286_data = ir1[2];
              ir1[2] = (v286_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v292_data = ir1[3];
              ir1[3] = (v292_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v298_data = ir1[4];
              ir1[4] = (v298_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v304_data = ir1[5];
              ir1[5] = (v304_data + (v270_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v314_data_pre = glb_m1[(v48_g && (((4194306 >> v371_lead) & 1) != 0)) ? (((sycl::popcount((4194306 & v41_bit))) + 45)) : (0)];
              float v314_data = (v48_g && (((4194306 >> v371_lead) & 1) != 0)) ? (v314_data_pre) : (0.0f);
              float v318_data = ir1[0];
              ir1[0] = (v318_data + (v314_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v324_data = ir1[1];
              ir1[1] = (v324_data + (v314_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v330_data = ir1[2];
              ir1[2] = (v330_data + (v314_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v336_data = ir1[3];
              ir1[3] = (v336_data + (v314_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v342_data = ir1[4];
              ir1[4] = (v342_data + (v314_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v348_data = ir1[5];
              ir1[5] = (v348_data + (v314_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              // r1 = ir1
              if (v48_g) {
                #pragma unroll
                for (int32_t v351_n1 = 0; v351_n1 < 6; ++v351_n1) {
                  float v353_data = ir1[v351_n1];
                  r1[v351_n1] = v353_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              if (v48_g) {
                #pragma unroll
                for (int32_t v355_i1 = 0; v355_i1 < 6; ++v355_i1) {
                  float v357_data = r1[v355_i1];
                  glb_m0[(v371_lead + (v355_i1 * 24))] = v357_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v377_i1 = 0; v377_i1 < 6; ++v377_i1) {
                float v380_data_pre = wrap_glb_m2[v372_g ? ((v371_lead + (v377_i1 * 7))) : (0)];
                float v380_data = v372_g ? (v380_data_pre) : (0.0f);
                r0[v377_i1] = v380_data;
              }
            }
            v367_flagWord = v368_flagWordNext;
            v368_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


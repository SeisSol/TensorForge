// === base name ===
kernel_dbb4b5a18285b867

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_dbb4b5a18285b867 = {{32, 1, 1}, 32, 64, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_dbb4b5a18285b867(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_dbb4b5a18285b867(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_dbb4b5a18285b867(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_dbb4b5a18285b867(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_dbb4b5a18285b867(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_dbb4b5a18285b867(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_dbb4b5a18285b867(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: lead_vectorize=1,pack_operands=runs
        // launch: 32 lanes (64 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 64×6(64×6) {0..64}×{0..6} strided
        //   m1 64×8(64×8) {0..64}×{0..8} none ordered runs-packed
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[64,6]],"name":"m0","ordered":false,"parts":1,"shape":[64,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[64,8]],"name":"m1","ordered":true,"parts":1,"shape":[64,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[64,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          float r0[6]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v487_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v487_flagSecondAhead < numElements0) ? v487_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          int32_t v495_lead = item.get_local_id(2) % 32;
          bool v496_g = v495_lead < 8;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v505_i1 = 0; v505_i1 < 6; ++v505_i1) {
            float v508_data_pre = peel_glb_m2[v496_g ? ((v495_lead + (v505_i1 * 8))) : (0)];
            float v508_data = v496_g ? (v508_data_pre) : (0.0f);
            r0[v505_i1] = v508_data;
          }
          uint32_t v491_flagWord = flagWordFirst;
          uint32_t v492_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v515_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v515_flagAheadAhead < numElements0) ? v515_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v491_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 384 + 0 + m0_extraOffset];
              float r1[12]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 64), (0, 6)] [(0, 8)]
              float ir1[12]{};
              bool v37_g = v495_lead < 7;
              float v39_data_pre = glb_m1[((v495_lead >= 1) && v37_g) ? ((v495_lead + -1)) : (0)];
              float v39_data = ((v495_lead >= 1) && v37_g) ? (v39_data_pre) : (0.0f);
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
              for (int32_t v509_i1 = 0; v509_i1 < 6; ++v509_i1) {
                float v512_data_pre = wrap_glb_m2[v496_g ? ((v495_lead + (v509_i1 * 8))) : (0)];
                float v512_data = v496_g ? (v512_data_pre) : (0.0f);
                r0[v509_i1] = v512_data;
              }
              ir1[10] = (v39_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0))));
              float v80_data_pre = glb_m1[((v495_lead >= 9) && (v495_lead < 27)) ? ((v495_lead + -3)) : (0)];
              float v80_data = ((v495_lead >= 9) && (v495_lead < 27)) ? (v80_data_pre) : (0.0f);
              float v84_data = ir1[0];
              ir1[0] = (v84_data + (v80_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v90_data = ir1[2];
              ir1[2] = (v90_data + (v80_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v96_data = ir1[4];
              ir1[4] = (v96_data + (v80_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v102_data = ir1[6];
              ir1[6] = (v102_data + (v80_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v108_data = ir1[8];
              ir1[8] = (v108_data + (v80_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v114_data = ir1[10];
              ir1[10] = (v114_data + (v80_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v119_data_pre = glb_m1[(v495_lead >= 31) ? ((v495_lead + -7)) : (0)];
              float v119_data = (v495_lead >= 31) ? (v119_data_pre) : (0.0f);
              float v121_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v123_data = ir1[0];
              ir1[0] = (v123_data + (v119_data * v121_bc));
              float v127_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v129_data = ir1[2];
              ir1[2] = (v129_data + (v119_data * v127_bc));
              float v133_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v135_data = ir1[4];
              ir1[4] = (v135_data + (v119_data * v133_bc));
              float v139_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v141_data = ir1[6];
              ir1[6] = (v141_data + (v119_data * v139_bc));
              float v145_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v147_data = ir1[8];
              ir1[8] = (v147_data + (v119_data * v145_bc));
              float v151_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v153_data = ir1[10];
              ir1[10] = (v153_data + (v119_data * v151_bc));
              int32_t v156_lead = v495_lead + 32_i32;
              float v160_data_pre = glb_m1[(v495_lead < 2) ? ((v156_lead + -7)) : (0)];
              float v160_data = (v495_lead < 2) ? (v160_data_pre) : (0.0f);
              ir1[1] = (v160_data * v121_bc);
              ir1[3] = (v160_data * v127_bc);
              ir1[5] = (v160_data * v133_bc);
              ir1[7] = (v160_data * v139_bc);
              ir1[9] = (v160_data * v145_bc);
              ir1[11] = (v160_data * v151_bc);
              float v199_data_pre = glb_m1[(v495_lead >= 2) ? ((v495_lead + 25)) : (0)];
              float v199_data = (v495_lead >= 2) ? (v199_data_pre) : (0.0f);
              float v201_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v203_data = ir1[0];
              ir1[0] = (v203_data + (v199_data * v201_bc));
              float v207_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v209_data = ir1[2];
              ir1[2] = (v209_data + (v199_data * v207_bc));
              float v213_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v215_data = ir1[4];
              ir1[4] = (v215_data + (v199_data * v213_bc));
              float v219_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v221_data = ir1[6];
              ir1[6] = (v221_data + (v199_data * v219_bc));
              float v225_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v227_data = ir1[8];
              ir1[8] = (v227_data + (v199_data * v225_bc));
              float v231_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v233_data = ir1[10];
              ir1[10] = (v233_data + (v199_data * v231_bc));
              float v238_data_pre = glb_m1[(v495_lead < 30) ? ((v156_lead + 25)) : (0)];
              float v238_data = (v495_lead < 30) ? (v238_data_pre) : (0.0f);
              float v242_data = ir1[1];
              ir1[1] = (v242_data + (v238_data * v201_bc));
              float v248_data = ir1[3];
              ir1[3] = (v248_data + (v238_data * v207_bc));
              float v254_data = ir1[5];
              ir1[5] = (v254_data + (v238_data * v213_bc));
              float v260_data = ir1[7];
              ir1[7] = (v260_data + (v238_data * v219_bc));
              float v266_data = ir1[9];
              ir1[9] = (v266_data + (v238_data * v225_bc));
              float v272_data = ir1[11];
              ir1[11] = (v272_data + (v238_data * v231_bc));
              bool v276_g = v495_lead >= 13;
              float v279_data_pre = glb_m1[(v276_g && (v495_lead < 19)) ? ((v156_lead + 42)) : (0)];
              float v279_data = (v276_g && (v495_lead < 19)) ? (v279_data_pre) : (0.0f);
              float v283_data = ir1[1];
              ir1[1] = (v283_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v289_data = ir1[3];
              ir1[3] = (v289_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v295_data = ir1[5];
              ir1[5] = (v295_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v301_data = ir1[7];
              ir1[7] = (v301_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v307_data = ir1[9];
              ir1[9] = (v307_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v313_data = ir1[11];
              ir1[11] = (v313_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              bool v318_g = v495_lead < 23;
              float v320_data_pre = glb_m1[((v495_lead >= 7) && v318_g) ? ((v495_lead + 86)) : (0)];
              float v320_data = ((v495_lead >= 7) && v318_g) ? (v320_data_pre) : (0.0f);
              float v324_data = ir1[0];
              ir1[0] = (v324_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v330_data = ir1[2];
              ir1[2] = (v330_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v336_data = ir1[4];
              ir1[4] = (v336_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v342_data = ir1[6];
              ir1[6] = (v342_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v348_data = ir1[8];
              ir1[8] = (v348_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v354_data = ir1[10];
              ir1[10] = (v354_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v360_data_pre = glb_m1[((v495_lead >= 5) && v37_g) ? ((v156_lead + 72)) : (0)];
              float v360_data = ((v495_lead >= 5) && v37_g) ? (v360_data_pre) : (0.0f);
              float v364_data = ir1[1];
              ir1[1] = (v364_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v370_data = ir1[3];
              ir1[3] = (v370_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v376_data = ir1[5];
              ir1[5] = (v376_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v382_data = ir1[7];
              ir1[7] = (v382_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v388_data = ir1[9];
              ir1[9] = (v388_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v394_data = ir1[11];
              ir1[11] = (v394_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v398_data_pre = glb_m1[v276_g ? ((v495_lead + 98)) : (0)];
              float v398_data = v276_g ? (v398_data_pre) : (0.0f);
              float v400_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v402_data = ir1[0];
              ir1[0] = (v402_data + (v398_data * v400_bc));
              float v406_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v408_data = ir1[2];
              ir1[2] = (v408_data + (v398_data * v406_bc));
              float v412_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v414_data = ir1[4];
              ir1[4] = (v414_data + (v398_data * v412_bc));
              float v418_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v420_data = ir1[6];
              ir1[6] = (v420_data + (v398_data * v418_bc));
              float v424_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v426_data = ir1[8];
              ir1[8] = (v426_data + (v398_data * v424_bc));
              float v430_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v432_data = ir1[10];
              ir1[10] = (v432_data + (v398_data * v430_bc));
              float v436_data_pre = glb_m1[v318_g ? ((v156_lead + 98)) : (0)];
              float v436_data = v318_g ? (v436_data_pre) : (0.0f);
              float v440_data = ir1[1];
              ir1[1] = (v440_data + (v436_data * v400_bc));
              float v446_data = ir1[3];
              ir1[3] = (v446_data + (v436_data * v406_bc));
              float v452_data = ir1[5];
              ir1[5] = (v452_data + (v436_data * v412_bc));
              float v458_data = ir1[7];
              ir1[7] = (v458_data + (v436_data * v418_bc));
              float v464_data = ir1[9];
              ir1[9] = (v464_data + (v436_data * v424_bc));
              float v470_data = ir1[11];
              ir1[11] = (v470_data + (v436_data * v430_bc));
              // r1 = ir1
              #pragma unroll
              for (int32_t v472_n0 = 0; v472_n0 < 2; ++v472_n0) {
                #pragma unroll
                for (int32_t v473_n1 = 0; v473_n1 < 6; ++v473_n1) {
                  int32_t v475_a = v472_n0 + (v473_n1 * 2);
                  float v476_data = ir1[v475_a];
                  r1[v475_a] = v476_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v477_i0 = 0; v477_i0 < 2; ++v477_i0) {
                int32_t v483_lead = v495_lead + (v477_i0 * 32);
                #pragma unroll
                for (int32_t v478_i1 = 0; v478_i1 < 6; ++v478_i1) {
                  float v481_data = r1[(v477_i0 + (v478_i1 * 2))];
                  glb_m0[(v483_lead + (v478_i1 * 64))] = v481_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v501_i1 = 0; v501_i1 < 6; ++v501_i1) {
                float v504_data_pre = wrap_glb_m2[v496_g ? ((v495_lead + (v501_i1 * 8))) : (0)];
                float v504_data = v496_g ? (v504_data_pre) : (0.0f);
                r0[v501_i1] = v504_data;
              }
            }
            v491_flagWord = v492_flagWordNext;
            v492_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


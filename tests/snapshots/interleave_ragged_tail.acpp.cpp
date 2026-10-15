// === base name ===
kernel_667ce17932a0f92e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_667ce17932a0f92e = {{32, 1, 1}, 32, 40, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_667ce17932a0f92e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_667ce17932a0f92e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_667ce17932a0f92e(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_667ce17932a0f92e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_667ce17932a0f92e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_667ce17932a0f92e(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_667ce17932a0f92e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (40 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none ordered
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":true,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          float r0[6]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v742_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v742_flagSecondAhead < numElements0) ? v742_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          int32_t v750_lead = item.get_local_id(2) % 32;
          bool v751_g = v750_lead < 8;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v760_i1 = 0; v760_i1 < 6; ++v760_i1) {
            float v763_data_pre = peel_glb_m2[v751_g ? ((v750_lead + (v760_i1 * 8))) : (0)];
            float v763_data = v751_g ? (v763_data_pre) : (0.0f);
            r0[v760_i1] = v763_data;
          }
          uint32_t v746_flagWord = flagWordFirst;
          uint32_t v747_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v770_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v770_flagAheadAhead < numElements0) ? v770_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v746_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 240 + 0 + m0_extraOffset];
              float r1[12]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 40), (0, 6)] [(0, 8)]
              float ir1[12]{};
              int32_t v32_a = v750_lead * 2;
              sycl::vec<float, 2> v34_data = *(sycl::vec<float, 2>*)&glb_m1[v32_a];
              float v35_p = v34_data[0];
              float v36_data = r0[0];
              float v37_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              ir1[0] = (v35_p * v37_bc);
              float v43_data = r0[1];
              float v44_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              ir1[2] = (v35_p * v44_bc);
              float v50_data = r0[2];
              float v51_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              ir1[4] = (v35_p * v51_bc);
              float v57_data = r0[3];
              float v58_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              ir1[6] = (v35_p * v58_bc);
              float v64_data = r0[4];
              float v65_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              ir1[8] = (v35_p * v65_bc);
              float v71_data = r0[5];
              float v72_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              ir1[10] = (v35_p * v72_bc);
              float v77_p = v34_data[1];
              ir1[1] = (v77_p * v37_bc);
              ir1[3] = (v77_p * v44_bc);
              ir1[5] = (v77_p * v51_bc);
              ir1[7] = (v77_p * v58_bc);
              ir1[9] = (v77_p * v65_bc);
              ir1[11] = (v77_p * v72_bc);
              sycl::vec<float, 2> v119_data = *(sycl::vec<float, 2>*)&glb_m1[(64 + v32_a)];
              float v120_p = v119_data[0];
              float v122_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v124_data = ir1[0];
              ir1[0] = (v124_data + (v120_p * v122_bc));
              float v129_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v131_data = ir1[2];
              ir1[2] = (v131_data + (v120_p * v129_bc));
              float v136_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v138_data = ir1[4];
              ir1[4] = (v138_data + (v120_p * v136_bc));
              float v143_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v145_data = ir1[6];
              ir1[6] = (v145_data + (v120_p * v143_bc));
              float v150_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v152_data = ir1[8];
              ir1[8] = (v152_data + (v120_p * v150_bc));
              float v157_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v159_data = ir1[10];
              ir1[10] = (v159_data + (v120_p * v157_bc));
              float v162_p = v119_data[1];
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v764_i1 = 0; v764_i1 < 6; ++v764_i1) {
                float v767_data_pre = wrap_glb_m2[v751_g ? ((v750_lead + (v764_i1 * 8))) : (0)];
                float v767_data = v751_g ? (v767_data_pre) : (0.0f);
                r0[v764_i1] = v767_data;
              }
              float v166_data = ir1[1];
              ir1[1] = (v166_data + (v162_p * v122_bc));
              float v173_data = ir1[3];
              ir1[3] = (v173_data + (v162_p * v129_bc));
              float v180_data = ir1[5];
              ir1[5] = (v180_data + (v162_p * v136_bc));
              float v187_data = ir1[7];
              ir1[7] = (v187_data + (v162_p * v143_bc));
              float v194_data = ir1[9];
              ir1[9] = (v194_data + (v162_p * v150_bc));
              float v201_data = ir1[11];
              ir1[11] = (v201_data + (v162_p * v157_bc));
              sycl::vec<float, 2> v204_data = *(sycl::vec<float, 2>*)&glb_m1[(128 + v32_a)];
              float v205_p = v204_data[0];
              float v207_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v209_data = ir1[0];
              ir1[0] = (v209_data + (v205_p * v207_bc));
              float v214_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v216_data = ir1[2];
              ir1[2] = (v216_data + (v205_p * v214_bc));
              float v221_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v223_data = ir1[4];
              ir1[4] = (v223_data + (v205_p * v221_bc));
              float v228_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v230_data = ir1[6];
              ir1[6] = (v230_data + (v205_p * v228_bc));
              float v235_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v237_data = ir1[8];
              ir1[8] = (v237_data + (v205_p * v235_bc));
              float v242_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v244_data = ir1[10];
              ir1[10] = (v244_data + (v205_p * v242_bc));
              float v247_p = v204_data[1];
              float v251_data = ir1[1];
              ir1[1] = (v251_data + (v247_p * v207_bc));
              float v258_data = ir1[3];
              ir1[3] = (v258_data + (v247_p * v214_bc));
              float v265_data = ir1[5];
              ir1[5] = (v265_data + (v247_p * v221_bc));
              float v272_data = ir1[7];
              ir1[7] = (v272_data + (v247_p * v228_bc));
              float v279_data = ir1[9];
              ir1[9] = (v279_data + (v247_p * v235_bc));
              float v286_data = ir1[11];
              ir1[11] = (v286_data + (v247_p * v242_bc));
              sycl::vec<float, 2> v289_data = *(sycl::vec<float, 2>*)&glb_m1[(192 + v32_a)];
              float v290_p = v289_data[0];
              float v292_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v294_data = ir1[0];
              ir1[0] = (v294_data + (v290_p * v292_bc));
              float v299_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v301_data = ir1[2];
              ir1[2] = (v301_data + (v290_p * v299_bc));
              float v306_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v308_data = ir1[4];
              ir1[4] = (v308_data + (v290_p * v306_bc));
              float v313_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v315_data = ir1[6];
              ir1[6] = (v315_data + (v290_p * v313_bc));
              float v320_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v322_data = ir1[8];
              ir1[8] = (v322_data + (v290_p * v320_bc));
              float v327_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v329_data = ir1[10];
              ir1[10] = (v329_data + (v290_p * v327_bc));
              float v332_p = v289_data[1];
              float v336_data = ir1[1];
              ir1[1] = (v336_data + (v332_p * v292_bc));
              float v343_data = ir1[3];
              ir1[3] = (v343_data + (v332_p * v299_bc));
              float v350_data = ir1[5];
              ir1[5] = (v350_data + (v332_p * v306_bc));
              float v357_data = ir1[7];
              ir1[7] = (v357_data + (v332_p * v313_bc));
              float v364_data = ir1[9];
              ir1[9] = (v364_data + (v332_p * v320_bc));
              float v371_data = ir1[11];
              ir1[11] = (v371_data + (v332_p * v327_bc));
              sycl::vec<float, 2> v374_data = *(sycl::vec<float, 2>*)&glb_m1[(256 + v32_a)];
              float v375_p = v374_data[0];
              float v377_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v379_data = ir1[0];
              ir1[0] = (v379_data + (v375_p * v377_bc));
              float v384_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v386_data = ir1[2];
              ir1[2] = (v386_data + (v375_p * v384_bc));
              float v391_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v393_data = ir1[4];
              ir1[4] = (v393_data + (v375_p * v391_bc));
              float v398_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v400_data = ir1[6];
              ir1[6] = (v400_data + (v375_p * v398_bc));
              float v405_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v407_data = ir1[8];
              ir1[8] = (v407_data + (v375_p * v405_bc));
              float v412_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v414_data = ir1[10];
              ir1[10] = (v414_data + (v375_p * v412_bc));
              float v417_p = v374_data[1];
              float v421_data = ir1[1];
              ir1[1] = (v421_data + (v417_p * v377_bc));
              float v428_data = ir1[3];
              ir1[3] = (v428_data + (v417_p * v384_bc));
              float v435_data = ir1[5];
              ir1[5] = (v435_data + (v417_p * v391_bc));
              float v442_data = ir1[7];
              ir1[7] = (v442_data + (v417_p * v398_bc));
              float v449_data = ir1[9];
              ir1[9] = (v449_data + (v417_p * v405_bc));
              float v456_data = ir1[11];
              ir1[11] = (v456_data + (v417_p * v412_bc));
              sycl::vec<float, 2> v459_data = *(sycl::vec<float, 2>*)&glb_m1[(320 + v32_a)];
              float v460_p = v459_data[0];
              float v462_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v464_data = ir1[0];
              ir1[0] = (v464_data + (v460_p * v462_bc));
              float v469_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v471_data = ir1[2];
              ir1[2] = (v471_data + (v460_p * v469_bc));
              float v476_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v478_data = ir1[4];
              ir1[4] = (v478_data + (v460_p * v476_bc));
              float v483_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v485_data = ir1[6];
              ir1[6] = (v485_data + (v460_p * v483_bc));
              float v490_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v492_data = ir1[8];
              ir1[8] = (v492_data + (v460_p * v490_bc));
              float v497_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v499_data = ir1[10];
              ir1[10] = (v499_data + (v460_p * v497_bc));
              float v502_p = v459_data[1];
              float v506_data = ir1[1];
              ir1[1] = (v506_data + (v502_p * v462_bc));
              float v513_data = ir1[3];
              ir1[3] = (v513_data + (v502_p * v469_bc));
              float v520_data = ir1[5];
              ir1[5] = (v520_data + (v502_p * v476_bc));
              float v527_data = ir1[7];
              ir1[7] = (v527_data + (v502_p * v483_bc));
              float v534_data = ir1[9];
              ir1[9] = (v534_data + (v502_p * v490_bc));
              float v541_data = ir1[11];
              ir1[11] = (v541_data + (v502_p * v497_bc));
              sycl::vec<float, 2> v544_data = *(sycl::vec<float, 2>*)&glb_m1[(384 + v32_a)];
              float v545_p = v544_data[0];
              float v547_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v549_data = ir1[0];
              ir1[0] = (v549_data + (v545_p * v547_bc));
              float v554_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v556_data = ir1[2];
              ir1[2] = (v556_data + (v545_p * v554_bc));
              float v561_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v563_data = ir1[4];
              ir1[4] = (v563_data + (v545_p * v561_bc));
              float v568_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v570_data = ir1[6];
              ir1[6] = (v570_data + (v545_p * v568_bc));
              float v575_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v577_data = ir1[8];
              ir1[8] = (v577_data + (v545_p * v575_bc));
              float v582_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v584_data = ir1[10];
              ir1[10] = (v584_data + (v545_p * v582_bc));
              float v587_p = v544_data[1];
              float v591_data = ir1[1];
              ir1[1] = (v591_data + (v587_p * v547_bc));
              float v598_data = ir1[3];
              ir1[3] = (v598_data + (v587_p * v554_bc));
              float v605_data = ir1[5];
              ir1[5] = (v605_data + (v587_p * v561_bc));
              float v612_data = ir1[7];
              ir1[7] = (v612_data + (v587_p * v568_bc));
              float v619_data = ir1[9];
              ir1[9] = (v619_data + (v587_p * v575_bc));
              float v626_data = ir1[11];
              ir1[11] = (v626_data + (v587_p * v582_bc));
              sycl::vec<float, 2> v629_data = *(sycl::vec<float, 2>*)&glb_m1[(448 + v32_a)];
              float v630_p = v629_data[0];
              float v632_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v634_data = ir1[0];
              ir1[0] = (v634_data + (v630_p * v632_bc));
              float v639_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v641_data = ir1[2];
              ir1[2] = (v641_data + (v630_p * v639_bc));
              float v646_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v648_data = ir1[4];
              ir1[4] = (v648_data + (v630_p * v646_bc));
              float v653_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v655_data = ir1[6];
              ir1[6] = (v655_data + (v630_p * v653_bc));
              float v660_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v662_data = ir1[8];
              ir1[8] = (v662_data + (v630_p * v660_bc));
              float v667_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v669_data = ir1[10];
              ir1[10] = (v669_data + (v630_p * v667_bc));
              float v672_p = v629_data[1];
              float v676_data = ir1[1];
              ir1[1] = (v676_data + (v672_p * v632_bc));
              float v683_data = ir1[3];
              ir1[3] = (v683_data + (v672_p * v639_bc));
              float v690_data = ir1[5];
              ir1[5] = (v690_data + (v672_p * v646_bc));
              float v697_data = ir1[7];
              ir1[7] = (v697_data + (v672_p * v653_bc));
              float v704_data = ir1[9];
              ir1[9] = (v704_data + (v672_p * v660_bc));
              float v711_data = ir1[11];
              ir1[11] = (v711_data + (v672_p * v667_bc));
              // r1 = ir1
              #pragma unroll
              for (int32_t v713_n0 = 0; v713_n0 < 1; ++v713_n0) {
                #pragma unroll
                for (int32_t v714_n1 = 0; v714_n1 < 6; ++v714_n1) {
                  int32_t v716_a = v713_n0 + (v714_n1 * 2);
                  float v717_data = ir1[v716_a];
                  r1[v716_a] = v717_data;
                }
              }
              if (v751_g) {
                #pragma unroll
                for (int32_t v719_n1 = 0; v719_n1 < 6; ++v719_n1) {
                  int32_t v721_a = 1 + (v719_n1 * 2);
                  float v722_data = ir1[v721_a];
                  r1[v721_a] = v722_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v723_i0 = 0; v723_i0 < 1; ++v723_i0) {
                int32_t v729_lead = v750_lead + (v723_i0 * 32);
                #pragma unroll
                for (int32_t v724_i1 = 0; v724_i1 < 6; ++v724_i1) {
                  float v727_data = r1[(v723_i0 + (v724_i1 * 2))];
                  glb_m0[(v729_lead + (v724_i1 * 40))] = v727_data;
                }
              }
              if (v751_g) {
                int32_t v738_lead = v750_lead + 32_i32;
                #pragma unroll
                for (int32_t v733_i1 = 0; v733_i1 < 6; ++v733_i1) {
                  float v736_data = r1[(1 + (v733_i1 * 2))];
                  glb_m0[(v738_lead + (v733_i1 * 40))] = v736_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v756_i1 = 0; v756_i1 < 6; ++v756_i1) {
                float v759_data_pre = wrap_glb_m2[v751_g ? ((v750_lead + (v756_i1 * 8))) : (0)];
                float v759_data = v751_g ? (v759_data_pre) : (0.0f);
                r0[v756_i1] = v759_data;
              }
            }
            v746_flagWord = v747_flagWordNext;
            v747_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


// === base name ===
kernel_96dc8646b74f36e2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_96dc8646b74f36e2 = {{32, 1, 1}, 32, 56, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_96dc8646b74f36e2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_96dc8646b74f36e2(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_96dc8646b74f36e2(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_96dc8646b74f36e2(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_96dc8646b74f36e2(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_96dc8646b74f36e2(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_96dc8646b74f36e2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: k_roll=2,trip_reads=1
        // launch: 32 lanes (56 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 56×6(56×6) {0..56}×{0..6} strided
        //   m1 56×16(56×16) {0..56}×{0..16} none ordered
        //   m2 16×6(16×6) {0..16}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[56,6]],"name":"m0","ordered":false,"parts":1,"shape":[56,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[56,16]],"name":"m1","ordered":true,"parts":1,"shape":[56,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,6]],"name":"m2","ordered":false,"parts":1,"shape":[16,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[56,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,16]},{"addressing":"strided","bbox":[[0,0],[16,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          float r0[6]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v1422_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1422_flagSecondAhead < numElements0) ? v1422_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 96 + 0 + m2_extraOffset];
          int32_t v1430_lead = item.get_local_id(2) % 32;
          bool v1431_g = v1430_lead < 16;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v1440_i1 = 0; v1440_i1 < 6; ++v1440_i1) {
            float v1443_data_pre = peel_glb_m2[v1431_g ? ((v1430_lead + (v1440_i1 * 16))) : (0)];
            float v1443_data = v1431_g ? (v1443_data_pre) : (0.0f);
            r0[v1440_i1] = v1443_data;
          }
          uint32_t v1426_flagWord = flagWordFirst;
          uint32_t v1427_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 96 + 0 + m2_extraOffset];
            size_t v1450_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1450_flagAheadAhead < numElements0) ? v1450_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v1426_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 336 + 0 + m0_extraOffset];
              float r1[12]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 56), (0, 6)] [(0, 16)]
              float ir1[12]{};
              int32_t v32_a = v1430_lead * 2;
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
              sycl::vec<float, 2> v714_data = *(sycl::vec<float, 2>*)&glb_m1[(512 + v32_a)];
              float v715_p = v714_data[0];
              float v717_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v719_data = ir1[0];
              ir1[0] = (v719_data + (v715_p * v717_bc));
              float v724_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v726_data = ir1[2];
              ir1[2] = (v726_data + (v715_p * v724_bc));
              float v731_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v733_data = ir1[4];
              ir1[4] = (v733_data + (v715_p * v731_bc));
              float v738_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v740_data = ir1[6];
              ir1[6] = (v740_data + (v715_p * v738_bc));
              float v745_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v747_data = ir1[8];
              ir1[8] = (v747_data + (v715_p * v745_bc));
              float v752_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8));
              float v754_data = ir1[10];
              ir1[10] = (v754_data + (v715_p * v752_bc));
              float v757_p = v714_data[1];
              float v761_data = ir1[1];
              ir1[1] = (v761_data + (v757_p * v717_bc));
              float v768_data = ir1[3];
              ir1[3] = (v768_data + (v757_p * v724_bc));
              float v775_data = ir1[5];
              ir1[5] = (v775_data + (v757_p * v731_bc));
              float v782_data = ir1[7];
              ir1[7] = (v782_data + (v757_p * v738_bc));
              float v789_data = ir1[9];
              ir1[9] = (v789_data + (v757_p * v745_bc));
              float v796_data = ir1[11];
              ir1[11] = (v796_data + (v757_p * v752_bc));
              sycl::vec<float, 2> v799_data = *(sycl::vec<float, 2>*)&glb_m1[(576 + v32_a)];
              float v800_p = v799_data[0];
              float v802_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v804_data = ir1[0];
              ir1[0] = (v804_data + (v800_p * v802_bc));
              float v809_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v811_data = ir1[2];
              ir1[2] = (v811_data + (v800_p * v809_bc));
              float v816_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v818_data = ir1[4];
              ir1[4] = (v818_data + (v800_p * v816_bc));
              float v823_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v825_data = ir1[6];
              ir1[6] = (v825_data + (v800_p * v823_bc));
              float v830_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v832_data = ir1[8];
              ir1[8] = (v832_data + (v800_p * v830_bc));
              float v837_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9));
              float v839_data = ir1[10];
              ir1[10] = (v839_data + (v800_p * v837_bc));
              float v842_p = v799_data[1];
              float v845_p = v842_p * v802_bc;
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v1444_i1 = 0; v1444_i1 < 6; ++v1444_i1) {
                float v1447_data_pre = wrap_glb_m2[v1431_g ? ((v1430_lead + (v1444_i1 * 16))) : (0)];
                float v1447_data = v1431_g ? (v1447_data_pre) : (0.0f);
                r0[v1444_i1] = v1447_data;
              }
              float v846_data = ir1[1];
              ir1[1] = (v846_data + v845_p);
              float v853_data = ir1[3];
              ir1[3] = (v853_data + (v842_p * v809_bc));
              float v860_data = ir1[5];
              ir1[5] = (v860_data + (v842_p * v816_bc));
              float v867_data = ir1[7];
              ir1[7] = (v867_data + (v842_p * v823_bc));
              float v874_data = ir1[9];
              ir1[9] = (v874_data + (v842_p * v830_bc));
              float v881_data = ir1[11];
              ir1[11] = (v881_data + (v842_p * v837_bc));
              sycl::vec<float, 2> v884_data = *(sycl::vec<float, 2>*)&glb_m1[(640 + v32_a)];
              float v885_p = v884_data[0];
              float v887_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v889_data = ir1[0];
              ir1[0] = (v889_data + (v885_p * v887_bc));
              float v894_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v896_data = ir1[2];
              ir1[2] = (v896_data + (v885_p * v894_bc));
              float v901_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v903_data = ir1[4];
              ir1[4] = (v903_data + (v885_p * v901_bc));
              float v908_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v910_data = ir1[6];
              ir1[6] = (v910_data + (v885_p * v908_bc));
              float v915_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v917_data = ir1[8];
              ir1[8] = (v917_data + (v885_p * v915_bc));
              float v922_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10));
              float v924_data = ir1[10];
              ir1[10] = (v924_data + (v885_p * v922_bc));
              float v927_p = v884_data[1];
              float v931_data = ir1[1];
              ir1[1] = (v931_data + (v927_p * v887_bc));
              float v938_data = ir1[3];
              ir1[3] = (v938_data + (v927_p * v894_bc));
              float v945_data = ir1[5];
              ir1[5] = (v945_data + (v927_p * v901_bc));
              float v952_data = ir1[7];
              ir1[7] = (v952_data + (v927_p * v908_bc));
              float v959_data = ir1[9];
              ir1[9] = (v959_data + (v927_p * v915_bc));
              float v966_data = ir1[11];
              ir1[11] = (v966_data + (v927_p * v922_bc));
              sycl::vec<float, 2> v969_data = *(sycl::vec<float, 2>*)&glb_m1[(704 + v32_a)];
              float v970_p = v969_data[0];
              float v972_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v974_data = ir1[0];
              ir1[0] = (v974_data + (v970_p * v972_bc));
              float v979_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v981_data = ir1[2];
              ir1[2] = (v981_data + (v970_p * v979_bc));
              float v986_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v988_data = ir1[4];
              ir1[4] = (v988_data + (v970_p * v986_bc));
              float v993_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v995_data = ir1[6];
              ir1[6] = (v995_data + (v970_p * v993_bc));
              float v1000_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1002_data = ir1[8];
              ir1[8] = (v1002_data + (v970_p * v1000_bc));
              float v1007_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11));
              float v1009_data = ir1[10];
              ir1[10] = (v1009_data + (v970_p * v1007_bc));
              float v1012_p = v969_data[1];
              float v1016_data = ir1[1];
              ir1[1] = (v1016_data + (v1012_p * v972_bc));
              float v1023_data = ir1[3];
              ir1[3] = (v1023_data + (v1012_p * v979_bc));
              float v1030_data = ir1[5];
              ir1[5] = (v1030_data + (v1012_p * v986_bc));
              float v1037_data = ir1[7];
              ir1[7] = (v1037_data + (v1012_p * v993_bc));
              float v1044_data = ir1[9];
              ir1[9] = (v1044_data + (v1012_p * v1000_bc));
              float v1051_data = ir1[11];
              ir1[11] = (v1051_data + (v1012_p * v1007_bc));
              sycl::vec<float, 2> v1054_data = *(sycl::vec<float, 2>*)&glb_m1[(768 + v32_a)];
              float v1055_p = v1054_data[0];
              float v1057_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1059_data = ir1[0];
              ir1[0] = (v1059_data + (v1055_p * v1057_bc));
              float v1064_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1066_data = ir1[2];
              ir1[2] = (v1066_data + (v1055_p * v1064_bc));
              float v1071_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1073_data = ir1[4];
              ir1[4] = (v1073_data + (v1055_p * v1071_bc));
              float v1078_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1080_data = ir1[6];
              ir1[6] = (v1080_data + (v1055_p * v1078_bc));
              float v1085_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1087_data = ir1[8];
              ir1[8] = (v1087_data + (v1055_p * v1085_bc));
              float v1092_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12));
              float v1094_data = ir1[10];
              ir1[10] = (v1094_data + (v1055_p * v1092_bc));
              float v1097_p = v1054_data[1];
              float v1101_data = ir1[1];
              ir1[1] = (v1101_data + (v1097_p * v1057_bc));
              float v1108_data = ir1[3];
              ir1[3] = (v1108_data + (v1097_p * v1064_bc));
              float v1115_data = ir1[5];
              ir1[5] = (v1115_data + (v1097_p * v1071_bc));
              float v1122_data = ir1[7];
              ir1[7] = (v1122_data + (v1097_p * v1078_bc));
              float v1129_data = ir1[9];
              ir1[9] = (v1129_data + (v1097_p * v1085_bc));
              float v1136_data = ir1[11];
              ir1[11] = (v1136_data + (v1097_p * v1092_bc));
              sycl::vec<float, 2> v1139_data = *(sycl::vec<float, 2>*)&glb_m1[(832 + v32_a)];
              float v1140_p = v1139_data[0];
              float v1142_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1144_data = ir1[0];
              ir1[0] = (v1144_data + (v1140_p * v1142_bc));
              float v1149_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1151_data = ir1[2];
              ir1[2] = (v1151_data + (v1140_p * v1149_bc));
              float v1156_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1158_data = ir1[4];
              ir1[4] = (v1158_data + (v1140_p * v1156_bc));
              float v1163_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1165_data = ir1[6];
              ir1[6] = (v1165_data + (v1140_p * v1163_bc));
              float v1170_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1172_data = ir1[8];
              ir1[8] = (v1172_data + (v1140_p * v1170_bc));
              float v1177_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13));
              float v1179_data = ir1[10];
              ir1[10] = (v1179_data + (v1140_p * v1177_bc));
              float v1182_p = v1139_data[1];
              float v1186_data = ir1[1];
              ir1[1] = (v1186_data + (v1182_p * v1142_bc));
              float v1193_data = ir1[3];
              ir1[3] = (v1193_data + (v1182_p * v1149_bc));
              float v1200_data = ir1[5];
              ir1[5] = (v1200_data + (v1182_p * v1156_bc));
              float v1207_data = ir1[7];
              ir1[7] = (v1207_data + (v1182_p * v1163_bc));
              float v1214_data = ir1[9];
              ir1[9] = (v1214_data + (v1182_p * v1170_bc));
              float v1221_data = ir1[11];
              ir1[11] = (v1221_data + (v1182_p * v1177_bc));
              sycl::vec<float, 2> v1224_data = *(sycl::vec<float, 2>*)&glb_m1[(896 + v32_a)];
              float v1225_p = v1224_data[0];
              float v1227_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1229_data = ir1[0];
              ir1[0] = (v1229_data + (v1225_p * v1227_bc));
              float v1234_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1236_data = ir1[2];
              ir1[2] = (v1236_data + (v1225_p * v1234_bc));
              float v1241_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1243_data = ir1[4];
              ir1[4] = (v1243_data + (v1225_p * v1241_bc));
              float v1248_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1250_data = ir1[6];
              ir1[6] = (v1250_data + (v1225_p * v1248_bc));
              float v1255_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1257_data = ir1[8];
              ir1[8] = (v1257_data + (v1225_p * v1255_bc));
              float v1262_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14));
              float v1264_data = ir1[10];
              ir1[10] = (v1264_data + (v1225_p * v1262_bc));
              float v1267_p = v1224_data[1];
              float v1271_data = ir1[1];
              ir1[1] = (v1271_data + (v1267_p * v1227_bc));
              float v1278_data = ir1[3];
              ir1[3] = (v1278_data + (v1267_p * v1234_bc));
              float v1285_data = ir1[5];
              ir1[5] = (v1285_data + (v1267_p * v1241_bc));
              float v1292_data = ir1[7];
              ir1[7] = (v1292_data + (v1267_p * v1248_bc));
              float v1299_data = ir1[9];
              ir1[9] = (v1299_data + (v1267_p * v1255_bc));
              float v1306_data = ir1[11];
              ir1[11] = (v1306_data + (v1267_p * v1262_bc));
              sycl::vec<float, 2> v1309_data = *(sycl::vec<float, 2>*)&glb_m1[(960 + v32_a)];
              float v1310_p = v1309_data[0];
              float v1312_bc = sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              float v1314_data = ir1[0];
              ir1[0] = (v1314_data + (v1310_p * v1312_bc));
              float v1319_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              float v1321_data = ir1[2];
              ir1[2] = (v1321_data + (v1310_p * v1319_bc));
              float v1326_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              float v1328_data = ir1[4];
              ir1[4] = (v1328_data + (v1310_p * v1326_bc));
              float v1333_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              float v1335_data = ir1[6];
              ir1[6] = (v1335_data + (v1310_p * v1333_bc));
              float v1340_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              float v1342_data = ir1[8];
              ir1[8] = (v1342_data + (v1310_p * v1340_bc));
              float v1347_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15));
              float v1349_data = ir1[10];
              ir1[10] = (v1349_data + (v1310_p * v1347_bc));
              float v1352_p = v1309_data[1];
              float v1356_data = ir1[1];
              ir1[1] = (v1356_data + (v1352_p * v1312_bc));
              float v1363_data = ir1[3];
              ir1[3] = (v1363_data + (v1352_p * v1319_bc));
              float v1370_data = ir1[5];
              ir1[5] = (v1370_data + (v1352_p * v1326_bc));
              float v1377_data = ir1[7];
              ir1[7] = (v1377_data + (v1352_p * v1333_bc));
              float v1384_data = ir1[9];
              ir1[9] = (v1384_data + (v1352_p * v1340_bc));
              float v1391_data = ir1[11];
              ir1[11] = (v1391_data + (v1352_p * v1347_bc));
              // r1 = ir1
              #pragma unroll
              for (int32_t v1393_n0 = 0; v1393_n0 < 1; ++v1393_n0) {
                #pragma unroll
                for (int32_t v1394_n1 = 0; v1394_n1 < 6; ++v1394_n1) {
                  int32_t v1396_a = v1393_n0 + (v1394_n1 * 2);
                  float v1397_data = ir1[v1396_a];
                  r1[v1396_a] = v1397_data;
                }
              }
              bool v1398_g = v1430_lead < 24;
              if (v1398_g) {
                #pragma unroll
                for (int32_t v1399_n1 = 0; v1399_n1 < 6; ++v1399_n1) {
                  int32_t v1401_a = 1 + (v1399_n1 * 2);
                  float v1402_data = ir1[v1401_a];
                  r1[v1401_a] = v1402_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v1403_i0 = 0; v1403_i0 < 1; ++v1403_i0) {
                int32_t v1409_lead = v1430_lead + (v1403_i0 * 32);
                #pragma unroll
                for (int32_t v1404_i1 = 0; v1404_i1 < 6; ++v1404_i1) {
                  float v1407_data = r1[(v1403_i0 + (v1404_i1 * 2))];
                  glb_m0[(v1409_lead + (v1404_i1 * 56))] = v1407_data;
                }
              }
              if (v1398_g) {
                int32_t v1418_lead = v1430_lead + 32_i32;
                #pragma unroll
                for (int32_t v1413_i1 = 0; v1413_i1 < 6; ++v1413_i1) {
                  float v1416_data = r1[(1 + (v1413_i1 * 2))];
                  glb_m0[(v1418_lead + (v1413_i1 * 56))] = v1416_data;
                }
              }
              item.barrier();
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v1436_i1 = 0; v1436_i1 < 6; ++v1436_i1) {
                float v1439_data_pre = wrap_glb_m2[v1431_g ? ((v1430_lead + (v1436_i1 * 16))) : (0)];
                float v1439_data = v1431_g ? (v1439_data_pre) : (0.0f);
                r0[v1436_i1] = v1439_data;
              }
            }
            v1426_flagWord = v1427_flagWordNext;
            v1427_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}


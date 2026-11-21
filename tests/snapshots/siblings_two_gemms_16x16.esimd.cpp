// === base name ===
kernel_ed59dc6c5bf2319d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ed59dc6c5bf2319d = {{1, 8, 1}, 32, 32, 1, 8, 8192, false, true, 2};
tensorforge::LaunchConfig launch_config_kernel_ed59dc6c5bf2319d(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_ed59dc6c5bf2319d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ed59dc6c5bf2319d(size_t numElements0, size_t numElements1, void* streamPtr) {
  (void)numElements0;
  (void)numElements1;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 8 - 1) / 8;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 2048 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ed59dc6c5bf2319d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ed59dc6c5bf2319d(numElements0, numElements1, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ed59dc6c5bf2319d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, numElements1, flags0, flags1);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_ed59dc6c5bf2319d_k;
inline void kernel_kernel_ed59dc6c5bf2319d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_ed59dc6c5bf2319d_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2048 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 8 per block = block 1x8x1, 8192 B shared, occupancy grid
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":2048},{"barrier":false,"mults_per_block":8,"shared_elements":2048}],"shared_bytes":8192,"shared_elements":2048,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"F","bbox":[[0,0],[16,16]],"name":"m5","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"siblings"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (256 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::intel_esimd::simd<float, 512> r0(0.0f);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v1328_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1328_flagSecondAhead < numElements0) ? v1328_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m1 = &m1[v3_batchId1 * 256 + 0 + m1_extraOffset];
          // r0 = load{g>r}(peel_glb_m1);
          #pragma unroll
          for (int32_t v1340_i1 = 0; v1340_i1 < 16; ++v1340_i1) {
            tensorforge::intel_esimd::simd<float, 32> v1342_data(0.0f);
            tensorforge::intel_esimd::simd<float, 16> v1342_data_part;
            v1342_data_part.copy_from(peel_glb_m1 + ((v1340_i1 * 16)));
            v1342_data.template select<16, 1>(0) = v1342_data_part;
            r0.template select<32, 1>((v1340_i1 * 32)) = v1342_data;
          }
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 256 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 128> v1350_ld;
          v1350_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 0));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v1350_ld);
          tensorforge::intel_esimd::simd<float, 128> v1351_ld;
          v1351_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 128));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 128), v1351_ld);
          uint32_t v1332_flagWord(flagWordFirst);
          uint32_t v1333_flagWordNext(flagWordSecond);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ wrap_glb_m1 = &m1[v11_batchId1 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 256 + 0 + m2_extraOffset];
            size_t v1356_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1356_flagAheadAhead < numElements0) ? v1356_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v1332_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 256 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 256 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 256 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v8_batchId0 * 256 + 0 + m5_extraOffset];
              tensorforge::intel_esimd::simd<float, 512> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 512> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v35_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 256> s0_w0 = tensorforge::slmLoad<float, 256>(s0 + 0);
              float v36_data = s0_w0[0];
              float v41_data = s0_w0[16];
              float v46_data = s0_w0[32];
              float v51_data = s0_w0[48];
              float v56_data = s0_w0[64];
              float v61_data = s0_w0[80];
              float v66_data = s0_w0[96];
              float v71_data = s0_w0[112];
              float v76_data = s0_w0[128];
              float v81_data = s0_w0[144];
              float v86_data = s0_w0[160];
              float v91_data = s0_w0[176];
              float v96_data = s0_w0[192];
              float v101_data = s0_w0[208];
              float v106_data = s0_w0[224];
              float v111_data = s0_w0[240];
              tensorforge::intel_esimd::simd<float, 32> v115_data(r0.template select<32, 1>(32));
              float v116_data = s0_w0[1];
              float v121_data = s0_w0[17];
              float v126_data = s0_w0[33];
              float v131_data = s0_w0[49];
              float v136_data = s0_w0[65];
              float v141_data = s0_w0[81];
              float v146_data = s0_w0[97];
              float v151_data = s0_w0[113];
              float v156_data = s0_w0[129];
              float v161_data = s0_w0[145];
              float v166_data = s0_w0[161];
              float v171_data = s0_w0[177];
              float v176_data = s0_w0[193];
              float v181_data = s0_w0[209];
              float v186_data = s0_w0[225];
              float v191_data = s0_w0[241];
              tensorforge::intel_esimd::simd<float, 32> v195_data(r0.template select<32, 1>(64));
              float v196_data = s0_w0[2];
              float v201_data = s0_w0[18];
              float v206_data = s0_w0[34];
              float v211_data = s0_w0[50];
              float v216_data = s0_w0[66];
              float v221_data = s0_w0[82];
              float v226_data = s0_w0[98];
              float v231_data = s0_w0[114];
              float v236_data = s0_w0[130];
              float v241_data = s0_w0[146];
              float v246_data = s0_w0[162];
              float v251_data = s0_w0[178];
              float v256_data = s0_w0[194];
              float v261_data = s0_w0[210];
              float v266_data = s0_w0[226];
              float v271_data = s0_w0[242];
              tensorforge::intel_esimd::simd<float, 32> v275_data(r0.template select<32, 1>(96));
              float v276_data = s0_w0[3];
              float v281_data = s0_w0[19];
              float v286_data = s0_w0[35];
              float v291_data = s0_w0[51];
              float v296_data = s0_w0[67];
              float v301_data = s0_w0[83];
              float v306_data = s0_w0[99];
              float v311_data = s0_w0[115];
              float v316_data = s0_w0[131];
              float v321_data = s0_w0[147];
              float v326_data = s0_w0[163];
              float v331_data = s0_w0[179];
              float v336_data = s0_w0[195];
              float v341_data = s0_w0[211];
              float v346_data = s0_w0[227];
              float v351_data = s0_w0[243];
              tensorforge::intel_esimd::simd<float, 32> v355_data(r0.template select<32, 1>(128));
              float v356_data = s0_w0[4];
              float v361_data = s0_w0[20];
              float v366_data = s0_w0[36];
              float v371_data = s0_w0[52];
              float v376_data = s0_w0[68];
              float v381_data = s0_w0[84];
              float v386_data = s0_w0[100];
              float v391_data = s0_w0[116];
              float v396_data = s0_w0[132];
              float v401_data = s0_w0[148];
              float v406_data = s0_w0[164];
              float v411_data = s0_w0[180];
              float v416_data = s0_w0[196];
              float v421_data = s0_w0[212];
              float v426_data = s0_w0[228];
              float v431_data = s0_w0[244];
              tensorforge::intel_esimd::simd<float, 32> v435_data(r0.template select<32, 1>(160));
              float v436_data = s0_w0[5];
              float v441_data = s0_w0[21];
              float v446_data = s0_w0[37];
              float v451_data = s0_w0[53];
              float v456_data = s0_w0[69];
              float v461_data = s0_w0[85];
              float v466_data = s0_w0[101];
              float v471_data = s0_w0[117];
              float v476_data = s0_w0[133];
              float v481_data = s0_w0[149];
              float v486_data = s0_w0[165];
              float v491_data = s0_w0[181];
              float v496_data = s0_w0[197];
              float v501_data = s0_w0[213];
              float v506_data = s0_w0[229];
              float v511_data = s0_w0[245];
              tensorforge::intel_esimd::simd<float, 32> v515_data(r0.template select<32, 1>(192));
              float v516_data = s0_w0[6];
              float v521_data = s0_w0[22];
              float v526_data = s0_w0[38];
              float v531_data = s0_w0[54];
              float v536_data = s0_w0[70];
              float v541_data = s0_w0[86];
              float v546_data = s0_w0[102];
              float v551_data = s0_w0[118];
              float v556_data = s0_w0[134];
              float v561_data = s0_w0[150];
              float v566_data = s0_w0[166];
              float v571_data = s0_w0[182];
              float v576_data = s0_w0[198];
              float v581_data = s0_w0[214];
              float v586_data = s0_w0[230];
              float v591_data = s0_w0[246];
              tensorforge::intel_esimd::simd<float, 32> v595_data(r0.template select<32, 1>(224));
              float v596_data = s0_w0[7];
              float v601_data = s0_w0[23];
              float v606_data = s0_w0[39];
              float v611_data = s0_w0[55];
              float v616_data = s0_w0[71];
              float v621_data = s0_w0[87];
              float v626_data = s0_w0[103];
              float v631_data = s0_w0[119];
              float v636_data = s0_w0[135];
              float v641_data = s0_w0[151];
              float v646_data = s0_w0[167];
              float v651_data = s0_w0[183];
              float v656_data = s0_w0[199];
              float v661_data = s0_w0[215];
              float v666_data = s0_w0[231];
              float v671_data = s0_w0[247];
              tensorforge::intel_esimd::simd<float, 32> v675_data(r0.template select<32, 1>(256));
              float v676_data = s0_w0[8];
              float v681_data = s0_w0[24];
              float v686_data = s0_w0[40];
              float v691_data = s0_w0[56];
              float v696_data = s0_w0[72];
              float v701_data = s0_w0[88];
              float v706_data = s0_w0[104];
              float v711_data = s0_w0[120];
              float v716_data = s0_w0[136];
              float v721_data = s0_w0[152];
              float v726_data = s0_w0[168];
              float v731_data = s0_w0[184];
              float v736_data = s0_w0[200];
              float v741_data = s0_w0[216];
              float v746_data = s0_w0[232];
              float v751_data = s0_w0[248];
              tensorforge::intel_esimd::simd<float, 32> v755_data(r0.template select<32, 1>(288));
              float v756_data = s0_w0[9];
              float v761_data = s0_w0[25];
              float v766_data = s0_w0[41];
              float v771_data = s0_w0[57];
              float v776_data = s0_w0[73];
              float v781_data = s0_w0[89];
              float v786_data = s0_w0[105];
              float v791_data = s0_w0[121];
              float v796_data = s0_w0[137];
              float v801_data = s0_w0[153];
              float v806_data = s0_w0[169];
              float v811_data = s0_w0[185];
              float v816_data = s0_w0[201];
              float v821_data = s0_w0[217];
              float v826_data = s0_w0[233];
              float v831_data = s0_w0[249];
              tensorforge::intel_esimd::simd<float, 32> v835_data(r0.template select<32, 1>(320));
              float v836_data = s0_w0[10];
              float v841_data = s0_w0[26];
              float v846_data = s0_w0[42];
              float v851_data = s0_w0[58];
              float v856_data = s0_w0[74];
              float v861_data = s0_w0[90];
              float v866_data = s0_w0[106];
              float v871_data = s0_w0[122];
              float v876_data = s0_w0[138];
              float v881_data = s0_w0[154];
              float v886_data = s0_w0[170];
              float v891_data = s0_w0[186];
              float v896_data = s0_w0[202];
              float v901_data = s0_w0[218];
              float v906_data = s0_w0[234];
              float v911_data = s0_w0[250];
              tensorforge::intel_esimd::simd<float, 32> v915_data(r0.template select<32, 1>(352));
              float v916_data = s0_w0[11];
              float v921_data = s0_w0[27];
              float v926_data = s0_w0[43];
              float v931_data = s0_w0[59];
              float v936_data = s0_w0[75];
              float v941_data = s0_w0[91];
              float v946_data = s0_w0[107];
              float v951_data = s0_w0[123];
              float v956_data = s0_w0[139];
              float v961_data = s0_w0[155];
              float v966_data = s0_w0[171];
              float v971_data = s0_w0[187];
              float v976_data = s0_w0[203];
              float v981_data = s0_w0[219];
              float v986_data = s0_w0[235];
              float v991_data = s0_w0[251];
              tensorforge::intel_esimd::simd<float, 32> v995_data(r0.template select<32, 1>(384));
              float v996_data = s0_w0[12];
              float v1001_data = s0_w0[28];
              float v1006_data = s0_w0[44];
              float v1011_data = s0_w0[60];
              float v1016_data = s0_w0[76];
              float v1021_data = s0_w0[92];
              float v1026_data = s0_w0[108];
              float v1031_data = s0_w0[124];
              float v1036_data = s0_w0[140];
              float v1041_data = s0_w0[156];
              float v1046_data = s0_w0[172];
              float v1051_data = s0_w0[188];
              float v1056_data = s0_w0[204];
              float v1061_data = s0_w0[220];
              float v1066_data = s0_w0[236];
              float v1071_data = s0_w0[252];
              tensorforge::intel_esimd::simd<float, 32> v1075_data(r0.template select<32, 1>(416));
              float v1076_data = s0_w0[13];
              float v1081_data = s0_w0[29];
              float v1086_data = s0_w0[45];
              float v1091_data = s0_w0[61];
              float v1096_data = s0_w0[77];
              float v1101_data = s0_w0[93];
              float v1106_data = s0_w0[109];
              float v1111_data = s0_w0[125];
              float v1116_data = s0_w0[141];
              float v1121_data = s0_w0[157];
              float v1126_data = s0_w0[173];
              float v1131_data = s0_w0[189];
              float v1136_data = s0_w0[205];
              float v1141_data = s0_w0[221];
              float v1146_data = s0_w0[237];
              float v1151_data = s0_w0[253];
              tensorforge::intel_esimd::simd<float, 32> v1155_data(r0.template select<32, 1>(448));
              float v1156_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v1159_p = ((((((((((((((v35_data * v36_data) + (v115_data * v116_data)) + (v195_data * v196_data)) + (v275_data * v276_data)) + (v355_data * v356_data)) + (v435_data * v436_data)) + (v515_data * v516_data)) + (v595_data * v596_data)) + (v675_data * v676_data)) + (v755_data * v756_data)) + (v835_data * v836_data)) + (v915_data * v916_data)) + (v995_data * v996_data)) + (v1075_data * v1076_data)) + (v1155_data * v1156_data);
              float v1161_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v1164_p = ((((((((((((((v35_data * v41_data) + (v115_data * v121_data)) + (v195_data * v201_data)) + (v275_data * v281_data)) + (v355_data * v361_data)) + (v435_data * v441_data)) + (v515_data * v521_data)) + (v595_data * v601_data)) + (v675_data * v681_data)) + (v755_data * v761_data)) + (v835_data * v841_data)) + (v915_data * v921_data)) + (v995_data * v1001_data)) + (v1075_data * v1081_data)) + (v1155_data * v1161_data);
              float v1166_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v1169_p = ((((((((((((((v35_data * v46_data) + (v115_data * v126_data)) + (v195_data * v206_data)) + (v275_data * v286_data)) + (v355_data * v366_data)) + (v435_data * v446_data)) + (v515_data * v526_data)) + (v595_data * v606_data)) + (v675_data * v686_data)) + (v755_data * v766_data)) + (v835_data * v846_data)) + (v915_data * v926_data)) + (v995_data * v1006_data)) + (v1075_data * v1086_data)) + (v1155_data * v1166_data);
              float v1171_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 32> v1174_p = ((((((((((((((v35_data * v51_data) + (v115_data * v131_data)) + (v195_data * v211_data)) + (v275_data * v291_data)) + (v355_data * v371_data)) + (v435_data * v451_data)) + (v515_data * v531_data)) + (v595_data * v611_data)) + (v675_data * v691_data)) + (v755_data * v771_data)) + (v835_data * v851_data)) + (v915_data * v931_data)) + (v995_data * v1011_data)) + (v1075_data * v1091_data)) + (v1155_data * v1171_data);
              float v1176_data = s0_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v1179_p = ((((((((((((((v35_data * v56_data) + (v115_data * v136_data)) + (v195_data * v216_data)) + (v275_data * v296_data)) + (v355_data * v376_data)) + (v435_data * v456_data)) + (v515_data * v536_data)) + (v595_data * v616_data)) + (v675_data * v696_data)) + (v755_data * v776_data)) + (v835_data * v856_data)) + (v915_data * v936_data)) + (v995_data * v1016_data)) + (v1075_data * v1096_data)) + (v1155_data * v1176_data);
              float v1181_data = s0_w0[94];
              tensorforge::intel_esimd::simd<float, 32> v1184_p = ((((((((((((((v35_data * v61_data) + (v115_data * v141_data)) + (v195_data * v221_data)) + (v275_data * v301_data)) + (v355_data * v381_data)) + (v435_data * v461_data)) + (v515_data * v541_data)) + (v595_data * v621_data)) + (v675_data * v701_data)) + (v755_data * v781_data)) + (v835_data * v861_data)) + (v915_data * v941_data)) + (v995_data * v1021_data)) + (v1075_data * v1101_data)) + (v1155_data * v1181_data);
              float v1186_data = s0_w0[110];
              tensorforge::intel_esimd::simd<float, 32> v1189_p = ((((((((((((((v35_data * v66_data) + (v115_data * v146_data)) + (v195_data * v226_data)) + (v275_data * v306_data)) + (v355_data * v386_data)) + (v435_data * v466_data)) + (v515_data * v546_data)) + (v595_data * v626_data)) + (v675_data * v706_data)) + (v755_data * v786_data)) + (v835_data * v866_data)) + (v915_data * v946_data)) + (v995_data * v1026_data)) + (v1075_data * v1106_data)) + (v1155_data * v1186_data);
              float v1191_data = s0_w0[126];
              tensorforge::intel_esimd::simd<float, 32> v1194_p = ((((((((((((((v35_data * v71_data) + (v115_data * v151_data)) + (v195_data * v231_data)) + (v275_data * v311_data)) + (v355_data * v391_data)) + (v435_data * v471_data)) + (v515_data * v551_data)) + (v595_data * v631_data)) + (v675_data * v711_data)) + (v755_data * v791_data)) + (v835_data * v871_data)) + (v915_data * v951_data)) + (v995_data * v1031_data)) + (v1075_data * v1111_data)) + (v1155_data * v1191_data);
              float v1196_data = s0_w0[142];
              tensorforge::intel_esimd::simd<float, 32> v1199_p = ((((((((((((((v35_data * v76_data) + (v115_data * v156_data)) + (v195_data * v236_data)) + (v275_data * v316_data)) + (v355_data * v396_data)) + (v435_data * v476_data)) + (v515_data * v556_data)) + (v595_data * v636_data)) + (v675_data * v716_data)) + (v755_data * v796_data)) + (v835_data * v876_data)) + (v915_data * v956_data)) + (v995_data * v1036_data)) + (v1075_data * v1116_data)) + (v1155_data * v1196_data);
              float v1201_data = s0_w0[158];
              tensorforge::intel_esimd::simd<float, 32> v1204_p = ((((((((((((((v35_data * v81_data) + (v115_data * v161_data)) + (v195_data * v241_data)) + (v275_data * v321_data)) + (v355_data * v401_data)) + (v435_data * v481_data)) + (v515_data * v561_data)) + (v595_data * v641_data)) + (v675_data * v721_data)) + (v755_data * v801_data)) + (v835_data * v881_data)) + (v915_data * v961_data)) + (v995_data * v1041_data)) + (v1075_data * v1121_data)) + (v1155_data * v1201_data);
              float v1206_data = s0_w0[174];
              tensorforge::intel_esimd::simd<float, 32> v1209_p = ((((((((((((((v35_data * v86_data) + (v115_data * v166_data)) + (v195_data * v246_data)) + (v275_data * v326_data)) + (v355_data * v406_data)) + (v435_data * v486_data)) + (v515_data * v566_data)) + (v595_data * v646_data)) + (v675_data * v726_data)) + (v755_data * v806_data)) + (v835_data * v886_data)) + (v915_data * v966_data)) + (v995_data * v1046_data)) + (v1075_data * v1126_data)) + (v1155_data * v1206_data);
              float v1211_data = s0_w0[190];
              tensorforge::intel_esimd::simd<float, 32> v1214_p = ((((((((((((((v35_data * v91_data) + (v115_data * v171_data)) + (v195_data * v251_data)) + (v275_data * v331_data)) + (v355_data * v411_data)) + (v435_data * v491_data)) + (v515_data * v571_data)) + (v595_data * v651_data)) + (v675_data * v731_data)) + (v755_data * v811_data)) + (v835_data * v891_data)) + (v915_data * v971_data)) + (v995_data * v1051_data)) + (v1075_data * v1131_data)) + (v1155_data * v1211_data);
              float v1216_data = s0_w0[206];
              tensorforge::intel_esimd::simd<float, 32> v1219_p = ((((((((((((((v35_data * v96_data) + (v115_data * v176_data)) + (v195_data * v256_data)) + (v275_data * v336_data)) + (v355_data * v416_data)) + (v435_data * v496_data)) + (v515_data * v576_data)) + (v595_data * v656_data)) + (v675_data * v736_data)) + (v755_data * v816_data)) + (v835_data * v896_data)) + (v915_data * v976_data)) + (v995_data * v1056_data)) + (v1075_data * v1136_data)) + (v1155_data * v1216_data);
              float v1221_data = s0_w0[222];
              tensorforge::intel_esimd::simd<float, 32> v1224_p = ((((((((((((((v35_data * v101_data) + (v115_data * v181_data)) + (v195_data * v261_data)) + (v275_data * v341_data)) + (v355_data * v421_data)) + (v435_data * v501_data)) + (v515_data * v581_data)) + (v595_data * v661_data)) + (v675_data * v741_data)) + (v755_data * v821_data)) + (v835_data * v901_data)) + (v915_data * v981_data)) + (v995_data * v1061_data)) + (v1075_data * v1141_data)) + (v1155_data * v1221_data);
              float v1226_data = s0_w0[238];
              tensorforge::intel_esimd::simd<float, 32> v1229_p = ((((((((((((((v35_data * v106_data) + (v115_data * v186_data)) + (v195_data * v266_data)) + (v275_data * v346_data)) + (v355_data * v426_data)) + (v435_data * v506_data)) + (v515_data * v586_data)) + (v595_data * v666_data)) + (v675_data * v746_data)) + (v755_data * v826_data)) + (v835_data * v906_data)) + (v915_data * v986_data)) + (v995_data * v1066_data)) + (v1075_data * v1146_data)) + (v1155_data * v1226_data);
              float v1231_data = s0_w0[254];
              tensorforge::intel_esimd::simd<float, 32> v1234_p = ((((((((((((((v35_data * v111_data) + (v115_data * v191_data)) + (v195_data * v271_data)) + (v275_data * v351_data)) + (v355_data * v431_data)) + (v435_data * v511_data)) + (v515_data * v591_data)) + (v595_data * v671_data)) + (v675_data * v751_data)) + (v755_data * v831_data)) + (v835_data * v911_data)) + (v915_data * v991_data)) + (v995_data * v1071_data)) + (v1075_data * v1151_data)) + (v1155_data * v1231_data);
              tensorforge::intel_esimd::simd<float, 32> v1235_data(r0.template select<32, 1>(480));
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v1344_i1 = 0; v1344_i1 < 16; ++v1344_i1) {
                tensorforge::intel_esimd::simd<float, 32> v1346_data(0.0f);
                tensorforge::intel_esimd::simd<float, 16> v1346_data_part;
                v1346_data_part.copy_from(wrap_glb_m1 + ((v1344_i1 * 16)));
                v1346_data.template select<16, 1>(0) = v1346_data_part;
                r0.template select<32, 1>((v1344_i1 * 32)) = v1346_data;
              }
              float v1236_data = s0[15];
              ir1.template select<32, 1>(0) = (v1159_p + (v1235_data * v1236_data));
              float v1241_data = s0[31];
              ir1.template select<32, 1>(32) = (v1164_p + (v1235_data * v1241_data));
              float v1246_data = s0[47];
              ir1.template select<32, 1>(64) = (v1169_p + (v1235_data * v1246_data));
              float v1251_data = s0[63];
              ir1.template select<32, 1>(96) = (v1174_p + (v1235_data * v1251_data));
              float v1256_data = s0[79];
              ir1.template select<32, 1>(128) = (v1179_p + (v1235_data * v1256_data));
              float v1261_data = s0[95];
              ir1.template select<32, 1>(160) = (v1184_p + (v1235_data * v1261_data));
              float v1266_data = s0[111];
              ir1.template select<32, 1>(192) = (v1189_p + (v1235_data * v1266_data));
              float v1271_data = s0[127];
              ir1.template select<32, 1>(224) = (v1194_p + (v1235_data * v1271_data));
              float v1276_data = s0[143];
              ir1.template select<32, 1>(256) = (v1199_p + (v1235_data * v1276_data));
              float v1281_data = s0[159];
              ir1.template select<32, 1>(288) = (v1204_p + (v1235_data * v1281_data));
              float v1286_data = s0[175];
              ir1.template select<32, 1>(320) = (v1209_p + (v1235_data * v1286_data));
              float v1291_data = s0[191];
              ir1.template select<32, 1>(352) = (v1214_p + (v1235_data * v1291_data));
              float v1296_data = s0[207];
              ir1.template select<32, 1>(384) = (v1219_p + (v1235_data * v1296_data));
              float v1301_data = s0[223];
              ir1.template select<32, 1>(416) = (v1224_p + (v1235_data * v1301_data));
              float v1306_data = s0[239];
              ir1.template select<32, 1>(448) = (v1229_p + (v1235_data * v1306_data));
              float v1311_data = s0[255];
              ir1.template select<32, 1>(480) = (v1234_p + (v1235_data * v1311_data));
              // r1 = ir1
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v1319_i1 = 0; v1319_i1 < 16; ++v1319_i1) {
                tensorforge::intel_esimd::simd<float, 16> v1322_data(ir1.template select<16, 1>((v1319_i1 * 32)));
                v1322_data.copy_to(glb_m0 + ((v1319_i1 * 16)));
              }
            }
            else {
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v1336_i1 = 0; v1336_i1 < 16; ++v1336_i1) {
                tensorforge::intel_esimd::simd<float, 32> v1338_data(0.0f);
                tensorforge::intel_esimd::simd<float, 16> v1338_data_part;
                v1338_data_part.copy_from(wrap_glb_m1 + ((v1336_i1 * 16)));
                v1338_data.template select<16, 1>(0) = v1338_data_part;
                r0.template select<32, 1>((v1336_i1 * 32)) = v1338_data;
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 128> v1352_ld;
            v1352_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v1352_ld);
            tensorforge::intel_esimd::simd<float, 128> v1353_ld;
            v1353_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 128));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 128), v1353_ld);
            v1332_flagWord = v1333_flagWordNext;
            v1333_flagWordNext = flagWordAhead;
          }
        }
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (256 * item.get_local_id(1) + 0);
          size_t v1367_batchId1 = (((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) + numElements0) % (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements1) ? ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) + numElements0) % (item.get_group_range(2) * item.get_group().get_local_range(1)) : 0u;
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::intel_esimd::simd<float, 512> r0(0.0f);
          const uint32_t flagWordFirst = flags1 == nullptr ? 1 : flags1[v1367_batchId1];
          size_t v2692_flagSecondAhead = v1367_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags1 == nullptr ? 1 : flags1[((v2692_flagSecondAhead < numElements1) ? v2692_flagSecondAhead : v1367_batchId1)];
          const float *const __restrict__ peel_glb_m4 = &m4[v1367_batchId1 * 256 + 0 + m4_extraOffset];
          // r0 = load{g>r}(peel_glb_m4);
          #pragma unroll
          for (int32_t v2704_i1 = 0; v2704_i1 < 16; ++v2704_i1) {
            tensorforge::intel_esimd::simd<float, 32> v2706_data(0.0f);
            tensorforge::intel_esimd::simd<float, 16> v2706_data_part;
            v2706_data_part.copy_from(peel_glb_m4 + ((v2704_i1 * 16)));
            v2706_data.template select<16, 1>(0) = v2706_data_part;
            r0.template select<32, 1>((v2704_i1 * 32)) = v2706_data;
          }
          const float *const __restrict__ peel_glb_m5 = &m5[v1367_batchId1 * 256 + 0 + m5_extraOffset];
          // s0 = load{g>s}(peel_glb_m5[0, 1])
          tensorforge::intel_esimd::simd<float, 128> v2714_ld;
          v2714_ld.copy_from(peel_glb_m5 + (0 + 0 + 4 * 0 + 0));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v2714_ld);
          tensorforge::intel_esimd::simd<float, 128> v2715_ld;
          v2715_ld.copy_from(peel_glb_m5 + (0 + 0 + 4 * 0 + 128));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 128), v2715_ld);
          uint32_t v2696_flagWord(flagWordFirst);
          uint32_t v2697_flagWordNext(flagWordSecond);
          for (size_t v1372_batchId0 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) + numElements0) % (item.get_group_range(2) * item.get_group().get_local_range(1)); v1372_batchId0 < numElements1; v1372_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v1373_ahead1 = v1372_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v1375_batchId1 = (v1373_ahead1 < numElements1) ? v1373_ahead1 : v1372_batchId0;
            const float *const __restrict__ wrap_glb_m4 = &m4[v1375_batchId1 * 256 + 0 + m4_extraOffset];
            const float *const __restrict__ wrap_glb_m5 = &m5[v1375_batchId1 * 256 + 0 + m5_extraOffset];
            size_t v2720_flagAheadAhead = v1375_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags1 == nullptr ? 1 : flags1[((v2720_flagAheadAhead < numElements1) ? v2720_flagAheadAhead : v1375_batchId1)];
            const bool allowed = static_cast<bool>(v2696_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v1372_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1372_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v1372_batchId0 * 256 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v1372_batchId0 * 256 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 512> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 512> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v1399_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 256> s0_w0 = tensorforge::slmLoad<float, 256>(s0 + 0);
              float v1400_data = s0_w0[0];
              float v1405_data = s0_w0[16];
              float v1410_data = s0_w0[32];
              float v1415_data = s0_w0[48];
              float v1420_data = s0_w0[64];
              float v1425_data = s0_w0[80];
              float v1430_data = s0_w0[96];
              float v1435_data = s0_w0[112];
              float v1440_data = s0_w0[128];
              float v1445_data = s0_w0[144];
              float v1450_data = s0_w0[160];
              float v1455_data = s0_w0[176];
              float v1460_data = s0_w0[192];
              float v1465_data = s0_w0[208];
              float v1470_data = s0_w0[224];
              float v1475_data = s0_w0[240];
              tensorforge::intel_esimd::simd<float, 32> v1479_data(r0.template select<32, 1>(32));
              float v1480_data = s0_w0[1];
              float v1485_data = s0_w0[17];
              float v1490_data = s0_w0[33];
              float v1495_data = s0_w0[49];
              float v1500_data = s0_w0[65];
              float v1505_data = s0_w0[81];
              float v1510_data = s0_w0[97];
              float v1515_data = s0_w0[113];
              float v1520_data = s0_w0[129];
              float v1525_data = s0_w0[145];
              float v1530_data = s0_w0[161];
              float v1535_data = s0_w0[177];
              float v1540_data = s0_w0[193];
              float v1545_data = s0_w0[209];
              float v1550_data = s0_w0[225];
              float v1555_data = s0_w0[241];
              tensorforge::intel_esimd::simd<float, 32> v1559_data(r0.template select<32, 1>(64));
              float v1560_data = s0_w0[2];
              float v1565_data = s0_w0[18];
              float v1570_data = s0_w0[34];
              float v1575_data = s0_w0[50];
              float v1580_data = s0_w0[66];
              float v1585_data = s0_w0[82];
              float v1590_data = s0_w0[98];
              float v1595_data = s0_w0[114];
              float v1600_data = s0_w0[130];
              float v1605_data = s0_w0[146];
              float v1610_data = s0_w0[162];
              float v1615_data = s0_w0[178];
              float v1620_data = s0_w0[194];
              float v1625_data = s0_w0[210];
              float v1630_data = s0_w0[226];
              float v1635_data = s0_w0[242];
              tensorforge::intel_esimd::simd<float, 32> v1639_data(r0.template select<32, 1>(96));
              float v1640_data = s0_w0[3];
              float v1645_data = s0_w0[19];
              float v1650_data = s0_w0[35];
              float v1655_data = s0_w0[51];
              float v1660_data = s0_w0[67];
              float v1665_data = s0_w0[83];
              float v1670_data = s0_w0[99];
              float v1675_data = s0_w0[115];
              float v1680_data = s0_w0[131];
              float v1685_data = s0_w0[147];
              float v1690_data = s0_w0[163];
              float v1695_data = s0_w0[179];
              float v1700_data = s0_w0[195];
              float v1705_data = s0_w0[211];
              float v1710_data = s0_w0[227];
              float v1715_data = s0_w0[243];
              tensorforge::intel_esimd::simd<float, 32> v1719_data(r0.template select<32, 1>(128));
              float v1720_data = s0_w0[4];
              float v1725_data = s0_w0[20];
              float v1730_data = s0_w0[36];
              float v1735_data = s0_w0[52];
              float v1740_data = s0_w0[68];
              float v1745_data = s0_w0[84];
              float v1750_data = s0_w0[100];
              float v1755_data = s0_w0[116];
              float v1760_data = s0_w0[132];
              float v1765_data = s0_w0[148];
              float v1770_data = s0_w0[164];
              float v1775_data = s0_w0[180];
              float v1780_data = s0_w0[196];
              float v1785_data = s0_w0[212];
              float v1790_data = s0_w0[228];
              float v1795_data = s0_w0[244];
              tensorforge::intel_esimd::simd<float, 32> v1799_data(r0.template select<32, 1>(160));
              float v1800_data = s0_w0[5];
              float v1805_data = s0_w0[21];
              float v1810_data = s0_w0[37];
              float v1815_data = s0_w0[53];
              float v1820_data = s0_w0[69];
              float v1825_data = s0_w0[85];
              float v1830_data = s0_w0[101];
              float v1835_data = s0_w0[117];
              float v1840_data = s0_w0[133];
              float v1845_data = s0_w0[149];
              float v1850_data = s0_w0[165];
              float v1855_data = s0_w0[181];
              float v1860_data = s0_w0[197];
              float v1865_data = s0_w0[213];
              float v1870_data = s0_w0[229];
              float v1875_data = s0_w0[245];
              tensorforge::intel_esimd::simd<float, 32> v1879_data(r0.template select<32, 1>(192));
              float v1880_data = s0_w0[6];
              float v1885_data = s0_w0[22];
              float v1890_data = s0_w0[38];
              float v1895_data = s0_w0[54];
              float v1900_data = s0_w0[70];
              float v1905_data = s0_w0[86];
              float v1910_data = s0_w0[102];
              float v1915_data = s0_w0[118];
              float v1920_data = s0_w0[134];
              float v1925_data = s0_w0[150];
              float v1930_data = s0_w0[166];
              float v1935_data = s0_w0[182];
              float v1940_data = s0_w0[198];
              float v1945_data = s0_w0[214];
              float v1950_data = s0_w0[230];
              float v1955_data = s0_w0[246];
              tensorforge::intel_esimd::simd<float, 32> v1959_data(r0.template select<32, 1>(224));
              float v1960_data = s0_w0[7];
              float v1965_data = s0_w0[23];
              float v1970_data = s0_w0[39];
              float v1975_data = s0_w0[55];
              float v1980_data = s0_w0[71];
              float v1985_data = s0_w0[87];
              float v1990_data = s0_w0[103];
              float v1995_data = s0_w0[119];
              float v2000_data = s0_w0[135];
              float v2005_data = s0_w0[151];
              float v2010_data = s0_w0[167];
              float v2015_data = s0_w0[183];
              float v2020_data = s0_w0[199];
              float v2025_data = s0_w0[215];
              float v2030_data = s0_w0[231];
              float v2035_data = s0_w0[247];
              tensorforge::intel_esimd::simd<float, 32> v2039_data(r0.template select<32, 1>(256));
              float v2040_data = s0_w0[8];
              float v2045_data = s0_w0[24];
              float v2050_data = s0_w0[40];
              float v2055_data = s0_w0[56];
              float v2060_data = s0_w0[72];
              float v2065_data = s0_w0[88];
              float v2070_data = s0_w0[104];
              float v2075_data = s0_w0[120];
              float v2080_data = s0_w0[136];
              float v2085_data = s0_w0[152];
              float v2090_data = s0_w0[168];
              float v2095_data = s0_w0[184];
              float v2100_data = s0_w0[200];
              float v2105_data = s0_w0[216];
              float v2110_data = s0_w0[232];
              float v2115_data = s0_w0[248];
              tensorforge::intel_esimd::simd<float, 32> v2119_data(r0.template select<32, 1>(288));
              float v2120_data = s0_w0[9];
              float v2125_data = s0_w0[25];
              float v2130_data = s0_w0[41];
              float v2135_data = s0_w0[57];
              float v2140_data = s0_w0[73];
              float v2145_data = s0_w0[89];
              float v2150_data = s0_w0[105];
              float v2155_data = s0_w0[121];
              float v2160_data = s0_w0[137];
              float v2165_data = s0_w0[153];
              float v2170_data = s0_w0[169];
              float v2175_data = s0_w0[185];
              float v2180_data = s0_w0[201];
              float v2185_data = s0_w0[217];
              float v2190_data = s0_w0[233];
              float v2195_data = s0_w0[249];
              tensorforge::intel_esimd::simd<float, 32> v2199_data(r0.template select<32, 1>(320));
              float v2200_data = s0_w0[10];
              float v2205_data = s0_w0[26];
              float v2210_data = s0_w0[42];
              float v2215_data = s0_w0[58];
              float v2220_data = s0_w0[74];
              float v2225_data = s0_w0[90];
              float v2230_data = s0_w0[106];
              float v2235_data = s0_w0[122];
              float v2240_data = s0_w0[138];
              float v2245_data = s0_w0[154];
              float v2250_data = s0_w0[170];
              float v2255_data = s0_w0[186];
              float v2260_data = s0_w0[202];
              float v2265_data = s0_w0[218];
              float v2270_data = s0_w0[234];
              float v2275_data = s0_w0[250];
              tensorforge::intel_esimd::simd<float, 32> v2279_data(r0.template select<32, 1>(352));
              float v2280_data = s0_w0[11];
              float v2285_data = s0_w0[27];
              float v2290_data = s0_w0[43];
              float v2295_data = s0_w0[59];
              float v2300_data = s0_w0[75];
              float v2305_data = s0_w0[91];
              float v2310_data = s0_w0[107];
              float v2315_data = s0_w0[123];
              float v2320_data = s0_w0[139];
              float v2325_data = s0_w0[155];
              float v2330_data = s0_w0[171];
              float v2335_data = s0_w0[187];
              float v2340_data = s0_w0[203];
              float v2345_data = s0_w0[219];
              float v2350_data = s0_w0[235];
              float v2355_data = s0_w0[251];
              tensorforge::intel_esimd::simd<float, 32> v2359_data(r0.template select<32, 1>(384));
              float v2360_data = s0_w0[12];
              float v2365_data = s0_w0[28];
              float v2370_data = s0_w0[44];
              float v2375_data = s0_w0[60];
              float v2380_data = s0_w0[76];
              float v2385_data = s0_w0[92];
              float v2390_data = s0_w0[108];
              float v2395_data = s0_w0[124];
              float v2400_data = s0_w0[140];
              float v2405_data = s0_w0[156];
              float v2410_data = s0_w0[172];
              float v2415_data = s0_w0[188];
              float v2420_data = s0_w0[204];
              float v2425_data = s0_w0[220];
              float v2430_data = s0_w0[236];
              float v2435_data = s0_w0[252];
              tensorforge::intel_esimd::simd<float, 32> v2439_data(r0.template select<32, 1>(416));
              float v2440_data = s0_w0[13];
              float v2445_data = s0_w0[29];
              float v2450_data = s0_w0[45];
              float v2455_data = s0_w0[61];
              float v2460_data = s0_w0[77];
              float v2465_data = s0_w0[93];
              float v2470_data = s0_w0[109];
              float v2475_data = s0_w0[125];
              float v2480_data = s0_w0[141];
              float v2485_data = s0_w0[157];
              float v2490_data = s0_w0[173];
              float v2495_data = s0_w0[189];
              float v2500_data = s0_w0[205];
              float v2505_data = s0_w0[221];
              float v2510_data = s0_w0[237];
              float v2515_data = s0_w0[253];
              tensorforge::intel_esimd::simd<float, 32> v2519_data(r0.template select<32, 1>(448));
              float v2520_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v2523_p = ((((((((((((((v1399_data * v1400_data) + (v1479_data * v1480_data)) + (v1559_data * v1560_data)) + (v1639_data * v1640_data)) + (v1719_data * v1720_data)) + (v1799_data * v1800_data)) + (v1879_data * v1880_data)) + (v1959_data * v1960_data)) + (v2039_data * v2040_data)) + (v2119_data * v2120_data)) + (v2199_data * v2200_data)) + (v2279_data * v2280_data)) + (v2359_data * v2360_data)) + (v2439_data * v2440_data)) + (v2519_data * v2520_data);
              float v2525_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v2528_p = ((((((((((((((v1399_data * v1405_data) + (v1479_data * v1485_data)) + (v1559_data * v1565_data)) + (v1639_data * v1645_data)) + (v1719_data * v1725_data)) + (v1799_data * v1805_data)) + (v1879_data * v1885_data)) + (v1959_data * v1965_data)) + (v2039_data * v2045_data)) + (v2119_data * v2125_data)) + (v2199_data * v2205_data)) + (v2279_data * v2285_data)) + (v2359_data * v2365_data)) + (v2439_data * v2445_data)) + (v2519_data * v2525_data);
              float v2530_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v2533_p = ((((((((((((((v1399_data * v1410_data) + (v1479_data * v1490_data)) + (v1559_data * v1570_data)) + (v1639_data * v1650_data)) + (v1719_data * v1730_data)) + (v1799_data * v1810_data)) + (v1879_data * v1890_data)) + (v1959_data * v1970_data)) + (v2039_data * v2050_data)) + (v2119_data * v2130_data)) + (v2199_data * v2210_data)) + (v2279_data * v2290_data)) + (v2359_data * v2370_data)) + (v2439_data * v2450_data)) + (v2519_data * v2530_data);
              float v2535_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 32> v2538_p = ((((((((((((((v1399_data * v1415_data) + (v1479_data * v1495_data)) + (v1559_data * v1575_data)) + (v1639_data * v1655_data)) + (v1719_data * v1735_data)) + (v1799_data * v1815_data)) + (v1879_data * v1895_data)) + (v1959_data * v1975_data)) + (v2039_data * v2055_data)) + (v2119_data * v2135_data)) + (v2199_data * v2215_data)) + (v2279_data * v2295_data)) + (v2359_data * v2375_data)) + (v2439_data * v2455_data)) + (v2519_data * v2535_data);
              float v2540_data = s0_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v2543_p = ((((((((((((((v1399_data * v1420_data) + (v1479_data * v1500_data)) + (v1559_data * v1580_data)) + (v1639_data * v1660_data)) + (v1719_data * v1740_data)) + (v1799_data * v1820_data)) + (v1879_data * v1900_data)) + (v1959_data * v1980_data)) + (v2039_data * v2060_data)) + (v2119_data * v2140_data)) + (v2199_data * v2220_data)) + (v2279_data * v2300_data)) + (v2359_data * v2380_data)) + (v2439_data * v2460_data)) + (v2519_data * v2540_data);
              float v2545_data = s0_w0[94];
              tensorforge::intel_esimd::simd<float, 32> v2548_p = ((((((((((((((v1399_data * v1425_data) + (v1479_data * v1505_data)) + (v1559_data * v1585_data)) + (v1639_data * v1665_data)) + (v1719_data * v1745_data)) + (v1799_data * v1825_data)) + (v1879_data * v1905_data)) + (v1959_data * v1985_data)) + (v2039_data * v2065_data)) + (v2119_data * v2145_data)) + (v2199_data * v2225_data)) + (v2279_data * v2305_data)) + (v2359_data * v2385_data)) + (v2439_data * v2465_data)) + (v2519_data * v2545_data);
              float v2550_data = s0_w0[110];
              tensorforge::intel_esimd::simd<float, 32> v2553_p = ((((((((((((((v1399_data * v1430_data) + (v1479_data * v1510_data)) + (v1559_data * v1590_data)) + (v1639_data * v1670_data)) + (v1719_data * v1750_data)) + (v1799_data * v1830_data)) + (v1879_data * v1910_data)) + (v1959_data * v1990_data)) + (v2039_data * v2070_data)) + (v2119_data * v2150_data)) + (v2199_data * v2230_data)) + (v2279_data * v2310_data)) + (v2359_data * v2390_data)) + (v2439_data * v2470_data)) + (v2519_data * v2550_data);
              float v2555_data = s0_w0[126];
              tensorforge::intel_esimd::simd<float, 32> v2558_p = ((((((((((((((v1399_data * v1435_data) + (v1479_data * v1515_data)) + (v1559_data * v1595_data)) + (v1639_data * v1675_data)) + (v1719_data * v1755_data)) + (v1799_data * v1835_data)) + (v1879_data * v1915_data)) + (v1959_data * v1995_data)) + (v2039_data * v2075_data)) + (v2119_data * v2155_data)) + (v2199_data * v2235_data)) + (v2279_data * v2315_data)) + (v2359_data * v2395_data)) + (v2439_data * v2475_data)) + (v2519_data * v2555_data);
              float v2560_data = s0_w0[142];
              tensorforge::intel_esimd::simd<float, 32> v2563_p = ((((((((((((((v1399_data * v1440_data) + (v1479_data * v1520_data)) + (v1559_data * v1600_data)) + (v1639_data * v1680_data)) + (v1719_data * v1760_data)) + (v1799_data * v1840_data)) + (v1879_data * v1920_data)) + (v1959_data * v2000_data)) + (v2039_data * v2080_data)) + (v2119_data * v2160_data)) + (v2199_data * v2240_data)) + (v2279_data * v2320_data)) + (v2359_data * v2400_data)) + (v2439_data * v2480_data)) + (v2519_data * v2560_data);
              float v2565_data = s0_w0[158];
              tensorforge::intel_esimd::simd<float, 32> v2568_p = ((((((((((((((v1399_data * v1445_data) + (v1479_data * v1525_data)) + (v1559_data * v1605_data)) + (v1639_data * v1685_data)) + (v1719_data * v1765_data)) + (v1799_data * v1845_data)) + (v1879_data * v1925_data)) + (v1959_data * v2005_data)) + (v2039_data * v2085_data)) + (v2119_data * v2165_data)) + (v2199_data * v2245_data)) + (v2279_data * v2325_data)) + (v2359_data * v2405_data)) + (v2439_data * v2485_data)) + (v2519_data * v2565_data);
              float v2570_data = s0_w0[174];
              tensorforge::intel_esimd::simd<float, 32> v2573_p = ((((((((((((((v1399_data * v1450_data) + (v1479_data * v1530_data)) + (v1559_data * v1610_data)) + (v1639_data * v1690_data)) + (v1719_data * v1770_data)) + (v1799_data * v1850_data)) + (v1879_data * v1930_data)) + (v1959_data * v2010_data)) + (v2039_data * v2090_data)) + (v2119_data * v2170_data)) + (v2199_data * v2250_data)) + (v2279_data * v2330_data)) + (v2359_data * v2410_data)) + (v2439_data * v2490_data)) + (v2519_data * v2570_data);
              float v2575_data = s0_w0[190];
              tensorforge::intel_esimd::simd<float, 32> v2578_p = ((((((((((((((v1399_data * v1455_data) + (v1479_data * v1535_data)) + (v1559_data * v1615_data)) + (v1639_data * v1695_data)) + (v1719_data * v1775_data)) + (v1799_data * v1855_data)) + (v1879_data * v1935_data)) + (v1959_data * v2015_data)) + (v2039_data * v2095_data)) + (v2119_data * v2175_data)) + (v2199_data * v2255_data)) + (v2279_data * v2335_data)) + (v2359_data * v2415_data)) + (v2439_data * v2495_data)) + (v2519_data * v2575_data);
              float v2580_data = s0_w0[206];
              tensorforge::intel_esimd::simd<float, 32> v2583_p = ((((((((((((((v1399_data * v1460_data) + (v1479_data * v1540_data)) + (v1559_data * v1620_data)) + (v1639_data * v1700_data)) + (v1719_data * v1780_data)) + (v1799_data * v1860_data)) + (v1879_data * v1940_data)) + (v1959_data * v2020_data)) + (v2039_data * v2100_data)) + (v2119_data * v2180_data)) + (v2199_data * v2260_data)) + (v2279_data * v2340_data)) + (v2359_data * v2420_data)) + (v2439_data * v2500_data)) + (v2519_data * v2580_data);
              float v2585_data = s0_w0[222];
              tensorforge::intel_esimd::simd<float, 32> v2588_p = ((((((((((((((v1399_data * v1465_data) + (v1479_data * v1545_data)) + (v1559_data * v1625_data)) + (v1639_data * v1705_data)) + (v1719_data * v1785_data)) + (v1799_data * v1865_data)) + (v1879_data * v1945_data)) + (v1959_data * v2025_data)) + (v2039_data * v2105_data)) + (v2119_data * v2185_data)) + (v2199_data * v2265_data)) + (v2279_data * v2345_data)) + (v2359_data * v2425_data)) + (v2439_data * v2505_data)) + (v2519_data * v2585_data);
              float v2590_data = s0_w0[238];
              tensorforge::intel_esimd::simd<float, 32> v2593_p = ((((((((((((((v1399_data * v1470_data) + (v1479_data * v1550_data)) + (v1559_data * v1630_data)) + (v1639_data * v1710_data)) + (v1719_data * v1790_data)) + (v1799_data * v1870_data)) + (v1879_data * v1950_data)) + (v1959_data * v2030_data)) + (v2039_data * v2110_data)) + (v2119_data * v2190_data)) + (v2199_data * v2270_data)) + (v2279_data * v2350_data)) + (v2359_data * v2430_data)) + (v2439_data * v2510_data)) + (v2519_data * v2590_data);
              float v2595_data = s0_w0[254];
              tensorforge::intel_esimd::simd<float, 32> v2598_p = ((((((((((((((v1399_data * v1475_data) + (v1479_data * v1555_data)) + (v1559_data * v1635_data)) + (v1639_data * v1715_data)) + (v1719_data * v1795_data)) + (v1799_data * v1875_data)) + (v1879_data * v1955_data)) + (v1959_data * v2035_data)) + (v2039_data * v2115_data)) + (v2119_data * v2195_data)) + (v2199_data * v2275_data)) + (v2279_data * v2355_data)) + (v2359_data * v2435_data)) + (v2439_data * v2515_data)) + (v2519_data * v2595_data);
              tensorforge::intel_esimd::simd<float, 32> v2599_data(r0.template select<32, 1>(480));
              // r0 = load{g>r}(wrap_glb_m4);
              #pragma unroll
              for (int32_t v2708_i1 = 0; v2708_i1 < 16; ++v2708_i1) {
                tensorforge::intel_esimd::simd<float, 32> v2710_data(0.0f);
                tensorforge::intel_esimd::simd<float, 16> v2710_data_part;
                v2710_data_part.copy_from(wrap_glb_m4 + ((v2708_i1 * 16)));
                v2710_data.template select<16, 1>(0) = v2710_data_part;
                r0.template select<32, 1>((v2708_i1 * 32)) = v2710_data;
              }
              float v2600_data = s0[15];
              ir1.template select<32, 1>(0) = (v2523_p + (v2599_data * v2600_data));
              float v2605_data = s0[31];
              ir1.template select<32, 1>(32) = (v2528_p + (v2599_data * v2605_data));
              float v2610_data = s0[47];
              ir1.template select<32, 1>(64) = (v2533_p + (v2599_data * v2610_data));
              float v2615_data = s0[63];
              ir1.template select<32, 1>(96) = (v2538_p + (v2599_data * v2615_data));
              float v2620_data = s0[79];
              ir1.template select<32, 1>(128) = (v2543_p + (v2599_data * v2620_data));
              float v2625_data = s0[95];
              ir1.template select<32, 1>(160) = (v2548_p + (v2599_data * v2625_data));
              float v2630_data = s0[111];
              ir1.template select<32, 1>(192) = (v2553_p + (v2599_data * v2630_data));
              float v2635_data = s0[127];
              ir1.template select<32, 1>(224) = (v2558_p + (v2599_data * v2635_data));
              float v2640_data = s0[143];
              ir1.template select<32, 1>(256) = (v2563_p + (v2599_data * v2640_data));
              float v2645_data = s0[159];
              ir1.template select<32, 1>(288) = (v2568_p + (v2599_data * v2645_data));
              float v2650_data = s0[175];
              ir1.template select<32, 1>(320) = (v2573_p + (v2599_data * v2650_data));
              float v2655_data = s0[191];
              ir1.template select<32, 1>(352) = (v2578_p + (v2599_data * v2655_data));
              float v2660_data = s0[207];
              ir1.template select<32, 1>(384) = (v2583_p + (v2599_data * v2660_data));
              float v2665_data = s0[223];
              ir1.template select<32, 1>(416) = (v2588_p + (v2599_data * v2665_data));
              float v2670_data = s0[239];
              ir1.template select<32, 1>(448) = (v2593_p + (v2599_data * v2670_data));
              float v2675_data = s0[255];
              ir1.template select<32, 1>(480) = (v2598_p + (v2599_data * v2675_data));
              // r1 = ir1
              // glb_m3 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v2683_i1 = 0; v2683_i1 < 16; ++v2683_i1) {
                tensorforge::intel_esimd::simd<float, 16> v2686_data(ir1.template select<16, 1>((v2683_i1 * 32)));
                v2686_data.copy_to(glb_m3 + ((v2683_i1 * 16)));
              }
            }
            else {
              // r0 = load{g>r}(wrap_glb_m4);
              #pragma unroll
              for (int32_t v2700_i1 = 0; v2700_i1 < 16; ++v2700_i1) {
                tensorforge::intel_esimd::simd<float, 32> v2702_data(0.0f);
                tensorforge::intel_esimd::simd<float, 16> v2702_data_part;
                v2702_data_part.copy_from(wrap_glb_m4 + ((v2700_i1 * 16)));
                v2702_data.template select<16, 1>(0) = v2702_data_part;
                r0.template select<32, 1>((v2700_i1 * 32)) = v2702_data;
              }
            }
            // s0 = load{g>s}(wrap_glb_m5[0, 1])
            tensorforge::intel_esimd::simd<float, 128> v2716_ld;
            v2716_ld.copy_from(wrap_glb_m5 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v2716_ld);
            tensorforge::intel_esimd::simd<float, 128> v2717_ld;
            v2717_ld.copy_from(wrap_glb_m5 + (0 + 0 + 4 * 0 + 128));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 128), v2717_ld);
            v2696_flagWord = v2697_flagWordNext;
            v2697_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


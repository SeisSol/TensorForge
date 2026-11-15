// === base name ===
kernel_15aedc64e8a07a37

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_15aedc64e8a07a37 = {{1, 16, 1}, 16, 48, 1, 16, 26624, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_15aedc64e8a07a37(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_15aedc64e8a07a37(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_15aedc64e8a07a37(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 3328 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_15aedc64e8a07a37(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_15aedc64e8a07a37(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_15aedc64e8a07a37(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_15aedc64e8a07a37_k;
inline void kernel_kernel_15aedc64e8a07a37(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_15aedc64e8a07a37_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<3328 * sizeof(double)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: amd_matrix_k4=1,lanes_per_mult=16
        // launch: 16 lanes (48 active) x 16 per block = block 1x16x1, 26624 B shared, occupancy grid
        // operands:
        //   m0 48×8(48×8) {0..48}×{0..8} strided
        //   m1 48×24(48×24) {0..48}×{0..24} none
        //   m2 24×8(24×8) {0..24}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":48,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3328}],"shared_bytes":26624,"shared_elements":3328,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[48,8]],"name":"m0","ordered":false,"parts":1,"shape":[48,8],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[48,24]],"name":"m1","ordered":false,"parts":1,"shape":[48,24],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[24,8]],"name":"m2","ordered":false,"parts":1,"shape":[24,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[48,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[48,8]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[48,24]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[48,24]},{"addressing":"strided","bbox":[[0,0],[24,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[24,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (208 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const double *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v2995_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v2995_flagSecondAhead < numElements0) ? v2995_flagSecondAhead : v3_batchId1)];
          const double *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 192 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 12; i += 2) {
            tensorforge::intel_esimd::simd<double, 32> v3003_ld;
            v3003_ld.copy_from(peel_glb_m2 + (0 + 0 + 2 * 0 + i * 16));
            tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + i * 16), v3003_ld);
          }
          uint32_t v2999_flagWord(flagWordFirst);
          uint32_t v3000_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const double *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 192 + 0 + m2_extraOffset];
            size_t v3007_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v3007_flagAheadAhead < numElements0) ? v3007_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v2999_flagWord);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v9_batchId0 * 384 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<double, 384> r0(0.0);
              // ir0 = +(glb_m1 * s0)
              // [(0, 48), (0, 8)] [(0, 24)]
              tensorforge::intel_esimd::simd<double, 384> ir0(0.0);
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<double, 16> v25_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 192> s0_w0 = tensorforge::slmLoad<double, 192>(s0 + 0);
              double v26_data = s0_w0[0];
              double v31_data = s0_w0[24];
              double v36_data = s0_w0[48];
              double v41_data = s0_w0[72];
              double v46_data = s0_w0[96];
              double v51_data = s0_w0[120];
              double v56_data = s0_w0[144];
              double v61_data = s0_w0[168];
              tensorforge::intel_esimd::simd<double, 16> v68_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<double, 16> v111_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v152_data(glb_m1_run1.template select<16, 1>(16));
              double v153_data = s0_w0[1];
              double v158_data = s0_w0[25];
              double v163_data = s0_w0[49];
              double v168_data = s0_w0[73];
              double v173_data = s0_w0[97];
              double v178_data = s0_w0[121];
              double v183_data = s0_w0[145];
              double v188_data = s0_w0[169];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<double, 16> v193_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v234_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<double, 16> v275_data(glb_m1_run3.template select<16, 1>(0));
              double v276_data = s0_w0[2];
              double v281_data = s0_w0[26];
              double v286_data = s0_w0[50];
              double v291_data = s0_w0[74];
              double v296_data = s0_w0[98];
              double v301_data = s0_w0[122];
              double v306_data = s0_w0[146];
              double v311_data = s0_w0[170];
              tensorforge::intel_esimd::simd<double, 16> v316_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run4;
              glb_m1_run4.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<double, 16> v357_data(glb_m1_run4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v398_data(glb_m1_run4.template select<16, 1>(16));
              double v399_data = s0_w0[3];
              double v404_data = s0_w0[27];
              double v409_data = s0_w0[51];
              double v414_data = s0_w0[75];
              double v419_data = s0_w0[99];
              double v424_data = s0_w0[123];
              double v429_data = s0_w0[147];
              double v434_data = s0_w0[171];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run5;
              glb_m1_run5.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<double, 16> v439_data(glb_m1_run5.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v480_data(glb_m1_run5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run6;
              glb_m1_run6.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<double, 16> v521_data(glb_m1_run6.template select<16, 1>(0));
              double v522_data = s0_w0[4];
              double v527_data = s0_w0[28];
              double v532_data = s0_w0[52];
              double v537_data = s0_w0[76];
              double v542_data = s0_w0[100];
              double v547_data = s0_w0[124];
              double v552_data = s0_w0[148];
              double v557_data = s0_w0[172];
              tensorforge::intel_esimd::simd<double, 16> v562_data(glb_m1_run6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run7;
              glb_m1_run7.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<double, 16> v603_data(glb_m1_run7.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v644_data(glb_m1_run7.template select<16, 1>(16));
              double v645_data = s0_w0[5];
              double v650_data = s0_w0[29];
              double v655_data = s0_w0[53];
              double v660_data = s0_w0[77];
              double v665_data = s0_w0[101];
              double v670_data = s0_w0[125];
              double v675_data = s0_w0[149];
              double v680_data = s0_w0[173];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run8;
              glb_m1_run8.copy_from(glb_m1 + (256_i32));
              tensorforge::intel_esimd::simd<double, 16> v685_data(glb_m1_run8.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v726_data(glb_m1_run8.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run9;
              glb_m1_run9.copy_from(glb_m1 + (288_i32));
              tensorforge::intel_esimd::simd<double, 16> v767_data(glb_m1_run9.template select<16, 1>(0));
              double v768_data = s0_w0[6];
              double v773_data = s0_w0[30];
              double v778_data = s0_w0[54];
              double v783_data = s0_w0[78];
              double v788_data = s0_w0[102];
              double v793_data = s0_w0[126];
              double v798_data = s0_w0[150];
              double v803_data = s0_w0[174];
              tensorforge::intel_esimd::simd<double, 16> v808_data(glb_m1_run9.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run10;
              glb_m1_run10.copy_from(glb_m1 + (320_i32));
              tensorforge::intel_esimd::simd<double, 16> v849_data(glb_m1_run10.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v890_data(glb_m1_run10.template select<16, 1>(16));
              double v891_data = s0_w0[7];
              double v896_data = s0_w0[31];
              double v901_data = s0_w0[55];
              double v906_data = s0_w0[79];
              double v911_data = s0_w0[103];
              double v916_data = s0_w0[127];
              double v921_data = s0_w0[151];
              double v926_data = s0_w0[175];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run11;
              glb_m1_run11.copy_from(glb_m1 + (352_i32));
              tensorforge::intel_esimd::simd<double, 16> v931_data(glb_m1_run11.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v972_data(glb_m1_run11.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run12;
              glb_m1_run12.copy_from(glb_m1 + (384_i32));
              tensorforge::intel_esimd::simd<double, 16> v1013_data(glb_m1_run12.template select<16, 1>(0));
              double v1014_data = s0_w0[8];
              double v1019_data = s0_w0[32];
              double v1024_data = s0_w0[56];
              double v1029_data = s0_w0[80];
              double v1034_data = s0_w0[104];
              double v1039_data = s0_w0[128];
              double v1044_data = s0_w0[152];
              double v1049_data = s0_w0[176];
              tensorforge::intel_esimd::simd<double, 16> v1054_data(glb_m1_run12.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run13;
              glb_m1_run13.copy_from(glb_m1 + (416_i32));
              tensorforge::intel_esimd::simd<double, 16> v1095_data(glb_m1_run13.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v1136_data(glb_m1_run13.template select<16, 1>(16));
              double v1137_data = s0_w0[9];
              double v1142_data = s0_w0[33];
              double v1147_data = s0_w0[57];
              double v1152_data = s0_w0[81];
              double v1157_data = s0_w0[105];
              double v1162_data = s0_w0[129];
              double v1167_data = s0_w0[153];
              double v1172_data = s0_w0[177];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run14;
              glb_m1_run14.copy_from(glb_m1 + (448_i32));
              tensorforge::intel_esimd::simd<double, 16> v1177_data(glb_m1_run14.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v1218_data(glb_m1_run14.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run15;
              glb_m1_run15.copy_from(glb_m1 + (480_i32));
              tensorforge::intel_esimd::simd<double, 16> v1259_data(glb_m1_run15.template select<16, 1>(0));
              double v1260_data = s0_w0[10];
              double v1265_data = s0_w0[34];
              double v1270_data = s0_w0[58];
              double v1275_data = s0_w0[82];
              double v1280_data = s0_w0[106];
              double v1285_data = s0_w0[130];
              double v1290_data = s0_w0[154];
              double v1295_data = s0_w0[178];
              tensorforge::intel_esimd::simd<double, 16> v1300_data(glb_m1_run15.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run16;
              glb_m1_run16.copy_from(glb_m1 + (512_i32));
              tensorforge::intel_esimd::simd<double, 16> v1341_data(glb_m1_run16.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v1382_data(glb_m1_run16.template select<16, 1>(16));
              double v1383_data = s0_w0[11];
              double v1388_data = s0_w0[35];
              double v1393_data = s0_w0[59];
              double v1398_data = s0_w0[83];
              double v1403_data = s0_w0[107];
              double v1408_data = s0_w0[131];
              double v1413_data = s0_w0[155];
              double v1418_data = s0_w0[179];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run17;
              glb_m1_run17.copy_from(glb_m1 + (544_i32));
              tensorforge::intel_esimd::simd<double, 16> v1423_data(glb_m1_run17.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v1464_data(glb_m1_run17.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run18;
              glb_m1_run18.copy_from(glb_m1 + (576_i32));
              tensorforge::intel_esimd::simd<double, 16> v1505_data(glb_m1_run18.template select<16, 1>(0));
              double v1506_data = s0_w0[12];
              double v1511_data = s0_w0[36];
              double v1516_data = s0_w0[60];
              double v1521_data = s0_w0[84];
              double v1526_data = s0_w0[108];
              double v1531_data = s0_w0[132];
              double v1536_data = s0_w0[156];
              double v1541_data = s0_w0[180];
              tensorforge::intel_esimd::simd<double, 16> v1546_data(glb_m1_run18.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run19;
              glb_m1_run19.copy_from(glb_m1 + (608_i32));
              tensorforge::intel_esimd::simd<double, 16> v1587_data(glb_m1_run19.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v1628_data(glb_m1_run19.template select<16, 1>(16));
              double v1629_data = s0_w0[13];
              double v1634_data = s0_w0[37];
              double v1639_data = s0_w0[61];
              double v1644_data = s0_w0[85];
              double v1649_data = s0_w0[109];
              double v1654_data = s0_w0[133];
              double v1659_data = s0_w0[157];
              double v1664_data = s0_w0[181];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run20;
              glb_m1_run20.copy_from(glb_m1 + (640_i32));
              tensorforge::intel_esimd::simd<double, 16> v1669_data(glb_m1_run20.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v1710_data(glb_m1_run20.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run21;
              glb_m1_run21.copy_from(glb_m1 + (672_i32));
              tensorforge::intel_esimd::simd<double, 16> v1751_data(glb_m1_run21.template select<16, 1>(0));
              double v1752_data = s0_w0[14];
              double v1757_data = s0_w0[38];
              double v1762_data = s0_w0[62];
              double v1767_data = s0_w0[86];
              double v1772_data = s0_w0[110];
              double v1777_data = s0_w0[134];
              double v1782_data = s0_w0[158];
              double v1787_data = s0_w0[182];
              tensorforge::intel_esimd::simd<double, 16> v1792_data(glb_m1_run21.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run22;
              glb_m1_run22.copy_from(glb_m1 + (704_i32));
              tensorforge::intel_esimd::simd<double, 16> v1833_data(glb_m1_run22.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v1874_data(glb_m1_run22.template select<16, 1>(16));
              double v1875_data = s0_w0[15];
              double v1880_data = s0_w0[39];
              double v1885_data = s0_w0[63];
              double v1890_data = s0_w0[87];
              double v1895_data = s0_w0[111];
              double v1900_data = s0_w0[135];
              double v1905_data = s0_w0[159];
              double v1910_data = s0_w0[183];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run23;
              glb_m1_run23.copy_from(glb_m1 + (736_i32));
              tensorforge::intel_esimd::simd<double, 16> v1915_data(glb_m1_run23.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v1956_data(glb_m1_run23.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run24;
              glb_m1_run24.copy_from(glb_m1 + (768_i32));
              tensorforge::intel_esimd::simd<double, 16> v1997_data(glb_m1_run24.template select<16, 1>(0));
              double v1998_data = s0_w0[16];
              double v2003_data = s0_w0[40];
              double v2008_data = s0_w0[64];
              double v2013_data = s0_w0[88];
              double v2018_data = s0_w0[112];
              double v2023_data = s0_w0[136];
              double v2028_data = s0_w0[160];
              double v2033_data = s0_w0[184];
              tensorforge::intel_esimd::simd<double, 16> v2038_data(glb_m1_run24.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run25;
              glb_m1_run25.copy_from(glb_m1 + (800_i32));
              tensorforge::intel_esimd::simd<double, 16> v2079_data(glb_m1_run25.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v2120_data(glb_m1_run25.template select<16, 1>(16));
              double v2121_data = s0_w0[17];
              double v2126_data = s0_w0[41];
              double v2131_data = s0_w0[65];
              double v2136_data = s0_w0[89];
              double v2141_data = s0_w0[113];
              double v2146_data = s0_w0[137];
              double v2151_data = s0_w0[161];
              double v2156_data = s0_w0[185];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run26;
              glb_m1_run26.copy_from(glb_m1 + (832_i32));
              tensorforge::intel_esimd::simd<double, 16> v2161_data(glb_m1_run26.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v2202_data(glb_m1_run26.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run27;
              glb_m1_run27.copy_from(glb_m1 + (864_i32));
              tensorforge::intel_esimd::simd<double, 16> v2243_data(glb_m1_run27.template select<16, 1>(0));
              double v2244_data = s0_w0[18];
              double v2249_data = s0_w0[42];
              double v2254_data = s0_w0[66];
              double v2259_data = s0_w0[90];
              double v2264_data = s0_w0[114];
              double v2269_data = s0_w0[138];
              double v2274_data = s0_w0[162];
              double v2279_data = s0_w0[186];
              tensorforge::intel_esimd::simd<double, 16> v2284_data(glb_m1_run27.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run28;
              glb_m1_run28.copy_from(glb_m1 + (896_i32));
              tensorforge::intel_esimd::simd<double, 16> v2325_data(glb_m1_run28.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v2366_data(glb_m1_run28.template select<16, 1>(16));
              double v2367_data = s0_w0[19];
              double v2372_data = s0_w0[43];
              double v2377_data = s0_w0[67];
              double v2382_data = s0_w0[91];
              double v2387_data = s0_w0[115];
              double v2392_data = s0_w0[139];
              double v2397_data = s0_w0[163];
              double v2402_data = s0_w0[187];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run29;
              glb_m1_run29.copy_from(glb_m1 + (928_i32));
              tensorforge::intel_esimd::simd<double, 16> v2407_data(glb_m1_run29.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v2448_data(glb_m1_run29.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run30;
              glb_m1_run30.copy_from(glb_m1 + (960_i32));
              tensorforge::intel_esimd::simd<double, 16> v2489_data(glb_m1_run30.template select<16, 1>(0));
              double v2490_data = s0_w0[20];
              double v2495_data = s0_w0[44];
              double v2500_data = s0_w0[68];
              double v2505_data = s0_w0[92];
              double v2510_data = s0_w0[116];
              double v2515_data = s0_w0[140];
              double v2520_data = s0_w0[164];
              double v2525_data = s0_w0[188];
              tensorforge::intel_esimd::simd<double, 16> v2530_data(glb_m1_run30.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run31;
              glb_m1_run31.copy_from(glb_m1 + (992_i32));
              tensorforge::intel_esimd::simd<double, 16> v2571_data(glb_m1_run31.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v2612_data(glb_m1_run31.template select<16, 1>(16));
              double v2613_data = s0_w0[21];
              double v2618_data = s0_w0[45];
              double v2623_data = s0_w0[69];
              double v2628_data = s0_w0[93];
              double v2633_data = s0_w0[117];
              double v2638_data = s0_w0[141];
              double v2643_data = s0_w0[165];
              double v2648_data = s0_w0[189];
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run32;
              glb_m1_run32.copy_from(glb_m1 + (1024_i32));
              tensorforge::intel_esimd::simd<double, 16> v2653_data(glb_m1_run32.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v2694_data(glb_m1_run32.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run33;
              glb_m1_run33.copy_from(glb_m1 + (1056_i32));
              tensorforge::intel_esimd::simd<double, 16> v2735_data(glb_m1_run33.template select<16, 1>(0));
              double v2736_data = s0_w0[22];
              double v2741_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v2744_p = ((((((((((((((((((((((v25_data * v31_data) + (v152_data * v158_data)) + (v275_data * v281_data)) + (v398_data * v404_data)) + (v521_data * v527_data)) + (v644_data * v650_data)) + (v767_data * v773_data)) + (v890_data * v896_data)) + (v1013_data * v1019_data)) + (v1136_data * v1142_data)) + (v1259_data * v1265_data)) + (v1382_data * v1388_data)) + (v1505_data * v1511_data)) + (v1628_data * v1634_data)) + (v1751_data * v1757_data)) + (v1874_data * v1880_data)) + (v1997_data * v2003_data)) + (v2120_data * v2126_data)) + (v2243_data * v2249_data)) + (v2366_data * v2372_data)) + (v2489_data * v2495_data)) + (v2612_data * v2618_data)) + (v2735_data * v2741_data);
              double v2746_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v2749_p = ((((((((((((((((((((((v25_data * v36_data) + (v152_data * v163_data)) + (v275_data * v286_data)) + (v398_data * v409_data)) + (v521_data * v532_data)) + (v644_data * v655_data)) + (v767_data * v778_data)) + (v890_data * v901_data)) + (v1013_data * v1024_data)) + (v1136_data * v1147_data)) + (v1259_data * v1270_data)) + (v1382_data * v1393_data)) + (v1505_data * v1516_data)) + (v1628_data * v1639_data)) + (v1751_data * v1762_data)) + (v1874_data * v1885_data)) + (v1997_data * v2008_data)) + (v2120_data * v2131_data)) + (v2243_data * v2254_data)) + (v2366_data * v2377_data)) + (v2489_data * v2500_data)) + (v2612_data * v2623_data)) + (v2735_data * v2746_data);
              double v2751_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v2754_p = ((((((((((((((((((((((v25_data * v41_data) + (v152_data * v168_data)) + (v275_data * v291_data)) + (v398_data * v414_data)) + (v521_data * v537_data)) + (v644_data * v660_data)) + (v767_data * v783_data)) + (v890_data * v906_data)) + (v1013_data * v1029_data)) + (v1136_data * v1152_data)) + (v1259_data * v1275_data)) + (v1382_data * v1398_data)) + (v1505_data * v1521_data)) + (v1628_data * v1644_data)) + (v1751_data * v1767_data)) + (v1874_data * v1890_data)) + (v1997_data * v2013_data)) + (v2120_data * v2136_data)) + (v2243_data * v2259_data)) + (v2366_data * v2382_data)) + (v2489_data * v2505_data)) + (v2612_data * v2628_data)) + (v2735_data * v2751_data);
              double v2756_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v2759_p = ((((((((((((((((((((((v25_data * v46_data) + (v152_data * v173_data)) + (v275_data * v296_data)) + (v398_data * v419_data)) + (v521_data * v542_data)) + (v644_data * v665_data)) + (v767_data * v788_data)) + (v890_data * v911_data)) + (v1013_data * v1034_data)) + (v1136_data * v1157_data)) + (v1259_data * v1280_data)) + (v1382_data * v1403_data)) + (v1505_data * v1526_data)) + (v1628_data * v1649_data)) + (v1751_data * v1772_data)) + (v1874_data * v1895_data)) + (v1997_data * v2018_data)) + (v2120_data * v2141_data)) + (v2243_data * v2264_data)) + (v2366_data * v2387_data)) + (v2489_data * v2510_data)) + (v2612_data * v2633_data)) + (v2735_data * v2756_data);
              double v2761_data = s0_w0[142];
              tensorforge::intel_esimd::simd<double, 16> v2764_p = ((((((((((((((((((((((v25_data * v51_data) + (v152_data * v178_data)) + (v275_data * v301_data)) + (v398_data * v424_data)) + (v521_data * v547_data)) + (v644_data * v670_data)) + (v767_data * v793_data)) + (v890_data * v916_data)) + (v1013_data * v1039_data)) + (v1136_data * v1162_data)) + (v1259_data * v1285_data)) + (v1382_data * v1408_data)) + (v1505_data * v1531_data)) + (v1628_data * v1654_data)) + (v1751_data * v1777_data)) + (v1874_data * v1900_data)) + (v1997_data * v2023_data)) + (v2120_data * v2146_data)) + (v2243_data * v2269_data)) + (v2366_data * v2392_data)) + (v2489_data * v2515_data)) + (v2612_data * v2638_data)) + (v2735_data * v2761_data);
              double v2766_data = s0_w0[166];
              tensorforge::intel_esimd::simd<double, 16> v2769_p = ((((((((((((((((((((((v25_data * v56_data) + (v152_data * v183_data)) + (v275_data * v306_data)) + (v398_data * v429_data)) + (v521_data * v552_data)) + (v644_data * v675_data)) + (v767_data * v798_data)) + (v890_data * v921_data)) + (v1013_data * v1044_data)) + (v1136_data * v1167_data)) + (v1259_data * v1290_data)) + (v1382_data * v1413_data)) + (v1505_data * v1536_data)) + (v1628_data * v1659_data)) + (v1751_data * v1782_data)) + (v1874_data * v1905_data)) + (v1997_data * v2028_data)) + (v2120_data * v2151_data)) + (v2243_data * v2274_data)) + (v2366_data * v2397_data)) + (v2489_data * v2520_data)) + (v2612_data * v2643_data)) + (v2735_data * v2766_data);
              double v2771_data = s0_w0[190];
              tensorforge::intel_esimd::simd<double, 16> v2774_p = ((((((((((((((((((((((v25_data * v61_data) + (v152_data * v188_data)) + (v275_data * v311_data)) + (v398_data * v434_data)) + (v521_data * v557_data)) + (v644_data * v680_data)) + (v767_data * v803_data)) + (v890_data * v926_data)) + (v1013_data * v1049_data)) + (v1136_data * v1172_data)) + (v1259_data * v1295_data)) + (v1382_data * v1418_data)) + (v1505_data * v1541_data)) + (v1628_data * v1664_data)) + (v1751_data * v1787_data)) + (v1874_data * v1910_data)) + (v1997_data * v2033_data)) + (v2120_data * v2156_data)) + (v2243_data * v2279_data)) + (v2366_data * v2402_data)) + (v2489_data * v2525_data)) + (v2612_data * v2648_data)) + (v2735_data * v2771_data);
              tensorforge::intel_esimd::simd<double, 16> v2776_data(glb_m1_run33.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v2780_p = ((((((((((((((((((((((v68_data * v26_data) + (v193_data * v153_data)) + (v316_data * v276_data)) + (v439_data * v399_data)) + (v562_data * v522_data)) + (v685_data * v645_data)) + (v808_data * v768_data)) + (v931_data * v891_data)) + (v1054_data * v1014_data)) + (v1177_data * v1137_data)) + (v1300_data * v1260_data)) + (v1423_data * v1383_data)) + (v1546_data * v1506_data)) + (v1669_data * v1629_data)) + (v1792_data * v1752_data)) + (v1915_data * v1875_data)) + (v2038_data * v1998_data)) + (v2161_data * v2121_data)) + (v2284_data * v2244_data)) + (v2407_data * v2367_data)) + (v2530_data * v2490_data)) + (v2653_data * v2613_data)) + (v2776_data * v2736_data);
              tensorforge::intel_esimd::simd<double, 16> v2785_p = ((((((((((((((((((((((v68_data * v31_data) + (v193_data * v158_data)) + (v316_data * v281_data)) + (v439_data * v404_data)) + (v562_data * v527_data)) + (v685_data * v650_data)) + (v808_data * v773_data)) + (v931_data * v896_data)) + (v1054_data * v1019_data)) + (v1177_data * v1142_data)) + (v1300_data * v1265_data)) + (v1423_data * v1388_data)) + (v1546_data * v1511_data)) + (v1669_data * v1634_data)) + (v1792_data * v1757_data)) + (v1915_data * v1880_data)) + (v2038_data * v2003_data)) + (v2161_data * v2126_data)) + (v2284_data * v2249_data)) + (v2407_data * v2372_data)) + (v2530_data * v2495_data)) + (v2653_data * v2618_data)) + (v2776_data * v2741_data);
              tensorforge::intel_esimd::simd<double, 16> v2790_p = ((((((((((((((((((((((v68_data * v36_data) + (v193_data * v163_data)) + (v316_data * v286_data)) + (v439_data * v409_data)) + (v562_data * v532_data)) + (v685_data * v655_data)) + (v808_data * v778_data)) + (v931_data * v901_data)) + (v1054_data * v1024_data)) + (v1177_data * v1147_data)) + (v1300_data * v1270_data)) + (v1423_data * v1393_data)) + (v1546_data * v1516_data)) + (v1669_data * v1639_data)) + (v1792_data * v1762_data)) + (v1915_data * v1885_data)) + (v2038_data * v2008_data)) + (v2161_data * v2131_data)) + (v2284_data * v2254_data)) + (v2407_data * v2377_data)) + (v2530_data * v2500_data)) + (v2653_data * v2623_data)) + (v2776_data * v2746_data);
              tensorforge::intel_esimd::simd<double, 16> v2795_p = ((((((((((((((((((((((v68_data * v41_data) + (v193_data * v168_data)) + (v316_data * v291_data)) + (v439_data * v414_data)) + (v562_data * v537_data)) + (v685_data * v660_data)) + (v808_data * v783_data)) + (v931_data * v906_data)) + (v1054_data * v1029_data)) + (v1177_data * v1152_data)) + (v1300_data * v1275_data)) + (v1423_data * v1398_data)) + (v1546_data * v1521_data)) + (v1669_data * v1644_data)) + (v1792_data * v1767_data)) + (v1915_data * v1890_data)) + (v2038_data * v2013_data)) + (v2161_data * v2136_data)) + (v2284_data * v2259_data)) + (v2407_data * v2382_data)) + (v2530_data * v2505_data)) + (v2653_data * v2628_data)) + (v2776_data * v2751_data);
              tensorforge::intel_esimd::simd<double, 16> v2800_p = ((((((((((((((((((((((v68_data * v46_data) + (v193_data * v173_data)) + (v316_data * v296_data)) + (v439_data * v419_data)) + (v562_data * v542_data)) + (v685_data * v665_data)) + (v808_data * v788_data)) + (v931_data * v911_data)) + (v1054_data * v1034_data)) + (v1177_data * v1157_data)) + (v1300_data * v1280_data)) + (v1423_data * v1403_data)) + (v1546_data * v1526_data)) + (v1669_data * v1649_data)) + (v1792_data * v1772_data)) + (v1915_data * v1895_data)) + (v2038_data * v2018_data)) + (v2161_data * v2141_data)) + (v2284_data * v2264_data)) + (v2407_data * v2387_data)) + (v2530_data * v2510_data)) + (v2653_data * v2633_data)) + (v2776_data * v2756_data);
              tensorforge::intel_esimd::simd<double, 16> v2805_p = ((((((((((((((((((((((v68_data * v51_data) + (v193_data * v178_data)) + (v316_data * v301_data)) + (v439_data * v424_data)) + (v562_data * v547_data)) + (v685_data * v670_data)) + (v808_data * v793_data)) + (v931_data * v916_data)) + (v1054_data * v1039_data)) + (v1177_data * v1162_data)) + (v1300_data * v1285_data)) + (v1423_data * v1408_data)) + (v1546_data * v1531_data)) + (v1669_data * v1654_data)) + (v1792_data * v1777_data)) + (v1915_data * v1900_data)) + (v2038_data * v2023_data)) + (v2161_data * v2146_data)) + (v2284_data * v2269_data)) + (v2407_data * v2392_data)) + (v2530_data * v2515_data)) + (v2653_data * v2638_data)) + (v2776_data * v2761_data);
              tensorforge::intel_esimd::simd<double, 16> v2810_p = ((((((((((((((((((((((v68_data * v56_data) + (v193_data * v183_data)) + (v316_data * v306_data)) + (v439_data * v429_data)) + (v562_data * v552_data)) + (v685_data * v675_data)) + (v808_data * v798_data)) + (v931_data * v921_data)) + (v1054_data * v1044_data)) + (v1177_data * v1167_data)) + (v1300_data * v1290_data)) + (v1423_data * v1413_data)) + (v1546_data * v1536_data)) + (v1669_data * v1659_data)) + (v1792_data * v1782_data)) + (v1915_data * v1905_data)) + (v2038_data * v2028_data)) + (v2161_data * v2151_data)) + (v2284_data * v2274_data)) + (v2407_data * v2397_data)) + (v2530_data * v2520_data)) + (v2653_data * v2643_data)) + (v2776_data * v2766_data);
              tensorforge::intel_esimd::simd<double, 16> v2815_p = ((((((((((((((((((((((v68_data * v61_data) + (v193_data * v188_data)) + (v316_data * v311_data)) + (v439_data * v434_data)) + (v562_data * v557_data)) + (v685_data * v680_data)) + (v808_data * v803_data)) + (v931_data * v926_data)) + (v1054_data * v1049_data)) + (v1177_data * v1172_data)) + (v1300_data * v1295_data)) + (v1423_data * v1418_data)) + (v1546_data * v1541_data)) + (v1669_data * v1664_data)) + (v1792_data * v1787_data)) + (v1915_data * v1910_data)) + (v2038_data * v2033_data)) + (v2161_data * v2156_data)) + (v2284_data * v2279_data)) + (v2407_data * v2402_data)) + (v2530_data * v2525_data)) + (v2653_data * v2648_data)) + (v2776_data * v2771_data);
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run34;
              glb_m1_run34.copy_from(glb_m1 + (1088_i32));
              tensorforge::intel_esimd::simd<double, 16> v2817_data(glb_m1_run34.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 16> v2821_p = ((((((((((((((((((((((v111_data * v26_data) + (v234_data * v153_data)) + (v357_data * v276_data)) + (v480_data * v399_data)) + (v603_data * v522_data)) + (v726_data * v645_data)) + (v849_data * v768_data)) + (v972_data * v891_data)) + (v1095_data * v1014_data)) + (v1218_data * v1137_data)) + (v1341_data * v1260_data)) + (v1464_data * v1383_data)) + (v1587_data * v1506_data)) + (v1710_data * v1629_data)) + (v1833_data * v1752_data)) + (v1956_data * v1875_data)) + (v2079_data * v1998_data)) + (v2202_data * v2121_data)) + (v2325_data * v2244_data)) + (v2448_data * v2367_data)) + (v2571_data * v2490_data)) + (v2694_data * v2613_data)) + (v2817_data * v2736_data);
              tensorforge::intel_esimd::simd<double, 16> v2826_p = ((((((((((((((((((((((v111_data * v31_data) + (v234_data * v158_data)) + (v357_data * v281_data)) + (v480_data * v404_data)) + (v603_data * v527_data)) + (v726_data * v650_data)) + (v849_data * v773_data)) + (v972_data * v896_data)) + (v1095_data * v1019_data)) + (v1218_data * v1142_data)) + (v1341_data * v1265_data)) + (v1464_data * v1388_data)) + (v1587_data * v1511_data)) + (v1710_data * v1634_data)) + (v1833_data * v1757_data)) + (v1956_data * v1880_data)) + (v2079_data * v2003_data)) + (v2202_data * v2126_data)) + (v2325_data * v2249_data)) + (v2448_data * v2372_data)) + (v2571_data * v2495_data)) + (v2694_data * v2618_data)) + (v2817_data * v2741_data);
              tensorforge::intel_esimd::simd<double, 16> v2831_p = ((((((((((((((((((((((v111_data * v36_data) + (v234_data * v163_data)) + (v357_data * v286_data)) + (v480_data * v409_data)) + (v603_data * v532_data)) + (v726_data * v655_data)) + (v849_data * v778_data)) + (v972_data * v901_data)) + (v1095_data * v1024_data)) + (v1218_data * v1147_data)) + (v1341_data * v1270_data)) + (v1464_data * v1393_data)) + (v1587_data * v1516_data)) + (v1710_data * v1639_data)) + (v1833_data * v1762_data)) + (v1956_data * v1885_data)) + (v2079_data * v2008_data)) + (v2202_data * v2131_data)) + (v2325_data * v2254_data)) + (v2448_data * v2377_data)) + (v2571_data * v2500_data)) + (v2694_data * v2623_data)) + (v2817_data * v2746_data);
              tensorforge::intel_esimd::simd<double, 16> v2836_p = ((((((((((((((((((((((v111_data * v41_data) + (v234_data * v168_data)) + (v357_data * v291_data)) + (v480_data * v414_data)) + (v603_data * v537_data)) + (v726_data * v660_data)) + (v849_data * v783_data)) + (v972_data * v906_data)) + (v1095_data * v1029_data)) + (v1218_data * v1152_data)) + (v1341_data * v1275_data)) + (v1464_data * v1398_data)) + (v1587_data * v1521_data)) + (v1710_data * v1644_data)) + (v1833_data * v1767_data)) + (v1956_data * v1890_data)) + (v2079_data * v2013_data)) + (v2202_data * v2136_data)) + (v2325_data * v2259_data)) + (v2448_data * v2382_data)) + (v2571_data * v2505_data)) + (v2694_data * v2628_data)) + (v2817_data * v2751_data);
              tensorforge::intel_esimd::simd<double, 16> v2841_p = ((((((((((((((((((((((v111_data * v46_data) + (v234_data * v173_data)) + (v357_data * v296_data)) + (v480_data * v419_data)) + (v603_data * v542_data)) + (v726_data * v665_data)) + (v849_data * v788_data)) + (v972_data * v911_data)) + (v1095_data * v1034_data)) + (v1218_data * v1157_data)) + (v1341_data * v1280_data)) + (v1464_data * v1403_data)) + (v1587_data * v1526_data)) + (v1710_data * v1649_data)) + (v1833_data * v1772_data)) + (v1956_data * v1895_data)) + (v2079_data * v2018_data)) + (v2202_data * v2141_data)) + (v2325_data * v2264_data)) + (v2448_data * v2387_data)) + (v2571_data * v2510_data)) + (v2694_data * v2633_data)) + (v2817_data * v2756_data);
              tensorforge::intel_esimd::simd<double, 16> v2846_p = ((((((((((((((((((((((v111_data * v51_data) + (v234_data * v178_data)) + (v357_data * v301_data)) + (v480_data * v424_data)) + (v603_data * v547_data)) + (v726_data * v670_data)) + (v849_data * v793_data)) + (v972_data * v916_data)) + (v1095_data * v1039_data)) + (v1218_data * v1162_data)) + (v1341_data * v1285_data)) + (v1464_data * v1408_data)) + (v1587_data * v1531_data)) + (v1710_data * v1654_data)) + (v1833_data * v1777_data)) + (v1956_data * v1900_data)) + (v2079_data * v2023_data)) + (v2202_data * v2146_data)) + (v2325_data * v2269_data)) + (v2448_data * v2392_data)) + (v2571_data * v2515_data)) + (v2694_data * v2638_data)) + (v2817_data * v2761_data);
              tensorforge::intel_esimd::simd<double, 16> v2851_p = ((((((((((((((((((((((v111_data * v56_data) + (v234_data * v183_data)) + (v357_data * v306_data)) + (v480_data * v429_data)) + (v603_data * v552_data)) + (v726_data * v675_data)) + (v849_data * v798_data)) + (v972_data * v921_data)) + (v1095_data * v1044_data)) + (v1218_data * v1167_data)) + (v1341_data * v1290_data)) + (v1464_data * v1413_data)) + (v1587_data * v1536_data)) + (v1710_data * v1659_data)) + (v1833_data * v1782_data)) + (v1956_data * v1905_data)) + (v2079_data * v2028_data)) + (v2202_data * v2151_data)) + (v2325_data * v2274_data)) + (v2448_data * v2397_data)) + (v2571_data * v2520_data)) + (v2694_data * v2643_data)) + (v2817_data * v2766_data);
              tensorforge::intel_esimd::simd<double, 16> v2856_p = ((((((((((((((((((((((v111_data * v61_data) + (v234_data * v188_data)) + (v357_data * v311_data)) + (v480_data * v434_data)) + (v603_data * v557_data)) + (v726_data * v680_data)) + (v849_data * v803_data)) + (v972_data * v926_data)) + (v1095_data * v1049_data)) + (v1218_data * v1172_data)) + (v1341_data * v1295_data)) + (v1464_data * v1418_data)) + (v1587_data * v1541_data)) + (v1710_data * v1664_data)) + (v1833_data * v1787_data)) + (v1956_data * v1910_data)) + (v2079_data * v2033_data)) + (v2202_data * v2156_data)) + (v2325_data * v2279_data)) + (v2448_data * v2402_data)) + (v2571_data * v2525_data)) + (v2694_data * v2648_data)) + (v2817_data * v2771_data);
              tensorforge::intel_esimd::simd<double, 16> v2858_data(glb_m1_run34.template select<16, 1>(16));
              double v2859_data = s0_w0[23];
              ir0.template select<16, 1>(0) = ((((((((((((((((((((((((v25_data * v26_data) + (v152_data * v153_data)) + (v275_data * v276_data)) + (v398_data * v399_data)) + (v521_data * v522_data)) + (v644_data * v645_data)) + (v767_data * v768_data)) + (v890_data * v891_data)) + (v1013_data * v1014_data)) + (v1136_data * v1137_data)) + (v1259_data * v1260_data)) + (v1382_data * v1383_data)) + (v1505_data * v1506_data)) + (v1628_data * v1629_data)) + (v1751_data * v1752_data)) + (v1874_data * v1875_data)) + (v1997_data * v1998_data)) + (v2120_data * v2121_data)) + (v2243_data * v2244_data)) + (v2366_data * v2367_data)) + (v2489_data * v2490_data)) + (v2612_data * v2613_data)) + (v2735_data * v2736_data)) + (v2858_data * v2859_data));
              double v2864_data = s0_w0[47];
              ir0.template select<16, 1>(48) = (v2744_p + (v2858_data * v2864_data));
              double v2869_data = s0_w0[71];
              ir0.template select<16, 1>(96) = (v2749_p + (v2858_data * v2869_data));
              double v2874_data = s0_w0[95];
              ir0.template select<16, 1>(144) = (v2754_p + (v2858_data * v2874_data));
              double v2879_data = s0_w0[119];
              ir0.template select<16, 1>(192) = (v2759_p + (v2858_data * v2879_data));
              double v2884_data = s0_w0[143];
              ir0.template select<16, 1>(240) = (v2764_p + (v2858_data * v2884_data));
              double v2889_data = s0_w0[167];
              ir0.template select<16, 1>(288) = (v2769_p + (v2858_data * v2889_data));
              double v2894_data = s0_w0[191];
              ir0.template select<16, 1>(336) = (v2774_p + (v2858_data * v2894_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run35;
              glb_m1_run35.copy_from(glb_m1 + (1120_i32));
              tensorforge::intel_esimd::simd<double, 16> v2899_data(glb_m1_run35.template select<16, 1>(0));
              ir0.template select<16, 1>(16) = (v2780_p + (v2899_data * v2859_data));
              ir0.template select<16, 1>(64) = (v2785_p + (v2899_data * v2864_data));
              ir0.template select<16, 1>(112) = (v2790_p + (v2899_data * v2869_data));
              ir0.template select<16, 1>(160) = (v2795_p + (v2899_data * v2874_data));
              ir0.template select<16, 1>(208) = (v2800_p + (v2899_data * v2879_data));
              ir0.template select<16, 1>(256) = (v2805_p + (v2899_data * v2884_data));
              ir0.template select<16, 1>(304) = (v2810_p + (v2899_data * v2889_data));
              ir0.template select<16, 1>(352) = (v2815_p + (v2899_data * v2894_data));
              tensorforge::intel_esimd::simd<double, 16> v2940_data(glb_m1_run35.template select<16, 1>(16));
              ir0.template select<16, 1>(32) = (v2821_p + (v2940_data * v2859_data));
              ir0.template select<16, 1>(80) = (v2826_p + (v2940_data * v2864_data));
              ir0.template select<16, 1>(128) = (v2831_p + (v2940_data * v2869_data));
              ir0.template select<16, 1>(176) = (v2836_p + (v2940_data * v2874_data));
              ir0.template select<16, 1>(224) = (v2841_p + (v2940_data * v2879_data));
              ir0.template select<16, 1>(272) = (v2846_p + (v2940_data * v2884_data));
              ir0.template select<16, 1>(320) = (v2851_p + (v2940_data * v2889_data));
              ir0.template select<16, 1>(368) = (v2856_p + (v2940_data * v2894_data));
              // r0 = ir0
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v2986_i0 = 0; v2986_i0 < 3; ++v2986_i0) {
                int32_t v2988_a = v2986_i0 * 16;
                #pragma unroll
                for (int32_t v2987_i1 = 0; v2987_i1 < 8; ++v2987_i1) {
                  int32_t v2989_a = v2987_i1 * 48;
                  tensorforge::intel_esimd::simd<double, 16> v2991_data(ir0.template select<16, 1>((v2988_a + v2989_a)));
                  v2991_data.copy_to(glb_m0 + ((v2988_a + v2989_a)));
                }
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 12; i += 2) {
              tensorforge::intel_esimd::simd<double, 32> v3004_ld;
              v3004_ld.copy_from(wrap_glb_m2 + (0 + 0 + 2 * 0 + i * 16));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + i * 16), v3004_ld);
            }
            v2999_flagWord = v3000_flagWordNext;
            v3000_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


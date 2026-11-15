// === base name ===
kernel_a0466580e83479b4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a0466580e83479b4 = {{1, 16, 1}, 16, 40, 1, 16, 36864, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a0466580e83479b4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a0466580e83479b4(const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a0466580e83479b4(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4608 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a0466580e83479b4(const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a0466580e83479b4(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a0466580e83479b4(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_a0466580e83479b4_k;
inline void kernel_kernel_a0466580e83479b4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_a0466580e83479b4_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4608 * sizeof(double)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: amd_matrix_k4=1,lanes_per_mult=16
        // launch: 16 lanes (40 active) x 16 per block = block 1x16x1, 36864 B shared, occupancy grid
        // operands:
        //   m0 30×9(30×9) {0..30}×{0..9} strided
        //   m1 9×9(9×9) {0..9}×{0..9} strided
        //   m2 40×9(40×9) {0..40}×{0..9} strided
        //   m3 40×30(40×30) {0..40}×{0..30} none
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k]@{0..40}×{1..30} × t0[k,j]@{1..30}×{0..9}
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":40,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4608}],"shared_bytes":36864,"shared_elements":4608,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"X","bbox":[[0,0],[30,9]],"name":"m0","ordered":false,"parts":1,"shape":[30,9],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[40,9]],"name":"m2","ordered":false,"parts":1,"shape":[40,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,30]],"name":"m3","ordered":false,"parts":1,"shape":[40,30],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[30,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[30,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[30,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[30,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[40,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[40,30]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[40,30]},{"addressing":"strided","bbox":[[1,0],[30,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[30,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (288 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const double *const __restrict__ glb_m3 = &m3[0];
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<double> s1 = localShrMem0 + (0);
          tensorforge::intel_esimd::simd<double, 288> r0(0.0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v4469_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v4469_flagSecondAhead < numElements0) ? v4469_flagSecondAhead : v3_batchId1)];
          const double *const __restrict__ peel_glb_m0 = &m0[v3_batchId1 * 270 + 0 + m0_extraOffset];
          // r0 = load{g>r}(peel_glb_m0);
          #pragma unroll
          for (int32_t v4493_i0 = 0; v4493_i0 < 1; ++v4493_i0) {
            int32_t v4494_lead = v4493_i0 * 16;
            #pragma unroll
            for (int32_t v4495_i1 = 0; v4495_i1 < 9; ++v4495_i1) {
              tensorforge::intel_esimd::simd<double, 16> v4498_data;
              v4498_data.copy_from(peel_glb_m0 + ((v4494_lead + (v4495_i1 * 30))));
              r0.template select<16, 1>((v4494_lead + (v4495_i1 * 32))) = v4498_data;
            }
          }
          #pragma unroll
          for (int32_t v4501_i1 = 0; v4501_i1 < 9; ++v4501_i1) {
            tensorforge::intel_esimd::simd<double, 16> v4504_data(0.0);
            tensorforge::intel_esimd::simd<double, 14> v4504_data_part;
            v4504_data_part.copy_from(peel_glb_m0 + ((16_i32 + (v4501_i1 * 30))));
            v4504_data.template select<14, 1>(0) = v4504_data_part;
            r0.template select<16, 1>((16 + (v4501_i1 * 32))) = v4504_data;
          }
          const double *const __restrict__ peel_glb_m1 = &m1[v3_batchId1 * 81 + 0 + m1_extraOffset];
          // s0 = load{g>s}(peel_glb_m1[0, 1])
          tensorforge::intel_esimd::simd<double, 32> v4523_ld;
          v4523_ld.copy_from(peel_glb_m1 + (0 + 0 + 2 * 0 + 0));
          tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v4523_ld);
          tensorforge::intel_esimd::simd<double, 32> v4524_ld;
          v4524_ld.copy_from(peel_glb_m1 + (0 + 0 + 2 * 0 + 32));
          tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v4524_ld);
          tensorforge::intel_esimd::simd<double, 16> v4525_ld;
          v4525_ld.copy_from(peel_glb_m1 + (0 + 0 + 1 * 0 + 64));
          tensorforge::slmStore<double, 16>(s0 + (0 + 0 + 1 * 0 + 64), v4525_ld);
          double v4526_ld = peel_glb_m1[0 + 0 + 1 * 0 + 80];
          s0[0 + 0 + 1 * 0 + 80] = v4526_ld;
          uint32_t v4473_flagWord(flagWordFirst);
          uint32_t v4474_flagWordNext(flagWordSecond);
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const double *const __restrict__ wrap_glb_m0 = &m0[v13_batchId1 * 270 + 0 + m0_extraOffset];
            const double *const __restrict__ wrap_glb_m1 = &m1[v13_batchId1 * 81 + 0 + m1_extraOffset];
            size_t v4533_flagAheadAhead = v13_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v4533_flagAheadAhead < numElements0) ? v4533_flagAheadAhead : v13_batchId1)];
            const bool allowed = static_cast<bool>(v4473_flagWord);
            if (allowed) {
              double *const __restrict__ glb_m2 = &m2[v10_batchId0 * 360 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 288> r1(0.0);
              // r1 = +(r0 * s0) + None
              // [(0, 30), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<double, 16> v45_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 88> s0_w0 = tensorforge::slmLoad<double, 88>(s0 + 0);
              double v46_data = s0_w0[0];
              double v51_data = s0_w0[9];
              double v56_data = s0_w0[18];
              double v61_data = s0_w0[27];
              double v66_data = s0_w0[36];
              double v71_data = s0_w0[45];
              double v76_data = s0_w0[54];
              double v81_data = s0_w0[63];
              double v86_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v90_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v135_data(r0.template select<16, 1>(32));
              double v136_data = s0_w0[1];
              double v141_data = s0_w0[10];
              double v146_data = s0_w0[19];
              double v151_data = s0_w0[28];
              double v156_data = s0_w0[37];
              double v161_data = s0_w0[46];
              double v166_data = s0_w0[55];
              double v171_data = s0_w0[64];
              double v176_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v180_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<double, 16> v225_data(r0.template select<16, 1>(64));
              double v226_data = s0_w0[2];
              double v231_data = s0_w0[11];
              double v236_data = s0_w0[20];
              double v241_data = s0_w0[29];
              double v246_data = s0_w0[38];
              double v251_data = s0_w0[47];
              double v256_data = s0_w0[56];
              double v261_data = s0_w0[65];
              double v266_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v270_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<double, 16> v315_data(r0.template select<16, 1>(96));
              double v316_data = s0_w0[3];
              double v321_data = s0_w0[12];
              double v326_data = s0_w0[21];
              double v331_data = s0_w0[30];
              double v336_data = s0_w0[39];
              double v341_data = s0_w0[48];
              double v346_data = s0_w0[57];
              double v351_data = s0_w0[66];
              double v356_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v360_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<double, 16> v405_data(r0.template select<16, 1>(128));
              double v406_data = s0_w0[4];
              double v411_data = s0_w0[13];
              double v416_data = s0_w0[22];
              double v421_data = s0_w0[31];
              double v426_data = s0_w0[40];
              double v431_data = s0_w0[49];
              double v436_data = s0_w0[58];
              double v441_data = s0_w0[67];
              double v446_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v450_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<double, 16> v495_data(r0.template select<16, 1>(160));
              double v496_data = s0_w0[5];
              double v501_data = s0_w0[14];
              double v506_data = s0_w0[23];
              double v511_data = s0_w0[32];
              double v516_data = s0_w0[41];
              double v521_data = s0_w0[50];
              double v526_data = s0_w0[59];
              double v531_data = s0_w0[68];
              double v536_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v540_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<double, 16> v585_data(r0.template select<16, 1>(192));
              double v586_data = s0_w0[6];
              double v591_data = s0_w0[15];
              double v596_data = s0_w0[24];
              double v601_data = s0_w0[33];
              double v606_data = s0_w0[42];
              double v611_data = s0_w0[51];
              double v616_data = s0_w0[60];
              double v621_data = s0_w0[69];
              double v626_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v630_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<double, 16> v675_data(r0.template select<16, 1>(224));
              double v676_data = s0_w0[7];
              double v681_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v684_p = (((((((v45_data * v51_data) + (v135_data * v141_data)) + (v225_data * v231_data)) + (v315_data * v321_data)) + (v405_data * v411_data)) + (v495_data * v501_data)) + (v585_data * v591_data)) + (v675_data * v681_data);
              double v686_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v689_p = (((((((v45_data * v56_data) + (v135_data * v146_data)) + (v225_data * v236_data)) + (v315_data * v326_data)) + (v405_data * v416_data)) + (v495_data * v506_data)) + (v585_data * v596_data)) + (v675_data * v686_data);
              double v691_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v694_p = (((((((v45_data * v61_data) + (v135_data * v151_data)) + (v225_data * v241_data)) + (v315_data * v331_data)) + (v405_data * v421_data)) + (v495_data * v511_data)) + (v585_data * v601_data)) + (v675_data * v691_data);
              double v696_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v699_p = (((((((v45_data * v66_data) + (v135_data * v156_data)) + (v225_data * v246_data)) + (v315_data * v336_data)) + (v405_data * v426_data)) + (v495_data * v516_data)) + (v585_data * v606_data)) + (v675_data * v696_data);
              double v701_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v704_p = (((((((v45_data * v71_data) + (v135_data * v161_data)) + (v225_data * v251_data)) + (v315_data * v341_data)) + (v405_data * v431_data)) + (v495_data * v521_data)) + (v585_data * v611_data)) + (v675_data * v701_data);
              double v706_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v709_p = (((((((v45_data * v76_data) + (v135_data * v166_data)) + (v225_data * v256_data)) + (v315_data * v346_data)) + (v405_data * v436_data)) + (v495_data * v526_data)) + (v585_data * v616_data)) + (v675_data * v706_data);
              double v711_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v714_p = (((((((v45_data * v81_data) + (v135_data * v171_data)) + (v225_data * v261_data)) + (v315_data * v351_data)) + (v405_data * v441_data)) + (v495_data * v531_data)) + (v585_data * v621_data)) + (v675_data * v711_data);
              double v716_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v719_p = (((((((v45_data * v86_data) + (v135_data * v176_data)) + (v225_data * v266_data)) + (v315_data * v356_data)) + (v405_data * v446_data)) + (v495_data * v536_data)) + (v585_data * v626_data)) + (v675_data * v716_data);
              tensorforge::intel_esimd::simd<double, 16> v720_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<double, 16> v724_p = (((((((v90_data * v46_data) + (v180_data * v136_data)) + (v270_data * v226_data)) + (v360_data * v316_data)) + (v450_data * v406_data)) + (v540_data * v496_data)) + (v630_data * v586_data)) + (v720_data * v676_data);
              tensorforge::intel_esimd::simd<double, 16> v729_p = (((((((v90_data * v51_data) + (v180_data * v141_data)) + (v270_data * v231_data)) + (v360_data * v321_data)) + (v450_data * v411_data)) + (v540_data * v501_data)) + (v630_data * v591_data)) + (v720_data * v681_data);
              tensorforge::intel_esimd::simd<double, 16> v734_p = (((((((v90_data * v56_data) + (v180_data * v146_data)) + (v270_data * v236_data)) + (v360_data * v326_data)) + (v450_data * v416_data)) + (v540_data * v506_data)) + (v630_data * v596_data)) + (v720_data * v686_data);
              tensorforge::intel_esimd::simd<double, 16> v739_p = (((((((v90_data * v61_data) + (v180_data * v151_data)) + (v270_data * v241_data)) + (v360_data * v331_data)) + (v450_data * v421_data)) + (v540_data * v511_data)) + (v630_data * v601_data)) + (v720_data * v691_data);
              tensorforge::intel_esimd::simd<double, 16> v744_p = (((((((v90_data * v66_data) + (v180_data * v156_data)) + (v270_data * v246_data)) + (v360_data * v336_data)) + (v450_data * v426_data)) + (v540_data * v516_data)) + (v630_data * v606_data)) + (v720_data * v696_data);
              tensorforge::intel_esimd::simd<double, 16> v749_p = (((((((v90_data * v71_data) + (v180_data * v161_data)) + (v270_data * v251_data)) + (v360_data * v341_data)) + (v450_data * v431_data)) + (v540_data * v521_data)) + (v630_data * v611_data)) + (v720_data * v701_data);
              tensorforge::intel_esimd::simd<double, 16> v754_p = (((((((v90_data * v76_data) + (v180_data * v166_data)) + (v270_data * v256_data)) + (v360_data * v346_data)) + (v450_data * v436_data)) + (v540_data * v526_data)) + (v630_data * v616_data)) + (v720_data * v706_data);
              tensorforge::intel_esimd::simd<double, 16> v759_p = (((((((v90_data * v81_data) + (v180_data * v171_data)) + (v270_data * v261_data)) + (v360_data * v351_data)) + (v450_data * v441_data)) + (v540_data * v531_data)) + (v630_data * v621_data)) + (v720_data * v711_data);
              tensorforge::intel_esimd::simd<double, 16> v764_p = (((((((v90_data * v86_data) + (v180_data * v176_data)) + (v270_data * v266_data)) + (v360_data * v356_data)) + (v450_data * v446_data)) + (v540_data * v536_data)) + (v630_data * v626_data)) + (v720_data * v716_data);
              tensorforge::intel_esimd::simd<double, 16> v765_data(r0.template select<16, 1>(256));
              double v766_data = s0_w0[8];
              r1.template select<16, 1>(0) = (((((((((v45_data * v46_data) + (v135_data * v136_data)) + (v225_data * v226_data)) + (v315_data * v316_data)) + (v405_data * v406_data)) + (v495_data * v496_data)) + (v585_data * v586_data)) + (v675_data * v676_data)) + (v765_data * v766_data));
              double v771_data = s0_w0[17];
              r1.template select<16, 1>(32) = (v684_p + (v765_data * v771_data));
              double v776_data = s0_w0[26];
              r1.template select<16, 1>(64) = (v689_p + (v765_data * v776_data));
              double v781_data = s0_w0[35];
              r1.template select<16, 1>(96) = (v694_p + (v765_data * v781_data));
              double v786_data = s0_w0[44];
              r1.template select<16, 1>(128) = (v699_p + (v765_data * v786_data));
              double v791_data = s0_w0[53];
              r1.template select<16, 1>(160) = (v704_p + (v765_data * v791_data));
              double v796_data = s0_w0[62];
              r1.template select<16, 1>(192) = (v709_p + (v765_data * v796_data));
              double v801_data = s0_w0[71];
              r1.template select<16, 1>(224) = (v714_p + (v765_data * v801_data));
              double v806_data = s0_w0[80];
              r1.template select<16, 1>(256) = (v719_p + (v765_data * v806_data));
              tensorforge::intel_esimd::simd<double, 16> v810_data(r0.template select<16, 1>(272));
              r1.template select<16, 1>(16) = (v724_p + (v810_data * v766_data));
              r1.template select<16, 1>(48) = (v729_p + (v810_data * v771_data));
              r1.template select<16, 1>(80) = (v734_p + (v810_data * v776_data));
              r1.template select<16, 1>(112) = (v739_p + (v810_data * v781_data));
              r1.template select<16, 1>(144) = (v744_p + (v810_data * v786_data));
              r1.template select<16, 1>(176) = (v749_p + (v810_data * v791_data));
              r1.template select<16, 1>(208) = (v754_p + (v810_data * v796_data));
              r1.template select<16, 1>(240) = (v759_p + (v810_data * v801_data));
              r1.template select<16, 1>(272) = (v764_p + (v810_data * v806_data));
              // r1 = 0 past (0, 30)
              tensorforge::intel_esimd::simd<int32_t, 16> v855_lead = tensorforge::intel_esimd::simd<int32_t, 16>(0, 1);
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v873_i0 = 0; v873_i0 < 1; ++v873_i0) {
                int32_t v875_a = v873_i0 * 16;
                #pragma unroll
                for (int32_t v874_i1 = 0; v874_i1 < 9; ++v874_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v878_data(r1.template select<16, 1>((v875_a + (v874_i1 * 32))));
                  tensorforge::slmStore<double, 16>(s1 + ((v875_a + (v874_i1 * 30))), v878_data);
                }
              }
              #pragma unroll
              for (int32_t v882_i1 = 0; v882_i1 < 9; ++v882_i1) {
                tensorforge::intel_esimd::simd<double, 14> v885_data(r1.template select<14, 1>((16 + (v882_i1 * 32))));
                tensorforge::slmStore<double, 14>(s1 + ((16_i32 + (v882_i1 * 30))), v885_data);
              }
              tensorforge::intel_esimd::simd<double, 432> r2(0.0);
              // ir2 = +(glb_m3 * s1)
              // [(0, 40), (0, 9)] [(1, 30)]
              tensorforge::intel_esimd::simd<double, 432> ir2(0.0);
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run0;
              glb_m3_run0.copy_from(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<double, 16> v896_data(glb_m3_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 271> s1_w1 = tensorforge::slmLoad<double, 271>(s1 + 1);
              double v897_data = s1_w1[0];
              double v902_data = s1_w1[30];
              double v907_data = s1_w1[60];
              double v912_data = s1_w1[90];
              double v917_data = s1_w1[120];
              double v922_data = s1_w1[150];
              double v927_data = s1_w1[180];
              double v932_data = s1_w1[210];
              double v937_data = s1_w1[240];
              tensorforge::intel_esimd::simd<double, 16> v944_data(glb_m3_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v991_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v991_data_part;
              v991_data_part.copy_from(glb_m3 + (72_i32));
              v991_data.template select<8, 1>(0) = v991_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run1;
              glb_m3_run1.copy_from(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<double, 16> v1037_data(glb_m3_run1.template select<16, 1>(0));
              double v1038_data = s1_w1[1];
              double v1043_data = s1_w1[31];
              double v1048_data = s1_w1[61];
              double v1053_data = s1_w1[91];
              double v1058_data = s1_w1[121];
              double v1063_data = s1_w1[151];
              double v1068_data = s1_w1[181];
              double v1073_data = s1_w1[211];
              double v1078_data = s1_w1[241];
              tensorforge::intel_esimd::simd<double, 16> v1083_data(glb_m3_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v1129_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v1129_data_part;
              v1129_data_part.copy_from(glb_m3 + (112_i32));
              v1129_data.template select<8, 1>(0) = v1129_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run2;
              glb_m3_run2.copy_from(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<double, 16> v1175_data(glb_m3_run2.template select<16, 1>(0));
              double v1176_data = s1_w1[2];
              double v1181_data = s1_w1[32];
              double v1186_data = s1_w1[62];
              double v1191_data = s1_w1[92];
              double v1196_data = s1_w1[122];
              double v1201_data = s1_w1[152];
              double v1206_data = s1_w1[182];
              double v1211_data = s1_w1[212];
              double v1216_data = s1_w1[242];
              tensorforge::intel_esimd::simd<double, 16> v1221_data(glb_m3_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v1267_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v1267_data_part;
              v1267_data_part.copy_from(glb_m3 + (152_i32));
              v1267_data.template select<8, 1>(0) = v1267_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run3;
              glb_m3_run3.copy_from(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<double, 16> v1313_data(glb_m3_run3.template select<16, 1>(0));
              double v1314_data = s1_w1[3];
              double v1319_data = s1_w1[33];
              double v1324_data = s1_w1[63];
              double v1329_data = s1_w1[93];
              double v1334_data = s1_w1[123];
              double v1339_data = s1_w1[153];
              double v1344_data = s1_w1[183];
              double v1349_data = s1_w1[213];
              double v1354_data = s1_w1[243];
              tensorforge::intel_esimd::simd<double, 16> v1359_data(glb_m3_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v1405_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v1405_data_part;
              v1405_data_part.copy_from(glb_m3 + (192_i32));
              v1405_data.template select<8, 1>(0) = v1405_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run4;
              glb_m3_run4.copy_from(glb_m3 + (200_i32));
              tensorforge::intel_esimd::simd<double, 16> v1451_data(glb_m3_run4.template select<16, 1>(0));
              double v1452_data = s1_w1[4];
              double v1457_data = s1_w1[34];
              double v1462_data = s1_w1[64];
              double v1467_data = s1_w1[94];
              double v1472_data = s1_w1[124];
              double v1477_data = s1_w1[154];
              double v1482_data = s1_w1[184];
              double v1487_data = s1_w1[214];
              double v1492_data = s1_w1[244];
              tensorforge::intel_esimd::simd<double, 16> v1497_data(glb_m3_run4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v1543_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v1543_data_part;
              v1543_data_part.copy_from(glb_m3 + (232_i32));
              v1543_data.template select<8, 1>(0) = v1543_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run5;
              glb_m3_run5.copy_from(glb_m3 + (240_i32));
              tensorforge::intel_esimd::simd<double, 16> v1589_data(glb_m3_run5.template select<16, 1>(0));
              double v1590_data = s1_w1[5];
              double v1595_data = s1_w1[35];
              double v1600_data = s1_w1[65];
              double v1605_data = s1_w1[95];
              double v1610_data = s1_w1[125];
              double v1615_data = s1_w1[155];
              double v1620_data = s1_w1[185];
              double v1625_data = s1_w1[215];
              double v1630_data = s1_w1[245];
              tensorforge::intel_esimd::simd<double, 16> v1635_data(glb_m3_run5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v1681_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v1681_data_part;
              v1681_data_part.copy_from(glb_m3 + (272_i32));
              v1681_data.template select<8, 1>(0) = v1681_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run6;
              glb_m3_run6.copy_from(glb_m3 + (280_i32));
              tensorforge::intel_esimd::simd<double, 16> v1727_data(glb_m3_run6.template select<16, 1>(0));
              double v1728_data = s1_w1[6];
              double v1733_data = s1_w1[36];
              double v1738_data = s1_w1[66];
              double v1743_data = s1_w1[96];
              double v1748_data = s1_w1[126];
              double v1753_data = s1_w1[156];
              double v1758_data = s1_w1[186];
              double v1763_data = s1_w1[216];
              double v1768_data = s1_w1[246];
              tensorforge::intel_esimd::simd<double, 16> v1773_data(glb_m3_run6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v1819_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v1819_data_part;
              v1819_data_part.copy_from(glb_m3 + (312_i32));
              v1819_data.template select<8, 1>(0) = v1819_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run7;
              glb_m3_run7.copy_from(glb_m3 + (320_i32));
              tensorforge::intel_esimd::simd<double, 16> v1865_data(glb_m3_run7.template select<16, 1>(0));
              double v1866_data = s1_w1[7];
              double v1871_data = s1_w1[37];
              double v1876_data = s1_w1[67];
              double v1881_data = s1_w1[97];
              double v1886_data = s1_w1[127];
              double v1891_data = s1_w1[157];
              double v1896_data = s1_w1[187];
              double v1901_data = s1_w1[217];
              double v1906_data = s1_w1[247];
              tensorforge::intel_esimd::simd<double, 16> v1911_data(glb_m3_run7.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v1957_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v1957_data_part;
              v1957_data_part.copy_from(glb_m3 + (352_i32));
              v1957_data.template select<8, 1>(0) = v1957_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run8;
              glb_m3_run8.copy_from(glb_m3 + (360_i32));
              tensorforge::intel_esimd::simd<double, 16> v2003_data(glb_m3_run8.template select<16, 1>(0));
              double v2004_data = s1_w1[8];
              double v2009_data = s1_w1[38];
              double v2014_data = s1_w1[68];
              double v2019_data = s1_w1[98];
              double v2024_data = s1_w1[128];
              double v2029_data = s1_w1[158];
              double v2034_data = s1_w1[188];
              double v2039_data = s1_w1[218];
              double v2044_data = s1_w1[248];
              tensorforge::intel_esimd::simd<double, 16> v2049_data(glb_m3_run8.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v2095_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v2095_data_part;
              v2095_data_part.copy_from(glb_m3 + (392_i32));
              v2095_data.template select<8, 1>(0) = v2095_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run9;
              glb_m3_run9.copy_from(glb_m3 + (400_i32));
              tensorforge::intel_esimd::simd<double, 16> v2141_data(glb_m3_run9.template select<16, 1>(0));
              double v2142_data = s1_w1[9];
              double v2147_data = s1_w1[39];
              double v2152_data = s1_w1[69];
              double v2157_data = s1_w1[99];
              double v2162_data = s1_w1[129];
              double v2167_data = s1_w1[159];
              double v2172_data = s1_w1[189];
              double v2177_data = s1_w1[219];
              double v2182_data = s1_w1[249];
              tensorforge::intel_esimd::simd<double, 16> v2187_data(glb_m3_run9.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v2233_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v2233_data_part;
              v2233_data_part.copy_from(glb_m3 + (432_i32));
              v2233_data.template select<8, 1>(0) = v2233_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run10;
              glb_m3_run10.copy_from(glb_m3 + (440_i32));
              tensorforge::intel_esimd::simd<double, 16> v2279_data(glb_m3_run10.template select<16, 1>(0));
              double v2280_data = s1_w1[10];
              double v2285_data = s1_w1[40];
              double v2290_data = s1_w1[70];
              double v2295_data = s1_w1[100];
              double v2300_data = s1_w1[130];
              double v2305_data = s1_w1[160];
              double v2310_data = s1_w1[190];
              double v2315_data = s1_w1[220];
              double v2320_data = s1_w1[250];
              tensorforge::intel_esimd::simd<double, 16> v2325_data(glb_m3_run10.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v2371_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v2371_data_part;
              v2371_data_part.copy_from(glb_m3 + (472_i32));
              v2371_data.template select<8, 1>(0) = v2371_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run11;
              glb_m3_run11.copy_from(glb_m3 + (480_i32));
              tensorforge::intel_esimd::simd<double, 16> v2417_data(glb_m3_run11.template select<16, 1>(0));
              double v2418_data = s1_w1[11];
              double v2423_data = s1_w1[41];
              double v2428_data = s1_w1[71];
              double v2433_data = s1_w1[101];
              double v2438_data = s1_w1[131];
              double v2443_data = s1_w1[161];
              double v2448_data = s1_w1[191];
              double v2453_data = s1_w1[221];
              double v2458_data = s1_w1[251];
              tensorforge::intel_esimd::simd<double, 16> v2463_data(glb_m3_run11.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v2509_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v2509_data_part;
              v2509_data_part.copy_from(glb_m3 + (512_i32));
              v2509_data.template select<8, 1>(0) = v2509_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run12;
              glb_m3_run12.copy_from(glb_m3 + (520_i32));
              tensorforge::intel_esimd::simd<double, 16> v2555_data(glb_m3_run12.template select<16, 1>(0));
              double v2556_data = s1_w1[12];
              double v2561_data = s1_w1[42];
              double v2566_data = s1_w1[72];
              double v2571_data = s1_w1[102];
              double v2576_data = s1_w1[132];
              double v2581_data = s1_w1[162];
              double v2586_data = s1_w1[192];
              double v2591_data = s1_w1[222];
              double v2596_data = s1_w1[252];
              tensorforge::intel_esimd::simd<double, 16> v2601_data(glb_m3_run12.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v2647_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v2647_data_part;
              v2647_data_part.copy_from(glb_m3 + (552_i32));
              v2647_data.template select<8, 1>(0) = v2647_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run13;
              glb_m3_run13.copy_from(glb_m3 + (560_i32));
              tensorforge::intel_esimd::simd<double, 16> v2693_data(glb_m3_run13.template select<16, 1>(0));
              double v2694_data = s1_w1[13];
              double v2699_data = s1_w1[43];
              double v2704_data = s1_w1[73];
              double v2709_data = s1_w1[103];
              double v2714_data = s1_w1[133];
              double v2719_data = s1_w1[163];
              double v2724_data = s1_w1[193];
              double v2729_data = s1_w1[223];
              double v2734_data = s1_w1[253];
              tensorforge::intel_esimd::simd<double, 16> v2739_data(glb_m3_run13.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v2785_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v2785_data_part;
              v2785_data_part.copy_from(glb_m3 + (592_i32));
              v2785_data.template select<8, 1>(0) = v2785_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run14;
              glb_m3_run14.copy_from(glb_m3 + (600_i32));
              tensorforge::intel_esimd::simd<double, 16> v2831_data(glb_m3_run14.template select<16, 1>(0));
              double v2832_data = s1_w1[14];
              double v2837_data = s1_w1[44];
              double v2842_data = s1_w1[74];
              double v2847_data = s1_w1[104];
              double v2852_data = s1_w1[134];
              double v2857_data = s1_w1[164];
              double v2862_data = s1_w1[194];
              double v2867_data = s1_w1[224];
              double v2872_data = s1_w1[254];
              tensorforge::intel_esimd::simd<double, 16> v2877_data(glb_m3_run14.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v2923_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v2923_data_part;
              v2923_data_part.copy_from(glb_m3 + (632_i32));
              v2923_data.template select<8, 1>(0) = v2923_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run15;
              glb_m3_run15.copy_from(glb_m3 + (640_i32));
              tensorforge::intel_esimd::simd<double, 16> v2969_data(glb_m3_run15.template select<16, 1>(0));
              double v2970_data = s1_w1[15];
              double v2975_data = s1_w1[45];
              double v2980_data = s1_w1[75];
              double v2985_data = s1_w1[105];
              double v2990_data = s1_w1[135];
              double v2995_data = s1_w1[165];
              double v3000_data = s1_w1[195];
              double v3005_data = s1_w1[225];
              double v3010_data = s1_w1[255];
              tensorforge::intel_esimd::simd<double, 16> v3015_data(glb_m3_run15.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v3061_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v3061_data_part;
              v3061_data_part.copy_from(glb_m3 + (672_i32));
              v3061_data.template select<8, 1>(0) = v3061_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run16;
              glb_m3_run16.copy_from(glb_m3 + (680_i32));
              tensorforge::intel_esimd::simd<double, 16> v3107_data(glb_m3_run16.template select<16, 1>(0));
              double v3108_data = s1_w1[16];
              double v3113_data = s1_w1[46];
              double v3118_data = s1_w1[76];
              double v3123_data = s1_w1[106];
              double v3128_data = s1_w1[136];
              double v3133_data = s1_w1[166];
              double v3138_data = s1_w1[196];
              double v3143_data = s1_w1[226];
              double v3148_data = s1_w1[256];
              tensorforge::intel_esimd::simd<double, 16> v3153_data(glb_m3_run16.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v3199_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v3199_data_part;
              v3199_data_part.copy_from(glb_m3 + (712_i32));
              v3199_data.template select<8, 1>(0) = v3199_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run17;
              glb_m3_run17.copy_from(glb_m3 + (720_i32));
              tensorforge::intel_esimd::simd<double, 16> v3245_data(glb_m3_run17.template select<16, 1>(0));
              double v3246_data = s1_w1[17];
              double v3251_data = s1_w1[47];
              tensorforge::intel_esimd::simd<double, 16> v3254_p = (((((((((((((((((v896_data * v902_data) + (v1037_data * v1043_data)) + (v1175_data * v1181_data)) + (v1313_data * v1319_data)) + (v1451_data * v1457_data)) + (v1589_data * v1595_data)) + (v1727_data * v1733_data)) + (v1865_data * v1871_data)) + (v2003_data * v2009_data)) + (v2141_data * v2147_data)) + (v2279_data * v2285_data)) + (v2417_data * v2423_data)) + (v2555_data * v2561_data)) + (v2693_data * v2699_data)) + (v2831_data * v2837_data)) + (v2969_data * v2975_data)) + (v3107_data * v3113_data)) + (v3245_data * v3251_data);
              double v3256_data = s1_w1[77];
              tensorforge::intel_esimd::simd<double, 16> v3259_p = (((((((((((((((((v896_data * v907_data) + (v1037_data * v1048_data)) + (v1175_data * v1186_data)) + (v1313_data * v1324_data)) + (v1451_data * v1462_data)) + (v1589_data * v1600_data)) + (v1727_data * v1738_data)) + (v1865_data * v1876_data)) + (v2003_data * v2014_data)) + (v2141_data * v2152_data)) + (v2279_data * v2290_data)) + (v2417_data * v2428_data)) + (v2555_data * v2566_data)) + (v2693_data * v2704_data)) + (v2831_data * v2842_data)) + (v2969_data * v2980_data)) + (v3107_data * v3118_data)) + (v3245_data * v3256_data);
              double v3261_data = s1_w1[107];
              tensorforge::intel_esimd::simd<double, 16> v3264_p = (((((((((((((((((v896_data * v912_data) + (v1037_data * v1053_data)) + (v1175_data * v1191_data)) + (v1313_data * v1329_data)) + (v1451_data * v1467_data)) + (v1589_data * v1605_data)) + (v1727_data * v1743_data)) + (v1865_data * v1881_data)) + (v2003_data * v2019_data)) + (v2141_data * v2157_data)) + (v2279_data * v2295_data)) + (v2417_data * v2433_data)) + (v2555_data * v2571_data)) + (v2693_data * v2709_data)) + (v2831_data * v2847_data)) + (v2969_data * v2985_data)) + (v3107_data * v3123_data)) + (v3245_data * v3261_data);
              double v3266_data = s1_w1[137];
              tensorforge::intel_esimd::simd<double, 16> v3269_p = (((((((((((((((((v896_data * v917_data) + (v1037_data * v1058_data)) + (v1175_data * v1196_data)) + (v1313_data * v1334_data)) + (v1451_data * v1472_data)) + (v1589_data * v1610_data)) + (v1727_data * v1748_data)) + (v1865_data * v1886_data)) + (v2003_data * v2024_data)) + (v2141_data * v2162_data)) + (v2279_data * v2300_data)) + (v2417_data * v2438_data)) + (v2555_data * v2576_data)) + (v2693_data * v2714_data)) + (v2831_data * v2852_data)) + (v2969_data * v2990_data)) + (v3107_data * v3128_data)) + (v3245_data * v3266_data);
              double v3271_data = s1_w1[167];
              tensorforge::intel_esimd::simd<double, 16> v3274_p = (((((((((((((((((v896_data * v922_data) + (v1037_data * v1063_data)) + (v1175_data * v1201_data)) + (v1313_data * v1339_data)) + (v1451_data * v1477_data)) + (v1589_data * v1615_data)) + (v1727_data * v1753_data)) + (v1865_data * v1891_data)) + (v2003_data * v2029_data)) + (v2141_data * v2167_data)) + (v2279_data * v2305_data)) + (v2417_data * v2443_data)) + (v2555_data * v2581_data)) + (v2693_data * v2719_data)) + (v2831_data * v2857_data)) + (v2969_data * v2995_data)) + (v3107_data * v3133_data)) + (v3245_data * v3271_data);
              double v3276_data = s1_w1[197];
              tensorforge::intel_esimd::simd<double, 16> v3279_p = (((((((((((((((((v896_data * v927_data) + (v1037_data * v1068_data)) + (v1175_data * v1206_data)) + (v1313_data * v1344_data)) + (v1451_data * v1482_data)) + (v1589_data * v1620_data)) + (v1727_data * v1758_data)) + (v1865_data * v1896_data)) + (v2003_data * v2034_data)) + (v2141_data * v2172_data)) + (v2279_data * v2310_data)) + (v2417_data * v2448_data)) + (v2555_data * v2586_data)) + (v2693_data * v2724_data)) + (v2831_data * v2862_data)) + (v2969_data * v3000_data)) + (v3107_data * v3138_data)) + (v3245_data * v3276_data);
              double v3281_data = s1_w1[227];
              tensorforge::intel_esimd::simd<double, 16> v3284_p = (((((((((((((((((v896_data * v932_data) + (v1037_data * v1073_data)) + (v1175_data * v1211_data)) + (v1313_data * v1349_data)) + (v1451_data * v1487_data)) + (v1589_data * v1625_data)) + (v1727_data * v1763_data)) + (v1865_data * v1901_data)) + (v2003_data * v2039_data)) + (v2141_data * v2177_data)) + (v2279_data * v2315_data)) + (v2417_data * v2453_data)) + (v2555_data * v2591_data)) + (v2693_data * v2729_data)) + (v2831_data * v2867_data)) + (v2969_data * v3005_data)) + (v3107_data * v3143_data)) + (v3245_data * v3281_data);
              double v3286_data = s1_w1[257];
              tensorforge::intel_esimd::simd<double, 16> v3289_p = (((((((((((((((((v896_data * v937_data) + (v1037_data * v1078_data)) + (v1175_data * v1216_data)) + (v1313_data * v1354_data)) + (v1451_data * v1492_data)) + (v1589_data * v1630_data)) + (v1727_data * v1768_data)) + (v1865_data * v1906_data)) + (v2003_data * v2044_data)) + (v2141_data * v2182_data)) + (v2279_data * v2320_data)) + (v2417_data * v2458_data)) + (v2555_data * v2596_data)) + (v2693_data * v2734_data)) + (v2831_data * v2872_data)) + (v2969_data * v3010_data)) + (v3107_data * v3148_data)) + (v3245_data * v3286_data);
              tensorforge::intel_esimd::simd<double, 16> v3291_data(glb_m3_run17.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v3295_p = (((((((((((((((((v944_data * v897_data) + (v1083_data * v1038_data)) + (v1221_data * v1176_data)) + (v1359_data * v1314_data)) + (v1497_data * v1452_data)) + (v1635_data * v1590_data)) + (v1773_data * v1728_data)) + (v1911_data * v1866_data)) + (v2049_data * v2004_data)) + (v2187_data * v2142_data)) + (v2325_data * v2280_data)) + (v2463_data * v2418_data)) + (v2601_data * v2556_data)) + (v2739_data * v2694_data)) + (v2877_data * v2832_data)) + (v3015_data * v2970_data)) + (v3153_data * v3108_data)) + (v3291_data * v3246_data);
              tensorforge::intel_esimd::simd<double, 16> v3300_p = (((((((((((((((((v944_data * v902_data) + (v1083_data * v1043_data)) + (v1221_data * v1181_data)) + (v1359_data * v1319_data)) + (v1497_data * v1457_data)) + (v1635_data * v1595_data)) + (v1773_data * v1733_data)) + (v1911_data * v1871_data)) + (v2049_data * v2009_data)) + (v2187_data * v2147_data)) + (v2325_data * v2285_data)) + (v2463_data * v2423_data)) + (v2601_data * v2561_data)) + (v2739_data * v2699_data)) + (v2877_data * v2837_data)) + (v3015_data * v2975_data)) + (v3153_data * v3113_data)) + (v3291_data * v3251_data);
              tensorforge::intel_esimd::simd<double, 16> v3305_p = (((((((((((((((((v944_data * v907_data) + (v1083_data * v1048_data)) + (v1221_data * v1186_data)) + (v1359_data * v1324_data)) + (v1497_data * v1462_data)) + (v1635_data * v1600_data)) + (v1773_data * v1738_data)) + (v1911_data * v1876_data)) + (v2049_data * v2014_data)) + (v2187_data * v2152_data)) + (v2325_data * v2290_data)) + (v2463_data * v2428_data)) + (v2601_data * v2566_data)) + (v2739_data * v2704_data)) + (v2877_data * v2842_data)) + (v3015_data * v2980_data)) + (v3153_data * v3118_data)) + (v3291_data * v3256_data);
              tensorforge::intel_esimd::simd<double, 16> v3310_p = (((((((((((((((((v944_data * v912_data) + (v1083_data * v1053_data)) + (v1221_data * v1191_data)) + (v1359_data * v1329_data)) + (v1497_data * v1467_data)) + (v1635_data * v1605_data)) + (v1773_data * v1743_data)) + (v1911_data * v1881_data)) + (v2049_data * v2019_data)) + (v2187_data * v2157_data)) + (v2325_data * v2295_data)) + (v2463_data * v2433_data)) + (v2601_data * v2571_data)) + (v2739_data * v2709_data)) + (v2877_data * v2847_data)) + (v3015_data * v2985_data)) + (v3153_data * v3123_data)) + (v3291_data * v3261_data);
              tensorforge::intel_esimd::simd<double, 16> v3315_p = (((((((((((((((((v944_data * v917_data) + (v1083_data * v1058_data)) + (v1221_data * v1196_data)) + (v1359_data * v1334_data)) + (v1497_data * v1472_data)) + (v1635_data * v1610_data)) + (v1773_data * v1748_data)) + (v1911_data * v1886_data)) + (v2049_data * v2024_data)) + (v2187_data * v2162_data)) + (v2325_data * v2300_data)) + (v2463_data * v2438_data)) + (v2601_data * v2576_data)) + (v2739_data * v2714_data)) + (v2877_data * v2852_data)) + (v3015_data * v2990_data)) + (v3153_data * v3128_data)) + (v3291_data * v3266_data);
              tensorforge::intel_esimd::simd<double, 16> v3320_p = (((((((((((((((((v944_data * v922_data) + (v1083_data * v1063_data)) + (v1221_data * v1201_data)) + (v1359_data * v1339_data)) + (v1497_data * v1477_data)) + (v1635_data * v1615_data)) + (v1773_data * v1753_data)) + (v1911_data * v1891_data)) + (v2049_data * v2029_data)) + (v2187_data * v2167_data)) + (v2325_data * v2305_data)) + (v2463_data * v2443_data)) + (v2601_data * v2581_data)) + (v2739_data * v2719_data)) + (v2877_data * v2857_data)) + (v3015_data * v2995_data)) + (v3153_data * v3133_data)) + (v3291_data * v3271_data);
              tensorforge::intel_esimd::simd<double, 16> v3325_p = (((((((((((((((((v944_data * v927_data) + (v1083_data * v1068_data)) + (v1221_data * v1206_data)) + (v1359_data * v1344_data)) + (v1497_data * v1482_data)) + (v1635_data * v1620_data)) + (v1773_data * v1758_data)) + (v1911_data * v1896_data)) + (v2049_data * v2034_data)) + (v2187_data * v2172_data)) + (v2325_data * v2310_data)) + (v2463_data * v2448_data)) + (v2601_data * v2586_data)) + (v2739_data * v2724_data)) + (v2877_data * v2862_data)) + (v3015_data * v3000_data)) + (v3153_data * v3138_data)) + (v3291_data * v3276_data);
              tensorforge::intel_esimd::simd<double, 16> v3330_p = (((((((((((((((((v944_data * v932_data) + (v1083_data * v1073_data)) + (v1221_data * v1211_data)) + (v1359_data * v1349_data)) + (v1497_data * v1487_data)) + (v1635_data * v1625_data)) + (v1773_data * v1763_data)) + (v1911_data * v1901_data)) + (v2049_data * v2039_data)) + (v2187_data * v2177_data)) + (v2325_data * v2315_data)) + (v2463_data * v2453_data)) + (v2601_data * v2591_data)) + (v2739_data * v2729_data)) + (v2877_data * v2867_data)) + (v3015_data * v3005_data)) + (v3153_data * v3143_data)) + (v3291_data * v3281_data);
              tensorforge::intel_esimd::simd<double, 16> v3335_p = (((((((((((((((((v944_data * v937_data) + (v1083_data * v1078_data)) + (v1221_data * v1216_data)) + (v1359_data * v1354_data)) + (v1497_data * v1492_data)) + (v1635_data * v1630_data)) + (v1773_data * v1768_data)) + (v1911_data * v1906_data)) + (v2049_data * v2044_data)) + (v2187_data * v2182_data)) + (v2325_data * v2320_data)) + (v2463_data * v2458_data)) + (v2601_data * v2596_data)) + (v2739_data * v2734_data)) + (v2877_data * v2872_data)) + (v3015_data * v3010_data)) + (v3153_data * v3148_data)) + (v3291_data * v3286_data);
              tensorforge::intel_esimd::simd<double, 16> v3337_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v3337_data_part;
              v3337_data_part.copy_from(glb_m3 + (752_i32));
              v3337_data.template select<8, 1>(0) = v3337_data_part;
              tensorforge::intel_esimd::simd<double, 16> v3341_p = (((((((((((((((((v991_data * v897_data) + (v1129_data * v1038_data)) + (v1267_data * v1176_data)) + (v1405_data * v1314_data)) + (v1543_data * v1452_data)) + (v1681_data * v1590_data)) + (v1819_data * v1728_data)) + (v1957_data * v1866_data)) + (v2095_data * v2004_data)) + (v2233_data * v2142_data)) + (v2371_data * v2280_data)) + (v2509_data * v2418_data)) + (v2647_data * v2556_data)) + (v2785_data * v2694_data)) + (v2923_data * v2832_data)) + (v3061_data * v2970_data)) + (v3199_data * v3108_data)) + (v3337_data * v3246_data);
              tensorforge::intel_esimd::simd<double, 16> v3346_p = (((((((((((((((((v991_data * v902_data) + (v1129_data * v1043_data)) + (v1267_data * v1181_data)) + (v1405_data * v1319_data)) + (v1543_data * v1457_data)) + (v1681_data * v1595_data)) + (v1819_data * v1733_data)) + (v1957_data * v1871_data)) + (v2095_data * v2009_data)) + (v2233_data * v2147_data)) + (v2371_data * v2285_data)) + (v2509_data * v2423_data)) + (v2647_data * v2561_data)) + (v2785_data * v2699_data)) + (v2923_data * v2837_data)) + (v3061_data * v2975_data)) + (v3199_data * v3113_data)) + (v3337_data * v3251_data);
              tensorforge::intel_esimd::simd<double, 16> v3351_p = (((((((((((((((((v991_data * v907_data) + (v1129_data * v1048_data)) + (v1267_data * v1186_data)) + (v1405_data * v1324_data)) + (v1543_data * v1462_data)) + (v1681_data * v1600_data)) + (v1819_data * v1738_data)) + (v1957_data * v1876_data)) + (v2095_data * v2014_data)) + (v2233_data * v2152_data)) + (v2371_data * v2290_data)) + (v2509_data * v2428_data)) + (v2647_data * v2566_data)) + (v2785_data * v2704_data)) + (v2923_data * v2842_data)) + (v3061_data * v2980_data)) + (v3199_data * v3118_data)) + (v3337_data * v3256_data);
              tensorforge::intel_esimd::simd<double, 16> v3356_p = (((((((((((((((((v991_data * v912_data) + (v1129_data * v1053_data)) + (v1267_data * v1191_data)) + (v1405_data * v1329_data)) + (v1543_data * v1467_data)) + (v1681_data * v1605_data)) + (v1819_data * v1743_data)) + (v1957_data * v1881_data)) + (v2095_data * v2019_data)) + (v2233_data * v2157_data)) + (v2371_data * v2295_data)) + (v2509_data * v2433_data)) + (v2647_data * v2571_data)) + (v2785_data * v2709_data)) + (v2923_data * v2847_data)) + (v3061_data * v2985_data)) + (v3199_data * v3123_data)) + (v3337_data * v3261_data);
              tensorforge::intel_esimd::simd<double, 16> v3361_p = (((((((((((((((((v991_data * v917_data) + (v1129_data * v1058_data)) + (v1267_data * v1196_data)) + (v1405_data * v1334_data)) + (v1543_data * v1472_data)) + (v1681_data * v1610_data)) + (v1819_data * v1748_data)) + (v1957_data * v1886_data)) + (v2095_data * v2024_data)) + (v2233_data * v2162_data)) + (v2371_data * v2300_data)) + (v2509_data * v2438_data)) + (v2647_data * v2576_data)) + (v2785_data * v2714_data)) + (v2923_data * v2852_data)) + (v3061_data * v2990_data)) + (v3199_data * v3128_data)) + (v3337_data * v3266_data);
              tensorforge::intel_esimd::simd<double, 16> v3366_p = (((((((((((((((((v991_data * v922_data) + (v1129_data * v1063_data)) + (v1267_data * v1201_data)) + (v1405_data * v1339_data)) + (v1543_data * v1477_data)) + (v1681_data * v1615_data)) + (v1819_data * v1753_data)) + (v1957_data * v1891_data)) + (v2095_data * v2029_data)) + (v2233_data * v2167_data)) + (v2371_data * v2305_data)) + (v2509_data * v2443_data)) + (v2647_data * v2581_data)) + (v2785_data * v2719_data)) + (v2923_data * v2857_data)) + (v3061_data * v2995_data)) + (v3199_data * v3133_data)) + (v3337_data * v3271_data);
              tensorforge::intel_esimd::simd<double, 16> v3371_p = (((((((((((((((((v991_data * v927_data) + (v1129_data * v1068_data)) + (v1267_data * v1206_data)) + (v1405_data * v1344_data)) + (v1543_data * v1482_data)) + (v1681_data * v1620_data)) + (v1819_data * v1758_data)) + (v1957_data * v1896_data)) + (v2095_data * v2034_data)) + (v2233_data * v2172_data)) + (v2371_data * v2310_data)) + (v2509_data * v2448_data)) + (v2647_data * v2586_data)) + (v2785_data * v2724_data)) + (v2923_data * v2862_data)) + (v3061_data * v3000_data)) + (v3199_data * v3138_data)) + (v3337_data * v3276_data);
              tensorforge::intel_esimd::simd<double, 16> v3376_p = (((((((((((((((((v991_data * v932_data) + (v1129_data * v1073_data)) + (v1267_data * v1211_data)) + (v1405_data * v1349_data)) + (v1543_data * v1487_data)) + (v1681_data * v1625_data)) + (v1819_data * v1763_data)) + (v1957_data * v1901_data)) + (v2095_data * v2039_data)) + (v2233_data * v2177_data)) + (v2371_data * v2315_data)) + (v2509_data * v2453_data)) + (v2647_data * v2591_data)) + (v2785_data * v2729_data)) + (v2923_data * v2867_data)) + (v3061_data * v3005_data)) + (v3199_data * v3143_data)) + (v3337_data * v3281_data);
              tensorforge::intel_esimd::simd<double, 16> v3381_p = (((((((((((((((((v991_data * v937_data) + (v1129_data * v1078_data)) + (v1267_data * v1216_data)) + (v1405_data * v1354_data)) + (v1543_data * v1492_data)) + (v1681_data * v1630_data)) + (v1819_data * v1768_data)) + (v1957_data * v1906_data)) + (v2095_data * v2044_data)) + (v2233_data * v2182_data)) + (v2371_data * v2320_data)) + (v2509_data * v2458_data)) + (v2647_data * v2596_data)) + (v2785_data * v2734_data)) + (v2923_data * v2872_data)) + (v3061_data * v3010_data)) + (v3199_data * v3148_data)) + (v3337_data * v3286_data);
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run18;
              glb_m3_run18.copy_from(glb_m3 + (760_i32));
              tensorforge::intel_esimd::simd<double, 16> v3383_data(glb_m3_run18.template select<16, 1>(0));
              double v3384_data = s1_w1[18];
              ir2.template select<16, 1>(0) = (((((((((((((((((((v896_data * v897_data) + (v1037_data * v1038_data)) + (v1175_data * v1176_data)) + (v1313_data * v1314_data)) + (v1451_data * v1452_data)) + (v1589_data * v1590_data)) + (v1727_data * v1728_data)) + (v1865_data * v1866_data)) + (v2003_data * v2004_data)) + (v2141_data * v2142_data)) + (v2279_data * v2280_data)) + (v2417_data * v2418_data)) + (v2555_data * v2556_data)) + (v2693_data * v2694_data)) + (v2831_data * v2832_data)) + (v2969_data * v2970_data)) + (v3107_data * v3108_data)) + (v3245_data * v3246_data)) + (v3383_data * v3384_data));
              double v3389_data = s1_w1[48];
              ir2.template select<16, 1>(48) = (v3254_p + (v3383_data * v3389_data));
              double v3394_data = s1_w1[78];
              ir2.template select<16, 1>(96) = (v3259_p + (v3383_data * v3394_data));
              double v3399_data = s1_w1[108];
              ir2.template select<16, 1>(144) = (v3264_p + (v3383_data * v3399_data));
              double v3404_data = s1_w1[138];
              ir2.template select<16, 1>(192) = (v3269_p + (v3383_data * v3404_data));
              double v3409_data = s1_w1[168];
              ir2.template select<16, 1>(240) = (v3274_p + (v3383_data * v3409_data));
              double v3414_data = s1_w1[198];
              ir2.template select<16, 1>(288) = (v3279_p + (v3383_data * v3414_data));
              double v3419_data = s1_w1[228];
              ir2.template select<16, 1>(336) = (v3284_p + (v3383_data * v3419_data));
              double v3424_data = s1_w1[258];
              ir2.template select<16, 1>(384) = (v3289_p + (v3383_data * v3424_data));
              tensorforge::intel_esimd::simd<double, 16> v3429_data(glb_m3_run18.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v3475_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v3475_data_part;
              v3475_data_part.copy_from(glb_m3 + (792_i32));
              v3475_data.template select<8, 1>(0) = v3475_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run19;
              glb_m3_run19.copy_from(glb_m3 + (816_i32));
              tensorforge::intel_esimd::simd<double, 16> v3521_data(glb_m3_run19.template select<16, 1>(0));
              double v3522_data = s1_w1[19];
              double v3527_data = s1_w1[49];
              double v3532_data = s1_w1[79];
              double v3537_data = s1_w1[109];
              double v3542_data = s1_w1[139];
              double v3547_data = s1_w1[169];
              double v3552_data = s1_w1[199];
              double v3557_data = s1_w1[229];
              double v3562_data = s1_w1[259];
              tensorforge::intel_esimd::simd<double, 16> v3567_data(glb_m3_run19.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run20;
              glb_m3_run20.copy_from(glb_m3 + (856_i32));
              tensorforge::intel_esimd::simd<double, 16> v3613_data(glb_m3_run20.template select<16, 1>(0));
              double v3614_data = s1_w1[20];
              double v3619_data = s1_w1[50];
              double v3624_data = s1_w1[80];
              double v3629_data = s1_w1[110];
              double v3634_data = s1_w1[140];
              double v3639_data = s1_w1[170];
              double v3644_data = s1_w1[200];
              double v3649_data = s1_w1[230];
              double v3654_data = s1_w1[260];
              tensorforge::intel_esimd::simd<double, 16> v3659_data(glb_m3_run20.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run21;
              glb_m3_run21.copy_from(glb_m3 + (896_i32));
              tensorforge::intel_esimd::simd<double, 16> v3705_data(glb_m3_run21.template select<16, 1>(0));
              double v3706_data = s1_w1[21];
              double v3711_data = s1_w1[51];
              double v3716_data = s1_w1[81];
              double v3721_data = s1_w1[111];
              double v3726_data = s1_w1[141];
              double v3731_data = s1_w1[171];
              double v3736_data = s1_w1[201];
              double v3741_data = s1_w1[231];
              double v3746_data = s1_w1[261];
              tensorforge::intel_esimd::simd<double, 16> v3751_data(glb_m3_run21.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run22;
              glb_m3_run22.copy_from(glb_m3 + (936_i32));
              tensorforge::intel_esimd::simd<double, 16> v3797_data(glb_m3_run22.template select<16, 1>(0));
              double v3798_data = s1_w1[22];
              double v3803_data = s1_w1[52];
              double v3808_data = s1_w1[82];
              double v3813_data = s1_w1[112];
              double v3818_data = s1_w1[142];
              double v3823_data = s1_w1[172];
              double v3828_data = s1_w1[202];
              double v3833_data = s1_w1[232];
              double v3838_data = s1_w1[262];
              tensorforge::intel_esimd::simd<double, 16> v3843_data(glb_m3_run22.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run23;
              glb_m3_run23.copy_from(glb_m3 + (976_i32));
              tensorforge::intel_esimd::simd<double, 16> v3889_data(glb_m3_run23.template select<16, 1>(0));
              double v3890_data = s1_w1[23];
              double v3895_data = s1_w1[53];
              double v3900_data = s1_w1[83];
              double v3905_data = s1_w1[113];
              double v3910_data = s1_w1[143];
              double v3915_data = s1_w1[173];
              double v3920_data = s1_w1[203];
              double v3925_data = s1_w1[233];
              double v3930_data = s1_w1[263];
              tensorforge::intel_esimd::simd<double, 16> v3935_data(glb_m3_run23.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run24;
              glb_m3_run24.copy_from(glb_m3 + (1016_i32));
              tensorforge::intel_esimd::simd<double, 16> v3981_data(glb_m3_run24.template select<16, 1>(0));
              double v3982_data = s1_w1[24];
              double v3987_data = s1_w1[54];
              double v3992_data = s1_w1[84];
              double v3997_data = s1_w1[114];
              double v4002_data = s1_w1[144];
              double v4007_data = s1_w1[174];
              double v4012_data = s1_w1[204];
              double v4017_data = s1_w1[234];
              double v4022_data = s1_w1[264];
              tensorforge::intel_esimd::simd<double, 16> v4027_data(glb_m3_run24.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run25;
              glb_m3_run25.copy_from(glb_m3 + (1056_i32));
              tensorforge::intel_esimd::simd<double, 16> v4073_data(glb_m3_run25.template select<16, 1>(0));
              double v4074_data = s1_w1[25];
              double v4079_data = s1_w1[55];
              double v4084_data = s1_w1[85];
              double v4089_data = s1_w1[115];
              double v4094_data = s1_w1[145];
              double v4099_data = s1_w1[175];
              double v4104_data = s1_w1[205];
              tensorforge::intel_esimd::simd<double, 16> v4107_p = (((((((v3325_p + (v3429_data * v3414_data)) + (v3521_data * v3552_data)) + (v3613_data * v3644_data)) + (v3705_data * v3736_data)) + (v3797_data * v3828_data)) + (v3889_data * v3920_data)) + (v3981_data * v4012_data)) + (v4073_data * v4104_data);
              double v4109_data = s1_w1[235];
              tensorforge::intel_esimd::simd<double, 16> v4112_p = (((((((v3330_p + (v3429_data * v3419_data)) + (v3521_data * v3557_data)) + (v3613_data * v3649_data)) + (v3705_data * v3741_data)) + (v3797_data * v3833_data)) + (v3889_data * v3925_data)) + (v3981_data * v4017_data)) + (v4073_data * v4109_data);
              double v4114_data = s1_w1[265];
              tensorforge::intel_esimd::simd<double, 16> v4117_p = (((((((v3335_p + (v3429_data * v3424_data)) + (v3521_data * v3562_data)) + (v3613_data * v3654_data)) + (v3705_data * v3746_data)) + (v3797_data * v3838_data)) + (v3889_data * v3930_data)) + (v3981_data * v4022_data)) + (v4073_data * v4114_data);
              tensorforge::intel_esimd::simd<double, 16> v4119_data(glb_m3_run25.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v4123_p = (((((((v3341_p + (v3475_data * v3384_data)) + (v3567_data * v3522_data)) + (v3659_data * v3614_data)) + (v3751_data * v3706_data)) + (v3843_data * v3798_data)) + (v3935_data * v3890_data)) + (v4027_data * v3982_data)) + (v4119_data * v4074_data);
              tensorforge::intel_esimd::simd<double, 16> v4128_p = (((((((v3346_p + (v3475_data * v3389_data)) + (v3567_data * v3527_data)) + (v3659_data * v3619_data)) + (v3751_data * v3711_data)) + (v3843_data * v3803_data)) + (v3935_data * v3895_data)) + (v4027_data * v3987_data)) + (v4119_data * v4079_data);
              tensorforge::intel_esimd::simd<double, 16> v4133_p = (((((((v3351_p + (v3475_data * v3394_data)) + (v3567_data * v3532_data)) + (v3659_data * v3624_data)) + (v3751_data * v3716_data)) + (v3843_data * v3808_data)) + (v3935_data * v3900_data)) + (v4027_data * v3992_data)) + (v4119_data * v4084_data);
              tensorforge::intel_esimd::simd<double, 16> v4138_p = (((((((v3356_p + (v3475_data * v3399_data)) + (v3567_data * v3537_data)) + (v3659_data * v3629_data)) + (v3751_data * v3721_data)) + (v3843_data * v3813_data)) + (v3935_data * v3905_data)) + (v4027_data * v3997_data)) + (v4119_data * v4089_data);
              tensorforge::intel_esimd::simd<double, 16> v4143_p = (((((((v3361_p + (v3475_data * v3404_data)) + (v3567_data * v3542_data)) + (v3659_data * v3634_data)) + (v3751_data * v3726_data)) + (v3843_data * v3818_data)) + (v3935_data * v3910_data)) + (v4027_data * v4002_data)) + (v4119_data * v4094_data);
              tensorforge::intel_esimd::simd<double, 16> v4148_p = (((((((v3366_p + (v3475_data * v3409_data)) + (v3567_data * v3547_data)) + (v3659_data * v3639_data)) + (v3751_data * v3731_data)) + (v3843_data * v3823_data)) + (v3935_data * v3915_data)) + (v4027_data * v4007_data)) + (v4119_data * v4099_data);
              tensorforge::intel_esimd::simd<double, 16> v4153_p = (((((((v3371_p + (v3475_data * v3414_data)) + (v3567_data * v3552_data)) + (v3659_data * v3644_data)) + (v3751_data * v3736_data)) + (v3843_data * v3828_data)) + (v3935_data * v3920_data)) + (v4027_data * v4012_data)) + (v4119_data * v4104_data);
              tensorforge::intel_esimd::simd<double, 16> v4158_p = (((((((v3376_p + (v3475_data * v3419_data)) + (v3567_data * v3557_data)) + (v3659_data * v3649_data)) + (v3751_data * v3741_data)) + (v3843_data * v3833_data)) + (v3935_data * v3925_data)) + (v4027_data * v4017_data)) + (v4119_data * v4109_data);
              tensorforge::intel_esimd::simd<double, 16> v4163_p = (((((((v3381_p + (v3475_data * v3424_data)) + (v3567_data * v3562_data)) + (v3659_data * v3654_data)) + (v3751_data * v3746_data)) + (v3843_data * v3838_data)) + (v3935_data * v3930_data)) + (v4027_data * v4022_data)) + (v4119_data * v4114_data);
              tensorforge::intel_esimd::simd<double, 16> v4165_data;
              v4165_data.copy_from(glb_m3 + (1096_i32));
              double v4166_data = s1_w1[26];
              tensorforge::intel_esimd::simd<double, 16> v4169_p = ((((((((v3295_p + (v3429_data * v3384_data)) + (v3521_data * v3522_data)) + (v3613_data * v3614_data)) + (v3705_data * v3706_data)) + (v3797_data * v3798_data)) + (v3889_data * v3890_data)) + (v3981_data * v3982_data)) + (v4073_data * v4074_data)) + (v4165_data * v4166_data);
              double v4171_data = s1_w1[56];
              tensorforge::intel_esimd::simd<double, 16> v4174_p = ((((((((v3300_p + (v3429_data * v3389_data)) + (v3521_data * v3527_data)) + (v3613_data * v3619_data)) + (v3705_data * v3711_data)) + (v3797_data * v3803_data)) + (v3889_data * v3895_data)) + (v3981_data * v3987_data)) + (v4073_data * v4079_data)) + (v4165_data * v4171_data);
              double v4176_data = s1_w1[86];
              tensorforge::intel_esimd::simd<double, 16> v4179_p = ((((((((v3305_p + (v3429_data * v3394_data)) + (v3521_data * v3532_data)) + (v3613_data * v3624_data)) + (v3705_data * v3716_data)) + (v3797_data * v3808_data)) + (v3889_data * v3900_data)) + (v3981_data * v3992_data)) + (v4073_data * v4084_data)) + (v4165_data * v4176_data);
              double v4181_data = s1_w1[116];
              tensorforge::intel_esimd::simd<double, 16> v4184_p = ((((((((v3310_p + (v3429_data * v3399_data)) + (v3521_data * v3537_data)) + (v3613_data * v3629_data)) + (v3705_data * v3721_data)) + (v3797_data * v3813_data)) + (v3889_data * v3905_data)) + (v3981_data * v3997_data)) + (v4073_data * v4089_data)) + (v4165_data * v4181_data);
              double v4186_data = s1_w1[146];
              tensorforge::intel_esimd::simd<double, 16> v4189_p = ((((((((v3315_p + (v3429_data * v3404_data)) + (v3521_data * v3542_data)) + (v3613_data * v3634_data)) + (v3705_data * v3726_data)) + (v3797_data * v3818_data)) + (v3889_data * v3910_data)) + (v3981_data * v4002_data)) + (v4073_data * v4094_data)) + (v4165_data * v4186_data);
              double v4191_data = s1_w1[176];
              tensorforge::intel_esimd::simd<double, 16> v4194_p = ((((((((v3320_p + (v3429_data * v3409_data)) + (v3521_data * v3547_data)) + (v3613_data * v3639_data)) + (v3705_data * v3731_data)) + (v3797_data * v3823_data)) + (v3889_data * v3915_data)) + (v3981_data * v4007_data)) + (v4073_data * v4099_data)) + (v4165_data * v4191_data);
              // r0 = load{g>r}(wrap_glb_m0);
              #pragma unroll
              for (int32_t v4507_i0 = 0; v4507_i0 < 1; ++v4507_i0) {
                int32_t v4508_lead = v4507_i0 * 16;
                #pragma unroll
                for (int32_t v4509_i1 = 0; v4509_i1 < 9; ++v4509_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v4512_data;
                  v4512_data.copy_from(wrap_glb_m0 + ((v4508_lead + (v4509_i1 * 30))));
                  r0.template select<16, 1>((v4508_lead + (v4509_i1 * 32))) = v4512_data;
                }
              }
              #pragma unroll
              for (int32_t v4515_i1 = 0; v4515_i1 < 9; ++v4515_i1) {
                tensorforge::intel_esimd::simd<double, 16> v4518_data(0.0);
                tensorforge::intel_esimd::simd<double, 14> v4518_data_part;
                v4518_data_part.copy_from(wrap_glb_m0 + ((16_i32 + (v4515_i1 * 30))));
                v4518_data.template select<14, 1>(0) = v4518_data_part;
                r0.template select<16, 1>((16 + (v4515_i1 * 32))) = v4518_data;
              }
              double v4196_data = s1[207];
              double v4201_data = s1[237];
              double v4206_data = s1[267];
              tensorforge::intel_esimd::simd<double, 16> v4211_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v4211_data_part;
              v4211_data_part.copy_from(glb_m3 + (1112_i32));
              v4211_data.template select<8, 1>(0) = v4211_data_part;
              tensorforge::intel_esimd::simd<double, 32> glb_m3_run26;
              glb_m3_run26.copy_from(glb_m3 + (1136_i32));
              tensorforge::intel_esimd::simd<double, 16> v4257_data(glb_m3_run26.template select<16, 1>(0));
              double v4258_data = s1[28];
              double v4263_data = s1[58];
              tensorforge::intel_esimd::simd<double, 16> v4266_p = v4174_p + (v4257_data * v4263_data);
              double v4268_data = s1[88];
              tensorforge::intel_esimd::simd<double, 16> v4271_p = v4179_p + (v4257_data * v4268_data);
              double v4273_data = s1[118];
              tensorforge::intel_esimd::simd<double, 16> v4276_p = v4184_p + (v4257_data * v4273_data);
              double v4278_data = s1[148];
              tensorforge::intel_esimd::simd<double, 16> v4281_p = v4189_p + (v4257_data * v4278_data);
              double v4283_data = s1[178];
              tensorforge::intel_esimd::simd<double, 16> v4286_p = v4194_p + (v4257_data * v4283_data);
              double v4288_data = s1[208];
              tensorforge::intel_esimd::simd<double, 16> v4291_p = (v4107_p + (v4165_data * v4196_data)) + (v4257_data * v4288_data);
              double v4293_data = s1[238];
              tensorforge::intel_esimd::simd<double, 16> v4296_p = (v4112_p + (v4165_data * v4201_data)) + (v4257_data * v4293_data);
              double v4298_data = s1[268];
              tensorforge::intel_esimd::simd<double, 16> v4301_p = (v4117_p + (v4165_data * v4206_data)) + (v4257_data * v4298_data);
              tensorforge::intel_esimd::simd<double, 16> v4303_data(glb_m3_run26.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<double, 16> v4307_p = (v4123_p + (v4211_data * v4166_data)) + (v4303_data * v4258_data);
              tensorforge::intel_esimd::simd<double, 16> v4312_p = (v4128_p + (v4211_data * v4171_data)) + (v4303_data * v4263_data);
              tensorforge::intel_esimd::simd<double, 16> v4317_p = (v4133_p + (v4211_data * v4176_data)) + (v4303_data * v4268_data);
              tensorforge::intel_esimd::simd<double, 16> v4322_p = (v4138_p + (v4211_data * v4181_data)) + (v4303_data * v4273_data);
              tensorforge::intel_esimd::simd<double, 16> v4327_p = (v4143_p + (v4211_data * v4186_data)) + (v4303_data * v4278_data);
              tensorforge::intel_esimd::simd<double, 16> v4332_p = (v4148_p + (v4211_data * v4191_data)) + (v4303_data * v4283_data);
              tensorforge::intel_esimd::simd<double, 16> v4337_p = (v4153_p + (v4211_data * v4196_data)) + (v4303_data * v4288_data);
              tensorforge::intel_esimd::simd<double, 16> v4342_p = (v4158_p + (v4211_data * v4201_data)) + (v4303_data * v4293_data);
              tensorforge::intel_esimd::simd<double, 16> v4347_p = (v4163_p + (v4211_data * v4206_data)) + (v4303_data * v4298_data);
              tensorforge::intel_esimd::simd<double, 16> v4349_data;
              v4349_data.copy_from(glb_m3 + (1176_i32));
              double v4350_data = s1[29];
              ir2.template select<16, 1>(16) = ((v4169_p + (v4257_data * v4258_data)) + (v4349_data * v4350_data));
              double v4355_data = s1[59];
              ir2.template select<16, 1>(64) = (v4266_p + (v4349_data * v4355_data));
              double v4360_data = s1[89];
              ir2.template select<16, 1>(112) = (v4271_p + (v4349_data * v4360_data));
              double v4365_data = s1[119];
              ir2.template select<16, 1>(160) = (v4276_p + (v4349_data * v4365_data));
              double v4370_data = s1[149];
              ir2.template select<16, 1>(208) = (v4281_p + (v4349_data * v4370_data));
              double v4375_data = s1[179];
              ir2.template select<16, 1>(256) = (v4286_p + (v4349_data * v4375_data));
              double v4380_data = s1[209];
              ir2.template select<16, 1>(304) = (v4291_p + (v4349_data * v4380_data));
              double v4385_data = s1[239];
              ir2.template select<16, 1>(352) = (v4296_p + (v4349_data * v4385_data));
              double v4390_data = s1[269];
              ir2.template select<16, 1>(400) = (v4301_p + (v4349_data * v4390_data));
              tensorforge::intel_esimd::simd<double, 16> v4395_data(0.0);
              tensorforge::intel_esimd::simd<double, 8> v4395_data_part;
              v4395_data_part.copy_from(glb_m3 + (1192_i32));
              v4395_data.template select<8, 1>(0) = v4395_data_part;
              ir2.template select<16, 1>(32) = (v4307_p + (v4395_data * v4350_data));
              ir2.template select<16, 1>(80) = (v4312_p + (v4395_data * v4355_data));
              ir2.template select<16, 1>(128) = (v4317_p + (v4395_data * v4360_data));
              ir2.template select<16, 1>(176) = (v4322_p + (v4395_data * v4365_data));
              ir2.template select<16, 1>(224) = (v4327_p + (v4395_data * v4370_data));
              ir2.template select<16, 1>(272) = (v4332_p + (v4395_data * v4375_data));
              ir2.template select<16, 1>(320) = (v4337_p + (v4395_data * v4380_data));
              ir2.template select<16, 1>(368) = (v4342_p + (v4395_data * v4385_data));
              ir2.template select<16, 1>(416) = (v4347_p + (v4395_data * v4390_data));
              // r2 = ir2
              // glb_m2 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v4450_i0 = 0; v4450_i0 < 2; ++v4450_i0) {
                int32_t v4452_a = v4450_i0 * 16;
                #pragma unroll
                for (int32_t v4451_i1 = 0; v4451_i1 < 9; ++v4451_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v4455_data(ir2.template select<16, 1>((v4452_a + (v4451_i1 * 48))));
                  v4455_data.copy_to(glb_m2 + ((v4452_a + (v4451_i1 * 40))));
                }
              }
              #pragma unroll
              for (int32_t v4459_i1 = 0; v4459_i1 < 9; ++v4459_i1) {
                tensorforge::intel_esimd::simd<double, 8> v4462_data(ir2.template select<8, 1>(((v4459_i1 * 48) + 32)));
                v4462_data.copy_to(glb_m2 + ((32_i32 + (v4459_i1 * 40))));
              }
            }
            else {
              // r0 = load{g>r}(wrap_glb_m0);
              #pragma unroll
              for (int32_t v4479_i0 = 0; v4479_i0 < 1; ++v4479_i0) {
                int32_t v4480_lead = v4479_i0 * 16;
                #pragma unroll
                for (int32_t v4481_i1 = 0; v4481_i1 < 9; ++v4481_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v4484_data;
                  v4484_data.copy_from(wrap_glb_m0 + ((v4480_lead + (v4481_i1 * 30))));
                  r0.template select<16, 1>((v4480_lead + (v4481_i1 * 32))) = v4484_data;
                }
              }
              #pragma unroll
              for (int32_t v4487_i1 = 0; v4487_i1 < 9; ++v4487_i1) {
                tensorforge::intel_esimd::simd<double, 16> v4490_data(0.0);
                tensorforge::intel_esimd::simd<double, 14> v4490_data_part;
                v4490_data_part.copy_from(wrap_glb_m0 + ((16_i32 + (v4487_i1 * 30))));
                v4490_data.template select<14, 1>(0) = v4490_data_part;
                r0.template select<16, 1>((16 + (v4487_i1 * 32))) = v4490_data;
              }
            }
            // s0 = load{g>s}(wrap_glb_m1[0, 1])
            tensorforge::intel_esimd::simd<double, 32> v4527_ld;
            v4527_ld.copy_from(wrap_glb_m1 + (0 + 0 + 2 * 0 + 0));
            tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v4527_ld);
            tensorforge::intel_esimd::simd<double, 32> v4528_ld;
            v4528_ld.copy_from(wrap_glb_m1 + (0 + 0 + 2 * 0 + 32));
            tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v4528_ld);
            tensorforge::intel_esimd::simd<double, 16> v4529_ld;
            v4529_ld.copy_from(wrap_glb_m1 + (0 + 0 + 1 * 0 + 64));
            tensorforge::slmStore<double, 16>(s0 + (0 + 0 + 1 * 0 + 64), v4529_ld);
            double v4530_ld = wrap_glb_m1[0 + 0 + 1 * 0 + 80];
            s0[0 + 0 + 1 * 0 + 80] = v4530_ld;
            v4473_flagWord = v4474_flagWordNext;
            v4474_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


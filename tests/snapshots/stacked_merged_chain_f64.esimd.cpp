// === base name ===
kernel_9f83c023ba433991

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9f83c023ba433991 = {{1, 16, 1}, 16, 20, 1, 16, 25600, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9f83c023ba433991(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9f83c023ba433991(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9f83c023ba433991(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 3200 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_9f83c023ba433991(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9f83c023ba433991(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_9f83c023ba433991(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m4, m4_extraOffset, m5, m6, m6_extraOffset, m7, m8, m8_extraOffset, m9, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_9f83c023ba433991_k;
inline void kernel_kernel_9f83c023ba433991(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_9f83c023ba433991_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<3200 * sizeof(double)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: amd_matrix_k4=1,lanes_per_mult=16,merge_variants=1,mma_d_layout=1,mma_d_layout_loops=1
        // launch: 16 lanes (20 active) x 16 per block = block 1x16x1, 25600 B shared, occupancy grid
        // operands:
        //   v0 9×9(9×9) {0..9}×{0..9} pointer_based
        //   v1 20×10(20×10) {0..20}×{0..10} none
        //   m0 20×9(20×9) {0..20}×{0..9} pointer_based
        //   m1 9×9(9×9) {0..9}×{0..9} pointer_based
        //   m2 20×9(20×9) {0..20}×{0..9} pointer_based
        //   m3 20×10(20×10) {0..20}×{0..10} none
        //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
        //   m5 20×10(20×10) {0..20}×{0..10} none
        //   m6 9×9(9×9) {0..9}×{0..9} pointer_based
        //   m7 20×10(20×10) {0..20}×{0..10} none
        //   m8 9×9(9×9) {0..9}×{0..9} pointer_based
        //   m9 20×10(20×10) {0..20}×{0..10} none
        // operations:
        //   for 4 iterations (any order):
        //     t0[i,j] = m0[i,k]@{0..10}×{0..9} × v0[k,j]
        //     m2[i,j] += v1[i,k] × t0[k,j]
        //     v0 ∈ {m1, m4, m6, m8}
        //     v1 ∈ {m3, m5, m7, m9}
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":20,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3200}],"shared_bytes":25600,"shared_elements":3200,"threads_per_mult":16},"loops":[{"body":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"v0","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"v1","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"holes":[{"members":["m1","m4","m6","m8"],"stand_in":"v0"},{"members":["m3","m5","m7","m9"],"stand_in":"v1"}],"iterations":4,"kind":"for","sequential":false}],"operands":[{"addressing":"pointer_based","alias":"variant0","bbox":[[0,0],[9,9]],"name":"v0","ordered":false,"parts":1,"shape":[9,9],"variant":true},{"addressing":"none","alias":"variant1","bbox":[[0,0],[20,10]],"name":"v1","ordered":false,"parts":1,"shape":[20,10],"variant":true},{"addressing":"pointer_based","alias":"I","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"pointer_based","alias":"S0","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[20,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"K0","bbox":[[0,0],[20,10]],"name":"m3","ordered":false,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S1","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K1","bbox":[[0,0],[20,10]],"name":"m5","ordered":false,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S2","bbox":[[0,0],[9,9]],"name":"m6","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K2","bbox":[[0,0],[20,10]],"name":"m7","ordered":false,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S3","bbox":[[0,0],[9,9]],"name":"m8","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K3","bbox":[[0,0],[20,10]],"name":"m9","ordered":false,"parts":1,"shape":[20,10],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m9","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (200 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const double *const __restrict__ glb_m3 = &m3[0];
          const double *const __restrict__ glb_m5 = &m5[0];
          const double *const __restrict__ glb_m7 = &m7[0];
          const double *const __restrict__ glb_m9 = &m9[0];
          tensorforge::intel_esimd::simd<double, 144> r0(0.0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v1469_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1469_flagSecondAhead < numElements0) ? v1469_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const double *const __restrict__ peel_glb_m0 = &m0[v3_batchId1][0 + m0_extraOffset];
          if (allowed_peel) {
            // r0 = load{g>r}(peel_glb_m0);
            #pragma unroll
            for (int32_t v1483_i1 = 0; v1483_i1 < 9; ++v1483_i1) {
              tensorforge::intel_esimd::simd<double, 10> v1485_data;
              v1485_data.copy_from(peel_glb_m0 + ((v1483_i1 * 20)));
              r0.template select<10, 1>((v1483_i1 * 16)) = v1485_data;
            }
          }
          uint32_t v1473_flagWord(flagWordFirst);
          uint32_t v1474_flagWordNext(flagWordSecond);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const double *const __restrict__ wrap_glb_m0 = &m0[v14_batchId1][0 + m0_extraOffset];
            size_t v1493_flagAheadAhead = v14_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1493_flagAheadAhead < numElements0) ? v1493_flagAheadAhead : v14_batchId1)];
            const bool allowed = static_cast<bool>(v1473_flagWord);
            const bool allowed_next = static_cast<bool>(v1474_flagWordNext);
            if (allowed) {
              const double *const __restrict__ glb_m1 = &m1[v11_batchId0][0 + m1_extraOffset];
              double *const __restrict__ glb_m2 = &m2[v11_batchId0][0 + m2_extraOffset];
              const double *const __restrict__ glb_m4 = &m4[v11_batchId0][0 + m4_extraOffset];
              const double *const __restrict__ glb_m6 = &m6[v11_batchId0][0 + m6_extraOffset];
              const double *const __restrict__ glb_m8 = &m8[v11_batchId0][0 + m8_extraOffset];
              tensorforge::SlmPtr<double> s0 = localShrMem0 + (96);
              tensorforge::SlmPtr<double> s1 = localShrMem0 + (0);
              tensorforge::intel_esimd::simd<int32_t, 16> v451_lead = tensorforge::intel_esimd::simd<int32_t, 16>(0, 1);
              const double *const peel_v0Table = (0 == 0) ? glb_m1 : (0 == 1) ? glb_m4 : (0 == 2) ? glb_m6 : glb_m8;
              const double *const __restrict__ peel_glb_v0 = peel_v0Table;
              // s0 = load{g>s}(peel_glb_v0[0, 1])
              tensorforge::intel_esimd::simd<double, 32> v1460_ld;
              v1460_ld.copy_from(peel_glb_v0 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v1460_ld);
              tensorforge::intel_esimd::simd<double, 32> v1461_ld;
              v1461_ld.copy_from(peel_glb_v0 + (0 + 0 + 2 * 0 + 32));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v1461_ld);
              tensorforge::intel_esimd::simd<double, 16> v1462_ld;
              v1462_ld.copy_from(peel_glb_v0 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<double, 16>(s0 + (0 + 0 + 1 * 0 + 64), v1462_ld);
              double v1463_ld = peel_glb_v0[0 + 0 + 1 * 0 + 80];
              s0[0 + 0 + 1 * 0 + 80] = v1463_ld;
              for (int32_t batchIdv6 = 0; batchIdv6 < 4; ++batchIdv6) {
                int32_t v1453_tripAhead = batchIdv6 + 1;
                int32_t v1455_trip = (v1453_tripAhead < 4) ? v1453_tripAhead : batchIdv6;
                const double *const wrap_v0Table = (v1455_trip == 0) ? glb_m1 : (v1455_trip == 1) ? glb_m4 : (v1455_trip == 2) ? glb_m6 : glb_m8;
                const double *const __restrict__ wrap_glb_v0 = wrap_v0Table;
                const double *const v1Table = (batchIdv6 == 0) ? glb_m3 : (batchIdv6 == 1) ? glb_m5 : (batchIdv6 == 2) ? glb_m7 : glb_m9;
                const double *const __restrict__ glb_v1 = v1Table;
                tensorforge::intel_esimd::simd<double, 288> r2(0.0);
                tensorforge::intel_esimd::simd<double, 144> r1(0.0);
                // r1 = +(r0 * s0) + None
                // [(0, 10), (0, 9)] [(0, 9)]
                tensorforge::intel_esimd::simd<double, 16> v46_data(r0.template select<16, 1>(0));
                tensorforge::intel_esimd::simd<double, 88> s0_w0 = tensorforge::slmLoad<double, 88>(s0 + 0);
                double v47_data = s0_w0[0];
                double v52_data = s0_w0[9];
                double v57_data = s0_w0[18];
                double v62_data = s0_w0[27];
                double v67_data = s0_w0[36];
                double v72_data = s0_w0[45];
                double v77_data = s0_w0[54];
                double v82_data = s0_w0[63];
                double v87_data = s0_w0[72];
                tensorforge::intel_esimd::simd<double, 16> v91_data(r0.template select<16, 1>(16));
                double v92_data = s0_w0[1];
                double v97_data = s0_w0[10];
                double v102_data = s0_w0[19];
                double v107_data = s0_w0[28];
                double v112_data = s0_w0[37];
                double v117_data = s0_w0[46];
                double v122_data = s0_w0[55];
                double v127_data = s0_w0[64];
                double v132_data = s0_w0[73];
                tensorforge::intel_esimd::simd<double, 16> v136_data(r0.template select<16, 1>(32));
                double v137_data = s0_w0[2];
                double v142_data = s0_w0[11];
                double v147_data = s0_w0[20];
                double v152_data = s0_w0[29];
                double v157_data = s0_w0[38];
                double v162_data = s0_w0[47];
                double v167_data = s0_w0[56];
                double v172_data = s0_w0[65];
                double v177_data = s0_w0[74];
                tensorforge::intel_esimd::simd<double, 16> v181_data(r0.template select<16, 1>(48));
                double v182_data = s0_w0[3];
                double v187_data = s0_w0[12];
                double v192_data = s0_w0[21];
                double v197_data = s0_w0[30];
                double v202_data = s0_w0[39];
                double v207_data = s0_w0[48];
                double v212_data = s0_w0[57];
                double v217_data = s0_w0[66];
                double v222_data = s0_w0[75];
                tensorforge::intel_esimd::simd<double, 16> v226_data(r0.template select<16, 1>(64));
                double v227_data = s0_w0[4];
                double v232_data = s0_w0[13];
                double v237_data = s0_w0[22];
                double v242_data = s0_w0[31];
                double v247_data = s0_w0[40];
                double v252_data = s0_w0[49];
                double v257_data = s0_w0[58];
                double v262_data = s0_w0[67];
                double v267_data = s0_w0[76];
                tensorforge::intel_esimd::simd<double, 16> v271_data(r0.template select<16, 1>(80));
                double v272_data = s0_w0[5];
                double v277_data = s0_w0[14];
                double v282_data = s0_w0[23];
                double v287_data = s0_w0[32];
                double v292_data = s0_w0[41];
                double v297_data = s0_w0[50];
                double v302_data = s0_w0[59];
                double v307_data = s0_w0[68];
                double v312_data = s0_w0[77];
                tensorforge::intel_esimd::simd<double, 16> v316_data(r0.template select<16, 1>(96));
                double v317_data = s0_w0[6];
                double v322_data = s0_w0[15];
                double v327_data = s0_w0[24];
                double v332_data = s0_w0[33];
                double v337_data = s0_w0[42];
                double v342_data = s0_w0[51];
                double v347_data = s0_w0[60];
                double v352_data = s0_w0[69];
                double v357_data = s0_w0[78];
                tensorforge::intel_esimd::simd<double, 16> v361_data(r0.template select<16, 1>(112));
                double v362_data = s0_w0[7];
                double v367_data = s0_w0[16];
                tensorforge::intel_esimd::simd<double, 16> v370_p = (((((((v46_data * v52_data) + (v91_data * v97_data)) + (v136_data * v142_data)) + (v181_data * v187_data)) + (v226_data * v232_data)) + (v271_data * v277_data)) + (v316_data * v322_data)) + (v361_data * v367_data);
                double v372_data = s0_w0[25];
                tensorforge::intel_esimd::simd<double, 16> v375_p = (((((((v46_data * v57_data) + (v91_data * v102_data)) + (v136_data * v147_data)) + (v181_data * v192_data)) + (v226_data * v237_data)) + (v271_data * v282_data)) + (v316_data * v327_data)) + (v361_data * v372_data);
                double v377_data = s0_w0[34];
                tensorforge::intel_esimd::simd<double, 16> v380_p = (((((((v46_data * v62_data) + (v91_data * v107_data)) + (v136_data * v152_data)) + (v181_data * v197_data)) + (v226_data * v242_data)) + (v271_data * v287_data)) + (v316_data * v332_data)) + (v361_data * v377_data);
                double v382_data = s0_w0[43];
                tensorforge::intel_esimd::simd<double, 16> v385_p = (((((((v46_data * v67_data) + (v91_data * v112_data)) + (v136_data * v157_data)) + (v181_data * v202_data)) + (v226_data * v247_data)) + (v271_data * v292_data)) + (v316_data * v337_data)) + (v361_data * v382_data);
                double v387_data = s0_w0[52];
                tensorforge::intel_esimd::simd<double, 16> v390_p = (((((((v46_data * v72_data) + (v91_data * v117_data)) + (v136_data * v162_data)) + (v181_data * v207_data)) + (v226_data * v252_data)) + (v271_data * v297_data)) + (v316_data * v342_data)) + (v361_data * v387_data);
                double v392_data = s0_w0[61];
                tensorforge::intel_esimd::simd<double, 16> v395_p = (((((((v46_data * v77_data) + (v91_data * v122_data)) + (v136_data * v167_data)) + (v181_data * v212_data)) + (v226_data * v257_data)) + (v271_data * v302_data)) + (v316_data * v347_data)) + (v361_data * v392_data);
                double v397_data = s0_w0[70];
                tensorforge::intel_esimd::simd<double, 16> v400_p = (((((((v46_data * v82_data) + (v91_data * v127_data)) + (v136_data * v172_data)) + (v181_data * v217_data)) + (v226_data * v262_data)) + (v271_data * v307_data)) + (v316_data * v352_data)) + (v361_data * v397_data);
                double v402_data = s0_w0[79];
                tensorforge::intel_esimd::simd<double, 16> v405_p = (((((((v46_data * v87_data) + (v91_data * v132_data)) + (v136_data * v177_data)) + (v181_data * v222_data)) + (v226_data * v267_data)) + (v271_data * v312_data)) + (v316_data * v357_data)) + (v361_data * v402_data);
                tensorforge::intel_esimd::simd<double, 16> v406_data(r0.template select<16, 1>(128));
                double v407_data = s0_w0[8];
                r1.template select<16, 1>(0) = (((((((((v46_data * v47_data) + (v91_data * v92_data)) + (v136_data * v137_data)) + (v181_data * v182_data)) + (v226_data * v227_data)) + (v271_data * v272_data)) + (v316_data * v317_data)) + (v361_data * v362_data)) + (v406_data * v407_data));
                double v412_data = s0_w0[17];
                r1.template select<16, 1>(16) = (v370_p + (v406_data * v412_data));
                double v417_data = s0_w0[26];
                r1.template select<16, 1>(32) = (v375_p + (v406_data * v417_data));
                double v422_data = s0_w0[35];
                r1.template select<16, 1>(48) = (v380_p + (v406_data * v422_data));
                double v427_data = s0_w0[44];
                r1.template select<16, 1>(64) = (v385_p + (v406_data * v427_data));
                double v432_data = s0_w0[53];
                r1.template select<16, 1>(80) = (v390_p + (v406_data * v432_data));
                double v437_data = s0_w0[62];
                r1.template select<16, 1>(96) = (v395_p + (v406_data * v437_data));
                double v442_data = s0_w0[71];
                r1.template select<16, 1>(112) = (v400_p + (v406_data * v442_data));
                double v447_data = s0_w0[80];
                r1.template select<16, 1>(128) = (v405_p + (v406_data * v447_data));
                // r1 = 0 past (0, 10)
                // s1 = store{r>s}(localShrMem0, r1);
                #pragma unroll
                for (int32_t v469_i1 = 0; v469_i1 < 9; ++v469_i1) {
                  tensorforge::intel_esimd::simd<double, 10> v472_data(r1.template select<10, 1>((v469_i1 * 16)));
                  tensorforge::slmStore<double, 10>(s1 + ((v469_i1 * 10)), v472_data);
                }
                tensorforge::intel_esimd::simd<double, 288> r3(0.0);
                // ir3 = +(glb_v1 * s1)
                // [(0, 20), (0, 9)] [(0, 10)]
                tensorforge::intel_esimd::simd<double, 288> ir3(0.0);
                tensorforge::intel_esimd::simd<double, 32> glb_v1_run0;
                glb_v1_run0.copy_from(glb_v1 + (0_i32));
                tensorforge::intel_esimd::simd<double, 16> v482_data(glb_v1_run0.template select<16, 1>(0));
                tensorforge::intel_esimd::simd<double, 96> s1_w1 = tensorforge::slmLoad<double, 96>(s1 + 0);
                double v483_data = s1_w1[0];
                double v488_data = s1_w1[10];
                double v493_data = s1_w1[20];
                double v498_data = s1_w1[30];
                double v503_data = s1_w1[40];
                double v508_data = s1_w1[50];
                double v513_data = s1_w1[60];
                double v518_data = s1_w1[70];
                double v523_data = s1_w1[80];
                tensorforge::intel_esimd::simd<double, 16> v529_data(glb_v1_run0.template select<16, 1>(16));
                tensorforge::intel_esimd::simd<double, 32> glb_v1_run1;
                glb_v1_run1.copy_from(glb_v1 + (20_i32));
                tensorforge::intel_esimd::simd<double, 16> v575_data(glb_v1_run1.template select<16, 1>(0));
                double v576_data = s1_w1[1];
                double v581_data = s1_w1[11];
                double v586_data = s1_w1[21];
                double v591_data = s1_w1[31];
                double v596_data = s1_w1[41];
                double v601_data = s1_w1[51];
                double v606_data = s1_w1[61];
                double v611_data = s1_w1[71];
                double v616_data = s1_w1[81];
                tensorforge::intel_esimd::simd<double, 16> v621_data(glb_v1_run1.template select<16, 1>(16));
                tensorforge::intel_esimd::simd<double, 32> glb_v1_run2;
                glb_v1_run2.copy_from(glb_v1 + (40_i32));
                tensorforge::intel_esimd::simd<double, 16> v667_data(glb_v1_run2.template select<16, 1>(0));
                double v668_data = s1_w1[2];
                double v673_data = s1_w1[12];
                double v678_data = s1_w1[22];
                double v683_data = s1_w1[32];
                double v688_data = s1_w1[42];
                double v693_data = s1_w1[52];
                double v698_data = s1_w1[62];
                double v703_data = s1_w1[72];
                double v708_data = s1_w1[82];
                tensorforge::intel_esimd::simd<double, 16> v713_data(glb_v1_run2.template select<16, 1>(16));
                tensorforge::intel_esimd::simd<double, 16> v747_p = ((v529_data * v513_data) + (v621_data * v606_data)) + (v713_data * v698_data);
                tensorforge::intel_esimd::simd<double, 16> v752_p = ((v529_data * v518_data) + (v621_data * v611_data)) + (v713_data * v703_data);
                tensorforge::intel_esimd::simd<double, 16> v757_p = ((v529_data * v523_data) + (v621_data * v616_data)) + (v713_data * v708_data);
                tensorforge::intel_esimd::simd<double, 32> glb_v1_run3;
                glb_v1_run3.copy_from(glb_v1 + (60_i32));
                tensorforge::intel_esimd::simd<double, 16> v759_data(glb_v1_run3.template select<16, 1>(0));
                double v760_data = s1_w1[3];
                tensorforge::intel_esimd::simd<double, 16> v763_p = (((v482_data * v483_data) + (v575_data * v576_data)) + (v667_data * v668_data)) + (v759_data * v760_data);
                double v765_data = s1_w1[13];
                tensorforge::intel_esimd::simd<double, 16> v768_p = (((v482_data * v488_data) + (v575_data * v581_data)) + (v667_data * v673_data)) + (v759_data * v765_data);
                double v770_data = s1_w1[23];
                tensorforge::intel_esimd::simd<double, 16> v773_p = (((v482_data * v493_data) + (v575_data * v586_data)) + (v667_data * v678_data)) + (v759_data * v770_data);
                double v775_data = s1_w1[33];
                tensorforge::intel_esimd::simd<double, 16> v778_p = (((v482_data * v498_data) + (v575_data * v591_data)) + (v667_data * v683_data)) + (v759_data * v775_data);
                double v780_data = s1_w1[43];
                tensorforge::intel_esimd::simd<double, 16> v783_p = (((v482_data * v503_data) + (v575_data * v596_data)) + (v667_data * v688_data)) + (v759_data * v780_data);
                double v785_data = s1_w1[53];
                tensorforge::intel_esimd::simd<double, 16> v788_p = (((v482_data * v508_data) + (v575_data * v601_data)) + (v667_data * v693_data)) + (v759_data * v785_data);
                double v790_data = s1_w1[63];
                tensorforge::intel_esimd::simd<double, 16> v793_p = (((v482_data * v513_data) + (v575_data * v606_data)) + (v667_data * v698_data)) + (v759_data * v790_data);
                double v795_data = s1_w1[73];
                tensorforge::intel_esimd::simd<double, 16> v798_p = (((v482_data * v518_data) + (v575_data * v611_data)) + (v667_data * v703_data)) + (v759_data * v795_data);
                double v800_data = s1_w1[83];
                tensorforge::intel_esimd::simd<double, 16> v803_p = (((v482_data * v523_data) + (v575_data * v616_data)) + (v667_data * v708_data)) + (v759_data * v800_data);
                tensorforge::intel_esimd::simd<double, 16> v805_data(glb_v1_run3.template select<16, 1>(16));
                tensorforge::intel_esimd::simd<double, 16> v809_p = (((v529_data * v483_data) + (v621_data * v576_data)) + (v713_data * v668_data)) + (v805_data * v760_data);
                tensorforge::intel_esimd::simd<double, 16> v814_p = (((v529_data * v488_data) + (v621_data * v581_data)) + (v713_data * v673_data)) + (v805_data * v765_data);
                tensorforge::intel_esimd::simd<double, 16> v819_p = (((v529_data * v493_data) + (v621_data * v586_data)) + (v713_data * v678_data)) + (v805_data * v770_data);
                tensorforge::intel_esimd::simd<double, 16> v824_p = (((v529_data * v498_data) + (v621_data * v591_data)) + (v713_data * v683_data)) + (v805_data * v775_data);
                tensorforge::intel_esimd::simd<double, 16> v829_p = (((v529_data * v503_data) + (v621_data * v596_data)) + (v713_data * v688_data)) + (v805_data * v780_data);
                tensorforge::intel_esimd::simd<double, 16> v834_p = (((v529_data * v508_data) + (v621_data * v601_data)) + (v713_data * v693_data)) + (v805_data * v785_data);
                // r2 = load{g>r}(glb_m2);
                #pragma unroll
                for (int32_t v1403_i0 = 0; v1403_i0 < 1; ++v1403_i0) {
                  int32_t v1405_lead = v1403_i0 * 16;
                  #pragma unroll
                  for (int32_t v1404_i1 = 0; v1404_i1 < 9; ++v1404_i1) {
                    tensorforge::intel_esimd::simd<double, 16> v1409_data;
                    v1409_data.copy_from(glb_m2 + ((v1405_lead + (v1404_i1 * 20))));
                    r2.template select<16, 1>((v1405_lead + (v1404_i1 * 32))) = v1409_data;
                  }
                }
                #pragma unroll
                for (int32_t v1412_i1 = 0; v1412_i1 < 9; ++v1412_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v1418_data(0.0);
                  tensorforge::intel_esimd::simd<double, 4> v1418_data_part;
                  v1418_data_part.copy_from(glb_m2 + ((16_i32 + (v1412_i1 * 20))));
                  v1418_data.template select<4, 1>(0) = v1418_data_part;
                  r2.template select<16, 1>((16 + (v1412_i1 * 32))) = v1418_data;
                }
                tensorforge::intel_esimd::simd<double, 32> glb_v1_run4;
                glb_v1_run4.copy_from(glb_v1 + (80_i32));
                tensorforge::intel_esimd::simd<double, 16> v851_data(glb_v1_run4.template select<16, 1>(0));
                tensorforge::intel_esimd::simd<double, 92> s1_w2 = tensorforge::slmLoad<double, 92>(s1 + 4);
                double v852_data = s1_w2[0];
                double v857_data = s1_w2[10];
                double v862_data = s1_w2[20];
                double v867_data = s1_w2[30];
                double v872_data = s1_w2[40];
                double v877_data = s1_w2[50];
                double v882_data = s1_w2[60];
                double v887_data = s1_w2[70];
                double v892_data = s1_w2[80];
                tensorforge::intel_esimd::simd<double, 16> v897_data(glb_v1_run4.template select<16, 1>(16));
                tensorforge::intel_esimd::simd<double, 32> glb_v1_run5;
                glb_v1_run5.copy_from(glb_v1 + (100_i32));
                tensorforge::intel_esimd::simd<double, 16> v943_data(glb_v1_run5.template select<16, 1>(0));
                double v944_data = s1_w2[1];
                double v949_data = s1_w2[11];
                double v954_data = s1_w2[21];
                double v959_data = s1_w2[31];
                double v964_data = s1_w2[41];
                double v969_data = s1_w2[51];
                double v974_data = s1_w2[61];
                double v979_data = s1_w2[71];
                double v984_data = s1_w2[81];
                tensorforge::intel_esimd::simd<double, 16> v989_data(glb_v1_run5.template select<16, 1>(16));
                tensorforge::intel_esimd::simd<double, 16> v1003_p = (v819_p + (v897_data * v862_data)) + (v989_data * v954_data);
                tensorforge::intel_esimd::simd<double, 16> v1008_p = (v824_p + (v897_data * v867_data)) + (v989_data * v959_data);
                tensorforge::intel_esimd::simd<double, 16> v1013_p = (v829_p + (v897_data * v872_data)) + (v989_data * v964_data);
                tensorforge::intel_esimd::simd<double, 16> v1018_p = (v834_p + (v897_data * v877_data)) + (v989_data * v969_data);
                tensorforge::intel_esimd::simd<double, 16> v1023_p = ((v747_p + (v805_data * v790_data)) + (v897_data * v882_data)) + (v989_data * v974_data);
                tensorforge::intel_esimd::simd<double, 16> v1028_p = ((v752_p + (v805_data * v795_data)) + (v897_data * v887_data)) + (v989_data * v979_data);
                tensorforge::intel_esimd::simd<double, 16> v1033_p = ((v757_p + (v805_data * v800_data)) + (v897_data * v892_data)) + (v989_data * v984_data);
                tensorforge::intel_esimd::simd<double, 32> glb_v1_run6;
                glb_v1_run6.copy_from(glb_v1 + (120_i32));
                tensorforge::intel_esimd::simd<double, 16> v1035_data(glb_v1_run6.template select<16, 1>(0));
                double v1036_data = s1_w2[2];
                tensorforge::intel_esimd::simd<double, 16> v1039_p = ((v763_p + (v851_data * v852_data)) + (v943_data * v944_data)) + (v1035_data * v1036_data);
                double v1041_data = s1_w2[12];
                tensorforge::intel_esimd::simd<double, 16> v1044_p = ((v768_p + (v851_data * v857_data)) + (v943_data * v949_data)) + (v1035_data * v1041_data);
                double v1046_data = s1_w2[22];
                tensorforge::intel_esimd::simd<double, 16> v1049_p = ((v773_p + (v851_data * v862_data)) + (v943_data * v954_data)) + (v1035_data * v1046_data);
                double v1051_data = s1_w2[32];
                tensorforge::intel_esimd::simd<double, 16> v1054_p = ((v778_p + (v851_data * v867_data)) + (v943_data * v959_data)) + (v1035_data * v1051_data);
                double v1056_data = s1_w2[42];
                tensorforge::intel_esimd::simd<double, 16> v1059_p = ((v783_p + (v851_data * v872_data)) + (v943_data * v964_data)) + (v1035_data * v1056_data);
                double v1061_data = s1_w2[52];
                tensorforge::intel_esimd::simd<double, 16> v1064_p = ((v788_p + (v851_data * v877_data)) + (v943_data * v969_data)) + (v1035_data * v1061_data);
                double v1066_data = s1_w2[62];
                tensorforge::intel_esimd::simd<double, 16> v1069_p = ((v793_p + (v851_data * v882_data)) + (v943_data * v974_data)) + (v1035_data * v1066_data);
                double v1071_data = s1_w2[72];
                tensorforge::intel_esimd::simd<double, 16> v1074_p = ((v798_p + (v851_data * v887_data)) + (v943_data * v979_data)) + (v1035_data * v1071_data);
                double v1076_data = s1_w2[82];
                tensorforge::intel_esimd::simd<double, 16> v1079_p = ((v803_p + (v851_data * v892_data)) + (v943_data * v984_data)) + (v1035_data * v1076_data);
                tensorforge::intel_esimd::simd<double, 16> v1081_data(glb_v1_run6.template select<16, 1>(16));
                tensorforge::intel_esimd::simd<double, 16> v1085_p = ((v809_p + (v897_data * v852_data)) + (v989_data * v944_data)) + (v1081_data * v1036_data);
                tensorforge::intel_esimd::simd<double, 16> v1090_p = ((v814_p + (v897_data * v857_data)) + (v989_data * v949_data)) + (v1081_data * v1041_data);
                // s0 = load{g>s}(wrap_glb_v0[0, 1])
                tensorforge::intel_esimd::simd<double, 32> v1464_ld;
                v1464_ld.copy_from(wrap_glb_v0 + (0 + 0 + 2 * 0 + 0));
                tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v1464_ld);
                tensorforge::intel_esimd::simd<double, 32> v1465_ld;
                v1465_ld.copy_from(wrap_glb_v0 + (0 + 0 + 2 * 0 + 32));
                tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v1465_ld);
                tensorforge::intel_esimd::simd<double, 16> v1466_ld;
                v1466_ld.copy_from(wrap_glb_v0 + (0 + 0 + 1 * 0 + 64));
                tensorforge::slmStore<double, 16>(s0 + (0 + 0 + 1 * 0 + 64), v1466_ld);
                double v1467_ld = wrap_glb_v0[0 + 0 + 1 * 0 + 80];
                s0[0 + 0 + 1 * 0 + 80] = v1467_ld;
                tensorforge::intel_esimd::simd<double, 32> glb_v1_run7;
                glb_v1_run7.copy_from(glb_v1 + (140_i32));
                tensorforge::intel_esimd::simd<double, 16> v1127_data(glb_v1_run7.template select<16, 1>(0));
                tensorforge::intel_esimd::simd<double, 89> s1_w3 = tensorforge::slmLoad<double, 89>(s1 + 7);
                double v1128_data = s1_w3[0];
                double v1133_data = s1_w3[10];
                double v1138_data = s1_w3[20];
                double v1143_data = s1_w3[30];
                double v1148_data = s1_w3[40];
                double v1153_data = s1_w3[50];
                double v1158_data = s1_w3[60];
                double v1163_data = s1_w3[70];
                double v1168_data = s1_w3[80];
                tensorforge::intel_esimd::simd<double, 16> v1173_data(glb_v1_run7.template select<16, 1>(16));
                tensorforge::intel_esimd::simd<double, 32> glb_v1_run8;
                glb_v1_run8.copy_from(glb_v1 + (160_i32));
                tensorforge::intel_esimd::simd<double, 16> v1219_data(glb_v1_run8.template select<16, 1>(0));
                double v1220_data = s1_w3[1];
                double v1225_data = s1_w3[11];
                tensorforge::intel_esimd::simd<double, 16> v1228_p = (v1044_p + (v1127_data * v1133_data)) + (v1219_data * v1225_data);
                double v1230_data = s1_w3[21];
                tensorforge::intel_esimd::simd<double, 16> v1233_p = (v1049_p + (v1127_data * v1138_data)) + (v1219_data * v1230_data);
                double v1235_data = s1_w3[31];
                tensorforge::intel_esimd::simd<double, 16> v1238_p = (v1054_p + (v1127_data * v1143_data)) + (v1219_data * v1235_data);
                double v1240_data = s1_w3[41];
                tensorforge::intel_esimd::simd<double, 16> v1243_p = (v1059_p + (v1127_data * v1148_data)) + (v1219_data * v1240_data);
                double v1245_data = s1_w3[51];
                tensorforge::intel_esimd::simd<double, 16> v1248_p = (v1064_p + (v1127_data * v1153_data)) + (v1219_data * v1245_data);
                double v1250_data = s1_w3[61];
                tensorforge::intel_esimd::simd<double, 16> v1253_p = (v1069_p + (v1127_data * v1158_data)) + (v1219_data * v1250_data);
                double v1255_data = s1_w3[71];
                tensorforge::intel_esimd::simd<double, 16> v1258_p = (v1074_p + (v1127_data * v1163_data)) + (v1219_data * v1255_data);
                double v1260_data = s1_w3[81];
                tensorforge::intel_esimd::simd<double, 16> v1263_p = (v1079_p + (v1127_data * v1168_data)) + (v1219_data * v1260_data);
                tensorforge::intel_esimd::simd<double, 16> v1265_data(glb_v1_run8.template select<16, 1>(16));
                tensorforge::intel_esimd::simd<double, 16> v1269_p = (v1085_p + (v1173_data * v1128_data)) + (v1265_data * v1220_data);
                tensorforge::intel_esimd::simd<double, 16> v1274_p = (v1090_p + (v1173_data * v1133_data)) + (v1265_data * v1225_data);
                tensorforge::intel_esimd::simd<double, 16> v1279_p = ((v1003_p + (v1081_data * v1046_data)) + (v1173_data * v1138_data)) + (v1265_data * v1230_data);
                tensorforge::intel_esimd::simd<double, 16> v1284_p = ((v1008_p + (v1081_data * v1051_data)) + (v1173_data * v1143_data)) + (v1265_data * v1235_data);
                tensorforge::intel_esimd::simd<double, 16> v1289_p = ((v1013_p + (v1081_data * v1056_data)) + (v1173_data * v1148_data)) + (v1265_data * v1240_data);
                tensorforge::intel_esimd::simd<double, 16> v1294_p = ((v1018_p + (v1081_data * v1061_data)) + (v1173_data * v1153_data)) + (v1265_data * v1245_data);
                tensorforge::intel_esimd::simd<double, 16> v1299_p = ((v1023_p + (v1081_data * v1066_data)) + (v1173_data * v1158_data)) + (v1265_data * v1250_data);
                tensorforge::intel_esimd::simd<double, 16> v1304_p = ((v1028_p + (v1081_data * v1071_data)) + (v1173_data * v1163_data)) + (v1265_data * v1255_data);
                tensorforge::intel_esimd::simd<double, 16> v1309_p = ((v1033_p + (v1081_data * v1076_data)) + (v1173_data * v1168_data)) + (v1265_data * v1260_data);
                tensorforge::intel_esimd::simd<double, 16> v1311_data;
                v1311_data.copy_from(glb_v1 + (180_i32));
                double v1312_data = s1_w3[2];
                ir3.template select<16, 1>(0) = (((v1039_p + (v1127_data * v1128_data)) + (v1219_data * v1220_data)) + (v1311_data * v1312_data));
                double v1317_data = s1_w3[12];
                ir3.template select<16, 1>(32) = (v1228_p + (v1311_data * v1317_data));
                double v1322_data = s1_w3[22];
                ir3.template select<16, 1>(64) = (v1233_p + (v1311_data * v1322_data));
                double v1327_data = s1_w3[32];
                ir3.template select<16, 1>(96) = (v1238_p + (v1311_data * v1327_data));
                double v1332_data = s1_w3[42];
                ir3.template select<16, 1>(128) = (v1243_p + (v1311_data * v1332_data));
                double v1337_data = s1_w3[52];
                ir3.template select<16, 1>(160) = (v1248_p + (v1311_data * v1337_data));
                double v1342_data = s1_w3[62];
                ir3.template select<16, 1>(192) = (v1253_p + (v1311_data * v1342_data));
                double v1347_data = s1_w3[72];
                ir3.template select<16, 1>(224) = (v1258_p + (v1311_data * v1347_data));
                double v1352_data = s1_w3[82];
                ir3.template select<16, 1>(256) = (v1263_p + (v1311_data * v1352_data));
                tensorforge::intel_esimd::simd<double, 16> v1357_data(0.0);
                tensorforge::intel_esimd::simd<double, 4> v1357_data_part;
                v1357_data_part.copy_from(glb_v1 + (196_i32));
                v1357_data.template select<4, 1>(0) = v1357_data_part;
                ir3.template select<16, 1>(16) = (v1269_p + (v1357_data * v1312_data));
                ir3.template select<16, 1>(48) = (v1274_p + (v1357_data * v1317_data));
                ir3.template select<16, 1>(80) = (v1279_p + (v1357_data * v1322_data));
                ir3.template select<16, 1>(112) = (v1284_p + (v1357_data * v1327_data));
                ir3.template select<16, 1>(144) = (v1289_p + (v1357_data * v1332_data));
                ir3.template select<16, 1>(176) = (v1294_p + (v1357_data * v1337_data));
                ir3.template select<16, 1>(208) = (v1299_p + (v1357_data * v1342_data));
                ir3.template select<16, 1>(240) = (v1304_p + (v1357_data * v1347_data));
                ir3.template select<16, 1>(272) = (v1309_p + (v1357_data * v1352_data));
                // r3 = ir3 + r2
                #pragma unroll
                for (int32_t v1421_n0 = 0; v1421_n0 < 1; ++v1421_n0) {
                  int32_t v1423_a = v1421_n0 * 16;
                  #pragma unroll
                  for (int32_t v1422_n1 = 0; v1422_n1 < 9; ++v1422_n1) {
                    int32_t v1425_a = v1423_a + (v1422_n1 * 32);
                    tensorforge::intel_esimd::simd<double, 16> v1426_data(ir3.template select<16, 1>(v1425_a));
                    tensorforge::intel_esimd::simd<double, 16> v1427_data(r2.template select<16, 1>(v1425_a));
                    r3.template select<16, 1>(v1425_a) = (v1427_data + v1426_data);
                  }
                }
                #pragma unroll
                for (int32_t v1429_n1 = 0; v1429_n1 < 9; ++v1429_n1) {
                  int32_t v1431_a = 16 + (v1429_n1 * 32);
                  tensorforge::intel_esimd::simd<double, 4> v1432_data(ir3.template select<4, 1>(v1431_a));
                  tensorforge::intel_esimd::simd<double, 4> v1433_data(r2.template select<4, 1>(v1431_a));
                  r3.template select<4, 1>(v1431_a) = (v1433_data + v1432_data);
                }
                // glb_m2 = store{r>g}(r3);
                #pragma unroll
                for (int32_t v1435_i0 = 0; v1435_i0 < 1; ++v1435_i0) {
                  int32_t v1437_a = v1435_i0 * 16;
                  #pragma unroll
                  for (int32_t v1436_i1 = 0; v1436_i1 < 9; ++v1436_i1) {
                    tensorforge::intel_esimd::simd<double, 16> v1440_data(r3.template select<16, 1>((v1437_a + (v1436_i1 * 32))));
                    v1440_data.copy_to(glb_m2 + ((v1437_a + (v1436_i1 * 20))));
                  }
                }
                #pragma unroll
                for (int32_t v1444_i1 = 0; v1444_i1 < 9; ++v1444_i1) {
                  tensorforge::intel_esimd::simd<double, 4> v1447_data(r3.template select<4, 1>((16 + (v1444_i1 * 32))));
                  v1447_data.copy_to(glb_m2 + ((16_i32 + (v1444_i1 * 20))));
                }
              }
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m0);
                #pragma unroll
                for (int32_t v1487_i1 = 0; v1487_i1 < 9; ++v1487_i1) {
                  tensorforge::intel_esimd::simd<double, 10> v1489_data;
                  v1489_data.copy_from(wrap_glb_m0 + ((v1487_i1 * 20)));
                  r0.template select<10, 1>((v1487_i1 * 16)) = v1489_data;
                }
              }
            }
            else {
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m0);
                #pragma unroll
                for (int32_t v1479_i1 = 0; v1479_i1 < 9; ++v1479_i1) {
                  tensorforge::intel_esimd::simd<double, 10> v1481_data;
                  v1481_data.copy_from(wrap_glb_m0 + ((v1479_i1 * 20)));
                  r0.template select<10, 1>((v1479_i1 * 16)) = v1481_data;
                }
              }
            }
            v1473_flagWord = v1474_flagWordNext;
            v1474_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


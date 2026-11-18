// === base name ===
kernel_7a3dfdb68a42df6d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7a3dfdb68a42df6d = {{1, 8, 1}, 32, 20, 1, 8, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7a3dfdb68a42df6d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7a3dfdb68a42df6d(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7a3dfdb68a42df6d(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 8 - 1) / 8;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 768 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7a3dfdb68a42df6d(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7a3dfdb68a42df6d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7a3dfdb68a42df6d(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_7a3dfdb68a42df6d_k;
inline void kernel_kernel_7a3dfdb68a42df6d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_7a3dfdb68a42df6d_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<768 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (20 active) x 8 per block = block 1x8x1, 3072 B shared, occupancy grid
        // operands:
        //   m0 20×9(20×9) {0..20}×{0..9} indexed_pointer(slots=2, slot=1)
        //   m1 20×9(20×9) {0..20}×{0..9} none
        //   m2 9×9(9×9) {0..9}×{0..9} indexed(offset=12)
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"indexed_pointer(slots=2, slot=1)","alias":"C","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"indexed(offset=12)","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"indexed_pointer(slots=2, slot=1)","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]},{"addressing":"indexed(offset=12)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (96 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::intel_esimd::simd<float, 288> r0(0.0f);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v463_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v463_flagSecondAhead < numElements0) ? v463_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const float *const __restrict__ peel_glb_m2 = &m2[static_cast<size_t>(elementIndices[v3_batchId1]) * 81 + 0 + 12 + m2_extraOffset];
          if (allowed_peel) {
            // s0 = load{g>s}(peel_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v473_ld;
            v473_ld.copy_from(peel_glb_m2 + (0 + 0 + 2 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 2 * 0 + 0), v473_ld);
            tensorforge::intel_esimd::simd<float, 17> v474_ld;
            v474_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 64));
            tensorforge::slmStore<float, 17>(s0 + (0 + 0 + 1 * 0 + 64), v474_ld);
          }
          float *const __restrict__ peel_glb_m0 = &m0[static_cast<size_t>(elementIndices[v3_batchId1]) * 2 + 1][0 + m0_extraOffset];
          if (allowed_peel) {
            // r0 = load{g>r}(peel_glb_m0);
            #pragma unroll
            for (int32_t v483_i1 = 0; v483_i1 < 9; ++v483_i1) {
              tensorforge::intel_esimd::simd<float, 32> v485_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v485_data_part;
              v485_data_part.copy_from(peel_glb_m0 + ((v483_i1 * 20)));
              v485_data.template select<20, 1>(0) = v485_data_part;
              r0.template select<32, 1>((v483_i1 * 32)) = v485_data;
            }
          }
          uint32_t v467_flagWord(flagWordFirst);
          uint32_t v468_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[static_cast<size_t>(elementIndices[v12_batchId1]) * 81 + 0 + 12 + m2_extraOffset];
            float *const __restrict__ wrap_glb_m0 = &m0[static_cast<size_t>(elementIndices[v12_batchId1]) * 2 + 1][0 + m0_extraOffset];
            size_t v493_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v493_flagAheadAhead < numElements0) ? v493_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v467_flagWord);
            const bool allowed_next = static_cast<bool>(v468_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[static_cast<size_t>(elementIndices[v9_batchId0]) * 2 + 1][0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 288> r1(0.0f);
              // ir1 = +(glb_m1 * s0)
              // [(0, 20), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 288> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v26_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v26_data_part;
              v26_data_part.copy_from(glb_m1 + (0_i32));
              v26_data.template select<20, 1>(0) = v26_data_part;
              tensorforge::intel_esimd::simd<float, 96> s0_w0 = tensorforge::slmLoad<float, 96>(s0 + 0);
              float v27_data = s0_w0[0];
              float v32_data = s0_w0[9];
              float v37_data = s0_w0[18];
              float v42_data = s0_w0[27];
              float v47_data = s0_w0[36];
              float v52_data = s0_w0[45];
              float v57_data = s0_w0[54];
              float v62_data = s0_w0[63];
              float v67_data = s0_w0[72];
              tensorforge::intel_esimd::simd<float, 32> v72_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v72_data_part;
              v72_data_part.copy_from(glb_m1 + (20_i32));
              v72_data.template select<20, 1>(0) = v72_data_part;
              float v73_data = s0_w0[1];
              float v78_data = s0_w0[10];
              float v83_data = s0_w0[19];
              float v88_data = s0_w0[28];
              float v93_data = s0_w0[37];
              float v98_data = s0_w0[46];
              float v103_data = s0_w0[55];
              float v108_data = s0_w0[64];
              float v113_data = s0_w0[73];
              tensorforge::intel_esimd::simd<float, 32> v118_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v118_data_part;
              v118_data_part.copy_from(glb_m1 + (40_i32));
              v118_data.template select<20, 1>(0) = v118_data_part;
              float v119_data = s0_w0[2];
              float v124_data = s0_w0[11];
              float v129_data = s0_w0[20];
              float v134_data = s0_w0[29];
              float v139_data = s0_w0[38];
              float v144_data = s0_w0[47];
              float v149_data = s0_w0[56];
              float v154_data = s0_w0[65];
              float v159_data = s0_w0[74];
              tensorforge::intel_esimd::simd<float, 32> v164_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v164_data_part;
              v164_data_part.copy_from(glb_m1 + (60_i32));
              v164_data.template select<20, 1>(0) = v164_data_part;
              float v165_data = s0_w0[3];
              float v170_data = s0_w0[12];
              float v175_data = s0_w0[21];
              float v180_data = s0_w0[30];
              float v185_data = s0_w0[39];
              float v190_data = s0_w0[48];
              float v195_data = s0_w0[57];
              float v200_data = s0_w0[66];
              float v205_data = s0_w0[75];
              tensorforge::intel_esimd::simd<float, 32> v210_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v210_data_part;
              v210_data_part.copy_from(glb_m1 + (80_i32));
              v210_data.template select<20, 1>(0) = v210_data_part;
              float v211_data = s0_w0[4];
              float v216_data = s0_w0[13];
              float v221_data = s0_w0[22];
              float v226_data = s0_w0[31];
              float v231_data = s0_w0[40];
              float v236_data = s0_w0[49];
              float v241_data = s0_w0[58];
              float v246_data = s0_w0[67];
              float v251_data = s0_w0[76];
              tensorforge::intel_esimd::simd<float, 32> v256_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v256_data_part;
              v256_data_part.copy_from(glb_m1 + (100_i32));
              v256_data.template select<20, 1>(0) = v256_data_part;
              float v257_data = s0_w0[5];
              float v262_data = s0_w0[14];
              float v267_data = s0_w0[23];
              float v272_data = s0_w0[32];
              float v277_data = s0_w0[41];
              float v282_data = s0_w0[50];
              float v287_data = s0_w0[59];
              float v292_data = s0_w0[68];
              float v297_data = s0_w0[77];
              tensorforge::intel_esimd::simd<float, 32> v302_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v302_data_part;
              v302_data_part.copy_from(glb_m1 + (120_i32));
              v302_data.template select<20, 1>(0) = v302_data_part;
              float v303_data = s0_w0[6];
              float v308_data = s0_w0[15];
              float v313_data = s0_w0[24];
              float v318_data = s0_w0[33];
              float v323_data = s0_w0[42];
              float v328_data = s0_w0[51];
              float v333_data = s0_w0[60];
              float v338_data = s0_w0[69];
              float v343_data = s0_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v348_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v348_data_part;
              v348_data_part.copy_from(glb_m1 + (140_i32));
              v348_data.template select<20, 1>(0) = v348_data_part;
              float v349_data = s0_w0[7];
              float v354_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v357_p = (((((((v26_data * v32_data) + (v72_data * v78_data)) + (v118_data * v124_data)) + (v164_data * v170_data)) + (v210_data * v216_data)) + (v256_data * v262_data)) + (v302_data * v308_data)) + (v348_data * v354_data);
              float v359_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v362_p = (((((((v26_data * v37_data) + (v72_data * v83_data)) + (v118_data * v129_data)) + (v164_data * v175_data)) + (v210_data * v221_data)) + (v256_data * v267_data)) + (v302_data * v313_data)) + (v348_data * v359_data);
              float v364_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v367_p = (((((((v26_data * v42_data) + (v72_data * v88_data)) + (v118_data * v134_data)) + (v164_data * v180_data)) + (v210_data * v226_data)) + (v256_data * v272_data)) + (v302_data * v318_data)) + (v348_data * v364_data);
              float v369_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v372_p = (((((((v26_data * v47_data) + (v72_data * v93_data)) + (v118_data * v139_data)) + (v164_data * v185_data)) + (v210_data * v231_data)) + (v256_data * v277_data)) + (v302_data * v323_data)) + (v348_data * v369_data);
              float v374_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v377_p = (((((((v26_data * v52_data) + (v72_data * v98_data)) + (v118_data * v144_data)) + (v164_data * v190_data)) + (v210_data * v236_data)) + (v256_data * v282_data)) + (v302_data * v328_data)) + (v348_data * v374_data);
              float v379_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 32> v382_p = (((((((v26_data * v57_data) + (v72_data * v103_data)) + (v118_data * v149_data)) + (v164_data * v195_data)) + (v210_data * v241_data)) + (v256_data * v287_data)) + (v302_data * v333_data)) + (v348_data * v379_data);
              float v384_data = s0_w0[70];
              tensorforge::intel_esimd::simd<float, 32> v387_p = (((((((v26_data * v62_data) + (v72_data * v108_data)) + (v118_data * v154_data)) + (v164_data * v200_data)) + (v210_data * v246_data)) + (v256_data * v292_data)) + (v302_data * v338_data)) + (v348_data * v384_data);
              float v389_data = s0_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v392_p = (((((((v26_data * v67_data) + (v72_data * v113_data)) + (v118_data * v159_data)) + (v164_data * v205_data)) + (v210_data * v251_data)) + (v256_data * v297_data)) + (v302_data * v343_data)) + (v348_data * v389_data);
              tensorforge::intel_esimd::simd<float, 32> v394_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v394_data_part;
              v394_data_part.copy_from(glb_m1 + (160_i32));
              v394_data.template select<20, 1>(0) = v394_data_part;
              float v395_data = s0_w0[8];
              ir1.template select<32, 1>(0) = (((((((((v26_data * v27_data) + (v72_data * v73_data)) + (v118_data * v119_data)) + (v164_data * v165_data)) + (v210_data * v211_data)) + (v256_data * v257_data)) + (v302_data * v303_data)) + (v348_data * v349_data)) + (v394_data * v395_data));
              float v400_data = s0_w0[17];
              ir1.template select<32, 1>(32) = (v357_p + (v394_data * v400_data));
              float v405_data = s0_w0[26];
              ir1.template select<32, 1>(64) = (v362_p + (v394_data * v405_data));
              float v410_data = s0_w0[35];
              ir1.template select<32, 1>(96) = (v367_p + (v394_data * v410_data));
              float v415_data = s0_w0[44];
              ir1.template select<32, 1>(128) = (v372_p + (v394_data * v415_data));
              float v420_data = s0_w0[53];
              ir1.template select<32, 1>(160) = (v377_p + (v394_data * v420_data));
              float v425_data = s0_w0[62];
              ir1.template select<32, 1>(192) = (v382_p + (v394_data * v425_data));
              float v430_data = s0_w0[71];
              ir1.template select<32, 1>(224) = (v387_p + (v394_data * v430_data));
              float v435_data = s0_w0[80];
              ir1.template select<32, 1>(256) = (v392_p + (v394_data * v435_data));
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v448_n1 = 0; v448_n1 < 9; ++v448_n1) {
                int32_t v449_a = v448_n1 * 32;
                tensorforge::intel_esimd::simd<float, 20> v451_data(ir1.template select<20, 1>(v449_a));
                tensorforge::intel_esimd::simd<float, 20> v452_data(r0.template select<20, 1>(v449_a));
                r1.template select<20, 1>(v449_a) = (v452_data + v451_data);
              }
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m0);
                #pragma unroll
                for (int32_t v487_i1 = 0; v487_i1 < 9; ++v487_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v489_data(0.0f);
                  tensorforge::intel_esimd::simd<float, 20> v489_data_part;
                  v489_data_part.copy_from(wrap_glb_m0 + ((v487_i1 * 20)));
                  v489_data.template select<20, 1>(0) = v489_data_part;
                  r0.template select<32, 1>((v487_i1 * 32)) = v489_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v454_i1 = 0; v454_i1 < 9; ++v454_i1) {
                tensorforge::intel_esimd::simd<float, 20> v457_data(r1.template select<20, 1>((v454_i1 * 32)));
                v457_data.copy_to(glb_m0 + ((v454_i1 * 20)));
              }
            }
            else {
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m0);
                #pragma unroll
                for (int32_t v479_i1 = 0; v479_i1 < 9; ++v479_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v481_data(0.0f);
                  tensorforge::intel_esimd::simd<float, 20> v481_data_part;
                  v481_data_part.copy_from(wrap_glb_m0 + ((v479_i1 * 20)));
                  v481_data.template select<20, 1>(0) = v481_data_part;
                  r0.template select<32, 1>((v479_i1 * 32)) = v481_data;
                }
              }
            }
            if (allowed_next) {
              // s0 = load{g>s}(wrap_glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v475_ld;
              v475_ld.copy_from(wrap_glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 2 * 0 + 0), v475_ld);
              tensorforge::intel_esimd::simd<float, 17> v476_ld;
              v476_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 17>(s0 + (0 + 0 + 1 * 0 + 64), v476_ld);
            }
            v467_flagWord = v468_flagWordNext;
            v468_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


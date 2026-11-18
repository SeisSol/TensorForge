// === base name ===
kernel_78ab7f0787ff8da4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_78ab7f0787ff8da4 = {{1, 16, 1}, 16, 9, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_78ab7f0787ff8da4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_78ab7f0787ff8da4(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_78ab7f0787ff8da4(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1664 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_78ab7f0787ff8da4(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_78ab7f0787ff8da4(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_78ab7f0787ff8da4(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_78ab7f0787ff8da4_k;
inline void kernel_kernel_78ab7f0787ff8da4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_78ab7f0787ff8da4_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1664 * sizeof(double)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (9 active) x 16 per block = block 1x16x1, 13312 B shared, occupancy grid
        // operands:
        //   m0 9×9(9×9) {0..9}×{0..9} indexed
        //   m1 9×9(9×9) {0..9}×{0..9} indexed(offset=8)
        //   m2 9×9(9×9) {0..9}×{0..9} indexed(stride=96)
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":9,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1664}],"shared_bytes":13312,"shared_elements":1664,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"indexed","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"indexed(offset=8)","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"indexed(stride=96)","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"indexed","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"indexed(offset=8)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"indexed(stride=96)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (104 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          tensorforge::intel_esimd::simd<double, 144> r0(0.0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v452_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v452_flagSecondAhead < numElements0) ? v452_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const double *const __restrict__ peel_glb_m1 = &m1[static_cast<size_t>(elementIndices[v3_batchId1]) * 81 + 0 + 8 + m1_extraOffset];
          if (allowed_peel) {
            // r0 = load{g>r}(peel_glb_m1);
            #pragma unroll
            for (int32_t v466_i1 = 0; v466_i1 < 9; ++v466_i1) {
              tensorforge::intel_esimd::simd<double, 16> v468_data(0.0);
              tensorforge::intel_esimd::simd<double, 9> v468_data_part;
              v468_data_part.copy_from(peel_glb_m1 + ((v466_i1 * 9)));
              v468_data.template select<9, 1>(0) = v468_data_part;
              r0.template select<16, 1>((v466_i1 * 16)) = v468_data;
            }
          }
          const double *const __restrict__ peel_glb_m2 = &m2[static_cast<size_t>(elementIndices[v3_batchId1]) * 96 + 0 + m2_extraOffset];
          if (allowed_peel) {
            // s0 = load{g>s}(peel_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<double, 32> v476_ld;
            v476_ld.copy_from(peel_glb_m2 + (0 + 0 + 2 * 0 + 0));
            tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v476_ld);
            tensorforge::intel_esimd::simd<double, 32> v477_ld;
            v477_ld.copy_from(peel_glb_m2 + (0 + 0 + 2 * 0 + 32));
            tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v477_ld);
            tensorforge::intel_esimd::simd<double, 16> v478_ld;
            v478_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 64));
            tensorforge::slmStore<double, 16>(s0 + (0 + 0 + 1 * 0 + 64), v478_ld);
            double v479_ld = peel_glb_m2[0 + 0 + 1 * 0 + 80];
            s0[0 + 0 + 1 * 0 + 80] = v479_ld;
          }
          uint32_t v456_flagWord(flagWordFirst);
          uint32_t v457_flagWordNext(flagWordSecond);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const double *const __restrict__ wrap_glb_m1 = &m1[static_cast<size_t>(elementIndices[v11_batchId1]) * 81 + 0 + 8 + m1_extraOffset];
            const double *const __restrict__ wrap_glb_m2 = &m2[static_cast<size_t>(elementIndices[v11_batchId1]) * 96 + 0 + m2_extraOffset];
            size_t v486_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v486_flagAheadAhead < numElements0) ? v486_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v456_flagWord);
            const bool allowed_next = static_cast<bool>(v457_flagWordNext);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[static_cast<size_t>(elementIndices[v8_batchId0]) * 81 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<double, 144> r1(0.0);
              // ir1 = +(r0 * s0)
              // [(0, 9), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<double, 144> ir1(0.0);
              tensorforge::intel_esimd::simd<double, 16> v34_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 80> s0_w0 = tensorforge::slmLoad<double, 80>(s0 + 0);
              double v35_data = s0_w0[0];
              double v40_data = s0_w0[9];
              double v45_data = s0_w0[18];
              double v50_data = s0_w0[27];
              double v55_data = s0_w0[36];
              double v60_data = s0_w0[45];
              double v65_data = s0_w0[54];
              double v70_data = s0_w0[63];
              double v75_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v79_data(r0.template select<16, 1>(16));
              double v80_data = s0_w0[1];
              double v85_data = s0_w0[10];
              double v90_data = s0_w0[19];
              double v95_data = s0_w0[28];
              double v100_data = s0_w0[37];
              double v105_data = s0_w0[46];
              double v110_data = s0_w0[55];
              double v115_data = s0_w0[64];
              double v120_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v124_data(r0.template select<16, 1>(32));
              double v125_data = s0_w0[2];
              double v130_data = s0_w0[11];
              double v135_data = s0_w0[20];
              double v140_data = s0_w0[29];
              double v145_data = s0_w0[38];
              double v150_data = s0_w0[47];
              double v155_data = s0_w0[56];
              double v160_data = s0_w0[65];
              double v165_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v169_data(r0.template select<16, 1>(48));
              double v170_data = s0_w0[3];
              double v175_data = s0_w0[12];
              double v180_data = s0_w0[21];
              double v185_data = s0_w0[30];
              double v190_data = s0_w0[39];
              double v195_data = s0_w0[48];
              double v200_data = s0_w0[57];
              double v205_data = s0_w0[66];
              double v210_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v214_data(r0.template select<16, 1>(64));
              double v215_data = s0_w0[4];
              double v220_data = s0_w0[13];
              double v225_data = s0_w0[22];
              double v230_data = s0_w0[31];
              double v235_data = s0_w0[40];
              double v240_data = s0_w0[49];
              double v245_data = s0_w0[58];
              double v250_data = s0_w0[67];
              double v255_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v259_data(r0.template select<16, 1>(80));
              double v260_data = s0_w0[5];
              double v265_data = s0_w0[14];
              double v270_data = s0_w0[23];
              double v275_data = s0_w0[32];
              double v280_data = s0_w0[41];
              double v285_data = s0_w0[50];
              double v290_data = s0_w0[59];
              double v295_data = s0_w0[68];
              double v300_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v304_data(r0.template select<16, 1>(96));
              double v305_data = s0_w0[6];
              double v310_data = s0_w0[15];
              double v315_data = s0_w0[24];
              double v320_data = s0_w0[33];
              double v325_data = s0_w0[42];
              double v330_data = s0_w0[51];
              double v335_data = s0_w0[60];
              double v340_data = s0_w0[69];
              double v345_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v349_data(r0.template select<16, 1>(112));
              double v350_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v353_p = (((((((v34_data * v35_data) + (v79_data * v80_data)) + (v124_data * v125_data)) + (v169_data * v170_data)) + (v214_data * v215_data)) + (v259_data * v260_data)) + (v304_data * v305_data)) + (v349_data * v350_data);
              double v355_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v358_p = (((((((v34_data * v40_data) + (v79_data * v85_data)) + (v124_data * v130_data)) + (v169_data * v175_data)) + (v214_data * v220_data)) + (v259_data * v265_data)) + (v304_data * v310_data)) + (v349_data * v355_data);
              double v360_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v363_p = (((((((v34_data * v45_data) + (v79_data * v90_data)) + (v124_data * v135_data)) + (v169_data * v180_data)) + (v214_data * v225_data)) + (v259_data * v270_data)) + (v304_data * v315_data)) + (v349_data * v360_data);
              double v365_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v368_p = (((((((v34_data * v50_data) + (v79_data * v95_data)) + (v124_data * v140_data)) + (v169_data * v185_data)) + (v214_data * v230_data)) + (v259_data * v275_data)) + (v304_data * v320_data)) + (v349_data * v365_data);
              double v370_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v373_p = (((((((v34_data * v55_data) + (v79_data * v100_data)) + (v124_data * v145_data)) + (v169_data * v190_data)) + (v214_data * v235_data)) + (v259_data * v280_data)) + (v304_data * v325_data)) + (v349_data * v370_data);
              double v375_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v378_p = (((((((v34_data * v60_data) + (v79_data * v105_data)) + (v124_data * v150_data)) + (v169_data * v195_data)) + (v214_data * v240_data)) + (v259_data * v285_data)) + (v304_data * v330_data)) + (v349_data * v375_data);
              double v380_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v383_p = (((((((v34_data * v65_data) + (v79_data * v110_data)) + (v124_data * v155_data)) + (v169_data * v200_data)) + (v214_data * v245_data)) + (v259_data * v290_data)) + (v304_data * v335_data)) + (v349_data * v380_data);
              double v385_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v388_p = (((((((v34_data * v70_data) + (v79_data * v115_data)) + (v124_data * v160_data)) + (v169_data * v205_data)) + (v214_data * v250_data)) + (v259_data * v295_data)) + (v304_data * v340_data)) + (v349_data * v385_data);
              double v390_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v393_p = (((((((v34_data * v75_data) + (v79_data * v120_data)) + (v124_data * v165_data)) + (v169_data * v210_data)) + (v214_data * v255_data)) + (v259_data * v300_data)) + (v304_data * v345_data)) + (v349_data * v390_data);
              tensorforge::intel_esimd::simd<double, 16> v394_data(r0.template select<16, 1>(128));
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m1);
                #pragma unroll
                for (int32_t v470_i1 = 0; v470_i1 < 9; ++v470_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v472_data(0.0);
                  tensorforge::intel_esimd::simd<double, 9> v472_data_part;
                  v472_data_part.copy_from(wrap_glb_m1 + ((v470_i1 * 9)));
                  v472_data.template select<9, 1>(0) = v472_data_part;
                  r0.template select<16, 1>((v470_i1 * 16)) = v472_data;
                }
              }
              double v395_data = s0[8];
              ir1.template select<16, 1>(0) = (v353_p + (v394_data * v395_data));
              double v400_data = s0[17];
              ir1.template select<16, 1>(16) = (v358_p + (v394_data * v400_data));
              double v405_data = s0[26];
              ir1.template select<16, 1>(32) = (v363_p + (v394_data * v405_data));
              double v410_data = s0[35];
              ir1.template select<16, 1>(48) = (v368_p + (v394_data * v410_data));
              double v415_data = s0[44];
              ir1.template select<16, 1>(64) = (v373_p + (v394_data * v415_data));
              double v420_data = s0[53];
              ir1.template select<16, 1>(80) = (v378_p + (v394_data * v420_data));
              double v425_data = s0[62];
              ir1.template select<16, 1>(96) = (v383_p + (v394_data * v425_data));
              double v430_data = s0[71];
              ir1.template select<16, 1>(112) = (v388_p + (v394_data * v430_data));
              double v435_data = s0[80];
              ir1.template select<16, 1>(128) = (v393_p + (v394_data * v435_data));
              // r1 = ir1
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v443_i1 = 0; v443_i1 < 9; ++v443_i1) {
                tensorforge::intel_esimd::simd<double, 9> v446_data(ir1.template select<9, 1>((v443_i1 * 16)));
                v446_data.copy_to(glb_m0 + ((v443_i1 * 9)));
              }
            }
            else {
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m1);
                #pragma unroll
                for (int32_t v462_i1 = 0; v462_i1 < 9; ++v462_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v464_data(0.0);
                  tensorforge::intel_esimd::simd<double, 9> v464_data_part;
                  v464_data_part.copy_from(wrap_glb_m1 + ((v462_i1 * 9)));
                  v464_data.template select<9, 1>(0) = v464_data_part;
                  r0.template select<16, 1>((v462_i1 * 16)) = v464_data;
                }
              }
            }
            if (allowed_next) {
              // s0 = load{g>s}(wrap_glb_m2[0, 1])
              tensorforge::intel_esimd::simd<double, 32> v480_ld;
              v480_ld.copy_from(wrap_glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v480_ld);
              tensorforge::intel_esimd::simd<double, 32> v481_ld;
              v481_ld.copy_from(wrap_glb_m2 + (0 + 0 + 2 * 0 + 32));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v481_ld);
              tensorforge::intel_esimd::simd<double, 16> v482_ld;
              v482_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<double, 16>(s0 + (0 + 0 + 1 * 0 + 64), v482_ld);
              double v483_ld = wrap_glb_m2[0 + 0 + 1 * 0 + 80];
              s0[0 + 0 + 1 * 0 + 80] = v483_ld;
            }
            v456_flagWord = v457_flagWordNext;
            v457_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


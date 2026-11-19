// === base name ===
kernel_d694a8bf391f9b5c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d694a8bf391f9b5c = {{1, 16, 1}, 16, 9, 1, 16, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d694a8bf391f9b5c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d694a8bf391f9b5c(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d694a8bf391f9b5c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_d694a8bf391f9b5c(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d694a8bf391f9b5c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d694a8bf391f9b5c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_d694a8bf391f9b5c_k;
inline void kernel_kernel_d694a8bf391f9b5c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_d694a8bf391f9b5c_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
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
          size_t v453_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v453_flagSecondAhead < numElements0) ? v453_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
          const double *const __restrict__ peel_glb_m1 = &m1[peel_elementIndex * 81 + 0 + 8 + m1_extraOffset];
          if (allowed_peel) {
            // r0 = load{g>r}(peel_glb_m1);
            #pragma unroll
            for (int32_t v469_i1 = 0; v469_i1 < 9; ++v469_i1) {
              tensorforge::intel_esimd::simd<double, 16> v471_data(0.0);
              tensorforge::intel_esimd::simd<double, 9> v471_data_part;
              v471_data_part.copy_from(peel_glb_m1 + ((v469_i1 * 9)));
              v471_data.template select<9, 1>(0) = v471_data_part;
              r0.template select<16, 1>((v469_i1 * 16)) = v471_data;
            }
          }
          const double *const __restrict__ peel_glb_m2 = &m2[peel_elementIndex * 96 + 0 + m2_extraOffset];
          if (allowed_peel) {
            // s0 = load{g>s}(peel_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<double, 32> v479_ld;
            v479_ld.copy_from(peel_glb_m2 + (0 + 0 + 2 * 0 + 0));
            tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v479_ld);
            tensorforge::intel_esimd::simd<double, 32> v480_ld;
            v480_ld.copy_from(peel_glb_m2 + (0 + 0 + 2 * 0 + 32));
            tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v480_ld);
            tensorforge::intel_esimd::simd<double, 16> v481_ld;
            v481_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 64));
            tensorforge::slmStore<double, 16>(s0 + (0 + 0 + 1 * 0 + 64), v481_ld);
            double v482_ld = peel_glb_m2[0 + 0 + 1 * 0 + 80];
            s0[0 + 0 + 1 * 0 + 80] = v482_ld;
          }
          uint32_t v457_flagWord(flagWordFirst);
          uint32_t v458_flagWordNext(flagWordSecond);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v11_batchId1]);
            const double *const __restrict__ wrap_glb_m1 = &m1[wrap_elementIndex * 81 + 0 + 8 + m1_extraOffset];
            const double *const __restrict__ wrap_glb_m2 = &m2[wrap_elementIndex * 96 + 0 + m2_extraOffset];
            size_t v489_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v489_flagAheadAhead < numElements0) ? v489_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v457_flagWord);
            const bool allowed_next = static_cast<bool>(v458_flagWordNext);
            if (allowed) {
              const size_t elementIndex = static_cast<size_t>(elementIndices[v8_batchId0]);
              double *const __restrict__ glb_m0 = &m0[elementIndex * 81 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<double, 144> r1(0.0);
              // ir1 = +(r0 * s0)
              // [(0, 9), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<double, 144> ir1(0.0);
              tensorforge::intel_esimd::simd<double, 16> v35_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 80> s0_w0 = tensorforge::slmLoad<double, 80>(s0 + 0);
              double v36_data = s0_w0[0];
              double v41_data = s0_w0[9];
              double v46_data = s0_w0[18];
              double v51_data = s0_w0[27];
              double v56_data = s0_w0[36];
              double v61_data = s0_w0[45];
              double v66_data = s0_w0[54];
              double v71_data = s0_w0[63];
              double v76_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v80_data(r0.template select<16, 1>(16));
              double v81_data = s0_w0[1];
              double v86_data = s0_w0[10];
              double v91_data = s0_w0[19];
              double v96_data = s0_w0[28];
              double v101_data = s0_w0[37];
              double v106_data = s0_w0[46];
              double v111_data = s0_w0[55];
              double v116_data = s0_w0[64];
              double v121_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v125_data(r0.template select<16, 1>(32));
              double v126_data = s0_w0[2];
              double v131_data = s0_w0[11];
              double v136_data = s0_w0[20];
              double v141_data = s0_w0[29];
              double v146_data = s0_w0[38];
              double v151_data = s0_w0[47];
              double v156_data = s0_w0[56];
              double v161_data = s0_w0[65];
              double v166_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v170_data(r0.template select<16, 1>(48));
              double v171_data = s0_w0[3];
              double v176_data = s0_w0[12];
              double v181_data = s0_w0[21];
              double v186_data = s0_w0[30];
              double v191_data = s0_w0[39];
              double v196_data = s0_w0[48];
              double v201_data = s0_w0[57];
              double v206_data = s0_w0[66];
              double v211_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v215_data(r0.template select<16, 1>(64));
              double v216_data = s0_w0[4];
              double v221_data = s0_w0[13];
              double v226_data = s0_w0[22];
              double v231_data = s0_w0[31];
              double v236_data = s0_w0[40];
              double v241_data = s0_w0[49];
              double v246_data = s0_w0[58];
              double v251_data = s0_w0[67];
              double v256_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v260_data(r0.template select<16, 1>(80));
              double v261_data = s0_w0[5];
              double v266_data = s0_w0[14];
              double v271_data = s0_w0[23];
              double v276_data = s0_w0[32];
              double v281_data = s0_w0[41];
              double v286_data = s0_w0[50];
              double v291_data = s0_w0[59];
              double v296_data = s0_w0[68];
              double v301_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v305_data(r0.template select<16, 1>(96));
              double v306_data = s0_w0[6];
              double v311_data = s0_w0[15];
              double v316_data = s0_w0[24];
              double v321_data = s0_w0[33];
              double v326_data = s0_w0[42];
              double v331_data = s0_w0[51];
              double v336_data = s0_w0[60];
              double v341_data = s0_w0[69];
              double v346_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v350_data(r0.template select<16, 1>(112));
              double v351_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v354_p = (((((((v35_data * v36_data) + (v80_data * v81_data)) + (v125_data * v126_data)) + (v170_data * v171_data)) + (v215_data * v216_data)) + (v260_data * v261_data)) + (v305_data * v306_data)) + (v350_data * v351_data);
              double v356_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v359_p = (((((((v35_data * v41_data) + (v80_data * v86_data)) + (v125_data * v131_data)) + (v170_data * v176_data)) + (v215_data * v221_data)) + (v260_data * v266_data)) + (v305_data * v311_data)) + (v350_data * v356_data);
              double v361_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v364_p = (((((((v35_data * v46_data) + (v80_data * v91_data)) + (v125_data * v136_data)) + (v170_data * v181_data)) + (v215_data * v226_data)) + (v260_data * v271_data)) + (v305_data * v316_data)) + (v350_data * v361_data);
              double v366_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v369_p = (((((((v35_data * v51_data) + (v80_data * v96_data)) + (v125_data * v141_data)) + (v170_data * v186_data)) + (v215_data * v231_data)) + (v260_data * v276_data)) + (v305_data * v321_data)) + (v350_data * v366_data);
              double v371_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v374_p = (((((((v35_data * v56_data) + (v80_data * v101_data)) + (v125_data * v146_data)) + (v170_data * v191_data)) + (v215_data * v236_data)) + (v260_data * v281_data)) + (v305_data * v326_data)) + (v350_data * v371_data);
              double v376_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v379_p = (((((((v35_data * v61_data) + (v80_data * v106_data)) + (v125_data * v151_data)) + (v170_data * v196_data)) + (v215_data * v241_data)) + (v260_data * v286_data)) + (v305_data * v331_data)) + (v350_data * v376_data);
              double v381_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v384_p = (((((((v35_data * v66_data) + (v80_data * v111_data)) + (v125_data * v156_data)) + (v170_data * v201_data)) + (v215_data * v246_data)) + (v260_data * v291_data)) + (v305_data * v336_data)) + (v350_data * v381_data);
              double v386_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v389_p = (((((((v35_data * v71_data) + (v80_data * v116_data)) + (v125_data * v161_data)) + (v170_data * v206_data)) + (v215_data * v251_data)) + (v260_data * v296_data)) + (v305_data * v341_data)) + (v350_data * v386_data);
              double v391_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v394_p = (((((((v35_data * v76_data) + (v80_data * v121_data)) + (v125_data * v166_data)) + (v170_data * v211_data)) + (v215_data * v256_data)) + (v260_data * v301_data)) + (v305_data * v346_data)) + (v350_data * v391_data);
              tensorforge::intel_esimd::simd<double, 16> v395_data(r0.template select<16, 1>(128));
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m1);
                #pragma unroll
                for (int32_t v473_i1 = 0; v473_i1 < 9; ++v473_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v475_data(0.0);
                  tensorforge::intel_esimd::simd<double, 9> v475_data_part;
                  v475_data_part.copy_from(wrap_glb_m1 + ((v473_i1 * 9)));
                  v475_data.template select<9, 1>(0) = v475_data_part;
                  r0.template select<16, 1>((v473_i1 * 16)) = v475_data;
                }
              }
              double v396_data = s0[8];
              ir1.template select<16, 1>(0) = (v354_p + (v395_data * v396_data));
              double v401_data = s0[17];
              ir1.template select<16, 1>(16) = (v359_p + (v395_data * v401_data));
              double v406_data = s0[26];
              ir1.template select<16, 1>(32) = (v364_p + (v395_data * v406_data));
              double v411_data = s0[35];
              ir1.template select<16, 1>(48) = (v369_p + (v395_data * v411_data));
              double v416_data = s0[44];
              ir1.template select<16, 1>(64) = (v374_p + (v395_data * v416_data));
              double v421_data = s0[53];
              ir1.template select<16, 1>(80) = (v379_p + (v395_data * v421_data));
              double v426_data = s0[62];
              ir1.template select<16, 1>(96) = (v384_p + (v395_data * v426_data));
              double v431_data = s0[71];
              ir1.template select<16, 1>(112) = (v389_p + (v395_data * v431_data));
              double v436_data = s0[80];
              ir1.template select<16, 1>(128) = (v394_p + (v395_data * v436_data));
              // r1 = ir1
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v444_i1 = 0; v444_i1 < 9; ++v444_i1) {
                tensorforge::intel_esimd::simd<double, 9> v447_data(ir1.template select<9, 1>((v444_i1 * 16)));
                v447_data.copy_to(glb_m0 + ((v444_i1 * 9)));
              }
            }
            else {
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m1);
                #pragma unroll
                for (int32_t v465_i1 = 0; v465_i1 < 9; ++v465_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v467_data(0.0);
                  tensorforge::intel_esimd::simd<double, 9> v467_data_part;
                  v467_data_part.copy_from(wrap_glb_m1 + ((v465_i1 * 9)));
                  v467_data.template select<9, 1>(0) = v467_data_part;
                  r0.template select<16, 1>((v465_i1 * 16)) = v467_data;
                }
              }
            }
            if (allowed_next) {
              // s0 = load{g>s}(wrap_glb_m2[0, 1])
              tensorforge::intel_esimd::simd<double, 32> v483_ld;
              v483_ld.copy_from(wrap_glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v483_ld);
              tensorforge::intel_esimd::simd<double, 32> v484_ld;
              v484_ld.copy_from(wrap_glb_m2 + (0 + 0 + 2 * 0 + 32));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v484_ld);
              tensorforge::intel_esimd::simd<double, 16> v485_ld;
              v485_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<double, 16>(s0 + (0 + 0 + 1 * 0 + 64), v485_ld);
              double v486_ld = wrap_glb_m2[0 + 0 + 1 * 0 + 80];
              s0[0 + 0 + 1 * 0 + 80] = v486_ld;
            }
            v457_flagWord = v458_flagWordNext;
            v458_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


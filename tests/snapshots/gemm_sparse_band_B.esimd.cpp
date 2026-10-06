// === base name ===
kernel_2b3b0a1ef08a996d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2b3b0a1ef08a996d = {{1, 16, 1}, 16, 16, 1, 16, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2b3b0a1ef08a996d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2b3b0a1ef08a996d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2b3b0a1ef08a996d(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1024 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_2b3b0a1ef08a996d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2b3b0a1ef08a996d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2b3b0a1ef08a996d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2b3b0a1ef08a996d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1024 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 4096 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1024}],"shared_bytes":4096,"shared_elements":1024,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (64 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (48);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 46 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                int32_t v25_lead = v23_i0 * 16;
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 16; ++v24_i1) {
                  int32_t v28_a = v25_lead + (v24_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v29_data;
                  v29_data.copy_from(glb_m1 + (v28_a));
                  r0.template select<16, 1>(v28_a) = v29_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<float, 14> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 14>(s0 + (0 + 0 + 1 * 0 + 32), v32_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(0));
              float v36_data = s0[0];
              tensorforge::intel_esimd::simd<float, 16> v38_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v38_data + (v35_data * v36_data));
              float v41_data = s0[2];
              tensorforge::intel_esimd::simd<float, 16> v43_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v43_data + (v35_data * v41_data));
              tensorforge::intel_esimd::simd<float, 16> v59_data(r0.template select<16, 1>(16));
              float v60_data = s0[1];
              tensorforge::intel_esimd::simd<float, 16> v62_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v62_data + (v59_data * v60_data));
              float v65_data = s0[3];
              tensorforge::intel_esimd::simd<float, 16> v67_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v67_data + (v59_data * v65_data));
              float v70_data = s0[5];
              tensorforge::intel_esimd::simd<float, 16> v72_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v72_data + (v59_data * v70_data));
              tensorforge::intel_esimd::simd<float, 16> v87_data(r0.template select<16, 1>(32));
              float v89_data = s0[4];
              tensorforge::intel_esimd::simd<float, 16> v91_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v91_data + (v87_data * v89_data));
              float v94_data = s0[6];
              tensorforge::intel_esimd::simd<float, 16> v96_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v96_data + (v87_data * v94_data));
              float v99_data = s0[8];
              tensorforge::intel_esimd::simd<float, 16> v101_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v101_data + (v87_data * v99_data));
              tensorforge::intel_esimd::simd<float, 16> v115_data(r0.template select<16, 1>(48));
              float v118_data = s0[7];
              tensorforge::intel_esimd::simd<float, 16> v120_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v120_data + (v115_data * v118_data));
              float v123_data = s0[9];
              tensorforge::intel_esimd::simd<float, 16> v125_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v125_data + (v115_data * v123_data));
              float v128_data = s0[11];
              tensorforge::intel_esimd::simd<float, 16> v130_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v130_data + (v115_data * v128_data));
              tensorforge::intel_esimd::simd<float, 16> v143_data(r0.template select<16, 1>(64));
              float v147_data = s0[10];
              tensorforge::intel_esimd::simd<float, 16> v149_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v149_data + (v143_data * v147_data));
              float v152_data = s0[12];
              tensorforge::intel_esimd::simd<float, 16> v154_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v154_data + (v143_data * v152_data));
              float v157_data = s0[14];
              tensorforge::intel_esimd::simd<float, 16> v159_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v159_data + (v143_data * v157_data));
              tensorforge::intel_esimd::simd<float, 16> v171_data(r0.template select<16, 1>(80));
              float v176_data = s0[13];
              tensorforge::intel_esimd::simd<float, 16> v178_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v178_data + (v171_data * v176_data));
              float v181_data = s0[15];
              tensorforge::intel_esimd::simd<float, 16> v183_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v183_data + (v171_data * v181_data));
              float v186_data = s0[17];
              tensorforge::intel_esimd::simd<float, 16> v188_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v188_data + (v171_data * v186_data));
              tensorforge::intel_esimd::simd<float, 16> v199_data(r0.template select<16, 1>(96));
              float v205_data = s0[16];
              tensorforge::intel_esimd::simd<float, 16> v207_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v207_data + (v199_data * v205_data));
              float v210_data = s0[18];
              tensorforge::intel_esimd::simd<float, 16> v212_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v212_data + (v199_data * v210_data));
              float v215_data = s0[20];
              tensorforge::intel_esimd::simd<float, 16> v217_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v217_data + (v199_data * v215_data));
              tensorforge::intel_esimd::simd<float, 16> v227_data(r0.template select<16, 1>(112));
              float v234_data = s0[19];
              tensorforge::intel_esimd::simd<float, 16> v236_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v236_data + (v227_data * v234_data));
              float v239_data = s0[21];
              tensorforge::intel_esimd::simd<float, 16> v241_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v241_data + (v227_data * v239_data));
              float v244_data = s0[23];
              tensorforge::intel_esimd::simd<float, 16> v246_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v246_data + (v227_data * v244_data));
              tensorforge::intel_esimd::simd<float, 16> v255_data(r0.template select<16, 1>(128));
              float v263_data = s0[22];
              tensorforge::intel_esimd::simd<float, 16> v265_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v265_data + (v255_data * v263_data));
              float v268_data = s0[24];
              tensorforge::intel_esimd::simd<float, 16> v270_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v270_data + (v255_data * v268_data));
              float v273_data = s0[26];
              tensorforge::intel_esimd::simd<float, 16> v275_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v275_data + (v255_data * v273_data));
              tensorforge::intel_esimd::simd<float, 16> v283_data(r0.template select<16, 1>(144));
              float v292_data = s0[25];
              tensorforge::intel_esimd::simd<float, 16> v294_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v294_data + (v283_data * v292_data));
              float v297_data = s0[27];
              tensorforge::intel_esimd::simd<float, 16> v299_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v299_data + (v283_data * v297_data));
              float v302_data = s0[29];
              tensorforge::intel_esimd::simd<float, 16> v304_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v304_data + (v283_data * v302_data));
              tensorforge::intel_esimd::simd<float, 16> v311_data(r0.template select<16, 1>(160));
              float v321_data = s0[28];
              tensorforge::intel_esimd::simd<float, 16> v323_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v323_data + (v311_data * v321_data));
              float v326_data = s0[30];
              tensorforge::intel_esimd::simd<float, 16> v328_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v328_data + (v311_data * v326_data));
              float v331_data = s0[32];
              tensorforge::intel_esimd::simd<float, 16> v333_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v333_data + (v311_data * v331_data));
              tensorforge::intel_esimd::simd<float, 16> v339_data(r0.template select<16, 1>(176));
              float v350_data = s0[31];
              tensorforge::intel_esimd::simd<float, 16> v352_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v352_data + (v339_data * v350_data));
              float v355_data = s0[33];
              tensorforge::intel_esimd::simd<float, 16> v357_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v357_data + (v339_data * v355_data));
              float v360_data = s0[35];
              tensorforge::intel_esimd::simd<float, 16> v362_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v362_data + (v339_data * v360_data));
              tensorforge::intel_esimd::simd<float, 16> v367_data(r0.template select<16, 1>(192));
              float v379_data = s0[34];
              tensorforge::intel_esimd::simd<float, 16> v381_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v381_data + (v367_data * v379_data));
              float v384_data = s0[36];
              tensorforge::intel_esimd::simd<float, 16> v386_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v386_data + (v367_data * v384_data));
              float v389_data = s0[38];
              tensorforge::intel_esimd::simd<float, 16> v391_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v391_data + (v367_data * v389_data));
              tensorforge::intel_esimd::simd<float, 16> v395_data(r0.template select<16, 1>(208));
              float v408_data = s0[37];
              tensorforge::intel_esimd::simd<float, 16> v410_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v410_data + (v395_data * v408_data));
              float v413_data = s0[39];
              tensorforge::intel_esimd::simd<float, 16> v415_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v415_data + (v395_data * v413_data));
              float v418_data = s0[41];
              tensorforge::intel_esimd::simd<float, 16> v420_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v420_data + (v395_data * v418_data));
              tensorforge::intel_esimd::simd<float, 16> v423_data(r0.template select<16, 1>(224));
              float v437_data = s0[40];
              tensorforge::intel_esimd::simd<float, 16> v439_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v439_data + (v423_data * v437_data));
              float v442_data = s0[42];
              tensorforge::intel_esimd::simd<float, 16> v444_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v444_data + (v423_data * v442_data));
              float v447_data = s0[44];
              tensorforge::intel_esimd::simd<float, 16> v449_data(ir1.template select<16, 1>(240));
              ir1.template select<16, 1>(240) = (v449_data + (v423_data * v447_data));
              tensorforge::intel_esimd::simd<float, 16> v451_data(r0.template select<16, 1>(240));
              float v466_data = s0[43];
              tensorforge::intel_esimd::simd<float, 16> v468_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v468_data + (v451_data * v466_data));
              float v471_data = s0[45];
              tensorforge::intel_esimd::simd<float, 16> v473_data(ir1.template select<16, 1>(240));
              ir1.template select<16, 1>(240) = (v473_data + (v451_data * v471_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v475_n0 = 0; v475_n0 < 1; ++v475_n0) {
                int32_t v477_a = v475_n0 * 16;
                #pragma unroll
                for (int32_t v476_n1 = 0; v476_n1 < 16; ++v476_n1) {
                  int32_t v479_a = v477_a + (v476_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v480_data(ir1.template select<16, 1>(v479_a));
                  r1.template select<16, 1>(v479_a) = v480_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v481_i0 = 0; v481_i0 < 1; ++v481_i0) {
                int32_t v483_a = v481_i0 * 16;
                #pragma unroll
                for (int32_t v482_i1 = 0; v482_i1 < 16; ++v482_i1) {
                  int32_t v485_a = v483_a + (v482_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v486_data(r1.template select<16, 1>(v485_a));
                  v486_data.copy_to(glb_m0 + (v485_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


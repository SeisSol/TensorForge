// === base name ===
kernel_3c2ad036c83474e9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3c2ad036c83474e9 = {{1, 16, 1}, 16, 16, 1, 16, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3c2ad036c83474e9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3c2ad036c83474e9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3c2ad036c83474e9(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_3c2ad036c83474e9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3c2ad036c83474e9(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3c2ad036c83474e9(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3c2ad036c83474e9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 46 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 16;
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 16; ++v21_i1) {
                  int32_t v25_a = v22_lead + (v21_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v26_data;
                  v26_data.copy_from(glb_m1 + (v25_a));
                  r0.template select<16, 1>(v25_a) = v26_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 0), v28_ld);
              tensorforge::intel_esimd::simd<float, 14> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 14>(s0 + (0 + 0 + 1 * 0 + 32), v29_ld);
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(0));
              float v33_data = s0[0];
              tensorforge::intel_esimd::simd<float, 16> v35_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v35_data + (v32_data * v33_data));
              float v38_data = s0[2];
              tensorforge::intel_esimd::simd<float, 16> v40_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v40_data + (v32_data * v38_data));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(16));
              float v57_data = s0[1];
              tensorforge::intel_esimd::simd<float, 16> v59_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v59_data + (v56_data * v57_data));
              float v62_data = s0[3];
              tensorforge::intel_esimd::simd<float, 16> v64_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v64_data + (v56_data * v62_data));
              float v67_data = s0[5];
              tensorforge::intel_esimd::simd<float, 16> v69_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v69_data + (v56_data * v67_data));
              tensorforge::intel_esimd::simd<float, 16> v84_data(r0.template select<16, 1>(32));
              float v86_data = s0[4];
              tensorforge::intel_esimd::simd<float, 16> v88_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v88_data + (v84_data * v86_data));
              float v91_data = s0[6];
              tensorforge::intel_esimd::simd<float, 16> v93_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v93_data + (v84_data * v91_data));
              float v96_data = s0[8];
              tensorforge::intel_esimd::simd<float, 16> v98_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v98_data + (v84_data * v96_data));
              tensorforge::intel_esimd::simd<float, 16> v112_data(r0.template select<16, 1>(48));
              float v115_data = s0[7];
              tensorforge::intel_esimd::simd<float, 16> v117_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v117_data + (v112_data * v115_data));
              float v120_data = s0[9];
              tensorforge::intel_esimd::simd<float, 16> v122_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v122_data + (v112_data * v120_data));
              float v125_data = s0[11];
              tensorforge::intel_esimd::simd<float, 16> v127_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v127_data + (v112_data * v125_data));
              tensorforge::intel_esimd::simd<float, 16> v140_data(r0.template select<16, 1>(64));
              float v144_data = s0[10];
              tensorforge::intel_esimd::simd<float, 16> v146_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v146_data + (v140_data * v144_data));
              float v149_data = s0[12];
              tensorforge::intel_esimd::simd<float, 16> v151_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v151_data + (v140_data * v149_data));
              float v154_data = s0[14];
              tensorforge::intel_esimd::simd<float, 16> v156_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v156_data + (v140_data * v154_data));
              tensorforge::intel_esimd::simd<float, 16> v168_data(r0.template select<16, 1>(80));
              float v173_data = s0[13];
              tensorforge::intel_esimd::simd<float, 16> v175_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v175_data + (v168_data * v173_data));
              float v178_data = s0[15];
              tensorforge::intel_esimd::simd<float, 16> v180_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v180_data + (v168_data * v178_data));
              float v183_data = s0[17];
              tensorforge::intel_esimd::simd<float, 16> v185_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v185_data + (v168_data * v183_data));
              tensorforge::intel_esimd::simd<float, 16> v196_data(r0.template select<16, 1>(96));
              float v202_data = s0[16];
              tensorforge::intel_esimd::simd<float, 16> v204_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v204_data + (v196_data * v202_data));
              float v207_data = s0[18];
              tensorforge::intel_esimd::simd<float, 16> v209_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v209_data + (v196_data * v207_data));
              float v212_data = s0[20];
              tensorforge::intel_esimd::simd<float, 16> v214_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v214_data + (v196_data * v212_data));
              tensorforge::intel_esimd::simd<float, 16> v224_data(r0.template select<16, 1>(112));
              float v231_data = s0[19];
              tensorforge::intel_esimd::simd<float, 16> v233_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v233_data + (v224_data * v231_data));
              float v236_data = s0[21];
              tensorforge::intel_esimd::simd<float, 16> v238_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v238_data + (v224_data * v236_data));
              float v241_data = s0[23];
              tensorforge::intel_esimd::simd<float, 16> v243_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v243_data + (v224_data * v241_data));
              tensorforge::intel_esimd::simd<float, 16> v252_data(r0.template select<16, 1>(128));
              float v260_data = s0[22];
              tensorforge::intel_esimd::simd<float, 16> v262_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v262_data + (v252_data * v260_data));
              float v265_data = s0[24];
              tensorforge::intel_esimd::simd<float, 16> v267_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v267_data + (v252_data * v265_data));
              float v270_data = s0[26];
              tensorforge::intel_esimd::simd<float, 16> v272_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v272_data + (v252_data * v270_data));
              tensorforge::intel_esimd::simd<float, 16> v280_data(r0.template select<16, 1>(144));
              float v289_data = s0[25];
              tensorforge::intel_esimd::simd<float, 16> v291_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v291_data + (v280_data * v289_data));
              float v294_data = s0[27];
              tensorforge::intel_esimd::simd<float, 16> v296_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v296_data + (v280_data * v294_data));
              float v299_data = s0[29];
              tensorforge::intel_esimd::simd<float, 16> v301_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v301_data + (v280_data * v299_data));
              tensorforge::intel_esimd::simd<float, 16> v308_data(r0.template select<16, 1>(160));
              float v318_data = s0[28];
              tensorforge::intel_esimd::simd<float, 16> v320_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v320_data + (v308_data * v318_data));
              float v323_data = s0[30];
              tensorforge::intel_esimd::simd<float, 16> v325_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v325_data + (v308_data * v323_data));
              float v328_data = s0[32];
              tensorforge::intel_esimd::simd<float, 16> v330_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v330_data + (v308_data * v328_data));
              tensorforge::intel_esimd::simd<float, 16> v336_data(r0.template select<16, 1>(176));
              float v347_data = s0[31];
              tensorforge::intel_esimd::simd<float, 16> v349_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v349_data + (v336_data * v347_data));
              float v352_data = s0[33];
              tensorforge::intel_esimd::simd<float, 16> v354_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v354_data + (v336_data * v352_data));
              float v357_data = s0[35];
              tensorforge::intel_esimd::simd<float, 16> v359_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v359_data + (v336_data * v357_data));
              tensorforge::intel_esimd::simd<float, 16> v364_data(r0.template select<16, 1>(192));
              float v376_data = s0[34];
              tensorforge::intel_esimd::simd<float, 16> v378_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v378_data + (v364_data * v376_data));
              float v381_data = s0[36];
              tensorforge::intel_esimd::simd<float, 16> v383_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v383_data + (v364_data * v381_data));
              float v386_data = s0[38];
              tensorforge::intel_esimd::simd<float, 16> v388_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v388_data + (v364_data * v386_data));
              tensorforge::intel_esimd::simd<float, 16> v392_data(r0.template select<16, 1>(208));
              float v405_data = s0[37];
              tensorforge::intel_esimd::simd<float, 16> v407_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v407_data + (v392_data * v405_data));
              float v410_data = s0[39];
              tensorforge::intel_esimd::simd<float, 16> v412_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v412_data + (v392_data * v410_data));
              float v415_data = s0[41];
              tensorforge::intel_esimd::simd<float, 16> v417_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v417_data + (v392_data * v415_data));
              tensorforge::intel_esimd::simd<float, 16> v420_data(r0.template select<16, 1>(224));
              float v434_data = s0[40];
              tensorforge::intel_esimd::simd<float, 16> v436_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v436_data + (v420_data * v434_data));
              float v439_data = s0[42];
              tensorforge::intel_esimd::simd<float, 16> v441_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v441_data + (v420_data * v439_data));
              float v444_data = s0[44];
              tensorforge::intel_esimd::simd<float, 16> v446_data(ir1.template select<16, 1>(240));
              ir1.template select<16, 1>(240) = (v446_data + (v420_data * v444_data));
              tensorforge::intel_esimd::simd<float, 16> v448_data(r0.template select<16, 1>(240));
              float v463_data = s0[43];
              tensorforge::intel_esimd::simd<float, 16> v465_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v465_data + (v448_data * v463_data));
              float v468_data = s0[45];
              tensorforge::intel_esimd::simd<float, 16> v470_data(ir1.template select<16, 1>(240));
              ir1.template select<16, 1>(240) = (v470_data + (v448_data * v468_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v472_n0 = 0; v472_n0 < 1; ++v472_n0) {
                int32_t v474_a = v472_n0 * 16;
                #pragma unroll
                for (int32_t v473_n1 = 0; v473_n1 < 16; ++v473_n1) {
                  int32_t v476_a = v474_a + (v473_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v477_data(ir1.template select<16, 1>(v476_a));
                  r1.template select<16, 1>(v476_a) = v477_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v478_i0 = 0; v478_i0 < 1; ++v478_i0) {
                int32_t v480_a = v478_i0 * 16;
                #pragma unroll
                for (int32_t v479_i1 = 0; v479_i1 < 16; ++v479_i1) {
                  int32_t v482_a = v480_a + (v479_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v483_data(r1.template select<16, 1>(v482_a));
                  v483_data.copy_to(glb_m0 + (v482_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


// === base name ===
kernel_46d569f5d263137f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_46d569f5d263137f = {{1, 16, 1}, 16, 16, 1, 16, 8192, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_46d569f5d263137f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_46d569f5d263137f(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_46d569f5d263137f(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1024 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_46d569f5d263137f(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_46d569f5d263137f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_46d569f5d263137f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_46d569f5d263137f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1024 * sizeof(double)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 8192 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1024}],"shared_bytes":8192,"shared_elements":1024,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (64 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<double> tempShrMem = localShrMem0 + (48);
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v5_batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v5_batchId0 * 256 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v5_batchId0 * 46 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v19_lead = v17_i0 * 16;
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
                  int32_t v22_a = v19_lead + (v18_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v23_data;
                  v23_data.copy_from(glb_m1 + (v22_a));
                  r0.template select<16, 1>(v22_a) = v23_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<double, 32> v25_ld;
              v25_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v25_ld);
              tensorforge::intel_esimd::simd<double, 14> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<double, 14>(s0 + (0 + 0 + 1 * 0 + 32), v26_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<double, 256> r1(0.0);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 256> ir1(0.0);
              tensorforge::intel_esimd::simd<double, 16> v29_data(r0.template select<16, 1>(0));
              double v30_data = s0[0];
              tensorforge::intel_esimd::simd<double, 16> v32_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v32_data + (v29_data * v30_data));
              double v35_data = s0[2];
              tensorforge::intel_esimd::simd<double, 16> v37_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v37_data + (v29_data * v35_data));
              tensorforge::intel_esimd::simd<double, 16> v53_data(r0.template select<16, 1>(16));
              double v54_data = s0[1];
              tensorforge::intel_esimd::simd<double, 16> v56_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v56_data + (v53_data * v54_data));
              double v59_data = s0[3];
              tensorforge::intel_esimd::simd<double, 16> v61_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v61_data + (v53_data * v59_data));
              double v64_data = s0[5];
              tensorforge::intel_esimd::simd<double, 16> v66_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v66_data + (v53_data * v64_data));
              tensorforge::intel_esimd::simd<double, 16> v81_data(r0.template select<16, 1>(32));
              double v83_data = s0[4];
              tensorforge::intel_esimd::simd<double, 16> v85_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v85_data + (v81_data * v83_data));
              double v88_data = s0[6];
              tensorforge::intel_esimd::simd<double, 16> v90_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v90_data + (v81_data * v88_data));
              double v93_data = s0[8];
              tensorforge::intel_esimd::simd<double, 16> v95_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v95_data + (v81_data * v93_data));
              tensorforge::intel_esimd::simd<double, 16> v109_data(r0.template select<16, 1>(48));
              double v112_data = s0[7];
              tensorforge::intel_esimd::simd<double, 16> v114_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v114_data + (v109_data * v112_data));
              double v117_data = s0[9];
              tensorforge::intel_esimd::simd<double, 16> v119_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v119_data + (v109_data * v117_data));
              double v122_data = s0[11];
              tensorforge::intel_esimd::simd<double, 16> v124_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v124_data + (v109_data * v122_data));
              tensorforge::intel_esimd::simd<double, 16> v137_data(r0.template select<16, 1>(64));
              double v141_data = s0[10];
              tensorforge::intel_esimd::simd<double, 16> v143_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v143_data + (v137_data * v141_data));
              double v146_data = s0[12];
              tensorforge::intel_esimd::simd<double, 16> v148_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v148_data + (v137_data * v146_data));
              double v151_data = s0[14];
              tensorforge::intel_esimd::simd<double, 16> v153_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v153_data + (v137_data * v151_data));
              tensorforge::intel_esimd::simd<double, 16> v165_data(r0.template select<16, 1>(80));
              double v170_data = s0[13];
              tensorforge::intel_esimd::simd<double, 16> v172_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v172_data + (v165_data * v170_data));
              double v175_data = s0[15];
              tensorforge::intel_esimd::simd<double, 16> v177_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v177_data + (v165_data * v175_data));
              double v180_data = s0[17];
              tensorforge::intel_esimd::simd<double, 16> v182_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v182_data + (v165_data * v180_data));
              tensorforge::intel_esimd::simd<double, 16> v193_data(r0.template select<16, 1>(96));
              double v199_data = s0[16];
              tensorforge::intel_esimd::simd<double, 16> v201_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v201_data + (v193_data * v199_data));
              double v204_data = s0[18];
              tensorforge::intel_esimd::simd<double, 16> v206_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v206_data + (v193_data * v204_data));
              double v209_data = s0[20];
              tensorforge::intel_esimd::simd<double, 16> v211_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v211_data + (v193_data * v209_data));
              tensorforge::intel_esimd::simd<double, 16> v221_data(r0.template select<16, 1>(112));
              double v228_data = s0[19];
              tensorforge::intel_esimd::simd<double, 16> v230_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v230_data + (v221_data * v228_data));
              double v233_data = s0[21];
              tensorforge::intel_esimd::simd<double, 16> v235_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v235_data + (v221_data * v233_data));
              double v238_data = s0[23];
              tensorforge::intel_esimd::simd<double, 16> v240_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v240_data + (v221_data * v238_data));
              tensorforge::intel_esimd::simd<double, 16> v249_data(r0.template select<16, 1>(128));
              double v257_data = s0[22];
              tensorforge::intel_esimd::simd<double, 16> v259_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v259_data + (v249_data * v257_data));
              double v262_data = s0[24];
              tensorforge::intel_esimd::simd<double, 16> v264_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v264_data + (v249_data * v262_data));
              double v267_data = s0[26];
              tensorforge::intel_esimd::simd<double, 16> v269_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v269_data + (v249_data * v267_data));
              tensorforge::intel_esimd::simd<double, 16> v277_data(r0.template select<16, 1>(144));
              double v286_data = s0[25];
              tensorforge::intel_esimd::simd<double, 16> v288_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v288_data + (v277_data * v286_data));
              double v291_data = s0[27];
              tensorforge::intel_esimd::simd<double, 16> v293_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v293_data + (v277_data * v291_data));
              double v296_data = s0[29];
              tensorforge::intel_esimd::simd<double, 16> v298_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v298_data + (v277_data * v296_data));
              tensorforge::intel_esimd::simd<double, 16> v305_data(r0.template select<16, 1>(160));
              double v315_data = s0[28];
              tensorforge::intel_esimd::simd<double, 16> v317_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v317_data + (v305_data * v315_data));
              double v320_data = s0[30];
              tensorforge::intel_esimd::simd<double, 16> v322_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v322_data + (v305_data * v320_data));
              double v325_data = s0[32];
              tensorforge::intel_esimd::simd<double, 16> v327_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v327_data + (v305_data * v325_data));
              tensorforge::intel_esimd::simd<double, 16> v333_data(r0.template select<16, 1>(176));
              double v344_data = s0[31];
              tensorforge::intel_esimd::simd<double, 16> v346_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v346_data + (v333_data * v344_data));
              double v349_data = s0[33];
              tensorforge::intel_esimd::simd<double, 16> v351_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v351_data + (v333_data * v349_data));
              double v354_data = s0[35];
              tensorforge::intel_esimd::simd<double, 16> v356_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v356_data + (v333_data * v354_data));
              tensorforge::intel_esimd::simd<double, 16> v361_data(r0.template select<16, 1>(192));
              double v373_data = s0[34];
              tensorforge::intel_esimd::simd<double, 16> v375_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v375_data + (v361_data * v373_data));
              double v378_data = s0[36];
              tensorforge::intel_esimd::simd<double, 16> v380_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v380_data + (v361_data * v378_data));
              double v383_data = s0[38];
              tensorforge::intel_esimd::simd<double, 16> v385_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v385_data + (v361_data * v383_data));
              tensorforge::intel_esimd::simd<double, 16> v389_data(r0.template select<16, 1>(208));
              double v402_data = s0[37];
              tensorforge::intel_esimd::simd<double, 16> v404_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v404_data + (v389_data * v402_data));
              double v407_data = s0[39];
              tensorforge::intel_esimd::simd<double, 16> v409_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v409_data + (v389_data * v407_data));
              double v412_data = s0[41];
              tensorforge::intel_esimd::simd<double, 16> v414_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v414_data + (v389_data * v412_data));
              tensorforge::intel_esimd::simd<double, 16> v417_data(r0.template select<16, 1>(224));
              double v431_data = s0[40];
              tensorforge::intel_esimd::simd<double, 16> v433_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v433_data + (v417_data * v431_data));
              double v436_data = s0[42];
              tensorforge::intel_esimd::simd<double, 16> v438_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v438_data + (v417_data * v436_data));
              double v441_data = s0[44];
              tensorforge::intel_esimd::simd<double, 16> v443_data(ir1.template select<16, 1>(240));
              ir1.template select<16, 1>(240) = (v443_data + (v417_data * v441_data));
              tensorforge::intel_esimd::simd<double, 16> v445_data(r0.template select<16, 1>(240));
              double v460_data = s0[43];
              tensorforge::intel_esimd::simd<double, 16> v462_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v462_data + (v445_data * v460_data));
              double v465_data = s0[45];
              tensorforge::intel_esimd::simd<double, 16> v467_data(ir1.template select<16, 1>(240));
              ir1.template select<16, 1>(240) = (v467_data + (v445_data * v465_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v469_n0 = 0; v469_n0 < 1; ++v469_n0) {
                int32_t v471_a = v469_n0 * 16;
                #pragma unroll
                for (int32_t v470_n1 = 0; v470_n1 < 16; ++v470_n1) {
                  int32_t v473_a = v471_a + (v470_n1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v474_data(ir1.template select<16, 1>(v473_a));
                  r1.template select<16, 1>(v473_a) = v474_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v475_i0 = 0; v475_i0 < 1; ++v475_i0) {
                int32_t v477_a = v475_i0 * 16;
                #pragma unroll
                for (int32_t v476_i1 = 0; v476_i1 < 16; ++v476_i1) {
                  int32_t v479_a = v477_a + (v476_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v480_data(r1.template select<16, 1>(v479_a));
                  v480_data.copy_to(glb_m0 + (v479_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


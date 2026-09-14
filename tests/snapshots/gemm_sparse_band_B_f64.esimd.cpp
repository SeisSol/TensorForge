// === base name ===
kernel_51c5ef22ffd74861

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_51c5ef22ffd74861 = {{1, 16, 1}, 16, 16, 1, 16, 8192, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_51c5ef22ffd74861(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_51c5ef22ffd74861(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_51c5ef22ffd74861(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_51c5ef22ffd74861(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_51c5ef22ffd74861(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_51c5ef22ffd74861(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_51c5ef22ffd74861(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const double *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 256 + 0 + m1_extraOffset];
            const double *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 46 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v5_batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v5_batchId0 * 256 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v5_batchId0 * 46 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                int32_t v21_lead = v19_i0 * 16;
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
                  int32_t v24_a = v21_lead + (v20_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v25_data;
                  v25_data.copy_from(glb_m1 + (v24_a));
                  r0.template select<16, 1>(v24_a) = v25_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<double, 32> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v27_ld);
              tensorforge::intel_esimd::simd<double, 14> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<double, 14>(s0 + (0 + 0 + 1 * 0 + 32), v28_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<double, 256> r1(0.0);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 256> ir1(0.0);
              tensorforge::intel_esimd::simd<double, 16> v31_data(r0.template select<16, 1>(0));
              double v32_data = s0[0];
              tensorforge::intel_esimd::simd<double, 16> v34_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v34_data + (v31_data * v32_data));
              double v37_data = s0[2];
              tensorforge::intel_esimd::simd<double, 16> v39_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v39_data + (v31_data * v37_data));
              tensorforge::intel_esimd::simd<double, 16> v55_data(r0.template select<16, 1>(16));
              double v56_data = s0[1];
              tensorforge::intel_esimd::simd<double, 16> v58_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v58_data + (v55_data * v56_data));
              double v61_data = s0[3];
              tensorforge::intel_esimd::simd<double, 16> v63_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v63_data + (v55_data * v61_data));
              double v66_data = s0[5];
              tensorforge::intel_esimd::simd<double, 16> v68_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v68_data + (v55_data * v66_data));
              tensorforge::intel_esimd::simd<double, 16> v83_data(r0.template select<16, 1>(32));
              double v85_data = s0[4];
              tensorforge::intel_esimd::simd<double, 16> v87_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v87_data + (v83_data * v85_data));
              double v90_data = s0[6];
              tensorforge::intel_esimd::simd<double, 16> v92_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v92_data + (v83_data * v90_data));
              double v95_data = s0[8];
              tensorforge::intel_esimd::simd<double, 16> v97_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v97_data + (v83_data * v95_data));
              tensorforge::intel_esimd::simd<double, 16> v111_data(r0.template select<16, 1>(48));
              double v114_data = s0[7];
              tensorforge::intel_esimd::simd<double, 16> v116_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v116_data + (v111_data * v114_data));
              double v119_data = s0[9];
              tensorforge::intel_esimd::simd<double, 16> v121_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v121_data + (v111_data * v119_data));
              double v124_data = s0[11];
              tensorforge::intel_esimd::simd<double, 16> v126_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v126_data + (v111_data * v124_data));
              tensorforge::intel_esimd::simd<double, 16> v139_data(r0.template select<16, 1>(64));
              double v143_data = s0[10];
              tensorforge::intel_esimd::simd<double, 16> v145_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v145_data + (v139_data * v143_data));
              double v148_data = s0[12];
              tensorforge::intel_esimd::simd<double, 16> v150_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v150_data + (v139_data * v148_data));
              double v153_data = s0[14];
              tensorforge::intel_esimd::simd<double, 16> v155_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v155_data + (v139_data * v153_data));
              tensorforge::intel_esimd::simd<double, 16> v167_data(r0.template select<16, 1>(80));
              double v172_data = s0[13];
              tensorforge::intel_esimd::simd<double, 16> v174_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v174_data + (v167_data * v172_data));
              double v177_data = s0[15];
              tensorforge::intel_esimd::simd<double, 16> v179_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v179_data + (v167_data * v177_data));
              double v182_data = s0[17];
              tensorforge::intel_esimd::simd<double, 16> v184_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v184_data + (v167_data * v182_data));
              tensorforge::intel_esimd::simd<double, 16> v195_data(r0.template select<16, 1>(96));
              double v201_data = s0[16];
              tensorforge::intel_esimd::simd<double, 16> v203_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v203_data + (v195_data * v201_data));
              double v206_data = s0[18];
              tensorforge::intel_esimd::simd<double, 16> v208_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v208_data + (v195_data * v206_data));
              double v211_data = s0[20];
              tensorforge::intel_esimd::simd<double, 16> v213_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v213_data + (v195_data * v211_data));
              tensorforge::intel_esimd::simd<double, 16> v223_data(r0.template select<16, 1>(112));
              double v230_data = s0[19];
              tensorforge::intel_esimd::simd<double, 16> v232_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v232_data + (v223_data * v230_data));
              double v235_data = s0[21];
              tensorforge::intel_esimd::simd<double, 16> v237_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v237_data + (v223_data * v235_data));
              double v240_data = s0[23];
              tensorforge::intel_esimd::simd<double, 16> v242_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v242_data + (v223_data * v240_data));
              tensorforge::intel_esimd::simd<double, 16> v251_data(r0.template select<16, 1>(128));
              double v259_data = s0[22];
              tensorforge::intel_esimd::simd<double, 16> v261_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v261_data + (v251_data * v259_data));
              double v264_data = s0[24];
              tensorforge::intel_esimd::simd<double, 16> v266_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v266_data + (v251_data * v264_data));
              double v269_data = s0[26];
              tensorforge::intel_esimd::simd<double, 16> v271_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v271_data + (v251_data * v269_data));
              tensorforge::intel_esimd::simd<double, 16> v279_data(r0.template select<16, 1>(144));
              double v288_data = s0[25];
              tensorforge::intel_esimd::simd<double, 16> v290_data(ir1.template select<16, 1>(128));
              ir1.template select<16, 1>(128) = (v290_data + (v279_data * v288_data));
              double v293_data = s0[27];
              tensorforge::intel_esimd::simd<double, 16> v295_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v295_data + (v279_data * v293_data));
              double v298_data = s0[29];
              tensorforge::intel_esimd::simd<double, 16> v300_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v300_data + (v279_data * v298_data));
              tensorforge::intel_esimd::simd<double, 16> v307_data(r0.template select<16, 1>(160));
              double v317_data = s0[28];
              tensorforge::intel_esimd::simd<double, 16> v319_data(ir1.template select<16, 1>(144));
              ir1.template select<16, 1>(144) = (v319_data + (v307_data * v317_data));
              double v322_data = s0[30];
              tensorforge::intel_esimd::simd<double, 16> v324_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v324_data + (v307_data * v322_data));
              double v327_data = s0[32];
              tensorforge::intel_esimd::simd<double, 16> v329_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v329_data + (v307_data * v327_data));
              tensorforge::intel_esimd::simd<double, 16> v335_data(r0.template select<16, 1>(176));
              double v346_data = s0[31];
              tensorforge::intel_esimd::simd<double, 16> v348_data(ir1.template select<16, 1>(160));
              ir1.template select<16, 1>(160) = (v348_data + (v335_data * v346_data));
              double v351_data = s0[33];
              tensorforge::intel_esimd::simd<double, 16> v353_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v353_data + (v335_data * v351_data));
              double v356_data = s0[35];
              tensorforge::intel_esimd::simd<double, 16> v358_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v358_data + (v335_data * v356_data));
              tensorforge::intel_esimd::simd<double, 16> v363_data(r0.template select<16, 1>(192));
              double v375_data = s0[34];
              tensorforge::intel_esimd::simd<double, 16> v377_data(ir1.template select<16, 1>(176));
              ir1.template select<16, 1>(176) = (v377_data + (v363_data * v375_data));
              double v380_data = s0[36];
              tensorforge::intel_esimd::simd<double, 16> v382_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v382_data + (v363_data * v380_data));
              double v385_data = s0[38];
              tensorforge::intel_esimd::simd<double, 16> v387_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v387_data + (v363_data * v385_data));
              tensorforge::intel_esimd::simd<double, 16> v391_data(r0.template select<16, 1>(208));
              double v404_data = s0[37];
              tensorforge::intel_esimd::simd<double, 16> v406_data(ir1.template select<16, 1>(192));
              ir1.template select<16, 1>(192) = (v406_data + (v391_data * v404_data));
              double v409_data = s0[39];
              tensorforge::intel_esimd::simd<double, 16> v411_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v411_data + (v391_data * v409_data));
              double v414_data = s0[41];
              tensorforge::intel_esimd::simd<double, 16> v416_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v416_data + (v391_data * v414_data));
              tensorforge::intel_esimd::simd<double, 16> v419_data(r0.template select<16, 1>(224));
              double v433_data = s0[40];
              tensorforge::intel_esimd::simd<double, 16> v435_data(ir1.template select<16, 1>(208));
              ir1.template select<16, 1>(208) = (v435_data + (v419_data * v433_data));
              double v438_data = s0[42];
              tensorforge::intel_esimd::simd<double, 16> v440_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v440_data + (v419_data * v438_data));
              double v443_data = s0[44];
              tensorforge::intel_esimd::simd<double, 16> v445_data(ir1.template select<16, 1>(240));
              ir1.template select<16, 1>(240) = (v445_data + (v419_data * v443_data));
              tensorforge::intel_esimd::simd<double, 16> v447_data(r0.template select<16, 1>(240));
              double v462_data = s0[43];
              tensorforge::intel_esimd::simd<double, 16> v464_data(ir1.template select<16, 1>(224));
              ir1.template select<16, 1>(224) = (v464_data + (v447_data * v462_data));
              double v467_data = s0[45];
              tensorforge::intel_esimd::simd<double, 16> v469_data(ir1.template select<16, 1>(240));
              ir1.template select<16, 1>(240) = (v469_data + (v447_data * v467_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v471_n0 = 0; v471_n0 < 1; ++v471_n0) {
                int32_t v473_a = v471_n0 * 16;
                #pragma unroll
                for (int32_t v472_n1 = 0; v472_n1 < 16; ++v472_n1) {
                  int32_t v475_a = v473_a + (v472_n1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v476_data(ir1.template select<16, 1>(v475_a));
                  r1.template select<16, 1>(v475_a) = v476_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v477_i0 = 0; v477_i0 < 1; ++v477_i0) {
                int32_t v479_a = v477_i0 * 16;
                #pragma unroll
                for (int32_t v478_i1 = 0; v478_i1 < 16; ++v478_i1) {
                  int32_t v481_a = v479_a + (v478_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v482_data(r1.template select<16, 1>(v481_a));
                  v482_data.copy_to(glb_m0 + (v481_a));
                }
              }
            }
            tensorforge::prefetchRunsL2<992, 32, 184>(&pf_glb_m1[0], &pf_glb_m1[248], &pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


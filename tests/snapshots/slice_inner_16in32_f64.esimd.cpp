// === base name ===
kernel_8938966fad8ac17a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8938966fad8ac17a = {{1, 16, 1}, 16, 16, 1, 16, 18432, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8938966fad8ac17a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8938966fad8ac17a(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8938966fad8ac17a(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2304 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_8938966fad8ac17a(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8938966fad8ac17a(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8938966fad8ac17a(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8938966fad8ac17a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(double)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 18432 B shared, occupancy grid
        // operands:
        //   m0 16×8(16×8) {0..16}×{0..8} strided
        //   m1 32×32(32×32) {0..32}×{0..32} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{8..24}×{8..24} × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":18432,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[8,8],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (144 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v8_batchId0 * 128 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v8_batchId0 * 1024 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v8_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 16;
                int32_t v24_off = v22_lead + 8;
                #pragma unroll
                for (int32_t v21_i1 = 8; v21_i1 < 24; ++v21_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v27_data;
                  v27_data.copy_from(glb_m1 + ((v24_off + (v21_i1 * 32))));
                  r0.template select<16, 1>((v22_lead + ((v21_i1 - 8) * 16))) = v27_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<double, 32> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<double, 32> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 32));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v32_ld);
              tensorforge::intel_esimd::simd<double, 32> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 64), v33_ld);
              tensorforge::intel_esimd::simd<double, 32> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 96));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 96), v34_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<double, 128> r1(0.0);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 128> ir1(0.0);
              tensorforge::intel_esimd::simd<double, 16> v37_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 128> s0_w0 = tensorforge::slmLoad<double, 128>(s0 + 0);
              double v38_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 16> v40_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v40_data + (v37_data * v38_data));
              double v43_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v45_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v45_data + (v37_data * v43_data));
              double v48_data = s0_w0[32];
              tensorforge::intel_esimd::simd<double, 16> v50_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v50_data + (v37_data * v48_data));
              double v53_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 16> v55_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v55_data + (v37_data * v53_data));
              double v58_data = s0_w0[64];
              tensorforge::intel_esimd::simd<double, 16> v60_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v60_data + (v37_data * v58_data));
              double v63_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 16> v65_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v65_data + (v37_data * v63_data));
              double v68_data = s0_w0[96];
              tensorforge::intel_esimd::simd<double, 16> v70_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v70_data + (v37_data * v68_data));
              double v73_data = s0_w0[112];
              tensorforge::intel_esimd::simd<double, 16> v75_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v75_data + (v37_data * v73_data));
              tensorforge::intel_esimd::simd<double, 16> v77_data(r0.template select<16, 1>(16));
              double v78_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 16> v80_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v80_data + (v77_data * v78_data));
              double v83_data = s0_w0[17];
              tensorforge::intel_esimd::simd<double, 16> v85_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v85_data + (v77_data * v83_data));
              double v88_data = s0_w0[33];
              tensorforge::intel_esimd::simd<double, 16> v90_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v90_data + (v77_data * v88_data));
              double v93_data = s0_w0[49];
              tensorforge::intel_esimd::simd<double, 16> v95_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v95_data + (v77_data * v93_data));
              double v98_data = s0_w0[65];
              tensorforge::intel_esimd::simd<double, 16> v100_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v100_data + (v77_data * v98_data));
              double v103_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 16> v105_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v105_data + (v77_data * v103_data));
              double v108_data = s0_w0[97];
              tensorforge::intel_esimd::simd<double, 16> v110_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v110_data + (v77_data * v108_data));
              double v113_data = s0_w0[113];
              tensorforge::intel_esimd::simd<double, 16> v115_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v115_data + (v77_data * v113_data));
              tensorforge::intel_esimd::simd<double, 16> v117_data(r0.template select<16, 1>(32));
              double v118_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 16> v120_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v120_data + (v117_data * v118_data));
              double v123_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 16> v125_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v125_data + (v117_data * v123_data));
              double v128_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v130_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v130_data + (v117_data * v128_data));
              double v133_data = s0_w0[50];
              tensorforge::intel_esimd::simd<double, 16> v135_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v135_data + (v117_data * v133_data));
              double v138_data = s0_w0[66];
              tensorforge::intel_esimd::simd<double, 16> v140_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v140_data + (v117_data * v138_data));
              double v143_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 16> v145_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v145_data + (v117_data * v143_data));
              double v148_data = s0_w0[98];
              tensorforge::intel_esimd::simd<double, 16> v150_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v150_data + (v117_data * v148_data));
              double v153_data = s0_w0[114];
              tensorforge::intel_esimd::simd<double, 16> v155_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v155_data + (v117_data * v153_data));
              tensorforge::intel_esimd::simd<double, 16> v157_data(r0.template select<16, 1>(48));
              double v158_data = s0_w0[3];
              tensorforge::intel_esimd::simd<double, 16> v160_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v160_data + (v157_data * v158_data));
              double v163_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 16> v165_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v165_data + (v157_data * v163_data));
              double v168_data = s0_w0[35];
              tensorforge::intel_esimd::simd<double, 16> v170_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v170_data + (v157_data * v168_data));
              double v173_data = s0_w0[51];
              tensorforge::intel_esimd::simd<double, 16> v175_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v175_data + (v157_data * v173_data));
              double v178_data = s0_w0[67];
              tensorforge::intel_esimd::simd<double, 16> v180_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v180_data + (v157_data * v178_data));
              double v183_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 16> v185_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v185_data + (v157_data * v183_data));
              double v188_data = s0_w0[99];
              tensorforge::intel_esimd::simd<double, 16> v190_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v190_data + (v157_data * v188_data));
              double v193_data = s0_w0[115];
              tensorforge::intel_esimd::simd<double, 16> v195_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v195_data + (v157_data * v193_data));
              tensorforge::intel_esimd::simd<double, 16> v197_data(r0.template select<16, 1>(64));
              double v198_data = s0_w0[4];
              tensorforge::intel_esimd::simd<double, 16> v200_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v200_data + (v197_data * v198_data));
              double v203_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 16> v205_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v205_data + (v197_data * v203_data));
              double v208_data = s0_w0[36];
              tensorforge::intel_esimd::simd<double, 16> v210_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v210_data + (v197_data * v208_data));
              double v213_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v215_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v215_data + (v197_data * v213_data));
              double v218_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 16> v220_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v220_data + (v197_data * v218_data));
              double v223_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 16> v225_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v225_data + (v197_data * v223_data));
              double v228_data = s0_w0[100];
              tensorforge::intel_esimd::simd<double, 16> v230_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v230_data + (v197_data * v228_data));
              double v233_data = s0_w0[116];
              tensorforge::intel_esimd::simd<double, 16> v235_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v235_data + (v197_data * v233_data));
              tensorforge::intel_esimd::simd<double, 16> v237_data(r0.template select<16, 1>(80));
              double v238_data = s0_w0[5];
              tensorforge::intel_esimd::simd<double, 16> v240_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v240_data + (v237_data * v238_data));
              double v243_data = s0_w0[21];
              tensorforge::intel_esimd::simd<double, 16> v245_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v245_data + (v237_data * v243_data));
              double v248_data = s0_w0[37];
              tensorforge::intel_esimd::simd<double, 16> v250_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v250_data + (v237_data * v248_data));
              double v253_data = s0_w0[53];
              tensorforge::intel_esimd::simd<double, 16> v255_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v255_data + (v237_data * v253_data));
              double v258_data = s0_w0[69];
              tensorforge::intel_esimd::simd<double, 16> v260_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v260_data + (v237_data * v258_data));
              double v263_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 16> v265_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v265_data + (v237_data * v263_data));
              double v268_data = s0_w0[101];
              tensorforge::intel_esimd::simd<double, 16> v270_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v270_data + (v237_data * v268_data));
              double v273_data = s0_w0[117];
              tensorforge::intel_esimd::simd<double, 16> v275_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v275_data + (v237_data * v273_data));
              tensorforge::intel_esimd::simd<double, 16> v277_data(r0.template select<16, 1>(96));
              double v278_data = s0_w0[6];
              tensorforge::intel_esimd::simd<double, 16> v280_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v280_data + (v277_data * v278_data));
              double v283_data = s0_w0[22];
              tensorforge::intel_esimd::simd<double, 16> v285_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v285_data + (v277_data * v283_data));
              double v288_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 16> v290_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v290_data + (v277_data * v288_data));
              double v293_data = s0_w0[54];
              tensorforge::intel_esimd::simd<double, 16> v295_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v295_data + (v277_data * v293_data));
              double v298_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v300_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v300_data + (v277_data * v298_data));
              double v303_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 16> v305_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v305_data + (v277_data * v303_data));
              double v308_data = s0_w0[102];
              tensorforge::intel_esimd::simd<double, 16> v310_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v310_data + (v277_data * v308_data));
              double v313_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v315_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v315_data + (v277_data * v313_data));
              tensorforge::intel_esimd::simd<double, 16> v317_data(r0.template select<16, 1>(112));
              double v318_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v320_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v320_data + (v317_data * v318_data));
              double v323_data = s0_w0[23];
              tensorforge::intel_esimd::simd<double, 16> v325_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v325_data + (v317_data * v323_data));
              double v328_data = s0_w0[39];
              tensorforge::intel_esimd::simd<double, 16> v330_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v330_data + (v317_data * v328_data));
              double v333_data = s0_w0[55];
              tensorforge::intel_esimd::simd<double, 16> v335_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v335_data + (v317_data * v333_data));
              double v338_data = s0_w0[71];
              tensorforge::intel_esimd::simd<double, 16> v340_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v340_data + (v317_data * v338_data));
              double v343_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 16> v345_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v345_data + (v317_data * v343_data));
              double v348_data = s0_w0[103];
              tensorforge::intel_esimd::simd<double, 16> v350_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v350_data + (v317_data * v348_data));
              double v353_data = s0_w0[119];
              tensorforge::intel_esimd::simd<double, 16> v355_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v355_data + (v317_data * v353_data));
              tensorforge::intel_esimd::simd<double, 16> v357_data(r0.template select<16, 1>(128));
              double v358_data = s0_w0[8];
              tensorforge::intel_esimd::simd<double, 16> v360_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v360_data + (v357_data * v358_data));
              double v363_data = s0_w0[24];
              tensorforge::intel_esimd::simd<double, 16> v365_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v365_data + (v357_data * v363_data));
              double v368_data = s0_w0[40];
              tensorforge::intel_esimd::simd<double, 16> v370_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v370_data + (v357_data * v368_data));
              double v373_data = s0_w0[56];
              tensorforge::intel_esimd::simd<double, 16> v375_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v375_data + (v357_data * v373_data));
              double v378_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v380_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v380_data + (v357_data * v378_data));
              double v383_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 16> v385_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v385_data + (v357_data * v383_data));
              double v388_data = s0_w0[104];
              tensorforge::intel_esimd::simd<double, 16> v390_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v390_data + (v357_data * v388_data));
              double v393_data = s0_w0[120];
              tensorforge::intel_esimd::simd<double, 16> v395_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v395_data + (v357_data * v393_data));
              tensorforge::intel_esimd::simd<double, 16> v397_data(r0.template select<16, 1>(144));
              double v398_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 16> v400_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v400_data + (v397_data * v398_data));
              double v403_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v405_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v405_data + (v397_data * v403_data));
              double v408_data = s0_w0[41];
              tensorforge::intel_esimd::simd<double, 16> v410_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v410_data + (v397_data * v408_data));
              double v413_data = s0_w0[57];
              tensorforge::intel_esimd::simd<double, 16> v415_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v415_data + (v397_data * v413_data));
              double v418_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v420_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v420_data + (v397_data * v418_data));
              double v423_data = s0_w0[89];
              tensorforge::intel_esimd::simd<double, 16> v425_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v425_data + (v397_data * v423_data));
              double v428_data = s0_w0[105];
              tensorforge::intel_esimd::simd<double, 16> v430_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v430_data + (v397_data * v428_data));
              double v433_data = s0_w0[121];
              tensorforge::intel_esimd::simd<double, 16> v435_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v435_data + (v397_data * v433_data));
              tensorforge::intel_esimd::simd<double, 16> v437_data(r0.template select<16, 1>(160));
              double v438_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 16> v440_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v440_data + (v437_data * v438_data));
              double v443_data = s0_w0[26];
              tensorforge::intel_esimd::simd<double, 16> v445_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v445_data + (v437_data * v443_data));
              double v448_data = s0_w0[42];
              tensorforge::intel_esimd::simd<double, 16> v450_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v450_data + (v437_data * v448_data));
              double v453_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 16> v455_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v455_data + (v437_data * v453_data));
              double v458_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v460_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v460_data + (v437_data * v458_data));
              double v463_data = s0_w0[90];
              tensorforge::intel_esimd::simd<double, 16> v465_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v465_data + (v437_data * v463_data));
              double v468_data = s0_w0[106];
              tensorforge::intel_esimd::simd<double, 16> v470_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v470_data + (v437_data * v468_data));
              double v473_data = s0_w0[122];
              tensorforge::intel_esimd::simd<double, 16> v475_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v475_data + (v437_data * v473_data));
              tensorforge::intel_esimd::simd<double, 16> v477_data(r0.template select<16, 1>(176));
              double v478_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 16> v480_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v480_data + (v477_data * v478_data));
              double v483_data = s0_w0[27];
              tensorforge::intel_esimd::simd<double, 16> v485_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v485_data + (v477_data * v483_data));
              double v488_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v490_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v490_data + (v477_data * v488_data));
              double v493_data = s0_w0[59];
              tensorforge::intel_esimd::simd<double, 16> v495_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v495_data + (v477_data * v493_data));
              double v498_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v500_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v500_data + (v477_data * v498_data));
              double v503_data = s0_w0[91];
              tensorforge::intel_esimd::simd<double, 16> v505_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v505_data + (v477_data * v503_data));
              double v508_data = s0_w0[107];
              tensorforge::intel_esimd::simd<double, 16> v510_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v510_data + (v477_data * v508_data));
              double v513_data = s0_w0[123];
              tensorforge::intel_esimd::simd<double, 16> v515_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v515_data + (v477_data * v513_data));
              tensorforge::intel_esimd::simd<double, 16> v517_data(r0.template select<16, 1>(192));
              double v518_data = s0_w0[12];
              tensorforge::intel_esimd::simd<double, 16> v520_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v520_data + (v517_data * v518_data));
              double v523_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 16> v525_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v525_data + (v517_data * v523_data));
              double v528_data = s0_w0[44];
              tensorforge::intel_esimd::simd<double, 16> v530_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v530_data + (v517_data * v528_data));
              double v533_data = s0_w0[60];
              tensorforge::intel_esimd::simd<double, 16> v535_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v535_data + (v517_data * v533_data));
              double v538_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v540_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v540_data + (v517_data * v538_data));
              double v543_data = s0_w0[92];
              tensorforge::intel_esimd::simd<double, 16> v545_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v545_data + (v517_data * v543_data));
              double v548_data = s0_w0[108];
              tensorforge::intel_esimd::simd<double, 16> v550_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v550_data + (v517_data * v548_data));
              double v553_data = s0_w0[124];
              tensorforge::intel_esimd::simd<double, 16> v555_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v555_data + (v517_data * v553_data));
              tensorforge::intel_esimd::simd<double, 16> v557_data(r0.template select<16, 1>(208));
              double v558_data = s0_w0[13];
              tensorforge::intel_esimd::simd<double, 16> v560_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v560_data + (v557_data * v558_data));
              double v563_data = s0_w0[29];
              tensorforge::intel_esimd::simd<double, 16> v565_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v565_data + (v557_data * v563_data));
              double v568_data = s0_w0[45];
              tensorforge::intel_esimd::simd<double, 16> v570_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v570_data + (v557_data * v568_data));
              double v573_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v575_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v575_data + (v557_data * v573_data));
              double v578_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v580_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v580_data + (v557_data * v578_data));
              double v583_data = s0_w0[93];
              tensorforge::intel_esimd::simd<double, 16> v585_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v585_data + (v557_data * v583_data));
              double v588_data = s0_w0[109];
              tensorforge::intel_esimd::simd<double, 16> v590_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v590_data + (v557_data * v588_data));
              double v593_data = s0_w0[125];
              tensorforge::intel_esimd::simd<double, 16> v595_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v595_data + (v557_data * v593_data));
              tensorforge::intel_esimd::simd<double, 16> v597_data(r0.template select<16, 1>(224));
              double v598_data = s0_w0[14];
              tensorforge::intel_esimd::simd<double, 16> v600_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v600_data + (v597_data * v598_data));
              double v603_data = s0_w0[30];
              tensorforge::intel_esimd::simd<double, 16> v605_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v605_data + (v597_data * v603_data));
              double v608_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v610_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v610_data + (v597_data * v608_data));
              double v613_data = s0_w0[62];
              tensorforge::intel_esimd::simd<double, 16> v615_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v615_data + (v597_data * v613_data));
              double v618_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v620_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v620_data + (v597_data * v618_data));
              double v623_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v625_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v625_data + (v597_data * v623_data));
              double v628_data = s0_w0[110];
              tensorforge::intel_esimd::simd<double, 16> v630_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v630_data + (v597_data * v628_data));
              double v633_data = s0_w0[126];
              tensorforge::intel_esimd::simd<double, 16> v635_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v635_data + (v597_data * v633_data));
              tensorforge::intel_esimd::simd<double, 16> v637_data(r0.template select<16, 1>(240));
              double v638_data = s0_w0[15];
              tensorforge::intel_esimd::simd<double, 16> v640_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v640_data + (v637_data * v638_data));
              double v643_data = s0_w0[31];
              tensorforge::intel_esimd::simd<double, 16> v645_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v645_data + (v637_data * v643_data));
              double v648_data = s0_w0[47];
              tensorforge::intel_esimd::simd<double, 16> v650_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v650_data + (v637_data * v648_data));
              double v653_data = s0_w0[63];
              tensorforge::intel_esimd::simd<double, 16> v655_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v655_data + (v637_data * v653_data));
              double v658_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v660_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v660_data + (v637_data * v658_data));
              double v663_data = s0_w0[95];
              tensorforge::intel_esimd::simd<double, 16> v665_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v665_data + (v637_data * v663_data));
              double v668_data = s0_w0[111];
              tensorforge::intel_esimd::simd<double, 16> v670_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v670_data + (v637_data * v668_data));
              double v673_data = s0_w0[127];
              tensorforge::intel_esimd::simd<double, 16> v675_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v675_data + (v637_data * v673_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v677_n0 = 0; v677_n0 < 1; ++v677_n0) {
                int32_t v679_a = v677_n0 * 16;
                #pragma unroll
                for (int32_t v678_n1 = 0; v678_n1 < 8; ++v678_n1) {
                  int32_t v681_a = v679_a + (v678_n1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v682_data(ir1.template select<16, 1>(v681_a));
                  r1.template select<16, 1>(v681_a) = v682_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v683_i0 = 0; v683_i0 < 1; ++v683_i0) {
                int32_t v685_a = v683_i0 * 16;
                #pragma unroll
                for (int32_t v684_i1 = 0; v684_i1 < 8; ++v684_i1) {
                  int32_t v687_a = v685_a + (v684_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v688_data(r1.template select<16, 1>(v687_a));
                  v688_data.copy_to(glb_m0 + (v687_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


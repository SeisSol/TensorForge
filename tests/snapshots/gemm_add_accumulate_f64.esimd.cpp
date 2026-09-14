// === base name ===
kernel_20ce8073c0dc56d5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_20ce8073c0dc56d5 = {{1, 16, 1}, 16, 12, 1, 16, 18432, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_20ce8073c0dc56d5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_20ce8073c0dc56d5(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_20ce8073c0dc56d5(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_20ce8073c0dc56d5(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_20ce8073c0dc56d5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_20ce8073c0dc56d5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_20ce8073c0dc56d5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(double)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 18432 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×16(12×16) {0..12}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":18432,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (144 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<double> tempShrMem = localShrMem0 + (128);
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v5_batchId0 * 96 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v5_batchId0 * 192 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v17_i1 = 0; v17_i1 < 16; ++v17_i1) {
                tensorforge::intel_esimd::simd<double, 12> v22_data;
                v22_data.copy_from(glb_m1 + ((v17_i1 * 12)));
                r0.template select<12, 1>((v17_i1 * 16)) = v22_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<double, 32> v25_ld;
              v25_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v25_ld);
              tensorforge::intel_esimd::simd<double, 32> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 32));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v26_ld);
              tensorforge::intel_esimd::simd<double, 32> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 64), v27_ld);
              tensorforge::intel_esimd::simd<double, 32> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 96));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 96), v28_ld);
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<double, 128> r1(0.0);
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
                tensorforge::intel_esimd::simd<double, 12> v35_data;
                v35_data.copy_from(glb_m0 + ((v30_i1 * 12)));
                r1.template select<12, 1>((v30_i1 * 16)) = v35_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<double, 128> r2(0.0);
              // ir2 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 128> ir2(0.0);
              tensorforge::intel_esimd::simd<double, 16> v40_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 128> s0_w0 = tensorforge::slmLoad<double, 128>(s0 + 0);
              double v41_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 16> v43_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v43_data + (v40_data * v41_data));
              double v46_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v48_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v48_data + (v40_data * v46_data));
              double v51_data = s0_w0[32];
              tensorforge::intel_esimd::simd<double, 16> v53_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v53_data + (v40_data * v51_data));
              double v56_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 16> v58_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v58_data + (v40_data * v56_data));
              double v61_data = s0_w0[64];
              tensorforge::intel_esimd::simd<double, 16> v63_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v63_data + (v40_data * v61_data));
              double v66_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 16> v68_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v68_data + (v40_data * v66_data));
              double v71_data = s0_w0[96];
              tensorforge::intel_esimd::simd<double, 16> v73_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v73_data + (v40_data * v71_data));
              double v76_data = s0_w0[112];
              tensorforge::intel_esimd::simd<double, 16> v78_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v78_data + (v40_data * v76_data));
              tensorforge::intel_esimd::simd<double, 16> v80_data(r0.template select<16, 1>(16));
              double v81_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 16> v83_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v83_data + (v80_data * v81_data));
              double v86_data = s0_w0[17];
              tensorforge::intel_esimd::simd<double, 16> v88_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v88_data + (v80_data * v86_data));
              double v91_data = s0_w0[33];
              tensorforge::intel_esimd::simd<double, 16> v93_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v93_data + (v80_data * v91_data));
              double v96_data = s0_w0[49];
              tensorforge::intel_esimd::simd<double, 16> v98_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v98_data + (v80_data * v96_data));
              double v101_data = s0_w0[65];
              tensorforge::intel_esimd::simd<double, 16> v103_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v103_data + (v80_data * v101_data));
              double v106_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 16> v108_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v108_data + (v80_data * v106_data));
              double v111_data = s0_w0[97];
              tensorforge::intel_esimd::simd<double, 16> v113_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v113_data + (v80_data * v111_data));
              double v116_data = s0_w0[113];
              tensorforge::intel_esimd::simd<double, 16> v118_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v118_data + (v80_data * v116_data));
              tensorforge::intel_esimd::simd<double, 16> v120_data(r0.template select<16, 1>(32));
              double v121_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 16> v123_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v123_data + (v120_data * v121_data));
              double v126_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 16> v128_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v128_data + (v120_data * v126_data));
              double v131_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v133_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v133_data + (v120_data * v131_data));
              double v136_data = s0_w0[50];
              tensorforge::intel_esimd::simd<double, 16> v138_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v138_data + (v120_data * v136_data));
              double v141_data = s0_w0[66];
              tensorforge::intel_esimd::simd<double, 16> v143_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v143_data + (v120_data * v141_data));
              double v146_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 16> v148_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v148_data + (v120_data * v146_data));
              double v151_data = s0_w0[98];
              tensorforge::intel_esimd::simd<double, 16> v153_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v153_data + (v120_data * v151_data));
              double v156_data = s0_w0[114];
              tensorforge::intel_esimd::simd<double, 16> v158_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v158_data + (v120_data * v156_data));
              tensorforge::intel_esimd::simd<double, 16> v160_data(r0.template select<16, 1>(48));
              double v161_data = s0_w0[3];
              tensorforge::intel_esimd::simd<double, 16> v163_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v163_data + (v160_data * v161_data));
              double v166_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 16> v168_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v168_data + (v160_data * v166_data));
              double v171_data = s0_w0[35];
              tensorforge::intel_esimd::simd<double, 16> v173_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v173_data + (v160_data * v171_data));
              double v176_data = s0_w0[51];
              tensorforge::intel_esimd::simd<double, 16> v178_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v178_data + (v160_data * v176_data));
              double v181_data = s0_w0[67];
              tensorforge::intel_esimd::simd<double, 16> v183_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v183_data + (v160_data * v181_data));
              double v186_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 16> v188_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v188_data + (v160_data * v186_data));
              double v191_data = s0_w0[99];
              tensorforge::intel_esimd::simd<double, 16> v193_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v193_data + (v160_data * v191_data));
              double v196_data = s0_w0[115];
              tensorforge::intel_esimd::simd<double, 16> v198_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v198_data + (v160_data * v196_data));
              tensorforge::intel_esimd::simd<double, 16> v200_data(r0.template select<16, 1>(64));
              double v201_data = s0_w0[4];
              tensorforge::intel_esimd::simd<double, 16> v203_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v203_data + (v200_data * v201_data));
              double v206_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 16> v208_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v208_data + (v200_data * v206_data));
              double v211_data = s0_w0[36];
              tensorforge::intel_esimd::simd<double, 16> v213_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v213_data + (v200_data * v211_data));
              double v216_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v218_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v218_data + (v200_data * v216_data));
              double v221_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 16> v223_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v223_data + (v200_data * v221_data));
              double v226_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 16> v228_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v228_data + (v200_data * v226_data));
              double v231_data = s0_w0[100];
              tensorforge::intel_esimd::simd<double, 16> v233_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v233_data + (v200_data * v231_data));
              double v236_data = s0_w0[116];
              tensorforge::intel_esimd::simd<double, 16> v238_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v238_data + (v200_data * v236_data));
              tensorforge::intel_esimd::simd<double, 16> v240_data(r0.template select<16, 1>(80));
              double v241_data = s0_w0[5];
              tensorforge::intel_esimd::simd<double, 16> v243_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v243_data + (v240_data * v241_data));
              double v246_data = s0_w0[21];
              tensorforge::intel_esimd::simd<double, 16> v248_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v248_data + (v240_data * v246_data));
              double v251_data = s0_w0[37];
              tensorforge::intel_esimd::simd<double, 16> v253_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v253_data + (v240_data * v251_data));
              double v256_data = s0_w0[53];
              tensorforge::intel_esimd::simd<double, 16> v258_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v258_data + (v240_data * v256_data));
              double v261_data = s0_w0[69];
              tensorforge::intel_esimd::simd<double, 16> v263_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v263_data + (v240_data * v261_data));
              double v266_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 16> v268_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v268_data + (v240_data * v266_data));
              double v271_data = s0_w0[101];
              tensorforge::intel_esimd::simd<double, 16> v273_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v273_data + (v240_data * v271_data));
              double v276_data = s0_w0[117];
              tensorforge::intel_esimd::simd<double, 16> v278_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v278_data + (v240_data * v276_data));
              tensorforge::intel_esimd::simd<double, 16> v280_data(r0.template select<16, 1>(96));
              double v281_data = s0_w0[6];
              tensorforge::intel_esimd::simd<double, 16> v283_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v283_data + (v280_data * v281_data));
              double v286_data = s0_w0[22];
              tensorforge::intel_esimd::simd<double, 16> v288_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v288_data + (v280_data * v286_data));
              double v291_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 16> v293_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v293_data + (v280_data * v291_data));
              double v296_data = s0_w0[54];
              tensorforge::intel_esimd::simd<double, 16> v298_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v298_data + (v280_data * v296_data));
              double v301_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v303_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v303_data + (v280_data * v301_data));
              double v306_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 16> v308_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v308_data + (v280_data * v306_data));
              double v311_data = s0_w0[102];
              tensorforge::intel_esimd::simd<double, 16> v313_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v313_data + (v280_data * v311_data));
              double v316_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v318_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v318_data + (v280_data * v316_data));
              tensorforge::intel_esimd::simd<double, 16> v320_data(r0.template select<16, 1>(112));
              double v321_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v323_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v323_data + (v320_data * v321_data));
              double v326_data = s0_w0[23];
              tensorforge::intel_esimd::simd<double, 16> v328_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v328_data + (v320_data * v326_data));
              double v331_data = s0_w0[39];
              tensorforge::intel_esimd::simd<double, 16> v333_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v333_data + (v320_data * v331_data));
              double v336_data = s0_w0[55];
              tensorforge::intel_esimd::simd<double, 16> v338_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v338_data + (v320_data * v336_data));
              double v341_data = s0_w0[71];
              tensorforge::intel_esimd::simd<double, 16> v343_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v343_data + (v320_data * v341_data));
              double v346_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 16> v348_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v348_data + (v320_data * v346_data));
              double v351_data = s0_w0[103];
              tensorforge::intel_esimd::simd<double, 16> v353_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v353_data + (v320_data * v351_data));
              double v356_data = s0_w0[119];
              tensorforge::intel_esimd::simd<double, 16> v358_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v358_data + (v320_data * v356_data));
              tensorforge::intel_esimd::simd<double, 16> v360_data(r0.template select<16, 1>(128));
              double v361_data = s0_w0[8];
              tensorforge::intel_esimd::simd<double, 16> v363_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v363_data + (v360_data * v361_data));
              double v366_data = s0_w0[24];
              tensorforge::intel_esimd::simd<double, 16> v368_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v368_data + (v360_data * v366_data));
              double v371_data = s0_w0[40];
              tensorforge::intel_esimd::simd<double, 16> v373_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v373_data + (v360_data * v371_data));
              double v376_data = s0_w0[56];
              tensorforge::intel_esimd::simd<double, 16> v378_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v378_data + (v360_data * v376_data));
              double v381_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v383_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v383_data + (v360_data * v381_data));
              double v386_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 16> v388_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v388_data + (v360_data * v386_data));
              double v391_data = s0_w0[104];
              tensorforge::intel_esimd::simd<double, 16> v393_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v393_data + (v360_data * v391_data));
              double v396_data = s0_w0[120];
              tensorforge::intel_esimd::simd<double, 16> v398_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v398_data + (v360_data * v396_data));
              tensorforge::intel_esimd::simd<double, 16> v400_data(r0.template select<16, 1>(144));
              double v401_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 16> v403_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v403_data + (v400_data * v401_data));
              double v406_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v408_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v408_data + (v400_data * v406_data));
              double v411_data = s0_w0[41];
              tensorforge::intel_esimd::simd<double, 16> v413_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v413_data + (v400_data * v411_data));
              double v416_data = s0_w0[57];
              tensorforge::intel_esimd::simd<double, 16> v418_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v418_data + (v400_data * v416_data));
              double v421_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v423_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v423_data + (v400_data * v421_data));
              double v426_data = s0_w0[89];
              tensorforge::intel_esimd::simd<double, 16> v428_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v428_data + (v400_data * v426_data));
              double v431_data = s0_w0[105];
              tensorforge::intel_esimd::simd<double, 16> v433_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v433_data + (v400_data * v431_data));
              double v436_data = s0_w0[121];
              tensorforge::intel_esimd::simd<double, 16> v438_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v438_data + (v400_data * v436_data));
              tensorforge::intel_esimd::simd<double, 16> v440_data(r0.template select<16, 1>(160));
              double v441_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 16> v443_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v443_data + (v440_data * v441_data));
              double v446_data = s0_w0[26];
              tensorforge::intel_esimd::simd<double, 16> v448_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v448_data + (v440_data * v446_data));
              double v451_data = s0_w0[42];
              tensorforge::intel_esimd::simd<double, 16> v453_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v453_data + (v440_data * v451_data));
              double v456_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 16> v458_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v458_data + (v440_data * v456_data));
              double v461_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v463_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v463_data + (v440_data * v461_data));
              double v466_data = s0_w0[90];
              tensorforge::intel_esimd::simd<double, 16> v468_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v468_data + (v440_data * v466_data));
              double v471_data = s0_w0[106];
              tensorforge::intel_esimd::simd<double, 16> v473_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v473_data + (v440_data * v471_data));
              double v476_data = s0_w0[122];
              tensorforge::intel_esimd::simd<double, 16> v478_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v478_data + (v440_data * v476_data));
              tensorforge::intel_esimd::simd<double, 16> v480_data(r0.template select<16, 1>(176));
              double v481_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 16> v483_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v483_data + (v480_data * v481_data));
              double v486_data = s0_w0[27];
              tensorforge::intel_esimd::simd<double, 16> v488_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v488_data + (v480_data * v486_data));
              double v491_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v493_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v493_data + (v480_data * v491_data));
              double v496_data = s0_w0[59];
              tensorforge::intel_esimd::simd<double, 16> v498_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v498_data + (v480_data * v496_data));
              double v501_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v503_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v503_data + (v480_data * v501_data));
              double v506_data = s0_w0[91];
              tensorforge::intel_esimd::simd<double, 16> v508_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v508_data + (v480_data * v506_data));
              double v511_data = s0_w0[107];
              tensorforge::intel_esimd::simd<double, 16> v513_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v513_data + (v480_data * v511_data));
              double v516_data = s0_w0[123];
              tensorforge::intel_esimd::simd<double, 16> v518_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v518_data + (v480_data * v516_data));
              tensorforge::intel_esimd::simd<double, 16> v520_data(r0.template select<16, 1>(192));
              double v521_data = s0_w0[12];
              tensorforge::intel_esimd::simd<double, 16> v523_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v523_data + (v520_data * v521_data));
              double v526_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 16> v528_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v528_data + (v520_data * v526_data));
              double v531_data = s0_w0[44];
              tensorforge::intel_esimd::simd<double, 16> v533_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v533_data + (v520_data * v531_data));
              double v536_data = s0_w0[60];
              tensorforge::intel_esimd::simd<double, 16> v538_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v538_data + (v520_data * v536_data));
              double v541_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v543_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v543_data + (v520_data * v541_data));
              double v546_data = s0_w0[92];
              tensorforge::intel_esimd::simd<double, 16> v548_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v548_data + (v520_data * v546_data));
              double v551_data = s0_w0[108];
              tensorforge::intel_esimd::simd<double, 16> v553_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v553_data + (v520_data * v551_data));
              double v556_data = s0_w0[124];
              tensorforge::intel_esimd::simd<double, 16> v558_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v558_data + (v520_data * v556_data));
              tensorforge::intel_esimd::simd<double, 16> v560_data(r0.template select<16, 1>(208));
              double v561_data = s0_w0[13];
              tensorforge::intel_esimd::simd<double, 16> v563_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v563_data + (v560_data * v561_data));
              double v566_data = s0_w0[29];
              tensorforge::intel_esimd::simd<double, 16> v568_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v568_data + (v560_data * v566_data));
              double v571_data = s0_w0[45];
              tensorforge::intel_esimd::simd<double, 16> v573_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v573_data + (v560_data * v571_data));
              double v576_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v578_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v578_data + (v560_data * v576_data));
              double v581_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v583_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v583_data + (v560_data * v581_data));
              double v586_data = s0_w0[93];
              tensorforge::intel_esimd::simd<double, 16> v588_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v588_data + (v560_data * v586_data));
              double v591_data = s0_w0[109];
              tensorforge::intel_esimd::simd<double, 16> v593_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v593_data + (v560_data * v591_data));
              double v596_data = s0_w0[125];
              tensorforge::intel_esimd::simd<double, 16> v598_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v598_data + (v560_data * v596_data));
              tensorforge::intel_esimd::simd<double, 16> v600_data(r0.template select<16, 1>(224));
              double v601_data = s0_w0[14];
              tensorforge::intel_esimd::simd<double, 16> v603_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v603_data + (v600_data * v601_data));
              double v606_data = s0_w0[30];
              tensorforge::intel_esimd::simd<double, 16> v608_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v608_data + (v600_data * v606_data));
              double v611_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v613_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v613_data + (v600_data * v611_data));
              double v616_data = s0_w0[62];
              tensorforge::intel_esimd::simd<double, 16> v618_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v618_data + (v600_data * v616_data));
              double v621_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v623_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v623_data + (v600_data * v621_data));
              double v626_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v628_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v628_data + (v600_data * v626_data));
              double v631_data = s0_w0[110];
              tensorforge::intel_esimd::simd<double, 16> v633_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v633_data + (v600_data * v631_data));
              double v636_data = s0_w0[126];
              tensorforge::intel_esimd::simd<double, 16> v638_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v638_data + (v600_data * v636_data));
              tensorforge::intel_esimd::simd<double, 16> v640_data(r0.template select<16, 1>(240));
              double v641_data = s0_w0[15];
              tensorforge::intel_esimd::simd<double, 16> v643_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v643_data + (v640_data * v641_data));
              double v646_data = s0_w0[31];
              tensorforge::intel_esimd::simd<double, 16> v648_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v648_data + (v640_data * v646_data));
              double v651_data = s0_w0[47];
              tensorforge::intel_esimd::simd<double, 16> v653_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v653_data + (v640_data * v651_data));
              double v656_data = s0_w0[63];
              tensorforge::intel_esimd::simd<double, 16> v658_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v658_data + (v640_data * v656_data));
              double v661_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v663_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v663_data + (v640_data * v661_data));
              double v666_data = s0_w0[95];
              tensorforge::intel_esimd::simd<double, 16> v668_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v668_data + (v640_data * v666_data));
              double v671_data = s0_w0[111];
              tensorforge::intel_esimd::simd<double, 16> v673_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v673_data + (v640_data * v671_data));
              double v676_data = s0_w0[127];
              tensorforge::intel_esimd::simd<double, 16> v678_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v678_data + (v640_data * v676_data));
              // r2 = ir2 + r1
              #pragma unroll
              for (int32_t v680_n1 = 0; v680_n1 < 8; ++v680_n1) {
                int32_t v681_a = v680_n1 * 16;
                tensorforge::intel_esimd::simd<double, 12> v683_data(ir2.template select<12, 1>(v681_a));
                tensorforge::intel_esimd::simd<double, 12> v684_data(r1.template select<12, 1>(v681_a));
                r2.template select<12, 1>(v681_a) = (v684_data + v683_data);
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v686_i1 = 0; v686_i1 < 8; ++v686_i1) {
                tensorforge::intel_esimd::simd<double, 12> v689_data(r2.template select<12, 1>((v686_i1 * 16)));
                v689_data.copy_to(glb_m0 + ((v686_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


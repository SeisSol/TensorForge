// === base name ===
kernel_51fef0e44ec0bc2d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_51fef0e44ec0bc2d = {{1, 16, 1}, 16, 16, 1, 16, 18432, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_51fef0e44ec0bc2d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_51fef0e44ec0bc2d(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_51fef0e44ec0bc2d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_51fef0e44ec0bc2d(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_51fef0e44ec0bc2d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_51fef0e44ec0bc2d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_51fef0e44ec0bc2d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(double)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 18432 B shared, occupancy grid
        // operands:
        //   m0 16×8(16×8) {0..16}×{0..8} strided
        //   m1 32×32(32×32) {0..32}×{0..32} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{8..24}×{8..24} × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":18432,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[8,8],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              double *const __restrict__ glb_m0 = &m0[v5_batchId0 * 128 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v5_batchId0 * 1024 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v19_lead = v17_i0 * 16;
                int32_t v21_off = v19_lead + 8;
                #pragma unroll
                for (int32_t v18_i1 = 8; v18_i1 < 24; ++v18_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v24_data;
                  v24_data.copy_from(glb_m1 + ((v21_off + (v18_i1 * 32))));
                  r0.template select<16, 1>((v19_lead + ((v18_i1 - 8) * 16))) = v24_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<double, 32> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v28_ld);
              tensorforge::intel_esimd::simd<double, 32> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 32));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v29_ld);
              tensorforge::intel_esimd::simd<double, 32> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 64), v30_ld);
              tensorforge::intel_esimd::simd<double, 32> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 96));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 96), v31_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<double, 128> r1(0.0);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 128> ir1(0.0);
              tensorforge::intel_esimd::simd<double, 16> v34_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 128> s0_w0 = tensorforge::slmLoad<double, 128>(s0 + 0);
              double v35_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 16> v37_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v37_data + (v34_data * v35_data));
              double v40_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v42_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v42_data + (v34_data * v40_data));
              double v45_data = s0_w0[32];
              tensorforge::intel_esimd::simd<double, 16> v47_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v47_data + (v34_data * v45_data));
              double v50_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 16> v52_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v52_data + (v34_data * v50_data));
              double v55_data = s0_w0[64];
              tensorforge::intel_esimd::simd<double, 16> v57_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v57_data + (v34_data * v55_data));
              double v60_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 16> v62_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v62_data + (v34_data * v60_data));
              double v65_data = s0_w0[96];
              tensorforge::intel_esimd::simd<double, 16> v67_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v67_data + (v34_data * v65_data));
              double v70_data = s0_w0[112];
              tensorforge::intel_esimd::simd<double, 16> v72_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v72_data + (v34_data * v70_data));
              tensorforge::intel_esimd::simd<double, 16> v74_data(r0.template select<16, 1>(16));
              double v75_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 16> v77_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v77_data + (v74_data * v75_data));
              double v80_data = s0_w0[17];
              tensorforge::intel_esimd::simd<double, 16> v82_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v82_data + (v74_data * v80_data));
              double v85_data = s0_w0[33];
              tensorforge::intel_esimd::simd<double, 16> v87_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v87_data + (v74_data * v85_data));
              double v90_data = s0_w0[49];
              tensorforge::intel_esimd::simd<double, 16> v92_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v92_data + (v74_data * v90_data));
              double v95_data = s0_w0[65];
              tensorforge::intel_esimd::simd<double, 16> v97_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v97_data + (v74_data * v95_data));
              double v100_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 16> v102_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v102_data + (v74_data * v100_data));
              double v105_data = s0_w0[97];
              tensorforge::intel_esimd::simd<double, 16> v107_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v107_data + (v74_data * v105_data));
              double v110_data = s0_w0[113];
              tensorforge::intel_esimd::simd<double, 16> v112_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v112_data + (v74_data * v110_data));
              tensorforge::intel_esimd::simd<double, 16> v114_data(r0.template select<16, 1>(32));
              double v115_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 16> v117_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v117_data + (v114_data * v115_data));
              double v120_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 16> v122_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v122_data + (v114_data * v120_data));
              double v125_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v127_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v127_data + (v114_data * v125_data));
              double v130_data = s0_w0[50];
              tensorforge::intel_esimd::simd<double, 16> v132_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v132_data + (v114_data * v130_data));
              double v135_data = s0_w0[66];
              tensorforge::intel_esimd::simd<double, 16> v137_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v137_data + (v114_data * v135_data));
              double v140_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 16> v142_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v142_data + (v114_data * v140_data));
              double v145_data = s0_w0[98];
              tensorforge::intel_esimd::simd<double, 16> v147_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v147_data + (v114_data * v145_data));
              double v150_data = s0_w0[114];
              tensorforge::intel_esimd::simd<double, 16> v152_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v152_data + (v114_data * v150_data));
              tensorforge::intel_esimd::simd<double, 16> v154_data(r0.template select<16, 1>(48));
              double v155_data = s0_w0[3];
              tensorforge::intel_esimd::simd<double, 16> v157_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v157_data + (v154_data * v155_data));
              double v160_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 16> v162_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v162_data + (v154_data * v160_data));
              double v165_data = s0_w0[35];
              tensorforge::intel_esimd::simd<double, 16> v167_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v167_data + (v154_data * v165_data));
              double v170_data = s0_w0[51];
              tensorforge::intel_esimd::simd<double, 16> v172_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v172_data + (v154_data * v170_data));
              double v175_data = s0_w0[67];
              tensorforge::intel_esimd::simd<double, 16> v177_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v177_data + (v154_data * v175_data));
              double v180_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 16> v182_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v182_data + (v154_data * v180_data));
              double v185_data = s0_w0[99];
              tensorforge::intel_esimd::simd<double, 16> v187_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v187_data + (v154_data * v185_data));
              double v190_data = s0_w0[115];
              tensorforge::intel_esimd::simd<double, 16> v192_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v192_data + (v154_data * v190_data));
              tensorforge::intel_esimd::simd<double, 16> v194_data(r0.template select<16, 1>(64));
              double v195_data = s0_w0[4];
              tensorforge::intel_esimd::simd<double, 16> v197_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v197_data + (v194_data * v195_data));
              double v200_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 16> v202_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v202_data + (v194_data * v200_data));
              double v205_data = s0_w0[36];
              tensorforge::intel_esimd::simd<double, 16> v207_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v207_data + (v194_data * v205_data));
              double v210_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v212_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v212_data + (v194_data * v210_data));
              double v215_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 16> v217_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v217_data + (v194_data * v215_data));
              double v220_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 16> v222_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v222_data + (v194_data * v220_data));
              double v225_data = s0_w0[100];
              tensorforge::intel_esimd::simd<double, 16> v227_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v227_data + (v194_data * v225_data));
              double v230_data = s0_w0[116];
              tensorforge::intel_esimd::simd<double, 16> v232_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v232_data + (v194_data * v230_data));
              tensorforge::intel_esimd::simd<double, 16> v234_data(r0.template select<16, 1>(80));
              double v235_data = s0_w0[5];
              tensorforge::intel_esimd::simd<double, 16> v237_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v237_data + (v234_data * v235_data));
              double v240_data = s0_w0[21];
              tensorforge::intel_esimd::simd<double, 16> v242_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v242_data + (v234_data * v240_data));
              double v245_data = s0_w0[37];
              tensorforge::intel_esimd::simd<double, 16> v247_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v247_data + (v234_data * v245_data));
              double v250_data = s0_w0[53];
              tensorforge::intel_esimd::simd<double, 16> v252_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v252_data + (v234_data * v250_data));
              double v255_data = s0_w0[69];
              tensorforge::intel_esimd::simd<double, 16> v257_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v257_data + (v234_data * v255_data));
              double v260_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 16> v262_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v262_data + (v234_data * v260_data));
              double v265_data = s0_w0[101];
              tensorforge::intel_esimd::simd<double, 16> v267_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v267_data + (v234_data * v265_data));
              double v270_data = s0_w0[117];
              tensorforge::intel_esimd::simd<double, 16> v272_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v272_data + (v234_data * v270_data));
              tensorforge::intel_esimd::simd<double, 16> v274_data(r0.template select<16, 1>(96));
              double v275_data = s0_w0[6];
              tensorforge::intel_esimd::simd<double, 16> v277_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v277_data + (v274_data * v275_data));
              double v280_data = s0_w0[22];
              tensorforge::intel_esimd::simd<double, 16> v282_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v282_data + (v274_data * v280_data));
              double v285_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 16> v287_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v287_data + (v274_data * v285_data));
              double v290_data = s0_w0[54];
              tensorforge::intel_esimd::simd<double, 16> v292_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v292_data + (v274_data * v290_data));
              double v295_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v297_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v297_data + (v274_data * v295_data));
              double v300_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 16> v302_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v302_data + (v274_data * v300_data));
              double v305_data = s0_w0[102];
              tensorforge::intel_esimd::simd<double, 16> v307_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v307_data + (v274_data * v305_data));
              double v310_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v312_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v312_data + (v274_data * v310_data));
              tensorforge::intel_esimd::simd<double, 16> v314_data(r0.template select<16, 1>(112));
              double v315_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v317_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v317_data + (v314_data * v315_data));
              double v320_data = s0_w0[23];
              tensorforge::intel_esimd::simd<double, 16> v322_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v322_data + (v314_data * v320_data));
              double v325_data = s0_w0[39];
              tensorforge::intel_esimd::simd<double, 16> v327_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v327_data + (v314_data * v325_data));
              double v330_data = s0_w0[55];
              tensorforge::intel_esimd::simd<double, 16> v332_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v332_data + (v314_data * v330_data));
              double v335_data = s0_w0[71];
              tensorforge::intel_esimd::simd<double, 16> v337_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v337_data + (v314_data * v335_data));
              double v340_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 16> v342_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v342_data + (v314_data * v340_data));
              double v345_data = s0_w0[103];
              tensorforge::intel_esimd::simd<double, 16> v347_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v347_data + (v314_data * v345_data));
              double v350_data = s0_w0[119];
              tensorforge::intel_esimd::simd<double, 16> v352_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v352_data + (v314_data * v350_data));
              tensorforge::intel_esimd::simd<double, 16> v354_data(r0.template select<16, 1>(128));
              double v355_data = s0_w0[8];
              tensorforge::intel_esimd::simd<double, 16> v357_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v357_data + (v354_data * v355_data));
              double v360_data = s0_w0[24];
              tensorforge::intel_esimd::simd<double, 16> v362_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v362_data + (v354_data * v360_data));
              double v365_data = s0_w0[40];
              tensorforge::intel_esimd::simd<double, 16> v367_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v367_data + (v354_data * v365_data));
              double v370_data = s0_w0[56];
              tensorforge::intel_esimd::simd<double, 16> v372_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v372_data + (v354_data * v370_data));
              double v375_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v377_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v377_data + (v354_data * v375_data));
              double v380_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 16> v382_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v382_data + (v354_data * v380_data));
              double v385_data = s0_w0[104];
              tensorforge::intel_esimd::simd<double, 16> v387_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v387_data + (v354_data * v385_data));
              double v390_data = s0_w0[120];
              tensorforge::intel_esimd::simd<double, 16> v392_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v392_data + (v354_data * v390_data));
              tensorforge::intel_esimd::simd<double, 16> v394_data(r0.template select<16, 1>(144));
              double v395_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 16> v397_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v397_data + (v394_data * v395_data));
              double v400_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v402_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v402_data + (v394_data * v400_data));
              double v405_data = s0_w0[41];
              tensorforge::intel_esimd::simd<double, 16> v407_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v407_data + (v394_data * v405_data));
              double v410_data = s0_w0[57];
              tensorforge::intel_esimd::simd<double, 16> v412_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v412_data + (v394_data * v410_data));
              double v415_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v417_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v417_data + (v394_data * v415_data));
              double v420_data = s0_w0[89];
              tensorforge::intel_esimd::simd<double, 16> v422_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v422_data + (v394_data * v420_data));
              double v425_data = s0_w0[105];
              tensorforge::intel_esimd::simd<double, 16> v427_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v427_data + (v394_data * v425_data));
              double v430_data = s0_w0[121];
              tensorforge::intel_esimd::simd<double, 16> v432_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v432_data + (v394_data * v430_data));
              tensorforge::intel_esimd::simd<double, 16> v434_data(r0.template select<16, 1>(160));
              double v435_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 16> v437_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v437_data + (v434_data * v435_data));
              double v440_data = s0_w0[26];
              tensorforge::intel_esimd::simd<double, 16> v442_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v442_data + (v434_data * v440_data));
              double v445_data = s0_w0[42];
              tensorforge::intel_esimd::simd<double, 16> v447_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v447_data + (v434_data * v445_data));
              double v450_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 16> v452_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v452_data + (v434_data * v450_data));
              double v455_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v457_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v457_data + (v434_data * v455_data));
              double v460_data = s0_w0[90];
              tensorforge::intel_esimd::simd<double, 16> v462_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v462_data + (v434_data * v460_data));
              double v465_data = s0_w0[106];
              tensorforge::intel_esimd::simd<double, 16> v467_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v467_data + (v434_data * v465_data));
              double v470_data = s0_w0[122];
              tensorforge::intel_esimd::simd<double, 16> v472_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v472_data + (v434_data * v470_data));
              tensorforge::intel_esimd::simd<double, 16> v474_data(r0.template select<16, 1>(176));
              double v475_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 16> v477_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v477_data + (v474_data * v475_data));
              double v480_data = s0_w0[27];
              tensorforge::intel_esimd::simd<double, 16> v482_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v482_data + (v474_data * v480_data));
              double v485_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v487_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v487_data + (v474_data * v485_data));
              double v490_data = s0_w0[59];
              tensorforge::intel_esimd::simd<double, 16> v492_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v492_data + (v474_data * v490_data));
              double v495_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v497_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v497_data + (v474_data * v495_data));
              double v500_data = s0_w0[91];
              tensorforge::intel_esimd::simd<double, 16> v502_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v502_data + (v474_data * v500_data));
              double v505_data = s0_w0[107];
              tensorforge::intel_esimd::simd<double, 16> v507_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v507_data + (v474_data * v505_data));
              double v510_data = s0_w0[123];
              tensorforge::intel_esimd::simd<double, 16> v512_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v512_data + (v474_data * v510_data));
              tensorforge::intel_esimd::simd<double, 16> v514_data(r0.template select<16, 1>(192));
              double v515_data = s0_w0[12];
              tensorforge::intel_esimd::simd<double, 16> v517_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v517_data + (v514_data * v515_data));
              double v520_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 16> v522_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v522_data + (v514_data * v520_data));
              double v525_data = s0_w0[44];
              tensorforge::intel_esimd::simd<double, 16> v527_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v527_data + (v514_data * v525_data));
              double v530_data = s0_w0[60];
              tensorforge::intel_esimd::simd<double, 16> v532_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v532_data + (v514_data * v530_data));
              double v535_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v537_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v537_data + (v514_data * v535_data));
              double v540_data = s0_w0[92];
              tensorforge::intel_esimd::simd<double, 16> v542_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v542_data + (v514_data * v540_data));
              double v545_data = s0_w0[108];
              tensorforge::intel_esimd::simd<double, 16> v547_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v547_data + (v514_data * v545_data));
              double v550_data = s0_w0[124];
              tensorforge::intel_esimd::simd<double, 16> v552_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v552_data + (v514_data * v550_data));
              tensorforge::intel_esimd::simd<double, 16> v554_data(r0.template select<16, 1>(208));
              double v555_data = s0_w0[13];
              tensorforge::intel_esimd::simd<double, 16> v557_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v557_data + (v554_data * v555_data));
              double v560_data = s0_w0[29];
              tensorforge::intel_esimd::simd<double, 16> v562_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v562_data + (v554_data * v560_data));
              double v565_data = s0_w0[45];
              tensorforge::intel_esimd::simd<double, 16> v567_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v567_data + (v554_data * v565_data));
              double v570_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v572_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v572_data + (v554_data * v570_data));
              double v575_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v577_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v577_data + (v554_data * v575_data));
              double v580_data = s0_w0[93];
              tensorforge::intel_esimd::simd<double, 16> v582_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v582_data + (v554_data * v580_data));
              double v585_data = s0_w0[109];
              tensorforge::intel_esimd::simd<double, 16> v587_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v587_data + (v554_data * v585_data));
              double v590_data = s0_w0[125];
              tensorforge::intel_esimd::simd<double, 16> v592_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v592_data + (v554_data * v590_data));
              tensorforge::intel_esimd::simd<double, 16> v594_data(r0.template select<16, 1>(224));
              double v595_data = s0_w0[14];
              tensorforge::intel_esimd::simd<double, 16> v597_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v597_data + (v594_data * v595_data));
              double v600_data = s0_w0[30];
              tensorforge::intel_esimd::simd<double, 16> v602_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v602_data + (v594_data * v600_data));
              double v605_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v607_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v607_data + (v594_data * v605_data));
              double v610_data = s0_w0[62];
              tensorforge::intel_esimd::simd<double, 16> v612_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v612_data + (v594_data * v610_data));
              double v615_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v617_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v617_data + (v594_data * v615_data));
              double v620_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v622_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v622_data + (v594_data * v620_data));
              double v625_data = s0_w0[110];
              tensorforge::intel_esimd::simd<double, 16> v627_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v627_data + (v594_data * v625_data));
              double v630_data = s0_w0[126];
              tensorforge::intel_esimd::simd<double, 16> v632_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v632_data + (v594_data * v630_data));
              tensorforge::intel_esimd::simd<double, 16> v634_data(r0.template select<16, 1>(240));
              double v635_data = s0_w0[15];
              tensorforge::intel_esimd::simd<double, 16> v637_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v637_data + (v634_data * v635_data));
              double v640_data = s0_w0[31];
              tensorforge::intel_esimd::simd<double, 16> v642_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v642_data + (v634_data * v640_data));
              double v645_data = s0_w0[47];
              tensorforge::intel_esimd::simd<double, 16> v647_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v647_data + (v634_data * v645_data));
              double v650_data = s0_w0[63];
              tensorforge::intel_esimd::simd<double, 16> v652_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v652_data + (v634_data * v650_data));
              double v655_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v657_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v657_data + (v634_data * v655_data));
              double v660_data = s0_w0[95];
              tensorforge::intel_esimd::simd<double, 16> v662_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v662_data + (v634_data * v660_data));
              double v665_data = s0_w0[111];
              tensorforge::intel_esimd::simd<double, 16> v667_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v667_data + (v634_data * v665_data));
              double v670_data = s0_w0[127];
              tensorforge::intel_esimd::simd<double, 16> v672_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v672_data + (v634_data * v670_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v674_n0 = 0; v674_n0 < 1; ++v674_n0) {
                int32_t v676_a = v674_n0 * 16;
                #pragma unroll
                for (int32_t v675_n1 = 0; v675_n1 < 8; ++v675_n1) {
                  int32_t v678_a = v676_a + (v675_n1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v679_data(ir1.template select<16, 1>(v678_a));
                  r1.template select<16, 1>(v678_a) = v679_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v680_i0 = 0; v680_i0 < 1; ++v680_i0) {
                int32_t v682_a = v680_i0 * 16;
                #pragma unroll
                for (int32_t v681_i1 = 0; v681_i1 < 8; ++v681_i1) {
                  int32_t v684_a = v682_a + (v681_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v685_data(r1.template select<16, 1>(v684_a));
                  v685_data.copy_to(glb_m0 + (v684_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


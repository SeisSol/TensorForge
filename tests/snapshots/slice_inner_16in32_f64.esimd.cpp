// === base name ===
kernel_5ee96d8be5e86e3e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5ee96d8be5e86e3e = {{1, 16, 1}, 16, 16, 1, 16, 18432, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5ee96d8be5e86e3e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5ee96d8be5e86e3e(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5ee96d8be5e86e3e(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 2304 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5ee96d8be5e86e3e(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5ee96d8be5e86e3e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5ee96d8be5e86e3e(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5ee96d8be5e86e3e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const double *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 1024 + 0 + m1_extraOffset];
            const double *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 128 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v5_batchId0 * 128 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v5_batchId0 * 1024 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                int32_t v21_lead = v19_i0 * 16;
                int32_t v23_off = v21_lead + 8;
                #pragma unroll
                for (int32_t v20_i1 = 8; v20_i1 < 24; ++v20_i1) {
                  tensorforge::intel_esimd::simd<double, 16> v26_data;
                  v26_data.copy_from(glb_m1 + ((v23_off + (v20_i1 * 32))));
                  r0.template select<16, 1>((v21_lead + ((v20_i1 - 8) * 16))) = v26_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<double, 32> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 0), v30_ld);
              tensorforge::intel_esimd::simd<double, 32> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 32));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 32), v31_ld);
              tensorforge::intel_esimd::simd<double, 32> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 64), v32_ld);
              tensorforge::intel_esimd::simd<double, 32> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 96));
              tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + 96), v33_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<double, 128> r1(0.0);
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 128> ir1(0.0);
              tensorforge::intel_esimd::simd<double, 16> v36_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 128> s0_w0 = tensorforge::slmLoad<double, 128>(s0 + 0);
              double v37_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 16> v39_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v39_data + (v36_data * v37_data));
              double v42_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v44_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v44_data + (v36_data * v42_data));
              double v47_data = s0_w0[32];
              tensorforge::intel_esimd::simd<double, 16> v49_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v49_data + (v36_data * v47_data));
              double v52_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 16> v54_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v54_data + (v36_data * v52_data));
              double v57_data = s0_w0[64];
              tensorforge::intel_esimd::simd<double, 16> v59_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v59_data + (v36_data * v57_data));
              double v62_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 16> v64_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v64_data + (v36_data * v62_data));
              double v67_data = s0_w0[96];
              tensorforge::intel_esimd::simd<double, 16> v69_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v69_data + (v36_data * v67_data));
              double v72_data = s0_w0[112];
              tensorforge::intel_esimd::simd<double, 16> v74_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v74_data + (v36_data * v72_data));
              tensorforge::intel_esimd::simd<double, 16> v76_data(r0.template select<16, 1>(16));
              double v77_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 16> v79_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v79_data + (v76_data * v77_data));
              double v82_data = s0_w0[17];
              tensorforge::intel_esimd::simd<double, 16> v84_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v84_data + (v76_data * v82_data));
              double v87_data = s0_w0[33];
              tensorforge::intel_esimd::simd<double, 16> v89_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v89_data + (v76_data * v87_data));
              double v92_data = s0_w0[49];
              tensorforge::intel_esimd::simd<double, 16> v94_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v94_data + (v76_data * v92_data));
              double v97_data = s0_w0[65];
              tensorforge::intel_esimd::simd<double, 16> v99_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v99_data + (v76_data * v97_data));
              double v102_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 16> v104_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v104_data + (v76_data * v102_data));
              double v107_data = s0_w0[97];
              tensorforge::intel_esimd::simd<double, 16> v109_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v109_data + (v76_data * v107_data));
              double v112_data = s0_w0[113];
              tensorforge::intel_esimd::simd<double, 16> v114_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v114_data + (v76_data * v112_data));
              tensorforge::intel_esimd::simd<double, 16> v116_data(r0.template select<16, 1>(32));
              double v117_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 16> v119_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v119_data + (v116_data * v117_data));
              double v122_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 16> v124_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v124_data + (v116_data * v122_data));
              double v127_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v129_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v129_data + (v116_data * v127_data));
              double v132_data = s0_w0[50];
              tensorforge::intel_esimd::simd<double, 16> v134_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v134_data + (v116_data * v132_data));
              double v137_data = s0_w0[66];
              tensorforge::intel_esimd::simd<double, 16> v139_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v139_data + (v116_data * v137_data));
              double v142_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 16> v144_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v144_data + (v116_data * v142_data));
              double v147_data = s0_w0[98];
              tensorforge::intel_esimd::simd<double, 16> v149_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v149_data + (v116_data * v147_data));
              double v152_data = s0_w0[114];
              tensorforge::intel_esimd::simd<double, 16> v154_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v154_data + (v116_data * v152_data));
              tensorforge::intel_esimd::simd<double, 16> v156_data(r0.template select<16, 1>(48));
              double v157_data = s0_w0[3];
              tensorforge::intel_esimd::simd<double, 16> v159_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v159_data + (v156_data * v157_data));
              double v162_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 16> v164_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v164_data + (v156_data * v162_data));
              double v167_data = s0_w0[35];
              tensorforge::intel_esimd::simd<double, 16> v169_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v169_data + (v156_data * v167_data));
              double v172_data = s0_w0[51];
              tensorforge::intel_esimd::simd<double, 16> v174_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v174_data + (v156_data * v172_data));
              double v177_data = s0_w0[67];
              tensorforge::intel_esimd::simd<double, 16> v179_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v179_data + (v156_data * v177_data));
              double v182_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 16> v184_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v184_data + (v156_data * v182_data));
              double v187_data = s0_w0[99];
              tensorforge::intel_esimd::simd<double, 16> v189_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v189_data + (v156_data * v187_data));
              double v192_data = s0_w0[115];
              tensorforge::intel_esimd::simd<double, 16> v194_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v194_data + (v156_data * v192_data));
              tensorforge::intel_esimd::simd<double, 16> v196_data(r0.template select<16, 1>(64));
              double v197_data = s0_w0[4];
              tensorforge::intel_esimd::simd<double, 16> v199_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v199_data + (v196_data * v197_data));
              double v202_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 16> v204_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v204_data + (v196_data * v202_data));
              double v207_data = s0_w0[36];
              tensorforge::intel_esimd::simd<double, 16> v209_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v209_data + (v196_data * v207_data));
              double v212_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v214_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v214_data + (v196_data * v212_data));
              double v217_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 16> v219_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v219_data + (v196_data * v217_data));
              double v222_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 16> v224_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v224_data + (v196_data * v222_data));
              double v227_data = s0_w0[100];
              tensorforge::intel_esimd::simd<double, 16> v229_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v229_data + (v196_data * v227_data));
              double v232_data = s0_w0[116];
              tensorforge::intel_esimd::simd<double, 16> v234_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v234_data + (v196_data * v232_data));
              tensorforge::intel_esimd::simd<double, 16> v236_data(r0.template select<16, 1>(80));
              double v237_data = s0_w0[5];
              tensorforge::intel_esimd::simd<double, 16> v239_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v239_data + (v236_data * v237_data));
              double v242_data = s0_w0[21];
              tensorforge::intel_esimd::simd<double, 16> v244_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v244_data + (v236_data * v242_data));
              double v247_data = s0_w0[37];
              tensorforge::intel_esimd::simd<double, 16> v249_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v249_data + (v236_data * v247_data));
              double v252_data = s0_w0[53];
              tensorforge::intel_esimd::simd<double, 16> v254_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v254_data + (v236_data * v252_data));
              double v257_data = s0_w0[69];
              tensorforge::intel_esimd::simd<double, 16> v259_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v259_data + (v236_data * v257_data));
              double v262_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 16> v264_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v264_data + (v236_data * v262_data));
              double v267_data = s0_w0[101];
              tensorforge::intel_esimd::simd<double, 16> v269_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v269_data + (v236_data * v267_data));
              double v272_data = s0_w0[117];
              tensorforge::intel_esimd::simd<double, 16> v274_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v274_data + (v236_data * v272_data));
              tensorforge::intel_esimd::simd<double, 16> v276_data(r0.template select<16, 1>(96));
              double v277_data = s0_w0[6];
              tensorforge::intel_esimd::simd<double, 16> v279_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v279_data + (v276_data * v277_data));
              double v282_data = s0_w0[22];
              tensorforge::intel_esimd::simd<double, 16> v284_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v284_data + (v276_data * v282_data));
              double v287_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 16> v289_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v289_data + (v276_data * v287_data));
              double v292_data = s0_w0[54];
              tensorforge::intel_esimd::simd<double, 16> v294_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v294_data + (v276_data * v292_data));
              double v297_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v299_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v299_data + (v276_data * v297_data));
              double v302_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 16> v304_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v304_data + (v276_data * v302_data));
              double v307_data = s0_w0[102];
              tensorforge::intel_esimd::simd<double, 16> v309_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v309_data + (v276_data * v307_data));
              double v312_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v314_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v314_data + (v276_data * v312_data));
              tensorforge::intel_esimd::simd<double, 16> v316_data(r0.template select<16, 1>(112));
              double v317_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v319_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v319_data + (v316_data * v317_data));
              double v322_data = s0_w0[23];
              tensorforge::intel_esimd::simd<double, 16> v324_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v324_data + (v316_data * v322_data));
              double v327_data = s0_w0[39];
              tensorforge::intel_esimd::simd<double, 16> v329_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v329_data + (v316_data * v327_data));
              double v332_data = s0_w0[55];
              tensorforge::intel_esimd::simd<double, 16> v334_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v334_data + (v316_data * v332_data));
              double v337_data = s0_w0[71];
              tensorforge::intel_esimd::simd<double, 16> v339_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v339_data + (v316_data * v337_data));
              double v342_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 16> v344_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v344_data + (v316_data * v342_data));
              double v347_data = s0_w0[103];
              tensorforge::intel_esimd::simd<double, 16> v349_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v349_data + (v316_data * v347_data));
              double v352_data = s0_w0[119];
              tensorforge::intel_esimd::simd<double, 16> v354_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v354_data + (v316_data * v352_data));
              tensorforge::intel_esimd::simd<double, 16> v356_data(r0.template select<16, 1>(128));
              double v357_data = s0_w0[8];
              tensorforge::intel_esimd::simd<double, 16> v359_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v359_data + (v356_data * v357_data));
              double v362_data = s0_w0[24];
              tensorforge::intel_esimd::simd<double, 16> v364_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v364_data + (v356_data * v362_data));
              double v367_data = s0_w0[40];
              tensorforge::intel_esimd::simd<double, 16> v369_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v369_data + (v356_data * v367_data));
              double v372_data = s0_w0[56];
              tensorforge::intel_esimd::simd<double, 16> v374_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v374_data + (v356_data * v372_data));
              double v377_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v379_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v379_data + (v356_data * v377_data));
              double v382_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 16> v384_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v384_data + (v356_data * v382_data));
              double v387_data = s0_w0[104];
              tensorforge::intel_esimd::simd<double, 16> v389_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v389_data + (v356_data * v387_data));
              double v392_data = s0_w0[120];
              tensorforge::intel_esimd::simd<double, 16> v394_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v394_data + (v356_data * v392_data));
              tensorforge::intel_esimd::simd<double, 16> v396_data(r0.template select<16, 1>(144));
              double v397_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 16> v399_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v399_data + (v396_data * v397_data));
              double v402_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v404_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v404_data + (v396_data * v402_data));
              double v407_data = s0_w0[41];
              tensorforge::intel_esimd::simd<double, 16> v409_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v409_data + (v396_data * v407_data));
              double v412_data = s0_w0[57];
              tensorforge::intel_esimd::simd<double, 16> v414_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v414_data + (v396_data * v412_data));
              double v417_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v419_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v419_data + (v396_data * v417_data));
              double v422_data = s0_w0[89];
              tensorforge::intel_esimd::simd<double, 16> v424_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v424_data + (v396_data * v422_data));
              double v427_data = s0_w0[105];
              tensorforge::intel_esimd::simd<double, 16> v429_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v429_data + (v396_data * v427_data));
              double v432_data = s0_w0[121];
              tensorforge::intel_esimd::simd<double, 16> v434_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v434_data + (v396_data * v432_data));
              tensorforge::intel_esimd::simd<double, 16> v436_data(r0.template select<16, 1>(160));
              double v437_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 16> v439_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v439_data + (v436_data * v437_data));
              double v442_data = s0_w0[26];
              tensorforge::intel_esimd::simd<double, 16> v444_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v444_data + (v436_data * v442_data));
              double v447_data = s0_w0[42];
              tensorforge::intel_esimd::simd<double, 16> v449_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v449_data + (v436_data * v447_data));
              double v452_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 16> v454_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v454_data + (v436_data * v452_data));
              double v457_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v459_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v459_data + (v436_data * v457_data));
              double v462_data = s0_w0[90];
              tensorforge::intel_esimd::simd<double, 16> v464_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v464_data + (v436_data * v462_data));
              double v467_data = s0_w0[106];
              tensorforge::intel_esimd::simd<double, 16> v469_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v469_data + (v436_data * v467_data));
              double v472_data = s0_w0[122];
              tensorforge::intel_esimd::simd<double, 16> v474_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v474_data + (v436_data * v472_data));
              tensorforge::intel_esimd::simd<double, 16> v476_data(r0.template select<16, 1>(176));
              double v477_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 16> v479_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v479_data + (v476_data * v477_data));
              double v482_data = s0_w0[27];
              tensorforge::intel_esimd::simd<double, 16> v484_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v484_data + (v476_data * v482_data));
              double v487_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v489_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v489_data + (v476_data * v487_data));
              double v492_data = s0_w0[59];
              tensorforge::intel_esimd::simd<double, 16> v494_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v494_data + (v476_data * v492_data));
              double v497_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v499_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v499_data + (v476_data * v497_data));
              double v502_data = s0_w0[91];
              tensorforge::intel_esimd::simd<double, 16> v504_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v504_data + (v476_data * v502_data));
              double v507_data = s0_w0[107];
              tensorforge::intel_esimd::simd<double, 16> v509_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v509_data + (v476_data * v507_data));
              double v512_data = s0_w0[123];
              tensorforge::intel_esimd::simd<double, 16> v514_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v514_data + (v476_data * v512_data));
              tensorforge::intel_esimd::simd<double, 16> v516_data(r0.template select<16, 1>(192));
              double v517_data = s0_w0[12];
              tensorforge::intel_esimd::simd<double, 16> v519_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v519_data + (v516_data * v517_data));
              double v522_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 16> v524_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v524_data + (v516_data * v522_data));
              double v527_data = s0_w0[44];
              tensorforge::intel_esimd::simd<double, 16> v529_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v529_data + (v516_data * v527_data));
              double v532_data = s0_w0[60];
              tensorforge::intel_esimd::simd<double, 16> v534_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v534_data + (v516_data * v532_data));
              double v537_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v539_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v539_data + (v516_data * v537_data));
              double v542_data = s0_w0[92];
              tensorforge::intel_esimd::simd<double, 16> v544_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v544_data + (v516_data * v542_data));
              double v547_data = s0_w0[108];
              tensorforge::intel_esimd::simd<double, 16> v549_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v549_data + (v516_data * v547_data));
              double v552_data = s0_w0[124];
              tensorforge::intel_esimd::simd<double, 16> v554_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v554_data + (v516_data * v552_data));
              tensorforge::intel_esimd::simd<double, 16> v556_data(r0.template select<16, 1>(208));
              double v557_data = s0_w0[13];
              tensorforge::intel_esimd::simd<double, 16> v559_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v559_data + (v556_data * v557_data));
              double v562_data = s0_w0[29];
              tensorforge::intel_esimd::simd<double, 16> v564_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v564_data + (v556_data * v562_data));
              double v567_data = s0_w0[45];
              tensorforge::intel_esimd::simd<double, 16> v569_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v569_data + (v556_data * v567_data));
              double v572_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v574_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v574_data + (v556_data * v572_data));
              double v577_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v579_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v579_data + (v556_data * v577_data));
              double v582_data = s0_w0[93];
              tensorforge::intel_esimd::simd<double, 16> v584_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v584_data + (v556_data * v582_data));
              double v587_data = s0_w0[109];
              tensorforge::intel_esimd::simd<double, 16> v589_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v589_data + (v556_data * v587_data));
              double v592_data = s0_w0[125];
              tensorforge::intel_esimd::simd<double, 16> v594_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v594_data + (v556_data * v592_data));
              tensorforge::intel_esimd::simd<double, 16> v596_data(r0.template select<16, 1>(224));
              double v597_data = s0_w0[14];
              tensorforge::intel_esimd::simd<double, 16> v599_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v599_data + (v596_data * v597_data));
              double v602_data = s0_w0[30];
              tensorforge::intel_esimd::simd<double, 16> v604_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v604_data + (v596_data * v602_data));
              double v607_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v609_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v609_data + (v596_data * v607_data));
              double v612_data = s0_w0[62];
              tensorforge::intel_esimd::simd<double, 16> v614_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v614_data + (v596_data * v612_data));
              double v617_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v619_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v619_data + (v596_data * v617_data));
              double v622_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v624_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v624_data + (v596_data * v622_data));
              double v627_data = s0_w0[110];
              tensorforge::intel_esimd::simd<double, 16> v629_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v629_data + (v596_data * v627_data));
              double v632_data = s0_w0[126];
              tensorforge::intel_esimd::simd<double, 16> v634_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v634_data + (v596_data * v632_data));
              tensorforge::intel_esimd::simd<double, 16> v636_data(r0.template select<16, 1>(240));
              double v637_data = s0_w0[15];
              tensorforge::intel_esimd::simd<double, 16> v639_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v639_data + (v636_data * v637_data));
              double v642_data = s0_w0[31];
              tensorforge::intel_esimd::simd<double, 16> v644_data(ir1.template select<16, 1>(16));
              ir1.template select<16, 1>(16) = (v644_data + (v636_data * v642_data));
              double v647_data = s0_w0[47];
              tensorforge::intel_esimd::simd<double, 16> v649_data(ir1.template select<16, 1>(32));
              ir1.template select<16, 1>(32) = (v649_data + (v636_data * v647_data));
              double v652_data = s0_w0[63];
              tensorforge::intel_esimd::simd<double, 16> v654_data(ir1.template select<16, 1>(48));
              ir1.template select<16, 1>(48) = (v654_data + (v636_data * v652_data));
              double v657_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v659_data(ir1.template select<16, 1>(64));
              ir1.template select<16, 1>(64) = (v659_data + (v636_data * v657_data));
              double v662_data = s0_w0[95];
              tensorforge::intel_esimd::simd<double, 16> v664_data(ir1.template select<16, 1>(80));
              ir1.template select<16, 1>(80) = (v664_data + (v636_data * v662_data));
              double v667_data = s0_w0[111];
              tensorforge::intel_esimd::simd<double, 16> v669_data(ir1.template select<16, 1>(96));
              ir1.template select<16, 1>(96) = (v669_data + (v636_data * v667_data));
              double v672_data = s0_w0[127];
              tensorforge::intel_esimd::simd<double, 16> v674_data(ir1.template select<16, 1>(112));
              ir1.template select<16, 1>(112) = (v674_data + (v636_data * v672_data));
              #pragma unroll
              for (int32_t v676_n0 = 0; v676_n0 < 1; ++v676_n0) {
                int32_t v678_a = v676_n0 * 16;
                #pragma unroll
                for (int32_t v677_n1 = 0; v677_n1 < 8; ++v677_n1) {
                  int32_t v680_a = v678_a + (v677_n1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v681_data(ir1.template select<16, 1>(v680_a));
                  r1.template select<16, 1>(v680_a) = v681_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v682_i0 = 0; v682_i0 < 1; ++v682_i0) {
                int32_t v684_a = v682_i0 * 16;
                #pragma unroll
                for (int32_t v683_i1 = 0; v683_i1 < 8; ++v683_i1) {
                  int32_t v686_a = v684_a + (v683_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v687_data(r1.template select<16, 1>(v686_a));
                  v687_data.copy_to(glb_m0 + (v686_a));
                }
              }
            }
            tensorforge::prefetchRunsL2<992, 992>(&pf_glb_m1[0], &pf_glb_m1[248]);
            tensorforge::prefetchRunsL2<992, 992>(&pf_glb_m1[496], &pf_glb_m1[744]);
            tensorforge::prefetchRunsL2<128, 512>(&pf_glb_m1[992], &pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


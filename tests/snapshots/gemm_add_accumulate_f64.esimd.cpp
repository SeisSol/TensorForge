// === base name ===
kernel_efafb907277b83dd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_efafb907277b83dd = {{1, 16, 1}, 16, 12, 1, 16, 18432, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_efafb907277b83dd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_efafb907277b83dd(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_efafb907277b83dd(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_efafb907277b83dd(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_efafb907277b83dd(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_efafb907277b83dd(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_efafb907277b83dd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(double)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 18432 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×16(12×16) {0..12}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":18432,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (144 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<double> tempShrMem = localShrMem0 + (128);
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v11_batchId0 * 96 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v11_batchId0 * 192 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v11_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                tensorforge::intel_esimd::simd<double, 12> v28_data;
                v28_data.copy_from(glb_m1 + ((v23_i1 * 12)));
                r0.template select<12, 1>((v23_i1 * 16)) = v28_data;
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
              tensorforge::intel_esimd::simd<double, 128> r1(0.0);
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
                tensorforge::intel_esimd::simd<double, 12> v41_data;
                v41_data.copy_from(glb_m0 + ((v36_i1 * 12)));
                r1.template select<12, 1>((v36_i1 * 16)) = v41_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<double, 128> r2(0.0);
              // ir2 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 128> ir2(0.0);
              tensorforge::intel_esimd::simd<double, 16> v46_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 128> s0_w0 = tensorforge::slmLoad<double, 128>(s0 + 0);
              double v47_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 16> v49_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v49_data + (v46_data * v47_data));
              double v52_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v54_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v54_data + (v46_data * v52_data));
              double v57_data = s0_w0[32];
              tensorforge::intel_esimd::simd<double, 16> v59_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v59_data + (v46_data * v57_data));
              double v62_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 16> v64_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v64_data + (v46_data * v62_data));
              double v67_data = s0_w0[64];
              tensorforge::intel_esimd::simd<double, 16> v69_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v69_data + (v46_data * v67_data));
              double v72_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 16> v74_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v74_data + (v46_data * v72_data));
              double v77_data = s0_w0[96];
              tensorforge::intel_esimd::simd<double, 16> v79_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v79_data + (v46_data * v77_data));
              double v82_data = s0_w0[112];
              tensorforge::intel_esimd::simd<double, 16> v84_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v84_data + (v46_data * v82_data));
              tensorforge::intel_esimd::simd<double, 16> v86_data(r0.template select<16, 1>(16));
              double v87_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 16> v89_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v89_data + (v86_data * v87_data));
              double v92_data = s0_w0[17];
              tensorforge::intel_esimd::simd<double, 16> v94_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v94_data + (v86_data * v92_data));
              double v97_data = s0_w0[33];
              tensorforge::intel_esimd::simd<double, 16> v99_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v99_data + (v86_data * v97_data));
              double v102_data = s0_w0[49];
              tensorforge::intel_esimd::simd<double, 16> v104_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v104_data + (v86_data * v102_data));
              double v107_data = s0_w0[65];
              tensorforge::intel_esimd::simd<double, 16> v109_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v109_data + (v86_data * v107_data));
              double v112_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 16> v114_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v114_data + (v86_data * v112_data));
              double v117_data = s0_w0[97];
              tensorforge::intel_esimd::simd<double, 16> v119_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v119_data + (v86_data * v117_data));
              double v122_data = s0_w0[113];
              tensorforge::intel_esimd::simd<double, 16> v124_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v124_data + (v86_data * v122_data));
              tensorforge::intel_esimd::simd<double, 16> v126_data(r0.template select<16, 1>(32));
              double v127_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 16> v129_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v129_data + (v126_data * v127_data));
              double v132_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 16> v134_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v134_data + (v126_data * v132_data));
              double v137_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v139_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v139_data + (v126_data * v137_data));
              double v142_data = s0_w0[50];
              tensorforge::intel_esimd::simd<double, 16> v144_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v144_data + (v126_data * v142_data));
              double v147_data = s0_w0[66];
              tensorforge::intel_esimd::simd<double, 16> v149_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v149_data + (v126_data * v147_data));
              double v152_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 16> v154_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v154_data + (v126_data * v152_data));
              double v157_data = s0_w0[98];
              tensorforge::intel_esimd::simd<double, 16> v159_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v159_data + (v126_data * v157_data));
              double v162_data = s0_w0[114];
              tensorforge::intel_esimd::simd<double, 16> v164_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v164_data + (v126_data * v162_data));
              tensorforge::intel_esimd::simd<double, 16> v166_data(r0.template select<16, 1>(48));
              double v167_data = s0_w0[3];
              tensorforge::intel_esimd::simd<double, 16> v169_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v169_data + (v166_data * v167_data));
              double v172_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 16> v174_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v174_data + (v166_data * v172_data));
              double v177_data = s0_w0[35];
              tensorforge::intel_esimd::simd<double, 16> v179_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v179_data + (v166_data * v177_data));
              double v182_data = s0_w0[51];
              tensorforge::intel_esimd::simd<double, 16> v184_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v184_data + (v166_data * v182_data));
              double v187_data = s0_w0[67];
              tensorforge::intel_esimd::simd<double, 16> v189_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v189_data + (v166_data * v187_data));
              double v192_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 16> v194_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v194_data + (v166_data * v192_data));
              double v197_data = s0_w0[99];
              tensorforge::intel_esimd::simd<double, 16> v199_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v199_data + (v166_data * v197_data));
              double v202_data = s0_w0[115];
              tensorforge::intel_esimd::simd<double, 16> v204_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v204_data + (v166_data * v202_data));
              tensorforge::intel_esimd::simd<double, 16> v206_data(r0.template select<16, 1>(64));
              double v207_data = s0_w0[4];
              tensorforge::intel_esimd::simd<double, 16> v209_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v209_data + (v206_data * v207_data));
              double v212_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 16> v214_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v214_data + (v206_data * v212_data));
              double v217_data = s0_w0[36];
              tensorforge::intel_esimd::simd<double, 16> v219_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v219_data + (v206_data * v217_data));
              double v222_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v224_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v224_data + (v206_data * v222_data));
              double v227_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 16> v229_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v229_data + (v206_data * v227_data));
              double v232_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 16> v234_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v234_data + (v206_data * v232_data));
              double v237_data = s0_w0[100];
              tensorforge::intel_esimd::simd<double, 16> v239_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v239_data + (v206_data * v237_data));
              double v242_data = s0_w0[116];
              tensorforge::intel_esimd::simd<double, 16> v244_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v244_data + (v206_data * v242_data));
              tensorforge::intel_esimd::simd<double, 16> v246_data(r0.template select<16, 1>(80));
              double v247_data = s0_w0[5];
              tensorforge::intel_esimd::simd<double, 16> v249_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v249_data + (v246_data * v247_data));
              double v252_data = s0_w0[21];
              tensorforge::intel_esimd::simd<double, 16> v254_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v254_data + (v246_data * v252_data));
              double v257_data = s0_w0[37];
              tensorforge::intel_esimd::simd<double, 16> v259_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v259_data + (v246_data * v257_data));
              double v262_data = s0_w0[53];
              tensorforge::intel_esimd::simd<double, 16> v264_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v264_data + (v246_data * v262_data));
              double v267_data = s0_w0[69];
              tensorforge::intel_esimd::simd<double, 16> v269_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v269_data + (v246_data * v267_data));
              double v272_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 16> v274_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v274_data + (v246_data * v272_data));
              double v277_data = s0_w0[101];
              tensorforge::intel_esimd::simd<double, 16> v279_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v279_data + (v246_data * v277_data));
              double v282_data = s0_w0[117];
              tensorforge::intel_esimd::simd<double, 16> v284_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v284_data + (v246_data * v282_data));
              tensorforge::intel_esimd::simd<double, 16> v286_data(r0.template select<16, 1>(96));
              double v287_data = s0_w0[6];
              tensorforge::intel_esimd::simd<double, 16> v289_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v289_data + (v286_data * v287_data));
              double v292_data = s0_w0[22];
              tensorforge::intel_esimd::simd<double, 16> v294_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v294_data + (v286_data * v292_data));
              double v297_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 16> v299_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v299_data + (v286_data * v297_data));
              double v302_data = s0_w0[54];
              tensorforge::intel_esimd::simd<double, 16> v304_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v304_data + (v286_data * v302_data));
              double v307_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v309_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v309_data + (v286_data * v307_data));
              double v312_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 16> v314_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v314_data + (v286_data * v312_data));
              double v317_data = s0_w0[102];
              tensorforge::intel_esimd::simd<double, 16> v319_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v319_data + (v286_data * v317_data));
              double v322_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v324_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v324_data + (v286_data * v322_data));
              tensorforge::intel_esimd::simd<double, 16> v326_data(r0.template select<16, 1>(112));
              double v327_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v329_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v329_data + (v326_data * v327_data));
              double v332_data = s0_w0[23];
              tensorforge::intel_esimd::simd<double, 16> v334_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v334_data + (v326_data * v332_data));
              double v337_data = s0_w0[39];
              tensorforge::intel_esimd::simd<double, 16> v339_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v339_data + (v326_data * v337_data));
              double v342_data = s0_w0[55];
              tensorforge::intel_esimd::simd<double, 16> v344_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v344_data + (v326_data * v342_data));
              double v347_data = s0_w0[71];
              tensorforge::intel_esimd::simd<double, 16> v349_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v349_data + (v326_data * v347_data));
              double v352_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 16> v354_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v354_data + (v326_data * v352_data));
              double v357_data = s0_w0[103];
              tensorforge::intel_esimd::simd<double, 16> v359_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v359_data + (v326_data * v357_data));
              double v362_data = s0_w0[119];
              tensorforge::intel_esimd::simd<double, 16> v364_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v364_data + (v326_data * v362_data));
              tensorforge::intel_esimd::simd<double, 16> v366_data(r0.template select<16, 1>(128));
              double v367_data = s0_w0[8];
              tensorforge::intel_esimd::simd<double, 16> v369_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v369_data + (v366_data * v367_data));
              double v372_data = s0_w0[24];
              tensorforge::intel_esimd::simd<double, 16> v374_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v374_data + (v366_data * v372_data));
              double v377_data = s0_w0[40];
              tensorforge::intel_esimd::simd<double, 16> v379_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v379_data + (v366_data * v377_data));
              double v382_data = s0_w0[56];
              tensorforge::intel_esimd::simd<double, 16> v384_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v384_data + (v366_data * v382_data));
              double v387_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v389_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v389_data + (v366_data * v387_data));
              double v392_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 16> v394_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v394_data + (v366_data * v392_data));
              double v397_data = s0_w0[104];
              tensorforge::intel_esimd::simd<double, 16> v399_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v399_data + (v366_data * v397_data));
              double v402_data = s0_w0[120];
              tensorforge::intel_esimd::simd<double, 16> v404_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v404_data + (v366_data * v402_data));
              tensorforge::intel_esimd::simd<double, 16> v406_data(r0.template select<16, 1>(144));
              double v407_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 16> v409_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v409_data + (v406_data * v407_data));
              double v412_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v414_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v414_data + (v406_data * v412_data));
              double v417_data = s0_w0[41];
              tensorforge::intel_esimd::simd<double, 16> v419_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v419_data + (v406_data * v417_data));
              double v422_data = s0_w0[57];
              tensorforge::intel_esimd::simd<double, 16> v424_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v424_data + (v406_data * v422_data));
              double v427_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v429_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v429_data + (v406_data * v427_data));
              double v432_data = s0_w0[89];
              tensorforge::intel_esimd::simd<double, 16> v434_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v434_data + (v406_data * v432_data));
              double v437_data = s0_w0[105];
              tensorforge::intel_esimd::simd<double, 16> v439_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v439_data + (v406_data * v437_data));
              double v442_data = s0_w0[121];
              tensorforge::intel_esimd::simd<double, 16> v444_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v444_data + (v406_data * v442_data));
              tensorforge::intel_esimd::simd<double, 16> v446_data(r0.template select<16, 1>(160));
              double v447_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 16> v449_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v449_data + (v446_data * v447_data));
              double v452_data = s0_w0[26];
              tensorforge::intel_esimd::simd<double, 16> v454_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v454_data + (v446_data * v452_data));
              double v457_data = s0_w0[42];
              tensorforge::intel_esimd::simd<double, 16> v459_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v459_data + (v446_data * v457_data));
              double v462_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 16> v464_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v464_data + (v446_data * v462_data));
              double v467_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v469_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v469_data + (v446_data * v467_data));
              double v472_data = s0_w0[90];
              tensorforge::intel_esimd::simd<double, 16> v474_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v474_data + (v446_data * v472_data));
              double v477_data = s0_w0[106];
              tensorforge::intel_esimd::simd<double, 16> v479_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v479_data + (v446_data * v477_data));
              double v482_data = s0_w0[122];
              tensorforge::intel_esimd::simd<double, 16> v484_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v484_data + (v446_data * v482_data));
              tensorforge::intel_esimd::simd<double, 16> v486_data(r0.template select<16, 1>(176));
              double v487_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 16> v489_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v489_data + (v486_data * v487_data));
              double v492_data = s0_w0[27];
              tensorforge::intel_esimd::simd<double, 16> v494_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v494_data + (v486_data * v492_data));
              double v497_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v499_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v499_data + (v486_data * v497_data));
              double v502_data = s0_w0[59];
              tensorforge::intel_esimd::simd<double, 16> v504_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v504_data + (v486_data * v502_data));
              double v507_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v509_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v509_data + (v486_data * v507_data));
              double v512_data = s0_w0[91];
              tensorforge::intel_esimd::simd<double, 16> v514_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v514_data + (v486_data * v512_data));
              double v517_data = s0_w0[107];
              tensorforge::intel_esimd::simd<double, 16> v519_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v519_data + (v486_data * v517_data));
              double v522_data = s0_w0[123];
              tensorforge::intel_esimd::simd<double, 16> v524_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v524_data + (v486_data * v522_data));
              tensorforge::intel_esimd::simd<double, 16> v526_data(r0.template select<16, 1>(192));
              double v527_data = s0_w0[12];
              tensorforge::intel_esimd::simd<double, 16> v529_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v529_data + (v526_data * v527_data));
              double v532_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 16> v534_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v534_data + (v526_data * v532_data));
              double v537_data = s0_w0[44];
              tensorforge::intel_esimd::simd<double, 16> v539_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v539_data + (v526_data * v537_data));
              double v542_data = s0_w0[60];
              tensorforge::intel_esimd::simd<double, 16> v544_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v544_data + (v526_data * v542_data));
              double v547_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v549_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v549_data + (v526_data * v547_data));
              double v552_data = s0_w0[92];
              tensorforge::intel_esimd::simd<double, 16> v554_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v554_data + (v526_data * v552_data));
              double v557_data = s0_w0[108];
              tensorforge::intel_esimd::simd<double, 16> v559_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v559_data + (v526_data * v557_data));
              double v562_data = s0_w0[124];
              tensorforge::intel_esimd::simd<double, 16> v564_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v564_data + (v526_data * v562_data));
              tensorforge::intel_esimd::simd<double, 16> v566_data(r0.template select<16, 1>(208));
              double v567_data = s0_w0[13];
              tensorforge::intel_esimd::simd<double, 16> v569_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v569_data + (v566_data * v567_data));
              double v572_data = s0_w0[29];
              tensorforge::intel_esimd::simd<double, 16> v574_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v574_data + (v566_data * v572_data));
              double v577_data = s0_w0[45];
              tensorforge::intel_esimd::simd<double, 16> v579_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v579_data + (v566_data * v577_data));
              double v582_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v584_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v584_data + (v566_data * v582_data));
              double v587_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v589_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v589_data + (v566_data * v587_data));
              double v592_data = s0_w0[93];
              tensorforge::intel_esimd::simd<double, 16> v594_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v594_data + (v566_data * v592_data));
              double v597_data = s0_w0[109];
              tensorforge::intel_esimd::simd<double, 16> v599_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v599_data + (v566_data * v597_data));
              double v602_data = s0_w0[125];
              tensorforge::intel_esimd::simd<double, 16> v604_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v604_data + (v566_data * v602_data));
              tensorforge::intel_esimd::simd<double, 16> v606_data(r0.template select<16, 1>(224));
              double v607_data = s0_w0[14];
              tensorforge::intel_esimd::simd<double, 16> v609_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v609_data + (v606_data * v607_data));
              double v612_data = s0_w0[30];
              tensorforge::intel_esimd::simd<double, 16> v614_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v614_data + (v606_data * v612_data));
              double v617_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v619_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v619_data + (v606_data * v617_data));
              double v622_data = s0_w0[62];
              tensorforge::intel_esimd::simd<double, 16> v624_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v624_data + (v606_data * v622_data));
              double v627_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v629_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v629_data + (v606_data * v627_data));
              double v632_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v634_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v634_data + (v606_data * v632_data));
              double v637_data = s0_w0[110];
              tensorforge::intel_esimd::simd<double, 16> v639_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v639_data + (v606_data * v637_data));
              double v642_data = s0_w0[126];
              tensorforge::intel_esimd::simd<double, 16> v644_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v644_data + (v606_data * v642_data));
              tensorforge::intel_esimd::simd<double, 16> v646_data(r0.template select<16, 1>(240));
              double v647_data = s0_w0[15];
              tensorforge::intel_esimd::simd<double, 16> v649_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v649_data + (v646_data * v647_data));
              double v652_data = s0_w0[31];
              tensorforge::intel_esimd::simd<double, 16> v654_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v654_data + (v646_data * v652_data));
              double v657_data = s0_w0[47];
              tensorforge::intel_esimd::simd<double, 16> v659_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v659_data + (v646_data * v657_data));
              double v662_data = s0_w0[63];
              tensorforge::intel_esimd::simd<double, 16> v664_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v664_data + (v646_data * v662_data));
              double v667_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v669_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v669_data + (v646_data * v667_data));
              double v672_data = s0_w0[95];
              tensorforge::intel_esimd::simd<double, 16> v674_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v674_data + (v646_data * v672_data));
              double v677_data = s0_w0[111];
              tensorforge::intel_esimd::simd<double, 16> v679_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v679_data + (v646_data * v677_data));
              double v682_data = s0_w0[127];
              tensorforge::intel_esimd::simd<double, 16> v684_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v684_data + (v646_data * v682_data));
              // r2 = ir2 + r1
              #pragma unroll
              for (int32_t v686_n1 = 0; v686_n1 < 8; ++v686_n1) {
                int32_t v687_a = v686_n1 * 16;
                tensorforge::intel_esimd::simd<double, 12> v689_data(ir2.template select<12, 1>(v687_a));
                tensorforge::intel_esimd::simd<double, 12> v690_data(r1.template select<12, 1>(v687_a));
                r2.template select<12, 1>(v687_a) = (v690_data + v689_data);
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v692_i1 = 0; v692_i1 < 8; ++v692_i1) {
                tensorforge::intel_esimd::simd<double, 12> v695_data(r2.template select<12, 1>((v692_i1 * 16)));
                v695_data.copy_to(glb_m0 + ((v692_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


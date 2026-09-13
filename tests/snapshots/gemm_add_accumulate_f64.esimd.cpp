// === base name ===
kernel_ffd4eb2f3f8f5d91

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ffd4eb2f3f8f5d91 = {{1, 16, 1}, 16, 12, 1, 16, 18432, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ffd4eb2f3f8f5d91(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ffd4eb2f3f8f5d91(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ffd4eb2f3f8f5d91(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_ffd4eb2f3f8f5d91(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ffd4eb2f3f8f5d91(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ffd4eb2f3f8f5d91(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ffd4eb2f3f8f5d91(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const double *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 192 + 0 + m1_extraOffset];
            const double *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 128 + 0 + m2_extraOffset];
            double *const __restrict__ pf_glb_m0 = &m0[v8_batchId1 * 96 + 0 + m0_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v5_batchId0 * 96 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v5_batchId0 * 192 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
                tensorforge::intel_esimd::simd<double, 12> v25_data;
                v25_data.copy_from(glb_m1 + ((v20_i1 * 12)));
                r0.template select<12, 1>((v20_i1 * 16)) = v25_data;
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
              tensorforge::intel_esimd::simd<double, 128> r1(0.0);
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
                tensorforge::intel_esimd::simd<double, 12> v38_data;
                v38_data.copy_from(glb_m0 + ((v33_i1 * 12)));
                r1.template select<12, 1>((v33_i1 * 16)) = v38_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<double, 128> r2(0.0);
              // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 128> ir2(0.0);
              tensorforge::intel_esimd::simd<double, 16> v43_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 128> s0_w0 = tensorforge::slmLoad<double, 128>(s0 + 0);
              double v44_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 16> v46_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v46_data + (v43_data * v44_data));
              double v49_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v51_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v51_data + (v43_data * v49_data));
              double v54_data = s0_w0[32];
              tensorforge::intel_esimd::simd<double, 16> v56_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v56_data + (v43_data * v54_data));
              double v59_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 16> v61_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v61_data + (v43_data * v59_data));
              double v64_data = s0_w0[64];
              tensorforge::intel_esimd::simd<double, 16> v66_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v66_data + (v43_data * v64_data));
              double v69_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 16> v71_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v71_data + (v43_data * v69_data));
              double v74_data = s0_w0[96];
              tensorforge::intel_esimd::simd<double, 16> v76_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v76_data + (v43_data * v74_data));
              double v79_data = s0_w0[112];
              tensorforge::intel_esimd::simd<double, 16> v81_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v81_data + (v43_data * v79_data));
              tensorforge::intel_esimd::simd<double, 16> v83_data(r0.template select<16, 1>(16));
              double v84_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 16> v86_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v86_data + (v83_data * v84_data));
              double v89_data = s0_w0[17];
              tensorforge::intel_esimd::simd<double, 16> v91_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v91_data + (v83_data * v89_data));
              double v94_data = s0_w0[33];
              tensorforge::intel_esimd::simd<double, 16> v96_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v96_data + (v83_data * v94_data));
              double v99_data = s0_w0[49];
              tensorforge::intel_esimd::simd<double, 16> v101_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v101_data + (v83_data * v99_data));
              double v104_data = s0_w0[65];
              tensorforge::intel_esimd::simd<double, 16> v106_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v106_data + (v83_data * v104_data));
              double v109_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 16> v111_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v111_data + (v83_data * v109_data));
              double v114_data = s0_w0[97];
              tensorforge::intel_esimd::simd<double, 16> v116_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v116_data + (v83_data * v114_data));
              double v119_data = s0_w0[113];
              tensorforge::intel_esimd::simd<double, 16> v121_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v121_data + (v83_data * v119_data));
              tensorforge::intel_esimd::simd<double, 16> v123_data(r0.template select<16, 1>(32));
              double v124_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 16> v126_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v126_data + (v123_data * v124_data));
              double v129_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 16> v131_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v131_data + (v123_data * v129_data));
              double v134_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v136_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v136_data + (v123_data * v134_data));
              double v139_data = s0_w0[50];
              tensorforge::intel_esimd::simd<double, 16> v141_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v141_data + (v123_data * v139_data));
              double v144_data = s0_w0[66];
              tensorforge::intel_esimd::simd<double, 16> v146_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v146_data + (v123_data * v144_data));
              double v149_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 16> v151_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v151_data + (v123_data * v149_data));
              double v154_data = s0_w0[98];
              tensorforge::intel_esimd::simd<double, 16> v156_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v156_data + (v123_data * v154_data));
              double v159_data = s0_w0[114];
              tensorforge::intel_esimd::simd<double, 16> v161_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v161_data + (v123_data * v159_data));
              tensorforge::intel_esimd::simd<double, 16> v163_data(r0.template select<16, 1>(48));
              double v164_data = s0_w0[3];
              tensorforge::intel_esimd::simd<double, 16> v166_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v166_data + (v163_data * v164_data));
              double v169_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 16> v171_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v171_data + (v163_data * v169_data));
              double v174_data = s0_w0[35];
              tensorforge::intel_esimd::simd<double, 16> v176_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v176_data + (v163_data * v174_data));
              double v179_data = s0_w0[51];
              tensorforge::intel_esimd::simd<double, 16> v181_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v181_data + (v163_data * v179_data));
              double v184_data = s0_w0[67];
              tensorforge::intel_esimd::simd<double, 16> v186_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v186_data + (v163_data * v184_data));
              double v189_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 16> v191_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v191_data + (v163_data * v189_data));
              double v194_data = s0_w0[99];
              tensorforge::intel_esimd::simd<double, 16> v196_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v196_data + (v163_data * v194_data));
              double v199_data = s0_w0[115];
              tensorforge::intel_esimd::simd<double, 16> v201_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v201_data + (v163_data * v199_data));
              tensorforge::intel_esimd::simd<double, 16> v203_data(r0.template select<16, 1>(64));
              double v204_data = s0_w0[4];
              tensorforge::intel_esimd::simd<double, 16> v206_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v206_data + (v203_data * v204_data));
              double v209_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 16> v211_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v211_data + (v203_data * v209_data));
              double v214_data = s0_w0[36];
              tensorforge::intel_esimd::simd<double, 16> v216_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v216_data + (v203_data * v214_data));
              double v219_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v221_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v221_data + (v203_data * v219_data));
              double v224_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 16> v226_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v226_data + (v203_data * v224_data));
              double v229_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 16> v231_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v231_data + (v203_data * v229_data));
              double v234_data = s0_w0[100];
              tensorforge::intel_esimd::simd<double, 16> v236_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v236_data + (v203_data * v234_data));
              double v239_data = s0_w0[116];
              tensorforge::intel_esimd::simd<double, 16> v241_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v241_data + (v203_data * v239_data));
              tensorforge::intel_esimd::simd<double, 16> v243_data(r0.template select<16, 1>(80));
              double v244_data = s0_w0[5];
              tensorforge::intel_esimd::simd<double, 16> v246_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v246_data + (v243_data * v244_data));
              double v249_data = s0_w0[21];
              tensorforge::intel_esimd::simd<double, 16> v251_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v251_data + (v243_data * v249_data));
              double v254_data = s0_w0[37];
              tensorforge::intel_esimd::simd<double, 16> v256_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v256_data + (v243_data * v254_data));
              double v259_data = s0_w0[53];
              tensorforge::intel_esimd::simd<double, 16> v261_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v261_data + (v243_data * v259_data));
              double v264_data = s0_w0[69];
              tensorforge::intel_esimd::simd<double, 16> v266_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v266_data + (v243_data * v264_data));
              double v269_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 16> v271_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v271_data + (v243_data * v269_data));
              double v274_data = s0_w0[101];
              tensorforge::intel_esimd::simd<double, 16> v276_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v276_data + (v243_data * v274_data));
              double v279_data = s0_w0[117];
              tensorforge::intel_esimd::simd<double, 16> v281_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v281_data + (v243_data * v279_data));
              tensorforge::intel_esimd::simd<double, 16> v283_data(r0.template select<16, 1>(96));
              double v284_data = s0_w0[6];
              tensorforge::intel_esimd::simd<double, 16> v286_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v286_data + (v283_data * v284_data));
              double v289_data = s0_w0[22];
              tensorforge::intel_esimd::simd<double, 16> v291_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v291_data + (v283_data * v289_data));
              double v294_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 16> v296_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v296_data + (v283_data * v294_data));
              double v299_data = s0_w0[54];
              tensorforge::intel_esimd::simd<double, 16> v301_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v301_data + (v283_data * v299_data));
              double v304_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v306_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v306_data + (v283_data * v304_data));
              double v309_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 16> v311_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v311_data + (v283_data * v309_data));
              double v314_data = s0_w0[102];
              tensorforge::intel_esimd::simd<double, 16> v316_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v316_data + (v283_data * v314_data));
              double v319_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v321_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v321_data + (v283_data * v319_data));
              tensorforge::intel_esimd::simd<double, 16> v323_data(r0.template select<16, 1>(112));
              double v324_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v326_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v326_data + (v323_data * v324_data));
              double v329_data = s0_w0[23];
              tensorforge::intel_esimd::simd<double, 16> v331_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v331_data + (v323_data * v329_data));
              double v334_data = s0_w0[39];
              tensorforge::intel_esimd::simd<double, 16> v336_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v336_data + (v323_data * v334_data));
              double v339_data = s0_w0[55];
              tensorforge::intel_esimd::simd<double, 16> v341_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v341_data + (v323_data * v339_data));
              double v344_data = s0_w0[71];
              tensorforge::intel_esimd::simd<double, 16> v346_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v346_data + (v323_data * v344_data));
              double v349_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 16> v351_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v351_data + (v323_data * v349_data));
              double v354_data = s0_w0[103];
              tensorforge::intel_esimd::simd<double, 16> v356_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v356_data + (v323_data * v354_data));
              double v359_data = s0_w0[119];
              tensorforge::intel_esimd::simd<double, 16> v361_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v361_data + (v323_data * v359_data));
              tensorforge::intel_esimd::simd<double, 16> v363_data(r0.template select<16, 1>(128));
              double v364_data = s0_w0[8];
              tensorforge::intel_esimd::simd<double, 16> v366_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v366_data + (v363_data * v364_data));
              double v369_data = s0_w0[24];
              tensorforge::intel_esimd::simd<double, 16> v371_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v371_data + (v363_data * v369_data));
              double v374_data = s0_w0[40];
              tensorforge::intel_esimd::simd<double, 16> v376_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v376_data + (v363_data * v374_data));
              double v379_data = s0_w0[56];
              tensorforge::intel_esimd::simd<double, 16> v381_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v381_data + (v363_data * v379_data));
              double v384_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v386_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v386_data + (v363_data * v384_data));
              double v389_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 16> v391_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v391_data + (v363_data * v389_data));
              double v394_data = s0_w0[104];
              tensorforge::intel_esimd::simd<double, 16> v396_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v396_data + (v363_data * v394_data));
              double v399_data = s0_w0[120];
              tensorforge::intel_esimd::simd<double, 16> v401_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v401_data + (v363_data * v399_data));
              tensorforge::intel_esimd::simd<double, 16> v403_data(r0.template select<16, 1>(144));
              double v404_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 16> v406_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v406_data + (v403_data * v404_data));
              double v409_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v411_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v411_data + (v403_data * v409_data));
              double v414_data = s0_w0[41];
              tensorforge::intel_esimd::simd<double, 16> v416_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v416_data + (v403_data * v414_data));
              double v419_data = s0_w0[57];
              tensorforge::intel_esimd::simd<double, 16> v421_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v421_data + (v403_data * v419_data));
              double v424_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v426_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v426_data + (v403_data * v424_data));
              double v429_data = s0_w0[89];
              tensorforge::intel_esimd::simd<double, 16> v431_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v431_data + (v403_data * v429_data));
              double v434_data = s0_w0[105];
              tensorforge::intel_esimd::simd<double, 16> v436_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v436_data + (v403_data * v434_data));
              double v439_data = s0_w0[121];
              tensorforge::intel_esimd::simd<double, 16> v441_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v441_data + (v403_data * v439_data));
              tensorforge::intel_esimd::simd<double, 16> v443_data(r0.template select<16, 1>(160));
              double v444_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 16> v446_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v446_data + (v443_data * v444_data));
              double v449_data = s0_w0[26];
              tensorforge::intel_esimd::simd<double, 16> v451_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v451_data + (v443_data * v449_data));
              double v454_data = s0_w0[42];
              tensorforge::intel_esimd::simd<double, 16> v456_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v456_data + (v443_data * v454_data));
              double v459_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 16> v461_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v461_data + (v443_data * v459_data));
              double v464_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v466_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v466_data + (v443_data * v464_data));
              double v469_data = s0_w0[90];
              tensorforge::intel_esimd::simd<double, 16> v471_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v471_data + (v443_data * v469_data));
              double v474_data = s0_w0[106];
              tensorforge::intel_esimd::simd<double, 16> v476_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v476_data + (v443_data * v474_data));
              double v479_data = s0_w0[122];
              tensorforge::intel_esimd::simd<double, 16> v481_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v481_data + (v443_data * v479_data));
              tensorforge::intel_esimd::simd<double, 16> v483_data(r0.template select<16, 1>(176));
              double v484_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 16> v486_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v486_data + (v483_data * v484_data));
              double v489_data = s0_w0[27];
              tensorforge::intel_esimd::simd<double, 16> v491_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v491_data + (v483_data * v489_data));
              double v494_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v496_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v496_data + (v483_data * v494_data));
              double v499_data = s0_w0[59];
              tensorforge::intel_esimd::simd<double, 16> v501_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v501_data + (v483_data * v499_data));
              double v504_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v506_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v506_data + (v483_data * v504_data));
              double v509_data = s0_w0[91];
              tensorforge::intel_esimd::simd<double, 16> v511_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v511_data + (v483_data * v509_data));
              double v514_data = s0_w0[107];
              tensorforge::intel_esimd::simd<double, 16> v516_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v516_data + (v483_data * v514_data));
              double v519_data = s0_w0[123];
              tensorforge::intel_esimd::simd<double, 16> v521_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v521_data + (v483_data * v519_data));
              tensorforge::intel_esimd::simd<double, 16> v523_data(r0.template select<16, 1>(192));
              double v524_data = s0_w0[12];
              tensorforge::intel_esimd::simd<double, 16> v526_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v526_data + (v523_data * v524_data));
              double v529_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 16> v531_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v531_data + (v523_data * v529_data));
              double v534_data = s0_w0[44];
              tensorforge::intel_esimd::simd<double, 16> v536_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v536_data + (v523_data * v534_data));
              double v539_data = s0_w0[60];
              tensorforge::intel_esimd::simd<double, 16> v541_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v541_data + (v523_data * v539_data));
              double v544_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v546_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v546_data + (v523_data * v544_data));
              double v549_data = s0_w0[92];
              tensorforge::intel_esimd::simd<double, 16> v551_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v551_data + (v523_data * v549_data));
              double v554_data = s0_w0[108];
              tensorforge::intel_esimd::simd<double, 16> v556_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v556_data + (v523_data * v554_data));
              double v559_data = s0_w0[124];
              tensorforge::intel_esimd::simd<double, 16> v561_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v561_data + (v523_data * v559_data));
              tensorforge::intel_esimd::simd<double, 16> v563_data(r0.template select<16, 1>(208));
              double v564_data = s0_w0[13];
              tensorforge::intel_esimd::simd<double, 16> v566_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v566_data + (v563_data * v564_data));
              double v569_data = s0_w0[29];
              tensorforge::intel_esimd::simd<double, 16> v571_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v571_data + (v563_data * v569_data));
              double v574_data = s0_w0[45];
              tensorforge::intel_esimd::simd<double, 16> v576_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v576_data + (v563_data * v574_data));
              double v579_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v581_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v581_data + (v563_data * v579_data));
              double v584_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v586_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v586_data + (v563_data * v584_data));
              double v589_data = s0_w0[93];
              tensorforge::intel_esimd::simd<double, 16> v591_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v591_data + (v563_data * v589_data));
              double v594_data = s0_w0[109];
              tensorforge::intel_esimd::simd<double, 16> v596_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v596_data + (v563_data * v594_data));
              double v599_data = s0_w0[125];
              tensorforge::intel_esimd::simd<double, 16> v601_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v601_data + (v563_data * v599_data));
              tensorforge::intel_esimd::simd<double, 16> v603_data(r0.template select<16, 1>(224));
              double v604_data = s0_w0[14];
              tensorforge::intel_esimd::simd<double, 16> v606_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v606_data + (v603_data * v604_data));
              double v609_data = s0_w0[30];
              tensorforge::intel_esimd::simd<double, 16> v611_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v611_data + (v603_data * v609_data));
              double v614_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v616_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v616_data + (v603_data * v614_data));
              double v619_data = s0_w0[62];
              tensorforge::intel_esimd::simd<double, 16> v621_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v621_data + (v603_data * v619_data));
              double v624_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v626_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v626_data + (v603_data * v624_data));
              double v629_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v631_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v631_data + (v603_data * v629_data));
              double v634_data = s0_w0[110];
              tensorforge::intel_esimd::simd<double, 16> v636_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v636_data + (v603_data * v634_data));
              double v639_data = s0_w0[126];
              tensorforge::intel_esimd::simd<double, 16> v641_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v641_data + (v603_data * v639_data));
              tensorforge::intel_esimd::simd<double, 16> v643_data(r0.template select<16, 1>(240));
              double v644_data = s0_w0[15];
              tensorforge::intel_esimd::simd<double, 16> v646_data(ir2.template select<16, 1>(0));
              ir2.template select<16, 1>(0) = (v646_data + (v643_data * v644_data));
              double v649_data = s0_w0[31];
              tensorforge::intel_esimd::simd<double, 16> v651_data(ir2.template select<16, 1>(16));
              ir2.template select<16, 1>(16) = (v651_data + (v643_data * v649_data));
              double v654_data = s0_w0[47];
              tensorforge::intel_esimd::simd<double, 16> v656_data(ir2.template select<16, 1>(32));
              ir2.template select<16, 1>(32) = (v656_data + (v643_data * v654_data));
              double v659_data = s0_w0[63];
              tensorforge::intel_esimd::simd<double, 16> v661_data(ir2.template select<16, 1>(48));
              ir2.template select<16, 1>(48) = (v661_data + (v643_data * v659_data));
              double v664_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v666_data(ir2.template select<16, 1>(64));
              ir2.template select<16, 1>(64) = (v666_data + (v643_data * v664_data));
              double v669_data = s0_w0[95];
              tensorforge::intel_esimd::simd<double, 16> v671_data(ir2.template select<16, 1>(80));
              ir2.template select<16, 1>(80) = (v671_data + (v643_data * v669_data));
              double v674_data = s0_w0[111];
              tensorforge::intel_esimd::simd<double, 16> v676_data(ir2.template select<16, 1>(96));
              ir2.template select<16, 1>(96) = (v676_data + (v643_data * v674_data));
              double v679_data = s0_w0[127];
              tensorforge::intel_esimd::simd<double, 16> v681_data(ir2.template select<16, 1>(112));
              ir2.template select<16, 1>(112) = (v681_data + (v643_data * v679_data));
              #pragma unroll
              for (int32_t v683_n1 = 0; v683_n1 < 8; ++v683_n1) {
                int32_t v684_a = v683_n1 * 16;
                tensorforge::intel_esimd::simd<double, 12> v686_data(ir2.template select<12, 1>(v684_a));
                tensorforge::intel_esimd::simd<double, 12> v687_data(r1.template select<12, 1>(v684_a));
                r2.template select<12, 1>(v684_a) = (v687_data + v686_data);
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v689_i1 = 0; v689_i1 < 8; ++v689_i1) {
                tensorforge::intel_esimd::simd<double, 12> v692_data(r2.template select<12, 1>((v689_i1 * 16)));
                v692_data.copy_to(glb_m0 + ((v689_i1 * 12)));
              }
            }
            tensorforge::prefetchRunsL2<768, 512, 384>(&pf_glb_m1[0], &pf_glb_m2[0], &pf_glb_m0[0]);
          }
        }
      }
    });
  });
}


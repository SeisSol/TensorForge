// === base name ===
kernel_6fc167805237fdd5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6fc167805237fdd5 = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6fc167805237fdd5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6fc167805237fdd5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6fc167805237fdd5(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4864 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_6fc167805237fdd5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6fc167805237fdd5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_6fc167805237fdd5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_6fc167805237fdd5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4864 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 19456 B shared, occupancy grid
        // operands:
        //   m0 6×12(6×12) {0..6}×{0..12} strided
        //   m1 12×12(12×12) {0..12}×{0..12} strided
        //   m2 6×12(6×12) {0..6}×{0..12} strided
        //   m3 12×12(12×12) {0..12}×{0..12} strided
        //   m4 2×12(2×12) {0..2}×{0..12} strided
        //   m5 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
        //   m3[i,j] = t0[i,j]
        //   t0[i,j]@{6..12}×{0..12} = m4[i,k] × m1[k,j]
        //   m5[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4864}],"shared_bytes":19456,"shared_elements":4864,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N2","bbox":[[0,0],[2,12]],"name":"m4","ordered":false,"parts":1,"shape":[2,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[2,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (304 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (288);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (144);
          for (size_t v12_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v12_batchId0 < numElements0; v12_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 72 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v12_batchId0 * 24 + 0 + m4_extraOffset];
              float *const __restrict__ glb_m5 = &m5[v12_batchId0 * 144 + 0 + m5_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
                tensorforge::intel_esimd::simd<float, 6> v32_data;
                v32_data.copy_from(glb_m0 + ((v27_i1 * 6)));
                r0.template select<6, 1>((v27_i1 * 16)) = v32_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v35_ld;
              v35_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v35_ld);
              tensorforge::intel_esimd::simd<float, 64> v36_ld;
              v36_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v36_ld);
              tensorforge::intel_esimd::simd<float, 16> v37_ld;
              v37_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v37_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
                tensorforge::intel_esimd::simd<float, 6> v44_data;
                v44_data.copy_from(glb_m2 + ((v39_i1 * 6)));
                r2.template select<6, 1>((v39_i1 * 16)) = v44_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v59_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v60_acc{};
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              float v65_bc = static_cast<float>(v64_data[0]);
              v60_acc += (v65_bc * v48_data);
              float v67_bc = static_cast<float>(v64_data[1]);
              v60_acc += (v67_bc * v49_data);
              float v69_bc = static_cast<float>(v64_data[2]);
              v60_acc += (v69_bc * v50_data);
              float v71_bc = static_cast<float>(v64_data[3]);
              v60_acc += (v71_bc * v51_data);
              float v73_bc = static_cast<float>(v64_data[4]);
              v60_acc += (v73_bc * v52_data);
              float v75_bc = static_cast<float>(v64_data[5]);
              v60_acc += (v75_bc * v53_data);
              float v77_bc = static_cast<float>(v64_data[6]);
              v60_acc += (v77_bc * v54_data);
              float v79_bc = static_cast<float>(v64_data[7]);
              v60_acc += (v79_bc * v55_data);
              float v81_bc = static_cast<float>(v64_data[8]);
              v60_acc += (v81_bc * v56_data);
              float v83_bc = static_cast<float>(v64_data[9]);
              v60_acc += (v83_bc * v57_data);
              float v85_bc = static_cast<float>(v64_data[10]);
              v60_acc += (v85_bc * v58_data);
              float v87_bc = static_cast<float>(v64_data[11]);
              v60_acc += (v87_bc * v59_data);
              r1.template select<16, 1>(0) = v60_acc;
              tensorforge::intel_esimd::simd<float, 16> v89_acc{};
              tensorforge::intel_esimd::simd<float, 16> v91_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              float v92_bc = static_cast<float>(v91_data[0]);
              v89_acc += (v92_bc * v48_data);
              float v94_bc = static_cast<float>(v91_data[1]);
              v89_acc += (v94_bc * v49_data);
              float v96_bc = static_cast<float>(v91_data[2]);
              v89_acc += (v96_bc * v50_data);
              float v98_bc = static_cast<float>(v91_data[3]);
              v89_acc += (v98_bc * v51_data);
              float v100_bc = static_cast<float>(v91_data[4]);
              v89_acc += (v100_bc * v52_data);
              float v102_bc = static_cast<float>(v91_data[5]);
              v89_acc += (v102_bc * v53_data);
              float v104_bc = static_cast<float>(v91_data[6]);
              v89_acc += (v104_bc * v54_data);
              float v106_bc = static_cast<float>(v91_data[7]);
              v89_acc += (v106_bc * v55_data);
              float v108_bc = static_cast<float>(v91_data[8]);
              v89_acc += (v108_bc * v56_data);
              float v110_bc = static_cast<float>(v91_data[9]);
              v89_acc += (v110_bc * v57_data);
              float v112_bc = static_cast<float>(v91_data[10]);
              v89_acc += (v112_bc * v58_data);
              float v114_bc = static_cast<float>(v91_data[11]);
              v89_acc += (v114_bc * v59_data);
              r1.template select<16, 1>(16) = v89_acc;
              tensorforge::intel_esimd::simd<float, 16> v116_acc{};
              tensorforge::intel_esimd::simd<float, 16> v118_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              float v119_bc = static_cast<float>(v118_data[0]);
              v116_acc += (v119_bc * v48_data);
              float v121_bc = static_cast<float>(v118_data[1]);
              v116_acc += (v121_bc * v49_data);
              float v123_bc = static_cast<float>(v118_data[2]);
              v116_acc += (v123_bc * v50_data);
              float v125_bc = static_cast<float>(v118_data[3]);
              v116_acc += (v125_bc * v51_data);
              float v127_bc = static_cast<float>(v118_data[4]);
              v116_acc += (v127_bc * v52_data);
              float v129_bc = static_cast<float>(v118_data[5]);
              v116_acc += (v129_bc * v53_data);
              float v131_bc = static_cast<float>(v118_data[6]);
              v116_acc += (v131_bc * v54_data);
              float v133_bc = static_cast<float>(v118_data[7]);
              v116_acc += (v133_bc * v55_data);
              float v135_bc = static_cast<float>(v118_data[8]);
              v116_acc += (v135_bc * v56_data);
              float v137_bc = static_cast<float>(v118_data[9]);
              v116_acc += (v137_bc * v57_data);
              float v139_bc = static_cast<float>(v118_data[10]);
              v116_acc += (v139_bc * v58_data);
              float v141_bc = static_cast<float>(v118_data[11]);
              v116_acc += (v141_bc * v59_data);
              r1.template select<16, 1>(32) = v116_acc;
              tensorforge::intel_esimd::simd<float, 16> v143_acc{};
              tensorforge::intel_esimd::simd<float, 16> v145_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              float v146_bc = static_cast<float>(v145_data[0]);
              v143_acc += (v146_bc * v48_data);
              float v148_bc = static_cast<float>(v145_data[1]);
              v143_acc += (v148_bc * v49_data);
              float v150_bc = static_cast<float>(v145_data[2]);
              v143_acc += (v150_bc * v50_data);
              float v152_bc = static_cast<float>(v145_data[3]);
              v143_acc += (v152_bc * v51_data);
              float v154_bc = static_cast<float>(v145_data[4]);
              v143_acc += (v154_bc * v52_data);
              float v156_bc = static_cast<float>(v145_data[5]);
              v143_acc += (v156_bc * v53_data);
              float v158_bc = static_cast<float>(v145_data[6]);
              v143_acc += (v158_bc * v54_data);
              float v160_bc = static_cast<float>(v145_data[7]);
              v143_acc += (v160_bc * v55_data);
              float v162_bc = static_cast<float>(v145_data[8]);
              v143_acc += (v162_bc * v56_data);
              float v164_bc = static_cast<float>(v145_data[9]);
              v143_acc += (v164_bc * v57_data);
              float v166_bc = static_cast<float>(v145_data[10]);
              v143_acc += (v166_bc * v58_data);
              float v168_bc = static_cast<float>(v145_data[11]);
              v143_acc += (v168_bc * v59_data);
              r1.template select<16, 1>(48) = v143_acc;
              tensorforge::intel_esimd::simd<float, 16> v170_acc{};
              tensorforge::intel_esimd::simd<float, 16> v172_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              float v173_bc = static_cast<float>(v172_data[0]);
              v170_acc += (v173_bc * v48_data);
              float v175_bc = static_cast<float>(v172_data[1]);
              v170_acc += (v175_bc * v49_data);
              float v177_bc = static_cast<float>(v172_data[2]);
              v170_acc += (v177_bc * v50_data);
              float v179_bc = static_cast<float>(v172_data[3]);
              v170_acc += (v179_bc * v51_data);
              float v181_bc = static_cast<float>(v172_data[4]);
              v170_acc += (v181_bc * v52_data);
              float v183_bc = static_cast<float>(v172_data[5]);
              v170_acc += (v183_bc * v53_data);
              float v185_bc = static_cast<float>(v172_data[6]);
              v170_acc += (v185_bc * v54_data);
              float v187_bc = static_cast<float>(v172_data[7]);
              v170_acc += (v187_bc * v55_data);
              float v189_bc = static_cast<float>(v172_data[8]);
              v170_acc += (v189_bc * v56_data);
              float v191_bc = static_cast<float>(v172_data[9]);
              v170_acc += (v191_bc * v57_data);
              float v193_bc = static_cast<float>(v172_data[10]);
              v170_acc += (v193_bc * v58_data);
              float v195_bc = static_cast<float>(v172_data[11]);
              v170_acc += (v195_bc * v59_data);
              r1.template select<16, 1>(64) = v170_acc;
              tensorforge::intel_esimd::simd<float, 16> v197_acc{};
              tensorforge::intel_esimd::simd<float, 16> v199_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              float v200_bc = static_cast<float>(v199_data[0]);
              v197_acc += (v200_bc * v48_data);
              float v202_bc = static_cast<float>(v199_data[1]);
              v197_acc += (v202_bc * v49_data);
              float v204_bc = static_cast<float>(v199_data[2]);
              v197_acc += (v204_bc * v50_data);
              float v206_bc = static_cast<float>(v199_data[3]);
              v197_acc += (v206_bc * v51_data);
              float v208_bc = static_cast<float>(v199_data[4]);
              v197_acc += (v208_bc * v52_data);
              float v210_bc = static_cast<float>(v199_data[5]);
              v197_acc += (v210_bc * v53_data);
              float v212_bc = static_cast<float>(v199_data[6]);
              v197_acc += (v212_bc * v54_data);
              float v214_bc = static_cast<float>(v199_data[7]);
              v197_acc += (v214_bc * v55_data);
              float v216_bc = static_cast<float>(v199_data[8]);
              v197_acc += (v216_bc * v56_data);
              float v218_bc = static_cast<float>(v199_data[9]);
              v197_acc += (v218_bc * v57_data);
              float v220_bc = static_cast<float>(v199_data[10]);
              v197_acc += (v220_bc * v58_data);
              float v222_bc = static_cast<float>(v199_data[11]);
              v197_acc += (v222_bc * v59_data);
              r1.template select<16, 1>(80) = v197_acc;
              tensorforge::intel_esimd::simd<float, 16> v224_acc{};
              tensorforge::intel_esimd::simd<float, 16> v226_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              float v227_bc = static_cast<float>(v226_data[0]);
              v224_acc += (v227_bc * v48_data);
              float v229_bc = static_cast<float>(v226_data[1]);
              v224_acc += (v229_bc * v49_data);
              float v231_bc = static_cast<float>(v226_data[2]);
              v224_acc += (v231_bc * v50_data);
              float v233_bc = static_cast<float>(v226_data[3]);
              v224_acc += (v233_bc * v51_data);
              float v235_bc = static_cast<float>(v226_data[4]);
              v224_acc += (v235_bc * v52_data);
              float v237_bc = static_cast<float>(v226_data[5]);
              v224_acc += (v237_bc * v53_data);
              float v239_bc = static_cast<float>(v226_data[6]);
              v224_acc += (v239_bc * v54_data);
              float v241_bc = static_cast<float>(v226_data[7]);
              v224_acc += (v241_bc * v55_data);
              float v243_bc = static_cast<float>(v226_data[8]);
              v224_acc += (v243_bc * v56_data);
              float v245_bc = static_cast<float>(v226_data[9]);
              v224_acc += (v245_bc * v57_data);
              float v247_bc = static_cast<float>(v226_data[10]);
              v224_acc += (v247_bc * v58_data);
              float v249_bc = static_cast<float>(v226_data[11]);
              v224_acc += (v249_bc * v59_data);
              r1.template select<16, 1>(96) = v224_acc;
              tensorforge::intel_esimd::simd<float, 16> v251_acc{};
              tensorforge::intel_esimd::simd<float, 16> v253_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              float v254_bc = static_cast<float>(v253_data[0]);
              v251_acc += (v254_bc * v48_data);
              float v256_bc = static_cast<float>(v253_data[1]);
              v251_acc += (v256_bc * v49_data);
              float v258_bc = static_cast<float>(v253_data[2]);
              v251_acc += (v258_bc * v50_data);
              float v260_bc = static_cast<float>(v253_data[3]);
              v251_acc += (v260_bc * v51_data);
              float v262_bc = static_cast<float>(v253_data[4]);
              v251_acc += (v262_bc * v52_data);
              float v264_bc = static_cast<float>(v253_data[5]);
              v251_acc += (v264_bc * v53_data);
              float v266_bc = static_cast<float>(v253_data[6]);
              v251_acc += (v266_bc * v54_data);
              float v268_bc = static_cast<float>(v253_data[7]);
              v251_acc += (v268_bc * v55_data);
              float v270_bc = static_cast<float>(v253_data[8]);
              v251_acc += (v270_bc * v56_data);
              float v272_bc = static_cast<float>(v253_data[9]);
              v251_acc += (v272_bc * v57_data);
              float v274_bc = static_cast<float>(v253_data[10]);
              v251_acc += (v274_bc * v58_data);
              float v276_bc = static_cast<float>(v253_data[11]);
              v251_acc += (v276_bc * v59_data);
              r1.template select<16, 1>(112) = v251_acc;
              tensorforge::intel_esimd::simd<float, 16> v278_acc{};
              tensorforge::intel_esimd::simd<float, 16> v280_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              float v281_bc = static_cast<float>(v280_data[0]);
              v278_acc += (v281_bc * v48_data);
              float v283_bc = static_cast<float>(v280_data[1]);
              v278_acc += (v283_bc * v49_data);
              float v285_bc = static_cast<float>(v280_data[2]);
              v278_acc += (v285_bc * v50_data);
              float v287_bc = static_cast<float>(v280_data[3]);
              v278_acc += (v287_bc * v51_data);
              float v289_bc = static_cast<float>(v280_data[4]);
              v278_acc += (v289_bc * v52_data);
              float v291_bc = static_cast<float>(v280_data[5]);
              v278_acc += (v291_bc * v53_data);
              float v293_bc = static_cast<float>(v280_data[6]);
              v278_acc += (v293_bc * v54_data);
              float v295_bc = static_cast<float>(v280_data[7]);
              v278_acc += (v295_bc * v55_data);
              float v297_bc = static_cast<float>(v280_data[8]);
              v278_acc += (v297_bc * v56_data);
              float v299_bc = static_cast<float>(v280_data[9]);
              v278_acc += (v299_bc * v57_data);
              float v301_bc = static_cast<float>(v280_data[10]);
              v278_acc += (v301_bc * v58_data);
              float v303_bc = static_cast<float>(v280_data[11]);
              v278_acc += (v303_bc * v59_data);
              r1.template select<16, 1>(128) = v278_acc;
              tensorforge::intel_esimd::simd<float, 16> v305_acc{};
              tensorforge::intel_esimd::simd<float, 16> v307_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              float v308_bc = static_cast<float>(v307_data[0]);
              v305_acc += (v308_bc * v48_data);
              float v310_bc = static_cast<float>(v307_data[1]);
              v305_acc += (v310_bc * v49_data);
              float v312_bc = static_cast<float>(v307_data[2]);
              v305_acc += (v312_bc * v50_data);
              float v314_bc = static_cast<float>(v307_data[3]);
              v305_acc += (v314_bc * v51_data);
              float v316_bc = static_cast<float>(v307_data[4]);
              v305_acc += (v316_bc * v52_data);
              float v318_bc = static_cast<float>(v307_data[5]);
              v305_acc += (v318_bc * v53_data);
              float v320_bc = static_cast<float>(v307_data[6]);
              v305_acc += (v320_bc * v54_data);
              float v322_bc = static_cast<float>(v307_data[7]);
              v305_acc += (v322_bc * v55_data);
              float v324_bc = static_cast<float>(v307_data[8]);
              v305_acc += (v324_bc * v56_data);
              float v326_bc = static_cast<float>(v307_data[9]);
              v305_acc += (v326_bc * v57_data);
              float v328_bc = static_cast<float>(v307_data[10]);
              v305_acc += (v328_bc * v58_data);
              float v330_bc = static_cast<float>(v307_data[11]);
              v305_acc += (v330_bc * v59_data);
              r1.template select<16, 1>(144) = v305_acc;
              tensorforge::intel_esimd::simd<float, 16> v332_acc{};
              tensorforge::intel_esimd::simd<float, 16> v334_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              float v335_bc = static_cast<float>(v334_data[0]);
              v332_acc += (v335_bc * v48_data);
              float v337_bc = static_cast<float>(v334_data[1]);
              v332_acc += (v337_bc * v49_data);
              float v339_bc = static_cast<float>(v334_data[2]);
              v332_acc += (v339_bc * v50_data);
              float v341_bc = static_cast<float>(v334_data[3]);
              v332_acc += (v341_bc * v51_data);
              float v343_bc = static_cast<float>(v334_data[4]);
              v332_acc += (v343_bc * v52_data);
              float v345_bc = static_cast<float>(v334_data[5]);
              v332_acc += (v345_bc * v53_data);
              float v347_bc = static_cast<float>(v334_data[6]);
              v332_acc += (v347_bc * v54_data);
              float v349_bc = static_cast<float>(v334_data[7]);
              v332_acc += (v349_bc * v55_data);
              float v351_bc = static_cast<float>(v334_data[8]);
              v332_acc += (v351_bc * v56_data);
              float v353_bc = static_cast<float>(v334_data[9]);
              v332_acc += (v353_bc * v57_data);
              float v355_bc = static_cast<float>(v334_data[10]);
              v332_acc += (v355_bc * v58_data);
              float v357_bc = static_cast<float>(v334_data[11]);
              v332_acc += (v357_bc * v59_data);
              r1.template select<16, 1>(160) = v332_acc;
              tensorforge::intel_esimd::simd<float, 16> v359_acc{};
              tensorforge::intel_esimd::simd<float, 16> v361_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              float v362_bc = static_cast<float>(v361_data[0]);
              v359_acc += (v362_bc * v48_data);
              float v364_bc = static_cast<float>(v361_data[1]);
              v359_acc += (v364_bc * v49_data);
              float v366_bc = static_cast<float>(v361_data[2]);
              v359_acc += (v366_bc * v50_data);
              float v368_bc = static_cast<float>(v361_data[3]);
              v359_acc += (v368_bc * v51_data);
              float v370_bc = static_cast<float>(v361_data[4]);
              v359_acc += (v370_bc * v52_data);
              float v372_bc = static_cast<float>(v361_data[5]);
              v359_acc += (v372_bc * v53_data);
              float v374_bc = static_cast<float>(v361_data[6]);
              v359_acc += (v374_bc * v54_data);
              float v376_bc = static_cast<float>(v361_data[7]);
              v359_acc += (v376_bc * v55_data);
              float v378_bc = static_cast<float>(v361_data[8]);
              v359_acc += (v378_bc * v56_data);
              float v380_bc = static_cast<float>(v361_data[9]);
              v359_acc += (v380_bc * v57_data);
              float v382_bc = static_cast<float>(v361_data[10]);
              v359_acc += (v382_bc * v58_data);
              float v384_bc = static_cast<float>(v361_data[11]);
              v359_acc += (v384_bc * v59_data);
              r1.template select<16, 1>(176) = v359_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v386_i1 = 0; v386_i1 < 12; ++v386_i1) {
                tensorforge::intel_esimd::simd<float, 6> v389_data(r1.template select<6, 1>((v386_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v386_i1 * 12)), v389_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // r5 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v395_i1 = 0; v395_i1 < 12; ++v395_i1) {
                tensorforge::intel_esimd::simd<float, 2> v400_data;
                v400_data.copy_from(glb_m4 + ((v395_i1 * 2)));
                r5.template select<2, 1>((v395_i1 * 16)) = v400_data;
              }
              // wait(r2 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s0)
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v405_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v406_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v407_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v408_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v409_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v410_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v411_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v412_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v413_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v414_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v415_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v416_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v417_acc{};
              v417_acc += (v65_bc * v405_data);
              v417_acc += (v67_bc * v406_data);
              v417_acc += (v69_bc * v407_data);
              v417_acc += (v71_bc * v408_data);
              v417_acc += (v73_bc * v409_data);
              v417_acc += (v75_bc * v410_data);
              v417_acc += (v77_bc * v411_data);
              v417_acc += (v79_bc * v412_data);
              v417_acc += (v81_bc * v413_data);
              v417_acc += (v83_bc * v414_data);
              v417_acc += (v85_bc * v415_data);
              v417_acc += (v87_bc * v416_data);
              ir3.template select<16, 1>(0) = v417_acc;
              tensorforge::intel_esimd::simd<float, 16> v446_acc{};
              v446_acc += (v92_bc * v405_data);
              v446_acc += (v94_bc * v406_data);
              v446_acc += (v96_bc * v407_data);
              v446_acc += (v98_bc * v408_data);
              v446_acc += (v100_bc * v409_data);
              v446_acc += (v102_bc * v410_data);
              v446_acc += (v104_bc * v411_data);
              v446_acc += (v106_bc * v412_data);
              v446_acc += (v108_bc * v413_data);
              v446_acc += (v110_bc * v414_data);
              v446_acc += (v112_bc * v415_data);
              v446_acc += (v114_bc * v416_data);
              ir3.template select<16, 1>(16) = v446_acc;
              tensorforge::intel_esimd::simd<float, 16> v473_acc{};
              v473_acc += (v119_bc * v405_data);
              v473_acc += (v121_bc * v406_data);
              v473_acc += (v123_bc * v407_data);
              v473_acc += (v125_bc * v408_data);
              v473_acc += (v127_bc * v409_data);
              v473_acc += (v129_bc * v410_data);
              v473_acc += (v131_bc * v411_data);
              v473_acc += (v133_bc * v412_data);
              v473_acc += (v135_bc * v413_data);
              v473_acc += (v137_bc * v414_data);
              v473_acc += (v139_bc * v415_data);
              v473_acc += (v141_bc * v416_data);
              ir3.template select<16, 1>(32) = v473_acc;
              tensorforge::intel_esimd::simd<float, 16> v500_acc{};
              v500_acc += (v146_bc * v405_data);
              v500_acc += (v148_bc * v406_data);
              v500_acc += (v150_bc * v407_data);
              v500_acc += (v152_bc * v408_data);
              v500_acc += (v154_bc * v409_data);
              v500_acc += (v156_bc * v410_data);
              v500_acc += (v158_bc * v411_data);
              v500_acc += (v160_bc * v412_data);
              v500_acc += (v162_bc * v413_data);
              v500_acc += (v164_bc * v414_data);
              v500_acc += (v166_bc * v415_data);
              v500_acc += (v168_bc * v416_data);
              ir3.template select<16, 1>(48) = v500_acc;
              tensorforge::intel_esimd::simd<float, 16> v527_acc{};
              v527_acc += (v173_bc * v405_data);
              v527_acc += (v175_bc * v406_data);
              v527_acc += (v177_bc * v407_data);
              v527_acc += (v179_bc * v408_data);
              v527_acc += (v181_bc * v409_data);
              v527_acc += (v183_bc * v410_data);
              v527_acc += (v185_bc * v411_data);
              v527_acc += (v187_bc * v412_data);
              v527_acc += (v189_bc * v413_data);
              v527_acc += (v191_bc * v414_data);
              v527_acc += (v193_bc * v415_data);
              v527_acc += (v195_bc * v416_data);
              ir3.template select<16, 1>(64) = v527_acc;
              tensorforge::intel_esimd::simd<float, 16> v554_acc{};
              v554_acc += (v200_bc * v405_data);
              v554_acc += (v202_bc * v406_data);
              v554_acc += (v204_bc * v407_data);
              v554_acc += (v206_bc * v408_data);
              v554_acc += (v208_bc * v409_data);
              v554_acc += (v210_bc * v410_data);
              v554_acc += (v212_bc * v411_data);
              v554_acc += (v214_bc * v412_data);
              v554_acc += (v216_bc * v413_data);
              v554_acc += (v218_bc * v414_data);
              v554_acc += (v220_bc * v415_data);
              v554_acc += (v222_bc * v416_data);
              ir3.template select<16, 1>(80) = v554_acc;
              tensorforge::intel_esimd::simd<float, 16> v581_acc{};
              v581_acc += (v227_bc * v405_data);
              v581_acc += (v229_bc * v406_data);
              v581_acc += (v231_bc * v407_data);
              v581_acc += (v233_bc * v408_data);
              v581_acc += (v235_bc * v409_data);
              v581_acc += (v237_bc * v410_data);
              v581_acc += (v239_bc * v411_data);
              v581_acc += (v241_bc * v412_data);
              v581_acc += (v243_bc * v413_data);
              v581_acc += (v245_bc * v414_data);
              v581_acc += (v247_bc * v415_data);
              v581_acc += (v249_bc * v416_data);
              ir3.template select<16, 1>(96) = v581_acc;
              tensorforge::intel_esimd::simd<float, 16> v608_acc{};
              v608_acc += (v254_bc * v405_data);
              v608_acc += (v256_bc * v406_data);
              v608_acc += (v258_bc * v407_data);
              v608_acc += (v260_bc * v408_data);
              v608_acc += (v262_bc * v409_data);
              v608_acc += (v264_bc * v410_data);
              v608_acc += (v266_bc * v411_data);
              v608_acc += (v268_bc * v412_data);
              v608_acc += (v270_bc * v413_data);
              v608_acc += (v272_bc * v414_data);
              v608_acc += (v274_bc * v415_data);
              v608_acc += (v276_bc * v416_data);
              ir3.template select<16, 1>(112) = v608_acc;
              tensorforge::intel_esimd::simd<float, 16> v635_acc{};
              v635_acc += (v281_bc * v405_data);
              v635_acc += (v283_bc * v406_data);
              v635_acc += (v285_bc * v407_data);
              v635_acc += (v287_bc * v408_data);
              v635_acc += (v289_bc * v409_data);
              v635_acc += (v291_bc * v410_data);
              v635_acc += (v293_bc * v411_data);
              v635_acc += (v295_bc * v412_data);
              v635_acc += (v297_bc * v413_data);
              v635_acc += (v299_bc * v414_data);
              v635_acc += (v301_bc * v415_data);
              v635_acc += (v303_bc * v416_data);
              ir3.template select<16, 1>(128) = v635_acc;
              tensorforge::intel_esimd::simd<float, 16> v662_acc{};
              v662_acc += (v308_bc * v405_data);
              v662_acc += (v310_bc * v406_data);
              v662_acc += (v312_bc * v407_data);
              v662_acc += (v314_bc * v408_data);
              v662_acc += (v316_bc * v409_data);
              v662_acc += (v318_bc * v410_data);
              v662_acc += (v320_bc * v411_data);
              v662_acc += (v322_bc * v412_data);
              v662_acc += (v324_bc * v413_data);
              v662_acc += (v326_bc * v414_data);
              v662_acc += (v328_bc * v415_data);
              v662_acc += (v330_bc * v416_data);
              ir3.template select<16, 1>(144) = v662_acc;
              tensorforge::intel_esimd::simd<float, 16> v689_acc{};
              v689_acc += (v335_bc * v405_data);
              v689_acc += (v337_bc * v406_data);
              v689_acc += (v339_bc * v407_data);
              v689_acc += (v341_bc * v408_data);
              v689_acc += (v343_bc * v409_data);
              v689_acc += (v345_bc * v410_data);
              v689_acc += (v347_bc * v411_data);
              v689_acc += (v349_bc * v412_data);
              v689_acc += (v351_bc * v413_data);
              v689_acc += (v353_bc * v414_data);
              v689_acc += (v355_bc * v415_data);
              v689_acc += (v357_bc * v416_data);
              ir3.template select<16, 1>(160) = v689_acc;
              tensorforge::intel_esimd::simd<float, 16> v716_acc{};
              v716_acc += (v362_bc * v405_data);
              v716_acc += (v364_bc * v406_data);
              v716_acc += (v366_bc * v407_data);
              v716_acc += (v368_bc * v408_data);
              v716_acc += (v370_bc * v409_data);
              v716_acc += (v372_bc * v410_data);
              v716_acc += (v374_bc * v411_data);
              v716_acc += (v376_bc * v412_data);
              v716_acc += (v378_bc * v413_data);
              v716_acc += (v380_bc * v414_data);
              v716_acc += (v382_bc * v415_data);
              v716_acc += (v384_bc * v416_data);
              ir3.template select<16, 1>(176) = v716_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v743_n1 = 0; v743_n1 < 12; ++v743_n1) {
                int32_t v744_a = v743_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v746_data(ir3.template select<6, 1>(v744_a));
                r3.template select<6, 1>(v744_a) = v746_data;
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v747_i1 = 0; v747_i1 < 12; ++v747_i1) {
                tensorforge::intel_esimd::simd<float, 6> v750_data(r3.template select<6, 1>((v747_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v747_i1 * 12))), v750_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // ir4 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir4(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v761_data(0.0f);
              v761_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v762_data(ir4.template select<16, 1>(0));
              ir4.template select<16, 1>(0) = (v762_data + v761_data);
              tensorforge::intel_esimd::simd<float, 16> v765_data(0.0f);
              v765_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v766_data(ir4.template select<16, 1>(16));
              ir4.template select<16, 1>(16) = (v766_data + v765_data);
              tensorforge::intel_esimd::simd<float, 16> v769_data(0.0f);
              v769_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v770_data(ir4.template select<16, 1>(32));
              ir4.template select<16, 1>(32) = (v770_data + v769_data);
              tensorforge::intel_esimd::simd<float, 16> v773_data(0.0f);
              v773_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v774_data(ir4.template select<16, 1>(48));
              ir4.template select<16, 1>(48) = (v774_data + v773_data);
              tensorforge::intel_esimd::simd<float, 16> v777_data(0.0f);
              v777_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v778_data(ir4.template select<16, 1>(64));
              ir4.template select<16, 1>(64) = (v778_data + v777_data);
              tensorforge::intel_esimd::simd<float, 16> v781_data(0.0f);
              v781_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v782_data(ir4.template select<16, 1>(80));
              ir4.template select<16, 1>(80) = (v782_data + v781_data);
              tensorforge::intel_esimd::simd<float, 16> v785_data(0.0f);
              v785_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v786_data(ir4.template select<16, 1>(96));
              ir4.template select<16, 1>(96) = (v786_data + v785_data);
              tensorforge::intel_esimd::simd<float, 16> v789_data(0.0f);
              v789_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v790_data(ir4.template select<16, 1>(112));
              ir4.template select<16, 1>(112) = (v790_data + v789_data);
              tensorforge::intel_esimd::simd<float, 16> v793_data(0.0f);
              v793_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v794_data(ir4.template select<16, 1>(128));
              ir4.template select<16, 1>(128) = (v794_data + v793_data);
              tensorforge::intel_esimd::simd<float, 16> v797_data(0.0f);
              v797_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v798_data(ir4.template select<16, 1>(144));
              ir4.template select<16, 1>(144) = (v798_data + v797_data);
              tensorforge::intel_esimd::simd<float, 16> v801_data(0.0f);
              v801_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v802_data(ir4.template select<16, 1>(160));
              ir4.template select<16, 1>(160) = (v802_data + v801_data);
              tensorforge::intel_esimd::simd<float, 16> v805_data(0.0f);
              v805_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v806_data(ir4.template select<16, 1>(176));
              ir4.template select<16, 1>(176) = (v806_data + v805_data);
              // r4 = ir4
              #pragma unroll
              for (int32_t v808_n1 = 0; v808_n1 < 12; ++v808_n1) {
                int32_t v809_a = v808_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v811_data(ir4.template select<12, 1>(v809_a));
                r4.template select<12, 1>(v809_a) = v811_data;
              }
              // glb_m3 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v812_i1 = 0; v812_i1 < 12; ++v812_i1) {
                tensorforge::intel_esimd::simd<float, 12> v815_data(r4.template select<12, 1>((v812_i1 * 16)));
                v815_data.copy_to(glb_m3 + ((v812_i1 * 12)));
              }
              // wait(r5 = load{g>r}(glb_m4););
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // ir6 = +(r5 * s0)
              // [(0, 2), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v822_data(r5.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v823_data(r5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v824_data(r5.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v825_data(r5.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v826_data(r5.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v827_data(r5.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v828_data(r5.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v829_data(r5.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v830_data(r5.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v831_data(r5.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v832_data(r5.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v833_data(r5.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v834_acc{};
              tensorforge::intel_esimd::simd<float, 16> v838_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v834_acc += ((static_cast<float>(v838_data[0])) * v822_data);
              v834_acc += ((static_cast<float>(v838_data[1])) * v823_data);
              v834_acc += ((static_cast<float>(v838_data[2])) * v824_data);
              v834_acc += ((static_cast<float>(v838_data[3])) * v825_data);
              v834_acc += ((static_cast<float>(v838_data[4])) * v826_data);
              v834_acc += ((static_cast<float>(v838_data[5])) * v827_data);
              v834_acc += ((static_cast<float>(v838_data[6])) * v828_data);
              v834_acc += ((static_cast<float>(v838_data[7])) * v829_data);
              v834_acc += ((static_cast<float>(v838_data[8])) * v830_data);
              v834_acc += ((static_cast<float>(v838_data[9])) * v831_data);
              v834_acc += ((static_cast<float>(v838_data[10])) * v832_data);
              v834_acc += ((static_cast<float>(v838_data[11])) * v833_data);
              ir6.template select<16, 1>(0) = v834_acc;
              tensorforge::intel_esimd::simd<float, 16> v863_acc{};
              tensorforge::intel_esimd::simd<float, 16> v865_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v863_acc += ((static_cast<float>(v865_data[0])) * v822_data);
              v863_acc += ((static_cast<float>(v865_data[1])) * v823_data);
              v863_acc += ((static_cast<float>(v865_data[2])) * v824_data);
              v863_acc += ((static_cast<float>(v865_data[3])) * v825_data);
              v863_acc += ((static_cast<float>(v865_data[4])) * v826_data);
              v863_acc += ((static_cast<float>(v865_data[5])) * v827_data);
              v863_acc += ((static_cast<float>(v865_data[6])) * v828_data);
              v863_acc += ((static_cast<float>(v865_data[7])) * v829_data);
              v863_acc += ((static_cast<float>(v865_data[8])) * v830_data);
              v863_acc += ((static_cast<float>(v865_data[9])) * v831_data);
              v863_acc += ((static_cast<float>(v865_data[10])) * v832_data);
              v863_acc += ((static_cast<float>(v865_data[11])) * v833_data);
              ir6.template select<16, 1>(16) = v863_acc;
              tensorforge::intel_esimd::simd<float, 16> v890_acc{};
              tensorforge::intel_esimd::simd<float, 16> v892_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v890_acc += ((static_cast<float>(v892_data[0])) * v822_data);
              v890_acc += ((static_cast<float>(v892_data[1])) * v823_data);
              v890_acc += ((static_cast<float>(v892_data[2])) * v824_data);
              v890_acc += ((static_cast<float>(v892_data[3])) * v825_data);
              v890_acc += ((static_cast<float>(v892_data[4])) * v826_data);
              v890_acc += ((static_cast<float>(v892_data[5])) * v827_data);
              v890_acc += ((static_cast<float>(v892_data[6])) * v828_data);
              v890_acc += ((static_cast<float>(v892_data[7])) * v829_data);
              v890_acc += ((static_cast<float>(v892_data[8])) * v830_data);
              v890_acc += ((static_cast<float>(v892_data[9])) * v831_data);
              v890_acc += ((static_cast<float>(v892_data[10])) * v832_data);
              v890_acc += ((static_cast<float>(v892_data[11])) * v833_data);
              ir6.template select<16, 1>(32) = v890_acc;
              tensorforge::intel_esimd::simd<float, 16> v917_acc{};
              tensorforge::intel_esimd::simd<float, 16> v919_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v917_acc += ((static_cast<float>(v919_data[0])) * v822_data);
              v917_acc += ((static_cast<float>(v919_data[1])) * v823_data);
              v917_acc += ((static_cast<float>(v919_data[2])) * v824_data);
              v917_acc += ((static_cast<float>(v919_data[3])) * v825_data);
              v917_acc += ((static_cast<float>(v919_data[4])) * v826_data);
              v917_acc += ((static_cast<float>(v919_data[5])) * v827_data);
              v917_acc += ((static_cast<float>(v919_data[6])) * v828_data);
              v917_acc += ((static_cast<float>(v919_data[7])) * v829_data);
              v917_acc += ((static_cast<float>(v919_data[8])) * v830_data);
              v917_acc += ((static_cast<float>(v919_data[9])) * v831_data);
              v917_acc += ((static_cast<float>(v919_data[10])) * v832_data);
              v917_acc += ((static_cast<float>(v919_data[11])) * v833_data);
              ir6.template select<16, 1>(48) = v917_acc;
              tensorforge::intel_esimd::simd<float, 16> v944_acc{};
              tensorforge::intel_esimd::simd<float, 16> v946_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v944_acc += ((static_cast<float>(v946_data[0])) * v822_data);
              v944_acc += ((static_cast<float>(v946_data[1])) * v823_data);
              v944_acc += ((static_cast<float>(v946_data[2])) * v824_data);
              v944_acc += ((static_cast<float>(v946_data[3])) * v825_data);
              v944_acc += ((static_cast<float>(v946_data[4])) * v826_data);
              v944_acc += ((static_cast<float>(v946_data[5])) * v827_data);
              v944_acc += ((static_cast<float>(v946_data[6])) * v828_data);
              v944_acc += ((static_cast<float>(v946_data[7])) * v829_data);
              v944_acc += ((static_cast<float>(v946_data[8])) * v830_data);
              v944_acc += ((static_cast<float>(v946_data[9])) * v831_data);
              v944_acc += ((static_cast<float>(v946_data[10])) * v832_data);
              v944_acc += ((static_cast<float>(v946_data[11])) * v833_data);
              ir6.template select<16, 1>(64) = v944_acc;
              tensorforge::intel_esimd::simd<float, 16> v971_acc{};
              tensorforge::intel_esimd::simd<float, 16> v973_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v971_acc += ((static_cast<float>(v973_data[0])) * v822_data);
              v971_acc += ((static_cast<float>(v973_data[1])) * v823_data);
              v971_acc += ((static_cast<float>(v973_data[2])) * v824_data);
              v971_acc += ((static_cast<float>(v973_data[3])) * v825_data);
              v971_acc += ((static_cast<float>(v973_data[4])) * v826_data);
              v971_acc += ((static_cast<float>(v973_data[5])) * v827_data);
              v971_acc += ((static_cast<float>(v973_data[6])) * v828_data);
              v971_acc += ((static_cast<float>(v973_data[7])) * v829_data);
              v971_acc += ((static_cast<float>(v973_data[8])) * v830_data);
              v971_acc += ((static_cast<float>(v973_data[9])) * v831_data);
              v971_acc += ((static_cast<float>(v973_data[10])) * v832_data);
              v971_acc += ((static_cast<float>(v973_data[11])) * v833_data);
              ir6.template select<16, 1>(80) = v971_acc;
              tensorforge::intel_esimd::simd<float, 16> v998_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1000_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v998_acc += ((static_cast<float>(v1000_data[0])) * v822_data);
              v998_acc += ((static_cast<float>(v1000_data[1])) * v823_data);
              v998_acc += ((static_cast<float>(v1000_data[2])) * v824_data);
              v998_acc += ((static_cast<float>(v1000_data[3])) * v825_data);
              v998_acc += ((static_cast<float>(v1000_data[4])) * v826_data);
              v998_acc += ((static_cast<float>(v1000_data[5])) * v827_data);
              v998_acc += ((static_cast<float>(v1000_data[6])) * v828_data);
              v998_acc += ((static_cast<float>(v1000_data[7])) * v829_data);
              v998_acc += ((static_cast<float>(v1000_data[8])) * v830_data);
              v998_acc += ((static_cast<float>(v1000_data[9])) * v831_data);
              v998_acc += ((static_cast<float>(v1000_data[10])) * v832_data);
              v998_acc += ((static_cast<float>(v1000_data[11])) * v833_data);
              ir6.template select<16, 1>(96) = v998_acc;
              tensorforge::intel_esimd::simd<float, 16> v1025_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1027_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v1025_acc += ((static_cast<float>(v1027_data[0])) * v822_data);
              v1025_acc += ((static_cast<float>(v1027_data[1])) * v823_data);
              v1025_acc += ((static_cast<float>(v1027_data[2])) * v824_data);
              v1025_acc += ((static_cast<float>(v1027_data[3])) * v825_data);
              v1025_acc += ((static_cast<float>(v1027_data[4])) * v826_data);
              v1025_acc += ((static_cast<float>(v1027_data[5])) * v827_data);
              v1025_acc += ((static_cast<float>(v1027_data[6])) * v828_data);
              v1025_acc += ((static_cast<float>(v1027_data[7])) * v829_data);
              v1025_acc += ((static_cast<float>(v1027_data[8])) * v830_data);
              v1025_acc += ((static_cast<float>(v1027_data[9])) * v831_data);
              v1025_acc += ((static_cast<float>(v1027_data[10])) * v832_data);
              v1025_acc += ((static_cast<float>(v1027_data[11])) * v833_data);
              ir6.template select<16, 1>(112) = v1025_acc;
              tensorforge::intel_esimd::simd<float, 16> v1052_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1054_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v1052_acc += ((static_cast<float>(v1054_data[0])) * v822_data);
              v1052_acc += ((static_cast<float>(v1054_data[1])) * v823_data);
              v1052_acc += ((static_cast<float>(v1054_data[2])) * v824_data);
              v1052_acc += ((static_cast<float>(v1054_data[3])) * v825_data);
              v1052_acc += ((static_cast<float>(v1054_data[4])) * v826_data);
              v1052_acc += ((static_cast<float>(v1054_data[5])) * v827_data);
              v1052_acc += ((static_cast<float>(v1054_data[6])) * v828_data);
              v1052_acc += ((static_cast<float>(v1054_data[7])) * v829_data);
              v1052_acc += ((static_cast<float>(v1054_data[8])) * v830_data);
              v1052_acc += ((static_cast<float>(v1054_data[9])) * v831_data);
              v1052_acc += ((static_cast<float>(v1054_data[10])) * v832_data);
              v1052_acc += ((static_cast<float>(v1054_data[11])) * v833_data);
              ir6.template select<16, 1>(128) = v1052_acc;
              tensorforge::intel_esimd::simd<float, 16> v1079_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1081_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              v1079_acc += ((static_cast<float>(v1081_data[0])) * v822_data);
              v1079_acc += ((static_cast<float>(v1081_data[1])) * v823_data);
              v1079_acc += ((static_cast<float>(v1081_data[2])) * v824_data);
              v1079_acc += ((static_cast<float>(v1081_data[3])) * v825_data);
              v1079_acc += ((static_cast<float>(v1081_data[4])) * v826_data);
              v1079_acc += ((static_cast<float>(v1081_data[5])) * v827_data);
              v1079_acc += ((static_cast<float>(v1081_data[6])) * v828_data);
              v1079_acc += ((static_cast<float>(v1081_data[7])) * v829_data);
              v1079_acc += ((static_cast<float>(v1081_data[8])) * v830_data);
              v1079_acc += ((static_cast<float>(v1081_data[9])) * v831_data);
              v1079_acc += ((static_cast<float>(v1081_data[10])) * v832_data);
              v1079_acc += ((static_cast<float>(v1081_data[11])) * v833_data);
              ir6.template select<16, 1>(144) = v1079_acc;
              tensorforge::intel_esimd::simd<float, 16> v1106_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1108_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              v1106_acc += ((static_cast<float>(v1108_data[0])) * v822_data);
              v1106_acc += ((static_cast<float>(v1108_data[1])) * v823_data);
              v1106_acc += ((static_cast<float>(v1108_data[2])) * v824_data);
              v1106_acc += ((static_cast<float>(v1108_data[3])) * v825_data);
              v1106_acc += ((static_cast<float>(v1108_data[4])) * v826_data);
              v1106_acc += ((static_cast<float>(v1108_data[5])) * v827_data);
              v1106_acc += ((static_cast<float>(v1108_data[6])) * v828_data);
              v1106_acc += ((static_cast<float>(v1108_data[7])) * v829_data);
              v1106_acc += ((static_cast<float>(v1108_data[8])) * v830_data);
              v1106_acc += ((static_cast<float>(v1108_data[9])) * v831_data);
              v1106_acc += ((static_cast<float>(v1108_data[10])) * v832_data);
              v1106_acc += ((static_cast<float>(v1108_data[11])) * v833_data);
              ir6.template select<16, 1>(160) = v1106_acc;
              tensorforge::intel_esimd::simd<float, 16> v1133_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1135_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              v1133_acc += ((static_cast<float>(v1135_data[0])) * v822_data);
              v1133_acc += ((static_cast<float>(v1135_data[1])) * v823_data);
              v1133_acc += ((static_cast<float>(v1135_data[2])) * v824_data);
              v1133_acc += ((static_cast<float>(v1135_data[3])) * v825_data);
              v1133_acc += ((static_cast<float>(v1135_data[4])) * v826_data);
              v1133_acc += ((static_cast<float>(v1135_data[5])) * v827_data);
              v1133_acc += ((static_cast<float>(v1135_data[6])) * v828_data);
              v1133_acc += ((static_cast<float>(v1135_data[7])) * v829_data);
              v1133_acc += ((static_cast<float>(v1135_data[8])) * v830_data);
              v1133_acc += ((static_cast<float>(v1135_data[9])) * v831_data);
              v1133_acc += ((static_cast<float>(v1135_data[10])) * v832_data);
              v1133_acc += ((static_cast<float>(v1135_data[11])) * v833_data);
              ir6.template select<16, 1>(176) = v1133_acc;
              // r6 = ir6
              #pragma unroll
              for (int32_t v1160_n1 = 0; v1160_n1 < 12; ++v1160_n1) {
                int32_t v1161_a = v1160_n1 * 16;
                tensorforge::intel_esimd::simd<float, 2> v1163_data(ir6.template select<2, 1>(v1161_a));
                r6.template select<2, 1>(v1161_a) = v1163_data;
              }
              // s1 = store{r>s, clear}(localShrMem0, r6);
              #pragma unroll
              for (int32_t v1164_z1 = 0; v1164_z1 < 12; ++v1164_z1) {
                s1[(8_i32 + (v1164_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v1171_i1 = 0; v1171_i1 < 12; ++v1171_i1) {
                tensorforge::intel_esimd::simd<float, 2> v1174_data(r6.template select<2, 1>((v1171_i1 * 16)));
                tensorforge::slmStore<float, 2>(s1 + ((6_i32 + (v1171_i1 * 12))), v1174_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r7(0.0f);
              // ir7 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir7(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v1185_data(0.0f);
              v1185_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v1186_data(ir7.template select<16, 1>(0));
              ir7.template select<16, 1>(0) = (v1186_data + v1185_data);
              tensorforge::intel_esimd::simd<float, 16> v1189_data(0.0f);
              v1189_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v1190_data(ir7.template select<16, 1>(16));
              ir7.template select<16, 1>(16) = (v1190_data + v1189_data);
              tensorforge::intel_esimd::simd<float, 16> v1193_data(0.0f);
              v1193_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v1194_data(ir7.template select<16, 1>(32));
              ir7.template select<16, 1>(32) = (v1194_data + v1193_data);
              tensorforge::intel_esimd::simd<float, 16> v1197_data(0.0f);
              v1197_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v1198_data(ir7.template select<16, 1>(48));
              ir7.template select<16, 1>(48) = (v1198_data + v1197_data);
              tensorforge::intel_esimd::simd<float, 16> v1201_data(0.0f);
              v1201_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v1202_data(ir7.template select<16, 1>(64));
              ir7.template select<16, 1>(64) = (v1202_data + v1201_data);
              tensorforge::intel_esimd::simd<float, 16> v1205_data(0.0f);
              v1205_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v1206_data(ir7.template select<16, 1>(80));
              ir7.template select<16, 1>(80) = (v1206_data + v1205_data);
              tensorforge::intel_esimd::simd<float, 16> v1209_data(0.0f);
              v1209_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v1210_data(ir7.template select<16, 1>(96));
              ir7.template select<16, 1>(96) = (v1210_data + v1209_data);
              tensorforge::intel_esimd::simd<float, 16> v1213_data(0.0f);
              v1213_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v1214_data(ir7.template select<16, 1>(112));
              ir7.template select<16, 1>(112) = (v1214_data + v1213_data);
              tensorforge::intel_esimd::simd<float, 16> v1217_data(0.0f);
              v1217_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v1218_data(ir7.template select<16, 1>(128));
              ir7.template select<16, 1>(128) = (v1218_data + v1217_data);
              tensorforge::intel_esimd::simd<float, 16> v1221_data(0.0f);
              v1221_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v1222_data(ir7.template select<16, 1>(144));
              ir7.template select<16, 1>(144) = (v1222_data + v1221_data);
              tensorforge::intel_esimd::simd<float, 16> v1225_data(0.0f);
              v1225_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v1226_data(ir7.template select<16, 1>(160));
              ir7.template select<16, 1>(160) = (v1226_data + v1225_data);
              tensorforge::intel_esimd::simd<float, 16> v1229_data(0.0f);
              v1229_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v1230_data(ir7.template select<16, 1>(176));
              ir7.template select<16, 1>(176) = (v1230_data + v1229_data);
              // r7 = ir7
              #pragma unroll
              for (int32_t v1232_n1 = 0; v1232_n1 < 12; ++v1232_n1) {
                int32_t v1233_a = v1232_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1235_data(ir7.template select<12, 1>(v1233_a));
                r7.template select<12, 1>(v1233_a) = v1235_data;
              }
              // glb_m5 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1236_i1 = 0; v1236_i1 < 12; ++v1236_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1239_data(r7.template select<12, 1>((v1236_i1 * 16)));
                v1239_data.copy_to(glb_m5 + ((v1236_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


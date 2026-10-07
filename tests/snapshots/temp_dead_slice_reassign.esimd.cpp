// === base name ===
kernel_3cacb9ab591edf08

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3cacb9ab591edf08 = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3cacb9ab591edf08(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3cacb9ab591edf08(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3cacb9ab591edf08(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_3cacb9ab591edf08(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3cacb9ab591edf08(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3cacb9ab591edf08(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3cacb9ab591edf08(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 6×12(6×12) {0..6}×{0..12} strided
        //   m1 12×12(12×12) {0..12}×{0..12} strided
        //   m2 6×12(6×12) {0..6}×{0..12} strided
        //   m3 6×12(6×12) {0..6}×{0..12} strided
        //   m4 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{6..12}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j] = m2[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m3[i,j]
        //   m4[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 72 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 144 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 12; ++v23_i1) {
                tensorforge::intel_esimd::simd<float, 6> v28_data;
                v28_data.copy_from(glb_m0 + ((v23_i1 * 6)));
                r0.template select<6, 1>((v23_i1 * 16)) = v28_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v31_ld;
              v31_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v32_ld);
              tensorforge::intel_esimd::simd<float, 16> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v33_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
                tensorforge::intel_esimd::simd<float, 6> v40_data;
                v40_data.copy_from(glb_m2 + ((v35_i1 * 6)));
                r2.template select<6, 1>((v35_i1 * 16)) = v40_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v56_acc{};
              tensorforge::intel_esimd::simd<float, 16> v60_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              float v61_bc = static_cast<float>(v60_data[0]);
              v56_acc += (v61_bc * v44_data);
              float v63_bc = static_cast<float>(v60_data[1]);
              v56_acc += (v63_bc * v45_data);
              float v65_bc = static_cast<float>(v60_data[2]);
              v56_acc += (v65_bc * v46_data);
              float v67_bc = static_cast<float>(v60_data[3]);
              v56_acc += (v67_bc * v47_data);
              float v69_bc = static_cast<float>(v60_data[4]);
              v56_acc += (v69_bc * v48_data);
              float v71_bc = static_cast<float>(v60_data[5]);
              v56_acc += (v71_bc * v49_data);
              float v73_bc = static_cast<float>(v60_data[6]);
              v56_acc += (v73_bc * v50_data);
              float v75_bc = static_cast<float>(v60_data[7]);
              v56_acc += (v75_bc * v51_data);
              float v77_bc = static_cast<float>(v60_data[8]);
              v56_acc += (v77_bc * v52_data);
              float v79_bc = static_cast<float>(v60_data[9]);
              v56_acc += (v79_bc * v53_data);
              float v81_bc = static_cast<float>(v60_data[10]);
              v56_acc += (v81_bc * v54_data);
              float v83_bc = static_cast<float>(v60_data[11]);
              v56_acc += (v83_bc * v55_data);
              r1.template select<16, 1>(0) = v56_acc;
              tensorforge::intel_esimd::simd<float, 16> v85_acc{};
              tensorforge::intel_esimd::simd<float, 16> v87_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              float v88_bc = static_cast<float>(v87_data[0]);
              v85_acc += (v88_bc * v44_data);
              float v90_bc = static_cast<float>(v87_data[1]);
              v85_acc += (v90_bc * v45_data);
              float v92_bc = static_cast<float>(v87_data[2]);
              v85_acc += (v92_bc * v46_data);
              float v94_bc = static_cast<float>(v87_data[3]);
              v85_acc += (v94_bc * v47_data);
              float v96_bc = static_cast<float>(v87_data[4]);
              v85_acc += (v96_bc * v48_data);
              float v98_bc = static_cast<float>(v87_data[5]);
              v85_acc += (v98_bc * v49_data);
              float v100_bc = static_cast<float>(v87_data[6]);
              v85_acc += (v100_bc * v50_data);
              float v102_bc = static_cast<float>(v87_data[7]);
              v85_acc += (v102_bc * v51_data);
              float v104_bc = static_cast<float>(v87_data[8]);
              v85_acc += (v104_bc * v52_data);
              float v106_bc = static_cast<float>(v87_data[9]);
              v85_acc += (v106_bc * v53_data);
              float v108_bc = static_cast<float>(v87_data[10]);
              v85_acc += (v108_bc * v54_data);
              float v110_bc = static_cast<float>(v87_data[11]);
              v85_acc += (v110_bc * v55_data);
              r1.template select<16, 1>(16) = v85_acc;
              tensorforge::intel_esimd::simd<float, 16> v112_acc{};
              tensorforge::intel_esimd::simd<float, 16> v114_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              float v115_bc = static_cast<float>(v114_data[0]);
              v112_acc += (v115_bc * v44_data);
              float v117_bc = static_cast<float>(v114_data[1]);
              v112_acc += (v117_bc * v45_data);
              float v119_bc = static_cast<float>(v114_data[2]);
              v112_acc += (v119_bc * v46_data);
              float v121_bc = static_cast<float>(v114_data[3]);
              v112_acc += (v121_bc * v47_data);
              float v123_bc = static_cast<float>(v114_data[4]);
              v112_acc += (v123_bc * v48_data);
              float v125_bc = static_cast<float>(v114_data[5]);
              v112_acc += (v125_bc * v49_data);
              float v127_bc = static_cast<float>(v114_data[6]);
              v112_acc += (v127_bc * v50_data);
              float v129_bc = static_cast<float>(v114_data[7]);
              v112_acc += (v129_bc * v51_data);
              float v131_bc = static_cast<float>(v114_data[8]);
              v112_acc += (v131_bc * v52_data);
              float v133_bc = static_cast<float>(v114_data[9]);
              v112_acc += (v133_bc * v53_data);
              float v135_bc = static_cast<float>(v114_data[10]);
              v112_acc += (v135_bc * v54_data);
              float v137_bc = static_cast<float>(v114_data[11]);
              v112_acc += (v137_bc * v55_data);
              r1.template select<16, 1>(32) = v112_acc;
              tensorforge::intel_esimd::simd<float, 16> v139_acc{};
              tensorforge::intel_esimd::simd<float, 16> v141_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              float v142_bc = static_cast<float>(v141_data[0]);
              v139_acc += (v142_bc * v44_data);
              float v144_bc = static_cast<float>(v141_data[1]);
              v139_acc += (v144_bc * v45_data);
              float v146_bc = static_cast<float>(v141_data[2]);
              v139_acc += (v146_bc * v46_data);
              float v148_bc = static_cast<float>(v141_data[3]);
              v139_acc += (v148_bc * v47_data);
              float v150_bc = static_cast<float>(v141_data[4]);
              v139_acc += (v150_bc * v48_data);
              float v152_bc = static_cast<float>(v141_data[5]);
              v139_acc += (v152_bc * v49_data);
              float v154_bc = static_cast<float>(v141_data[6]);
              v139_acc += (v154_bc * v50_data);
              float v156_bc = static_cast<float>(v141_data[7]);
              v139_acc += (v156_bc * v51_data);
              float v158_bc = static_cast<float>(v141_data[8]);
              v139_acc += (v158_bc * v52_data);
              float v160_bc = static_cast<float>(v141_data[9]);
              v139_acc += (v160_bc * v53_data);
              float v162_bc = static_cast<float>(v141_data[10]);
              v139_acc += (v162_bc * v54_data);
              float v164_bc = static_cast<float>(v141_data[11]);
              v139_acc += (v164_bc * v55_data);
              r1.template select<16, 1>(48) = v139_acc;
              tensorforge::intel_esimd::simd<float, 16> v166_acc{};
              tensorforge::intel_esimd::simd<float, 16> v168_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              float v169_bc = static_cast<float>(v168_data[0]);
              v166_acc += (v169_bc * v44_data);
              float v171_bc = static_cast<float>(v168_data[1]);
              v166_acc += (v171_bc * v45_data);
              float v173_bc = static_cast<float>(v168_data[2]);
              v166_acc += (v173_bc * v46_data);
              float v175_bc = static_cast<float>(v168_data[3]);
              v166_acc += (v175_bc * v47_data);
              float v177_bc = static_cast<float>(v168_data[4]);
              v166_acc += (v177_bc * v48_data);
              float v179_bc = static_cast<float>(v168_data[5]);
              v166_acc += (v179_bc * v49_data);
              float v181_bc = static_cast<float>(v168_data[6]);
              v166_acc += (v181_bc * v50_data);
              float v183_bc = static_cast<float>(v168_data[7]);
              v166_acc += (v183_bc * v51_data);
              float v185_bc = static_cast<float>(v168_data[8]);
              v166_acc += (v185_bc * v52_data);
              float v187_bc = static_cast<float>(v168_data[9]);
              v166_acc += (v187_bc * v53_data);
              float v189_bc = static_cast<float>(v168_data[10]);
              v166_acc += (v189_bc * v54_data);
              float v191_bc = static_cast<float>(v168_data[11]);
              v166_acc += (v191_bc * v55_data);
              r1.template select<16, 1>(64) = v166_acc;
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              tensorforge::intel_esimd::simd<float, 16> v195_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              float v196_bc = static_cast<float>(v195_data[0]);
              v193_acc += (v196_bc * v44_data);
              float v198_bc = static_cast<float>(v195_data[1]);
              v193_acc += (v198_bc * v45_data);
              float v200_bc = static_cast<float>(v195_data[2]);
              v193_acc += (v200_bc * v46_data);
              float v202_bc = static_cast<float>(v195_data[3]);
              v193_acc += (v202_bc * v47_data);
              float v204_bc = static_cast<float>(v195_data[4]);
              v193_acc += (v204_bc * v48_data);
              float v206_bc = static_cast<float>(v195_data[5]);
              v193_acc += (v206_bc * v49_data);
              float v208_bc = static_cast<float>(v195_data[6]);
              v193_acc += (v208_bc * v50_data);
              float v210_bc = static_cast<float>(v195_data[7]);
              v193_acc += (v210_bc * v51_data);
              float v212_bc = static_cast<float>(v195_data[8]);
              v193_acc += (v212_bc * v52_data);
              float v214_bc = static_cast<float>(v195_data[9]);
              v193_acc += (v214_bc * v53_data);
              float v216_bc = static_cast<float>(v195_data[10]);
              v193_acc += (v216_bc * v54_data);
              float v218_bc = static_cast<float>(v195_data[11]);
              v193_acc += (v218_bc * v55_data);
              r1.template select<16, 1>(80) = v193_acc;
              tensorforge::intel_esimd::simd<float, 16> v220_acc{};
              tensorforge::intel_esimd::simd<float, 16> v222_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              float v223_bc = static_cast<float>(v222_data[0]);
              v220_acc += (v223_bc * v44_data);
              float v225_bc = static_cast<float>(v222_data[1]);
              v220_acc += (v225_bc * v45_data);
              float v227_bc = static_cast<float>(v222_data[2]);
              v220_acc += (v227_bc * v46_data);
              float v229_bc = static_cast<float>(v222_data[3]);
              v220_acc += (v229_bc * v47_data);
              float v231_bc = static_cast<float>(v222_data[4]);
              v220_acc += (v231_bc * v48_data);
              float v233_bc = static_cast<float>(v222_data[5]);
              v220_acc += (v233_bc * v49_data);
              float v235_bc = static_cast<float>(v222_data[6]);
              v220_acc += (v235_bc * v50_data);
              float v237_bc = static_cast<float>(v222_data[7]);
              v220_acc += (v237_bc * v51_data);
              float v239_bc = static_cast<float>(v222_data[8]);
              v220_acc += (v239_bc * v52_data);
              float v241_bc = static_cast<float>(v222_data[9]);
              v220_acc += (v241_bc * v53_data);
              float v243_bc = static_cast<float>(v222_data[10]);
              v220_acc += (v243_bc * v54_data);
              float v245_bc = static_cast<float>(v222_data[11]);
              v220_acc += (v245_bc * v55_data);
              r1.template select<16, 1>(96) = v220_acc;
              tensorforge::intel_esimd::simd<float, 16> v247_acc{};
              tensorforge::intel_esimd::simd<float, 16> v249_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              float v250_bc = static_cast<float>(v249_data[0]);
              v247_acc += (v250_bc * v44_data);
              float v252_bc = static_cast<float>(v249_data[1]);
              v247_acc += (v252_bc * v45_data);
              float v254_bc = static_cast<float>(v249_data[2]);
              v247_acc += (v254_bc * v46_data);
              float v256_bc = static_cast<float>(v249_data[3]);
              v247_acc += (v256_bc * v47_data);
              float v258_bc = static_cast<float>(v249_data[4]);
              v247_acc += (v258_bc * v48_data);
              float v260_bc = static_cast<float>(v249_data[5]);
              v247_acc += (v260_bc * v49_data);
              float v262_bc = static_cast<float>(v249_data[6]);
              v247_acc += (v262_bc * v50_data);
              float v264_bc = static_cast<float>(v249_data[7]);
              v247_acc += (v264_bc * v51_data);
              float v266_bc = static_cast<float>(v249_data[8]);
              v247_acc += (v266_bc * v52_data);
              float v268_bc = static_cast<float>(v249_data[9]);
              v247_acc += (v268_bc * v53_data);
              float v270_bc = static_cast<float>(v249_data[10]);
              v247_acc += (v270_bc * v54_data);
              float v272_bc = static_cast<float>(v249_data[11]);
              v247_acc += (v272_bc * v55_data);
              r1.template select<16, 1>(112) = v247_acc;
              tensorforge::intel_esimd::simd<float, 16> v274_acc{};
              tensorforge::intel_esimd::simd<float, 16> v276_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              float v277_bc = static_cast<float>(v276_data[0]);
              v274_acc += (v277_bc * v44_data);
              float v279_bc = static_cast<float>(v276_data[1]);
              v274_acc += (v279_bc * v45_data);
              float v281_bc = static_cast<float>(v276_data[2]);
              v274_acc += (v281_bc * v46_data);
              float v283_bc = static_cast<float>(v276_data[3]);
              v274_acc += (v283_bc * v47_data);
              float v285_bc = static_cast<float>(v276_data[4]);
              v274_acc += (v285_bc * v48_data);
              float v287_bc = static_cast<float>(v276_data[5]);
              v274_acc += (v287_bc * v49_data);
              float v289_bc = static_cast<float>(v276_data[6]);
              v274_acc += (v289_bc * v50_data);
              float v291_bc = static_cast<float>(v276_data[7]);
              v274_acc += (v291_bc * v51_data);
              float v293_bc = static_cast<float>(v276_data[8]);
              v274_acc += (v293_bc * v52_data);
              float v295_bc = static_cast<float>(v276_data[9]);
              v274_acc += (v295_bc * v53_data);
              float v297_bc = static_cast<float>(v276_data[10]);
              v274_acc += (v297_bc * v54_data);
              float v299_bc = static_cast<float>(v276_data[11]);
              v274_acc += (v299_bc * v55_data);
              r1.template select<16, 1>(128) = v274_acc;
              tensorforge::intel_esimd::simd<float, 16> v301_acc{};
              tensorforge::intel_esimd::simd<float, 16> v303_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              float v304_bc = static_cast<float>(v303_data[0]);
              v301_acc += (v304_bc * v44_data);
              float v306_bc = static_cast<float>(v303_data[1]);
              v301_acc += (v306_bc * v45_data);
              float v308_bc = static_cast<float>(v303_data[2]);
              v301_acc += (v308_bc * v46_data);
              float v310_bc = static_cast<float>(v303_data[3]);
              v301_acc += (v310_bc * v47_data);
              float v312_bc = static_cast<float>(v303_data[4]);
              v301_acc += (v312_bc * v48_data);
              float v314_bc = static_cast<float>(v303_data[5]);
              v301_acc += (v314_bc * v49_data);
              float v316_bc = static_cast<float>(v303_data[6]);
              v301_acc += (v316_bc * v50_data);
              float v318_bc = static_cast<float>(v303_data[7]);
              v301_acc += (v318_bc * v51_data);
              float v320_bc = static_cast<float>(v303_data[8]);
              v301_acc += (v320_bc * v52_data);
              float v322_bc = static_cast<float>(v303_data[9]);
              v301_acc += (v322_bc * v53_data);
              float v324_bc = static_cast<float>(v303_data[10]);
              v301_acc += (v324_bc * v54_data);
              float v326_bc = static_cast<float>(v303_data[11]);
              v301_acc += (v326_bc * v55_data);
              r1.template select<16, 1>(144) = v301_acc;
              tensorforge::intel_esimd::simd<float, 16> v328_acc{};
              tensorforge::intel_esimd::simd<float, 16> v330_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              float v331_bc = static_cast<float>(v330_data[0]);
              v328_acc += (v331_bc * v44_data);
              float v333_bc = static_cast<float>(v330_data[1]);
              v328_acc += (v333_bc * v45_data);
              float v335_bc = static_cast<float>(v330_data[2]);
              v328_acc += (v335_bc * v46_data);
              float v337_bc = static_cast<float>(v330_data[3]);
              v328_acc += (v337_bc * v47_data);
              float v339_bc = static_cast<float>(v330_data[4]);
              v328_acc += (v339_bc * v48_data);
              float v341_bc = static_cast<float>(v330_data[5]);
              v328_acc += (v341_bc * v49_data);
              float v343_bc = static_cast<float>(v330_data[6]);
              v328_acc += (v343_bc * v50_data);
              float v345_bc = static_cast<float>(v330_data[7]);
              v328_acc += (v345_bc * v51_data);
              float v347_bc = static_cast<float>(v330_data[8]);
              v328_acc += (v347_bc * v52_data);
              float v349_bc = static_cast<float>(v330_data[9]);
              v328_acc += (v349_bc * v53_data);
              float v351_bc = static_cast<float>(v330_data[10]);
              v328_acc += (v351_bc * v54_data);
              float v353_bc = static_cast<float>(v330_data[11]);
              v328_acc += (v353_bc * v55_data);
              r1.template select<16, 1>(160) = v328_acc;
              tensorforge::intel_esimd::simd<float, 16> v355_acc{};
              tensorforge::intel_esimd::simd<float, 16> v357_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              float v358_bc = static_cast<float>(v357_data[0]);
              v355_acc += (v358_bc * v44_data);
              float v360_bc = static_cast<float>(v357_data[1]);
              v355_acc += (v360_bc * v45_data);
              float v362_bc = static_cast<float>(v357_data[2]);
              v355_acc += (v362_bc * v46_data);
              float v364_bc = static_cast<float>(v357_data[3]);
              v355_acc += (v364_bc * v47_data);
              float v366_bc = static_cast<float>(v357_data[4]);
              v355_acc += (v366_bc * v48_data);
              float v368_bc = static_cast<float>(v357_data[5]);
              v355_acc += (v368_bc * v49_data);
              float v370_bc = static_cast<float>(v357_data[6]);
              v355_acc += (v370_bc * v50_data);
              float v372_bc = static_cast<float>(v357_data[7]);
              v355_acc += (v372_bc * v51_data);
              float v374_bc = static_cast<float>(v357_data[8]);
              v355_acc += (v374_bc * v52_data);
              float v376_bc = static_cast<float>(v357_data[9]);
              v355_acc += (v376_bc * v53_data);
              float v378_bc = static_cast<float>(v357_data[10]);
              v355_acc += (v378_bc * v54_data);
              float v380_bc = static_cast<float>(v357_data[11]);
              v355_acc += (v380_bc * v55_data);
              r1.template select<16, 1>(176) = v355_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v382_i1 = 0; v382_i1 < 12; ++v382_i1) {
                tensorforge::intel_esimd::simd<float, 6> v385_data(r1.template select<6, 1>((v382_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v382_i1 * 12))), v385_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v392_i1 = 0; v392_i1 < 12; ++v392_i1) {
                tensorforge::intel_esimd::simd<float, 6> v397_data;
                v397_data.copy_from(glb_m3 + ((v392_i1 * 6)));
                r4.template select<6, 1>((v392_i1 * 16)) = v397_data;
              }
              // wait(r2 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s0)
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v402_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v403_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v404_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v405_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v406_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v407_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v408_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v409_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v410_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v411_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v412_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v413_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v414_acc{};
              v414_acc += (v61_bc * v402_data);
              v414_acc += (v63_bc * v403_data);
              v414_acc += (v65_bc * v404_data);
              v414_acc += (v67_bc * v405_data);
              v414_acc += (v69_bc * v406_data);
              v414_acc += (v71_bc * v407_data);
              v414_acc += (v73_bc * v408_data);
              v414_acc += (v75_bc * v409_data);
              v414_acc += (v77_bc * v410_data);
              v414_acc += (v79_bc * v411_data);
              v414_acc += (v81_bc * v412_data);
              v414_acc += (v83_bc * v413_data);
              ir3.template select<16, 1>(0) = v414_acc;
              tensorforge::intel_esimd::simd<float, 16> v443_acc{};
              v443_acc += (v88_bc * v402_data);
              v443_acc += (v90_bc * v403_data);
              v443_acc += (v92_bc * v404_data);
              v443_acc += (v94_bc * v405_data);
              v443_acc += (v96_bc * v406_data);
              v443_acc += (v98_bc * v407_data);
              v443_acc += (v100_bc * v408_data);
              v443_acc += (v102_bc * v409_data);
              v443_acc += (v104_bc * v410_data);
              v443_acc += (v106_bc * v411_data);
              v443_acc += (v108_bc * v412_data);
              v443_acc += (v110_bc * v413_data);
              ir3.template select<16, 1>(16) = v443_acc;
              tensorforge::intel_esimd::simd<float, 16> v470_acc{};
              v470_acc += (v115_bc * v402_data);
              v470_acc += (v117_bc * v403_data);
              v470_acc += (v119_bc * v404_data);
              v470_acc += (v121_bc * v405_data);
              v470_acc += (v123_bc * v406_data);
              v470_acc += (v125_bc * v407_data);
              v470_acc += (v127_bc * v408_data);
              v470_acc += (v129_bc * v409_data);
              v470_acc += (v131_bc * v410_data);
              v470_acc += (v133_bc * v411_data);
              v470_acc += (v135_bc * v412_data);
              v470_acc += (v137_bc * v413_data);
              ir3.template select<16, 1>(32) = v470_acc;
              tensorforge::intel_esimd::simd<float, 16> v497_acc{};
              v497_acc += (v142_bc * v402_data);
              v497_acc += (v144_bc * v403_data);
              v497_acc += (v146_bc * v404_data);
              v497_acc += (v148_bc * v405_data);
              v497_acc += (v150_bc * v406_data);
              v497_acc += (v152_bc * v407_data);
              v497_acc += (v154_bc * v408_data);
              v497_acc += (v156_bc * v409_data);
              v497_acc += (v158_bc * v410_data);
              v497_acc += (v160_bc * v411_data);
              v497_acc += (v162_bc * v412_data);
              v497_acc += (v164_bc * v413_data);
              ir3.template select<16, 1>(48) = v497_acc;
              tensorforge::intel_esimd::simd<float, 16> v524_acc{};
              v524_acc += (v169_bc * v402_data);
              v524_acc += (v171_bc * v403_data);
              v524_acc += (v173_bc * v404_data);
              v524_acc += (v175_bc * v405_data);
              v524_acc += (v177_bc * v406_data);
              v524_acc += (v179_bc * v407_data);
              v524_acc += (v181_bc * v408_data);
              v524_acc += (v183_bc * v409_data);
              v524_acc += (v185_bc * v410_data);
              v524_acc += (v187_bc * v411_data);
              v524_acc += (v189_bc * v412_data);
              v524_acc += (v191_bc * v413_data);
              ir3.template select<16, 1>(64) = v524_acc;
              tensorforge::intel_esimd::simd<float, 16> v551_acc{};
              v551_acc += (v196_bc * v402_data);
              v551_acc += (v198_bc * v403_data);
              v551_acc += (v200_bc * v404_data);
              v551_acc += (v202_bc * v405_data);
              v551_acc += (v204_bc * v406_data);
              v551_acc += (v206_bc * v407_data);
              v551_acc += (v208_bc * v408_data);
              v551_acc += (v210_bc * v409_data);
              v551_acc += (v212_bc * v410_data);
              v551_acc += (v214_bc * v411_data);
              v551_acc += (v216_bc * v412_data);
              v551_acc += (v218_bc * v413_data);
              ir3.template select<16, 1>(80) = v551_acc;
              tensorforge::intel_esimd::simd<float, 16> v578_acc{};
              v578_acc += (v223_bc * v402_data);
              v578_acc += (v225_bc * v403_data);
              v578_acc += (v227_bc * v404_data);
              v578_acc += (v229_bc * v405_data);
              v578_acc += (v231_bc * v406_data);
              v578_acc += (v233_bc * v407_data);
              v578_acc += (v235_bc * v408_data);
              v578_acc += (v237_bc * v409_data);
              v578_acc += (v239_bc * v410_data);
              v578_acc += (v241_bc * v411_data);
              v578_acc += (v243_bc * v412_data);
              v578_acc += (v245_bc * v413_data);
              ir3.template select<16, 1>(96) = v578_acc;
              tensorforge::intel_esimd::simd<float, 16> v605_acc{};
              v605_acc += (v250_bc * v402_data);
              v605_acc += (v252_bc * v403_data);
              v605_acc += (v254_bc * v404_data);
              v605_acc += (v256_bc * v405_data);
              v605_acc += (v258_bc * v406_data);
              v605_acc += (v260_bc * v407_data);
              v605_acc += (v262_bc * v408_data);
              v605_acc += (v264_bc * v409_data);
              v605_acc += (v266_bc * v410_data);
              v605_acc += (v268_bc * v411_data);
              v605_acc += (v270_bc * v412_data);
              v605_acc += (v272_bc * v413_data);
              ir3.template select<16, 1>(112) = v605_acc;
              tensorforge::intel_esimd::simd<float, 16> v632_acc{};
              v632_acc += (v277_bc * v402_data);
              v632_acc += (v279_bc * v403_data);
              v632_acc += (v281_bc * v404_data);
              v632_acc += (v283_bc * v405_data);
              v632_acc += (v285_bc * v406_data);
              v632_acc += (v287_bc * v407_data);
              v632_acc += (v289_bc * v408_data);
              v632_acc += (v291_bc * v409_data);
              v632_acc += (v293_bc * v410_data);
              v632_acc += (v295_bc * v411_data);
              v632_acc += (v297_bc * v412_data);
              v632_acc += (v299_bc * v413_data);
              ir3.template select<16, 1>(128) = v632_acc;
              tensorforge::intel_esimd::simd<float, 16> v659_acc{};
              v659_acc += (v304_bc * v402_data);
              v659_acc += (v306_bc * v403_data);
              v659_acc += (v308_bc * v404_data);
              v659_acc += (v310_bc * v405_data);
              v659_acc += (v312_bc * v406_data);
              v659_acc += (v314_bc * v407_data);
              v659_acc += (v316_bc * v408_data);
              v659_acc += (v318_bc * v409_data);
              v659_acc += (v320_bc * v410_data);
              v659_acc += (v322_bc * v411_data);
              v659_acc += (v324_bc * v412_data);
              v659_acc += (v326_bc * v413_data);
              ir3.template select<16, 1>(144) = v659_acc;
              tensorforge::intel_esimd::simd<float, 16> v686_acc{};
              v686_acc += (v331_bc * v402_data);
              v686_acc += (v333_bc * v403_data);
              v686_acc += (v335_bc * v404_data);
              v686_acc += (v337_bc * v405_data);
              v686_acc += (v339_bc * v406_data);
              v686_acc += (v341_bc * v407_data);
              v686_acc += (v343_bc * v408_data);
              v686_acc += (v345_bc * v409_data);
              v686_acc += (v347_bc * v410_data);
              v686_acc += (v349_bc * v411_data);
              v686_acc += (v351_bc * v412_data);
              v686_acc += (v353_bc * v413_data);
              ir3.template select<16, 1>(160) = v686_acc;
              tensorforge::intel_esimd::simd<float, 16> v713_acc{};
              v713_acc += (v358_bc * v402_data);
              v713_acc += (v360_bc * v403_data);
              v713_acc += (v362_bc * v404_data);
              v713_acc += (v364_bc * v405_data);
              v713_acc += (v366_bc * v406_data);
              v713_acc += (v368_bc * v407_data);
              v713_acc += (v370_bc * v408_data);
              v713_acc += (v372_bc * v409_data);
              v713_acc += (v374_bc * v410_data);
              v713_acc += (v376_bc * v411_data);
              v713_acc += (v378_bc * v412_data);
              v713_acc += (v380_bc * v413_data);
              ir3.template select<16, 1>(176) = v713_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v740_n1 = 0; v740_n1 < 12; ++v740_n1) {
                int32_t v741_a = v740_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v743_data(ir3.template select<6, 1>(v741_a));
                r3.template select<6, 1>(v741_a) = v743_data;
              }
              // s1 = store{r>s, clear}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v744_z1 = 0; v744_z1 < 12; ++v744_z1) {
                s1[(6_i32 + (v744_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v751_i1 = 0; v751_i1 < 12; ++v751_i1) {
                tensorforge::intel_esimd::simd<float, 6> v754_data(r3.template select<6, 1>((v751_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v751_i1 * 12)), v754_data);
              }
              // wait(r4 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // ir5 = +(r4)
              // [(0, 6), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v761_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v762_data(ir5.template select<16, 1>(0));
              ir5.template select<16, 1>(0) = (v762_data + v761_data);
              tensorforge::intel_esimd::simd<float, 16> v764_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v765_data(ir5.template select<16, 1>(16));
              ir5.template select<16, 1>(16) = (v765_data + v764_data);
              tensorforge::intel_esimd::simd<float, 16> v767_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v768_data(ir5.template select<16, 1>(32));
              ir5.template select<16, 1>(32) = (v768_data + v767_data);
              tensorforge::intel_esimd::simd<float, 16> v770_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v771_data(ir5.template select<16, 1>(48));
              ir5.template select<16, 1>(48) = (v771_data + v770_data);
              tensorforge::intel_esimd::simd<float, 16> v773_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v774_data(ir5.template select<16, 1>(64));
              ir5.template select<16, 1>(64) = (v774_data + v773_data);
              tensorforge::intel_esimd::simd<float, 16> v776_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v777_data(ir5.template select<16, 1>(80));
              ir5.template select<16, 1>(80) = (v777_data + v776_data);
              tensorforge::intel_esimd::simd<float, 16> v779_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v780_data(ir5.template select<16, 1>(96));
              ir5.template select<16, 1>(96) = (v780_data + v779_data);
              tensorforge::intel_esimd::simd<float, 16> v782_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v783_data(ir5.template select<16, 1>(112));
              ir5.template select<16, 1>(112) = (v783_data + v782_data);
              tensorforge::intel_esimd::simd<float, 16> v785_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v786_data(ir5.template select<16, 1>(128));
              ir5.template select<16, 1>(128) = (v786_data + v785_data);
              tensorforge::intel_esimd::simd<float, 16> v788_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v789_data(ir5.template select<16, 1>(144));
              ir5.template select<16, 1>(144) = (v789_data + v788_data);
              tensorforge::intel_esimd::simd<float, 16> v791_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v792_data(ir5.template select<16, 1>(160));
              ir5.template select<16, 1>(160) = (v792_data + v791_data);
              tensorforge::intel_esimd::simd<float, 16> v794_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v795_data(ir5.template select<16, 1>(176));
              ir5.template select<16, 1>(176) = (v795_data + v794_data);
              // r5 = ir5
              #pragma unroll
              for (int32_t v797_n1 = 0; v797_n1 < 12; ++v797_n1) {
                int32_t v798_a = v797_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v800_data(ir5.template select<6, 1>(v798_a));
                r5.template select<6, 1>(v798_a) = v800_data;
              }
              // s1 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v801_i1 = 0; v801_i1 < 12; ++v801_i1) {
                tensorforge::intel_esimd::simd<float, 6> v804_data(r5.template select<6, 1>((v801_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v801_i1 * 12))), v804_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // ir6 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v815_data(0.0f);
              v815_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v816_data(ir6.template select<16, 1>(0));
              ir6.template select<16, 1>(0) = (v816_data + v815_data);
              tensorforge::intel_esimd::simd<float, 16> v819_data(0.0f);
              v819_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v820_data(ir6.template select<16, 1>(16));
              ir6.template select<16, 1>(16) = (v820_data + v819_data);
              tensorforge::intel_esimd::simd<float, 16> v823_data(0.0f);
              v823_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v824_data(ir6.template select<16, 1>(32));
              ir6.template select<16, 1>(32) = (v824_data + v823_data);
              tensorforge::intel_esimd::simd<float, 16> v827_data(0.0f);
              v827_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v828_data(ir6.template select<16, 1>(48));
              ir6.template select<16, 1>(48) = (v828_data + v827_data);
              tensorforge::intel_esimd::simd<float, 16> v831_data(0.0f);
              v831_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v832_data(ir6.template select<16, 1>(64));
              ir6.template select<16, 1>(64) = (v832_data + v831_data);
              tensorforge::intel_esimd::simd<float, 16> v835_data(0.0f);
              v835_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v836_data(ir6.template select<16, 1>(80));
              ir6.template select<16, 1>(80) = (v836_data + v835_data);
              tensorforge::intel_esimd::simd<float, 16> v839_data(0.0f);
              v839_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v840_data(ir6.template select<16, 1>(96));
              ir6.template select<16, 1>(96) = (v840_data + v839_data);
              tensorforge::intel_esimd::simd<float, 16> v843_data(0.0f);
              v843_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v844_data(ir6.template select<16, 1>(112));
              ir6.template select<16, 1>(112) = (v844_data + v843_data);
              tensorforge::intel_esimd::simd<float, 16> v847_data(0.0f);
              v847_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v848_data(ir6.template select<16, 1>(128));
              ir6.template select<16, 1>(128) = (v848_data + v847_data);
              tensorforge::intel_esimd::simd<float, 16> v851_data(0.0f);
              v851_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v852_data(ir6.template select<16, 1>(144));
              ir6.template select<16, 1>(144) = (v852_data + v851_data);
              tensorforge::intel_esimd::simd<float, 16> v855_data(0.0f);
              v855_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v856_data(ir6.template select<16, 1>(160));
              ir6.template select<16, 1>(160) = (v856_data + v855_data);
              tensorforge::intel_esimd::simd<float, 16> v859_data(0.0f);
              v859_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v860_data(ir6.template select<16, 1>(176));
              ir6.template select<16, 1>(176) = (v860_data + v859_data);
              // r6 = ir6
              #pragma unroll
              for (int32_t v862_n1 = 0; v862_n1 < 12; ++v862_n1) {
                int32_t v863_a = v862_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v865_data(ir6.template select<12, 1>(v863_a));
                r6.template select<12, 1>(v863_a) = v865_data;
              }
              // glb_m4 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v866_i1 = 0; v866_i1 < 12; ++v866_i1) {
                tensorforge::intel_esimd::simd<float, 12> v869_data(r6.template select<12, 1>((v866_i1 * 16)));
                v869_data.copy_to(glb_m4 + ((v866_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


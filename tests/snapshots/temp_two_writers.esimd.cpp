// === base name ===
kernel_759f3297a838b70d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_759f3297a838b70d = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_759f3297a838b70d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_759f3297a838b70d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_759f3297a838b70d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_759f3297a838b70d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_759f3297a838b70d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_759f3297a838b70d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_759f3297a838b70d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 32×32(6×12) {0..6}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(6×12) {0..6}×{0..12} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        //   m4 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
        //   m3[i,j] = m4[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
              float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 144 + 0 + m4_extraOffset];
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
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v382_i1 = 0; v382_i1 < 12; ++v382_i1) {
                tensorforge::intel_esimd::simd<float, 6> v387_data;
                v387_data.copy_from(glb_m2 + ((v382_i1 * 6)));
                r2.template select<6, 1>((v382_i1 * 16)) = v387_data;
              }
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v47_acc{};
              tensorforge::intel_esimd::simd<float, 16> v51_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              float v52_bc = static_cast<float>(v51_data[0]);
              v47_acc += (v52_bc * v35_data);
              float v54_bc = static_cast<float>(v51_data[1]);
              v47_acc += (v54_bc * v36_data);
              float v56_bc = static_cast<float>(v51_data[2]);
              v47_acc += (v56_bc * v37_data);
              float v58_bc = static_cast<float>(v51_data[3]);
              v47_acc += (v58_bc * v38_data);
              float v60_bc = static_cast<float>(v51_data[4]);
              v47_acc += (v60_bc * v39_data);
              float v62_bc = static_cast<float>(v51_data[5]);
              v47_acc += (v62_bc * v40_data);
              float v64_bc = static_cast<float>(v51_data[6]);
              v47_acc += (v64_bc * v41_data);
              float v66_bc = static_cast<float>(v51_data[7]);
              v47_acc += (v66_bc * v42_data);
              float v68_bc = static_cast<float>(v51_data[8]);
              v47_acc += (v68_bc * v43_data);
              float v70_bc = static_cast<float>(v51_data[9]);
              v47_acc += (v70_bc * v44_data);
              float v72_bc = static_cast<float>(v51_data[10]);
              v47_acc += (v72_bc * v45_data);
              float v74_bc = static_cast<float>(v51_data[11]);
              v47_acc += (v74_bc * v46_data);
              r1.template select<16, 1>(0) = v47_acc;
              tensorforge::intel_esimd::simd<float, 16> v76_acc{};
              tensorforge::intel_esimd::simd<float, 16> v78_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              float v79_bc = static_cast<float>(v78_data[0]);
              v76_acc += (v79_bc * v35_data);
              float v81_bc = static_cast<float>(v78_data[1]);
              v76_acc += (v81_bc * v36_data);
              float v83_bc = static_cast<float>(v78_data[2]);
              v76_acc += (v83_bc * v37_data);
              float v85_bc = static_cast<float>(v78_data[3]);
              v76_acc += (v85_bc * v38_data);
              float v87_bc = static_cast<float>(v78_data[4]);
              v76_acc += (v87_bc * v39_data);
              float v89_bc = static_cast<float>(v78_data[5]);
              v76_acc += (v89_bc * v40_data);
              float v91_bc = static_cast<float>(v78_data[6]);
              v76_acc += (v91_bc * v41_data);
              float v93_bc = static_cast<float>(v78_data[7]);
              v76_acc += (v93_bc * v42_data);
              float v95_bc = static_cast<float>(v78_data[8]);
              v76_acc += (v95_bc * v43_data);
              float v97_bc = static_cast<float>(v78_data[9]);
              v76_acc += (v97_bc * v44_data);
              float v99_bc = static_cast<float>(v78_data[10]);
              v76_acc += (v99_bc * v45_data);
              float v101_bc = static_cast<float>(v78_data[11]);
              v76_acc += (v101_bc * v46_data);
              r1.template select<16, 1>(16) = v76_acc;
              tensorforge::intel_esimd::simd<float, 16> v103_acc{};
              tensorforge::intel_esimd::simd<float, 16> v105_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              float v106_bc = static_cast<float>(v105_data[0]);
              v103_acc += (v106_bc * v35_data);
              float v108_bc = static_cast<float>(v105_data[1]);
              v103_acc += (v108_bc * v36_data);
              float v110_bc = static_cast<float>(v105_data[2]);
              v103_acc += (v110_bc * v37_data);
              float v112_bc = static_cast<float>(v105_data[3]);
              v103_acc += (v112_bc * v38_data);
              float v114_bc = static_cast<float>(v105_data[4]);
              v103_acc += (v114_bc * v39_data);
              float v116_bc = static_cast<float>(v105_data[5]);
              v103_acc += (v116_bc * v40_data);
              float v118_bc = static_cast<float>(v105_data[6]);
              v103_acc += (v118_bc * v41_data);
              float v120_bc = static_cast<float>(v105_data[7]);
              v103_acc += (v120_bc * v42_data);
              float v122_bc = static_cast<float>(v105_data[8]);
              v103_acc += (v122_bc * v43_data);
              float v124_bc = static_cast<float>(v105_data[9]);
              v103_acc += (v124_bc * v44_data);
              float v126_bc = static_cast<float>(v105_data[10]);
              v103_acc += (v126_bc * v45_data);
              float v128_bc = static_cast<float>(v105_data[11]);
              v103_acc += (v128_bc * v46_data);
              r1.template select<16, 1>(32) = v103_acc;
              tensorforge::intel_esimd::simd<float, 16> v130_acc{};
              tensorforge::intel_esimd::simd<float, 16> v132_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              float v133_bc = static_cast<float>(v132_data[0]);
              v130_acc += (v133_bc * v35_data);
              float v135_bc = static_cast<float>(v132_data[1]);
              v130_acc += (v135_bc * v36_data);
              float v137_bc = static_cast<float>(v132_data[2]);
              v130_acc += (v137_bc * v37_data);
              float v139_bc = static_cast<float>(v132_data[3]);
              v130_acc += (v139_bc * v38_data);
              float v141_bc = static_cast<float>(v132_data[4]);
              v130_acc += (v141_bc * v39_data);
              float v143_bc = static_cast<float>(v132_data[5]);
              v130_acc += (v143_bc * v40_data);
              float v145_bc = static_cast<float>(v132_data[6]);
              v130_acc += (v145_bc * v41_data);
              float v147_bc = static_cast<float>(v132_data[7]);
              v130_acc += (v147_bc * v42_data);
              float v149_bc = static_cast<float>(v132_data[8]);
              v130_acc += (v149_bc * v43_data);
              float v151_bc = static_cast<float>(v132_data[9]);
              v130_acc += (v151_bc * v44_data);
              float v153_bc = static_cast<float>(v132_data[10]);
              v130_acc += (v153_bc * v45_data);
              float v155_bc = static_cast<float>(v132_data[11]);
              v130_acc += (v155_bc * v46_data);
              r1.template select<16, 1>(48) = v130_acc;
              tensorforge::intel_esimd::simd<float, 16> v157_acc{};
              tensorforge::intel_esimd::simd<float, 16> v159_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              float v160_bc = static_cast<float>(v159_data[0]);
              v157_acc += (v160_bc * v35_data);
              float v162_bc = static_cast<float>(v159_data[1]);
              v157_acc += (v162_bc * v36_data);
              float v164_bc = static_cast<float>(v159_data[2]);
              v157_acc += (v164_bc * v37_data);
              float v166_bc = static_cast<float>(v159_data[3]);
              v157_acc += (v166_bc * v38_data);
              float v168_bc = static_cast<float>(v159_data[4]);
              v157_acc += (v168_bc * v39_data);
              float v170_bc = static_cast<float>(v159_data[5]);
              v157_acc += (v170_bc * v40_data);
              float v172_bc = static_cast<float>(v159_data[6]);
              v157_acc += (v172_bc * v41_data);
              float v174_bc = static_cast<float>(v159_data[7]);
              v157_acc += (v174_bc * v42_data);
              float v176_bc = static_cast<float>(v159_data[8]);
              v157_acc += (v176_bc * v43_data);
              float v178_bc = static_cast<float>(v159_data[9]);
              v157_acc += (v178_bc * v44_data);
              float v180_bc = static_cast<float>(v159_data[10]);
              v157_acc += (v180_bc * v45_data);
              float v182_bc = static_cast<float>(v159_data[11]);
              v157_acc += (v182_bc * v46_data);
              r1.template select<16, 1>(64) = v157_acc;
              tensorforge::intel_esimd::simd<float, 16> v184_acc{};
              tensorforge::intel_esimd::simd<float, 16> v186_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              float v187_bc = static_cast<float>(v186_data[0]);
              v184_acc += (v187_bc * v35_data);
              float v189_bc = static_cast<float>(v186_data[1]);
              v184_acc += (v189_bc * v36_data);
              float v191_bc = static_cast<float>(v186_data[2]);
              v184_acc += (v191_bc * v37_data);
              float v193_bc = static_cast<float>(v186_data[3]);
              v184_acc += (v193_bc * v38_data);
              float v195_bc = static_cast<float>(v186_data[4]);
              v184_acc += (v195_bc * v39_data);
              float v197_bc = static_cast<float>(v186_data[5]);
              v184_acc += (v197_bc * v40_data);
              float v199_bc = static_cast<float>(v186_data[6]);
              v184_acc += (v199_bc * v41_data);
              float v201_bc = static_cast<float>(v186_data[7]);
              v184_acc += (v201_bc * v42_data);
              float v203_bc = static_cast<float>(v186_data[8]);
              v184_acc += (v203_bc * v43_data);
              float v205_bc = static_cast<float>(v186_data[9]);
              v184_acc += (v205_bc * v44_data);
              float v207_bc = static_cast<float>(v186_data[10]);
              v184_acc += (v207_bc * v45_data);
              float v209_bc = static_cast<float>(v186_data[11]);
              v184_acc += (v209_bc * v46_data);
              r1.template select<16, 1>(80) = v184_acc;
              tensorforge::intel_esimd::simd<float, 16> v211_acc{};
              tensorforge::intel_esimd::simd<float, 16> v213_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              float v214_bc = static_cast<float>(v213_data[0]);
              v211_acc += (v214_bc * v35_data);
              float v216_bc = static_cast<float>(v213_data[1]);
              v211_acc += (v216_bc * v36_data);
              float v218_bc = static_cast<float>(v213_data[2]);
              v211_acc += (v218_bc * v37_data);
              float v220_bc = static_cast<float>(v213_data[3]);
              v211_acc += (v220_bc * v38_data);
              float v222_bc = static_cast<float>(v213_data[4]);
              v211_acc += (v222_bc * v39_data);
              float v224_bc = static_cast<float>(v213_data[5]);
              v211_acc += (v224_bc * v40_data);
              float v226_bc = static_cast<float>(v213_data[6]);
              v211_acc += (v226_bc * v41_data);
              float v228_bc = static_cast<float>(v213_data[7]);
              v211_acc += (v228_bc * v42_data);
              float v230_bc = static_cast<float>(v213_data[8]);
              v211_acc += (v230_bc * v43_data);
              float v232_bc = static_cast<float>(v213_data[9]);
              v211_acc += (v232_bc * v44_data);
              float v234_bc = static_cast<float>(v213_data[10]);
              v211_acc += (v234_bc * v45_data);
              float v236_bc = static_cast<float>(v213_data[11]);
              v211_acc += (v236_bc * v46_data);
              r1.template select<16, 1>(96) = v211_acc;
              tensorforge::intel_esimd::simd<float, 16> v238_acc{};
              tensorforge::intel_esimd::simd<float, 16> v240_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              float v241_bc = static_cast<float>(v240_data[0]);
              v238_acc += (v241_bc * v35_data);
              float v243_bc = static_cast<float>(v240_data[1]);
              v238_acc += (v243_bc * v36_data);
              float v245_bc = static_cast<float>(v240_data[2]);
              v238_acc += (v245_bc * v37_data);
              float v247_bc = static_cast<float>(v240_data[3]);
              v238_acc += (v247_bc * v38_data);
              float v249_bc = static_cast<float>(v240_data[4]);
              v238_acc += (v249_bc * v39_data);
              float v251_bc = static_cast<float>(v240_data[5]);
              v238_acc += (v251_bc * v40_data);
              float v253_bc = static_cast<float>(v240_data[6]);
              v238_acc += (v253_bc * v41_data);
              float v255_bc = static_cast<float>(v240_data[7]);
              v238_acc += (v255_bc * v42_data);
              float v257_bc = static_cast<float>(v240_data[8]);
              v238_acc += (v257_bc * v43_data);
              float v259_bc = static_cast<float>(v240_data[9]);
              v238_acc += (v259_bc * v44_data);
              float v261_bc = static_cast<float>(v240_data[10]);
              v238_acc += (v261_bc * v45_data);
              float v263_bc = static_cast<float>(v240_data[11]);
              v238_acc += (v263_bc * v46_data);
              r1.template select<16, 1>(112) = v238_acc;
              tensorforge::intel_esimd::simd<float, 16> v265_acc{};
              tensorforge::intel_esimd::simd<float, 16> v267_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              float v268_bc = static_cast<float>(v267_data[0]);
              v265_acc += (v268_bc * v35_data);
              float v270_bc = static_cast<float>(v267_data[1]);
              v265_acc += (v270_bc * v36_data);
              float v272_bc = static_cast<float>(v267_data[2]);
              v265_acc += (v272_bc * v37_data);
              float v274_bc = static_cast<float>(v267_data[3]);
              v265_acc += (v274_bc * v38_data);
              float v276_bc = static_cast<float>(v267_data[4]);
              v265_acc += (v276_bc * v39_data);
              float v278_bc = static_cast<float>(v267_data[5]);
              v265_acc += (v278_bc * v40_data);
              float v280_bc = static_cast<float>(v267_data[6]);
              v265_acc += (v280_bc * v41_data);
              float v282_bc = static_cast<float>(v267_data[7]);
              v265_acc += (v282_bc * v42_data);
              float v284_bc = static_cast<float>(v267_data[8]);
              v265_acc += (v284_bc * v43_data);
              float v286_bc = static_cast<float>(v267_data[9]);
              v265_acc += (v286_bc * v44_data);
              float v288_bc = static_cast<float>(v267_data[10]);
              v265_acc += (v288_bc * v45_data);
              float v290_bc = static_cast<float>(v267_data[11]);
              v265_acc += (v290_bc * v46_data);
              r1.template select<16, 1>(128) = v265_acc;
              tensorforge::intel_esimd::simd<float, 16> v292_acc{};
              tensorforge::intel_esimd::simd<float, 16> v294_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              float v295_bc = static_cast<float>(v294_data[0]);
              v292_acc += (v295_bc * v35_data);
              float v297_bc = static_cast<float>(v294_data[1]);
              v292_acc += (v297_bc * v36_data);
              float v299_bc = static_cast<float>(v294_data[2]);
              v292_acc += (v299_bc * v37_data);
              float v301_bc = static_cast<float>(v294_data[3]);
              v292_acc += (v301_bc * v38_data);
              float v303_bc = static_cast<float>(v294_data[4]);
              v292_acc += (v303_bc * v39_data);
              float v305_bc = static_cast<float>(v294_data[5]);
              v292_acc += (v305_bc * v40_data);
              float v307_bc = static_cast<float>(v294_data[6]);
              v292_acc += (v307_bc * v41_data);
              float v309_bc = static_cast<float>(v294_data[7]);
              v292_acc += (v309_bc * v42_data);
              float v311_bc = static_cast<float>(v294_data[8]);
              v292_acc += (v311_bc * v43_data);
              float v313_bc = static_cast<float>(v294_data[9]);
              v292_acc += (v313_bc * v44_data);
              float v315_bc = static_cast<float>(v294_data[10]);
              v292_acc += (v315_bc * v45_data);
              float v317_bc = static_cast<float>(v294_data[11]);
              v292_acc += (v317_bc * v46_data);
              r1.template select<16, 1>(144) = v292_acc;
              tensorforge::intel_esimd::simd<float, 16> v319_acc{};
              tensorforge::intel_esimd::simd<float, 16> v321_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              float v322_bc = static_cast<float>(v321_data[0]);
              v319_acc += (v322_bc * v35_data);
              float v324_bc = static_cast<float>(v321_data[1]);
              v319_acc += (v324_bc * v36_data);
              float v326_bc = static_cast<float>(v321_data[2]);
              v319_acc += (v326_bc * v37_data);
              float v328_bc = static_cast<float>(v321_data[3]);
              v319_acc += (v328_bc * v38_data);
              float v330_bc = static_cast<float>(v321_data[4]);
              v319_acc += (v330_bc * v39_data);
              float v332_bc = static_cast<float>(v321_data[5]);
              v319_acc += (v332_bc * v40_data);
              float v334_bc = static_cast<float>(v321_data[6]);
              v319_acc += (v334_bc * v41_data);
              float v336_bc = static_cast<float>(v321_data[7]);
              v319_acc += (v336_bc * v42_data);
              float v338_bc = static_cast<float>(v321_data[8]);
              v319_acc += (v338_bc * v43_data);
              float v340_bc = static_cast<float>(v321_data[9]);
              v319_acc += (v340_bc * v44_data);
              float v342_bc = static_cast<float>(v321_data[10]);
              v319_acc += (v342_bc * v45_data);
              float v344_bc = static_cast<float>(v321_data[11]);
              v319_acc += (v344_bc * v46_data);
              r1.template select<16, 1>(160) = v319_acc;
              tensorforge::intel_esimd::simd<float, 16> v346_acc{};
              tensorforge::intel_esimd::simd<float, 16> v348_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              float v349_bc = static_cast<float>(v348_data[0]);
              v346_acc += (v349_bc * v35_data);
              float v351_bc = static_cast<float>(v348_data[1]);
              v346_acc += (v351_bc * v36_data);
              float v353_bc = static_cast<float>(v348_data[2]);
              v346_acc += (v353_bc * v37_data);
              float v355_bc = static_cast<float>(v348_data[3]);
              v346_acc += (v355_bc * v38_data);
              float v357_bc = static_cast<float>(v348_data[4]);
              v346_acc += (v357_bc * v39_data);
              float v359_bc = static_cast<float>(v348_data[5]);
              v346_acc += (v359_bc * v40_data);
              float v361_bc = static_cast<float>(v348_data[6]);
              v346_acc += (v361_bc * v41_data);
              float v363_bc = static_cast<float>(v348_data[7]);
              v346_acc += (v363_bc * v42_data);
              float v365_bc = static_cast<float>(v348_data[8]);
              v346_acc += (v365_bc * v43_data);
              float v367_bc = static_cast<float>(v348_data[9]);
              v346_acc += (v367_bc * v44_data);
              float v369_bc = static_cast<float>(v348_data[10]);
              v346_acc += (v369_bc * v45_data);
              float v371_bc = static_cast<float>(v348_data[11]);
              v346_acc += (v371_bc * v46_data);
              r1.template select<16, 1>(176) = v346_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v373_i1 = 0; v373_i1 < 12; ++v373_i1) {
                tensorforge::intel_esimd::simd<float, 6> v376_data(r1.template select<6, 1>((v373_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v373_i1 * 12)), v376_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v744_i1 = 0; v744_i1 < 12; ++v744_i1) {
                tensorforge::intel_esimd::simd<float, 12> v749_data;
                v749_data.copy_from(glb_m4 + ((v744_i1 * 12)));
                r4.template select<12, 1>((v744_i1 * 16)) = v749_data;
              }
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s0)
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v392_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v393_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v394_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v395_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v396_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v397_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v398_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v399_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v400_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v401_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v402_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v403_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v404_acc{};
              v404_acc += (v52_bc * v392_data);
              v404_acc += (v54_bc * v393_data);
              v404_acc += (v56_bc * v394_data);
              v404_acc += (v58_bc * v395_data);
              v404_acc += (v60_bc * v396_data);
              v404_acc += (v62_bc * v397_data);
              v404_acc += (v64_bc * v398_data);
              v404_acc += (v66_bc * v399_data);
              v404_acc += (v68_bc * v400_data);
              v404_acc += (v70_bc * v401_data);
              v404_acc += (v72_bc * v402_data);
              v404_acc += (v74_bc * v403_data);
              ir3.template select<16, 1>(0) = v404_acc;
              tensorforge::intel_esimd::simd<float, 16> v433_acc{};
              v433_acc += (v79_bc * v392_data);
              v433_acc += (v81_bc * v393_data);
              v433_acc += (v83_bc * v394_data);
              v433_acc += (v85_bc * v395_data);
              v433_acc += (v87_bc * v396_data);
              v433_acc += (v89_bc * v397_data);
              v433_acc += (v91_bc * v398_data);
              v433_acc += (v93_bc * v399_data);
              v433_acc += (v95_bc * v400_data);
              v433_acc += (v97_bc * v401_data);
              v433_acc += (v99_bc * v402_data);
              v433_acc += (v101_bc * v403_data);
              ir3.template select<16, 1>(16) = v433_acc;
              tensorforge::intel_esimd::simd<float, 16> v460_acc{};
              v460_acc += (v106_bc * v392_data);
              v460_acc += (v108_bc * v393_data);
              v460_acc += (v110_bc * v394_data);
              v460_acc += (v112_bc * v395_data);
              v460_acc += (v114_bc * v396_data);
              v460_acc += (v116_bc * v397_data);
              v460_acc += (v118_bc * v398_data);
              v460_acc += (v120_bc * v399_data);
              v460_acc += (v122_bc * v400_data);
              v460_acc += (v124_bc * v401_data);
              v460_acc += (v126_bc * v402_data);
              v460_acc += (v128_bc * v403_data);
              ir3.template select<16, 1>(32) = v460_acc;
              tensorforge::intel_esimd::simd<float, 16> v487_acc{};
              v487_acc += (v133_bc * v392_data);
              v487_acc += (v135_bc * v393_data);
              v487_acc += (v137_bc * v394_data);
              v487_acc += (v139_bc * v395_data);
              v487_acc += (v141_bc * v396_data);
              v487_acc += (v143_bc * v397_data);
              v487_acc += (v145_bc * v398_data);
              v487_acc += (v147_bc * v399_data);
              v487_acc += (v149_bc * v400_data);
              v487_acc += (v151_bc * v401_data);
              v487_acc += (v153_bc * v402_data);
              v487_acc += (v155_bc * v403_data);
              ir3.template select<16, 1>(48) = v487_acc;
              tensorforge::intel_esimd::simd<float, 16> v514_acc{};
              v514_acc += (v160_bc * v392_data);
              v514_acc += (v162_bc * v393_data);
              v514_acc += (v164_bc * v394_data);
              v514_acc += (v166_bc * v395_data);
              v514_acc += (v168_bc * v396_data);
              v514_acc += (v170_bc * v397_data);
              v514_acc += (v172_bc * v398_data);
              v514_acc += (v174_bc * v399_data);
              v514_acc += (v176_bc * v400_data);
              v514_acc += (v178_bc * v401_data);
              v514_acc += (v180_bc * v402_data);
              v514_acc += (v182_bc * v403_data);
              ir3.template select<16, 1>(64) = v514_acc;
              tensorforge::intel_esimd::simd<float, 16> v541_acc{};
              v541_acc += (v187_bc * v392_data);
              v541_acc += (v189_bc * v393_data);
              v541_acc += (v191_bc * v394_data);
              v541_acc += (v193_bc * v395_data);
              v541_acc += (v195_bc * v396_data);
              v541_acc += (v197_bc * v397_data);
              v541_acc += (v199_bc * v398_data);
              v541_acc += (v201_bc * v399_data);
              v541_acc += (v203_bc * v400_data);
              v541_acc += (v205_bc * v401_data);
              v541_acc += (v207_bc * v402_data);
              v541_acc += (v209_bc * v403_data);
              ir3.template select<16, 1>(80) = v541_acc;
              tensorforge::intel_esimd::simd<float, 16> v568_acc{};
              v568_acc += (v214_bc * v392_data);
              v568_acc += (v216_bc * v393_data);
              v568_acc += (v218_bc * v394_data);
              v568_acc += (v220_bc * v395_data);
              v568_acc += (v222_bc * v396_data);
              v568_acc += (v224_bc * v397_data);
              v568_acc += (v226_bc * v398_data);
              v568_acc += (v228_bc * v399_data);
              v568_acc += (v230_bc * v400_data);
              v568_acc += (v232_bc * v401_data);
              v568_acc += (v234_bc * v402_data);
              v568_acc += (v236_bc * v403_data);
              ir3.template select<16, 1>(96) = v568_acc;
              tensorforge::intel_esimd::simd<float, 16> v595_acc{};
              v595_acc += (v241_bc * v392_data);
              v595_acc += (v243_bc * v393_data);
              v595_acc += (v245_bc * v394_data);
              v595_acc += (v247_bc * v395_data);
              v595_acc += (v249_bc * v396_data);
              v595_acc += (v251_bc * v397_data);
              v595_acc += (v253_bc * v398_data);
              v595_acc += (v255_bc * v399_data);
              v595_acc += (v257_bc * v400_data);
              v595_acc += (v259_bc * v401_data);
              v595_acc += (v261_bc * v402_data);
              v595_acc += (v263_bc * v403_data);
              ir3.template select<16, 1>(112) = v595_acc;
              tensorforge::intel_esimd::simd<float, 16> v622_acc{};
              v622_acc += (v268_bc * v392_data);
              v622_acc += (v270_bc * v393_data);
              v622_acc += (v272_bc * v394_data);
              v622_acc += (v274_bc * v395_data);
              v622_acc += (v276_bc * v396_data);
              v622_acc += (v278_bc * v397_data);
              v622_acc += (v280_bc * v398_data);
              v622_acc += (v282_bc * v399_data);
              v622_acc += (v284_bc * v400_data);
              v622_acc += (v286_bc * v401_data);
              v622_acc += (v288_bc * v402_data);
              v622_acc += (v290_bc * v403_data);
              ir3.template select<16, 1>(128) = v622_acc;
              tensorforge::intel_esimd::simd<float, 16> v649_acc{};
              v649_acc += (v295_bc * v392_data);
              v649_acc += (v297_bc * v393_data);
              v649_acc += (v299_bc * v394_data);
              v649_acc += (v301_bc * v395_data);
              v649_acc += (v303_bc * v396_data);
              v649_acc += (v305_bc * v397_data);
              v649_acc += (v307_bc * v398_data);
              v649_acc += (v309_bc * v399_data);
              v649_acc += (v311_bc * v400_data);
              v649_acc += (v313_bc * v401_data);
              v649_acc += (v315_bc * v402_data);
              v649_acc += (v317_bc * v403_data);
              ir3.template select<16, 1>(144) = v649_acc;
              tensorforge::intel_esimd::simd<float, 16> v676_acc{};
              v676_acc += (v322_bc * v392_data);
              v676_acc += (v324_bc * v393_data);
              v676_acc += (v326_bc * v394_data);
              v676_acc += (v328_bc * v395_data);
              v676_acc += (v330_bc * v396_data);
              v676_acc += (v332_bc * v397_data);
              v676_acc += (v334_bc * v398_data);
              v676_acc += (v336_bc * v399_data);
              v676_acc += (v338_bc * v400_data);
              v676_acc += (v340_bc * v401_data);
              v676_acc += (v342_bc * v402_data);
              v676_acc += (v344_bc * v403_data);
              ir3.template select<16, 1>(160) = v676_acc;
              tensorforge::intel_esimd::simd<float, 16> v703_acc{};
              v703_acc += (v349_bc * v392_data);
              v703_acc += (v351_bc * v393_data);
              v703_acc += (v353_bc * v394_data);
              v703_acc += (v355_bc * v395_data);
              v703_acc += (v357_bc * v396_data);
              v703_acc += (v359_bc * v397_data);
              v703_acc += (v361_bc * v398_data);
              v703_acc += (v363_bc * v399_data);
              v703_acc += (v365_bc * v400_data);
              v703_acc += (v367_bc * v401_data);
              v703_acc += (v369_bc * v402_data);
              v703_acc += (v371_bc * v403_data);
              ir3.template select<16, 1>(176) = v703_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v730_n1 = 0; v730_n1 < 12; ++v730_n1) {
                int32_t v731_a = v730_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v733_data(ir3.template select<6, 1>(v731_a));
                r3.template select<6, 1>(v731_a) = v733_data;
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v734_i1 = 0; v734_i1 < 12; ++v734_i1) {
                tensorforge::intel_esimd::simd<float, 6> v737_data(r3.template select<6, 1>((v734_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v734_i1 * 12))), v737_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // ir5 = +(r4 * s1)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v754_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v755_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v756_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v757_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v758_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v759_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v760_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v761_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v762_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v763_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v764_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v765_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v766_acc{};
              tensorforge::intel_esimd::simd<float, 16> v770_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v766_acc += ((static_cast<float>(v770_data[0])) * v754_data);
              v766_acc += ((static_cast<float>(v770_data[1])) * v755_data);
              v766_acc += ((static_cast<float>(v770_data[2])) * v756_data);
              v766_acc += ((static_cast<float>(v770_data[3])) * v757_data);
              v766_acc += ((static_cast<float>(v770_data[4])) * v758_data);
              v766_acc += ((static_cast<float>(v770_data[5])) * v759_data);
              v766_acc += ((static_cast<float>(v770_data[6])) * v760_data);
              v766_acc += ((static_cast<float>(v770_data[7])) * v761_data);
              v766_acc += ((static_cast<float>(v770_data[8])) * v762_data);
              v766_acc += ((static_cast<float>(v770_data[9])) * v763_data);
              v766_acc += ((static_cast<float>(v770_data[10])) * v764_data);
              v766_acc += ((static_cast<float>(v770_data[11])) * v765_data);
              ir5.template select<16, 1>(0) = v766_acc;
              tensorforge::intel_esimd::simd<float, 16> v795_acc{};
              tensorforge::intel_esimd::simd<float, 16> v797_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v795_acc += ((static_cast<float>(v797_data[0])) * v754_data);
              v795_acc += ((static_cast<float>(v797_data[1])) * v755_data);
              v795_acc += ((static_cast<float>(v797_data[2])) * v756_data);
              v795_acc += ((static_cast<float>(v797_data[3])) * v757_data);
              v795_acc += ((static_cast<float>(v797_data[4])) * v758_data);
              v795_acc += ((static_cast<float>(v797_data[5])) * v759_data);
              v795_acc += ((static_cast<float>(v797_data[6])) * v760_data);
              v795_acc += ((static_cast<float>(v797_data[7])) * v761_data);
              v795_acc += ((static_cast<float>(v797_data[8])) * v762_data);
              v795_acc += ((static_cast<float>(v797_data[9])) * v763_data);
              v795_acc += ((static_cast<float>(v797_data[10])) * v764_data);
              v795_acc += ((static_cast<float>(v797_data[11])) * v765_data);
              ir5.template select<16, 1>(16) = v795_acc;
              tensorforge::intel_esimd::simd<float, 16> v822_acc{};
              tensorforge::intel_esimd::simd<float, 16> v824_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v822_acc += ((static_cast<float>(v824_data[0])) * v754_data);
              v822_acc += ((static_cast<float>(v824_data[1])) * v755_data);
              v822_acc += ((static_cast<float>(v824_data[2])) * v756_data);
              v822_acc += ((static_cast<float>(v824_data[3])) * v757_data);
              v822_acc += ((static_cast<float>(v824_data[4])) * v758_data);
              v822_acc += ((static_cast<float>(v824_data[5])) * v759_data);
              v822_acc += ((static_cast<float>(v824_data[6])) * v760_data);
              v822_acc += ((static_cast<float>(v824_data[7])) * v761_data);
              v822_acc += ((static_cast<float>(v824_data[8])) * v762_data);
              v822_acc += ((static_cast<float>(v824_data[9])) * v763_data);
              v822_acc += ((static_cast<float>(v824_data[10])) * v764_data);
              v822_acc += ((static_cast<float>(v824_data[11])) * v765_data);
              ir5.template select<16, 1>(32) = v822_acc;
              tensorforge::intel_esimd::simd<float, 16> v849_acc{};
              tensorforge::intel_esimd::simd<float, 16> v851_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v849_acc += ((static_cast<float>(v851_data[0])) * v754_data);
              v849_acc += ((static_cast<float>(v851_data[1])) * v755_data);
              v849_acc += ((static_cast<float>(v851_data[2])) * v756_data);
              v849_acc += ((static_cast<float>(v851_data[3])) * v757_data);
              v849_acc += ((static_cast<float>(v851_data[4])) * v758_data);
              v849_acc += ((static_cast<float>(v851_data[5])) * v759_data);
              v849_acc += ((static_cast<float>(v851_data[6])) * v760_data);
              v849_acc += ((static_cast<float>(v851_data[7])) * v761_data);
              v849_acc += ((static_cast<float>(v851_data[8])) * v762_data);
              v849_acc += ((static_cast<float>(v851_data[9])) * v763_data);
              v849_acc += ((static_cast<float>(v851_data[10])) * v764_data);
              v849_acc += ((static_cast<float>(v851_data[11])) * v765_data);
              ir5.template select<16, 1>(48) = v849_acc;
              tensorforge::intel_esimd::simd<float, 16> v876_acc{};
              tensorforge::intel_esimd::simd<float, 16> v878_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v876_acc += ((static_cast<float>(v878_data[0])) * v754_data);
              v876_acc += ((static_cast<float>(v878_data[1])) * v755_data);
              v876_acc += ((static_cast<float>(v878_data[2])) * v756_data);
              v876_acc += ((static_cast<float>(v878_data[3])) * v757_data);
              v876_acc += ((static_cast<float>(v878_data[4])) * v758_data);
              v876_acc += ((static_cast<float>(v878_data[5])) * v759_data);
              v876_acc += ((static_cast<float>(v878_data[6])) * v760_data);
              v876_acc += ((static_cast<float>(v878_data[7])) * v761_data);
              v876_acc += ((static_cast<float>(v878_data[8])) * v762_data);
              v876_acc += ((static_cast<float>(v878_data[9])) * v763_data);
              v876_acc += ((static_cast<float>(v878_data[10])) * v764_data);
              v876_acc += ((static_cast<float>(v878_data[11])) * v765_data);
              ir5.template select<16, 1>(64) = v876_acc;
              tensorforge::intel_esimd::simd<float, 16> v903_acc{};
              tensorforge::intel_esimd::simd<float, 16> v905_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v903_acc += ((static_cast<float>(v905_data[0])) * v754_data);
              v903_acc += ((static_cast<float>(v905_data[1])) * v755_data);
              v903_acc += ((static_cast<float>(v905_data[2])) * v756_data);
              v903_acc += ((static_cast<float>(v905_data[3])) * v757_data);
              v903_acc += ((static_cast<float>(v905_data[4])) * v758_data);
              v903_acc += ((static_cast<float>(v905_data[5])) * v759_data);
              v903_acc += ((static_cast<float>(v905_data[6])) * v760_data);
              v903_acc += ((static_cast<float>(v905_data[7])) * v761_data);
              v903_acc += ((static_cast<float>(v905_data[8])) * v762_data);
              v903_acc += ((static_cast<float>(v905_data[9])) * v763_data);
              v903_acc += ((static_cast<float>(v905_data[10])) * v764_data);
              v903_acc += ((static_cast<float>(v905_data[11])) * v765_data);
              ir5.template select<16, 1>(80) = v903_acc;
              tensorforge::intel_esimd::simd<float, 16> v930_acc{};
              tensorforge::intel_esimd::simd<float, 16> v932_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v930_acc += ((static_cast<float>(v932_data[0])) * v754_data);
              v930_acc += ((static_cast<float>(v932_data[1])) * v755_data);
              v930_acc += ((static_cast<float>(v932_data[2])) * v756_data);
              v930_acc += ((static_cast<float>(v932_data[3])) * v757_data);
              v930_acc += ((static_cast<float>(v932_data[4])) * v758_data);
              v930_acc += ((static_cast<float>(v932_data[5])) * v759_data);
              v930_acc += ((static_cast<float>(v932_data[6])) * v760_data);
              v930_acc += ((static_cast<float>(v932_data[7])) * v761_data);
              v930_acc += ((static_cast<float>(v932_data[8])) * v762_data);
              v930_acc += ((static_cast<float>(v932_data[9])) * v763_data);
              v930_acc += ((static_cast<float>(v932_data[10])) * v764_data);
              v930_acc += ((static_cast<float>(v932_data[11])) * v765_data);
              ir5.template select<16, 1>(96) = v930_acc;
              tensorforge::intel_esimd::simd<float, 16> v957_acc{};
              tensorforge::intel_esimd::simd<float, 16> v959_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v957_acc += ((static_cast<float>(v959_data[0])) * v754_data);
              v957_acc += ((static_cast<float>(v959_data[1])) * v755_data);
              v957_acc += ((static_cast<float>(v959_data[2])) * v756_data);
              v957_acc += ((static_cast<float>(v959_data[3])) * v757_data);
              v957_acc += ((static_cast<float>(v959_data[4])) * v758_data);
              v957_acc += ((static_cast<float>(v959_data[5])) * v759_data);
              v957_acc += ((static_cast<float>(v959_data[6])) * v760_data);
              v957_acc += ((static_cast<float>(v959_data[7])) * v761_data);
              v957_acc += ((static_cast<float>(v959_data[8])) * v762_data);
              v957_acc += ((static_cast<float>(v959_data[9])) * v763_data);
              v957_acc += ((static_cast<float>(v959_data[10])) * v764_data);
              v957_acc += ((static_cast<float>(v959_data[11])) * v765_data);
              ir5.template select<16, 1>(112) = v957_acc;
              tensorforge::intel_esimd::simd<float, 16> v984_acc{};
              tensorforge::intel_esimd::simd<float, 16> v986_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v984_acc += ((static_cast<float>(v986_data[0])) * v754_data);
              v984_acc += ((static_cast<float>(v986_data[1])) * v755_data);
              v984_acc += ((static_cast<float>(v986_data[2])) * v756_data);
              v984_acc += ((static_cast<float>(v986_data[3])) * v757_data);
              v984_acc += ((static_cast<float>(v986_data[4])) * v758_data);
              v984_acc += ((static_cast<float>(v986_data[5])) * v759_data);
              v984_acc += ((static_cast<float>(v986_data[6])) * v760_data);
              v984_acc += ((static_cast<float>(v986_data[7])) * v761_data);
              v984_acc += ((static_cast<float>(v986_data[8])) * v762_data);
              v984_acc += ((static_cast<float>(v986_data[9])) * v763_data);
              v984_acc += ((static_cast<float>(v986_data[10])) * v764_data);
              v984_acc += ((static_cast<float>(v986_data[11])) * v765_data);
              ir5.template select<16, 1>(128) = v984_acc;
              tensorforge::intel_esimd::simd<float, 16> v1011_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1013_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v1011_acc += ((static_cast<float>(v1013_data[0])) * v754_data);
              v1011_acc += ((static_cast<float>(v1013_data[1])) * v755_data);
              v1011_acc += ((static_cast<float>(v1013_data[2])) * v756_data);
              v1011_acc += ((static_cast<float>(v1013_data[3])) * v757_data);
              v1011_acc += ((static_cast<float>(v1013_data[4])) * v758_data);
              v1011_acc += ((static_cast<float>(v1013_data[5])) * v759_data);
              v1011_acc += ((static_cast<float>(v1013_data[6])) * v760_data);
              v1011_acc += ((static_cast<float>(v1013_data[7])) * v761_data);
              v1011_acc += ((static_cast<float>(v1013_data[8])) * v762_data);
              v1011_acc += ((static_cast<float>(v1013_data[9])) * v763_data);
              v1011_acc += ((static_cast<float>(v1013_data[10])) * v764_data);
              v1011_acc += ((static_cast<float>(v1013_data[11])) * v765_data);
              ir5.template select<16, 1>(144) = v1011_acc;
              tensorforge::intel_esimd::simd<float, 16> v1038_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1040_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v1038_acc += ((static_cast<float>(v1040_data[0])) * v754_data);
              v1038_acc += ((static_cast<float>(v1040_data[1])) * v755_data);
              v1038_acc += ((static_cast<float>(v1040_data[2])) * v756_data);
              v1038_acc += ((static_cast<float>(v1040_data[3])) * v757_data);
              v1038_acc += ((static_cast<float>(v1040_data[4])) * v758_data);
              v1038_acc += ((static_cast<float>(v1040_data[5])) * v759_data);
              v1038_acc += ((static_cast<float>(v1040_data[6])) * v760_data);
              v1038_acc += ((static_cast<float>(v1040_data[7])) * v761_data);
              v1038_acc += ((static_cast<float>(v1040_data[8])) * v762_data);
              v1038_acc += ((static_cast<float>(v1040_data[9])) * v763_data);
              v1038_acc += ((static_cast<float>(v1040_data[10])) * v764_data);
              v1038_acc += ((static_cast<float>(v1040_data[11])) * v765_data);
              ir5.template select<16, 1>(160) = v1038_acc;
              tensorforge::intel_esimd::simd<float, 16> v1065_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1067_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v1065_acc += ((static_cast<float>(v1067_data[0])) * v754_data);
              v1065_acc += ((static_cast<float>(v1067_data[1])) * v755_data);
              v1065_acc += ((static_cast<float>(v1067_data[2])) * v756_data);
              v1065_acc += ((static_cast<float>(v1067_data[3])) * v757_data);
              v1065_acc += ((static_cast<float>(v1067_data[4])) * v758_data);
              v1065_acc += ((static_cast<float>(v1067_data[5])) * v759_data);
              v1065_acc += ((static_cast<float>(v1067_data[6])) * v760_data);
              v1065_acc += ((static_cast<float>(v1067_data[7])) * v761_data);
              v1065_acc += ((static_cast<float>(v1067_data[8])) * v762_data);
              v1065_acc += ((static_cast<float>(v1067_data[9])) * v763_data);
              v1065_acc += ((static_cast<float>(v1067_data[10])) * v764_data);
              v1065_acc += ((static_cast<float>(v1067_data[11])) * v765_data);
              ir5.template select<16, 1>(176) = v1065_acc;
              // r5 = ir5
              #pragma unroll
              for (int32_t v1092_n1 = 0; v1092_n1 < 12; ++v1092_n1) {
                int32_t v1093_a = v1092_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1095_data(ir5.template select<12, 1>(v1093_a));
                r5.template select<12, 1>(v1093_a) = v1095_data;
              }
              // glb_m3 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1096_i1 = 0; v1096_i1 < 12; ++v1096_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1099_data(r5.template select<12, 1>((v1096_i1 * 16)));
                v1099_data.copy_to(glb_m3 + ((v1096_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


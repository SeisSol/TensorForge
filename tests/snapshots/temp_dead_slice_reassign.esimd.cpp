// === base name ===
kernel_e8ce6742ac020bc4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e8ce6742ac020bc4 = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e8ce6742ac020bc4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e8ce6742ac020bc4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e8ce6742ac020bc4(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_e8ce6742ac020bc4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e8ce6742ac020bc4(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e8ce6742ac020bc4(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e8ce6742ac020bc4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4864 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 19456 B shared, occupancy grid
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4864}],"shared_bytes":19456,"shared_elements":4864,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (304 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (288);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (144);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 72 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 144 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v20_i1 = 0; v20_i1 < 12; ++v20_i1) {
                tensorforge::intel_esimd::simd<float, 6> v25_data;
                v25_data.copy_from(glb_m0 + ((v20_i1 * 6)));
                r0.template select<6, 1>((v20_i1 * 16)) = v25_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v28_ld;
              v28_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v28_ld);
              tensorforge::intel_esimd::simd<float, 64> v29_ld;
              v29_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v29_ld);
              tensorforge::intel_esimd::simd<float, 16> v30_ld;
              v30_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v30_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
                tensorforge::intel_esimd::simd<float, 6> v37_data;
                v37_data.copy_from(glb_m2 + ((v32_i1 * 6)));
                r2.template select<6, 1>((v32_i1 * 16)) = v37_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v53_acc{};
              tensorforge::intel_esimd::simd<float, 16> v57_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              float v58_bc = static_cast<float>(v57_data[0]);
              v53_acc += (v58_bc * v41_data);
              float v60_bc = static_cast<float>(v57_data[1]);
              v53_acc += (v60_bc * v42_data);
              float v62_bc = static_cast<float>(v57_data[2]);
              v53_acc += (v62_bc * v43_data);
              float v64_bc = static_cast<float>(v57_data[3]);
              v53_acc += (v64_bc * v44_data);
              float v66_bc = static_cast<float>(v57_data[4]);
              v53_acc += (v66_bc * v45_data);
              float v68_bc = static_cast<float>(v57_data[5]);
              v53_acc += (v68_bc * v46_data);
              float v70_bc = static_cast<float>(v57_data[6]);
              v53_acc += (v70_bc * v47_data);
              float v72_bc = static_cast<float>(v57_data[7]);
              v53_acc += (v72_bc * v48_data);
              float v74_bc = static_cast<float>(v57_data[8]);
              v53_acc += (v74_bc * v49_data);
              float v76_bc = static_cast<float>(v57_data[9]);
              v53_acc += (v76_bc * v50_data);
              float v78_bc = static_cast<float>(v57_data[10]);
              v53_acc += (v78_bc * v51_data);
              float v80_bc = static_cast<float>(v57_data[11]);
              v53_acc += (v80_bc * v52_data);
              r1.template select<16, 1>(0) = v53_acc;
              tensorforge::intel_esimd::simd<float, 16> v82_acc{};
              tensorforge::intel_esimd::simd<float, 16> v84_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              float v85_bc = static_cast<float>(v84_data[0]);
              v82_acc += (v85_bc * v41_data);
              float v87_bc = static_cast<float>(v84_data[1]);
              v82_acc += (v87_bc * v42_data);
              float v89_bc = static_cast<float>(v84_data[2]);
              v82_acc += (v89_bc * v43_data);
              float v91_bc = static_cast<float>(v84_data[3]);
              v82_acc += (v91_bc * v44_data);
              float v93_bc = static_cast<float>(v84_data[4]);
              v82_acc += (v93_bc * v45_data);
              float v95_bc = static_cast<float>(v84_data[5]);
              v82_acc += (v95_bc * v46_data);
              float v97_bc = static_cast<float>(v84_data[6]);
              v82_acc += (v97_bc * v47_data);
              float v99_bc = static_cast<float>(v84_data[7]);
              v82_acc += (v99_bc * v48_data);
              float v101_bc = static_cast<float>(v84_data[8]);
              v82_acc += (v101_bc * v49_data);
              float v103_bc = static_cast<float>(v84_data[9]);
              v82_acc += (v103_bc * v50_data);
              float v105_bc = static_cast<float>(v84_data[10]);
              v82_acc += (v105_bc * v51_data);
              float v107_bc = static_cast<float>(v84_data[11]);
              v82_acc += (v107_bc * v52_data);
              r1.template select<16, 1>(16) = v82_acc;
              tensorforge::intel_esimd::simd<float, 16> v109_acc{};
              tensorforge::intel_esimd::simd<float, 16> v111_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              float v112_bc = static_cast<float>(v111_data[0]);
              v109_acc += (v112_bc * v41_data);
              float v114_bc = static_cast<float>(v111_data[1]);
              v109_acc += (v114_bc * v42_data);
              float v116_bc = static_cast<float>(v111_data[2]);
              v109_acc += (v116_bc * v43_data);
              float v118_bc = static_cast<float>(v111_data[3]);
              v109_acc += (v118_bc * v44_data);
              float v120_bc = static_cast<float>(v111_data[4]);
              v109_acc += (v120_bc * v45_data);
              float v122_bc = static_cast<float>(v111_data[5]);
              v109_acc += (v122_bc * v46_data);
              float v124_bc = static_cast<float>(v111_data[6]);
              v109_acc += (v124_bc * v47_data);
              float v126_bc = static_cast<float>(v111_data[7]);
              v109_acc += (v126_bc * v48_data);
              float v128_bc = static_cast<float>(v111_data[8]);
              v109_acc += (v128_bc * v49_data);
              float v130_bc = static_cast<float>(v111_data[9]);
              v109_acc += (v130_bc * v50_data);
              float v132_bc = static_cast<float>(v111_data[10]);
              v109_acc += (v132_bc * v51_data);
              float v134_bc = static_cast<float>(v111_data[11]);
              v109_acc += (v134_bc * v52_data);
              r1.template select<16, 1>(32) = v109_acc;
              tensorforge::intel_esimd::simd<float, 16> v136_acc{};
              tensorforge::intel_esimd::simd<float, 16> v138_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              float v139_bc = static_cast<float>(v138_data[0]);
              v136_acc += (v139_bc * v41_data);
              float v141_bc = static_cast<float>(v138_data[1]);
              v136_acc += (v141_bc * v42_data);
              float v143_bc = static_cast<float>(v138_data[2]);
              v136_acc += (v143_bc * v43_data);
              float v145_bc = static_cast<float>(v138_data[3]);
              v136_acc += (v145_bc * v44_data);
              float v147_bc = static_cast<float>(v138_data[4]);
              v136_acc += (v147_bc * v45_data);
              float v149_bc = static_cast<float>(v138_data[5]);
              v136_acc += (v149_bc * v46_data);
              float v151_bc = static_cast<float>(v138_data[6]);
              v136_acc += (v151_bc * v47_data);
              float v153_bc = static_cast<float>(v138_data[7]);
              v136_acc += (v153_bc * v48_data);
              float v155_bc = static_cast<float>(v138_data[8]);
              v136_acc += (v155_bc * v49_data);
              float v157_bc = static_cast<float>(v138_data[9]);
              v136_acc += (v157_bc * v50_data);
              float v159_bc = static_cast<float>(v138_data[10]);
              v136_acc += (v159_bc * v51_data);
              float v161_bc = static_cast<float>(v138_data[11]);
              v136_acc += (v161_bc * v52_data);
              r1.template select<16, 1>(48) = v136_acc;
              tensorforge::intel_esimd::simd<float, 16> v163_acc{};
              tensorforge::intel_esimd::simd<float, 16> v165_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              float v166_bc = static_cast<float>(v165_data[0]);
              v163_acc += (v166_bc * v41_data);
              float v168_bc = static_cast<float>(v165_data[1]);
              v163_acc += (v168_bc * v42_data);
              float v170_bc = static_cast<float>(v165_data[2]);
              v163_acc += (v170_bc * v43_data);
              float v172_bc = static_cast<float>(v165_data[3]);
              v163_acc += (v172_bc * v44_data);
              float v174_bc = static_cast<float>(v165_data[4]);
              v163_acc += (v174_bc * v45_data);
              float v176_bc = static_cast<float>(v165_data[5]);
              v163_acc += (v176_bc * v46_data);
              float v178_bc = static_cast<float>(v165_data[6]);
              v163_acc += (v178_bc * v47_data);
              float v180_bc = static_cast<float>(v165_data[7]);
              v163_acc += (v180_bc * v48_data);
              float v182_bc = static_cast<float>(v165_data[8]);
              v163_acc += (v182_bc * v49_data);
              float v184_bc = static_cast<float>(v165_data[9]);
              v163_acc += (v184_bc * v50_data);
              float v186_bc = static_cast<float>(v165_data[10]);
              v163_acc += (v186_bc * v51_data);
              float v188_bc = static_cast<float>(v165_data[11]);
              v163_acc += (v188_bc * v52_data);
              r1.template select<16, 1>(64) = v163_acc;
              tensorforge::intel_esimd::simd<float, 16> v190_acc{};
              tensorforge::intel_esimd::simd<float, 16> v192_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              float v193_bc = static_cast<float>(v192_data[0]);
              v190_acc += (v193_bc * v41_data);
              float v195_bc = static_cast<float>(v192_data[1]);
              v190_acc += (v195_bc * v42_data);
              float v197_bc = static_cast<float>(v192_data[2]);
              v190_acc += (v197_bc * v43_data);
              float v199_bc = static_cast<float>(v192_data[3]);
              v190_acc += (v199_bc * v44_data);
              float v201_bc = static_cast<float>(v192_data[4]);
              v190_acc += (v201_bc * v45_data);
              float v203_bc = static_cast<float>(v192_data[5]);
              v190_acc += (v203_bc * v46_data);
              float v205_bc = static_cast<float>(v192_data[6]);
              v190_acc += (v205_bc * v47_data);
              float v207_bc = static_cast<float>(v192_data[7]);
              v190_acc += (v207_bc * v48_data);
              float v209_bc = static_cast<float>(v192_data[8]);
              v190_acc += (v209_bc * v49_data);
              float v211_bc = static_cast<float>(v192_data[9]);
              v190_acc += (v211_bc * v50_data);
              float v213_bc = static_cast<float>(v192_data[10]);
              v190_acc += (v213_bc * v51_data);
              float v215_bc = static_cast<float>(v192_data[11]);
              v190_acc += (v215_bc * v52_data);
              r1.template select<16, 1>(80) = v190_acc;
              tensorforge::intel_esimd::simd<float, 16> v217_acc{};
              tensorforge::intel_esimd::simd<float, 16> v219_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              float v220_bc = static_cast<float>(v219_data[0]);
              v217_acc += (v220_bc * v41_data);
              float v222_bc = static_cast<float>(v219_data[1]);
              v217_acc += (v222_bc * v42_data);
              float v224_bc = static_cast<float>(v219_data[2]);
              v217_acc += (v224_bc * v43_data);
              float v226_bc = static_cast<float>(v219_data[3]);
              v217_acc += (v226_bc * v44_data);
              float v228_bc = static_cast<float>(v219_data[4]);
              v217_acc += (v228_bc * v45_data);
              float v230_bc = static_cast<float>(v219_data[5]);
              v217_acc += (v230_bc * v46_data);
              float v232_bc = static_cast<float>(v219_data[6]);
              v217_acc += (v232_bc * v47_data);
              float v234_bc = static_cast<float>(v219_data[7]);
              v217_acc += (v234_bc * v48_data);
              float v236_bc = static_cast<float>(v219_data[8]);
              v217_acc += (v236_bc * v49_data);
              float v238_bc = static_cast<float>(v219_data[9]);
              v217_acc += (v238_bc * v50_data);
              float v240_bc = static_cast<float>(v219_data[10]);
              v217_acc += (v240_bc * v51_data);
              float v242_bc = static_cast<float>(v219_data[11]);
              v217_acc += (v242_bc * v52_data);
              r1.template select<16, 1>(96) = v217_acc;
              tensorforge::intel_esimd::simd<float, 16> v244_acc{};
              tensorforge::intel_esimd::simd<float, 16> v246_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              float v247_bc = static_cast<float>(v246_data[0]);
              v244_acc += (v247_bc * v41_data);
              float v249_bc = static_cast<float>(v246_data[1]);
              v244_acc += (v249_bc * v42_data);
              float v251_bc = static_cast<float>(v246_data[2]);
              v244_acc += (v251_bc * v43_data);
              float v253_bc = static_cast<float>(v246_data[3]);
              v244_acc += (v253_bc * v44_data);
              float v255_bc = static_cast<float>(v246_data[4]);
              v244_acc += (v255_bc * v45_data);
              float v257_bc = static_cast<float>(v246_data[5]);
              v244_acc += (v257_bc * v46_data);
              float v259_bc = static_cast<float>(v246_data[6]);
              v244_acc += (v259_bc * v47_data);
              float v261_bc = static_cast<float>(v246_data[7]);
              v244_acc += (v261_bc * v48_data);
              float v263_bc = static_cast<float>(v246_data[8]);
              v244_acc += (v263_bc * v49_data);
              float v265_bc = static_cast<float>(v246_data[9]);
              v244_acc += (v265_bc * v50_data);
              float v267_bc = static_cast<float>(v246_data[10]);
              v244_acc += (v267_bc * v51_data);
              float v269_bc = static_cast<float>(v246_data[11]);
              v244_acc += (v269_bc * v52_data);
              r1.template select<16, 1>(112) = v244_acc;
              tensorforge::intel_esimd::simd<float, 16> v271_acc{};
              tensorforge::intel_esimd::simd<float, 16> v273_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              float v274_bc = static_cast<float>(v273_data[0]);
              v271_acc += (v274_bc * v41_data);
              float v276_bc = static_cast<float>(v273_data[1]);
              v271_acc += (v276_bc * v42_data);
              float v278_bc = static_cast<float>(v273_data[2]);
              v271_acc += (v278_bc * v43_data);
              float v280_bc = static_cast<float>(v273_data[3]);
              v271_acc += (v280_bc * v44_data);
              float v282_bc = static_cast<float>(v273_data[4]);
              v271_acc += (v282_bc * v45_data);
              float v284_bc = static_cast<float>(v273_data[5]);
              v271_acc += (v284_bc * v46_data);
              float v286_bc = static_cast<float>(v273_data[6]);
              v271_acc += (v286_bc * v47_data);
              float v288_bc = static_cast<float>(v273_data[7]);
              v271_acc += (v288_bc * v48_data);
              float v290_bc = static_cast<float>(v273_data[8]);
              v271_acc += (v290_bc * v49_data);
              float v292_bc = static_cast<float>(v273_data[9]);
              v271_acc += (v292_bc * v50_data);
              float v294_bc = static_cast<float>(v273_data[10]);
              v271_acc += (v294_bc * v51_data);
              float v296_bc = static_cast<float>(v273_data[11]);
              v271_acc += (v296_bc * v52_data);
              r1.template select<16, 1>(128) = v271_acc;
              tensorforge::intel_esimd::simd<float, 16> v298_acc{};
              tensorforge::intel_esimd::simd<float, 16> v300_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              float v301_bc = static_cast<float>(v300_data[0]);
              v298_acc += (v301_bc * v41_data);
              float v303_bc = static_cast<float>(v300_data[1]);
              v298_acc += (v303_bc * v42_data);
              float v305_bc = static_cast<float>(v300_data[2]);
              v298_acc += (v305_bc * v43_data);
              float v307_bc = static_cast<float>(v300_data[3]);
              v298_acc += (v307_bc * v44_data);
              float v309_bc = static_cast<float>(v300_data[4]);
              v298_acc += (v309_bc * v45_data);
              float v311_bc = static_cast<float>(v300_data[5]);
              v298_acc += (v311_bc * v46_data);
              float v313_bc = static_cast<float>(v300_data[6]);
              v298_acc += (v313_bc * v47_data);
              float v315_bc = static_cast<float>(v300_data[7]);
              v298_acc += (v315_bc * v48_data);
              float v317_bc = static_cast<float>(v300_data[8]);
              v298_acc += (v317_bc * v49_data);
              float v319_bc = static_cast<float>(v300_data[9]);
              v298_acc += (v319_bc * v50_data);
              float v321_bc = static_cast<float>(v300_data[10]);
              v298_acc += (v321_bc * v51_data);
              float v323_bc = static_cast<float>(v300_data[11]);
              v298_acc += (v323_bc * v52_data);
              r1.template select<16, 1>(144) = v298_acc;
              tensorforge::intel_esimd::simd<float, 16> v325_acc{};
              tensorforge::intel_esimd::simd<float, 16> v327_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              float v328_bc = static_cast<float>(v327_data[0]);
              v325_acc += (v328_bc * v41_data);
              float v330_bc = static_cast<float>(v327_data[1]);
              v325_acc += (v330_bc * v42_data);
              float v332_bc = static_cast<float>(v327_data[2]);
              v325_acc += (v332_bc * v43_data);
              float v334_bc = static_cast<float>(v327_data[3]);
              v325_acc += (v334_bc * v44_data);
              float v336_bc = static_cast<float>(v327_data[4]);
              v325_acc += (v336_bc * v45_data);
              float v338_bc = static_cast<float>(v327_data[5]);
              v325_acc += (v338_bc * v46_data);
              float v340_bc = static_cast<float>(v327_data[6]);
              v325_acc += (v340_bc * v47_data);
              float v342_bc = static_cast<float>(v327_data[7]);
              v325_acc += (v342_bc * v48_data);
              float v344_bc = static_cast<float>(v327_data[8]);
              v325_acc += (v344_bc * v49_data);
              float v346_bc = static_cast<float>(v327_data[9]);
              v325_acc += (v346_bc * v50_data);
              float v348_bc = static_cast<float>(v327_data[10]);
              v325_acc += (v348_bc * v51_data);
              float v350_bc = static_cast<float>(v327_data[11]);
              v325_acc += (v350_bc * v52_data);
              r1.template select<16, 1>(160) = v325_acc;
              tensorforge::intel_esimd::simd<float, 16> v352_acc{};
              tensorforge::intel_esimd::simd<float, 16> v354_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              float v355_bc = static_cast<float>(v354_data[0]);
              v352_acc += (v355_bc * v41_data);
              float v357_bc = static_cast<float>(v354_data[1]);
              v352_acc += (v357_bc * v42_data);
              float v359_bc = static_cast<float>(v354_data[2]);
              v352_acc += (v359_bc * v43_data);
              float v361_bc = static_cast<float>(v354_data[3]);
              v352_acc += (v361_bc * v44_data);
              float v363_bc = static_cast<float>(v354_data[4]);
              v352_acc += (v363_bc * v45_data);
              float v365_bc = static_cast<float>(v354_data[5]);
              v352_acc += (v365_bc * v46_data);
              float v367_bc = static_cast<float>(v354_data[6]);
              v352_acc += (v367_bc * v47_data);
              float v369_bc = static_cast<float>(v354_data[7]);
              v352_acc += (v369_bc * v48_data);
              float v371_bc = static_cast<float>(v354_data[8]);
              v352_acc += (v371_bc * v49_data);
              float v373_bc = static_cast<float>(v354_data[9]);
              v352_acc += (v373_bc * v50_data);
              float v375_bc = static_cast<float>(v354_data[10]);
              v352_acc += (v375_bc * v51_data);
              float v377_bc = static_cast<float>(v354_data[11]);
              v352_acc += (v377_bc * v52_data);
              r1.template select<16, 1>(176) = v352_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v379_i1 = 0; v379_i1 < 12; ++v379_i1) {
                tensorforge::intel_esimd::simd<float, 6> v382_data(r1.template select<6, 1>((v379_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v379_i1 * 12))), v382_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v389_i1 = 0; v389_i1 < 12; ++v389_i1) {
                tensorforge::intel_esimd::simd<float, 6> v394_data;
                v394_data.copy_from(glb_m3 + ((v389_i1 * 6)));
                r4.template select<6, 1>((v389_i1 * 16)) = v394_data;
              }
              // wait(r2 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s0)
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v399_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v400_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v401_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v402_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v403_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v404_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v405_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v406_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v407_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v408_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v409_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v410_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v411_acc{};
              v411_acc += (v58_bc * v399_data);
              v411_acc += (v60_bc * v400_data);
              v411_acc += (v62_bc * v401_data);
              v411_acc += (v64_bc * v402_data);
              v411_acc += (v66_bc * v403_data);
              v411_acc += (v68_bc * v404_data);
              v411_acc += (v70_bc * v405_data);
              v411_acc += (v72_bc * v406_data);
              v411_acc += (v74_bc * v407_data);
              v411_acc += (v76_bc * v408_data);
              v411_acc += (v78_bc * v409_data);
              v411_acc += (v80_bc * v410_data);
              ir3.template select<16, 1>(0) = v411_acc;
              tensorforge::intel_esimd::simd<float, 16> v440_acc{};
              v440_acc += (v85_bc * v399_data);
              v440_acc += (v87_bc * v400_data);
              v440_acc += (v89_bc * v401_data);
              v440_acc += (v91_bc * v402_data);
              v440_acc += (v93_bc * v403_data);
              v440_acc += (v95_bc * v404_data);
              v440_acc += (v97_bc * v405_data);
              v440_acc += (v99_bc * v406_data);
              v440_acc += (v101_bc * v407_data);
              v440_acc += (v103_bc * v408_data);
              v440_acc += (v105_bc * v409_data);
              v440_acc += (v107_bc * v410_data);
              ir3.template select<16, 1>(16) = v440_acc;
              tensorforge::intel_esimd::simd<float, 16> v467_acc{};
              v467_acc += (v112_bc * v399_data);
              v467_acc += (v114_bc * v400_data);
              v467_acc += (v116_bc * v401_data);
              v467_acc += (v118_bc * v402_data);
              v467_acc += (v120_bc * v403_data);
              v467_acc += (v122_bc * v404_data);
              v467_acc += (v124_bc * v405_data);
              v467_acc += (v126_bc * v406_data);
              v467_acc += (v128_bc * v407_data);
              v467_acc += (v130_bc * v408_data);
              v467_acc += (v132_bc * v409_data);
              v467_acc += (v134_bc * v410_data);
              ir3.template select<16, 1>(32) = v467_acc;
              tensorforge::intel_esimd::simd<float, 16> v494_acc{};
              v494_acc += (v139_bc * v399_data);
              v494_acc += (v141_bc * v400_data);
              v494_acc += (v143_bc * v401_data);
              v494_acc += (v145_bc * v402_data);
              v494_acc += (v147_bc * v403_data);
              v494_acc += (v149_bc * v404_data);
              v494_acc += (v151_bc * v405_data);
              v494_acc += (v153_bc * v406_data);
              v494_acc += (v155_bc * v407_data);
              v494_acc += (v157_bc * v408_data);
              v494_acc += (v159_bc * v409_data);
              v494_acc += (v161_bc * v410_data);
              ir3.template select<16, 1>(48) = v494_acc;
              tensorforge::intel_esimd::simd<float, 16> v521_acc{};
              v521_acc += (v166_bc * v399_data);
              v521_acc += (v168_bc * v400_data);
              v521_acc += (v170_bc * v401_data);
              v521_acc += (v172_bc * v402_data);
              v521_acc += (v174_bc * v403_data);
              v521_acc += (v176_bc * v404_data);
              v521_acc += (v178_bc * v405_data);
              v521_acc += (v180_bc * v406_data);
              v521_acc += (v182_bc * v407_data);
              v521_acc += (v184_bc * v408_data);
              v521_acc += (v186_bc * v409_data);
              v521_acc += (v188_bc * v410_data);
              ir3.template select<16, 1>(64) = v521_acc;
              tensorforge::intel_esimd::simd<float, 16> v548_acc{};
              v548_acc += (v193_bc * v399_data);
              v548_acc += (v195_bc * v400_data);
              v548_acc += (v197_bc * v401_data);
              v548_acc += (v199_bc * v402_data);
              v548_acc += (v201_bc * v403_data);
              v548_acc += (v203_bc * v404_data);
              v548_acc += (v205_bc * v405_data);
              v548_acc += (v207_bc * v406_data);
              v548_acc += (v209_bc * v407_data);
              v548_acc += (v211_bc * v408_data);
              v548_acc += (v213_bc * v409_data);
              v548_acc += (v215_bc * v410_data);
              ir3.template select<16, 1>(80) = v548_acc;
              tensorforge::intel_esimd::simd<float, 16> v575_acc{};
              v575_acc += (v220_bc * v399_data);
              v575_acc += (v222_bc * v400_data);
              v575_acc += (v224_bc * v401_data);
              v575_acc += (v226_bc * v402_data);
              v575_acc += (v228_bc * v403_data);
              v575_acc += (v230_bc * v404_data);
              v575_acc += (v232_bc * v405_data);
              v575_acc += (v234_bc * v406_data);
              v575_acc += (v236_bc * v407_data);
              v575_acc += (v238_bc * v408_data);
              v575_acc += (v240_bc * v409_data);
              v575_acc += (v242_bc * v410_data);
              ir3.template select<16, 1>(96) = v575_acc;
              tensorforge::intel_esimd::simd<float, 16> v602_acc{};
              v602_acc += (v247_bc * v399_data);
              v602_acc += (v249_bc * v400_data);
              v602_acc += (v251_bc * v401_data);
              v602_acc += (v253_bc * v402_data);
              v602_acc += (v255_bc * v403_data);
              v602_acc += (v257_bc * v404_data);
              v602_acc += (v259_bc * v405_data);
              v602_acc += (v261_bc * v406_data);
              v602_acc += (v263_bc * v407_data);
              v602_acc += (v265_bc * v408_data);
              v602_acc += (v267_bc * v409_data);
              v602_acc += (v269_bc * v410_data);
              ir3.template select<16, 1>(112) = v602_acc;
              tensorforge::intel_esimd::simd<float, 16> v629_acc{};
              v629_acc += (v274_bc * v399_data);
              v629_acc += (v276_bc * v400_data);
              v629_acc += (v278_bc * v401_data);
              v629_acc += (v280_bc * v402_data);
              v629_acc += (v282_bc * v403_data);
              v629_acc += (v284_bc * v404_data);
              v629_acc += (v286_bc * v405_data);
              v629_acc += (v288_bc * v406_data);
              v629_acc += (v290_bc * v407_data);
              v629_acc += (v292_bc * v408_data);
              v629_acc += (v294_bc * v409_data);
              v629_acc += (v296_bc * v410_data);
              ir3.template select<16, 1>(128) = v629_acc;
              tensorforge::intel_esimd::simd<float, 16> v656_acc{};
              v656_acc += (v301_bc * v399_data);
              v656_acc += (v303_bc * v400_data);
              v656_acc += (v305_bc * v401_data);
              v656_acc += (v307_bc * v402_data);
              v656_acc += (v309_bc * v403_data);
              v656_acc += (v311_bc * v404_data);
              v656_acc += (v313_bc * v405_data);
              v656_acc += (v315_bc * v406_data);
              v656_acc += (v317_bc * v407_data);
              v656_acc += (v319_bc * v408_data);
              v656_acc += (v321_bc * v409_data);
              v656_acc += (v323_bc * v410_data);
              ir3.template select<16, 1>(144) = v656_acc;
              tensorforge::intel_esimd::simd<float, 16> v683_acc{};
              v683_acc += (v328_bc * v399_data);
              v683_acc += (v330_bc * v400_data);
              v683_acc += (v332_bc * v401_data);
              v683_acc += (v334_bc * v402_data);
              v683_acc += (v336_bc * v403_data);
              v683_acc += (v338_bc * v404_data);
              v683_acc += (v340_bc * v405_data);
              v683_acc += (v342_bc * v406_data);
              v683_acc += (v344_bc * v407_data);
              v683_acc += (v346_bc * v408_data);
              v683_acc += (v348_bc * v409_data);
              v683_acc += (v350_bc * v410_data);
              ir3.template select<16, 1>(160) = v683_acc;
              tensorforge::intel_esimd::simd<float, 16> v710_acc{};
              v710_acc += (v355_bc * v399_data);
              v710_acc += (v357_bc * v400_data);
              v710_acc += (v359_bc * v401_data);
              v710_acc += (v361_bc * v402_data);
              v710_acc += (v363_bc * v403_data);
              v710_acc += (v365_bc * v404_data);
              v710_acc += (v367_bc * v405_data);
              v710_acc += (v369_bc * v406_data);
              v710_acc += (v371_bc * v407_data);
              v710_acc += (v373_bc * v408_data);
              v710_acc += (v375_bc * v409_data);
              v710_acc += (v377_bc * v410_data);
              ir3.template select<16, 1>(176) = v710_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v737_n1 = 0; v737_n1 < 12; ++v737_n1) {
                int32_t v738_a = v737_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v740_data(ir3.template select<6, 1>(v738_a));
                r3.template select<6, 1>(v738_a) = v740_data;
              }
              // s1 = store{r>s, clear}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v741_z1 = 0; v741_z1 < 12; ++v741_z1) {
                s1[(6_i32 + (v741_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v748_i1 = 0; v748_i1 < 12; ++v748_i1) {
                tensorforge::intel_esimd::simd<float, 6> v751_data(r3.template select<6, 1>((v748_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v748_i1 * 12)), v751_data);
              }
              // wait(r4 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // ir5 = +(r4)
              // [(0, 6), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v758_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v759_data(ir5.template select<16, 1>(0));
              ir5.template select<16, 1>(0) = (v759_data + v758_data);
              tensorforge::intel_esimd::simd<float, 16> v761_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v762_data(ir5.template select<16, 1>(16));
              ir5.template select<16, 1>(16) = (v762_data + v761_data);
              tensorforge::intel_esimd::simd<float, 16> v764_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v765_data(ir5.template select<16, 1>(32));
              ir5.template select<16, 1>(32) = (v765_data + v764_data);
              tensorforge::intel_esimd::simd<float, 16> v767_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v768_data(ir5.template select<16, 1>(48));
              ir5.template select<16, 1>(48) = (v768_data + v767_data);
              tensorforge::intel_esimd::simd<float, 16> v770_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v771_data(ir5.template select<16, 1>(64));
              ir5.template select<16, 1>(64) = (v771_data + v770_data);
              tensorforge::intel_esimd::simd<float, 16> v773_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v774_data(ir5.template select<16, 1>(80));
              ir5.template select<16, 1>(80) = (v774_data + v773_data);
              tensorforge::intel_esimd::simd<float, 16> v776_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v777_data(ir5.template select<16, 1>(96));
              ir5.template select<16, 1>(96) = (v777_data + v776_data);
              tensorforge::intel_esimd::simd<float, 16> v779_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v780_data(ir5.template select<16, 1>(112));
              ir5.template select<16, 1>(112) = (v780_data + v779_data);
              tensorforge::intel_esimd::simd<float, 16> v782_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v783_data(ir5.template select<16, 1>(128));
              ir5.template select<16, 1>(128) = (v783_data + v782_data);
              tensorforge::intel_esimd::simd<float, 16> v785_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v786_data(ir5.template select<16, 1>(144));
              ir5.template select<16, 1>(144) = (v786_data + v785_data);
              tensorforge::intel_esimd::simd<float, 16> v788_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v789_data(ir5.template select<16, 1>(160));
              ir5.template select<16, 1>(160) = (v789_data + v788_data);
              tensorforge::intel_esimd::simd<float, 16> v791_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v792_data(ir5.template select<16, 1>(176));
              ir5.template select<16, 1>(176) = (v792_data + v791_data);
              // r5 = ir5
              #pragma unroll
              for (int32_t v794_n1 = 0; v794_n1 < 12; ++v794_n1) {
                int32_t v795_a = v794_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v797_data(ir5.template select<6, 1>(v795_a));
                r5.template select<6, 1>(v795_a) = v797_data;
              }
              // s1 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v798_i1 = 0; v798_i1 < 12; ++v798_i1) {
                tensorforge::intel_esimd::simd<float, 6> v801_data(r5.template select<6, 1>((v798_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v798_i1 * 12))), v801_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // ir6 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v812_data(0.0f);
              v812_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v813_data(ir6.template select<16, 1>(0));
              ir6.template select<16, 1>(0) = (v813_data + v812_data);
              tensorforge::intel_esimd::simd<float, 16> v816_data(0.0f);
              v816_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v817_data(ir6.template select<16, 1>(16));
              ir6.template select<16, 1>(16) = (v817_data + v816_data);
              tensorforge::intel_esimd::simd<float, 16> v820_data(0.0f);
              v820_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v821_data(ir6.template select<16, 1>(32));
              ir6.template select<16, 1>(32) = (v821_data + v820_data);
              tensorforge::intel_esimd::simd<float, 16> v824_data(0.0f);
              v824_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v825_data(ir6.template select<16, 1>(48));
              ir6.template select<16, 1>(48) = (v825_data + v824_data);
              tensorforge::intel_esimd::simd<float, 16> v828_data(0.0f);
              v828_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v829_data(ir6.template select<16, 1>(64));
              ir6.template select<16, 1>(64) = (v829_data + v828_data);
              tensorforge::intel_esimd::simd<float, 16> v832_data(0.0f);
              v832_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v833_data(ir6.template select<16, 1>(80));
              ir6.template select<16, 1>(80) = (v833_data + v832_data);
              tensorforge::intel_esimd::simd<float, 16> v836_data(0.0f);
              v836_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v837_data(ir6.template select<16, 1>(96));
              ir6.template select<16, 1>(96) = (v837_data + v836_data);
              tensorforge::intel_esimd::simd<float, 16> v840_data(0.0f);
              v840_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v841_data(ir6.template select<16, 1>(112));
              ir6.template select<16, 1>(112) = (v841_data + v840_data);
              tensorforge::intel_esimd::simd<float, 16> v844_data(0.0f);
              v844_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v845_data(ir6.template select<16, 1>(128));
              ir6.template select<16, 1>(128) = (v845_data + v844_data);
              tensorforge::intel_esimd::simd<float, 16> v848_data(0.0f);
              v848_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v849_data(ir6.template select<16, 1>(144));
              ir6.template select<16, 1>(144) = (v849_data + v848_data);
              tensorforge::intel_esimd::simd<float, 16> v852_data(0.0f);
              v852_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v853_data(ir6.template select<16, 1>(160));
              ir6.template select<16, 1>(160) = (v853_data + v852_data);
              tensorforge::intel_esimd::simd<float, 16> v856_data(0.0f);
              v856_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v857_data(ir6.template select<16, 1>(176));
              ir6.template select<16, 1>(176) = (v857_data + v856_data);
              // r6 = ir6
              #pragma unroll
              for (int32_t v859_n1 = 0; v859_n1 < 12; ++v859_n1) {
                int32_t v860_a = v859_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v862_data(ir6.template select<12, 1>(v860_a));
                r6.template select<12, 1>(v860_a) = v862_data;
              }
              // glb_m4 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v863_i1 = 0; v863_i1 < 12; ++v863_i1) {
                tensorforge::intel_esimd::simd<float, 12> v866_data(r6.template select<12, 1>((v863_i1 * 16)));
                v866_data.copy_to(glb_m4 + ((v863_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


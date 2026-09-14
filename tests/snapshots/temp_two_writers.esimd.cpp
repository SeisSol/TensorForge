// === base name ===
kernel_0f06d13edce8c5f4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0f06d13edce8c5f4 = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0f06d13edce8c5f4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0f06d13edce8c5f4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0f06d13edce8c5f4(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_0f06d13edce8c5f4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0f06d13edce8c5f4(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0f06d13edce8c5f4(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0f06d13edce8c5f4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4864 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 19456 B shared, occupancy grid
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4864}],"shared_bytes":19456,"shared_elements":4864,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 144 + 0 + m4_extraOffset];
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
                tensorforge::slmStore<float, 6>(s1 + ((v379_i1 * 12)), v382_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v388_i1 = 0; v388_i1 < 12; ++v388_i1) {
                tensorforge::intel_esimd::simd<float, 12> v393_data;
                v393_data.copy_from(glb_m4 + ((v388_i1 * 12)));
                r4.template select<12, 1>((v388_i1 * 16)) = v393_data;
              }
              // wait(r2 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s0)
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v398_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v399_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v400_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v401_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v402_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v403_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v404_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v405_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v406_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v407_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v408_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v409_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v410_acc{};
              v410_acc += (v58_bc * v398_data);
              v410_acc += (v60_bc * v399_data);
              v410_acc += (v62_bc * v400_data);
              v410_acc += (v64_bc * v401_data);
              v410_acc += (v66_bc * v402_data);
              v410_acc += (v68_bc * v403_data);
              v410_acc += (v70_bc * v404_data);
              v410_acc += (v72_bc * v405_data);
              v410_acc += (v74_bc * v406_data);
              v410_acc += (v76_bc * v407_data);
              v410_acc += (v78_bc * v408_data);
              v410_acc += (v80_bc * v409_data);
              ir3.template select<16, 1>(0) = v410_acc;
              tensorforge::intel_esimd::simd<float, 16> v439_acc{};
              v439_acc += (v85_bc * v398_data);
              v439_acc += (v87_bc * v399_data);
              v439_acc += (v89_bc * v400_data);
              v439_acc += (v91_bc * v401_data);
              v439_acc += (v93_bc * v402_data);
              v439_acc += (v95_bc * v403_data);
              v439_acc += (v97_bc * v404_data);
              v439_acc += (v99_bc * v405_data);
              v439_acc += (v101_bc * v406_data);
              v439_acc += (v103_bc * v407_data);
              v439_acc += (v105_bc * v408_data);
              v439_acc += (v107_bc * v409_data);
              ir3.template select<16, 1>(16) = v439_acc;
              tensorforge::intel_esimd::simd<float, 16> v466_acc{};
              v466_acc += (v112_bc * v398_data);
              v466_acc += (v114_bc * v399_data);
              v466_acc += (v116_bc * v400_data);
              v466_acc += (v118_bc * v401_data);
              v466_acc += (v120_bc * v402_data);
              v466_acc += (v122_bc * v403_data);
              v466_acc += (v124_bc * v404_data);
              v466_acc += (v126_bc * v405_data);
              v466_acc += (v128_bc * v406_data);
              v466_acc += (v130_bc * v407_data);
              v466_acc += (v132_bc * v408_data);
              v466_acc += (v134_bc * v409_data);
              ir3.template select<16, 1>(32) = v466_acc;
              tensorforge::intel_esimd::simd<float, 16> v493_acc{};
              v493_acc += (v139_bc * v398_data);
              v493_acc += (v141_bc * v399_data);
              v493_acc += (v143_bc * v400_data);
              v493_acc += (v145_bc * v401_data);
              v493_acc += (v147_bc * v402_data);
              v493_acc += (v149_bc * v403_data);
              v493_acc += (v151_bc * v404_data);
              v493_acc += (v153_bc * v405_data);
              v493_acc += (v155_bc * v406_data);
              v493_acc += (v157_bc * v407_data);
              v493_acc += (v159_bc * v408_data);
              v493_acc += (v161_bc * v409_data);
              ir3.template select<16, 1>(48) = v493_acc;
              tensorforge::intel_esimd::simd<float, 16> v520_acc{};
              v520_acc += (v166_bc * v398_data);
              v520_acc += (v168_bc * v399_data);
              v520_acc += (v170_bc * v400_data);
              v520_acc += (v172_bc * v401_data);
              v520_acc += (v174_bc * v402_data);
              v520_acc += (v176_bc * v403_data);
              v520_acc += (v178_bc * v404_data);
              v520_acc += (v180_bc * v405_data);
              v520_acc += (v182_bc * v406_data);
              v520_acc += (v184_bc * v407_data);
              v520_acc += (v186_bc * v408_data);
              v520_acc += (v188_bc * v409_data);
              ir3.template select<16, 1>(64) = v520_acc;
              tensorforge::intel_esimd::simd<float, 16> v547_acc{};
              v547_acc += (v193_bc * v398_data);
              v547_acc += (v195_bc * v399_data);
              v547_acc += (v197_bc * v400_data);
              v547_acc += (v199_bc * v401_data);
              v547_acc += (v201_bc * v402_data);
              v547_acc += (v203_bc * v403_data);
              v547_acc += (v205_bc * v404_data);
              v547_acc += (v207_bc * v405_data);
              v547_acc += (v209_bc * v406_data);
              v547_acc += (v211_bc * v407_data);
              v547_acc += (v213_bc * v408_data);
              v547_acc += (v215_bc * v409_data);
              ir3.template select<16, 1>(80) = v547_acc;
              tensorforge::intel_esimd::simd<float, 16> v574_acc{};
              v574_acc += (v220_bc * v398_data);
              v574_acc += (v222_bc * v399_data);
              v574_acc += (v224_bc * v400_data);
              v574_acc += (v226_bc * v401_data);
              v574_acc += (v228_bc * v402_data);
              v574_acc += (v230_bc * v403_data);
              v574_acc += (v232_bc * v404_data);
              v574_acc += (v234_bc * v405_data);
              v574_acc += (v236_bc * v406_data);
              v574_acc += (v238_bc * v407_data);
              v574_acc += (v240_bc * v408_data);
              v574_acc += (v242_bc * v409_data);
              ir3.template select<16, 1>(96) = v574_acc;
              tensorforge::intel_esimd::simd<float, 16> v601_acc{};
              v601_acc += (v247_bc * v398_data);
              v601_acc += (v249_bc * v399_data);
              v601_acc += (v251_bc * v400_data);
              v601_acc += (v253_bc * v401_data);
              v601_acc += (v255_bc * v402_data);
              v601_acc += (v257_bc * v403_data);
              v601_acc += (v259_bc * v404_data);
              v601_acc += (v261_bc * v405_data);
              v601_acc += (v263_bc * v406_data);
              v601_acc += (v265_bc * v407_data);
              v601_acc += (v267_bc * v408_data);
              v601_acc += (v269_bc * v409_data);
              ir3.template select<16, 1>(112) = v601_acc;
              tensorforge::intel_esimd::simd<float, 16> v628_acc{};
              v628_acc += (v274_bc * v398_data);
              v628_acc += (v276_bc * v399_data);
              v628_acc += (v278_bc * v400_data);
              v628_acc += (v280_bc * v401_data);
              v628_acc += (v282_bc * v402_data);
              v628_acc += (v284_bc * v403_data);
              v628_acc += (v286_bc * v404_data);
              v628_acc += (v288_bc * v405_data);
              v628_acc += (v290_bc * v406_data);
              v628_acc += (v292_bc * v407_data);
              v628_acc += (v294_bc * v408_data);
              v628_acc += (v296_bc * v409_data);
              ir3.template select<16, 1>(128) = v628_acc;
              tensorforge::intel_esimd::simd<float, 16> v655_acc{};
              v655_acc += (v301_bc * v398_data);
              v655_acc += (v303_bc * v399_data);
              v655_acc += (v305_bc * v400_data);
              v655_acc += (v307_bc * v401_data);
              v655_acc += (v309_bc * v402_data);
              v655_acc += (v311_bc * v403_data);
              v655_acc += (v313_bc * v404_data);
              v655_acc += (v315_bc * v405_data);
              v655_acc += (v317_bc * v406_data);
              v655_acc += (v319_bc * v407_data);
              v655_acc += (v321_bc * v408_data);
              v655_acc += (v323_bc * v409_data);
              ir3.template select<16, 1>(144) = v655_acc;
              tensorforge::intel_esimd::simd<float, 16> v682_acc{};
              v682_acc += (v328_bc * v398_data);
              v682_acc += (v330_bc * v399_data);
              v682_acc += (v332_bc * v400_data);
              v682_acc += (v334_bc * v401_data);
              v682_acc += (v336_bc * v402_data);
              v682_acc += (v338_bc * v403_data);
              v682_acc += (v340_bc * v404_data);
              v682_acc += (v342_bc * v405_data);
              v682_acc += (v344_bc * v406_data);
              v682_acc += (v346_bc * v407_data);
              v682_acc += (v348_bc * v408_data);
              v682_acc += (v350_bc * v409_data);
              ir3.template select<16, 1>(160) = v682_acc;
              tensorforge::intel_esimd::simd<float, 16> v709_acc{};
              v709_acc += (v355_bc * v398_data);
              v709_acc += (v357_bc * v399_data);
              v709_acc += (v359_bc * v400_data);
              v709_acc += (v361_bc * v401_data);
              v709_acc += (v363_bc * v402_data);
              v709_acc += (v365_bc * v403_data);
              v709_acc += (v367_bc * v404_data);
              v709_acc += (v369_bc * v405_data);
              v709_acc += (v371_bc * v406_data);
              v709_acc += (v373_bc * v407_data);
              v709_acc += (v375_bc * v408_data);
              v709_acc += (v377_bc * v409_data);
              ir3.template select<16, 1>(176) = v709_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v736_n1 = 0; v736_n1 < 12; ++v736_n1) {
                int32_t v737_a = v736_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v739_data(ir3.template select<6, 1>(v737_a));
                r3.template select<6, 1>(v737_a) = v739_data;
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v740_i1 = 0; v740_i1 < 12; ++v740_i1) {
                tensorforge::intel_esimd::simd<float, 6> v743_data(r3.template select<6, 1>((v740_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v740_i1 * 12))), v743_data);
              }
              // wait(r4 = load{g>r}(glb_m4););
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // ir5 = +(r4 * s1)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v751_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v752_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v753_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v754_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v755_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v756_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v757_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v758_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v759_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v760_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v761_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v762_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v763_acc{};
              tensorforge::intel_esimd::simd<float, 16> v767_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v763_acc += ((static_cast<float>(v767_data[0])) * v751_data);
              v763_acc += ((static_cast<float>(v767_data[1])) * v752_data);
              v763_acc += ((static_cast<float>(v767_data[2])) * v753_data);
              v763_acc += ((static_cast<float>(v767_data[3])) * v754_data);
              v763_acc += ((static_cast<float>(v767_data[4])) * v755_data);
              v763_acc += ((static_cast<float>(v767_data[5])) * v756_data);
              v763_acc += ((static_cast<float>(v767_data[6])) * v757_data);
              v763_acc += ((static_cast<float>(v767_data[7])) * v758_data);
              v763_acc += ((static_cast<float>(v767_data[8])) * v759_data);
              v763_acc += ((static_cast<float>(v767_data[9])) * v760_data);
              v763_acc += ((static_cast<float>(v767_data[10])) * v761_data);
              v763_acc += ((static_cast<float>(v767_data[11])) * v762_data);
              ir5.template select<16, 1>(0) = v763_acc;
              tensorforge::intel_esimd::simd<float, 16> v792_acc{};
              tensorforge::intel_esimd::simd<float, 16> v794_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v792_acc += ((static_cast<float>(v794_data[0])) * v751_data);
              v792_acc += ((static_cast<float>(v794_data[1])) * v752_data);
              v792_acc += ((static_cast<float>(v794_data[2])) * v753_data);
              v792_acc += ((static_cast<float>(v794_data[3])) * v754_data);
              v792_acc += ((static_cast<float>(v794_data[4])) * v755_data);
              v792_acc += ((static_cast<float>(v794_data[5])) * v756_data);
              v792_acc += ((static_cast<float>(v794_data[6])) * v757_data);
              v792_acc += ((static_cast<float>(v794_data[7])) * v758_data);
              v792_acc += ((static_cast<float>(v794_data[8])) * v759_data);
              v792_acc += ((static_cast<float>(v794_data[9])) * v760_data);
              v792_acc += ((static_cast<float>(v794_data[10])) * v761_data);
              v792_acc += ((static_cast<float>(v794_data[11])) * v762_data);
              ir5.template select<16, 1>(16) = v792_acc;
              tensorforge::intel_esimd::simd<float, 16> v819_acc{};
              tensorforge::intel_esimd::simd<float, 16> v821_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v819_acc += ((static_cast<float>(v821_data[0])) * v751_data);
              v819_acc += ((static_cast<float>(v821_data[1])) * v752_data);
              v819_acc += ((static_cast<float>(v821_data[2])) * v753_data);
              v819_acc += ((static_cast<float>(v821_data[3])) * v754_data);
              v819_acc += ((static_cast<float>(v821_data[4])) * v755_data);
              v819_acc += ((static_cast<float>(v821_data[5])) * v756_data);
              v819_acc += ((static_cast<float>(v821_data[6])) * v757_data);
              v819_acc += ((static_cast<float>(v821_data[7])) * v758_data);
              v819_acc += ((static_cast<float>(v821_data[8])) * v759_data);
              v819_acc += ((static_cast<float>(v821_data[9])) * v760_data);
              v819_acc += ((static_cast<float>(v821_data[10])) * v761_data);
              v819_acc += ((static_cast<float>(v821_data[11])) * v762_data);
              ir5.template select<16, 1>(32) = v819_acc;
              tensorforge::intel_esimd::simd<float, 16> v846_acc{};
              tensorforge::intel_esimd::simd<float, 16> v848_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v846_acc += ((static_cast<float>(v848_data[0])) * v751_data);
              v846_acc += ((static_cast<float>(v848_data[1])) * v752_data);
              v846_acc += ((static_cast<float>(v848_data[2])) * v753_data);
              v846_acc += ((static_cast<float>(v848_data[3])) * v754_data);
              v846_acc += ((static_cast<float>(v848_data[4])) * v755_data);
              v846_acc += ((static_cast<float>(v848_data[5])) * v756_data);
              v846_acc += ((static_cast<float>(v848_data[6])) * v757_data);
              v846_acc += ((static_cast<float>(v848_data[7])) * v758_data);
              v846_acc += ((static_cast<float>(v848_data[8])) * v759_data);
              v846_acc += ((static_cast<float>(v848_data[9])) * v760_data);
              v846_acc += ((static_cast<float>(v848_data[10])) * v761_data);
              v846_acc += ((static_cast<float>(v848_data[11])) * v762_data);
              ir5.template select<16, 1>(48) = v846_acc;
              tensorforge::intel_esimd::simd<float, 16> v873_acc{};
              tensorforge::intel_esimd::simd<float, 16> v875_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v873_acc += ((static_cast<float>(v875_data[0])) * v751_data);
              v873_acc += ((static_cast<float>(v875_data[1])) * v752_data);
              v873_acc += ((static_cast<float>(v875_data[2])) * v753_data);
              v873_acc += ((static_cast<float>(v875_data[3])) * v754_data);
              v873_acc += ((static_cast<float>(v875_data[4])) * v755_data);
              v873_acc += ((static_cast<float>(v875_data[5])) * v756_data);
              v873_acc += ((static_cast<float>(v875_data[6])) * v757_data);
              v873_acc += ((static_cast<float>(v875_data[7])) * v758_data);
              v873_acc += ((static_cast<float>(v875_data[8])) * v759_data);
              v873_acc += ((static_cast<float>(v875_data[9])) * v760_data);
              v873_acc += ((static_cast<float>(v875_data[10])) * v761_data);
              v873_acc += ((static_cast<float>(v875_data[11])) * v762_data);
              ir5.template select<16, 1>(64) = v873_acc;
              tensorforge::intel_esimd::simd<float, 16> v900_acc{};
              tensorforge::intel_esimd::simd<float, 16> v902_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v900_acc += ((static_cast<float>(v902_data[0])) * v751_data);
              v900_acc += ((static_cast<float>(v902_data[1])) * v752_data);
              v900_acc += ((static_cast<float>(v902_data[2])) * v753_data);
              v900_acc += ((static_cast<float>(v902_data[3])) * v754_data);
              v900_acc += ((static_cast<float>(v902_data[4])) * v755_data);
              v900_acc += ((static_cast<float>(v902_data[5])) * v756_data);
              v900_acc += ((static_cast<float>(v902_data[6])) * v757_data);
              v900_acc += ((static_cast<float>(v902_data[7])) * v758_data);
              v900_acc += ((static_cast<float>(v902_data[8])) * v759_data);
              v900_acc += ((static_cast<float>(v902_data[9])) * v760_data);
              v900_acc += ((static_cast<float>(v902_data[10])) * v761_data);
              v900_acc += ((static_cast<float>(v902_data[11])) * v762_data);
              ir5.template select<16, 1>(80) = v900_acc;
              tensorforge::intel_esimd::simd<float, 16> v927_acc{};
              tensorforge::intel_esimd::simd<float, 16> v929_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v927_acc += ((static_cast<float>(v929_data[0])) * v751_data);
              v927_acc += ((static_cast<float>(v929_data[1])) * v752_data);
              v927_acc += ((static_cast<float>(v929_data[2])) * v753_data);
              v927_acc += ((static_cast<float>(v929_data[3])) * v754_data);
              v927_acc += ((static_cast<float>(v929_data[4])) * v755_data);
              v927_acc += ((static_cast<float>(v929_data[5])) * v756_data);
              v927_acc += ((static_cast<float>(v929_data[6])) * v757_data);
              v927_acc += ((static_cast<float>(v929_data[7])) * v758_data);
              v927_acc += ((static_cast<float>(v929_data[8])) * v759_data);
              v927_acc += ((static_cast<float>(v929_data[9])) * v760_data);
              v927_acc += ((static_cast<float>(v929_data[10])) * v761_data);
              v927_acc += ((static_cast<float>(v929_data[11])) * v762_data);
              ir5.template select<16, 1>(96) = v927_acc;
              tensorforge::intel_esimd::simd<float, 16> v954_acc{};
              tensorforge::intel_esimd::simd<float, 16> v956_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v954_acc += ((static_cast<float>(v956_data[0])) * v751_data);
              v954_acc += ((static_cast<float>(v956_data[1])) * v752_data);
              v954_acc += ((static_cast<float>(v956_data[2])) * v753_data);
              v954_acc += ((static_cast<float>(v956_data[3])) * v754_data);
              v954_acc += ((static_cast<float>(v956_data[4])) * v755_data);
              v954_acc += ((static_cast<float>(v956_data[5])) * v756_data);
              v954_acc += ((static_cast<float>(v956_data[6])) * v757_data);
              v954_acc += ((static_cast<float>(v956_data[7])) * v758_data);
              v954_acc += ((static_cast<float>(v956_data[8])) * v759_data);
              v954_acc += ((static_cast<float>(v956_data[9])) * v760_data);
              v954_acc += ((static_cast<float>(v956_data[10])) * v761_data);
              v954_acc += ((static_cast<float>(v956_data[11])) * v762_data);
              ir5.template select<16, 1>(112) = v954_acc;
              tensorforge::intel_esimd::simd<float, 16> v981_acc{};
              tensorforge::intel_esimd::simd<float, 16> v983_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v981_acc += ((static_cast<float>(v983_data[0])) * v751_data);
              v981_acc += ((static_cast<float>(v983_data[1])) * v752_data);
              v981_acc += ((static_cast<float>(v983_data[2])) * v753_data);
              v981_acc += ((static_cast<float>(v983_data[3])) * v754_data);
              v981_acc += ((static_cast<float>(v983_data[4])) * v755_data);
              v981_acc += ((static_cast<float>(v983_data[5])) * v756_data);
              v981_acc += ((static_cast<float>(v983_data[6])) * v757_data);
              v981_acc += ((static_cast<float>(v983_data[7])) * v758_data);
              v981_acc += ((static_cast<float>(v983_data[8])) * v759_data);
              v981_acc += ((static_cast<float>(v983_data[9])) * v760_data);
              v981_acc += ((static_cast<float>(v983_data[10])) * v761_data);
              v981_acc += ((static_cast<float>(v983_data[11])) * v762_data);
              ir5.template select<16, 1>(128) = v981_acc;
              tensorforge::intel_esimd::simd<float, 16> v1008_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1010_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v1008_acc += ((static_cast<float>(v1010_data[0])) * v751_data);
              v1008_acc += ((static_cast<float>(v1010_data[1])) * v752_data);
              v1008_acc += ((static_cast<float>(v1010_data[2])) * v753_data);
              v1008_acc += ((static_cast<float>(v1010_data[3])) * v754_data);
              v1008_acc += ((static_cast<float>(v1010_data[4])) * v755_data);
              v1008_acc += ((static_cast<float>(v1010_data[5])) * v756_data);
              v1008_acc += ((static_cast<float>(v1010_data[6])) * v757_data);
              v1008_acc += ((static_cast<float>(v1010_data[7])) * v758_data);
              v1008_acc += ((static_cast<float>(v1010_data[8])) * v759_data);
              v1008_acc += ((static_cast<float>(v1010_data[9])) * v760_data);
              v1008_acc += ((static_cast<float>(v1010_data[10])) * v761_data);
              v1008_acc += ((static_cast<float>(v1010_data[11])) * v762_data);
              ir5.template select<16, 1>(144) = v1008_acc;
              tensorforge::intel_esimd::simd<float, 16> v1035_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1037_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v1035_acc += ((static_cast<float>(v1037_data[0])) * v751_data);
              v1035_acc += ((static_cast<float>(v1037_data[1])) * v752_data);
              v1035_acc += ((static_cast<float>(v1037_data[2])) * v753_data);
              v1035_acc += ((static_cast<float>(v1037_data[3])) * v754_data);
              v1035_acc += ((static_cast<float>(v1037_data[4])) * v755_data);
              v1035_acc += ((static_cast<float>(v1037_data[5])) * v756_data);
              v1035_acc += ((static_cast<float>(v1037_data[6])) * v757_data);
              v1035_acc += ((static_cast<float>(v1037_data[7])) * v758_data);
              v1035_acc += ((static_cast<float>(v1037_data[8])) * v759_data);
              v1035_acc += ((static_cast<float>(v1037_data[9])) * v760_data);
              v1035_acc += ((static_cast<float>(v1037_data[10])) * v761_data);
              v1035_acc += ((static_cast<float>(v1037_data[11])) * v762_data);
              ir5.template select<16, 1>(160) = v1035_acc;
              tensorforge::intel_esimd::simd<float, 16> v1062_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1064_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v1062_acc += ((static_cast<float>(v1064_data[0])) * v751_data);
              v1062_acc += ((static_cast<float>(v1064_data[1])) * v752_data);
              v1062_acc += ((static_cast<float>(v1064_data[2])) * v753_data);
              v1062_acc += ((static_cast<float>(v1064_data[3])) * v754_data);
              v1062_acc += ((static_cast<float>(v1064_data[4])) * v755_data);
              v1062_acc += ((static_cast<float>(v1064_data[5])) * v756_data);
              v1062_acc += ((static_cast<float>(v1064_data[6])) * v757_data);
              v1062_acc += ((static_cast<float>(v1064_data[7])) * v758_data);
              v1062_acc += ((static_cast<float>(v1064_data[8])) * v759_data);
              v1062_acc += ((static_cast<float>(v1064_data[9])) * v760_data);
              v1062_acc += ((static_cast<float>(v1064_data[10])) * v761_data);
              v1062_acc += ((static_cast<float>(v1064_data[11])) * v762_data);
              ir5.template select<16, 1>(176) = v1062_acc;
              // r5 = ir5
              #pragma unroll
              for (int32_t v1089_n1 = 0; v1089_n1 < 12; ++v1089_n1) {
                int32_t v1090_a = v1089_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1092_data(ir5.template select<12, 1>(v1090_a));
                r5.template select<12, 1>(v1090_a) = v1092_data;
              }
              // glb_m3 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1093_i1 = 0; v1093_i1 < 12; ++v1093_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1096_data(r5.template select<12, 1>((v1093_i1 * 16)));
                v1096_data.copy_to(glb_m3 + ((v1093_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


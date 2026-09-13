// === base name ===
kernel_8e2ca611d4c70458

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8e2ca611d4c70458 = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8e2ca611d4c70458(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8e2ca611d4c70458(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8e2ca611d4c70458(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4864 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_8e2ca611d4c70458(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8e2ca611d4c70458(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8e2ca611d4c70458(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8e2ca611d4c70458(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const float *const __restrict__ pf_glb_m0 = &m0[v9_batchId1 * 72 + 0 + m0_extraOffset];
            const float *const __restrict__ pf_glb_m1 = &m1[v9_batchId1 * 144 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v9_batchId1 * 72 + 0 + m2_extraOffset];
            const float *const __restrict__ pf_glb_m4 = &m4[v9_batchId1 * 144 + 0 + m4_extraOffset];
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
              for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                tensorforge::intel_esimd::simd<float, 6> v29_data;
                v29_data.copy_from(glb_m0 + ((v24_i1 * 6)));
                r0.template select<6, 1>((v24_i1 * 16)) = v29_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v32_ld);
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v33_ld);
              tensorforge::intel_esimd::simd<float, 16> v34_ld;
              v34_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v34_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
                tensorforge::intel_esimd::simd<float, 6> v41_data;
                v41_data.copy_from(glb_m2 + ((v36_i1 * 6)));
                r2.template select<6, 1>((v36_i1 * 16)) = v41_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v57_acc{};
              tensorforge::intel_esimd::simd<float, 16> v61_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              float v62_bc = static_cast<float>(v61_data[0]);
              v57_acc += (v62_bc * v45_data);
              float v64_bc = static_cast<float>(v61_data[1]);
              v57_acc += (v64_bc * v46_data);
              float v66_bc = static_cast<float>(v61_data[2]);
              v57_acc += (v66_bc * v47_data);
              float v68_bc = static_cast<float>(v61_data[3]);
              v57_acc += (v68_bc * v48_data);
              float v70_bc = static_cast<float>(v61_data[4]);
              v57_acc += (v70_bc * v49_data);
              float v72_bc = static_cast<float>(v61_data[5]);
              v57_acc += (v72_bc * v50_data);
              float v74_bc = static_cast<float>(v61_data[6]);
              v57_acc += (v74_bc * v51_data);
              float v76_bc = static_cast<float>(v61_data[7]);
              v57_acc += (v76_bc * v52_data);
              float v78_bc = static_cast<float>(v61_data[8]);
              v57_acc += (v78_bc * v53_data);
              float v80_bc = static_cast<float>(v61_data[9]);
              v57_acc += (v80_bc * v54_data);
              float v82_bc = static_cast<float>(v61_data[10]);
              v57_acc += (v82_bc * v55_data);
              float v84_bc = static_cast<float>(v61_data[11]);
              v57_acc += (v84_bc * v56_data);
              r1.template select<16, 1>(0) = v57_acc;
              tensorforge::intel_esimd::simd<float, 16> v86_acc{};
              tensorforge::intel_esimd::simd<float, 16> v88_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              float v89_bc = static_cast<float>(v88_data[0]);
              v86_acc += (v89_bc * v45_data);
              float v91_bc = static_cast<float>(v88_data[1]);
              v86_acc += (v91_bc * v46_data);
              float v93_bc = static_cast<float>(v88_data[2]);
              v86_acc += (v93_bc * v47_data);
              float v95_bc = static_cast<float>(v88_data[3]);
              v86_acc += (v95_bc * v48_data);
              float v97_bc = static_cast<float>(v88_data[4]);
              v86_acc += (v97_bc * v49_data);
              float v99_bc = static_cast<float>(v88_data[5]);
              v86_acc += (v99_bc * v50_data);
              float v101_bc = static_cast<float>(v88_data[6]);
              v86_acc += (v101_bc * v51_data);
              float v103_bc = static_cast<float>(v88_data[7]);
              v86_acc += (v103_bc * v52_data);
              float v105_bc = static_cast<float>(v88_data[8]);
              v86_acc += (v105_bc * v53_data);
              float v107_bc = static_cast<float>(v88_data[9]);
              v86_acc += (v107_bc * v54_data);
              float v109_bc = static_cast<float>(v88_data[10]);
              v86_acc += (v109_bc * v55_data);
              float v111_bc = static_cast<float>(v88_data[11]);
              v86_acc += (v111_bc * v56_data);
              r1.template select<16, 1>(16) = v86_acc;
              tensorforge::intel_esimd::simd<float, 16> v113_acc{};
              tensorforge::intel_esimd::simd<float, 16> v115_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              float v116_bc = static_cast<float>(v115_data[0]);
              v113_acc += (v116_bc * v45_data);
              float v118_bc = static_cast<float>(v115_data[1]);
              v113_acc += (v118_bc * v46_data);
              float v120_bc = static_cast<float>(v115_data[2]);
              v113_acc += (v120_bc * v47_data);
              float v122_bc = static_cast<float>(v115_data[3]);
              v113_acc += (v122_bc * v48_data);
              float v124_bc = static_cast<float>(v115_data[4]);
              v113_acc += (v124_bc * v49_data);
              float v126_bc = static_cast<float>(v115_data[5]);
              v113_acc += (v126_bc * v50_data);
              float v128_bc = static_cast<float>(v115_data[6]);
              v113_acc += (v128_bc * v51_data);
              float v130_bc = static_cast<float>(v115_data[7]);
              v113_acc += (v130_bc * v52_data);
              float v132_bc = static_cast<float>(v115_data[8]);
              v113_acc += (v132_bc * v53_data);
              float v134_bc = static_cast<float>(v115_data[9]);
              v113_acc += (v134_bc * v54_data);
              float v136_bc = static_cast<float>(v115_data[10]);
              v113_acc += (v136_bc * v55_data);
              float v138_bc = static_cast<float>(v115_data[11]);
              v113_acc += (v138_bc * v56_data);
              r1.template select<16, 1>(32) = v113_acc;
              tensorforge::intel_esimd::simd<float, 16> v140_acc{};
              tensorforge::intel_esimd::simd<float, 16> v142_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              float v143_bc = static_cast<float>(v142_data[0]);
              v140_acc += (v143_bc * v45_data);
              float v145_bc = static_cast<float>(v142_data[1]);
              v140_acc += (v145_bc * v46_data);
              float v147_bc = static_cast<float>(v142_data[2]);
              v140_acc += (v147_bc * v47_data);
              float v149_bc = static_cast<float>(v142_data[3]);
              v140_acc += (v149_bc * v48_data);
              float v151_bc = static_cast<float>(v142_data[4]);
              v140_acc += (v151_bc * v49_data);
              float v153_bc = static_cast<float>(v142_data[5]);
              v140_acc += (v153_bc * v50_data);
              float v155_bc = static_cast<float>(v142_data[6]);
              v140_acc += (v155_bc * v51_data);
              float v157_bc = static_cast<float>(v142_data[7]);
              v140_acc += (v157_bc * v52_data);
              float v159_bc = static_cast<float>(v142_data[8]);
              v140_acc += (v159_bc * v53_data);
              float v161_bc = static_cast<float>(v142_data[9]);
              v140_acc += (v161_bc * v54_data);
              float v163_bc = static_cast<float>(v142_data[10]);
              v140_acc += (v163_bc * v55_data);
              float v165_bc = static_cast<float>(v142_data[11]);
              v140_acc += (v165_bc * v56_data);
              r1.template select<16, 1>(48) = v140_acc;
              tensorforge::intel_esimd::simd<float, 16> v167_acc{};
              tensorforge::intel_esimd::simd<float, 16> v169_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              float v170_bc = static_cast<float>(v169_data[0]);
              v167_acc += (v170_bc * v45_data);
              float v172_bc = static_cast<float>(v169_data[1]);
              v167_acc += (v172_bc * v46_data);
              float v174_bc = static_cast<float>(v169_data[2]);
              v167_acc += (v174_bc * v47_data);
              float v176_bc = static_cast<float>(v169_data[3]);
              v167_acc += (v176_bc * v48_data);
              float v178_bc = static_cast<float>(v169_data[4]);
              v167_acc += (v178_bc * v49_data);
              float v180_bc = static_cast<float>(v169_data[5]);
              v167_acc += (v180_bc * v50_data);
              float v182_bc = static_cast<float>(v169_data[6]);
              v167_acc += (v182_bc * v51_data);
              float v184_bc = static_cast<float>(v169_data[7]);
              v167_acc += (v184_bc * v52_data);
              float v186_bc = static_cast<float>(v169_data[8]);
              v167_acc += (v186_bc * v53_data);
              float v188_bc = static_cast<float>(v169_data[9]);
              v167_acc += (v188_bc * v54_data);
              float v190_bc = static_cast<float>(v169_data[10]);
              v167_acc += (v190_bc * v55_data);
              float v192_bc = static_cast<float>(v169_data[11]);
              v167_acc += (v192_bc * v56_data);
              r1.template select<16, 1>(64) = v167_acc;
              tensorforge::intel_esimd::simd<float, 16> v194_acc{};
              tensorforge::intel_esimd::simd<float, 16> v196_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              float v197_bc = static_cast<float>(v196_data[0]);
              v194_acc += (v197_bc * v45_data);
              float v199_bc = static_cast<float>(v196_data[1]);
              v194_acc += (v199_bc * v46_data);
              float v201_bc = static_cast<float>(v196_data[2]);
              v194_acc += (v201_bc * v47_data);
              float v203_bc = static_cast<float>(v196_data[3]);
              v194_acc += (v203_bc * v48_data);
              float v205_bc = static_cast<float>(v196_data[4]);
              v194_acc += (v205_bc * v49_data);
              float v207_bc = static_cast<float>(v196_data[5]);
              v194_acc += (v207_bc * v50_data);
              float v209_bc = static_cast<float>(v196_data[6]);
              v194_acc += (v209_bc * v51_data);
              float v211_bc = static_cast<float>(v196_data[7]);
              v194_acc += (v211_bc * v52_data);
              float v213_bc = static_cast<float>(v196_data[8]);
              v194_acc += (v213_bc * v53_data);
              float v215_bc = static_cast<float>(v196_data[9]);
              v194_acc += (v215_bc * v54_data);
              float v217_bc = static_cast<float>(v196_data[10]);
              v194_acc += (v217_bc * v55_data);
              float v219_bc = static_cast<float>(v196_data[11]);
              v194_acc += (v219_bc * v56_data);
              r1.template select<16, 1>(80) = v194_acc;
              tensorforge::intel_esimd::simd<float, 16> v221_acc{};
              tensorforge::intel_esimd::simd<float, 16> v223_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              float v224_bc = static_cast<float>(v223_data[0]);
              v221_acc += (v224_bc * v45_data);
              float v226_bc = static_cast<float>(v223_data[1]);
              v221_acc += (v226_bc * v46_data);
              float v228_bc = static_cast<float>(v223_data[2]);
              v221_acc += (v228_bc * v47_data);
              float v230_bc = static_cast<float>(v223_data[3]);
              v221_acc += (v230_bc * v48_data);
              float v232_bc = static_cast<float>(v223_data[4]);
              v221_acc += (v232_bc * v49_data);
              float v234_bc = static_cast<float>(v223_data[5]);
              v221_acc += (v234_bc * v50_data);
              float v236_bc = static_cast<float>(v223_data[6]);
              v221_acc += (v236_bc * v51_data);
              float v238_bc = static_cast<float>(v223_data[7]);
              v221_acc += (v238_bc * v52_data);
              float v240_bc = static_cast<float>(v223_data[8]);
              v221_acc += (v240_bc * v53_data);
              float v242_bc = static_cast<float>(v223_data[9]);
              v221_acc += (v242_bc * v54_data);
              float v244_bc = static_cast<float>(v223_data[10]);
              v221_acc += (v244_bc * v55_data);
              float v246_bc = static_cast<float>(v223_data[11]);
              v221_acc += (v246_bc * v56_data);
              r1.template select<16, 1>(96) = v221_acc;
              tensorforge::intel_esimd::simd<float, 16> v248_acc{};
              tensorforge::intel_esimd::simd<float, 16> v250_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              float v251_bc = static_cast<float>(v250_data[0]);
              v248_acc += (v251_bc * v45_data);
              float v253_bc = static_cast<float>(v250_data[1]);
              v248_acc += (v253_bc * v46_data);
              float v255_bc = static_cast<float>(v250_data[2]);
              v248_acc += (v255_bc * v47_data);
              float v257_bc = static_cast<float>(v250_data[3]);
              v248_acc += (v257_bc * v48_data);
              float v259_bc = static_cast<float>(v250_data[4]);
              v248_acc += (v259_bc * v49_data);
              float v261_bc = static_cast<float>(v250_data[5]);
              v248_acc += (v261_bc * v50_data);
              float v263_bc = static_cast<float>(v250_data[6]);
              v248_acc += (v263_bc * v51_data);
              float v265_bc = static_cast<float>(v250_data[7]);
              v248_acc += (v265_bc * v52_data);
              float v267_bc = static_cast<float>(v250_data[8]);
              v248_acc += (v267_bc * v53_data);
              float v269_bc = static_cast<float>(v250_data[9]);
              v248_acc += (v269_bc * v54_data);
              float v271_bc = static_cast<float>(v250_data[10]);
              v248_acc += (v271_bc * v55_data);
              float v273_bc = static_cast<float>(v250_data[11]);
              v248_acc += (v273_bc * v56_data);
              r1.template select<16, 1>(112) = v248_acc;
              tensorforge::intel_esimd::simd<float, 16> v275_acc{};
              tensorforge::intel_esimd::simd<float, 16> v277_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              float v278_bc = static_cast<float>(v277_data[0]);
              v275_acc += (v278_bc * v45_data);
              float v280_bc = static_cast<float>(v277_data[1]);
              v275_acc += (v280_bc * v46_data);
              float v282_bc = static_cast<float>(v277_data[2]);
              v275_acc += (v282_bc * v47_data);
              float v284_bc = static_cast<float>(v277_data[3]);
              v275_acc += (v284_bc * v48_data);
              float v286_bc = static_cast<float>(v277_data[4]);
              v275_acc += (v286_bc * v49_data);
              float v288_bc = static_cast<float>(v277_data[5]);
              v275_acc += (v288_bc * v50_data);
              float v290_bc = static_cast<float>(v277_data[6]);
              v275_acc += (v290_bc * v51_data);
              float v292_bc = static_cast<float>(v277_data[7]);
              v275_acc += (v292_bc * v52_data);
              float v294_bc = static_cast<float>(v277_data[8]);
              v275_acc += (v294_bc * v53_data);
              float v296_bc = static_cast<float>(v277_data[9]);
              v275_acc += (v296_bc * v54_data);
              float v298_bc = static_cast<float>(v277_data[10]);
              v275_acc += (v298_bc * v55_data);
              float v300_bc = static_cast<float>(v277_data[11]);
              v275_acc += (v300_bc * v56_data);
              r1.template select<16, 1>(128) = v275_acc;
              tensorforge::intel_esimd::simd<float, 16> v302_acc{};
              tensorforge::intel_esimd::simd<float, 16> v304_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              float v305_bc = static_cast<float>(v304_data[0]);
              v302_acc += (v305_bc * v45_data);
              float v307_bc = static_cast<float>(v304_data[1]);
              v302_acc += (v307_bc * v46_data);
              float v309_bc = static_cast<float>(v304_data[2]);
              v302_acc += (v309_bc * v47_data);
              float v311_bc = static_cast<float>(v304_data[3]);
              v302_acc += (v311_bc * v48_data);
              float v313_bc = static_cast<float>(v304_data[4]);
              v302_acc += (v313_bc * v49_data);
              float v315_bc = static_cast<float>(v304_data[5]);
              v302_acc += (v315_bc * v50_data);
              float v317_bc = static_cast<float>(v304_data[6]);
              v302_acc += (v317_bc * v51_data);
              float v319_bc = static_cast<float>(v304_data[7]);
              v302_acc += (v319_bc * v52_data);
              float v321_bc = static_cast<float>(v304_data[8]);
              v302_acc += (v321_bc * v53_data);
              float v323_bc = static_cast<float>(v304_data[9]);
              v302_acc += (v323_bc * v54_data);
              float v325_bc = static_cast<float>(v304_data[10]);
              v302_acc += (v325_bc * v55_data);
              float v327_bc = static_cast<float>(v304_data[11]);
              v302_acc += (v327_bc * v56_data);
              r1.template select<16, 1>(144) = v302_acc;
              tensorforge::intel_esimd::simd<float, 16> v329_acc{};
              tensorforge::intel_esimd::simd<float, 16> v331_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              float v332_bc = static_cast<float>(v331_data[0]);
              v329_acc += (v332_bc * v45_data);
              float v334_bc = static_cast<float>(v331_data[1]);
              v329_acc += (v334_bc * v46_data);
              float v336_bc = static_cast<float>(v331_data[2]);
              v329_acc += (v336_bc * v47_data);
              float v338_bc = static_cast<float>(v331_data[3]);
              v329_acc += (v338_bc * v48_data);
              float v340_bc = static_cast<float>(v331_data[4]);
              v329_acc += (v340_bc * v49_data);
              float v342_bc = static_cast<float>(v331_data[5]);
              v329_acc += (v342_bc * v50_data);
              float v344_bc = static_cast<float>(v331_data[6]);
              v329_acc += (v344_bc * v51_data);
              float v346_bc = static_cast<float>(v331_data[7]);
              v329_acc += (v346_bc * v52_data);
              float v348_bc = static_cast<float>(v331_data[8]);
              v329_acc += (v348_bc * v53_data);
              float v350_bc = static_cast<float>(v331_data[9]);
              v329_acc += (v350_bc * v54_data);
              float v352_bc = static_cast<float>(v331_data[10]);
              v329_acc += (v352_bc * v55_data);
              float v354_bc = static_cast<float>(v331_data[11]);
              v329_acc += (v354_bc * v56_data);
              r1.template select<16, 1>(160) = v329_acc;
              tensorforge::intel_esimd::simd<float, 16> v356_acc{};
              tensorforge::intel_esimd::simd<float, 16> v358_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              float v359_bc = static_cast<float>(v358_data[0]);
              v356_acc += (v359_bc * v45_data);
              float v361_bc = static_cast<float>(v358_data[1]);
              v356_acc += (v361_bc * v46_data);
              float v363_bc = static_cast<float>(v358_data[2]);
              v356_acc += (v363_bc * v47_data);
              float v365_bc = static_cast<float>(v358_data[3]);
              v356_acc += (v365_bc * v48_data);
              float v367_bc = static_cast<float>(v358_data[4]);
              v356_acc += (v367_bc * v49_data);
              float v369_bc = static_cast<float>(v358_data[5]);
              v356_acc += (v369_bc * v50_data);
              float v371_bc = static_cast<float>(v358_data[6]);
              v356_acc += (v371_bc * v51_data);
              float v373_bc = static_cast<float>(v358_data[7]);
              v356_acc += (v373_bc * v52_data);
              float v375_bc = static_cast<float>(v358_data[8]);
              v356_acc += (v375_bc * v53_data);
              float v377_bc = static_cast<float>(v358_data[9]);
              v356_acc += (v377_bc * v54_data);
              float v379_bc = static_cast<float>(v358_data[10]);
              v356_acc += (v379_bc * v55_data);
              float v381_bc = static_cast<float>(v358_data[11]);
              v356_acc += (v381_bc * v56_data);
              r1.template select<16, 1>(176) = v356_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v383_i1 = 0; v383_i1 < 12; ++v383_i1) {
                tensorforge::intel_esimd::simd<float, 6> v386_data(r1.template select<6, 1>((v383_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v383_i1 * 12)), v386_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v392_i1 = 0; v392_i1 < 12; ++v392_i1) {
                tensorforge::intel_esimd::simd<float, 12> v397_data;
                v397_data.copy_from(glb_m4 + ((v392_i1 * 12)));
                r4.template select<12, 1>((v392_i1 * 16)) = v397_data;
              }
              // wait(r2 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // r3 = +(r2 * s0) + None
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
              v414_acc += (v62_bc * v402_data);
              v414_acc += (v64_bc * v403_data);
              v414_acc += (v66_bc * v404_data);
              v414_acc += (v68_bc * v405_data);
              v414_acc += (v70_bc * v406_data);
              v414_acc += (v72_bc * v407_data);
              v414_acc += (v74_bc * v408_data);
              v414_acc += (v76_bc * v409_data);
              v414_acc += (v78_bc * v410_data);
              v414_acc += (v80_bc * v411_data);
              v414_acc += (v82_bc * v412_data);
              v414_acc += (v84_bc * v413_data);
              ir3.template select<16, 1>(0) = v414_acc;
              tensorforge::intel_esimd::simd<float, 16> v443_acc{};
              v443_acc += (v89_bc * v402_data);
              v443_acc += (v91_bc * v403_data);
              v443_acc += (v93_bc * v404_data);
              v443_acc += (v95_bc * v405_data);
              v443_acc += (v97_bc * v406_data);
              v443_acc += (v99_bc * v407_data);
              v443_acc += (v101_bc * v408_data);
              v443_acc += (v103_bc * v409_data);
              v443_acc += (v105_bc * v410_data);
              v443_acc += (v107_bc * v411_data);
              v443_acc += (v109_bc * v412_data);
              v443_acc += (v111_bc * v413_data);
              ir3.template select<16, 1>(16) = v443_acc;
              tensorforge::intel_esimd::simd<float, 16> v470_acc{};
              v470_acc += (v116_bc * v402_data);
              v470_acc += (v118_bc * v403_data);
              v470_acc += (v120_bc * v404_data);
              v470_acc += (v122_bc * v405_data);
              v470_acc += (v124_bc * v406_data);
              v470_acc += (v126_bc * v407_data);
              v470_acc += (v128_bc * v408_data);
              v470_acc += (v130_bc * v409_data);
              v470_acc += (v132_bc * v410_data);
              v470_acc += (v134_bc * v411_data);
              v470_acc += (v136_bc * v412_data);
              v470_acc += (v138_bc * v413_data);
              ir3.template select<16, 1>(32) = v470_acc;
              tensorforge::intel_esimd::simd<float, 16> v497_acc{};
              v497_acc += (v143_bc * v402_data);
              v497_acc += (v145_bc * v403_data);
              v497_acc += (v147_bc * v404_data);
              v497_acc += (v149_bc * v405_data);
              v497_acc += (v151_bc * v406_data);
              v497_acc += (v153_bc * v407_data);
              v497_acc += (v155_bc * v408_data);
              v497_acc += (v157_bc * v409_data);
              v497_acc += (v159_bc * v410_data);
              v497_acc += (v161_bc * v411_data);
              v497_acc += (v163_bc * v412_data);
              v497_acc += (v165_bc * v413_data);
              ir3.template select<16, 1>(48) = v497_acc;
              tensorforge::intel_esimd::simd<float, 16> v524_acc{};
              v524_acc += (v170_bc * v402_data);
              v524_acc += (v172_bc * v403_data);
              v524_acc += (v174_bc * v404_data);
              v524_acc += (v176_bc * v405_data);
              v524_acc += (v178_bc * v406_data);
              v524_acc += (v180_bc * v407_data);
              v524_acc += (v182_bc * v408_data);
              v524_acc += (v184_bc * v409_data);
              v524_acc += (v186_bc * v410_data);
              v524_acc += (v188_bc * v411_data);
              v524_acc += (v190_bc * v412_data);
              v524_acc += (v192_bc * v413_data);
              ir3.template select<16, 1>(64) = v524_acc;
              tensorforge::intel_esimd::simd<float, 16> v551_acc{};
              v551_acc += (v197_bc * v402_data);
              v551_acc += (v199_bc * v403_data);
              v551_acc += (v201_bc * v404_data);
              v551_acc += (v203_bc * v405_data);
              v551_acc += (v205_bc * v406_data);
              v551_acc += (v207_bc * v407_data);
              v551_acc += (v209_bc * v408_data);
              v551_acc += (v211_bc * v409_data);
              v551_acc += (v213_bc * v410_data);
              v551_acc += (v215_bc * v411_data);
              v551_acc += (v217_bc * v412_data);
              v551_acc += (v219_bc * v413_data);
              ir3.template select<16, 1>(80) = v551_acc;
              tensorforge::intel_esimd::simd<float, 16> v578_acc{};
              v578_acc += (v224_bc * v402_data);
              v578_acc += (v226_bc * v403_data);
              v578_acc += (v228_bc * v404_data);
              v578_acc += (v230_bc * v405_data);
              v578_acc += (v232_bc * v406_data);
              v578_acc += (v234_bc * v407_data);
              v578_acc += (v236_bc * v408_data);
              v578_acc += (v238_bc * v409_data);
              v578_acc += (v240_bc * v410_data);
              v578_acc += (v242_bc * v411_data);
              v578_acc += (v244_bc * v412_data);
              v578_acc += (v246_bc * v413_data);
              ir3.template select<16, 1>(96) = v578_acc;
              tensorforge::intel_esimd::simd<float, 16> v605_acc{};
              v605_acc += (v251_bc * v402_data);
              v605_acc += (v253_bc * v403_data);
              v605_acc += (v255_bc * v404_data);
              v605_acc += (v257_bc * v405_data);
              v605_acc += (v259_bc * v406_data);
              v605_acc += (v261_bc * v407_data);
              v605_acc += (v263_bc * v408_data);
              v605_acc += (v265_bc * v409_data);
              v605_acc += (v267_bc * v410_data);
              v605_acc += (v269_bc * v411_data);
              v605_acc += (v271_bc * v412_data);
              v605_acc += (v273_bc * v413_data);
              ir3.template select<16, 1>(112) = v605_acc;
              tensorforge::intel_esimd::simd<float, 16> v632_acc{};
              v632_acc += (v278_bc * v402_data);
              v632_acc += (v280_bc * v403_data);
              v632_acc += (v282_bc * v404_data);
              v632_acc += (v284_bc * v405_data);
              v632_acc += (v286_bc * v406_data);
              v632_acc += (v288_bc * v407_data);
              v632_acc += (v290_bc * v408_data);
              v632_acc += (v292_bc * v409_data);
              v632_acc += (v294_bc * v410_data);
              v632_acc += (v296_bc * v411_data);
              v632_acc += (v298_bc * v412_data);
              v632_acc += (v300_bc * v413_data);
              ir3.template select<16, 1>(128) = v632_acc;
              tensorforge::intel_esimd::simd<float, 16> v659_acc{};
              v659_acc += (v305_bc * v402_data);
              v659_acc += (v307_bc * v403_data);
              v659_acc += (v309_bc * v404_data);
              v659_acc += (v311_bc * v405_data);
              v659_acc += (v313_bc * v406_data);
              v659_acc += (v315_bc * v407_data);
              v659_acc += (v317_bc * v408_data);
              v659_acc += (v319_bc * v409_data);
              v659_acc += (v321_bc * v410_data);
              v659_acc += (v323_bc * v411_data);
              v659_acc += (v325_bc * v412_data);
              v659_acc += (v327_bc * v413_data);
              ir3.template select<16, 1>(144) = v659_acc;
              tensorforge::intel_esimd::simd<float, 16> v686_acc{};
              v686_acc += (v332_bc * v402_data);
              v686_acc += (v334_bc * v403_data);
              v686_acc += (v336_bc * v404_data);
              v686_acc += (v338_bc * v405_data);
              v686_acc += (v340_bc * v406_data);
              v686_acc += (v342_bc * v407_data);
              v686_acc += (v344_bc * v408_data);
              v686_acc += (v346_bc * v409_data);
              v686_acc += (v348_bc * v410_data);
              v686_acc += (v350_bc * v411_data);
              v686_acc += (v352_bc * v412_data);
              v686_acc += (v354_bc * v413_data);
              ir3.template select<16, 1>(160) = v686_acc;
              tensorforge::intel_esimd::simd<float, 16> v713_acc{};
              v713_acc += (v359_bc * v402_data);
              v713_acc += (v361_bc * v403_data);
              v713_acc += (v363_bc * v404_data);
              v713_acc += (v365_bc * v405_data);
              v713_acc += (v367_bc * v406_data);
              v713_acc += (v369_bc * v407_data);
              v713_acc += (v371_bc * v408_data);
              v713_acc += (v373_bc * v409_data);
              v713_acc += (v375_bc * v410_data);
              v713_acc += (v377_bc * v411_data);
              v713_acc += (v379_bc * v412_data);
              v713_acc += (v381_bc * v413_data);
              ir3.template select<16, 1>(176) = v713_acc;
              #pragma unroll
              for (int32_t v740_n1 = 0; v740_n1 < 12; ++v740_n1) {
                int32_t v741_a = v740_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v743_data(ir3.template select<6, 1>(v741_a));
                r3.template select<6, 1>(v741_a) = v743_data;
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v744_i1 = 0; v744_i1 < 12; ++v744_i1) {
                tensorforge::intel_esimd::simd<float, 6> v747_data(r3.template select<6, 1>((v744_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v744_i1 * 12))), v747_data);
              }
              // wait(r4 = load{g>r}(glb_m4););
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // r5 = +(r4 * s1) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v755_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v756_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v757_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v758_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v759_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v760_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v761_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v762_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v763_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v764_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v765_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v766_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v767_acc{};
              tensorforge::intel_esimd::simd<float, 16> v771_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v767_acc += ((static_cast<float>(v771_data[0])) * v755_data);
              v767_acc += ((static_cast<float>(v771_data[1])) * v756_data);
              v767_acc += ((static_cast<float>(v771_data[2])) * v757_data);
              v767_acc += ((static_cast<float>(v771_data[3])) * v758_data);
              v767_acc += ((static_cast<float>(v771_data[4])) * v759_data);
              v767_acc += ((static_cast<float>(v771_data[5])) * v760_data);
              v767_acc += ((static_cast<float>(v771_data[6])) * v761_data);
              v767_acc += ((static_cast<float>(v771_data[7])) * v762_data);
              v767_acc += ((static_cast<float>(v771_data[8])) * v763_data);
              v767_acc += ((static_cast<float>(v771_data[9])) * v764_data);
              v767_acc += ((static_cast<float>(v771_data[10])) * v765_data);
              v767_acc += ((static_cast<float>(v771_data[11])) * v766_data);
              ir5.template select<16, 1>(0) = v767_acc;
              tensorforge::intel_esimd::simd<float, 16> v796_acc{};
              tensorforge::intel_esimd::simd<float, 16> v798_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v796_acc += ((static_cast<float>(v798_data[0])) * v755_data);
              v796_acc += ((static_cast<float>(v798_data[1])) * v756_data);
              v796_acc += ((static_cast<float>(v798_data[2])) * v757_data);
              v796_acc += ((static_cast<float>(v798_data[3])) * v758_data);
              v796_acc += ((static_cast<float>(v798_data[4])) * v759_data);
              v796_acc += ((static_cast<float>(v798_data[5])) * v760_data);
              v796_acc += ((static_cast<float>(v798_data[6])) * v761_data);
              v796_acc += ((static_cast<float>(v798_data[7])) * v762_data);
              v796_acc += ((static_cast<float>(v798_data[8])) * v763_data);
              v796_acc += ((static_cast<float>(v798_data[9])) * v764_data);
              v796_acc += ((static_cast<float>(v798_data[10])) * v765_data);
              v796_acc += ((static_cast<float>(v798_data[11])) * v766_data);
              ir5.template select<16, 1>(16) = v796_acc;
              tensorforge::intel_esimd::simd<float, 16> v823_acc{};
              tensorforge::intel_esimd::simd<float, 16> v825_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v823_acc += ((static_cast<float>(v825_data[0])) * v755_data);
              v823_acc += ((static_cast<float>(v825_data[1])) * v756_data);
              v823_acc += ((static_cast<float>(v825_data[2])) * v757_data);
              v823_acc += ((static_cast<float>(v825_data[3])) * v758_data);
              v823_acc += ((static_cast<float>(v825_data[4])) * v759_data);
              v823_acc += ((static_cast<float>(v825_data[5])) * v760_data);
              v823_acc += ((static_cast<float>(v825_data[6])) * v761_data);
              v823_acc += ((static_cast<float>(v825_data[7])) * v762_data);
              v823_acc += ((static_cast<float>(v825_data[8])) * v763_data);
              v823_acc += ((static_cast<float>(v825_data[9])) * v764_data);
              v823_acc += ((static_cast<float>(v825_data[10])) * v765_data);
              v823_acc += ((static_cast<float>(v825_data[11])) * v766_data);
              ir5.template select<16, 1>(32) = v823_acc;
              tensorforge::intel_esimd::simd<float, 16> v850_acc{};
              tensorforge::intel_esimd::simd<float, 16> v852_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v850_acc += ((static_cast<float>(v852_data[0])) * v755_data);
              v850_acc += ((static_cast<float>(v852_data[1])) * v756_data);
              v850_acc += ((static_cast<float>(v852_data[2])) * v757_data);
              v850_acc += ((static_cast<float>(v852_data[3])) * v758_data);
              v850_acc += ((static_cast<float>(v852_data[4])) * v759_data);
              v850_acc += ((static_cast<float>(v852_data[5])) * v760_data);
              v850_acc += ((static_cast<float>(v852_data[6])) * v761_data);
              v850_acc += ((static_cast<float>(v852_data[7])) * v762_data);
              v850_acc += ((static_cast<float>(v852_data[8])) * v763_data);
              v850_acc += ((static_cast<float>(v852_data[9])) * v764_data);
              v850_acc += ((static_cast<float>(v852_data[10])) * v765_data);
              v850_acc += ((static_cast<float>(v852_data[11])) * v766_data);
              ir5.template select<16, 1>(48) = v850_acc;
              tensorforge::intel_esimd::simd<float, 16> v877_acc{};
              tensorforge::intel_esimd::simd<float, 16> v879_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v877_acc += ((static_cast<float>(v879_data[0])) * v755_data);
              v877_acc += ((static_cast<float>(v879_data[1])) * v756_data);
              v877_acc += ((static_cast<float>(v879_data[2])) * v757_data);
              v877_acc += ((static_cast<float>(v879_data[3])) * v758_data);
              v877_acc += ((static_cast<float>(v879_data[4])) * v759_data);
              v877_acc += ((static_cast<float>(v879_data[5])) * v760_data);
              v877_acc += ((static_cast<float>(v879_data[6])) * v761_data);
              v877_acc += ((static_cast<float>(v879_data[7])) * v762_data);
              v877_acc += ((static_cast<float>(v879_data[8])) * v763_data);
              v877_acc += ((static_cast<float>(v879_data[9])) * v764_data);
              v877_acc += ((static_cast<float>(v879_data[10])) * v765_data);
              v877_acc += ((static_cast<float>(v879_data[11])) * v766_data);
              ir5.template select<16, 1>(64) = v877_acc;
              tensorforge::intel_esimd::simd<float, 16> v904_acc{};
              tensorforge::intel_esimd::simd<float, 16> v906_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v904_acc += ((static_cast<float>(v906_data[0])) * v755_data);
              v904_acc += ((static_cast<float>(v906_data[1])) * v756_data);
              v904_acc += ((static_cast<float>(v906_data[2])) * v757_data);
              v904_acc += ((static_cast<float>(v906_data[3])) * v758_data);
              v904_acc += ((static_cast<float>(v906_data[4])) * v759_data);
              v904_acc += ((static_cast<float>(v906_data[5])) * v760_data);
              v904_acc += ((static_cast<float>(v906_data[6])) * v761_data);
              v904_acc += ((static_cast<float>(v906_data[7])) * v762_data);
              v904_acc += ((static_cast<float>(v906_data[8])) * v763_data);
              v904_acc += ((static_cast<float>(v906_data[9])) * v764_data);
              v904_acc += ((static_cast<float>(v906_data[10])) * v765_data);
              v904_acc += ((static_cast<float>(v906_data[11])) * v766_data);
              ir5.template select<16, 1>(80) = v904_acc;
              tensorforge::intel_esimd::simd<float, 16> v931_acc{};
              tensorforge::intel_esimd::simd<float, 16> v933_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v931_acc += ((static_cast<float>(v933_data[0])) * v755_data);
              v931_acc += ((static_cast<float>(v933_data[1])) * v756_data);
              v931_acc += ((static_cast<float>(v933_data[2])) * v757_data);
              v931_acc += ((static_cast<float>(v933_data[3])) * v758_data);
              v931_acc += ((static_cast<float>(v933_data[4])) * v759_data);
              v931_acc += ((static_cast<float>(v933_data[5])) * v760_data);
              v931_acc += ((static_cast<float>(v933_data[6])) * v761_data);
              v931_acc += ((static_cast<float>(v933_data[7])) * v762_data);
              v931_acc += ((static_cast<float>(v933_data[8])) * v763_data);
              v931_acc += ((static_cast<float>(v933_data[9])) * v764_data);
              v931_acc += ((static_cast<float>(v933_data[10])) * v765_data);
              v931_acc += ((static_cast<float>(v933_data[11])) * v766_data);
              ir5.template select<16, 1>(96) = v931_acc;
              tensorforge::intel_esimd::simd<float, 16> v958_acc{};
              tensorforge::intel_esimd::simd<float, 16> v960_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v958_acc += ((static_cast<float>(v960_data[0])) * v755_data);
              v958_acc += ((static_cast<float>(v960_data[1])) * v756_data);
              v958_acc += ((static_cast<float>(v960_data[2])) * v757_data);
              v958_acc += ((static_cast<float>(v960_data[3])) * v758_data);
              v958_acc += ((static_cast<float>(v960_data[4])) * v759_data);
              v958_acc += ((static_cast<float>(v960_data[5])) * v760_data);
              v958_acc += ((static_cast<float>(v960_data[6])) * v761_data);
              v958_acc += ((static_cast<float>(v960_data[7])) * v762_data);
              v958_acc += ((static_cast<float>(v960_data[8])) * v763_data);
              v958_acc += ((static_cast<float>(v960_data[9])) * v764_data);
              v958_acc += ((static_cast<float>(v960_data[10])) * v765_data);
              v958_acc += ((static_cast<float>(v960_data[11])) * v766_data);
              ir5.template select<16, 1>(112) = v958_acc;
              tensorforge::intel_esimd::simd<float, 16> v985_acc{};
              tensorforge::intel_esimd::simd<float, 16> v987_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v985_acc += ((static_cast<float>(v987_data[0])) * v755_data);
              v985_acc += ((static_cast<float>(v987_data[1])) * v756_data);
              v985_acc += ((static_cast<float>(v987_data[2])) * v757_data);
              v985_acc += ((static_cast<float>(v987_data[3])) * v758_data);
              v985_acc += ((static_cast<float>(v987_data[4])) * v759_data);
              v985_acc += ((static_cast<float>(v987_data[5])) * v760_data);
              v985_acc += ((static_cast<float>(v987_data[6])) * v761_data);
              v985_acc += ((static_cast<float>(v987_data[7])) * v762_data);
              v985_acc += ((static_cast<float>(v987_data[8])) * v763_data);
              v985_acc += ((static_cast<float>(v987_data[9])) * v764_data);
              v985_acc += ((static_cast<float>(v987_data[10])) * v765_data);
              v985_acc += ((static_cast<float>(v987_data[11])) * v766_data);
              ir5.template select<16, 1>(128) = v985_acc;
              tensorforge::intel_esimd::simd<float, 16> v1012_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1014_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v1012_acc += ((static_cast<float>(v1014_data[0])) * v755_data);
              v1012_acc += ((static_cast<float>(v1014_data[1])) * v756_data);
              v1012_acc += ((static_cast<float>(v1014_data[2])) * v757_data);
              v1012_acc += ((static_cast<float>(v1014_data[3])) * v758_data);
              v1012_acc += ((static_cast<float>(v1014_data[4])) * v759_data);
              v1012_acc += ((static_cast<float>(v1014_data[5])) * v760_data);
              v1012_acc += ((static_cast<float>(v1014_data[6])) * v761_data);
              v1012_acc += ((static_cast<float>(v1014_data[7])) * v762_data);
              v1012_acc += ((static_cast<float>(v1014_data[8])) * v763_data);
              v1012_acc += ((static_cast<float>(v1014_data[9])) * v764_data);
              v1012_acc += ((static_cast<float>(v1014_data[10])) * v765_data);
              v1012_acc += ((static_cast<float>(v1014_data[11])) * v766_data);
              ir5.template select<16, 1>(144) = v1012_acc;
              tensorforge::intel_esimd::simd<float, 16> v1039_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1041_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v1039_acc += ((static_cast<float>(v1041_data[0])) * v755_data);
              v1039_acc += ((static_cast<float>(v1041_data[1])) * v756_data);
              v1039_acc += ((static_cast<float>(v1041_data[2])) * v757_data);
              v1039_acc += ((static_cast<float>(v1041_data[3])) * v758_data);
              v1039_acc += ((static_cast<float>(v1041_data[4])) * v759_data);
              v1039_acc += ((static_cast<float>(v1041_data[5])) * v760_data);
              v1039_acc += ((static_cast<float>(v1041_data[6])) * v761_data);
              v1039_acc += ((static_cast<float>(v1041_data[7])) * v762_data);
              v1039_acc += ((static_cast<float>(v1041_data[8])) * v763_data);
              v1039_acc += ((static_cast<float>(v1041_data[9])) * v764_data);
              v1039_acc += ((static_cast<float>(v1041_data[10])) * v765_data);
              v1039_acc += ((static_cast<float>(v1041_data[11])) * v766_data);
              ir5.template select<16, 1>(160) = v1039_acc;
              tensorforge::intel_esimd::simd<float, 16> v1066_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1068_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v1066_acc += ((static_cast<float>(v1068_data[0])) * v755_data);
              v1066_acc += ((static_cast<float>(v1068_data[1])) * v756_data);
              v1066_acc += ((static_cast<float>(v1068_data[2])) * v757_data);
              v1066_acc += ((static_cast<float>(v1068_data[3])) * v758_data);
              v1066_acc += ((static_cast<float>(v1068_data[4])) * v759_data);
              v1066_acc += ((static_cast<float>(v1068_data[5])) * v760_data);
              v1066_acc += ((static_cast<float>(v1068_data[6])) * v761_data);
              v1066_acc += ((static_cast<float>(v1068_data[7])) * v762_data);
              v1066_acc += ((static_cast<float>(v1068_data[8])) * v763_data);
              v1066_acc += ((static_cast<float>(v1068_data[9])) * v764_data);
              v1066_acc += ((static_cast<float>(v1068_data[10])) * v765_data);
              v1066_acc += ((static_cast<float>(v1068_data[11])) * v766_data);
              ir5.template select<16, 1>(176) = v1066_acc;
              #pragma unroll
              for (int32_t v1093_n1 = 0; v1093_n1 < 12; ++v1093_n1) {
                int32_t v1094_a = v1093_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1096_data(ir5.template select<12, 1>(v1094_a));
                r5.template select<12, 1>(v1094_a) = v1096_data;
              }
              // glb_m3 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1097_i1 = 0; v1097_i1 < 12; ++v1097_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1100_data(r5.template select<12, 1>((v1097_i1 * 16)));
                v1100_data.copy_to(glb_m3 + ((v1097_i1 * 12)));
              }
            }
            tensorforge::prefetchRunsL2<288, 576, 288, 576>(&pf_glb_m0[0], &pf_glb_m1[0], &pf_glb_m2[0], &pf_glb_m4[0]);
          }
        }
      }
    });
  });
}


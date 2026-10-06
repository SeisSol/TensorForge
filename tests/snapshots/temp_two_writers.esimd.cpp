// === base name ===
kernel_d0b50539b16db57f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d0b50539b16db57f = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d0b50539b16db57f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d0b50539b16db57f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d0b50539b16db57f(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_d0b50539b16db57f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d0b50539b16db57f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d0b50539b16db57f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d0b50539b16db57f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4864 * sizeof(float)>(); {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4864}],"shared_bytes":19456,"shared_elements":4864,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
              const float *const __restrict__ glb_m4 = &m4[v12_batchId0 * 144 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
                tensorforge::intel_esimd::simd<float, 6> v31_data;
                v31_data.copy_from(glb_m0 + ((v26_i1 * 6)));
                r0.template select<6, 1>((v26_i1 * 16)) = v31_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v34_ld;
              v34_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v34_ld);
              tensorforge::intel_esimd::simd<float, 64> v35_ld;
              v35_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v35_ld);
              tensorforge::intel_esimd::simd<float, 16> v36_ld;
              v36_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v36_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
                tensorforge::intel_esimd::simd<float, 6> v43_data;
                v43_data.copy_from(glb_m2 + ((v38_i1 * 6)));
                r2.template select<6, 1>((v38_i1 * 16)) = v43_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v59_acc{};
              tensorforge::intel_esimd::simd<float, 16> v63_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              float v64_bc = static_cast<float>(v63_data[0]);
              v59_acc += (v64_bc * v47_data);
              float v66_bc = static_cast<float>(v63_data[1]);
              v59_acc += (v66_bc * v48_data);
              float v68_bc = static_cast<float>(v63_data[2]);
              v59_acc += (v68_bc * v49_data);
              float v70_bc = static_cast<float>(v63_data[3]);
              v59_acc += (v70_bc * v50_data);
              float v72_bc = static_cast<float>(v63_data[4]);
              v59_acc += (v72_bc * v51_data);
              float v74_bc = static_cast<float>(v63_data[5]);
              v59_acc += (v74_bc * v52_data);
              float v76_bc = static_cast<float>(v63_data[6]);
              v59_acc += (v76_bc * v53_data);
              float v78_bc = static_cast<float>(v63_data[7]);
              v59_acc += (v78_bc * v54_data);
              float v80_bc = static_cast<float>(v63_data[8]);
              v59_acc += (v80_bc * v55_data);
              float v82_bc = static_cast<float>(v63_data[9]);
              v59_acc += (v82_bc * v56_data);
              float v84_bc = static_cast<float>(v63_data[10]);
              v59_acc += (v84_bc * v57_data);
              float v86_bc = static_cast<float>(v63_data[11]);
              v59_acc += (v86_bc * v58_data);
              r1.template select<16, 1>(0) = v59_acc;
              tensorforge::intel_esimd::simd<float, 16> v88_acc{};
              tensorforge::intel_esimd::simd<float, 16> v90_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              float v91_bc = static_cast<float>(v90_data[0]);
              v88_acc += (v91_bc * v47_data);
              float v93_bc = static_cast<float>(v90_data[1]);
              v88_acc += (v93_bc * v48_data);
              float v95_bc = static_cast<float>(v90_data[2]);
              v88_acc += (v95_bc * v49_data);
              float v97_bc = static_cast<float>(v90_data[3]);
              v88_acc += (v97_bc * v50_data);
              float v99_bc = static_cast<float>(v90_data[4]);
              v88_acc += (v99_bc * v51_data);
              float v101_bc = static_cast<float>(v90_data[5]);
              v88_acc += (v101_bc * v52_data);
              float v103_bc = static_cast<float>(v90_data[6]);
              v88_acc += (v103_bc * v53_data);
              float v105_bc = static_cast<float>(v90_data[7]);
              v88_acc += (v105_bc * v54_data);
              float v107_bc = static_cast<float>(v90_data[8]);
              v88_acc += (v107_bc * v55_data);
              float v109_bc = static_cast<float>(v90_data[9]);
              v88_acc += (v109_bc * v56_data);
              float v111_bc = static_cast<float>(v90_data[10]);
              v88_acc += (v111_bc * v57_data);
              float v113_bc = static_cast<float>(v90_data[11]);
              v88_acc += (v113_bc * v58_data);
              r1.template select<16, 1>(16) = v88_acc;
              tensorforge::intel_esimd::simd<float, 16> v115_acc{};
              tensorforge::intel_esimd::simd<float, 16> v117_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              float v118_bc = static_cast<float>(v117_data[0]);
              v115_acc += (v118_bc * v47_data);
              float v120_bc = static_cast<float>(v117_data[1]);
              v115_acc += (v120_bc * v48_data);
              float v122_bc = static_cast<float>(v117_data[2]);
              v115_acc += (v122_bc * v49_data);
              float v124_bc = static_cast<float>(v117_data[3]);
              v115_acc += (v124_bc * v50_data);
              float v126_bc = static_cast<float>(v117_data[4]);
              v115_acc += (v126_bc * v51_data);
              float v128_bc = static_cast<float>(v117_data[5]);
              v115_acc += (v128_bc * v52_data);
              float v130_bc = static_cast<float>(v117_data[6]);
              v115_acc += (v130_bc * v53_data);
              float v132_bc = static_cast<float>(v117_data[7]);
              v115_acc += (v132_bc * v54_data);
              float v134_bc = static_cast<float>(v117_data[8]);
              v115_acc += (v134_bc * v55_data);
              float v136_bc = static_cast<float>(v117_data[9]);
              v115_acc += (v136_bc * v56_data);
              float v138_bc = static_cast<float>(v117_data[10]);
              v115_acc += (v138_bc * v57_data);
              float v140_bc = static_cast<float>(v117_data[11]);
              v115_acc += (v140_bc * v58_data);
              r1.template select<16, 1>(32) = v115_acc;
              tensorforge::intel_esimd::simd<float, 16> v142_acc{};
              tensorforge::intel_esimd::simd<float, 16> v144_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              float v145_bc = static_cast<float>(v144_data[0]);
              v142_acc += (v145_bc * v47_data);
              float v147_bc = static_cast<float>(v144_data[1]);
              v142_acc += (v147_bc * v48_data);
              float v149_bc = static_cast<float>(v144_data[2]);
              v142_acc += (v149_bc * v49_data);
              float v151_bc = static_cast<float>(v144_data[3]);
              v142_acc += (v151_bc * v50_data);
              float v153_bc = static_cast<float>(v144_data[4]);
              v142_acc += (v153_bc * v51_data);
              float v155_bc = static_cast<float>(v144_data[5]);
              v142_acc += (v155_bc * v52_data);
              float v157_bc = static_cast<float>(v144_data[6]);
              v142_acc += (v157_bc * v53_data);
              float v159_bc = static_cast<float>(v144_data[7]);
              v142_acc += (v159_bc * v54_data);
              float v161_bc = static_cast<float>(v144_data[8]);
              v142_acc += (v161_bc * v55_data);
              float v163_bc = static_cast<float>(v144_data[9]);
              v142_acc += (v163_bc * v56_data);
              float v165_bc = static_cast<float>(v144_data[10]);
              v142_acc += (v165_bc * v57_data);
              float v167_bc = static_cast<float>(v144_data[11]);
              v142_acc += (v167_bc * v58_data);
              r1.template select<16, 1>(48) = v142_acc;
              tensorforge::intel_esimd::simd<float, 16> v169_acc{};
              tensorforge::intel_esimd::simd<float, 16> v171_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              float v172_bc = static_cast<float>(v171_data[0]);
              v169_acc += (v172_bc * v47_data);
              float v174_bc = static_cast<float>(v171_data[1]);
              v169_acc += (v174_bc * v48_data);
              float v176_bc = static_cast<float>(v171_data[2]);
              v169_acc += (v176_bc * v49_data);
              float v178_bc = static_cast<float>(v171_data[3]);
              v169_acc += (v178_bc * v50_data);
              float v180_bc = static_cast<float>(v171_data[4]);
              v169_acc += (v180_bc * v51_data);
              float v182_bc = static_cast<float>(v171_data[5]);
              v169_acc += (v182_bc * v52_data);
              float v184_bc = static_cast<float>(v171_data[6]);
              v169_acc += (v184_bc * v53_data);
              float v186_bc = static_cast<float>(v171_data[7]);
              v169_acc += (v186_bc * v54_data);
              float v188_bc = static_cast<float>(v171_data[8]);
              v169_acc += (v188_bc * v55_data);
              float v190_bc = static_cast<float>(v171_data[9]);
              v169_acc += (v190_bc * v56_data);
              float v192_bc = static_cast<float>(v171_data[10]);
              v169_acc += (v192_bc * v57_data);
              float v194_bc = static_cast<float>(v171_data[11]);
              v169_acc += (v194_bc * v58_data);
              r1.template select<16, 1>(64) = v169_acc;
              tensorforge::intel_esimd::simd<float, 16> v196_acc{};
              tensorforge::intel_esimd::simd<float, 16> v198_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              float v199_bc = static_cast<float>(v198_data[0]);
              v196_acc += (v199_bc * v47_data);
              float v201_bc = static_cast<float>(v198_data[1]);
              v196_acc += (v201_bc * v48_data);
              float v203_bc = static_cast<float>(v198_data[2]);
              v196_acc += (v203_bc * v49_data);
              float v205_bc = static_cast<float>(v198_data[3]);
              v196_acc += (v205_bc * v50_data);
              float v207_bc = static_cast<float>(v198_data[4]);
              v196_acc += (v207_bc * v51_data);
              float v209_bc = static_cast<float>(v198_data[5]);
              v196_acc += (v209_bc * v52_data);
              float v211_bc = static_cast<float>(v198_data[6]);
              v196_acc += (v211_bc * v53_data);
              float v213_bc = static_cast<float>(v198_data[7]);
              v196_acc += (v213_bc * v54_data);
              float v215_bc = static_cast<float>(v198_data[8]);
              v196_acc += (v215_bc * v55_data);
              float v217_bc = static_cast<float>(v198_data[9]);
              v196_acc += (v217_bc * v56_data);
              float v219_bc = static_cast<float>(v198_data[10]);
              v196_acc += (v219_bc * v57_data);
              float v221_bc = static_cast<float>(v198_data[11]);
              v196_acc += (v221_bc * v58_data);
              r1.template select<16, 1>(80) = v196_acc;
              tensorforge::intel_esimd::simd<float, 16> v223_acc{};
              tensorforge::intel_esimd::simd<float, 16> v225_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              float v226_bc = static_cast<float>(v225_data[0]);
              v223_acc += (v226_bc * v47_data);
              float v228_bc = static_cast<float>(v225_data[1]);
              v223_acc += (v228_bc * v48_data);
              float v230_bc = static_cast<float>(v225_data[2]);
              v223_acc += (v230_bc * v49_data);
              float v232_bc = static_cast<float>(v225_data[3]);
              v223_acc += (v232_bc * v50_data);
              float v234_bc = static_cast<float>(v225_data[4]);
              v223_acc += (v234_bc * v51_data);
              float v236_bc = static_cast<float>(v225_data[5]);
              v223_acc += (v236_bc * v52_data);
              float v238_bc = static_cast<float>(v225_data[6]);
              v223_acc += (v238_bc * v53_data);
              float v240_bc = static_cast<float>(v225_data[7]);
              v223_acc += (v240_bc * v54_data);
              float v242_bc = static_cast<float>(v225_data[8]);
              v223_acc += (v242_bc * v55_data);
              float v244_bc = static_cast<float>(v225_data[9]);
              v223_acc += (v244_bc * v56_data);
              float v246_bc = static_cast<float>(v225_data[10]);
              v223_acc += (v246_bc * v57_data);
              float v248_bc = static_cast<float>(v225_data[11]);
              v223_acc += (v248_bc * v58_data);
              r1.template select<16, 1>(96) = v223_acc;
              tensorforge::intel_esimd::simd<float, 16> v250_acc{};
              tensorforge::intel_esimd::simd<float, 16> v252_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              float v253_bc = static_cast<float>(v252_data[0]);
              v250_acc += (v253_bc * v47_data);
              float v255_bc = static_cast<float>(v252_data[1]);
              v250_acc += (v255_bc * v48_data);
              float v257_bc = static_cast<float>(v252_data[2]);
              v250_acc += (v257_bc * v49_data);
              float v259_bc = static_cast<float>(v252_data[3]);
              v250_acc += (v259_bc * v50_data);
              float v261_bc = static_cast<float>(v252_data[4]);
              v250_acc += (v261_bc * v51_data);
              float v263_bc = static_cast<float>(v252_data[5]);
              v250_acc += (v263_bc * v52_data);
              float v265_bc = static_cast<float>(v252_data[6]);
              v250_acc += (v265_bc * v53_data);
              float v267_bc = static_cast<float>(v252_data[7]);
              v250_acc += (v267_bc * v54_data);
              float v269_bc = static_cast<float>(v252_data[8]);
              v250_acc += (v269_bc * v55_data);
              float v271_bc = static_cast<float>(v252_data[9]);
              v250_acc += (v271_bc * v56_data);
              float v273_bc = static_cast<float>(v252_data[10]);
              v250_acc += (v273_bc * v57_data);
              float v275_bc = static_cast<float>(v252_data[11]);
              v250_acc += (v275_bc * v58_data);
              r1.template select<16, 1>(112) = v250_acc;
              tensorforge::intel_esimd::simd<float, 16> v277_acc{};
              tensorforge::intel_esimd::simd<float, 16> v279_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              float v280_bc = static_cast<float>(v279_data[0]);
              v277_acc += (v280_bc * v47_data);
              float v282_bc = static_cast<float>(v279_data[1]);
              v277_acc += (v282_bc * v48_data);
              float v284_bc = static_cast<float>(v279_data[2]);
              v277_acc += (v284_bc * v49_data);
              float v286_bc = static_cast<float>(v279_data[3]);
              v277_acc += (v286_bc * v50_data);
              float v288_bc = static_cast<float>(v279_data[4]);
              v277_acc += (v288_bc * v51_data);
              float v290_bc = static_cast<float>(v279_data[5]);
              v277_acc += (v290_bc * v52_data);
              float v292_bc = static_cast<float>(v279_data[6]);
              v277_acc += (v292_bc * v53_data);
              float v294_bc = static_cast<float>(v279_data[7]);
              v277_acc += (v294_bc * v54_data);
              float v296_bc = static_cast<float>(v279_data[8]);
              v277_acc += (v296_bc * v55_data);
              float v298_bc = static_cast<float>(v279_data[9]);
              v277_acc += (v298_bc * v56_data);
              float v300_bc = static_cast<float>(v279_data[10]);
              v277_acc += (v300_bc * v57_data);
              float v302_bc = static_cast<float>(v279_data[11]);
              v277_acc += (v302_bc * v58_data);
              r1.template select<16, 1>(128) = v277_acc;
              tensorforge::intel_esimd::simd<float, 16> v304_acc{};
              tensorforge::intel_esimd::simd<float, 16> v306_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              float v307_bc = static_cast<float>(v306_data[0]);
              v304_acc += (v307_bc * v47_data);
              float v309_bc = static_cast<float>(v306_data[1]);
              v304_acc += (v309_bc * v48_data);
              float v311_bc = static_cast<float>(v306_data[2]);
              v304_acc += (v311_bc * v49_data);
              float v313_bc = static_cast<float>(v306_data[3]);
              v304_acc += (v313_bc * v50_data);
              float v315_bc = static_cast<float>(v306_data[4]);
              v304_acc += (v315_bc * v51_data);
              float v317_bc = static_cast<float>(v306_data[5]);
              v304_acc += (v317_bc * v52_data);
              float v319_bc = static_cast<float>(v306_data[6]);
              v304_acc += (v319_bc * v53_data);
              float v321_bc = static_cast<float>(v306_data[7]);
              v304_acc += (v321_bc * v54_data);
              float v323_bc = static_cast<float>(v306_data[8]);
              v304_acc += (v323_bc * v55_data);
              float v325_bc = static_cast<float>(v306_data[9]);
              v304_acc += (v325_bc * v56_data);
              float v327_bc = static_cast<float>(v306_data[10]);
              v304_acc += (v327_bc * v57_data);
              float v329_bc = static_cast<float>(v306_data[11]);
              v304_acc += (v329_bc * v58_data);
              r1.template select<16, 1>(144) = v304_acc;
              tensorforge::intel_esimd::simd<float, 16> v331_acc{};
              tensorforge::intel_esimd::simd<float, 16> v333_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              float v334_bc = static_cast<float>(v333_data[0]);
              v331_acc += (v334_bc * v47_data);
              float v336_bc = static_cast<float>(v333_data[1]);
              v331_acc += (v336_bc * v48_data);
              float v338_bc = static_cast<float>(v333_data[2]);
              v331_acc += (v338_bc * v49_data);
              float v340_bc = static_cast<float>(v333_data[3]);
              v331_acc += (v340_bc * v50_data);
              float v342_bc = static_cast<float>(v333_data[4]);
              v331_acc += (v342_bc * v51_data);
              float v344_bc = static_cast<float>(v333_data[5]);
              v331_acc += (v344_bc * v52_data);
              float v346_bc = static_cast<float>(v333_data[6]);
              v331_acc += (v346_bc * v53_data);
              float v348_bc = static_cast<float>(v333_data[7]);
              v331_acc += (v348_bc * v54_data);
              float v350_bc = static_cast<float>(v333_data[8]);
              v331_acc += (v350_bc * v55_data);
              float v352_bc = static_cast<float>(v333_data[9]);
              v331_acc += (v352_bc * v56_data);
              float v354_bc = static_cast<float>(v333_data[10]);
              v331_acc += (v354_bc * v57_data);
              float v356_bc = static_cast<float>(v333_data[11]);
              v331_acc += (v356_bc * v58_data);
              r1.template select<16, 1>(160) = v331_acc;
              tensorforge::intel_esimd::simd<float, 16> v358_acc{};
              tensorforge::intel_esimd::simd<float, 16> v360_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              float v361_bc = static_cast<float>(v360_data[0]);
              v358_acc += (v361_bc * v47_data);
              float v363_bc = static_cast<float>(v360_data[1]);
              v358_acc += (v363_bc * v48_data);
              float v365_bc = static_cast<float>(v360_data[2]);
              v358_acc += (v365_bc * v49_data);
              float v367_bc = static_cast<float>(v360_data[3]);
              v358_acc += (v367_bc * v50_data);
              float v369_bc = static_cast<float>(v360_data[4]);
              v358_acc += (v369_bc * v51_data);
              float v371_bc = static_cast<float>(v360_data[5]);
              v358_acc += (v371_bc * v52_data);
              float v373_bc = static_cast<float>(v360_data[6]);
              v358_acc += (v373_bc * v53_data);
              float v375_bc = static_cast<float>(v360_data[7]);
              v358_acc += (v375_bc * v54_data);
              float v377_bc = static_cast<float>(v360_data[8]);
              v358_acc += (v377_bc * v55_data);
              float v379_bc = static_cast<float>(v360_data[9]);
              v358_acc += (v379_bc * v56_data);
              float v381_bc = static_cast<float>(v360_data[10]);
              v358_acc += (v381_bc * v57_data);
              float v383_bc = static_cast<float>(v360_data[11]);
              v358_acc += (v383_bc * v58_data);
              r1.template select<16, 1>(176) = v358_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v385_i1 = 0; v385_i1 < 12; ++v385_i1) {
                tensorforge::intel_esimd::simd<float, 6> v388_data(r1.template select<6, 1>((v385_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v385_i1 * 12)), v388_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v394_i1 = 0; v394_i1 < 12; ++v394_i1) {
                tensorforge::intel_esimd::simd<float, 12> v399_data;
                v399_data.copy_from(glb_m4 + ((v394_i1 * 12)));
                r4.template select<12, 1>((v394_i1 * 16)) = v399_data;
              }
              // wait(r2 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s0)
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v404_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v405_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v406_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v407_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v408_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v409_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v410_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v411_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v412_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v413_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v414_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v415_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v416_acc{};
              v416_acc += (v64_bc * v404_data);
              v416_acc += (v66_bc * v405_data);
              v416_acc += (v68_bc * v406_data);
              v416_acc += (v70_bc * v407_data);
              v416_acc += (v72_bc * v408_data);
              v416_acc += (v74_bc * v409_data);
              v416_acc += (v76_bc * v410_data);
              v416_acc += (v78_bc * v411_data);
              v416_acc += (v80_bc * v412_data);
              v416_acc += (v82_bc * v413_data);
              v416_acc += (v84_bc * v414_data);
              v416_acc += (v86_bc * v415_data);
              ir3.template select<16, 1>(0) = v416_acc;
              tensorforge::intel_esimd::simd<float, 16> v445_acc{};
              v445_acc += (v91_bc * v404_data);
              v445_acc += (v93_bc * v405_data);
              v445_acc += (v95_bc * v406_data);
              v445_acc += (v97_bc * v407_data);
              v445_acc += (v99_bc * v408_data);
              v445_acc += (v101_bc * v409_data);
              v445_acc += (v103_bc * v410_data);
              v445_acc += (v105_bc * v411_data);
              v445_acc += (v107_bc * v412_data);
              v445_acc += (v109_bc * v413_data);
              v445_acc += (v111_bc * v414_data);
              v445_acc += (v113_bc * v415_data);
              ir3.template select<16, 1>(16) = v445_acc;
              tensorforge::intel_esimd::simd<float, 16> v472_acc{};
              v472_acc += (v118_bc * v404_data);
              v472_acc += (v120_bc * v405_data);
              v472_acc += (v122_bc * v406_data);
              v472_acc += (v124_bc * v407_data);
              v472_acc += (v126_bc * v408_data);
              v472_acc += (v128_bc * v409_data);
              v472_acc += (v130_bc * v410_data);
              v472_acc += (v132_bc * v411_data);
              v472_acc += (v134_bc * v412_data);
              v472_acc += (v136_bc * v413_data);
              v472_acc += (v138_bc * v414_data);
              v472_acc += (v140_bc * v415_data);
              ir3.template select<16, 1>(32) = v472_acc;
              tensorforge::intel_esimd::simd<float, 16> v499_acc{};
              v499_acc += (v145_bc * v404_data);
              v499_acc += (v147_bc * v405_data);
              v499_acc += (v149_bc * v406_data);
              v499_acc += (v151_bc * v407_data);
              v499_acc += (v153_bc * v408_data);
              v499_acc += (v155_bc * v409_data);
              v499_acc += (v157_bc * v410_data);
              v499_acc += (v159_bc * v411_data);
              v499_acc += (v161_bc * v412_data);
              v499_acc += (v163_bc * v413_data);
              v499_acc += (v165_bc * v414_data);
              v499_acc += (v167_bc * v415_data);
              ir3.template select<16, 1>(48) = v499_acc;
              tensorforge::intel_esimd::simd<float, 16> v526_acc{};
              v526_acc += (v172_bc * v404_data);
              v526_acc += (v174_bc * v405_data);
              v526_acc += (v176_bc * v406_data);
              v526_acc += (v178_bc * v407_data);
              v526_acc += (v180_bc * v408_data);
              v526_acc += (v182_bc * v409_data);
              v526_acc += (v184_bc * v410_data);
              v526_acc += (v186_bc * v411_data);
              v526_acc += (v188_bc * v412_data);
              v526_acc += (v190_bc * v413_data);
              v526_acc += (v192_bc * v414_data);
              v526_acc += (v194_bc * v415_data);
              ir3.template select<16, 1>(64) = v526_acc;
              tensorforge::intel_esimd::simd<float, 16> v553_acc{};
              v553_acc += (v199_bc * v404_data);
              v553_acc += (v201_bc * v405_data);
              v553_acc += (v203_bc * v406_data);
              v553_acc += (v205_bc * v407_data);
              v553_acc += (v207_bc * v408_data);
              v553_acc += (v209_bc * v409_data);
              v553_acc += (v211_bc * v410_data);
              v553_acc += (v213_bc * v411_data);
              v553_acc += (v215_bc * v412_data);
              v553_acc += (v217_bc * v413_data);
              v553_acc += (v219_bc * v414_data);
              v553_acc += (v221_bc * v415_data);
              ir3.template select<16, 1>(80) = v553_acc;
              tensorforge::intel_esimd::simd<float, 16> v580_acc{};
              v580_acc += (v226_bc * v404_data);
              v580_acc += (v228_bc * v405_data);
              v580_acc += (v230_bc * v406_data);
              v580_acc += (v232_bc * v407_data);
              v580_acc += (v234_bc * v408_data);
              v580_acc += (v236_bc * v409_data);
              v580_acc += (v238_bc * v410_data);
              v580_acc += (v240_bc * v411_data);
              v580_acc += (v242_bc * v412_data);
              v580_acc += (v244_bc * v413_data);
              v580_acc += (v246_bc * v414_data);
              v580_acc += (v248_bc * v415_data);
              ir3.template select<16, 1>(96) = v580_acc;
              tensorforge::intel_esimd::simd<float, 16> v607_acc{};
              v607_acc += (v253_bc * v404_data);
              v607_acc += (v255_bc * v405_data);
              v607_acc += (v257_bc * v406_data);
              v607_acc += (v259_bc * v407_data);
              v607_acc += (v261_bc * v408_data);
              v607_acc += (v263_bc * v409_data);
              v607_acc += (v265_bc * v410_data);
              v607_acc += (v267_bc * v411_data);
              v607_acc += (v269_bc * v412_data);
              v607_acc += (v271_bc * v413_data);
              v607_acc += (v273_bc * v414_data);
              v607_acc += (v275_bc * v415_data);
              ir3.template select<16, 1>(112) = v607_acc;
              tensorforge::intel_esimd::simd<float, 16> v634_acc{};
              v634_acc += (v280_bc * v404_data);
              v634_acc += (v282_bc * v405_data);
              v634_acc += (v284_bc * v406_data);
              v634_acc += (v286_bc * v407_data);
              v634_acc += (v288_bc * v408_data);
              v634_acc += (v290_bc * v409_data);
              v634_acc += (v292_bc * v410_data);
              v634_acc += (v294_bc * v411_data);
              v634_acc += (v296_bc * v412_data);
              v634_acc += (v298_bc * v413_data);
              v634_acc += (v300_bc * v414_data);
              v634_acc += (v302_bc * v415_data);
              ir3.template select<16, 1>(128) = v634_acc;
              tensorforge::intel_esimd::simd<float, 16> v661_acc{};
              v661_acc += (v307_bc * v404_data);
              v661_acc += (v309_bc * v405_data);
              v661_acc += (v311_bc * v406_data);
              v661_acc += (v313_bc * v407_data);
              v661_acc += (v315_bc * v408_data);
              v661_acc += (v317_bc * v409_data);
              v661_acc += (v319_bc * v410_data);
              v661_acc += (v321_bc * v411_data);
              v661_acc += (v323_bc * v412_data);
              v661_acc += (v325_bc * v413_data);
              v661_acc += (v327_bc * v414_data);
              v661_acc += (v329_bc * v415_data);
              ir3.template select<16, 1>(144) = v661_acc;
              tensorforge::intel_esimd::simd<float, 16> v688_acc{};
              v688_acc += (v334_bc * v404_data);
              v688_acc += (v336_bc * v405_data);
              v688_acc += (v338_bc * v406_data);
              v688_acc += (v340_bc * v407_data);
              v688_acc += (v342_bc * v408_data);
              v688_acc += (v344_bc * v409_data);
              v688_acc += (v346_bc * v410_data);
              v688_acc += (v348_bc * v411_data);
              v688_acc += (v350_bc * v412_data);
              v688_acc += (v352_bc * v413_data);
              v688_acc += (v354_bc * v414_data);
              v688_acc += (v356_bc * v415_data);
              ir3.template select<16, 1>(160) = v688_acc;
              tensorforge::intel_esimd::simd<float, 16> v715_acc{};
              v715_acc += (v361_bc * v404_data);
              v715_acc += (v363_bc * v405_data);
              v715_acc += (v365_bc * v406_data);
              v715_acc += (v367_bc * v407_data);
              v715_acc += (v369_bc * v408_data);
              v715_acc += (v371_bc * v409_data);
              v715_acc += (v373_bc * v410_data);
              v715_acc += (v375_bc * v411_data);
              v715_acc += (v377_bc * v412_data);
              v715_acc += (v379_bc * v413_data);
              v715_acc += (v381_bc * v414_data);
              v715_acc += (v383_bc * v415_data);
              ir3.template select<16, 1>(176) = v715_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v742_n1 = 0; v742_n1 < 12; ++v742_n1) {
                int32_t v743_a = v742_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v745_data(ir3.template select<6, 1>(v743_a));
                r3.template select<6, 1>(v743_a) = v745_data;
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v746_i1 = 0; v746_i1 < 12; ++v746_i1) {
                tensorforge::intel_esimd::simd<float, 6> v749_data(r3.template select<6, 1>((v746_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v746_i1 * 12))), v749_data);
              }
              // wait(r4 = load{g>r}(glb_m4););
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // ir5 = +(r4 * s1)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v757_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v758_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v759_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v760_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v761_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v762_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v763_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v764_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v765_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v766_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v767_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v768_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v769_acc{};
              tensorforge::intel_esimd::simd<float, 16> v773_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v769_acc += ((static_cast<float>(v773_data[0])) * v757_data);
              v769_acc += ((static_cast<float>(v773_data[1])) * v758_data);
              v769_acc += ((static_cast<float>(v773_data[2])) * v759_data);
              v769_acc += ((static_cast<float>(v773_data[3])) * v760_data);
              v769_acc += ((static_cast<float>(v773_data[4])) * v761_data);
              v769_acc += ((static_cast<float>(v773_data[5])) * v762_data);
              v769_acc += ((static_cast<float>(v773_data[6])) * v763_data);
              v769_acc += ((static_cast<float>(v773_data[7])) * v764_data);
              v769_acc += ((static_cast<float>(v773_data[8])) * v765_data);
              v769_acc += ((static_cast<float>(v773_data[9])) * v766_data);
              v769_acc += ((static_cast<float>(v773_data[10])) * v767_data);
              v769_acc += ((static_cast<float>(v773_data[11])) * v768_data);
              ir5.template select<16, 1>(0) = v769_acc;
              tensorforge::intel_esimd::simd<float, 16> v798_acc{};
              tensorforge::intel_esimd::simd<float, 16> v800_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v798_acc += ((static_cast<float>(v800_data[0])) * v757_data);
              v798_acc += ((static_cast<float>(v800_data[1])) * v758_data);
              v798_acc += ((static_cast<float>(v800_data[2])) * v759_data);
              v798_acc += ((static_cast<float>(v800_data[3])) * v760_data);
              v798_acc += ((static_cast<float>(v800_data[4])) * v761_data);
              v798_acc += ((static_cast<float>(v800_data[5])) * v762_data);
              v798_acc += ((static_cast<float>(v800_data[6])) * v763_data);
              v798_acc += ((static_cast<float>(v800_data[7])) * v764_data);
              v798_acc += ((static_cast<float>(v800_data[8])) * v765_data);
              v798_acc += ((static_cast<float>(v800_data[9])) * v766_data);
              v798_acc += ((static_cast<float>(v800_data[10])) * v767_data);
              v798_acc += ((static_cast<float>(v800_data[11])) * v768_data);
              ir5.template select<16, 1>(16) = v798_acc;
              tensorforge::intel_esimd::simd<float, 16> v825_acc{};
              tensorforge::intel_esimd::simd<float, 16> v827_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v825_acc += ((static_cast<float>(v827_data[0])) * v757_data);
              v825_acc += ((static_cast<float>(v827_data[1])) * v758_data);
              v825_acc += ((static_cast<float>(v827_data[2])) * v759_data);
              v825_acc += ((static_cast<float>(v827_data[3])) * v760_data);
              v825_acc += ((static_cast<float>(v827_data[4])) * v761_data);
              v825_acc += ((static_cast<float>(v827_data[5])) * v762_data);
              v825_acc += ((static_cast<float>(v827_data[6])) * v763_data);
              v825_acc += ((static_cast<float>(v827_data[7])) * v764_data);
              v825_acc += ((static_cast<float>(v827_data[8])) * v765_data);
              v825_acc += ((static_cast<float>(v827_data[9])) * v766_data);
              v825_acc += ((static_cast<float>(v827_data[10])) * v767_data);
              v825_acc += ((static_cast<float>(v827_data[11])) * v768_data);
              ir5.template select<16, 1>(32) = v825_acc;
              tensorforge::intel_esimd::simd<float, 16> v852_acc{};
              tensorforge::intel_esimd::simd<float, 16> v854_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v852_acc += ((static_cast<float>(v854_data[0])) * v757_data);
              v852_acc += ((static_cast<float>(v854_data[1])) * v758_data);
              v852_acc += ((static_cast<float>(v854_data[2])) * v759_data);
              v852_acc += ((static_cast<float>(v854_data[3])) * v760_data);
              v852_acc += ((static_cast<float>(v854_data[4])) * v761_data);
              v852_acc += ((static_cast<float>(v854_data[5])) * v762_data);
              v852_acc += ((static_cast<float>(v854_data[6])) * v763_data);
              v852_acc += ((static_cast<float>(v854_data[7])) * v764_data);
              v852_acc += ((static_cast<float>(v854_data[8])) * v765_data);
              v852_acc += ((static_cast<float>(v854_data[9])) * v766_data);
              v852_acc += ((static_cast<float>(v854_data[10])) * v767_data);
              v852_acc += ((static_cast<float>(v854_data[11])) * v768_data);
              ir5.template select<16, 1>(48) = v852_acc;
              tensorforge::intel_esimd::simd<float, 16> v879_acc{};
              tensorforge::intel_esimd::simd<float, 16> v881_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v879_acc += ((static_cast<float>(v881_data[0])) * v757_data);
              v879_acc += ((static_cast<float>(v881_data[1])) * v758_data);
              v879_acc += ((static_cast<float>(v881_data[2])) * v759_data);
              v879_acc += ((static_cast<float>(v881_data[3])) * v760_data);
              v879_acc += ((static_cast<float>(v881_data[4])) * v761_data);
              v879_acc += ((static_cast<float>(v881_data[5])) * v762_data);
              v879_acc += ((static_cast<float>(v881_data[6])) * v763_data);
              v879_acc += ((static_cast<float>(v881_data[7])) * v764_data);
              v879_acc += ((static_cast<float>(v881_data[8])) * v765_data);
              v879_acc += ((static_cast<float>(v881_data[9])) * v766_data);
              v879_acc += ((static_cast<float>(v881_data[10])) * v767_data);
              v879_acc += ((static_cast<float>(v881_data[11])) * v768_data);
              ir5.template select<16, 1>(64) = v879_acc;
              tensorforge::intel_esimd::simd<float, 16> v906_acc{};
              tensorforge::intel_esimd::simd<float, 16> v908_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v906_acc += ((static_cast<float>(v908_data[0])) * v757_data);
              v906_acc += ((static_cast<float>(v908_data[1])) * v758_data);
              v906_acc += ((static_cast<float>(v908_data[2])) * v759_data);
              v906_acc += ((static_cast<float>(v908_data[3])) * v760_data);
              v906_acc += ((static_cast<float>(v908_data[4])) * v761_data);
              v906_acc += ((static_cast<float>(v908_data[5])) * v762_data);
              v906_acc += ((static_cast<float>(v908_data[6])) * v763_data);
              v906_acc += ((static_cast<float>(v908_data[7])) * v764_data);
              v906_acc += ((static_cast<float>(v908_data[8])) * v765_data);
              v906_acc += ((static_cast<float>(v908_data[9])) * v766_data);
              v906_acc += ((static_cast<float>(v908_data[10])) * v767_data);
              v906_acc += ((static_cast<float>(v908_data[11])) * v768_data);
              ir5.template select<16, 1>(80) = v906_acc;
              tensorforge::intel_esimd::simd<float, 16> v933_acc{};
              tensorforge::intel_esimd::simd<float, 16> v935_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v933_acc += ((static_cast<float>(v935_data[0])) * v757_data);
              v933_acc += ((static_cast<float>(v935_data[1])) * v758_data);
              v933_acc += ((static_cast<float>(v935_data[2])) * v759_data);
              v933_acc += ((static_cast<float>(v935_data[3])) * v760_data);
              v933_acc += ((static_cast<float>(v935_data[4])) * v761_data);
              v933_acc += ((static_cast<float>(v935_data[5])) * v762_data);
              v933_acc += ((static_cast<float>(v935_data[6])) * v763_data);
              v933_acc += ((static_cast<float>(v935_data[7])) * v764_data);
              v933_acc += ((static_cast<float>(v935_data[8])) * v765_data);
              v933_acc += ((static_cast<float>(v935_data[9])) * v766_data);
              v933_acc += ((static_cast<float>(v935_data[10])) * v767_data);
              v933_acc += ((static_cast<float>(v935_data[11])) * v768_data);
              ir5.template select<16, 1>(96) = v933_acc;
              tensorforge::intel_esimd::simd<float, 16> v960_acc{};
              tensorforge::intel_esimd::simd<float, 16> v962_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v960_acc += ((static_cast<float>(v962_data[0])) * v757_data);
              v960_acc += ((static_cast<float>(v962_data[1])) * v758_data);
              v960_acc += ((static_cast<float>(v962_data[2])) * v759_data);
              v960_acc += ((static_cast<float>(v962_data[3])) * v760_data);
              v960_acc += ((static_cast<float>(v962_data[4])) * v761_data);
              v960_acc += ((static_cast<float>(v962_data[5])) * v762_data);
              v960_acc += ((static_cast<float>(v962_data[6])) * v763_data);
              v960_acc += ((static_cast<float>(v962_data[7])) * v764_data);
              v960_acc += ((static_cast<float>(v962_data[8])) * v765_data);
              v960_acc += ((static_cast<float>(v962_data[9])) * v766_data);
              v960_acc += ((static_cast<float>(v962_data[10])) * v767_data);
              v960_acc += ((static_cast<float>(v962_data[11])) * v768_data);
              ir5.template select<16, 1>(112) = v960_acc;
              tensorforge::intel_esimd::simd<float, 16> v987_acc{};
              tensorforge::intel_esimd::simd<float, 16> v989_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v987_acc += ((static_cast<float>(v989_data[0])) * v757_data);
              v987_acc += ((static_cast<float>(v989_data[1])) * v758_data);
              v987_acc += ((static_cast<float>(v989_data[2])) * v759_data);
              v987_acc += ((static_cast<float>(v989_data[3])) * v760_data);
              v987_acc += ((static_cast<float>(v989_data[4])) * v761_data);
              v987_acc += ((static_cast<float>(v989_data[5])) * v762_data);
              v987_acc += ((static_cast<float>(v989_data[6])) * v763_data);
              v987_acc += ((static_cast<float>(v989_data[7])) * v764_data);
              v987_acc += ((static_cast<float>(v989_data[8])) * v765_data);
              v987_acc += ((static_cast<float>(v989_data[9])) * v766_data);
              v987_acc += ((static_cast<float>(v989_data[10])) * v767_data);
              v987_acc += ((static_cast<float>(v989_data[11])) * v768_data);
              ir5.template select<16, 1>(128) = v987_acc;
              tensorforge::intel_esimd::simd<float, 16> v1014_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1016_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v1014_acc += ((static_cast<float>(v1016_data[0])) * v757_data);
              v1014_acc += ((static_cast<float>(v1016_data[1])) * v758_data);
              v1014_acc += ((static_cast<float>(v1016_data[2])) * v759_data);
              v1014_acc += ((static_cast<float>(v1016_data[3])) * v760_data);
              v1014_acc += ((static_cast<float>(v1016_data[4])) * v761_data);
              v1014_acc += ((static_cast<float>(v1016_data[5])) * v762_data);
              v1014_acc += ((static_cast<float>(v1016_data[6])) * v763_data);
              v1014_acc += ((static_cast<float>(v1016_data[7])) * v764_data);
              v1014_acc += ((static_cast<float>(v1016_data[8])) * v765_data);
              v1014_acc += ((static_cast<float>(v1016_data[9])) * v766_data);
              v1014_acc += ((static_cast<float>(v1016_data[10])) * v767_data);
              v1014_acc += ((static_cast<float>(v1016_data[11])) * v768_data);
              ir5.template select<16, 1>(144) = v1014_acc;
              tensorforge::intel_esimd::simd<float, 16> v1041_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1043_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v1041_acc += ((static_cast<float>(v1043_data[0])) * v757_data);
              v1041_acc += ((static_cast<float>(v1043_data[1])) * v758_data);
              v1041_acc += ((static_cast<float>(v1043_data[2])) * v759_data);
              v1041_acc += ((static_cast<float>(v1043_data[3])) * v760_data);
              v1041_acc += ((static_cast<float>(v1043_data[4])) * v761_data);
              v1041_acc += ((static_cast<float>(v1043_data[5])) * v762_data);
              v1041_acc += ((static_cast<float>(v1043_data[6])) * v763_data);
              v1041_acc += ((static_cast<float>(v1043_data[7])) * v764_data);
              v1041_acc += ((static_cast<float>(v1043_data[8])) * v765_data);
              v1041_acc += ((static_cast<float>(v1043_data[9])) * v766_data);
              v1041_acc += ((static_cast<float>(v1043_data[10])) * v767_data);
              v1041_acc += ((static_cast<float>(v1043_data[11])) * v768_data);
              ir5.template select<16, 1>(160) = v1041_acc;
              tensorforge::intel_esimd::simd<float, 16> v1068_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1070_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v1068_acc += ((static_cast<float>(v1070_data[0])) * v757_data);
              v1068_acc += ((static_cast<float>(v1070_data[1])) * v758_data);
              v1068_acc += ((static_cast<float>(v1070_data[2])) * v759_data);
              v1068_acc += ((static_cast<float>(v1070_data[3])) * v760_data);
              v1068_acc += ((static_cast<float>(v1070_data[4])) * v761_data);
              v1068_acc += ((static_cast<float>(v1070_data[5])) * v762_data);
              v1068_acc += ((static_cast<float>(v1070_data[6])) * v763_data);
              v1068_acc += ((static_cast<float>(v1070_data[7])) * v764_data);
              v1068_acc += ((static_cast<float>(v1070_data[8])) * v765_data);
              v1068_acc += ((static_cast<float>(v1070_data[9])) * v766_data);
              v1068_acc += ((static_cast<float>(v1070_data[10])) * v767_data);
              v1068_acc += ((static_cast<float>(v1070_data[11])) * v768_data);
              ir5.template select<16, 1>(176) = v1068_acc;
              // r5 = ir5
              #pragma unroll
              for (int32_t v1095_n1 = 0; v1095_n1 < 12; ++v1095_n1) {
                int32_t v1096_a = v1095_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1098_data(ir5.template select<12, 1>(v1096_a));
                r5.template select<12, 1>(v1096_a) = v1098_data;
              }
              // glb_m3 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1099_i1 = 0; v1099_i1 < 12; ++v1099_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1102_data(r5.template select<12, 1>((v1099_i1 * 16)));
                v1102_data.copy_to(glb_m3 + ((v1099_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


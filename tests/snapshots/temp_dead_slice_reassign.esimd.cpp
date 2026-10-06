// === base name ===
kernel_e0d0163942d0380c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e0d0163942d0380c = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e0d0163942d0380c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e0d0163942d0380c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e0d0163942d0380c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_e0d0163942d0380c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e0d0163942d0380c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e0d0163942d0380c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e0d0163942d0380c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 6×12(6×12) {0..6}×{0..12} strided
        //   m4 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{6..12}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j] = m2[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m3[i,j]
        //   m4[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4864}],"shared_bytes":19456,"shared_elements":4864,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
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
              const float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 72 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v12_batchId0 * 144 + 0 + m4_extraOffset];
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
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v385_i1 * 12))), v388_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v395_i1 = 0; v395_i1 < 12; ++v395_i1) {
                tensorforge::intel_esimd::simd<float, 6> v400_data;
                v400_data.copy_from(glb_m3 + ((v395_i1 * 6)));
                r4.template select<6, 1>((v395_i1 * 16)) = v400_data;
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
              v417_acc += (v64_bc * v405_data);
              v417_acc += (v66_bc * v406_data);
              v417_acc += (v68_bc * v407_data);
              v417_acc += (v70_bc * v408_data);
              v417_acc += (v72_bc * v409_data);
              v417_acc += (v74_bc * v410_data);
              v417_acc += (v76_bc * v411_data);
              v417_acc += (v78_bc * v412_data);
              v417_acc += (v80_bc * v413_data);
              v417_acc += (v82_bc * v414_data);
              v417_acc += (v84_bc * v415_data);
              v417_acc += (v86_bc * v416_data);
              ir3.template select<16, 1>(0) = v417_acc;
              tensorforge::intel_esimd::simd<float, 16> v446_acc{};
              v446_acc += (v91_bc * v405_data);
              v446_acc += (v93_bc * v406_data);
              v446_acc += (v95_bc * v407_data);
              v446_acc += (v97_bc * v408_data);
              v446_acc += (v99_bc * v409_data);
              v446_acc += (v101_bc * v410_data);
              v446_acc += (v103_bc * v411_data);
              v446_acc += (v105_bc * v412_data);
              v446_acc += (v107_bc * v413_data);
              v446_acc += (v109_bc * v414_data);
              v446_acc += (v111_bc * v415_data);
              v446_acc += (v113_bc * v416_data);
              ir3.template select<16, 1>(16) = v446_acc;
              tensorforge::intel_esimd::simd<float, 16> v473_acc{};
              v473_acc += (v118_bc * v405_data);
              v473_acc += (v120_bc * v406_data);
              v473_acc += (v122_bc * v407_data);
              v473_acc += (v124_bc * v408_data);
              v473_acc += (v126_bc * v409_data);
              v473_acc += (v128_bc * v410_data);
              v473_acc += (v130_bc * v411_data);
              v473_acc += (v132_bc * v412_data);
              v473_acc += (v134_bc * v413_data);
              v473_acc += (v136_bc * v414_data);
              v473_acc += (v138_bc * v415_data);
              v473_acc += (v140_bc * v416_data);
              ir3.template select<16, 1>(32) = v473_acc;
              tensorforge::intel_esimd::simd<float, 16> v500_acc{};
              v500_acc += (v145_bc * v405_data);
              v500_acc += (v147_bc * v406_data);
              v500_acc += (v149_bc * v407_data);
              v500_acc += (v151_bc * v408_data);
              v500_acc += (v153_bc * v409_data);
              v500_acc += (v155_bc * v410_data);
              v500_acc += (v157_bc * v411_data);
              v500_acc += (v159_bc * v412_data);
              v500_acc += (v161_bc * v413_data);
              v500_acc += (v163_bc * v414_data);
              v500_acc += (v165_bc * v415_data);
              v500_acc += (v167_bc * v416_data);
              ir3.template select<16, 1>(48) = v500_acc;
              tensorforge::intel_esimd::simd<float, 16> v527_acc{};
              v527_acc += (v172_bc * v405_data);
              v527_acc += (v174_bc * v406_data);
              v527_acc += (v176_bc * v407_data);
              v527_acc += (v178_bc * v408_data);
              v527_acc += (v180_bc * v409_data);
              v527_acc += (v182_bc * v410_data);
              v527_acc += (v184_bc * v411_data);
              v527_acc += (v186_bc * v412_data);
              v527_acc += (v188_bc * v413_data);
              v527_acc += (v190_bc * v414_data);
              v527_acc += (v192_bc * v415_data);
              v527_acc += (v194_bc * v416_data);
              ir3.template select<16, 1>(64) = v527_acc;
              tensorforge::intel_esimd::simd<float, 16> v554_acc{};
              v554_acc += (v199_bc * v405_data);
              v554_acc += (v201_bc * v406_data);
              v554_acc += (v203_bc * v407_data);
              v554_acc += (v205_bc * v408_data);
              v554_acc += (v207_bc * v409_data);
              v554_acc += (v209_bc * v410_data);
              v554_acc += (v211_bc * v411_data);
              v554_acc += (v213_bc * v412_data);
              v554_acc += (v215_bc * v413_data);
              v554_acc += (v217_bc * v414_data);
              v554_acc += (v219_bc * v415_data);
              v554_acc += (v221_bc * v416_data);
              ir3.template select<16, 1>(80) = v554_acc;
              tensorforge::intel_esimd::simd<float, 16> v581_acc{};
              v581_acc += (v226_bc * v405_data);
              v581_acc += (v228_bc * v406_data);
              v581_acc += (v230_bc * v407_data);
              v581_acc += (v232_bc * v408_data);
              v581_acc += (v234_bc * v409_data);
              v581_acc += (v236_bc * v410_data);
              v581_acc += (v238_bc * v411_data);
              v581_acc += (v240_bc * v412_data);
              v581_acc += (v242_bc * v413_data);
              v581_acc += (v244_bc * v414_data);
              v581_acc += (v246_bc * v415_data);
              v581_acc += (v248_bc * v416_data);
              ir3.template select<16, 1>(96) = v581_acc;
              tensorforge::intel_esimd::simd<float, 16> v608_acc{};
              v608_acc += (v253_bc * v405_data);
              v608_acc += (v255_bc * v406_data);
              v608_acc += (v257_bc * v407_data);
              v608_acc += (v259_bc * v408_data);
              v608_acc += (v261_bc * v409_data);
              v608_acc += (v263_bc * v410_data);
              v608_acc += (v265_bc * v411_data);
              v608_acc += (v267_bc * v412_data);
              v608_acc += (v269_bc * v413_data);
              v608_acc += (v271_bc * v414_data);
              v608_acc += (v273_bc * v415_data);
              v608_acc += (v275_bc * v416_data);
              ir3.template select<16, 1>(112) = v608_acc;
              tensorforge::intel_esimd::simd<float, 16> v635_acc{};
              v635_acc += (v280_bc * v405_data);
              v635_acc += (v282_bc * v406_data);
              v635_acc += (v284_bc * v407_data);
              v635_acc += (v286_bc * v408_data);
              v635_acc += (v288_bc * v409_data);
              v635_acc += (v290_bc * v410_data);
              v635_acc += (v292_bc * v411_data);
              v635_acc += (v294_bc * v412_data);
              v635_acc += (v296_bc * v413_data);
              v635_acc += (v298_bc * v414_data);
              v635_acc += (v300_bc * v415_data);
              v635_acc += (v302_bc * v416_data);
              ir3.template select<16, 1>(128) = v635_acc;
              tensorforge::intel_esimd::simd<float, 16> v662_acc{};
              v662_acc += (v307_bc * v405_data);
              v662_acc += (v309_bc * v406_data);
              v662_acc += (v311_bc * v407_data);
              v662_acc += (v313_bc * v408_data);
              v662_acc += (v315_bc * v409_data);
              v662_acc += (v317_bc * v410_data);
              v662_acc += (v319_bc * v411_data);
              v662_acc += (v321_bc * v412_data);
              v662_acc += (v323_bc * v413_data);
              v662_acc += (v325_bc * v414_data);
              v662_acc += (v327_bc * v415_data);
              v662_acc += (v329_bc * v416_data);
              ir3.template select<16, 1>(144) = v662_acc;
              tensorforge::intel_esimd::simd<float, 16> v689_acc{};
              v689_acc += (v334_bc * v405_data);
              v689_acc += (v336_bc * v406_data);
              v689_acc += (v338_bc * v407_data);
              v689_acc += (v340_bc * v408_data);
              v689_acc += (v342_bc * v409_data);
              v689_acc += (v344_bc * v410_data);
              v689_acc += (v346_bc * v411_data);
              v689_acc += (v348_bc * v412_data);
              v689_acc += (v350_bc * v413_data);
              v689_acc += (v352_bc * v414_data);
              v689_acc += (v354_bc * v415_data);
              v689_acc += (v356_bc * v416_data);
              ir3.template select<16, 1>(160) = v689_acc;
              tensorforge::intel_esimd::simd<float, 16> v716_acc{};
              v716_acc += (v361_bc * v405_data);
              v716_acc += (v363_bc * v406_data);
              v716_acc += (v365_bc * v407_data);
              v716_acc += (v367_bc * v408_data);
              v716_acc += (v369_bc * v409_data);
              v716_acc += (v371_bc * v410_data);
              v716_acc += (v373_bc * v411_data);
              v716_acc += (v375_bc * v412_data);
              v716_acc += (v377_bc * v413_data);
              v716_acc += (v379_bc * v414_data);
              v716_acc += (v381_bc * v415_data);
              v716_acc += (v383_bc * v416_data);
              ir3.template select<16, 1>(176) = v716_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v743_n1 = 0; v743_n1 < 12; ++v743_n1) {
                int32_t v744_a = v743_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v746_data(ir3.template select<6, 1>(v744_a));
                r3.template select<6, 1>(v744_a) = v746_data;
              }
              // s1 = store{r>s, clear}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v747_z1 = 0; v747_z1 < 12; ++v747_z1) {
                s1[(6_i32 + (v747_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v754_i1 = 0; v754_i1 < 12; ++v754_i1) {
                tensorforge::intel_esimd::simd<float, 6> v757_data(r3.template select<6, 1>((v754_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v754_i1 * 12)), v757_data);
              }
              // wait(r4 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // ir5 = +(r4)
              // [(0, 6), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v764_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v765_data(ir5.template select<16, 1>(0));
              ir5.template select<16, 1>(0) = (v765_data + v764_data);
              tensorforge::intel_esimd::simd<float, 16> v767_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v768_data(ir5.template select<16, 1>(16));
              ir5.template select<16, 1>(16) = (v768_data + v767_data);
              tensorforge::intel_esimd::simd<float, 16> v770_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v771_data(ir5.template select<16, 1>(32));
              ir5.template select<16, 1>(32) = (v771_data + v770_data);
              tensorforge::intel_esimd::simd<float, 16> v773_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v774_data(ir5.template select<16, 1>(48));
              ir5.template select<16, 1>(48) = (v774_data + v773_data);
              tensorforge::intel_esimd::simd<float, 16> v776_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v777_data(ir5.template select<16, 1>(64));
              ir5.template select<16, 1>(64) = (v777_data + v776_data);
              tensorforge::intel_esimd::simd<float, 16> v779_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v780_data(ir5.template select<16, 1>(80));
              ir5.template select<16, 1>(80) = (v780_data + v779_data);
              tensorforge::intel_esimd::simd<float, 16> v782_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v783_data(ir5.template select<16, 1>(96));
              ir5.template select<16, 1>(96) = (v783_data + v782_data);
              tensorforge::intel_esimd::simd<float, 16> v785_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v786_data(ir5.template select<16, 1>(112));
              ir5.template select<16, 1>(112) = (v786_data + v785_data);
              tensorforge::intel_esimd::simd<float, 16> v788_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v789_data(ir5.template select<16, 1>(128));
              ir5.template select<16, 1>(128) = (v789_data + v788_data);
              tensorforge::intel_esimd::simd<float, 16> v791_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v792_data(ir5.template select<16, 1>(144));
              ir5.template select<16, 1>(144) = (v792_data + v791_data);
              tensorforge::intel_esimd::simd<float, 16> v794_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v795_data(ir5.template select<16, 1>(160));
              ir5.template select<16, 1>(160) = (v795_data + v794_data);
              tensorforge::intel_esimd::simd<float, 16> v797_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v798_data(ir5.template select<16, 1>(176));
              ir5.template select<16, 1>(176) = (v798_data + v797_data);
              // r5 = ir5
              #pragma unroll
              for (int32_t v800_n1 = 0; v800_n1 < 12; ++v800_n1) {
                int32_t v801_a = v800_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v803_data(ir5.template select<6, 1>(v801_a));
                r5.template select<6, 1>(v801_a) = v803_data;
              }
              // s1 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v804_i1 = 0; v804_i1 < 12; ++v804_i1) {
                tensorforge::intel_esimd::simd<float, 6> v807_data(r5.template select<6, 1>((v804_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v804_i1 * 12))), v807_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // ir6 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v818_data(0.0f);
              v818_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v819_data(ir6.template select<16, 1>(0));
              ir6.template select<16, 1>(0) = (v819_data + v818_data);
              tensorforge::intel_esimd::simd<float, 16> v822_data(0.0f);
              v822_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v823_data(ir6.template select<16, 1>(16));
              ir6.template select<16, 1>(16) = (v823_data + v822_data);
              tensorforge::intel_esimd::simd<float, 16> v826_data(0.0f);
              v826_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v827_data(ir6.template select<16, 1>(32));
              ir6.template select<16, 1>(32) = (v827_data + v826_data);
              tensorforge::intel_esimd::simd<float, 16> v830_data(0.0f);
              v830_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v831_data(ir6.template select<16, 1>(48));
              ir6.template select<16, 1>(48) = (v831_data + v830_data);
              tensorforge::intel_esimd::simd<float, 16> v834_data(0.0f);
              v834_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v835_data(ir6.template select<16, 1>(64));
              ir6.template select<16, 1>(64) = (v835_data + v834_data);
              tensorforge::intel_esimd::simd<float, 16> v838_data(0.0f);
              v838_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v839_data(ir6.template select<16, 1>(80));
              ir6.template select<16, 1>(80) = (v839_data + v838_data);
              tensorforge::intel_esimd::simd<float, 16> v842_data(0.0f);
              v842_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v843_data(ir6.template select<16, 1>(96));
              ir6.template select<16, 1>(96) = (v843_data + v842_data);
              tensorforge::intel_esimd::simd<float, 16> v846_data(0.0f);
              v846_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v847_data(ir6.template select<16, 1>(112));
              ir6.template select<16, 1>(112) = (v847_data + v846_data);
              tensorforge::intel_esimd::simd<float, 16> v850_data(0.0f);
              v850_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v851_data(ir6.template select<16, 1>(128));
              ir6.template select<16, 1>(128) = (v851_data + v850_data);
              tensorforge::intel_esimd::simd<float, 16> v854_data(0.0f);
              v854_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v855_data(ir6.template select<16, 1>(144));
              ir6.template select<16, 1>(144) = (v855_data + v854_data);
              tensorforge::intel_esimd::simd<float, 16> v858_data(0.0f);
              v858_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v859_data(ir6.template select<16, 1>(160));
              ir6.template select<16, 1>(160) = (v859_data + v858_data);
              tensorforge::intel_esimd::simd<float, 16> v862_data(0.0f);
              v862_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v863_data(ir6.template select<16, 1>(176));
              ir6.template select<16, 1>(176) = (v863_data + v862_data);
              // r6 = ir6
              #pragma unroll
              for (int32_t v865_n1 = 0; v865_n1 < 12; ++v865_n1) {
                int32_t v866_a = v865_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v868_data(ir6.template select<12, 1>(v866_a));
                r6.template select<12, 1>(v866_a) = v868_data;
              }
              // glb_m4 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v869_i1 = 0; v869_i1 < 12; ++v869_i1) {
                tensorforge::intel_esimd::simd<float, 12> v872_data(r6.template select<12, 1>((v869_i1 * 16)));
                v872_data.copy_to(glb_m4 + ((v869_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


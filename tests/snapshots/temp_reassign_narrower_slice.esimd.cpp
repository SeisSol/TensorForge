// === base name ===
kernel_4ce34356b49b67cf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4ce34356b49b67cf = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4ce34356b49b67cf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4ce34356b49b67cf(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4ce34356b49b67cf(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4ce34356b49b67cf(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4ce34356b49b67cf(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4ce34356b49b67cf(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4ce34356b49b67cf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 12×12(12×12) {0..12}×{0..12} strided
        //   m4 2×12(2×12) {0..2}×{0..12} strided
        //   m5 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
        //   m3[i,j] = t0[i,j]
        //   t0[i,j]@{6..12}×{0..12} = m4[i,k] × m1[k,j]
        //   m5[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4864}],"shared_bytes":19456,"shared_elements":4864,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N2","bbox":[[0,0],[2,12]],"name":"m4","ordered":false,"parts":1,"shape":[2,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[2,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
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
              const float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 24 + 0 + m4_extraOffset];
              float *const __restrict__ glb_m5 = &m5[v6_batchId0 * 144 + 0 + m5_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
                tensorforge::intel_esimd::simd<float, 6> v26_data;
                v26_data.copy_from(glb_m0 + ((v21_i1 * 6)));
                r0.template select<6, 1>((v21_i1 * 16)) = v26_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v29_ld;
              v29_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v29_ld);
              tensorforge::intel_esimd::simd<float, 64> v30_ld;
              v30_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v30_ld);
              tensorforge::intel_esimd::simd<float, 16> v31_ld;
              v31_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v31_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
                tensorforge::intel_esimd::simd<float, 6> v38_data;
                v38_data.copy_from(glb_m2 + ((v33_i1 * 6)));
                r2.template select<6, 1>((v33_i1 * 16)) = v38_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v54_acc{};
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              float v59_bc = static_cast<float>(v58_data[0]);
              v54_acc += (v59_bc * v42_data);
              float v61_bc = static_cast<float>(v58_data[1]);
              v54_acc += (v61_bc * v43_data);
              float v63_bc = static_cast<float>(v58_data[2]);
              v54_acc += (v63_bc * v44_data);
              float v65_bc = static_cast<float>(v58_data[3]);
              v54_acc += (v65_bc * v45_data);
              float v67_bc = static_cast<float>(v58_data[4]);
              v54_acc += (v67_bc * v46_data);
              float v69_bc = static_cast<float>(v58_data[5]);
              v54_acc += (v69_bc * v47_data);
              float v71_bc = static_cast<float>(v58_data[6]);
              v54_acc += (v71_bc * v48_data);
              float v73_bc = static_cast<float>(v58_data[7]);
              v54_acc += (v73_bc * v49_data);
              float v75_bc = static_cast<float>(v58_data[8]);
              v54_acc += (v75_bc * v50_data);
              float v77_bc = static_cast<float>(v58_data[9]);
              v54_acc += (v77_bc * v51_data);
              float v79_bc = static_cast<float>(v58_data[10]);
              v54_acc += (v79_bc * v52_data);
              float v81_bc = static_cast<float>(v58_data[11]);
              v54_acc += (v81_bc * v53_data);
              r1.template select<16, 1>(0) = v54_acc;
              tensorforge::intel_esimd::simd<float, 16> v83_acc{};
              tensorforge::intel_esimd::simd<float, 16> v85_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              float v86_bc = static_cast<float>(v85_data[0]);
              v83_acc += (v86_bc * v42_data);
              float v88_bc = static_cast<float>(v85_data[1]);
              v83_acc += (v88_bc * v43_data);
              float v90_bc = static_cast<float>(v85_data[2]);
              v83_acc += (v90_bc * v44_data);
              float v92_bc = static_cast<float>(v85_data[3]);
              v83_acc += (v92_bc * v45_data);
              float v94_bc = static_cast<float>(v85_data[4]);
              v83_acc += (v94_bc * v46_data);
              float v96_bc = static_cast<float>(v85_data[5]);
              v83_acc += (v96_bc * v47_data);
              float v98_bc = static_cast<float>(v85_data[6]);
              v83_acc += (v98_bc * v48_data);
              float v100_bc = static_cast<float>(v85_data[7]);
              v83_acc += (v100_bc * v49_data);
              float v102_bc = static_cast<float>(v85_data[8]);
              v83_acc += (v102_bc * v50_data);
              float v104_bc = static_cast<float>(v85_data[9]);
              v83_acc += (v104_bc * v51_data);
              float v106_bc = static_cast<float>(v85_data[10]);
              v83_acc += (v106_bc * v52_data);
              float v108_bc = static_cast<float>(v85_data[11]);
              v83_acc += (v108_bc * v53_data);
              r1.template select<16, 1>(16) = v83_acc;
              tensorforge::intel_esimd::simd<float, 16> v110_acc{};
              tensorforge::intel_esimd::simd<float, 16> v112_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              float v113_bc = static_cast<float>(v112_data[0]);
              v110_acc += (v113_bc * v42_data);
              float v115_bc = static_cast<float>(v112_data[1]);
              v110_acc += (v115_bc * v43_data);
              float v117_bc = static_cast<float>(v112_data[2]);
              v110_acc += (v117_bc * v44_data);
              float v119_bc = static_cast<float>(v112_data[3]);
              v110_acc += (v119_bc * v45_data);
              float v121_bc = static_cast<float>(v112_data[4]);
              v110_acc += (v121_bc * v46_data);
              float v123_bc = static_cast<float>(v112_data[5]);
              v110_acc += (v123_bc * v47_data);
              float v125_bc = static_cast<float>(v112_data[6]);
              v110_acc += (v125_bc * v48_data);
              float v127_bc = static_cast<float>(v112_data[7]);
              v110_acc += (v127_bc * v49_data);
              float v129_bc = static_cast<float>(v112_data[8]);
              v110_acc += (v129_bc * v50_data);
              float v131_bc = static_cast<float>(v112_data[9]);
              v110_acc += (v131_bc * v51_data);
              float v133_bc = static_cast<float>(v112_data[10]);
              v110_acc += (v133_bc * v52_data);
              float v135_bc = static_cast<float>(v112_data[11]);
              v110_acc += (v135_bc * v53_data);
              r1.template select<16, 1>(32) = v110_acc;
              tensorforge::intel_esimd::simd<float, 16> v137_acc{};
              tensorforge::intel_esimd::simd<float, 16> v139_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              float v140_bc = static_cast<float>(v139_data[0]);
              v137_acc += (v140_bc * v42_data);
              float v142_bc = static_cast<float>(v139_data[1]);
              v137_acc += (v142_bc * v43_data);
              float v144_bc = static_cast<float>(v139_data[2]);
              v137_acc += (v144_bc * v44_data);
              float v146_bc = static_cast<float>(v139_data[3]);
              v137_acc += (v146_bc * v45_data);
              float v148_bc = static_cast<float>(v139_data[4]);
              v137_acc += (v148_bc * v46_data);
              float v150_bc = static_cast<float>(v139_data[5]);
              v137_acc += (v150_bc * v47_data);
              float v152_bc = static_cast<float>(v139_data[6]);
              v137_acc += (v152_bc * v48_data);
              float v154_bc = static_cast<float>(v139_data[7]);
              v137_acc += (v154_bc * v49_data);
              float v156_bc = static_cast<float>(v139_data[8]);
              v137_acc += (v156_bc * v50_data);
              float v158_bc = static_cast<float>(v139_data[9]);
              v137_acc += (v158_bc * v51_data);
              float v160_bc = static_cast<float>(v139_data[10]);
              v137_acc += (v160_bc * v52_data);
              float v162_bc = static_cast<float>(v139_data[11]);
              v137_acc += (v162_bc * v53_data);
              r1.template select<16, 1>(48) = v137_acc;
              tensorforge::intel_esimd::simd<float, 16> v164_acc{};
              tensorforge::intel_esimd::simd<float, 16> v166_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              float v167_bc = static_cast<float>(v166_data[0]);
              v164_acc += (v167_bc * v42_data);
              float v169_bc = static_cast<float>(v166_data[1]);
              v164_acc += (v169_bc * v43_data);
              float v171_bc = static_cast<float>(v166_data[2]);
              v164_acc += (v171_bc * v44_data);
              float v173_bc = static_cast<float>(v166_data[3]);
              v164_acc += (v173_bc * v45_data);
              float v175_bc = static_cast<float>(v166_data[4]);
              v164_acc += (v175_bc * v46_data);
              float v177_bc = static_cast<float>(v166_data[5]);
              v164_acc += (v177_bc * v47_data);
              float v179_bc = static_cast<float>(v166_data[6]);
              v164_acc += (v179_bc * v48_data);
              float v181_bc = static_cast<float>(v166_data[7]);
              v164_acc += (v181_bc * v49_data);
              float v183_bc = static_cast<float>(v166_data[8]);
              v164_acc += (v183_bc * v50_data);
              float v185_bc = static_cast<float>(v166_data[9]);
              v164_acc += (v185_bc * v51_data);
              float v187_bc = static_cast<float>(v166_data[10]);
              v164_acc += (v187_bc * v52_data);
              float v189_bc = static_cast<float>(v166_data[11]);
              v164_acc += (v189_bc * v53_data);
              r1.template select<16, 1>(64) = v164_acc;
              tensorforge::intel_esimd::simd<float, 16> v191_acc{};
              tensorforge::intel_esimd::simd<float, 16> v193_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              float v194_bc = static_cast<float>(v193_data[0]);
              v191_acc += (v194_bc * v42_data);
              float v196_bc = static_cast<float>(v193_data[1]);
              v191_acc += (v196_bc * v43_data);
              float v198_bc = static_cast<float>(v193_data[2]);
              v191_acc += (v198_bc * v44_data);
              float v200_bc = static_cast<float>(v193_data[3]);
              v191_acc += (v200_bc * v45_data);
              float v202_bc = static_cast<float>(v193_data[4]);
              v191_acc += (v202_bc * v46_data);
              float v204_bc = static_cast<float>(v193_data[5]);
              v191_acc += (v204_bc * v47_data);
              float v206_bc = static_cast<float>(v193_data[6]);
              v191_acc += (v206_bc * v48_data);
              float v208_bc = static_cast<float>(v193_data[7]);
              v191_acc += (v208_bc * v49_data);
              float v210_bc = static_cast<float>(v193_data[8]);
              v191_acc += (v210_bc * v50_data);
              float v212_bc = static_cast<float>(v193_data[9]);
              v191_acc += (v212_bc * v51_data);
              float v214_bc = static_cast<float>(v193_data[10]);
              v191_acc += (v214_bc * v52_data);
              float v216_bc = static_cast<float>(v193_data[11]);
              v191_acc += (v216_bc * v53_data);
              r1.template select<16, 1>(80) = v191_acc;
              tensorforge::intel_esimd::simd<float, 16> v218_acc{};
              tensorforge::intel_esimd::simd<float, 16> v220_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              float v221_bc = static_cast<float>(v220_data[0]);
              v218_acc += (v221_bc * v42_data);
              float v223_bc = static_cast<float>(v220_data[1]);
              v218_acc += (v223_bc * v43_data);
              float v225_bc = static_cast<float>(v220_data[2]);
              v218_acc += (v225_bc * v44_data);
              float v227_bc = static_cast<float>(v220_data[3]);
              v218_acc += (v227_bc * v45_data);
              float v229_bc = static_cast<float>(v220_data[4]);
              v218_acc += (v229_bc * v46_data);
              float v231_bc = static_cast<float>(v220_data[5]);
              v218_acc += (v231_bc * v47_data);
              float v233_bc = static_cast<float>(v220_data[6]);
              v218_acc += (v233_bc * v48_data);
              float v235_bc = static_cast<float>(v220_data[7]);
              v218_acc += (v235_bc * v49_data);
              float v237_bc = static_cast<float>(v220_data[8]);
              v218_acc += (v237_bc * v50_data);
              float v239_bc = static_cast<float>(v220_data[9]);
              v218_acc += (v239_bc * v51_data);
              float v241_bc = static_cast<float>(v220_data[10]);
              v218_acc += (v241_bc * v52_data);
              float v243_bc = static_cast<float>(v220_data[11]);
              v218_acc += (v243_bc * v53_data);
              r1.template select<16, 1>(96) = v218_acc;
              tensorforge::intel_esimd::simd<float, 16> v245_acc{};
              tensorforge::intel_esimd::simd<float, 16> v247_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              float v248_bc = static_cast<float>(v247_data[0]);
              v245_acc += (v248_bc * v42_data);
              float v250_bc = static_cast<float>(v247_data[1]);
              v245_acc += (v250_bc * v43_data);
              float v252_bc = static_cast<float>(v247_data[2]);
              v245_acc += (v252_bc * v44_data);
              float v254_bc = static_cast<float>(v247_data[3]);
              v245_acc += (v254_bc * v45_data);
              float v256_bc = static_cast<float>(v247_data[4]);
              v245_acc += (v256_bc * v46_data);
              float v258_bc = static_cast<float>(v247_data[5]);
              v245_acc += (v258_bc * v47_data);
              float v260_bc = static_cast<float>(v247_data[6]);
              v245_acc += (v260_bc * v48_data);
              float v262_bc = static_cast<float>(v247_data[7]);
              v245_acc += (v262_bc * v49_data);
              float v264_bc = static_cast<float>(v247_data[8]);
              v245_acc += (v264_bc * v50_data);
              float v266_bc = static_cast<float>(v247_data[9]);
              v245_acc += (v266_bc * v51_data);
              float v268_bc = static_cast<float>(v247_data[10]);
              v245_acc += (v268_bc * v52_data);
              float v270_bc = static_cast<float>(v247_data[11]);
              v245_acc += (v270_bc * v53_data);
              r1.template select<16, 1>(112) = v245_acc;
              tensorforge::intel_esimd::simd<float, 16> v272_acc{};
              tensorforge::intel_esimd::simd<float, 16> v274_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              float v275_bc = static_cast<float>(v274_data[0]);
              v272_acc += (v275_bc * v42_data);
              float v277_bc = static_cast<float>(v274_data[1]);
              v272_acc += (v277_bc * v43_data);
              float v279_bc = static_cast<float>(v274_data[2]);
              v272_acc += (v279_bc * v44_data);
              float v281_bc = static_cast<float>(v274_data[3]);
              v272_acc += (v281_bc * v45_data);
              float v283_bc = static_cast<float>(v274_data[4]);
              v272_acc += (v283_bc * v46_data);
              float v285_bc = static_cast<float>(v274_data[5]);
              v272_acc += (v285_bc * v47_data);
              float v287_bc = static_cast<float>(v274_data[6]);
              v272_acc += (v287_bc * v48_data);
              float v289_bc = static_cast<float>(v274_data[7]);
              v272_acc += (v289_bc * v49_data);
              float v291_bc = static_cast<float>(v274_data[8]);
              v272_acc += (v291_bc * v50_data);
              float v293_bc = static_cast<float>(v274_data[9]);
              v272_acc += (v293_bc * v51_data);
              float v295_bc = static_cast<float>(v274_data[10]);
              v272_acc += (v295_bc * v52_data);
              float v297_bc = static_cast<float>(v274_data[11]);
              v272_acc += (v297_bc * v53_data);
              r1.template select<16, 1>(128) = v272_acc;
              tensorforge::intel_esimd::simd<float, 16> v299_acc{};
              tensorforge::intel_esimd::simd<float, 16> v301_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              float v302_bc = static_cast<float>(v301_data[0]);
              v299_acc += (v302_bc * v42_data);
              float v304_bc = static_cast<float>(v301_data[1]);
              v299_acc += (v304_bc * v43_data);
              float v306_bc = static_cast<float>(v301_data[2]);
              v299_acc += (v306_bc * v44_data);
              float v308_bc = static_cast<float>(v301_data[3]);
              v299_acc += (v308_bc * v45_data);
              float v310_bc = static_cast<float>(v301_data[4]);
              v299_acc += (v310_bc * v46_data);
              float v312_bc = static_cast<float>(v301_data[5]);
              v299_acc += (v312_bc * v47_data);
              float v314_bc = static_cast<float>(v301_data[6]);
              v299_acc += (v314_bc * v48_data);
              float v316_bc = static_cast<float>(v301_data[7]);
              v299_acc += (v316_bc * v49_data);
              float v318_bc = static_cast<float>(v301_data[8]);
              v299_acc += (v318_bc * v50_data);
              float v320_bc = static_cast<float>(v301_data[9]);
              v299_acc += (v320_bc * v51_data);
              float v322_bc = static_cast<float>(v301_data[10]);
              v299_acc += (v322_bc * v52_data);
              float v324_bc = static_cast<float>(v301_data[11]);
              v299_acc += (v324_bc * v53_data);
              r1.template select<16, 1>(144) = v299_acc;
              tensorforge::intel_esimd::simd<float, 16> v326_acc{};
              tensorforge::intel_esimd::simd<float, 16> v328_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              float v329_bc = static_cast<float>(v328_data[0]);
              v326_acc += (v329_bc * v42_data);
              float v331_bc = static_cast<float>(v328_data[1]);
              v326_acc += (v331_bc * v43_data);
              float v333_bc = static_cast<float>(v328_data[2]);
              v326_acc += (v333_bc * v44_data);
              float v335_bc = static_cast<float>(v328_data[3]);
              v326_acc += (v335_bc * v45_data);
              float v337_bc = static_cast<float>(v328_data[4]);
              v326_acc += (v337_bc * v46_data);
              float v339_bc = static_cast<float>(v328_data[5]);
              v326_acc += (v339_bc * v47_data);
              float v341_bc = static_cast<float>(v328_data[6]);
              v326_acc += (v341_bc * v48_data);
              float v343_bc = static_cast<float>(v328_data[7]);
              v326_acc += (v343_bc * v49_data);
              float v345_bc = static_cast<float>(v328_data[8]);
              v326_acc += (v345_bc * v50_data);
              float v347_bc = static_cast<float>(v328_data[9]);
              v326_acc += (v347_bc * v51_data);
              float v349_bc = static_cast<float>(v328_data[10]);
              v326_acc += (v349_bc * v52_data);
              float v351_bc = static_cast<float>(v328_data[11]);
              v326_acc += (v351_bc * v53_data);
              r1.template select<16, 1>(160) = v326_acc;
              tensorforge::intel_esimd::simd<float, 16> v353_acc{};
              tensorforge::intel_esimd::simd<float, 16> v355_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              float v356_bc = static_cast<float>(v355_data[0]);
              v353_acc += (v356_bc * v42_data);
              float v358_bc = static_cast<float>(v355_data[1]);
              v353_acc += (v358_bc * v43_data);
              float v360_bc = static_cast<float>(v355_data[2]);
              v353_acc += (v360_bc * v44_data);
              float v362_bc = static_cast<float>(v355_data[3]);
              v353_acc += (v362_bc * v45_data);
              float v364_bc = static_cast<float>(v355_data[4]);
              v353_acc += (v364_bc * v46_data);
              float v366_bc = static_cast<float>(v355_data[5]);
              v353_acc += (v366_bc * v47_data);
              float v368_bc = static_cast<float>(v355_data[6]);
              v353_acc += (v368_bc * v48_data);
              float v370_bc = static_cast<float>(v355_data[7]);
              v353_acc += (v370_bc * v49_data);
              float v372_bc = static_cast<float>(v355_data[8]);
              v353_acc += (v372_bc * v50_data);
              float v374_bc = static_cast<float>(v355_data[9]);
              v353_acc += (v374_bc * v51_data);
              float v376_bc = static_cast<float>(v355_data[10]);
              v353_acc += (v376_bc * v52_data);
              float v378_bc = static_cast<float>(v355_data[11]);
              v353_acc += (v378_bc * v53_data);
              r1.template select<16, 1>(176) = v353_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v380_i1 = 0; v380_i1 < 12; ++v380_i1) {
                tensorforge::intel_esimd::simd<float, 6> v383_data(r1.template select<6, 1>((v380_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v380_i1 * 12)), v383_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // r5 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v389_i1 = 0; v389_i1 < 12; ++v389_i1) {
                tensorforge::intel_esimd::simd<float, 2> v394_data;
                v394_data.copy_from(glb_m4 + ((v389_i1 * 2)));
                r5.template select<2, 1>((v389_i1 * 16)) = v394_data;
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
              v411_acc += (v59_bc * v399_data);
              v411_acc += (v61_bc * v400_data);
              v411_acc += (v63_bc * v401_data);
              v411_acc += (v65_bc * v402_data);
              v411_acc += (v67_bc * v403_data);
              v411_acc += (v69_bc * v404_data);
              v411_acc += (v71_bc * v405_data);
              v411_acc += (v73_bc * v406_data);
              v411_acc += (v75_bc * v407_data);
              v411_acc += (v77_bc * v408_data);
              v411_acc += (v79_bc * v409_data);
              v411_acc += (v81_bc * v410_data);
              ir3.template select<16, 1>(0) = v411_acc;
              tensorforge::intel_esimd::simd<float, 16> v440_acc{};
              v440_acc += (v86_bc * v399_data);
              v440_acc += (v88_bc * v400_data);
              v440_acc += (v90_bc * v401_data);
              v440_acc += (v92_bc * v402_data);
              v440_acc += (v94_bc * v403_data);
              v440_acc += (v96_bc * v404_data);
              v440_acc += (v98_bc * v405_data);
              v440_acc += (v100_bc * v406_data);
              v440_acc += (v102_bc * v407_data);
              v440_acc += (v104_bc * v408_data);
              v440_acc += (v106_bc * v409_data);
              v440_acc += (v108_bc * v410_data);
              ir3.template select<16, 1>(16) = v440_acc;
              tensorforge::intel_esimd::simd<float, 16> v467_acc{};
              v467_acc += (v113_bc * v399_data);
              v467_acc += (v115_bc * v400_data);
              v467_acc += (v117_bc * v401_data);
              v467_acc += (v119_bc * v402_data);
              v467_acc += (v121_bc * v403_data);
              v467_acc += (v123_bc * v404_data);
              v467_acc += (v125_bc * v405_data);
              v467_acc += (v127_bc * v406_data);
              v467_acc += (v129_bc * v407_data);
              v467_acc += (v131_bc * v408_data);
              v467_acc += (v133_bc * v409_data);
              v467_acc += (v135_bc * v410_data);
              ir3.template select<16, 1>(32) = v467_acc;
              tensorforge::intel_esimd::simd<float, 16> v494_acc{};
              v494_acc += (v140_bc * v399_data);
              v494_acc += (v142_bc * v400_data);
              v494_acc += (v144_bc * v401_data);
              v494_acc += (v146_bc * v402_data);
              v494_acc += (v148_bc * v403_data);
              v494_acc += (v150_bc * v404_data);
              v494_acc += (v152_bc * v405_data);
              v494_acc += (v154_bc * v406_data);
              v494_acc += (v156_bc * v407_data);
              v494_acc += (v158_bc * v408_data);
              v494_acc += (v160_bc * v409_data);
              v494_acc += (v162_bc * v410_data);
              ir3.template select<16, 1>(48) = v494_acc;
              tensorforge::intel_esimd::simd<float, 16> v521_acc{};
              v521_acc += (v167_bc * v399_data);
              v521_acc += (v169_bc * v400_data);
              v521_acc += (v171_bc * v401_data);
              v521_acc += (v173_bc * v402_data);
              v521_acc += (v175_bc * v403_data);
              v521_acc += (v177_bc * v404_data);
              v521_acc += (v179_bc * v405_data);
              v521_acc += (v181_bc * v406_data);
              v521_acc += (v183_bc * v407_data);
              v521_acc += (v185_bc * v408_data);
              v521_acc += (v187_bc * v409_data);
              v521_acc += (v189_bc * v410_data);
              ir3.template select<16, 1>(64) = v521_acc;
              tensorforge::intel_esimd::simd<float, 16> v548_acc{};
              v548_acc += (v194_bc * v399_data);
              v548_acc += (v196_bc * v400_data);
              v548_acc += (v198_bc * v401_data);
              v548_acc += (v200_bc * v402_data);
              v548_acc += (v202_bc * v403_data);
              v548_acc += (v204_bc * v404_data);
              v548_acc += (v206_bc * v405_data);
              v548_acc += (v208_bc * v406_data);
              v548_acc += (v210_bc * v407_data);
              v548_acc += (v212_bc * v408_data);
              v548_acc += (v214_bc * v409_data);
              v548_acc += (v216_bc * v410_data);
              ir3.template select<16, 1>(80) = v548_acc;
              tensorforge::intel_esimd::simd<float, 16> v575_acc{};
              v575_acc += (v221_bc * v399_data);
              v575_acc += (v223_bc * v400_data);
              v575_acc += (v225_bc * v401_data);
              v575_acc += (v227_bc * v402_data);
              v575_acc += (v229_bc * v403_data);
              v575_acc += (v231_bc * v404_data);
              v575_acc += (v233_bc * v405_data);
              v575_acc += (v235_bc * v406_data);
              v575_acc += (v237_bc * v407_data);
              v575_acc += (v239_bc * v408_data);
              v575_acc += (v241_bc * v409_data);
              v575_acc += (v243_bc * v410_data);
              ir3.template select<16, 1>(96) = v575_acc;
              tensorforge::intel_esimd::simd<float, 16> v602_acc{};
              v602_acc += (v248_bc * v399_data);
              v602_acc += (v250_bc * v400_data);
              v602_acc += (v252_bc * v401_data);
              v602_acc += (v254_bc * v402_data);
              v602_acc += (v256_bc * v403_data);
              v602_acc += (v258_bc * v404_data);
              v602_acc += (v260_bc * v405_data);
              v602_acc += (v262_bc * v406_data);
              v602_acc += (v264_bc * v407_data);
              v602_acc += (v266_bc * v408_data);
              v602_acc += (v268_bc * v409_data);
              v602_acc += (v270_bc * v410_data);
              ir3.template select<16, 1>(112) = v602_acc;
              tensorforge::intel_esimd::simd<float, 16> v629_acc{};
              v629_acc += (v275_bc * v399_data);
              v629_acc += (v277_bc * v400_data);
              v629_acc += (v279_bc * v401_data);
              v629_acc += (v281_bc * v402_data);
              v629_acc += (v283_bc * v403_data);
              v629_acc += (v285_bc * v404_data);
              v629_acc += (v287_bc * v405_data);
              v629_acc += (v289_bc * v406_data);
              v629_acc += (v291_bc * v407_data);
              v629_acc += (v293_bc * v408_data);
              v629_acc += (v295_bc * v409_data);
              v629_acc += (v297_bc * v410_data);
              ir3.template select<16, 1>(128) = v629_acc;
              tensorforge::intel_esimd::simd<float, 16> v656_acc{};
              v656_acc += (v302_bc * v399_data);
              v656_acc += (v304_bc * v400_data);
              v656_acc += (v306_bc * v401_data);
              v656_acc += (v308_bc * v402_data);
              v656_acc += (v310_bc * v403_data);
              v656_acc += (v312_bc * v404_data);
              v656_acc += (v314_bc * v405_data);
              v656_acc += (v316_bc * v406_data);
              v656_acc += (v318_bc * v407_data);
              v656_acc += (v320_bc * v408_data);
              v656_acc += (v322_bc * v409_data);
              v656_acc += (v324_bc * v410_data);
              ir3.template select<16, 1>(144) = v656_acc;
              tensorforge::intel_esimd::simd<float, 16> v683_acc{};
              v683_acc += (v329_bc * v399_data);
              v683_acc += (v331_bc * v400_data);
              v683_acc += (v333_bc * v401_data);
              v683_acc += (v335_bc * v402_data);
              v683_acc += (v337_bc * v403_data);
              v683_acc += (v339_bc * v404_data);
              v683_acc += (v341_bc * v405_data);
              v683_acc += (v343_bc * v406_data);
              v683_acc += (v345_bc * v407_data);
              v683_acc += (v347_bc * v408_data);
              v683_acc += (v349_bc * v409_data);
              v683_acc += (v351_bc * v410_data);
              ir3.template select<16, 1>(160) = v683_acc;
              tensorforge::intel_esimd::simd<float, 16> v710_acc{};
              v710_acc += (v356_bc * v399_data);
              v710_acc += (v358_bc * v400_data);
              v710_acc += (v360_bc * v401_data);
              v710_acc += (v362_bc * v402_data);
              v710_acc += (v364_bc * v403_data);
              v710_acc += (v366_bc * v404_data);
              v710_acc += (v368_bc * v405_data);
              v710_acc += (v370_bc * v406_data);
              v710_acc += (v372_bc * v407_data);
              v710_acc += (v374_bc * v408_data);
              v710_acc += (v376_bc * v409_data);
              v710_acc += (v378_bc * v410_data);
              ir3.template select<16, 1>(176) = v710_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v737_n1 = 0; v737_n1 < 12; ++v737_n1) {
                int32_t v738_a = v737_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v740_data(ir3.template select<6, 1>(v738_a));
                r3.template select<6, 1>(v738_a) = v740_data;
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v741_i1 = 0; v741_i1 < 12; ++v741_i1) {
                tensorforge::intel_esimd::simd<float, 6> v744_data(r3.template select<6, 1>((v741_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v741_i1 * 12))), v744_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // ir4 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir4(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v755_data(0.0f);
              v755_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v756_data(ir4.template select<16, 1>(0));
              ir4.template select<16, 1>(0) = (v756_data + v755_data);
              tensorforge::intel_esimd::simd<float, 16> v759_data(0.0f);
              v759_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v760_data(ir4.template select<16, 1>(16));
              ir4.template select<16, 1>(16) = (v760_data + v759_data);
              tensorforge::intel_esimd::simd<float, 16> v763_data(0.0f);
              v763_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v764_data(ir4.template select<16, 1>(32));
              ir4.template select<16, 1>(32) = (v764_data + v763_data);
              tensorforge::intel_esimd::simd<float, 16> v767_data(0.0f);
              v767_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v768_data(ir4.template select<16, 1>(48));
              ir4.template select<16, 1>(48) = (v768_data + v767_data);
              tensorforge::intel_esimd::simd<float, 16> v771_data(0.0f);
              v771_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v772_data(ir4.template select<16, 1>(64));
              ir4.template select<16, 1>(64) = (v772_data + v771_data);
              tensorforge::intel_esimd::simd<float, 16> v775_data(0.0f);
              v775_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v776_data(ir4.template select<16, 1>(80));
              ir4.template select<16, 1>(80) = (v776_data + v775_data);
              tensorforge::intel_esimd::simd<float, 16> v779_data(0.0f);
              v779_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v780_data(ir4.template select<16, 1>(96));
              ir4.template select<16, 1>(96) = (v780_data + v779_data);
              tensorforge::intel_esimd::simd<float, 16> v783_data(0.0f);
              v783_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v784_data(ir4.template select<16, 1>(112));
              ir4.template select<16, 1>(112) = (v784_data + v783_data);
              tensorforge::intel_esimd::simd<float, 16> v787_data(0.0f);
              v787_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v788_data(ir4.template select<16, 1>(128));
              ir4.template select<16, 1>(128) = (v788_data + v787_data);
              tensorforge::intel_esimd::simd<float, 16> v791_data(0.0f);
              v791_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v792_data(ir4.template select<16, 1>(144));
              ir4.template select<16, 1>(144) = (v792_data + v791_data);
              tensorforge::intel_esimd::simd<float, 16> v795_data(0.0f);
              v795_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v796_data(ir4.template select<16, 1>(160));
              ir4.template select<16, 1>(160) = (v796_data + v795_data);
              tensorforge::intel_esimd::simd<float, 16> v799_data(0.0f);
              v799_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v800_data(ir4.template select<16, 1>(176));
              ir4.template select<16, 1>(176) = (v800_data + v799_data);
              // r4 = ir4
              #pragma unroll
              for (int32_t v802_n1 = 0; v802_n1 < 12; ++v802_n1) {
                int32_t v803_a = v802_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v805_data(ir4.template select<12, 1>(v803_a));
                r4.template select<12, 1>(v803_a) = v805_data;
              }
              // glb_m3 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v806_i1 = 0; v806_i1 < 12; ++v806_i1) {
                tensorforge::intel_esimd::simd<float, 12> v809_data(r4.template select<12, 1>((v806_i1 * 16)));
                v809_data.copy_to(glb_m3 + ((v806_i1 * 12)));
              }
              // wait(r5 = load{g>r}(glb_m4););
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // ir6 = +(r5 * s0)
              // [(0, 2), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v816_data(r5.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v817_data(r5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v818_data(r5.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v819_data(r5.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v820_data(r5.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v821_data(r5.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v822_data(r5.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v823_data(r5.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v824_data(r5.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v825_data(r5.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v826_data(r5.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v827_data(r5.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v828_acc{};
              tensorforge::intel_esimd::simd<float, 16> v832_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v828_acc += ((static_cast<float>(v832_data[0])) * v816_data);
              v828_acc += ((static_cast<float>(v832_data[1])) * v817_data);
              v828_acc += ((static_cast<float>(v832_data[2])) * v818_data);
              v828_acc += ((static_cast<float>(v832_data[3])) * v819_data);
              v828_acc += ((static_cast<float>(v832_data[4])) * v820_data);
              v828_acc += ((static_cast<float>(v832_data[5])) * v821_data);
              v828_acc += ((static_cast<float>(v832_data[6])) * v822_data);
              v828_acc += ((static_cast<float>(v832_data[7])) * v823_data);
              v828_acc += ((static_cast<float>(v832_data[8])) * v824_data);
              v828_acc += ((static_cast<float>(v832_data[9])) * v825_data);
              v828_acc += ((static_cast<float>(v832_data[10])) * v826_data);
              v828_acc += ((static_cast<float>(v832_data[11])) * v827_data);
              ir6.template select<16, 1>(0) = v828_acc;
              tensorforge::intel_esimd::simd<float, 16> v857_acc{};
              tensorforge::intel_esimd::simd<float, 16> v859_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v857_acc += ((static_cast<float>(v859_data[0])) * v816_data);
              v857_acc += ((static_cast<float>(v859_data[1])) * v817_data);
              v857_acc += ((static_cast<float>(v859_data[2])) * v818_data);
              v857_acc += ((static_cast<float>(v859_data[3])) * v819_data);
              v857_acc += ((static_cast<float>(v859_data[4])) * v820_data);
              v857_acc += ((static_cast<float>(v859_data[5])) * v821_data);
              v857_acc += ((static_cast<float>(v859_data[6])) * v822_data);
              v857_acc += ((static_cast<float>(v859_data[7])) * v823_data);
              v857_acc += ((static_cast<float>(v859_data[8])) * v824_data);
              v857_acc += ((static_cast<float>(v859_data[9])) * v825_data);
              v857_acc += ((static_cast<float>(v859_data[10])) * v826_data);
              v857_acc += ((static_cast<float>(v859_data[11])) * v827_data);
              ir6.template select<16, 1>(16) = v857_acc;
              tensorforge::intel_esimd::simd<float, 16> v884_acc{};
              tensorforge::intel_esimd::simd<float, 16> v886_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v884_acc += ((static_cast<float>(v886_data[0])) * v816_data);
              v884_acc += ((static_cast<float>(v886_data[1])) * v817_data);
              v884_acc += ((static_cast<float>(v886_data[2])) * v818_data);
              v884_acc += ((static_cast<float>(v886_data[3])) * v819_data);
              v884_acc += ((static_cast<float>(v886_data[4])) * v820_data);
              v884_acc += ((static_cast<float>(v886_data[5])) * v821_data);
              v884_acc += ((static_cast<float>(v886_data[6])) * v822_data);
              v884_acc += ((static_cast<float>(v886_data[7])) * v823_data);
              v884_acc += ((static_cast<float>(v886_data[8])) * v824_data);
              v884_acc += ((static_cast<float>(v886_data[9])) * v825_data);
              v884_acc += ((static_cast<float>(v886_data[10])) * v826_data);
              v884_acc += ((static_cast<float>(v886_data[11])) * v827_data);
              ir6.template select<16, 1>(32) = v884_acc;
              tensorforge::intel_esimd::simd<float, 16> v911_acc{};
              tensorforge::intel_esimd::simd<float, 16> v913_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v911_acc += ((static_cast<float>(v913_data[0])) * v816_data);
              v911_acc += ((static_cast<float>(v913_data[1])) * v817_data);
              v911_acc += ((static_cast<float>(v913_data[2])) * v818_data);
              v911_acc += ((static_cast<float>(v913_data[3])) * v819_data);
              v911_acc += ((static_cast<float>(v913_data[4])) * v820_data);
              v911_acc += ((static_cast<float>(v913_data[5])) * v821_data);
              v911_acc += ((static_cast<float>(v913_data[6])) * v822_data);
              v911_acc += ((static_cast<float>(v913_data[7])) * v823_data);
              v911_acc += ((static_cast<float>(v913_data[8])) * v824_data);
              v911_acc += ((static_cast<float>(v913_data[9])) * v825_data);
              v911_acc += ((static_cast<float>(v913_data[10])) * v826_data);
              v911_acc += ((static_cast<float>(v913_data[11])) * v827_data);
              ir6.template select<16, 1>(48) = v911_acc;
              tensorforge::intel_esimd::simd<float, 16> v938_acc{};
              tensorforge::intel_esimd::simd<float, 16> v940_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v938_acc += ((static_cast<float>(v940_data[0])) * v816_data);
              v938_acc += ((static_cast<float>(v940_data[1])) * v817_data);
              v938_acc += ((static_cast<float>(v940_data[2])) * v818_data);
              v938_acc += ((static_cast<float>(v940_data[3])) * v819_data);
              v938_acc += ((static_cast<float>(v940_data[4])) * v820_data);
              v938_acc += ((static_cast<float>(v940_data[5])) * v821_data);
              v938_acc += ((static_cast<float>(v940_data[6])) * v822_data);
              v938_acc += ((static_cast<float>(v940_data[7])) * v823_data);
              v938_acc += ((static_cast<float>(v940_data[8])) * v824_data);
              v938_acc += ((static_cast<float>(v940_data[9])) * v825_data);
              v938_acc += ((static_cast<float>(v940_data[10])) * v826_data);
              v938_acc += ((static_cast<float>(v940_data[11])) * v827_data);
              ir6.template select<16, 1>(64) = v938_acc;
              tensorforge::intel_esimd::simd<float, 16> v965_acc{};
              tensorforge::intel_esimd::simd<float, 16> v967_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v965_acc += ((static_cast<float>(v967_data[0])) * v816_data);
              v965_acc += ((static_cast<float>(v967_data[1])) * v817_data);
              v965_acc += ((static_cast<float>(v967_data[2])) * v818_data);
              v965_acc += ((static_cast<float>(v967_data[3])) * v819_data);
              v965_acc += ((static_cast<float>(v967_data[4])) * v820_data);
              v965_acc += ((static_cast<float>(v967_data[5])) * v821_data);
              v965_acc += ((static_cast<float>(v967_data[6])) * v822_data);
              v965_acc += ((static_cast<float>(v967_data[7])) * v823_data);
              v965_acc += ((static_cast<float>(v967_data[8])) * v824_data);
              v965_acc += ((static_cast<float>(v967_data[9])) * v825_data);
              v965_acc += ((static_cast<float>(v967_data[10])) * v826_data);
              v965_acc += ((static_cast<float>(v967_data[11])) * v827_data);
              ir6.template select<16, 1>(80) = v965_acc;
              tensorforge::intel_esimd::simd<float, 16> v992_acc{};
              tensorforge::intel_esimd::simd<float, 16> v994_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v992_acc += ((static_cast<float>(v994_data[0])) * v816_data);
              v992_acc += ((static_cast<float>(v994_data[1])) * v817_data);
              v992_acc += ((static_cast<float>(v994_data[2])) * v818_data);
              v992_acc += ((static_cast<float>(v994_data[3])) * v819_data);
              v992_acc += ((static_cast<float>(v994_data[4])) * v820_data);
              v992_acc += ((static_cast<float>(v994_data[5])) * v821_data);
              v992_acc += ((static_cast<float>(v994_data[6])) * v822_data);
              v992_acc += ((static_cast<float>(v994_data[7])) * v823_data);
              v992_acc += ((static_cast<float>(v994_data[8])) * v824_data);
              v992_acc += ((static_cast<float>(v994_data[9])) * v825_data);
              v992_acc += ((static_cast<float>(v994_data[10])) * v826_data);
              v992_acc += ((static_cast<float>(v994_data[11])) * v827_data);
              ir6.template select<16, 1>(96) = v992_acc;
              tensorforge::intel_esimd::simd<float, 16> v1019_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1021_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v1019_acc += ((static_cast<float>(v1021_data[0])) * v816_data);
              v1019_acc += ((static_cast<float>(v1021_data[1])) * v817_data);
              v1019_acc += ((static_cast<float>(v1021_data[2])) * v818_data);
              v1019_acc += ((static_cast<float>(v1021_data[3])) * v819_data);
              v1019_acc += ((static_cast<float>(v1021_data[4])) * v820_data);
              v1019_acc += ((static_cast<float>(v1021_data[5])) * v821_data);
              v1019_acc += ((static_cast<float>(v1021_data[6])) * v822_data);
              v1019_acc += ((static_cast<float>(v1021_data[7])) * v823_data);
              v1019_acc += ((static_cast<float>(v1021_data[8])) * v824_data);
              v1019_acc += ((static_cast<float>(v1021_data[9])) * v825_data);
              v1019_acc += ((static_cast<float>(v1021_data[10])) * v826_data);
              v1019_acc += ((static_cast<float>(v1021_data[11])) * v827_data);
              ir6.template select<16, 1>(112) = v1019_acc;
              tensorforge::intel_esimd::simd<float, 16> v1046_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1048_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v1046_acc += ((static_cast<float>(v1048_data[0])) * v816_data);
              v1046_acc += ((static_cast<float>(v1048_data[1])) * v817_data);
              v1046_acc += ((static_cast<float>(v1048_data[2])) * v818_data);
              v1046_acc += ((static_cast<float>(v1048_data[3])) * v819_data);
              v1046_acc += ((static_cast<float>(v1048_data[4])) * v820_data);
              v1046_acc += ((static_cast<float>(v1048_data[5])) * v821_data);
              v1046_acc += ((static_cast<float>(v1048_data[6])) * v822_data);
              v1046_acc += ((static_cast<float>(v1048_data[7])) * v823_data);
              v1046_acc += ((static_cast<float>(v1048_data[8])) * v824_data);
              v1046_acc += ((static_cast<float>(v1048_data[9])) * v825_data);
              v1046_acc += ((static_cast<float>(v1048_data[10])) * v826_data);
              v1046_acc += ((static_cast<float>(v1048_data[11])) * v827_data);
              ir6.template select<16, 1>(128) = v1046_acc;
              tensorforge::intel_esimd::simd<float, 16> v1073_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1075_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              v1073_acc += ((static_cast<float>(v1075_data[0])) * v816_data);
              v1073_acc += ((static_cast<float>(v1075_data[1])) * v817_data);
              v1073_acc += ((static_cast<float>(v1075_data[2])) * v818_data);
              v1073_acc += ((static_cast<float>(v1075_data[3])) * v819_data);
              v1073_acc += ((static_cast<float>(v1075_data[4])) * v820_data);
              v1073_acc += ((static_cast<float>(v1075_data[5])) * v821_data);
              v1073_acc += ((static_cast<float>(v1075_data[6])) * v822_data);
              v1073_acc += ((static_cast<float>(v1075_data[7])) * v823_data);
              v1073_acc += ((static_cast<float>(v1075_data[8])) * v824_data);
              v1073_acc += ((static_cast<float>(v1075_data[9])) * v825_data);
              v1073_acc += ((static_cast<float>(v1075_data[10])) * v826_data);
              v1073_acc += ((static_cast<float>(v1075_data[11])) * v827_data);
              ir6.template select<16, 1>(144) = v1073_acc;
              tensorforge::intel_esimd::simd<float, 16> v1100_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1102_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              v1100_acc += ((static_cast<float>(v1102_data[0])) * v816_data);
              v1100_acc += ((static_cast<float>(v1102_data[1])) * v817_data);
              v1100_acc += ((static_cast<float>(v1102_data[2])) * v818_data);
              v1100_acc += ((static_cast<float>(v1102_data[3])) * v819_data);
              v1100_acc += ((static_cast<float>(v1102_data[4])) * v820_data);
              v1100_acc += ((static_cast<float>(v1102_data[5])) * v821_data);
              v1100_acc += ((static_cast<float>(v1102_data[6])) * v822_data);
              v1100_acc += ((static_cast<float>(v1102_data[7])) * v823_data);
              v1100_acc += ((static_cast<float>(v1102_data[8])) * v824_data);
              v1100_acc += ((static_cast<float>(v1102_data[9])) * v825_data);
              v1100_acc += ((static_cast<float>(v1102_data[10])) * v826_data);
              v1100_acc += ((static_cast<float>(v1102_data[11])) * v827_data);
              ir6.template select<16, 1>(160) = v1100_acc;
              tensorforge::intel_esimd::simd<float, 16> v1127_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1129_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              v1127_acc += ((static_cast<float>(v1129_data[0])) * v816_data);
              v1127_acc += ((static_cast<float>(v1129_data[1])) * v817_data);
              v1127_acc += ((static_cast<float>(v1129_data[2])) * v818_data);
              v1127_acc += ((static_cast<float>(v1129_data[3])) * v819_data);
              v1127_acc += ((static_cast<float>(v1129_data[4])) * v820_data);
              v1127_acc += ((static_cast<float>(v1129_data[5])) * v821_data);
              v1127_acc += ((static_cast<float>(v1129_data[6])) * v822_data);
              v1127_acc += ((static_cast<float>(v1129_data[7])) * v823_data);
              v1127_acc += ((static_cast<float>(v1129_data[8])) * v824_data);
              v1127_acc += ((static_cast<float>(v1129_data[9])) * v825_data);
              v1127_acc += ((static_cast<float>(v1129_data[10])) * v826_data);
              v1127_acc += ((static_cast<float>(v1129_data[11])) * v827_data);
              ir6.template select<16, 1>(176) = v1127_acc;
              // r6 = ir6
              #pragma unroll
              for (int32_t v1154_n1 = 0; v1154_n1 < 12; ++v1154_n1) {
                int32_t v1155_a = v1154_n1 * 16;
                tensorforge::intel_esimd::simd<float, 2> v1157_data(ir6.template select<2, 1>(v1155_a));
                r6.template select<2, 1>(v1155_a) = v1157_data;
              }
              // s1 = store{r>s, clear}(localShrMem0, r6);
              #pragma unroll
              for (int32_t v1158_z1 = 0; v1158_z1 < 12; ++v1158_z1) {
                s1[(8_i32 + (v1158_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v1165_i1 = 0; v1165_i1 < 12; ++v1165_i1) {
                tensorforge::intel_esimd::simd<float, 2> v1168_data(r6.template select<2, 1>((v1165_i1 * 16)));
                tensorforge::slmStore<float, 2>(s1 + ((6_i32 + (v1165_i1 * 12))), v1168_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r7(0.0f);
              // ir7 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir7(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v1179_data(0.0f);
              v1179_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v1180_data(ir7.template select<16, 1>(0));
              ir7.template select<16, 1>(0) = (v1180_data + v1179_data);
              tensorforge::intel_esimd::simd<float, 16> v1183_data(0.0f);
              v1183_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v1184_data(ir7.template select<16, 1>(16));
              ir7.template select<16, 1>(16) = (v1184_data + v1183_data);
              tensorforge::intel_esimd::simd<float, 16> v1187_data(0.0f);
              v1187_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v1188_data(ir7.template select<16, 1>(32));
              ir7.template select<16, 1>(32) = (v1188_data + v1187_data);
              tensorforge::intel_esimd::simd<float, 16> v1191_data(0.0f);
              v1191_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v1192_data(ir7.template select<16, 1>(48));
              ir7.template select<16, 1>(48) = (v1192_data + v1191_data);
              tensorforge::intel_esimd::simd<float, 16> v1195_data(0.0f);
              v1195_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v1196_data(ir7.template select<16, 1>(64));
              ir7.template select<16, 1>(64) = (v1196_data + v1195_data);
              tensorforge::intel_esimd::simd<float, 16> v1199_data(0.0f);
              v1199_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v1200_data(ir7.template select<16, 1>(80));
              ir7.template select<16, 1>(80) = (v1200_data + v1199_data);
              tensorforge::intel_esimd::simd<float, 16> v1203_data(0.0f);
              v1203_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v1204_data(ir7.template select<16, 1>(96));
              ir7.template select<16, 1>(96) = (v1204_data + v1203_data);
              tensorforge::intel_esimd::simd<float, 16> v1207_data(0.0f);
              v1207_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v1208_data(ir7.template select<16, 1>(112));
              ir7.template select<16, 1>(112) = (v1208_data + v1207_data);
              tensorforge::intel_esimd::simd<float, 16> v1211_data(0.0f);
              v1211_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v1212_data(ir7.template select<16, 1>(128));
              ir7.template select<16, 1>(128) = (v1212_data + v1211_data);
              tensorforge::intel_esimd::simd<float, 16> v1215_data(0.0f);
              v1215_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v1216_data(ir7.template select<16, 1>(144));
              ir7.template select<16, 1>(144) = (v1216_data + v1215_data);
              tensorforge::intel_esimd::simd<float, 16> v1219_data(0.0f);
              v1219_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v1220_data(ir7.template select<16, 1>(160));
              ir7.template select<16, 1>(160) = (v1220_data + v1219_data);
              tensorforge::intel_esimd::simd<float, 16> v1223_data(0.0f);
              v1223_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v1224_data(ir7.template select<16, 1>(176));
              ir7.template select<16, 1>(176) = (v1224_data + v1223_data);
              // r7 = ir7
              #pragma unroll
              for (int32_t v1226_n1 = 0; v1226_n1 < 12; ++v1226_n1) {
                int32_t v1227_a = v1226_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1229_data(ir7.template select<12, 1>(v1227_a));
                r7.template select<12, 1>(v1227_a) = v1229_data;
              }
              // glb_m5 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1230_i1 = 0; v1230_i1 < 12; ++v1230_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1233_data(r7.template select<12, 1>((v1230_i1 * 16)));
                v1233_data.copy_to(glb_m5 + ((v1230_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


// === base name ===
kernel_905e0b5092832d2c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_905e0b5092832d2c = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_905e0b5092832d2c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_905e0b5092832d2c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_905e0b5092832d2c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_905e0b5092832d2c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_905e0b5092832d2c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_905e0b5092832d2c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_905e0b5092832d2c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 12×12(12×12) {0..12}×{0..12} strided
        //   m4 2×12(2×12) {0..2}×{0..12} strided
        //   m5 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
        //   m3[i,j] = t0[i,j]
        //   t0[i,j]@{6..12}×{0..12} = m4[i,k] × m1[k,j]
        //   m5[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N2","bbox":[[0,0],[2,12]],"name":"m4","ordered":false,"parts":1,"shape":[2,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[2,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
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
              const float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 24 + 0 + m4_extraOffset];
              float *const __restrict__ glb_m5 = &m5[v9_batchId0 * 144 + 0 + m5_extraOffset];
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
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v383_i1 = 0; v383_i1 < 12; ++v383_i1) {
                tensorforge::intel_esimd::simd<float, 6> v388_data;
                v388_data.copy_from(glb_m2 + ((v383_i1 * 6)));
                r2.template select<6, 1>((v383_i1 * 16)) = v388_data;
              }
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v48_acc{};
              tensorforge::intel_esimd::simd<float, 16> v52_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              float v53_bc = static_cast<float>(v52_data[0]);
              v48_acc += (v53_bc * v36_data);
              float v55_bc = static_cast<float>(v52_data[1]);
              v48_acc += (v55_bc * v37_data);
              float v57_bc = static_cast<float>(v52_data[2]);
              v48_acc += (v57_bc * v38_data);
              float v59_bc = static_cast<float>(v52_data[3]);
              v48_acc += (v59_bc * v39_data);
              float v61_bc = static_cast<float>(v52_data[4]);
              v48_acc += (v61_bc * v40_data);
              float v63_bc = static_cast<float>(v52_data[5]);
              v48_acc += (v63_bc * v41_data);
              float v65_bc = static_cast<float>(v52_data[6]);
              v48_acc += (v65_bc * v42_data);
              float v67_bc = static_cast<float>(v52_data[7]);
              v48_acc += (v67_bc * v43_data);
              float v69_bc = static_cast<float>(v52_data[8]);
              v48_acc += (v69_bc * v44_data);
              float v71_bc = static_cast<float>(v52_data[9]);
              v48_acc += (v71_bc * v45_data);
              float v73_bc = static_cast<float>(v52_data[10]);
              v48_acc += (v73_bc * v46_data);
              float v75_bc = static_cast<float>(v52_data[11]);
              v48_acc += (v75_bc * v47_data);
              r1.template select<16, 1>(0) = v48_acc;
              tensorforge::intel_esimd::simd<float, 16> v77_acc{};
              tensorforge::intel_esimd::simd<float, 16> v79_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              float v80_bc = static_cast<float>(v79_data[0]);
              v77_acc += (v80_bc * v36_data);
              float v82_bc = static_cast<float>(v79_data[1]);
              v77_acc += (v82_bc * v37_data);
              float v84_bc = static_cast<float>(v79_data[2]);
              v77_acc += (v84_bc * v38_data);
              float v86_bc = static_cast<float>(v79_data[3]);
              v77_acc += (v86_bc * v39_data);
              float v88_bc = static_cast<float>(v79_data[4]);
              v77_acc += (v88_bc * v40_data);
              float v90_bc = static_cast<float>(v79_data[5]);
              v77_acc += (v90_bc * v41_data);
              float v92_bc = static_cast<float>(v79_data[6]);
              v77_acc += (v92_bc * v42_data);
              float v94_bc = static_cast<float>(v79_data[7]);
              v77_acc += (v94_bc * v43_data);
              float v96_bc = static_cast<float>(v79_data[8]);
              v77_acc += (v96_bc * v44_data);
              float v98_bc = static_cast<float>(v79_data[9]);
              v77_acc += (v98_bc * v45_data);
              float v100_bc = static_cast<float>(v79_data[10]);
              v77_acc += (v100_bc * v46_data);
              float v102_bc = static_cast<float>(v79_data[11]);
              v77_acc += (v102_bc * v47_data);
              r1.template select<16, 1>(16) = v77_acc;
              tensorforge::intel_esimd::simd<float, 16> v104_acc{};
              tensorforge::intel_esimd::simd<float, 16> v106_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              float v107_bc = static_cast<float>(v106_data[0]);
              v104_acc += (v107_bc * v36_data);
              float v109_bc = static_cast<float>(v106_data[1]);
              v104_acc += (v109_bc * v37_data);
              float v111_bc = static_cast<float>(v106_data[2]);
              v104_acc += (v111_bc * v38_data);
              float v113_bc = static_cast<float>(v106_data[3]);
              v104_acc += (v113_bc * v39_data);
              float v115_bc = static_cast<float>(v106_data[4]);
              v104_acc += (v115_bc * v40_data);
              float v117_bc = static_cast<float>(v106_data[5]);
              v104_acc += (v117_bc * v41_data);
              float v119_bc = static_cast<float>(v106_data[6]);
              v104_acc += (v119_bc * v42_data);
              float v121_bc = static_cast<float>(v106_data[7]);
              v104_acc += (v121_bc * v43_data);
              float v123_bc = static_cast<float>(v106_data[8]);
              v104_acc += (v123_bc * v44_data);
              float v125_bc = static_cast<float>(v106_data[9]);
              v104_acc += (v125_bc * v45_data);
              float v127_bc = static_cast<float>(v106_data[10]);
              v104_acc += (v127_bc * v46_data);
              float v129_bc = static_cast<float>(v106_data[11]);
              v104_acc += (v129_bc * v47_data);
              r1.template select<16, 1>(32) = v104_acc;
              tensorforge::intel_esimd::simd<float, 16> v131_acc{};
              tensorforge::intel_esimd::simd<float, 16> v133_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              float v134_bc = static_cast<float>(v133_data[0]);
              v131_acc += (v134_bc * v36_data);
              float v136_bc = static_cast<float>(v133_data[1]);
              v131_acc += (v136_bc * v37_data);
              float v138_bc = static_cast<float>(v133_data[2]);
              v131_acc += (v138_bc * v38_data);
              float v140_bc = static_cast<float>(v133_data[3]);
              v131_acc += (v140_bc * v39_data);
              float v142_bc = static_cast<float>(v133_data[4]);
              v131_acc += (v142_bc * v40_data);
              float v144_bc = static_cast<float>(v133_data[5]);
              v131_acc += (v144_bc * v41_data);
              float v146_bc = static_cast<float>(v133_data[6]);
              v131_acc += (v146_bc * v42_data);
              float v148_bc = static_cast<float>(v133_data[7]);
              v131_acc += (v148_bc * v43_data);
              float v150_bc = static_cast<float>(v133_data[8]);
              v131_acc += (v150_bc * v44_data);
              float v152_bc = static_cast<float>(v133_data[9]);
              v131_acc += (v152_bc * v45_data);
              float v154_bc = static_cast<float>(v133_data[10]);
              v131_acc += (v154_bc * v46_data);
              float v156_bc = static_cast<float>(v133_data[11]);
              v131_acc += (v156_bc * v47_data);
              r1.template select<16, 1>(48) = v131_acc;
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              float v161_bc = static_cast<float>(v160_data[0]);
              v158_acc += (v161_bc * v36_data);
              float v163_bc = static_cast<float>(v160_data[1]);
              v158_acc += (v163_bc * v37_data);
              float v165_bc = static_cast<float>(v160_data[2]);
              v158_acc += (v165_bc * v38_data);
              float v167_bc = static_cast<float>(v160_data[3]);
              v158_acc += (v167_bc * v39_data);
              float v169_bc = static_cast<float>(v160_data[4]);
              v158_acc += (v169_bc * v40_data);
              float v171_bc = static_cast<float>(v160_data[5]);
              v158_acc += (v171_bc * v41_data);
              float v173_bc = static_cast<float>(v160_data[6]);
              v158_acc += (v173_bc * v42_data);
              float v175_bc = static_cast<float>(v160_data[7]);
              v158_acc += (v175_bc * v43_data);
              float v177_bc = static_cast<float>(v160_data[8]);
              v158_acc += (v177_bc * v44_data);
              float v179_bc = static_cast<float>(v160_data[9]);
              v158_acc += (v179_bc * v45_data);
              float v181_bc = static_cast<float>(v160_data[10]);
              v158_acc += (v181_bc * v46_data);
              float v183_bc = static_cast<float>(v160_data[11]);
              v158_acc += (v183_bc * v47_data);
              r1.template select<16, 1>(64) = v158_acc;
              tensorforge::intel_esimd::simd<float, 16> v185_acc{};
              tensorforge::intel_esimd::simd<float, 16> v187_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              float v188_bc = static_cast<float>(v187_data[0]);
              v185_acc += (v188_bc * v36_data);
              float v190_bc = static_cast<float>(v187_data[1]);
              v185_acc += (v190_bc * v37_data);
              float v192_bc = static_cast<float>(v187_data[2]);
              v185_acc += (v192_bc * v38_data);
              float v194_bc = static_cast<float>(v187_data[3]);
              v185_acc += (v194_bc * v39_data);
              float v196_bc = static_cast<float>(v187_data[4]);
              v185_acc += (v196_bc * v40_data);
              float v198_bc = static_cast<float>(v187_data[5]);
              v185_acc += (v198_bc * v41_data);
              float v200_bc = static_cast<float>(v187_data[6]);
              v185_acc += (v200_bc * v42_data);
              float v202_bc = static_cast<float>(v187_data[7]);
              v185_acc += (v202_bc * v43_data);
              float v204_bc = static_cast<float>(v187_data[8]);
              v185_acc += (v204_bc * v44_data);
              float v206_bc = static_cast<float>(v187_data[9]);
              v185_acc += (v206_bc * v45_data);
              float v208_bc = static_cast<float>(v187_data[10]);
              v185_acc += (v208_bc * v46_data);
              float v210_bc = static_cast<float>(v187_data[11]);
              v185_acc += (v210_bc * v47_data);
              r1.template select<16, 1>(80) = v185_acc;
              tensorforge::intel_esimd::simd<float, 16> v212_acc{};
              tensorforge::intel_esimd::simd<float, 16> v214_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              float v215_bc = static_cast<float>(v214_data[0]);
              v212_acc += (v215_bc * v36_data);
              float v217_bc = static_cast<float>(v214_data[1]);
              v212_acc += (v217_bc * v37_data);
              float v219_bc = static_cast<float>(v214_data[2]);
              v212_acc += (v219_bc * v38_data);
              float v221_bc = static_cast<float>(v214_data[3]);
              v212_acc += (v221_bc * v39_data);
              float v223_bc = static_cast<float>(v214_data[4]);
              v212_acc += (v223_bc * v40_data);
              float v225_bc = static_cast<float>(v214_data[5]);
              v212_acc += (v225_bc * v41_data);
              float v227_bc = static_cast<float>(v214_data[6]);
              v212_acc += (v227_bc * v42_data);
              float v229_bc = static_cast<float>(v214_data[7]);
              v212_acc += (v229_bc * v43_data);
              float v231_bc = static_cast<float>(v214_data[8]);
              v212_acc += (v231_bc * v44_data);
              float v233_bc = static_cast<float>(v214_data[9]);
              v212_acc += (v233_bc * v45_data);
              float v235_bc = static_cast<float>(v214_data[10]);
              v212_acc += (v235_bc * v46_data);
              float v237_bc = static_cast<float>(v214_data[11]);
              v212_acc += (v237_bc * v47_data);
              r1.template select<16, 1>(96) = v212_acc;
              tensorforge::intel_esimd::simd<float, 16> v239_acc{};
              tensorforge::intel_esimd::simd<float, 16> v241_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              float v242_bc = static_cast<float>(v241_data[0]);
              v239_acc += (v242_bc * v36_data);
              float v244_bc = static_cast<float>(v241_data[1]);
              v239_acc += (v244_bc * v37_data);
              float v246_bc = static_cast<float>(v241_data[2]);
              v239_acc += (v246_bc * v38_data);
              float v248_bc = static_cast<float>(v241_data[3]);
              v239_acc += (v248_bc * v39_data);
              float v250_bc = static_cast<float>(v241_data[4]);
              v239_acc += (v250_bc * v40_data);
              float v252_bc = static_cast<float>(v241_data[5]);
              v239_acc += (v252_bc * v41_data);
              float v254_bc = static_cast<float>(v241_data[6]);
              v239_acc += (v254_bc * v42_data);
              float v256_bc = static_cast<float>(v241_data[7]);
              v239_acc += (v256_bc * v43_data);
              float v258_bc = static_cast<float>(v241_data[8]);
              v239_acc += (v258_bc * v44_data);
              float v260_bc = static_cast<float>(v241_data[9]);
              v239_acc += (v260_bc * v45_data);
              float v262_bc = static_cast<float>(v241_data[10]);
              v239_acc += (v262_bc * v46_data);
              float v264_bc = static_cast<float>(v241_data[11]);
              v239_acc += (v264_bc * v47_data);
              r1.template select<16, 1>(112) = v239_acc;
              tensorforge::intel_esimd::simd<float, 16> v266_acc{};
              tensorforge::intel_esimd::simd<float, 16> v268_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              float v269_bc = static_cast<float>(v268_data[0]);
              v266_acc += (v269_bc * v36_data);
              float v271_bc = static_cast<float>(v268_data[1]);
              v266_acc += (v271_bc * v37_data);
              float v273_bc = static_cast<float>(v268_data[2]);
              v266_acc += (v273_bc * v38_data);
              float v275_bc = static_cast<float>(v268_data[3]);
              v266_acc += (v275_bc * v39_data);
              float v277_bc = static_cast<float>(v268_data[4]);
              v266_acc += (v277_bc * v40_data);
              float v279_bc = static_cast<float>(v268_data[5]);
              v266_acc += (v279_bc * v41_data);
              float v281_bc = static_cast<float>(v268_data[6]);
              v266_acc += (v281_bc * v42_data);
              float v283_bc = static_cast<float>(v268_data[7]);
              v266_acc += (v283_bc * v43_data);
              float v285_bc = static_cast<float>(v268_data[8]);
              v266_acc += (v285_bc * v44_data);
              float v287_bc = static_cast<float>(v268_data[9]);
              v266_acc += (v287_bc * v45_data);
              float v289_bc = static_cast<float>(v268_data[10]);
              v266_acc += (v289_bc * v46_data);
              float v291_bc = static_cast<float>(v268_data[11]);
              v266_acc += (v291_bc * v47_data);
              r1.template select<16, 1>(128) = v266_acc;
              tensorforge::intel_esimd::simd<float, 16> v293_acc{};
              tensorforge::intel_esimd::simd<float, 16> v295_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              float v296_bc = static_cast<float>(v295_data[0]);
              v293_acc += (v296_bc * v36_data);
              float v298_bc = static_cast<float>(v295_data[1]);
              v293_acc += (v298_bc * v37_data);
              float v300_bc = static_cast<float>(v295_data[2]);
              v293_acc += (v300_bc * v38_data);
              float v302_bc = static_cast<float>(v295_data[3]);
              v293_acc += (v302_bc * v39_data);
              float v304_bc = static_cast<float>(v295_data[4]);
              v293_acc += (v304_bc * v40_data);
              float v306_bc = static_cast<float>(v295_data[5]);
              v293_acc += (v306_bc * v41_data);
              float v308_bc = static_cast<float>(v295_data[6]);
              v293_acc += (v308_bc * v42_data);
              float v310_bc = static_cast<float>(v295_data[7]);
              v293_acc += (v310_bc * v43_data);
              float v312_bc = static_cast<float>(v295_data[8]);
              v293_acc += (v312_bc * v44_data);
              float v314_bc = static_cast<float>(v295_data[9]);
              v293_acc += (v314_bc * v45_data);
              float v316_bc = static_cast<float>(v295_data[10]);
              v293_acc += (v316_bc * v46_data);
              float v318_bc = static_cast<float>(v295_data[11]);
              v293_acc += (v318_bc * v47_data);
              r1.template select<16, 1>(144) = v293_acc;
              tensorforge::intel_esimd::simd<float, 16> v320_acc{};
              tensorforge::intel_esimd::simd<float, 16> v322_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              float v323_bc = static_cast<float>(v322_data[0]);
              v320_acc += (v323_bc * v36_data);
              float v325_bc = static_cast<float>(v322_data[1]);
              v320_acc += (v325_bc * v37_data);
              float v327_bc = static_cast<float>(v322_data[2]);
              v320_acc += (v327_bc * v38_data);
              float v329_bc = static_cast<float>(v322_data[3]);
              v320_acc += (v329_bc * v39_data);
              float v331_bc = static_cast<float>(v322_data[4]);
              v320_acc += (v331_bc * v40_data);
              float v333_bc = static_cast<float>(v322_data[5]);
              v320_acc += (v333_bc * v41_data);
              float v335_bc = static_cast<float>(v322_data[6]);
              v320_acc += (v335_bc * v42_data);
              float v337_bc = static_cast<float>(v322_data[7]);
              v320_acc += (v337_bc * v43_data);
              float v339_bc = static_cast<float>(v322_data[8]);
              v320_acc += (v339_bc * v44_data);
              float v341_bc = static_cast<float>(v322_data[9]);
              v320_acc += (v341_bc * v45_data);
              float v343_bc = static_cast<float>(v322_data[10]);
              v320_acc += (v343_bc * v46_data);
              float v345_bc = static_cast<float>(v322_data[11]);
              v320_acc += (v345_bc * v47_data);
              r1.template select<16, 1>(160) = v320_acc;
              tensorforge::intel_esimd::simd<float, 16> v347_acc{};
              tensorforge::intel_esimd::simd<float, 16> v349_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              float v350_bc = static_cast<float>(v349_data[0]);
              v347_acc += (v350_bc * v36_data);
              float v352_bc = static_cast<float>(v349_data[1]);
              v347_acc += (v352_bc * v37_data);
              float v354_bc = static_cast<float>(v349_data[2]);
              v347_acc += (v354_bc * v38_data);
              float v356_bc = static_cast<float>(v349_data[3]);
              v347_acc += (v356_bc * v39_data);
              float v358_bc = static_cast<float>(v349_data[4]);
              v347_acc += (v358_bc * v40_data);
              float v360_bc = static_cast<float>(v349_data[5]);
              v347_acc += (v360_bc * v41_data);
              float v362_bc = static_cast<float>(v349_data[6]);
              v347_acc += (v362_bc * v42_data);
              float v364_bc = static_cast<float>(v349_data[7]);
              v347_acc += (v364_bc * v43_data);
              float v366_bc = static_cast<float>(v349_data[8]);
              v347_acc += (v366_bc * v44_data);
              float v368_bc = static_cast<float>(v349_data[9]);
              v347_acc += (v368_bc * v45_data);
              float v370_bc = static_cast<float>(v349_data[10]);
              v347_acc += (v370_bc * v46_data);
              float v372_bc = static_cast<float>(v349_data[11]);
              v347_acc += (v372_bc * v47_data);
              r1.template select<16, 1>(176) = v347_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v374_i1 = 0; v374_i1 < 12; ++v374_i1) {
                tensorforge::intel_esimd::simd<float, 6> v377_data(r1.template select<6, 1>((v374_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v374_i1 * 12)), v377_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // r5 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v809_i1 = 0; v809_i1 < 12; ++v809_i1) {
                tensorforge::intel_esimd::simd<float, 2> v814_data;
                v814_data.copy_from(glb_m4 + ((v809_i1 * 2)));
                r5.template select<2, 1>((v809_i1 * 16)) = v814_data;
              }
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s0)
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v393_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v394_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v395_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v396_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v397_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v398_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v399_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v400_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v401_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v402_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v403_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v404_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v405_acc{};
              v405_acc += (v53_bc * v393_data);
              v405_acc += (v55_bc * v394_data);
              v405_acc += (v57_bc * v395_data);
              v405_acc += (v59_bc * v396_data);
              v405_acc += (v61_bc * v397_data);
              v405_acc += (v63_bc * v398_data);
              v405_acc += (v65_bc * v399_data);
              v405_acc += (v67_bc * v400_data);
              v405_acc += (v69_bc * v401_data);
              v405_acc += (v71_bc * v402_data);
              v405_acc += (v73_bc * v403_data);
              v405_acc += (v75_bc * v404_data);
              ir3.template select<16, 1>(0) = v405_acc;
              tensorforge::intel_esimd::simd<float, 16> v434_acc{};
              v434_acc += (v80_bc * v393_data);
              v434_acc += (v82_bc * v394_data);
              v434_acc += (v84_bc * v395_data);
              v434_acc += (v86_bc * v396_data);
              v434_acc += (v88_bc * v397_data);
              v434_acc += (v90_bc * v398_data);
              v434_acc += (v92_bc * v399_data);
              v434_acc += (v94_bc * v400_data);
              v434_acc += (v96_bc * v401_data);
              v434_acc += (v98_bc * v402_data);
              v434_acc += (v100_bc * v403_data);
              v434_acc += (v102_bc * v404_data);
              ir3.template select<16, 1>(16) = v434_acc;
              tensorforge::intel_esimd::simd<float, 16> v461_acc{};
              v461_acc += (v107_bc * v393_data);
              v461_acc += (v109_bc * v394_data);
              v461_acc += (v111_bc * v395_data);
              v461_acc += (v113_bc * v396_data);
              v461_acc += (v115_bc * v397_data);
              v461_acc += (v117_bc * v398_data);
              v461_acc += (v119_bc * v399_data);
              v461_acc += (v121_bc * v400_data);
              v461_acc += (v123_bc * v401_data);
              v461_acc += (v125_bc * v402_data);
              v461_acc += (v127_bc * v403_data);
              v461_acc += (v129_bc * v404_data);
              ir3.template select<16, 1>(32) = v461_acc;
              tensorforge::intel_esimd::simd<float, 16> v488_acc{};
              v488_acc += (v134_bc * v393_data);
              v488_acc += (v136_bc * v394_data);
              v488_acc += (v138_bc * v395_data);
              v488_acc += (v140_bc * v396_data);
              v488_acc += (v142_bc * v397_data);
              v488_acc += (v144_bc * v398_data);
              v488_acc += (v146_bc * v399_data);
              v488_acc += (v148_bc * v400_data);
              v488_acc += (v150_bc * v401_data);
              v488_acc += (v152_bc * v402_data);
              v488_acc += (v154_bc * v403_data);
              v488_acc += (v156_bc * v404_data);
              ir3.template select<16, 1>(48) = v488_acc;
              tensorforge::intel_esimd::simd<float, 16> v515_acc{};
              v515_acc += (v161_bc * v393_data);
              v515_acc += (v163_bc * v394_data);
              v515_acc += (v165_bc * v395_data);
              v515_acc += (v167_bc * v396_data);
              v515_acc += (v169_bc * v397_data);
              v515_acc += (v171_bc * v398_data);
              v515_acc += (v173_bc * v399_data);
              v515_acc += (v175_bc * v400_data);
              v515_acc += (v177_bc * v401_data);
              v515_acc += (v179_bc * v402_data);
              v515_acc += (v181_bc * v403_data);
              v515_acc += (v183_bc * v404_data);
              ir3.template select<16, 1>(64) = v515_acc;
              tensorforge::intel_esimd::simd<float, 16> v542_acc{};
              v542_acc += (v188_bc * v393_data);
              v542_acc += (v190_bc * v394_data);
              v542_acc += (v192_bc * v395_data);
              v542_acc += (v194_bc * v396_data);
              v542_acc += (v196_bc * v397_data);
              v542_acc += (v198_bc * v398_data);
              v542_acc += (v200_bc * v399_data);
              v542_acc += (v202_bc * v400_data);
              v542_acc += (v204_bc * v401_data);
              v542_acc += (v206_bc * v402_data);
              v542_acc += (v208_bc * v403_data);
              v542_acc += (v210_bc * v404_data);
              ir3.template select<16, 1>(80) = v542_acc;
              tensorforge::intel_esimd::simd<float, 16> v569_acc{};
              v569_acc += (v215_bc * v393_data);
              v569_acc += (v217_bc * v394_data);
              v569_acc += (v219_bc * v395_data);
              v569_acc += (v221_bc * v396_data);
              v569_acc += (v223_bc * v397_data);
              v569_acc += (v225_bc * v398_data);
              v569_acc += (v227_bc * v399_data);
              v569_acc += (v229_bc * v400_data);
              v569_acc += (v231_bc * v401_data);
              v569_acc += (v233_bc * v402_data);
              v569_acc += (v235_bc * v403_data);
              v569_acc += (v237_bc * v404_data);
              ir3.template select<16, 1>(96) = v569_acc;
              tensorforge::intel_esimd::simd<float, 16> v596_acc{};
              v596_acc += (v242_bc * v393_data);
              v596_acc += (v244_bc * v394_data);
              v596_acc += (v246_bc * v395_data);
              v596_acc += (v248_bc * v396_data);
              v596_acc += (v250_bc * v397_data);
              v596_acc += (v252_bc * v398_data);
              v596_acc += (v254_bc * v399_data);
              v596_acc += (v256_bc * v400_data);
              v596_acc += (v258_bc * v401_data);
              v596_acc += (v260_bc * v402_data);
              v596_acc += (v262_bc * v403_data);
              v596_acc += (v264_bc * v404_data);
              ir3.template select<16, 1>(112) = v596_acc;
              tensorforge::intel_esimd::simd<float, 16> v623_acc{};
              v623_acc += (v269_bc * v393_data);
              v623_acc += (v271_bc * v394_data);
              v623_acc += (v273_bc * v395_data);
              v623_acc += (v275_bc * v396_data);
              v623_acc += (v277_bc * v397_data);
              v623_acc += (v279_bc * v398_data);
              v623_acc += (v281_bc * v399_data);
              v623_acc += (v283_bc * v400_data);
              v623_acc += (v285_bc * v401_data);
              v623_acc += (v287_bc * v402_data);
              v623_acc += (v289_bc * v403_data);
              v623_acc += (v291_bc * v404_data);
              ir3.template select<16, 1>(128) = v623_acc;
              tensorforge::intel_esimd::simd<float, 16> v650_acc{};
              v650_acc += (v296_bc * v393_data);
              v650_acc += (v298_bc * v394_data);
              v650_acc += (v300_bc * v395_data);
              v650_acc += (v302_bc * v396_data);
              v650_acc += (v304_bc * v397_data);
              v650_acc += (v306_bc * v398_data);
              v650_acc += (v308_bc * v399_data);
              v650_acc += (v310_bc * v400_data);
              v650_acc += (v312_bc * v401_data);
              v650_acc += (v314_bc * v402_data);
              v650_acc += (v316_bc * v403_data);
              v650_acc += (v318_bc * v404_data);
              ir3.template select<16, 1>(144) = v650_acc;
              tensorforge::intel_esimd::simd<float, 16> v677_acc{};
              v677_acc += (v323_bc * v393_data);
              v677_acc += (v325_bc * v394_data);
              v677_acc += (v327_bc * v395_data);
              v677_acc += (v329_bc * v396_data);
              v677_acc += (v331_bc * v397_data);
              v677_acc += (v333_bc * v398_data);
              v677_acc += (v335_bc * v399_data);
              v677_acc += (v337_bc * v400_data);
              v677_acc += (v339_bc * v401_data);
              v677_acc += (v341_bc * v402_data);
              v677_acc += (v343_bc * v403_data);
              v677_acc += (v345_bc * v404_data);
              ir3.template select<16, 1>(160) = v677_acc;
              tensorforge::intel_esimd::simd<float, 16> v704_acc{};
              v704_acc += (v350_bc * v393_data);
              v704_acc += (v352_bc * v394_data);
              v704_acc += (v354_bc * v395_data);
              v704_acc += (v356_bc * v396_data);
              v704_acc += (v358_bc * v397_data);
              v704_acc += (v360_bc * v398_data);
              v704_acc += (v362_bc * v399_data);
              v704_acc += (v364_bc * v400_data);
              v704_acc += (v366_bc * v401_data);
              v704_acc += (v368_bc * v402_data);
              v704_acc += (v370_bc * v403_data);
              v704_acc += (v372_bc * v404_data);
              ir3.template select<16, 1>(176) = v704_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v731_n1 = 0; v731_n1 < 12; ++v731_n1) {
                int32_t v732_a = v731_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v734_data(ir3.template select<6, 1>(v732_a));
                r3.template select<6, 1>(v732_a) = v734_data;
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v735_i1 = 0; v735_i1 < 12; ++v735_i1) {
                tensorforge::intel_esimd::simd<float, 6> v738_data(r3.template select<6, 1>((v735_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v735_i1 * 12))), v738_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // ir4 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir4(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v749_data(0.0f);
              v749_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v750_data(ir4.template select<16, 1>(0));
              ir4.template select<16, 1>(0) = (v750_data + v749_data);
              tensorforge::intel_esimd::simd<float, 16> v753_data(0.0f);
              v753_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v754_data(ir4.template select<16, 1>(16));
              ir4.template select<16, 1>(16) = (v754_data + v753_data);
              tensorforge::intel_esimd::simd<float, 16> v757_data(0.0f);
              v757_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v758_data(ir4.template select<16, 1>(32));
              ir4.template select<16, 1>(32) = (v758_data + v757_data);
              tensorforge::intel_esimd::simd<float, 16> v761_data(0.0f);
              v761_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v762_data(ir4.template select<16, 1>(48));
              ir4.template select<16, 1>(48) = (v762_data + v761_data);
              tensorforge::intel_esimd::simd<float, 16> v765_data(0.0f);
              v765_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v766_data(ir4.template select<16, 1>(64));
              ir4.template select<16, 1>(64) = (v766_data + v765_data);
              tensorforge::intel_esimd::simd<float, 16> v769_data(0.0f);
              v769_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v770_data(ir4.template select<16, 1>(80));
              ir4.template select<16, 1>(80) = (v770_data + v769_data);
              tensorforge::intel_esimd::simd<float, 16> v773_data(0.0f);
              v773_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v774_data(ir4.template select<16, 1>(96));
              ir4.template select<16, 1>(96) = (v774_data + v773_data);
              tensorforge::intel_esimd::simd<float, 16> v777_data(0.0f);
              v777_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v778_data(ir4.template select<16, 1>(112));
              ir4.template select<16, 1>(112) = (v778_data + v777_data);
              tensorforge::intel_esimd::simd<float, 16> v781_data(0.0f);
              v781_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v782_data(ir4.template select<16, 1>(128));
              ir4.template select<16, 1>(128) = (v782_data + v781_data);
              tensorforge::intel_esimd::simd<float, 16> v785_data(0.0f);
              v785_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v786_data(ir4.template select<16, 1>(144));
              ir4.template select<16, 1>(144) = (v786_data + v785_data);
              tensorforge::intel_esimd::simd<float, 16> v789_data(0.0f);
              v789_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v790_data(ir4.template select<16, 1>(160));
              ir4.template select<16, 1>(160) = (v790_data + v789_data);
              tensorforge::intel_esimd::simd<float, 16> v793_data(0.0f);
              v793_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v794_data(ir4.template select<16, 1>(176));
              ir4.template select<16, 1>(176) = (v794_data + v793_data);
              // r4 = ir4
              #pragma unroll
              for (int32_t v796_n1 = 0; v796_n1 < 12; ++v796_n1) {
                int32_t v797_a = v796_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v799_data(ir4.template select<12, 1>(v797_a));
                r4.template select<12, 1>(v797_a) = v799_data;
              }
              // glb_m3 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v800_i1 = 0; v800_i1 < 12; ++v800_i1) {
                tensorforge::intel_esimd::simd<float, 12> v803_data(r4.template select<12, 1>((v800_i1 * 16)));
                v803_data.copy_to(glb_m3 + ((v800_i1 * 12)));
              }
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // ir6 = +(r5 * s0)
              // [(0, 2), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v819_data(r5.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v820_data(r5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v821_data(r5.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v822_data(r5.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v823_data(r5.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v824_data(r5.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v825_data(r5.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v826_data(r5.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v827_data(r5.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v828_data(r5.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v829_data(r5.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v830_data(r5.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v831_acc{};
              v831_acc += (v53_bc * v819_data);
              v831_acc += (v55_bc * v820_data);
              v831_acc += (v57_bc * v821_data);
              v831_acc += (v59_bc * v822_data);
              v831_acc += (v61_bc * v823_data);
              v831_acc += (v63_bc * v824_data);
              v831_acc += (v65_bc * v825_data);
              v831_acc += (v67_bc * v826_data);
              v831_acc += (v69_bc * v827_data);
              v831_acc += (v71_bc * v828_data);
              v831_acc += (v73_bc * v829_data);
              v831_acc += (v75_bc * v830_data);
              ir6.template select<16, 1>(0) = v831_acc;
              tensorforge::intel_esimd::simd<float, 16> v860_acc{};
              v860_acc += (v80_bc * v819_data);
              v860_acc += (v82_bc * v820_data);
              v860_acc += (v84_bc * v821_data);
              v860_acc += (v86_bc * v822_data);
              v860_acc += (v88_bc * v823_data);
              v860_acc += (v90_bc * v824_data);
              v860_acc += (v92_bc * v825_data);
              v860_acc += (v94_bc * v826_data);
              v860_acc += (v96_bc * v827_data);
              v860_acc += (v98_bc * v828_data);
              v860_acc += (v100_bc * v829_data);
              v860_acc += (v102_bc * v830_data);
              ir6.template select<16, 1>(16) = v860_acc;
              tensorforge::intel_esimd::simd<float, 16> v887_acc{};
              v887_acc += (v107_bc * v819_data);
              v887_acc += (v109_bc * v820_data);
              v887_acc += (v111_bc * v821_data);
              v887_acc += (v113_bc * v822_data);
              v887_acc += (v115_bc * v823_data);
              v887_acc += (v117_bc * v824_data);
              v887_acc += (v119_bc * v825_data);
              v887_acc += (v121_bc * v826_data);
              v887_acc += (v123_bc * v827_data);
              v887_acc += (v125_bc * v828_data);
              v887_acc += (v127_bc * v829_data);
              v887_acc += (v129_bc * v830_data);
              ir6.template select<16, 1>(32) = v887_acc;
              tensorforge::intel_esimd::simd<float, 16> v914_acc{};
              v914_acc += (v134_bc * v819_data);
              v914_acc += (v136_bc * v820_data);
              v914_acc += (v138_bc * v821_data);
              v914_acc += (v140_bc * v822_data);
              v914_acc += (v142_bc * v823_data);
              v914_acc += (v144_bc * v824_data);
              v914_acc += (v146_bc * v825_data);
              v914_acc += (v148_bc * v826_data);
              v914_acc += (v150_bc * v827_data);
              v914_acc += (v152_bc * v828_data);
              v914_acc += (v154_bc * v829_data);
              v914_acc += (v156_bc * v830_data);
              ir6.template select<16, 1>(48) = v914_acc;
              tensorforge::intel_esimd::simd<float, 16> v941_acc{};
              v941_acc += (v161_bc * v819_data);
              v941_acc += (v163_bc * v820_data);
              v941_acc += (v165_bc * v821_data);
              v941_acc += (v167_bc * v822_data);
              v941_acc += (v169_bc * v823_data);
              v941_acc += (v171_bc * v824_data);
              v941_acc += (v173_bc * v825_data);
              v941_acc += (v175_bc * v826_data);
              v941_acc += (v177_bc * v827_data);
              v941_acc += (v179_bc * v828_data);
              v941_acc += (v181_bc * v829_data);
              v941_acc += (v183_bc * v830_data);
              ir6.template select<16, 1>(64) = v941_acc;
              tensorforge::intel_esimd::simd<float, 16> v968_acc{};
              v968_acc += (v188_bc * v819_data);
              v968_acc += (v190_bc * v820_data);
              v968_acc += (v192_bc * v821_data);
              v968_acc += (v194_bc * v822_data);
              v968_acc += (v196_bc * v823_data);
              v968_acc += (v198_bc * v824_data);
              v968_acc += (v200_bc * v825_data);
              v968_acc += (v202_bc * v826_data);
              v968_acc += (v204_bc * v827_data);
              v968_acc += (v206_bc * v828_data);
              v968_acc += (v208_bc * v829_data);
              v968_acc += (v210_bc * v830_data);
              ir6.template select<16, 1>(80) = v968_acc;
              tensorforge::intel_esimd::simd<float, 16> v995_acc{};
              v995_acc += (v215_bc * v819_data);
              v995_acc += (v217_bc * v820_data);
              v995_acc += (v219_bc * v821_data);
              v995_acc += (v221_bc * v822_data);
              v995_acc += (v223_bc * v823_data);
              v995_acc += (v225_bc * v824_data);
              v995_acc += (v227_bc * v825_data);
              v995_acc += (v229_bc * v826_data);
              v995_acc += (v231_bc * v827_data);
              v995_acc += (v233_bc * v828_data);
              v995_acc += (v235_bc * v829_data);
              v995_acc += (v237_bc * v830_data);
              ir6.template select<16, 1>(96) = v995_acc;
              tensorforge::intel_esimd::simd<float, 16> v1022_acc{};
              v1022_acc += (v242_bc * v819_data);
              v1022_acc += (v244_bc * v820_data);
              v1022_acc += (v246_bc * v821_data);
              v1022_acc += (v248_bc * v822_data);
              v1022_acc += (v250_bc * v823_data);
              v1022_acc += (v252_bc * v824_data);
              v1022_acc += (v254_bc * v825_data);
              v1022_acc += (v256_bc * v826_data);
              v1022_acc += (v258_bc * v827_data);
              v1022_acc += (v260_bc * v828_data);
              v1022_acc += (v262_bc * v829_data);
              v1022_acc += (v264_bc * v830_data);
              ir6.template select<16, 1>(112) = v1022_acc;
              tensorforge::intel_esimd::simd<float, 16> v1049_acc{};
              v1049_acc += (v269_bc * v819_data);
              v1049_acc += (v271_bc * v820_data);
              v1049_acc += (v273_bc * v821_data);
              v1049_acc += (v275_bc * v822_data);
              v1049_acc += (v277_bc * v823_data);
              v1049_acc += (v279_bc * v824_data);
              v1049_acc += (v281_bc * v825_data);
              v1049_acc += (v283_bc * v826_data);
              v1049_acc += (v285_bc * v827_data);
              v1049_acc += (v287_bc * v828_data);
              v1049_acc += (v289_bc * v829_data);
              v1049_acc += (v291_bc * v830_data);
              ir6.template select<16, 1>(128) = v1049_acc;
              tensorforge::intel_esimd::simd<float, 16> v1076_acc{};
              v1076_acc += (v296_bc * v819_data);
              v1076_acc += (v298_bc * v820_data);
              v1076_acc += (v300_bc * v821_data);
              v1076_acc += (v302_bc * v822_data);
              v1076_acc += (v304_bc * v823_data);
              v1076_acc += (v306_bc * v824_data);
              v1076_acc += (v308_bc * v825_data);
              v1076_acc += (v310_bc * v826_data);
              v1076_acc += (v312_bc * v827_data);
              v1076_acc += (v314_bc * v828_data);
              v1076_acc += (v316_bc * v829_data);
              v1076_acc += (v318_bc * v830_data);
              ir6.template select<16, 1>(144) = v1076_acc;
              tensorforge::intel_esimd::simd<float, 16> v1103_acc{};
              v1103_acc += (v323_bc * v819_data);
              v1103_acc += (v325_bc * v820_data);
              v1103_acc += (v327_bc * v821_data);
              v1103_acc += (v329_bc * v822_data);
              v1103_acc += (v331_bc * v823_data);
              v1103_acc += (v333_bc * v824_data);
              v1103_acc += (v335_bc * v825_data);
              v1103_acc += (v337_bc * v826_data);
              v1103_acc += (v339_bc * v827_data);
              v1103_acc += (v341_bc * v828_data);
              v1103_acc += (v343_bc * v829_data);
              v1103_acc += (v345_bc * v830_data);
              ir6.template select<16, 1>(160) = v1103_acc;
              tensorforge::intel_esimd::simd<float, 16> v1130_acc{};
              v1130_acc += (v350_bc * v819_data);
              v1130_acc += (v352_bc * v820_data);
              v1130_acc += (v354_bc * v821_data);
              v1130_acc += (v356_bc * v822_data);
              v1130_acc += (v358_bc * v823_data);
              v1130_acc += (v360_bc * v824_data);
              v1130_acc += (v362_bc * v825_data);
              v1130_acc += (v364_bc * v826_data);
              v1130_acc += (v366_bc * v827_data);
              v1130_acc += (v368_bc * v828_data);
              v1130_acc += (v370_bc * v829_data);
              v1130_acc += (v372_bc * v830_data);
              ir6.template select<16, 1>(176) = v1130_acc;
              // r6 = ir6
              #pragma unroll
              for (int32_t v1157_n1 = 0; v1157_n1 < 12; ++v1157_n1) {
                int32_t v1158_a = v1157_n1 * 16;
                tensorforge::intel_esimd::simd<float, 2> v1160_data(ir6.template select<2, 1>(v1158_a));
                r6.template select<2, 1>(v1158_a) = v1160_data;
              }
              // s1 = store{r>s, clear}(localShrMem0, r6);
              #pragma unroll
              for (int32_t v1161_z1 = 0; v1161_z1 < 12; ++v1161_z1) {
                s1[(8_i32 + (v1161_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v1168_i1 = 0; v1168_i1 < 12; ++v1168_i1) {
                tensorforge::intel_esimd::simd<float, 2> v1171_data(r6.template select<2, 1>((v1168_i1 * 16)));
                tensorforge::slmStore<float, 2>(s1 + ((6_i32 + (v1168_i1 * 12))), v1171_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r7(0.0f);
              // ir7 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir7(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v1182_data(0.0f);
              v1182_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v1183_data(ir7.template select<16, 1>(0));
              ir7.template select<16, 1>(0) = (v1183_data + v1182_data);
              tensorforge::intel_esimd::simd<float, 16> v1186_data(0.0f);
              v1186_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v1187_data(ir7.template select<16, 1>(16));
              ir7.template select<16, 1>(16) = (v1187_data + v1186_data);
              tensorforge::intel_esimd::simd<float, 16> v1190_data(0.0f);
              v1190_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v1191_data(ir7.template select<16, 1>(32));
              ir7.template select<16, 1>(32) = (v1191_data + v1190_data);
              tensorforge::intel_esimd::simd<float, 16> v1194_data(0.0f);
              v1194_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v1195_data(ir7.template select<16, 1>(48));
              ir7.template select<16, 1>(48) = (v1195_data + v1194_data);
              tensorforge::intel_esimd::simd<float, 16> v1198_data(0.0f);
              v1198_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v1199_data(ir7.template select<16, 1>(64));
              ir7.template select<16, 1>(64) = (v1199_data + v1198_data);
              tensorforge::intel_esimd::simd<float, 16> v1202_data(0.0f);
              v1202_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v1203_data(ir7.template select<16, 1>(80));
              ir7.template select<16, 1>(80) = (v1203_data + v1202_data);
              tensorforge::intel_esimd::simd<float, 16> v1206_data(0.0f);
              v1206_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v1207_data(ir7.template select<16, 1>(96));
              ir7.template select<16, 1>(96) = (v1207_data + v1206_data);
              tensorforge::intel_esimd::simd<float, 16> v1210_data(0.0f);
              v1210_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v1211_data(ir7.template select<16, 1>(112));
              ir7.template select<16, 1>(112) = (v1211_data + v1210_data);
              tensorforge::intel_esimd::simd<float, 16> v1214_data(0.0f);
              v1214_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v1215_data(ir7.template select<16, 1>(128));
              ir7.template select<16, 1>(128) = (v1215_data + v1214_data);
              tensorforge::intel_esimd::simd<float, 16> v1218_data(0.0f);
              v1218_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v1219_data(ir7.template select<16, 1>(144));
              ir7.template select<16, 1>(144) = (v1219_data + v1218_data);
              tensorforge::intel_esimd::simd<float, 16> v1222_data(0.0f);
              v1222_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v1223_data(ir7.template select<16, 1>(160));
              ir7.template select<16, 1>(160) = (v1223_data + v1222_data);
              tensorforge::intel_esimd::simd<float, 16> v1226_data(0.0f);
              v1226_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v1227_data(ir7.template select<16, 1>(176));
              ir7.template select<16, 1>(176) = (v1227_data + v1226_data);
              // r7 = ir7
              #pragma unroll
              for (int32_t v1229_n1 = 0; v1229_n1 < 12; ++v1229_n1) {
                int32_t v1230_a = v1229_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1232_data(ir7.template select<12, 1>(v1230_a));
                r7.template select<12, 1>(v1230_a) = v1232_data;
              }
              // glb_m5 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1233_i1 = 0; v1233_i1 < 12; ++v1233_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1236_data(r7.template select<12, 1>((v1233_i1 * 16)));
                v1236_data.copy_to(glb_m5 + ((v1233_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


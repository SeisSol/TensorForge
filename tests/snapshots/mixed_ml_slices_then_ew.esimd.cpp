// === base name ===
kernel_65fac80109253459

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_65fac80109253459 = {{1, 32, 1}, 8, 8, 1, 32, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_65fac80109253459(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_65fac80109253459(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_65fac80109253459(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 32, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 32 - 1) / 32;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 32;
  config.block[2] = 1;
  config.sharedMemBytes = 3328 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_65fac80109253459(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_65fac80109253459(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_65fac80109253459(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_65fac80109253459(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<3328 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 32 per block = block 1x32x1, 13312 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×4(8×4) {0..8}×{0..4} strided
        //   m2 8×4(8×4) {0..8}×{0..4} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j]@{0..8}×{0..4} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{0..8}×{4..8} = m0[i,k] × m2[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (104 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v13_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v13_batchId0 < numElements0; v13_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v14_ahead1 = v13_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v16_batchId1 = (v14_ahead1 < numElements0) ? v14_ahead1 : v13_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v13_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v13_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v13_batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v13_batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v13_batchId0 * 64 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
                int32_t v28_lead = v26_i0 * 8;
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
                  int32_t v31_a = v28_lead + (v27_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v32_data;
                  v32_data.copy_from(glb_m0 + (v31_a));
                  r0.template select<8, 1>(v31_a) = v32_data;
                }
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v34_ld;
              v34_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v34_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 0), v35_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 32> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v37_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> s0_w0 = tensorforge::slmLoad<float, 32>(s0 + 0);
              float v38_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v40_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v40_data + (v37_data * v38_data));
              float v43_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v45_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v45_data + (v37_data * v43_data));
              float v48_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v50_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v50_data + (v37_data * v48_data));
              float v53_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v55_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v55_data + (v37_data * v53_data));
              tensorforge::intel_esimd::simd<float, 8> v57_data(r0.template select<8, 1>(8));
              float v58_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v60_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v60_data + (v57_data * v58_data));
              float v63_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v65_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v65_data + (v57_data * v63_data));
              float v68_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v70_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v70_data + (v57_data * v68_data));
              float v73_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v75_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v75_data + (v57_data * v73_data));
              tensorforge::intel_esimd::simd<float, 8> v77_data(r0.template select<8, 1>(16));
              float v78_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v80_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v80_data + (v77_data * v78_data));
              float v83_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v85_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v85_data + (v77_data * v83_data));
              float v88_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v90_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v90_data + (v77_data * v88_data));
              float v93_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v95_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v95_data + (v77_data * v93_data));
              tensorforge::intel_esimd::simd<float, 8> v97_data(r0.template select<8, 1>(24));
              float v98_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v100_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v100_data + (v97_data * v98_data));
              float v103_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v105_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v105_data + (v97_data * v103_data));
              float v108_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v110_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v110_data + (v97_data * v108_data));
              float v113_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v115_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v115_data + (v97_data * v113_data));
              tensorforge::intel_esimd::simd<float, 8> v117_data(r0.template select<8, 1>(32));
              float v118_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v120_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v120_data + (v117_data * v118_data));
              float v123_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v125_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v125_data + (v117_data * v123_data));
              float v128_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v130_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v130_data + (v117_data * v128_data));
              float v133_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v135_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v135_data + (v117_data * v133_data));
              tensorforge::intel_esimd::simd<float, 8> v137_data(r0.template select<8, 1>(40));
              float v138_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v140_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v140_data + (v137_data * v138_data));
              float v143_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v145_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v145_data + (v137_data * v143_data));
              float v148_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v150_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v150_data + (v137_data * v148_data));
              float v153_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v155_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v155_data + (v137_data * v153_data));
              tensorforge::intel_esimd::simd<float, 8> v157_data(r0.template select<8, 1>(48));
              float v158_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v160_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v160_data + (v157_data * v158_data));
              float v163_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v165_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v165_data + (v157_data * v163_data));
              float v168_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v170_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v170_data + (v157_data * v168_data));
              float v173_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v175_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v175_data + (v157_data * v173_data));
              tensorforge::intel_esimd::simd<float, 8> v177_data(r0.template select<8, 1>(56));
              float v178_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v180_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v180_data + (v177_data * v178_data));
              float v183_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v185_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v185_data + (v177_data * v183_data));
              float v188_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v190_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v190_data + (v177_data * v188_data));
              float v193_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v195_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v195_data + (v177_data * v193_data));
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v197_i0 = 0; v197_i0 < 1; ++v197_i0) {
                int32_t v199_a = v197_i0 * 8;
                #pragma unroll
                for (int32_t v198_i1 = 0; v198_i1 < 4; ++v198_i1) {
                  int32_t v201_a = v199_a + (v198_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v202_data(r1.template select<8, 1>(v201_a));
                  tensorforge::slmStore<float, 8>(s1 + (v201_a), v202_data);
                }
              }
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 32> r2(0.0f);
              // ir2 = +(r0 * s2)
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 32> ir2(0.0f);
              tensorforge::intel_esimd::simd<float, 32> s2_w1 = tensorforge::slmLoad<float, 32>(s2 + 0);
              float v208_data = s2_w1[0];
              tensorforge::intel_esimd::simd<float, 8> v210_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v210_data + (v37_data * v208_data));
              float v213_data = s2_w1[8];
              tensorforge::intel_esimd::simd<float, 8> v215_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v215_data + (v37_data * v213_data));
              float v218_data = s2_w1[16];
              tensorforge::intel_esimd::simd<float, 8> v220_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v220_data + (v37_data * v218_data));
              float v223_data = s2_w1[24];
              tensorforge::intel_esimd::simd<float, 8> v225_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v225_data + (v37_data * v223_data));
              float v228_data = s2_w1[1];
              tensorforge::intel_esimd::simd<float, 8> v230_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v230_data + (v57_data * v228_data));
              float v233_data = s2_w1[9];
              tensorforge::intel_esimd::simd<float, 8> v235_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v235_data + (v57_data * v233_data));
              float v238_data = s2_w1[17];
              tensorforge::intel_esimd::simd<float, 8> v240_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v240_data + (v57_data * v238_data));
              float v243_data = s2_w1[25];
              tensorforge::intel_esimd::simd<float, 8> v245_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v245_data + (v57_data * v243_data));
              float v248_data = s2_w1[2];
              tensorforge::intel_esimd::simd<float, 8> v250_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v250_data + (v77_data * v248_data));
              float v253_data = s2_w1[10];
              tensorforge::intel_esimd::simd<float, 8> v255_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v255_data + (v77_data * v253_data));
              float v258_data = s2_w1[18];
              tensorforge::intel_esimd::simd<float, 8> v260_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v260_data + (v77_data * v258_data));
              float v263_data = s2_w1[26];
              tensorforge::intel_esimd::simd<float, 8> v265_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v265_data + (v77_data * v263_data));
              float v268_data = s2_w1[3];
              tensorforge::intel_esimd::simd<float, 8> v270_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v270_data + (v97_data * v268_data));
              float v273_data = s2_w1[11];
              tensorforge::intel_esimd::simd<float, 8> v275_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v275_data + (v97_data * v273_data));
              float v278_data = s2_w1[19];
              tensorforge::intel_esimd::simd<float, 8> v280_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v280_data + (v97_data * v278_data));
              float v283_data = s2_w1[27];
              tensorforge::intel_esimd::simd<float, 8> v285_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v285_data + (v97_data * v283_data));
              float v288_data = s2_w1[4];
              tensorforge::intel_esimd::simd<float, 8> v290_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v290_data + (v117_data * v288_data));
              float v293_data = s2_w1[12];
              tensorforge::intel_esimd::simd<float, 8> v295_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v295_data + (v117_data * v293_data));
              float v298_data = s2_w1[20];
              tensorforge::intel_esimd::simd<float, 8> v300_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v300_data + (v117_data * v298_data));
              float v303_data = s2_w1[28];
              tensorforge::intel_esimd::simd<float, 8> v305_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v305_data + (v117_data * v303_data));
              float v308_data = s2_w1[5];
              tensorforge::intel_esimd::simd<float, 8> v310_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v310_data + (v137_data * v308_data));
              float v313_data = s2_w1[13];
              tensorforge::intel_esimd::simd<float, 8> v315_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v315_data + (v137_data * v313_data));
              float v318_data = s2_w1[21];
              tensorforge::intel_esimd::simd<float, 8> v320_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v320_data + (v137_data * v318_data));
              float v323_data = s2_w1[29];
              tensorforge::intel_esimd::simd<float, 8> v325_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v325_data + (v137_data * v323_data));
              float v328_data = s2_w1[6];
              tensorforge::intel_esimd::simd<float, 8> v330_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v330_data + (v157_data * v328_data));
              float v333_data = s2_w1[14];
              tensorforge::intel_esimd::simd<float, 8> v335_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v335_data + (v157_data * v333_data));
              float v338_data = s2_w1[22];
              tensorforge::intel_esimd::simd<float, 8> v340_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v340_data + (v157_data * v338_data));
              float v343_data = s2_w1[30];
              tensorforge::intel_esimd::simd<float, 8> v345_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v345_data + (v157_data * v343_data));
              float v348_data = s2_w1[7];
              tensorforge::intel_esimd::simd<float, 8> v350_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v350_data + (v177_data * v348_data));
              float v353_data = s2_w1[15];
              tensorforge::intel_esimd::simd<float, 8> v355_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v355_data + (v177_data * v353_data));
              float v358_data = s2_w1[23];
              tensorforge::intel_esimd::simd<float, 8> v360_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v360_data + (v177_data * v358_data));
              float v363_data = s2_w1[31];
              tensorforge::intel_esimd::simd<float, 8> v365_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v365_data + (v177_data * v363_data));
              // r2 = ir2
              #pragma unroll
              for (int32_t v367_n0 = 0; v367_n0 < 1; ++v367_n0) {
                int32_t v369_a = v367_n0 * 8;
                #pragma unroll
                for (int32_t v368_n1 = 0; v368_n1 < 4; ++v368_n1) {
                  int32_t v371_a = v369_a + (v368_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v372_data(ir2.template select<8, 1>(v371_a));
                  r2.template select<8, 1>(v371_a) = v372_data;
                }
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v373_i0 = 0; v373_i0 < 1; ++v373_i0) {
                int32_t v375_a = v373_i0 * 8;
                #pragma unroll
                for (int32_t v374_i1 = 0; v374_i1 < 4; ++v374_i1) {
                  tensorforge::intel_esimd::simd<float, 8> v378_data(r2.template select<8, 1>((v375_a + (v374_i1 * 8))));
                  tensorforge::slmStore<float, 8>(s1 + ((v375_a + ((v374_i1 + 4) * 8))), v378_data);
                }
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v383_k0 = 0; v383_k0 < 1; ++v383_k0) {
                int32_t v385_lead = v383_k0 * 8;
                #pragma unroll
                for (int32_t v384_k1 = 0; v384_k1 < 8; ++v384_k1) {
                  int32_t v388_a = v385_lead + (v384_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v389_data = tensorforge::slmLoad<float, 8>(s1 + (v388_a));
                  (tensorforge::intel_esimd::abs(v389_data)).copy_to(glb_m3 + (v388_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


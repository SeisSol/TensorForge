// === base name ===
kernel_57c01cb42aa9d49c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_57c01cb42aa9d49c = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_57c01cb42aa9d49c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_57c01cb42aa9d49c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_57c01cb42aa9d49c(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_57c01cb42aa9d49c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_57c01cb42aa9d49c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_57c01cb42aa9d49c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_57c01cb42aa9d49c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 32 per block = block 1x32x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        //   m4 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t0[i,j] += m2[i,k] × m3[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (72 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v10_batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v10_batchId0 * 64 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v26_lead = v24_i0 * 8;
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                  int32_t v29_a = v26_lead + (v25_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v30_data;
                  v30_data.copy_from(glb_m0 + (v29_a));
                  r0.template select<8, 1>(v29_a) = v30_data;
                }
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v32_ld;
              v32_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v32_ld);
              tensorforge::intel_esimd::simd<float, 32> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 32), v33_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 64> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
                int32_t v37_lead = v35_i0 * 8;
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
                  int32_t v40_a = v37_lead + (v36_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v41_data;
                  v41_data.copy_from(glb_m2 + (v40_a));
                  r2.template select<8, 1>(v40_a) = v41_data;
                }
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v44_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s0_w0 = tensorforge::slmLoad<float, 64>(s0 + 0);
              float v45_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v47_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v47_data + (v44_data * v45_data));
              float v50_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v52_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v52_data + (v44_data * v50_data));
              float v55_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v57_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v57_data + (v44_data * v55_data));
              float v60_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v62_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v62_data + (v44_data * v60_data));
              float v65_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v67_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v67_data + (v44_data * v65_data));
              float v70_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v72_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v72_data + (v44_data * v70_data));
              float v75_data = s0_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v77_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v77_data + (v44_data * v75_data));
              float v80_data = s0_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v82_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v82_data + (v44_data * v80_data));
              tensorforge::intel_esimd::simd<float, 8> v84_data(r0.template select<8, 1>(8));
              float v85_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v87_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v87_data + (v84_data * v85_data));
              float v90_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v92_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v92_data + (v84_data * v90_data));
              float v95_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v97_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v97_data + (v84_data * v95_data));
              float v100_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v102_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v102_data + (v84_data * v100_data));
              float v105_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v107_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v107_data + (v84_data * v105_data));
              float v110_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v112_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v112_data + (v84_data * v110_data));
              float v115_data = s0_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v117_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v117_data + (v84_data * v115_data));
              float v120_data = s0_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v122_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v122_data + (v84_data * v120_data));
              tensorforge::intel_esimd::simd<float, 8> v124_data(r0.template select<8, 1>(16));
              float v125_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v127_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v127_data + (v124_data * v125_data));
              float v130_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v132_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v132_data + (v124_data * v130_data));
              float v135_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v137_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v137_data + (v124_data * v135_data));
              float v140_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v142_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v142_data + (v124_data * v140_data));
              float v145_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v147_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v147_data + (v124_data * v145_data));
              float v150_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v152_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v152_data + (v124_data * v150_data));
              float v155_data = s0_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v157_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v157_data + (v124_data * v155_data));
              float v160_data = s0_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v162_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v162_data + (v124_data * v160_data));
              tensorforge::intel_esimd::simd<float, 8> v164_data(r0.template select<8, 1>(24));
              float v165_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v167_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v167_data + (v164_data * v165_data));
              float v170_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v172_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v172_data + (v164_data * v170_data));
              float v175_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v177_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v177_data + (v164_data * v175_data));
              float v180_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v182_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v182_data + (v164_data * v180_data));
              float v185_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v187_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v187_data + (v164_data * v185_data));
              float v190_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v192_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v192_data + (v164_data * v190_data));
              float v195_data = s0_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v197_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v197_data + (v164_data * v195_data));
              float v200_data = s0_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v202_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v202_data + (v164_data * v200_data));
              tensorforge::intel_esimd::simd<float, 8> v204_data(r0.template select<8, 1>(32));
              float v205_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v207_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v207_data + (v204_data * v205_data));
              float v210_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v212_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v212_data + (v204_data * v210_data));
              float v215_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v217_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v217_data + (v204_data * v215_data));
              float v220_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v222_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v222_data + (v204_data * v220_data));
              float v225_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v227_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v227_data + (v204_data * v225_data));
              float v230_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v232_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v232_data + (v204_data * v230_data));
              float v235_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v237_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v237_data + (v204_data * v235_data));
              float v240_data = s0_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v242_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v242_data + (v204_data * v240_data));
              tensorforge::intel_esimd::simd<float, 8> v244_data(r0.template select<8, 1>(40));
              float v245_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v247_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v247_data + (v244_data * v245_data));
              float v250_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v252_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v252_data + (v244_data * v250_data));
              float v255_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v257_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v257_data + (v244_data * v255_data));
              float v260_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v262_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v262_data + (v244_data * v260_data));
              float v265_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v267_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v267_data + (v244_data * v265_data));
              float v270_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v272_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v272_data + (v244_data * v270_data));
              float v275_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v277_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v277_data + (v244_data * v275_data));
              float v280_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v282_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v282_data + (v244_data * v280_data));
              tensorforge::intel_esimd::simd<float, 8> v284_data(r0.template select<8, 1>(48));
              float v285_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v287_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v287_data + (v284_data * v285_data));
              float v290_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v292_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v292_data + (v284_data * v290_data));
              float v295_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v297_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v297_data + (v284_data * v295_data));
              float v300_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v302_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v302_data + (v284_data * v300_data));
              float v305_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v307_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v307_data + (v284_data * v305_data));
              float v310_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v312_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v312_data + (v284_data * v310_data));
              float v315_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v317_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v317_data + (v284_data * v315_data));
              float v320_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v322_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v322_data + (v284_data * v320_data));
              tensorforge::intel_esimd::simd<float, 8> v324_data(r0.template select<8, 1>(56));
              float v325_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v327_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v327_data + (v324_data * v325_data));
              float v330_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v332_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v332_data + (v324_data * v330_data));
              float v335_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v337_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v337_data + (v324_data * v335_data));
              float v340_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v342_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v342_data + (v324_data * v340_data));
              float v345_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v347_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v347_data + (v324_data * v345_data));
              float v350_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v352_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v352_data + (v324_data * v350_data));
              float v355_data = s0_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v357_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v357_data + (v324_data * v355_data));
              float v360_data = s0_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v362_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v362_data + (v324_data * v360_data));
              // s2 = load{g>s}(glb_m3[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v364_ld;
              v364_ld.copy_from(glb_m3 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 0), v364_ld);
              tensorforge::intel_esimd::simd<float, 32> v365_ld;
              v365_ld.copy_from(glb_m3 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 32), v365_ld);
              // wait(r2 = load{g>r}(glb_m2););
              // wait(s2 = load{g>s}(glb_m3[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r3(0.0f);
              // ir3 = +(r2 * s2)
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 64> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v368_data(r2.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s2_w1 = tensorforge::slmLoad<float, 64>(s2 + 0);
              float v369_data = s2_w1[0];
              tensorforge::intel_esimd::simd<float, 8> v371_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v371_data + (v368_data * v369_data));
              float v374_data = s2_w1[8];
              tensorforge::intel_esimd::simd<float, 8> v376_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v376_data + (v368_data * v374_data));
              float v379_data = s2_w1[16];
              tensorforge::intel_esimd::simd<float, 8> v381_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v381_data + (v368_data * v379_data));
              float v384_data = s2_w1[24];
              tensorforge::intel_esimd::simd<float, 8> v386_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v386_data + (v368_data * v384_data));
              float v389_data = s2_w1[32];
              tensorforge::intel_esimd::simd<float, 8> v391_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v391_data + (v368_data * v389_data));
              float v394_data = s2_w1[40];
              tensorforge::intel_esimd::simd<float, 8> v396_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v396_data + (v368_data * v394_data));
              float v399_data = s2_w1[48];
              tensorforge::intel_esimd::simd<float, 8> v401_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v401_data + (v368_data * v399_data));
              float v404_data = s2_w1[56];
              tensorforge::intel_esimd::simd<float, 8> v406_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v406_data + (v368_data * v404_data));
              tensorforge::intel_esimd::simd<float, 8> v408_data(r2.template select<8, 1>(8));
              float v409_data = s2_w1[1];
              tensorforge::intel_esimd::simd<float, 8> v411_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v411_data + (v408_data * v409_data));
              float v414_data = s2_w1[9];
              tensorforge::intel_esimd::simd<float, 8> v416_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v416_data + (v408_data * v414_data));
              float v419_data = s2_w1[17];
              tensorforge::intel_esimd::simd<float, 8> v421_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v421_data + (v408_data * v419_data));
              float v424_data = s2_w1[25];
              tensorforge::intel_esimd::simd<float, 8> v426_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v426_data + (v408_data * v424_data));
              float v429_data = s2_w1[33];
              tensorforge::intel_esimd::simd<float, 8> v431_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v431_data + (v408_data * v429_data));
              float v434_data = s2_w1[41];
              tensorforge::intel_esimd::simd<float, 8> v436_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v436_data + (v408_data * v434_data));
              float v439_data = s2_w1[49];
              tensorforge::intel_esimd::simd<float, 8> v441_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v441_data + (v408_data * v439_data));
              float v444_data = s2_w1[57];
              tensorforge::intel_esimd::simd<float, 8> v446_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v446_data + (v408_data * v444_data));
              tensorforge::intel_esimd::simd<float, 8> v448_data(r2.template select<8, 1>(16));
              float v449_data = s2_w1[2];
              tensorforge::intel_esimd::simd<float, 8> v451_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v451_data + (v448_data * v449_data));
              float v454_data = s2_w1[10];
              tensorforge::intel_esimd::simd<float, 8> v456_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v456_data + (v448_data * v454_data));
              float v459_data = s2_w1[18];
              tensorforge::intel_esimd::simd<float, 8> v461_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v461_data + (v448_data * v459_data));
              float v464_data = s2_w1[26];
              tensorforge::intel_esimd::simd<float, 8> v466_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v466_data + (v448_data * v464_data));
              float v469_data = s2_w1[34];
              tensorforge::intel_esimd::simd<float, 8> v471_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v471_data + (v448_data * v469_data));
              float v474_data = s2_w1[42];
              tensorforge::intel_esimd::simd<float, 8> v476_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v476_data + (v448_data * v474_data));
              float v479_data = s2_w1[50];
              tensorforge::intel_esimd::simd<float, 8> v481_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v481_data + (v448_data * v479_data));
              float v484_data = s2_w1[58];
              tensorforge::intel_esimd::simd<float, 8> v486_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v486_data + (v448_data * v484_data));
              tensorforge::intel_esimd::simd<float, 8> v488_data(r2.template select<8, 1>(24));
              float v489_data = s2_w1[3];
              tensorforge::intel_esimd::simd<float, 8> v491_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v491_data + (v488_data * v489_data));
              float v494_data = s2_w1[11];
              tensorforge::intel_esimd::simd<float, 8> v496_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v496_data + (v488_data * v494_data));
              float v499_data = s2_w1[19];
              tensorforge::intel_esimd::simd<float, 8> v501_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v501_data + (v488_data * v499_data));
              float v504_data = s2_w1[27];
              tensorforge::intel_esimd::simd<float, 8> v506_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v506_data + (v488_data * v504_data));
              float v509_data = s2_w1[35];
              tensorforge::intel_esimd::simd<float, 8> v511_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v511_data + (v488_data * v509_data));
              float v514_data = s2_w1[43];
              tensorforge::intel_esimd::simd<float, 8> v516_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v516_data + (v488_data * v514_data));
              float v519_data = s2_w1[51];
              tensorforge::intel_esimd::simd<float, 8> v521_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v521_data + (v488_data * v519_data));
              float v524_data = s2_w1[59];
              tensorforge::intel_esimd::simd<float, 8> v526_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v526_data + (v488_data * v524_data));
              tensorforge::intel_esimd::simd<float, 8> v528_data(r2.template select<8, 1>(32));
              float v529_data = s2_w1[4];
              tensorforge::intel_esimd::simd<float, 8> v531_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v531_data + (v528_data * v529_data));
              float v534_data = s2_w1[12];
              tensorforge::intel_esimd::simd<float, 8> v536_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v536_data + (v528_data * v534_data));
              float v539_data = s2_w1[20];
              tensorforge::intel_esimd::simd<float, 8> v541_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v541_data + (v528_data * v539_data));
              float v544_data = s2_w1[28];
              tensorforge::intel_esimd::simd<float, 8> v546_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v546_data + (v528_data * v544_data));
              float v549_data = s2_w1[36];
              tensorforge::intel_esimd::simd<float, 8> v551_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v551_data + (v528_data * v549_data));
              float v554_data = s2_w1[44];
              tensorforge::intel_esimd::simd<float, 8> v556_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v556_data + (v528_data * v554_data));
              float v559_data = s2_w1[52];
              tensorforge::intel_esimd::simd<float, 8> v561_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v561_data + (v528_data * v559_data));
              float v564_data = s2_w1[60];
              tensorforge::intel_esimd::simd<float, 8> v566_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v566_data + (v528_data * v564_data));
              tensorforge::intel_esimd::simd<float, 8> v568_data(r2.template select<8, 1>(40));
              float v569_data = s2_w1[5];
              tensorforge::intel_esimd::simd<float, 8> v571_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v571_data + (v568_data * v569_data));
              float v574_data = s2_w1[13];
              tensorforge::intel_esimd::simd<float, 8> v576_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v576_data + (v568_data * v574_data));
              float v579_data = s2_w1[21];
              tensorforge::intel_esimd::simd<float, 8> v581_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v581_data + (v568_data * v579_data));
              float v584_data = s2_w1[29];
              tensorforge::intel_esimd::simd<float, 8> v586_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v586_data + (v568_data * v584_data));
              float v589_data = s2_w1[37];
              tensorforge::intel_esimd::simd<float, 8> v591_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v591_data + (v568_data * v589_data));
              float v594_data = s2_w1[45];
              tensorforge::intel_esimd::simd<float, 8> v596_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v596_data + (v568_data * v594_data));
              float v599_data = s2_w1[53];
              tensorforge::intel_esimd::simd<float, 8> v601_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v601_data + (v568_data * v599_data));
              float v604_data = s2_w1[61];
              tensorforge::intel_esimd::simd<float, 8> v606_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v606_data + (v568_data * v604_data));
              tensorforge::intel_esimd::simd<float, 8> v608_data(r2.template select<8, 1>(48));
              float v609_data = s2_w1[6];
              tensorforge::intel_esimd::simd<float, 8> v611_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v611_data + (v608_data * v609_data));
              float v614_data = s2_w1[14];
              tensorforge::intel_esimd::simd<float, 8> v616_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v616_data + (v608_data * v614_data));
              float v619_data = s2_w1[22];
              tensorforge::intel_esimd::simd<float, 8> v621_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v621_data + (v608_data * v619_data));
              float v624_data = s2_w1[30];
              tensorforge::intel_esimd::simd<float, 8> v626_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v626_data + (v608_data * v624_data));
              float v629_data = s2_w1[38];
              tensorforge::intel_esimd::simd<float, 8> v631_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v631_data + (v608_data * v629_data));
              float v634_data = s2_w1[46];
              tensorforge::intel_esimd::simd<float, 8> v636_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v636_data + (v608_data * v634_data));
              float v639_data = s2_w1[54];
              tensorforge::intel_esimd::simd<float, 8> v641_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v641_data + (v608_data * v639_data));
              float v644_data = s2_w1[62];
              tensorforge::intel_esimd::simd<float, 8> v646_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v646_data + (v608_data * v644_data));
              tensorforge::intel_esimd::simd<float, 8> v648_data(r2.template select<8, 1>(56));
              float v649_data = s2_w1[7];
              tensorforge::intel_esimd::simd<float, 8> v651_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v651_data + (v648_data * v649_data));
              float v654_data = s2_w1[15];
              tensorforge::intel_esimd::simd<float, 8> v656_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v656_data + (v648_data * v654_data));
              float v659_data = s2_w1[23];
              tensorforge::intel_esimd::simd<float, 8> v661_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v661_data + (v648_data * v659_data));
              float v664_data = s2_w1[31];
              tensorforge::intel_esimd::simd<float, 8> v666_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v666_data + (v648_data * v664_data));
              float v669_data = s2_w1[39];
              tensorforge::intel_esimd::simd<float, 8> v671_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v671_data + (v648_data * v669_data));
              float v674_data = s2_w1[47];
              tensorforge::intel_esimd::simd<float, 8> v676_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v676_data + (v648_data * v674_data));
              float v679_data = s2_w1[55];
              tensorforge::intel_esimd::simd<float, 8> v681_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v681_data + (v648_data * v679_data));
              float v684_data = s2_w1[63];
              tensorforge::intel_esimd::simd<float, 8> v686_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v686_data + (v648_data * v684_data));
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v688_n0 = 0; v688_n0 < 1; ++v688_n0) {
                int32_t v690_a = v688_n0 * 8;
                #pragma unroll
                for (int32_t v689_n1 = 0; v689_n1 < 8; ++v689_n1) {
                  int32_t v692_a = v690_a + (v689_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v693_data(ir3.template select<8, 1>(v692_a));
                  tensorforge::intel_esimd::simd<float, 8> v694_data(r1.template select<8, 1>(v692_a));
                  r3.template select<8, 1>(v692_a) = (v694_data + v693_data);
                }
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v696_i0 = 0; v696_i0 < 1; ++v696_i0) {
                int32_t v698_a = v696_i0 * 8;
                #pragma unroll
                for (int32_t v697_i1 = 0; v697_i1 < 8; ++v697_i1) {
                  int32_t v700_a = v698_a + (v697_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v701_data(r3.template select<8, 1>(v700_a));
                  tensorforge::slmStore<float, 8>(s1 + (v700_a), v701_data);
                }
              }
              // glb_m4 = abs(s1)
              #pragma unroll
              for (int32_t v704_k0 = 0; v704_k0 < 1; ++v704_k0) {
                int32_t v706_lead = v704_k0 * 8;
                #pragma unroll
                for (int32_t v705_k1 = 0; v705_k1 < 8; ++v705_k1) {
                  int32_t v709_a = v706_lead + (v705_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v710_data = tensorforge::slmLoad<float, 8>(s1 + (v709_a));
                  (tensorforge::intel_esimd::abs(v710_data)).copy_to(glb_m4 + (v709_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


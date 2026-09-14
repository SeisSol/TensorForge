// === base name ===
kernel_99b6824bd29d2096

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_99b6824bd29d2096 = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_99b6824bd29d2096(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_99b6824bd29d2096(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_99b6824bd29d2096(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_99b6824bd29d2096(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_99b6824bd29d2096(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_99b6824bd29d2096(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_99b6824bd29d2096(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 32 per block = block 1x32x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (72 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 64 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v20_lead = v18_i0 * 8;
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 8; ++v19_i1) {
                  int32_t v23_a = v20_lead + (v19_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v24_data;
                  v24_data.copy_from(glb_m0 + (v23_a));
                  r0.template select<8, 1>(v23_a) = v24_data;
                }
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v26_ld;
              v26_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v26_ld);
              tensorforge::intel_esimd::simd<float, 32> v27_ld;
              v27_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 32), v27_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v29_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s0_w0 = tensorforge::slmLoad<float, 64>(s0 + 0);
              float v30_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v32_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v32_data + (v29_data * v30_data));
              float v35_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v37_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v37_data + (v29_data * v35_data));
              float v40_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v42_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v42_data + (v29_data * v40_data));
              float v45_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v47_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v47_data + (v29_data * v45_data));
              float v50_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v52_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v52_data + (v29_data * v50_data));
              float v55_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v57_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v57_data + (v29_data * v55_data));
              float v60_data = s0_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v62_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v62_data + (v29_data * v60_data));
              float v65_data = s0_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v67_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v67_data + (v29_data * v65_data));
              tensorforge::intel_esimd::simd<float, 8> v69_data(r0.template select<8, 1>(8));
              float v70_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v72_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v72_data + (v69_data * v70_data));
              float v75_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v77_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v77_data + (v69_data * v75_data));
              float v80_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v82_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v82_data + (v69_data * v80_data));
              float v85_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v87_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v87_data + (v69_data * v85_data));
              float v90_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v92_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v92_data + (v69_data * v90_data));
              float v95_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v97_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v97_data + (v69_data * v95_data));
              float v100_data = s0_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v102_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v102_data + (v69_data * v100_data));
              float v105_data = s0_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v107_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v107_data + (v69_data * v105_data));
              tensorforge::intel_esimd::simd<float, 8> v109_data(r0.template select<8, 1>(16));
              float v110_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v112_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v112_data + (v109_data * v110_data));
              float v115_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v117_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v117_data + (v109_data * v115_data));
              float v120_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v122_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v122_data + (v109_data * v120_data));
              float v125_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v127_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v127_data + (v109_data * v125_data));
              float v130_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v132_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v132_data + (v109_data * v130_data));
              float v135_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v137_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v137_data + (v109_data * v135_data));
              float v140_data = s0_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v142_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v142_data + (v109_data * v140_data));
              float v145_data = s0_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v147_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v147_data + (v109_data * v145_data));
              tensorforge::intel_esimd::simd<float, 8> v149_data(r0.template select<8, 1>(24));
              float v150_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v152_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v152_data + (v149_data * v150_data));
              float v155_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v157_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v157_data + (v149_data * v155_data));
              float v160_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v162_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v162_data + (v149_data * v160_data));
              float v165_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v167_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v167_data + (v149_data * v165_data));
              float v170_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v172_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v172_data + (v149_data * v170_data));
              float v175_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v177_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v177_data + (v149_data * v175_data));
              float v180_data = s0_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v182_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v182_data + (v149_data * v180_data));
              float v185_data = s0_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v187_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v187_data + (v149_data * v185_data));
              tensorforge::intel_esimd::simd<float, 8> v189_data(r0.template select<8, 1>(32));
              float v190_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v192_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v192_data + (v189_data * v190_data));
              float v195_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v197_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v197_data + (v189_data * v195_data));
              float v200_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v202_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v202_data + (v189_data * v200_data));
              float v205_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v207_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v207_data + (v189_data * v205_data));
              float v210_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v212_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v212_data + (v189_data * v210_data));
              float v215_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v217_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v217_data + (v189_data * v215_data));
              float v220_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v222_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v222_data + (v189_data * v220_data));
              float v225_data = s0_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v227_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v227_data + (v189_data * v225_data));
              tensorforge::intel_esimd::simd<float, 8> v229_data(r0.template select<8, 1>(40));
              float v230_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v232_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v232_data + (v229_data * v230_data));
              float v235_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v237_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v237_data + (v229_data * v235_data));
              float v240_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v242_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v242_data + (v229_data * v240_data));
              float v245_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v247_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v247_data + (v229_data * v245_data));
              float v250_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v252_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v252_data + (v229_data * v250_data));
              float v255_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v257_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v257_data + (v229_data * v255_data));
              float v260_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v262_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v262_data + (v229_data * v260_data));
              float v265_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v267_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v267_data + (v229_data * v265_data));
              tensorforge::intel_esimd::simd<float, 8> v269_data(r0.template select<8, 1>(48));
              float v270_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v272_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v272_data + (v269_data * v270_data));
              float v275_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v277_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v277_data + (v269_data * v275_data));
              float v280_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v282_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v282_data + (v269_data * v280_data));
              float v285_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v287_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v287_data + (v269_data * v285_data));
              float v290_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v292_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v292_data + (v269_data * v290_data));
              float v295_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v297_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v297_data + (v269_data * v295_data));
              float v300_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v302_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v302_data + (v269_data * v300_data));
              float v305_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v307_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v307_data + (v269_data * v305_data));
              tensorforge::intel_esimd::simd<float, 8> v309_data(r0.template select<8, 1>(56));
              float v310_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v312_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v312_data + (v309_data * v310_data));
              float v315_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v317_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v317_data + (v309_data * v315_data));
              float v320_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v322_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v322_data + (v309_data * v320_data));
              float v325_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v327_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v327_data + (v309_data * v325_data));
              float v330_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v332_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v332_data + (v309_data * v330_data));
              float v335_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v337_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v337_data + (v309_data * v335_data));
              float v340_data = s0_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v342_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v342_data + (v309_data * v340_data));
              float v345_data = s0_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v347_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v347_data + (v309_data * v345_data));
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v349_i0 = 0; v349_i0 < 1; ++v349_i0) {
                int32_t v351_a = v349_i0 * 8;
                #pragma unroll
                for (int32_t v350_i1 = 0; v350_i1 < 8; ++v350_i1) {
                  int32_t v353_a = v351_a + (v350_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v354_data(r1.template select<8, 1>(v353_a));
                  tensorforge::slmStore<float, 8>(s1 + (v353_a), v354_data);
                }
              }
              // glb_m2 = abs(s1)
              #pragma unroll
              for (int32_t v357_k0 = 0; v357_k0 < 1; ++v357_k0) {
                int32_t v359_lead = v357_k0 * 8;
                #pragma unroll
                for (int32_t v358_k1 = 0; v358_k1 < 8; ++v358_k1) {
                  int32_t v362_a = v359_lead + (v358_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v363_data = tensorforge::slmLoad<float, 8>(s1 + (v362_a));
                  (tensorforge::intel_esimd::abs(v363_data)).copy_to(glb_m2 + (v362_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


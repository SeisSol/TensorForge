// === base name ===
kernel_4ee635026b2ab260

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4ee635026b2ab260 = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4ee635026b2ab260(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4ee635026b2ab260(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4ee635026b2ab260(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4ee635026b2ab260(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4ee635026b2ab260(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4ee635026b2ab260(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4ee635026b2ab260(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v13_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v13_batchId0 < numElements0; v13_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v14_ahead1 = v13_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v16_batchId1 = (v14_ahead1 < numElements0) ? v14_ahead1 : v13_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v13_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v13_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v13_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v13_batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v13_batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v13_batchId0 * 64 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
                int32_t v29_lead = v27_i0 * 8;
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
                  int32_t v32_a = v29_lead + (v28_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v33_data;
                  v33_data.copy_from(glb_m0 + (v32_a));
                  r0.template select<8, 1>(v32_a) = v33_data;
                }
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v35_ld;
              v35_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v35_ld);
              tensorforge::intel_esimd::simd<float, 32> v36_ld;
              v36_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 32), v36_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 64> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v38_i0 = 0; v38_i0 < 1; ++v38_i0) {
                int32_t v40_lead = v38_i0 * 8;
                #pragma unroll
                for (int32_t v39_i1 = 0; v39_i1 < 8; ++v39_i1) {
                  int32_t v43_a = v40_lead + (v39_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v44_data;
                  v44_data.copy_from(glb_m2 + (v43_a));
                  r2.template select<8, 1>(v43_a) = v44_data;
                }
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v47_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s0_w0 = tensorforge::slmLoad<float, 64>(s0 + 0);
              float v48_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v50_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v50_data + (v47_data * v48_data));
              float v53_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v55_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v55_data + (v47_data * v53_data));
              float v58_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v60_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v60_data + (v47_data * v58_data));
              float v63_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v65_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v65_data + (v47_data * v63_data));
              float v68_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v70_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v70_data + (v47_data * v68_data));
              float v73_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v75_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v75_data + (v47_data * v73_data));
              float v78_data = s0_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v80_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v80_data + (v47_data * v78_data));
              float v83_data = s0_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v85_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v85_data + (v47_data * v83_data));
              tensorforge::intel_esimd::simd<float, 8> v87_data(r0.template select<8, 1>(8));
              float v88_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v90_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v90_data + (v87_data * v88_data));
              float v93_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v95_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v95_data + (v87_data * v93_data));
              float v98_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v100_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v100_data + (v87_data * v98_data));
              float v103_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v105_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v105_data + (v87_data * v103_data));
              float v108_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v110_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v110_data + (v87_data * v108_data));
              float v113_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v115_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v115_data + (v87_data * v113_data));
              float v118_data = s0_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v120_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v120_data + (v87_data * v118_data));
              float v123_data = s0_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v125_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v125_data + (v87_data * v123_data));
              tensorforge::intel_esimd::simd<float, 8> v127_data(r0.template select<8, 1>(16));
              float v128_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v130_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v130_data + (v127_data * v128_data));
              float v133_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v135_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v135_data + (v127_data * v133_data));
              float v138_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v140_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v140_data + (v127_data * v138_data));
              float v143_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v145_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v145_data + (v127_data * v143_data));
              float v148_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v150_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v150_data + (v127_data * v148_data));
              float v153_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v155_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v155_data + (v127_data * v153_data));
              float v158_data = s0_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v160_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v160_data + (v127_data * v158_data));
              float v163_data = s0_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v165_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v165_data + (v127_data * v163_data));
              tensorforge::intel_esimd::simd<float, 8> v167_data(r0.template select<8, 1>(24));
              float v168_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v170_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v170_data + (v167_data * v168_data));
              float v173_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v175_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v175_data + (v167_data * v173_data));
              float v178_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v180_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v180_data + (v167_data * v178_data));
              float v183_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v185_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v185_data + (v167_data * v183_data));
              float v188_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v190_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v190_data + (v167_data * v188_data));
              float v193_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v195_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v195_data + (v167_data * v193_data));
              float v198_data = s0_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v200_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v200_data + (v167_data * v198_data));
              float v203_data = s0_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v205_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v205_data + (v167_data * v203_data));
              tensorforge::intel_esimd::simd<float, 8> v207_data(r0.template select<8, 1>(32));
              float v208_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v210_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v210_data + (v207_data * v208_data));
              float v213_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v215_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v215_data + (v207_data * v213_data));
              float v218_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v220_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v220_data + (v207_data * v218_data));
              float v223_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v225_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v225_data + (v207_data * v223_data));
              float v228_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v230_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v230_data + (v207_data * v228_data));
              float v233_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v235_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v235_data + (v207_data * v233_data));
              float v238_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v240_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v240_data + (v207_data * v238_data));
              float v243_data = s0_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v245_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v245_data + (v207_data * v243_data));
              tensorforge::intel_esimd::simd<float, 8> v247_data(r0.template select<8, 1>(40));
              float v248_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v250_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v250_data + (v247_data * v248_data));
              float v253_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v255_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v255_data + (v247_data * v253_data));
              float v258_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v260_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v260_data + (v247_data * v258_data));
              float v263_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v265_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v265_data + (v247_data * v263_data));
              float v268_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v270_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v270_data + (v247_data * v268_data));
              float v273_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v275_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v275_data + (v247_data * v273_data));
              float v278_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v280_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v280_data + (v247_data * v278_data));
              float v283_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v285_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v285_data + (v247_data * v283_data));
              tensorforge::intel_esimd::simd<float, 8> v287_data(r0.template select<8, 1>(48));
              float v288_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v290_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v290_data + (v287_data * v288_data));
              float v293_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v295_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v295_data + (v287_data * v293_data));
              float v298_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v300_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v300_data + (v287_data * v298_data));
              float v303_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v305_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v305_data + (v287_data * v303_data));
              float v308_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v310_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v310_data + (v287_data * v308_data));
              float v313_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v315_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v315_data + (v287_data * v313_data));
              float v318_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v320_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v320_data + (v287_data * v318_data));
              float v323_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v325_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v325_data + (v287_data * v323_data));
              tensorforge::intel_esimd::simd<float, 8> v327_data(r0.template select<8, 1>(56));
              float v328_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v330_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v330_data + (v327_data * v328_data));
              float v333_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v335_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v335_data + (v327_data * v333_data));
              float v338_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v340_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v340_data + (v327_data * v338_data));
              float v343_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v345_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v345_data + (v327_data * v343_data));
              float v348_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v350_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v350_data + (v327_data * v348_data));
              float v353_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v355_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v355_data + (v327_data * v353_data));
              float v358_data = s0_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v360_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v360_data + (v327_data * v358_data));
              float v363_data = s0_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v365_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v365_data + (v327_data * v363_data));
              // s2 = load{g>s}(glb_m3[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v367_ld;
              v367_ld.copy_from(glb_m3 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 0), v367_ld);
              tensorforge::intel_esimd::simd<float, 32> v368_ld;
              v368_ld.copy_from(glb_m3 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 32), v368_ld);
              // wait(r2 = load{g>r}(glb_m2););
              // wait(s2 = load{g>s}(glb_m3[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r3(0.0f);
              // ir3 = +(r2 * s2)
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 64> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v371_data(r2.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s2_w1 = tensorforge::slmLoad<float, 64>(s2 + 0);
              float v372_data = s2_w1[0];
              tensorforge::intel_esimd::simd<float, 8> v374_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v374_data + (v371_data * v372_data));
              float v377_data = s2_w1[8];
              tensorforge::intel_esimd::simd<float, 8> v379_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v379_data + (v371_data * v377_data));
              float v382_data = s2_w1[16];
              tensorforge::intel_esimd::simd<float, 8> v384_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v384_data + (v371_data * v382_data));
              float v387_data = s2_w1[24];
              tensorforge::intel_esimd::simd<float, 8> v389_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v389_data + (v371_data * v387_data));
              float v392_data = s2_w1[32];
              tensorforge::intel_esimd::simd<float, 8> v394_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v394_data + (v371_data * v392_data));
              float v397_data = s2_w1[40];
              tensorforge::intel_esimd::simd<float, 8> v399_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v399_data + (v371_data * v397_data));
              float v402_data = s2_w1[48];
              tensorforge::intel_esimd::simd<float, 8> v404_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v404_data + (v371_data * v402_data));
              float v407_data = s2_w1[56];
              tensorforge::intel_esimd::simd<float, 8> v409_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v409_data + (v371_data * v407_data));
              tensorforge::intel_esimd::simd<float, 8> v411_data(r2.template select<8, 1>(8));
              float v412_data = s2_w1[1];
              tensorforge::intel_esimd::simd<float, 8> v414_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v414_data + (v411_data * v412_data));
              float v417_data = s2_w1[9];
              tensorforge::intel_esimd::simd<float, 8> v419_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v419_data + (v411_data * v417_data));
              float v422_data = s2_w1[17];
              tensorforge::intel_esimd::simd<float, 8> v424_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v424_data + (v411_data * v422_data));
              float v427_data = s2_w1[25];
              tensorforge::intel_esimd::simd<float, 8> v429_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v429_data + (v411_data * v427_data));
              float v432_data = s2_w1[33];
              tensorforge::intel_esimd::simd<float, 8> v434_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v434_data + (v411_data * v432_data));
              float v437_data = s2_w1[41];
              tensorforge::intel_esimd::simd<float, 8> v439_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v439_data + (v411_data * v437_data));
              float v442_data = s2_w1[49];
              tensorforge::intel_esimd::simd<float, 8> v444_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v444_data + (v411_data * v442_data));
              float v447_data = s2_w1[57];
              tensorforge::intel_esimd::simd<float, 8> v449_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v449_data + (v411_data * v447_data));
              tensorforge::intel_esimd::simd<float, 8> v451_data(r2.template select<8, 1>(16));
              float v452_data = s2_w1[2];
              tensorforge::intel_esimd::simd<float, 8> v454_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v454_data + (v451_data * v452_data));
              float v457_data = s2_w1[10];
              tensorforge::intel_esimd::simd<float, 8> v459_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v459_data + (v451_data * v457_data));
              float v462_data = s2_w1[18];
              tensorforge::intel_esimd::simd<float, 8> v464_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v464_data + (v451_data * v462_data));
              float v467_data = s2_w1[26];
              tensorforge::intel_esimd::simd<float, 8> v469_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v469_data + (v451_data * v467_data));
              float v472_data = s2_w1[34];
              tensorforge::intel_esimd::simd<float, 8> v474_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v474_data + (v451_data * v472_data));
              float v477_data = s2_w1[42];
              tensorforge::intel_esimd::simd<float, 8> v479_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v479_data + (v451_data * v477_data));
              float v482_data = s2_w1[50];
              tensorforge::intel_esimd::simd<float, 8> v484_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v484_data + (v451_data * v482_data));
              float v487_data = s2_w1[58];
              tensorforge::intel_esimd::simd<float, 8> v489_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v489_data + (v451_data * v487_data));
              tensorforge::intel_esimd::simd<float, 8> v491_data(r2.template select<8, 1>(24));
              float v492_data = s2_w1[3];
              tensorforge::intel_esimd::simd<float, 8> v494_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v494_data + (v491_data * v492_data));
              float v497_data = s2_w1[11];
              tensorforge::intel_esimd::simd<float, 8> v499_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v499_data + (v491_data * v497_data));
              float v502_data = s2_w1[19];
              tensorforge::intel_esimd::simd<float, 8> v504_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v504_data + (v491_data * v502_data));
              float v507_data = s2_w1[27];
              tensorforge::intel_esimd::simd<float, 8> v509_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v509_data + (v491_data * v507_data));
              float v512_data = s2_w1[35];
              tensorforge::intel_esimd::simd<float, 8> v514_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v514_data + (v491_data * v512_data));
              float v517_data = s2_w1[43];
              tensorforge::intel_esimd::simd<float, 8> v519_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v519_data + (v491_data * v517_data));
              float v522_data = s2_w1[51];
              tensorforge::intel_esimd::simd<float, 8> v524_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v524_data + (v491_data * v522_data));
              float v527_data = s2_w1[59];
              tensorforge::intel_esimd::simd<float, 8> v529_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v529_data + (v491_data * v527_data));
              tensorforge::intel_esimd::simd<float, 8> v531_data(r2.template select<8, 1>(32));
              float v532_data = s2_w1[4];
              tensorforge::intel_esimd::simd<float, 8> v534_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v534_data + (v531_data * v532_data));
              float v537_data = s2_w1[12];
              tensorforge::intel_esimd::simd<float, 8> v539_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v539_data + (v531_data * v537_data));
              float v542_data = s2_w1[20];
              tensorforge::intel_esimd::simd<float, 8> v544_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v544_data + (v531_data * v542_data));
              float v547_data = s2_w1[28];
              tensorforge::intel_esimd::simd<float, 8> v549_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v549_data + (v531_data * v547_data));
              float v552_data = s2_w1[36];
              tensorforge::intel_esimd::simd<float, 8> v554_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v554_data + (v531_data * v552_data));
              float v557_data = s2_w1[44];
              tensorforge::intel_esimd::simd<float, 8> v559_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v559_data + (v531_data * v557_data));
              float v562_data = s2_w1[52];
              tensorforge::intel_esimd::simd<float, 8> v564_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v564_data + (v531_data * v562_data));
              float v567_data = s2_w1[60];
              tensorforge::intel_esimd::simd<float, 8> v569_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v569_data + (v531_data * v567_data));
              tensorforge::intel_esimd::simd<float, 8> v571_data(r2.template select<8, 1>(40));
              float v572_data = s2_w1[5];
              tensorforge::intel_esimd::simd<float, 8> v574_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v574_data + (v571_data * v572_data));
              float v577_data = s2_w1[13];
              tensorforge::intel_esimd::simd<float, 8> v579_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v579_data + (v571_data * v577_data));
              float v582_data = s2_w1[21];
              tensorforge::intel_esimd::simd<float, 8> v584_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v584_data + (v571_data * v582_data));
              float v587_data = s2_w1[29];
              tensorforge::intel_esimd::simd<float, 8> v589_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v589_data + (v571_data * v587_data));
              float v592_data = s2_w1[37];
              tensorforge::intel_esimd::simd<float, 8> v594_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v594_data + (v571_data * v592_data));
              float v597_data = s2_w1[45];
              tensorforge::intel_esimd::simd<float, 8> v599_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v599_data + (v571_data * v597_data));
              float v602_data = s2_w1[53];
              tensorforge::intel_esimd::simd<float, 8> v604_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v604_data + (v571_data * v602_data));
              float v607_data = s2_w1[61];
              tensorforge::intel_esimd::simd<float, 8> v609_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v609_data + (v571_data * v607_data));
              tensorforge::intel_esimd::simd<float, 8> v611_data(r2.template select<8, 1>(48));
              float v612_data = s2_w1[6];
              tensorforge::intel_esimd::simd<float, 8> v614_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v614_data + (v611_data * v612_data));
              float v617_data = s2_w1[14];
              tensorforge::intel_esimd::simd<float, 8> v619_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v619_data + (v611_data * v617_data));
              float v622_data = s2_w1[22];
              tensorforge::intel_esimd::simd<float, 8> v624_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v624_data + (v611_data * v622_data));
              float v627_data = s2_w1[30];
              tensorforge::intel_esimd::simd<float, 8> v629_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v629_data + (v611_data * v627_data));
              float v632_data = s2_w1[38];
              tensorforge::intel_esimd::simd<float, 8> v634_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v634_data + (v611_data * v632_data));
              float v637_data = s2_w1[46];
              tensorforge::intel_esimd::simd<float, 8> v639_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v639_data + (v611_data * v637_data));
              float v642_data = s2_w1[54];
              tensorforge::intel_esimd::simd<float, 8> v644_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v644_data + (v611_data * v642_data));
              float v647_data = s2_w1[62];
              tensorforge::intel_esimd::simd<float, 8> v649_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v649_data + (v611_data * v647_data));
              tensorforge::intel_esimd::simd<float, 8> v651_data(r2.template select<8, 1>(56));
              float v652_data = s2_w1[7];
              tensorforge::intel_esimd::simd<float, 8> v654_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v654_data + (v651_data * v652_data));
              float v657_data = s2_w1[15];
              tensorforge::intel_esimd::simd<float, 8> v659_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v659_data + (v651_data * v657_data));
              float v662_data = s2_w1[23];
              tensorforge::intel_esimd::simd<float, 8> v664_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v664_data + (v651_data * v662_data));
              float v667_data = s2_w1[31];
              tensorforge::intel_esimd::simd<float, 8> v669_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v669_data + (v651_data * v667_data));
              float v672_data = s2_w1[39];
              tensorforge::intel_esimd::simd<float, 8> v674_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v674_data + (v651_data * v672_data));
              float v677_data = s2_w1[47];
              tensorforge::intel_esimd::simd<float, 8> v679_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v679_data + (v651_data * v677_data));
              float v682_data = s2_w1[55];
              tensorforge::intel_esimd::simd<float, 8> v684_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v684_data + (v651_data * v682_data));
              float v687_data = s2_w1[63];
              tensorforge::intel_esimd::simd<float, 8> v689_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v689_data + (v651_data * v687_data));
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v691_n0 = 0; v691_n0 < 1; ++v691_n0) {
                int32_t v693_a = v691_n0 * 8;
                #pragma unroll
                for (int32_t v692_n1 = 0; v692_n1 < 8; ++v692_n1) {
                  int32_t v695_a = v693_a + (v692_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v696_data(ir3.template select<8, 1>(v695_a));
                  tensorforge::intel_esimd::simd<float, 8> v697_data(r1.template select<8, 1>(v695_a));
                  r3.template select<8, 1>(v695_a) = (v697_data + v696_data);
                }
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v699_i0 = 0; v699_i0 < 1; ++v699_i0) {
                int32_t v701_a = v699_i0 * 8;
                #pragma unroll
                for (int32_t v700_i1 = 0; v700_i1 < 8; ++v700_i1) {
                  int32_t v703_a = v701_a + (v700_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v704_data(r3.template select<8, 1>(v703_a));
                  tensorforge::slmStore<float, 8>(s1 + (v703_a), v704_data);
                }
              }
              // glb_m4 = abs(s1)
              #pragma unroll
              for (int32_t v707_k0 = 0; v707_k0 < 1; ++v707_k0) {
                int32_t v709_lead = v707_k0 * 8;
                #pragma unroll
                for (int32_t v708_k1 = 0; v708_k1 < 8; ++v708_k1) {
                  int32_t v712_a = v709_lead + (v708_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v713_data = tensorforge::slmLoad<float, 8>(s1 + (v712_a));
                  (tensorforge::intel_esimd::abs(v713_data)).copy_to(glb_m4 + (v712_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


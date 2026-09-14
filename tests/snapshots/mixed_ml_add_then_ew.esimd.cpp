// === base name ===
kernel_a53e36aef4650697

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a53e36aef4650697 = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a53e36aef4650697(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a53e36aef4650697(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a53e36aef4650697(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_a53e36aef4650697(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a53e36aef4650697(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a53e36aef4650697(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a53e36aef4650697(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        //   m4 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t0[i,j] += m2[i,k] × m3[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (72 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 64 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
                int32_t v23_lead = v21_i0 * 8;
                #pragma unroll
                for (int32_t v22_i1 = 0; v22_i1 < 8; ++v22_i1) {
                  int32_t v26_a = v23_lead + (v22_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v27_data;
                  v27_data.copy_from(glb_m0 + (v26_a));
                  r0.template select<8, 1>(v26_a) = v27_data;
                }
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v29_ld;
              v29_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v29_ld);
              tensorforge::intel_esimd::simd<float, 32> v30_ld;
              v30_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 32), v30_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 64> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v34_lead = v32_i0 * 8;
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
                  int32_t v37_a = v34_lead + (v33_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v38_data;
                  v38_data.copy_from(glb_m2 + (v37_a));
                  r2.template select<8, 1>(v37_a) = v38_data;
                }
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v41_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s0_w0 = tensorforge::slmLoad<float, 64>(s0 + 0);
              float v42_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v44_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v44_data + (v41_data * v42_data));
              float v47_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v49_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v49_data + (v41_data * v47_data));
              float v52_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v54_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v54_data + (v41_data * v52_data));
              float v57_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v59_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v59_data + (v41_data * v57_data));
              float v62_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v64_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v64_data + (v41_data * v62_data));
              float v67_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v69_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v69_data + (v41_data * v67_data));
              float v72_data = s0_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v74_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v74_data + (v41_data * v72_data));
              float v77_data = s0_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v79_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v79_data + (v41_data * v77_data));
              tensorforge::intel_esimd::simd<float, 8> v81_data(r0.template select<8, 1>(8));
              float v82_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v84_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v84_data + (v81_data * v82_data));
              float v87_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v89_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v89_data + (v81_data * v87_data));
              float v92_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v94_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v94_data + (v81_data * v92_data));
              float v97_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v99_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v99_data + (v81_data * v97_data));
              float v102_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v104_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v104_data + (v81_data * v102_data));
              float v107_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v109_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v109_data + (v81_data * v107_data));
              float v112_data = s0_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v114_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v114_data + (v81_data * v112_data));
              float v117_data = s0_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v119_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v119_data + (v81_data * v117_data));
              tensorforge::intel_esimd::simd<float, 8> v121_data(r0.template select<8, 1>(16));
              float v122_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v124_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v124_data + (v121_data * v122_data));
              float v127_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v129_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v129_data + (v121_data * v127_data));
              float v132_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v134_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v134_data + (v121_data * v132_data));
              float v137_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v139_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v139_data + (v121_data * v137_data));
              float v142_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v144_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v144_data + (v121_data * v142_data));
              float v147_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v149_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v149_data + (v121_data * v147_data));
              float v152_data = s0_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v154_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v154_data + (v121_data * v152_data));
              float v157_data = s0_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v159_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v159_data + (v121_data * v157_data));
              tensorforge::intel_esimd::simd<float, 8> v161_data(r0.template select<8, 1>(24));
              float v162_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v164_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v164_data + (v161_data * v162_data));
              float v167_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v169_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v169_data + (v161_data * v167_data));
              float v172_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v174_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v174_data + (v161_data * v172_data));
              float v177_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v179_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v179_data + (v161_data * v177_data));
              float v182_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v184_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v184_data + (v161_data * v182_data));
              float v187_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v189_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v189_data + (v161_data * v187_data));
              float v192_data = s0_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v194_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v194_data + (v161_data * v192_data));
              float v197_data = s0_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v199_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v199_data + (v161_data * v197_data));
              tensorforge::intel_esimd::simd<float, 8> v201_data(r0.template select<8, 1>(32));
              float v202_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v204_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v204_data + (v201_data * v202_data));
              float v207_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v209_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v209_data + (v201_data * v207_data));
              float v212_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v214_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v214_data + (v201_data * v212_data));
              float v217_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v219_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v219_data + (v201_data * v217_data));
              float v222_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v224_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v224_data + (v201_data * v222_data));
              float v227_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v229_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v229_data + (v201_data * v227_data));
              float v232_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v234_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v234_data + (v201_data * v232_data));
              float v237_data = s0_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v239_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v239_data + (v201_data * v237_data));
              tensorforge::intel_esimd::simd<float, 8> v241_data(r0.template select<8, 1>(40));
              float v242_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v244_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v244_data + (v241_data * v242_data));
              float v247_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v249_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v249_data + (v241_data * v247_data));
              float v252_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v254_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v254_data + (v241_data * v252_data));
              float v257_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v259_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v259_data + (v241_data * v257_data));
              float v262_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v264_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v264_data + (v241_data * v262_data));
              float v267_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v269_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v269_data + (v241_data * v267_data));
              float v272_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v274_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v274_data + (v241_data * v272_data));
              float v277_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v279_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v279_data + (v241_data * v277_data));
              tensorforge::intel_esimd::simd<float, 8> v281_data(r0.template select<8, 1>(48));
              float v282_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v284_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v284_data + (v281_data * v282_data));
              float v287_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v289_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v289_data + (v281_data * v287_data));
              float v292_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v294_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v294_data + (v281_data * v292_data));
              float v297_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v299_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v299_data + (v281_data * v297_data));
              float v302_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v304_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v304_data + (v281_data * v302_data));
              float v307_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v309_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v309_data + (v281_data * v307_data));
              float v312_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v314_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v314_data + (v281_data * v312_data));
              float v317_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v319_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v319_data + (v281_data * v317_data));
              tensorforge::intel_esimd::simd<float, 8> v321_data(r0.template select<8, 1>(56));
              float v322_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v324_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v324_data + (v321_data * v322_data));
              float v327_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v329_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v329_data + (v321_data * v327_data));
              float v332_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v334_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v334_data + (v321_data * v332_data));
              float v337_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v339_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v339_data + (v321_data * v337_data));
              float v342_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v344_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v344_data + (v321_data * v342_data));
              float v347_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v349_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v349_data + (v321_data * v347_data));
              float v352_data = s0_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v354_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v354_data + (v321_data * v352_data));
              float v357_data = s0_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v359_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v359_data + (v321_data * v357_data));
              // s2 = load{g>s}(glb_m3[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v361_ld;
              v361_ld.copy_from(glb_m3 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 0), v361_ld);
              tensorforge::intel_esimd::simd<float, 32> v362_ld;
              v362_ld.copy_from(glb_m3 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 32), v362_ld);
              // wait(r2 = load{g>r}(glb_m2););
              // wait(s2 = load{g>s}(glb_m3[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r3(0.0f);
              // ir3 = +(r2 * s2)
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 64> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v365_data(r2.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s2_w1 = tensorforge::slmLoad<float, 64>(s2 + 0);
              float v366_data = s2_w1[0];
              tensorforge::intel_esimd::simd<float, 8> v368_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v368_data + (v365_data * v366_data));
              float v371_data = s2_w1[8];
              tensorforge::intel_esimd::simd<float, 8> v373_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v373_data + (v365_data * v371_data));
              float v376_data = s2_w1[16];
              tensorforge::intel_esimd::simd<float, 8> v378_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v378_data + (v365_data * v376_data));
              float v381_data = s2_w1[24];
              tensorforge::intel_esimd::simd<float, 8> v383_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v383_data + (v365_data * v381_data));
              float v386_data = s2_w1[32];
              tensorforge::intel_esimd::simd<float, 8> v388_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v388_data + (v365_data * v386_data));
              float v391_data = s2_w1[40];
              tensorforge::intel_esimd::simd<float, 8> v393_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v393_data + (v365_data * v391_data));
              float v396_data = s2_w1[48];
              tensorforge::intel_esimd::simd<float, 8> v398_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v398_data + (v365_data * v396_data));
              float v401_data = s2_w1[56];
              tensorforge::intel_esimd::simd<float, 8> v403_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v403_data + (v365_data * v401_data));
              tensorforge::intel_esimd::simd<float, 8> v405_data(r2.template select<8, 1>(8));
              float v406_data = s2_w1[1];
              tensorforge::intel_esimd::simd<float, 8> v408_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v408_data + (v405_data * v406_data));
              float v411_data = s2_w1[9];
              tensorforge::intel_esimd::simd<float, 8> v413_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v413_data + (v405_data * v411_data));
              float v416_data = s2_w1[17];
              tensorforge::intel_esimd::simd<float, 8> v418_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v418_data + (v405_data * v416_data));
              float v421_data = s2_w1[25];
              tensorforge::intel_esimd::simd<float, 8> v423_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v423_data + (v405_data * v421_data));
              float v426_data = s2_w1[33];
              tensorforge::intel_esimd::simd<float, 8> v428_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v428_data + (v405_data * v426_data));
              float v431_data = s2_w1[41];
              tensorforge::intel_esimd::simd<float, 8> v433_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v433_data + (v405_data * v431_data));
              float v436_data = s2_w1[49];
              tensorforge::intel_esimd::simd<float, 8> v438_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v438_data + (v405_data * v436_data));
              float v441_data = s2_w1[57];
              tensorforge::intel_esimd::simd<float, 8> v443_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v443_data + (v405_data * v441_data));
              tensorforge::intel_esimd::simd<float, 8> v445_data(r2.template select<8, 1>(16));
              float v446_data = s2_w1[2];
              tensorforge::intel_esimd::simd<float, 8> v448_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v448_data + (v445_data * v446_data));
              float v451_data = s2_w1[10];
              tensorforge::intel_esimd::simd<float, 8> v453_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v453_data + (v445_data * v451_data));
              float v456_data = s2_w1[18];
              tensorforge::intel_esimd::simd<float, 8> v458_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v458_data + (v445_data * v456_data));
              float v461_data = s2_w1[26];
              tensorforge::intel_esimd::simd<float, 8> v463_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v463_data + (v445_data * v461_data));
              float v466_data = s2_w1[34];
              tensorforge::intel_esimd::simd<float, 8> v468_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v468_data + (v445_data * v466_data));
              float v471_data = s2_w1[42];
              tensorforge::intel_esimd::simd<float, 8> v473_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v473_data + (v445_data * v471_data));
              float v476_data = s2_w1[50];
              tensorforge::intel_esimd::simd<float, 8> v478_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v478_data + (v445_data * v476_data));
              float v481_data = s2_w1[58];
              tensorforge::intel_esimd::simd<float, 8> v483_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v483_data + (v445_data * v481_data));
              tensorforge::intel_esimd::simd<float, 8> v485_data(r2.template select<8, 1>(24));
              float v486_data = s2_w1[3];
              tensorforge::intel_esimd::simd<float, 8> v488_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v488_data + (v485_data * v486_data));
              float v491_data = s2_w1[11];
              tensorforge::intel_esimd::simd<float, 8> v493_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v493_data + (v485_data * v491_data));
              float v496_data = s2_w1[19];
              tensorforge::intel_esimd::simd<float, 8> v498_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v498_data + (v485_data * v496_data));
              float v501_data = s2_w1[27];
              tensorforge::intel_esimd::simd<float, 8> v503_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v503_data + (v485_data * v501_data));
              float v506_data = s2_w1[35];
              tensorforge::intel_esimd::simd<float, 8> v508_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v508_data + (v485_data * v506_data));
              float v511_data = s2_w1[43];
              tensorforge::intel_esimd::simd<float, 8> v513_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v513_data + (v485_data * v511_data));
              float v516_data = s2_w1[51];
              tensorforge::intel_esimd::simd<float, 8> v518_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v518_data + (v485_data * v516_data));
              float v521_data = s2_w1[59];
              tensorforge::intel_esimd::simd<float, 8> v523_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v523_data + (v485_data * v521_data));
              tensorforge::intel_esimd::simd<float, 8> v525_data(r2.template select<8, 1>(32));
              float v526_data = s2_w1[4];
              tensorforge::intel_esimd::simd<float, 8> v528_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v528_data + (v525_data * v526_data));
              float v531_data = s2_w1[12];
              tensorforge::intel_esimd::simd<float, 8> v533_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v533_data + (v525_data * v531_data));
              float v536_data = s2_w1[20];
              tensorforge::intel_esimd::simd<float, 8> v538_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v538_data + (v525_data * v536_data));
              float v541_data = s2_w1[28];
              tensorforge::intel_esimd::simd<float, 8> v543_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v543_data + (v525_data * v541_data));
              float v546_data = s2_w1[36];
              tensorforge::intel_esimd::simd<float, 8> v548_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v548_data + (v525_data * v546_data));
              float v551_data = s2_w1[44];
              tensorforge::intel_esimd::simd<float, 8> v553_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v553_data + (v525_data * v551_data));
              float v556_data = s2_w1[52];
              tensorforge::intel_esimd::simd<float, 8> v558_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v558_data + (v525_data * v556_data));
              float v561_data = s2_w1[60];
              tensorforge::intel_esimd::simd<float, 8> v563_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v563_data + (v525_data * v561_data));
              tensorforge::intel_esimd::simd<float, 8> v565_data(r2.template select<8, 1>(40));
              float v566_data = s2_w1[5];
              tensorforge::intel_esimd::simd<float, 8> v568_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v568_data + (v565_data * v566_data));
              float v571_data = s2_w1[13];
              tensorforge::intel_esimd::simd<float, 8> v573_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v573_data + (v565_data * v571_data));
              float v576_data = s2_w1[21];
              tensorforge::intel_esimd::simd<float, 8> v578_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v578_data + (v565_data * v576_data));
              float v581_data = s2_w1[29];
              tensorforge::intel_esimd::simd<float, 8> v583_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v583_data + (v565_data * v581_data));
              float v586_data = s2_w1[37];
              tensorforge::intel_esimd::simd<float, 8> v588_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v588_data + (v565_data * v586_data));
              float v591_data = s2_w1[45];
              tensorforge::intel_esimd::simd<float, 8> v593_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v593_data + (v565_data * v591_data));
              float v596_data = s2_w1[53];
              tensorforge::intel_esimd::simd<float, 8> v598_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v598_data + (v565_data * v596_data));
              float v601_data = s2_w1[61];
              tensorforge::intel_esimd::simd<float, 8> v603_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v603_data + (v565_data * v601_data));
              tensorforge::intel_esimd::simd<float, 8> v605_data(r2.template select<8, 1>(48));
              float v606_data = s2_w1[6];
              tensorforge::intel_esimd::simd<float, 8> v608_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v608_data + (v605_data * v606_data));
              float v611_data = s2_w1[14];
              tensorforge::intel_esimd::simd<float, 8> v613_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v613_data + (v605_data * v611_data));
              float v616_data = s2_w1[22];
              tensorforge::intel_esimd::simd<float, 8> v618_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v618_data + (v605_data * v616_data));
              float v621_data = s2_w1[30];
              tensorforge::intel_esimd::simd<float, 8> v623_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v623_data + (v605_data * v621_data));
              float v626_data = s2_w1[38];
              tensorforge::intel_esimd::simd<float, 8> v628_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v628_data + (v605_data * v626_data));
              float v631_data = s2_w1[46];
              tensorforge::intel_esimd::simd<float, 8> v633_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v633_data + (v605_data * v631_data));
              float v636_data = s2_w1[54];
              tensorforge::intel_esimd::simd<float, 8> v638_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v638_data + (v605_data * v636_data));
              float v641_data = s2_w1[62];
              tensorforge::intel_esimd::simd<float, 8> v643_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v643_data + (v605_data * v641_data));
              tensorforge::intel_esimd::simd<float, 8> v645_data(r2.template select<8, 1>(56));
              float v646_data = s2_w1[7];
              tensorforge::intel_esimd::simd<float, 8> v648_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v648_data + (v645_data * v646_data));
              float v651_data = s2_w1[15];
              tensorforge::intel_esimd::simd<float, 8> v653_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v653_data + (v645_data * v651_data));
              float v656_data = s2_w1[23];
              tensorforge::intel_esimd::simd<float, 8> v658_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v658_data + (v645_data * v656_data));
              float v661_data = s2_w1[31];
              tensorforge::intel_esimd::simd<float, 8> v663_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v663_data + (v645_data * v661_data));
              float v666_data = s2_w1[39];
              tensorforge::intel_esimd::simd<float, 8> v668_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v668_data + (v645_data * v666_data));
              float v671_data = s2_w1[47];
              tensorforge::intel_esimd::simd<float, 8> v673_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v673_data + (v645_data * v671_data));
              float v676_data = s2_w1[55];
              tensorforge::intel_esimd::simd<float, 8> v678_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v678_data + (v645_data * v676_data));
              float v681_data = s2_w1[63];
              tensorforge::intel_esimd::simd<float, 8> v683_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v683_data + (v645_data * v681_data));
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v685_n0 = 0; v685_n0 < 1; ++v685_n0) {
                int32_t v687_a = v685_n0 * 8;
                #pragma unroll
                for (int32_t v686_n1 = 0; v686_n1 < 8; ++v686_n1) {
                  int32_t v689_a = v687_a + (v686_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v690_data(ir3.template select<8, 1>(v689_a));
                  tensorforge::intel_esimd::simd<float, 8> v691_data(r1.template select<8, 1>(v689_a));
                  r3.template select<8, 1>(v689_a) = (v691_data + v690_data);
                }
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v693_i0 = 0; v693_i0 < 1; ++v693_i0) {
                int32_t v695_a = v693_i0 * 8;
                #pragma unroll
                for (int32_t v694_i1 = 0; v694_i1 < 8; ++v694_i1) {
                  int32_t v697_a = v695_a + (v694_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v698_data(r3.template select<8, 1>(v697_a));
                  tensorforge::slmStore<float, 8>(s1 + (v697_a), v698_data);
                }
              }
              // glb_m4 = abs(s1)
              #pragma unroll
              for (int32_t v701_k0 = 0; v701_k0 < 1; ++v701_k0) {
                int32_t v703_lead = v701_k0 * 8;
                #pragma unroll
                for (int32_t v702_k1 = 0; v702_k1 < 8; ++v702_k1) {
                  int32_t v706_a = v703_lead + (v702_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v707_data = tensorforge::slmLoad<float, 8>(s1 + (v706_a));
                  (tensorforge::intel_esimd::abs(v707_data)).copy_to(glb_m4 + (v706_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


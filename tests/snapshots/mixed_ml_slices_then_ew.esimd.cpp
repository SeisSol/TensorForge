// === base name ===
kernel_4ffb4a098587d382

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4ffb4a098587d382 = {{1, 32, 1}, 8, 8, 1, 32, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4ffb4a098587d382(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4ffb4a098587d382(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4ffb4a098587d382(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4ffb4a098587d382(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4ffb4a098587d382(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4ffb4a098587d382(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4ffb4a098587d382(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<3328 * sizeof(float)>(); {
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (104 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 8;
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
                  int32_t v25_a = v22_lead + (v21_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v26_data;
                  v26_data.copy_from(glb_m0 + (v25_a));
                  r0.template select<8, 1>(v25_a) = v26_data;
                }
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v28_ld;
              v28_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v28_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 0), v29_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 32> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v31_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> s0_w0 = tensorforge::slmLoad<float, 32>(s0 + 0);
              float v32_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v34_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v34_data + (v31_data * v32_data));
              float v37_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v39_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v39_data + (v31_data * v37_data));
              float v42_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v44_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v44_data + (v31_data * v42_data));
              float v47_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v49_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v49_data + (v31_data * v47_data));
              tensorforge::intel_esimd::simd<float, 8> v51_data(r0.template select<8, 1>(8));
              float v52_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v54_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v54_data + (v51_data * v52_data));
              float v57_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v59_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v59_data + (v51_data * v57_data));
              float v62_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v64_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v64_data + (v51_data * v62_data));
              float v67_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v69_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v69_data + (v51_data * v67_data));
              tensorforge::intel_esimd::simd<float, 8> v71_data(r0.template select<8, 1>(16));
              float v72_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v74_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v74_data + (v71_data * v72_data));
              float v77_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v79_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v79_data + (v71_data * v77_data));
              float v82_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v84_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v84_data + (v71_data * v82_data));
              float v87_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v89_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v89_data + (v71_data * v87_data));
              tensorforge::intel_esimd::simd<float, 8> v91_data(r0.template select<8, 1>(24));
              float v92_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v94_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v94_data + (v91_data * v92_data));
              float v97_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v99_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v99_data + (v91_data * v97_data));
              float v102_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v104_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v104_data + (v91_data * v102_data));
              float v107_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v109_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v109_data + (v91_data * v107_data));
              tensorforge::intel_esimd::simd<float, 8> v111_data(r0.template select<8, 1>(32));
              float v112_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v114_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v114_data + (v111_data * v112_data));
              float v117_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v119_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v119_data + (v111_data * v117_data));
              float v122_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v124_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v124_data + (v111_data * v122_data));
              float v127_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v129_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v129_data + (v111_data * v127_data));
              tensorforge::intel_esimd::simd<float, 8> v131_data(r0.template select<8, 1>(40));
              float v132_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v134_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v134_data + (v131_data * v132_data));
              float v137_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v139_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v139_data + (v131_data * v137_data));
              float v142_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v144_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v144_data + (v131_data * v142_data));
              float v147_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v149_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v149_data + (v131_data * v147_data));
              tensorforge::intel_esimd::simd<float, 8> v151_data(r0.template select<8, 1>(48));
              float v152_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v154_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v154_data + (v151_data * v152_data));
              float v157_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v159_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v159_data + (v151_data * v157_data));
              float v162_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v164_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v164_data + (v151_data * v162_data));
              float v167_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v169_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v169_data + (v151_data * v167_data));
              tensorforge::intel_esimd::simd<float, 8> v171_data(r0.template select<8, 1>(56));
              float v172_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v174_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v174_data + (v171_data * v172_data));
              float v177_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v179_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v179_data + (v171_data * v177_data));
              float v182_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v184_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v184_data + (v171_data * v182_data));
              float v187_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v189_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v189_data + (v171_data * v187_data));
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v191_i0 = 0; v191_i0 < 1; ++v191_i0) {
                int32_t v193_a = v191_i0 * 8;
                #pragma unroll
                for (int32_t v192_i1 = 0; v192_i1 < 4; ++v192_i1) {
                  int32_t v195_a = v193_a + (v192_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v196_data(r1.template select<8, 1>(v195_a));
                  tensorforge::slmStore<float, 8>(s1 + (v195_a), v196_data);
                }
              }
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 32> r2(0.0f);
              // ir2 = +(r0 * s2)
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 32> ir2(0.0f);
              tensorforge::intel_esimd::simd<float, 32> s2_w1 = tensorforge::slmLoad<float, 32>(s2 + 0);
              float v202_data = s2_w1[0];
              tensorforge::intel_esimd::simd<float, 8> v204_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v204_data + (v31_data * v202_data));
              float v207_data = s2_w1[8];
              tensorforge::intel_esimd::simd<float, 8> v209_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v209_data + (v31_data * v207_data));
              float v212_data = s2_w1[16];
              tensorforge::intel_esimd::simd<float, 8> v214_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v214_data + (v31_data * v212_data));
              float v217_data = s2_w1[24];
              tensorforge::intel_esimd::simd<float, 8> v219_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v219_data + (v31_data * v217_data));
              float v222_data = s2_w1[1];
              tensorforge::intel_esimd::simd<float, 8> v224_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v224_data + (v51_data * v222_data));
              float v227_data = s2_w1[9];
              tensorforge::intel_esimd::simd<float, 8> v229_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v229_data + (v51_data * v227_data));
              float v232_data = s2_w1[17];
              tensorforge::intel_esimd::simd<float, 8> v234_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v234_data + (v51_data * v232_data));
              float v237_data = s2_w1[25];
              tensorforge::intel_esimd::simd<float, 8> v239_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v239_data + (v51_data * v237_data));
              float v242_data = s2_w1[2];
              tensorforge::intel_esimd::simd<float, 8> v244_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v244_data + (v71_data * v242_data));
              float v247_data = s2_w1[10];
              tensorforge::intel_esimd::simd<float, 8> v249_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v249_data + (v71_data * v247_data));
              float v252_data = s2_w1[18];
              tensorforge::intel_esimd::simd<float, 8> v254_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v254_data + (v71_data * v252_data));
              float v257_data = s2_w1[26];
              tensorforge::intel_esimd::simd<float, 8> v259_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v259_data + (v71_data * v257_data));
              float v262_data = s2_w1[3];
              tensorforge::intel_esimd::simd<float, 8> v264_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v264_data + (v91_data * v262_data));
              float v267_data = s2_w1[11];
              tensorforge::intel_esimd::simd<float, 8> v269_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v269_data + (v91_data * v267_data));
              float v272_data = s2_w1[19];
              tensorforge::intel_esimd::simd<float, 8> v274_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v274_data + (v91_data * v272_data));
              float v277_data = s2_w1[27];
              tensorforge::intel_esimd::simd<float, 8> v279_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v279_data + (v91_data * v277_data));
              float v282_data = s2_w1[4];
              tensorforge::intel_esimd::simd<float, 8> v284_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v284_data + (v111_data * v282_data));
              float v287_data = s2_w1[12];
              tensorforge::intel_esimd::simd<float, 8> v289_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v289_data + (v111_data * v287_data));
              float v292_data = s2_w1[20];
              tensorforge::intel_esimd::simd<float, 8> v294_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v294_data + (v111_data * v292_data));
              float v297_data = s2_w1[28];
              tensorforge::intel_esimd::simd<float, 8> v299_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v299_data + (v111_data * v297_data));
              float v302_data = s2_w1[5];
              tensorforge::intel_esimd::simd<float, 8> v304_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v304_data + (v131_data * v302_data));
              float v307_data = s2_w1[13];
              tensorforge::intel_esimd::simd<float, 8> v309_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v309_data + (v131_data * v307_data));
              float v312_data = s2_w1[21];
              tensorforge::intel_esimd::simd<float, 8> v314_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v314_data + (v131_data * v312_data));
              float v317_data = s2_w1[29];
              tensorforge::intel_esimd::simd<float, 8> v319_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v319_data + (v131_data * v317_data));
              float v322_data = s2_w1[6];
              tensorforge::intel_esimd::simd<float, 8> v324_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v324_data + (v151_data * v322_data));
              float v327_data = s2_w1[14];
              tensorforge::intel_esimd::simd<float, 8> v329_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v329_data + (v151_data * v327_data));
              float v332_data = s2_w1[22];
              tensorforge::intel_esimd::simd<float, 8> v334_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v334_data + (v151_data * v332_data));
              float v337_data = s2_w1[30];
              tensorforge::intel_esimd::simd<float, 8> v339_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v339_data + (v151_data * v337_data));
              float v342_data = s2_w1[7];
              tensorforge::intel_esimd::simd<float, 8> v344_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v344_data + (v171_data * v342_data));
              float v347_data = s2_w1[15];
              tensorforge::intel_esimd::simd<float, 8> v349_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v349_data + (v171_data * v347_data));
              float v352_data = s2_w1[23];
              tensorforge::intel_esimd::simd<float, 8> v354_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v354_data + (v171_data * v352_data));
              float v357_data = s2_w1[31];
              tensorforge::intel_esimd::simd<float, 8> v359_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v359_data + (v171_data * v357_data));
              // r2 = ir2
              #pragma unroll
              for (int32_t v361_n0 = 0; v361_n0 < 1; ++v361_n0) {
                int32_t v363_a = v361_n0 * 8;
                #pragma unroll
                for (int32_t v362_n1 = 0; v362_n1 < 4; ++v362_n1) {
                  int32_t v365_a = v363_a + (v362_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v366_data(ir2.template select<8, 1>(v365_a));
                  r2.template select<8, 1>(v365_a) = v366_data;
                }
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v367_i0 = 0; v367_i0 < 1; ++v367_i0) {
                int32_t v369_a = v367_i0 * 8;
                #pragma unroll
                for (int32_t v368_i1 = 0; v368_i1 < 4; ++v368_i1) {
                  tensorforge::intel_esimd::simd<float, 8> v372_data(r2.template select<8, 1>((v369_a + (v368_i1 * 8))));
                  tensorforge::slmStore<float, 8>(s1 + ((v369_a + ((v368_i1 + 4) * 8))), v372_data);
                }
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v377_k0 = 0; v377_k0 < 1; ++v377_k0) {
                int32_t v379_lead = v377_k0 * 8;
                #pragma unroll
                for (int32_t v378_k1 = 0; v378_k1 < 8; ++v378_k1) {
                  int32_t v382_a = v379_lead + (v378_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v383_data = tensorforge::slmLoad<float, 8>(s1 + (v382_a));
                  (tensorforge::intel_esimd::abs(v383_data)).copy_to(glb_m3 + (v382_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


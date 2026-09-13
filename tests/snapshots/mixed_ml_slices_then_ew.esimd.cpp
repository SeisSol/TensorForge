// === base name ===
kernel_5be871864761db53

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5be871864761db53 = {{1, 32, 1}, 8, 8, 1, 32, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5be871864761db53(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5be871864761db53(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5be871864761db53(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 32, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 32;
  config.block[2] = 1;
  config.sharedMemBytes = 3328 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5be871864761db53(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5be871864761db53(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5be871864761db53(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5be871864761db53(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const float *const __restrict__ pf_glb_m0 = &m0[v10_batchId1 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ pf_glb_m1 = &m1[v10_batchId1 * 32 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v10_batchId1 * 32 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                int32_t v25_lead = v23_i0 * 8;
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
                  int32_t v28_a = v25_lead + (v24_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v29_data;
                  v29_data.copy_from(glb_m0 + (v28_a));
                  r0.template select<8, 1>(v28_a) = v29_data;
                }
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v31_ld;
              v31_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v31_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 0), v32_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 32> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v34_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> s0_w0 = tensorforge::slmLoad<float, 32>(s0 + 0);
              float v35_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v37_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v37_data + (v34_data * v35_data));
              float v40_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v42_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v42_data + (v34_data * v40_data));
              float v45_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v47_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v47_data + (v34_data * v45_data));
              float v50_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v52_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v52_data + (v34_data * v50_data));
              tensorforge::intel_esimd::simd<float, 8> v54_data(r0.template select<8, 1>(8));
              float v55_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v57_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v57_data + (v54_data * v55_data));
              float v60_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v62_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v62_data + (v54_data * v60_data));
              float v65_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v67_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v67_data + (v54_data * v65_data));
              float v70_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v72_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v72_data + (v54_data * v70_data));
              tensorforge::intel_esimd::simd<float, 8> v74_data(r0.template select<8, 1>(16));
              float v75_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v77_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v77_data + (v74_data * v75_data));
              float v80_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v82_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v82_data + (v74_data * v80_data));
              float v85_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v87_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v87_data + (v74_data * v85_data));
              float v90_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v92_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v92_data + (v74_data * v90_data));
              tensorforge::intel_esimd::simd<float, 8> v94_data(r0.template select<8, 1>(24));
              float v95_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v97_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v97_data + (v94_data * v95_data));
              float v100_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v102_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v102_data + (v94_data * v100_data));
              float v105_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v107_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v107_data + (v94_data * v105_data));
              float v110_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v112_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v112_data + (v94_data * v110_data));
              tensorforge::intel_esimd::simd<float, 8> v114_data(r0.template select<8, 1>(32));
              float v115_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v117_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v117_data + (v114_data * v115_data));
              float v120_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v122_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v122_data + (v114_data * v120_data));
              float v125_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v127_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v127_data + (v114_data * v125_data));
              float v130_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v132_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v132_data + (v114_data * v130_data));
              tensorforge::intel_esimd::simd<float, 8> v134_data(r0.template select<8, 1>(40));
              float v135_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v137_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v137_data + (v134_data * v135_data));
              float v140_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v142_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v142_data + (v134_data * v140_data));
              float v145_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v147_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v147_data + (v134_data * v145_data));
              float v150_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v152_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v152_data + (v134_data * v150_data));
              tensorforge::intel_esimd::simd<float, 8> v154_data(r0.template select<8, 1>(48));
              float v155_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v157_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v157_data + (v154_data * v155_data));
              float v160_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v162_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v162_data + (v154_data * v160_data));
              float v165_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v167_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v167_data + (v154_data * v165_data));
              float v170_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v172_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v172_data + (v154_data * v170_data));
              tensorforge::intel_esimd::simd<float, 8> v174_data(r0.template select<8, 1>(56));
              float v175_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v177_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v177_data + (v174_data * v175_data));
              float v180_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v182_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v182_data + (v174_data * v180_data));
              float v185_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v187_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v187_data + (v174_data * v185_data));
              float v190_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v192_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v192_data + (v174_data * v190_data));
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v194_i0 = 0; v194_i0 < 1; ++v194_i0) {
                int32_t v196_a = v194_i0 * 8;
                #pragma unroll
                for (int32_t v195_i1 = 0; v195_i1 < 4; ++v195_i1) {
                  int32_t v198_a = v196_a + (v195_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v199_data(r1.template select<8, 1>(v198_a));
                  tensorforge::slmStore<float, 8>(s1 + (v198_a), v199_data);
                }
              }
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 32> r2(0.0f);
              // ir2 = +(r0 * s2)
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 32> ir2(0.0f);
              tensorforge::intel_esimd::simd<float, 32> s2_w1 = tensorforge::slmLoad<float, 32>(s2 + 0);
              float v205_data = s2_w1[0];
              tensorforge::intel_esimd::simd<float, 8> v207_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v207_data + (v34_data * v205_data));
              float v210_data = s2_w1[8];
              tensorforge::intel_esimd::simd<float, 8> v212_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v212_data + (v34_data * v210_data));
              float v215_data = s2_w1[16];
              tensorforge::intel_esimd::simd<float, 8> v217_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v217_data + (v34_data * v215_data));
              float v220_data = s2_w1[24];
              tensorforge::intel_esimd::simd<float, 8> v222_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v222_data + (v34_data * v220_data));
              float v225_data = s2_w1[1];
              tensorforge::intel_esimd::simd<float, 8> v227_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v227_data + (v54_data * v225_data));
              float v230_data = s2_w1[9];
              tensorforge::intel_esimd::simd<float, 8> v232_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v232_data + (v54_data * v230_data));
              float v235_data = s2_w1[17];
              tensorforge::intel_esimd::simd<float, 8> v237_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v237_data + (v54_data * v235_data));
              float v240_data = s2_w1[25];
              tensorforge::intel_esimd::simd<float, 8> v242_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v242_data + (v54_data * v240_data));
              float v245_data = s2_w1[2];
              tensorforge::intel_esimd::simd<float, 8> v247_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v247_data + (v74_data * v245_data));
              float v250_data = s2_w1[10];
              tensorforge::intel_esimd::simd<float, 8> v252_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v252_data + (v74_data * v250_data));
              float v255_data = s2_w1[18];
              tensorforge::intel_esimd::simd<float, 8> v257_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v257_data + (v74_data * v255_data));
              float v260_data = s2_w1[26];
              tensorforge::intel_esimd::simd<float, 8> v262_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v262_data + (v74_data * v260_data));
              float v265_data = s2_w1[3];
              tensorforge::intel_esimd::simd<float, 8> v267_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v267_data + (v94_data * v265_data));
              float v270_data = s2_w1[11];
              tensorforge::intel_esimd::simd<float, 8> v272_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v272_data + (v94_data * v270_data));
              float v275_data = s2_w1[19];
              tensorforge::intel_esimd::simd<float, 8> v277_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v277_data + (v94_data * v275_data));
              float v280_data = s2_w1[27];
              tensorforge::intel_esimd::simd<float, 8> v282_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v282_data + (v94_data * v280_data));
              float v285_data = s2_w1[4];
              tensorforge::intel_esimd::simd<float, 8> v287_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v287_data + (v114_data * v285_data));
              float v290_data = s2_w1[12];
              tensorforge::intel_esimd::simd<float, 8> v292_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v292_data + (v114_data * v290_data));
              float v295_data = s2_w1[20];
              tensorforge::intel_esimd::simd<float, 8> v297_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v297_data + (v114_data * v295_data));
              float v300_data = s2_w1[28];
              tensorforge::intel_esimd::simd<float, 8> v302_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v302_data + (v114_data * v300_data));
              float v305_data = s2_w1[5];
              tensorforge::intel_esimd::simd<float, 8> v307_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v307_data + (v134_data * v305_data));
              float v310_data = s2_w1[13];
              tensorforge::intel_esimd::simd<float, 8> v312_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v312_data + (v134_data * v310_data));
              float v315_data = s2_w1[21];
              tensorforge::intel_esimd::simd<float, 8> v317_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v317_data + (v134_data * v315_data));
              float v320_data = s2_w1[29];
              tensorforge::intel_esimd::simd<float, 8> v322_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v322_data + (v134_data * v320_data));
              float v325_data = s2_w1[6];
              tensorforge::intel_esimd::simd<float, 8> v327_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v327_data + (v154_data * v325_data));
              float v330_data = s2_w1[14];
              tensorforge::intel_esimd::simd<float, 8> v332_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v332_data + (v154_data * v330_data));
              float v335_data = s2_w1[22];
              tensorforge::intel_esimd::simd<float, 8> v337_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v337_data + (v154_data * v335_data));
              float v340_data = s2_w1[30];
              tensorforge::intel_esimd::simd<float, 8> v342_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v342_data + (v154_data * v340_data));
              float v345_data = s2_w1[7];
              tensorforge::intel_esimd::simd<float, 8> v347_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v347_data + (v174_data * v345_data));
              float v350_data = s2_w1[15];
              tensorforge::intel_esimd::simd<float, 8> v352_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v352_data + (v174_data * v350_data));
              float v355_data = s2_w1[23];
              tensorforge::intel_esimd::simd<float, 8> v357_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v357_data + (v174_data * v355_data));
              float v360_data = s2_w1[31];
              tensorforge::intel_esimd::simd<float, 8> v362_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v362_data + (v174_data * v360_data));
              // r2 = ir2
              #pragma unroll
              for (int32_t v364_n0 = 0; v364_n0 < 1; ++v364_n0) {
                int32_t v366_a = v364_n0 * 8;
                #pragma unroll
                for (int32_t v365_n1 = 0; v365_n1 < 4; ++v365_n1) {
                  int32_t v368_a = v366_a + (v365_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v369_data(ir2.template select<8, 1>(v368_a));
                  r2.template select<8, 1>(v368_a) = v369_data;
                }
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v370_i0 = 0; v370_i0 < 1; ++v370_i0) {
                int32_t v372_a = v370_i0 * 8;
                #pragma unroll
                for (int32_t v371_i1 = 0; v371_i1 < 4; ++v371_i1) {
                  tensorforge::intel_esimd::simd<float, 8> v375_data(r2.template select<8, 1>((v372_a + (v371_i1 * 8))));
                  tensorforge::slmStore<float, 8>(s1 + ((v372_a + ((v371_i1 + 4) * 8))), v375_data);
                }
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v380_k0 = 0; v380_k0 < 1; ++v380_k0) {
                int32_t v382_lead = v380_k0 * 8;
                #pragma unroll
                for (int32_t v381_k1 = 0; v381_k1 < 8; ++v381_k1) {
                  int32_t v385_a = v382_lead + (v381_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v386_data = tensorforge::slmLoad<float, 8>(s1 + (v385_a));
                  (tensorforge::intel_esimd::abs(v386_data)).copy_to(glb_m3 + (v385_a));
                }
              }
            }
            tensorforge::prefetchRunsL2<256, 128, 128>(&pf_glb_m0[0], &pf_glb_m1[0], &pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


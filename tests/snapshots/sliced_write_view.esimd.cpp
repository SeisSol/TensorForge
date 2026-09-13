// === base name ===
kernel_1035bcd3f7a3e7f6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1035bcd3f7a3e7f6 = {{1, 8, 1}, 32, 32, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1035bcd3f7a3e7f6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1035bcd3f7a3e7f6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1035bcd3f7a3e7f6(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 1408 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_1035bcd3f7a3e7f6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1035bcd3f7a3e7f6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1035bcd3f7a3e7f6(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1035bcd3f7a3e7f6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1408 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 8 per block = block 1x8x1, 5632 B shared, occupancy grid
        // operands:
        //   m0 32×13(32×13) {0..32}×{0..13} strided
        //   m1 32×13(32×13) {0..32}×{0..13} strided
        //   m2 13×13(13×13) {0..13}×{0..13} strided
        //   m3 32×13(32×13) {0..32}×{0..13} strided
        //   m4 13×13(13×13) {0..13}×{0..13} strided
        // operations:
        //   m0[i,j]@{0..32}×{8..9} = m1[i,k]@{0..32}×{10..13} × m2[k,j]@{10..13}×{8..9}
        //   m3[i,j] = m0[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,0],[13,1]],"is_tmp":false,"name":"m2","offset":[0,8],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (176);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v9_batchId1 * 416 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v9_batchId1 * 169 + 0 + m2_extraOffset];
            float *const __restrict__ pf_glb_m0 = &m0[v9_batchId1 * 416 + 0 + m0_extraOffset];
            const float *const __restrict__ pf_glb_m4 = &m4[v9_batchId1 * 169 + 0 + m4_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 169 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 169 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 96> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v26_lead = v24_i0 * 32;
                #pragma unroll
                for (int32_t v25_i1 = 10; v25_i1 < 13; ++v25_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v30_data;
                  v30_data.copy_from(glb_m1 + ((v26_lead + (v25_i1 * 32))));
                  r0.template select<32, 1>((v26_lead + ((v25_i1 - 10) * 32))) = v30_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v34_ld);
              tensorforge::intel_esimd::simd<float, 32> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 128), v35_ld);
              tensorforge::intel_esimd::simd<float, 9> v36_ld;
              v36_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 160), v36_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 32> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 32), (0, 1)] [(10, 13)]
              tensorforge::intel_esimd::simd<float, 32> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v39_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> s0_w0 = tensorforge::slmLoad<float, 16>(s0 + 114);
              float v40_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v42_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v42_data + (v39_data * v40_data));
              tensorforge::intel_esimd::simd<float, 32> v44_data(r0.template select<32, 1>(32));
              float v45_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v47_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v47_data + (v44_data * v45_data));
              tensorforge::intel_esimd::simd<float, 32> v49_data(r0.template select<32, 1>(64));
              float v50_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v52_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v52_data + (v49_data * v50_data));
              #pragma unroll
              for (int32_t v54_n0 = 0; v54_n0 < 1; ++v54_n0) {
                int32_t v56_a = v54_n0 * 32;
                #pragma unroll
                for (int32_t v55_n1 = 0; v55_n1 < 1; ++v55_n1) {
                  int32_t v58_a = v56_a + (v55_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v59_data(ir1.template select<32, 1>(v58_a));
                  r1.template select<32, 1>(v58_a) = v59_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v60_i0 = 0; v60_i0 < 1; ++v60_i0) {
                int32_t v62_a = v60_i0 * 32;
                #pragma unroll
                for (int32_t v61_i1 = 0; v61_i1 < 1; ++v61_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v65_data(r1.template select<32, 1>((v62_a + (v61_i1 * 32))));
                  v65_data.copy_to(glb_m0 + ((v62_a + ((v61_i1 + 8) * 32))));
                }
              }
              tensorforge::intel_esimd::simd<float, 416> r2(0.0f);
              // r2 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v71_i0 = 0; v71_i0 < 1; ++v71_i0) {
                int32_t v73_lead = v71_i0 * 32;
                #pragma unroll
                for (int32_t v72_i1 = 0; v72_i1 < 13; ++v72_i1) {
                  int32_t v76_a = v73_lead + (v72_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v77_data;
                  v77_data.copy_from(glb_m0 + (v76_a));
                  r2.template select<32, 1>(v76_a) = v77_data;
                }
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v79_ld;
              v79_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s1 + (0 + 0 + 4 * 0 + 0), v79_ld);
              tensorforge::intel_esimd::simd<float, 32> v80_ld;
              v80_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 1 * 0 + 128), v80_ld);
              tensorforge::intel_esimd::simd<float, 9> v81_ld;
              v81_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s1 + (0 + 0 + 1 * 0 + 160), v81_ld);
              // wait(r2 = load{g>r}(glb_m0););
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 416> r3(0.0f);
              // r3 = +(r2 * s1) + None
              // [(0, 32), (0, 13)] [(0, 13)]
              tensorforge::intel_esimd::simd<float, 416> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v84_data(r2.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 176> s1_w1 = tensorforge::slmLoad<float, 176>(s1 + 0);
              float v85_data = s1_w1[0];
              tensorforge::intel_esimd::simd<float, 32> v87_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v87_data + (v84_data * v85_data));
              float v90_data = s1_w1[13];
              tensorforge::intel_esimd::simd<float, 32> v92_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v92_data + (v84_data * v90_data));
              float v95_data = s1_w1[26];
              tensorforge::intel_esimd::simd<float, 32> v97_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v97_data + (v84_data * v95_data));
              float v100_data = s1_w1[39];
              tensorforge::intel_esimd::simd<float, 32> v102_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v102_data + (v84_data * v100_data));
              float v105_data = s1_w1[52];
              tensorforge::intel_esimd::simd<float, 32> v107_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v107_data + (v84_data * v105_data));
              float v110_data = s1_w1[65];
              tensorforge::intel_esimd::simd<float, 32> v112_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v112_data + (v84_data * v110_data));
              float v115_data = s1_w1[78];
              tensorforge::intel_esimd::simd<float, 32> v117_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v117_data + (v84_data * v115_data));
              float v120_data = s1_w1[91];
              tensorforge::intel_esimd::simd<float, 32> v122_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v122_data + (v84_data * v120_data));
              float v125_data = s1_w1[104];
              tensorforge::intel_esimd::simd<float, 32> v127_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v127_data + (v84_data * v125_data));
              float v130_data = s1_w1[117];
              tensorforge::intel_esimd::simd<float, 32> v132_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v132_data + (v84_data * v130_data));
              float v135_data = s1_w1[130];
              tensorforge::intel_esimd::simd<float, 32> v137_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v137_data + (v84_data * v135_data));
              float v140_data = s1_w1[143];
              tensorforge::intel_esimd::simd<float, 32> v142_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v142_data + (v84_data * v140_data));
              float v145_data = s1_w1[156];
              tensorforge::intel_esimd::simd<float, 32> v147_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v147_data + (v84_data * v145_data));
              tensorforge::intel_esimd::simd<float, 32> v149_data(r2.template select<32, 1>(32));
              float v150_data = s1_w1[1];
              tensorforge::intel_esimd::simd<float, 32> v152_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v152_data + (v149_data * v150_data));
              float v155_data = s1_w1[14];
              tensorforge::intel_esimd::simd<float, 32> v157_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v157_data + (v149_data * v155_data));
              float v160_data = s1_w1[27];
              tensorforge::intel_esimd::simd<float, 32> v162_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v162_data + (v149_data * v160_data));
              float v165_data = s1_w1[40];
              tensorforge::intel_esimd::simd<float, 32> v167_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v167_data + (v149_data * v165_data));
              float v170_data = s1_w1[53];
              tensorforge::intel_esimd::simd<float, 32> v172_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v172_data + (v149_data * v170_data));
              float v175_data = s1_w1[66];
              tensorforge::intel_esimd::simd<float, 32> v177_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v177_data + (v149_data * v175_data));
              float v180_data = s1_w1[79];
              tensorforge::intel_esimd::simd<float, 32> v182_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v182_data + (v149_data * v180_data));
              float v185_data = s1_w1[92];
              tensorforge::intel_esimd::simd<float, 32> v187_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v187_data + (v149_data * v185_data));
              float v190_data = s1_w1[105];
              tensorforge::intel_esimd::simd<float, 32> v192_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v192_data + (v149_data * v190_data));
              float v195_data = s1_w1[118];
              tensorforge::intel_esimd::simd<float, 32> v197_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v197_data + (v149_data * v195_data));
              float v200_data = s1_w1[131];
              tensorforge::intel_esimd::simd<float, 32> v202_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v202_data + (v149_data * v200_data));
              float v205_data = s1_w1[144];
              tensorforge::intel_esimd::simd<float, 32> v207_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v207_data + (v149_data * v205_data));
              float v210_data = s1_w1[157];
              tensorforge::intel_esimd::simd<float, 32> v212_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v212_data + (v149_data * v210_data));
              tensorforge::intel_esimd::simd<float, 32> v214_data(r2.template select<32, 1>(64));
              float v215_data = s1_w1[2];
              tensorforge::intel_esimd::simd<float, 32> v217_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v217_data + (v214_data * v215_data));
              float v220_data = s1_w1[15];
              tensorforge::intel_esimd::simd<float, 32> v222_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v222_data + (v214_data * v220_data));
              float v225_data = s1_w1[28];
              tensorforge::intel_esimd::simd<float, 32> v227_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v227_data + (v214_data * v225_data));
              float v230_data = s1_w1[41];
              tensorforge::intel_esimd::simd<float, 32> v232_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v232_data + (v214_data * v230_data));
              float v235_data = s1_w1[54];
              tensorforge::intel_esimd::simd<float, 32> v237_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v237_data + (v214_data * v235_data));
              float v240_data = s1_w1[67];
              tensorforge::intel_esimd::simd<float, 32> v242_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v242_data + (v214_data * v240_data));
              float v245_data = s1_w1[80];
              tensorforge::intel_esimd::simd<float, 32> v247_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v247_data + (v214_data * v245_data));
              float v250_data = s1_w1[93];
              tensorforge::intel_esimd::simd<float, 32> v252_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v252_data + (v214_data * v250_data));
              float v255_data = s1_w1[106];
              tensorforge::intel_esimd::simd<float, 32> v257_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v257_data + (v214_data * v255_data));
              float v260_data = s1_w1[119];
              tensorforge::intel_esimd::simd<float, 32> v262_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v262_data + (v214_data * v260_data));
              float v265_data = s1_w1[132];
              tensorforge::intel_esimd::simd<float, 32> v267_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v267_data + (v214_data * v265_data));
              float v270_data = s1_w1[145];
              tensorforge::intel_esimd::simd<float, 32> v272_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v272_data + (v214_data * v270_data));
              float v275_data = s1_w1[158];
              tensorforge::intel_esimd::simd<float, 32> v277_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v277_data + (v214_data * v275_data));
              tensorforge::intel_esimd::simd<float, 32> v279_data(r2.template select<32, 1>(96));
              float v280_data = s1_w1[3];
              tensorforge::intel_esimd::simd<float, 32> v282_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v282_data + (v279_data * v280_data));
              float v285_data = s1_w1[16];
              tensorforge::intel_esimd::simd<float, 32> v287_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v287_data + (v279_data * v285_data));
              float v290_data = s1_w1[29];
              tensorforge::intel_esimd::simd<float, 32> v292_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v292_data + (v279_data * v290_data));
              float v295_data = s1_w1[42];
              tensorforge::intel_esimd::simd<float, 32> v297_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v297_data + (v279_data * v295_data));
              float v300_data = s1_w1[55];
              tensorforge::intel_esimd::simd<float, 32> v302_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v302_data + (v279_data * v300_data));
              float v305_data = s1_w1[68];
              tensorforge::intel_esimd::simd<float, 32> v307_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v307_data + (v279_data * v305_data));
              float v310_data = s1_w1[81];
              tensorforge::intel_esimd::simd<float, 32> v312_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v312_data + (v279_data * v310_data));
              float v315_data = s1_w1[94];
              tensorforge::intel_esimd::simd<float, 32> v317_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v317_data + (v279_data * v315_data));
              float v320_data = s1_w1[107];
              tensorforge::intel_esimd::simd<float, 32> v322_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v322_data + (v279_data * v320_data));
              float v325_data = s1_w1[120];
              tensorforge::intel_esimd::simd<float, 32> v327_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v327_data + (v279_data * v325_data));
              float v330_data = s1_w1[133];
              tensorforge::intel_esimd::simd<float, 32> v332_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v332_data + (v279_data * v330_data));
              float v335_data = s1_w1[146];
              tensorforge::intel_esimd::simd<float, 32> v337_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v337_data + (v279_data * v335_data));
              float v340_data = s1_w1[159];
              tensorforge::intel_esimd::simd<float, 32> v342_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v342_data + (v279_data * v340_data));
              tensorforge::intel_esimd::simd<float, 32> v344_data(r2.template select<32, 1>(128));
              float v345_data = s1_w1[4];
              tensorforge::intel_esimd::simd<float, 32> v347_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v347_data + (v344_data * v345_data));
              float v350_data = s1_w1[17];
              tensorforge::intel_esimd::simd<float, 32> v352_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v352_data + (v344_data * v350_data));
              float v355_data = s1_w1[30];
              tensorforge::intel_esimd::simd<float, 32> v357_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v357_data + (v344_data * v355_data));
              float v360_data = s1_w1[43];
              tensorforge::intel_esimd::simd<float, 32> v362_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v362_data + (v344_data * v360_data));
              float v365_data = s1_w1[56];
              tensorforge::intel_esimd::simd<float, 32> v367_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v367_data + (v344_data * v365_data));
              float v370_data = s1_w1[69];
              tensorforge::intel_esimd::simd<float, 32> v372_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v372_data + (v344_data * v370_data));
              float v375_data = s1_w1[82];
              tensorforge::intel_esimd::simd<float, 32> v377_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v377_data + (v344_data * v375_data));
              float v380_data = s1_w1[95];
              tensorforge::intel_esimd::simd<float, 32> v382_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v382_data + (v344_data * v380_data));
              float v385_data = s1_w1[108];
              tensorforge::intel_esimd::simd<float, 32> v387_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v387_data + (v344_data * v385_data));
              float v390_data = s1_w1[121];
              tensorforge::intel_esimd::simd<float, 32> v392_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v392_data + (v344_data * v390_data));
              float v395_data = s1_w1[134];
              tensorforge::intel_esimd::simd<float, 32> v397_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v397_data + (v344_data * v395_data));
              float v400_data = s1_w1[147];
              tensorforge::intel_esimd::simd<float, 32> v402_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v402_data + (v344_data * v400_data));
              float v405_data = s1_w1[160];
              tensorforge::intel_esimd::simd<float, 32> v407_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v407_data + (v344_data * v405_data));
              tensorforge::intel_esimd::simd<float, 32> v409_data(r2.template select<32, 1>(160));
              float v410_data = s1_w1[5];
              tensorforge::intel_esimd::simd<float, 32> v412_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v412_data + (v409_data * v410_data));
              float v415_data = s1_w1[18];
              tensorforge::intel_esimd::simd<float, 32> v417_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v417_data + (v409_data * v415_data));
              float v420_data = s1_w1[31];
              tensorforge::intel_esimd::simd<float, 32> v422_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v422_data + (v409_data * v420_data));
              float v425_data = s1_w1[44];
              tensorforge::intel_esimd::simd<float, 32> v427_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v427_data + (v409_data * v425_data));
              float v430_data = s1_w1[57];
              tensorforge::intel_esimd::simd<float, 32> v432_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v432_data + (v409_data * v430_data));
              float v435_data = s1_w1[70];
              tensorforge::intel_esimd::simd<float, 32> v437_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v437_data + (v409_data * v435_data));
              float v440_data = s1_w1[83];
              tensorforge::intel_esimd::simd<float, 32> v442_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v442_data + (v409_data * v440_data));
              float v445_data = s1_w1[96];
              tensorforge::intel_esimd::simd<float, 32> v447_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v447_data + (v409_data * v445_data));
              float v450_data = s1_w1[109];
              tensorforge::intel_esimd::simd<float, 32> v452_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v452_data + (v409_data * v450_data));
              float v455_data = s1_w1[122];
              tensorforge::intel_esimd::simd<float, 32> v457_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v457_data + (v409_data * v455_data));
              float v460_data = s1_w1[135];
              tensorforge::intel_esimd::simd<float, 32> v462_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v462_data + (v409_data * v460_data));
              float v465_data = s1_w1[148];
              tensorforge::intel_esimd::simd<float, 32> v467_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v467_data + (v409_data * v465_data));
              float v470_data = s1_w1[161];
              tensorforge::intel_esimd::simd<float, 32> v472_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v472_data + (v409_data * v470_data));
              tensorforge::intel_esimd::simd<float, 32> v474_data(r2.template select<32, 1>(192));
              float v475_data = s1_w1[6];
              tensorforge::intel_esimd::simd<float, 32> v477_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v477_data + (v474_data * v475_data));
              float v480_data = s1_w1[19];
              tensorforge::intel_esimd::simd<float, 32> v482_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v482_data + (v474_data * v480_data));
              float v485_data = s1_w1[32];
              tensorforge::intel_esimd::simd<float, 32> v487_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v487_data + (v474_data * v485_data));
              float v490_data = s1_w1[45];
              tensorforge::intel_esimd::simd<float, 32> v492_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v492_data + (v474_data * v490_data));
              float v495_data = s1_w1[58];
              tensorforge::intel_esimd::simd<float, 32> v497_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v497_data + (v474_data * v495_data));
              float v500_data = s1_w1[71];
              tensorforge::intel_esimd::simd<float, 32> v502_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v502_data + (v474_data * v500_data));
              float v505_data = s1_w1[84];
              tensorforge::intel_esimd::simd<float, 32> v507_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v507_data + (v474_data * v505_data));
              float v510_data = s1_w1[97];
              tensorforge::intel_esimd::simd<float, 32> v512_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v512_data + (v474_data * v510_data));
              float v515_data = s1_w1[110];
              tensorforge::intel_esimd::simd<float, 32> v517_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v517_data + (v474_data * v515_data));
              float v520_data = s1_w1[123];
              tensorforge::intel_esimd::simd<float, 32> v522_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v522_data + (v474_data * v520_data));
              float v525_data = s1_w1[136];
              tensorforge::intel_esimd::simd<float, 32> v527_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v527_data + (v474_data * v525_data));
              float v530_data = s1_w1[149];
              tensorforge::intel_esimd::simd<float, 32> v532_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v532_data + (v474_data * v530_data));
              float v535_data = s1_w1[162];
              tensorforge::intel_esimd::simd<float, 32> v537_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v537_data + (v474_data * v535_data));
              tensorforge::intel_esimd::simd<float, 32> v539_data(r2.template select<32, 1>(224));
              float v540_data = s1_w1[7];
              tensorforge::intel_esimd::simd<float, 32> v542_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v542_data + (v539_data * v540_data));
              float v545_data = s1_w1[20];
              tensorforge::intel_esimd::simd<float, 32> v547_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v547_data + (v539_data * v545_data));
              float v550_data = s1_w1[33];
              tensorforge::intel_esimd::simd<float, 32> v552_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v552_data + (v539_data * v550_data));
              float v555_data = s1_w1[46];
              tensorforge::intel_esimd::simd<float, 32> v557_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v557_data + (v539_data * v555_data));
              float v560_data = s1_w1[59];
              tensorforge::intel_esimd::simd<float, 32> v562_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v562_data + (v539_data * v560_data));
              float v565_data = s1_w1[72];
              tensorforge::intel_esimd::simd<float, 32> v567_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v567_data + (v539_data * v565_data));
              float v570_data = s1_w1[85];
              tensorforge::intel_esimd::simd<float, 32> v572_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v572_data + (v539_data * v570_data));
              float v575_data = s1_w1[98];
              tensorforge::intel_esimd::simd<float, 32> v577_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v577_data + (v539_data * v575_data));
              float v580_data = s1_w1[111];
              tensorforge::intel_esimd::simd<float, 32> v582_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v582_data + (v539_data * v580_data));
              float v585_data = s1_w1[124];
              tensorforge::intel_esimd::simd<float, 32> v587_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v587_data + (v539_data * v585_data));
              float v590_data = s1_w1[137];
              tensorforge::intel_esimd::simd<float, 32> v592_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v592_data + (v539_data * v590_data));
              float v595_data = s1_w1[150];
              tensorforge::intel_esimd::simd<float, 32> v597_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v597_data + (v539_data * v595_data));
              float v600_data = s1_w1[163];
              tensorforge::intel_esimd::simd<float, 32> v602_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v602_data + (v539_data * v600_data));
              tensorforge::intel_esimd::simd<float, 32> v604_data(r2.template select<32, 1>(256));
              float v605_data = s1_w1[8];
              tensorforge::intel_esimd::simd<float, 32> v607_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v607_data + (v604_data * v605_data));
              float v610_data = s1_w1[21];
              tensorforge::intel_esimd::simd<float, 32> v612_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v612_data + (v604_data * v610_data));
              float v615_data = s1_w1[34];
              tensorforge::intel_esimd::simd<float, 32> v617_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v617_data + (v604_data * v615_data));
              float v620_data = s1_w1[47];
              tensorforge::intel_esimd::simd<float, 32> v622_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v622_data + (v604_data * v620_data));
              float v625_data = s1_w1[60];
              tensorforge::intel_esimd::simd<float, 32> v627_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v627_data + (v604_data * v625_data));
              float v630_data = s1_w1[73];
              tensorforge::intel_esimd::simd<float, 32> v632_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v632_data + (v604_data * v630_data));
              float v635_data = s1_w1[86];
              tensorforge::intel_esimd::simd<float, 32> v637_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v637_data + (v604_data * v635_data));
              float v640_data = s1_w1[99];
              tensorforge::intel_esimd::simd<float, 32> v642_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v642_data + (v604_data * v640_data));
              float v645_data = s1_w1[112];
              tensorforge::intel_esimd::simd<float, 32> v647_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v647_data + (v604_data * v645_data));
              float v650_data = s1_w1[125];
              tensorforge::intel_esimd::simd<float, 32> v652_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v652_data + (v604_data * v650_data));
              float v655_data = s1_w1[138];
              tensorforge::intel_esimd::simd<float, 32> v657_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v657_data + (v604_data * v655_data));
              float v660_data = s1_w1[151];
              tensorforge::intel_esimd::simd<float, 32> v662_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v662_data + (v604_data * v660_data));
              float v665_data = s1_w1[164];
              tensorforge::intel_esimd::simd<float, 32> v667_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v667_data + (v604_data * v665_data));
              tensorforge::intel_esimd::simd<float, 32> v669_data(r2.template select<32, 1>(288));
              float v670_data = s1_w1[9];
              tensorforge::intel_esimd::simd<float, 32> v672_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v672_data + (v669_data * v670_data));
              float v675_data = s1_w1[22];
              tensorforge::intel_esimd::simd<float, 32> v677_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v677_data + (v669_data * v675_data));
              float v680_data = s1_w1[35];
              tensorforge::intel_esimd::simd<float, 32> v682_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v682_data + (v669_data * v680_data));
              float v685_data = s1_w1[48];
              tensorforge::intel_esimd::simd<float, 32> v687_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v687_data + (v669_data * v685_data));
              float v690_data = s1_w1[61];
              tensorforge::intel_esimd::simd<float, 32> v692_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v692_data + (v669_data * v690_data));
              float v695_data = s1_w1[74];
              tensorforge::intel_esimd::simd<float, 32> v697_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v697_data + (v669_data * v695_data));
              float v700_data = s1_w1[87];
              tensorforge::intel_esimd::simd<float, 32> v702_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v702_data + (v669_data * v700_data));
              float v705_data = s1_w1[100];
              tensorforge::intel_esimd::simd<float, 32> v707_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v707_data + (v669_data * v705_data));
              float v710_data = s1_w1[113];
              tensorforge::intel_esimd::simd<float, 32> v712_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v712_data + (v669_data * v710_data));
              float v715_data = s1_w1[126];
              tensorforge::intel_esimd::simd<float, 32> v717_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v717_data + (v669_data * v715_data));
              float v720_data = s1_w1[139];
              tensorforge::intel_esimd::simd<float, 32> v722_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v722_data + (v669_data * v720_data));
              float v725_data = s1_w1[152];
              tensorforge::intel_esimd::simd<float, 32> v727_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v727_data + (v669_data * v725_data));
              float v730_data = s1_w1[165];
              tensorforge::intel_esimd::simd<float, 32> v732_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v732_data + (v669_data * v730_data));
              tensorforge::intel_esimd::simd<float, 32> v734_data(r2.template select<32, 1>(320));
              float v735_data = s1_w1[10];
              tensorforge::intel_esimd::simd<float, 32> v737_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v737_data + (v734_data * v735_data));
              float v740_data = s1_w1[23];
              tensorforge::intel_esimd::simd<float, 32> v742_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v742_data + (v734_data * v740_data));
              float v745_data = s1_w1[36];
              tensorforge::intel_esimd::simd<float, 32> v747_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v747_data + (v734_data * v745_data));
              float v750_data = s1_w1[49];
              tensorforge::intel_esimd::simd<float, 32> v752_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v752_data + (v734_data * v750_data));
              float v755_data = s1_w1[62];
              tensorforge::intel_esimd::simd<float, 32> v757_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v757_data + (v734_data * v755_data));
              float v760_data = s1_w1[75];
              tensorforge::intel_esimd::simd<float, 32> v762_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v762_data + (v734_data * v760_data));
              float v765_data = s1_w1[88];
              tensorforge::intel_esimd::simd<float, 32> v767_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v767_data + (v734_data * v765_data));
              float v770_data = s1_w1[101];
              tensorforge::intel_esimd::simd<float, 32> v772_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v772_data + (v734_data * v770_data));
              float v775_data = s1_w1[114];
              tensorforge::intel_esimd::simd<float, 32> v777_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v777_data + (v734_data * v775_data));
              float v780_data = s1_w1[127];
              tensorforge::intel_esimd::simd<float, 32> v782_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v782_data + (v734_data * v780_data));
              float v785_data = s1_w1[140];
              tensorforge::intel_esimd::simd<float, 32> v787_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v787_data + (v734_data * v785_data));
              float v790_data = s1_w1[153];
              tensorforge::intel_esimd::simd<float, 32> v792_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v792_data + (v734_data * v790_data));
              float v795_data = s1_w1[166];
              tensorforge::intel_esimd::simd<float, 32> v797_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v797_data + (v734_data * v795_data));
              tensorforge::intel_esimd::simd<float, 32> v799_data(r2.template select<32, 1>(352));
              float v800_data = s1_w1[11];
              tensorforge::intel_esimd::simd<float, 32> v802_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v802_data + (v799_data * v800_data));
              float v805_data = s1_w1[24];
              tensorforge::intel_esimd::simd<float, 32> v807_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v807_data + (v799_data * v805_data));
              float v810_data = s1_w1[37];
              tensorforge::intel_esimd::simd<float, 32> v812_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v812_data + (v799_data * v810_data));
              float v815_data = s1_w1[50];
              tensorforge::intel_esimd::simd<float, 32> v817_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v817_data + (v799_data * v815_data));
              float v820_data = s1_w1[63];
              tensorforge::intel_esimd::simd<float, 32> v822_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v822_data + (v799_data * v820_data));
              float v825_data = s1_w1[76];
              tensorforge::intel_esimd::simd<float, 32> v827_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v827_data + (v799_data * v825_data));
              float v830_data = s1_w1[89];
              tensorforge::intel_esimd::simd<float, 32> v832_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v832_data + (v799_data * v830_data));
              float v835_data = s1_w1[102];
              tensorforge::intel_esimd::simd<float, 32> v837_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v837_data + (v799_data * v835_data));
              float v840_data = s1_w1[115];
              tensorforge::intel_esimd::simd<float, 32> v842_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v842_data + (v799_data * v840_data));
              float v845_data = s1_w1[128];
              tensorforge::intel_esimd::simd<float, 32> v847_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v847_data + (v799_data * v845_data));
              float v850_data = s1_w1[141];
              tensorforge::intel_esimd::simd<float, 32> v852_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v852_data + (v799_data * v850_data));
              float v855_data = s1_w1[154];
              tensorforge::intel_esimd::simd<float, 32> v857_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v857_data + (v799_data * v855_data));
              float v860_data = s1_w1[167];
              tensorforge::intel_esimd::simd<float, 32> v862_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v862_data + (v799_data * v860_data));
              tensorforge::intel_esimd::simd<float, 32> v864_data(r2.template select<32, 1>(384));
              float v865_data = s1_w1[12];
              tensorforge::intel_esimd::simd<float, 32> v867_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v867_data + (v864_data * v865_data));
              float v870_data = s1_w1[25];
              tensorforge::intel_esimd::simd<float, 32> v872_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v872_data + (v864_data * v870_data));
              float v875_data = s1_w1[38];
              tensorforge::intel_esimd::simd<float, 32> v877_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v877_data + (v864_data * v875_data));
              float v880_data = s1_w1[51];
              tensorforge::intel_esimd::simd<float, 32> v882_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v882_data + (v864_data * v880_data));
              float v885_data = s1_w1[64];
              tensorforge::intel_esimd::simd<float, 32> v887_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v887_data + (v864_data * v885_data));
              float v890_data = s1_w1[77];
              tensorforge::intel_esimd::simd<float, 32> v892_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v892_data + (v864_data * v890_data));
              float v895_data = s1_w1[90];
              tensorforge::intel_esimd::simd<float, 32> v897_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v897_data + (v864_data * v895_data));
              float v900_data = s1_w1[103];
              tensorforge::intel_esimd::simd<float, 32> v902_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v902_data + (v864_data * v900_data));
              float v905_data = s1_w1[116];
              tensorforge::intel_esimd::simd<float, 32> v907_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v907_data + (v864_data * v905_data));
              float v910_data = s1_w1[129];
              tensorforge::intel_esimd::simd<float, 32> v912_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v912_data + (v864_data * v910_data));
              float v915_data = s1_w1[142];
              tensorforge::intel_esimd::simd<float, 32> v917_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v917_data + (v864_data * v915_data));
              float v920_data = s1_w1[155];
              tensorforge::intel_esimd::simd<float, 32> v922_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v922_data + (v864_data * v920_data));
              float v925_data = s1_w1[168];
              tensorforge::intel_esimd::simd<float, 32> v927_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v927_data + (v864_data * v925_data));
              #pragma unroll
              for (int32_t v929_n0 = 0; v929_n0 < 1; ++v929_n0) {
                int32_t v931_a = v929_n0 * 32;
                #pragma unroll
                for (int32_t v930_n1 = 0; v930_n1 < 13; ++v930_n1) {
                  int32_t v933_a = v931_a + (v930_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v934_data(ir3.template select<32, 1>(v933_a));
                  r3.template select<32, 1>(v933_a) = v934_data;
                }
              }
              // glb_m3 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v935_i0 = 0; v935_i0 < 1; ++v935_i0) {
                int32_t v937_a = v935_i0 * 32;
                #pragma unroll
                for (int32_t v936_i1 = 0; v936_i1 < 13; ++v936_i1) {
                  int32_t v939_a = v937_a + (v936_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v940_data(r3.template select<32, 1>(v939_a));
                  v940_data.copy_to(glb_m3 + (v939_a));
                }
              }
            }
            tensorforge::prefetchL2<416>(&pf_glb_m1[0]);
            tensorforge::prefetchL2<169>(&pf_glb_m2[0]);
            tensorforge::prefetchL2<416>(&pf_glb_m0[0]);
            tensorforge::prefetchL2<169>(&pf_glb_m4[0]);
          }
        }
      }
    });
  });
}


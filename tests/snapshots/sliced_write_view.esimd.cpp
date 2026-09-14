// === base name ===
kernel_136ef179ad88d921

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_136ef179ad88d921 = {{1, 8, 1}, 32, 32, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_136ef179ad88d921(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_136ef179ad88d921(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_136ef179ad88d921(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 8 - 1) / 8;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 1408 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_136ef179ad88d921(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_136ef179ad88d921(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_136ef179ad88d921(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_136ef179ad88d921(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 32;
                #pragma unroll
                for (int32_t v21_i1 = 10; v21_i1 < 13; ++v21_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v26_data;
                  v26_data.copy_from(glb_m1 + ((v22_lead + (v21_i1 * 32))));
                  r0.template select<32, 1>((v22_lead + ((v21_i1 - 10) * 32))) = v26_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v30_ld);
              tensorforge::intel_esimd::simd<float, 32> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 128), v31_ld);
              tensorforge::intel_esimd::simd<float, 9> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 160), v32_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 32> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 32), (0, 1)] [(10, 13)]
              tensorforge::intel_esimd::simd<float, 32> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v35_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> s0_w0 = tensorforge::slmLoad<float, 16>(s0 + 114);
              float v36_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v38_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v38_data + (v35_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v40_data(r0.template select<32, 1>(32));
              float v41_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v43_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v43_data + (v40_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v45_data(r0.template select<32, 1>(64));
              float v46_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v48_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v48_data + (v45_data * v46_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v50_n0 = 0; v50_n0 < 1; ++v50_n0) {
                int32_t v52_a = v50_n0 * 32;
                #pragma unroll
                for (int32_t v51_n1 = 0; v51_n1 < 1; ++v51_n1) {
                  int32_t v54_a = v52_a + (v51_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v55_data(ir1.template select<32, 1>(v54_a));
                  r1.template select<32, 1>(v54_a) = v55_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v56_i0 = 0; v56_i0 < 1; ++v56_i0) {
                int32_t v58_a = v56_i0 * 32;
                #pragma unroll
                for (int32_t v57_i1 = 0; v57_i1 < 1; ++v57_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v61_data(r1.template select<32, 1>((v58_a + (v57_i1 * 32))));
                  v61_data.copy_to(glb_m0 + ((v58_a + ((v57_i1 + 8) * 32))));
                }
              }
              tensorforge::intel_esimd::simd<float, 416> r2(0.0f);
              // r2 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v67_i0 = 0; v67_i0 < 1; ++v67_i0) {
                int32_t v69_lead = v67_i0 * 32;
                #pragma unroll
                for (int32_t v68_i1 = 0; v68_i1 < 13; ++v68_i1) {
                  int32_t v72_a = v69_lead + (v68_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v73_data;
                  v73_data.copy_from(glb_m0 + (v72_a));
                  r2.template select<32, 1>(v72_a) = v73_data;
                }
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v75_ld;
              v75_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s1 + (0 + 0 + 4 * 0 + 0), v75_ld);
              tensorforge::intel_esimd::simd<float, 32> v76_ld;
              v76_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 1 * 0 + 128), v76_ld);
              tensorforge::intel_esimd::simd<float, 9> v77_ld;
              v77_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s1 + (0 + 0 + 1 * 0 + 160), v77_ld);
              // wait(r2 = load{g>r}(glb_m0););
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 416> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 32), (0, 13)] [(0, 13)]
              tensorforge::intel_esimd::simd<float, 416> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v80_data(r2.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 176> s1_w1 = tensorforge::slmLoad<float, 176>(s1 + 0);
              float v81_data = s1_w1[0];
              tensorforge::intel_esimd::simd<float, 32> v83_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v83_data + (v80_data * v81_data));
              float v86_data = s1_w1[13];
              tensorforge::intel_esimd::simd<float, 32> v88_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v88_data + (v80_data * v86_data));
              float v91_data = s1_w1[26];
              tensorforge::intel_esimd::simd<float, 32> v93_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v93_data + (v80_data * v91_data));
              float v96_data = s1_w1[39];
              tensorforge::intel_esimd::simd<float, 32> v98_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v98_data + (v80_data * v96_data));
              float v101_data = s1_w1[52];
              tensorforge::intel_esimd::simd<float, 32> v103_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v103_data + (v80_data * v101_data));
              float v106_data = s1_w1[65];
              tensorforge::intel_esimd::simd<float, 32> v108_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v108_data + (v80_data * v106_data));
              float v111_data = s1_w1[78];
              tensorforge::intel_esimd::simd<float, 32> v113_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v113_data + (v80_data * v111_data));
              float v116_data = s1_w1[91];
              tensorforge::intel_esimd::simd<float, 32> v118_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v118_data + (v80_data * v116_data));
              float v121_data = s1_w1[104];
              tensorforge::intel_esimd::simd<float, 32> v123_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v123_data + (v80_data * v121_data));
              float v126_data = s1_w1[117];
              tensorforge::intel_esimd::simd<float, 32> v128_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v128_data + (v80_data * v126_data));
              float v131_data = s1_w1[130];
              tensorforge::intel_esimd::simd<float, 32> v133_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v133_data + (v80_data * v131_data));
              float v136_data = s1_w1[143];
              tensorforge::intel_esimd::simd<float, 32> v138_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v138_data + (v80_data * v136_data));
              float v141_data = s1_w1[156];
              tensorforge::intel_esimd::simd<float, 32> v143_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v143_data + (v80_data * v141_data));
              tensorforge::intel_esimd::simd<float, 32> v145_data(r2.template select<32, 1>(32));
              float v146_data = s1_w1[1];
              tensorforge::intel_esimd::simd<float, 32> v148_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v148_data + (v145_data * v146_data));
              float v151_data = s1_w1[14];
              tensorforge::intel_esimd::simd<float, 32> v153_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v153_data + (v145_data * v151_data));
              float v156_data = s1_w1[27];
              tensorforge::intel_esimd::simd<float, 32> v158_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v158_data + (v145_data * v156_data));
              float v161_data = s1_w1[40];
              tensorforge::intel_esimd::simd<float, 32> v163_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v163_data + (v145_data * v161_data));
              float v166_data = s1_w1[53];
              tensorforge::intel_esimd::simd<float, 32> v168_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v168_data + (v145_data * v166_data));
              float v171_data = s1_w1[66];
              tensorforge::intel_esimd::simd<float, 32> v173_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v173_data + (v145_data * v171_data));
              float v176_data = s1_w1[79];
              tensorforge::intel_esimd::simd<float, 32> v178_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v178_data + (v145_data * v176_data));
              float v181_data = s1_w1[92];
              tensorforge::intel_esimd::simd<float, 32> v183_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v183_data + (v145_data * v181_data));
              float v186_data = s1_w1[105];
              tensorforge::intel_esimd::simd<float, 32> v188_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v188_data + (v145_data * v186_data));
              float v191_data = s1_w1[118];
              tensorforge::intel_esimd::simd<float, 32> v193_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v193_data + (v145_data * v191_data));
              float v196_data = s1_w1[131];
              tensorforge::intel_esimd::simd<float, 32> v198_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v198_data + (v145_data * v196_data));
              float v201_data = s1_w1[144];
              tensorforge::intel_esimd::simd<float, 32> v203_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v203_data + (v145_data * v201_data));
              float v206_data = s1_w1[157];
              tensorforge::intel_esimd::simd<float, 32> v208_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v208_data + (v145_data * v206_data));
              tensorforge::intel_esimd::simd<float, 32> v210_data(r2.template select<32, 1>(64));
              float v211_data = s1_w1[2];
              tensorforge::intel_esimd::simd<float, 32> v213_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v213_data + (v210_data * v211_data));
              float v216_data = s1_w1[15];
              tensorforge::intel_esimd::simd<float, 32> v218_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v218_data + (v210_data * v216_data));
              float v221_data = s1_w1[28];
              tensorforge::intel_esimd::simd<float, 32> v223_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v223_data + (v210_data * v221_data));
              float v226_data = s1_w1[41];
              tensorforge::intel_esimd::simd<float, 32> v228_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v228_data + (v210_data * v226_data));
              float v231_data = s1_w1[54];
              tensorforge::intel_esimd::simd<float, 32> v233_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v233_data + (v210_data * v231_data));
              float v236_data = s1_w1[67];
              tensorforge::intel_esimd::simd<float, 32> v238_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v238_data + (v210_data * v236_data));
              float v241_data = s1_w1[80];
              tensorforge::intel_esimd::simd<float, 32> v243_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v243_data + (v210_data * v241_data));
              float v246_data = s1_w1[93];
              tensorforge::intel_esimd::simd<float, 32> v248_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v248_data + (v210_data * v246_data));
              float v251_data = s1_w1[106];
              tensorforge::intel_esimd::simd<float, 32> v253_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v253_data + (v210_data * v251_data));
              float v256_data = s1_w1[119];
              tensorforge::intel_esimd::simd<float, 32> v258_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v258_data + (v210_data * v256_data));
              float v261_data = s1_w1[132];
              tensorforge::intel_esimd::simd<float, 32> v263_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v263_data + (v210_data * v261_data));
              float v266_data = s1_w1[145];
              tensorforge::intel_esimd::simd<float, 32> v268_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v268_data + (v210_data * v266_data));
              float v271_data = s1_w1[158];
              tensorforge::intel_esimd::simd<float, 32> v273_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v273_data + (v210_data * v271_data));
              tensorforge::intel_esimd::simd<float, 32> v275_data(r2.template select<32, 1>(96));
              float v276_data = s1_w1[3];
              tensorforge::intel_esimd::simd<float, 32> v278_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v278_data + (v275_data * v276_data));
              float v281_data = s1_w1[16];
              tensorforge::intel_esimd::simd<float, 32> v283_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v283_data + (v275_data * v281_data));
              float v286_data = s1_w1[29];
              tensorforge::intel_esimd::simd<float, 32> v288_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v288_data + (v275_data * v286_data));
              float v291_data = s1_w1[42];
              tensorforge::intel_esimd::simd<float, 32> v293_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v293_data + (v275_data * v291_data));
              float v296_data = s1_w1[55];
              tensorforge::intel_esimd::simd<float, 32> v298_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v298_data + (v275_data * v296_data));
              float v301_data = s1_w1[68];
              tensorforge::intel_esimd::simd<float, 32> v303_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v303_data + (v275_data * v301_data));
              float v306_data = s1_w1[81];
              tensorforge::intel_esimd::simd<float, 32> v308_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v308_data + (v275_data * v306_data));
              float v311_data = s1_w1[94];
              tensorforge::intel_esimd::simd<float, 32> v313_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v313_data + (v275_data * v311_data));
              float v316_data = s1_w1[107];
              tensorforge::intel_esimd::simd<float, 32> v318_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v318_data + (v275_data * v316_data));
              float v321_data = s1_w1[120];
              tensorforge::intel_esimd::simd<float, 32> v323_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v323_data + (v275_data * v321_data));
              float v326_data = s1_w1[133];
              tensorforge::intel_esimd::simd<float, 32> v328_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v328_data + (v275_data * v326_data));
              float v331_data = s1_w1[146];
              tensorforge::intel_esimd::simd<float, 32> v333_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v333_data + (v275_data * v331_data));
              float v336_data = s1_w1[159];
              tensorforge::intel_esimd::simd<float, 32> v338_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v338_data + (v275_data * v336_data));
              tensorforge::intel_esimd::simd<float, 32> v340_data(r2.template select<32, 1>(128));
              float v341_data = s1_w1[4];
              tensorforge::intel_esimd::simd<float, 32> v343_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v343_data + (v340_data * v341_data));
              float v346_data = s1_w1[17];
              tensorforge::intel_esimd::simd<float, 32> v348_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v348_data + (v340_data * v346_data));
              float v351_data = s1_w1[30];
              tensorforge::intel_esimd::simd<float, 32> v353_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v353_data + (v340_data * v351_data));
              float v356_data = s1_w1[43];
              tensorforge::intel_esimd::simd<float, 32> v358_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v358_data + (v340_data * v356_data));
              float v361_data = s1_w1[56];
              tensorforge::intel_esimd::simd<float, 32> v363_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v363_data + (v340_data * v361_data));
              float v366_data = s1_w1[69];
              tensorforge::intel_esimd::simd<float, 32> v368_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v368_data + (v340_data * v366_data));
              float v371_data = s1_w1[82];
              tensorforge::intel_esimd::simd<float, 32> v373_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v373_data + (v340_data * v371_data));
              float v376_data = s1_w1[95];
              tensorforge::intel_esimd::simd<float, 32> v378_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v378_data + (v340_data * v376_data));
              float v381_data = s1_w1[108];
              tensorforge::intel_esimd::simd<float, 32> v383_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v383_data + (v340_data * v381_data));
              float v386_data = s1_w1[121];
              tensorforge::intel_esimd::simd<float, 32> v388_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v388_data + (v340_data * v386_data));
              float v391_data = s1_w1[134];
              tensorforge::intel_esimd::simd<float, 32> v393_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v393_data + (v340_data * v391_data));
              float v396_data = s1_w1[147];
              tensorforge::intel_esimd::simd<float, 32> v398_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v398_data + (v340_data * v396_data));
              float v401_data = s1_w1[160];
              tensorforge::intel_esimd::simd<float, 32> v403_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v403_data + (v340_data * v401_data));
              tensorforge::intel_esimd::simd<float, 32> v405_data(r2.template select<32, 1>(160));
              float v406_data = s1_w1[5];
              tensorforge::intel_esimd::simd<float, 32> v408_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v408_data + (v405_data * v406_data));
              float v411_data = s1_w1[18];
              tensorforge::intel_esimd::simd<float, 32> v413_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v413_data + (v405_data * v411_data));
              float v416_data = s1_w1[31];
              tensorforge::intel_esimd::simd<float, 32> v418_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v418_data + (v405_data * v416_data));
              float v421_data = s1_w1[44];
              tensorforge::intel_esimd::simd<float, 32> v423_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v423_data + (v405_data * v421_data));
              float v426_data = s1_w1[57];
              tensorforge::intel_esimd::simd<float, 32> v428_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v428_data + (v405_data * v426_data));
              float v431_data = s1_w1[70];
              tensorforge::intel_esimd::simd<float, 32> v433_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v433_data + (v405_data * v431_data));
              float v436_data = s1_w1[83];
              tensorforge::intel_esimd::simd<float, 32> v438_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v438_data + (v405_data * v436_data));
              float v441_data = s1_w1[96];
              tensorforge::intel_esimd::simd<float, 32> v443_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v443_data + (v405_data * v441_data));
              float v446_data = s1_w1[109];
              tensorforge::intel_esimd::simd<float, 32> v448_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v448_data + (v405_data * v446_data));
              float v451_data = s1_w1[122];
              tensorforge::intel_esimd::simd<float, 32> v453_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v453_data + (v405_data * v451_data));
              float v456_data = s1_w1[135];
              tensorforge::intel_esimd::simd<float, 32> v458_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v458_data + (v405_data * v456_data));
              float v461_data = s1_w1[148];
              tensorforge::intel_esimd::simd<float, 32> v463_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v463_data + (v405_data * v461_data));
              float v466_data = s1_w1[161];
              tensorforge::intel_esimd::simd<float, 32> v468_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v468_data + (v405_data * v466_data));
              tensorforge::intel_esimd::simd<float, 32> v470_data(r2.template select<32, 1>(192));
              float v471_data = s1_w1[6];
              tensorforge::intel_esimd::simd<float, 32> v473_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v473_data + (v470_data * v471_data));
              float v476_data = s1_w1[19];
              tensorforge::intel_esimd::simd<float, 32> v478_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v478_data + (v470_data * v476_data));
              float v481_data = s1_w1[32];
              tensorforge::intel_esimd::simd<float, 32> v483_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v483_data + (v470_data * v481_data));
              float v486_data = s1_w1[45];
              tensorforge::intel_esimd::simd<float, 32> v488_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v488_data + (v470_data * v486_data));
              float v491_data = s1_w1[58];
              tensorforge::intel_esimd::simd<float, 32> v493_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v493_data + (v470_data * v491_data));
              float v496_data = s1_w1[71];
              tensorforge::intel_esimd::simd<float, 32> v498_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v498_data + (v470_data * v496_data));
              float v501_data = s1_w1[84];
              tensorforge::intel_esimd::simd<float, 32> v503_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v503_data + (v470_data * v501_data));
              float v506_data = s1_w1[97];
              tensorforge::intel_esimd::simd<float, 32> v508_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v508_data + (v470_data * v506_data));
              float v511_data = s1_w1[110];
              tensorforge::intel_esimd::simd<float, 32> v513_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v513_data + (v470_data * v511_data));
              float v516_data = s1_w1[123];
              tensorforge::intel_esimd::simd<float, 32> v518_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v518_data + (v470_data * v516_data));
              float v521_data = s1_w1[136];
              tensorforge::intel_esimd::simd<float, 32> v523_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v523_data + (v470_data * v521_data));
              float v526_data = s1_w1[149];
              tensorforge::intel_esimd::simd<float, 32> v528_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v528_data + (v470_data * v526_data));
              float v531_data = s1_w1[162];
              tensorforge::intel_esimd::simd<float, 32> v533_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v533_data + (v470_data * v531_data));
              tensorforge::intel_esimd::simd<float, 32> v535_data(r2.template select<32, 1>(224));
              float v536_data = s1_w1[7];
              tensorforge::intel_esimd::simd<float, 32> v538_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v538_data + (v535_data * v536_data));
              float v541_data = s1_w1[20];
              tensorforge::intel_esimd::simd<float, 32> v543_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v543_data + (v535_data * v541_data));
              float v546_data = s1_w1[33];
              tensorforge::intel_esimd::simd<float, 32> v548_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v548_data + (v535_data * v546_data));
              float v551_data = s1_w1[46];
              tensorforge::intel_esimd::simd<float, 32> v553_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v553_data + (v535_data * v551_data));
              float v556_data = s1_w1[59];
              tensorforge::intel_esimd::simd<float, 32> v558_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v558_data + (v535_data * v556_data));
              float v561_data = s1_w1[72];
              tensorforge::intel_esimd::simd<float, 32> v563_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v563_data + (v535_data * v561_data));
              float v566_data = s1_w1[85];
              tensorforge::intel_esimd::simd<float, 32> v568_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v568_data + (v535_data * v566_data));
              float v571_data = s1_w1[98];
              tensorforge::intel_esimd::simd<float, 32> v573_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v573_data + (v535_data * v571_data));
              float v576_data = s1_w1[111];
              tensorforge::intel_esimd::simd<float, 32> v578_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v578_data + (v535_data * v576_data));
              float v581_data = s1_w1[124];
              tensorforge::intel_esimd::simd<float, 32> v583_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v583_data + (v535_data * v581_data));
              float v586_data = s1_w1[137];
              tensorforge::intel_esimd::simd<float, 32> v588_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v588_data + (v535_data * v586_data));
              float v591_data = s1_w1[150];
              tensorforge::intel_esimd::simd<float, 32> v593_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v593_data + (v535_data * v591_data));
              float v596_data = s1_w1[163];
              tensorforge::intel_esimd::simd<float, 32> v598_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v598_data + (v535_data * v596_data));
              tensorforge::intel_esimd::simd<float, 32> v600_data(r2.template select<32, 1>(256));
              float v601_data = s1_w1[8];
              tensorforge::intel_esimd::simd<float, 32> v603_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v603_data + (v600_data * v601_data));
              float v606_data = s1_w1[21];
              tensorforge::intel_esimd::simd<float, 32> v608_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v608_data + (v600_data * v606_data));
              float v611_data = s1_w1[34];
              tensorforge::intel_esimd::simd<float, 32> v613_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v613_data + (v600_data * v611_data));
              float v616_data = s1_w1[47];
              tensorforge::intel_esimd::simd<float, 32> v618_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v618_data + (v600_data * v616_data));
              float v621_data = s1_w1[60];
              tensorforge::intel_esimd::simd<float, 32> v623_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v623_data + (v600_data * v621_data));
              float v626_data = s1_w1[73];
              tensorforge::intel_esimd::simd<float, 32> v628_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v628_data + (v600_data * v626_data));
              float v631_data = s1_w1[86];
              tensorforge::intel_esimd::simd<float, 32> v633_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v633_data + (v600_data * v631_data));
              float v636_data = s1_w1[99];
              tensorforge::intel_esimd::simd<float, 32> v638_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v638_data + (v600_data * v636_data));
              float v641_data = s1_w1[112];
              tensorforge::intel_esimd::simd<float, 32> v643_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v643_data + (v600_data * v641_data));
              float v646_data = s1_w1[125];
              tensorforge::intel_esimd::simd<float, 32> v648_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v648_data + (v600_data * v646_data));
              float v651_data = s1_w1[138];
              tensorforge::intel_esimd::simd<float, 32> v653_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v653_data + (v600_data * v651_data));
              float v656_data = s1_w1[151];
              tensorforge::intel_esimd::simd<float, 32> v658_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v658_data + (v600_data * v656_data));
              float v661_data = s1_w1[164];
              tensorforge::intel_esimd::simd<float, 32> v663_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v663_data + (v600_data * v661_data));
              tensorforge::intel_esimd::simd<float, 32> v665_data(r2.template select<32, 1>(288));
              float v666_data = s1_w1[9];
              tensorforge::intel_esimd::simd<float, 32> v668_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v668_data + (v665_data * v666_data));
              float v671_data = s1_w1[22];
              tensorforge::intel_esimd::simd<float, 32> v673_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v673_data + (v665_data * v671_data));
              float v676_data = s1_w1[35];
              tensorforge::intel_esimd::simd<float, 32> v678_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v678_data + (v665_data * v676_data));
              float v681_data = s1_w1[48];
              tensorforge::intel_esimd::simd<float, 32> v683_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v683_data + (v665_data * v681_data));
              float v686_data = s1_w1[61];
              tensorforge::intel_esimd::simd<float, 32> v688_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v688_data + (v665_data * v686_data));
              float v691_data = s1_w1[74];
              tensorforge::intel_esimd::simd<float, 32> v693_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v693_data + (v665_data * v691_data));
              float v696_data = s1_w1[87];
              tensorforge::intel_esimd::simd<float, 32> v698_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v698_data + (v665_data * v696_data));
              float v701_data = s1_w1[100];
              tensorforge::intel_esimd::simd<float, 32> v703_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v703_data + (v665_data * v701_data));
              float v706_data = s1_w1[113];
              tensorforge::intel_esimd::simd<float, 32> v708_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v708_data + (v665_data * v706_data));
              float v711_data = s1_w1[126];
              tensorforge::intel_esimd::simd<float, 32> v713_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v713_data + (v665_data * v711_data));
              float v716_data = s1_w1[139];
              tensorforge::intel_esimd::simd<float, 32> v718_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v718_data + (v665_data * v716_data));
              float v721_data = s1_w1[152];
              tensorforge::intel_esimd::simd<float, 32> v723_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v723_data + (v665_data * v721_data));
              float v726_data = s1_w1[165];
              tensorforge::intel_esimd::simd<float, 32> v728_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v728_data + (v665_data * v726_data));
              tensorforge::intel_esimd::simd<float, 32> v730_data(r2.template select<32, 1>(320));
              float v731_data = s1_w1[10];
              tensorforge::intel_esimd::simd<float, 32> v733_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v733_data + (v730_data * v731_data));
              float v736_data = s1_w1[23];
              tensorforge::intel_esimd::simd<float, 32> v738_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v738_data + (v730_data * v736_data));
              float v741_data = s1_w1[36];
              tensorforge::intel_esimd::simd<float, 32> v743_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v743_data + (v730_data * v741_data));
              float v746_data = s1_w1[49];
              tensorforge::intel_esimd::simd<float, 32> v748_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v748_data + (v730_data * v746_data));
              float v751_data = s1_w1[62];
              tensorforge::intel_esimd::simd<float, 32> v753_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v753_data + (v730_data * v751_data));
              float v756_data = s1_w1[75];
              tensorforge::intel_esimd::simd<float, 32> v758_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v758_data + (v730_data * v756_data));
              float v761_data = s1_w1[88];
              tensorforge::intel_esimd::simd<float, 32> v763_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v763_data + (v730_data * v761_data));
              float v766_data = s1_w1[101];
              tensorforge::intel_esimd::simd<float, 32> v768_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v768_data + (v730_data * v766_data));
              float v771_data = s1_w1[114];
              tensorforge::intel_esimd::simd<float, 32> v773_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v773_data + (v730_data * v771_data));
              float v776_data = s1_w1[127];
              tensorforge::intel_esimd::simd<float, 32> v778_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v778_data + (v730_data * v776_data));
              float v781_data = s1_w1[140];
              tensorforge::intel_esimd::simd<float, 32> v783_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v783_data + (v730_data * v781_data));
              float v786_data = s1_w1[153];
              tensorforge::intel_esimd::simd<float, 32> v788_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v788_data + (v730_data * v786_data));
              float v791_data = s1_w1[166];
              tensorforge::intel_esimd::simd<float, 32> v793_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v793_data + (v730_data * v791_data));
              tensorforge::intel_esimd::simd<float, 32> v795_data(r2.template select<32, 1>(352));
              float v796_data = s1_w1[11];
              tensorforge::intel_esimd::simd<float, 32> v798_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v798_data + (v795_data * v796_data));
              float v801_data = s1_w1[24];
              tensorforge::intel_esimd::simd<float, 32> v803_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v803_data + (v795_data * v801_data));
              float v806_data = s1_w1[37];
              tensorforge::intel_esimd::simd<float, 32> v808_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v808_data + (v795_data * v806_data));
              float v811_data = s1_w1[50];
              tensorforge::intel_esimd::simd<float, 32> v813_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v813_data + (v795_data * v811_data));
              float v816_data = s1_w1[63];
              tensorforge::intel_esimd::simd<float, 32> v818_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v818_data + (v795_data * v816_data));
              float v821_data = s1_w1[76];
              tensorforge::intel_esimd::simd<float, 32> v823_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v823_data + (v795_data * v821_data));
              float v826_data = s1_w1[89];
              tensorforge::intel_esimd::simd<float, 32> v828_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v828_data + (v795_data * v826_data));
              float v831_data = s1_w1[102];
              tensorforge::intel_esimd::simd<float, 32> v833_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v833_data + (v795_data * v831_data));
              float v836_data = s1_w1[115];
              tensorforge::intel_esimd::simd<float, 32> v838_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v838_data + (v795_data * v836_data));
              float v841_data = s1_w1[128];
              tensorforge::intel_esimd::simd<float, 32> v843_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v843_data + (v795_data * v841_data));
              float v846_data = s1_w1[141];
              tensorforge::intel_esimd::simd<float, 32> v848_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v848_data + (v795_data * v846_data));
              float v851_data = s1_w1[154];
              tensorforge::intel_esimd::simd<float, 32> v853_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v853_data + (v795_data * v851_data));
              float v856_data = s1_w1[167];
              tensorforge::intel_esimd::simd<float, 32> v858_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v858_data + (v795_data * v856_data));
              tensorforge::intel_esimd::simd<float, 32> v860_data(r2.template select<32, 1>(384));
              float v861_data = s1_w1[12];
              tensorforge::intel_esimd::simd<float, 32> v863_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v863_data + (v860_data * v861_data));
              float v866_data = s1_w1[25];
              tensorforge::intel_esimd::simd<float, 32> v868_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v868_data + (v860_data * v866_data));
              float v871_data = s1_w1[38];
              tensorforge::intel_esimd::simd<float, 32> v873_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v873_data + (v860_data * v871_data));
              float v876_data = s1_w1[51];
              tensorforge::intel_esimd::simd<float, 32> v878_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v878_data + (v860_data * v876_data));
              float v881_data = s1_w1[64];
              tensorforge::intel_esimd::simd<float, 32> v883_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v883_data + (v860_data * v881_data));
              float v886_data = s1_w1[77];
              tensorforge::intel_esimd::simd<float, 32> v888_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v888_data + (v860_data * v886_data));
              float v891_data = s1_w1[90];
              tensorforge::intel_esimd::simd<float, 32> v893_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v893_data + (v860_data * v891_data));
              float v896_data = s1_w1[103];
              tensorforge::intel_esimd::simd<float, 32> v898_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v898_data + (v860_data * v896_data));
              float v901_data = s1_w1[116];
              tensorforge::intel_esimd::simd<float, 32> v903_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v903_data + (v860_data * v901_data));
              float v906_data = s1_w1[129];
              tensorforge::intel_esimd::simd<float, 32> v908_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v908_data + (v860_data * v906_data));
              float v911_data = s1_w1[142];
              tensorforge::intel_esimd::simd<float, 32> v913_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v913_data + (v860_data * v911_data));
              float v916_data = s1_w1[155];
              tensorforge::intel_esimd::simd<float, 32> v918_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v918_data + (v860_data * v916_data));
              float v921_data = s1_w1[168];
              tensorforge::intel_esimd::simd<float, 32> v923_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v923_data + (v860_data * v921_data));
              // r3 = ir3
              #pragma unroll
              for (int32_t v925_n0 = 0; v925_n0 < 1; ++v925_n0) {
                int32_t v927_a = v925_n0 * 32;
                #pragma unroll
                for (int32_t v926_n1 = 0; v926_n1 < 13; ++v926_n1) {
                  int32_t v929_a = v927_a + (v926_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v930_data(ir3.template select<32, 1>(v929_a));
                  r3.template select<32, 1>(v929_a) = v930_data;
                }
              }
              // glb_m3 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v931_i0 = 0; v931_i0 < 1; ++v931_i0) {
                int32_t v933_a = v931_i0 * 32;
                #pragma unroll
                for (int32_t v932_i1 = 0; v932_i1 < 13; ++v932_i1) {
                  int32_t v935_a = v933_a + (v932_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v936_data(r3.template select<32, 1>(v935_a));
                  v936_data.copy_to(glb_m3 + (v935_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


// === base name ===
kernel_28c34dd53c9d3758

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_28c34dd53c9d3758 = {{1, 8, 1}, 32, 32, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_28c34dd53c9d3758(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_28c34dd53c9d3758(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_28c34dd53c9d3758(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_28c34dd53c9d3758(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_28c34dd53c9d3758(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_28c34dd53c9d3758(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_28c34dd53c9d3758(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1408 * sizeof(float)>(); {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,0],[13,1]],"is_tmp":false,"name":"m2","offset":[0,8],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 169 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 169 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 96> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                int32_t v25_lead = v23_i0 * 32;
                #pragma unroll
                for (int32_t v24_i1 = 10; v24_i1 < 13; ++v24_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v29_data;
                  v29_data.copy_from(glb_m1 + ((v25_lead + (v24_i1 * 32))));
                  r0.template select<32, 1>((v25_lead + ((v24_i1 - 10) * 32))) = v29_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v33_ld);
              tensorforge::intel_esimd::simd<float, 32> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 128), v34_ld);
              tensorforge::intel_esimd::simd<float, 9> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 160), v35_ld);
              tensorforge::intel_esimd::simd<float, 32> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 32), (0, 1)] [(10, 13)]
              tensorforge::intel_esimd::simd<float, 32> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v38_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> s0_w0 = tensorforge::slmLoad<float, 16>(s0 + 114);
              float v39_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v41_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v41_data + (v38_data * v39_data));
              tensorforge::intel_esimd::simd<float, 32> v43_data(r0.template select<32, 1>(32));
              float v44_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v46_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v46_data + (v43_data * v44_data));
              tensorforge::intel_esimd::simd<float, 32> v48_data(r0.template select<32, 1>(64));
              float v49_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v51_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v51_data + (v48_data * v49_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v53_n0 = 0; v53_n0 < 1; ++v53_n0) {
                int32_t v55_a = v53_n0 * 32;
                #pragma unroll
                for (int32_t v54_n1 = 0; v54_n1 < 1; ++v54_n1) {
                  int32_t v57_a = v55_a + (v54_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v58_data(ir1.template select<32, 1>(v57_a));
                  r1.template select<32, 1>(v57_a) = v58_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v59_i0 = 0; v59_i0 < 1; ++v59_i0) {
                int32_t v61_a = v59_i0 * 32;
                #pragma unroll
                for (int32_t v60_i1 = 0; v60_i1 < 1; ++v60_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v64_data(r1.template select<32, 1>((v61_a + (v60_i1 * 32))));
                  v64_data.copy_to(glb_m0 + ((v61_a + ((v60_i1 + 8) * 32))));
                }
              }
              tensorforge::intel_esimd::simd<float, 416> r2(0.0f);
              // r2 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v70_i0 = 0; v70_i0 < 1; ++v70_i0) {
                int32_t v72_lead = v70_i0 * 32;
                #pragma unroll
                for (int32_t v71_i1 = 0; v71_i1 < 13; ++v71_i1) {
                  int32_t v75_a = v72_lead + (v71_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v76_data;
                  v76_data.copy_from(glb_m0 + (v75_a));
                  r2.template select<32, 1>(v75_a) = v76_data;
                }
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v78_ld;
              v78_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s1 + (0 + 0 + 4 * 0 + 0), v78_ld);
              tensorforge::intel_esimd::simd<float, 32> v79_ld;
              v79_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 1 * 0 + 128), v79_ld);
              tensorforge::intel_esimd::simd<float, 9> v80_ld;
              v80_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s1 + (0 + 0 + 1 * 0 + 160), v80_ld);
              tensorforge::intel_esimd::simd<float, 416> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 32), (0, 13)] [(0, 13)]
              tensorforge::intel_esimd::simd<float, 416> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v83_data(r2.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 176> s1_w1 = tensorforge::slmLoad<float, 176>(s1 + 0);
              float v84_data = s1_w1[0];
              tensorforge::intel_esimd::simd<float, 32> v86_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v86_data + (v83_data * v84_data));
              float v89_data = s1_w1[13];
              tensorforge::intel_esimd::simd<float, 32> v91_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v91_data + (v83_data * v89_data));
              float v94_data = s1_w1[26];
              tensorforge::intel_esimd::simd<float, 32> v96_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v96_data + (v83_data * v94_data));
              float v99_data = s1_w1[39];
              tensorforge::intel_esimd::simd<float, 32> v101_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v101_data + (v83_data * v99_data));
              float v104_data = s1_w1[52];
              tensorforge::intel_esimd::simd<float, 32> v106_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v106_data + (v83_data * v104_data));
              float v109_data = s1_w1[65];
              tensorforge::intel_esimd::simd<float, 32> v111_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v111_data + (v83_data * v109_data));
              float v114_data = s1_w1[78];
              tensorforge::intel_esimd::simd<float, 32> v116_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v116_data + (v83_data * v114_data));
              float v119_data = s1_w1[91];
              tensorforge::intel_esimd::simd<float, 32> v121_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v121_data + (v83_data * v119_data));
              float v124_data = s1_w1[104];
              tensorforge::intel_esimd::simd<float, 32> v126_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v126_data + (v83_data * v124_data));
              float v129_data = s1_w1[117];
              tensorforge::intel_esimd::simd<float, 32> v131_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v131_data + (v83_data * v129_data));
              float v134_data = s1_w1[130];
              tensorforge::intel_esimd::simd<float, 32> v136_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v136_data + (v83_data * v134_data));
              float v139_data = s1_w1[143];
              tensorforge::intel_esimd::simd<float, 32> v141_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v141_data + (v83_data * v139_data));
              float v144_data = s1_w1[156];
              tensorforge::intel_esimd::simd<float, 32> v146_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v146_data + (v83_data * v144_data));
              tensorforge::intel_esimd::simd<float, 32> v148_data(r2.template select<32, 1>(32));
              float v149_data = s1_w1[1];
              tensorforge::intel_esimd::simd<float, 32> v151_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v151_data + (v148_data * v149_data));
              float v154_data = s1_w1[14];
              tensorforge::intel_esimd::simd<float, 32> v156_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v156_data + (v148_data * v154_data));
              float v159_data = s1_w1[27];
              tensorforge::intel_esimd::simd<float, 32> v161_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v161_data + (v148_data * v159_data));
              float v164_data = s1_w1[40];
              tensorforge::intel_esimd::simd<float, 32> v166_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v166_data + (v148_data * v164_data));
              float v169_data = s1_w1[53];
              tensorforge::intel_esimd::simd<float, 32> v171_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v171_data + (v148_data * v169_data));
              float v174_data = s1_w1[66];
              tensorforge::intel_esimd::simd<float, 32> v176_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v176_data + (v148_data * v174_data));
              float v179_data = s1_w1[79];
              tensorforge::intel_esimd::simd<float, 32> v181_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v181_data + (v148_data * v179_data));
              float v184_data = s1_w1[92];
              tensorforge::intel_esimd::simd<float, 32> v186_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v186_data + (v148_data * v184_data));
              float v189_data = s1_w1[105];
              tensorforge::intel_esimd::simd<float, 32> v191_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v191_data + (v148_data * v189_data));
              float v194_data = s1_w1[118];
              tensorforge::intel_esimd::simd<float, 32> v196_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v196_data + (v148_data * v194_data));
              float v199_data = s1_w1[131];
              tensorforge::intel_esimd::simd<float, 32> v201_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v201_data + (v148_data * v199_data));
              float v204_data = s1_w1[144];
              tensorforge::intel_esimd::simd<float, 32> v206_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v206_data + (v148_data * v204_data));
              float v209_data = s1_w1[157];
              tensorforge::intel_esimd::simd<float, 32> v211_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v211_data + (v148_data * v209_data));
              tensorforge::intel_esimd::simd<float, 32> v213_data(r2.template select<32, 1>(64));
              float v214_data = s1_w1[2];
              tensorforge::intel_esimd::simd<float, 32> v216_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v216_data + (v213_data * v214_data));
              float v219_data = s1_w1[15];
              tensorforge::intel_esimd::simd<float, 32> v221_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v221_data + (v213_data * v219_data));
              float v224_data = s1_w1[28];
              tensorforge::intel_esimd::simd<float, 32> v226_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v226_data + (v213_data * v224_data));
              float v229_data = s1_w1[41];
              tensorforge::intel_esimd::simd<float, 32> v231_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v231_data + (v213_data * v229_data));
              float v234_data = s1_w1[54];
              tensorforge::intel_esimd::simd<float, 32> v236_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v236_data + (v213_data * v234_data));
              float v239_data = s1_w1[67];
              tensorforge::intel_esimd::simd<float, 32> v241_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v241_data + (v213_data * v239_data));
              float v244_data = s1_w1[80];
              tensorforge::intel_esimd::simd<float, 32> v246_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v246_data + (v213_data * v244_data));
              float v249_data = s1_w1[93];
              tensorforge::intel_esimd::simd<float, 32> v251_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v251_data + (v213_data * v249_data));
              float v254_data = s1_w1[106];
              tensorforge::intel_esimd::simd<float, 32> v256_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v256_data + (v213_data * v254_data));
              float v259_data = s1_w1[119];
              tensorforge::intel_esimd::simd<float, 32> v261_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v261_data + (v213_data * v259_data));
              float v264_data = s1_w1[132];
              tensorforge::intel_esimd::simd<float, 32> v266_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v266_data + (v213_data * v264_data));
              float v269_data = s1_w1[145];
              tensorforge::intel_esimd::simd<float, 32> v271_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v271_data + (v213_data * v269_data));
              float v274_data = s1_w1[158];
              tensorforge::intel_esimd::simd<float, 32> v276_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v276_data + (v213_data * v274_data));
              tensorforge::intel_esimd::simd<float, 32> v278_data(r2.template select<32, 1>(96));
              float v279_data = s1_w1[3];
              tensorforge::intel_esimd::simd<float, 32> v281_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v281_data + (v278_data * v279_data));
              float v284_data = s1_w1[16];
              tensorforge::intel_esimd::simd<float, 32> v286_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v286_data + (v278_data * v284_data));
              float v289_data = s1_w1[29];
              tensorforge::intel_esimd::simd<float, 32> v291_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v291_data + (v278_data * v289_data));
              float v294_data = s1_w1[42];
              tensorforge::intel_esimd::simd<float, 32> v296_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v296_data + (v278_data * v294_data));
              float v299_data = s1_w1[55];
              tensorforge::intel_esimd::simd<float, 32> v301_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v301_data + (v278_data * v299_data));
              float v304_data = s1_w1[68];
              tensorforge::intel_esimd::simd<float, 32> v306_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v306_data + (v278_data * v304_data));
              float v309_data = s1_w1[81];
              tensorforge::intel_esimd::simd<float, 32> v311_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v311_data + (v278_data * v309_data));
              float v314_data = s1_w1[94];
              tensorforge::intel_esimd::simd<float, 32> v316_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v316_data + (v278_data * v314_data));
              float v319_data = s1_w1[107];
              tensorforge::intel_esimd::simd<float, 32> v321_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v321_data + (v278_data * v319_data));
              float v324_data = s1_w1[120];
              tensorforge::intel_esimd::simd<float, 32> v326_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v326_data + (v278_data * v324_data));
              float v329_data = s1_w1[133];
              tensorforge::intel_esimd::simd<float, 32> v331_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v331_data + (v278_data * v329_data));
              float v334_data = s1_w1[146];
              tensorforge::intel_esimd::simd<float, 32> v336_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v336_data + (v278_data * v334_data));
              float v339_data = s1_w1[159];
              tensorforge::intel_esimd::simd<float, 32> v341_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v341_data + (v278_data * v339_data));
              tensorforge::intel_esimd::simd<float, 32> v343_data(r2.template select<32, 1>(128));
              float v344_data = s1_w1[4];
              tensorforge::intel_esimd::simd<float, 32> v346_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v346_data + (v343_data * v344_data));
              float v349_data = s1_w1[17];
              tensorforge::intel_esimd::simd<float, 32> v351_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v351_data + (v343_data * v349_data));
              float v354_data = s1_w1[30];
              tensorforge::intel_esimd::simd<float, 32> v356_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v356_data + (v343_data * v354_data));
              float v359_data = s1_w1[43];
              tensorforge::intel_esimd::simd<float, 32> v361_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v361_data + (v343_data * v359_data));
              float v364_data = s1_w1[56];
              tensorforge::intel_esimd::simd<float, 32> v366_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v366_data + (v343_data * v364_data));
              float v369_data = s1_w1[69];
              tensorforge::intel_esimd::simd<float, 32> v371_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v371_data + (v343_data * v369_data));
              float v374_data = s1_w1[82];
              tensorforge::intel_esimd::simd<float, 32> v376_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v376_data + (v343_data * v374_data));
              float v379_data = s1_w1[95];
              tensorforge::intel_esimd::simd<float, 32> v381_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v381_data + (v343_data * v379_data));
              float v384_data = s1_w1[108];
              tensorforge::intel_esimd::simd<float, 32> v386_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v386_data + (v343_data * v384_data));
              float v389_data = s1_w1[121];
              tensorforge::intel_esimd::simd<float, 32> v391_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v391_data + (v343_data * v389_data));
              float v394_data = s1_w1[134];
              tensorforge::intel_esimd::simd<float, 32> v396_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v396_data + (v343_data * v394_data));
              float v399_data = s1_w1[147];
              tensorforge::intel_esimd::simd<float, 32> v401_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v401_data + (v343_data * v399_data));
              float v404_data = s1_w1[160];
              tensorforge::intel_esimd::simd<float, 32> v406_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v406_data + (v343_data * v404_data));
              tensorforge::intel_esimd::simd<float, 32> v408_data(r2.template select<32, 1>(160));
              float v409_data = s1_w1[5];
              tensorforge::intel_esimd::simd<float, 32> v411_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v411_data + (v408_data * v409_data));
              float v414_data = s1_w1[18];
              tensorforge::intel_esimd::simd<float, 32> v416_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v416_data + (v408_data * v414_data));
              float v419_data = s1_w1[31];
              tensorforge::intel_esimd::simd<float, 32> v421_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v421_data + (v408_data * v419_data));
              float v424_data = s1_w1[44];
              tensorforge::intel_esimd::simd<float, 32> v426_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v426_data + (v408_data * v424_data));
              float v429_data = s1_w1[57];
              tensorforge::intel_esimd::simd<float, 32> v431_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v431_data + (v408_data * v429_data));
              float v434_data = s1_w1[70];
              tensorforge::intel_esimd::simd<float, 32> v436_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v436_data + (v408_data * v434_data));
              float v439_data = s1_w1[83];
              tensorforge::intel_esimd::simd<float, 32> v441_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v441_data + (v408_data * v439_data));
              float v444_data = s1_w1[96];
              tensorforge::intel_esimd::simd<float, 32> v446_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v446_data + (v408_data * v444_data));
              float v449_data = s1_w1[109];
              tensorforge::intel_esimd::simd<float, 32> v451_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v451_data + (v408_data * v449_data));
              float v454_data = s1_w1[122];
              tensorforge::intel_esimd::simd<float, 32> v456_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v456_data + (v408_data * v454_data));
              float v459_data = s1_w1[135];
              tensorforge::intel_esimd::simd<float, 32> v461_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v461_data + (v408_data * v459_data));
              float v464_data = s1_w1[148];
              tensorforge::intel_esimd::simd<float, 32> v466_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v466_data + (v408_data * v464_data));
              float v469_data = s1_w1[161];
              tensorforge::intel_esimd::simd<float, 32> v471_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v471_data + (v408_data * v469_data));
              tensorforge::intel_esimd::simd<float, 32> v473_data(r2.template select<32, 1>(192));
              float v474_data = s1_w1[6];
              tensorforge::intel_esimd::simd<float, 32> v476_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v476_data + (v473_data * v474_data));
              float v479_data = s1_w1[19];
              tensorforge::intel_esimd::simd<float, 32> v481_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v481_data + (v473_data * v479_data));
              float v484_data = s1_w1[32];
              tensorforge::intel_esimd::simd<float, 32> v486_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v486_data + (v473_data * v484_data));
              float v489_data = s1_w1[45];
              tensorforge::intel_esimd::simd<float, 32> v491_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v491_data + (v473_data * v489_data));
              float v494_data = s1_w1[58];
              tensorforge::intel_esimd::simd<float, 32> v496_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v496_data + (v473_data * v494_data));
              float v499_data = s1_w1[71];
              tensorforge::intel_esimd::simd<float, 32> v501_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v501_data + (v473_data * v499_data));
              float v504_data = s1_w1[84];
              tensorforge::intel_esimd::simd<float, 32> v506_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v506_data + (v473_data * v504_data));
              float v509_data = s1_w1[97];
              tensorforge::intel_esimd::simd<float, 32> v511_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v511_data + (v473_data * v509_data));
              float v514_data = s1_w1[110];
              tensorforge::intel_esimd::simd<float, 32> v516_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v516_data + (v473_data * v514_data));
              float v519_data = s1_w1[123];
              tensorforge::intel_esimd::simd<float, 32> v521_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v521_data + (v473_data * v519_data));
              float v524_data = s1_w1[136];
              tensorforge::intel_esimd::simd<float, 32> v526_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v526_data + (v473_data * v524_data));
              float v529_data = s1_w1[149];
              tensorforge::intel_esimd::simd<float, 32> v531_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v531_data + (v473_data * v529_data));
              float v534_data = s1_w1[162];
              tensorforge::intel_esimd::simd<float, 32> v536_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v536_data + (v473_data * v534_data));
              tensorforge::intel_esimd::simd<float, 32> v538_data(r2.template select<32, 1>(224));
              float v539_data = s1_w1[7];
              tensorforge::intel_esimd::simd<float, 32> v541_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v541_data + (v538_data * v539_data));
              float v544_data = s1_w1[20];
              tensorforge::intel_esimd::simd<float, 32> v546_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v546_data + (v538_data * v544_data));
              float v549_data = s1_w1[33];
              tensorforge::intel_esimd::simd<float, 32> v551_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v551_data + (v538_data * v549_data));
              float v554_data = s1_w1[46];
              tensorforge::intel_esimd::simd<float, 32> v556_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v556_data + (v538_data * v554_data));
              float v559_data = s1_w1[59];
              tensorforge::intel_esimd::simd<float, 32> v561_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v561_data + (v538_data * v559_data));
              float v564_data = s1_w1[72];
              tensorforge::intel_esimd::simd<float, 32> v566_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v566_data + (v538_data * v564_data));
              float v569_data = s1_w1[85];
              tensorforge::intel_esimd::simd<float, 32> v571_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v571_data + (v538_data * v569_data));
              float v574_data = s1_w1[98];
              tensorforge::intel_esimd::simd<float, 32> v576_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v576_data + (v538_data * v574_data));
              float v579_data = s1_w1[111];
              tensorforge::intel_esimd::simd<float, 32> v581_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v581_data + (v538_data * v579_data));
              float v584_data = s1_w1[124];
              tensorforge::intel_esimd::simd<float, 32> v586_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v586_data + (v538_data * v584_data));
              float v589_data = s1_w1[137];
              tensorforge::intel_esimd::simd<float, 32> v591_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v591_data + (v538_data * v589_data));
              float v594_data = s1_w1[150];
              tensorforge::intel_esimd::simd<float, 32> v596_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v596_data + (v538_data * v594_data));
              float v599_data = s1_w1[163];
              tensorforge::intel_esimd::simd<float, 32> v601_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v601_data + (v538_data * v599_data));
              tensorforge::intel_esimd::simd<float, 32> v603_data(r2.template select<32, 1>(256));
              float v604_data = s1_w1[8];
              tensorforge::intel_esimd::simd<float, 32> v606_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v606_data + (v603_data * v604_data));
              float v609_data = s1_w1[21];
              tensorforge::intel_esimd::simd<float, 32> v611_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v611_data + (v603_data * v609_data));
              float v614_data = s1_w1[34];
              tensorforge::intel_esimd::simd<float, 32> v616_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v616_data + (v603_data * v614_data));
              float v619_data = s1_w1[47];
              tensorforge::intel_esimd::simd<float, 32> v621_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v621_data + (v603_data * v619_data));
              float v624_data = s1_w1[60];
              tensorforge::intel_esimd::simd<float, 32> v626_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v626_data + (v603_data * v624_data));
              float v629_data = s1_w1[73];
              tensorforge::intel_esimd::simd<float, 32> v631_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v631_data + (v603_data * v629_data));
              float v634_data = s1_w1[86];
              tensorforge::intel_esimd::simd<float, 32> v636_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v636_data + (v603_data * v634_data));
              float v639_data = s1_w1[99];
              tensorforge::intel_esimd::simd<float, 32> v641_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v641_data + (v603_data * v639_data));
              float v644_data = s1_w1[112];
              tensorforge::intel_esimd::simd<float, 32> v646_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v646_data + (v603_data * v644_data));
              float v649_data = s1_w1[125];
              tensorforge::intel_esimd::simd<float, 32> v651_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v651_data + (v603_data * v649_data));
              float v654_data = s1_w1[138];
              tensorforge::intel_esimd::simd<float, 32> v656_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v656_data + (v603_data * v654_data));
              float v659_data = s1_w1[151];
              tensorforge::intel_esimd::simd<float, 32> v661_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v661_data + (v603_data * v659_data));
              float v664_data = s1_w1[164];
              tensorforge::intel_esimd::simd<float, 32> v666_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v666_data + (v603_data * v664_data));
              tensorforge::intel_esimd::simd<float, 32> v668_data(r2.template select<32, 1>(288));
              float v669_data = s1_w1[9];
              tensorforge::intel_esimd::simd<float, 32> v671_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v671_data + (v668_data * v669_data));
              float v674_data = s1_w1[22];
              tensorforge::intel_esimd::simd<float, 32> v676_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v676_data + (v668_data * v674_data));
              float v679_data = s1_w1[35];
              tensorforge::intel_esimd::simd<float, 32> v681_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v681_data + (v668_data * v679_data));
              float v684_data = s1_w1[48];
              tensorforge::intel_esimd::simd<float, 32> v686_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v686_data + (v668_data * v684_data));
              float v689_data = s1_w1[61];
              tensorforge::intel_esimd::simd<float, 32> v691_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v691_data + (v668_data * v689_data));
              float v694_data = s1_w1[74];
              tensorforge::intel_esimd::simd<float, 32> v696_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v696_data + (v668_data * v694_data));
              float v699_data = s1_w1[87];
              tensorforge::intel_esimd::simd<float, 32> v701_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v701_data + (v668_data * v699_data));
              float v704_data = s1_w1[100];
              tensorforge::intel_esimd::simd<float, 32> v706_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v706_data + (v668_data * v704_data));
              float v709_data = s1_w1[113];
              tensorforge::intel_esimd::simd<float, 32> v711_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v711_data + (v668_data * v709_data));
              float v714_data = s1_w1[126];
              tensorforge::intel_esimd::simd<float, 32> v716_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v716_data + (v668_data * v714_data));
              float v719_data = s1_w1[139];
              tensorforge::intel_esimd::simd<float, 32> v721_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v721_data + (v668_data * v719_data));
              float v724_data = s1_w1[152];
              tensorforge::intel_esimd::simd<float, 32> v726_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v726_data + (v668_data * v724_data));
              float v729_data = s1_w1[165];
              tensorforge::intel_esimd::simd<float, 32> v731_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v731_data + (v668_data * v729_data));
              tensorforge::intel_esimd::simd<float, 32> v733_data(r2.template select<32, 1>(320));
              float v734_data = s1_w1[10];
              tensorforge::intel_esimd::simd<float, 32> v736_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v736_data + (v733_data * v734_data));
              float v739_data = s1_w1[23];
              tensorforge::intel_esimd::simd<float, 32> v741_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v741_data + (v733_data * v739_data));
              float v744_data = s1_w1[36];
              tensorforge::intel_esimd::simd<float, 32> v746_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v746_data + (v733_data * v744_data));
              float v749_data = s1_w1[49];
              tensorforge::intel_esimd::simd<float, 32> v751_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v751_data + (v733_data * v749_data));
              float v754_data = s1_w1[62];
              tensorforge::intel_esimd::simd<float, 32> v756_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v756_data + (v733_data * v754_data));
              float v759_data = s1_w1[75];
              tensorforge::intel_esimd::simd<float, 32> v761_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v761_data + (v733_data * v759_data));
              float v764_data = s1_w1[88];
              tensorforge::intel_esimd::simd<float, 32> v766_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v766_data + (v733_data * v764_data));
              float v769_data = s1_w1[101];
              tensorforge::intel_esimd::simd<float, 32> v771_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v771_data + (v733_data * v769_data));
              float v774_data = s1_w1[114];
              tensorforge::intel_esimd::simd<float, 32> v776_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v776_data + (v733_data * v774_data));
              float v779_data = s1_w1[127];
              tensorforge::intel_esimd::simd<float, 32> v781_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v781_data + (v733_data * v779_data));
              float v784_data = s1_w1[140];
              tensorforge::intel_esimd::simd<float, 32> v786_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v786_data + (v733_data * v784_data));
              float v789_data = s1_w1[153];
              tensorforge::intel_esimd::simd<float, 32> v791_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v791_data + (v733_data * v789_data));
              float v794_data = s1_w1[166];
              tensorforge::intel_esimd::simd<float, 32> v796_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v796_data + (v733_data * v794_data));
              tensorforge::intel_esimd::simd<float, 32> v798_data(r2.template select<32, 1>(352));
              float v799_data = s1_w1[11];
              tensorforge::intel_esimd::simd<float, 32> v801_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v801_data + (v798_data * v799_data));
              float v804_data = s1_w1[24];
              tensorforge::intel_esimd::simd<float, 32> v806_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v806_data + (v798_data * v804_data));
              float v809_data = s1_w1[37];
              tensorforge::intel_esimd::simd<float, 32> v811_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v811_data + (v798_data * v809_data));
              float v814_data = s1_w1[50];
              tensorforge::intel_esimd::simd<float, 32> v816_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v816_data + (v798_data * v814_data));
              float v819_data = s1_w1[63];
              tensorforge::intel_esimd::simd<float, 32> v821_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v821_data + (v798_data * v819_data));
              float v824_data = s1_w1[76];
              tensorforge::intel_esimd::simd<float, 32> v826_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v826_data + (v798_data * v824_data));
              float v829_data = s1_w1[89];
              tensorforge::intel_esimd::simd<float, 32> v831_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v831_data + (v798_data * v829_data));
              float v834_data = s1_w1[102];
              tensorforge::intel_esimd::simd<float, 32> v836_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v836_data + (v798_data * v834_data));
              float v839_data = s1_w1[115];
              tensorforge::intel_esimd::simd<float, 32> v841_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v841_data + (v798_data * v839_data));
              float v844_data = s1_w1[128];
              tensorforge::intel_esimd::simd<float, 32> v846_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v846_data + (v798_data * v844_data));
              float v849_data = s1_w1[141];
              tensorforge::intel_esimd::simd<float, 32> v851_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v851_data + (v798_data * v849_data));
              float v854_data = s1_w1[154];
              tensorforge::intel_esimd::simd<float, 32> v856_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v856_data + (v798_data * v854_data));
              float v859_data = s1_w1[167];
              tensorforge::intel_esimd::simd<float, 32> v861_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v861_data + (v798_data * v859_data));
              tensorforge::intel_esimd::simd<float, 32> v863_data(r2.template select<32, 1>(384));
              float v864_data = s1_w1[12];
              tensorforge::intel_esimd::simd<float, 32> v866_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v866_data + (v863_data * v864_data));
              float v869_data = s1_w1[25];
              tensorforge::intel_esimd::simd<float, 32> v871_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v871_data + (v863_data * v869_data));
              float v874_data = s1_w1[38];
              tensorforge::intel_esimd::simd<float, 32> v876_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v876_data + (v863_data * v874_data));
              float v879_data = s1_w1[51];
              tensorforge::intel_esimd::simd<float, 32> v881_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v881_data + (v863_data * v879_data));
              float v884_data = s1_w1[64];
              tensorforge::intel_esimd::simd<float, 32> v886_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v886_data + (v863_data * v884_data));
              float v889_data = s1_w1[77];
              tensorforge::intel_esimd::simd<float, 32> v891_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v891_data + (v863_data * v889_data));
              float v894_data = s1_w1[90];
              tensorforge::intel_esimd::simd<float, 32> v896_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v896_data + (v863_data * v894_data));
              float v899_data = s1_w1[103];
              tensorforge::intel_esimd::simd<float, 32> v901_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v901_data + (v863_data * v899_data));
              float v904_data = s1_w1[116];
              tensorforge::intel_esimd::simd<float, 32> v906_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v906_data + (v863_data * v904_data));
              float v909_data = s1_w1[129];
              tensorforge::intel_esimd::simd<float, 32> v911_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v911_data + (v863_data * v909_data));
              float v914_data = s1_w1[142];
              tensorforge::intel_esimd::simd<float, 32> v916_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v916_data + (v863_data * v914_data));
              float v919_data = s1_w1[155];
              tensorforge::intel_esimd::simd<float, 32> v921_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v921_data + (v863_data * v919_data));
              float v924_data = s1_w1[168];
              tensorforge::intel_esimd::simd<float, 32> v926_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v926_data + (v863_data * v924_data));
              // r3 = ir3
              #pragma unroll
              for (int32_t v928_n0 = 0; v928_n0 < 1; ++v928_n0) {
                int32_t v930_a = v928_n0 * 32;
                #pragma unroll
                for (int32_t v929_n1 = 0; v929_n1 < 13; ++v929_n1) {
                  int32_t v932_a = v930_a + (v929_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v933_data(ir3.template select<32, 1>(v932_a));
                  r3.template select<32, 1>(v932_a) = v933_data;
                }
              }
              // glb_m3 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v934_i0 = 0; v934_i0 < 1; ++v934_i0) {
                int32_t v936_a = v934_i0 * 32;
                #pragma unroll
                for (int32_t v935_i1 = 0; v935_i1 < 13; ++v935_i1) {
                  int32_t v938_a = v936_a + (v935_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v939_data(r3.template select<32, 1>(v938_a));
                  v939_data.copy_to(glb_m3 + (v938_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


// === base name ===
kernel_687fc0ae49185375

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_687fc0ae49185375 = {{1, 8, 1}, 32, 32, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_687fc0ae49185375(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_687fc0ae49185375(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_687fc0ae49185375(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_687fc0ae49185375(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_687fc0ae49185375(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_687fc0ae49185375(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_687fc0ae49185375(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (176);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v12_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v12_batchId0 < numElements0; v12_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 169 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v12_batchId0 * 169 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 96> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
                int32_t v28_lead = v26_i0 * 32;
                #pragma unroll
                for (int32_t v27_i1 = 10; v27_i1 < 13; ++v27_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v32_data;
                  v32_data.copy_from(glb_m1 + ((v28_lead + (v27_i1 * 32))));
                  r0.template select<32, 1>((v28_lead + ((v27_i1 - 10) * 32))) = v32_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v36_ld;
              v36_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v36_ld);
              tensorforge::intel_esimd::simd<float, 32> v37_ld;
              v37_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 128), v37_ld);
              tensorforge::intel_esimd::simd<float, 9> v38_ld;
              v38_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 160), v38_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 32> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 32), (0, 1)] [(10, 13)]
              tensorforge::intel_esimd::simd<float, 32> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v41_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> s0_w0 = tensorforge::slmLoad<float, 16>(s0 + 114);
              float v42_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v44_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v44_data + (v41_data * v42_data));
              tensorforge::intel_esimd::simd<float, 32> v46_data(r0.template select<32, 1>(32));
              float v47_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v49_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v49_data + (v46_data * v47_data));
              tensorforge::intel_esimd::simd<float, 32> v51_data(r0.template select<32, 1>(64));
              float v52_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v54_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v54_data + (v51_data * v52_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v56_n0 = 0; v56_n0 < 1; ++v56_n0) {
                int32_t v58_a = v56_n0 * 32;
                #pragma unroll
                for (int32_t v57_n1 = 0; v57_n1 < 1; ++v57_n1) {
                  int32_t v60_a = v58_a + (v57_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v61_data(ir1.template select<32, 1>(v60_a));
                  r1.template select<32, 1>(v60_a) = v61_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v62_i0 = 0; v62_i0 < 1; ++v62_i0) {
                int32_t v64_a = v62_i0 * 32;
                #pragma unroll
                for (int32_t v63_i1 = 0; v63_i1 < 1; ++v63_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v67_data(r1.template select<32, 1>((v64_a + (v63_i1 * 32))));
                  v67_data.copy_to(glb_m0 + ((v64_a + ((v63_i1 + 8) * 32))));
                }
              }
              tensorforge::intel_esimd::simd<float, 416> r2(0.0f);
              // r2 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v73_i0 = 0; v73_i0 < 1; ++v73_i0) {
                int32_t v75_lead = v73_i0 * 32;
                #pragma unroll
                for (int32_t v74_i1 = 0; v74_i1 < 13; ++v74_i1) {
                  int32_t v78_a = v75_lead + (v74_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v79_data;
                  v79_data.copy_from(glb_m0 + (v78_a));
                  r2.template select<32, 1>(v78_a) = v79_data;
                }
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v81_ld;
              v81_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s1 + (0 + 0 + 4 * 0 + 0), v81_ld);
              tensorforge::intel_esimd::simd<float, 32> v82_ld;
              v82_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 1 * 0 + 128), v82_ld);
              tensorforge::intel_esimd::simd<float, 9> v83_ld;
              v83_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s1 + (0 + 0 + 1 * 0 + 160), v83_ld);
              // wait(r2 = load{g>r}(glb_m0););
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 416> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 32), (0, 13)] [(0, 13)]
              tensorforge::intel_esimd::simd<float, 416> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v86_data(r2.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 176> s1_w1 = tensorforge::slmLoad<float, 176>(s1 + 0);
              float v87_data = s1_w1[0];
              tensorforge::intel_esimd::simd<float, 32> v89_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v89_data + (v86_data * v87_data));
              float v92_data = s1_w1[13];
              tensorforge::intel_esimd::simd<float, 32> v94_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v94_data + (v86_data * v92_data));
              float v97_data = s1_w1[26];
              tensorforge::intel_esimd::simd<float, 32> v99_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v99_data + (v86_data * v97_data));
              float v102_data = s1_w1[39];
              tensorforge::intel_esimd::simd<float, 32> v104_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v104_data + (v86_data * v102_data));
              float v107_data = s1_w1[52];
              tensorforge::intel_esimd::simd<float, 32> v109_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v109_data + (v86_data * v107_data));
              float v112_data = s1_w1[65];
              tensorforge::intel_esimd::simd<float, 32> v114_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v114_data + (v86_data * v112_data));
              float v117_data = s1_w1[78];
              tensorforge::intel_esimd::simd<float, 32> v119_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v119_data + (v86_data * v117_data));
              float v122_data = s1_w1[91];
              tensorforge::intel_esimd::simd<float, 32> v124_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v124_data + (v86_data * v122_data));
              float v127_data = s1_w1[104];
              tensorforge::intel_esimd::simd<float, 32> v129_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v129_data + (v86_data * v127_data));
              float v132_data = s1_w1[117];
              tensorforge::intel_esimd::simd<float, 32> v134_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v134_data + (v86_data * v132_data));
              float v137_data = s1_w1[130];
              tensorforge::intel_esimd::simd<float, 32> v139_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v139_data + (v86_data * v137_data));
              float v142_data = s1_w1[143];
              tensorforge::intel_esimd::simd<float, 32> v144_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v144_data + (v86_data * v142_data));
              float v147_data = s1_w1[156];
              tensorforge::intel_esimd::simd<float, 32> v149_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v149_data + (v86_data * v147_data));
              tensorforge::intel_esimd::simd<float, 32> v151_data(r2.template select<32, 1>(32));
              float v152_data = s1_w1[1];
              tensorforge::intel_esimd::simd<float, 32> v154_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v154_data + (v151_data * v152_data));
              float v157_data = s1_w1[14];
              tensorforge::intel_esimd::simd<float, 32> v159_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v159_data + (v151_data * v157_data));
              float v162_data = s1_w1[27];
              tensorforge::intel_esimd::simd<float, 32> v164_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v164_data + (v151_data * v162_data));
              float v167_data = s1_w1[40];
              tensorforge::intel_esimd::simd<float, 32> v169_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v169_data + (v151_data * v167_data));
              float v172_data = s1_w1[53];
              tensorforge::intel_esimd::simd<float, 32> v174_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v174_data + (v151_data * v172_data));
              float v177_data = s1_w1[66];
              tensorforge::intel_esimd::simd<float, 32> v179_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v179_data + (v151_data * v177_data));
              float v182_data = s1_w1[79];
              tensorforge::intel_esimd::simd<float, 32> v184_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v184_data + (v151_data * v182_data));
              float v187_data = s1_w1[92];
              tensorforge::intel_esimd::simd<float, 32> v189_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v189_data + (v151_data * v187_data));
              float v192_data = s1_w1[105];
              tensorforge::intel_esimd::simd<float, 32> v194_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v194_data + (v151_data * v192_data));
              float v197_data = s1_w1[118];
              tensorforge::intel_esimd::simd<float, 32> v199_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v199_data + (v151_data * v197_data));
              float v202_data = s1_w1[131];
              tensorforge::intel_esimd::simd<float, 32> v204_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v204_data + (v151_data * v202_data));
              float v207_data = s1_w1[144];
              tensorforge::intel_esimd::simd<float, 32> v209_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v209_data + (v151_data * v207_data));
              float v212_data = s1_w1[157];
              tensorforge::intel_esimd::simd<float, 32> v214_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v214_data + (v151_data * v212_data));
              tensorforge::intel_esimd::simd<float, 32> v216_data(r2.template select<32, 1>(64));
              float v217_data = s1_w1[2];
              tensorforge::intel_esimd::simd<float, 32> v219_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v219_data + (v216_data * v217_data));
              float v222_data = s1_w1[15];
              tensorforge::intel_esimd::simd<float, 32> v224_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v224_data + (v216_data * v222_data));
              float v227_data = s1_w1[28];
              tensorforge::intel_esimd::simd<float, 32> v229_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v229_data + (v216_data * v227_data));
              float v232_data = s1_w1[41];
              tensorforge::intel_esimd::simd<float, 32> v234_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v234_data + (v216_data * v232_data));
              float v237_data = s1_w1[54];
              tensorforge::intel_esimd::simd<float, 32> v239_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v239_data + (v216_data * v237_data));
              float v242_data = s1_w1[67];
              tensorforge::intel_esimd::simd<float, 32> v244_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v244_data + (v216_data * v242_data));
              float v247_data = s1_w1[80];
              tensorforge::intel_esimd::simd<float, 32> v249_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v249_data + (v216_data * v247_data));
              float v252_data = s1_w1[93];
              tensorforge::intel_esimd::simd<float, 32> v254_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v254_data + (v216_data * v252_data));
              float v257_data = s1_w1[106];
              tensorforge::intel_esimd::simd<float, 32> v259_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v259_data + (v216_data * v257_data));
              float v262_data = s1_w1[119];
              tensorforge::intel_esimd::simd<float, 32> v264_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v264_data + (v216_data * v262_data));
              float v267_data = s1_w1[132];
              tensorforge::intel_esimd::simd<float, 32> v269_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v269_data + (v216_data * v267_data));
              float v272_data = s1_w1[145];
              tensorforge::intel_esimd::simd<float, 32> v274_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v274_data + (v216_data * v272_data));
              float v277_data = s1_w1[158];
              tensorforge::intel_esimd::simd<float, 32> v279_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v279_data + (v216_data * v277_data));
              tensorforge::intel_esimd::simd<float, 32> v281_data(r2.template select<32, 1>(96));
              float v282_data = s1_w1[3];
              tensorforge::intel_esimd::simd<float, 32> v284_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v284_data + (v281_data * v282_data));
              float v287_data = s1_w1[16];
              tensorforge::intel_esimd::simd<float, 32> v289_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v289_data + (v281_data * v287_data));
              float v292_data = s1_w1[29];
              tensorforge::intel_esimd::simd<float, 32> v294_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v294_data + (v281_data * v292_data));
              float v297_data = s1_w1[42];
              tensorforge::intel_esimd::simd<float, 32> v299_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v299_data + (v281_data * v297_data));
              float v302_data = s1_w1[55];
              tensorforge::intel_esimd::simd<float, 32> v304_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v304_data + (v281_data * v302_data));
              float v307_data = s1_w1[68];
              tensorforge::intel_esimd::simd<float, 32> v309_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v309_data + (v281_data * v307_data));
              float v312_data = s1_w1[81];
              tensorforge::intel_esimd::simd<float, 32> v314_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v314_data + (v281_data * v312_data));
              float v317_data = s1_w1[94];
              tensorforge::intel_esimd::simd<float, 32> v319_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v319_data + (v281_data * v317_data));
              float v322_data = s1_w1[107];
              tensorforge::intel_esimd::simd<float, 32> v324_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v324_data + (v281_data * v322_data));
              float v327_data = s1_w1[120];
              tensorforge::intel_esimd::simd<float, 32> v329_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v329_data + (v281_data * v327_data));
              float v332_data = s1_w1[133];
              tensorforge::intel_esimd::simd<float, 32> v334_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v334_data + (v281_data * v332_data));
              float v337_data = s1_w1[146];
              tensorforge::intel_esimd::simd<float, 32> v339_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v339_data + (v281_data * v337_data));
              float v342_data = s1_w1[159];
              tensorforge::intel_esimd::simd<float, 32> v344_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v344_data + (v281_data * v342_data));
              tensorforge::intel_esimd::simd<float, 32> v346_data(r2.template select<32, 1>(128));
              float v347_data = s1_w1[4];
              tensorforge::intel_esimd::simd<float, 32> v349_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v349_data + (v346_data * v347_data));
              float v352_data = s1_w1[17];
              tensorforge::intel_esimd::simd<float, 32> v354_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v354_data + (v346_data * v352_data));
              float v357_data = s1_w1[30];
              tensorforge::intel_esimd::simd<float, 32> v359_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v359_data + (v346_data * v357_data));
              float v362_data = s1_w1[43];
              tensorforge::intel_esimd::simd<float, 32> v364_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v364_data + (v346_data * v362_data));
              float v367_data = s1_w1[56];
              tensorforge::intel_esimd::simd<float, 32> v369_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v369_data + (v346_data * v367_data));
              float v372_data = s1_w1[69];
              tensorforge::intel_esimd::simd<float, 32> v374_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v374_data + (v346_data * v372_data));
              float v377_data = s1_w1[82];
              tensorforge::intel_esimd::simd<float, 32> v379_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v379_data + (v346_data * v377_data));
              float v382_data = s1_w1[95];
              tensorforge::intel_esimd::simd<float, 32> v384_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v384_data + (v346_data * v382_data));
              float v387_data = s1_w1[108];
              tensorforge::intel_esimd::simd<float, 32> v389_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v389_data + (v346_data * v387_data));
              float v392_data = s1_w1[121];
              tensorforge::intel_esimd::simd<float, 32> v394_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v394_data + (v346_data * v392_data));
              float v397_data = s1_w1[134];
              tensorforge::intel_esimd::simd<float, 32> v399_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v399_data + (v346_data * v397_data));
              float v402_data = s1_w1[147];
              tensorforge::intel_esimd::simd<float, 32> v404_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v404_data + (v346_data * v402_data));
              float v407_data = s1_w1[160];
              tensorforge::intel_esimd::simd<float, 32> v409_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v409_data + (v346_data * v407_data));
              tensorforge::intel_esimd::simd<float, 32> v411_data(r2.template select<32, 1>(160));
              float v412_data = s1_w1[5];
              tensorforge::intel_esimd::simd<float, 32> v414_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v414_data + (v411_data * v412_data));
              float v417_data = s1_w1[18];
              tensorforge::intel_esimd::simd<float, 32> v419_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v419_data + (v411_data * v417_data));
              float v422_data = s1_w1[31];
              tensorforge::intel_esimd::simd<float, 32> v424_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v424_data + (v411_data * v422_data));
              float v427_data = s1_w1[44];
              tensorforge::intel_esimd::simd<float, 32> v429_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v429_data + (v411_data * v427_data));
              float v432_data = s1_w1[57];
              tensorforge::intel_esimd::simd<float, 32> v434_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v434_data + (v411_data * v432_data));
              float v437_data = s1_w1[70];
              tensorforge::intel_esimd::simd<float, 32> v439_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v439_data + (v411_data * v437_data));
              float v442_data = s1_w1[83];
              tensorforge::intel_esimd::simd<float, 32> v444_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v444_data + (v411_data * v442_data));
              float v447_data = s1_w1[96];
              tensorforge::intel_esimd::simd<float, 32> v449_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v449_data + (v411_data * v447_data));
              float v452_data = s1_w1[109];
              tensorforge::intel_esimd::simd<float, 32> v454_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v454_data + (v411_data * v452_data));
              float v457_data = s1_w1[122];
              tensorforge::intel_esimd::simd<float, 32> v459_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v459_data + (v411_data * v457_data));
              float v462_data = s1_w1[135];
              tensorforge::intel_esimd::simd<float, 32> v464_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v464_data + (v411_data * v462_data));
              float v467_data = s1_w1[148];
              tensorforge::intel_esimd::simd<float, 32> v469_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v469_data + (v411_data * v467_data));
              float v472_data = s1_w1[161];
              tensorforge::intel_esimd::simd<float, 32> v474_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v474_data + (v411_data * v472_data));
              tensorforge::intel_esimd::simd<float, 32> v476_data(r2.template select<32, 1>(192));
              float v477_data = s1_w1[6];
              tensorforge::intel_esimd::simd<float, 32> v479_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v479_data + (v476_data * v477_data));
              float v482_data = s1_w1[19];
              tensorforge::intel_esimd::simd<float, 32> v484_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v484_data + (v476_data * v482_data));
              float v487_data = s1_w1[32];
              tensorforge::intel_esimd::simd<float, 32> v489_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v489_data + (v476_data * v487_data));
              float v492_data = s1_w1[45];
              tensorforge::intel_esimd::simd<float, 32> v494_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v494_data + (v476_data * v492_data));
              float v497_data = s1_w1[58];
              tensorforge::intel_esimd::simd<float, 32> v499_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v499_data + (v476_data * v497_data));
              float v502_data = s1_w1[71];
              tensorforge::intel_esimd::simd<float, 32> v504_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v504_data + (v476_data * v502_data));
              float v507_data = s1_w1[84];
              tensorforge::intel_esimd::simd<float, 32> v509_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v509_data + (v476_data * v507_data));
              float v512_data = s1_w1[97];
              tensorforge::intel_esimd::simd<float, 32> v514_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v514_data + (v476_data * v512_data));
              float v517_data = s1_w1[110];
              tensorforge::intel_esimd::simd<float, 32> v519_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v519_data + (v476_data * v517_data));
              float v522_data = s1_w1[123];
              tensorforge::intel_esimd::simd<float, 32> v524_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v524_data + (v476_data * v522_data));
              float v527_data = s1_w1[136];
              tensorforge::intel_esimd::simd<float, 32> v529_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v529_data + (v476_data * v527_data));
              float v532_data = s1_w1[149];
              tensorforge::intel_esimd::simd<float, 32> v534_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v534_data + (v476_data * v532_data));
              float v537_data = s1_w1[162];
              tensorforge::intel_esimd::simd<float, 32> v539_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v539_data + (v476_data * v537_data));
              tensorforge::intel_esimd::simd<float, 32> v541_data(r2.template select<32, 1>(224));
              float v542_data = s1_w1[7];
              tensorforge::intel_esimd::simd<float, 32> v544_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v544_data + (v541_data * v542_data));
              float v547_data = s1_w1[20];
              tensorforge::intel_esimd::simd<float, 32> v549_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v549_data + (v541_data * v547_data));
              float v552_data = s1_w1[33];
              tensorforge::intel_esimd::simd<float, 32> v554_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v554_data + (v541_data * v552_data));
              float v557_data = s1_w1[46];
              tensorforge::intel_esimd::simd<float, 32> v559_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v559_data + (v541_data * v557_data));
              float v562_data = s1_w1[59];
              tensorforge::intel_esimd::simd<float, 32> v564_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v564_data + (v541_data * v562_data));
              float v567_data = s1_w1[72];
              tensorforge::intel_esimd::simd<float, 32> v569_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v569_data + (v541_data * v567_data));
              float v572_data = s1_w1[85];
              tensorforge::intel_esimd::simd<float, 32> v574_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v574_data + (v541_data * v572_data));
              float v577_data = s1_w1[98];
              tensorforge::intel_esimd::simd<float, 32> v579_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v579_data + (v541_data * v577_data));
              float v582_data = s1_w1[111];
              tensorforge::intel_esimd::simd<float, 32> v584_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v584_data + (v541_data * v582_data));
              float v587_data = s1_w1[124];
              tensorforge::intel_esimd::simd<float, 32> v589_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v589_data + (v541_data * v587_data));
              float v592_data = s1_w1[137];
              tensorforge::intel_esimd::simd<float, 32> v594_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v594_data + (v541_data * v592_data));
              float v597_data = s1_w1[150];
              tensorforge::intel_esimd::simd<float, 32> v599_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v599_data + (v541_data * v597_data));
              float v602_data = s1_w1[163];
              tensorforge::intel_esimd::simd<float, 32> v604_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v604_data + (v541_data * v602_data));
              tensorforge::intel_esimd::simd<float, 32> v606_data(r2.template select<32, 1>(256));
              float v607_data = s1_w1[8];
              tensorforge::intel_esimd::simd<float, 32> v609_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v609_data + (v606_data * v607_data));
              float v612_data = s1_w1[21];
              tensorforge::intel_esimd::simd<float, 32> v614_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v614_data + (v606_data * v612_data));
              float v617_data = s1_w1[34];
              tensorforge::intel_esimd::simd<float, 32> v619_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v619_data + (v606_data * v617_data));
              float v622_data = s1_w1[47];
              tensorforge::intel_esimd::simd<float, 32> v624_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v624_data + (v606_data * v622_data));
              float v627_data = s1_w1[60];
              tensorforge::intel_esimd::simd<float, 32> v629_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v629_data + (v606_data * v627_data));
              float v632_data = s1_w1[73];
              tensorforge::intel_esimd::simd<float, 32> v634_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v634_data + (v606_data * v632_data));
              float v637_data = s1_w1[86];
              tensorforge::intel_esimd::simd<float, 32> v639_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v639_data + (v606_data * v637_data));
              float v642_data = s1_w1[99];
              tensorforge::intel_esimd::simd<float, 32> v644_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v644_data + (v606_data * v642_data));
              float v647_data = s1_w1[112];
              tensorforge::intel_esimd::simd<float, 32> v649_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v649_data + (v606_data * v647_data));
              float v652_data = s1_w1[125];
              tensorforge::intel_esimd::simd<float, 32> v654_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v654_data + (v606_data * v652_data));
              float v657_data = s1_w1[138];
              tensorforge::intel_esimd::simd<float, 32> v659_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v659_data + (v606_data * v657_data));
              float v662_data = s1_w1[151];
              tensorforge::intel_esimd::simd<float, 32> v664_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v664_data + (v606_data * v662_data));
              float v667_data = s1_w1[164];
              tensorforge::intel_esimd::simd<float, 32> v669_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v669_data + (v606_data * v667_data));
              tensorforge::intel_esimd::simd<float, 32> v671_data(r2.template select<32, 1>(288));
              float v672_data = s1_w1[9];
              tensorforge::intel_esimd::simd<float, 32> v674_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v674_data + (v671_data * v672_data));
              float v677_data = s1_w1[22];
              tensorforge::intel_esimd::simd<float, 32> v679_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v679_data + (v671_data * v677_data));
              float v682_data = s1_w1[35];
              tensorforge::intel_esimd::simd<float, 32> v684_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v684_data + (v671_data * v682_data));
              float v687_data = s1_w1[48];
              tensorforge::intel_esimd::simd<float, 32> v689_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v689_data + (v671_data * v687_data));
              float v692_data = s1_w1[61];
              tensorforge::intel_esimd::simd<float, 32> v694_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v694_data + (v671_data * v692_data));
              float v697_data = s1_w1[74];
              tensorforge::intel_esimd::simd<float, 32> v699_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v699_data + (v671_data * v697_data));
              float v702_data = s1_w1[87];
              tensorforge::intel_esimd::simd<float, 32> v704_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v704_data + (v671_data * v702_data));
              float v707_data = s1_w1[100];
              tensorforge::intel_esimd::simd<float, 32> v709_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v709_data + (v671_data * v707_data));
              float v712_data = s1_w1[113];
              tensorforge::intel_esimd::simd<float, 32> v714_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v714_data + (v671_data * v712_data));
              float v717_data = s1_w1[126];
              tensorforge::intel_esimd::simd<float, 32> v719_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v719_data + (v671_data * v717_data));
              float v722_data = s1_w1[139];
              tensorforge::intel_esimd::simd<float, 32> v724_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v724_data + (v671_data * v722_data));
              float v727_data = s1_w1[152];
              tensorforge::intel_esimd::simd<float, 32> v729_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v729_data + (v671_data * v727_data));
              float v732_data = s1_w1[165];
              tensorforge::intel_esimd::simd<float, 32> v734_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v734_data + (v671_data * v732_data));
              tensorforge::intel_esimd::simd<float, 32> v736_data(r2.template select<32, 1>(320));
              float v737_data = s1_w1[10];
              tensorforge::intel_esimd::simd<float, 32> v739_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v739_data + (v736_data * v737_data));
              float v742_data = s1_w1[23];
              tensorforge::intel_esimd::simd<float, 32> v744_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v744_data + (v736_data * v742_data));
              float v747_data = s1_w1[36];
              tensorforge::intel_esimd::simd<float, 32> v749_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v749_data + (v736_data * v747_data));
              float v752_data = s1_w1[49];
              tensorforge::intel_esimd::simd<float, 32> v754_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v754_data + (v736_data * v752_data));
              float v757_data = s1_w1[62];
              tensorforge::intel_esimd::simd<float, 32> v759_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v759_data + (v736_data * v757_data));
              float v762_data = s1_w1[75];
              tensorforge::intel_esimd::simd<float, 32> v764_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v764_data + (v736_data * v762_data));
              float v767_data = s1_w1[88];
              tensorforge::intel_esimd::simd<float, 32> v769_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v769_data + (v736_data * v767_data));
              float v772_data = s1_w1[101];
              tensorforge::intel_esimd::simd<float, 32> v774_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v774_data + (v736_data * v772_data));
              float v777_data = s1_w1[114];
              tensorforge::intel_esimd::simd<float, 32> v779_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v779_data + (v736_data * v777_data));
              float v782_data = s1_w1[127];
              tensorforge::intel_esimd::simd<float, 32> v784_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v784_data + (v736_data * v782_data));
              float v787_data = s1_w1[140];
              tensorforge::intel_esimd::simd<float, 32> v789_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v789_data + (v736_data * v787_data));
              float v792_data = s1_w1[153];
              tensorforge::intel_esimd::simd<float, 32> v794_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v794_data + (v736_data * v792_data));
              float v797_data = s1_w1[166];
              tensorforge::intel_esimd::simd<float, 32> v799_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v799_data + (v736_data * v797_data));
              tensorforge::intel_esimd::simd<float, 32> v801_data(r2.template select<32, 1>(352));
              float v802_data = s1_w1[11];
              tensorforge::intel_esimd::simd<float, 32> v804_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v804_data + (v801_data * v802_data));
              float v807_data = s1_w1[24];
              tensorforge::intel_esimd::simd<float, 32> v809_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v809_data + (v801_data * v807_data));
              float v812_data = s1_w1[37];
              tensorforge::intel_esimd::simd<float, 32> v814_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v814_data + (v801_data * v812_data));
              float v817_data = s1_w1[50];
              tensorforge::intel_esimd::simd<float, 32> v819_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v819_data + (v801_data * v817_data));
              float v822_data = s1_w1[63];
              tensorforge::intel_esimd::simd<float, 32> v824_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v824_data + (v801_data * v822_data));
              float v827_data = s1_w1[76];
              tensorforge::intel_esimd::simd<float, 32> v829_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v829_data + (v801_data * v827_data));
              float v832_data = s1_w1[89];
              tensorforge::intel_esimd::simd<float, 32> v834_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v834_data + (v801_data * v832_data));
              float v837_data = s1_w1[102];
              tensorforge::intel_esimd::simd<float, 32> v839_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v839_data + (v801_data * v837_data));
              float v842_data = s1_w1[115];
              tensorforge::intel_esimd::simd<float, 32> v844_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v844_data + (v801_data * v842_data));
              float v847_data = s1_w1[128];
              tensorforge::intel_esimd::simd<float, 32> v849_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v849_data + (v801_data * v847_data));
              float v852_data = s1_w1[141];
              tensorforge::intel_esimd::simd<float, 32> v854_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v854_data + (v801_data * v852_data));
              float v857_data = s1_w1[154];
              tensorforge::intel_esimd::simd<float, 32> v859_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v859_data + (v801_data * v857_data));
              float v862_data = s1_w1[167];
              tensorforge::intel_esimd::simd<float, 32> v864_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v864_data + (v801_data * v862_data));
              tensorforge::intel_esimd::simd<float, 32> v866_data(r2.template select<32, 1>(384));
              float v867_data = s1_w1[12];
              tensorforge::intel_esimd::simd<float, 32> v869_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v869_data + (v866_data * v867_data));
              float v872_data = s1_w1[25];
              tensorforge::intel_esimd::simd<float, 32> v874_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v874_data + (v866_data * v872_data));
              float v877_data = s1_w1[38];
              tensorforge::intel_esimd::simd<float, 32> v879_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v879_data + (v866_data * v877_data));
              float v882_data = s1_w1[51];
              tensorforge::intel_esimd::simd<float, 32> v884_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v884_data + (v866_data * v882_data));
              float v887_data = s1_w1[64];
              tensorforge::intel_esimd::simd<float, 32> v889_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v889_data + (v866_data * v887_data));
              float v892_data = s1_w1[77];
              tensorforge::intel_esimd::simd<float, 32> v894_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v894_data + (v866_data * v892_data));
              float v897_data = s1_w1[90];
              tensorforge::intel_esimd::simd<float, 32> v899_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v899_data + (v866_data * v897_data));
              float v902_data = s1_w1[103];
              tensorforge::intel_esimd::simd<float, 32> v904_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v904_data + (v866_data * v902_data));
              float v907_data = s1_w1[116];
              tensorforge::intel_esimd::simd<float, 32> v909_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v909_data + (v866_data * v907_data));
              float v912_data = s1_w1[129];
              tensorforge::intel_esimd::simd<float, 32> v914_data(ir3.template select<32, 1>(288));
              ir3.template select<32, 1>(288) = (v914_data + (v866_data * v912_data));
              float v917_data = s1_w1[142];
              tensorforge::intel_esimd::simd<float, 32> v919_data(ir3.template select<32, 1>(320));
              ir3.template select<32, 1>(320) = (v919_data + (v866_data * v917_data));
              float v922_data = s1_w1[155];
              tensorforge::intel_esimd::simd<float, 32> v924_data(ir3.template select<32, 1>(352));
              ir3.template select<32, 1>(352) = (v924_data + (v866_data * v922_data));
              float v927_data = s1_w1[168];
              tensorforge::intel_esimd::simd<float, 32> v929_data(ir3.template select<32, 1>(384));
              ir3.template select<32, 1>(384) = (v929_data + (v866_data * v927_data));
              // r3 = ir3
              #pragma unroll
              for (int32_t v931_n0 = 0; v931_n0 < 1; ++v931_n0) {
                int32_t v933_a = v931_n0 * 32;
                #pragma unroll
                for (int32_t v932_n1 = 0; v932_n1 < 13; ++v932_n1) {
                  int32_t v935_a = v933_a + (v932_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v936_data(ir3.template select<32, 1>(v935_a));
                  r3.template select<32, 1>(v935_a) = v936_data;
                }
              }
              // glb_m3 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v937_i0 = 0; v937_i0 < 1; ++v937_i0) {
                int32_t v939_a = v937_i0 * 32;
                #pragma unroll
                for (int32_t v938_i1 = 0; v938_i1 < 13; ++v938_i1) {
                  int32_t v941_a = v939_a + (v938_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v942_data(r3.template select<32, 1>(v941_a));
                  v942_data.copy_to(glb_m3 + (v941_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


// === base name ===
kernel_b5474370464a5bfd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b5474370464a5bfd = {{1, 32, 1}, 8, 8, 1, 32, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b5474370464a5bfd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b5474370464a5bfd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b5474370464a5bfd(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_b5474370464a5bfd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b5474370464a5bfd(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b5474370464a5bfd(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b5474370464a5bfd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<3328 * sizeof(float)>(); {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":3328}],"shared_bytes":13312,"shared_elements":3328,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (104 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (64);
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v10_batchId0 * 64 + 0 + m3_extraOffset];
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
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v201_ld;
              v201_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 0), v201_ld);
              tensorforge::intel_esimd::simd<float, 32> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v33_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> s0_w0 = tensorforge::slmLoad<float, 32>(s0 + 0);
              float v34_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v36_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v36_data + (v33_data * v34_data));
              float v39_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v41_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v41_data + (v33_data * v39_data));
              float v44_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v46_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v46_data + (v33_data * v44_data));
              float v49_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v51_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v51_data + (v33_data * v49_data));
              tensorforge::intel_esimd::simd<float, 8> v53_data(r0.template select<8, 1>(8));
              float v54_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v56_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v56_data + (v53_data * v54_data));
              float v59_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v61_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v61_data + (v53_data * v59_data));
              float v64_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v66_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v66_data + (v53_data * v64_data));
              float v69_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v71_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v71_data + (v53_data * v69_data));
              tensorforge::intel_esimd::simd<float, 8> v73_data(r0.template select<8, 1>(16));
              float v74_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v76_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v76_data + (v73_data * v74_data));
              float v79_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v81_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v81_data + (v73_data * v79_data));
              float v84_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v86_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v86_data + (v73_data * v84_data));
              float v89_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v91_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v91_data + (v73_data * v89_data));
              tensorforge::intel_esimd::simd<float, 8> v93_data(r0.template select<8, 1>(24));
              float v94_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v96_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v96_data + (v93_data * v94_data));
              float v99_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v101_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v101_data + (v93_data * v99_data));
              float v104_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v106_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v106_data + (v93_data * v104_data));
              float v109_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v111_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v111_data + (v93_data * v109_data));
              tensorforge::intel_esimd::simd<float, 8> v113_data(r0.template select<8, 1>(32));
              float v114_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v116_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v116_data + (v113_data * v114_data));
              float v119_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v121_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v121_data + (v113_data * v119_data));
              float v124_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v126_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v126_data + (v113_data * v124_data));
              float v129_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v131_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v131_data + (v113_data * v129_data));
              tensorforge::intel_esimd::simd<float, 8> v133_data(r0.template select<8, 1>(40));
              float v134_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v136_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v136_data + (v133_data * v134_data));
              float v139_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v141_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v141_data + (v133_data * v139_data));
              float v144_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v146_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v146_data + (v133_data * v144_data));
              float v149_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v151_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v151_data + (v133_data * v149_data));
              tensorforge::intel_esimd::simd<float, 8> v153_data(r0.template select<8, 1>(48));
              float v154_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v156_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v156_data + (v153_data * v154_data));
              float v159_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v161_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v161_data + (v153_data * v159_data));
              float v164_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v166_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v166_data + (v153_data * v164_data));
              float v169_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v171_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v171_data + (v153_data * v169_data));
              tensorforge::intel_esimd::simd<float, 8> v173_data(r0.template select<8, 1>(56));
              float v174_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v176_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v176_data + (v173_data * v174_data));
              float v179_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v181_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v181_data + (v173_data * v179_data));
              float v184_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v186_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v186_data + (v173_data * v184_data));
              float v189_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v191_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v191_data + (v173_data * v189_data));
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v193_i0 = 0; v193_i0 < 1; ++v193_i0) {
                int32_t v195_a = v193_i0 * 8;
                #pragma unroll
                for (int32_t v194_i1 = 0; v194_i1 < 4; ++v194_i1) {
                  int32_t v197_a = v195_a + (v194_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v198_data(r1.template select<8, 1>(v197_a));
                  tensorforge::slmStore<float, 8>(s1 + (v197_a), v198_data);
                }
              }
              tensorforge::intel_esimd::simd<float, 32> r2(0.0f);
              // ir2 = +(r0 * s2)
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 32> ir2(0.0f);
              tensorforge::intel_esimd::simd<float, 32> s2_w1 = tensorforge::slmLoad<float, 32>(s2 + 0);
              float v205_data = s2_w1[0];
              tensorforge::intel_esimd::simd<float, 8> v207_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v207_data + (v33_data * v205_data));
              float v210_data = s2_w1[8];
              tensorforge::intel_esimd::simd<float, 8> v212_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v212_data + (v33_data * v210_data));
              float v215_data = s2_w1[16];
              tensorforge::intel_esimd::simd<float, 8> v217_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v217_data + (v33_data * v215_data));
              float v220_data = s2_w1[24];
              tensorforge::intel_esimd::simd<float, 8> v222_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v222_data + (v33_data * v220_data));
              float v225_data = s2_w1[1];
              tensorforge::intel_esimd::simd<float, 8> v227_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v227_data + (v53_data * v225_data));
              float v230_data = s2_w1[9];
              tensorforge::intel_esimd::simd<float, 8> v232_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v232_data + (v53_data * v230_data));
              float v235_data = s2_w1[17];
              tensorforge::intel_esimd::simd<float, 8> v237_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v237_data + (v53_data * v235_data));
              float v240_data = s2_w1[25];
              tensorforge::intel_esimd::simd<float, 8> v242_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v242_data + (v53_data * v240_data));
              float v245_data = s2_w1[2];
              tensorforge::intel_esimd::simd<float, 8> v247_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v247_data + (v73_data * v245_data));
              float v250_data = s2_w1[10];
              tensorforge::intel_esimd::simd<float, 8> v252_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v252_data + (v73_data * v250_data));
              float v255_data = s2_w1[18];
              tensorforge::intel_esimd::simd<float, 8> v257_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v257_data + (v73_data * v255_data));
              float v260_data = s2_w1[26];
              tensorforge::intel_esimd::simd<float, 8> v262_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v262_data + (v73_data * v260_data));
              float v265_data = s2_w1[3];
              tensorforge::intel_esimd::simd<float, 8> v267_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v267_data + (v93_data * v265_data));
              float v270_data = s2_w1[11];
              tensorforge::intel_esimd::simd<float, 8> v272_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v272_data + (v93_data * v270_data));
              float v275_data = s2_w1[19];
              tensorforge::intel_esimd::simd<float, 8> v277_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v277_data + (v93_data * v275_data));
              float v280_data = s2_w1[27];
              tensorforge::intel_esimd::simd<float, 8> v282_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v282_data + (v93_data * v280_data));
              float v285_data = s2_w1[4];
              tensorforge::intel_esimd::simd<float, 8> v287_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v287_data + (v113_data * v285_data));
              float v290_data = s2_w1[12];
              tensorforge::intel_esimd::simd<float, 8> v292_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v292_data + (v113_data * v290_data));
              float v295_data = s2_w1[20];
              tensorforge::intel_esimd::simd<float, 8> v297_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v297_data + (v113_data * v295_data));
              float v300_data = s2_w1[28];
              tensorforge::intel_esimd::simd<float, 8> v302_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v302_data + (v113_data * v300_data));
              float v305_data = s2_w1[5];
              tensorforge::intel_esimd::simd<float, 8> v307_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v307_data + (v133_data * v305_data));
              float v310_data = s2_w1[13];
              tensorforge::intel_esimd::simd<float, 8> v312_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v312_data + (v133_data * v310_data));
              float v315_data = s2_w1[21];
              tensorforge::intel_esimd::simd<float, 8> v317_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v317_data + (v133_data * v315_data));
              float v320_data = s2_w1[29];
              tensorforge::intel_esimd::simd<float, 8> v322_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v322_data + (v133_data * v320_data));
              float v325_data = s2_w1[6];
              tensorforge::intel_esimd::simd<float, 8> v327_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v327_data + (v153_data * v325_data));
              float v330_data = s2_w1[14];
              tensorforge::intel_esimd::simd<float, 8> v332_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v332_data + (v153_data * v330_data));
              float v335_data = s2_w1[22];
              tensorforge::intel_esimd::simd<float, 8> v337_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v337_data + (v153_data * v335_data));
              float v340_data = s2_w1[30];
              tensorforge::intel_esimd::simd<float, 8> v342_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v342_data + (v153_data * v340_data));
              float v345_data = s2_w1[7];
              tensorforge::intel_esimd::simd<float, 8> v347_data(ir2.template select<8, 1>(0));
              ir2.template select<8, 1>(0) = (v347_data + (v173_data * v345_data));
              float v350_data = s2_w1[15];
              tensorforge::intel_esimd::simd<float, 8> v352_data(ir2.template select<8, 1>(8));
              ir2.template select<8, 1>(8) = (v352_data + (v173_data * v350_data));
              float v355_data = s2_w1[23];
              tensorforge::intel_esimd::simd<float, 8> v357_data(ir2.template select<8, 1>(16));
              ir2.template select<8, 1>(16) = (v357_data + (v173_data * v355_data));
              float v360_data = s2_w1[31];
              tensorforge::intel_esimd::simd<float, 8> v362_data(ir2.template select<8, 1>(24));
              ir2.template select<8, 1>(24) = (v362_data + (v173_data * v360_data));
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
          }
        }
      }
    });
  });
}


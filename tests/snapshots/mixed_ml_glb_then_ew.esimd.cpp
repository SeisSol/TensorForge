// === base name ===
kernel_295f524a4f6458b9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_295f524a4f6458b9 = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_295f524a4f6458b9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_295f524a4f6458b9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_295f524a4f6458b9(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_295f524a4f6458b9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_295f524a4f6458b9(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_295f524a4f6458b9(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_295f524a4f6458b9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   C = abs(M)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"M","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (72 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 64 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
                int32_t v23_lead = v21_i0 * 8;
                #pragma unroll
                for (int32_t v22_i1 = 0; v22_i1 < 8; ++v22_i1) {
                  int32_t v26_a = v23_lead + (v22_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v27_data;
                  v27_data.copy_from(glb_m1 + (v26_a));
                  r0.template select<8, 1>(v26_a) = v27_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v29_ld);
              tensorforge::intel_esimd::simd<float, 32> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 32), v30_ld);
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 64> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v33_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s0_w0 = tensorforge::slmLoad<float, 64>(s0 + 0);
              float v34_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v36_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v36_data + (v33_data * v34_data));
              float v39_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v41_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v41_data + (v33_data * v39_data));
              float v44_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v46_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v46_data + (v33_data * v44_data));
              float v49_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v51_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v51_data + (v33_data * v49_data));
              float v54_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v56_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v56_data + (v33_data * v54_data));
              float v59_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v61_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v61_data + (v33_data * v59_data));
              float v64_data = s0_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v66_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v66_data + (v33_data * v64_data));
              float v69_data = s0_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v71_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v71_data + (v33_data * v69_data));
              tensorforge::intel_esimd::simd<float, 8> v73_data(r0.template select<8, 1>(8));
              float v74_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v76_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v76_data + (v73_data * v74_data));
              float v79_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v81_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v81_data + (v73_data * v79_data));
              float v84_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v86_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v86_data + (v73_data * v84_data));
              float v89_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v91_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v91_data + (v73_data * v89_data));
              float v94_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v96_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v96_data + (v73_data * v94_data));
              float v99_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v101_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v101_data + (v73_data * v99_data));
              float v104_data = s0_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v106_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v106_data + (v73_data * v104_data));
              float v109_data = s0_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v111_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v111_data + (v73_data * v109_data));
              tensorforge::intel_esimd::simd<float, 8> v113_data(r0.template select<8, 1>(16));
              float v114_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v116_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v116_data + (v113_data * v114_data));
              float v119_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v121_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v121_data + (v113_data * v119_data));
              float v124_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v126_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v126_data + (v113_data * v124_data));
              float v129_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v131_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v131_data + (v113_data * v129_data));
              float v134_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v136_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v136_data + (v113_data * v134_data));
              float v139_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v141_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v141_data + (v113_data * v139_data));
              float v144_data = s0_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v146_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v146_data + (v113_data * v144_data));
              float v149_data = s0_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v151_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v151_data + (v113_data * v149_data));
              tensorforge::intel_esimd::simd<float, 8> v153_data(r0.template select<8, 1>(24));
              float v154_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v156_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v156_data + (v153_data * v154_data));
              float v159_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v161_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v161_data + (v153_data * v159_data));
              float v164_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v166_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v166_data + (v153_data * v164_data));
              float v169_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v171_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v171_data + (v153_data * v169_data));
              float v174_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v176_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v176_data + (v153_data * v174_data));
              float v179_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v181_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v181_data + (v153_data * v179_data));
              float v184_data = s0_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v186_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v186_data + (v153_data * v184_data));
              float v189_data = s0_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v191_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v191_data + (v153_data * v189_data));
              tensorforge::intel_esimd::simd<float, 8> v193_data(r0.template select<8, 1>(32));
              float v194_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v196_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v196_data + (v193_data * v194_data));
              float v199_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v201_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v201_data + (v193_data * v199_data));
              float v204_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v206_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v206_data + (v193_data * v204_data));
              float v209_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v211_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v211_data + (v193_data * v209_data));
              float v214_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v216_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v216_data + (v193_data * v214_data));
              float v219_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v221_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v221_data + (v193_data * v219_data));
              float v224_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v226_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v226_data + (v193_data * v224_data));
              float v229_data = s0_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v231_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v231_data + (v193_data * v229_data));
              tensorforge::intel_esimd::simd<float, 8> v233_data(r0.template select<8, 1>(40));
              float v234_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v236_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v236_data + (v233_data * v234_data));
              float v239_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v241_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v241_data + (v233_data * v239_data));
              float v244_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v246_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v246_data + (v233_data * v244_data));
              float v249_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v251_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v251_data + (v233_data * v249_data));
              float v254_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v256_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v256_data + (v233_data * v254_data));
              float v259_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v261_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v261_data + (v233_data * v259_data));
              float v264_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v266_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v266_data + (v233_data * v264_data));
              float v269_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v271_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v271_data + (v233_data * v269_data));
              tensorforge::intel_esimd::simd<float, 8> v273_data(r0.template select<8, 1>(48));
              float v274_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v276_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v276_data + (v273_data * v274_data));
              float v279_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v281_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v281_data + (v273_data * v279_data));
              float v284_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v286_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v286_data + (v273_data * v284_data));
              float v289_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v291_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v291_data + (v273_data * v289_data));
              float v294_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v296_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v296_data + (v273_data * v294_data));
              float v299_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v301_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v301_data + (v273_data * v299_data));
              float v304_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v306_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v306_data + (v273_data * v304_data));
              float v309_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v311_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v311_data + (v273_data * v309_data));
              tensorforge::intel_esimd::simd<float, 8> v313_data(r0.template select<8, 1>(56));
              float v314_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v316_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v316_data + (v313_data * v314_data));
              float v319_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v321_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v321_data + (v313_data * v319_data));
              float v324_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v326_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v326_data + (v313_data * v324_data));
              float v329_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v331_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v331_data + (v313_data * v329_data));
              float v334_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v336_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v336_data + (v313_data * v334_data));
              float v339_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v341_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v341_data + (v313_data * v339_data));
              float v344_data = s0_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v346_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v346_data + (v313_data * v344_data));
              float v349_data = s0_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v351_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v351_data + (v313_data * v349_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v353_n0 = 0; v353_n0 < 1; ++v353_n0) {
                int32_t v355_a = v353_n0 * 8;
                #pragma unroll
                for (int32_t v354_n1 = 0; v354_n1 < 8; ++v354_n1) {
                  int32_t v357_a = v355_a + (v354_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v358_data(ir1.template select<8, 1>(v357_a));
                  r1.template select<8, 1>(v357_a) = v358_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v359_i0 = 0; v359_i0 < 1; ++v359_i0) {
                int32_t v361_a = v359_i0 * 8;
                #pragma unroll
                for (int32_t v360_i1 = 0; v360_i1 < 8; ++v360_i1) {
                  int32_t v363_a = v361_a + (v360_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v364_data(r1.template select<8, 1>(v363_a));
                  v364_data.copy_to(glb_m0 + (v363_a));
                }
              }
              // glb_m3 = abs(glb_m0)
              #pragma unroll
              for (int32_t v367_k0 = 0; v367_k0 < 1; ++v367_k0) {
                int32_t v369_lead = v367_k0 * 8;
                #pragma unroll
                for (int32_t v368_k1 = 0; v368_k1 < 8; ++v368_k1) {
                  int32_t v372_a = v369_lead + (v368_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v373_data;
                  v373_data.copy_from(glb_m0 + (v372_a));
                  (tensorforge::intel_esimd::abs(v373_data)).copy_to(glb_m3 + (v372_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


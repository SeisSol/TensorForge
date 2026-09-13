// === base name ===
kernel_84e6496a532d9349

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_84e6496a532d9349 = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_84e6496a532d9349(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_84e6496a532d9349(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_84e6496a532d9349(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_84e6496a532d9349(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_84e6496a532d9349(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_84e6496a532d9349(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_84e6496a532d9349(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
        // operations:
        //   TMP = abs(A)
        //   m1[i,j] = t0[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (72 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 64 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 64 + 0 + m2_extraOffset];
              // s1 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 4 * 0 + 0), v17_ld);
              tensorforge::intel_esimd::simd<float, 32> v18_ld;
              v18_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 4 * 0 + 32), v18_ld);
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = abs(glb_m0)
              #pragma unroll
              for (int32_t v20_k0 = 0; v20_k0 < 1; ++v20_k0) {
                int32_t v22_lead = v20_k0 * 8;
                #pragma unroll
                for (int32_t v21_k1 = 0; v21_k1 < 8; ++v21_k1) {
                  int32_t v25_a = v22_lead + (v21_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v26_data;
                  v26_data.copy_from(glb_m0 + (v25_a));
                  r0.template select<8, 1>(v25_a) = (tensorforge::intel_esimd::abs(v26_data));
                }
              }
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // ir1 = +(r0 * s1)
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 64> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v31_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s1_w0 = tensorforge::slmLoad<float, 64>(s1 + 0);
              float v32_data = s1_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v34_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v34_data + (v31_data * v32_data));
              float v37_data = s1_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v39_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v39_data + (v31_data * v37_data));
              float v42_data = s1_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v44_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v44_data + (v31_data * v42_data));
              float v47_data = s1_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v49_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v49_data + (v31_data * v47_data));
              float v52_data = s1_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v54_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v54_data + (v31_data * v52_data));
              float v57_data = s1_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v59_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v59_data + (v31_data * v57_data));
              float v62_data = s1_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v64_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v64_data + (v31_data * v62_data));
              float v67_data = s1_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v69_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v69_data + (v31_data * v67_data));
              tensorforge::intel_esimd::simd<float, 8> v71_data(r0.template select<8, 1>(8));
              float v72_data = s1_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v74_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v74_data + (v71_data * v72_data));
              float v77_data = s1_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v79_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v79_data + (v71_data * v77_data));
              float v82_data = s1_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v84_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v84_data + (v71_data * v82_data));
              float v87_data = s1_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v89_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v89_data + (v71_data * v87_data));
              float v92_data = s1_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v94_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v94_data + (v71_data * v92_data));
              float v97_data = s1_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v99_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v99_data + (v71_data * v97_data));
              float v102_data = s1_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v104_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v104_data + (v71_data * v102_data));
              float v107_data = s1_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v109_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v109_data + (v71_data * v107_data));
              tensorforge::intel_esimd::simd<float, 8> v111_data(r0.template select<8, 1>(16));
              float v112_data = s1_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v114_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v114_data + (v111_data * v112_data));
              float v117_data = s1_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v119_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v119_data + (v111_data * v117_data));
              float v122_data = s1_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v124_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v124_data + (v111_data * v122_data));
              float v127_data = s1_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v129_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v129_data + (v111_data * v127_data));
              float v132_data = s1_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v134_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v134_data + (v111_data * v132_data));
              float v137_data = s1_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v139_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v139_data + (v111_data * v137_data));
              float v142_data = s1_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v144_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v144_data + (v111_data * v142_data));
              float v147_data = s1_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v149_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v149_data + (v111_data * v147_data));
              tensorforge::intel_esimd::simd<float, 8> v151_data(r0.template select<8, 1>(24));
              float v152_data = s1_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v154_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v154_data + (v151_data * v152_data));
              float v157_data = s1_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v159_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v159_data + (v151_data * v157_data));
              float v162_data = s1_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v164_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v164_data + (v151_data * v162_data));
              float v167_data = s1_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v169_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v169_data + (v151_data * v167_data));
              float v172_data = s1_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v174_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v174_data + (v151_data * v172_data));
              float v177_data = s1_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v179_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v179_data + (v151_data * v177_data));
              float v182_data = s1_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v184_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v184_data + (v151_data * v182_data));
              float v187_data = s1_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v189_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v189_data + (v151_data * v187_data));
              tensorforge::intel_esimd::simd<float, 8> v191_data(r0.template select<8, 1>(32));
              float v192_data = s1_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v194_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v194_data + (v191_data * v192_data));
              float v197_data = s1_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v199_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v199_data + (v191_data * v197_data));
              float v202_data = s1_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v204_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v204_data + (v191_data * v202_data));
              float v207_data = s1_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v209_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v209_data + (v191_data * v207_data));
              float v212_data = s1_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v214_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v214_data + (v191_data * v212_data));
              float v217_data = s1_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v219_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v219_data + (v191_data * v217_data));
              float v222_data = s1_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v224_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v224_data + (v191_data * v222_data));
              float v227_data = s1_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v229_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v229_data + (v191_data * v227_data));
              tensorforge::intel_esimd::simd<float, 8> v231_data(r0.template select<8, 1>(40));
              float v232_data = s1_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v234_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v234_data + (v231_data * v232_data));
              float v237_data = s1_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v239_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v239_data + (v231_data * v237_data));
              float v242_data = s1_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v244_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v244_data + (v231_data * v242_data));
              float v247_data = s1_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v249_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v249_data + (v231_data * v247_data));
              float v252_data = s1_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v254_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v254_data + (v231_data * v252_data));
              float v257_data = s1_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v259_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v259_data + (v231_data * v257_data));
              float v262_data = s1_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v264_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v264_data + (v231_data * v262_data));
              float v267_data = s1_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v269_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v269_data + (v231_data * v267_data));
              tensorforge::intel_esimd::simd<float, 8> v271_data(r0.template select<8, 1>(48));
              float v272_data = s1_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v274_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v274_data + (v271_data * v272_data));
              float v277_data = s1_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v279_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v279_data + (v271_data * v277_data));
              float v282_data = s1_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v284_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v284_data + (v271_data * v282_data));
              float v287_data = s1_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v289_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v289_data + (v271_data * v287_data));
              float v292_data = s1_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v294_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v294_data + (v271_data * v292_data));
              float v297_data = s1_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v299_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v299_data + (v271_data * v297_data));
              float v302_data = s1_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v304_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v304_data + (v271_data * v302_data));
              float v307_data = s1_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v309_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v309_data + (v271_data * v307_data));
              tensorforge::intel_esimd::simd<float, 8> v311_data(r0.template select<8, 1>(56));
              float v312_data = s1_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v314_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v314_data + (v311_data * v312_data));
              float v317_data = s1_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v319_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v319_data + (v311_data * v317_data));
              float v322_data = s1_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v324_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v324_data + (v311_data * v322_data));
              float v327_data = s1_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v329_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v329_data + (v311_data * v327_data));
              float v332_data = s1_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v334_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v334_data + (v311_data * v332_data));
              float v337_data = s1_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v339_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v339_data + (v311_data * v337_data));
              float v342_data = s1_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v344_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v344_data + (v311_data * v342_data));
              float v347_data = s1_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v349_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v349_data + (v311_data * v347_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v351_n0 = 0; v351_n0 < 1; ++v351_n0) {
                int32_t v353_a = v351_n0 * 8;
                #pragma unroll
                for (int32_t v352_n1 = 0; v352_n1 < 8; ++v352_n1) {
                  int32_t v355_a = v353_a + (v352_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v356_data(ir1.template select<8, 1>(v355_a));
                  r1.template select<8, 1>(v355_a) = v356_data;
                }
              }
              // glb_m1 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v357_i0 = 0; v357_i0 < 1; ++v357_i0) {
                int32_t v359_a = v357_i0 * 8;
                #pragma unroll
                for (int32_t v358_i1 = 0; v358_i1 < 8; ++v358_i1) {
                  int32_t v361_a = v359_a + (v358_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v362_data(r1.template select<8, 1>(v361_a));
                  v362_data.copy_to(glb_m1 + (v361_a));
                }
              }
            }
            tensorforge::prefetchL2<64>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


// === base name ===
kernel_e3caf16b37aac009

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e3caf16b37aac009 = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e3caf16b37aac009(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e3caf16b37aac009(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e3caf16b37aac009(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_e3caf16b37aac009(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e3caf16b37aac009(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e3caf16b37aac009(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e3caf16b37aac009(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v11_batchId0 * 64 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v26_lead = v24_i0 * 8;
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                  int32_t v29_a = v26_lead + (v25_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v30_data;
                  v30_data.copy_from(glb_m1 + (v29_a));
                  r0.template select<8, 1>(v29_a) = v30_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v32_ld);
              tensorforge::intel_esimd::simd<float, 32> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 32), v33_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 64> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v36_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s0_w0 = tensorforge::slmLoad<float, 64>(s0 + 0);
              float v37_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v39_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v39_data + (v36_data * v37_data));
              float v42_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v44_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v44_data + (v36_data * v42_data));
              float v47_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v49_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v49_data + (v36_data * v47_data));
              float v52_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v54_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v54_data + (v36_data * v52_data));
              float v57_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v59_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v59_data + (v36_data * v57_data));
              float v62_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v64_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v64_data + (v36_data * v62_data));
              float v67_data = s0_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v69_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v69_data + (v36_data * v67_data));
              float v72_data = s0_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v74_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v74_data + (v36_data * v72_data));
              tensorforge::intel_esimd::simd<float, 8> v76_data(r0.template select<8, 1>(8));
              float v77_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v79_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v79_data + (v76_data * v77_data));
              float v82_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v84_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v84_data + (v76_data * v82_data));
              float v87_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v89_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v89_data + (v76_data * v87_data));
              float v92_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v94_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v94_data + (v76_data * v92_data));
              float v97_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v99_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v99_data + (v76_data * v97_data));
              float v102_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v104_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v104_data + (v76_data * v102_data));
              float v107_data = s0_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v109_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v109_data + (v76_data * v107_data));
              float v112_data = s0_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v114_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v114_data + (v76_data * v112_data));
              tensorforge::intel_esimd::simd<float, 8> v116_data(r0.template select<8, 1>(16));
              float v117_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v119_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v119_data + (v116_data * v117_data));
              float v122_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v124_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v124_data + (v116_data * v122_data));
              float v127_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v129_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v129_data + (v116_data * v127_data));
              float v132_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v134_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v134_data + (v116_data * v132_data));
              float v137_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v139_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v139_data + (v116_data * v137_data));
              float v142_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v144_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v144_data + (v116_data * v142_data));
              float v147_data = s0_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v149_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v149_data + (v116_data * v147_data));
              float v152_data = s0_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v154_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v154_data + (v116_data * v152_data));
              tensorforge::intel_esimd::simd<float, 8> v156_data(r0.template select<8, 1>(24));
              float v157_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v159_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v159_data + (v156_data * v157_data));
              float v162_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v164_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v164_data + (v156_data * v162_data));
              float v167_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v169_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v169_data + (v156_data * v167_data));
              float v172_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v174_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v174_data + (v156_data * v172_data));
              float v177_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v179_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v179_data + (v156_data * v177_data));
              float v182_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v184_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v184_data + (v156_data * v182_data));
              float v187_data = s0_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v189_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v189_data + (v156_data * v187_data));
              float v192_data = s0_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v194_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v194_data + (v156_data * v192_data));
              tensorforge::intel_esimd::simd<float, 8> v196_data(r0.template select<8, 1>(32));
              float v197_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v199_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v199_data + (v196_data * v197_data));
              float v202_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v204_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v204_data + (v196_data * v202_data));
              float v207_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v209_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v209_data + (v196_data * v207_data));
              float v212_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v214_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v214_data + (v196_data * v212_data));
              float v217_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v219_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v219_data + (v196_data * v217_data));
              float v222_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v224_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v224_data + (v196_data * v222_data));
              float v227_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v229_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v229_data + (v196_data * v227_data));
              float v232_data = s0_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v234_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v234_data + (v196_data * v232_data));
              tensorforge::intel_esimd::simd<float, 8> v236_data(r0.template select<8, 1>(40));
              float v237_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v239_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v239_data + (v236_data * v237_data));
              float v242_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v244_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v244_data + (v236_data * v242_data));
              float v247_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v249_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v249_data + (v236_data * v247_data));
              float v252_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v254_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v254_data + (v236_data * v252_data));
              float v257_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v259_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v259_data + (v236_data * v257_data));
              float v262_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v264_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v264_data + (v236_data * v262_data));
              float v267_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v269_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v269_data + (v236_data * v267_data));
              float v272_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v274_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v274_data + (v236_data * v272_data));
              tensorforge::intel_esimd::simd<float, 8> v276_data(r0.template select<8, 1>(48));
              float v277_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v279_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v279_data + (v276_data * v277_data));
              float v282_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v284_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v284_data + (v276_data * v282_data));
              float v287_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v289_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v289_data + (v276_data * v287_data));
              float v292_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v294_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v294_data + (v276_data * v292_data));
              float v297_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v299_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v299_data + (v276_data * v297_data));
              float v302_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v304_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v304_data + (v276_data * v302_data));
              float v307_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v309_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v309_data + (v276_data * v307_data));
              float v312_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v314_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v314_data + (v276_data * v312_data));
              tensorforge::intel_esimd::simd<float, 8> v316_data(r0.template select<8, 1>(56));
              float v317_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v319_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v319_data + (v316_data * v317_data));
              float v322_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v324_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v324_data + (v316_data * v322_data));
              float v327_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v329_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v329_data + (v316_data * v327_data));
              float v332_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v334_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v334_data + (v316_data * v332_data));
              float v337_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v339_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v339_data + (v316_data * v337_data));
              float v342_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v344_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v344_data + (v316_data * v342_data));
              float v347_data = s0_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v349_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v349_data + (v316_data * v347_data));
              float v352_data = s0_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v354_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v354_data + (v316_data * v352_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v356_n0 = 0; v356_n0 < 1; ++v356_n0) {
                int32_t v358_a = v356_n0 * 8;
                #pragma unroll
                for (int32_t v357_n1 = 0; v357_n1 < 8; ++v357_n1) {
                  int32_t v360_a = v358_a + (v357_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v361_data(ir1.template select<8, 1>(v360_a));
                  r1.template select<8, 1>(v360_a) = v361_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v362_i0 = 0; v362_i0 < 1; ++v362_i0) {
                int32_t v364_a = v362_i0 * 8;
                #pragma unroll
                for (int32_t v363_i1 = 0; v363_i1 < 8; ++v363_i1) {
                  int32_t v366_a = v364_a + (v363_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v367_data(r1.template select<8, 1>(v366_a));
                  v367_data.copy_to(glb_m0 + (v366_a));
                }
              }
              // glb_m3 = abs(glb_m0)
              #pragma unroll
              for (int32_t v370_k0 = 0; v370_k0 < 1; ++v370_k0) {
                int32_t v372_lead = v370_k0 * 8;
                #pragma unroll
                for (int32_t v371_k1 = 0; v371_k1 < 8; ++v371_k1) {
                  int32_t v375_a = v372_lead + (v371_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v376_data;
                  v376_data.copy_from(glb_m0 + (v375_a));
                  (tensorforge::intel_esimd::abs(v376_data)).copy_to(glb_m3 + (v375_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


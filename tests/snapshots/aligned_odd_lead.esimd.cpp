// === base name ===
kernel_b3b0d82030bc422a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b3b0d82030bc422a = {{1, 8, 1}, 32, 35, 1, 8, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b3b0d82030bc422a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b3b0d82030bc422a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b3b0d82030bc422a(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b3b0d82030bc422a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b3b0d82030bc422a(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b3b0d82030bc422a(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b3b0d82030bc422a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<256 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (35 active) x 8 per block = block 1x8x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 35×4(35×4) {0..35}×{0..4} strided
        //   m1 35×8(35×8) {0..35}×{0..8} strided
        //   m2 8×4(8×4) {0..8}×{0..4} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":35,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[35,4]],"name":"m0","ordered":false,"parts":1,"shape":[35,4],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[35,8]],"name":"m1","ordered":false,"parts":1,"shape":[35,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,4]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[35,4]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[35,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[35,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (32 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 32 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 512> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 32;
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v26_data;
                  v26_data.copy_from(glb_m1 + ((v22_lead + (v21_i1 * 35))));
                  r0.template select<32, 1>((v22_lead + (v21_i1 * 64))) = v26_data;
                }
              }
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
                tensorforge::intel_esimd::simd<float, 3> v35_data;
                v35_data.copy_from(glb_m1 + ((32_i32 + (v29_i1 * 35))));
                r0.template select<3, 1>((32 + (v29_i1 * 64))) = v35_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v38_ld;
              v38_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v38_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 35), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v41_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> s0_w0 = tensorforge::slmLoad<float, 32>(s0 + 0);
              float v42_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v44_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v44_data + (v41_data * v42_data));
              float v47_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v49_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v49_data + (v41_data * v47_data));
              float v52_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v54_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v54_data + (v41_data * v52_data));
              float v57_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v59_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v59_data + (v41_data * v57_data));
              tensorforge::intel_esimd::simd<float, 32> v61_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v64_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v64_data + (v61_data * v42_data));
              tensorforge::intel_esimd::simd<float, 32> v69_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v69_data + (v61_data * v47_data));
              tensorforge::intel_esimd::simd<float, 32> v74_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v74_data + (v61_data * v52_data));
              tensorforge::intel_esimd::simd<float, 32> v79_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v79_data + (v61_data * v57_data));
              tensorforge::intel_esimd::simd<float, 32> v81_data(r0.template select<32, 1>(64));
              float v82_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v84_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v84_data + (v81_data * v82_data));
              float v87_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v89_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v89_data + (v81_data * v87_data));
              float v92_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v94_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v94_data + (v81_data * v92_data));
              float v97_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v99_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v99_data + (v81_data * v97_data));
              tensorforge::intel_esimd::simd<float, 32> v101_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v104_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v104_data + (v101_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v109_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v109_data + (v101_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v114_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v114_data + (v101_data * v92_data));
              tensorforge::intel_esimd::simd<float, 32> v119_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v119_data + (v101_data * v97_data));
              tensorforge::intel_esimd::simd<float, 32> v121_data(r0.template select<32, 1>(128));
              float v122_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v124_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v124_data + (v121_data * v122_data));
              float v127_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v129_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v129_data + (v121_data * v127_data));
              float v132_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v134_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v134_data + (v121_data * v132_data));
              float v137_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v139_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v139_data + (v121_data * v137_data));
              tensorforge::intel_esimd::simd<float, 32> v141_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v144_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v144_data + (v141_data * v122_data));
              tensorforge::intel_esimd::simd<float, 32> v149_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v149_data + (v141_data * v127_data));
              tensorforge::intel_esimd::simd<float, 32> v154_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v154_data + (v141_data * v132_data));
              tensorforge::intel_esimd::simd<float, 32> v159_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v159_data + (v141_data * v137_data));
              tensorforge::intel_esimd::simd<float, 32> v161_data(r0.template select<32, 1>(192));
              float v162_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v164_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v164_data + (v161_data * v162_data));
              float v167_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v169_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v169_data + (v161_data * v167_data));
              float v172_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v174_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v174_data + (v161_data * v172_data));
              float v177_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v179_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v179_data + (v161_data * v177_data));
              tensorforge::intel_esimd::simd<float, 32> v181_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v184_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v184_data + (v181_data * v162_data));
              tensorforge::intel_esimd::simd<float, 32> v189_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v189_data + (v181_data * v167_data));
              tensorforge::intel_esimd::simd<float, 32> v194_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v194_data + (v181_data * v172_data));
              tensorforge::intel_esimd::simd<float, 32> v199_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v199_data + (v181_data * v177_data));
              tensorforge::intel_esimd::simd<float, 32> v201_data(r0.template select<32, 1>(256));
              float v202_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v204_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v204_data + (v201_data * v202_data));
              float v207_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v209_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v209_data + (v201_data * v207_data));
              float v212_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v214_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v214_data + (v201_data * v212_data));
              float v217_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v219_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v219_data + (v201_data * v217_data));
              tensorforge::intel_esimd::simd<float, 32> v221_data(r0.template select<32, 1>(288));
              tensorforge::intel_esimd::simd<float, 32> v224_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v224_data + (v221_data * v202_data));
              tensorforge::intel_esimd::simd<float, 32> v229_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v229_data + (v221_data * v207_data));
              tensorforge::intel_esimd::simd<float, 32> v234_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v234_data + (v221_data * v212_data));
              tensorforge::intel_esimd::simd<float, 32> v239_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v239_data + (v221_data * v217_data));
              tensorforge::intel_esimd::simd<float, 32> v241_data(r0.template select<32, 1>(320));
              float v242_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v244_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v244_data + (v241_data * v242_data));
              float v247_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v249_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v249_data + (v241_data * v247_data));
              float v252_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v254_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v254_data + (v241_data * v252_data));
              float v257_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v259_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v259_data + (v241_data * v257_data));
              tensorforge::intel_esimd::simd<float, 32> v261_data(r0.template select<32, 1>(352));
              tensorforge::intel_esimd::simd<float, 32> v264_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v264_data + (v261_data * v242_data));
              tensorforge::intel_esimd::simd<float, 32> v269_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v269_data + (v261_data * v247_data));
              tensorforge::intel_esimd::simd<float, 32> v274_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v274_data + (v261_data * v252_data));
              tensorforge::intel_esimd::simd<float, 32> v279_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v279_data + (v261_data * v257_data));
              tensorforge::intel_esimd::simd<float, 32> v281_data(r0.template select<32, 1>(384));
              float v282_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v284_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v284_data + (v281_data * v282_data));
              float v287_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v289_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v289_data + (v281_data * v287_data));
              float v292_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v294_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v294_data + (v281_data * v292_data));
              float v297_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v299_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v299_data + (v281_data * v297_data));
              tensorforge::intel_esimd::simd<float, 32> v301_data(r0.template select<32, 1>(416));
              tensorforge::intel_esimd::simd<float, 32> v304_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v304_data + (v301_data * v282_data));
              tensorforge::intel_esimd::simd<float, 32> v309_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v309_data + (v301_data * v287_data));
              tensorforge::intel_esimd::simd<float, 32> v314_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v314_data + (v301_data * v292_data));
              tensorforge::intel_esimd::simd<float, 32> v319_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v319_data + (v301_data * v297_data));
              tensorforge::intel_esimd::simd<float, 32> v321_data(r0.template select<32, 1>(448));
              float v322_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v324_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v324_data + (v321_data * v322_data));
              float v327_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v329_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v329_data + (v321_data * v327_data));
              float v332_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v334_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v334_data + (v321_data * v332_data));
              float v337_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v339_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v339_data + (v321_data * v337_data));
              tensorforge::intel_esimd::simd<float, 32> v341_data(r0.template select<32, 1>(480));
              tensorforge::intel_esimd::simd<float, 32> v344_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v344_data + (v341_data * v322_data));
              tensorforge::intel_esimd::simd<float, 32> v349_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v349_data + (v341_data * v327_data));
              tensorforge::intel_esimd::simd<float, 32> v354_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v354_data + (v341_data * v332_data));
              tensorforge::intel_esimd::simd<float, 32> v359_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v359_data + (v341_data * v337_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v361_n0 = 0; v361_n0 < 1; ++v361_n0) {
                int32_t v363_a = v361_n0 * 32;
                #pragma unroll
                for (int32_t v362_n1 = 0; v362_n1 < 4; ++v362_n1) {
                  int32_t v365_a = v363_a + (v362_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v366_data(ir1.template select<32, 1>(v365_a));
                  r1.template select<32, 1>(v365_a) = v366_data;
                }
              }
              #pragma unroll
              for (int32_t v367_n1 = 0; v367_n1 < 4; ++v367_n1) {
                int32_t v369_a = 32 + (v367_n1 * 64);
                tensorforge::intel_esimd::simd<float, 3> v370_data(ir1.template select<3, 1>(v369_a));
                r1.template select<3, 1>(v369_a) = v370_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v371_i0 = 0; v371_i0 < 1; ++v371_i0) {
                int32_t v373_a = v371_i0 * 32;
                #pragma unroll
                for (int32_t v372_i1 = 0; v372_i1 < 4; ++v372_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v376_data(r1.template select<32, 1>((v373_a + (v372_i1 * 64))));
                  v376_data.copy_to(glb_m0 + ((v373_a + (v372_i1 * 35))));
                }
              }
              #pragma unroll
              for (int32_t v380_i1 = 0; v380_i1 < 4; ++v380_i1) {
                tensorforge::intel_esimd::simd<float, 3> v383_data(r1.template select<3, 1>((32 + (v380_i1 * 64))));
                v383_data.copy_to(glb_m0 + ((32_i32 + (v380_i1 * 35))));
              }
            }
          }
        }
      }
    });
  });
}


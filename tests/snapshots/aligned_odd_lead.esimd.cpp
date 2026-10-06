// === base name ===
kernel_d797100d92798264

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d797100d92798264 = {{1, 8, 1}, 32, 35, 1, 8, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d797100d92798264(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d797100d92798264(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d797100d92798264(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_d797100d92798264(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d797100d92798264(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d797100d92798264(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d797100d92798264(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (32);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 32 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 512> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                int32_t v25_lead = v23_i0 * 32;
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v29_data;
                  v29_data.copy_from(glb_m1 + ((v25_lead + (v24_i1 * 35))));
                  r0.template select<32, 1>((v25_lead + (v24_i1 * 64))) = v29_data;
                }
              }
              #pragma unroll
              for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
                tensorforge::intel_esimd::simd<float, 3> v38_data;
                v38_data.copy_from(glb_m1 + ((32_i32 + (v32_i1 * 35))));
                r0.template select<3, 1>((32 + (v32_i1 * 64))) = v38_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v41_ld;
              v41_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v41_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 35), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v44_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> s0_w0 = tensorforge::slmLoad<float, 32>(s0 + 0);
              float v45_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v47_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v47_data + (v44_data * v45_data));
              float v50_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v52_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v52_data + (v44_data * v50_data));
              float v55_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v57_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v57_data + (v44_data * v55_data));
              float v60_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v62_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v62_data + (v44_data * v60_data));
              tensorforge::intel_esimd::simd<float, 32> v64_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v67_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v67_data + (v64_data * v45_data));
              tensorforge::intel_esimd::simd<float, 32> v72_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v72_data + (v64_data * v50_data));
              tensorforge::intel_esimd::simd<float, 32> v77_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v77_data + (v64_data * v55_data));
              tensorforge::intel_esimd::simd<float, 32> v82_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v82_data + (v64_data * v60_data));
              tensorforge::intel_esimd::simd<float, 32> v84_data(r0.template select<32, 1>(64));
              float v85_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v87_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v87_data + (v84_data * v85_data));
              float v90_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v92_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v92_data + (v84_data * v90_data));
              float v95_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v97_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v97_data + (v84_data * v95_data));
              float v100_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v102_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v102_data + (v84_data * v100_data));
              tensorforge::intel_esimd::simd<float, 32> v104_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v107_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v107_data + (v104_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v112_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v112_data + (v104_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v117_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v117_data + (v104_data * v95_data));
              tensorforge::intel_esimd::simd<float, 32> v122_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v122_data + (v104_data * v100_data));
              tensorforge::intel_esimd::simd<float, 32> v124_data(r0.template select<32, 1>(128));
              float v125_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v127_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v127_data + (v124_data * v125_data));
              float v130_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v132_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v132_data + (v124_data * v130_data));
              float v135_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v137_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v137_data + (v124_data * v135_data));
              float v140_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v142_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v142_data + (v124_data * v140_data));
              tensorforge::intel_esimd::simd<float, 32> v144_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v147_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v147_data + (v144_data * v125_data));
              tensorforge::intel_esimd::simd<float, 32> v152_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v152_data + (v144_data * v130_data));
              tensorforge::intel_esimd::simd<float, 32> v157_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v157_data + (v144_data * v135_data));
              tensorforge::intel_esimd::simd<float, 32> v162_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v162_data + (v144_data * v140_data));
              tensorforge::intel_esimd::simd<float, 32> v164_data(r0.template select<32, 1>(192));
              float v165_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v167_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v167_data + (v164_data * v165_data));
              float v170_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v172_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v172_data + (v164_data * v170_data));
              float v175_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v177_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v177_data + (v164_data * v175_data));
              float v180_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v182_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v182_data + (v164_data * v180_data));
              tensorforge::intel_esimd::simd<float, 32> v184_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v187_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v187_data + (v184_data * v165_data));
              tensorforge::intel_esimd::simd<float, 32> v192_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v192_data + (v184_data * v170_data));
              tensorforge::intel_esimd::simd<float, 32> v197_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v197_data + (v184_data * v175_data));
              tensorforge::intel_esimd::simd<float, 32> v202_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v202_data + (v184_data * v180_data));
              tensorforge::intel_esimd::simd<float, 32> v204_data(r0.template select<32, 1>(256));
              float v205_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v207_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v207_data + (v204_data * v205_data));
              float v210_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v212_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v212_data + (v204_data * v210_data));
              float v215_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v217_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v217_data + (v204_data * v215_data));
              float v220_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v222_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v222_data + (v204_data * v220_data));
              tensorforge::intel_esimd::simd<float, 32> v224_data(r0.template select<32, 1>(288));
              tensorforge::intel_esimd::simd<float, 32> v227_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v227_data + (v224_data * v205_data));
              tensorforge::intel_esimd::simd<float, 32> v232_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v232_data + (v224_data * v210_data));
              tensorforge::intel_esimd::simd<float, 32> v237_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v237_data + (v224_data * v215_data));
              tensorforge::intel_esimd::simd<float, 32> v242_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v242_data + (v224_data * v220_data));
              tensorforge::intel_esimd::simd<float, 32> v244_data(r0.template select<32, 1>(320));
              float v245_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v247_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v247_data + (v244_data * v245_data));
              float v250_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v252_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v252_data + (v244_data * v250_data));
              float v255_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v257_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v257_data + (v244_data * v255_data));
              float v260_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v262_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v262_data + (v244_data * v260_data));
              tensorforge::intel_esimd::simd<float, 32> v264_data(r0.template select<32, 1>(352));
              tensorforge::intel_esimd::simd<float, 32> v267_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v267_data + (v264_data * v245_data));
              tensorforge::intel_esimd::simd<float, 32> v272_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v272_data + (v264_data * v250_data));
              tensorforge::intel_esimd::simd<float, 32> v277_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v277_data + (v264_data * v255_data));
              tensorforge::intel_esimd::simd<float, 32> v282_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v282_data + (v264_data * v260_data));
              tensorforge::intel_esimd::simd<float, 32> v284_data(r0.template select<32, 1>(384));
              float v285_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v287_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v287_data + (v284_data * v285_data));
              float v290_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v292_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v292_data + (v284_data * v290_data));
              float v295_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v297_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v297_data + (v284_data * v295_data));
              float v300_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v302_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v302_data + (v284_data * v300_data));
              tensorforge::intel_esimd::simd<float, 32> v304_data(r0.template select<32, 1>(416));
              tensorforge::intel_esimd::simd<float, 32> v307_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v307_data + (v304_data * v285_data));
              tensorforge::intel_esimd::simd<float, 32> v312_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v312_data + (v304_data * v290_data));
              tensorforge::intel_esimd::simd<float, 32> v317_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v317_data + (v304_data * v295_data));
              tensorforge::intel_esimd::simd<float, 32> v322_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v322_data + (v304_data * v300_data));
              tensorforge::intel_esimd::simd<float, 32> v324_data(r0.template select<32, 1>(448));
              float v325_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v327_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v327_data + (v324_data * v325_data));
              float v330_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v332_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v332_data + (v324_data * v330_data));
              float v335_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v337_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v337_data + (v324_data * v335_data));
              float v340_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v342_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v342_data + (v324_data * v340_data));
              tensorforge::intel_esimd::simd<float, 32> v344_data(r0.template select<32, 1>(480));
              tensorforge::intel_esimd::simd<float, 32> v347_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v347_data + (v344_data * v325_data));
              tensorforge::intel_esimd::simd<float, 32> v352_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v352_data + (v344_data * v330_data));
              tensorforge::intel_esimd::simd<float, 32> v357_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v357_data + (v344_data * v335_data));
              tensorforge::intel_esimd::simd<float, 32> v362_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v362_data + (v344_data * v340_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v364_n0 = 0; v364_n0 < 1; ++v364_n0) {
                int32_t v366_a = v364_n0 * 32;
                #pragma unroll
                for (int32_t v365_n1 = 0; v365_n1 < 4; ++v365_n1) {
                  int32_t v368_a = v366_a + (v365_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v369_data(ir1.template select<32, 1>(v368_a));
                  r1.template select<32, 1>(v368_a) = v369_data;
                }
              }
              #pragma unroll
              for (int32_t v370_n1 = 0; v370_n1 < 4; ++v370_n1) {
                int32_t v372_a = 32 + (v370_n1 * 64);
                tensorforge::intel_esimd::simd<float, 3> v373_data(ir1.template select<3, 1>(v372_a));
                r1.template select<3, 1>(v372_a) = v373_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v374_i0 = 0; v374_i0 < 1; ++v374_i0) {
                int32_t v376_a = v374_i0 * 32;
                #pragma unroll
                for (int32_t v375_i1 = 0; v375_i1 < 4; ++v375_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v379_data(r1.template select<32, 1>((v376_a + (v375_i1 * 64))));
                  v379_data.copy_to(glb_m0 + ((v376_a + (v375_i1 * 35))));
                }
              }
              #pragma unroll
              for (int32_t v383_i1 = 0; v383_i1 < 4; ++v383_i1) {
                tensorforge::intel_esimd::simd<float, 3> v386_data(r1.template select<3, 1>((32 + (v383_i1 * 64))));
                v386_data.copy_to(glb_m0 + ((32_i32 + (v383_i1 * 35))));
              }
            }
          }
        }
      }
    });
  });
}


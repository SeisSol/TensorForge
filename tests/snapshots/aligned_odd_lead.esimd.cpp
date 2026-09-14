// === base name ===
kernel_7fa3a9f4fa35ea26

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7fa3a9f4fa35ea26 = {{1, 8, 1}, 32, 35, 1, 8, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7fa3a9f4fa35ea26(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7fa3a9f4fa35ea26(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7fa3a9f4fa35ea26(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_7fa3a9f4fa35ea26(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7fa3a9f4fa35ea26(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7fa3a9f4fa35ea26(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7fa3a9f4fa35ea26(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<256 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (35 active) x 8 per block = block 1x8x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 35×4(35×4) {0..35}×{0..4} strided
        //   m1 35×8(35×8) {0..35}×{0..8} strided
        //   m2 8×4(8×4) {0..8}×{0..4} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":35,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[35,4]],"name":"m0","ordered":false,"parts":1,"shape":[35,4],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[35,8]],"name":"m1","ordered":false,"parts":1,"shape":[35,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,4]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[35,4]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[35,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[35,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (32 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (32);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 32 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 512> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v19_lead = v17_i0 * 32;
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 8; ++v18_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v23_data;
                  v23_data.copy_from(glb_m1 + ((v19_lead + (v18_i1 * 35))));
                  r0.template select<32, 1>((v19_lead + (v18_i1 * 64))) = v23_data;
                }
              }
              #pragma unroll
              for (int32_t v26_i1 = 0; v26_i1 < 8; ++v26_i1) {
                tensorforge::intel_esimd::simd<float, 3> v32_data;
                v32_data.copy_from(glb_m1 + ((32_i32 + (v26_i1 * 35))));
                r0.template select<3, 1>((32 + (v26_i1 * 64))) = v32_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v35_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 35), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v38_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> s0_w0 = tensorforge::slmLoad<float, 32>(s0 + 0);
              float v39_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v41_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v41_data + (v38_data * v39_data));
              float v44_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v46_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v46_data + (v38_data * v44_data));
              float v49_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v51_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v51_data + (v38_data * v49_data));
              float v54_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v56_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v56_data + (v38_data * v54_data));
              tensorforge::intel_esimd::simd<float, 32> v58_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v61_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v61_data + (v58_data * v39_data));
              tensorforge::intel_esimd::simd<float, 32> v66_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v66_data + (v58_data * v44_data));
              tensorforge::intel_esimd::simd<float, 32> v71_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v71_data + (v58_data * v49_data));
              tensorforge::intel_esimd::simd<float, 32> v76_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v76_data + (v58_data * v54_data));
              tensorforge::intel_esimd::simd<float, 32> v78_data(r0.template select<32, 1>(64));
              float v79_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v81_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v81_data + (v78_data * v79_data));
              float v84_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v86_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v86_data + (v78_data * v84_data));
              float v89_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v91_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v91_data + (v78_data * v89_data));
              float v94_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v96_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v96_data + (v78_data * v94_data));
              tensorforge::intel_esimd::simd<float, 32> v98_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v101_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v101_data + (v98_data * v79_data));
              tensorforge::intel_esimd::simd<float, 32> v106_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v106_data + (v98_data * v84_data));
              tensorforge::intel_esimd::simd<float, 32> v111_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v111_data + (v98_data * v89_data));
              tensorforge::intel_esimd::simd<float, 32> v116_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v116_data + (v98_data * v94_data));
              tensorforge::intel_esimd::simd<float, 32> v118_data(r0.template select<32, 1>(128));
              float v119_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v121_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v121_data + (v118_data * v119_data));
              float v124_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v126_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v126_data + (v118_data * v124_data));
              float v129_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v131_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v131_data + (v118_data * v129_data));
              float v134_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v136_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v136_data + (v118_data * v134_data));
              tensorforge::intel_esimd::simd<float, 32> v138_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v141_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v141_data + (v138_data * v119_data));
              tensorforge::intel_esimd::simd<float, 32> v146_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v146_data + (v138_data * v124_data));
              tensorforge::intel_esimd::simd<float, 32> v151_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v151_data + (v138_data * v129_data));
              tensorforge::intel_esimd::simd<float, 32> v156_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v156_data + (v138_data * v134_data));
              tensorforge::intel_esimd::simd<float, 32> v158_data(r0.template select<32, 1>(192));
              float v159_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v161_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v161_data + (v158_data * v159_data));
              float v164_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v166_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v166_data + (v158_data * v164_data));
              float v169_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v171_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v171_data + (v158_data * v169_data));
              float v174_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v176_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v176_data + (v158_data * v174_data));
              tensorforge::intel_esimd::simd<float, 32> v178_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v181_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v181_data + (v178_data * v159_data));
              tensorforge::intel_esimd::simd<float, 32> v186_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v186_data + (v178_data * v164_data));
              tensorforge::intel_esimd::simd<float, 32> v191_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v191_data + (v178_data * v169_data));
              tensorforge::intel_esimd::simd<float, 32> v196_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v196_data + (v178_data * v174_data));
              tensorforge::intel_esimd::simd<float, 32> v198_data(r0.template select<32, 1>(256));
              float v199_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v201_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v201_data + (v198_data * v199_data));
              float v204_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v206_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v206_data + (v198_data * v204_data));
              float v209_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v211_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v211_data + (v198_data * v209_data));
              float v214_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v216_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v216_data + (v198_data * v214_data));
              tensorforge::intel_esimd::simd<float, 32> v218_data(r0.template select<32, 1>(288));
              tensorforge::intel_esimd::simd<float, 32> v221_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v221_data + (v218_data * v199_data));
              tensorforge::intel_esimd::simd<float, 32> v226_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v226_data + (v218_data * v204_data));
              tensorforge::intel_esimd::simd<float, 32> v231_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v231_data + (v218_data * v209_data));
              tensorforge::intel_esimd::simd<float, 32> v236_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v236_data + (v218_data * v214_data));
              tensorforge::intel_esimd::simd<float, 32> v238_data(r0.template select<32, 1>(320));
              float v239_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v241_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v241_data + (v238_data * v239_data));
              float v244_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v246_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v246_data + (v238_data * v244_data));
              float v249_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v251_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v251_data + (v238_data * v249_data));
              float v254_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v256_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v256_data + (v238_data * v254_data));
              tensorforge::intel_esimd::simd<float, 32> v258_data(r0.template select<32, 1>(352));
              tensorforge::intel_esimd::simd<float, 32> v261_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v261_data + (v258_data * v239_data));
              tensorforge::intel_esimd::simd<float, 32> v266_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v266_data + (v258_data * v244_data));
              tensorforge::intel_esimd::simd<float, 32> v271_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v271_data + (v258_data * v249_data));
              tensorforge::intel_esimd::simd<float, 32> v276_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v276_data + (v258_data * v254_data));
              tensorforge::intel_esimd::simd<float, 32> v278_data(r0.template select<32, 1>(384));
              float v279_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v281_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v281_data + (v278_data * v279_data));
              float v284_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v286_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v286_data + (v278_data * v284_data));
              float v289_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v291_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v291_data + (v278_data * v289_data));
              float v294_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v296_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v296_data + (v278_data * v294_data));
              tensorforge::intel_esimd::simd<float, 32> v298_data(r0.template select<32, 1>(416));
              tensorforge::intel_esimd::simd<float, 32> v301_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v301_data + (v298_data * v279_data));
              tensorforge::intel_esimd::simd<float, 32> v306_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v306_data + (v298_data * v284_data));
              tensorforge::intel_esimd::simd<float, 32> v311_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v311_data + (v298_data * v289_data));
              tensorforge::intel_esimd::simd<float, 32> v316_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v316_data + (v298_data * v294_data));
              tensorforge::intel_esimd::simd<float, 32> v318_data(r0.template select<32, 1>(448));
              float v319_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v321_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v321_data + (v318_data * v319_data));
              float v324_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v326_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v326_data + (v318_data * v324_data));
              float v329_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v331_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v331_data + (v318_data * v329_data));
              float v334_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v336_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v336_data + (v318_data * v334_data));
              tensorforge::intel_esimd::simd<float, 32> v338_data(r0.template select<32, 1>(480));
              tensorforge::intel_esimd::simd<float, 32> v341_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v341_data + (v338_data * v319_data));
              tensorforge::intel_esimd::simd<float, 32> v346_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v346_data + (v338_data * v324_data));
              tensorforge::intel_esimd::simd<float, 32> v351_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v351_data + (v338_data * v329_data));
              tensorforge::intel_esimd::simd<float, 32> v356_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v356_data + (v338_data * v334_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v358_n0 = 0; v358_n0 < 1; ++v358_n0) {
                int32_t v360_a = v358_n0 * 32;
                #pragma unroll
                for (int32_t v359_n1 = 0; v359_n1 < 4; ++v359_n1) {
                  int32_t v362_a = v360_a + (v359_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v363_data(ir1.template select<32, 1>(v362_a));
                  r1.template select<32, 1>(v362_a) = v363_data;
                }
              }
              #pragma unroll
              for (int32_t v364_n1 = 0; v364_n1 < 4; ++v364_n1) {
                int32_t v366_a = 32 + (v364_n1 * 64);
                tensorforge::intel_esimd::simd<float, 3> v367_data(ir1.template select<3, 1>(v366_a));
                r1.template select<3, 1>(v366_a) = v367_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v368_i0 = 0; v368_i0 < 1; ++v368_i0) {
                int32_t v370_a = v368_i0 * 32;
                #pragma unroll
                for (int32_t v369_i1 = 0; v369_i1 < 4; ++v369_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v373_data(r1.template select<32, 1>((v370_a + (v369_i1 * 64))));
                  v373_data.copy_to(glb_m0 + ((v370_a + (v369_i1 * 35))));
                }
              }
              #pragma unroll
              for (int32_t v377_i1 = 0; v377_i1 < 4; ++v377_i1) {
                tensorforge::intel_esimd::simd<float, 3> v380_data(r1.template select<3, 1>((32 + (v377_i1 * 64))));
                v380_data.copy_to(glb_m0 + ((32_i32 + (v377_i1 * 35))));
              }
            }
          }
        }
      }
    });
  });
}


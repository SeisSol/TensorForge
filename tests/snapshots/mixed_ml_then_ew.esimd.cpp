// === base name ===
kernel_c9d255d2a84f575b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c9d255d2a84f575b = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c9d255d2a84f575b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c9d255d2a84f575b(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c9d255d2a84f575b(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_c9d255d2a84f575b(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c9d255d2a84f575b(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c9d255d2a84f575b(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c9d255d2a84f575b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (72 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v12_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v12_batchId0 < numElements0; v12_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 64 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v26_lead = v24_i0 * 8;
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                  int32_t v29_a = v26_lead + (v25_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v30_data;
                  v30_data.copy_from(glb_m0 + (v29_a));
                  r0.template select<8, 1>(v29_a) = v30_data;
                }
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v32_ld;
              v32_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v32_ld);
              tensorforge::intel_esimd::simd<float, 32> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 32), v33_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v35_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s0_w0 = tensorforge::slmLoad<float, 64>(s0 + 0);
              float v36_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v38_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v38_data + (v35_data * v36_data));
              float v41_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v43_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v43_data + (v35_data * v41_data));
              float v46_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v48_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v48_data + (v35_data * v46_data));
              float v51_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v53_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v53_data + (v35_data * v51_data));
              float v56_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v58_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v58_data + (v35_data * v56_data));
              float v61_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v63_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v63_data + (v35_data * v61_data));
              float v66_data = s0_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v68_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v68_data + (v35_data * v66_data));
              float v71_data = s0_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v73_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v73_data + (v35_data * v71_data));
              tensorforge::intel_esimd::simd<float, 8> v75_data(r0.template select<8, 1>(8));
              float v76_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v78_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v78_data + (v75_data * v76_data));
              float v81_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v83_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v83_data + (v75_data * v81_data));
              float v86_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v88_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v88_data + (v75_data * v86_data));
              float v91_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v93_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v93_data + (v75_data * v91_data));
              float v96_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v98_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v98_data + (v75_data * v96_data));
              float v101_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v103_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v103_data + (v75_data * v101_data));
              float v106_data = s0_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v108_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v108_data + (v75_data * v106_data));
              float v111_data = s0_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v113_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v113_data + (v75_data * v111_data));
              tensorforge::intel_esimd::simd<float, 8> v115_data(r0.template select<8, 1>(16));
              float v116_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v118_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v118_data + (v115_data * v116_data));
              float v121_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v123_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v123_data + (v115_data * v121_data));
              float v126_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v128_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v128_data + (v115_data * v126_data));
              float v131_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v133_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v133_data + (v115_data * v131_data));
              float v136_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v138_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v138_data + (v115_data * v136_data));
              float v141_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v143_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v143_data + (v115_data * v141_data));
              float v146_data = s0_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v148_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v148_data + (v115_data * v146_data));
              float v151_data = s0_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v153_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v153_data + (v115_data * v151_data));
              tensorforge::intel_esimd::simd<float, 8> v155_data(r0.template select<8, 1>(24));
              float v156_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v158_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v158_data + (v155_data * v156_data));
              float v161_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v163_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v163_data + (v155_data * v161_data));
              float v166_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v168_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v168_data + (v155_data * v166_data));
              float v171_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v173_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v173_data + (v155_data * v171_data));
              float v176_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v178_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v178_data + (v155_data * v176_data));
              float v181_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v183_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v183_data + (v155_data * v181_data));
              float v186_data = s0_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v188_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v188_data + (v155_data * v186_data));
              float v191_data = s0_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v193_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v193_data + (v155_data * v191_data));
              tensorforge::intel_esimd::simd<float, 8> v195_data(r0.template select<8, 1>(32));
              float v196_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v198_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v198_data + (v195_data * v196_data));
              float v201_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v203_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v203_data + (v195_data * v201_data));
              float v206_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v208_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v208_data + (v195_data * v206_data));
              float v211_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v213_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v213_data + (v195_data * v211_data));
              float v216_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v218_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v218_data + (v195_data * v216_data));
              float v221_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v223_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v223_data + (v195_data * v221_data));
              float v226_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v228_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v228_data + (v195_data * v226_data));
              float v231_data = s0_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v233_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v233_data + (v195_data * v231_data));
              tensorforge::intel_esimd::simd<float, 8> v235_data(r0.template select<8, 1>(40));
              float v236_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v238_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v238_data + (v235_data * v236_data));
              float v241_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v243_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v243_data + (v235_data * v241_data));
              float v246_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v248_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v248_data + (v235_data * v246_data));
              float v251_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v253_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v253_data + (v235_data * v251_data));
              float v256_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v258_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v258_data + (v235_data * v256_data));
              float v261_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v263_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v263_data + (v235_data * v261_data));
              float v266_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v268_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v268_data + (v235_data * v266_data));
              float v271_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v273_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v273_data + (v235_data * v271_data));
              tensorforge::intel_esimd::simd<float, 8> v275_data(r0.template select<8, 1>(48));
              float v276_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v278_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v278_data + (v275_data * v276_data));
              float v281_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v283_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v283_data + (v275_data * v281_data));
              float v286_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v288_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v288_data + (v275_data * v286_data));
              float v291_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v293_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v293_data + (v275_data * v291_data));
              float v296_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v298_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v298_data + (v275_data * v296_data));
              float v301_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v303_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v303_data + (v275_data * v301_data));
              float v306_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v308_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v308_data + (v275_data * v306_data));
              float v311_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v313_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v313_data + (v275_data * v311_data));
              tensorforge::intel_esimd::simd<float, 8> v315_data(r0.template select<8, 1>(56));
              float v316_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v318_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v318_data + (v315_data * v316_data));
              float v321_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v323_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v323_data + (v315_data * v321_data));
              float v326_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v328_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v328_data + (v315_data * v326_data));
              float v331_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v333_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v333_data + (v315_data * v331_data));
              float v336_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v338_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v338_data + (v315_data * v336_data));
              float v341_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v343_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v343_data + (v315_data * v341_data));
              float v346_data = s0_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v348_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v348_data + (v315_data * v346_data));
              float v351_data = s0_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v353_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v353_data + (v315_data * v351_data));
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v355_i0 = 0; v355_i0 < 1; ++v355_i0) {
                int32_t v357_a = v355_i0 * 8;
                #pragma unroll
                for (int32_t v356_i1 = 0; v356_i1 < 8; ++v356_i1) {
                  int32_t v359_a = v357_a + (v356_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v360_data(r1.template select<8, 1>(v359_a));
                  tensorforge::slmStore<float, 8>(s1 + (v359_a), v360_data);
                }
              }
              // glb_m2 = abs(s1)
              #pragma unroll
              for (int32_t v363_k0 = 0; v363_k0 < 1; ++v363_k0) {
                int32_t v365_lead = v363_k0 * 8;
                #pragma unroll
                for (int32_t v364_k1 = 0; v364_k1 < 8; ++v364_k1) {
                  int32_t v368_a = v365_lead + (v364_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v369_data = tensorforge::slmLoad<float, 8>(s1 + (v368_a));
                  (tensorforge::intel_esimd::abs(v369_data)).copy_to(glb_m2 + (v368_a));
                }
              }
            }
          }
        }
      }
    });
  });
}


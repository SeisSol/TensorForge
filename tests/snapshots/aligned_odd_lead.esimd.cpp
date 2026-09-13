// === base name ===
kernel_f0d4d5811fe13935

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f0d4d5811fe13935 = {{1, 8, 1}, 32, 35, 1, 8, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f0d4d5811fe13935(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f0d4d5811fe13935(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f0d4d5811fe13935(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f0d4d5811fe13935(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f0d4d5811fe13935(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f0d4d5811fe13935(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f0d4d5811fe13935(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 280 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 32 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 32 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 512> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                int32_t v21_lead = v19_i0 * 32;
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 8; ++v20_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v25_data;
                  v25_data.copy_from(glb_m1 + ((v21_lead + (v20_i1 * 35))));
                  r0.template select<32, 1>((v21_lead + (v20_i1 * 64))) = v25_data;
                }
              }
              #pragma unroll
              for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
                tensorforge::intel_esimd::simd<float, 3> v34_data;
                v34_data.copy_from(glb_m1 + ((32_i32 + (v28_i1 * 35))));
                r0.template select<3, 1>((32 + (v28_i1 * 64))) = v34_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v37_ld;
              v37_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v37_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v40_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> s0_w0 = tensorforge::slmLoad<float, 32>(s0 + 0);
              float v41_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v43_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v43_data + (v40_data * v41_data));
              float v46_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v48_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v48_data + (v40_data * v46_data));
              float v51_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v53_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v53_data + (v40_data * v51_data));
              float v56_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v58_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v58_data + (v40_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v60_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v63_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v63_data + (v60_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v68_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v68_data + (v60_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v73_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v73_data + (v60_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v78_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v78_data + (v60_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v80_data(r0.template select<32, 1>(64));
              float v81_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v83_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v83_data + (v80_data * v81_data));
              float v86_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v88_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v88_data + (v80_data * v86_data));
              float v91_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v93_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v93_data + (v80_data * v91_data));
              float v96_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v98_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v98_data + (v80_data * v96_data));
              tensorforge::intel_esimd::simd<float, 32> v100_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v103_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v103_data + (v100_data * v81_data));
              tensorforge::intel_esimd::simd<float, 32> v108_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v108_data + (v100_data * v86_data));
              tensorforge::intel_esimd::simd<float, 32> v113_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v113_data + (v100_data * v91_data));
              tensorforge::intel_esimd::simd<float, 32> v118_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v118_data + (v100_data * v96_data));
              tensorforge::intel_esimd::simd<float, 32> v120_data(r0.template select<32, 1>(128));
              float v121_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v123_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v123_data + (v120_data * v121_data));
              float v126_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v128_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v128_data + (v120_data * v126_data));
              float v131_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v133_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v133_data + (v120_data * v131_data));
              float v136_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v138_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v138_data + (v120_data * v136_data));
              tensorforge::intel_esimd::simd<float, 32> v140_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v143_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v143_data + (v140_data * v121_data));
              tensorforge::intel_esimd::simd<float, 32> v148_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v148_data + (v140_data * v126_data));
              tensorforge::intel_esimd::simd<float, 32> v153_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v153_data + (v140_data * v131_data));
              tensorforge::intel_esimd::simd<float, 32> v158_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v158_data + (v140_data * v136_data));
              tensorforge::intel_esimd::simd<float, 32> v160_data(r0.template select<32, 1>(192));
              float v161_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v163_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v163_data + (v160_data * v161_data));
              float v166_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v168_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v168_data + (v160_data * v166_data));
              float v171_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v173_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v173_data + (v160_data * v171_data));
              float v176_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v178_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v178_data + (v160_data * v176_data));
              tensorforge::intel_esimd::simd<float, 32> v180_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v183_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v183_data + (v180_data * v161_data));
              tensorforge::intel_esimd::simd<float, 32> v188_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v188_data + (v180_data * v166_data));
              tensorforge::intel_esimd::simd<float, 32> v193_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v193_data + (v180_data * v171_data));
              tensorforge::intel_esimd::simd<float, 32> v198_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v198_data + (v180_data * v176_data));
              tensorforge::intel_esimd::simd<float, 32> v200_data(r0.template select<32, 1>(256));
              float v201_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v203_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v203_data + (v200_data * v201_data));
              float v206_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v208_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v208_data + (v200_data * v206_data));
              float v211_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v213_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v213_data + (v200_data * v211_data));
              float v216_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v218_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v218_data + (v200_data * v216_data));
              tensorforge::intel_esimd::simd<float, 32> v220_data(r0.template select<32, 1>(288));
              tensorforge::intel_esimd::simd<float, 32> v223_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v223_data + (v220_data * v201_data));
              tensorforge::intel_esimd::simd<float, 32> v228_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v228_data + (v220_data * v206_data));
              tensorforge::intel_esimd::simd<float, 32> v233_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v233_data + (v220_data * v211_data));
              tensorforge::intel_esimd::simd<float, 32> v238_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v238_data + (v220_data * v216_data));
              tensorforge::intel_esimd::simd<float, 32> v240_data(r0.template select<32, 1>(320));
              float v241_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v243_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v243_data + (v240_data * v241_data));
              float v246_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v248_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v248_data + (v240_data * v246_data));
              float v251_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v253_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v253_data + (v240_data * v251_data));
              float v256_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v258_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v258_data + (v240_data * v256_data));
              tensorforge::intel_esimd::simd<float, 32> v260_data(r0.template select<32, 1>(352));
              tensorforge::intel_esimd::simd<float, 32> v263_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v263_data + (v260_data * v241_data));
              tensorforge::intel_esimd::simd<float, 32> v268_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v268_data + (v260_data * v246_data));
              tensorforge::intel_esimd::simd<float, 32> v273_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v273_data + (v260_data * v251_data));
              tensorforge::intel_esimd::simd<float, 32> v278_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v278_data + (v260_data * v256_data));
              tensorforge::intel_esimd::simd<float, 32> v280_data(r0.template select<32, 1>(384));
              float v281_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v283_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v283_data + (v280_data * v281_data));
              float v286_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v288_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v288_data + (v280_data * v286_data));
              float v291_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v293_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v293_data + (v280_data * v291_data));
              float v296_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v298_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v298_data + (v280_data * v296_data));
              tensorforge::intel_esimd::simd<float, 32> v300_data(r0.template select<32, 1>(416));
              tensorforge::intel_esimd::simd<float, 32> v303_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v303_data + (v300_data * v281_data));
              tensorforge::intel_esimd::simd<float, 32> v308_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v308_data + (v300_data * v286_data));
              tensorforge::intel_esimd::simd<float, 32> v313_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v313_data + (v300_data * v291_data));
              tensorforge::intel_esimd::simd<float, 32> v318_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v318_data + (v300_data * v296_data));
              tensorforge::intel_esimd::simd<float, 32> v320_data(r0.template select<32, 1>(448));
              float v321_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v323_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v323_data + (v320_data * v321_data));
              float v326_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v328_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v328_data + (v320_data * v326_data));
              float v331_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v333_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v333_data + (v320_data * v331_data));
              float v336_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v338_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v338_data + (v320_data * v336_data));
              tensorforge::intel_esimd::simd<float, 32> v340_data(r0.template select<32, 1>(480));
              tensorforge::intel_esimd::simd<float, 32> v343_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v343_data + (v340_data * v321_data));
              tensorforge::intel_esimd::simd<float, 32> v348_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v348_data + (v340_data * v326_data));
              tensorforge::intel_esimd::simd<float, 32> v353_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v353_data + (v340_data * v331_data));
              tensorforge::intel_esimd::simd<float, 32> v358_data(ir1.template select<32, 1>(224));
              ir1.template select<32, 1>(224) = (v358_data + (v340_data * v336_data));
              #pragma unroll
              for (int32_t v360_n0 = 0; v360_n0 < 1; ++v360_n0) {
                int32_t v362_a = v360_n0 * 32;
                #pragma unroll
                for (int32_t v361_n1 = 0; v361_n1 < 4; ++v361_n1) {
                  int32_t v364_a = v362_a + (v361_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v365_data(ir1.template select<32, 1>(v364_a));
                  r1.template select<32, 1>(v364_a) = v365_data;
                }
              }
              #pragma unroll
              for (int32_t v366_n1 = 0; v366_n1 < 4; ++v366_n1) {
                int32_t v368_a = 32 + (v366_n1 * 64);
                tensorforge::intel_esimd::simd<float, 3> v369_data(ir1.template select<3, 1>(v368_a));
                r1.template select<3, 1>(v368_a) = v369_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v370_i0 = 0; v370_i0 < 1; ++v370_i0) {
                int32_t v372_a = v370_i0 * 32;
                #pragma unroll
                for (int32_t v371_i1 = 0; v371_i1 < 4; ++v371_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v375_data(r1.template select<32, 1>((v372_a + (v371_i1 * 64))));
                  v375_data.copy_to(glb_m0 + ((v372_a + (v371_i1 * 35))));
                }
              }
              #pragma unroll
              for (int32_t v379_i1 = 0; v379_i1 < 4; ++v379_i1) {
                tensorforge::intel_esimd::simd<float, 3> v382_data(r1.template select<3, 1>((32 + (v379_i1 * 64))));
                v382_data.copy_to(glb_m0 + ((32_i32 + (v379_i1 * 35))));
              }
            }
            tensorforge::prefetchRunsL2<1120, 128>(&pf_glb_m1[0], &pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


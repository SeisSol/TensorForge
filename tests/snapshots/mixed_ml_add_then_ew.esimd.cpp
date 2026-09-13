// === base name ===
kernel_f80407273666d245

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f80407273666d245 = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f80407273666d245(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f80407273666d245(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f80407273666d245(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_f80407273666d245(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f80407273666d245(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f80407273666d245(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f80407273666d245(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        //   m4 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t0[i,j] += m2[i,k] × m3[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (72 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const float *const __restrict__ pf_glb_m0 = &m0[v10_batchId1 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ pf_glb_m1 = &m1[v10_batchId1 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v10_batchId1 * 64 + 0 + m2_extraOffset];
            const float *const __restrict__ pf_glb_m3 = &m3[v10_batchId1 * 64 + 0 + m3_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 64 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
                int32_t v27_lead = v25_i0 * 8;
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 8; ++v26_i1) {
                  int32_t v30_a = v27_lead + (v26_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v31_data;
                  v31_data.copy_from(glb_m0 + (v30_a));
                  r0.template select<8, 1>(v30_a) = v31_data;
                }
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v33_ld);
              tensorforge::intel_esimd::simd<float, 32> v34_ld;
              v34_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 32), v34_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 64> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v36_i0 = 0; v36_i0 < 1; ++v36_i0) {
                int32_t v38_lead = v36_i0 * 8;
                #pragma unroll
                for (int32_t v37_i1 = 0; v37_i1 < 8; ++v37_i1) {
                  int32_t v41_a = v38_lead + (v37_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v42_data;
                  v42_data.copy_from(glb_m2 + (v41_a));
                  r2.template select<8, 1>(v41_a) = v42_data;
                }
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v45_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s0_w0 = tensorforge::slmLoad<float, 64>(s0 + 0);
              float v46_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v48_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v48_data + (v45_data * v46_data));
              float v51_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v53_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v53_data + (v45_data * v51_data));
              float v56_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v58_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v58_data + (v45_data * v56_data));
              float v61_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v63_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v63_data + (v45_data * v61_data));
              float v66_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v68_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v68_data + (v45_data * v66_data));
              float v71_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v73_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v73_data + (v45_data * v71_data));
              float v76_data = s0_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v78_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v78_data + (v45_data * v76_data));
              float v81_data = s0_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v83_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v83_data + (v45_data * v81_data));
              tensorforge::intel_esimd::simd<float, 8> v85_data(r0.template select<8, 1>(8));
              float v86_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v88_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v88_data + (v85_data * v86_data));
              float v91_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v93_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v93_data + (v85_data * v91_data));
              float v96_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v98_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v98_data + (v85_data * v96_data));
              float v101_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v103_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v103_data + (v85_data * v101_data));
              float v106_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v108_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v108_data + (v85_data * v106_data));
              float v111_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v113_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v113_data + (v85_data * v111_data));
              float v116_data = s0_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v118_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v118_data + (v85_data * v116_data));
              float v121_data = s0_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v123_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v123_data + (v85_data * v121_data));
              tensorforge::intel_esimd::simd<float, 8> v125_data(r0.template select<8, 1>(16));
              float v126_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v128_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v128_data + (v125_data * v126_data));
              float v131_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v133_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v133_data + (v125_data * v131_data));
              float v136_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v138_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v138_data + (v125_data * v136_data));
              float v141_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v143_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v143_data + (v125_data * v141_data));
              float v146_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v148_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v148_data + (v125_data * v146_data));
              float v151_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v153_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v153_data + (v125_data * v151_data));
              float v156_data = s0_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v158_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v158_data + (v125_data * v156_data));
              float v161_data = s0_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v163_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v163_data + (v125_data * v161_data));
              tensorforge::intel_esimd::simd<float, 8> v165_data(r0.template select<8, 1>(24));
              float v166_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v168_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v168_data + (v165_data * v166_data));
              float v171_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v173_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v173_data + (v165_data * v171_data));
              float v176_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v178_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v178_data + (v165_data * v176_data));
              float v181_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v183_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v183_data + (v165_data * v181_data));
              float v186_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v188_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v188_data + (v165_data * v186_data));
              float v191_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v193_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v193_data + (v165_data * v191_data));
              float v196_data = s0_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v198_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v198_data + (v165_data * v196_data));
              float v201_data = s0_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v203_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v203_data + (v165_data * v201_data));
              tensorforge::intel_esimd::simd<float, 8> v205_data(r0.template select<8, 1>(32));
              float v206_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v208_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v208_data + (v205_data * v206_data));
              float v211_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v213_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v213_data + (v205_data * v211_data));
              float v216_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v218_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v218_data + (v205_data * v216_data));
              float v221_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v223_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v223_data + (v205_data * v221_data));
              float v226_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v228_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v228_data + (v205_data * v226_data));
              float v231_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v233_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v233_data + (v205_data * v231_data));
              float v236_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v238_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v238_data + (v205_data * v236_data));
              float v241_data = s0_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v243_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v243_data + (v205_data * v241_data));
              tensorforge::intel_esimd::simd<float, 8> v245_data(r0.template select<8, 1>(40));
              float v246_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v248_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v248_data + (v245_data * v246_data));
              float v251_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v253_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v253_data + (v245_data * v251_data));
              float v256_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v258_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v258_data + (v245_data * v256_data));
              float v261_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v263_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v263_data + (v245_data * v261_data));
              float v266_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v268_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v268_data + (v245_data * v266_data));
              float v271_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v273_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v273_data + (v245_data * v271_data));
              float v276_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v278_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v278_data + (v245_data * v276_data));
              float v281_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v283_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v283_data + (v245_data * v281_data));
              tensorforge::intel_esimd::simd<float, 8> v285_data(r0.template select<8, 1>(48));
              float v286_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v288_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v288_data + (v285_data * v286_data));
              float v291_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v293_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v293_data + (v285_data * v291_data));
              float v296_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v298_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v298_data + (v285_data * v296_data));
              float v301_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v303_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v303_data + (v285_data * v301_data));
              float v306_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v308_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v308_data + (v285_data * v306_data));
              float v311_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v313_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v313_data + (v285_data * v311_data));
              float v316_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v318_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v318_data + (v285_data * v316_data));
              float v321_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v323_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v323_data + (v285_data * v321_data));
              tensorforge::intel_esimd::simd<float, 8> v325_data(r0.template select<8, 1>(56));
              float v326_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v328_data(r1.template select<8, 1>(0));
              r1.template select<8, 1>(0) = (v328_data + (v325_data * v326_data));
              float v331_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v333_data(r1.template select<8, 1>(8));
              r1.template select<8, 1>(8) = (v333_data + (v325_data * v331_data));
              float v336_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v338_data(r1.template select<8, 1>(16));
              r1.template select<8, 1>(16) = (v338_data + (v325_data * v336_data));
              float v341_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v343_data(r1.template select<8, 1>(24));
              r1.template select<8, 1>(24) = (v343_data + (v325_data * v341_data));
              float v346_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v348_data(r1.template select<8, 1>(32));
              r1.template select<8, 1>(32) = (v348_data + (v325_data * v346_data));
              float v351_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v353_data(r1.template select<8, 1>(40));
              r1.template select<8, 1>(40) = (v353_data + (v325_data * v351_data));
              float v356_data = s0_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v358_data(r1.template select<8, 1>(48));
              r1.template select<8, 1>(48) = (v358_data + (v325_data * v356_data));
              float v361_data = s0_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v363_data(r1.template select<8, 1>(56));
              r1.template select<8, 1>(56) = (v363_data + (v325_data * v361_data));
              // s2 = load{g>s}(glb_m3[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v365_ld;
              v365_ld.copy_from(glb_m3 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 0), v365_ld);
              tensorforge::intel_esimd::simd<float, 32> v366_ld;
              v366_ld.copy_from(glb_m3 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 4 * 0 + 32), v366_ld);
              // wait(r2 = load{g>r}(glb_m2););
              // wait(s2 = load{g>s}(glb_m3[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r3(0.0f);
              // ir3 = +(r2 * s2)
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 64> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v369_data(r2.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s2_w1 = tensorforge::slmLoad<float, 64>(s2 + 0);
              float v370_data = s2_w1[0];
              tensorforge::intel_esimd::simd<float, 8> v372_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v372_data + (v369_data * v370_data));
              float v375_data = s2_w1[8];
              tensorforge::intel_esimd::simd<float, 8> v377_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v377_data + (v369_data * v375_data));
              float v380_data = s2_w1[16];
              tensorforge::intel_esimd::simd<float, 8> v382_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v382_data + (v369_data * v380_data));
              float v385_data = s2_w1[24];
              tensorforge::intel_esimd::simd<float, 8> v387_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v387_data + (v369_data * v385_data));
              float v390_data = s2_w1[32];
              tensorforge::intel_esimd::simd<float, 8> v392_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v392_data + (v369_data * v390_data));
              float v395_data = s2_w1[40];
              tensorforge::intel_esimd::simd<float, 8> v397_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v397_data + (v369_data * v395_data));
              float v400_data = s2_w1[48];
              tensorforge::intel_esimd::simd<float, 8> v402_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v402_data + (v369_data * v400_data));
              float v405_data = s2_w1[56];
              tensorforge::intel_esimd::simd<float, 8> v407_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v407_data + (v369_data * v405_data));
              tensorforge::intel_esimd::simd<float, 8> v409_data(r2.template select<8, 1>(8));
              float v410_data = s2_w1[1];
              tensorforge::intel_esimd::simd<float, 8> v412_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v412_data + (v409_data * v410_data));
              float v415_data = s2_w1[9];
              tensorforge::intel_esimd::simd<float, 8> v417_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v417_data + (v409_data * v415_data));
              float v420_data = s2_w1[17];
              tensorforge::intel_esimd::simd<float, 8> v422_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v422_data + (v409_data * v420_data));
              float v425_data = s2_w1[25];
              tensorforge::intel_esimd::simd<float, 8> v427_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v427_data + (v409_data * v425_data));
              float v430_data = s2_w1[33];
              tensorforge::intel_esimd::simd<float, 8> v432_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v432_data + (v409_data * v430_data));
              float v435_data = s2_w1[41];
              tensorforge::intel_esimd::simd<float, 8> v437_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v437_data + (v409_data * v435_data));
              float v440_data = s2_w1[49];
              tensorforge::intel_esimd::simd<float, 8> v442_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v442_data + (v409_data * v440_data));
              float v445_data = s2_w1[57];
              tensorforge::intel_esimd::simd<float, 8> v447_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v447_data + (v409_data * v445_data));
              tensorforge::intel_esimd::simd<float, 8> v449_data(r2.template select<8, 1>(16));
              float v450_data = s2_w1[2];
              tensorforge::intel_esimd::simd<float, 8> v452_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v452_data + (v449_data * v450_data));
              float v455_data = s2_w1[10];
              tensorforge::intel_esimd::simd<float, 8> v457_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v457_data + (v449_data * v455_data));
              float v460_data = s2_w1[18];
              tensorforge::intel_esimd::simd<float, 8> v462_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v462_data + (v449_data * v460_data));
              float v465_data = s2_w1[26];
              tensorforge::intel_esimd::simd<float, 8> v467_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v467_data + (v449_data * v465_data));
              float v470_data = s2_w1[34];
              tensorforge::intel_esimd::simd<float, 8> v472_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v472_data + (v449_data * v470_data));
              float v475_data = s2_w1[42];
              tensorforge::intel_esimd::simd<float, 8> v477_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v477_data + (v449_data * v475_data));
              float v480_data = s2_w1[50];
              tensorforge::intel_esimd::simd<float, 8> v482_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v482_data + (v449_data * v480_data));
              float v485_data = s2_w1[58];
              tensorforge::intel_esimd::simd<float, 8> v487_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v487_data + (v449_data * v485_data));
              tensorforge::intel_esimd::simd<float, 8> v489_data(r2.template select<8, 1>(24));
              float v490_data = s2_w1[3];
              tensorforge::intel_esimd::simd<float, 8> v492_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v492_data + (v489_data * v490_data));
              float v495_data = s2_w1[11];
              tensorforge::intel_esimd::simd<float, 8> v497_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v497_data + (v489_data * v495_data));
              float v500_data = s2_w1[19];
              tensorforge::intel_esimd::simd<float, 8> v502_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v502_data + (v489_data * v500_data));
              float v505_data = s2_w1[27];
              tensorforge::intel_esimd::simd<float, 8> v507_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v507_data + (v489_data * v505_data));
              float v510_data = s2_w1[35];
              tensorforge::intel_esimd::simd<float, 8> v512_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v512_data + (v489_data * v510_data));
              float v515_data = s2_w1[43];
              tensorforge::intel_esimd::simd<float, 8> v517_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v517_data + (v489_data * v515_data));
              float v520_data = s2_w1[51];
              tensorforge::intel_esimd::simd<float, 8> v522_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v522_data + (v489_data * v520_data));
              float v525_data = s2_w1[59];
              tensorforge::intel_esimd::simd<float, 8> v527_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v527_data + (v489_data * v525_data));
              tensorforge::intel_esimd::simd<float, 8> v529_data(r2.template select<8, 1>(32));
              float v530_data = s2_w1[4];
              tensorforge::intel_esimd::simd<float, 8> v532_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v532_data + (v529_data * v530_data));
              float v535_data = s2_w1[12];
              tensorforge::intel_esimd::simd<float, 8> v537_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v537_data + (v529_data * v535_data));
              float v540_data = s2_w1[20];
              tensorforge::intel_esimd::simd<float, 8> v542_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v542_data + (v529_data * v540_data));
              float v545_data = s2_w1[28];
              tensorforge::intel_esimd::simd<float, 8> v547_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v547_data + (v529_data * v545_data));
              float v550_data = s2_w1[36];
              tensorforge::intel_esimd::simd<float, 8> v552_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v552_data + (v529_data * v550_data));
              float v555_data = s2_w1[44];
              tensorforge::intel_esimd::simd<float, 8> v557_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v557_data + (v529_data * v555_data));
              float v560_data = s2_w1[52];
              tensorforge::intel_esimd::simd<float, 8> v562_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v562_data + (v529_data * v560_data));
              float v565_data = s2_w1[60];
              tensorforge::intel_esimd::simd<float, 8> v567_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v567_data + (v529_data * v565_data));
              tensorforge::intel_esimd::simd<float, 8> v569_data(r2.template select<8, 1>(40));
              float v570_data = s2_w1[5];
              tensorforge::intel_esimd::simd<float, 8> v572_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v572_data + (v569_data * v570_data));
              float v575_data = s2_w1[13];
              tensorforge::intel_esimd::simd<float, 8> v577_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v577_data + (v569_data * v575_data));
              float v580_data = s2_w1[21];
              tensorforge::intel_esimd::simd<float, 8> v582_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v582_data + (v569_data * v580_data));
              float v585_data = s2_w1[29];
              tensorforge::intel_esimd::simd<float, 8> v587_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v587_data + (v569_data * v585_data));
              float v590_data = s2_w1[37];
              tensorforge::intel_esimd::simd<float, 8> v592_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v592_data + (v569_data * v590_data));
              float v595_data = s2_w1[45];
              tensorforge::intel_esimd::simd<float, 8> v597_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v597_data + (v569_data * v595_data));
              float v600_data = s2_w1[53];
              tensorforge::intel_esimd::simd<float, 8> v602_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v602_data + (v569_data * v600_data));
              float v605_data = s2_w1[61];
              tensorforge::intel_esimd::simd<float, 8> v607_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v607_data + (v569_data * v605_data));
              tensorforge::intel_esimd::simd<float, 8> v609_data(r2.template select<8, 1>(48));
              float v610_data = s2_w1[6];
              tensorforge::intel_esimd::simd<float, 8> v612_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v612_data + (v609_data * v610_data));
              float v615_data = s2_w1[14];
              tensorforge::intel_esimd::simd<float, 8> v617_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v617_data + (v609_data * v615_data));
              float v620_data = s2_w1[22];
              tensorforge::intel_esimd::simd<float, 8> v622_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v622_data + (v609_data * v620_data));
              float v625_data = s2_w1[30];
              tensorforge::intel_esimd::simd<float, 8> v627_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v627_data + (v609_data * v625_data));
              float v630_data = s2_w1[38];
              tensorforge::intel_esimd::simd<float, 8> v632_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v632_data + (v609_data * v630_data));
              float v635_data = s2_w1[46];
              tensorforge::intel_esimd::simd<float, 8> v637_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v637_data + (v609_data * v635_data));
              float v640_data = s2_w1[54];
              tensorforge::intel_esimd::simd<float, 8> v642_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v642_data + (v609_data * v640_data));
              float v645_data = s2_w1[62];
              tensorforge::intel_esimd::simd<float, 8> v647_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v647_data + (v609_data * v645_data));
              tensorforge::intel_esimd::simd<float, 8> v649_data(r2.template select<8, 1>(56));
              float v650_data = s2_w1[7];
              tensorforge::intel_esimd::simd<float, 8> v652_data(ir3.template select<8, 1>(0));
              ir3.template select<8, 1>(0) = (v652_data + (v649_data * v650_data));
              float v655_data = s2_w1[15];
              tensorforge::intel_esimd::simd<float, 8> v657_data(ir3.template select<8, 1>(8));
              ir3.template select<8, 1>(8) = (v657_data + (v649_data * v655_data));
              float v660_data = s2_w1[23];
              tensorforge::intel_esimd::simd<float, 8> v662_data(ir3.template select<8, 1>(16));
              ir3.template select<8, 1>(16) = (v662_data + (v649_data * v660_data));
              float v665_data = s2_w1[31];
              tensorforge::intel_esimd::simd<float, 8> v667_data(ir3.template select<8, 1>(24));
              ir3.template select<8, 1>(24) = (v667_data + (v649_data * v665_data));
              float v670_data = s2_w1[39];
              tensorforge::intel_esimd::simd<float, 8> v672_data(ir3.template select<8, 1>(32));
              ir3.template select<8, 1>(32) = (v672_data + (v649_data * v670_data));
              float v675_data = s2_w1[47];
              tensorforge::intel_esimd::simd<float, 8> v677_data(ir3.template select<8, 1>(40));
              ir3.template select<8, 1>(40) = (v677_data + (v649_data * v675_data));
              float v680_data = s2_w1[55];
              tensorforge::intel_esimd::simd<float, 8> v682_data(ir3.template select<8, 1>(48));
              ir3.template select<8, 1>(48) = (v682_data + (v649_data * v680_data));
              float v685_data = s2_w1[63];
              tensorforge::intel_esimd::simd<float, 8> v687_data(ir3.template select<8, 1>(56));
              ir3.template select<8, 1>(56) = (v687_data + (v649_data * v685_data));
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v689_n0 = 0; v689_n0 < 1; ++v689_n0) {
                int32_t v691_a = v689_n0 * 8;
                #pragma unroll
                for (int32_t v690_n1 = 0; v690_n1 < 8; ++v690_n1) {
                  int32_t v693_a = v691_a + (v690_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v694_data(ir3.template select<8, 1>(v693_a));
                  tensorforge::intel_esimd::simd<float, 8> v695_data(r1.template select<8, 1>(v693_a));
                  r3.template select<8, 1>(v693_a) = (v695_data + v694_data);
                }
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v697_i0 = 0; v697_i0 < 1; ++v697_i0) {
                int32_t v699_a = v697_i0 * 8;
                #pragma unroll
                for (int32_t v698_i1 = 0; v698_i1 < 8; ++v698_i1) {
                  int32_t v701_a = v699_a + (v698_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v702_data(r3.template select<8, 1>(v701_a));
                  tensorforge::slmStore<float, 8>(s1 + (v701_a), v702_data);
                }
              }
              // glb_m4 = abs(s1)
              #pragma unroll
              for (int32_t v705_k0 = 0; v705_k0 < 1; ++v705_k0) {
                int32_t v707_lead = v705_k0 * 8;
                #pragma unroll
                for (int32_t v706_k1 = 0; v706_k1 < 8; ++v706_k1) {
                  int32_t v710_a = v707_lead + (v706_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v711_data = tensorforge::slmLoad<float, 8>(s1 + (v710_a));
                  (tensorforge::intel_esimd::abs(v711_data)).copy_to(glb_m4 + (v710_a));
                }
              }
            }
            tensorforge::prefetchRunsL2<256, 256, 256, 256>(&pf_glb_m0[0], &pf_glb_m1[0], &pf_glb_m2[0], &pf_glb_m3[0]);
          }
        }
      }
    });
  });
}


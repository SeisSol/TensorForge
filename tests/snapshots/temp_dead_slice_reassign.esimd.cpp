// === base name ===
kernel_5e60e00ca08c7339

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5e60e00ca08c7339 = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5e60e00ca08c7339(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5e60e00ca08c7339(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5e60e00ca08c7339(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5e60e00ca08c7339(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5e60e00ca08c7339(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5e60e00ca08c7339(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5e60e00ca08c7339(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 6×12(6×12) {0..6}×{0..12} strided
        //   m1 12×12(12×12) {0..12}×{0..12} strided
        //   m2 6×12(6×12) {0..6}×{0..12} strided
        //   m3 6×12(6×12) {0..6}×{0..12} strided
        //   m4 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{6..12}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j] = m2[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m3[i,j]
        //   m4[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 72 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 144 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 12; ++v23_i1) {
                tensorforge::intel_esimd::simd<float, 6> v28_data;
                v28_data.copy_from(glb_m0 + ((v23_i1 * 6)));
                r0.template select<6, 1>((v23_i1 * 16)) = v28_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v31_ld;
              v31_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v32_ld);
              tensorforge::intel_esimd::simd<float, 16> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v33_ld);
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v383_i1 = 0; v383_i1 < 12; ++v383_i1) {
                tensorforge::intel_esimd::simd<float, 6> v388_data;
                v388_data.copy_from(glb_m2 + ((v383_i1 * 6)));
                r2.template select<6, 1>((v383_i1 * 16)) = v388_data;
              }
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v47_acc{};
              tensorforge::intel_esimd::simd<float, 16> v51_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              float v52_bc = static_cast<float>(v51_data[0]);
              v47_acc += (v52_bc * v35_data);
              float v54_bc = static_cast<float>(v51_data[1]);
              v47_acc += (v54_bc * v36_data);
              float v56_bc = static_cast<float>(v51_data[2]);
              v47_acc += (v56_bc * v37_data);
              float v58_bc = static_cast<float>(v51_data[3]);
              v47_acc += (v58_bc * v38_data);
              float v60_bc = static_cast<float>(v51_data[4]);
              v47_acc += (v60_bc * v39_data);
              float v62_bc = static_cast<float>(v51_data[5]);
              v47_acc += (v62_bc * v40_data);
              float v64_bc = static_cast<float>(v51_data[6]);
              v47_acc += (v64_bc * v41_data);
              float v66_bc = static_cast<float>(v51_data[7]);
              v47_acc += (v66_bc * v42_data);
              float v68_bc = static_cast<float>(v51_data[8]);
              v47_acc += (v68_bc * v43_data);
              float v70_bc = static_cast<float>(v51_data[9]);
              v47_acc += (v70_bc * v44_data);
              float v72_bc = static_cast<float>(v51_data[10]);
              v47_acc += (v72_bc * v45_data);
              float v74_bc = static_cast<float>(v51_data[11]);
              v47_acc += (v74_bc * v46_data);
              r1.template select<16, 1>(0) = v47_acc;
              tensorforge::intel_esimd::simd<float, 16> v76_acc{};
              tensorforge::intel_esimd::simd<float, 16> v78_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              float v79_bc = static_cast<float>(v78_data[0]);
              v76_acc += (v79_bc * v35_data);
              float v81_bc = static_cast<float>(v78_data[1]);
              v76_acc += (v81_bc * v36_data);
              float v83_bc = static_cast<float>(v78_data[2]);
              v76_acc += (v83_bc * v37_data);
              float v85_bc = static_cast<float>(v78_data[3]);
              v76_acc += (v85_bc * v38_data);
              float v87_bc = static_cast<float>(v78_data[4]);
              v76_acc += (v87_bc * v39_data);
              float v89_bc = static_cast<float>(v78_data[5]);
              v76_acc += (v89_bc * v40_data);
              float v91_bc = static_cast<float>(v78_data[6]);
              v76_acc += (v91_bc * v41_data);
              float v93_bc = static_cast<float>(v78_data[7]);
              v76_acc += (v93_bc * v42_data);
              float v95_bc = static_cast<float>(v78_data[8]);
              v76_acc += (v95_bc * v43_data);
              float v97_bc = static_cast<float>(v78_data[9]);
              v76_acc += (v97_bc * v44_data);
              float v99_bc = static_cast<float>(v78_data[10]);
              v76_acc += (v99_bc * v45_data);
              float v101_bc = static_cast<float>(v78_data[11]);
              v76_acc += (v101_bc * v46_data);
              r1.template select<16, 1>(16) = v76_acc;
              tensorforge::intel_esimd::simd<float, 16> v103_acc{};
              tensorforge::intel_esimd::simd<float, 16> v105_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              float v106_bc = static_cast<float>(v105_data[0]);
              v103_acc += (v106_bc * v35_data);
              float v108_bc = static_cast<float>(v105_data[1]);
              v103_acc += (v108_bc * v36_data);
              float v110_bc = static_cast<float>(v105_data[2]);
              v103_acc += (v110_bc * v37_data);
              float v112_bc = static_cast<float>(v105_data[3]);
              v103_acc += (v112_bc * v38_data);
              float v114_bc = static_cast<float>(v105_data[4]);
              v103_acc += (v114_bc * v39_data);
              float v116_bc = static_cast<float>(v105_data[5]);
              v103_acc += (v116_bc * v40_data);
              float v118_bc = static_cast<float>(v105_data[6]);
              v103_acc += (v118_bc * v41_data);
              float v120_bc = static_cast<float>(v105_data[7]);
              v103_acc += (v120_bc * v42_data);
              float v122_bc = static_cast<float>(v105_data[8]);
              v103_acc += (v122_bc * v43_data);
              float v124_bc = static_cast<float>(v105_data[9]);
              v103_acc += (v124_bc * v44_data);
              float v126_bc = static_cast<float>(v105_data[10]);
              v103_acc += (v126_bc * v45_data);
              float v128_bc = static_cast<float>(v105_data[11]);
              v103_acc += (v128_bc * v46_data);
              r1.template select<16, 1>(32) = v103_acc;
              tensorforge::intel_esimd::simd<float, 16> v130_acc{};
              tensorforge::intel_esimd::simd<float, 16> v132_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              float v133_bc = static_cast<float>(v132_data[0]);
              v130_acc += (v133_bc * v35_data);
              float v135_bc = static_cast<float>(v132_data[1]);
              v130_acc += (v135_bc * v36_data);
              float v137_bc = static_cast<float>(v132_data[2]);
              v130_acc += (v137_bc * v37_data);
              float v139_bc = static_cast<float>(v132_data[3]);
              v130_acc += (v139_bc * v38_data);
              float v141_bc = static_cast<float>(v132_data[4]);
              v130_acc += (v141_bc * v39_data);
              float v143_bc = static_cast<float>(v132_data[5]);
              v130_acc += (v143_bc * v40_data);
              float v145_bc = static_cast<float>(v132_data[6]);
              v130_acc += (v145_bc * v41_data);
              float v147_bc = static_cast<float>(v132_data[7]);
              v130_acc += (v147_bc * v42_data);
              float v149_bc = static_cast<float>(v132_data[8]);
              v130_acc += (v149_bc * v43_data);
              float v151_bc = static_cast<float>(v132_data[9]);
              v130_acc += (v151_bc * v44_data);
              float v153_bc = static_cast<float>(v132_data[10]);
              v130_acc += (v153_bc * v45_data);
              float v155_bc = static_cast<float>(v132_data[11]);
              v130_acc += (v155_bc * v46_data);
              r1.template select<16, 1>(48) = v130_acc;
              tensorforge::intel_esimd::simd<float, 16> v157_acc{};
              tensorforge::intel_esimd::simd<float, 16> v159_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              float v160_bc = static_cast<float>(v159_data[0]);
              v157_acc += (v160_bc * v35_data);
              float v162_bc = static_cast<float>(v159_data[1]);
              v157_acc += (v162_bc * v36_data);
              float v164_bc = static_cast<float>(v159_data[2]);
              v157_acc += (v164_bc * v37_data);
              float v166_bc = static_cast<float>(v159_data[3]);
              v157_acc += (v166_bc * v38_data);
              float v168_bc = static_cast<float>(v159_data[4]);
              v157_acc += (v168_bc * v39_data);
              float v170_bc = static_cast<float>(v159_data[5]);
              v157_acc += (v170_bc * v40_data);
              float v172_bc = static_cast<float>(v159_data[6]);
              v157_acc += (v172_bc * v41_data);
              float v174_bc = static_cast<float>(v159_data[7]);
              v157_acc += (v174_bc * v42_data);
              float v176_bc = static_cast<float>(v159_data[8]);
              v157_acc += (v176_bc * v43_data);
              float v178_bc = static_cast<float>(v159_data[9]);
              v157_acc += (v178_bc * v44_data);
              float v180_bc = static_cast<float>(v159_data[10]);
              v157_acc += (v180_bc * v45_data);
              float v182_bc = static_cast<float>(v159_data[11]);
              v157_acc += (v182_bc * v46_data);
              r1.template select<16, 1>(64) = v157_acc;
              tensorforge::intel_esimd::simd<float, 16> v184_acc{};
              tensorforge::intel_esimd::simd<float, 16> v186_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              float v187_bc = static_cast<float>(v186_data[0]);
              v184_acc += (v187_bc * v35_data);
              float v189_bc = static_cast<float>(v186_data[1]);
              v184_acc += (v189_bc * v36_data);
              float v191_bc = static_cast<float>(v186_data[2]);
              v184_acc += (v191_bc * v37_data);
              float v193_bc = static_cast<float>(v186_data[3]);
              v184_acc += (v193_bc * v38_data);
              float v195_bc = static_cast<float>(v186_data[4]);
              v184_acc += (v195_bc * v39_data);
              float v197_bc = static_cast<float>(v186_data[5]);
              v184_acc += (v197_bc * v40_data);
              float v199_bc = static_cast<float>(v186_data[6]);
              v184_acc += (v199_bc * v41_data);
              float v201_bc = static_cast<float>(v186_data[7]);
              v184_acc += (v201_bc * v42_data);
              float v203_bc = static_cast<float>(v186_data[8]);
              v184_acc += (v203_bc * v43_data);
              float v205_bc = static_cast<float>(v186_data[9]);
              v184_acc += (v205_bc * v44_data);
              float v207_bc = static_cast<float>(v186_data[10]);
              v184_acc += (v207_bc * v45_data);
              float v209_bc = static_cast<float>(v186_data[11]);
              v184_acc += (v209_bc * v46_data);
              r1.template select<16, 1>(80) = v184_acc;
              tensorforge::intel_esimd::simd<float, 16> v211_acc{};
              tensorforge::intel_esimd::simd<float, 16> v213_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              float v214_bc = static_cast<float>(v213_data[0]);
              v211_acc += (v214_bc * v35_data);
              float v216_bc = static_cast<float>(v213_data[1]);
              v211_acc += (v216_bc * v36_data);
              float v218_bc = static_cast<float>(v213_data[2]);
              v211_acc += (v218_bc * v37_data);
              float v220_bc = static_cast<float>(v213_data[3]);
              v211_acc += (v220_bc * v38_data);
              float v222_bc = static_cast<float>(v213_data[4]);
              v211_acc += (v222_bc * v39_data);
              float v224_bc = static_cast<float>(v213_data[5]);
              v211_acc += (v224_bc * v40_data);
              float v226_bc = static_cast<float>(v213_data[6]);
              v211_acc += (v226_bc * v41_data);
              float v228_bc = static_cast<float>(v213_data[7]);
              v211_acc += (v228_bc * v42_data);
              float v230_bc = static_cast<float>(v213_data[8]);
              v211_acc += (v230_bc * v43_data);
              float v232_bc = static_cast<float>(v213_data[9]);
              v211_acc += (v232_bc * v44_data);
              float v234_bc = static_cast<float>(v213_data[10]);
              v211_acc += (v234_bc * v45_data);
              float v236_bc = static_cast<float>(v213_data[11]);
              v211_acc += (v236_bc * v46_data);
              r1.template select<16, 1>(96) = v211_acc;
              tensorforge::intel_esimd::simd<float, 16> v238_acc{};
              tensorforge::intel_esimd::simd<float, 16> v240_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              float v241_bc = static_cast<float>(v240_data[0]);
              v238_acc += (v241_bc * v35_data);
              float v243_bc = static_cast<float>(v240_data[1]);
              v238_acc += (v243_bc * v36_data);
              float v245_bc = static_cast<float>(v240_data[2]);
              v238_acc += (v245_bc * v37_data);
              float v247_bc = static_cast<float>(v240_data[3]);
              v238_acc += (v247_bc * v38_data);
              float v249_bc = static_cast<float>(v240_data[4]);
              v238_acc += (v249_bc * v39_data);
              float v251_bc = static_cast<float>(v240_data[5]);
              v238_acc += (v251_bc * v40_data);
              float v253_bc = static_cast<float>(v240_data[6]);
              v238_acc += (v253_bc * v41_data);
              float v255_bc = static_cast<float>(v240_data[7]);
              v238_acc += (v255_bc * v42_data);
              float v257_bc = static_cast<float>(v240_data[8]);
              v238_acc += (v257_bc * v43_data);
              float v259_bc = static_cast<float>(v240_data[9]);
              v238_acc += (v259_bc * v44_data);
              float v261_bc = static_cast<float>(v240_data[10]);
              v238_acc += (v261_bc * v45_data);
              float v263_bc = static_cast<float>(v240_data[11]);
              v238_acc += (v263_bc * v46_data);
              r1.template select<16, 1>(112) = v238_acc;
              tensorforge::intel_esimd::simd<float, 16> v265_acc{};
              tensorforge::intel_esimd::simd<float, 16> v267_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              float v268_bc = static_cast<float>(v267_data[0]);
              v265_acc += (v268_bc * v35_data);
              float v270_bc = static_cast<float>(v267_data[1]);
              v265_acc += (v270_bc * v36_data);
              float v272_bc = static_cast<float>(v267_data[2]);
              v265_acc += (v272_bc * v37_data);
              float v274_bc = static_cast<float>(v267_data[3]);
              v265_acc += (v274_bc * v38_data);
              float v276_bc = static_cast<float>(v267_data[4]);
              v265_acc += (v276_bc * v39_data);
              float v278_bc = static_cast<float>(v267_data[5]);
              v265_acc += (v278_bc * v40_data);
              float v280_bc = static_cast<float>(v267_data[6]);
              v265_acc += (v280_bc * v41_data);
              float v282_bc = static_cast<float>(v267_data[7]);
              v265_acc += (v282_bc * v42_data);
              float v284_bc = static_cast<float>(v267_data[8]);
              v265_acc += (v284_bc * v43_data);
              float v286_bc = static_cast<float>(v267_data[9]);
              v265_acc += (v286_bc * v44_data);
              float v288_bc = static_cast<float>(v267_data[10]);
              v265_acc += (v288_bc * v45_data);
              float v290_bc = static_cast<float>(v267_data[11]);
              v265_acc += (v290_bc * v46_data);
              r1.template select<16, 1>(128) = v265_acc;
              tensorforge::intel_esimd::simd<float, 16> v292_acc{};
              tensorforge::intel_esimd::simd<float, 16> v294_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              float v295_bc = static_cast<float>(v294_data[0]);
              v292_acc += (v295_bc * v35_data);
              float v297_bc = static_cast<float>(v294_data[1]);
              v292_acc += (v297_bc * v36_data);
              float v299_bc = static_cast<float>(v294_data[2]);
              v292_acc += (v299_bc * v37_data);
              float v301_bc = static_cast<float>(v294_data[3]);
              v292_acc += (v301_bc * v38_data);
              float v303_bc = static_cast<float>(v294_data[4]);
              v292_acc += (v303_bc * v39_data);
              float v305_bc = static_cast<float>(v294_data[5]);
              v292_acc += (v305_bc * v40_data);
              float v307_bc = static_cast<float>(v294_data[6]);
              v292_acc += (v307_bc * v41_data);
              float v309_bc = static_cast<float>(v294_data[7]);
              v292_acc += (v309_bc * v42_data);
              float v311_bc = static_cast<float>(v294_data[8]);
              v292_acc += (v311_bc * v43_data);
              float v313_bc = static_cast<float>(v294_data[9]);
              v292_acc += (v313_bc * v44_data);
              float v315_bc = static_cast<float>(v294_data[10]);
              v292_acc += (v315_bc * v45_data);
              float v317_bc = static_cast<float>(v294_data[11]);
              v292_acc += (v317_bc * v46_data);
              r1.template select<16, 1>(144) = v292_acc;
              tensorforge::intel_esimd::simd<float, 16> v319_acc{};
              tensorforge::intel_esimd::simd<float, 16> v321_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              float v322_bc = static_cast<float>(v321_data[0]);
              v319_acc += (v322_bc * v35_data);
              float v324_bc = static_cast<float>(v321_data[1]);
              v319_acc += (v324_bc * v36_data);
              float v326_bc = static_cast<float>(v321_data[2]);
              v319_acc += (v326_bc * v37_data);
              float v328_bc = static_cast<float>(v321_data[3]);
              v319_acc += (v328_bc * v38_data);
              float v330_bc = static_cast<float>(v321_data[4]);
              v319_acc += (v330_bc * v39_data);
              float v332_bc = static_cast<float>(v321_data[5]);
              v319_acc += (v332_bc * v40_data);
              float v334_bc = static_cast<float>(v321_data[6]);
              v319_acc += (v334_bc * v41_data);
              float v336_bc = static_cast<float>(v321_data[7]);
              v319_acc += (v336_bc * v42_data);
              float v338_bc = static_cast<float>(v321_data[8]);
              v319_acc += (v338_bc * v43_data);
              float v340_bc = static_cast<float>(v321_data[9]);
              v319_acc += (v340_bc * v44_data);
              float v342_bc = static_cast<float>(v321_data[10]);
              v319_acc += (v342_bc * v45_data);
              float v344_bc = static_cast<float>(v321_data[11]);
              v319_acc += (v344_bc * v46_data);
              r1.template select<16, 1>(160) = v319_acc;
              tensorforge::intel_esimd::simd<float, 16> v346_acc{};
              tensorforge::intel_esimd::simd<float, 16> v348_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              float v349_bc = static_cast<float>(v348_data[0]);
              v346_acc += (v349_bc * v35_data);
              float v351_bc = static_cast<float>(v348_data[1]);
              v346_acc += (v351_bc * v36_data);
              float v353_bc = static_cast<float>(v348_data[2]);
              v346_acc += (v353_bc * v37_data);
              float v355_bc = static_cast<float>(v348_data[3]);
              v346_acc += (v355_bc * v38_data);
              float v357_bc = static_cast<float>(v348_data[4]);
              v346_acc += (v357_bc * v39_data);
              float v359_bc = static_cast<float>(v348_data[5]);
              v346_acc += (v359_bc * v40_data);
              float v361_bc = static_cast<float>(v348_data[6]);
              v346_acc += (v361_bc * v41_data);
              float v363_bc = static_cast<float>(v348_data[7]);
              v346_acc += (v363_bc * v42_data);
              float v365_bc = static_cast<float>(v348_data[8]);
              v346_acc += (v365_bc * v43_data);
              float v367_bc = static_cast<float>(v348_data[9]);
              v346_acc += (v367_bc * v44_data);
              float v369_bc = static_cast<float>(v348_data[10]);
              v346_acc += (v369_bc * v45_data);
              float v371_bc = static_cast<float>(v348_data[11]);
              v346_acc += (v371_bc * v46_data);
              r1.template select<16, 1>(176) = v346_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v373_i1 = 0; v373_i1 < 12; ++v373_i1) {
                tensorforge::intel_esimd::simd<float, 6> v376_data(r1.template select<6, 1>((v373_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v373_i1 * 12))), v376_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v751_i1 = 0; v751_i1 < 12; ++v751_i1) {
                tensorforge::intel_esimd::simd<float, 6> v756_data;
                v756_data.copy_from(glb_m3 + ((v751_i1 * 6)));
                r4.template select<6, 1>((v751_i1 * 16)) = v756_data;
              }
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s0)
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v393_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v394_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v395_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v396_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v397_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v398_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v399_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v400_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v401_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v402_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v403_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v404_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v405_acc{};
              v405_acc += (v52_bc * v393_data);
              v405_acc += (v54_bc * v394_data);
              v405_acc += (v56_bc * v395_data);
              v405_acc += (v58_bc * v396_data);
              v405_acc += (v60_bc * v397_data);
              v405_acc += (v62_bc * v398_data);
              v405_acc += (v64_bc * v399_data);
              v405_acc += (v66_bc * v400_data);
              v405_acc += (v68_bc * v401_data);
              v405_acc += (v70_bc * v402_data);
              v405_acc += (v72_bc * v403_data);
              v405_acc += (v74_bc * v404_data);
              ir3.template select<16, 1>(0) = v405_acc;
              tensorforge::intel_esimd::simd<float, 16> v434_acc{};
              v434_acc += (v79_bc * v393_data);
              v434_acc += (v81_bc * v394_data);
              v434_acc += (v83_bc * v395_data);
              v434_acc += (v85_bc * v396_data);
              v434_acc += (v87_bc * v397_data);
              v434_acc += (v89_bc * v398_data);
              v434_acc += (v91_bc * v399_data);
              v434_acc += (v93_bc * v400_data);
              v434_acc += (v95_bc * v401_data);
              v434_acc += (v97_bc * v402_data);
              v434_acc += (v99_bc * v403_data);
              v434_acc += (v101_bc * v404_data);
              ir3.template select<16, 1>(16) = v434_acc;
              tensorforge::intel_esimd::simd<float, 16> v461_acc{};
              v461_acc += (v106_bc * v393_data);
              v461_acc += (v108_bc * v394_data);
              v461_acc += (v110_bc * v395_data);
              v461_acc += (v112_bc * v396_data);
              v461_acc += (v114_bc * v397_data);
              v461_acc += (v116_bc * v398_data);
              v461_acc += (v118_bc * v399_data);
              v461_acc += (v120_bc * v400_data);
              v461_acc += (v122_bc * v401_data);
              v461_acc += (v124_bc * v402_data);
              v461_acc += (v126_bc * v403_data);
              v461_acc += (v128_bc * v404_data);
              ir3.template select<16, 1>(32) = v461_acc;
              tensorforge::intel_esimd::simd<float, 16> v488_acc{};
              v488_acc += (v133_bc * v393_data);
              v488_acc += (v135_bc * v394_data);
              v488_acc += (v137_bc * v395_data);
              v488_acc += (v139_bc * v396_data);
              v488_acc += (v141_bc * v397_data);
              v488_acc += (v143_bc * v398_data);
              v488_acc += (v145_bc * v399_data);
              v488_acc += (v147_bc * v400_data);
              v488_acc += (v149_bc * v401_data);
              v488_acc += (v151_bc * v402_data);
              v488_acc += (v153_bc * v403_data);
              v488_acc += (v155_bc * v404_data);
              ir3.template select<16, 1>(48) = v488_acc;
              tensorforge::intel_esimd::simd<float, 16> v515_acc{};
              v515_acc += (v160_bc * v393_data);
              v515_acc += (v162_bc * v394_data);
              v515_acc += (v164_bc * v395_data);
              v515_acc += (v166_bc * v396_data);
              v515_acc += (v168_bc * v397_data);
              v515_acc += (v170_bc * v398_data);
              v515_acc += (v172_bc * v399_data);
              v515_acc += (v174_bc * v400_data);
              v515_acc += (v176_bc * v401_data);
              v515_acc += (v178_bc * v402_data);
              v515_acc += (v180_bc * v403_data);
              v515_acc += (v182_bc * v404_data);
              ir3.template select<16, 1>(64) = v515_acc;
              tensorforge::intel_esimd::simd<float, 16> v542_acc{};
              v542_acc += (v187_bc * v393_data);
              v542_acc += (v189_bc * v394_data);
              v542_acc += (v191_bc * v395_data);
              v542_acc += (v193_bc * v396_data);
              v542_acc += (v195_bc * v397_data);
              v542_acc += (v197_bc * v398_data);
              v542_acc += (v199_bc * v399_data);
              v542_acc += (v201_bc * v400_data);
              v542_acc += (v203_bc * v401_data);
              v542_acc += (v205_bc * v402_data);
              v542_acc += (v207_bc * v403_data);
              v542_acc += (v209_bc * v404_data);
              ir3.template select<16, 1>(80) = v542_acc;
              tensorforge::intel_esimd::simd<float, 16> v569_acc{};
              v569_acc += (v214_bc * v393_data);
              v569_acc += (v216_bc * v394_data);
              v569_acc += (v218_bc * v395_data);
              v569_acc += (v220_bc * v396_data);
              v569_acc += (v222_bc * v397_data);
              v569_acc += (v224_bc * v398_data);
              v569_acc += (v226_bc * v399_data);
              v569_acc += (v228_bc * v400_data);
              v569_acc += (v230_bc * v401_data);
              v569_acc += (v232_bc * v402_data);
              v569_acc += (v234_bc * v403_data);
              v569_acc += (v236_bc * v404_data);
              ir3.template select<16, 1>(96) = v569_acc;
              tensorforge::intel_esimd::simd<float, 16> v596_acc{};
              v596_acc += (v241_bc * v393_data);
              v596_acc += (v243_bc * v394_data);
              v596_acc += (v245_bc * v395_data);
              v596_acc += (v247_bc * v396_data);
              v596_acc += (v249_bc * v397_data);
              v596_acc += (v251_bc * v398_data);
              v596_acc += (v253_bc * v399_data);
              v596_acc += (v255_bc * v400_data);
              v596_acc += (v257_bc * v401_data);
              v596_acc += (v259_bc * v402_data);
              v596_acc += (v261_bc * v403_data);
              v596_acc += (v263_bc * v404_data);
              ir3.template select<16, 1>(112) = v596_acc;
              tensorforge::intel_esimd::simd<float, 16> v623_acc{};
              v623_acc += (v268_bc * v393_data);
              v623_acc += (v270_bc * v394_data);
              v623_acc += (v272_bc * v395_data);
              v623_acc += (v274_bc * v396_data);
              v623_acc += (v276_bc * v397_data);
              v623_acc += (v278_bc * v398_data);
              v623_acc += (v280_bc * v399_data);
              v623_acc += (v282_bc * v400_data);
              v623_acc += (v284_bc * v401_data);
              v623_acc += (v286_bc * v402_data);
              v623_acc += (v288_bc * v403_data);
              v623_acc += (v290_bc * v404_data);
              ir3.template select<16, 1>(128) = v623_acc;
              tensorforge::intel_esimd::simd<float, 16> v650_acc{};
              v650_acc += (v295_bc * v393_data);
              v650_acc += (v297_bc * v394_data);
              v650_acc += (v299_bc * v395_data);
              v650_acc += (v301_bc * v396_data);
              v650_acc += (v303_bc * v397_data);
              v650_acc += (v305_bc * v398_data);
              v650_acc += (v307_bc * v399_data);
              v650_acc += (v309_bc * v400_data);
              v650_acc += (v311_bc * v401_data);
              v650_acc += (v313_bc * v402_data);
              v650_acc += (v315_bc * v403_data);
              v650_acc += (v317_bc * v404_data);
              ir3.template select<16, 1>(144) = v650_acc;
              tensorforge::intel_esimd::simd<float, 16> v677_acc{};
              v677_acc += (v322_bc * v393_data);
              v677_acc += (v324_bc * v394_data);
              v677_acc += (v326_bc * v395_data);
              v677_acc += (v328_bc * v396_data);
              v677_acc += (v330_bc * v397_data);
              v677_acc += (v332_bc * v398_data);
              v677_acc += (v334_bc * v399_data);
              v677_acc += (v336_bc * v400_data);
              v677_acc += (v338_bc * v401_data);
              v677_acc += (v340_bc * v402_data);
              v677_acc += (v342_bc * v403_data);
              v677_acc += (v344_bc * v404_data);
              ir3.template select<16, 1>(160) = v677_acc;
              tensorforge::intel_esimd::simd<float, 16> v704_acc{};
              v704_acc += (v349_bc * v393_data);
              v704_acc += (v351_bc * v394_data);
              v704_acc += (v353_bc * v395_data);
              v704_acc += (v355_bc * v396_data);
              v704_acc += (v357_bc * v397_data);
              v704_acc += (v359_bc * v398_data);
              v704_acc += (v361_bc * v399_data);
              v704_acc += (v363_bc * v400_data);
              v704_acc += (v365_bc * v401_data);
              v704_acc += (v367_bc * v402_data);
              v704_acc += (v369_bc * v403_data);
              v704_acc += (v371_bc * v404_data);
              ir3.template select<16, 1>(176) = v704_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v731_n1 = 0; v731_n1 < 12; ++v731_n1) {
                int32_t v732_a = v731_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v734_data(ir3.template select<6, 1>(v732_a));
                r3.template select<6, 1>(v732_a) = v734_data;
              }
              // s1 = store{r>s, clear}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v735_z1 = 0; v735_z1 < 12; ++v735_z1) {
                s1[(6_i32 + (v735_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v742_i1 = 0; v742_i1 < 12; ++v742_i1) {
                tensorforge::intel_esimd::simd<float, 6> v745_data(r3.template select<6, 1>((v742_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((v742_i1 * 12)), v745_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r5(0.0f);
              // ir5 = +(r4)
              // [(0, 6), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v761_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v762_data(ir5.template select<16, 1>(0));
              ir5.template select<16, 1>(0) = (v762_data + v761_data);
              tensorforge::intel_esimd::simd<float, 16> v764_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v765_data(ir5.template select<16, 1>(16));
              ir5.template select<16, 1>(16) = (v765_data + v764_data);
              tensorforge::intel_esimd::simd<float, 16> v767_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v768_data(ir5.template select<16, 1>(32));
              ir5.template select<16, 1>(32) = (v768_data + v767_data);
              tensorforge::intel_esimd::simd<float, 16> v770_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v771_data(ir5.template select<16, 1>(48));
              ir5.template select<16, 1>(48) = (v771_data + v770_data);
              tensorforge::intel_esimd::simd<float, 16> v773_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v774_data(ir5.template select<16, 1>(64));
              ir5.template select<16, 1>(64) = (v774_data + v773_data);
              tensorforge::intel_esimd::simd<float, 16> v776_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v777_data(ir5.template select<16, 1>(80));
              ir5.template select<16, 1>(80) = (v777_data + v776_data);
              tensorforge::intel_esimd::simd<float, 16> v779_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v780_data(ir5.template select<16, 1>(96));
              ir5.template select<16, 1>(96) = (v780_data + v779_data);
              tensorforge::intel_esimd::simd<float, 16> v782_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v783_data(ir5.template select<16, 1>(112));
              ir5.template select<16, 1>(112) = (v783_data + v782_data);
              tensorforge::intel_esimd::simd<float, 16> v785_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v786_data(ir5.template select<16, 1>(128));
              ir5.template select<16, 1>(128) = (v786_data + v785_data);
              tensorforge::intel_esimd::simd<float, 16> v788_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v789_data(ir5.template select<16, 1>(144));
              ir5.template select<16, 1>(144) = (v789_data + v788_data);
              tensorforge::intel_esimd::simd<float, 16> v791_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v792_data(ir5.template select<16, 1>(160));
              ir5.template select<16, 1>(160) = (v792_data + v791_data);
              tensorforge::intel_esimd::simd<float, 16> v794_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v795_data(ir5.template select<16, 1>(176));
              ir5.template select<16, 1>(176) = (v795_data + v794_data);
              // r5 = ir5
              #pragma unroll
              for (int32_t v797_n1 = 0; v797_n1 < 12; ++v797_n1) {
                int32_t v798_a = v797_n1 * 16;
                tensorforge::intel_esimd::simd<float, 6> v800_data(ir5.template select<6, 1>(v798_a));
                r5.template select<6, 1>(v798_a) = v800_data;
              }
              // s1 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v801_i1 = 0; v801_i1 < 12; ++v801_i1) {
                tensorforge::intel_esimd::simd<float, 6> v804_data(r5.template select<6, 1>((v801_i1 * 16)));
                tensorforge::slmStore<float, 6>(s1 + ((6_i32 + (v801_i1 * 12))), v804_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // ir6 = +(s1)
              // [(0, 12), (0, 12)] []
              tensorforge::intel_esimd::simd<float, 192> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v815_data(0.0f);
              v815_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v816_data(ir6.template select<16, 1>(0));
              ir6.template select<16, 1>(0) = (v816_data + v815_data);
              tensorforge::intel_esimd::simd<float, 16> v819_data(0.0f);
              v819_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v820_data(ir6.template select<16, 1>(16));
              ir6.template select<16, 1>(16) = (v820_data + v819_data);
              tensorforge::intel_esimd::simd<float, 16> v823_data(0.0f);
              v823_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v824_data(ir6.template select<16, 1>(32));
              ir6.template select<16, 1>(32) = (v824_data + v823_data);
              tensorforge::intel_esimd::simd<float, 16> v827_data(0.0f);
              v827_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v828_data(ir6.template select<16, 1>(48));
              ir6.template select<16, 1>(48) = (v828_data + v827_data);
              tensorforge::intel_esimd::simd<float, 16> v831_data(0.0f);
              v831_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v832_data(ir6.template select<16, 1>(64));
              ir6.template select<16, 1>(64) = (v832_data + v831_data);
              tensorforge::intel_esimd::simd<float, 16> v835_data(0.0f);
              v835_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v836_data(ir6.template select<16, 1>(80));
              ir6.template select<16, 1>(80) = (v836_data + v835_data);
              tensorforge::intel_esimd::simd<float, 16> v839_data(0.0f);
              v839_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v840_data(ir6.template select<16, 1>(96));
              ir6.template select<16, 1>(96) = (v840_data + v839_data);
              tensorforge::intel_esimd::simd<float, 16> v843_data(0.0f);
              v843_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v844_data(ir6.template select<16, 1>(112));
              ir6.template select<16, 1>(112) = (v844_data + v843_data);
              tensorforge::intel_esimd::simd<float, 16> v847_data(0.0f);
              v847_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v848_data(ir6.template select<16, 1>(128));
              ir6.template select<16, 1>(128) = (v848_data + v847_data);
              tensorforge::intel_esimd::simd<float, 16> v851_data(0.0f);
              v851_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v852_data(ir6.template select<16, 1>(144));
              ir6.template select<16, 1>(144) = (v852_data + v851_data);
              tensorforge::intel_esimd::simd<float, 16> v855_data(0.0f);
              v855_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v856_data(ir6.template select<16, 1>(160));
              ir6.template select<16, 1>(160) = (v856_data + v855_data);
              tensorforge::intel_esimd::simd<float, 16> v859_data(0.0f);
              v859_data.template select<12, 1>(0) = tensorforge::slmLoad<float, 12>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v860_data(ir6.template select<16, 1>(176));
              ir6.template select<16, 1>(176) = (v860_data + v859_data);
              // r6 = ir6
              #pragma unroll
              for (int32_t v862_n1 = 0; v862_n1 < 12; ++v862_n1) {
                int32_t v863_a = v862_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v865_data(ir6.template select<12, 1>(v863_a));
                r6.template select<12, 1>(v863_a) = v865_data;
              }
              // glb_m4 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v866_i1 = 0; v866_i1 < 12; ++v866_i1) {
                tensorforge::intel_esimd::simd<float, 12> v869_data(r6.template select<12, 1>((v866_i1 * 16)));
                v869_data.copy_to(glb_m4 + ((v866_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


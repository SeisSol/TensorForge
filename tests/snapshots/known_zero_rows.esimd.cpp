// === base name ===
kernel_e0a716931c9b9546

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e0a716931c9b9546 = {{1, 32, 1}, 32, 40, 1, 32, 7424, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e0a716931c9b9546(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e0a716931c9b9546(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e0a716931c9b9546(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1856 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e0a716931c9b9546(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e0a716931c9b9546(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e0a716931c9b9546(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e0a716931c9b9546(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1856 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (40 active) x 32 per block = block 1x32x1, 7424 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":1856}],"shared_bytes":7424,"shared_elements":1856,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (48 * item.get_local_id(1) + 320);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (48);
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 32> v5_ld;
            v5_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v5_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 32> v6_ld;
            v6_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v6_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 32> v7_ld;
            v7_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v7_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 32> v8_ld;
            v8_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v8_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 32> v9_ld;
            v9_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v9_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 32> v10_ld;
            v10_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v10_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 32> v11_ld;
            v11_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v11_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 32> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 32> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 32> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v14_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v17_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v17_batchId0 < numElements0; v17_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v18_ahead1 = v17_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v20_batchId1 = (v18_ahead1 < numElements0) ? v18_ahead1 : v17_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v17_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v17_batchId0 * 240 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v17_batchId0 * 48 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v27_ld);
              tensorforge::intel_esimd::simd<float, 16> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v28_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 40), (0, 6)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 384> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v35_data(0.0f);
              v35_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 48> s0_w0 = tensorforge::slmLoad<float, 48>(s0 + 0);
              float v36_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v38_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v38_data + (v35_data * v36_data));
              float v41_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v43_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v43_data + (v35_data * v41_data));
              float v46_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v48_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v48_data + (v35_data * v46_data));
              float v51_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v53_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v53_data + (v35_data * v51_data));
              float v56_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v58_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v58_data + (v35_data * v56_data));
              float v61_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v63_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v63_data + (v35_data * v61_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 32> v66_data(glb_m1_run0.template select<32, 1>(0));
              float v67_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v69_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v69_data + (v66_data * v67_data));
              float v72_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v74_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v74_data + (v66_data * v72_data));
              float v77_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v79_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v79_data + (v66_data * v77_data));
              float v82_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v84_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v84_data + (v66_data * v82_data));
              float v87_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v89_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v89_data + (v66_data * v87_data));
              float v92_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v94_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v94_data + (v66_data * v92_data));
              tensorforge::intel_esimd::simd<float, 32> v97_data(glb_m1_run0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v100_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v100_data + (v97_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v105_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v105_data + (v97_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v110_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v110_data + (v97_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v115_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v115_data + (v97_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v120_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v120_data + (v97_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v125_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v125_data + (v97_data * v92_data));
              tensorforge::intel_esimd::simd<float, 32> v128_data(0.0f);
              v128_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (112_i32));
              float v129_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v131_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v131_data + (v128_data * v129_data));
              float v134_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v136_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v136_data + (v128_data * v134_data));
              float v139_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v141_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v141_data + (v128_data * v139_data));
              float v144_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v146_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v146_data + (v128_data * v144_data));
              float v149_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v151_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v151_data + (v128_data * v149_data));
              float v154_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v156_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v156_data + (v128_data * v154_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 32> v159_data(glb_m1_run1.template select<32, 1>(0));
              float v160_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v162_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v162_data + (v159_data * v160_data));
              float v165_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v167_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v167_data + (v159_data * v165_data));
              float v170_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v172_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v172_data + (v159_data * v170_data));
              float v175_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v177_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v177_data + (v159_data * v175_data));
              float v180_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v182_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v182_data + (v159_data * v180_data));
              float v185_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v187_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v187_data + (v159_data * v185_data));
              tensorforge::intel_esimd::simd<float, 32> v190_data(glb_m1_run1.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v193_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v193_data + (v190_data * v160_data));
              tensorforge::intel_esimd::simd<float, 32> v198_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v198_data + (v190_data * v165_data));
              tensorforge::intel_esimd::simd<float, 32> v203_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v203_data + (v190_data * v170_data));
              tensorforge::intel_esimd::simd<float, 32> v208_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v208_data + (v190_data * v175_data));
              tensorforge::intel_esimd::simd<float, 32> v213_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v213_data + (v190_data * v180_data));
              tensorforge::intel_esimd::simd<float, 32> v218_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v218_data + (v190_data * v185_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 32> v221_data(glb_m1_run2.template select<32, 1>(0));
              float v222_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v224_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v224_data + (v221_data * v222_data));
              float v227_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v229_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v229_data + (v221_data * v227_data));
              float v232_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v234_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v234_data + (v221_data * v232_data));
              float v237_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v239_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v239_data + (v221_data * v237_data));
              float v242_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v244_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v244_data + (v221_data * v242_data));
              float v247_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v249_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v249_data + (v221_data * v247_data));
              tensorforge::intel_esimd::simd<float, 32> v252_data(glb_m1_run2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v255_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v255_data + (v252_data * v222_data));
              tensorforge::intel_esimd::simd<float, 32> v260_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v260_data + (v252_data * v227_data));
              tensorforge::intel_esimd::simd<float, 32> v265_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v265_data + (v252_data * v232_data));
              tensorforge::intel_esimd::simd<float, 32> v270_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v270_data + (v252_data * v237_data));
              tensorforge::intel_esimd::simd<float, 32> v275_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v275_data + (v252_data * v242_data));
              tensorforge::intel_esimd::simd<float, 32> v280_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v280_data + (v252_data * v247_data));
              tensorforge::intel_esimd::simd<float, 32> v283_data(0.0f);
              v283_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (232_i32));
              float v284_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v286_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v286_data + (v283_data * v284_data));
              float v289_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v291_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v291_data + (v283_data * v289_data));
              float v294_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v296_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v296_data + (v283_data * v294_data));
              float v299_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v301_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v301_data + (v283_data * v299_data));
              float v304_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v306_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v306_data + (v283_data * v304_data));
              float v309_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v311_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v311_data + (v283_data * v309_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 32> v314_data(glb_m1_run3.template select<32, 1>(0));
              float v315_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v317_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v317_data + (v314_data * v315_data));
              float v320_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v322_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v322_data + (v314_data * v320_data));
              float v325_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v327_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v327_data + (v314_data * v325_data));
              float v330_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v332_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v332_data + (v314_data * v330_data));
              float v335_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v337_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v337_data + (v314_data * v335_data));
              float v340_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v342_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v342_data + (v314_data * v340_data));
              tensorforge::intel_esimd::simd<float, 32> v345_data(glb_m1_run3.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v348_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v348_data + (v345_data * v315_data));
              tensorforge::intel_esimd::simd<float, 32> v353_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v353_data + (v345_data * v320_data));
              tensorforge::intel_esimd::simd<float, 32> v358_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v358_data + (v345_data * v325_data));
              tensorforge::intel_esimd::simd<float, 32> v363_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v363_data + (v345_data * v330_data));
              tensorforge::intel_esimd::simd<float, 32> v368_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v368_data + (v345_data * v335_data));
              tensorforge::intel_esimd::simd<float, 32> v373_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v373_data + (v345_data * v340_data));
              tensorforge::intel_esimd::simd<float, 32> v376_data = tensorforge::slmLoad<float, 32>(glb_m1 + (280_i32));
              float v377_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v379_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v379_data + (v376_data * v377_data));
              float v382_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v384_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v384_data + (v376_data * v382_data));
              float v387_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v389_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v389_data + (v376_data * v387_data));
              float v392_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v394_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v394_data + (v376_data * v392_data));
              float v397_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v399_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v399_data + (v376_data * v397_data));
              float v402_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v404_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v404_data + (v376_data * v402_data));
              tensorforge::intel_esimd::simd<float, 32> v407_data(0.0f);
              v407_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (312_i32));
              tensorforge::intel_esimd::simd<float, 32> v410_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v410_data + (v407_data * v377_data));
              tensorforge::intel_esimd::simd<float, 32> v415_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v415_data + (v407_data * v382_data));
              tensorforge::intel_esimd::simd<float, 32> v420_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v420_data + (v407_data * v387_data));
              tensorforge::intel_esimd::simd<float, 32> v425_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v425_data + (v407_data * v392_data));
              tensorforge::intel_esimd::simd<float, 32> v430_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v430_data + (v407_data * v397_data));
              tensorforge::intel_esimd::simd<float, 32> v435_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v435_data + (v407_data * v402_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v437_n0 = 0; v437_n0 < 1; ++v437_n0) {
                int32_t v439_a = v437_n0 * 32;
                #pragma unroll
                for (int32_t v438_n1 = 0; v438_n1 < 6; ++v438_n1) {
                  int32_t v441_a = v439_a + (v438_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v442_data(ir0.template select<32, 1>(v441_a));
                  r0.template select<32, 1>(v441_a) = v442_data;
                }
              }
              #pragma unroll
              for (int32_t v443_n1 = 0; v443_n1 < 6; ++v443_n1) {
                int32_t v445_a = 32 + (v443_n1 * 64);
                tensorforge::intel_esimd::simd<float, 8> v446_data(ir0.template select<8, 1>(v445_a));
                r0.template select<8, 1>(v445_a) = v446_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v447_i0 = 0; v447_i0 < 1; ++v447_i0) {
                int32_t v449_a = v447_i0 * 32;
                #pragma unroll
                for (int32_t v448_i1 = 0; v448_i1 < 6; ++v448_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v452_data(r0.template select<32, 1>((v449_a + (v448_i1 * 64))));
                  v452_data.copy_to(glb_m0 + ((v449_a + (v448_i1 * 40))));
                }
              }
              #pragma unroll
              for (int32_t v456_i1 = 0; v456_i1 < 6; ++v456_i1) {
                tensorforge::intel_esimd::simd<float, 8> v459_data(r0.template select<8, 1>((32 + (v456_i1 * 64))));
                v459_data.copy_to(glb_m0 + ((32_i32 + (v456_i1 * 40))));
              }
            }
          }
        }
      }
    });
  });
}


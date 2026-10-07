// === base name ===
kernel_93370731bb32fb1a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_93370731bb32fb1a = {{1, 32, 1}, 32, 40, 1, 32, 7424, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_93370731bb32fb1a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_93370731bb32fb1a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_93370731bb32fb1a(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_93370731bb32fb1a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_93370731bb32fb1a(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_93370731bb32fb1a(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_93370731bb32fb1a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1856 * sizeof(float)>(); {
        using namespace tensorforge::literals;
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
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (48 * item.get_local_id(1) + 320);
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 32> v9_ld;
            v9_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v9_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 32> v10_ld;
            v10_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v10_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 32> v11_ld;
            v11_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v11_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 32> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 32> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 32> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 32> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 32> v16_ld;
            v16_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 32> v17_ld;
            v17_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 32> v18_ld;
            v18_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v18_ld);
          }
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v20_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v20_batchId0 < numElements0; v20_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v21_ahead1 = v20_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v23_batchId1 = (v21_ahead1 < numElements0) ? v21_ahead1 : v20_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v20_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v20_batchId0 * 240 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v20_batchId0 * 48 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v30_ld);
              tensorforge::intel_esimd::simd<float, 16> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v31_ld);
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 40), (0, 6)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 384> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v38_data(0.0f);
              v38_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 48> s0_w0 = tensorforge::slmLoad<float, 48>(s0 + 0);
              float v39_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v41_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v41_data + (v38_data * v39_data));
              float v44_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v46_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v46_data + (v38_data * v44_data));
              float v49_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v51_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v51_data + (v38_data * v49_data));
              float v54_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v56_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v56_data + (v38_data * v54_data));
              float v59_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v61_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v61_data + (v38_data * v59_data));
              float v64_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v66_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v66_data + (v38_data * v64_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 32> v69_data(glb_m1_run0.template select<32, 1>(0));
              float v70_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v72_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v72_data + (v69_data * v70_data));
              float v75_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v77_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v77_data + (v69_data * v75_data));
              float v80_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v82_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v82_data + (v69_data * v80_data));
              float v85_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v87_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v87_data + (v69_data * v85_data));
              float v90_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v92_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v92_data + (v69_data * v90_data));
              float v95_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v97_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v97_data + (v69_data * v95_data));
              tensorforge::intel_esimd::simd<float, 32> v100_data(glb_m1_run0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v103_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v103_data + (v100_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v108_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v108_data + (v100_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v113_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v113_data + (v100_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v118_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v118_data + (v100_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v123_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v123_data + (v100_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v128_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v128_data + (v100_data * v95_data));
              tensorforge::intel_esimd::simd<float, 32> v131_data(0.0f);
              v131_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (112_i32));
              float v132_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v134_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v134_data + (v131_data * v132_data));
              float v137_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v139_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v139_data + (v131_data * v137_data));
              float v142_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v144_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v144_data + (v131_data * v142_data));
              float v147_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v149_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v149_data + (v131_data * v147_data));
              float v152_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v154_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v154_data + (v131_data * v152_data));
              float v157_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v159_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v159_data + (v131_data * v157_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 32> v162_data(glb_m1_run1.template select<32, 1>(0));
              float v163_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v165_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v165_data + (v162_data * v163_data));
              float v168_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v170_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v170_data + (v162_data * v168_data));
              float v173_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v175_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v175_data + (v162_data * v173_data));
              float v178_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v180_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v180_data + (v162_data * v178_data));
              float v183_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v185_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v185_data + (v162_data * v183_data));
              float v188_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v190_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v190_data + (v162_data * v188_data));
              tensorforge::intel_esimd::simd<float, 32> v193_data(glb_m1_run1.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v196_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v196_data + (v193_data * v163_data));
              tensorforge::intel_esimd::simd<float, 32> v201_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v201_data + (v193_data * v168_data));
              tensorforge::intel_esimd::simd<float, 32> v206_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v206_data + (v193_data * v173_data));
              tensorforge::intel_esimd::simd<float, 32> v211_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v211_data + (v193_data * v178_data));
              tensorforge::intel_esimd::simd<float, 32> v216_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v216_data + (v193_data * v183_data));
              tensorforge::intel_esimd::simd<float, 32> v221_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v221_data + (v193_data * v188_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 32> v224_data(glb_m1_run2.template select<32, 1>(0));
              float v225_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v227_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v227_data + (v224_data * v225_data));
              float v230_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v232_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v232_data + (v224_data * v230_data));
              float v235_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v237_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v237_data + (v224_data * v235_data));
              float v240_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v242_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v242_data + (v224_data * v240_data));
              float v245_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v247_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v247_data + (v224_data * v245_data));
              float v250_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v252_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v252_data + (v224_data * v250_data));
              tensorforge::intel_esimd::simd<float, 32> v255_data(glb_m1_run2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v258_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v258_data + (v255_data * v225_data));
              tensorforge::intel_esimd::simd<float, 32> v263_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v263_data + (v255_data * v230_data));
              tensorforge::intel_esimd::simd<float, 32> v268_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v268_data + (v255_data * v235_data));
              tensorforge::intel_esimd::simd<float, 32> v273_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v273_data + (v255_data * v240_data));
              tensorforge::intel_esimd::simd<float, 32> v278_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v278_data + (v255_data * v245_data));
              tensorforge::intel_esimd::simd<float, 32> v283_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v283_data + (v255_data * v250_data));
              tensorforge::intel_esimd::simd<float, 32> v286_data(0.0f);
              v286_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (232_i32));
              float v287_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v289_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v289_data + (v286_data * v287_data));
              float v292_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v294_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v294_data + (v286_data * v292_data));
              float v297_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v299_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v299_data + (v286_data * v297_data));
              float v302_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v304_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v304_data + (v286_data * v302_data));
              float v307_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v309_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v309_data + (v286_data * v307_data));
              float v312_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v314_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v314_data + (v286_data * v312_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 32> v317_data(glb_m1_run3.template select<32, 1>(0));
              float v318_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v320_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v320_data + (v317_data * v318_data));
              float v323_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v325_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v325_data + (v317_data * v323_data));
              float v328_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v330_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v330_data + (v317_data * v328_data));
              float v333_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v335_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v335_data + (v317_data * v333_data));
              float v338_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v340_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v340_data + (v317_data * v338_data));
              float v343_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v345_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v345_data + (v317_data * v343_data));
              tensorforge::intel_esimd::simd<float, 32> v348_data(glb_m1_run3.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v351_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v351_data + (v348_data * v318_data));
              tensorforge::intel_esimd::simd<float, 32> v356_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v356_data + (v348_data * v323_data));
              tensorforge::intel_esimd::simd<float, 32> v361_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v361_data + (v348_data * v328_data));
              tensorforge::intel_esimd::simd<float, 32> v366_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v366_data + (v348_data * v333_data));
              tensorforge::intel_esimd::simd<float, 32> v371_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v371_data + (v348_data * v338_data));
              tensorforge::intel_esimd::simd<float, 32> v376_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v376_data + (v348_data * v343_data));
              tensorforge::intel_esimd::simd<float, 32> v379_data = tensorforge::slmLoad<float, 32>(glb_m1 + (280_i32));
              float v380_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v382_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v382_data + (v379_data * v380_data));
              float v385_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v387_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v387_data + (v379_data * v385_data));
              float v390_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v392_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v392_data + (v379_data * v390_data));
              float v395_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v397_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v397_data + (v379_data * v395_data));
              float v400_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v402_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v402_data + (v379_data * v400_data));
              float v405_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v407_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v407_data + (v379_data * v405_data));
              tensorforge::intel_esimd::simd<float, 32> v410_data(0.0f);
              v410_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (312_i32));
              tensorforge::intel_esimd::simd<float, 32> v413_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v413_data + (v410_data * v380_data));
              tensorforge::intel_esimd::simd<float, 32> v418_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v418_data + (v410_data * v385_data));
              tensorforge::intel_esimd::simd<float, 32> v423_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v423_data + (v410_data * v390_data));
              tensorforge::intel_esimd::simd<float, 32> v428_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v428_data + (v410_data * v395_data));
              tensorforge::intel_esimd::simd<float, 32> v433_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v433_data + (v410_data * v400_data));
              tensorforge::intel_esimd::simd<float, 32> v438_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v438_data + (v410_data * v405_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v440_n0 = 0; v440_n0 < 1; ++v440_n0) {
                int32_t v442_a = v440_n0 * 32;
                #pragma unroll
                for (int32_t v441_n1 = 0; v441_n1 < 6; ++v441_n1) {
                  int32_t v444_a = v442_a + (v441_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v445_data(ir0.template select<32, 1>(v444_a));
                  r0.template select<32, 1>(v444_a) = v445_data;
                }
              }
              #pragma unroll
              for (int32_t v446_n1 = 0; v446_n1 < 6; ++v446_n1) {
                int32_t v448_a = 32 + (v446_n1 * 64);
                tensorforge::intel_esimd::simd<float, 8> v449_data(ir0.template select<8, 1>(v448_a));
                r0.template select<8, 1>(v448_a) = v449_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v450_i0 = 0; v450_i0 < 1; ++v450_i0) {
                int32_t v452_a = v450_i0 * 32;
                #pragma unroll
                for (int32_t v451_i1 = 0; v451_i1 < 6; ++v451_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v455_data(r0.template select<32, 1>((v452_a + (v451_i1 * 64))));
                  v455_data.copy_to(glb_m0 + ((v452_a + (v451_i1 * 40))));
                }
              }
              #pragma unroll
              for (int32_t v459_i1 = 0; v459_i1 < 6; ++v459_i1) {
                tensorforge::intel_esimd::simd<float, 8> v462_data(r0.template select<8, 1>((32 + (v459_i1 * 64))));
                v462_data.copy_to(glb_m0 + ((32_i32 + (v459_i1 * 40))));
              }
            }
          }
        }
      }
    });
  });
}


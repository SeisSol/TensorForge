// === base name ===
kernel_1f61f55400873eb7

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1f61f55400873eb7 = {{1, 32, 1}, 32, 40, 1, 32, 7424, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1f61f55400873eb7(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1f61f55400873eb7(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1f61f55400873eb7(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_1f61f55400873eb7(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1f61f55400873eb7(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1f61f55400873eb7(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1f61f55400873eb7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (48);
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 32> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 32> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 32> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 32> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 32> v16_ld;
            v16_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 32> v17_ld;
            v17_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 32> v18_ld;
            v18_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 32> v19_ld;
            v19_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 32> v20_ld;
            v20_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v20_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 32> v21_ld;
            v21_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v21_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v23_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v23_batchId0 < numElements0; v23_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v24_ahead1 = v23_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v26_batchId1 = (v24_ahead1 < numElements0) ? v24_ahead1 : v23_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v23_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v23_batchId0 * 240 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v23_batchId0 * 48 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v33_ld);
              tensorforge::intel_esimd::simd<float, 16> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v34_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 40), (0, 6)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 384> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v41_data(0.0f);
              v41_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 48> s0_w0 = tensorforge::slmLoad<float, 48>(s0 + 0);
              float v42_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v44_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v44_data + (v41_data * v42_data));
              float v47_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v49_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v49_data + (v41_data * v47_data));
              float v52_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v54_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v54_data + (v41_data * v52_data));
              float v57_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v59_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v59_data + (v41_data * v57_data));
              float v62_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v64_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v64_data + (v41_data * v62_data));
              float v67_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v69_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v69_data + (v41_data * v67_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 32> v72_data(glb_m1_run0.template select<32, 1>(0));
              float v73_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v75_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v75_data + (v72_data * v73_data));
              float v78_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v80_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v80_data + (v72_data * v78_data));
              float v83_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v85_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v85_data + (v72_data * v83_data));
              float v88_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v90_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v90_data + (v72_data * v88_data));
              float v93_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v95_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v95_data + (v72_data * v93_data));
              float v98_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v100_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v100_data + (v72_data * v98_data));
              tensorforge::intel_esimd::simd<float, 32> v103_data(glb_m1_run0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v106_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v106_data + (v103_data * v73_data));
              tensorforge::intel_esimd::simd<float, 32> v111_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v111_data + (v103_data * v78_data));
              tensorforge::intel_esimd::simd<float, 32> v116_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v116_data + (v103_data * v83_data));
              tensorforge::intel_esimd::simd<float, 32> v121_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v121_data + (v103_data * v88_data));
              tensorforge::intel_esimd::simd<float, 32> v126_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v126_data + (v103_data * v93_data));
              tensorforge::intel_esimd::simd<float, 32> v131_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v131_data + (v103_data * v98_data));
              tensorforge::intel_esimd::simd<float, 32> v134_data(0.0f);
              v134_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (112_i32));
              float v135_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v137_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v137_data + (v134_data * v135_data));
              float v140_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v142_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v142_data + (v134_data * v140_data));
              float v145_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v147_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v147_data + (v134_data * v145_data));
              float v150_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v152_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v152_data + (v134_data * v150_data));
              float v155_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v157_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v157_data + (v134_data * v155_data));
              float v160_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v162_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v162_data + (v134_data * v160_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 32> v165_data(glb_m1_run1.template select<32, 1>(0));
              float v166_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v168_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v168_data + (v165_data * v166_data));
              float v171_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v173_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v173_data + (v165_data * v171_data));
              float v176_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v178_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v178_data + (v165_data * v176_data));
              float v181_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v183_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v183_data + (v165_data * v181_data));
              float v186_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v188_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v188_data + (v165_data * v186_data));
              float v191_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v193_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v193_data + (v165_data * v191_data));
              tensorforge::intel_esimd::simd<float, 32> v196_data(glb_m1_run1.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v199_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v199_data + (v196_data * v166_data));
              tensorforge::intel_esimd::simd<float, 32> v204_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v204_data + (v196_data * v171_data));
              tensorforge::intel_esimd::simd<float, 32> v209_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v209_data + (v196_data * v176_data));
              tensorforge::intel_esimd::simd<float, 32> v214_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v214_data + (v196_data * v181_data));
              tensorforge::intel_esimd::simd<float, 32> v219_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v219_data + (v196_data * v186_data));
              tensorforge::intel_esimd::simd<float, 32> v224_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v224_data + (v196_data * v191_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2 = tensorforge::slmLoad<float, 64>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 32> v227_data(glb_m1_run2.template select<32, 1>(0));
              float v228_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v230_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v230_data + (v227_data * v228_data));
              float v233_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v235_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v235_data + (v227_data * v233_data));
              float v238_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v240_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v240_data + (v227_data * v238_data));
              float v243_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v245_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v245_data + (v227_data * v243_data));
              float v248_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v250_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v250_data + (v227_data * v248_data));
              float v253_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v255_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v255_data + (v227_data * v253_data));
              tensorforge::intel_esimd::simd<float, 32> v258_data(glb_m1_run2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v261_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v261_data + (v258_data * v228_data));
              tensorforge::intel_esimd::simd<float, 32> v266_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v266_data + (v258_data * v233_data));
              tensorforge::intel_esimd::simd<float, 32> v271_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v271_data + (v258_data * v238_data));
              tensorforge::intel_esimd::simd<float, 32> v276_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v276_data + (v258_data * v243_data));
              tensorforge::intel_esimd::simd<float, 32> v281_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v281_data + (v258_data * v248_data));
              tensorforge::intel_esimd::simd<float, 32> v286_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v286_data + (v258_data * v253_data));
              tensorforge::intel_esimd::simd<float, 32> v289_data(0.0f);
              v289_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (232_i32));
              float v290_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v292_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v292_data + (v289_data * v290_data));
              float v295_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v297_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v297_data + (v289_data * v295_data));
              float v300_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v302_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v302_data + (v289_data * v300_data));
              float v305_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v307_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v307_data + (v289_data * v305_data));
              float v310_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v312_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v312_data + (v289_data * v310_data));
              float v315_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v317_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v317_data + (v289_data * v315_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3 = tensorforge::slmLoad<float, 64>(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 32> v320_data(glb_m1_run3.template select<32, 1>(0));
              float v321_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v323_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v323_data + (v320_data * v321_data));
              float v326_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v328_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v328_data + (v320_data * v326_data));
              float v331_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v333_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v333_data + (v320_data * v331_data));
              float v336_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v338_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v338_data + (v320_data * v336_data));
              float v341_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v343_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v343_data + (v320_data * v341_data));
              float v346_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v348_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v348_data + (v320_data * v346_data));
              tensorforge::intel_esimd::simd<float, 32> v351_data(glb_m1_run3.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v354_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v354_data + (v351_data * v321_data));
              tensorforge::intel_esimd::simd<float, 32> v359_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v359_data + (v351_data * v326_data));
              tensorforge::intel_esimd::simd<float, 32> v364_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v364_data + (v351_data * v331_data));
              tensorforge::intel_esimd::simd<float, 32> v369_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v369_data + (v351_data * v336_data));
              tensorforge::intel_esimd::simd<float, 32> v374_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v374_data + (v351_data * v341_data));
              tensorforge::intel_esimd::simd<float, 32> v379_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v379_data + (v351_data * v346_data));
              tensorforge::intel_esimd::simd<float, 32> v382_data = tensorforge::slmLoad<float, 32>(glb_m1 + (280_i32));
              float v383_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v385_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v385_data + (v382_data * v383_data));
              float v388_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v390_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v390_data + (v382_data * v388_data));
              float v393_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v395_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v395_data + (v382_data * v393_data));
              float v398_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v400_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v400_data + (v382_data * v398_data));
              float v403_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v405_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v405_data + (v382_data * v403_data));
              float v408_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v410_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v410_data + (v382_data * v408_data));
              tensorforge::intel_esimd::simd<float, 32> v413_data(0.0f);
              v413_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (312_i32));
              tensorforge::intel_esimd::simd<float, 32> v416_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v416_data + (v413_data * v383_data));
              tensorforge::intel_esimd::simd<float, 32> v421_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v421_data + (v413_data * v388_data));
              tensorforge::intel_esimd::simd<float, 32> v426_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v426_data + (v413_data * v393_data));
              tensorforge::intel_esimd::simd<float, 32> v431_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v431_data + (v413_data * v398_data));
              tensorforge::intel_esimd::simd<float, 32> v436_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v436_data + (v413_data * v403_data));
              tensorforge::intel_esimd::simd<float, 32> v441_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v441_data + (v413_data * v408_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v443_n0 = 0; v443_n0 < 1; ++v443_n0) {
                int32_t v445_a = v443_n0 * 32;
                #pragma unroll
                for (int32_t v444_n1 = 0; v444_n1 < 6; ++v444_n1) {
                  int32_t v447_a = v445_a + (v444_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v448_data(ir0.template select<32, 1>(v447_a));
                  r0.template select<32, 1>(v447_a) = v448_data;
                }
              }
              #pragma unroll
              for (int32_t v449_n1 = 0; v449_n1 < 6; ++v449_n1) {
                int32_t v451_a = 32 + (v449_n1 * 64);
                tensorforge::intel_esimd::simd<float, 8> v452_data(ir0.template select<8, 1>(v451_a));
                r0.template select<8, 1>(v451_a) = v452_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v453_i0 = 0; v453_i0 < 1; ++v453_i0) {
                int32_t v455_a = v453_i0 * 32;
                #pragma unroll
                for (int32_t v454_i1 = 0; v454_i1 < 6; ++v454_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v458_data(r0.template select<32, 1>((v455_a + (v454_i1 * 64))));
                  v458_data.copy_to(glb_m0 + ((v455_a + (v454_i1 * 40))));
                }
              }
              #pragma unroll
              for (int32_t v462_i1 = 0; v462_i1 < 6; ++v462_i1) {
                tensorforge::intel_esimd::simd<float, 8> v465_data(r0.template select<8, 1>((32 + (v462_i1 * 64))));
                v465_data.copy_to(glb_m0 + ((32_i32 + (v462_i1 * 40))));
              }
            }
          }
        }
      }
    });
  });
}


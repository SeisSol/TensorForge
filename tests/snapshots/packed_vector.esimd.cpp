// === base name ===
kernel_864883913f10f23f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_864883913f10f23f = {{1, 8, 1}, 32, 64, 1, 8, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_864883913f10f23f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_864883913f10f23f(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_864883913f10f23f(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 384 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_864883913f10f23f(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_864883913f10f23f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_864883913f10f23f(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_864883913f10f23f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<384 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: lead_vectorize=1,pack_operands=runs
        // launch: 32 lanes (64 active) x 8 per block = block 1x8x1, 1536 B shared, occupancy grid
        // operands:
        //   m0 64×6(64×6) {0..64}×{0..6} strided
        //   m1 64×8(64×8) {0..64}×{0..8} none
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[64,6]],"name":"m0","ordered":false,"parts":1,"shape":[64,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[64,8]],"name":"m1","ordered":false,"parts":1,"shape":[64,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[64,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (48 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v383_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v383_flagSecondAhead < numElements0) ? v383_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 32> v391_ld;
          v391_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 0));
          tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v391_ld);
          tensorforge::intel_esimd::simd<float, 16> v392_ld;
          v392_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 32));
          tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v392_ld);
          uint32_t v387_flagWord(flagWordFirst);
          uint32_t v388_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v397_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v397_flagAheadAhead < numElements0) ? v397_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v387_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 384 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 64), (0, 6)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 384> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v26_data;
              v26_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 48> s0_w0 = tensorforge::slmLoad<float, 48>(s0 + 0);
              float v27_data = s0_w0[0];
              ir0.template select<32, 1>(0) = (v26_data * v27_data);
              float v32_data = s0_w0[8];
              ir0.template select<32, 1>(64) = (v26_data * v32_data);
              float v37_data = s0_w0[16];
              ir0.template select<32, 1>(128) = (v26_data * v37_data);
              float v42_data = s0_w0[24];
              ir0.template select<32, 1>(192) = (v26_data * v42_data);
              float v47_data = s0_w0[32];
              ir0.template select<32, 1>(256) = (v26_data * v47_data);
              float v52_data = s0_w0[40];
              ir0.template select<32, 1>(320) = (v26_data * v52_data);
              tensorforge::intel_esimd::simd<float, 32> v57_data;
              v57_data.copy_from(glb_m1 + (64_i32));
              float v58_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v60_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v60_data + (v57_data * v58_data));
              float v63_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v65_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v65_data + (v57_data * v63_data));
              float v68_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v70_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v70_data + (v57_data * v68_data));
              float v73_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v75_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v75_data + (v57_data * v73_data));
              float v78_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v80_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v80_data + (v57_data * v78_data));
              float v83_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v85_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v85_data + (v57_data * v83_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 32> v88_data(glb_m1_run0.template select<32, 1>(0));
              float v89_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v91_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v91_data + (v88_data * v89_data));
              float v94_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v96_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v96_data + (v88_data * v94_data));
              float v99_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v101_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v101_data + (v88_data * v99_data));
              float v104_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v106_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v106_data + (v88_data * v104_data));
              float v109_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v111_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v111_data + (v88_data * v109_data));
              float v114_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v116_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v116_data + (v88_data * v114_data));
              tensorforge::intel_esimd::simd<float, 32> v121_data(glb_m1_run0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v121_data * v89_data);
              ir0.template select<32, 1>(96) = (v121_data * v94_data);
              ir0.template select<32, 1>(160) = (v121_data * v99_data);
              ir0.template select<32, 1>(224) = (v121_data * v104_data);
              ir0.template select<32, 1>(288) = (v121_data * v109_data);
              ir0.template select<32, 1>(352) = (v121_data * v114_data);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 32> v152_data(glb_m1_run1.template select<32, 1>(0));
              float v153_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v155_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v155_data + (v152_data * v153_data));
              float v158_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v160_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v160_data + (v152_data * v158_data));
              float v163_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v165_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v165_data + (v152_data * v163_data));
              float v168_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v170_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v170_data + (v152_data * v168_data));
              float v173_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v175_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v175_data + (v152_data * v173_data));
              float v178_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v180_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v180_data + (v152_data * v178_data));
              tensorforge::intel_esimd::simd<float, 32> v183_data(glb_m1_run1.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v186_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v186_data + (v183_data * v153_data));
              tensorforge::intel_esimd::simd<float, 32> v191_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v191_data + (v183_data * v158_data));
              tensorforge::intel_esimd::simd<float, 32> v196_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v196_data + (v183_data * v163_data));
              tensorforge::intel_esimd::simd<float, 32> v201_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v201_data + (v183_data * v168_data));
              tensorforge::intel_esimd::simd<float, 32> v206_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v206_data + (v183_data * v173_data));
              tensorforge::intel_esimd::simd<float, 32> v211_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v211_data + (v183_data * v178_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (288_i32));
              tensorforge::intel_esimd::simd<float, 32> v214_data(glb_m1_run2.template select<32, 1>(0));
              float v215_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v217_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v217_data + (v214_data * v215_data));
              float v220_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v222_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v222_data + (v214_data * v220_data));
              float v225_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v227_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v227_data + (v214_data * v225_data));
              float v230_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v232_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v232_data + (v214_data * v230_data));
              float v235_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v237_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v237_data + (v214_data * v235_data));
              float v240_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v242_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v242_data + (v214_data * v240_data));
              tensorforge::intel_esimd::simd<float, 32> v245_data(glb_m1_run2.template select<32, 1>(32));
              float v246_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v248_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v248_data + (v245_data * v246_data));
              float v251_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v253_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v253_data + (v245_data * v251_data));
              float v256_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v258_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v258_data + (v245_data * v256_data));
              float v261_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v263_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v263_data + (v245_data * v261_data));
              float v266_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v268_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v268_data + (v245_data * v266_data));
              float v271_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v273_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v273_data + (v245_data * v271_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (416_i32));
              tensorforge::intel_esimd::simd<float, 32> v276_data(glb_m1_run3.template select<32, 1>(0));
              float v277_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v279_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v279_data + (v276_data * v277_data));
              float v282_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v284_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v284_data + (v276_data * v282_data));
              float v287_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v289_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v289_data + (v276_data * v287_data));
              float v292_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v294_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v294_data + (v276_data * v292_data));
              float v297_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v299_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v299_data + (v276_data * v297_data));
              float v302_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v304_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v304_data + (v276_data * v302_data));
              tensorforge::intel_esimd::simd<float, 32> v307_data(glb_m1_run3.template select<32, 1>(32));
              float v308_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v310_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v310_data + (v307_data * v308_data));
              float v313_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v315_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v315_data + (v307_data * v313_data));
              float v318_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v320_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v320_data + (v307_data * v318_data));
              float v323_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v325_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v325_data + (v307_data * v323_data));
              float v328_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v330_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v330_data + (v307_data * v328_data));
              float v333_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v335_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v335_data + (v307_data * v333_data));
              tensorforge::intel_esimd::simd<float, 32> v338_data;
              v338_data.copy_from(glb_m1 + (480_i32));
              tensorforge::intel_esimd::simd<float, 32> v341_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v341_data + (v338_data * v308_data));
              tensorforge::intel_esimd::simd<float, 32> v346_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v346_data + (v338_data * v313_data));
              tensorforge::intel_esimd::simd<float, 32> v351_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v351_data + (v338_data * v318_data));
              tensorforge::intel_esimd::simd<float, 32> v356_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v356_data + (v338_data * v323_data));
              tensorforge::intel_esimd::simd<float, 32> v361_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v361_data + (v338_data * v328_data));
              tensorforge::intel_esimd::simd<float, 32> v366_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v366_data + (v338_data * v333_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v368_n0 = 0; v368_n0 < 2; ++v368_n0) {
                int32_t v370_a = v368_n0 * 32;
                #pragma unroll
                for (int32_t v369_n1 = 0; v369_n1 < 6; ++v369_n1) {
                  int32_t v372_a = v370_a + (v369_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v373_data(ir0.template select<32, 1>(v372_a));
                  r0.template select<32, 1>(v372_a) = v373_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v374_i0 = 0; v374_i0 < 2; ++v374_i0) {
                int32_t v376_a = v374_i0 * 32;
                #pragma unroll
                for (int32_t v375_i1 = 0; v375_i1 < 6; ++v375_i1) {
                  int32_t v378_a = v376_a + (v375_i1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v379_data(r0.template select<32, 1>(v378_a));
                  v379_data.copy_to(glb_m0 + (v378_a));
                }
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 32> v393_ld;
            v393_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 0));
            tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v393_ld);
            tensorforge::intel_esimd::simd<float, 16> v394_ld;
            v394_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 32));
            tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v394_ld);
            v387_flagWord = v388_flagWordNext;
            v388_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


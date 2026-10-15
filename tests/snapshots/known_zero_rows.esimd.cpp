// === base name ===
kernel_5fd10aea623fb9b6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5fd10aea623fb9b6 = {{1, 8, 1}, 32, 40, 1, 8, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5fd10aea623fb9b6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5fd10aea623fb9b6(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5fd10aea623fb9b6(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_5fd10aea623fb9b6(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5fd10aea623fb9b6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5fd10aea623fb9b6(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5fd10aea623fb9b6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<384 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (40 active) x 8 per block = block 1x8x1, 1536 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (48 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v458_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v458_flagSecondAhead < numElements0) ? v458_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 32> v466_ld;
          v466_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 0));
          tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v466_ld);
          tensorforge::intel_esimd::simd<float, 16> v467_ld;
          v467_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 32));
          tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v467_ld);
          uint32_t v462_flagWord(flagWordFirst);
          uint32_t v463_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v472_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v472_flagAheadAhead < numElements0) ? v472_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v462_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 240 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 40), (0, 6)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 384> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v27_data(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v27_data_part;
              v27_data_part.copy_from(glb_m1 + (32_i32));
              v27_data.template select<8, 1>(0) = v27_data_part;
              tensorforge::intel_esimd::simd<float, 48> s0_w0 = tensorforge::slmLoad<float, 48>(s0 + 0);
              float v28_data = s0_w0[0];
              ir0.template select<32, 1>(32) = (v27_data * v28_data);
              float v33_data = s0_w0[8];
              ir0.template select<32, 1>(96) = (v27_data * v33_data);
              float v38_data = s0_w0[16];
              ir0.template select<32, 1>(160) = (v27_data * v38_data);
              float v43_data = s0_w0[24];
              ir0.template select<32, 1>(224) = (v27_data * v43_data);
              float v48_data = s0_w0[32];
              ir0.template select<32, 1>(288) = (v27_data * v48_data);
              float v53_data = s0_w0[40];
              ir0.template select<32, 1>(352) = (v27_data * v53_data);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 32> v58_data(glb_m1_run0.template select<32, 1>(0));
              float v59_data = s0_w0[1];
              ir0.template select<32, 1>(0) = (v58_data * v59_data);
              float v64_data = s0_w0[9];
              ir0.template select<32, 1>(64) = (v58_data * v64_data);
              float v69_data = s0_w0[17];
              ir0.template select<32, 1>(128) = (v58_data * v69_data);
              float v74_data = s0_w0[25];
              ir0.template select<32, 1>(192) = (v58_data * v74_data);
              float v79_data = s0_w0[33];
              ir0.template select<32, 1>(256) = (v58_data * v79_data);
              float v84_data = s0_w0[41];
              ir0.template select<32, 1>(320) = (v58_data * v84_data);
              tensorforge::intel_esimd::simd<float, 32> v89_data(glb_m1_run0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v92_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v92_data + (v89_data * v59_data));
              tensorforge::intel_esimd::simd<float, 32> v97_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v97_data + (v89_data * v64_data));
              tensorforge::intel_esimd::simd<float, 32> v102_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v102_data + (v89_data * v69_data));
              tensorforge::intel_esimd::simd<float, 32> v107_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v107_data + (v89_data * v74_data));
              tensorforge::intel_esimd::simd<float, 32> v112_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v112_data + (v89_data * v79_data));
              tensorforge::intel_esimd::simd<float, 32> v117_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v117_data + (v89_data * v84_data));
              tensorforge::intel_esimd::simd<float, 32> v120_data(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v120_data_part;
              v120_data_part.copy_from(glb_m1 + (112_i32));
              v120_data.template select<8, 1>(0) = v120_data_part;
              float v121_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v123_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v123_data + (v120_data * v121_data));
              float v126_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v128_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v128_data + (v120_data * v126_data));
              float v131_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v133_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v133_data + (v120_data * v131_data));
              float v136_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v138_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v138_data + (v120_data * v136_data));
              float v141_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v143_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v143_data + (v120_data * v141_data));
              float v146_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v148_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v148_data + (v120_data * v146_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 32> v151_data(glb_m1_run1.template select<32, 1>(0));
              float v152_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v154_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v154_data + (v151_data * v152_data));
              float v157_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v159_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v159_data + (v151_data * v157_data));
              float v162_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v164_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v164_data + (v151_data * v162_data));
              float v167_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v169_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v169_data + (v151_data * v167_data));
              float v172_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v174_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v174_data + (v151_data * v172_data));
              float v177_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v179_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v179_data + (v151_data * v177_data));
              tensorforge::intel_esimd::simd<float, 32> v182_data(glb_m1_run1.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v185_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v185_data + (v182_data * v152_data));
              tensorforge::intel_esimd::simd<float, 32> v190_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v190_data + (v182_data * v157_data));
              tensorforge::intel_esimd::simd<float, 32> v195_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v195_data + (v182_data * v162_data));
              tensorforge::intel_esimd::simd<float, 32> v200_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v200_data + (v182_data * v167_data));
              tensorforge::intel_esimd::simd<float, 32> v205_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v205_data + (v182_data * v172_data));
              tensorforge::intel_esimd::simd<float, 32> v210_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v210_data + (v182_data * v177_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 32> v213_data(glb_m1_run2.template select<32, 1>(0));
              float v214_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v216_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v216_data + (v213_data * v214_data));
              float v219_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v221_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v221_data + (v213_data * v219_data));
              float v224_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v226_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v226_data + (v213_data * v224_data));
              float v229_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v231_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v231_data + (v213_data * v229_data));
              float v234_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v236_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v236_data + (v213_data * v234_data));
              float v239_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v241_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v241_data + (v213_data * v239_data));
              tensorforge::intel_esimd::simd<float, 32> v244_data(glb_m1_run2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v247_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v247_data + (v244_data * v214_data));
              tensorforge::intel_esimd::simd<float, 32> v252_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v252_data + (v244_data * v219_data));
              tensorforge::intel_esimd::simd<float, 32> v257_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v257_data + (v244_data * v224_data));
              tensorforge::intel_esimd::simd<float, 32> v262_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v262_data + (v244_data * v229_data));
              tensorforge::intel_esimd::simd<float, 32> v267_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v267_data + (v244_data * v234_data));
              tensorforge::intel_esimd::simd<float, 32> v272_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v272_data + (v244_data * v239_data));
              tensorforge::intel_esimd::simd<float, 32> v275_data(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v275_data_part;
              v275_data_part.copy_from(glb_m1 + (232_i32));
              v275_data.template select<8, 1>(0) = v275_data_part;
              float v276_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v278_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v278_data + (v275_data * v276_data));
              float v281_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v283_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v283_data + (v275_data * v281_data));
              float v286_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v288_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v288_data + (v275_data * v286_data));
              float v291_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v293_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v293_data + (v275_data * v291_data));
              float v296_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v298_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v298_data + (v275_data * v296_data));
              float v301_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v303_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v303_data + (v275_data * v301_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 32> v306_data(glb_m1_run3.template select<32, 1>(0));
              float v307_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v309_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v309_data + (v306_data * v307_data));
              float v312_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v314_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v314_data + (v306_data * v312_data));
              float v317_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v319_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v319_data + (v306_data * v317_data));
              float v322_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v324_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v324_data + (v306_data * v322_data));
              float v327_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v329_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v329_data + (v306_data * v327_data));
              float v332_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v334_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v334_data + (v306_data * v332_data));
              tensorforge::intel_esimd::simd<float, 32> v337_data(glb_m1_run3.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v340_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v340_data + (v337_data * v307_data));
              tensorforge::intel_esimd::simd<float, 32> v345_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v345_data + (v337_data * v312_data));
              tensorforge::intel_esimd::simd<float, 32> v350_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v350_data + (v337_data * v317_data));
              tensorforge::intel_esimd::simd<float, 32> v355_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v355_data + (v337_data * v322_data));
              tensorforge::intel_esimd::simd<float, 32> v360_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v360_data + (v337_data * v327_data));
              tensorforge::intel_esimd::simd<float, 32> v365_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v365_data + (v337_data * v332_data));
              tensorforge::intel_esimd::simd<float, 32> v368_data;
              v368_data.copy_from(glb_m1 + (280_i32));
              float v369_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v371_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v371_data + (v368_data * v369_data));
              float v374_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v376_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v376_data + (v368_data * v374_data));
              float v379_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v381_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v381_data + (v368_data * v379_data));
              float v384_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v386_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v386_data + (v368_data * v384_data));
              float v389_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v391_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v391_data + (v368_data * v389_data));
              float v394_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v396_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v396_data + (v368_data * v394_data));
              tensorforge::intel_esimd::simd<float, 32> v399_data(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v399_data_part;
              v399_data_part.copy_from(glb_m1 + (312_i32));
              v399_data.template select<8, 1>(0) = v399_data_part;
              tensorforge::intel_esimd::simd<float, 32> v402_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v402_data + (v399_data * v369_data));
              tensorforge::intel_esimd::simd<float, 32> v407_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v407_data + (v399_data * v374_data));
              tensorforge::intel_esimd::simd<float, 32> v412_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v412_data + (v399_data * v379_data));
              tensorforge::intel_esimd::simd<float, 32> v417_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v417_data + (v399_data * v384_data));
              tensorforge::intel_esimd::simd<float, 32> v422_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v422_data + (v399_data * v389_data));
              tensorforge::intel_esimd::simd<float, 32> v427_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v427_data + (v399_data * v394_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v429_n0 = 0; v429_n0 < 1; ++v429_n0) {
                int32_t v431_a = v429_n0 * 32;
                #pragma unroll
                for (int32_t v430_n1 = 0; v430_n1 < 6; ++v430_n1) {
                  int32_t v433_a = v431_a + (v430_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v434_data(ir0.template select<32, 1>(v433_a));
                  r0.template select<32, 1>(v433_a) = v434_data;
                }
              }
              #pragma unroll
              for (int32_t v435_n1 = 0; v435_n1 < 6; ++v435_n1) {
                int32_t v437_a = 32 + (v435_n1 * 64);
                tensorforge::intel_esimd::simd<float, 8> v438_data(ir0.template select<8, 1>(v437_a));
                r0.template select<8, 1>(v437_a) = v438_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v439_i0 = 0; v439_i0 < 1; ++v439_i0) {
                int32_t v441_a = v439_i0 * 32;
                #pragma unroll
                for (int32_t v440_i1 = 0; v440_i1 < 6; ++v440_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v444_data(r0.template select<32, 1>((v441_a + (v440_i1 * 64))));
                  v444_data.copy_to(glb_m0 + ((v441_a + (v440_i1 * 40))));
                }
              }
              #pragma unroll
              for (int32_t v448_i1 = 0; v448_i1 < 6; ++v448_i1) {
                tensorforge::intel_esimd::simd<float, 8> v451_data(r0.template select<8, 1>((32 + (v448_i1 * 64))));
                v451_data.copy_to(glb_m0 + ((32_i32 + (v448_i1 * 40))));
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 32> v468_ld;
            v468_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 0));
            tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v468_ld);
            tensorforge::intel_esimd::simd<float, 16> v469_ld;
            v469_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 32));
            tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v469_ld);
            v462_flagWord = v463_flagWordNext;
            v463_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


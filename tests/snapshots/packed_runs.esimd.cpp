// === base name ===
kernel_561a3d9c1d39bb75

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_561a3d9c1d39bb75 = {{1, 8, 1}, 32, 40, 1, 8, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_561a3d9c1d39bb75(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_561a3d9c1d39bb75(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_561a3d9c1d39bb75(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_561a3d9c1d39bb75(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_561a3d9c1d39bb75(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_561a3d9c1d39bb75(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_561a3d9c1d39bb75(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<384 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: pack_operands=runs
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
          size_t v365_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v365_flagSecondAhead < numElements0) ? v365_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 32> v373_ld;
          v373_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 0));
          tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v373_ld);
          tensorforge::intel_esimd::simd<float, 16> v374_ld;
          v374_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 32));
          tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v374_ld);
          uint32_t v369_flagWord(flagWordFirst);
          uint32_t v370_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v379_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v379_flagAheadAhead < numElements0) ? v379_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v369_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 240 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 40), (0, 6)] [(0, 8)]
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
              v57_data.copy_from(glb_m1 + (40_i32));
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
              glb_m1_run0.copy_from(glb_m1 + (80_i32));
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
              tensorforge::intel_esimd::simd<float, 32> v120_data(glb_m1_run0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v120_data * v89_data);
              ir0.template select<32, 1>(96) = (v120_data * v94_data);
              ir0.template select<32, 1>(160) = (v120_data * v99_data);
              ir0.template select<32, 1>(224) = (v120_data * v104_data);
              ir0.template select<32, 1>(288) = (v120_data * v109_data);
              ir0.template select<32, 1>(352) = (v120_data * v114_data);
              tensorforge::intel_esimd::simd<float, 32> v151_data(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v151_data_part;
              v151_data_part.copy_from(glb_m1 + (152_i32));
              v151_data.template select<8, 1>(0) = v151_data_part;
              float v152_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v154_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v154_data + (v151_data * v152_data));
              float v157_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v159_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v159_data + (v151_data * v157_data));
              float v162_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v164_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v164_data + (v151_data * v162_data));
              float v167_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v169_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v169_data + (v151_data * v167_data));
              float v172_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v174_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v174_data + (v151_data * v172_data));
              float v177_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v179_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v179_data + (v151_data * v177_data));
              tensorforge::intel_esimd::simd<float, 32> v182_data;
              v182_data.copy_from(glb_m1 + (160_i32));
              float v183_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v185_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v185_data + (v182_data * v183_data));
              float v188_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v190_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v190_data + (v182_data * v188_data));
              float v193_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v195_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v195_data + (v182_data * v193_data));
              float v198_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v200_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v200_data + (v182_data * v198_data));
              float v203_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v205_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v205_data + (v182_data * v203_data));
              float v208_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v210_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v210_data + (v182_data * v208_data));
              tensorforge::intel_esimd::simd<float, 32> v213_data;
              v213_data.copy_from(glb_m1 + (200_i32));
              float v214_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v216_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v216_data + (v213_data * v214_data));
              float v219_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v221_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v221_data + (v213_data * v219_data));
              float v224_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v226_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v226_data + (v213_data * v224_data));
              float v229_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v231_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v231_data + (v213_data * v229_data));
              float v234_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v236_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v236_data + (v213_data * v234_data));
              float v239_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v241_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v241_data + (v213_data * v239_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 32> v244_data(glb_m1_run1.template select<32, 1>(0));
              float v245_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v247_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v247_data + (v244_data * v245_data));
              float v250_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v252_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v252_data + (v244_data * v250_data));
              float v255_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v257_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v257_data + (v244_data * v255_data));
              float v260_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v262_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v262_data + (v244_data * v260_data));
              float v265_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v267_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v267_data + (v244_data * v265_data));
              float v270_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v272_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v272_data + (v244_data * v270_data));
              tensorforge::intel_esimd::simd<float, 32> v275_data(glb_m1_run1.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v278_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v278_data + (v275_data * v245_data));
              tensorforge::intel_esimd::simd<float, 32> v283_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v283_data + (v275_data * v250_data));
              tensorforge::intel_esimd::simd<float, 32> v288_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v288_data + (v275_data * v255_data));
              tensorforge::intel_esimd::simd<float, 32> v293_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v293_data + (v275_data * v260_data));
              tensorforge::intel_esimd::simd<float, 32> v298_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v298_data + (v275_data * v265_data));
              tensorforge::intel_esimd::simd<float, 32> v303_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v303_data + (v275_data * v270_data));
              tensorforge::intel_esimd::simd<float, 32> v306_data;
              v306_data.copy_from(glb_m1 + (280_i32));
              float v307_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v309_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v309_data + (v306_data * v307_data));
              float v312_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v314_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v314_data + (v306_data * v312_data));
              float v317_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v319_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v319_data + (v306_data * v317_data));
              float v322_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v324_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v324_data + (v306_data * v322_data));
              float v327_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v329_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v329_data + (v306_data * v327_data));
              float v332_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v334_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v334_data + (v306_data * v332_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v336_n0 = 0; v336_n0 < 1; ++v336_n0) {
                int32_t v338_a = v336_n0 * 32;
                #pragma unroll
                for (int32_t v337_n1 = 0; v337_n1 < 6; ++v337_n1) {
                  int32_t v340_a = v338_a + (v337_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v341_data(ir0.template select<32, 1>(v340_a));
                  r0.template select<32, 1>(v340_a) = v341_data;
                }
              }
              #pragma unroll
              for (int32_t v342_n1 = 0; v342_n1 < 6; ++v342_n1) {
                int32_t v344_a = 32 + (v342_n1 * 64);
                tensorforge::intel_esimd::simd<float, 8> v345_data(ir0.template select<8, 1>(v344_a));
                r0.template select<8, 1>(v344_a) = v345_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v346_i0 = 0; v346_i0 < 1; ++v346_i0) {
                int32_t v348_a = v346_i0 * 32;
                #pragma unroll
                for (int32_t v347_i1 = 0; v347_i1 < 6; ++v347_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v351_data(r0.template select<32, 1>((v348_a + (v347_i1 * 64))));
                  v351_data.copy_to(glb_m0 + ((v348_a + (v347_i1 * 40))));
                }
              }
              #pragma unroll
              for (int32_t v355_i1 = 0; v355_i1 < 6; ++v355_i1) {
                tensorforge::intel_esimd::simd<float, 8> v358_data(r0.template select<8, 1>((32 + (v355_i1 * 64))));
                v358_data.copy_to(glb_m0 + ((32_i32 + (v355_i1 * 40))));
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 32> v375_ld;
            v375_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 0));
            tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v375_ld);
            tensorforge::intel_esimd::simd<float, 16> v376_ld;
            v376_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 32));
            tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v376_ld);
            v369_flagWord = v370_flagWordNext;
            v370_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


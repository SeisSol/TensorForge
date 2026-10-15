// === base name ===
kernel_1f28e3233cbda68c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1f28e3233cbda68c = {{1, 8, 1}, 32, 40, 1, 8, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1f28e3233cbda68c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1f28e3233cbda68c(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1f28e3233cbda68c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_1f28e3233cbda68c(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1f28e3233cbda68c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1f28e3233cbda68c(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1f28e3233cbda68c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<384 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: pack_operands=rows
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
          size_t v302_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v302_flagSecondAhead < numElements0) ? v302_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 32> v310_ld;
          v310_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 0));
          tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v310_ld);
          tensorforge::intel_esimd::simd<float, 16> v311_ld;
          v311_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 32));
          tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v311_ld);
          uint32_t v306_flagWord(flagWordFirst);
          uint32_t v307_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v316_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v316_flagAheadAhead < numElements0) ? v316_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v306_flagWord);
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
              tensorforge::intel_esimd::simd<float, 32> v88_data;
              v88_data.copy_from(glb_m1 + (80_i32));
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
              tensorforge::intel_esimd::simd<float, 32> v119_data;
              v119_data.copy_from(glb_m1 + (120_i32));
              float v120_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v122_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v122_data + (v119_data * v120_data));
              float v125_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v127_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v127_data + (v119_data * v125_data));
              float v130_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v132_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v132_data + (v119_data * v130_data));
              float v135_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v137_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v137_data + (v119_data * v135_data));
              float v140_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v142_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v142_data + (v119_data * v140_data));
              float v145_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v147_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v147_data + (v119_data * v145_data));
              tensorforge::intel_esimd::simd<float, 32> v150_data;
              v150_data.copy_from(glb_m1 + (160_i32));
              float v151_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v153_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v153_data + (v150_data * v151_data));
              float v156_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v158_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v158_data + (v150_data * v156_data));
              float v161_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v163_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v163_data + (v150_data * v161_data));
              float v166_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v168_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v168_data + (v150_data * v166_data));
              float v171_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v173_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v173_data + (v150_data * v171_data));
              float v176_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v178_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v178_data + (v150_data * v176_data));
              tensorforge::intel_esimd::simd<float, 32> v181_data;
              v181_data.copy_from(glb_m1 + (200_i32));
              float v182_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v184_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v184_data + (v181_data * v182_data));
              float v187_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v189_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v189_data + (v181_data * v187_data));
              float v192_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v194_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v194_data + (v181_data * v192_data));
              float v197_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v199_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v199_data + (v181_data * v197_data));
              float v202_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v204_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v204_data + (v181_data * v202_data));
              float v207_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v209_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v209_data + (v181_data * v207_data));
              tensorforge::intel_esimd::simd<float, 32> v212_data;
              v212_data.copy_from(glb_m1 + (240_i32));
              float v213_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v215_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v215_data + (v212_data * v213_data));
              float v218_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v220_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v220_data + (v212_data * v218_data));
              float v223_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v225_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v225_data + (v212_data * v223_data));
              float v228_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v230_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v230_data + (v212_data * v228_data));
              float v233_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v235_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v235_data + (v212_data * v233_data));
              float v238_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v240_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v240_data + (v212_data * v238_data));
              tensorforge::intel_esimd::simd<float, 32> v243_data;
              v243_data.copy_from(glb_m1 + (280_i32));
              float v244_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v246_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v246_data + (v243_data * v244_data));
              float v249_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v251_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v251_data + (v243_data * v249_data));
              float v254_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v256_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v256_data + (v243_data * v254_data));
              float v259_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v261_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v261_data + (v243_data * v259_data));
              float v264_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v266_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v266_data + (v243_data * v264_data));
              float v269_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v271_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v271_data + (v243_data * v269_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v273_n0 = 0; v273_n0 < 1; ++v273_n0) {
                int32_t v275_a = v273_n0 * 32;
                #pragma unroll
                for (int32_t v274_n1 = 0; v274_n1 < 6; ++v274_n1) {
                  int32_t v277_a = v275_a + (v274_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v278_data(ir0.template select<32, 1>(v277_a));
                  r0.template select<32, 1>(v277_a) = v278_data;
                }
              }
              #pragma unroll
              for (int32_t v279_n1 = 0; v279_n1 < 6; ++v279_n1) {
                int32_t v281_a = 32 + (v279_n1 * 64);
                tensorforge::intel_esimd::simd<float, 8> v282_data(ir0.template select<8, 1>(v281_a));
                r0.template select<8, 1>(v281_a) = v282_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v283_i0 = 0; v283_i0 < 1; ++v283_i0) {
                int32_t v285_a = v283_i0 * 32;
                #pragma unroll
                for (int32_t v284_i1 = 0; v284_i1 < 6; ++v284_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v288_data(r0.template select<32, 1>((v285_a + (v284_i1 * 64))));
                  v288_data.copy_to(glb_m0 + ((v285_a + (v284_i1 * 40))));
                }
              }
              #pragma unroll
              for (int32_t v292_i1 = 0; v292_i1 < 6; ++v292_i1) {
                tensorforge::intel_esimd::simd<float, 8> v295_data(r0.template select<8, 1>((32 + (v292_i1 * 64))));
                v295_data.copy_to(glb_m0 + ((32_i32 + (v292_i1 * 40))));
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 32> v312_ld;
            v312_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 0));
            tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v312_ld);
            tensorforge::intel_esimd::simd<float, 16> v313_ld;
            v313_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 32));
            tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v313_ld);
            v306_flagWord = v307_flagWordNext;
            v307_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


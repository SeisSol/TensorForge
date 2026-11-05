// === base name ===
kernel_d4d884cf2d46b8e2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d4d884cf2d46b8e2 = {{1, 32, 1}, 8, 8, 1, 32, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d4d884cf2d46b8e2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d4d884cf2d46b8e2(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d4d884cf2d46b8e2(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1792 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_d4d884cf2d46b8e2(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d4d884cf2d46b8e2(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d4d884cf2d46b8e2(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d4d884cf2d46b8e2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1792 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 32 per block = block 1x32x1, 7168 B shared, occupancy grid
        // operands:
        //   m0 8×9(8×9) {0..8}×{0..9} strided
        //   m1 8×16(8×4) {0..8}×{9..13} none
        //   m2 16×9(4×9) {9..13}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":1792}],"shared_bytes":7168,"shared_elements":1792,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[8,9]],"name":"m0","ordered":false,"parts":1,"shape":[8,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,9],[8,13]],"name":"m1","ordered":false,"parts":1,"shape":[8,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[9,0],[13,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,9],[8,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,16]},{"addressing":"strided","bbox":[[9,0],[13,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (56 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v224_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v224_flagSecondAhead < numElements0) ? v224_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 36 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 32> v232_ld;
          v232_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 0));
          tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v232_ld);
          tensorforge::intel_esimd::simd<float, 4> v233_ld;
          v233_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 32));
          tensorforge::slmStore<float, 4>(s0 + (0 + 0 + 1 * 0 + 32), v233_ld);
          uint32_t v228_flagWord(flagWordFirst);
          uint32_t v229_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 36 + 0 + m2_extraOffset];
            size_t v238_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v238_flagAheadAhead < numElements0) ? v238_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v228_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 72 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 72> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 8), (0, 9)] [(9, 13)]
              tensorforge::intel_esimd::simd<float, 72> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 32> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 8> v26_data(glb_m1_run0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 48> s0_w0 = tensorforge::slmLoad<float, 48>(s0 + 0);
              float v27_data = s0_w0[0];
              float v32_data = s0_w0[4];
              float v37_data = s0_w0[8];
              float v42_data = s0_w0[12];
              float v47_data = s0_w0[16];
              float v52_data = s0_w0[20];
              float v57_data = s0_w0[24];
              float v62_data = s0_w0[28];
              float v67_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v72_data(glb_m1_run0.template select<8, 1>(8));
              float v73_data = s0_w0[1];
              float v78_data = s0_w0[5];
              float v83_data = s0_w0[9];
              float v88_data = s0_w0[13];
              float v93_data = s0_w0[17];
              float v98_data = s0_w0[21];
              float v103_data = s0_w0[25];
              float v108_data = s0_w0[29];
              float v113_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v118_data(glb_m1_run0.template select<8, 1>(16));
              float v119_data = s0_w0[2];
              float v124_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v127_p = ((v26_data * v32_data) + (v72_data * v78_data)) + (v118_data * v124_data);
              float v129_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v132_p = ((v26_data * v37_data) + (v72_data * v83_data)) + (v118_data * v129_data);
              float v134_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v137_p = ((v26_data * v42_data) + (v72_data * v88_data)) + (v118_data * v134_data);
              float v139_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v142_p = ((v26_data * v47_data) + (v72_data * v93_data)) + (v118_data * v139_data);
              float v144_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v147_p = ((v26_data * v52_data) + (v72_data * v98_data)) + (v118_data * v144_data);
              float v149_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v152_p = ((v26_data * v57_data) + (v72_data * v103_data)) + (v118_data * v149_data);
              float v154_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v157_p = ((v26_data * v62_data) + (v72_data * v108_data)) + (v118_data * v154_data);
              float v159_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v162_p = ((v26_data * v67_data) + (v72_data * v113_data)) + (v118_data * v159_data);
              tensorforge::intel_esimd::simd<float, 8> v164_data(glb_m1_run0.template select<8, 1>(24));
              float v165_data = s0_w0[3];
              ir0.template select<8, 1>(0) = ((((v26_data * v27_data) + (v72_data * v73_data)) + (v118_data * v119_data)) + (v164_data * v165_data));
              float v170_data = s0_w0[7];
              ir0.template select<8, 1>(8) = (v127_p + (v164_data * v170_data));
              float v175_data = s0_w0[11];
              ir0.template select<8, 1>(16) = (v132_p + (v164_data * v175_data));
              float v180_data = s0_w0[15];
              ir0.template select<8, 1>(24) = (v137_p + (v164_data * v180_data));
              float v185_data = s0_w0[19];
              ir0.template select<8, 1>(32) = (v142_p + (v164_data * v185_data));
              float v190_data = s0_w0[23];
              ir0.template select<8, 1>(40) = (v147_p + (v164_data * v190_data));
              float v195_data = s0_w0[27];
              ir0.template select<8, 1>(48) = (v152_p + (v164_data * v195_data));
              float v200_data = s0_w0[31];
              ir0.template select<8, 1>(56) = (v157_p + (v164_data * v200_data));
              float v205_data = s0_w0[35];
              ir0.template select<8, 1>(64) = (v162_p + (v164_data * v205_data));
              // r0 = ir0
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v215_i0 = 0; v215_i0 < 1; ++v215_i0) {
                int32_t v217_a = v215_i0 * 8;
                #pragma unroll
                for (int32_t v216_i1 = 0; v216_i1 < 9; ++v216_i1) {
                  int32_t v218_a = v216_i1 * 8;
                  tensorforge::intel_esimd::simd<float, 8> v220_data(ir0.template select<8, 1>(v218_a));
                  v220_data.copy_to(glb_m0 + ((v217_a + v218_a)));
                }
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 32> v234_ld;
            v234_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v234_ld);
            tensorforge::intel_esimd::simd<float, 4> v235_ld;
            v235_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 32));
            tensorforge::slmStore<float, 4>(s0 + (0 + 0 + 1 * 0 + 32), v235_ld);
            v228_flagWord = v229_flagWordNext;
            v229_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


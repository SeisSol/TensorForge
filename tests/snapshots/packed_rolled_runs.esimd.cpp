// === base name ===
kernel_b1c0142ecaf938ae

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b1c0142ecaf938ae = {{1, 8, 1}, 32, 64, 1, 8, 18432, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b1c0142ecaf938ae(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b1c0142ecaf938ae(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b1c0142ecaf938ae(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4608 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b1c0142ecaf938ae(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b1c0142ecaf938ae(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b1c0142ecaf938ae(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b1c0142ecaf938ae(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4608 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: k_roll=4,pack_operands=runs
        // launch: 32 lanes (64 active) x 8 per block = block 1x8x1, 18432 B shared, occupancy grid
        // operands:
        //   m0 64×6(64×6) {0..64}×{0..6} strided
        //   m1 64×96(64×96) {0..64}×{0..96} none
        //   m2 96×6(96×6) {0..96}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":4608}],"shared_bytes":18432,"shared_elements":4608,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[64,6]],"name":"m0","ordered":false,"parts":1,"shape":[64,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[64,96]],"name":"m1","ordered":false,"parts":1,"shape":[64,96],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[96,6]],"name":"m2","ordered":false,"parts":1,"shape":[96,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[64,96]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,96]},{"addressing":"strided","bbox":[[0,0],[96,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[96,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (576 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v115_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v115_flagSecondAhead < numElements0) ? v115_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 576 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 128> v123_ld;
          v123_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 0));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v123_ld);
          tensorforge::intel_esimd::simd<float, 128> v124_ld;
          v124_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 128));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 128), v124_ld);
          tensorforge::intel_esimd::simd<float, 128> v125_ld;
          v125_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 256));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 256), v125_ld);
          tensorforge::intel_esimd::simd<float, 128> v126_ld;
          v126_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 384));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 384), v126_ld);
          tensorforge::intel_esimd::simd<float, 64> v127_ld;
          v127_ld.copy_from(peel_glb_m2 + (0 + 0 + 2 * 0 + 512));
          tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 2 * 0 + 512), v127_ld);
          uint32_t v119_flagWord(flagWordFirst);
          uint32_t v120_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 576 + 0 + m2_extraOffset];
            size_t v135_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v135_flagAheadAhead < numElements0) ? v135_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v119_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 384 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 64), (0, 6)] [(0, 96)]
              tensorforge::intel_esimd::simd<float, 384> ir0(0.0f);
              #pragma unroll 4
              for (int32_t v26_k0 = 0; v26_k0 < 96; ++v26_k0) {
                int32_t v29_a = v26_k0 * 64;
                tensorforge::intel_esimd::simd<float, 32> v31_data;
                v31_data.copy_from(glb_m1 + (v29_a));
                float v33_data = s0[v26_k0];
                tensorforge::intel_esimd::simd<float, 32> v35_data(ir0.template select<32, 1>(0));
                ir0.template select<32, 1>(0) = (v35_data + (v31_data * v33_data));
                float v39_data = s0[(v26_k0 + 96)];
                tensorforge::intel_esimd::simd<float, 32> v41_data(ir0.template select<32, 1>(64));
                ir0.template select<32, 1>(64) = (v41_data + (v31_data * v39_data));
                float v45_data = s0[(v26_k0 + 192)];
                tensorforge::intel_esimd::simd<float, 32> v47_data(ir0.template select<32, 1>(128));
                ir0.template select<32, 1>(128) = (v47_data + (v31_data * v45_data));
                float v51_data = s0[(v26_k0 + 288)];
                tensorforge::intel_esimd::simd<float, 32> v53_data(ir0.template select<32, 1>(192));
                ir0.template select<32, 1>(192) = (v53_data + (v31_data * v51_data));
                float v57_data = s0[(v26_k0 + 384)];
                tensorforge::intel_esimd::simd<float, 32> v59_data(ir0.template select<32, 1>(256));
                ir0.template select<32, 1>(256) = (v59_data + (v31_data * v57_data));
                float v63_data = s0[(v26_k0 + 480)];
                tensorforge::intel_esimd::simd<float, 32> v65_data(ir0.template select<32, 1>(320));
                ir0.template select<32, 1>(320) = (v65_data + (v31_data * v63_data));
                tensorforge::intel_esimd::simd<float, 32> v70_data;
                v70_data.copy_from(glb_m1 + ((32_i32 + v29_a)));
                tensorforge::intel_esimd::simd<float, 32> v73_data(ir0.template select<32, 1>(32));
                ir0.template select<32, 1>(32) = (v73_data + (v70_data * v33_data));
                tensorforge::intel_esimd::simd<float, 32> v78_data(ir0.template select<32, 1>(96));
                ir0.template select<32, 1>(96) = (v78_data + (v70_data * v39_data));
                tensorforge::intel_esimd::simd<float, 32> v83_data(ir0.template select<32, 1>(160));
                ir0.template select<32, 1>(160) = (v83_data + (v70_data * v45_data));
                tensorforge::intel_esimd::simd<float, 32> v88_data(ir0.template select<32, 1>(224));
                ir0.template select<32, 1>(224) = (v88_data + (v70_data * v51_data));
                tensorforge::intel_esimd::simd<float, 32> v93_data(ir0.template select<32, 1>(288));
                ir0.template select<32, 1>(288) = (v93_data + (v70_data * v57_data));
                tensorforge::intel_esimd::simd<float, 32> v98_data(ir0.template select<32, 1>(352));
                ir0.template select<32, 1>(352) = (v98_data + (v70_data * v63_data));
              }
              // r0 = ir0
              #pragma unroll
              for (int32_t v100_n0 = 0; v100_n0 < 2; ++v100_n0) {
                int32_t v102_a = v100_n0 * 32;
                #pragma unroll
                for (int32_t v101_n1 = 0; v101_n1 < 6; ++v101_n1) {
                  int32_t v104_a = v102_a + (v101_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v105_data(ir0.template select<32, 1>(v104_a));
                  r0.template select<32, 1>(v104_a) = v105_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v106_i0 = 0; v106_i0 < 2; ++v106_i0) {
                int32_t v108_a = v106_i0 * 32;
                #pragma unroll
                for (int32_t v107_i1 = 0; v107_i1 < 6; ++v107_i1) {
                  int32_t v110_a = v108_a + (v107_i1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v111_data(r0.template select<32, 1>(v110_a));
                  v111_data.copy_to(glb_m0 + (v110_a));
                }
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 128> v128_ld;
            v128_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v128_ld);
            tensorforge::intel_esimd::simd<float, 128> v129_ld;
            v129_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 128));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 128), v129_ld);
            tensorforge::intel_esimd::simd<float, 128> v130_ld;
            v130_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 256));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 256), v130_ld);
            tensorforge::intel_esimd::simd<float, 128> v131_ld;
            v131_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 384));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 384), v131_ld);
            tensorforge::intel_esimd::simd<float, 64> v132_ld;
            v132_ld.copy_from(wrap_glb_m2 + (0 + 0 + 2 * 0 + 512));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 2 * 0 + 512), v132_ld);
            v119_flagWord = v120_flagWordNext;
            v120_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


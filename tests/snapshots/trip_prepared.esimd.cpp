// === base name ===
kernel_38ba3847a7cd03f3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_38ba3847a7cd03f3 = {{1, 8, 1}, 32, 56, 1, 8, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_38ba3847a7cd03f3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_38ba3847a7cd03f3(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_38ba3847a7cd03f3(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 768 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_38ba3847a7cd03f3(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_38ba3847a7cd03f3(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_38ba3847a7cd03f3(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_38ba3847a7cd03f3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<768 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: k_roll=2,trip_reads=1
        // launch: 32 lanes (56 active) x 8 per block = block 1x8x1, 3072 B shared, occupancy grid
        // operands:
        //   m0 56×6(56×6) {0..56}×{0..6} strided
        //   m1 56×16(56×16) {0..56}×{0..16} none
        //   m2 16×6(16×6) {0..16}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[56,6]],"name":"m0","ordered":false,"parts":1,"shape":[56,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[56,16]],"name":"m1","ordered":false,"parts":1,"shape":[56,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,6]],"name":"m2","ordered":false,"parts":1,"shape":[16,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[56,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,16]},{"addressing":"strided","bbox":[[0,0],[16,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (96 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v125_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v125_flagSecondAhead < numElements0) ? v125_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 96 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 64> v133_ld;
          v133_ld.copy_from(peel_glb_m2 + (0 + 0 + 2 * 0 + 0));
          tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 2 * 0 + 0), v133_ld);
          tensorforge::intel_esimd::simd<float, 32> v134_ld;
          v134_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 64));
          tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 64), v134_ld);
          uint32_t v129_flagWord(flagWordFirst);
          uint32_t v130_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 96 + 0 + m2_extraOffset];
            size_t v139_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v139_flagAheadAhead < numElements0) ? v139_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v129_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 336 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 56), (0, 6)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 384> ir0(0.0f);
              #pragma unroll 2
              for (int32_t v23_k0 = 0; v23_k0 < 16; ++v23_k0) {
                int32_t v26_a = v23_k0 * 56;
                tensorforge::intel_esimd::simd<float, 32> v28_data;
                v28_data.copy_from(glb_m1 + (v26_a));
                float v30_data = s0[v23_k0];
                tensorforge::intel_esimd::simd<float, 32> v32_data(ir0.template select<32, 1>(0));
                ir0.template select<32, 1>(0) = (v32_data + (v28_data * v30_data));
                float v36_data = s0[(v23_k0 + 16)];
                tensorforge::intel_esimd::simd<float, 32> v38_data(ir0.template select<32, 1>(64));
                ir0.template select<32, 1>(64) = (v38_data + (v28_data * v36_data));
                float v42_data = s0[(v23_k0 + 32)];
                tensorforge::intel_esimd::simd<float, 32> v44_data(ir0.template select<32, 1>(128));
                ir0.template select<32, 1>(128) = (v44_data + (v28_data * v42_data));
                float v48_data = s0[(v23_k0 + 48)];
                tensorforge::intel_esimd::simd<float, 32> v50_data(ir0.template select<32, 1>(192));
                ir0.template select<32, 1>(192) = (v50_data + (v28_data * v48_data));
                float v54_data = s0[(v23_k0 + 64)];
                tensorforge::intel_esimd::simd<float, 32> v56_data(ir0.template select<32, 1>(256));
                ir0.template select<32, 1>(256) = (v56_data + (v28_data * v54_data));
                float v60_data = s0[(v23_k0 + 80)];
                tensorforge::intel_esimd::simd<float, 32> v62_data(ir0.template select<32, 1>(320));
                ir0.template select<32, 1>(320) = (v62_data + (v28_data * v60_data));
                tensorforge::intel_esimd::simd<float, 32> v66_data(0.0f);
                tensorforge::intel_esimd::simd<float, 24> v66_data_part;
                v66_data_part.copy_from(glb_m1 + ((32_i32 + v26_a)));
                v66_data.template select<24, 1>(0) = v66_data_part;
                tensorforge::intel_esimd::simd<float, 32> v69_data(ir0.template select<32, 1>(32));
                ir0.template select<32, 1>(32) = (v69_data + (v66_data * v30_data));
                tensorforge::intel_esimd::simd<float, 32> v74_data(ir0.template select<32, 1>(96));
                ir0.template select<32, 1>(96) = (v74_data + (v66_data * v36_data));
                tensorforge::intel_esimd::simd<float, 32> v79_data(ir0.template select<32, 1>(160));
                ir0.template select<32, 1>(160) = (v79_data + (v66_data * v42_data));
                tensorforge::intel_esimd::simd<float, 32> v84_data(ir0.template select<32, 1>(224));
                ir0.template select<32, 1>(224) = (v84_data + (v66_data * v48_data));
                tensorforge::intel_esimd::simd<float, 32> v89_data(ir0.template select<32, 1>(288));
                ir0.template select<32, 1>(288) = (v89_data + (v66_data * v54_data));
                tensorforge::intel_esimd::simd<float, 32> v94_data(ir0.template select<32, 1>(352));
                ir0.template select<32, 1>(352) = (v94_data + (v66_data * v60_data));
              }
              // r0 = ir0
              #pragma unroll
              for (int32_t v96_n0 = 0; v96_n0 < 1; ++v96_n0) {
                int32_t v98_a = v96_n0 * 32;
                #pragma unroll
                for (int32_t v97_n1 = 0; v97_n1 < 6; ++v97_n1) {
                  int32_t v100_a = v98_a + (v97_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v101_data(ir0.template select<32, 1>(v100_a));
                  r0.template select<32, 1>(v100_a) = v101_data;
                }
              }
              #pragma unroll
              for (int32_t v102_n1 = 0; v102_n1 < 6; ++v102_n1) {
                int32_t v104_a = 32 + (v102_n1 * 64);
                tensorforge::intel_esimd::simd<float, 24> v105_data(ir0.template select<24, 1>(v104_a));
                r0.template select<24, 1>(v104_a) = v105_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v106_i0 = 0; v106_i0 < 1; ++v106_i0) {
                int32_t v108_a = v106_i0 * 32;
                #pragma unroll
                for (int32_t v107_i1 = 0; v107_i1 < 6; ++v107_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v111_data(r0.template select<32, 1>((v108_a + (v107_i1 * 64))));
                  v111_data.copy_to(glb_m0 + ((v108_a + (v107_i1 * 56))));
                }
              }
              #pragma unroll
              for (int32_t v115_i1 = 0; v115_i1 < 6; ++v115_i1) {
                tensorforge::intel_esimd::simd<float, 24> v118_data(r0.template select<24, 1>((32 + (v115_i1 * 64))));
                v118_data.copy_to(glb_m0 + ((32_i32 + (v115_i1 * 56))));
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v135_ld;
            v135_ld.copy_from(wrap_glb_m2 + (0 + 0 + 2 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 2 * 0 + 0), v135_ld);
            tensorforge::intel_esimd::simd<float, 32> v136_ld;
            v136_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 64));
            tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 64), v136_ld);
            v129_flagWord = v130_flagWordNext;
            v130_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


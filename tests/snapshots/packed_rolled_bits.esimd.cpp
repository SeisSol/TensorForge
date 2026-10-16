// === base name ===
kernel_85d0f0f7aad27372

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_85d0f0f7aad27372 = {{1, 8, 1}, 32, 32, 1, 8, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_85d0f0f7aad27372(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_85d0f0f7aad27372(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_85d0f0f7aad27372(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_85d0f0f7aad27372(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_85d0f0f7aad27372(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_85d0f0f7aad27372(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_85d0f0f7aad27372(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: k_roll=4,pack_operands=bits
        // launch: 32 lanes x 8 per block = block 1x8x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 32×6(32×6) {0..32}×{0..6} strided
        //   m1 32×48(32×48) {0..32}×{0..48} none
        //   m2 48×6(48×6) {0..48}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[32,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[32,48]],"name":"m1","ordered":false,"parts":1,"shape":[32,48],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[48,6]],"name":"m2","ordered":false,"parts":1,"shape":[48,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[32,48]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,48]},{"addressing":"strided","bbox":[[0,0],[48,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[48,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (288 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v80_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v80_flagSecondAhead < numElements0) ? v80_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 288 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 128> v88_ld;
          v88_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 0));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v88_ld);
          tensorforge::intel_esimd::simd<float, 128> v89_ld;
          v89_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 128));
          tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 128), v89_ld);
          tensorforge::intel_esimd::simd<float, 32> v90_ld;
          v90_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 256));
          tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 256), v90_ld);
          uint32_t v84_flagWord(flagWordFirst);
          uint32_t v85_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 288 + 0 + m2_extraOffset];
            size_t v96_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v96_flagAheadAhead < numElements0) ? v96_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v84_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 192 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 32), (0, 6)] [(0, 48)]
              tensorforge::intel_esimd::simd<float, 192> ir0(0.0f);
              #pragma unroll 4
              for (int32_t v24_k0 = 0; v24_k0 < 48; ++v24_k0) {
                tensorforge::intel_esimd::simd<float, 32> v29_data;
                v29_data.copy_from(glb_m1 + ((v24_k0 * 32)));
                float v31_data = s0[v24_k0];
                tensorforge::intel_esimd::simd<float, 32> v33_data(ir0.template select<32, 1>(0));
                ir0.template select<32, 1>(0) = (v33_data + (v29_data * v31_data));
                float v37_data = s0[(v24_k0 + 48)];
                tensorforge::intel_esimd::simd<float, 32> v39_data(ir0.template select<32, 1>(32));
                ir0.template select<32, 1>(32) = (v39_data + (v29_data * v37_data));
                float v43_data = s0[(v24_k0 + 96)];
                tensorforge::intel_esimd::simd<float, 32> v45_data(ir0.template select<32, 1>(64));
                ir0.template select<32, 1>(64) = (v45_data + (v29_data * v43_data));
                float v49_data = s0[(v24_k0 + 144)];
                tensorforge::intel_esimd::simd<float, 32> v51_data(ir0.template select<32, 1>(96));
                ir0.template select<32, 1>(96) = (v51_data + (v29_data * v49_data));
                float v55_data = s0[(v24_k0 + 192)];
                tensorforge::intel_esimd::simd<float, 32> v57_data(ir0.template select<32, 1>(128));
                ir0.template select<32, 1>(128) = (v57_data + (v29_data * v55_data));
                float v61_data = s0[(v24_k0 + 240)];
                tensorforge::intel_esimd::simd<float, 32> v63_data(ir0.template select<32, 1>(160));
                ir0.template select<32, 1>(160) = (v63_data + (v29_data * v61_data));
              }
              // r0 = ir0
              #pragma unroll
              for (int32_t v65_n0 = 0; v65_n0 < 1; ++v65_n0) {
                int32_t v67_a = v65_n0 * 32;
                #pragma unroll
                for (int32_t v66_n1 = 0; v66_n1 < 6; ++v66_n1) {
                  int32_t v69_a = v67_a + (v66_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v70_data(ir0.template select<32, 1>(v69_a));
                  r0.template select<32, 1>(v69_a) = v70_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v71_i0 = 0; v71_i0 < 1; ++v71_i0) {
                int32_t v73_a = v71_i0 * 32;
                #pragma unroll
                for (int32_t v72_i1 = 0; v72_i1 < 6; ++v72_i1) {
                  int32_t v75_a = v73_a + (v72_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v76_data(r0.template select<32, 1>(v75_a));
                  v76_data.copy_to(glb_m0 + (v75_a));
                }
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 128> v91_ld;
            v91_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v91_ld);
            tensorforge::intel_esimd::simd<float, 128> v92_ld;
            v92_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 128));
            tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 128), v92_ld);
            tensorforge::intel_esimd::simd<float, 32> v93_ld;
            v93_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 256));
            tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 256), v93_ld);
            v84_flagWord = v85_flagWordNext;
            v85_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


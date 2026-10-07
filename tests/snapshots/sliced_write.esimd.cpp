// === base name ===
kernel_7ec77aceb30e8605

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7ec77aceb30e8605 = {{1, 8, 1}, 32, 32, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7ec77aceb30e8605(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7ec77aceb30e8605(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7ec77aceb30e8605(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1408 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7ec77aceb30e8605(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7ec77aceb30e8605(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7ec77aceb30e8605(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7ec77aceb30e8605(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1408 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 8 per block = block 1x8x1, 5632 B shared, occupancy grid
        // operands:
        //   m0 32×13(32×13) {0..32}×{0..13} strided
        //   m1 32×13(32×13) {0..32}×{0..13} strided
        //   m2 13×13(13×13) {0..13}×{0..13} strided
        // operations:
        //   m0[i,j]@{0..32}×{6..13} = m1[i,k]@{0..32}×{10..13} × m2[k,j]@{10..13}×{6..13}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,6],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,6],[13,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 169 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 96> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 32;
                #pragma unroll
                for (int32_t v21_i1 = 10; v21_i1 < 13; ++v21_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v26_data;
                  v26_data.copy_from(glb_m1 + ((v22_lead + (v21_i1 * 32))));
                  r0.template select<32, 1>((v22_lead + ((v21_i1 - 10) * 32))) = v26_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v30_ld);
              tensorforge::intel_esimd::simd<float, 32> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 128), v31_ld);
              tensorforge::intel_esimd::simd<float, 9> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 160), v32_ld);
              tensorforge::intel_esimd::simd<float, 224> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 32), (6, 13)] [(10, 13)]
              tensorforge::intel_esimd::simd<float, 224> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v35_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 88> s0_w0 = tensorforge::slmLoad<float, 88>(s0 + 88);
              float v36_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v38_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v38_data + (v35_data * v36_data));
              float v41_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v43_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v43_data + (v35_data * v41_data));
              float v46_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v48_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v48_data + (v35_data * v46_data));
              float v51_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v53_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v53_data + (v35_data * v51_data));
              float v56_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v58_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v58_data + (v35_data * v56_data));
              float v61_data = s0_w0[65];
              tensorforge::intel_esimd::simd<float, 32> v63_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v63_data + (v35_data * v61_data));
              float v66_data = s0_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v68_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v68_data + (v35_data * v66_data));
              tensorforge::intel_esimd::simd<float, 32> v70_data(r0.template select<32, 1>(32));
              float v71_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v73_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v73_data + (v70_data * v71_data));
              float v76_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v78_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v78_data + (v70_data * v76_data));
              float v81_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v83_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v83_data + (v70_data * v81_data));
              float v86_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v88_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v88_data + (v70_data * v86_data));
              float v91_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 32> v93_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v93_data + (v70_data * v91_data));
              float v96_data = s0_w0[66];
              tensorforge::intel_esimd::simd<float, 32> v98_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v98_data + (v70_data * v96_data));
              float v101_data = s0_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v103_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v103_data + (v70_data * v101_data));
              tensorforge::intel_esimd::simd<float, 32> v105_data(r0.template select<32, 1>(64));
              float v106_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v108_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v108_data + (v105_data * v106_data));
              float v111_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v113_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v113_data + (v105_data * v111_data));
              float v116_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v118_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v118_data + (v105_data * v116_data));
              float v121_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v123_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v123_data + (v105_data * v121_data));
              float v126_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 32> v128_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v128_data + (v105_data * v126_data));
              float v131_data = s0_w0[67];
              tensorforge::intel_esimd::simd<float, 32> v133_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v133_data + (v105_data * v131_data));
              float v136_data = s0_w0[80];
              tensorforge::intel_esimd::simd<float, 32> v138_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v138_data + (v105_data * v136_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v140_n0 = 0; v140_n0 < 1; ++v140_n0) {
                int32_t v142_a = v140_n0 * 32;
                #pragma unroll
                for (int32_t v141_n1 = 6; v141_n1 < 13; ++v141_n1) {
                  int32_t v145_a = v142_a + ((v141_n1 - 6) * 32);
                  tensorforge::intel_esimd::simd<float, 32> v146_data(ir1.template select<32, 1>(v145_a));
                  r1.template select<32, 1>(v145_a) = v146_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v147_i0 = 0; v147_i0 < 1; ++v147_i0) {
                int32_t v149_lead = v147_i0 * 32;
                glb_m0[v149_lead] = 0.0f;
                int32_t v152_a = v149_lead + 32;
                glb_m0[v152_a] = 0.0f;
                int32_t v153_a = v149_lead + 64;
                glb_m0[v153_a] = 0.0f;
                int32_t v154_a = v149_lead + 96;
                glb_m0[v154_a] = 0.0f;
                int32_t v155_a = v149_lead + 128;
                glb_m0[v155_a] = 0.0f;
                int32_t v156_a = v149_lead + 160;
                glb_m0[v156_a] = 0.0f;
                tensorforge::intel_esimd::simd<float, 32> v158_data(r1.template select<32, 1>(v149_lead));
                int32_t v159_a = v149_lead + 192;
                v158_data.copy_to(glb_m0 + (v159_a));
                tensorforge::intel_esimd::simd<float, 32> v161_data(r1.template select<32, 1>(v152_a));
                v161_data.copy_to(glb_m0 + ((v149_lead + 224)));
                tensorforge::intel_esimd::simd<float, 32> v164_data(r1.template select<32, 1>(v153_a));
                v164_data.copy_to(glb_m0 + ((v149_lead + 256)));
                tensorforge::intel_esimd::simd<float, 32> v167_data(r1.template select<32, 1>(v154_a));
                v167_data.copy_to(glb_m0 + ((v149_lead + 288)));
                tensorforge::intel_esimd::simd<float, 32> v170_data(r1.template select<32, 1>(v155_a));
                v170_data.copy_to(glb_m0 + ((v149_lead + 320)));
                tensorforge::intel_esimd::simd<float, 32> v173_data(r1.template select<32, 1>(v156_a));
                v173_data.copy_to(glb_m0 + ((v149_lead + 352)));
                tensorforge::intel_esimd::simd<float, 32> v176_data(r1.template select<32, 1>(v159_a));
                v176_data.copy_to(glb_m0 + ((v149_lead + 384)));
              }
            }
          }
        }
      }
    });
  });
}


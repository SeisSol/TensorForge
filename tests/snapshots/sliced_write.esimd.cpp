// === base name ===
kernel_7b4a4c241f001a0f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7b4a4c241f001a0f = {{1, 8, 1}, 32, 32, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7b4a4c241f001a0f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7b4a4c241f001a0f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7b4a4c241f001a0f(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_7b4a4c241f001a0f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7b4a4c241f001a0f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7b4a4c241f001a0f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7b4a4c241f001a0f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (176);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 169 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 96> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                int32_t v25_lead = v23_i0 * 32;
                #pragma unroll
                for (int32_t v24_i1 = 10; v24_i1 < 13; ++v24_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v29_data;
                  v29_data.copy_from(glb_m1 + ((v25_lead + (v24_i1 * 32))));
                  r0.template select<32, 1>((v25_lead + ((v24_i1 - 10) * 32))) = v29_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v33_ld);
              tensorforge::intel_esimd::simd<float, 32> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 128), v34_ld);
              tensorforge::intel_esimd::simd<float, 9> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 160), v35_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 224> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 32), (6, 13)] [(10, 13)]
              tensorforge::intel_esimd::simd<float, 224> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v38_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 88> s0_w0 = tensorforge::slmLoad<float, 88>(s0 + 88);
              float v39_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v41_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v41_data + (v38_data * v39_data));
              float v44_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v46_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v46_data + (v38_data * v44_data));
              float v49_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v51_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v51_data + (v38_data * v49_data));
              float v54_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v56_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v56_data + (v38_data * v54_data));
              float v59_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v61_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v61_data + (v38_data * v59_data));
              float v64_data = s0_w0[65];
              tensorforge::intel_esimd::simd<float, 32> v66_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v66_data + (v38_data * v64_data));
              float v69_data = s0_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v71_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v71_data + (v38_data * v69_data));
              tensorforge::intel_esimd::simd<float, 32> v73_data(r0.template select<32, 1>(32));
              float v74_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v76_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v76_data + (v73_data * v74_data));
              float v79_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v81_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v81_data + (v73_data * v79_data));
              float v84_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v86_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v86_data + (v73_data * v84_data));
              float v89_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v91_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v91_data + (v73_data * v89_data));
              float v94_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 32> v96_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v96_data + (v73_data * v94_data));
              float v99_data = s0_w0[66];
              tensorforge::intel_esimd::simd<float, 32> v101_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v101_data + (v73_data * v99_data));
              float v104_data = s0_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v106_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v106_data + (v73_data * v104_data));
              tensorforge::intel_esimd::simd<float, 32> v108_data(r0.template select<32, 1>(64));
              float v109_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v111_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v111_data + (v108_data * v109_data));
              float v114_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v116_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v116_data + (v108_data * v114_data));
              float v119_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v121_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v121_data + (v108_data * v119_data));
              float v124_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v126_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v126_data + (v108_data * v124_data));
              float v129_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 32> v131_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v131_data + (v108_data * v129_data));
              float v134_data = s0_w0[67];
              tensorforge::intel_esimd::simd<float, 32> v136_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v136_data + (v108_data * v134_data));
              float v139_data = s0_w0[80];
              tensorforge::intel_esimd::simd<float, 32> v141_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v141_data + (v108_data * v139_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v143_n0 = 0; v143_n0 < 1; ++v143_n0) {
                int32_t v145_a = v143_n0 * 32;
                #pragma unroll
                for (int32_t v144_n1 = 6; v144_n1 < 13; ++v144_n1) {
                  int32_t v148_a = v145_a + ((v144_n1 - 6) * 32);
                  tensorforge::intel_esimd::simd<float, 32> v149_data(ir1.template select<32, 1>(v148_a));
                  r1.template select<32, 1>(v148_a) = v149_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v150_i0 = 0; v150_i0 < 1; ++v150_i0) {
                int32_t v152_lead = v150_i0 * 32;
                glb_m0[v152_lead] = 0.0f;
                int32_t v155_a = v152_lead + 32;
                glb_m0[v155_a] = 0.0f;
                int32_t v156_a = v152_lead + 64;
                glb_m0[v156_a] = 0.0f;
                int32_t v157_a = v152_lead + 96;
                glb_m0[v157_a] = 0.0f;
                int32_t v158_a = v152_lead + 128;
                glb_m0[v158_a] = 0.0f;
                int32_t v159_a = v152_lead + 160;
                glb_m0[v159_a] = 0.0f;
                tensorforge::intel_esimd::simd<float, 32> v161_data(r1.template select<32, 1>(v152_lead));
                int32_t v162_a = v152_lead + 192;
                v161_data.copy_to(glb_m0 + (v162_a));
                tensorforge::intel_esimd::simd<float, 32> v164_data(r1.template select<32, 1>(v155_a));
                v164_data.copy_to(glb_m0 + ((v152_lead + 224)));
                tensorforge::intel_esimd::simd<float, 32> v167_data(r1.template select<32, 1>(v156_a));
                v167_data.copy_to(glb_m0 + ((v152_lead + 256)));
                tensorforge::intel_esimd::simd<float, 32> v170_data(r1.template select<32, 1>(v157_a));
                v170_data.copy_to(glb_m0 + ((v152_lead + 288)));
                tensorforge::intel_esimd::simd<float, 32> v173_data(r1.template select<32, 1>(v158_a));
                v173_data.copy_to(glb_m0 + ((v152_lead + 320)));
                tensorforge::intel_esimd::simd<float, 32> v176_data(r1.template select<32, 1>(v159_a));
                v176_data.copy_to(glb_m0 + ((v152_lead + 352)));
                tensorforge::intel_esimd::simd<float, 32> v179_data(r1.template select<32, 1>(v162_a));
                v179_data.copy_to(glb_m0 + ((v152_lead + 384)));
              }
            }
          }
        }
      }
    });
  });
}


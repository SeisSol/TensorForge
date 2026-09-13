// === base name ===
kernel_d5c085ee699745f0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d5c085ee699745f0 = {{1, 8, 1}, 32, 32, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d5c085ee699745f0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d5c085ee699745f0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d5c085ee699745f0(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 1408 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_d5c085ee699745f0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d5c085ee699745f0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d5c085ee699745f0(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d5c085ee699745f0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1408 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 8 per block = block 1x8x1, 5632 B shared, occupancy grid
        // operands:
        //   m0 32×13(32×13) {0..32}×{0..13} strided
        //   m1 32×13(32×13) {0..32}×{0..13} strided
        //   m2 13×13(13×13) {0..13}×{0..13} strided
        // operations:
        //   m0[i,j]@{0..32}×{6..13} = m1[i,k]@{0..32}×{10..13} × m2[k,j]@{10..13}×{6..13}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,6],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,6],[13,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (176);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 416 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 169 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 169 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 96> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                int32_t v21_lead = v19_i0 * 32;
                #pragma unroll
                for (int32_t v20_i1 = 10; v20_i1 < 13; ++v20_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v25_data;
                  v25_data.copy_from(glb_m1 + ((v21_lead + (v20_i1 * 32))));
                  r0.template select<32, 1>((v21_lead + ((v20_i1 - 10) * 32))) = v25_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 128>(s0 + (0 + 0 + 4 * 0 + 0), v29_ld);
              tensorforge::intel_esimd::simd<float, 32> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 128), v30_ld);
              tensorforge::intel_esimd::simd<float, 9> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 160), v31_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 224> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              tensorforge::intel_esimd::simd<float, 224> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v34_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 88> s0_w0 = tensorforge::slmLoad<float, 88>(s0 + 88);
              float v35_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v37_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v37_data + (v34_data * v35_data));
              float v40_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v42_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v42_data + (v34_data * v40_data));
              float v45_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v47_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v47_data + (v34_data * v45_data));
              float v50_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v52_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v52_data + (v34_data * v50_data));
              float v55_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v57_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v57_data + (v34_data * v55_data));
              float v60_data = s0_w0[65];
              tensorforge::intel_esimd::simd<float, 32> v62_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v62_data + (v34_data * v60_data));
              float v65_data = s0_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v67_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v67_data + (v34_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v69_data(r0.template select<32, 1>(32));
              float v70_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v72_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v72_data + (v69_data * v70_data));
              float v75_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v77_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v77_data + (v69_data * v75_data));
              float v80_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v82_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v82_data + (v69_data * v80_data));
              float v85_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v87_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v87_data + (v69_data * v85_data));
              float v90_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 32> v92_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v92_data + (v69_data * v90_data));
              float v95_data = s0_w0[66];
              tensorforge::intel_esimd::simd<float, 32> v97_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v97_data + (v69_data * v95_data));
              float v100_data = s0_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v102_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v102_data + (v69_data * v100_data));
              tensorforge::intel_esimd::simd<float, 32> v104_data(r0.template select<32, 1>(64));
              float v105_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v107_data(ir1.template select<32, 1>(0));
              ir1.template select<32, 1>(0) = (v107_data + (v104_data * v105_data));
              float v110_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v112_data(ir1.template select<32, 1>(32));
              ir1.template select<32, 1>(32) = (v112_data + (v104_data * v110_data));
              float v115_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v117_data(ir1.template select<32, 1>(64));
              ir1.template select<32, 1>(64) = (v117_data + (v104_data * v115_data));
              float v120_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v122_data(ir1.template select<32, 1>(96));
              ir1.template select<32, 1>(96) = (v122_data + (v104_data * v120_data));
              float v125_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 32> v127_data(ir1.template select<32, 1>(128));
              ir1.template select<32, 1>(128) = (v127_data + (v104_data * v125_data));
              float v130_data = s0_w0[67];
              tensorforge::intel_esimd::simd<float, 32> v132_data(ir1.template select<32, 1>(160));
              ir1.template select<32, 1>(160) = (v132_data + (v104_data * v130_data));
              float v135_data = s0_w0[80];
              tensorforge::intel_esimd::simd<float, 32> v137_data(ir1.template select<32, 1>(192));
              ir1.template select<32, 1>(192) = (v137_data + (v104_data * v135_data));
              #pragma unroll
              for (int32_t v139_n0 = 0; v139_n0 < 1; ++v139_n0) {
                int32_t v141_a = v139_n0 * 32;
                #pragma unroll
                for (int32_t v140_n1 = 6; v140_n1 < 13; ++v140_n1) {
                  int32_t v144_a = v141_a + ((v140_n1 - 6) * 32);
                  tensorforge::intel_esimd::simd<float, 32> v145_data(ir1.template select<32, 1>(v144_a));
                  r1.template select<32, 1>(v144_a) = v145_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v146_i0 = 0; v146_i0 < 1; ++v146_i0) {
                int32_t v148_lead = v146_i0 * 32;
                glb_m0[v148_lead] = 0.0f;
                int32_t v151_a = v148_lead + 32;
                glb_m0[v151_a] = 0.0f;
                int32_t v152_a = v148_lead + 64;
                glb_m0[v152_a] = 0.0f;
                int32_t v153_a = v148_lead + 96;
                glb_m0[v153_a] = 0.0f;
                int32_t v154_a = v148_lead + 128;
                glb_m0[v154_a] = 0.0f;
                int32_t v155_a = v148_lead + 160;
                glb_m0[v155_a] = 0.0f;
                tensorforge::intel_esimd::simd<float, 32> v157_data(r1.template select<32, 1>(v148_lead));
                int32_t v158_a = v148_lead + 192;
                v157_data.copy_to(glb_m0 + (v158_a));
                tensorforge::intel_esimd::simd<float, 32> v160_data(r1.template select<32, 1>(v151_a));
                v160_data.copy_to(glb_m0 + ((v148_lead + 224)));
                tensorforge::intel_esimd::simd<float, 32> v163_data(r1.template select<32, 1>(v152_a));
                v163_data.copy_to(glb_m0 + ((v148_lead + 256)));
                tensorforge::intel_esimd::simd<float, 32> v166_data(r1.template select<32, 1>(v153_a));
                v166_data.copy_to(glb_m0 + ((v148_lead + 288)));
                tensorforge::intel_esimd::simd<float, 32> v169_data(r1.template select<32, 1>(v154_a));
                v169_data.copy_to(glb_m0 + ((v148_lead + 320)));
                tensorforge::intel_esimd::simd<float, 32> v172_data(r1.template select<32, 1>(v155_a));
                v172_data.copy_to(glb_m0 + ((v148_lead + 352)));
                tensorforge::intel_esimd::simd<float, 32> v175_data(r1.template select<32, 1>(v158_a));
                v175_data.copy_to(glb_m0 + ((v148_lead + 384)));
              }
            }
            tensorforge::prefetchL2<416>(&pf_glb_m1[0]);
            tensorforge::prefetchL2<169>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


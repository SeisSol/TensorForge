// === base name ===
kernel_12876e1bfc9758bb

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_12876e1bfc9758bb = {{1, 8, 1}, 32, 32, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_12876e1bfc9758bb(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_12876e1bfc9758bb(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_12876e1bfc9758bb(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_12876e1bfc9758bb(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_12876e1bfc9758bb(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_12876e1bfc9758bb(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_12876e1bfc9758bb(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      // generated with TensorForge. Version: 0.0.1
      // options: default
      // launch: 32 lanes x 8 per block = block 1x8x1, 0 B shared, occupancy grid
      // operands:
      //   m0 32(32) {0..32} pointer_based
      //   m1 32×3(32×3) {0..32}×{0..3} pointer_based
      //   m2 32×3(32×3) {0..32}×{0..3} pointer_based
      // operations:
      //   t0[i] = m0[i]
      //   t1[i,j] = m1[i,j]
      //   t2[i,j] = t0[i]
      //   t2[i,j] += t1[i,j]
      //   m2[i,j] = t2[i,j]
      // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0],[32]],"name":"m0","ordered":false,"parts":1,"shape":[32],"variant":false},{"addressing":"pointer_based","alias":"B","bbox":[[0,0],[32,3]],"name":"m1","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,3]],"name":"m2","ordered":false,"parts":1,"shape":[32,3],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[32]],"is_tmp":true,"name":"t0","offset":[0],"shape":[32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0],[32]],"is_tmp":false,"name":"m0","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,3]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[32]],"is_tmp":true,"name":"t0","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,3]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
      {
        const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
        const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
        const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
        for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
          size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
          const float *const __restrict__ pf_glb_m0 = &m0[v4_batchId1][0 + m0_extraOffset];
          const float *const __restrict__ pf_glb_m1 = &m1[v4_batchId1][0 + m1_extraOffset];
          const bool allowed_next = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId1]);
          const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
          if (allowed) {
            const float *const __restrict__ glb_m0 = &m0[v1_batchId0][0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v1_batchId0][0 + m1_extraOffset];
            float *const __restrict__ glb_m2 = &m2[v1_batchId0][0 + m2_extraOffset];
            tensorforge::intel_esimd::simd<float, 32> r0(0.0f);
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
              int32_t v17_lead = v16_i0 * 32;
              tensorforge::intel_esimd::simd<float, 32> v19_data;
              v19_data.copy_from(glb_m0 + (v17_lead));
              r0.template select<32, 1>(v17_lead) = v19_data;
            }
            tensorforge::intel_esimd::simd<float, 96> r2(0.0f);
            // r2 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
              int32_t v23_lead = v21_i0 * 32;
              #pragma unroll
              for (int32_t v22_i1 = 0; v22_i1 < 3; ++v22_i1) {
                int32_t v26_a = v23_lead + (v22_i1 * 32);
                tensorforge::intel_esimd::simd<float, 32> v27_data;
                v27_data.copy_from(glb_m1 + (v26_a));
                r2.template select<32, 1>(v26_a) = v27_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            tensorforge::intel_esimd::simd<float, 32> r1(0.0f);
            // r1 = +(r0) + None
            // [(0, 32)] []
            tensorforge::intel_esimd::simd<float, 32> v30_data(r0.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v31_data(r1.template select<32, 1>(0));
            r1.template select<32, 1>(0) = (v31_data + v30_data);
            // wait(r2 = load{g>r}(glb_m1););
            tensorforge::intel_esimd::simd<float, 96> r3(0.0f);
            // r3 = +(r2) + None
            // [(0, 32), (0, 3)] []
            tensorforge::intel_esimd::simd<float, 32> v34_data(r2.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v35_data(r3.template select<32, 1>(0));
            r3.template select<32, 1>(0) = (v35_data + v34_data);
            tensorforge::intel_esimd::simd<float, 32> v37_data(r2.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v38_data(r3.template select<32, 1>(32));
            r3.template select<32, 1>(32) = (v38_data + v37_data);
            tensorforge::intel_esimd::simd<float, 32> v40_data(r2.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v41_data(r3.template select<32, 1>(64));
            r3.template select<32, 1>(64) = (v41_data + v40_data);
            tensorforge::intel_esimd::simd<float, 96> r4(0.0f);
            // r4 = +(r1) + None
            // [(0, 32), (0, 3)] []
            tensorforge::intel_esimd::simd<float, 32> v44_data(r1.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v45_data(r4.template select<32, 1>(0));
            r4.template select<32, 1>(0) = (v45_data + v44_data);
            tensorforge::intel_esimd::simd<float, 32> v48_data(r4.template select<32, 1>(32));
            r4.template select<32, 1>(32) = (v48_data + v44_data);
            tensorforge::intel_esimd::simd<float, 32> v51_data(r4.template select<32, 1>(64));
            r4.template select<32, 1>(64) = (v51_data + v44_data);
            tensorforge::intel_esimd::simd<float, 96> r5(0.0f);
            // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
            // [(0, 32), (0, 3)] []
            tensorforge::intel_esimd::simd<float, 96> ir5(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v55_data(r3.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v56_data(ir5.template select<32, 1>(0));
            ir5.template select<32, 1>(0) = (v56_data + v55_data);
            tensorforge::intel_esimd::simd<float, 32> v58_data(r3.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v59_data(ir5.template select<32, 1>(32));
            ir5.template select<32, 1>(32) = (v59_data + v58_data);
            tensorforge::intel_esimd::simd<float, 32> v61_data(r3.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v62_data(ir5.template select<32, 1>(64));
            ir5.template select<32, 1>(64) = (v62_data + v61_data);
            #pragma unroll
            for (int32_t v64_n0 = 0; v64_n0 < 1; ++v64_n0) {
              int32_t v66_a = v64_n0 * 32;
              #pragma unroll
              for (int32_t v65_n1 = 0; v65_n1 < 3; ++v65_n1) {
                int32_t v68_a = v66_a + (v65_n1 * 32);
                tensorforge::intel_esimd::simd<float, 32> v69_data(ir5.template select<32, 1>(v68_a));
                tensorforge::intel_esimd::simd<float, 32> v70_data(r4.template select<32, 1>(v68_a));
                r5.template select<32, 1>(v68_a) = (v70_data + v69_data);
              }
            }
            tensorforge::intel_esimd::simd<float, 96> r6(0.0f);
            // r6 = +(r5) + None
            // [(0, 32), (0, 3)] []
            tensorforge::intel_esimd::simd<float, 96> ir6(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v74_data(r5.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v75_data(ir6.template select<32, 1>(0));
            ir6.template select<32, 1>(0) = (v75_data + v74_data);
            tensorforge::intel_esimd::simd<float, 32> v77_data(r5.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v78_data(ir6.template select<32, 1>(32));
            ir6.template select<32, 1>(32) = (v78_data + v77_data);
            tensorforge::intel_esimd::simd<float, 32> v80_data(r5.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v81_data(ir6.template select<32, 1>(64));
            ir6.template select<32, 1>(64) = (v81_data + v80_data);
            #pragma unroll
            for (int32_t v83_n0 = 0; v83_n0 < 1; ++v83_n0) {
              int32_t v85_a = v83_n0 * 32;
              #pragma unroll
              for (int32_t v84_n1 = 0; v84_n1 < 3; ++v84_n1) {
                int32_t v87_a = v85_a + (v84_n1 * 32);
                tensorforge::intel_esimd::simd<float, 32> v88_data(ir6.template select<32, 1>(v87_a));
                r6.template select<32, 1>(v87_a) = v88_data;
              }
            }
            // glb_m2 = store{r>g}(r6);
            #pragma unroll
            for (int32_t v89_i0 = 0; v89_i0 < 1; ++v89_i0) {
              int32_t v91_a = v89_i0 * 32;
              #pragma unroll
              for (int32_t v90_i1 = 0; v90_i1 < 3; ++v90_i1) {
                int32_t v93_a = v91_a + (v90_i1 * 32);
                tensorforge::intel_esimd::simd<float, 32> v94_data(r6.template select<32, 1>(v93_a));
                v94_data.copy_to(glb_m2 + (v93_a));
              }
            }
          }
          if (allowed_next) {
            tensorforge::prefetchRunsL2<128, 384>(&pf_glb_m0[0], &pf_glb_m1[0]);
          }
        }
      }
    });
  });
}


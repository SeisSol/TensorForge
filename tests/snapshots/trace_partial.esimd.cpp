// === base name ===
kernel_8b7edbec27c3ad2c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8b7edbec27c3ad2c = {{1, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8b7edbec27c3ad2c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8b7edbec27c3ad2c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8b7edbec27c3ad2c(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_8b7edbec27c3ad2c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8b7edbec27c3ad2c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8b7edbec27c3ad2c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8b7edbec27c3ad2c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<256 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16(16) {0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i] = m1[i,k]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"OUT","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (16 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (0);
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v7_batchId1 * 256 + 0 + m1_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 256 + 0 + m1_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
                int32_t v18_lead = v16_i0 * 16;
                #pragma unroll
                for (int32_t v17_i1 = 0; v17_i1 < 16; ++v17_i1) {
                  int32_t v21_a = v18_lead + (v17_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v22_data;
                  v22_data.copy_from(glb_m1 + (v21_a));
                  r0.template select<16, 1>(v21_a) = v22_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 16> r1(0.0f);
              // r1 = +(r0) + None
              // [(0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 16> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v26_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v27_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v27_data + v26_data);
              tensorforge::intel_esimd::simd<float, 16> v29_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v30_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v30_data + v29_data);
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v33_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v33_data + v32_data);
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v36_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v36_data + v35_data);
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v39_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v39_data + v38_data);
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v42_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v42_data + v41_data);
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v45_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v45_data + v44_data);
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v48_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v48_data + v47_data);
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v51_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v51_data + v50_data);
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v54_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v54_data + v53_data);
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v57_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v57_data + v56_data);
              tensorforge::intel_esimd::simd<float, 16> v59_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v60_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v60_data + v59_data);
              tensorforge::intel_esimd::simd<float, 16> v62_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v63_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v63_data + v62_data);
              tensorforge::intel_esimd::simd<float, 16> v65_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v66_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v66_data + v65_data);
              tensorforge::intel_esimd::simd<float, 16> v68_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v69_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v69_data + v68_data);
              tensorforge::intel_esimd::simd<float, 16> v71_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v72_data(ir1.template select<16, 1>(0));
              ir1.template select<16, 1>(0) = (v72_data + v71_data);
              #pragma unroll
              for (int32_t v74_n0 = 0; v74_n0 < 1; ++v74_n0) {
                int32_t v75_a = v74_n0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v76_data(ir1.template select<16, 1>(v75_a));
                r1.template select<16, 1>(v75_a) = v76_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v77_i0 = 0; v77_i0 < 1; ++v77_i0) {
                int32_t v78_a = v77_i0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v79_data(r1.template select<16, 1>(v78_a));
                v79_data.copy_to(glb_m0 + (v78_a));
              }
            }
            tensorforge::prefetchL2<256>(&pf_glb_m1[0]);
          }
        }
      }
    });
  });
}


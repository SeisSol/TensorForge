// === base name ===
kernel_da9d751ac97609bf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_da9d751ac97609bf = {{1, 16, 1}, 16, 16, 1, 16, 5120, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_da9d751ac97609bf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_da9d751ac97609bf(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_da9d751ac97609bf(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 1280 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_da9d751ac97609bf(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_da9d751ac97609bf(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_da9d751ac97609bf(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_da9d751ac97609bf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1280 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 5120 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8(8) {0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   OUT = +(TMP, dims=[1])
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1280}],"shared_bytes":5120,"shared_elements":1280,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[8]],"name":"m2","ordered":false,"parts":1,"shape":[8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[8]],"is_tmp":false,"name":"m2","offset":[0],"shape":[8]},"kind":"reduction","op":"+","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (80 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 8 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 128> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v18_i1 = 0; v18_i1 < 8; ++v18_i1) {
                tensorforge::intel_esimd::simd<float, 8> v23_data;
                v23_data.copy_from(glb_m0 + ((v18_i1 * 8)));
                r0.template select<8, 1>((v18_i1 * 16)) = v23_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v26_ld;
              v26_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v26_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v28_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v29_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v30_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v31_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v36_acc{};
              tensorforge::intel_esimd::simd<float, 16> v40_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v36_acc += ((static_cast<float>(v40_data[0])) * v28_data);
              v36_acc += ((static_cast<float>(v40_data[1])) * v29_data);
              v36_acc += ((static_cast<float>(v40_data[2])) * v30_data);
              v36_acc += ((static_cast<float>(v40_data[3])) * v31_data);
              v36_acc += ((static_cast<float>(v40_data[4])) * v32_data);
              v36_acc += ((static_cast<float>(v40_data[5])) * v33_data);
              v36_acc += ((static_cast<float>(v40_data[6])) * v34_data);
              v36_acc += ((static_cast<float>(v40_data[7])) * v35_data);
              r1.template select<16, 1>(0) = v36_acc;
              tensorforge::intel_esimd::simd<float, 16> v57_acc{};
              tensorforge::intel_esimd::simd<float, 16> v59_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              v57_acc += ((static_cast<float>(v59_data[0])) * v28_data);
              v57_acc += ((static_cast<float>(v59_data[1])) * v29_data);
              v57_acc += ((static_cast<float>(v59_data[2])) * v30_data);
              v57_acc += ((static_cast<float>(v59_data[3])) * v31_data);
              v57_acc += ((static_cast<float>(v59_data[4])) * v32_data);
              v57_acc += ((static_cast<float>(v59_data[5])) * v33_data);
              v57_acc += ((static_cast<float>(v59_data[6])) * v34_data);
              v57_acc += ((static_cast<float>(v59_data[7])) * v35_data);
              r1.template select<16, 1>(16) = v57_acc;
              tensorforge::intel_esimd::simd<float, 16> v76_acc{};
              tensorforge::intel_esimd::simd<float, 16> v78_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v76_acc += ((static_cast<float>(v78_data[0])) * v28_data);
              v76_acc += ((static_cast<float>(v78_data[1])) * v29_data);
              v76_acc += ((static_cast<float>(v78_data[2])) * v30_data);
              v76_acc += ((static_cast<float>(v78_data[3])) * v31_data);
              v76_acc += ((static_cast<float>(v78_data[4])) * v32_data);
              v76_acc += ((static_cast<float>(v78_data[5])) * v33_data);
              v76_acc += ((static_cast<float>(v78_data[6])) * v34_data);
              v76_acc += ((static_cast<float>(v78_data[7])) * v35_data);
              r1.template select<16, 1>(32) = v76_acc;
              tensorforge::intel_esimd::simd<float, 16> v95_acc{};
              tensorforge::intel_esimd::simd<float, 16> v97_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v95_acc += ((static_cast<float>(v97_data[0])) * v28_data);
              v95_acc += ((static_cast<float>(v97_data[1])) * v29_data);
              v95_acc += ((static_cast<float>(v97_data[2])) * v30_data);
              v95_acc += ((static_cast<float>(v97_data[3])) * v31_data);
              v95_acc += ((static_cast<float>(v97_data[4])) * v32_data);
              v95_acc += ((static_cast<float>(v97_data[5])) * v33_data);
              v95_acc += ((static_cast<float>(v97_data[6])) * v34_data);
              v95_acc += ((static_cast<float>(v97_data[7])) * v35_data);
              r1.template select<16, 1>(48) = v95_acc;
              tensorforge::intel_esimd::simd<float, 16> v114_acc{};
              tensorforge::intel_esimd::simd<float, 16> v116_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v114_acc += ((static_cast<float>(v116_data[0])) * v28_data);
              v114_acc += ((static_cast<float>(v116_data[1])) * v29_data);
              v114_acc += ((static_cast<float>(v116_data[2])) * v30_data);
              v114_acc += ((static_cast<float>(v116_data[3])) * v31_data);
              v114_acc += ((static_cast<float>(v116_data[4])) * v32_data);
              v114_acc += ((static_cast<float>(v116_data[5])) * v33_data);
              v114_acc += ((static_cast<float>(v116_data[6])) * v34_data);
              v114_acc += ((static_cast<float>(v116_data[7])) * v35_data);
              r1.template select<16, 1>(64) = v114_acc;
              tensorforge::intel_esimd::simd<float, 16> v133_acc{};
              tensorforge::intel_esimd::simd<float, 16> v135_data = tensorforge::slmLoad<float, 16>(s0 + (40_i32));
              v133_acc += ((static_cast<float>(v135_data[0])) * v28_data);
              v133_acc += ((static_cast<float>(v135_data[1])) * v29_data);
              v133_acc += ((static_cast<float>(v135_data[2])) * v30_data);
              v133_acc += ((static_cast<float>(v135_data[3])) * v31_data);
              v133_acc += ((static_cast<float>(v135_data[4])) * v32_data);
              v133_acc += ((static_cast<float>(v135_data[5])) * v33_data);
              v133_acc += ((static_cast<float>(v135_data[6])) * v34_data);
              v133_acc += ((static_cast<float>(v135_data[7])) * v35_data);
              r1.template select<16, 1>(80) = v133_acc;
              tensorforge::intel_esimd::simd<float, 16> v152_acc{};
              tensorforge::intel_esimd::simd<float, 16> v154_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v152_acc += ((static_cast<float>(v154_data[0])) * v28_data);
              v152_acc += ((static_cast<float>(v154_data[1])) * v29_data);
              v152_acc += ((static_cast<float>(v154_data[2])) * v30_data);
              v152_acc += ((static_cast<float>(v154_data[3])) * v31_data);
              v152_acc += ((static_cast<float>(v154_data[4])) * v32_data);
              v152_acc += ((static_cast<float>(v154_data[5])) * v33_data);
              v152_acc += ((static_cast<float>(v154_data[6])) * v34_data);
              v152_acc += ((static_cast<float>(v154_data[7])) * v35_data);
              r1.template select<16, 1>(96) = v152_acc;
              tensorforge::intel_esimd::simd<float, 16> v171_acc{};
              tensorforge::intel_esimd::simd<float, 16> v173_data = tensorforge::slmLoad<float, 16>(s0 + (56_i32));
              v171_acc += ((static_cast<float>(v173_data[0])) * v28_data);
              v171_acc += ((static_cast<float>(v173_data[1])) * v29_data);
              v171_acc += ((static_cast<float>(v173_data[2])) * v30_data);
              v171_acc += ((static_cast<float>(v173_data[3])) * v31_data);
              v171_acc += ((static_cast<float>(v173_data[4])) * v32_data);
              v171_acc += ((static_cast<float>(v173_data[5])) * v33_data);
              v171_acc += ((static_cast<float>(v173_data[6])) * v34_data);
              v171_acc += ((static_cast<float>(v173_data[7])) * v35_data);
              r1.template select<16, 1>(112) = v171_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v190_i1 = 0; v190_i1 < 8; ++v190_i1) {
                tensorforge::intel_esimd::simd<float, 8> v193_data(r1.template select<8, 1>((v190_i1 * 16)));
                tensorforge::slmStore<float, 8>(s1 + ((v190_i1 * 8)), v193_data);
              }
              // glb_m2 = +(s1, dims=[1])
              tensorforge::intel_esimd::simd<float, 8> v199_acc0(0.0f);
              #pragma unroll
              for (int32_t v198_r1 = 0; v198_r1 < 8; ++v198_r1) {
                tensorforge::intel_esimd::simd<float, 8> v204_data = tensorforge::slmLoad<float, 8>(s1 + ((v198_r1 * 8)));
                v199_acc0 = (v199_acc0 + v204_data);
              }
              v199_acc0.copy_to(glb_m2 + (0_i32));
            }
          }
        }
      }
    });
  });
}


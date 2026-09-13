// === base name ===
kernel_575ccda9425a5e68

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_575ccda9425a5e68 = {{1, 16, 1}, 16, 16, 1, 16, 5120, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_575ccda9425a5e68(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_575ccda9425a5e68(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_575ccda9425a5e68(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1280 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_575ccda9425a5e68(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_575ccda9425a5e68(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_575ccda9425a5e68(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_575ccda9425a5e68(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1280 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 5120 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   TMP = abs(A)
        //   m1[i,j] = t0[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1280}],"shared_bytes":5120,"shared_elements":1280,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (80 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 64 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 64 + 0 + m2_extraOffset];
              // s1 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v17_ld);
              tensorforge::intel_esimd::simd<float, 128> r0(0.0f);
              // r0 = abs(glb_m0)
              #pragma unroll
              for (int32_t v19_k1 = 0; v19_k1 < 8; ++v19_k1) {
                tensorforge::intel_esimd::simd<float, 8> v24_data;
                v24_data.copy_from(glb_m0 + ((v19_k1 * 8)));
                r0.template select<8, 1>((v19_k1 * 16)) = (tensorforge::intel_esimd::abs(v24_data));
              }
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = +(r0 * s1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v30_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v31_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v38_acc{};
              tensorforge::intel_esimd::simd<float, 16> v42_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v38_acc += ((static_cast<float>(v42_data[0])) * v30_data);
              v38_acc += ((static_cast<float>(v42_data[1])) * v31_data);
              v38_acc += ((static_cast<float>(v42_data[2])) * v32_data);
              v38_acc += ((static_cast<float>(v42_data[3])) * v33_data);
              v38_acc += ((static_cast<float>(v42_data[4])) * v34_data);
              v38_acc += ((static_cast<float>(v42_data[5])) * v35_data);
              v38_acc += ((static_cast<float>(v42_data[6])) * v36_data);
              v38_acc += ((static_cast<float>(v42_data[7])) * v37_data);
              ir1.template select<16, 1>(0) = v38_acc;
              tensorforge::intel_esimd::simd<float, 16> v59_acc{};
              tensorforge::intel_esimd::simd<float, 16> v61_data = tensorforge::slmLoad<float, 16>(s1 + (8_i32));
              v59_acc += ((static_cast<float>(v61_data[0])) * v30_data);
              v59_acc += ((static_cast<float>(v61_data[1])) * v31_data);
              v59_acc += ((static_cast<float>(v61_data[2])) * v32_data);
              v59_acc += ((static_cast<float>(v61_data[3])) * v33_data);
              v59_acc += ((static_cast<float>(v61_data[4])) * v34_data);
              v59_acc += ((static_cast<float>(v61_data[5])) * v35_data);
              v59_acc += ((static_cast<float>(v61_data[6])) * v36_data);
              v59_acc += ((static_cast<float>(v61_data[7])) * v37_data);
              ir1.template select<16, 1>(16) = v59_acc;
              tensorforge::intel_esimd::simd<float, 16> v78_acc{};
              tensorforge::intel_esimd::simd<float, 16> v80_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v78_acc += ((static_cast<float>(v80_data[0])) * v30_data);
              v78_acc += ((static_cast<float>(v80_data[1])) * v31_data);
              v78_acc += ((static_cast<float>(v80_data[2])) * v32_data);
              v78_acc += ((static_cast<float>(v80_data[3])) * v33_data);
              v78_acc += ((static_cast<float>(v80_data[4])) * v34_data);
              v78_acc += ((static_cast<float>(v80_data[5])) * v35_data);
              v78_acc += ((static_cast<float>(v80_data[6])) * v36_data);
              v78_acc += ((static_cast<float>(v80_data[7])) * v37_data);
              ir1.template select<16, 1>(32) = v78_acc;
              tensorforge::intel_esimd::simd<float, 16> v97_acc{};
              tensorforge::intel_esimd::simd<float, 16> v99_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v97_acc += ((static_cast<float>(v99_data[0])) * v30_data);
              v97_acc += ((static_cast<float>(v99_data[1])) * v31_data);
              v97_acc += ((static_cast<float>(v99_data[2])) * v32_data);
              v97_acc += ((static_cast<float>(v99_data[3])) * v33_data);
              v97_acc += ((static_cast<float>(v99_data[4])) * v34_data);
              v97_acc += ((static_cast<float>(v99_data[5])) * v35_data);
              v97_acc += ((static_cast<float>(v99_data[6])) * v36_data);
              v97_acc += ((static_cast<float>(v99_data[7])) * v37_data);
              ir1.template select<16, 1>(48) = v97_acc;
              tensorforge::intel_esimd::simd<float, 16> v116_acc{};
              tensorforge::intel_esimd::simd<float, 16> v118_data = tensorforge::slmLoad<float, 16>(s1 + (32_i32));
              v116_acc += ((static_cast<float>(v118_data[0])) * v30_data);
              v116_acc += ((static_cast<float>(v118_data[1])) * v31_data);
              v116_acc += ((static_cast<float>(v118_data[2])) * v32_data);
              v116_acc += ((static_cast<float>(v118_data[3])) * v33_data);
              v116_acc += ((static_cast<float>(v118_data[4])) * v34_data);
              v116_acc += ((static_cast<float>(v118_data[5])) * v35_data);
              v116_acc += ((static_cast<float>(v118_data[6])) * v36_data);
              v116_acc += ((static_cast<float>(v118_data[7])) * v37_data);
              ir1.template select<16, 1>(64) = v116_acc;
              tensorforge::intel_esimd::simd<float, 16> v135_acc{};
              tensorforge::intel_esimd::simd<float, 16> v137_data = tensorforge::slmLoad<float, 16>(s1 + (40_i32));
              v135_acc += ((static_cast<float>(v137_data[0])) * v30_data);
              v135_acc += ((static_cast<float>(v137_data[1])) * v31_data);
              v135_acc += ((static_cast<float>(v137_data[2])) * v32_data);
              v135_acc += ((static_cast<float>(v137_data[3])) * v33_data);
              v135_acc += ((static_cast<float>(v137_data[4])) * v34_data);
              v135_acc += ((static_cast<float>(v137_data[5])) * v35_data);
              v135_acc += ((static_cast<float>(v137_data[6])) * v36_data);
              v135_acc += ((static_cast<float>(v137_data[7])) * v37_data);
              ir1.template select<16, 1>(80) = v135_acc;
              tensorforge::intel_esimd::simd<float, 16> v154_acc{};
              tensorforge::intel_esimd::simd<float, 16> v156_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v154_acc += ((static_cast<float>(v156_data[0])) * v30_data);
              v154_acc += ((static_cast<float>(v156_data[1])) * v31_data);
              v154_acc += ((static_cast<float>(v156_data[2])) * v32_data);
              v154_acc += ((static_cast<float>(v156_data[3])) * v33_data);
              v154_acc += ((static_cast<float>(v156_data[4])) * v34_data);
              v154_acc += ((static_cast<float>(v156_data[5])) * v35_data);
              v154_acc += ((static_cast<float>(v156_data[6])) * v36_data);
              v154_acc += ((static_cast<float>(v156_data[7])) * v37_data);
              ir1.template select<16, 1>(96) = v154_acc;
              tensorforge::intel_esimd::simd<float, 16> v173_acc{};
              tensorforge::intel_esimd::simd<float, 16> v175_data = tensorforge::slmLoad<float, 16>(s1 + (56_i32));
              v173_acc += ((static_cast<float>(v175_data[0])) * v30_data);
              v173_acc += ((static_cast<float>(v175_data[1])) * v31_data);
              v173_acc += ((static_cast<float>(v175_data[2])) * v32_data);
              v173_acc += ((static_cast<float>(v175_data[3])) * v33_data);
              v173_acc += ((static_cast<float>(v175_data[4])) * v34_data);
              v173_acc += ((static_cast<float>(v175_data[5])) * v35_data);
              v173_acc += ((static_cast<float>(v175_data[6])) * v36_data);
              v173_acc += ((static_cast<float>(v175_data[7])) * v37_data);
              ir1.template select<16, 1>(112) = v173_acc;
              #pragma unroll
              for (int32_t v192_n1 = 0; v192_n1 < 8; ++v192_n1) {
                int32_t v193_a = v192_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v195_data(ir1.template select<8, 1>(v193_a));
                r1.template select<8, 1>(v193_a) = v195_data;
              }
              // glb_m1 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v196_i1 = 0; v196_i1 < 8; ++v196_i1) {
                tensorforge::intel_esimd::simd<float, 8> v199_data(r1.template select<8, 1>((v196_i1 * 16)));
                v199_data.copy_to(glb_m1 + ((v196_i1 * 8)));
              }
            }
            tensorforge::prefetchL2<64>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}


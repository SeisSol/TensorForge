// === base name ===
kernel_c989b4f1189b13a5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c989b4f1189b13a5 = {{1, 16, 1}, 16, 9, 1, 16, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c989b4f1189b13a5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c989b4f1189b13a5(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c989b4f1189b13a5(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1792 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_c989b4f1189b13a5(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c989b4f1189b13a5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c989b4f1189b13a5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c989b4f1189b13a5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1792 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (9 active) x 16 per block = block 1x16x1, 7168 B shared, occupancy grid
        // operands:
        //   m0 9×9(9×9) {0..9}×{0..9} strided
        //   m1 9×9(9×9) {0..9}×{0..9} strided
        //   m2 9×9(9×9) {0..9}×{0..9} strided
        //   m3 ()  scalar
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j] × m3[]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":9,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1792}],"shared_bytes":7168,"shared_elements":1792,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[0,1],[0,1],[]],"target":[[0,-1],[-1,1],[]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (112 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 81 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 9; ++v23_i1) {
                tensorforge::intel_esimd::simd<float, 9> v28_data;
                v28_data.copy_from(glb_m1 + ((v23_i1 * 9)));
                r0.template select<9, 1>((v23_i1 * 16)) = v28_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<float, 16> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 64), v32_ld);
              float v33_ld = glb_m2[0 + 0 + 1 * 0 + 80];
              s0[0 + 0 + 1 * 0 + 80] = v33_ld;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 9), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v45_acc{};
              tensorforge::intel_esimd::simd<float, 16> v49_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v45_acc += ((static_cast<float>(v49_data[0])) * v36_data);
              v45_acc += ((static_cast<float>(v49_data[1])) * v37_data);
              v45_acc += ((static_cast<float>(v49_data[2])) * v38_data);
              v45_acc += ((static_cast<float>(v49_data[3])) * v39_data);
              v45_acc += ((static_cast<float>(v49_data[4])) * v40_data);
              v45_acc += ((static_cast<float>(v49_data[5])) * v41_data);
              v45_acc += ((static_cast<float>(v49_data[6])) * v42_data);
              v45_acc += ((static_cast<float>(v49_data[7])) * v43_data);
              v45_acc += ((static_cast<float>(v49_data[8])) * v44_data);
              ir1.template select<16, 1>(0) = v45_acc;
              tensorforge::intel_esimd::simd<float, 16> v68_acc{};
              tensorforge::intel_esimd::simd<float, 16> v70_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              v68_acc += ((static_cast<float>(v70_data[0])) * v36_data);
              v68_acc += ((static_cast<float>(v70_data[1])) * v37_data);
              v68_acc += ((static_cast<float>(v70_data[2])) * v38_data);
              v68_acc += ((static_cast<float>(v70_data[3])) * v39_data);
              v68_acc += ((static_cast<float>(v70_data[4])) * v40_data);
              v68_acc += ((static_cast<float>(v70_data[5])) * v41_data);
              v68_acc += ((static_cast<float>(v70_data[6])) * v42_data);
              v68_acc += ((static_cast<float>(v70_data[7])) * v43_data);
              v68_acc += ((static_cast<float>(v70_data[8])) * v44_data);
              ir1.template select<16, 1>(16) = v68_acc;
              tensorforge::intel_esimd::simd<float, 16> v89_acc{};
              tensorforge::intel_esimd::simd<float, 16> v91_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              v89_acc += ((static_cast<float>(v91_data[0])) * v36_data);
              v89_acc += ((static_cast<float>(v91_data[1])) * v37_data);
              v89_acc += ((static_cast<float>(v91_data[2])) * v38_data);
              v89_acc += ((static_cast<float>(v91_data[3])) * v39_data);
              v89_acc += ((static_cast<float>(v91_data[4])) * v40_data);
              v89_acc += ((static_cast<float>(v91_data[5])) * v41_data);
              v89_acc += ((static_cast<float>(v91_data[6])) * v42_data);
              v89_acc += ((static_cast<float>(v91_data[7])) * v43_data);
              v89_acc += ((static_cast<float>(v91_data[8])) * v44_data);
              ir1.template select<16, 1>(32) = v89_acc;
              tensorforge::intel_esimd::simd<float, 16> v110_acc{};
              tensorforge::intel_esimd::simd<float, 16> v112_data = tensorforge::slmLoad<float, 16>(s0 + (27_i32));
              v110_acc += ((static_cast<float>(v112_data[0])) * v36_data);
              v110_acc += ((static_cast<float>(v112_data[1])) * v37_data);
              v110_acc += ((static_cast<float>(v112_data[2])) * v38_data);
              v110_acc += ((static_cast<float>(v112_data[3])) * v39_data);
              v110_acc += ((static_cast<float>(v112_data[4])) * v40_data);
              v110_acc += ((static_cast<float>(v112_data[5])) * v41_data);
              v110_acc += ((static_cast<float>(v112_data[6])) * v42_data);
              v110_acc += ((static_cast<float>(v112_data[7])) * v43_data);
              v110_acc += ((static_cast<float>(v112_data[8])) * v44_data);
              ir1.template select<16, 1>(48) = v110_acc;
              tensorforge::intel_esimd::simd<float, 16> v131_acc{};
              tensorforge::intel_esimd::simd<float, 16> v133_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v131_acc += ((static_cast<float>(v133_data[0])) * v36_data);
              v131_acc += ((static_cast<float>(v133_data[1])) * v37_data);
              v131_acc += ((static_cast<float>(v133_data[2])) * v38_data);
              v131_acc += ((static_cast<float>(v133_data[3])) * v39_data);
              v131_acc += ((static_cast<float>(v133_data[4])) * v40_data);
              v131_acc += ((static_cast<float>(v133_data[5])) * v41_data);
              v131_acc += ((static_cast<float>(v133_data[6])) * v42_data);
              v131_acc += ((static_cast<float>(v133_data[7])) * v43_data);
              v131_acc += ((static_cast<float>(v133_data[8])) * v44_data);
              ir1.template select<16, 1>(64) = v131_acc;
              tensorforge::intel_esimd::simd<float, 16> v152_acc{};
              tensorforge::intel_esimd::simd<float, 16> v154_data = tensorforge::slmLoad<float, 16>(s0 + (45_i32));
              v152_acc += ((static_cast<float>(v154_data[0])) * v36_data);
              v152_acc += ((static_cast<float>(v154_data[1])) * v37_data);
              v152_acc += ((static_cast<float>(v154_data[2])) * v38_data);
              v152_acc += ((static_cast<float>(v154_data[3])) * v39_data);
              v152_acc += ((static_cast<float>(v154_data[4])) * v40_data);
              v152_acc += ((static_cast<float>(v154_data[5])) * v41_data);
              v152_acc += ((static_cast<float>(v154_data[6])) * v42_data);
              v152_acc += ((static_cast<float>(v154_data[7])) * v43_data);
              v152_acc += ((static_cast<float>(v154_data[8])) * v44_data);
              ir1.template select<16, 1>(80) = v152_acc;
              tensorforge::intel_esimd::simd<float, 16> v173_acc{};
              tensorforge::intel_esimd::simd<float, 16> v175_data = tensorforge::slmLoad<float, 16>(s0 + (54_i32));
              v173_acc += ((static_cast<float>(v175_data[0])) * v36_data);
              v173_acc += ((static_cast<float>(v175_data[1])) * v37_data);
              v173_acc += ((static_cast<float>(v175_data[2])) * v38_data);
              v173_acc += ((static_cast<float>(v175_data[3])) * v39_data);
              v173_acc += ((static_cast<float>(v175_data[4])) * v40_data);
              v173_acc += ((static_cast<float>(v175_data[5])) * v41_data);
              v173_acc += ((static_cast<float>(v175_data[6])) * v42_data);
              v173_acc += ((static_cast<float>(v175_data[7])) * v43_data);
              v173_acc += ((static_cast<float>(v175_data[8])) * v44_data);
              ir1.template select<16, 1>(96) = v173_acc;
              tensorforge::intel_esimd::simd<float, 16> v194_acc{};
              tensorforge::intel_esimd::simd<float, 16> v196_data = tensorforge::slmLoad<float, 16>(s0 + (63_i32));
              v194_acc += ((static_cast<float>(v196_data[0])) * v36_data);
              v194_acc += ((static_cast<float>(v196_data[1])) * v37_data);
              v194_acc += ((static_cast<float>(v196_data[2])) * v38_data);
              v194_acc += ((static_cast<float>(v196_data[3])) * v39_data);
              v194_acc += ((static_cast<float>(v196_data[4])) * v40_data);
              v194_acc += ((static_cast<float>(v196_data[5])) * v41_data);
              v194_acc += ((static_cast<float>(v196_data[6])) * v42_data);
              v194_acc += ((static_cast<float>(v196_data[7])) * v43_data);
              v194_acc += ((static_cast<float>(v196_data[8])) * v44_data);
              ir1.template select<16, 1>(112) = v194_acc;
              tensorforge::intel_esimd::simd<float, 16> v215_acc{};
              tensorforge::intel_esimd::simd<float, 16> v217_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v215_acc += ((static_cast<float>(v217_data[0])) * v36_data);
              v215_acc += ((static_cast<float>(v217_data[1])) * v37_data);
              v215_acc += ((static_cast<float>(v217_data[2])) * v38_data);
              v215_acc += ((static_cast<float>(v217_data[3])) * v39_data);
              v215_acc += ((static_cast<float>(v217_data[4])) * v40_data);
              v215_acc += ((static_cast<float>(v217_data[5])) * v41_data);
              v215_acc += ((static_cast<float>(v217_data[6])) * v42_data);
              v215_acc += ((static_cast<float>(v217_data[7])) * v43_data);
              v215_acc += ((static_cast<float>(v217_data[8])) * v44_data);
              ir1.template select<16, 1>(128) = v215_acc;
              // r1 = ir1 * glb_m3
              #pragma unroll
              for (int32_t v237_n1 = 0; v237_n1 < 9; ++v237_n1) {
                int32_t v238_a = v237_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v240_data(ir1.template select<9, 1>(v238_a));
                r1.template select<9, 1>(v238_a) = (v240_data * 13.0f);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v242_i1 = 0; v242_i1 < 9; ++v242_i1) {
                tensorforge::intel_esimd::simd<float, 9> v245_data(r1.template select<9, 1>((v242_i1 * 16)));
                v245_data.copy_to(glb_m0 + ((v242_i1 * 9)));
              }
            }
          }
        }
      }
    });
  });
}


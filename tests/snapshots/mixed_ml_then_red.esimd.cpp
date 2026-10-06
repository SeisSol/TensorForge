// === base name ===
kernel_8e5b2fec1c73f8fb

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8e5b2fec1c73f8fb = {{1, 16, 1}, 16, 16, 1, 16, 5120, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8e5b2fec1c73f8fb(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8e5b2fec1c73f8fb(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8e5b2fec1c73f8fb(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_8e5b2fec1c73f8fb(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8e5b2fec1c73f8fb(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8e5b2fec1c73f8fb(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8e5b2fec1c73f8fb(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1280 * sizeof(float)>(); {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1280}],"shared_bytes":5120,"shared_elements":1280,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[8]],"name":"m2","ordered":false,"parts":1,"shape":[8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[8]],"is_tmp":false,"name":"m2","offset":[0],"shape":[8]},"kind":"reduction","op":"+","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (80 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v12_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v12_batchId0 < numElements0; v12_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 8 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 128> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
                tensorforge::intel_esimd::simd<float, 8> v29_data;
                v29_data.copy_from(glb_m0 + ((v24_i1 * 8)));
                r0.template select<8, 1>((v24_i1 * 16)) = v29_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v32_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v42_acc{};
              tensorforge::intel_esimd::simd<float, 16> v46_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v42_acc += ((static_cast<float>(v46_data[0])) * v34_data);
              v42_acc += ((static_cast<float>(v46_data[1])) * v35_data);
              v42_acc += ((static_cast<float>(v46_data[2])) * v36_data);
              v42_acc += ((static_cast<float>(v46_data[3])) * v37_data);
              v42_acc += ((static_cast<float>(v46_data[4])) * v38_data);
              v42_acc += ((static_cast<float>(v46_data[5])) * v39_data);
              v42_acc += ((static_cast<float>(v46_data[6])) * v40_data);
              v42_acc += ((static_cast<float>(v46_data[7])) * v41_data);
              r1.template select<16, 1>(0) = v42_acc;
              tensorforge::intel_esimd::simd<float, 16> v63_acc{};
              tensorforge::intel_esimd::simd<float, 16> v65_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              v63_acc += ((static_cast<float>(v65_data[0])) * v34_data);
              v63_acc += ((static_cast<float>(v65_data[1])) * v35_data);
              v63_acc += ((static_cast<float>(v65_data[2])) * v36_data);
              v63_acc += ((static_cast<float>(v65_data[3])) * v37_data);
              v63_acc += ((static_cast<float>(v65_data[4])) * v38_data);
              v63_acc += ((static_cast<float>(v65_data[5])) * v39_data);
              v63_acc += ((static_cast<float>(v65_data[6])) * v40_data);
              v63_acc += ((static_cast<float>(v65_data[7])) * v41_data);
              r1.template select<16, 1>(16) = v63_acc;
              tensorforge::intel_esimd::simd<float, 16> v82_acc{};
              tensorforge::intel_esimd::simd<float, 16> v84_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v82_acc += ((static_cast<float>(v84_data[0])) * v34_data);
              v82_acc += ((static_cast<float>(v84_data[1])) * v35_data);
              v82_acc += ((static_cast<float>(v84_data[2])) * v36_data);
              v82_acc += ((static_cast<float>(v84_data[3])) * v37_data);
              v82_acc += ((static_cast<float>(v84_data[4])) * v38_data);
              v82_acc += ((static_cast<float>(v84_data[5])) * v39_data);
              v82_acc += ((static_cast<float>(v84_data[6])) * v40_data);
              v82_acc += ((static_cast<float>(v84_data[7])) * v41_data);
              r1.template select<16, 1>(32) = v82_acc;
              tensorforge::intel_esimd::simd<float, 16> v101_acc{};
              tensorforge::intel_esimd::simd<float, 16> v103_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v101_acc += ((static_cast<float>(v103_data[0])) * v34_data);
              v101_acc += ((static_cast<float>(v103_data[1])) * v35_data);
              v101_acc += ((static_cast<float>(v103_data[2])) * v36_data);
              v101_acc += ((static_cast<float>(v103_data[3])) * v37_data);
              v101_acc += ((static_cast<float>(v103_data[4])) * v38_data);
              v101_acc += ((static_cast<float>(v103_data[5])) * v39_data);
              v101_acc += ((static_cast<float>(v103_data[6])) * v40_data);
              v101_acc += ((static_cast<float>(v103_data[7])) * v41_data);
              r1.template select<16, 1>(48) = v101_acc;
              tensorforge::intel_esimd::simd<float, 16> v120_acc{};
              tensorforge::intel_esimd::simd<float, 16> v122_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v120_acc += ((static_cast<float>(v122_data[0])) * v34_data);
              v120_acc += ((static_cast<float>(v122_data[1])) * v35_data);
              v120_acc += ((static_cast<float>(v122_data[2])) * v36_data);
              v120_acc += ((static_cast<float>(v122_data[3])) * v37_data);
              v120_acc += ((static_cast<float>(v122_data[4])) * v38_data);
              v120_acc += ((static_cast<float>(v122_data[5])) * v39_data);
              v120_acc += ((static_cast<float>(v122_data[6])) * v40_data);
              v120_acc += ((static_cast<float>(v122_data[7])) * v41_data);
              r1.template select<16, 1>(64) = v120_acc;
              tensorforge::intel_esimd::simd<float, 16> v139_acc{};
              tensorforge::intel_esimd::simd<float, 16> v141_data = tensorforge::slmLoad<float, 16>(s0 + (40_i32));
              v139_acc += ((static_cast<float>(v141_data[0])) * v34_data);
              v139_acc += ((static_cast<float>(v141_data[1])) * v35_data);
              v139_acc += ((static_cast<float>(v141_data[2])) * v36_data);
              v139_acc += ((static_cast<float>(v141_data[3])) * v37_data);
              v139_acc += ((static_cast<float>(v141_data[4])) * v38_data);
              v139_acc += ((static_cast<float>(v141_data[5])) * v39_data);
              v139_acc += ((static_cast<float>(v141_data[6])) * v40_data);
              v139_acc += ((static_cast<float>(v141_data[7])) * v41_data);
              r1.template select<16, 1>(80) = v139_acc;
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v158_acc += ((static_cast<float>(v160_data[0])) * v34_data);
              v158_acc += ((static_cast<float>(v160_data[1])) * v35_data);
              v158_acc += ((static_cast<float>(v160_data[2])) * v36_data);
              v158_acc += ((static_cast<float>(v160_data[3])) * v37_data);
              v158_acc += ((static_cast<float>(v160_data[4])) * v38_data);
              v158_acc += ((static_cast<float>(v160_data[5])) * v39_data);
              v158_acc += ((static_cast<float>(v160_data[6])) * v40_data);
              v158_acc += ((static_cast<float>(v160_data[7])) * v41_data);
              r1.template select<16, 1>(96) = v158_acc;
              tensorforge::intel_esimd::simd<float, 16> v177_acc{};
              tensorforge::intel_esimd::simd<float, 16> v179_data = tensorforge::slmLoad<float, 16>(s0 + (56_i32));
              v177_acc += ((static_cast<float>(v179_data[0])) * v34_data);
              v177_acc += ((static_cast<float>(v179_data[1])) * v35_data);
              v177_acc += ((static_cast<float>(v179_data[2])) * v36_data);
              v177_acc += ((static_cast<float>(v179_data[3])) * v37_data);
              v177_acc += ((static_cast<float>(v179_data[4])) * v38_data);
              v177_acc += ((static_cast<float>(v179_data[5])) * v39_data);
              v177_acc += ((static_cast<float>(v179_data[6])) * v40_data);
              v177_acc += ((static_cast<float>(v179_data[7])) * v41_data);
              r1.template select<16, 1>(112) = v177_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v196_i1 = 0; v196_i1 < 8; ++v196_i1) {
                tensorforge::intel_esimd::simd<float, 8> v199_data(r1.template select<8, 1>((v196_i1 * 16)));
                tensorforge::slmStore<float, 8>(s1 + ((v196_i1 * 8)), v199_data);
              }
              // glb_m2 = +(s1, dims=[1])
              tensorforge::intel_esimd::simd<float, 8> v205_acc0(0.0f);
              #pragma unroll
              for (int32_t v204_r1 = 0; v204_r1 < 8; ++v204_r1) {
                tensorforge::intel_esimd::simd<float, 8> v210_data = tensorforge::slmLoad<float, 8>(s1 + ((v204_r1 * 8)));
                v205_acc0 = (v205_acc0 + v210_data);
              }
              v205_acc0.copy_to(glb_m2 + (0_i32));
            }
          }
        }
      }
    });
  });
}


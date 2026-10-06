// === base name ===
kernel_f2ebbc45ecc197e2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f2ebbc45ecc197e2 = {{1, 16, 1}, 16, 16, 1, 16, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f2ebbc45ecc197e2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f2ebbc45ecc197e2(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f2ebbc45ecc197e2(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1024 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f2ebbc45ecc197e2(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f2ebbc45ecc197e2(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f2ebbc45ecc197e2(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f2ebbc45ecc197e2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1024 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 4096 B shared, occupancy grid
        // operands:
        //   m0 40(40) {0..40} strided
        //   m1 40(40) {0..40} strided
        // operations:
        //   P = abs(A)
        //   M = max(P, dims=[0])
        //   C = mul(A, M)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1024}],"shared_bytes":4096,"shared_elements":1024,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0],[40]],"name":"m0","ordered":false,"parts":1,"shape":[40],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0],[40]],"name":"m1","ordered":false,"parts":1,"shape":[40],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[40]],"is_tmp":true,"name":"t0","offset":[0],"shape":[40]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0],[40]],"is_tmp":false,"name":"m0","offset":[0],"shape":[40]}],"permute":[[0]],"scalars":[],"target":[[0]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[],[]],"is_tmp":true,"name":"t1","offset":[],"shape":[]},"kind":"reduction","op":"max","ops":[{"addressing":"pointer_based","bbox":[[0],[40]],"is_tmp":true,"name":"t0","offset":[0],"shape":[40]}],"permute":[[0]],"target":[[-1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[40]],"is_tmp":false,"name":"m1","offset":[0],"shape":[40]},"kind":"elementwise","op":"MUL","ops":[{"addressing":"strided","bbox":[[0],[40]],"is_tmp":false,"name":"m0","offset":[0],"shape":[40]},{"addressing":"pointer_based","bbox":[[],[]],"is_tmp":true,"name":"t1","offset":[],"shape":[]}],"permute":[[0],[]],"scalars":[],"target":[[0],[]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (64 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (48);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 40 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 40 + 0 + m1_extraOffset];
              float r0[48]{};
              // r0 = abs(glb_m0)
              #pragma unroll
              for (int32_t v22_k0 = 0; v22_k0 < 2; ++v22_k0) {
                int32_t v23_lead = v22_k0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v25_data;
                v25_data.copy_from(glb_m0 + (v23_lead));
                (tensorforge::intel_esimd::abs(v25_data)).copy_to(r0 + (v23_lead));
              }
              tensorforge::intel_esimd::simd<float, 8> v30_data;
              v30_data.copy_from(glb_m0 + (32_i32));
              (tensorforge::intel_esimd::abs(v30_data)).copy_to(r0 + (32));
              // s0 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 2; ++v32_i0) {
                int32_t v33_a = v32_i0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v34_data;
                v34_data.copy_from(r0 + (v33_a));
                tensorforge::slmStore<float, 16>(s0 + (v33_a), v34_data);
              }
              tensorforge::intel_esimd::simd<float, 8> v36_data;
              v36_data.copy_from(r0 + (32));
              tensorforge::slmStore<float, 8>(s0 + (32_i32), v36_data);
              float r1[1]{};
              // r1 = max(s0, dims=[0])
              tensorforge::intel_esimd::simd<float, 48> s0_run0 = tensorforge::slmLoad<float, 48>(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v44_data(s0_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v47_data(s0_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v52_data(s0_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v54_own(-INFINITY);
              v54_own.merge(tensorforge::intel_esimd::simd<float, 16>(v52_data), ((tensorforge::intel_esimd::simd<int32_t, 16>(0, 1)) < 8));
              r1[0] = (tensorforge::intel_esimd::hmax<float>((tensorforge::intel_esimd::max((tensorforge::intel_esimd::max(v44_data, v47_data)), v54_own))));
              // glb_m1 = mul(glb_m0, r1)
              float v61_data = r1[0];
              #pragma unroll
              for (int32_t v57_k0 = 0; v57_k0 < 2; ++v57_k0) {
                int32_t v58_lead = v57_k0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v60_data;
                v60_data.copy_from(glb_m0 + (v58_lead));
                ((v60_data * v61_data)).copy_to(glb_m1 + (v58_lead));
              }
              tensorforge::intel_esimd::simd<float, 8> v63_data;
              v63_data.copy_from(glb_m0 + (32_i32));
              float v64_data = r1[0];
              ((v63_data * v64_data)).copy_to(glb_m1 + (32_i32));
            }
          }
        }
      }
    });
  });
}


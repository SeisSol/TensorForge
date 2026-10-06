// === base name ===
kernel_14f5cec9161bdecf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_14f5cec9161bdecf = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_14f5cec9161bdecf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_14f5cec9161bdecf(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_14f5cec9161bdecf(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_14f5cec9161bdecf(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_14f5cec9161bdecf(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_14f5cec9161bdecf(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_14f5cec9161bdecf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(4×12) {4..8}×{0..12} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   D = abs(N)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (144);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v11_batchId0 * 48 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                tensorforge::intel_esimd::simd<float, 12> v29_data;
                v29_data.copy_from(glb_m1 + ((v24_i1 * 12)));
                r0.template select<12, 1>((v24_i1 * 16)) = v29_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v32_ld);
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v33_ld);
              tensorforge::intel_esimd::simd<float, 16> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v34_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v49_acc{};
              tensorforge::intel_esimd::simd<float, 16> v53_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v49_acc += ((static_cast<float>(v53_data[0])) * v37_data);
              v49_acc += ((static_cast<float>(v53_data[1])) * v38_data);
              v49_acc += ((static_cast<float>(v53_data[2])) * v39_data);
              v49_acc += ((static_cast<float>(v53_data[3])) * v40_data);
              v49_acc += ((static_cast<float>(v53_data[4])) * v41_data);
              v49_acc += ((static_cast<float>(v53_data[5])) * v42_data);
              v49_acc += ((static_cast<float>(v53_data[6])) * v43_data);
              v49_acc += ((static_cast<float>(v53_data[7])) * v44_data);
              v49_acc += ((static_cast<float>(v53_data[8])) * v45_data);
              v49_acc += ((static_cast<float>(v53_data[9])) * v46_data);
              v49_acc += ((static_cast<float>(v53_data[10])) * v47_data);
              v49_acc += ((static_cast<float>(v53_data[11])) * v48_data);
              ir1.template select<16, 1>(0) = v49_acc;
              tensorforge::intel_esimd::simd<float, 16> v78_acc{};
              tensorforge::intel_esimd::simd<float, 16> v80_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v78_acc += ((static_cast<float>(v80_data[0])) * v37_data);
              v78_acc += ((static_cast<float>(v80_data[1])) * v38_data);
              v78_acc += ((static_cast<float>(v80_data[2])) * v39_data);
              v78_acc += ((static_cast<float>(v80_data[3])) * v40_data);
              v78_acc += ((static_cast<float>(v80_data[4])) * v41_data);
              v78_acc += ((static_cast<float>(v80_data[5])) * v42_data);
              v78_acc += ((static_cast<float>(v80_data[6])) * v43_data);
              v78_acc += ((static_cast<float>(v80_data[7])) * v44_data);
              v78_acc += ((static_cast<float>(v80_data[8])) * v45_data);
              v78_acc += ((static_cast<float>(v80_data[9])) * v46_data);
              v78_acc += ((static_cast<float>(v80_data[10])) * v47_data);
              v78_acc += ((static_cast<float>(v80_data[11])) * v48_data);
              ir1.template select<16, 1>(16) = v78_acc;
              tensorforge::intel_esimd::simd<float, 16> v105_acc{};
              tensorforge::intel_esimd::simd<float, 16> v107_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v105_acc += ((static_cast<float>(v107_data[0])) * v37_data);
              v105_acc += ((static_cast<float>(v107_data[1])) * v38_data);
              v105_acc += ((static_cast<float>(v107_data[2])) * v39_data);
              v105_acc += ((static_cast<float>(v107_data[3])) * v40_data);
              v105_acc += ((static_cast<float>(v107_data[4])) * v41_data);
              v105_acc += ((static_cast<float>(v107_data[5])) * v42_data);
              v105_acc += ((static_cast<float>(v107_data[6])) * v43_data);
              v105_acc += ((static_cast<float>(v107_data[7])) * v44_data);
              v105_acc += ((static_cast<float>(v107_data[8])) * v45_data);
              v105_acc += ((static_cast<float>(v107_data[9])) * v46_data);
              v105_acc += ((static_cast<float>(v107_data[10])) * v47_data);
              v105_acc += ((static_cast<float>(v107_data[11])) * v48_data);
              ir1.template select<16, 1>(32) = v105_acc;
              tensorforge::intel_esimd::simd<float, 16> v132_acc{};
              tensorforge::intel_esimd::simd<float, 16> v134_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v132_acc += ((static_cast<float>(v134_data[0])) * v37_data);
              v132_acc += ((static_cast<float>(v134_data[1])) * v38_data);
              v132_acc += ((static_cast<float>(v134_data[2])) * v39_data);
              v132_acc += ((static_cast<float>(v134_data[3])) * v40_data);
              v132_acc += ((static_cast<float>(v134_data[4])) * v41_data);
              v132_acc += ((static_cast<float>(v134_data[5])) * v42_data);
              v132_acc += ((static_cast<float>(v134_data[6])) * v43_data);
              v132_acc += ((static_cast<float>(v134_data[7])) * v44_data);
              v132_acc += ((static_cast<float>(v134_data[8])) * v45_data);
              v132_acc += ((static_cast<float>(v134_data[9])) * v46_data);
              v132_acc += ((static_cast<float>(v134_data[10])) * v47_data);
              v132_acc += ((static_cast<float>(v134_data[11])) * v48_data);
              ir1.template select<16, 1>(48) = v132_acc;
              tensorforge::intel_esimd::simd<float, 16> v159_acc{};
              tensorforge::intel_esimd::simd<float, 16> v161_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v159_acc += ((static_cast<float>(v161_data[0])) * v37_data);
              v159_acc += ((static_cast<float>(v161_data[1])) * v38_data);
              v159_acc += ((static_cast<float>(v161_data[2])) * v39_data);
              v159_acc += ((static_cast<float>(v161_data[3])) * v40_data);
              v159_acc += ((static_cast<float>(v161_data[4])) * v41_data);
              v159_acc += ((static_cast<float>(v161_data[5])) * v42_data);
              v159_acc += ((static_cast<float>(v161_data[6])) * v43_data);
              v159_acc += ((static_cast<float>(v161_data[7])) * v44_data);
              v159_acc += ((static_cast<float>(v161_data[8])) * v45_data);
              v159_acc += ((static_cast<float>(v161_data[9])) * v46_data);
              v159_acc += ((static_cast<float>(v161_data[10])) * v47_data);
              v159_acc += ((static_cast<float>(v161_data[11])) * v48_data);
              ir1.template select<16, 1>(64) = v159_acc;
              tensorforge::intel_esimd::simd<float, 16> v186_acc{};
              tensorforge::intel_esimd::simd<float, 16> v188_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v186_acc += ((static_cast<float>(v188_data[0])) * v37_data);
              v186_acc += ((static_cast<float>(v188_data[1])) * v38_data);
              v186_acc += ((static_cast<float>(v188_data[2])) * v39_data);
              v186_acc += ((static_cast<float>(v188_data[3])) * v40_data);
              v186_acc += ((static_cast<float>(v188_data[4])) * v41_data);
              v186_acc += ((static_cast<float>(v188_data[5])) * v42_data);
              v186_acc += ((static_cast<float>(v188_data[6])) * v43_data);
              v186_acc += ((static_cast<float>(v188_data[7])) * v44_data);
              v186_acc += ((static_cast<float>(v188_data[8])) * v45_data);
              v186_acc += ((static_cast<float>(v188_data[9])) * v46_data);
              v186_acc += ((static_cast<float>(v188_data[10])) * v47_data);
              v186_acc += ((static_cast<float>(v188_data[11])) * v48_data);
              ir1.template select<16, 1>(80) = v186_acc;
              tensorforge::intel_esimd::simd<float, 16> v213_acc{};
              tensorforge::intel_esimd::simd<float, 16> v215_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v213_acc += ((static_cast<float>(v215_data[0])) * v37_data);
              v213_acc += ((static_cast<float>(v215_data[1])) * v38_data);
              v213_acc += ((static_cast<float>(v215_data[2])) * v39_data);
              v213_acc += ((static_cast<float>(v215_data[3])) * v40_data);
              v213_acc += ((static_cast<float>(v215_data[4])) * v41_data);
              v213_acc += ((static_cast<float>(v215_data[5])) * v42_data);
              v213_acc += ((static_cast<float>(v215_data[6])) * v43_data);
              v213_acc += ((static_cast<float>(v215_data[7])) * v44_data);
              v213_acc += ((static_cast<float>(v215_data[8])) * v45_data);
              v213_acc += ((static_cast<float>(v215_data[9])) * v46_data);
              v213_acc += ((static_cast<float>(v215_data[10])) * v47_data);
              v213_acc += ((static_cast<float>(v215_data[11])) * v48_data);
              ir1.template select<16, 1>(96) = v213_acc;
              tensorforge::intel_esimd::simd<float, 16> v240_acc{};
              tensorforge::intel_esimd::simd<float, 16> v242_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v240_acc += ((static_cast<float>(v242_data[0])) * v37_data);
              v240_acc += ((static_cast<float>(v242_data[1])) * v38_data);
              v240_acc += ((static_cast<float>(v242_data[2])) * v39_data);
              v240_acc += ((static_cast<float>(v242_data[3])) * v40_data);
              v240_acc += ((static_cast<float>(v242_data[4])) * v41_data);
              v240_acc += ((static_cast<float>(v242_data[5])) * v42_data);
              v240_acc += ((static_cast<float>(v242_data[6])) * v43_data);
              v240_acc += ((static_cast<float>(v242_data[7])) * v44_data);
              v240_acc += ((static_cast<float>(v242_data[8])) * v45_data);
              v240_acc += ((static_cast<float>(v242_data[9])) * v46_data);
              v240_acc += ((static_cast<float>(v242_data[10])) * v47_data);
              v240_acc += ((static_cast<float>(v242_data[11])) * v48_data);
              ir1.template select<16, 1>(112) = v240_acc;
              tensorforge::intel_esimd::simd<float, 16> v267_acc{};
              tensorforge::intel_esimd::simd<float, 16> v269_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v267_acc += ((static_cast<float>(v269_data[0])) * v37_data);
              v267_acc += ((static_cast<float>(v269_data[1])) * v38_data);
              v267_acc += ((static_cast<float>(v269_data[2])) * v39_data);
              v267_acc += ((static_cast<float>(v269_data[3])) * v40_data);
              v267_acc += ((static_cast<float>(v269_data[4])) * v41_data);
              v267_acc += ((static_cast<float>(v269_data[5])) * v42_data);
              v267_acc += ((static_cast<float>(v269_data[6])) * v43_data);
              v267_acc += ((static_cast<float>(v269_data[7])) * v44_data);
              v267_acc += ((static_cast<float>(v269_data[8])) * v45_data);
              v267_acc += ((static_cast<float>(v269_data[9])) * v46_data);
              v267_acc += ((static_cast<float>(v269_data[10])) * v47_data);
              v267_acc += ((static_cast<float>(v269_data[11])) * v48_data);
              ir1.template select<16, 1>(128) = v267_acc;
              tensorforge::intel_esimd::simd<float, 16> v294_acc{};
              tensorforge::intel_esimd::simd<float, 16> v296_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              v294_acc += ((static_cast<float>(v296_data[0])) * v37_data);
              v294_acc += ((static_cast<float>(v296_data[1])) * v38_data);
              v294_acc += ((static_cast<float>(v296_data[2])) * v39_data);
              v294_acc += ((static_cast<float>(v296_data[3])) * v40_data);
              v294_acc += ((static_cast<float>(v296_data[4])) * v41_data);
              v294_acc += ((static_cast<float>(v296_data[5])) * v42_data);
              v294_acc += ((static_cast<float>(v296_data[6])) * v43_data);
              v294_acc += ((static_cast<float>(v296_data[7])) * v44_data);
              v294_acc += ((static_cast<float>(v296_data[8])) * v45_data);
              v294_acc += ((static_cast<float>(v296_data[9])) * v46_data);
              v294_acc += ((static_cast<float>(v296_data[10])) * v47_data);
              v294_acc += ((static_cast<float>(v296_data[11])) * v48_data);
              ir1.template select<16, 1>(144) = v294_acc;
              tensorforge::intel_esimd::simd<float, 16> v321_acc{};
              tensorforge::intel_esimd::simd<float, 16> v323_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              v321_acc += ((static_cast<float>(v323_data[0])) * v37_data);
              v321_acc += ((static_cast<float>(v323_data[1])) * v38_data);
              v321_acc += ((static_cast<float>(v323_data[2])) * v39_data);
              v321_acc += ((static_cast<float>(v323_data[3])) * v40_data);
              v321_acc += ((static_cast<float>(v323_data[4])) * v41_data);
              v321_acc += ((static_cast<float>(v323_data[5])) * v42_data);
              v321_acc += ((static_cast<float>(v323_data[6])) * v43_data);
              v321_acc += ((static_cast<float>(v323_data[7])) * v44_data);
              v321_acc += ((static_cast<float>(v323_data[8])) * v45_data);
              v321_acc += ((static_cast<float>(v323_data[9])) * v46_data);
              v321_acc += ((static_cast<float>(v323_data[10])) * v47_data);
              v321_acc += ((static_cast<float>(v323_data[11])) * v48_data);
              ir1.template select<16, 1>(160) = v321_acc;
              tensorforge::intel_esimd::simd<float, 16> v348_acc{};
              tensorforge::intel_esimd::simd<float, 16> v350_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              v348_acc += ((static_cast<float>(v350_data[0])) * v37_data);
              v348_acc += ((static_cast<float>(v350_data[1])) * v38_data);
              v348_acc += ((static_cast<float>(v350_data[2])) * v39_data);
              v348_acc += ((static_cast<float>(v350_data[3])) * v40_data);
              v348_acc += ((static_cast<float>(v350_data[4])) * v41_data);
              v348_acc += ((static_cast<float>(v350_data[5])) * v42_data);
              v348_acc += ((static_cast<float>(v350_data[6])) * v43_data);
              v348_acc += ((static_cast<float>(v350_data[7])) * v44_data);
              v348_acc += ((static_cast<float>(v350_data[8])) * v45_data);
              v348_acc += ((static_cast<float>(v350_data[9])) * v46_data);
              v348_acc += ((static_cast<float>(v350_data[10])) * v47_data);
              v348_acc += ((static_cast<float>(v350_data[11])) * v48_data);
              ir1.template select<16, 1>(176) = v348_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v375_n1 = 0; v375_n1 < 12; ++v375_n1) {
                int32_t v376_a = v375_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v378_data(ir1.template select<12, 1>(v376_a));
                r1.template select<12, 1>(v376_a) = v378_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v379_i1 = 0; v379_i1 < 12; ++v379_i1) {
                tensorforge::intel_esimd::simd<float, 12> v382_data(r1.template select<12, 1>((v379_i1 * 16)));
                v382_data.copy_to(glb_m0 + ((v379_i1 * 12)));
              }
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = abs(glb_m3)
              #pragma unroll
              for (int32_t v388_k1 = 0; v388_k1 < 12; ++v388_k1) {
                tensorforge::intel_esimd::simd<float, 4> v395_data;
                v395_data.copy_from(glb_m3 + ((v388_k1 * 4)));
                r2.template select<4, 1>((v388_k1 * 16)) = (tensorforge::intel_esimd::abs(v395_data));
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v399_i1 = 0; v399_i1 < 12; ++v399_i1) {
                tensorforge::intel_esimd::simd<float, 4> v402_data(r2.template select<4, 1>((v399_i1 * 16)));
                v402_data.copy_to(glb_m0 + ((4_i32 + (v399_i1 * 12))));
              }
              #pragma unroll
              for (int32_t v408_z1 = 0; v408_z1 < 12; ++v408_z1) {
                glb_m0[(v408_z1 * 12)] = 0.0f;
              }
              #pragma unroll
              for (int32_t v414_z1 = 0; v414_z1 < 12; ++v414_z1) {
                glb_m0[(8_i32 + (v414_z1 * 12))] = 0.0f;
              }
            }
          }
        }
      }
    });
  });
}


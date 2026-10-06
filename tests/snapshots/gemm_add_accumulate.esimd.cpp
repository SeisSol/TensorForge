// === base name ===
kernel_7a53329855962b11

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7a53329855962b11 = {{1, 16, 1}, 16, 12, 1, 16, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7a53329855962b11(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7a53329855962b11(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7a53329855962b11(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7a53329855962b11(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7a53329855962b11(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7a53329855962b11(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7a53329855962b11(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×16(12×16) {0..12}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (144 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (128);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                tensorforge::intel_esimd::simd<float, 12> v28_data;
                v28_data.copy_from(glb_m1 + ((v23_i1 * 12)));
                r0.template select<12, 1>((v23_i1 * 16)) = v28_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v32_ld);
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
                tensorforge::intel_esimd::simd<float, 12> v39_data;
                v39_data.copy_from(glb_m0 + ((v34_i1 * 12)));
                r1.template select<12, 1>((v34_i1 * 16)) = v39_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 128> r2(0.0f);
              // ir2 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 128> ir2(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v59_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v60_acc{};
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v60_acc += ((static_cast<float>(v64_data[0])) * v44_data);
              v60_acc += ((static_cast<float>(v64_data[1])) * v45_data);
              v60_acc += ((static_cast<float>(v64_data[2])) * v46_data);
              v60_acc += ((static_cast<float>(v64_data[3])) * v47_data);
              v60_acc += ((static_cast<float>(v64_data[4])) * v48_data);
              v60_acc += ((static_cast<float>(v64_data[5])) * v49_data);
              v60_acc += ((static_cast<float>(v64_data[6])) * v50_data);
              v60_acc += ((static_cast<float>(v64_data[7])) * v51_data);
              v60_acc += ((static_cast<float>(v64_data[8])) * v52_data);
              v60_acc += ((static_cast<float>(v64_data[9])) * v53_data);
              v60_acc += ((static_cast<float>(v64_data[10])) * v54_data);
              v60_acc += ((static_cast<float>(v64_data[11])) * v55_data);
              v60_acc += ((static_cast<float>(v64_data[12])) * v56_data);
              v60_acc += ((static_cast<float>(v64_data[13])) * v57_data);
              v60_acc += ((static_cast<float>(v64_data[14])) * v58_data);
              v60_acc += ((static_cast<float>(v64_data[15])) * v59_data);
              ir2.template select<16, 1>(0) = v60_acc;
              tensorforge::intel_esimd::simd<float, 16> v97_acc{};
              tensorforge::intel_esimd::simd<float, 16> v99_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v97_acc += ((static_cast<float>(v99_data[0])) * v44_data);
              v97_acc += ((static_cast<float>(v99_data[1])) * v45_data);
              v97_acc += ((static_cast<float>(v99_data[2])) * v46_data);
              v97_acc += ((static_cast<float>(v99_data[3])) * v47_data);
              v97_acc += ((static_cast<float>(v99_data[4])) * v48_data);
              v97_acc += ((static_cast<float>(v99_data[5])) * v49_data);
              v97_acc += ((static_cast<float>(v99_data[6])) * v50_data);
              v97_acc += ((static_cast<float>(v99_data[7])) * v51_data);
              v97_acc += ((static_cast<float>(v99_data[8])) * v52_data);
              v97_acc += ((static_cast<float>(v99_data[9])) * v53_data);
              v97_acc += ((static_cast<float>(v99_data[10])) * v54_data);
              v97_acc += ((static_cast<float>(v99_data[11])) * v55_data);
              v97_acc += ((static_cast<float>(v99_data[12])) * v56_data);
              v97_acc += ((static_cast<float>(v99_data[13])) * v57_data);
              v97_acc += ((static_cast<float>(v99_data[14])) * v58_data);
              v97_acc += ((static_cast<float>(v99_data[15])) * v59_data);
              ir2.template select<16, 1>(16) = v97_acc;
              tensorforge::intel_esimd::simd<float, 16> v132_acc{};
              tensorforge::intel_esimd::simd<float, 16> v134_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v132_acc += ((static_cast<float>(v134_data[0])) * v44_data);
              v132_acc += ((static_cast<float>(v134_data[1])) * v45_data);
              v132_acc += ((static_cast<float>(v134_data[2])) * v46_data);
              v132_acc += ((static_cast<float>(v134_data[3])) * v47_data);
              v132_acc += ((static_cast<float>(v134_data[4])) * v48_data);
              v132_acc += ((static_cast<float>(v134_data[5])) * v49_data);
              v132_acc += ((static_cast<float>(v134_data[6])) * v50_data);
              v132_acc += ((static_cast<float>(v134_data[7])) * v51_data);
              v132_acc += ((static_cast<float>(v134_data[8])) * v52_data);
              v132_acc += ((static_cast<float>(v134_data[9])) * v53_data);
              v132_acc += ((static_cast<float>(v134_data[10])) * v54_data);
              v132_acc += ((static_cast<float>(v134_data[11])) * v55_data);
              v132_acc += ((static_cast<float>(v134_data[12])) * v56_data);
              v132_acc += ((static_cast<float>(v134_data[13])) * v57_data);
              v132_acc += ((static_cast<float>(v134_data[14])) * v58_data);
              v132_acc += ((static_cast<float>(v134_data[15])) * v59_data);
              ir2.template select<16, 1>(32) = v132_acc;
              tensorforge::intel_esimd::simd<float, 16> v167_acc{};
              tensorforge::intel_esimd::simd<float, 16> v169_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v167_acc += ((static_cast<float>(v169_data[0])) * v44_data);
              v167_acc += ((static_cast<float>(v169_data[1])) * v45_data);
              v167_acc += ((static_cast<float>(v169_data[2])) * v46_data);
              v167_acc += ((static_cast<float>(v169_data[3])) * v47_data);
              v167_acc += ((static_cast<float>(v169_data[4])) * v48_data);
              v167_acc += ((static_cast<float>(v169_data[5])) * v49_data);
              v167_acc += ((static_cast<float>(v169_data[6])) * v50_data);
              v167_acc += ((static_cast<float>(v169_data[7])) * v51_data);
              v167_acc += ((static_cast<float>(v169_data[8])) * v52_data);
              v167_acc += ((static_cast<float>(v169_data[9])) * v53_data);
              v167_acc += ((static_cast<float>(v169_data[10])) * v54_data);
              v167_acc += ((static_cast<float>(v169_data[11])) * v55_data);
              v167_acc += ((static_cast<float>(v169_data[12])) * v56_data);
              v167_acc += ((static_cast<float>(v169_data[13])) * v57_data);
              v167_acc += ((static_cast<float>(v169_data[14])) * v58_data);
              v167_acc += ((static_cast<float>(v169_data[15])) * v59_data);
              ir2.template select<16, 1>(48) = v167_acc;
              tensorforge::intel_esimd::simd<float, 16> v202_acc{};
              tensorforge::intel_esimd::simd<float, 16> v204_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v202_acc += ((static_cast<float>(v204_data[0])) * v44_data);
              v202_acc += ((static_cast<float>(v204_data[1])) * v45_data);
              v202_acc += ((static_cast<float>(v204_data[2])) * v46_data);
              v202_acc += ((static_cast<float>(v204_data[3])) * v47_data);
              v202_acc += ((static_cast<float>(v204_data[4])) * v48_data);
              v202_acc += ((static_cast<float>(v204_data[5])) * v49_data);
              v202_acc += ((static_cast<float>(v204_data[6])) * v50_data);
              v202_acc += ((static_cast<float>(v204_data[7])) * v51_data);
              v202_acc += ((static_cast<float>(v204_data[8])) * v52_data);
              v202_acc += ((static_cast<float>(v204_data[9])) * v53_data);
              v202_acc += ((static_cast<float>(v204_data[10])) * v54_data);
              v202_acc += ((static_cast<float>(v204_data[11])) * v55_data);
              v202_acc += ((static_cast<float>(v204_data[12])) * v56_data);
              v202_acc += ((static_cast<float>(v204_data[13])) * v57_data);
              v202_acc += ((static_cast<float>(v204_data[14])) * v58_data);
              v202_acc += ((static_cast<float>(v204_data[15])) * v59_data);
              ir2.template select<16, 1>(64) = v202_acc;
              tensorforge::intel_esimd::simd<float, 16> v237_acc{};
              tensorforge::intel_esimd::simd<float, 16> v239_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v237_acc += ((static_cast<float>(v239_data[0])) * v44_data);
              v237_acc += ((static_cast<float>(v239_data[1])) * v45_data);
              v237_acc += ((static_cast<float>(v239_data[2])) * v46_data);
              v237_acc += ((static_cast<float>(v239_data[3])) * v47_data);
              v237_acc += ((static_cast<float>(v239_data[4])) * v48_data);
              v237_acc += ((static_cast<float>(v239_data[5])) * v49_data);
              v237_acc += ((static_cast<float>(v239_data[6])) * v50_data);
              v237_acc += ((static_cast<float>(v239_data[7])) * v51_data);
              v237_acc += ((static_cast<float>(v239_data[8])) * v52_data);
              v237_acc += ((static_cast<float>(v239_data[9])) * v53_data);
              v237_acc += ((static_cast<float>(v239_data[10])) * v54_data);
              v237_acc += ((static_cast<float>(v239_data[11])) * v55_data);
              v237_acc += ((static_cast<float>(v239_data[12])) * v56_data);
              v237_acc += ((static_cast<float>(v239_data[13])) * v57_data);
              v237_acc += ((static_cast<float>(v239_data[14])) * v58_data);
              v237_acc += ((static_cast<float>(v239_data[15])) * v59_data);
              ir2.template select<16, 1>(80) = v237_acc;
              tensorforge::intel_esimd::simd<float, 16> v272_acc{};
              tensorforge::intel_esimd::simd<float, 16> v274_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v272_acc += ((static_cast<float>(v274_data[0])) * v44_data);
              v272_acc += ((static_cast<float>(v274_data[1])) * v45_data);
              v272_acc += ((static_cast<float>(v274_data[2])) * v46_data);
              v272_acc += ((static_cast<float>(v274_data[3])) * v47_data);
              v272_acc += ((static_cast<float>(v274_data[4])) * v48_data);
              v272_acc += ((static_cast<float>(v274_data[5])) * v49_data);
              v272_acc += ((static_cast<float>(v274_data[6])) * v50_data);
              v272_acc += ((static_cast<float>(v274_data[7])) * v51_data);
              v272_acc += ((static_cast<float>(v274_data[8])) * v52_data);
              v272_acc += ((static_cast<float>(v274_data[9])) * v53_data);
              v272_acc += ((static_cast<float>(v274_data[10])) * v54_data);
              v272_acc += ((static_cast<float>(v274_data[11])) * v55_data);
              v272_acc += ((static_cast<float>(v274_data[12])) * v56_data);
              v272_acc += ((static_cast<float>(v274_data[13])) * v57_data);
              v272_acc += ((static_cast<float>(v274_data[14])) * v58_data);
              v272_acc += ((static_cast<float>(v274_data[15])) * v59_data);
              ir2.template select<16, 1>(96) = v272_acc;
              tensorforge::intel_esimd::simd<float, 16> v307_acc{};
              tensorforge::intel_esimd::simd<float, 16> v309_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v307_acc += ((static_cast<float>(v309_data[0])) * v44_data);
              v307_acc += ((static_cast<float>(v309_data[1])) * v45_data);
              v307_acc += ((static_cast<float>(v309_data[2])) * v46_data);
              v307_acc += ((static_cast<float>(v309_data[3])) * v47_data);
              v307_acc += ((static_cast<float>(v309_data[4])) * v48_data);
              v307_acc += ((static_cast<float>(v309_data[5])) * v49_data);
              v307_acc += ((static_cast<float>(v309_data[6])) * v50_data);
              v307_acc += ((static_cast<float>(v309_data[7])) * v51_data);
              v307_acc += ((static_cast<float>(v309_data[8])) * v52_data);
              v307_acc += ((static_cast<float>(v309_data[9])) * v53_data);
              v307_acc += ((static_cast<float>(v309_data[10])) * v54_data);
              v307_acc += ((static_cast<float>(v309_data[11])) * v55_data);
              v307_acc += ((static_cast<float>(v309_data[12])) * v56_data);
              v307_acc += ((static_cast<float>(v309_data[13])) * v57_data);
              v307_acc += ((static_cast<float>(v309_data[14])) * v58_data);
              v307_acc += ((static_cast<float>(v309_data[15])) * v59_data);
              ir2.template select<16, 1>(112) = v307_acc;
              // r2 = ir2 + r1
              #pragma unroll
              for (int32_t v342_n1 = 0; v342_n1 < 8; ++v342_n1) {
                int32_t v343_a = v342_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v345_data(ir2.template select<12, 1>(v343_a));
                tensorforge::intel_esimd::simd<float, 12> v346_data(r1.template select<12, 1>(v343_a));
                r2.template select<12, 1>(v343_a) = (v346_data + v345_data);
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v348_i1 = 0; v348_i1 < 8; ++v348_i1) {
                tensorforge::intel_esimd::simd<float, 12> v351_data(r2.template select<12, 1>((v348_i1 * 16)));
                v351_data.copy_to(glb_m0 + ((v348_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


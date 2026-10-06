// === base name ===
kernel_4ccaf099d03df400

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4ccaf099d03df400 = {{1, 16, 1}, 16, 12, 1, 16, 23552, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4ccaf099d03df400(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4ccaf099d03df400(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4ccaf099d03df400(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 5888 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_4ccaf099d03df400(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4ccaf099d03df400(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4ccaf099d03df400(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4ccaf099d03df400(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5888 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 23552 B shared, occupancy grid
        // operands:
        //   m0 12×16(12×16) {0..12}×{0..16} strided
        //   m1 12×20(12×20) {0..12}×{0..20} strided
        //   m2 16×20(16×20) {0..16}×{0..20} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[j,k]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":5888}],"shared_bytes":23552,"shared_elements":5888,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,20]],"name":"m2","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"strided","bbox":[[0,0],[16,20]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,20]}],"permute":[[0,1],[1,0]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (368 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (352);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 320 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 320> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 20; ++v23_i1) {
                tensorforge::intel_esimd::simd<float, 12> v28_data;
                v28_data.copy_from(glb_m1 + ((v23_i1 * 12)));
                r0.template select<12, 1>((v23_i1 * 16)) = v28_data;
              }
              // s0 = load{g>s}(glb_m2[1, 0])
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
                int32_t v33_lead = v31_i0 * 16;
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 20; ++v32_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v37_data;
                  v37_data.copy_from(glb_m2 + ((v33_lead + (v32_i1 * 16))));
                  tensorforge::slmStore<float, 16>(s0 + ((v33_lead + (v32_i1 * 17))), v37_data);
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[1, 0]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 16)] [(0, 20)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(256));
              tensorforge::intel_esimd::simd<float, 16> v59_data(r0.template select<16, 1>(272));
              tensorforge::intel_esimd::simd<float, 16> v60_data(r0.template select<16, 1>(288));
              tensorforge::intel_esimd::simd<float, 16> v61_data(r0.template select<16, 1>(304));
              tensorforge::intel_esimd::simd<float, 16> v62_acc{};
              tensorforge::intel_esimd::simd<float, 16> v67_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v62_acc += ((static_cast<float>(v67_data[0])) * v42_data);
              v62_acc += ((static_cast<float>(v67_data[1])) * v43_data);
              v62_acc += ((static_cast<float>(v67_data[2])) * v44_data);
              v62_acc += ((static_cast<float>(v67_data[3])) * v45_data);
              v62_acc += ((static_cast<float>(v67_data[4])) * v46_data);
              v62_acc += ((static_cast<float>(v67_data[5])) * v47_data);
              v62_acc += ((static_cast<float>(v67_data[6])) * v48_data);
              v62_acc += ((static_cast<float>(v67_data[7])) * v49_data);
              v62_acc += ((static_cast<float>(v67_data[8])) * v50_data);
              v62_acc += ((static_cast<float>(v67_data[9])) * v51_data);
              v62_acc += ((static_cast<float>(v67_data[10])) * v52_data);
              v62_acc += ((static_cast<float>(v67_data[11])) * v53_data);
              v62_acc += ((static_cast<float>(v67_data[12])) * v54_data);
              v62_acc += ((static_cast<float>(v67_data[13])) * v55_data);
              v62_acc += ((static_cast<float>(v67_data[14])) * v56_data);
              v62_acc += ((static_cast<float>(v67_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v104_data = tensorforge::slmLoad<float, 16>(s0 + (272_i32));
              v62_acc += ((static_cast<float>(v104_data[0])) * v58_data);
              v62_acc += ((static_cast<float>(v104_data[1])) * v59_data);
              v62_acc += ((static_cast<float>(v104_data[2])) * v60_data);
              v62_acc += ((static_cast<float>(v104_data[3])) * v61_data);
              ir1.template select<16, 1>(0) = v62_acc;
              tensorforge::intel_esimd::simd<float, 16> v113_acc{};
              tensorforge::intel_esimd::simd<float, 16> v115_data = tensorforge::slmLoad<float, 16>(s0 + (1_i32));
              v113_acc += ((static_cast<float>(v115_data[0])) * v42_data);
              v113_acc += ((static_cast<float>(v115_data[1])) * v43_data);
              v113_acc += ((static_cast<float>(v115_data[2])) * v44_data);
              v113_acc += ((static_cast<float>(v115_data[3])) * v45_data);
              v113_acc += ((static_cast<float>(v115_data[4])) * v46_data);
              v113_acc += ((static_cast<float>(v115_data[5])) * v47_data);
              v113_acc += ((static_cast<float>(v115_data[6])) * v48_data);
              v113_acc += ((static_cast<float>(v115_data[7])) * v49_data);
              v113_acc += ((static_cast<float>(v115_data[8])) * v50_data);
              v113_acc += ((static_cast<float>(v115_data[9])) * v51_data);
              v113_acc += ((static_cast<float>(v115_data[10])) * v52_data);
              v113_acc += ((static_cast<float>(v115_data[11])) * v53_data);
              v113_acc += ((static_cast<float>(v115_data[12])) * v54_data);
              v113_acc += ((static_cast<float>(v115_data[13])) * v55_data);
              v113_acc += ((static_cast<float>(v115_data[14])) * v56_data);
              v113_acc += ((static_cast<float>(v115_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v149_data = tensorforge::slmLoad<float, 16>(s0 + (273_i32));
              v113_acc += ((static_cast<float>(v149_data[0])) * v58_data);
              v113_acc += ((static_cast<float>(v149_data[1])) * v59_data);
              v113_acc += ((static_cast<float>(v149_data[2])) * v60_data);
              v113_acc += ((static_cast<float>(v149_data[3])) * v61_data);
              ir1.template select<16, 1>(16) = v113_acc;
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data = tensorforge::slmLoad<float, 16>(s0 + (2_i32));
              v158_acc += ((static_cast<float>(v160_data[0])) * v42_data);
              v158_acc += ((static_cast<float>(v160_data[1])) * v43_data);
              v158_acc += ((static_cast<float>(v160_data[2])) * v44_data);
              v158_acc += ((static_cast<float>(v160_data[3])) * v45_data);
              v158_acc += ((static_cast<float>(v160_data[4])) * v46_data);
              v158_acc += ((static_cast<float>(v160_data[5])) * v47_data);
              v158_acc += ((static_cast<float>(v160_data[6])) * v48_data);
              v158_acc += ((static_cast<float>(v160_data[7])) * v49_data);
              v158_acc += ((static_cast<float>(v160_data[8])) * v50_data);
              v158_acc += ((static_cast<float>(v160_data[9])) * v51_data);
              v158_acc += ((static_cast<float>(v160_data[10])) * v52_data);
              v158_acc += ((static_cast<float>(v160_data[11])) * v53_data);
              v158_acc += ((static_cast<float>(v160_data[12])) * v54_data);
              v158_acc += ((static_cast<float>(v160_data[13])) * v55_data);
              v158_acc += ((static_cast<float>(v160_data[14])) * v56_data);
              v158_acc += ((static_cast<float>(v160_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v194_data = tensorforge::slmLoad<float, 16>(s0 + (274_i32));
              v158_acc += ((static_cast<float>(v194_data[0])) * v58_data);
              v158_acc += ((static_cast<float>(v194_data[1])) * v59_data);
              v158_acc += ((static_cast<float>(v194_data[2])) * v60_data);
              v158_acc += ((static_cast<float>(v194_data[3])) * v61_data);
              ir1.template select<16, 1>(32) = v158_acc;
              tensorforge::intel_esimd::simd<float, 16> v203_acc{};
              tensorforge::intel_esimd::simd<float, 16> v205_data = tensorforge::slmLoad<float, 16>(s0 + (3_i32));
              v203_acc += ((static_cast<float>(v205_data[0])) * v42_data);
              v203_acc += ((static_cast<float>(v205_data[1])) * v43_data);
              v203_acc += ((static_cast<float>(v205_data[2])) * v44_data);
              v203_acc += ((static_cast<float>(v205_data[3])) * v45_data);
              v203_acc += ((static_cast<float>(v205_data[4])) * v46_data);
              v203_acc += ((static_cast<float>(v205_data[5])) * v47_data);
              v203_acc += ((static_cast<float>(v205_data[6])) * v48_data);
              v203_acc += ((static_cast<float>(v205_data[7])) * v49_data);
              v203_acc += ((static_cast<float>(v205_data[8])) * v50_data);
              v203_acc += ((static_cast<float>(v205_data[9])) * v51_data);
              v203_acc += ((static_cast<float>(v205_data[10])) * v52_data);
              v203_acc += ((static_cast<float>(v205_data[11])) * v53_data);
              v203_acc += ((static_cast<float>(v205_data[12])) * v54_data);
              v203_acc += ((static_cast<float>(v205_data[13])) * v55_data);
              v203_acc += ((static_cast<float>(v205_data[14])) * v56_data);
              v203_acc += ((static_cast<float>(v205_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v239_data = tensorforge::slmLoad<float, 16>(s0 + (275_i32));
              v203_acc += ((static_cast<float>(v239_data[0])) * v58_data);
              v203_acc += ((static_cast<float>(v239_data[1])) * v59_data);
              v203_acc += ((static_cast<float>(v239_data[2])) * v60_data);
              v203_acc += ((static_cast<float>(v239_data[3])) * v61_data);
              ir1.template select<16, 1>(48) = v203_acc;
              tensorforge::intel_esimd::simd<float, 16> v248_acc{};
              tensorforge::intel_esimd::simd<float, 16> v250_data = tensorforge::slmLoad<float, 16>(s0 + (4_i32));
              v248_acc += ((static_cast<float>(v250_data[0])) * v42_data);
              v248_acc += ((static_cast<float>(v250_data[1])) * v43_data);
              v248_acc += ((static_cast<float>(v250_data[2])) * v44_data);
              v248_acc += ((static_cast<float>(v250_data[3])) * v45_data);
              v248_acc += ((static_cast<float>(v250_data[4])) * v46_data);
              v248_acc += ((static_cast<float>(v250_data[5])) * v47_data);
              v248_acc += ((static_cast<float>(v250_data[6])) * v48_data);
              v248_acc += ((static_cast<float>(v250_data[7])) * v49_data);
              v248_acc += ((static_cast<float>(v250_data[8])) * v50_data);
              v248_acc += ((static_cast<float>(v250_data[9])) * v51_data);
              v248_acc += ((static_cast<float>(v250_data[10])) * v52_data);
              v248_acc += ((static_cast<float>(v250_data[11])) * v53_data);
              v248_acc += ((static_cast<float>(v250_data[12])) * v54_data);
              v248_acc += ((static_cast<float>(v250_data[13])) * v55_data);
              v248_acc += ((static_cast<float>(v250_data[14])) * v56_data);
              v248_acc += ((static_cast<float>(v250_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v284_data = tensorforge::slmLoad<float, 16>(s0 + (276_i32));
              v248_acc += ((static_cast<float>(v284_data[0])) * v58_data);
              v248_acc += ((static_cast<float>(v284_data[1])) * v59_data);
              v248_acc += ((static_cast<float>(v284_data[2])) * v60_data);
              v248_acc += ((static_cast<float>(v284_data[3])) * v61_data);
              ir1.template select<16, 1>(64) = v248_acc;
              tensorforge::intel_esimd::simd<float, 16> v293_acc{};
              tensorforge::intel_esimd::simd<float, 16> v295_data = tensorforge::slmLoad<float, 16>(s0 + (5_i32));
              v293_acc += ((static_cast<float>(v295_data[0])) * v42_data);
              v293_acc += ((static_cast<float>(v295_data[1])) * v43_data);
              v293_acc += ((static_cast<float>(v295_data[2])) * v44_data);
              v293_acc += ((static_cast<float>(v295_data[3])) * v45_data);
              v293_acc += ((static_cast<float>(v295_data[4])) * v46_data);
              v293_acc += ((static_cast<float>(v295_data[5])) * v47_data);
              v293_acc += ((static_cast<float>(v295_data[6])) * v48_data);
              v293_acc += ((static_cast<float>(v295_data[7])) * v49_data);
              v293_acc += ((static_cast<float>(v295_data[8])) * v50_data);
              v293_acc += ((static_cast<float>(v295_data[9])) * v51_data);
              v293_acc += ((static_cast<float>(v295_data[10])) * v52_data);
              v293_acc += ((static_cast<float>(v295_data[11])) * v53_data);
              v293_acc += ((static_cast<float>(v295_data[12])) * v54_data);
              v293_acc += ((static_cast<float>(v295_data[13])) * v55_data);
              v293_acc += ((static_cast<float>(v295_data[14])) * v56_data);
              v293_acc += ((static_cast<float>(v295_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v329_data = tensorforge::slmLoad<float, 16>(s0 + (277_i32));
              v293_acc += ((static_cast<float>(v329_data[0])) * v58_data);
              v293_acc += ((static_cast<float>(v329_data[1])) * v59_data);
              v293_acc += ((static_cast<float>(v329_data[2])) * v60_data);
              v293_acc += ((static_cast<float>(v329_data[3])) * v61_data);
              ir1.template select<16, 1>(80) = v293_acc;
              tensorforge::intel_esimd::simd<float, 16> v338_acc{};
              tensorforge::intel_esimd::simd<float, 16> v340_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              v338_acc += ((static_cast<float>(v340_data[0])) * v42_data);
              v338_acc += ((static_cast<float>(v340_data[1])) * v43_data);
              v338_acc += ((static_cast<float>(v340_data[2])) * v44_data);
              v338_acc += ((static_cast<float>(v340_data[3])) * v45_data);
              v338_acc += ((static_cast<float>(v340_data[4])) * v46_data);
              v338_acc += ((static_cast<float>(v340_data[5])) * v47_data);
              v338_acc += ((static_cast<float>(v340_data[6])) * v48_data);
              v338_acc += ((static_cast<float>(v340_data[7])) * v49_data);
              v338_acc += ((static_cast<float>(v340_data[8])) * v50_data);
              v338_acc += ((static_cast<float>(v340_data[9])) * v51_data);
              v338_acc += ((static_cast<float>(v340_data[10])) * v52_data);
              v338_acc += ((static_cast<float>(v340_data[11])) * v53_data);
              v338_acc += ((static_cast<float>(v340_data[12])) * v54_data);
              v338_acc += ((static_cast<float>(v340_data[13])) * v55_data);
              v338_acc += ((static_cast<float>(v340_data[14])) * v56_data);
              v338_acc += ((static_cast<float>(v340_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v374_data = tensorforge::slmLoad<float, 16>(s0 + (278_i32));
              v338_acc += ((static_cast<float>(v374_data[0])) * v58_data);
              v338_acc += ((static_cast<float>(v374_data[1])) * v59_data);
              v338_acc += ((static_cast<float>(v374_data[2])) * v60_data);
              v338_acc += ((static_cast<float>(v374_data[3])) * v61_data);
              ir1.template select<16, 1>(96) = v338_acc;
              tensorforge::intel_esimd::simd<float, 16> v383_acc{};
              tensorforge::intel_esimd::simd<float, 16> v385_data = tensorforge::slmLoad<float, 16>(s0 + (7_i32));
              v383_acc += ((static_cast<float>(v385_data[0])) * v42_data);
              v383_acc += ((static_cast<float>(v385_data[1])) * v43_data);
              v383_acc += ((static_cast<float>(v385_data[2])) * v44_data);
              v383_acc += ((static_cast<float>(v385_data[3])) * v45_data);
              v383_acc += ((static_cast<float>(v385_data[4])) * v46_data);
              v383_acc += ((static_cast<float>(v385_data[5])) * v47_data);
              v383_acc += ((static_cast<float>(v385_data[6])) * v48_data);
              v383_acc += ((static_cast<float>(v385_data[7])) * v49_data);
              v383_acc += ((static_cast<float>(v385_data[8])) * v50_data);
              v383_acc += ((static_cast<float>(v385_data[9])) * v51_data);
              v383_acc += ((static_cast<float>(v385_data[10])) * v52_data);
              v383_acc += ((static_cast<float>(v385_data[11])) * v53_data);
              v383_acc += ((static_cast<float>(v385_data[12])) * v54_data);
              v383_acc += ((static_cast<float>(v385_data[13])) * v55_data);
              v383_acc += ((static_cast<float>(v385_data[14])) * v56_data);
              v383_acc += ((static_cast<float>(v385_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v419_data = tensorforge::slmLoad<float, 16>(s0 + (279_i32));
              v383_acc += ((static_cast<float>(v419_data[0])) * v58_data);
              v383_acc += ((static_cast<float>(v419_data[1])) * v59_data);
              v383_acc += ((static_cast<float>(v419_data[2])) * v60_data);
              v383_acc += ((static_cast<float>(v419_data[3])) * v61_data);
              ir1.template select<16, 1>(112) = v383_acc;
              tensorforge::intel_esimd::simd<float, 16> v428_acc{};
              tensorforge::intel_esimd::simd<float, 16> v430_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              v428_acc += ((static_cast<float>(v430_data[0])) * v42_data);
              v428_acc += ((static_cast<float>(v430_data[1])) * v43_data);
              v428_acc += ((static_cast<float>(v430_data[2])) * v44_data);
              v428_acc += ((static_cast<float>(v430_data[3])) * v45_data);
              v428_acc += ((static_cast<float>(v430_data[4])) * v46_data);
              v428_acc += ((static_cast<float>(v430_data[5])) * v47_data);
              v428_acc += ((static_cast<float>(v430_data[6])) * v48_data);
              v428_acc += ((static_cast<float>(v430_data[7])) * v49_data);
              v428_acc += ((static_cast<float>(v430_data[8])) * v50_data);
              v428_acc += ((static_cast<float>(v430_data[9])) * v51_data);
              v428_acc += ((static_cast<float>(v430_data[10])) * v52_data);
              v428_acc += ((static_cast<float>(v430_data[11])) * v53_data);
              v428_acc += ((static_cast<float>(v430_data[12])) * v54_data);
              v428_acc += ((static_cast<float>(v430_data[13])) * v55_data);
              v428_acc += ((static_cast<float>(v430_data[14])) * v56_data);
              v428_acc += ((static_cast<float>(v430_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v464_data = tensorforge::slmLoad<float, 16>(s0 + (280_i32));
              v428_acc += ((static_cast<float>(v464_data[0])) * v58_data);
              v428_acc += ((static_cast<float>(v464_data[1])) * v59_data);
              v428_acc += ((static_cast<float>(v464_data[2])) * v60_data);
              v428_acc += ((static_cast<float>(v464_data[3])) * v61_data);
              ir1.template select<16, 1>(128) = v428_acc;
              tensorforge::intel_esimd::simd<float, 16> v473_acc{};
              tensorforge::intel_esimd::simd<float, 16> v475_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              v473_acc += ((static_cast<float>(v475_data[0])) * v42_data);
              v473_acc += ((static_cast<float>(v475_data[1])) * v43_data);
              v473_acc += ((static_cast<float>(v475_data[2])) * v44_data);
              v473_acc += ((static_cast<float>(v475_data[3])) * v45_data);
              v473_acc += ((static_cast<float>(v475_data[4])) * v46_data);
              v473_acc += ((static_cast<float>(v475_data[5])) * v47_data);
              v473_acc += ((static_cast<float>(v475_data[6])) * v48_data);
              v473_acc += ((static_cast<float>(v475_data[7])) * v49_data);
              v473_acc += ((static_cast<float>(v475_data[8])) * v50_data);
              v473_acc += ((static_cast<float>(v475_data[9])) * v51_data);
              v473_acc += ((static_cast<float>(v475_data[10])) * v52_data);
              v473_acc += ((static_cast<float>(v475_data[11])) * v53_data);
              v473_acc += ((static_cast<float>(v475_data[12])) * v54_data);
              v473_acc += ((static_cast<float>(v475_data[13])) * v55_data);
              v473_acc += ((static_cast<float>(v475_data[14])) * v56_data);
              v473_acc += ((static_cast<float>(v475_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v509_data = tensorforge::slmLoad<float, 16>(s0 + (281_i32));
              v473_acc += ((static_cast<float>(v509_data[0])) * v58_data);
              v473_acc += ((static_cast<float>(v509_data[1])) * v59_data);
              v473_acc += ((static_cast<float>(v509_data[2])) * v60_data);
              v473_acc += ((static_cast<float>(v509_data[3])) * v61_data);
              ir1.template select<16, 1>(144) = v473_acc;
              tensorforge::intel_esimd::simd<float, 16> v518_acc{};
              tensorforge::intel_esimd::simd<float, 16> v520_data = tensorforge::slmLoad<float, 16>(s0 + (10_i32));
              v518_acc += ((static_cast<float>(v520_data[0])) * v42_data);
              v518_acc += ((static_cast<float>(v520_data[1])) * v43_data);
              v518_acc += ((static_cast<float>(v520_data[2])) * v44_data);
              v518_acc += ((static_cast<float>(v520_data[3])) * v45_data);
              v518_acc += ((static_cast<float>(v520_data[4])) * v46_data);
              v518_acc += ((static_cast<float>(v520_data[5])) * v47_data);
              v518_acc += ((static_cast<float>(v520_data[6])) * v48_data);
              v518_acc += ((static_cast<float>(v520_data[7])) * v49_data);
              v518_acc += ((static_cast<float>(v520_data[8])) * v50_data);
              v518_acc += ((static_cast<float>(v520_data[9])) * v51_data);
              v518_acc += ((static_cast<float>(v520_data[10])) * v52_data);
              v518_acc += ((static_cast<float>(v520_data[11])) * v53_data);
              v518_acc += ((static_cast<float>(v520_data[12])) * v54_data);
              v518_acc += ((static_cast<float>(v520_data[13])) * v55_data);
              v518_acc += ((static_cast<float>(v520_data[14])) * v56_data);
              v518_acc += ((static_cast<float>(v520_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v554_data = tensorforge::slmLoad<float, 16>(s0 + (282_i32));
              v518_acc += ((static_cast<float>(v554_data[0])) * v58_data);
              v518_acc += ((static_cast<float>(v554_data[1])) * v59_data);
              v518_acc += ((static_cast<float>(v554_data[2])) * v60_data);
              v518_acc += ((static_cast<float>(v554_data[3])) * v61_data);
              ir1.template select<16, 1>(160) = v518_acc;
              tensorforge::intel_esimd::simd<float, 16> v563_acc{};
              tensorforge::intel_esimd::simd<float, 16> v565_data = tensorforge::slmLoad<float, 16>(s0 + (11_i32));
              v563_acc += ((static_cast<float>(v565_data[0])) * v42_data);
              v563_acc += ((static_cast<float>(v565_data[1])) * v43_data);
              v563_acc += ((static_cast<float>(v565_data[2])) * v44_data);
              v563_acc += ((static_cast<float>(v565_data[3])) * v45_data);
              v563_acc += ((static_cast<float>(v565_data[4])) * v46_data);
              v563_acc += ((static_cast<float>(v565_data[5])) * v47_data);
              v563_acc += ((static_cast<float>(v565_data[6])) * v48_data);
              v563_acc += ((static_cast<float>(v565_data[7])) * v49_data);
              v563_acc += ((static_cast<float>(v565_data[8])) * v50_data);
              v563_acc += ((static_cast<float>(v565_data[9])) * v51_data);
              v563_acc += ((static_cast<float>(v565_data[10])) * v52_data);
              v563_acc += ((static_cast<float>(v565_data[11])) * v53_data);
              v563_acc += ((static_cast<float>(v565_data[12])) * v54_data);
              v563_acc += ((static_cast<float>(v565_data[13])) * v55_data);
              v563_acc += ((static_cast<float>(v565_data[14])) * v56_data);
              v563_acc += ((static_cast<float>(v565_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v599_data = tensorforge::slmLoad<float, 16>(s0 + (283_i32));
              v563_acc += ((static_cast<float>(v599_data[0])) * v58_data);
              v563_acc += ((static_cast<float>(v599_data[1])) * v59_data);
              v563_acc += ((static_cast<float>(v599_data[2])) * v60_data);
              v563_acc += ((static_cast<float>(v599_data[3])) * v61_data);
              ir1.template select<16, 1>(176) = v563_acc;
              tensorforge::intel_esimd::simd<float, 16> v608_acc{};
              tensorforge::intel_esimd::simd<float, 16> v610_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v608_acc += ((static_cast<float>(v610_data[0])) * v42_data);
              v608_acc += ((static_cast<float>(v610_data[1])) * v43_data);
              v608_acc += ((static_cast<float>(v610_data[2])) * v44_data);
              v608_acc += ((static_cast<float>(v610_data[3])) * v45_data);
              v608_acc += ((static_cast<float>(v610_data[4])) * v46_data);
              v608_acc += ((static_cast<float>(v610_data[5])) * v47_data);
              v608_acc += ((static_cast<float>(v610_data[6])) * v48_data);
              v608_acc += ((static_cast<float>(v610_data[7])) * v49_data);
              v608_acc += ((static_cast<float>(v610_data[8])) * v50_data);
              v608_acc += ((static_cast<float>(v610_data[9])) * v51_data);
              v608_acc += ((static_cast<float>(v610_data[10])) * v52_data);
              v608_acc += ((static_cast<float>(v610_data[11])) * v53_data);
              v608_acc += ((static_cast<float>(v610_data[12])) * v54_data);
              v608_acc += ((static_cast<float>(v610_data[13])) * v55_data);
              v608_acc += ((static_cast<float>(v610_data[14])) * v56_data);
              v608_acc += ((static_cast<float>(v610_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v644_data = tensorforge::slmLoad<float, 16>(s0 + (284_i32));
              v608_acc += ((static_cast<float>(v644_data[0])) * v58_data);
              v608_acc += ((static_cast<float>(v644_data[1])) * v59_data);
              v608_acc += ((static_cast<float>(v644_data[2])) * v60_data);
              v608_acc += ((static_cast<float>(v644_data[3])) * v61_data);
              ir1.template select<16, 1>(192) = v608_acc;
              tensorforge::intel_esimd::simd<float, 16> v653_acc{};
              tensorforge::intel_esimd::simd<float, 16> v655_data = tensorforge::slmLoad<float, 16>(s0 + (13_i32));
              v653_acc += ((static_cast<float>(v655_data[0])) * v42_data);
              v653_acc += ((static_cast<float>(v655_data[1])) * v43_data);
              v653_acc += ((static_cast<float>(v655_data[2])) * v44_data);
              v653_acc += ((static_cast<float>(v655_data[3])) * v45_data);
              v653_acc += ((static_cast<float>(v655_data[4])) * v46_data);
              v653_acc += ((static_cast<float>(v655_data[5])) * v47_data);
              v653_acc += ((static_cast<float>(v655_data[6])) * v48_data);
              v653_acc += ((static_cast<float>(v655_data[7])) * v49_data);
              v653_acc += ((static_cast<float>(v655_data[8])) * v50_data);
              v653_acc += ((static_cast<float>(v655_data[9])) * v51_data);
              v653_acc += ((static_cast<float>(v655_data[10])) * v52_data);
              v653_acc += ((static_cast<float>(v655_data[11])) * v53_data);
              v653_acc += ((static_cast<float>(v655_data[12])) * v54_data);
              v653_acc += ((static_cast<float>(v655_data[13])) * v55_data);
              v653_acc += ((static_cast<float>(v655_data[14])) * v56_data);
              v653_acc += ((static_cast<float>(v655_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v689_data = tensorforge::slmLoad<float, 16>(s0 + (285_i32));
              v653_acc += ((static_cast<float>(v689_data[0])) * v58_data);
              v653_acc += ((static_cast<float>(v689_data[1])) * v59_data);
              v653_acc += ((static_cast<float>(v689_data[2])) * v60_data);
              v653_acc += ((static_cast<float>(v689_data[3])) * v61_data);
              ir1.template select<16, 1>(208) = v653_acc;
              tensorforge::intel_esimd::simd<float, 16> v698_acc{};
              tensorforge::intel_esimd::simd<float, 16> v700_data = tensorforge::slmLoad<float, 16>(s0 + (14_i32));
              v698_acc += ((static_cast<float>(v700_data[0])) * v42_data);
              v698_acc += ((static_cast<float>(v700_data[1])) * v43_data);
              v698_acc += ((static_cast<float>(v700_data[2])) * v44_data);
              v698_acc += ((static_cast<float>(v700_data[3])) * v45_data);
              v698_acc += ((static_cast<float>(v700_data[4])) * v46_data);
              v698_acc += ((static_cast<float>(v700_data[5])) * v47_data);
              v698_acc += ((static_cast<float>(v700_data[6])) * v48_data);
              v698_acc += ((static_cast<float>(v700_data[7])) * v49_data);
              v698_acc += ((static_cast<float>(v700_data[8])) * v50_data);
              v698_acc += ((static_cast<float>(v700_data[9])) * v51_data);
              v698_acc += ((static_cast<float>(v700_data[10])) * v52_data);
              v698_acc += ((static_cast<float>(v700_data[11])) * v53_data);
              v698_acc += ((static_cast<float>(v700_data[12])) * v54_data);
              v698_acc += ((static_cast<float>(v700_data[13])) * v55_data);
              v698_acc += ((static_cast<float>(v700_data[14])) * v56_data);
              v698_acc += ((static_cast<float>(v700_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v734_data = tensorforge::slmLoad<float, 16>(s0 + (286_i32));
              v698_acc += ((static_cast<float>(v734_data[0])) * v58_data);
              v698_acc += ((static_cast<float>(v734_data[1])) * v59_data);
              v698_acc += ((static_cast<float>(v734_data[2])) * v60_data);
              v698_acc += ((static_cast<float>(v734_data[3])) * v61_data);
              ir1.template select<16, 1>(224) = v698_acc;
              tensorforge::intel_esimd::simd<float, 16> v743_acc{};
              tensorforge::intel_esimd::simd<float, 16> v745_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v743_acc += ((static_cast<float>(v745_data[0])) * v42_data);
              v743_acc += ((static_cast<float>(v745_data[1])) * v43_data);
              v743_acc += ((static_cast<float>(v745_data[2])) * v44_data);
              v743_acc += ((static_cast<float>(v745_data[3])) * v45_data);
              v743_acc += ((static_cast<float>(v745_data[4])) * v46_data);
              v743_acc += ((static_cast<float>(v745_data[5])) * v47_data);
              v743_acc += ((static_cast<float>(v745_data[6])) * v48_data);
              v743_acc += ((static_cast<float>(v745_data[7])) * v49_data);
              v743_acc += ((static_cast<float>(v745_data[8])) * v50_data);
              v743_acc += ((static_cast<float>(v745_data[9])) * v51_data);
              v743_acc += ((static_cast<float>(v745_data[10])) * v52_data);
              v743_acc += ((static_cast<float>(v745_data[11])) * v53_data);
              v743_acc += ((static_cast<float>(v745_data[12])) * v54_data);
              v743_acc += ((static_cast<float>(v745_data[13])) * v55_data);
              v743_acc += ((static_cast<float>(v745_data[14])) * v56_data);
              v743_acc += ((static_cast<float>(v745_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v779_data = tensorforge::slmLoad<float, 16>(s0 + (287_i32));
              v743_acc += ((static_cast<float>(v779_data[0])) * v58_data);
              v743_acc += ((static_cast<float>(v779_data[1])) * v59_data);
              v743_acc += ((static_cast<float>(v779_data[2])) * v60_data);
              v743_acc += ((static_cast<float>(v779_data[3])) * v61_data);
              ir1.template select<16, 1>(240) = v743_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v788_n1 = 0; v788_n1 < 16; ++v788_n1) {
                int32_t v789_a = v788_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v791_data(ir1.template select<12, 1>(v789_a));
                r1.template select<12, 1>(v789_a) = v791_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v792_i1 = 0; v792_i1 < 16; ++v792_i1) {
                tensorforge::intel_esimd::simd<float, 12> v795_data(r1.template select<12, 1>((v792_i1 * 16)));
                v795_data.copy_to(glb_m0 + ((v792_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}


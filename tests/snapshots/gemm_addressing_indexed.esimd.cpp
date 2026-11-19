// === base name ===
kernel_766af004634c9efc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_766af004634c9efc = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_766af004634c9efc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_766af004634c9efc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_766af004634c9efc(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4352 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_766af004634c9efc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_766af004634c9efc(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_766af004634c9efc(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2, m2_extraOffset, m2_slot, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_766af004634c9efc_k;
inline void kernel_kernel_766af004634c9efc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_766af004634c9efc_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4352 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 17408 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided(stride=300, offset=8)
        //   m1 16×16(16×16) {0..16}×{0..16} indexed(stride=runtime)
        //   m2 16×16(16×16) {0..16}×{0..16} indexed_pointer(offset=4, slots=4, slot=runtime)
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4352}],"shared_bytes":17408,"shared_elements":4352,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided(stride=300, offset=8)","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided(stride=300, offset=8)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v628_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v628_flagSecondAhead < numElements0) ? v628_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
          const float *const __restrict__ peel_glb_m1 = &m1[peel_elementIndex * m1_stride + 0 + m1_extraOffset];
          if (allowed_peel) {
            // r0 = load{g>r}(peel_glb_m1);
            #pragma unroll
            for (int32_t v646_i0 = 0; v646_i0 < 1; ++v646_i0) {
              int32_t v647_lead = v646_i0 * 16;
              #pragma unroll
              for (int32_t v648_i1 = 0; v648_i1 < 16; ++v648_i1) {
                int32_t v650_a = v647_lead + (v648_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v651_data;
                v651_data.copy_from(peel_glb_m1 + (v650_a));
                r0.template select<16, 1>(v650_a) = v651_data;
              }
            }
          }
          const float *const __restrict__ peel_glb_m2 = &m2[peel_elementIndex * 4 + m2_slot][0 + 4 + m2_extraOffset];
          if (allowed_peel) {
            // s0 = load{g>s}(peel_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v660_ld;
            v660_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v660_ld);
            tensorforge::intel_esimd::simd<float, 64> v661_ld;
            v661_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v661_ld);
            tensorforge::intel_esimd::simd<float, 64> v662_ld;
            v662_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 128));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v662_ld);
            tensorforge::intel_esimd::simd<float, 64> v663_ld;
            v663_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 192));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v663_ld);
          }
          uint32_t v632_flagWord(flagWordFirst);
          uint32_t v633_flagWordNext(flagWordSecond);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v11_batchId1]);
            const float *const __restrict__ wrap_glb_m1 = &m1[wrap_elementIndex * m1_stride + 0 + m1_extraOffset];
            const float *const __restrict__ wrap_glb_m2 = &m2[wrap_elementIndex * 4 + m2_slot][0 + 4 + m2_extraOffset];
            size_t v670_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v670_flagAheadAhead < numElements0) ? v670_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v632_flagWord);
            const bool allowed_next = static_cast<bool>(v633_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 300 + 0 + 8 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v51_acc{};
              tensorforge::intel_esimd::simd<float, 16> v55_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v51_acc += ((static_cast<float>(v55_data[0])) * v35_data);
              v51_acc += ((static_cast<float>(v55_data[1])) * v36_data);
              v51_acc += ((static_cast<float>(v55_data[2])) * v37_data);
              v51_acc += ((static_cast<float>(v55_data[3])) * v38_data);
              v51_acc += ((static_cast<float>(v55_data[4])) * v39_data);
              v51_acc += ((static_cast<float>(v55_data[5])) * v40_data);
              v51_acc += ((static_cast<float>(v55_data[6])) * v41_data);
              v51_acc += ((static_cast<float>(v55_data[7])) * v42_data);
              v51_acc += ((static_cast<float>(v55_data[8])) * v43_data);
              v51_acc += ((static_cast<float>(v55_data[9])) * v44_data);
              v51_acc += ((static_cast<float>(v55_data[10])) * v45_data);
              v51_acc += ((static_cast<float>(v55_data[11])) * v46_data);
              v51_acc += ((static_cast<float>(v55_data[12])) * v47_data);
              v51_acc += ((static_cast<float>(v55_data[13])) * v48_data);
              v51_acc += ((static_cast<float>(v55_data[14])) * v49_data);
              v51_acc += ((static_cast<float>(v55_data[15])) * v50_data);
              ir1.template select<16, 1>(0) = v51_acc;
              tensorforge::intel_esimd::simd<float, 16> v88_acc{};
              tensorforge::intel_esimd::simd<float, 16> v90_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v88_acc += ((static_cast<float>(v90_data[0])) * v35_data);
              v88_acc += ((static_cast<float>(v90_data[1])) * v36_data);
              v88_acc += ((static_cast<float>(v90_data[2])) * v37_data);
              v88_acc += ((static_cast<float>(v90_data[3])) * v38_data);
              v88_acc += ((static_cast<float>(v90_data[4])) * v39_data);
              v88_acc += ((static_cast<float>(v90_data[5])) * v40_data);
              v88_acc += ((static_cast<float>(v90_data[6])) * v41_data);
              v88_acc += ((static_cast<float>(v90_data[7])) * v42_data);
              v88_acc += ((static_cast<float>(v90_data[8])) * v43_data);
              v88_acc += ((static_cast<float>(v90_data[9])) * v44_data);
              v88_acc += ((static_cast<float>(v90_data[10])) * v45_data);
              v88_acc += ((static_cast<float>(v90_data[11])) * v46_data);
              v88_acc += ((static_cast<float>(v90_data[12])) * v47_data);
              v88_acc += ((static_cast<float>(v90_data[13])) * v48_data);
              v88_acc += ((static_cast<float>(v90_data[14])) * v49_data);
              v88_acc += ((static_cast<float>(v90_data[15])) * v50_data);
              ir1.template select<16, 1>(16) = v88_acc;
              tensorforge::intel_esimd::simd<float, 16> v123_acc{};
              tensorforge::intel_esimd::simd<float, 16> v125_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v123_acc += ((static_cast<float>(v125_data[0])) * v35_data);
              v123_acc += ((static_cast<float>(v125_data[1])) * v36_data);
              v123_acc += ((static_cast<float>(v125_data[2])) * v37_data);
              v123_acc += ((static_cast<float>(v125_data[3])) * v38_data);
              v123_acc += ((static_cast<float>(v125_data[4])) * v39_data);
              v123_acc += ((static_cast<float>(v125_data[5])) * v40_data);
              v123_acc += ((static_cast<float>(v125_data[6])) * v41_data);
              v123_acc += ((static_cast<float>(v125_data[7])) * v42_data);
              v123_acc += ((static_cast<float>(v125_data[8])) * v43_data);
              v123_acc += ((static_cast<float>(v125_data[9])) * v44_data);
              v123_acc += ((static_cast<float>(v125_data[10])) * v45_data);
              v123_acc += ((static_cast<float>(v125_data[11])) * v46_data);
              v123_acc += ((static_cast<float>(v125_data[12])) * v47_data);
              v123_acc += ((static_cast<float>(v125_data[13])) * v48_data);
              v123_acc += ((static_cast<float>(v125_data[14])) * v49_data);
              v123_acc += ((static_cast<float>(v125_data[15])) * v50_data);
              ir1.template select<16, 1>(32) = v123_acc;
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v158_acc += ((static_cast<float>(v160_data[0])) * v35_data);
              v158_acc += ((static_cast<float>(v160_data[1])) * v36_data);
              v158_acc += ((static_cast<float>(v160_data[2])) * v37_data);
              v158_acc += ((static_cast<float>(v160_data[3])) * v38_data);
              v158_acc += ((static_cast<float>(v160_data[4])) * v39_data);
              v158_acc += ((static_cast<float>(v160_data[5])) * v40_data);
              v158_acc += ((static_cast<float>(v160_data[6])) * v41_data);
              v158_acc += ((static_cast<float>(v160_data[7])) * v42_data);
              v158_acc += ((static_cast<float>(v160_data[8])) * v43_data);
              v158_acc += ((static_cast<float>(v160_data[9])) * v44_data);
              v158_acc += ((static_cast<float>(v160_data[10])) * v45_data);
              v158_acc += ((static_cast<float>(v160_data[11])) * v46_data);
              v158_acc += ((static_cast<float>(v160_data[12])) * v47_data);
              v158_acc += ((static_cast<float>(v160_data[13])) * v48_data);
              v158_acc += ((static_cast<float>(v160_data[14])) * v49_data);
              v158_acc += ((static_cast<float>(v160_data[15])) * v50_data);
              ir1.template select<16, 1>(48) = v158_acc;
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              tensorforge::intel_esimd::simd<float, 16> v195_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v193_acc += ((static_cast<float>(v195_data[0])) * v35_data);
              v193_acc += ((static_cast<float>(v195_data[1])) * v36_data);
              v193_acc += ((static_cast<float>(v195_data[2])) * v37_data);
              v193_acc += ((static_cast<float>(v195_data[3])) * v38_data);
              v193_acc += ((static_cast<float>(v195_data[4])) * v39_data);
              v193_acc += ((static_cast<float>(v195_data[5])) * v40_data);
              v193_acc += ((static_cast<float>(v195_data[6])) * v41_data);
              v193_acc += ((static_cast<float>(v195_data[7])) * v42_data);
              v193_acc += ((static_cast<float>(v195_data[8])) * v43_data);
              v193_acc += ((static_cast<float>(v195_data[9])) * v44_data);
              v193_acc += ((static_cast<float>(v195_data[10])) * v45_data);
              v193_acc += ((static_cast<float>(v195_data[11])) * v46_data);
              v193_acc += ((static_cast<float>(v195_data[12])) * v47_data);
              v193_acc += ((static_cast<float>(v195_data[13])) * v48_data);
              v193_acc += ((static_cast<float>(v195_data[14])) * v49_data);
              v193_acc += ((static_cast<float>(v195_data[15])) * v50_data);
              ir1.template select<16, 1>(64) = v193_acc;
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              tensorforge::intel_esimd::simd<float, 16> v230_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v228_acc += ((static_cast<float>(v230_data[0])) * v35_data);
              v228_acc += ((static_cast<float>(v230_data[1])) * v36_data);
              v228_acc += ((static_cast<float>(v230_data[2])) * v37_data);
              v228_acc += ((static_cast<float>(v230_data[3])) * v38_data);
              v228_acc += ((static_cast<float>(v230_data[4])) * v39_data);
              v228_acc += ((static_cast<float>(v230_data[5])) * v40_data);
              tensorforge::intel_esimd::simd<float, 16> v244_p = (static_cast<float>(v230_data[6])) * v41_data;
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m1);
                #pragma unroll
                for (int32_t v652_i0 = 0; v652_i0 < 1; ++v652_i0) {
                  int32_t v653_lead = v652_i0 * 16;
                  #pragma unroll
                  for (int32_t v654_i1 = 0; v654_i1 < 16; ++v654_i1) {
                    int32_t v656_a = v653_lead + (v654_i1 * 16);
                    tensorforge::intel_esimd::simd<float, 16> v657_data;
                    v657_data.copy_from(wrap_glb_m1 + (v656_a));
                    r0.template select<16, 1>(v656_a) = v657_data;
                  }
                }
              }
              v228_acc += v244_p;
              v228_acc += ((static_cast<float>(v230_data[7])) * v42_data);
              v228_acc += ((static_cast<float>(v230_data[8])) * v43_data);
              v228_acc += ((static_cast<float>(v230_data[9])) * v44_data);
              v228_acc += ((static_cast<float>(v230_data[10])) * v45_data);
              v228_acc += ((static_cast<float>(v230_data[11])) * v46_data);
              v228_acc += ((static_cast<float>(v230_data[12])) * v47_data);
              v228_acc += ((static_cast<float>(v230_data[13])) * v48_data);
              v228_acc += ((static_cast<float>(v230_data[14])) * v49_data);
              v228_acc += ((static_cast<float>(v230_data[15])) * v50_data);
              ir1.template select<16, 1>(80) = v228_acc;
              tensorforge::intel_esimd::simd<float, 16> v263_acc{};
              tensorforge::intel_esimd::simd<float, 16> v265_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v263_acc += ((static_cast<float>(v265_data[0])) * v35_data);
              v263_acc += ((static_cast<float>(v265_data[1])) * v36_data);
              v263_acc += ((static_cast<float>(v265_data[2])) * v37_data);
              v263_acc += ((static_cast<float>(v265_data[3])) * v38_data);
              v263_acc += ((static_cast<float>(v265_data[4])) * v39_data);
              v263_acc += ((static_cast<float>(v265_data[5])) * v40_data);
              v263_acc += ((static_cast<float>(v265_data[6])) * v41_data);
              v263_acc += ((static_cast<float>(v265_data[7])) * v42_data);
              v263_acc += ((static_cast<float>(v265_data[8])) * v43_data);
              v263_acc += ((static_cast<float>(v265_data[9])) * v44_data);
              v263_acc += ((static_cast<float>(v265_data[10])) * v45_data);
              v263_acc += ((static_cast<float>(v265_data[11])) * v46_data);
              v263_acc += ((static_cast<float>(v265_data[12])) * v47_data);
              v263_acc += ((static_cast<float>(v265_data[13])) * v48_data);
              v263_acc += ((static_cast<float>(v265_data[14])) * v49_data);
              v263_acc += ((static_cast<float>(v265_data[15])) * v50_data);
              ir1.template select<16, 1>(96) = v263_acc;
              tensorforge::intel_esimd::simd<float, 16> v298_acc{};
              tensorforge::intel_esimd::simd<float, 16> v300_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v298_acc += ((static_cast<float>(v300_data[0])) * v35_data);
              v298_acc += ((static_cast<float>(v300_data[1])) * v36_data);
              v298_acc += ((static_cast<float>(v300_data[2])) * v37_data);
              v298_acc += ((static_cast<float>(v300_data[3])) * v38_data);
              v298_acc += ((static_cast<float>(v300_data[4])) * v39_data);
              v298_acc += ((static_cast<float>(v300_data[5])) * v40_data);
              v298_acc += ((static_cast<float>(v300_data[6])) * v41_data);
              v298_acc += ((static_cast<float>(v300_data[7])) * v42_data);
              v298_acc += ((static_cast<float>(v300_data[8])) * v43_data);
              v298_acc += ((static_cast<float>(v300_data[9])) * v44_data);
              v298_acc += ((static_cast<float>(v300_data[10])) * v45_data);
              v298_acc += ((static_cast<float>(v300_data[11])) * v46_data);
              v298_acc += ((static_cast<float>(v300_data[12])) * v47_data);
              v298_acc += ((static_cast<float>(v300_data[13])) * v48_data);
              v298_acc += ((static_cast<float>(v300_data[14])) * v49_data);
              v298_acc += ((static_cast<float>(v300_data[15])) * v50_data);
              ir1.template select<16, 1>(112) = v298_acc;
              tensorforge::intel_esimd::simd<float, 16> v333_acc{};
              tensorforge::intel_esimd::simd<float, 16> v335_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v333_acc += ((static_cast<float>(v335_data[0])) * v35_data);
              v333_acc += ((static_cast<float>(v335_data[1])) * v36_data);
              v333_acc += ((static_cast<float>(v335_data[2])) * v37_data);
              v333_acc += ((static_cast<float>(v335_data[3])) * v38_data);
              v333_acc += ((static_cast<float>(v335_data[4])) * v39_data);
              v333_acc += ((static_cast<float>(v335_data[5])) * v40_data);
              v333_acc += ((static_cast<float>(v335_data[6])) * v41_data);
              v333_acc += ((static_cast<float>(v335_data[7])) * v42_data);
              v333_acc += ((static_cast<float>(v335_data[8])) * v43_data);
              v333_acc += ((static_cast<float>(v335_data[9])) * v44_data);
              v333_acc += ((static_cast<float>(v335_data[10])) * v45_data);
              v333_acc += ((static_cast<float>(v335_data[11])) * v46_data);
              v333_acc += ((static_cast<float>(v335_data[12])) * v47_data);
              v333_acc += ((static_cast<float>(v335_data[13])) * v48_data);
              v333_acc += ((static_cast<float>(v335_data[14])) * v49_data);
              v333_acc += ((static_cast<float>(v335_data[15])) * v50_data);
              ir1.template select<16, 1>(128) = v333_acc;
              tensorforge::intel_esimd::simd<float, 16> v368_acc{};
              tensorforge::intel_esimd::simd<float, 16> v370_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v368_acc += ((static_cast<float>(v370_data[0])) * v35_data);
              v368_acc += ((static_cast<float>(v370_data[1])) * v36_data);
              v368_acc += ((static_cast<float>(v370_data[2])) * v37_data);
              v368_acc += ((static_cast<float>(v370_data[3])) * v38_data);
              v368_acc += ((static_cast<float>(v370_data[4])) * v39_data);
              v368_acc += ((static_cast<float>(v370_data[5])) * v40_data);
              v368_acc += ((static_cast<float>(v370_data[6])) * v41_data);
              v368_acc += ((static_cast<float>(v370_data[7])) * v42_data);
              v368_acc += ((static_cast<float>(v370_data[8])) * v43_data);
              v368_acc += ((static_cast<float>(v370_data[9])) * v44_data);
              v368_acc += ((static_cast<float>(v370_data[10])) * v45_data);
              v368_acc += ((static_cast<float>(v370_data[11])) * v46_data);
              v368_acc += ((static_cast<float>(v370_data[12])) * v47_data);
              v368_acc += ((static_cast<float>(v370_data[13])) * v48_data);
              v368_acc += ((static_cast<float>(v370_data[14])) * v49_data);
              v368_acc += ((static_cast<float>(v370_data[15])) * v50_data);
              ir1.template select<16, 1>(144) = v368_acc;
              tensorforge::intel_esimd::simd<float, 16> v403_acc{};
              tensorforge::intel_esimd::simd<float, 16> v405_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v403_acc += ((static_cast<float>(v405_data[0])) * v35_data);
              v403_acc += ((static_cast<float>(v405_data[1])) * v36_data);
              v403_acc += ((static_cast<float>(v405_data[2])) * v37_data);
              v403_acc += ((static_cast<float>(v405_data[3])) * v38_data);
              v403_acc += ((static_cast<float>(v405_data[4])) * v39_data);
              v403_acc += ((static_cast<float>(v405_data[5])) * v40_data);
              v403_acc += ((static_cast<float>(v405_data[6])) * v41_data);
              v403_acc += ((static_cast<float>(v405_data[7])) * v42_data);
              v403_acc += ((static_cast<float>(v405_data[8])) * v43_data);
              v403_acc += ((static_cast<float>(v405_data[9])) * v44_data);
              v403_acc += ((static_cast<float>(v405_data[10])) * v45_data);
              v403_acc += ((static_cast<float>(v405_data[11])) * v46_data);
              v403_acc += ((static_cast<float>(v405_data[12])) * v47_data);
              v403_acc += ((static_cast<float>(v405_data[13])) * v48_data);
              v403_acc += ((static_cast<float>(v405_data[14])) * v49_data);
              v403_acc += ((static_cast<float>(v405_data[15])) * v50_data);
              ir1.template select<16, 1>(160) = v403_acc;
              tensorforge::intel_esimd::simd<float, 16> v438_acc{};
              tensorforge::intel_esimd::simd<float, 16> v440_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v438_acc += ((static_cast<float>(v440_data[0])) * v35_data);
              v438_acc += ((static_cast<float>(v440_data[1])) * v36_data);
              v438_acc += ((static_cast<float>(v440_data[2])) * v37_data);
              v438_acc += ((static_cast<float>(v440_data[3])) * v38_data);
              v438_acc += ((static_cast<float>(v440_data[4])) * v39_data);
              v438_acc += ((static_cast<float>(v440_data[5])) * v40_data);
              v438_acc += ((static_cast<float>(v440_data[6])) * v41_data);
              v438_acc += ((static_cast<float>(v440_data[7])) * v42_data);
              v438_acc += ((static_cast<float>(v440_data[8])) * v43_data);
              v438_acc += ((static_cast<float>(v440_data[9])) * v44_data);
              v438_acc += ((static_cast<float>(v440_data[10])) * v45_data);
              v438_acc += ((static_cast<float>(v440_data[11])) * v46_data);
              v438_acc += ((static_cast<float>(v440_data[12])) * v47_data);
              v438_acc += ((static_cast<float>(v440_data[13])) * v48_data);
              v438_acc += ((static_cast<float>(v440_data[14])) * v49_data);
              v438_acc += ((static_cast<float>(v440_data[15])) * v50_data);
              ir1.template select<16, 1>(176) = v438_acc;
              tensorforge::intel_esimd::simd<float, 16> v473_acc{};
              tensorforge::intel_esimd::simd<float, 16> v475_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v473_acc += ((static_cast<float>(v475_data[0])) * v35_data);
              v473_acc += ((static_cast<float>(v475_data[1])) * v36_data);
              v473_acc += ((static_cast<float>(v475_data[2])) * v37_data);
              v473_acc += ((static_cast<float>(v475_data[3])) * v38_data);
              v473_acc += ((static_cast<float>(v475_data[4])) * v39_data);
              v473_acc += ((static_cast<float>(v475_data[5])) * v40_data);
              v473_acc += ((static_cast<float>(v475_data[6])) * v41_data);
              v473_acc += ((static_cast<float>(v475_data[7])) * v42_data);
              v473_acc += ((static_cast<float>(v475_data[8])) * v43_data);
              v473_acc += ((static_cast<float>(v475_data[9])) * v44_data);
              v473_acc += ((static_cast<float>(v475_data[10])) * v45_data);
              v473_acc += ((static_cast<float>(v475_data[11])) * v46_data);
              v473_acc += ((static_cast<float>(v475_data[12])) * v47_data);
              v473_acc += ((static_cast<float>(v475_data[13])) * v48_data);
              v473_acc += ((static_cast<float>(v475_data[14])) * v49_data);
              v473_acc += ((static_cast<float>(v475_data[15])) * v50_data);
              ir1.template select<16, 1>(192) = v473_acc;
              tensorforge::intel_esimd::simd<float, 16> v508_acc{};
              tensorforge::intel_esimd::simd<float, 16> v510_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v508_acc += ((static_cast<float>(v510_data[0])) * v35_data);
              v508_acc += ((static_cast<float>(v510_data[1])) * v36_data);
              v508_acc += ((static_cast<float>(v510_data[2])) * v37_data);
              v508_acc += ((static_cast<float>(v510_data[3])) * v38_data);
              v508_acc += ((static_cast<float>(v510_data[4])) * v39_data);
              v508_acc += ((static_cast<float>(v510_data[5])) * v40_data);
              v508_acc += ((static_cast<float>(v510_data[6])) * v41_data);
              v508_acc += ((static_cast<float>(v510_data[7])) * v42_data);
              v508_acc += ((static_cast<float>(v510_data[8])) * v43_data);
              v508_acc += ((static_cast<float>(v510_data[9])) * v44_data);
              v508_acc += ((static_cast<float>(v510_data[10])) * v45_data);
              v508_acc += ((static_cast<float>(v510_data[11])) * v46_data);
              v508_acc += ((static_cast<float>(v510_data[12])) * v47_data);
              v508_acc += ((static_cast<float>(v510_data[13])) * v48_data);
              v508_acc += ((static_cast<float>(v510_data[14])) * v49_data);
              v508_acc += ((static_cast<float>(v510_data[15])) * v50_data);
              ir1.template select<16, 1>(208) = v508_acc;
              tensorforge::intel_esimd::simd<float, 16> v543_acc{};
              tensorforge::intel_esimd::simd<float, 16> v545_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v543_acc += ((static_cast<float>(v545_data[0])) * v35_data);
              v543_acc += ((static_cast<float>(v545_data[1])) * v36_data);
              v543_acc += ((static_cast<float>(v545_data[2])) * v37_data);
              v543_acc += ((static_cast<float>(v545_data[3])) * v38_data);
              v543_acc += ((static_cast<float>(v545_data[4])) * v39_data);
              v543_acc += ((static_cast<float>(v545_data[5])) * v40_data);
              v543_acc += ((static_cast<float>(v545_data[6])) * v41_data);
              v543_acc += ((static_cast<float>(v545_data[7])) * v42_data);
              v543_acc += ((static_cast<float>(v545_data[8])) * v43_data);
              v543_acc += ((static_cast<float>(v545_data[9])) * v44_data);
              v543_acc += ((static_cast<float>(v545_data[10])) * v45_data);
              v543_acc += ((static_cast<float>(v545_data[11])) * v46_data);
              v543_acc += ((static_cast<float>(v545_data[12])) * v47_data);
              v543_acc += ((static_cast<float>(v545_data[13])) * v48_data);
              v543_acc += ((static_cast<float>(v545_data[14])) * v49_data);
              v543_acc += ((static_cast<float>(v545_data[15])) * v50_data);
              ir1.template select<16, 1>(224) = v543_acc;
              tensorforge::intel_esimd::simd<float, 16> v578_acc{};
              tensorforge::intel_esimd::simd<float, 16> v580_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v578_acc += ((static_cast<float>(v580_data[0])) * v35_data);
              v578_acc += ((static_cast<float>(v580_data[1])) * v36_data);
              v578_acc += ((static_cast<float>(v580_data[2])) * v37_data);
              v578_acc += ((static_cast<float>(v580_data[3])) * v38_data);
              v578_acc += ((static_cast<float>(v580_data[4])) * v39_data);
              v578_acc += ((static_cast<float>(v580_data[5])) * v40_data);
              v578_acc += ((static_cast<float>(v580_data[6])) * v41_data);
              v578_acc += ((static_cast<float>(v580_data[7])) * v42_data);
              v578_acc += ((static_cast<float>(v580_data[8])) * v43_data);
              v578_acc += ((static_cast<float>(v580_data[9])) * v44_data);
              v578_acc += ((static_cast<float>(v580_data[10])) * v45_data);
              v578_acc += ((static_cast<float>(v580_data[11])) * v46_data);
              v578_acc += ((static_cast<float>(v580_data[12])) * v47_data);
              v578_acc += ((static_cast<float>(v580_data[13])) * v48_data);
              v578_acc += ((static_cast<float>(v580_data[14])) * v49_data);
              v578_acc += ((static_cast<float>(v580_data[15])) * v50_data);
              ir1.template select<16, 1>(240) = v578_acc;
              // r1 = ir1
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v619_i0 = 0; v619_i0 < 1; ++v619_i0) {
                int32_t v621_a = v619_i0 * 16;
                #pragma unroll
                for (int32_t v620_i1 = 0; v620_i1 < 16; ++v620_i1) {
                  int32_t v622_a = v620_i1 * 16;
                  tensorforge::intel_esimd::simd<float, 16> v624_data(ir1.template select<16, 1>(v622_a));
                  v624_data.copy_to(glb_m0 + ((v621_a + v622_a)));
                }
              }
            }
            else {
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m1);
                #pragma unroll
                for (int32_t v640_i0 = 0; v640_i0 < 1; ++v640_i0) {
                  int32_t v641_lead = v640_i0 * 16;
                  #pragma unroll
                  for (int32_t v642_i1 = 0; v642_i1 < 16; ++v642_i1) {
                    int32_t v644_a = v641_lead + (v642_i1 * 16);
                    tensorforge::intel_esimd::simd<float, 16> v645_data;
                    v645_data.copy_from(wrap_glb_m1 + (v644_a));
                    r0.template select<16, 1>(v644_a) = v645_data;
                  }
                }
              }
            }
            if (allowed_next) {
              // s0 = load{g>s}(wrap_glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v664_ld;
              v664_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v664_ld);
              tensorforge::intel_esimd::simd<float, 64> v665_ld;
              v665_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v665_ld);
              tensorforge::intel_esimd::simd<float, 64> v666_ld;
              v666_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 128));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v666_ld);
              tensorforge::intel_esimd::simd<float, 64> v667_ld;
              v667_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 192));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v667_ld);
            }
            v632_flagWord = v633_flagWordNext;
            v633_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}


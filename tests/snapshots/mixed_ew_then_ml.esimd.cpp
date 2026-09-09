// === base name ===
kernel_e67ccede26e8b078

// === header ===
void launcher_kernel_e67ccede26e8b078(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e67ccede26e8b078(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e67ccede26e8b078(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e67ccede26e8b078(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // TMP = abs(A)
        // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float* __restrict__ s1 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              // s1 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v10_ld;
              v10_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v10_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              float r0[128]{};
              // r0 = abs(glb_m0)
              #pragma unroll
              for (int32_t v12_k1 = 0; v12_k1 < 8; ++v12_k1) {
                tensorforge::intel_esimd::simd<float, 8> v17_data;
                v17_data.copy_from(glb_m0 + ((v12_k1 * 8)));
                (tensorforge::intel_esimd::abs(v17_data)).copy_to(r0 + ((v12_k1 * 16)));
              }
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v31_acc{};
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(s1 + (0_i32));
              v31_acc += ((v35_data[0]) * v23_data);
              v31_acc += ((v35_data[1]) * v24_data);
              v31_acc += ((v35_data[2]) * v25_data);
              v31_acc += ((v35_data[3]) * v26_data);
              v31_acc += ((v35_data[4]) * v27_data);
              v31_acc += ((v35_data[5]) * v28_data);
              v31_acc += ((v35_data[6]) * v29_data);
              v31_acc += ((v35_data[7]) * v30_data);
              v31_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v52_acc{};
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(s1 + (8_i32));
              v52_acc += ((v56_data[0]) * v23_data);
              v52_acc += ((v56_data[1]) * v24_data);
              v52_acc += ((v56_data[2]) * v25_data);
              v52_acc += ((v56_data[3]) * v26_data);
              v52_acc += ((v56_data[4]) * v27_data);
              v52_acc += ((v56_data[5]) * v28_data);
              v52_acc += ((v56_data[6]) * v29_data);
              v52_acc += ((v56_data[7]) * v30_data);
              v52_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v73_acc{};
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(s1 + (16_i32));
              v73_acc += ((v77_data[0]) * v23_data);
              v73_acc += ((v77_data[1]) * v24_data);
              v73_acc += ((v77_data[2]) * v25_data);
              v73_acc += ((v77_data[3]) * v26_data);
              v73_acc += ((v77_data[4]) * v27_data);
              v73_acc += ((v77_data[5]) * v28_data);
              v73_acc += ((v77_data[6]) * v29_data);
              v73_acc += ((v77_data[7]) * v30_data);
              v73_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v94_acc{};
              tensorforge::intel_esimd::simd<float, 16> v98_data;
              v98_data.copy_from(s1 + (24_i32));
              v94_acc += ((v98_data[0]) * v23_data);
              v94_acc += ((v98_data[1]) * v24_data);
              v94_acc += ((v98_data[2]) * v25_data);
              v94_acc += ((v98_data[3]) * v26_data);
              v94_acc += ((v98_data[4]) * v27_data);
              v94_acc += ((v98_data[5]) * v28_data);
              v94_acc += ((v98_data[6]) * v29_data);
              v94_acc += ((v98_data[7]) * v30_data);
              v94_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v115_acc{};
              tensorforge::intel_esimd::simd<float, 16> v119_data;
              v119_data.copy_from(s1 + (32_i32));
              v115_acc += ((v119_data[0]) * v23_data);
              v115_acc += ((v119_data[1]) * v24_data);
              v115_acc += ((v119_data[2]) * v25_data);
              v115_acc += ((v119_data[3]) * v26_data);
              v115_acc += ((v119_data[4]) * v27_data);
              v115_acc += ((v119_data[5]) * v28_data);
              v115_acc += ((v119_data[6]) * v29_data);
              v115_acc += ((v119_data[7]) * v30_data);
              v115_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v136_acc{};
              tensorforge::intel_esimd::simd<float, 16> v140_data;
              v140_data.copy_from(s1 + (40_i32));
              v136_acc += ((v140_data[0]) * v23_data);
              v136_acc += ((v140_data[1]) * v24_data);
              v136_acc += ((v140_data[2]) * v25_data);
              v136_acc += ((v140_data[3]) * v26_data);
              v136_acc += ((v140_data[4]) * v27_data);
              v136_acc += ((v140_data[5]) * v28_data);
              v136_acc += ((v140_data[6]) * v29_data);
              v136_acc += ((v140_data[7]) * v30_data);
              v136_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v157_acc{};
              tensorforge::intel_esimd::simd<float, 16> v161_data;
              v161_data.copy_from(s1 + (48_i32));
              v157_acc += ((v161_data[0]) * v23_data);
              v157_acc += ((v161_data[1]) * v24_data);
              v157_acc += ((v161_data[2]) * v25_data);
              v157_acc += ((v161_data[3]) * v26_data);
              v157_acc += ((v161_data[4]) * v27_data);
              v157_acc += ((v161_data[5]) * v28_data);
              v157_acc += ((v161_data[6]) * v29_data);
              v157_acc += ((v161_data[7]) * v30_data);
              v157_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v178_acc{};
              tensorforge::intel_esimd::simd<float, 16> v182_data;
              v182_data.copy_from(s1 + (56_i32));
              v178_acc += ((v182_data[0]) * v23_data);
              v178_acc += ((v182_data[1]) * v24_data);
              v178_acc += ((v182_data[2]) * v25_data);
              v178_acc += ((v182_data[3]) * v26_data);
              v178_acc += ((v182_data[4]) * v27_data);
              v178_acc += ((v182_data[5]) * v28_data);
              v178_acc += ((v182_data[6]) * v29_data);
              v178_acc += ((v182_data[7]) * v30_data);
              v178_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v199_n1 = 0; v199_n1 < 8; ++v199_n1) {
                int32_t v200_a = v199_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v202_data;
                v202_data.copy_from(ir1 + (v200_a));
                v202_data.copy_to(r1 + (v200_a));
              }
              // glb_m1 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v205_i1 = 0; v205_i1 < 8; ++v205_i1) {
                tensorforge::intel_esimd::simd<float, 8> v208_data;
                v208_data.copy_from(r1 + ((v205_i1 * 16)));
                v208_data.copy_to(glb_m1 + ((v205_i1 * 8)));
              }
            }
          }
        }
      });
    }
  });
}


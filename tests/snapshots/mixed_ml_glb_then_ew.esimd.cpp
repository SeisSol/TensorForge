// === base name ===
kernel_8c9d1a8467

// === header ===
void launcher_kernel_8c9d1a8467(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8c9d1a8467(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8c9d1a8467(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8c9d1a8467(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(M)
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v7_i1 = 0; v7_i1 < 8; ++v7_i1) {
                tensorforge::intel_esimd::simd<float, 8> v12_data;
                v12_data.copy_from(glb_m1 + ((v7_i1 * 8)));
                v12_data.copy_to(r0 + ((v7_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v16_ld;
              v16_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v16_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v27_acc{};
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(s0 + (0_i32));
              v27_acc += ((v31_data[0]) * v19_data);
              v27_acc += ((v31_data[1]) * v20_data);
              v27_acc += ((v31_data[2]) * v21_data);
              v27_acc += ((v31_data[3]) * v22_data);
              v27_acc += ((v31_data[4]) * v23_data);
              v27_acc += ((v31_data[5]) * v24_data);
              v27_acc += ((v31_data[6]) * v25_data);
              v27_acc += ((v31_data[7]) * v26_data);
              v27_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v48_acc{};
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(s0 + (8_i32));
              v48_acc += ((v52_data[0]) * v19_data);
              v48_acc += ((v52_data[1]) * v20_data);
              v48_acc += ((v52_data[2]) * v21_data);
              v48_acc += ((v52_data[3]) * v22_data);
              v48_acc += ((v52_data[4]) * v23_data);
              v48_acc += ((v52_data[5]) * v24_data);
              v48_acc += ((v52_data[6]) * v25_data);
              v48_acc += ((v52_data[7]) * v26_data);
              v48_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v69_acc{};
              tensorforge::intel_esimd::simd<float, 16> v73_data;
              v73_data.copy_from(s0 + (16_i32));
              v69_acc += ((v73_data[0]) * v19_data);
              v69_acc += ((v73_data[1]) * v20_data);
              v69_acc += ((v73_data[2]) * v21_data);
              v69_acc += ((v73_data[3]) * v22_data);
              v69_acc += ((v73_data[4]) * v23_data);
              v69_acc += ((v73_data[5]) * v24_data);
              v69_acc += ((v73_data[6]) * v25_data);
              v69_acc += ((v73_data[7]) * v26_data);
              v69_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v90_acc{};
              tensorforge::intel_esimd::simd<float, 16> v94_data;
              v94_data.copy_from(s0 + (24_i32));
              v90_acc += ((v94_data[0]) * v19_data);
              v90_acc += ((v94_data[1]) * v20_data);
              v90_acc += ((v94_data[2]) * v21_data);
              v90_acc += ((v94_data[3]) * v22_data);
              v90_acc += ((v94_data[4]) * v23_data);
              v90_acc += ((v94_data[5]) * v24_data);
              v90_acc += ((v94_data[6]) * v25_data);
              v90_acc += ((v94_data[7]) * v26_data);
              v90_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v111_acc{};
              tensorforge::intel_esimd::simd<float, 16> v115_data;
              v115_data.copy_from(s0 + (32_i32));
              v111_acc += ((v115_data[0]) * v19_data);
              v111_acc += ((v115_data[1]) * v20_data);
              v111_acc += ((v115_data[2]) * v21_data);
              v111_acc += ((v115_data[3]) * v22_data);
              v111_acc += ((v115_data[4]) * v23_data);
              v111_acc += ((v115_data[5]) * v24_data);
              v111_acc += ((v115_data[6]) * v25_data);
              v111_acc += ((v115_data[7]) * v26_data);
              v111_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v132_acc{};
              tensorforge::intel_esimd::simd<float, 16> v136_data;
              v136_data.copy_from(s0 + (40_i32));
              v132_acc += ((v136_data[0]) * v19_data);
              v132_acc += ((v136_data[1]) * v20_data);
              v132_acc += ((v136_data[2]) * v21_data);
              v132_acc += ((v136_data[3]) * v22_data);
              v132_acc += ((v136_data[4]) * v23_data);
              v132_acc += ((v136_data[5]) * v24_data);
              v132_acc += ((v136_data[6]) * v25_data);
              v132_acc += ((v136_data[7]) * v26_data);
              v132_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v153_acc{};
              tensorforge::intel_esimd::simd<float, 16> v157_data;
              v157_data.copy_from(s0 + (48_i32));
              v153_acc += ((v157_data[0]) * v19_data);
              v153_acc += ((v157_data[1]) * v20_data);
              v153_acc += ((v157_data[2]) * v21_data);
              v153_acc += ((v157_data[3]) * v22_data);
              v153_acc += ((v157_data[4]) * v23_data);
              v153_acc += ((v157_data[5]) * v24_data);
              v153_acc += ((v157_data[6]) * v25_data);
              v153_acc += ((v157_data[7]) * v26_data);
              v153_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v174_acc{};
              tensorforge::intel_esimd::simd<float, 16> v178_data;
              v178_data.copy_from(s0 + (56_i32));
              v174_acc += ((v178_data[0]) * v19_data);
              v174_acc += ((v178_data[1]) * v20_data);
              v174_acc += ((v178_data[2]) * v21_data);
              v174_acc += ((v178_data[3]) * v22_data);
              v174_acc += ((v178_data[4]) * v23_data);
              v174_acc += ((v178_data[5]) * v24_data);
              v174_acc += ((v178_data[6]) * v25_data);
              v174_acc += ((v178_data[7]) * v26_data);
              v174_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v195_n1 = 0; v195_n1 < 8; ++v195_n1) {
                int32_t v196_a = v195_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v198_data;
                v198_data.copy_from(ir1 + (v196_a));
                v198_data.copy_to(r1 + (v196_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v201_i1 = 0; v201_i1 < 8; ++v201_i1) {
                tensorforge::intel_esimd::simd<float, 8> v204_data;
                v204_data.copy_from(r1 + ((v201_i1 * 16)));
                v204_data.copy_to(glb_m0 + ((v201_i1 * 8)));
              }
              // glb_m3 = abs(glb_m0)
              #pragma unroll
              for (int32_t v209_k1 = 0; v209_k1 < 8; ++v209_k1) {
                int32_t v212_a = v209_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v214_data;
                v214_data.copy_from(glb_m0 + (v212_a));
                (tensorforge::intel_esimd::abs(v214_data)).copy_to(glb_m3 + (v212_a));
              }
            }
          }
        }
      });
    }
  });
}


// === base name ===
kernel_3d1d547970e724f8

// === header ===
void launcher_kernel_3d1d547970e724f8(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3d1d547970e724f8(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3d1d547970e724f8(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3d1d547970e724f8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×4(8×4) {0..8}×{0..4} strided
        // m2 8×4(8×4) {0..8}×{0..4} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        // C = abs(TMP)
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float* __restrict__ s0 = &localShrMem0[0];
          float* __restrict__ s1 = &localShrMem0[0];
          float* __restrict__ s2 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v14_i1 = 0; v14_i1 < 8; ++v14_i1) {
                tensorforge::intel_esimd::simd<float, 8> v19_data;
                v19_data.copy_from(glb_m0 + ((v14_i1 * 8)));
                v19_data.copy_to(r0 + ((v14_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v22_ld;
              v22_ld.copy_from(glb_m1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v22_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[64]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v32_acc{};
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(s0 + (0_i32));
              v32_acc += ((v36_data[0]) * v24_data);
              v32_acc += ((v36_data[1]) * v25_data);
              v32_acc += ((v36_data[2]) * v26_data);
              v32_acc += ((v36_data[3]) * v27_data);
              v32_acc += ((v36_data[4]) * v28_data);
              v32_acc += ((v36_data[5]) * v29_data);
              v32_acc += ((v36_data[6]) * v30_data);
              v32_acc += ((v36_data[7]) * v31_data);
              v32_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v53_acc{};
              tensorforge::intel_esimd::simd<float, 16> v57_data;
              v57_data.copy_from(s0 + (8_i32));
              v53_acc += ((v57_data[0]) * v24_data);
              v53_acc += ((v57_data[1]) * v25_data);
              v53_acc += ((v57_data[2]) * v26_data);
              v53_acc += ((v57_data[3]) * v27_data);
              v53_acc += ((v57_data[4]) * v28_data);
              v53_acc += ((v57_data[5]) * v29_data);
              v53_acc += ((v57_data[6]) * v30_data);
              v53_acc += ((v57_data[7]) * v31_data);
              v53_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v74_acc{};
              tensorforge::intel_esimd::simd<float, 16> v78_data;
              v78_data.copy_from(s0 + (16_i32));
              v74_acc += ((v78_data[0]) * v24_data);
              v74_acc += ((v78_data[1]) * v25_data);
              v74_acc += ((v78_data[2]) * v26_data);
              v74_acc += ((v78_data[3]) * v27_data);
              v74_acc += ((v78_data[4]) * v28_data);
              v74_acc += ((v78_data[5]) * v29_data);
              v74_acc += ((v78_data[6]) * v30_data);
              v74_acc += ((v78_data[7]) * v31_data);
              v74_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v95_acc{};
              tensorforge::intel_esimd::simd<float, 16> v99_data;
              v99_data.copy_from(s0 + (24_i32));
              v95_acc += ((v99_data[0]) * v24_data);
              v95_acc += ((v99_data[1]) * v25_data);
              v95_acc += ((v99_data[2]) * v26_data);
              v95_acc += ((v99_data[3]) * v27_data);
              v95_acc += ((v99_data[4]) * v28_data);
              v95_acc += ((v99_data[5]) * v29_data);
              v95_acc += ((v99_data[6]) * v30_data);
              v95_acc += ((v99_data[7]) * v31_data);
              v95_acc.copy_to(r1 + (48));
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v116_i1 = 0; v116_i1 < 4; ++v116_i1) {
                tensorforge::intel_esimd::simd<float, 8> v119_data;
                v119_data.copy_from(r1 + ((v116_i1 * 16)));
                v119_data.copy_to(s1 + ((v116_i1 * 8)));
              }
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v124_ld;
              v124_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v124_ld.copy_to(s2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              float r2[64]{};
              // r2 = +(r0 * s2) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir2[64]{};
              tensorforge::intel_esimd::simd<float, 16> v135_acc{};
              tensorforge::intel_esimd::simd<float, 16> v139_data;
              v139_data.copy_from(s2 + (0_i32));
              v135_acc += ((v139_data[0]) * v24_data);
              v135_acc += ((v139_data[1]) * v25_data);
              v135_acc += ((v139_data[2]) * v26_data);
              v135_acc += ((v139_data[3]) * v27_data);
              v135_acc += ((v139_data[4]) * v28_data);
              v135_acc += ((v139_data[5]) * v29_data);
              v135_acc += ((v139_data[6]) * v30_data);
              v135_acc += ((v139_data[7]) * v31_data);
              v135_acc.copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v156_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data;
              v160_data.copy_from(s2 + (8_i32));
              v156_acc += ((v160_data[0]) * v24_data);
              v156_acc += ((v160_data[1]) * v25_data);
              v156_acc += ((v160_data[2]) * v26_data);
              v156_acc += ((v160_data[3]) * v27_data);
              v156_acc += ((v160_data[4]) * v28_data);
              v156_acc += ((v160_data[5]) * v29_data);
              v156_acc += ((v160_data[6]) * v30_data);
              v156_acc += ((v160_data[7]) * v31_data);
              v156_acc.copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v177_acc{};
              tensorforge::intel_esimd::simd<float, 16> v181_data;
              v181_data.copy_from(s2 + (16_i32));
              v177_acc += ((v181_data[0]) * v24_data);
              v177_acc += ((v181_data[1]) * v25_data);
              v177_acc += ((v181_data[2]) * v26_data);
              v177_acc += ((v181_data[3]) * v27_data);
              v177_acc += ((v181_data[4]) * v28_data);
              v177_acc += ((v181_data[5]) * v29_data);
              v177_acc += ((v181_data[6]) * v30_data);
              v177_acc += ((v181_data[7]) * v31_data);
              v177_acc.copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v198_acc{};
              tensorforge::intel_esimd::simd<float, 16> v202_data;
              v202_data.copy_from(s2 + (24_i32));
              v198_acc += ((v202_data[0]) * v24_data);
              v198_acc += ((v202_data[1]) * v25_data);
              v198_acc += ((v202_data[2]) * v26_data);
              v198_acc += ((v202_data[3]) * v27_data);
              v198_acc += ((v202_data[4]) * v28_data);
              v198_acc += ((v202_data[5]) * v29_data);
              v198_acc += ((v202_data[6]) * v30_data);
              v198_acc += ((v202_data[7]) * v31_data);
              v198_acc.copy_to(ir2 + (48));
              #pragma unroll
              for (int32_t v219_n1 = 0; v219_n1 < 4; ++v219_n1) {
                int32_t v220_a = v219_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v222_data;
                v222_data.copy_from(ir2 + (v220_a));
                v222_data.copy_to(r2 + (v220_a));
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v225_i1 = 0; v225_i1 < 4; ++v225_i1) {
                tensorforge::intel_esimd::simd<float, 8> v228_data;
                v228_data.copy_from(r2 + ((v225_i1 * 16)));
                v228_data.copy_to(s1 + (((v225_i1 + 4) * 8)));
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v234_k1 = 0; v234_k1 < 8; ++v234_k1) {
                int32_t v237_a = v234_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v239_data;
                v239_data.copy_from(s1 + (v237_a));
                (tensorforge::intel_esimd::abs(v239_data)).copy_to(glb_m3 + (v237_a));
              }
            }
          }
        }
      });
    }
  });
}


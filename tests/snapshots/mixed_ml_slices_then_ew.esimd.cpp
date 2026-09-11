// === base name ===
kernel_b4e03a7c60d1e595

// === header ===
void launcher_kernel_b4e03a7c60d1e595(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_b4e03a7c60d1e595(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b4e03a7c60d1e595(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b4e03a7c60d1e595(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          float * __restrict__ s0 = &localShrMem0[0];
          float * __restrict__ s2 = &localShrMem0[64];
          float * __restrict__ s1 = &localShrMem0[0];
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v5_batchId0 * 64 + 0 + m3_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v18_i1 = 0; v18_i1 < 8; ++v18_i1) {
                tensorforge::intel_esimd::simd<float, 8> v23_data;
                v23_data.copy_from(glb_m0 + ((v18_i1 * 8)));
                v23_data.copy_to(r0 + ((v18_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v26_ld;
              v26_ld.copy_from(glb_m1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v26_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v27_ld.copy_to(s2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[64]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v37_acc{};
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(s0 + (0_i32));
              v37_acc += ((static_cast<float>(v41_data[0])) * v29_data);
              v37_acc += ((static_cast<float>(v41_data[1])) * v30_data);
              v37_acc += ((static_cast<float>(v41_data[2])) * v31_data);
              v37_acc += ((static_cast<float>(v41_data[3])) * v32_data);
              v37_acc += ((static_cast<float>(v41_data[4])) * v33_data);
              v37_acc += ((static_cast<float>(v41_data[5])) * v34_data);
              v37_acc += ((static_cast<float>(v41_data[6])) * v35_data);
              v37_acc += ((static_cast<float>(v41_data[7])) * v36_data);
              v37_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v58_acc{};
              tensorforge::intel_esimd::simd<float, 16> v62_data;
              v62_data.copy_from(s0 + (8_i32));
              v58_acc += ((static_cast<float>(v62_data[0])) * v29_data);
              v58_acc += ((static_cast<float>(v62_data[1])) * v30_data);
              v58_acc += ((static_cast<float>(v62_data[2])) * v31_data);
              v58_acc += ((static_cast<float>(v62_data[3])) * v32_data);
              v58_acc += ((static_cast<float>(v62_data[4])) * v33_data);
              v58_acc += ((static_cast<float>(v62_data[5])) * v34_data);
              v58_acc += ((static_cast<float>(v62_data[6])) * v35_data);
              v58_acc += ((static_cast<float>(v62_data[7])) * v36_data);
              v58_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v79_acc{};
              tensorforge::intel_esimd::simd<float, 16> v83_data;
              v83_data.copy_from(s0 + (16_i32));
              v79_acc += ((static_cast<float>(v83_data[0])) * v29_data);
              v79_acc += ((static_cast<float>(v83_data[1])) * v30_data);
              v79_acc += ((static_cast<float>(v83_data[2])) * v31_data);
              v79_acc += ((static_cast<float>(v83_data[3])) * v32_data);
              v79_acc += ((static_cast<float>(v83_data[4])) * v33_data);
              v79_acc += ((static_cast<float>(v83_data[5])) * v34_data);
              v79_acc += ((static_cast<float>(v83_data[6])) * v35_data);
              v79_acc += ((static_cast<float>(v83_data[7])) * v36_data);
              v79_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v100_acc{};
              tensorforge::intel_esimd::simd<float, 16> v104_data;
              v104_data.copy_from(s0 + (24_i32));
              v100_acc += ((static_cast<float>(v104_data[0])) * v29_data);
              v100_acc += ((static_cast<float>(v104_data[1])) * v30_data);
              v100_acc += ((static_cast<float>(v104_data[2])) * v31_data);
              v100_acc += ((static_cast<float>(v104_data[3])) * v32_data);
              v100_acc += ((static_cast<float>(v104_data[4])) * v33_data);
              v100_acc += ((static_cast<float>(v104_data[5])) * v34_data);
              v100_acc += ((static_cast<float>(v104_data[6])) * v35_data);
              v100_acc += ((static_cast<float>(v104_data[7])) * v36_data);
              v100_acc.copy_to(r1 + (48));
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v121_i1 = 0; v121_i1 < 4; ++v121_i1) {
                tensorforge::intel_esimd::simd<float, 8> v124_data;
                v124_data.copy_from(r1 + ((v121_i1 * 16)));
                v124_data.copy_to(s1 + ((v121_i1 * 8)));
              }
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              float r2[64]{};
              // r2 = +(r0 * s2) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir2[64]{};
              tensorforge::intel_esimd::simd<float, 16> v139_acc{};
              tensorforge::intel_esimd::simd<float, 16> v143_data;
              v143_data.copy_from(s2 + (0_i32));
              v139_acc += ((static_cast<float>(v143_data[0])) * v29_data);
              v139_acc += ((static_cast<float>(v143_data[1])) * v30_data);
              v139_acc += ((static_cast<float>(v143_data[2])) * v31_data);
              v139_acc += ((static_cast<float>(v143_data[3])) * v32_data);
              v139_acc += ((static_cast<float>(v143_data[4])) * v33_data);
              v139_acc += ((static_cast<float>(v143_data[5])) * v34_data);
              v139_acc += ((static_cast<float>(v143_data[6])) * v35_data);
              v139_acc += ((static_cast<float>(v143_data[7])) * v36_data);
              v139_acc.copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v160_acc{};
              tensorforge::intel_esimd::simd<float, 16> v164_data;
              v164_data.copy_from(s2 + (8_i32));
              v160_acc += ((static_cast<float>(v164_data[0])) * v29_data);
              v160_acc += ((static_cast<float>(v164_data[1])) * v30_data);
              v160_acc += ((static_cast<float>(v164_data[2])) * v31_data);
              v160_acc += ((static_cast<float>(v164_data[3])) * v32_data);
              v160_acc += ((static_cast<float>(v164_data[4])) * v33_data);
              v160_acc += ((static_cast<float>(v164_data[5])) * v34_data);
              v160_acc += ((static_cast<float>(v164_data[6])) * v35_data);
              v160_acc += ((static_cast<float>(v164_data[7])) * v36_data);
              v160_acc.copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v181_acc{};
              tensorforge::intel_esimd::simd<float, 16> v185_data;
              v185_data.copy_from(s2 + (16_i32));
              v181_acc += ((static_cast<float>(v185_data[0])) * v29_data);
              v181_acc += ((static_cast<float>(v185_data[1])) * v30_data);
              v181_acc += ((static_cast<float>(v185_data[2])) * v31_data);
              v181_acc += ((static_cast<float>(v185_data[3])) * v32_data);
              v181_acc += ((static_cast<float>(v185_data[4])) * v33_data);
              v181_acc += ((static_cast<float>(v185_data[5])) * v34_data);
              v181_acc += ((static_cast<float>(v185_data[6])) * v35_data);
              v181_acc += ((static_cast<float>(v185_data[7])) * v36_data);
              v181_acc.copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v202_acc{};
              tensorforge::intel_esimd::simd<float, 16> v206_data;
              v206_data.copy_from(s2 + (24_i32));
              v202_acc += ((static_cast<float>(v206_data[0])) * v29_data);
              v202_acc += ((static_cast<float>(v206_data[1])) * v30_data);
              v202_acc += ((static_cast<float>(v206_data[2])) * v31_data);
              v202_acc += ((static_cast<float>(v206_data[3])) * v32_data);
              v202_acc += ((static_cast<float>(v206_data[4])) * v33_data);
              v202_acc += ((static_cast<float>(v206_data[5])) * v34_data);
              v202_acc += ((static_cast<float>(v206_data[6])) * v35_data);
              v202_acc += ((static_cast<float>(v206_data[7])) * v36_data);
              v202_acc.copy_to(ir2 + (48));
              #pragma unroll
              for (int32_t v223_n1 = 0; v223_n1 < 4; ++v223_n1) {
                int32_t v224_a = v223_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v226_data;
                v226_data.copy_from(ir2 + (v224_a));
                v226_data.copy_to(r2 + (v224_a));
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v229_i1 = 0; v229_i1 < 4; ++v229_i1) {
                tensorforge::intel_esimd::simd<float, 8> v232_data;
                v232_data.copy_from(r2 + ((v229_i1 * 16)));
                v232_data.copy_to(s1 + (((v229_i1 + 4) * 8)));
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v238_k1 = 0; v238_k1 < 8; ++v238_k1) {
                int32_t v241_a = v238_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v243_data;
                v243_data.copy_from(s1 + (v241_a));
                (tensorforge::intel_esimd::abs(v243_data)).copy_to(glb_m3 + (v241_a));
              }
            }
          }
        }
      });
    }
  });
}


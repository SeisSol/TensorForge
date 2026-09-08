// === base name ===
kernel_a587425bdd

// === header ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a587425bdd(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a587425bdd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v6_ld;
              v6_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v6_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              float r0[128]{};
              // r0 = abs(glb_m0)
              #pragma unroll
              for (int32_t v8_k1 = 0; v8_k1 < 8; ++v8_k1) {
                tensorforge::intel_esimd::simd<float, 8> v13_data;
                v13_data.copy_from(glb_m0 + ((v8_k1 * 8)));
                (tensorforge::intel_esimd::abs(v13_data)).copy_to(r0 + ((v8_k1 * 16)));
              }
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s1) + None
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
              tensorforge::intel_esimd::simd<float, 16> v28_lin;
              v28_lin.copy_from(s1 + (0 + item.get_local_id(0) * 1));
              float v29_bc = v28_lin[0];
              v27_acc += (v29_bc * v19_data);
              float v31_bc = v28_lin[1];
              v27_acc += (v31_bc * v20_data);
              float v33_bc = v28_lin[2];
              v27_acc += (v33_bc * v21_data);
              float v35_bc = v28_lin[3];
              v27_acc += (v35_bc * v22_data);
              float v37_bc = v28_lin[4];
              v27_acc += (v37_bc * v23_data);
              float v39_bc = v28_lin[5];
              v27_acc += (v39_bc * v24_data);
              float v41_bc = v28_lin[6];
              v27_acc += (v41_bc * v25_data);
              float v43_bc = v28_lin[7];
              v27_acc += (v43_bc * v26_data);
              v27_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v45_acc{};
              v45_acc += (v29_bc * v19_data);
              v45_acc += (v31_bc * v20_data);
              v45_acc += (v33_bc * v21_data);
              v45_acc += (v35_bc * v22_data);
              v45_acc += (v37_bc * v23_data);
              v45_acc += (v39_bc * v24_data);
              v45_acc += (v41_bc * v25_data);
              v45_acc += (v43_bc * v26_data);
              v45_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v63_acc{};
              v63_acc += (v29_bc * v19_data);
              v63_acc += (v31_bc * v20_data);
              v63_acc += (v33_bc * v21_data);
              v63_acc += (v35_bc * v22_data);
              v63_acc += (v37_bc * v23_data);
              v63_acc += (v39_bc * v24_data);
              v63_acc += (v41_bc * v25_data);
              v63_acc += (v43_bc * v26_data);
              v63_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v81_acc{};
              v81_acc += (v29_bc * v19_data);
              v81_acc += (v31_bc * v20_data);
              v81_acc += (v33_bc * v21_data);
              v81_acc += (v35_bc * v22_data);
              v81_acc += (v37_bc * v23_data);
              v81_acc += (v39_bc * v24_data);
              v81_acc += (v41_bc * v25_data);
              v81_acc += (v43_bc * v26_data);
              v81_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v99_acc{};
              v99_acc += (v29_bc * v19_data);
              v99_acc += (v31_bc * v20_data);
              v99_acc += (v33_bc * v21_data);
              v99_acc += (v35_bc * v22_data);
              v99_acc += (v37_bc * v23_data);
              v99_acc += (v39_bc * v24_data);
              v99_acc += (v41_bc * v25_data);
              v99_acc += (v43_bc * v26_data);
              v99_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v117_acc{};
              v117_acc += (v29_bc * v19_data);
              v117_acc += (v31_bc * v20_data);
              v117_acc += (v33_bc * v21_data);
              v117_acc += (v35_bc * v22_data);
              v117_acc += (v37_bc * v23_data);
              v117_acc += (v39_bc * v24_data);
              v117_acc += (v41_bc * v25_data);
              v117_acc += (v43_bc * v26_data);
              v117_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v135_acc{};
              v135_acc += (v29_bc * v19_data);
              v135_acc += (v31_bc * v20_data);
              v135_acc += (v33_bc * v21_data);
              v135_acc += (v35_bc * v22_data);
              v135_acc += (v37_bc * v23_data);
              v135_acc += (v39_bc * v24_data);
              v135_acc += (v41_bc * v25_data);
              v135_acc += (v43_bc * v26_data);
              v135_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v153_acc{};
              v153_acc += (v29_bc * v19_data);
              v153_acc += (v31_bc * v20_data);
              v153_acc += (v33_bc * v21_data);
              v153_acc += (v35_bc * v22_data);
              v153_acc += (v37_bc * v23_data);
              v153_acc += (v39_bc * v24_data);
              v153_acc += (v41_bc * v25_data);
              v153_acc += (v43_bc * v26_data);
              v153_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v171_n1 = 0; v171_n1 < 8; ++v171_n1) {
                int32_t v172_a = v171_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v174_data;
                v174_data.copy_from(ir1 + (v172_a));
                v174_data.copy_to(r1 + (v172_a));
              }
              // glb_m1 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v177_i1 = 0; v177_i1 < 8; ++v177_i1) {
                tensorforge::intel_esimd::simd<float, 8> v180_data;
                v180_data.copy_from(r1 + ((v177_i1 * 16)));
                v180_data.copy_to(glb_m1 + ((v177_i1 * 8)));
              }
            }
          }
        }
      });
    }
  });
}


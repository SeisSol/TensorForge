// === base name ===
kernel_8ab0d0fff0

// === header ===
void launcher_kernel_8ab0d0fff0(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8ab0d0fff0(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8ab0d0fff0(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8ab0d0fff0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(TMP)
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 8; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 8> v11_data;
                v11_data.copy_from(glb_m0 + ((v6_i1 * 8)));
                v11_data.copy_to(r0 + ((v6_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v15_ld;
              v15_ld.copy_from(glb_m1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v15_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v17_data;
              v17_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v18_data;
              v18_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v25_acc{};
              tensorforge::intel_esimd::simd<float, 16> v26_lin;
              v26_lin.copy_from(s0 + (0 + item.get_local_id(0) * 1));
              float v27_bc = v26_lin[0];
              v25_acc += (v27_bc * v17_data);
              float v29_bc = v26_lin[1];
              v25_acc += (v29_bc * v18_data);
              float v31_bc = v26_lin[2];
              v25_acc += (v31_bc * v19_data);
              float v33_bc = v26_lin[3];
              v25_acc += (v33_bc * v20_data);
              float v35_bc = v26_lin[4];
              v25_acc += (v35_bc * v21_data);
              float v37_bc = v26_lin[5];
              v25_acc += (v37_bc * v22_data);
              float v39_bc = v26_lin[6];
              v25_acc += (v39_bc * v23_data);
              float v41_bc = v26_lin[7];
              v25_acc += (v41_bc * v24_data);
              v25_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v43_acc{};
              v43_acc += (v27_bc * v17_data);
              v43_acc += (v29_bc * v18_data);
              v43_acc += (v31_bc * v19_data);
              v43_acc += (v33_bc * v20_data);
              v43_acc += (v35_bc * v21_data);
              v43_acc += (v37_bc * v22_data);
              v43_acc += (v39_bc * v23_data);
              v43_acc += (v41_bc * v24_data);
              v43_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v61_acc{};
              v61_acc += (v27_bc * v17_data);
              v61_acc += (v29_bc * v18_data);
              v61_acc += (v31_bc * v19_data);
              v61_acc += (v33_bc * v20_data);
              v61_acc += (v35_bc * v21_data);
              v61_acc += (v37_bc * v22_data);
              v61_acc += (v39_bc * v23_data);
              v61_acc += (v41_bc * v24_data);
              v61_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v79_acc{};
              v79_acc += (v27_bc * v17_data);
              v79_acc += (v29_bc * v18_data);
              v79_acc += (v31_bc * v19_data);
              v79_acc += (v33_bc * v20_data);
              v79_acc += (v35_bc * v21_data);
              v79_acc += (v37_bc * v22_data);
              v79_acc += (v39_bc * v23_data);
              v79_acc += (v41_bc * v24_data);
              v79_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v97_acc{};
              v97_acc += (v27_bc * v17_data);
              v97_acc += (v29_bc * v18_data);
              v97_acc += (v31_bc * v19_data);
              v97_acc += (v33_bc * v20_data);
              v97_acc += (v35_bc * v21_data);
              v97_acc += (v37_bc * v22_data);
              v97_acc += (v39_bc * v23_data);
              v97_acc += (v41_bc * v24_data);
              v97_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v115_acc{};
              v115_acc += (v27_bc * v17_data);
              v115_acc += (v29_bc * v18_data);
              v115_acc += (v31_bc * v19_data);
              v115_acc += (v33_bc * v20_data);
              v115_acc += (v35_bc * v21_data);
              v115_acc += (v37_bc * v22_data);
              v115_acc += (v39_bc * v23_data);
              v115_acc += (v41_bc * v24_data);
              v115_acc.copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v133_acc{};
              v133_acc += (v27_bc * v17_data);
              v133_acc += (v29_bc * v18_data);
              v133_acc += (v31_bc * v19_data);
              v133_acc += (v33_bc * v20_data);
              v133_acc += (v35_bc * v21_data);
              v133_acc += (v37_bc * v22_data);
              v133_acc += (v39_bc * v23_data);
              v133_acc += (v41_bc * v24_data);
              v133_acc.copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v151_acc{};
              v151_acc += (v27_bc * v17_data);
              v151_acc += (v29_bc * v18_data);
              v151_acc += (v31_bc * v19_data);
              v151_acc += (v33_bc * v20_data);
              v151_acc += (v35_bc * v21_data);
              v151_acc += (v37_bc * v22_data);
              v151_acc += (v39_bc * v23_data);
              v151_acc += (v41_bc * v24_data);
              v151_acc.copy_to(r1 + (112));
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v170_i1 = 0; v170_i1 < 8; ++v170_i1) {
                tensorforge::intel_esimd::simd<float, 8> v173_data;
                v173_data.copy_from(r1 + ((v170_i1 * 16)));
                v173_data.copy_to(s1 + ((v170_i1 * 8)));
              }
              // glb_m2 = abs(s1)
              #pragma unroll
              for (int32_t v178_k1 = 0; v178_k1 < 8; ++v178_k1) {
                int32_t v181_a = v178_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v183_data;
                v183_data.copy_from(s1 + (v181_a));
                (tensorforge::intel_esimd::abs(v183_data)).copy_to(glb_m2 + (v181_a));
              }
            }
          }
        }
      });
    }
  });
}


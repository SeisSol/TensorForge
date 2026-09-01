// === base name ===
kernel_49337a255f

// === header ===
void launcher_kernel_49337a255f(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_49337a255f(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_49337a255f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_49337a255f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // TMP = +(A, dims=[1])
        // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8(8) {0..8} pointer_based({0..8})[0]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float r1[128]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 8; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 8> v11_data;
                v11_data.copy_from(glb_m2 + ((v6_i1 * 8)));
                v11_data.copy_to(r1 + ((v6_i1 * 16)));
              }
              float r0[16]{};
              // r0 = +(glb_m0, dims=[1])
              tensorforge::intel_esimd::simd<float, 8> v16_acc0(0.0f);
              #pragma unroll
              for (int32_t v15_r1 = 0; v15_r1 < 8; ++v15_r1) {
                tensorforge::intel_esimd::simd<float, 8> v21_data;
                v21_data.copy_from(glb_m0 + ((v15_r1 * 8)));
                v16_acc0 = (v16_acc0 + v21_data);
              }
              v16_acc0.copy_to(r0 + (0));
              // wait(r1 = load{g>r}(glb_m2););
              float r2[128]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] []
              float ir2[128]{};
              tensorforge::intel_esimd::simd<float, 8> v26_data;
              v26_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 8> v27_data;
              v27_data.copy_from(r1 + (0));
              tensorforge::intel_esimd::simd<float, 8> v29_data;
              v29_data.copy_from(ir2 + (0));
              (v29_data + (v26_data * v27_data)).copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 8> v32_data;
              v32_data.copy_from(r1 + (16));
              tensorforge::intel_esimd::simd<float, 8> v34_data;
              v34_data.copy_from(ir2 + (16));
              (v34_data + (v26_data * v32_data)).copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 8> v37_data;
              v37_data.copy_from(r1 + (32));
              tensorforge::intel_esimd::simd<float, 8> v39_data;
              v39_data.copy_from(ir2 + (32));
              (v39_data + (v26_data * v37_data)).copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 8> v42_data;
              v42_data.copy_from(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v44_data;
              v44_data.copy_from(ir2 + (48));
              (v44_data + (v26_data * v42_data)).copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 8> v47_data;
              v47_data.copy_from(r1 + (64));
              tensorforge::intel_esimd::simd<float, 8> v49_data;
              v49_data.copy_from(ir2 + (64));
              (v49_data + (v26_data * v47_data)).copy_to(ir2 + (64));
              tensorforge::intel_esimd::simd<float, 8> v52_data;
              v52_data.copy_from(r1 + (80));
              tensorforge::intel_esimd::simd<float, 8> v54_data;
              v54_data.copy_from(ir2 + (80));
              (v54_data + (v26_data * v52_data)).copy_to(ir2 + (80));
              tensorforge::intel_esimd::simd<float, 8> v57_data;
              v57_data.copy_from(r1 + (96));
              tensorforge::intel_esimd::simd<float, 8> v59_data;
              v59_data.copy_from(ir2 + (96));
              (v59_data + (v26_data * v57_data)).copy_to(ir2 + (96));
              tensorforge::intel_esimd::simd<float, 8> v62_data;
              v62_data.copy_from(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v64_data;
              v64_data.copy_from(ir2 + (112));
              (v64_data + (v26_data * v62_data)).copy_to(ir2 + (112));
              #pragma unroll
              for (int32_t v66_n1 = 0; v66_n1 < 8; ++v66_n1) {
                int32_t v67_a = v66_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v69_data;
                v69_data.copy_from(ir2 + (v67_a));
                v69_data.copy_to(r2 + (v67_a));
              }
              // glb_m1 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v72_i1 = 0; v72_i1 < 8; ++v72_i1) {
                tensorforge::intel_esimd::simd<float, 8> v75_data;
                v75_data.copy_from(r2 + ((v72_i1 * 16)));
                v75_data.copy_to(glb_m1 + ((v72_i1 * 8)));
              }
            }
          }
        }
      });
    }
  });
}


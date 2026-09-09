// === base name ===
kernel_129a0de0f2786d11

// === header ===
void launcher_kernel_129a0de0f2786d11(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_129a0de0f2786d11(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_129a0de0f2786d11(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_129a0de0f2786d11(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float r1[128]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v10_i1 = 0; v10_i1 < 8; ++v10_i1) {
                tensorforge::intel_esimd::simd<float, 8> v15_data;
                v15_data.copy_from(glb_m2 + ((v10_i1 * 8)));
                v15_data.copy_to(r1 + ((v10_i1 * 16)));
              }
              float r0[16]{};
              // r0 = +(glb_m0, dims=[1])
              tensorforge::intel_esimd::simd<float, 8> v20_acc0(0.0f);
              #pragma unroll
              for (int32_t v19_r1 = 0; v19_r1 < 8; ++v19_r1) {
                tensorforge::intel_esimd::simd<float, 8> v25_data;
                v25_data.copy_from(glb_m0 + ((v19_r1 * 8)));
                v20_acc0 = (v20_acc0 + v25_data);
              }
              v20_acc0.copy_to(r0 + (0));
              // wait(r1 = load{g>r}(glb_m2););
              float r2[128]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] []
              float ir2[128]{};
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v31_acc{};
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r1 + (0));
              v31_acc += ((v32_data[0]) * v30_data);
              v31_acc.copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v35_acc{};
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r1 + (16));
              v35_acc += ((v36_data[0]) * v30_data);
              v35_acc.copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v39_acc{};
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r1 + (32));
              v39_acc += ((v40_data[0]) * v30_data);
              v39_acc.copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v43_acc{};
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r1 + (48));
              v43_acc += ((v44_data[0]) * v30_data);
              v43_acc.copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v47_acc{};
              tensorforge::intel_esimd::simd<float, 16> v48_data;
              v48_data.copy_from(r1 + (64));
              v47_acc += ((v48_data[0]) * v30_data);
              v47_acc.copy_to(ir2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v51_acc{};
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(r1 + (80));
              v51_acc += ((v52_data[0]) * v30_data);
              v51_acc.copy_to(ir2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v55_acc{};
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(r1 + (96));
              v55_acc += ((v56_data[0]) * v30_data);
              v55_acc.copy_to(ir2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v59_acc{};
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(r1 + (112));
              v59_acc += ((v60_data[0]) * v30_data);
              v59_acc.copy_to(ir2 + (112));
              #pragma unroll
              for (int32_t v63_n1 = 0; v63_n1 < 8; ++v63_n1) {
                int32_t v64_a = v63_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v66_data;
                v66_data.copy_from(ir2 + (v64_a));
                v66_data.copy_to(r2 + (v64_a));
              }
              // glb_m1 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v69_i1 = 0; v69_i1 < 8; ++v69_i1) {
                tensorforge::intel_esimd::simd<float, 8> v72_data;
                v72_data.copy_from(r2 + ((v69_i1 * 16)));
                v72_data.copy_to(glb_m1 + ((v69_i1 * 8)));
              }
            }
          }
        }
      });
    }
  });
}


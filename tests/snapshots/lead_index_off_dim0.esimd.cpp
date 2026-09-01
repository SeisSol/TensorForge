// === base name ===
kernel_75d3097b00

// === header ===
void launcher_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_75d3097b00(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_75d3097b00(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (128, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 20×9(20×9) {0..20}×{0..9} strided
        // m1 1×20(1×20) {0..1}×{0..20} strided
        // m2 1×9(1×9) {0..1}×{0..9} strided
        // m0 20×9(20×9) {0..20}×{0..9} strided({0..20}×{0..9})[0, 1] = m1 1×20(1×20) {0..1}×{0..20} strided({0..1}×{0..20})[-1, 0]×m2 1×9(1×9) {0..1}×{0..9} strided({0..1}×{0..9})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[16];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 180 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 20 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 9 + 0 + m2_extraOffset];
              float r0[32]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
                tensorforge::intel_esimd::simd<float, 20> v10_data;
                v10_data.copy_from(glb_m1 + (v6_i0));
                v10_data.copy_to(r0 + (v6_i0));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              if (item.get_local_id(0) < 9) {
                tensorforge::intel_esimd::simd<float, 32> v13_ld;
                v13_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 0));
                v13_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[288]{};
              // r1 = +(r0 * s0) + None
              // [(0, 20), (0, 9)] [(0, 1)]
              float ir1[288]{};
              tensorforge::intel_esimd::simd<float, 20> v16_data;
              v16_data.copy_from(r0 + (0));
              float v17_data = s0[0];
              tensorforge::intel_esimd::simd<float, 20> v19_data;
              v19_data.copy_from(ir1 + (0));
              (v19_data + (v16_data * v17_data)).copy_to(ir1 + (0));
              float v22_data = s0[1];
              tensorforge::intel_esimd::simd<float, 20> v24_data;
              v24_data.copy_from(ir1 + (32));
              (v24_data + (v16_data * v22_data)).copy_to(ir1 + (32));
              float v27_data = s0[2];
              tensorforge::intel_esimd::simd<float, 20> v29_data;
              v29_data.copy_from(ir1 + (64));
              (v29_data + (v16_data * v27_data)).copy_to(ir1 + (64));
              float v32_data = s0[3];
              tensorforge::intel_esimd::simd<float, 20> v34_data;
              v34_data.copy_from(ir1 + (96));
              (v34_data + (v16_data * v32_data)).copy_to(ir1 + (96));
              float v37_data = s0[4];
              tensorforge::intel_esimd::simd<float, 20> v39_data;
              v39_data.copy_from(ir1 + (128));
              (v39_data + (v16_data * v37_data)).copy_to(ir1 + (128));
              float v42_data = s0[5];
              tensorforge::intel_esimd::simd<float, 20> v44_data;
              v44_data.copy_from(ir1 + (160));
              (v44_data + (v16_data * v42_data)).copy_to(ir1 + (160));
              float v47_data = s0[6];
              tensorforge::intel_esimd::simd<float, 20> v49_data;
              v49_data.copy_from(ir1 + (192));
              (v49_data + (v16_data * v47_data)).copy_to(ir1 + (192));
              float v52_data = s0[7];
              tensorforge::intel_esimd::simd<float, 20> v54_data;
              v54_data.copy_from(ir1 + (224));
              (v54_data + (v16_data * v52_data)).copy_to(ir1 + (224));
              float v57_data = s0[8];
              tensorforge::intel_esimd::simd<float, 20> v59_data;
              v59_data.copy_from(ir1 + (256));
              (v59_data + (v16_data * v57_data)).copy_to(ir1 + (256));
              #pragma unroll
              for (int32_t v61_n1 = 0; v61_n1 < 9; ++v61_n1) {
                int32_t v62_a = v61_n1 * 32;
                tensorforge::intel_esimd::simd<float, 20> v64_data;
                v64_data.copy_from(ir1 + (v62_a));
                v64_data.copy_to(r1 + (v62_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v67_i1 = 0; v67_i1 < 9; ++v67_i1) {
                tensorforge::intel_esimd::simd<float, 20> v70_data;
                v70_data.copy_from(r1 + ((v67_i1 * 32)));
                v70_data.copy_to(glb_m0 + ((v67_i1 * 20)));
              }
            }
          }
        }
      });
    }
  });
}


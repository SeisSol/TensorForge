// === base name ===
kernel_a7d5d30824

// === header ===
void launcher_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a7d5d30824(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a7d5d30824(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16(16) {0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m0 16(16) {0..16} strided({0..16})[0] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              float r0[1]{};
              // r0 = +(glb_m1) + None
              // [(0, 16)] [(0, 16)]
              float ir0[1]{};
              int32_t v6_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v10_data;
              v10_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v11_data;
              v11_data.copy_from(ir0 + (0));
              (v11_data + v10_data).copy_to(ir0 + (0));
              int32_t v15_a = 0_i32 + 16;
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(ir0 + (0));
              (v20_data + v19_data).copy_to(ir0 + (0));
              int32_t v24_a = 0_i32 + 32;
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(ir0 + (0));
              (v29_data + v28_data).copy_to(ir0 + (0));
              int32_t v33_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(ir0 + (0));
              (v38_data + v37_data).copy_to(ir0 + (0));
              int32_t v42_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(ir0 + (0));
              (v47_data + v46_data).copy_to(ir0 + (0));
              int32_t v51_a = 0_i32 + 80;
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(ir0 + (0));
              (v56_data + v55_data).copy_to(ir0 + (0));
              int32_t v60_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v64_data;
              v64_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data;
              v65_data.copy_from(ir0 + (0));
              (v65_data + v64_data).copy_to(ir0 + (0));
              int32_t v69_a = 0_i32 + 112;
              tensorforge::intel_esimd::simd<float, 16> v73_data;
              v73_data.copy_from(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<float, 16> v74_data;
              v74_data.copy_from(ir0 + (0));
              (v74_data + v73_data).copy_to(ir0 + (0));
              int32_t v78_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 16> v82_data;
              v82_data.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v83_data;
              v83_data.copy_from(ir0 + (0));
              (v83_data + v82_data).copy_to(ir0 + (0));
              int32_t v87_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v91_data;
              v91_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(ir0 + (0));
              (v92_data + v91_data).copy_to(ir0 + (0));
              int32_t v96_a = 0_i32 + 160;
              tensorforge::intel_esimd::simd<float, 16> v100_data;
              v100_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v101_data;
              v101_data.copy_from(ir0 + (0));
              (v101_data + v100_data).copy_to(ir0 + (0));
              int32_t v105_a = 0_i32 + 176;
              tensorforge::intel_esimd::simd<float, 16> v109_data;
              v109_data.copy_from(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<float, 16> v110_data;
              v110_data.copy_from(ir0 + (0));
              (v110_data + v109_data).copy_to(ir0 + (0));
              int32_t v114_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v118_data;
              v118_data.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v119_data;
              v119_data.copy_from(ir0 + (0));
              (v119_data + v118_data).copy_to(ir0 + (0));
              int32_t v123_a = 0_i32 + 208;
              tensorforge::intel_esimd::simd<float, 16> v127_data;
              v127_data.copy_from(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<float, 16> v128_data;
              v128_data.copy_from(ir0 + (0));
              (v128_data + v127_data).copy_to(ir0 + (0));
              int32_t v132_a = 0_i32 + 224;
              tensorforge::intel_esimd::simd<float, 16> v136_data;
              v136_data.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<float, 16> v137_data;
              v137_data.copy_from(ir0 + (0));
              (v137_data + v136_data).copy_to(ir0 + (0));
              int32_t v141_a = 0_i32 + 240;
              tensorforge::intel_esimd::simd<float, 16> v145_data;
              v145_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v146_data;
              v146_data.copy_from(ir0 + (0));
              (v146_data + v145_data).copy_to(ir0 + (0));
              #pragma unroll
              for (int32_t v148_n0 = 0; v148_n0 < 1; ++v148_n0) {
                tensorforge::intel_esimd::simd<float, 16> v149_data;
                v149_data.copy_from(ir0 + (v148_n0));
                v149_data.copy_to(r0 + (v148_n0));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v150_i0 = 0; v150_i0 < 1; ++v150_i0) {
                tensorforge::intel_esimd::simd<float, 16> v151_data;
                v151_data.copy_from(r0 + (v150_i0));
                v151_data.copy_to(glb_m0 + ((v150_i0 * 16)));
              }
            }
          }
        }
      });
    }
  });
}


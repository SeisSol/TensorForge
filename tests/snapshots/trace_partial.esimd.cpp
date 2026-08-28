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
              tensorforge::intel_esimd::simd<int32_t, 16> v4_lead = tensorforge::intel_esimd::simd<int32_t, 16>(0, 1);
              int32_t v7_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v11_data;
              v11_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v12_data;
              v12_data.copy_from(ir0 + (0));
              (v12_data + v11_data).copy_to(ir0 + (0));
              int32_t v17_a = 0_i32 + 16;
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(ir0 + (0));
              (v22_data + v21_data).copy_to(ir0 + (0));
              int32_t v27_a = 0_i32 + 32;
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(ir0 + (0));
              (v32_data + v31_data).copy_to(ir0 + (0));
              int32_t v37_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(ir0 + (0));
              (v42_data + v41_data).copy_to(ir0 + (0));
              int32_t v47_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(ir0 + (0));
              (v52_data + v51_data).copy_to(ir0 + (0));
              int32_t v57_a = 0_i32 + 80;
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data;
              v62_data.copy_from(ir0 + (0));
              (v62_data + v61_data).copy_to(ir0 + (0));
              int32_t v67_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_data;
              v72_data.copy_from(ir0 + (0));
              (v72_data + v71_data).copy_to(ir0 + (0));
              int32_t v77_a = 0_i32 + 112;
              tensorforge::intel_esimd::simd<float, 16> v81_data;
              v81_data.copy_from(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<float, 16> v82_data;
              v82_data.copy_from(ir0 + (0));
              (v82_data + v81_data).copy_to(ir0 + (0));
              int32_t v87_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 16> v91_data;
              v91_data.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(ir0 + (0));
              (v92_data + v91_data).copy_to(ir0 + (0));
              int32_t v97_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v101_data;
              v101_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v102_data;
              v102_data.copy_from(ir0 + (0));
              (v102_data + v101_data).copy_to(ir0 + (0));
              int32_t v107_a = 0_i32 + 160;
              tensorforge::intel_esimd::simd<float, 16> v111_data;
              v111_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v112_data;
              v112_data.copy_from(ir0 + (0));
              (v112_data + v111_data).copy_to(ir0 + (0));
              int32_t v117_a = 0_i32 + 176;
              tensorforge::intel_esimd::simd<float, 16> v121_data;
              v121_data.copy_from(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<float, 16> v122_data;
              v122_data.copy_from(ir0 + (0));
              (v122_data + v121_data).copy_to(ir0 + (0));
              int32_t v127_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v131_data;
              v131_data.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v132_data;
              v132_data.copy_from(ir0 + (0));
              (v132_data + v131_data).copy_to(ir0 + (0));
              int32_t v137_a = 0_i32 + 208;
              tensorforge::intel_esimd::simd<float, 16> v141_data;
              v141_data.copy_from(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<float, 16> v142_data;
              v142_data.copy_from(ir0 + (0));
              (v142_data + v141_data).copy_to(ir0 + (0));
              int32_t v147_a = 0_i32 + 224;
              tensorforge::intel_esimd::simd<float, 16> v151_data;
              v151_data.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<float, 16> v152_data;
              v152_data.copy_from(ir0 + (0));
              (v152_data + v151_data).copy_to(ir0 + (0));
              int32_t v157_a = 0_i32 + 240;
              tensorforge::intel_esimd::simd<float, 16> v161_data;
              v161_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v162_data;
              v162_data.copy_from(ir0 + (0));
              (v162_data + v161_data).copy_to(ir0 + (0));
              #pragma unroll
              for (int32_t v165_n0 = 0; v165_n0 < 1; ++v165_n0) {
                tensorforge::intel_esimd::simd<float, 16> v166_data;
                v166_data.copy_from(ir0 + (v165_n0));
                v166_data.copy_to(r0 + (v165_n0));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v168_i0 = 0; v168_i0 < 1; ++v168_i0) {
                tensorforge::intel_esimd::simd<float, 16> v169_data;
                v169_data.copy_from(r0 + (v168_i0));
                v169_data.copy_to(glb_m0 + ((v168_i0 * 16)));
              }
            }
          }
        }
      });
    }
  });
}


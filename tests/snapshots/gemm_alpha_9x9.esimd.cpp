// === base name ===
kernel_2faad4dc2e7974d3

// === header ===
void launcher_kernel_2faad4dc2e7974d3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_2faad4dc2e7974d3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2faad4dc2e7974d3(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2faad4dc2e7974d3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 9×9(9×9) {0..9}×{0..9} strided
        // m1 9×9(9×9) {0..9}×{0..9} strided
        // m2 9×9(9×9) {0..9}×{0..9} strided
        // m3 ()  scalar
        // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float r0[144]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v11_i1 = 0; v11_i1 < 9; ++v11_i1) {
                tensorforge::intel_esimd::simd<float, 9> v16_data;
                v16_data.copy_from(glb_m1 + ((v11_i1 * 9)));
                v16_data.copy_to(r0 + ((v11_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v19_ld;
              v19_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v19_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 16> v20_ld;
              v20_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              v20_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              if (item.get_local_id(0) < 1) {
                tensorforge::intel_esimd::simd<float, 16> v21_ld;
                v21_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 80));
                v21_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 80));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[144]{};
              // r1 = +(r0 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir1[144]{};
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
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v33_acc{};
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(s0 + (0_i32));
              v33_acc += ((v37_data[0]) * v24_data);
              v33_acc += ((v37_data[1]) * v25_data);
              v33_acc += ((v37_data[2]) * v26_data);
              v33_acc += ((v37_data[3]) * v27_data);
              v33_acc += ((v37_data[4]) * v28_data);
              v33_acc += ((v37_data[5]) * v29_data);
              v33_acc += ((v37_data[6]) * v30_data);
              v33_acc += ((v37_data[7]) * v31_data);
              v33_acc += ((v37_data[8]) * v32_data);
              v33_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v56_acc{};
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(s0 + (9_i32));
              v56_acc += ((v60_data[0]) * v24_data);
              v56_acc += ((v60_data[1]) * v25_data);
              v56_acc += ((v60_data[2]) * v26_data);
              v56_acc += ((v60_data[3]) * v27_data);
              v56_acc += ((v60_data[4]) * v28_data);
              v56_acc += ((v60_data[5]) * v29_data);
              v56_acc += ((v60_data[6]) * v30_data);
              v56_acc += ((v60_data[7]) * v31_data);
              v56_acc += ((v60_data[8]) * v32_data);
              v56_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v79_acc{};
              tensorforge::intel_esimd::simd<float, 16> v83_data;
              v83_data.copy_from(s0 + (18_i32));
              v79_acc += ((v83_data[0]) * v24_data);
              v79_acc += ((v83_data[1]) * v25_data);
              v79_acc += ((v83_data[2]) * v26_data);
              v79_acc += ((v83_data[3]) * v27_data);
              v79_acc += ((v83_data[4]) * v28_data);
              v79_acc += ((v83_data[5]) * v29_data);
              v79_acc += ((v83_data[6]) * v30_data);
              v79_acc += ((v83_data[7]) * v31_data);
              v79_acc += ((v83_data[8]) * v32_data);
              v79_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v102_acc{};
              tensorforge::intel_esimd::simd<float, 16> v106_data;
              v106_data.copy_from(s0 + (27_i32));
              v102_acc += ((v106_data[0]) * v24_data);
              v102_acc += ((v106_data[1]) * v25_data);
              v102_acc += ((v106_data[2]) * v26_data);
              v102_acc += ((v106_data[3]) * v27_data);
              v102_acc += ((v106_data[4]) * v28_data);
              v102_acc += ((v106_data[5]) * v29_data);
              v102_acc += ((v106_data[6]) * v30_data);
              v102_acc += ((v106_data[7]) * v31_data);
              v102_acc += ((v106_data[8]) * v32_data);
              v102_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v125_acc{};
              tensorforge::intel_esimd::simd<float, 16> v129_data;
              v129_data.copy_from(s0 + (36_i32));
              v125_acc += ((v129_data[0]) * v24_data);
              v125_acc += ((v129_data[1]) * v25_data);
              v125_acc += ((v129_data[2]) * v26_data);
              v125_acc += ((v129_data[3]) * v27_data);
              v125_acc += ((v129_data[4]) * v28_data);
              v125_acc += ((v129_data[5]) * v29_data);
              v125_acc += ((v129_data[6]) * v30_data);
              v125_acc += ((v129_data[7]) * v31_data);
              v125_acc += ((v129_data[8]) * v32_data);
              v125_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v148_acc{};
              tensorforge::intel_esimd::simd<float, 16> v152_data;
              v152_data.copy_from(s0 + (45_i32));
              v148_acc += ((v152_data[0]) * v24_data);
              v148_acc += ((v152_data[1]) * v25_data);
              v148_acc += ((v152_data[2]) * v26_data);
              v148_acc += ((v152_data[3]) * v27_data);
              v148_acc += ((v152_data[4]) * v28_data);
              v148_acc += ((v152_data[5]) * v29_data);
              v148_acc += ((v152_data[6]) * v30_data);
              v148_acc += ((v152_data[7]) * v31_data);
              v148_acc += ((v152_data[8]) * v32_data);
              v148_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v171_acc{};
              tensorforge::intel_esimd::simd<float, 16> v175_data;
              v175_data.copy_from(s0 + (54_i32));
              v171_acc += ((v175_data[0]) * v24_data);
              v171_acc += ((v175_data[1]) * v25_data);
              v171_acc += ((v175_data[2]) * v26_data);
              v171_acc += ((v175_data[3]) * v27_data);
              v171_acc += ((v175_data[4]) * v28_data);
              v171_acc += ((v175_data[5]) * v29_data);
              v171_acc += ((v175_data[6]) * v30_data);
              v171_acc += ((v175_data[7]) * v31_data);
              v171_acc += ((v175_data[8]) * v32_data);
              v171_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v194_acc{};
              tensorforge::intel_esimd::simd<float, 16> v198_data;
              v198_data.copy_from(s0 + (63_i32));
              v194_acc += ((v198_data[0]) * v24_data);
              v194_acc += ((v198_data[1]) * v25_data);
              v194_acc += ((v198_data[2]) * v26_data);
              v194_acc += ((v198_data[3]) * v27_data);
              v194_acc += ((v198_data[4]) * v28_data);
              v194_acc += ((v198_data[5]) * v29_data);
              v194_acc += ((v198_data[6]) * v30_data);
              v194_acc += ((v198_data[7]) * v31_data);
              v194_acc += ((v198_data[8]) * v32_data);
              v194_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v217_acc{};
              tensorforge::intel_esimd::simd<float, 16> v221_data;
              v221_data.copy_from(s0 + (72_i32));
              v217_acc += ((v221_data[0]) * v24_data);
              v217_acc += ((v221_data[1]) * v25_data);
              v217_acc += ((v221_data[2]) * v26_data);
              v217_acc += ((v221_data[3]) * v27_data);
              v217_acc += ((v221_data[4]) * v28_data);
              v217_acc += ((v221_data[5]) * v29_data);
              v217_acc += ((v221_data[6]) * v30_data);
              v217_acc += ((v221_data[7]) * v31_data);
              v217_acc += ((v221_data[8]) * v32_data);
              v217_acc.copy_to(ir1 + (128));
              #pragma unroll
              for (int32_t v241_n1 = 0; v241_n1 < 9; ++v241_n1) {
                int32_t v242_a = v241_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v244_data;
                v244_data.copy_from(ir1 + (v242_a));
                (v244_data * 13.0f).copy_to(r1 + (v242_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v248_i1 = 0; v248_i1 < 9; ++v248_i1) {
                tensorforge::intel_esimd::simd<float, 9> v251_data;
                v251_data.copy_from(r1 + ((v248_i1 * 16)));
                v251_data.copy_to(glb_m0 + ((v248_i1 * 9)));
              }
            }
          }
        }
      });
    }
  });
}


// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08a27dccde(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08a27dccde(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float r0[144]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 9; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 9> v11_data;
                v11_data.copy_from(glb_m1 + ((v6_i1 * 9)));
                v11_data.copy_to(r0 + ((v6_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v15_ld;
              v15_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v15_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 16> v16_ld;
              v16_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              v16_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              if (item.get_local_id(0) < 1) {
                tensorforge::intel_esimd::simd<float, 16> v17_ld;
                v17_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 80));
                v17_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 80));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[144]{};
              // r1 = +(r0 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir1[144]{};
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v29_acc{};
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(s0 + (0_i32));
              v29_acc += ((v33_data[0]) * v20_data);
              v29_acc += ((v33_data[1]) * v21_data);
              v29_acc += ((v33_data[2]) * v22_data);
              v29_acc += ((v33_data[3]) * v23_data);
              v29_acc += ((v33_data[4]) * v24_data);
              v29_acc += ((v33_data[5]) * v25_data);
              v29_acc += ((v33_data[6]) * v26_data);
              v29_acc += ((v33_data[7]) * v27_data);
              v29_acc += ((v33_data[8]) * v28_data);
              v29_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v52_acc{};
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(s0 + (9_i32));
              v52_acc += ((v56_data[0]) * v20_data);
              v52_acc += ((v56_data[1]) * v21_data);
              v52_acc += ((v56_data[2]) * v22_data);
              v52_acc += ((v56_data[3]) * v23_data);
              v52_acc += ((v56_data[4]) * v24_data);
              v52_acc += ((v56_data[5]) * v25_data);
              v52_acc += ((v56_data[6]) * v26_data);
              v52_acc += ((v56_data[7]) * v27_data);
              v52_acc += ((v56_data[8]) * v28_data);
              v52_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v75_acc{};
              tensorforge::intel_esimd::simd<float, 16> v79_data;
              v79_data.copy_from(s0 + (18_i32));
              v75_acc += ((v79_data[0]) * v20_data);
              v75_acc += ((v79_data[1]) * v21_data);
              v75_acc += ((v79_data[2]) * v22_data);
              v75_acc += ((v79_data[3]) * v23_data);
              v75_acc += ((v79_data[4]) * v24_data);
              v75_acc += ((v79_data[5]) * v25_data);
              v75_acc += ((v79_data[6]) * v26_data);
              v75_acc += ((v79_data[7]) * v27_data);
              v75_acc += ((v79_data[8]) * v28_data);
              v75_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v98_acc{};
              tensorforge::intel_esimd::simd<float, 16> v102_data;
              v102_data.copy_from(s0 + (27_i32));
              v98_acc += ((v102_data[0]) * v20_data);
              v98_acc += ((v102_data[1]) * v21_data);
              v98_acc += ((v102_data[2]) * v22_data);
              v98_acc += ((v102_data[3]) * v23_data);
              v98_acc += ((v102_data[4]) * v24_data);
              v98_acc += ((v102_data[5]) * v25_data);
              v98_acc += ((v102_data[6]) * v26_data);
              v98_acc += ((v102_data[7]) * v27_data);
              v98_acc += ((v102_data[8]) * v28_data);
              v98_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v121_acc{};
              tensorforge::intel_esimd::simd<float, 16> v125_data;
              v125_data.copy_from(s0 + (36_i32));
              v121_acc += ((v125_data[0]) * v20_data);
              v121_acc += ((v125_data[1]) * v21_data);
              v121_acc += ((v125_data[2]) * v22_data);
              v121_acc += ((v125_data[3]) * v23_data);
              v121_acc += ((v125_data[4]) * v24_data);
              v121_acc += ((v125_data[5]) * v25_data);
              v121_acc += ((v125_data[6]) * v26_data);
              v121_acc += ((v125_data[7]) * v27_data);
              v121_acc += ((v125_data[8]) * v28_data);
              v121_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v144_acc{};
              tensorforge::intel_esimd::simd<float, 16> v148_data;
              v148_data.copy_from(s0 + (45_i32));
              v144_acc += ((v148_data[0]) * v20_data);
              v144_acc += ((v148_data[1]) * v21_data);
              v144_acc += ((v148_data[2]) * v22_data);
              v144_acc += ((v148_data[3]) * v23_data);
              v144_acc += ((v148_data[4]) * v24_data);
              v144_acc += ((v148_data[5]) * v25_data);
              v144_acc += ((v148_data[6]) * v26_data);
              v144_acc += ((v148_data[7]) * v27_data);
              v144_acc += ((v148_data[8]) * v28_data);
              v144_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v167_acc{};
              tensorforge::intel_esimd::simd<float, 16> v171_data;
              v171_data.copy_from(s0 + (54_i32));
              v167_acc += ((v171_data[0]) * v20_data);
              v167_acc += ((v171_data[1]) * v21_data);
              v167_acc += ((v171_data[2]) * v22_data);
              v167_acc += ((v171_data[3]) * v23_data);
              v167_acc += ((v171_data[4]) * v24_data);
              v167_acc += ((v171_data[5]) * v25_data);
              v167_acc += ((v171_data[6]) * v26_data);
              v167_acc += ((v171_data[7]) * v27_data);
              v167_acc += ((v171_data[8]) * v28_data);
              v167_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v190_acc{};
              tensorforge::intel_esimd::simd<float, 16> v194_data;
              v194_data.copy_from(s0 + (63_i32));
              v190_acc += ((v194_data[0]) * v20_data);
              v190_acc += ((v194_data[1]) * v21_data);
              v190_acc += ((v194_data[2]) * v22_data);
              v190_acc += ((v194_data[3]) * v23_data);
              v190_acc += ((v194_data[4]) * v24_data);
              v190_acc += ((v194_data[5]) * v25_data);
              v190_acc += ((v194_data[6]) * v26_data);
              v190_acc += ((v194_data[7]) * v27_data);
              v190_acc += ((v194_data[8]) * v28_data);
              v190_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v213_acc{};
              tensorforge::intel_esimd::simd<float, 16> v217_data;
              v217_data.copy_from(s0 + (72_i32));
              v213_acc += ((v217_data[0]) * v20_data);
              v213_acc += ((v217_data[1]) * v21_data);
              v213_acc += ((v217_data[2]) * v22_data);
              v213_acc += ((v217_data[3]) * v23_data);
              v213_acc += ((v217_data[4]) * v24_data);
              v213_acc += ((v217_data[5]) * v25_data);
              v213_acc += ((v217_data[6]) * v26_data);
              v213_acc += ((v217_data[7]) * v27_data);
              v213_acc += ((v217_data[8]) * v28_data);
              v213_acc.copy_to(ir1 + (128));
              #pragma unroll
              for (int32_t v237_n1 = 0; v237_n1 < 9; ++v237_n1) {
                int32_t v238_a = v237_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v240_data;
                v240_data.copy_from(ir1 + (v238_a));
                (v240_data * 13.0f).copy_to(r1 + (v238_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v244_i1 = 0; v244_i1 < 9; ++v244_i1) {
                tensorforge::intel_esimd::simd<float, 9> v247_data;
                v247_data.copy_from(r1 + ((v244_i1 * 16)));
                v247_data.copy_to(glb_m0 + ((v244_i1 * 9)));
              }
            }
          }
        }
      });
    }
  });
}


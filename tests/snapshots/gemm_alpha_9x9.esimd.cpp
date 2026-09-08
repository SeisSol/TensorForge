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
              tensorforge::intel_esimd::simd<float, 16> v30_lin;
              v30_lin.copy_from(s0 + (0 + item.get_local_id(0) * 1));
              float v31_bc = v30_lin[0];
              v29_acc += (v31_bc * v20_data);
              float v33_bc = v30_lin[1];
              v29_acc += (v33_bc * v21_data);
              float v35_bc = v30_lin[2];
              v29_acc += (v35_bc * v22_data);
              float v37_bc = v30_lin[3];
              v29_acc += (v37_bc * v23_data);
              float v39_bc = v30_lin[4];
              v29_acc += (v39_bc * v24_data);
              float v41_bc = v30_lin[5];
              v29_acc += (v41_bc * v25_data);
              float v43_bc = v30_lin[6];
              v29_acc += (v43_bc * v26_data);
              float v45_bc = v30_lin[7];
              v29_acc += (v45_bc * v27_data);
              float v47_bc = v30_lin[8];
              v29_acc += (v47_bc * v28_data);
              v29_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v49_acc{};
              v49_acc += (v31_bc * v20_data);
              v49_acc += (v33_bc * v21_data);
              v49_acc += (v35_bc * v22_data);
              v49_acc += (v37_bc * v23_data);
              v49_acc += (v39_bc * v24_data);
              v49_acc += (v41_bc * v25_data);
              v49_acc += (v43_bc * v26_data);
              v49_acc += (v45_bc * v27_data);
              v49_acc += (v47_bc * v28_data);
              v49_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v69_acc{};
              v69_acc += (v31_bc * v20_data);
              v69_acc += (v33_bc * v21_data);
              v69_acc += (v35_bc * v22_data);
              v69_acc += (v37_bc * v23_data);
              v69_acc += (v39_bc * v24_data);
              v69_acc += (v41_bc * v25_data);
              v69_acc += (v43_bc * v26_data);
              v69_acc += (v45_bc * v27_data);
              v69_acc += (v47_bc * v28_data);
              v69_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v89_acc{};
              v89_acc += (v31_bc * v20_data);
              v89_acc += (v33_bc * v21_data);
              v89_acc += (v35_bc * v22_data);
              v89_acc += (v37_bc * v23_data);
              v89_acc += (v39_bc * v24_data);
              v89_acc += (v41_bc * v25_data);
              v89_acc += (v43_bc * v26_data);
              v89_acc += (v45_bc * v27_data);
              v89_acc += (v47_bc * v28_data);
              v89_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v109_acc{};
              v109_acc += (v31_bc * v20_data);
              v109_acc += (v33_bc * v21_data);
              v109_acc += (v35_bc * v22_data);
              v109_acc += (v37_bc * v23_data);
              v109_acc += (v39_bc * v24_data);
              v109_acc += (v41_bc * v25_data);
              v109_acc += (v43_bc * v26_data);
              v109_acc += (v45_bc * v27_data);
              v109_acc += (v47_bc * v28_data);
              v109_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v129_acc{};
              v129_acc += (v31_bc * v20_data);
              v129_acc += (v33_bc * v21_data);
              v129_acc += (v35_bc * v22_data);
              v129_acc += (v37_bc * v23_data);
              v129_acc += (v39_bc * v24_data);
              v129_acc += (v41_bc * v25_data);
              v129_acc += (v43_bc * v26_data);
              v129_acc += (v45_bc * v27_data);
              v129_acc += (v47_bc * v28_data);
              v129_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v149_acc{};
              v149_acc += (v31_bc * v20_data);
              v149_acc += (v33_bc * v21_data);
              v149_acc += (v35_bc * v22_data);
              v149_acc += (v37_bc * v23_data);
              v149_acc += (v39_bc * v24_data);
              v149_acc += (v41_bc * v25_data);
              v149_acc += (v43_bc * v26_data);
              v149_acc += (v45_bc * v27_data);
              v149_acc += (v47_bc * v28_data);
              v149_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v169_acc{};
              v169_acc += (v31_bc * v20_data);
              v169_acc += (v33_bc * v21_data);
              v169_acc += (v35_bc * v22_data);
              v169_acc += (v37_bc * v23_data);
              v169_acc += (v39_bc * v24_data);
              v169_acc += (v41_bc * v25_data);
              v169_acc += (v43_bc * v26_data);
              v169_acc += (v45_bc * v27_data);
              v169_acc += (v47_bc * v28_data);
              v169_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v189_acc{};
              v189_acc += (v31_bc * v20_data);
              v189_acc += (v33_bc * v21_data);
              v189_acc += (v35_bc * v22_data);
              v189_acc += (v37_bc * v23_data);
              v189_acc += (v39_bc * v24_data);
              v189_acc += (v41_bc * v25_data);
              v189_acc += (v43_bc * v26_data);
              v189_acc += (v45_bc * v27_data);
              v189_acc += (v47_bc * v28_data);
              v189_acc.copy_to(ir1 + (128));
              #pragma unroll
              for (int32_t v210_n1 = 0; v210_n1 < 9; ++v210_n1) {
                int32_t v211_a = v210_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v213_data;
                v213_data.copy_from(ir1 + (v211_a));
                (v213_data * 13.0f).copy_to(r1 + (v211_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v217_i1 = 0; v217_i1 < 9; ++v217_i1) {
                tensorforge::intel_esimd::simd<float, 9> v220_data;
                v220_data.copy_from(r1 + ((v217_i1 * 16)));
                v220_data.copy_to(glb_m0 + ((v217_i1 * 9)));
              }
            }
          }
        }
      });
    }
  });
}


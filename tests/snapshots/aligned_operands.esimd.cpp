// === base name ===
kernel_21138a3fa2

// === header ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_21138a3fa2(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_21138a3fa2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
                int32_t v8_lead = v6_i0 * 16;
                #pragma unroll
                for (int32_t v7_i1 = 0; v7_i1 < 16; ++v7_i1) {
                  int32_t v11_a = v8_lead + (v7_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v12_data;
                  v12_data.copy_from(glb_m1 + (v11_a));
                  v12_data.copy_to(r0 + (v11_a));
                }
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v17_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v18_ld;
              v18_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v18_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v37_acc{};
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(s0 + (0_i32));
              v37_acc += ((v41_data[0]) * v21_data);
              v37_acc += ((v41_data[1]) * v22_data);
              v37_acc += ((v41_data[2]) * v23_data);
              v37_acc += ((v41_data[3]) * v24_data);
              v37_acc += ((v41_data[4]) * v25_data);
              v37_acc += ((v41_data[5]) * v26_data);
              v37_acc += ((v41_data[6]) * v27_data);
              v37_acc += ((v41_data[7]) * v28_data);
              v37_acc += ((v41_data[8]) * v29_data);
              v37_acc += ((v41_data[9]) * v30_data);
              v37_acc += ((v41_data[10]) * v31_data);
              v37_acc += ((v41_data[11]) * v32_data);
              v37_acc += ((v41_data[12]) * v33_data);
              v37_acc += ((v41_data[13]) * v34_data);
              v37_acc += ((v41_data[14]) * v35_data);
              v37_acc += ((v41_data[15]) * v36_data);
              v37_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v74_acc{};
              tensorforge::intel_esimd::simd<float, 16> v78_data;
              v78_data.copy_from(s0 + (16_i32));
              v74_acc += ((v78_data[0]) * v21_data);
              v74_acc += ((v78_data[1]) * v22_data);
              v74_acc += ((v78_data[2]) * v23_data);
              v74_acc += ((v78_data[3]) * v24_data);
              v74_acc += ((v78_data[4]) * v25_data);
              v74_acc += ((v78_data[5]) * v26_data);
              v74_acc += ((v78_data[6]) * v27_data);
              v74_acc += ((v78_data[7]) * v28_data);
              v74_acc += ((v78_data[8]) * v29_data);
              v74_acc += ((v78_data[9]) * v30_data);
              v74_acc += ((v78_data[10]) * v31_data);
              v74_acc += ((v78_data[11]) * v32_data);
              v74_acc += ((v78_data[12]) * v33_data);
              v74_acc += ((v78_data[13]) * v34_data);
              v74_acc += ((v78_data[14]) * v35_data);
              v74_acc += ((v78_data[15]) * v36_data);
              v74_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v111_acc{};
              tensorforge::intel_esimd::simd<float, 16> v115_data;
              v115_data.copy_from(s0 + (32_i32));
              v111_acc += ((v115_data[0]) * v21_data);
              v111_acc += ((v115_data[1]) * v22_data);
              v111_acc += ((v115_data[2]) * v23_data);
              v111_acc += ((v115_data[3]) * v24_data);
              v111_acc += ((v115_data[4]) * v25_data);
              v111_acc += ((v115_data[5]) * v26_data);
              v111_acc += ((v115_data[6]) * v27_data);
              v111_acc += ((v115_data[7]) * v28_data);
              v111_acc += ((v115_data[8]) * v29_data);
              v111_acc += ((v115_data[9]) * v30_data);
              v111_acc += ((v115_data[10]) * v31_data);
              v111_acc += ((v115_data[11]) * v32_data);
              v111_acc += ((v115_data[12]) * v33_data);
              v111_acc += ((v115_data[13]) * v34_data);
              v111_acc += ((v115_data[14]) * v35_data);
              v111_acc += ((v115_data[15]) * v36_data);
              v111_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v148_acc{};
              tensorforge::intel_esimd::simd<float, 16> v152_data;
              v152_data.copy_from(s0 + (48_i32));
              v148_acc += ((v152_data[0]) * v21_data);
              v148_acc += ((v152_data[1]) * v22_data);
              v148_acc += ((v152_data[2]) * v23_data);
              v148_acc += ((v152_data[3]) * v24_data);
              v148_acc += ((v152_data[4]) * v25_data);
              v148_acc += ((v152_data[5]) * v26_data);
              v148_acc += ((v152_data[6]) * v27_data);
              v148_acc += ((v152_data[7]) * v28_data);
              v148_acc += ((v152_data[8]) * v29_data);
              v148_acc += ((v152_data[9]) * v30_data);
              v148_acc += ((v152_data[10]) * v31_data);
              v148_acc += ((v152_data[11]) * v32_data);
              v148_acc += ((v152_data[12]) * v33_data);
              v148_acc += ((v152_data[13]) * v34_data);
              v148_acc += ((v152_data[14]) * v35_data);
              v148_acc += ((v152_data[15]) * v36_data);
              v148_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v185_acc{};
              tensorforge::intel_esimd::simd<float, 16> v189_data;
              v189_data.copy_from(s0 + (64_i32));
              v185_acc += ((v189_data[0]) * v21_data);
              v185_acc += ((v189_data[1]) * v22_data);
              v185_acc += ((v189_data[2]) * v23_data);
              v185_acc += ((v189_data[3]) * v24_data);
              v185_acc += ((v189_data[4]) * v25_data);
              v185_acc += ((v189_data[5]) * v26_data);
              v185_acc += ((v189_data[6]) * v27_data);
              v185_acc += ((v189_data[7]) * v28_data);
              v185_acc += ((v189_data[8]) * v29_data);
              v185_acc += ((v189_data[9]) * v30_data);
              v185_acc += ((v189_data[10]) * v31_data);
              v185_acc += ((v189_data[11]) * v32_data);
              v185_acc += ((v189_data[12]) * v33_data);
              v185_acc += ((v189_data[13]) * v34_data);
              v185_acc += ((v189_data[14]) * v35_data);
              v185_acc += ((v189_data[15]) * v36_data);
              v185_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v222_acc{};
              tensorforge::intel_esimd::simd<float, 16> v226_data;
              v226_data.copy_from(s0 + (80_i32));
              v222_acc += ((v226_data[0]) * v21_data);
              v222_acc += ((v226_data[1]) * v22_data);
              v222_acc += ((v226_data[2]) * v23_data);
              v222_acc += ((v226_data[3]) * v24_data);
              v222_acc += ((v226_data[4]) * v25_data);
              v222_acc += ((v226_data[5]) * v26_data);
              v222_acc += ((v226_data[6]) * v27_data);
              v222_acc += ((v226_data[7]) * v28_data);
              v222_acc += ((v226_data[8]) * v29_data);
              v222_acc += ((v226_data[9]) * v30_data);
              v222_acc += ((v226_data[10]) * v31_data);
              v222_acc += ((v226_data[11]) * v32_data);
              v222_acc += ((v226_data[12]) * v33_data);
              v222_acc += ((v226_data[13]) * v34_data);
              v222_acc += ((v226_data[14]) * v35_data);
              v222_acc += ((v226_data[15]) * v36_data);
              v222_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v259_acc{};
              tensorforge::intel_esimd::simd<float, 16> v263_data;
              v263_data.copy_from(s0 + (96_i32));
              v259_acc += ((v263_data[0]) * v21_data);
              v259_acc += ((v263_data[1]) * v22_data);
              v259_acc += ((v263_data[2]) * v23_data);
              v259_acc += ((v263_data[3]) * v24_data);
              v259_acc += ((v263_data[4]) * v25_data);
              v259_acc += ((v263_data[5]) * v26_data);
              v259_acc += ((v263_data[6]) * v27_data);
              v259_acc += ((v263_data[7]) * v28_data);
              v259_acc += ((v263_data[8]) * v29_data);
              v259_acc += ((v263_data[9]) * v30_data);
              v259_acc += ((v263_data[10]) * v31_data);
              v259_acc += ((v263_data[11]) * v32_data);
              v259_acc += ((v263_data[12]) * v33_data);
              v259_acc += ((v263_data[13]) * v34_data);
              v259_acc += ((v263_data[14]) * v35_data);
              v259_acc += ((v263_data[15]) * v36_data);
              v259_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v296_acc{};
              tensorforge::intel_esimd::simd<float, 16> v300_data;
              v300_data.copy_from(s0 + (112_i32));
              v296_acc += ((v300_data[0]) * v21_data);
              v296_acc += ((v300_data[1]) * v22_data);
              v296_acc += ((v300_data[2]) * v23_data);
              v296_acc += ((v300_data[3]) * v24_data);
              v296_acc += ((v300_data[4]) * v25_data);
              v296_acc += ((v300_data[5]) * v26_data);
              v296_acc += ((v300_data[6]) * v27_data);
              v296_acc += ((v300_data[7]) * v28_data);
              v296_acc += ((v300_data[8]) * v29_data);
              v296_acc += ((v300_data[9]) * v30_data);
              v296_acc += ((v300_data[10]) * v31_data);
              v296_acc += ((v300_data[11]) * v32_data);
              v296_acc += ((v300_data[12]) * v33_data);
              v296_acc += ((v300_data[13]) * v34_data);
              v296_acc += ((v300_data[14]) * v35_data);
              v296_acc += ((v300_data[15]) * v36_data);
              v296_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v333_n0 = 0; v333_n0 < 1; ++v333_n0) {
                int32_t v335_a = v333_n0 * 16;
                #pragma unroll
                for (int32_t v334_n1 = 0; v334_n1 < 8; ++v334_n1) {
                  int32_t v337_a = v335_a + (v334_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v338_data;
                  v338_data.copy_from(ir1 + (v337_a));
                  v338_data.copy_to(r1 + (v337_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v342_i0 = 0; v342_i0 < 1; ++v342_i0) {
                int32_t v344_a = v342_i0 * 16;
                #pragma unroll
                for (int32_t v343_i1 = 0; v343_i1 < 8; ++v343_i1) {
                  int32_t v346_a = v344_a + (v343_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v347_data;
                  v347_data.copy_from(r1 + (v346_a));
                  v347_data.copy_to(glb_m0 + (v346_a));
                }
              }
            }
          }
        }
      });
    }
  });
}


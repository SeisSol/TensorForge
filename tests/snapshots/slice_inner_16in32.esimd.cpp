// === base name ===
kernel_87f2838a59

// === header ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_87f2838a59(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_87f2838a59(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 32×32(32×32) {0..32}×{0..32} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
                int32_t v8_lead = v6_i0 * 16;
                int32_t v10_off = v8_lead + 8;
                #pragma unroll
                for (int32_t v7_i1 = 8; v7_i1 < 24; ++v7_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v13_data;
                  v13_data.copy_from(glb_m1 + ((v10_off + (v7_i1 * 32))));
                  v13_data.copy_to(r0 + ((v8_lead + ((v7_i1 - 8) * 16))));
                }
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
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
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 5) & 31))));
              v37_acc += ((v44_data[0]) * v21_data);
              v37_acc += ((v44_data[1]) * v22_data);
              v37_acc += ((v44_data[2]) * v23_data);
              v37_acc += ((v44_data[3]) * v24_data);
              v37_acc += ((v44_data[4]) * v25_data);
              v37_acc += ((v44_data[5]) * v26_data);
              v37_acc += ((v44_data[6]) * v27_data);
              v37_acc += ((v44_data[7]) * v28_data);
              v37_acc += ((v44_data[8]) * v29_data);
              v37_acc += ((v44_data[9]) * v30_data);
              v37_acc += ((v44_data[10]) * v31_data);
              v37_acc += ((v44_data[11]) * v32_data);
              v37_acc += ((v44_data[12]) * v33_data);
              v37_acc += ((v44_data[13]) * v34_data);
              v37_acc += ((v44_data[14]) * v35_data);
              v37_acc += ((v44_data[15]) * v36_data);
              v37_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v77_acc{};
              tensorforge::intel_esimd::simd<float, 16> v84_data;
              v84_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 5) & 31))));
              v77_acc += ((v84_data[0]) * v21_data);
              v77_acc += ((v84_data[1]) * v22_data);
              v77_acc += ((v84_data[2]) * v23_data);
              v77_acc += ((v84_data[3]) * v24_data);
              v77_acc += ((v84_data[4]) * v25_data);
              v77_acc += ((v84_data[5]) * v26_data);
              v77_acc += ((v84_data[6]) * v27_data);
              v77_acc += ((v84_data[7]) * v28_data);
              v77_acc += ((v84_data[8]) * v29_data);
              v77_acc += ((v84_data[9]) * v30_data);
              v77_acc += ((v84_data[10]) * v31_data);
              v77_acc += ((v84_data[11]) * v32_data);
              v77_acc += ((v84_data[12]) * v33_data);
              v77_acc += ((v84_data[13]) * v34_data);
              v77_acc += ((v84_data[14]) * v35_data);
              v77_acc += ((v84_data[15]) * v36_data);
              v77_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v117_acc{};
              tensorforge::intel_esimd::simd<float, 16> v124_data;
              v124_data.copy_from(s0 + ((32_i32 ^ ((32_i32 >> 5) & 31))));
              v117_acc += ((v124_data[0]) * v21_data);
              v117_acc += ((v124_data[1]) * v22_data);
              v117_acc += ((v124_data[2]) * v23_data);
              v117_acc += ((v124_data[3]) * v24_data);
              v117_acc += ((v124_data[4]) * v25_data);
              v117_acc += ((v124_data[5]) * v26_data);
              v117_acc += ((v124_data[6]) * v27_data);
              v117_acc += ((v124_data[7]) * v28_data);
              v117_acc += ((v124_data[8]) * v29_data);
              v117_acc += ((v124_data[9]) * v30_data);
              v117_acc += ((v124_data[10]) * v31_data);
              v117_acc += ((v124_data[11]) * v32_data);
              v117_acc += ((v124_data[12]) * v33_data);
              v117_acc += ((v124_data[13]) * v34_data);
              v117_acc += ((v124_data[14]) * v35_data);
              v117_acc += ((v124_data[15]) * v36_data);
              v117_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v157_acc{};
              tensorforge::intel_esimd::simd<float, 16> v164_data;
              v164_data.copy_from(s0 + ((48_i32 ^ ((48_i32 >> 5) & 31))));
              v157_acc += ((v164_data[0]) * v21_data);
              v157_acc += ((v164_data[1]) * v22_data);
              v157_acc += ((v164_data[2]) * v23_data);
              v157_acc += ((v164_data[3]) * v24_data);
              v157_acc += ((v164_data[4]) * v25_data);
              v157_acc += ((v164_data[5]) * v26_data);
              v157_acc += ((v164_data[6]) * v27_data);
              v157_acc += ((v164_data[7]) * v28_data);
              v157_acc += ((v164_data[8]) * v29_data);
              v157_acc += ((v164_data[9]) * v30_data);
              v157_acc += ((v164_data[10]) * v31_data);
              v157_acc += ((v164_data[11]) * v32_data);
              v157_acc += ((v164_data[12]) * v33_data);
              v157_acc += ((v164_data[13]) * v34_data);
              v157_acc += ((v164_data[14]) * v35_data);
              v157_acc += ((v164_data[15]) * v36_data);
              v157_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v197_acc{};
              tensorforge::intel_esimd::simd<float, 16> v204_data;
              v204_data.copy_from(s0 + ((64_i32 ^ ((64_i32 >> 5) & 31))));
              v197_acc += ((v204_data[0]) * v21_data);
              v197_acc += ((v204_data[1]) * v22_data);
              v197_acc += ((v204_data[2]) * v23_data);
              v197_acc += ((v204_data[3]) * v24_data);
              v197_acc += ((v204_data[4]) * v25_data);
              v197_acc += ((v204_data[5]) * v26_data);
              v197_acc += ((v204_data[6]) * v27_data);
              v197_acc += ((v204_data[7]) * v28_data);
              v197_acc += ((v204_data[8]) * v29_data);
              v197_acc += ((v204_data[9]) * v30_data);
              v197_acc += ((v204_data[10]) * v31_data);
              v197_acc += ((v204_data[11]) * v32_data);
              v197_acc += ((v204_data[12]) * v33_data);
              v197_acc += ((v204_data[13]) * v34_data);
              v197_acc += ((v204_data[14]) * v35_data);
              v197_acc += ((v204_data[15]) * v36_data);
              v197_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v237_acc{};
              tensorforge::intel_esimd::simd<float, 16> v244_data;
              v244_data.copy_from(s0 + ((80_i32 ^ ((80_i32 >> 5) & 31))));
              v237_acc += ((v244_data[0]) * v21_data);
              v237_acc += ((v244_data[1]) * v22_data);
              v237_acc += ((v244_data[2]) * v23_data);
              v237_acc += ((v244_data[3]) * v24_data);
              v237_acc += ((v244_data[4]) * v25_data);
              v237_acc += ((v244_data[5]) * v26_data);
              v237_acc += ((v244_data[6]) * v27_data);
              v237_acc += ((v244_data[7]) * v28_data);
              v237_acc += ((v244_data[8]) * v29_data);
              v237_acc += ((v244_data[9]) * v30_data);
              v237_acc += ((v244_data[10]) * v31_data);
              v237_acc += ((v244_data[11]) * v32_data);
              v237_acc += ((v244_data[12]) * v33_data);
              v237_acc += ((v244_data[13]) * v34_data);
              v237_acc += ((v244_data[14]) * v35_data);
              v237_acc += ((v244_data[15]) * v36_data);
              v237_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v277_acc{};
              tensorforge::intel_esimd::simd<float, 16> v284_data;
              v284_data.copy_from(s0 + ((96_i32 ^ ((96_i32 >> 5) & 31))));
              v277_acc += ((v284_data[0]) * v21_data);
              v277_acc += ((v284_data[1]) * v22_data);
              v277_acc += ((v284_data[2]) * v23_data);
              v277_acc += ((v284_data[3]) * v24_data);
              v277_acc += ((v284_data[4]) * v25_data);
              v277_acc += ((v284_data[5]) * v26_data);
              v277_acc += ((v284_data[6]) * v27_data);
              v277_acc += ((v284_data[7]) * v28_data);
              v277_acc += ((v284_data[8]) * v29_data);
              v277_acc += ((v284_data[9]) * v30_data);
              v277_acc += ((v284_data[10]) * v31_data);
              v277_acc += ((v284_data[11]) * v32_data);
              v277_acc += ((v284_data[12]) * v33_data);
              v277_acc += ((v284_data[13]) * v34_data);
              v277_acc += ((v284_data[14]) * v35_data);
              v277_acc += ((v284_data[15]) * v36_data);
              v277_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v317_acc{};
              tensorforge::intel_esimd::simd<float, 16> v324_data;
              v324_data.copy_from(s0 + ((112_i32 ^ ((112_i32 >> 5) & 31))));
              v317_acc += ((v324_data[0]) * v21_data);
              v317_acc += ((v324_data[1]) * v22_data);
              v317_acc += ((v324_data[2]) * v23_data);
              v317_acc += ((v324_data[3]) * v24_data);
              v317_acc += ((v324_data[4]) * v25_data);
              v317_acc += ((v324_data[5]) * v26_data);
              v317_acc += ((v324_data[6]) * v27_data);
              v317_acc += ((v324_data[7]) * v28_data);
              v317_acc += ((v324_data[8]) * v29_data);
              v317_acc += ((v324_data[9]) * v30_data);
              v317_acc += ((v324_data[10]) * v31_data);
              v317_acc += ((v324_data[11]) * v32_data);
              v317_acc += ((v324_data[12]) * v33_data);
              v317_acc += ((v324_data[13]) * v34_data);
              v317_acc += ((v324_data[14]) * v35_data);
              v317_acc += ((v324_data[15]) * v36_data);
              v317_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v357_n0 = 0; v357_n0 < 1; ++v357_n0) {
                int32_t v359_a = v357_n0 * 16;
                #pragma unroll
                for (int32_t v358_n1 = 0; v358_n1 < 8; ++v358_n1) {
                  int32_t v361_a = v359_a + (v358_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v362_data;
                  v362_data.copy_from(ir1 + (v361_a));
                  v362_data.copy_to(r1 + (v361_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v366_i0 = 0; v366_i0 < 1; ++v366_i0) {
                int32_t v368_a = v366_i0 * 16;
                #pragma unroll
                for (int32_t v367_i1 = 0; v367_i1 < 8; ++v367_i1) {
                  int32_t v370_a = v368_a + (v367_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v371_data;
                  v371_data.copy_from(r1 + (v370_a));
                  v371_data.copy_to(glb_m0 + (v370_a));
                }
              }
            }
          }
        }
      });
    }
  });
}


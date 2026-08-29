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
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[128]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir0[128]{};
              tensorforge::intel_esimd::simd<float, 16> v9_data;
              v9_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v13_data;
              v13_data.copy_from(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v17_data;
              v17_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v53_data;
              v53_data.copy_from(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<float, 16> v57_data;
              v57_data.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data;
              v65_data.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<float, 16> v69_data;
              v69_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_acc{};
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 4) & 15))));
              v70_acc += ((v77_data[0]) * v9_data);
              v70_acc += ((v77_data[1]) * v13_data);
              v70_acc += ((v77_data[2]) * v17_data);
              v70_acc += ((v77_data[3]) * v21_data);
              v70_acc += ((v77_data[4]) * v25_data);
              v70_acc += ((v77_data[5]) * v29_data);
              v70_acc += ((v77_data[6]) * v33_data);
              v70_acc += ((v77_data[7]) * v37_data);
              v70_acc += ((v77_data[8]) * v41_data);
              v70_acc += ((v77_data[9]) * v45_data);
              v70_acc += ((v77_data[10]) * v49_data);
              v70_acc += ((v77_data[11]) * v53_data);
              v70_acc += ((v77_data[12]) * v57_data);
              v70_acc += ((v77_data[13]) * v61_data);
              v70_acc += ((v77_data[14]) * v65_data);
              v70_acc += ((v77_data[15]) * v69_data);
              v70_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v110_acc{};
              tensorforge::intel_esimd::simd<float, 16> v117_data;
              v117_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 4) & 15))));
              v110_acc += ((v117_data[0]) * v9_data);
              v110_acc += ((v117_data[1]) * v13_data);
              v110_acc += ((v117_data[2]) * v17_data);
              v110_acc += ((v117_data[3]) * v21_data);
              v110_acc += ((v117_data[4]) * v25_data);
              v110_acc += ((v117_data[5]) * v29_data);
              v110_acc += ((v117_data[6]) * v33_data);
              v110_acc += ((v117_data[7]) * v37_data);
              v110_acc += ((v117_data[8]) * v41_data);
              v110_acc += ((v117_data[9]) * v45_data);
              v110_acc += ((v117_data[10]) * v49_data);
              v110_acc += ((v117_data[11]) * v53_data);
              v110_acc += ((v117_data[12]) * v57_data);
              v110_acc += ((v117_data[13]) * v61_data);
              v110_acc += ((v117_data[14]) * v65_data);
              v110_acc += ((v117_data[15]) * v69_data);
              v110_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v150_acc{};
              tensorforge::intel_esimd::simd<float, 16> v157_data;
              v157_data.copy_from(s0 + ((32_i32 ^ ((32_i32 >> 4) & 15))));
              v150_acc += ((v157_data[0]) * v9_data);
              v150_acc += ((v157_data[1]) * v13_data);
              v150_acc += ((v157_data[2]) * v17_data);
              v150_acc += ((v157_data[3]) * v21_data);
              v150_acc += ((v157_data[4]) * v25_data);
              v150_acc += ((v157_data[5]) * v29_data);
              v150_acc += ((v157_data[6]) * v33_data);
              v150_acc += ((v157_data[7]) * v37_data);
              v150_acc += ((v157_data[8]) * v41_data);
              v150_acc += ((v157_data[9]) * v45_data);
              v150_acc += ((v157_data[10]) * v49_data);
              v150_acc += ((v157_data[11]) * v53_data);
              v150_acc += ((v157_data[12]) * v57_data);
              v150_acc += ((v157_data[13]) * v61_data);
              v150_acc += ((v157_data[14]) * v65_data);
              v150_acc += ((v157_data[15]) * v69_data);
              v150_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v190_acc{};
              tensorforge::intel_esimd::simd<float, 16> v197_data;
              v197_data.copy_from(s0 + ((48_i32 ^ ((48_i32 >> 4) & 15))));
              v190_acc += ((v197_data[0]) * v9_data);
              v190_acc += ((v197_data[1]) * v13_data);
              v190_acc += ((v197_data[2]) * v17_data);
              v190_acc += ((v197_data[3]) * v21_data);
              v190_acc += ((v197_data[4]) * v25_data);
              v190_acc += ((v197_data[5]) * v29_data);
              v190_acc += ((v197_data[6]) * v33_data);
              v190_acc += ((v197_data[7]) * v37_data);
              v190_acc += ((v197_data[8]) * v41_data);
              v190_acc += ((v197_data[9]) * v45_data);
              v190_acc += ((v197_data[10]) * v49_data);
              v190_acc += ((v197_data[11]) * v53_data);
              v190_acc += ((v197_data[12]) * v57_data);
              v190_acc += ((v197_data[13]) * v61_data);
              v190_acc += ((v197_data[14]) * v65_data);
              v190_acc += ((v197_data[15]) * v69_data);
              v190_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v230_acc{};
              tensorforge::intel_esimd::simd<float, 16> v237_data;
              v237_data.copy_from(s0 + ((64_i32 ^ ((64_i32 >> 4) & 15))));
              v230_acc += ((v237_data[0]) * v9_data);
              v230_acc += ((v237_data[1]) * v13_data);
              v230_acc += ((v237_data[2]) * v17_data);
              v230_acc += ((v237_data[3]) * v21_data);
              v230_acc += ((v237_data[4]) * v25_data);
              v230_acc += ((v237_data[5]) * v29_data);
              v230_acc += ((v237_data[6]) * v33_data);
              v230_acc += ((v237_data[7]) * v37_data);
              v230_acc += ((v237_data[8]) * v41_data);
              v230_acc += ((v237_data[9]) * v45_data);
              v230_acc += ((v237_data[10]) * v49_data);
              v230_acc += ((v237_data[11]) * v53_data);
              v230_acc += ((v237_data[12]) * v57_data);
              v230_acc += ((v237_data[13]) * v61_data);
              v230_acc += ((v237_data[14]) * v65_data);
              v230_acc += ((v237_data[15]) * v69_data);
              v230_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v270_acc{};
              tensorforge::intel_esimd::simd<float, 16> v277_data;
              v277_data.copy_from(s0 + ((80_i32 ^ ((80_i32 >> 4) & 15))));
              v270_acc += ((v277_data[0]) * v9_data);
              v270_acc += ((v277_data[1]) * v13_data);
              v270_acc += ((v277_data[2]) * v17_data);
              v270_acc += ((v277_data[3]) * v21_data);
              v270_acc += ((v277_data[4]) * v25_data);
              v270_acc += ((v277_data[5]) * v29_data);
              v270_acc += ((v277_data[6]) * v33_data);
              v270_acc += ((v277_data[7]) * v37_data);
              v270_acc += ((v277_data[8]) * v41_data);
              v270_acc += ((v277_data[9]) * v45_data);
              v270_acc += ((v277_data[10]) * v49_data);
              v270_acc += ((v277_data[11]) * v53_data);
              v270_acc += ((v277_data[12]) * v57_data);
              v270_acc += ((v277_data[13]) * v61_data);
              v270_acc += ((v277_data[14]) * v65_data);
              v270_acc += ((v277_data[15]) * v69_data);
              v270_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v310_acc{};
              tensorforge::intel_esimd::simd<float, 16> v317_data;
              v317_data.copy_from(s0 + ((96_i32 ^ ((96_i32 >> 4) & 15))));
              v310_acc += ((v317_data[0]) * v9_data);
              v310_acc += ((v317_data[1]) * v13_data);
              v310_acc += ((v317_data[2]) * v17_data);
              v310_acc += ((v317_data[3]) * v21_data);
              v310_acc += ((v317_data[4]) * v25_data);
              v310_acc += ((v317_data[5]) * v29_data);
              v310_acc += ((v317_data[6]) * v33_data);
              v310_acc += ((v317_data[7]) * v37_data);
              v310_acc += ((v317_data[8]) * v41_data);
              v310_acc += ((v317_data[9]) * v45_data);
              v310_acc += ((v317_data[10]) * v49_data);
              v310_acc += ((v317_data[11]) * v53_data);
              v310_acc += ((v317_data[12]) * v57_data);
              v310_acc += ((v317_data[13]) * v61_data);
              v310_acc += ((v317_data[14]) * v65_data);
              v310_acc += ((v317_data[15]) * v69_data);
              v310_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v350_acc{};
              tensorforge::intel_esimd::simd<float, 16> v357_data;
              v357_data.copy_from(s0 + ((112_i32 ^ ((112_i32 >> 4) & 15))));
              v350_acc += ((v357_data[0]) * v9_data);
              v350_acc += ((v357_data[1]) * v13_data);
              v350_acc += ((v357_data[2]) * v17_data);
              v350_acc += ((v357_data[3]) * v21_data);
              v350_acc += ((v357_data[4]) * v25_data);
              v350_acc += ((v357_data[5]) * v29_data);
              v350_acc += ((v357_data[6]) * v33_data);
              v350_acc += ((v357_data[7]) * v37_data);
              v350_acc += ((v357_data[8]) * v41_data);
              v350_acc += ((v357_data[9]) * v45_data);
              v350_acc += ((v357_data[10]) * v49_data);
              v350_acc += ((v357_data[11]) * v53_data);
              v350_acc += ((v357_data[12]) * v57_data);
              v350_acc += ((v357_data[13]) * v61_data);
              v350_acc += ((v357_data[14]) * v65_data);
              v350_acc += ((v357_data[15]) * v69_data);
              v350_acc.copy_to(ir0 + (112));
              #pragma unroll
              for (int32_t v390_n0 = 0; v390_n0 < 1; ++v390_n0) {
                int32_t v392_a = v390_n0 * 16;
                #pragma unroll
                for (int32_t v391_n1 = 0; v391_n1 < 8; ++v391_n1) {
                  int32_t v394_a = v392_a + (v391_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v395_data;
                  v395_data.copy_from(ir0 + (v394_a));
                  v395_data.copy_to(r0 + (v394_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v399_i0 = 0; v399_i0 < 1; ++v399_i0) {
                int32_t v401_a = v399_i0 * 16;
                #pragma unroll
                for (int32_t v400_i1 = 0; v400_i1 < 8; ++v400_i1) {
                  int32_t v403_a = v401_a + (v400_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v404_data;
                  v404_data.copy_from(r0 + (v403_a));
                  v404_data.copy_to(glb_m0 + (v403_a));
                }
              }
            }
          }
        }
      });
    }
  });
}


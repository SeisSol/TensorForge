// === base name ===
kernel_ead773dd51

// === header ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ead773dd51(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ead773dd51(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(32×16) {0..32}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(32×16) {0..32}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 16; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 12> v12_data;
                v12_data.copy_from(glb_m1 + ((4_i32 + (v6_i1 * 32))));
                v12_data.copy_to(r0 + ((v6_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v18_data;
              v18_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v34_acc{};
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 5) & 31))));
              v34_acc += ((v41_data[0]) * v18_data);
              v34_acc += ((v41_data[1]) * v19_data);
              v34_acc += ((v41_data[2]) * v20_data);
              v34_acc += ((v41_data[3]) * v21_data);
              v34_acc += ((v41_data[4]) * v22_data);
              v34_acc += ((v41_data[5]) * v23_data);
              v34_acc += ((v41_data[6]) * v24_data);
              v34_acc += ((v41_data[7]) * v25_data);
              v34_acc += ((v41_data[8]) * v26_data);
              v34_acc += ((v41_data[9]) * v27_data);
              v34_acc += ((v41_data[10]) * v28_data);
              v34_acc += ((v41_data[11]) * v29_data);
              v34_acc += ((v41_data[12]) * v30_data);
              v34_acc += ((v41_data[13]) * v31_data);
              v34_acc += ((v41_data[14]) * v32_data);
              v34_acc += ((v41_data[15]) * v33_data);
              v34_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v74_acc{};
              tensorforge::intel_esimd::simd<float, 16> v81_data;
              v81_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 5) & 31))));
              v74_acc += ((v81_data[0]) * v18_data);
              v74_acc += ((v81_data[1]) * v19_data);
              v74_acc += ((v81_data[2]) * v20_data);
              v74_acc += ((v81_data[3]) * v21_data);
              v74_acc += ((v81_data[4]) * v22_data);
              v74_acc += ((v81_data[5]) * v23_data);
              v74_acc += ((v81_data[6]) * v24_data);
              v74_acc += ((v81_data[7]) * v25_data);
              v74_acc += ((v81_data[8]) * v26_data);
              v74_acc += ((v81_data[9]) * v27_data);
              v74_acc += ((v81_data[10]) * v28_data);
              v74_acc += ((v81_data[11]) * v29_data);
              v74_acc += ((v81_data[12]) * v30_data);
              v74_acc += ((v81_data[13]) * v31_data);
              v74_acc += ((v81_data[14]) * v32_data);
              v74_acc += ((v81_data[15]) * v33_data);
              v74_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v114_acc{};
              tensorforge::intel_esimd::simd<float, 16> v121_data;
              v121_data.copy_from(s0 + ((32_i32 ^ ((32_i32 >> 5) & 31))));
              v114_acc += ((v121_data[0]) * v18_data);
              v114_acc += ((v121_data[1]) * v19_data);
              v114_acc += ((v121_data[2]) * v20_data);
              v114_acc += ((v121_data[3]) * v21_data);
              v114_acc += ((v121_data[4]) * v22_data);
              v114_acc += ((v121_data[5]) * v23_data);
              v114_acc += ((v121_data[6]) * v24_data);
              v114_acc += ((v121_data[7]) * v25_data);
              v114_acc += ((v121_data[8]) * v26_data);
              v114_acc += ((v121_data[9]) * v27_data);
              v114_acc += ((v121_data[10]) * v28_data);
              v114_acc += ((v121_data[11]) * v29_data);
              v114_acc += ((v121_data[12]) * v30_data);
              v114_acc += ((v121_data[13]) * v31_data);
              v114_acc += ((v121_data[14]) * v32_data);
              v114_acc += ((v121_data[15]) * v33_data);
              v114_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v154_acc{};
              tensorforge::intel_esimd::simd<float, 16> v161_data;
              v161_data.copy_from(s0 + ((48_i32 ^ ((48_i32 >> 5) & 31))));
              v154_acc += ((v161_data[0]) * v18_data);
              v154_acc += ((v161_data[1]) * v19_data);
              v154_acc += ((v161_data[2]) * v20_data);
              v154_acc += ((v161_data[3]) * v21_data);
              v154_acc += ((v161_data[4]) * v22_data);
              v154_acc += ((v161_data[5]) * v23_data);
              v154_acc += ((v161_data[6]) * v24_data);
              v154_acc += ((v161_data[7]) * v25_data);
              v154_acc += ((v161_data[8]) * v26_data);
              v154_acc += ((v161_data[9]) * v27_data);
              v154_acc += ((v161_data[10]) * v28_data);
              v154_acc += ((v161_data[11]) * v29_data);
              v154_acc += ((v161_data[12]) * v30_data);
              v154_acc += ((v161_data[13]) * v31_data);
              v154_acc += ((v161_data[14]) * v32_data);
              v154_acc += ((v161_data[15]) * v33_data);
              v154_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v194_acc{};
              tensorforge::intel_esimd::simd<float, 16> v201_data;
              v201_data.copy_from(s0 + ((64_i32 ^ ((64_i32 >> 5) & 31))));
              v194_acc += ((v201_data[0]) * v18_data);
              v194_acc += ((v201_data[1]) * v19_data);
              v194_acc += ((v201_data[2]) * v20_data);
              v194_acc += ((v201_data[3]) * v21_data);
              v194_acc += ((v201_data[4]) * v22_data);
              v194_acc += ((v201_data[5]) * v23_data);
              v194_acc += ((v201_data[6]) * v24_data);
              v194_acc += ((v201_data[7]) * v25_data);
              v194_acc += ((v201_data[8]) * v26_data);
              v194_acc += ((v201_data[9]) * v27_data);
              v194_acc += ((v201_data[10]) * v28_data);
              v194_acc += ((v201_data[11]) * v29_data);
              v194_acc += ((v201_data[12]) * v30_data);
              v194_acc += ((v201_data[13]) * v31_data);
              v194_acc += ((v201_data[14]) * v32_data);
              v194_acc += ((v201_data[15]) * v33_data);
              v194_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v234_acc{};
              tensorforge::intel_esimd::simd<float, 16> v241_data;
              v241_data.copy_from(s0 + ((80_i32 ^ ((80_i32 >> 5) & 31))));
              v234_acc += ((v241_data[0]) * v18_data);
              v234_acc += ((v241_data[1]) * v19_data);
              v234_acc += ((v241_data[2]) * v20_data);
              v234_acc += ((v241_data[3]) * v21_data);
              v234_acc += ((v241_data[4]) * v22_data);
              v234_acc += ((v241_data[5]) * v23_data);
              v234_acc += ((v241_data[6]) * v24_data);
              v234_acc += ((v241_data[7]) * v25_data);
              v234_acc += ((v241_data[8]) * v26_data);
              v234_acc += ((v241_data[9]) * v27_data);
              v234_acc += ((v241_data[10]) * v28_data);
              v234_acc += ((v241_data[11]) * v29_data);
              v234_acc += ((v241_data[12]) * v30_data);
              v234_acc += ((v241_data[13]) * v31_data);
              v234_acc += ((v241_data[14]) * v32_data);
              v234_acc += ((v241_data[15]) * v33_data);
              v234_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v274_acc{};
              tensorforge::intel_esimd::simd<float, 16> v281_data;
              v281_data.copy_from(s0 + ((96_i32 ^ ((96_i32 >> 5) & 31))));
              v274_acc += ((v281_data[0]) * v18_data);
              v274_acc += ((v281_data[1]) * v19_data);
              v274_acc += ((v281_data[2]) * v20_data);
              v274_acc += ((v281_data[3]) * v21_data);
              v274_acc += ((v281_data[4]) * v22_data);
              v274_acc += ((v281_data[5]) * v23_data);
              v274_acc += ((v281_data[6]) * v24_data);
              v274_acc += ((v281_data[7]) * v25_data);
              v274_acc += ((v281_data[8]) * v26_data);
              v274_acc += ((v281_data[9]) * v27_data);
              v274_acc += ((v281_data[10]) * v28_data);
              v274_acc += ((v281_data[11]) * v29_data);
              v274_acc += ((v281_data[12]) * v30_data);
              v274_acc += ((v281_data[13]) * v31_data);
              v274_acc += ((v281_data[14]) * v32_data);
              v274_acc += ((v281_data[15]) * v33_data);
              v274_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v314_acc{};
              tensorforge::intel_esimd::simd<float, 16> v321_data;
              v321_data.copy_from(s0 + ((112_i32 ^ ((112_i32 >> 5) & 31))));
              v314_acc += ((v321_data[0]) * v18_data);
              v314_acc += ((v321_data[1]) * v19_data);
              v314_acc += ((v321_data[2]) * v20_data);
              v314_acc += ((v321_data[3]) * v21_data);
              v314_acc += ((v321_data[4]) * v22_data);
              v314_acc += ((v321_data[5]) * v23_data);
              v314_acc += ((v321_data[6]) * v24_data);
              v314_acc += ((v321_data[7]) * v25_data);
              v314_acc += ((v321_data[8]) * v26_data);
              v314_acc += ((v321_data[9]) * v27_data);
              v314_acc += ((v321_data[10]) * v28_data);
              v314_acc += ((v321_data[11]) * v29_data);
              v314_acc += ((v321_data[12]) * v30_data);
              v314_acc += ((v321_data[13]) * v31_data);
              v314_acc += ((v321_data[14]) * v32_data);
              v314_acc += ((v321_data[15]) * v33_data);
              v314_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v354_n1 = 0; v354_n1 < 8; ++v354_n1) {
                int32_t v355_a = v354_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v357_data;
                v357_data.copy_from(ir1 + (v355_a));
                v357_data.copy_to(r1 + (v355_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v360_i1 = 0; v360_i1 < 8; ++v360_i1) {
                tensorforge::intel_esimd::simd<float, 12> v363_data;
                v363_data.copy_from(r1 + ((v360_i1 * 16)));
                v363_data.copy_to(glb_m0 + ((v360_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}


// === base name ===
kernel_7ea37d09b508bf0c

// === header ===
void launcher_kernel_7ea37d09b508bf0c(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7ea37d09b508bf0c(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7ea37d09b508bf0c(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7ea37d09b508bf0c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v11_i1 = 0; v11_i1 < 16; ++v11_i1) {
                tensorforge::intel_esimd::simd<float, 12> v17_data;
                v17_data.copy_from(glb_m1 + ((4_i32 + (v11_i1 * 32))));
                v17_data.copy_to(r0 + ((v11_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v20_ld;
              v20_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v20_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v21_ld;
              v21_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v21_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir1[128]{};
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
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v40_acc{};
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(s0 + (0_i32));
              v40_acc += ((v44_data[0]) * v24_data);
              v40_acc += ((v44_data[1]) * v25_data);
              v40_acc += ((v44_data[2]) * v26_data);
              v40_acc += ((v44_data[3]) * v27_data);
              v40_acc += ((v44_data[4]) * v28_data);
              v40_acc += ((v44_data[5]) * v29_data);
              v40_acc += ((v44_data[6]) * v30_data);
              v40_acc += ((v44_data[7]) * v31_data);
              v40_acc += ((v44_data[8]) * v32_data);
              v40_acc += ((v44_data[9]) * v33_data);
              v40_acc += ((v44_data[10]) * v34_data);
              v40_acc += ((v44_data[11]) * v35_data);
              v40_acc += ((v44_data[12]) * v36_data);
              v40_acc += ((v44_data[13]) * v37_data);
              v40_acc += ((v44_data[14]) * v38_data);
              v40_acc += ((v44_data[15]) * v39_data);
              v40_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v77_acc{};
              tensorforge::intel_esimd::simd<float, 16> v81_data;
              v81_data.copy_from(s0 + (16_i32));
              v77_acc += ((v81_data[0]) * v24_data);
              v77_acc += ((v81_data[1]) * v25_data);
              v77_acc += ((v81_data[2]) * v26_data);
              v77_acc += ((v81_data[3]) * v27_data);
              v77_acc += ((v81_data[4]) * v28_data);
              v77_acc += ((v81_data[5]) * v29_data);
              v77_acc += ((v81_data[6]) * v30_data);
              v77_acc += ((v81_data[7]) * v31_data);
              v77_acc += ((v81_data[8]) * v32_data);
              v77_acc += ((v81_data[9]) * v33_data);
              v77_acc += ((v81_data[10]) * v34_data);
              v77_acc += ((v81_data[11]) * v35_data);
              v77_acc += ((v81_data[12]) * v36_data);
              v77_acc += ((v81_data[13]) * v37_data);
              v77_acc += ((v81_data[14]) * v38_data);
              v77_acc += ((v81_data[15]) * v39_data);
              v77_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v114_acc{};
              tensorforge::intel_esimd::simd<float, 16> v118_data;
              v118_data.copy_from(s0 + (32_i32));
              v114_acc += ((v118_data[0]) * v24_data);
              v114_acc += ((v118_data[1]) * v25_data);
              v114_acc += ((v118_data[2]) * v26_data);
              v114_acc += ((v118_data[3]) * v27_data);
              v114_acc += ((v118_data[4]) * v28_data);
              v114_acc += ((v118_data[5]) * v29_data);
              v114_acc += ((v118_data[6]) * v30_data);
              v114_acc += ((v118_data[7]) * v31_data);
              v114_acc += ((v118_data[8]) * v32_data);
              v114_acc += ((v118_data[9]) * v33_data);
              v114_acc += ((v118_data[10]) * v34_data);
              v114_acc += ((v118_data[11]) * v35_data);
              v114_acc += ((v118_data[12]) * v36_data);
              v114_acc += ((v118_data[13]) * v37_data);
              v114_acc += ((v118_data[14]) * v38_data);
              v114_acc += ((v118_data[15]) * v39_data);
              v114_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v151_acc{};
              tensorforge::intel_esimd::simd<float, 16> v155_data;
              v155_data.copy_from(s0 + (48_i32));
              v151_acc += ((v155_data[0]) * v24_data);
              v151_acc += ((v155_data[1]) * v25_data);
              v151_acc += ((v155_data[2]) * v26_data);
              v151_acc += ((v155_data[3]) * v27_data);
              v151_acc += ((v155_data[4]) * v28_data);
              v151_acc += ((v155_data[5]) * v29_data);
              v151_acc += ((v155_data[6]) * v30_data);
              v151_acc += ((v155_data[7]) * v31_data);
              v151_acc += ((v155_data[8]) * v32_data);
              v151_acc += ((v155_data[9]) * v33_data);
              v151_acc += ((v155_data[10]) * v34_data);
              v151_acc += ((v155_data[11]) * v35_data);
              v151_acc += ((v155_data[12]) * v36_data);
              v151_acc += ((v155_data[13]) * v37_data);
              v151_acc += ((v155_data[14]) * v38_data);
              v151_acc += ((v155_data[15]) * v39_data);
              v151_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v188_acc{};
              tensorforge::intel_esimd::simd<float, 16> v192_data;
              v192_data.copy_from(s0 + (64_i32));
              v188_acc += ((v192_data[0]) * v24_data);
              v188_acc += ((v192_data[1]) * v25_data);
              v188_acc += ((v192_data[2]) * v26_data);
              v188_acc += ((v192_data[3]) * v27_data);
              v188_acc += ((v192_data[4]) * v28_data);
              v188_acc += ((v192_data[5]) * v29_data);
              v188_acc += ((v192_data[6]) * v30_data);
              v188_acc += ((v192_data[7]) * v31_data);
              v188_acc += ((v192_data[8]) * v32_data);
              v188_acc += ((v192_data[9]) * v33_data);
              v188_acc += ((v192_data[10]) * v34_data);
              v188_acc += ((v192_data[11]) * v35_data);
              v188_acc += ((v192_data[12]) * v36_data);
              v188_acc += ((v192_data[13]) * v37_data);
              v188_acc += ((v192_data[14]) * v38_data);
              v188_acc += ((v192_data[15]) * v39_data);
              v188_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v225_acc{};
              tensorforge::intel_esimd::simd<float, 16> v229_data;
              v229_data.copy_from(s0 + (80_i32));
              v225_acc += ((v229_data[0]) * v24_data);
              v225_acc += ((v229_data[1]) * v25_data);
              v225_acc += ((v229_data[2]) * v26_data);
              v225_acc += ((v229_data[3]) * v27_data);
              v225_acc += ((v229_data[4]) * v28_data);
              v225_acc += ((v229_data[5]) * v29_data);
              v225_acc += ((v229_data[6]) * v30_data);
              v225_acc += ((v229_data[7]) * v31_data);
              v225_acc += ((v229_data[8]) * v32_data);
              v225_acc += ((v229_data[9]) * v33_data);
              v225_acc += ((v229_data[10]) * v34_data);
              v225_acc += ((v229_data[11]) * v35_data);
              v225_acc += ((v229_data[12]) * v36_data);
              v225_acc += ((v229_data[13]) * v37_data);
              v225_acc += ((v229_data[14]) * v38_data);
              v225_acc += ((v229_data[15]) * v39_data);
              v225_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v262_acc{};
              tensorforge::intel_esimd::simd<float, 16> v266_data;
              v266_data.copy_from(s0 + (96_i32));
              v262_acc += ((v266_data[0]) * v24_data);
              v262_acc += ((v266_data[1]) * v25_data);
              v262_acc += ((v266_data[2]) * v26_data);
              v262_acc += ((v266_data[3]) * v27_data);
              v262_acc += ((v266_data[4]) * v28_data);
              v262_acc += ((v266_data[5]) * v29_data);
              v262_acc += ((v266_data[6]) * v30_data);
              v262_acc += ((v266_data[7]) * v31_data);
              v262_acc += ((v266_data[8]) * v32_data);
              v262_acc += ((v266_data[9]) * v33_data);
              v262_acc += ((v266_data[10]) * v34_data);
              v262_acc += ((v266_data[11]) * v35_data);
              v262_acc += ((v266_data[12]) * v36_data);
              v262_acc += ((v266_data[13]) * v37_data);
              v262_acc += ((v266_data[14]) * v38_data);
              v262_acc += ((v266_data[15]) * v39_data);
              v262_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v299_acc{};
              tensorforge::intel_esimd::simd<float, 16> v303_data;
              v303_data.copy_from(s0 + (112_i32));
              v299_acc += ((v303_data[0]) * v24_data);
              v299_acc += ((v303_data[1]) * v25_data);
              v299_acc += ((v303_data[2]) * v26_data);
              v299_acc += ((v303_data[3]) * v27_data);
              v299_acc += ((v303_data[4]) * v28_data);
              v299_acc += ((v303_data[5]) * v29_data);
              v299_acc += ((v303_data[6]) * v30_data);
              v299_acc += ((v303_data[7]) * v31_data);
              v299_acc += ((v303_data[8]) * v32_data);
              v299_acc += ((v303_data[9]) * v33_data);
              v299_acc += ((v303_data[10]) * v34_data);
              v299_acc += ((v303_data[11]) * v35_data);
              v299_acc += ((v303_data[12]) * v36_data);
              v299_acc += ((v303_data[13]) * v37_data);
              v299_acc += ((v303_data[14]) * v38_data);
              v299_acc += ((v303_data[15]) * v39_data);
              v299_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v336_n1 = 0; v336_n1 < 8; ++v336_n1) {
                int32_t v337_a = v336_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v339_data;
                v339_data.copy_from(ir1 + (v337_a));
                v339_data.copy_to(r1 + (v337_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v342_i1 = 0; v342_i1 < 8; ++v342_i1) {
                tensorforge::intel_esimd::simd<float, 12> v345_data;
                v345_data.copy_from(r1 + ((v342_i1 * 16)));
                v345_data.copy_to(glb_m0 + ((v342_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}


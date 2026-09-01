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
              tensorforge::intel_esimd::simd<float, 64> v16_ld;
              v16_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v16_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v17_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir1[128]{};
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
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v36_acc{};
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(s0 + (0_i32));
              v36_acc += ((v40_data[0]) * v20_data);
              v36_acc += ((v40_data[1]) * v21_data);
              v36_acc += ((v40_data[2]) * v22_data);
              v36_acc += ((v40_data[3]) * v23_data);
              v36_acc += ((v40_data[4]) * v24_data);
              v36_acc += ((v40_data[5]) * v25_data);
              v36_acc += ((v40_data[6]) * v26_data);
              v36_acc += ((v40_data[7]) * v27_data);
              v36_acc += ((v40_data[8]) * v28_data);
              v36_acc += ((v40_data[9]) * v29_data);
              v36_acc += ((v40_data[10]) * v30_data);
              v36_acc += ((v40_data[11]) * v31_data);
              v36_acc += ((v40_data[12]) * v32_data);
              v36_acc += ((v40_data[13]) * v33_data);
              v36_acc += ((v40_data[14]) * v34_data);
              v36_acc += ((v40_data[15]) * v35_data);
              v36_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v73_acc{};
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(s0 + (16_i32));
              v73_acc += ((v77_data[0]) * v20_data);
              v73_acc += ((v77_data[1]) * v21_data);
              v73_acc += ((v77_data[2]) * v22_data);
              v73_acc += ((v77_data[3]) * v23_data);
              v73_acc += ((v77_data[4]) * v24_data);
              v73_acc += ((v77_data[5]) * v25_data);
              v73_acc += ((v77_data[6]) * v26_data);
              v73_acc += ((v77_data[7]) * v27_data);
              v73_acc += ((v77_data[8]) * v28_data);
              v73_acc += ((v77_data[9]) * v29_data);
              v73_acc += ((v77_data[10]) * v30_data);
              v73_acc += ((v77_data[11]) * v31_data);
              v73_acc += ((v77_data[12]) * v32_data);
              v73_acc += ((v77_data[13]) * v33_data);
              v73_acc += ((v77_data[14]) * v34_data);
              v73_acc += ((v77_data[15]) * v35_data);
              v73_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v110_acc{};
              tensorforge::intel_esimd::simd<float, 16> v114_data;
              v114_data.copy_from(s0 + (32_i32));
              v110_acc += ((v114_data[0]) * v20_data);
              v110_acc += ((v114_data[1]) * v21_data);
              v110_acc += ((v114_data[2]) * v22_data);
              v110_acc += ((v114_data[3]) * v23_data);
              v110_acc += ((v114_data[4]) * v24_data);
              v110_acc += ((v114_data[5]) * v25_data);
              v110_acc += ((v114_data[6]) * v26_data);
              v110_acc += ((v114_data[7]) * v27_data);
              v110_acc += ((v114_data[8]) * v28_data);
              v110_acc += ((v114_data[9]) * v29_data);
              v110_acc += ((v114_data[10]) * v30_data);
              v110_acc += ((v114_data[11]) * v31_data);
              v110_acc += ((v114_data[12]) * v32_data);
              v110_acc += ((v114_data[13]) * v33_data);
              v110_acc += ((v114_data[14]) * v34_data);
              v110_acc += ((v114_data[15]) * v35_data);
              v110_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v147_acc{};
              tensorforge::intel_esimd::simd<float, 16> v151_data;
              v151_data.copy_from(s0 + (48_i32));
              v147_acc += ((v151_data[0]) * v20_data);
              v147_acc += ((v151_data[1]) * v21_data);
              v147_acc += ((v151_data[2]) * v22_data);
              v147_acc += ((v151_data[3]) * v23_data);
              v147_acc += ((v151_data[4]) * v24_data);
              v147_acc += ((v151_data[5]) * v25_data);
              v147_acc += ((v151_data[6]) * v26_data);
              v147_acc += ((v151_data[7]) * v27_data);
              v147_acc += ((v151_data[8]) * v28_data);
              v147_acc += ((v151_data[9]) * v29_data);
              v147_acc += ((v151_data[10]) * v30_data);
              v147_acc += ((v151_data[11]) * v31_data);
              v147_acc += ((v151_data[12]) * v32_data);
              v147_acc += ((v151_data[13]) * v33_data);
              v147_acc += ((v151_data[14]) * v34_data);
              v147_acc += ((v151_data[15]) * v35_data);
              v147_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v184_acc{};
              tensorforge::intel_esimd::simd<float, 16> v188_data;
              v188_data.copy_from(s0 + (64_i32));
              v184_acc += ((v188_data[0]) * v20_data);
              v184_acc += ((v188_data[1]) * v21_data);
              v184_acc += ((v188_data[2]) * v22_data);
              v184_acc += ((v188_data[3]) * v23_data);
              v184_acc += ((v188_data[4]) * v24_data);
              v184_acc += ((v188_data[5]) * v25_data);
              v184_acc += ((v188_data[6]) * v26_data);
              v184_acc += ((v188_data[7]) * v27_data);
              v184_acc += ((v188_data[8]) * v28_data);
              v184_acc += ((v188_data[9]) * v29_data);
              v184_acc += ((v188_data[10]) * v30_data);
              v184_acc += ((v188_data[11]) * v31_data);
              v184_acc += ((v188_data[12]) * v32_data);
              v184_acc += ((v188_data[13]) * v33_data);
              v184_acc += ((v188_data[14]) * v34_data);
              v184_acc += ((v188_data[15]) * v35_data);
              v184_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v221_acc{};
              tensorforge::intel_esimd::simd<float, 16> v225_data;
              v225_data.copy_from(s0 + (80_i32));
              v221_acc += ((v225_data[0]) * v20_data);
              v221_acc += ((v225_data[1]) * v21_data);
              v221_acc += ((v225_data[2]) * v22_data);
              v221_acc += ((v225_data[3]) * v23_data);
              v221_acc += ((v225_data[4]) * v24_data);
              v221_acc += ((v225_data[5]) * v25_data);
              v221_acc += ((v225_data[6]) * v26_data);
              v221_acc += ((v225_data[7]) * v27_data);
              v221_acc += ((v225_data[8]) * v28_data);
              v221_acc += ((v225_data[9]) * v29_data);
              v221_acc += ((v225_data[10]) * v30_data);
              v221_acc += ((v225_data[11]) * v31_data);
              v221_acc += ((v225_data[12]) * v32_data);
              v221_acc += ((v225_data[13]) * v33_data);
              v221_acc += ((v225_data[14]) * v34_data);
              v221_acc += ((v225_data[15]) * v35_data);
              v221_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v258_acc{};
              tensorforge::intel_esimd::simd<float, 16> v262_data;
              v262_data.copy_from(s0 + (96_i32));
              v258_acc += ((v262_data[0]) * v20_data);
              v258_acc += ((v262_data[1]) * v21_data);
              v258_acc += ((v262_data[2]) * v22_data);
              v258_acc += ((v262_data[3]) * v23_data);
              v258_acc += ((v262_data[4]) * v24_data);
              v258_acc += ((v262_data[5]) * v25_data);
              v258_acc += ((v262_data[6]) * v26_data);
              v258_acc += ((v262_data[7]) * v27_data);
              v258_acc += ((v262_data[8]) * v28_data);
              v258_acc += ((v262_data[9]) * v29_data);
              v258_acc += ((v262_data[10]) * v30_data);
              v258_acc += ((v262_data[11]) * v31_data);
              v258_acc += ((v262_data[12]) * v32_data);
              v258_acc += ((v262_data[13]) * v33_data);
              v258_acc += ((v262_data[14]) * v34_data);
              v258_acc += ((v262_data[15]) * v35_data);
              v258_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v295_acc{};
              tensorforge::intel_esimd::simd<float, 16> v299_data;
              v299_data.copy_from(s0 + (112_i32));
              v295_acc += ((v299_data[0]) * v20_data);
              v295_acc += ((v299_data[1]) * v21_data);
              v295_acc += ((v299_data[2]) * v22_data);
              v295_acc += ((v299_data[3]) * v23_data);
              v295_acc += ((v299_data[4]) * v24_data);
              v295_acc += ((v299_data[5]) * v25_data);
              v295_acc += ((v299_data[6]) * v26_data);
              v295_acc += ((v299_data[7]) * v27_data);
              v295_acc += ((v299_data[8]) * v28_data);
              v295_acc += ((v299_data[9]) * v29_data);
              v295_acc += ((v299_data[10]) * v30_data);
              v295_acc += ((v299_data[11]) * v31_data);
              v295_acc += ((v299_data[12]) * v32_data);
              v295_acc += ((v299_data[13]) * v33_data);
              v295_acc += ((v299_data[14]) * v34_data);
              v295_acc += ((v299_data[15]) * v35_data);
              v295_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v332_n1 = 0; v332_n1 < 8; ++v332_n1) {
                int32_t v333_a = v332_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v335_data;
                v335_data.copy_from(ir1 + (v333_a));
                v335_data.copy_to(r1 + (v333_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v338_i1 = 0; v338_i1 < 8; ++v338_i1) {
                tensorforge::intel_esimd::simd<float, 12> v341_data;
                v341_data.copy_from(r1 + ((v338_i1 * 16)));
                v341_data.copy_to(glb_m0 + ((v338_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}


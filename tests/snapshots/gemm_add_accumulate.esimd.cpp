// === base name ===
kernel_5e7da3148f

// === header ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5e7da3148f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5e7da3148f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×16(12×16) {0..12}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 16; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 12> v11_data;
                v11_data.copy_from(glb_m1 + ((v6_i1 * 12)));
                v11_data.copy_to(r0 + ((v6_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v15_ld;
              v15_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v15_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v16_ld;
              v16_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v16_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              float r1[128]{};
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v18_i1 = 0; v18_i1 < 8; ++v18_i1) {
                tensorforge::intel_esimd::simd<float, 12> v23_data;
                v23_data.copy_from(glb_m0 + ((v18_i1 * 12)));
                v23_data.copy_to(r1 + ((v18_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              float r2[128]{};
              // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[128]{};
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v44_acc{};
              tensorforge::intel_esimd::simd<float, 16> v48_data;
              v48_data.copy_from(s0 + (0_i32));
              v44_acc += ((v48_data[0]) * v28_data);
              v44_acc += ((v48_data[1]) * v29_data);
              v44_acc += ((v48_data[2]) * v30_data);
              v44_acc += ((v48_data[3]) * v31_data);
              v44_acc += ((v48_data[4]) * v32_data);
              v44_acc += ((v48_data[5]) * v33_data);
              v44_acc += ((v48_data[6]) * v34_data);
              v44_acc += ((v48_data[7]) * v35_data);
              v44_acc += ((v48_data[8]) * v36_data);
              v44_acc += ((v48_data[9]) * v37_data);
              v44_acc += ((v48_data[10]) * v38_data);
              v44_acc += ((v48_data[11]) * v39_data);
              v44_acc += ((v48_data[12]) * v40_data);
              v44_acc += ((v48_data[13]) * v41_data);
              v44_acc += ((v48_data[14]) * v42_data);
              v44_acc += ((v48_data[15]) * v43_data);
              v44_acc.copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v81_acc{};
              tensorforge::intel_esimd::simd<float, 16> v85_data;
              v85_data.copy_from(s0 + (16_i32));
              v81_acc += ((v85_data[0]) * v28_data);
              v81_acc += ((v85_data[1]) * v29_data);
              v81_acc += ((v85_data[2]) * v30_data);
              v81_acc += ((v85_data[3]) * v31_data);
              v81_acc += ((v85_data[4]) * v32_data);
              v81_acc += ((v85_data[5]) * v33_data);
              v81_acc += ((v85_data[6]) * v34_data);
              v81_acc += ((v85_data[7]) * v35_data);
              v81_acc += ((v85_data[8]) * v36_data);
              v81_acc += ((v85_data[9]) * v37_data);
              v81_acc += ((v85_data[10]) * v38_data);
              v81_acc += ((v85_data[11]) * v39_data);
              v81_acc += ((v85_data[12]) * v40_data);
              v81_acc += ((v85_data[13]) * v41_data);
              v81_acc += ((v85_data[14]) * v42_data);
              v81_acc += ((v85_data[15]) * v43_data);
              v81_acc.copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v118_acc{};
              tensorforge::intel_esimd::simd<float, 16> v122_data;
              v122_data.copy_from(s0 + (32_i32));
              v118_acc += ((v122_data[0]) * v28_data);
              v118_acc += ((v122_data[1]) * v29_data);
              v118_acc += ((v122_data[2]) * v30_data);
              v118_acc += ((v122_data[3]) * v31_data);
              v118_acc += ((v122_data[4]) * v32_data);
              v118_acc += ((v122_data[5]) * v33_data);
              v118_acc += ((v122_data[6]) * v34_data);
              v118_acc += ((v122_data[7]) * v35_data);
              v118_acc += ((v122_data[8]) * v36_data);
              v118_acc += ((v122_data[9]) * v37_data);
              v118_acc += ((v122_data[10]) * v38_data);
              v118_acc += ((v122_data[11]) * v39_data);
              v118_acc += ((v122_data[12]) * v40_data);
              v118_acc += ((v122_data[13]) * v41_data);
              v118_acc += ((v122_data[14]) * v42_data);
              v118_acc += ((v122_data[15]) * v43_data);
              v118_acc.copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v155_acc{};
              tensorforge::intel_esimd::simd<float, 16> v159_data;
              v159_data.copy_from(s0 + (48_i32));
              v155_acc += ((v159_data[0]) * v28_data);
              v155_acc += ((v159_data[1]) * v29_data);
              v155_acc += ((v159_data[2]) * v30_data);
              v155_acc += ((v159_data[3]) * v31_data);
              v155_acc += ((v159_data[4]) * v32_data);
              v155_acc += ((v159_data[5]) * v33_data);
              v155_acc += ((v159_data[6]) * v34_data);
              v155_acc += ((v159_data[7]) * v35_data);
              v155_acc += ((v159_data[8]) * v36_data);
              v155_acc += ((v159_data[9]) * v37_data);
              v155_acc += ((v159_data[10]) * v38_data);
              v155_acc += ((v159_data[11]) * v39_data);
              v155_acc += ((v159_data[12]) * v40_data);
              v155_acc += ((v159_data[13]) * v41_data);
              v155_acc += ((v159_data[14]) * v42_data);
              v155_acc += ((v159_data[15]) * v43_data);
              v155_acc.copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v192_acc{};
              tensorforge::intel_esimd::simd<float, 16> v196_data;
              v196_data.copy_from(s0 + (64_i32));
              v192_acc += ((v196_data[0]) * v28_data);
              v192_acc += ((v196_data[1]) * v29_data);
              v192_acc += ((v196_data[2]) * v30_data);
              v192_acc += ((v196_data[3]) * v31_data);
              v192_acc += ((v196_data[4]) * v32_data);
              v192_acc += ((v196_data[5]) * v33_data);
              v192_acc += ((v196_data[6]) * v34_data);
              v192_acc += ((v196_data[7]) * v35_data);
              v192_acc += ((v196_data[8]) * v36_data);
              v192_acc += ((v196_data[9]) * v37_data);
              v192_acc += ((v196_data[10]) * v38_data);
              v192_acc += ((v196_data[11]) * v39_data);
              v192_acc += ((v196_data[12]) * v40_data);
              v192_acc += ((v196_data[13]) * v41_data);
              v192_acc += ((v196_data[14]) * v42_data);
              v192_acc += ((v196_data[15]) * v43_data);
              v192_acc.copy_to(ir2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v229_acc{};
              tensorforge::intel_esimd::simd<float, 16> v233_data;
              v233_data.copy_from(s0 + (80_i32));
              v229_acc += ((v233_data[0]) * v28_data);
              v229_acc += ((v233_data[1]) * v29_data);
              v229_acc += ((v233_data[2]) * v30_data);
              v229_acc += ((v233_data[3]) * v31_data);
              v229_acc += ((v233_data[4]) * v32_data);
              v229_acc += ((v233_data[5]) * v33_data);
              v229_acc += ((v233_data[6]) * v34_data);
              v229_acc += ((v233_data[7]) * v35_data);
              v229_acc += ((v233_data[8]) * v36_data);
              v229_acc += ((v233_data[9]) * v37_data);
              v229_acc += ((v233_data[10]) * v38_data);
              v229_acc += ((v233_data[11]) * v39_data);
              v229_acc += ((v233_data[12]) * v40_data);
              v229_acc += ((v233_data[13]) * v41_data);
              v229_acc += ((v233_data[14]) * v42_data);
              v229_acc += ((v233_data[15]) * v43_data);
              v229_acc.copy_to(ir2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v266_acc{};
              tensorforge::intel_esimd::simd<float, 16> v270_data;
              v270_data.copy_from(s0 + (96_i32));
              v266_acc += ((v270_data[0]) * v28_data);
              v266_acc += ((v270_data[1]) * v29_data);
              v266_acc += ((v270_data[2]) * v30_data);
              v266_acc += ((v270_data[3]) * v31_data);
              v266_acc += ((v270_data[4]) * v32_data);
              v266_acc += ((v270_data[5]) * v33_data);
              v266_acc += ((v270_data[6]) * v34_data);
              v266_acc += ((v270_data[7]) * v35_data);
              v266_acc += ((v270_data[8]) * v36_data);
              v266_acc += ((v270_data[9]) * v37_data);
              v266_acc += ((v270_data[10]) * v38_data);
              v266_acc += ((v270_data[11]) * v39_data);
              v266_acc += ((v270_data[12]) * v40_data);
              v266_acc += ((v270_data[13]) * v41_data);
              v266_acc += ((v270_data[14]) * v42_data);
              v266_acc += ((v270_data[15]) * v43_data);
              v266_acc.copy_to(ir2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v303_acc{};
              tensorforge::intel_esimd::simd<float, 16> v307_data;
              v307_data.copy_from(s0 + (112_i32));
              v303_acc += ((v307_data[0]) * v28_data);
              v303_acc += ((v307_data[1]) * v29_data);
              v303_acc += ((v307_data[2]) * v30_data);
              v303_acc += ((v307_data[3]) * v31_data);
              v303_acc += ((v307_data[4]) * v32_data);
              v303_acc += ((v307_data[5]) * v33_data);
              v303_acc += ((v307_data[6]) * v34_data);
              v303_acc += ((v307_data[7]) * v35_data);
              v303_acc += ((v307_data[8]) * v36_data);
              v303_acc += ((v307_data[9]) * v37_data);
              v303_acc += ((v307_data[10]) * v38_data);
              v303_acc += ((v307_data[11]) * v39_data);
              v303_acc += ((v307_data[12]) * v40_data);
              v303_acc += ((v307_data[13]) * v41_data);
              v303_acc += ((v307_data[14]) * v42_data);
              v303_acc += ((v307_data[15]) * v43_data);
              v303_acc.copy_to(ir2 + (112));
              #pragma unroll
              for (int32_t v340_n1 = 0; v340_n1 < 8; ++v340_n1) {
                int32_t v341_a = v340_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v343_data;
                v343_data.copy_from(ir2 + (v341_a));
                tensorforge::intel_esimd::simd<float, 12> v346_data;
                v346_data.copy_from(r1 + (v341_a));
                (v346_data + v343_data).copy_to(r2 + (v341_a));
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v350_i1 = 0; v350_i1 < 8; ++v350_i1) {
                tensorforge::intel_esimd::simd<float, 12> v353_data;
                v353_data.copy_from(r2 + ((v350_i1 * 16)));
                v353_data.copy_to(glb_m0 + ((v350_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}


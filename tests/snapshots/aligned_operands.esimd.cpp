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
              tensorforge::intel_esimd::simd<float, 16> v74_data;
              v74_data.copy_from(s0 + (0_i32));
              v70_acc += ((v74_data[0]) * v9_data);
              v70_acc += ((v74_data[1]) * v13_data);
              v70_acc += ((v74_data[2]) * v17_data);
              v70_acc += ((v74_data[3]) * v21_data);
              v70_acc += ((v74_data[4]) * v25_data);
              v70_acc += ((v74_data[5]) * v29_data);
              v70_acc += ((v74_data[6]) * v33_data);
              v70_acc += ((v74_data[7]) * v37_data);
              v70_acc += ((v74_data[8]) * v41_data);
              v70_acc += ((v74_data[9]) * v45_data);
              v70_acc += ((v74_data[10]) * v49_data);
              v70_acc += ((v74_data[11]) * v53_data);
              v70_acc += ((v74_data[12]) * v57_data);
              v70_acc += ((v74_data[13]) * v61_data);
              v70_acc += ((v74_data[14]) * v65_data);
              v70_acc += ((v74_data[15]) * v69_data);
              v70_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v107_acc{};
              tensorforge::intel_esimd::simd<float, 16> v111_data;
              v111_data.copy_from(s0 + (16_i32));
              v107_acc += ((v111_data[0]) * v9_data);
              v107_acc += ((v111_data[1]) * v13_data);
              v107_acc += ((v111_data[2]) * v17_data);
              v107_acc += ((v111_data[3]) * v21_data);
              v107_acc += ((v111_data[4]) * v25_data);
              v107_acc += ((v111_data[5]) * v29_data);
              v107_acc += ((v111_data[6]) * v33_data);
              v107_acc += ((v111_data[7]) * v37_data);
              v107_acc += ((v111_data[8]) * v41_data);
              v107_acc += ((v111_data[9]) * v45_data);
              v107_acc += ((v111_data[10]) * v49_data);
              v107_acc += ((v111_data[11]) * v53_data);
              v107_acc += ((v111_data[12]) * v57_data);
              v107_acc += ((v111_data[13]) * v61_data);
              v107_acc += ((v111_data[14]) * v65_data);
              v107_acc += ((v111_data[15]) * v69_data);
              v107_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v144_acc{};
              tensorforge::intel_esimd::simd<float, 16> v148_data;
              v148_data.copy_from(s0 + (32_i32));
              v144_acc += ((v148_data[0]) * v9_data);
              v144_acc += ((v148_data[1]) * v13_data);
              v144_acc += ((v148_data[2]) * v17_data);
              v144_acc += ((v148_data[3]) * v21_data);
              v144_acc += ((v148_data[4]) * v25_data);
              v144_acc += ((v148_data[5]) * v29_data);
              v144_acc += ((v148_data[6]) * v33_data);
              v144_acc += ((v148_data[7]) * v37_data);
              v144_acc += ((v148_data[8]) * v41_data);
              v144_acc += ((v148_data[9]) * v45_data);
              v144_acc += ((v148_data[10]) * v49_data);
              v144_acc += ((v148_data[11]) * v53_data);
              v144_acc += ((v148_data[12]) * v57_data);
              v144_acc += ((v148_data[13]) * v61_data);
              v144_acc += ((v148_data[14]) * v65_data);
              v144_acc += ((v148_data[15]) * v69_data);
              v144_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v181_acc{};
              tensorforge::intel_esimd::simd<float, 16> v185_data;
              v185_data.copy_from(s0 + (48_i32));
              v181_acc += ((v185_data[0]) * v9_data);
              v181_acc += ((v185_data[1]) * v13_data);
              v181_acc += ((v185_data[2]) * v17_data);
              v181_acc += ((v185_data[3]) * v21_data);
              v181_acc += ((v185_data[4]) * v25_data);
              v181_acc += ((v185_data[5]) * v29_data);
              v181_acc += ((v185_data[6]) * v33_data);
              v181_acc += ((v185_data[7]) * v37_data);
              v181_acc += ((v185_data[8]) * v41_data);
              v181_acc += ((v185_data[9]) * v45_data);
              v181_acc += ((v185_data[10]) * v49_data);
              v181_acc += ((v185_data[11]) * v53_data);
              v181_acc += ((v185_data[12]) * v57_data);
              v181_acc += ((v185_data[13]) * v61_data);
              v181_acc += ((v185_data[14]) * v65_data);
              v181_acc += ((v185_data[15]) * v69_data);
              v181_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v218_acc{};
              tensorforge::intel_esimd::simd<float, 16> v222_data;
              v222_data.copy_from(s0 + (64_i32));
              v218_acc += ((v222_data[0]) * v9_data);
              v218_acc += ((v222_data[1]) * v13_data);
              v218_acc += ((v222_data[2]) * v17_data);
              v218_acc += ((v222_data[3]) * v21_data);
              v218_acc += ((v222_data[4]) * v25_data);
              v218_acc += ((v222_data[5]) * v29_data);
              v218_acc += ((v222_data[6]) * v33_data);
              v218_acc += ((v222_data[7]) * v37_data);
              v218_acc += ((v222_data[8]) * v41_data);
              v218_acc += ((v222_data[9]) * v45_data);
              v218_acc += ((v222_data[10]) * v49_data);
              v218_acc += ((v222_data[11]) * v53_data);
              v218_acc += ((v222_data[12]) * v57_data);
              v218_acc += ((v222_data[13]) * v61_data);
              v218_acc += ((v222_data[14]) * v65_data);
              v218_acc += ((v222_data[15]) * v69_data);
              v218_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v255_acc{};
              tensorforge::intel_esimd::simd<float, 16> v259_data;
              v259_data.copy_from(s0 + (80_i32));
              v255_acc += ((v259_data[0]) * v9_data);
              v255_acc += ((v259_data[1]) * v13_data);
              v255_acc += ((v259_data[2]) * v17_data);
              v255_acc += ((v259_data[3]) * v21_data);
              v255_acc += ((v259_data[4]) * v25_data);
              v255_acc += ((v259_data[5]) * v29_data);
              v255_acc += ((v259_data[6]) * v33_data);
              v255_acc += ((v259_data[7]) * v37_data);
              v255_acc += ((v259_data[8]) * v41_data);
              v255_acc += ((v259_data[9]) * v45_data);
              v255_acc += ((v259_data[10]) * v49_data);
              v255_acc += ((v259_data[11]) * v53_data);
              v255_acc += ((v259_data[12]) * v57_data);
              v255_acc += ((v259_data[13]) * v61_data);
              v255_acc += ((v259_data[14]) * v65_data);
              v255_acc += ((v259_data[15]) * v69_data);
              v255_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v292_acc{};
              tensorforge::intel_esimd::simd<float, 16> v296_data;
              v296_data.copy_from(s0 + (96_i32));
              v292_acc += ((v296_data[0]) * v9_data);
              v292_acc += ((v296_data[1]) * v13_data);
              v292_acc += ((v296_data[2]) * v17_data);
              v292_acc += ((v296_data[3]) * v21_data);
              v292_acc += ((v296_data[4]) * v25_data);
              v292_acc += ((v296_data[5]) * v29_data);
              v292_acc += ((v296_data[6]) * v33_data);
              v292_acc += ((v296_data[7]) * v37_data);
              v292_acc += ((v296_data[8]) * v41_data);
              v292_acc += ((v296_data[9]) * v45_data);
              v292_acc += ((v296_data[10]) * v49_data);
              v292_acc += ((v296_data[11]) * v53_data);
              v292_acc += ((v296_data[12]) * v57_data);
              v292_acc += ((v296_data[13]) * v61_data);
              v292_acc += ((v296_data[14]) * v65_data);
              v292_acc += ((v296_data[15]) * v69_data);
              v292_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v329_acc{};
              tensorforge::intel_esimd::simd<float, 16> v333_data;
              v333_data.copy_from(s0 + (112_i32));
              v329_acc += ((v333_data[0]) * v9_data);
              v329_acc += ((v333_data[1]) * v13_data);
              v329_acc += ((v333_data[2]) * v17_data);
              v329_acc += ((v333_data[3]) * v21_data);
              v329_acc += ((v333_data[4]) * v25_data);
              v329_acc += ((v333_data[5]) * v29_data);
              v329_acc += ((v333_data[6]) * v33_data);
              v329_acc += ((v333_data[7]) * v37_data);
              v329_acc += ((v333_data[8]) * v41_data);
              v329_acc += ((v333_data[9]) * v45_data);
              v329_acc += ((v333_data[10]) * v49_data);
              v329_acc += ((v333_data[11]) * v53_data);
              v329_acc += ((v333_data[12]) * v57_data);
              v329_acc += ((v333_data[13]) * v61_data);
              v329_acc += ((v333_data[14]) * v65_data);
              v329_acc += ((v333_data[15]) * v69_data);
              v329_acc.copy_to(ir0 + (112));
              #pragma unroll
              for (int32_t v366_n0 = 0; v366_n0 < 1; ++v366_n0) {
                int32_t v368_a = v366_n0 * 16;
                #pragma unroll
                for (int32_t v367_n1 = 0; v367_n1 < 8; ++v367_n1) {
                  int32_t v370_a = v368_a + (v367_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v371_data;
                  v371_data.copy_from(ir0 + (v370_a));
                  v371_data.copy_to(r0 + (v370_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v375_i0 = 0; v375_i0 < 1; ++v375_i0) {
                int32_t v377_a = v375_i0 * 16;
                #pragma unroll
                for (int32_t v376_i1 = 0; v376_i1 < 8; ++v376_i1) {
                  int32_t v379_a = v377_a + (v376_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v380_data;
                  v380_data.copy_from(r0 + (v379_a));
                  v380_data.copy_to(glb_m0 + (v379_a));
                }
              }
            }
          }
        }
      });
    }
  });
}


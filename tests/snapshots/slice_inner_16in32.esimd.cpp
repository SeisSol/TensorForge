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
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[128]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir0[128]{};
              tensorforge::intel_esimd::simd<float, 16> v10_data;
              v10_data.copy_from(glb_m1 + (264_i32));
              tensorforge::intel_esimd::simd<float, 16> v15_data;
              v15_data.copy_from(glb_m1 + (296_i32));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(glb_m1 + (328_i32));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(glb_m1 + (360_i32));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(glb_m1 + (392_i32));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(glb_m1 + (424_i32));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(glb_m1 + (456_i32));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(glb_m1 + (488_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(glb_m1 + (520_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(glb_m1 + (552_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(glb_m1 + (584_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data;
              v65_data.copy_from(glb_m1 + (616_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data;
              v70_data.copy_from(glb_m1 + (648_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data;
              v75_data.copy_from(glb_m1 + (680_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data;
              v80_data.copy_from(glb_m1 + (712_i32));
              tensorforge::intel_esimd::simd<float, 16> v85_data;
              v85_data.copy_from(glb_m1 + (744_i32));
              tensorforge::intel_esimd::simd<float, 16> v86_acc{};
              tensorforge::intel_esimd::simd<float, 16> v90_data;
              v90_data.copy_from(s0 + (0_i32));
              v86_acc += ((v90_data[0]) * v10_data);
              v86_acc += ((v90_data[1]) * v15_data);
              v86_acc += ((v90_data[2]) * v20_data);
              v86_acc += ((v90_data[3]) * v25_data);
              v86_acc += ((v90_data[4]) * v30_data);
              v86_acc += ((v90_data[5]) * v35_data);
              v86_acc += ((v90_data[6]) * v40_data);
              v86_acc += ((v90_data[7]) * v45_data);
              v86_acc += ((v90_data[8]) * v50_data);
              v86_acc += ((v90_data[9]) * v55_data);
              v86_acc += ((v90_data[10]) * v60_data);
              v86_acc += ((v90_data[11]) * v65_data);
              v86_acc += ((v90_data[12]) * v70_data);
              v86_acc += ((v90_data[13]) * v75_data);
              v86_acc += ((v90_data[14]) * v80_data);
              v86_acc += ((v90_data[15]) * v85_data);
              v86_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v123_acc{};
              tensorforge::intel_esimd::simd<float, 16> v127_data;
              v127_data.copy_from(s0 + (16_i32));
              v123_acc += ((v127_data[0]) * v10_data);
              v123_acc += ((v127_data[1]) * v15_data);
              v123_acc += ((v127_data[2]) * v20_data);
              v123_acc += ((v127_data[3]) * v25_data);
              v123_acc += ((v127_data[4]) * v30_data);
              v123_acc += ((v127_data[5]) * v35_data);
              v123_acc += ((v127_data[6]) * v40_data);
              v123_acc += ((v127_data[7]) * v45_data);
              v123_acc += ((v127_data[8]) * v50_data);
              v123_acc += ((v127_data[9]) * v55_data);
              v123_acc += ((v127_data[10]) * v60_data);
              v123_acc += ((v127_data[11]) * v65_data);
              v123_acc += ((v127_data[12]) * v70_data);
              v123_acc += ((v127_data[13]) * v75_data);
              v123_acc += ((v127_data[14]) * v80_data);
              v123_acc += ((v127_data[15]) * v85_data);
              v123_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v160_acc{};
              tensorforge::intel_esimd::simd<float, 16> v164_data;
              v164_data.copy_from(s0 + (32_i32));
              v160_acc += ((v164_data[0]) * v10_data);
              v160_acc += ((v164_data[1]) * v15_data);
              v160_acc += ((v164_data[2]) * v20_data);
              v160_acc += ((v164_data[3]) * v25_data);
              v160_acc += ((v164_data[4]) * v30_data);
              v160_acc += ((v164_data[5]) * v35_data);
              v160_acc += ((v164_data[6]) * v40_data);
              v160_acc += ((v164_data[7]) * v45_data);
              v160_acc += ((v164_data[8]) * v50_data);
              v160_acc += ((v164_data[9]) * v55_data);
              v160_acc += ((v164_data[10]) * v60_data);
              v160_acc += ((v164_data[11]) * v65_data);
              v160_acc += ((v164_data[12]) * v70_data);
              v160_acc += ((v164_data[13]) * v75_data);
              v160_acc += ((v164_data[14]) * v80_data);
              v160_acc += ((v164_data[15]) * v85_data);
              v160_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v197_acc{};
              tensorforge::intel_esimd::simd<float, 16> v201_data;
              v201_data.copy_from(s0 + (48_i32));
              v197_acc += ((v201_data[0]) * v10_data);
              v197_acc += ((v201_data[1]) * v15_data);
              v197_acc += ((v201_data[2]) * v20_data);
              v197_acc += ((v201_data[3]) * v25_data);
              v197_acc += ((v201_data[4]) * v30_data);
              v197_acc += ((v201_data[5]) * v35_data);
              v197_acc += ((v201_data[6]) * v40_data);
              v197_acc += ((v201_data[7]) * v45_data);
              v197_acc += ((v201_data[8]) * v50_data);
              v197_acc += ((v201_data[9]) * v55_data);
              v197_acc += ((v201_data[10]) * v60_data);
              v197_acc += ((v201_data[11]) * v65_data);
              v197_acc += ((v201_data[12]) * v70_data);
              v197_acc += ((v201_data[13]) * v75_data);
              v197_acc += ((v201_data[14]) * v80_data);
              v197_acc += ((v201_data[15]) * v85_data);
              v197_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v234_acc{};
              tensorforge::intel_esimd::simd<float, 16> v238_data;
              v238_data.copy_from(s0 + (64_i32));
              v234_acc += ((v238_data[0]) * v10_data);
              v234_acc += ((v238_data[1]) * v15_data);
              v234_acc += ((v238_data[2]) * v20_data);
              v234_acc += ((v238_data[3]) * v25_data);
              v234_acc += ((v238_data[4]) * v30_data);
              v234_acc += ((v238_data[5]) * v35_data);
              v234_acc += ((v238_data[6]) * v40_data);
              v234_acc += ((v238_data[7]) * v45_data);
              v234_acc += ((v238_data[8]) * v50_data);
              v234_acc += ((v238_data[9]) * v55_data);
              v234_acc += ((v238_data[10]) * v60_data);
              v234_acc += ((v238_data[11]) * v65_data);
              v234_acc += ((v238_data[12]) * v70_data);
              v234_acc += ((v238_data[13]) * v75_data);
              v234_acc += ((v238_data[14]) * v80_data);
              v234_acc += ((v238_data[15]) * v85_data);
              v234_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v271_acc{};
              tensorforge::intel_esimd::simd<float, 16> v275_data;
              v275_data.copy_from(s0 + (80_i32));
              v271_acc += ((v275_data[0]) * v10_data);
              v271_acc += ((v275_data[1]) * v15_data);
              v271_acc += ((v275_data[2]) * v20_data);
              v271_acc += ((v275_data[3]) * v25_data);
              v271_acc += ((v275_data[4]) * v30_data);
              v271_acc += ((v275_data[5]) * v35_data);
              v271_acc += ((v275_data[6]) * v40_data);
              v271_acc += ((v275_data[7]) * v45_data);
              v271_acc += ((v275_data[8]) * v50_data);
              v271_acc += ((v275_data[9]) * v55_data);
              v271_acc += ((v275_data[10]) * v60_data);
              v271_acc += ((v275_data[11]) * v65_data);
              v271_acc += ((v275_data[12]) * v70_data);
              v271_acc += ((v275_data[13]) * v75_data);
              v271_acc += ((v275_data[14]) * v80_data);
              v271_acc += ((v275_data[15]) * v85_data);
              v271_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v308_acc{};
              tensorforge::intel_esimd::simd<float, 16> v312_data;
              v312_data.copy_from(s0 + (96_i32));
              v308_acc += ((v312_data[0]) * v10_data);
              v308_acc += ((v312_data[1]) * v15_data);
              v308_acc += ((v312_data[2]) * v20_data);
              v308_acc += ((v312_data[3]) * v25_data);
              v308_acc += ((v312_data[4]) * v30_data);
              v308_acc += ((v312_data[5]) * v35_data);
              v308_acc += ((v312_data[6]) * v40_data);
              v308_acc += ((v312_data[7]) * v45_data);
              v308_acc += ((v312_data[8]) * v50_data);
              v308_acc += ((v312_data[9]) * v55_data);
              v308_acc += ((v312_data[10]) * v60_data);
              v308_acc += ((v312_data[11]) * v65_data);
              v308_acc += ((v312_data[12]) * v70_data);
              v308_acc += ((v312_data[13]) * v75_data);
              v308_acc += ((v312_data[14]) * v80_data);
              v308_acc += ((v312_data[15]) * v85_data);
              v308_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v345_acc{};
              tensorforge::intel_esimd::simd<float, 16> v349_data;
              v349_data.copy_from(s0 + (112_i32));
              v345_acc += ((v349_data[0]) * v10_data);
              v345_acc += ((v349_data[1]) * v15_data);
              v345_acc += ((v349_data[2]) * v20_data);
              v345_acc += ((v349_data[3]) * v25_data);
              v345_acc += ((v349_data[4]) * v30_data);
              v345_acc += ((v349_data[5]) * v35_data);
              v345_acc += ((v349_data[6]) * v40_data);
              v345_acc += ((v349_data[7]) * v45_data);
              v345_acc += ((v349_data[8]) * v50_data);
              v345_acc += ((v349_data[9]) * v55_data);
              v345_acc += ((v349_data[10]) * v60_data);
              v345_acc += ((v349_data[11]) * v65_data);
              v345_acc += ((v349_data[12]) * v70_data);
              v345_acc += ((v349_data[13]) * v75_data);
              v345_acc += ((v349_data[14]) * v80_data);
              v345_acc += ((v349_data[15]) * v85_data);
              v345_acc.copy_to(ir0 + (112));
              #pragma unroll
              for (int32_t v382_n0 = 0; v382_n0 < 1; ++v382_n0) {
                int32_t v384_a = v382_n0 * 16;
                #pragma unroll
                for (int32_t v383_n1 = 0; v383_n1 < 8; ++v383_n1) {
                  int32_t v386_a = v384_a + (v383_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v387_data;
                  v387_data.copy_from(ir0 + (v386_a));
                  v387_data.copy_to(r0 + (v386_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v391_i0 = 0; v391_i0 < 1; ++v391_i0) {
                int32_t v393_a = v391_i0 * 16;
                #pragma unroll
                for (int32_t v392_i1 = 0; v392_i1 < 8; ++v392_i1) {
                  int32_t v395_a = v393_a + (v392_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v396_data;
                  v396_data.copy_from(r0 + (v395_a));
                  v396_data.copy_to(glb_m0 + (v395_a));
                }
              }
            }
          }
        }
      });
    }
  });
}


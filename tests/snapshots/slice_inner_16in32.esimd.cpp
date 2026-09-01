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
              tensorforge::intel_esimd::simd<float, 64> v19_ld;
              v19_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v19_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v20_ld;
              v20_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v20_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v39_acc{};
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(s0 + (0_i32));
              v39_acc += ((v43_data[0]) * v23_data);
              v39_acc += ((v43_data[1]) * v24_data);
              v39_acc += ((v43_data[2]) * v25_data);
              v39_acc += ((v43_data[3]) * v26_data);
              v39_acc += ((v43_data[4]) * v27_data);
              v39_acc += ((v43_data[5]) * v28_data);
              v39_acc += ((v43_data[6]) * v29_data);
              v39_acc += ((v43_data[7]) * v30_data);
              v39_acc += ((v43_data[8]) * v31_data);
              v39_acc += ((v43_data[9]) * v32_data);
              v39_acc += ((v43_data[10]) * v33_data);
              v39_acc += ((v43_data[11]) * v34_data);
              v39_acc += ((v43_data[12]) * v35_data);
              v39_acc += ((v43_data[13]) * v36_data);
              v39_acc += ((v43_data[14]) * v37_data);
              v39_acc += ((v43_data[15]) * v38_data);
              v39_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v76_acc{};
              tensorforge::intel_esimd::simd<float, 16> v80_data;
              v80_data.copy_from(s0 + (16_i32));
              v76_acc += ((v80_data[0]) * v23_data);
              v76_acc += ((v80_data[1]) * v24_data);
              v76_acc += ((v80_data[2]) * v25_data);
              v76_acc += ((v80_data[3]) * v26_data);
              v76_acc += ((v80_data[4]) * v27_data);
              v76_acc += ((v80_data[5]) * v28_data);
              v76_acc += ((v80_data[6]) * v29_data);
              v76_acc += ((v80_data[7]) * v30_data);
              v76_acc += ((v80_data[8]) * v31_data);
              v76_acc += ((v80_data[9]) * v32_data);
              v76_acc += ((v80_data[10]) * v33_data);
              v76_acc += ((v80_data[11]) * v34_data);
              v76_acc += ((v80_data[12]) * v35_data);
              v76_acc += ((v80_data[13]) * v36_data);
              v76_acc += ((v80_data[14]) * v37_data);
              v76_acc += ((v80_data[15]) * v38_data);
              v76_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v113_acc{};
              tensorforge::intel_esimd::simd<float, 16> v117_data;
              v117_data.copy_from(s0 + (32_i32));
              v113_acc += ((v117_data[0]) * v23_data);
              v113_acc += ((v117_data[1]) * v24_data);
              v113_acc += ((v117_data[2]) * v25_data);
              v113_acc += ((v117_data[3]) * v26_data);
              v113_acc += ((v117_data[4]) * v27_data);
              v113_acc += ((v117_data[5]) * v28_data);
              v113_acc += ((v117_data[6]) * v29_data);
              v113_acc += ((v117_data[7]) * v30_data);
              v113_acc += ((v117_data[8]) * v31_data);
              v113_acc += ((v117_data[9]) * v32_data);
              v113_acc += ((v117_data[10]) * v33_data);
              v113_acc += ((v117_data[11]) * v34_data);
              v113_acc += ((v117_data[12]) * v35_data);
              v113_acc += ((v117_data[13]) * v36_data);
              v113_acc += ((v117_data[14]) * v37_data);
              v113_acc += ((v117_data[15]) * v38_data);
              v113_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v150_acc{};
              tensorforge::intel_esimd::simd<float, 16> v154_data;
              v154_data.copy_from(s0 + (48_i32));
              v150_acc += ((v154_data[0]) * v23_data);
              v150_acc += ((v154_data[1]) * v24_data);
              v150_acc += ((v154_data[2]) * v25_data);
              v150_acc += ((v154_data[3]) * v26_data);
              v150_acc += ((v154_data[4]) * v27_data);
              v150_acc += ((v154_data[5]) * v28_data);
              v150_acc += ((v154_data[6]) * v29_data);
              v150_acc += ((v154_data[7]) * v30_data);
              v150_acc += ((v154_data[8]) * v31_data);
              v150_acc += ((v154_data[9]) * v32_data);
              v150_acc += ((v154_data[10]) * v33_data);
              v150_acc += ((v154_data[11]) * v34_data);
              v150_acc += ((v154_data[12]) * v35_data);
              v150_acc += ((v154_data[13]) * v36_data);
              v150_acc += ((v154_data[14]) * v37_data);
              v150_acc += ((v154_data[15]) * v38_data);
              v150_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v187_acc{};
              tensorforge::intel_esimd::simd<float, 16> v191_data;
              v191_data.copy_from(s0 + (64_i32));
              v187_acc += ((v191_data[0]) * v23_data);
              v187_acc += ((v191_data[1]) * v24_data);
              v187_acc += ((v191_data[2]) * v25_data);
              v187_acc += ((v191_data[3]) * v26_data);
              v187_acc += ((v191_data[4]) * v27_data);
              v187_acc += ((v191_data[5]) * v28_data);
              v187_acc += ((v191_data[6]) * v29_data);
              v187_acc += ((v191_data[7]) * v30_data);
              v187_acc += ((v191_data[8]) * v31_data);
              v187_acc += ((v191_data[9]) * v32_data);
              v187_acc += ((v191_data[10]) * v33_data);
              v187_acc += ((v191_data[11]) * v34_data);
              v187_acc += ((v191_data[12]) * v35_data);
              v187_acc += ((v191_data[13]) * v36_data);
              v187_acc += ((v191_data[14]) * v37_data);
              v187_acc += ((v191_data[15]) * v38_data);
              v187_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v224_acc{};
              tensorforge::intel_esimd::simd<float, 16> v228_data;
              v228_data.copy_from(s0 + (80_i32));
              v224_acc += ((v228_data[0]) * v23_data);
              v224_acc += ((v228_data[1]) * v24_data);
              v224_acc += ((v228_data[2]) * v25_data);
              v224_acc += ((v228_data[3]) * v26_data);
              v224_acc += ((v228_data[4]) * v27_data);
              v224_acc += ((v228_data[5]) * v28_data);
              v224_acc += ((v228_data[6]) * v29_data);
              v224_acc += ((v228_data[7]) * v30_data);
              v224_acc += ((v228_data[8]) * v31_data);
              v224_acc += ((v228_data[9]) * v32_data);
              v224_acc += ((v228_data[10]) * v33_data);
              v224_acc += ((v228_data[11]) * v34_data);
              v224_acc += ((v228_data[12]) * v35_data);
              v224_acc += ((v228_data[13]) * v36_data);
              v224_acc += ((v228_data[14]) * v37_data);
              v224_acc += ((v228_data[15]) * v38_data);
              v224_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v261_acc{};
              tensorforge::intel_esimd::simd<float, 16> v265_data;
              v265_data.copy_from(s0 + (96_i32));
              v261_acc += ((v265_data[0]) * v23_data);
              v261_acc += ((v265_data[1]) * v24_data);
              v261_acc += ((v265_data[2]) * v25_data);
              v261_acc += ((v265_data[3]) * v26_data);
              v261_acc += ((v265_data[4]) * v27_data);
              v261_acc += ((v265_data[5]) * v28_data);
              v261_acc += ((v265_data[6]) * v29_data);
              v261_acc += ((v265_data[7]) * v30_data);
              v261_acc += ((v265_data[8]) * v31_data);
              v261_acc += ((v265_data[9]) * v32_data);
              v261_acc += ((v265_data[10]) * v33_data);
              v261_acc += ((v265_data[11]) * v34_data);
              v261_acc += ((v265_data[12]) * v35_data);
              v261_acc += ((v265_data[13]) * v36_data);
              v261_acc += ((v265_data[14]) * v37_data);
              v261_acc += ((v265_data[15]) * v38_data);
              v261_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v298_acc{};
              tensorforge::intel_esimd::simd<float, 16> v302_data;
              v302_data.copy_from(s0 + (112_i32));
              v298_acc += ((v302_data[0]) * v23_data);
              v298_acc += ((v302_data[1]) * v24_data);
              v298_acc += ((v302_data[2]) * v25_data);
              v298_acc += ((v302_data[3]) * v26_data);
              v298_acc += ((v302_data[4]) * v27_data);
              v298_acc += ((v302_data[5]) * v28_data);
              v298_acc += ((v302_data[6]) * v29_data);
              v298_acc += ((v302_data[7]) * v30_data);
              v298_acc += ((v302_data[8]) * v31_data);
              v298_acc += ((v302_data[9]) * v32_data);
              v298_acc += ((v302_data[10]) * v33_data);
              v298_acc += ((v302_data[11]) * v34_data);
              v298_acc += ((v302_data[12]) * v35_data);
              v298_acc += ((v302_data[13]) * v36_data);
              v298_acc += ((v302_data[14]) * v37_data);
              v298_acc += ((v302_data[15]) * v38_data);
              v298_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v335_n0 = 0; v335_n0 < 1; ++v335_n0) {
                int32_t v337_a = v335_n0 * 16;
                #pragma unroll
                for (int32_t v336_n1 = 0; v336_n1 < 8; ++v336_n1) {
                  int32_t v339_a = v337_a + (v336_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v340_data;
                  v340_data.copy_from(ir1 + (v339_a));
                  v340_data.copy_to(r1 + (v339_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v344_i0 = 0; v344_i0 < 1; ++v344_i0) {
                int32_t v346_a = v344_i0 * 16;
                #pragma unroll
                for (int32_t v345_i1 = 0; v345_i1 < 8; ++v345_i1) {
                  int32_t v348_a = v346_a + (v345_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v349_data;
                  v349_data.copy_from(r1 + (v348_a));
                  v349_data.copy_to(glb_m0 + (v348_a));
                }
              }
            }
          }
        }
      });
    }
  });
}


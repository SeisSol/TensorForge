// === base name ===
kernel_93ef4bb989135a04

// === header ===
void launcher_kernel_93ef4bb989135a04(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_93ef4bb989135a04(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_93ef4bb989135a04(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_93ef4bb989135a04(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                int32_t v13_lead = v11_i0 * 16;
                int32_t v15_off = v13_lead + 8;
                #pragma unroll
                for (int32_t v12_i1 = 8; v12_i1 < 24; ++v12_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v18_data;
                  v18_data.copy_from(glb_m1 + ((v15_off + (v12_i1 * 32))));
                  v18_data.copy_to(r0 + ((v13_lead + ((v12_i1 - 8) * 16))));
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v23_ld;
              v23_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v23_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v24_ld;
              v24_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v24_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v43_acc{};
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(s0 + (0_i32));
              v43_acc += ((v47_data[0]) * v27_data);
              v43_acc += ((v47_data[1]) * v28_data);
              v43_acc += ((v47_data[2]) * v29_data);
              v43_acc += ((v47_data[3]) * v30_data);
              v43_acc += ((v47_data[4]) * v31_data);
              v43_acc += ((v47_data[5]) * v32_data);
              v43_acc += ((v47_data[6]) * v33_data);
              v43_acc += ((v47_data[7]) * v34_data);
              v43_acc += ((v47_data[8]) * v35_data);
              v43_acc += ((v47_data[9]) * v36_data);
              v43_acc += ((v47_data[10]) * v37_data);
              v43_acc += ((v47_data[11]) * v38_data);
              v43_acc += ((v47_data[12]) * v39_data);
              v43_acc += ((v47_data[13]) * v40_data);
              v43_acc += ((v47_data[14]) * v41_data);
              v43_acc += ((v47_data[15]) * v42_data);
              v43_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v80_acc{};
              tensorforge::intel_esimd::simd<float, 16> v84_data;
              v84_data.copy_from(s0 + (16_i32));
              v80_acc += ((v84_data[0]) * v27_data);
              v80_acc += ((v84_data[1]) * v28_data);
              v80_acc += ((v84_data[2]) * v29_data);
              v80_acc += ((v84_data[3]) * v30_data);
              v80_acc += ((v84_data[4]) * v31_data);
              v80_acc += ((v84_data[5]) * v32_data);
              v80_acc += ((v84_data[6]) * v33_data);
              v80_acc += ((v84_data[7]) * v34_data);
              v80_acc += ((v84_data[8]) * v35_data);
              v80_acc += ((v84_data[9]) * v36_data);
              v80_acc += ((v84_data[10]) * v37_data);
              v80_acc += ((v84_data[11]) * v38_data);
              v80_acc += ((v84_data[12]) * v39_data);
              v80_acc += ((v84_data[13]) * v40_data);
              v80_acc += ((v84_data[14]) * v41_data);
              v80_acc += ((v84_data[15]) * v42_data);
              v80_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v117_acc{};
              tensorforge::intel_esimd::simd<float, 16> v121_data;
              v121_data.copy_from(s0 + (32_i32));
              v117_acc += ((v121_data[0]) * v27_data);
              v117_acc += ((v121_data[1]) * v28_data);
              v117_acc += ((v121_data[2]) * v29_data);
              v117_acc += ((v121_data[3]) * v30_data);
              v117_acc += ((v121_data[4]) * v31_data);
              v117_acc += ((v121_data[5]) * v32_data);
              v117_acc += ((v121_data[6]) * v33_data);
              v117_acc += ((v121_data[7]) * v34_data);
              v117_acc += ((v121_data[8]) * v35_data);
              v117_acc += ((v121_data[9]) * v36_data);
              v117_acc += ((v121_data[10]) * v37_data);
              v117_acc += ((v121_data[11]) * v38_data);
              v117_acc += ((v121_data[12]) * v39_data);
              v117_acc += ((v121_data[13]) * v40_data);
              v117_acc += ((v121_data[14]) * v41_data);
              v117_acc += ((v121_data[15]) * v42_data);
              v117_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v154_acc{};
              tensorforge::intel_esimd::simd<float, 16> v158_data;
              v158_data.copy_from(s0 + (48_i32));
              v154_acc += ((v158_data[0]) * v27_data);
              v154_acc += ((v158_data[1]) * v28_data);
              v154_acc += ((v158_data[2]) * v29_data);
              v154_acc += ((v158_data[3]) * v30_data);
              v154_acc += ((v158_data[4]) * v31_data);
              v154_acc += ((v158_data[5]) * v32_data);
              v154_acc += ((v158_data[6]) * v33_data);
              v154_acc += ((v158_data[7]) * v34_data);
              v154_acc += ((v158_data[8]) * v35_data);
              v154_acc += ((v158_data[9]) * v36_data);
              v154_acc += ((v158_data[10]) * v37_data);
              v154_acc += ((v158_data[11]) * v38_data);
              v154_acc += ((v158_data[12]) * v39_data);
              v154_acc += ((v158_data[13]) * v40_data);
              v154_acc += ((v158_data[14]) * v41_data);
              v154_acc += ((v158_data[15]) * v42_data);
              v154_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v191_acc{};
              tensorforge::intel_esimd::simd<float, 16> v195_data;
              v195_data.copy_from(s0 + (64_i32));
              v191_acc += ((v195_data[0]) * v27_data);
              v191_acc += ((v195_data[1]) * v28_data);
              v191_acc += ((v195_data[2]) * v29_data);
              v191_acc += ((v195_data[3]) * v30_data);
              v191_acc += ((v195_data[4]) * v31_data);
              v191_acc += ((v195_data[5]) * v32_data);
              v191_acc += ((v195_data[6]) * v33_data);
              v191_acc += ((v195_data[7]) * v34_data);
              v191_acc += ((v195_data[8]) * v35_data);
              v191_acc += ((v195_data[9]) * v36_data);
              v191_acc += ((v195_data[10]) * v37_data);
              v191_acc += ((v195_data[11]) * v38_data);
              v191_acc += ((v195_data[12]) * v39_data);
              v191_acc += ((v195_data[13]) * v40_data);
              v191_acc += ((v195_data[14]) * v41_data);
              v191_acc += ((v195_data[15]) * v42_data);
              v191_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              tensorforge::intel_esimd::simd<float, 16> v232_data;
              v232_data.copy_from(s0 + (80_i32));
              v228_acc += ((v232_data[0]) * v27_data);
              v228_acc += ((v232_data[1]) * v28_data);
              v228_acc += ((v232_data[2]) * v29_data);
              v228_acc += ((v232_data[3]) * v30_data);
              v228_acc += ((v232_data[4]) * v31_data);
              v228_acc += ((v232_data[5]) * v32_data);
              v228_acc += ((v232_data[6]) * v33_data);
              v228_acc += ((v232_data[7]) * v34_data);
              v228_acc += ((v232_data[8]) * v35_data);
              v228_acc += ((v232_data[9]) * v36_data);
              v228_acc += ((v232_data[10]) * v37_data);
              v228_acc += ((v232_data[11]) * v38_data);
              v228_acc += ((v232_data[12]) * v39_data);
              v228_acc += ((v232_data[13]) * v40_data);
              v228_acc += ((v232_data[14]) * v41_data);
              v228_acc += ((v232_data[15]) * v42_data);
              v228_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v265_acc{};
              tensorforge::intel_esimd::simd<float, 16> v269_data;
              v269_data.copy_from(s0 + (96_i32));
              v265_acc += ((v269_data[0]) * v27_data);
              v265_acc += ((v269_data[1]) * v28_data);
              v265_acc += ((v269_data[2]) * v29_data);
              v265_acc += ((v269_data[3]) * v30_data);
              v265_acc += ((v269_data[4]) * v31_data);
              v265_acc += ((v269_data[5]) * v32_data);
              v265_acc += ((v269_data[6]) * v33_data);
              v265_acc += ((v269_data[7]) * v34_data);
              v265_acc += ((v269_data[8]) * v35_data);
              v265_acc += ((v269_data[9]) * v36_data);
              v265_acc += ((v269_data[10]) * v37_data);
              v265_acc += ((v269_data[11]) * v38_data);
              v265_acc += ((v269_data[12]) * v39_data);
              v265_acc += ((v269_data[13]) * v40_data);
              v265_acc += ((v269_data[14]) * v41_data);
              v265_acc += ((v269_data[15]) * v42_data);
              v265_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v302_acc{};
              tensorforge::intel_esimd::simd<float, 16> v306_data;
              v306_data.copy_from(s0 + (112_i32));
              v302_acc += ((v306_data[0]) * v27_data);
              v302_acc += ((v306_data[1]) * v28_data);
              v302_acc += ((v306_data[2]) * v29_data);
              v302_acc += ((v306_data[3]) * v30_data);
              v302_acc += ((v306_data[4]) * v31_data);
              v302_acc += ((v306_data[5]) * v32_data);
              v302_acc += ((v306_data[6]) * v33_data);
              v302_acc += ((v306_data[7]) * v34_data);
              v302_acc += ((v306_data[8]) * v35_data);
              v302_acc += ((v306_data[9]) * v36_data);
              v302_acc += ((v306_data[10]) * v37_data);
              v302_acc += ((v306_data[11]) * v38_data);
              v302_acc += ((v306_data[12]) * v39_data);
              v302_acc += ((v306_data[13]) * v40_data);
              v302_acc += ((v306_data[14]) * v41_data);
              v302_acc += ((v306_data[15]) * v42_data);
              v302_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v339_n0 = 0; v339_n0 < 1; ++v339_n0) {
                int32_t v341_a = v339_n0 * 16;
                #pragma unroll
                for (int32_t v340_n1 = 0; v340_n1 < 8; ++v340_n1) {
                  int32_t v343_a = v341_a + (v340_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v344_data;
                  v344_data.copy_from(ir1 + (v343_a));
                  v344_data.copy_to(r1 + (v343_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v348_i0 = 0; v348_i0 < 1; ++v348_i0) {
                int32_t v350_a = v348_i0 * 16;
                #pragma unroll
                for (int32_t v349_i1 = 0; v349_i1 < 8; ++v349_i1) {
                  int32_t v352_a = v350_a + (v349_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v353_data;
                  v353_data.copy_from(r1 + (v352_a));
                  v353_data.copy_to(glb_m0 + (v352_a));
                }
              }
            }
          }
        }
      });
    }
  });
}


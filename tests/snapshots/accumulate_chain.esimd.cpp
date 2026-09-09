// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8a03a3cd0d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8a03a3cd0d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×12(12×12) {0..12}×{0..12} strided
        // m2 12×8(12×8) {0..12}×{0..8} strided
        // m3 12×12(12×12) {0..12}×{0..12} strided
        // m4 12×8(12×8) {0..12}×{0..8} strided
        // m5 12×12(12×12) {0..12}×{0..12} strided
        // m6 12×8(12×8) {0..12}×{0..8} strided
        // m7 12×12(12×12) {0..12}×{0..12} strided
        // m8 12×8(12×8) {0..12}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
              float r0[192]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v12_i1 = 0; v12_i1 < 12; ++v12_i1) {
                tensorforge::intel_esimd::simd<float, 12> v17_data;
                v17_data.copy_from(glb_m1 + ((v12_i1 * 12)));
                v17_data.copy_to(r0 + ((v12_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v21_ld;
              v21_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v21_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v22_ld;
              v22_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v22_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              float r2[192]{};
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                tensorforge::intel_esimd::simd<float, 12> v29_data;
                v29_data.copy_from(glb_m3 + ((v24_i1 * 12)));
                v29_data.copy_to(r2 + ((v24_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v46_acc{};
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(s0 + (0_i32));
              v46_acc += ((v50_data[0]) * v34_data);
              v46_acc += ((v50_data[1]) * v35_data);
              v46_acc += ((v50_data[2]) * v36_data);
              v46_acc += ((v50_data[3]) * v37_data);
              v46_acc += ((v50_data[4]) * v38_data);
              v46_acc += ((v50_data[5]) * v39_data);
              v46_acc += ((v50_data[6]) * v40_data);
              v46_acc += ((v50_data[7]) * v41_data);
              v46_acc += ((v50_data[8]) * v42_data);
              v46_acc += ((v50_data[9]) * v43_data);
              v46_acc += ((v50_data[10]) * v44_data);
              v46_acc += ((v50_data[11]) * v45_data);
              v46_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v75_acc{};
              tensorforge::intel_esimd::simd<float, 16> v79_data;
              v79_data.copy_from(s0 + (12_i32));
              v75_acc += ((v79_data[0]) * v34_data);
              v75_acc += ((v79_data[1]) * v35_data);
              v75_acc += ((v79_data[2]) * v36_data);
              v75_acc += ((v79_data[3]) * v37_data);
              v75_acc += ((v79_data[4]) * v38_data);
              v75_acc += ((v79_data[5]) * v39_data);
              v75_acc += ((v79_data[6]) * v40_data);
              v75_acc += ((v79_data[7]) * v41_data);
              v75_acc += ((v79_data[8]) * v42_data);
              v75_acc += ((v79_data[9]) * v43_data);
              v75_acc += ((v79_data[10]) * v44_data);
              v75_acc += ((v79_data[11]) * v45_data);
              v75_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v104_acc{};
              tensorforge::intel_esimd::simd<float, 16> v108_data;
              v108_data.copy_from(s0 + (24_i32));
              v104_acc += ((v108_data[0]) * v34_data);
              v104_acc += ((v108_data[1]) * v35_data);
              v104_acc += ((v108_data[2]) * v36_data);
              v104_acc += ((v108_data[3]) * v37_data);
              v104_acc += ((v108_data[4]) * v38_data);
              v104_acc += ((v108_data[5]) * v39_data);
              v104_acc += ((v108_data[6]) * v40_data);
              v104_acc += ((v108_data[7]) * v41_data);
              v104_acc += ((v108_data[8]) * v42_data);
              v104_acc += ((v108_data[9]) * v43_data);
              v104_acc += ((v108_data[10]) * v44_data);
              v104_acc += ((v108_data[11]) * v45_data);
              v104_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v133_acc{};
              tensorforge::intel_esimd::simd<float, 16> v137_data;
              v137_data.copy_from(s0 + (36_i32));
              v133_acc += ((v137_data[0]) * v34_data);
              v133_acc += ((v137_data[1]) * v35_data);
              v133_acc += ((v137_data[2]) * v36_data);
              v133_acc += ((v137_data[3]) * v37_data);
              v133_acc += ((v137_data[4]) * v38_data);
              v133_acc += ((v137_data[5]) * v39_data);
              v133_acc += ((v137_data[6]) * v40_data);
              v133_acc += ((v137_data[7]) * v41_data);
              v133_acc += ((v137_data[8]) * v42_data);
              v133_acc += ((v137_data[9]) * v43_data);
              v133_acc += ((v137_data[10]) * v44_data);
              v133_acc += ((v137_data[11]) * v45_data);
              v133_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v162_acc{};
              tensorforge::intel_esimd::simd<float, 16> v166_data;
              v166_data.copy_from(s0 + (48_i32));
              v162_acc += ((v166_data[0]) * v34_data);
              v162_acc += ((v166_data[1]) * v35_data);
              v162_acc += ((v166_data[2]) * v36_data);
              v162_acc += ((v166_data[3]) * v37_data);
              v162_acc += ((v166_data[4]) * v38_data);
              v162_acc += ((v166_data[5]) * v39_data);
              v162_acc += ((v166_data[6]) * v40_data);
              v162_acc += ((v166_data[7]) * v41_data);
              v162_acc += ((v166_data[8]) * v42_data);
              v162_acc += ((v166_data[9]) * v43_data);
              v162_acc += ((v166_data[10]) * v44_data);
              v162_acc += ((v166_data[11]) * v45_data);
              v162_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v191_acc{};
              tensorforge::intel_esimd::simd<float, 16> v195_data;
              v195_data.copy_from(s0 + (60_i32));
              v191_acc += ((v195_data[0]) * v34_data);
              v191_acc += ((v195_data[1]) * v35_data);
              v191_acc += ((v195_data[2]) * v36_data);
              v191_acc += ((v195_data[3]) * v37_data);
              v191_acc += ((v195_data[4]) * v38_data);
              v191_acc += ((v195_data[5]) * v39_data);
              v191_acc += ((v195_data[6]) * v40_data);
              v191_acc += ((v195_data[7]) * v41_data);
              v191_acc += ((v195_data[8]) * v42_data);
              v191_acc += ((v195_data[9]) * v43_data);
              v191_acc += ((v195_data[10]) * v44_data);
              v191_acc += ((v195_data[11]) * v45_data);
              v191_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v220_acc{};
              tensorforge::intel_esimd::simd<float, 16> v224_data;
              v224_data.copy_from(s0 + (72_i32));
              v220_acc += ((v224_data[0]) * v34_data);
              v220_acc += ((v224_data[1]) * v35_data);
              v220_acc += ((v224_data[2]) * v36_data);
              v220_acc += ((v224_data[3]) * v37_data);
              v220_acc += ((v224_data[4]) * v38_data);
              v220_acc += ((v224_data[5]) * v39_data);
              v220_acc += ((v224_data[6]) * v40_data);
              v220_acc += ((v224_data[7]) * v41_data);
              v220_acc += ((v224_data[8]) * v42_data);
              v220_acc += ((v224_data[9]) * v43_data);
              v220_acc += ((v224_data[10]) * v44_data);
              v220_acc += ((v224_data[11]) * v45_data);
              v220_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v249_acc{};
              tensorforge::intel_esimd::simd<float, 16> v253_data;
              v253_data.copy_from(s0 + (84_i32));
              v249_acc += ((v253_data[0]) * v34_data);
              v249_acc += ((v253_data[1]) * v35_data);
              v249_acc += ((v253_data[2]) * v36_data);
              v249_acc += ((v253_data[3]) * v37_data);
              v249_acc += ((v253_data[4]) * v38_data);
              v249_acc += ((v253_data[5]) * v39_data);
              v249_acc += ((v253_data[6]) * v40_data);
              v249_acc += ((v253_data[7]) * v41_data);
              v249_acc += ((v253_data[8]) * v42_data);
              v249_acc += ((v253_data[9]) * v43_data);
              v249_acc += ((v253_data[10]) * v44_data);
              v249_acc += ((v253_data[11]) * v45_data);
              v249_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v278_n1 = 0; v278_n1 < 8; ++v278_n1) {
                int32_t v279_a = v278_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v281_data;
                v281_data.copy_from(ir1 + (v279_a));
                v281_data.copy_to(r1 + (v279_a));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v285_ld;
              v285_ld.copy_from(glb_m4 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v285_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v286_ld;
              v286_ld.copy_from(glb_m4 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v286_ld.copy_to(s1 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r2 = load{g>r}(glb_m3););
              float r4[192]{};
              // r4 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v288_i1 = 0; v288_i1 < 12; ++v288_i1) {
                tensorforge::intel_esimd::simd<float, 12> v293_data;
                v293_data.copy_from(glb_m5 + ((v288_i1 * 12)));
                v293_data.copy_to(r4 + ((v288_i1 * 16)));
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r3[128]{};
              // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir3[128]{};
              tensorforge::intel_esimd::simd<float, 16> v298_data;
              v298_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v299_data;
              v299_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v300_data;
              v300_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v301_data;
              v301_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v302_data;
              v302_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v303_data;
              v303_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v304_data;
              v304_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v305_data;
              v305_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v306_data;
              v306_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v307_data;
              v307_data.copy_from(r2 + (144));
              tensorforge::intel_esimd::simd<float, 16> v308_data;
              v308_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v309_data;
              v309_data.copy_from(r2 + (176));
              tensorforge::intel_esimd::simd<float, 16> v310_acc{};
              tensorforge::intel_esimd::simd<float, 16> v314_data;
              v314_data.copy_from(s1 + (0_i32));
              v310_acc += ((v314_data[0]) * v298_data);
              v310_acc += ((v314_data[1]) * v299_data);
              v310_acc += ((v314_data[2]) * v300_data);
              v310_acc += ((v314_data[3]) * v301_data);
              v310_acc += ((v314_data[4]) * v302_data);
              v310_acc += ((v314_data[5]) * v303_data);
              v310_acc += ((v314_data[6]) * v304_data);
              v310_acc += ((v314_data[7]) * v305_data);
              v310_acc += ((v314_data[8]) * v306_data);
              v310_acc += ((v314_data[9]) * v307_data);
              v310_acc += ((v314_data[10]) * v308_data);
              v310_acc += ((v314_data[11]) * v309_data);
              v310_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v339_acc{};
              tensorforge::intel_esimd::simd<float, 16> v343_data;
              v343_data.copy_from(s1 + (12_i32));
              v339_acc += ((v343_data[0]) * v298_data);
              v339_acc += ((v343_data[1]) * v299_data);
              v339_acc += ((v343_data[2]) * v300_data);
              v339_acc += ((v343_data[3]) * v301_data);
              v339_acc += ((v343_data[4]) * v302_data);
              v339_acc += ((v343_data[5]) * v303_data);
              v339_acc += ((v343_data[6]) * v304_data);
              v339_acc += ((v343_data[7]) * v305_data);
              v339_acc += ((v343_data[8]) * v306_data);
              v339_acc += ((v343_data[9]) * v307_data);
              v339_acc += ((v343_data[10]) * v308_data);
              v339_acc += ((v343_data[11]) * v309_data);
              v339_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v368_acc{};
              tensorforge::intel_esimd::simd<float, 16> v372_data;
              v372_data.copy_from(s1 + (24_i32));
              v368_acc += ((v372_data[0]) * v298_data);
              v368_acc += ((v372_data[1]) * v299_data);
              v368_acc += ((v372_data[2]) * v300_data);
              v368_acc += ((v372_data[3]) * v301_data);
              v368_acc += ((v372_data[4]) * v302_data);
              v368_acc += ((v372_data[5]) * v303_data);
              v368_acc += ((v372_data[6]) * v304_data);
              v368_acc += ((v372_data[7]) * v305_data);
              v368_acc += ((v372_data[8]) * v306_data);
              v368_acc += ((v372_data[9]) * v307_data);
              v368_acc += ((v372_data[10]) * v308_data);
              v368_acc += ((v372_data[11]) * v309_data);
              v368_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v397_acc{};
              tensorforge::intel_esimd::simd<float, 16> v401_data;
              v401_data.copy_from(s1 + (36_i32));
              v397_acc += ((v401_data[0]) * v298_data);
              v397_acc += ((v401_data[1]) * v299_data);
              v397_acc += ((v401_data[2]) * v300_data);
              v397_acc += ((v401_data[3]) * v301_data);
              v397_acc += ((v401_data[4]) * v302_data);
              v397_acc += ((v401_data[5]) * v303_data);
              v397_acc += ((v401_data[6]) * v304_data);
              v397_acc += ((v401_data[7]) * v305_data);
              v397_acc += ((v401_data[8]) * v306_data);
              v397_acc += ((v401_data[9]) * v307_data);
              v397_acc += ((v401_data[10]) * v308_data);
              v397_acc += ((v401_data[11]) * v309_data);
              v397_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v426_acc{};
              tensorforge::intel_esimd::simd<float, 16> v430_data;
              v430_data.copy_from(s1 + (48_i32));
              v426_acc += ((v430_data[0]) * v298_data);
              v426_acc += ((v430_data[1]) * v299_data);
              v426_acc += ((v430_data[2]) * v300_data);
              v426_acc += ((v430_data[3]) * v301_data);
              v426_acc += ((v430_data[4]) * v302_data);
              v426_acc += ((v430_data[5]) * v303_data);
              v426_acc += ((v430_data[6]) * v304_data);
              v426_acc += ((v430_data[7]) * v305_data);
              v426_acc += ((v430_data[8]) * v306_data);
              v426_acc += ((v430_data[9]) * v307_data);
              v426_acc += ((v430_data[10]) * v308_data);
              v426_acc += ((v430_data[11]) * v309_data);
              v426_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v455_acc{};
              tensorforge::intel_esimd::simd<float, 16> v459_data;
              v459_data.copy_from(s1 + (60_i32));
              v455_acc += ((v459_data[0]) * v298_data);
              v455_acc += ((v459_data[1]) * v299_data);
              v455_acc += ((v459_data[2]) * v300_data);
              v455_acc += ((v459_data[3]) * v301_data);
              v455_acc += ((v459_data[4]) * v302_data);
              v455_acc += ((v459_data[5]) * v303_data);
              v455_acc += ((v459_data[6]) * v304_data);
              v455_acc += ((v459_data[7]) * v305_data);
              v455_acc += ((v459_data[8]) * v306_data);
              v455_acc += ((v459_data[9]) * v307_data);
              v455_acc += ((v459_data[10]) * v308_data);
              v455_acc += ((v459_data[11]) * v309_data);
              v455_acc.copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 16> v484_acc{};
              tensorforge::intel_esimd::simd<float, 16> v488_data;
              v488_data.copy_from(s1 + (72_i32));
              v484_acc += ((v488_data[0]) * v298_data);
              v484_acc += ((v488_data[1]) * v299_data);
              v484_acc += ((v488_data[2]) * v300_data);
              v484_acc += ((v488_data[3]) * v301_data);
              v484_acc += ((v488_data[4]) * v302_data);
              v484_acc += ((v488_data[5]) * v303_data);
              v484_acc += ((v488_data[6]) * v304_data);
              v484_acc += ((v488_data[7]) * v305_data);
              v484_acc += ((v488_data[8]) * v306_data);
              v484_acc += ((v488_data[9]) * v307_data);
              v484_acc += ((v488_data[10]) * v308_data);
              v484_acc += ((v488_data[11]) * v309_data);
              v484_acc.copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v513_acc{};
              tensorforge::intel_esimd::simd<float, 16> v517_data;
              v517_data.copy_from(s1 + (84_i32));
              v513_acc += ((v517_data[0]) * v298_data);
              v513_acc += ((v517_data[1]) * v299_data);
              v513_acc += ((v517_data[2]) * v300_data);
              v513_acc += ((v517_data[3]) * v301_data);
              v513_acc += ((v517_data[4]) * v302_data);
              v513_acc += ((v517_data[5]) * v303_data);
              v513_acc += ((v517_data[6]) * v304_data);
              v513_acc += ((v517_data[7]) * v305_data);
              v513_acc += ((v517_data[8]) * v306_data);
              v513_acc += ((v517_data[9]) * v307_data);
              v513_acc += ((v517_data[10]) * v308_data);
              v513_acc += ((v517_data[11]) * v309_data);
              v513_acc.copy_to(ir3 + (112));
              #pragma unroll
              for (int32_t v542_n1 = 0; v542_n1 < 8; ++v542_n1) {
                int32_t v543_a = v542_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v545_data;
                v545_data.copy_from(ir3 + (v543_a));
                tensorforge::intel_esimd::simd<float, 12> v548_data;
                v548_data.copy_from(r1 + (v543_a));
                (v548_data + v545_data).copy_to(r3 + (v543_a));
              }
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m6[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v553_ld;
              v553_ld.copy_from(glb_m6 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v553_ld.copy_to(s2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v554_ld;
              v554_ld.copy_from(glb_m6 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v554_ld.copy_to(s2 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r4 = load{g>r}(glb_m5););
              float r6[192]{};
              // r6 = load{g>r}(glb_m7);
              #pragma unroll
              for (int32_t v556_i1 = 0; v556_i1 < 12; ++v556_i1) {
                tensorforge::intel_esimd::simd<float, 12> v561_data;
                v561_data.copy_from(glb_m7 + ((v556_i1 * 12)));
                v561_data.copy_to(r6 + ((v556_i1 * 16)));
              }
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              float r5[128]{};
              // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[128]{};
              tensorforge::intel_esimd::simd<float, 16> v566_data;
              v566_data.copy_from(r4 + (0));
              tensorforge::intel_esimd::simd<float, 16> v567_data;
              v567_data.copy_from(r4 + (16));
              tensorforge::intel_esimd::simd<float, 16> v568_data;
              v568_data.copy_from(r4 + (32));
              tensorforge::intel_esimd::simd<float, 16> v569_data;
              v569_data.copy_from(r4 + (48));
              tensorforge::intel_esimd::simd<float, 16> v570_data;
              v570_data.copy_from(r4 + (64));
              tensorforge::intel_esimd::simd<float, 16> v571_data;
              v571_data.copy_from(r4 + (80));
              tensorforge::intel_esimd::simd<float, 16> v572_data;
              v572_data.copy_from(r4 + (96));
              tensorforge::intel_esimd::simd<float, 16> v573_data;
              v573_data.copy_from(r4 + (112));
              tensorforge::intel_esimd::simd<float, 16> v574_data;
              v574_data.copy_from(r4 + (128));
              tensorforge::intel_esimd::simd<float, 16> v575_data;
              v575_data.copy_from(r4 + (144));
              tensorforge::intel_esimd::simd<float, 16> v576_data;
              v576_data.copy_from(r4 + (160));
              tensorforge::intel_esimd::simd<float, 16> v577_data;
              v577_data.copy_from(r4 + (176));
              tensorforge::intel_esimd::simd<float, 16> v578_acc{};
              tensorforge::intel_esimd::simd<float, 16> v582_data;
              v582_data.copy_from(s2 + (0_i32));
              v578_acc += ((v582_data[0]) * v566_data);
              v578_acc += ((v582_data[1]) * v567_data);
              v578_acc += ((v582_data[2]) * v568_data);
              v578_acc += ((v582_data[3]) * v569_data);
              v578_acc += ((v582_data[4]) * v570_data);
              v578_acc += ((v582_data[5]) * v571_data);
              v578_acc += ((v582_data[6]) * v572_data);
              v578_acc += ((v582_data[7]) * v573_data);
              v578_acc += ((v582_data[8]) * v574_data);
              v578_acc += ((v582_data[9]) * v575_data);
              v578_acc += ((v582_data[10]) * v576_data);
              v578_acc += ((v582_data[11]) * v577_data);
              v578_acc.copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 16> v607_acc{};
              tensorforge::intel_esimd::simd<float, 16> v611_data;
              v611_data.copy_from(s2 + (12_i32));
              v607_acc += ((v611_data[0]) * v566_data);
              v607_acc += ((v611_data[1]) * v567_data);
              v607_acc += ((v611_data[2]) * v568_data);
              v607_acc += ((v611_data[3]) * v569_data);
              v607_acc += ((v611_data[4]) * v570_data);
              v607_acc += ((v611_data[5]) * v571_data);
              v607_acc += ((v611_data[6]) * v572_data);
              v607_acc += ((v611_data[7]) * v573_data);
              v607_acc += ((v611_data[8]) * v574_data);
              v607_acc += ((v611_data[9]) * v575_data);
              v607_acc += ((v611_data[10]) * v576_data);
              v607_acc += ((v611_data[11]) * v577_data);
              v607_acc.copy_to(ir5 + (16));
              tensorforge::intel_esimd::simd<float, 16> v636_acc{};
              tensorforge::intel_esimd::simd<float, 16> v640_data;
              v640_data.copy_from(s2 + (24_i32));
              v636_acc += ((v640_data[0]) * v566_data);
              v636_acc += ((v640_data[1]) * v567_data);
              v636_acc += ((v640_data[2]) * v568_data);
              v636_acc += ((v640_data[3]) * v569_data);
              v636_acc += ((v640_data[4]) * v570_data);
              v636_acc += ((v640_data[5]) * v571_data);
              v636_acc += ((v640_data[6]) * v572_data);
              v636_acc += ((v640_data[7]) * v573_data);
              v636_acc += ((v640_data[8]) * v574_data);
              v636_acc += ((v640_data[9]) * v575_data);
              v636_acc += ((v640_data[10]) * v576_data);
              v636_acc += ((v640_data[11]) * v577_data);
              v636_acc.copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 16> v665_acc{};
              tensorforge::intel_esimd::simd<float, 16> v669_data;
              v669_data.copy_from(s2 + (36_i32));
              v665_acc += ((v669_data[0]) * v566_data);
              v665_acc += ((v669_data[1]) * v567_data);
              v665_acc += ((v669_data[2]) * v568_data);
              v665_acc += ((v669_data[3]) * v569_data);
              v665_acc += ((v669_data[4]) * v570_data);
              v665_acc += ((v669_data[5]) * v571_data);
              v665_acc += ((v669_data[6]) * v572_data);
              v665_acc += ((v669_data[7]) * v573_data);
              v665_acc += ((v669_data[8]) * v574_data);
              v665_acc += ((v669_data[9]) * v575_data);
              v665_acc += ((v669_data[10]) * v576_data);
              v665_acc += ((v669_data[11]) * v577_data);
              v665_acc.copy_to(ir5 + (48));
              tensorforge::intel_esimd::simd<float, 16> v694_acc{};
              tensorforge::intel_esimd::simd<float, 16> v698_data;
              v698_data.copy_from(s2 + (48_i32));
              v694_acc += ((v698_data[0]) * v566_data);
              v694_acc += ((v698_data[1]) * v567_data);
              v694_acc += ((v698_data[2]) * v568_data);
              v694_acc += ((v698_data[3]) * v569_data);
              v694_acc += ((v698_data[4]) * v570_data);
              v694_acc += ((v698_data[5]) * v571_data);
              v694_acc += ((v698_data[6]) * v572_data);
              v694_acc += ((v698_data[7]) * v573_data);
              v694_acc += ((v698_data[8]) * v574_data);
              v694_acc += ((v698_data[9]) * v575_data);
              v694_acc += ((v698_data[10]) * v576_data);
              v694_acc += ((v698_data[11]) * v577_data);
              v694_acc.copy_to(ir5 + (64));
              tensorforge::intel_esimd::simd<float, 16> v723_acc{};
              tensorforge::intel_esimd::simd<float, 16> v727_data;
              v727_data.copy_from(s2 + (60_i32));
              v723_acc += ((v727_data[0]) * v566_data);
              v723_acc += ((v727_data[1]) * v567_data);
              v723_acc += ((v727_data[2]) * v568_data);
              v723_acc += ((v727_data[3]) * v569_data);
              v723_acc += ((v727_data[4]) * v570_data);
              v723_acc += ((v727_data[5]) * v571_data);
              v723_acc += ((v727_data[6]) * v572_data);
              v723_acc += ((v727_data[7]) * v573_data);
              v723_acc += ((v727_data[8]) * v574_data);
              v723_acc += ((v727_data[9]) * v575_data);
              v723_acc += ((v727_data[10]) * v576_data);
              v723_acc += ((v727_data[11]) * v577_data);
              v723_acc.copy_to(ir5 + (80));
              tensorforge::intel_esimd::simd<float, 16> v752_acc{};
              tensorforge::intel_esimd::simd<float, 16> v756_data;
              v756_data.copy_from(s2 + (72_i32));
              v752_acc += ((v756_data[0]) * v566_data);
              v752_acc += ((v756_data[1]) * v567_data);
              v752_acc += ((v756_data[2]) * v568_data);
              v752_acc += ((v756_data[3]) * v569_data);
              v752_acc += ((v756_data[4]) * v570_data);
              v752_acc += ((v756_data[5]) * v571_data);
              v752_acc += ((v756_data[6]) * v572_data);
              v752_acc += ((v756_data[7]) * v573_data);
              v752_acc += ((v756_data[8]) * v574_data);
              v752_acc += ((v756_data[9]) * v575_data);
              v752_acc += ((v756_data[10]) * v576_data);
              v752_acc += ((v756_data[11]) * v577_data);
              v752_acc.copy_to(ir5 + (96));
              tensorforge::intel_esimd::simd<float, 16> v781_acc{};
              tensorforge::intel_esimd::simd<float, 16> v785_data;
              v785_data.copy_from(s2 + (84_i32));
              v781_acc += ((v785_data[0]) * v566_data);
              v781_acc += ((v785_data[1]) * v567_data);
              v781_acc += ((v785_data[2]) * v568_data);
              v781_acc += ((v785_data[3]) * v569_data);
              v781_acc += ((v785_data[4]) * v570_data);
              v781_acc += ((v785_data[5]) * v571_data);
              v781_acc += ((v785_data[6]) * v572_data);
              v781_acc += ((v785_data[7]) * v573_data);
              v781_acc += ((v785_data[8]) * v574_data);
              v781_acc += ((v785_data[9]) * v575_data);
              v781_acc += ((v785_data[10]) * v576_data);
              v781_acc += ((v785_data[11]) * v577_data);
              v781_acc.copy_to(ir5 + (112));
              #pragma unroll
              for (int32_t v810_n1 = 0; v810_n1 < 8; ++v810_n1) {
                int32_t v811_a = v810_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v813_data;
                v813_data.copy_from(ir5 + (v811_a));
                tensorforge::intel_esimd::simd<float, 12> v816_data;
                v816_data.copy_from(r3 + (v811_a));
                (v816_data + v813_data).copy_to(r5 + (v811_a));
              }
              float* __restrict__ s3 = &localShrMem0[0];
              // s3 = load{g>s}(glb_m8[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v821_ld;
              v821_ld.copy_from(glb_m8 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v821_ld.copy_to(s3 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v822_ld;
              v822_ld.copy_from(glb_m8 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v822_ld.copy_to(s3 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r6 = load{g>r}(glb_m7););
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              float r7[128]{};
              // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir7[128]{};
              tensorforge::intel_esimd::simd<float, 16> v825_data;
              v825_data.copy_from(r6 + (0));
              tensorforge::intel_esimd::simd<float, 16> v826_data;
              v826_data.copy_from(r6 + (16));
              tensorforge::intel_esimd::simd<float, 16> v827_data;
              v827_data.copy_from(r6 + (32));
              tensorforge::intel_esimd::simd<float, 16> v828_data;
              v828_data.copy_from(r6 + (48));
              tensorforge::intel_esimd::simd<float, 16> v829_data;
              v829_data.copy_from(r6 + (64));
              tensorforge::intel_esimd::simd<float, 16> v830_data;
              v830_data.copy_from(r6 + (80));
              tensorforge::intel_esimd::simd<float, 16> v831_data;
              v831_data.copy_from(r6 + (96));
              tensorforge::intel_esimd::simd<float, 16> v832_data;
              v832_data.copy_from(r6 + (112));
              tensorforge::intel_esimd::simd<float, 16> v833_data;
              v833_data.copy_from(r6 + (128));
              tensorforge::intel_esimd::simd<float, 16> v834_data;
              v834_data.copy_from(r6 + (144));
              tensorforge::intel_esimd::simd<float, 16> v835_data;
              v835_data.copy_from(r6 + (160));
              tensorforge::intel_esimd::simd<float, 16> v836_data;
              v836_data.copy_from(r6 + (176));
              tensorforge::intel_esimd::simd<float, 16> v837_acc{};
              tensorforge::intel_esimd::simd<float, 16> v841_data;
              v841_data.copy_from(s3 + (0_i32));
              v837_acc += ((v841_data[0]) * v825_data);
              v837_acc += ((v841_data[1]) * v826_data);
              v837_acc += ((v841_data[2]) * v827_data);
              v837_acc += ((v841_data[3]) * v828_data);
              v837_acc += ((v841_data[4]) * v829_data);
              v837_acc += ((v841_data[5]) * v830_data);
              v837_acc += ((v841_data[6]) * v831_data);
              v837_acc += ((v841_data[7]) * v832_data);
              v837_acc += ((v841_data[8]) * v833_data);
              v837_acc += ((v841_data[9]) * v834_data);
              v837_acc += ((v841_data[10]) * v835_data);
              v837_acc += ((v841_data[11]) * v836_data);
              v837_acc.copy_to(ir7 + (0));
              tensorforge::intel_esimd::simd<float, 16> v866_acc{};
              tensorforge::intel_esimd::simd<float, 16> v870_data;
              v870_data.copy_from(s3 + (12_i32));
              v866_acc += ((v870_data[0]) * v825_data);
              v866_acc += ((v870_data[1]) * v826_data);
              v866_acc += ((v870_data[2]) * v827_data);
              v866_acc += ((v870_data[3]) * v828_data);
              v866_acc += ((v870_data[4]) * v829_data);
              v866_acc += ((v870_data[5]) * v830_data);
              v866_acc += ((v870_data[6]) * v831_data);
              v866_acc += ((v870_data[7]) * v832_data);
              v866_acc += ((v870_data[8]) * v833_data);
              v866_acc += ((v870_data[9]) * v834_data);
              v866_acc += ((v870_data[10]) * v835_data);
              v866_acc += ((v870_data[11]) * v836_data);
              v866_acc.copy_to(ir7 + (16));
              tensorforge::intel_esimd::simd<float, 16> v895_acc{};
              tensorforge::intel_esimd::simd<float, 16> v899_data;
              v899_data.copy_from(s3 + (24_i32));
              v895_acc += ((v899_data[0]) * v825_data);
              v895_acc += ((v899_data[1]) * v826_data);
              v895_acc += ((v899_data[2]) * v827_data);
              v895_acc += ((v899_data[3]) * v828_data);
              v895_acc += ((v899_data[4]) * v829_data);
              v895_acc += ((v899_data[5]) * v830_data);
              v895_acc += ((v899_data[6]) * v831_data);
              v895_acc += ((v899_data[7]) * v832_data);
              v895_acc += ((v899_data[8]) * v833_data);
              v895_acc += ((v899_data[9]) * v834_data);
              v895_acc += ((v899_data[10]) * v835_data);
              v895_acc += ((v899_data[11]) * v836_data);
              v895_acc.copy_to(ir7 + (32));
              tensorforge::intel_esimd::simd<float, 16> v924_acc{};
              tensorforge::intel_esimd::simd<float, 16> v928_data;
              v928_data.copy_from(s3 + (36_i32));
              v924_acc += ((v928_data[0]) * v825_data);
              v924_acc += ((v928_data[1]) * v826_data);
              v924_acc += ((v928_data[2]) * v827_data);
              v924_acc += ((v928_data[3]) * v828_data);
              v924_acc += ((v928_data[4]) * v829_data);
              v924_acc += ((v928_data[5]) * v830_data);
              v924_acc += ((v928_data[6]) * v831_data);
              v924_acc += ((v928_data[7]) * v832_data);
              v924_acc += ((v928_data[8]) * v833_data);
              v924_acc += ((v928_data[9]) * v834_data);
              v924_acc += ((v928_data[10]) * v835_data);
              v924_acc += ((v928_data[11]) * v836_data);
              v924_acc.copy_to(ir7 + (48));
              tensorforge::intel_esimd::simd<float, 16> v953_acc{};
              tensorforge::intel_esimd::simd<float, 16> v957_data;
              v957_data.copy_from(s3 + (48_i32));
              v953_acc += ((v957_data[0]) * v825_data);
              v953_acc += ((v957_data[1]) * v826_data);
              v953_acc += ((v957_data[2]) * v827_data);
              v953_acc += ((v957_data[3]) * v828_data);
              v953_acc += ((v957_data[4]) * v829_data);
              v953_acc += ((v957_data[5]) * v830_data);
              v953_acc += ((v957_data[6]) * v831_data);
              v953_acc += ((v957_data[7]) * v832_data);
              v953_acc += ((v957_data[8]) * v833_data);
              v953_acc += ((v957_data[9]) * v834_data);
              v953_acc += ((v957_data[10]) * v835_data);
              v953_acc += ((v957_data[11]) * v836_data);
              v953_acc.copy_to(ir7 + (64));
              tensorforge::intel_esimd::simd<float, 16> v982_acc{};
              tensorforge::intel_esimd::simd<float, 16> v986_data;
              v986_data.copy_from(s3 + (60_i32));
              v982_acc += ((v986_data[0]) * v825_data);
              v982_acc += ((v986_data[1]) * v826_data);
              v982_acc += ((v986_data[2]) * v827_data);
              v982_acc += ((v986_data[3]) * v828_data);
              v982_acc += ((v986_data[4]) * v829_data);
              v982_acc += ((v986_data[5]) * v830_data);
              v982_acc += ((v986_data[6]) * v831_data);
              v982_acc += ((v986_data[7]) * v832_data);
              v982_acc += ((v986_data[8]) * v833_data);
              v982_acc += ((v986_data[9]) * v834_data);
              v982_acc += ((v986_data[10]) * v835_data);
              v982_acc += ((v986_data[11]) * v836_data);
              v982_acc.copy_to(ir7 + (80));
              tensorforge::intel_esimd::simd<float, 16> v1011_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1015_data;
              v1015_data.copy_from(s3 + (72_i32));
              v1011_acc += ((v1015_data[0]) * v825_data);
              v1011_acc += ((v1015_data[1]) * v826_data);
              v1011_acc += ((v1015_data[2]) * v827_data);
              v1011_acc += ((v1015_data[3]) * v828_data);
              v1011_acc += ((v1015_data[4]) * v829_data);
              v1011_acc += ((v1015_data[5]) * v830_data);
              v1011_acc += ((v1015_data[6]) * v831_data);
              v1011_acc += ((v1015_data[7]) * v832_data);
              v1011_acc += ((v1015_data[8]) * v833_data);
              v1011_acc += ((v1015_data[9]) * v834_data);
              v1011_acc += ((v1015_data[10]) * v835_data);
              v1011_acc += ((v1015_data[11]) * v836_data);
              v1011_acc.copy_to(ir7 + (96));
              tensorforge::intel_esimd::simd<float, 16> v1040_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1044_data;
              v1044_data.copy_from(s3 + (84_i32));
              v1040_acc += ((v1044_data[0]) * v825_data);
              v1040_acc += ((v1044_data[1]) * v826_data);
              v1040_acc += ((v1044_data[2]) * v827_data);
              v1040_acc += ((v1044_data[3]) * v828_data);
              v1040_acc += ((v1044_data[4]) * v829_data);
              v1040_acc += ((v1044_data[5]) * v830_data);
              v1040_acc += ((v1044_data[6]) * v831_data);
              v1040_acc += ((v1044_data[7]) * v832_data);
              v1040_acc += ((v1044_data[8]) * v833_data);
              v1040_acc += ((v1044_data[9]) * v834_data);
              v1040_acc += ((v1044_data[10]) * v835_data);
              v1040_acc += ((v1044_data[11]) * v836_data);
              v1040_acc.copy_to(ir7 + (112));
              #pragma unroll
              for (int32_t v1069_n1 = 0; v1069_n1 < 8; ++v1069_n1) {
                int32_t v1070_a = v1069_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1072_data;
                v1072_data.copy_from(ir7 + (v1070_a));
                tensorforge::intel_esimd::simd<float, 12> v1075_data;
                v1075_data.copy_from(r5 + (v1070_a));
                (v1075_data + v1072_data).copy_to(r7 + (v1070_a));
              }
              // glb_m0 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1079_i1 = 0; v1079_i1 < 8; ++v1079_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1082_data;
                v1082_data.copy_from(r7 + ((v1079_i1 * 16)));
                v1082_data.copy_to(glb_m0 + ((v1079_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}


// === base name ===
kernel_f94e030d8c

// === header ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f94e030d8c(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f94e030d8c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (9472, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 20×12(20×12) {0..20}×{0..12} strided
        // m2 20×16(20×16) {0..20}×{0..16} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 20×12(20×12) {0..20}×{0..12} strided({0..20}×{0..12})[-1, 0]×m2 20×16(20×16) {0..20}×{0..16} strided({0..20}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[592 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[576];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[320];
              // s0 = load{g>s}(glb_m1[1, 0])
              #pragma unroll
              for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
                int32_t v8_lead = v6_i0 * 16;
                #pragma unroll
                for (int32_t v7_i1 = 0; v7_i1 < 12; ++v7_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v12_data;
                  v12_data.copy_from(glb_m1 + ((v8_lead + (v7_i1 * 20))));
                  int32_t v16_a = v8_lead + (v7_i1 * 21);
                  v12_data.copy_to(s0 + ((v16_a ^ ((v16_a >> 2) & 3))));
                }
              }
              #pragma unroll
              for (int32_t v20_i1 = 0; v20_i1 < 12; ++v20_i1) {
                tensorforge::intel_esimd::simd<float, 4> v26_data;
                v26_data.copy_from(glb_m1 + ((16_i32 + (v20_i1 * 20))));
                int32_t v31_a = 16_i32 + (v20_i1 * 21);
                v26_data.copy_to(s0 + ((v31_a ^ ((v31_a >> 2) & 3))));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                *(sycl::vec<float, 4>*)&s1[0 + 0 + 4 * item.get_local_id(0) + i * 16] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + i * 16];
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(s0 * s1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v53_data;
              v53_data.copy_from(s0 + ((1_i32 ^ ((1_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(s0 + ((2_i32 ^ ((2_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v69_data;
              v69_data.copy_from(s0 + ((3_i32 ^ ((3_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(s0 + ((4_i32 ^ ((4_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v85_data;
              v85_data.copy_from(s0 + ((5_i32 ^ ((5_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v93_data;
              v93_data.copy_from(s0 + ((6_i32 ^ ((6_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v101_data;
              v101_data.copy_from(s0 + ((7_i32 ^ ((7_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v109_data;
              v109_data.copy_from(s0 + ((8_i32 ^ ((8_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v117_data;
              v117_data.copy_from(s0 + ((9_i32 ^ ((9_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v125_data;
              v125_data.copy_from(s0 + ((10_i32 ^ ((10_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v133_data;
              v133_data.copy_from(s0 + ((11_i32 ^ ((11_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v141_data;
              v141_data.copy_from(s0 + ((12_i32 ^ ((12_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v149_data;
              v149_data.copy_from(s0 + ((13_i32 ^ ((13_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v157_data;
              v157_data.copy_from(s0 + ((14_i32 ^ ((14_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v165_data;
              v165_data.copy_from(s0 + ((15_i32 ^ ((15_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v173_data;
              v173_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v181_data;
              v181_data.copy_from(s0 + ((17_i32 ^ ((17_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v189_data;
              v189_data.copy_from(s0 + ((18_i32 ^ ((18_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v197_data;
              v197_data.copy_from(s0 + ((19_i32 ^ ((19_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v198_acc{};
              tensorforge::intel_esimd::simd<float, 16> v205_data;
              v205_data.copy_from(s1 + ((0_i32 ^ ((0_i32 >> 5) & 31))));
              v198_acc += ((v205_data[0]) * v45_data);
              v198_acc += ((v205_data[1]) * v53_data);
              v198_acc += ((v205_data[2]) * v61_data);
              v198_acc += ((v205_data[3]) * v69_data);
              v198_acc += ((v205_data[4]) * v77_data);
              v198_acc += ((v205_data[5]) * v85_data);
              v198_acc += ((v205_data[6]) * v93_data);
              v198_acc += ((v205_data[7]) * v101_data);
              v198_acc += ((v205_data[8]) * v109_data);
              v198_acc += ((v205_data[9]) * v117_data);
              v198_acc += ((v205_data[10]) * v125_data);
              v198_acc += ((v205_data[11]) * v133_data);
              v198_acc += ((v205_data[12]) * v141_data);
              v198_acc += ((v205_data[13]) * v149_data);
              v198_acc += ((v205_data[14]) * v157_data);
              v198_acc += ((v205_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v244_data;
              v244_data.copy_from(s1 + ((16_i32 ^ ((16_i32 >> 5) & 31))));
              v198_acc += ((v244_data[0]) * v173_data);
              v198_acc += ((v244_data[1]) * v181_data);
              v198_acc += ((v244_data[2]) * v189_data);
              v198_acc += ((v244_data[3]) * v197_data);
              v198_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v253_acc{};
              tensorforge::intel_esimd::simd<float, 16> v260_data;
              v260_data.copy_from(s1 + ((20_i32 ^ ((20_i32 >> 5) & 31))));
              v253_acc += ((v260_data[0]) * v45_data);
              v253_acc += ((v260_data[1]) * v53_data);
              v253_acc += ((v260_data[2]) * v61_data);
              v253_acc += ((v260_data[3]) * v69_data);
              v253_acc += ((v260_data[4]) * v77_data);
              v253_acc += ((v260_data[5]) * v85_data);
              v253_acc += ((v260_data[6]) * v93_data);
              v253_acc += ((v260_data[7]) * v101_data);
              v253_acc += ((v260_data[8]) * v109_data);
              v253_acc += ((v260_data[9]) * v117_data);
              v253_acc += ((v260_data[10]) * v125_data);
              v253_acc += ((v260_data[11]) * v133_data);
              v253_acc += ((v260_data[12]) * v141_data);
              v253_acc += ((v260_data[13]) * v149_data);
              v253_acc += ((v260_data[14]) * v157_data);
              v253_acc += ((v260_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v299_data;
              v299_data.copy_from(s1 + ((36_i32 ^ ((36_i32 >> 5) & 31))));
              v253_acc += ((v299_data[0]) * v173_data);
              v253_acc += ((v299_data[1]) * v181_data);
              v253_acc += ((v299_data[2]) * v189_data);
              v253_acc += ((v299_data[3]) * v197_data);
              v253_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v308_acc{};
              tensorforge::intel_esimd::simd<float, 16> v315_data;
              v315_data.copy_from(s1 + ((40_i32 ^ ((40_i32 >> 5) & 31))));
              v308_acc += ((v315_data[0]) * v45_data);
              v308_acc += ((v315_data[1]) * v53_data);
              v308_acc += ((v315_data[2]) * v61_data);
              v308_acc += ((v315_data[3]) * v69_data);
              v308_acc += ((v315_data[4]) * v77_data);
              v308_acc += ((v315_data[5]) * v85_data);
              v308_acc += ((v315_data[6]) * v93_data);
              v308_acc += ((v315_data[7]) * v101_data);
              v308_acc += ((v315_data[8]) * v109_data);
              v308_acc += ((v315_data[9]) * v117_data);
              v308_acc += ((v315_data[10]) * v125_data);
              v308_acc += ((v315_data[11]) * v133_data);
              v308_acc += ((v315_data[12]) * v141_data);
              v308_acc += ((v315_data[13]) * v149_data);
              v308_acc += ((v315_data[14]) * v157_data);
              v308_acc += ((v315_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v354_data;
              v354_data.copy_from(s1 + ((56_i32 ^ ((56_i32 >> 5) & 31))));
              v308_acc += ((v354_data[0]) * v173_data);
              v308_acc += ((v354_data[1]) * v181_data);
              v308_acc += ((v354_data[2]) * v189_data);
              v308_acc += ((v354_data[3]) * v197_data);
              v308_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v363_acc{};
              tensorforge::intel_esimd::simd<float, 16> v370_data;
              v370_data.copy_from(s1 + ((60_i32 ^ ((60_i32 >> 5) & 31))));
              v363_acc += ((v370_data[0]) * v45_data);
              v363_acc += ((v370_data[1]) * v53_data);
              v363_acc += ((v370_data[2]) * v61_data);
              v363_acc += ((v370_data[3]) * v69_data);
              v363_acc += ((v370_data[4]) * v77_data);
              v363_acc += ((v370_data[5]) * v85_data);
              v363_acc += ((v370_data[6]) * v93_data);
              v363_acc += ((v370_data[7]) * v101_data);
              v363_acc += ((v370_data[8]) * v109_data);
              v363_acc += ((v370_data[9]) * v117_data);
              v363_acc += ((v370_data[10]) * v125_data);
              v363_acc += ((v370_data[11]) * v133_data);
              v363_acc += ((v370_data[12]) * v141_data);
              v363_acc += ((v370_data[13]) * v149_data);
              v363_acc += ((v370_data[14]) * v157_data);
              v363_acc += ((v370_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v409_data;
              v409_data.copy_from(s1 + ((76_i32 ^ ((76_i32 >> 5) & 31))));
              v363_acc += ((v409_data[0]) * v173_data);
              v363_acc += ((v409_data[1]) * v181_data);
              v363_acc += ((v409_data[2]) * v189_data);
              v363_acc += ((v409_data[3]) * v197_data);
              v363_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v418_acc{};
              tensorforge::intel_esimd::simd<float, 16> v425_data;
              v425_data.copy_from(s1 + ((80_i32 ^ ((80_i32 >> 5) & 31))));
              v418_acc += ((v425_data[0]) * v45_data);
              v418_acc += ((v425_data[1]) * v53_data);
              v418_acc += ((v425_data[2]) * v61_data);
              v418_acc += ((v425_data[3]) * v69_data);
              v418_acc += ((v425_data[4]) * v77_data);
              v418_acc += ((v425_data[5]) * v85_data);
              v418_acc += ((v425_data[6]) * v93_data);
              v418_acc += ((v425_data[7]) * v101_data);
              v418_acc += ((v425_data[8]) * v109_data);
              v418_acc += ((v425_data[9]) * v117_data);
              v418_acc += ((v425_data[10]) * v125_data);
              v418_acc += ((v425_data[11]) * v133_data);
              v418_acc += ((v425_data[12]) * v141_data);
              v418_acc += ((v425_data[13]) * v149_data);
              v418_acc += ((v425_data[14]) * v157_data);
              v418_acc += ((v425_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v464_data;
              v464_data.copy_from(s1 + ((96_i32 ^ ((96_i32 >> 5) & 31))));
              v418_acc += ((v464_data[0]) * v173_data);
              v418_acc += ((v464_data[1]) * v181_data);
              v418_acc += ((v464_data[2]) * v189_data);
              v418_acc += ((v464_data[3]) * v197_data);
              v418_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v473_acc{};
              tensorforge::intel_esimd::simd<float, 16> v480_data;
              v480_data.copy_from(s1 + ((100_i32 ^ ((100_i32 >> 5) & 31))));
              v473_acc += ((v480_data[0]) * v45_data);
              v473_acc += ((v480_data[1]) * v53_data);
              v473_acc += ((v480_data[2]) * v61_data);
              v473_acc += ((v480_data[3]) * v69_data);
              v473_acc += ((v480_data[4]) * v77_data);
              v473_acc += ((v480_data[5]) * v85_data);
              v473_acc += ((v480_data[6]) * v93_data);
              v473_acc += ((v480_data[7]) * v101_data);
              v473_acc += ((v480_data[8]) * v109_data);
              v473_acc += ((v480_data[9]) * v117_data);
              v473_acc += ((v480_data[10]) * v125_data);
              v473_acc += ((v480_data[11]) * v133_data);
              v473_acc += ((v480_data[12]) * v141_data);
              v473_acc += ((v480_data[13]) * v149_data);
              v473_acc += ((v480_data[14]) * v157_data);
              v473_acc += ((v480_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v519_data;
              v519_data.copy_from(s1 + ((116_i32 ^ ((116_i32 >> 5) & 31))));
              v473_acc += ((v519_data[0]) * v173_data);
              v473_acc += ((v519_data[1]) * v181_data);
              v473_acc += ((v519_data[2]) * v189_data);
              v473_acc += ((v519_data[3]) * v197_data);
              v473_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v528_acc{};
              tensorforge::intel_esimd::simd<float, 16> v535_data;
              v535_data.copy_from(s1 + ((120_i32 ^ ((120_i32 >> 5) & 31))));
              v528_acc += ((v535_data[0]) * v45_data);
              v528_acc += ((v535_data[1]) * v53_data);
              v528_acc += ((v535_data[2]) * v61_data);
              v528_acc += ((v535_data[3]) * v69_data);
              v528_acc += ((v535_data[4]) * v77_data);
              v528_acc += ((v535_data[5]) * v85_data);
              v528_acc += ((v535_data[6]) * v93_data);
              v528_acc += ((v535_data[7]) * v101_data);
              v528_acc += ((v535_data[8]) * v109_data);
              v528_acc += ((v535_data[9]) * v117_data);
              v528_acc += ((v535_data[10]) * v125_data);
              v528_acc += ((v535_data[11]) * v133_data);
              v528_acc += ((v535_data[12]) * v141_data);
              v528_acc += ((v535_data[13]) * v149_data);
              v528_acc += ((v535_data[14]) * v157_data);
              v528_acc += ((v535_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v574_data;
              v574_data.copy_from(s1 + ((136_i32 ^ ((136_i32 >> 5) & 31))));
              v528_acc += ((v574_data[0]) * v173_data);
              v528_acc += ((v574_data[1]) * v181_data);
              v528_acc += ((v574_data[2]) * v189_data);
              v528_acc += ((v574_data[3]) * v197_data);
              v528_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v583_acc{};
              tensorforge::intel_esimd::simd<float, 16> v590_data;
              v590_data.copy_from(s1 + ((140_i32 ^ ((140_i32 >> 5) & 31))));
              v583_acc += ((v590_data[0]) * v45_data);
              v583_acc += ((v590_data[1]) * v53_data);
              v583_acc += ((v590_data[2]) * v61_data);
              v583_acc += ((v590_data[3]) * v69_data);
              v583_acc += ((v590_data[4]) * v77_data);
              v583_acc += ((v590_data[5]) * v85_data);
              v583_acc += ((v590_data[6]) * v93_data);
              v583_acc += ((v590_data[7]) * v101_data);
              v583_acc += ((v590_data[8]) * v109_data);
              v583_acc += ((v590_data[9]) * v117_data);
              v583_acc += ((v590_data[10]) * v125_data);
              v583_acc += ((v590_data[11]) * v133_data);
              v583_acc += ((v590_data[12]) * v141_data);
              v583_acc += ((v590_data[13]) * v149_data);
              v583_acc += ((v590_data[14]) * v157_data);
              v583_acc += ((v590_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v629_data;
              v629_data.copy_from(s1 + ((156_i32 ^ ((156_i32 >> 5) & 31))));
              v583_acc += ((v629_data[0]) * v173_data);
              v583_acc += ((v629_data[1]) * v181_data);
              v583_acc += ((v629_data[2]) * v189_data);
              v583_acc += ((v629_data[3]) * v197_data);
              v583_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v638_acc{};
              tensorforge::intel_esimd::simd<float, 16> v645_data;
              v645_data.copy_from(s1 + ((160_i32 ^ ((160_i32 >> 5) & 31))));
              v638_acc += ((v645_data[0]) * v45_data);
              v638_acc += ((v645_data[1]) * v53_data);
              v638_acc += ((v645_data[2]) * v61_data);
              v638_acc += ((v645_data[3]) * v69_data);
              v638_acc += ((v645_data[4]) * v77_data);
              v638_acc += ((v645_data[5]) * v85_data);
              v638_acc += ((v645_data[6]) * v93_data);
              v638_acc += ((v645_data[7]) * v101_data);
              v638_acc += ((v645_data[8]) * v109_data);
              v638_acc += ((v645_data[9]) * v117_data);
              v638_acc += ((v645_data[10]) * v125_data);
              v638_acc += ((v645_data[11]) * v133_data);
              v638_acc += ((v645_data[12]) * v141_data);
              v638_acc += ((v645_data[13]) * v149_data);
              v638_acc += ((v645_data[14]) * v157_data);
              v638_acc += ((v645_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v684_data;
              v684_data.copy_from(s1 + ((176_i32 ^ ((176_i32 >> 5) & 31))));
              v638_acc += ((v684_data[0]) * v173_data);
              v638_acc += ((v684_data[1]) * v181_data);
              v638_acc += ((v684_data[2]) * v189_data);
              v638_acc += ((v684_data[3]) * v197_data);
              v638_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v693_acc{};
              tensorforge::intel_esimd::simd<float, 16> v700_data;
              v700_data.copy_from(s1 + ((180_i32 ^ ((180_i32 >> 5) & 31))));
              v693_acc += ((v700_data[0]) * v45_data);
              v693_acc += ((v700_data[1]) * v53_data);
              v693_acc += ((v700_data[2]) * v61_data);
              v693_acc += ((v700_data[3]) * v69_data);
              v693_acc += ((v700_data[4]) * v77_data);
              v693_acc += ((v700_data[5]) * v85_data);
              v693_acc += ((v700_data[6]) * v93_data);
              v693_acc += ((v700_data[7]) * v101_data);
              v693_acc += ((v700_data[8]) * v109_data);
              v693_acc += ((v700_data[9]) * v117_data);
              v693_acc += ((v700_data[10]) * v125_data);
              v693_acc += ((v700_data[11]) * v133_data);
              v693_acc += ((v700_data[12]) * v141_data);
              v693_acc += ((v700_data[13]) * v149_data);
              v693_acc += ((v700_data[14]) * v157_data);
              v693_acc += ((v700_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v739_data;
              v739_data.copy_from(s1 + ((196_i32 ^ ((196_i32 >> 5) & 31))));
              v693_acc += ((v739_data[0]) * v173_data);
              v693_acc += ((v739_data[1]) * v181_data);
              v693_acc += ((v739_data[2]) * v189_data);
              v693_acc += ((v739_data[3]) * v197_data);
              v693_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v748_acc{};
              tensorforge::intel_esimd::simd<float, 16> v755_data;
              v755_data.copy_from(s1 + ((200_i32 ^ ((200_i32 >> 5) & 31))));
              v748_acc += ((v755_data[0]) * v45_data);
              v748_acc += ((v755_data[1]) * v53_data);
              v748_acc += ((v755_data[2]) * v61_data);
              v748_acc += ((v755_data[3]) * v69_data);
              v748_acc += ((v755_data[4]) * v77_data);
              v748_acc += ((v755_data[5]) * v85_data);
              v748_acc += ((v755_data[6]) * v93_data);
              v748_acc += ((v755_data[7]) * v101_data);
              v748_acc += ((v755_data[8]) * v109_data);
              v748_acc += ((v755_data[9]) * v117_data);
              v748_acc += ((v755_data[10]) * v125_data);
              v748_acc += ((v755_data[11]) * v133_data);
              v748_acc += ((v755_data[12]) * v141_data);
              v748_acc += ((v755_data[13]) * v149_data);
              v748_acc += ((v755_data[14]) * v157_data);
              v748_acc += ((v755_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v794_data;
              v794_data.copy_from(s1 + ((216_i32 ^ ((216_i32 >> 5) & 31))));
              v748_acc += ((v794_data[0]) * v173_data);
              v748_acc += ((v794_data[1]) * v181_data);
              v748_acc += ((v794_data[2]) * v189_data);
              v748_acc += ((v794_data[3]) * v197_data);
              v748_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v803_acc{};
              tensorforge::intel_esimd::simd<float, 16> v810_data;
              v810_data.copy_from(s1 + ((220_i32 ^ ((220_i32 >> 5) & 31))));
              v803_acc += ((v810_data[0]) * v45_data);
              v803_acc += ((v810_data[1]) * v53_data);
              v803_acc += ((v810_data[2]) * v61_data);
              v803_acc += ((v810_data[3]) * v69_data);
              v803_acc += ((v810_data[4]) * v77_data);
              v803_acc += ((v810_data[5]) * v85_data);
              v803_acc += ((v810_data[6]) * v93_data);
              v803_acc += ((v810_data[7]) * v101_data);
              v803_acc += ((v810_data[8]) * v109_data);
              v803_acc += ((v810_data[9]) * v117_data);
              v803_acc += ((v810_data[10]) * v125_data);
              v803_acc += ((v810_data[11]) * v133_data);
              v803_acc += ((v810_data[12]) * v141_data);
              v803_acc += ((v810_data[13]) * v149_data);
              v803_acc += ((v810_data[14]) * v157_data);
              v803_acc += ((v810_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v849_data;
              v849_data.copy_from(s1 + ((236_i32 ^ ((236_i32 >> 5) & 31))));
              v803_acc += ((v849_data[0]) * v173_data);
              v803_acc += ((v849_data[1]) * v181_data);
              v803_acc += ((v849_data[2]) * v189_data);
              v803_acc += ((v849_data[3]) * v197_data);
              v803_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v858_acc{};
              tensorforge::intel_esimd::simd<float, 16> v865_data;
              v865_data.copy_from(s1 + ((240_i32 ^ ((240_i32 >> 5) & 31))));
              v858_acc += ((v865_data[0]) * v45_data);
              v858_acc += ((v865_data[1]) * v53_data);
              v858_acc += ((v865_data[2]) * v61_data);
              v858_acc += ((v865_data[3]) * v69_data);
              v858_acc += ((v865_data[4]) * v77_data);
              v858_acc += ((v865_data[5]) * v85_data);
              v858_acc += ((v865_data[6]) * v93_data);
              v858_acc += ((v865_data[7]) * v101_data);
              v858_acc += ((v865_data[8]) * v109_data);
              v858_acc += ((v865_data[9]) * v117_data);
              v858_acc += ((v865_data[10]) * v125_data);
              v858_acc += ((v865_data[11]) * v133_data);
              v858_acc += ((v865_data[12]) * v141_data);
              v858_acc += ((v865_data[13]) * v149_data);
              v858_acc += ((v865_data[14]) * v157_data);
              v858_acc += ((v865_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v904_data;
              v904_data.copy_from(s1 + ((256_i32 ^ ((256_i32 >> 5) & 31))));
              v858_acc += ((v904_data[0]) * v173_data);
              v858_acc += ((v904_data[1]) * v181_data);
              v858_acc += ((v904_data[2]) * v189_data);
              v858_acc += ((v904_data[3]) * v197_data);
              v858_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v913_acc{};
              tensorforge::intel_esimd::simd<float, 16> v920_data;
              v920_data.copy_from(s1 + ((260_i32 ^ ((260_i32 >> 5) & 31))));
              v913_acc += ((v920_data[0]) * v45_data);
              v913_acc += ((v920_data[1]) * v53_data);
              v913_acc += ((v920_data[2]) * v61_data);
              v913_acc += ((v920_data[3]) * v69_data);
              v913_acc += ((v920_data[4]) * v77_data);
              v913_acc += ((v920_data[5]) * v85_data);
              v913_acc += ((v920_data[6]) * v93_data);
              v913_acc += ((v920_data[7]) * v101_data);
              v913_acc += ((v920_data[8]) * v109_data);
              v913_acc += ((v920_data[9]) * v117_data);
              v913_acc += ((v920_data[10]) * v125_data);
              v913_acc += ((v920_data[11]) * v133_data);
              v913_acc += ((v920_data[12]) * v141_data);
              v913_acc += ((v920_data[13]) * v149_data);
              v913_acc += ((v920_data[14]) * v157_data);
              v913_acc += ((v920_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v959_data;
              v959_data.copy_from(s1 + ((276_i32 ^ ((276_i32 >> 5) & 31))));
              v913_acc += ((v959_data[0]) * v173_data);
              v913_acc += ((v959_data[1]) * v181_data);
              v913_acc += ((v959_data[2]) * v189_data);
              v913_acc += ((v959_data[3]) * v197_data);
              v913_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v968_acc{};
              tensorforge::intel_esimd::simd<float, 16> v975_data;
              v975_data.copy_from(s1 + ((280_i32 ^ ((280_i32 >> 5) & 31))));
              v968_acc += ((v975_data[0]) * v45_data);
              v968_acc += ((v975_data[1]) * v53_data);
              v968_acc += ((v975_data[2]) * v61_data);
              v968_acc += ((v975_data[3]) * v69_data);
              v968_acc += ((v975_data[4]) * v77_data);
              v968_acc += ((v975_data[5]) * v85_data);
              v968_acc += ((v975_data[6]) * v93_data);
              v968_acc += ((v975_data[7]) * v101_data);
              v968_acc += ((v975_data[8]) * v109_data);
              v968_acc += ((v975_data[9]) * v117_data);
              v968_acc += ((v975_data[10]) * v125_data);
              v968_acc += ((v975_data[11]) * v133_data);
              v968_acc += ((v975_data[12]) * v141_data);
              v968_acc += ((v975_data[13]) * v149_data);
              v968_acc += ((v975_data[14]) * v157_data);
              v968_acc += ((v975_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v1014_data;
              v1014_data.copy_from(s1 + ((296_i32 ^ ((296_i32 >> 5) & 31))));
              v968_acc += ((v1014_data[0]) * v173_data);
              v968_acc += ((v1014_data[1]) * v181_data);
              v968_acc += ((v1014_data[2]) * v189_data);
              v968_acc += ((v1014_data[3]) * v197_data);
              v968_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v1023_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1030_data;
              v1030_data.copy_from(s1 + ((300_i32 ^ ((300_i32 >> 5) & 31))));
              v1023_acc += ((v1030_data[0]) * v45_data);
              v1023_acc += ((v1030_data[1]) * v53_data);
              v1023_acc += ((v1030_data[2]) * v61_data);
              v1023_acc += ((v1030_data[3]) * v69_data);
              v1023_acc += ((v1030_data[4]) * v77_data);
              v1023_acc += ((v1030_data[5]) * v85_data);
              v1023_acc += ((v1030_data[6]) * v93_data);
              v1023_acc += ((v1030_data[7]) * v101_data);
              v1023_acc += ((v1030_data[8]) * v109_data);
              v1023_acc += ((v1030_data[9]) * v117_data);
              v1023_acc += ((v1030_data[10]) * v125_data);
              v1023_acc += ((v1030_data[11]) * v133_data);
              v1023_acc += ((v1030_data[12]) * v141_data);
              v1023_acc += ((v1030_data[13]) * v149_data);
              v1023_acc += ((v1030_data[14]) * v157_data);
              v1023_acc += ((v1030_data[15]) * v165_data);
              tensorforge::intel_esimd::simd<float, 16> v1069_data;
              v1069_data.copy_from(s1 + ((316_i32 ^ ((316_i32 >> 5) & 31))));
              v1023_acc += ((v1069_data[0]) * v173_data);
              v1023_acc += ((v1069_data[1]) * v181_data);
              v1023_acc += ((v1069_data[2]) * v189_data);
              v1023_acc += ((v1069_data[3]) * v197_data);
              v1023_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v1078_n1 = 0; v1078_n1 < 16; ++v1078_n1) {
                int32_t v1079_a = v1078_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1081_data;
                v1081_data.copy_from(ir0 + (v1079_a));
                v1081_data.copy_to(r0 + (v1079_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v1084_i1 = 0; v1084_i1 < 16; ++v1084_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1087_data;
                v1087_data.copy_from(r0 + ((v1084_i1 * 16)));
                v1087_data.copy_to(glb_m0 + ((v1084_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}


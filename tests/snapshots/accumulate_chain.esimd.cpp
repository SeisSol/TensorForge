// === base name ===
kernel_79b5ddae311858a9

// === header ===
void launcher_kernel_79b5ddae311858a9(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_79b5ddae311858a9(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_79b5ddae311858a9(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_79b5ddae311858a9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          float* __restrict__ s0 = &localShrMem0[0];
          float* __restrict__ s1 = &localShrMem0[0];
          float* __restrict__ s2 = &localShrMem0[0];
          float* __restrict__ s3 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
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
              for (int32_t v20_i1 = 0; v20_i1 < 12; ++v20_i1) {
                tensorforge::intel_esimd::simd<float, 12> v25_data;
                v25_data.copy_from(glb_m1 + ((v20_i1 * 12)));
                v25_data.copy_to(r0 + ((v20_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v28_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v29_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              float r2[192]{};
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v31_i1 = 0; v31_i1 < 12; ++v31_i1) {
                tensorforge::intel_esimd::simd<float, 12> v36_data;
                v36_data.copy_from(glb_m3 + ((v31_i1 * 12)));
                v36_data.copy_to(r2 + ((v31_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v48_data;
              v48_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v53_acc{};
              tensorforge::intel_esimd::simd<float, 16> v57_data;
              v57_data.copy_from(s0 + (0_i32));
              v53_acc += ((v57_data[0]) * v41_data);
              v53_acc += ((v57_data[1]) * v42_data);
              v53_acc += ((v57_data[2]) * v43_data);
              v53_acc += ((v57_data[3]) * v44_data);
              v53_acc += ((v57_data[4]) * v45_data);
              v53_acc += ((v57_data[5]) * v46_data);
              v53_acc += ((v57_data[6]) * v47_data);
              v53_acc += ((v57_data[7]) * v48_data);
              v53_acc += ((v57_data[8]) * v49_data);
              v53_acc += ((v57_data[9]) * v50_data);
              v53_acc += ((v57_data[10]) * v51_data);
              v53_acc += ((v57_data[11]) * v52_data);
              v53_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v82_acc{};
              tensorforge::intel_esimd::simd<float, 16> v86_data;
              v86_data.copy_from(s0 + (12_i32));
              v82_acc += ((v86_data[0]) * v41_data);
              v82_acc += ((v86_data[1]) * v42_data);
              v82_acc += ((v86_data[2]) * v43_data);
              v82_acc += ((v86_data[3]) * v44_data);
              v82_acc += ((v86_data[4]) * v45_data);
              v82_acc += ((v86_data[5]) * v46_data);
              v82_acc += ((v86_data[6]) * v47_data);
              v82_acc += ((v86_data[7]) * v48_data);
              v82_acc += ((v86_data[8]) * v49_data);
              v82_acc += ((v86_data[9]) * v50_data);
              v82_acc += ((v86_data[10]) * v51_data);
              v82_acc += ((v86_data[11]) * v52_data);
              v82_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v111_acc{};
              tensorforge::intel_esimd::simd<float, 16> v115_data;
              v115_data.copy_from(s0 + (24_i32));
              v111_acc += ((v115_data[0]) * v41_data);
              v111_acc += ((v115_data[1]) * v42_data);
              v111_acc += ((v115_data[2]) * v43_data);
              v111_acc += ((v115_data[3]) * v44_data);
              v111_acc += ((v115_data[4]) * v45_data);
              v111_acc += ((v115_data[5]) * v46_data);
              v111_acc += ((v115_data[6]) * v47_data);
              v111_acc += ((v115_data[7]) * v48_data);
              v111_acc += ((v115_data[8]) * v49_data);
              v111_acc += ((v115_data[9]) * v50_data);
              v111_acc += ((v115_data[10]) * v51_data);
              v111_acc += ((v115_data[11]) * v52_data);
              v111_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v140_acc{};
              tensorforge::intel_esimd::simd<float, 16> v144_data;
              v144_data.copy_from(s0 + (36_i32));
              v140_acc += ((v144_data[0]) * v41_data);
              v140_acc += ((v144_data[1]) * v42_data);
              v140_acc += ((v144_data[2]) * v43_data);
              v140_acc += ((v144_data[3]) * v44_data);
              v140_acc += ((v144_data[4]) * v45_data);
              v140_acc += ((v144_data[5]) * v46_data);
              v140_acc += ((v144_data[6]) * v47_data);
              v140_acc += ((v144_data[7]) * v48_data);
              v140_acc += ((v144_data[8]) * v49_data);
              v140_acc += ((v144_data[9]) * v50_data);
              v140_acc += ((v144_data[10]) * v51_data);
              v140_acc += ((v144_data[11]) * v52_data);
              v140_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v169_acc{};
              tensorforge::intel_esimd::simd<float, 16> v173_data;
              v173_data.copy_from(s0 + (48_i32));
              v169_acc += ((v173_data[0]) * v41_data);
              v169_acc += ((v173_data[1]) * v42_data);
              v169_acc += ((v173_data[2]) * v43_data);
              v169_acc += ((v173_data[3]) * v44_data);
              v169_acc += ((v173_data[4]) * v45_data);
              v169_acc += ((v173_data[5]) * v46_data);
              v169_acc += ((v173_data[6]) * v47_data);
              v169_acc += ((v173_data[7]) * v48_data);
              v169_acc += ((v173_data[8]) * v49_data);
              v169_acc += ((v173_data[9]) * v50_data);
              v169_acc += ((v173_data[10]) * v51_data);
              v169_acc += ((v173_data[11]) * v52_data);
              v169_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v198_acc{};
              tensorforge::intel_esimd::simd<float, 16> v202_data;
              v202_data.copy_from(s0 + (60_i32));
              v198_acc += ((v202_data[0]) * v41_data);
              v198_acc += ((v202_data[1]) * v42_data);
              v198_acc += ((v202_data[2]) * v43_data);
              v198_acc += ((v202_data[3]) * v44_data);
              v198_acc += ((v202_data[4]) * v45_data);
              v198_acc += ((v202_data[5]) * v46_data);
              v198_acc += ((v202_data[6]) * v47_data);
              v198_acc += ((v202_data[7]) * v48_data);
              v198_acc += ((v202_data[8]) * v49_data);
              v198_acc += ((v202_data[9]) * v50_data);
              v198_acc += ((v202_data[10]) * v51_data);
              v198_acc += ((v202_data[11]) * v52_data);
              v198_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v227_acc{};
              tensorforge::intel_esimd::simd<float, 16> v231_data;
              v231_data.copy_from(s0 + (72_i32));
              v227_acc += ((v231_data[0]) * v41_data);
              v227_acc += ((v231_data[1]) * v42_data);
              v227_acc += ((v231_data[2]) * v43_data);
              v227_acc += ((v231_data[3]) * v44_data);
              v227_acc += ((v231_data[4]) * v45_data);
              v227_acc += ((v231_data[5]) * v46_data);
              v227_acc += ((v231_data[6]) * v47_data);
              v227_acc += ((v231_data[7]) * v48_data);
              v227_acc += ((v231_data[8]) * v49_data);
              v227_acc += ((v231_data[9]) * v50_data);
              v227_acc += ((v231_data[10]) * v51_data);
              v227_acc += ((v231_data[11]) * v52_data);
              v227_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v256_acc{};
              tensorforge::intel_esimd::simd<float, 16> v260_data;
              v260_data.copy_from(s0 + (84_i32));
              v256_acc += ((v260_data[0]) * v41_data);
              v256_acc += ((v260_data[1]) * v42_data);
              v256_acc += ((v260_data[2]) * v43_data);
              v256_acc += ((v260_data[3]) * v44_data);
              v256_acc += ((v260_data[4]) * v45_data);
              v256_acc += ((v260_data[5]) * v46_data);
              v256_acc += ((v260_data[6]) * v47_data);
              v256_acc += ((v260_data[7]) * v48_data);
              v256_acc += ((v260_data[8]) * v49_data);
              v256_acc += ((v260_data[9]) * v50_data);
              v256_acc += ((v260_data[10]) * v51_data);
              v256_acc += ((v260_data[11]) * v52_data);
              v256_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v285_n1 = 0; v285_n1 < 8; ++v285_n1) {
                int32_t v286_a = v285_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v288_data;
                v288_data.copy_from(ir1 + (v286_a));
                v288_data.copy_to(r1 + (v286_a));
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v291_ld;
              v291_ld.copy_from(glb_m4 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v291_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v292_ld;
              v292_ld.copy_from(glb_m4 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v292_ld.copy_to(s1 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r2 = load{g>r}(glb_m3););
              float r4[192]{};
              // r4 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v294_i1 = 0; v294_i1 < 12; ++v294_i1) {
                tensorforge::intel_esimd::simd<float, 12> v299_data;
                v299_data.copy_from(glb_m5 + ((v294_i1 * 12)));
                v299_data.copy_to(r4 + ((v294_i1 * 16)));
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r3[128]{};
              // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir3[128]{};
              tensorforge::intel_esimd::simd<float, 16> v304_data;
              v304_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v305_data;
              v305_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v306_data;
              v306_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v307_data;
              v307_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v308_data;
              v308_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v309_data;
              v309_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v310_data;
              v310_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v311_data;
              v311_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v312_data;
              v312_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v313_data;
              v313_data.copy_from(r2 + (144));
              tensorforge::intel_esimd::simd<float, 16> v314_data;
              v314_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v315_data;
              v315_data.copy_from(r2 + (176));
              tensorforge::intel_esimd::simd<float, 16> v316_acc{};
              tensorforge::intel_esimd::simd<float, 16> v320_data;
              v320_data.copy_from(s1 + (0_i32));
              v316_acc += ((v320_data[0]) * v304_data);
              v316_acc += ((v320_data[1]) * v305_data);
              v316_acc += ((v320_data[2]) * v306_data);
              v316_acc += ((v320_data[3]) * v307_data);
              v316_acc += ((v320_data[4]) * v308_data);
              v316_acc += ((v320_data[5]) * v309_data);
              v316_acc += ((v320_data[6]) * v310_data);
              v316_acc += ((v320_data[7]) * v311_data);
              v316_acc += ((v320_data[8]) * v312_data);
              v316_acc += ((v320_data[9]) * v313_data);
              v316_acc += ((v320_data[10]) * v314_data);
              v316_acc += ((v320_data[11]) * v315_data);
              v316_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v345_acc{};
              tensorforge::intel_esimd::simd<float, 16> v349_data;
              v349_data.copy_from(s1 + (12_i32));
              v345_acc += ((v349_data[0]) * v304_data);
              v345_acc += ((v349_data[1]) * v305_data);
              v345_acc += ((v349_data[2]) * v306_data);
              v345_acc += ((v349_data[3]) * v307_data);
              v345_acc += ((v349_data[4]) * v308_data);
              v345_acc += ((v349_data[5]) * v309_data);
              v345_acc += ((v349_data[6]) * v310_data);
              v345_acc += ((v349_data[7]) * v311_data);
              v345_acc += ((v349_data[8]) * v312_data);
              v345_acc += ((v349_data[9]) * v313_data);
              v345_acc += ((v349_data[10]) * v314_data);
              v345_acc += ((v349_data[11]) * v315_data);
              v345_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v374_acc{};
              tensorforge::intel_esimd::simd<float, 16> v378_data;
              v378_data.copy_from(s1 + (24_i32));
              v374_acc += ((v378_data[0]) * v304_data);
              v374_acc += ((v378_data[1]) * v305_data);
              v374_acc += ((v378_data[2]) * v306_data);
              v374_acc += ((v378_data[3]) * v307_data);
              v374_acc += ((v378_data[4]) * v308_data);
              v374_acc += ((v378_data[5]) * v309_data);
              v374_acc += ((v378_data[6]) * v310_data);
              v374_acc += ((v378_data[7]) * v311_data);
              v374_acc += ((v378_data[8]) * v312_data);
              v374_acc += ((v378_data[9]) * v313_data);
              v374_acc += ((v378_data[10]) * v314_data);
              v374_acc += ((v378_data[11]) * v315_data);
              v374_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v403_acc{};
              tensorforge::intel_esimd::simd<float, 16> v407_data;
              v407_data.copy_from(s1 + (36_i32));
              v403_acc += ((v407_data[0]) * v304_data);
              v403_acc += ((v407_data[1]) * v305_data);
              v403_acc += ((v407_data[2]) * v306_data);
              v403_acc += ((v407_data[3]) * v307_data);
              v403_acc += ((v407_data[4]) * v308_data);
              v403_acc += ((v407_data[5]) * v309_data);
              v403_acc += ((v407_data[6]) * v310_data);
              v403_acc += ((v407_data[7]) * v311_data);
              v403_acc += ((v407_data[8]) * v312_data);
              v403_acc += ((v407_data[9]) * v313_data);
              v403_acc += ((v407_data[10]) * v314_data);
              v403_acc += ((v407_data[11]) * v315_data);
              v403_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v432_acc{};
              tensorforge::intel_esimd::simd<float, 16> v436_data;
              v436_data.copy_from(s1 + (48_i32));
              v432_acc += ((v436_data[0]) * v304_data);
              v432_acc += ((v436_data[1]) * v305_data);
              v432_acc += ((v436_data[2]) * v306_data);
              v432_acc += ((v436_data[3]) * v307_data);
              v432_acc += ((v436_data[4]) * v308_data);
              v432_acc += ((v436_data[5]) * v309_data);
              v432_acc += ((v436_data[6]) * v310_data);
              v432_acc += ((v436_data[7]) * v311_data);
              v432_acc += ((v436_data[8]) * v312_data);
              v432_acc += ((v436_data[9]) * v313_data);
              v432_acc += ((v436_data[10]) * v314_data);
              v432_acc += ((v436_data[11]) * v315_data);
              v432_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v461_acc{};
              tensorforge::intel_esimd::simd<float, 16> v465_data;
              v465_data.copy_from(s1 + (60_i32));
              v461_acc += ((v465_data[0]) * v304_data);
              v461_acc += ((v465_data[1]) * v305_data);
              v461_acc += ((v465_data[2]) * v306_data);
              v461_acc += ((v465_data[3]) * v307_data);
              v461_acc += ((v465_data[4]) * v308_data);
              v461_acc += ((v465_data[5]) * v309_data);
              v461_acc += ((v465_data[6]) * v310_data);
              v461_acc += ((v465_data[7]) * v311_data);
              v461_acc += ((v465_data[8]) * v312_data);
              v461_acc += ((v465_data[9]) * v313_data);
              v461_acc += ((v465_data[10]) * v314_data);
              v461_acc += ((v465_data[11]) * v315_data);
              v461_acc.copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 16> v490_acc{};
              tensorforge::intel_esimd::simd<float, 16> v494_data;
              v494_data.copy_from(s1 + (72_i32));
              v490_acc += ((v494_data[0]) * v304_data);
              v490_acc += ((v494_data[1]) * v305_data);
              v490_acc += ((v494_data[2]) * v306_data);
              v490_acc += ((v494_data[3]) * v307_data);
              v490_acc += ((v494_data[4]) * v308_data);
              v490_acc += ((v494_data[5]) * v309_data);
              v490_acc += ((v494_data[6]) * v310_data);
              v490_acc += ((v494_data[7]) * v311_data);
              v490_acc += ((v494_data[8]) * v312_data);
              v490_acc += ((v494_data[9]) * v313_data);
              v490_acc += ((v494_data[10]) * v314_data);
              v490_acc += ((v494_data[11]) * v315_data);
              v490_acc.copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v519_acc{};
              tensorforge::intel_esimd::simd<float, 16> v523_data;
              v523_data.copy_from(s1 + (84_i32));
              v519_acc += ((v523_data[0]) * v304_data);
              v519_acc += ((v523_data[1]) * v305_data);
              v519_acc += ((v523_data[2]) * v306_data);
              v519_acc += ((v523_data[3]) * v307_data);
              v519_acc += ((v523_data[4]) * v308_data);
              v519_acc += ((v523_data[5]) * v309_data);
              v519_acc += ((v523_data[6]) * v310_data);
              v519_acc += ((v523_data[7]) * v311_data);
              v519_acc += ((v523_data[8]) * v312_data);
              v519_acc += ((v523_data[9]) * v313_data);
              v519_acc += ((v523_data[10]) * v314_data);
              v519_acc += ((v523_data[11]) * v315_data);
              v519_acc.copy_to(ir3 + (112));
              #pragma unroll
              for (int32_t v548_n1 = 0; v548_n1 < 8; ++v548_n1) {
                int32_t v549_a = v548_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v551_data;
                v551_data.copy_from(ir3 + (v549_a));
                tensorforge::intel_esimd::simd<float, 12> v554_data;
                v554_data.copy_from(r1 + (v549_a));
                (v554_data + v551_data).copy_to(r3 + (v549_a));
              }
              // s2 = load{g>s}(glb_m6[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v558_ld;
              v558_ld.copy_from(glb_m6 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v558_ld.copy_to(s2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v559_ld;
              v559_ld.copy_from(glb_m6 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v559_ld.copy_to(s2 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r4 = load{g>r}(glb_m5););
              float r6[192]{};
              // r6 = load{g>r}(glb_m7);
              #pragma unroll
              for (int32_t v561_i1 = 0; v561_i1 < 12; ++v561_i1) {
                tensorforge::intel_esimd::simd<float, 12> v566_data;
                v566_data.copy_from(glb_m7 + ((v561_i1 * 12)));
                v566_data.copy_to(r6 + ((v561_i1 * 16)));
              }
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              float r5[128]{};
              // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[128]{};
              tensorforge::intel_esimd::simd<float, 16> v571_data;
              v571_data.copy_from(r4 + (0));
              tensorforge::intel_esimd::simd<float, 16> v572_data;
              v572_data.copy_from(r4 + (16));
              tensorforge::intel_esimd::simd<float, 16> v573_data;
              v573_data.copy_from(r4 + (32));
              tensorforge::intel_esimd::simd<float, 16> v574_data;
              v574_data.copy_from(r4 + (48));
              tensorforge::intel_esimd::simd<float, 16> v575_data;
              v575_data.copy_from(r4 + (64));
              tensorforge::intel_esimd::simd<float, 16> v576_data;
              v576_data.copy_from(r4 + (80));
              tensorforge::intel_esimd::simd<float, 16> v577_data;
              v577_data.copy_from(r4 + (96));
              tensorforge::intel_esimd::simd<float, 16> v578_data;
              v578_data.copy_from(r4 + (112));
              tensorforge::intel_esimd::simd<float, 16> v579_data;
              v579_data.copy_from(r4 + (128));
              tensorforge::intel_esimd::simd<float, 16> v580_data;
              v580_data.copy_from(r4 + (144));
              tensorforge::intel_esimd::simd<float, 16> v581_data;
              v581_data.copy_from(r4 + (160));
              tensorforge::intel_esimd::simd<float, 16> v582_data;
              v582_data.copy_from(r4 + (176));
              tensorforge::intel_esimd::simd<float, 16> v583_acc{};
              tensorforge::intel_esimd::simd<float, 16> v587_data;
              v587_data.copy_from(s2 + (0_i32));
              v583_acc += ((v587_data[0]) * v571_data);
              v583_acc += ((v587_data[1]) * v572_data);
              v583_acc += ((v587_data[2]) * v573_data);
              v583_acc += ((v587_data[3]) * v574_data);
              v583_acc += ((v587_data[4]) * v575_data);
              v583_acc += ((v587_data[5]) * v576_data);
              v583_acc += ((v587_data[6]) * v577_data);
              v583_acc += ((v587_data[7]) * v578_data);
              v583_acc += ((v587_data[8]) * v579_data);
              v583_acc += ((v587_data[9]) * v580_data);
              v583_acc += ((v587_data[10]) * v581_data);
              v583_acc += ((v587_data[11]) * v582_data);
              v583_acc.copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 16> v612_acc{};
              tensorforge::intel_esimd::simd<float, 16> v616_data;
              v616_data.copy_from(s2 + (12_i32));
              v612_acc += ((v616_data[0]) * v571_data);
              v612_acc += ((v616_data[1]) * v572_data);
              v612_acc += ((v616_data[2]) * v573_data);
              v612_acc += ((v616_data[3]) * v574_data);
              v612_acc += ((v616_data[4]) * v575_data);
              v612_acc += ((v616_data[5]) * v576_data);
              v612_acc += ((v616_data[6]) * v577_data);
              v612_acc += ((v616_data[7]) * v578_data);
              v612_acc += ((v616_data[8]) * v579_data);
              v612_acc += ((v616_data[9]) * v580_data);
              v612_acc += ((v616_data[10]) * v581_data);
              v612_acc += ((v616_data[11]) * v582_data);
              v612_acc.copy_to(ir5 + (16));
              tensorforge::intel_esimd::simd<float, 16> v641_acc{};
              tensorforge::intel_esimd::simd<float, 16> v645_data;
              v645_data.copy_from(s2 + (24_i32));
              v641_acc += ((v645_data[0]) * v571_data);
              v641_acc += ((v645_data[1]) * v572_data);
              v641_acc += ((v645_data[2]) * v573_data);
              v641_acc += ((v645_data[3]) * v574_data);
              v641_acc += ((v645_data[4]) * v575_data);
              v641_acc += ((v645_data[5]) * v576_data);
              v641_acc += ((v645_data[6]) * v577_data);
              v641_acc += ((v645_data[7]) * v578_data);
              v641_acc += ((v645_data[8]) * v579_data);
              v641_acc += ((v645_data[9]) * v580_data);
              v641_acc += ((v645_data[10]) * v581_data);
              v641_acc += ((v645_data[11]) * v582_data);
              v641_acc.copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 16> v670_acc{};
              tensorforge::intel_esimd::simd<float, 16> v674_data;
              v674_data.copy_from(s2 + (36_i32));
              v670_acc += ((v674_data[0]) * v571_data);
              v670_acc += ((v674_data[1]) * v572_data);
              v670_acc += ((v674_data[2]) * v573_data);
              v670_acc += ((v674_data[3]) * v574_data);
              v670_acc += ((v674_data[4]) * v575_data);
              v670_acc += ((v674_data[5]) * v576_data);
              v670_acc += ((v674_data[6]) * v577_data);
              v670_acc += ((v674_data[7]) * v578_data);
              v670_acc += ((v674_data[8]) * v579_data);
              v670_acc += ((v674_data[9]) * v580_data);
              v670_acc += ((v674_data[10]) * v581_data);
              v670_acc += ((v674_data[11]) * v582_data);
              v670_acc.copy_to(ir5 + (48));
              tensorforge::intel_esimd::simd<float, 16> v699_acc{};
              tensorforge::intel_esimd::simd<float, 16> v703_data;
              v703_data.copy_from(s2 + (48_i32));
              v699_acc += ((v703_data[0]) * v571_data);
              v699_acc += ((v703_data[1]) * v572_data);
              v699_acc += ((v703_data[2]) * v573_data);
              v699_acc += ((v703_data[3]) * v574_data);
              v699_acc += ((v703_data[4]) * v575_data);
              v699_acc += ((v703_data[5]) * v576_data);
              v699_acc += ((v703_data[6]) * v577_data);
              v699_acc += ((v703_data[7]) * v578_data);
              v699_acc += ((v703_data[8]) * v579_data);
              v699_acc += ((v703_data[9]) * v580_data);
              v699_acc += ((v703_data[10]) * v581_data);
              v699_acc += ((v703_data[11]) * v582_data);
              v699_acc.copy_to(ir5 + (64));
              tensorforge::intel_esimd::simd<float, 16> v728_acc{};
              tensorforge::intel_esimd::simd<float, 16> v732_data;
              v732_data.copy_from(s2 + (60_i32));
              v728_acc += ((v732_data[0]) * v571_data);
              v728_acc += ((v732_data[1]) * v572_data);
              v728_acc += ((v732_data[2]) * v573_data);
              v728_acc += ((v732_data[3]) * v574_data);
              v728_acc += ((v732_data[4]) * v575_data);
              v728_acc += ((v732_data[5]) * v576_data);
              v728_acc += ((v732_data[6]) * v577_data);
              v728_acc += ((v732_data[7]) * v578_data);
              v728_acc += ((v732_data[8]) * v579_data);
              v728_acc += ((v732_data[9]) * v580_data);
              v728_acc += ((v732_data[10]) * v581_data);
              v728_acc += ((v732_data[11]) * v582_data);
              v728_acc.copy_to(ir5 + (80));
              tensorforge::intel_esimd::simd<float, 16> v757_acc{};
              tensorforge::intel_esimd::simd<float, 16> v761_data;
              v761_data.copy_from(s2 + (72_i32));
              v757_acc += ((v761_data[0]) * v571_data);
              v757_acc += ((v761_data[1]) * v572_data);
              v757_acc += ((v761_data[2]) * v573_data);
              v757_acc += ((v761_data[3]) * v574_data);
              v757_acc += ((v761_data[4]) * v575_data);
              v757_acc += ((v761_data[5]) * v576_data);
              v757_acc += ((v761_data[6]) * v577_data);
              v757_acc += ((v761_data[7]) * v578_data);
              v757_acc += ((v761_data[8]) * v579_data);
              v757_acc += ((v761_data[9]) * v580_data);
              v757_acc += ((v761_data[10]) * v581_data);
              v757_acc += ((v761_data[11]) * v582_data);
              v757_acc.copy_to(ir5 + (96));
              tensorforge::intel_esimd::simd<float, 16> v786_acc{};
              tensorforge::intel_esimd::simd<float, 16> v790_data;
              v790_data.copy_from(s2 + (84_i32));
              v786_acc += ((v790_data[0]) * v571_data);
              v786_acc += ((v790_data[1]) * v572_data);
              v786_acc += ((v790_data[2]) * v573_data);
              v786_acc += ((v790_data[3]) * v574_data);
              v786_acc += ((v790_data[4]) * v575_data);
              v786_acc += ((v790_data[5]) * v576_data);
              v786_acc += ((v790_data[6]) * v577_data);
              v786_acc += ((v790_data[7]) * v578_data);
              v786_acc += ((v790_data[8]) * v579_data);
              v786_acc += ((v790_data[9]) * v580_data);
              v786_acc += ((v790_data[10]) * v581_data);
              v786_acc += ((v790_data[11]) * v582_data);
              v786_acc.copy_to(ir5 + (112));
              #pragma unroll
              for (int32_t v815_n1 = 0; v815_n1 < 8; ++v815_n1) {
                int32_t v816_a = v815_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v818_data;
                v818_data.copy_from(ir5 + (v816_a));
                tensorforge::intel_esimd::simd<float, 12> v821_data;
                v821_data.copy_from(r3 + (v816_a));
                (v821_data + v818_data).copy_to(r5 + (v816_a));
              }
              // s3 = load{g>s}(glb_m8[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v825_ld;
              v825_ld.copy_from(glb_m8 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v825_ld.copy_to(s3 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v826_ld;
              v826_ld.copy_from(glb_m8 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v826_ld.copy_to(s3 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r6 = load{g>r}(glb_m7););
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              float r7[128]{};
              // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir7[128]{};
              tensorforge::intel_esimd::simd<float, 16> v829_data;
              v829_data.copy_from(r6 + (0));
              tensorforge::intel_esimd::simd<float, 16> v830_data;
              v830_data.copy_from(r6 + (16));
              tensorforge::intel_esimd::simd<float, 16> v831_data;
              v831_data.copy_from(r6 + (32));
              tensorforge::intel_esimd::simd<float, 16> v832_data;
              v832_data.copy_from(r6 + (48));
              tensorforge::intel_esimd::simd<float, 16> v833_data;
              v833_data.copy_from(r6 + (64));
              tensorforge::intel_esimd::simd<float, 16> v834_data;
              v834_data.copy_from(r6 + (80));
              tensorforge::intel_esimd::simd<float, 16> v835_data;
              v835_data.copy_from(r6 + (96));
              tensorforge::intel_esimd::simd<float, 16> v836_data;
              v836_data.copy_from(r6 + (112));
              tensorforge::intel_esimd::simd<float, 16> v837_data;
              v837_data.copy_from(r6 + (128));
              tensorforge::intel_esimd::simd<float, 16> v838_data;
              v838_data.copy_from(r6 + (144));
              tensorforge::intel_esimd::simd<float, 16> v839_data;
              v839_data.copy_from(r6 + (160));
              tensorforge::intel_esimd::simd<float, 16> v840_data;
              v840_data.copy_from(r6 + (176));
              tensorforge::intel_esimd::simd<float, 16> v841_acc{};
              tensorforge::intel_esimd::simd<float, 16> v845_data;
              v845_data.copy_from(s3 + (0_i32));
              v841_acc += ((v845_data[0]) * v829_data);
              v841_acc += ((v845_data[1]) * v830_data);
              v841_acc += ((v845_data[2]) * v831_data);
              v841_acc += ((v845_data[3]) * v832_data);
              v841_acc += ((v845_data[4]) * v833_data);
              v841_acc += ((v845_data[5]) * v834_data);
              v841_acc += ((v845_data[6]) * v835_data);
              v841_acc += ((v845_data[7]) * v836_data);
              v841_acc += ((v845_data[8]) * v837_data);
              v841_acc += ((v845_data[9]) * v838_data);
              v841_acc += ((v845_data[10]) * v839_data);
              v841_acc += ((v845_data[11]) * v840_data);
              v841_acc.copy_to(ir7 + (0));
              tensorforge::intel_esimd::simd<float, 16> v870_acc{};
              tensorforge::intel_esimd::simd<float, 16> v874_data;
              v874_data.copy_from(s3 + (12_i32));
              v870_acc += ((v874_data[0]) * v829_data);
              v870_acc += ((v874_data[1]) * v830_data);
              v870_acc += ((v874_data[2]) * v831_data);
              v870_acc += ((v874_data[3]) * v832_data);
              v870_acc += ((v874_data[4]) * v833_data);
              v870_acc += ((v874_data[5]) * v834_data);
              v870_acc += ((v874_data[6]) * v835_data);
              v870_acc += ((v874_data[7]) * v836_data);
              v870_acc += ((v874_data[8]) * v837_data);
              v870_acc += ((v874_data[9]) * v838_data);
              v870_acc += ((v874_data[10]) * v839_data);
              v870_acc += ((v874_data[11]) * v840_data);
              v870_acc.copy_to(ir7 + (16));
              tensorforge::intel_esimd::simd<float, 16> v899_acc{};
              tensorforge::intel_esimd::simd<float, 16> v903_data;
              v903_data.copy_from(s3 + (24_i32));
              v899_acc += ((v903_data[0]) * v829_data);
              v899_acc += ((v903_data[1]) * v830_data);
              v899_acc += ((v903_data[2]) * v831_data);
              v899_acc += ((v903_data[3]) * v832_data);
              v899_acc += ((v903_data[4]) * v833_data);
              v899_acc += ((v903_data[5]) * v834_data);
              v899_acc += ((v903_data[6]) * v835_data);
              v899_acc += ((v903_data[7]) * v836_data);
              v899_acc += ((v903_data[8]) * v837_data);
              v899_acc += ((v903_data[9]) * v838_data);
              v899_acc += ((v903_data[10]) * v839_data);
              v899_acc += ((v903_data[11]) * v840_data);
              v899_acc.copy_to(ir7 + (32));
              tensorforge::intel_esimd::simd<float, 16> v928_acc{};
              tensorforge::intel_esimd::simd<float, 16> v932_data;
              v932_data.copy_from(s3 + (36_i32));
              v928_acc += ((v932_data[0]) * v829_data);
              v928_acc += ((v932_data[1]) * v830_data);
              v928_acc += ((v932_data[2]) * v831_data);
              v928_acc += ((v932_data[3]) * v832_data);
              v928_acc += ((v932_data[4]) * v833_data);
              v928_acc += ((v932_data[5]) * v834_data);
              v928_acc += ((v932_data[6]) * v835_data);
              v928_acc += ((v932_data[7]) * v836_data);
              v928_acc += ((v932_data[8]) * v837_data);
              v928_acc += ((v932_data[9]) * v838_data);
              v928_acc += ((v932_data[10]) * v839_data);
              v928_acc += ((v932_data[11]) * v840_data);
              v928_acc.copy_to(ir7 + (48));
              tensorforge::intel_esimd::simd<float, 16> v957_acc{};
              tensorforge::intel_esimd::simd<float, 16> v961_data;
              v961_data.copy_from(s3 + (48_i32));
              v957_acc += ((v961_data[0]) * v829_data);
              v957_acc += ((v961_data[1]) * v830_data);
              v957_acc += ((v961_data[2]) * v831_data);
              v957_acc += ((v961_data[3]) * v832_data);
              v957_acc += ((v961_data[4]) * v833_data);
              v957_acc += ((v961_data[5]) * v834_data);
              v957_acc += ((v961_data[6]) * v835_data);
              v957_acc += ((v961_data[7]) * v836_data);
              v957_acc += ((v961_data[8]) * v837_data);
              v957_acc += ((v961_data[9]) * v838_data);
              v957_acc += ((v961_data[10]) * v839_data);
              v957_acc += ((v961_data[11]) * v840_data);
              v957_acc.copy_to(ir7 + (64));
              tensorforge::intel_esimd::simd<float, 16> v986_acc{};
              tensorforge::intel_esimd::simd<float, 16> v990_data;
              v990_data.copy_from(s3 + (60_i32));
              v986_acc += ((v990_data[0]) * v829_data);
              v986_acc += ((v990_data[1]) * v830_data);
              v986_acc += ((v990_data[2]) * v831_data);
              v986_acc += ((v990_data[3]) * v832_data);
              v986_acc += ((v990_data[4]) * v833_data);
              v986_acc += ((v990_data[5]) * v834_data);
              v986_acc += ((v990_data[6]) * v835_data);
              v986_acc += ((v990_data[7]) * v836_data);
              v986_acc += ((v990_data[8]) * v837_data);
              v986_acc += ((v990_data[9]) * v838_data);
              v986_acc += ((v990_data[10]) * v839_data);
              v986_acc += ((v990_data[11]) * v840_data);
              v986_acc.copy_to(ir7 + (80));
              tensorforge::intel_esimd::simd<float, 16> v1015_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1019_data;
              v1019_data.copy_from(s3 + (72_i32));
              v1015_acc += ((v1019_data[0]) * v829_data);
              v1015_acc += ((v1019_data[1]) * v830_data);
              v1015_acc += ((v1019_data[2]) * v831_data);
              v1015_acc += ((v1019_data[3]) * v832_data);
              v1015_acc += ((v1019_data[4]) * v833_data);
              v1015_acc += ((v1019_data[5]) * v834_data);
              v1015_acc += ((v1019_data[6]) * v835_data);
              v1015_acc += ((v1019_data[7]) * v836_data);
              v1015_acc += ((v1019_data[8]) * v837_data);
              v1015_acc += ((v1019_data[9]) * v838_data);
              v1015_acc += ((v1019_data[10]) * v839_data);
              v1015_acc += ((v1019_data[11]) * v840_data);
              v1015_acc.copy_to(ir7 + (96));
              tensorforge::intel_esimd::simd<float, 16> v1044_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1048_data;
              v1048_data.copy_from(s3 + (84_i32));
              v1044_acc += ((v1048_data[0]) * v829_data);
              v1044_acc += ((v1048_data[1]) * v830_data);
              v1044_acc += ((v1048_data[2]) * v831_data);
              v1044_acc += ((v1048_data[3]) * v832_data);
              v1044_acc += ((v1048_data[4]) * v833_data);
              v1044_acc += ((v1048_data[5]) * v834_data);
              v1044_acc += ((v1048_data[6]) * v835_data);
              v1044_acc += ((v1048_data[7]) * v836_data);
              v1044_acc += ((v1048_data[8]) * v837_data);
              v1044_acc += ((v1048_data[9]) * v838_data);
              v1044_acc += ((v1048_data[10]) * v839_data);
              v1044_acc += ((v1048_data[11]) * v840_data);
              v1044_acc.copy_to(ir7 + (112));
              #pragma unroll
              for (int32_t v1073_n1 = 0; v1073_n1 < 8; ++v1073_n1) {
                int32_t v1074_a = v1073_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1076_data;
                v1076_data.copy_from(ir7 + (v1074_a));
                tensorforge::intel_esimd::simd<float, 12> v1079_data;
                v1079_data.copy_from(r5 + (v1074_a));
                (v1079_data + v1076_data).copy_to(r7 + (v1074_a));
              }
              // glb_m0 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1083_i1 = 0; v1083_i1 < 8; ++v1083_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1086_data;
                v1086_data.copy_from(r7 + ((v1083_i1 * 16)));
                v1086_data.copy_to(glb_m0 + ((v1083_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}


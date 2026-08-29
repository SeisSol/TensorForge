// === base name ===
kernel_7ab185b978

// === header ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7ab185b978(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7ab185b978(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (3072, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×9(32×9) {0..32}×{0..9} pointer_based
        // m1 16×9(16×9) {0..16}×{0..9} pointer_based
        // m2 16×9(16×9) {0..16}×{0..9} pointer_based
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based
        // m4 9×9(9×9) {0..9}×{0..9} pointer_based
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[384 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[384];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0][0 + m4_extraOffset];
              float r0[288]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
                int32_t v10_lead = v8_i0 * 32;
                #pragma unroll
                for (int32_t v9_i1 = 0; v9_i1 < 9; ++v9_i1) {
                  int32_t v13_a = v10_lead + (v9_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v14_data;
                  v14_data.copy_from(glb_m0 + (v13_a));
                  v14_data.copy_to(r0 + (v13_a));
                }
              }
              float r2[288]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 16> v24_data;
                v24_data.copy_from(glb_m1 + ((v19_i1 * 16)));
                v24_data.copy_to(r2 + ((v19_i1 * 32)));
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[288]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 32> v28_data;
              v28_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 32> v29_data;
              v29_data.copy_from(r1 + (0));
              (v29_data + v28_data).copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 32> v31_data;
              v31_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 32> v32_data;
              v32_data.copy_from(r1 + (32));
              (v32_data + v31_data).copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 32> v34_data;
              v34_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 32> v35_data;
              v35_data.copy_from(r1 + (64));
              (v35_data + v34_data).copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 32> v37_data;
              v37_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 32> v38_data;
              v38_data.copy_from(r1 + (96));
              (v38_data + v37_data).copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 32> v40_data;
              v40_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 32> v41_data;
              v41_data.copy_from(r1 + (128));
              (v41_data + v40_data).copy_to(r1 + (128));
              tensorforge::intel_esimd::simd<float, 32> v43_data;
              v43_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 32> v44_data;
              v44_data.copy_from(r1 + (160));
              (v44_data + v43_data).copy_to(r1 + (160));
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(r1 + (192));
              (v47_data + v46_data).copy_to(r1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v49_data;
              v49_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(r1 + (224));
              (v50_data + v49_data).copy_to(r1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v52_data;
              v52_data.copy_from(r0 + (256));
              tensorforge::intel_esimd::simd<float, 32> v53_data;
              v53_data.copy_from(r1 + (256));
              (v53_data + v52_data).copy_to(r1 + (256));
              float* __restrict__ s0 = &localShrMem0[96];
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v56_i0 = 0; v56_i0 < 1; ++v56_i0) {
                int32_t v58_a = v56_i0 * 32;
                #pragma unroll
                for (int32_t v57_i1 = 0; v57_i1 < 9; ++v57_i1) {
                  int32_t v60_a = v58_a + (v57_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v61_data;
                  v61_data.copy_from(r1 + (v60_a));
                  v61_data.copy_to(s0 + ((v60_a ^ ((v60_a >> 5) & 31))));
                }
              }
              float r4[288]{};
              // r4 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v70_i1 = 0; v70_i1 < 9; ++v70_i1) {
                tensorforge::intel_esimd::simd<float, 16> v75_data;
                v75_data.copy_from(glb_m2 + ((v70_i1 * 16)));
                v75_data.copy_to(r4 + ((v70_i1 * 32)));
              }
              // wait(r2 = load{g>r}(glb_m1););
              float r3[288]{};
              // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir3[288]{};
              tensorforge::intel_esimd::simd<float, 16> v80_data;
              v80_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v81_data;
              v81_data.copy_from(ir3 + (0));
              (v81_data + v80_data).copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v83_data;
              v83_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v84_data;
              v84_data.copy_from(ir3 + (32));
              (v84_data + v83_data).copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v86_data;
              v86_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v87_data;
              v87_data.copy_from(ir3 + (64));
              (v87_data + v86_data).copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v90_data;
              v90_data.copy_from(ir3 + (96));
              (v90_data + v89_data).copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v93_data;
              v93_data.copy_from(ir3 + (128));
              (v93_data + v92_data).copy_to(ir3 + (128));
              tensorforge::intel_esimd::simd<float, 16> v95_data;
              v95_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v96_data;
              v96_data.copy_from(ir3 + (160));
              (v96_data + v95_data).copy_to(ir3 + (160));
              tensorforge::intel_esimd::simd<float, 16> v98_data;
              v98_data.copy_from(r2 + (192));
              tensorforge::intel_esimd::simd<float, 16> v99_data;
              v99_data.copy_from(ir3 + (192));
              (v99_data + v98_data).copy_to(ir3 + (192));
              tensorforge::intel_esimd::simd<float, 16> v101_data;
              v101_data.copy_from(r2 + (224));
              tensorforge::intel_esimd::simd<float, 16> v102_data;
              v102_data.copy_from(ir3 + (224));
              (v102_data + v101_data).copy_to(ir3 + (224));
              tensorforge::intel_esimd::simd<float, 16> v104_data;
              v104_data.copy_from(r2 + (256));
              tensorforge::intel_esimd::simd<float, 16> v105_data;
              v105_data.copy_from(ir3 + (256));
              (v105_data + v104_data).copy_to(ir3 + (256));
              #pragma unroll
              for (int32_t v107_n1 = 0; v107_n1 < 9; ++v107_n1) {
                int32_t v108_a = v107_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v110_data;
                v110_data.copy_from(ir3 + (v108_a));
                tensorforge::intel_esimd::simd<float, 16> v118_data;
                v118_data.copy_from(s0 + ((v108_a ^ ((v108_a >> 5) & 31))));
                (v118_data + v110_data).copy_to(r3 + (v108_a));
              }
              // s0 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v122_i1 = 0; v122_i1 < 9; ++v122_i1) {
                int32_t v123_a = v122_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v125_data;
                v125_data.copy_from(r3 + (v123_a));
                v125_data.copy_to(s0 + ((v123_a ^ ((v123_a >> 5) & 31))));
              }
              // wait(r4 = load{g>r}(glb_m2););
              float r5[288]{};
              // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir5[288]{};
              tensorforge::intel_esimd::simd<float, 16> v135_data;
              v135_data.copy_from(r4 + (0));
              tensorforge::intel_esimd::simd<float, 16> v136_data;
              v136_data.copy_from(ir5 + (0));
              (v136_data + v135_data).copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 16> v138_data;
              v138_data.copy_from(r4 + (32));
              tensorforge::intel_esimd::simd<float, 16> v139_data;
              v139_data.copy_from(ir5 + (32));
              (v139_data + v138_data).copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 16> v141_data;
              v141_data.copy_from(r4 + (64));
              tensorforge::intel_esimd::simd<float, 16> v142_data;
              v142_data.copy_from(ir5 + (64));
              (v142_data + v141_data).copy_to(ir5 + (64));
              tensorforge::intel_esimd::simd<float, 16> v144_data;
              v144_data.copy_from(r4 + (96));
              tensorforge::intel_esimd::simd<float, 16> v145_data;
              v145_data.copy_from(ir5 + (96));
              (v145_data + v144_data).copy_to(ir5 + (96));
              tensorforge::intel_esimd::simd<float, 16> v147_data;
              v147_data.copy_from(r4 + (128));
              tensorforge::intel_esimd::simd<float, 16> v148_data;
              v148_data.copy_from(ir5 + (128));
              (v148_data + v147_data).copy_to(ir5 + (128));
              tensorforge::intel_esimd::simd<float, 16> v150_data;
              v150_data.copy_from(r4 + (160));
              tensorforge::intel_esimd::simd<float, 16> v151_data;
              v151_data.copy_from(ir5 + (160));
              (v151_data + v150_data).copy_to(ir5 + (160));
              tensorforge::intel_esimd::simd<float, 16> v153_data;
              v153_data.copy_from(r4 + (192));
              tensorforge::intel_esimd::simd<float, 16> v154_data;
              v154_data.copy_from(ir5 + (192));
              (v154_data + v153_data).copy_to(ir5 + (192));
              tensorforge::intel_esimd::simd<float, 16> v156_data;
              v156_data.copy_from(r4 + (224));
              tensorforge::intel_esimd::simd<float, 16> v157_data;
              v157_data.copy_from(ir5 + (224));
              (v157_data + v156_data).copy_to(ir5 + (224));
              tensorforge::intel_esimd::simd<float, 16> v159_data;
              v159_data.copy_from(r4 + (256));
              tensorforge::intel_esimd::simd<float, 16> v160_data;
              v160_data.copy_from(ir5 + (256));
              (v160_data + v159_data).copy_to(ir5 + (256));
              #pragma unroll
              for (int32_t v162_n1 = 0; v162_n1 < 9; ++v162_n1) {
                int32_t v163_a = v162_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v165_data;
                v165_data.copy_from(ir5 + (v163_a));
                tensorforge::intel_esimd::simd<float, 16> v173_data;
                v173_data.copy_from(s0 + ((v163_a ^ ((v163_a >> 5) & 31))));
                (v173_data + v165_data).copy_to(r5 + (v163_a));
              }
              // s0 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v177_i1 = 0; v177_i1 < 9; ++v177_i1) {
                int32_t v178_a = v177_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v180_data;
                v180_data.copy_from(r5 + (v178_a));
                v180_data.copy_to(s0 + ((v178_a ^ ((v178_a >> 5) & 31))));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m4[0, 1])
              *(sycl::vec<float, 2>*)&s1[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<float, 2>*)&glb_m4[0 + 0 + 2 * item.get_local_id(0) + 0];
              if (item.get_local_id(0) < 17) {
                s1[0 + 0 + 1 * item.get_local_id(0) + 64] = glb_m4[0 + 0 + 1 * item.get_local_id(0) + 64];
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r6[288]{};
              // r6 = +(s0 * s1) + None
              // [(0, 32), (0, 9)] [(0, 9)]
              float ir6[288]{};
              int32_t v194_sw = 0_i32 >> 5;
              tensorforge::intel_esimd::simd<float, 32> v197_data;
              v197_data.copy_from(s0 + ((0_i32 ^ (v194_sw & 31))));
              float v198_data = s1[0];
              tensorforge::intel_esimd::simd<float, 32> v200_data;
              v200_data.copy_from(ir6 + (0));
              (v200_data + (v197_data * v198_data)).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v208_data;
              v208_data.copy_from(s0 + ((0_i32 ^ (v194_sw & 31))));
              float v209_data = s1[9];
              tensorforge::intel_esimd::simd<float, 32> v211_data;
              v211_data.copy_from(ir6 + (32));
              (v211_data + (v208_data * v209_data)).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v219_data;
              v219_data.copy_from(s0 + ((0_i32 ^ (v194_sw & 31))));
              float v220_data = s1[18];
              tensorforge::intel_esimd::simd<float, 32> v222_data;
              v222_data.copy_from(ir6 + (64));
              (v222_data + (v219_data * v220_data)).copy_to(ir6 + (64));
              tensorforge::intel_esimd::simd<float, 32> v230_data;
              v230_data.copy_from(s0 + ((0_i32 ^ (v194_sw & 31))));
              float v231_data = s1[27];
              tensorforge::intel_esimd::simd<float, 32> v233_data;
              v233_data.copy_from(ir6 + (96));
              (v233_data + (v230_data * v231_data)).copy_to(ir6 + (96));
              tensorforge::intel_esimd::simd<float, 32> v241_data;
              v241_data.copy_from(s0 + ((0_i32 ^ (v194_sw & 31))));
              float v242_data = s1[36];
              tensorforge::intel_esimd::simd<float, 32> v244_data;
              v244_data.copy_from(ir6 + (128));
              (v244_data + (v241_data * v242_data)).copy_to(ir6 + (128));
              tensorforge::intel_esimd::simd<float, 32> v252_data;
              v252_data.copy_from(s0 + ((0_i32 ^ (v194_sw & 31))));
              float v253_data = s1[45];
              tensorforge::intel_esimd::simd<float, 32> v255_data;
              v255_data.copy_from(ir6 + (160));
              (v255_data + (v252_data * v253_data)).copy_to(ir6 + (160));
              tensorforge::intel_esimd::simd<float, 32> v263_data;
              v263_data.copy_from(s0 + ((0_i32 ^ (v194_sw & 31))));
              float v264_data = s1[54];
              tensorforge::intel_esimd::simd<float, 32> v266_data;
              v266_data.copy_from(ir6 + (192));
              (v266_data + (v263_data * v264_data)).copy_to(ir6 + (192));
              tensorforge::intel_esimd::simd<float, 32> v274_data;
              v274_data.copy_from(s0 + ((0_i32 ^ (v194_sw & 31))));
              float v275_data = s1[63];
              tensorforge::intel_esimd::simd<float, 32> v277_data;
              v277_data.copy_from(ir6 + (224));
              (v277_data + (v274_data * v275_data)).copy_to(ir6 + (224));
              tensorforge::intel_esimd::simd<float, 32> v285_data;
              v285_data.copy_from(s0 + ((0_i32 ^ (v194_sw & 31))));
              float v286_data = s1[72];
              tensorforge::intel_esimd::simd<float, 32> v288_data;
              v288_data.copy_from(ir6 + (256));
              (v288_data + (v285_data * v286_data)).copy_to(ir6 + (256));
              int32_t v293_sw = 32_i32 >> 5;
              tensorforge::intel_esimd::simd<float, 32> v296_data;
              v296_data.copy_from(s0 + ((32_i32 ^ (v293_sw & 31))));
              float v297_data = s1[1];
              tensorforge::intel_esimd::simd<float, 32> v299_data;
              v299_data.copy_from(ir6 + (0));
              (v299_data + (v296_data * v297_data)).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v307_data;
              v307_data.copy_from(s0 + ((32_i32 ^ (v293_sw & 31))));
              float v308_data = s1[10];
              tensorforge::intel_esimd::simd<float, 32> v310_data;
              v310_data.copy_from(ir6 + (32));
              (v310_data + (v307_data * v308_data)).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v318_data;
              v318_data.copy_from(s0 + ((32_i32 ^ (v293_sw & 31))));
              float v319_data = s1[19];
              tensorforge::intel_esimd::simd<float, 32> v321_data;
              v321_data.copy_from(ir6 + (64));
              (v321_data + (v318_data * v319_data)).copy_to(ir6 + (64));
              tensorforge::intel_esimd::simd<float, 32> v329_data;
              v329_data.copy_from(s0 + ((32_i32 ^ (v293_sw & 31))));
              float v330_data = s1[28];
              tensorforge::intel_esimd::simd<float, 32> v332_data;
              v332_data.copy_from(ir6 + (96));
              (v332_data + (v329_data * v330_data)).copy_to(ir6 + (96));
              tensorforge::intel_esimd::simd<float, 32> v340_data;
              v340_data.copy_from(s0 + ((32_i32 ^ (v293_sw & 31))));
              float v341_data = s1[37];
              tensorforge::intel_esimd::simd<float, 32> v343_data;
              v343_data.copy_from(ir6 + (128));
              (v343_data + (v340_data * v341_data)).copy_to(ir6 + (128));
              tensorforge::intel_esimd::simd<float, 32> v351_data;
              v351_data.copy_from(s0 + ((32_i32 ^ (v293_sw & 31))));
              float v352_data = s1[46];
              tensorforge::intel_esimd::simd<float, 32> v354_data;
              v354_data.copy_from(ir6 + (160));
              (v354_data + (v351_data * v352_data)).copy_to(ir6 + (160));
              tensorforge::intel_esimd::simd<float, 32> v362_data;
              v362_data.copy_from(s0 + ((32_i32 ^ (v293_sw & 31))));
              float v363_data = s1[55];
              tensorforge::intel_esimd::simd<float, 32> v365_data;
              v365_data.copy_from(ir6 + (192));
              (v365_data + (v362_data * v363_data)).copy_to(ir6 + (192));
              tensorforge::intel_esimd::simd<float, 32> v373_data;
              v373_data.copy_from(s0 + ((32_i32 ^ (v293_sw & 31))));
              float v374_data = s1[64];
              tensorforge::intel_esimd::simd<float, 32> v376_data;
              v376_data.copy_from(ir6 + (224));
              (v376_data + (v373_data * v374_data)).copy_to(ir6 + (224));
              tensorforge::intel_esimd::simd<float, 32> v384_data;
              v384_data.copy_from(s0 + ((32_i32 ^ (v293_sw & 31))));
              float v385_data = s1[73];
              tensorforge::intel_esimd::simd<float, 32> v387_data;
              v387_data.copy_from(ir6 + (256));
              (v387_data + (v384_data * v385_data)).copy_to(ir6 + (256));
              int32_t v392_sw = 64_i32 >> 5;
              tensorforge::intel_esimd::simd<float, 32> v395_data;
              v395_data.copy_from(s0 + ((64_i32 ^ (v392_sw & 31))));
              float v396_data = s1[2];
              tensorforge::intel_esimd::simd<float, 32> v398_data;
              v398_data.copy_from(ir6 + (0));
              (v398_data + (v395_data * v396_data)).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v406_data;
              v406_data.copy_from(s0 + ((64_i32 ^ (v392_sw & 31))));
              float v407_data = s1[11];
              tensorforge::intel_esimd::simd<float, 32> v409_data;
              v409_data.copy_from(ir6 + (32));
              (v409_data + (v406_data * v407_data)).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v417_data;
              v417_data.copy_from(s0 + ((64_i32 ^ (v392_sw & 31))));
              float v418_data = s1[20];
              tensorforge::intel_esimd::simd<float, 32> v420_data;
              v420_data.copy_from(ir6 + (64));
              (v420_data + (v417_data * v418_data)).copy_to(ir6 + (64));
              tensorforge::intel_esimd::simd<float, 32> v428_data;
              v428_data.copy_from(s0 + ((64_i32 ^ (v392_sw & 31))));
              float v429_data = s1[29];
              tensorforge::intel_esimd::simd<float, 32> v431_data;
              v431_data.copy_from(ir6 + (96));
              (v431_data + (v428_data * v429_data)).copy_to(ir6 + (96));
              tensorforge::intel_esimd::simd<float, 32> v439_data;
              v439_data.copy_from(s0 + ((64_i32 ^ (v392_sw & 31))));
              float v440_data = s1[38];
              tensorforge::intel_esimd::simd<float, 32> v442_data;
              v442_data.copy_from(ir6 + (128));
              (v442_data + (v439_data * v440_data)).copy_to(ir6 + (128));
              tensorforge::intel_esimd::simd<float, 32> v450_data;
              v450_data.copy_from(s0 + ((64_i32 ^ (v392_sw & 31))));
              float v451_data = s1[47];
              tensorforge::intel_esimd::simd<float, 32> v453_data;
              v453_data.copy_from(ir6 + (160));
              (v453_data + (v450_data * v451_data)).copy_to(ir6 + (160));
              tensorforge::intel_esimd::simd<float, 32> v461_data;
              v461_data.copy_from(s0 + ((64_i32 ^ (v392_sw & 31))));
              float v462_data = s1[56];
              tensorforge::intel_esimd::simd<float, 32> v464_data;
              v464_data.copy_from(ir6 + (192));
              (v464_data + (v461_data * v462_data)).copy_to(ir6 + (192));
              tensorforge::intel_esimd::simd<float, 32> v472_data;
              v472_data.copy_from(s0 + ((64_i32 ^ (v392_sw & 31))));
              float v473_data = s1[65];
              tensorforge::intel_esimd::simd<float, 32> v475_data;
              v475_data.copy_from(ir6 + (224));
              (v475_data + (v472_data * v473_data)).copy_to(ir6 + (224));
              tensorforge::intel_esimd::simd<float, 32> v483_data;
              v483_data.copy_from(s0 + ((64_i32 ^ (v392_sw & 31))));
              float v484_data = s1[74];
              tensorforge::intel_esimd::simd<float, 32> v486_data;
              v486_data.copy_from(ir6 + (256));
              (v486_data + (v483_data * v484_data)).copy_to(ir6 + (256));
              int32_t v491_sw = 96_i32 >> 5;
              tensorforge::intel_esimd::simd<float, 32> v494_data;
              v494_data.copy_from(s0 + ((96_i32 ^ (v491_sw & 31))));
              float v495_data = s1[3];
              tensorforge::intel_esimd::simd<float, 32> v497_data;
              v497_data.copy_from(ir6 + (0));
              (v497_data + (v494_data * v495_data)).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v505_data;
              v505_data.copy_from(s0 + ((96_i32 ^ (v491_sw & 31))));
              float v506_data = s1[12];
              tensorforge::intel_esimd::simd<float, 32> v508_data;
              v508_data.copy_from(ir6 + (32));
              (v508_data + (v505_data * v506_data)).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v516_data;
              v516_data.copy_from(s0 + ((96_i32 ^ (v491_sw & 31))));
              float v517_data = s1[21];
              tensorforge::intel_esimd::simd<float, 32> v519_data;
              v519_data.copy_from(ir6 + (64));
              (v519_data + (v516_data * v517_data)).copy_to(ir6 + (64));
              tensorforge::intel_esimd::simd<float, 32> v527_data;
              v527_data.copy_from(s0 + ((96_i32 ^ (v491_sw & 31))));
              float v528_data = s1[30];
              tensorforge::intel_esimd::simd<float, 32> v530_data;
              v530_data.copy_from(ir6 + (96));
              (v530_data + (v527_data * v528_data)).copy_to(ir6 + (96));
              tensorforge::intel_esimd::simd<float, 32> v538_data;
              v538_data.copy_from(s0 + ((96_i32 ^ (v491_sw & 31))));
              float v539_data = s1[39];
              tensorforge::intel_esimd::simd<float, 32> v541_data;
              v541_data.copy_from(ir6 + (128));
              (v541_data + (v538_data * v539_data)).copy_to(ir6 + (128));
              tensorforge::intel_esimd::simd<float, 32> v549_data;
              v549_data.copy_from(s0 + ((96_i32 ^ (v491_sw & 31))));
              float v550_data = s1[48];
              tensorforge::intel_esimd::simd<float, 32> v552_data;
              v552_data.copy_from(ir6 + (160));
              (v552_data + (v549_data * v550_data)).copy_to(ir6 + (160));
              tensorforge::intel_esimd::simd<float, 32> v560_data;
              v560_data.copy_from(s0 + ((96_i32 ^ (v491_sw & 31))));
              float v561_data = s1[57];
              tensorforge::intel_esimd::simd<float, 32> v563_data;
              v563_data.copy_from(ir6 + (192));
              (v563_data + (v560_data * v561_data)).copy_to(ir6 + (192));
              tensorforge::intel_esimd::simd<float, 32> v571_data;
              v571_data.copy_from(s0 + ((96_i32 ^ (v491_sw & 31))));
              float v572_data = s1[66];
              tensorforge::intel_esimd::simd<float, 32> v574_data;
              v574_data.copy_from(ir6 + (224));
              (v574_data + (v571_data * v572_data)).copy_to(ir6 + (224));
              tensorforge::intel_esimd::simd<float, 32> v582_data;
              v582_data.copy_from(s0 + ((96_i32 ^ (v491_sw & 31))));
              float v583_data = s1[75];
              tensorforge::intel_esimd::simd<float, 32> v585_data;
              v585_data.copy_from(ir6 + (256));
              (v585_data + (v582_data * v583_data)).copy_to(ir6 + (256));
              int32_t v590_sw = 128_i32 >> 5;
              tensorforge::intel_esimd::simd<float, 32> v593_data;
              v593_data.copy_from(s0 + ((128_i32 ^ (v590_sw & 31))));
              float v594_data = s1[4];
              tensorforge::intel_esimd::simd<float, 32> v596_data;
              v596_data.copy_from(ir6 + (0));
              (v596_data + (v593_data * v594_data)).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v604_data;
              v604_data.copy_from(s0 + ((128_i32 ^ (v590_sw & 31))));
              float v605_data = s1[13];
              tensorforge::intel_esimd::simd<float, 32> v607_data;
              v607_data.copy_from(ir6 + (32));
              (v607_data + (v604_data * v605_data)).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v615_data;
              v615_data.copy_from(s0 + ((128_i32 ^ (v590_sw & 31))));
              float v616_data = s1[22];
              tensorforge::intel_esimd::simd<float, 32> v618_data;
              v618_data.copy_from(ir6 + (64));
              (v618_data + (v615_data * v616_data)).copy_to(ir6 + (64));
              tensorforge::intel_esimd::simd<float, 32> v626_data;
              v626_data.copy_from(s0 + ((128_i32 ^ (v590_sw & 31))));
              float v627_data = s1[31];
              tensorforge::intel_esimd::simd<float, 32> v629_data;
              v629_data.copy_from(ir6 + (96));
              (v629_data + (v626_data * v627_data)).copy_to(ir6 + (96));
              tensorforge::intel_esimd::simd<float, 32> v637_data;
              v637_data.copy_from(s0 + ((128_i32 ^ (v590_sw & 31))));
              float v638_data = s1[40];
              tensorforge::intel_esimd::simd<float, 32> v640_data;
              v640_data.copy_from(ir6 + (128));
              (v640_data + (v637_data * v638_data)).copy_to(ir6 + (128));
              tensorforge::intel_esimd::simd<float, 32> v648_data;
              v648_data.copy_from(s0 + ((128_i32 ^ (v590_sw & 31))));
              float v649_data = s1[49];
              tensorforge::intel_esimd::simd<float, 32> v651_data;
              v651_data.copy_from(ir6 + (160));
              (v651_data + (v648_data * v649_data)).copy_to(ir6 + (160));
              tensorforge::intel_esimd::simd<float, 32> v659_data;
              v659_data.copy_from(s0 + ((128_i32 ^ (v590_sw & 31))));
              float v660_data = s1[58];
              tensorforge::intel_esimd::simd<float, 32> v662_data;
              v662_data.copy_from(ir6 + (192));
              (v662_data + (v659_data * v660_data)).copy_to(ir6 + (192));
              tensorforge::intel_esimd::simd<float, 32> v670_data;
              v670_data.copy_from(s0 + ((128_i32 ^ (v590_sw & 31))));
              float v671_data = s1[67];
              tensorforge::intel_esimd::simd<float, 32> v673_data;
              v673_data.copy_from(ir6 + (224));
              (v673_data + (v670_data * v671_data)).copy_to(ir6 + (224));
              tensorforge::intel_esimd::simd<float, 32> v681_data;
              v681_data.copy_from(s0 + ((128_i32 ^ (v590_sw & 31))));
              float v682_data = s1[76];
              tensorforge::intel_esimd::simd<float, 32> v684_data;
              v684_data.copy_from(ir6 + (256));
              (v684_data + (v681_data * v682_data)).copy_to(ir6 + (256));
              int32_t v689_sw = 160_i32 >> 5;
              tensorforge::intel_esimd::simd<float, 32> v692_data;
              v692_data.copy_from(s0 + ((160_i32 ^ (v689_sw & 31))));
              float v693_data = s1[5];
              tensorforge::intel_esimd::simd<float, 32> v695_data;
              v695_data.copy_from(ir6 + (0));
              (v695_data + (v692_data * v693_data)).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v703_data;
              v703_data.copy_from(s0 + ((160_i32 ^ (v689_sw & 31))));
              float v704_data = s1[14];
              tensorforge::intel_esimd::simd<float, 32> v706_data;
              v706_data.copy_from(ir6 + (32));
              (v706_data + (v703_data * v704_data)).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v714_data;
              v714_data.copy_from(s0 + ((160_i32 ^ (v689_sw & 31))));
              float v715_data = s1[23];
              tensorforge::intel_esimd::simd<float, 32> v717_data;
              v717_data.copy_from(ir6 + (64));
              (v717_data + (v714_data * v715_data)).copy_to(ir6 + (64));
              tensorforge::intel_esimd::simd<float, 32> v725_data;
              v725_data.copy_from(s0 + ((160_i32 ^ (v689_sw & 31))));
              float v726_data = s1[32];
              tensorforge::intel_esimd::simd<float, 32> v728_data;
              v728_data.copy_from(ir6 + (96));
              (v728_data + (v725_data * v726_data)).copy_to(ir6 + (96));
              tensorforge::intel_esimd::simd<float, 32> v736_data;
              v736_data.copy_from(s0 + ((160_i32 ^ (v689_sw & 31))));
              float v737_data = s1[41];
              tensorforge::intel_esimd::simd<float, 32> v739_data;
              v739_data.copy_from(ir6 + (128));
              (v739_data + (v736_data * v737_data)).copy_to(ir6 + (128));
              tensorforge::intel_esimd::simd<float, 32> v747_data;
              v747_data.copy_from(s0 + ((160_i32 ^ (v689_sw & 31))));
              float v748_data = s1[50];
              tensorforge::intel_esimd::simd<float, 32> v750_data;
              v750_data.copy_from(ir6 + (160));
              (v750_data + (v747_data * v748_data)).copy_to(ir6 + (160));
              tensorforge::intel_esimd::simd<float, 32> v758_data;
              v758_data.copy_from(s0 + ((160_i32 ^ (v689_sw & 31))));
              float v759_data = s1[59];
              tensorforge::intel_esimd::simd<float, 32> v761_data;
              v761_data.copy_from(ir6 + (192));
              (v761_data + (v758_data * v759_data)).copy_to(ir6 + (192));
              tensorforge::intel_esimd::simd<float, 32> v769_data;
              v769_data.copy_from(s0 + ((160_i32 ^ (v689_sw & 31))));
              float v770_data = s1[68];
              tensorforge::intel_esimd::simd<float, 32> v772_data;
              v772_data.copy_from(ir6 + (224));
              (v772_data + (v769_data * v770_data)).copy_to(ir6 + (224));
              tensorforge::intel_esimd::simd<float, 32> v780_data;
              v780_data.copy_from(s0 + ((160_i32 ^ (v689_sw & 31))));
              float v781_data = s1[77];
              tensorforge::intel_esimd::simd<float, 32> v783_data;
              v783_data.copy_from(ir6 + (256));
              (v783_data + (v780_data * v781_data)).copy_to(ir6 + (256));
              int32_t v788_sw = 192_i32 >> 5;
              tensorforge::intel_esimd::simd<float, 32> v791_data;
              v791_data.copy_from(s0 + ((192_i32 ^ (v788_sw & 31))));
              float v792_data = s1[6];
              tensorforge::intel_esimd::simd<float, 32> v794_data;
              v794_data.copy_from(ir6 + (0));
              (v794_data + (v791_data * v792_data)).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v802_data;
              v802_data.copy_from(s0 + ((192_i32 ^ (v788_sw & 31))));
              float v803_data = s1[15];
              tensorforge::intel_esimd::simd<float, 32> v805_data;
              v805_data.copy_from(ir6 + (32));
              (v805_data + (v802_data * v803_data)).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v813_data;
              v813_data.copy_from(s0 + ((192_i32 ^ (v788_sw & 31))));
              float v814_data = s1[24];
              tensorforge::intel_esimd::simd<float, 32> v816_data;
              v816_data.copy_from(ir6 + (64));
              (v816_data + (v813_data * v814_data)).copy_to(ir6 + (64));
              tensorforge::intel_esimd::simd<float, 32> v824_data;
              v824_data.copy_from(s0 + ((192_i32 ^ (v788_sw & 31))));
              float v825_data = s1[33];
              tensorforge::intel_esimd::simd<float, 32> v827_data;
              v827_data.copy_from(ir6 + (96));
              (v827_data + (v824_data * v825_data)).copy_to(ir6 + (96));
              tensorforge::intel_esimd::simd<float, 32> v835_data;
              v835_data.copy_from(s0 + ((192_i32 ^ (v788_sw & 31))));
              float v836_data = s1[42];
              tensorforge::intel_esimd::simd<float, 32> v838_data;
              v838_data.copy_from(ir6 + (128));
              (v838_data + (v835_data * v836_data)).copy_to(ir6 + (128));
              tensorforge::intel_esimd::simd<float, 32> v846_data;
              v846_data.copy_from(s0 + ((192_i32 ^ (v788_sw & 31))));
              float v847_data = s1[51];
              tensorforge::intel_esimd::simd<float, 32> v849_data;
              v849_data.copy_from(ir6 + (160));
              (v849_data + (v846_data * v847_data)).copy_to(ir6 + (160));
              tensorforge::intel_esimd::simd<float, 32> v857_data;
              v857_data.copy_from(s0 + ((192_i32 ^ (v788_sw & 31))));
              float v858_data = s1[60];
              tensorforge::intel_esimd::simd<float, 32> v860_data;
              v860_data.copy_from(ir6 + (192));
              (v860_data + (v857_data * v858_data)).copy_to(ir6 + (192));
              tensorforge::intel_esimd::simd<float, 32> v868_data;
              v868_data.copy_from(s0 + ((192_i32 ^ (v788_sw & 31))));
              float v869_data = s1[69];
              tensorforge::intel_esimd::simd<float, 32> v871_data;
              v871_data.copy_from(ir6 + (224));
              (v871_data + (v868_data * v869_data)).copy_to(ir6 + (224));
              tensorforge::intel_esimd::simd<float, 32> v879_data;
              v879_data.copy_from(s0 + ((192_i32 ^ (v788_sw & 31))));
              float v880_data = s1[78];
              tensorforge::intel_esimd::simd<float, 32> v882_data;
              v882_data.copy_from(ir6 + (256));
              (v882_data + (v879_data * v880_data)).copy_to(ir6 + (256));
              int32_t v887_sw = 224_i32 >> 5;
              tensorforge::intel_esimd::simd<float, 32> v890_data;
              v890_data.copy_from(s0 + ((224_i32 ^ (v887_sw & 31))));
              float v891_data = s1[7];
              tensorforge::intel_esimd::simd<float, 32> v893_data;
              v893_data.copy_from(ir6 + (0));
              (v893_data + (v890_data * v891_data)).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v901_data;
              v901_data.copy_from(s0 + ((224_i32 ^ (v887_sw & 31))));
              float v902_data = s1[16];
              tensorforge::intel_esimd::simd<float, 32> v904_data;
              v904_data.copy_from(ir6 + (32));
              (v904_data + (v901_data * v902_data)).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v912_data;
              v912_data.copy_from(s0 + ((224_i32 ^ (v887_sw & 31))));
              float v913_data = s1[25];
              tensorforge::intel_esimd::simd<float, 32> v915_data;
              v915_data.copy_from(ir6 + (64));
              (v915_data + (v912_data * v913_data)).copy_to(ir6 + (64));
              tensorforge::intel_esimd::simd<float, 32> v923_data;
              v923_data.copy_from(s0 + ((224_i32 ^ (v887_sw & 31))));
              float v924_data = s1[34];
              tensorforge::intel_esimd::simd<float, 32> v926_data;
              v926_data.copy_from(ir6 + (96));
              (v926_data + (v923_data * v924_data)).copy_to(ir6 + (96));
              tensorforge::intel_esimd::simd<float, 32> v934_data;
              v934_data.copy_from(s0 + ((224_i32 ^ (v887_sw & 31))));
              float v935_data = s1[43];
              tensorforge::intel_esimd::simd<float, 32> v937_data;
              v937_data.copy_from(ir6 + (128));
              (v937_data + (v934_data * v935_data)).copy_to(ir6 + (128));
              tensorforge::intel_esimd::simd<float, 32> v945_data;
              v945_data.copy_from(s0 + ((224_i32 ^ (v887_sw & 31))));
              float v946_data = s1[52];
              tensorforge::intel_esimd::simd<float, 32> v948_data;
              v948_data.copy_from(ir6 + (160));
              (v948_data + (v945_data * v946_data)).copy_to(ir6 + (160));
              tensorforge::intel_esimd::simd<float, 32> v956_data;
              v956_data.copy_from(s0 + ((224_i32 ^ (v887_sw & 31))));
              float v957_data = s1[61];
              tensorforge::intel_esimd::simd<float, 32> v959_data;
              v959_data.copy_from(ir6 + (192));
              (v959_data + (v956_data * v957_data)).copy_to(ir6 + (192));
              tensorforge::intel_esimd::simd<float, 32> v967_data;
              v967_data.copy_from(s0 + ((224_i32 ^ (v887_sw & 31))));
              float v968_data = s1[70];
              tensorforge::intel_esimd::simd<float, 32> v970_data;
              v970_data.copy_from(ir6 + (224));
              (v970_data + (v967_data * v968_data)).copy_to(ir6 + (224));
              tensorforge::intel_esimd::simd<float, 32> v978_data;
              v978_data.copy_from(s0 + ((224_i32 ^ (v887_sw & 31))));
              float v979_data = s1[79];
              tensorforge::intel_esimd::simd<float, 32> v981_data;
              v981_data.copy_from(ir6 + (256));
              (v981_data + (v978_data * v979_data)).copy_to(ir6 + (256));
              int32_t v986_sw = 256_i32 >> 5;
              tensorforge::intel_esimd::simd<float, 32> v989_data;
              v989_data.copy_from(s0 + ((256_i32 ^ (v986_sw & 31))));
              float v990_data = s1[8];
              tensorforge::intel_esimd::simd<float, 32> v992_data;
              v992_data.copy_from(ir6 + (0));
              (v992_data + (v989_data * v990_data)).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v1000_data;
              v1000_data.copy_from(s0 + ((256_i32 ^ (v986_sw & 31))));
              float v1001_data = s1[17];
              tensorforge::intel_esimd::simd<float, 32> v1003_data;
              v1003_data.copy_from(ir6 + (32));
              (v1003_data + (v1000_data * v1001_data)).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v1011_data;
              v1011_data.copy_from(s0 + ((256_i32 ^ (v986_sw & 31))));
              float v1012_data = s1[26];
              tensorforge::intel_esimd::simd<float, 32> v1014_data;
              v1014_data.copy_from(ir6 + (64));
              (v1014_data + (v1011_data * v1012_data)).copy_to(ir6 + (64));
              tensorforge::intel_esimd::simd<float, 32> v1022_data;
              v1022_data.copy_from(s0 + ((256_i32 ^ (v986_sw & 31))));
              float v1023_data = s1[35];
              tensorforge::intel_esimd::simd<float, 32> v1025_data;
              v1025_data.copy_from(ir6 + (96));
              (v1025_data + (v1022_data * v1023_data)).copy_to(ir6 + (96));
              tensorforge::intel_esimd::simd<float, 32> v1033_data;
              v1033_data.copy_from(s0 + ((256_i32 ^ (v986_sw & 31))));
              float v1034_data = s1[44];
              tensorforge::intel_esimd::simd<float, 32> v1036_data;
              v1036_data.copy_from(ir6 + (128));
              (v1036_data + (v1033_data * v1034_data)).copy_to(ir6 + (128));
              tensorforge::intel_esimd::simd<float, 32> v1044_data;
              v1044_data.copy_from(s0 + ((256_i32 ^ (v986_sw & 31))));
              float v1045_data = s1[53];
              tensorforge::intel_esimd::simd<float, 32> v1047_data;
              v1047_data.copy_from(ir6 + (160));
              (v1047_data + (v1044_data * v1045_data)).copy_to(ir6 + (160));
              tensorforge::intel_esimd::simd<float, 32> v1055_data;
              v1055_data.copy_from(s0 + ((256_i32 ^ (v986_sw & 31))));
              float v1056_data = s1[62];
              tensorforge::intel_esimd::simd<float, 32> v1058_data;
              v1058_data.copy_from(ir6 + (192));
              (v1058_data + (v1055_data * v1056_data)).copy_to(ir6 + (192));
              tensorforge::intel_esimd::simd<float, 32> v1066_data;
              v1066_data.copy_from(s0 + ((256_i32 ^ (v986_sw & 31))));
              float v1067_data = s1[71];
              tensorforge::intel_esimd::simd<float, 32> v1069_data;
              v1069_data.copy_from(ir6 + (224));
              (v1069_data + (v1066_data * v1067_data)).copy_to(ir6 + (224));
              tensorforge::intel_esimd::simd<float, 32> v1077_data;
              v1077_data.copy_from(s0 + ((256_i32 ^ (v986_sw & 31))));
              float v1078_data = s1[80];
              tensorforge::intel_esimd::simd<float, 32> v1080_data;
              v1080_data.copy_from(ir6 + (256));
              (v1080_data + (v1077_data * v1078_data)).copy_to(ir6 + (256));
              #pragma unroll
              for (int32_t v1082_n0 = 0; v1082_n0 < 1; ++v1082_n0) {
                int32_t v1084_a = v1082_n0 * 32;
                #pragma unroll
                for (int32_t v1083_n1 = 0; v1083_n1 < 9; ++v1083_n1) {
                  int32_t v1086_a = v1084_a + (v1083_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v1087_data;
                  v1087_data.copy_from(ir6 + (v1086_a));
                  v1087_data.copy_to(r6 + (v1086_a));
                }
              }
              // glb_m3 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v1091_i0 = 0; v1091_i0 < 1; ++v1091_i0) {
                int32_t v1093_a = v1091_i0 * 32;
                #pragma unroll
                for (int32_t v1092_i1 = 0; v1092_i1 < 9; ++v1092_i1) {
                  int32_t v1095_a = v1093_a + (v1092_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v1096_data;
                  v1096_data.copy_from(r6 + (v1095_a));
                  v1096_data.copy_to(glb_m3 + (v1095_a));
                }
              }
            }
          }
        }
      });
    }
  });
}


// === base name ===
kernel_671a350836

// === header ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_671a350836(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_671a350836(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 64×13(64×13) {0..64}×{0..13} pointer_based
        // m1 6(6) {0..6} none
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
        // t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..64}×{0..13}×{0..6})[0, 1, 2] = m0 64×13(64×13) {0..64}×{0..13} pointer_based({0..64}×{0..13})[0, 1]×m1 6(6) {0..6} none({0..6})[2]
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based({0..15}×{0..1}×{0..6})[0, 1, 2] += t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..15}×{0..1}×{0..6})[0, 1, 2]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const float *const __restrict__ glb_m1 = &m1[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[832]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v4_i0 = 0; v4_i0 < 2; ++v4_i0) {
                int32_t v6_lead = v4_i0 * 32;
                #pragma unroll
                for (int32_t v5_i1 = 0; v5_i1 < 13; ++v5_i1) {
                  int32_t v9_a = v6_lead + (v5_i1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v10_data;
                  v10_data.copy_from(glb_m0 + (v9_a));
                  v10_data.copy_to(r0 + (v9_a));
                }
              }
              float r2[384]{};
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v15_i1 = 0; v15_i1 < 1; ++v15_i1) {
                int32_t v23_a = 20_i32 + ((v15_i1 + 12) * 64);
                int32_t v28_a = 20 + (v15_i1 * 64);
                #pragma unroll
                for (int32_t v16_i2 = 0; v16_i2 < 6; ++v16_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v25_data;
                  v25_data.copy_from(glb_m2 + ((v23_a + (v16_i2 * 832))));
                  v25_data.copy_to(r2 + ((v28_a + (v16_i2 * 64))));
                }
              }
              #pragma unroll
              for (int32_t v30_i1 = 0; v30_i1 < 1; ++v30_i1) {
                int32_t v38_a = 32_i32 + ((v30_i1 + 12) * 64);
                int32_t v43_a = 32 + (v30_i1 * 64);
                #pragma unroll
                for (int32_t v31_i2 = 0; v31_i2 < 6; ++v31_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v40_data;
                  v40_data.copy_from(glb_m2 + ((v38_a + (v31_i2 * 832))));
                  v40_data.copy_to(r2 + ((v43_a + (v31_i2 * 64))));
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[4992]{};
              // r1 = +(r0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(r0 + (0));
              float v47_data = glb_m1[0];
              tensorforge::intel_esimd::simd<float, 32> v49_data;
              v49_data.copy_from(r1 + (0));
              (v49_data + (v46_data * v47_data)).copy_to(r1 + (0));
              float v52_data = glb_m1[1];
              tensorforge::intel_esimd::simd<float, 32> v54_data;
              v54_data.copy_from(r1 + (832));
              (v54_data + (v46_data * v52_data)).copy_to(r1 + (832));
              float v57_data = glb_m1[2];
              tensorforge::intel_esimd::simd<float, 32> v59_data;
              v59_data.copy_from(r1 + (1664));
              (v59_data + (v46_data * v57_data)).copy_to(r1 + (1664));
              float v62_data = glb_m1[3];
              tensorforge::intel_esimd::simd<float, 32> v64_data;
              v64_data.copy_from(r1 + (2496));
              (v64_data + (v46_data * v62_data)).copy_to(r1 + (2496));
              float v67_data = glb_m1[4];
              tensorforge::intel_esimd::simd<float, 32> v69_data;
              v69_data.copy_from(r1 + (3328));
              (v69_data + (v46_data * v67_data)).copy_to(r1 + (3328));
              float v72_data = glb_m1[5];
              tensorforge::intel_esimd::simd<float, 32> v74_data;
              v74_data.copy_from(r1 + (4160));
              (v74_data + (v46_data * v72_data)).copy_to(r1 + (4160));
              tensorforge::intel_esimd::simd<float, 32> v76_data;
              v76_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 32> v79_data;
              v79_data.copy_from(r1 + (64));
              (v79_data + (v76_data * v47_data)).copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 32> v84_data;
              v84_data.copy_from(r1 + (896));
              (v84_data + (v76_data * v52_data)).copy_to(r1 + (896));
              tensorforge::intel_esimd::simd<float, 32> v89_data;
              v89_data.copy_from(r1 + (1728));
              (v89_data + (v76_data * v57_data)).copy_to(r1 + (1728));
              tensorforge::intel_esimd::simd<float, 32> v94_data;
              v94_data.copy_from(r1 + (2560));
              (v94_data + (v76_data * v62_data)).copy_to(r1 + (2560));
              tensorforge::intel_esimd::simd<float, 32> v99_data;
              v99_data.copy_from(r1 + (3392));
              (v99_data + (v76_data * v67_data)).copy_to(r1 + (3392));
              tensorforge::intel_esimd::simd<float, 32> v104_data;
              v104_data.copy_from(r1 + (4224));
              (v104_data + (v76_data * v72_data)).copy_to(r1 + (4224));
              tensorforge::intel_esimd::simd<float, 32> v106_data;
              v106_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 32> v109_data;
              v109_data.copy_from(r1 + (128));
              (v109_data + (v106_data * v47_data)).copy_to(r1 + (128));
              tensorforge::intel_esimd::simd<float, 32> v114_data;
              v114_data.copy_from(r1 + (960));
              (v114_data + (v106_data * v52_data)).copy_to(r1 + (960));
              tensorforge::intel_esimd::simd<float, 32> v119_data;
              v119_data.copy_from(r1 + (1792));
              (v119_data + (v106_data * v57_data)).copy_to(r1 + (1792));
              tensorforge::intel_esimd::simd<float, 32> v124_data;
              v124_data.copy_from(r1 + (2624));
              (v124_data + (v106_data * v62_data)).copy_to(r1 + (2624));
              tensorforge::intel_esimd::simd<float, 32> v129_data;
              v129_data.copy_from(r1 + (3456));
              (v129_data + (v106_data * v67_data)).copy_to(r1 + (3456));
              tensorforge::intel_esimd::simd<float, 32> v134_data;
              v134_data.copy_from(r1 + (4288));
              (v134_data + (v106_data * v72_data)).copy_to(r1 + (4288));
              tensorforge::intel_esimd::simd<float, 32> v136_data;
              v136_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 32> v139_data;
              v139_data.copy_from(r1 + (192));
              (v139_data + (v136_data * v47_data)).copy_to(r1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v144_data;
              v144_data.copy_from(r1 + (1024));
              (v144_data + (v136_data * v52_data)).copy_to(r1 + (1024));
              tensorforge::intel_esimd::simd<float, 32> v149_data;
              v149_data.copy_from(r1 + (1856));
              (v149_data + (v136_data * v57_data)).copy_to(r1 + (1856));
              tensorforge::intel_esimd::simd<float, 32> v154_data;
              v154_data.copy_from(r1 + (2688));
              (v154_data + (v136_data * v62_data)).copy_to(r1 + (2688));
              tensorforge::intel_esimd::simd<float, 32> v159_data;
              v159_data.copy_from(r1 + (3520));
              (v159_data + (v136_data * v67_data)).copy_to(r1 + (3520));
              tensorforge::intel_esimd::simd<float, 32> v164_data;
              v164_data.copy_from(r1 + (4352));
              (v164_data + (v136_data * v72_data)).copy_to(r1 + (4352));
              tensorforge::intel_esimd::simd<float, 32> v166_data;
              v166_data.copy_from(r0 + (256));
              tensorforge::intel_esimd::simd<float, 32> v169_data;
              v169_data.copy_from(r1 + (256));
              (v169_data + (v166_data * v47_data)).copy_to(r1 + (256));
              tensorforge::intel_esimd::simd<float, 32> v174_data;
              v174_data.copy_from(r1 + (1088));
              (v174_data + (v166_data * v52_data)).copy_to(r1 + (1088));
              tensorforge::intel_esimd::simd<float, 32> v179_data;
              v179_data.copy_from(r1 + (1920));
              (v179_data + (v166_data * v57_data)).copy_to(r1 + (1920));
              tensorforge::intel_esimd::simd<float, 32> v184_data;
              v184_data.copy_from(r1 + (2752));
              (v184_data + (v166_data * v62_data)).copy_to(r1 + (2752));
              tensorforge::intel_esimd::simd<float, 32> v189_data;
              v189_data.copy_from(r1 + (3584));
              (v189_data + (v166_data * v67_data)).copy_to(r1 + (3584));
              tensorforge::intel_esimd::simd<float, 32> v194_data;
              v194_data.copy_from(r1 + (4416));
              (v194_data + (v166_data * v72_data)).copy_to(r1 + (4416));
              tensorforge::intel_esimd::simd<float, 32> v196_data;
              v196_data.copy_from(r0 + (320));
              tensorforge::intel_esimd::simd<float, 32> v199_data;
              v199_data.copy_from(r1 + (320));
              (v199_data + (v196_data * v47_data)).copy_to(r1 + (320));
              tensorforge::intel_esimd::simd<float, 32> v204_data;
              v204_data.copy_from(r1 + (1152));
              (v204_data + (v196_data * v52_data)).copy_to(r1 + (1152));
              tensorforge::intel_esimd::simd<float, 32> v209_data;
              v209_data.copy_from(r1 + (1984));
              (v209_data + (v196_data * v57_data)).copy_to(r1 + (1984));
              tensorforge::intel_esimd::simd<float, 32> v214_data;
              v214_data.copy_from(r1 + (2816));
              (v214_data + (v196_data * v62_data)).copy_to(r1 + (2816));
              tensorforge::intel_esimd::simd<float, 32> v219_data;
              v219_data.copy_from(r1 + (3648));
              (v219_data + (v196_data * v67_data)).copy_to(r1 + (3648));
              tensorforge::intel_esimd::simd<float, 32> v224_data;
              v224_data.copy_from(r1 + (4480));
              (v224_data + (v196_data * v72_data)).copy_to(r1 + (4480));
              tensorforge::intel_esimd::simd<float, 32> v226_data;
              v226_data.copy_from(r0 + (384));
              tensorforge::intel_esimd::simd<float, 32> v229_data;
              v229_data.copy_from(r1 + (384));
              (v229_data + (v226_data * v47_data)).copy_to(r1 + (384));
              tensorforge::intel_esimd::simd<float, 32> v234_data;
              v234_data.copy_from(r1 + (1216));
              (v234_data + (v226_data * v52_data)).copy_to(r1 + (1216));
              tensorforge::intel_esimd::simd<float, 32> v239_data;
              v239_data.copy_from(r1 + (2048));
              (v239_data + (v226_data * v57_data)).copy_to(r1 + (2048));
              tensorforge::intel_esimd::simd<float, 32> v244_data;
              v244_data.copy_from(r1 + (2880));
              (v244_data + (v226_data * v62_data)).copy_to(r1 + (2880));
              tensorforge::intel_esimd::simd<float, 32> v249_data;
              v249_data.copy_from(r1 + (3712));
              (v249_data + (v226_data * v67_data)).copy_to(r1 + (3712));
              tensorforge::intel_esimd::simd<float, 32> v254_data;
              v254_data.copy_from(r1 + (4544));
              (v254_data + (v226_data * v72_data)).copy_to(r1 + (4544));
              tensorforge::intel_esimd::simd<float, 32> v256_data;
              v256_data.copy_from(r0 + (448));
              tensorforge::intel_esimd::simd<float, 32> v259_data;
              v259_data.copy_from(r1 + (448));
              (v259_data + (v256_data * v47_data)).copy_to(r1 + (448));
              tensorforge::intel_esimd::simd<float, 32> v264_data;
              v264_data.copy_from(r1 + (1280));
              (v264_data + (v256_data * v52_data)).copy_to(r1 + (1280));
              tensorforge::intel_esimd::simd<float, 32> v269_data;
              v269_data.copy_from(r1 + (2112));
              (v269_data + (v256_data * v57_data)).copy_to(r1 + (2112));
              tensorforge::intel_esimd::simd<float, 32> v274_data;
              v274_data.copy_from(r1 + (2944));
              (v274_data + (v256_data * v62_data)).copy_to(r1 + (2944));
              tensorforge::intel_esimd::simd<float, 32> v279_data;
              v279_data.copy_from(r1 + (3776));
              (v279_data + (v256_data * v67_data)).copy_to(r1 + (3776));
              tensorforge::intel_esimd::simd<float, 32> v284_data;
              v284_data.copy_from(r1 + (4608));
              (v284_data + (v256_data * v72_data)).copy_to(r1 + (4608));
              tensorforge::intel_esimd::simd<float, 32> v286_data;
              v286_data.copy_from(r0 + (512));
              tensorforge::intel_esimd::simd<float, 32> v289_data;
              v289_data.copy_from(r1 + (512));
              (v289_data + (v286_data * v47_data)).copy_to(r1 + (512));
              tensorforge::intel_esimd::simd<float, 32> v294_data;
              v294_data.copy_from(r1 + (1344));
              (v294_data + (v286_data * v52_data)).copy_to(r1 + (1344));
              tensorforge::intel_esimd::simd<float, 32> v299_data;
              v299_data.copy_from(r1 + (2176));
              (v299_data + (v286_data * v57_data)).copy_to(r1 + (2176));
              tensorforge::intel_esimd::simd<float, 32> v304_data;
              v304_data.copy_from(r1 + (3008));
              (v304_data + (v286_data * v62_data)).copy_to(r1 + (3008));
              tensorforge::intel_esimd::simd<float, 32> v309_data;
              v309_data.copy_from(r1 + (3840));
              (v309_data + (v286_data * v67_data)).copy_to(r1 + (3840));
              tensorforge::intel_esimd::simd<float, 32> v314_data;
              v314_data.copy_from(r1 + (4672));
              (v314_data + (v286_data * v72_data)).copy_to(r1 + (4672));
              tensorforge::intel_esimd::simd<float, 32> v316_data;
              v316_data.copy_from(r0 + (576));
              tensorforge::intel_esimd::simd<float, 32> v319_data;
              v319_data.copy_from(r1 + (576));
              (v319_data + (v316_data * v47_data)).copy_to(r1 + (576));
              tensorforge::intel_esimd::simd<float, 32> v324_data;
              v324_data.copy_from(r1 + (1408));
              (v324_data + (v316_data * v52_data)).copy_to(r1 + (1408));
              tensorforge::intel_esimd::simd<float, 32> v329_data;
              v329_data.copy_from(r1 + (2240));
              (v329_data + (v316_data * v57_data)).copy_to(r1 + (2240));
              tensorforge::intel_esimd::simd<float, 32> v334_data;
              v334_data.copy_from(r1 + (3072));
              (v334_data + (v316_data * v62_data)).copy_to(r1 + (3072));
              tensorforge::intel_esimd::simd<float, 32> v339_data;
              v339_data.copy_from(r1 + (3904));
              (v339_data + (v316_data * v67_data)).copy_to(r1 + (3904));
              tensorforge::intel_esimd::simd<float, 32> v344_data;
              v344_data.copy_from(r1 + (4736));
              (v344_data + (v316_data * v72_data)).copy_to(r1 + (4736));
              tensorforge::intel_esimd::simd<float, 32> v346_data;
              v346_data.copy_from(r0 + (640));
              tensorforge::intel_esimd::simd<float, 32> v349_data;
              v349_data.copy_from(r1 + (640));
              (v349_data + (v346_data * v47_data)).copy_to(r1 + (640));
              tensorforge::intel_esimd::simd<float, 32> v354_data;
              v354_data.copy_from(r1 + (1472));
              (v354_data + (v346_data * v52_data)).copy_to(r1 + (1472));
              tensorforge::intel_esimd::simd<float, 32> v359_data;
              v359_data.copy_from(r1 + (2304));
              (v359_data + (v346_data * v57_data)).copy_to(r1 + (2304));
              tensorforge::intel_esimd::simd<float, 32> v364_data;
              v364_data.copy_from(r1 + (3136));
              (v364_data + (v346_data * v62_data)).copy_to(r1 + (3136));
              tensorforge::intel_esimd::simd<float, 32> v369_data;
              v369_data.copy_from(r1 + (3968));
              (v369_data + (v346_data * v67_data)).copy_to(r1 + (3968));
              tensorforge::intel_esimd::simd<float, 32> v374_data;
              v374_data.copy_from(r1 + (4800));
              (v374_data + (v346_data * v72_data)).copy_to(r1 + (4800));
              tensorforge::intel_esimd::simd<float, 32> v376_data;
              v376_data.copy_from(r0 + (704));
              tensorforge::intel_esimd::simd<float, 32> v379_data;
              v379_data.copy_from(r1 + (704));
              (v379_data + (v376_data * v47_data)).copy_to(r1 + (704));
              tensorforge::intel_esimd::simd<float, 32> v384_data;
              v384_data.copy_from(r1 + (1536));
              (v384_data + (v376_data * v52_data)).copy_to(r1 + (1536));
              tensorforge::intel_esimd::simd<float, 32> v389_data;
              v389_data.copy_from(r1 + (2368));
              (v389_data + (v376_data * v57_data)).copy_to(r1 + (2368));
              tensorforge::intel_esimd::simd<float, 32> v394_data;
              v394_data.copy_from(r1 + (3200));
              (v394_data + (v376_data * v62_data)).copy_to(r1 + (3200));
              tensorforge::intel_esimd::simd<float, 32> v399_data;
              v399_data.copy_from(r1 + (4032));
              (v399_data + (v376_data * v67_data)).copy_to(r1 + (4032));
              tensorforge::intel_esimd::simd<float, 32> v404_data;
              v404_data.copy_from(r1 + (4864));
              (v404_data + (v376_data * v72_data)).copy_to(r1 + (4864));
              tensorforge::intel_esimd::simd<float, 32> v406_data;
              v406_data.copy_from(r0 + (768));
              tensorforge::intel_esimd::simd<float, 32> v409_data;
              v409_data.copy_from(r1 + (768));
              (v409_data + (v406_data * v47_data)).copy_to(r1 + (768));
              tensorforge::intel_esimd::simd<float, 32> v414_data;
              v414_data.copy_from(r1 + (1600));
              (v414_data + (v406_data * v52_data)).copy_to(r1 + (1600));
              tensorforge::intel_esimd::simd<float, 32> v419_data;
              v419_data.copy_from(r1 + (2432));
              (v419_data + (v406_data * v57_data)).copy_to(r1 + (2432));
              tensorforge::intel_esimd::simd<float, 32> v424_data;
              v424_data.copy_from(r1 + (3264));
              (v424_data + (v406_data * v62_data)).copy_to(r1 + (3264));
              tensorforge::intel_esimd::simd<float, 32> v429_data;
              v429_data.copy_from(r1 + (4096));
              (v429_data + (v406_data * v67_data)).copy_to(r1 + (4096));
              tensorforge::intel_esimd::simd<float, 32> v434_data;
              v434_data.copy_from(r1 + (4928));
              (v434_data + (v406_data * v72_data)).copy_to(r1 + (4928));
              tensorforge::intel_esimd::simd<float, 32> v436_data;
              v436_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 32> v439_data;
              v439_data.copy_from(r1 + (32));
              (v439_data + (v436_data * v47_data)).copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 32> v444_data;
              v444_data.copy_from(r1 + (864));
              (v444_data + (v436_data * v52_data)).copy_to(r1 + (864));
              tensorforge::intel_esimd::simd<float, 32> v449_data;
              v449_data.copy_from(r1 + (1696));
              (v449_data + (v436_data * v57_data)).copy_to(r1 + (1696));
              tensorforge::intel_esimd::simd<float, 32> v454_data;
              v454_data.copy_from(r1 + (2528));
              (v454_data + (v436_data * v62_data)).copy_to(r1 + (2528));
              tensorforge::intel_esimd::simd<float, 32> v459_data;
              v459_data.copy_from(r1 + (3360));
              (v459_data + (v436_data * v67_data)).copy_to(r1 + (3360));
              tensorforge::intel_esimd::simd<float, 32> v464_data;
              v464_data.copy_from(r1 + (4192));
              (v464_data + (v436_data * v72_data)).copy_to(r1 + (4192));
              tensorforge::intel_esimd::simd<float, 32> v466_data;
              v466_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 32> v469_data;
              v469_data.copy_from(r1 + (96));
              (v469_data + (v466_data * v47_data)).copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 32> v474_data;
              v474_data.copy_from(r1 + (928));
              (v474_data + (v466_data * v52_data)).copy_to(r1 + (928));
              tensorforge::intel_esimd::simd<float, 32> v479_data;
              v479_data.copy_from(r1 + (1760));
              (v479_data + (v466_data * v57_data)).copy_to(r1 + (1760));
              tensorforge::intel_esimd::simd<float, 32> v484_data;
              v484_data.copy_from(r1 + (2592));
              (v484_data + (v466_data * v62_data)).copy_to(r1 + (2592));
              tensorforge::intel_esimd::simd<float, 32> v489_data;
              v489_data.copy_from(r1 + (3424));
              (v489_data + (v466_data * v67_data)).copy_to(r1 + (3424));
              tensorforge::intel_esimd::simd<float, 32> v494_data;
              v494_data.copy_from(r1 + (4256));
              (v494_data + (v466_data * v72_data)).copy_to(r1 + (4256));
              tensorforge::intel_esimd::simd<float, 32> v496_data;
              v496_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 32> v499_data;
              v499_data.copy_from(r1 + (160));
              (v499_data + (v496_data * v47_data)).copy_to(r1 + (160));
              tensorforge::intel_esimd::simd<float, 32> v504_data;
              v504_data.copy_from(r1 + (992));
              (v504_data + (v496_data * v52_data)).copy_to(r1 + (992));
              tensorforge::intel_esimd::simd<float, 32> v509_data;
              v509_data.copy_from(r1 + (1824));
              (v509_data + (v496_data * v57_data)).copy_to(r1 + (1824));
              tensorforge::intel_esimd::simd<float, 32> v514_data;
              v514_data.copy_from(r1 + (2656));
              (v514_data + (v496_data * v62_data)).copy_to(r1 + (2656));
              tensorforge::intel_esimd::simd<float, 32> v519_data;
              v519_data.copy_from(r1 + (3488));
              (v519_data + (v496_data * v67_data)).copy_to(r1 + (3488));
              tensorforge::intel_esimd::simd<float, 32> v524_data;
              v524_data.copy_from(r1 + (4320));
              (v524_data + (v496_data * v72_data)).copy_to(r1 + (4320));
              tensorforge::intel_esimd::simd<float, 32> v526_data;
              v526_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 32> v529_data;
              v529_data.copy_from(r1 + (224));
              (v529_data + (v526_data * v47_data)).copy_to(r1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v534_data;
              v534_data.copy_from(r1 + (1056));
              (v534_data + (v526_data * v52_data)).copy_to(r1 + (1056));
              tensorforge::intel_esimd::simd<float, 32> v539_data;
              v539_data.copy_from(r1 + (1888));
              (v539_data + (v526_data * v57_data)).copy_to(r1 + (1888));
              tensorforge::intel_esimd::simd<float, 32> v544_data;
              v544_data.copy_from(r1 + (2720));
              (v544_data + (v526_data * v62_data)).copy_to(r1 + (2720));
              tensorforge::intel_esimd::simd<float, 32> v549_data;
              v549_data.copy_from(r1 + (3552));
              (v549_data + (v526_data * v67_data)).copy_to(r1 + (3552));
              tensorforge::intel_esimd::simd<float, 32> v554_data;
              v554_data.copy_from(r1 + (4384));
              (v554_data + (v526_data * v72_data)).copy_to(r1 + (4384));
              tensorforge::intel_esimd::simd<float, 32> v556_data;
              v556_data.copy_from(r0 + (288));
              tensorforge::intel_esimd::simd<float, 32> v559_data;
              v559_data.copy_from(r1 + (288));
              (v559_data + (v556_data * v47_data)).copy_to(r1 + (288));
              tensorforge::intel_esimd::simd<float, 32> v564_data;
              v564_data.copy_from(r1 + (1120));
              (v564_data + (v556_data * v52_data)).copy_to(r1 + (1120));
              tensorforge::intel_esimd::simd<float, 32> v569_data;
              v569_data.copy_from(r1 + (1952));
              (v569_data + (v556_data * v57_data)).copy_to(r1 + (1952));
              tensorforge::intel_esimd::simd<float, 32> v574_data;
              v574_data.copy_from(r1 + (2784));
              (v574_data + (v556_data * v62_data)).copy_to(r1 + (2784));
              tensorforge::intel_esimd::simd<float, 32> v579_data;
              v579_data.copy_from(r1 + (3616));
              (v579_data + (v556_data * v67_data)).copy_to(r1 + (3616));
              tensorforge::intel_esimd::simd<float, 32> v584_data;
              v584_data.copy_from(r1 + (4448));
              (v584_data + (v556_data * v72_data)).copy_to(r1 + (4448));
              tensorforge::intel_esimd::simd<float, 32> v586_data;
              v586_data.copy_from(r0 + (352));
              tensorforge::intel_esimd::simd<float, 32> v589_data;
              v589_data.copy_from(r1 + (352));
              (v589_data + (v586_data * v47_data)).copy_to(r1 + (352));
              tensorforge::intel_esimd::simd<float, 32> v594_data;
              v594_data.copy_from(r1 + (1184));
              (v594_data + (v586_data * v52_data)).copy_to(r1 + (1184));
              tensorforge::intel_esimd::simd<float, 32> v599_data;
              v599_data.copy_from(r1 + (2016));
              (v599_data + (v586_data * v57_data)).copy_to(r1 + (2016));
              tensorforge::intel_esimd::simd<float, 32> v604_data;
              v604_data.copy_from(r1 + (2848));
              (v604_data + (v586_data * v62_data)).copy_to(r1 + (2848));
              tensorforge::intel_esimd::simd<float, 32> v609_data;
              v609_data.copy_from(r1 + (3680));
              (v609_data + (v586_data * v67_data)).copy_to(r1 + (3680));
              tensorforge::intel_esimd::simd<float, 32> v614_data;
              v614_data.copy_from(r1 + (4512));
              (v614_data + (v586_data * v72_data)).copy_to(r1 + (4512));
              tensorforge::intel_esimd::simd<float, 32> v616_data;
              v616_data.copy_from(r0 + (416));
              tensorforge::intel_esimd::simd<float, 32> v619_data;
              v619_data.copy_from(r1 + (416));
              (v619_data + (v616_data * v47_data)).copy_to(r1 + (416));
              tensorforge::intel_esimd::simd<float, 32> v624_data;
              v624_data.copy_from(r1 + (1248));
              (v624_data + (v616_data * v52_data)).copy_to(r1 + (1248));
              tensorforge::intel_esimd::simd<float, 32> v629_data;
              v629_data.copy_from(r1 + (2080));
              (v629_data + (v616_data * v57_data)).copy_to(r1 + (2080));
              tensorforge::intel_esimd::simd<float, 32> v634_data;
              v634_data.copy_from(r1 + (2912));
              (v634_data + (v616_data * v62_data)).copy_to(r1 + (2912));
              tensorforge::intel_esimd::simd<float, 32> v639_data;
              v639_data.copy_from(r1 + (3744));
              (v639_data + (v616_data * v67_data)).copy_to(r1 + (3744));
              tensorforge::intel_esimd::simd<float, 32> v644_data;
              v644_data.copy_from(r1 + (4576));
              (v644_data + (v616_data * v72_data)).copy_to(r1 + (4576));
              tensorforge::intel_esimd::simd<float, 32> v646_data;
              v646_data.copy_from(r0 + (480));
              tensorforge::intel_esimd::simd<float, 32> v649_data;
              v649_data.copy_from(r1 + (480));
              (v649_data + (v646_data * v47_data)).copy_to(r1 + (480));
              tensorforge::intel_esimd::simd<float, 32> v654_data;
              v654_data.copy_from(r1 + (1312));
              (v654_data + (v646_data * v52_data)).copy_to(r1 + (1312));
              tensorforge::intel_esimd::simd<float, 32> v659_data;
              v659_data.copy_from(r1 + (2144));
              (v659_data + (v646_data * v57_data)).copy_to(r1 + (2144));
              tensorforge::intel_esimd::simd<float, 32> v664_data;
              v664_data.copy_from(r1 + (2976));
              (v664_data + (v646_data * v62_data)).copy_to(r1 + (2976));
              tensorforge::intel_esimd::simd<float, 32> v669_data;
              v669_data.copy_from(r1 + (3808));
              (v669_data + (v646_data * v67_data)).copy_to(r1 + (3808));
              tensorforge::intel_esimd::simd<float, 32> v674_data;
              v674_data.copy_from(r1 + (4640));
              (v674_data + (v646_data * v72_data)).copy_to(r1 + (4640));
              tensorforge::intel_esimd::simd<float, 32> v676_data;
              v676_data.copy_from(r0 + (544));
              tensorforge::intel_esimd::simd<float, 32> v679_data;
              v679_data.copy_from(r1 + (544));
              (v679_data + (v676_data * v47_data)).copy_to(r1 + (544));
              tensorforge::intel_esimd::simd<float, 32> v684_data;
              v684_data.copy_from(r1 + (1376));
              (v684_data + (v676_data * v52_data)).copy_to(r1 + (1376));
              tensorforge::intel_esimd::simd<float, 32> v689_data;
              v689_data.copy_from(r1 + (2208));
              (v689_data + (v676_data * v57_data)).copy_to(r1 + (2208));
              tensorforge::intel_esimd::simd<float, 32> v694_data;
              v694_data.copy_from(r1 + (3040));
              (v694_data + (v676_data * v62_data)).copy_to(r1 + (3040));
              tensorforge::intel_esimd::simd<float, 32> v699_data;
              v699_data.copy_from(r1 + (3872));
              (v699_data + (v676_data * v67_data)).copy_to(r1 + (3872));
              tensorforge::intel_esimd::simd<float, 32> v704_data;
              v704_data.copy_from(r1 + (4704));
              (v704_data + (v676_data * v72_data)).copy_to(r1 + (4704));
              tensorforge::intel_esimd::simd<float, 32> v706_data;
              v706_data.copy_from(r0 + (608));
              tensorforge::intel_esimd::simd<float, 32> v709_data;
              v709_data.copy_from(r1 + (608));
              (v709_data + (v706_data * v47_data)).copy_to(r1 + (608));
              tensorforge::intel_esimd::simd<float, 32> v714_data;
              v714_data.copy_from(r1 + (1440));
              (v714_data + (v706_data * v52_data)).copy_to(r1 + (1440));
              tensorforge::intel_esimd::simd<float, 32> v719_data;
              v719_data.copy_from(r1 + (2272));
              (v719_data + (v706_data * v57_data)).copy_to(r1 + (2272));
              tensorforge::intel_esimd::simd<float, 32> v724_data;
              v724_data.copy_from(r1 + (3104));
              (v724_data + (v706_data * v62_data)).copy_to(r1 + (3104));
              tensorforge::intel_esimd::simd<float, 32> v729_data;
              v729_data.copy_from(r1 + (3936));
              (v729_data + (v706_data * v67_data)).copy_to(r1 + (3936));
              tensorforge::intel_esimd::simd<float, 32> v734_data;
              v734_data.copy_from(r1 + (4768));
              (v734_data + (v706_data * v72_data)).copy_to(r1 + (4768));
              tensorforge::intel_esimd::simd<float, 32> v736_data;
              v736_data.copy_from(r0 + (672));
              tensorforge::intel_esimd::simd<float, 32> v739_data;
              v739_data.copy_from(r1 + (672));
              (v739_data + (v736_data * v47_data)).copy_to(r1 + (672));
              tensorforge::intel_esimd::simd<float, 32> v744_data;
              v744_data.copy_from(r1 + (1504));
              (v744_data + (v736_data * v52_data)).copy_to(r1 + (1504));
              tensorforge::intel_esimd::simd<float, 32> v749_data;
              v749_data.copy_from(r1 + (2336));
              (v749_data + (v736_data * v57_data)).copy_to(r1 + (2336));
              tensorforge::intel_esimd::simd<float, 32> v754_data;
              v754_data.copy_from(r1 + (3168));
              (v754_data + (v736_data * v62_data)).copy_to(r1 + (3168));
              tensorforge::intel_esimd::simd<float, 32> v759_data;
              v759_data.copy_from(r1 + (4000));
              (v759_data + (v736_data * v67_data)).copy_to(r1 + (4000));
              tensorforge::intel_esimd::simd<float, 32> v764_data;
              v764_data.copy_from(r1 + (4832));
              (v764_data + (v736_data * v72_data)).copy_to(r1 + (4832));
              tensorforge::intel_esimd::simd<float, 32> v766_data;
              v766_data.copy_from(r0 + (736));
              tensorforge::intel_esimd::simd<float, 32> v769_data;
              v769_data.copy_from(r1 + (736));
              (v769_data + (v766_data * v47_data)).copy_to(r1 + (736));
              tensorforge::intel_esimd::simd<float, 32> v774_data;
              v774_data.copy_from(r1 + (1568));
              (v774_data + (v766_data * v52_data)).copy_to(r1 + (1568));
              tensorforge::intel_esimd::simd<float, 32> v779_data;
              v779_data.copy_from(r1 + (2400));
              (v779_data + (v766_data * v57_data)).copy_to(r1 + (2400));
              tensorforge::intel_esimd::simd<float, 32> v784_data;
              v784_data.copy_from(r1 + (3232));
              (v784_data + (v766_data * v62_data)).copy_to(r1 + (3232));
              tensorforge::intel_esimd::simd<float, 32> v789_data;
              v789_data.copy_from(r1 + (4064));
              (v789_data + (v766_data * v67_data)).copy_to(r1 + (4064));
              tensorforge::intel_esimd::simd<float, 32> v794_data;
              v794_data.copy_from(r1 + (4896));
              (v794_data + (v766_data * v72_data)).copy_to(r1 + (4896));
              tensorforge::intel_esimd::simd<float, 32> v796_data;
              v796_data.copy_from(r0 + (800));
              tensorforge::intel_esimd::simd<float, 32> v799_data;
              v799_data.copy_from(r1 + (800));
              (v799_data + (v796_data * v47_data)).copy_to(r1 + (800));
              tensorforge::intel_esimd::simd<float, 32> v804_data;
              v804_data.copy_from(r1 + (1632));
              (v804_data + (v796_data * v52_data)).copy_to(r1 + (1632));
              tensorforge::intel_esimd::simd<float, 32> v809_data;
              v809_data.copy_from(r1 + (2464));
              (v809_data + (v796_data * v57_data)).copy_to(r1 + (2464));
              tensorforge::intel_esimd::simd<float, 32> v814_data;
              v814_data.copy_from(r1 + (3296));
              (v814_data + (v796_data * v62_data)).copy_to(r1 + (3296));
              tensorforge::intel_esimd::simd<float, 32> v819_data;
              v819_data.copy_from(r1 + (4128));
              (v819_data + (v796_data * v67_data)).copy_to(r1 + (4128));
              tensorforge::intel_esimd::simd<float, 32> v824_data;
              v824_data.copy_from(r1 + (4960));
              (v824_data + (v796_data * v72_data)).copy_to(r1 + (4960));
              // wait(r2 = load{g>r}(glb_m2););
              float r3[384]{};
              // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(20, 35), (0, 1), (0, 6)] []
              float ir3[384]{};
              tensorforge::intel_esimd::simd<float, 12> v828_data;
              v828_data.copy_from(r1 + (788));
              tensorforge::intel_esimd::simd<float, 12> v829_data;
              v829_data.copy_from(ir3 + (20));
              (v829_data + v828_data).copy_to(ir3 + (20));
              tensorforge::intel_esimd::simd<float, 12> v831_data;
              v831_data.copy_from(r1 + (1620));
              tensorforge::intel_esimd::simd<float, 12> v832_data;
              v832_data.copy_from(ir3 + (84));
              (v832_data + v831_data).copy_to(ir3 + (84));
              tensorforge::intel_esimd::simd<float, 12> v834_data;
              v834_data.copy_from(r1 + (2452));
              tensorforge::intel_esimd::simd<float, 12> v835_data;
              v835_data.copy_from(ir3 + (148));
              (v835_data + v834_data).copy_to(ir3 + (148));
              tensorforge::intel_esimd::simd<float, 12> v837_data;
              v837_data.copy_from(r1 + (3284));
              tensorforge::intel_esimd::simd<float, 12> v838_data;
              v838_data.copy_from(ir3 + (212));
              (v838_data + v837_data).copy_to(ir3 + (212));
              tensorforge::intel_esimd::simd<float, 12> v840_data;
              v840_data.copy_from(r1 + (4116));
              tensorforge::intel_esimd::simd<float, 12> v841_data;
              v841_data.copy_from(ir3 + (276));
              (v841_data + v840_data).copy_to(ir3 + (276));
              tensorforge::intel_esimd::simd<float, 12> v843_data;
              v843_data.copy_from(r1 + (4948));
              tensorforge::intel_esimd::simd<float, 12> v844_data;
              v844_data.copy_from(ir3 + (340));
              (v844_data + v843_data).copy_to(ir3 + (340));
              tensorforge::intel_esimd::simd<float, 3> v846_data;
              v846_data.copy_from(r1 + (800));
              tensorforge::intel_esimd::simd<float, 3> v847_data;
              v847_data.copy_from(ir3 + (32));
              (v847_data + v846_data).copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 3> v849_data;
              v849_data.copy_from(r1 + (1632));
              tensorforge::intel_esimd::simd<float, 3> v850_data;
              v850_data.copy_from(ir3 + (96));
              (v850_data + v849_data).copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 3> v852_data;
              v852_data.copy_from(r1 + (2464));
              tensorforge::intel_esimd::simd<float, 3> v853_data;
              v853_data.copy_from(ir3 + (160));
              (v853_data + v852_data).copy_to(ir3 + (160));
              tensorforge::intel_esimd::simd<float, 3> v855_data;
              v855_data.copy_from(r1 + (3296));
              tensorforge::intel_esimd::simd<float, 3> v856_data;
              v856_data.copy_from(ir3 + (224));
              (v856_data + v855_data).copy_to(ir3 + (224));
              tensorforge::intel_esimd::simd<float, 3> v858_data;
              v858_data.copy_from(r1 + (4128));
              tensorforge::intel_esimd::simd<float, 3> v859_data;
              v859_data.copy_from(ir3 + (288));
              (v859_data + v858_data).copy_to(ir3 + (288));
              tensorforge::intel_esimd::simd<float, 3> v861_data;
              v861_data.copy_from(r1 + (4960));
              tensorforge::intel_esimd::simd<float, 3> v862_data;
              v862_data.copy_from(ir3 + (352));
              (v862_data + v861_data).copy_to(ir3 + (352));
              #pragma unroll
              for (int32_t v864_n1 = 0; v864_n1 < 1; ++v864_n1) {
                int32_t v868_a = 20 + (v864_n1 * 64);
                #pragma unroll
                for (int32_t v865_n2 = 0; v865_n2 < 6; ++v865_n2) {
                  int32_t v867_a = v865_n2 * 64;
                  tensorforge::intel_esimd::simd<float, 12> v870_data;
                  v870_data.copy_from(ir3 + ((v868_a + v867_a)));
                  tensorforge::intel_esimd::simd<float, 12> v875_data;
                  v875_data.copy_from(r2 + ((v868_a + v867_a)));
                  (v875_data + v870_data).copy_to(r3 + ((v868_a + v867_a)));
                }
              }
              #pragma unroll
              for (int32_t v881_n1 = 0; v881_n1 < 1; ++v881_n1) {
                int32_t v885_a = 32 + (v881_n1 * 64);
                #pragma unroll
                for (int32_t v882_n2 = 0; v882_n2 < 6; ++v882_n2) {
                  int32_t v884_a = v882_n2 * 64;
                  tensorforge::intel_esimd::simd<float, 3> v887_data;
                  v887_data.copy_from(ir3 + ((v885_a + v884_a)));
                  tensorforge::intel_esimd::simd<float, 3> v892_data;
                  v892_data.copy_from(r2 + ((v885_a + v884_a)));
                  (v892_data + v887_data).copy_to(r3 + ((v885_a + v884_a)));
                }
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v898_i1 = 0; v898_i1 < 1; ++v898_i1) {
                int32_t v902_a = 20 + (v898_i1 * 64);
                int32_t v911_a = 20_i32 + ((v898_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v899_i2 = 0; v899_i2 < 6; ++v899_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v904_data;
                  v904_data.copy_from(r3 + ((v902_a + (v899_i2 * 64))));
                  v904_data.copy_to(glb_m2 + ((v911_a + (v899_i2 * 832))));
                }
              }
              #pragma unroll
              for (int32_t v913_i1 = 0; v913_i1 < 1; ++v913_i1) {
                int32_t v917_a = 32 + (v913_i1 * 64);
                int32_t v926_a = 32_i32 + ((v913_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v914_i2 = 0; v914_i2 < 6; ++v914_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v919_data;
                  v919_data.copy_from(r3 + ((v917_a + (v914_i2 * 64))));
                  v919_data.copy_to(glb_m2 + ((v926_a + (v914_i2 * 832))));
                }
              }
            }
          }
        }
      });
    }
  });
}


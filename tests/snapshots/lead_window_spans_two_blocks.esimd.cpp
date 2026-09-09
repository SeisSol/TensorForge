// === base name ===
kernel_0ed39c5001b6158f

// === header ===
void launcher_kernel_0ed39c5001b6158f(const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_0ed39c5001b6158f(const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0ed39c5001b6158f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0ed39c5001b6158f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[832]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v8_i0 = 0; v8_i0 < 2; ++v8_i0) {
                int32_t v10_lead = v8_i0 * 32;
                #pragma unroll
                for (int32_t v9_i1 = 0; v9_i1 < 13; ++v9_i1) {
                  int32_t v13_a = v10_lead + (v9_i1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v14_data;
                  v14_data.copy_from(glb_m0 + (v13_a));
                  v14_data.copy_to(r0 + (v13_a));
                }
              }
              float r2[384]{};
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 1; ++v19_i1) {
                int32_t v27_a = 20_i32 + ((v19_i1 + 12) * 64);
                int32_t v32_a = 20 + (v19_i1 * 64);
                #pragma unroll
                for (int32_t v20_i2 = 0; v20_i2 < 6; ++v20_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v29_data;
                  v29_data.copy_from(glb_m2 + ((v27_a + (v20_i2 * 832))));
                  v29_data.copy_to(r2 + ((v32_a + (v20_i2 * 64))));
                }
              }
              #pragma unroll
              for (int32_t v34_i1 = 0; v34_i1 < 1; ++v34_i1) {
                int32_t v42_a = 32_i32 + ((v34_i1 + 12) * 64);
                int32_t v47_a = 32 + (v34_i1 * 64);
                #pragma unroll
                for (int32_t v35_i2 = 0; v35_i2 < 6; ++v35_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v44_data;
                  v44_data.copy_from(glb_m2 + ((v42_a + (v35_i2 * 832))));
                  v44_data.copy_to(r2 + ((v47_a + (v35_i2 * 64))));
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[4992]{};
              // r1 = +(r0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(r0 + (0));
              float v51_data = glb_m1[0];
              tensorforge::intel_esimd::simd<float, 32> v53_data;
              v53_data.copy_from(r1 + (0));
              (v53_data + (v50_data * v51_data)).copy_to(r1 + (0));
              float v56_data = glb_m1[1];
              tensorforge::intel_esimd::simd<float, 32> v58_data;
              v58_data.copy_from(r1 + (832));
              (v58_data + (v50_data * v56_data)).copy_to(r1 + (832));
              float v61_data = glb_m1[2];
              tensorforge::intel_esimd::simd<float, 32> v63_data;
              v63_data.copy_from(r1 + (1664));
              (v63_data + (v50_data * v61_data)).copy_to(r1 + (1664));
              float v66_data = glb_m1[3];
              tensorforge::intel_esimd::simd<float, 32> v68_data;
              v68_data.copy_from(r1 + (2496));
              (v68_data + (v50_data * v66_data)).copy_to(r1 + (2496));
              float v71_data = glb_m1[4];
              tensorforge::intel_esimd::simd<float, 32> v73_data;
              v73_data.copy_from(r1 + (3328));
              (v73_data + (v50_data * v71_data)).copy_to(r1 + (3328));
              float v76_data = glb_m1[5];
              tensorforge::intel_esimd::simd<float, 32> v78_data;
              v78_data.copy_from(r1 + (4160));
              (v78_data + (v50_data * v76_data)).copy_to(r1 + (4160));
              tensorforge::intel_esimd::simd<float, 32> v80_data;
              v80_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 32> v83_data;
              v83_data.copy_from(r1 + (64));
              (v83_data + (v80_data * v51_data)).copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 32> v88_data;
              v88_data.copy_from(r1 + (896));
              (v88_data + (v80_data * v56_data)).copy_to(r1 + (896));
              tensorforge::intel_esimd::simd<float, 32> v93_data;
              v93_data.copy_from(r1 + (1728));
              (v93_data + (v80_data * v61_data)).copy_to(r1 + (1728));
              tensorforge::intel_esimd::simd<float, 32> v98_data;
              v98_data.copy_from(r1 + (2560));
              (v98_data + (v80_data * v66_data)).copy_to(r1 + (2560));
              tensorforge::intel_esimd::simd<float, 32> v103_data;
              v103_data.copy_from(r1 + (3392));
              (v103_data + (v80_data * v71_data)).copy_to(r1 + (3392));
              tensorforge::intel_esimd::simd<float, 32> v108_data;
              v108_data.copy_from(r1 + (4224));
              (v108_data + (v80_data * v76_data)).copy_to(r1 + (4224));
              tensorforge::intel_esimd::simd<float, 32> v110_data;
              v110_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 32> v113_data;
              v113_data.copy_from(r1 + (128));
              (v113_data + (v110_data * v51_data)).copy_to(r1 + (128));
              tensorforge::intel_esimd::simd<float, 32> v118_data;
              v118_data.copy_from(r1 + (960));
              (v118_data + (v110_data * v56_data)).copy_to(r1 + (960));
              tensorforge::intel_esimd::simd<float, 32> v123_data;
              v123_data.copy_from(r1 + (1792));
              (v123_data + (v110_data * v61_data)).copy_to(r1 + (1792));
              tensorforge::intel_esimd::simd<float, 32> v128_data;
              v128_data.copy_from(r1 + (2624));
              (v128_data + (v110_data * v66_data)).copy_to(r1 + (2624));
              tensorforge::intel_esimd::simd<float, 32> v133_data;
              v133_data.copy_from(r1 + (3456));
              (v133_data + (v110_data * v71_data)).copy_to(r1 + (3456));
              tensorforge::intel_esimd::simd<float, 32> v138_data;
              v138_data.copy_from(r1 + (4288));
              (v138_data + (v110_data * v76_data)).copy_to(r1 + (4288));
              tensorforge::intel_esimd::simd<float, 32> v140_data;
              v140_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 32> v143_data;
              v143_data.copy_from(r1 + (192));
              (v143_data + (v140_data * v51_data)).copy_to(r1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v148_data;
              v148_data.copy_from(r1 + (1024));
              (v148_data + (v140_data * v56_data)).copy_to(r1 + (1024));
              tensorforge::intel_esimd::simd<float, 32> v153_data;
              v153_data.copy_from(r1 + (1856));
              (v153_data + (v140_data * v61_data)).copy_to(r1 + (1856));
              tensorforge::intel_esimd::simd<float, 32> v158_data;
              v158_data.copy_from(r1 + (2688));
              (v158_data + (v140_data * v66_data)).copy_to(r1 + (2688));
              tensorforge::intel_esimd::simd<float, 32> v163_data;
              v163_data.copy_from(r1 + (3520));
              (v163_data + (v140_data * v71_data)).copy_to(r1 + (3520));
              tensorforge::intel_esimd::simd<float, 32> v168_data;
              v168_data.copy_from(r1 + (4352));
              (v168_data + (v140_data * v76_data)).copy_to(r1 + (4352));
              tensorforge::intel_esimd::simd<float, 32> v170_data;
              v170_data.copy_from(r0 + (256));
              tensorforge::intel_esimd::simd<float, 32> v173_data;
              v173_data.copy_from(r1 + (256));
              (v173_data + (v170_data * v51_data)).copy_to(r1 + (256));
              tensorforge::intel_esimd::simd<float, 32> v178_data;
              v178_data.copy_from(r1 + (1088));
              (v178_data + (v170_data * v56_data)).copy_to(r1 + (1088));
              tensorforge::intel_esimd::simd<float, 32> v183_data;
              v183_data.copy_from(r1 + (1920));
              (v183_data + (v170_data * v61_data)).copy_to(r1 + (1920));
              tensorforge::intel_esimd::simd<float, 32> v188_data;
              v188_data.copy_from(r1 + (2752));
              (v188_data + (v170_data * v66_data)).copy_to(r1 + (2752));
              tensorforge::intel_esimd::simd<float, 32> v193_data;
              v193_data.copy_from(r1 + (3584));
              (v193_data + (v170_data * v71_data)).copy_to(r1 + (3584));
              tensorforge::intel_esimd::simd<float, 32> v198_data;
              v198_data.copy_from(r1 + (4416));
              (v198_data + (v170_data * v76_data)).copy_to(r1 + (4416));
              tensorforge::intel_esimd::simd<float, 32> v200_data;
              v200_data.copy_from(r0 + (320));
              tensorforge::intel_esimd::simd<float, 32> v203_data;
              v203_data.copy_from(r1 + (320));
              (v203_data + (v200_data * v51_data)).copy_to(r1 + (320));
              tensorforge::intel_esimd::simd<float, 32> v208_data;
              v208_data.copy_from(r1 + (1152));
              (v208_data + (v200_data * v56_data)).copy_to(r1 + (1152));
              tensorforge::intel_esimd::simd<float, 32> v213_data;
              v213_data.copy_from(r1 + (1984));
              (v213_data + (v200_data * v61_data)).copy_to(r1 + (1984));
              tensorforge::intel_esimd::simd<float, 32> v218_data;
              v218_data.copy_from(r1 + (2816));
              (v218_data + (v200_data * v66_data)).copy_to(r1 + (2816));
              tensorforge::intel_esimd::simd<float, 32> v223_data;
              v223_data.copy_from(r1 + (3648));
              (v223_data + (v200_data * v71_data)).copy_to(r1 + (3648));
              tensorforge::intel_esimd::simd<float, 32> v228_data;
              v228_data.copy_from(r1 + (4480));
              (v228_data + (v200_data * v76_data)).copy_to(r1 + (4480));
              tensorforge::intel_esimd::simd<float, 32> v230_data;
              v230_data.copy_from(r0 + (384));
              tensorforge::intel_esimd::simd<float, 32> v233_data;
              v233_data.copy_from(r1 + (384));
              (v233_data + (v230_data * v51_data)).copy_to(r1 + (384));
              tensorforge::intel_esimd::simd<float, 32> v238_data;
              v238_data.copy_from(r1 + (1216));
              (v238_data + (v230_data * v56_data)).copy_to(r1 + (1216));
              tensorforge::intel_esimd::simd<float, 32> v243_data;
              v243_data.copy_from(r1 + (2048));
              (v243_data + (v230_data * v61_data)).copy_to(r1 + (2048));
              tensorforge::intel_esimd::simd<float, 32> v248_data;
              v248_data.copy_from(r1 + (2880));
              (v248_data + (v230_data * v66_data)).copy_to(r1 + (2880));
              tensorforge::intel_esimd::simd<float, 32> v253_data;
              v253_data.copy_from(r1 + (3712));
              (v253_data + (v230_data * v71_data)).copy_to(r1 + (3712));
              tensorforge::intel_esimd::simd<float, 32> v258_data;
              v258_data.copy_from(r1 + (4544));
              (v258_data + (v230_data * v76_data)).copy_to(r1 + (4544));
              tensorforge::intel_esimd::simd<float, 32> v260_data;
              v260_data.copy_from(r0 + (448));
              tensorforge::intel_esimd::simd<float, 32> v263_data;
              v263_data.copy_from(r1 + (448));
              (v263_data + (v260_data * v51_data)).copy_to(r1 + (448));
              tensorforge::intel_esimd::simd<float, 32> v268_data;
              v268_data.copy_from(r1 + (1280));
              (v268_data + (v260_data * v56_data)).copy_to(r1 + (1280));
              tensorforge::intel_esimd::simd<float, 32> v273_data;
              v273_data.copy_from(r1 + (2112));
              (v273_data + (v260_data * v61_data)).copy_to(r1 + (2112));
              tensorforge::intel_esimd::simd<float, 32> v278_data;
              v278_data.copy_from(r1 + (2944));
              (v278_data + (v260_data * v66_data)).copy_to(r1 + (2944));
              tensorforge::intel_esimd::simd<float, 32> v283_data;
              v283_data.copy_from(r1 + (3776));
              (v283_data + (v260_data * v71_data)).copy_to(r1 + (3776));
              tensorforge::intel_esimd::simd<float, 32> v288_data;
              v288_data.copy_from(r1 + (4608));
              (v288_data + (v260_data * v76_data)).copy_to(r1 + (4608));
              tensorforge::intel_esimd::simd<float, 32> v290_data;
              v290_data.copy_from(r0 + (512));
              tensorforge::intel_esimd::simd<float, 32> v293_data;
              v293_data.copy_from(r1 + (512));
              (v293_data + (v290_data * v51_data)).copy_to(r1 + (512));
              tensorforge::intel_esimd::simd<float, 32> v298_data;
              v298_data.copy_from(r1 + (1344));
              (v298_data + (v290_data * v56_data)).copy_to(r1 + (1344));
              tensorforge::intel_esimd::simd<float, 32> v303_data;
              v303_data.copy_from(r1 + (2176));
              (v303_data + (v290_data * v61_data)).copy_to(r1 + (2176));
              tensorforge::intel_esimd::simd<float, 32> v308_data;
              v308_data.copy_from(r1 + (3008));
              (v308_data + (v290_data * v66_data)).copy_to(r1 + (3008));
              tensorforge::intel_esimd::simd<float, 32> v313_data;
              v313_data.copy_from(r1 + (3840));
              (v313_data + (v290_data * v71_data)).copy_to(r1 + (3840));
              tensorforge::intel_esimd::simd<float, 32> v318_data;
              v318_data.copy_from(r1 + (4672));
              (v318_data + (v290_data * v76_data)).copy_to(r1 + (4672));
              tensorforge::intel_esimd::simd<float, 32> v320_data;
              v320_data.copy_from(r0 + (576));
              tensorforge::intel_esimd::simd<float, 32> v323_data;
              v323_data.copy_from(r1 + (576));
              (v323_data + (v320_data * v51_data)).copy_to(r1 + (576));
              tensorforge::intel_esimd::simd<float, 32> v328_data;
              v328_data.copy_from(r1 + (1408));
              (v328_data + (v320_data * v56_data)).copy_to(r1 + (1408));
              tensorforge::intel_esimd::simd<float, 32> v333_data;
              v333_data.copy_from(r1 + (2240));
              (v333_data + (v320_data * v61_data)).copy_to(r1 + (2240));
              tensorforge::intel_esimd::simd<float, 32> v338_data;
              v338_data.copy_from(r1 + (3072));
              (v338_data + (v320_data * v66_data)).copy_to(r1 + (3072));
              tensorforge::intel_esimd::simd<float, 32> v343_data;
              v343_data.copy_from(r1 + (3904));
              (v343_data + (v320_data * v71_data)).copy_to(r1 + (3904));
              tensorforge::intel_esimd::simd<float, 32> v348_data;
              v348_data.copy_from(r1 + (4736));
              (v348_data + (v320_data * v76_data)).copy_to(r1 + (4736));
              tensorforge::intel_esimd::simd<float, 32> v350_data;
              v350_data.copy_from(r0 + (640));
              tensorforge::intel_esimd::simd<float, 32> v353_data;
              v353_data.copy_from(r1 + (640));
              (v353_data + (v350_data * v51_data)).copy_to(r1 + (640));
              tensorforge::intel_esimd::simd<float, 32> v358_data;
              v358_data.copy_from(r1 + (1472));
              (v358_data + (v350_data * v56_data)).copy_to(r1 + (1472));
              tensorforge::intel_esimd::simd<float, 32> v363_data;
              v363_data.copy_from(r1 + (2304));
              (v363_data + (v350_data * v61_data)).copy_to(r1 + (2304));
              tensorforge::intel_esimd::simd<float, 32> v368_data;
              v368_data.copy_from(r1 + (3136));
              (v368_data + (v350_data * v66_data)).copy_to(r1 + (3136));
              tensorforge::intel_esimd::simd<float, 32> v373_data;
              v373_data.copy_from(r1 + (3968));
              (v373_data + (v350_data * v71_data)).copy_to(r1 + (3968));
              tensorforge::intel_esimd::simd<float, 32> v378_data;
              v378_data.copy_from(r1 + (4800));
              (v378_data + (v350_data * v76_data)).copy_to(r1 + (4800));
              tensorforge::intel_esimd::simd<float, 32> v380_data;
              v380_data.copy_from(r0 + (704));
              tensorforge::intel_esimd::simd<float, 32> v383_data;
              v383_data.copy_from(r1 + (704));
              (v383_data + (v380_data * v51_data)).copy_to(r1 + (704));
              tensorforge::intel_esimd::simd<float, 32> v388_data;
              v388_data.copy_from(r1 + (1536));
              (v388_data + (v380_data * v56_data)).copy_to(r1 + (1536));
              tensorforge::intel_esimd::simd<float, 32> v393_data;
              v393_data.copy_from(r1 + (2368));
              (v393_data + (v380_data * v61_data)).copy_to(r1 + (2368));
              tensorforge::intel_esimd::simd<float, 32> v398_data;
              v398_data.copy_from(r1 + (3200));
              (v398_data + (v380_data * v66_data)).copy_to(r1 + (3200));
              tensorforge::intel_esimd::simd<float, 32> v403_data;
              v403_data.copy_from(r1 + (4032));
              (v403_data + (v380_data * v71_data)).copy_to(r1 + (4032));
              tensorforge::intel_esimd::simd<float, 32> v408_data;
              v408_data.copy_from(r1 + (4864));
              (v408_data + (v380_data * v76_data)).copy_to(r1 + (4864));
              tensorforge::intel_esimd::simd<float, 32> v410_data;
              v410_data.copy_from(r0 + (768));
              tensorforge::intel_esimd::simd<float, 32> v413_data;
              v413_data.copy_from(r1 + (768));
              (v413_data + (v410_data * v51_data)).copy_to(r1 + (768));
              tensorforge::intel_esimd::simd<float, 32> v418_data;
              v418_data.copy_from(r1 + (1600));
              (v418_data + (v410_data * v56_data)).copy_to(r1 + (1600));
              tensorforge::intel_esimd::simd<float, 32> v423_data;
              v423_data.copy_from(r1 + (2432));
              (v423_data + (v410_data * v61_data)).copy_to(r1 + (2432));
              tensorforge::intel_esimd::simd<float, 32> v428_data;
              v428_data.copy_from(r1 + (3264));
              (v428_data + (v410_data * v66_data)).copy_to(r1 + (3264));
              tensorforge::intel_esimd::simd<float, 32> v433_data;
              v433_data.copy_from(r1 + (4096));
              (v433_data + (v410_data * v71_data)).copy_to(r1 + (4096));
              tensorforge::intel_esimd::simd<float, 32> v438_data;
              v438_data.copy_from(r1 + (4928));
              (v438_data + (v410_data * v76_data)).copy_to(r1 + (4928));
              tensorforge::intel_esimd::simd<float, 32> v440_data;
              v440_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 32> v443_data;
              v443_data.copy_from(r1 + (32));
              (v443_data + (v440_data * v51_data)).copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 32> v448_data;
              v448_data.copy_from(r1 + (864));
              (v448_data + (v440_data * v56_data)).copy_to(r1 + (864));
              tensorforge::intel_esimd::simd<float, 32> v453_data;
              v453_data.copy_from(r1 + (1696));
              (v453_data + (v440_data * v61_data)).copy_to(r1 + (1696));
              tensorforge::intel_esimd::simd<float, 32> v458_data;
              v458_data.copy_from(r1 + (2528));
              (v458_data + (v440_data * v66_data)).copy_to(r1 + (2528));
              tensorforge::intel_esimd::simd<float, 32> v463_data;
              v463_data.copy_from(r1 + (3360));
              (v463_data + (v440_data * v71_data)).copy_to(r1 + (3360));
              tensorforge::intel_esimd::simd<float, 32> v468_data;
              v468_data.copy_from(r1 + (4192));
              (v468_data + (v440_data * v76_data)).copy_to(r1 + (4192));
              tensorforge::intel_esimd::simd<float, 32> v470_data;
              v470_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 32> v473_data;
              v473_data.copy_from(r1 + (96));
              (v473_data + (v470_data * v51_data)).copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 32> v478_data;
              v478_data.copy_from(r1 + (928));
              (v478_data + (v470_data * v56_data)).copy_to(r1 + (928));
              tensorforge::intel_esimd::simd<float, 32> v483_data;
              v483_data.copy_from(r1 + (1760));
              (v483_data + (v470_data * v61_data)).copy_to(r1 + (1760));
              tensorforge::intel_esimd::simd<float, 32> v488_data;
              v488_data.copy_from(r1 + (2592));
              (v488_data + (v470_data * v66_data)).copy_to(r1 + (2592));
              tensorforge::intel_esimd::simd<float, 32> v493_data;
              v493_data.copy_from(r1 + (3424));
              (v493_data + (v470_data * v71_data)).copy_to(r1 + (3424));
              tensorforge::intel_esimd::simd<float, 32> v498_data;
              v498_data.copy_from(r1 + (4256));
              (v498_data + (v470_data * v76_data)).copy_to(r1 + (4256));
              tensorforge::intel_esimd::simd<float, 32> v500_data;
              v500_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 32> v503_data;
              v503_data.copy_from(r1 + (160));
              (v503_data + (v500_data * v51_data)).copy_to(r1 + (160));
              tensorforge::intel_esimd::simd<float, 32> v508_data;
              v508_data.copy_from(r1 + (992));
              (v508_data + (v500_data * v56_data)).copy_to(r1 + (992));
              tensorforge::intel_esimd::simd<float, 32> v513_data;
              v513_data.copy_from(r1 + (1824));
              (v513_data + (v500_data * v61_data)).copy_to(r1 + (1824));
              tensorforge::intel_esimd::simd<float, 32> v518_data;
              v518_data.copy_from(r1 + (2656));
              (v518_data + (v500_data * v66_data)).copy_to(r1 + (2656));
              tensorforge::intel_esimd::simd<float, 32> v523_data;
              v523_data.copy_from(r1 + (3488));
              (v523_data + (v500_data * v71_data)).copy_to(r1 + (3488));
              tensorforge::intel_esimd::simd<float, 32> v528_data;
              v528_data.copy_from(r1 + (4320));
              (v528_data + (v500_data * v76_data)).copy_to(r1 + (4320));
              tensorforge::intel_esimd::simd<float, 32> v530_data;
              v530_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 32> v533_data;
              v533_data.copy_from(r1 + (224));
              (v533_data + (v530_data * v51_data)).copy_to(r1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v538_data;
              v538_data.copy_from(r1 + (1056));
              (v538_data + (v530_data * v56_data)).copy_to(r1 + (1056));
              tensorforge::intel_esimd::simd<float, 32> v543_data;
              v543_data.copy_from(r1 + (1888));
              (v543_data + (v530_data * v61_data)).copy_to(r1 + (1888));
              tensorforge::intel_esimd::simd<float, 32> v548_data;
              v548_data.copy_from(r1 + (2720));
              (v548_data + (v530_data * v66_data)).copy_to(r1 + (2720));
              tensorforge::intel_esimd::simd<float, 32> v553_data;
              v553_data.copy_from(r1 + (3552));
              (v553_data + (v530_data * v71_data)).copy_to(r1 + (3552));
              tensorforge::intel_esimd::simd<float, 32> v558_data;
              v558_data.copy_from(r1 + (4384));
              (v558_data + (v530_data * v76_data)).copy_to(r1 + (4384));
              tensorforge::intel_esimd::simd<float, 32> v560_data;
              v560_data.copy_from(r0 + (288));
              tensorforge::intel_esimd::simd<float, 32> v563_data;
              v563_data.copy_from(r1 + (288));
              (v563_data + (v560_data * v51_data)).copy_to(r1 + (288));
              tensorforge::intel_esimd::simd<float, 32> v568_data;
              v568_data.copy_from(r1 + (1120));
              (v568_data + (v560_data * v56_data)).copy_to(r1 + (1120));
              tensorforge::intel_esimd::simd<float, 32> v573_data;
              v573_data.copy_from(r1 + (1952));
              (v573_data + (v560_data * v61_data)).copy_to(r1 + (1952));
              tensorforge::intel_esimd::simd<float, 32> v578_data;
              v578_data.copy_from(r1 + (2784));
              (v578_data + (v560_data * v66_data)).copy_to(r1 + (2784));
              tensorforge::intel_esimd::simd<float, 32> v583_data;
              v583_data.copy_from(r1 + (3616));
              (v583_data + (v560_data * v71_data)).copy_to(r1 + (3616));
              tensorforge::intel_esimd::simd<float, 32> v588_data;
              v588_data.copy_from(r1 + (4448));
              (v588_data + (v560_data * v76_data)).copy_to(r1 + (4448));
              tensorforge::intel_esimd::simd<float, 32> v590_data;
              v590_data.copy_from(r0 + (352));
              tensorforge::intel_esimd::simd<float, 32> v593_data;
              v593_data.copy_from(r1 + (352));
              (v593_data + (v590_data * v51_data)).copy_to(r1 + (352));
              tensorforge::intel_esimd::simd<float, 32> v598_data;
              v598_data.copy_from(r1 + (1184));
              (v598_data + (v590_data * v56_data)).copy_to(r1 + (1184));
              tensorforge::intel_esimd::simd<float, 32> v603_data;
              v603_data.copy_from(r1 + (2016));
              (v603_data + (v590_data * v61_data)).copy_to(r1 + (2016));
              tensorforge::intel_esimd::simd<float, 32> v608_data;
              v608_data.copy_from(r1 + (2848));
              (v608_data + (v590_data * v66_data)).copy_to(r1 + (2848));
              tensorforge::intel_esimd::simd<float, 32> v613_data;
              v613_data.copy_from(r1 + (3680));
              (v613_data + (v590_data * v71_data)).copy_to(r1 + (3680));
              tensorforge::intel_esimd::simd<float, 32> v618_data;
              v618_data.copy_from(r1 + (4512));
              (v618_data + (v590_data * v76_data)).copy_to(r1 + (4512));
              tensorforge::intel_esimd::simd<float, 32> v620_data;
              v620_data.copy_from(r0 + (416));
              tensorforge::intel_esimd::simd<float, 32> v623_data;
              v623_data.copy_from(r1 + (416));
              (v623_data + (v620_data * v51_data)).copy_to(r1 + (416));
              tensorforge::intel_esimd::simd<float, 32> v628_data;
              v628_data.copy_from(r1 + (1248));
              (v628_data + (v620_data * v56_data)).copy_to(r1 + (1248));
              tensorforge::intel_esimd::simd<float, 32> v633_data;
              v633_data.copy_from(r1 + (2080));
              (v633_data + (v620_data * v61_data)).copy_to(r1 + (2080));
              tensorforge::intel_esimd::simd<float, 32> v638_data;
              v638_data.copy_from(r1 + (2912));
              (v638_data + (v620_data * v66_data)).copy_to(r1 + (2912));
              tensorforge::intel_esimd::simd<float, 32> v643_data;
              v643_data.copy_from(r1 + (3744));
              (v643_data + (v620_data * v71_data)).copy_to(r1 + (3744));
              tensorforge::intel_esimd::simd<float, 32> v648_data;
              v648_data.copy_from(r1 + (4576));
              (v648_data + (v620_data * v76_data)).copy_to(r1 + (4576));
              tensorforge::intel_esimd::simd<float, 32> v650_data;
              v650_data.copy_from(r0 + (480));
              tensorforge::intel_esimd::simd<float, 32> v653_data;
              v653_data.copy_from(r1 + (480));
              (v653_data + (v650_data * v51_data)).copy_to(r1 + (480));
              tensorforge::intel_esimd::simd<float, 32> v658_data;
              v658_data.copy_from(r1 + (1312));
              (v658_data + (v650_data * v56_data)).copy_to(r1 + (1312));
              tensorforge::intel_esimd::simd<float, 32> v663_data;
              v663_data.copy_from(r1 + (2144));
              (v663_data + (v650_data * v61_data)).copy_to(r1 + (2144));
              tensorforge::intel_esimd::simd<float, 32> v668_data;
              v668_data.copy_from(r1 + (2976));
              (v668_data + (v650_data * v66_data)).copy_to(r1 + (2976));
              tensorforge::intel_esimd::simd<float, 32> v673_data;
              v673_data.copy_from(r1 + (3808));
              (v673_data + (v650_data * v71_data)).copy_to(r1 + (3808));
              tensorforge::intel_esimd::simd<float, 32> v678_data;
              v678_data.copy_from(r1 + (4640));
              (v678_data + (v650_data * v76_data)).copy_to(r1 + (4640));
              tensorforge::intel_esimd::simd<float, 32> v680_data;
              v680_data.copy_from(r0 + (544));
              tensorforge::intel_esimd::simd<float, 32> v683_data;
              v683_data.copy_from(r1 + (544));
              (v683_data + (v680_data * v51_data)).copy_to(r1 + (544));
              tensorforge::intel_esimd::simd<float, 32> v688_data;
              v688_data.copy_from(r1 + (1376));
              (v688_data + (v680_data * v56_data)).copy_to(r1 + (1376));
              tensorforge::intel_esimd::simd<float, 32> v693_data;
              v693_data.copy_from(r1 + (2208));
              (v693_data + (v680_data * v61_data)).copy_to(r1 + (2208));
              tensorforge::intel_esimd::simd<float, 32> v698_data;
              v698_data.copy_from(r1 + (3040));
              (v698_data + (v680_data * v66_data)).copy_to(r1 + (3040));
              tensorforge::intel_esimd::simd<float, 32> v703_data;
              v703_data.copy_from(r1 + (3872));
              (v703_data + (v680_data * v71_data)).copy_to(r1 + (3872));
              tensorforge::intel_esimd::simd<float, 32> v708_data;
              v708_data.copy_from(r1 + (4704));
              (v708_data + (v680_data * v76_data)).copy_to(r1 + (4704));
              tensorforge::intel_esimd::simd<float, 32> v710_data;
              v710_data.copy_from(r0 + (608));
              tensorforge::intel_esimd::simd<float, 32> v713_data;
              v713_data.copy_from(r1 + (608));
              (v713_data + (v710_data * v51_data)).copy_to(r1 + (608));
              tensorforge::intel_esimd::simd<float, 32> v718_data;
              v718_data.copy_from(r1 + (1440));
              (v718_data + (v710_data * v56_data)).copy_to(r1 + (1440));
              tensorforge::intel_esimd::simd<float, 32> v723_data;
              v723_data.copy_from(r1 + (2272));
              (v723_data + (v710_data * v61_data)).copy_to(r1 + (2272));
              tensorforge::intel_esimd::simd<float, 32> v728_data;
              v728_data.copy_from(r1 + (3104));
              (v728_data + (v710_data * v66_data)).copy_to(r1 + (3104));
              tensorforge::intel_esimd::simd<float, 32> v733_data;
              v733_data.copy_from(r1 + (3936));
              (v733_data + (v710_data * v71_data)).copy_to(r1 + (3936));
              tensorforge::intel_esimd::simd<float, 32> v738_data;
              v738_data.copy_from(r1 + (4768));
              (v738_data + (v710_data * v76_data)).copy_to(r1 + (4768));
              tensorforge::intel_esimd::simd<float, 32> v740_data;
              v740_data.copy_from(r0 + (672));
              tensorforge::intel_esimd::simd<float, 32> v743_data;
              v743_data.copy_from(r1 + (672));
              (v743_data + (v740_data * v51_data)).copy_to(r1 + (672));
              tensorforge::intel_esimd::simd<float, 32> v748_data;
              v748_data.copy_from(r1 + (1504));
              (v748_data + (v740_data * v56_data)).copy_to(r1 + (1504));
              tensorforge::intel_esimd::simd<float, 32> v753_data;
              v753_data.copy_from(r1 + (2336));
              (v753_data + (v740_data * v61_data)).copy_to(r1 + (2336));
              tensorforge::intel_esimd::simd<float, 32> v758_data;
              v758_data.copy_from(r1 + (3168));
              (v758_data + (v740_data * v66_data)).copy_to(r1 + (3168));
              tensorforge::intel_esimd::simd<float, 32> v763_data;
              v763_data.copy_from(r1 + (4000));
              (v763_data + (v740_data * v71_data)).copy_to(r1 + (4000));
              tensorforge::intel_esimd::simd<float, 32> v768_data;
              v768_data.copy_from(r1 + (4832));
              (v768_data + (v740_data * v76_data)).copy_to(r1 + (4832));
              tensorforge::intel_esimd::simd<float, 32> v770_data;
              v770_data.copy_from(r0 + (736));
              tensorforge::intel_esimd::simd<float, 32> v773_data;
              v773_data.copy_from(r1 + (736));
              (v773_data + (v770_data * v51_data)).copy_to(r1 + (736));
              tensorforge::intel_esimd::simd<float, 32> v778_data;
              v778_data.copy_from(r1 + (1568));
              (v778_data + (v770_data * v56_data)).copy_to(r1 + (1568));
              tensorforge::intel_esimd::simd<float, 32> v783_data;
              v783_data.copy_from(r1 + (2400));
              (v783_data + (v770_data * v61_data)).copy_to(r1 + (2400));
              tensorforge::intel_esimd::simd<float, 32> v788_data;
              v788_data.copy_from(r1 + (3232));
              (v788_data + (v770_data * v66_data)).copy_to(r1 + (3232));
              tensorforge::intel_esimd::simd<float, 32> v793_data;
              v793_data.copy_from(r1 + (4064));
              (v793_data + (v770_data * v71_data)).copy_to(r1 + (4064));
              tensorforge::intel_esimd::simd<float, 32> v798_data;
              v798_data.copy_from(r1 + (4896));
              (v798_data + (v770_data * v76_data)).copy_to(r1 + (4896));
              tensorforge::intel_esimd::simd<float, 32> v800_data;
              v800_data.copy_from(r0 + (800));
              tensorforge::intel_esimd::simd<float, 32> v803_data;
              v803_data.copy_from(r1 + (800));
              (v803_data + (v800_data * v51_data)).copy_to(r1 + (800));
              tensorforge::intel_esimd::simd<float, 32> v808_data;
              v808_data.copy_from(r1 + (1632));
              (v808_data + (v800_data * v56_data)).copy_to(r1 + (1632));
              tensorforge::intel_esimd::simd<float, 32> v813_data;
              v813_data.copy_from(r1 + (2464));
              (v813_data + (v800_data * v61_data)).copy_to(r1 + (2464));
              tensorforge::intel_esimd::simd<float, 32> v818_data;
              v818_data.copy_from(r1 + (3296));
              (v818_data + (v800_data * v66_data)).copy_to(r1 + (3296));
              tensorforge::intel_esimd::simd<float, 32> v823_data;
              v823_data.copy_from(r1 + (4128));
              (v823_data + (v800_data * v71_data)).copy_to(r1 + (4128));
              tensorforge::intel_esimd::simd<float, 32> v828_data;
              v828_data.copy_from(r1 + (4960));
              (v828_data + (v800_data * v76_data)).copy_to(r1 + (4960));
              // wait(r2 = load{g>r}(glb_m2););
              float r3[384]{};
              // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(20, 35), (0, 1), (0, 6)] []
              float ir3[384]{};
              tensorforge::intel_esimd::simd<float, 12> v832_data;
              v832_data.copy_from(r1 + (788));
              tensorforge::intel_esimd::simd<float, 12> v833_data;
              v833_data.copy_from(ir3 + (20));
              (v833_data + v832_data).copy_to(ir3 + (20));
              tensorforge::intel_esimd::simd<float, 12> v835_data;
              v835_data.copy_from(r1 + (1620));
              tensorforge::intel_esimd::simd<float, 12> v836_data;
              v836_data.copy_from(ir3 + (84));
              (v836_data + v835_data).copy_to(ir3 + (84));
              tensorforge::intel_esimd::simd<float, 12> v838_data;
              v838_data.copy_from(r1 + (2452));
              tensorforge::intel_esimd::simd<float, 12> v839_data;
              v839_data.copy_from(ir3 + (148));
              (v839_data + v838_data).copy_to(ir3 + (148));
              tensorforge::intel_esimd::simd<float, 12> v841_data;
              v841_data.copy_from(r1 + (3284));
              tensorforge::intel_esimd::simd<float, 12> v842_data;
              v842_data.copy_from(ir3 + (212));
              (v842_data + v841_data).copy_to(ir3 + (212));
              tensorforge::intel_esimd::simd<float, 12> v844_data;
              v844_data.copy_from(r1 + (4116));
              tensorforge::intel_esimd::simd<float, 12> v845_data;
              v845_data.copy_from(ir3 + (276));
              (v845_data + v844_data).copy_to(ir3 + (276));
              tensorforge::intel_esimd::simd<float, 12> v847_data;
              v847_data.copy_from(r1 + (4948));
              tensorforge::intel_esimd::simd<float, 12> v848_data;
              v848_data.copy_from(ir3 + (340));
              (v848_data + v847_data).copy_to(ir3 + (340));
              tensorforge::intel_esimd::simd<float, 3> v850_data;
              v850_data.copy_from(r1 + (800));
              tensorforge::intel_esimd::simd<float, 3> v851_data;
              v851_data.copy_from(ir3 + (32));
              (v851_data + v850_data).copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 3> v853_data;
              v853_data.copy_from(r1 + (1632));
              tensorforge::intel_esimd::simd<float, 3> v854_data;
              v854_data.copy_from(ir3 + (96));
              (v854_data + v853_data).copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 3> v856_data;
              v856_data.copy_from(r1 + (2464));
              tensorforge::intel_esimd::simd<float, 3> v857_data;
              v857_data.copy_from(ir3 + (160));
              (v857_data + v856_data).copy_to(ir3 + (160));
              tensorforge::intel_esimd::simd<float, 3> v859_data;
              v859_data.copy_from(r1 + (3296));
              tensorforge::intel_esimd::simd<float, 3> v860_data;
              v860_data.copy_from(ir3 + (224));
              (v860_data + v859_data).copy_to(ir3 + (224));
              tensorforge::intel_esimd::simd<float, 3> v862_data;
              v862_data.copy_from(r1 + (4128));
              tensorforge::intel_esimd::simd<float, 3> v863_data;
              v863_data.copy_from(ir3 + (288));
              (v863_data + v862_data).copy_to(ir3 + (288));
              tensorforge::intel_esimd::simd<float, 3> v865_data;
              v865_data.copy_from(r1 + (4960));
              tensorforge::intel_esimd::simd<float, 3> v866_data;
              v866_data.copy_from(ir3 + (352));
              (v866_data + v865_data).copy_to(ir3 + (352));
              #pragma unroll
              for (int32_t v868_n1 = 0; v868_n1 < 1; ++v868_n1) {
                int32_t v872_a = 20 + (v868_n1 * 64);
                #pragma unroll
                for (int32_t v869_n2 = 0; v869_n2 < 6; ++v869_n2) {
                  int32_t v871_a = v869_n2 * 64;
                  tensorforge::intel_esimd::simd<float, 12> v874_data;
                  v874_data.copy_from(ir3 + ((v872_a + v871_a)));
                  tensorforge::intel_esimd::simd<float, 12> v879_data;
                  v879_data.copy_from(r2 + ((v872_a + v871_a)));
                  (v879_data + v874_data).copy_to(r3 + ((v872_a + v871_a)));
                }
              }
              #pragma unroll
              for (int32_t v885_n1 = 0; v885_n1 < 1; ++v885_n1) {
                int32_t v889_a = 32 + (v885_n1 * 64);
                #pragma unroll
                for (int32_t v886_n2 = 0; v886_n2 < 6; ++v886_n2) {
                  int32_t v888_a = v886_n2 * 64;
                  tensorforge::intel_esimd::simd<float, 3> v891_data;
                  v891_data.copy_from(ir3 + ((v889_a + v888_a)));
                  tensorforge::intel_esimd::simd<float, 3> v896_data;
                  v896_data.copy_from(r2 + ((v889_a + v888_a)));
                  (v896_data + v891_data).copy_to(r3 + ((v889_a + v888_a)));
                }
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v902_i1 = 0; v902_i1 < 1; ++v902_i1) {
                int32_t v906_a = 20 + (v902_i1 * 64);
                int32_t v915_a = 20_i32 + ((v902_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v903_i2 = 0; v903_i2 < 6; ++v903_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v908_data;
                  v908_data.copy_from(r3 + ((v906_a + (v903_i2 * 64))));
                  v908_data.copy_to(glb_m2 + ((v915_a + (v903_i2 * 832))));
                }
              }
              #pragma unroll
              for (int32_t v917_i1 = 0; v917_i1 < 1; ++v917_i1) {
                int32_t v921_a = 32 + (v917_i1 * 64);
                int32_t v930_a = 32_i32 + ((v917_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v918_i2 = 0; v918_i2 < 6; ++v918_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v923_data;
                  v923_data.copy_from(r3 + ((v921_a + (v918_i2 * 64))));
                  v923_data.copy_to(glb_m2 + ((v930_a + (v918_i2 * 832))));
                }
              }
            }
          }
        }
      });
    }
  });
}


// === base name ===
kernel_69f2bb9311

// === header ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_69f2bb9311(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_69f2bb9311(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 35×4(35×4) {0..35}×{0..4} strided
        // m1 35×8(35×8) {0..35}×{0..8} strided
        // m2 8×4(8×4) {0..8}×{0..4} strided
        // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[32 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[32];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float r0[512]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
                int32_t v8_lead = v6_i0 * 32;
                #pragma unroll
                for (int32_t v7_i1 = 0; v7_i1 < 8; ++v7_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v12_data;
                  v12_data.copy_from(glb_m1 + ((v8_lead + (v7_i1 * 35))));
                  v12_data.copy_to(r0 + ((v8_lead + (v7_i1 * 64))));
                }
              }
              #pragma unroll
              for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
                tensorforge::intel_esimd::simd<float, 3> v22_data;
                v22_data.copy_from(glb_m1 + ((32_i32 + (v16_i1 * 35))));
                v22_data.copy_to(r0 + ((32 + (v16_i1 * 64))));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              s0[0 + 0 + 1 * item.get_local_id(0) + 0] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 0];
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[256]{};
              // r1 = +(r0 * s0) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir1[256]{};
              tensorforge::intel_esimd::simd<float, 32> v28_data;
              v28_data.copy_from(r0 + (0));
              float v29_data = s0[0];
              tensorforge::intel_esimd::simd<float, 32> v31_data;
              v31_data.copy_from(ir1 + (0));
              (v31_data + (v28_data * v29_data)).copy_to(ir1 + (0));
              float v34_data = s0[8];
              tensorforge::intel_esimd::simd<float, 32> v36_data;
              v36_data.copy_from(ir1 + (64));
              (v36_data + (v28_data * v34_data)).copy_to(ir1 + (64));
              float v39_data = s0[16];
              tensorforge::intel_esimd::simd<float, 32> v41_data;
              v41_data.copy_from(ir1 + (128));
              (v41_data + (v28_data * v39_data)).copy_to(ir1 + (128));
              float v44_data = s0[24];
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(ir1 + (192));
              (v46_data + (v28_data * v44_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v48_data;
              v48_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 3> v51_data;
              v51_data.copy_from(ir1 + (32));
              (v51_data + (v48_data * v29_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v56_data;
              v56_data.copy_from(ir1 + (96));
              (v56_data + (v48_data * v34_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v61_data;
              v61_data.copy_from(ir1 + (160));
              (v61_data + (v48_data * v39_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v66_data;
              v66_data.copy_from(ir1 + (224));
              (v66_data + (v48_data * v44_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v68_data;
              v68_data.copy_from(r0 + (64));
              float v69_data = s0[1];
              tensorforge::intel_esimd::simd<float, 32> v71_data;
              v71_data.copy_from(ir1 + (0));
              (v71_data + (v68_data * v69_data)).copy_to(ir1 + (0));
              float v74_data = s0[9];
              tensorforge::intel_esimd::simd<float, 32> v76_data;
              v76_data.copy_from(ir1 + (64));
              (v76_data + (v68_data * v74_data)).copy_to(ir1 + (64));
              float v79_data = s0[17];
              tensorforge::intel_esimd::simd<float, 32> v81_data;
              v81_data.copy_from(ir1 + (128));
              (v81_data + (v68_data * v79_data)).copy_to(ir1 + (128));
              float v84_data = s0[25];
              tensorforge::intel_esimd::simd<float, 32> v86_data;
              v86_data.copy_from(ir1 + (192));
              (v86_data + (v68_data * v84_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v88_data;
              v88_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 3> v91_data;
              v91_data.copy_from(ir1 + (32));
              (v91_data + (v88_data * v69_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v96_data;
              v96_data.copy_from(ir1 + (96));
              (v96_data + (v88_data * v74_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v101_data;
              v101_data.copy_from(ir1 + (160));
              (v101_data + (v88_data * v79_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v106_data;
              v106_data.copy_from(ir1 + (224));
              (v106_data + (v88_data * v84_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v108_data;
              v108_data.copy_from(r0 + (128));
              float v109_data = s0[2];
              tensorforge::intel_esimd::simd<float, 32> v111_data;
              v111_data.copy_from(ir1 + (0));
              (v111_data + (v108_data * v109_data)).copy_to(ir1 + (0));
              float v114_data = s0[10];
              tensorforge::intel_esimd::simd<float, 32> v116_data;
              v116_data.copy_from(ir1 + (64));
              (v116_data + (v108_data * v114_data)).copy_to(ir1 + (64));
              float v119_data = s0[18];
              tensorforge::intel_esimd::simd<float, 32> v121_data;
              v121_data.copy_from(ir1 + (128));
              (v121_data + (v108_data * v119_data)).copy_to(ir1 + (128));
              float v124_data = s0[26];
              tensorforge::intel_esimd::simd<float, 32> v126_data;
              v126_data.copy_from(ir1 + (192));
              (v126_data + (v108_data * v124_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v128_data;
              v128_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 3> v131_data;
              v131_data.copy_from(ir1 + (32));
              (v131_data + (v128_data * v109_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v136_data;
              v136_data.copy_from(ir1 + (96));
              (v136_data + (v128_data * v114_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v141_data;
              v141_data.copy_from(ir1 + (160));
              (v141_data + (v128_data * v119_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v146_data;
              v146_data.copy_from(ir1 + (224));
              (v146_data + (v128_data * v124_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v148_data;
              v148_data.copy_from(r0 + (192));
              float v149_data = s0[3];
              tensorforge::intel_esimd::simd<float, 32> v151_data;
              v151_data.copy_from(ir1 + (0));
              (v151_data + (v148_data * v149_data)).copy_to(ir1 + (0));
              float v154_data = s0[11];
              tensorforge::intel_esimd::simd<float, 32> v156_data;
              v156_data.copy_from(ir1 + (64));
              (v156_data + (v148_data * v154_data)).copy_to(ir1 + (64));
              float v159_data = s0[19];
              tensorforge::intel_esimd::simd<float, 32> v161_data;
              v161_data.copy_from(ir1 + (128));
              (v161_data + (v148_data * v159_data)).copy_to(ir1 + (128));
              float v164_data = s0[27];
              tensorforge::intel_esimd::simd<float, 32> v166_data;
              v166_data.copy_from(ir1 + (192));
              (v166_data + (v148_data * v164_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v168_data;
              v168_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 3> v171_data;
              v171_data.copy_from(ir1 + (32));
              (v171_data + (v168_data * v149_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v176_data;
              v176_data.copy_from(ir1 + (96));
              (v176_data + (v168_data * v154_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v181_data;
              v181_data.copy_from(ir1 + (160));
              (v181_data + (v168_data * v159_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v186_data;
              v186_data.copy_from(ir1 + (224));
              (v186_data + (v168_data * v164_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v188_data;
              v188_data.copy_from(r0 + (256));
              float v189_data = s0[4];
              tensorforge::intel_esimd::simd<float, 32> v191_data;
              v191_data.copy_from(ir1 + (0));
              (v191_data + (v188_data * v189_data)).copy_to(ir1 + (0));
              float v194_data = s0[12];
              tensorforge::intel_esimd::simd<float, 32> v196_data;
              v196_data.copy_from(ir1 + (64));
              (v196_data + (v188_data * v194_data)).copy_to(ir1 + (64));
              float v199_data = s0[20];
              tensorforge::intel_esimd::simd<float, 32> v201_data;
              v201_data.copy_from(ir1 + (128));
              (v201_data + (v188_data * v199_data)).copy_to(ir1 + (128));
              float v204_data = s0[28];
              tensorforge::intel_esimd::simd<float, 32> v206_data;
              v206_data.copy_from(ir1 + (192));
              (v206_data + (v188_data * v204_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v208_data;
              v208_data.copy_from(r0 + (288));
              tensorforge::intel_esimd::simd<float, 3> v211_data;
              v211_data.copy_from(ir1 + (32));
              (v211_data + (v208_data * v189_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v216_data;
              v216_data.copy_from(ir1 + (96));
              (v216_data + (v208_data * v194_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v221_data;
              v221_data.copy_from(ir1 + (160));
              (v221_data + (v208_data * v199_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v226_data;
              v226_data.copy_from(ir1 + (224));
              (v226_data + (v208_data * v204_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v228_data;
              v228_data.copy_from(r0 + (320));
              float v229_data = s0[5];
              tensorforge::intel_esimd::simd<float, 32> v231_data;
              v231_data.copy_from(ir1 + (0));
              (v231_data + (v228_data * v229_data)).copy_to(ir1 + (0));
              float v234_data = s0[13];
              tensorforge::intel_esimd::simd<float, 32> v236_data;
              v236_data.copy_from(ir1 + (64));
              (v236_data + (v228_data * v234_data)).copy_to(ir1 + (64));
              float v239_data = s0[21];
              tensorforge::intel_esimd::simd<float, 32> v241_data;
              v241_data.copy_from(ir1 + (128));
              (v241_data + (v228_data * v239_data)).copy_to(ir1 + (128));
              float v244_data = s0[29];
              tensorforge::intel_esimd::simd<float, 32> v246_data;
              v246_data.copy_from(ir1 + (192));
              (v246_data + (v228_data * v244_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v248_data;
              v248_data.copy_from(r0 + (352));
              tensorforge::intel_esimd::simd<float, 3> v251_data;
              v251_data.copy_from(ir1 + (32));
              (v251_data + (v248_data * v229_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v256_data;
              v256_data.copy_from(ir1 + (96));
              (v256_data + (v248_data * v234_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v261_data;
              v261_data.copy_from(ir1 + (160));
              (v261_data + (v248_data * v239_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v266_data;
              v266_data.copy_from(ir1 + (224));
              (v266_data + (v248_data * v244_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v268_data;
              v268_data.copy_from(r0 + (384));
              float v269_data = s0[6];
              tensorforge::intel_esimd::simd<float, 32> v271_data;
              v271_data.copy_from(ir1 + (0));
              (v271_data + (v268_data * v269_data)).copy_to(ir1 + (0));
              float v274_data = s0[14];
              tensorforge::intel_esimd::simd<float, 32> v276_data;
              v276_data.copy_from(ir1 + (64));
              (v276_data + (v268_data * v274_data)).copy_to(ir1 + (64));
              float v279_data = s0[22];
              tensorforge::intel_esimd::simd<float, 32> v281_data;
              v281_data.copy_from(ir1 + (128));
              (v281_data + (v268_data * v279_data)).copy_to(ir1 + (128));
              float v284_data = s0[30];
              tensorforge::intel_esimd::simd<float, 32> v286_data;
              v286_data.copy_from(ir1 + (192));
              (v286_data + (v268_data * v284_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v288_data;
              v288_data.copy_from(r0 + (416));
              tensorforge::intel_esimd::simd<float, 3> v291_data;
              v291_data.copy_from(ir1 + (32));
              (v291_data + (v288_data * v269_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v296_data;
              v296_data.copy_from(ir1 + (96));
              (v296_data + (v288_data * v274_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v301_data;
              v301_data.copy_from(ir1 + (160));
              (v301_data + (v288_data * v279_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v306_data;
              v306_data.copy_from(ir1 + (224));
              (v306_data + (v288_data * v284_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v308_data;
              v308_data.copy_from(r0 + (448));
              float v309_data = s0[7];
              tensorforge::intel_esimd::simd<float, 32> v311_data;
              v311_data.copy_from(ir1 + (0));
              (v311_data + (v308_data * v309_data)).copy_to(ir1 + (0));
              float v314_data = s0[15];
              tensorforge::intel_esimd::simd<float, 32> v316_data;
              v316_data.copy_from(ir1 + (64));
              (v316_data + (v308_data * v314_data)).copy_to(ir1 + (64));
              float v319_data = s0[23];
              tensorforge::intel_esimd::simd<float, 32> v321_data;
              v321_data.copy_from(ir1 + (128));
              (v321_data + (v308_data * v319_data)).copy_to(ir1 + (128));
              float v324_data = s0[31];
              tensorforge::intel_esimd::simd<float, 32> v326_data;
              v326_data.copy_from(ir1 + (192));
              (v326_data + (v308_data * v324_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v328_data;
              v328_data.copy_from(r0 + (480));
              tensorforge::intel_esimd::simd<float, 3> v331_data;
              v331_data.copy_from(ir1 + (32));
              (v331_data + (v328_data * v309_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v336_data;
              v336_data.copy_from(ir1 + (96));
              (v336_data + (v328_data * v314_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v341_data;
              v341_data.copy_from(ir1 + (160));
              (v341_data + (v328_data * v319_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v346_data;
              v346_data.copy_from(ir1 + (224));
              (v346_data + (v328_data * v324_data)).copy_to(ir1 + (224));
              #pragma unroll
              for (int32_t v348_n0 = 0; v348_n0 < 1; ++v348_n0) {
                int32_t v350_a = v348_n0 * 32;
                #pragma unroll
                for (int32_t v349_n1 = 0; v349_n1 < 4; ++v349_n1) {
                  int32_t v352_a = v350_a + (v349_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v353_data;
                  v353_data.copy_from(ir1 + (v352_a));
                  v353_data.copy_to(r1 + (v352_a));
                }
              }
              #pragma unroll
              for (int32_t v357_n1 = 0; v357_n1 < 4; ++v357_n1) {
                int32_t v359_a = 32 + (v357_n1 * 64);
                tensorforge::intel_esimd::simd<float, 3> v360_data;
                v360_data.copy_from(ir1 + (v359_a));
                v360_data.copy_to(r1 + (v359_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v363_i0 = 0; v363_i0 < 1; ++v363_i0) {
                int32_t v365_a = v363_i0 * 32;
                #pragma unroll
                for (int32_t v364_i1 = 0; v364_i1 < 4; ++v364_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v368_data;
                  v368_data.copy_from(r1 + ((v365_a + (v364_i1 * 64))));
                  v368_data.copy_to(glb_m0 + ((v365_a + (v364_i1 * 35))));
                }
              }
              #pragma unroll
              for (int32_t v373_i1 = 0; v373_i1 < 4; ++v373_i1) {
                tensorforge::intel_esimd::simd<float, 3> v376_data;
                v376_data.copy_from(r1 + ((32 + (v373_i1 * 64))));
                v376_data.copy_to(glb_m0 + ((32_i32 + (v373_i1 * 35))));
              }
            }
          }
        }
      });
    }
  });
}


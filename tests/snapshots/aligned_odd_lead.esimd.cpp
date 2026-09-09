// === base name ===
kernel_c3122a6db4f79ad7

// === header ===
void launcher_kernel_c3122a6db4f79ad7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_c3122a6db4f79ad7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c3122a6db4f79ad7(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c3122a6db4f79ad7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float r0[512]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                int32_t v13_lead = v11_i0 * 32;
                #pragma unroll
                for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v17_data;
                  v17_data.copy_from(glb_m1 + ((v13_lead + (v12_i1 * 35))));
                  v17_data.copy_to(r0 + ((v13_lead + (v12_i1 * 64))));
                }
              }
              #pragma unroll
              for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
                tensorforge::intel_esimd::simd<float, 3> v27_data;
                v27_data.copy_from(glb_m1 + ((32_i32 + (v21_i1 * 35))));
                v27_data.copy_to(r0 + ((32 + (v21_i1 * 64))));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              v30_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[256]{};
              // r1 = +(r0 * s0) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir1[256]{};
              tensorforge::intel_esimd::simd<float, 32> v33_data;
              v33_data.copy_from(r0 + (0));
              float v34_data = s0[0];
              tensorforge::intel_esimd::simd<float, 32> v36_data;
              v36_data.copy_from(ir1 + (0));
              (v36_data + (v33_data * v34_data)).copy_to(ir1 + (0));
              float v39_data = s0[8];
              tensorforge::intel_esimd::simd<float, 32> v41_data;
              v41_data.copy_from(ir1 + (64));
              (v41_data + (v33_data * v39_data)).copy_to(ir1 + (64));
              float v44_data = s0[16];
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(ir1 + (128));
              (v46_data + (v33_data * v44_data)).copy_to(ir1 + (128));
              float v49_data = s0[24];
              tensorforge::intel_esimd::simd<float, 32> v51_data;
              v51_data.copy_from(ir1 + (192));
              (v51_data + (v33_data * v49_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v53_data;
              v53_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 3> v56_data;
              v56_data.copy_from(ir1 + (32));
              (v56_data + (v53_data * v34_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v61_data;
              v61_data.copy_from(ir1 + (96));
              (v61_data + (v53_data * v39_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v66_data;
              v66_data.copy_from(ir1 + (160));
              (v66_data + (v53_data * v44_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v71_data;
              v71_data.copy_from(ir1 + (224));
              (v71_data + (v53_data * v49_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v73_data;
              v73_data.copy_from(r0 + (64));
              float v74_data = s0[1];
              tensorforge::intel_esimd::simd<float, 32> v76_data;
              v76_data.copy_from(ir1 + (0));
              (v76_data + (v73_data * v74_data)).copy_to(ir1 + (0));
              float v79_data = s0[9];
              tensorforge::intel_esimd::simd<float, 32> v81_data;
              v81_data.copy_from(ir1 + (64));
              (v81_data + (v73_data * v79_data)).copy_to(ir1 + (64));
              float v84_data = s0[17];
              tensorforge::intel_esimd::simd<float, 32> v86_data;
              v86_data.copy_from(ir1 + (128));
              (v86_data + (v73_data * v84_data)).copy_to(ir1 + (128));
              float v89_data = s0[25];
              tensorforge::intel_esimd::simd<float, 32> v91_data;
              v91_data.copy_from(ir1 + (192));
              (v91_data + (v73_data * v89_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v93_data;
              v93_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 3> v96_data;
              v96_data.copy_from(ir1 + (32));
              (v96_data + (v93_data * v74_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v101_data;
              v101_data.copy_from(ir1 + (96));
              (v101_data + (v93_data * v79_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v106_data;
              v106_data.copy_from(ir1 + (160));
              (v106_data + (v93_data * v84_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v111_data;
              v111_data.copy_from(ir1 + (224));
              (v111_data + (v93_data * v89_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v113_data;
              v113_data.copy_from(r0 + (128));
              float v114_data = s0[2];
              tensorforge::intel_esimd::simd<float, 32> v116_data;
              v116_data.copy_from(ir1 + (0));
              (v116_data + (v113_data * v114_data)).copy_to(ir1 + (0));
              float v119_data = s0[10];
              tensorforge::intel_esimd::simd<float, 32> v121_data;
              v121_data.copy_from(ir1 + (64));
              (v121_data + (v113_data * v119_data)).copy_to(ir1 + (64));
              float v124_data = s0[18];
              tensorforge::intel_esimd::simd<float, 32> v126_data;
              v126_data.copy_from(ir1 + (128));
              (v126_data + (v113_data * v124_data)).copy_to(ir1 + (128));
              float v129_data = s0[26];
              tensorforge::intel_esimd::simd<float, 32> v131_data;
              v131_data.copy_from(ir1 + (192));
              (v131_data + (v113_data * v129_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v133_data;
              v133_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 3> v136_data;
              v136_data.copy_from(ir1 + (32));
              (v136_data + (v133_data * v114_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v141_data;
              v141_data.copy_from(ir1 + (96));
              (v141_data + (v133_data * v119_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v146_data;
              v146_data.copy_from(ir1 + (160));
              (v146_data + (v133_data * v124_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v151_data;
              v151_data.copy_from(ir1 + (224));
              (v151_data + (v133_data * v129_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v153_data;
              v153_data.copy_from(r0 + (192));
              float v154_data = s0[3];
              tensorforge::intel_esimd::simd<float, 32> v156_data;
              v156_data.copy_from(ir1 + (0));
              (v156_data + (v153_data * v154_data)).copy_to(ir1 + (0));
              float v159_data = s0[11];
              tensorforge::intel_esimd::simd<float, 32> v161_data;
              v161_data.copy_from(ir1 + (64));
              (v161_data + (v153_data * v159_data)).copy_to(ir1 + (64));
              float v164_data = s0[19];
              tensorforge::intel_esimd::simd<float, 32> v166_data;
              v166_data.copy_from(ir1 + (128));
              (v166_data + (v153_data * v164_data)).copy_to(ir1 + (128));
              float v169_data = s0[27];
              tensorforge::intel_esimd::simd<float, 32> v171_data;
              v171_data.copy_from(ir1 + (192));
              (v171_data + (v153_data * v169_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v173_data;
              v173_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 3> v176_data;
              v176_data.copy_from(ir1 + (32));
              (v176_data + (v173_data * v154_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v181_data;
              v181_data.copy_from(ir1 + (96));
              (v181_data + (v173_data * v159_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v186_data;
              v186_data.copy_from(ir1 + (160));
              (v186_data + (v173_data * v164_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v191_data;
              v191_data.copy_from(ir1 + (224));
              (v191_data + (v173_data * v169_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v193_data;
              v193_data.copy_from(r0 + (256));
              float v194_data = s0[4];
              tensorforge::intel_esimd::simd<float, 32> v196_data;
              v196_data.copy_from(ir1 + (0));
              (v196_data + (v193_data * v194_data)).copy_to(ir1 + (0));
              float v199_data = s0[12];
              tensorforge::intel_esimd::simd<float, 32> v201_data;
              v201_data.copy_from(ir1 + (64));
              (v201_data + (v193_data * v199_data)).copy_to(ir1 + (64));
              float v204_data = s0[20];
              tensorforge::intel_esimd::simd<float, 32> v206_data;
              v206_data.copy_from(ir1 + (128));
              (v206_data + (v193_data * v204_data)).copy_to(ir1 + (128));
              float v209_data = s0[28];
              tensorforge::intel_esimd::simd<float, 32> v211_data;
              v211_data.copy_from(ir1 + (192));
              (v211_data + (v193_data * v209_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v213_data;
              v213_data.copy_from(r0 + (288));
              tensorforge::intel_esimd::simd<float, 3> v216_data;
              v216_data.copy_from(ir1 + (32));
              (v216_data + (v213_data * v194_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v221_data;
              v221_data.copy_from(ir1 + (96));
              (v221_data + (v213_data * v199_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v226_data;
              v226_data.copy_from(ir1 + (160));
              (v226_data + (v213_data * v204_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v231_data;
              v231_data.copy_from(ir1 + (224));
              (v231_data + (v213_data * v209_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v233_data;
              v233_data.copy_from(r0 + (320));
              float v234_data = s0[5];
              tensorforge::intel_esimd::simd<float, 32> v236_data;
              v236_data.copy_from(ir1 + (0));
              (v236_data + (v233_data * v234_data)).copy_to(ir1 + (0));
              float v239_data = s0[13];
              tensorforge::intel_esimd::simd<float, 32> v241_data;
              v241_data.copy_from(ir1 + (64));
              (v241_data + (v233_data * v239_data)).copy_to(ir1 + (64));
              float v244_data = s0[21];
              tensorforge::intel_esimd::simd<float, 32> v246_data;
              v246_data.copy_from(ir1 + (128));
              (v246_data + (v233_data * v244_data)).copy_to(ir1 + (128));
              float v249_data = s0[29];
              tensorforge::intel_esimd::simd<float, 32> v251_data;
              v251_data.copy_from(ir1 + (192));
              (v251_data + (v233_data * v249_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v253_data;
              v253_data.copy_from(r0 + (352));
              tensorforge::intel_esimd::simd<float, 3> v256_data;
              v256_data.copy_from(ir1 + (32));
              (v256_data + (v253_data * v234_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v261_data;
              v261_data.copy_from(ir1 + (96));
              (v261_data + (v253_data * v239_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v266_data;
              v266_data.copy_from(ir1 + (160));
              (v266_data + (v253_data * v244_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v271_data;
              v271_data.copy_from(ir1 + (224));
              (v271_data + (v253_data * v249_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v273_data;
              v273_data.copy_from(r0 + (384));
              float v274_data = s0[6];
              tensorforge::intel_esimd::simd<float, 32> v276_data;
              v276_data.copy_from(ir1 + (0));
              (v276_data + (v273_data * v274_data)).copy_to(ir1 + (0));
              float v279_data = s0[14];
              tensorforge::intel_esimd::simd<float, 32> v281_data;
              v281_data.copy_from(ir1 + (64));
              (v281_data + (v273_data * v279_data)).copy_to(ir1 + (64));
              float v284_data = s0[22];
              tensorforge::intel_esimd::simd<float, 32> v286_data;
              v286_data.copy_from(ir1 + (128));
              (v286_data + (v273_data * v284_data)).copy_to(ir1 + (128));
              float v289_data = s0[30];
              tensorforge::intel_esimd::simd<float, 32> v291_data;
              v291_data.copy_from(ir1 + (192));
              (v291_data + (v273_data * v289_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v293_data;
              v293_data.copy_from(r0 + (416));
              tensorforge::intel_esimd::simd<float, 3> v296_data;
              v296_data.copy_from(ir1 + (32));
              (v296_data + (v293_data * v274_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v301_data;
              v301_data.copy_from(ir1 + (96));
              (v301_data + (v293_data * v279_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v306_data;
              v306_data.copy_from(ir1 + (160));
              (v306_data + (v293_data * v284_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v311_data;
              v311_data.copy_from(ir1 + (224));
              (v311_data + (v293_data * v289_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v313_data;
              v313_data.copy_from(r0 + (448));
              float v314_data = s0[7];
              tensorforge::intel_esimd::simd<float, 32> v316_data;
              v316_data.copy_from(ir1 + (0));
              (v316_data + (v313_data * v314_data)).copy_to(ir1 + (0));
              float v319_data = s0[15];
              tensorforge::intel_esimd::simd<float, 32> v321_data;
              v321_data.copy_from(ir1 + (64));
              (v321_data + (v313_data * v319_data)).copy_to(ir1 + (64));
              float v324_data = s0[23];
              tensorforge::intel_esimd::simd<float, 32> v326_data;
              v326_data.copy_from(ir1 + (128));
              (v326_data + (v313_data * v324_data)).copy_to(ir1 + (128));
              float v329_data = s0[31];
              tensorforge::intel_esimd::simd<float, 32> v331_data;
              v331_data.copy_from(ir1 + (192));
              (v331_data + (v313_data * v329_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v333_data;
              v333_data.copy_from(r0 + (480));
              tensorforge::intel_esimd::simd<float, 3> v336_data;
              v336_data.copy_from(ir1 + (32));
              (v336_data + (v333_data * v314_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v341_data;
              v341_data.copy_from(ir1 + (96));
              (v341_data + (v333_data * v319_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v346_data;
              v346_data.copy_from(ir1 + (160));
              (v346_data + (v333_data * v324_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v351_data;
              v351_data.copy_from(ir1 + (224));
              (v351_data + (v333_data * v329_data)).copy_to(ir1 + (224));
              #pragma unroll
              for (int32_t v353_n0 = 0; v353_n0 < 1; ++v353_n0) {
                int32_t v355_a = v353_n0 * 32;
                #pragma unroll
                for (int32_t v354_n1 = 0; v354_n1 < 4; ++v354_n1) {
                  int32_t v357_a = v355_a + (v354_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v358_data;
                  v358_data.copy_from(ir1 + (v357_a));
                  v358_data.copy_to(r1 + (v357_a));
                }
              }
              #pragma unroll
              for (int32_t v362_n1 = 0; v362_n1 < 4; ++v362_n1) {
                int32_t v364_a = 32 + (v362_n1 * 64);
                tensorforge::intel_esimd::simd<float, 3> v365_data;
                v365_data.copy_from(ir1 + (v364_a));
                v365_data.copy_to(r1 + (v364_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v368_i0 = 0; v368_i0 < 1; ++v368_i0) {
                int32_t v370_a = v368_i0 * 32;
                #pragma unroll
                for (int32_t v369_i1 = 0; v369_i1 < 4; ++v369_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v373_data;
                  v373_data.copy_from(r1 + ((v370_a + (v369_i1 * 64))));
                  v373_data.copy_to(glb_m0 + ((v370_a + (v369_i1 * 35))));
                }
              }
              #pragma unroll
              for (int32_t v378_i1 = 0; v378_i1 < 4; ++v378_i1) {
                tensorforge::intel_esimd::simd<float, 3> v381_data;
                v381_data.copy_from(r1 + ((32 + (v378_i1 * 64))));
                v381_data.copy_to(glb_m0 + ((32_i32 + (v378_i1 * 35))));
              }
            }
          }
        }
      });
    }
  });
}


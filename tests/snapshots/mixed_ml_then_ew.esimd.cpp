// === base name ===
kernel_8ab0d0fff0

// === header ===
void launcher_kernel_8ab0d0fff0(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8ab0d0fff0(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8ab0d0fff0(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8ab0d0fff0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(TMP)
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 8; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 8> v11_data;
                v11_data.copy_from(glb_m0 + ((v6_i1 * 8)));
                v11_data.copy_to(r0 + ((v6_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m1[0 + 0 + 4 * item.get_local_id(0) + 0];
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v16_data;
              v16_data.copy_from(r0 + (0));
              float v17_data = s0[0];
              tensorforge::intel_esimd::simd<float, 8> v19_data;
              v19_data.copy_from(r1 + (0));
              (v19_data + (v16_data * v17_data)).copy_to(r1 + (0));
              float v22_data = s0[8];
              tensorforge::intel_esimd::simd<float, 8> v24_data;
              v24_data.copy_from(r1 + (16));
              (v24_data + (v16_data * v22_data)).copy_to(r1 + (16));
              float v27_data = s0[16];
              tensorforge::intel_esimd::simd<float, 8> v29_data;
              v29_data.copy_from(r1 + (32));
              (v29_data + (v16_data * v27_data)).copy_to(r1 + (32));
              float v32_data = s0[24];
              tensorforge::intel_esimd::simd<float, 8> v34_data;
              v34_data.copy_from(r1 + (48));
              (v34_data + (v16_data * v32_data)).copy_to(r1 + (48));
              float v37_data = s0[33];
              tensorforge::intel_esimd::simd<float, 8> v39_data;
              v39_data.copy_from(r1 + (64));
              (v39_data + (v16_data * v37_data)).copy_to(r1 + (64));
              float v42_data = s0[41];
              tensorforge::intel_esimd::simd<float, 8> v44_data;
              v44_data.copy_from(r1 + (80));
              (v44_data + (v16_data * v42_data)).copy_to(r1 + (80));
              float v47_data = s0[49];
              tensorforge::intel_esimd::simd<float, 8> v49_data;
              v49_data.copy_from(r1 + (96));
              (v49_data + (v16_data * v47_data)).copy_to(r1 + (96));
              float v52_data = s0[57];
              tensorforge::intel_esimd::simd<float, 8> v54_data;
              v54_data.copy_from(r1 + (112));
              (v54_data + (v16_data * v52_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v56_data;
              v56_data.copy_from(r0 + (16));
              float v57_data = s0[1];
              tensorforge::intel_esimd::simd<float, 8> v59_data;
              v59_data.copy_from(r1 + (0));
              (v59_data + (v56_data * v57_data)).copy_to(r1 + (0));
              float v62_data = s0[9];
              tensorforge::intel_esimd::simd<float, 8> v64_data;
              v64_data.copy_from(r1 + (16));
              (v64_data + (v56_data * v62_data)).copy_to(r1 + (16));
              float v67_data = s0[17];
              tensorforge::intel_esimd::simd<float, 8> v69_data;
              v69_data.copy_from(r1 + (32));
              (v69_data + (v56_data * v67_data)).copy_to(r1 + (32));
              float v72_data = s0[25];
              tensorforge::intel_esimd::simd<float, 8> v74_data;
              v74_data.copy_from(r1 + (48));
              (v74_data + (v56_data * v72_data)).copy_to(r1 + (48));
              float v77_data = s0[32];
              tensorforge::intel_esimd::simd<float, 8> v79_data;
              v79_data.copy_from(r1 + (64));
              (v79_data + (v56_data * v77_data)).copy_to(r1 + (64));
              float v82_data = s0[40];
              tensorforge::intel_esimd::simd<float, 8> v84_data;
              v84_data.copy_from(r1 + (80));
              (v84_data + (v56_data * v82_data)).copy_to(r1 + (80));
              float v87_data = s0[48];
              tensorforge::intel_esimd::simd<float, 8> v89_data;
              v89_data.copy_from(r1 + (96));
              (v89_data + (v56_data * v87_data)).copy_to(r1 + (96));
              float v92_data = s0[56];
              tensorforge::intel_esimd::simd<float, 8> v94_data;
              v94_data.copy_from(r1 + (112));
              (v94_data + (v56_data * v92_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v96_data;
              v96_data.copy_from(r0 + (32));
              float v97_data = s0[2];
              tensorforge::intel_esimd::simd<float, 8> v99_data;
              v99_data.copy_from(r1 + (0));
              (v99_data + (v96_data * v97_data)).copy_to(r1 + (0));
              float v102_data = s0[10];
              tensorforge::intel_esimd::simd<float, 8> v104_data;
              v104_data.copy_from(r1 + (16));
              (v104_data + (v96_data * v102_data)).copy_to(r1 + (16));
              float v107_data = s0[18];
              tensorforge::intel_esimd::simd<float, 8> v109_data;
              v109_data.copy_from(r1 + (32));
              (v109_data + (v96_data * v107_data)).copy_to(r1 + (32));
              float v112_data = s0[26];
              tensorforge::intel_esimd::simd<float, 8> v114_data;
              v114_data.copy_from(r1 + (48));
              (v114_data + (v96_data * v112_data)).copy_to(r1 + (48));
              float v117_data = s0[35];
              tensorforge::intel_esimd::simd<float, 8> v119_data;
              v119_data.copy_from(r1 + (64));
              (v119_data + (v96_data * v117_data)).copy_to(r1 + (64));
              float v122_data = s0[43];
              tensorforge::intel_esimd::simd<float, 8> v124_data;
              v124_data.copy_from(r1 + (80));
              (v124_data + (v96_data * v122_data)).copy_to(r1 + (80));
              float v127_data = s0[51];
              tensorforge::intel_esimd::simd<float, 8> v129_data;
              v129_data.copy_from(r1 + (96));
              (v129_data + (v96_data * v127_data)).copy_to(r1 + (96));
              float v132_data = s0[59];
              tensorforge::intel_esimd::simd<float, 8> v134_data;
              v134_data.copy_from(r1 + (112));
              (v134_data + (v96_data * v132_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v136_data;
              v136_data.copy_from(r0 + (48));
              float v137_data = s0[3];
              tensorforge::intel_esimd::simd<float, 8> v139_data;
              v139_data.copy_from(r1 + (0));
              (v139_data + (v136_data * v137_data)).copy_to(r1 + (0));
              float v142_data = s0[11];
              tensorforge::intel_esimd::simd<float, 8> v144_data;
              v144_data.copy_from(r1 + (16));
              (v144_data + (v136_data * v142_data)).copy_to(r1 + (16));
              float v147_data = s0[19];
              tensorforge::intel_esimd::simd<float, 8> v149_data;
              v149_data.copy_from(r1 + (32));
              (v149_data + (v136_data * v147_data)).copy_to(r1 + (32));
              float v152_data = s0[27];
              tensorforge::intel_esimd::simd<float, 8> v154_data;
              v154_data.copy_from(r1 + (48));
              (v154_data + (v136_data * v152_data)).copy_to(r1 + (48));
              float v157_data = s0[34];
              tensorforge::intel_esimd::simd<float, 8> v159_data;
              v159_data.copy_from(r1 + (64));
              (v159_data + (v136_data * v157_data)).copy_to(r1 + (64));
              float v162_data = s0[42];
              tensorforge::intel_esimd::simd<float, 8> v164_data;
              v164_data.copy_from(r1 + (80));
              (v164_data + (v136_data * v162_data)).copy_to(r1 + (80));
              float v167_data = s0[50];
              tensorforge::intel_esimd::simd<float, 8> v169_data;
              v169_data.copy_from(r1 + (96));
              (v169_data + (v136_data * v167_data)).copy_to(r1 + (96));
              float v172_data = s0[58];
              tensorforge::intel_esimd::simd<float, 8> v174_data;
              v174_data.copy_from(r1 + (112));
              (v174_data + (v136_data * v172_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v176_data;
              v176_data.copy_from(r0 + (64));
              float v177_data = s0[4];
              tensorforge::intel_esimd::simd<float, 8> v179_data;
              v179_data.copy_from(r1 + (0));
              (v179_data + (v176_data * v177_data)).copy_to(r1 + (0));
              float v182_data = s0[12];
              tensorforge::intel_esimd::simd<float, 8> v184_data;
              v184_data.copy_from(r1 + (16));
              (v184_data + (v176_data * v182_data)).copy_to(r1 + (16));
              float v187_data = s0[20];
              tensorforge::intel_esimd::simd<float, 8> v189_data;
              v189_data.copy_from(r1 + (32));
              (v189_data + (v176_data * v187_data)).copy_to(r1 + (32));
              float v192_data = s0[28];
              tensorforge::intel_esimd::simd<float, 8> v194_data;
              v194_data.copy_from(r1 + (48));
              (v194_data + (v176_data * v192_data)).copy_to(r1 + (48));
              float v197_data = s0[37];
              tensorforge::intel_esimd::simd<float, 8> v199_data;
              v199_data.copy_from(r1 + (64));
              (v199_data + (v176_data * v197_data)).copy_to(r1 + (64));
              float v202_data = s0[45];
              tensorforge::intel_esimd::simd<float, 8> v204_data;
              v204_data.copy_from(r1 + (80));
              (v204_data + (v176_data * v202_data)).copy_to(r1 + (80));
              float v207_data = s0[53];
              tensorforge::intel_esimd::simd<float, 8> v209_data;
              v209_data.copy_from(r1 + (96));
              (v209_data + (v176_data * v207_data)).copy_to(r1 + (96));
              float v212_data = s0[61];
              tensorforge::intel_esimd::simd<float, 8> v214_data;
              v214_data.copy_from(r1 + (112));
              (v214_data + (v176_data * v212_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v216_data;
              v216_data.copy_from(r0 + (80));
              float v217_data = s0[5];
              tensorforge::intel_esimd::simd<float, 8> v219_data;
              v219_data.copy_from(r1 + (0));
              (v219_data + (v216_data * v217_data)).copy_to(r1 + (0));
              float v222_data = s0[13];
              tensorforge::intel_esimd::simd<float, 8> v224_data;
              v224_data.copy_from(r1 + (16));
              (v224_data + (v216_data * v222_data)).copy_to(r1 + (16));
              float v227_data = s0[21];
              tensorforge::intel_esimd::simd<float, 8> v229_data;
              v229_data.copy_from(r1 + (32));
              (v229_data + (v216_data * v227_data)).copy_to(r1 + (32));
              float v232_data = s0[29];
              tensorforge::intel_esimd::simd<float, 8> v234_data;
              v234_data.copy_from(r1 + (48));
              (v234_data + (v216_data * v232_data)).copy_to(r1 + (48));
              float v237_data = s0[36];
              tensorforge::intel_esimd::simd<float, 8> v239_data;
              v239_data.copy_from(r1 + (64));
              (v239_data + (v216_data * v237_data)).copy_to(r1 + (64));
              float v242_data = s0[44];
              tensorforge::intel_esimd::simd<float, 8> v244_data;
              v244_data.copy_from(r1 + (80));
              (v244_data + (v216_data * v242_data)).copy_to(r1 + (80));
              float v247_data = s0[52];
              tensorforge::intel_esimd::simd<float, 8> v249_data;
              v249_data.copy_from(r1 + (96));
              (v249_data + (v216_data * v247_data)).copy_to(r1 + (96));
              float v252_data = s0[60];
              tensorforge::intel_esimd::simd<float, 8> v254_data;
              v254_data.copy_from(r1 + (112));
              (v254_data + (v216_data * v252_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v256_data;
              v256_data.copy_from(r0 + (96));
              float v257_data = s0[6];
              tensorforge::intel_esimd::simd<float, 8> v259_data;
              v259_data.copy_from(r1 + (0));
              (v259_data + (v256_data * v257_data)).copy_to(r1 + (0));
              float v262_data = s0[14];
              tensorforge::intel_esimd::simd<float, 8> v264_data;
              v264_data.copy_from(r1 + (16));
              (v264_data + (v256_data * v262_data)).copy_to(r1 + (16));
              float v267_data = s0[22];
              tensorforge::intel_esimd::simd<float, 8> v269_data;
              v269_data.copy_from(r1 + (32));
              (v269_data + (v256_data * v267_data)).copy_to(r1 + (32));
              float v272_data = s0[30];
              tensorforge::intel_esimd::simd<float, 8> v274_data;
              v274_data.copy_from(r1 + (48));
              (v274_data + (v256_data * v272_data)).copy_to(r1 + (48));
              float v277_data = s0[39];
              tensorforge::intel_esimd::simd<float, 8> v279_data;
              v279_data.copy_from(r1 + (64));
              (v279_data + (v256_data * v277_data)).copy_to(r1 + (64));
              float v282_data = s0[47];
              tensorforge::intel_esimd::simd<float, 8> v284_data;
              v284_data.copy_from(r1 + (80));
              (v284_data + (v256_data * v282_data)).copy_to(r1 + (80));
              float v287_data = s0[55];
              tensorforge::intel_esimd::simd<float, 8> v289_data;
              v289_data.copy_from(r1 + (96));
              (v289_data + (v256_data * v287_data)).copy_to(r1 + (96));
              float v292_data = s0[63];
              tensorforge::intel_esimd::simd<float, 8> v294_data;
              v294_data.copy_from(r1 + (112));
              (v294_data + (v256_data * v292_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v296_data;
              v296_data.copy_from(r0 + (112));
              float v297_data = s0[7];
              tensorforge::intel_esimd::simd<float, 8> v299_data;
              v299_data.copy_from(r1 + (0));
              (v299_data + (v296_data * v297_data)).copy_to(r1 + (0));
              float v302_data = s0[15];
              tensorforge::intel_esimd::simd<float, 8> v304_data;
              v304_data.copy_from(r1 + (16));
              (v304_data + (v296_data * v302_data)).copy_to(r1 + (16));
              float v307_data = s0[23];
              tensorforge::intel_esimd::simd<float, 8> v309_data;
              v309_data.copy_from(r1 + (32));
              (v309_data + (v296_data * v307_data)).copy_to(r1 + (32));
              float v312_data = s0[31];
              tensorforge::intel_esimd::simd<float, 8> v314_data;
              v314_data.copy_from(r1 + (48));
              (v314_data + (v296_data * v312_data)).copy_to(r1 + (48));
              float v317_data = s0[38];
              tensorforge::intel_esimd::simd<float, 8> v319_data;
              v319_data.copy_from(r1 + (64));
              (v319_data + (v296_data * v317_data)).copy_to(r1 + (64));
              float v322_data = s0[46];
              tensorforge::intel_esimd::simd<float, 8> v324_data;
              v324_data.copy_from(r1 + (80));
              (v324_data + (v296_data * v322_data)).copy_to(r1 + (80));
              float v327_data = s0[54];
              tensorforge::intel_esimd::simd<float, 8> v329_data;
              v329_data.copy_from(r1 + (96));
              (v329_data + (v296_data * v327_data)).copy_to(r1 + (96));
              float v332_data = s0[62];
              tensorforge::intel_esimd::simd<float, 8> v334_data;
              v334_data.copy_from(r1 + (112));
              (v334_data + (v296_data * v332_data)).copy_to(r1 + (112));
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v337_i1 = 0; v337_i1 < 8; ++v337_i1) {
                tensorforge::intel_esimd::simd<float, 8> v340_data;
                v340_data.copy_from(r1 + ((v337_i1 * 16)));
                int32_t v343_a = v337_i1 * 8;
                v340_data.copy_to(s1 + ((v343_a ^ ((v343_a >> 5) & 31))));
              }
              // glb_m2 = abs(s1)
              #pragma unroll
              for (int32_t v348_k1 = 0; v348_k1 < 8; ++v348_k1) {
                int32_t v351_a = v348_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v356_data;
                v356_data.copy_from(s1 + ((v351_a ^ ((v351_a >> 5) & 31))));
                (tensorforge::intel_esimd::abs(v356_data)).copy_to(glb_m2 + (v351_a));
              }
            }
          }
        }
      });
    }
  });
}


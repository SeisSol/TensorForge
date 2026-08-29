// === base name ===
kernel_a587425bdd

// === header ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a587425bdd(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a587425bdd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // TMP = abs(A)
        // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s1[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              float r0[128]{};
              // r0 = abs(glb_m0)
              #pragma unroll
              for (int32_t v7_k1 = 0; v7_k1 < 8; ++v7_k1) {
                tensorforge::intel_esimd::simd<float, 8> v12_data;
                v12_data.copy_from(glb_m0 + ((v7_k1 * 8)));
                (tensorforge::intel_esimd::abs(v12_data)).copy_to(r0 + ((v7_k1 * 16)));
              }
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 8> v18_data;
              v18_data.copy_from(r0 + (0));
              float v19_data = s1[0];
              tensorforge::intel_esimd::simd<float, 8> v21_data;
              v21_data.copy_from(ir1 + (0));
              (v21_data + (v18_data * v19_data)).copy_to(ir1 + (0));
              float v24_data = s1[8];
              tensorforge::intel_esimd::simd<float, 8> v26_data;
              v26_data.copy_from(ir1 + (16));
              (v26_data + (v18_data * v24_data)).copy_to(ir1 + (16));
              float v29_data = s1[16];
              tensorforge::intel_esimd::simd<float, 8> v31_data;
              v31_data.copy_from(ir1 + (32));
              (v31_data + (v18_data * v29_data)).copy_to(ir1 + (32));
              float v34_data = s1[24];
              tensorforge::intel_esimd::simd<float, 8> v36_data;
              v36_data.copy_from(ir1 + (48));
              (v36_data + (v18_data * v34_data)).copy_to(ir1 + (48));
              float v39_data = s1[33];
              tensorforge::intel_esimd::simd<float, 8> v41_data;
              v41_data.copy_from(ir1 + (64));
              (v41_data + (v18_data * v39_data)).copy_to(ir1 + (64));
              float v44_data = s1[41];
              tensorforge::intel_esimd::simd<float, 8> v46_data;
              v46_data.copy_from(ir1 + (80));
              (v46_data + (v18_data * v44_data)).copy_to(ir1 + (80));
              float v49_data = s1[49];
              tensorforge::intel_esimd::simd<float, 8> v51_data;
              v51_data.copy_from(ir1 + (96));
              (v51_data + (v18_data * v49_data)).copy_to(ir1 + (96));
              float v54_data = s1[57];
              tensorforge::intel_esimd::simd<float, 8> v56_data;
              v56_data.copy_from(ir1 + (112));
              (v56_data + (v18_data * v54_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v58_data;
              v58_data.copy_from(r0 + (16));
              float v59_data = s1[1];
              tensorforge::intel_esimd::simd<float, 8> v61_data;
              v61_data.copy_from(ir1 + (0));
              (v61_data + (v58_data * v59_data)).copy_to(ir1 + (0));
              float v64_data = s1[9];
              tensorforge::intel_esimd::simd<float, 8> v66_data;
              v66_data.copy_from(ir1 + (16));
              (v66_data + (v58_data * v64_data)).copy_to(ir1 + (16));
              float v69_data = s1[17];
              tensorforge::intel_esimd::simd<float, 8> v71_data;
              v71_data.copy_from(ir1 + (32));
              (v71_data + (v58_data * v69_data)).copy_to(ir1 + (32));
              float v74_data = s1[25];
              tensorforge::intel_esimd::simd<float, 8> v76_data;
              v76_data.copy_from(ir1 + (48));
              (v76_data + (v58_data * v74_data)).copy_to(ir1 + (48));
              float v79_data = s1[32];
              tensorforge::intel_esimd::simd<float, 8> v81_data;
              v81_data.copy_from(ir1 + (64));
              (v81_data + (v58_data * v79_data)).copy_to(ir1 + (64));
              float v84_data = s1[40];
              tensorforge::intel_esimd::simd<float, 8> v86_data;
              v86_data.copy_from(ir1 + (80));
              (v86_data + (v58_data * v84_data)).copy_to(ir1 + (80));
              float v89_data = s1[48];
              tensorforge::intel_esimd::simd<float, 8> v91_data;
              v91_data.copy_from(ir1 + (96));
              (v91_data + (v58_data * v89_data)).copy_to(ir1 + (96));
              float v94_data = s1[56];
              tensorforge::intel_esimd::simd<float, 8> v96_data;
              v96_data.copy_from(ir1 + (112));
              (v96_data + (v58_data * v94_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v98_data;
              v98_data.copy_from(r0 + (32));
              float v99_data = s1[2];
              tensorforge::intel_esimd::simd<float, 8> v101_data;
              v101_data.copy_from(ir1 + (0));
              (v101_data + (v98_data * v99_data)).copy_to(ir1 + (0));
              float v104_data = s1[10];
              tensorforge::intel_esimd::simd<float, 8> v106_data;
              v106_data.copy_from(ir1 + (16));
              (v106_data + (v98_data * v104_data)).copy_to(ir1 + (16));
              float v109_data = s1[18];
              tensorforge::intel_esimd::simd<float, 8> v111_data;
              v111_data.copy_from(ir1 + (32));
              (v111_data + (v98_data * v109_data)).copy_to(ir1 + (32));
              float v114_data = s1[26];
              tensorforge::intel_esimd::simd<float, 8> v116_data;
              v116_data.copy_from(ir1 + (48));
              (v116_data + (v98_data * v114_data)).copy_to(ir1 + (48));
              float v119_data = s1[35];
              tensorforge::intel_esimd::simd<float, 8> v121_data;
              v121_data.copy_from(ir1 + (64));
              (v121_data + (v98_data * v119_data)).copy_to(ir1 + (64));
              float v124_data = s1[43];
              tensorforge::intel_esimd::simd<float, 8> v126_data;
              v126_data.copy_from(ir1 + (80));
              (v126_data + (v98_data * v124_data)).copy_to(ir1 + (80));
              float v129_data = s1[51];
              tensorforge::intel_esimd::simd<float, 8> v131_data;
              v131_data.copy_from(ir1 + (96));
              (v131_data + (v98_data * v129_data)).copy_to(ir1 + (96));
              float v134_data = s1[59];
              tensorforge::intel_esimd::simd<float, 8> v136_data;
              v136_data.copy_from(ir1 + (112));
              (v136_data + (v98_data * v134_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v138_data;
              v138_data.copy_from(r0 + (48));
              float v139_data = s1[3];
              tensorforge::intel_esimd::simd<float, 8> v141_data;
              v141_data.copy_from(ir1 + (0));
              (v141_data + (v138_data * v139_data)).copy_to(ir1 + (0));
              float v144_data = s1[11];
              tensorforge::intel_esimd::simd<float, 8> v146_data;
              v146_data.copy_from(ir1 + (16));
              (v146_data + (v138_data * v144_data)).copy_to(ir1 + (16));
              float v149_data = s1[19];
              tensorforge::intel_esimd::simd<float, 8> v151_data;
              v151_data.copy_from(ir1 + (32));
              (v151_data + (v138_data * v149_data)).copy_to(ir1 + (32));
              float v154_data = s1[27];
              tensorforge::intel_esimd::simd<float, 8> v156_data;
              v156_data.copy_from(ir1 + (48));
              (v156_data + (v138_data * v154_data)).copy_to(ir1 + (48));
              float v159_data = s1[34];
              tensorforge::intel_esimd::simd<float, 8> v161_data;
              v161_data.copy_from(ir1 + (64));
              (v161_data + (v138_data * v159_data)).copy_to(ir1 + (64));
              float v164_data = s1[42];
              tensorforge::intel_esimd::simd<float, 8> v166_data;
              v166_data.copy_from(ir1 + (80));
              (v166_data + (v138_data * v164_data)).copy_to(ir1 + (80));
              float v169_data = s1[50];
              tensorforge::intel_esimd::simd<float, 8> v171_data;
              v171_data.copy_from(ir1 + (96));
              (v171_data + (v138_data * v169_data)).copy_to(ir1 + (96));
              float v174_data = s1[58];
              tensorforge::intel_esimd::simd<float, 8> v176_data;
              v176_data.copy_from(ir1 + (112));
              (v176_data + (v138_data * v174_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v178_data;
              v178_data.copy_from(r0 + (64));
              float v179_data = s1[4];
              tensorforge::intel_esimd::simd<float, 8> v181_data;
              v181_data.copy_from(ir1 + (0));
              (v181_data + (v178_data * v179_data)).copy_to(ir1 + (0));
              float v184_data = s1[12];
              tensorforge::intel_esimd::simd<float, 8> v186_data;
              v186_data.copy_from(ir1 + (16));
              (v186_data + (v178_data * v184_data)).copy_to(ir1 + (16));
              float v189_data = s1[20];
              tensorforge::intel_esimd::simd<float, 8> v191_data;
              v191_data.copy_from(ir1 + (32));
              (v191_data + (v178_data * v189_data)).copy_to(ir1 + (32));
              float v194_data = s1[28];
              tensorforge::intel_esimd::simd<float, 8> v196_data;
              v196_data.copy_from(ir1 + (48));
              (v196_data + (v178_data * v194_data)).copy_to(ir1 + (48));
              float v199_data = s1[37];
              tensorforge::intel_esimd::simd<float, 8> v201_data;
              v201_data.copy_from(ir1 + (64));
              (v201_data + (v178_data * v199_data)).copy_to(ir1 + (64));
              float v204_data = s1[45];
              tensorforge::intel_esimd::simd<float, 8> v206_data;
              v206_data.copy_from(ir1 + (80));
              (v206_data + (v178_data * v204_data)).copy_to(ir1 + (80));
              float v209_data = s1[53];
              tensorforge::intel_esimd::simd<float, 8> v211_data;
              v211_data.copy_from(ir1 + (96));
              (v211_data + (v178_data * v209_data)).copy_to(ir1 + (96));
              float v214_data = s1[61];
              tensorforge::intel_esimd::simd<float, 8> v216_data;
              v216_data.copy_from(ir1 + (112));
              (v216_data + (v178_data * v214_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v218_data;
              v218_data.copy_from(r0 + (80));
              float v219_data = s1[5];
              tensorforge::intel_esimd::simd<float, 8> v221_data;
              v221_data.copy_from(ir1 + (0));
              (v221_data + (v218_data * v219_data)).copy_to(ir1 + (0));
              float v224_data = s1[13];
              tensorforge::intel_esimd::simd<float, 8> v226_data;
              v226_data.copy_from(ir1 + (16));
              (v226_data + (v218_data * v224_data)).copy_to(ir1 + (16));
              float v229_data = s1[21];
              tensorforge::intel_esimd::simd<float, 8> v231_data;
              v231_data.copy_from(ir1 + (32));
              (v231_data + (v218_data * v229_data)).copy_to(ir1 + (32));
              float v234_data = s1[29];
              tensorforge::intel_esimd::simd<float, 8> v236_data;
              v236_data.copy_from(ir1 + (48));
              (v236_data + (v218_data * v234_data)).copy_to(ir1 + (48));
              float v239_data = s1[36];
              tensorforge::intel_esimd::simd<float, 8> v241_data;
              v241_data.copy_from(ir1 + (64));
              (v241_data + (v218_data * v239_data)).copy_to(ir1 + (64));
              float v244_data = s1[44];
              tensorforge::intel_esimd::simd<float, 8> v246_data;
              v246_data.copy_from(ir1 + (80));
              (v246_data + (v218_data * v244_data)).copy_to(ir1 + (80));
              float v249_data = s1[52];
              tensorforge::intel_esimd::simd<float, 8> v251_data;
              v251_data.copy_from(ir1 + (96));
              (v251_data + (v218_data * v249_data)).copy_to(ir1 + (96));
              float v254_data = s1[60];
              tensorforge::intel_esimd::simd<float, 8> v256_data;
              v256_data.copy_from(ir1 + (112));
              (v256_data + (v218_data * v254_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v258_data;
              v258_data.copy_from(r0 + (96));
              float v259_data = s1[6];
              tensorforge::intel_esimd::simd<float, 8> v261_data;
              v261_data.copy_from(ir1 + (0));
              (v261_data + (v258_data * v259_data)).copy_to(ir1 + (0));
              float v264_data = s1[14];
              tensorforge::intel_esimd::simd<float, 8> v266_data;
              v266_data.copy_from(ir1 + (16));
              (v266_data + (v258_data * v264_data)).copy_to(ir1 + (16));
              float v269_data = s1[22];
              tensorforge::intel_esimd::simd<float, 8> v271_data;
              v271_data.copy_from(ir1 + (32));
              (v271_data + (v258_data * v269_data)).copy_to(ir1 + (32));
              float v274_data = s1[30];
              tensorforge::intel_esimd::simd<float, 8> v276_data;
              v276_data.copy_from(ir1 + (48));
              (v276_data + (v258_data * v274_data)).copy_to(ir1 + (48));
              float v279_data = s1[39];
              tensorforge::intel_esimd::simd<float, 8> v281_data;
              v281_data.copy_from(ir1 + (64));
              (v281_data + (v258_data * v279_data)).copy_to(ir1 + (64));
              float v284_data = s1[47];
              tensorforge::intel_esimd::simd<float, 8> v286_data;
              v286_data.copy_from(ir1 + (80));
              (v286_data + (v258_data * v284_data)).copy_to(ir1 + (80));
              float v289_data = s1[55];
              tensorforge::intel_esimd::simd<float, 8> v291_data;
              v291_data.copy_from(ir1 + (96));
              (v291_data + (v258_data * v289_data)).copy_to(ir1 + (96));
              float v294_data = s1[63];
              tensorforge::intel_esimd::simd<float, 8> v296_data;
              v296_data.copy_from(ir1 + (112));
              (v296_data + (v258_data * v294_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v298_data;
              v298_data.copy_from(r0 + (112));
              float v299_data = s1[7];
              tensorforge::intel_esimd::simd<float, 8> v301_data;
              v301_data.copy_from(ir1 + (0));
              (v301_data + (v298_data * v299_data)).copy_to(ir1 + (0));
              float v304_data = s1[15];
              tensorforge::intel_esimd::simd<float, 8> v306_data;
              v306_data.copy_from(ir1 + (16));
              (v306_data + (v298_data * v304_data)).copy_to(ir1 + (16));
              float v309_data = s1[23];
              tensorforge::intel_esimd::simd<float, 8> v311_data;
              v311_data.copy_from(ir1 + (32));
              (v311_data + (v298_data * v309_data)).copy_to(ir1 + (32));
              float v314_data = s1[31];
              tensorforge::intel_esimd::simd<float, 8> v316_data;
              v316_data.copy_from(ir1 + (48));
              (v316_data + (v298_data * v314_data)).copy_to(ir1 + (48));
              float v319_data = s1[38];
              tensorforge::intel_esimd::simd<float, 8> v321_data;
              v321_data.copy_from(ir1 + (64));
              (v321_data + (v298_data * v319_data)).copy_to(ir1 + (64));
              float v324_data = s1[46];
              tensorforge::intel_esimd::simd<float, 8> v326_data;
              v326_data.copy_from(ir1 + (80));
              (v326_data + (v298_data * v324_data)).copy_to(ir1 + (80));
              float v329_data = s1[54];
              tensorforge::intel_esimd::simd<float, 8> v331_data;
              v331_data.copy_from(ir1 + (96));
              (v331_data + (v298_data * v329_data)).copy_to(ir1 + (96));
              float v334_data = s1[62];
              tensorforge::intel_esimd::simd<float, 8> v336_data;
              v336_data.copy_from(ir1 + (112));
              (v336_data + (v298_data * v334_data)).copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v338_n1 = 0; v338_n1 < 8; ++v338_n1) {
                int32_t v339_a = v338_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v341_data;
                v341_data.copy_from(ir1 + (v339_a));
                v341_data.copy_to(r1 + (v339_a));
              }
              // glb_m1 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v344_i1 = 0; v344_i1 < 8; ++v344_i1) {
                tensorforge::intel_esimd::simd<float, 8> v347_data;
                v347_data.copy_from(r1 + ((v344_i1 * 16)));
                v347_data.copy_to(glb_m1 + ((v344_i1 * 8)));
              }
            }
          }
        }
      });
    }
  });
}


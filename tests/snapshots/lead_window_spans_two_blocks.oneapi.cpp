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
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[156]{};
              // r0 = +(glb_m0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              auto& ir0 = r0;
              int32_t v6_lead = item.get_local_id(0) % 32;
              int32_t v8_a = 0_i32 + 0;
              int32_t v10_a = 0_i32 + 0;
              int32_t v12_a = 0_i32 + 0;
              int32_t v14_a = 0_i32 + 0;
              int32_t v16_a = 0_i32 + 0;
              int32_t v18_a = 0_i32 + 0;
              int32_t v20_a = 0_i32 + 64;
              int32_t v22_a = 0_i32 + 64;
              int32_t v24_a = 0_i32 + 64;
              int32_t v26_a = 0_i32 + 64;
              int32_t v28_a = 0_i32 + 64;
              int32_t v30_a = 0_i32 + 64;
              int32_t v32_a = 0_i32 + 128;
              int32_t v34_a = 0_i32 + 128;
              int32_t v36_a = 0_i32 + 128;
              int32_t v38_a = 0_i32 + 128;
              int32_t v40_a = 0_i32 + 128;
              int32_t v42_a = 0_i32 + 128;
              int32_t v44_a = 0_i32 + 192;
              int32_t v46_a = 0_i32 + 192;
              int32_t v48_a = 0_i32 + 192;
              int32_t v50_a = 0_i32 + 192;
              int32_t v52_a = 0_i32 + 192;
              int32_t v54_a = 0_i32 + 192;
              int32_t v56_a = 0_i32 + 256;
              int32_t v58_a = 0_i32 + 256;
              int32_t v60_a = 0_i32 + 256;
              int32_t v62_a = 0_i32 + 256;
              int32_t v64_a = 0_i32 + 256;
              int32_t v66_a = 0_i32 + 256;
              int32_t v68_a = 0_i32 + 320;
              int32_t v70_a = 0_i32 + 320;
              int32_t v72_a = 0_i32 + 320;
              int32_t v74_a = 0_i32 + 320;
              int32_t v76_a = 0_i32 + 320;
              int32_t v78_a = 0_i32 + 320;
              int32_t v80_a = 0_i32 + 384;
              int32_t v82_a = 0_i32 + 384;
              int32_t v84_a = 0_i32 + 384;
              int32_t v86_a = 0_i32 + 384;
              int32_t v88_a = 0_i32 + 384;
              int32_t v90_a = 0_i32 + 384;
              int32_t v92_a = 0_i32 + 448;
              int32_t v94_a = 0_i32 + 448;
              int32_t v96_a = 0_i32 + 448;
              int32_t v98_a = 0_i32 + 448;
              int32_t v100_a = 0_i32 + 448;
              int32_t v102_a = 0_i32 + 448;
              int32_t v104_a = 0_i32 + 512;
              int32_t v106_a = 0_i32 + 512;
              int32_t v108_a = 0_i32 + 512;
              int32_t v110_a = 0_i32 + 512;
              int32_t v112_a = 0_i32 + 512;
              int32_t v114_a = 0_i32 + 512;
              int32_t v116_a = 0_i32 + 576;
              int32_t v118_a = 0_i32 + 576;
              int32_t v120_a = 0_i32 + 576;
              int32_t v122_a = 0_i32 + 576;
              int32_t v124_a = 0_i32 + 576;
              int32_t v126_a = 0_i32 + 576;
              int32_t v128_a = 0_i32 + 640;
              int32_t v130_a = 0_i32 + 640;
              int32_t v132_a = 0_i32 + 640;
              int32_t v134_a = 0_i32 + 640;
              int32_t v136_a = 0_i32 + 640;
              int32_t v138_a = 0_i32 + 640;
              int32_t v140_a = 0_i32 + 704;
              int32_t v142_a = 0_i32 + 704;
              int32_t v144_a = 0_i32 + 704;
              int32_t v146_a = 0_i32 + 704;
              int32_t v148_a = 0_i32 + 704;
              int32_t v150_a = 0_i32 + 704;
              int32_t v152_a = 0_i32 + 768;
              int32_t v154_a = 0_i32 + 768;
              int32_t v156_a = 0_i32 + 768;
              int32_t v158_a = 0_i32 + 768;
              int32_t v160_a = 0_i32 + 768;
              int32_t v162_a = 0_i32 + 768;
              int32_t v164_a = 32_i32 + 0;
              int32_t v166_a = 32_i32 + 0;
              int32_t v168_a = 32_i32 + 0;
              int32_t v170_a = 32_i32 + 0;
              int32_t v172_a = 32_i32 + 0;
              int32_t v174_a = 32_i32 + 0;
              int32_t v176_a = 32_i32 + 64;
              int32_t v178_a = 32_i32 + 64;
              int32_t v180_a = 32_i32 + 64;
              int32_t v182_a = 32_i32 + 64;
              int32_t v184_a = 32_i32 + 64;
              int32_t v186_a = 32_i32 + 64;
              int32_t v188_a = 32_i32 + 128;
              int32_t v190_a = 32_i32 + 128;
              int32_t v192_a = 32_i32 + 128;
              int32_t v194_a = 32_i32 + 128;
              int32_t v196_a = 32_i32 + 128;
              int32_t v198_a = 32_i32 + 128;
              int32_t v200_a = 32_i32 + 192;
              int32_t v202_a = 32_i32 + 192;
              int32_t v204_a = 32_i32 + 192;
              int32_t v206_a = 32_i32 + 192;
              int32_t v208_a = 32_i32 + 192;
              int32_t v210_a = 32_i32 + 192;
              int32_t v212_a = 32_i32 + 256;
              int32_t v214_a = 32_i32 + 256;
              int32_t v216_a = 32_i32 + 256;
              int32_t v218_a = 32_i32 + 256;
              int32_t v220_a = 32_i32 + 256;
              int32_t v222_a = 32_i32 + 256;
              int32_t v224_a = 32_i32 + 320;
              int32_t v226_a = 32_i32 + 320;
              int32_t v228_a = 32_i32 + 320;
              int32_t v230_a = 32_i32 + 320;
              int32_t v232_a = 32_i32 + 320;
              int32_t v234_a = 32_i32 + 320;
              int32_t v236_a = 32_i32 + 384;
              int32_t v238_a = 32_i32 + 384;
              int32_t v240_a = 32_i32 + 384;
              int32_t v242_a = 32_i32 + 384;
              int32_t v244_a = 32_i32 + 384;
              int32_t v246_a = 32_i32 + 384;
              int32_t v248_a = 32_i32 + 448;
              int32_t v250_a = 32_i32 + 448;
              int32_t v252_a = 32_i32 + 448;
              int32_t v254_a = 32_i32 + 448;
              int32_t v256_a = 32_i32 + 448;
              int32_t v258_a = 32_i32 + 448;
              int32_t v260_a = 32_i32 + 512;
              int32_t v262_a = 32_i32 + 512;
              int32_t v264_a = 32_i32 + 512;
              int32_t v266_a = 32_i32 + 512;
              int32_t v268_a = 32_i32 + 512;
              int32_t v270_a = 32_i32 + 512;
              int32_t v272_a = 32_i32 + 576;
              int32_t v274_a = 32_i32 + 576;
              int32_t v276_a = 32_i32 + 576;
              int32_t v278_a = 32_i32 + 576;
              int32_t v280_a = 32_i32 + 576;
              int32_t v282_a = 32_i32 + 576;
              int32_t v284_a = 32_i32 + 640;
              int32_t v286_a = 32_i32 + 640;
              int32_t v288_a = 32_i32 + 640;
              int32_t v290_a = 32_i32 + 640;
              int32_t v292_a = 32_i32 + 640;
              int32_t v294_a = 32_i32 + 640;
              int32_t v296_a = 32_i32 + 704;
              int32_t v298_a = 32_i32 + 704;
              int32_t v300_a = 32_i32 + 704;
              int32_t v302_a = 32_i32 + 704;
              int32_t v304_a = 32_i32 + 704;
              int32_t v306_a = 32_i32 + 704;
              int32_t v308_a = 32_i32 + 768;
              int32_t v310_a = 32_i32 + 768;
              int32_t v312_a = 32_i32 + 768;
              int32_t v314_a = 32_i32 + 768;
              int32_t v316_a = 32_i32 + 768;
              int32_t v318_a = 32_i32 + 768;
              float r1[156]{};
              // r1 = +(r0) + name: glb_m2, type: SymbolType.Global, lead: [0]
              // [(20, 35), (0, 1), (0, 6)] []
              float ir1[12]{};
              if (v6_lead >= 20) {
                #pragma unroll
                for (int32_t v330_n1 = 0; v330_n1 < 1; ++v330_n1) {
                  int32_t v332_a = v330_n1 * 2;
                  int32_t v338_a = (v330_n1 + 12) * 64;
                  #pragma unroll
                  for (int32_t v331_n2 = 0; v331_n2 < 6; ++v331_n2) {
                    int32_t v333_a = v331_n2 * 2;
                    int32_t v335_a = v332_a + v333_a;
                    int32_t v341_a = v338_a + (v331_n2 * 832);
                    int32_t v346_a = v332_a + v333_a;
                    v342_p = r1[v346_a];
                  }
                }
              }
              if (v6_lead < 3) {
                #pragma unroll
                for (int32_t v348_n1 = 0; v348_n1 < 1; ++v348_n1) {
                  int32_t v352_a = 1 + (v348_n1 * 2);
                  int32_t v358_a = 32_i32 + ((v348_n1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v349_n2 = 0; v349_n2 < 6; ++v349_n2) {
                    int32_t v351_a = v349_n2 * 2;
                    int32_t v353_a = v352_a + v351_a;
                    int32_t v359_a = v358_a + (v349_n2 * 832);
                    int32_t v364_a = v352_a + v351_a;
                    v360_p = r1[v364_a];
                  }
                }
              }
              // glb_m2 = store{r>g}(r1);
              if (v6_lead >= 20) {
                #pragma unroll
                for (int32_t v369_i1 = 0; v369_i1 < 1; ++v369_i1) {
                  int32_t v371_a = v369_i1 * 2;
                  int32_t v377_a = (v369_i1 + 12) * 64;
                  #pragma unroll
                  for (int32_t v370_i2 = 0; v370_i2 < 6; ++v370_i2) {
                    int32_t v374_a = v371_a + (v370_i2 * 2);
                    int32_t v380_a = v377_a + (v370_i2 * 832);
                    None.copy_to(glb_m2[v380_a]);
                  }
                }
              }
              if (v6_lead < 3) {
                #pragma unroll
                for (int32_t v382_i1 = 0; v382_i1 < 1; ++v382_i1) {
                  int32_t v386_a = 1 + (v382_i1 * 2);
                  int32_t v392_a = 32_i32 + ((v382_i1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v383_i2 = 0; v383_i2 < 6; ++v383_i2) {
                    int32_t v387_a = v386_a + (v383_i2 * 2);
                    int32_t v393_a = v392_a + (v383_i2 * 832);
                    None.copy_to(glb_m2[v393_a]);
                  }
                }
              }
            }
          }
        }
      });
    }
  });
}


// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8a03a3cd0d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8a03a3cd0d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
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
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              ;
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir0[8]{};
              int32_t v14_lead = item.get_local_id(0) % 16;
              if (v14_lead < 12) {
                int32_t v17_a = 0_i32 + 0;
                int32_t v19_a = 0_i32 + 0;
                int32_t v21_a = 0_i32 + 0;
                int32_t v23_a = 0_i32 + 0;
                int32_t v25_a = 0_i32 + 0;
                int32_t v27_a = 0_i32 + 0;
                int32_t v29_a = 0_i32 + 0;
                int32_t v31_a = 0_i32 + 0;
              }
              if (v14_lead < 12) {
                int32_t v37_a = 0_i32 + 12;
                int32_t v39_a = 0_i32 + 12;
                int32_t v41_a = 0_i32 + 12;
                int32_t v43_a = 0_i32 + 12;
                int32_t v45_a = 0_i32 + 12;
                int32_t v47_a = 0_i32 + 12;
                int32_t v49_a = 0_i32 + 12;
                int32_t v51_a = 0_i32 + 12;
              }
              if (v14_lead < 12) {
                int32_t v57_a = 0_i32 + 24;
                int32_t v59_a = 0_i32 + 24;
                int32_t v61_a = 0_i32 + 24;
                int32_t v63_a = 0_i32 + 24;
                int32_t v65_a = 0_i32 + 24;
                int32_t v67_a = 0_i32 + 24;
                int32_t v69_a = 0_i32 + 24;
                int32_t v71_a = 0_i32 + 24;
              }
              if (v14_lead < 12) {
                int32_t v77_a = 0_i32 + 36;
                int32_t v79_a = 0_i32 + 36;
                int32_t v81_a = 0_i32 + 36;
                int32_t v83_a = 0_i32 + 36;
                int32_t v85_a = 0_i32 + 36;
                int32_t v87_a = 0_i32 + 36;
                int32_t v89_a = 0_i32 + 36;
                int32_t v91_a = 0_i32 + 36;
              }
              if (v14_lead < 12) {
                int32_t v97_a = 0_i32 + 48;
                int32_t v99_a = 0_i32 + 48;
                int32_t v101_a = 0_i32 + 48;
                int32_t v103_a = 0_i32 + 48;
                int32_t v105_a = 0_i32 + 48;
                int32_t v107_a = 0_i32 + 48;
                int32_t v109_a = 0_i32 + 48;
                int32_t v111_a = 0_i32 + 48;
              }
              if (v14_lead < 12) {
                int32_t v117_a = 0_i32 + 60;
                int32_t v119_a = 0_i32 + 60;
                int32_t v121_a = 0_i32 + 60;
                int32_t v123_a = 0_i32 + 60;
                int32_t v125_a = 0_i32 + 60;
                int32_t v127_a = 0_i32 + 60;
                int32_t v129_a = 0_i32 + 60;
                int32_t v131_a = 0_i32 + 60;
              }
              if (v14_lead < 12) {
                int32_t v137_a = 0_i32 + 72;
                int32_t v139_a = 0_i32 + 72;
                int32_t v141_a = 0_i32 + 72;
                int32_t v143_a = 0_i32 + 72;
                int32_t v145_a = 0_i32 + 72;
                int32_t v147_a = 0_i32 + 72;
                int32_t v149_a = 0_i32 + 72;
                int32_t v151_a = 0_i32 + 72;
              }
              if (v14_lead < 12) {
                int32_t v157_a = 0_i32 + 84;
                int32_t v159_a = 0_i32 + 84;
                int32_t v161_a = 0_i32 + 84;
                int32_t v163_a = 0_i32 + 84;
                int32_t v165_a = 0_i32 + 84;
                int32_t v167_a = 0_i32 + 84;
                int32_t v169_a = 0_i32 + 84;
                int32_t v171_a = 0_i32 + 84;
              }
              if (v14_lead < 12) {
                int32_t v177_a = 0_i32 + 96;
                int32_t v179_a = 0_i32 + 96;
                int32_t v181_a = 0_i32 + 96;
                int32_t v183_a = 0_i32 + 96;
                int32_t v185_a = 0_i32 + 96;
                int32_t v187_a = 0_i32 + 96;
                int32_t v189_a = 0_i32 + 96;
                int32_t v191_a = 0_i32 + 96;
              }
              if (v14_lead < 12) {
                int32_t v197_a = 0_i32 + 108;
                int32_t v199_a = 0_i32 + 108;
                int32_t v201_a = 0_i32 + 108;
                int32_t v203_a = 0_i32 + 108;
                int32_t v205_a = 0_i32 + 108;
                int32_t v207_a = 0_i32 + 108;
                int32_t v209_a = 0_i32 + 108;
                int32_t v211_a = 0_i32 + 108;
              }
              if (v14_lead < 12) {
                int32_t v217_a = 0_i32 + 120;
                int32_t v219_a = 0_i32 + 120;
                int32_t v221_a = 0_i32 + 120;
                int32_t v223_a = 0_i32 + 120;
                int32_t v225_a = 0_i32 + 120;
                int32_t v227_a = 0_i32 + 120;
                int32_t v229_a = 0_i32 + 120;
                int32_t v231_a = 0_i32 + 120;
              }
              if (v14_lead < 12) {
                int32_t v237_a = 0_i32 + 132;
                int32_t v239_a = 0_i32 + 132;
                int32_t v241_a = 0_i32 + 132;
                int32_t v243_a = 0_i32 + 132;
                int32_t v245_a = 0_i32 + 132;
                int32_t v247_a = 0_i32 + 132;
                int32_t v249_a = 0_i32 + 132;
                int32_t v251_a = 0_i32 + 132;
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v256_n1 = 0; v256_n1 < 8; ++v256_n1) {
                  int32_t v257_a = 0 + v256_n1;
                  int32_t v258_a = 0 + v256_n1;
                  None = r0[v258_a];
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v263_i1 = 0; v263_i1 < 8; ++v263_i1) {
                  int32_t v264_a = 0 + v263_i1;
                  int32_t v267_a = 0_i32 + (v263_i1 * 12);
                  None.copy_to(glb_m0[v267_a]);
                }
              }
              ;
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m4[0, 1])
              *(sycl::vec<float, 4>*)&s1[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m4[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s1[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m4[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r1[8]{};
              ;
              // r1 = +(glb_m3 * s1) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir1[8]{};
              if (v14_lead < 12) {
                int32_t v276_a = 0_i32 + 0;
                int32_t v278_a = 0_i32 + 0;
                int32_t v280_a = 0_i32 + 0;
                int32_t v282_a = 0_i32 + 0;
                int32_t v284_a = 0_i32 + 0;
                int32_t v286_a = 0_i32 + 0;
                int32_t v288_a = 0_i32 + 0;
                int32_t v290_a = 0_i32 + 0;
              }
              if (v14_lead < 12) {
                int32_t v296_a = 0_i32 + 12;
                int32_t v298_a = 0_i32 + 12;
                int32_t v300_a = 0_i32 + 12;
                int32_t v302_a = 0_i32 + 12;
                int32_t v304_a = 0_i32 + 12;
                int32_t v306_a = 0_i32 + 12;
                int32_t v308_a = 0_i32 + 12;
                int32_t v310_a = 0_i32 + 12;
              }
              if (v14_lead < 12) {
                int32_t v316_a = 0_i32 + 24;
                int32_t v318_a = 0_i32 + 24;
                int32_t v320_a = 0_i32 + 24;
                int32_t v322_a = 0_i32 + 24;
                int32_t v324_a = 0_i32 + 24;
                int32_t v326_a = 0_i32 + 24;
                int32_t v328_a = 0_i32 + 24;
                int32_t v330_a = 0_i32 + 24;
              }
              if (v14_lead < 12) {
                int32_t v336_a = 0_i32 + 36;
                int32_t v338_a = 0_i32 + 36;
                int32_t v340_a = 0_i32 + 36;
                int32_t v342_a = 0_i32 + 36;
                int32_t v344_a = 0_i32 + 36;
                int32_t v346_a = 0_i32 + 36;
                int32_t v348_a = 0_i32 + 36;
                int32_t v350_a = 0_i32 + 36;
              }
              if (v14_lead < 12) {
                int32_t v356_a = 0_i32 + 48;
                int32_t v358_a = 0_i32 + 48;
                int32_t v360_a = 0_i32 + 48;
                int32_t v362_a = 0_i32 + 48;
                int32_t v364_a = 0_i32 + 48;
                int32_t v366_a = 0_i32 + 48;
                int32_t v368_a = 0_i32 + 48;
                int32_t v370_a = 0_i32 + 48;
              }
              if (v14_lead < 12) {
                int32_t v376_a = 0_i32 + 60;
                int32_t v378_a = 0_i32 + 60;
                int32_t v380_a = 0_i32 + 60;
                int32_t v382_a = 0_i32 + 60;
                int32_t v384_a = 0_i32 + 60;
                int32_t v386_a = 0_i32 + 60;
                int32_t v388_a = 0_i32 + 60;
                int32_t v390_a = 0_i32 + 60;
              }
              if (v14_lead < 12) {
                int32_t v396_a = 0_i32 + 72;
                int32_t v398_a = 0_i32 + 72;
                int32_t v400_a = 0_i32 + 72;
                int32_t v402_a = 0_i32 + 72;
                int32_t v404_a = 0_i32 + 72;
                int32_t v406_a = 0_i32 + 72;
                int32_t v408_a = 0_i32 + 72;
                int32_t v410_a = 0_i32 + 72;
              }
              if (v14_lead < 12) {
                int32_t v416_a = 0_i32 + 84;
                int32_t v418_a = 0_i32 + 84;
                int32_t v420_a = 0_i32 + 84;
                int32_t v422_a = 0_i32 + 84;
                int32_t v424_a = 0_i32 + 84;
                int32_t v426_a = 0_i32 + 84;
                int32_t v428_a = 0_i32 + 84;
                int32_t v430_a = 0_i32 + 84;
              }
              if (v14_lead < 12) {
                int32_t v436_a = 0_i32 + 96;
                int32_t v438_a = 0_i32 + 96;
                int32_t v440_a = 0_i32 + 96;
                int32_t v442_a = 0_i32 + 96;
                int32_t v444_a = 0_i32 + 96;
                int32_t v446_a = 0_i32 + 96;
                int32_t v448_a = 0_i32 + 96;
                int32_t v450_a = 0_i32 + 96;
              }
              if (v14_lead < 12) {
                int32_t v456_a = 0_i32 + 108;
                int32_t v458_a = 0_i32 + 108;
                int32_t v460_a = 0_i32 + 108;
                int32_t v462_a = 0_i32 + 108;
                int32_t v464_a = 0_i32 + 108;
                int32_t v466_a = 0_i32 + 108;
                int32_t v468_a = 0_i32 + 108;
                int32_t v470_a = 0_i32 + 108;
              }
              if (v14_lead < 12) {
                int32_t v476_a = 0_i32 + 120;
                int32_t v478_a = 0_i32 + 120;
                int32_t v480_a = 0_i32 + 120;
                int32_t v482_a = 0_i32 + 120;
                int32_t v484_a = 0_i32 + 120;
                int32_t v486_a = 0_i32 + 120;
                int32_t v488_a = 0_i32 + 120;
                int32_t v490_a = 0_i32 + 120;
              }
              if (v14_lead < 12) {
                int32_t v496_a = 0_i32 + 132;
                int32_t v498_a = 0_i32 + 132;
                int32_t v500_a = 0_i32 + 132;
                int32_t v502_a = 0_i32 + 132;
                int32_t v504_a = 0_i32 + 132;
                int32_t v506_a = 0_i32 + 132;
                int32_t v508_a = 0_i32 + 132;
                int32_t v510_a = 0_i32 + 132;
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v515_n1 = 0; v515_n1 < 8; ++v515_n1) {
                  int32_t v516_a = 0 + v515_n1;
                  int32_t v519_a = 0_i32 + (v515_n1 * 12);
                  int32_t v521_a = 0 + v515_n1;
                  v520_p = r1[v521_a];
                }
              }
              // glb_m0 = store{r>g}(r1);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v526_i1 = 0; v526_i1 < 8; ++v526_i1) {
                  int32_t v527_a = 0 + v526_i1;
                  int32_t v530_a = 0_i32 + (v526_i1 * 12);
                  None.copy_to(glb_m0[v530_a]);
                }
              }
              ;
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m6[0, 1])
              *(sycl::vec<float, 4>*)&s2[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m6[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s2[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m6[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              float r2[8]{};
              ;
              // r2 = +(glb_m5 * s2) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              if (v14_lead < 12) {
                int32_t v539_a = 0_i32 + 0;
                int32_t v541_a = 0_i32 + 0;
                int32_t v543_a = 0_i32 + 0;
                int32_t v545_a = 0_i32 + 0;
                int32_t v547_a = 0_i32 + 0;
                int32_t v549_a = 0_i32 + 0;
                int32_t v551_a = 0_i32 + 0;
                int32_t v553_a = 0_i32 + 0;
              }
              if (v14_lead < 12) {
                int32_t v559_a = 0_i32 + 12;
                int32_t v561_a = 0_i32 + 12;
                int32_t v563_a = 0_i32 + 12;
                int32_t v565_a = 0_i32 + 12;
                int32_t v567_a = 0_i32 + 12;
                int32_t v569_a = 0_i32 + 12;
                int32_t v571_a = 0_i32 + 12;
                int32_t v573_a = 0_i32 + 12;
              }
              if (v14_lead < 12) {
                int32_t v579_a = 0_i32 + 24;
                int32_t v581_a = 0_i32 + 24;
                int32_t v583_a = 0_i32 + 24;
                int32_t v585_a = 0_i32 + 24;
                int32_t v587_a = 0_i32 + 24;
                int32_t v589_a = 0_i32 + 24;
                int32_t v591_a = 0_i32 + 24;
                int32_t v593_a = 0_i32 + 24;
              }
              if (v14_lead < 12) {
                int32_t v599_a = 0_i32 + 36;
                int32_t v601_a = 0_i32 + 36;
                int32_t v603_a = 0_i32 + 36;
                int32_t v605_a = 0_i32 + 36;
                int32_t v607_a = 0_i32 + 36;
                int32_t v609_a = 0_i32 + 36;
                int32_t v611_a = 0_i32 + 36;
                int32_t v613_a = 0_i32 + 36;
              }
              if (v14_lead < 12) {
                int32_t v619_a = 0_i32 + 48;
                int32_t v621_a = 0_i32 + 48;
                int32_t v623_a = 0_i32 + 48;
                int32_t v625_a = 0_i32 + 48;
                int32_t v627_a = 0_i32 + 48;
                int32_t v629_a = 0_i32 + 48;
                int32_t v631_a = 0_i32 + 48;
                int32_t v633_a = 0_i32 + 48;
              }
              if (v14_lead < 12) {
                int32_t v639_a = 0_i32 + 60;
                int32_t v641_a = 0_i32 + 60;
                int32_t v643_a = 0_i32 + 60;
                int32_t v645_a = 0_i32 + 60;
                int32_t v647_a = 0_i32 + 60;
                int32_t v649_a = 0_i32 + 60;
                int32_t v651_a = 0_i32 + 60;
                int32_t v653_a = 0_i32 + 60;
              }
              if (v14_lead < 12) {
                int32_t v659_a = 0_i32 + 72;
                int32_t v661_a = 0_i32 + 72;
                int32_t v663_a = 0_i32 + 72;
                int32_t v665_a = 0_i32 + 72;
                int32_t v667_a = 0_i32 + 72;
                int32_t v669_a = 0_i32 + 72;
                int32_t v671_a = 0_i32 + 72;
                int32_t v673_a = 0_i32 + 72;
              }
              if (v14_lead < 12) {
                int32_t v679_a = 0_i32 + 84;
                int32_t v681_a = 0_i32 + 84;
                int32_t v683_a = 0_i32 + 84;
                int32_t v685_a = 0_i32 + 84;
                int32_t v687_a = 0_i32 + 84;
                int32_t v689_a = 0_i32 + 84;
                int32_t v691_a = 0_i32 + 84;
                int32_t v693_a = 0_i32 + 84;
              }
              if (v14_lead < 12) {
                int32_t v699_a = 0_i32 + 96;
                int32_t v701_a = 0_i32 + 96;
                int32_t v703_a = 0_i32 + 96;
                int32_t v705_a = 0_i32 + 96;
                int32_t v707_a = 0_i32 + 96;
                int32_t v709_a = 0_i32 + 96;
                int32_t v711_a = 0_i32 + 96;
                int32_t v713_a = 0_i32 + 96;
              }
              if (v14_lead < 12) {
                int32_t v719_a = 0_i32 + 108;
                int32_t v721_a = 0_i32 + 108;
                int32_t v723_a = 0_i32 + 108;
                int32_t v725_a = 0_i32 + 108;
                int32_t v727_a = 0_i32 + 108;
                int32_t v729_a = 0_i32 + 108;
                int32_t v731_a = 0_i32 + 108;
                int32_t v733_a = 0_i32 + 108;
              }
              if (v14_lead < 12) {
                int32_t v739_a = 0_i32 + 120;
                int32_t v741_a = 0_i32 + 120;
                int32_t v743_a = 0_i32 + 120;
                int32_t v745_a = 0_i32 + 120;
                int32_t v747_a = 0_i32 + 120;
                int32_t v749_a = 0_i32 + 120;
                int32_t v751_a = 0_i32 + 120;
                int32_t v753_a = 0_i32 + 120;
              }
              if (v14_lead < 12) {
                int32_t v759_a = 0_i32 + 132;
                int32_t v761_a = 0_i32 + 132;
                int32_t v763_a = 0_i32 + 132;
                int32_t v765_a = 0_i32 + 132;
                int32_t v767_a = 0_i32 + 132;
                int32_t v769_a = 0_i32 + 132;
                int32_t v771_a = 0_i32 + 132;
                int32_t v773_a = 0_i32 + 132;
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v778_n1 = 0; v778_n1 < 8; ++v778_n1) {
                  int32_t v779_a = 0 + v778_n1;
                  int32_t v782_a = 0_i32 + (v778_n1 * 12);
                  int32_t v784_a = 0 + v778_n1;
                  v783_p = r2[v784_a];
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v789_i1 = 0; v789_i1 < 8; ++v789_i1) {
                  int32_t v790_a = 0 + v789_i1;
                  int32_t v793_a = 0_i32 + (v789_i1 * 12);
                  None.copy_to(glb_m0[v793_a]);
                }
              }
              ;
              float* __restrict__ s3 = &localShrMem0[0];
              // s3 = load{g>s}(glb_m8[0, 1])
              *(sycl::vec<float, 4>*)&s3[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m8[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s3[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m8[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              float r3[8]{};
              ;
              // r3 = +(glb_m7 * s3) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir3[8]{};
              if (v14_lead < 12) {
                int32_t v802_a = 0_i32 + 0;
                int32_t v804_a = 0_i32 + 0;
                int32_t v806_a = 0_i32 + 0;
                int32_t v808_a = 0_i32 + 0;
                int32_t v810_a = 0_i32 + 0;
                int32_t v812_a = 0_i32 + 0;
                int32_t v814_a = 0_i32 + 0;
                int32_t v816_a = 0_i32 + 0;
              }
              if (v14_lead < 12) {
                int32_t v822_a = 0_i32 + 12;
                int32_t v824_a = 0_i32 + 12;
                int32_t v826_a = 0_i32 + 12;
                int32_t v828_a = 0_i32 + 12;
                int32_t v830_a = 0_i32 + 12;
                int32_t v832_a = 0_i32 + 12;
                int32_t v834_a = 0_i32 + 12;
                int32_t v836_a = 0_i32 + 12;
              }
              if (v14_lead < 12) {
                int32_t v842_a = 0_i32 + 24;
                int32_t v844_a = 0_i32 + 24;
                int32_t v846_a = 0_i32 + 24;
                int32_t v848_a = 0_i32 + 24;
                int32_t v850_a = 0_i32 + 24;
                int32_t v852_a = 0_i32 + 24;
                int32_t v854_a = 0_i32 + 24;
                int32_t v856_a = 0_i32 + 24;
              }
              if (v14_lead < 12) {
                int32_t v862_a = 0_i32 + 36;
                int32_t v864_a = 0_i32 + 36;
                int32_t v866_a = 0_i32 + 36;
                int32_t v868_a = 0_i32 + 36;
                int32_t v870_a = 0_i32 + 36;
                int32_t v872_a = 0_i32 + 36;
                int32_t v874_a = 0_i32 + 36;
                int32_t v876_a = 0_i32 + 36;
              }
              if (v14_lead < 12) {
                int32_t v882_a = 0_i32 + 48;
                int32_t v884_a = 0_i32 + 48;
                int32_t v886_a = 0_i32 + 48;
                int32_t v888_a = 0_i32 + 48;
                int32_t v890_a = 0_i32 + 48;
                int32_t v892_a = 0_i32 + 48;
                int32_t v894_a = 0_i32 + 48;
                int32_t v896_a = 0_i32 + 48;
              }
              if (v14_lead < 12) {
                int32_t v902_a = 0_i32 + 60;
                int32_t v904_a = 0_i32 + 60;
                int32_t v906_a = 0_i32 + 60;
                int32_t v908_a = 0_i32 + 60;
                int32_t v910_a = 0_i32 + 60;
                int32_t v912_a = 0_i32 + 60;
                int32_t v914_a = 0_i32 + 60;
                int32_t v916_a = 0_i32 + 60;
              }
              if (v14_lead < 12) {
                int32_t v922_a = 0_i32 + 72;
                int32_t v924_a = 0_i32 + 72;
                int32_t v926_a = 0_i32 + 72;
                int32_t v928_a = 0_i32 + 72;
                int32_t v930_a = 0_i32 + 72;
                int32_t v932_a = 0_i32 + 72;
                int32_t v934_a = 0_i32 + 72;
                int32_t v936_a = 0_i32 + 72;
              }
              if (v14_lead < 12) {
                int32_t v942_a = 0_i32 + 84;
                int32_t v944_a = 0_i32 + 84;
                int32_t v946_a = 0_i32 + 84;
                int32_t v948_a = 0_i32 + 84;
                int32_t v950_a = 0_i32 + 84;
                int32_t v952_a = 0_i32 + 84;
                int32_t v954_a = 0_i32 + 84;
                int32_t v956_a = 0_i32 + 84;
              }
              if (v14_lead < 12) {
                int32_t v962_a = 0_i32 + 96;
                int32_t v964_a = 0_i32 + 96;
                int32_t v966_a = 0_i32 + 96;
                int32_t v968_a = 0_i32 + 96;
                int32_t v970_a = 0_i32 + 96;
                int32_t v972_a = 0_i32 + 96;
                int32_t v974_a = 0_i32 + 96;
                int32_t v976_a = 0_i32 + 96;
              }
              if (v14_lead < 12) {
                int32_t v982_a = 0_i32 + 108;
                int32_t v984_a = 0_i32 + 108;
                int32_t v986_a = 0_i32 + 108;
                int32_t v988_a = 0_i32 + 108;
                int32_t v990_a = 0_i32 + 108;
                int32_t v992_a = 0_i32 + 108;
                int32_t v994_a = 0_i32 + 108;
                int32_t v996_a = 0_i32 + 108;
              }
              if (v14_lead < 12) {
                int32_t v1002_a = 0_i32 + 120;
                int32_t v1004_a = 0_i32 + 120;
                int32_t v1006_a = 0_i32 + 120;
                int32_t v1008_a = 0_i32 + 120;
                int32_t v1010_a = 0_i32 + 120;
                int32_t v1012_a = 0_i32 + 120;
                int32_t v1014_a = 0_i32 + 120;
                int32_t v1016_a = 0_i32 + 120;
              }
              if (v14_lead < 12) {
                int32_t v1022_a = 0_i32 + 132;
                int32_t v1024_a = 0_i32 + 132;
                int32_t v1026_a = 0_i32 + 132;
                int32_t v1028_a = 0_i32 + 132;
                int32_t v1030_a = 0_i32 + 132;
                int32_t v1032_a = 0_i32 + 132;
                int32_t v1034_a = 0_i32 + 132;
                int32_t v1036_a = 0_i32 + 132;
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1041_n1 = 0; v1041_n1 < 8; ++v1041_n1) {
                  int32_t v1042_a = 0 + v1041_n1;
                  int32_t v1045_a = 0_i32 + (v1041_n1 * 12);
                  int32_t v1047_a = 0 + v1041_n1;
                  v1046_p = r3[v1047_a];
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1052_i1 = 0; v1052_i1 < 8; ++v1052_i1) {
                  int32_t v1053_a = 0 + v1052_i1;
                  int32_t v1056_a = 0_i32 + (v1052_i1 * 12);
                  None.copy_to(glb_m0[v1056_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}


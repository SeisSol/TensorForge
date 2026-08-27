// === base name ===
kernel_151d4e8604

// === header ===
void launcher_kernel_151d4e8604(float* m0, unsigned m0_extraOffset, const float* m1, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_151d4e8604(float* m0, unsigned m0_extraOffset, const float* m1, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_151d4e8604(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_151d4e8604(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (4352, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} none
        // m2 16×16(16×16) {0..16}×{0..16} strided
        // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} none({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[272 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[256];
          const float *const __restrict__ glb_m1 = &m1[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 128] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 128];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 192] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 192];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[16]{};
              ;
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir0[16]{};
              int32_t v10_a = 0_i32 + 0;
              int32_t v12_a = 0_i32 + 0;
              int32_t v14_a = 0_i32 + 0;
              int32_t v16_a = 0_i32 + 0;
              int32_t v18_a = 0_i32 + 0;
              int32_t v20_a = 0_i32 + 0;
              int32_t v22_a = 0_i32 + 0;
              int32_t v24_a = 0_i32 + 0;
              int32_t v26_a = 0_i32 + 0;
              int32_t v28_a = 0_i32 + 0;
              int32_t v30_a = 0_i32 + 0;
              int32_t v32_a = 0_i32 + 0;
              int32_t v34_a = 0_i32 + 0;
              int32_t v36_a = 0_i32 + 0;
              int32_t v38_a = 0_i32 + 0;
              int32_t v40_a = 0_i32 + 0;
              int32_t v45_a = 0_i32 + 16;
              int32_t v47_a = 0_i32 + 16;
              int32_t v49_a = 0_i32 + 16;
              int32_t v51_a = 0_i32 + 16;
              int32_t v53_a = 0_i32 + 16;
              int32_t v55_a = 0_i32 + 16;
              int32_t v57_a = 0_i32 + 16;
              int32_t v59_a = 0_i32 + 16;
              int32_t v61_a = 0_i32 + 16;
              int32_t v63_a = 0_i32 + 16;
              int32_t v65_a = 0_i32 + 16;
              int32_t v67_a = 0_i32 + 16;
              int32_t v69_a = 0_i32 + 16;
              int32_t v71_a = 0_i32 + 16;
              int32_t v73_a = 0_i32 + 16;
              int32_t v75_a = 0_i32 + 16;
              int32_t v80_a = 0_i32 + 32;
              int32_t v82_a = 0_i32 + 32;
              int32_t v84_a = 0_i32 + 32;
              int32_t v86_a = 0_i32 + 32;
              int32_t v88_a = 0_i32 + 32;
              int32_t v90_a = 0_i32 + 32;
              int32_t v92_a = 0_i32 + 32;
              int32_t v94_a = 0_i32 + 32;
              int32_t v96_a = 0_i32 + 32;
              int32_t v98_a = 0_i32 + 32;
              int32_t v100_a = 0_i32 + 32;
              int32_t v102_a = 0_i32 + 32;
              int32_t v104_a = 0_i32 + 32;
              int32_t v106_a = 0_i32 + 32;
              int32_t v108_a = 0_i32 + 32;
              int32_t v110_a = 0_i32 + 32;
              int32_t v115_a = 0_i32 + 48;
              int32_t v117_a = 0_i32 + 48;
              int32_t v119_a = 0_i32 + 48;
              int32_t v121_a = 0_i32 + 48;
              int32_t v123_a = 0_i32 + 48;
              int32_t v125_a = 0_i32 + 48;
              int32_t v127_a = 0_i32 + 48;
              int32_t v129_a = 0_i32 + 48;
              int32_t v131_a = 0_i32 + 48;
              int32_t v133_a = 0_i32 + 48;
              int32_t v135_a = 0_i32 + 48;
              int32_t v137_a = 0_i32 + 48;
              int32_t v139_a = 0_i32 + 48;
              int32_t v141_a = 0_i32 + 48;
              int32_t v143_a = 0_i32 + 48;
              int32_t v145_a = 0_i32 + 48;
              int32_t v150_a = 0_i32 + 64;
              int32_t v152_a = 0_i32 + 64;
              int32_t v154_a = 0_i32 + 64;
              int32_t v156_a = 0_i32 + 64;
              int32_t v158_a = 0_i32 + 64;
              int32_t v160_a = 0_i32 + 64;
              int32_t v162_a = 0_i32 + 64;
              int32_t v164_a = 0_i32 + 64;
              int32_t v166_a = 0_i32 + 64;
              int32_t v168_a = 0_i32 + 64;
              int32_t v170_a = 0_i32 + 64;
              int32_t v172_a = 0_i32 + 64;
              int32_t v174_a = 0_i32 + 64;
              int32_t v176_a = 0_i32 + 64;
              int32_t v178_a = 0_i32 + 64;
              int32_t v180_a = 0_i32 + 64;
              int32_t v185_a = 0_i32 + 80;
              int32_t v187_a = 0_i32 + 80;
              int32_t v189_a = 0_i32 + 80;
              int32_t v191_a = 0_i32 + 80;
              int32_t v193_a = 0_i32 + 80;
              int32_t v195_a = 0_i32 + 80;
              int32_t v197_a = 0_i32 + 80;
              int32_t v199_a = 0_i32 + 80;
              int32_t v201_a = 0_i32 + 80;
              int32_t v203_a = 0_i32 + 80;
              int32_t v205_a = 0_i32 + 80;
              int32_t v207_a = 0_i32 + 80;
              int32_t v209_a = 0_i32 + 80;
              int32_t v211_a = 0_i32 + 80;
              int32_t v213_a = 0_i32 + 80;
              int32_t v215_a = 0_i32 + 80;
              int32_t v220_a = 0_i32 + 96;
              int32_t v222_a = 0_i32 + 96;
              int32_t v224_a = 0_i32 + 96;
              int32_t v226_a = 0_i32 + 96;
              int32_t v228_a = 0_i32 + 96;
              int32_t v230_a = 0_i32 + 96;
              int32_t v232_a = 0_i32 + 96;
              int32_t v234_a = 0_i32 + 96;
              int32_t v236_a = 0_i32 + 96;
              int32_t v238_a = 0_i32 + 96;
              int32_t v240_a = 0_i32 + 96;
              int32_t v242_a = 0_i32 + 96;
              int32_t v244_a = 0_i32 + 96;
              int32_t v246_a = 0_i32 + 96;
              int32_t v248_a = 0_i32 + 96;
              int32_t v250_a = 0_i32 + 96;
              int32_t v255_a = 0_i32 + 112;
              int32_t v257_a = 0_i32 + 112;
              int32_t v259_a = 0_i32 + 112;
              int32_t v261_a = 0_i32 + 112;
              int32_t v263_a = 0_i32 + 112;
              int32_t v265_a = 0_i32 + 112;
              int32_t v267_a = 0_i32 + 112;
              int32_t v269_a = 0_i32 + 112;
              int32_t v271_a = 0_i32 + 112;
              int32_t v273_a = 0_i32 + 112;
              int32_t v275_a = 0_i32 + 112;
              int32_t v277_a = 0_i32 + 112;
              int32_t v279_a = 0_i32 + 112;
              int32_t v281_a = 0_i32 + 112;
              int32_t v283_a = 0_i32 + 112;
              int32_t v285_a = 0_i32 + 112;
              int32_t v290_a = 0_i32 + 128;
              int32_t v292_a = 0_i32 + 128;
              int32_t v294_a = 0_i32 + 128;
              int32_t v296_a = 0_i32 + 128;
              int32_t v298_a = 0_i32 + 128;
              int32_t v300_a = 0_i32 + 128;
              int32_t v302_a = 0_i32 + 128;
              int32_t v304_a = 0_i32 + 128;
              int32_t v306_a = 0_i32 + 128;
              int32_t v308_a = 0_i32 + 128;
              int32_t v310_a = 0_i32 + 128;
              int32_t v312_a = 0_i32 + 128;
              int32_t v314_a = 0_i32 + 128;
              int32_t v316_a = 0_i32 + 128;
              int32_t v318_a = 0_i32 + 128;
              int32_t v320_a = 0_i32 + 128;
              int32_t v325_a = 0_i32 + 144;
              int32_t v327_a = 0_i32 + 144;
              int32_t v329_a = 0_i32 + 144;
              int32_t v331_a = 0_i32 + 144;
              int32_t v333_a = 0_i32 + 144;
              int32_t v335_a = 0_i32 + 144;
              int32_t v337_a = 0_i32 + 144;
              int32_t v339_a = 0_i32 + 144;
              int32_t v341_a = 0_i32 + 144;
              int32_t v343_a = 0_i32 + 144;
              int32_t v345_a = 0_i32 + 144;
              int32_t v347_a = 0_i32 + 144;
              int32_t v349_a = 0_i32 + 144;
              int32_t v351_a = 0_i32 + 144;
              int32_t v353_a = 0_i32 + 144;
              int32_t v355_a = 0_i32 + 144;
              int32_t v360_a = 0_i32 + 160;
              int32_t v362_a = 0_i32 + 160;
              int32_t v364_a = 0_i32 + 160;
              int32_t v366_a = 0_i32 + 160;
              int32_t v368_a = 0_i32 + 160;
              int32_t v370_a = 0_i32 + 160;
              int32_t v372_a = 0_i32 + 160;
              int32_t v374_a = 0_i32 + 160;
              int32_t v376_a = 0_i32 + 160;
              int32_t v378_a = 0_i32 + 160;
              int32_t v380_a = 0_i32 + 160;
              int32_t v382_a = 0_i32 + 160;
              int32_t v384_a = 0_i32 + 160;
              int32_t v386_a = 0_i32 + 160;
              int32_t v388_a = 0_i32 + 160;
              int32_t v390_a = 0_i32 + 160;
              int32_t v395_a = 0_i32 + 176;
              int32_t v397_a = 0_i32 + 176;
              int32_t v399_a = 0_i32 + 176;
              int32_t v401_a = 0_i32 + 176;
              int32_t v403_a = 0_i32 + 176;
              int32_t v405_a = 0_i32 + 176;
              int32_t v407_a = 0_i32 + 176;
              int32_t v409_a = 0_i32 + 176;
              int32_t v411_a = 0_i32 + 176;
              int32_t v413_a = 0_i32 + 176;
              int32_t v415_a = 0_i32 + 176;
              int32_t v417_a = 0_i32 + 176;
              int32_t v419_a = 0_i32 + 176;
              int32_t v421_a = 0_i32 + 176;
              int32_t v423_a = 0_i32 + 176;
              int32_t v425_a = 0_i32 + 176;
              int32_t v430_a = 0_i32 + 192;
              int32_t v432_a = 0_i32 + 192;
              int32_t v434_a = 0_i32 + 192;
              int32_t v436_a = 0_i32 + 192;
              int32_t v438_a = 0_i32 + 192;
              int32_t v440_a = 0_i32 + 192;
              int32_t v442_a = 0_i32 + 192;
              int32_t v444_a = 0_i32 + 192;
              int32_t v446_a = 0_i32 + 192;
              int32_t v448_a = 0_i32 + 192;
              int32_t v450_a = 0_i32 + 192;
              int32_t v452_a = 0_i32 + 192;
              int32_t v454_a = 0_i32 + 192;
              int32_t v456_a = 0_i32 + 192;
              int32_t v458_a = 0_i32 + 192;
              int32_t v460_a = 0_i32 + 192;
              int32_t v465_a = 0_i32 + 208;
              int32_t v467_a = 0_i32 + 208;
              int32_t v469_a = 0_i32 + 208;
              int32_t v471_a = 0_i32 + 208;
              int32_t v473_a = 0_i32 + 208;
              int32_t v475_a = 0_i32 + 208;
              int32_t v477_a = 0_i32 + 208;
              int32_t v479_a = 0_i32 + 208;
              int32_t v481_a = 0_i32 + 208;
              int32_t v483_a = 0_i32 + 208;
              int32_t v485_a = 0_i32 + 208;
              int32_t v487_a = 0_i32 + 208;
              int32_t v489_a = 0_i32 + 208;
              int32_t v491_a = 0_i32 + 208;
              int32_t v493_a = 0_i32 + 208;
              int32_t v495_a = 0_i32 + 208;
              int32_t v500_a = 0_i32 + 224;
              int32_t v502_a = 0_i32 + 224;
              int32_t v504_a = 0_i32 + 224;
              int32_t v506_a = 0_i32 + 224;
              int32_t v508_a = 0_i32 + 224;
              int32_t v510_a = 0_i32 + 224;
              int32_t v512_a = 0_i32 + 224;
              int32_t v514_a = 0_i32 + 224;
              int32_t v516_a = 0_i32 + 224;
              int32_t v518_a = 0_i32 + 224;
              int32_t v520_a = 0_i32 + 224;
              int32_t v522_a = 0_i32 + 224;
              int32_t v524_a = 0_i32 + 224;
              int32_t v526_a = 0_i32 + 224;
              int32_t v528_a = 0_i32 + 224;
              int32_t v530_a = 0_i32 + 224;
              int32_t v535_a = 0_i32 + 240;
              int32_t v537_a = 0_i32 + 240;
              int32_t v539_a = 0_i32 + 240;
              int32_t v541_a = 0_i32 + 240;
              int32_t v543_a = 0_i32 + 240;
              int32_t v545_a = 0_i32 + 240;
              int32_t v547_a = 0_i32 + 240;
              int32_t v549_a = 0_i32 + 240;
              int32_t v551_a = 0_i32 + 240;
              int32_t v553_a = 0_i32 + 240;
              int32_t v555_a = 0_i32 + 240;
              int32_t v557_a = 0_i32 + 240;
              int32_t v559_a = 0_i32 + 240;
              int32_t v561_a = 0_i32 + 240;
              int32_t v563_a = 0_i32 + 240;
              int32_t v565_a = 0_i32 + 240;
              #pragma unroll
              for (int32_t v569_n0 = 0; v569_n0 < 1; ++v569_n0) {
                #pragma unroll
                for (int32_t v570_n1 = 0; v570_n1 < 16; ++v570_n1) {
                  int32_t v571_a = v569_n0 + v570_n1;
                  int32_t v572_a = v569_n0 + v570_n1;
                  None = r0[v572_a];
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v576_i0 = 0; v576_i0 < 1; ++v576_i0) {
                int32_t v579_lead = v576_i0 * 16;
                #pragma unroll
                for (int32_t v577_i1 = 0; v577_i1 < 16; ++v577_i1) {
                  int32_t v578_a = v576_i0 + v577_i1;
                  int32_t v581_a = v579_lead + (v577_i1 * 16);
                  None.copy_to(glb_m0[v581_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}


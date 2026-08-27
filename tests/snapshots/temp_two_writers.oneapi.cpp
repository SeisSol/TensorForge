// === base name ===
kernel_3e24e7feaf

// === header ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3e24e7feaf(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3e24e7feaf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×32(6×12) {0..6}×{0..12} strided
        // m1 32×32(12×12) {0..12}×{0..12} strided
        // m2 32×32(6×12) {0..6}×{0..12} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // m4 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m0 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
        // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m2 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
        // m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, 1] = m4 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..12}×{0..12})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m1[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m1[0 + 0 + 4 * item.get_local_id(0) + 64];
              s0[0 + 0 + 1 * item.get_local_id(0) + 128] = glb_m1[0 + 0 + 1 * item.get_local_id(0) + 128];
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r0[12]{};
              ;
              // r0 = +(glb_m0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              auto& ir0 = r0;
              int32_t v9_lead = item.get_local_id(0) % 16;
              if (v9_lead < 6) {
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
              }
              if (v9_lead < 6) {
                int32_t v40_a = 0_i32 + 6;
                int32_t v42_a = 0_i32 + 6;
                int32_t v44_a = 0_i32 + 6;
                int32_t v46_a = 0_i32 + 6;
                int32_t v48_a = 0_i32 + 6;
                int32_t v50_a = 0_i32 + 6;
                int32_t v52_a = 0_i32 + 6;
                int32_t v54_a = 0_i32 + 6;
                int32_t v56_a = 0_i32 + 6;
                int32_t v58_a = 0_i32 + 6;
                int32_t v60_a = 0_i32 + 6;
                int32_t v62_a = 0_i32 + 6;
              }
              if (v9_lead < 6) {
                int32_t v68_a = 0_i32 + 12;
                int32_t v70_a = 0_i32 + 12;
                int32_t v72_a = 0_i32 + 12;
                int32_t v74_a = 0_i32 + 12;
                int32_t v76_a = 0_i32 + 12;
                int32_t v78_a = 0_i32 + 12;
                int32_t v80_a = 0_i32 + 12;
                int32_t v82_a = 0_i32 + 12;
                int32_t v84_a = 0_i32 + 12;
                int32_t v86_a = 0_i32 + 12;
                int32_t v88_a = 0_i32 + 12;
                int32_t v90_a = 0_i32 + 12;
              }
              if (v9_lead < 6) {
                int32_t v96_a = 0_i32 + 18;
                int32_t v98_a = 0_i32 + 18;
                int32_t v100_a = 0_i32 + 18;
                int32_t v102_a = 0_i32 + 18;
                int32_t v104_a = 0_i32 + 18;
                int32_t v106_a = 0_i32 + 18;
                int32_t v108_a = 0_i32 + 18;
                int32_t v110_a = 0_i32 + 18;
                int32_t v112_a = 0_i32 + 18;
                int32_t v114_a = 0_i32 + 18;
                int32_t v116_a = 0_i32 + 18;
                int32_t v118_a = 0_i32 + 18;
              }
              if (v9_lead < 6) {
                int32_t v124_a = 0_i32 + 24;
                int32_t v126_a = 0_i32 + 24;
                int32_t v128_a = 0_i32 + 24;
                int32_t v130_a = 0_i32 + 24;
                int32_t v132_a = 0_i32 + 24;
                int32_t v134_a = 0_i32 + 24;
                int32_t v136_a = 0_i32 + 24;
                int32_t v138_a = 0_i32 + 24;
                int32_t v140_a = 0_i32 + 24;
                int32_t v142_a = 0_i32 + 24;
                int32_t v144_a = 0_i32 + 24;
                int32_t v146_a = 0_i32 + 24;
              }
              if (v9_lead < 6) {
                int32_t v152_a = 0_i32 + 30;
                int32_t v154_a = 0_i32 + 30;
                int32_t v156_a = 0_i32 + 30;
                int32_t v158_a = 0_i32 + 30;
                int32_t v160_a = 0_i32 + 30;
                int32_t v162_a = 0_i32 + 30;
                int32_t v164_a = 0_i32 + 30;
                int32_t v166_a = 0_i32 + 30;
                int32_t v168_a = 0_i32 + 30;
                int32_t v170_a = 0_i32 + 30;
                int32_t v172_a = 0_i32 + 30;
                int32_t v174_a = 0_i32 + 30;
              }
              if (v9_lead < 6) {
                int32_t v180_a = 0_i32 + 36;
                int32_t v182_a = 0_i32 + 36;
                int32_t v184_a = 0_i32 + 36;
                int32_t v186_a = 0_i32 + 36;
                int32_t v188_a = 0_i32 + 36;
                int32_t v190_a = 0_i32 + 36;
                int32_t v192_a = 0_i32 + 36;
                int32_t v194_a = 0_i32 + 36;
                int32_t v196_a = 0_i32 + 36;
                int32_t v198_a = 0_i32 + 36;
                int32_t v200_a = 0_i32 + 36;
                int32_t v202_a = 0_i32 + 36;
              }
              if (v9_lead < 6) {
                int32_t v208_a = 0_i32 + 42;
                int32_t v210_a = 0_i32 + 42;
                int32_t v212_a = 0_i32 + 42;
                int32_t v214_a = 0_i32 + 42;
                int32_t v216_a = 0_i32 + 42;
                int32_t v218_a = 0_i32 + 42;
                int32_t v220_a = 0_i32 + 42;
                int32_t v222_a = 0_i32 + 42;
                int32_t v224_a = 0_i32 + 42;
                int32_t v226_a = 0_i32 + 42;
                int32_t v228_a = 0_i32 + 42;
                int32_t v230_a = 0_i32 + 42;
              }
              if (v9_lead < 6) {
                int32_t v236_a = 0_i32 + 48;
                int32_t v238_a = 0_i32 + 48;
                int32_t v240_a = 0_i32 + 48;
                int32_t v242_a = 0_i32 + 48;
                int32_t v244_a = 0_i32 + 48;
                int32_t v246_a = 0_i32 + 48;
                int32_t v248_a = 0_i32 + 48;
                int32_t v250_a = 0_i32 + 48;
                int32_t v252_a = 0_i32 + 48;
                int32_t v254_a = 0_i32 + 48;
                int32_t v256_a = 0_i32 + 48;
                int32_t v258_a = 0_i32 + 48;
              }
              if (v9_lead < 6) {
                int32_t v264_a = 0_i32 + 54;
                int32_t v266_a = 0_i32 + 54;
                int32_t v268_a = 0_i32 + 54;
                int32_t v270_a = 0_i32 + 54;
                int32_t v272_a = 0_i32 + 54;
                int32_t v274_a = 0_i32 + 54;
                int32_t v276_a = 0_i32 + 54;
                int32_t v278_a = 0_i32 + 54;
                int32_t v280_a = 0_i32 + 54;
                int32_t v282_a = 0_i32 + 54;
                int32_t v284_a = 0_i32 + 54;
                int32_t v286_a = 0_i32 + 54;
              }
              if (v9_lead < 6) {
                int32_t v292_a = 0_i32 + 60;
                int32_t v294_a = 0_i32 + 60;
                int32_t v296_a = 0_i32 + 60;
                int32_t v298_a = 0_i32 + 60;
                int32_t v300_a = 0_i32 + 60;
                int32_t v302_a = 0_i32 + 60;
                int32_t v304_a = 0_i32 + 60;
                int32_t v306_a = 0_i32 + 60;
                int32_t v308_a = 0_i32 + 60;
                int32_t v310_a = 0_i32 + 60;
                int32_t v312_a = 0_i32 + 60;
                int32_t v314_a = 0_i32 + 60;
              }
              if (v9_lead < 6) {
                int32_t v320_a = 0_i32 + 66;
                int32_t v322_a = 0_i32 + 66;
                int32_t v324_a = 0_i32 + 66;
                int32_t v326_a = 0_i32 + 66;
                int32_t v328_a = 0_i32 + 66;
                int32_t v330_a = 0_i32 + 66;
                int32_t v332_a = 0_i32 + 66;
                int32_t v334_a = 0_i32 + 66;
                int32_t v336_a = 0_i32 + 66;
                int32_t v338_a = 0_i32 + 66;
                int32_t v340_a = 0_i32 + 66;
                int32_t v342_a = 0_i32 + 66;
              }
              ;
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r0);
              if (v9_lead < 6) {
                #pragma unroll
                for (int32_t v348_i1 = 0; v348_i1 < 12; ++v348_i1) {
                  int32_t v349_a = 0 + v348_i1;
                  int32_t v352_a = 0_i32 + (v348_i1 * 12);
                  None = s1[v352_a];
                }
              }
              float r1[12]{};
              // r1 = +(glb_m2 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir1[12]{};
              if (v9_lead < 6) {
                int32_t v360_a = 0_i32 + 0;
                int32_t v362_a = 0_i32 + 0;
                int32_t v364_a = 0_i32 + 0;
                int32_t v366_a = 0_i32 + 0;
                int32_t v368_a = 0_i32 + 0;
                int32_t v370_a = 0_i32 + 0;
                int32_t v372_a = 0_i32 + 0;
                int32_t v374_a = 0_i32 + 0;
                int32_t v376_a = 0_i32 + 0;
                int32_t v378_a = 0_i32 + 0;
                int32_t v380_a = 0_i32 + 0;
                int32_t v382_a = 0_i32 + 0;
              }
              if (v9_lead < 6) {
                int32_t v388_a = 0_i32 + 6;
                int32_t v390_a = 0_i32 + 6;
                int32_t v392_a = 0_i32 + 6;
                int32_t v394_a = 0_i32 + 6;
                int32_t v396_a = 0_i32 + 6;
                int32_t v398_a = 0_i32 + 6;
                int32_t v400_a = 0_i32 + 6;
                int32_t v402_a = 0_i32 + 6;
                int32_t v404_a = 0_i32 + 6;
                int32_t v406_a = 0_i32 + 6;
                int32_t v408_a = 0_i32 + 6;
                int32_t v410_a = 0_i32 + 6;
              }
              if (v9_lead < 6) {
                int32_t v416_a = 0_i32 + 12;
                int32_t v418_a = 0_i32 + 12;
                int32_t v420_a = 0_i32 + 12;
                int32_t v422_a = 0_i32 + 12;
                int32_t v424_a = 0_i32 + 12;
                int32_t v426_a = 0_i32 + 12;
                int32_t v428_a = 0_i32 + 12;
                int32_t v430_a = 0_i32 + 12;
                int32_t v432_a = 0_i32 + 12;
                int32_t v434_a = 0_i32 + 12;
                int32_t v436_a = 0_i32 + 12;
                int32_t v438_a = 0_i32 + 12;
              }
              if (v9_lead < 6) {
                int32_t v444_a = 0_i32 + 18;
                int32_t v446_a = 0_i32 + 18;
                int32_t v448_a = 0_i32 + 18;
                int32_t v450_a = 0_i32 + 18;
                int32_t v452_a = 0_i32 + 18;
                int32_t v454_a = 0_i32 + 18;
                int32_t v456_a = 0_i32 + 18;
                int32_t v458_a = 0_i32 + 18;
                int32_t v460_a = 0_i32 + 18;
                int32_t v462_a = 0_i32 + 18;
                int32_t v464_a = 0_i32 + 18;
                int32_t v466_a = 0_i32 + 18;
              }
              if (v9_lead < 6) {
                int32_t v472_a = 0_i32 + 24;
                int32_t v474_a = 0_i32 + 24;
                int32_t v476_a = 0_i32 + 24;
                int32_t v478_a = 0_i32 + 24;
                int32_t v480_a = 0_i32 + 24;
                int32_t v482_a = 0_i32 + 24;
                int32_t v484_a = 0_i32 + 24;
                int32_t v486_a = 0_i32 + 24;
                int32_t v488_a = 0_i32 + 24;
                int32_t v490_a = 0_i32 + 24;
                int32_t v492_a = 0_i32 + 24;
                int32_t v494_a = 0_i32 + 24;
              }
              if (v9_lead < 6) {
                int32_t v500_a = 0_i32 + 30;
                int32_t v502_a = 0_i32 + 30;
                int32_t v504_a = 0_i32 + 30;
                int32_t v506_a = 0_i32 + 30;
                int32_t v508_a = 0_i32 + 30;
                int32_t v510_a = 0_i32 + 30;
                int32_t v512_a = 0_i32 + 30;
                int32_t v514_a = 0_i32 + 30;
                int32_t v516_a = 0_i32 + 30;
                int32_t v518_a = 0_i32 + 30;
                int32_t v520_a = 0_i32 + 30;
                int32_t v522_a = 0_i32 + 30;
              }
              if (v9_lead < 6) {
                int32_t v528_a = 0_i32 + 36;
                int32_t v530_a = 0_i32 + 36;
                int32_t v532_a = 0_i32 + 36;
                int32_t v534_a = 0_i32 + 36;
                int32_t v536_a = 0_i32 + 36;
                int32_t v538_a = 0_i32 + 36;
                int32_t v540_a = 0_i32 + 36;
                int32_t v542_a = 0_i32 + 36;
                int32_t v544_a = 0_i32 + 36;
                int32_t v546_a = 0_i32 + 36;
                int32_t v548_a = 0_i32 + 36;
                int32_t v550_a = 0_i32 + 36;
              }
              if (v9_lead < 6) {
                int32_t v556_a = 0_i32 + 42;
                int32_t v558_a = 0_i32 + 42;
                int32_t v560_a = 0_i32 + 42;
                int32_t v562_a = 0_i32 + 42;
                int32_t v564_a = 0_i32 + 42;
                int32_t v566_a = 0_i32 + 42;
                int32_t v568_a = 0_i32 + 42;
                int32_t v570_a = 0_i32 + 42;
                int32_t v572_a = 0_i32 + 42;
                int32_t v574_a = 0_i32 + 42;
                int32_t v576_a = 0_i32 + 42;
                int32_t v578_a = 0_i32 + 42;
              }
              if (v9_lead < 6) {
                int32_t v584_a = 0_i32 + 48;
                int32_t v586_a = 0_i32 + 48;
                int32_t v588_a = 0_i32 + 48;
                int32_t v590_a = 0_i32 + 48;
                int32_t v592_a = 0_i32 + 48;
                int32_t v594_a = 0_i32 + 48;
                int32_t v596_a = 0_i32 + 48;
                int32_t v598_a = 0_i32 + 48;
                int32_t v600_a = 0_i32 + 48;
                int32_t v602_a = 0_i32 + 48;
                int32_t v604_a = 0_i32 + 48;
                int32_t v606_a = 0_i32 + 48;
              }
              if (v9_lead < 6) {
                int32_t v612_a = 0_i32 + 54;
                int32_t v614_a = 0_i32 + 54;
                int32_t v616_a = 0_i32 + 54;
                int32_t v618_a = 0_i32 + 54;
                int32_t v620_a = 0_i32 + 54;
                int32_t v622_a = 0_i32 + 54;
                int32_t v624_a = 0_i32 + 54;
                int32_t v626_a = 0_i32 + 54;
                int32_t v628_a = 0_i32 + 54;
                int32_t v630_a = 0_i32 + 54;
                int32_t v632_a = 0_i32 + 54;
                int32_t v634_a = 0_i32 + 54;
              }
              if (v9_lead < 6) {
                int32_t v640_a = 0_i32 + 60;
                int32_t v642_a = 0_i32 + 60;
                int32_t v644_a = 0_i32 + 60;
                int32_t v646_a = 0_i32 + 60;
                int32_t v648_a = 0_i32 + 60;
                int32_t v650_a = 0_i32 + 60;
                int32_t v652_a = 0_i32 + 60;
                int32_t v654_a = 0_i32 + 60;
                int32_t v656_a = 0_i32 + 60;
                int32_t v658_a = 0_i32 + 60;
                int32_t v660_a = 0_i32 + 60;
                int32_t v662_a = 0_i32 + 60;
              }
              if (v9_lead < 6) {
                int32_t v668_a = 0_i32 + 66;
                int32_t v670_a = 0_i32 + 66;
                int32_t v672_a = 0_i32 + 66;
                int32_t v674_a = 0_i32 + 66;
                int32_t v676_a = 0_i32 + 66;
                int32_t v678_a = 0_i32 + 66;
                int32_t v680_a = 0_i32 + 66;
                int32_t v682_a = 0_i32 + 66;
                int32_t v684_a = 0_i32 + 66;
                int32_t v686_a = 0_i32 + 66;
                int32_t v688_a = 0_i32 + 66;
                int32_t v690_a = 0_i32 + 66;
              }
              if (v9_lead < 6) {
                #pragma unroll
                for (int32_t v695_n1 = 0; v695_n1 < 12; ++v695_n1) {
                  int32_t v696_a = 0 + v695_n1;
                  int32_t v697_a = 0 + v695_n1;
                  None = r1[v697_a];
                }
              }
              ;
              // s1 = store{r>s}(localShrMem0, r1);
              if (v9_lead < 6) {
                #pragma unroll
                for (int32_t v702_i1 = 0; v702_i1 < 12; ++v702_i1) {
                  int32_t v703_a = 0 + v702_i1;
                  int32_t v707_a = 6_i32 + (v702_i1 * 12);
                  None = s1[v707_a];
                }
              }
              float r2[12]{};
              ;
              // r2 = +(glb_m4 * s1) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir2[12]{};
              if (v9_lead < 12) {
                int32_t v715_a = 0_i32 + 0;
                int32_t v717_a = 0_i32 + 0;
                int32_t v719_a = 0_i32 + 0;
                int32_t v721_a = 0_i32 + 0;
                int32_t v723_a = 0_i32 + 0;
                int32_t v725_a = 0_i32 + 0;
                int32_t v727_a = 0_i32 + 0;
                int32_t v729_a = 0_i32 + 0;
                int32_t v731_a = 0_i32 + 0;
                int32_t v733_a = 0_i32 + 0;
                int32_t v735_a = 0_i32 + 0;
                int32_t v737_a = 0_i32 + 0;
              }
              if (v9_lead < 12) {
                int32_t v743_a = 0_i32 + 12;
                int32_t v745_a = 0_i32 + 12;
                int32_t v747_a = 0_i32 + 12;
                int32_t v749_a = 0_i32 + 12;
                int32_t v751_a = 0_i32 + 12;
                int32_t v753_a = 0_i32 + 12;
                int32_t v755_a = 0_i32 + 12;
                int32_t v757_a = 0_i32 + 12;
                int32_t v759_a = 0_i32 + 12;
                int32_t v761_a = 0_i32 + 12;
                int32_t v763_a = 0_i32 + 12;
                int32_t v765_a = 0_i32 + 12;
              }
              if (v9_lead < 12) {
                int32_t v771_a = 0_i32 + 24;
                int32_t v773_a = 0_i32 + 24;
                int32_t v775_a = 0_i32 + 24;
                int32_t v777_a = 0_i32 + 24;
                int32_t v779_a = 0_i32 + 24;
                int32_t v781_a = 0_i32 + 24;
                int32_t v783_a = 0_i32 + 24;
                int32_t v785_a = 0_i32 + 24;
                int32_t v787_a = 0_i32 + 24;
                int32_t v789_a = 0_i32 + 24;
                int32_t v791_a = 0_i32 + 24;
                int32_t v793_a = 0_i32 + 24;
              }
              if (v9_lead < 12) {
                int32_t v799_a = 0_i32 + 36;
                int32_t v801_a = 0_i32 + 36;
                int32_t v803_a = 0_i32 + 36;
                int32_t v805_a = 0_i32 + 36;
                int32_t v807_a = 0_i32 + 36;
                int32_t v809_a = 0_i32 + 36;
                int32_t v811_a = 0_i32 + 36;
                int32_t v813_a = 0_i32 + 36;
                int32_t v815_a = 0_i32 + 36;
                int32_t v817_a = 0_i32 + 36;
                int32_t v819_a = 0_i32 + 36;
                int32_t v821_a = 0_i32 + 36;
              }
              if (v9_lead < 12) {
                int32_t v827_a = 0_i32 + 48;
                int32_t v829_a = 0_i32 + 48;
                int32_t v831_a = 0_i32 + 48;
                int32_t v833_a = 0_i32 + 48;
                int32_t v835_a = 0_i32 + 48;
                int32_t v837_a = 0_i32 + 48;
                int32_t v839_a = 0_i32 + 48;
                int32_t v841_a = 0_i32 + 48;
                int32_t v843_a = 0_i32 + 48;
                int32_t v845_a = 0_i32 + 48;
                int32_t v847_a = 0_i32 + 48;
                int32_t v849_a = 0_i32 + 48;
              }
              if (v9_lead < 12) {
                int32_t v855_a = 0_i32 + 60;
                int32_t v857_a = 0_i32 + 60;
                int32_t v859_a = 0_i32 + 60;
                int32_t v861_a = 0_i32 + 60;
                int32_t v863_a = 0_i32 + 60;
                int32_t v865_a = 0_i32 + 60;
                int32_t v867_a = 0_i32 + 60;
                int32_t v869_a = 0_i32 + 60;
                int32_t v871_a = 0_i32 + 60;
                int32_t v873_a = 0_i32 + 60;
                int32_t v875_a = 0_i32 + 60;
                int32_t v877_a = 0_i32 + 60;
              }
              if (v9_lead < 12) {
                int32_t v883_a = 0_i32 + 72;
                int32_t v885_a = 0_i32 + 72;
                int32_t v887_a = 0_i32 + 72;
                int32_t v889_a = 0_i32 + 72;
                int32_t v891_a = 0_i32 + 72;
                int32_t v893_a = 0_i32 + 72;
                int32_t v895_a = 0_i32 + 72;
                int32_t v897_a = 0_i32 + 72;
                int32_t v899_a = 0_i32 + 72;
                int32_t v901_a = 0_i32 + 72;
                int32_t v903_a = 0_i32 + 72;
                int32_t v905_a = 0_i32 + 72;
              }
              if (v9_lead < 12) {
                int32_t v911_a = 0_i32 + 84;
                int32_t v913_a = 0_i32 + 84;
                int32_t v915_a = 0_i32 + 84;
                int32_t v917_a = 0_i32 + 84;
                int32_t v919_a = 0_i32 + 84;
                int32_t v921_a = 0_i32 + 84;
                int32_t v923_a = 0_i32 + 84;
                int32_t v925_a = 0_i32 + 84;
                int32_t v927_a = 0_i32 + 84;
                int32_t v929_a = 0_i32 + 84;
                int32_t v931_a = 0_i32 + 84;
                int32_t v933_a = 0_i32 + 84;
              }
              if (v9_lead < 12) {
                int32_t v939_a = 0_i32 + 96;
                int32_t v941_a = 0_i32 + 96;
                int32_t v943_a = 0_i32 + 96;
                int32_t v945_a = 0_i32 + 96;
                int32_t v947_a = 0_i32 + 96;
                int32_t v949_a = 0_i32 + 96;
                int32_t v951_a = 0_i32 + 96;
                int32_t v953_a = 0_i32 + 96;
                int32_t v955_a = 0_i32 + 96;
                int32_t v957_a = 0_i32 + 96;
                int32_t v959_a = 0_i32 + 96;
                int32_t v961_a = 0_i32 + 96;
              }
              if (v9_lead < 12) {
                int32_t v967_a = 0_i32 + 108;
                int32_t v969_a = 0_i32 + 108;
                int32_t v971_a = 0_i32 + 108;
                int32_t v973_a = 0_i32 + 108;
                int32_t v975_a = 0_i32 + 108;
                int32_t v977_a = 0_i32 + 108;
                int32_t v979_a = 0_i32 + 108;
                int32_t v981_a = 0_i32 + 108;
                int32_t v983_a = 0_i32 + 108;
                int32_t v985_a = 0_i32 + 108;
                int32_t v987_a = 0_i32 + 108;
                int32_t v989_a = 0_i32 + 108;
              }
              if (v9_lead < 12) {
                int32_t v995_a = 0_i32 + 120;
                int32_t v997_a = 0_i32 + 120;
                int32_t v999_a = 0_i32 + 120;
                int32_t v1001_a = 0_i32 + 120;
                int32_t v1003_a = 0_i32 + 120;
                int32_t v1005_a = 0_i32 + 120;
                int32_t v1007_a = 0_i32 + 120;
                int32_t v1009_a = 0_i32 + 120;
                int32_t v1011_a = 0_i32 + 120;
                int32_t v1013_a = 0_i32 + 120;
                int32_t v1015_a = 0_i32 + 120;
                int32_t v1017_a = 0_i32 + 120;
              }
              if (v9_lead < 12) {
                int32_t v1023_a = 0_i32 + 132;
                int32_t v1025_a = 0_i32 + 132;
                int32_t v1027_a = 0_i32 + 132;
                int32_t v1029_a = 0_i32 + 132;
                int32_t v1031_a = 0_i32 + 132;
                int32_t v1033_a = 0_i32 + 132;
                int32_t v1035_a = 0_i32 + 132;
                int32_t v1037_a = 0_i32 + 132;
                int32_t v1039_a = 0_i32 + 132;
                int32_t v1041_a = 0_i32 + 132;
                int32_t v1043_a = 0_i32 + 132;
                int32_t v1045_a = 0_i32 + 132;
              }
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v1050_n1 = 0; v1050_n1 < 12; ++v1050_n1) {
                  int32_t v1051_a = 0 + v1050_n1;
                  int32_t v1052_a = 0 + v1050_n1;
                  None = r2[v1052_a];
                }
              }
              // glb_m3 = store{r>g}(r2);
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v1057_i1 = 0; v1057_i1 < 12; ++v1057_i1) {
                  int32_t v1058_a = 0 + v1057_i1;
                  int32_t v1061_a = 0_i32 + (v1057_i1 * 12);
                  None.copy_to(glb_m3[v1061_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}


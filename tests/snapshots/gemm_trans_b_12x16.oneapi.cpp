// === base name ===
kernel_e7f2438624

// === header ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e7f2438624(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e7f2438624(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (5888, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 12×20(12×20) {0..12}×{0..20} strided
        // m2 16×20(16×20) {0..16}×{0..20} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 12×20(12×20) {0..12}×{0..20} strided({0..12}×{0..20})[0, -1]×m2 16×20(16×20) {0..16}×{0..20} strided({0..16}×{0..20})[1, -1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[368 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[352];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[1, 0])
              int32_t v6_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
                int32_t v9_lead = v7_i0 * 16;
                #pragma unroll
                for (int32_t v8_i1 = 0; v8_i1 < 20; ++v8_i1) {
                  int32_t v11_a = v9_lead + (v8_i1 * 16);
                  int32_t v14_a = v9_lead + (v8_i1 * 17);
                  None = s0[v14_a];
                }
              }
              // wait(s0 = load{g>s}(glb_m2[1, 0]));
              float r0[16]{};
              ;
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[16]{};
              if (v6_lead < 12) {
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
                int32_t v42_a = 0_i32 + 0;
                int32_t v44_a = 0_i32 + 0;
                int32_t v46_a = 0_i32 + 0;
                int32_t v48_a = 0_i32 + 0;
                int32_t v50_a = 0_i32 + 0;
                int32_t v52_a = 0_i32 + 0;
              }
              if (v6_lead < 12) {
                int32_t v58_a = 0_i32 + 12;
                int32_t v60_a = 0_i32 + 12;
                int32_t v62_a = 0_i32 + 12;
                int32_t v64_a = 0_i32 + 12;
                int32_t v66_a = 0_i32 + 12;
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
              }
              if (v6_lead < 12) {
                int32_t v94_a = 0_i32 + 24;
                int32_t v96_a = 0_i32 + 24;
                int32_t v98_a = 0_i32 + 24;
                int32_t v100_a = 0_i32 + 24;
                int32_t v102_a = 0_i32 + 24;
                int32_t v104_a = 0_i32 + 24;
                int32_t v106_a = 0_i32 + 24;
                int32_t v108_a = 0_i32 + 24;
                int32_t v110_a = 0_i32 + 24;
                int32_t v112_a = 0_i32 + 24;
                int32_t v114_a = 0_i32 + 24;
                int32_t v116_a = 0_i32 + 24;
                int32_t v118_a = 0_i32 + 24;
                int32_t v120_a = 0_i32 + 24;
                int32_t v122_a = 0_i32 + 24;
                int32_t v124_a = 0_i32 + 24;
              }
              if (v6_lead < 12) {
                int32_t v130_a = 0_i32 + 36;
                int32_t v132_a = 0_i32 + 36;
                int32_t v134_a = 0_i32 + 36;
                int32_t v136_a = 0_i32 + 36;
                int32_t v138_a = 0_i32 + 36;
                int32_t v140_a = 0_i32 + 36;
                int32_t v142_a = 0_i32 + 36;
                int32_t v144_a = 0_i32 + 36;
                int32_t v146_a = 0_i32 + 36;
                int32_t v148_a = 0_i32 + 36;
                int32_t v150_a = 0_i32 + 36;
                int32_t v152_a = 0_i32 + 36;
                int32_t v154_a = 0_i32 + 36;
                int32_t v156_a = 0_i32 + 36;
                int32_t v158_a = 0_i32 + 36;
                int32_t v160_a = 0_i32 + 36;
              }
              if (v6_lead < 12) {
                int32_t v166_a = 0_i32 + 48;
                int32_t v168_a = 0_i32 + 48;
                int32_t v170_a = 0_i32 + 48;
                int32_t v172_a = 0_i32 + 48;
                int32_t v174_a = 0_i32 + 48;
                int32_t v176_a = 0_i32 + 48;
                int32_t v178_a = 0_i32 + 48;
                int32_t v180_a = 0_i32 + 48;
                int32_t v182_a = 0_i32 + 48;
                int32_t v184_a = 0_i32 + 48;
                int32_t v186_a = 0_i32 + 48;
                int32_t v188_a = 0_i32 + 48;
                int32_t v190_a = 0_i32 + 48;
                int32_t v192_a = 0_i32 + 48;
                int32_t v194_a = 0_i32 + 48;
                int32_t v196_a = 0_i32 + 48;
              }
              if (v6_lead < 12) {
                int32_t v202_a = 0_i32 + 60;
                int32_t v204_a = 0_i32 + 60;
                int32_t v206_a = 0_i32 + 60;
                int32_t v208_a = 0_i32 + 60;
                int32_t v210_a = 0_i32 + 60;
                int32_t v212_a = 0_i32 + 60;
                int32_t v214_a = 0_i32 + 60;
                int32_t v216_a = 0_i32 + 60;
                int32_t v218_a = 0_i32 + 60;
                int32_t v220_a = 0_i32 + 60;
                int32_t v222_a = 0_i32 + 60;
                int32_t v224_a = 0_i32 + 60;
                int32_t v226_a = 0_i32 + 60;
                int32_t v228_a = 0_i32 + 60;
                int32_t v230_a = 0_i32 + 60;
                int32_t v232_a = 0_i32 + 60;
              }
              if (v6_lead < 12) {
                int32_t v238_a = 0_i32 + 72;
                int32_t v240_a = 0_i32 + 72;
                int32_t v242_a = 0_i32 + 72;
                int32_t v244_a = 0_i32 + 72;
                int32_t v246_a = 0_i32 + 72;
                int32_t v248_a = 0_i32 + 72;
                int32_t v250_a = 0_i32 + 72;
                int32_t v252_a = 0_i32 + 72;
                int32_t v254_a = 0_i32 + 72;
                int32_t v256_a = 0_i32 + 72;
                int32_t v258_a = 0_i32 + 72;
                int32_t v260_a = 0_i32 + 72;
                int32_t v262_a = 0_i32 + 72;
                int32_t v264_a = 0_i32 + 72;
                int32_t v266_a = 0_i32 + 72;
                int32_t v268_a = 0_i32 + 72;
              }
              if (v6_lead < 12) {
                int32_t v274_a = 0_i32 + 84;
                int32_t v276_a = 0_i32 + 84;
                int32_t v278_a = 0_i32 + 84;
                int32_t v280_a = 0_i32 + 84;
                int32_t v282_a = 0_i32 + 84;
                int32_t v284_a = 0_i32 + 84;
                int32_t v286_a = 0_i32 + 84;
                int32_t v288_a = 0_i32 + 84;
                int32_t v290_a = 0_i32 + 84;
                int32_t v292_a = 0_i32 + 84;
                int32_t v294_a = 0_i32 + 84;
                int32_t v296_a = 0_i32 + 84;
                int32_t v298_a = 0_i32 + 84;
                int32_t v300_a = 0_i32 + 84;
                int32_t v302_a = 0_i32 + 84;
                int32_t v304_a = 0_i32 + 84;
              }
              if (v6_lead < 12) {
                int32_t v310_a = 0_i32 + 96;
                int32_t v312_a = 0_i32 + 96;
                int32_t v314_a = 0_i32 + 96;
                int32_t v316_a = 0_i32 + 96;
                int32_t v318_a = 0_i32 + 96;
                int32_t v320_a = 0_i32 + 96;
                int32_t v322_a = 0_i32 + 96;
                int32_t v324_a = 0_i32 + 96;
                int32_t v326_a = 0_i32 + 96;
                int32_t v328_a = 0_i32 + 96;
                int32_t v330_a = 0_i32 + 96;
                int32_t v332_a = 0_i32 + 96;
                int32_t v334_a = 0_i32 + 96;
                int32_t v336_a = 0_i32 + 96;
                int32_t v338_a = 0_i32 + 96;
                int32_t v340_a = 0_i32 + 96;
              }
              if (v6_lead < 12) {
                int32_t v346_a = 0_i32 + 108;
                int32_t v348_a = 0_i32 + 108;
                int32_t v350_a = 0_i32 + 108;
                int32_t v352_a = 0_i32 + 108;
                int32_t v354_a = 0_i32 + 108;
                int32_t v356_a = 0_i32 + 108;
                int32_t v358_a = 0_i32 + 108;
                int32_t v360_a = 0_i32 + 108;
                int32_t v362_a = 0_i32 + 108;
                int32_t v364_a = 0_i32 + 108;
                int32_t v366_a = 0_i32 + 108;
                int32_t v368_a = 0_i32 + 108;
                int32_t v370_a = 0_i32 + 108;
                int32_t v372_a = 0_i32 + 108;
                int32_t v374_a = 0_i32 + 108;
                int32_t v376_a = 0_i32 + 108;
              }
              if (v6_lead < 12) {
                int32_t v382_a = 0_i32 + 120;
                int32_t v384_a = 0_i32 + 120;
                int32_t v386_a = 0_i32 + 120;
                int32_t v388_a = 0_i32 + 120;
                int32_t v390_a = 0_i32 + 120;
                int32_t v392_a = 0_i32 + 120;
                int32_t v394_a = 0_i32 + 120;
                int32_t v396_a = 0_i32 + 120;
                int32_t v398_a = 0_i32 + 120;
                int32_t v400_a = 0_i32 + 120;
                int32_t v402_a = 0_i32 + 120;
                int32_t v404_a = 0_i32 + 120;
                int32_t v406_a = 0_i32 + 120;
                int32_t v408_a = 0_i32 + 120;
                int32_t v410_a = 0_i32 + 120;
                int32_t v412_a = 0_i32 + 120;
              }
              if (v6_lead < 12) {
                int32_t v418_a = 0_i32 + 132;
                int32_t v420_a = 0_i32 + 132;
                int32_t v422_a = 0_i32 + 132;
                int32_t v424_a = 0_i32 + 132;
                int32_t v426_a = 0_i32 + 132;
                int32_t v428_a = 0_i32 + 132;
                int32_t v430_a = 0_i32 + 132;
                int32_t v432_a = 0_i32 + 132;
                int32_t v434_a = 0_i32 + 132;
                int32_t v436_a = 0_i32 + 132;
                int32_t v438_a = 0_i32 + 132;
                int32_t v440_a = 0_i32 + 132;
                int32_t v442_a = 0_i32 + 132;
                int32_t v444_a = 0_i32 + 132;
                int32_t v446_a = 0_i32 + 132;
                int32_t v448_a = 0_i32 + 132;
              }
              if (v6_lead < 12) {
                int32_t v454_a = 0_i32 + 144;
                int32_t v456_a = 0_i32 + 144;
                int32_t v458_a = 0_i32 + 144;
                int32_t v460_a = 0_i32 + 144;
                int32_t v462_a = 0_i32 + 144;
                int32_t v464_a = 0_i32 + 144;
                int32_t v466_a = 0_i32 + 144;
                int32_t v468_a = 0_i32 + 144;
                int32_t v470_a = 0_i32 + 144;
                int32_t v472_a = 0_i32 + 144;
                int32_t v474_a = 0_i32 + 144;
                int32_t v476_a = 0_i32 + 144;
                int32_t v478_a = 0_i32 + 144;
                int32_t v480_a = 0_i32 + 144;
                int32_t v482_a = 0_i32 + 144;
                int32_t v484_a = 0_i32 + 144;
              }
              if (v6_lead < 12) {
                int32_t v490_a = 0_i32 + 156;
                int32_t v492_a = 0_i32 + 156;
                int32_t v494_a = 0_i32 + 156;
                int32_t v496_a = 0_i32 + 156;
                int32_t v498_a = 0_i32 + 156;
                int32_t v500_a = 0_i32 + 156;
                int32_t v502_a = 0_i32 + 156;
                int32_t v504_a = 0_i32 + 156;
                int32_t v506_a = 0_i32 + 156;
                int32_t v508_a = 0_i32 + 156;
                int32_t v510_a = 0_i32 + 156;
                int32_t v512_a = 0_i32 + 156;
                int32_t v514_a = 0_i32 + 156;
                int32_t v516_a = 0_i32 + 156;
                int32_t v518_a = 0_i32 + 156;
                int32_t v520_a = 0_i32 + 156;
              }
              if (v6_lead < 12) {
                int32_t v526_a = 0_i32 + 168;
                int32_t v528_a = 0_i32 + 168;
                int32_t v530_a = 0_i32 + 168;
                int32_t v532_a = 0_i32 + 168;
                int32_t v534_a = 0_i32 + 168;
                int32_t v536_a = 0_i32 + 168;
                int32_t v538_a = 0_i32 + 168;
                int32_t v540_a = 0_i32 + 168;
                int32_t v542_a = 0_i32 + 168;
                int32_t v544_a = 0_i32 + 168;
                int32_t v546_a = 0_i32 + 168;
                int32_t v548_a = 0_i32 + 168;
                int32_t v550_a = 0_i32 + 168;
                int32_t v552_a = 0_i32 + 168;
                int32_t v554_a = 0_i32 + 168;
                int32_t v556_a = 0_i32 + 168;
              }
              if (v6_lead < 12) {
                int32_t v562_a = 0_i32 + 180;
                int32_t v564_a = 0_i32 + 180;
                int32_t v566_a = 0_i32 + 180;
                int32_t v568_a = 0_i32 + 180;
                int32_t v570_a = 0_i32 + 180;
                int32_t v572_a = 0_i32 + 180;
                int32_t v574_a = 0_i32 + 180;
                int32_t v576_a = 0_i32 + 180;
                int32_t v578_a = 0_i32 + 180;
                int32_t v580_a = 0_i32 + 180;
                int32_t v582_a = 0_i32 + 180;
                int32_t v584_a = 0_i32 + 180;
                int32_t v586_a = 0_i32 + 180;
                int32_t v588_a = 0_i32 + 180;
                int32_t v590_a = 0_i32 + 180;
                int32_t v592_a = 0_i32 + 180;
              }
              if (v6_lead < 12) {
                int32_t v598_a = 0_i32 + 192;
                int32_t v600_a = 0_i32 + 192;
                int32_t v602_a = 0_i32 + 192;
                int32_t v604_a = 0_i32 + 192;
                int32_t v606_a = 0_i32 + 192;
                int32_t v608_a = 0_i32 + 192;
                int32_t v610_a = 0_i32 + 192;
                int32_t v612_a = 0_i32 + 192;
                int32_t v614_a = 0_i32 + 192;
                int32_t v616_a = 0_i32 + 192;
                int32_t v618_a = 0_i32 + 192;
                int32_t v620_a = 0_i32 + 192;
                int32_t v622_a = 0_i32 + 192;
                int32_t v624_a = 0_i32 + 192;
                int32_t v626_a = 0_i32 + 192;
                int32_t v628_a = 0_i32 + 192;
              }
              if (v6_lead < 12) {
                int32_t v634_a = 0_i32 + 204;
                int32_t v636_a = 0_i32 + 204;
                int32_t v638_a = 0_i32 + 204;
                int32_t v640_a = 0_i32 + 204;
                int32_t v642_a = 0_i32 + 204;
                int32_t v644_a = 0_i32 + 204;
                int32_t v646_a = 0_i32 + 204;
                int32_t v648_a = 0_i32 + 204;
                int32_t v650_a = 0_i32 + 204;
                int32_t v652_a = 0_i32 + 204;
                int32_t v654_a = 0_i32 + 204;
                int32_t v656_a = 0_i32 + 204;
                int32_t v658_a = 0_i32 + 204;
                int32_t v660_a = 0_i32 + 204;
                int32_t v662_a = 0_i32 + 204;
                int32_t v664_a = 0_i32 + 204;
              }
              if (v6_lead < 12) {
                int32_t v670_a = 0_i32 + 216;
                int32_t v672_a = 0_i32 + 216;
                int32_t v674_a = 0_i32 + 216;
                int32_t v676_a = 0_i32 + 216;
                int32_t v678_a = 0_i32 + 216;
                int32_t v680_a = 0_i32 + 216;
                int32_t v682_a = 0_i32 + 216;
                int32_t v684_a = 0_i32 + 216;
                int32_t v686_a = 0_i32 + 216;
                int32_t v688_a = 0_i32 + 216;
                int32_t v690_a = 0_i32 + 216;
                int32_t v692_a = 0_i32 + 216;
                int32_t v694_a = 0_i32 + 216;
                int32_t v696_a = 0_i32 + 216;
                int32_t v698_a = 0_i32 + 216;
                int32_t v700_a = 0_i32 + 216;
              }
              if (v6_lead < 12) {
                int32_t v706_a = 0_i32 + 228;
                int32_t v708_a = 0_i32 + 228;
                int32_t v710_a = 0_i32 + 228;
                int32_t v712_a = 0_i32 + 228;
                int32_t v714_a = 0_i32 + 228;
                int32_t v716_a = 0_i32 + 228;
                int32_t v718_a = 0_i32 + 228;
                int32_t v720_a = 0_i32 + 228;
                int32_t v722_a = 0_i32 + 228;
                int32_t v724_a = 0_i32 + 228;
                int32_t v726_a = 0_i32 + 228;
                int32_t v728_a = 0_i32 + 228;
                int32_t v730_a = 0_i32 + 228;
                int32_t v732_a = 0_i32 + 228;
                int32_t v734_a = 0_i32 + 228;
                int32_t v736_a = 0_i32 + 228;
              }
              if (v6_lead < 12) {
                #pragma unroll
                for (int32_t v741_n1 = 0; v741_n1 < 16; ++v741_n1) {
                  int32_t v742_a = 0 + v741_n1;
                  int32_t v743_a = 0 + v741_n1;
                  None = r0[v743_a];
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v6_lead < 12) {
                #pragma unroll
                for (int32_t v748_i1 = 0; v748_i1 < 16; ++v748_i1) {
                  int32_t v749_a = 0 + v748_i1;
                  int32_t v752_a = 0_i32 + (v748_i1 * 12);
                  None.copy_to(glb_m0[v752_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}


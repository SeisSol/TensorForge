// === base name ===
kernel_f94e030d8c

// === header ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f94e030d8c(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f94e030d8c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (9472, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 20×12(20×12) {0..20}×{0..12} strided
        // m2 20×16(20×16) {0..20}×{0..16} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 20×12(20×12) {0..20}×{0..12} strided({0..20}×{0..12})[-1, 0]×m2 20×16(20×16) {0..20}×{0..16} strided({0..20}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[592 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[576];
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
              float* __restrict__ s0 = &localShrMem0[320];
              // s0 = load{g>s}(glb_m1[1, 0])
              int32_t v6_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
                int32_t v9_lead = v7_i0 * 16;
                #pragma unroll
                for (int32_t v8_i1 = 0; v8_i1 < 12; ++v8_i1) {
                  int32_t v11_a = v9_lead + (v8_i1 * 20);
                  int32_t v14_a = v9_lead + (v8_i1 * 21);
                  None = s0[v14_a];
                }
              }
              if (v6_lead < 4) {
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 12; ++v16_i1) {
                  int32_t v19_a = 16_i32 + (v16_i1 * 20);
                  int32_t v22_a = 16_i32 + (v16_i1 * 21);
                  None = s0[v22_a];
                }
              }
              float* __restrict__ s1 = &localShrMem0[0];
              {
                // s1 = load{g>s}(glb_m2[0, 1])
                #pragma unroll
                for (int32_t i = 0; i < 20; i += 4) {
                  *(sycl::vec<float, 4>*)&s1[0 + 0 + 4 * item.get_local_id(0) + i * 16] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + i * 16];
                }
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r0[16]{};
              ;
              // r0 = +(s0 * s1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[16]{};
              if (v6_lead < 12) {
                int32_t v32_a = 0 + 0_i32;
                int32_t v35_a = 0 + 0_i32;
                int32_t v38_a = 0 + 0_i32;
                int32_t v41_a = 0 + 0_i32;
                int32_t v44_a = 0 + 0_i32;
                int32_t v47_a = 0 + 0_i32;
                int32_t v50_a = 0 + 0_i32;
                int32_t v53_a = 0 + 0_i32;
                int32_t v56_a = 0 + 0_i32;
                int32_t v59_a = 0 + 0_i32;
                int32_t v62_a = 0 + 0_i32;
                int32_t v65_a = 0 + 0_i32;
                int32_t v68_a = 0 + 0_i32;
                int32_t v71_a = 0 + 0_i32;
                int32_t v74_a = 0 + 0_i32;
                int32_t v77_a = 0 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v84_a = 1 + 0_i32;
                int32_t v87_a = 1 + 0_i32;
                int32_t v90_a = 1 + 0_i32;
                int32_t v93_a = 1 + 0_i32;
                int32_t v96_a = 1 + 0_i32;
                int32_t v99_a = 1 + 0_i32;
                int32_t v102_a = 1 + 0_i32;
                int32_t v105_a = 1 + 0_i32;
                int32_t v108_a = 1 + 0_i32;
                int32_t v111_a = 1 + 0_i32;
                int32_t v114_a = 1 + 0_i32;
                int32_t v117_a = 1 + 0_i32;
                int32_t v120_a = 1 + 0_i32;
                int32_t v123_a = 1 + 0_i32;
                int32_t v126_a = 1 + 0_i32;
                int32_t v129_a = 1 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v136_a = 2 + 0_i32;
                int32_t v139_a = 2 + 0_i32;
                int32_t v142_a = 2 + 0_i32;
                int32_t v145_a = 2 + 0_i32;
                int32_t v148_a = 2 + 0_i32;
                int32_t v151_a = 2 + 0_i32;
                int32_t v154_a = 2 + 0_i32;
                int32_t v157_a = 2 + 0_i32;
                int32_t v160_a = 2 + 0_i32;
                int32_t v163_a = 2 + 0_i32;
                int32_t v166_a = 2 + 0_i32;
                int32_t v169_a = 2 + 0_i32;
                int32_t v172_a = 2 + 0_i32;
                int32_t v175_a = 2 + 0_i32;
                int32_t v178_a = 2 + 0_i32;
                int32_t v181_a = 2 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v188_a = 3 + 0_i32;
                int32_t v191_a = 3 + 0_i32;
                int32_t v194_a = 3 + 0_i32;
                int32_t v197_a = 3 + 0_i32;
                int32_t v200_a = 3 + 0_i32;
                int32_t v203_a = 3 + 0_i32;
                int32_t v206_a = 3 + 0_i32;
                int32_t v209_a = 3 + 0_i32;
                int32_t v212_a = 3 + 0_i32;
                int32_t v215_a = 3 + 0_i32;
                int32_t v218_a = 3 + 0_i32;
                int32_t v221_a = 3 + 0_i32;
                int32_t v224_a = 3 + 0_i32;
                int32_t v227_a = 3 + 0_i32;
                int32_t v230_a = 3 + 0_i32;
                int32_t v233_a = 3 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v240_a = 4 + 0_i32;
                int32_t v243_a = 4 + 0_i32;
                int32_t v246_a = 4 + 0_i32;
                int32_t v249_a = 4 + 0_i32;
                int32_t v252_a = 4 + 0_i32;
                int32_t v255_a = 4 + 0_i32;
                int32_t v258_a = 4 + 0_i32;
                int32_t v261_a = 4 + 0_i32;
                int32_t v264_a = 4 + 0_i32;
                int32_t v267_a = 4 + 0_i32;
                int32_t v270_a = 4 + 0_i32;
                int32_t v273_a = 4 + 0_i32;
                int32_t v276_a = 4 + 0_i32;
                int32_t v279_a = 4 + 0_i32;
                int32_t v282_a = 4 + 0_i32;
                int32_t v285_a = 4 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v292_a = 5 + 0_i32;
                int32_t v295_a = 5 + 0_i32;
                int32_t v298_a = 5 + 0_i32;
                int32_t v301_a = 5 + 0_i32;
                int32_t v304_a = 5 + 0_i32;
                int32_t v307_a = 5 + 0_i32;
                int32_t v310_a = 5 + 0_i32;
                int32_t v313_a = 5 + 0_i32;
                int32_t v316_a = 5 + 0_i32;
                int32_t v319_a = 5 + 0_i32;
                int32_t v322_a = 5 + 0_i32;
                int32_t v325_a = 5 + 0_i32;
                int32_t v328_a = 5 + 0_i32;
                int32_t v331_a = 5 + 0_i32;
                int32_t v334_a = 5 + 0_i32;
                int32_t v337_a = 5 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v344_a = 6 + 0_i32;
                int32_t v347_a = 6 + 0_i32;
                int32_t v350_a = 6 + 0_i32;
                int32_t v353_a = 6 + 0_i32;
                int32_t v356_a = 6 + 0_i32;
                int32_t v359_a = 6 + 0_i32;
                int32_t v362_a = 6 + 0_i32;
                int32_t v365_a = 6 + 0_i32;
                int32_t v368_a = 6 + 0_i32;
                int32_t v371_a = 6 + 0_i32;
                int32_t v374_a = 6 + 0_i32;
                int32_t v377_a = 6 + 0_i32;
                int32_t v380_a = 6 + 0_i32;
                int32_t v383_a = 6 + 0_i32;
                int32_t v386_a = 6 + 0_i32;
                int32_t v389_a = 6 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v396_a = 7 + 0_i32;
                int32_t v399_a = 7 + 0_i32;
                int32_t v402_a = 7 + 0_i32;
                int32_t v405_a = 7 + 0_i32;
                int32_t v408_a = 7 + 0_i32;
                int32_t v411_a = 7 + 0_i32;
                int32_t v414_a = 7 + 0_i32;
                int32_t v417_a = 7 + 0_i32;
                int32_t v420_a = 7 + 0_i32;
                int32_t v423_a = 7 + 0_i32;
                int32_t v426_a = 7 + 0_i32;
                int32_t v429_a = 7 + 0_i32;
                int32_t v432_a = 7 + 0_i32;
                int32_t v435_a = 7 + 0_i32;
                int32_t v438_a = 7 + 0_i32;
                int32_t v441_a = 7 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v448_a = 8 + 0_i32;
                int32_t v451_a = 8 + 0_i32;
                int32_t v454_a = 8 + 0_i32;
                int32_t v457_a = 8 + 0_i32;
                int32_t v460_a = 8 + 0_i32;
                int32_t v463_a = 8 + 0_i32;
                int32_t v466_a = 8 + 0_i32;
                int32_t v469_a = 8 + 0_i32;
                int32_t v472_a = 8 + 0_i32;
                int32_t v475_a = 8 + 0_i32;
                int32_t v478_a = 8 + 0_i32;
                int32_t v481_a = 8 + 0_i32;
                int32_t v484_a = 8 + 0_i32;
                int32_t v487_a = 8 + 0_i32;
                int32_t v490_a = 8 + 0_i32;
                int32_t v493_a = 8 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v500_a = 9 + 0_i32;
                int32_t v503_a = 9 + 0_i32;
                int32_t v506_a = 9 + 0_i32;
                int32_t v509_a = 9 + 0_i32;
                int32_t v512_a = 9 + 0_i32;
                int32_t v515_a = 9 + 0_i32;
                int32_t v518_a = 9 + 0_i32;
                int32_t v521_a = 9 + 0_i32;
                int32_t v524_a = 9 + 0_i32;
                int32_t v527_a = 9 + 0_i32;
                int32_t v530_a = 9 + 0_i32;
                int32_t v533_a = 9 + 0_i32;
                int32_t v536_a = 9 + 0_i32;
                int32_t v539_a = 9 + 0_i32;
                int32_t v542_a = 9 + 0_i32;
                int32_t v545_a = 9 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v552_a = 10 + 0_i32;
                int32_t v555_a = 10 + 0_i32;
                int32_t v558_a = 10 + 0_i32;
                int32_t v561_a = 10 + 0_i32;
                int32_t v564_a = 10 + 0_i32;
                int32_t v567_a = 10 + 0_i32;
                int32_t v570_a = 10 + 0_i32;
                int32_t v573_a = 10 + 0_i32;
                int32_t v576_a = 10 + 0_i32;
                int32_t v579_a = 10 + 0_i32;
                int32_t v582_a = 10 + 0_i32;
                int32_t v585_a = 10 + 0_i32;
                int32_t v588_a = 10 + 0_i32;
                int32_t v591_a = 10 + 0_i32;
                int32_t v594_a = 10 + 0_i32;
                int32_t v597_a = 10 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v604_a = 11 + 0_i32;
                int32_t v607_a = 11 + 0_i32;
                int32_t v610_a = 11 + 0_i32;
                int32_t v613_a = 11 + 0_i32;
                int32_t v616_a = 11 + 0_i32;
                int32_t v619_a = 11 + 0_i32;
                int32_t v622_a = 11 + 0_i32;
                int32_t v625_a = 11 + 0_i32;
                int32_t v628_a = 11 + 0_i32;
                int32_t v631_a = 11 + 0_i32;
                int32_t v634_a = 11 + 0_i32;
                int32_t v637_a = 11 + 0_i32;
                int32_t v640_a = 11 + 0_i32;
                int32_t v643_a = 11 + 0_i32;
                int32_t v646_a = 11 + 0_i32;
                int32_t v649_a = 11 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v656_a = 12 + 0_i32;
                int32_t v659_a = 12 + 0_i32;
                int32_t v662_a = 12 + 0_i32;
                int32_t v665_a = 12 + 0_i32;
                int32_t v668_a = 12 + 0_i32;
                int32_t v671_a = 12 + 0_i32;
                int32_t v674_a = 12 + 0_i32;
                int32_t v677_a = 12 + 0_i32;
                int32_t v680_a = 12 + 0_i32;
                int32_t v683_a = 12 + 0_i32;
                int32_t v686_a = 12 + 0_i32;
                int32_t v689_a = 12 + 0_i32;
                int32_t v692_a = 12 + 0_i32;
                int32_t v695_a = 12 + 0_i32;
                int32_t v698_a = 12 + 0_i32;
                int32_t v701_a = 12 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v708_a = 13 + 0_i32;
                int32_t v711_a = 13 + 0_i32;
                int32_t v714_a = 13 + 0_i32;
                int32_t v717_a = 13 + 0_i32;
                int32_t v720_a = 13 + 0_i32;
                int32_t v723_a = 13 + 0_i32;
                int32_t v726_a = 13 + 0_i32;
                int32_t v729_a = 13 + 0_i32;
                int32_t v732_a = 13 + 0_i32;
                int32_t v735_a = 13 + 0_i32;
                int32_t v738_a = 13 + 0_i32;
                int32_t v741_a = 13 + 0_i32;
                int32_t v744_a = 13 + 0_i32;
                int32_t v747_a = 13 + 0_i32;
                int32_t v750_a = 13 + 0_i32;
                int32_t v753_a = 13 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v760_a = 14 + 0_i32;
                int32_t v763_a = 14 + 0_i32;
                int32_t v766_a = 14 + 0_i32;
                int32_t v769_a = 14 + 0_i32;
                int32_t v772_a = 14 + 0_i32;
                int32_t v775_a = 14 + 0_i32;
                int32_t v778_a = 14 + 0_i32;
                int32_t v781_a = 14 + 0_i32;
                int32_t v784_a = 14 + 0_i32;
                int32_t v787_a = 14 + 0_i32;
                int32_t v790_a = 14 + 0_i32;
                int32_t v793_a = 14 + 0_i32;
                int32_t v796_a = 14 + 0_i32;
                int32_t v799_a = 14 + 0_i32;
                int32_t v802_a = 14 + 0_i32;
                int32_t v805_a = 14 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v812_a = 15 + 0_i32;
                int32_t v815_a = 15 + 0_i32;
                int32_t v818_a = 15 + 0_i32;
                int32_t v821_a = 15 + 0_i32;
                int32_t v824_a = 15 + 0_i32;
                int32_t v827_a = 15 + 0_i32;
                int32_t v830_a = 15 + 0_i32;
                int32_t v833_a = 15 + 0_i32;
                int32_t v836_a = 15 + 0_i32;
                int32_t v839_a = 15 + 0_i32;
                int32_t v842_a = 15 + 0_i32;
                int32_t v845_a = 15 + 0_i32;
                int32_t v848_a = 15 + 0_i32;
                int32_t v851_a = 15 + 0_i32;
                int32_t v854_a = 15 + 0_i32;
                int32_t v857_a = 15 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v864_a = 16 + 0_i32;
                int32_t v867_a = 16 + 0_i32;
                int32_t v870_a = 16 + 0_i32;
                int32_t v873_a = 16 + 0_i32;
                int32_t v876_a = 16 + 0_i32;
                int32_t v879_a = 16 + 0_i32;
                int32_t v882_a = 16 + 0_i32;
                int32_t v885_a = 16 + 0_i32;
                int32_t v888_a = 16 + 0_i32;
                int32_t v891_a = 16 + 0_i32;
                int32_t v894_a = 16 + 0_i32;
                int32_t v897_a = 16 + 0_i32;
                int32_t v900_a = 16 + 0_i32;
                int32_t v903_a = 16 + 0_i32;
                int32_t v906_a = 16 + 0_i32;
                int32_t v909_a = 16 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v916_a = 17 + 0_i32;
                int32_t v919_a = 17 + 0_i32;
                int32_t v922_a = 17 + 0_i32;
                int32_t v925_a = 17 + 0_i32;
                int32_t v928_a = 17 + 0_i32;
                int32_t v931_a = 17 + 0_i32;
                int32_t v934_a = 17 + 0_i32;
                int32_t v937_a = 17 + 0_i32;
                int32_t v940_a = 17 + 0_i32;
                int32_t v943_a = 17 + 0_i32;
                int32_t v946_a = 17 + 0_i32;
                int32_t v949_a = 17 + 0_i32;
                int32_t v952_a = 17 + 0_i32;
                int32_t v955_a = 17 + 0_i32;
                int32_t v958_a = 17 + 0_i32;
                int32_t v961_a = 17 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v968_a = 18 + 0_i32;
                int32_t v971_a = 18 + 0_i32;
                int32_t v974_a = 18 + 0_i32;
                int32_t v977_a = 18 + 0_i32;
                int32_t v980_a = 18 + 0_i32;
                int32_t v983_a = 18 + 0_i32;
                int32_t v986_a = 18 + 0_i32;
                int32_t v989_a = 18 + 0_i32;
                int32_t v992_a = 18 + 0_i32;
                int32_t v995_a = 18 + 0_i32;
                int32_t v998_a = 18 + 0_i32;
                int32_t v1001_a = 18 + 0_i32;
                int32_t v1004_a = 18 + 0_i32;
                int32_t v1007_a = 18 + 0_i32;
                int32_t v1010_a = 18 + 0_i32;
                int32_t v1013_a = 18 + 0_i32;
              }
              if (v6_lead < 12) {
                int32_t v1020_a = 19 + 0_i32;
                int32_t v1023_a = 19 + 0_i32;
                int32_t v1026_a = 19 + 0_i32;
                int32_t v1029_a = 19 + 0_i32;
                int32_t v1032_a = 19 + 0_i32;
                int32_t v1035_a = 19 + 0_i32;
                int32_t v1038_a = 19 + 0_i32;
                int32_t v1041_a = 19 + 0_i32;
                int32_t v1044_a = 19 + 0_i32;
                int32_t v1047_a = 19 + 0_i32;
                int32_t v1050_a = 19 + 0_i32;
                int32_t v1053_a = 19 + 0_i32;
                int32_t v1056_a = 19 + 0_i32;
                int32_t v1059_a = 19 + 0_i32;
                int32_t v1062_a = 19 + 0_i32;
                int32_t v1065_a = 19 + 0_i32;
              }
              if (v6_lead < 12) {
                #pragma unroll
                for (int32_t v1070_n1 = 0; v1070_n1 < 16; ++v1070_n1) {
                  int32_t v1071_a = 0 + v1070_n1;
                  int32_t v1072_a = 0 + v1070_n1;
                  None = r0[v1072_a];
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v6_lead < 12) {
                #pragma unroll
                for (int32_t v1077_i1 = 0; v1077_i1 < 16; ++v1077_i1) {
                  int32_t v1078_a = 0 + v1077_i1;
                  int32_t v1081_a = 0_i32 + (v1077_i1 * 12);
                  None.copy_to(glb_m0[v1081_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}


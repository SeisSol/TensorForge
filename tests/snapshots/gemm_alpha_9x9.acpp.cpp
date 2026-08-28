// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08a27dccde(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08a27dccde(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 9×9(9×9) {0..9}×{0..9} strided
        // m1 9×9(9×9) {0..9}×{0..9} strided
        // m2 9×9(9×9) {0..9}×{0..9} strided
        // m3 ()  scalar
        // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              s0[0 + 0 + 1 * item.get_local_id(0) + 64] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 64];
              if (item.get_local_id(0) < 1) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 80] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 80];
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[9]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir0[9]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 9) {
                float v16_data = glb_m1[v8_lead];
                float v17_data = s0[0];
                float v19_data = ir0[0];
                ir0[0] = (v19_data + (v16_data * v17_data));
                float v27_data = glb_m1[v8_lead];
                float v28_data = s0[9];
                float v30_data = ir0[1];
                ir0[1] = (v30_data + (v27_data * v28_data));
                float v38_data = glb_m1[v8_lead];
                float v39_data = s0[18];
                float v41_data = ir0[2];
                ir0[2] = (v41_data + (v38_data * v39_data));
                float v49_data = glb_m1[v8_lead];
                float v50_data = s0[27];
                float v52_data = ir0[3];
                ir0[3] = (v52_data + (v49_data * v50_data));
                float v60_data = glb_m1[v8_lead];
                float v61_data = s0[36];
                float v63_data = ir0[4];
                ir0[4] = (v63_data + (v60_data * v61_data));
                float v71_data = glb_m1[v8_lead];
                float v72_data = s0[45];
                float v74_data = ir0[5];
                ir0[5] = (v74_data + (v71_data * v72_data));
                float v82_data = glb_m1[v8_lead];
                float v83_data = s0[54];
                float v85_data = ir0[6];
                ir0[6] = (v85_data + (v82_data * v83_data));
                float v93_data = glb_m1[v8_lead];
                float v94_data = s0[63];
                float v96_data = ir0[7];
                ir0[7] = (v96_data + (v93_data * v94_data));
                float v104_data = glb_m1[v8_lead];
                float v105_data = s0[72];
                float v107_data = ir0[8];
                ir0[8] = (v107_data + (v104_data * v105_data));
              }
              if (v8_lead < 9) {
                float v119_data = glb_m1[(v8_lead + 9)];
                float v120_data = s0[1];
                float v122_data = ir0[0];
                ir0[0] = (v122_data + (v119_data * v120_data));
                float v130_data = glb_m1[(v8_lead + 9)];
                float v131_data = s0[10];
                float v133_data = ir0[1];
                ir0[1] = (v133_data + (v130_data * v131_data));
                float v141_data = glb_m1[(v8_lead + 9)];
                float v142_data = s0[19];
                float v144_data = ir0[2];
                ir0[2] = (v144_data + (v141_data * v142_data));
                float v152_data = glb_m1[(v8_lead + 9)];
                float v153_data = s0[28];
                float v155_data = ir0[3];
                ir0[3] = (v155_data + (v152_data * v153_data));
                float v163_data = glb_m1[(v8_lead + 9)];
                float v164_data = s0[37];
                float v166_data = ir0[4];
                ir0[4] = (v166_data + (v163_data * v164_data));
                float v174_data = glb_m1[(v8_lead + 9)];
                float v175_data = s0[46];
                float v177_data = ir0[5];
                ir0[5] = (v177_data + (v174_data * v175_data));
                float v185_data = glb_m1[(v8_lead + 9)];
                float v186_data = s0[55];
                float v188_data = ir0[6];
                ir0[6] = (v188_data + (v185_data * v186_data));
                float v196_data = glb_m1[(v8_lead + 9)];
                float v197_data = s0[64];
                float v199_data = ir0[7];
                ir0[7] = (v199_data + (v196_data * v197_data));
                float v207_data = glb_m1[(v8_lead + 9)];
                float v208_data = s0[73];
                float v210_data = ir0[8];
                ir0[8] = (v210_data + (v207_data * v208_data));
              }
              if (v8_lead < 9) {
                float v222_data = glb_m1[(v8_lead + 18)];
                float v223_data = s0[2];
                float v225_data = ir0[0];
                ir0[0] = (v225_data + (v222_data * v223_data));
                float v233_data = glb_m1[(v8_lead + 18)];
                float v234_data = s0[11];
                float v236_data = ir0[1];
                ir0[1] = (v236_data + (v233_data * v234_data));
                float v244_data = glb_m1[(v8_lead + 18)];
                float v245_data = s0[20];
                float v247_data = ir0[2];
                ir0[2] = (v247_data + (v244_data * v245_data));
                float v255_data = glb_m1[(v8_lead + 18)];
                float v256_data = s0[29];
                float v258_data = ir0[3];
                ir0[3] = (v258_data + (v255_data * v256_data));
                float v266_data = glb_m1[(v8_lead + 18)];
                float v267_data = s0[38];
                float v269_data = ir0[4];
                ir0[4] = (v269_data + (v266_data * v267_data));
                float v277_data = glb_m1[(v8_lead + 18)];
                float v278_data = s0[47];
                float v280_data = ir0[5];
                ir0[5] = (v280_data + (v277_data * v278_data));
                float v288_data = glb_m1[(v8_lead + 18)];
                float v289_data = s0[56];
                float v291_data = ir0[6];
                ir0[6] = (v291_data + (v288_data * v289_data));
                float v299_data = glb_m1[(v8_lead + 18)];
                float v300_data = s0[65];
                float v302_data = ir0[7];
                ir0[7] = (v302_data + (v299_data * v300_data));
                float v310_data = glb_m1[(v8_lead + 18)];
                float v311_data = s0[74];
                float v313_data = ir0[8];
                ir0[8] = (v313_data + (v310_data * v311_data));
              }
              if (v8_lead < 9) {
                float v325_data = glb_m1[(v8_lead + 27)];
                float v326_data = s0[3];
                float v328_data = ir0[0];
                ir0[0] = (v328_data + (v325_data * v326_data));
                float v336_data = glb_m1[(v8_lead + 27)];
                float v337_data = s0[12];
                float v339_data = ir0[1];
                ir0[1] = (v339_data + (v336_data * v337_data));
                float v347_data = glb_m1[(v8_lead + 27)];
                float v348_data = s0[21];
                float v350_data = ir0[2];
                ir0[2] = (v350_data + (v347_data * v348_data));
                float v358_data = glb_m1[(v8_lead + 27)];
                float v359_data = s0[30];
                float v361_data = ir0[3];
                ir0[3] = (v361_data + (v358_data * v359_data));
                float v369_data = glb_m1[(v8_lead + 27)];
                float v370_data = s0[39];
                float v372_data = ir0[4];
                ir0[4] = (v372_data + (v369_data * v370_data));
                float v380_data = glb_m1[(v8_lead + 27)];
                float v381_data = s0[48];
                float v383_data = ir0[5];
                ir0[5] = (v383_data + (v380_data * v381_data));
                float v391_data = glb_m1[(v8_lead + 27)];
                float v392_data = s0[57];
                float v394_data = ir0[6];
                ir0[6] = (v394_data + (v391_data * v392_data));
                float v402_data = glb_m1[(v8_lead + 27)];
                float v403_data = s0[66];
                float v405_data = ir0[7];
                ir0[7] = (v405_data + (v402_data * v403_data));
                float v413_data = glb_m1[(v8_lead + 27)];
                float v414_data = s0[75];
                float v416_data = ir0[8];
                ir0[8] = (v416_data + (v413_data * v414_data));
              }
              if (v8_lead < 9) {
                float v428_data = glb_m1[(v8_lead + 36)];
                float v429_data = s0[4];
                float v431_data = ir0[0];
                ir0[0] = (v431_data + (v428_data * v429_data));
                float v439_data = glb_m1[(v8_lead + 36)];
                float v440_data = s0[13];
                float v442_data = ir0[1];
                ir0[1] = (v442_data + (v439_data * v440_data));
                float v450_data = glb_m1[(v8_lead + 36)];
                float v451_data = s0[22];
                float v453_data = ir0[2];
                ir0[2] = (v453_data + (v450_data * v451_data));
                float v461_data = glb_m1[(v8_lead + 36)];
                float v462_data = s0[31];
                float v464_data = ir0[3];
                ir0[3] = (v464_data + (v461_data * v462_data));
                float v472_data = glb_m1[(v8_lead + 36)];
                float v473_data = s0[40];
                float v475_data = ir0[4];
                ir0[4] = (v475_data + (v472_data * v473_data));
                float v483_data = glb_m1[(v8_lead + 36)];
                float v484_data = s0[49];
                float v486_data = ir0[5];
                ir0[5] = (v486_data + (v483_data * v484_data));
                float v494_data = glb_m1[(v8_lead + 36)];
                float v495_data = s0[58];
                float v497_data = ir0[6];
                ir0[6] = (v497_data + (v494_data * v495_data));
                float v505_data = glb_m1[(v8_lead + 36)];
                float v506_data = s0[67];
                float v508_data = ir0[7];
                ir0[7] = (v508_data + (v505_data * v506_data));
                float v516_data = glb_m1[(v8_lead + 36)];
                float v517_data = s0[76];
                float v519_data = ir0[8];
                ir0[8] = (v519_data + (v516_data * v517_data));
              }
              if (v8_lead < 9) {
                float v531_data = glb_m1[(v8_lead + 45)];
                float v532_data = s0[5];
                float v534_data = ir0[0];
                ir0[0] = (v534_data + (v531_data * v532_data));
                float v542_data = glb_m1[(v8_lead + 45)];
                float v543_data = s0[14];
                float v545_data = ir0[1];
                ir0[1] = (v545_data + (v542_data * v543_data));
                float v553_data = glb_m1[(v8_lead + 45)];
                float v554_data = s0[23];
                float v556_data = ir0[2];
                ir0[2] = (v556_data + (v553_data * v554_data));
                float v564_data = glb_m1[(v8_lead + 45)];
                float v565_data = s0[32];
                float v567_data = ir0[3];
                ir0[3] = (v567_data + (v564_data * v565_data));
                float v575_data = glb_m1[(v8_lead + 45)];
                float v576_data = s0[41];
                float v578_data = ir0[4];
                ir0[4] = (v578_data + (v575_data * v576_data));
                float v586_data = glb_m1[(v8_lead + 45)];
                float v587_data = s0[50];
                float v589_data = ir0[5];
                ir0[5] = (v589_data + (v586_data * v587_data));
                float v597_data = glb_m1[(v8_lead + 45)];
                float v598_data = s0[59];
                float v600_data = ir0[6];
                ir0[6] = (v600_data + (v597_data * v598_data));
                float v608_data = glb_m1[(v8_lead + 45)];
                float v609_data = s0[68];
                float v611_data = ir0[7];
                ir0[7] = (v611_data + (v608_data * v609_data));
                float v619_data = glb_m1[(v8_lead + 45)];
                float v620_data = s0[77];
                float v622_data = ir0[8];
                ir0[8] = (v622_data + (v619_data * v620_data));
              }
              if (v8_lead < 9) {
                float v634_data = glb_m1[(v8_lead + 54)];
                float v635_data = s0[6];
                float v637_data = ir0[0];
                ir0[0] = (v637_data + (v634_data * v635_data));
                float v645_data = glb_m1[(v8_lead + 54)];
                float v646_data = s0[15];
                float v648_data = ir0[1];
                ir0[1] = (v648_data + (v645_data * v646_data));
                float v656_data = glb_m1[(v8_lead + 54)];
                float v657_data = s0[24];
                float v659_data = ir0[2];
                ir0[2] = (v659_data + (v656_data * v657_data));
                float v667_data = glb_m1[(v8_lead + 54)];
                float v668_data = s0[33];
                float v670_data = ir0[3];
                ir0[3] = (v670_data + (v667_data * v668_data));
                float v678_data = glb_m1[(v8_lead + 54)];
                float v679_data = s0[42];
                float v681_data = ir0[4];
                ir0[4] = (v681_data + (v678_data * v679_data));
                float v689_data = glb_m1[(v8_lead + 54)];
                float v690_data = s0[51];
                float v692_data = ir0[5];
                ir0[5] = (v692_data + (v689_data * v690_data));
                float v700_data = glb_m1[(v8_lead + 54)];
                float v701_data = s0[60];
                float v703_data = ir0[6];
                ir0[6] = (v703_data + (v700_data * v701_data));
                float v711_data = glb_m1[(v8_lead + 54)];
                float v712_data = s0[69];
                float v714_data = ir0[7];
                ir0[7] = (v714_data + (v711_data * v712_data));
                float v722_data = glb_m1[(v8_lead + 54)];
                float v723_data = s0[78];
                float v725_data = ir0[8];
                ir0[8] = (v725_data + (v722_data * v723_data));
              }
              if (v8_lead < 9) {
                float v737_data = glb_m1[(v8_lead + 63)];
                float v738_data = s0[7];
                float v740_data = ir0[0];
                ir0[0] = (v740_data + (v737_data * v738_data));
                float v748_data = glb_m1[(v8_lead + 63)];
                float v749_data = s0[16];
                float v751_data = ir0[1];
                ir0[1] = (v751_data + (v748_data * v749_data));
                float v759_data = glb_m1[(v8_lead + 63)];
                float v760_data = s0[25];
                float v762_data = ir0[2];
                ir0[2] = (v762_data + (v759_data * v760_data));
                float v770_data = glb_m1[(v8_lead + 63)];
                float v771_data = s0[34];
                float v773_data = ir0[3];
                ir0[3] = (v773_data + (v770_data * v771_data));
                float v781_data = glb_m1[(v8_lead + 63)];
                float v782_data = s0[43];
                float v784_data = ir0[4];
                ir0[4] = (v784_data + (v781_data * v782_data));
                float v792_data = glb_m1[(v8_lead + 63)];
                float v793_data = s0[52];
                float v795_data = ir0[5];
                ir0[5] = (v795_data + (v792_data * v793_data));
                float v803_data = glb_m1[(v8_lead + 63)];
                float v804_data = s0[61];
                float v806_data = ir0[6];
                ir0[6] = (v806_data + (v803_data * v804_data));
                float v814_data = glb_m1[(v8_lead + 63)];
                float v815_data = s0[70];
                float v817_data = ir0[7];
                ir0[7] = (v817_data + (v814_data * v815_data));
                float v825_data = glb_m1[(v8_lead + 63)];
                float v826_data = s0[79];
                float v828_data = ir0[8];
                ir0[8] = (v828_data + (v825_data * v826_data));
              }
              if (v8_lead < 9) {
                float v840_data = glb_m1[(v8_lead + 72)];
                float v841_data = s0[8];
                float v843_data = ir0[0];
                ir0[0] = (v843_data + (v840_data * v841_data));
                float v851_data = glb_m1[(v8_lead + 72)];
                float v852_data = s0[17];
                float v854_data = ir0[1];
                ir0[1] = (v854_data + (v851_data * v852_data));
                float v862_data = glb_m1[(v8_lead + 72)];
                float v863_data = s0[26];
                float v865_data = ir0[2];
                ir0[2] = (v865_data + (v862_data * v863_data));
                float v873_data = glb_m1[(v8_lead + 72)];
                float v874_data = s0[35];
                float v876_data = ir0[3];
                ir0[3] = (v876_data + (v873_data * v874_data));
                float v884_data = glb_m1[(v8_lead + 72)];
                float v885_data = s0[44];
                float v887_data = ir0[4];
                ir0[4] = (v887_data + (v884_data * v885_data));
                float v895_data = glb_m1[(v8_lead + 72)];
                float v896_data = s0[53];
                float v898_data = ir0[5];
                ir0[5] = (v898_data + (v895_data * v896_data));
                float v906_data = glb_m1[(v8_lead + 72)];
                float v907_data = s0[62];
                float v909_data = ir0[6];
                ir0[6] = (v909_data + (v906_data * v907_data));
                float v917_data = glb_m1[(v8_lead + 72)];
                float v918_data = s0[71];
                float v920_data = ir0[7];
                ir0[7] = (v920_data + (v917_data * v918_data));
                float v928_data = glb_m1[(v8_lead + 72)];
                float v929_data = s0[80];
                float v931_data = ir0[8];
                ir0[8] = (v931_data + (v928_data * v929_data));
              }
              if (v8_lead < 9) {
                #pragma unroll
                for (int32_t v938_n1 = 0; v938_n1 < 9; ++v938_n1) {
                  float v940_data = ir0[v938_n1];
                  r0[v938_n1] = (v940_data * 13.0f);
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 9) {
                #pragma unroll
                for (int32_t v947_i1 = 0; v947_i1 < 9; ++v947_i1) {
                  float v949_data = r0[v947_i1];
                  glb_m0[(v8_lead + (v947_i1 * 9))] = v949_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


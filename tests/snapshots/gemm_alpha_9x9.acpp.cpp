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
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
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
                int32_t v15_a = v8_lead + 0;
                float v22_data = glb_m1[v8_lead];
                float v23_data = s0[0];
                float v25_data = ir0[0];
                ir0[0] = (v25_data + (v22_data * v23_data));
                int32_t v32_a = v8_lead + 0;
                float v39_data = glb_m1[v8_lead];
                float v40_data = s0[9];
                float v42_data = ir0[1];
                ir0[1] = (v42_data + (v39_data * v40_data));
                int32_t v49_a = v8_lead + 0;
                float v56_data = glb_m1[v8_lead];
                float v57_data = s0[18];
                float v59_data = ir0[2];
                ir0[2] = (v59_data + (v56_data * v57_data));
                int32_t v66_a = v8_lead + 0;
                float v73_data = glb_m1[v8_lead];
                float v74_data = s0[27];
                float v76_data = ir0[3];
                ir0[3] = (v76_data + (v73_data * v74_data));
                int32_t v83_a = v8_lead + 0;
                float v90_data = glb_m1[v8_lead];
                float v91_data = s0[36];
                float v93_data = ir0[4];
                ir0[4] = (v93_data + (v90_data * v91_data));
                int32_t v100_a = v8_lead + 0;
                float v107_data = glb_m1[v8_lead];
                float v108_data = s0[45];
                float v110_data = ir0[5];
                ir0[5] = (v110_data + (v107_data * v108_data));
                int32_t v117_a = v8_lead + 0;
                float v124_data = glb_m1[v8_lead];
                float v125_data = s0[54];
                float v127_data = ir0[6];
                ir0[6] = (v127_data + (v124_data * v125_data));
                int32_t v134_a = v8_lead + 0;
                float v141_data = glb_m1[v8_lead];
                float v142_data = s0[63];
                float v144_data = ir0[7];
                ir0[7] = (v144_data + (v141_data * v142_data));
                int32_t v151_a = v8_lead + 0;
                float v158_data = glb_m1[v8_lead];
                float v159_data = s0[72];
                float v161_data = ir0[8];
                ir0[8] = (v161_data + (v158_data * v159_data));
              }
              if (v8_lead < 9) {
                int32_t v172_a = v8_lead + 9;
                float v179_data = glb_m1[(v8_lead + 9)];
                float v180_data = s0[1];
                float v182_data = ir0[0];
                ir0[0] = (v182_data + (v179_data * v180_data));
                int32_t v189_a = v8_lead + 9;
                float v196_data = glb_m1[(v8_lead + 9)];
                float v197_data = s0[10];
                float v199_data = ir0[1];
                ir0[1] = (v199_data + (v196_data * v197_data));
                int32_t v206_a = v8_lead + 9;
                float v213_data = glb_m1[(v8_lead + 9)];
                float v214_data = s0[19];
                float v216_data = ir0[2];
                ir0[2] = (v216_data + (v213_data * v214_data));
                int32_t v223_a = v8_lead + 9;
                float v230_data = glb_m1[(v8_lead + 9)];
                float v231_data = s0[28];
                float v233_data = ir0[3];
                ir0[3] = (v233_data + (v230_data * v231_data));
                int32_t v240_a = v8_lead + 9;
                float v247_data = glb_m1[(v8_lead + 9)];
                float v248_data = s0[37];
                float v250_data = ir0[4];
                ir0[4] = (v250_data + (v247_data * v248_data));
                int32_t v257_a = v8_lead + 9;
                float v264_data = glb_m1[(v8_lead + 9)];
                float v265_data = s0[46];
                float v267_data = ir0[5];
                ir0[5] = (v267_data + (v264_data * v265_data));
                int32_t v274_a = v8_lead + 9;
                float v281_data = glb_m1[(v8_lead + 9)];
                float v282_data = s0[55];
                float v284_data = ir0[6];
                ir0[6] = (v284_data + (v281_data * v282_data));
                int32_t v291_a = v8_lead + 9;
                float v298_data = glb_m1[(v8_lead + 9)];
                float v299_data = s0[64];
                float v301_data = ir0[7];
                ir0[7] = (v301_data + (v298_data * v299_data));
                int32_t v308_a = v8_lead + 9;
                float v315_data = glb_m1[(v8_lead + 9)];
                float v316_data = s0[73];
                float v318_data = ir0[8];
                ir0[8] = (v318_data + (v315_data * v316_data));
              }
              if (v8_lead < 9) {
                int32_t v329_a = v8_lead + 18;
                float v336_data = glb_m1[(v8_lead + 18)];
                float v337_data = s0[2];
                float v339_data = ir0[0];
                ir0[0] = (v339_data + (v336_data * v337_data));
                int32_t v346_a = v8_lead + 18;
                float v353_data = glb_m1[(v8_lead + 18)];
                float v354_data = s0[11];
                float v356_data = ir0[1];
                ir0[1] = (v356_data + (v353_data * v354_data));
                int32_t v363_a = v8_lead + 18;
                float v370_data = glb_m1[(v8_lead + 18)];
                float v371_data = s0[20];
                float v373_data = ir0[2];
                ir0[2] = (v373_data + (v370_data * v371_data));
                int32_t v380_a = v8_lead + 18;
                float v387_data = glb_m1[(v8_lead + 18)];
                float v388_data = s0[29];
                float v390_data = ir0[3];
                ir0[3] = (v390_data + (v387_data * v388_data));
                int32_t v397_a = v8_lead + 18;
                float v404_data = glb_m1[(v8_lead + 18)];
                float v405_data = s0[38];
                float v407_data = ir0[4];
                ir0[4] = (v407_data + (v404_data * v405_data));
                int32_t v414_a = v8_lead + 18;
                float v421_data = glb_m1[(v8_lead + 18)];
                float v422_data = s0[47];
                float v424_data = ir0[5];
                ir0[5] = (v424_data + (v421_data * v422_data));
                int32_t v431_a = v8_lead + 18;
                float v438_data = glb_m1[(v8_lead + 18)];
                float v439_data = s0[56];
                float v441_data = ir0[6];
                ir0[6] = (v441_data + (v438_data * v439_data));
                int32_t v448_a = v8_lead + 18;
                float v455_data = glb_m1[(v8_lead + 18)];
                float v456_data = s0[65];
                float v458_data = ir0[7];
                ir0[7] = (v458_data + (v455_data * v456_data));
                int32_t v465_a = v8_lead + 18;
                float v472_data = glb_m1[(v8_lead + 18)];
                float v473_data = s0[74];
                float v475_data = ir0[8];
                ir0[8] = (v475_data + (v472_data * v473_data));
              }
              if (v8_lead < 9) {
                int32_t v486_a = v8_lead + 27;
                float v493_data = glb_m1[(v8_lead + 27)];
                float v494_data = s0[3];
                float v496_data = ir0[0];
                ir0[0] = (v496_data + (v493_data * v494_data));
                int32_t v503_a = v8_lead + 27;
                float v510_data = glb_m1[(v8_lead + 27)];
                float v511_data = s0[12];
                float v513_data = ir0[1];
                ir0[1] = (v513_data + (v510_data * v511_data));
                int32_t v520_a = v8_lead + 27;
                float v527_data = glb_m1[(v8_lead + 27)];
                float v528_data = s0[21];
                float v530_data = ir0[2];
                ir0[2] = (v530_data + (v527_data * v528_data));
                int32_t v537_a = v8_lead + 27;
                float v544_data = glb_m1[(v8_lead + 27)];
                float v545_data = s0[30];
                float v547_data = ir0[3];
                ir0[3] = (v547_data + (v544_data * v545_data));
                int32_t v554_a = v8_lead + 27;
                float v561_data = glb_m1[(v8_lead + 27)];
                float v562_data = s0[39];
                float v564_data = ir0[4];
                ir0[4] = (v564_data + (v561_data * v562_data));
                int32_t v571_a = v8_lead + 27;
                float v578_data = glb_m1[(v8_lead + 27)];
                float v579_data = s0[48];
                float v581_data = ir0[5];
                ir0[5] = (v581_data + (v578_data * v579_data));
                int32_t v588_a = v8_lead + 27;
                float v595_data = glb_m1[(v8_lead + 27)];
                float v596_data = s0[57];
                float v598_data = ir0[6];
                ir0[6] = (v598_data + (v595_data * v596_data));
                int32_t v605_a = v8_lead + 27;
                float v612_data = glb_m1[(v8_lead + 27)];
                float v613_data = s0[66];
                float v615_data = ir0[7];
                ir0[7] = (v615_data + (v612_data * v613_data));
                int32_t v622_a = v8_lead + 27;
                float v629_data = glb_m1[(v8_lead + 27)];
                float v630_data = s0[75];
                float v632_data = ir0[8];
                ir0[8] = (v632_data + (v629_data * v630_data));
              }
              if (v8_lead < 9) {
                int32_t v643_a = v8_lead + 36;
                float v650_data = glb_m1[(v8_lead + 36)];
                float v651_data = s0[4];
                float v653_data = ir0[0];
                ir0[0] = (v653_data + (v650_data * v651_data));
                int32_t v660_a = v8_lead + 36;
                float v667_data = glb_m1[(v8_lead + 36)];
                float v668_data = s0[13];
                float v670_data = ir0[1];
                ir0[1] = (v670_data + (v667_data * v668_data));
                int32_t v677_a = v8_lead + 36;
                float v684_data = glb_m1[(v8_lead + 36)];
                float v685_data = s0[22];
                float v687_data = ir0[2];
                ir0[2] = (v687_data + (v684_data * v685_data));
                int32_t v694_a = v8_lead + 36;
                float v701_data = glb_m1[(v8_lead + 36)];
                float v702_data = s0[31];
                float v704_data = ir0[3];
                ir0[3] = (v704_data + (v701_data * v702_data));
                int32_t v711_a = v8_lead + 36;
                float v718_data = glb_m1[(v8_lead + 36)];
                float v719_data = s0[40];
                float v721_data = ir0[4];
                ir0[4] = (v721_data + (v718_data * v719_data));
                int32_t v728_a = v8_lead + 36;
                float v735_data = glb_m1[(v8_lead + 36)];
                float v736_data = s0[49];
                float v738_data = ir0[5];
                ir0[5] = (v738_data + (v735_data * v736_data));
                int32_t v745_a = v8_lead + 36;
                float v752_data = glb_m1[(v8_lead + 36)];
                float v753_data = s0[58];
                float v755_data = ir0[6];
                ir0[6] = (v755_data + (v752_data * v753_data));
                int32_t v762_a = v8_lead + 36;
                float v769_data = glb_m1[(v8_lead + 36)];
                float v770_data = s0[67];
                float v772_data = ir0[7];
                ir0[7] = (v772_data + (v769_data * v770_data));
                int32_t v779_a = v8_lead + 36;
                float v786_data = glb_m1[(v8_lead + 36)];
                float v787_data = s0[76];
                float v789_data = ir0[8];
                ir0[8] = (v789_data + (v786_data * v787_data));
              }
              if (v8_lead < 9) {
                int32_t v800_a = v8_lead + 45;
                float v807_data = glb_m1[(v8_lead + 45)];
                float v808_data = s0[5];
                float v810_data = ir0[0];
                ir0[0] = (v810_data + (v807_data * v808_data));
                int32_t v817_a = v8_lead + 45;
                float v824_data = glb_m1[(v8_lead + 45)];
                float v825_data = s0[14];
                float v827_data = ir0[1];
                ir0[1] = (v827_data + (v824_data * v825_data));
                int32_t v834_a = v8_lead + 45;
                float v841_data = glb_m1[(v8_lead + 45)];
                float v842_data = s0[23];
                float v844_data = ir0[2];
                ir0[2] = (v844_data + (v841_data * v842_data));
                int32_t v851_a = v8_lead + 45;
                float v858_data = glb_m1[(v8_lead + 45)];
                float v859_data = s0[32];
                float v861_data = ir0[3];
                ir0[3] = (v861_data + (v858_data * v859_data));
                int32_t v868_a = v8_lead + 45;
                float v875_data = glb_m1[(v8_lead + 45)];
                float v876_data = s0[41];
                float v878_data = ir0[4];
                ir0[4] = (v878_data + (v875_data * v876_data));
                int32_t v885_a = v8_lead + 45;
                float v892_data = glb_m1[(v8_lead + 45)];
                float v893_data = s0[50];
                float v895_data = ir0[5];
                ir0[5] = (v895_data + (v892_data * v893_data));
                int32_t v902_a = v8_lead + 45;
                float v909_data = glb_m1[(v8_lead + 45)];
                float v910_data = s0[59];
                float v912_data = ir0[6];
                ir0[6] = (v912_data + (v909_data * v910_data));
                int32_t v919_a = v8_lead + 45;
                float v926_data = glb_m1[(v8_lead + 45)];
                float v927_data = s0[68];
                float v929_data = ir0[7];
                ir0[7] = (v929_data + (v926_data * v927_data));
                int32_t v936_a = v8_lead + 45;
                float v943_data = glb_m1[(v8_lead + 45)];
                float v944_data = s0[77];
                float v946_data = ir0[8];
                ir0[8] = (v946_data + (v943_data * v944_data));
              }
              if (v8_lead < 9) {
                int32_t v957_a = v8_lead + 54;
                float v964_data = glb_m1[(v8_lead + 54)];
                float v965_data = s0[6];
                float v967_data = ir0[0];
                ir0[0] = (v967_data + (v964_data * v965_data));
                int32_t v974_a = v8_lead + 54;
                float v981_data = glb_m1[(v8_lead + 54)];
                float v982_data = s0[15];
                float v984_data = ir0[1];
                ir0[1] = (v984_data + (v981_data * v982_data));
                int32_t v991_a = v8_lead + 54;
                float v998_data = glb_m1[(v8_lead + 54)];
                float v999_data = s0[24];
                float v1001_data = ir0[2];
                ir0[2] = (v1001_data + (v998_data * v999_data));
                int32_t v1008_a = v8_lead + 54;
                float v1015_data = glb_m1[(v8_lead + 54)];
                float v1016_data = s0[33];
                float v1018_data = ir0[3];
                ir0[3] = (v1018_data + (v1015_data * v1016_data));
                int32_t v1025_a = v8_lead + 54;
                float v1032_data = glb_m1[(v8_lead + 54)];
                float v1033_data = s0[42];
                float v1035_data = ir0[4];
                ir0[4] = (v1035_data + (v1032_data * v1033_data));
                int32_t v1042_a = v8_lead + 54;
                float v1049_data = glb_m1[(v8_lead + 54)];
                float v1050_data = s0[51];
                float v1052_data = ir0[5];
                ir0[5] = (v1052_data + (v1049_data * v1050_data));
                int32_t v1059_a = v8_lead + 54;
                float v1066_data = glb_m1[(v8_lead + 54)];
                float v1067_data = s0[60];
                float v1069_data = ir0[6];
                ir0[6] = (v1069_data + (v1066_data * v1067_data));
                int32_t v1076_a = v8_lead + 54;
                float v1083_data = glb_m1[(v8_lead + 54)];
                float v1084_data = s0[69];
                float v1086_data = ir0[7];
                ir0[7] = (v1086_data + (v1083_data * v1084_data));
                int32_t v1093_a = v8_lead + 54;
                float v1100_data = glb_m1[(v8_lead + 54)];
                float v1101_data = s0[78];
                float v1103_data = ir0[8];
                ir0[8] = (v1103_data + (v1100_data * v1101_data));
              }
              if (v8_lead < 9) {
                int32_t v1114_a = v8_lead + 63;
                float v1121_data = glb_m1[(v8_lead + 63)];
                float v1122_data = s0[7];
                float v1124_data = ir0[0];
                ir0[0] = (v1124_data + (v1121_data * v1122_data));
                int32_t v1131_a = v8_lead + 63;
                float v1138_data = glb_m1[(v8_lead + 63)];
                float v1139_data = s0[16];
                float v1141_data = ir0[1];
                ir0[1] = (v1141_data + (v1138_data * v1139_data));
                int32_t v1148_a = v8_lead + 63;
                float v1155_data = glb_m1[(v8_lead + 63)];
                float v1156_data = s0[25];
                float v1158_data = ir0[2];
                ir0[2] = (v1158_data + (v1155_data * v1156_data));
                int32_t v1165_a = v8_lead + 63;
                float v1172_data = glb_m1[(v8_lead + 63)];
                float v1173_data = s0[34];
                float v1175_data = ir0[3];
                ir0[3] = (v1175_data + (v1172_data * v1173_data));
                int32_t v1182_a = v8_lead + 63;
                float v1189_data = glb_m1[(v8_lead + 63)];
                float v1190_data = s0[43];
                float v1192_data = ir0[4];
                ir0[4] = (v1192_data + (v1189_data * v1190_data));
                int32_t v1199_a = v8_lead + 63;
                float v1206_data = glb_m1[(v8_lead + 63)];
                float v1207_data = s0[52];
                float v1209_data = ir0[5];
                ir0[5] = (v1209_data + (v1206_data * v1207_data));
                int32_t v1216_a = v8_lead + 63;
                float v1223_data = glb_m1[(v8_lead + 63)];
                float v1224_data = s0[61];
                float v1226_data = ir0[6];
                ir0[6] = (v1226_data + (v1223_data * v1224_data));
                int32_t v1233_a = v8_lead + 63;
                float v1240_data = glb_m1[(v8_lead + 63)];
                float v1241_data = s0[70];
                float v1243_data = ir0[7];
                ir0[7] = (v1243_data + (v1240_data * v1241_data));
                int32_t v1250_a = v8_lead + 63;
                float v1257_data = glb_m1[(v8_lead + 63)];
                float v1258_data = s0[79];
                float v1260_data = ir0[8];
                ir0[8] = (v1260_data + (v1257_data * v1258_data));
              }
              if (v8_lead < 9) {
                int32_t v1271_a = v8_lead + 72;
                float v1278_data = glb_m1[(v8_lead + 72)];
                float v1279_data = s0[8];
                float v1281_data = ir0[0];
                ir0[0] = (v1281_data + (v1278_data * v1279_data));
                int32_t v1288_a = v8_lead + 72;
                float v1295_data = glb_m1[(v8_lead + 72)];
                float v1296_data = s0[17];
                float v1298_data = ir0[1];
                ir0[1] = (v1298_data + (v1295_data * v1296_data));
                int32_t v1305_a = v8_lead + 72;
                float v1312_data = glb_m1[(v8_lead + 72)];
                float v1313_data = s0[26];
                float v1315_data = ir0[2];
                ir0[2] = (v1315_data + (v1312_data * v1313_data));
                int32_t v1322_a = v8_lead + 72;
                float v1329_data = glb_m1[(v8_lead + 72)];
                float v1330_data = s0[35];
                float v1332_data = ir0[3];
                ir0[3] = (v1332_data + (v1329_data * v1330_data));
                int32_t v1339_a = v8_lead + 72;
                float v1346_data = glb_m1[(v8_lead + 72)];
                float v1347_data = s0[44];
                float v1349_data = ir0[4];
                ir0[4] = (v1349_data + (v1346_data * v1347_data));
                int32_t v1356_a = v8_lead + 72;
                float v1363_data = glb_m1[(v8_lead + 72)];
                float v1364_data = s0[53];
                float v1366_data = ir0[5];
                ir0[5] = (v1366_data + (v1363_data * v1364_data));
                int32_t v1373_a = v8_lead + 72;
                float v1380_data = glb_m1[(v8_lead + 72)];
                float v1381_data = s0[62];
                float v1383_data = ir0[6];
                ir0[6] = (v1383_data + (v1380_data * v1381_data));
                int32_t v1390_a = v8_lead + 72;
                float v1397_data = glb_m1[(v8_lead + 72)];
                float v1398_data = s0[71];
                float v1400_data = ir0[7];
                ir0[7] = (v1400_data + (v1397_data * v1398_data));
                int32_t v1407_a = v8_lead + 72;
                float v1414_data = glb_m1[(v8_lead + 72)];
                float v1415_data = s0[80];
                float v1417_data = ir0[8];
                ir0[8] = (v1417_data + (v1414_data * v1415_data));
              }
              if (v8_lead < 9) {
                #pragma unroll
                for (int32_t v1424_n1 = 0; v1424_n1 < 9; ++v1424_n1) {
                  int32_t v1425_a = 0 + v1424_n1;
                  float v1427_data = ir0[v1424_n1];
                  int32_t v1429_a = 0 + v1424_n1;
                  r0[v1424_n1] = (v1427_data * 13.0f);
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 9) {
                #pragma unroll
                for (int32_t v1435_i1 = 0; v1435_i1 < 9; ++v1435_i1) {
                  int32_t v1436_a = 0 + v1435_i1;
                  float v1438_data = r0[v1435_i1];
                  int32_t v1445_a = v8_lead + (v1435_i1 * 9);
                  glb_m0[v1445_a] = v1438_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


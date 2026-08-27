// === base name ===
kernel_5e7da3148f

// === header ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5e7da3148f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5e7da3148f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×16(12×16) {0..12}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              item.barrier();
              // r0 = +(glb_m1 * s0) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[8]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                int32_t v15_a = v8_lead + 0;
                float v22_data = glb_m1[v8_lead];
                float v23_data = s0[0];
                float v25_data = ir0[0];
                ir0[0] = (v25_data + (v22_data * v23_data));
                int32_t v32_a = v8_lead + 0;
                float v39_data = glb_m1[v8_lead];
                float v40_data = s0[16];
                float v42_data = ir0[1];
                ir0[1] = (v42_data + (v39_data * v40_data));
                int32_t v49_a = v8_lead + 0;
                float v56_data = glb_m1[v8_lead];
                float v57_data = s0[32];
                float v59_data = ir0[2];
                ir0[2] = (v59_data + (v56_data * v57_data));
                int32_t v66_a = v8_lead + 0;
                float v73_data = glb_m1[v8_lead];
                float v74_data = s0[48];
                float v76_data = ir0[3];
                ir0[3] = (v76_data + (v73_data * v74_data));
                int32_t v83_a = v8_lead + 0;
                float v90_data = glb_m1[v8_lead];
                float v91_data = s0[64];
                float v93_data = ir0[4];
                ir0[4] = (v93_data + (v90_data * v91_data));
                int32_t v100_a = v8_lead + 0;
                float v107_data = glb_m1[v8_lead];
                float v108_data = s0[80];
                float v110_data = ir0[5];
                ir0[5] = (v110_data + (v107_data * v108_data));
                int32_t v117_a = v8_lead + 0;
                float v124_data = glb_m1[v8_lead];
                float v125_data = s0[96];
                float v127_data = ir0[6];
                ir0[6] = (v127_data + (v124_data * v125_data));
                int32_t v134_a = v8_lead + 0;
                float v141_data = glb_m1[v8_lead];
                float v142_data = s0[112];
                float v144_data = ir0[7];
                ir0[7] = (v144_data + (v141_data * v142_data));
              }
              if (v8_lead < 12) {
                int32_t v155_a = v8_lead + 12;
                float v162_data = glb_m1[(v8_lead + 12)];
                float v163_data = s0[1];
                float v165_data = ir0[0];
                ir0[0] = (v165_data + (v162_data * v163_data));
                int32_t v172_a = v8_lead + 12;
                float v179_data = glb_m1[(v8_lead + 12)];
                float v180_data = s0[17];
                float v182_data = ir0[1];
                ir0[1] = (v182_data + (v179_data * v180_data));
                int32_t v189_a = v8_lead + 12;
                float v196_data = glb_m1[(v8_lead + 12)];
                float v197_data = s0[33];
                float v199_data = ir0[2];
                ir0[2] = (v199_data + (v196_data * v197_data));
                int32_t v206_a = v8_lead + 12;
                float v213_data = glb_m1[(v8_lead + 12)];
                float v214_data = s0[49];
                float v216_data = ir0[3];
                ir0[3] = (v216_data + (v213_data * v214_data));
                int32_t v223_a = v8_lead + 12;
                float v230_data = glb_m1[(v8_lead + 12)];
                float v231_data = s0[65];
                float v233_data = ir0[4];
                ir0[4] = (v233_data + (v230_data * v231_data));
                int32_t v240_a = v8_lead + 12;
                float v247_data = glb_m1[(v8_lead + 12)];
                float v248_data = s0[81];
                float v250_data = ir0[5];
                ir0[5] = (v250_data + (v247_data * v248_data));
                int32_t v257_a = v8_lead + 12;
                float v264_data = glb_m1[(v8_lead + 12)];
                float v265_data = s0[97];
                float v267_data = ir0[6];
                ir0[6] = (v267_data + (v264_data * v265_data));
                int32_t v274_a = v8_lead + 12;
                float v281_data = glb_m1[(v8_lead + 12)];
                float v282_data = s0[113];
                float v284_data = ir0[7];
                ir0[7] = (v284_data + (v281_data * v282_data));
              }
              if (v8_lead < 12) {
                int32_t v295_a = v8_lead + 24;
                float v302_data = glb_m1[(v8_lead + 24)];
                float v303_data = s0[2];
                float v305_data = ir0[0];
                ir0[0] = (v305_data + (v302_data * v303_data));
                int32_t v312_a = v8_lead + 24;
                float v319_data = glb_m1[(v8_lead + 24)];
                float v320_data = s0[18];
                float v322_data = ir0[1];
                ir0[1] = (v322_data + (v319_data * v320_data));
                int32_t v329_a = v8_lead + 24;
                float v336_data = glb_m1[(v8_lead + 24)];
                float v337_data = s0[34];
                float v339_data = ir0[2];
                ir0[2] = (v339_data + (v336_data * v337_data));
                int32_t v346_a = v8_lead + 24;
                float v353_data = glb_m1[(v8_lead + 24)];
                float v354_data = s0[50];
                float v356_data = ir0[3];
                ir0[3] = (v356_data + (v353_data * v354_data));
                int32_t v363_a = v8_lead + 24;
                float v370_data = glb_m1[(v8_lead + 24)];
                float v371_data = s0[66];
                float v373_data = ir0[4];
                ir0[4] = (v373_data + (v370_data * v371_data));
                int32_t v380_a = v8_lead + 24;
                float v387_data = glb_m1[(v8_lead + 24)];
                float v388_data = s0[82];
                float v390_data = ir0[5];
                ir0[5] = (v390_data + (v387_data * v388_data));
                int32_t v397_a = v8_lead + 24;
                float v404_data = glb_m1[(v8_lead + 24)];
                float v405_data = s0[98];
                float v407_data = ir0[6];
                ir0[6] = (v407_data + (v404_data * v405_data));
                int32_t v414_a = v8_lead + 24;
                float v421_data = glb_m1[(v8_lead + 24)];
                float v422_data = s0[114];
                float v424_data = ir0[7];
                ir0[7] = (v424_data + (v421_data * v422_data));
              }
              if (v8_lead < 12) {
                int32_t v435_a = v8_lead + 36;
                float v442_data = glb_m1[(v8_lead + 36)];
                float v443_data = s0[3];
                float v445_data = ir0[0];
                ir0[0] = (v445_data + (v442_data * v443_data));
                int32_t v452_a = v8_lead + 36;
                float v459_data = glb_m1[(v8_lead + 36)];
                float v460_data = s0[19];
                float v462_data = ir0[1];
                ir0[1] = (v462_data + (v459_data * v460_data));
                int32_t v469_a = v8_lead + 36;
                float v476_data = glb_m1[(v8_lead + 36)];
                float v477_data = s0[35];
                float v479_data = ir0[2];
                ir0[2] = (v479_data + (v476_data * v477_data));
                int32_t v486_a = v8_lead + 36;
                float v493_data = glb_m1[(v8_lead + 36)];
                float v494_data = s0[51];
                float v496_data = ir0[3];
                ir0[3] = (v496_data + (v493_data * v494_data));
                int32_t v503_a = v8_lead + 36;
                float v510_data = glb_m1[(v8_lead + 36)];
                float v511_data = s0[67];
                float v513_data = ir0[4];
                ir0[4] = (v513_data + (v510_data * v511_data));
                int32_t v520_a = v8_lead + 36;
                float v527_data = glb_m1[(v8_lead + 36)];
                float v528_data = s0[83];
                float v530_data = ir0[5];
                ir0[5] = (v530_data + (v527_data * v528_data));
                int32_t v537_a = v8_lead + 36;
                float v544_data = glb_m1[(v8_lead + 36)];
                float v545_data = s0[99];
                float v547_data = ir0[6];
                ir0[6] = (v547_data + (v544_data * v545_data));
                int32_t v554_a = v8_lead + 36;
                float v561_data = glb_m1[(v8_lead + 36)];
                float v562_data = s0[115];
                float v564_data = ir0[7];
                ir0[7] = (v564_data + (v561_data * v562_data));
              }
              if (v8_lead < 12) {
                int32_t v575_a = v8_lead + 48;
                float v582_data = glb_m1[(v8_lead + 48)];
                float v583_data = s0[4];
                float v585_data = ir0[0];
                ir0[0] = (v585_data + (v582_data * v583_data));
                int32_t v592_a = v8_lead + 48;
                float v599_data = glb_m1[(v8_lead + 48)];
                float v600_data = s0[20];
                float v602_data = ir0[1];
                ir0[1] = (v602_data + (v599_data * v600_data));
                int32_t v609_a = v8_lead + 48;
                float v616_data = glb_m1[(v8_lead + 48)];
                float v617_data = s0[36];
                float v619_data = ir0[2];
                ir0[2] = (v619_data + (v616_data * v617_data));
                int32_t v626_a = v8_lead + 48;
                float v633_data = glb_m1[(v8_lead + 48)];
                float v634_data = s0[52];
                float v636_data = ir0[3];
                ir0[3] = (v636_data + (v633_data * v634_data));
                int32_t v643_a = v8_lead + 48;
                float v650_data = glb_m1[(v8_lead + 48)];
                float v651_data = s0[68];
                float v653_data = ir0[4];
                ir0[4] = (v653_data + (v650_data * v651_data));
                int32_t v660_a = v8_lead + 48;
                float v667_data = glb_m1[(v8_lead + 48)];
                float v668_data = s0[84];
                float v670_data = ir0[5];
                ir0[5] = (v670_data + (v667_data * v668_data));
                int32_t v677_a = v8_lead + 48;
                float v684_data = glb_m1[(v8_lead + 48)];
                float v685_data = s0[100];
                float v687_data = ir0[6];
                ir0[6] = (v687_data + (v684_data * v685_data));
                int32_t v694_a = v8_lead + 48;
                float v701_data = glb_m1[(v8_lead + 48)];
                float v702_data = s0[116];
                float v704_data = ir0[7];
                ir0[7] = (v704_data + (v701_data * v702_data));
              }
              if (v8_lead < 12) {
                int32_t v715_a = v8_lead + 60;
                float v722_data = glb_m1[(v8_lead + 60)];
                float v723_data = s0[5];
                float v725_data = ir0[0];
                ir0[0] = (v725_data + (v722_data * v723_data));
                int32_t v732_a = v8_lead + 60;
                float v739_data = glb_m1[(v8_lead + 60)];
                float v740_data = s0[21];
                float v742_data = ir0[1];
                ir0[1] = (v742_data + (v739_data * v740_data));
                int32_t v749_a = v8_lead + 60;
                float v756_data = glb_m1[(v8_lead + 60)];
                float v757_data = s0[37];
                float v759_data = ir0[2];
                ir0[2] = (v759_data + (v756_data * v757_data));
                int32_t v766_a = v8_lead + 60;
                float v773_data = glb_m1[(v8_lead + 60)];
                float v774_data = s0[53];
                float v776_data = ir0[3];
                ir0[3] = (v776_data + (v773_data * v774_data));
                int32_t v783_a = v8_lead + 60;
                float v790_data = glb_m1[(v8_lead + 60)];
                float v791_data = s0[69];
                float v793_data = ir0[4];
                ir0[4] = (v793_data + (v790_data * v791_data));
                int32_t v800_a = v8_lead + 60;
                float v807_data = glb_m1[(v8_lead + 60)];
                float v808_data = s0[85];
                float v810_data = ir0[5];
                ir0[5] = (v810_data + (v807_data * v808_data));
                int32_t v817_a = v8_lead + 60;
                float v824_data = glb_m1[(v8_lead + 60)];
                float v825_data = s0[101];
                float v827_data = ir0[6];
                ir0[6] = (v827_data + (v824_data * v825_data));
                int32_t v834_a = v8_lead + 60;
                float v841_data = glb_m1[(v8_lead + 60)];
                float v842_data = s0[117];
                float v844_data = ir0[7];
                ir0[7] = (v844_data + (v841_data * v842_data));
              }
              if (v8_lead < 12) {
                int32_t v855_a = v8_lead + 72;
                float v862_data = glb_m1[(v8_lead + 72)];
                float v863_data = s0[6];
                float v865_data = ir0[0];
                ir0[0] = (v865_data + (v862_data * v863_data));
                int32_t v872_a = v8_lead + 72;
                float v879_data = glb_m1[(v8_lead + 72)];
                float v880_data = s0[22];
                float v882_data = ir0[1];
                ir0[1] = (v882_data + (v879_data * v880_data));
                int32_t v889_a = v8_lead + 72;
                float v896_data = glb_m1[(v8_lead + 72)];
                float v897_data = s0[38];
                float v899_data = ir0[2];
                ir0[2] = (v899_data + (v896_data * v897_data));
                int32_t v906_a = v8_lead + 72;
                float v913_data = glb_m1[(v8_lead + 72)];
                float v914_data = s0[54];
                float v916_data = ir0[3];
                ir0[3] = (v916_data + (v913_data * v914_data));
                int32_t v923_a = v8_lead + 72;
                float v930_data = glb_m1[(v8_lead + 72)];
                float v931_data = s0[70];
                float v933_data = ir0[4];
                ir0[4] = (v933_data + (v930_data * v931_data));
                int32_t v940_a = v8_lead + 72;
                float v947_data = glb_m1[(v8_lead + 72)];
                float v948_data = s0[86];
                float v950_data = ir0[5];
                ir0[5] = (v950_data + (v947_data * v948_data));
                int32_t v957_a = v8_lead + 72;
                float v964_data = glb_m1[(v8_lead + 72)];
                float v965_data = s0[102];
                float v967_data = ir0[6];
                ir0[6] = (v967_data + (v964_data * v965_data));
                int32_t v974_a = v8_lead + 72;
                float v981_data = glb_m1[(v8_lead + 72)];
                float v982_data = s0[118];
                float v984_data = ir0[7];
                ir0[7] = (v984_data + (v981_data * v982_data));
              }
              if (v8_lead < 12) {
                int32_t v995_a = v8_lead + 84;
                float v1002_data = glb_m1[(v8_lead + 84)];
                float v1003_data = s0[7];
                float v1005_data = ir0[0];
                ir0[0] = (v1005_data + (v1002_data * v1003_data));
                int32_t v1012_a = v8_lead + 84;
                float v1019_data = glb_m1[(v8_lead + 84)];
                float v1020_data = s0[23];
                float v1022_data = ir0[1];
                ir0[1] = (v1022_data + (v1019_data * v1020_data));
                int32_t v1029_a = v8_lead + 84;
                float v1036_data = glb_m1[(v8_lead + 84)];
                float v1037_data = s0[39];
                float v1039_data = ir0[2];
                ir0[2] = (v1039_data + (v1036_data * v1037_data));
                int32_t v1046_a = v8_lead + 84;
                float v1053_data = glb_m1[(v8_lead + 84)];
                float v1054_data = s0[55];
                float v1056_data = ir0[3];
                ir0[3] = (v1056_data + (v1053_data * v1054_data));
                int32_t v1063_a = v8_lead + 84;
                float v1070_data = glb_m1[(v8_lead + 84)];
                float v1071_data = s0[71];
                float v1073_data = ir0[4];
                ir0[4] = (v1073_data + (v1070_data * v1071_data));
                int32_t v1080_a = v8_lead + 84;
                float v1087_data = glb_m1[(v8_lead + 84)];
                float v1088_data = s0[87];
                float v1090_data = ir0[5];
                ir0[5] = (v1090_data + (v1087_data * v1088_data));
                int32_t v1097_a = v8_lead + 84;
                float v1104_data = glb_m1[(v8_lead + 84)];
                float v1105_data = s0[103];
                float v1107_data = ir0[6];
                ir0[6] = (v1107_data + (v1104_data * v1105_data));
                int32_t v1114_a = v8_lead + 84;
                float v1121_data = glb_m1[(v8_lead + 84)];
                float v1122_data = s0[119];
                float v1124_data = ir0[7];
                ir0[7] = (v1124_data + (v1121_data * v1122_data));
              }
              if (v8_lead < 12) {
                int32_t v1135_a = v8_lead + 96;
                float v1142_data = glb_m1[(v8_lead + 96)];
                float v1143_data = s0[8];
                float v1145_data = ir0[0];
                ir0[0] = (v1145_data + (v1142_data * v1143_data));
                int32_t v1152_a = v8_lead + 96;
                float v1159_data = glb_m1[(v8_lead + 96)];
                float v1160_data = s0[24];
                float v1162_data = ir0[1];
                ir0[1] = (v1162_data + (v1159_data * v1160_data));
                int32_t v1169_a = v8_lead + 96;
                float v1176_data = glb_m1[(v8_lead + 96)];
                float v1177_data = s0[40];
                float v1179_data = ir0[2];
                ir0[2] = (v1179_data + (v1176_data * v1177_data));
                int32_t v1186_a = v8_lead + 96;
                float v1193_data = glb_m1[(v8_lead + 96)];
                float v1194_data = s0[56];
                float v1196_data = ir0[3];
                ir0[3] = (v1196_data + (v1193_data * v1194_data));
                int32_t v1203_a = v8_lead + 96;
                float v1210_data = glb_m1[(v8_lead + 96)];
                float v1211_data = s0[72];
                float v1213_data = ir0[4];
                ir0[4] = (v1213_data + (v1210_data * v1211_data));
                int32_t v1220_a = v8_lead + 96;
                float v1227_data = glb_m1[(v8_lead + 96)];
                float v1228_data = s0[88];
                float v1230_data = ir0[5];
                ir0[5] = (v1230_data + (v1227_data * v1228_data));
                int32_t v1237_a = v8_lead + 96;
                float v1244_data = glb_m1[(v8_lead + 96)];
                float v1245_data = s0[104];
                float v1247_data = ir0[6];
                ir0[6] = (v1247_data + (v1244_data * v1245_data));
                int32_t v1254_a = v8_lead + 96;
                float v1261_data = glb_m1[(v8_lead + 96)];
                float v1262_data = s0[120];
                float v1264_data = ir0[7];
                ir0[7] = (v1264_data + (v1261_data * v1262_data));
              }
              if (v8_lead < 12) {
                int32_t v1275_a = v8_lead + 108;
                float v1282_data = glb_m1[(v8_lead + 108)];
                float v1283_data = s0[9];
                float v1285_data = ir0[0];
                ir0[0] = (v1285_data + (v1282_data * v1283_data));
                int32_t v1292_a = v8_lead + 108;
                float v1299_data = glb_m1[(v8_lead + 108)];
                float v1300_data = s0[25];
                float v1302_data = ir0[1];
                ir0[1] = (v1302_data + (v1299_data * v1300_data));
                int32_t v1309_a = v8_lead + 108;
                float v1316_data = glb_m1[(v8_lead + 108)];
                float v1317_data = s0[41];
                float v1319_data = ir0[2];
                ir0[2] = (v1319_data + (v1316_data * v1317_data));
                int32_t v1326_a = v8_lead + 108;
                float v1333_data = glb_m1[(v8_lead + 108)];
                float v1334_data = s0[57];
                float v1336_data = ir0[3];
                ir0[3] = (v1336_data + (v1333_data * v1334_data));
                int32_t v1343_a = v8_lead + 108;
                float v1350_data = glb_m1[(v8_lead + 108)];
                float v1351_data = s0[73];
                float v1353_data = ir0[4];
                ir0[4] = (v1353_data + (v1350_data * v1351_data));
                int32_t v1360_a = v8_lead + 108;
                float v1367_data = glb_m1[(v8_lead + 108)];
                float v1368_data = s0[89];
                float v1370_data = ir0[5];
                ir0[5] = (v1370_data + (v1367_data * v1368_data));
                int32_t v1377_a = v8_lead + 108;
                float v1384_data = glb_m1[(v8_lead + 108)];
                float v1385_data = s0[105];
                float v1387_data = ir0[6];
                ir0[6] = (v1387_data + (v1384_data * v1385_data));
                int32_t v1394_a = v8_lead + 108;
                float v1401_data = glb_m1[(v8_lead + 108)];
                float v1402_data = s0[121];
                float v1404_data = ir0[7];
                ir0[7] = (v1404_data + (v1401_data * v1402_data));
              }
              if (v8_lead < 12) {
                int32_t v1415_a = v8_lead + 120;
                float v1422_data = glb_m1[(v8_lead + 120)];
                float v1423_data = s0[10];
                float v1425_data = ir0[0];
                ir0[0] = (v1425_data + (v1422_data * v1423_data));
                int32_t v1432_a = v8_lead + 120;
                float v1439_data = glb_m1[(v8_lead + 120)];
                float v1440_data = s0[26];
                float v1442_data = ir0[1];
                ir0[1] = (v1442_data + (v1439_data * v1440_data));
                int32_t v1449_a = v8_lead + 120;
                float v1456_data = glb_m1[(v8_lead + 120)];
                float v1457_data = s0[42];
                float v1459_data = ir0[2];
                ir0[2] = (v1459_data + (v1456_data * v1457_data));
                int32_t v1466_a = v8_lead + 120;
                float v1473_data = glb_m1[(v8_lead + 120)];
                float v1474_data = s0[58];
                float v1476_data = ir0[3];
                ir0[3] = (v1476_data + (v1473_data * v1474_data));
                int32_t v1483_a = v8_lead + 120;
                float v1490_data = glb_m1[(v8_lead + 120)];
                float v1491_data = s0[74];
                float v1493_data = ir0[4];
                ir0[4] = (v1493_data + (v1490_data * v1491_data));
                int32_t v1500_a = v8_lead + 120;
                float v1507_data = glb_m1[(v8_lead + 120)];
                float v1508_data = s0[90];
                float v1510_data = ir0[5];
                ir0[5] = (v1510_data + (v1507_data * v1508_data));
                int32_t v1517_a = v8_lead + 120;
                float v1524_data = glb_m1[(v8_lead + 120)];
                float v1525_data = s0[106];
                float v1527_data = ir0[6];
                ir0[6] = (v1527_data + (v1524_data * v1525_data));
                int32_t v1534_a = v8_lead + 120;
                float v1541_data = glb_m1[(v8_lead + 120)];
                float v1542_data = s0[122];
                float v1544_data = ir0[7];
                ir0[7] = (v1544_data + (v1541_data * v1542_data));
              }
              if (v8_lead < 12) {
                int32_t v1555_a = v8_lead + 132;
                float v1562_data = glb_m1[(v8_lead + 132)];
                float v1563_data = s0[11];
                float v1565_data = ir0[0];
                ir0[0] = (v1565_data + (v1562_data * v1563_data));
                int32_t v1572_a = v8_lead + 132;
                float v1579_data = glb_m1[(v8_lead + 132)];
                float v1580_data = s0[27];
                float v1582_data = ir0[1];
                ir0[1] = (v1582_data + (v1579_data * v1580_data));
                int32_t v1589_a = v8_lead + 132;
                float v1596_data = glb_m1[(v8_lead + 132)];
                float v1597_data = s0[43];
                float v1599_data = ir0[2];
                ir0[2] = (v1599_data + (v1596_data * v1597_data));
                int32_t v1606_a = v8_lead + 132;
                float v1613_data = glb_m1[(v8_lead + 132)];
                float v1614_data = s0[59];
                float v1616_data = ir0[3];
                ir0[3] = (v1616_data + (v1613_data * v1614_data));
                int32_t v1623_a = v8_lead + 132;
                float v1630_data = glb_m1[(v8_lead + 132)];
                float v1631_data = s0[75];
                float v1633_data = ir0[4];
                ir0[4] = (v1633_data + (v1630_data * v1631_data));
                int32_t v1640_a = v8_lead + 132;
                float v1647_data = glb_m1[(v8_lead + 132)];
                float v1648_data = s0[91];
                float v1650_data = ir0[5];
                ir0[5] = (v1650_data + (v1647_data * v1648_data));
                int32_t v1657_a = v8_lead + 132;
                float v1664_data = glb_m1[(v8_lead + 132)];
                float v1665_data = s0[107];
                float v1667_data = ir0[6];
                ir0[6] = (v1667_data + (v1664_data * v1665_data));
                int32_t v1674_a = v8_lead + 132;
                float v1681_data = glb_m1[(v8_lead + 132)];
                float v1682_data = s0[123];
                float v1684_data = ir0[7];
                ir0[7] = (v1684_data + (v1681_data * v1682_data));
              }
              if (v8_lead < 12) {
                int32_t v1695_a = v8_lead + 144;
                float v1702_data = glb_m1[(v8_lead + 144)];
                float v1703_data = s0[12];
                float v1705_data = ir0[0];
                ir0[0] = (v1705_data + (v1702_data * v1703_data));
                int32_t v1712_a = v8_lead + 144;
                float v1719_data = glb_m1[(v8_lead + 144)];
                float v1720_data = s0[28];
                float v1722_data = ir0[1];
                ir0[1] = (v1722_data + (v1719_data * v1720_data));
                int32_t v1729_a = v8_lead + 144;
                float v1736_data = glb_m1[(v8_lead + 144)];
                float v1737_data = s0[44];
                float v1739_data = ir0[2];
                ir0[2] = (v1739_data + (v1736_data * v1737_data));
                int32_t v1746_a = v8_lead + 144;
                float v1753_data = glb_m1[(v8_lead + 144)];
                float v1754_data = s0[60];
                float v1756_data = ir0[3];
                ir0[3] = (v1756_data + (v1753_data * v1754_data));
                int32_t v1763_a = v8_lead + 144;
                float v1770_data = glb_m1[(v8_lead + 144)];
                float v1771_data = s0[76];
                float v1773_data = ir0[4];
                ir0[4] = (v1773_data + (v1770_data * v1771_data));
                int32_t v1780_a = v8_lead + 144;
                float v1787_data = glb_m1[(v8_lead + 144)];
                float v1788_data = s0[92];
                float v1790_data = ir0[5];
                ir0[5] = (v1790_data + (v1787_data * v1788_data));
                int32_t v1797_a = v8_lead + 144;
                float v1804_data = glb_m1[(v8_lead + 144)];
                float v1805_data = s0[108];
                float v1807_data = ir0[6];
                ir0[6] = (v1807_data + (v1804_data * v1805_data));
                int32_t v1814_a = v8_lead + 144;
                float v1821_data = glb_m1[(v8_lead + 144)];
                float v1822_data = s0[124];
                float v1824_data = ir0[7];
                ir0[7] = (v1824_data + (v1821_data * v1822_data));
              }
              if (v8_lead < 12) {
                int32_t v1835_a = v8_lead + 156;
                float v1842_data = glb_m1[(v8_lead + 156)];
                float v1843_data = s0[13];
                float v1845_data = ir0[0];
                ir0[0] = (v1845_data + (v1842_data * v1843_data));
                int32_t v1852_a = v8_lead + 156;
                float v1859_data = glb_m1[(v8_lead + 156)];
                float v1860_data = s0[29];
                float v1862_data = ir0[1];
                ir0[1] = (v1862_data + (v1859_data * v1860_data));
                int32_t v1869_a = v8_lead + 156;
                float v1876_data = glb_m1[(v8_lead + 156)];
                float v1877_data = s0[45];
                float v1879_data = ir0[2];
                ir0[2] = (v1879_data + (v1876_data * v1877_data));
                int32_t v1886_a = v8_lead + 156;
                float v1893_data = glb_m1[(v8_lead + 156)];
                float v1894_data = s0[61];
                float v1896_data = ir0[3];
                ir0[3] = (v1896_data + (v1893_data * v1894_data));
                int32_t v1903_a = v8_lead + 156;
                float v1910_data = glb_m1[(v8_lead + 156)];
                float v1911_data = s0[77];
                float v1913_data = ir0[4];
                ir0[4] = (v1913_data + (v1910_data * v1911_data));
                int32_t v1920_a = v8_lead + 156;
                float v1927_data = glb_m1[(v8_lead + 156)];
                float v1928_data = s0[93];
                float v1930_data = ir0[5];
                ir0[5] = (v1930_data + (v1927_data * v1928_data));
                int32_t v1937_a = v8_lead + 156;
                float v1944_data = glb_m1[(v8_lead + 156)];
                float v1945_data = s0[109];
                float v1947_data = ir0[6];
                ir0[6] = (v1947_data + (v1944_data * v1945_data));
                int32_t v1954_a = v8_lead + 156;
                float v1961_data = glb_m1[(v8_lead + 156)];
                float v1962_data = s0[125];
                float v1964_data = ir0[7];
                ir0[7] = (v1964_data + (v1961_data * v1962_data));
              }
              if (v8_lead < 12) {
                int32_t v1975_a = v8_lead + 168;
                float v1982_data = glb_m1[(v8_lead + 168)];
                float v1983_data = s0[14];
                float v1985_data = ir0[0];
                ir0[0] = (v1985_data + (v1982_data * v1983_data));
                int32_t v1992_a = v8_lead + 168;
                float v1999_data = glb_m1[(v8_lead + 168)];
                float v2000_data = s0[30];
                float v2002_data = ir0[1];
                ir0[1] = (v2002_data + (v1999_data * v2000_data));
                int32_t v2009_a = v8_lead + 168;
                float v2016_data = glb_m1[(v8_lead + 168)];
                float v2017_data = s0[46];
                float v2019_data = ir0[2];
                ir0[2] = (v2019_data + (v2016_data * v2017_data));
                int32_t v2026_a = v8_lead + 168;
                float v2033_data = glb_m1[(v8_lead + 168)];
                float v2034_data = s0[62];
                float v2036_data = ir0[3];
                ir0[3] = (v2036_data + (v2033_data * v2034_data));
                int32_t v2043_a = v8_lead + 168;
                float v2050_data = glb_m1[(v8_lead + 168)];
                float v2051_data = s0[78];
                float v2053_data = ir0[4];
                ir0[4] = (v2053_data + (v2050_data * v2051_data));
                int32_t v2060_a = v8_lead + 168;
                float v2067_data = glb_m1[(v8_lead + 168)];
                float v2068_data = s0[94];
                float v2070_data = ir0[5];
                ir0[5] = (v2070_data + (v2067_data * v2068_data));
                int32_t v2077_a = v8_lead + 168;
                float v2084_data = glb_m1[(v8_lead + 168)];
                float v2085_data = s0[110];
                float v2087_data = ir0[6];
                ir0[6] = (v2087_data + (v2084_data * v2085_data));
                int32_t v2094_a = v8_lead + 168;
                float v2101_data = glb_m1[(v8_lead + 168)];
                float v2102_data = s0[126];
                float v2104_data = ir0[7];
                ir0[7] = (v2104_data + (v2101_data * v2102_data));
              }
              if (v8_lead < 12) {
                int32_t v2115_a = v8_lead + 180;
                float v2122_data = glb_m1[(v8_lead + 180)];
                float v2123_data = s0[15];
                float v2125_data = ir0[0];
                ir0[0] = (v2125_data + (v2122_data * v2123_data));
                int32_t v2132_a = v8_lead + 180;
                float v2139_data = glb_m1[(v8_lead + 180)];
                float v2140_data = s0[31];
                float v2142_data = ir0[1];
                ir0[1] = (v2142_data + (v2139_data * v2140_data));
                int32_t v2149_a = v8_lead + 180;
                float v2156_data = glb_m1[(v8_lead + 180)];
                float v2157_data = s0[47];
                float v2159_data = ir0[2];
                ir0[2] = (v2159_data + (v2156_data * v2157_data));
                int32_t v2166_a = v8_lead + 180;
                float v2173_data = glb_m1[(v8_lead + 180)];
                float v2174_data = s0[63];
                float v2176_data = ir0[3];
                ir0[3] = (v2176_data + (v2173_data * v2174_data));
                int32_t v2183_a = v8_lead + 180;
                float v2190_data = glb_m1[(v8_lead + 180)];
                float v2191_data = s0[79];
                float v2193_data = ir0[4];
                ir0[4] = (v2193_data + (v2190_data * v2191_data));
                int32_t v2200_a = v8_lead + 180;
                float v2207_data = glb_m1[(v8_lead + 180)];
                float v2208_data = s0[95];
                float v2210_data = ir0[5];
                ir0[5] = (v2210_data + (v2207_data * v2208_data));
                int32_t v2217_a = v8_lead + 180;
                float v2224_data = glb_m1[(v8_lead + 180)];
                float v2225_data = s0[111];
                float v2227_data = ir0[6];
                ir0[6] = (v2227_data + (v2224_data * v2225_data));
                int32_t v2234_a = v8_lead + 180;
                float v2241_data = glb_m1[(v8_lead + 180)];
                float v2242_data = s0[127];
                float v2244_data = ir0[7];
                ir0[7] = (v2244_data + (v2241_data * v2242_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v2250_n1 = 0; v2250_n1 < 8; ++v2250_n1) {
                  int32_t v2251_a = 0 + v2250_n1;
                  float v2253_data = ir0[v2250_n1];
                  int32_t v2259_a = v2250_n1 * 12;
                  int32_t v2260_a = v8_lead + v2259_a;
                  float v2268_data = glb_m0[(v8_lead + v2259_a)];
                  int32_t v2270_a = 0 + v2250_n1;
                  r0[v2250_n1] = (v2268_data + v2253_data);
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v2276_i1 = 0; v2276_i1 < 8; ++v2276_i1) {
                  int32_t v2277_a = 0 + v2276_i1;
                  float v2279_data = r0[v2276_i1];
                  int32_t v2286_a = v8_lead + (v2276_i1 * 12);
                  glb_m0[v2286_a] = v2279_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


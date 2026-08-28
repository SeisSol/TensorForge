// === base name ===
kernel_f61651fe59

// === header ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f61651fe59(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f61651fe59(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(12×16) {4..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(12×16) {4..16}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              alignas(16) float r0[8]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[8]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                int32_t v17_a = ((v8_lead + 4) - 4) + 0;
                float v26_data = glb_m1[((v8_lead + 4) - 4)];
                float v27_data = s0[0];
                float v29_data = ir0[0];
                ir0[0] = (v29_data + (v26_data * v27_data));
                int32_t v38_a = ((v8_lead + 4) - 4) + 0;
                float v47_data = glb_m1[((v8_lead + 4) - 4)];
                float v48_data = s0[16];
                float v50_data = ir0[1];
                ir0[1] = (v50_data + (v47_data * v48_data));
                int32_t v59_a = ((v8_lead + 4) - 4) + 0;
                float v68_data = glb_m1[((v8_lead + 4) - 4)];
                float v69_data = s0[32];
                float v71_data = ir0[2];
                ir0[2] = (v71_data + (v68_data * v69_data));
                int32_t v80_a = ((v8_lead + 4) - 4) + 0;
                float v89_data = glb_m1[((v8_lead + 4) - 4)];
                float v90_data = s0[48];
                float v92_data = ir0[3];
                ir0[3] = (v92_data + (v89_data * v90_data));
                int32_t v101_a = ((v8_lead + 4) - 4) + 0;
                float v110_data = glb_m1[((v8_lead + 4) - 4)];
                float v111_data = s0[64];
                float v113_data = ir0[4];
                ir0[4] = (v113_data + (v110_data * v111_data));
                int32_t v122_a = ((v8_lead + 4) - 4) + 0;
                float v131_data = glb_m1[((v8_lead + 4) - 4)];
                float v132_data = s0[80];
                float v134_data = ir0[5];
                ir0[5] = (v134_data + (v131_data * v132_data));
                int32_t v143_a = ((v8_lead + 4) - 4) + 0;
                float v152_data = glb_m1[((v8_lead + 4) - 4)];
                float v153_data = s0[96];
                float v155_data = ir0[6];
                ir0[6] = (v155_data + (v152_data * v153_data));
                int32_t v164_a = ((v8_lead + 4) - 4) + 0;
                float v173_data = glb_m1[((v8_lead + 4) - 4)];
                float v174_data = s0[112];
                float v176_data = ir0[7];
                ir0[7] = (v176_data + (v173_data * v174_data));
              }
              if (v8_lead < 12) {
                int32_t v189_a = ((v8_lead + 4) - 4) + 12;
                float v198_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v199_data = s0[1];
                float v201_data = ir0[0];
                ir0[0] = (v201_data + (v198_data * v199_data));
                int32_t v210_a = ((v8_lead + 4) - 4) + 12;
                float v219_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v220_data = s0[17];
                float v222_data = ir0[1];
                ir0[1] = (v222_data + (v219_data * v220_data));
                int32_t v231_a = ((v8_lead + 4) - 4) + 12;
                float v240_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v241_data = s0[33];
                float v243_data = ir0[2];
                ir0[2] = (v243_data + (v240_data * v241_data));
                int32_t v252_a = ((v8_lead + 4) - 4) + 12;
                float v261_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v262_data = s0[49];
                float v264_data = ir0[3];
                ir0[3] = (v264_data + (v261_data * v262_data));
                int32_t v273_a = ((v8_lead + 4) - 4) + 12;
                float v282_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v283_data = s0[65];
                float v285_data = ir0[4];
                ir0[4] = (v285_data + (v282_data * v283_data));
                int32_t v294_a = ((v8_lead + 4) - 4) + 12;
                float v303_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v304_data = s0[81];
                float v306_data = ir0[5];
                ir0[5] = (v306_data + (v303_data * v304_data));
                int32_t v315_a = ((v8_lead + 4) - 4) + 12;
                float v324_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v325_data = s0[97];
                float v327_data = ir0[6];
                ir0[6] = (v327_data + (v324_data * v325_data));
                int32_t v336_a = ((v8_lead + 4) - 4) + 12;
                float v345_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v346_data = s0[113];
                float v348_data = ir0[7];
                ir0[7] = (v348_data + (v345_data * v346_data));
              }
              if (v8_lead < 12) {
                int32_t v361_a = ((v8_lead + 4) - 4) + 24;
                float v370_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v371_data = s0[2];
                float v373_data = ir0[0];
                ir0[0] = (v373_data + (v370_data * v371_data));
                int32_t v382_a = ((v8_lead + 4) - 4) + 24;
                float v391_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v392_data = s0[18];
                float v394_data = ir0[1];
                ir0[1] = (v394_data + (v391_data * v392_data));
                int32_t v403_a = ((v8_lead + 4) - 4) + 24;
                float v412_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v413_data = s0[34];
                float v415_data = ir0[2];
                ir0[2] = (v415_data + (v412_data * v413_data));
                int32_t v424_a = ((v8_lead + 4) - 4) + 24;
                float v433_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v434_data = s0[50];
                float v436_data = ir0[3];
                ir0[3] = (v436_data + (v433_data * v434_data));
                int32_t v445_a = ((v8_lead + 4) - 4) + 24;
                float v454_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v455_data = s0[66];
                float v457_data = ir0[4];
                ir0[4] = (v457_data + (v454_data * v455_data));
                int32_t v466_a = ((v8_lead + 4) - 4) + 24;
                float v475_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v476_data = s0[82];
                float v478_data = ir0[5];
                ir0[5] = (v478_data + (v475_data * v476_data));
                int32_t v487_a = ((v8_lead + 4) - 4) + 24;
                float v496_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v497_data = s0[98];
                float v499_data = ir0[6];
                ir0[6] = (v499_data + (v496_data * v497_data));
                int32_t v508_a = ((v8_lead + 4) - 4) + 24;
                float v517_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v518_data = s0[114];
                float v520_data = ir0[7];
                ir0[7] = (v520_data + (v517_data * v518_data));
              }
              if (v8_lead < 12) {
                int32_t v533_a = ((v8_lead + 4) - 4) + 36;
                float v542_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v543_data = s0[3];
                float v545_data = ir0[0];
                ir0[0] = (v545_data + (v542_data * v543_data));
                int32_t v554_a = ((v8_lead + 4) - 4) + 36;
                float v563_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v564_data = s0[19];
                float v566_data = ir0[1];
                ir0[1] = (v566_data + (v563_data * v564_data));
                int32_t v575_a = ((v8_lead + 4) - 4) + 36;
                float v584_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v585_data = s0[35];
                float v587_data = ir0[2];
                ir0[2] = (v587_data + (v584_data * v585_data));
                int32_t v596_a = ((v8_lead + 4) - 4) + 36;
                float v605_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v606_data = s0[51];
                float v608_data = ir0[3];
                ir0[3] = (v608_data + (v605_data * v606_data));
                int32_t v617_a = ((v8_lead + 4) - 4) + 36;
                float v626_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v627_data = s0[67];
                float v629_data = ir0[4];
                ir0[4] = (v629_data + (v626_data * v627_data));
                int32_t v638_a = ((v8_lead + 4) - 4) + 36;
                float v647_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v648_data = s0[83];
                float v650_data = ir0[5];
                ir0[5] = (v650_data + (v647_data * v648_data));
                int32_t v659_a = ((v8_lead + 4) - 4) + 36;
                float v668_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v669_data = s0[99];
                float v671_data = ir0[6];
                ir0[6] = (v671_data + (v668_data * v669_data));
                int32_t v680_a = ((v8_lead + 4) - 4) + 36;
                float v689_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v690_data = s0[115];
                float v692_data = ir0[7];
                ir0[7] = (v692_data + (v689_data * v690_data));
              }
              if (v8_lead < 12) {
                int32_t v705_a = ((v8_lead + 4) - 4) + 48;
                float v714_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v715_data = s0[4];
                float v717_data = ir0[0];
                ir0[0] = (v717_data + (v714_data * v715_data));
                int32_t v726_a = ((v8_lead + 4) - 4) + 48;
                float v735_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v736_data = s0[20];
                float v738_data = ir0[1];
                ir0[1] = (v738_data + (v735_data * v736_data));
                int32_t v747_a = ((v8_lead + 4) - 4) + 48;
                float v756_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v757_data = s0[36];
                float v759_data = ir0[2];
                ir0[2] = (v759_data + (v756_data * v757_data));
                int32_t v768_a = ((v8_lead + 4) - 4) + 48;
                float v777_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v778_data = s0[52];
                float v780_data = ir0[3];
                ir0[3] = (v780_data + (v777_data * v778_data));
                int32_t v789_a = ((v8_lead + 4) - 4) + 48;
                float v798_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v799_data = s0[68];
                float v801_data = ir0[4];
                ir0[4] = (v801_data + (v798_data * v799_data));
                int32_t v810_a = ((v8_lead + 4) - 4) + 48;
                float v819_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v820_data = s0[84];
                float v822_data = ir0[5];
                ir0[5] = (v822_data + (v819_data * v820_data));
                int32_t v831_a = ((v8_lead + 4) - 4) + 48;
                float v840_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v841_data = s0[100];
                float v843_data = ir0[6];
                ir0[6] = (v843_data + (v840_data * v841_data));
                int32_t v852_a = ((v8_lead + 4) - 4) + 48;
                float v861_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v862_data = s0[116];
                float v864_data = ir0[7];
                ir0[7] = (v864_data + (v861_data * v862_data));
              }
              if (v8_lead < 12) {
                int32_t v877_a = ((v8_lead + 4) - 4) + 60;
                float v886_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v887_data = s0[5];
                float v889_data = ir0[0];
                ir0[0] = (v889_data + (v886_data * v887_data));
                int32_t v898_a = ((v8_lead + 4) - 4) + 60;
                float v907_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v908_data = s0[21];
                float v910_data = ir0[1];
                ir0[1] = (v910_data + (v907_data * v908_data));
                int32_t v919_a = ((v8_lead + 4) - 4) + 60;
                float v928_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v929_data = s0[37];
                float v931_data = ir0[2];
                ir0[2] = (v931_data + (v928_data * v929_data));
                int32_t v940_a = ((v8_lead + 4) - 4) + 60;
                float v949_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v950_data = s0[53];
                float v952_data = ir0[3];
                ir0[3] = (v952_data + (v949_data * v950_data));
                int32_t v961_a = ((v8_lead + 4) - 4) + 60;
                float v970_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v971_data = s0[69];
                float v973_data = ir0[4];
                ir0[4] = (v973_data + (v970_data * v971_data));
                int32_t v982_a = ((v8_lead + 4) - 4) + 60;
                float v991_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v992_data = s0[85];
                float v994_data = ir0[5];
                ir0[5] = (v994_data + (v991_data * v992_data));
                int32_t v1003_a = ((v8_lead + 4) - 4) + 60;
                float v1012_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v1013_data = s0[101];
                float v1015_data = ir0[6];
                ir0[6] = (v1015_data + (v1012_data * v1013_data));
                int32_t v1024_a = ((v8_lead + 4) - 4) + 60;
                float v1033_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v1034_data = s0[117];
                float v1036_data = ir0[7];
                ir0[7] = (v1036_data + (v1033_data * v1034_data));
              }
              if (v8_lead < 12) {
                int32_t v1049_a = ((v8_lead + 4) - 4) + 72;
                float v1058_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v1059_data = s0[6];
                float v1061_data = ir0[0];
                ir0[0] = (v1061_data + (v1058_data * v1059_data));
                int32_t v1070_a = ((v8_lead + 4) - 4) + 72;
                float v1079_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v1080_data = s0[22];
                float v1082_data = ir0[1];
                ir0[1] = (v1082_data + (v1079_data * v1080_data));
                int32_t v1091_a = ((v8_lead + 4) - 4) + 72;
                float v1100_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v1101_data = s0[38];
                float v1103_data = ir0[2];
                ir0[2] = (v1103_data + (v1100_data * v1101_data));
                int32_t v1112_a = ((v8_lead + 4) - 4) + 72;
                float v1121_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v1122_data = s0[54];
                float v1124_data = ir0[3];
                ir0[3] = (v1124_data + (v1121_data * v1122_data));
                int32_t v1133_a = ((v8_lead + 4) - 4) + 72;
                float v1142_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v1143_data = s0[70];
                float v1145_data = ir0[4];
                ir0[4] = (v1145_data + (v1142_data * v1143_data));
                int32_t v1154_a = ((v8_lead + 4) - 4) + 72;
                float v1163_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v1164_data = s0[86];
                float v1166_data = ir0[5];
                ir0[5] = (v1166_data + (v1163_data * v1164_data));
                int32_t v1175_a = ((v8_lead + 4) - 4) + 72;
                float v1184_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v1185_data = s0[102];
                float v1187_data = ir0[6];
                ir0[6] = (v1187_data + (v1184_data * v1185_data));
                int32_t v1196_a = ((v8_lead + 4) - 4) + 72;
                float v1205_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v1206_data = s0[118];
                float v1208_data = ir0[7];
                ir0[7] = (v1208_data + (v1205_data * v1206_data));
              }
              if (v8_lead < 12) {
                int32_t v1221_a = ((v8_lead + 4) - 4) + 84;
                float v1230_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v1231_data = s0[7];
                float v1233_data = ir0[0];
                ir0[0] = (v1233_data + (v1230_data * v1231_data));
                int32_t v1242_a = ((v8_lead + 4) - 4) + 84;
                float v1251_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v1252_data = s0[23];
                float v1254_data = ir0[1];
                ir0[1] = (v1254_data + (v1251_data * v1252_data));
                int32_t v1263_a = ((v8_lead + 4) - 4) + 84;
                float v1272_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v1273_data = s0[39];
                float v1275_data = ir0[2];
                ir0[2] = (v1275_data + (v1272_data * v1273_data));
                int32_t v1284_a = ((v8_lead + 4) - 4) + 84;
                float v1293_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v1294_data = s0[55];
                float v1296_data = ir0[3];
                ir0[3] = (v1296_data + (v1293_data * v1294_data));
                int32_t v1305_a = ((v8_lead + 4) - 4) + 84;
                float v1314_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v1315_data = s0[71];
                float v1317_data = ir0[4];
                ir0[4] = (v1317_data + (v1314_data * v1315_data));
                int32_t v1326_a = ((v8_lead + 4) - 4) + 84;
                float v1335_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v1336_data = s0[87];
                float v1338_data = ir0[5];
                ir0[5] = (v1338_data + (v1335_data * v1336_data));
                int32_t v1347_a = ((v8_lead + 4) - 4) + 84;
                float v1356_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v1357_data = s0[103];
                float v1359_data = ir0[6];
                ir0[6] = (v1359_data + (v1356_data * v1357_data));
                int32_t v1368_a = ((v8_lead + 4) - 4) + 84;
                float v1377_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v1378_data = s0[119];
                float v1380_data = ir0[7];
                ir0[7] = (v1380_data + (v1377_data * v1378_data));
              }
              if (v8_lead < 12) {
                int32_t v1393_a = ((v8_lead + 4) - 4) + 96;
                float v1402_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v1403_data = s0[8];
                float v1405_data = ir0[0];
                ir0[0] = (v1405_data + (v1402_data * v1403_data));
                int32_t v1414_a = ((v8_lead + 4) - 4) + 96;
                float v1423_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v1424_data = s0[24];
                float v1426_data = ir0[1];
                ir0[1] = (v1426_data + (v1423_data * v1424_data));
                int32_t v1435_a = ((v8_lead + 4) - 4) + 96;
                float v1444_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v1445_data = s0[40];
                float v1447_data = ir0[2];
                ir0[2] = (v1447_data + (v1444_data * v1445_data));
                int32_t v1456_a = ((v8_lead + 4) - 4) + 96;
                float v1465_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v1466_data = s0[56];
                float v1468_data = ir0[3];
                ir0[3] = (v1468_data + (v1465_data * v1466_data));
                int32_t v1477_a = ((v8_lead + 4) - 4) + 96;
                float v1486_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v1487_data = s0[72];
                float v1489_data = ir0[4];
                ir0[4] = (v1489_data + (v1486_data * v1487_data));
                int32_t v1498_a = ((v8_lead + 4) - 4) + 96;
                float v1507_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v1508_data = s0[88];
                float v1510_data = ir0[5];
                ir0[5] = (v1510_data + (v1507_data * v1508_data));
                int32_t v1519_a = ((v8_lead + 4) - 4) + 96;
                float v1528_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v1529_data = s0[104];
                float v1531_data = ir0[6];
                ir0[6] = (v1531_data + (v1528_data * v1529_data));
                int32_t v1540_a = ((v8_lead + 4) - 4) + 96;
                float v1549_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v1550_data = s0[120];
                float v1552_data = ir0[7];
                ir0[7] = (v1552_data + (v1549_data * v1550_data));
              }
              if (v8_lead < 12) {
                int32_t v1565_a = ((v8_lead + 4) - 4) + 108;
                float v1574_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1575_data = s0[9];
                float v1577_data = ir0[0];
                ir0[0] = (v1577_data + (v1574_data * v1575_data));
                int32_t v1586_a = ((v8_lead + 4) - 4) + 108;
                float v1595_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1596_data = s0[25];
                float v1598_data = ir0[1];
                ir0[1] = (v1598_data + (v1595_data * v1596_data));
                int32_t v1607_a = ((v8_lead + 4) - 4) + 108;
                float v1616_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1617_data = s0[41];
                float v1619_data = ir0[2];
                ir0[2] = (v1619_data + (v1616_data * v1617_data));
                int32_t v1628_a = ((v8_lead + 4) - 4) + 108;
                float v1637_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1638_data = s0[57];
                float v1640_data = ir0[3];
                ir0[3] = (v1640_data + (v1637_data * v1638_data));
                int32_t v1649_a = ((v8_lead + 4) - 4) + 108;
                float v1658_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1659_data = s0[73];
                float v1661_data = ir0[4];
                ir0[4] = (v1661_data + (v1658_data * v1659_data));
                int32_t v1670_a = ((v8_lead + 4) - 4) + 108;
                float v1679_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1680_data = s0[89];
                float v1682_data = ir0[5];
                ir0[5] = (v1682_data + (v1679_data * v1680_data));
                int32_t v1691_a = ((v8_lead + 4) - 4) + 108;
                float v1700_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1701_data = s0[105];
                float v1703_data = ir0[6];
                ir0[6] = (v1703_data + (v1700_data * v1701_data));
                int32_t v1712_a = ((v8_lead + 4) - 4) + 108;
                float v1721_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1722_data = s0[121];
                float v1724_data = ir0[7];
                ir0[7] = (v1724_data + (v1721_data * v1722_data));
              }
              if (v8_lead < 12) {
                int32_t v1737_a = ((v8_lead + 4) - 4) + 120;
                float v1746_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1747_data = s0[10];
                float v1749_data = ir0[0];
                ir0[0] = (v1749_data + (v1746_data * v1747_data));
                int32_t v1758_a = ((v8_lead + 4) - 4) + 120;
                float v1767_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1768_data = s0[26];
                float v1770_data = ir0[1];
                ir0[1] = (v1770_data + (v1767_data * v1768_data));
                int32_t v1779_a = ((v8_lead + 4) - 4) + 120;
                float v1788_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1789_data = s0[42];
                float v1791_data = ir0[2];
                ir0[2] = (v1791_data + (v1788_data * v1789_data));
                int32_t v1800_a = ((v8_lead + 4) - 4) + 120;
                float v1809_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1810_data = s0[58];
                float v1812_data = ir0[3];
                ir0[3] = (v1812_data + (v1809_data * v1810_data));
                int32_t v1821_a = ((v8_lead + 4) - 4) + 120;
                float v1830_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1831_data = s0[74];
                float v1833_data = ir0[4];
                ir0[4] = (v1833_data + (v1830_data * v1831_data));
                int32_t v1842_a = ((v8_lead + 4) - 4) + 120;
                float v1851_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1852_data = s0[90];
                float v1854_data = ir0[5];
                ir0[5] = (v1854_data + (v1851_data * v1852_data));
                int32_t v1863_a = ((v8_lead + 4) - 4) + 120;
                float v1872_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1873_data = s0[106];
                float v1875_data = ir0[6];
                ir0[6] = (v1875_data + (v1872_data * v1873_data));
                int32_t v1884_a = ((v8_lead + 4) - 4) + 120;
                float v1893_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1894_data = s0[122];
                float v1896_data = ir0[7];
                ir0[7] = (v1896_data + (v1893_data * v1894_data));
              }
              if (v8_lead < 12) {
                int32_t v1909_a = ((v8_lead + 4) - 4) + 132;
                float v1918_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1919_data = s0[11];
                float v1921_data = ir0[0];
                ir0[0] = (v1921_data + (v1918_data * v1919_data));
                int32_t v1930_a = ((v8_lead + 4) - 4) + 132;
                float v1939_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1940_data = s0[27];
                float v1942_data = ir0[1];
                ir0[1] = (v1942_data + (v1939_data * v1940_data));
                int32_t v1951_a = ((v8_lead + 4) - 4) + 132;
                float v1960_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1961_data = s0[43];
                float v1963_data = ir0[2];
                ir0[2] = (v1963_data + (v1960_data * v1961_data));
                int32_t v1972_a = ((v8_lead + 4) - 4) + 132;
                float v1981_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1982_data = s0[59];
                float v1984_data = ir0[3];
                ir0[3] = (v1984_data + (v1981_data * v1982_data));
                int32_t v1993_a = ((v8_lead + 4) - 4) + 132;
                float v2002_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v2003_data = s0[75];
                float v2005_data = ir0[4];
                ir0[4] = (v2005_data + (v2002_data * v2003_data));
                int32_t v2014_a = ((v8_lead + 4) - 4) + 132;
                float v2023_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v2024_data = s0[91];
                float v2026_data = ir0[5];
                ir0[5] = (v2026_data + (v2023_data * v2024_data));
                int32_t v2035_a = ((v8_lead + 4) - 4) + 132;
                float v2044_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v2045_data = s0[107];
                float v2047_data = ir0[6];
                ir0[6] = (v2047_data + (v2044_data * v2045_data));
                int32_t v2056_a = ((v8_lead + 4) - 4) + 132;
                float v2065_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v2066_data = s0[123];
                float v2068_data = ir0[7];
                ir0[7] = (v2068_data + (v2065_data * v2066_data));
              }
              if (v8_lead < 12) {
                int32_t v2081_a = ((v8_lead + 4) - 4) + 144;
                float v2090_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v2091_data = s0[12];
                float v2093_data = ir0[0];
                ir0[0] = (v2093_data + (v2090_data * v2091_data));
                int32_t v2102_a = ((v8_lead + 4) - 4) + 144;
                float v2111_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v2112_data = s0[28];
                float v2114_data = ir0[1];
                ir0[1] = (v2114_data + (v2111_data * v2112_data));
                int32_t v2123_a = ((v8_lead + 4) - 4) + 144;
                float v2132_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v2133_data = s0[44];
                float v2135_data = ir0[2];
                ir0[2] = (v2135_data + (v2132_data * v2133_data));
                int32_t v2144_a = ((v8_lead + 4) - 4) + 144;
                float v2153_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v2154_data = s0[60];
                float v2156_data = ir0[3];
                ir0[3] = (v2156_data + (v2153_data * v2154_data));
                int32_t v2165_a = ((v8_lead + 4) - 4) + 144;
                float v2174_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v2175_data = s0[76];
                float v2177_data = ir0[4];
                ir0[4] = (v2177_data + (v2174_data * v2175_data));
                int32_t v2186_a = ((v8_lead + 4) - 4) + 144;
                float v2195_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v2196_data = s0[92];
                float v2198_data = ir0[5];
                ir0[5] = (v2198_data + (v2195_data * v2196_data));
                int32_t v2207_a = ((v8_lead + 4) - 4) + 144;
                float v2216_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v2217_data = s0[108];
                float v2219_data = ir0[6];
                ir0[6] = (v2219_data + (v2216_data * v2217_data));
                int32_t v2228_a = ((v8_lead + 4) - 4) + 144;
                float v2237_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v2238_data = s0[124];
                float v2240_data = ir0[7];
                ir0[7] = (v2240_data + (v2237_data * v2238_data));
              }
              if (v8_lead < 12) {
                int32_t v2253_a = ((v8_lead + 4) - 4) + 156;
                float v2262_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v2263_data = s0[13];
                float v2265_data = ir0[0];
                ir0[0] = (v2265_data + (v2262_data * v2263_data));
                int32_t v2274_a = ((v8_lead + 4) - 4) + 156;
                float v2283_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v2284_data = s0[29];
                float v2286_data = ir0[1];
                ir0[1] = (v2286_data + (v2283_data * v2284_data));
                int32_t v2295_a = ((v8_lead + 4) - 4) + 156;
                float v2304_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v2305_data = s0[45];
                float v2307_data = ir0[2];
                ir0[2] = (v2307_data + (v2304_data * v2305_data));
                int32_t v2316_a = ((v8_lead + 4) - 4) + 156;
                float v2325_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v2326_data = s0[61];
                float v2328_data = ir0[3];
                ir0[3] = (v2328_data + (v2325_data * v2326_data));
                int32_t v2337_a = ((v8_lead + 4) - 4) + 156;
                float v2346_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v2347_data = s0[77];
                float v2349_data = ir0[4];
                ir0[4] = (v2349_data + (v2346_data * v2347_data));
                int32_t v2358_a = ((v8_lead + 4) - 4) + 156;
                float v2367_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v2368_data = s0[93];
                float v2370_data = ir0[5];
                ir0[5] = (v2370_data + (v2367_data * v2368_data));
                int32_t v2379_a = ((v8_lead + 4) - 4) + 156;
                float v2388_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v2389_data = s0[109];
                float v2391_data = ir0[6];
                ir0[6] = (v2391_data + (v2388_data * v2389_data));
                int32_t v2400_a = ((v8_lead + 4) - 4) + 156;
                float v2409_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v2410_data = s0[125];
                float v2412_data = ir0[7];
                ir0[7] = (v2412_data + (v2409_data * v2410_data));
              }
              if (v8_lead < 12) {
                int32_t v2425_a = ((v8_lead + 4) - 4) + 168;
                float v2434_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v2435_data = s0[14];
                float v2437_data = ir0[0];
                ir0[0] = (v2437_data + (v2434_data * v2435_data));
                int32_t v2446_a = ((v8_lead + 4) - 4) + 168;
                float v2455_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v2456_data = s0[30];
                float v2458_data = ir0[1];
                ir0[1] = (v2458_data + (v2455_data * v2456_data));
                int32_t v2467_a = ((v8_lead + 4) - 4) + 168;
                float v2476_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v2477_data = s0[46];
                float v2479_data = ir0[2];
                ir0[2] = (v2479_data + (v2476_data * v2477_data));
                int32_t v2488_a = ((v8_lead + 4) - 4) + 168;
                float v2497_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v2498_data = s0[62];
                float v2500_data = ir0[3];
                ir0[3] = (v2500_data + (v2497_data * v2498_data));
                int32_t v2509_a = ((v8_lead + 4) - 4) + 168;
                float v2518_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v2519_data = s0[78];
                float v2521_data = ir0[4];
                ir0[4] = (v2521_data + (v2518_data * v2519_data));
                int32_t v2530_a = ((v8_lead + 4) - 4) + 168;
                float v2539_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v2540_data = s0[94];
                float v2542_data = ir0[5];
                ir0[5] = (v2542_data + (v2539_data * v2540_data));
                int32_t v2551_a = ((v8_lead + 4) - 4) + 168;
                float v2560_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v2561_data = s0[110];
                float v2563_data = ir0[6];
                ir0[6] = (v2563_data + (v2560_data * v2561_data));
                int32_t v2572_a = ((v8_lead + 4) - 4) + 168;
                float v2581_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v2582_data = s0[126];
                float v2584_data = ir0[7];
                ir0[7] = (v2584_data + (v2581_data * v2582_data));
              }
              if (v8_lead < 12) {
                int32_t v2597_a = ((v8_lead + 4) - 4) + 180;
                float v2606_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v2607_data = s0[15];
                float v2609_data = ir0[0];
                ir0[0] = (v2609_data + (v2606_data * v2607_data));
                int32_t v2618_a = ((v8_lead + 4) - 4) + 180;
                float v2627_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v2628_data = s0[31];
                float v2630_data = ir0[1];
                ir0[1] = (v2630_data + (v2627_data * v2628_data));
                int32_t v2639_a = ((v8_lead + 4) - 4) + 180;
                float v2648_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v2649_data = s0[47];
                float v2651_data = ir0[2];
                ir0[2] = (v2651_data + (v2648_data * v2649_data));
                int32_t v2660_a = ((v8_lead + 4) - 4) + 180;
                float v2669_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v2670_data = s0[63];
                float v2672_data = ir0[3];
                ir0[3] = (v2672_data + (v2669_data * v2670_data));
                int32_t v2681_a = ((v8_lead + 4) - 4) + 180;
                float v2690_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v2691_data = s0[79];
                float v2693_data = ir0[4];
                ir0[4] = (v2693_data + (v2690_data * v2691_data));
                int32_t v2702_a = ((v8_lead + 4) - 4) + 180;
                float v2711_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v2712_data = s0[95];
                float v2714_data = ir0[5];
                ir0[5] = (v2714_data + (v2711_data * v2712_data));
                int32_t v2723_a = ((v8_lead + 4) - 4) + 180;
                float v2732_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v2733_data = s0[111];
                float v2735_data = ir0[6];
                ir0[6] = (v2735_data + (v2732_data * v2733_data));
                int32_t v2744_a = ((v8_lead + 4) - 4) + 180;
                float v2753_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v2754_data = s0[127];
                float v2756_data = ir0[7];
                ir0[7] = (v2756_data + (v2753_data * v2754_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v2762_n1 = 0; v2762_n1 < 8; ++v2762_n1) {
                  int32_t v2763_a = 0 + v2762_n1;
                  float v2765_data = ir0[v2762_n1];
                  r0[v2762_n1] = v2765_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v2771_i1 = 0; v2771_i1 < 8; ++v2771_i1) {
                  int32_t v2772_a = 0 + v2771_i1;
                  float v2774_data = r0[v2771_i1];
                  glb_m0[(v8_lead + (v2771_i1 * 12))] = v2774_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


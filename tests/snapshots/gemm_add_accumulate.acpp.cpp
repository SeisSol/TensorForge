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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
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
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[8]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                float v16_data = glb_m1[v8_lead];
                float v17_data = s0[0];
                float v19_data = ir0[0];
                ir0[0] = (v19_data + (v16_data * v17_data));
                float v27_data = glb_m1[v8_lead];
                float v28_data = s0[16];
                float v30_data = ir0[1];
                ir0[1] = (v30_data + (v27_data * v28_data));
                float v38_data = glb_m1[v8_lead];
                float v39_data = s0[33];
                float v41_data = ir0[2];
                ir0[2] = (v41_data + (v38_data * v39_data));
                float v49_data = glb_m1[v8_lead];
                float v50_data = s0[49];
                float v52_data = ir0[3];
                ir0[3] = (v52_data + (v49_data * v50_data));
                float v60_data = glb_m1[v8_lead];
                float v61_data = s0[66];
                float v63_data = ir0[4];
                ir0[4] = (v63_data + (v60_data * v61_data));
                float v71_data = glb_m1[v8_lead];
                float v72_data = s0[82];
                float v74_data = ir0[5];
                ir0[5] = (v74_data + (v71_data * v72_data));
                float v82_data = glb_m1[v8_lead];
                float v83_data = s0[99];
                float v85_data = ir0[6];
                ir0[6] = (v85_data + (v82_data * v83_data));
                float v93_data = glb_m1[v8_lead];
                float v94_data = s0[115];
                float v96_data = ir0[7];
                ir0[7] = (v96_data + (v93_data * v94_data));
              }
              if (v8_lead < 12) {
                float v108_data = glb_m1[(v8_lead + 12)];
                float v109_data = s0[1];
                float v111_data = ir0[0];
                ir0[0] = (v111_data + (v108_data * v109_data));
                float v119_data = glb_m1[(v8_lead + 12)];
                float v120_data = s0[17];
                float v122_data = ir0[1];
                ir0[1] = (v122_data + (v119_data * v120_data));
                float v130_data = glb_m1[(v8_lead + 12)];
                float v131_data = s0[32];
                float v133_data = ir0[2];
                ir0[2] = (v133_data + (v130_data * v131_data));
                float v141_data = glb_m1[(v8_lead + 12)];
                float v142_data = s0[48];
                float v144_data = ir0[3];
                ir0[3] = (v144_data + (v141_data * v142_data));
                float v152_data = glb_m1[(v8_lead + 12)];
                float v153_data = s0[67];
                float v155_data = ir0[4];
                ir0[4] = (v155_data + (v152_data * v153_data));
                float v163_data = glb_m1[(v8_lead + 12)];
                float v164_data = s0[83];
                float v166_data = ir0[5];
                ir0[5] = (v166_data + (v163_data * v164_data));
                float v174_data = glb_m1[(v8_lead + 12)];
                float v175_data = s0[98];
                float v177_data = ir0[6];
                ir0[6] = (v177_data + (v174_data * v175_data));
                float v185_data = glb_m1[(v8_lead + 12)];
                float v186_data = s0[114];
                float v188_data = ir0[7];
                ir0[7] = (v188_data + (v185_data * v186_data));
              }
              if (v8_lead < 12) {
                float v200_data = glb_m1[(v8_lead + 24)];
                float v201_data = s0[2];
                float v203_data = ir0[0];
                ir0[0] = (v203_data + (v200_data * v201_data));
                float v211_data = glb_m1[(v8_lead + 24)];
                float v212_data = s0[18];
                float v214_data = ir0[1];
                ir0[1] = (v214_data + (v211_data * v212_data));
                float v222_data = glb_m1[(v8_lead + 24)];
                float v223_data = s0[35];
                float v225_data = ir0[2];
                ir0[2] = (v225_data + (v222_data * v223_data));
                float v233_data = glb_m1[(v8_lead + 24)];
                float v234_data = s0[51];
                float v236_data = ir0[3];
                ir0[3] = (v236_data + (v233_data * v234_data));
                float v244_data = glb_m1[(v8_lead + 24)];
                float v245_data = s0[64];
                float v247_data = ir0[4];
                ir0[4] = (v247_data + (v244_data * v245_data));
                float v255_data = glb_m1[(v8_lead + 24)];
                float v256_data = s0[80];
                float v258_data = ir0[5];
                ir0[5] = (v258_data + (v255_data * v256_data));
                float v266_data = glb_m1[(v8_lead + 24)];
                float v267_data = s0[97];
                float v269_data = ir0[6];
                ir0[6] = (v269_data + (v266_data * v267_data));
                float v277_data = glb_m1[(v8_lead + 24)];
                float v278_data = s0[113];
                float v280_data = ir0[7];
                ir0[7] = (v280_data + (v277_data * v278_data));
              }
              if (v8_lead < 12) {
                float v292_data = glb_m1[(v8_lead + 36)];
                float v293_data = s0[3];
                float v295_data = ir0[0];
                ir0[0] = (v295_data + (v292_data * v293_data));
                float v303_data = glb_m1[(v8_lead + 36)];
                float v304_data = s0[19];
                float v306_data = ir0[1];
                ir0[1] = (v306_data + (v303_data * v304_data));
                float v314_data = glb_m1[(v8_lead + 36)];
                float v315_data = s0[34];
                float v317_data = ir0[2];
                ir0[2] = (v317_data + (v314_data * v315_data));
                float v325_data = glb_m1[(v8_lead + 36)];
                float v326_data = s0[50];
                float v328_data = ir0[3];
                ir0[3] = (v328_data + (v325_data * v326_data));
                float v336_data = glb_m1[(v8_lead + 36)];
                float v337_data = s0[65];
                float v339_data = ir0[4];
                ir0[4] = (v339_data + (v336_data * v337_data));
                float v347_data = glb_m1[(v8_lead + 36)];
                float v348_data = s0[81];
                float v350_data = ir0[5];
                ir0[5] = (v350_data + (v347_data * v348_data));
                float v358_data = glb_m1[(v8_lead + 36)];
                float v359_data = s0[96];
                float v361_data = ir0[6];
                ir0[6] = (v361_data + (v358_data * v359_data));
                float v369_data = glb_m1[(v8_lead + 36)];
                float v370_data = s0[112];
                float v372_data = ir0[7];
                ir0[7] = (v372_data + (v369_data * v370_data));
              }
              if (v8_lead < 12) {
                float v384_data = glb_m1[(v8_lead + 48)];
                float v385_data = s0[4];
                float v387_data = ir0[0];
                ir0[0] = (v387_data + (v384_data * v385_data));
                float v395_data = glb_m1[(v8_lead + 48)];
                float v396_data = s0[20];
                float v398_data = ir0[1];
                ir0[1] = (v398_data + (v395_data * v396_data));
                float v406_data = glb_m1[(v8_lead + 48)];
                float v407_data = s0[37];
                float v409_data = ir0[2];
                ir0[2] = (v409_data + (v406_data * v407_data));
                float v417_data = glb_m1[(v8_lead + 48)];
                float v418_data = s0[53];
                float v420_data = ir0[3];
                ir0[3] = (v420_data + (v417_data * v418_data));
                float v428_data = glb_m1[(v8_lead + 48)];
                float v429_data = s0[70];
                float v431_data = ir0[4];
                ir0[4] = (v431_data + (v428_data * v429_data));
                float v439_data = glb_m1[(v8_lead + 48)];
                float v440_data = s0[86];
                float v442_data = ir0[5];
                ir0[5] = (v442_data + (v439_data * v440_data));
                float v450_data = glb_m1[(v8_lead + 48)];
                float v451_data = s0[103];
                float v453_data = ir0[6];
                ir0[6] = (v453_data + (v450_data * v451_data));
                float v461_data = glb_m1[(v8_lead + 48)];
                float v462_data = s0[119];
                float v464_data = ir0[7];
                ir0[7] = (v464_data + (v461_data * v462_data));
              }
              if (v8_lead < 12) {
                float v476_data = glb_m1[(v8_lead + 60)];
                float v477_data = s0[5];
                float v479_data = ir0[0];
                ir0[0] = (v479_data + (v476_data * v477_data));
                float v487_data = glb_m1[(v8_lead + 60)];
                float v488_data = s0[21];
                float v490_data = ir0[1];
                ir0[1] = (v490_data + (v487_data * v488_data));
                float v498_data = glb_m1[(v8_lead + 60)];
                float v499_data = s0[36];
                float v501_data = ir0[2];
                ir0[2] = (v501_data + (v498_data * v499_data));
                float v509_data = glb_m1[(v8_lead + 60)];
                float v510_data = s0[52];
                float v512_data = ir0[3];
                ir0[3] = (v512_data + (v509_data * v510_data));
                float v520_data = glb_m1[(v8_lead + 60)];
                float v521_data = s0[71];
                float v523_data = ir0[4];
                ir0[4] = (v523_data + (v520_data * v521_data));
                float v531_data = glb_m1[(v8_lead + 60)];
                float v532_data = s0[87];
                float v534_data = ir0[5];
                ir0[5] = (v534_data + (v531_data * v532_data));
                float v542_data = glb_m1[(v8_lead + 60)];
                float v543_data = s0[102];
                float v545_data = ir0[6];
                ir0[6] = (v545_data + (v542_data * v543_data));
                float v553_data = glb_m1[(v8_lead + 60)];
                float v554_data = s0[118];
                float v556_data = ir0[7];
                ir0[7] = (v556_data + (v553_data * v554_data));
              }
              if (v8_lead < 12) {
                float v568_data = glb_m1[(v8_lead + 72)];
                float v569_data = s0[6];
                float v571_data = ir0[0];
                ir0[0] = (v571_data + (v568_data * v569_data));
                float v579_data = glb_m1[(v8_lead + 72)];
                float v580_data = s0[22];
                float v582_data = ir0[1];
                ir0[1] = (v582_data + (v579_data * v580_data));
                float v590_data = glb_m1[(v8_lead + 72)];
                float v591_data = s0[39];
                float v593_data = ir0[2];
                ir0[2] = (v593_data + (v590_data * v591_data));
                float v601_data = glb_m1[(v8_lead + 72)];
                float v602_data = s0[55];
                float v604_data = ir0[3];
                ir0[3] = (v604_data + (v601_data * v602_data));
                float v612_data = glb_m1[(v8_lead + 72)];
                float v613_data = s0[68];
                float v615_data = ir0[4];
                ir0[4] = (v615_data + (v612_data * v613_data));
                float v623_data = glb_m1[(v8_lead + 72)];
                float v624_data = s0[84];
                float v626_data = ir0[5];
                ir0[5] = (v626_data + (v623_data * v624_data));
                float v634_data = glb_m1[(v8_lead + 72)];
                float v635_data = s0[101];
                float v637_data = ir0[6];
                ir0[6] = (v637_data + (v634_data * v635_data));
                float v645_data = glb_m1[(v8_lead + 72)];
                float v646_data = s0[117];
                float v648_data = ir0[7];
                ir0[7] = (v648_data + (v645_data * v646_data));
              }
              if (v8_lead < 12) {
                float v660_data = glb_m1[(v8_lead + 84)];
                float v661_data = s0[7];
                float v663_data = ir0[0];
                ir0[0] = (v663_data + (v660_data * v661_data));
                float v671_data = glb_m1[(v8_lead + 84)];
                float v672_data = s0[23];
                float v674_data = ir0[1];
                ir0[1] = (v674_data + (v671_data * v672_data));
                float v682_data = glb_m1[(v8_lead + 84)];
                float v683_data = s0[38];
                float v685_data = ir0[2];
                ir0[2] = (v685_data + (v682_data * v683_data));
                float v693_data = glb_m1[(v8_lead + 84)];
                float v694_data = s0[54];
                float v696_data = ir0[3];
                ir0[3] = (v696_data + (v693_data * v694_data));
                float v704_data = glb_m1[(v8_lead + 84)];
                float v705_data = s0[69];
                float v707_data = ir0[4];
                ir0[4] = (v707_data + (v704_data * v705_data));
                float v715_data = glb_m1[(v8_lead + 84)];
                float v716_data = s0[85];
                float v718_data = ir0[5];
                ir0[5] = (v718_data + (v715_data * v716_data));
                float v726_data = glb_m1[(v8_lead + 84)];
                float v727_data = s0[100];
                float v729_data = ir0[6];
                ir0[6] = (v729_data + (v726_data * v727_data));
                float v737_data = glb_m1[(v8_lead + 84)];
                float v738_data = s0[116];
                float v740_data = ir0[7];
                ir0[7] = (v740_data + (v737_data * v738_data));
              }
              if (v8_lead < 12) {
                float v752_data = glb_m1[(v8_lead + 96)];
                float v753_data = s0[8];
                float v755_data = ir0[0];
                ir0[0] = (v755_data + (v752_data * v753_data));
                float v763_data = glb_m1[(v8_lead + 96)];
                float v764_data = s0[24];
                float v766_data = ir0[1];
                ir0[1] = (v766_data + (v763_data * v764_data));
                float v774_data = glb_m1[(v8_lead + 96)];
                float v775_data = s0[41];
                float v777_data = ir0[2];
                ir0[2] = (v777_data + (v774_data * v775_data));
                float v785_data = glb_m1[(v8_lead + 96)];
                float v786_data = s0[57];
                float v788_data = ir0[3];
                ir0[3] = (v788_data + (v785_data * v786_data));
                float v796_data = glb_m1[(v8_lead + 96)];
                float v797_data = s0[74];
                float v799_data = ir0[4];
                ir0[4] = (v799_data + (v796_data * v797_data));
                float v807_data = glb_m1[(v8_lead + 96)];
                float v808_data = s0[90];
                float v810_data = ir0[5];
                ir0[5] = (v810_data + (v807_data * v808_data));
                float v818_data = glb_m1[(v8_lead + 96)];
                float v819_data = s0[107];
                float v821_data = ir0[6];
                ir0[6] = (v821_data + (v818_data * v819_data));
                float v829_data = glb_m1[(v8_lead + 96)];
                float v830_data = s0[123];
                float v832_data = ir0[7];
                ir0[7] = (v832_data + (v829_data * v830_data));
              }
              if (v8_lead < 12) {
                float v844_data = glb_m1[(v8_lead + 108)];
                float v845_data = s0[9];
                float v847_data = ir0[0];
                ir0[0] = (v847_data + (v844_data * v845_data));
                float v855_data = glb_m1[(v8_lead + 108)];
                float v856_data = s0[25];
                float v858_data = ir0[1];
                ir0[1] = (v858_data + (v855_data * v856_data));
                float v866_data = glb_m1[(v8_lead + 108)];
                float v867_data = s0[40];
                float v869_data = ir0[2];
                ir0[2] = (v869_data + (v866_data * v867_data));
                float v877_data = glb_m1[(v8_lead + 108)];
                float v878_data = s0[56];
                float v880_data = ir0[3];
                ir0[3] = (v880_data + (v877_data * v878_data));
                float v888_data = glb_m1[(v8_lead + 108)];
                float v889_data = s0[75];
                float v891_data = ir0[4];
                ir0[4] = (v891_data + (v888_data * v889_data));
                float v899_data = glb_m1[(v8_lead + 108)];
                float v900_data = s0[91];
                float v902_data = ir0[5];
                ir0[5] = (v902_data + (v899_data * v900_data));
                float v910_data = glb_m1[(v8_lead + 108)];
                float v911_data = s0[106];
                float v913_data = ir0[6];
                ir0[6] = (v913_data + (v910_data * v911_data));
                float v921_data = glb_m1[(v8_lead + 108)];
                float v922_data = s0[122];
                float v924_data = ir0[7];
                ir0[7] = (v924_data + (v921_data * v922_data));
              }
              if (v8_lead < 12) {
                float v936_data = glb_m1[(v8_lead + 120)];
                float v937_data = s0[10];
                float v939_data = ir0[0];
                ir0[0] = (v939_data + (v936_data * v937_data));
                float v947_data = glb_m1[(v8_lead + 120)];
                float v948_data = s0[26];
                float v950_data = ir0[1];
                ir0[1] = (v950_data + (v947_data * v948_data));
                float v958_data = glb_m1[(v8_lead + 120)];
                float v959_data = s0[43];
                float v961_data = ir0[2];
                ir0[2] = (v961_data + (v958_data * v959_data));
                float v969_data = glb_m1[(v8_lead + 120)];
                float v970_data = s0[59];
                float v972_data = ir0[3];
                ir0[3] = (v972_data + (v969_data * v970_data));
                float v980_data = glb_m1[(v8_lead + 120)];
                float v981_data = s0[72];
                float v983_data = ir0[4];
                ir0[4] = (v983_data + (v980_data * v981_data));
                float v991_data = glb_m1[(v8_lead + 120)];
                float v992_data = s0[88];
                float v994_data = ir0[5];
                ir0[5] = (v994_data + (v991_data * v992_data));
                float v1002_data = glb_m1[(v8_lead + 120)];
                float v1003_data = s0[105];
                float v1005_data = ir0[6];
                ir0[6] = (v1005_data + (v1002_data * v1003_data));
                float v1013_data = glb_m1[(v8_lead + 120)];
                float v1014_data = s0[121];
                float v1016_data = ir0[7];
                ir0[7] = (v1016_data + (v1013_data * v1014_data));
              }
              if (v8_lead < 12) {
                float v1028_data = glb_m1[(v8_lead + 132)];
                float v1029_data = s0[11];
                float v1031_data = ir0[0];
                ir0[0] = (v1031_data + (v1028_data * v1029_data));
                float v1039_data = glb_m1[(v8_lead + 132)];
                float v1040_data = s0[27];
                float v1042_data = ir0[1];
                ir0[1] = (v1042_data + (v1039_data * v1040_data));
                float v1050_data = glb_m1[(v8_lead + 132)];
                float v1051_data = s0[42];
                float v1053_data = ir0[2];
                ir0[2] = (v1053_data + (v1050_data * v1051_data));
                float v1061_data = glb_m1[(v8_lead + 132)];
                float v1062_data = s0[58];
                float v1064_data = ir0[3];
                ir0[3] = (v1064_data + (v1061_data * v1062_data));
                float v1072_data = glb_m1[(v8_lead + 132)];
                float v1073_data = s0[73];
                float v1075_data = ir0[4];
                ir0[4] = (v1075_data + (v1072_data * v1073_data));
                float v1083_data = glb_m1[(v8_lead + 132)];
                float v1084_data = s0[89];
                float v1086_data = ir0[5];
                ir0[5] = (v1086_data + (v1083_data * v1084_data));
                float v1094_data = glb_m1[(v8_lead + 132)];
                float v1095_data = s0[104];
                float v1097_data = ir0[6];
                ir0[6] = (v1097_data + (v1094_data * v1095_data));
                float v1105_data = glb_m1[(v8_lead + 132)];
                float v1106_data = s0[120];
                float v1108_data = ir0[7];
                ir0[7] = (v1108_data + (v1105_data * v1106_data));
              }
              if (v8_lead < 12) {
                float v1120_data = glb_m1[(v8_lead + 144)];
                float v1121_data = s0[12];
                float v1123_data = ir0[0];
                ir0[0] = (v1123_data + (v1120_data * v1121_data));
                float v1131_data = glb_m1[(v8_lead + 144)];
                float v1132_data = s0[28];
                float v1134_data = ir0[1];
                ir0[1] = (v1134_data + (v1131_data * v1132_data));
                float v1142_data = glb_m1[(v8_lead + 144)];
                float v1143_data = s0[45];
                float v1145_data = ir0[2];
                ir0[2] = (v1145_data + (v1142_data * v1143_data));
                float v1153_data = glb_m1[(v8_lead + 144)];
                float v1154_data = s0[61];
                float v1156_data = ir0[3];
                ir0[3] = (v1156_data + (v1153_data * v1154_data));
                float v1164_data = glb_m1[(v8_lead + 144)];
                float v1165_data = s0[78];
                float v1167_data = ir0[4];
                ir0[4] = (v1167_data + (v1164_data * v1165_data));
                float v1175_data = glb_m1[(v8_lead + 144)];
                float v1176_data = s0[94];
                float v1178_data = ir0[5];
                ir0[5] = (v1178_data + (v1175_data * v1176_data));
                float v1186_data = glb_m1[(v8_lead + 144)];
                float v1187_data = s0[111];
                float v1189_data = ir0[6];
                ir0[6] = (v1189_data + (v1186_data * v1187_data));
                float v1197_data = glb_m1[(v8_lead + 144)];
                float v1198_data = s0[127];
                float v1200_data = ir0[7];
                ir0[7] = (v1200_data + (v1197_data * v1198_data));
              }
              if (v8_lead < 12) {
                float v1212_data = glb_m1[(v8_lead + 156)];
                float v1213_data = s0[13];
                float v1215_data = ir0[0];
                ir0[0] = (v1215_data + (v1212_data * v1213_data));
                float v1223_data = glb_m1[(v8_lead + 156)];
                float v1224_data = s0[29];
                float v1226_data = ir0[1];
                ir0[1] = (v1226_data + (v1223_data * v1224_data));
                float v1234_data = glb_m1[(v8_lead + 156)];
                float v1235_data = s0[44];
                float v1237_data = ir0[2];
                ir0[2] = (v1237_data + (v1234_data * v1235_data));
                float v1245_data = glb_m1[(v8_lead + 156)];
                float v1246_data = s0[60];
                float v1248_data = ir0[3];
                ir0[3] = (v1248_data + (v1245_data * v1246_data));
                float v1256_data = glb_m1[(v8_lead + 156)];
                float v1257_data = s0[79];
                float v1259_data = ir0[4];
                ir0[4] = (v1259_data + (v1256_data * v1257_data));
                float v1267_data = glb_m1[(v8_lead + 156)];
                float v1268_data = s0[95];
                float v1270_data = ir0[5];
                ir0[5] = (v1270_data + (v1267_data * v1268_data));
                float v1278_data = glb_m1[(v8_lead + 156)];
                float v1279_data = s0[110];
                float v1281_data = ir0[6];
                ir0[6] = (v1281_data + (v1278_data * v1279_data));
                float v1289_data = glb_m1[(v8_lead + 156)];
                float v1290_data = s0[126];
                float v1292_data = ir0[7];
                ir0[7] = (v1292_data + (v1289_data * v1290_data));
              }
              if (v8_lead < 12) {
                float v1304_data = glb_m1[(v8_lead + 168)];
                float v1305_data = s0[14];
                float v1307_data = ir0[0];
                ir0[0] = (v1307_data + (v1304_data * v1305_data));
                float v1315_data = glb_m1[(v8_lead + 168)];
                float v1316_data = s0[30];
                float v1318_data = ir0[1];
                ir0[1] = (v1318_data + (v1315_data * v1316_data));
                float v1326_data = glb_m1[(v8_lead + 168)];
                float v1327_data = s0[47];
                float v1329_data = ir0[2];
                ir0[2] = (v1329_data + (v1326_data * v1327_data));
                float v1337_data = glb_m1[(v8_lead + 168)];
                float v1338_data = s0[63];
                float v1340_data = ir0[3];
                ir0[3] = (v1340_data + (v1337_data * v1338_data));
                float v1348_data = glb_m1[(v8_lead + 168)];
                float v1349_data = s0[76];
                float v1351_data = ir0[4];
                ir0[4] = (v1351_data + (v1348_data * v1349_data));
                float v1359_data = glb_m1[(v8_lead + 168)];
                float v1360_data = s0[92];
                float v1362_data = ir0[5];
                ir0[5] = (v1362_data + (v1359_data * v1360_data));
                float v1370_data = glb_m1[(v8_lead + 168)];
                float v1371_data = s0[109];
                float v1373_data = ir0[6];
                ir0[6] = (v1373_data + (v1370_data * v1371_data));
                float v1381_data = glb_m1[(v8_lead + 168)];
                float v1382_data = s0[125];
                float v1384_data = ir0[7];
                ir0[7] = (v1384_data + (v1381_data * v1382_data));
              }
              if (v8_lead < 12) {
                float v1396_data = glb_m1[(v8_lead + 180)];
                float v1397_data = s0[15];
                float v1399_data = ir0[0];
                ir0[0] = (v1399_data + (v1396_data * v1397_data));
                float v1407_data = glb_m1[(v8_lead + 180)];
                float v1408_data = s0[31];
                float v1410_data = ir0[1];
                ir0[1] = (v1410_data + (v1407_data * v1408_data));
                float v1418_data = glb_m1[(v8_lead + 180)];
                float v1419_data = s0[46];
                float v1421_data = ir0[2];
                ir0[2] = (v1421_data + (v1418_data * v1419_data));
                float v1429_data = glb_m1[(v8_lead + 180)];
                float v1430_data = s0[62];
                float v1432_data = ir0[3];
                ir0[3] = (v1432_data + (v1429_data * v1430_data));
                float v1440_data = glb_m1[(v8_lead + 180)];
                float v1441_data = s0[77];
                float v1443_data = ir0[4];
                ir0[4] = (v1443_data + (v1440_data * v1441_data));
                float v1451_data = glb_m1[(v8_lead + 180)];
                float v1452_data = s0[93];
                float v1454_data = ir0[5];
                ir0[5] = (v1454_data + (v1451_data * v1452_data));
                float v1462_data = glb_m1[(v8_lead + 180)];
                float v1463_data = s0[108];
                float v1465_data = ir0[6];
                ir0[6] = (v1465_data + (v1462_data * v1463_data));
                float v1473_data = glb_m1[(v8_lead + 180)];
                float v1474_data = s0[124];
                float v1476_data = ir0[7];
                ir0[7] = (v1476_data + (v1473_data * v1474_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1482_n1 = 0; v1482_n1 < 8; ++v1482_n1) {
                  float v1484_data = ir0[v1482_n1];
                  float v1492_data = glb_m0[(v8_lead + (v1482_n1 * 12))];
                  r0[v1482_n1] = (v1492_data + v1484_data);
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1499_i1 = 0; v1499_i1 < 8; ++v1499_i1) {
                  float v1501_data = r0[v1499_i1];
                  glb_m0[(v8_lead + (v1499_i1 * 12))] = v1501_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


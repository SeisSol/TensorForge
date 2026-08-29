// === base name ===
kernel_16c847f49d

// === header ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_16c847f49d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_16c847f49d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2304, cgh); {
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
          double* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              double* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<double, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<double, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 0];
              *(sycl::vec<double, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 32] = *(sycl::vec<double, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 32];
              *(sycl::vec<double, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<double, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 64];
              *(sycl::vec<double, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 96] = *(sycl::vec<double, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 96];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              double r0[8]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              double ir0[8]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                double v16_data = glb_m1[v8_lead];
                double v17_data = s0[0];
                double v19_data = ir0[0];
                ir0[0] = (v19_data + (v16_data * v17_data));
                double v27_data = glb_m1[v8_lead];
                double v28_data = s0[17];
                double v30_data = ir0[1];
                ir0[1] = (v30_data + (v27_data * v28_data));
                double v38_data = glb_m1[v8_lead];
                double v39_data = s0[34];
                double v41_data = ir0[2];
                ir0[2] = (v41_data + (v38_data * v39_data));
                double v49_data = glb_m1[v8_lead];
                double v50_data = s0[51];
                double v52_data = ir0[3];
                ir0[3] = (v52_data + (v49_data * v50_data));
                double v60_data = glb_m1[v8_lead];
                double v61_data = s0[68];
                double v63_data = ir0[4];
                ir0[4] = (v63_data + (v60_data * v61_data));
                double v71_data = glb_m1[v8_lead];
                double v72_data = s0[85];
                double v74_data = ir0[5];
                ir0[5] = (v74_data + (v71_data * v72_data));
                double v82_data = glb_m1[v8_lead];
                double v83_data = s0[102];
                double v85_data = ir0[6];
                ir0[6] = (v85_data + (v82_data * v83_data));
                double v93_data = glb_m1[v8_lead];
                double v94_data = s0[119];
                double v96_data = ir0[7];
                ir0[7] = (v96_data + (v93_data * v94_data));
              }
              if (v8_lead < 12) {
                double v108_data = glb_m1[(v8_lead + 12)];
                double v109_data = s0[1];
                double v111_data = ir0[0];
                ir0[0] = (v111_data + (v108_data * v109_data));
                double v119_data = glb_m1[(v8_lead + 12)];
                double v120_data = s0[16];
                double v122_data = ir0[1];
                ir0[1] = (v122_data + (v119_data * v120_data));
                double v130_data = glb_m1[(v8_lead + 12)];
                double v131_data = s0[35];
                double v133_data = ir0[2];
                ir0[2] = (v133_data + (v130_data * v131_data));
                double v141_data = glb_m1[(v8_lead + 12)];
                double v142_data = s0[50];
                double v144_data = ir0[3];
                ir0[3] = (v144_data + (v141_data * v142_data));
                double v152_data = glb_m1[(v8_lead + 12)];
                double v153_data = s0[69];
                double v155_data = ir0[4];
                ir0[4] = (v155_data + (v152_data * v153_data));
                double v163_data = glb_m1[(v8_lead + 12)];
                double v164_data = s0[84];
                double v166_data = ir0[5];
                ir0[5] = (v166_data + (v163_data * v164_data));
                double v174_data = glb_m1[(v8_lead + 12)];
                double v175_data = s0[103];
                double v177_data = ir0[6];
                ir0[6] = (v177_data + (v174_data * v175_data));
                double v185_data = glb_m1[(v8_lead + 12)];
                double v186_data = s0[118];
                double v188_data = ir0[7];
                ir0[7] = (v188_data + (v185_data * v186_data));
              }
              if (v8_lead < 12) {
                double v200_data = glb_m1[(v8_lead + 24)];
                double v201_data = s0[2];
                double v203_data = ir0[0];
                ir0[0] = (v203_data + (v200_data * v201_data));
                double v211_data = glb_m1[(v8_lead + 24)];
                double v212_data = s0[19];
                double v214_data = ir0[1];
                ir0[1] = (v214_data + (v211_data * v212_data));
                double v222_data = glb_m1[(v8_lead + 24)];
                double v223_data = s0[32];
                double v225_data = ir0[2];
                ir0[2] = (v225_data + (v222_data * v223_data));
                double v233_data = glb_m1[(v8_lead + 24)];
                double v234_data = s0[49];
                double v236_data = ir0[3];
                ir0[3] = (v236_data + (v233_data * v234_data));
                double v244_data = glb_m1[(v8_lead + 24)];
                double v245_data = s0[70];
                double v247_data = ir0[4];
                ir0[4] = (v247_data + (v244_data * v245_data));
                double v255_data = glb_m1[(v8_lead + 24)];
                double v256_data = s0[87];
                double v258_data = ir0[5];
                ir0[5] = (v258_data + (v255_data * v256_data));
                double v266_data = glb_m1[(v8_lead + 24)];
                double v267_data = s0[100];
                double v269_data = ir0[6];
                ir0[6] = (v269_data + (v266_data * v267_data));
                double v277_data = glb_m1[(v8_lead + 24)];
                double v278_data = s0[117];
                double v280_data = ir0[7];
                ir0[7] = (v280_data + (v277_data * v278_data));
              }
              if (v8_lead < 12) {
                double v292_data = glb_m1[(v8_lead + 36)];
                double v293_data = s0[3];
                double v295_data = ir0[0];
                ir0[0] = (v295_data + (v292_data * v293_data));
                double v303_data = glb_m1[(v8_lead + 36)];
                double v304_data = s0[18];
                double v306_data = ir0[1];
                ir0[1] = (v306_data + (v303_data * v304_data));
                double v314_data = glb_m1[(v8_lead + 36)];
                double v315_data = s0[33];
                double v317_data = ir0[2];
                ir0[2] = (v317_data + (v314_data * v315_data));
                double v325_data = glb_m1[(v8_lead + 36)];
                double v326_data = s0[48];
                double v328_data = ir0[3];
                ir0[3] = (v328_data + (v325_data * v326_data));
                double v336_data = glb_m1[(v8_lead + 36)];
                double v337_data = s0[71];
                double v339_data = ir0[4];
                ir0[4] = (v339_data + (v336_data * v337_data));
                double v347_data = glb_m1[(v8_lead + 36)];
                double v348_data = s0[86];
                double v350_data = ir0[5];
                ir0[5] = (v350_data + (v347_data * v348_data));
                double v358_data = glb_m1[(v8_lead + 36)];
                double v359_data = s0[101];
                double v361_data = ir0[6];
                ir0[6] = (v361_data + (v358_data * v359_data));
                double v369_data = glb_m1[(v8_lead + 36)];
                double v370_data = s0[116];
                double v372_data = ir0[7];
                ir0[7] = (v372_data + (v369_data * v370_data));
              }
              if (v8_lead < 12) {
                double v384_data = glb_m1[(v8_lead + 48)];
                double v385_data = s0[4];
                double v387_data = ir0[0];
                ir0[0] = (v387_data + (v384_data * v385_data));
                double v395_data = glb_m1[(v8_lead + 48)];
                double v396_data = s0[21];
                double v398_data = ir0[1];
                ir0[1] = (v398_data + (v395_data * v396_data));
                double v406_data = glb_m1[(v8_lead + 48)];
                double v407_data = s0[38];
                double v409_data = ir0[2];
                ir0[2] = (v409_data + (v406_data * v407_data));
                double v417_data = glb_m1[(v8_lead + 48)];
                double v418_data = s0[55];
                double v420_data = ir0[3];
                ir0[3] = (v420_data + (v417_data * v418_data));
                double v428_data = glb_m1[(v8_lead + 48)];
                double v429_data = s0[64];
                double v431_data = ir0[4];
                ir0[4] = (v431_data + (v428_data * v429_data));
                double v439_data = glb_m1[(v8_lead + 48)];
                double v440_data = s0[81];
                double v442_data = ir0[5];
                ir0[5] = (v442_data + (v439_data * v440_data));
                double v450_data = glb_m1[(v8_lead + 48)];
                double v451_data = s0[98];
                double v453_data = ir0[6];
                ir0[6] = (v453_data + (v450_data * v451_data));
                double v461_data = glb_m1[(v8_lead + 48)];
                double v462_data = s0[115];
                double v464_data = ir0[7];
                ir0[7] = (v464_data + (v461_data * v462_data));
              }
              if (v8_lead < 12) {
                double v476_data = glb_m1[(v8_lead + 60)];
                double v477_data = s0[5];
                double v479_data = ir0[0];
                ir0[0] = (v479_data + (v476_data * v477_data));
                double v487_data = glb_m1[(v8_lead + 60)];
                double v488_data = s0[20];
                double v490_data = ir0[1];
                ir0[1] = (v490_data + (v487_data * v488_data));
                double v498_data = glb_m1[(v8_lead + 60)];
                double v499_data = s0[39];
                double v501_data = ir0[2];
                ir0[2] = (v501_data + (v498_data * v499_data));
                double v509_data = glb_m1[(v8_lead + 60)];
                double v510_data = s0[54];
                double v512_data = ir0[3];
                ir0[3] = (v512_data + (v509_data * v510_data));
                double v520_data = glb_m1[(v8_lead + 60)];
                double v521_data = s0[65];
                double v523_data = ir0[4];
                ir0[4] = (v523_data + (v520_data * v521_data));
                double v531_data = glb_m1[(v8_lead + 60)];
                double v532_data = s0[80];
                double v534_data = ir0[5];
                ir0[5] = (v534_data + (v531_data * v532_data));
                double v542_data = glb_m1[(v8_lead + 60)];
                double v543_data = s0[99];
                double v545_data = ir0[6];
                ir0[6] = (v545_data + (v542_data * v543_data));
                double v553_data = glb_m1[(v8_lead + 60)];
                double v554_data = s0[114];
                double v556_data = ir0[7];
                ir0[7] = (v556_data + (v553_data * v554_data));
              }
              if (v8_lead < 12) {
                double v568_data = glb_m1[(v8_lead + 72)];
                double v569_data = s0[6];
                double v571_data = ir0[0];
                ir0[0] = (v571_data + (v568_data * v569_data));
                double v579_data = glb_m1[(v8_lead + 72)];
                double v580_data = s0[23];
                double v582_data = ir0[1];
                ir0[1] = (v582_data + (v579_data * v580_data));
                double v590_data = glb_m1[(v8_lead + 72)];
                double v591_data = s0[36];
                double v593_data = ir0[2];
                ir0[2] = (v593_data + (v590_data * v591_data));
                double v601_data = glb_m1[(v8_lead + 72)];
                double v602_data = s0[53];
                double v604_data = ir0[3];
                ir0[3] = (v604_data + (v601_data * v602_data));
                double v612_data = glb_m1[(v8_lead + 72)];
                double v613_data = s0[66];
                double v615_data = ir0[4];
                ir0[4] = (v615_data + (v612_data * v613_data));
                double v623_data = glb_m1[(v8_lead + 72)];
                double v624_data = s0[83];
                double v626_data = ir0[5];
                ir0[5] = (v626_data + (v623_data * v624_data));
                double v634_data = glb_m1[(v8_lead + 72)];
                double v635_data = s0[96];
                double v637_data = ir0[6];
                ir0[6] = (v637_data + (v634_data * v635_data));
                double v645_data = glb_m1[(v8_lead + 72)];
                double v646_data = s0[113];
                double v648_data = ir0[7];
                ir0[7] = (v648_data + (v645_data * v646_data));
              }
              if (v8_lead < 12) {
                double v660_data = glb_m1[(v8_lead + 84)];
                double v661_data = s0[7];
                double v663_data = ir0[0];
                ir0[0] = (v663_data + (v660_data * v661_data));
                double v671_data = glb_m1[(v8_lead + 84)];
                double v672_data = s0[22];
                double v674_data = ir0[1];
                ir0[1] = (v674_data + (v671_data * v672_data));
                double v682_data = glb_m1[(v8_lead + 84)];
                double v683_data = s0[37];
                double v685_data = ir0[2];
                ir0[2] = (v685_data + (v682_data * v683_data));
                double v693_data = glb_m1[(v8_lead + 84)];
                double v694_data = s0[52];
                double v696_data = ir0[3];
                ir0[3] = (v696_data + (v693_data * v694_data));
                double v704_data = glb_m1[(v8_lead + 84)];
                double v705_data = s0[67];
                double v707_data = ir0[4];
                ir0[4] = (v707_data + (v704_data * v705_data));
                double v715_data = glb_m1[(v8_lead + 84)];
                double v716_data = s0[82];
                double v718_data = ir0[5];
                ir0[5] = (v718_data + (v715_data * v716_data));
                double v726_data = glb_m1[(v8_lead + 84)];
                double v727_data = s0[97];
                double v729_data = ir0[6];
                ir0[6] = (v729_data + (v726_data * v727_data));
                double v737_data = glb_m1[(v8_lead + 84)];
                double v738_data = s0[112];
                double v740_data = ir0[7];
                ir0[7] = (v740_data + (v737_data * v738_data));
              }
              if (v8_lead < 12) {
                double v752_data = glb_m1[(v8_lead + 96)];
                double v753_data = s0[8];
                double v755_data = ir0[0];
                ir0[0] = (v755_data + (v752_data * v753_data));
                double v763_data = glb_m1[(v8_lead + 96)];
                double v764_data = s0[25];
                double v766_data = ir0[1];
                ir0[1] = (v766_data + (v763_data * v764_data));
                double v774_data = glb_m1[(v8_lead + 96)];
                double v775_data = s0[42];
                double v777_data = ir0[2];
                ir0[2] = (v777_data + (v774_data * v775_data));
                double v785_data = glb_m1[(v8_lead + 96)];
                double v786_data = s0[59];
                double v788_data = ir0[3];
                ir0[3] = (v788_data + (v785_data * v786_data));
                double v796_data = glb_m1[(v8_lead + 96)];
                double v797_data = s0[76];
                double v799_data = ir0[4];
                ir0[4] = (v799_data + (v796_data * v797_data));
                double v807_data = glb_m1[(v8_lead + 96)];
                double v808_data = s0[93];
                double v810_data = ir0[5];
                ir0[5] = (v810_data + (v807_data * v808_data));
                double v818_data = glb_m1[(v8_lead + 96)];
                double v819_data = s0[110];
                double v821_data = ir0[6];
                ir0[6] = (v821_data + (v818_data * v819_data));
                double v829_data = glb_m1[(v8_lead + 96)];
                double v830_data = s0[127];
                double v832_data = ir0[7];
                ir0[7] = (v832_data + (v829_data * v830_data));
              }
              if (v8_lead < 12) {
                double v844_data = glb_m1[(v8_lead + 108)];
                double v845_data = s0[9];
                double v847_data = ir0[0];
                ir0[0] = (v847_data + (v844_data * v845_data));
                double v855_data = glb_m1[(v8_lead + 108)];
                double v856_data = s0[24];
                double v858_data = ir0[1];
                ir0[1] = (v858_data + (v855_data * v856_data));
                double v866_data = glb_m1[(v8_lead + 108)];
                double v867_data = s0[43];
                double v869_data = ir0[2];
                ir0[2] = (v869_data + (v866_data * v867_data));
                double v877_data = glb_m1[(v8_lead + 108)];
                double v878_data = s0[58];
                double v880_data = ir0[3];
                ir0[3] = (v880_data + (v877_data * v878_data));
                double v888_data = glb_m1[(v8_lead + 108)];
                double v889_data = s0[77];
                double v891_data = ir0[4];
                ir0[4] = (v891_data + (v888_data * v889_data));
                double v899_data = glb_m1[(v8_lead + 108)];
                double v900_data = s0[92];
                double v902_data = ir0[5];
                ir0[5] = (v902_data + (v899_data * v900_data));
                double v910_data = glb_m1[(v8_lead + 108)];
                double v911_data = s0[111];
                double v913_data = ir0[6];
                ir0[6] = (v913_data + (v910_data * v911_data));
                double v921_data = glb_m1[(v8_lead + 108)];
                double v922_data = s0[126];
                double v924_data = ir0[7];
                ir0[7] = (v924_data + (v921_data * v922_data));
              }
              if (v8_lead < 12) {
                double v936_data = glb_m1[(v8_lead + 120)];
                double v937_data = s0[10];
                double v939_data = ir0[0];
                ir0[0] = (v939_data + (v936_data * v937_data));
                double v947_data = glb_m1[(v8_lead + 120)];
                double v948_data = s0[27];
                double v950_data = ir0[1];
                ir0[1] = (v950_data + (v947_data * v948_data));
                double v958_data = glb_m1[(v8_lead + 120)];
                double v959_data = s0[40];
                double v961_data = ir0[2];
                ir0[2] = (v961_data + (v958_data * v959_data));
                double v969_data = glb_m1[(v8_lead + 120)];
                double v970_data = s0[57];
                double v972_data = ir0[3];
                ir0[3] = (v972_data + (v969_data * v970_data));
                double v980_data = glb_m1[(v8_lead + 120)];
                double v981_data = s0[78];
                double v983_data = ir0[4];
                ir0[4] = (v983_data + (v980_data * v981_data));
                double v991_data = glb_m1[(v8_lead + 120)];
                double v992_data = s0[95];
                double v994_data = ir0[5];
                ir0[5] = (v994_data + (v991_data * v992_data));
                double v1002_data = glb_m1[(v8_lead + 120)];
                double v1003_data = s0[108];
                double v1005_data = ir0[6];
                ir0[6] = (v1005_data + (v1002_data * v1003_data));
                double v1013_data = glb_m1[(v8_lead + 120)];
                double v1014_data = s0[125];
                double v1016_data = ir0[7];
                ir0[7] = (v1016_data + (v1013_data * v1014_data));
              }
              if (v8_lead < 12) {
                double v1028_data = glb_m1[(v8_lead + 132)];
                double v1029_data = s0[11];
                double v1031_data = ir0[0];
                ir0[0] = (v1031_data + (v1028_data * v1029_data));
                double v1039_data = glb_m1[(v8_lead + 132)];
                double v1040_data = s0[26];
                double v1042_data = ir0[1];
                ir0[1] = (v1042_data + (v1039_data * v1040_data));
                double v1050_data = glb_m1[(v8_lead + 132)];
                double v1051_data = s0[41];
                double v1053_data = ir0[2];
                ir0[2] = (v1053_data + (v1050_data * v1051_data));
                double v1061_data = glb_m1[(v8_lead + 132)];
                double v1062_data = s0[56];
                double v1064_data = ir0[3];
                ir0[3] = (v1064_data + (v1061_data * v1062_data));
                double v1072_data = glb_m1[(v8_lead + 132)];
                double v1073_data = s0[79];
                double v1075_data = ir0[4];
                ir0[4] = (v1075_data + (v1072_data * v1073_data));
                double v1083_data = glb_m1[(v8_lead + 132)];
                double v1084_data = s0[94];
                double v1086_data = ir0[5];
                ir0[5] = (v1086_data + (v1083_data * v1084_data));
                double v1094_data = glb_m1[(v8_lead + 132)];
                double v1095_data = s0[109];
                double v1097_data = ir0[6];
                ir0[6] = (v1097_data + (v1094_data * v1095_data));
                double v1105_data = glb_m1[(v8_lead + 132)];
                double v1106_data = s0[124];
                double v1108_data = ir0[7];
                ir0[7] = (v1108_data + (v1105_data * v1106_data));
              }
              if (v8_lead < 12) {
                double v1120_data = glb_m1[(v8_lead + 144)];
                double v1121_data = s0[12];
                double v1123_data = ir0[0];
                ir0[0] = (v1123_data + (v1120_data * v1121_data));
                double v1131_data = glb_m1[(v8_lead + 144)];
                double v1132_data = s0[29];
                double v1134_data = ir0[1];
                ir0[1] = (v1134_data + (v1131_data * v1132_data));
                double v1142_data = glb_m1[(v8_lead + 144)];
                double v1143_data = s0[46];
                double v1145_data = ir0[2];
                ir0[2] = (v1145_data + (v1142_data * v1143_data));
                double v1153_data = glb_m1[(v8_lead + 144)];
                double v1154_data = s0[63];
                double v1156_data = ir0[3];
                ir0[3] = (v1156_data + (v1153_data * v1154_data));
                double v1164_data = glb_m1[(v8_lead + 144)];
                double v1165_data = s0[72];
                double v1167_data = ir0[4];
                ir0[4] = (v1167_data + (v1164_data * v1165_data));
                double v1175_data = glb_m1[(v8_lead + 144)];
                double v1176_data = s0[89];
                double v1178_data = ir0[5];
                ir0[5] = (v1178_data + (v1175_data * v1176_data));
                double v1186_data = glb_m1[(v8_lead + 144)];
                double v1187_data = s0[106];
                double v1189_data = ir0[6];
                ir0[6] = (v1189_data + (v1186_data * v1187_data));
                double v1197_data = glb_m1[(v8_lead + 144)];
                double v1198_data = s0[123];
                double v1200_data = ir0[7];
                ir0[7] = (v1200_data + (v1197_data * v1198_data));
              }
              if (v8_lead < 12) {
                double v1212_data = glb_m1[(v8_lead + 156)];
                double v1213_data = s0[13];
                double v1215_data = ir0[0];
                ir0[0] = (v1215_data + (v1212_data * v1213_data));
                double v1223_data = glb_m1[(v8_lead + 156)];
                double v1224_data = s0[28];
                double v1226_data = ir0[1];
                ir0[1] = (v1226_data + (v1223_data * v1224_data));
                double v1234_data = glb_m1[(v8_lead + 156)];
                double v1235_data = s0[47];
                double v1237_data = ir0[2];
                ir0[2] = (v1237_data + (v1234_data * v1235_data));
                double v1245_data = glb_m1[(v8_lead + 156)];
                double v1246_data = s0[62];
                double v1248_data = ir0[3];
                ir0[3] = (v1248_data + (v1245_data * v1246_data));
                double v1256_data = glb_m1[(v8_lead + 156)];
                double v1257_data = s0[73];
                double v1259_data = ir0[4];
                ir0[4] = (v1259_data + (v1256_data * v1257_data));
                double v1267_data = glb_m1[(v8_lead + 156)];
                double v1268_data = s0[88];
                double v1270_data = ir0[5];
                ir0[5] = (v1270_data + (v1267_data * v1268_data));
                double v1278_data = glb_m1[(v8_lead + 156)];
                double v1279_data = s0[107];
                double v1281_data = ir0[6];
                ir0[6] = (v1281_data + (v1278_data * v1279_data));
                double v1289_data = glb_m1[(v8_lead + 156)];
                double v1290_data = s0[122];
                double v1292_data = ir0[7];
                ir0[7] = (v1292_data + (v1289_data * v1290_data));
              }
              if (v8_lead < 12) {
                double v1304_data = glb_m1[(v8_lead + 168)];
                double v1305_data = s0[14];
                double v1307_data = ir0[0];
                ir0[0] = (v1307_data + (v1304_data * v1305_data));
                double v1315_data = glb_m1[(v8_lead + 168)];
                double v1316_data = s0[31];
                double v1318_data = ir0[1];
                ir0[1] = (v1318_data + (v1315_data * v1316_data));
                double v1326_data = glb_m1[(v8_lead + 168)];
                double v1327_data = s0[44];
                double v1329_data = ir0[2];
                ir0[2] = (v1329_data + (v1326_data * v1327_data));
                double v1337_data = glb_m1[(v8_lead + 168)];
                double v1338_data = s0[61];
                double v1340_data = ir0[3];
                ir0[3] = (v1340_data + (v1337_data * v1338_data));
                double v1348_data = glb_m1[(v8_lead + 168)];
                double v1349_data = s0[74];
                double v1351_data = ir0[4];
                ir0[4] = (v1351_data + (v1348_data * v1349_data));
                double v1359_data = glb_m1[(v8_lead + 168)];
                double v1360_data = s0[91];
                double v1362_data = ir0[5];
                ir0[5] = (v1362_data + (v1359_data * v1360_data));
                double v1370_data = glb_m1[(v8_lead + 168)];
                double v1371_data = s0[104];
                double v1373_data = ir0[6];
                ir0[6] = (v1373_data + (v1370_data * v1371_data));
                double v1381_data = glb_m1[(v8_lead + 168)];
                double v1382_data = s0[121];
                double v1384_data = ir0[7];
                ir0[7] = (v1384_data + (v1381_data * v1382_data));
              }
              if (v8_lead < 12) {
                double v1396_data = glb_m1[(v8_lead + 180)];
                double v1397_data = s0[15];
                double v1399_data = ir0[0];
                ir0[0] = (v1399_data + (v1396_data * v1397_data));
                double v1407_data = glb_m1[(v8_lead + 180)];
                double v1408_data = s0[30];
                double v1410_data = ir0[1];
                ir0[1] = (v1410_data + (v1407_data * v1408_data));
                double v1418_data = glb_m1[(v8_lead + 180)];
                double v1419_data = s0[45];
                double v1421_data = ir0[2];
                ir0[2] = (v1421_data + (v1418_data * v1419_data));
                double v1429_data = glb_m1[(v8_lead + 180)];
                double v1430_data = s0[60];
                double v1432_data = ir0[3];
                ir0[3] = (v1432_data + (v1429_data * v1430_data));
                double v1440_data = glb_m1[(v8_lead + 180)];
                double v1441_data = s0[75];
                double v1443_data = ir0[4];
                ir0[4] = (v1443_data + (v1440_data * v1441_data));
                double v1451_data = glb_m1[(v8_lead + 180)];
                double v1452_data = s0[90];
                double v1454_data = ir0[5];
                ir0[5] = (v1454_data + (v1451_data * v1452_data));
                double v1462_data = glb_m1[(v8_lead + 180)];
                double v1463_data = s0[105];
                double v1465_data = ir0[6];
                ir0[6] = (v1465_data + (v1462_data * v1463_data));
                double v1473_data = glb_m1[(v8_lead + 180)];
                double v1474_data = s0[120];
                double v1476_data = ir0[7];
                ir0[7] = (v1476_data + (v1473_data * v1474_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1482_n1 = 0; v1482_n1 < 8; ++v1482_n1) {
                  double v1484_data = ir0[v1482_n1];
                  double v1492_data = glb_m0[(v8_lead + (v1482_n1 * 12))];
                  r0[v1482_n1] = (v1492_data + v1484_data);
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1499_i1 = 0; v1499_i1 < 8; ++v1499_i1) {
                  double v1501_data = r0[v1499_i1];
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


// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08703cce1d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08703cce1d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1536, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×32(12×6) {0..12}×{0..6} strided
        // m1 32×32(6×6) {0..6}×{0..6} strided
        // m2 32×32(12×6) {0..12}×{0..6} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
        // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[96 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[80];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              *(sycl::vec<float, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<float, 2>*)&glb_m1[0 + 0 + 2 * item.get_local_id(0) + 0];
              if (item.get_local_id(0) < 4) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 32] = glb_m1[0 + 0 + 1 * item.get_local_id(0) + 32];
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r0[6]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                int32_t v15_a = v8_lead + 0;
                float v22_data = glb_m0[v8_lead];
                float v23_data = s0[0];
                float v25_data = r0[0];
                r0[0] = (v25_data + (v22_data * v23_data));
                int32_t v32_a = v8_lead + 0;
                float v39_data = glb_m0[v8_lead];
                float v40_data = s0[6];
                float v42_data = r0[1];
                r0[1] = (v42_data + (v39_data * v40_data));
                int32_t v49_a = v8_lead + 0;
                float v56_data = glb_m0[v8_lead];
                float v57_data = s0[12];
                float v59_data = r0[2];
                r0[2] = (v59_data + (v56_data * v57_data));
                int32_t v66_a = v8_lead + 0;
                float v73_data = glb_m0[v8_lead];
                float v74_data = s0[18];
                float v76_data = r0[3];
                r0[3] = (v76_data + (v73_data * v74_data));
                int32_t v83_a = v8_lead + 0;
                float v90_data = glb_m0[v8_lead];
                float v91_data = s0[24];
                float v93_data = r0[4];
                r0[4] = (v93_data + (v90_data * v91_data));
                int32_t v100_a = v8_lead + 0;
                float v107_data = glb_m0[v8_lead];
                float v108_data = s0[30];
                float v110_data = r0[5];
                r0[5] = (v110_data + (v107_data * v108_data));
              }
              if (v8_lead < 12) {
                int32_t v121_a = v8_lead + 12;
                float v128_data = glb_m0[(v8_lead + 12)];
                float v129_data = s0[1];
                float v131_data = r0[0];
                r0[0] = (v131_data + (v128_data * v129_data));
                int32_t v138_a = v8_lead + 12;
                float v145_data = glb_m0[(v8_lead + 12)];
                float v146_data = s0[7];
                float v148_data = r0[1];
                r0[1] = (v148_data + (v145_data * v146_data));
                int32_t v155_a = v8_lead + 12;
                float v162_data = glb_m0[(v8_lead + 12)];
                float v163_data = s0[13];
                float v165_data = r0[2];
                r0[2] = (v165_data + (v162_data * v163_data));
                int32_t v172_a = v8_lead + 12;
                float v179_data = glb_m0[(v8_lead + 12)];
                float v180_data = s0[19];
                float v182_data = r0[3];
                r0[3] = (v182_data + (v179_data * v180_data));
                int32_t v189_a = v8_lead + 12;
                float v196_data = glb_m0[(v8_lead + 12)];
                float v197_data = s0[25];
                float v199_data = r0[4];
                r0[4] = (v199_data + (v196_data * v197_data));
                int32_t v206_a = v8_lead + 12;
                float v213_data = glb_m0[(v8_lead + 12)];
                float v214_data = s0[31];
                float v216_data = r0[5];
                r0[5] = (v216_data + (v213_data * v214_data));
              }
              if (v8_lead < 12) {
                int32_t v227_a = v8_lead + 24;
                float v234_data = glb_m0[(v8_lead + 24)];
                float v235_data = s0[2];
                float v237_data = r0[0];
                r0[0] = (v237_data + (v234_data * v235_data));
                int32_t v244_a = v8_lead + 24;
                float v251_data = glb_m0[(v8_lead + 24)];
                float v252_data = s0[8];
                float v254_data = r0[1];
                r0[1] = (v254_data + (v251_data * v252_data));
                int32_t v261_a = v8_lead + 24;
                float v268_data = glb_m0[(v8_lead + 24)];
                float v269_data = s0[14];
                float v271_data = r0[2];
                r0[2] = (v271_data + (v268_data * v269_data));
                int32_t v278_a = v8_lead + 24;
                float v285_data = glb_m0[(v8_lead + 24)];
                float v286_data = s0[20];
                float v288_data = r0[3];
                r0[3] = (v288_data + (v285_data * v286_data));
                int32_t v295_a = v8_lead + 24;
                float v302_data = glb_m0[(v8_lead + 24)];
                float v303_data = s0[26];
                float v305_data = r0[4];
                r0[4] = (v305_data + (v302_data * v303_data));
                int32_t v312_a = v8_lead + 24;
                float v319_data = glb_m0[(v8_lead + 24)];
                float v320_data = s0[32];
                float v322_data = r0[5];
                r0[5] = (v322_data + (v319_data * v320_data));
              }
              if (v8_lead < 12) {
                int32_t v333_a = v8_lead + 36;
                float v340_data = glb_m0[(v8_lead + 36)];
                float v341_data = s0[3];
                float v343_data = r0[0];
                r0[0] = (v343_data + (v340_data * v341_data));
                int32_t v350_a = v8_lead + 36;
                float v357_data = glb_m0[(v8_lead + 36)];
                float v358_data = s0[9];
                float v360_data = r0[1];
                r0[1] = (v360_data + (v357_data * v358_data));
                int32_t v367_a = v8_lead + 36;
                float v374_data = glb_m0[(v8_lead + 36)];
                float v375_data = s0[15];
                float v377_data = r0[2];
                r0[2] = (v377_data + (v374_data * v375_data));
                int32_t v384_a = v8_lead + 36;
                float v391_data = glb_m0[(v8_lead + 36)];
                float v392_data = s0[21];
                float v394_data = r0[3];
                r0[3] = (v394_data + (v391_data * v392_data));
                int32_t v401_a = v8_lead + 36;
                float v408_data = glb_m0[(v8_lead + 36)];
                float v409_data = s0[27];
                float v411_data = r0[4];
                r0[4] = (v411_data + (v408_data * v409_data));
                int32_t v418_a = v8_lead + 36;
                float v425_data = glb_m0[(v8_lead + 36)];
                float v426_data = s0[33];
                float v428_data = r0[5];
                r0[5] = (v428_data + (v425_data * v426_data));
              }
              if (v8_lead < 12) {
                int32_t v439_a = v8_lead + 48;
                float v446_data = glb_m0[(v8_lead + 48)];
                float v447_data = s0[4];
                float v449_data = r0[0];
                r0[0] = (v449_data + (v446_data * v447_data));
                int32_t v456_a = v8_lead + 48;
                float v463_data = glb_m0[(v8_lead + 48)];
                float v464_data = s0[10];
                float v466_data = r0[1];
                r0[1] = (v466_data + (v463_data * v464_data));
                int32_t v473_a = v8_lead + 48;
                float v480_data = glb_m0[(v8_lead + 48)];
                float v481_data = s0[16];
                float v483_data = r0[2];
                r0[2] = (v483_data + (v480_data * v481_data));
                int32_t v490_a = v8_lead + 48;
                float v497_data = glb_m0[(v8_lead + 48)];
                float v498_data = s0[22];
                float v500_data = r0[3];
                r0[3] = (v500_data + (v497_data * v498_data));
                int32_t v507_a = v8_lead + 48;
                float v514_data = glb_m0[(v8_lead + 48)];
                float v515_data = s0[28];
                float v517_data = r0[4];
                r0[4] = (v517_data + (v514_data * v515_data));
                int32_t v524_a = v8_lead + 48;
                float v531_data = glb_m0[(v8_lead + 48)];
                float v532_data = s0[34];
                float v534_data = r0[5];
                r0[5] = (v534_data + (v531_data * v532_data));
              }
              if (v8_lead < 12) {
                int32_t v545_a = v8_lead + 60;
                float v552_data = glb_m0[(v8_lead + 60)];
                float v553_data = s0[5];
                float v555_data = r0[0];
                r0[0] = (v555_data + (v552_data * v553_data));
                int32_t v562_a = v8_lead + 60;
                float v569_data = glb_m0[(v8_lead + 60)];
                float v570_data = s0[11];
                float v572_data = r0[1];
                r0[1] = (v572_data + (v569_data * v570_data));
                int32_t v579_a = v8_lead + 60;
                float v586_data = glb_m0[(v8_lead + 60)];
                float v587_data = s0[17];
                float v589_data = r0[2];
                r0[2] = (v589_data + (v586_data * v587_data));
                int32_t v596_a = v8_lead + 60;
                float v603_data = glb_m0[(v8_lead + 60)];
                float v604_data = s0[23];
                float v606_data = r0[3];
                r0[3] = (v606_data + (v603_data * v604_data));
                int32_t v613_a = v8_lead + 60;
                float v620_data = glb_m0[(v8_lead + 60)];
                float v621_data = s0[29];
                float v623_data = r0[4];
                r0[4] = (v623_data + (v620_data * v621_data));
                int32_t v630_a = v8_lead + 60;
                float v637_data = glb_m0[(v8_lead + 60)];
                float v638_data = s0[35];
                float v640_data = r0[5];
                r0[5] = (v640_data + (v637_data * v638_data));
              }
              sycl::group_barrier(item.get_sub_group());
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v647_i1 = 0; v647_i1 < 6; ++v647_i1) {
                  int32_t v648_a = 0 + v647_i1;
                  float v650_data = r0[v647_i1];
                  int32_t v657_a = v8_lead + (v647_i1 * 12);
                  s1[v657_a] = v650_data;
                }
              }
              float r1[6]{};
              sycl::group_barrier(item.get_sub_group());
              // r1 = +(glb_m3 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir1[6]{};
              if (v8_lead < 12) {
                int32_t v669_a = v8_lead + 0;
                float v676_data = glb_m3[v8_lead];
                float v677_data = s1[0];
                float v679_data = ir1[0];
                ir1[0] = (v679_data + (v676_data * v677_data));
                int32_t v686_a = v8_lead + 0;
                float v693_data = glb_m3[v8_lead];
                float v694_data = s1[12];
                float v696_data = ir1[1];
                ir1[1] = (v696_data + (v693_data * v694_data));
                int32_t v703_a = v8_lead + 0;
                float v710_data = glb_m3[v8_lead];
                float v711_data = s1[24];
                float v713_data = ir1[2];
                ir1[2] = (v713_data + (v710_data * v711_data));
                int32_t v720_a = v8_lead + 0;
                float v727_data = glb_m3[v8_lead];
                float v728_data = s1[36];
                float v730_data = ir1[3];
                ir1[3] = (v730_data + (v727_data * v728_data));
                int32_t v737_a = v8_lead + 0;
                float v744_data = glb_m3[v8_lead];
                float v745_data = s1[48];
                float v747_data = ir1[4];
                ir1[4] = (v747_data + (v744_data * v745_data));
                int32_t v754_a = v8_lead + 0;
                float v761_data = glb_m3[v8_lead];
                float v762_data = s1[60];
                float v764_data = ir1[5];
                ir1[5] = (v764_data + (v761_data * v762_data));
              }
              if (v8_lead < 12) {
                int32_t v775_a = v8_lead + 12;
                float v782_data = glb_m3[(v8_lead + 12)];
                float v783_data = s1[1];
                float v785_data = ir1[0];
                ir1[0] = (v785_data + (v782_data * v783_data));
                int32_t v792_a = v8_lead + 12;
                float v799_data = glb_m3[(v8_lead + 12)];
                float v800_data = s1[13];
                float v802_data = ir1[1];
                ir1[1] = (v802_data + (v799_data * v800_data));
                int32_t v809_a = v8_lead + 12;
                float v816_data = glb_m3[(v8_lead + 12)];
                float v817_data = s1[25];
                float v819_data = ir1[2];
                ir1[2] = (v819_data + (v816_data * v817_data));
                int32_t v826_a = v8_lead + 12;
                float v833_data = glb_m3[(v8_lead + 12)];
                float v834_data = s1[37];
                float v836_data = ir1[3];
                ir1[3] = (v836_data + (v833_data * v834_data));
                int32_t v843_a = v8_lead + 12;
                float v850_data = glb_m3[(v8_lead + 12)];
                float v851_data = s1[49];
                float v853_data = ir1[4];
                ir1[4] = (v853_data + (v850_data * v851_data));
                int32_t v860_a = v8_lead + 12;
                float v867_data = glb_m3[(v8_lead + 12)];
                float v868_data = s1[61];
                float v870_data = ir1[5];
                ir1[5] = (v870_data + (v867_data * v868_data));
              }
              if (v8_lead < 12) {
                int32_t v881_a = v8_lead + 24;
                float v888_data = glb_m3[(v8_lead + 24)];
                float v889_data = s1[2];
                float v891_data = ir1[0];
                ir1[0] = (v891_data + (v888_data * v889_data));
                int32_t v898_a = v8_lead + 24;
                float v905_data = glb_m3[(v8_lead + 24)];
                float v906_data = s1[14];
                float v908_data = ir1[1];
                ir1[1] = (v908_data + (v905_data * v906_data));
                int32_t v915_a = v8_lead + 24;
                float v922_data = glb_m3[(v8_lead + 24)];
                float v923_data = s1[26];
                float v925_data = ir1[2];
                ir1[2] = (v925_data + (v922_data * v923_data));
                int32_t v932_a = v8_lead + 24;
                float v939_data = glb_m3[(v8_lead + 24)];
                float v940_data = s1[38];
                float v942_data = ir1[3];
                ir1[3] = (v942_data + (v939_data * v940_data));
                int32_t v949_a = v8_lead + 24;
                float v956_data = glb_m3[(v8_lead + 24)];
                float v957_data = s1[50];
                float v959_data = ir1[4];
                ir1[4] = (v959_data + (v956_data * v957_data));
                int32_t v966_a = v8_lead + 24;
                float v973_data = glb_m3[(v8_lead + 24)];
                float v974_data = s1[62];
                float v976_data = ir1[5];
                ir1[5] = (v976_data + (v973_data * v974_data));
              }
              if (v8_lead < 12) {
                int32_t v987_a = v8_lead + 36;
                float v994_data = glb_m3[(v8_lead + 36)];
                float v995_data = s1[3];
                float v997_data = ir1[0];
                ir1[0] = (v997_data + (v994_data * v995_data));
                int32_t v1004_a = v8_lead + 36;
                float v1011_data = glb_m3[(v8_lead + 36)];
                float v1012_data = s1[15];
                float v1014_data = ir1[1];
                ir1[1] = (v1014_data + (v1011_data * v1012_data));
                int32_t v1021_a = v8_lead + 36;
                float v1028_data = glb_m3[(v8_lead + 36)];
                float v1029_data = s1[27];
                float v1031_data = ir1[2];
                ir1[2] = (v1031_data + (v1028_data * v1029_data));
                int32_t v1038_a = v8_lead + 36;
                float v1045_data = glb_m3[(v8_lead + 36)];
                float v1046_data = s1[39];
                float v1048_data = ir1[3];
                ir1[3] = (v1048_data + (v1045_data * v1046_data));
                int32_t v1055_a = v8_lead + 36;
                float v1062_data = glb_m3[(v8_lead + 36)];
                float v1063_data = s1[51];
                float v1065_data = ir1[4];
                ir1[4] = (v1065_data + (v1062_data * v1063_data));
                int32_t v1072_a = v8_lead + 36;
                float v1079_data = glb_m3[(v8_lead + 36)];
                float v1080_data = s1[63];
                float v1082_data = ir1[5];
                ir1[5] = (v1082_data + (v1079_data * v1080_data));
              }
              if (v8_lead < 12) {
                int32_t v1093_a = v8_lead + 48;
                float v1100_data = glb_m3[(v8_lead + 48)];
                float v1101_data = s1[4];
                float v1103_data = ir1[0];
                ir1[0] = (v1103_data + (v1100_data * v1101_data));
                int32_t v1110_a = v8_lead + 48;
                float v1117_data = glb_m3[(v8_lead + 48)];
                float v1118_data = s1[16];
                float v1120_data = ir1[1];
                ir1[1] = (v1120_data + (v1117_data * v1118_data));
                int32_t v1127_a = v8_lead + 48;
                float v1134_data = glb_m3[(v8_lead + 48)];
                float v1135_data = s1[28];
                float v1137_data = ir1[2];
                ir1[2] = (v1137_data + (v1134_data * v1135_data));
                int32_t v1144_a = v8_lead + 48;
                float v1151_data = glb_m3[(v8_lead + 48)];
                float v1152_data = s1[40];
                float v1154_data = ir1[3];
                ir1[3] = (v1154_data + (v1151_data * v1152_data));
                int32_t v1161_a = v8_lead + 48;
                float v1168_data = glb_m3[(v8_lead + 48)];
                float v1169_data = s1[52];
                float v1171_data = ir1[4];
                ir1[4] = (v1171_data + (v1168_data * v1169_data));
                int32_t v1178_a = v8_lead + 48;
                float v1185_data = glb_m3[(v8_lead + 48)];
                float v1186_data = s1[64];
                float v1188_data = ir1[5];
                ir1[5] = (v1188_data + (v1185_data * v1186_data));
              }
              if (v8_lead < 12) {
                int32_t v1199_a = v8_lead + 60;
                float v1206_data = glb_m3[(v8_lead + 60)];
                float v1207_data = s1[5];
                float v1209_data = ir1[0];
                ir1[0] = (v1209_data + (v1206_data * v1207_data));
                int32_t v1216_a = v8_lead + 60;
                float v1223_data = glb_m3[(v8_lead + 60)];
                float v1224_data = s1[17];
                float v1226_data = ir1[1];
                ir1[1] = (v1226_data + (v1223_data * v1224_data));
                int32_t v1233_a = v8_lead + 60;
                float v1240_data = glb_m3[(v8_lead + 60)];
                float v1241_data = s1[29];
                float v1243_data = ir1[2];
                ir1[2] = (v1243_data + (v1240_data * v1241_data));
                int32_t v1250_a = v8_lead + 60;
                float v1257_data = glb_m3[(v8_lead + 60)];
                float v1258_data = s1[41];
                float v1260_data = ir1[3];
                ir1[3] = (v1260_data + (v1257_data * v1258_data));
                int32_t v1267_a = v8_lead + 60;
                float v1274_data = glb_m3[(v8_lead + 60)];
                float v1275_data = s1[53];
                float v1277_data = ir1[4];
                ir1[4] = (v1277_data + (v1274_data * v1275_data));
                int32_t v1284_a = v8_lead + 60;
                float v1291_data = glb_m3[(v8_lead + 60)];
                float v1292_data = s1[65];
                float v1294_data = ir1[5];
                ir1[5] = (v1294_data + (v1291_data * v1292_data));
              }
              if (v8_lead < 12) {
                int32_t v1305_a = v8_lead + 72;
                float v1312_data = glb_m3[(v8_lead + 72)];
                float v1313_data = s1[6];
                float v1315_data = ir1[0];
                ir1[0] = (v1315_data + (v1312_data * v1313_data));
                int32_t v1322_a = v8_lead + 72;
                float v1329_data = glb_m3[(v8_lead + 72)];
                float v1330_data = s1[18];
                float v1332_data = ir1[1];
                ir1[1] = (v1332_data + (v1329_data * v1330_data));
                int32_t v1339_a = v8_lead + 72;
                float v1346_data = glb_m3[(v8_lead + 72)];
                float v1347_data = s1[30];
                float v1349_data = ir1[2];
                ir1[2] = (v1349_data + (v1346_data * v1347_data));
                int32_t v1356_a = v8_lead + 72;
                float v1363_data = glb_m3[(v8_lead + 72)];
                float v1364_data = s1[42];
                float v1366_data = ir1[3];
                ir1[3] = (v1366_data + (v1363_data * v1364_data));
                int32_t v1373_a = v8_lead + 72;
                float v1380_data = glb_m3[(v8_lead + 72)];
                float v1381_data = s1[54];
                float v1383_data = ir1[4];
                ir1[4] = (v1383_data + (v1380_data * v1381_data));
                int32_t v1390_a = v8_lead + 72;
                float v1397_data = glb_m3[(v8_lead + 72)];
                float v1398_data = s1[66];
                float v1400_data = ir1[5];
                ir1[5] = (v1400_data + (v1397_data * v1398_data));
              }
              if (v8_lead < 12) {
                int32_t v1411_a = v8_lead + 84;
                float v1418_data = glb_m3[(v8_lead + 84)];
                float v1419_data = s1[7];
                float v1421_data = ir1[0];
                ir1[0] = (v1421_data + (v1418_data * v1419_data));
                int32_t v1428_a = v8_lead + 84;
                float v1435_data = glb_m3[(v8_lead + 84)];
                float v1436_data = s1[19];
                float v1438_data = ir1[1];
                ir1[1] = (v1438_data + (v1435_data * v1436_data));
                int32_t v1445_a = v8_lead + 84;
                float v1452_data = glb_m3[(v8_lead + 84)];
                float v1453_data = s1[31];
                float v1455_data = ir1[2];
                ir1[2] = (v1455_data + (v1452_data * v1453_data));
                int32_t v1462_a = v8_lead + 84;
                float v1469_data = glb_m3[(v8_lead + 84)];
                float v1470_data = s1[43];
                float v1472_data = ir1[3];
                ir1[3] = (v1472_data + (v1469_data * v1470_data));
                int32_t v1479_a = v8_lead + 84;
                float v1486_data = glb_m3[(v8_lead + 84)];
                float v1487_data = s1[55];
                float v1489_data = ir1[4];
                ir1[4] = (v1489_data + (v1486_data * v1487_data));
                int32_t v1496_a = v8_lead + 84;
                float v1503_data = glb_m3[(v8_lead + 84)];
                float v1504_data = s1[67];
                float v1506_data = ir1[5];
                ir1[5] = (v1506_data + (v1503_data * v1504_data));
              }
              if (v8_lead < 12) {
                int32_t v1517_a = v8_lead + 96;
                float v1524_data = glb_m3[(v8_lead + 96)];
                float v1525_data = s1[8];
                float v1527_data = ir1[0];
                ir1[0] = (v1527_data + (v1524_data * v1525_data));
                int32_t v1534_a = v8_lead + 96;
                float v1541_data = glb_m3[(v8_lead + 96)];
                float v1542_data = s1[20];
                float v1544_data = ir1[1];
                ir1[1] = (v1544_data + (v1541_data * v1542_data));
                int32_t v1551_a = v8_lead + 96;
                float v1558_data = glb_m3[(v8_lead + 96)];
                float v1559_data = s1[32];
                float v1561_data = ir1[2];
                ir1[2] = (v1561_data + (v1558_data * v1559_data));
                int32_t v1568_a = v8_lead + 96;
                float v1575_data = glb_m3[(v8_lead + 96)];
                float v1576_data = s1[44];
                float v1578_data = ir1[3];
                ir1[3] = (v1578_data + (v1575_data * v1576_data));
                int32_t v1585_a = v8_lead + 96;
                float v1592_data = glb_m3[(v8_lead + 96)];
                float v1593_data = s1[56];
                float v1595_data = ir1[4];
                ir1[4] = (v1595_data + (v1592_data * v1593_data));
                int32_t v1602_a = v8_lead + 96;
                float v1609_data = glb_m3[(v8_lead + 96)];
                float v1610_data = s1[68];
                float v1612_data = ir1[5];
                ir1[5] = (v1612_data + (v1609_data * v1610_data));
              }
              if (v8_lead < 12) {
                int32_t v1623_a = v8_lead + 108;
                float v1630_data = glb_m3[(v8_lead + 108)];
                float v1631_data = s1[9];
                float v1633_data = ir1[0];
                ir1[0] = (v1633_data + (v1630_data * v1631_data));
                int32_t v1640_a = v8_lead + 108;
                float v1647_data = glb_m3[(v8_lead + 108)];
                float v1648_data = s1[21];
                float v1650_data = ir1[1];
                ir1[1] = (v1650_data + (v1647_data * v1648_data));
                int32_t v1657_a = v8_lead + 108;
                float v1664_data = glb_m3[(v8_lead + 108)];
                float v1665_data = s1[33];
                float v1667_data = ir1[2];
                ir1[2] = (v1667_data + (v1664_data * v1665_data));
                int32_t v1674_a = v8_lead + 108;
                float v1681_data = glb_m3[(v8_lead + 108)];
                float v1682_data = s1[45];
                float v1684_data = ir1[3];
                ir1[3] = (v1684_data + (v1681_data * v1682_data));
                int32_t v1691_a = v8_lead + 108;
                float v1698_data = glb_m3[(v8_lead + 108)];
                float v1699_data = s1[57];
                float v1701_data = ir1[4];
                ir1[4] = (v1701_data + (v1698_data * v1699_data));
                int32_t v1708_a = v8_lead + 108;
                float v1715_data = glb_m3[(v8_lead + 108)];
                float v1716_data = s1[69];
                float v1718_data = ir1[5];
                ir1[5] = (v1718_data + (v1715_data * v1716_data));
              }
              if (v8_lead < 12) {
                int32_t v1729_a = v8_lead + 120;
                float v1736_data = glb_m3[(v8_lead + 120)];
                float v1737_data = s1[10];
                float v1739_data = ir1[0];
                ir1[0] = (v1739_data + (v1736_data * v1737_data));
                int32_t v1746_a = v8_lead + 120;
                float v1753_data = glb_m3[(v8_lead + 120)];
                float v1754_data = s1[22];
                float v1756_data = ir1[1];
                ir1[1] = (v1756_data + (v1753_data * v1754_data));
                int32_t v1763_a = v8_lead + 120;
                float v1770_data = glb_m3[(v8_lead + 120)];
                float v1771_data = s1[34];
                float v1773_data = ir1[2];
                ir1[2] = (v1773_data + (v1770_data * v1771_data));
                int32_t v1780_a = v8_lead + 120;
                float v1787_data = glb_m3[(v8_lead + 120)];
                float v1788_data = s1[46];
                float v1790_data = ir1[3];
                ir1[3] = (v1790_data + (v1787_data * v1788_data));
                int32_t v1797_a = v8_lead + 120;
                float v1804_data = glb_m3[(v8_lead + 120)];
                float v1805_data = s1[58];
                float v1807_data = ir1[4];
                ir1[4] = (v1807_data + (v1804_data * v1805_data));
                int32_t v1814_a = v8_lead + 120;
                float v1821_data = glb_m3[(v8_lead + 120)];
                float v1822_data = s1[70];
                float v1824_data = ir1[5];
                ir1[5] = (v1824_data + (v1821_data * v1822_data));
              }
              if (v8_lead < 12) {
                int32_t v1835_a = v8_lead + 132;
                float v1842_data = glb_m3[(v8_lead + 132)];
                float v1843_data = s1[11];
                float v1845_data = ir1[0];
                ir1[0] = (v1845_data + (v1842_data * v1843_data));
                int32_t v1852_a = v8_lead + 132;
                float v1859_data = glb_m3[(v8_lead + 132)];
                float v1860_data = s1[23];
                float v1862_data = ir1[1];
                ir1[1] = (v1862_data + (v1859_data * v1860_data));
                int32_t v1869_a = v8_lead + 132;
                float v1876_data = glb_m3[(v8_lead + 132)];
                float v1877_data = s1[35];
                float v1879_data = ir1[2];
                ir1[2] = (v1879_data + (v1876_data * v1877_data));
                int32_t v1886_a = v8_lead + 132;
                float v1893_data = glb_m3[(v8_lead + 132)];
                float v1894_data = s1[47];
                float v1896_data = ir1[3];
                ir1[3] = (v1896_data + (v1893_data * v1894_data));
                int32_t v1903_a = v8_lead + 132;
                float v1910_data = glb_m3[(v8_lead + 132)];
                float v1911_data = s1[59];
                float v1913_data = ir1[4];
                ir1[4] = (v1913_data + (v1910_data * v1911_data));
                int32_t v1920_a = v8_lead + 132;
                float v1927_data = glb_m3[(v8_lead + 132)];
                float v1928_data = s1[71];
                float v1930_data = ir1[5];
                ir1[5] = (v1930_data + (v1927_data * v1928_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1936_n1 = 0; v1936_n1 < 6; ++v1936_n1) {
                  int32_t v1937_a = 0 + v1936_n1;
                  float v1939_data = ir1[v1936_n1];
                  int32_t v1940_a = 0 + v1936_n1;
                  r1[v1936_n1] = v1939_data;
                }
              }
              // glb_m2 = store{r>g}(r1);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1946_i1 = 0; v1946_i1 < 6; ++v1946_i1) {
                  int32_t v1947_a = 0 + v1946_i1;
                  float v1949_data = r1[v1946_i1];
                  int32_t v1956_a = v8_lead + (v1946_i1 * 12);
                  glb_m2[v1956_a] = v1949_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


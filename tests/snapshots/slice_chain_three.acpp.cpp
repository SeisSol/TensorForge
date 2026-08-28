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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
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
                float v16_data = glb_m0[v8_lead];
                float v17_data = s0[0];
                float v19_data = r0[0];
                r0[0] = (v19_data + (v16_data * v17_data));
                float v27_data = glb_m0[v8_lead];
                float v28_data = s0[6];
                float v30_data = r0[1];
                r0[1] = (v30_data + (v27_data * v28_data));
                float v38_data = glb_m0[v8_lead];
                float v39_data = s0[12];
                float v41_data = r0[2];
                r0[2] = (v41_data + (v38_data * v39_data));
                float v49_data = glb_m0[v8_lead];
                float v50_data = s0[18];
                float v52_data = r0[3];
                r0[3] = (v52_data + (v49_data * v50_data));
                float v60_data = glb_m0[v8_lead];
                float v61_data = s0[24];
                float v63_data = r0[4];
                r0[4] = (v63_data + (v60_data * v61_data));
                float v71_data = glb_m0[v8_lead];
                float v72_data = s0[30];
                float v74_data = r0[5];
                r0[5] = (v74_data + (v71_data * v72_data));
              }
              if (v8_lead < 12) {
                float v86_data = glb_m0[(v8_lead + 12)];
                float v87_data = s0[1];
                float v89_data = r0[0];
                r0[0] = (v89_data + (v86_data * v87_data));
                float v97_data = glb_m0[(v8_lead + 12)];
                float v98_data = s0[7];
                float v100_data = r0[1];
                r0[1] = (v100_data + (v97_data * v98_data));
                float v108_data = glb_m0[(v8_lead + 12)];
                float v109_data = s0[13];
                float v111_data = r0[2];
                r0[2] = (v111_data + (v108_data * v109_data));
                float v119_data = glb_m0[(v8_lead + 12)];
                float v120_data = s0[19];
                float v122_data = r0[3];
                r0[3] = (v122_data + (v119_data * v120_data));
                float v130_data = glb_m0[(v8_lead + 12)];
                float v131_data = s0[25];
                float v133_data = r0[4];
                r0[4] = (v133_data + (v130_data * v131_data));
                float v141_data = glb_m0[(v8_lead + 12)];
                float v142_data = s0[31];
                float v144_data = r0[5];
                r0[5] = (v144_data + (v141_data * v142_data));
              }
              if (v8_lead < 12) {
                float v156_data = glb_m0[(v8_lead + 24)];
                float v157_data = s0[2];
                float v159_data = r0[0];
                r0[0] = (v159_data + (v156_data * v157_data));
                float v167_data = glb_m0[(v8_lead + 24)];
                float v168_data = s0[8];
                float v170_data = r0[1];
                r0[1] = (v170_data + (v167_data * v168_data));
                float v178_data = glb_m0[(v8_lead + 24)];
                float v179_data = s0[14];
                float v181_data = r0[2];
                r0[2] = (v181_data + (v178_data * v179_data));
                float v189_data = glb_m0[(v8_lead + 24)];
                float v190_data = s0[20];
                float v192_data = r0[3];
                r0[3] = (v192_data + (v189_data * v190_data));
                float v200_data = glb_m0[(v8_lead + 24)];
                float v201_data = s0[26];
                float v203_data = r0[4];
                r0[4] = (v203_data + (v200_data * v201_data));
                float v211_data = glb_m0[(v8_lead + 24)];
                float v212_data = s0[32];
                float v214_data = r0[5];
                r0[5] = (v214_data + (v211_data * v212_data));
              }
              if (v8_lead < 12) {
                float v226_data = glb_m0[(v8_lead + 36)];
                float v227_data = s0[3];
                float v229_data = r0[0];
                r0[0] = (v229_data + (v226_data * v227_data));
                float v237_data = glb_m0[(v8_lead + 36)];
                float v238_data = s0[9];
                float v240_data = r0[1];
                r0[1] = (v240_data + (v237_data * v238_data));
                float v248_data = glb_m0[(v8_lead + 36)];
                float v249_data = s0[15];
                float v251_data = r0[2];
                r0[2] = (v251_data + (v248_data * v249_data));
                float v259_data = glb_m0[(v8_lead + 36)];
                float v260_data = s0[21];
                float v262_data = r0[3];
                r0[3] = (v262_data + (v259_data * v260_data));
                float v270_data = glb_m0[(v8_lead + 36)];
                float v271_data = s0[27];
                float v273_data = r0[4];
                r0[4] = (v273_data + (v270_data * v271_data));
                float v281_data = glb_m0[(v8_lead + 36)];
                float v282_data = s0[33];
                float v284_data = r0[5];
                r0[5] = (v284_data + (v281_data * v282_data));
              }
              if (v8_lead < 12) {
                float v296_data = glb_m0[(v8_lead + 48)];
                float v297_data = s0[4];
                float v299_data = r0[0];
                r0[0] = (v299_data + (v296_data * v297_data));
                float v307_data = glb_m0[(v8_lead + 48)];
                float v308_data = s0[10];
                float v310_data = r0[1];
                r0[1] = (v310_data + (v307_data * v308_data));
                float v318_data = glb_m0[(v8_lead + 48)];
                float v319_data = s0[16];
                float v321_data = r0[2];
                r0[2] = (v321_data + (v318_data * v319_data));
                float v329_data = glb_m0[(v8_lead + 48)];
                float v330_data = s0[22];
                float v332_data = r0[3];
                r0[3] = (v332_data + (v329_data * v330_data));
                float v340_data = glb_m0[(v8_lead + 48)];
                float v341_data = s0[28];
                float v343_data = r0[4];
                r0[4] = (v343_data + (v340_data * v341_data));
                float v351_data = glb_m0[(v8_lead + 48)];
                float v352_data = s0[34];
                float v354_data = r0[5];
                r0[5] = (v354_data + (v351_data * v352_data));
              }
              if (v8_lead < 12) {
                float v366_data = glb_m0[(v8_lead + 60)];
                float v367_data = s0[5];
                float v369_data = r0[0];
                r0[0] = (v369_data + (v366_data * v367_data));
                float v377_data = glb_m0[(v8_lead + 60)];
                float v378_data = s0[11];
                float v380_data = r0[1];
                r0[1] = (v380_data + (v377_data * v378_data));
                float v388_data = glb_m0[(v8_lead + 60)];
                float v389_data = s0[17];
                float v391_data = r0[2];
                r0[2] = (v391_data + (v388_data * v389_data));
                float v399_data = glb_m0[(v8_lead + 60)];
                float v400_data = s0[23];
                float v402_data = r0[3];
                r0[3] = (v402_data + (v399_data * v400_data));
                float v410_data = glb_m0[(v8_lead + 60)];
                float v411_data = s0[29];
                float v413_data = r0[4];
                r0[4] = (v413_data + (v410_data * v411_data));
                float v421_data = glb_m0[(v8_lead + 60)];
                float v422_data = s0[35];
                float v424_data = r0[5];
                r0[5] = (v424_data + (v421_data * v422_data));
              }
              sycl::group_barrier(item.get_sub_group());
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v431_i1 = 0; v431_i1 < 6; ++v431_i1) {
                  float v433_data = r0[v431_i1];
                  s1[(v8_lead + (v431_i1 * 12))] = v433_data;
                }
              }
              float r1[6]{};
              sycl::group_barrier(item.get_sub_group());
              // r1 = +(glb_m3 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir1[6]{};
              if (v8_lead < 12) {
                float v453_data = glb_m3[v8_lead];
                float v454_data = s1[0];
                float v456_data = ir1[0];
                ir1[0] = (v456_data + (v453_data * v454_data));
                float v464_data = glb_m3[v8_lead];
                float v465_data = s1[12];
                float v467_data = ir1[1];
                ir1[1] = (v467_data + (v464_data * v465_data));
                float v475_data = glb_m3[v8_lead];
                float v476_data = s1[24];
                float v478_data = ir1[2];
                ir1[2] = (v478_data + (v475_data * v476_data));
                float v486_data = glb_m3[v8_lead];
                float v487_data = s1[36];
                float v489_data = ir1[3];
                ir1[3] = (v489_data + (v486_data * v487_data));
                float v497_data = glb_m3[v8_lead];
                float v498_data = s1[48];
                float v500_data = ir1[4];
                ir1[4] = (v500_data + (v497_data * v498_data));
                float v508_data = glb_m3[v8_lead];
                float v509_data = s1[60];
                float v511_data = ir1[5];
                ir1[5] = (v511_data + (v508_data * v509_data));
              }
              if (v8_lead < 12) {
                float v523_data = glb_m3[(v8_lead + 12)];
                float v524_data = s1[1];
                float v526_data = ir1[0];
                ir1[0] = (v526_data + (v523_data * v524_data));
                float v534_data = glb_m3[(v8_lead + 12)];
                float v535_data = s1[13];
                float v537_data = ir1[1];
                ir1[1] = (v537_data + (v534_data * v535_data));
                float v545_data = glb_m3[(v8_lead + 12)];
                float v546_data = s1[25];
                float v548_data = ir1[2];
                ir1[2] = (v548_data + (v545_data * v546_data));
                float v556_data = glb_m3[(v8_lead + 12)];
                float v557_data = s1[37];
                float v559_data = ir1[3];
                ir1[3] = (v559_data + (v556_data * v557_data));
                float v567_data = glb_m3[(v8_lead + 12)];
                float v568_data = s1[49];
                float v570_data = ir1[4];
                ir1[4] = (v570_data + (v567_data * v568_data));
                float v578_data = glb_m3[(v8_lead + 12)];
                float v579_data = s1[61];
                float v581_data = ir1[5];
                ir1[5] = (v581_data + (v578_data * v579_data));
              }
              if (v8_lead < 12) {
                float v593_data = glb_m3[(v8_lead + 24)];
                float v594_data = s1[2];
                float v596_data = ir1[0];
                ir1[0] = (v596_data + (v593_data * v594_data));
                float v604_data = glb_m3[(v8_lead + 24)];
                float v605_data = s1[14];
                float v607_data = ir1[1];
                ir1[1] = (v607_data + (v604_data * v605_data));
                float v615_data = glb_m3[(v8_lead + 24)];
                float v616_data = s1[26];
                float v618_data = ir1[2];
                ir1[2] = (v618_data + (v615_data * v616_data));
                float v626_data = glb_m3[(v8_lead + 24)];
                float v627_data = s1[38];
                float v629_data = ir1[3];
                ir1[3] = (v629_data + (v626_data * v627_data));
                float v637_data = glb_m3[(v8_lead + 24)];
                float v638_data = s1[50];
                float v640_data = ir1[4];
                ir1[4] = (v640_data + (v637_data * v638_data));
                float v648_data = glb_m3[(v8_lead + 24)];
                float v649_data = s1[62];
                float v651_data = ir1[5];
                ir1[5] = (v651_data + (v648_data * v649_data));
              }
              if (v8_lead < 12) {
                float v663_data = glb_m3[(v8_lead + 36)];
                float v664_data = s1[3];
                float v666_data = ir1[0];
                ir1[0] = (v666_data + (v663_data * v664_data));
                float v674_data = glb_m3[(v8_lead + 36)];
                float v675_data = s1[15];
                float v677_data = ir1[1];
                ir1[1] = (v677_data + (v674_data * v675_data));
                float v685_data = glb_m3[(v8_lead + 36)];
                float v686_data = s1[27];
                float v688_data = ir1[2];
                ir1[2] = (v688_data + (v685_data * v686_data));
                float v696_data = glb_m3[(v8_lead + 36)];
                float v697_data = s1[39];
                float v699_data = ir1[3];
                ir1[3] = (v699_data + (v696_data * v697_data));
                float v707_data = glb_m3[(v8_lead + 36)];
                float v708_data = s1[51];
                float v710_data = ir1[4];
                ir1[4] = (v710_data + (v707_data * v708_data));
                float v718_data = glb_m3[(v8_lead + 36)];
                float v719_data = s1[63];
                float v721_data = ir1[5];
                ir1[5] = (v721_data + (v718_data * v719_data));
              }
              if (v8_lead < 12) {
                float v733_data = glb_m3[(v8_lead + 48)];
                float v734_data = s1[4];
                float v736_data = ir1[0];
                ir1[0] = (v736_data + (v733_data * v734_data));
                float v744_data = glb_m3[(v8_lead + 48)];
                float v745_data = s1[16];
                float v747_data = ir1[1];
                ir1[1] = (v747_data + (v744_data * v745_data));
                float v755_data = glb_m3[(v8_lead + 48)];
                float v756_data = s1[28];
                float v758_data = ir1[2];
                ir1[2] = (v758_data + (v755_data * v756_data));
                float v766_data = glb_m3[(v8_lead + 48)];
                float v767_data = s1[40];
                float v769_data = ir1[3];
                ir1[3] = (v769_data + (v766_data * v767_data));
                float v777_data = glb_m3[(v8_lead + 48)];
                float v778_data = s1[52];
                float v780_data = ir1[4];
                ir1[4] = (v780_data + (v777_data * v778_data));
                float v788_data = glb_m3[(v8_lead + 48)];
                float v789_data = s1[64];
                float v791_data = ir1[5];
                ir1[5] = (v791_data + (v788_data * v789_data));
              }
              if (v8_lead < 12) {
                float v803_data = glb_m3[(v8_lead + 60)];
                float v804_data = s1[5];
                float v806_data = ir1[0];
                ir1[0] = (v806_data + (v803_data * v804_data));
                float v814_data = glb_m3[(v8_lead + 60)];
                float v815_data = s1[17];
                float v817_data = ir1[1];
                ir1[1] = (v817_data + (v814_data * v815_data));
                float v825_data = glb_m3[(v8_lead + 60)];
                float v826_data = s1[29];
                float v828_data = ir1[2];
                ir1[2] = (v828_data + (v825_data * v826_data));
                float v836_data = glb_m3[(v8_lead + 60)];
                float v837_data = s1[41];
                float v839_data = ir1[3];
                ir1[3] = (v839_data + (v836_data * v837_data));
                float v847_data = glb_m3[(v8_lead + 60)];
                float v848_data = s1[53];
                float v850_data = ir1[4];
                ir1[4] = (v850_data + (v847_data * v848_data));
                float v858_data = glb_m3[(v8_lead + 60)];
                float v859_data = s1[65];
                float v861_data = ir1[5];
                ir1[5] = (v861_data + (v858_data * v859_data));
              }
              if (v8_lead < 12) {
                float v873_data = glb_m3[(v8_lead + 72)];
                float v874_data = s1[6];
                float v876_data = ir1[0];
                ir1[0] = (v876_data + (v873_data * v874_data));
                float v884_data = glb_m3[(v8_lead + 72)];
                float v885_data = s1[18];
                float v887_data = ir1[1];
                ir1[1] = (v887_data + (v884_data * v885_data));
                float v895_data = glb_m3[(v8_lead + 72)];
                float v896_data = s1[30];
                float v898_data = ir1[2];
                ir1[2] = (v898_data + (v895_data * v896_data));
                float v906_data = glb_m3[(v8_lead + 72)];
                float v907_data = s1[42];
                float v909_data = ir1[3];
                ir1[3] = (v909_data + (v906_data * v907_data));
                float v917_data = glb_m3[(v8_lead + 72)];
                float v918_data = s1[54];
                float v920_data = ir1[4];
                ir1[4] = (v920_data + (v917_data * v918_data));
                float v928_data = glb_m3[(v8_lead + 72)];
                float v929_data = s1[66];
                float v931_data = ir1[5];
                ir1[5] = (v931_data + (v928_data * v929_data));
              }
              if (v8_lead < 12) {
                float v943_data = glb_m3[(v8_lead + 84)];
                float v944_data = s1[7];
                float v946_data = ir1[0];
                ir1[0] = (v946_data + (v943_data * v944_data));
                float v954_data = glb_m3[(v8_lead + 84)];
                float v955_data = s1[19];
                float v957_data = ir1[1];
                ir1[1] = (v957_data + (v954_data * v955_data));
                float v965_data = glb_m3[(v8_lead + 84)];
                float v966_data = s1[31];
                float v968_data = ir1[2];
                ir1[2] = (v968_data + (v965_data * v966_data));
                float v976_data = glb_m3[(v8_lead + 84)];
                float v977_data = s1[43];
                float v979_data = ir1[3];
                ir1[3] = (v979_data + (v976_data * v977_data));
                float v987_data = glb_m3[(v8_lead + 84)];
                float v988_data = s1[55];
                float v990_data = ir1[4];
                ir1[4] = (v990_data + (v987_data * v988_data));
                float v998_data = glb_m3[(v8_lead + 84)];
                float v999_data = s1[67];
                float v1001_data = ir1[5];
                ir1[5] = (v1001_data + (v998_data * v999_data));
              }
              if (v8_lead < 12) {
                float v1013_data = glb_m3[(v8_lead + 96)];
                float v1014_data = s1[8];
                float v1016_data = ir1[0];
                ir1[0] = (v1016_data + (v1013_data * v1014_data));
                float v1024_data = glb_m3[(v8_lead + 96)];
                float v1025_data = s1[20];
                float v1027_data = ir1[1];
                ir1[1] = (v1027_data + (v1024_data * v1025_data));
                float v1035_data = glb_m3[(v8_lead + 96)];
                float v1036_data = s1[32];
                float v1038_data = ir1[2];
                ir1[2] = (v1038_data + (v1035_data * v1036_data));
                float v1046_data = glb_m3[(v8_lead + 96)];
                float v1047_data = s1[44];
                float v1049_data = ir1[3];
                ir1[3] = (v1049_data + (v1046_data * v1047_data));
                float v1057_data = glb_m3[(v8_lead + 96)];
                float v1058_data = s1[56];
                float v1060_data = ir1[4];
                ir1[4] = (v1060_data + (v1057_data * v1058_data));
                float v1068_data = glb_m3[(v8_lead + 96)];
                float v1069_data = s1[68];
                float v1071_data = ir1[5];
                ir1[5] = (v1071_data + (v1068_data * v1069_data));
              }
              if (v8_lead < 12) {
                float v1083_data = glb_m3[(v8_lead + 108)];
                float v1084_data = s1[9];
                float v1086_data = ir1[0];
                ir1[0] = (v1086_data + (v1083_data * v1084_data));
                float v1094_data = glb_m3[(v8_lead + 108)];
                float v1095_data = s1[21];
                float v1097_data = ir1[1];
                ir1[1] = (v1097_data + (v1094_data * v1095_data));
                float v1105_data = glb_m3[(v8_lead + 108)];
                float v1106_data = s1[33];
                float v1108_data = ir1[2];
                ir1[2] = (v1108_data + (v1105_data * v1106_data));
                float v1116_data = glb_m3[(v8_lead + 108)];
                float v1117_data = s1[45];
                float v1119_data = ir1[3];
                ir1[3] = (v1119_data + (v1116_data * v1117_data));
                float v1127_data = glb_m3[(v8_lead + 108)];
                float v1128_data = s1[57];
                float v1130_data = ir1[4];
                ir1[4] = (v1130_data + (v1127_data * v1128_data));
                float v1138_data = glb_m3[(v8_lead + 108)];
                float v1139_data = s1[69];
                float v1141_data = ir1[5];
                ir1[5] = (v1141_data + (v1138_data * v1139_data));
              }
              if (v8_lead < 12) {
                float v1153_data = glb_m3[(v8_lead + 120)];
                float v1154_data = s1[10];
                float v1156_data = ir1[0];
                ir1[0] = (v1156_data + (v1153_data * v1154_data));
                float v1164_data = glb_m3[(v8_lead + 120)];
                float v1165_data = s1[22];
                float v1167_data = ir1[1];
                ir1[1] = (v1167_data + (v1164_data * v1165_data));
                float v1175_data = glb_m3[(v8_lead + 120)];
                float v1176_data = s1[34];
                float v1178_data = ir1[2];
                ir1[2] = (v1178_data + (v1175_data * v1176_data));
                float v1186_data = glb_m3[(v8_lead + 120)];
                float v1187_data = s1[46];
                float v1189_data = ir1[3];
                ir1[3] = (v1189_data + (v1186_data * v1187_data));
                float v1197_data = glb_m3[(v8_lead + 120)];
                float v1198_data = s1[58];
                float v1200_data = ir1[4];
                ir1[4] = (v1200_data + (v1197_data * v1198_data));
                float v1208_data = glb_m3[(v8_lead + 120)];
                float v1209_data = s1[70];
                float v1211_data = ir1[5];
                ir1[5] = (v1211_data + (v1208_data * v1209_data));
              }
              if (v8_lead < 12) {
                float v1223_data = glb_m3[(v8_lead + 132)];
                float v1224_data = s1[11];
                float v1226_data = ir1[0];
                ir1[0] = (v1226_data + (v1223_data * v1224_data));
                float v1234_data = glb_m3[(v8_lead + 132)];
                float v1235_data = s1[23];
                float v1237_data = ir1[1];
                ir1[1] = (v1237_data + (v1234_data * v1235_data));
                float v1245_data = glb_m3[(v8_lead + 132)];
                float v1246_data = s1[35];
                float v1248_data = ir1[2];
                ir1[2] = (v1248_data + (v1245_data * v1246_data));
                float v1256_data = glb_m3[(v8_lead + 132)];
                float v1257_data = s1[47];
                float v1259_data = ir1[3];
                ir1[3] = (v1259_data + (v1256_data * v1257_data));
                float v1267_data = glb_m3[(v8_lead + 132)];
                float v1268_data = s1[59];
                float v1270_data = ir1[4];
                ir1[4] = (v1270_data + (v1267_data * v1268_data));
                float v1278_data = glb_m3[(v8_lead + 132)];
                float v1279_data = s1[71];
                float v1281_data = ir1[5];
                ir1[5] = (v1281_data + (v1278_data * v1279_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1287_n1 = 0; v1287_n1 < 6; ++v1287_n1) {
                  float v1289_data = ir1[v1287_n1];
                  r1[v1287_n1] = v1289_data;
                }
              }
              // glb_m2 = store{r>g}(r1);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1295_i1 = 0; v1295_i1 < 6; ++v1295_i1) {
                  float v1297_data = r1[v1295_i1];
                  glb_m2[(v8_lead + (v1295_i1 * 12))] = v1297_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


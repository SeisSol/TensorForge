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
                float v28_data = s0[7];
                float v30_data = r0[1];
                r0[1] = (v30_data + (v27_data * v28_data));
                float v38_data = glb_m0[v8_lead];
                float v39_data = s0[15];
                float v41_data = r0[2];
                r0[2] = (v41_data + (v38_data * v39_data));
                float v49_data = glb_m0[v8_lead];
                float v50_data = s0[18];
                float v52_data = r0[3];
                r0[3] = (v52_data + (v49_data * v50_data));
                float v60_data = glb_m0[v8_lead];
                float v61_data = s0[26];
                float v63_data = r0[4];
                r0[4] = (v63_data + (v60_data * v61_data));
                float v71_data = glb_m0[v8_lead];
                float v72_data = s0[29];
                float v74_data = r0[5];
                r0[5] = (v74_data + (v71_data * v72_data));
              }
              if (v8_lead < 12) {
                float v86_data = glb_m0[(v8_lead + 12)];
                float v87_data = s0[1];
                float v89_data = r0[0];
                r0[0] = (v89_data + (v86_data * v87_data));
                float v97_data = glb_m0[(v8_lead + 12)];
                float v98_data = s0[6];
                float v100_data = r0[1];
                r0[1] = (v100_data + (v97_data * v98_data));
                float v108_data = glb_m0[(v8_lead + 12)];
                float v109_data = s0[14];
                float v111_data = r0[2];
                r0[2] = (v111_data + (v108_data * v109_data));
                float v119_data = glb_m0[(v8_lead + 12)];
                float v120_data = s0[19];
                float v122_data = r0[3];
                r0[3] = (v122_data + (v119_data * v120_data));
                float v130_data = glb_m0[(v8_lead + 12)];
                float v131_data = s0[27];
                float v133_data = r0[4];
                r0[4] = (v133_data + (v130_data * v131_data));
                float v141_data = glb_m0[(v8_lead + 12)];
                float v142_data = s0[28];
                float v144_data = r0[5];
                r0[5] = (v144_data + (v141_data * v142_data));
              }
              if (v8_lead < 12) {
                float v156_data = glb_m0[(v8_lead + 24)];
                float v157_data = s0[2];
                float v159_data = r0[0];
                r0[0] = (v159_data + (v156_data * v157_data));
                float v167_data = glb_m0[(v8_lead + 24)];
                float v168_data = s0[10];
                float v170_data = r0[1];
                r0[1] = (v170_data + (v167_data * v168_data));
                float v178_data = glb_m0[(v8_lead + 24)];
                float v179_data = s0[13];
                float v181_data = r0[2];
                r0[2] = (v181_data + (v178_data * v179_data));
                float v189_data = glb_m0[(v8_lead + 24)];
                float v190_data = s0[21];
                float v192_data = r0[3];
                r0[3] = (v192_data + (v189_data * v190_data));
                float v200_data = glb_m0[(v8_lead + 24)];
                float v201_data = s0[24];
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
                float v238_data = s0[11];
                float v240_data = r0[1];
                r0[1] = (v240_data + (v237_data * v238_data));
                float v248_data = glb_m0[(v8_lead + 36)];
                float v249_data = s0[12];
                float v251_data = r0[2];
                r0[2] = (v251_data + (v248_data * v249_data));
                float v259_data = glb_m0[(v8_lead + 36)];
                float v260_data = s0[20];
                float v262_data = r0[3];
                r0[3] = (v262_data + (v259_data * v260_data));
                float v270_data = glb_m0[(v8_lead + 36)];
                float v271_data = s0[25];
                float v273_data = r0[4];
                r0[4] = (v273_data + (v270_data * v271_data));
                float v281_data = glb_m0[(v8_lead + 36)];
                float v282_data = s0[33];
                float v284_data = r0[5];
                r0[5] = (v284_data + (v281_data * v282_data));
              }
              if (v8_lead < 12) {
                float v296_data = glb_m0[(v8_lead + 48)];
                float v297_data = s0[5];
                float v299_data = r0[0];
                r0[0] = (v299_data + (v296_data * v297_data));
                float v307_data = glb_m0[(v8_lead + 48)];
                float v308_data = s0[8];
                float v310_data = r0[1];
                r0[1] = (v310_data + (v307_data * v308_data));
                float v318_data = glb_m0[(v8_lead + 48)];
                float v319_data = s0[16];
                float v321_data = r0[2];
                r0[2] = (v321_data + (v318_data * v319_data));
                float v329_data = glb_m0[(v8_lead + 48)];
                float v330_data = s0[23];
                float v332_data = r0[3];
                r0[3] = (v332_data + (v329_data * v330_data));
                float v340_data = glb_m0[(v8_lead + 48)];
                float v341_data = s0[31];
                float v343_data = r0[4];
                r0[4] = (v343_data + (v340_data * v341_data));
                float v351_data = glb_m0[(v8_lead + 48)];
                float v352_data = s0[34];
                float v354_data = r0[5];
                r0[5] = (v354_data + (v351_data * v352_data));
              }
              if (v8_lead < 12) {
                float v366_data = glb_m0[(v8_lead + 60)];
                float v367_data = s0[4];
                float v369_data = r0[0];
                r0[0] = (v369_data + (v366_data * v367_data));
                float v377_data = glb_m0[(v8_lead + 60)];
                float v378_data = s0[9];
                float v380_data = r0[1];
                r0[1] = (v380_data + (v377_data * v378_data));
                float v388_data = glb_m0[(v8_lead + 60)];
                float v389_data = s0[17];
                float v391_data = r0[2];
                r0[2] = (v391_data + (v388_data * v389_data));
                float v399_data = glb_m0[(v8_lead + 60)];
                float v400_data = s0[22];
                float v402_data = r0[3];
                r0[3] = (v402_data + (v399_data * v400_data));
                float v410_data = glb_m0[(v8_lead + 60)];
                float v411_data = s0[30];
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
                  int32_t v440_a = v8_lead + (v431_i1 * 12);
                  s1[(v440_a ^ ((v440_a >> 3) & 7))] = v433_data;
                }
              }
              float r1[6]{};
              sycl::group_barrier(item.get_sub_group());
              // r1 = +(glb_m3 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir1[6]{};
              if (v8_lead < 12) {
                float v456_data = glb_m3[v8_lead];
                float v457_data = s1[0];
                float v459_data = ir1[0];
                ir1[0] = (v459_data + (v456_data * v457_data));
                float v467_data = glb_m3[v8_lead];
                float v468_data = s1[13];
                float v470_data = ir1[1];
                ir1[1] = (v470_data + (v467_data * v468_data));
                float v478_data = glb_m3[v8_lead];
                float v479_data = s1[27];
                float v481_data = ir1[2];
                ir1[2] = (v481_data + (v478_data * v479_data));
                float v489_data = glb_m3[v8_lead];
                float v490_data = s1[32];
                float v492_data = ir1[3];
                ir1[3] = (v492_data + (v489_data * v490_data));
                float v500_data = glb_m3[v8_lead];
                float v501_data = s1[54];
                float v503_data = ir1[4];
                ir1[4] = (v503_data + (v500_data * v501_data));
                float v511_data = glb_m3[v8_lead];
                float v512_data = s1[59];
                float v514_data = ir1[5];
                ir1[5] = (v514_data + (v511_data * v512_data));
              }
              if (v8_lead < 12) {
                float v526_data = glb_m3[(v8_lead + 12)];
                float v527_data = s1[1];
                float v529_data = ir1[0];
                ir1[0] = (v529_data + (v526_data * v527_data));
                float v537_data = glb_m3[(v8_lead + 12)];
                float v538_data = s1[12];
                float v540_data = ir1[1];
                ir1[1] = (v540_data + (v537_data * v538_data));
                float v548_data = glb_m3[(v8_lead + 12)];
                float v549_data = s1[26];
                float v551_data = ir1[2];
                ir1[2] = (v551_data + (v548_data * v549_data));
                float v559_data = glb_m3[(v8_lead + 12)];
                float v560_data = s1[33];
                float v562_data = ir1[3];
                ir1[3] = (v562_data + (v559_data * v560_data));
                float v570_data = glb_m3[(v8_lead + 12)];
                float v571_data = s1[55];
                float v573_data = ir1[4];
                ir1[4] = (v573_data + (v570_data * v571_data));
                float v581_data = glb_m3[(v8_lead + 12)];
                float v582_data = s1[58];
                float v584_data = ir1[5];
                ir1[5] = (v584_data + (v581_data * v582_data));
              }
              if (v8_lead < 12) {
                float v596_data = glb_m3[(v8_lead + 24)];
                float v597_data = s1[2];
                float v599_data = ir1[0];
                ir1[0] = (v599_data + (v596_data * v597_data));
                float v607_data = glb_m3[(v8_lead + 24)];
                float v608_data = s1[15];
                float v610_data = ir1[1];
                ir1[1] = (v610_data + (v607_data * v608_data));
                float v618_data = glb_m3[(v8_lead + 24)];
                float v619_data = s1[25];
                float v621_data = ir1[2];
                ir1[2] = (v621_data + (v618_data * v619_data));
                float v629_data = glb_m3[(v8_lead + 24)];
                float v630_data = s1[34];
                float v632_data = ir1[3];
                ir1[3] = (v632_data + (v629_data * v630_data));
                float v640_data = glb_m3[(v8_lead + 24)];
                float v641_data = s1[52];
                float v643_data = ir1[4];
                ir1[4] = (v643_data + (v640_data * v641_data));
                float v651_data = glb_m3[(v8_lead + 24)];
                float v652_data = s1[57];
                float v654_data = ir1[5];
                ir1[5] = (v654_data + (v651_data * v652_data));
              }
              if (v8_lead < 12) {
                float v666_data = glb_m3[(v8_lead + 36)];
                float v667_data = s1[3];
                float v669_data = ir1[0];
                ir1[0] = (v669_data + (v666_data * v667_data));
                float v677_data = glb_m3[(v8_lead + 36)];
                float v678_data = s1[14];
                float v680_data = ir1[1];
                ir1[1] = (v680_data + (v677_data * v678_data));
                float v688_data = glb_m3[(v8_lead + 36)];
                float v689_data = s1[24];
                float v691_data = ir1[2];
                ir1[2] = (v691_data + (v688_data * v689_data));
                float v699_data = glb_m3[(v8_lead + 36)];
                float v700_data = s1[35];
                float v702_data = ir1[3];
                ir1[3] = (v702_data + (v699_data * v700_data));
                float v710_data = glb_m3[(v8_lead + 36)];
                float v711_data = s1[53];
                float v713_data = ir1[4];
                ir1[4] = (v713_data + (v710_data * v711_data));
                float v721_data = glb_m3[(v8_lead + 36)];
                float v722_data = s1[56];
                float v724_data = ir1[5];
                ir1[5] = (v724_data + (v721_data * v722_data));
              }
              if (v8_lead < 12) {
                float v736_data = glb_m3[(v8_lead + 48)];
                float v737_data = s1[4];
                float v739_data = ir1[0];
                ir1[0] = (v739_data + (v736_data * v737_data));
                float v747_data = glb_m3[(v8_lead + 48)];
                float v748_data = s1[18];
                float v750_data = ir1[1];
                ir1[1] = (v750_data + (v747_data * v748_data));
                float v758_data = glb_m3[(v8_lead + 48)];
                float v759_data = s1[31];
                float v761_data = ir1[2];
                ir1[2] = (v761_data + (v758_data * v759_data));
                float v769_data = glb_m3[(v8_lead + 48)];
                float v770_data = s1[45];
                float v772_data = ir1[3];
                ir1[3] = (v772_data + (v769_data * v770_data));
                float v780_data = glb_m3[(v8_lead + 48)];
                float v781_data = s1[50];
                float v783_data = ir1[4];
                ir1[4] = (v783_data + (v780_data * v781_data));
                float v791_data = glb_m3[(v8_lead + 48)];
                float v792_data = s1[64];
                float v794_data = ir1[5];
                ir1[5] = (v794_data + (v791_data * v792_data));
              }
              if (v8_lead < 12) {
                float v806_data = glb_m3[(v8_lead + 60)];
                float v807_data = s1[5];
                float v809_data = ir1[0];
                ir1[0] = (v809_data + (v806_data * v807_data));
                float v817_data = glb_m3[(v8_lead + 60)];
                float v818_data = s1[19];
                float v820_data = ir1[1];
                ir1[1] = (v820_data + (v817_data * v818_data));
                float v828_data = glb_m3[(v8_lead + 60)];
                float v829_data = s1[30];
                float v831_data = ir1[2];
                ir1[2] = (v831_data + (v828_data * v829_data));
                float v839_data = glb_m3[(v8_lead + 60)];
                float v840_data = s1[44];
                float v842_data = ir1[3];
                ir1[3] = (v842_data + (v839_data * v840_data));
                float v850_data = glb_m3[(v8_lead + 60)];
                float v851_data = s1[51];
                float v853_data = ir1[4];
                ir1[4] = (v853_data + (v850_data * v851_data));
                float v861_data = glb_m3[(v8_lead + 60)];
                float v862_data = s1[65];
                float v864_data = ir1[5];
                ir1[5] = (v864_data + (v861_data * v862_data));
              }
              if (v8_lead < 12) {
                float v876_data = glb_m3[(v8_lead + 72)];
                float v877_data = s1[6];
                float v879_data = ir1[0];
                ir1[0] = (v879_data + (v876_data * v877_data));
                float v887_data = glb_m3[(v8_lead + 72)];
                float v888_data = s1[16];
                float v890_data = ir1[1];
                ir1[1] = (v890_data + (v887_data * v888_data));
                float v898_data = glb_m3[(v8_lead + 72)];
                float v899_data = s1[29];
                float v901_data = ir1[2];
                ir1[2] = (v901_data + (v898_data * v899_data));
                float v909_data = glb_m3[(v8_lead + 72)];
                float v910_data = s1[47];
                float v912_data = ir1[3];
                ir1[3] = (v912_data + (v909_data * v910_data));
                float v920_data = glb_m3[(v8_lead + 72)];
                float v921_data = s1[48];
                float v923_data = ir1[4];
                ir1[4] = (v923_data + (v920_data * v921_data));
                float v931_data = glb_m3[(v8_lead + 72)];
                float v932_data = s1[66];
                float v934_data = ir1[5];
                ir1[5] = (v934_data + (v931_data * v932_data));
              }
              if (v8_lead < 12) {
                float v946_data = glb_m3[(v8_lead + 84)];
                float v947_data = s1[7];
                float v949_data = ir1[0];
                ir1[0] = (v949_data + (v946_data * v947_data));
                float v957_data = glb_m3[(v8_lead + 84)];
                float v958_data = s1[17];
                float v960_data = ir1[1];
                ir1[1] = (v960_data + (v957_data * v958_data));
                float v968_data = glb_m3[(v8_lead + 84)];
                float v969_data = s1[28];
                float v971_data = ir1[2];
                ir1[2] = (v971_data + (v968_data * v969_data));
                float v979_data = glb_m3[(v8_lead + 84)];
                float v980_data = s1[46];
                float v982_data = ir1[3];
                ir1[3] = (v982_data + (v979_data * v980_data));
                float v990_data = glb_m3[(v8_lead + 84)];
                float v991_data = s1[49];
                float v993_data = ir1[4];
                ir1[4] = (v993_data + (v990_data * v991_data));
                float v1001_data = glb_m3[(v8_lead + 84)];
                float v1002_data = s1[67];
                float v1004_data = ir1[5];
                ir1[5] = (v1004_data + (v1001_data * v1002_data));
              }
              if (v8_lead < 12) {
                float v1016_data = glb_m3[(v8_lead + 96)];
                float v1017_data = s1[9];
                float v1019_data = ir1[0];
                ir1[0] = (v1019_data + (v1016_data * v1017_data));
                float v1027_data = glb_m3[(v8_lead + 96)];
                float v1028_data = s1[22];
                float v1030_data = ir1[1];
                ir1[1] = (v1030_data + (v1027_data * v1028_data));
                float v1038_data = glb_m3[(v8_lead + 96)];
                float v1039_data = s1[36];
                float v1041_data = ir1[2];
                ir1[2] = (v1041_data + (v1038_data * v1039_data));
                float v1049_data = glb_m3[(v8_lead + 96)];
                float v1050_data = s1[41];
                float v1052_data = ir1[3];
                ir1[3] = (v1052_data + (v1049_data * v1050_data));
                float v1060_data = glb_m3[(v8_lead + 96)];
                float v1061_data = s1[63];
                float v1063_data = ir1[4];
                ir1[4] = (v1063_data + (v1060_data * v1061_data));
                float v1071_data = glb_m3[(v8_lead + 96)];
                float v1072_data = s1[68];
                float v1074_data = ir1[5];
                ir1[5] = (v1074_data + (v1071_data * v1072_data));
              }
              if (v8_lead < 12) {
                float v1086_data = glb_m3[(v8_lead + 108)];
                float v1087_data = s1[8];
                float v1089_data = ir1[0];
                ir1[0] = (v1089_data + (v1086_data * v1087_data));
                float v1097_data = glb_m3[(v8_lead + 108)];
                float v1098_data = s1[23];
                float v1100_data = ir1[1];
                ir1[1] = (v1100_data + (v1097_data * v1098_data));
                float v1108_data = glb_m3[(v8_lead + 108)];
                float v1109_data = s1[37];
                float v1111_data = ir1[2];
                ir1[2] = (v1111_data + (v1108_data * v1109_data));
                float v1119_data = glb_m3[(v8_lead + 108)];
                float v1120_data = s1[40];
                float v1122_data = ir1[3];
                ir1[3] = (v1122_data + (v1119_data * v1120_data));
                float v1130_data = glb_m3[(v8_lead + 108)];
                float v1131_data = s1[62];
                float v1133_data = ir1[4];
                ir1[4] = (v1133_data + (v1130_data * v1131_data));
                float v1141_data = glb_m3[(v8_lead + 108)];
                float v1142_data = s1[69];
                float v1144_data = ir1[5];
                ir1[5] = (v1144_data + (v1141_data * v1142_data));
              }
              if (v8_lead < 12) {
                float v1156_data = glb_m3[(v8_lead + 120)];
                float v1157_data = s1[11];
                float v1159_data = ir1[0];
                ir1[0] = (v1159_data + (v1156_data * v1157_data));
                float v1167_data = glb_m3[(v8_lead + 120)];
                float v1168_data = s1[20];
                float v1170_data = ir1[1];
                ir1[1] = (v1170_data + (v1167_data * v1168_data));
                float v1178_data = glb_m3[(v8_lead + 120)];
                float v1179_data = s1[38];
                float v1181_data = ir1[2];
                ir1[2] = (v1181_data + (v1178_data * v1179_data));
                float v1189_data = glb_m3[(v8_lead + 120)];
                float v1190_data = s1[43];
                float v1192_data = ir1[3];
                ir1[3] = (v1192_data + (v1189_data * v1190_data));
                float v1200_data = glb_m3[(v8_lead + 120)];
                float v1201_data = s1[61];
                float v1203_data = ir1[4];
                ir1[4] = (v1203_data + (v1200_data * v1201_data));
                float v1211_data = glb_m3[(v8_lead + 120)];
                float v1212_data = s1[70];
                float v1214_data = ir1[5];
                ir1[5] = (v1214_data + (v1211_data * v1212_data));
              }
              if (v8_lead < 12) {
                float v1226_data = glb_m3[(v8_lead + 132)];
                float v1227_data = s1[10];
                float v1229_data = ir1[0];
                ir1[0] = (v1229_data + (v1226_data * v1227_data));
                float v1237_data = glb_m3[(v8_lead + 132)];
                float v1238_data = s1[21];
                float v1240_data = ir1[1];
                ir1[1] = (v1240_data + (v1237_data * v1238_data));
                float v1248_data = glb_m3[(v8_lead + 132)];
                float v1249_data = s1[39];
                float v1251_data = ir1[2];
                ir1[2] = (v1251_data + (v1248_data * v1249_data));
                float v1259_data = glb_m3[(v8_lead + 132)];
                float v1260_data = s1[42];
                float v1262_data = ir1[3];
                ir1[3] = (v1262_data + (v1259_data * v1260_data));
                float v1270_data = glb_m3[(v8_lead + 132)];
                float v1271_data = s1[60];
                float v1273_data = ir1[4];
                ir1[4] = (v1273_data + (v1270_data * v1271_data));
                float v1281_data = glb_m3[(v8_lead + 132)];
                float v1282_data = s1[71];
                float v1284_data = ir1[5];
                ir1[5] = (v1284_data + (v1281_data * v1282_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1290_n1 = 0; v1290_n1 < 6; ++v1290_n1) {
                  float v1292_data = ir1[v1290_n1];
                  r1[v1290_n1] = v1292_data;
                }
              }
              // glb_m2 = store{r>g}(r1);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1298_i1 = 0; v1298_i1 < 6; ++v1298_i1) {
                  float v1300_data = r1[v1298_i1];
                  glb_m2[(v8_lead + (v1298_i1 * 12))] = v1300_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


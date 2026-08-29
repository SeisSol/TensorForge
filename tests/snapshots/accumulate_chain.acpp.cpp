// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8a03a3cd0d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8a03a3cd0d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×12(12×12) {0..12}×{0..12} strided
        // m2 12×8(12×8) {0..12}×{0..8} strided
        // m3 12×12(12×12) {0..12}×{0..12} strided
        // m4 12×8(12×8) {0..12}×{0..8} strided
        // m5 12×12(12×12) {0..12}×{0..12} strided
        // m6 12×8(12×8) {0..12}×{0..8} strided
        // m7 12×12(12×12) {0..12}×{0..12} strided
        // m8 12×8(12×8) {0..12}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir0[8]{};
              int32_t v14_lead = item.get_local_id(0) % 16;
              if (v14_lead < 12) {
                float v22_data = glb_m1[v14_lead];
                float v23_data = s0[0];
                float v25_data = ir0[0];
                ir0[0] = (v25_data + (v22_data * v23_data));
                float v33_data = glb_m1[v14_lead];
                float v34_data = s0[12];
                float v36_data = ir0[1];
                ir0[1] = (v36_data + (v33_data * v34_data));
                float v44_data = glb_m1[v14_lead];
                float v45_data = s0[24];
                float v47_data = ir0[2];
                ir0[2] = (v47_data + (v44_data * v45_data));
                float v55_data = glb_m1[v14_lead];
                float v56_data = s0[37];
                float v58_data = ir0[3];
                ir0[3] = (v58_data + (v55_data * v56_data));
                float v66_data = glb_m1[v14_lead];
                float v67_data = s0[49];
                float v69_data = ir0[4];
                ir0[4] = (v69_data + (v66_data * v67_data));
                float v77_data = glb_m1[v14_lead];
                float v78_data = s0[61];
                float v80_data = ir0[5];
                ir0[5] = (v80_data + (v77_data * v78_data));
                float v88_data = glb_m1[v14_lead];
                float v89_data = s0[74];
                float v91_data = ir0[6];
                ir0[6] = (v91_data + (v88_data * v89_data));
                float v99_data = glb_m1[v14_lead];
                float v100_data = s0[86];
                float v102_data = ir0[7];
                ir0[7] = (v102_data + (v99_data * v100_data));
              }
              if (v14_lead < 12) {
                float v114_data = glb_m1[(v14_lead + 12)];
                float v115_data = s0[1];
                float v117_data = ir0[0];
                ir0[0] = (v117_data + (v114_data * v115_data));
                float v125_data = glb_m1[(v14_lead + 12)];
                float v126_data = s0[13];
                float v128_data = ir0[1];
                ir0[1] = (v128_data + (v125_data * v126_data));
                float v136_data = glb_m1[(v14_lead + 12)];
                float v137_data = s0[25];
                float v139_data = ir0[2];
                ir0[2] = (v139_data + (v136_data * v137_data));
                float v147_data = glb_m1[(v14_lead + 12)];
                float v148_data = s0[36];
                float v150_data = ir0[3];
                ir0[3] = (v150_data + (v147_data * v148_data));
                float v158_data = glb_m1[(v14_lead + 12)];
                float v159_data = s0[48];
                float v161_data = ir0[4];
                ir0[4] = (v161_data + (v158_data * v159_data));
                float v169_data = glb_m1[(v14_lead + 12)];
                float v170_data = s0[60];
                float v172_data = ir0[5];
                ir0[5] = (v172_data + (v169_data * v170_data));
                float v180_data = glb_m1[(v14_lead + 12)];
                float v181_data = s0[75];
                float v183_data = ir0[6];
                ir0[6] = (v183_data + (v180_data * v181_data));
                float v191_data = glb_m1[(v14_lead + 12)];
                float v192_data = s0[87];
                float v194_data = ir0[7];
                ir0[7] = (v194_data + (v191_data * v192_data));
              }
              if (v14_lead < 12) {
                float v206_data = glb_m1[(v14_lead + 24)];
                float v207_data = s0[2];
                float v209_data = ir0[0];
                ir0[0] = (v209_data + (v206_data * v207_data));
                float v217_data = glb_m1[(v14_lead + 24)];
                float v218_data = s0[14];
                float v220_data = ir0[1];
                ir0[1] = (v220_data + (v217_data * v218_data));
                float v228_data = glb_m1[(v14_lead + 24)];
                float v229_data = s0[26];
                float v231_data = ir0[2];
                ir0[2] = (v231_data + (v228_data * v229_data));
                float v239_data = glb_m1[(v14_lead + 24)];
                float v240_data = s0[39];
                float v242_data = ir0[3];
                ir0[3] = (v242_data + (v239_data * v240_data));
                float v250_data = glb_m1[(v14_lead + 24)];
                float v251_data = s0[51];
                float v253_data = ir0[4];
                ir0[4] = (v253_data + (v250_data * v251_data));
                float v261_data = glb_m1[(v14_lead + 24)];
                float v262_data = s0[63];
                float v264_data = ir0[5];
                ir0[5] = (v264_data + (v261_data * v262_data));
                float v272_data = glb_m1[(v14_lead + 24)];
                float v273_data = s0[72];
                float v275_data = ir0[6];
                ir0[6] = (v275_data + (v272_data * v273_data));
                float v283_data = glb_m1[(v14_lead + 24)];
                float v284_data = s0[84];
                float v286_data = ir0[7];
                ir0[7] = (v286_data + (v283_data * v284_data));
              }
              if (v14_lead < 12) {
                float v298_data = glb_m1[(v14_lead + 36)];
                float v299_data = s0[3];
                float v301_data = ir0[0];
                ir0[0] = (v301_data + (v298_data * v299_data));
                float v309_data = glb_m1[(v14_lead + 36)];
                float v310_data = s0[15];
                float v312_data = ir0[1];
                ir0[1] = (v312_data + (v309_data * v310_data));
                float v320_data = glb_m1[(v14_lead + 36)];
                float v321_data = s0[27];
                float v323_data = ir0[2];
                ir0[2] = (v323_data + (v320_data * v321_data));
                float v331_data = glb_m1[(v14_lead + 36)];
                float v332_data = s0[38];
                float v334_data = ir0[3];
                ir0[3] = (v334_data + (v331_data * v332_data));
                float v342_data = glb_m1[(v14_lead + 36)];
                float v343_data = s0[50];
                float v345_data = ir0[4];
                ir0[4] = (v345_data + (v342_data * v343_data));
                float v353_data = glb_m1[(v14_lead + 36)];
                float v354_data = s0[62];
                float v356_data = ir0[5];
                ir0[5] = (v356_data + (v353_data * v354_data));
                float v364_data = glb_m1[(v14_lead + 36)];
                float v365_data = s0[73];
                float v367_data = ir0[6];
                ir0[6] = (v367_data + (v364_data * v365_data));
                float v375_data = glb_m1[(v14_lead + 36)];
                float v376_data = s0[85];
                float v378_data = ir0[7];
                ir0[7] = (v378_data + (v375_data * v376_data));
              }
              if (v14_lead < 12) {
                float v390_data = glb_m1[(v14_lead + 48)];
                float v391_data = s0[4];
                float v393_data = ir0[0];
                ir0[0] = (v393_data + (v390_data * v391_data));
                float v401_data = glb_m1[(v14_lead + 48)];
                float v402_data = s0[16];
                float v404_data = ir0[1];
                ir0[1] = (v404_data + (v401_data * v402_data));
                float v412_data = glb_m1[(v14_lead + 48)];
                float v413_data = s0[28];
                float v415_data = ir0[2];
                ir0[2] = (v415_data + (v412_data * v413_data));
                float v423_data = glb_m1[(v14_lead + 48)];
                float v424_data = s0[41];
                float v426_data = ir0[3];
                ir0[3] = (v426_data + (v423_data * v424_data));
                float v434_data = glb_m1[(v14_lead + 48)];
                float v435_data = s0[53];
                float v437_data = ir0[4];
                ir0[4] = (v437_data + (v434_data * v435_data));
                float v445_data = glb_m1[(v14_lead + 48)];
                float v446_data = s0[66];
                float v448_data = ir0[5];
                ir0[5] = (v448_data + (v445_data * v446_data));
                float v456_data = glb_m1[(v14_lead + 48)];
                float v457_data = s0[78];
                float v459_data = ir0[6];
                ir0[6] = (v459_data + (v456_data * v457_data));
                float v467_data = glb_m1[(v14_lead + 48)];
                float v468_data = s0[90];
                float v470_data = ir0[7];
                ir0[7] = (v470_data + (v467_data * v468_data));
              }
              if (v14_lead < 12) {
                float v482_data = glb_m1[(v14_lead + 60)];
                float v483_data = s0[5];
                float v485_data = ir0[0];
                ir0[0] = (v485_data + (v482_data * v483_data));
                float v493_data = glb_m1[(v14_lead + 60)];
                float v494_data = s0[17];
                float v496_data = ir0[1];
                ir0[1] = (v496_data + (v493_data * v494_data));
                float v504_data = glb_m1[(v14_lead + 60)];
                float v505_data = s0[29];
                float v507_data = ir0[2];
                ir0[2] = (v507_data + (v504_data * v505_data));
                float v515_data = glb_m1[(v14_lead + 60)];
                float v516_data = s0[40];
                float v518_data = ir0[3];
                ir0[3] = (v518_data + (v515_data * v516_data));
                float v526_data = glb_m1[(v14_lead + 60)];
                float v527_data = s0[52];
                float v529_data = ir0[4];
                ir0[4] = (v529_data + (v526_data * v527_data));
                float v537_data = glb_m1[(v14_lead + 60)];
                float v538_data = s0[67];
                float v540_data = ir0[5];
                ir0[5] = (v540_data + (v537_data * v538_data));
                float v548_data = glb_m1[(v14_lead + 60)];
                float v549_data = s0[79];
                float v551_data = ir0[6];
                ir0[6] = (v551_data + (v548_data * v549_data));
                float v559_data = glb_m1[(v14_lead + 60)];
                float v560_data = s0[91];
                float v562_data = ir0[7];
                ir0[7] = (v562_data + (v559_data * v560_data));
              }
              if (v14_lead < 12) {
                float v574_data = glb_m1[(v14_lead + 72)];
                float v575_data = s0[6];
                float v577_data = ir0[0];
                ir0[0] = (v577_data + (v574_data * v575_data));
                float v585_data = glb_m1[(v14_lead + 72)];
                float v586_data = s0[18];
                float v588_data = ir0[1];
                ir0[1] = (v588_data + (v585_data * v586_data));
                float v596_data = glb_m1[(v14_lead + 72)];
                float v597_data = s0[30];
                float v599_data = ir0[2];
                ir0[2] = (v599_data + (v596_data * v597_data));
                float v607_data = glb_m1[(v14_lead + 72)];
                float v608_data = s0[43];
                float v610_data = ir0[3];
                ir0[3] = (v610_data + (v607_data * v608_data));
                float v618_data = glb_m1[(v14_lead + 72)];
                float v619_data = s0[55];
                float v621_data = ir0[4];
                ir0[4] = (v621_data + (v618_data * v619_data));
                float v629_data = glb_m1[(v14_lead + 72)];
                float v630_data = s0[64];
                float v632_data = ir0[5];
                ir0[5] = (v632_data + (v629_data * v630_data));
                float v640_data = glb_m1[(v14_lead + 72)];
                float v641_data = s0[76];
                float v643_data = ir0[6];
                ir0[6] = (v643_data + (v640_data * v641_data));
                float v651_data = glb_m1[(v14_lead + 72)];
                float v652_data = s0[88];
                float v654_data = ir0[7];
                ir0[7] = (v654_data + (v651_data * v652_data));
              }
              if (v14_lead < 12) {
                float v666_data = glb_m1[(v14_lead + 84)];
                float v667_data = s0[7];
                float v669_data = ir0[0];
                ir0[0] = (v669_data + (v666_data * v667_data));
                float v677_data = glb_m1[(v14_lead + 84)];
                float v678_data = s0[19];
                float v680_data = ir0[1];
                ir0[1] = (v680_data + (v677_data * v678_data));
                float v688_data = glb_m1[(v14_lead + 84)];
                float v689_data = s0[31];
                float v691_data = ir0[2];
                ir0[2] = (v691_data + (v688_data * v689_data));
                float v699_data = glb_m1[(v14_lead + 84)];
                float v700_data = s0[42];
                float v702_data = ir0[3];
                ir0[3] = (v702_data + (v699_data * v700_data));
                float v710_data = glb_m1[(v14_lead + 84)];
                float v711_data = s0[54];
                float v713_data = ir0[4];
                ir0[4] = (v713_data + (v710_data * v711_data));
                float v721_data = glb_m1[(v14_lead + 84)];
                float v722_data = s0[65];
                float v724_data = ir0[5];
                ir0[5] = (v724_data + (v721_data * v722_data));
                float v732_data = glb_m1[(v14_lead + 84)];
                float v733_data = s0[77];
                float v735_data = ir0[6];
                ir0[6] = (v735_data + (v732_data * v733_data));
                float v743_data = glb_m1[(v14_lead + 84)];
                float v744_data = s0[89];
                float v746_data = ir0[7];
                ir0[7] = (v746_data + (v743_data * v744_data));
              }
              if (v14_lead < 12) {
                float v758_data = glb_m1[(v14_lead + 96)];
                float v759_data = s0[8];
                float v761_data = ir0[0];
                ir0[0] = (v761_data + (v758_data * v759_data));
                float v769_data = glb_m1[(v14_lead + 96)];
                float v770_data = s0[20];
                float v772_data = ir0[1];
                ir0[1] = (v772_data + (v769_data * v770_data));
                float v780_data = glb_m1[(v14_lead + 96)];
                float v781_data = s0[33];
                float v783_data = ir0[2];
                ir0[2] = (v783_data + (v780_data * v781_data));
                float v791_data = glb_m1[(v14_lead + 96)];
                float v792_data = s0[45];
                float v794_data = ir0[3];
                ir0[3] = (v794_data + (v791_data * v792_data));
                float v802_data = glb_m1[(v14_lead + 96)];
                float v803_data = s0[57];
                float v805_data = ir0[4];
                ir0[4] = (v805_data + (v802_data * v803_data));
                float v813_data = glb_m1[(v14_lead + 96)];
                float v814_data = s0[70];
                float v816_data = ir0[5];
                ir0[5] = (v816_data + (v813_data * v814_data));
                float v824_data = glb_m1[(v14_lead + 96)];
                float v825_data = s0[82];
                float v827_data = ir0[6];
                ir0[6] = (v827_data + (v824_data * v825_data));
                float v835_data = glb_m1[(v14_lead + 96)];
                float v836_data = s0[94];
                float v838_data = ir0[7];
                ir0[7] = (v838_data + (v835_data * v836_data));
              }
              if (v14_lead < 12) {
                float v850_data = glb_m1[(v14_lead + 108)];
                float v851_data = s0[9];
                float v853_data = ir0[0];
                ir0[0] = (v853_data + (v850_data * v851_data));
                float v861_data = glb_m1[(v14_lead + 108)];
                float v862_data = s0[21];
                float v864_data = ir0[1];
                ir0[1] = (v864_data + (v861_data * v862_data));
                float v872_data = glb_m1[(v14_lead + 108)];
                float v873_data = s0[32];
                float v875_data = ir0[2];
                ir0[2] = (v875_data + (v872_data * v873_data));
                float v883_data = glb_m1[(v14_lead + 108)];
                float v884_data = s0[44];
                float v886_data = ir0[3];
                ir0[3] = (v886_data + (v883_data * v884_data));
                float v894_data = glb_m1[(v14_lead + 108)];
                float v895_data = s0[56];
                float v897_data = ir0[4];
                ir0[4] = (v897_data + (v894_data * v895_data));
                float v905_data = glb_m1[(v14_lead + 108)];
                float v906_data = s0[71];
                float v908_data = ir0[5];
                ir0[5] = (v908_data + (v905_data * v906_data));
                float v916_data = glb_m1[(v14_lead + 108)];
                float v917_data = s0[83];
                float v919_data = ir0[6];
                ir0[6] = (v919_data + (v916_data * v917_data));
                float v927_data = glb_m1[(v14_lead + 108)];
                float v928_data = s0[95];
                float v930_data = ir0[7];
                ir0[7] = (v930_data + (v927_data * v928_data));
              }
              if (v14_lead < 12) {
                float v942_data = glb_m1[(v14_lead + 120)];
                float v943_data = s0[10];
                float v945_data = ir0[0];
                ir0[0] = (v945_data + (v942_data * v943_data));
                float v953_data = glb_m1[(v14_lead + 120)];
                float v954_data = s0[22];
                float v956_data = ir0[1];
                ir0[1] = (v956_data + (v953_data * v954_data));
                float v964_data = glb_m1[(v14_lead + 120)];
                float v965_data = s0[35];
                float v967_data = ir0[2];
                ir0[2] = (v967_data + (v964_data * v965_data));
                float v975_data = glb_m1[(v14_lead + 120)];
                float v976_data = s0[47];
                float v978_data = ir0[3];
                ir0[3] = (v978_data + (v975_data * v976_data));
                float v986_data = glb_m1[(v14_lead + 120)];
                float v987_data = s0[59];
                float v989_data = ir0[4];
                ir0[4] = (v989_data + (v986_data * v987_data));
                float v997_data = glb_m1[(v14_lead + 120)];
                float v998_data = s0[68];
                float v1000_data = ir0[5];
                ir0[5] = (v1000_data + (v997_data * v998_data));
                float v1008_data = glb_m1[(v14_lead + 120)];
                float v1009_data = s0[80];
                float v1011_data = ir0[6];
                ir0[6] = (v1011_data + (v1008_data * v1009_data));
                float v1019_data = glb_m1[(v14_lead + 120)];
                float v1020_data = s0[92];
                float v1022_data = ir0[7];
                ir0[7] = (v1022_data + (v1019_data * v1020_data));
              }
              if (v14_lead < 12) {
                float v1034_data = glb_m1[(v14_lead + 132)];
                float v1035_data = s0[11];
                float v1037_data = ir0[0];
                ir0[0] = (v1037_data + (v1034_data * v1035_data));
                float v1045_data = glb_m1[(v14_lead + 132)];
                float v1046_data = s0[23];
                float v1048_data = ir0[1];
                ir0[1] = (v1048_data + (v1045_data * v1046_data));
                float v1056_data = glb_m1[(v14_lead + 132)];
                float v1057_data = s0[34];
                float v1059_data = ir0[2];
                ir0[2] = (v1059_data + (v1056_data * v1057_data));
                float v1067_data = glb_m1[(v14_lead + 132)];
                float v1068_data = s0[46];
                float v1070_data = ir0[3];
                ir0[3] = (v1070_data + (v1067_data * v1068_data));
                float v1078_data = glb_m1[(v14_lead + 132)];
                float v1079_data = s0[58];
                float v1081_data = ir0[4];
                ir0[4] = (v1081_data + (v1078_data * v1079_data));
                float v1089_data = glb_m1[(v14_lead + 132)];
                float v1090_data = s0[69];
                float v1092_data = ir0[5];
                ir0[5] = (v1092_data + (v1089_data * v1090_data));
                float v1100_data = glb_m1[(v14_lead + 132)];
                float v1101_data = s0[81];
                float v1103_data = ir0[6];
                ir0[6] = (v1103_data + (v1100_data * v1101_data));
                float v1111_data = glb_m1[(v14_lead + 132)];
                float v1112_data = s0[93];
                float v1114_data = ir0[7];
                ir0[7] = (v1114_data + (v1111_data * v1112_data));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1120_n1 = 0; v1120_n1 < 8; ++v1120_n1) {
                  float v1122_data = ir0[v1120_n1];
                  r0[v1120_n1] = v1122_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1128_i1 = 0; v1128_i1 < 8; ++v1128_i1) {
                  float v1130_data = r0[v1128_i1];
                  glb_m0[(v14_lead + (v1128_i1 * 12))] = v1130_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m4[0, 1])
              *(sycl::vec<float, 4>*)&s1[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m4[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s1[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m4[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r1[8]{};
              sycl::group_barrier(item.get_sub_group());
              // r1 = +(glb_m3 * s1) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir1[8]{};
              if (v14_lead < 12) {
                float v1151_data = glb_m3[v14_lead];
                float v1152_data = s1[0];
                float v1154_data = ir1[0];
                ir1[0] = (v1154_data + (v1151_data * v1152_data));
                float v1162_data = glb_m3[v14_lead];
                float v1163_data = s1[12];
                float v1165_data = ir1[1];
                ir1[1] = (v1165_data + (v1162_data * v1163_data));
                float v1173_data = glb_m3[v14_lead];
                float v1174_data = s1[24];
                float v1176_data = ir1[2];
                ir1[2] = (v1176_data + (v1173_data * v1174_data));
                float v1184_data = glb_m3[v14_lead];
                float v1185_data = s1[37];
                float v1187_data = ir1[3];
                ir1[3] = (v1187_data + (v1184_data * v1185_data));
                float v1195_data = glb_m3[v14_lead];
                float v1196_data = s1[49];
                float v1198_data = ir1[4];
                ir1[4] = (v1198_data + (v1195_data * v1196_data));
                float v1206_data = glb_m3[v14_lead];
                float v1207_data = s1[61];
                float v1209_data = ir1[5];
                ir1[5] = (v1209_data + (v1206_data * v1207_data));
                float v1217_data = glb_m3[v14_lead];
                float v1218_data = s1[74];
                float v1220_data = ir1[6];
                ir1[6] = (v1220_data + (v1217_data * v1218_data));
                float v1228_data = glb_m3[v14_lead];
                float v1229_data = s1[86];
                float v1231_data = ir1[7];
                ir1[7] = (v1231_data + (v1228_data * v1229_data));
              }
              if (v14_lead < 12) {
                float v1243_data = glb_m3[(v14_lead + 12)];
                float v1244_data = s1[1];
                float v1246_data = ir1[0];
                ir1[0] = (v1246_data + (v1243_data * v1244_data));
                float v1254_data = glb_m3[(v14_lead + 12)];
                float v1255_data = s1[13];
                float v1257_data = ir1[1];
                ir1[1] = (v1257_data + (v1254_data * v1255_data));
                float v1265_data = glb_m3[(v14_lead + 12)];
                float v1266_data = s1[25];
                float v1268_data = ir1[2];
                ir1[2] = (v1268_data + (v1265_data * v1266_data));
                float v1276_data = glb_m3[(v14_lead + 12)];
                float v1277_data = s1[36];
                float v1279_data = ir1[3];
                ir1[3] = (v1279_data + (v1276_data * v1277_data));
                float v1287_data = glb_m3[(v14_lead + 12)];
                float v1288_data = s1[48];
                float v1290_data = ir1[4];
                ir1[4] = (v1290_data + (v1287_data * v1288_data));
                float v1298_data = glb_m3[(v14_lead + 12)];
                float v1299_data = s1[60];
                float v1301_data = ir1[5];
                ir1[5] = (v1301_data + (v1298_data * v1299_data));
                float v1309_data = glb_m3[(v14_lead + 12)];
                float v1310_data = s1[75];
                float v1312_data = ir1[6];
                ir1[6] = (v1312_data + (v1309_data * v1310_data));
                float v1320_data = glb_m3[(v14_lead + 12)];
                float v1321_data = s1[87];
                float v1323_data = ir1[7];
                ir1[7] = (v1323_data + (v1320_data * v1321_data));
              }
              if (v14_lead < 12) {
                float v1335_data = glb_m3[(v14_lead + 24)];
                float v1336_data = s1[2];
                float v1338_data = ir1[0];
                ir1[0] = (v1338_data + (v1335_data * v1336_data));
                float v1346_data = glb_m3[(v14_lead + 24)];
                float v1347_data = s1[14];
                float v1349_data = ir1[1];
                ir1[1] = (v1349_data + (v1346_data * v1347_data));
                float v1357_data = glb_m3[(v14_lead + 24)];
                float v1358_data = s1[26];
                float v1360_data = ir1[2];
                ir1[2] = (v1360_data + (v1357_data * v1358_data));
                float v1368_data = glb_m3[(v14_lead + 24)];
                float v1369_data = s1[39];
                float v1371_data = ir1[3];
                ir1[3] = (v1371_data + (v1368_data * v1369_data));
                float v1379_data = glb_m3[(v14_lead + 24)];
                float v1380_data = s1[51];
                float v1382_data = ir1[4];
                ir1[4] = (v1382_data + (v1379_data * v1380_data));
                float v1390_data = glb_m3[(v14_lead + 24)];
                float v1391_data = s1[63];
                float v1393_data = ir1[5];
                ir1[5] = (v1393_data + (v1390_data * v1391_data));
                float v1401_data = glb_m3[(v14_lead + 24)];
                float v1402_data = s1[72];
                float v1404_data = ir1[6];
                ir1[6] = (v1404_data + (v1401_data * v1402_data));
                float v1412_data = glb_m3[(v14_lead + 24)];
                float v1413_data = s1[84];
                float v1415_data = ir1[7];
                ir1[7] = (v1415_data + (v1412_data * v1413_data));
              }
              if (v14_lead < 12) {
                float v1427_data = glb_m3[(v14_lead + 36)];
                float v1428_data = s1[3];
                float v1430_data = ir1[0];
                ir1[0] = (v1430_data + (v1427_data * v1428_data));
                float v1438_data = glb_m3[(v14_lead + 36)];
                float v1439_data = s1[15];
                float v1441_data = ir1[1];
                ir1[1] = (v1441_data + (v1438_data * v1439_data));
                float v1449_data = glb_m3[(v14_lead + 36)];
                float v1450_data = s1[27];
                float v1452_data = ir1[2];
                ir1[2] = (v1452_data + (v1449_data * v1450_data));
                float v1460_data = glb_m3[(v14_lead + 36)];
                float v1461_data = s1[38];
                float v1463_data = ir1[3];
                ir1[3] = (v1463_data + (v1460_data * v1461_data));
                float v1471_data = glb_m3[(v14_lead + 36)];
                float v1472_data = s1[50];
                float v1474_data = ir1[4];
                ir1[4] = (v1474_data + (v1471_data * v1472_data));
                float v1482_data = glb_m3[(v14_lead + 36)];
                float v1483_data = s1[62];
                float v1485_data = ir1[5];
                ir1[5] = (v1485_data + (v1482_data * v1483_data));
                float v1493_data = glb_m3[(v14_lead + 36)];
                float v1494_data = s1[73];
                float v1496_data = ir1[6];
                ir1[6] = (v1496_data + (v1493_data * v1494_data));
                float v1504_data = glb_m3[(v14_lead + 36)];
                float v1505_data = s1[85];
                float v1507_data = ir1[7];
                ir1[7] = (v1507_data + (v1504_data * v1505_data));
              }
              if (v14_lead < 12) {
                float v1519_data = glb_m3[(v14_lead + 48)];
                float v1520_data = s1[4];
                float v1522_data = ir1[0];
                ir1[0] = (v1522_data + (v1519_data * v1520_data));
                float v1530_data = glb_m3[(v14_lead + 48)];
                float v1531_data = s1[16];
                float v1533_data = ir1[1];
                ir1[1] = (v1533_data + (v1530_data * v1531_data));
                float v1541_data = glb_m3[(v14_lead + 48)];
                float v1542_data = s1[28];
                float v1544_data = ir1[2];
                ir1[2] = (v1544_data + (v1541_data * v1542_data));
                float v1552_data = glb_m3[(v14_lead + 48)];
                float v1553_data = s1[41];
                float v1555_data = ir1[3];
                ir1[3] = (v1555_data + (v1552_data * v1553_data));
                float v1563_data = glb_m3[(v14_lead + 48)];
                float v1564_data = s1[53];
                float v1566_data = ir1[4];
                ir1[4] = (v1566_data + (v1563_data * v1564_data));
                float v1574_data = glb_m3[(v14_lead + 48)];
                float v1575_data = s1[66];
                float v1577_data = ir1[5];
                ir1[5] = (v1577_data + (v1574_data * v1575_data));
                float v1585_data = glb_m3[(v14_lead + 48)];
                float v1586_data = s1[78];
                float v1588_data = ir1[6];
                ir1[6] = (v1588_data + (v1585_data * v1586_data));
                float v1596_data = glb_m3[(v14_lead + 48)];
                float v1597_data = s1[90];
                float v1599_data = ir1[7];
                ir1[7] = (v1599_data + (v1596_data * v1597_data));
              }
              if (v14_lead < 12) {
                float v1611_data = glb_m3[(v14_lead + 60)];
                float v1612_data = s1[5];
                float v1614_data = ir1[0];
                ir1[0] = (v1614_data + (v1611_data * v1612_data));
                float v1622_data = glb_m3[(v14_lead + 60)];
                float v1623_data = s1[17];
                float v1625_data = ir1[1];
                ir1[1] = (v1625_data + (v1622_data * v1623_data));
                float v1633_data = glb_m3[(v14_lead + 60)];
                float v1634_data = s1[29];
                float v1636_data = ir1[2];
                ir1[2] = (v1636_data + (v1633_data * v1634_data));
                float v1644_data = glb_m3[(v14_lead + 60)];
                float v1645_data = s1[40];
                float v1647_data = ir1[3];
                ir1[3] = (v1647_data + (v1644_data * v1645_data));
                float v1655_data = glb_m3[(v14_lead + 60)];
                float v1656_data = s1[52];
                float v1658_data = ir1[4];
                ir1[4] = (v1658_data + (v1655_data * v1656_data));
                float v1666_data = glb_m3[(v14_lead + 60)];
                float v1667_data = s1[67];
                float v1669_data = ir1[5];
                ir1[5] = (v1669_data + (v1666_data * v1667_data));
                float v1677_data = glb_m3[(v14_lead + 60)];
                float v1678_data = s1[79];
                float v1680_data = ir1[6];
                ir1[6] = (v1680_data + (v1677_data * v1678_data));
                float v1688_data = glb_m3[(v14_lead + 60)];
                float v1689_data = s1[91];
                float v1691_data = ir1[7];
                ir1[7] = (v1691_data + (v1688_data * v1689_data));
              }
              if (v14_lead < 12) {
                float v1703_data = glb_m3[(v14_lead + 72)];
                float v1704_data = s1[6];
                float v1706_data = ir1[0];
                ir1[0] = (v1706_data + (v1703_data * v1704_data));
                float v1714_data = glb_m3[(v14_lead + 72)];
                float v1715_data = s1[18];
                float v1717_data = ir1[1];
                ir1[1] = (v1717_data + (v1714_data * v1715_data));
                float v1725_data = glb_m3[(v14_lead + 72)];
                float v1726_data = s1[30];
                float v1728_data = ir1[2];
                ir1[2] = (v1728_data + (v1725_data * v1726_data));
                float v1736_data = glb_m3[(v14_lead + 72)];
                float v1737_data = s1[43];
                float v1739_data = ir1[3];
                ir1[3] = (v1739_data + (v1736_data * v1737_data));
                float v1747_data = glb_m3[(v14_lead + 72)];
                float v1748_data = s1[55];
                float v1750_data = ir1[4];
                ir1[4] = (v1750_data + (v1747_data * v1748_data));
                float v1758_data = glb_m3[(v14_lead + 72)];
                float v1759_data = s1[64];
                float v1761_data = ir1[5];
                ir1[5] = (v1761_data + (v1758_data * v1759_data));
                float v1769_data = glb_m3[(v14_lead + 72)];
                float v1770_data = s1[76];
                float v1772_data = ir1[6];
                ir1[6] = (v1772_data + (v1769_data * v1770_data));
                float v1780_data = glb_m3[(v14_lead + 72)];
                float v1781_data = s1[88];
                float v1783_data = ir1[7];
                ir1[7] = (v1783_data + (v1780_data * v1781_data));
              }
              if (v14_lead < 12) {
                float v1795_data = glb_m3[(v14_lead + 84)];
                float v1796_data = s1[7];
                float v1798_data = ir1[0];
                ir1[0] = (v1798_data + (v1795_data * v1796_data));
                float v1806_data = glb_m3[(v14_lead + 84)];
                float v1807_data = s1[19];
                float v1809_data = ir1[1];
                ir1[1] = (v1809_data + (v1806_data * v1807_data));
                float v1817_data = glb_m3[(v14_lead + 84)];
                float v1818_data = s1[31];
                float v1820_data = ir1[2];
                ir1[2] = (v1820_data + (v1817_data * v1818_data));
                float v1828_data = glb_m3[(v14_lead + 84)];
                float v1829_data = s1[42];
                float v1831_data = ir1[3];
                ir1[3] = (v1831_data + (v1828_data * v1829_data));
                float v1839_data = glb_m3[(v14_lead + 84)];
                float v1840_data = s1[54];
                float v1842_data = ir1[4];
                ir1[4] = (v1842_data + (v1839_data * v1840_data));
                float v1850_data = glb_m3[(v14_lead + 84)];
                float v1851_data = s1[65];
                float v1853_data = ir1[5];
                ir1[5] = (v1853_data + (v1850_data * v1851_data));
                float v1861_data = glb_m3[(v14_lead + 84)];
                float v1862_data = s1[77];
                float v1864_data = ir1[6];
                ir1[6] = (v1864_data + (v1861_data * v1862_data));
                float v1872_data = glb_m3[(v14_lead + 84)];
                float v1873_data = s1[89];
                float v1875_data = ir1[7];
                ir1[7] = (v1875_data + (v1872_data * v1873_data));
              }
              if (v14_lead < 12) {
                float v1887_data = glb_m3[(v14_lead + 96)];
                float v1888_data = s1[8];
                float v1890_data = ir1[0];
                ir1[0] = (v1890_data + (v1887_data * v1888_data));
                float v1898_data = glb_m3[(v14_lead + 96)];
                float v1899_data = s1[20];
                float v1901_data = ir1[1];
                ir1[1] = (v1901_data + (v1898_data * v1899_data));
                float v1909_data = glb_m3[(v14_lead + 96)];
                float v1910_data = s1[33];
                float v1912_data = ir1[2];
                ir1[2] = (v1912_data + (v1909_data * v1910_data));
                float v1920_data = glb_m3[(v14_lead + 96)];
                float v1921_data = s1[45];
                float v1923_data = ir1[3];
                ir1[3] = (v1923_data + (v1920_data * v1921_data));
                float v1931_data = glb_m3[(v14_lead + 96)];
                float v1932_data = s1[57];
                float v1934_data = ir1[4];
                ir1[4] = (v1934_data + (v1931_data * v1932_data));
                float v1942_data = glb_m3[(v14_lead + 96)];
                float v1943_data = s1[70];
                float v1945_data = ir1[5];
                ir1[5] = (v1945_data + (v1942_data * v1943_data));
                float v1953_data = glb_m3[(v14_lead + 96)];
                float v1954_data = s1[82];
                float v1956_data = ir1[6];
                ir1[6] = (v1956_data + (v1953_data * v1954_data));
                float v1964_data = glb_m3[(v14_lead + 96)];
                float v1965_data = s1[94];
                float v1967_data = ir1[7];
                ir1[7] = (v1967_data + (v1964_data * v1965_data));
              }
              if (v14_lead < 12) {
                float v1979_data = glb_m3[(v14_lead + 108)];
                float v1980_data = s1[9];
                float v1982_data = ir1[0];
                ir1[0] = (v1982_data + (v1979_data * v1980_data));
                float v1990_data = glb_m3[(v14_lead + 108)];
                float v1991_data = s1[21];
                float v1993_data = ir1[1];
                ir1[1] = (v1993_data + (v1990_data * v1991_data));
                float v2001_data = glb_m3[(v14_lead + 108)];
                float v2002_data = s1[32];
                float v2004_data = ir1[2];
                ir1[2] = (v2004_data + (v2001_data * v2002_data));
                float v2012_data = glb_m3[(v14_lead + 108)];
                float v2013_data = s1[44];
                float v2015_data = ir1[3];
                ir1[3] = (v2015_data + (v2012_data * v2013_data));
                float v2023_data = glb_m3[(v14_lead + 108)];
                float v2024_data = s1[56];
                float v2026_data = ir1[4];
                ir1[4] = (v2026_data + (v2023_data * v2024_data));
                float v2034_data = glb_m3[(v14_lead + 108)];
                float v2035_data = s1[71];
                float v2037_data = ir1[5];
                ir1[5] = (v2037_data + (v2034_data * v2035_data));
                float v2045_data = glb_m3[(v14_lead + 108)];
                float v2046_data = s1[83];
                float v2048_data = ir1[6];
                ir1[6] = (v2048_data + (v2045_data * v2046_data));
                float v2056_data = glb_m3[(v14_lead + 108)];
                float v2057_data = s1[95];
                float v2059_data = ir1[7];
                ir1[7] = (v2059_data + (v2056_data * v2057_data));
              }
              if (v14_lead < 12) {
                float v2071_data = glb_m3[(v14_lead + 120)];
                float v2072_data = s1[10];
                float v2074_data = ir1[0];
                ir1[0] = (v2074_data + (v2071_data * v2072_data));
                float v2082_data = glb_m3[(v14_lead + 120)];
                float v2083_data = s1[22];
                float v2085_data = ir1[1];
                ir1[1] = (v2085_data + (v2082_data * v2083_data));
                float v2093_data = glb_m3[(v14_lead + 120)];
                float v2094_data = s1[35];
                float v2096_data = ir1[2];
                ir1[2] = (v2096_data + (v2093_data * v2094_data));
                float v2104_data = glb_m3[(v14_lead + 120)];
                float v2105_data = s1[47];
                float v2107_data = ir1[3];
                ir1[3] = (v2107_data + (v2104_data * v2105_data));
                float v2115_data = glb_m3[(v14_lead + 120)];
                float v2116_data = s1[59];
                float v2118_data = ir1[4];
                ir1[4] = (v2118_data + (v2115_data * v2116_data));
                float v2126_data = glb_m3[(v14_lead + 120)];
                float v2127_data = s1[68];
                float v2129_data = ir1[5];
                ir1[5] = (v2129_data + (v2126_data * v2127_data));
                float v2137_data = glb_m3[(v14_lead + 120)];
                float v2138_data = s1[80];
                float v2140_data = ir1[6];
                ir1[6] = (v2140_data + (v2137_data * v2138_data));
                float v2148_data = glb_m3[(v14_lead + 120)];
                float v2149_data = s1[92];
                float v2151_data = ir1[7];
                ir1[7] = (v2151_data + (v2148_data * v2149_data));
              }
              if (v14_lead < 12) {
                float v2163_data = glb_m3[(v14_lead + 132)];
                float v2164_data = s1[11];
                float v2166_data = ir1[0];
                ir1[0] = (v2166_data + (v2163_data * v2164_data));
                float v2174_data = glb_m3[(v14_lead + 132)];
                float v2175_data = s1[23];
                float v2177_data = ir1[1];
                ir1[1] = (v2177_data + (v2174_data * v2175_data));
                float v2185_data = glb_m3[(v14_lead + 132)];
                float v2186_data = s1[34];
                float v2188_data = ir1[2];
                ir1[2] = (v2188_data + (v2185_data * v2186_data));
                float v2196_data = glb_m3[(v14_lead + 132)];
                float v2197_data = s1[46];
                float v2199_data = ir1[3];
                ir1[3] = (v2199_data + (v2196_data * v2197_data));
                float v2207_data = glb_m3[(v14_lead + 132)];
                float v2208_data = s1[58];
                float v2210_data = ir1[4];
                ir1[4] = (v2210_data + (v2207_data * v2208_data));
                float v2218_data = glb_m3[(v14_lead + 132)];
                float v2219_data = s1[69];
                float v2221_data = ir1[5];
                ir1[5] = (v2221_data + (v2218_data * v2219_data));
                float v2229_data = glb_m3[(v14_lead + 132)];
                float v2230_data = s1[81];
                float v2232_data = ir1[6];
                ir1[6] = (v2232_data + (v2229_data * v2230_data));
                float v2240_data = glb_m3[(v14_lead + 132)];
                float v2241_data = s1[93];
                float v2243_data = ir1[7];
                ir1[7] = (v2243_data + (v2240_data * v2241_data));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v2249_n1 = 0; v2249_n1 < 8; ++v2249_n1) {
                  float v2251_data = ir1[v2249_n1];
                  float v2259_data = glb_m0[(v14_lead + (v2249_n1 * 12))];
                  r1[v2249_n1] = (v2259_data + v2251_data);
                }
              }
              // glb_m0 = store{r>g}(r1);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v2266_i1 = 0; v2266_i1 < 8; ++v2266_i1) {
                  float v2268_data = r1[v2266_i1];
                  glb_m0[(v14_lead + (v2266_i1 * 12))] = v2268_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m6[0, 1])
              *(sycl::vec<float, 4>*)&s2[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m6[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s2[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m6[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              float r2[8]{};
              sycl::group_barrier(item.get_sub_group());
              // r2 = +(glb_m5 * s2) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              if (v14_lead < 12) {
                float v2289_data = glb_m5[v14_lead];
                float v2290_data = s2[0];
                float v2292_data = ir2[0];
                ir2[0] = (v2292_data + (v2289_data * v2290_data));
                float v2300_data = glb_m5[v14_lead];
                float v2301_data = s2[12];
                float v2303_data = ir2[1];
                ir2[1] = (v2303_data + (v2300_data * v2301_data));
                float v2311_data = glb_m5[v14_lead];
                float v2312_data = s2[24];
                float v2314_data = ir2[2];
                ir2[2] = (v2314_data + (v2311_data * v2312_data));
                float v2322_data = glb_m5[v14_lead];
                float v2323_data = s2[37];
                float v2325_data = ir2[3];
                ir2[3] = (v2325_data + (v2322_data * v2323_data));
                float v2333_data = glb_m5[v14_lead];
                float v2334_data = s2[49];
                float v2336_data = ir2[4];
                ir2[4] = (v2336_data + (v2333_data * v2334_data));
                float v2344_data = glb_m5[v14_lead];
                float v2345_data = s2[61];
                float v2347_data = ir2[5];
                ir2[5] = (v2347_data + (v2344_data * v2345_data));
                float v2355_data = glb_m5[v14_lead];
                float v2356_data = s2[74];
                float v2358_data = ir2[6];
                ir2[6] = (v2358_data + (v2355_data * v2356_data));
                float v2366_data = glb_m5[v14_lead];
                float v2367_data = s2[86];
                float v2369_data = ir2[7];
                ir2[7] = (v2369_data + (v2366_data * v2367_data));
              }
              if (v14_lead < 12) {
                float v2381_data = glb_m5[(v14_lead + 12)];
                float v2382_data = s2[1];
                float v2384_data = ir2[0];
                ir2[0] = (v2384_data + (v2381_data * v2382_data));
                float v2392_data = glb_m5[(v14_lead + 12)];
                float v2393_data = s2[13];
                float v2395_data = ir2[1];
                ir2[1] = (v2395_data + (v2392_data * v2393_data));
                float v2403_data = glb_m5[(v14_lead + 12)];
                float v2404_data = s2[25];
                float v2406_data = ir2[2];
                ir2[2] = (v2406_data + (v2403_data * v2404_data));
                float v2414_data = glb_m5[(v14_lead + 12)];
                float v2415_data = s2[36];
                float v2417_data = ir2[3];
                ir2[3] = (v2417_data + (v2414_data * v2415_data));
                float v2425_data = glb_m5[(v14_lead + 12)];
                float v2426_data = s2[48];
                float v2428_data = ir2[4];
                ir2[4] = (v2428_data + (v2425_data * v2426_data));
                float v2436_data = glb_m5[(v14_lead + 12)];
                float v2437_data = s2[60];
                float v2439_data = ir2[5];
                ir2[5] = (v2439_data + (v2436_data * v2437_data));
                float v2447_data = glb_m5[(v14_lead + 12)];
                float v2448_data = s2[75];
                float v2450_data = ir2[6];
                ir2[6] = (v2450_data + (v2447_data * v2448_data));
                float v2458_data = glb_m5[(v14_lead + 12)];
                float v2459_data = s2[87];
                float v2461_data = ir2[7];
                ir2[7] = (v2461_data + (v2458_data * v2459_data));
              }
              if (v14_lead < 12) {
                float v2473_data = glb_m5[(v14_lead + 24)];
                float v2474_data = s2[2];
                float v2476_data = ir2[0];
                ir2[0] = (v2476_data + (v2473_data * v2474_data));
                float v2484_data = glb_m5[(v14_lead + 24)];
                float v2485_data = s2[14];
                float v2487_data = ir2[1];
                ir2[1] = (v2487_data + (v2484_data * v2485_data));
                float v2495_data = glb_m5[(v14_lead + 24)];
                float v2496_data = s2[26];
                float v2498_data = ir2[2];
                ir2[2] = (v2498_data + (v2495_data * v2496_data));
                float v2506_data = glb_m5[(v14_lead + 24)];
                float v2507_data = s2[39];
                float v2509_data = ir2[3];
                ir2[3] = (v2509_data + (v2506_data * v2507_data));
                float v2517_data = glb_m5[(v14_lead + 24)];
                float v2518_data = s2[51];
                float v2520_data = ir2[4];
                ir2[4] = (v2520_data + (v2517_data * v2518_data));
                float v2528_data = glb_m5[(v14_lead + 24)];
                float v2529_data = s2[63];
                float v2531_data = ir2[5];
                ir2[5] = (v2531_data + (v2528_data * v2529_data));
                float v2539_data = glb_m5[(v14_lead + 24)];
                float v2540_data = s2[72];
                float v2542_data = ir2[6];
                ir2[6] = (v2542_data + (v2539_data * v2540_data));
                float v2550_data = glb_m5[(v14_lead + 24)];
                float v2551_data = s2[84];
                float v2553_data = ir2[7];
                ir2[7] = (v2553_data + (v2550_data * v2551_data));
              }
              if (v14_lead < 12) {
                float v2565_data = glb_m5[(v14_lead + 36)];
                float v2566_data = s2[3];
                float v2568_data = ir2[0];
                ir2[0] = (v2568_data + (v2565_data * v2566_data));
                float v2576_data = glb_m5[(v14_lead + 36)];
                float v2577_data = s2[15];
                float v2579_data = ir2[1];
                ir2[1] = (v2579_data + (v2576_data * v2577_data));
                float v2587_data = glb_m5[(v14_lead + 36)];
                float v2588_data = s2[27];
                float v2590_data = ir2[2];
                ir2[2] = (v2590_data + (v2587_data * v2588_data));
                float v2598_data = glb_m5[(v14_lead + 36)];
                float v2599_data = s2[38];
                float v2601_data = ir2[3];
                ir2[3] = (v2601_data + (v2598_data * v2599_data));
                float v2609_data = glb_m5[(v14_lead + 36)];
                float v2610_data = s2[50];
                float v2612_data = ir2[4];
                ir2[4] = (v2612_data + (v2609_data * v2610_data));
                float v2620_data = glb_m5[(v14_lead + 36)];
                float v2621_data = s2[62];
                float v2623_data = ir2[5];
                ir2[5] = (v2623_data + (v2620_data * v2621_data));
                float v2631_data = glb_m5[(v14_lead + 36)];
                float v2632_data = s2[73];
                float v2634_data = ir2[6];
                ir2[6] = (v2634_data + (v2631_data * v2632_data));
                float v2642_data = glb_m5[(v14_lead + 36)];
                float v2643_data = s2[85];
                float v2645_data = ir2[7];
                ir2[7] = (v2645_data + (v2642_data * v2643_data));
              }
              if (v14_lead < 12) {
                float v2657_data = glb_m5[(v14_lead + 48)];
                float v2658_data = s2[4];
                float v2660_data = ir2[0];
                ir2[0] = (v2660_data + (v2657_data * v2658_data));
                float v2668_data = glb_m5[(v14_lead + 48)];
                float v2669_data = s2[16];
                float v2671_data = ir2[1];
                ir2[1] = (v2671_data + (v2668_data * v2669_data));
                float v2679_data = glb_m5[(v14_lead + 48)];
                float v2680_data = s2[28];
                float v2682_data = ir2[2];
                ir2[2] = (v2682_data + (v2679_data * v2680_data));
                float v2690_data = glb_m5[(v14_lead + 48)];
                float v2691_data = s2[41];
                float v2693_data = ir2[3];
                ir2[3] = (v2693_data + (v2690_data * v2691_data));
                float v2701_data = glb_m5[(v14_lead + 48)];
                float v2702_data = s2[53];
                float v2704_data = ir2[4];
                ir2[4] = (v2704_data + (v2701_data * v2702_data));
                float v2712_data = glb_m5[(v14_lead + 48)];
                float v2713_data = s2[66];
                float v2715_data = ir2[5];
                ir2[5] = (v2715_data + (v2712_data * v2713_data));
                float v2723_data = glb_m5[(v14_lead + 48)];
                float v2724_data = s2[78];
                float v2726_data = ir2[6];
                ir2[6] = (v2726_data + (v2723_data * v2724_data));
                float v2734_data = glb_m5[(v14_lead + 48)];
                float v2735_data = s2[90];
                float v2737_data = ir2[7];
                ir2[7] = (v2737_data + (v2734_data * v2735_data));
              }
              if (v14_lead < 12) {
                float v2749_data = glb_m5[(v14_lead + 60)];
                float v2750_data = s2[5];
                float v2752_data = ir2[0];
                ir2[0] = (v2752_data + (v2749_data * v2750_data));
                float v2760_data = glb_m5[(v14_lead + 60)];
                float v2761_data = s2[17];
                float v2763_data = ir2[1];
                ir2[1] = (v2763_data + (v2760_data * v2761_data));
                float v2771_data = glb_m5[(v14_lead + 60)];
                float v2772_data = s2[29];
                float v2774_data = ir2[2];
                ir2[2] = (v2774_data + (v2771_data * v2772_data));
                float v2782_data = glb_m5[(v14_lead + 60)];
                float v2783_data = s2[40];
                float v2785_data = ir2[3];
                ir2[3] = (v2785_data + (v2782_data * v2783_data));
                float v2793_data = glb_m5[(v14_lead + 60)];
                float v2794_data = s2[52];
                float v2796_data = ir2[4];
                ir2[4] = (v2796_data + (v2793_data * v2794_data));
                float v2804_data = glb_m5[(v14_lead + 60)];
                float v2805_data = s2[67];
                float v2807_data = ir2[5];
                ir2[5] = (v2807_data + (v2804_data * v2805_data));
                float v2815_data = glb_m5[(v14_lead + 60)];
                float v2816_data = s2[79];
                float v2818_data = ir2[6];
                ir2[6] = (v2818_data + (v2815_data * v2816_data));
                float v2826_data = glb_m5[(v14_lead + 60)];
                float v2827_data = s2[91];
                float v2829_data = ir2[7];
                ir2[7] = (v2829_data + (v2826_data * v2827_data));
              }
              if (v14_lead < 12) {
                float v2841_data = glb_m5[(v14_lead + 72)];
                float v2842_data = s2[6];
                float v2844_data = ir2[0];
                ir2[0] = (v2844_data + (v2841_data * v2842_data));
                float v2852_data = glb_m5[(v14_lead + 72)];
                float v2853_data = s2[18];
                float v2855_data = ir2[1];
                ir2[1] = (v2855_data + (v2852_data * v2853_data));
                float v2863_data = glb_m5[(v14_lead + 72)];
                float v2864_data = s2[30];
                float v2866_data = ir2[2];
                ir2[2] = (v2866_data + (v2863_data * v2864_data));
                float v2874_data = glb_m5[(v14_lead + 72)];
                float v2875_data = s2[43];
                float v2877_data = ir2[3];
                ir2[3] = (v2877_data + (v2874_data * v2875_data));
                float v2885_data = glb_m5[(v14_lead + 72)];
                float v2886_data = s2[55];
                float v2888_data = ir2[4];
                ir2[4] = (v2888_data + (v2885_data * v2886_data));
                float v2896_data = glb_m5[(v14_lead + 72)];
                float v2897_data = s2[64];
                float v2899_data = ir2[5];
                ir2[5] = (v2899_data + (v2896_data * v2897_data));
                float v2907_data = glb_m5[(v14_lead + 72)];
                float v2908_data = s2[76];
                float v2910_data = ir2[6];
                ir2[6] = (v2910_data + (v2907_data * v2908_data));
                float v2918_data = glb_m5[(v14_lead + 72)];
                float v2919_data = s2[88];
                float v2921_data = ir2[7];
                ir2[7] = (v2921_data + (v2918_data * v2919_data));
              }
              if (v14_lead < 12) {
                float v2933_data = glb_m5[(v14_lead + 84)];
                float v2934_data = s2[7];
                float v2936_data = ir2[0];
                ir2[0] = (v2936_data + (v2933_data * v2934_data));
                float v2944_data = glb_m5[(v14_lead + 84)];
                float v2945_data = s2[19];
                float v2947_data = ir2[1];
                ir2[1] = (v2947_data + (v2944_data * v2945_data));
                float v2955_data = glb_m5[(v14_lead + 84)];
                float v2956_data = s2[31];
                float v2958_data = ir2[2];
                ir2[2] = (v2958_data + (v2955_data * v2956_data));
                float v2966_data = glb_m5[(v14_lead + 84)];
                float v2967_data = s2[42];
                float v2969_data = ir2[3];
                ir2[3] = (v2969_data + (v2966_data * v2967_data));
                float v2977_data = glb_m5[(v14_lead + 84)];
                float v2978_data = s2[54];
                float v2980_data = ir2[4];
                ir2[4] = (v2980_data + (v2977_data * v2978_data));
                float v2988_data = glb_m5[(v14_lead + 84)];
                float v2989_data = s2[65];
                float v2991_data = ir2[5];
                ir2[5] = (v2991_data + (v2988_data * v2989_data));
                float v2999_data = glb_m5[(v14_lead + 84)];
                float v3000_data = s2[77];
                float v3002_data = ir2[6];
                ir2[6] = (v3002_data + (v2999_data * v3000_data));
                float v3010_data = glb_m5[(v14_lead + 84)];
                float v3011_data = s2[89];
                float v3013_data = ir2[7];
                ir2[7] = (v3013_data + (v3010_data * v3011_data));
              }
              if (v14_lead < 12) {
                float v3025_data = glb_m5[(v14_lead + 96)];
                float v3026_data = s2[8];
                float v3028_data = ir2[0];
                ir2[0] = (v3028_data + (v3025_data * v3026_data));
                float v3036_data = glb_m5[(v14_lead + 96)];
                float v3037_data = s2[20];
                float v3039_data = ir2[1];
                ir2[1] = (v3039_data + (v3036_data * v3037_data));
                float v3047_data = glb_m5[(v14_lead + 96)];
                float v3048_data = s2[33];
                float v3050_data = ir2[2];
                ir2[2] = (v3050_data + (v3047_data * v3048_data));
                float v3058_data = glb_m5[(v14_lead + 96)];
                float v3059_data = s2[45];
                float v3061_data = ir2[3];
                ir2[3] = (v3061_data + (v3058_data * v3059_data));
                float v3069_data = glb_m5[(v14_lead + 96)];
                float v3070_data = s2[57];
                float v3072_data = ir2[4];
                ir2[4] = (v3072_data + (v3069_data * v3070_data));
                float v3080_data = glb_m5[(v14_lead + 96)];
                float v3081_data = s2[70];
                float v3083_data = ir2[5];
                ir2[5] = (v3083_data + (v3080_data * v3081_data));
                float v3091_data = glb_m5[(v14_lead + 96)];
                float v3092_data = s2[82];
                float v3094_data = ir2[6];
                ir2[6] = (v3094_data + (v3091_data * v3092_data));
                float v3102_data = glb_m5[(v14_lead + 96)];
                float v3103_data = s2[94];
                float v3105_data = ir2[7];
                ir2[7] = (v3105_data + (v3102_data * v3103_data));
              }
              if (v14_lead < 12) {
                float v3117_data = glb_m5[(v14_lead + 108)];
                float v3118_data = s2[9];
                float v3120_data = ir2[0];
                ir2[0] = (v3120_data + (v3117_data * v3118_data));
                float v3128_data = glb_m5[(v14_lead + 108)];
                float v3129_data = s2[21];
                float v3131_data = ir2[1];
                ir2[1] = (v3131_data + (v3128_data * v3129_data));
                float v3139_data = glb_m5[(v14_lead + 108)];
                float v3140_data = s2[32];
                float v3142_data = ir2[2];
                ir2[2] = (v3142_data + (v3139_data * v3140_data));
                float v3150_data = glb_m5[(v14_lead + 108)];
                float v3151_data = s2[44];
                float v3153_data = ir2[3];
                ir2[3] = (v3153_data + (v3150_data * v3151_data));
                float v3161_data = glb_m5[(v14_lead + 108)];
                float v3162_data = s2[56];
                float v3164_data = ir2[4];
                ir2[4] = (v3164_data + (v3161_data * v3162_data));
                float v3172_data = glb_m5[(v14_lead + 108)];
                float v3173_data = s2[71];
                float v3175_data = ir2[5];
                ir2[5] = (v3175_data + (v3172_data * v3173_data));
                float v3183_data = glb_m5[(v14_lead + 108)];
                float v3184_data = s2[83];
                float v3186_data = ir2[6];
                ir2[6] = (v3186_data + (v3183_data * v3184_data));
                float v3194_data = glb_m5[(v14_lead + 108)];
                float v3195_data = s2[95];
                float v3197_data = ir2[7];
                ir2[7] = (v3197_data + (v3194_data * v3195_data));
              }
              if (v14_lead < 12) {
                float v3209_data = glb_m5[(v14_lead + 120)];
                float v3210_data = s2[10];
                float v3212_data = ir2[0];
                ir2[0] = (v3212_data + (v3209_data * v3210_data));
                float v3220_data = glb_m5[(v14_lead + 120)];
                float v3221_data = s2[22];
                float v3223_data = ir2[1];
                ir2[1] = (v3223_data + (v3220_data * v3221_data));
                float v3231_data = glb_m5[(v14_lead + 120)];
                float v3232_data = s2[35];
                float v3234_data = ir2[2];
                ir2[2] = (v3234_data + (v3231_data * v3232_data));
                float v3242_data = glb_m5[(v14_lead + 120)];
                float v3243_data = s2[47];
                float v3245_data = ir2[3];
                ir2[3] = (v3245_data + (v3242_data * v3243_data));
                float v3253_data = glb_m5[(v14_lead + 120)];
                float v3254_data = s2[59];
                float v3256_data = ir2[4];
                ir2[4] = (v3256_data + (v3253_data * v3254_data));
                float v3264_data = glb_m5[(v14_lead + 120)];
                float v3265_data = s2[68];
                float v3267_data = ir2[5];
                ir2[5] = (v3267_data + (v3264_data * v3265_data));
                float v3275_data = glb_m5[(v14_lead + 120)];
                float v3276_data = s2[80];
                float v3278_data = ir2[6];
                ir2[6] = (v3278_data + (v3275_data * v3276_data));
                float v3286_data = glb_m5[(v14_lead + 120)];
                float v3287_data = s2[92];
                float v3289_data = ir2[7];
                ir2[7] = (v3289_data + (v3286_data * v3287_data));
              }
              if (v14_lead < 12) {
                float v3301_data = glb_m5[(v14_lead + 132)];
                float v3302_data = s2[11];
                float v3304_data = ir2[0];
                ir2[0] = (v3304_data + (v3301_data * v3302_data));
                float v3312_data = glb_m5[(v14_lead + 132)];
                float v3313_data = s2[23];
                float v3315_data = ir2[1];
                ir2[1] = (v3315_data + (v3312_data * v3313_data));
                float v3323_data = glb_m5[(v14_lead + 132)];
                float v3324_data = s2[34];
                float v3326_data = ir2[2];
                ir2[2] = (v3326_data + (v3323_data * v3324_data));
                float v3334_data = glb_m5[(v14_lead + 132)];
                float v3335_data = s2[46];
                float v3337_data = ir2[3];
                ir2[3] = (v3337_data + (v3334_data * v3335_data));
                float v3345_data = glb_m5[(v14_lead + 132)];
                float v3346_data = s2[58];
                float v3348_data = ir2[4];
                ir2[4] = (v3348_data + (v3345_data * v3346_data));
                float v3356_data = glb_m5[(v14_lead + 132)];
                float v3357_data = s2[69];
                float v3359_data = ir2[5];
                ir2[5] = (v3359_data + (v3356_data * v3357_data));
                float v3367_data = glb_m5[(v14_lead + 132)];
                float v3368_data = s2[81];
                float v3370_data = ir2[6];
                ir2[6] = (v3370_data + (v3367_data * v3368_data));
                float v3378_data = glb_m5[(v14_lead + 132)];
                float v3379_data = s2[93];
                float v3381_data = ir2[7];
                ir2[7] = (v3381_data + (v3378_data * v3379_data));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v3387_n1 = 0; v3387_n1 < 8; ++v3387_n1) {
                  float v3389_data = ir2[v3387_n1];
                  float v3397_data = glb_m0[(v14_lead + (v3387_n1 * 12))];
                  r2[v3387_n1] = (v3397_data + v3389_data);
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v3404_i1 = 0; v3404_i1 < 8; ++v3404_i1) {
                  float v3406_data = r2[v3404_i1];
                  glb_m0[(v14_lead + (v3404_i1 * 12))] = v3406_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              float* __restrict__ s3 = &localShrMem0[0];
              // s3 = load{g>s}(glb_m8[0, 1])
              *(sycl::vec<float, 4>*)&s3[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m8[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s3[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m8[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              float r3[8]{};
              sycl::group_barrier(item.get_sub_group());
              // r3 = +(glb_m7 * s3) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir3[8]{};
              if (v14_lead < 12) {
                float v3427_data = glb_m7[v14_lead];
                float v3428_data = s3[0];
                float v3430_data = ir3[0];
                ir3[0] = (v3430_data + (v3427_data * v3428_data));
                float v3438_data = glb_m7[v14_lead];
                float v3439_data = s3[12];
                float v3441_data = ir3[1];
                ir3[1] = (v3441_data + (v3438_data * v3439_data));
                float v3449_data = glb_m7[v14_lead];
                float v3450_data = s3[24];
                float v3452_data = ir3[2];
                ir3[2] = (v3452_data + (v3449_data * v3450_data));
                float v3460_data = glb_m7[v14_lead];
                float v3461_data = s3[37];
                float v3463_data = ir3[3];
                ir3[3] = (v3463_data + (v3460_data * v3461_data));
                float v3471_data = glb_m7[v14_lead];
                float v3472_data = s3[49];
                float v3474_data = ir3[4];
                ir3[4] = (v3474_data + (v3471_data * v3472_data));
                float v3482_data = glb_m7[v14_lead];
                float v3483_data = s3[61];
                float v3485_data = ir3[5];
                ir3[5] = (v3485_data + (v3482_data * v3483_data));
                float v3493_data = glb_m7[v14_lead];
                float v3494_data = s3[74];
                float v3496_data = ir3[6];
                ir3[6] = (v3496_data + (v3493_data * v3494_data));
                float v3504_data = glb_m7[v14_lead];
                float v3505_data = s3[86];
                float v3507_data = ir3[7];
                ir3[7] = (v3507_data + (v3504_data * v3505_data));
              }
              if (v14_lead < 12) {
                float v3519_data = glb_m7[(v14_lead + 12)];
                float v3520_data = s3[1];
                float v3522_data = ir3[0];
                ir3[0] = (v3522_data + (v3519_data * v3520_data));
                float v3530_data = glb_m7[(v14_lead + 12)];
                float v3531_data = s3[13];
                float v3533_data = ir3[1];
                ir3[1] = (v3533_data + (v3530_data * v3531_data));
                float v3541_data = glb_m7[(v14_lead + 12)];
                float v3542_data = s3[25];
                float v3544_data = ir3[2];
                ir3[2] = (v3544_data + (v3541_data * v3542_data));
                float v3552_data = glb_m7[(v14_lead + 12)];
                float v3553_data = s3[36];
                float v3555_data = ir3[3];
                ir3[3] = (v3555_data + (v3552_data * v3553_data));
                float v3563_data = glb_m7[(v14_lead + 12)];
                float v3564_data = s3[48];
                float v3566_data = ir3[4];
                ir3[4] = (v3566_data + (v3563_data * v3564_data));
                float v3574_data = glb_m7[(v14_lead + 12)];
                float v3575_data = s3[60];
                float v3577_data = ir3[5];
                ir3[5] = (v3577_data + (v3574_data * v3575_data));
                float v3585_data = glb_m7[(v14_lead + 12)];
                float v3586_data = s3[75];
                float v3588_data = ir3[6];
                ir3[6] = (v3588_data + (v3585_data * v3586_data));
                float v3596_data = glb_m7[(v14_lead + 12)];
                float v3597_data = s3[87];
                float v3599_data = ir3[7];
                ir3[7] = (v3599_data + (v3596_data * v3597_data));
              }
              if (v14_lead < 12) {
                float v3611_data = glb_m7[(v14_lead + 24)];
                float v3612_data = s3[2];
                float v3614_data = ir3[0];
                ir3[0] = (v3614_data + (v3611_data * v3612_data));
                float v3622_data = glb_m7[(v14_lead + 24)];
                float v3623_data = s3[14];
                float v3625_data = ir3[1];
                ir3[1] = (v3625_data + (v3622_data * v3623_data));
                float v3633_data = glb_m7[(v14_lead + 24)];
                float v3634_data = s3[26];
                float v3636_data = ir3[2];
                ir3[2] = (v3636_data + (v3633_data * v3634_data));
                float v3644_data = glb_m7[(v14_lead + 24)];
                float v3645_data = s3[39];
                float v3647_data = ir3[3];
                ir3[3] = (v3647_data + (v3644_data * v3645_data));
                float v3655_data = glb_m7[(v14_lead + 24)];
                float v3656_data = s3[51];
                float v3658_data = ir3[4];
                ir3[4] = (v3658_data + (v3655_data * v3656_data));
                float v3666_data = glb_m7[(v14_lead + 24)];
                float v3667_data = s3[63];
                float v3669_data = ir3[5];
                ir3[5] = (v3669_data + (v3666_data * v3667_data));
                float v3677_data = glb_m7[(v14_lead + 24)];
                float v3678_data = s3[72];
                float v3680_data = ir3[6];
                ir3[6] = (v3680_data + (v3677_data * v3678_data));
                float v3688_data = glb_m7[(v14_lead + 24)];
                float v3689_data = s3[84];
                float v3691_data = ir3[7];
                ir3[7] = (v3691_data + (v3688_data * v3689_data));
              }
              if (v14_lead < 12) {
                float v3703_data = glb_m7[(v14_lead + 36)];
                float v3704_data = s3[3];
                float v3706_data = ir3[0];
                ir3[0] = (v3706_data + (v3703_data * v3704_data));
                float v3714_data = glb_m7[(v14_lead + 36)];
                float v3715_data = s3[15];
                float v3717_data = ir3[1];
                ir3[1] = (v3717_data + (v3714_data * v3715_data));
                float v3725_data = glb_m7[(v14_lead + 36)];
                float v3726_data = s3[27];
                float v3728_data = ir3[2];
                ir3[2] = (v3728_data + (v3725_data * v3726_data));
                float v3736_data = glb_m7[(v14_lead + 36)];
                float v3737_data = s3[38];
                float v3739_data = ir3[3];
                ir3[3] = (v3739_data + (v3736_data * v3737_data));
                float v3747_data = glb_m7[(v14_lead + 36)];
                float v3748_data = s3[50];
                float v3750_data = ir3[4];
                ir3[4] = (v3750_data + (v3747_data * v3748_data));
                float v3758_data = glb_m7[(v14_lead + 36)];
                float v3759_data = s3[62];
                float v3761_data = ir3[5];
                ir3[5] = (v3761_data + (v3758_data * v3759_data));
                float v3769_data = glb_m7[(v14_lead + 36)];
                float v3770_data = s3[73];
                float v3772_data = ir3[6];
                ir3[6] = (v3772_data + (v3769_data * v3770_data));
                float v3780_data = glb_m7[(v14_lead + 36)];
                float v3781_data = s3[85];
                float v3783_data = ir3[7];
                ir3[7] = (v3783_data + (v3780_data * v3781_data));
              }
              if (v14_lead < 12) {
                float v3795_data = glb_m7[(v14_lead + 48)];
                float v3796_data = s3[4];
                float v3798_data = ir3[0];
                ir3[0] = (v3798_data + (v3795_data * v3796_data));
                float v3806_data = glb_m7[(v14_lead + 48)];
                float v3807_data = s3[16];
                float v3809_data = ir3[1];
                ir3[1] = (v3809_data + (v3806_data * v3807_data));
                float v3817_data = glb_m7[(v14_lead + 48)];
                float v3818_data = s3[28];
                float v3820_data = ir3[2];
                ir3[2] = (v3820_data + (v3817_data * v3818_data));
                float v3828_data = glb_m7[(v14_lead + 48)];
                float v3829_data = s3[41];
                float v3831_data = ir3[3];
                ir3[3] = (v3831_data + (v3828_data * v3829_data));
                float v3839_data = glb_m7[(v14_lead + 48)];
                float v3840_data = s3[53];
                float v3842_data = ir3[4];
                ir3[4] = (v3842_data + (v3839_data * v3840_data));
                float v3850_data = glb_m7[(v14_lead + 48)];
                float v3851_data = s3[66];
                float v3853_data = ir3[5];
                ir3[5] = (v3853_data + (v3850_data * v3851_data));
                float v3861_data = glb_m7[(v14_lead + 48)];
                float v3862_data = s3[78];
                float v3864_data = ir3[6];
                ir3[6] = (v3864_data + (v3861_data * v3862_data));
                float v3872_data = glb_m7[(v14_lead + 48)];
                float v3873_data = s3[90];
                float v3875_data = ir3[7];
                ir3[7] = (v3875_data + (v3872_data * v3873_data));
              }
              if (v14_lead < 12) {
                float v3887_data = glb_m7[(v14_lead + 60)];
                float v3888_data = s3[5];
                float v3890_data = ir3[0];
                ir3[0] = (v3890_data + (v3887_data * v3888_data));
                float v3898_data = glb_m7[(v14_lead + 60)];
                float v3899_data = s3[17];
                float v3901_data = ir3[1];
                ir3[1] = (v3901_data + (v3898_data * v3899_data));
                float v3909_data = glb_m7[(v14_lead + 60)];
                float v3910_data = s3[29];
                float v3912_data = ir3[2];
                ir3[2] = (v3912_data + (v3909_data * v3910_data));
                float v3920_data = glb_m7[(v14_lead + 60)];
                float v3921_data = s3[40];
                float v3923_data = ir3[3];
                ir3[3] = (v3923_data + (v3920_data * v3921_data));
                float v3931_data = glb_m7[(v14_lead + 60)];
                float v3932_data = s3[52];
                float v3934_data = ir3[4];
                ir3[4] = (v3934_data + (v3931_data * v3932_data));
                float v3942_data = glb_m7[(v14_lead + 60)];
                float v3943_data = s3[67];
                float v3945_data = ir3[5];
                ir3[5] = (v3945_data + (v3942_data * v3943_data));
                float v3953_data = glb_m7[(v14_lead + 60)];
                float v3954_data = s3[79];
                float v3956_data = ir3[6];
                ir3[6] = (v3956_data + (v3953_data * v3954_data));
                float v3964_data = glb_m7[(v14_lead + 60)];
                float v3965_data = s3[91];
                float v3967_data = ir3[7];
                ir3[7] = (v3967_data + (v3964_data * v3965_data));
              }
              if (v14_lead < 12) {
                float v3979_data = glb_m7[(v14_lead + 72)];
                float v3980_data = s3[6];
                float v3982_data = ir3[0];
                ir3[0] = (v3982_data + (v3979_data * v3980_data));
                float v3990_data = glb_m7[(v14_lead + 72)];
                float v3991_data = s3[18];
                float v3993_data = ir3[1];
                ir3[1] = (v3993_data + (v3990_data * v3991_data));
                float v4001_data = glb_m7[(v14_lead + 72)];
                float v4002_data = s3[30];
                float v4004_data = ir3[2];
                ir3[2] = (v4004_data + (v4001_data * v4002_data));
                float v4012_data = glb_m7[(v14_lead + 72)];
                float v4013_data = s3[43];
                float v4015_data = ir3[3];
                ir3[3] = (v4015_data + (v4012_data * v4013_data));
                float v4023_data = glb_m7[(v14_lead + 72)];
                float v4024_data = s3[55];
                float v4026_data = ir3[4];
                ir3[4] = (v4026_data + (v4023_data * v4024_data));
                float v4034_data = glb_m7[(v14_lead + 72)];
                float v4035_data = s3[64];
                float v4037_data = ir3[5];
                ir3[5] = (v4037_data + (v4034_data * v4035_data));
                float v4045_data = glb_m7[(v14_lead + 72)];
                float v4046_data = s3[76];
                float v4048_data = ir3[6];
                ir3[6] = (v4048_data + (v4045_data * v4046_data));
                float v4056_data = glb_m7[(v14_lead + 72)];
                float v4057_data = s3[88];
                float v4059_data = ir3[7];
                ir3[7] = (v4059_data + (v4056_data * v4057_data));
              }
              if (v14_lead < 12) {
                float v4071_data = glb_m7[(v14_lead + 84)];
                float v4072_data = s3[7];
                float v4074_data = ir3[0];
                ir3[0] = (v4074_data + (v4071_data * v4072_data));
                float v4082_data = glb_m7[(v14_lead + 84)];
                float v4083_data = s3[19];
                float v4085_data = ir3[1];
                ir3[1] = (v4085_data + (v4082_data * v4083_data));
                float v4093_data = glb_m7[(v14_lead + 84)];
                float v4094_data = s3[31];
                float v4096_data = ir3[2];
                ir3[2] = (v4096_data + (v4093_data * v4094_data));
                float v4104_data = glb_m7[(v14_lead + 84)];
                float v4105_data = s3[42];
                float v4107_data = ir3[3];
                ir3[3] = (v4107_data + (v4104_data * v4105_data));
                float v4115_data = glb_m7[(v14_lead + 84)];
                float v4116_data = s3[54];
                float v4118_data = ir3[4];
                ir3[4] = (v4118_data + (v4115_data * v4116_data));
                float v4126_data = glb_m7[(v14_lead + 84)];
                float v4127_data = s3[65];
                float v4129_data = ir3[5];
                ir3[5] = (v4129_data + (v4126_data * v4127_data));
                float v4137_data = glb_m7[(v14_lead + 84)];
                float v4138_data = s3[77];
                float v4140_data = ir3[6];
                ir3[6] = (v4140_data + (v4137_data * v4138_data));
                float v4148_data = glb_m7[(v14_lead + 84)];
                float v4149_data = s3[89];
                float v4151_data = ir3[7];
                ir3[7] = (v4151_data + (v4148_data * v4149_data));
              }
              if (v14_lead < 12) {
                float v4163_data = glb_m7[(v14_lead + 96)];
                float v4164_data = s3[8];
                float v4166_data = ir3[0];
                ir3[0] = (v4166_data + (v4163_data * v4164_data));
                float v4174_data = glb_m7[(v14_lead + 96)];
                float v4175_data = s3[20];
                float v4177_data = ir3[1];
                ir3[1] = (v4177_data + (v4174_data * v4175_data));
                float v4185_data = glb_m7[(v14_lead + 96)];
                float v4186_data = s3[33];
                float v4188_data = ir3[2];
                ir3[2] = (v4188_data + (v4185_data * v4186_data));
                float v4196_data = glb_m7[(v14_lead + 96)];
                float v4197_data = s3[45];
                float v4199_data = ir3[3];
                ir3[3] = (v4199_data + (v4196_data * v4197_data));
                float v4207_data = glb_m7[(v14_lead + 96)];
                float v4208_data = s3[57];
                float v4210_data = ir3[4];
                ir3[4] = (v4210_data + (v4207_data * v4208_data));
                float v4218_data = glb_m7[(v14_lead + 96)];
                float v4219_data = s3[70];
                float v4221_data = ir3[5];
                ir3[5] = (v4221_data + (v4218_data * v4219_data));
                float v4229_data = glb_m7[(v14_lead + 96)];
                float v4230_data = s3[82];
                float v4232_data = ir3[6];
                ir3[6] = (v4232_data + (v4229_data * v4230_data));
                float v4240_data = glb_m7[(v14_lead + 96)];
                float v4241_data = s3[94];
                float v4243_data = ir3[7];
                ir3[7] = (v4243_data + (v4240_data * v4241_data));
              }
              if (v14_lead < 12) {
                float v4255_data = glb_m7[(v14_lead + 108)];
                float v4256_data = s3[9];
                float v4258_data = ir3[0];
                ir3[0] = (v4258_data + (v4255_data * v4256_data));
                float v4266_data = glb_m7[(v14_lead + 108)];
                float v4267_data = s3[21];
                float v4269_data = ir3[1];
                ir3[1] = (v4269_data + (v4266_data * v4267_data));
                float v4277_data = glb_m7[(v14_lead + 108)];
                float v4278_data = s3[32];
                float v4280_data = ir3[2];
                ir3[2] = (v4280_data + (v4277_data * v4278_data));
                float v4288_data = glb_m7[(v14_lead + 108)];
                float v4289_data = s3[44];
                float v4291_data = ir3[3];
                ir3[3] = (v4291_data + (v4288_data * v4289_data));
                float v4299_data = glb_m7[(v14_lead + 108)];
                float v4300_data = s3[56];
                float v4302_data = ir3[4];
                ir3[4] = (v4302_data + (v4299_data * v4300_data));
                float v4310_data = glb_m7[(v14_lead + 108)];
                float v4311_data = s3[71];
                float v4313_data = ir3[5];
                ir3[5] = (v4313_data + (v4310_data * v4311_data));
                float v4321_data = glb_m7[(v14_lead + 108)];
                float v4322_data = s3[83];
                float v4324_data = ir3[6];
                ir3[6] = (v4324_data + (v4321_data * v4322_data));
                float v4332_data = glb_m7[(v14_lead + 108)];
                float v4333_data = s3[95];
                float v4335_data = ir3[7];
                ir3[7] = (v4335_data + (v4332_data * v4333_data));
              }
              if (v14_lead < 12) {
                float v4347_data = glb_m7[(v14_lead + 120)];
                float v4348_data = s3[10];
                float v4350_data = ir3[0];
                ir3[0] = (v4350_data + (v4347_data * v4348_data));
                float v4358_data = glb_m7[(v14_lead + 120)];
                float v4359_data = s3[22];
                float v4361_data = ir3[1];
                ir3[1] = (v4361_data + (v4358_data * v4359_data));
                float v4369_data = glb_m7[(v14_lead + 120)];
                float v4370_data = s3[35];
                float v4372_data = ir3[2];
                ir3[2] = (v4372_data + (v4369_data * v4370_data));
                float v4380_data = glb_m7[(v14_lead + 120)];
                float v4381_data = s3[47];
                float v4383_data = ir3[3];
                ir3[3] = (v4383_data + (v4380_data * v4381_data));
                float v4391_data = glb_m7[(v14_lead + 120)];
                float v4392_data = s3[59];
                float v4394_data = ir3[4];
                ir3[4] = (v4394_data + (v4391_data * v4392_data));
                float v4402_data = glb_m7[(v14_lead + 120)];
                float v4403_data = s3[68];
                float v4405_data = ir3[5];
                ir3[5] = (v4405_data + (v4402_data * v4403_data));
                float v4413_data = glb_m7[(v14_lead + 120)];
                float v4414_data = s3[80];
                float v4416_data = ir3[6];
                ir3[6] = (v4416_data + (v4413_data * v4414_data));
                float v4424_data = glb_m7[(v14_lead + 120)];
                float v4425_data = s3[92];
                float v4427_data = ir3[7];
                ir3[7] = (v4427_data + (v4424_data * v4425_data));
              }
              if (v14_lead < 12) {
                float v4439_data = glb_m7[(v14_lead + 132)];
                float v4440_data = s3[11];
                float v4442_data = ir3[0];
                ir3[0] = (v4442_data + (v4439_data * v4440_data));
                float v4450_data = glb_m7[(v14_lead + 132)];
                float v4451_data = s3[23];
                float v4453_data = ir3[1];
                ir3[1] = (v4453_data + (v4450_data * v4451_data));
                float v4461_data = glb_m7[(v14_lead + 132)];
                float v4462_data = s3[34];
                float v4464_data = ir3[2];
                ir3[2] = (v4464_data + (v4461_data * v4462_data));
                float v4472_data = glb_m7[(v14_lead + 132)];
                float v4473_data = s3[46];
                float v4475_data = ir3[3];
                ir3[3] = (v4475_data + (v4472_data * v4473_data));
                float v4483_data = glb_m7[(v14_lead + 132)];
                float v4484_data = s3[58];
                float v4486_data = ir3[4];
                ir3[4] = (v4486_data + (v4483_data * v4484_data));
                float v4494_data = glb_m7[(v14_lead + 132)];
                float v4495_data = s3[69];
                float v4497_data = ir3[5];
                ir3[5] = (v4497_data + (v4494_data * v4495_data));
                float v4505_data = glb_m7[(v14_lead + 132)];
                float v4506_data = s3[81];
                float v4508_data = ir3[6];
                ir3[6] = (v4508_data + (v4505_data * v4506_data));
                float v4516_data = glb_m7[(v14_lead + 132)];
                float v4517_data = s3[93];
                float v4519_data = ir3[7];
                ir3[7] = (v4519_data + (v4516_data * v4517_data));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v4525_n1 = 0; v4525_n1 < 8; ++v4525_n1) {
                  float v4527_data = ir3[v4525_n1];
                  float v4535_data = glb_m0[(v14_lead + (v4525_n1 * 12))];
                  r3[v4525_n1] = (v4535_data + v4527_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v4542_i1 = 0; v4542_i1 < 8; ++v4542_i1) {
                  float v4544_data = r3[v4542_i1];
                  glb_m0[(v14_lead + (v4542_i1 * 12))] = v4544_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


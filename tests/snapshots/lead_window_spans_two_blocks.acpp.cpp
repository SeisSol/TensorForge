// === base name ===
kernel_671a350836

// === header ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_671a350836(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_671a350836(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 64×13(64×13) {0..64}×{0..13} pointer_based
        // m1 6(6) {0..6} none
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
        // t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..64}×{0..13}×{0..6})[0, 1, 2] = m0 64×13(64×13) {0..64}×{0..13} pointer_based({0..64}×{0..13})[0, 1]×m1 6(6) {0..6} none({0..6})[2]
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based({0..15}×{0..1}×{0..6})[0, 1, 2] += t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..15}×{0..1}×{0..6})[0, 1, 2]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const float *const __restrict__ glb_m1 = &m1[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[26]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v6_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v7_i0 = 0; v7_i0 < 2; ++v7_i0) {
                int32_t v13_lead = v6_lead + (v7_i0 * 32);
                #pragma unroll
                for (int32_t v8_i1 = 0; v8_i1 < 13; ++v8_i1) {
                  float v16_data = glb_m0[(v13_lead + (v8_i1 * 64))];
                  r0[(v7_i0 + (v8_i1 * 2))] = v16_data;
                }
              }
              float r2[12]{};
              // r2 = load{g>r}(glb_m2);
              if (v6_lead >= 20) {
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 1; ++v24_i1) {
                  int32_t v34_a = v6_lead + ((v24_i1 + 12) * 64);
                  int32_t v37_a = v24_i1 * 2;
                  #pragma unroll
                  for (int32_t v25_i2 = 0; v25_i2 < 6; ++v25_i2) {
                    float v36_data = glb_m2[(v34_a + (v25_i2 * 832))];
                    r2[(v37_a + (v25_i2 * 2))] = v36_data;
                  }
                }
              }
              if (v6_lead < 3) {
                int32_t v48_lead = v6_lead + 32_i32;
                #pragma unroll
                for (int32_t v42_i1 = 0; v42_i1 < 1; ++v42_i1) {
                  int32_t v52_a = v48_lead + ((v42_i1 + 12) * 64);
                  int32_t v57_a = 1 + (v42_i1 * 2);
                  #pragma unroll
                  for (int32_t v43_i2 = 0; v43_i2 < 6; ++v43_i2) {
                    float v54_data = glb_m2[(v52_a + (v43_i2 * 832))];
                    r2[(v57_a + (v43_i2 * 2))] = v54_data;
                  }
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[156]{};
              // r1 = +(r0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              float v63_data = r0[0];
              float v64_data = glb_m1[0];
              float v66_data = r1[0];
              r1[0] = (v66_data + (v63_data * v64_data));
              float v69_data = glb_m1[1];
              float v71_data = r1[26];
              r1[26] = (v71_data + (v63_data * v69_data));
              float v74_data = glb_m1[2];
              float v76_data = r1[52];
              r1[52] = (v76_data + (v63_data * v74_data));
              float v79_data = glb_m1[3];
              float v81_data = r1[78];
              r1[78] = (v81_data + (v63_data * v79_data));
              float v84_data = glb_m1[4];
              float v86_data = r1[104];
              r1[104] = (v86_data + (v63_data * v84_data));
              float v89_data = glb_m1[5];
              float v91_data = r1[130];
              r1[130] = (v91_data + (v63_data * v89_data));
              float v93_data = r0[2];
              float v96_data = r1[2];
              r1[2] = (v96_data + (v93_data * v64_data));
              float v101_data = r1[28];
              r1[28] = (v101_data + (v93_data * v69_data));
              float v106_data = r1[54];
              r1[54] = (v106_data + (v93_data * v74_data));
              float v111_data = r1[80];
              r1[80] = (v111_data + (v93_data * v79_data));
              float v116_data = r1[106];
              r1[106] = (v116_data + (v93_data * v84_data));
              float v121_data = r1[132];
              r1[132] = (v121_data + (v93_data * v89_data));
              float v123_data = r0[4];
              float v126_data = r1[4];
              r1[4] = (v126_data + (v123_data * v64_data));
              float v131_data = r1[30];
              r1[30] = (v131_data + (v123_data * v69_data));
              float v136_data = r1[56];
              r1[56] = (v136_data + (v123_data * v74_data));
              float v141_data = r1[82];
              r1[82] = (v141_data + (v123_data * v79_data));
              float v146_data = r1[108];
              r1[108] = (v146_data + (v123_data * v84_data));
              float v151_data = r1[134];
              r1[134] = (v151_data + (v123_data * v89_data));
              float v153_data = r0[6];
              float v156_data = r1[6];
              r1[6] = (v156_data + (v153_data * v64_data));
              float v161_data = r1[32];
              r1[32] = (v161_data + (v153_data * v69_data));
              float v166_data = r1[58];
              r1[58] = (v166_data + (v153_data * v74_data));
              float v171_data = r1[84];
              r1[84] = (v171_data + (v153_data * v79_data));
              float v176_data = r1[110];
              r1[110] = (v176_data + (v153_data * v84_data));
              float v181_data = r1[136];
              r1[136] = (v181_data + (v153_data * v89_data));
              float v183_data = r0[8];
              float v186_data = r1[8];
              r1[8] = (v186_data + (v183_data * v64_data));
              float v191_data = r1[34];
              r1[34] = (v191_data + (v183_data * v69_data));
              float v196_data = r1[60];
              r1[60] = (v196_data + (v183_data * v74_data));
              float v201_data = r1[86];
              r1[86] = (v201_data + (v183_data * v79_data));
              float v206_data = r1[112];
              r1[112] = (v206_data + (v183_data * v84_data));
              float v211_data = r1[138];
              r1[138] = (v211_data + (v183_data * v89_data));
              float v213_data = r0[10];
              float v216_data = r1[10];
              r1[10] = (v216_data + (v213_data * v64_data));
              float v221_data = r1[36];
              r1[36] = (v221_data + (v213_data * v69_data));
              float v226_data = r1[62];
              r1[62] = (v226_data + (v213_data * v74_data));
              float v231_data = r1[88];
              r1[88] = (v231_data + (v213_data * v79_data));
              float v236_data = r1[114];
              r1[114] = (v236_data + (v213_data * v84_data));
              float v241_data = r1[140];
              r1[140] = (v241_data + (v213_data * v89_data));
              float v243_data = r0[12];
              float v246_data = r1[12];
              r1[12] = (v246_data + (v243_data * v64_data));
              float v251_data = r1[38];
              r1[38] = (v251_data + (v243_data * v69_data));
              float v256_data = r1[64];
              r1[64] = (v256_data + (v243_data * v74_data));
              float v261_data = r1[90];
              r1[90] = (v261_data + (v243_data * v79_data));
              float v266_data = r1[116];
              r1[116] = (v266_data + (v243_data * v84_data));
              float v271_data = r1[142];
              r1[142] = (v271_data + (v243_data * v89_data));
              float v273_data = r0[14];
              float v276_data = r1[14];
              r1[14] = (v276_data + (v273_data * v64_data));
              float v281_data = r1[40];
              r1[40] = (v281_data + (v273_data * v69_data));
              float v286_data = r1[66];
              r1[66] = (v286_data + (v273_data * v74_data));
              float v291_data = r1[92];
              r1[92] = (v291_data + (v273_data * v79_data));
              float v296_data = r1[118];
              r1[118] = (v296_data + (v273_data * v84_data));
              float v301_data = r1[144];
              r1[144] = (v301_data + (v273_data * v89_data));
              float v303_data = r0[16];
              float v306_data = r1[16];
              r1[16] = (v306_data + (v303_data * v64_data));
              float v311_data = r1[42];
              r1[42] = (v311_data + (v303_data * v69_data));
              float v316_data = r1[68];
              r1[68] = (v316_data + (v303_data * v74_data));
              float v321_data = r1[94];
              r1[94] = (v321_data + (v303_data * v79_data));
              float v326_data = r1[120];
              r1[120] = (v326_data + (v303_data * v84_data));
              float v331_data = r1[146];
              r1[146] = (v331_data + (v303_data * v89_data));
              float v333_data = r0[18];
              float v336_data = r1[18];
              r1[18] = (v336_data + (v333_data * v64_data));
              float v341_data = r1[44];
              r1[44] = (v341_data + (v333_data * v69_data));
              float v346_data = r1[70];
              r1[70] = (v346_data + (v333_data * v74_data));
              float v351_data = r1[96];
              r1[96] = (v351_data + (v333_data * v79_data));
              float v356_data = r1[122];
              r1[122] = (v356_data + (v333_data * v84_data));
              float v361_data = r1[148];
              r1[148] = (v361_data + (v333_data * v89_data));
              float v363_data = r0[20];
              float v366_data = r1[20];
              r1[20] = (v366_data + (v363_data * v64_data));
              float v371_data = r1[46];
              r1[46] = (v371_data + (v363_data * v69_data));
              float v376_data = r1[72];
              r1[72] = (v376_data + (v363_data * v74_data));
              float v381_data = r1[98];
              r1[98] = (v381_data + (v363_data * v79_data));
              float v386_data = r1[124];
              r1[124] = (v386_data + (v363_data * v84_data));
              float v391_data = r1[150];
              r1[150] = (v391_data + (v363_data * v89_data));
              float v393_data = r0[22];
              float v396_data = r1[22];
              r1[22] = (v396_data + (v393_data * v64_data));
              float v401_data = r1[48];
              r1[48] = (v401_data + (v393_data * v69_data));
              float v406_data = r1[74];
              r1[74] = (v406_data + (v393_data * v74_data));
              float v411_data = r1[100];
              r1[100] = (v411_data + (v393_data * v79_data));
              float v416_data = r1[126];
              r1[126] = (v416_data + (v393_data * v84_data));
              float v421_data = r1[152];
              r1[152] = (v421_data + (v393_data * v89_data));
              float v423_data = r0[24];
              float v426_data = r1[24];
              r1[24] = (v426_data + (v423_data * v64_data));
              float v431_data = r1[50];
              r1[50] = (v431_data + (v423_data * v69_data));
              float v436_data = r1[76];
              r1[76] = (v436_data + (v423_data * v74_data));
              float v441_data = r1[102];
              r1[102] = (v441_data + (v423_data * v79_data));
              float v446_data = r1[128];
              r1[128] = (v446_data + (v423_data * v84_data));
              float v451_data = r1[154];
              r1[154] = (v451_data + (v423_data * v89_data));
              float v453_data = r0[1];
              float v456_data = r1[1];
              r1[1] = (v456_data + (v453_data * v64_data));
              float v461_data = r1[27];
              r1[27] = (v461_data + (v453_data * v69_data));
              float v466_data = r1[53];
              r1[53] = (v466_data + (v453_data * v74_data));
              float v471_data = r1[79];
              r1[79] = (v471_data + (v453_data * v79_data));
              float v476_data = r1[105];
              r1[105] = (v476_data + (v453_data * v84_data));
              float v481_data = r1[131];
              r1[131] = (v481_data + (v453_data * v89_data));
              float v483_data = r0[3];
              float v486_data = r1[3];
              r1[3] = (v486_data + (v483_data * v64_data));
              float v491_data = r1[29];
              r1[29] = (v491_data + (v483_data * v69_data));
              float v496_data = r1[55];
              r1[55] = (v496_data + (v483_data * v74_data));
              float v501_data = r1[81];
              r1[81] = (v501_data + (v483_data * v79_data));
              float v506_data = r1[107];
              r1[107] = (v506_data + (v483_data * v84_data));
              float v511_data = r1[133];
              r1[133] = (v511_data + (v483_data * v89_data));
              float v513_data = r0[5];
              float v516_data = r1[5];
              r1[5] = (v516_data + (v513_data * v64_data));
              float v521_data = r1[31];
              r1[31] = (v521_data + (v513_data * v69_data));
              float v526_data = r1[57];
              r1[57] = (v526_data + (v513_data * v74_data));
              float v531_data = r1[83];
              r1[83] = (v531_data + (v513_data * v79_data));
              float v536_data = r1[109];
              r1[109] = (v536_data + (v513_data * v84_data));
              float v541_data = r1[135];
              r1[135] = (v541_data + (v513_data * v89_data));
              float v543_data = r0[7];
              float v546_data = r1[7];
              r1[7] = (v546_data + (v543_data * v64_data));
              float v551_data = r1[33];
              r1[33] = (v551_data + (v543_data * v69_data));
              float v556_data = r1[59];
              r1[59] = (v556_data + (v543_data * v74_data));
              float v561_data = r1[85];
              r1[85] = (v561_data + (v543_data * v79_data));
              float v566_data = r1[111];
              r1[111] = (v566_data + (v543_data * v84_data));
              float v571_data = r1[137];
              r1[137] = (v571_data + (v543_data * v89_data));
              float v573_data = r0[9];
              float v576_data = r1[9];
              r1[9] = (v576_data + (v573_data * v64_data));
              float v581_data = r1[35];
              r1[35] = (v581_data + (v573_data * v69_data));
              float v586_data = r1[61];
              r1[61] = (v586_data + (v573_data * v74_data));
              float v591_data = r1[87];
              r1[87] = (v591_data + (v573_data * v79_data));
              float v596_data = r1[113];
              r1[113] = (v596_data + (v573_data * v84_data));
              float v601_data = r1[139];
              r1[139] = (v601_data + (v573_data * v89_data));
              float v603_data = r0[11];
              float v606_data = r1[11];
              r1[11] = (v606_data + (v603_data * v64_data));
              float v611_data = r1[37];
              r1[37] = (v611_data + (v603_data * v69_data));
              float v616_data = r1[63];
              r1[63] = (v616_data + (v603_data * v74_data));
              float v621_data = r1[89];
              r1[89] = (v621_data + (v603_data * v79_data));
              float v626_data = r1[115];
              r1[115] = (v626_data + (v603_data * v84_data));
              float v631_data = r1[141];
              r1[141] = (v631_data + (v603_data * v89_data));
              float v633_data = r0[13];
              float v636_data = r1[13];
              r1[13] = (v636_data + (v633_data * v64_data));
              float v641_data = r1[39];
              r1[39] = (v641_data + (v633_data * v69_data));
              float v646_data = r1[65];
              r1[65] = (v646_data + (v633_data * v74_data));
              float v651_data = r1[91];
              r1[91] = (v651_data + (v633_data * v79_data));
              float v656_data = r1[117];
              r1[117] = (v656_data + (v633_data * v84_data));
              float v661_data = r1[143];
              r1[143] = (v661_data + (v633_data * v89_data));
              float v663_data = r0[15];
              float v666_data = r1[15];
              r1[15] = (v666_data + (v663_data * v64_data));
              float v671_data = r1[41];
              r1[41] = (v671_data + (v663_data * v69_data));
              float v676_data = r1[67];
              r1[67] = (v676_data + (v663_data * v74_data));
              float v681_data = r1[93];
              r1[93] = (v681_data + (v663_data * v79_data));
              float v686_data = r1[119];
              r1[119] = (v686_data + (v663_data * v84_data));
              float v691_data = r1[145];
              r1[145] = (v691_data + (v663_data * v89_data));
              float v693_data = r0[17];
              float v696_data = r1[17];
              r1[17] = (v696_data + (v693_data * v64_data));
              float v701_data = r1[43];
              r1[43] = (v701_data + (v693_data * v69_data));
              float v706_data = r1[69];
              r1[69] = (v706_data + (v693_data * v74_data));
              float v711_data = r1[95];
              r1[95] = (v711_data + (v693_data * v79_data));
              float v716_data = r1[121];
              r1[121] = (v716_data + (v693_data * v84_data));
              float v721_data = r1[147];
              r1[147] = (v721_data + (v693_data * v89_data));
              float v723_data = r0[19];
              float v726_data = r1[19];
              r1[19] = (v726_data + (v723_data * v64_data));
              float v731_data = r1[45];
              r1[45] = (v731_data + (v723_data * v69_data));
              float v736_data = r1[71];
              r1[71] = (v736_data + (v723_data * v74_data));
              float v741_data = r1[97];
              r1[97] = (v741_data + (v723_data * v79_data));
              float v746_data = r1[123];
              r1[123] = (v746_data + (v723_data * v84_data));
              float v751_data = r1[149];
              r1[149] = (v751_data + (v723_data * v89_data));
              float v753_data = r0[21];
              float v756_data = r1[21];
              r1[21] = (v756_data + (v753_data * v64_data));
              float v761_data = r1[47];
              r1[47] = (v761_data + (v753_data * v69_data));
              float v766_data = r1[73];
              r1[73] = (v766_data + (v753_data * v74_data));
              float v771_data = r1[99];
              r1[99] = (v771_data + (v753_data * v79_data));
              float v776_data = r1[125];
              r1[125] = (v776_data + (v753_data * v84_data));
              float v781_data = r1[151];
              r1[151] = (v781_data + (v753_data * v89_data));
              float v783_data = r0[23];
              float v786_data = r1[23];
              r1[23] = (v786_data + (v783_data * v64_data));
              float v791_data = r1[49];
              r1[49] = (v791_data + (v783_data * v69_data));
              float v796_data = r1[75];
              r1[75] = (v796_data + (v783_data * v74_data));
              float v801_data = r1[101];
              r1[101] = (v801_data + (v783_data * v79_data));
              float v806_data = r1[127];
              r1[127] = (v806_data + (v783_data * v84_data));
              float v811_data = r1[153];
              r1[153] = (v811_data + (v783_data * v89_data));
              float v813_data = r0[25];
              float v816_data = r1[25];
              r1[25] = (v816_data + (v813_data * v64_data));
              float v821_data = r1[51];
              r1[51] = (v821_data + (v813_data * v69_data));
              float v826_data = r1[77];
              r1[77] = (v826_data + (v813_data * v74_data));
              float v831_data = r1[103];
              r1[103] = (v831_data + (v813_data * v79_data));
              float v836_data = r1[129];
              r1[129] = (v836_data + (v813_data * v84_data));
              float v841_data = r1[155];
              r1[155] = (v841_data + (v813_data * v89_data));
              // wait(r2 = load{g>r}(glb_m2););
              float r3[12]{};
              // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(20, 35), (0, 1), (0, 6)] []
              float ir3[12]{};
              if (v6_lead >= 20) {
                float v849_data = r1[24];
                float v850_data = ir3[0];
                ir3[0] = (v850_data + v849_data);
                float v852_data = r1[50];
                float v853_data = ir3[2];
                ir3[2] = (v853_data + v852_data);
                float v855_data = r1[76];
                float v856_data = ir3[4];
                ir3[4] = (v856_data + v855_data);
                float v858_data = r1[102];
                float v859_data = ir3[6];
                ir3[6] = (v859_data + v858_data);
                float v861_data = r1[128];
                float v862_data = ir3[8];
                ir3[8] = (v862_data + v861_data);
                float v864_data = r1[154];
                float v865_data = ir3[10];
                ir3[10] = (v865_data + v864_data);
              }
              if (v6_lead < 3) {
                float v868_data = r1[25];
                float v869_data = ir3[1];
                ir3[1] = (v869_data + v868_data);
                float v871_data = r1[51];
                float v872_data = ir3[3];
                ir3[3] = (v872_data + v871_data);
                float v874_data = r1[77];
                float v875_data = ir3[5];
                ir3[5] = (v875_data + v874_data);
                float v877_data = r1[103];
                float v878_data = ir3[7];
                ir3[7] = (v878_data + v877_data);
                float v880_data = r1[129];
                float v881_data = ir3[9];
                ir3[9] = (v881_data + v880_data);
                float v883_data = r1[155];
                float v884_data = ir3[11];
                ir3[11] = (v884_data + v883_data);
              }
              if (v6_lead >= 20) {
                #pragma unroll
                for (int32_t v890_n1 = 0; v890_n1 < 1; ++v890_n1) {
                  int32_t v892_a = v890_n1 * 2;
                  #pragma unroll
                  for (int32_t v891_n2 = 0; v891_n2 < 6; ++v891_n2) {
                    int32_t v895_a = v892_a + (v891_n2 * 2);
                    float v896_data = ir3[v895_a];
                    float v901_data = r2[v895_a];
                    r3[v895_a] = (v901_data + v896_data);
                  }
                }
              }
              if (v6_lead < 3) {
                #pragma unroll
                for (int32_t v908_n1 = 0; v908_n1 < 1; ++v908_n1) {
                  int32_t v912_a = 1 + (v908_n1 * 2);
                  #pragma unroll
                  for (int32_t v909_n2 = 0; v909_n2 < 6; ++v909_n2) {
                    int32_t v911_a = v909_n2 * 2;
                    float v914_data = ir3[(v912_a + v911_a)];
                    float v919_data = r2[(v912_a + v911_a)];
                    r3[(v912_a + v911_a)] = (v919_data + v914_data);
                  }
                }
              }
              // glb_m2 = store{r>g}(r3);
              if (v6_lead >= 20) {
                #pragma unroll
                for (int32_t v929_i1 = 0; v929_i1 < 1; ++v929_i1) {
                  int32_t v931_a = v929_i1 * 2;
                  int32_t v944_a = v6_lead + ((v929_i1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v930_i2 = 0; v930_i2 < 6; ++v930_i2) {
                    float v935_data = r3[(v931_a + (v930_i2 * 2))];
                    glb_m2[(v944_a + (v930_i2 * 832))] = v935_data;
                  }
                }
              }
              if (v6_lead < 3) {
                int32_t v958_lead = v6_lead + 32_i32;
                #pragma unroll
                for (int32_t v947_i1 = 0; v947_i1 < 1; ++v947_i1) {
                  int32_t v951_a = 1 + (v947_i1 * 2);
                  int32_t v962_a = v958_lead + ((v947_i1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v948_i2 = 0; v948_i2 < 6; ++v948_i2) {
                    float v953_data = r3[(v951_a + (v948_i2 * 2))];
                    glb_m2[(v962_a + (v948_i2 * 832))] = v953_data;
                  }
                }
              }
            }
          }
        }
      });
    }
  });
}


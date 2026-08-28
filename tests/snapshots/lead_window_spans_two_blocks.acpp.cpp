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
              float r0[156]{};
              // r0 = +(glb_m0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              int32_t v6_lead = item.get_local_id(0) % 32;
              float v13_data = glb_m0[v6_lead];
              float v14_data = glb_m1[0];
              float v16_data = r0[0];
              r0[0] = (v16_data + (v13_data * v14_data));
              float v24_data = glb_m0[v6_lead];
              float v25_data = glb_m1[1];
              float v27_data = r0[26];
              r0[26] = (v27_data + (v24_data * v25_data));
              float v35_data = glb_m0[v6_lead];
              float v36_data = glb_m1[2];
              float v38_data = r0[52];
              r0[52] = (v38_data + (v35_data * v36_data));
              float v46_data = glb_m0[v6_lead];
              float v47_data = glb_m1[3];
              float v49_data = r0[78];
              r0[78] = (v49_data + (v46_data * v47_data));
              float v57_data = glb_m0[v6_lead];
              float v58_data = glb_m1[4];
              float v60_data = r0[104];
              r0[104] = (v60_data + (v57_data * v58_data));
              float v68_data = glb_m0[v6_lead];
              float v69_data = glb_m1[5];
              float v71_data = r0[130];
              r0[130] = (v71_data + (v68_data * v69_data));
              float v79_data = glb_m0[(v6_lead + 64)];
              float v82_data = r0[2];
              r0[2] = (v82_data + (v79_data * v14_data));
              float v90_data = glb_m0[(v6_lead + 64)];
              float v93_data = r0[28];
              r0[28] = (v93_data + (v90_data * v25_data));
              float v101_data = glb_m0[(v6_lead + 64)];
              float v104_data = r0[54];
              r0[54] = (v104_data + (v101_data * v36_data));
              float v112_data = glb_m0[(v6_lead + 64)];
              float v115_data = r0[80];
              r0[80] = (v115_data + (v112_data * v47_data));
              float v123_data = glb_m0[(v6_lead + 64)];
              float v126_data = r0[106];
              r0[106] = (v126_data + (v123_data * v58_data));
              float v134_data = glb_m0[(v6_lead + 64)];
              float v137_data = r0[132];
              r0[132] = (v137_data + (v134_data * v69_data));
              float v145_data = glb_m0[(v6_lead + 128)];
              float v148_data = r0[4];
              r0[4] = (v148_data + (v145_data * v14_data));
              float v156_data = glb_m0[(v6_lead + 128)];
              float v159_data = r0[30];
              r0[30] = (v159_data + (v156_data * v25_data));
              float v167_data = glb_m0[(v6_lead + 128)];
              float v170_data = r0[56];
              r0[56] = (v170_data + (v167_data * v36_data));
              float v178_data = glb_m0[(v6_lead + 128)];
              float v181_data = r0[82];
              r0[82] = (v181_data + (v178_data * v47_data));
              float v189_data = glb_m0[(v6_lead + 128)];
              float v192_data = r0[108];
              r0[108] = (v192_data + (v189_data * v58_data));
              float v200_data = glb_m0[(v6_lead + 128)];
              float v203_data = r0[134];
              r0[134] = (v203_data + (v200_data * v69_data));
              float v211_data = glb_m0[(v6_lead + 192)];
              float v214_data = r0[6];
              r0[6] = (v214_data + (v211_data * v14_data));
              float v222_data = glb_m0[(v6_lead + 192)];
              float v225_data = r0[32];
              r0[32] = (v225_data + (v222_data * v25_data));
              float v233_data = glb_m0[(v6_lead + 192)];
              float v236_data = r0[58];
              r0[58] = (v236_data + (v233_data * v36_data));
              float v244_data = glb_m0[(v6_lead + 192)];
              float v247_data = r0[84];
              r0[84] = (v247_data + (v244_data * v47_data));
              float v255_data = glb_m0[(v6_lead + 192)];
              float v258_data = r0[110];
              r0[110] = (v258_data + (v255_data * v58_data));
              float v266_data = glb_m0[(v6_lead + 192)];
              float v269_data = r0[136];
              r0[136] = (v269_data + (v266_data * v69_data));
              float v277_data = glb_m0[(v6_lead + 256)];
              float v280_data = r0[8];
              r0[8] = (v280_data + (v277_data * v14_data));
              float v288_data = glb_m0[(v6_lead + 256)];
              float v291_data = r0[34];
              r0[34] = (v291_data + (v288_data * v25_data));
              float v299_data = glb_m0[(v6_lead + 256)];
              float v302_data = r0[60];
              r0[60] = (v302_data + (v299_data * v36_data));
              float v310_data = glb_m0[(v6_lead + 256)];
              float v313_data = r0[86];
              r0[86] = (v313_data + (v310_data * v47_data));
              float v321_data = glb_m0[(v6_lead + 256)];
              float v324_data = r0[112];
              r0[112] = (v324_data + (v321_data * v58_data));
              float v332_data = glb_m0[(v6_lead + 256)];
              float v335_data = r0[138];
              r0[138] = (v335_data + (v332_data * v69_data));
              float v343_data = glb_m0[(v6_lead + 320)];
              float v346_data = r0[10];
              r0[10] = (v346_data + (v343_data * v14_data));
              float v354_data = glb_m0[(v6_lead + 320)];
              float v357_data = r0[36];
              r0[36] = (v357_data + (v354_data * v25_data));
              float v365_data = glb_m0[(v6_lead + 320)];
              float v368_data = r0[62];
              r0[62] = (v368_data + (v365_data * v36_data));
              float v376_data = glb_m0[(v6_lead + 320)];
              float v379_data = r0[88];
              r0[88] = (v379_data + (v376_data * v47_data));
              float v387_data = glb_m0[(v6_lead + 320)];
              float v390_data = r0[114];
              r0[114] = (v390_data + (v387_data * v58_data));
              float v398_data = glb_m0[(v6_lead + 320)];
              float v401_data = r0[140];
              r0[140] = (v401_data + (v398_data * v69_data));
              float v409_data = glb_m0[(v6_lead + 384)];
              float v412_data = r0[12];
              r0[12] = (v412_data + (v409_data * v14_data));
              float v420_data = glb_m0[(v6_lead + 384)];
              float v423_data = r0[38];
              r0[38] = (v423_data + (v420_data * v25_data));
              float v431_data = glb_m0[(v6_lead + 384)];
              float v434_data = r0[64];
              r0[64] = (v434_data + (v431_data * v36_data));
              float v442_data = glb_m0[(v6_lead + 384)];
              float v445_data = r0[90];
              r0[90] = (v445_data + (v442_data * v47_data));
              float v453_data = glb_m0[(v6_lead + 384)];
              float v456_data = r0[116];
              r0[116] = (v456_data + (v453_data * v58_data));
              float v464_data = glb_m0[(v6_lead + 384)];
              float v467_data = r0[142];
              r0[142] = (v467_data + (v464_data * v69_data));
              float v475_data = glb_m0[(v6_lead + 448)];
              float v478_data = r0[14];
              r0[14] = (v478_data + (v475_data * v14_data));
              float v486_data = glb_m0[(v6_lead + 448)];
              float v489_data = r0[40];
              r0[40] = (v489_data + (v486_data * v25_data));
              float v497_data = glb_m0[(v6_lead + 448)];
              float v500_data = r0[66];
              r0[66] = (v500_data + (v497_data * v36_data));
              float v508_data = glb_m0[(v6_lead + 448)];
              float v511_data = r0[92];
              r0[92] = (v511_data + (v508_data * v47_data));
              float v519_data = glb_m0[(v6_lead + 448)];
              float v522_data = r0[118];
              r0[118] = (v522_data + (v519_data * v58_data));
              float v530_data = glb_m0[(v6_lead + 448)];
              float v533_data = r0[144];
              r0[144] = (v533_data + (v530_data * v69_data));
              float v541_data = glb_m0[(v6_lead + 512)];
              float v544_data = r0[16];
              r0[16] = (v544_data + (v541_data * v14_data));
              float v552_data = glb_m0[(v6_lead + 512)];
              float v555_data = r0[42];
              r0[42] = (v555_data + (v552_data * v25_data));
              float v563_data = glb_m0[(v6_lead + 512)];
              float v566_data = r0[68];
              r0[68] = (v566_data + (v563_data * v36_data));
              float v574_data = glb_m0[(v6_lead + 512)];
              float v577_data = r0[94];
              r0[94] = (v577_data + (v574_data * v47_data));
              float v585_data = glb_m0[(v6_lead + 512)];
              float v588_data = r0[120];
              r0[120] = (v588_data + (v585_data * v58_data));
              float v596_data = glb_m0[(v6_lead + 512)];
              float v599_data = r0[146];
              r0[146] = (v599_data + (v596_data * v69_data));
              float v607_data = glb_m0[(v6_lead + 576)];
              float v610_data = r0[18];
              r0[18] = (v610_data + (v607_data * v14_data));
              float v618_data = glb_m0[(v6_lead + 576)];
              float v621_data = r0[44];
              r0[44] = (v621_data + (v618_data * v25_data));
              float v629_data = glb_m0[(v6_lead + 576)];
              float v632_data = r0[70];
              r0[70] = (v632_data + (v629_data * v36_data));
              float v640_data = glb_m0[(v6_lead + 576)];
              float v643_data = r0[96];
              r0[96] = (v643_data + (v640_data * v47_data));
              float v651_data = glb_m0[(v6_lead + 576)];
              float v654_data = r0[122];
              r0[122] = (v654_data + (v651_data * v58_data));
              float v662_data = glb_m0[(v6_lead + 576)];
              float v665_data = r0[148];
              r0[148] = (v665_data + (v662_data * v69_data));
              float v673_data = glb_m0[(v6_lead + 640)];
              float v676_data = r0[20];
              r0[20] = (v676_data + (v673_data * v14_data));
              float v684_data = glb_m0[(v6_lead + 640)];
              float v687_data = r0[46];
              r0[46] = (v687_data + (v684_data * v25_data));
              float v695_data = glb_m0[(v6_lead + 640)];
              float v698_data = r0[72];
              r0[72] = (v698_data + (v695_data * v36_data));
              float v706_data = glb_m0[(v6_lead + 640)];
              float v709_data = r0[98];
              r0[98] = (v709_data + (v706_data * v47_data));
              float v717_data = glb_m0[(v6_lead + 640)];
              float v720_data = r0[124];
              r0[124] = (v720_data + (v717_data * v58_data));
              float v728_data = glb_m0[(v6_lead + 640)];
              float v731_data = r0[150];
              r0[150] = (v731_data + (v728_data * v69_data));
              float v739_data = glb_m0[(v6_lead + 704)];
              float v742_data = r0[22];
              r0[22] = (v742_data + (v739_data * v14_data));
              float v750_data = glb_m0[(v6_lead + 704)];
              float v753_data = r0[48];
              r0[48] = (v753_data + (v750_data * v25_data));
              float v761_data = glb_m0[(v6_lead + 704)];
              float v764_data = r0[74];
              r0[74] = (v764_data + (v761_data * v36_data));
              float v772_data = glb_m0[(v6_lead + 704)];
              float v775_data = r0[100];
              r0[100] = (v775_data + (v772_data * v47_data));
              float v783_data = glb_m0[(v6_lead + 704)];
              float v786_data = r0[126];
              r0[126] = (v786_data + (v783_data * v58_data));
              float v794_data = glb_m0[(v6_lead + 704)];
              float v797_data = r0[152];
              r0[152] = (v797_data + (v794_data * v69_data));
              float v805_data = glb_m0[(v6_lead + 768)];
              float v808_data = r0[24];
              r0[24] = (v808_data + (v805_data * v14_data));
              float v816_data = glb_m0[(v6_lead + 768)];
              float v819_data = r0[50];
              r0[50] = (v819_data + (v816_data * v25_data));
              float v827_data = glb_m0[(v6_lead + 768)];
              float v830_data = r0[76];
              r0[76] = (v830_data + (v827_data * v36_data));
              float v838_data = glb_m0[(v6_lead + 768)];
              float v841_data = r0[102];
              r0[102] = (v841_data + (v838_data * v47_data));
              float v849_data = glb_m0[(v6_lead + 768)];
              float v852_data = r0[128];
              r0[128] = (v852_data + (v849_data * v58_data));
              float v860_data = glb_m0[(v6_lead + 768)];
              float v863_data = r0[154];
              r0[154] = (v863_data + (v860_data * v69_data));
              float v871_data = glb_m0[(v6_lead + 32_i32)];
              float v874_data = r0[1];
              r0[1] = (v874_data + (v871_data * v14_data));
              float v882_data = glb_m0[(v6_lead + 32_i32)];
              float v885_data = r0[27];
              r0[27] = (v885_data + (v882_data * v25_data));
              float v893_data = glb_m0[(v6_lead + 32_i32)];
              float v896_data = r0[53];
              r0[53] = (v896_data + (v893_data * v36_data));
              float v904_data = glb_m0[(v6_lead + 32_i32)];
              float v907_data = r0[79];
              r0[79] = (v907_data + (v904_data * v47_data));
              float v915_data = glb_m0[(v6_lead + 32_i32)];
              float v918_data = r0[105];
              r0[105] = (v918_data + (v915_data * v58_data));
              float v926_data = glb_m0[(v6_lead + 32_i32)];
              float v929_data = r0[131];
              r0[131] = (v929_data + (v926_data * v69_data));
              float v937_data = glb_m0[((v6_lead + 32_i32) + 64)];
              float v940_data = r0[3];
              r0[3] = (v940_data + (v937_data * v14_data));
              float v948_data = glb_m0[((v6_lead + 32_i32) + 64)];
              float v951_data = r0[29];
              r0[29] = (v951_data + (v948_data * v25_data));
              float v959_data = glb_m0[((v6_lead + 32_i32) + 64)];
              float v962_data = r0[55];
              r0[55] = (v962_data + (v959_data * v36_data));
              float v970_data = glb_m0[((v6_lead + 32_i32) + 64)];
              float v973_data = r0[81];
              r0[81] = (v973_data + (v970_data * v47_data));
              float v981_data = glb_m0[((v6_lead + 32_i32) + 64)];
              float v984_data = r0[107];
              r0[107] = (v984_data + (v981_data * v58_data));
              float v992_data = glb_m0[((v6_lead + 32_i32) + 64)];
              float v995_data = r0[133];
              r0[133] = (v995_data + (v992_data * v69_data));
              float v1003_data = glb_m0[((v6_lead + 32_i32) + 128)];
              float v1006_data = r0[5];
              r0[5] = (v1006_data + (v1003_data * v14_data));
              float v1014_data = glb_m0[((v6_lead + 32_i32) + 128)];
              float v1017_data = r0[31];
              r0[31] = (v1017_data + (v1014_data * v25_data));
              float v1025_data = glb_m0[((v6_lead + 32_i32) + 128)];
              float v1028_data = r0[57];
              r0[57] = (v1028_data + (v1025_data * v36_data));
              float v1036_data = glb_m0[((v6_lead + 32_i32) + 128)];
              float v1039_data = r0[83];
              r0[83] = (v1039_data + (v1036_data * v47_data));
              float v1047_data = glb_m0[((v6_lead + 32_i32) + 128)];
              float v1050_data = r0[109];
              r0[109] = (v1050_data + (v1047_data * v58_data));
              float v1058_data = glb_m0[((v6_lead + 32_i32) + 128)];
              float v1061_data = r0[135];
              r0[135] = (v1061_data + (v1058_data * v69_data));
              float v1069_data = glb_m0[((v6_lead + 32_i32) + 192)];
              float v1072_data = r0[7];
              r0[7] = (v1072_data + (v1069_data * v14_data));
              float v1080_data = glb_m0[((v6_lead + 32_i32) + 192)];
              float v1083_data = r0[33];
              r0[33] = (v1083_data + (v1080_data * v25_data));
              float v1091_data = glb_m0[((v6_lead + 32_i32) + 192)];
              float v1094_data = r0[59];
              r0[59] = (v1094_data + (v1091_data * v36_data));
              float v1102_data = glb_m0[((v6_lead + 32_i32) + 192)];
              float v1105_data = r0[85];
              r0[85] = (v1105_data + (v1102_data * v47_data));
              float v1113_data = glb_m0[((v6_lead + 32_i32) + 192)];
              float v1116_data = r0[111];
              r0[111] = (v1116_data + (v1113_data * v58_data));
              float v1124_data = glb_m0[((v6_lead + 32_i32) + 192)];
              float v1127_data = r0[137];
              r0[137] = (v1127_data + (v1124_data * v69_data));
              float v1135_data = glb_m0[((v6_lead + 32_i32) + 256)];
              float v1138_data = r0[9];
              r0[9] = (v1138_data + (v1135_data * v14_data));
              float v1146_data = glb_m0[((v6_lead + 32_i32) + 256)];
              float v1149_data = r0[35];
              r0[35] = (v1149_data + (v1146_data * v25_data));
              float v1157_data = glb_m0[((v6_lead + 32_i32) + 256)];
              float v1160_data = r0[61];
              r0[61] = (v1160_data + (v1157_data * v36_data));
              float v1168_data = glb_m0[((v6_lead + 32_i32) + 256)];
              float v1171_data = r0[87];
              r0[87] = (v1171_data + (v1168_data * v47_data));
              float v1179_data = glb_m0[((v6_lead + 32_i32) + 256)];
              float v1182_data = r0[113];
              r0[113] = (v1182_data + (v1179_data * v58_data));
              float v1190_data = glb_m0[((v6_lead + 32_i32) + 256)];
              float v1193_data = r0[139];
              r0[139] = (v1193_data + (v1190_data * v69_data));
              float v1201_data = glb_m0[((v6_lead + 32_i32) + 320)];
              float v1204_data = r0[11];
              r0[11] = (v1204_data + (v1201_data * v14_data));
              float v1212_data = glb_m0[((v6_lead + 32_i32) + 320)];
              float v1215_data = r0[37];
              r0[37] = (v1215_data + (v1212_data * v25_data));
              float v1223_data = glb_m0[((v6_lead + 32_i32) + 320)];
              float v1226_data = r0[63];
              r0[63] = (v1226_data + (v1223_data * v36_data));
              float v1234_data = glb_m0[((v6_lead + 32_i32) + 320)];
              float v1237_data = r0[89];
              r0[89] = (v1237_data + (v1234_data * v47_data));
              float v1245_data = glb_m0[((v6_lead + 32_i32) + 320)];
              float v1248_data = r0[115];
              r0[115] = (v1248_data + (v1245_data * v58_data));
              float v1256_data = glb_m0[((v6_lead + 32_i32) + 320)];
              float v1259_data = r0[141];
              r0[141] = (v1259_data + (v1256_data * v69_data));
              float v1267_data = glb_m0[((v6_lead + 32_i32) + 384)];
              float v1270_data = r0[13];
              r0[13] = (v1270_data + (v1267_data * v14_data));
              float v1278_data = glb_m0[((v6_lead + 32_i32) + 384)];
              float v1281_data = r0[39];
              r0[39] = (v1281_data + (v1278_data * v25_data));
              float v1289_data = glb_m0[((v6_lead + 32_i32) + 384)];
              float v1292_data = r0[65];
              r0[65] = (v1292_data + (v1289_data * v36_data));
              float v1300_data = glb_m0[((v6_lead + 32_i32) + 384)];
              float v1303_data = r0[91];
              r0[91] = (v1303_data + (v1300_data * v47_data));
              float v1311_data = glb_m0[((v6_lead + 32_i32) + 384)];
              float v1314_data = r0[117];
              r0[117] = (v1314_data + (v1311_data * v58_data));
              float v1322_data = glb_m0[((v6_lead + 32_i32) + 384)];
              float v1325_data = r0[143];
              r0[143] = (v1325_data + (v1322_data * v69_data));
              float v1333_data = glb_m0[((v6_lead + 32_i32) + 448)];
              float v1336_data = r0[15];
              r0[15] = (v1336_data + (v1333_data * v14_data));
              float v1344_data = glb_m0[((v6_lead + 32_i32) + 448)];
              float v1347_data = r0[41];
              r0[41] = (v1347_data + (v1344_data * v25_data));
              float v1355_data = glb_m0[((v6_lead + 32_i32) + 448)];
              float v1358_data = r0[67];
              r0[67] = (v1358_data + (v1355_data * v36_data));
              float v1366_data = glb_m0[((v6_lead + 32_i32) + 448)];
              float v1369_data = r0[93];
              r0[93] = (v1369_data + (v1366_data * v47_data));
              float v1377_data = glb_m0[((v6_lead + 32_i32) + 448)];
              float v1380_data = r0[119];
              r0[119] = (v1380_data + (v1377_data * v58_data));
              float v1388_data = glb_m0[((v6_lead + 32_i32) + 448)];
              float v1391_data = r0[145];
              r0[145] = (v1391_data + (v1388_data * v69_data));
              float v1399_data = glb_m0[((v6_lead + 32_i32) + 512)];
              float v1402_data = r0[17];
              r0[17] = (v1402_data + (v1399_data * v14_data));
              float v1410_data = glb_m0[((v6_lead + 32_i32) + 512)];
              float v1413_data = r0[43];
              r0[43] = (v1413_data + (v1410_data * v25_data));
              float v1421_data = glb_m0[((v6_lead + 32_i32) + 512)];
              float v1424_data = r0[69];
              r0[69] = (v1424_data + (v1421_data * v36_data));
              float v1432_data = glb_m0[((v6_lead + 32_i32) + 512)];
              float v1435_data = r0[95];
              r0[95] = (v1435_data + (v1432_data * v47_data));
              float v1443_data = glb_m0[((v6_lead + 32_i32) + 512)];
              float v1446_data = r0[121];
              r0[121] = (v1446_data + (v1443_data * v58_data));
              float v1454_data = glb_m0[((v6_lead + 32_i32) + 512)];
              float v1457_data = r0[147];
              r0[147] = (v1457_data + (v1454_data * v69_data));
              float v1465_data = glb_m0[((v6_lead + 32_i32) + 576)];
              float v1468_data = r0[19];
              r0[19] = (v1468_data + (v1465_data * v14_data));
              float v1476_data = glb_m0[((v6_lead + 32_i32) + 576)];
              float v1479_data = r0[45];
              r0[45] = (v1479_data + (v1476_data * v25_data));
              float v1487_data = glb_m0[((v6_lead + 32_i32) + 576)];
              float v1490_data = r0[71];
              r0[71] = (v1490_data + (v1487_data * v36_data));
              float v1498_data = glb_m0[((v6_lead + 32_i32) + 576)];
              float v1501_data = r0[97];
              r0[97] = (v1501_data + (v1498_data * v47_data));
              float v1509_data = glb_m0[((v6_lead + 32_i32) + 576)];
              float v1512_data = r0[123];
              r0[123] = (v1512_data + (v1509_data * v58_data));
              float v1520_data = glb_m0[((v6_lead + 32_i32) + 576)];
              float v1523_data = r0[149];
              r0[149] = (v1523_data + (v1520_data * v69_data));
              float v1531_data = glb_m0[((v6_lead + 32_i32) + 640)];
              float v1534_data = r0[21];
              r0[21] = (v1534_data + (v1531_data * v14_data));
              float v1542_data = glb_m0[((v6_lead + 32_i32) + 640)];
              float v1545_data = r0[47];
              r0[47] = (v1545_data + (v1542_data * v25_data));
              float v1553_data = glb_m0[((v6_lead + 32_i32) + 640)];
              float v1556_data = r0[73];
              r0[73] = (v1556_data + (v1553_data * v36_data));
              float v1564_data = glb_m0[((v6_lead + 32_i32) + 640)];
              float v1567_data = r0[99];
              r0[99] = (v1567_data + (v1564_data * v47_data));
              float v1575_data = glb_m0[((v6_lead + 32_i32) + 640)];
              float v1578_data = r0[125];
              r0[125] = (v1578_data + (v1575_data * v58_data));
              float v1586_data = glb_m0[((v6_lead + 32_i32) + 640)];
              float v1589_data = r0[151];
              r0[151] = (v1589_data + (v1586_data * v69_data));
              float v1597_data = glb_m0[((v6_lead + 32_i32) + 704)];
              float v1600_data = r0[23];
              r0[23] = (v1600_data + (v1597_data * v14_data));
              float v1608_data = glb_m0[((v6_lead + 32_i32) + 704)];
              float v1611_data = r0[49];
              r0[49] = (v1611_data + (v1608_data * v25_data));
              float v1619_data = glb_m0[((v6_lead + 32_i32) + 704)];
              float v1622_data = r0[75];
              r0[75] = (v1622_data + (v1619_data * v36_data));
              float v1630_data = glb_m0[((v6_lead + 32_i32) + 704)];
              float v1633_data = r0[101];
              r0[101] = (v1633_data + (v1630_data * v47_data));
              float v1641_data = glb_m0[((v6_lead + 32_i32) + 704)];
              float v1644_data = r0[127];
              r0[127] = (v1644_data + (v1641_data * v58_data));
              float v1652_data = glb_m0[((v6_lead + 32_i32) + 704)];
              float v1655_data = r0[153];
              r0[153] = (v1655_data + (v1652_data * v69_data));
              float v1663_data = glb_m0[((v6_lead + 32_i32) + 768)];
              float v1666_data = r0[25];
              r0[25] = (v1666_data + (v1663_data * v14_data));
              float v1674_data = glb_m0[((v6_lead + 32_i32) + 768)];
              float v1677_data = r0[51];
              r0[51] = (v1677_data + (v1674_data * v25_data));
              float v1685_data = glb_m0[((v6_lead + 32_i32) + 768)];
              float v1688_data = r0[77];
              r0[77] = (v1688_data + (v1685_data * v36_data));
              float v1696_data = glb_m0[((v6_lead + 32_i32) + 768)];
              float v1699_data = r0[103];
              r0[103] = (v1699_data + (v1696_data * v47_data));
              float v1707_data = glb_m0[((v6_lead + 32_i32) + 768)];
              float v1710_data = r0[129];
              r0[129] = (v1710_data + (v1707_data * v58_data));
              float v1718_data = glb_m0[((v6_lead + 32_i32) + 768)];
              float v1721_data = r0[155];
              r0[155] = (v1721_data + (v1718_data * v69_data));
              float r1[12]{};
              // r1 = +(r0) + name: glb_m2, type: SymbolType.Global, lead: [0]
              // [(20, 35), (0, 1), (0, 6)] []
              float ir1[12]{};
              if (v6_lead >= 20) {
                float v1729_data = r0[24];
                float v1730_data = ir1[0];
                ir1[0] = (v1730_data + v1729_data);
                float v1732_data = r0[50];
                float v1733_data = ir1[2];
                ir1[2] = (v1733_data + v1732_data);
                float v1735_data = r0[76];
                float v1736_data = ir1[4];
                ir1[4] = (v1736_data + v1735_data);
                float v1738_data = r0[102];
                float v1739_data = ir1[6];
                ir1[6] = (v1739_data + v1738_data);
                float v1741_data = r0[128];
                float v1742_data = ir1[8];
                ir1[8] = (v1742_data + v1741_data);
                float v1744_data = r0[154];
                float v1745_data = ir1[10];
                ir1[10] = (v1745_data + v1744_data);
              }
              if (v6_lead < 3) {
                float v1748_data = r0[25];
                float v1749_data = ir1[1];
                ir1[1] = (v1749_data + v1748_data);
                float v1751_data = r0[51];
                float v1752_data = ir1[3];
                ir1[3] = (v1752_data + v1751_data);
                float v1754_data = r0[77];
                float v1755_data = ir1[5];
                ir1[5] = (v1755_data + v1754_data);
                float v1757_data = r0[103];
                float v1758_data = ir1[7];
                ir1[7] = (v1758_data + v1757_data);
                float v1760_data = r0[129];
                float v1761_data = ir1[9];
                ir1[9] = (v1761_data + v1760_data);
                float v1763_data = r0[155];
                float v1764_data = ir1[11];
                ir1[11] = (v1764_data + v1763_data);
              }
              if (v6_lead >= 20) {
                #pragma unroll
                for (int32_t v1770_n1 = 0; v1770_n1 < 1; ++v1770_n1) {
                  int32_t v1772_a = v1770_n1 * 2;
                  int32_t v1785_a = v6_lead + ((v1770_n1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v1771_n2 = 0; v1771_n2 < 6; ++v1771_n2) {
                    int32_t v1775_a = v1772_a + (v1771_n2 * 2);
                    float v1776_data = ir1[v1775_a];
                    float v1787_data = glb_m2[(v1785_a + (v1771_n2 * 832))];
                    r1[v1775_a] = (v1787_data + v1776_data);
                  }
                }
              }
              if (v6_lead < 3) {
                int32_t v1805_lead = v6_lead + 32_i32;
                #pragma unroll
                for (int32_t v1794_n1 = 0; v1794_n1 < 1; ++v1794_n1) {
                  int32_t v1798_a = 1 + (v1794_n1 * 2);
                  int32_t v1809_a = v1805_lead + ((v1794_n1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v1795_n2 = 0; v1795_n2 < 6; ++v1795_n2) {
                    int32_t v1797_a = v1795_n2 * 2;
                    float v1800_data = ir1[(v1798_a + v1797_a)];
                    float v1811_data = glb_m2[(v1809_a + (v1795_n2 * 832))];
                    r1[(v1798_a + v1797_a)] = (v1811_data + v1800_data);
                  }
                }
              }
              // glb_m2 = store{r>g}(r1);
              if (v6_lead >= 20) {
                #pragma unroll
                for (int32_t v1821_i1 = 0; v1821_i1 < 1; ++v1821_i1) {
                  int32_t v1823_a = v1821_i1 * 2;
                  int32_t v1836_a = v6_lead + ((v1821_i1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v1822_i2 = 0; v1822_i2 < 6; ++v1822_i2) {
                    float v1827_data = r1[(v1823_a + (v1822_i2 * 2))];
                    glb_m2[(v1836_a + (v1822_i2 * 832))] = v1827_data;
                  }
                }
              }
              if (v6_lead < 3) {
                int32_t v1850_lead = v6_lead + 32_i32;
                #pragma unroll
                for (int32_t v1839_i1 = 0; v1839_i1 < 1; ++v1839_i1) {
                  int32_t v1843_a = 1 + (v1839_i1 * 2);
                  int32_t v1854_a = v1850_lead + ((v1839_i1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v1840_i2 = 0; v1840_i2 < 6; ++v1840_i2) {
                    float v1845_data = r1[(v1843_a + (v1840_i2 * 2))];
                    glb_m2[(v1854_a + (v1840_i2 * 832))] = v1845_data;
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


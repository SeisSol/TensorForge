// === base name ===
kernel_ead773dd51

// === header ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ead773dd51(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ead773dd51(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(32×16) {0..32}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(32×16) {0..32}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
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
                int32_t v16_a = (v8_lead + 4) + 0;
                float v24_data = glb_m1[(v8_lead + 4)];
                float v25_data = s0[0];
                float v27_data = ir0[0];
                ir0[0] = (v27_data + (v24_data * v25_data));
                int32_t v35_a = (v8_lead + 4) + 0;
                float v43_data = glb_m1[(v8_lead + 4)];
                float v44_data = s0[16];
                float v46_data = ir0[1];
                ir0[1] = (v46_data + (v43_data * v44_data));
                int32_t v54_a = (v8_lead + 4) + 0;
                float v62_data = glb_m1[(v8_lead + 4)];
                float v63_data = s0[32];
                float v65_data = ir0[2];
                ir0[2] = (v65_data + (v62_data * v63_data));
                int32_t v73_a = (v8_lead + 4) + 0;
                float v81_data = glb_m1[(v8_lead + 4)];
                float v82_data = s0[48];
                float v84_data = ir0[3];
                ir0[3] = (v84_data + (v81_data * v82_data));
                int32_t v92_a = (v8_lead + 4) + 0;
                float v100_data = glb_m1[(v8_lead + 4)];
                float v101_data = s0[64];
                float v103_data = ir0[4];
                ir0[4] = (v103_data + (v100_data * v101_data));
                int32_t v111_a = (v8_lead + 4) + 0;
                float v119_data = glb_m1[(v8_lead + 4)];
                float v120_data = s0[80];
                float v122_data = ir0[5];
                ir0[5] = (v122_data + (v119_data * v120_data));
                int32_t v130_a = (v8_lead + 4) + 0;
                float v138_data = glb_m1[(v8_lead + 4)];
                float v139_data = s0[96];
                float v141_data = ir0[6];
                ir0[6] = (v141_data + (v138_data * v139_data));
                int32_t v149_a = (v8_lead + 4) + 0;
                float v157_data = glb_m1[(v8_lead + 4)];
                float v158_data = s0[112];
                float v160_data = ir0[7];
                ir0[7] = (v160_data + (v157_data * v158_data));
              }
              if (v8_lead < 12) {
                int32_t v172_a = (v8_lead + 4) + 32;
                float v180_data = glb_m1[((v8_lead + 4) + 32)];
                float v181_data = s0[1];
                float v183_data = ir0[0];
                ir0[0] = (v183_data + (v180_data * v181_data));
                int32_t v191_a = (v8_lead + 4) + 32;
                float v199_data = glb_m1[((v8_lead + 4) + 32)];
                float v200_data = s0[17];
                float v202_data = ir0[1];
                ir0[1] = (v202_data + (v199_data * v200_data));
                int32_t v210_a = (v8_lead + 4) + 32;
                float v218_data = glb_m1[((v8_lead + 4) + 32)];
                float v219_data = s0[33];
                float v221_data = ir0[2];
                ir0[2] = (v221_data + (v218_data * v219_data));
                int32_t v229_a = (v8_lead + 4) + 32;
                float v237_data = glb_m1[((v8_lead + 4) + 32)];
                float v238_data = s0[49];
                float v240_data = ir0[3];
                ir0[3] = (v240_data + (v237_data * v238_data));
                int32_t v248_a = (v8_lead + 4) + 32;
                float v256_data = glb_m1[((v8_lead + 4) + 32)];
                float v257_data = s0[65];
                float v259_data = ir0[4];
                ir0[4] = (v259_data + (v256_data * v257_data));
                int32_t v267_a = (v8_lead + 4) + 32;
                float v275_data = glb_m1[((v8_lead + 4) + 32)];
                float v276_data = s0[81];
                float v278_data = ir0[5];
                ir0[5] = (v278_data + (v275_data * v276_data));
                int32_t v286_a = (v8_lead + 4) + 32;
                float v294_data = glb_m1[((v8_lead + 4) + 32)];
                float v295_data = s0[97];
                float v297_data = ir0[6];
                ir0[6] = (v297_data + (v294_data * v295_data));
                int32_t v305_a = (v8_lead + 4) + 32;
                float v313_data = glb_m1[((v8_lead + 4) + 32)];
                float v314_data = s0[113];
                float v316_data = ir0[7];
                ir0[7] = (v316_data + (v313_data * v314_data));
              }
              if (v8_lead < 12) {
                int32_t v328_a = (v8_lead + 4) + 64;
                float v336_data = glb_m1[((v8_lead + 4) + 64)];
                float v337_data = s0[2];
                float v339_data = ir0[0];
                ir0[0] = (v339_data + (v336_data * v337_data));
                int32_t v347_a = (v8_lead + 4) + 64;
                float v355_data = glb_m1[((v8_lead + 4) + 64)];
                float v356_data = s0[18];
                float v358_data = ir0[1];
                ir0[1] = (v358_data + (v355_data * v356_data));
                int32_t v366_a = (v8_lead + 4) + 64;
                float v374_data = glb_m1[((v8_lead + 4) + 64)];
                float v375_data = s0[34];
                float v377_data = ir0[2];
                ir0[2] = (v377_data + (v374_data * v375_data));
                int32_t v385_a = (v8_lead + 4) + 64;
                float v393_data = glb_m1[((v8_lead + 4) + 64)];
                float v394_data = s0[50];
                float v396_data = ir0[3];
                ir0[3] = (v396_data + (v393_data * v394_data));
                int32_t v404_a = (v8_lead + 4) + 64;
                float v412_data = glb_m1[((v8_lead + 4) + 64)];
                float v413_data = s0[66];
                float v415_data = ir0[4];
                ir0[4] = (v415_data + (v412_data * v413_data));
                int32_t v423_a = (v8_lead + 4) + 64;
                float v431_data = glb_m1[((v8_lead + 4) + 64)];
                float v432_data = s0[82];
                float v434_data = ir0[5];
                ir0[5] = (v434_data + (v431_data * v432_data));
                int32_t v442_a = (v8_lead + 4) + 64;
                float v450_data = glb_m1[((v8_lead + 4) + 64)];
                float v451_data = s0[98];
                float v453_data = ir0[6];
                ir0[6] = (v453_data + (v450_data * v451_data));
                int32_t v461_a = (v8_lead + 4) + 64;
                float v469_data = glb_m1[((v8_lead + 4) + 64)];
                float v470_data = s0[114];
                float v472_data = ir0[7];
                ir0[7] = (v472_data + (v469_data * v470_data));
              }
              if (v8_lead < 12) {
                int32_t v484_a = (v8_lead + 4) + 96;
                float v492_data = glb_m1[((v8_lead + 4) + 96)];
                float v493_data = s0[3];
                float v495_data = ir0[0];
                ir0[0] = (v495_data + (v492_data * v493_data));
                int32_t v503_a = (v8_lead + 4) + 96;
                float v511_data = glb_m1[((v8_lead + 4) + 96)];
                float v512_data = s0[19];
                float v514_data = ir0[1];
                ir0[1] = (v514_data + (v511_data * v512_data));
                int32_t v522_a = (v8_lead + 4) + 96;
                float v530_data = glb_m1[((v8_lead + 4) + 96)];
                float v531_data = s0[35];
                float v533_data = ir0[2];
                ir0[2] = (v533_data + (v530_data * v531_data));
                int32_t v541_a = (v8_lead + 4) + 96;
                float v549_data = glb_m1[((v8_lead + 4) + 96)];
                float v550_data = s0[51];
                float v552_data = ir0[3];
                ir0[3] = (v552_data + (v549_data * v550_data));
                int32_t v560_a = (v8_lead + 4) + 96;
                float v568_data = glb_m1[((v8_lead + 4) + 96)];
                float v569_data = s0[67];
                float v571_data = ir0[4];
                ir0[4] = (v571_data + (v568_data * v569_data));
                int32_t v579_a = (v8_lead + 4) + 96;
                float v587_data = glb_m1[((v8_lead + 4) + 96)];
                float v588_data = s0[83];
                float v590_data = ir0[5];
                ir0[5] = (v590_data + (v587_data * v588_data));
                int32_t v598_a = (v8_lead + 4) + 96;
                float v606_data = glb_m1[((v8_lead + 4) + 96)];
                float v607_data = s0[99];
                float v609_data = ir0[6];
                ir0[6] = (v609_data + (v606_data * v607_data));
                int32_t v617_a = (v8_lead + 4) + 96;
                float v625_data = glb_m1[((v8_lead + 4) + 96)];
                float v626_data = s0[115];
                float v628_data = ir0[7];
                ir0[7] = (v628_data + (v625_data * v626_data));
              }
              if (v8_lead < 12) {
                int32_t v640_a = (v8_lead + 4) + 128;
                float v648_data = glb_m1[((v8_lead + 4) + 128)];
                float v649_data = s0[4];
                float v651_data = ir0[0];
                ir0[0] = (v651_data + (v648_data * v649_data));
                int32_t v659_a = (v8_lead + 4) + 128;
                float v667_data = glb_m1[((v8_lead + 4) + 128)];
                float v668_data = s0[20];
                float v670_data = ir0[1];
                ir0[1] = (v670_data + (v667_data * v668_data));
                int32_t v678_a = (v8_lead + 4) + 128;
                float v686_data = glb_m1[((v8_lead + 4) + 128)];
                float v687_data = s0[36];
                float v689_data = ir0[2];
                ir0[2] = (v689_data + (v686_data * v687_data));
                int32_t v697_a = (v8_lead + 4) + 128;
                float v705_data = glb_m1[((v8_lead + 4) + 128)];
                float v706_data = s0[52];
                float v708_data = ir0[3];
                ir0[3] = (v708_data + (v705_data * v706_data));
                int32_t v716_a = (v8_lead + 4) + 128;
                float v724_data = glb_m1[((v8_lead + 4) + 128)];
                float v725_data = s0[68];
                float v727_data = ir0[4];
                ir0[4] = (v727_data + (v724_data * v725_data));
                int32_t v735_a = (v8_lead + 4) + 128;
                float v743_data = glb_m1[((v8_lead + 4) + 128)];
                float v744_data = s0[84];
                float v746_data = ir0[5];
                ir0[5] = (v746_data + (v743_data * v744_data));
                int32_t v754_a = (v8_lead + 4) + 128;
                float v762_data = glb_m1[((v8_lead + 4) + 128)];
                float v763_data = s0[100];
                float v765_data = ir0[6];
                ir0[6] = (v765_data + (v762_data * v763_data));
                int32_t v773_a = (v8_lead + 4) + 128;
                float v781_data = glb_m1[((v8_lead + 4) + 128)];
                float v782_data = s0[116];
                float v784_data = ir0[7];
                ir0[7] = (v784_data + (v781_data * v782_data));
              }
              if (v8_lead < 12) {
                int32_t v796_a = (v8_lead + 4) + 160;
                float v804_data = glb_m1[((v8_lead + 4) + 160)];
                float v805_data = s0[5];
                float v807_data = ir0[0];
                ir0[0] = (v807_data + (v804_data * v805_data));
                int32_t v815_a = (v8_lead + 4) + 160;
                float v823_data = glb_m1[((v8_lead + 4) + 160)];
                float v824_data = s0[21];
                float v826_data = ir0[1];
                ir0[1] = (v826_data + (v823_data * v824_data));
                int32_t v834_a = (v8_lead + 4) + 160;
                float v842_data = glb_m1[((v8_lead + 4) + 160)];
                float v843_data = s0[37];
                float v845_data = ir0[2];
                ir0[2] = (v845_data + (v842_data * v843_data));
                int32_t v853_a = (v8_lead + 4) + 160;
                float v861_data = glb_m1[((v8_lead + 4) + 160)];
                float v862_data = s0[53];
                float v864_data = ir0[3];
                ir0[3] = (v864_data + (v861_data * v862_data));
                int32_t v872_a = (v8_lead + 4) + 160;
                float v880_data = glb_m1[((v8_lead + 4) + 160)];
                float v881_data = s0[69];
                float v883_data = ir0[4];
                ir0[4] = (v883_data + (v880_data * v881_data));
                int32_t v891_a = (v8_lead + 4) + 160;
                float v899_data = glb_m1[((v8_lead + 4) + 160)];
                float v900_data = s0[85];
                float v902_data = ir0[5];
                ir0[5] = (v902_data + (v899_data * v900_data));
                int32_t v910_a = (v8_lead + 4) + 160;
                float v918_data = glb_m1[((v8_lead + 4) + 160)];
                float v919_data = s0[101];
                float v921_data = ir0[6];
                ir0[6] = (v921_data + (v918_data * v919_data));
                int32_t v929_a = (v8_lead + 4) + 160;
                float v937_data = glb_m1[((v8_lead + 4) + 160)];
                float v938_data = s0[117];
                float v940_data = ir0[7];
                ir0[7] = (v940_data + (v937_data * v938_data));
              }
              if (v8_lead < 12) {
                int32_t v952_a = (v8_lead + 4) + 192;
                float v960_data = glb_m1[((v8_lead + 4) + 192)];
                float v961_data = s0[6];
                float v963_data = ir0[0];
                ir0[0] = (v963_data + (v960_data * v961_data));
                int32_t v971_a = (v8_lead + 4) + 192;
                float v979_data = glb_m1[((v8_lead + 4) + 192)];
                float v980_data = s0[22];
                float v982_data = ir0[1];
                ir0[1] = (v982_data + (v979_data * v980_data));
                int32_t v990_a = (v8_lead + 4) + 192;
                float v998_data = glb_m1[((v8_lead + 4) + 192)];
                float v999_data = s0[38];
                float v1001_data = ir0[2];
                ir0[2] = (v1001_data + (v998_data * v999_data));
                int32_t v1009_a = (v8_lead + 4) + 192;
                float v1017_data = glb_m1[((v8_lead + 4) + 192)];
                float v1018_data = s0[54];
                float v1020_data = ir0[3];
                ir0[3] = (v1020_data + (v1017_data * v1018_data));
                int32_t v1028_a = (v8_lead + 4) + 192;
                float v1036_data = glb_m1[((v8_lead + 4) + 192)];
                float v1037_data = s0[70];
                float v1039_data = ir0[4];
                ir0[4] = (v1039_data + (v1036_data * v1037_data));
                int32_t v1047_a = (v8_lead + 4) + 192;
                float v1055_data = glb_m1[((v8_lead + 4) + 192)];
                float v1056_data = s0[86];
                float v1058_data = ir0[5];
                ir0[5] = (v1058_data + (v1055_data * v1056_data));
                int32_t v1066_a = (v8_lead + 4) + 192;
                float v1074_data = glb_m1[((v8_lead + 4) + 192)];
                float v1075_data = s0[102];
                float v1077_data = ir0[6];
                ir0[6] = (v1077_data + (v1074_data * v1075_data));
                int32_t v1085_a = (v8_lead + 4) + 192;
                float v1093_data = glb_m1[((v8_lead + 4) + 192)];
                float v1094_data = s0[118];
                float v1096_data = ir0[7];
                ir0[7] = (v1096_data + (v1093_data * v1094_data));
              }
              if (v8_lead < 12) {
                int32_t v1108_a = (v8_lead + 4) + 224;
                float v1116_data = glb_m1[((v8_lead + 4) + 224)];
                float v1117_data = s0[7];
                float v1119_data = ir0[0];
                ir0[0] = (v1119_data + (v1116_data * v1117_data));
                int32_t v1127_a = (v8_lead + 4) + 224;
                float v1135_data = glb_m1[((v8_lead + 4) + 224)];
                float v1136_data = s0[23];
                float v1138_data = ir0[1];
                ir0[1] = (v1138_data + (v1135_data * v1136_data));
                int32_t v1146_a = (v8_lead + 4) + 224;
                float v1154_data = glb_m1[((v8_lead + 4) + 224)];
                float v1155_data = s0[39];
                float v1157_data = ir0[2];
                ir0[2] = (v1157_data + (v1154_data * v1155_data));
                int32_t v1165_a = (v8_lead + 4) + 224;
                float v1173_data = glb_m1[((v8_lead + 4) + 224)];
                float v1174_data = s0[55];
                float v1176_data = ir0[3];
                ir0[3] = (v1176_data + (v1173_data * v1174_data));
                int32_t v1184_a = (v8_lead + 4) + 224;
                float v1192_data = glb_m1[((v8_lead + 4) + 224)];
                float v1193_data = s0[71];
                float v1195_data = ir0[4];
                ir0[4] = (v1195_data + (v1192_data * v1193_data));
                int32_t v1203_a = (v8_lead + 4) + 224;
                float v1211_data = glb_m1[((v8_lead + 4) + 224)];
                float v1212_data = s0[87];
                float v1214_data = ir0[5];
                ir0[5] = (v1214_data + (v1211_data * v1212_data));
                int32_t v1222_a = (v8_lead + 4) + 224;
                float v1230_data = glb_m1[((v8_lead + 4) + 224)];
                float v1231_data = s0[103];
                float v1233_data = ir0[6];
                ir0[6] = (v1233_data + (v1230_data * v1231_data));
                int32_t v1241_a = (v8_lead + 4) + 224;
                float v1249_data = glb_m1[((v8_lead + 4) + 224)];
                float v1250_data = s0[119];
                float v1252_data = ir0[7];
                ir0[7] = (v1252_data + (v1249_data * v1250_data));
              }
              if (v8_lead < 12) {
                int32_t v1264_a = (v8_lead + 4) + 256;
                float v1272_data = glb_m1[((v8_lead + 4) + 256)];
                float v1273_data = s0[8];
                float v1275_data = ir0[0];
                ir0[0] = (v1275_data + (v1272_data * v1273_data));
                int32_t v1283_a = (v8_lead + 4) + 256;
                float v1291_data = glb_m1[((v8_lead + 4) + 256)];
                float v1292_data = s0[24];
                float v1294_data = ir0[1];
                ir0[1] = (v1294_data + (v1291_data * v1292_data));
                int32_t v1302_a = (v8_lead + 4) + 256;
                float v1310_data = glb_m1[((v8_lead + 4) + 256)];
                float v1311_data = s0[40];
                float v1313_data = ir0[2];
                ir0[2] = (v1313_data + (v1310_data * v1311_data));
                int32_t v1321_a = (v8_lead + 4) + 256;
                float v1329_data = glb_m1[((v8_lead + 4) + 256)];
                float v1330_data = s0[56];
                float v1332_data = ir0[3];
                ir0[3] = (v1332_data + (v1329_data * v1330_data));
                int32_t v1340_a = (v8_lead + 4) + 256;
                float v1348_data = glb_m1[((v8_lead + 4) + 256)];
                float v1349_data = s0[72];
                float v1351_data = ir0[4];
                ir0[4] = (v1351_data + (v1348_data * v1349_data));
                int32_t v1359_a = (v8_lead + 4) + 256;
                float v1367_data = glb_m1[((v8_lead + 4) + 256)];
                float v1368_data = s0[88];
                float v1370_data = ir0[5];
                ir0[5] = (v1370_data + (v1367_data * v1368_data));
                int32_t v1378_a = (v8_lead + 4) + 256;
                float v1386_data = glb_m1[((v8_lead + 4) + 256)];
                float v1387_data = s0[104];
                float v1389_data = ir0[6];
                ir0[6] = (v1389_data + (v1386_data * v1387_data));
                int32_t v1397_a = (v8_lead + 4) + 256;
                float v1405_data = glb_m1[((v8_lead + 4) + 256)];
                float v1406_data = s0[120];
                float v1408_data = ir0[7];
                ir0[7] = (v1408_data + (v1405_data * v1406_data));
              }
              if (v8_lead < 12) {
                int32_t v1420_a = (v8_lead + 4) + 288;
                float v1428_data = glb_m1[((v8_lead + 4) + 288)];
                float v1429_data = s0[9];
                float v1431_data = ir0[0];
                ir0[0] = (v1431_data + (v1428_data * v1429_data));
                int32_t v1439_a = (v8_lead + 4) + 288;
                float v1447_data = glb_m1[((v8_lead + 4) + 288)];
                float v1448_data = s0[25];
                float v1450_data = ir0[1];
                ir0[1] = (v1450_data + (v1447_data * v1448_data));
                int32_t v1458_a = (v8_lead + 4) + 288;
                float v1466_data = glb_m1[((v8_lead + 4) + 288)];
                float v1467_data = s0[41];
                float v1469_data = ir0[2];
                ir0[2] = (v1469_data + (v1466_data * v1467_data));
                int32_t v1477_a = (v8_lead + 4) + 288;
                float v1485_data = glb_m1[((v8_lead + 4) + 288)];
                float v1486_data = s0[57];
                float v1488_data = ir0[3];
                ir0[3] = (v1488_data + (v1485_data * v1486_data));
                int32_t v1496_a = (v8_lead + 4) + 288;
                float v1504_data = glb_m1[((v8_lead + 4) + 288)];
                float v1505_data = s0[73];
                float v1507_data = ir0[4];
                ir0[4] = (v1507_data + (v1504_data * v1505_data));
                int32_t v1515_a = (v8_lead + 4) + 288;
                float v1523_data = glb_m1[((v8_lead + 4) + 288)];
                float v1524_data = s0[89];
                float v1526_data = ir0[5];
                ir0[5] = (v1526_data + (v1523_data * v1524_data));
                int32_t v1534_a = (v8_lead + 4) + 288;
                float v1542_data = glb_m1[((v8_lead + 4) + 288)];
                float v1543_data = s0[105];
                float v1545_data = ir0[6];
                ir0[6] = (v1545_data + (v1542_data * v1543_data));
                int32_t v1553_a = (v8_lead + 4) + 288;
                float v1561_data = glb_m1[((v8_lead + 4) + 288)];
                float v1562_data = s0[121];
                float v1564_data = ir0[7];
                ir0[7] = (v1564_data + (v1561_data * v1562_data));
              }
              if (v8_lead < 12) {
                int32_t v1576_a = (v8_lead + 4) + 320;
                float v1584_data = glb_m1[((v8_lead + 4) + 320)];
                float v1585_data = s0[10];
                float v1587_data = ir0[0];
                ir0[0] = (v1587_data + (v1584_data * v1585_data));
                int32_t v1595_a = (v8_lead + 4) + 320;
                float v1603_data = glb_m1[((v8_lead + 4) + 320)];
                float v1604_data = s0[26];
                float v1606_data = ir0[1];
                ir0[1] = (v1606_data + (v1603_data * v1604_data));
                int32_t v1614_a = (v8_lead + 4) + 320;
                float v1622_data = glb_m1[((v8_lead + 4) + 320)];
                float v1623_data = s0[42];
                float v1625_data = ir0[2];
                ir0[2] = (v1625_data + (v1622_data * v1623_data));
                int32_t v1633_a = (v8_lead + 4) + 320;
                float v1641_data = glb_m1[((v8_lead + 4) + 320)];
                float v1642_data = s0[58];
                float v1644_data = ir0[3];
                ir0[3] = (v1644_data + (v1641_data * v1642_data));
                int32_t v1652_a = (v8_lead + 4) + 320;
                float v1660_data = glb_m1[((v8_lead + 4) + 320)];
                float v1661_data = s0[74];
                float v1663_data = ir0[4];
                ir0[4] = (v1663_data + (v1660_data * v1661_data));
                int32_t v1671_a = (v8_lead + 4) + 320;
                float v1679_data = glb_m1[((v8_lead + 4) + 320)];
                float v1680_data = s0[90];
                float v1682_data = ir0[5];
                ir0[5] = (v1682_data + (v1679_data * v1680_data));
                int32_t v1690_a = (v8_lead + 4) + 320;
                float v1698_data = glb_m1[((v8_lead + 4) + 320)];
                float v1699_data = s0[106];
                float v1701_data = ir0[6];
                ir0[6] = (v1701_data + (v1698_data * v1699_data));
                int32_t v1709_a = (v8_lead + 4) + 320;
                float v1717_data = glb_m1[((v8_lead + 4) + 320)];
                float v1718_data = s0[122];
                float v1720_data = ir0[7];
                ir0[7] = (v1720_data + (v1717_data * v1718_data));
              }
              if (v8_lead < 12) {
                int32_t v1732_a = (v8_lead + 4) + 352;
                float v1740_data = glb_m1[((v8_lead + 4) + 352)];
                float v1741_data = s0[11];
                float v1743_data = ir0[0];
                ir0[0] = (v1743_data + (v1740_data * v1741_data));
                int32_t v1751_a = (v8_lead + 4) + 352;
                float v1759_data = glb_m1[((v8_lead + 4) + 352)];
                float v1760_data = s0[27];
                float v1762_data = ir0[1];
                ir0[1] = (v1762_data + (v1759_data * v1760_data));
                int32_t v1770_a = (v8_lead + 4) + 352;
                float v1778_data = glb_m1[((v8_lead + 4) + 352)];
                float v1779_data = s0[43];
                float v1781_data = ir0[2];
                ir0[2] = (v1781_data + (v1778_data * v1779_data));
                int32_t v1789_a = (v8_lead + 4) + 352;
                float v1797_data = glb_m1[((v8_lead + 4) + 352)];
                float v1798_data = s0[59];
                float v1800_data = ir0[3];
                ir0[3] = (v1800_data + (v1797_data * v1798_data));
                int32_t v1808_a = (v8_lead + 4) + 352;
                float v1816_data = glb_m1[((v8_lead + 4) + 352)];
                float v1817_data = s0[75];
                float v1819_data = ir0[4];
                ir0[4] = (v1819_data + (v1816_data * v1817_data));
                int32_t v1827_a = (v8_lead + 4) + 352;
                float v1835_data = glb_m1[((v8_lead + 4) + 352)];
                float v1836_data = s0[91];
                float v1838_data = ir0[5];
                ir0[5] = (v1838_data + (v1835_data * v1836_data));
                int32_t v1846_a = (v8_lead + 4) + 352;
                float v1854_data = glb_m1[((v8_lead + 4) + 352)];
                float v1855_data = s0[107];
                float v1857_data = ir0[6];
                ir0[6] = (v1857_data + (v1854_data * v1855_data));
                int32_t v1865_a = (v8_lead + 4) + 352;
                float v1873_data = glb_m1[((v8_lead + 4) + 352)];
                float v1874_data = s0[123];
                float v1876_data = ir0[7];
                ir0[7] = (v1876_data + (v1873_data * v1874_data));
              }
              if (v8_lead < 12) {
                int32_t v1888_a = (v8_lead + 4) + 384;
                float v1896_data = glb_m1[((v8_lead + 4) + 384)];
                float v1897_data = s0[12];
                float v1899_data = ir0[0];
                ir0[0] = (v1899_data + (v1896_data * v1897_data));
                int32_t v1907_a = (v8_lead + 4) + 384;
                float v1915_data = glb_m1[((v8_lead + 4) + 384)];
                float v1916_data = s0[28];
                float v1918_data = ir0[1];
                ir0[1] = (v1918_data + (v1915_data * v1916_data));
                int32_t v1926_a = (v8_lead + 4) + 384;
                float v1934_data = glb_m1[((v8_lead + 4) + 384)];
                float v1935_data = s0[44];
                float v1937_data = ir0[2];
                ir0[2] = (v1937_data + (v1934_data * v1935_data));
                int32_t v1945_a = (v8_lead + 4) + 384;
                float v1953_data = glb_m1[((v8_lead + 4) + 384)];
                float v1954_data = s0[60];
                float v1956_data = ir0[3];
                ir0[3] = (v1956_data + (v1953_data * v1954_data));
                int32_t v1964_a = (v8_lead + 4) + 384;
                float v1972_data = glb_m1[((v8_lead + 4) + 384)];
                float v1973_data = s0[76];
                float v1975_data = ir0[4];
                ir0[4] = (v1975_data + (v1972_data * v1973_data));
                int32_t v1983_a = (v8_lead + 4) + 384;
                float v1991_data = glb_m1[((v8_lead + 4) + 384)];
                float v1992_data = s0[92];
                float v1994_data = ir0[5];
                ir0[5] = (v1994_data + (v1991_data * v1992_data));
                int32_t v2002_a = (v8_lead + 4) + 384;
                float v2010_data = glb_m1[((v8_lead + 4) + 384)];
                float v2011_data = s0[108];
                float v2013_data = ir0[6];
                ir0[6] = (v2013_data + (v2010_data * v2011_data));
                int32_t v2021_a = (v8_lead + 4) + 384;
                float v2029_data = glb_m1[((v8_lead + 4) + 384)];
                float v2030_data = s0[124];
                float v2032_data = ir0[7];
                ir0[7] = (v2032_data + (v2029_data * v2030_data));
              }
              if (v8_lead < 12) {
                int32_t v2044_a = (v8_lead + 4) + 416;
                float v2052_data = glb_m1[((v8_lead + 4) + 416)];
                float v2053_data = s0[13];
                float v2055_data = ir0[0];
                ir0[0] = (v2055_data + (v2052_data * v2053_data));
                int32_t v2063_a = (v8_lead + 4) + 416;
                float v2071_data = glb_m1[((v8_lead + 4) + 416)];
                float v2072_data = s0[29];
                float v2074_data = ir0[1];
                ir0[1] = (v2074_data + (v2071_data * v2072_data));
                int32_t v2082_a = (v8_lead + 4) + 416;
                float v2090_data = glb_m1[((v8_lead + 4) + 416)];
                float v2091_data = s0[45];
                float v2093_data = ir0[2];
                ir0[2] = (v2093_data + (v2090_data * v2091_data));
                int32_t v2101_a = (v8_lead + 4) + 416;
                float v2109_data = glb_m1[((v8_lead + 4) + 416)];
                float v2110_data = s0[61];
                float v2112_data = ir0[3];
                ir0[3] = (v2112_data + (v2109_data * v2110_data));
                int32_t v2120_a = (v8_lead + 4) + 416;
                float v2128_data = glb_m1[((v8_lead + 4) + 416)];
                float v2129_data = s0[77];
                float v2131_data = ir0[4];
                ir0[4] = (v2131_data + (v2128_data * v2129_data));
                int32_t v2139_a = (v8_lead + 4) + 416;
                float v2147_data = glb_m1[((v8_lead + 4) + 416)];
                float v2148_data = s0[93];
                float v2150_data = ir0[5];
                ir0[5] = (v2150_data + (v2147_data * v2148_data));
                int32_t v2158_a = (v8_lead + 4) + 416;
                float v2166_data = glb_m1[((v8_lead + 4) + 416)];
                float v2167_data = s0[109];
                float v2169_data = ir0[6];
                ir0[6] = (v2169_data + (v2166_data * v2167_data));
                int32_t v2177_a = (v8_lead + 4) + 416;
                float v2185_data = glb_m1[((v8_lead + 4) + 416)];
                float v2186_data = s0[125];
                float v2188_data = ir0[7];
                ir0[7] = (v2188_data + (v2185_data * v2186_data));
              }
              if (v8_lead < 12) {
                int32_t v2200_a = (v8_lead + 4) + 448;
                float v2208_data = glb_m1[((v8_lead + 4) + 448)];
                float v2209_data = s0[14];
                float v2211_data = ir0[0];
                ir0[0] = (v2211_data + (v2208_data * v2209_data));
                int32_t v2219_a = (v8_lead + 4) + 448;
                float v2227_data = glb_m1[((v8_lead + 4) + 448)];
                float v2228_data = s0[30];
                float v2230_data = ir0[1];
                ir0[1] = (v2230_data + (v2227_data * v2228_data));
                int32_t v2238_a = (v8_lead + 4) + 448;
                float v2246_data = glb_m1[((v8_lead + 4) + 448)];
                float v2247_data = s0[46];
                float v2249_data = ir0[2];
                ir0[2] = (v2249_data + (v2246_data * v2247_data));
                int32_t v2257_a = (v8_lead + 4) + 448;
                float v2265_data = glb_m1[((v8_lead + 4) + 448)];
                float v2266_data = s0[62];
                float v2268_data = ir0[3];
                ir0[3] = (v2268_data + (v2265_data * v2266_data));
                int32_t v2276_a = (v8_lead + 4) + 448;
                float v2284_data = glb_m1[((v8_lead + 4) + 448)];
                float v2285_data = s0[78];
                float v2287_data = ir0[4];
                ir0[4] = (v2287_data + (v2284_data * v2285_data));
                int32_t v2295_a = (v8_lead + 4) + 448;
                float v2303_data = glb_m1[((v8_lead + 4) + 448)];
                float v2304_data = s0[94];
                float v2306_data = ir0[5];
                ir0[5] = (v2306_data + (v2303_data * v2304_data));
                int32_t v2314_a = (v8_lead + 4) + 448;
                float v2322_data = glb_m1[((v8_lead + 4) + 448)];
                float v2323_data = s0[110];
                float v2325_data = ir0[6];
                ir0[6] = (v2325_data + (v2322_data * v2323_data));
                int32_t v2333_a = (v8_lead + 4) + 448;
                float v2341_data = glb_m1[((v8_lead + 4) + 448)];
                float v2342_data = s0[126];
                float v2344_data = ir0[7];
                ir0[7] = (v2344_data + (v2341_data * v2342_data));
              }
              if (v8_lead < 12) {
                int32_t v2356_a = (v8_lead + 4) + 480;
                float v2364_data = glb_m1[((v8_lead + 4) + 480)];
                float v2365_data = s0[15];
                float v2367_data = ir0[0];
                ir0[0] = (v2367_data + (v2364_data * v2365_data));
                int32_t v2375_a = (v8_lead + 4) + 480;
                float v2383_data = glb_m1[((v8_lead + 4) + 480)];
                float v2384_data = s0[31];
                float v2386_data = ir0[1];
                ir0[1] = (v2386_data + (v2383_data * v2384_data));
                int32_t v2394_a = (v8_lead + 4) + 480;
                float v2402_data = glb_m1[((v8_lead + 4) + 480)];
                float v2403_data = s0[47];
                float v2405_data = ir0[2];
                ir0[2] = (v2405_data + (v2402_data * v2403_data));
                int32_t v2413_a = (v8_lead + 4) + 480;
                float v2421_data = glb_m1[((v8_lead + 4) + 480)];
                float v2422_data = s0[63];
                float v2424_data = ir0[3];
                ir0[3] = (v2424_data + (v2421_data * v2422_data));
                int32_t v2432_a = (v8_lead + 4) + 480;
                float v2440_data = glb_m1[((v8_lead + 4) + 480)];
                float v2441_data = s0[79];
                float v2443_data = ir0[4];
                ir0[4] = (v2443_data + (v2440_data * v2441_data));
                int32_t v2451_a = (v8_lead + 4) + 480;
                float v2459_data = glb_m1[((v8_lead + 4) + 480)];
                float v2460_data = s0[95];
                float v2462_data = ir0[5];
                ir0[5] = (v2462_data + (v2459_data * v2460_data));
                int32_t v2470_a = (v8_lead + 4) + 480;
                float v2478_data = glb_m1[((v8_lead + 4) + 480)];
                float v2479_data = s0[111];
                float v2481_data = ir0[6];
                ir0[6] = (v2481_data + (v2478_data * v2479_data));
                int32_t v2489_a = (v8_lead + 4) + 480;
                float v2497_data = glb_m1[((v8_lead + 4) + 480)];
                float v2498_data = s0[127];
                float v2500_data = ir0[7];
                ir0[7] = (v2500_data + (v2497_data * v2498_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v2506_n1 = 0; v2506_n1 < 8; ++v2506_n1) {
                  int32_t v2507_a = 0 + v2506_n1;
                  float v2509_data = ir0[v2506_n1];
                  r0[v2506_n1] = v2509_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v2515_i1 = 0; v2515_i1 < 8; ++v2515_i1) {
                  int32_t v2516_a = 0 + v2515_i1;
                  float v2518_data = r0[v2515_i1];
                  glb_m0[(v8_lead + (v2515_i1 * 12))] = v2518_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


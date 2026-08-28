// === base name ===
kernel_417e1ddcc4

// === header ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_417e1ddcc4(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_417e1ddcc4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1024, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×16(16×16) {0..16}×{0..16} strided
        // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          double* localShrMem0 = &totalShrMem[64 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[48];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
              double* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<double, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<double, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 0];
              if (item.get_local_id(0) < 14) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 32] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 32];
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              double r0[16]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              double ir0[16]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              int32_t v14_a = v8_lead + 0;
              double v21_data = glb_m1[v8_lead];
              double v22_data = s0[0];
              double v24_data = ir0[0];
              ir0[0] = (v24_data + (v21_data * v22_data));
              int32_t v31_a = v8_lead + 0;
              double v38_data = glb_m1[v8_lead];
              double v39_data = s0[2];
              double v41_data = ir0[1];
              ir0[1] = (v41_data + (v38_data * v39_data));
              int32_t v48_a = v8_lead + 0;
              int32_t v61_a = v8_lead + 0;
              int32_t v74_a = v8_lead + 0;
              int32_t v87_a = v8_lead + 0;
              int32_t v100_a = v8_lead + 0;
              int32_t v113_a = v8_lead + 0;
              int32_t v126_a = v8_lead + 0;
              int32_t v139_a = v8_lead + 0;
              int32_t v152_a = v8_lead + 0;
              int32_t v165_a = v8_lead + 0;
              int32_t v178_a = v8_lead + 0;
              int32_t v191_a = v8_lead + 0;
              int32_t v204_a = v8_lead + 0;
              int32_t v217_a = v8_lead + 0;
              int32_t v233_a = v8_lead + 16;
              double v240_data = glb_m1[(v8_lead + 16)];
              double v241_data = s0[1];
              double v243_data = ir0[0];
              ir0[0] = (v243_data + (v240_data * v241_data));
              int32_t v250_a = v8_lead + 16;
              double v257_data = glb_m1[(v8_lead + 16)];
              double v258_data = s0[3];
              double v260_data = ir0[1];
              ir0[1] = (v260_data + (v257_data * v258_data));
              int32_t v267_a = v8_lead + 16;
              double v274_data = glb_m1[(v8_lead + 16)];
              double v275_data = s0[5];
              double v277_data = ir0[2];
              ir0[2] = (v277_data + (v274_data * v275_data));
              int32_t v284_a = v8_lead + 16;
              int32_t v297_a = v8_lead + 16;
              int32_t v310_a = v8_lead + 16;
              int32_t v323_a = v8_lead + 16;
              int32_t v336_a = v8_lead + 16;
              int32_t v349_a = v8_lead + 16;
              int32_t v362_a = v8_lead + 16;
              int32_t v375_a = v8_lead + 16;
              int32_t v388_a = v8_lead + 16;
              int32_t v401_a = v8_lead + 16;
              int32_t v414_a = v8_lead + 16;
              int32_t v427_a = v8_lead + 16;
              int32_t v440_a = v8_lead + 16;
              int32_t v456_a = v8_lead + 32;
              int32_t v469_a = v8_lead + 32;
              double v476_data = glb_m1[(v8_lead + 32)];
              double v477_data = s0[4];
              double v479_data = ir0[1];
              ir0[1] = (v479_data + (v476_data * v477_data));
              int32_t v486_a = v8_lead + 32;
              double v493_data = glb_m1[(v8_lead + 32)];
              double v494_data = s0[6];
              double v496_data = ir0[2];
              ir0[2] = (v496_data + (v493_data * v494_data));
              int32_t v503_a = v8_lead + 32;
              double v510_data = glb_m1[(v8_lead + 32)];
              double v511_data = s0[8];
              double v513_data = ir0[3];
              ir0[3] = (v513_data + (v510_data * v511_data));
              int32_t v520_a = v8_lead + 32;
              int32_t v533_a = v8_lead + 32;
              int32_t v546_a = v8_lead + 32;
              int32_t v559_a = v8_lead + 32;
              int32_t v572_a = v8_lead + 32;
              int32_t v585_a = v8_lead + 32;
              int32_t v598_a = v8_lead + 32;
              int32_t v611_a = v8_lead + 32;
              int32_t v624_a = v8_lead + 32;
              int32_t v637_a = v8_lead + 32;
              int32_t v650_a = v8_lead + 32;
              int32_t v663_a = v8_lead + 32;
              int32_t v679_a = v8_lead + 48;
              int32_t v692_a = v8_lead + 48;
              int32_t v705_a = v8_lead + 48;
              double v712_data = glb_m1[(v8_lead + 48)];
              double v713_data = s0[7];
              double v715_data = ir0[2];
              ir0[2] = (v715_data + (v712_data * v713_data));
              int32_t v722_a = v8_lead + 48;
              double v729_data = glb_m1[(v8_lead + 48)];
              double v730_data = s0[9];
              double v732_data = ir0[3];
              ir0[3] = (v732_data + (v729_data * v730_data));
              int32_t v739_a = v8_lead + 48;
              double v746_data = glb_m1[(v8_lead + 48)];
              double v747_data = s0[11];
              double v749_data = ir0[4];
              ir0[4] = (v749_data + (v746_data * v747_data));
              int32_t v756_a = v8_lead + 48;
              int32_t v769_a = v8_lead + 48;
              int32_t v782_a = v8_lead + 48;
              int32_t v795_a = v8_lead + 48;
              int32_t v808_a = v8_lead + 48;
              int32_t v821_a = v8_lead + 48;
              int32_t v834_a = v8_lead + 48;
              int32_t v847_a = v8_lead + 48;
              int32_t v860_a = v8_lead + 48;
              int32_t v873_a = v8_lead + 48;
              int32_t v886_a = v8_lead + 48;
              int32_t v902_a = v8_lead + 64;
              int32_t v915_a = v8_lead + 64;
              int32_t v928_a = v8_lead + 64;
              int32_t v941_a = v8_lead + 64;
              double v948_data = glb_m1[(v8_lead + 64)];
              double v949_data = s0[10];
              double v951_data = ir0[3];
              ir0[3] = (v951_data + (v948_data * v949_data));
              int32_t v958_a = v8_lead + 64;
              double v965_data = glb_m1[(v8_lead + 64)];
              double v966_data = s0[12];
              double v968_data = ir0[4];
              ir0[4] = (v968_data + (v965_data * v966_data));
              int32_t v975_a = v8_lead + 64;
              double v982_data = glb_m1[(v8_lead + 64)];
              double v983_data = s0[14];
              double v985_data = ir0[5];
              ir0[5] = (v985_data + (v982_data * v983_data));
              int32_t v992_a = v8_lead + 64;
              int32_t v1005_a = v8_lead + 64;
              int32_t v1018_a = v8_lead + 64;
              int32_t v1031_a = v8_lead + 64;
              int32_t v1044_a = v8_lead + 64;
              int32_t v1057_a = v8_lead + 64;
              int32_t v1070_a = v8_lead + 64;
              int32_t v1083_a = v8_lead + 64;
              int32_t v1096_a = v8_lead + 64;
              int32_t v1109_a = v8_lead + 64;
              int32_t v1125_a = v8_lead + 80;
              int32_t v1138_a = v8_lead + 80;
              int32_t v1151_a = v8_lead + 80;
              int32_t v1164_a = v8_lead + 80;
              int32_t v1177_a = v8_lead + 80;
              double v1184_data = glb_m1[(v8_lead + 80)];
              double v1185_data = s0[13];
              double v1187_data = ir0[4];
              ir0[4] = (v1187_data + (v1184_data * v1185_data));
              int32_t v1194_a = v8_lead + 80;
              double v1201_data = glb_m1[(v8_lead + 80)];
              double v1202_data = s0[15];
              double v1204_data = ir0[5];
              ir0[5] = (v1204_data + (v1201_data * v1202_data));
              int32_t v1211_a = v8_lead + 80;
              double v1218_data = glb_m1[(v8_lead + 80)];
              double v1219_data = s0[17];
              double v1221_data = ir0[6];
              ir0[6] = (v1221_data + (v1218_data * v1219_data));
              int32_t v1228_a = v8_lead + 80;
              int32_t v1241_a = v8_lead + 80;
              int32_t v1254_a = v8_lead + 80;
              int32_t v1267_a = v8_lead + 80;
              int32_t v1280_a = v8_lead + 80;
              int32_t v1293_a = v8_lead + 80;
              int32_t v1306_a = v8_lead + 80;
              int32_t v1319_a = v8_lead + 80;
              int32_t v1332_a = v8_lead + 80;
              int32_t v1348_a = v8_lead + 96;
              int32_t v1361_a = v8_lead + 96;
              int32_t v1374_a = v8_lead + 96;
              int32_t v1387_a = v8_lead + 96;
              int32_t v1400_a = v8_lead + 96;
              int32_t v1413_a = v8_lead + 96;
              double v1420_data = glb_m1[(v8_lead + 96)];
              double v1421_data = s0[16];
              double v1423_data = ir0[5];
              ir0[5] = (v1423_data + (v1420_data * v1421_data));
              int32_t v1430_a = v8_lead + 96;
              double v1437_data = glb_m1[(v8_lead + 96)];
              double v1438_data = s0[18];
              double v1440_data = ir0[6];
              ir0[6] = (v1440_data + (v1437_data * v1438_data));
              int32_t v1447_a = v8_lead + 96;
              double v1454_data = glb_m1[(v8_lead + 96)];
              double v1455_data = s0[20];
              double v1457_data = ir0[7];
              ir0[7] = (v1457_data + (v1454_data * v1455_data));
              int32_t v1464_a = v8_lead + 96;
              int32_t v1477_a = v8_lead + 96;
              int32_t v1490_a = v8_lead + 96;
              int32_t v1503_a = v8_lead + 96;
              int32_t v1516_a = v8_lead + 96;
              int32_t v1529_a = v8_lead + 96;
              int32_t v1542_a = v8_lead + 96;
              int32_t v1555_a = v8_lead + 96;
              int32_t v1571_a = v8_lead + 112;
              int32_t v1584_a = v8_lead + 112;
              int32_t v1597_a = v8_lead + 112;
              int32_t v1610_a = v8_lead + 112;
              int32_t v1623_a = v8_lead + 112;
              int32_t v1636_a = v8_lead + 112;
              int32_t v1649_a = v8_lead + 112;
              double v1656_data = glb_m1[(v8_lead + 112)];
              double v1657_data = s0[19];
              double v1659_data = ir0[6];
              ir0[6] = (v1659_data + (v1656_data * v1657_data));
              int32_t v1666_a = v8_lead + 112;
              double v1673_data = glb_m1[(v8_lead + 112)];
              double v1674_data = s0[21];
              double v1676_data = ir0[7];
              ir0[7] = (v1676_data + (v1673_data * v1674_data));
              int32_t v1683_a = v8_lead + 112;
              double v1690_data = glb_m1[(v8_lead + 112)];
              double v1691_data = s0[23];
              double v1693_data = ir0[8];
              ir0[8] = (v1693_data + (v1690_data * v1691_data));
              int32_t v1700_a = v8_lead + 112;
              int32_t v1713_a = v8_lead + 112;
              int32_t v1726_a = v8_lead + 112;
              int32_t v1739_a = v8_lead + 112;
              int32_t v1752_a = v8_lead + 112;
              int32_t v1765_a = v8_lead + 112;
              int32_t v1778_a = v8_lead + 112;
              int32_t v1794_a = v8_lead + 128;
              int32_t v1807_a = v8_lead + 128;
              int32_t v1820_a = v8_lead + 128;
              int32_t v1833_a = v8_lead + 128;
              int32_t v1846_a = v8_lead + 128;
              int32_t v1859_a = v8_lead + 128;
              int32_t v1872_a = v8_lead + 128;
              int32_t v1885_a = v8_lead + 128;
              double v1892_data = glb_m1[(v8_lead + 128)];
              double v1893_data = s0[22];
              double v1895_data = ir0[7];
              ir0[7] = (v1895_data + (v1892_data * v1893_data));
              int32_t v1902_a = v8_lead + 128;
              double v1909_data = glb_m1[(v8_lead + 128)];
              double v1910_data = s0[24];
              double v1912_data = ir0[8];
              ir0[8] = (v1912_data + (v1909_data * v1910_data));
              int32_t v1919_a = v8_lead + 128;
              double v1926_data = glb_m1[(v8_lead + 128)];
              double v1927_data = s0[26];
              double v1929_data = ir0[9];
              ir0[9] = (v1929_data + (v1926_data * v1927_data));
              int32_t v1936_a = v8_lead + 128;
              int32_t v1949_a = v8_lead + 128;
              int32_t v1962_a = v8_lead + 128;
              int32_t v1975_a = v8_lead + 128;
              int32_t v1988_a = v8_lead + 128;
              int32_t v2001_a = v8_lead + 128;
              int32_t v2017_a = v8_lead + 144;
              int32_t v2030_a = v8_lead + 144;
              int32_t v2043_a = v8_lead + 144;
              int32_t v2056_a = v8_lead + 144;
              int32_t v2069_a = v8_lead + 144;
              int32_t v2082_a = v8_lead + 144;
              int32_t v2095_a = v8_lead + 144;
              int32_t v2108_a = v8_lead + 144;
              int32_t v2121_a = v8_lead + 144;
              double v2128_data = glb_m1[(v8_lead + 144)];
              double v2129_data = s0[25];
              double v2131_data = ir0[8];
              ir0[8] = (v2131_data + (v2128_data * v2129_data));
              int32_t v2138_a = v8_lead + 144;
              double v2145_data = glb_m1[(v8_lead + 144)];
              double v2146_data = s0[27];
              double v2148_data = ir0[9];
              ir0[9] = (v2148_data + (v2145_data * v2146_data));
              int32_t v2155_a = v8_lead + 144;
              double v2162_data = glb_m1[(v8_lead + 144)];
              double v2163_data = s0[29];
              double v2165_data = ir0[10];
              ir0[10] = (v2165_data + (v2162_data * v2163_data));
              int32_t v2172_a = v8_lead + 144;
              int32_t v2185_a = v8_lead + 144;
              int32_t v2198_a = v8_lead + 144;
              int32_t v2211_a = v8_lead + 144;
              int32_t v2224_a = v8_lead + 144;
              int32_t v2240_a = v8_lead + 160;
              int32_t v2253_a = v8_lead + 160;
              int32_t v2266_a = v8_lead + 160;
              int32_t v2279_a = v8_lead + 160;
              int32_t v2292_a = v8_lead + 160;
              int32_t v2305_a = v8_lead + 160;
              int32_t v2318_a = v8_lead + 160;
              int32_t v2331_a = v8_lead + 160;
              int32_t v2344_a = v8_lead + 160;
              int32_t v2357_a = v8_lead + 160;
              double v2364_data = glb_m1[(v8_lead + 160)];
              double v2365_data = s0[28];
              double v2367_data = ir0[9];
              ir0[9] = (v2367_data + (v2364_data * v2365_data));
              int32_t v2374_a = v8_lead + 160;
              double v2381_data = glb_m1[(v8_lead + 160)];
              double v2382_data = s0[30];
              double v2384_data = ir0[10];
              ir0[10] = (v2384_data + (v2381_data * v2382_data));
              int32_t v2391_a = v8_lead + 160;
              double v2398_data = glb_m1[(v8_lead + 160)];
              double v2399_data = s0[32];
              double v2401_data = ir0[11];
              ir0[11] = (v2401_data + (v2398_data * v2399_data));
              int32_t v2408_a = v8_lead + 160;
              int32_t v2421_a = v8_lead + 160;
              int32_t v2434_a = v8_lead + 160;
              int32_t v2447_a = v8_lead + 160;
              int32_t v2463_a = v8_lead + 176;
              int32_t v2476_a = v8_lead + 176;
              int32_t v2489_a = v8_lead + 176;
              int32_t v2502_a = v8_lead + 176;
              int32_t v2515_a = v8_lead + 176;
              int32_t v2528_a = v8_lead + 176;
              int32_t v2541_a = v8_lead + 176;
              int32_t v2554_a = v8_lead + 176;
              int32_t v2567_a = v8_lead + 176;
              int32_t v2580_a = v8_lead + 176;
              int32_t v2593_a = v8_lead + 176;
              double v2600_data = glb_m1[(v8_lead + 176)];
              double v2601_data = s0[31];
              double v2603_data = ir0[10];
              ir0[10] = (v2603_data + (v2600_data * v2601_data));
              int32_t v2610_a = v8_lead + 176;
              double v2617_data = glb_m1[(v8_lead + 176)];
              double v2618_data = s0[33];
              double v2620_data = ir0[11];
              ir0[11] = (v2620_data + (v2617_data * v2618_data));
              int32_t v2627_a = v8_lead + 176;
              double v2634_data = glb_m1[(v8_lead + 176)];
              double v2635_data = s0[35];
              double v2637_data = ir0[12];
              ir0[12] = (v2637_data + (v2634_data * v2635_data));
              int32_t v2644_a = v8_lead + 176;
              int32_t v2657_a = v8_lead + 176;
              int32_t v2670_a = v8_lead + 176;
              int32_t v2686_a = v8_lead + 192;
              int32_t v2699_a = v8_lead + 192;
              int32_t v2712_a = v8_lead + 192;
              int32_t v2725_a = v8_lead + 192;
              int32_t v2738_a = v8_lead + 192;
              int32_t v2751_a = v8_lead + 192;
              int32_t v2764_a = v8_lead + 192;
              int32_t v2777_a = v8_lead + 192;
              int32_t v2790_a = v8_lead + 192;
              int32_t v2803_a = v8_lead + 192;
              int32_t v2816_a = v8_lead + 192;
              int32_t v2829_a = v8_lead + 192;
              double v2836_data = glb_m1[(v8_lead + 192)];
              double v2837_data = s0[34];
              double v2839_data = ir0[11];
              ir0[11] = (v2839_data + (v2836_data * v2837_data));
              int32_t v2846_a = v8_lead + 192;
              double v2853_data = glb_m1[(v8_lead + 192)];
              double v2854_data = s0[36];
              double v2856_data = ir0[12];
              ir0[12] = (v2856_data + (v2853_data * v2854_data));
              int32_t v2863_a = v8_lead + 192;
              double v2870_data = glb_m1[(v8_lead + 192)];
              double v2871_data = s0[38];
              double v2873_data = ir0[13];
              ir0[13] = (v2873_data + (v2870_data * v2871_data));
              int32_t v2880_a = v8_lead + 192;
              int32_t v2893_a = v8_lead + 192;
              int32_t v2909_a = v8_lead + 208;
              int32_t v2922_a = v8_lead + 208;
              int32_t v2935_a = v8_lead + 208;
              int32_t v2948_a = v8_lead + 208;
              int32_t v2961_a = v8_lead + 208;
              int32_t v2974_a = v8_lead + 208;
              int32_t v2987_a = v8_lead + 208;
              int32_t v3000_a = v8_lead + 208;
              int32_t v3013_a = v8_lead + 208;
              int32_t v3026_a = v8_lead + 208;
              int32_t v3039_a = v8_lead + 208;
              int32_t v3052_a = v8_lead + 208;
              int32_t v3065_a = v8_lead + 208;
              double v3072_data = glb_m1[(v8_lead + 208)];
              double v3073_data = s0[37];
              double v3075_data = ir0[12];
              ir0[12] = (v3075_data + (v3072_data * v3073_data));
              int32_t v3082_a = v8_lead + 208;
              double v3089_data = glb_m1[(v8_lead + 208)];
              double v3090_data = s0[39];
              double v3092_data = ir0[13];
              ir0[13] = (v3092_data + (v3089_data * v3090_data));
              int32_t v3099_a = v8_lead + 208;
              double v3106_data = glb_m1[(v8_lead + 208)];
              double v3107_data = s0[41];
              double v3109_data = ir0[14];
              ir0[14] = (v3109_data + (v3106_data * v3107_data));
              int32_t v3116_a = v8_lead + 208;
              int32_t v3132_a = v8_lead + 224;
              int32_t v3145_a = v8_lead + 224;
              int32_t v3158_a = v8_lead + 224;
              int32_t v3171_a = v8_lead + 224;
              int32_t v3184_a = v8_lead + 224;
              int32_t v3197_a = v8_lead + 224;
              int32_t v3210_a = v8_lead + 224;
              int32_t v3223_a = v8_lead + 224;
              int32_t v3236_a = v8_lead + 224;
              int32_t v3249_a = v8_lead + 224;
              int32_t v3262_a = v8_lead + 224;
              int32_t v3275_a = v8_lead + 224;
              int32_t v3288_a = v8_lead + 224;
              int32_t v3301_a = v8_lead + 224;
              double v3308_data = glb_m1[(v8_lead + 224)];
              double v3309_data = s0[40];
              double v3311_data = ir0[13];
              ir0[13] = (v3311_data + (v3308_data * v3309_data));
              int32_t v3318_a = v8_lead + 224;
              double v3325_data = glb_m1[(v8_lead + 224)];
              double v3326_data = s0[42];
              double v3328_data = ir0[14];
              ir0[14] = (v3328_data + (v3325_data * v3326_data));
              int32_t v3335_a = v8_lead + 224;
              double v3342_data = glb_m1[(v8_lead + 224)];
              double v3343_data = s0[44];
              double v3345_data = ir0[15];
              ir0[15] = (v3345_data + (v3342_data * v3343_data));
              int32_t v3355_a = v8_lead + 240;
              int32_t v3368_a = v8_lead + 240;
              int32_t v3381_a = v8_lead + 240;
              int32_t v3394_a = v8_lead + 240;
              int32_t v3407_a = v8_lead + 240;
              int32_t v3420_a = v8_lead + 240;
              int32_t v3433_a = v8_lead + 240;
              int32_t v3446_a = v8_lead + 240;
              int32_t v3459_a = v8_lead + 240;
              int32_t v3472_a = v8_lead + 240;
              int32_t v3485_a = v8_lead + 240;
              int32_t v3498_a = v8_lead + 240;
              int32_t v3511_a = v8_lead + 240;
              int32_t v3524_a = v8_lead + 240;
              int32_t v3537_a = v8_lead + 240;
              double v3544_data = glb_m1[(v8_lead + 240)];
              double v3545_data = s0[43];
              double v3547_data = ir0[14];
              ir0[14] = (v3547_data + (v3544_data * v3545_data));
              int32_t v3554_a = v8_lead + 240;
              double v3561_data = glb_m1[(v8_lead + 240)];
              double v3562_data = s0[45];
              double v3564_data = ir0[15];
              ir0[15] = (v3564_data + (v3561_data * v3562_data));
              #pragma unroll
              for (int32_t v3569_n0 = 0; v3569_n0 < 1; ++v3569_n0) {
                #pragma unroll
                for (int32_t v3570_n1 = 0; v3570_n1 < 16; ++v3570_n1) {
                  int32_t v3571_a = v3569_n0 + v3570_n1;
                  int32_t v3572_a = v3569_n0 + v3570_n1;
                  double v3573_data = ir0[v3572_a];
                  r0[v3572_a] = v3573_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v3578_i0 = 0; v3578_i0 < 1; ++v3578_i0) {
                int32_t v3587_lead = v8_lead + (v3578_i0 * 16);
                #pragma unroll
                for (int32_t v3579_i1 = 0; v3579_i1 < 16; ++v3579_i1) {
                  int32_t v3580_a = v3578_i0 + v3579_i1;
                  double v3582_data = r0[(v3578_i0 + v3579_i1)];
                  glb_m0[(v3587_lead + (v3579_i1 * 16))] = v3582_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


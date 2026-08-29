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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
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
              double v15_data = glb_m1[v8_lead];
              double v19_data = s0[(0 ^ ((0 >> 4) & 15))];
              double v21_data = ir0[0];
              ir0[0] = (v21_data + (v15_data * v19_data));
              double v29_data = glb_m1[v8_lead];
              double v33_data = s0[(2 ^ ((2 >> 4) & 15))];
              double v35_data = ir0[1];
              ir0[1] = (v35_data + (v29_data * v33_data));
              double v144_data = glb_m1[(v8_lead + 16)];
              double v148_data = s0[(1 ^ ((1 >> 4) & 15))];
              double v150_data = ir0[0];
              ir0[0] = (v150_data + (v144_data * v148_data));
              double v158_data = glb_m1[(v8_lead + 16)];
              double v162_data = s0[(3 ^ ((3 >> 4) & 15))];
              double v164_data = ir0[1];
              ir0[1] = (v164_data + (v158_data * v162_data));
              double v172_data = glb_m1[(v8_lead + 16)];
              double v176_data = s0[(5 ^ ((5 >> 4) & 15))];
              double v178_data = ir0[2];
              ir0[2] = (v178_data + (v172_data * v176_data));
              double v287_data = glb_m1[(v8_lead + 32)];
              double v291_data = s0[(4 ^ ((4 >> 4) & 15))];
              double v293_data = ir0[1];
              ir0[1] = (v293_data + (v287_data * v291_data));
              double v301_data = glb_m1[(v8_lead + 32)];
              double v305_data = s0[(6 ^ ((6 >> 4) & 15))];
              double v307_data = ir0[2];
              ir0[2] = (v307_data + (v301_data * v305_data));
              double v315_data = glb_m1[(v8_lead + 32)];
              double v319_data = s0[(8 ^ ((8 >> 4) & 15))];
              double v321_data = ir0[3];
              ir0[3] = (v321_data + (v315_data * v319_data));
              double v430_data = glb_m1[(v8_lead + 48)];
              double v434_data = s0[(7 ^ ((7 >> 4) & 15))];
              double v436_data = ir0[2];
              ir0[2] = (v436_data + (v430_data * v434_data));
              double v444_data = glb_m1[(v8_lead + 48)];
              double v448_data = s0[(9 ^ ((9 >> 4) & 15))];
              double v450_data = ir0[3];
              ir0[3] = (v450_data + (v444_data * v448_data));
              double v458_data = glb_m1[(v8_lead + 48)];
              double v462_data = s0[(11 ^ ((11 >> 4) & 15))];
              double v464_data = ir0[4];
              ir0[4] = (v464_data + (v458_data * v462_data));
              double v573_data = glb_m1[(v8_lead + 64)];
              double v577_data = s0[(10 ^ ((10 >> 4) & 15))];
              double v579_data = ir0[3];
              ir0[3] = (v579_data + (v573_data * v577_data));
              double v587_data = glb_m1[(v8_lead + 64)];
              double v591_data = s0[(12 ^ ((12 >> 4) & 15))];
              double v593_data = ir0[4];
              ir0[4] = (v593_data + (v587_data * v591_data));
              double v601_data = glb_m1[(v8_lead + 64)];
              double v605_data = s0[(14 ^ ((14 >> 4) & 15))];
              double v607_data = ir0[5];
              ir0[5] = (v607_data + (v601_data * v605_data));
              double v716_data = glb_m1[(v8_lead + 80)];
              double v720_data = s0[(13 ^ ((13 >> 4) & 15))];
              double v722_data = ir0[4];
              ir0[4] = (v722_data + (v716_data * v720_data));
              double v730_data = glb_m1[(v8_lead + 80)];
              double v734_data = s0[(15 ^ ((15 >> 4) & 15))];
              double v736_data = ir0[5];
              ir0[5] = (v736_data + (v730_data * v734_data));
              double v744_data = glb_m1[(v8_lead + 80)];
              double v748_data = s0[(17 ^ ((17 >> 4) & 15))];
              double v750_data = ir0[6];
              ir0[6] = (v750_data + (v744_data * v748_data));
              double v859_data = glb_m1[(v8_lead + 96)];
              double v863_data = s0[(16 ^ ((16 >> 4) & 15))];
              double v865_data = ir0[5];
              ir0[5] = (v865_data + (v859_data * v863_data));
              double v873_data = glb_m1[(v8_lead + 96)];
              double v877_data = s0[(18 ^ ((18 >> 4) & 15))];
              double v879_data = ir0[6];
              ir0[6] = (v879_data + (v873_data * v877_data));
              double v887_data = glb_m1[(v8_lead + 96)];
              double v891_data = s0[(20 ^ ((20 >> 4) & 15))];
              double v893_data = ir0[7];
              ir0[7] = (v893_data + (v887_data * v891_data));
              double v1002_data = glb_m1[(v8_lead + 112)];
              double v1006_data = s0[(19 ^ ((19 >> 4) & 15))];
              double v1008_data = ir0[6];
              ir0[6] = (v1008_data + (v1002_data * v1006_data));
              double v1016_data = glb_m1[(v8_lead + 112)];
              double v1020_data = s0[(21 ^ ((21 >> 4) & 15))];
              double v1022_data = ir0[7];
              ir0[7] = (v1022_data + (v1016_data * v1020_data));
              double v1030_data = glb_m1[(v8_lead + 112)];
              double v1034_data = s0[(23 ^ ((23 >> 4) & 15))];
              double v1036_data = ir0[8];
              ir0[8] = (v1036_data + (v1030_data * v1034_data));
              double v1145_data = glb_m1[(v8_lead + 128)];
              double v1149_data = s0[(22 ^ ((22 >> 4) & 15))];
              double v1151_data = ir0[7];
              ir0[7] = (v1151_data + (v1145_data * v1149_data));
              double v1159_data = glb_m1[(v8_lead + 128)];
              double v1163_data = s0[(24 ^ ((24 >> 4) & 15))];
              double v1165_data = ir0[8];
              ir0[8] = (v1165_data + (v1159_data * v1163_data));
              double v1173_data = glb_m1[(v8_lead + 128)];
              double v1177_data = s0[(26 ^ ((26 >> 4) & 15))];
              double v1179_data = ir0[9];
              ir0[9] = (v1179_data + (v1173_data * v1177_data));
              double v1288_data = glb_m1[(v8_lead + 144)];
              double v1292_data = s0[(25 ^ ((25 >> 4) & 15))];
              double v1294_data = ir0[8];
              ir0[8] = (v1294_data + (v1288_data * v1292_data));
              double v1302_data = glb_m1[(v8_lead + 144)];
              double v1306_data = s0[(27 ^ ((27 >> 4) & 15))];
              double v1308_data = ir0[9];
              ir0[9] = (v1308_data + (v1302_data * v1306_data));
              double v1316_data = glb_m1[(v8_lead + 144)];
              double v1320_data = s0[(29 ^ ((29 >> 4) & 15))];
              double v1322_data = ir0[10];
              ir0[10] = (v1322_data + (v1316_data * v1320_data));
              double v1431_data = glb_m1[(v8_lead + 160)];
              double v1435_data = s0[(28 ^ ((28 >> 4) & 15))];
              double v1437_data = ir0[9];
              ir0[9] = (v1437_data + (v1431_data * v1435_data));
              double v1445_data = glb_m1[(v8_lead + 160)];
              double v1449_data = s0[(30 ^ ((30 >> 4) & 15))];
              double v1451_data = ir0[10];
              ir0[10] = (v1451_data + (v1445_data * v1449_data));
              double v1459_data = glb_m1[(v8_lead + 160)];
              double v1463_data = s0[(32 ^ ((32 >> 4) & 15))];
              double v1465_data = ir0[11];
              ir0[11] = (v1465_data + (v1459_data * v1463_data));
              double v1574_data = glb_m1[(v8_lead + 176)];
              double v1578_data = s0[(31 ^ ((31 >> 4) & 15))];
              double v1580_data = ir0[10];
              ir0[10] = (v1580_data + (v1574_data * v1578_data));
              double v1588_data = glb_m1[(v8_lead + 176)];
              double v1592_data = s0[(33 ^ ((33 >> 4) & 15))];
              double v1594_data = ir0[11];
              ir0[11] = (v1594_data + (v1588_data * v1592_data));
              double v1602_data = glb_m1[(v8_lead + 176)];
              double v1606_data = s0[(35 ^ ((35 >> 4) & 15))];
              double v1608_data = ir0[12];
              ir0[12] = (v1608_data + (v1602_data * v1606_data));
              double v1717_data = glb_m1[(v8_lead + 192)];
              double v1721_data = s0[(34 ^ ((34 >> 4) & 15))];
              double v1723_data = ir0[11];
              ir0[11] = (v1723_data + (v1717_data * v1721_data));
              double v1731_data = glb_m1[(v8_lead + 192)];
              double v1735_data = s0[(36 ^ ((36 >> 4) & 15))];
              double v1737_data = ir0[12];
              ir0[12] = (v1737_data + (v1731_data * v1735_data));
              double v1745_data = glb_m1[(v8_lead + 192)];
              double v1749_data = s0[(38 ^ ((38 >> 4) & 15))];
              double v1751_data = ir0[13];
              ir0[13] = (v1751_data + (v1745_data * v1749_data));
              double v1860_data = glb_m1[(v8_lead + 208)];
              double v1864_data = s0[(37 ^ ((37 >> 4) & 15))];
              double v1866_data = ir0[12];
              ir0[12] = (v1866_data + (v1860_data * v1864_data));
              double v1874_data = glb_m1[(v8_lead + 208)];
              double v1878_data = s0[(39 ^ ((39 >> 4) & 15))];
              double v1880_data = ir0[13];
              ir0[13] = (v1880_data + (v1874_data * v1878_data));
              double v1888_data = glb_m1[(v8_lead + 208)];
              double v1892_data = s0[(41 ^ ((41 >> 4) & 15))];
              double v1894_data = ir0[14];
              ir0[14] = (v1894_data + (v1888_data * v1892_data));
              double v2003_data = glb_m1[(v8_lead + 224)];
              double v2007_data = s0[(40 ^ ((40 >> 4) & 15))];
              double v2009_data = ir0[13];
              ir0[13] = (v2009_data + (v2003_data * v2007_data));
              double v2017_data = glb_m1[(v8_lead + 224)];
              double v2021_data = s0[(42 ^ ((42 >> 4) & 15))];
              double v2023_data = ir0[14];
              ir0[14] = (v2023_data + (v2017_data * v2021_data));
              double v2031_data = glb_m1[(v8_lead + 224)];
              double v2035_data = s0[(44 ^ ((44 >> 4) & 15))];
              double v2037_data = ir0[15];
              ir0[15] = (v2037_data + (v2031_data * v2035_data));
              double v2146_data = glb_m1[(v8_lead + 240)];
              double v2150_data = s0[(43 ^ ((43 >> 4) & 15))];
              double v2152_data = ir0[14];
              ir0[14] = (v2152_data + (v2146_data * v2150_data));
              double v2160_data = glb_m1[(v8_lead + 240)];
              double v2164_data = s0[(45 ^ ((45 >> 4) & 15))];
              double v2166_data = ir0[15];
              ir0[15] = (v2166_data + (v2160_data * v2164_data));
              #pragma unroll
              for (int32_t v2171_n0 = 0; v2171_n0 < 1; ++v2171_n0) {
                #pragma unroll
                for (int32_t v2172_n1 = 0; v2172_n1 < 16; ++v2172_n1) {
                  int32_t v2173_a = v2171_n0 + v2172_n1;
                  double v2174_data = ir0[v2173_a];
                  r0[v2173_a] = v2174_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v2179_i0 = 0; v2179_i0 < 1; ++v2179_i0) {
                int32_t v2187_lead = v8_lead + (v2179_i0 * 16);
                #pragma unroll
                for (int32_t v2180_i1 = 0; v2180_i1 < 16; ++v2180_i1) {
                  double v2182_data = r0[(v2179_i0 + v2180_i1)];
                  glb_m0[(v2187_lead + (v2180_i1 * 16))] = v2182_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


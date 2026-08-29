// === base name ===
kernel_30948bd44e

// === header ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_30948bd44e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_30948bd44e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1024, cgh); {
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
          float* localShrMem0 = &totalShrMem[64 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[48];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<float, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 0];
              if (item.get_local_id(0) < 14) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 32] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 32];
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[16]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir0[16]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              float v15_data = glb_m1[v8_lead];
              float v19_data = s0[(0 ^ ((0 >> 5) & 31))];
              float v21_data = ir0[0];
              ir0[0] = (v21_data + (v15_data * v19_data));
              float v29_data = glb_m1[v8_lead];
              float v33_data = s0[(2 ^ ((2 >> 5) & 31))];
              float v35_data = ir0[1];
              ir0[1] = (v35_data + (v29_data * v33_data));
              float v144_data = glb_m1[(v8_lead + 16)];
              float v148_data = s0[(1 ^ ((1 >> 5) & 31))];
              float v150_data = ir0[0];
              ir0[0] = (v150_data + (v144_data * v148_data));
              float v158_data = glb_m1[(v8_lead + 16)];
              float v162_data = s0[(3 ^ ((3 >> 5) & 31))];
              float v164_data = ir0[1];
              ir0[1] = (v164_data + (v158_data * v162_data));
              float v172_data = glb_m1[(v8_lead + 16)];
              float v176_data = s0[(5 ^ ((5 >> 5) & 31))];
              float v178_data = ir0[2];
              ir0[2] = (v178_data + (v172_data * v176_data));
              float v287_data = glb_m1[(v8_lead + 32)];
              float v291_data = s0[(4 ^ ((4 >> 5) & 31))];
              float v293_data = ir0[1];
              ir0[1] = (v293_data + (v287_data * v291_data));
              float v301_data = glb_m1[(v8_lead + 32)];
              float v305_data = s0[(6 ^ ((6 >> 5) & 31))];
              float v307_data = ir0[2];
              ir0[2] = (v307_data + (v301_data * v305_data));
              float v315_data = glb_m1[(v8_lead + 32)];
              float v319_data = s0[(8 ^ ((8 >> 5) & 31))];
              float v321_data = ir0[3];
              ir0[3] = (v321_data + (v315_data * v319_data));
              float v430_data = glb_m1[(v8_lead + 48)];
              float v434_data = s0[(7 ^ ((7 >> 5) & 31))];
              float v436_data = ir0[2];
              ir0[2] = (v436_data + (v430_data * v434_data));
              float v444_data = glb_m1[(v8_lead + 48)];
              float v448_data = s0[(9 ^ ((9 >> 5) & 31))];
              float v450_data = ir0[3];
              ir0[3] = (v450_data + (v444_data * v448_data));
              float v458_data = glb_m1[(v8_lead + 48)];
              float v462_data = s0[(11 ^ ((11 >> 5) & 31))];
              float v464_data = ir0[4];
              ir0[4] = (v464_data + (v458_data * v462_data));
              float v573_data = glb_m1[(v8_lead + 64)];
              float v577_data = s0[(10 ^ ((10 >> 5) & 31))];
              float v579_data = ir0[3];
              ir0[3] = (v579_data + (v573_data * v577_data));
              float v587_data = glb_m1[(v8_lead + 64)];
              float v591_data = s0[(12 ^ ((12 >> 5) & 31))];
              float v593_data = ir0[4];
              ir0[4] = (v593_data + (v587_data * v591_data));
              float v601_data = glb_m1[(v8_lead + 64)];
              float v605_data = s0[(14 ^ ((14 >> 5) & 31))];
              float v607_data = ir0[5];
              ir0[5] = (v607_data + (v601_data * v605_data));
              float v716_data = glb_m1[(v8_lead + 80)];
              float v720_data = s0[(13 ^ ((13 >> 5) & 31))];
              float v722_data = ir0[4];
              ir0[4] = (v722_data + (v716_data * v720_data));
              float v730_data = glb_m1[(v8_lead + 80)];
              float v734_data = s0[(15 ^ ((15 >> 5) & 31))];
              float v736_data = ir0[5];
              ir0[5] = (v736_data + (v730_data * v734_data));
              float v744_data = glb_m1[(v8_lead + 80)];
              float v748_data = s0[(17 ^ ((17 >> 5) & 31))];
              float v750_data = ir0[6];
              ir0[6] = (v750_data + (v744_data * v748_data));
              float v859_data = glb_m1[(v8_lead + 96)];
              float v863_data = s0[(16 ^ ((16 >> 5) & 31))];
              float v865_data = ir0[5];
              ir0[5] = (v865_data + (v859_data * v863_data));
              float v873_data = glb_m1[(v8_lead + 96)];
              float v877_data = s0[(18 ^ ((18 >> 5) & 31))];
              float v879_data = ir0[6];
              ir0[6] = (v879_data + (v873_data * v877_data));
              float v887_data = glb_m1[(v8_lead + 96)];
              float v891_data = s0[(20 ^ ((20 >> 5) & 31))];
              float v893_data = ir0[7];
              ir0[7] = (v893_data + (v887_data * v891_data));
              float v1002_data = glb_m1[(v8_lead + 112)];
              float v1006_data = s0[(19 ^ ((19 >> 5) & 31))];
              float v1008_data = ir0[6];
              ir0[6] = (v1008_data + (v1002_data * v1006_data));
              float v1016_data = glb_m1[(v8_lead + 112)];
              float v1020_data = s0[(21 ^ ((21 >> 5) & 31))];
              float v1022_data = ir0[7];
              ir0[7] = (v1022_data + (v1016_data * v1020_data));
              float v1030_data = glb_m1[(v8_lead + 112)];
              float v1034_data = s0[(23 ^ ((23 >> 5) & 31))];
              float v1036_data = ir0[8];
              ir0[8] = (v1036_data + (v1030_data * v1034_data));
              float v1145_data = glb_m1[(v8_lead + 128)];
              float v1149_data = s0[(22 ^ ((22 >> 5) & 31))];
              float v1151_data = ir0[7];
              ir0[7] = (v1151_data + (v1145_data * v1149_data));
              float v1159_data = glb_m1[(v8_lead + 128)];
              float v1163_data = s0[(24 ^ ((24 >> 5) & 31))];
              float v1165_data = ir0[8];
              ir0[8] = (v1165_data + (v1159_data * v1163_data));
              float v1173_data = glb_m1[(v8_lead + 128)];
              float v1177_data = s0[(26 ^ ((26 >> 5) & 31))];
              float v1179_data = ir0[9];
              ir0[9] = (v1179_data + (v1173_data * v1177_data));
              float v1288_data = glb_m1[(v8_lead + 144)];
              float v1292_data = s0[(25 ^ ((25 >> 5) & 31))];
              float v1294_data = ir0[8];
              ir0[8] = (v1294_data + (v1288_data * v1292_data));
              float v1302_data = glb_m1[(v8_lead + 144)];
              float v1306_data = s0[(27 ^ ((27 >> 5) & 31))];
              float v1308_data = ir0[9];
              ir0[9] = (v1308_data + (v1302_data * v1306_data));
              float v1316_data = glb_m1[(v8_lead + 144)];
              float v1320_data = s0[(29 ^ ((29 >> 5) & 31))];
              float v1322_data = ir0[10];
              ir0[10] = (v1322_data + (v1316_data * v1320_data));
              float v1431_data = glb_m1[(v8_lead + 160)];
              float v1435_data = s0[(28 ^ ((28 >> 5) & 31))];
              float v1437_data = ir0[9];
              ir0[9] = (v1437_data + (v1431_data * v1435_data));
              float v1445_data = glb_m1[(v8_lead + 160)];
              float v1449_data = s0[(30 ^ ((30 >> 5) & 31))];
              float v1451_data = ir0[10];
              ir0[10] = (v1451_data + (v1445_data * v1449_data));
              float v1459_data = glb_m1[(v8_lead + 160)];
              float v1463_data = s0[(32 ^ ((32 >> 5) & 31))];
              float v1465_data = ir0[11];
              ir0[11] = (v1465_data + (v1459_data * v1463_data));
              float v1574_data = glb_m1[(v8_lead + 176)];
              float v1578_data = s0[(31 ^ ((31 >> 5) & 31))];
              float v1580_data = ir0[10];
              ir0[10] = (v1580_data + (v1574_data * v1578_data));
              float v1588_data = glb_m1[(v8_lead + 176)];
              float v1592_data = s0[(33 ^ ((33 >> 5) & 31))];
              float v1594_data = ir0[11];
              ir0[11] = (v1594_data + (v1588_data * v1592_data));
              float v1602_data = glb_m1[(v8_lead + 176)];
              float v1606_data = s0[(35 ^ ((35 >> 5) & 31))];
              float v1608_data = ir0[12];
              ir0[12] = (v1608_data + (v1602_data * v1606_data));
              float v1717_data = glb_m1[(v8_lead + 192)];
              float v1721_data = s0[(34 ^ ((34 >> 5) & 31))];
              float v1723_data = ir0[11];
              ir0[11] = (v1723_data + (v1717_data * v1721_data));
              float v1731_data = glb_m1[(v8_lead + 192)];
              float v1735_data = s0[(36 ^ ((36 >> 5) & 31))];
              float v1737_data = ir0[12];
              ir0[12] = (v1737_data + (v1731_data * v1735_data));
              float v1745_data = glb_m1[(v8_lead + 192)];
              float v1749_data = s0[(38 ^ ((38 >> 5) & 31))];
              float v1751_data = ir0[13];
              ir0[13] = (v1751_data + (v1745_data * v1749_data));
              float v1860_data = glb_m1[(v8_lead + 208)];
              float v1864_data = s0[(37 ^ ((37 >> 5) & 31))];
              float v1866_data = ir0[12];
              ir0[12] = (v1866_data + (v1860_data * v1864_data));
              float v1874_data = glb_m1[(v8_lead + 208)];
              float v1878_data = s0[(39 ^ ((39 >> 5) & 31))];
              float v1880_data = ir0[13];
              ir0[13] = (v1880_data + (v1874_data * v1878_data));
              float v1888_data = glb_m1[(v8_lead + 208)];
              float v1892_data = s0[(41 ^ ((41 >> 5) & 31))];
              float v1894_data = ir0[14];
              ir0[14] = (v1894_data + (v1888_data * v1892_data));
              float v2003_data = glb_m1[(v8_lead + 224)];
              float v2007_data = s0[(40 ^ ((40 >> 5) & 31))];
              float v2009_data = ir0[13];
              ir0[13] = (v2009_data + (v2003_data * v2007_data));
              float v2017_data = glb_m1[(v8_lead + 224)];
              float v2021_data = s0[(42 ^ ((42 >> 5) & 31))];
              float v2023_data = ir0[14];
              ir0[14] = (v2023_data + (v2017_data * v2021_data));
              float v2031_data = glb_m1[(v8_lead + 224)];
              float v2035_data = s0[(44 ^ ((44 >> 5) & 31))];
              float v2037_data = ir0[15];
              ir0[15] = (v2037_data + (v2031_data * v2035_data));
              float v2146_data = glb_m1[(v8_lead + 240)];
              float v2150_data = s0[(43 ^ ((43 >> 5) & 31))];
              float v2152_data = ir0[14];
              ir0[14] = (v2152_data + (v2146_data * v2150_data));
              float v2160_data = glb_m1[(v8_lead + 240)];
              float v2164_data = s0[(45 ^ ((45 >> 5) & 31))];
              float v2166_data = ir0[15];
              ir0[15] = (v2166_data + (v2160_data * v2164_data));
              #pragma unroll
              for (int32_t v2171_n0 = 0; v2171_n0 < 1; ++v2171_n0) {
                #pragma unroll
                for (int32_t v2172_n1 = 0; v2172_n1 < 16; ++v2172_n1) {
                  int32_t v2173_a = v2171_n0 + v2172_n1;
                  float v2174_data = ir0[v2173_a];
                  r0[v2173_a] = v2174_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v2179_i0 = 0; v2179_i0 < 1; ++v2179_i0) {
                int32_t v2187_lead = v8_lead + (v2179_i0 * 16);
                #pragma unroll
                for (int32_t v2180_i1 = 0; v2180_i1 < 16; ++v2180_i1) {
                  float v2182_data = r0[(v2179_i0 + v2180_i1)];
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


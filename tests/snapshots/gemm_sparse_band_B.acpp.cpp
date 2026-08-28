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
              float v16_data = s0[0];
              float v18_data = ir0[0];
              ir0[0] = (v18_data + (v15_data * v16_data));
              float v26_data = glb_m1[v8_lead];
              float v27_data = s0[2];
              float v29_data = ir0[1];
              ir0[1] = (v29_data + (v26_data * v27_data));
              float v138_data = glb_m1[(v8_lead + 16)];
              float v139_data = s0[1];
              float v141_data = ir0[0];
              ir0[0] = (v141_data + (v138_data * v139_data));
              float v149_data = glb_m1[(v8_lead + 16)];
              float v150_data = s0[3];
              float v152_data = ir0[1];
              ir0[1] = (v152_data + (v149_data * v150_data));
              float v160_data = glb_m1[(v8_lead + 16)];
              float v161_data = s0[5];
              float v163_data = ir0[2];
              ir0[2] = (v163_data + (v160_data * v161_data));
              float v272_data = glb_m1[(v8_lead + 32)];
              float v273_data = s0[4];
              float v275_data = ir0[1];
              ir0[1] = (v275_data + (v272_data * v273_data));
              float v283_data = glb_m1[(v8_lead + 32)];
              float v284_data = s0[6];
              float v286_data = ir0[2];
              ir0[2] = (v286_data + (v283_data * v284_data));
              float v294_data = glb_m1[(v8_lead + 32)];
              float v295_data = s0[8];
              float v297_data = ir0[3];
              ir0[3] = (v297_data + (v294_data * v295_data));
              float v406_data = glb_m1[(v8_lead + 48)];
              float v407_data = s0[7];
              float v409_data = ir0[2];
              ir0[2] = (v409_data + (v406_data * v407_data));
              float v417_data = glb_m1[(v8_lead + 48)];
              float v418_data = s0[9];
              float v420_data = ir0[3];
              ir0[3] = (v420_data + (v417_data * v418_data));
              float v428_data = glb_m1[(v8_lead + 48)];
              float v429_data = s0[11];
              float v431_data = ir0[4];
              ir0[4] = (v431_data + (v428_data * v429_data));
              float v540_data = glb_m1[(v8_lead + 64)];
              float v541_data = s0[10];
              float v543_data = ir0[3];
              ir0[3] = (v543_data + (v540_data * v541_data));
              float v551_data = glb_m1[(v8_lead + 64)];
              float v552_data = s0[12];
              float v554_data = ir0[4];
              ir0[4] = (v554_data + (v551_data * v552_data));
              float v562_data = glb_m1[(v8_lead + 64)];
              float v563_data = s0[14];
              float v565_data = ir0[5];
              ir0[5] = (v565_data + (v562_data * v563_data));
              float v674_data = glb_m1[(v8_lead + 80)];
              float v675_data = s0[13];
              float v677_data = ir0[4];
              ir0[4] = (v677_data + (v674_data * v675_data));
              float v685_data = glb_m1[(v8_lead + 80)];
              float v686_data = s0[15];
              float v688_data = ir0[5];
              ir0[5] = (v688_data + (v685_data * v686_data));
              float v696_data = glb_m1[(v8_lead + 80)];
              float v697_data = s0[17];
              float v699_data = ir0[6];
              ir0[6] = (v699_data + (v696_data * v697_data));
              float v808_data = glb_m1[(v8_lead + 96)];
              float v809_data = s0[16];
              float v811_data = ir0[5];
              ir0[5] = (v811_data + (v808_data * v809_data));
              float v819_data = glb_m1[(v8_lead + 96)];
              float v820_data = s0[18];
              float v822_data = ir0[6];
              ir0[6] = (v822_data + (v819_data * v820_data));
              float v830_data = glb_m1[(v8_lead + 96)];
              float v831_data = s0[20];
              float v833_data = ir0[7];
              ir0[7] = (v833_data + (v830_data * v831_data));
              float v942_data = glb_m1[(v8_lead + 112)];
              float v943_data = s0[19];
              float v945_data = ir0[6];
              ir0[6] = (v945_data + (v942_data * v943_data));
              float v953_data = glb_m1[(v8_lead + 112)];
              float v954_data = s0[21];
              float v956_data = ir0[7];
              ir0[7] = (v956_data + (v953_data * v954_data));
              float v964_data = glb_m1[(v8_lead + 112)];
              float v965_data = s0[23];
              float v967_data = ir0[8];
              ir0[8] = (v967_data + (v964_data * v965_data));
              float v1076_data = glb_m1[(v8_lead + 128)];
              float v1077_data = s0[22];
              float v1079_data = ir0[7];
              ir0[7] = (v1079_data + (v1076_data * v1077_data));
              float v1087_data = glb_m1[(v8_lead + 128)];
              float v1088_data = s0[24];
              float v1090_data = ir0[8];
              ir0[8] = (v1090_data + (v1087_data * v1088_data));
              float v1098_data = glb_m1[(v8_lead + 128)];
              float v1099_data = s0[26];
              float v1101_data = ir0[9];
              ir0[9] = (v1101_data + (v1098_data * v1099_data));
              float v1210_data = glb_m1[(v8_lead + 144)];
              float v1211_data = s0[25];
              float v1213_data = ir0[8];
              ir0[8] = (v1213_data + (v1210_data * v1211_data));
              float v1221_data = glb_m1[(v8_lead + 144)];
              float v1222_data = s0[27];
              float v1224_data = ir0[9];
              ir0[9] = (v1224_data + (v1221_data * v1222_data));
              float v1232_data = glb_m1[(v8_lead + 144)];
              float v1233_data = s0[29];
              float v1235_data = ir0[10];
              ir0[10] = (v1235_data + (v1232_data * v1233_data));
              float v1344_data = glb_m1[(v8_lead + 160)];
              float v1345_data = s0[28];
              float v1347_data = ir0[9];
              ir0[9] = (v1347_data + (v1344_data * v1345_data));
              float v1355_data = glb_m1[(v8_lead + 160)];
              float v1356_data = s0[30];
              float v1358_data = ir0[10];
              ir0[10] = (v1358_data + (v1355_data * v1356_data));
              float v1366_data = glb_m1[(v8_lead + 160)];
              float v1367_data = s0[32];
              float v1369_data = ir0[11];
              ir0[11] = (v1369_data + (v1366_data * v1367_data));
              float v1478_data = glb_m1[(v8_lead + 176)];
              float v1479_data = s0[31];
              float v1481_data = ir0[10];
              ir0[10] = (v1481_data + (v1478_data * v1479_data));
              float v1489_data = glb_m1[(v8_lead + 176)];
              float v1490_data = s0[33];
              float v1492_data = ir0[11];
              ir0[11] = (v1492_data + (v1489_data * v1490_data));
              float v1500_data = glb_m1[(v8_lead + 176)];
              float v1501_data = s0[35];
              float v1503_data = ir0[12];
              ir0[12] = (v1503_data + (v1500_data * v1501_data));
              float v1612_data = glb_m1[(v8_lead + 192)];
              float v1613_data = s0[34];
              float v1615_data = ir0[11];
              ir0[11] = (v1615_data + (v1612_data * v1613_data));
              float v1623_data = glb_m1[(v8_lead + 192)];
              float v1624_data = s0[36];
              float v1626_data = ir0[12];
              ir0[12] = (v1626_data + (v1623_data * v1624_data));
              float v1634_data = glb_m1[(v8_lead + 192)];
              float v1635_data = s0[38];
              float v1637_data = ir0[13];
              ir0[13] = (v1637_data + (v1634_data * v1635_data));
              float v1746_data = glb_m1[(v8_lead + 208)];
              float v1747_data = s0[37];
              float v1749_data = ir0[12];
              ir0[12] = (v1749_data + (v1746_data * v1747_data));
              float v1757_data = glb_m1[(v8_lead + 208)];
              float v1758_data = s0[39];
              float v1760_data = ir0[13];
              ir0[13] = (v1760_data + (v1757_data * v1758_data));
              float v1768_data = glb_m1[(v8_lead + 208)];
              float v1769_data = s0[41];
              float v1771_data = ir0[14];
              ir0[14] = (v1771_data + (v1768_data * v1769_data));
              float v1880_data = glb_m1[(v8_lead + 224)];
              float v1881_data = s0[40];
              float v1883_data = ir0[13];
              ir0[13] = (v1883_data + (v1880_data * v1881_data));
              float v1891_data = glb_m1[(v8_lead + 224)];
              float v1892_data = s0[42];
              float v1894_data = ir0[14];
              ir0[14] = (v1894_data + (v1891_data * v1892_data));
              float v1902_data = glb_m1[(v8_lead + 224)];
              float v1903_data = s0[44];
              float v1905_data = ir0[15];
              ir0[15] = (v1905_data + (v1902_data * v1903_data));
              float v2014_data = glb_m1[(v8_lead + 240)];
              float v2015_data = s0[43];
              float v2017_data = ir0[14];
              ir0[14] = (v2017_data + (v2014_data * v2015_data));
              float v2025_data = glb_m1[(v8_lead + 240)];
              float v2026_data = s0[45];
              float v2028_data = ir0[15];
              ir0[15] = (v2028_data + (v2025_data * v2026_data));
              #pragma unroll
              for (int32_t v2033_n0 = 0; v2033_n0 < 1; ++v2033_n0) {
                #pragma unroll
                for (int32_t v2034_n1 = 0; v2034_n1 < 16; ++v2034_n1) {
                  int32_t v2035_a = v2033_n0 + v2034_n1;
                  float v2036_data = ir0[v2035_a];
                  r0[v2035_a] = v2036_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v2041_i0 = 0; v2041_i0 < 1; ++v2041_i0) {
                int32_t v2049_lead = v8_lead + (v2041_i0 * 16);
                #pragma unroll
                for (int32_t v2042_i1 = 0; v2042_i1 < 16; ++v2042_i1) {
                  float v2044_data = r0[(v2041_i0 + v2042_i1)];
                  glb_m0[(v2049_lead + (v2042_i1 * 16))] = v2044_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


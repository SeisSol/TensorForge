// === base name ===
kernel_b3de4beae629cab2

// === header ===
void launcher_kernel_b3de4beae629cab2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_b3de4beae629cab2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b3de4beae629cab2(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b3de4beae629cab2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 12×20(12×20) {0..12}×{0..20} strided
        // m2 16×20(16×20) {0..16}×{0..20} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 12×20(12×20) {0..12}×{0..20} strided({0..12}×{0..20})[0, -1]×m2 16×20(16×20) {0..16}×{0..20} strided({0..16}×{0..20})[1, -1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 320 + 0 + m2_extraOffset];
              float r0[20]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(0) % 16;
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 20; ++v18_i1) {
                  float v26_data = glb_m1[(v16_lead + (v18_i1 * 12))];
                  r0[v18_i1] = v26_data;
                }
              }
              float r1[20]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v38_lead = v16_lead + (v32_i0 * 16);
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 20; ++v33_i1) {
                  float v41_data = glb_m2[(v38_lead + (v33_i1 * 16))];
                  r1[(v32_i0 + v33_i1)] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir2[16]{};
              if (v16_lead < 12) {
                float v49_data = r0[0];
                float v50_data = r1[0];
                float v53_data = ir2[0];
                ir2[0] = (v53_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 0))));
                float v59_data = ir2[1];
                ir2[1] = (v59_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 1))));
                float v65_data = ir2[2];
                ir2[2] = (v65_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 2))));
                float v71_data = ir2[3];
                ir2[3] = (v71_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 3))));
                float v77_data = ir2[4];
                ir2[4] = (v77_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 4))));
                float v83_data = ir2[5];
                ir2[5] = (v83_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 5))));
                float v89_data = ir2[6];
                ir2[6] = (v89_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 6))));
                float v95_data = ir2[7];
                ir2[7] = (v95_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 7))));
                float v101_data = ir2[8];
                ir2[8] = (v101_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 8))));
                float v107_data = ir2[9];
                ir2[9] = (v107_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 9))));
                float v113_data = ir2[10];
                ir2[10] = (v113_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 10))));
                float v119_data = ir2[11];
                ir2[11] = (v119_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 11))));
                float v125_data = ir2[12];
                ir2[12] = (v125_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 12))));
                float v131_data = ir2[13];
                ir2[13] = (v131_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 13))));
                float v137_data = ir2[14];
                ir2[14] = (v137_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 14))));
                float v143_data = ir2[15];
                ir2[15] = (v143_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 15))));
              }
              if (v16_lead < 12) {
                float v149_data = r0[1];
                float v150_data = r1[1];
                float v153_data = ir2[0];
                ir2[0] = (v153_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 0))));
                float v159_data = ir2[1];
                ir2[1] = (v159_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 1))));
                float v165_data = ir2[2];
                ir2[2] = (v165_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 2))));
                float v171_data = ir2[3];
                ir2[3] = (v171_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 3))));
                float v177_data = ir2[4];
                ir2[4] = (v177_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 4))));
                float v183_data = ir2[5];
                ir2[5] = (v183_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 5))));
                float v189_data = ir2[6];
                ir2[6] = (v189_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 6))));
                float v195_data = ir2[7];
                ir2[7] = (v195_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 7))));
                float v201_data = ir2[8];
                ir2[8] = (v201_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 8))));
                float v207_data = ir2[9];
                ir2[9] = (v207_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 9))));
                float v213_data = ir2[10];
                ir2[10] = (v213_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 10))));
                float v219_data = ir2[11];
                ir2[11] = (v219_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 11))));
                float v225_data = ir2[12];
                ir2[12] = (v225_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 12))));
                float v231_data = ir2[13];
                ir2[13] = (v231_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 13))));
                float v237_data = ir2[14];
                ir2[14] = (v237_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 14))));
                float v243_data = ir2[15];
                ir2[15] = (v243_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 15))));
              }
              if (v16_lead < 12) {
                float v249_data = r0[2];
                float v250_data = r1[2];
                float v253_data = ir2[0];
                ir2[0] = (v253_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 0))));
                float v259_data = ir2[1];
                ir2[1] = (v259_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 1))));
                float v265_data = ir2[2];
                ir2[2] = (v265_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 2))));
                float v271_data = ir2[3];
                ir2[3] = (v271_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 3))));
                float v277_data = ir2[4];
                ir2[4] = (v277_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 4))));
                float v283_data = ir2[5];
                ir2[5] = (v283_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 5))));
                float v289_data = ir2[6];
                ir2[6] = (v289_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 6))));
                float v295_data = ir2[7];
                ir2[7] = (v295_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 7))));
                float v301_data = ir2[8];
                ir2[8] = (v301_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 8))));
                float v307_data = ir2[9];
                ir2[9] = (v307_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 9))));
                float v313_data = ir2[10];
                ir2[10] = (v313_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 10))));
                float v319_data = ir2[11];
                ir2[11] = (v319_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 11))));
                float v325_data = ir2[12];
                ir2[12] = (v325_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 12))));
                float v331_data = ir2[13];
                ir2[13] = (v331_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 13))));
                float v337_data = ir2[14];
                ir2[14] = (v337_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 14))));
                float v343_data = ir2[15];
                ir2[15] = (v343_data + (v249_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 15))));
              }
              if (v16_lead < 12) {
                float v349_data = r0[3];
                float v350_data = r1[3];
                float v353_data = ir2[0];
                ir2[0] = (v353_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 0))));
                float v359_data = ir2[1];
                ir2[1] = (v359_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 1))));
                float v365_data = ir2[2];
                ir2[2] = (v365_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 2))));
                float v371_data = ir2[3];
                ir2[3] = (v371_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 3))));
                float v377_data = ir2[4];
                ir2[4] = (v377_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 4))));
                float v383_data = ir2[5];
                ir2[5] = (v383_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 5))));
                float v389_data = ir2[6];
                ir2[6] = (v389_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 6))));
                float v395_data = ir2[7];
                ir2[7] = (v395_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 7))));
                float v401_data = ir2[8];
                ir2[8] = (v401_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 8))));
                float v407_data = ir2[9];
                ir2[9] = (v407_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 9))));
                float v413_data = ir2[10];
                ir2[10] = (v413_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 10))));
                float v419_data = ir2[11];
                ir2[11] = (v419_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 11))));
                float v425_data = ir2[12];
                ir2[12] = (v425_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 12))));
                float v431_data = ir2[13];
                ir2[13] = (v431_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 13))));
                float v437_data = ir2[14];
                ir2[14] = (v437_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 14))));
                float v443_data = ir2[15];
                ir2[15] = (v443_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 15))));
              }
              if (v16_lead < 12) {
                float v449_data = r0[4];
                float v450_data = r1[4];
                float v453_data = ir2[0];
                ir2[0] = (v453_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 0))));
                float v459_data = ir2[1];
                ir2[1] = (v459_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 1))));
                float v465_data = ir2[2];
                ir2[2] = (v465_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 2))));
                float v471_data = ir2[3];
                ir2[3] = (v471_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 3))));
                float v477_data = ir2[4];
                ir2[4] = (v477_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 4))));
                float v483_data = ir2[5];
                ir2[5] = (v483_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 5))));
                float v489_data = ir2[6];
                ir2[6] = (v489_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 6))));
                float v495_data = ir2[7];
                ir2[7] = (v495_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 7))));
                float v501_data = ir2[8];
                ir2[8] = (v501_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 8))));
                float v507_data = ir2[9];
                ir2[9] = (v507_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 9))));
                float v513_data = ir2[10];
                ir2[10] = (v513_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 10))));
                float v519_data = ir2[11];
                ir2[11] = (v519_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 11))));
                float v525_data = ir2[12];
                ir2[12] = (v525_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 12))));
                float v531_data = ir2[13];
                ir2[13] = (v531_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 13))));
                float v537_data = ir2[14];
                ir2[14] = (v537_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 14))));
                float v543_data = ir2[15];
                ir2[15] = (v543_data + (v449_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 15))));
              }
              if (v16_lead < 12) {
                float v549_data = r0[5];
                float v550_data = r1[5];
                float v553_data = ir2[0];
                ir2[0] = (v553_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 0))));
                float v559_data = ir2[1];
                ir2[1] = (v559_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 1))));
                float v565_data = ir2[2];
                ir2[2] = (v565_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 2))));
                float v571_data = ir2[3];
                ir2[3] = (v571_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 3))));
                float v577_data = ir2[4];
                ir2[4] = (v577_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 4))));
                float v583_data = ir2[5];
                ir2[5] = (v583_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 5))));
                float v589_data = ir2[6];
                ir2[6] = (v589_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 6))));
                float v595_data = ir2[7];
                ir2[7] = (v595_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 7))));
                float v601_data = ir2[8];
                ir2[8] = (v601_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 8))));
                float v607_data = ir2[9];
                ir2[9] = (v607_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 9))));
                float v613_data = ir2[10];
                ir2[10] = (v613_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 10))));
                float v619_data = ir2[11];
                ir2[11] = (v619_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 11))));
                float v625_data = ir2[12];
                ir2[12] = (v625_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 12))));
                float v631_data = ir2[13];
                ir2[13] = (v631_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 13))));
                float v637_data = ir2[14];
                ir2[14] = (v637_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 14))));
                float v643_data = ir2[15];
                ir2[15] = (v643_data + (v549_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 15))));
              }
              if (v16_lead < 12) {
                float v649_data = r0[6];
                float v650_data = r1[6];
                float v653_data = ir2[0];
                ir2[0] = (v653_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 0))));
                float v659_data = ir2[1];
                ir2[1] = (v659_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 1))));
                float v665_data = ir2[2];
                ir2[2] = (v665_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 2))));
                float v671_data = ir2[3];
                ir2[3] = (v671_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 3))));
                float v677_data = ir2[4];
                ir2[4] = (v677_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 4))));
                float v683_data = ir2[5];
                ir2[5] = (v683_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 5))));
                float v689_data = ir2[6];
                ir2[6] = (v689_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 6))));
                float v695_data = ir2[7];
                ir2[7] = (v695_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 7))));
                float v701_data = ir2[8];
                ir2[8] = (v701_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 8))));
                float v707_data = ir2[9];
                ir2[9] = (v707_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 9))));
                float v713_data = ir2[10];
                ir2[10] = (v713_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 10))));
                float v719_data = ir2[11];
                ir2[11] = (v719_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 11))));
                float v725_data = ir2[12];
                ir2[12] = (v725_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 12))));
                float v731_data = ir2[13];
                ir2[13] = (v731_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 13))));
                float v737_data = ir2[14];
                ir2[14] = (v737_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 14))));
                float v743_data = ir2[15];
                ir2[15] = (v743_data + (v649_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 15))));
              }
              if (v16_lead < 12) {
                float v749_data = r0[7];
                float v750_data = r1[7];
                float v753_data = ir2[0];
                ir2[0] = (v753_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 0))));
                float v759_data = ir2[1];
                ir2[1] = (v759_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 1))));
                float v765_data = ir2[2];
                ir2[2] = (v765_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 2))));
                float v771_data = ir2[3];
                ir2[3] = (v771_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 3))));
                float v777_data = ir2[4];
                ir2[4] = (v777_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 4))));
                float v783_data = ir2[5];
                ir2[5] = (v783_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 5))));
                float v789_data = ir2[6];
                ir2[6] = (v789_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 6))));
                float v795_data = ir2[7];
                ir2[7] = (v795_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 7))));
                float v801_data = ir2[8];
                ir2[8] = (v801_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 8))));
                float v807_data = ir2[9];
                ir2[9] = (v807_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 9))));
                float v813_data = ir2[10];
                ir2[10] = (v813_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 10))));
                float v819_data = ir2[11];
                ir2[11] = (v819_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 11))));
                float v825_data = ir2[12];
                ir2[12] = (v825_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 12))));
                float v831_data = ir2[13];
                ir2[13] = (v831_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 13))));
                float v837_data = ir2[14];
                ir2[14] = (v837_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 14))));
                float v843_data = ir2[15];
                ir2[15] = (v843_data + (v749_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 15))));
              }
              if (v16_lead < 12) {
                float v849_data = r0[8];
                float v850_data = r1[8];
                float v853_data = ir2[0];
                ir2[0] = (v853_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 0))));
                float v859_data = ir2[1];
                ir2[1] = (v859_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 1))));
                float v865_data = ir2[2];
                ir2[2] = (v865_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 2))));
                float v871_data = ir2[3];
                ir2[3] = (v871_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 3))));
                float v877_data = ir2[4];
                ir2[4] = (v877_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 4))));
                float v883_data = ir2[5];
                ir2[5] = (v883_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 5))));
                float v889_data = ir2[6];
                ir2[6] = (v889_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 6))));
                float v895_data = ir2[7];
                ir2[7] = (v895_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 7))));
                float v901_data = ir2[8];
                ir2[8] = (v901_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 8))));
                float v907_data = ir2[9];
                ir2[9] = (v907_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 9))));
                float v913_data = ir2[10];
                ir2[10] = (v913_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 10))));
                float v919_data = ir2[11];
                ir2[11] = (v919_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 11))));
                float v925_data = ir2[12];
                ir2[12] = (v925_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 12))));
                float v931_data = ir2[13];
                ir2[13] = (v931_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 13))));
                float v937_data = ir2[14];
                ir2[14] = (v937_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 14))));
                float v943_data = ir2[15];
                ir2[15] = (v943_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 15))));
              }
              if (v16_lead < 12) {
                float v949_data = r0[9];
                float v950_data = r1[9];
                float v953_data = ir2[0];
                ir2[0] = (v953_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 0))));
                float v959_data = ir2[1];
                ir2[1] = (v959_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 1))));
                float v965_data = ir2[2];
                ir2[2] = (v965_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 2))));
                float v971_data = ir2[3];
                ir2[3] = (v971_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 3))));
                float v977_data = ir2[4];
                ir2[4] = (v977_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 4))));
                float v983_data = ir2[5];
                ir2[5] = (v983_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 5))));
                float v989_data = ir2[6];
                ir2[6] = (v989_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 6))));
                float v995_data = ir2[7];
                ir2[7] = (v995_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 7))));
                float v1001_data = ir2[8];
                ir2[8] = (v1001_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 8))));
                float v1007_data = ir2[9];
                ir2[9] = (v1007_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 9))));
                float v1013_data = ir2[10];
                ir2[10] = (v1013_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 10))));
                float v1019_data = ir2[11];
                ir2[11] = (v1019_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 11))));
                float v1025_data = ir2[12];
                ir2[12] = (v1025_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 12))));
                float v1031_data = ir2[13];
                ir2[13] = (v1031_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 13))));
                float v1037_data = ir2[14];
                ir2[14] = (v1037_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 14))));
                float v1043_data = ir2[15];
                ir2[15] = (v1043_data + (v949_data * (sycl::group_broadcast(item.get_sub_group(), v950_data, 15))));
              }
              if (v16_lead < 12) {
                float v1049_data = r0[10];
                float v1050_data = r1[10];
                float v1053_data = ir2[0];
                ir2[0] = (v1053_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 0))));
                float v1059_data = ir2[1];
                ir2[1] = (v1059_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 1))));
                float v1065_data = ir2[2];
                ir2[2] = (v1065_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 2))));
                float v1071_data = ir2[3];
                ir2[3] = (v1071_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 3))));
                float v1077_data = ir2[4];
                ir2[4] = (v1077_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 4))));
                float v1083_data = ir2[5];
                ir2[5] = (v1083_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 5))));
                float v1089_data = ir2[6];
                ir2[6] = (v1089_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 6))));
                float v1095_data = ir2[7];
                ir2[7] = (v1095_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 7))));
                float v1101_data = ir2[8];
                ir2[8] = (v1101_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 8))));
                float v1107_data = ir2[9];
                ir2[9] = (v1107_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 9))));
                float v1113_data = ir2[10];
                ir2[10] = (v1113_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 10))));
                float v1119_data = ir2[11];
                ir2[11] = (v1119_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 11))));
                float v1125_data = ir2[12];
                ir2[12] = (v1125_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 12))));
                float v1131_data = ir2[13];
                ir2[13] = (v1131_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 13))));
                float v1137_data = ir2[14];
                ir2[14] = (v1137_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 14))));
                float v1143_data = ir2[15];
                ir2[15] = (v1143_data + (v1049_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 15))));
              }
              if (v16_lead < 12) {
                float v1149_data = r0[11];
                float v1150_data = r1[11];
                float v1153_data = ir2[0];
                ir2[0] = (v1153_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 0))));
                float v1159_data = ir2[1];
                ir2[1] = (v1159_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 1))));
                float v1165_data = ir2[2];
                ir2[2] = (v1165_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 2))));
                float v1171_data = ir2[3];
                ir2[3] = (v1171_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 3))));
                float v1177_data = ir2[4];
                ir2[4] = (v1177_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 4))));
                float v1183_data = ir2[5];
                ir2[5] = (v1183_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 5))));
                float v1189_data = ir2[6];
                ir2[6] = (v1189_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 6))));
                float v1195_data = ir2[7];
                ir2[7] = (v1195_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 7))));
                float v1201_data = ir2[8];
                ir2[8] = (v1201_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 8))));
                float v1207_data = ir2[9];
                ir2[9] = (v1207_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 9))));
                float v1213_data = ir2[10];
                ir2[10] = (v1213_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 10))));
                float v1219_data = ir2[11];
                ir2[11] = (v1219_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 11))));
                float v1225_data = ir2[12];
                ir2[12] = (v1225_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 12))));
                float v1231_data = ir2[13];
                ir2[13] = (v1231_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 13))));
                float v1237_data = ir2[14];
                ir2[14] = (v1237_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 14))));
                float v1243_data = ir2[15];
                ir2[15] = (v1243_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 15))));
              }
              if (v16_lead < 12) {
                float v1249_data = r0[12];
                float v1250_data = r1[12];
                float v1253_data = ir2[0];
                ir2[0] = (v1253_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 0))));
                float v1259_data = ir2[1];
                ir2[1] = (v1259_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 1))));
                float v1265_data = ir2[2];
                ir2[2] = (v1265_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 2))));
                float v1271_data = ir2[3];
                ir2[3] = (v1271_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 3))));
                float v1277_data = ir2[4];
                ir2[4] = (v1277_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 4))));
                float v1283_data = ir2[5];
                ir2[5] = (v1283_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 5))));
                float v1289_data = ir2[6];
                ir2[6] = (v1289_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 6))));
                float v1295_data = ir2[7];
                ir2[7] = (v1295_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 7))));
                float v1301_data = ir2[8];
                ir2[8] = (v1301_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 8))));
                float v1307_data = ir2[9];
                ir2[9] = (v1307_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 9))));
                float v1313_data = ir2[10];
                ir2[10] = (v1313_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 10))));
                float v1319_data = ir2[11];
                ir2[11] = (v1319_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 11))));
                float v1325_data = ir2[12];
                ir2[12] = (v1325_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 12))));
                float v1331_data = ir2[13];
                ir2[13] = (v1331_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 13))));
                float v1337_data = ir2[14];
                ir2[14] = (v1337_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 14))));
                float v1343_data = ir2[15];
                ir2[15] = (v1343_data + (v1249_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 15))));
              }
              if (v16_lead < 12) {
                float v1349_data = r0[13];
                float v1350_data = r1[13];
                float v1353_data = ir2[0];
                ir2[0] = (v1353_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 0))));
                float v1359_data = ir2[1];
                ir2[1] = (v1359_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 1))));
                float v1365_data = ir2[2];
                ir2[2] = (v1365_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 2))));
                float v1371_data = ir2[3];
                ir2[3] = (v1371_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 3))));
                float v1377_data = ir2[4];
                ir2[4] = (v1377_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 4))));
                float v1383_data = ir2[5];
                ir2[5] = (v1383_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 5))));
                float v1389_data = ir2[6];
                ir2[6] = (v1389_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 6))));
                float v1395_data = ir2[7];
                ir2[7] = (v1395_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 7))));
                float v1401_data = ir2[8];
                ir2[8] = (v1401_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 8))));
                float v1407_data = ir2[9];
                ir2[9] = (v1407_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 9))));
                float v1413_data = ir2[10];
                ir2[10] = (v1413_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 10))));
                float v1419_data = ir2[11];
                ir2[11] = (v1419_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 11))));
                float v1425_data = ir2[12];
                ir2[12] = (v1425_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 12))));
                float v1431_data = ir2[13];
                ir2[13] = (v1431_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 13))));
                float v1437_data = ir2[14];
                ir2[14] = (v1437_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 14))));
                float v1443_data = ir2[15];
                ir2[15] = (v1443_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v1350_data, 15))));
              }
              if (v16_lead < 12) {
                float v1449_data = r0[14];
                float v1450_data = r1[14];
                float v1453_data = ir2[0];
                ir2[0] = (v1453_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 0))));
                float v1459_data = ir2[1];
                ir2[1] = (v1459_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 1))));
                float v1465_data = ir2[2];
                ir2[2] = (v1465_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 2))));
                float v1471_data = ir2[3];
                ir2[3] = (v1471_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 3))));
                float v1477_data = ir2[4];
                ir2[4] = (v1477_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 4))));
                float v1483_data = ir2[5];
                ir2[5] = (v1483_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 5))));
                float v1489_data = ir2[6];
                ir2[6] = (v1489_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 6))));
                float v1495_data = ir2[7];
                ir2[7] = (v1495_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 7))));
                float v1501_data = ir2[8];
                ir2[8] = (v1501_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 8))));
                float v1507_data = ir2[9];
                ir2[9] = (v1507_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 9))));
                float v1513_data = ir2[10];
                ir2[10] = (v1513_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 10))));
                float v1519_data = ir2[11];
                ir2[11] = (v1519_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 11))));
                float v1525_data = ir2[12];
                ir2[12] = (v1525_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 12))));
                float v1531_data = ir2[13];
                ir2[13] = (v1531_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 13))));
                float v1537_data = ir2[14];
                ir2[14] = (v1537_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 14))));
                float v1543_data = ir2[15];
                ir2[15] = (v1543_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 15))));
              }
              if (v16_lead < 12) {
                float v1549_data = r0[15];
                float v1550_data = r1[15];
                float v1553_data = ir2[0];
                ir2[0] = (v1553_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 0))));
                float v1559_data = ir2[1];
                ir2[1] = (v1559_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 1))));
                float v1565_data = ir2[2];
                ir2[2] = (v1565_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 2))));
                float v1571_data = ir2[3];
                ir2[3] = (v1571_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 3))));
                float v1577_data = ir2[4];
                ir2[4] = (v1577_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 4))));
                float v1583_data = ir2[5];
                ir2[5] = (v1583_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 5))));
                float v1589_data = ir2[6];
                ir2[6] = (v1589_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 6))));
                float v1595_data = ir2[7];
                ir2[7] = (v1595_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 7))));
                float v1601_data = ir2[8];
                ir2[8] = (v1601_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 8))));
                float v1607_data = ir2[9];
                ir2[9] = (v1607_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 9))));
                float v1613_data = ir2[10];
                ir2[10] = (v1613_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 10))));
                float v1619_data = ir2[11];
                ir2[11] = (v1619_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 11))));
                float v1625_data = ir2[12];
                ir2[12] = (v1625_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 12))));
                float v1631_data = ir2[13];
                ir2[13] = (v1631_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 13))));
                float v1637_data = ir2[14];
                ir2[14] = (v1637_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 14))));
                float v1643_data = ir2[15];
                ir2[15] = (v1643_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 15))));
              }
              if (v16_lead < 12) {
                float v1649_data = r0[16];
                float v1650_data = r1[16];
                float v1653_data = ir2[0];
                ir2[0] = (v1653_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 0))));
                float v1659_data = ir2[1];
                ir2[1] = (v1659_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 1))));
                float v1665_data = ir2[2];
                ir2[2] = (v1665_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 2))));
                float v1671_data = ir2[3];
                ir2[3] = (v1671_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 3))));
                float v1677_data = ir2[4];
                ir2[4] = (v1677_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 4))));
                float v1683_data = ir2[5];
                ir2[5] = (v1683_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 5))));
                float v1689_data = ir2[6];
                ir2[6] = (v1689_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 6))));
                float v1695_data = ir2[7];
                ir2[7] = (v1695_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 7))));
                float v1701_data = ir2[8];
                ir2[8] = (v1701_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 8))));
                float v1707_data = ir2[9];
                ir2[9] = (v1707_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 9))));
                float v1713_data = ir2[10];
                ir2[10] = (v1713_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 10))));
                float v1719_data = ir2[11];
                ir2[11] = (v1719_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 11))));
                float v1725_data = ir2[12];
                ir2[12] = (v1725_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 12))));
                float v1731_data = ir2[13];
                ir2[13] = (v1731_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 13))));
                float v1737_data = ir2[14];
                ir2[14] = (v1737_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 14))));
                float v1743_data = ir2[15];
                ir2[15] = (v1743_data + (v1649_data * (sycl::group_broadcast(item.get_sub_group(), v1650_data, 15))));
              }
              if (v16_lead < 12) {
                float v1749_data = r0[17];
                float v1750_data = r1[17];
                float v1753_data = ir2[0];
                ir2[0] = (v1753_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 0))));
                float v1759_data = ir2[1];
                ir2[1] = (v1759_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 1))));
                float v1765_data = ir2[2];
                ir2[2] = (v1765_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 2))));
                float v1771_data = ir2[3];
                ir2[3] = (v1771_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 3))));
                float v1777_data = ir2[4];
                ir2[4] = (v1777_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 4))));
                float v1783_data = ir2[5];
                ir2[5] = (v1783_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 5))));
                float v1789_data = ir2[6];
                ir2[6] = (v1789_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 6))));
                float v1795_data = ir2[7];
                ir2[7] = (v1795_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 7))));
                float v1801_data = ir2[8];
                ir2[8] = (v1801_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 8))));
                float v1807_data = ir2[9];
                ir2[9] = (v1807_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 9))));
                float v1813_data = ir2[10];
                ir2[10] = (v1813_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 10))));
                float v1819_data = ir2[11];
                ir2[11] = (v1819_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 11))));
                float v1825_data = ir2[12];
                ir2[12] = (v1825_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 12))));
                float v1831_data = ir2[13];
                ir2[13] = (v1831_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 13))));
                float v1837_data = ir2[14];
                ir2[14] = (v1837_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 14))));
                float v1843_data = ir2[15];
                ir2[15] = (v1843_data + (v1749_data * (sycl::group_broadcast(item.get_sub_group(), v1750_data, 15))));
              }
              if (v16_lead < 12) {
                float v1849_data = r0[18];
                float v1850_data = r1[18];
                float v1853_data = ir2[0];
                ir2[0] = (v1853_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 0))));
                float v1859_data = ir2[1];
                ir2[1] = (v1859_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 1))));
                float v1865_data = ir2[2];
                ir2[2] = (v1865_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 2))));
                float v1871_data = ir2[3];
                ir2[3] = (v1871_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 3))));
                float v1877_data = ir2[4];
                ir2[4] = (v1877_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 4))));
                float v1883_data = ir2[5];
                ir2[5] = (v1883_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 5))));
                float v1889_data = ir2[6];
                ir2[6] = (v1889_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 6))));
                float v1895_data = ir2[7];
                ir2[7] = (v1895_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 7))));
                float v1901_data = ir2[8];
                ir2[8] = (v1901_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 8))));
                float v1907_data = ir2[9];
                ir2[9] = (v1907_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 9))));
                float v1913_data = ir2[10];
                ir2[10] = (v1913_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 10))));
                float v1919_data = ir2[11];
                ir2[11] = (v1919_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 11))));
                float v1925_data = ir2[12];
                ir2[12] = (v1925_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 12))));
                float v1931_data = ir2[13];
                ir2[13] = (v1931_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 13))));
                float v1937_data = ir2[14];
                ir2[14] = (v1937_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 14))));
                float v1943_data = ir2[15];
                ir2[15] = (v1943_data + (v1849_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 15))));
              }
              if (v16_lead < 12) {
                float v1949_data = r0[19];
                float v1950_data = r1[19];
                float v1953_data = ir2[0];
                ir2[0] = (v1953_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 0))));
                float v1959_data = ir2[1];
                ir2[1] = (v1959_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 1))));
                float v1965_data = ir2[2];
                ir2[2] = (v1965_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 2))));
                float v1971_data = ir2[3];
                ir2[3] = (v1971_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 3))));
                float v1977_data = ir2[4];
                ir2[4] = (v1977_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 4))));
                float v1983_data = ir2[5];
                ir2[5] = (v1983_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 5))));
                float v1989_data = ir2[6];
                ir2[6] = (v1989_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 6))));
                float v1995_data = ir2[7];
                ir2[7] = (v1995_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 7))));
                float v2001_data = ir2[8];
                ir2[8] = (v2001_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 8))));
                float v2007_data = ir2[9];
                ir2[9] = (v2007_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 9))));
                float v2013_data = ir2[10];
                ir2[10] = (v2013_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 10))));
                float v2019_data = ir2[11];
                ir2[11] = (v2019_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 11))));
                float v2025_data = ir2[12];
                ir2[12] = (v2025_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 12))));
                float v2031_data = ir2[13];
                ir2[13] = (v2031_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 13))));
                float v2037_data = ir2[14];
                ir2[14] = (v2037_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 14))));
                float v2043_data = ir2[15];
                ir2[15] = (v2043_data + (v1949_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 15))));
              }
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v2049_n1 = 0; v2049_n1 < 16; ++v2049_n1) {
                  float v2051_data = ir2[v2049_n1];
                  r2[v2049_n1] = v2051_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v2057_i1 = 0; v2057_i1 < 16; ++v2057_i1) {
                  float v2059_data = r2[v2057_i1];
                  glb_m0[(v16_lead + (v2057_i1 * 12))] = v2059_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}


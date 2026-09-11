// === base name ===
kernel_99d12608fc1be459

// === header ===
void launcher_kernel_99d12608fc1be459(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_99d12608fc1be459(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_99d12608fc1be459(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_99d12608fc1be459(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×8(12×8) {4..16}×{0..8} strided
        // m1 16×16(12×16) {4..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(12×8) {4..16}×{0..8} strided({4..16}×{0..8})[0, 1] = m1 16×16(12×16) {4..16}×{0..16} strided({4..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(0) % 16;
              if (v16_lead < 12) {
                int32_t v25_a = (v16_lead + 4) - 4;
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
                  float v28_data = glb_m1[(v25_a + (v18_i1 * 12))];
                  r0[v18_i1] = v28_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
                int32_t v40_lead = v16_lead + (v34_i0 * 16);
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
                  float v43_data = glb_m2[(v40_lead + (v35_i1 * 16))];
                  r1[(v34_i0 + v35_i1)] = v43_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(16, 28), (0, 8)] [(0, 16)]
              float ir2[8]{};
              if (v16_lead < 12) {
                float v51_data = r0[0];
                float v52_data = r1[0];
                float v55_data = ir2[0];
                ir2[0] = (v55_data + (v51_data * (sycl::group_broadcast(item.get_sub_group(), v52_data, 0))));
                float v58_data = r1[1];
                float v61_data = ir2[1];
                ir2[1] = (v61_data + (v51_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 0))));
                float v64_data = r1[2];
                float v67_data = ir2[2];
                ir2[2] = (v67_data + (v51_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 0))));
                float v70_data = r1[3];
                float v73_data = ir2[3];
                ir2[3] = (v73_data + (v51_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 0))));
                float v76_data = r1[4];
                float v79_data = ir2[4];
                ir2[4] = (v79_data + (v51_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 0))));
                float v82_data = r1[5];
                float v85_data = ir2[5];
                ir2[5] = (v85_data + (v51_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 0))));
                float v88_data = r1[6];
                float v91_data = ir2[6];
                ir2[6] = (v91_data + (v51_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 0))));
                float v94_data = r1[7];
                float v97_data = ir2[7];
                ir2[7] = (v97_data + (v51_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 0))));
              }
              if (v16_lead < 12) {
                float v103_data = r0[1];
                float v104_data = r1[0];
                float v107_data = ir2[0];
                ir2[0] = (v107_data + (v103_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 1))));
                float v110_data = r1[1];
                float v113_data = ir2[1];
                ir2[1] = (v113_data + (v103_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 1))));
                float v116_data = r1[2];
                float v119_data = ir2[2];
                ir2[2] = (v119_data + (v103_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 1))));
                float v122_data = r1[3];
                float v125_data = ir2[3];
                ir2[3] = (v125_data + (v103_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 1))));
                float v128_data = r1[4];
                float v131_data = ir2[4];
                ir2[4] = (v131_data + (v103_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 1))));
                float v134_data = r1[5];
                float v137_data = ir2[5];
                ir2[5] = (v137_data + (v103_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 1))));
                float v140_data = r1[6];
                float v143_data = ir2[6];
                ir2[6] = (v143_data + (v103_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 1))));
                float v146_data = r1[7];
                float v149_data = ir2[7];
                ir2[7] = (v149_data + (v103_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 1))));
              }
              if (v16_lead < 12) {
                float v155_data = r0[2];
                float v156_data = r1[0];
                float v159_data = ir2[0];
                ir2[0] = (v159_data + (v155_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 2))));
                float v162_data = r1[1];
                float v165_data = ir2[1];
                ir2[1] = (v165_data + (v155_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 2))));
                float v168_data = r1[2];
                float v171_data = ir2[2];
                ir2[2] = (v171_data + (v155_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 2))));
                float v174_data = r1[3];
                float v177_data = ir2[3];
                ir2[3] = (v177_data + (v155_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 2))));
                float v180_data = r1[4];
                float v183_data = ir2[4];
                ir2[4] = (v183_data + (v155_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 2))));
                float v186_data = r1[5];
                float v189_data = ir2[5];
                ir2[5] = (v189_data + (v155_data * (sycl::group_broadcast(item.get_sub_group(), v186_data, 2))));
                float v192_data = r1[6];
                float v195_data = ir2[6];
                ir2[6] = (v195_data + (v155_data * (sycl::group_broadcast(item.get_sub_group(), v192_data, 2))));
                float v198_data = r1[7];
                float v201_data = ir2[7];
                ir2[7] = (v201_data + (v155_data * (sycl::group_broadcast(item.get_sub_group(), v198_data, 2))));
              }
              if (v16_lead < 12) {
                float v207_data = r0[3];
                float v208_data = r1[0];
                float v211_data = ir2[0];
                ir2[0] = (v211_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v208_data, 3))));
                float v214_data = r1[1];
                float v217_data = ir2[1];
                ir2[1] = (v217_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v214_data, 3))));
                float v220_data = r1[2];
                float v223_data = ir2[2];
                ir2[2] = (v223_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v220_data, 3))));
                float v226_data = r1[3];
                float v229_data = ir2[3];
                ir2[3] = (v229_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v226_data, 3))));
                float v232_data = r1[4];
                float v235_data = ir2[4];
                ir2[4] = (v235_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v232_data, 3))));
                float v238_data = r1[5];
                float v241_data = ir2[5];
                ir2[5] = (v241_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v238_data, 3))));
                float v244_data = r1[6];
                float v247_data = ir2[6];
                ir2[6] = (v247_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v244_data, 3))));
                float v250_data = r1[7];
                float v253_data = ir2[7];
                ir2[7] = (v253_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 3))));
              }
              if (v16_lead < 12) {
                float v259_data = r0[4];
                float v260_data = r1[0];
                float v263_data = ir2[0];
                ir2[0] = (v263_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v260_data, 4))));
                float v266_data = r1[1];
                float v269_data = ir2[1];
                ir2[1] = (v269_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v266_data, 4))));
                float v272_data = r1[2];
                float v275_data = ir2[2];
                ir2[2] = (v275_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v272_data, 4))));
                float v278_data = r1[3];
                float v281_data = ir2[3];
                ir2[3] = (v281_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v278_data, 4))));
                float v284_data = r1[4];
                float v287_data = ir2[4];
                ir2[4] = (v287_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v284_data, 4))));
                float v290_data = r1[5];
                float v293_data = ir2[5];
                ir2[5] = (v293_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 4))));
                float v296_data = r1[6];
                float v299_data = ir2[6];
                ir2[6] = (v299_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v296_data, 4))));
                float v302_data = r1[7];
                float v305_data = ir2[7];
                ir2[7] = (v305_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v302_data, 4))));
              }
              if (v16_lead < 12) {
                float v311_data = r0[5];
                float v312_data = r1[0];
                float v315_data = ir2[0];
                ir2[0] = (v315_data + (v311_data * (sycl::group_broadcast(item.get_sub_group(), v312_data, 5))));
                float v318_data = r1[1];
                float v321_data = ir2[1];
                ir2[1] = (v321_data + (v311_data * (sycl::group_broadcast(item.get_sub_group(), v318_data, 5))));
                float v324_data = r1[2];
                float v327_data = ir2[2];
                ir2[2] = (v327_data + (v311_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 5))));
                float v330_data = r1[3];
                float v333_data = ir2[3];
                ir2[3] = (v333_data + (v311_data * (sycl::group_broadcast(item.get_sub_group(), v330_data, 5))));
                float v336_data = r1[4];
                float v339_data = ir2[4];
                ir2[4] = (v339_data + (v311_data * (sycl::group_broadcast(item.get_sub_group(), v336_data, 5))));
                float v342_data = r1[5];
                float v345_data = ir2[5];
                ir2[5] = (v345_data + (v311_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 5))));
                float v348_data = r1[6];
                float v351_data = ir2[6];
                ir2[6] = (v351_data + (v311_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 5))));
                float v354_data = r1[7];
                float v357_data = ir2[7];
                ir2[7] = (v357_data + (v311_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 5))));
              }
              if (v16_lead < 12) {
                float v363_data = r0[6];
                float v364_data = r1[0];
                float v367_data = ir2[0];
                ir2[0] = (v367_data + (v363_data * (sycl::group_broadcast(item.get_sub_group(), v364_data, 6))));
                float v370_data = r1[1];
                float v373_data = ir2[1];
                ir2[1] = (v373_data + (v363_data * (sycl::group_broadcast(item.get_sub_group(), v370_data, 6))));
                float v376_data = r1[2];
                float v379_data = ir2[2];
                ir2[2] = (v379_data + (v363_data * (sycl::group_broadcast(item.get_sub_group(), v376_data, 6))));
                float v382_data = r1[3];
                float v385_data = ir2[3];
                ir2[3] = (v385_data + (v363_data * (sycl::group_broadcast(item.get_sub_group(), v382_data, 6))));
                float v388_data = r1[4];
                float v391_data = ir2[4];
                ir2[4] = (v391_data + (v363_data * (sycl::group_broadcast(item.get_sub_group(), v388_data, 6))));
                float v394_data = r1[5];
                float v397_data = ir2[5];
                ir2[5] = (v397_data + (v363_data * (sycl::group_broadcast(item.get_sub_group(), v394_data, 6))));
                float v400_data = r1[6];
                float v403_data = ir2[6];
                ir2[6] = (v403_data + (v363_data * (sycl::group_broadcast(item.get_sub_group(), v400_data, 6))));
                float v406_data = r1[7];
                float v409_data = ir2[7];
                ir2[7] = (v409_data + (v363_data * (sycl::group_broadcast(item.get_sub_group(), v406_data, 6))));
              }
              if (v16_lead < 12) {
                float v415_data = r0[7];
                float v416_data = r1[0];
                float v419_data = ir2[0];
                ir2[0] = (v419_data + (v415_data * (sycl::group_broadcast(item.get_sub_group(), v416_data, 7))));
                float v422_data = r1[1];
                float v425_data = ir2[1];
                ir2[1] = (v425_data + (v415_data * (sycl::group_broadcast(item.get_sub_group(), v422_data, 7))));
                float v428_data = r1[2];
                float v431_data = ir2[2];
                ir2[2] = (v431_data + (v415_data * (sycl::group_broadcast(item.get_sub_group(), v428_data, 7))));
                float v434_data = r1[3];
                float v437_data = ir2[3];
                ir2[3] = (v437_data + (v415_data * (sycl::group_broadcast(item.get_sub_group(), v434_data, 7))));
                float v440_data = r1[4];
                float v443_data = ir2[4];
                ir2[4] = (v443_data + (v415_data * (sycl::group_broadcast(item.get_sub_group(), v440_data, 7))));
                float v446_data = r1[5];
                float v449_data = ir2[5];
                ir2[5] = (v449_data + (v415_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 7))));
                float v452_data = r1[6];
                float v455_data = ir2[6];
                ir2[6] = (v455_data + (v415_data * (sycl::group_broadcast(item.get_sub_group(), v452_data, 7))));
                float v458_data = r1[7];
                float v461_data = ir2[7];
                ir2[7] = (v461_data + (v415_data * (sycl::group_broadcast(item.get_sub_group(), v458_data, 7))));
              }
              if (v16_lead < 12) {
                float v467_data = r0[8];
                float v468_data = r1[0];
                float v471_data = ir2[0];
                ir2[0] = (v471_data + (v467_data * (sycl::group_broadcast(item.get_sub_group(), v468_data, 8))));
                float v474_data = r1[1];
                float v477_data = ir2[1];
                ir2[1] = (v477_data + (v467_data * (sycl::group_broadcast(item.get_sub_group(), v474_data, 8))));
                float v480_data = r1[2];
                float v483_data = ir2[2];
                ir2[2] = (v483_data + (v467_data * (sycl::group_broadcast(item.get_sub_group(), v480_data, 8))));
                float v486_data = r1[3];
                float v489_data = ir2[3];
                ir2[3] = (v489_data + (v467_data * (sycl::group_broadcast(item.get_sub_group(), v486_data, 8))));
                float v492_data = r1[4];
                float v495_data = ir2[4];
                ir2[4] = (v495_data + (v467_data * (sycl::group_broadcast(item.get_sub_group(), v492_data, 8))));
                float v498_data = r1[5];
                float v501_data = ir2[5];
                ir2[5] = (v501_data + (v467_data * (sycl::group_broadcast(item.get_sub_group(), v498_data, 8))));
                float v504_data = r1[6];
                float v507_data = ir2[6];
                ir2[6] = (v507_data + (v467_data * (sycl::group_broadcast(item.get_sub_group(), v504_data, 8))));
                float v510_data = r1[7];
                float v513_data = ir2[7];
                ir2[7] = (v513_data + (v467_data * (sycl::group_broadcast(item.get_sub_group(), v510_data, 8))));
              }
              if (v16_lead < 12) {
                float v519_data = r0[9];
                float v520_data = r1[0];
                float v523_data = ir2[0];
                ir2[0] = (v523_data + (v519_data * (sycl::group_broadcast(item.get_sub_group(), v520_data, 9))));
                float v526_data = r1[1];
                float v529_data = ir2[1];
                ir2[1] = (v529_data + (v519_data * (sycl::group_broadcast(item.get_sub_group(), v526_data, 9))));
                float v532_data = r1[2];
                float v535_data = ir2[2];
                ir2[2] = (v535_data + (v519_data * (sycl::group_broadcast(item.get_sub_group(), v532_data, 9))));
                float v538_data = r1[3];
                float v541_data = ir2[3];
                ir2[3] = (v541_data + (v519_data * (sycl::group_broadcast(item.get_sub_group(), v538_data, 9))));
                float v544_data = r1[4];
                float v547_data = ir2[4];
                ir2[4] = (v547_data + (v519_data * (sycl::group_broadcast(item.get_sub_group(), v544_data, 9))));
                float v550_data = r1[5];
                float v553_data = ir2[5];
                ir2[5] = (v553_data + (v519_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 9))));
                float v556_data = r1[6];
                float v559_data = ir2[6];
                ir2[6] = (v559_data + (v519_data * (sycl::group_broadcast(item.get_sub_group(), v556_data, 9))));
                float v562_data = r1[7];
                float v565_data = ir2[7];
                ir2[7] = (v565_data + (v519_data * (sycl::group_broadcast(item.get_sub_group(), v562_data, 9))));
              }
              if (v16_lead < 12) {
                float v571_data = r0[10];
                float v572_data = r1[0];
                float v575_data = ir2[0];
                ir2[0] = (v575_data + (v571_data * (sycl::group_broadcast(item.get_sub_group(), v572_data, 10))));
                float v578_data = r1[1];
                float v581_data = ir2[1];
                ir2[1] = (v581_data + (v571_data * (sycl::group_broadcast(item.get_sub_group(), v578_data, 10))));
                float v584_data = r1[2];
                float v587_data = ir2[2];
                ir2[2] = (v587_data + (v571_data * (sycl::group_broadcast(item.get_sub_group(), v584_data, 10))));
                float v590_data = r1[3];
                float v593_data = ir2[3];
                ir2[3] = (v593_data + (v571_data * (sycl::group_broadcast(item.get_sub_group(), v590_data, 10))));
                float v596_data = r1[4];
                float v599_data = ir2[4];
                ir2[4] = (v599_data + (v571_data * (sycl::group_broadcast(item.get_sub_group(), v596_data, 10))));
                float v602_data = r1[5];
                float v605_data = ir2[5];
                ir2[5] = (v605_data + (v571_data * (sycl::group_broadcast(item.get_sub_group(), v602_data, 10))));
                float v608_data = r1[6];
                float v611_data = ir2[6];
                ir2[6] = (v611_data + (v571_data * (sycl::group_broadcast(item.get_sub_group(), v608_data, 10))));
                float v614_data = r1[7];
                float v617_data = ir2[7];
                ir2[7] = (v617_data + (v571_data * (sycl::group_broadcast(item.get_sub_group(), v614_data, 10))));
              }
              if (v16_lead < 12) {
                float v623_data = r0[11];
                float v624_data = r1[0];
                float v627_data = ir2[0];
                ir2[0] = (v627_data + (v623_data * (sycl::group_broadcast(item.get_sub_group(), v624_data, 11))));
                float v630_data = r1[1];
                float v633_data = ir2[1];
                ir2[1] = (v633_data + (v623_data * (sycl::group_broadcast(item.get_sub_group(), v630_data, 11))));
                float v636_data = r1[2];
                float v639_data = ir2[2];
                ir2[2] = (v639_data + (v623_data * (sycl::group_broadcast(item.get_sub_group(), v636_data, 11))));
                float v642_data = r1[3];
                float v645_data = ir2[3];
                ir2[3] = (v645_data + (v623_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 11))));
                float v648_data = r1[4];
                float v651_data = ir2[4];
                ir2[4] = (v651_data + (v623_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 11))));
                float v654_data = r1[5];
                float v657_data = ir2[5];
                ir2[5] = (v657_data + (v623_data * (sycl::group_broadcast(item.get_sub_group(), v654_data, 11))));
                float v660_data = r1[6];
                float v663_data = ir2[6];
                ir2[6] = (v663_data + (v623_data * (sycl::group_broadcast(item.get_sub_group(), v660_data, 11))));
                float v666_data = r1[7];
                float v669_data = ir2[7];
                ir2[7] = (v669_data + (v623_data * (sycl::group_broadcast(item.get_sub_group(), v666_data, 11))));
              }
              if (v16_lead < 12) {
                float v675_data = r0[12];
                float v676_data = r1[0];
                float v679_data = ir2[0];
                ir2[0] = (v679_data + (v675_data * (sycl::group_broadcast(item.get_sub_group(), v676_data, 12))));
                float v682_data = r1[1];
                float v685_data = ir2[1];
                ir2[1] = (v685_data + (v675_data * (sycl::group_broadcast(item.get_sub_group(), v682_data, 12))));
                float v688_data = r1[2];
                float v691_data = ir2[2];
                ir2[2] = (v691_data + (v675_data * (sycl::group_broadcast(item.get_sub_group(), v688_data, 12))));
                float v694_data = r1[3];
                float v697_data = ir2[3];
                ir2[3] = (v697_data + (v675_data * (sycl::group_broadcast(item.get_sub_group(), v694_data, 12))));
                float v700_data = r1[4];
                float v703_data = ir2[4];
                ir2[4] = (v703_data + (v675_data * (sycl::group_broadcast(item.get_sub_group(), v700_data, 12))));
                float v706_data = r1[5];
                float v709_data = ir2[5];
                ir2[5] = (v709_data + (v675_data * (sycl::group_broadcast(item.get_sub_group(), v706_data, 12))));
                float v712_data = r1[6];
                float v715_data = ir2[6];
                ir2[6] = (v715_data + (v675_data * (sycl::group_broadcast(item.get_sub_group(), v712_data, 12))));
                float v718_data = r1[7];
                float v721_data = ir2[7];
                ir2[7] = (v721_data + (v675_data * (sycl::group_broadcast(item.get_sub_group(), v718_data, 12))));
              }
              if (v16_lead < 12) {
                float v727_data = r0[13];
                float v728_data = r1[0];
                float v731_data = ir2[0];
                ir2[0] = (v731_data + (v727_data * (sycl::group_broadcast(item.get_sub_group(), v728_data, 13))));
                float v734_data = r1[1];
                float v737_data = ir2[1];
                ir2[1] = (v737_data + (v727_data * (sycl::group_broadcast(item.get_sub_group(), v734_data, 13))));
                float v740_data = r1[2];
                float v743_data = ir2[2];
                ir2[2] = (v743_data + (v727_data * (sycl::group_broadcast(item.get_sub_group(), v740_data, 13))));
                float v746_data = r1[3];
                float v749_data = ir2[3];
                ir2[3] = (v749_data + (v727_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 13))));
                float v752_data = r1[4];
                float v755_data = ir2[4];
                ir2[4] = (v755_data + (v727_data * (sycl::group_broadcast(item.get_sub_group(), v752_data, 13))));
                float v758_data = r1[5];
                float v761_data = ir2[5];
                ir2[5] = (v761_data + (v727_data * (sycl::group_broadcast(item.get_sub_group(), v758_data, 13))));
                float v764_data = r1[6];
                float v767_data = ir2[6];
                ir2[6] = (v767_data + (v727_data * (sycl::group_broadcast(item.get_sub_group(), v764_data, 13))));
                float v770_data = r1[7];
                float v773_data = ir2[7];
                ir2[7] = (v773_data + (v727_data * (sycl::group_broadcast(item.get_sub_group(), v770_data, 13))));
              }
              if (v16_lead < 12) {
                float v779_data = r0[14];
                float v780_data = r1[0];
                float v783_data = ir2[0];
                ir2[0] = (v783_data + (v779_data * (sycl::group_broadcast(item.get_sub_group(), v780_data, 14))));
                float v786_data = r1[1];
                float v789_data = ir2[1];
                ir2[1] = (v789_data + (v779_data * (sycl::group_broadcast(item.get_sub_group(), v786_data, 14))));
                float v792_data = r1[2];
                float v795_data = ir2[2];
                ir2[2] = (v795_data + (v779_data * (sycl::group_broadcast(item.get_sub_group(), v792_data, 14))));
                float v798_data = r1[3];
                float v801_data = ir2[3];
                ir2[3] = (v801_data + (v779_data * (sycl::group_broadcast(item.get_sub_group(), v798_data, 14))));
                float v804_data = r1[4];
                float v807_data = ir2[4];
                ir2[4] = (v807_data + (v779_data * (sycl::group_broadcast(item.get_sub_group(), v804_data, 14))));
                float v810_data = r1[5];
                float v813_data = ir2[5];
                ir2[5] = (v813_data + (v779_data * (sycl::group_broadcast(item.get_sub_group(), v810_data, 14))));
                float v816_data = r1[6];
                float v819_data = ir2[6];
                ir2[6] = (v819_data + (v779_data * (sycl::group_broadcast(item.get_sub_group(), v816_data, 14))));
                float v822_data = r1[7];
                float v825_data = ir2[7];
                ir2[7] = (v825_data + (v779_data * (sycl::group_broadcast(item.get_sub_group(), v822_data, 14))));
              }
              if (v16_lead < 12) {
                float v831_data = r0[15];
                float v832_data = r1[0];
                float v835_data = ir2[0];
                ir2[0] = (v835_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v832_data, 15))));
                float v838_data = r1[1];
                float v841_data = ir2[1];
                ir2[1] = (v841_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v838_data, 15))));
                float v844_data = r1[2];
                float v847_data = ir2[2];
                ir2[2] = (v847_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v844_data, 15))));
                float v850_data = r1[3];
                float v853_data = ir2[3];
                ir2[3] = (v853_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 15))));
                float v856_data = r1[4];
                float v859_data = ir2[4];
                ir2[4] = (v859_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v856_data, 15))));
                float v862_data = r1[5];
                float v865_data = ir2[5];
                ir2[5] = (v865_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v862_data, 15))));
                float v868_data = r1[6];
                float v871_data = ir2[6];
                ir2[6] = (v871_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v868_data, 15))));
                float v874_data = r1[7];
                float v877_data = ir2[7];
                ir2[7] = (v877_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v874_data, 15))));
              }
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v883_n1 = 0; v883_n1 < 8; ++v883_n1) {
                  float v885_data = ir2[v883_n1];
                  r2[v883_n1] = v885_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v16_lead < 12) {
                int32_t v900_a = ((v16_lead + 16_i32) + -12) - 4;
                #pragma unroll
                for (int32_t v891_i1 = 0; v891_i1 < 8; ++v891_i1) {
                  float v893_data = r2[v891_i1];
                  glb_m0[(v900_a + (v891_i1 * 12))] = v893_data;
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


// === base name ===
kernel_edb1a2398989ea63

// === header ===
void launcher_kernel_edb1a2398989ea63(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_edb1a2398989ea63(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_edb1a2398989ea63(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_edb1a2398989ea63(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×16(12×16) {0..12}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v12_lead = item.get_local_id(0) % 16;
              if (v12_lead < 12) {
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 16; ++v14_i1) {
                  float v22_data = glb_m1[(v12_lead + (v14_i1 * 12))];
                  r0[v14_i1] = v22_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
                int32_t v34_lead = v12_lead + (v28_i0 * 16);
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
                  float v37_data = glb_m2[(v34_lead + (v29_i1 * 16))];
                  r1[(v28_i0 + v29_i1)] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = load{g>r}(glb_m0);
              if (v12_lead < 12) {
                #pragma unroll
                for (int32_t v44_i1 = 0; v44_i1 < 8; ++v44_i1) {
                  float v52_data = glb_m0[(v12_lead + (v44_i1 * 12))];
                  r2[v44_i1] = v52_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              // wait(r2 = load{g>r}(glb_m0););
              float r3[8]{};
              // r3 = +(r0 * r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir3[8]{};
              if (v12_lead < 12) {
                float v60_data = r0[0];
                float v61_data = r1[0];
                float v64_data = ir3[0];
                ir3[0] = (v64_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
                float v67_data = r1[1];
                float v70_data = ir3[1];
                ir3[1] = (v70_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
                float v73_data = r1[2];
                float v76_data = ir3[2];
                ir3[2] = (v76_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
                float v79_data = r1[3];
                float v82_data = ir3[3];
                ir3[3] = (v82_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
                float v85_data = r1[4];
                float v88_data = ir3[4];
                ir3[4] = (v88_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
                float v91_data = r1[5];
                float v94_data = ir3[5];
                ir3[5] = (v94_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 0))));
                float v97_data = r1[6];
                float v100_data = ir3[6];
                ir3[6] = (v100_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 0))));
                float v103_data = r1[7];
                float v106_data = ir3[7];
                ir3[7] = (v106_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 0))));
              }
              if (v12_lead < 12) {
                float v112_data = r0[1];
                float v113_data = r1[0];
                float v116_data = ir3[0];
                ir3[0] = (v116_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 1))));
                float v119_data = r1[1];
                float v122_data = ir3[1];
                ir3[1] = (v122_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 1))));
                float v125_data = r1[2];
                float v128_data = ir3[2];
                ir3[2] = (v128_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 1))));
                float v131_data = r1[3];
                float v134_data = ir3[3];
                ir3[3] = (v134_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 1))));
                float v137_data = r1[4];
                float v140_data = ir3[4];
                ir3[4] = (v140_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 1))));
                float v143_data = r1[5];
                float v146_data = ir3[5];
                ir3[5] = (v146_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 1))));
                float v149_data = r1[6];
                float v152_data = ir3[6];
                ir3[6] = (v152_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 1))));
                float v155_data = r1[7];
                float v158_data = ir3[7];
                ir3[7] = (v158_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 1))));
              }
              if (v12_lead < 12) {
                float v164_data = r0[2];
                float v165_data = r1[0];
                float v168_data = ir3[0];
                ir3[0] = (v168_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 2))));
                float v171_data = r1[1];
                float v174_data = ir3[1];
                ir3[1] = (v174_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
                float v177_data = r1[2];
                float v180_data = ir3[2];
                ir3[2] = (v180_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 2))));
                float v183_data = r1[3];
                float v186_data = ir3[3];
                ir3[3] = (v186_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 2))));
                float v189_data = r1[4];
                float v192_data = ir3[4];
                ir3[4] = (v192_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 2))));
                float v195_data = r1[5];
                float v198_data = ir3[5];
                ir3[5] = (v198_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 2))));
                float v201_data = r1[6];
                float v204_data = ir3[6];
                ir3[6] = (v204_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 2))));
                float v207_data = r1[7];
                float v210_data = ir3[7];
                ir3[7] = (v210_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 2))));
              }
              if (v12_lead < 12) {
                float v216_data = r0[3];
                float v217_data = r1[0];
                float v220_data = ir3[0];
                ir3[0] = (v220_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v217_data, 3))));
                float v223_data = r1[1];
                float v226_data = ir3[1];
                ir3[1] = (v226_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v223_data, 3))));
                float v229_data = r1[2];
                float v232_data = ir3[2];
                ir3[2] = (v232_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v229_data, 3))));
                float v235_data = r1[3];
                float v238_data = ir3[3];
                ir3[3] = (v238_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v235_data, 3))));
                float v241_data = r1[4];
                float v244_data = ir3[4];
                ir3[4] = (v244_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v241_data, 3))));
                float v247_data = r1[5];
                float v250_data = ir3[5];
                ir3[5] = (v250_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v247_data, 3))));
                float v253_data = r1[6];
                float v256_data = ir3[6];
                ir3[6] = (v256_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v253_data, 3))));
                float v259_data = r1[7];
                float v262_data = ir3[7];
                ir3[7] = (v262_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v259_data, 3))));
              }
              if (v12_lead < 12) {
                float v268_data = r0[4];
                float v269_data = r1[0];
                float v272_data = ir3[0];
                ir3[0] = (v272_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v269_data, 4))));
                float v275_data = r1[1];
                float v278_data = ir3[1];
                ir3[1] = (v278_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 4))));
                float v281_data = r1[2];
                float v284_data = ir3[2];
                ir3[2] = (v284_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v281_data, 4))));
                float v287_data = r1[3];
                float v290_data = ir3[3];
                ir3[3] = (v290_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v287_data, 4))));
                float v293_data = r1[4];
                float v296_data = ir3[4];
                ir3[4] = (v296_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v293_data, 4))));
                float v299_data = r1[5];
                float v302_data = ir3[5];
                ir3[5] = (v302_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v299_data, 4))));
                float v305_data = r1[6];
                float v308_data = ir3[6];
                ir3[6] = (v308_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 4))));
                float v311_data = r1[7];
                float v314_data = ir3[7];
                ir3[7] = (v314_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v311_data, 4))));
              }
              if (v12_lead < 12) {
                float v320_data = r0[5];
                float v321_data = r1[0];
                float v324_data = ir3[0];
                ir3[0] = (v324_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v321_data, 5))));
                float v327_data = r1[1];
                float v330_data = ir3[1];
                ir3[1] = (v330_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 5))));
                float v333_data = r1[2];
                float v336_data = ir3[2];
                ir3[2] = (v336_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 5))));
                float v339_data = r1[3];
                float v342_data = ir3[3];
                ir3[3] = (v342_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 5))));
                float v345_data = r1[4];
                float v348_data = ir3[4];
                ir3[4] = (v348_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 5))));
                float v351_data = r1[5];
                float v354_data = ir3[5];
                ir3[5] = (v354_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v351_data, 5))));
                float v357_data = r1[6];
                float v360_data = ir3[6];
                ir3[6] = (v360_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v357_data, 5))));
                float v363_data = r1[7];
                float v366_data = ir3[7];
                ir3[7] = (v366_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v363_data, 5))));
              }
              if (v12_lead < 12) {
                float v372_data = r0[6];
                float v373_data = r1[0];
                float v376_data = ir3[0];
                ir3[0] = (v376_data + (v372_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 6))));
                float v379_data = r1[1];
                float v382_data = ir3[1];
                ir3[1] = (v382_data + (v372_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 6))));
                float v385_data = r1[2];
                float v388_data = ir3[2];
                ir3[2] = (v388_data + (v372_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 6))));
                float v391_data = r1[3];
                float v394_data = ir3[3];
                ir3[3] = (v394_data + (v372_data * (sycl::group_broadcast(item.get_sub_group(), v391_data, 6))));
                float v397_data = r1[4];
                float v400_data = ir3[4];
                ir3[4] = (v400_data + (v372_data * (sycl::group_broadcast(item.get_sub_group(), v397_data, 6))));
                float v403_data = r1[5];
                float v406_data = ir3[5];
                ir3[5] = (v406_data + (v372_data * (sycl::group_broadcast(item.get_sub_group(), v403_data, 6))));
                float v409_data = r1[6];
                float v412_data = ir3[6];
                ir3[6] = (v412_data + (v372_data * (sycl::group_broadcast(item.get_sub_group(), v409_data, 6))));
                float v415_data = r1[7];
                float v418_data = ir3[7];
                ir3[7] = (v418_data + (v372_data * (sycl::group_broadcast(item.get_sub_group(), v415_data, 6))));
              }
              if (v12_lead < 12) {
                float v424_data = r0[7];
                float v425_data = r1[0];
                float v428_data = ir3[0];
                ir3[0] = (v428_data + (v424_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 7))));
                float v431_data = r1[1];
                float v434_data = ir3[1];
                ir3[1] = (v434_data + (v424_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 7))));
                float v437_data = r1[2];
                float v440_data = ir3[2];
                ir3[2] = (v440_data + (v424_data * (sycl::group_broadcast(item.get_sub_group(), v437_data, 7))));
                float v443_data = r1[3];
                float v446_data = ir3[3];
                ir3[3] = (v446_data + (v424_data * (sycl::group_broadcast(item.get_sub_group(), v443_data, 7))));
                float v449_data = r1[4];
                float v452_data = ir3[4];
                ir3[4] = (v452_data + (v424_data * (sycl::group_broadcast(item.get_sub_group(), v449_data, 7))));
                float v455_data = r1[5];
                float v458_data = ir3[5];
                ir3[5] = (v458_data + (v424_data * (sycl::group_broadcast(item.get_sub_group(), v455_data, 7))));
                float v461_data = r1[6];
                float v464_data = ir3[6];
                ir3[6] = (v464_data + (v424_data * (sycl::group_broadcast(item.get_sub_group(), v461_data, 7))));
                float v467_data = r1[7];
                float v470_data = ir3[7];
                ir3[7] = (v470_data + (v424_data * (sycl::group_broadcast(item.get_sub_group(), v467_data, 7))));
              }
              if (v12_lead < 12) {
                float v476_data = r0[8];
                float v477_data = r1[0];
                float v480_data = ir3[0];
                ir3[0] = (v480_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v477_data, 8))));
                float v483_data = r1[1];
                float v486_data = ir3[1];
                ir3[1] = (v486_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v483_data, 8))));
                float v489_data = r1[2];
                float v492_data = ir3[2];
                ir3[2] = (v492_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v489_data, 8))));
                float v495_data = r1[3];
                float v498_data = ir3[3];
                ir3[3] = (v498_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v495_data, 8))));
                float v501_data = r1[4];
                float v504_data = ir3[4];
                ir3[4] = (v504_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v501_data, 8))));
                float v507_data = r1[5];
                float v510_data = ir3[5];
                ir3[5] = (v510_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v507_data, 8))));
                float v513_data = r1[6];
                float v516_data = ir3[6];
                ir3[6] = (v516_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v513_data, 8))));
                float v519_data = r1[7];
                float v522_data = ir3[7];
                ir3[7] = (v522_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v519_data, 8))));
              }
              if (v12_lead < 12) {
                float v528_data = r0[9];
                float v529_data = r1[0];
                float v532_data = ir3[0];
                ir3[0] = (v532_data + (v528_data * (sycl::group_broadcast(item.get_sub_group(), v529_data, 9))));
                float v535_data = r1[1];
                float v538_data = ir3[1];
                ir3[1] = (v538_data + (v528_data * (sycl::group_broadcast(item.get_sub_group(), v535_data, 9))));
                float v541_data = r1[2];
                float v544_data = ir3[2];
                ir3[2] = (v544_data + (v528_data * (sycl::group_broadcast(item.get_sub_group(), v541_data, 9))));
                float v547_data = r1[3];
                float v550_data = ir3[3];
                ir3[3] = (v550_data + (v528_data * (sycl::group_broadcast(item.get_sub_group(), v547_data, 9))));
                float v553_data = r1[4];
                float v556_data = ir3[4];
                ir3[4] = (v556_data + (v528_data * (sycl::group_broadcast(item.get_sub_group(), v553_data, 9))));
                float v559_data = r1[5];
                float v562_data = ir3[5];
                ir3[5] = (v562_data + (v528_data * (sycl::group_broadcast(item.get_sub_group(), v559_data, 9))));
                float v565_data = r1[6];
                float v568_data = ir3[6];
                ir3[6] = (v568_data + (v528_data * (sycl::group_broadcast(item.get_sub_group(), v565_data, 9))));
                float v571_data = r1[7];
                float v574_data = ir3[7];
                ir3[7] = (v574_data + (v528_data * (sycl::group_broadcast(item.get_sub_group(), v571_data, 9))));
              }
              if (v12_lead < 12) {
                float v580_data = r0[10];
                float v581_data = r1[0];
                float v584_data = ir3[0];
                ir3[0] = (v584_data + (v580_data * (sycl::group_broadcast(item.get_sub_group(), v581_data, 10))));
                float v587_data = r1[1];
                float v590_data = ir3[1];
                ir3[1] = (v590_data + (v580_data * (sycl::group_broadcast(item.get_sub_group(), v587_data, 10))));
                float v593_data = r1[2];
                float v596_data = ir3[2];
                ir3[2] = (v596_data + (v580_data * (sycl::group_broadcast(item.get_sub_group(), v593_data, 10))));
                float v599_data = r1[3];
                float v602_data = ir3[3];
                ir3[3] = (v602_data + (v580_data * (sycl::group_broadcast(item.get_sub_group(), v599_data, 10))));
                float v605_data = r1[4];
                float v608_data = ir3[4];
                ir3[4] = (v608_data + (v580_data * (sycl::group_broadcast(item.get_sub_group(), v605_data, 10))));
                float v611_data = r1[5];
                float v614_data = ir3[5];
                ir3[5] = (v614_data + (v580_data * (sycl::group_broadcast(item.get_sub_group(), v611_data, 10))));
                float v617_data = r1[6];
                float v620_data = ir3[6];
                ir3[6] = (v620_data + (v580_data * (sycl::group_broadcast(item.get_sub_group(), v617_data, 10))));
                float v623_data = r1[7];
                float v626_data = ir3[7];
                ir3[7] = (v626_data + (v580_data * (sycl::group_broadcast(item.get_sub_group(), v623_data, 10))));
              }
              if (v12_lead < 12) {
                float v632_data = r0[11];
                float v633_data = r1[0];
                float v636_data = ir3[0];
                ir3[0] = (v636_data + (v632_data * (sycl::group_broadcast(item.get_sub_group(), v633_data, 11))));
                float v639_data = r1[1];
                float v642_data = ir3[1];
                ir3[1] = (v642_data + (v632_data * (sycl::group_broadcast(item.get_sub_group(), v639_data, 11))));
                float v645_data = r1[2];
                float v648_data = ir3[2];
                ir3[2] = (v648_data + (v632_data * (sycl::group_broadcast(item.get_sub_group(), v645_data, 11))));
                float v651_data = r1[3];
                float v654_data = ir3[3];
                ir3[3] = (v654_data + (v632_data * (sycl::group_broadcast(item.get_sub_group(), v651_data, 11))));
                float v657_data = r1[4];
                float v660_data = ir3[4];
                ir3[4] = (v660_data + (v632_data * (sycl::group_broadcast(item.get_sub_group(), v657_data, 11))));
                float v663_data = r1[5];
                float v666_data = ir3[5];
                ir3[5] = (v666_data + (v632_data * (sycl::group_broadcast(item.get_sub_group(), v663_data, 11))));
                float v669_data = r1[6];
                float v672_data = ir3[6];
                ir3[6] = (v672_data + (v632_data * (sycl::group_broadcast(item.get_sub_group(), v669_data, 11))));
                float v675_data = r1[7];
                float v678_data = ir3[7];
                ir3[7] = (v678_data + (v632_data * (sycl::group_broadcast(item.get_sub_group(), v675_data, 11))));
              }
              if (v12_lead < 12) {
                float v684_data = r0[12];
                float v685_data = r1[0];
                float v688_data = ir3[0];
                ir3[0] = (v688_data + (v684_data * (sycl::group_broadcast(item.get_sub_group(), v685_data, 12))));
                float v691_data = r1[1];
                float v694_data = ir3[1];
                ir3[1] = (v694_data + (v684_data * (sycl::group_broadcast(item.get_sub_group(), v691_data, 12))));
                float v697_data = r1[2];
                float v700_data = ir3[2];
                ir3[2] = (v700_data + (v684_data * (sycl::group_broadcast(item.get_sub_group(), v697_data, 12))));
                float v703_data = r1[3];
                float v706_data = ir3[3];
                ir3[3] = (v706_data + (v684_data * (sycl::group_broadcast(item.get_sub_group(), v703_data, 12))));
                float v709_data = r1[4];
                float v712_data = ir3[4];
                ir3[4] = (v712_data + (v684_data * (sycl::group_broadcast(item.get_sub_group(), v709_data, 12))));
                float v715_data = r1[5];
                float v718_data = ir3[5];
                ir3[5] = (v718_data + (v684_data * (sycl::group_broadcast(item.get_sub_group(), v715_data, 12))));
                float v721_data = r1[6];
                float v724_data = ir3[6];
                ir3[6] = (v724_data + (v684_data * (sycl::group_broadcast(item.get_sub_group(), v721_data, 12))));
                float v727_data = r1[7];
                float v730_data = ir3[7];
                ir3[7] = (v730_data + (v684_data * (sycl::group_broadcast(item.get_sub_group(), v727_data, 12))));
              }
              if (v12_lead < 12) {
                float v736_data = r0[13];
                float v737_data = r1[0];
                float v740_data = ir3[0];
                ir3[0] = (v740_data + (v736_data * (sycl::group_broadcast(item.get_sub_group(), v737_data, 13))));
                float v743_data = r1[1];
                float v746_data = ir3[1];
                ir3[1] = (v746_data + (v736_data * (sycl::group_broadcast(item.get_sub_group(), v743_data, 13))));
                float v749_data = r1[2];
                float v752_data = ir3[2];
                ir3[2] = (v752_data + (v736_data * (sycl::group_broadcast(item.get_sub_group(), v749_data, 13))));
                float v755_data = r1[3];
                float v758_data = ir3[3];
                ir3[3] = (v758_data + (v736_data * (sycl::group_broadcast(item.get_sub_group(), v755_data, 13))));
                float v761_data = r1[4];
                float v764_data = ir3[4];
                ir3[4] = (v764_data + (v736_data * (sycl::group_broadcast(item.get_sub_group(), v761_data, 13))));
                float v767_data = r1[5];
                float v770_data = ir3[5];
                ir3[5] = (v770_data + (v736_data * (sycl::group_broadcast(item.get_sub_group(), v767_data, 13))));
                float v773_data = r1[6];
                float v776_data = ir3[6];
                ir3[6] = (v776_data + (v736_data * (sycl::group_broadcast(item.get_sub_group(), v773_data, 13))));
                float v779_data = r1[7];
                float v782_data = ir3[7];
                ir3[7] = (v782_data + (v736_data * (sycl::group_broadcast(item.get_sub_group(), v779_data, 13))));
              }
              if (v12_lead < 12) {
                float v788_data = r0[14];
                float v789_data = r1[0];
                float v792_data = ir3[0];
                ir3[0] = (v792_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v789_data, 14))));
                float v795_data = r1[1];
                float v798_data = ir3[1];
                ir3[1] = (v798_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v795_data, 14))));
                float v801_data = r1[2];
                float v804_data = ir3[2];
                ir3[2] = (v804_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v801_data, 14))));
                float v807_data = r1[3];
                float v810_data = ir3[3];
                ir3[3] = (v810_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v807_data, 14))));
                float v813_data = r1[4];
                float v816_data = ir3[4];
                ir3[4] = (v816_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v813_data, 14))));
                float v819_data = r1[5];
                float v822_data = ir3[5];
                ir3[5] = (v822_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v819_data, 14))));
                float v825_data = r1[6];
                float v828_data = ir3[6];
                ir3[6] = (v828_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v825_data, 14))));
                float v831_data = r1[7];
                float v834_data = ir3[7];
                ir3[7] = (v834_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v831_data, 14))));
              }
              if (v12_lead < 12) {
                float v840_data = r0[15];
                float v841_data = r1[0];
                float v844_data = ir3[0];
                ir3[0] = (v844_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v841_data, 15))));
                float v847_data = r1[1];
                float v850_data = ir3[1];
                ir3[1] = (v850_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v847_data, 15))));
                float v853_data = r1[2];
                float v856_data = ir3[2];
                ir3[2] = (v856_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v853_data, 15))));
                float v859_data = r1[3];
                float v862_data = ir3[3];
                ir3[3] = (v862_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v859_data, 15))));
                float v865_data = r1[4];
                float v868_data = ir3[4];
                ir3[4] = (v868_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v865_data, 15))));
                float v871_data = r1[5];
                float v874_data = ir3[5];
                ir3[5] = (v874_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v871_data, 15))));
                float v877_data = r1[6];
                float v880_data = ir3[6];
                ir3[6] = (v880_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v877_data, 15))));
                float v883_data = r1[7];
                float v886_data = ir3[7];
                ir3[7] = (v886_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v883_data, 15))));
              }
              if (v12_lead < 12) {
                #pragma unroll
                for (int32_t v892_n1 = 0; v892_n1 < 8; ++v892_n1) {
                  float v894_data = ir3[v892_n1];
                  float v896_data = r2[v892_n1];
                  r3[v892_n1] = (v896_data + v894_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v12_lead < 12) {
                #pragma unroll
                for (int32_t v903_i1 = 0; v903_i1 < 8; ++v903_i1) {
                  float v905_data = r3[v903_i1];
                  glb_m0[(v12_lead + (v903_i1 * 12))] = v905_data;
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


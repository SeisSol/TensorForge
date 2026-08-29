// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8a03a3cd0d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8a03a3cd0d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×12(12×12) {0..12}×{0..12} strided
        // m2 12×8(12×8) {0..12}×{0..8} strided
        // m3 12×12(12×12) {0..12}×{0..12} strided
        // m4 12×8(12×8) {0..12}×{0..8} strided
        // m5 12×12(12×12) {0..12}×{0..12} strided
        // m6 12×8(12×8) {0..12}×{0..8} strided
        // m7 12×12(12×12) {0..12}×{0..12} strided
        // m8 12×8(12×8) {0..12}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v14_lead = item.get_local_id(0) % 16;
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 12; ++v16_i1) {
                  float v24_data = glb_m1[(v14_lead + (v16_i1 * 12))];
                  r0[v16_i1] = v24_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              float v27_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v27_lin;
              float v28_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v28_lin;
              float v29_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v29_lin;
              float v30_lin = glb_m2[48 + item.get_local_id(0) * 1];
              r1[3] = v30_lin;
              float v31_lin = glb_m2[64 + item.get_local_id(0) * 1];
              r1[4] = v31_lin;
              float v32_lin = glb_m2[80 + item.get_local_id(0) * 1];
              r1[5] = v32_lin;
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
                  float v46_data = glb_m3[(v14_lead + (v38_i1 * 12))];
                  r3[v38_i1] = v46_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              if (v14_lead < 12) {
                float v54_data = r0[0];
                float v55_data = r1[0];
                float v58_data = ir2[0];
                ir2[0] = (v58_data + (v54_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 0))));
                float v61_data = r1[1];
                float v64_data = ir2[1];
                ir2[1] = (v64_data + (v54_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
                float v67_data = r1[2];
                float v70_data = ir2[2];
                ir2[2] = (v70_data + (v54_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
                float v73_data = r1[3];
                float v76_data = ir2[3];
                ir2[3] = (v76_data + (v54_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
                float v79_data = r1[4];
                float v82_data = ir2[4];
                ir2[4] = (v82_data + (v54_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
                float v85_data = r1[5];
                float v88_data = ir2[5];
                ir2[5] = (v88_data + (v54_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
                float v91_data = r1[6];
                float v94_data = ir2[6];
                ir2[6] = (v94_data + (v54_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 0))));
                float v97_data = r1[7];
                float v100_data = ir2[7];
                ir2[7] = (v100_data + (v54_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 0))));
              }
              if (v14_lead < 12) {
                float v106_data = r0[1];
                float v107_data = r1[0];
                float v110_data = ir2[0];
                ir2[0] = (v110_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 1))));
                float v113_data = r1[1];
                float v116_data = ir2[1];
                ir2[1] = (v116_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 1))));
                float v119_data = r1[2];
                float v122_data = ir2[2];
                ir2[2] = (v122_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 1))));
                float v125_data = r1[3];
                float v128_data = ir2[3];
                ir2[3] = (v128_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 1))));
                float v131_data = r1[4];
                float v134_data = ir2[4];
                ir2[4] = (v134_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 1))));
                float v137_data = r1[5];
                float v140_data = ir2[5];
                ir2[5] = (v140_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 1))));
                float v143_data = r1[6];
                float v146_data = ir2[6];
                ir2[6] = (v146_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 1))));
                float v149_data = r1[7];
                float v152_data = ir2[7];
                ir2[7] = (v152_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 1))));
              }
              if (v14_lead < 12) {
                float v158_data = r0[2];
                float v159_data = r1[0];
                float v162_data = ir2[0];
                ir2[0] = (v162_data + (v158_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 2))));
                float v165_data = r1[1];
                float v168_data = ir2[1];
                ir2[1] = (v168_data + (v158_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 2))));
                float v171_data = r1[2];
                float v174_data = ir2[2];
                ir2[2] = (v174_data + (v158_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
                float v177_data = r1[3];
                float v180_data = ir2[3];
                ir2[3] = (v180_data + (v158_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 2))));
                float v183_data = r1[4];
                float v186_data = ir2[4];
                ir2[4] = (v186_data + (v158_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 2))));
                float v189_data = r1[5];
                float v192_data = ir2[5];
                ir2[5] = (v192_data + (v158_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 2))));
                float v195_data = r1[6];
                float v198_data = ir2[6];
                ir2[6] = (v198_data + (v158_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 2))));
                float v201_data = r1[7];
                float v204_data = ir2[7];
                ir2[7] = (v204_data + (v158_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 2))));
              }
              if (v14_lead < 12) {
                float v210_data = r0[3];
                float v211_data = r1[0];
                float v214_data = ir2[0];
                ir2[0] = (v214_data + (v210_data * (sycl::group_broadcast(item.get_sub_group(), v211_data, 3))));
                float v217_data = r1[1];
                float v220_data = ir2[1];
                ir2[1] = (v220_data + (v210_data * (sycl::group_broadcast(item.get_sub_group(), v217_data, 3))));
                float v223_data = r1[2];
                float v226_data = ir2[2];
                ir2[2] = (v226_data + (v210_data * (sycl::group_broadcast(item.get_sub_group(), v223_data, 3))));
                float v229_data = r1[3];
                float v232_data = ir2[3];
                ir2[3] = (v232_data + (v210_data * (sycl::group_broadcast(item.get_sub_group(), v229_data, 3))));
                float v235_data = r1[4];
                float v238_data = ir2[4];
                ir2[4] = (v238_data + (v210_data * (sycl::group_broadcast(item.get_sub_group(), v235_data, 3))));
                float v241_data = r1[5];
                float v244_data = ir2[5];
                ir2[5] = (v244_data + (v210_data * (sycl::group_broadcast(item.get_sub_group(), v241_data, 3))));
                float v247_data = r1[6];
                float v250_data = ir2[6];
                ir2[6] = (v250_data + (v210_data * (sycl::group_broadcast(item.get_sub_group(), v247_data, 3))));
                float v253_data = r1[7];
                float v256_data = ir2[7];
                ir2[7] = (v256_data + (v210_data * (sycl::group_broadcast(item.get_sub_group(), v253_data, 3))));
              }
              if (v14_lead < 12) {
                float v262_data = r0[4];
                float v263_data = r1[0];
                float v266_data = ir2[0];
                ir2[0] = (v266_data + (v262_data * (sycl::group_broadcast(item.get_sub_group(), v263_data, 4))));
                float v269_data = r1[1];
                float v272_data = ir2[1];
                ir2[1] = (v272_data + (v262_data * (sycl::group_broadcast(item.get_sub_group(), v269_data, 4))));
                float v275_data = r1[2];
                float v278_data = ir2[2];
                ir2[2] = (v278_data + (v262_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 4))));
                float v281_data = r1[3];
                float v284_data = ir2[3];
                ir2[3] = (v284_data + (v262_data * (sycl::group_broadcast(item.get_sub_group(), v281_data, 4))));
                float v287_data = r1[4];
                float v290_data = ir2[4];
                ir2[4] = (v290_data + (v262_data * (sycl::group_broadcast(item.get_sub_group(), v287_data, 4))));
                float v293_data = r1[5];
                float v296_data = ir2[5];
                ir2[5] = (v296_data + (v262_data * (sycl::group_broadcast(item.get_sub_group(), v293_data, 4))));
                float v299_data = r1[6];
                float v302_data = ir2[6];
                ir2[6] = (v302_data + (v262_data * (sycl::group_broadcast(item.get_sub_group(), v299_data, 4))));
                float v305_data = r1[7];
                float v308_data = ir2[7];
                ir2[7] = (v308_data + (v262_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 4))));
              }
              if (v14_lead < 12) {
                float v314_data = r0[5];
                float v315_data = r1[0];
                float v318_data = ir2[0];
                ir2[0] = (v318_data + (v314_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 5))));
                float v321_data = r1[1];
                float v324_data = ir2[1];
                ir2[1] = (v324_data + (v314_data * (sycl::group_broadcast(item.get_sub_group(), v321_data, 5))));
                float v327_data = r1[2];
                float v330_data = ir2[2];
                ir2[2] = (v330_data + (v314_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 5))));
                float v333_data = r1[3];
                float v336_data = ir2[3];
                ir2[3] = (v336_data + (v314_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 5))));
                float v339_data = r1[4];
                float v342_data = ir2[4];
                ir2[4] = (v342_data + (v314_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 5))));
                float v345_data = r1[5];
                float v348_data = ir2[5];
                ir2[5] = (v348_data + (v314_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 5))));
                float v351_data = r1[6];
                float v354_data = ir2[6];
                ir2[6] = (v354_data + (v314_data * (sycl::group_broadcast(item.get_sub_group(), v351_data, 5))));
                float v357_data = r1[7];
                float v360_data = ir2[7];
                ir2[7] = (v360_data + (v314_data * (sycl::group_broadcast(item.get_sub_group(), v357_data, 5))));
              }
              if (v14_lead < 12) {
                float v366_data = r0[6];
                float v367_data = r1[0];
                float v370_data = ir2[0];
                ir2[0] = (v370_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 6))));
                float v373_data = r1[1];
                float v376_data = ir2[1];
                ir2[1] = (v376_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 6))));
                float v379_data = r1[2];
                float v382_data = ir2[2];
                ir2[2] = (v382_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 6))));
                float v385_data = r1[3];
                float v388_data = ir2[3];
                ir2[3] = (v388_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 6))));
                float v391_data = r1[4];
                float v394_data = ir2[4];
                ir2[4] = (v394_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v391_data, 6))));
                float v397_data = r1[5];
                float v400_data = ir2[5];
                ir2[5] = (v400_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v397_data, 6))));
                float v403_data = r1[6];
                float v406_data = ir2[6];
                ir2[6] = (v406_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v403_data, 6))));
                float v409_data = r1[7];
                float v412_data = ir2[7];
                ir2[7] = (v412_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v409_data, 6))));
              }
              if (v14_lead < 12) {
                float v418_data = r0[7];
                float v419_data = r1[0];
                float v422_data = ir2[0];
                ir2[0] = (v422_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 7))));
                float v425_data = r1[1];
                float v428_data = ir2[1];
                ir2[1] = (v428_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 7))));
                float v431_data = r1[2];
                float v434_data = ir2[2];
                ir2[2] = (v434_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 7))));
                float v437_data = r1[3];
                float v440_data = ir2[3];
                ir2[3] = (v440_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v437_data, 7))));
                float v443_data = r1[4];
                float v446_data = ir2[4];
                ir2[4] = (v446_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v443_data, 7))));
                float v449_data = r1[5];
                float v452_data = ir2[5];
                ir2[5] = (v452_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v449_data, 7))));
                float v455_data = r1[6];
                float v458_data = ir2[6];
                ir2[6] = (v458_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v455_data, 7))));
                float v461_data = r1[7];
                float v464_data = ir2[7];
                ir2[7] = (v464_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v461_data, 7))));
              }
              if (v14_lead < 12) {
                float v470_data = r0[8];
                float v471_data = r1[0];
                float v474_data = ir2[0];
                ir2[0] = (v474_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v471_data, 8))));
                float v477_data = r1[1];
                float v480_data = ir2[1];
                ir2[1] = (v480_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v477_data, 8))));
                float v483_data = r1[2];
                float v486_data = ir2[2];
                ir2[2] = (v486_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v483_data, 8))));
                float v489_data = r1[3];
                float v492_data = ir2[3];
                ir2[3] = (v492_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v489_data, 8))));
                float v495_data = r1[4];
                float v498_data = ir2[4];
                ir2[4] = (v498_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v495_data, 8))));
                float v501_data = r1[5];
                float v504_data = ir2[5];
                ir2[5] = (v504_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v501_data, 8))));
                float v507_data = r1[6];
                float v510_data = ir2[6];
                ir2[6] = (v510_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v507_data, 8))));
                float v513_data = r1[7];
                float v516_data = ir2[7];
                ir2[7] = (v516_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v513_data, 8))));
              }
              if (v14_lead < 12) {
                float v522_data = r0[9];
                float v523_data = r1[0];
                float v526_data = ir2[0];
                ir2[0] = (v526_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 9))));
                float v529_data = r1[1];
                float v532_data = ir2[1];
                ir2[1] = (v532_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v529_data, 9))));
                float v535_data = r1[2];
                float v538_data = ir2[2];
                ir2[2] = (v538_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v535_data, 9))));
                float v541_data = r1[3];
                float v544_data = ir2[3];
                ir2[3] = (v544_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v541_data, 9))));
                float v547_data = r1[4];
                float v550_data = ir2[4];
                ir2[4] = (v550_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v547_data, 9))));
                float v553_data = r1[5];
                float v556_data = ir2[5];
                ir2[5] = (v556_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v553_data, 9))));
                float v559_data = r1[6];
                float v562_data = ir2[6];
                ir2[6] = (v562_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v559_data, 9))));
                float v565_data = r1[7];
                float v568_data = ir2[7];
                ir2[7] = (v568_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v565_data, 9))));
              }
              if (v14_lead < 12) {
                float v574_data = r0[10];
                float v575_data = r1[0];
                float v578_data = ir2[0];
                ir2[0] = (v578_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v575_data, 10))));
                float v581_data = r1[1];
                float v584_data = ir2[1];
                ir2[1] = (v584_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v581_data, 10))));
                float v587_data = r1[2];
                float v590_data = ir2[2];
                ir2[2] = (v590_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v587_data, 10))));
                float v593_data = r1[3];
                float v596_data = ir2[3];
                ir2[3] = (v596_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v593_data, 10))));
                float v599_data = r1[4];
                float v602_data = ir2[4];
                ir2[4] = (v602_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v599_data, 10))));
                float v605_data = r1[5];
                float v608_data = ir2[5];
                ir2[5] = (v608_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v605_data, 10))));
                float v611_data = r1[6];
                float v614_data = ir2[6];
                ir2[6] = (v614_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v611_data, 10))));
                float v617_data = r1[7];
                float v620_data = ir2[7];
                ir2[7] = (v620_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v617_data, 10))));
              }
              if (v14_lead < 12) {
                float v626_data = r0[11];
                float v627_data = r1[0];
                float v630_data = ir2[0];
                ir2[0] = (v630_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v627_data, 11))));
                float v633_data = r1[1];
                float v636_data = ir2[1];
                ir2[1] = (v636_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v633_data, 11))));
                float v639_data = r1[2];
                float v642_data = ir2[2];
                ir2[2] = (v642_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v639_data, 11))));
                float v645_data = r1[3];
                float v648_data = ir2[3];
                ir2[3] = (v648_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v645_data, 11))));
                float v651_data = r1[4];
                float v654_data = ir2[4];
                ir2[4] = (v654_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v651_data, 11))));
                float v657_data = r1[5];
                float v660_data = ir2[5];
                ir2[5] = (v660_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v657_data, 11))));
                float v663_data = r1[6];
                float v666_data = ir2[6];
                ir2[6] = (v666_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v663_data, 11))));
                float v669_data = r1[7];
                float v672_data = ir2[7];
                ir2[7] = (v672_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v669_data, 11))));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v678_n1 = 0; v678_n1 < 8; ++v678_n1) {
                  float v680_data = ir2[v678_n1];
                  r2[v678_n1] = v680_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              float v683_lin = glb_m4[0 + item.get_local_id(0) * 1];
              r4[0] = v683_lin;
              float v684_lin = glb_m4[16 + item.get_local_id(0) * 1];
              r4[1] = v684_lin;
              float v685_lin = glb_m4[32 + item.get_local_id(0) * 1];
              r4[2] = v685_lin;
              float v686_lin = glb_m4[48 + item.get_local_id(0) * 1];
              r4[3] = v686_lin;
              float v687_lin = glb_m4[64 + item.get_local_id(0) * 1];
              r4[4] = v687_lin;
              float v688_lin = glb_m4[80 + item.get_local_id(0) * 1];
              r4[5] = v688_lin;
              // wait(r3 = load{g>r}(glb_m3););
              float r6[12]{};
              // r6 = load{g>r}(glb_m5);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v694_i1 = 0; v694_i1 < 12; ++v694_i1) {
                  float v702_data = glb_m5[(v14_lead + (v694_i1 * 12))];
                  r6[v694_i1] = v702_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[8]{};
              if (v14_lead < 12) {
                float v710_data = r3[0];
                float v711_data = r4[0];
                float v714_data = ir5[0];
                ir5[0] = (v714_data + (v710_data * (sycl::group_broadcast(item.get_sub_group(), v711_data, 0))));
                float v717_data = r4[1];
                float v720_data = ir5[1];
                ir5[1] = (v720_data + (v710_data * (sycl::group_broadcast(item.get_sub_group(), v717_data, 0))));
                float v723_data = r4[2];
                float v726_data = ir5[2];
                ir5[2] = (v726_data + (v710_data * (sycl::group_broadcast(item.get_sub_group(), v723_data, 0))));
                float v729_data = r4[3];
                float v732_data = ir5[3];
                ir5[3] = (v732_data + (v710_data * (sycl::group_broadcast(item.get_sub_group(), v729_data, 0))));
                float v735_data = r4[4];
                float v738_data = ir5[4];
                ir5[4] = (v738_data + (v710_data * (sycl::group_broadcast(item.get_sub_group(), v735_data, 0))));
                float v741_data = r4[5];
                float v744_data = ir5[5];
                ir5[5] = (v744_data + (v710_data * (sycl::group_broadcast(item.get_sub_group(), v741_data, 0))));
                float v747_data = r4[6];
                float v750_data = ir5[6];
                ir5[6] = (v750_data + (v710_data * (sycl::group_broadcast(item.get_sub_group(), v747_data, 0))));
                float v753_data = r4[7];
                float v756_data = ir5[7];
                ir5[7] = (v756_data + (v710_data * (sycl::group_broadcast(item.get_sub_group(), v753_data, 0))));
              }
              if (v14_lead < 12) {
                float v762_data = r3[1];
                float v763_data = r4[0];
                float v766_data = ir5[0];
                ir5[0] = (v766_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v763_data, 1))));
                float v769_data = r4[1];
                float v772_data = ir5[1];
                ir5[1] = (v772_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v769_data, 1))));
                float v775_data = r4[2];
                float v778_data = ir5[2];
                ir5[2] = (v778_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v775_data, 1))));
                float v781_data = r4[3];
                float v784_data = ir5[3];
                ir5[3] = (v784_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v781_data, 1))));
                float v787_data = r4[4];
                float v790_data = ir5[4];
                ir5[4] = (v790_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v787_data, 1))));
                float v793_data = r4[5];
                float v796_data = ir5[5];
                ir5[5] = (v796_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v793_data, 1))));
                float v799_data = r4[6];
                float v802_data = ir5[6];
                ir5[6] = (v802_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v799_data, 1))));
                float v805_data = r4[7];
                float v808_data = ir5[7];
                ir5[7] = (v808_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v805_data, 1))));
              }
              if (v14_lead < 12) {
                float v814_data = r3[2];
                float v815_data = r4[0];
                float v818_data = ir5[0];
                ir5[0] = (v818_data + (v814_data * (sycl::group_broadcast(item.get_sub_group(), v815_data, 2))));
                float v821_data = r4[1];
                float v824_data = ir5[1];
                ir5[1] = (v824_data + (v814_data * (sycl::group_broadcast(item.get_sub_group(), v821_data, 2))));
                float v827_data = r4[2];
                float v830_data = ir5[2];
                ir5[2] = (v830_data + (v814_data * (sycl::group_broadcast(item.get_sub_group(), v827_data, 2))));
                float v833_data = r4[3];
                float v836_data = ir5[3];
                ir5[3] = (v836_data + (v814_data * (sycl::group_broadcast(item.get_sub_group(), v833_data, 2))));
                float v839_data = r4[4];
                float v842_data = ir5[4];
                ir5[4] = (v842_data + (v814_data * (sycl::group_broadcast(item.get_sub_group(), v839_data, 2))));
                float v845_data = r4[5];
                float v848_data = ir5[5];
                ir5[5] = (v848_data + (v814_data * (sycl::group_broadcast(item.get_sub_group(), v845_data, 2))));
                float v851_data = r4[6];
                float v854_data = ir5[6];
                ir5[6] = (v854_data + (v814_data * (sycl::group_broadcast(item.get_sub_group(), v851_data, 2))));
                float v857_data = r4[7];
                float v860_data = ir5[7];
                ir5[7] = (v860_data + (v814_data * (sycl::group_broadcast(item.get_sub_group(), v857_data, 2))));
              }
              if (v14_lead < 12) {
                float v866_data = r3[3];
                float v867_data = r4[0];
                float v870_data = ir5[0];
                ir5[0] = (v870_data + (v866_data * (sycl::group_broadcast(item.get_sub_group(), v867_data, 3))));
                float v873_data = r4[1];
                float v876_data = ir5[1];
                ir5[1] = (v876_data + (v866_data * (sycl::group_broadcast(item.get_sub_group(), v873_data, 3))));
                float v879_data = r4[2];
                float v882_data = ir5[2];
                ir5[2] = (v882_data + (v866_data * (sycl::group_broadcast(item.get_sub_group(), v879_data, 3))));
                float v885_data = r4[3];
                float v888_data = ir5[3];
                ir5[3] = (v888_data + (v866_data * (sycl::group_broadcast(item.get_sub_group(), v885_data, 3))));
                float v891_data = r4[4];
                float v894_data = ir5[4];
                ir5[4] = (v894_data + (v866_data * (sycl::group_broadcast(item.get_sub_group(), v891_data, 3))));
                float v897_data = r4[5];
                float v900_data = ir5[5];
                ir5[5] = (v900_data + (v866_data * (sycl::group_broadcast(item.get_sub_group(), v897_data, 3))));
                float v903_data = r4[6];
                float v906_data = ir5[6];
                ir5[6] = (v906_data + (v866_data * (sycl::group_broadcast(item.get_sub_group(), v903_data, 3))));
                float v909_data = r4[7];
                float v912_data = ir5[7];
                ir5[7] = (v912_data + (v866_data * (sycl::group_broadcast(item.get_sub_group(), v909_data, 3))));
              }
              if (v14_lead < 12) {
                float v918_data = r3[4];
                float v919_data = r4[0];
                float v922_data = ir5[0];
                ir5[0] = (v922_data + (v918_data * (sycl::group_broadcast(item.get_sub_group(), v919_data, 4))));
                float v925_data = r4[1];
                float v928_data = ir5[1];
                ir5[1] = (v928_data + (v918_data * (sycl::group_broadcast(item.get_sub_group(), v925_data, 4))));
                float v931_data = r4[2];
                float v934_data = ir5[2];
                ir5[2] = (v934_data + (v918_data * (sycl::group_broadcast(item.get_sub_group(), v931_data, 4))));
                float v937_data = r4[3];
                float v940_data = ir5[3];
                ir5[3] = (v940_data + (v918_data * (sycl::group_broadcast(item.get_sub_group(), v937_data, 4))));
                float v943_data = r4[4];
                float v946_data = ir5[4];
                ir5[4] = (v946_data + (v918_data * (sycl::group_broadcast(item.get_sub_group(), v943_data, 4))));
                float v949_data = r4[5];
                float v952_data = ir5[5];
                ir5[5] = (v952_data + (v918_data * (sycl::group_broadcast(item.get_sub_group(), v949_data, 4))));
                float v955_data = r4[6];
                float v958_data = ir5[6];
                ir5[6] = (v958_data + (v918_data * (sycl::group_broadcast(item.get_sub_group(), v955_data, 4))));
                float v961_data = r4[7];
                float v964_data = ir5[7];
                ir5[7] = (v964_data + (v918_data * (sycl::group_broadcast(item.get_sub_group(), v961_data, 4))));
              }
              if (v14_lead < 12) {
                float v970_data = r3[5];
                float v971_data = r4[0];
                float v974_data = ir5[0];
                ir5[0] = (v974_data + (v970_data * (sycl::group_broadcast(item.get_sub_group(), v971_data, 5))));
                float v977_data = r4[1];
                float v980_data = ir5[1];
                ir5[1] = (v980_data + (v970_data * (sycl::group_broadcast(item.get_sub_group(), v977_data, 5))));
                float v983_data = r4[2];
                float v986_data = ir5[2];
                ir5[2] = (v986_data + (v970_data * (sycl::group_broadcast(item.get_sub_group(), v983_data, 5))));
                float v989_data = r4[3];
                float v992_data = ir5[3];
                ir5[3] = (v992_data + (v970_data * (sycl::group_broadcast(item.get_sub_group(), v989_data, 5))));
                float v995_data = r4[4];
                float v998_data = ir5[4];
                ir5[4] = (v998_data + (v970_data * (sycl::group_broadcast(item.get_sub_group(), v995_data, 5))));
                float v1001_data = r4[5];
                float v1004_data = ir5[5];
                ir5[5] = (v1004_data + (v970_data * (sycl::group_broadcast(item.get_sub_group(), v1001_data, 5))));
                float v1007_data = r4[6];
                float v1010_data = ir5[6];
                ir5[6] = (v1010_data + (v970_data * (sycl::group_broadcast(item.get_sub_group(), v1007_data, 5))));
                float v1013_data = r4[7];
                float v1016_data = ir5[7];
                ir5[7] = (v1016_data + (v970_data * (sycl::group_broadcast(item.get_sub_group(), v1013_data, 5))));
              }
              if (v14_lead < 12) {
                float v1022_data = r3[6];
                float v1023_data = r4[0];
                float v1026_data = ir5[0];
                ir5[0] = (v1026_data + (v1022_data * (sycl::group_broadcast(item.get_sub_group(), v1023_data, 6))));
                float v1029_data = r4[1];
                float v1032_data = ir5[1];
                ir5[1] = (v1032_data + (v1022_data * (sycl::group_broadcast(item.get_sub_group(), v1029_data, 6))));
                float v1035_data = r4[2];
                float v1038_data = ir5[2];
                ir5[2] = (v1038_data + (v1022_data * (sycl::group_broadcast(item.get_sub_group(), v1035_data, 6))));
                float v1041_data = r4[3];
                float v1044_data = ir5[3];
                ir5[3] = (v1044_data + (v1022_data * (sycl::group_broadcast(item.get_sub_group(), v1041_data, 6))));
                float v1047_data = r4[4];
                float v1050_data = ir5[4];
                ir5[4] = (v1050_data + (v1022_data * (sycl::group_broadcast(item.get_sub_group(), v1047_data, 6))));
                float v1053_data = r4[5];
                float v1056_data = ir5[5];
                ir5[5] = (v1056_data + (v1022_data * (sycl::group_broadcast(item.get_sub_group(), v1053_data, 6))));
                float v1059_data = r4[6];
                float v1062_data = ir5[6];
                ir5[6] = (v1062_data + (v1022_data * (sycl::group_broadcast(item.get_sub_group(), v1059_data, 6))));
                float v1065_data = r4[7];
                float v1068_data = ir5[7];
                ir5[7] = (v1068_data + (v1022_data * (sycl::group_broadcast(item.get_sub_group(), v1065_data, 6))));
              }
              if (v14_lead < 12) {
                float v1074_data = r3[7];
                float v1075_data = r4[0];
                float v1078_data = ir5[0];
                ir5[0] = (v1078_data + (v1074_data * (sycl::group_broadcast(item.get_sub_group(), v1075_data, 7))));
                float v1081_data = r4[1];
                float v1084_data = ir5[1];
                ir5[1] = (v1084_data + (v1074_data * (sycl::group_broadcast(item.get_sub_group(), v1081_data, 7))));
                float v1087_data = r4[2];
                float v1090_data = ir5[2];
                ir5[2] = (v1090_data + (v1074_data * (sycl::group_broadcast(item.get_sub_group(), v1087_data, 7))));
                float v1093_data = r4[3];
                float v1096_data = ir5[3];
                ir5[3] = (v1096_data + (v1074_data * (sycl::group_broadcast(item.get_sub_group(), v1093_data, 7))));
                float v1099_data = r4[4];
                float v1102_data = ir5[4];
                ir5[4] = (v1102_data + (v1074_data * (sycl::group_broadcast(item.get_sub_group(), v1099_data, 7))));
                float v1105_data = r4[5];
                float v1108_data = ir5[5];
                ir5[5] = (v1108_data + (v1074_data * (sycl::group_broadcast(item.get_sub_group(), v1105_data, 7))));
                float v1111_data = r4[6];
                float v1114_data = ir5[6];
                ir5[6] = (v1114_data + (v1074_data * (sycl::group_broadcast(item.get_sub_group(), v1111_data, 7))));
                float v1117_data = r4[7];
                float v1120_data = ir5[7];
                ir5[7] = (v1120_data + (v1074_data * (sycl::group_broadcast(item.get_sub_group(), v1117_data, 7))));
              }
              if (v14_lead < 12) {
                float v1126_data = r3[8];
                float v1127_data = r4[0];
                float v1130_data = ir5[0];
                ir5[0] = (v1130_data + (v1126_data * (sycl::group_broadcast(item.get_sub_group(), v1127_data, 8))));
                float v1133_data = r4[1];
                float v1136_data = ir5[1];
                ir5[1] = (v1136_data + (v1126_data * (sycl::group_broadcast(item.get_sub_group(), v1133_data, 8))));
                float v1139_data = r4[2];
                float v1142_data = ir5[2];
                ir5[2] = (v1142_data + (v1126_data * (sycl::group_broadcast(item.get_sub_group(), v1139_data, 8))));
                float v1145_data = r4[3];
                float v1148_data = ir5[3];
                ir5[3] = (v1148_data + (v1126_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 8))));
                float v1151_data = r4[4];
                float v1154_data = ir5[4];
                ir5[4] = (v1154_data + (v1126_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 8))));
                float v1157_data = r4[5];
                float v1160_data = ir5[5];
                ir5[5] = (v1160_data + (v1126_data * (sycl::group_broadcast(item.get_sub_group(), v1157_data, 8))));
                float v1163_data = r4[6];
                float v1166_data = ir5[6];
                ir5[6] = (v1166_data + (v1126_data * (sycl::group_broadcast(item.get_sub_group(), v1163_data, 8))));
                float v1169_data = r4[7];
                float v1172_data = ir5[7];
                ir5[7] = (v1172_data + (v1126_data * (sycl::group_broadcast(item.get_sub_group(), v1169_data, 8))));
              }
              if (v14_lead < 12) {
                float v1178_data = r3[9];
                float v1179_data = r4[0];
                float v1182_data = ir5[0];
                ir5[0] = (v1182_data + (v1178_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 9))));
                float v1185_data = r4[1];
                float v1188_data = ir5[1];
                ir5[1] = (v1188_data + (v1178_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 9))));
                float v1191_data = r4[2];
                float v1194_data = ir5[2];
                ir5[2] = (v1194_data + (v1178_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 9))));
                float v1197_data = r4[3];
                float v1200_data = ir5[3];
                ir5[3] = (v1200_data + (v1178_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 9))));
                float v1203_data = r4[4];
                float v1206_data = ir5[4];
                ir5[4] = (v1206_data + (v1178_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 9))));
                float v1209_data = r4[5];
                float v1212_data = ir5[5];
                ir5[5] = (v1212_data + (v1178_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 9))));
                float v1215_data = r4[6];
                float v1218_data = ir5[6];
                ir5[6] = (v1218_data + (v1178_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 9))));
                float v1221_data = r4[7];
                float v1224_data = ir5[7];
                ir5[7] = (v1224_data + (v1178_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 9))));
              }
              if (v14_lead < 12) {
                float v1230_data = r3[10];
                float v1231_data = r4[0];
                float v1234_data = ir5[0];
                ir5[0] = (v1234_data + (v1230_data * (sycl::group_broadcast(item.get_sub_group(), v1231_data, 10))));
                float v1237_data = r4[1];
                float v1240_data = ir5[1];
                ir5[1] = (v1240_data + (v1230_data * (sycl::group_broadcast(item.get_sub_group(), v1237_data, 10))));
                float v1243_data = r4[2];
                float v1246_data = ir5[2];
                ir5[2] = (v1246_data + (v1230_data * (sycl::group_broadcast(item.get_sub_group(), v1243_data, 10))));
                float v1249_data = r4[3];
                float v1252_data = ir5[3];
                ir5[3] = (v1252_data + (v1230_data * (sycl::group_broadcast(item.get_sub_group(), v1249_data, 10))));
                float v1255_data = r4[4];
                float v1258_data = ir5[4];
                ir5[4] = (v1258_data + (v1230_data * (sycl::group_broadcast(item.get_sub_group(), v1255_data, 10))));
                float v1261_data = r4[5];
                float v1264_data = ir5[5];
                ir5[5] = (v1264_data + (v1230_data * (sycl::group_broadcast(item.get_sub_group(), v1261_data, 10))));
                float v1267_data = r4[6];
                float v1270_data = ir5[6];
                ir5[6] = (v1270_data + (v1230_data * (sycl::group_broadcast(item.get_sub_group(), v1267_data, 10))));
                float v1273_data = r4[7];
                float v1276_data = ir5[7];
                ir5[7] = (v1276_data + (v1230_data * (sycl::group_broadcast(item.get_sub_group(), v1273_data, 10))));
              }
              if (v14_lead < 12) {
                float v1282_data = r3[11];
                float v1283_data = r4[0];
                float v1286_data = ir5[0];
                ir5[0] = (v1286_data + (v1282_data * (sycl::group_broadcast(item.get_sub_group(), v1283_data, 11))));
                float v1289_data = r4[1];
                float v1292_data = ir5[1];
                ir5[1] = (v1292_data + (v1282_data * (sycl::group_broadcast(item.get_sub_group(), v1289_data, 11))));
                float v1295_data = r4[2];
                float v1298_data = ir5[2];
                ir5[2] = (v1298_data + (v1282_data * (sycl::group_broadcast(item.get_sub_group(), v1295_data, 11))));
                float v1301_data = r4[3];
                float v1304_data = ir5[3];
                ir5[3] = (v1304_data + (v1282_data * (sycl::group_broadcast(item.get_sub_group(), v1301_data, 11))));
                float v1307_data = r4[4];
                float v1310_data = ir5[4];
                ir5[4] = (v1310_data + (v1282_data * (sycl::group_broadcast(item.get_sub_group(), v1307_data, 11))));
                float v1313_data = r4[5];
                float v1316_data = ir5[5];
                ir5[5] = (v1316_data + (v1282_data * (sycl::group_broadcast(item.get_sub_group(), v1313_data, 11))));
                float v1319_data = r4[6];
                float v1322_data = ir5[6];
                ir5[6] = (v1322_data + (v1282_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 11))));
                float v1325_data = r4[7];
                float v1328_data = ir5[7];
                ir5[7] = (v1328_data + (v1282_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 11))));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1334_n1 = 0; v1334_n1 < 8; ++v1334_n1) {
                  float v1336_data = ir5[v1334_n1];
                  float v1338_data = r2[v1334_n1];
                  r5[v1334_n1] = (v1338_data + v1336_data);
                }
              }
              float r7[8]{};
              // r7 = load{g>r}(glb_m6);
              float v1342_lin = glb_m6[0 + item.get_local_id(0) * 1];
              r7[0] = v1342_lin;
              float v1343_lin = glb_m6[16 + item.get_local_id(0) * 1];
              r7[1] = v1343_lin;
              float v1344_lin = glb_m6[32 + item.get_local_id(0) * 1];
              r7[2] = v1344_lin;
              float v1345_lin = glb_m6[48 + item.get_local_id(0) * 1];
              r7[3] = v1345_lin;
              float v1346_lin = glb_m6[64 + item.get_local_id(0) * 1];
              r7[4] = v1346_lin;
              float v1347_lin = glb_m6[80 + item.get_local_id(0) * 1];
              r7[5] = v1347_lin;
              // wait(r6 = load{g>r}(glb_m5););
              float r9[12]{};
              // r9 = load{g>r}(glb_m7);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1353_i1 = 0; v1353_i1 < 12; ++v1353_i1) {
                  float v1361_data = glb_m7[(v14_lead + (v1353_i1 * 12))];
                  r9[v1353_i1] = v1361_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m6););
              float r8[8]{};
              // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir8[8]{};
              if (v14_lead < 12) {
                float v1369_data = r6[0];
                float v1370_data = r7[0];
                float v1373_data = ir8[0];
                ir8[0] = (v1373_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1370_data, 0))));
                float v1376_data = r7[1];
                float v1379_data = ir8[1];
                ir8[1] = (v1379_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1376_data, 0))));
                float v1382_data = r7[2];
                float v1385_data = ir8[2];
                ir8[2] = (v1385_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1382_data, 0))));
                float v1388_data = r7[3];
                float v1391_data = ir8[3];
                ir8[3] = (v1391_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1388_data, 0))));
                float v1394_data = r7[4];
                float v1397_data = ir8[4];
                ir8[4] = (v1397_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1394_data, 0))));
                float v1400_data = r7[5];
                float v1403_data = ir8[5];
                ir8[5] = (v1403_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1400_data, 0))));
                float v1406_data = r7[6];
                float v1409_data = ir8[6];
                ir8[6] = (v1409_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1406_data, 0))));
                float v1412_data = r7[7];
                float v1415_data = ir8[7];
                ir8[7] = (v1415_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1412_data, 0))));
              }
              if (v14_lead < 12) {
                float v1421_data = r6[1];
                float v1422_data = r7[0];
                float v1425_data = ir8[0];
                ir8[0] = (v1425_data + (v1421_data * (sycl::group_broadcast(item.get_sub_group(), v1422_data, 1))));
                float v1428_data = r7[1];
                float v1431_data = ir8[1];
                ir8[1] = (v1431_data + (v1421_data * (sycl::group_broadcast(item.get_sub_group(), v1428_data, 1))));
                float v1434_data = r7[2];
                float v1437_data = ir8[2];
                ir8[2] = (v1437_data + (v1421_data * (sycl::group_broadcast(item.get_sub_group(), v1434_data, 1))));
                float v1440_data = r7[3];
                float v1443_data = ir8[3];
                ir8[3] = (v1443_data + (v1421_data * (sycl::group_broadcast(item.get_sub_group(), v1440_data, 1))));
                float v1446_data = r7[4];
                float v1449_data = ir8[4];
                ir8[4] = (v1449_data + (v1421_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 1))));
                float v1452_data = r7[5];
                float v1455_data = ir8[5];
                ir8[5] = (v1455_data + (v1421_data * (sycl::group_broadcast(item.get_sub_group(), v1452_data, 1))));
                float v1458_data = r7[6];
                float v1461_data = ir8[6];
                ir8[6] = (v1461_data + (v1421_data * (sycl::group_broadcast(item.get_sub_group(), v1458_data, 1))));
                float v1464_data = r7[7];
                float v1467_data = ir8[7];
                ir8[7] = (v1467_data + (v1421_data * (sycl::group_broadcast(item.get_sub_group(), v1464_data, 1))));
              }
              if (v14_lead < 12) {
                float v1473_data = r6[2];
                float v1474_data = r7[0];
                float v1477_data = ir8[0];
                ir8[0] = (v1477_data + (v1473_data * (sycl::group_broadcast(item.get_sub_group(), v1474_data, 2))));
                float v1480_data = r7[1];
                float v1483_data = ir8[1];
                ir8[1] = (v1483_data + (v1473_data * (sycl::group_broadcast(item.get_sub_group(), v1480_data, 2))));
                float v1486_data = r7[2];
                float v1489_data = ir8[2];
                ir8[2] = (v1489_data + (v1473_data * (sycl::group_broadcast(item.get_sub_group(), v1486_data, 2))));
                float v1492_data = r7[3];
                float v1495_data = ir8[3];
                ir8[3] = (v1495_data + (v1473_data * (sycl::group_broadcast(item.get_sub_group(), v1492_data, 2))));
                float v1498_data = r7[4];
                float v1501_data = ir8[4];
                ir8[4] = (v1501_data + (v1473_data * (sycl::group_broadcast(item.get_sub_group(), v1498_data, 2))));
                float v1504_data = r7[5];
                float v1507_data = ir8[5];
                ir8[5] = (v1507_data + (v1473_data * (sycl::group_broadcast(item.get_sub_group(), v1504_data, 2))));
                float v1510_data = r7[6];
                float v1513_data = ir8[6];
                ir8[6] = (v1513_data + (v1473_data * (sycl::group_broadcast(item.get_sub_group(), v1510_data, 2))));
                float v1516_data = r7[7];
                float v1519_data = ir8[7];
                ir8[7] = (v1519_data + (v1473_data * (sycl::group_broadcast(item.get_sub_group(), v1516_data, 2))));
              }
              if (v14_lead < 12) {
                float v1525_data = r6[3];
                float v1526_data = r7[0];
                float v1529_data = ir8[0];
                ir8[0] = (v1529_data + (v1525_data * (sycl::group_broadcast(item.get_sub_group(), v1526_data, 3))));
                float v1532_data = r7[1];
                float v1535_data = ir8[1];
                ir8[1] = (v1535_data + (v1525_data * (sycl::group_broadcast(item.get_sub_group(), v1532_data, 3))));
                float v1538_data = r7[2];
                float v1541_data = ir8[2];
                ir8[2] = (v1541_data + (v1525_data * (sycl::group_broadcast(item.get_sub_group(), v1538_data, 3))));
                float v1544_data = r7[3];
                float v1547_data = ir8[3];
                ir8[3] = (v1547_data + (v1525_data * (sycl::group_broadcast(item.get_sub_group(), v1544_data, 3))));
                float v1550_data = r7[4];
                float v1553_data = ir8[4];
                ir8[4] = (v1553_data + (v1525_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 3))));
                float v1556_data = r7[5];
                float v1559_data = ir8[5];
                ir8[5] = (v1559_data + (v1525_data * (sycl::group_broadcast(item.get_sub_group(), v1556_data, 3))));
                float v1562_data = r7[6];
                float v1565_data = ir8[6];
                ir8[6] = (v1565_data + (v1525_data * (sycl::group_broadcast(item.get_sub_group(), v1562_data, 3))));
                float v1568_data = r7[7];
                float v1571_data = ir8[7];
                ir8[7] = (v1571_data + (v1525_data * (sycl::group_broadcast(item.get_sub_group(), v1568_data, 3))));
              }
              if (v14_lead < 12) {
                float v1577_data = r6[4];
                float v1578_data = r7[0];
                float v1581_data = ir8[0];
                ir8[0] = (v1581_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1578_data, 4))));
                float v1584_data = r7[1];
                float v1587_data = ir8[1];
                ir8[1] = (v1587_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1584_data, 4))));
                float v1590_data = r7[2];
                float v1593_data = ir8[2];
                ir8[2] = (v1593_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1590_data, 4))));
                float v1596_data = r7[3];
                float v1599_data = ir8[3];
                ir8[3] = (v1599_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1596_data, 4))));
                float v1602_data = r7[4];
                float v1605_data = ir8[4];
                ir8[4] = (v1605_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1602_data, 4))));
                float v1608_data = r7[5];
                float v1611_data = ir8[5];
                ir8[5] = (v1611_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1608_data, 4))));
                float v1614_data = r7[6];
                float v1617_data = ir8[6];
                ir8[6] = (v1617_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1614_data, 4))));
                float v1620_data = r7[7];
                float v1623_data = ir8[7];
                ir8[7] = (v1623_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1620_data, 4))));
              }
              if (v14_lead < 12) {
                float v1629_data = r6[5];
                float v1630_data = r7[0];
                float v1633_data = ir8[0];
                ir8[0] = (v1633_data + (v1629_data * (sycl::group_broadcast(item.get_sub_group(), v1630_data, 5))));
                float v1636_data = r7[1];
                float v1639_data = ir8[1];
                ir8[1] = (v1639_data + (v1629_data * (sycl::group_broadcast(item.get_sub_group(), v1636_data, 5))));
                float v1642_data = r7[2];
                float v1645_data = ir8[2];
                ir8[2] = (v1645_data + (v1629_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 5))));
                float v1648_data = r7[3];
                float v1651_data = ir8[3];
                ir8[3] = (v1651_data + (v1629_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 5))));
                float v1654_data = r7[4];
                float v1657_data = ir8[4];
                ir8[4] = (v1657_data + (v1629_data * (sycl::group_broadcast(item.get_sub_group(), v1654_data, 5))));
                float v1660_data = r7[5];
                float v1663_data = ir8[5];
                ir8[5] = (v1663_data + (v1629_data * (sycl::group_broadcast(item.get_sub_group(), v1660_data, 5))));
                float v1666_data = r7[6];
                float v1669_data = ir8[6];
                ir8[6] = (v1669_data + (v1629_data * (sycl::group_broadcast(item.get_sub_group(), v1666_data, 5))));
                float v1672_data = r7[7];
                float v1675_data = ir8[7];
                ir8[7] = (v1675_data + (v1629_data * (sycl::group_broadcast(item.get_sub_group(), v1672_data, 5))));
              }
              if (v14_lead < 12) {
                float v1681_data = r6[6];
                float v1682_data = r7[0];
                float v1685_data = ir8[0];
                ir8[0] = (v1685_data + (v1681_data * (sycl::group_broadcast(item.get_sub_group(), v1682_data, 6))));
                float v1688_data = r7[1];
                float v1691_data = ir8[1];
                ir8[1] = (v1691_data + (v1681_data * (sycl::group_broadcast(item.get_sub_group(), v1688_data, 6))));
                float v1694_data = r7[2];
                float v1697_data = ir8[2];
                ir8[2] = (v1697_data + (v1681_data * (sycl::group_broadcast(item.get_sub_group(), v1694_data, 6))));
                float v1700_data = r7[3];
                float v1703_data = ir8[3];
                ir8[3] = (v1703_data + (v1681_data * (sycl::group_broadcast(item.get_sub_group(), v1700_data, 6))));
                float v1706_data = r7[4];
                float v1709_data = ir8[4];
                ir8[4] = (v1709_data + (v1681_data * (sycl::group_broadcast(item.get_sub_group(), v1706_data, 6))));
                float v1712_data = r7[5];
                float v1715_data = ir8[5];
                ir8[5] = (v1715_data + (v1681_data * (sycl::group_broadcast(item.get_sub_group(), v1712_data, 6))));
                float v1718_data = r7[6];
                float v1721_data = ir8[6];
                ir8[6] = (v1721_data + (v1681_data * (sycl::group_broadcast(item.get_sub_group(), v1718_data, 6))));
                float v1724_data = r7[7];
                float v1727_data = ir8[7];
                ir8[7] = (v1727_data + (v1681_data * (sycl::group_broadcast(item.get_sub_group(), v1724_data, 6))));
              }
              if (v14_lead < 12) {
                float v1733_data = r6[7];
                float v1734_data = r7[0];
                float v1737_data = ir8[0];
                ir8[0] = (v1737_data + (v1733_data * (sycl::group_broadcast(item.get_sub_group(), v1734_data, 7))));
                float v1740_data = r7[1];
                float v1743_data = ir8[1];
                ir8[1] = (v1743_data + (v1733_data * (sycl::group_broadcast(item.get_sub_group(), v1740_data, 7))));
                float v1746_data = r7[2];
                float v1749_data = ir8[2];
                ir8[2] = (v1749_data + (v1733_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 7))));
                float v1752_data = r7[3];
                float v1755_data = ir8[3];
                ir8[3] = (v1755_data + (v1733_data * (sycl::group_broadcast(item.get_sub_group(), v1752_data, 7))));
                float v1758_data = r7[4];
                float v1761_data = ir8[4];
                ir8[4] = (v1761_data + (v1733_data * (sycl::group_broadcast(item.get_sub_group(), v1758_data, 7))));
                float v1764_data = r7[5];
                float v1767_data = ir8[5];
                ir8[5] = (v1767_data + (v1733_data * (sycl::group_broadcast(item.get_sub_group(), v1764_data, 7))));
                float v1770_data = r7[6];
                float v1773_data = ir8[6];
                ir8[6] = (v1773_data + (v1733_data * (sycl::group_broadcast(item.get_sub_group(), v1770_data, 7))));
                float v1776_data = r7[7];
                float v1779_data = ir8[7];
                ir8[7] = (v1779_data + (v1733_data * (sycl::group_broadcast(item.get_sub_group(), v1776_data, 7))));
              }
              if (v14_lead < 12) {
                float v1785_data = r6[8];
                float v1786_data = r7[0];
                float v1789_data = ir8[0];
                ir8[0] = (v1789_data + (v1785_data * (sycl::group_broadcast(item.get_sub_group(), v1786_data, 8))));
                float v1792_data = r7[1];
                float v1795_data = ir8[1];
                ir8[1] = (v1795_data + (v1785_data * (sycl::group_broadcast(item.get_sub_group(), v1792_data, 8))));
                float v1798_data = r7[2];
                float v1801_data = ir8[2];
                ir8[2] = (v1801_data + (v1785_data * (sycl::group_broadcast(item.get_sub_group(), v1798_data, 8))));
                float v1804_data = r7[3];
                float v1807_data = ir8[3];
                ir8[3] = (v1807_data + (v1785_data * (sycl::group_broadcast(item.get_sub_group(), v1804_data, 8))));
                float v1810_data = r7[4];
                float v1813_data = ir8[4];
                ir8[4] = (v1813_data + (v1785_data * (sycl::group_broadcast(item.get_sub_group(), v1810_data, 8))));
                float v1816_data = r7[5];
                float v1819_data = ir8[5];
                ir8[5] = (v1819_data + (v1785_data * (sycl::group_broadcast(item.get_sub_group(), v1816_data, 8))));
                float v1822_data = r7[6];
                float v1825_data = ir8[6];
                ir8[6] = (v1825_data + (v1785_data * (sycl::group_broadcast(item.get_sub_group(), v1822_data, 8))));
                float v1828_data = r7[7];
                float v1831_data = ir8[7];
                ir8[7] = (v1831_data + (v1785_data * (sycl::group_broadcast(item.get_sub_group(), v1828_data, 8))));
              }
              if (v14_lead < 12) {
                float v1837_data = r6[9];
                float v1838_data = r7[0];
                float v1841_data = ir8[0];
                ir8[0] = (v1841_data + (v1837_data * (sycl::group_broadcast(item.get_sub_group(), v1838_data, 9))));
                float v1844_data = r7[1];
                float v1847_data = ir8[1];
                ir8[1] = (v1847_data + (v1837_data * (sycl::group_broadcast(item.get_sub_group(), v1844_data, 9))));
                float v1850_data = r7[2];
                float v1853_data = ir8[2];
                ir8[2] = (v1853_data + (v1837_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 9))));
                float v1856_data = r7[3];
                float v1859_data = ir8[3];
                ir8[3] = (v1859_data + (v1837_data * (sycl::group_broadcast(item.get_sub_group(), v1856_data, 9))));
                float v1862_data = r7[4];
                float v1865_data = ir8[4];
                ir8[4] = (v1865_data + (v1837_data * (sycl::group_broadcast(item.get_sub_group(), v1862_data, 9))));
                float v1868_data = r7[5];
                float v1871_data = ir8[5];
                ir8[5] = (v1871_data + (v1837_data * (sycl::group_broadcast(item.get_sub_group(), v1868_data, 9))));
                float v1874_data = r7[6];
                float v1877_data = ir8[6];
                ir8[6] = (v1877_data + (v1837_data * (sycl::group_broadcast(item.get_sub_group(), v1874_data, 9))));
                float v1880_data = r7[7];
                float v1883_data = ir8[7];
                ir8[7] = (v1883_data + (v1837_data * (sycl::group_broadcast(item.get_sub_group(), v1880_data, 9))));
              }
              if (v14_lead < 12) {
                float v1889_data = r6[10];
                float v1890_data = r7[0];
                float v1893_data = ir8[0];
                ir8[0] = (v1893_data + (v1889_data * (sycl::group_broadcast(item.get_sub_group(), v1890_data, 10))));
                float v1896_data = r7[1];
                float v1899_data = ir8[1];
                ir8[1] = (v1899_data + (v1889_data * (sycl::group_broadcast(item.get_sub_group(), v1896_data, 10))));
                float v1902_data = r7[2];
                float v1905_data = ir8[2];
                ir8[2] = (v1905_data + (v1889_data * (sycl::group_broadcast(item.get_sub_group(), v1902_data, 10))));
                float v1908_data = r7[3];
                float v1911_data = ir8[3];
                ir8[3] = (v1911_data + (v1889_data * (sycl::group_broadcast(item.get_sub_group(), v1908_data, 10))));
                float v1914_data = r7[4];
                float v1917_data = ir8[4];
                ir8[4] = (v1917_data + (v1889_data * (sycl::group_broadcast(item.get_sub_group(), v1914_data, 10))));
                float v1920_data = r7[5];
                float v1923_data = ir8[5];
                ir8[5] = (v1923_data + (v1889_data * (sycl::group_broadcast(item.get_sub_group(), v1920_data, 10))));
                float v1926_data = r7[6];
                float v1929_data = ir8[6];
                ir8[6] = (v1929_data + (v1889_data * (sycl::group_broadcast(item.get_sub_group(), v1926_data, 10))));
                float v1932_data = r7[7];
                float v1935_data = ir8[7];
                ir8[7] = (v1935_data + (v1889_data * (sycl::group_broadcast(item.get_sub_group(), v1932_data, 10))));
              }
              if (v14_lead < 12) {
                float v1941_data = r6[11];
                float v1942_data = r7[0];
                float v1945_data = ir8[0];
                ir8[0] = (v1945_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 11))));
                float v1948_data = r7[1];
                float v1951_data = ir8[1];
                ir8[1] = (v1951_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 11))));
                float v1954_data = r7[2];
                float v1957_data = ir8[2];
                ir8[2] = (v1957_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1954_data, 11))));
                float v1960_data = r7[3];
                float v1963_data = ir8[3];
                ir8[3] = (v1963_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1960_data, 11))));
                float v1966_data = r7[4];
                float v1969_data = ir8[4];
                ir8[4] = (v1969_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1966_data, 11))));
                float v1972_data = r7[5];
                float v1975_data = ir8[5];
                ir8[5] = (v1975_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1972_data, 11))));
                float v1978_data = r7[6];
                float v1981_data = ir8[6];
                ir8[6] = (v1981_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1978_data, 11))));
                float v1984_data = r7[7];
                float v1987_data = ir8[7];
                ir8[7] = (v1987_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1984_data, 11))));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1993_n1 = 0; v1993_n1 < 8; ++v1993_n1) {
                  float v1995_data = ir8[v1993_n1];
                  float v1997_data = r5[v1993_n1];
                  r8[v1993_n1] = (v1997_data + v1995_data);
                }
              }
              float r10[8]{};
              // r10 = load{g>r}(glb_m8);
              float v2001_lin = glb_m8[0 + item.get_local_id(0) * 1];
              r10[0] = v2001_lin;
              float v2002_lin = glb_m8[16 + item.get_local_id(0) * 1];
              r10[1] = v2002_lin;
              float v2003_lin = glb_m8[32 + item.get_local_id(0) * 1];
              r10[2] = v2003_lin;
              float v2004_lin = glb_m8[48 + item.get_local_id(0) * 1];
              r10[3] = v2004_lin;
              float v2005_lin = glb_m8[64 + item.get_local_id(0) * 1];
              r10[4] = v2005_lin;
              float v2006_lin = glb_m8[80 + item.get_local_id(0) * 1];
              r10[5] = v2006_lin;
              // wait(r9 = load{g>r}(glb_m7););
              // wait(r10 = load{g>r}(glb_m8););
              float r11[8]{};
              // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir11[8]{};
              if (v14_lead < 12) {
                float v2013_data = r9[0];
                float v2014_data = r10[0];
                float v2017_data = ir11[0];
                ir11[0] = (v2017_data + (v2013_data * (sycl::group_broadcast(item.get_sub_group(), v2014_data, 0))));
                float v2020_data = r10[1];
                float v2023_data = ir11[1];
                ir11[1] = (v2023_data + (v2013_data * (sycl::group_broadcast(item.get_sub_group(), v2020_data, 0))));
                float v2026_data = r10[2];
                float v2029_data = ir11[2];
                ir11[2] = (v2029_data + (v2013_data * (sycl::group_broadcast(item.get_sub_group(), v2026_data, 0))));
                float v2032_data = r10[3];
                float v2035_data = ir11[3];
                ir11[3] = (v2035_data + (v2013_data * (sycl::group_broadcast(item.get_sub_group(), v2032_data, 0))));
                float v2038_data = r10[4];
                float v2041_data = ir11[4];
                ir11[4] = (v2041_data + (v2013_data * (sycl::group_broadcast(item.get_sub_group(), v2038_data, 0))));
                float v2044_data = r10[5];
                float v2047_data = ir11[5];
                ir11[5] = (v2047_data + (v2013_data * (sycl::group_broadcast(item.get_sub_group(), v2044_data, 0))));
                float v2050_data = r10[6];
                float v2053_data = ir11[6];
                ir11[6] = (v2053_data + (v2013_data * (sycl::group_broadcast(item.get_sub_group(), v2050_data, 0))));
                float v2056_data = r10[7];
                float v2059_data = ir11[7];
                ir11[7] = (v2059_data + (v2013_data * (sycl::group_broadcast(item.get_sub_group(), v2056_data, 0))));
              }
              if (v14_lead < 12) {
                float v2065_data = r9[1];
                float v2066_data = r10[0];
                float v2069_data = ir11[0];
                ir11[0] = (v2069_data + (v2065_data * (sycl::group_broadcast(item.get_sub_group(), v2066_data, 1))));
                float v2072_data = r10[1];
                float v2075_data = ir11[1];
                ir11[1] = (v2075_data + (v2065_data * (sycl::group_broadcast(item.get_sub_group(), v2072_data, 1))));
                float v2078_data = r10[2];
                float v2081_data = ir11[2];
                ir11[2] = (v2081_data + (v2065_data * (sycl::group_broadcast(item.get_sub_group(), v2078_data, 1))));
                float v2084_data = r10[3];
                float v2087_data = ir11[3];
                ir11[3] = (v2087_data + (v2065_data * (sycl::group_broadcast(item.get_sub_group(), v2084_data, 1))));
                float v2090_data = r10[4];
                float v2093_data = ir11[4];
                ir11[4] = (v2093_data + (v2065_data * (sycl::group_broadcast(item.get_sub_group(), v2090_data, 1))));
                float v2096_data = r10[5];
                float v2099_data = ir11[5];
                ir11[5] = (v2099_data + (v2065_data * (sycl::group_broadcast(item.get_sub_group(), v2096_data, 1))));
                float v2102_data = r10[6];
                float v2105_data = ir11[6];
                ir11[6] = (v2105_data + (v2065_data * (sycl::group_broadcast(item.get_sub_group(), v2102_data, 1))));
                float v2108_data = r10[7];
                float v2111_data = ir11[7];
                ir11[7] = (v2111_data + (v2065_data * (sycl::group_broadcast(item.get_sub_group(), v2108_data, 1))));
              }
              if (v14_lead < 12) {
                float v2117_data = r9[2];
                float v2118_data = r10[0];
                float v2121_data = ir11[0];
                ir11[0] = (v2121_data + (v2117_data * (sycl::group_broadcast(item.get_sub_group(), v2118_data, 2))));
                float v2124_data = r10[1];
                float v2127_data = ir11[1];
                ir11[1] = (v2127_data + (v2117_data * (sycl::group_broadcast(item.get_sub_group(), v2124_data, 2))));
                float v2130_data = r10[2];
                float v2133_data = ir11[2];
                ir11[2] = (v2133_data + (v2117_data * (sycl::group_broadcast(item.get_sub_group(), v2130_data, 2))));
                float v2136_data = r10[3];
                float v2139_data = ir11[3];
                ir11[3] = (v2139_data + (v2117_data * (sycl::group_broadcast(item.get_sub_group(), v2136_data, 2))));
                float v2142_data = r10[4];
                float v2145_data = ir11[4];
                ir11[4] = (v2145_data + (v2117_data * (sycl::group_broadcast(item.get_sub_group(), v2142_data, 2))));
                float v2148_data = r10[5];
                float v2151_data = ir11[5];
                ir11[5] = (v2151_data + (v2117_data * (sycl::group_broadcast(item.get_sub_group(), v2148_data, 2))));
                float v2154_data = r10[6];
                float v2157_data = ir11[6];
                ir11[6] = (v2157_data + (v2117_data * (sycl::group_broadcast(item.get_sub_group(), v2154_data, 2))));
                float v2160_data = r10[7];
                float v2163_data = ir11[7];
                ir11[7] = (v2163_data + (v2117_data * (sycl::group_broadcast(item.get_sub_group(), v2160_data, 2))));
              }
              if (v14_lead < 12) {
                float v2169_data = r9[3];
                float v2170_data = r10[0];
                float v2173_data = ir11[0];
                ir11[0] = (v2173_data + (v2169_data * (sycl::group_broadcast(item.get_sub_group(), v2170_data, 3))));
                float v2176_data = r10[1];
                float v2179_data = ir11[1];
                ir11[1] = (v2179_data + (v2169_data * (sycl::group_broadcast(item.get_sub_group(), v2176_data, 3))));
                float v2182_data = r10[2];
                float v2185_data = ir11[2];
                ir11[2] = (v2185_data + (v2169_data * (sycl::group_broadcast(item.get_sub_group(), v2182_data, 3))));
                float v2188_data = r10[3];
                float v2191_data = ir11[3];
                ir11[3] = (v2191_data + (v2169_data * (sycl::group_broadcast(item.get_sub_group(), v2188_data, 3))));
                float v2194_data = r10[4];
                float v2197_data = ir11[4];
                ir11[4] = (v2197_data + (v2169_data * (sycl::group_broadcast(item.get_sub_group(), v2194_data, 3))));
                float v2200_data = r10[5];
                float v2203_data = ir11[5];
                ir11[5] = (v2203_data + (v2169_data * (sycl::group_broadcast(item.get_sub_group(), v2200_data, 3))));
                float v2206_data = r10[6];
                float v2209_data = ir11[6];
                ir11[6] = (v2209_data + (v2169_data * (sycl::group_broadcast(item.get_sub_group(), v2206_data, 3))));
                float v2212_data = r10[7];
                float v2215_data = ir11[7];
                ir11[7] = (v2215_data + (v2169_data * (sycl::group_broadcast(item.get_sub_group(), v2212_data, 3))));
              }
              if (v14_lead < 12) {
                float v2221_data = r9[4];
                float v2222_data = r10[0];
                float v2225_data = ir11[0];
                ir11[0] = (v2225_data + (v2221_data * (sycl::group_broadcast(item.get_sub_group(), v2222_data, 4))));
                float v2228_data = r10[1];
                float v2231_data = ir11[1];
                ir11[1] = (v2231_data + (v2221_data * (sycl::group_broadcast(item.get_sub_group(), v2228_data, 4))));
                float v2234_data = r10[2];
                float v2237_data = ir11[2];
                ir11[2] = (v2237_data + (v2221_data * (sycl::group_broadcast(item.get_sub_group(), v2234_data, 4))));
                float v2240_data = r10[3];
                float v2243_data = ir11[3];
                ir11[3] = (v2243_data + (v2221_data * (sycl::group_broadcast(item.get_sub_group(), v2240_data, 4))));
                float v2246_data = r10[4];
                float v2249_data = ir11[4];
                ir11[4] = (v2249_data + (v2221_data * (sycl::group_broadcast(item.get_sub_group(), v2246_data, 4))));
                float v2252_data = r10[5];
                float v2255_data = ir11[5];
                ir11[5] = (v2255_data + (v2221_data * (sycl::group_broadcast(item.get_sub_group(), v2252_data, 4))));
                float v2258_data = r10[6];
                float v2261_data = ir11[6];
                ir11[6] = (v2261_data + (v2221_data * (sycl::group_broadcast(item.get_sub_group(), v2258_data, 4))));
                float v2264_data = r10[7];
                float v2267_data = ir11[7];
                ir11[7] = (v2267_data + (v2221_data * (sycl::group_broadcast(item.get_sub_group(), v2264_data, 4))));
              }
              if (v14_lead < 12) {
                float v2273_data = r9[5];
                float v2274_data = r10[0];
                float v2277_data = ir11[0];
                ir11[0] = (v2277_data + (v2273_data * (sycl::group_broadcast(item.get_sub_group(), v2274_data, 5))));
                float v2280_data = r10[1];
                float v2283_data = ir11[1];
                ir11[1] = (v2283_data + (v2273_data * (sycl::group_broadcast(item.get_sub_group(), v2280_data, 5))));
                float v2286_data = r10[2];
                float v2289_data = ir11[2];
                ir11[2] = (v2289_data + (v2273_data * (sycl::group_broadcast(item.get_sub_group(), v2286_data, 5))));
                float v2292_data = r10[3];
                float v2295_data = ir11[3];
                ir11[3] = (v2295_data + (v2273_data * (sycl::group_broadcast(item.get_sub_group(), v2292_data, 5))));
                float v2298_data = r10[4];
                float v2301_data = ir11[4];
                ir11[4] = (v2301_data + (v2273_data * (sycl::group_broadcast(item.get_sub_group(), v2298_data, 5))));
                float v2304_data = r10[5];
                float v2307_data = ir11[5];
                ir11[5] = (v2307_data + (v2273_data * (sycl::group_broadcast(item.get_sub_group(), v2304_data, 5))));
                float v2310_data = r10[6];
                float v2313_data = ir11[6];
                ir11[6] = (v2313_data + (v2273_data * (sycl::group_broadcast(item.get_sub_group(), v2310_data, 5))));
                float v2316_data = r10[7];
                float v2319_data = ir11[7];
                ir11[7] = (v2319_data + (v2273_data * (sycl::group_broadcast(item.get_sub_group(), v2316_data, 5))));
              }
              if (v14_lead < 12) {
                float v2325_data = r9[6];
                float v2326_data = r10[0];
                float v2329_data = ir11[0];
                ir11[0] = (v2329_data + (v2325_data * (sycl::group_broadcast(item.get_sub_group(), v2326_data, 6))));
                float v2332_data = r10[1];
                float v2335_data = ir11[1];
                ir11[1] = (v2335_data + (v2325_data * (sycl::group_broadcast(item.get_sub_group(), v2332_data, 6))));
                float v2338_data = r10[2];
                float v2341_data = ir11[2];
                ir11[2] = (v2341_data + (v2325_data * (sycl::group_broadcast(item.get_sub_group(), v2338_data, 6))));
                float v2344_data = r10[3];
                float v2347_data = ir11[3];
                ir11[3] = (v2347_data + (v2325_data * (sycl::group_broadcast(item.get_sub_group(), v2344_data, 6))));
                float v2350_data = r10[4];
                float v2353_data = ir11[4];
                ir11[4] = (v2353_data + (v2325_data * (sycl::group_broadcast(item.get_sub_group(), v2350_data, 6))));
                float v2356_data = r10[5];
                float v2359_data = ir11[5];
                ir11[5] = (v2359_data + (v2325_data * (sycl::group_broadcast(item.get_sub_group(), v2356_data, 6))));
                float v2362_data = r10[6];
                float v2365_data = ir11[6];
                ir11[6] = (v2365_data + (v2325_data * (sycl::group_broadcast(item.get_sub_group(), v2362_data, 6))));
                float v2368_data = r10[7];
                float v2371_data = ir11[7];
                ir11[7] = (v2371_data + (v2325_data * (sycl::group_broadcast(item.get_sub_group(), v2368_data, 6))));
              }
              if (v14_lead < 12) {
                float v2377_data = r9[7];
                float v2378_data = r10[0];
                float v2381_data = ir11[0];
                ir11[0] = (v2381_data + (v2377_data * (sycl::group_broadcast(item.get_sub_group(), v2378_data, 7))));
                float v2384_data = r10[1];
                float v2387_data = ir11[1];
                ir11[1] = (v2387_data + (v2377_data * (sycl::group_broadcast(item.get_sub_group(), v2384_data, 7))));
                float v2390_data = r10[2];
                float v2393_data = ir11[2];
                ir11[2] = (v2393_data + (v2377_data * (sycl::group_broadcast(item.get_sub_group(), v2390_data, 7))));
                float v2396_data = r10[3];
                float v2399_data = ir11[3];
                ir11[3] = (v2399_data + (v2377_data * (sycl::group_broadcast(item.get_sub_group(), v2396_data, 7))));
                float v2402_data = r10[4];
                float v2405_data = ir11[4];
                ir11[4] = (v2405_data + (v2377_data * (sycl::group_broadcast(item.get_sub_group(), v2402_data, 7))));
                float v2408_data = r10[5];
                float v2411_data = ir11[5];
                ir11[5] = (v2411_data + (v2377_data * (sycl::group_broadcast(item.get_sub_group(), v2408_data, 7))));
                float v2414_data = r10[6];
                float v2417_data = ir11[6];
                ir11[6] = (v2417_data + (v2377_data * (sycl::group_broadcast(item.get_sub_group(), v2414_data, 7))));
                float v2420_data = r10[7];
                float v2423_data = ir11[7];
                ir11[7] = (v2423_data + (v2377_data * (sycl::group_broadcast(item.get_sub_group(), v2420_data, 7))));
              }
              if (v14_lead < 12) {
                float v2429_data = r9[8];
                float v2430_data = r10[0];
                float v2433_data = ir11[0];
                ir11[0] = (v2433_data + (v2429_data * (sycl::group_broadcast(item.get_sub_group(), v2430_data, 8))));
                float v2436_data = r10[1];
                float v2439_data = ir11[1];
                ir11[1] = (v2439_data + (v2429_data * (sycl::group_broadcast(item.get_sub_group(), v2436_data, 8))));
                float v2442_data = r10[2];
                float v2445_data = ir11[2];
                ir11[2] = (v2445_data + (v2429_data * (sycl::group_broadcast(item.get_sub_group(), v2442_data, 8))));
                float v2448_data = r10[3];
                float v2451_data = ir11[3];
                ir11[3] = (v2451_data + (v2429_data * (sycl::group_broadcast(item.get_sub_group(), v2448_data, 8))));
                float v2454_data = r10[4];
                float v2457_data = ir11[4];
                ir11[4] = (v2457_data + (v2429_data * (sycl::group_broadcast(item.get_sub_group(), v2454_data, 8))));
                float v2460_data = r10[5];
                float v2463_data = ir11[5];
                ir11[5] = (v2463_data + (v2429_data * (sycl::group_broadcast(item.get_sub_group(), v2460_data, 8))));
                float v2466_data = r10[6];
                float v2469_data = ir11[6];
                ir11[6] = (v2469_data + (v2429_data * (sycl::group_broadcast(item.get_sub_group(), v2466_data, 8))));
                float v2472_data = r10[7];
                float v2475_data = ir11[7];
                ir11[7] = (v2475_data + (v2429_data * (sycl::group_broadcast(item.get_sub_group(), v2472_data, 8))));
              }
              if (v14_lead < 12) {
                float v2481_data = r9[9];
                float v2482_data = r10[0];
                float v2485_data = ir11[0];
                ir11[0] = (v2485_data + (v2481_data * (sycl::group_broadcast(item.get_sub_group(), v2482_data, 9))));
                float v2488_data = r10[1];
                float v2491_data = ir11[1];
                ir11[1] = (v2491_data + (v2481_data * (sycl::group_broadcast(item.get_sub_group(), v2488_data, 9))));
                float v2494_data = r10[2];
                float v2497_data = ir11[2];
                ir11[2] = (v2497_data + (v2481_data * (sycl::group_broadcast(item.get_sub_group(), v2494_data, 9))));
                float v2500_data = r10[3];
                float v2503_data = ir11[3];
                ir11[3] = (v2503_data + (v2481_data * (sycl::group_broadcast(item.get_sub_group(), v2500_data, 9))));
                float v2506_data = r10[4];
                float v2509_data = ir11[4];
                ir11[4] = (v2509_data + (v2481_data * (sycl::group_broadcast(item.get_sub_group(), v2506_data, 9))));
                float v2512_data = r10[5];
                float v2515_data = ir11[5];
                ir11[5] = (v2515_data + (v2481_data * (sycl::group_broadcast(item.get_sub_group(), v2512_data, 9))));
                float v2518_data = r10[6];
                float v2521_data = ir11[6];
                ir11[6] = (v2521_data + (v2481_data * (sycl::group_broadcast(item.get_sub_group(), v2518_data, 9))));
                float v2524_data = r10[7];
                float v2527_data = ir11[7];
                ir11[7] = (v2527_data + (v2481_data * (sycl::group_broadcast(item.get_sub_group(), v2524_data, 9))));
              }
              if (v14_lead < 12) {
                float v2533_data = r9[10];
                float v2534_data = r10[0];
                float v2537_data = ir11[0];
                ir11[0] = (v2537_data + (v2533_data * (sycl::group_broadcast(item.get_sub_group(), v2534_data, 10))));
                float v2540_data = r10[1];
                float v2543_data = ir11[1];
                ir11[1] = (v2543_data + (v2533_data * (sycl::group_broadcast(item.get_sub_group(), v2540_data, 10))));
                float v2546_data = r10[2];
                float v2549_data = ir11[2];
                ir11[2] = (v2549_data + (v2533_data * (sycl::group_broadcast(item.get_sub_group(), v2546_data, 10))));
                float v2552_data = r10[3];
                float v2555_data = ir11[3];
                ir11[3] = (v2555_data + (v2533_data * (sycl::group_broadcast(item.get_sub_group(), v2552_data, 10))));
                float v2558_data = r10[4];
                float v2561_data = ir11[4];
                ir11[4] = (v2561_data + (v2533_data * (sycl::group_broadcast(item.get_sub_group(), v2558_data, 10))));
                float v2564_data = r10[5];
                float v2567_data = ir11[5];
                ir11[5] = (v2567_data + (v2533_data * (sycl::group_broadcast(item.get_sub_group(), v2564_data, 10))));
                float v2570_data = r10[6];
                float v2573_data = ir11[6];
                ir11[6] = (v2573_data + (v2533_data * (sycl::group_broadcast(item.get_sub_group(), v2570_data, 10))));
                float v2576_data = r10[7];
                float v2579_data = ir11[7];
                ir11[7] = (v2579_data + (v2533_data * (sycl::group_broadcast(item.get_sub_group(), v2576_data, 10))));
              }
              if (v14_lead < 12) {
                float v2585_data = r9[11];
                float v2586_data = r10[0];
                float v2589_data = ir11[0];
                ir11[0] = (v2589_data + (v2585_data * (sycl::group_broadcast(item.get_sub_group(), v2586_data, 11))));
                float v2592_data = r10[1];
                float v2595_data = ir11[1];
                ir11[1] = (v2595_data + (v2585_data * (sycl::group_broadcast(item.get_sub_group(), v2592_data, 11))));
                float v2598_data = r10[2];
                float v2601_data = ir11[2];
                ir11[2] = (v2601_data + (v2585_data * (sycl::group_broadcast(item.get_sub_group(), v2598_data, 11))));
                float v2604_data = r10[3];
                float v2607_data = ir11[3];
                ir11[3] = (v2607_data + (v2585_data * (sycl::group_broadcast(item.get_sub_group(), v2604_data, 11))));
                float v2610_data = r10[4];
                float v2613_data = ir11[4];
                ir11[4] = (v2613_data + (v2585_data * (sycl::group_broadcast(item.get_sub_group(), v2610_data, 11))));
                float v2616_data = r10[5];
                float v2619_data = ir11[5];
                ir11[5] = (v2619_data + (v2585_data * (sycl::group_broadcast(item.get_sub_group(), v2616_data, 11))));
                float v2622_data = r10[6];
                float v2625_data = ir11[6];
                ir11[6] = (v2625_data + (v2585_data * (sycl::group_broadcast(item.get_sub_group(), v2622_data, 11))));
                float v2628_data = r10[7];
                float v2631_data = ir11[7];
                ir11[7] = (v2631_data + (v2585_data * (sycl::group_broadcast(item.get_sub_group(), v2628_data, 11))));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v2637_n1 = 0; v2637_n1 < 8; ++v2637_n1) {
                  float v2639_data = ir11[v2637_n1];
                  float v2641_data = r8[v2637_n1];
                  r11[v2637_n1] = (v2641_data + v2639_data);
                }
              }
              // glb_m0 = store{r>g}(r11);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v2648_i1 = 0; v2648_i1 < 8; ++v2648_i1) {
                  float v2650_data = r11[v2648_i1];
                  glb_m0[(v14_lead + (v2648_i1 * 12))] = v2650_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


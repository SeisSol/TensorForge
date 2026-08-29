// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08a27dccde(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08a27dccde(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 9×9(9×9) {0..9}×{0..9} strided
        // m1 9×9(9×9) {0..9}×{0..9} strided
        // m2 9×9(9×9) {0..9}×{0..9} strided
        // m3 ()  scalar
        // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 9) {
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 9; ++v10_i1) {
                  float v18_data = glb_m1[(v8_lead + (v10_i1 * 9))];
                  r0[v10_i1] = v18_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              float v21_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v21_lin;
              float v22_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v22_lin;
              float v23_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v23_lin;
              float v24_lin = glb_m2[48 + item.get_local_id(0) * 1];
              r1[3] = v24_lin;
              float v25_lin = glb_m2[64 + item.get_local_id(0) * 1];
              r1[4] = v25_lin;
              float v26_lin = glb_m2[80 + item.get_local_id(0) * 1];
              r1[5] = v26_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir2[9]{};
              if (v8_lead < 9) {
                float v33_data = r0[0];
                float v34_data = r1[0];
                float v37_data = ir2[0];
                ir2[0] = (v37_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v34_data, 0))));
                float v40_data = r1[1];
                float v43_data = ir2[1];
                ir2[1] = (v43_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v40_data, 0))));
                float v46_data = r1[2];
                float v49_data = ir2[2];
                ir2[2] = (v49_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 0))));
                float v52_data = r1[3];
                float v55_data = ir2[3];
                ir2[3] = (v55_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v52_data, 0))));
                float v58_data = r1[4];
                float v61_data = ir2[4];
                ir2[4] = (v61_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 0))));
                float v64_data = r1[5];
                float v67_data = ir2[5];
                ir2[5] = (v67_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 0))));
                float v70_data = r1[6];
                float v73_data = ir2[6];
                ir2[6] = (v73_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 0))));
                float v76_data = r1[7];
                float v79_data = ir2[7];
                ir2[7] = (v79_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 0))));
                float v82_data = r1[8];
                float v85_data = ir2[8];
                ir2[8] = (v85_data + (v33_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 0))));
              }
              if (v8_lead < 9) {
                float v91_data = r0[1];
                float v92_data = r1[0];
                float v95_data = ir2[0];
                ir2[0] = (v95_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 1))));
                float v98_data = r1[1];
                float v101_data = ir2[1];
                ir2[1] = (v101_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 1))));
                float v104_data = r1[2];
                float v107_data = ir2[2];
                ir2[2] = (v107_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 1))));
                float v110_data = r1[3];
                float v113_data = ir2[3];
                ir2[3] = (v113_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 1))));
                float v116_data = r1[4];
                float v119_data = ir2[4];
                ir2[4] = (v119_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 1))));
                float v122_data = r1[5];
                float v125_data = ir2[5];
                ir2[5] = (v125_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 1))));
                float v128_data = r1[6];
                float v131_data = ir2[6];
                ir2[6] = (v131_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 1))));
                float v134_data = r1[7];
                float v137_data = ir2[7];
                ir2[7] = (v137_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 1))));
                float v140_data = r1[8];
                float v143_data = ir2[8];
                ir2[8] = (v143_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 1))));
              }
              if (v8_lead < 9) {
                float v149_data = r0[2];
                float v150_data = r1[0];
                float v153_data = ir2[0];
                ir2[0] = (v153_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 2))));
                float v156_data = r1[1];
                float v159_data = ir2[1];
                ir2[1] = (v159_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 2))));
                float v162_data = r1[2];
                float v165_data = ir2[2];
                ir2[2] = (v165_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 2))));
                float v168_data = r1[3];
                float v171_data = ir2[3];
                ir2[3] = (v171_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 2))));
                float v174_data = r1[4];
                float v177_data = ir2[4];
                ir2[4] = (v177_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 2))));
                float v180_data = r1[5];
                float v183_data = ir2[5];
                ir2[5] = (v183_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 2))));
                float v186_data = r1[6];
                float v189_data = ir2[6];
                ir2[6] = (v189_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v186_data, 2))));
                float v192_data = r1[7];
                float v195_data = ir2[7];
                ir2[7] = (v195_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v192_data, 2))));
                float v198_data = r1[8];
                float v201_data = ir2[8];
                ir2[8] = (v201_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v198_data, 2))));
              }
              if (v8_lead < 9) {
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
                float v256_data = r1[8];
                float v259_data = ir2[8];
                ir2[8] = (v259_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v256_data, 3))));
              }
              if (v8_lead < 9) {
                float v265_data = r0[4];
                float v266_data = r1[0];
                float v269_data = ir2[0];
                ir2[0] = (v269_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v266_data, 4))));
                float v272_data = r1[1];
                float v275_data = ir2[1];
                ir2[1] = (v275_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v272_data, 4))));
                float v278_data = r1[2];
                float v281_data = ir2[2];
                ir2[2] = (v281_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v278_data, 4))));
                float v284_data = r1[3];
                float v287_data = ir2[3];
                ir2[3] = (v287_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v284_data, 4))));
                float v290_data = r1[4];
                float v293_data = ir2[4];
                ir2[4] = (v293_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 4))));
                float v296_data = r1[5];
                float v299_data = ir2[5];
                ir2[5] = (v299_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v296_data, 4))));
                float v302_data = r1[6];
                float v305_data = ir2[6];
                ir2[6] = (v305_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v302_data, 4))));
                float v308_data = r1[7];
                float v311_data = ir2[7];
                ir2[7] = (v311_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v308_data, 4))));
                float v314_data = r1[8];
                float v317_data = ir2[8];
                ir2[8] = (v317_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v314_data, 4))));
              }
              if (v8_lead < 9) {
                float v323_data = r0[5];
                float v324_data = r1[0];
                float v327_data = ir2[0];
                ir2[0] = (v327_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 5))));
                float v330_data = r1[1];
                float v333_data = ir2[1];
                ir2[1] = (v333_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v330_data, 5))));
                float v336_data = r1[2];
                float v339_data = ir2[2];
                ir2[2] = (v339_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v336_data, 5))));
                float v342_data = r1[3];
                float v345_data = ir2[3];
                ir2[3] = (v345_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 5))));
                float v348_data = r1[4];
                float v351_data = ir2[4];
                ir2[4] = (v351_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 5))));
                float v354_data = r1[5];
                float v357_data = ir2[5];
                ir2[5] = (v357_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 5))));
                float v360_data = r1[6];
                float v363_data = ir2[6];
                ir2[6] = (v363_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v360_data, 5))));
                float v366_data = r1[7];
                float v369_data = ir2[7];
                ir2[7] = (v369_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v366_data, 5))));
                float v372_data = r1[8];
                float v375_data = ir2[8];
                ir2[8] = (v375_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v372_data, 5))));
              }
              if (v8_lead < 9) {
                float v381_data = r0[6];
                float v382_data = r1[0];
                float v385_data = ir2[0];
                ir2[0] = (v385_data + (v381_data * (sycl::group_broadcast(item.get_sub_group(), v382_data, 6))));
                float v388_data = r1[1];
                float v391_data = ir2[1];
                ir2[1] = (v391_data + (v381_data * (sycl::group_broadcast(item.get_sub_group(), v388_data, 6))));
                float v394_data = r1[2];
                float v397_data = ir2[2];
                ir2[2] = (v397_data + (v381_data * (sycl::group_broadcast(item.get_sub_group(), v394_data, 6))));
                float v400_data = r1[3];
                float v403_data = ir2[3];
                ir2[3] = (v403_data + (v381_data * (sycl::group_broadcast(item.get_sub_group(), v400_data, 6))));
                float v406_data = r1[4];
                float v409_data = ir2[4];
                ir2[4] = (v409_data + (v381_data * (sycl::group_broadcast(item.get_sub_group(), v406_data, 6))));
                float v412_data = r1[5];
                float v415_data = ir2[5];
                ir2[5] = (v415_data + (v381_data * (sycl::group_broadcast(item.get_sub_group(), v412_data, 6))));
                float v418_data = r1[6];
                float v421_data = ir2[6];
                ir2[6] = (v421_data + (v381_data * (sycl::group_broadcast(item.get_sub_group(), v418_data, 6))));
                float v424_data = r1[7];
                float v427_data = ir2[7];
                ir2[7] = (v427_data + (v381_data * (sycl::group_broadcast(item.get_sub_group(), v424_data, 6))));
                float v430_data = r1[8];
                float v433_data = ir2[8];
                ir2[8] = (v433_data + (v381_data * (sycl::group_broadcast(item.get_sub_group(), v430_data, 6))));
              }
              if (v8_lead < 9) {
                float v439_data = r0[7];
                float v440_data = r1[0];
                float v443_data = ir2[0];
                ir2[0] = (v443_data + (v439_data * (sycl::group_broadcast(item.get_sub_group(), v440_data, 7))));
                float v446_data = r1[1];
                float v449_data = ir2[1];
                ir2[1] = (v449_data + (v439_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 7))));
                float v452_data = r1[2];
                float v455_data = ir2[2];
                ir2[2] = (v455_data + (v439_data * (sycl::group_broadcast(item.get_sub_group(), v452_data, 7))));
                float v458_data = r1[3];
                float v461_data = ir2[3];
                ir2[3] = (v461_data + (v439_data * (sycl::group_broadcast(item.get_sub_group(), v458_data, 7))));
                float v464_data = r1[4];
                float v467_data = ir2[4];
                ir2[4] = (v467_data + (v439_data * (sycl::group_broadcast(item.get_sub_group(), v464_data, 7))));
                float v470_data = r1[5];
                float v473_data = ir2[5];
                ir2[5] = (v473_data + (v439_data * (sycl::group_broadcast(item.get_sub_group(), v470_data, 7))));
                float v476_data = r1[6];
                float v479_data = ir2[6];
                ir2[6] = (v479_data + (v439_data * (sycl::group_broadcast(item.get_sub_group(), v476_data, 7))));
                float v482_data = r1[7];
                float v485_data = ir2[7];
                ir2[7] = (v485_data + (v439_data * (sycl::group_broadcast(item.get_sub_group(), v482_data, 7))));
                float v488_data = r1[8];
                float v491_data = ir2[8];
                ir2[8] = (v491_data + (v439_data * (sycl::group_broadcast(item.get_sub_group(), v488_data, 7))));
              }
              if (v8_lead < 9) {
                float v497_data = r0[8];
                float v498_data = r1[0];
                float v501_data = ir2[0];
                ir2[0] = (v501_data + (v497_data * (sycl::group_broadcast(item.get_sub_group(), v498_data, 8))));
                float v504_data = r1[1];
                float v507_data = ir2[1];
                ir2[1] = (v507_data + (v497_data * (sycl::group_broadcast(item.get_sub_group(), v504_data, 8))));
                float v510_data = r1[2];
                float v513_data = ir2[2];
                ir2[2] = (v513_data + (v497_data * (sycl::group_broadcast(item.get_sub_group(), v510_data, 8))));
                float v516_data = r1[3];
                float v519_data = ir2[3];
                ir2[3] = (v519_data + (v497_data * (sycl::group_broadcast(item.get_sub_group(), v516_data, 8))));
                float v522_data = r1[4];
                float v525_data = ir2[4];
                ir2[4] = (v525_data + (v497_data * (sycl::group_broadcast(item.get_sub_group(), v522_data, 8))));
                float v528_data = r1[5];
                float v531_data = ir2[5];
                ir2[5] = (v531_data + (v497_data * (sycl::group_broadcast(item.get_sub_group(), v528_data, 8))));
                float v534_data = r1[6];
                float v537_data = ir2[6];
                ir2[6] = (v537_data + (v497_data * (sycl::group_broadcast(item.get_sub_group(), v534_data, 8))));
                float v540_data = r1[7];
                float v543_data = ir2[7];
                ir2[7] = (v543_data + (v497_data * (sycl::group_broadcast(item.get_sub_group(), v540_data, 8))));
                float v546_data = r1[8];
                float v549_data = ir2[8];
                ir2[8] = (v549_data + (v497_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 8))));
              }
              if (v8_lead < 9) {
                #pragma unroll
                for (int32_t v556_n1 = 0; v556_n1 < 9; ++v556_n1) {
                  float v558_data = ir2[v556_n1];
                  r2[v556_n1] = (v558_data * 13.0f);
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v8_lead < 9) {
                #pragma unroll
                for (int32_t v565_i1 = 0; v565_i1 < 9; ++v565_i1) {
                  float v567_data = r2[v565_i1];
                  glb_m0[(v8_lead + (v565_i1 * 9))] = v567_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


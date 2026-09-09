// === base name ===
kernel_8162b17515

// === header ===
void launcher_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8162b17515(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8162b17515(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 24×9(24×9) {0..24}×{0..9} strided
        // m1 24×24(24×24) {0..24}×{0..24} strided
        // m2 24×9(24×9) {0..24}×{0..9} strided
        // m0 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[0, 1] = m1 24×24(24×24) {0..24}×{0..24} strided({0..24}×{0..24})[0, -1]×m2 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 216 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 576 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 216 + 0 + m2_extraOffset];
              float r0[24]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v6_lead = item.get_local_id(0) % 32;
              bool v7_g = v6_lead < 24;
              if (v7_g) {
                #pragma unroll
                for (int32_t v8_i1 = 0; v8_i1 < 24; ++v8_i1) {
                  float v16_data = glb_m1[(v6_lead + (v8_i1 * 24))];
                  r0[v8_i1] = v16_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              if (v7_g) {
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 9; ++v23_i1) {
                  float v31_data = glb_m2[(v6_lead + (v23_i1 * 24))];
                  r1[v23_i1] = v31_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 24), (0, 9)] [(0, 24)]
              float ir2[9]{};
              if (v7_g) {
                float v39_data = r0[0];
                float v40_data = r1[0];
                float v43_data = ir2[0];
                ir2[0] = (v43_data + (v39_data * (sycl::group_broadcast(item.get_sub_group(), v40_data, 0))));
                float v46_data = r1[1];
                float v49_data = ir2[1];
                ir2[1] = (v49_data + (v39_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 0))));
                float v52_data = r1[2];
                float v55_data = ir2[2];
                ir2[2] = (v55_data + (v39_data * (sycl::group_broadcast(item.get_sub_group(), v52_data, 0))));
                float v58_data = r1[3];
                float v61_data = ir2[3];
                ir2[3] = (v61_data + (v39_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 0))));
                float v64_data = r1[4];
                float v67_data = ir2[4];
                ir2[4] = (v67_data + (v39_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 0))));
                float v70_data = r1[5];
                float v73_data = ir2[5];
                ir2[5] = (v73_data + (v39_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 0))));
                float v76_data = r1[6];
                float v79_data = ir2[6];
                ir2[6] = (v79_data + (v39_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 0))));
                float v82_data = r1[7];
                float v85_data = ir2[7];
                ir2[7] = (v85_data + (v39_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 0))));
                float v88_data = r1[8];
                float v91_data = ir2[8];
                ir2[8] = (v91_data + (v39_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 0))));
              }
              if (v7_g) {
                float v97_data = r0[1];
                float v98_data = r1[0];
                float v101_data = ir2[0];
                ir2[0] = (v101_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 1))));
                float v104_data = r1[1];
                float v107_data = ir2[1];
                ir2[1] = (v107_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 1))));
                float v110_data = r1[2];
                float v113_data = ir2[2];
                ir2[2] = (v113_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 1))));
                float v116_data = r1[3];
                float v119_data = ir2[3];
                ir2[3] = (v119_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 1))));
                float v122_data = r1[4];
                float v125_data = ir2[4];
                ir2[4] = (v125_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 1))));
                float v128_data = r1[5];
                float v131_data = ir2[5];
                ir2[5] = (v131_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 1))));
                float v134_data = r1[6];
                float v137_data = ir2[6];
                ir2[6] = (v137_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 1))));
                float v140_data = r1[7];
                float v143_data = ir2[7];
                ir2[7] = (v143_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 1))));
                float v146_data = r1[8];
                float v149_data = ir2[8];
                ir2[8] = (v149_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 1))));
              }
              if (v7_g) {
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
                float v204_data = r1[8];
                float v207_data = ir2[8];
                ir2[8] = (v207_data + (v155_data * (sycl::group_broadcast(item.get_sub_group(), v204_data, 2))));
              }
              if (v7_g) {
                float v213_data = r0[3];
                float v214_data = r1[0];
                float v217_data = ir2[0];
                ir2[0] = (v217_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v214_data, 3))));
                float v220_data = r1[1];
                float v223_data = ir2[1];
                ir2[1] = (v223_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v220_data, 3))));
                float v226_data = r1[2];
                float v229_data = ir2[2];
                ir2[2] = (v229_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v226_data, 3))));
                float v232_data = r1[3];
                float v235_data = ir2[3];
                ir2[3] = (v235_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v232_data, 3))));
                float v238_data = r1[4];
                float v241_data = ir2[4];
                ir2[4] = (v241_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v238_data, 3))));
                float v244_data = r1[5];
                float v247_data = ir2[5];
                ir2[5] = (v247_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v244_data, 3))));
                float v250_data = r1[6];
                float v253_data = ir2[6];
                ir2[6] = (v253_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 3))));
                float v256_data = r1[7];
                float v259_data = ir2[7];
                ir2[7] = (v259_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v256_data, 3))));
                float v262_data = r1[8];
                float v265_data = ir2[8];
                ir2[8] = (v265_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v262_data, 3))));
              }
              if (v7_g) {
                float v271_data = r0[4];
                float v272_data = r1[0];
                float v275_data = ir2[0];
                ir2[0] = (v275_data + (v271_data * (sycl::group_broadcast(item.get_sub_group(), v272_data, 4))));
                float v278_data = r1[1];
                float v281_data = ir2[1];
                ir2[1] = (v281_data + (v271_data * (sycl::group_broadcast(item.get_sub_group(), v278_data, 4))));
                float v284_data = r1[2];
                float v287_data = ir2[2];
                ir2[2] = (v287_data + (v271_data * (sycl::group_broadcast(item.get_sub_group(), v284_data, 4))));
                float v290_data = r1[3];
                float v293_data = ir2[3];
                ir2[3] = (v293_data + (v271_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 4))));
                float v296_data = r1[4];
                float v299_data = ir2[4];
                ir2[4] = (v299_data + (v271_data * (sycl::group_broadcast(item.get_sub_group(), v296_data, 4))));
                float v302_data = r1[5];
                float v305_data = ir2[5];
                ir2[5] = (v305_data + (v271_data * (sycl::group_broadcast(item.get_sub_group(), v302_data, 4))));
                float v308_data = r1[6];
                float v311_data = ir2[6];
                ir2[6] = (v311_data + (v271_data * (sycl::group_broadcast(item.get_sub_group(), v308_data, 4))));
                float v314_data = r1[7];
                float v317_data = ir2[7];
                ir2[7] = (v317_data + (v271_data * (sycl::group_broadcast(item.get_sub_group(), v314_data, 4))));
                float v320_data = r1[8];
                float v323_data = ir2[8];
                ir2[8] = (v323_data + (v271_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 4))));
              }
              if (v7_g) {
                float v329_data = r0[5];
                float v330_data = r1[0];
                float v333_data = ir2[0];
                ir2[0] = (v333_data + (v329_data * (sycl::group_broadcast(item.get_sub_group(), v330_data, 5))));
                float v336_data = r1[1];
                float v339_data = ir2[1];
                ir2[1] = (v339_data + (v329_data * (sycl::group_broadcast(item.get_sub_group(), v336_data, 5))));
                float v342_data = r1[2];
                float v345_data = ir2[2];
                ir2[2] = (v345_data + (v329_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 5))));
                float v348_data = r1[3];
                float v351_data = ir2[3];
                ir2[3] = (v351_data + (v329_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 5))));
                float v354_data = r1[4];
                float v357_data = ir2[4];
                ir2[4] = (v357_data + (v329_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 5))));
                float v360_data = r1[5];
                float v363_data = ir2[5];
                ir2[5] = (v363_data + (v329_data * (sycl::group_broadcast(item.get_sub_group(), v360_data, 5))));
                float v366_data = r1[6];
                float v369_data = ir2[6];
                ir2[6] = (v369_data + (v329_data * (sycl::group_broadcast(item.get_sub_group(), v366_data, 5))));
                float v372_data = r1[7];
                float v375_data = ir2[7];
                ir2[7] = (v375_data + (v329_data * (sycl::group_broadcast(item.get_sub_group(), v372_data, 5))));
                float v378_data = r1[8];
                float v381_data = ir2[8];
                ir2[8] = (v381_data + (v329_data * (sycl::group_broadcast(item.get_sub_group(), v378_data, 5))));
              }
              if (v7_g) {
                float v387_data = r0[6];
                float v388_data = r1[0];
                float v391_data = ir2[0];
                ir2[0] = (v391_data + (v387_data * (sycl::group_broadcast(item.get_sub_group(), v388_data, 6))));
                float v394_data = r1[1];
                float v397_data = ir2[1];
                ir2[1] = (v397_data + (v387_data * (sycl::group_broadcast(item.get_sub_group(), v394_data, 6))));
                float v400_data = r1[2];
                float v403_data = ir2[2];
                ir2[2] = (v403_data + (v387_data * (sycl::group_broadcast(item.get_sub_group(), v400_data, 6))));
                float v406_data = r1[3];
                float v409_data = ir2[3];
                ir2[3] = (v409_data + (v387_data * (sycl::group_broadcast(item.get_sub_group(), v406_data, 6))));
                float v412_data = r1[4];
                float v415_data = ir2[4];
                ir2[4] = (v415_data + (v387_data * (sycl::group_broadcast(item.get_sub_group(), v412_data, 6))));
                float v418_data = r1[5];
                float v421_data = ir2[5];
                ir2[5] = (v421_data + (v387_data * (sycl::group_broadcast(item.get_sub_group(), v418_data, 6))));
                float v424_data = r1[6];
                float v427_data = ir2[6];
                ir2[6] = (v427_data + (v387_data * (sycl::group_broadcast(item.get_sub_group(), v424_data, 6))));
                float v430_data = r1[7];
                float v433_data = ir2[7];
                ir2[7] = (v433_data + (v387_data * (sycl::group_broadcast(item.get_sub_group(), v430_data, 6))));
                float v436_data = r1[8];
                float v439_data = ir2[8];
                ir2[8] = (v439_data + (v387_data * (sycl::group_broadcast(item.get_sub_group(), v436_data, 6))));
              }
              if (v7_g) {
                float v445_data = r0[7];
                float v446_data = r1[0];
                float v449_data = ir2[0];
                ir2[0] = (v449_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 7))));
                float v452_data = r1[1];
                float v455_data = ir2[1];
                ir2[1] = (v455_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v452_data, 7))));
                float v458_data = r1[2];
                float v461_data = ir2[2];
                ir2[2] = (v461_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v458_data, 7))));
                float v464_data = r1[3];
                float v467_data = ir2[3];
                ir2[3] = (v467_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v464_data, 7))));
                float v470_data = r1[4];
                float v473_data = ir2[4];
                ir2[4] = (v473_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v470_data, 7))));
                float v476_data = r1[5];
                float v479_data = ir2[5];
                ir2[5] = (v479_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v476_data, 7))));
                float v482_data = r1[6];
                float v485_data = ir2[6];
                ir2[6] = (v485_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v482_data, 7))));
                float v488_data = r1[7];
                float v491_data = ir2[7];
                ir2[7] = (v491_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v488_data, 7))));
                float v494_data = r1[8];
                float v497_data = ir2[8];
                ir2[8] = (v497_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v494_data, 7))));
              }
              if (v7_g) {
                float v503_data = r0[8];
                float v504_data = r1[0];
                float v507_data = ir2[0];
                ir2[0] = (v507_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v504_data, 8))));
                float v510_data = r1[1];
                float v513_data = ir2[1];
                ir2[1] = (v513_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v510_data, 8))));
                float v516_data = r1[2];
                float v519_data = ir2[2];
                ir2[2] = (v519_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v516_data, 8))));
                float v522_data = r1[3];
                float v525_data = ir2[3];
                ir2[3] = (v525_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v522_data, 8))));
                float v528_data = r1[4];
                float v531_data = ir2[4];
                ir2[4] = (v531_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v528_data, 8))));
                float v534_data = r1[5];
                float v537_data = ir2[5];
                ir2[5] = (v537_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v534_data, 8))));
                float v540_data = r1[6];
                float v543_data = ir2[6];
                ir2[6] = (v543_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v540_data, 8))));
                float v546_data = r1[7];
                float v549_data = ir2[7];
                ir2[7] = (v549_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 8))));
                float v552_data = r1[8];
                float v555_data = ir2[8];
                ir2[8] = (v555_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v552_data, 8))));
              }
              if (v7_g) {
                float v561_data = r0[9];
                float v562_data = r1[0];
                float v565_data = ir2[0];
                ir2[0] = (v565_data + (v561_data * (sycl::group_broadcast(item.get_sub_group(), v562_data, 9))));
                float v568_data = r1[1];
                float v571_data = ir2[1];
                ir2[1] = (v571_data + (v561_data * (sycl::group_broadcast(item.get_sub_group(), v568_data, 9))));
                float v574_data = r1[2];
                float v577_data = ir2[2];
                ir2[2] = (v577_data + (v561_data * (sycl::group_broadcast(item.get_sub_group(), v574_data, 9))));
                float v580_data = r1[3];
                float v583_data = ir2[3];
                ir2[3] = (v583_data + (v561_data * (sycl::group_broadcast(item.get_sub_group(), v580_data, 9))));
                float v586_data = r1[4];
                float v589_data = ir2[4];
                ir2[4] = (v589_data + (v561_data * (sycl::group_broadcast(item.get_sub_group(), v586_data, 9))));
                float v592_data = r1[5];
                float v595_data = ir2[5];
                ir2[5] = (v595_data + (v561_data * (sycl::group_broadcast(item.get_sub_group(), v592_data, 9))));
                float v598_data = r1[6];
                float v601_data = ir2[6];
                ir2[6] = (v601_data + (v561_data * (sycl::group_broadcast(item.get_sub_group(), v598_data, 9))));
                float v604_data = r1[7];
                float v607_data = ir2[7];
                ir2[7] = (v607_data + (v561_data * (sycl::group_broadcast(item.get_sub_group(), v604_data, 9))));
                float v610_data = r1[8];
                float v613_data = ir2[8];
                ir2[8] = (v613_data + (v561_data * (sycl::group_broadcast(item.get_sub_group(), v610_data, 9))));
              }
              if (v7_g) {
                float v619_data = r0[10];
                float v620_data = r1[0];
                float v623_data = ir2[0];
                ir2[0] = (v623_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v620_data, 10))));
                float v626_data = r1[1];
                float v629_data = ir2[1];
                ir2[1] = (v629_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v626_data, 10))));
                float v632_data = r1[2];
                float v635_data = ir2[2];
                ir2[2] = (v635_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v632_data, 10))));
                float v638_data = r1[3];
                float v641_data = ir2[3];
                ir2[3] = (v641_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v638_data, 10))));
                float v644_data = r1[4];
                float v647_data = ir2[4];
                ir2[4] = (v647_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v644_data, 10))));
                float v650_data = r1[5];
                float v653_data = ir2[5];
                ir2[5] = (v653_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 10))));
                float v656_data = r1[6];
                float v659_data = ir2[6];
                ir2[6] = (v659_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v656_data, 10))));
                float v662_data = r1[7];
                float v665_data = ir2[7];
                ir2[7] = (v665_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v662_data, 10))));
                float v668_data = r1[8];
                float v671_data = ir2[8];
                ir2[8] = (v671_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v668_data, 10))));
              }
              if (v7_g) {
                float v677_data = r0[11];
                float v678_data = r1[0];
                float v681_data = ir2[0];
                ir2[0] = (v681_data + (v677_data * (sycl::group_broadcast(item.get_sub_group(), v678_data, 11))));
                float v684_data = r1[1];
                float v687_data = ir2[1];
                ir2[1] = (v687_data + (v677_data * (sycl::group_broadcast(item.get_sub_group(), v684_data, 11))));
                float v690_data = r1[2];
                float v693_data = ir2[2];
                ir2[2] = (v693_data + (v677_data * (sycl::group_broadcast(item.get_sub_group(), v690_data, 11))));
                float v696_data = r1[3];
                float v699_data = ir2[3];
                ir2[3] = (v699_data + (v677_data * (sycl::group_broadcast(item.get_sub_group(), v696_data, 11))));
                float v702_data = r1[4];
                float v705_data = ir2[4];
                ir2[4] = (v705_data + (v677_data * (sycl::group_broadcast(item.get_sub_group(), v702_data, 11))));
                float v708_data = r1[5];
                float v711_data = ir2[5];
                ir2[5] = (v711_data + (v677_data * (sycl::group_broadcast(item.get_sub_group(), v708_data, 11))));
                float v714_data = r1[6];
                float v717_data = ir2[6];
                ir2[6] = (v717_data + (v677_data * (sycl::group_broadcast(item.get_sub_group(), v714_data, 11))));
                float v720_data = r1[7];
                float v723_data = ir2[7];
                ir2[7] = (v723_data + (v677_data * (sycl::group_broadcast(item.get_sub_group(), v720_data, 11))));
                float v726_data = r1[8];
                float v729_data = ir2[8];
                ir2[8] = (v729_data + (v677_data * (sycl::group_broadcast(item.get_sub_group(), v726_data, 11))));
              }
              if (v7_g) {
                float v735_data = r0[12];
                float v736_data = r1[0];
                float v739_data = ir2[0];
                ir2[0] = (v739_data + (v735_data * (sycl::group_broadcast(item.get_sub_group(), v736_data, 12))));
                float v742_data = r1[1];
                float v745_data = ir2[1];
                ir2[1] = (v745_data + (v735_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 12))));
                float v748_data = r1[2];
                float v751_data = ir2[2];
                ir2[2] = (v751_data + (v735_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 12))));
                float v754_data = r1[3];
                float v757_data = ir2[3];
                ir2[3] = (v757_data + (v735_data * (sycl::group_broadcast(item.get_sub_group(), v754_data, 12))));
                float v760_data = r1[4];
                float v763_data = ir2[4];
                ir2[4] = (v763_data + (v735_data * (sycl::group_broadcast(item.get_sub_group(), v760_data, 12))));
                float v766_data = r1[5];
                float v769_data = ir2[5];
                ir2[5] = (v769_data + (v735_data * (sycl::group_broadcast(item.get_sub_group(), v766_data, 12))));
                float v772_data = r1[6];
                float v775_data = ir2[6];
                ir2[6] = (v775_data + (v735_data * (sycl::group_broadcast(item.get_sub_group(), v772_data, 12))));
                float v778_data = r1[7];
                float v781_data = ir2[7];
                ir2[7] = (v781_data + (v735_data * (sycl::group_broadcast(item.get_sub_group(), v778_data, 12))));
                float v784_data = r1[8];
                float v787_data = ir2[8];
                ir2[8] = (v787_data + (v735_data * (sycl::group_broadcast(item.get_sub_group(), v784_data, 12))));
              }
              if (v7_g) {
                float v793_data = r0[13];
                float v794_data = r1[0];
                float v797_data = ir2[0];
                ir2[0] = (v797_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v794_data, 13))));
                float v800_data = r1[1];
                float v803_data = ir2[1];
                ir2[1] = (v803_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v800_data, 13))));
                float v806_data = r1[2];
                float v809_data = ir2[2];
                ir2[2] = (v809_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v806_data, 13))));
                float v812_data = r1[3];
                float v815_data = ir2[3];
                ir2[3] = (v815_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v812_data, 13))));
                float v818_data = r1[4];
                float v821_data = ir2[4];
                ir2[4] = (v821_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v818_data, 13))));
                float v824_data = r1[5];
                float v827_data = ir2[5];
                ir2[5] = (v827_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v824_data, 13))));
                float v830_data = r1[6];
                float v833_data = ir2[6];
                ir2[6] = (v833_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v830_data, 13))));
                float v836_data = r1[7];
                float v839_data = ir2[7];
                ir2[7] = (v839_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v836_data, 13))));
                float v842_data = r1[8];
                float v845_data = ir2[8];
                ir2[8] = (v845_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 13))));
              }
              if (v7_g) {
                float v851_data = r0[14];
                float v852_data = r1[0];
                float v855_data = ir2[0];
                ir2[0] = (v855_data + (v851_data * (sycl::group_broadcast(item.get_sub_group(), v852_data, 14))));
                float v858_data = r1[1];
                float v861_data = ir2[1];
                ir2[1] = (v861_data + (v851_data * (sycl::group_broadcast(item.get_sub_group(), v858_data, 14))));
                float v864_data = r1[2];
                float v867_data = ir2[2];
                ir2[2] = (v867_data + (v851_data * (sycl::group_broadcast(item.get_sub_group(), v864_data, 14))));
                float v870_data = r1[3];
                float v873_data = ir2[3];
                ir2[3] = (v873_data + (v851_data * (sycl::group_broadcast(item.get_sub_group(), v870_data, 14))));
                float v876_data = r1[4];
                float v879_data = ir2[4];
                ir2[4] = (v879_data + (v851_data * (sycl::group_broadcast(item.get_sub_group(), v876_data, 14))));
                float v882_data = r1[5];
                float v885_data = ir2[5];
                ir2[5] = (v885_data + (v851_data * (sycl::group_broadcast(item.get_sub_group(), v882_data, 14))));
                float v888_data = r1[6];
                float v891_data = ir2[6];
                ir2[6] = (v891_data + (v851_data * (sycl::group_broadcast(item.get_sub_group(), v888_data, 14))));
                float v894_data = r1[7];
                float v897_data = ir2[7];
                ir2[7] = (v897_data + (v851_data * (sycl::group_broadcast(item.get_sub_group(), v894_data, 14))));
                float v900_data = r1[8];
                float v903_data = ir2[8];
                ir2[8] = (v903_data + (v851_data * (sycl::group_broadcast(item.get_sub_group(), v900_data, 14))));
              }
              if (v7_g) {
                float v909_data = r0[15];
                float v910_data = r1[0];
                float v913_data = ir2[0];
                ir2[0] = (v913_data + (v909_data * (sycl::group_broadcast(item.get_sub_group(), v910_data, 15))));
                float v916_data = r1[1];
                float v919_data = ir2[1];
                ir2[1] = (v919_data + (v909_data * (sycl::group_broadcast(item.get_sub_group(), v916_data, 15))));
                float v922_data = r1[2];
                float v925_data = ir2[2];
                ir2[2] = (v925_data + (v909_data * (sycl::group_broadcast(item.get_sub_group(), v922_data, 15))));
                float v928_data = r1[3];
                float v931_data = ir2[3];
                ir2[3] = (v931_data + (v909_data * (sycl::group_broadcast(item.get_sub_group(), v928_data, 15))));
                float v934_data = r1[4];
                float v937_data = ir2[4];
                ir2[4] = (v937_data + (v909_data * (sycl::group_broadcast(item.get_sub_group(), v934_data, 15))));
                float v940_data = r1[5];
                float v943_data = ir2[5];
                ir2[5] = (v943_data + (v909_data * (sycl::group_broadcast(item.get_sub_group(), v940_data, 15))));
                float v946_data = r1[6];
                float v949_data = ir2[6];
                ir2[6] = (v949_data + (v909_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 15))));
                float v952_data = r1[7];
                float v955_data = ir2[7];
                ir2[7] = (v955_data + (v909_data * (sycl::group_broadcast(item.get_sub_group(), v952_data, 15))));
                float v958_data = r1[8];
                float v961_data = ir2[8];
                ir2[8] = (v961_data + (v909_data * (sycl::group_broadcast(item.get_sub_group(), v958_data, 15))));
              }
              if (v7_g) {
                float v967_data = r0[16];
                float v968_data = r1[0];
                float v971_data = ir2[0];
                ir2[0] = (v971_data + (v967_data * (sycl::group_broadcast(item.get_sub_group(), v968_data, 16))));
                float v974_data = r1[1];
                float v977_data = ir2[1];
                ir2[1] = (v977_data + (v967_data * (sycl::group_broadcast(item.get_sub_group(), v974_data, 16))));
                float v980_data = r1[2];
                float v983_data = ir2[2];
                ir2[2] = (v983_data + (v967_data * (sycl::group_broadcast(item.get_sub_group(), v980_data, 16))));
                float v986_data = r1[3];
                float v989_data = ir2[3];
                ir2[3] = (v989_data + (v967_data * (sycl::group_broadcast(item.get_sub_group(), v986_data, 16))));
                float v992_data = r1[4];
                float v995_data = ir2[4];
                ir2[4] = (v995_data + (v967_data * (sycl::group_broadcast(item.get_sub_group(), v992_data, 16))));
                float v998_data = r1[5];
                float v1001_data = ir2[5];
                ir2[5] = (v1001_data + (v967_data * (sycl::group_broadcast(item.get_sub_group(), v998_data, 16))));
                float v1004_data = r1[6];
                float v1007_data = ir2[6];
                ir2[6] = (v1007_data + (v967_data * (sycl::group_broadcast(item.get_sub_group(), v1004_data, 16))));
                float v1010_data = r1[7];
                float v1013_data = ir2[7];
                ir2[7] = (v1013_data + (v967_data * (sycl::group_broadcast(item.get_sub_group(), v1010_data, 16))));
                float v1016_data = r1[8];
                float v1019_data = ir2[8];
                ir2[8] = (v1019_data + (v967_data * (sycl::group_broadcast(item.get_sub_group(), v1016_data, 16))));
              }
              if (v7_g) {
                float v1025_data = r0[17];
                float v1026_data = r1[0];
                float v1029_data = ir2[0];
                ir2[0] = (v1029_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v1026_data, 17))));
                float v1032_data = r1[1];
                float v1035_data = ir2[1];
                ir2[1] = (v1035_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v1032_data, 17))));
                float v1038_data = r1[2];
                float v1041_data = ir2[2];
                ir2[2] = (v1041_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v1038_data, 17))));
                float v1044_data = r1[3];
                float v1047_data = ir2[3];
                ir2[3] = (v1047_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v1044_data, 17))));
                float v1050_data = r1[4];
                float v1053_data = ir2[4];
                ir2[4] = (v1053_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v1050_data, 17))));
                float v1056_data = r1[5];
                float v1059_data = ir2[5];
                ir2[5] = (v1059_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v1056_data, 17))));
                float v1062_data = r1[6];
                float v1065_data = ir2[6];
                ir2[6] = (v1065_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v1062_data, 17))));
                float v1068_data = r1[7];
                float v1071_data = ir2[7];
                ir2[7] = (v1071_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v1068_data, 17))));
                float v1074_data = r1[8];
                float v1077_data = ir2[8];
                ir2[8] = (v1077_data + (v1025_data * (sycl::group_broadcast(item.get_sub_group(), v1074_data, 17))));
              }
              if (v7_g) {
                float v1083_data = r0[18];
                float v1084_data = r1[0];
                float v1087_data = ir2[0];
                ir2[0] = (v1087_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v1084_data, 18))));
                float v1090_data = r1[1];
                float v1093_data = ir2[1];
                ir2[1] = (v1093_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v1090_data, 18))));
                float v1096_data = r1[2];
                float v1099_data = ir2[2];
                ir2[2] = (v1099_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v1096_data, 18))));
                float v1102_data = r1[3];
                float v1105_data = ir2[3];
                ir2[3] = (v1105_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v1102_data, 18))));
                float v1108_data = r1[4];
                float v1111_data = ir2[4];
                ir2[4] = (v1111_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v1108_data, 18))));
                float v1114_data = r1[5];
                float v1117_data = ir2[5];
                ir2[5] = (v1117_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v1114_data, 18))));
                float v1120_data = r1[6];
                float v1123_data = ir2[6];
                ir2[6] = (v1123_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v1120_data, 18))));
                float v1126_data = r1[7];
                float v1129_data = ir2[7];
                ir2[7] = (v1129_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v1126_data, 18))));
                float v1132_data = r1[8];
                float v1135_data = ir2[8];
                ir2[8] = (v1135_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v1132_data, 18))));
              }
              if (v7_g) {
                float v1141_data = r0[19];
                float v1142_data = r1[0];
                float v1145_data = ir2[0];
                ir2[0] = (v1145_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 19))));
                float v1148_data = r1[1];
                float v1151_data = ir2[1];
                ir2[1] = (v1151_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 19))));
                float v1154_data = r1[2];
                float v1157_data = ir2[2];
                ir2[2] = (v1157_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1154_data, 19))));
                float v1160_data = r1[3];
                float v1163_data = ir2[3];
                ir2[3] = (v1163_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1160_data, 19))));
                float v1166_data = r1[4];
                float v1169_data = ir2[4];
                ir2[4] = (v1169_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1166_data, 19))));
                float v1172_data = r1[5];
                float v1175_data = ir2[5];
                ir2[5] = (v1175_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1172_data, 19))));
                float v1178_data = r1[6];
                float v1181_data = ir2[6];
                ir2[6] = (v1181_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1178_data, 19))));
                float v1184_data = r1[7];
                float v1187_data = ir2[7];
                ir2[7] = (v1187_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1184_data, 19))));
                float v1190_data = r1[8];
                float v1193_data = ir2[8];
                ir2[8] = (v1193_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1190_data, 19))));
              }
              if (v7_g) {
                float v1199_data = r0[20];
                float v1200_data = r1[0];
                float v1203_data = ir2[0];
                ir2[0] = (v1203_data + (v1199_data * (sycl::group_broadcast(item.get_sub_group(), v1200_data, 20))));
                float v1206_data = r1[1];
                float v1209_data = ir2[1];
                ir2[1] = (v1209_data + (v1199_data * (sycl::group_broadcast(item.get_sub_group(), v1206_data, 20))));
                float v1212_data = r1[2];
                float v1215_data = ir2[2];
                ir2[2] = (v1215_data + (v1199_data * (sycl::group_broadcast(item.get_sub_group(), v1212_data, 20))));
                float v1218_data = r1[3];
                float v1221_data = ir2[3];
                ir2[3] = (v1221_data + (v1199_data * (sycl::group_broadcast(item.get_sub_group(), v1218_data, 20))));
                float v1224_data = r1[4];
                float v1227_data = ir2[4];
                ir2[4] = (v1227_data + (v1199_data * (sycl::group_broadcast(item.get_sub_group(), v1224_data, 20))));
                float v1230_data = r1[5];
                float v1233_data = ir2[5];
                ir2[5] = (v1233_data + (v1199_data * (sycl::group_broadcast(item.get_sub_group(), v1230_data, 20))));
                float v1236_data = r1[6];
                float v1239_data = ir2[6];
                ir2[6] = (v1239_data + (v1199_data * (sycl::group_broadcast(item.get_sub_group(), v1236_data, 20))));
                float v1242_data = r1[7];
                float v1245_data = ir2[7];
                ir2[7] = (v1245_data + (v1199_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 20))));
                float v1248_data = r1[8];
                float v1251_data = ir2[8];
                ir2[8] = (v1251_data + (v1199_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 20))));
              }
              if (v7_g) {
                float v1257_data = r0[21];
                float v1258_data = r1[0];
                float v1261_data = ir2[0];
                ir2[0] = (v1261_data + (v1257_data * (sycl::group_broadcast(item.get_sub_group(), v1258_data, 21))));
                float v1264_data = r1[1];
                float v1267_data = ir2[1];
                ir2[1] = (v1267_data + (v1257_data * (sycl::group_broadcast(item.get_sub_group(), v1264_data, 21))));
                float v1270_data = r1[2];
                float v1273_data = ir2[2];
                ir2[2] = (v1273_data + (v1257_data * (sycl::group_broadcast(item.get_sub_group(), v1270_data, 21))));
                float v1276_data = r1[3];
                float v1279_data = ir2[3];
                ir2[3] = (v1279_data + (v1257_data * (sycl::group_broadcast(item.get_sub_group(), v1276_data, 21))));
                float v1282_data = r1[4];
                float v1285_data = ir2[4];
                ir2[4] = (v1285_data + (v1257_data * (sycl::group_broadcast(item.get_sub_group(), v1282_data, 21))));
                float v1288_data = r1[5];
                float v1291_data = ir2[5];
                ir2[5] = (v1291_data + (v1257_data * (sycl::group_broadcast(item.get_sub_group(), v1288_data, 21))));
                float v1294_data = r1[6];
                float v1297_data = ir2[6];
                ir2[6] = (v1297_data + (v1257_data * (sycl::group_broadcast(item.get_sub_group(), v1294_data, 21))));
                float v1300_data = r1[7];
                float v1303_data = ir2[7];
                ir2[7] = (v1303_data + (v1257_data * (sycl::group_broadcast(item.get_sub_group(), v1300_data, 21))));
                float v1306_data = r1[8];
                float v1309_data = ir2[8];
                ir2[8] = (v1309_data + (v1257_data * (sycl::group_broadcast(item.get_sub_group(), v1306_data, 21))));
              }
              if (v7_g) {
                float v1315_data = r0[22];
                float v1316_data = r1[0];
                float v1319_data = ir2[0];
                ir2[0] = (v1319_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v1316_data, 22))));
                float v1322_data = r1[1];
                float v1325_data = ir2[1];
                ir2[1] = (v1325_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v1322_data, 22))));
                float v1328_data = r1[2];
                float v1331_data = ir2[2];
                ir2[2] = (v1331_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v1328_data, 22))));
                float v1334_data = r1[3];
                float v1337_data = ir2[3];
                ir2[3] = (v1337_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v1334_data, 22))));
                float v1340_data = r1[4];
                float v1343_data = ir2[4];
                ir2[4] = (v1343_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v1340_data, 22))));
                float v1346_data = r1[5];
                float v1349_data = ir2[5];
                ir2[5] = (v1349_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 22))));
                float v1352_data = r1[6];
                float v1355_data = ir2[6];
                ir2[6] = (v1355_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v1352_data, 22))));
                float v1358_data = r1[7];
                float v1361_data = ir2[7];
                ir2[7] = (v1361_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v1358_data, 22))));
                float v1364_data = r1[8];
                float v1367_data = ir2[8];
                ir2[8] = (v1367_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v1364_data, 22))));
              }
              if (v7_g) {
                float v1373_data = r0[23];
                float v1374_data = r1[0];
                float v1377_data = ir2[0];
                ir2[0] = (v1377_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1374_data, 23))));
                float v1380_data = r1[1];
                float v1383_data = ir2[1];
                ir2[1] = (v1383_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1380_data, 23))));
                float v1386_data = r1[2];
                float v1389_data = ir2[2];
                ir2[2] = (v1389_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1386_data, 23))));
                float v1392_data = r1[3];
                float v1395_data = ir2[3];
                ir2[3] = (v1395_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1392_data, 23))));
                float v1398_data = r1[4];
                float v1401_data = ir2[4];
                ir2[4] = (v1401_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1398_data, 23))));
                float v1404_data = r1[5];
                float v1407_data = ir2[5];
                ir2[5] = (v1407_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1404_data, 23))));
                float v1410_data = r1[6];
                float v1413_data = ir2[6];
                ir2[6] = (v1413_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1410_data, 23))));
                float v1416_data = r1[7];
                float v1419_data = ir2[7];
                ir2[7] = (v1419_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1416_data, 23))));
                float v1422_data = r1[8];
                float v1425_data = ir2[8];
                ir2[8] = (v1425_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1422_data, 23))));
              }
              if (v7_g) {
                #pragma unroll
                for (int32_t v1431_n1 = 0; v1431_n1 < 9; ++v1431_n1) {
                  float v1433_data = ir2[v1431_n1];
                  r2[v1431_n1] = v1433_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v7_g) {
                #pragma unroll
                for (int32_t v1439_i1 = 0; v1439_i1 < 9; ++v1439_i1) {
                  float v1441_data = r2[v1439_i1];
                  glb_m0[(v6_lead + (v1439_i1 * 24))] = v1441_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


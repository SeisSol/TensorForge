// === base name ===
kernel_5e7da3148f

// === header ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5e7da3148f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5e7da3148f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 16; ++v10_i1) {
                  float v18_data = glb_m1[(v8_lead + (v10_i1 * 12))];
                  r0[v10_i1] = v18_data;
                }
              }
              float r1[8]{};
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
              float v27_lin = glb_m2[96 + item.get_local_id(0) * 1];
              r1[6] = v27_lin;
              float v28_lin = glb_m2[112 + item.get_local_id(0) * 1];
              r1[7] = v28_lin;
              // wait(r0 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = load{g>r}(glb_m0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
                  float v42_data = glb_m0[(v8_lead + (v34_i1 * 12))];
                  r2[v34_i1] = v42_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              // wait(r2 = load{g>r}(glb_m0););
              float r3[8]{};
              // r3 = +(r0 * r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir3[8]{};
              if (v8_lead < 12) {
                float v50_data = r0[0];
                float v51_data = r1[0];
                float v54_data = ir3[0];
                ir3[0] = (v54_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 0))));
                float v57_data = r1[1];
                float v60_data = ir3[1];
                ir3[1] = (v60_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 0))));
                float v63_data = r1[2];
                float v66_data = ir3[2];
                ir3[2] = (v66_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 0))));
                float v69_data = r1[3];
                float v72_data = ir3[3];
                ir3[3] = (v72_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 0))));
                float v75_data = r1[4];
                float v78_data = ir3[4];
                ir3[4] = (v78_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 0))));
                float v81_data = r1[5];
                float v84_data = ir3[5];
                ir3[5] = (v84_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 0))));
                float v87_data = r1[6];
                float v90_data = ir3[6];
                ir3[6] = (v90_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 0))));
                float v93_data = r1[7];
                float v96_data = ir3[7];
                ir3[7] = (v96_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 0))));
              }
              if (v8_lead < 12) {
                float v102_data = r0[1];
                float v103_data = r1[0];
                float v106_data = ir3[0];
                ir3[0] = (v106_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 1))));
                float v109_data = r1[1];
                float v112_data = ir3[1];
                ir3[1] = (v112_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 1))));
                float v115_data = r1[2];
                float v118_data = ir3[2];
                ir3[2] = (v118_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 1))));
                float v121_data = r1[3];
                float v124_data = ir3[3];
                ir3[3] = (v124_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 1))));
                float v127_data = r1[4];
                float v130_data = ir3[4];
                ir3[4] = (v130_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 1))));
                float v133_data = r1[5];
                float v136_data = ir3[5];
                ir3[5] = (v136_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 1))));
                float v139_data = r1[6];
                float v142_data = ir3[6];
                ir3[6] = (v142_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 1))));
                float v145_data = r1[7];
                float v148_data = ir3[7];
                ir3[7] = (v148_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 1))));
              }
              if (v8_lead < 12) {
                float v154_data = r0[2];
                float v155_data = r1[0];
                float v158_data = ir3[0];
                ir3[0] = (v158_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 2))));
                float v161_data = r1[1];
                float v164_data = ir3[1];
                ir3[1] = (v164_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 2))));
                float v167_data = r1[2];
                float v170_data = ir3[2];
                ir3[2] = (v170_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 2))));
                float v173_data = r1[3];
                float v176_data = ir3[3];
                ir3[3] = (v176_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 2))));
                float v179_data = r1[4];
                float v182_data = ir3[4];
                ir3[4] = (v182_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 2))));
                float v185_data = r1[5];
                float v188_data = ir3[5];
                ir3[5] = (v188_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 2))));
                float v191_data = r1[6];
                float v194_data = ir3[6];
                ir3[6] = (v194_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 2))));
                float v197_data = r1[7];
                float v200_data = ir3[7];
                ir3[7] = (v200_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 2))));
              }
              if (v8_lead < 12) {
                float v206_data = r0[3];
                float v207_data = r1[0];
                float v210_data = ir3[0];
                ir3[0] = (v210_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 3))));
                float v213_data = r1[1];
                float v216_data = ir3[1];
                ir3[1] = (v216_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v213_data, 3))));
                float v219_data = r1[2];
                float v222_data = ir3[2];
                ir3[2] = (v222_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v219_data, 3))));
                float v225_data = r1[3];
                float v228_data = ir3[3];
                ir3[3] = (v228_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v225_data, 3))));
                float v231_data = r1[4];
                float v234_data = ir3[4];
                ir3[4] = (v234_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v231_data, 3))));
                float v237_data = r1[5];
                float v240_data = ir3[5];
                ir3[5] = (v240_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v237_data, 3))));
                float v243_data = r1[6];
                float v246_data = ir3[6];
                ir3[6] = (v246_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v243_data, 3))));
                float v249_data = r1[7];
                float v252_data = ir3[7];
                ir3[7] = (v252_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v249_data, 3))));
              }
              if (v8_lead < 12) {
                float v258_data = r0[4];
                float v259_data = r1[0];
                float v262_data = ir3[0];
                ir3[0] = (v262_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v259_data, 4))));
                float v265_data = r1[1];
                float v268_data = ir3[1];
                ir3[1] = (v268_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v265_data, 4))));
                float v271_data = r1[2];
                float v274_data = ir3[2];
                ir3[2] = (v274_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v271_data, 4))));
                float v277_data = r1[3];
                float v280_data = ir3[3];
                ir3[3] = (v280_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v277_data, 4))));
                float v283_data = r1[4];
                float v286_data = ir3[4];
                ir3[4] = (v286_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v283_data, 4))));
                float v289_data = r1[5];
                float v292_data = ir3[5];
                ir3[5] = (v292_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v289_data, 4))));
                float v295_data = r1[6];
                float v298_data = ir3[6];
                ir3[6] = (v298_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v295_data, 4))));
                float v301_data = r1[7];
                float v304_data = ir3[7];
                ir3[7] = (v304_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v301_data, 4))));
              }
              if (v8_lead < 12) {
                float v310_data = r0[5];
                float v311_data = r1[0];
                float v314_data = ir3[0];
                ir3[0] = (v314_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v311_data, 5))));
                float v317_data = r1[1];
                float v320_data = ir3[1];
                ir3[1] = (v320_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v317_data, 5))));
                float v323_data = r1[2];
                float v326_data = ir3[2];
                ir3[2] = (v326_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v323_data, 5))));
                float v329_data = r1[3];
                float v332_data = ir3[3];
                ir3[3] = (v332_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v329_data, 5))));
                float v335_data = r1[4];
                float v338_data = ir3[4];
                ir3[4] = (v338_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 5))));
                float v341_data = r1[5];
                float v344_data = ir3[5];
                ir3[5] = (v344_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v341_data, 5))));
                float v347_data = r1[6];
                float v350_data = ir3[6];
                ir3[6] = (v350_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v347_data, 5))));
                float v353_data = r1[7];
                float v356_data = ir3[7];
                ir3[7] = (v356_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v353_data, 5))));
              }
              if (v8_lead < 12) {
                float v362_data = r0[6];
                float v363_data = r1[0];
                float v366_data = ir3[0];
                ir3[0] = (v366_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v363_data, 6))));
                float v369_data = r1[1];
                float v372_data = ir3[1];
                ir3[1] = (v372_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 6))));
                float v375_data = r1[2];
                float v378_data = ir3[2];
                ir3[2] = (v378_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v375_data, 6))));
                float v381_data = r1[3];
                float v384_data = ir3[3];
                ir3[3] = (v384_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v381_data, 6))));
                float v387_data = r1[4];
                float v390_data = ir3[4];
                ir3[4] = (v390_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v387_data, 6))));
                float v393_data = r1[5];
                float v396_data = ir3[5];
                ir3[5] = (v396_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v393_data, 6))));
                float v399_data = r1[6];
                float v402_data = ir3[6];
                ir3[6] = (v402_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 6))));
                float v405_data = r1[7];
                float v408_data = ir3[7];
                ir3[7] = (v408_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v405_data, 6))));
              }
              if (v8_lead < 12) {
                float v414_data = r0[7];
                float v415_data = r1[0];
                float v418_data = ir3[0];
                ir3[0] = (v418_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v415_data, 7))));
                float v421_data = r1[1];
                float v424_data = ir3[1];
                ir3[1] = (v424_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v421_data, 7))));
                float v427_data = r1[2];
                float v430_data = ir3[2];
                ir3[2] = (v430_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v427_data, 7))));
                float v433_data = r1[3];
                float v436_data = ir3[3];
                ir3[3] = (v436_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v433_data, 7))));
                float v439_data = r1[4];
                float v442_data = ir3[4];
                ir3[4] = (v442_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v439_data, 7))));
                float v445_data = r1[5];
                float v448_data = ir3[5];
                ir3[5] = (v448_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v445_data, 7))));
                float v451_data = r1[6];
                float v454_data = ir3[6];
                ir3[6] = (v454_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v451_data, 7))));
                float v457_data = r1[7];
                float v460_data = ir3[7];
                ir3[7] = (v460_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v457_data, 7))));
              }
              if (v8_lead < 12) {
                float v466_data = r0[8];
                float v467_data = r1[0];
                float v470_data = ir3[0];
                ir3[0] = (v470_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v467_data, 8))));
                float v473_data = r1[1];
                float v476_data = ir3[1];
                ir3[1] = (v476_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v473_data, 8))));
                float v479_data = r1[2];
                float v482_data = ir3[2];
                ir3[2] = (v482_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v479_data, 8))));
                float v485_data = r1[3];
                float v488_data = ir3[3];
                ir3[3] = (v488_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v485_data, 8))));
                float v491_data = r1[4];
                float v494_data = ir3[4];
                ir3[4] = (v494_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v491_data, 8))));
                float v497_data = r1[5];
                float v500_data = ir3[5];
                ir3[5] = (v500_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v497_data, 8))));
                float v503_data = r1[6];
                float v506_data = ir3[6];
                ir3[6] = (v506_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v503_data, 8))));
                float v509_data = r1[7];
                float v512_data = ir3[7];
                ir3[7] = (v512_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v509_data, 8))));
              }
              if (v8_lead < 12) {
                float v518_data = r0[9];
                float v519_data = r1[0];
                float v522_data = ir3[0];
                ir3[0] = (v522_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v519_data, 9))));
                float v525_data = r1[1];
                float v528_data = ir3[1];
                ir3[1] = (v528_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v525_data, 9))));
                float v531_data = r1[2];
                float v534_data = ir3[2];
                ir3[2] = (v534_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v531_data, 9))));
                float v537_data = r1[3];
                float v540_data = ir3[3];
                ir3[3] = (v540_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v537_data, 9))));
                float v543_data = r1[4];
                float v546_data = ir3[4];
                ir3[4] = (v546_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v543_data, 9))));
                float v549_data = r1[5];
                float v552_data = ir3[5];
                ir3[5] = (v552_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v549_data, 9))));
                float v555_data = r1[6];
                float v558_data = ir3[6];
                ir3[6] = (v558_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v555_data, 9))));
                float v561_data = r1[7];
                float v564_data = ir3[7];
                ir3[7] = (v564_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v561_data, 9))));
              }
              if (v8_lead < 12) {
                float v570_data = r0[10];
                float v571_data = r1[0];
                float v574_data = ir3[0];
                ir3[0] = (v574_data + (v570_data * (sycl::group_broadcast(item.get_sub_group(), v571_data, 10))));
                float v577_data = r1[1];
                float v580_data = ir3[1];
                ir3[1] = (v580_data + (v570_data * (sycl::group_broadcast(item.get_sub_group(), v577_data, 10))));
                float v583_data = r1[2];
                float v586_data = ir3[2];
                ir3[2] = (v586_data + (v570_data * (sycl::group_broadcast(item.get_sub_group(), v583_data, 10))));
                float v589_data = r1[3];
                float v592_data = ir3[3];
                ir3[3] = (v592_data + (v570_data * (sycl::group_broadcast(item.get_sub_group(), v589_data, 10))));
                float v595_data = r1[4];
                float v598_data = ir3[4];
                ir3[4] = (v598_data + (v570_data * (sycl::group_broadcast(item.get_sub_group(), v595_data, 10))));
                float v601_data = r1[5];
                float v604_data = ir3[5];
                ir3[5] = (v604_data + (v570_data * (sycl::group_broadcast(item.get_sub_group(), v601_data, 10))));
                float v607_data = r1[6];
                float v610_data = ir3[6];
                ir3[6] = (v610_data + (v570_data * (sycl::group_broadcast(item.get_sub_group(), v607_data, 10))));
                float v613_data = r1[7];
                float v616_data = ir3[7];
                ir3[7] = (v616_data + (v570_data * (sycl::group_broadcast(item.get_sub_group(), v613_data, 10))));
              }
              if (v8_lead < 12) {
                float v622_data = r0[11];
                float v623_data = r1[0];
                float v626_data = ir3[0];
                ir3[0] = (v626_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v623_data, 11))));
                float v629_data = r1[1];
                float v632_data = ir3[1];
                ir3[1] = (v632_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v629_data, 11))));
                float v635_data = r1[2];
                float v638_data = ir3[2];
                ir3[2] = (v638_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v635_data, 11))));
                float v641_data = r1[3];
                float v644_data = ir3[3];
                ir3[3] = (v644_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v641_data, 11))));
                float v647_data = r1[4];
                float v650_data = ir3[4];
                ir3[4] = (v650_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v647_data, 11))));
                float v653_data = r1[5];
                float v656_data = ir3[5];
                ir3[5] = (v656_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v653_data, 11))));
                float v659_data = r1[6];
                float v662_data = ir3[6];
                ir3[6] = (v662_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v659_data, 11))));
                float v665_data = r1[7];
                float v668_data = ir3[7];
                ir3[7] = (v668_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v665_data, 11))));
              }
              if (v8_lead < 12) {
                float v674_data = r0[12];
                float v675_data = r1[0];
                float v678_data = ir3[0];
                ir3[0] = (v678_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v675_data, 12))));
                float v681_data = r1[1];
                float v684_data = ir3[1];
                ir3[1] = (v684_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v681_data, 12))));
                float v687_data = r1[2];
                float v690_data = ir3[2];
                ir3[2] = (v690_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v687_data, 12))));
                float v693_data = r1[3];
                float v696_data = ir3[3];
                ir3[3] = (v696_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v693_data, 12))));
                float v699_data = r1[4];
                float v702_data = ir3[4];
                ir3[4] = (v702_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v699_data, 12))));
                float v705_data = r1[5];
                float v708_data = ir3[5];
                ir3[5] = (v708_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v705_data, 12))));
                float v711_data = r1[6];
                float v714_data = ir3[6];
                ir3[6] = (v714_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v711_data, 12))));
                float v717_data = r1[7];
                float v720_data = ir3[7];
                ir3[7] = (v720_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v717_data, 12))));
              }
              if (v8_lead < 12) {
                float v726_data = r0[13];
                float v727_data = r1[0];
                float v730_data = ir3[0];
                ir3[0] = (v730_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v727_data, 13))));
                float v733_data = r1[1];
                float v736_data = ir3[1];
                ir3[1] = (v736_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v733_data, 13))));
                float v739_data = r1[2];
                float v742_data = ir3[2];
                ir3[2] = (v742_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v739_data, 13))));
                float v745_data = r1[3];
                float v748_data = ir3[3];
                ir3[3] = (v748_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v745_data, 13))));
                float v751_data = r1[4];
                float v754_data = ir3[4];
                ir3[4] = (v754_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v751_data, 13))));
                float v757_data = r1[5];
                float v760_data = ir3[5];
                ir3[5] = (v760_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v757_data, 13))));
                float v763_data = r1[6];
                float v766_data = ir3[6];
                ir3[6] = (v766_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v763_data, 13))));
                float v769_data = r1[7];
                float v772_data = ir3[7];
                ir3[7] = (v772_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v769_data, 13))));
              }
              if (v8_lead < 12) {
                float v778_data = r0[14];
                float v779_data = r1[0];
                float v782_data = ir3[0];
                ir3[0] = (v782_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v779_data, 14))));
                float v785_data = r1[1];
                float v788_data = ir3[1];
                ir3[1] = (v788_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v785_data, 14))));
                float v791_data = r1[2];
                float v794_data = ir3[2];
                ir3[2] = (v794_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v791_data, 14))));
                float v797_data = r1[3];
                float v800_data = ir3[3];
                ir3[3] = (v800_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v797_data, 14))));
                float v803_data = r1[4];
                float v806_data = ir3[4];
                ir3[4] = (v806_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v803_data, 14))));
                float v809_data = r1[5];
                float v812_data = ir3[5];
                ir3[5] = (v812_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v809_data, 14))));
                float v815_data = r1[6];
                float v818_data = ir3[6];
                ir3[6] = (v818_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v815_data, 14))));
                float v821_data = r1[7];
                float v824_data = ir3[7];
                ir3[7] = (v824_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v821_data, 14))));
              }
              if (v8_lead < 12) {
                float v830_data = r0[15];
                float v831_data = r1[0];
                float v834_data = ir3[0];
                ir3[0] = (v834_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v831_data, 15))));
                float v837_data = r1[1];
                float v840_data = ir3[1];
                ir3[1] = (v840_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v837_data, 15))));
                float v843_data = r1[2];
                float v846_data = ir3[2];
                ir3[2] = (v846_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v843_data, 15))));
                float v849_data = r1[3];
                float v852_data = ir3[3];
                ir3[3] = (v852_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v849_data, 15))));
                float v855_data = r1[4];
                float v858_data = ir3[4];
                ir3[4] = (v858_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v855_data, 15))));
                float v861_data = r1[5];
                float v864_data = ir3[5];
                ir3[5] = (v864_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v861_data, 15))));
                float v867_data = r1[6];
                float v870_data = ir3[6];
                ir3[6] = (v870_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v867_data, 15))));
                float v873_data = r1[7];
                float v876_data = ir3[7];
                ir3[7] = (v876_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v873_data, 15))));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v882_n1 = 0; v882_n1 < 8; ++v882_n1) {
                  float v884_data = ir3[v882_n1];
                  float v886_data = r2[v882_n1];
                  r3[v882_n1] = (v886_data + v884_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v893_i1 = 0; v893_i1 < 8; ++v893_i1) {
                  float v895_data = r3[v893_i1];
                  glb_m0[(v8_lead + (v893_i1 * 12))] = v895_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


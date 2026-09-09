// === base name ===
kernel_c9cca2c79ae04629

// === header ===
void launcher_kernel_c9cca2c79ae04629(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_c9cca2c79ae04629(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c9cca2c79ae04629(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c9cca2c79ae04629(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
                int32_t v21_a = (v12_lead + 4) - 4;
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 16; ++v14_i1) {
                  float v24_data = glb_m1[(v21_a + (v14_i1 * 12))];
                  r0[v14_i1] = v24_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v30_i0 = 0; v30_i0 < 1; ++v30_i0) {
                int32_t v36_lead = v12_lead + (v30_i0 * 16);
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
                  float v39_data = glb_m2[(v36_lead + (v31_i1 * 16))];
                  r1[(v30_i0 + v31_i1)] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(16, 28), (0, 8)] [(0, 16)]
              float ir2[8]{};
              if (v12_lead < 12) {
                float v47_data = r0[0];
                float v48_data = r1[0];
                float v51_data = ir2[0];
                ir2[0] = (v51_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 0))));
                float v54_data = r1[1];
                float v57_data = ir2[1];
                ir2[1] = (v57_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v54_data, 0))));
                float v60_data = r1[2];
                float v63_data = ir2[2];
                ir2[2] = (v63_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v60_data, 0))));
                float v66_data = r1[3];
                float v69_data = ir2[3];
                ir2[3] = (v69_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 0))));
                float v72_data = r1[4];
                float v75_data = ir2[4];
                ir2[4] = (v75_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 0))));
                float v78_data = r1[5];
                float v81_data = ir2[5];
                ir2[5] = (v81_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 0))));
                float v84_data = r1[6];
                float v87_data = ir2[6];
                ir2[6] = (v87_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 0))));
                float v90_data = r1[7];
                float v93_data = ir2[7];
                ir2[7] = (v93_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 0))));
              }
              if (v12_lead < 12) {
                float v99_data = r0[1];
                float v100_data = r1[0];
                float v103_data = ir2[0];
                ir2[0] = (v103_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 1))));
                float v106_data = r1[1];
                float v109_data = ir2[1];
                ir2[1] = (v109_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 1))));
                float v112_data = r1[2];
                float v115_data = ir2[2];
                ir2[2] = (v115_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 1))));
                float v118_data = r1[3];
                float v121_data = ir2[3];
                ir2[3] = (v121_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 1))));
                float v124_data = r1[4];
                float v127_data = ir2[4];
                ir2[4] = (v127_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 1))));
                float v130_data = r1[5];
                float v133_data = ir2[5];
                ir2[5] = (v133_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 1))));
                float v136_data = r1[6];
                float v139_data = ir2[6];
                ir2[6] = (v139_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 1))));
                float v142_data = r1[7];
                float v145_data = ir2[7];
                ir2[7] = (v145_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 1))));
              }
              if (v12_lead < 12) {
                float v151_data = r0[2];
                float v152_data = r1[0];
                float v155_data = ir2[0];
                ir2[0] = (v155_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 2))));
                float v158_data = r1[1];
                float v161_data = ir2[1];
                ir2[1] = (v161_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v158_data, 2))));
                float v164_data = r1[2];
                float v167_data = ir2[2];
                ir2[2] = (v167_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v164_data, 2))));
                float v170_data = r1[3];
                float v173_data = ir2[3];
                ir2[3] = (v173_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v170_data, 2))));
                float v176_data = r1[4];
                float v179_data = ir2[4];
                ir2[4] = (v179_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v176_data, 2))));
                float v182_data = r1[5];
                float v185_data = ir2[5];
                ir2[5] = (v185_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v182_data, 2))));
                float v188_data = r1[6];
                float v191_data = ir2[6];
                ir2[6] = (v191_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v188_data, 2))));
                float v194_data = r1[7];
                float v197_data = ir2[7];
                ir2[7] = (v197_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v194_data, 2))));
              }
              if (v12_lead < 12) {
                float v203_data = r0[3];
                float v204_data = r1[0];
                float v207_data = ir2[0];
                ir2[0] = (v207_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v204_data, 3))));
                float v210_data = r1[1];
                float v213_data = ir2[1];
                ir2[1] = (v213_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v210_data, 3))));
                float v216_data = r1[2];
                float v219_data = ir2[2];
                ir2[2] = (v219_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v216_data, 3))));
                float v222_data = r1[3];
                float v225_data = ir2[3];
                ir2[3] = (v225_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v222_data, 3))));
                float v228_data = r1[4];
                float v231_data = ir2[4];
                ir2[4] = (v231_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v228_data, 3))));
                float v234_data = r1[5];
                float v237_data = ir2[5];
                ir2[5] = (v237_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v234_data, 3))));
                float v240_data = r1[6];
                float v243_data = ir2[6];
                ir2[6] = (v243_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v240_data, 3))));
                float v246_data = r1[7];
                float v249_data = ir2[7];
                ir2[7] = (v249_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 3))));
              }
              if (v12_lead < 12) {
                float v255_data = r0[4];
                float v256_data = r1[0];
                float v259_data = ir2[0];
                ir2[0] = (v259_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v256_data, 4))));
                float v262_data = r1[1];
                float v265_data = ir2[1];
                ir2[1] = (v265_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v262_data, 4))));
                float v268_data = r1[2];
                float v271_data = ir2[2];
                ir2[2] = (v271_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v268_data, 4))));
                float v274_data = r1[3];
                float v277_data = ir2[3];
                ir2[3] = (v277_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v274_data, 4))));
                float v280_data = r1[4];
                float v283_data = ir2[4];
                ir2[4] = (v283_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v280_data, 4))));
                float v286_data = r1[5];
                float v289_data = ir2[5];
                ir2[5] = (v289_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v286_data, 4))));
                float v292_data = r1[6];
                float v295_data = ir2[6];
                ir2[6] = (v295_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v292_data, 4))));
                float v298_data = r1[7];
                float v301_data = ir2[7];
                ir2[7] = (v301_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v298_data, 4))));
              }
              if (v12_lead < 12) {
                float v307_data = r0[5];
                float v308_data = r1[0];
                float v311_data = ir2[0];
                ir2[0] = (v311_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v308_data, 5))));
                float v314_data = r1[1];
                float v317_data = ir2[1];
                ir2[1] = (v317_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v314_data, 5))));
                float v320_data = r1[2];
                float v323_data = ir2[2];
                ir2[2] = (v323_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 5))));
                float v326_data = r1[3];
                float v329_data = ir2[3];
                ir2[3] = (v329_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v326_data, 5))));
                float v332_data = r1[4];
                float v335_data = ir2[4];
                ir2[4] = (v335_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v332_data, 5))));
                float v338_data = r1[5];
                float v341_data = ir2[5];
                ir2[5] = (v341_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v338_data, 5))));
                float v344_data = r1[6];
                float v347_data = ir2[6];
                ir2[6] = (v347_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v344_data, 5))));
                float v350_data = r1[7];
                float v353_data = ir2[7];
                ir2[7] = (v353_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 5))));
              }
              if (v12_lead < 12) {
                float v359_data = r0[6];
                float v360_data = r1[0];
                float v363_data = ir2[0];
                ir2[0] = (v363_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v360_data, 6))));
                float v366_data = r1[1];
                float v369_data = ir2[1];
                ir2[1] = (v369_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v366_data, 6))));
                float v372_data = r1[2];
                float v375_data = ir2[2];
                ir2[2] = (v375_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v372_data, 6))));
                float v378_data = r1[3];
                float v381_data = ir2[3];
                ir2[3] = (v381_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v378_data, 6))));
                float v384_data = r1[4];
                float v387_data = ir2[4];
                ir2[4] = (v387_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 6))));
                float v390_data = r1[5];
                float v393_data = ir2[5];
                ir2[5] = (v393_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v390_data, 6))));
                float v396_data = r1[6];
                float v399_data = ir2[6];
                ir2[6] = (v399_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v396_data, 6))));
                float v402_data = r1[7];
                float v405_data = ir2[7];
                ir2[7] = (v405_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v402_data, 6))));
              }
              if (v12_lead < 12) {
                float v411_data = r0[7];
                float v412_data = r1[0];
                float v415_data = ir2[0];
                ir2[0] = (v415_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v412_data, 7))));
                float v418_data = r1[1];
                float v421_data = ir2[1];
                ir2[1] = (v421_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v418_data, 7))));
                float v424_data = r1[2];
                float v427_data = ir2[2];
                ir2[2] = (v427_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v424_data, 7))));
                float v430_data = r1[3];
                float v433_data = ir2[3];
                ir2[3] = (v433_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v430_data, 7))));
                float v436_data = r1[4];
                float v439_data = ir2[4];
                ir2[4] = (v439_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v436_data, 7))));
                float v442_data = r1[5];
                float v445_data = ir2[5];
                ir2[5] = (v445_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 7))));
                float v448_data = r1[6];
                float v451_data = ir2[6];
                ir2[6] = (v451_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 7))));
                float v454_data = r1[7];
                float v457_data = ir2[7];
                ir2[7] = (v457_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v454_data, 7))));
              }
              if (v12_lead < 12) {
                float v463_data = r0[8];
                float v464_data = r1[0];
                float v467_data = ir2[0];
                ir2[0] = (v467_data + (v463_data * (sycl::group_broadcast(item.get_sub_group(), v464_data, 8))));
                float v470_data = r1[1];
                float v473_data = ir2[1];
                ir2[1] = (v473_data + (v463_data * (sycl::group_broadcast(item.get_sub_group(), v470_data, 8))));
                float v476_data = r1[2];
                float v479_data = ir2[2];
                ir2[2] = (v479_data + (v463_data * (sycl::group_broadcast(item.get_sub_group(), v476_data, 8))));
                float v482_data = r1[3];
                float v485_data = ir2[3];
                ir2[3] = (v485_data + (v463_data * (sycl::group_broadcast(item.get_sub_group(), v482_data, 8))));
                float v488_data = r1[4];
                float v491_data = ir2[4];
                ir2[4] = (v491_data + (v463_data * (sycl::group_broadcast(item.get_sub_group(), v488_data, 8))));
                float v494_data = r1[5];
                float v497_data = ir2[5];
                ir2[5] = (v497_data + (v463_data * (sycl::group_broadcast(item.get_sub_group(), v494_data, 8))));
                float v500_data = r1[6];
                float v503_data = ir2[6];
                ir2[6] = (v503_data + (v463_data * (sycl::group_broadcast(item.get_sub_group(), v500_data, 8))));
                float v506_data = r1[7];
                float v509_data = ir2[7];
                ir2[7] = (v509_data + (v463_data * (sycl::group_broadcast(item.get_sub_group(), v506_data, 8))));
              }
              if (v12_lead < 12) {
                float v515_data = r0[9];
                float v516_data = r1[0];
                float v519_data = ir2[0];
                ir2[0] = (v519_data + (v515_data * (sycl::group_broadcast(item.get_sub_group(), v516_data, 9))));
                float v522_data = r1[1];
                float v525_data = ir2[1];
                ir2[1] = (v525_data + (v515_data * (sycl::group_broadcast(item.get_sub_group(), v522_data, 9))));
                float v528_data = r1[2];
                float v531_data = ir2[2];
                ir2[2] = (v531_data + (v515_data * (sycl::group_broadcast(item.get_sub_group(), v528_data, 9))));
                float v534_data = r1[3];
                float v537_data = ir2[3];
                ir2[3] = (v537_data + (v515_data * (sycl::group_broadcast(item.get_sub_group(), v534_data, 9))));
                float v540_data = r1[4];
                float v543_data = ir2[4];
                ir2[4] = (v543_data + (v515_data * (sycl::group_broadcast(item.get_sub_group(), v540_data, 9))));
                float v546_data = r1[5];
                float v549_data = ir2[5];
                ir2[5] = (v549_data + (v515_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 9))));
                float v552_data = r1[6];
                float v555_data = ir2[6];
                ir2[6] = (v555_data + (v515_data * (sycl::group_broadcast(item.get_sub_group(), v552_data, 9))));
                float v558_data = r1[7];
                float v561_data = ir2[7];
                ir2[7] = (v561_data + (v515_data * (sycl::group_broadcast(item.get_sub_group(), v558_data, 9))));
              }
              if (v12_lead < 12) {
                float v567_data = r0[10];
                float v568_data = r1[0];
                float v571_data = ir2[0];
                ir2[0] = (v571_data + (v567_data * (sycl::group_broadcast(item.get_sub_group(), v568_data, 10))));
                float v574_data = r1[1];
                float v577_data = ir2[1];
                ir2[1] = (v577_data + (v567_data * (sycl::group_broadcast(item.get_sub_group(), v574_data, 10))));
                float v580_data = r1[2];
                float v583_data = ir2[2];
                ir2[2] = (v583_data + (v567_data * (sycl::group_broadcast(item.get_sub_group(), v580_data, 10))));
                float v586_data = r1[3];
                float v589_data = ir2[3];
                ir2[3] = (v589_data + (v567_data * (sycl::group_broadcast(item.get_sub_group(), v586_data, 10))));
                float v592_data = r1[4];
                float v595_data = ir2[4];
                ir2[4] = (v595_data + (v567_data * (sycl::group_broadcast(item.get_sub_group(), v592_data, 10))));
                float v598_data = r1[5];
                float v601_data = ir2[5];
                ir2[5] = (v601_data + (v567_data * (sycl::group_broadcast(item.get_sub_group(), v598_data, 10))));
                float v604_data = r1[6];
                float v607_data = ir2[6];
                ir2[6] = (v607_data + (v567_data * (sycl::group_broadcast(item.get_sub_group(), v604_data, 10))));
                float v610_data = r1[7];
                float v613_data = ir2[7];
                ir2[7] = (v613_data + (v567_data * (sycl::group_broadcast(item.get_sub_group(), v610_data, 10))));
              }
              if (v12_lead < 12) {
                float v619_data = r0[11];
                float v620_data = r1[0];
                float v623_data = ir2[0];
                ir2[0] = (v623_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v620_data, 11))));
                float v626_data = r1[1];
                float v629_data = ir2[1];
                ir2[1] = (v629_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v626_data, 11))));
                float v632_data = r1[2];
                float v635_data = ir2[2];
                ir2[2] = (v635_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v632_data, 11))));
                float v638_data = r1[3];
                float v641_data = ir2[3];
                ir2[3] = (v641_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v638_data, 11))));
                float v644_data = r1[4];
                float v647_data = ir2[4];
                ir2[4] = (v647_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v644_data, 11))));
                float v650_data = r1[5];
                float v653_data = ir2[5];
                ir2[5] = (v653_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v650_data, 11))));
                float v656_data = r1[6];
                float v659_data = ir2[6];
                ir2[6] = (v659_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v656_data, 11))));
                float v662_data = r1[7];
                float v665_data = ir2[7];
                ir2[7] = (v665_data + (v619_data * (sycl::group_broadcast(item.get_sub_group(), v662_data, 11))));
              }
              if (v12_lead < 12) {
                float v671_data = r0[12];
                float v672_data = r1[0];
                float v675_data = ir2[0];
                ir2[0] = (v675_data + (v671_data * (sycl::group_broadcast(item.get_sub_group(), v672_data, 12))));
                float v678_data = r1[1];
                float v681_data = ir2[1];
                ir2[1] = (v681_data + (v671_data * (sycl::group_broadcast(item.get_sub_group(), v678_data, 12))));
                float v684_data = r1[2];
                float v687_data = ir2[2];
                ir2[2] = (v687_data + (v671_data * (sycl::group_broadcast(item.get_sub_group(), v684_data, 12))));
                float v690_data = r1[3];
                float v693_data = ir2[3];
                ir2[3] = (v693_data + (v671_data * (sycl::group_broadcast(item.get_sub_group(), v690_data, 12))));
                float v696_data = r1[4];
                float v699_data = ir2[4];
                ir2[4] = (v699_data + (v671_data * (sycl::group_broadcast(item.get_sub_group(), v696_data, 12))));
                float v702_data = r1[5];
                float v705_data = ir2[5];
                ir2[5] = (v705_data + (v671_data * (sycl::group_broadcast(item.get_sub_group(), v702_data, 12))));
                float v708_data = r1[6];
                float v711_data = ir2[6];
                ir2[6] = (v711_data + (v671_data * (sycl::group_broadcast(item.get_sub_group(), v708_data, 12))));
                float v714_data = r1[7];
                float v717_data = ir2[7];
                ir2[7] = (v717_data + (v671_data * (sycl::group_broadcast(item.get_sub_group(), v714_data, 12))));
              }
              if (v12_lead < 12) {
                float v723_data = r0[13];
                float v724_data = r1[0];
                float v727_data = ir2[0];
                ir2[0] = (v727_data + (v723_data * (sycl::group_broadcast(item.get_sub_group(), v724_data, 13))));
                float v730_data = r1[1];
                float v733_data = ir2[1];
                ir2[1] = (v733_data + (v723_data * (sycl::group_broadcast(item.get_sub_group(), v730_data, 13))));
                float v736_data = r1[2];
                float v739_data = ir2[2];
                ir2[2] = (v739_data + (v723_data * (sycl::group_broadcast(item.get_sub_group(), v736_data, 13))));
                float v742_data = r1[3];
                float v745_data = ir2[3];
                ir2[3] = (v745_data + (v723_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 13))));
                float v748_data = r1[4];
                float v751_data = ir2[4];
                ir2[4] = (v751_data + (v723_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 13))));
                float v754_data = r1[5];
                float v757_data = ir2[5];
                ir2[5] = (v757_data + (v723_data * (sycl::group_broadcast(item.get_sub_group(), v754_data, 13))));
                float v760_data = r1[6];
                float v763_data = ir2[6];
                ir2[6] = (v763_data + (v723_data * (sycl::group_broadcast(item.get_sub_group(), v760_data, 13))));
                float v766_data = r1[7];
                float v769_data = ir2[7];
                ir2[7] = (v769_data + (v723_data * (sycl::group_broadcast(item.get_sub_group(), v766_data, 13))));
              }
              if (v12_lead < 12) {
                float v775_data = r0[14];
                float v776_data = r1[0];
                float v779_data = ir2[0];
                ir2[0] = (v779_data + (v775_data * (sycl::group_broadcast(item.get_sub_group(), v776_data, 14))));
                float v782_data = r1[1];
                float v785_data = ir2[1];
                ir2[1] = (v785_data + (v775_data * (sycl::group_broadcast(item.get_sub_group(), v782_data, 14))));
                float v788_data = r1[2];
                float v791_data = ir2[2];
                ir2[2] = (v791_data + (v775_data * (sycl::group_broadcast(item.get_sub_group(), v788_data, 14))));
                float v794_data = r1[3];
                float v797_data = ir2[3];
                ir2[3] = (v797_data + (v775_data * (sycl::group_broadcast(item.get_sub_group(), v794_data, 14))));
                float v800_data = r1[4];
                float v803_data = ir2[4];
                ir2[4] = (v803_data + (v775_data * (sycl::group_broadcast(item.get_sub_group(), v800_data, 14))));
                float v806_data = r1[5];
                float v809_data = ir2[5];
                ir2[5] = (v809_data + (v775_data * (sycl::group_broadcast(item.get_sub_group(), v806_data, 14))));
                float v812_data = r1[6];
                float v815_data = ir2[6];
                ir2[6] = (v815_data + (v775_data * (sycl::group_broadcast(item.get_sub_group(), v812_data, 14))));
                float v818_data = r1[7];
                float v821_data = ir2[7];
                ir2[7] = (v821_data + (v775_data * (sycl::group_broadcast(item.get_sub_group(), v818_data, 14))));
              }
              if (v12_lead < 12) {
                float v827_data = r0[15];
                float v828_data = r1[0];
                float v831_data = ir2[0];
                ir2[0] = (v831_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v828_data, 15))));
                float v834_data = r1[1];
                float v837_data = ir2[1];
                ir2[1] = (v837_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v834_data, 15))));
                float v840_data = r1[2];
                float v843_data = ir2[2];
                ir2[2] = (v843_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v840_data, 15))));
                float v846_data = r1[3];
                float v849_data = ir2[3];
                ir2[3] = (v849_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 15))));
                float v852_data = r1[4];
                float v855_data = ir2[4];
                ir2[4] = (v855_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v852_data, 15))));
                float v858_data = r1[5];
                float v861_data = ir2[5];
                ir2[5] = (v861_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v858_data, 15))));
                float v864_data = r1[6];
                float v867_data = ir2[6];
                ir2[6] = (v867_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v864_data, 15))));
                float v870_data = r1[7];
                float v873_data = ir2[7];
                ir2[7] = (v873_data + (v827_data * (sycl::group_broadcast(item.get_sub_group(), v870_data, 15))));
              }
              if (v12_lead < 12) {
                #pragma unroll
                for (int32_t v879_n1 = 0; v879_n1 < 8; ++v879_n1) {
                  float v881_data = ir2[v879_n1];
                  r2[v879_n1] = v881_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v12_lead < 12) {
                int32_t v896_a = ((v12_lead + 16_i32) + -12) - 4;
                #pragma unroll
                for (int32_t v887_i1 = 0; v887_i1 < 8; ++v887_i1) {
                  float v889_data = r2[v887_i1];
                  glb_m0[(v896_a + (v887_i1 * 12))] = v889_data;
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


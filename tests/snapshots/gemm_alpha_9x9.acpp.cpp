// === base name ===
kernel_98748807e30b6fdd

// === header ===
void launcher_kernel_98748807e30b6fdd(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_98748807e30b6fdd(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_98748807e30b6fdd(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_98748807e30b6fdd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v12_lead = item.get_local_id(0) % 16;
              if (v12_lead < 9) {
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 9; ++v14_i1) {
                  float v22_data = glb_m1[(v12_lead + (v14_i1 * 9))];
                  r0[v14_i1] = v22_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              if (v12_lead < 9) {
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
                  float v37_data = glb_m2[(v12_lead + (v29_i1 * 9))];
                  r1[v29_i1] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir2[9]{};
              if (v12_lead < 9) {
                float v45_data = r0[0];
                float v46_data = r1[0];
                float v49_data = ir2[0];
                ir2[0] = (v49_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 0))));
                float v52_data = r1[1];
                float v55_data = ir2[1];
                ir2[1] = (v55_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v52_data, 0))));
                float v58_data = r1[2];
                float v61_data = ir2[2];
                ir2[2] = (v61_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 0))));
                float v64_data = r1[3];
                float v67_data = ir2[3];
                ir2[3] = (v67_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 0))));
                float v70_data = r1[4];
                float v73_data = ir2[4];
                ir2[4] = (v73_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 0))));
                float v76_data = r1[5];
                float v79_data = ir2[5];
                ir2[5] = (v79_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 0))));
                float v82_data = r1[6];
                float v85_data = ir2[6];
                ir2[6] = (v85_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 0))));
                float v88_data = r1[7];
                float v91_data = ir2[7];
                ir2[7] = (v91_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 0))));
                float v94_data = r1[8];
                float v97_data = ir2[8];
                ir2[8] = (v97_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 0))));
              }
              if (v12_lead < 9) {
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
                float v152_data = r1[8];
                float v155_data = ir2[8];
                ir2[8] = (v155_data + (v103_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 1))));
              }
              if (v12_lead < 9) {
                float v161_data = r0[2];
                float v162_data = r1[0];
                float v165_data = ir2[0];
                ir2[0] = (v165_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 2))));
                float v168_data = r1[1];
                float v171_data = ir2[1];
                ir2[1] = (v171_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 2))));
                float v174_data = r1[2];
                float v177_data = ir2[2];
                ir2[2] = (v177_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 2))));
                float v180_data = r1[3];
                float v183_data = ir2[3];
                ir2[3] = (v183_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 2))));
                float v186_data = r1[4];
                float v189_data = ir2[4];
                ir2[4] = (v189_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v186_data, 2))));
                float v192_data = r1[5];
                float v195_data = ir2[5];
                ir2[5] = (v195_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v192_data, 2))));
                float v198_data = r1[6];
                float v201_data = ir2[6];
                ir2[6] = (v201_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v198_data, 2))));
                float v204_data = r1[7];
                float v207_data = ir2[7];
                ir2[7] = (v207_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v204_data, 2))));
                float v210_data = r1[8];
                float v213_data = ir2[8];
                ir2[8] = (v213_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v210_data, 2))));
              }
              if (v12_lead < 9) {
                float v219_data = r0[3];
                float v220_data = r1[0];
                float v223_data = ir2[0];
                ir2[0] = (v223_data + (v219_data * (sycl::group_broadcast(item.get_sub_group(), v220_data, 3))));
                float v226_data = r1[1];
                float v229_data = ir2[1];
                ir2[1] = (v229_data + (v219_data * (sycl::group_broadcast(item.get_sub_group(), v226_data, 3))));
                float v232_data = r1[2];
                float v235_data = ir2[2];
                ir2[2] = (v235_data + (v219_data * (sycl::group_broadcast(item.get_sub_group(), v232_data, 3))));
                float v238_data = r1[3];
                float v241_data = ir2[3];
                ir2[3] = (v241_data + (v219_data * (sycl::group_broadcast(item.get_sub_group(), v238_data, 3))));
                float v244_data = r1[4];
                float v247_data = ir2[4];
                ir2[4] = (v247_data + (v219_data * (sycl::group_broadcast(item.get_sub_group(), v244_data, 3))));
                float v250_data = r1[5];
                float v253_data = ir2[5];
                ir2[5] = (v253_data + (v219_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 3))));
                float v256_data = r1[6];
                float v259_data = ir2[6];
                ir2[6] = (v259_data + (v219_data * (sycl::group_broadcast(item.get_sub_group(), v256_data, 3))));
                float v262_data = r1[7];
                float v265_data = ir2[7];
                ir2[7] = (v265_data + (v219_data * (sycl::group_broadcast(item.get_sub_group(), v262_data, 3))));
                float v268_data = r1[8];
                float v271_data = ir2[8];
                ir2[8] = (v271_data + (v219_data * (sycl::group_broadcast(item.get_sub_group(), v268_data, 3))));
              }
              if (v12_lead < 9) {
                float v277_data = r0[4];
                float v278_data = r1[0];
                float v281_data = ir2[0];
                ir2[0] = (v281_data + (v277_data * (sycl::group_broadcast(item.get_sub_group(), v278_data, 4))));
                float v284_data = r1[1];
                float v287_data = ir2[1];
                ir2[1] = (v287_data + (v277_data * (sycl::group_broadcast(item.get_sub_group(), v284_data, 4))));
                float v290_data = r1[2];
                float v293_data = ir2[2];
                ir2[2] = (v293_data + (v277_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 4))));
                float v296_data = r1[3];
                float v299_data = ir2[3];
                ir2[3] = (v299_data + (v277_data * (sycl::group_broadcast(item.get_sub_group(), v296_data, 4))));
                float v302_data = r1[4];
                float v305_data = ir2[4];
                ir2[4] = (v305_data + (v277_data * (sycl::group_broadcast(item.get_sub_group(), v302_data, 4))));
                float v308_data = r1[5];
                float v311_data = ir2[5];
                ir2[5] = (v311_data + (v277_data * (sycl::group_broadcast(item.get_sub_group(), v308_data, 4))));
                float v314_data = r1[6];
                float v317_data = ir2[6];
                ir2[6] = (v317_data + (v277_data * (sycl::group_broadcast(item.get_sub_group(), v314_data, 4))));
                float v320_data = r1[7];
                float v323_data = ir2[7];
                ir2[7] = (v323_data + (v277_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 4))));
                float v326_data = r1[8];
                float v329_data = ir2[8];
                ir2[8] = (v329_data + (v277_data * (sycl::group_broadcast(item.get_sub_group(), v326_data, 4))));
              }
              if (v12_lead < 9) {
                float v335_data = r0[5];
                float v336_data = r1[0];
                float v339_data = ir2[0];
                ir2[0] = (v339_data + (v335_data * (sycl::group_broadcast(item.get_sub_group(), v336_data, 5))));
                float v342_data = r1[1];
                float v345_data = ir2[1];
                ir2[1] = (v345_data + (v335_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 5))));
                float v348_data = r1[2];
                float v351_data = ir2[2];
                ir2[2] = (v351_data + (v335_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 5))));
                float v354_data = r1[3];
                float v357_data = ir2[3];
                ir2[3] = (v357_data + (v335_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 5))));
                float v360_data = r1[4];
                float v363_data = ir2[4];
                ir2[4] = (v363_data + (v335_data * (sycl::group_broadcast(item.get_sub_group(), v360_data, 5))));
                float v366_data = r1[5];
                float v369_data = ir2[5];
                ir2[5] = (v369_data + (v335_data * (sycl::group_broadcast(item.get_sub_group(), v366_data, 5))));
                float v372_data = r1[6];
                float v375_data = ir2[6];
                ir2[6] = (v375_data + (v335_data * (sycl::group_broadcast(item.get_sub_group(), v372_data, 5))));
                float v378_data = r1[7];
                float v381_data = ir2[7];
                ir2[7] = (v381_data + (v335_data * (sycl::group_broadcast(item.get_sub_group(), v378_data, 5))));
                float v384_data = r1[8];
                float v387_data = ir2[8];
                ir2[8] = (v387_data + (v335_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 5))));
              }
              if (v12_lead < 9) {
                float v393_data = r0[6];
                float v394_data = r1[0];
                float v397_data = ir2[0];
                ir2[0] = (v397_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v394_data, 6))));
                float v400_data = r1[1];
                float v403_data = ir2[1];
                ir2[1] = (v403_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v400_data, 6))));
                float v406_data = r1[2];
                float v409_data = ir2[2];
                ir2[2] = (v409_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v406_data, 6))));
                float v412_data = r1[3];
                float v415_data = ir2[3];
                ir2[3] = (v415_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v412_data, 6))));
                float v418_data = r1[4];
                float v421_data = ir2[4];
                ir2[4] = (v421_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v418_data, 6))));
                float v424_data = r1[5];
                float v427_data = ir2[5];
                ir2[5] = (v427_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v424_data, 6))));
                float v430_data = r1[6];
                float v433_data = ir2[6];
                ir2[6] = (v433_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v430_data, 6))));
                float v436_data = r1[7];
                float v439_data = ir2[7];
                ir2[7] = (v439_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v436_data, 6))));
                float v442_data = r1[8];
                float v445_data = ir2[8];
                ir2[8] = (v445_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 6))));
              }
              if (v12_lead < 9) {
                float v451_data = r0[7];
                float v452_data = r1[0];
                float v455_data = ir2[0];
                ir2[0] = (v455_data + (v451_data * (sycl::group_broadcast(item.get_sub_group(), v452_data, 7))));
                float v458_data = r1[1];
                float v461_data = ir2[1];
                ir2[1] = (v461_data + (v451_data * (sycl::group_broadcast(item.get_sub_group(), v458_data, 7))));
                float v464_data = r1[2];
                float v467_data = ir2[2];
                ir2[2] = (v467_data + (v451_data * (sycl::group_broadcast(item.get_sub_group(), v464_data, 7))));
                float v470_data = r1[3];
                float v473_data = ir2[3];
                ir2[3] = (v473_data + (v451_data * (sycl::group_broadcast(item.get_sub_group(), v470_data, 7))));
                float v476_data = r1[4];
                float v479_data = ir2[4];
                ir2[4] = (v479_data + (v451_data * (sycl::group_broadcast(item.get_sub_group(), v476_data, 7))));
                float v482_data = r1[5];
                float v485_data = ir2[5];
                ir2[5] = (v485_data + (v451_data * (sycl::group_broadcast(item.get_sub_group(), v482_data, 7))));
                float v488_data = r1[6];
                float v491_data = ir2[6];
                ir2[6] = (v491_data + (v451_data * (sycl::group_broadcast(item.get_sub_group(), v488_data, 7))));
                float v494_data = r1[7];
                float v497_data = ir2[7];
                ir2[7] = (v497_data + (v451_data * (sycl::group_broadcast(item.get_sub_group(), v494_data, 7))));
                float v500_data = r1[8];
                float v503_data = ir2[8];
                ir2[8] = (v503_data + (v451_data * (sycl::group_broadcast(item.get_sub_group(), v500_data, 7))));
              }
              if (v12_lead < 9) {
                float v509_data = r0[8];
                float v510_data = r1[0];
                float v513_data = ir2[0];
                ir2[0] = (v513_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v510_data, 8))));
                float v516_data = r1[1];
                float v519_data = ir2[1];
                ir2[1] = (v519_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v516_data, 8))));
                float v522_data = r1[2];
                float v525_data = ir2[2];
                ir2[2] = (v525_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v522_data, 8))));
                float v528_data = r1[3];
                float v531_data = ir2[3];
                ir2[3] = (v531_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v528_data, 8))));
                float v534_data = r1[4];
                float v537_data = ir2[4];
                ir2[4] = (v537_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v534_data, 8))));
                float v540_data = r1[5];
                float v543_data = ir2[5];
                ir2[5] = (v543_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v540_data, 8))));
                float v546_data = r1[6];
                float v549_data = ir2[6];
                ir2[6] = (v549_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 8))));
                float v552_data = r1[7];
                float v555_data = ir2[7];
                ir2[7] = (v555_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v552_data, 8))));
                float v558_data = r1[8];
                float v561_data = ir2[8];
                ir2[8] = (v561_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v558_data, 8))));
              }
              if (v12_lead < 9) {
                #pragma unroll
                for (int32_t v568_n1 = 0; v568_n1 < 9; ++v568_n1) {
                  float v570_data = ir2[v568_n1];
                  r2[v568_n1] = (v570_data * 13.0f);
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v12_lead < 9) {
                #pragma unroll
                for (int32_t v577_i1 = 0; v577_i1 < 9; ++v577_i1) {
                  float v579_data = r2[v577_i1];
                  glb_m0[(v12_lead + (v577_i1 * 9))] = v579_data;
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


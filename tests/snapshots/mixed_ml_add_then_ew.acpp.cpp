// === base name ===
kernel_609dd06e89

// === header ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_609dd06e89(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_609dd06e89(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // m4 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(TMP)
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[batchId0 * 64 + 0 + m4_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v10_lead = item.get_local_id(0) % 16;
              bool v11_g = v10_lead < 8;
              if (v11_g) {
                #pragma unroll
                for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
                  float v20_data = glb_m0[(v10_lead + (v12_i1 * 8))];
                  r0[v12_i1] = v20_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v11_g) {
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
                  float v35_data = glb_m1[(v10_lead + (v27_i1 * 8))];
                  r1[v27_i1] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[8]{};
              // r3 = load{g>r}(glb_m2);
              if (v11_g) {
                #pragma unroll
                for (int32_t v42_i1 = 0; v42_i1 < 8; ++v42_i1) {
                  float v50_data = glb_m2[(v10_lead + (v42_i1 * 8))];
                  r3[v42_i1] = v50_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              if (v11_g) {
                float v57_data = r0[0];
                float v58_data = r1[0];
                float v61_data = r2[0];
                r2[0] = (v61_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 0))));
                float v64_data = r1[1];
                float v67_data = r2[1];
                r2[1] = (v67_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 0))));
                float v70_data = r1[2];
                float v73_data = r2[2];
                r2[2] = (v73_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 0))));
                float v76_data = r1[3];
                float v79_data = r2[3];
                r2[3] = (v79_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 0))));
                float v82_data = r1[4];
                float v85_data = r2[4];
                r2[4] = (v85_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 0))));
                float v88_data = r1[5];
                float v91_data = r2[5];
                r2[5] = (v91_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 0))));
                float v94_data = r1[6];
                float v97_data = r2[6];
                r2[6] = (v97_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 0))));
                float v100_data = r1[7];
                float v103_data = r2[7];
                r2[7] = (v103_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 0))));
              }
              if (v11_g) {
                float v109_data = r0[1];
                float v110_data = r1[0];
                float v113_data = r2[0];
                r2[0] = (v113_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 1))));
                float v116_data = r1[1];
                float v119_data = r2[1];
                r2[1] = (v119_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 1))));
                float v122_data = r1[2];
                float v125_data = r2[2];
                r2[2] = (v125_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 1))));
                float v128_data = r1[3];
                float v131_data = r2[3];
                r2[3] = (v131_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 1))));
                float v134_data = r1[4];
                float v137_data = r2[4];
                r2[4] = (v137_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 1))));
                float v140_data = r1[5];
                float v143_data = r2[5];
                r2[5] = (v143_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 1))));
                float v146_data = r1[6];
                float v149_data = r2[6];
                r2[6] = (v149_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 1))));
                float v152_data = r1[7];
                float v155_data = r2[7];
                r2[7] = (v155_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 1))));
              }
              if (v11_g) {
                float v161_data = r0[2];
                float v162_data = r1[0];
                float v165_data = r2[0];
                r2[0] = (v165_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 2))));
                float v168_data = r1[1];
                float v171_data = r2[1];
                r2[1] = (v171_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 2))));
                float v174_data = r1[2];
                float v177_data = r2[2];
                r2[2] = (v177_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 2))));
                float v180_data = r1[3];
                float v183_data = r2[3];
                r2[3] = (v183_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 2))));
                float v186_data = r1[4];
                float v189_data = r2[4];
                r2[4] = (v189_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v186_data, 2))));
                float v192_data = r1[5];
                float v195_data = r2[5];
                r2[5] = (v195_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v192_data, 2))));
                float v198_data = r1[6];
                float v201_data = r2[6];
                r2[6] = (v201_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v198_data, 2))));
                float v204_data = r1[7];
                float v207_data = r2[7];
                r2[7] = (v207_data + (v161_data * (sycl::group_broadcast(item.get_sub_group(), v204_data, 2))));
              }
              if (v11_g) {
                float v213_data = r0[3];
                float v214_data = r1[0];
                float v217_data = r2[0];
                r2[0] = (v217_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v214_data, 3))));
                float v220_data = r1[1];
                float v223_data = r2[1];
                r2[1] = (v223_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v220_data, 3))));
                float v226_data = r1[2];
                float v229_data = r2[2];
                r2[2] = (v229_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v226_data, 3))));
                float v232_data = r1[3];
                float v235_data = r2[3];
                r2[3] = (v235_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v232_data, 3))));
                float v238_data = r1[4];
                float v241_data = r2[4];
                r2[4] = (v241_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v238_data, 3))));
                float v244_data = r1[5];
                float v247_data = r2[5];
                r2[5] = (v247_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v244_data, 3))));
                float v250_data = r1[6];
                float v253_data = r2[6];
                r2[6] = (v253_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v250_data, 3))));
                float v256_data = r1[7];
                float v259_data = r2[7];
                r2[7] = (v259_data + (v213_data * (sycl::group_broadcast(item.get_sub_group(), v256_data, 3))));
              }
              if (v11_g) {
                float v265_data = r0[4];
                float v266_data = r1[0];
                float v269_data = r2[0];
                r2[0] = (v269_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v266_data, 4))));
                float v272_data = r1[1];
                float v275_data = r2[1];
                r2[1] = (v275_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v272_data, 4))));
                float v278_data = r1[2];
                float v281_data = r2[2];
                r2[2] = (v281_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v278_data, 4))));
                float v284_data = r1[3];
                float v287_data = r2[3];
                r2[3] = (v287_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v284_data, 4))));
                float v290_data = r1[4];
                float v293_data = r2[4];
                r2[4] = (v293_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 4))));
                float v296_data = r1[5];
                float v299_data = r2[5];
                r2[5] = (v299_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v296_data, 4))));
                float v302_data = r1[6];
                float v305_data = r2[6];
                r2[6] = (v305_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v302_data, 4))));
                float v308_data = r1[7];
                float v311_data = r2[7];
                r2[7] = (v311_data + (v265_data * (sycl::group_broadcast(item.get_sub_group(), v308_data, 4))));
              }
              if (v11_g) {
                float v317_data = r0[5];
                float v318_data = r1[0];
                float v321_data = r2[0];
                r2[0] = (v321_data + (v317_data * (sycl::group_broadcast(item.get_sub_group(), v318_data, 5))));
                float v324_data = r1[1];
                float v327_data = r2[1];
                r2[1] = (v327_data + (v317_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 5))));
                float v330_data = r1[2];
                float v333_data = r2[2];
                r2[2] = (v333_data + (v317_data * (sycl::group_broadcast(item.get_sub_group(), v330_data, 5))));
                float v336_data = r1[3];
                float v339_data = r2[3];
                r2[3] = (v339_data + (v317_data * (sycl::group_broadcast(item.get_sub_group(), v336_data, 5))));
                float v342_data = r1[4];
                float v345_data = r2[4];
                r2[4] = (v345_data + (v317_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 5))));
                float v348_data = r1[5];
                float v351_data = r2[5];
                r2[5] = (v351_data + (v317_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 5))));
                float v354_data = r1[6];
                float v357_data = r2[6];
                r2[6] = (v357_data + (v317_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 5))));
                float v360_data = r1[7];
                float v363_data = r2[7];
                r2[7] = (v363_data + (v317_data * (sycl::group_broadcast(item.get_sub_group(), v360_data, 5))));
              }
              if (v11_g) {
                float v369_data = r0[6];
                float v370_data = r1[0];
                float v373_data = r2[0];
                r2[0] = (v373_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v370_data, 6))));
                float v376_data = r1[1];
                float v379_data = r2[1];
                r2[1] = (v379_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v376_data, 6))));
                float v382_data = r1[2];
                float v385_data = r2[2];
                r2[2] = (v385_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v382_data, 6))));
                float v388_data = r1[3];
                float v391_data = r2[3];
                r2[3] = (v391_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v388_data, 6))));
                float v394_data = r1[4];
                float v397_data = r2[4];
                r2[4] = (v397_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v394_data, 6))));
                float v400_data = r1[5];
                float v403_data = r2[5];
                r2[5] = (v403_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v400_data, 6))));
                float v406_data = r1[6];
                float v409_data = r2[6];
                r2[6] = (v409_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v406_data, 6))));
                float v412_data = r1[7];
                float v415_data = r2[7];
                r2[7] = (v415_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v412_data, 6))));
              }
              if (v11_g) {
                float v421_data = r0[7];
                float v422_data = r1[0];
                float v425_data = r2[0];
                r2[0] = (v425_data + (v421_data * (sycl::group_broadcast(item.get_sub_group(), v422_data, 7))));
                float v428_data = r1[1];
                float v431_data = r2[1];
                r2[1] = (v431_data + (v421_data * (sycl::group_broadcast(item.get_sub_group(), v428_data, 7))));
                float v434_data = r1[2];
                float v437_data = r2[2];
                r2[2] = (v437_data + (v421_data * (sycl::group_broadcast(item.get_sub_group(), v434_data, 7))));
                float v440_data = r1[3];
                float v443_data = r2[3];
                r2[3] = (v443_data + (v421_data * (sycl::group_broadcast(item.get_sub_group(), v440_data, 7))));
                float v446_data = r1[4];
                float v449_data = r2[4];
                r2[4] = (v449_data + (v421_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 7))));
                float v452_data = r1[5];
                float v455_data = r2[5];
                r2[5] = (v455_data + (v421_data * (sycl::group_broadcast(item.get_sub_group(), v452_data, 7))));
                float v458_data = r1[6];
                float v461_data = r2[6];
                r2[6] = (v461_data + (v421_data * (sycl::group_broadcast(item.get_sub_group(), v458_data, 7))));
                float v464_data = r1[7];
                float v467_data = r2[7];
                r2[7] = (v467_data + (v421_data * (sycl::group_broadcast(item.get_sub_group(), v464_data, 7))));
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m3);
              if (v11_g) {
                #pragma unroll
                for (int32_t v474_i1 = 0; v474_i1 < 8; ++v474_i1) {
                  float v482_data = glb_m3[(v10_lead + (v474_i1 * 8))];
                  r4[v474_i1] = v482_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              // wait(r4 = load{g>r}(glb_m3););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir5[8]{};
              if (v11_g) {
                float v490_data = r3[0];
                float v491_data = r4[0];
                float v494_data = ir5[0];
                ir5[0] = (v494_data + (v490_data * (sycl::group_broadcast(item.get_sub_group(), v491_data, 0))));
                float v497_data = r4[1];
                float v500_data = ir5[1];
                ir5[1] = (v500_data + (v490_data * (sycl::group_broadcast(item.get_sub_group(), v497_data, 0))));
                float v503_data = r4[2];
                float v506_data = ir5[2];
                ir5[2] = (v506_data + (v490_data * (sycl::group_broadcast(item.get_sub_group(), v503_data, 0))));
                float v509_data = r4[3];
                float v512_data = ir5[3];
                ir5[3] = (v512_data + (v490_data * (sycl::group_broadcast(item.get_sub_group(), v509_data, 0))));
                float v515_data = r4[4];
                float v518_data = ir5[4];
                ir5[4] = (v518_data + (v490_data * (sycl::group_broadcast(item.get_sub_group(), v515_data, 0))));
                float v521_data = r4[5];
                float v524_data = ir5[5];
                ir5[5] = (v524_data + (v490_data * (sycl::group_broadcast(item.get_sub_group(), v521_data, 0))));
                float v527_data = r4[6];
                float v530_data = ir5[6];
                ir5[6] = (v530_data + (v490_data * (sycl::group_broadcast(item.get_sub_group(), v527_data, 0))));
                float v533_data = r4[7];
                float v536_data = ir5[7];
                ir5[7] = (v536_data + (v490_data * (sycl::group_broadcast(item.get_sub_group(), v533_data, 0))));
              }
              if (v11_g) {
                float v542_data = r3[1];
                float v543_data = r4[0];
                float v546_data = ir5[0];
                ir5[0] = (v546_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v543_data, 1))));
                float v549_data = r4[1];
                float v552_data = ir5[1];
                ir5[1] = (v552_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v549_data, 1))));
                float v555_data = r4[2];
                float v558_data = ir5[2];
                ir5[2] = (v558_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v555_data, 1))));
                float v561_data = r4[3];
                float v564_data = ir5[3];
                ir5[3] = (v564_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v561_data, 1))));
                float v567_data = r4[4];
                float v570_data = ir5[4];
                ir5[4] = (v570_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v567_data, 1))));
                float v573_data = r4[5];
                float v576_data = ir5[5];
                ir5[5] = (v576_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v573_data, 1))));
                float v579_data = r4[6];
                float v582_data = ir5[6];
                ir5[6] = (v582_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v579_data, 1))));
                float v585_data = r4[7];
                float v588_data = ir5[7];
                ir5[7] = (v588_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v585_data, 1))));
              }
              if (v11_g) {
                float v594_data = r3[2];
                float v595_data = r4[0];
                float v598_data = ir5[0];
                ir5[0] = (v598_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v595_data, 2))));
                float v601_data = r4[1];
                float v604_data = ir5[1];
                ir5[1] = (v604_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v601_data, 2))));
                float v607_data = r4[2];
                float v610_data = ir5[2];
                ir5[2] = (v610_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v607_data, 2))));
                float v613_data = r4[3];
                float v616_data = ir5[3];
                ir5[3] = (v616_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v613_data, 2))));
                float v619_data = r4[4];
                float v622_data = ir5[4];
                ir5[4] = (v622_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v619_data, 2))));
                float v625_data = r4[5];
                float v628_data = ir5[5];
                ir5[5] = (v628_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v625_data, 2))));
                float v631_data = r4[6];
                float v634_data = ir5[6];
                ir5[6] = (v634_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v631_data, 2))));
                float v637_data = r4[7];
                float v640_data = ir5[7];
                ir5[7] = (v640_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v637_data, 2))));
              }
              if (v11_g) {
                float v646_data = r3[3];
                float v647_data = r4[0];
                float v650_data = ir5[0];
                ir5[0] = (v650_data + (v646_data * (sycl::group_broadcast(item.get_sub_group(), v647_data, 3))));
                float v653_data = r4[1];
                float v656_data = ir5[1];
                ir5[1] = (v656_data + (v646_data * (sycl::group_broadcast(item.get_sub_group(), v653_data, 3))));
                float v659_data = r4[2];
                float v662_data = ir5[2];
                ir5[2] = (v662_data + (v646_data * (sycl::group_broadcast(item.get_sub_group(), v659_data, 3))));
                float v665_data = r4[3];
                float v668_data = ir5[3];
                ir5[3] = (v668_data + (v646_data * (sycl::group_broadcast(item.get_sub_group(), v665_data, 3))));
                float v671_data = r4[4];
                float v674_data = ir5[4];
                ir5[4] = (v674_data + (v646_data * (sycl::group_broadcast(item.get_sub_group(), v671_data, 3))));
                float v677_data = r4[5];
                float v680_data = ir5[5];
                ir5[5] = (v680_data + (v646_data * (sycl::group_broadcast(item.get_sub_group(), v677_data, 3))));
                float v683_data = r4[6];
                float v686_data = ir5[6];
                ir5[6] = (v686_data + (v646_data * (sycl::group_broadcast(item.get_sub_group(), v683_data, 3))));
                float v689_data = r4[7];
                float v692_data = ir5[7];
                ir5[7] = (v692_data + (v646_data * (sycl::group_broadcast(item.get_sub_group(), v689_data, 3))));
              }
              if (v11_g) {
                float v698_data = r3[4];
                float v699_data = r4[0];
                float v702_data = ir5[0];
                ir5[0] = (v702_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v699_data, 4))));
                float v705_data = r4[1];
                float v708_data = ir5[1];
                ir5[1] = (v708_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v705_data, 4))));
                float v711_data = r4[2];
                float v714_data = ir5[2];
                ir5[2] = (v714_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v711_data, 4))));
                float v717_data = r4[3];
                float v720_data = ir5[3];
                ir5[3] = (v720_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v717_data, 4))));
                float v723_data = r4[4];
                float v726_data = ir5[4];
                ir5[4] = (v726_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v723_data, 4))));
                float v729_data = r4[5];
                float v732_data = ir5[5];
                ir5[5] = (v732_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v729_data, 4))));
                float v735_data = r4[6];
                float v738_data = ir5[6];
                ir5[6] = (v738_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v735_data, 4))));
                float v741_data = r4[7];
                float v744_data = ir5[7];
                ir5[7] = (v744_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v741_data, 4))));
              }
              if (v11_g) {
                float v750_data = r3[5];
                float v751_data = r4[0];
                float v754_data = ir5[0];
                ir5[0] = (v754_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v751_data, 5))));
                float v757_data = r4[1];
                float v760_data = ir5[1];
                ir5[1] = (v760_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v757_data, 5))));
                float v763_data = r4[2];
                float v766_data = ir5[2];
                ir5[2] = (v766_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v763_data, 5))));
                float v769_data = r4[3];
                float v772_data = ir5[3];
                ir5[3] = (v772_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v769_data, 5))));
                float v775_data = r4[4];
                float v778_data = ir5[4];
                ir5[4] = (v778_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v775_data, 5))));
                float v781_data = r4[5];
                float v784_data = ir5[5];
                ir5[5] = (v784_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v781_data, 5))));
                float v787_data = r4[6];
                float v790_data = ir5[6];
                ir5[6] = (v790_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v787_data, 5))));
                float v793_data = r4[7];
                float v796_data = ir5[7];
                ir5[7] = (v796_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v793_data, 5))));
              }
              if (v11_g) {
                float v802_data = r3[6];
                float v803_data = r4[0];
                float v806_data = ir5[0];
                ir5[0] = (v806_data + (v802_data * (sycl::group_broadcast(item.get_sub_group(), v803_data, 6))));
                float v809_data = r4[1];
                float v812_data = ir5[1];
                ir5[1] = (v812_data + (v802_data * (sycl::group_broadcast(item.get_sub_group(), v809_data, 6))));
                float v815_data = r4[2];
                float v818_data = ir5[2];
                ir5[2] = (v818_data + (v802_data * (sycl::group_broadcast(item.get_sub_group(), v815_data, 6))));
                float v821_data = r4[3];
                float v824_data = ir5[3];
                ir5[3] = (v824_data + (v802_data * (sycl::group_broadcast(item.get_sub_group(), v821_data, 6))));
                float v827_data = r4[4];
                float v830_data = ir5[4];
                ir5[4] = (v830_data + (v802_data * (sycl::group_broadcast(item.get_sub_group(), v827_data, 6))));
                float v833_data = r4[5];
                float v836_data = ir5[5];
                ir5[5] = (v836_data + (v802_data * (sycl::group_broadcast(item.get_sub_group(), v833_data, 6))));
                float v839_data = r4[6];
                float v842_data = ir5[6];
                ir5[6] = (v842_data + (v802_data * (sycl::group_broadcast(item.get_sub_group(), v839_data, 6))));
                float v845_data = r4[7];
                float v848_data = ir5[7];
                ir5[7] = (v848_data + (v802_data * (sycl::group_broadcast(item.get_sub_group(), v845_data, 6))));
              }
              if (v11_g) {
                float v854_data = r3[7];
                float v855_data = r4[0];
                float v858_data = ir5[0];
                ir5[0] = (v858_data + (v854_data * (sycl::group_broadcast(item.get_sub_group(), v855_data, 7))));
                float v861_data = r4[1];
                float v864_data = ir5[1];
                ir5[1] = (v864_data + (v854_data * (sycl::group_broadcast(item.get_sub_group(), v861_data, 7))));
                float v867_data = r4[2];
                float v870_data = ir5[2];
                ir5[2] = (v870_data + (v854_data * (sycl::group_broadcast(item.get_sub_group(), v867_data, 7))));
                float v873_data = r4[3];
                float v876_data = ir5[3];
                ir5[3] = (v876_data + (v854_data * (sycl::group_broadcast(item.get_sub_group(), v873_data, 7))));
                float v879_data = r4[4];
                float v882_data = ir5[4];
                ir5[4] = (v882_data + (v854_data * (sycl::group_broadcast(item.get_sub_group(), v879_data, 7))));
                float v885_data = r4[5];
                float v888_data = ir5[5];
                ir5[5] = (v888_data + (v854_data * (sycl::group_broadcast(item.get_sub_group(), v885_data, 7))));
                float v891_data = r4[6];
                float v894_data = ir5[6];
                ir5[6] = (v894_data + (v854_data * (sycl::group_broadcast(item.get_sub_group(), v891_data, 7))));
                float v897_data = r4[7];
                float v900_data = ir5[7];
                ir5[7] = (v900_data + (v854_data * (sycl::group_broadcast(item.get_sub_group(), v897_data, 7))));
              }
              if (v11_g) {
                #pragma unroll
                for (int32_t v906_n1 = 0; v906_n1 < 8; ++v906_n1) {
                  float v908_data = ir5[v906_n1];
                  float v910_data = r2[v906_n1];
                  r5[v906_n1] = (v910_data + v908_data);
                }
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = store{r>s}(localShrMem0, r5);
              if (v11_g) {
                #pragma unroll
                for (int32_t v918_i1 = 0; v918_i1 < 8; ++v918_i1) {
                  float v920_data = r5[v918_i1];
                  int32_t v927_a = v10_lead + (v918_i1 * 8);
                  s0[(v927_a ^ ((v927_a >> 5) & 31))] = v920_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m4 = abs(s0)
              if (v11_g) {
                #pragma unroll
                for (int32_t v935_k1 = 0; v935_k1 < 8; ++v935_k1) {
                  int32_t v942_a = v10_lead + (v935_k1 * 8);
                  float v946_data = s0[(v942_a ^ ((v942_a >> 5) & 31))];
                  glb_m4[v942_a] = (sycl::fabs(v946_data));
                }
              }
            }
          }
        }
      });
    }
  });
}


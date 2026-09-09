// === base name ===
kernel_fe1d67c50c105e21

// === header ===
void launcher_kernel_fe1d67c50c105e21(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_fe1d67c50c105e21(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_fe1d67c50c105e21(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_fe1d67c50c105e21(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×32(12×6) {0..12}×{0..6} strided
        // m1 32×32(6×6) {0..6}×{0..6} strided
        // m2 32×32(12×6) {0..12}×{0..6} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
        // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
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
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              float r0[6]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v13_lead = item.get_local_id(0) % 16;
              if (v13_lead < 12) {
                #pragma unroll
                for (int32_t v15_i1 = 0; v15_i1 < 6; ++v15_i1) {
                  float v23_data = glb_m0[(v13_lead + (v15_i1 * 12))];
                  r0[v15_i1] = v23_data;
                }
              }
              float r1[6]{};
              // r1 = load{g>r}(glb_m1);
              if (v13_lead < 6) {
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 6; ++v30_i1) {
                  float v38_data = glb_m1[(v13_lead + (v30_i1 * 6))];
                  r1[v30_i1] = v38_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v13_lead < 12) {
                #pragma unroll
                for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
                  float v53_data = glb_m3[(v13_lead + (v45_i1 * 12))];
                  r3[v45_i1] = v53_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              if (v13_lead < 12) {
                float v60_data = r0[0];
                float v61_data = r1[0];
                float v64_data = r2[0];
                r2[0] = (v64_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
                float v67_data = r1[1];
                float v70_data = r2[1];
                r2[1] = (v70_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
                float v73_data = r1[2];
                float v76_data = r2[2];
                r2[2] = (v76_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
                float v79_data = r1[3];
                float v82_data = r2[3];
                r2[3] = (v82_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
                float v85_data = r1[4];
                float v88_data = r2[4];
                r2[4] = (v88_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
                float v91_data = r1[5];
                float v94_data = r2[5];
                r2[5] = (v94_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 0))));
              }
              if (v13_lead < 12) {
                float v100_data = r0[1];
                float v101_data = r1[0];
                float v104_data = r2[0];
                r2[0] = (v104_data + (v100_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 1))));
                float v107_data = r1[1];
                float v110_data = r2[1];
                r2[1] = (v110_data + (v100_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 1))));
                float v113_data = r1[2];
                float v116_data = r2[2];
                r2[2] = (v116_data + (v100_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 1))));
                float v119_data = r1[3];
                float v122_data = r2[3];
                r2[3] = (v122_data + (v100_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 1))));
                float v125_data = r1[4];
                float v128_data = r2[4];
                r2[4] = (v128_data + (v100_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 1))));
                float v131_data = r1[5];
                float v134_data = r2[5];
                r2[5] = (v134_data + (v100_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 1))));
              }
              if (v13_lead < 12) {
                float v140_data = r0[2];
                float v141_data = r1[0];
                float v144_data = r2[0];
                r2[0] = (v144_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 2))));
                float v147_data = r1[1];
                float v150_data = r2[1];
                r2[1] = (v150_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 2))));
                float v153_data = r1[2];
                float v156_data = r2[2];
                r2[2] = (v156_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 2))));
                float v159_data = r1[3];
                float v162_data = r2[3];
                r2[3] = (v162_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 2))));
                float v165_data = r1[4];
                float v168_data = r2[4];
                r2[4] = (v168_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 2))));
                float v171_data = r1[5];
                float v174_data = r2[5];
                r2[5] = (v174_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
              }
              if (v13_lead < 12) {
                float v180_data = r0[3];
                float v181_data = r1[0];
                float v184_data = r2[0];
                r2[0] = (v184_data + (v180_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 3))));
                float v187_data = r1[1];
                float v190_data = r2[1];
                r2[1] = (v190_data + (v180_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 3))));
                float v193_data = r1[2];
                float v196_data = r2[2];
                r2[2] = (v196_data + (v180_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 3))));
                float v199_data = r1[3];
                float v202_data = r2[3];
                r2[3] = (v202_data + (v180_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 3))));
                float v205_data = r1[4];
                float v208_data = r2[4];
                r2[4] = (v208_data + (v180_data * (sycl::group_broadcast(item.get_sub_group(), v205_data, 3))));
                float v211_data = r1[5];
                float v214_data = r2[5];
                r2[5] = (v214_data + (v180_data * (sycl::group_broadcast(item.get_sub_group(), v211_data, 3))));
              }
              if (v13_lead < 12) {
                float v220_data = r0[4];
                float v221_data = r1[0];
                float v224_data = r2[0];
                r2[0] = (v224_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v221_data, 4))));
                float v227_data = r1[1];
                float v230_data = r2[1];
                r2[1] = (v230_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v227_data, 4))));
                float v233_data = r1[2];
                float v236_data = r2[2];
                r2[2] = (v236_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v233_data, 4))));
                float v239_data = r1[3];
                float v242_data = r2[3];
                r2[3] = (v242_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 4))));
                float v245_data = r1[4];
                float v248_data = r2[4];
                r2[4] = (v248_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 4))));
                float v251_data = r1[5];
                float v254_data = r2[5];
                r2[5] = (v254_data + (v220_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 4))));
              }
              if (v13_lead < 12) {
                float v260_data = r0[5];
                float v261_data = r1[0];
                float v264_data = r2[0];
                r2[0] = (v264_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v261_data, 5))));
                float v267_data = r1[1];
                float v270_data = r2[1];
                r2[1] = (v270_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v267_data, 5))));
                float v273_data = r1[2];
                float v276_data = r2[2];
                r2[2] = (v276_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v273_data, 5))));
                float v279_data = r1[3];
                float v282_data = r2[3];
                r2[3] = (v282_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 5))));
                float v285_data = r1[4];
                float v288_data = r2[4];
                r2[4] = (v288_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v285_data, 5))));
                float v291_data = r1[5];
                float v294_data = r2[5];
                r2[5] = (v294_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 5))));
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r4[6]{};
              // r4 = +(r3 * r2) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir4[6]{};
              if (v13_lead < 12) {
                float v302_data = r3[0];
                float v303_data = r2[0];
                float v306_data = ir4[0];
                ir4[0] = (v306_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v303_data, 0))));
                float v309_data = r2[1];
                float v312_data = ir4[1];
                ir4[1] = (v312_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 0))));
                float v315_data = r2[2];
                float v318_data = ir4[2];
                ir4[2] = (v318_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 0))));
                float v321_data = r2[3];
                float v324_data = ir4[3];
                ir4[3] = (v324_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v321_data, 0))));
                float v327_data = r2[4];
                float v330_data = ir4[4];
                ir4[4] = (v330_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 0))));
                float v333_data = r2[5];
                float v336_data = ir4[5];
                ir4[5] = (v336_data + (v302_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 0))));
              }
              if (v13_lead < 12) {
                float v342_data = r3[1];
                float v343_data = r2[0];
                float v346_data = ir4[0];
                ir4[0] = (v346_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 1))));
                float v349_data = r2[1];
                float v352_data = ir4[1];
                ir4[1] = (v352_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 1))));
                float v355_data = r2[2];
                float v358_data = ir4[2];
                ir4[2] = (v358_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 1))));
                float v361_data = r2[3];
                float v364_data = ir4[3];
                ir4[3] = (v364_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 1))));
                float v367_data = r2[4];
                float v370_data = ir4[4];
                ir4[4] = (v370_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 1))));
                float v373_data = r2[5];
                float v376_data = ir4[5];
                ir4[5] = (v376_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 1))));
              }
              if (v13_lead < 12) {
                float v382_data = r3[2];
                float v383_data = r2[0];
                float v386_data = ir4[0];
                ir4[0] = (v386_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v383_data, 2))));
                float v389_data = r2[1];
                float v392_data = ir4[1];
                ir4[1] = (v392_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 2))));
                float v395_data = r2[2];
                float v398_data = ir4[2];
                ir4[2] = (v398_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 2))));
                float v401_data = r2[3];
                float v404_data = ir4[3];
                ir4[3] = (v404_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 2))));
                float v407_data = r2[4];
                float v410_data = ir4[4];
                ir4[4] = (v410_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 2))));
                float v413_data = r2[5];
                float v416_data = ir4[5];
                ir4[5] = (v416_data + (v382_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 2))));
              }
              if (v13_lead < 12) {
                float v422_data = r3[3];
                float v423_data = r2[0];
                float v426_data = ir4[0];
                ir4[0] = (v426_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v423_data, 3))));
                float v429_data = r2[1];
                float v432_data = ir4[1];
                ir4[1] = (v432_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v429_data, 3))));
                float v435_data = r2[2];
                float v438_data = ir4[2];
                ir4[2] = (v438_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 3))));
                float v441_data = r2[3];
                float v444_data = ir4[3];
                ir4[3] = (v444_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v441_data, 3))));
                float v447_data = r2[4];
                float v450_data = ir4[4];
                ir4[4] = (v450_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 3))));
                float v453_data = r2[5];
                float v456_data = ir4[5];
                ir4[5] = (v456_data + (v422_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 3))));
              }
              if (v13_lead < 12) {
                float v462_data = r3[4];
                float v463_data = r2[0];
                float v466_data = ir4[0];
                ir4[0] = (v466_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v463_data, 4))));
                float v469_data = r2[1];
                float v472_data = ir4[1];
                ir4[1] = (v472_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v469_data, 4))));
                float v475_data = r2[2];
                float v478_data = ir4[2];
                ir4[2] = (v478_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v475_data, 4))));
                float v481_data = r2[3];
                float v484_data = ir4[3];
                ir4[3] = (v484_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v481_data, 4))));
                float v487_data = r2[4];
                float v490_data = ir4[4];
                ir4[4] = (v490_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v487_data, 4))));
                float v493_data = r2[5];
                float v496_data = ir4[5];
                ir4[5] = (v496_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v493_data, 4))));
              }
              if (v13_lead < 12) {
                float v502_data = r3[5];
                float v503_data = r2[0];
                float v506_data = ir4[0];
                ir4[0] = (v506_data + (v502_data * (sycl::group_broadcast(item.get_sub_group(), v503_data, 5))));
                float v509_data = r2[1];
                float v512_data = ir4[1];
                ir4[1] = (v512_data + (v502_data * (sycl::group_broadcast(item.get_sub_group(), v509_data, 5))));
                float v515_data = r2[2];
                float v518_data = ir4[2];
                ir4[2] = (v518_data + (v502_data * (sycl::group_broadcast(item.get_sub_group(), v515_data, 5))));
                float v521_data = r2[3];
                float v524_data = ir4[3];
                ir4[3] = (v524_data + (v502_data * (sycl::group_broadcast(item.get_sub_group(), v521_data, 5))));
                float v527_data = r2[4];
                float v530_data = ir4[4];
                ir4[4] = (v530_data + (v502_data * (sycl::group_broadcast(item.get_sub_group(), v527_data, 5))));
                float v533_data = r2[5];
                float v536_data = ir4[5];
                ir4[5] = (v536_data + (v502_data * (sycl::group_broadcast(item.get_sub_group(), v533_data, 5))));
              }
              if (v13_lead < 12) {
                float v542_data = r3[6];
                float v543_data = r2[0];
                float v546_data = ir4[0];
                ir4[0] = (v546_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v543_data, 6))));
                float v549_data = r2[1];
                float v552_data = ir4[1];
                ir4[1] = (v552_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v549_data, 6))));
                float v555_data = r2[2];
                float v558_data = ir4[2];
                ir4[2] = (v558_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v555_data, 6))));
                float v561_data = r2[3];
                float v564_data = ir4[3];
                ir4[3] = (v564_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v561_data, 6))));
                float v567_data = r2[4];
                float v570_data = ir4[4];
                ir4[4] = (v570_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v567_data, 6))));
                float v573_data = r2[5];
                float v576_data = ir4[5];
                ir4[5] = (v576_data + (v542_data * (sycl::group_broadcast(item.get_sub_group(), v573_data, 6))));
              }
              if (v13_lead < 12) {
                float v582_data = r3[7];
                float v583_data = r2[0];
                float v586_data = ir4[0];
                ir4[0] = (v586_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v583_data, 7))));
                float v589_data = r2[1];
                float v592_data = ir4[1];
                ir4[1] = (v592_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v589_data, 7))));
                float v595_data = r2[2];
                float v598_data = ir4[2];
                ir4[2] = (v598_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v595_data, 7))));
                float v601_data = r2[3];
                float v604_data = ir4[3];
                ir4[3] = (v604_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v601_data, 7))));
                float v607_data = r2[4];
                float v610_data = ir4[4];
                ir4[4] = (v610_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v607_data, 7))));
                float v613_data = r2[5];
                float v616_data = ir4[5];
                ir4[5] = (v616_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v613_data, 7))));
              }
              if (v13_lead < 12) {
                float v622_data = r3[8];
                float v623_data = r2[0];
                float v626_data = ir4[0];
                ir4[0] = (v626_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v623_data, 8))));
                float v629_data = r2[1];
                float v632_data = ir4[1];
                ir4[1] = (v632_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v629_data, 8))));
                float v635_data = r2[2];
                float v638_data = ir4[2];
                ir4[2] = (v638_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v635_data, 8))));
                float v641_data = r2[3];
                float v644_data = ir4[3];
                ir4[3] = (v644_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v641_data, 8))));
                float v647_data = r2[4];
                float v650_data = ir4[4];
                ir4[4] = (v650_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v647_data, 8))));
                float v653_data = r2[5];
                float v656_data = ir4[5];
                ir4[5] = (v656_data + (v622_data * (sycl::group_broadcast(item.get_sub_group(), v653_data, 8))));
              }
              if (v13_lead < 12) {
                float v662_data = r3[9];
                float v663_data = r2[0];
                float v666_data = ir4[0];
                ir4[0] = (v666_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v663_data, 9))));
                float v669_data = r2[1];
                float v672_data = ir4[1];
                ir4[1] = (v672_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v669_data, 9))));
                float v675_data = r2[2];
                float v678_data = ir4[2];
                ir4[2] = (v678_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v675_data, 9))));
                float v681_data = r2[3];
                float v684_data = ir4[3];
                ir4[3] = (v684_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v681_data, 9))));
                float v687_data = r2[4];
                float v690_data = ir4[4];
                ir4[4] = (v690_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v687_data, 9))));
                float v693_data = r2[5];
                float v696_data = ir4[5];
                ir4[5] = (v696_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v693_data, 9))));
              }
              if (v13_lead < 12) {
                float v702_data = r3[10];
                float v703_data = r2[0];
                float v706_data = ir4[0];
                ir4[0] = (v706_data + (v702_data * (sycl::group_broadcast(item.get_sub_group(), v703_data, 10))));
                float v709_data = r2[1];
                float v712_data = ir4[1];
                ir4[1] = (v712_data + (v702_data * (sycl::group_broadcast(item.get_sub_group(), v709_data, 10))));
                float v715_data = r2[2];
                float v718_data = ir4[2];
                ir4[2] = (v718_data + (v702_data * (sycl::group_broadcast(item.get_sub_group(), v715_data, 10))));
                float v721_data = r2[3];
                float v724_data = ir4[3];
                ir4[3] = (v724_data + (v702_data * (sycl::group_broadcast(item.get_sub_group(), v721_data, 10))));
                float v727_data = r2[4];
                float v730_data = ir4[4];
                ir4[4] = (v730_data + (v702_data * (sycl::group_broadcast(item.get_sub_group(), v727_data, 10))));
                float v733_data = r2[5];
                float v736_data = ir4[5];
                ir4[5] = (v736_data + (v702_data * (sycl::group_broadcast(item.get_sub_group(), v733_data, 10))));
              }
              if (v13_lead < 12) {
                float v742_data = r3[11];
                float v743_data = r2[0];
                float v746_data = ir4[0];
                ir4[0] = (v746_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v743_data, 11))));
                float v749_data = r2[1];
                float v752_data = ir4[1];
                ir4[1] = (v752_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v749_data, 11))));
                float v755_data = r2[2];
                float v758_data = ir4[2];
                ir4[2] = (v758_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v755_data, 11))));
                float v761_data = r2[3];
                float v764_data = ir4[3];
                ir4[3] = (v764_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v761_data, 11))));
                float v767_data = r2[4];
                float v770_data = ir4[4];
                ir4[4] = (v770_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v767_data, 11))));
                float v773_data = r2[5];
                float v776_data = ir4[5];
                ir4[5] = (v776_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v773_data, 11))));
              }
              if (v13_lead < 12) {
                #pragma unroll
                for (int32_t v782_n1 = 0; v782_n1 < 6; ++v782_n1) {
                  float v784_data = ir4[v782_n1];
                  r4[v782_n1] = v784_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v13_lead < 12) {
                #pragma unroll
                for (int32_t v790_i1 = 0; v790_i1 < 6; ++v790_i1) {
                  float v792_data = r4[v790_i1];
                  glb_m2[(v13_lead + (v790_i1 * 12))] = v792_data;
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


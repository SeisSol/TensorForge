// === base name ===
kernel_32dccc6a83e5241d

// === header ===
void launcher_kernel_32dccc6a83e5241d(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_32dccc6a83e5241d(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_32dccc6a83e5241d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_32dccc6a83e5241d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×32(6×12) {0..6}×{0..12} strided
        // m1 32×32(12×12) {0..12}×{0..12} strided
        // m2 32×32(6×12) {0..6}×{0..12} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // m4 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m0 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
        // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m2 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
        // m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, 1] = m4 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..12}×{0..12})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v15_lead = item.get_local_id(0) % 16;
              if (v15_lead < 6) {
                #pragma unroll
                for (int32_t v17_i1 = 0; v17_i1 < 12; ++v17_i1) {
                  float v25_data = glb_m0[(v15_lead + (v17_i1 * 6))];
                  r0[v17_i1] = v25_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              if (v15_lead < 12) {
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
                  float v40_data = glb_m1[(v15_lead + (v32_i1 * 12))];
                  r1[v32_i1] = v40_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v15_lead < 6) {
                #pragma unroll
                for (int32_t v47_i1 = 0; v47_i1 < 12; ++v47_i1) {
                  float v55_data = glb_m2[(v15_lead + (v47_i1 * 6))];
                  r3[v47_i1] = v55_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              if (v15_lead < 6) {
                float v62_data = r0[0];
                float v63_data = r1[0];
                float v66_data = r2[0];
                r2[0] = (v66_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 0))));
                float v69_data = r1[1];
                float v72_data = r2[1];
                r2[1] = (v72_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 0))));
                float v75_data = r1[2];
                float v78_data = r2[2];
                r2[2] = (v78_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 0))));
                float v81_data = r1[3];
                float v84_data = r2[3];
                r2[3] = (v84_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 0))));
                float v87_data = r1[4];
                float v90_data = r2[4];
                r2[4] = (v90_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 0))));
                float v93_data = r1[5];
                float v96_data = r2[5];
                r2[5] = (v96_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 0))));
                float v99_data = r1[6];
                float v102_data = r2[6];
                r2[6] = (v102_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 0))));
                float v105_data = r1[7];
                float v108_data = r2[7];
                r2[7] = (v108_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 0))));
                float v111_data = r1[8];
                float v114_data = r2[8];
                r2[8] = (v114_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 0))));
                float v117_data = r1[9];
                float v120_data = r2[9];
                r2[9] = (v120_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 0))));
                float v123_data = r1[10];
                float v126_data = r2[10];
                r2[10] = (v126_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 0))));
                float v129_data = r1[11];
                float v132_data = r2[11];
                r2[11] = (v132_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 0))));
              }
              if (v15_lead < 6) {
                float v138_data = r0[1];
                float v139_data = r1[0];
                float v142_data = r2[0];
                r2[0] = (v142_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 1))));
                float v145_data = r1[1];
                float v148_data = r2[1];
                r2[1] = (v148_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 1))));
                float v151_data = r1[2];
                float v154_data = r2[2];
                r2[2] = (v154_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 1))));
                float v157_data = r1[3];
                float v160_data = r2[3];
                r2[3] = (v160_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 1))));
                float v163_data = r1[4];
                float v166_data = r2[4];
                r2[4] = (v166_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 1))));
                float v169_data = r1[5];
                float v172_data = r2[5];
                r2[5] = (v172_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 1))));
                float v175_data = r1[6];
                float v178_data = r2[6];
                r2[6] = (v178_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 1))));
                float v181_data = r1[7];
                float v184_data = r2[7];
                r2[7] = (v184_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 1))));
                float v187_data = r1[8];
                float v190_data = r2[8];
                r2[8] = (v190_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 1))));
                float v193_data = r1[9];
                float v196_data = r2[9];
                r2[9] = (v196_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 1))));
                float v199_data = r1[10];
                float v202_data = r2[10];
                r2[10] = (v202_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 1))));
                float v205_data = r1[11];
                float v208_data = r2[11];
                r2[11] = (v208_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v205_data, 1))));
              }
              if (v15_lead < 6) {
                float v214_data = r0[2];
                float v215_data = r1[0];
                float v218_data = r2[0];
                r2[0] = (v218_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v215_data, 2))));
                float v221_data = r1[1];
                float v224_data = r2[1];
                r2[1] = (v224_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v221_data, 2))));
                float v227_data = r1[2];
                float v230_data = r2[2];
                r2[2] = (v230_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v227_data, 2))));
                float v233_data = r1[3];
                float v236_data = r2[3];
                r2[3] = (v236_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v233_data, 2))));
                float v239_data = r1[4];
                float v242_data = r2[4];
                r2[4] = (v242_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 2))));
                float v245_data = r1[5];
                float v248_data = r2[5];
                r2[5] = (v248_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 2))));
                float v251_data = r1[6];
                float v254_data = r2[6];
                r2[6] = (v254_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 2))));
                float v257_data = r1[7];
                float v260_data = r2[7];
                r2[7] = (v260_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v257_data, 2))));
                float v263_data = r1[8];
                float v266_data = r2[8];
                r2[8] = (v266_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v263_data, 2))));
                float v269_data = r1[9];
                float v272_data = r2[9];
                r2[9] = (v272_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v269_data, 2))));
                float v275_data = r1[10];
                float v278_data = r2[10];
                r2[10] = (v278_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 2))));
                float v281_data = r1[11];
                float v284_data = r2[11];
                r2[11] = (v284_data + (v214_data * (sycl::group_broadcast(item.get_sub_group(), v281_data, 2))));
              }
              if (v15_lead < 6) {
                float v290_data = r0[3];
                float v291_data = r1[0];
                float v294_data = r2[0];
                r2[0] = (v294_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 3))));
                float v297_data = r1[1];
                float v300_data = r2[1];
                r2[1] = (v300_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v297_data, 3))));
                float v303_data = r1[2];
                float v306_data = r2[2];
                r2[2] = (v306_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v303_data, 3))));
                float v309_data = r1[3];
                float v312_data = r2[3];
                r2[3] = (v312_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 3))));
                float v315_data = r1[4];
                float v318_data = r2[4];
                r2[4] = (v318_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 3))));
                float v321_data = r1[5];
                float v324_data = r2[5];
                r2[5] = (v324_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v321_data, 3))));
                float v327_data = r1[6];
                float v330_data = r2[6];
                r2[6] = (v330_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 3))));
                float v333_data = r1[7];
                float v336_data = r2[7];
                r2[7] = (v336_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 3))));
                float v339_data = r1[8];
                float v342_data = r2[8];
                r2[8] = (v342_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 3))));
                float v345_data = r1[9];
                float v348_data = r2[9];
                r2[9] = (v348_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 3))));
                float v351_data = r1[10];
                float v354_data = r2[10];
                r2[10] = (v354_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v351_data, 3))));
                float v357_data = r1[11];
                float v360_data = r2[11];
                r2[11] = (v360_data + (v290_data * (sycl::group_broadcast(item.get_sub_group(), v357_data, 3))));
              }
              if (v15_lead < 6) {
                float v366_data = r0[4];
                float v367_data = r1[0];
                float v370_data = r2[0];
                r2[0] = (v370_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 4))));
                float v373_data = r1[1];
                float v376_data = r2[1];
                r2[1] = (v376_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 4))));
                float v379_data = r1[2];
                float v382_data = r2[2];
                r2[2] = (v382_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 4))));
                float v385_data = r1[3];
                float v388_data = r2[3];
                r2[3] = (v388_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 4))));
                float v391_data = r1[4];
                float v394_data = r2[4];
                r2[4] = (v394_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v391_data, 4))));
                float v397_data = r1[5];
                float v400_data = r2[5];
                r2[5] = (v400_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v397_data, 4))));
                float v403_data = r1[6];
                float v406_data = r2[6];
                r2[6] = (v406_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v403_data, 4))));
                float v409_data = r1[7];
                float v412_data = r2[7];
                r2[7] = (v412_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v409_data, 4))));
                float v415_data = r1[8];
                float v418_data = r2[8];
                r2[8] = (v418_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v415_data, 4))));
                float v421_data = r1[9];
                float v424_data = r2[9];
                r2[9] = (v424_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v421_data, 4))));
                float v427_data = r1[10];
                float v430_data = r2[10];
                r2[10] = (v430_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v427_data, 4))));
                float v433_data = r1[11];
                float v436_data = r2[11];
                r2[11] = (v436_data + (v366_data * (sycl::group_broadcast(item.get_sub_group(), v433_data, 4))));
              }
              if (v15_lead < 6) {
                float v442_data = r0[5];
                float v443_data = r1[0];
                float v446_data = r2[0];
                r2[0] = (v446_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v443_data, 5))));
                float v449_data = r1[1];
                float v452_data = r2[1];
                r2[1] = (v452_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v449_data, 5))));
                float v455_data = r1[2];
                float v458_data = r2[2];
                r2[2] = (v458_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v455_data, 5))));
                float v461_data = r1[3];
                float v464_data = r2[3];
                r2[3] = (v464_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v461_data, 5))));
                float v467_data = r1[4];
                float v470_data = r2[4];
                r2[4] = (v470_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v467_data, 5))));
                float v473_data = r1[5];
                float v476_data = r2[5];
                r2[5] = (v476_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v473_data, 5))));
                float v479_data = r1[6];
                float v482_data = r2[6];
                r2[6] = (v482_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v479_data, 5))));
                float v485_data = r1[7];
                float v488_data = r2[7];
                r2[7] = (v488_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v485_data, 5))));
                float v491_data = r1[8];
                float v494_data = r2[8];
                r2[8] = (v494_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v491_data, 5))));
                float v497_data = r1[9];
                float v500_data = r2[9];
                r2[9] = (v500_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v497_data, 5))));
                float v503_data = r1[10];
                float v506_data = r2[10];
                r2[10] = (v506_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v503_data, 5))));
                float v509_data = r1[11];
                float v512_data = r2[11];
                r2[11] = (v512_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v509_data, 5))));
              }
              if (v15_lead < 6) {
                float v518_data = r0[6];
                float v519_data = r1[0];
                float v522_data = r2[0];
                r2[0] = (v522_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v519_data, 6))));
                float v525_data = r1[1];
                float v528_data = r2[1];
                r2[1] = (v528_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v525_data, 6))));
                float v531_data = r1[2];
                float v534_data = r2[2];
                r2[2] = (v534_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v531_data, 6))));
                float v537_data = r1[3];
                float v540_data = r2[3];
                r2[3] = (v540_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v537_data, 6))));
                float v543_data = r1[4];
                float v546_data = r2[4];
                r2[4] = (v546_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v543_data, 6))));
                float v549_data = r1[5];
                float v552_data = r2[5];
                r2[5] = (v552_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v549_data, 6))));
                float v555_data = r1[6];
                float v558_data = r2[6];
                r2[6] = (v558_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v555_data, 6))));
                float v561_data = r1[7];
                float v564_data = r2[7];
                r2[7] = (v564_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v561_data, 6))));
                float v567_data = r1[8];
                float v570_data = r2[8];
                r2[8] = (v570_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v567_data, 6))));
                float v573_data = r1[9];
                float v576_data = r2[9];
                r2[9] = (v576_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v573_data, 6))));
                float v579_data = r1[10];
                float v582_data = r2[10];
                r2[10] = (v582_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v579_data, 6))));
                float v585_data = r1[11];
                float v588_data = r2[11];
                r2[11] = (v588_data + (v518_data * (sycl::group_broadcast(item.get_sub_group(), v585_data, 6))));
              }
              if (v15_lead < 6) {
                float v594_data = r0[7];
                float v595_data = r1[0];
                float v598_data = r2[0];
                r2[0] = (v598_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v595_data, 7))));
                float v601_data = r1[1];
                float v604_data = r2[1];
                r2[1] = (v604_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v601_data, 7))));
                float v607_data = r1[2];
                float v610_data = r2[2];
                r2[2] = (v610_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v607_data, 7))));
                float v613_data = r1[3];
                float v616_data = r2[3];
                r2[3] = (v616_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v613_data, 7))));
                float v619_data = r1[4];
                float v622_data = r2[4];
                r2[4] = (v622_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v619_data, 7))));
                float v625_data = r1[5];
                float v628_data = r2[5];
                r2[5] = (v628_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v625_data, 7))));
                float v631_data = r1[6];
                float v634_data = r2[6];
                r2[6] = (v634_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v631_data, 7))));
                float v637_data = r1[7];
                float v640_data = r2[7];
                r2[7] = (v640_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v637_data, 7))));
                float v643_data = r1[8];
                float v646_data = r2[8];
                r2[8] = (v646_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v643_data, 7))));
                float v649_data = r1[9];
                float v652_data = r2[9];
                r2[9] = (v652_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v649_data, 7))));
                float v655_data = r1[10];
                float v658_data = r2[10];
                r2[10] = (v658_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v655_data, 7))));
                float v661_data = r1[11];
                float v664_data = r2[11];
                r2[11] = (v664_data + (v594_data * (sycl::group_broadcast(item.get_sub_group(), v661_data, 7))));
              }
              if (v15_lead < 6) {
                float v670_data = r0[8];
                float v671_data = r1[0];
                float v674_data = r2[0];
                r2[0] = (v674_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v671_data, 8))));
                float v677_data = r1[1];
                float v680_data = r2[1];
                r2[1] = (v680_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v677_data, 8))));
                float v683_data = r1[2];
                float v686_data = r2[2];
                r2[2] = (v686_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v683_data, 8))));
                float v689_data = r1[3];
                float v692_data = r2[3];
                r2[3] = (v692_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v689_data, 8))));
                float v695_data = r1[4];
                float v698_data = r2[4];
                r2[4] = (v698_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v695_data, 8))));
                float v701_data = r1[5];
                float v704_data = r2[5];
                r2[5] = (v704_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v701_data, 8))));
                float v707_data = r1[6];
                float v710_data = r2[6];
                r2[6] = (v710_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v707_data, 8))));
                float v713_data = r1[7];
                float v716_data = r2[7];
                r2[7] = (v716_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v713_data, 8))));
                float v719_data = r1[8];
                float v722_data = r2[8];
                r2[8] = (v722_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v719_data, 8))));
                float v725_data = r1[9];
                float v728_data = r2[9];
                r2[9] = (v728_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v725_data, 8))));
                float v731_data = r1[10];
                float v734_data = r2[10];
                r2[10] = (v734_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v731_data, 8))));
                float v737_data = r1[11];
                float v740_data = r2[11];
                r2[11] = (v740_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v737_data, 8))));
              }
              if (v15_lead < 6) {
                float v746_data = r0[9];
                float v747_data = r1[0];
                float v750_data = r2[0];
                r2[0] = (v750_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v747_data, 9))));
                float v753_data = r1[1];
                float v756_data = r2[1];
                r2[1] = (v756_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v753_data, 9))));
                float v759_data = r1[2];
                float v762_data = r2[2];
                r2[2] = (v762_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v759_data, 9))));
                float v765_data = r1[3];
                float v768_data = r2[3];
                r2[3] = (v768_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v765_data, 9))));
                float v771_data = r1[4];
                float v774_data = r2[4];
                r2[4] = (v774_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v771_data, 9))));
                float v777_data = r1[5];
                float v780_data = r2[5];
                r2[5] = (v780_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v777_data, 9))));
                float v783_data = r1[6];
                float v786_data = r2[6];
                r2[6] = (v786_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v783_data, 9))));
                float v789_data = r1[7];
                float v792_data = r2[7];
                r2[7] = (v792_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v789_data, 9))));
                float v795_data = r1[8];
                float v798_data = r2[8];
                r2[8] = (v798_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v795_data, 9))));
                float v801_data = r1[9];
                float v804_data = r2[9];
                r2[9] = (v804_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v801_data, 9))));
                float v807_data = r1[10];
                float v810_data = r2[10];
                r2[10] = (v810_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v807_data, 9))));
                float v813_data = r1[11];
                float v816_data = r2[11];
                r2[11] = (v816_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v813_data, 9))));
              }
              if (v15_lead < 6) {
                float v822_data = r0[10];
                float v823_data = r1[0];
                float v826_data = r2[0];
                r2[0] = (v826_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v823_data, 10))));
                float v829_data = r1[1];
                float v832_data = r2[1];
                r2[1] = (v832_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v829_data, 10))));
                float v835_data = r1[2];
                float v838_data = r2[2];
                r2[2] = (v838_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v835_data, 10))));
                float v841_data = r1[3];
                float v844_data = r2[3];
                r2[3] = (v844_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v841_data, 10))));
                float v847_data = r1[4];
                float v850_data = r2[4];
                r2[4] = (v850_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v847_data, 10))));
                float v853_data = r1[5];
                float v856_data = r2[5];
                r2[5] = (v856_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v853_data, 10))));
                float v859_data = r1[6];
                float v862_data = r2[6];
                r2[6] = (v862_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v859_data, 10))));
                float v865_data = r1[7];
                float v868_data = r2[7];
                r2[7] = (v868_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v865_data, 10))));
                float v871_data = r1[8];
                float v874_data = r2[8];
                r2[8] = (v874_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v871_data, 10))));
                float v877_data = r1[9];
                float v880_data = r2[9];
                r2[9] = (v880_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v877_data, 10))));
                float v883_data = r1[10];
                float v886_data = r2[10];
                r2[10] = (v886_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v883_data, 10))));
                float v889_data = r1[11];
                float v892_data = r2[11];
                r2[11] = (v892_data + (v822_data * (sycl::group_broadcast(item.get_sub_group(), v889_data, 10))));
              }
              if (v15_lead < 6) {
                float v898_data = r0[11];
                float v899_data = r1[0];
                float v902_data = r2[0];
                r2[0] = (v902_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v899_data, 11))));
                float v905_data = r1[1];
                float v908_data = r2[1];
                r2[1] = (v908_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v905_data, 11))));
                float v911_data = r1[2];
                float v914_data = r2[2];
                r2[2] = (v914_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v911_data, 11))));
                float v917_data = r1[3];
                float v920_data = r2[3];
                r2[3] = (v920_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v917_data, 11))));
                float v923_data = r1[4];
                float v926_data = r2[4];
                r2[4] = (v926_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v923_data, 11))));
                float v929_data = r1[5];
                float v932_data = r2[5];
                r2[5] = (v932_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v929_data, 11))));
                float v935_data = r1[6];
                float v938_data = r2[6];
                r2[6] = (v938_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v935_data, 11))));
                float v941_data = r1[7];
                float v944_data = r2[7];
                r2[7] = (v944_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v941_data, 11))));
                float v947_data = r1[8];
                float v950_data = r2[8];
                r2[8] = (v950_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v947_data, 11))));
                float v953_data = r1[9];
                float v956_data = r2[9];
                r2[9] = (v956_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v953_data, 11))));
                float v959_data = r1[10];
                float v962_data = r2[10];
                r2[10] = (v962_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v959_data, 11))));
                float v965_data = r1[11];
                float v968_data = r2[11];
                r2[11] = (v968_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v965_data, 11))));
              }
              // s0 = store{r>s}(localShrMem0, r2);
              if (v15_lead < 6) {
                #pragma unroll
                for (int32_t v974_i1 = 0; v974_i1 < 12; ++v974_i1) {
                  float v976_data = r2[v974_i1];
                  int32_t v983_a = v15_lead + (v974_i1 * 12);
                  s0[(v983_a ^ ((v983_a >> 4) & 15))] = v976_data;
                }
              }
              float r5[12]{};
              // r5 = load{g>r}(glb_m4);
              if (v15_lead < 12) {
                #pragma unroll
                for (int32_t v992_i1 = 0; v992_i1 < 12; ++v992_i1) {
                  float v1000_data = glb_m4[(v15_lead + (v992_i1 * 12))];
                  r5[v992_i1] = v1000_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[12]{};
              // r4 = +(r3 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              if (v15_lead < 6) {
                float v1008_data = r3[0];
                float v1009_data = r1[0];
                float v1012_data = ir4[0];
                ir4[0] = (v1012_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1009_data, 0))));
                float v1015_data = r1[1];
                float v1018_data = ir4[1];
                ir4[1] = (v1018_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1015_data, 0))));
                float v1021_data = r1[2];
                float v1024_data = ir4[2];
                ir4[2] = (v1024_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1021_data, 0))));
                float v1027_data = r1[3];
                float v1030_data = ir4[3];
                ir4[3] = (v1030_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1027_data, 0))));
                float v1033_data = r1[4];
                float v1036_data = ir4[4];
                ir4[4] = (v1036_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1033_data, 0))));
                float v1039_data = r1[5];
                float v1042_data = ir4[5];
                ir4[5] = (v1042_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1039_data, 0))));
                float v1045_data = r1[6];
                float v1048_data = ir4[6];
                ir4[6] = (v1048_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1045_data, 0))));
                float v1051_data = r1[7];
                float v1054_data = ir4[7];
                ir4[7] = (v1054_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1051_data, 0))));
                float v1057_data = r1[8];
                float v1060_data = ir4[8];
                ir4[8] = (v1060_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1057_data, 0))));
                float v1063_data = r1[9];
                float v1066_data = ir4[9];
                ir4[9] = (v1066_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1063_data, 0))));
                float v1069_data = r1[10];
                float v1072_data = ir4[10];
                ir4[10] = (v1072_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1069_data, 0))));
                float v1075_data = r1[11];
                float v1078_data = ir4[11];
                ir4[11] = (v1078_data + (v1008_data * (sycl::group_broadcast(item.get_sub_group(), v1075_data, 0))));
              }
              if (v15_lead < 6) {
                float v1084_data = r3[1];
                float v1085_data = r1[0];
                float v1088_data = ir4[0];
                ir4[0] = (v1088_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1085_data, 1))));
                float v1091_data = r1[1];
                float v1094_data = ir4[1];
                ir4[1] = (v1094_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1091_data, 1))));
                float v1097_data = r1[2];
                float v1100_data = ir4[2];
                ir4[2] = (v1100_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1097_data, 1))));
                float v1103_data = r1[3];
                float v1106_data = ir4[3];
                ir4[3] = (v1106_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1103_data, 1))));
                float v1109_data = r1[4];
                float v1112_data = ir4[4];
                ir4[4] = (v1112_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1109_data, 1))));
                float v1115_data = r1[5];
                float v1118_data = ir4[5];
                ir4[5] = (v1118_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1115_data, 1))));
                float v1121_data = r1[6];
                float v1124_data = ir4[6];
                ir4[6] = (v1124_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1121_data, 1))));
                float v1127_data = r1[7];
                float v1130_data = ir4[7];
                ir4[7] = (v1130_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1127_data, 1))));
                float v1133_data = r1[8];
                float v1136_data = ir4[8];
                ir4[8] = (v1136_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1133_data, 1))));
                float v1139_data = r1[9];
                float v1142_data = ir4[9];
                ir4[9] = (v1142_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1139_data, 1))));
                float v1145_data = r1[10];
                float v1148_data = ir4[10];
                ir4[10] = (v1148_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1145_data, 1))));
                float v1151_data = r1[11];
                float v1154_data = ir4[11];
                ir4[11] = (v1154_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v1151_data, 1))));
              }
              if (v15_lead < 6) {
                float v1160_data = r3[2];
                float v1161_data = r1[0];
                float v1164_data = ir4[0];
                ir4[0] = (v1164_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 2))));
                float v1167_data = r1[1];
                float v1170_data = ir4[1];
                ir4[1] = (v1170_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 2))));
                float v1173_data = r1[2];
                float v1176_data = ir4[2];
                ir4[2] = (v1176_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 2))));
                float v1179_data = r1[3];
                float v1182_data = ir4[3];
                ir4[3] = (v1182_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 2))));
                float v1185_data = r1[4];
                float v1188_data = ir4[4];
                ir4[4] = (v1188_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 2))));
                float v1191_data = r1[5];
                float v1194_data = ir4[5];
                ir4[5] = (v1194_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 2))));
                float v1197_data = r1[6];
                float v1200_data = ir4[6];
                ir4[6] = (v1200_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 2))));
                float v1203_data = r1[7];
                float v1206_data = ir4[7];
                ir4[7] = (v1206_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 2))));
                float v1209_data = r1[8];
                float v1212_data = ir4[8];
                ir4[8] = (v1212_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 2))));
                float v1215_data = r1[9];
                float v1218_data = ir4[9];
                ir4[9] = (v1218_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 2))));
                float v1221_data = r1[10];
                float v1224_data = ir4[10];
                ir4[10] = (v1224_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 2))));
                float v1227_data = r1[11];
                float v1230_data = ir4[11];
                ir4[11] = (v1230_data + (v1160_data * (sycl::group_broadcast(item.get_sub_group(), v1227_data, 2))));
              }
              if (v15_lead < 6) {
                float v1236_data = r3[3];
                float v1237_data = r1[0];
                float v1240_data = ir4[0];
                ir4[0] = (v1240_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1237_data, 3))));
                float v1243_data = r1[1];
                float v1246_data = ir4[1];
                ir4[1] = (v1246_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1243_data, 3))));
                float v1249_data = r1[2];
                float v1252_data = ir4[2];
                ir4[2] = (v1252_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1249_data, 3))));
                float v1255_data = r1[3];
                float v1258_data = ir4[3];
                ir4[3] = (v1258_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1255_data, 3))));
                float v1261_data = r1[4];
                float v1264_data = ir4[4];
                ir4[4] = (v1264_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1261_data, 3))));
                float v1267_data = r1[5];
                float v1270_data = ir4[5];
                ir4[5] = (v1270_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1267_data, 3))));
                float v1273_data = r1[6];
                float v1276_data = ir4[6];
                ir4[6] = (v1276_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1273_data, 3))));
                float v1279_data = r1[7];
                float v1282_data = ir4[7];
                ir4[7] = (v1282_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1279_data, 3))));
                float v1285_data = r1[8];
                float v1288_data = ir4[8];
                ir4[8] = (v1288_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1285_data, 3))));
                float v1291_data = r1[9];
                float v1294_data = ir4[9];
                ir4[9] = (v1294_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1291_data, 3))));
                float v1297_data = r1[10];
                float v1300_data = ir4[10];
                ir4[10] = (v1300_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1297_data, 3))));
                float v1303_data = r1[11];
                float v1306_data = ir4[11];
                ir4[11] = (v1306_data + (v1236_data * (sycl::group_broadcast(item.get_sub_group(), v1303_data, 3))));
              }
              if (v15_lead < 6) {
                float v1312_data = r3[4];
                float v1313_data = r1[0];
                float v1316_data = ir4[0];
                ir4[0] = (v1316_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1313_data, 4))));
                float v1319_data = r1[1];
                float v1322_data = ir4[1];
                ir4[1] = (v1322_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 4))));
                float v1325_data = r1[2];
                float v1328_data = ir4[2];
                ir4[2] = (v1328_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 4))));
                float v1331_data = r1[3];
                float v1334_data = ir4[3];
                ir4[3] = (v1334_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 4))));
                float v1337_data = r1[4];
                float v1340_data = ir4[4];
                ir4[4] = (v1340_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 4))));
                float v1343_data = r1[5];
                float v1346_data = ir4[5];
                ir4[5] = (v1346_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 4))));
                float v1349_data = r1[6];
                float v1352_data = ir4[6];
                ir4[6] = (v1352_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 4))));
                float v1355_data = r1[7];
                float v1358_data = ir4[7];
                ir4[7] = (v1358_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 4))));
                float v1361_data = r1[8];
                float v1364_data = ir4[8];
                ir4[8] = (v1364_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 4))));
                float v1367_data = r1[9];
                float v1370_data = ir4[9];
                ir4[9] = (v1370_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1367_data, 4))));
                float v1373_data = r1[10];
                float v1376_data = ir4[10];
                ir4[10] = (v1376_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1373_data, 4))));
                float v1379_data = r1[11];
                float v1382_data = ir4[11];
                ir4[11] = (v1382_data + (v1312_data * (sycl::group_broadcast(item.get_sub_group(), v1379_data, 4))));
              }
              if (v15_lead < 6) {
                float v1388_data = r3[5];
                float v1389_data = r1[0];
                float v1392_data = ir4[0];
                ir4[0] = (v1392_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1389_data, 5))));
                float v1395_data = r1[1];
                float v1398_data = ir4[1];
                ir4[1] = (v1398_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1395_data, 5))));
                float v1401_data = r1[2];
                float v1404_data = ir4[2];
                ir4[2] = (v1404_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1401_data, 5))));
                float v1407_data = r1[3];
                float v1410_data = ir4[3];
                ir4[3] = (v1410_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1407_data, 5))));
                float v1413_data = r1[4];
                float v1416_data = ir4[4];
                ir4[4] = (v1416_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1413_data, 5))));
                float v1419_data = r1[5];
                float v1422_data = ir4[5];
                ir4[5] = (v1422_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1419_data, 5))));
                float v1425_data = r1[6];
                float v1428_data = ir4[6];
                ir4[6] = (v1428_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1425_data, 5))));
                float v1431_data = r1[7];
                float v1434_data = ir4[7];
                ir4[7] = (v1434_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1431_data, 5))));
                float v1437_data = r1[8];
                float v1440_data = ir4[8];
                ir4[8] = (v1440_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1437_data, 5))));
                float v1443_data = r1[9];
                float v1446_data = ir4[9];
                ir4[9] = (v1446_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1443_data, 5))));
                float v1449_data = r1[10];
                float v1452_data = ir4[10];
                ir4[10] = (v1452_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1449_data, 5))));
                float v1455_data = r1[11];
                float v1458_data = ir4[11];
                ir4[11] = (v1458_data + (v1388_data * (sycl::group_broadcast(item.get_sub_group(), v1455_data, 5))));
              }
              if (v15_lead < 6) {
                float v1464_data = r3[6];
                float v1465_data = r1[0];
                float v1468_data = ir4[0];
                ir4[0] = (v1468_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1465_data, 6))));
                float v1471_data = r1[1];
                float v1474_data = ir4[1];
                ir4[1] = (v1474_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1471_data, 6))));
                float v1477_data = r1[2];
                float v1480_data = ir4[2];
                ir4[2] = (v1480_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1477_data, 6))));
                float v1483_data = r1[3];
                float v1486_data = ir4[3];
                ir4[3] = (v1486_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1483_data, 6))));
                float v1489_data = r1[4];
                float v1492_data = ir4[4];
                ir4[4] = (v1492_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1489_data, 6))));
                float v1495_data = r1[5];
                float v1498_data = ir4[5];
                ir4[5] = (v1498_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1495_data, 6))));
                float v1501_data = r1[6];
                float v1504_data = ir4[6];
                ir4[6] = (v1504_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1501_data, 6))));
                float v1507_data = r1[7];
                float v1510_data = ir4[7];
                ir4[7] = (v1510_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1507_data, 6))));
                float v1513_data = r1[8];
                float v1516_data = ir4[8];
                ir4[8] = (v1516_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1513_data, 6))));
                float v1519_data = r1[9];
                float v1522_data = ir4[9];
                ir4[9] = (v1522_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1519_data, 6))));
                float v1525_data = r1[10];
                float v1528_data = ir4[10];
                ir4[10] = (v1528_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1525_data, 6))));
                float v1531_data = r1[11];
                float v1534_data = ir4[11];
                ir4[11] = (v1534_data + (v1464_data * (sycl::group_broadcast(item.get_sub_group(), v1531_data, 6))));
              }
              if (v15_lead < 6) {
                float v1540_data = r3[7];
                float v1541_data = r1[0];
                float v1544_data = ir4[0];
                ir4[0] = (v1544_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1541_data, 7))));
                float v1547_data = r1[1];
                float v1550_data = ir4[1];
                ir4[1] = (v1550_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1547_data, 7))));
                float v1553_data = r1[2];
                float v1556_data = ir4[2];
                ir4[2] = (v1556_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1553_data, 7))));
                float v1559_data = r1[3];
                float v1562_data = ir4[3];
                ir4[3] = (v1562_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1559_data, 7))));
                float v1565_data = r1[4];
                float v1568_data = ir4[4];
                ir4[4] = (v1568_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1565_data, 7))));
                float v1571_data = r1[5];
                float v1574_data = ir4[5];
                ir4[5] = (v1574_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1571_data, 7))));
                float v1577_data = r1[6];
                float v1580_data = ir4[6];
                ir4[6] = (v1580_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1577_data, 7))));
                float v1583_data = r1[7];
                float v1586_data = ir4[7];
                ir4[7] = (v1586_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1583_data, 7))));
                float v1589_data = r1[8];
                float v1592_data = ir4[8];
                ir4[8] = (v1592_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1589_data, 7))));
                float v1595_data = r1[9];
                float v1598_data = ir4[9];
                ir4[9] = (v1598_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1595_data, 7))));
                float v1601_data = r1[10];
                float v1604_data = ir4[10];
                ir4[10] = (v1604_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1601_data, 7))));
                float v1607_data = r1[11];
                float v1610_data = ir4[11];
                ir4[11] = (v1610_data + (v1540_data * (sycl::group_broadcast(item.get_sub_group(), v1607_data, 7))));
              }
              if (v15_lead < 6) {
                float v1616_data = r3[8];
                float v1617_data = r1[0];
                float v1620_data = ir4[0];
                ir4[0] = (v1620_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1617_data, 8))));
                float v1623_data = r1[1];
                float v1626_data = ir4[1];
                ir4[1] = (v1626_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1623_data, 8))));
                float v1629_data = r1[2];
                float v1632_data = ir4[2];
                ir4[2] = (v1632_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1629_data, 8))));
                float v1635_data = r1[3];
                float v1638_data = ir4[3];
                ir4[3] = (v1638_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1635_data, 8))));
                float v1641_data = r1[4];
                float v1644_data = ir4[4];
                ir4[4] = (v1644_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1641_data, 8))));
                float v1647_data = r1[5];
                float v1650_data = ir4[5];
                ir4[5] = (v1650_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1647_data, 8))));
                float v1653_data = r1[6];
                float v1656_data = ir4[6];
                ir4[6] = (v1656_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1653_data, 8))));
                float v1659_data = r1[7];
                float v1662_data = ir4[7];
                ir4[7] = (v1662_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1659_data, 8))));
                float v1665_data = r1[8];
                float v1668_data = ir4[8];
                ir4[8] = (v1668_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1665_data, 8))));
                float v1671_data = r1[9];
                float v1674_data = ir4[9];
                ir4[9] = (v1674_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1671_data, 8))));
                float v1677_data = r1[10];
                float v1680_data = ir4[10];
                ir4[10] = (v1680_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1677_data, 8))));
                float v1683_data = r1[11];
                float v1686_data = ir4[11];
                ir4[11] = (v1686_data + (v1616_data * (sycl::group_broadcast(item.get_sub_group(), v1683_data, 8))));
              }
              if (v15_lead < 6) {
                float v1692_data = r3[9];
                float v1693_data = r1[0];
                float v1696_data = ir4[0];
                ir4[0] = (v1696_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1693_data, 9))));
                float v1699_data = r1[1];
                float v1702_data = ir4[1];
                ir4[1] = (v1702_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1699_data, 9))));
                float v1705_data = r1[2];
                float v1708_data = ir4[2];
                ir4[2] = (v1708_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1705_data, 9))));
                float v1711_data = r1[3];
                float v1714_data = ir4[3];
                ir4[3] = (v1714_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1711_data, 9))));
                float v1717_data = r1[4];
                float v1720_data = ir4[4];
                ir4[4] = (v1720_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1717_data, 9))));
                float v1723_data = r1[5];
                float v1726_data = ir4[5];
                ir4[5] = (v1726_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1723_data, 9))));
                float v1729_data = r1[6];
                float v1732_data = ir4[6];
                ir4[6] = (v1732_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1729_data, 9))));
                float v1735_data = r1[7];
                float v1738_data = ir4[7];
                ir4[7] = (v1738_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1735_data, 9))));
                float v1741_data = r1[8];
                float v1744_data = ir4[8];
                ir4[8] = (v1744_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1741_data, 9))));
                float v1747_data = r1[9];
                float v1750_data = ir4[9];
                ir4[9] = (v1750_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1747_data, 9))));
                float v1753_data = r1[10];
                float v1756_data = ir4[10];
                ir4[10] = (v1756_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1753_data, 9))));
                float v1759_data = r1[11];
                float v1762_data = ir4[11];
                ir4[11] = (v1762_data + (v1692_data * (sycl::group_broadcast(item.get_sub_group(), v1759_data, 9))));
              }
              if (v15_lead < 6) {
                float v1768_data = r3[10];
                float v1769_data = r1[0];
                float v1772_data = ir4[0];
                ir4[0] = (v1772_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1769_data, 10))));
                float v1775_data = r1[1];
                float v1778_data = ir4[1];
                ir4[1] = (v1778_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1775_data, 10))));
                float v1781_data = r1[2];
                float v1784_data = ir4[2];
                ir4[2] = (v1784_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1781_data, 10))));
                float v1787_data = r1[3];
                float v1790_data = ir4[3];
                ir4[3] = (v1790_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1787_data, 10))));
                float v1793_data = r1[4];
                float v1796_data = ir4[4];
                ir4[4] = (v1796_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1793_data, 10))));
                float v1799_data = r1[5];
                float v1802_data = ir4[5];
                ir4[5] = (v1802_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1799_data, 10))));
                float v1805_data = r1[6];
                float v1808_data = ir4[6];
                ir4[6] = (v1808_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1805_data, 10))));
                float v1811_data = r1[7];
                float v1814_data = ir4[7];
                ir4[7] = (v1814_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1811_data, 10))));
                float v1817_data = r1[8];
                float v1820_data = ir4[8];
                ir4[8] = (v1820_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1817_data, 10))));
                float v1823_data = r1[9];
                float v1826_data = ir4[9];
                ir4[9] = (v1826_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1823_data, 10))));
                float v1829_data = r1[10];
                float v1832_data = ir4[10];
                ir4[10] = (v1832_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1829_data, 10))));
                float v1835_data = r1[11];
                float v1838_data = ir4[11];
                ir4[11] = (v1838_data + (v1768_data * (sycl::group_broadcast(item.get_sub_group(), v1835_data, 10))));
              }
              if (v15_lead < 6) {
                float v1844_data = r3[11];
                float v1845_data = r1[0];
                float v1848_data = ir4[0];
                ir4[0] = (v1848_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1845_data, 11))));
                float v1851_data = r1[1];
                float v1854_data = ir4[1];
                ir4[1] = (v1854_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1851_data, 11))));
                float v1857_data = r1[2];
                float v1860_data = ir4[2];
                ir4[2] = (v1860_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1857_data, 11))));
                float v1863_data = r1[3];
                float v1866_data = ir4[3];
                ir4[3] = (v1866_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1863_data, 11))));
                float v1869_data = r1[4];
                float v1872_data = ir4[4];
                ir4[4] = (v1872_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1869_data, 11))));
                float v1875_data = r1[5];
                float v1878_data = ir4[5];
                ir4[5] = (v1878_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1875_data, 11))));
                float v1881_data = r1[6];
                float v1884_data = ir4[6];
                ir4[6] = (v1884_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1881_data, 11))));
                float v1887_data = r1[7];
                float v1890_data = ir4[7];
                ir4[7] = (v1890_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1887_data, 11))));
                float v1893_data = r1[8];
                float v1896_data = ir4[8];
                ir4[8] = (v1896_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1893_data, 11))));
                float v1899_data = r1[9];
                float v1902_data = ir4[9];
                ir4[9] = (v1902_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1899_data, 11))));
                float v1905_data = r1[10];
                float v1908_data = ir4[10];
                ir4[10] = (v1908_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1905_data, 11))));
                float v1911_data = r1[11];
                float v1914_data = ir4[11];
                ir4[11] = (v1914_data + (v1844_data * (sycl::group_broadcast(item.get_sub_group(), v1911_data, 11))));
              }
              if (v15_lead < 6) {
                #pragma unroll
                for (int32_t v1920_n1 = 0; v1920_n1 < 12; ++v1920_n1) {
                  float v1922_data = ir4[v1920_n1];
                  r4[v1920_n1] = v1922_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v15_lead < 6) {
                int32_t v1936_off = v15_lead + 6;
                #pragma unroll
                for (int32_t v1928_i1 = 0; v1928_i1 < 12; ++v1928_i1) {
                  float v1930_data = r4[v1928_i1];
                  int32_t v1938_a = v1936_off + (v1928_i1 * 12);
                  s0[(v1938_a ^ ((v1938_a >> 4) & 15))] = v1930_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m4););
              float r6[12]{};
              sycl::group_barrier(item.get_sub_group());
              // r6 = +(r5 * s0) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir6[12]{};
              if (v15_lead < 12) {
                float v1948_data = r5[0];
                float v1949_data = s0[0];
                float v1951_data = ir6[0];
                ir6[0] = (v1951_data + (v1948_data * v1949_data));
                float v1954_data = s0[12];
                float v1956_data = ir6[1];
                ir6[1] = (v1956_data + (v1948_data * v1954_data));
                float v1959_data = s0[25];
                float v1961_data = ir6[2];
                ir6[2] = (v1961_data + (v1948_data * v1959_data));
                float v1964_data = s0[38];
                float v1966_data = ir6[3];
                ir6[3] = (v1966_data + (v1948_data * v1964_data));
                float v1969_data = s0[51];
                float v1971_data = ir6[4];
                ir6[4] = (v1971_data + (v1948_data * v1969_data));
                float v1974_data = s0[63];
                float v1976_data = ir6[5];
                ir6[5] = (v1976_data + (v1948_data * v1974_data));
                float v1979_data = s0[76];
                float v1981_data = ir6[6];
                ir6[6] = (v1981_data + (v1948_data * v1979_data));
                float v1984_data = s0[81];
                float v1986_data = ir6[7];
                ir6[7] = (v1986_data + (v1948_data * v1984_data));
                float v1989_data = s0[102];
                float v1991_data = ir6[8];
                ir6[8] = (v1991_data + (v1948_data * v1989_data));
                float v1994_data = s0[106];
                float v1996_data = ir6[9];
                ir6[9] = (v1996_data + (v1948_data * v1994_data));
                float v1999_data = s0[127];
                float v2001_data = ir6[10];
                ir6[10] = (v2001_data + (v1948_data * v1999_data));
                float v2004_data = s0[140];
                float v2006_data = ir6[11];
                ir6[11] = (v2006_data + (v1948_data * v2004_data));
              }
              if (v15_lead < 12) {
                float v2012_data = r5[1];
                float v2013_data = s0[1];
                float v2015_data = ir6[0];
                ir6[0] = (v2015_data + (v2012_data * v2013_data));
                float v2018_data = s0[13];
                float v2020_data = ir6[1];
                ir6[1] = (v2020_data + (v2012_data * v2018_data));
                float v2023_data = s0[24];
                float v2025_data = ir6[2];
                ir6[2] = (v2025_data + (v2012_data * v2023_data));
                float v2028_data = s0[39];
                float v2030_data = ir6[3];
                ir6[3] = (v2030_data + (v2012_data * v2028_data));
                float v2033_data = s0[50];
                float v2035_data = ir6[4];
                ir6[4] = (v2035_data + (v2012_data * v2033_data));
                float v2038_data = s0[62];
                float v2040_data = ir6[5];
                ir6[5] = (v2040_data + (v2012_data * v2038_data));
                float v2043_data = s0[77];
                float v2045_data = ir6[6];
                ir6[6] = (v2045_data + (v2012_data * v2043_data));
                float v2048_data = s0[80];
                float v2050_data = ir6[7];
                ir6[7] = (v2050_data + (v2012_data * v2048_data));
                float v2053_data = s0[103];
                float v2055_data = ir6[8];
                ir6[8] = (v2055_data + (v2012_data * v2053_data));
                float v2058_data = s0[107];
                float v2060_data = ir6[9];
                ir6[9] = (v2060_data + (v2012_data * v2058_data));
                float v2063_data = s0[126];
                float v2065_data = ir6[10];
                ir6[10] = (v2065_data + (v2012_data * v2063_data));
                float v2068_data = s0[141];
                float v2070_data = ir6[11];
                ir6[11] = (v2070_data + (v2012_data * v2068_data));
              }
              if (v15_lead < 12) {
                float v2076_data = r5[2];
                float v2077_data = s0[2];
                float v2079_data = ir6[0];
                ir6[0] = (v2079_data + (v2076_data * v2077_data));
                float v2082_data = s0[14];
                float v2084_data = ir6[1];
                ir6[1] = (v2084_data + (v2076_data * v2082_data));
                float v2087_data = s0[27];
                float v2089_data = ir6[2];
                ir6[2] = (v2089_data + (v2076_data * v2087_data));
                float v2092_data = s0[36];
                float v2094_data = ir6[3];
                ir6[3] = (v2094_data + (v2076_data * v2092_data));
                float v2097_data = s0[49];
                float v2099_data = ir6[4];
                ir6[4] = (v2099_data + (v2076_data * v2097_data));
                float v2102_data = s0[61];
                float v2104_data = ir6[5];
                ir6[5] = (v2104_data + (v2076_data * v2102_data));
                float v2107_data = s0[78];
                float v2109_data = ir6[6];
                ir6[6] = (v2109_data + (v2076_data * v2107_data));
                float v2112_data = s0[83];
                float v2114_data = ir6[7];
                ir6[7] = (v2114_data + (v2076_data * v2112_data));
                float v2117_data = s0[100];
                float v2119_data = ir6[8];
                ir6[8] = (v2119_data + (v2076_data * v2117_data));
                float v2122_data = s0[104];
                float v2124_data = ir6[9];
                ir6[9] = (v2124_data + (v2076_data * v2122_data));
                float v2127_data = s0[125];
                float v2129_data = ir6[10];
                ir6[10] = (v2129_data + (v2076_data * v2127_data));
                float v2132_data = s0[142];
                float v2134_data = ir6[11];
                ir6[11] = (v2134_data + (v2076_data * v2132_data));
              }
              if (v15_lead < 12) {
                float v2140_data = r5[3];
                float v2141_data = s0[3];
                float v2143_data = ir6[0];
                ir6[0] = (v2143_data + (v2140_data * v2141_data));
                float v2146_data = s0[15];
                float v2148_data = ir6[1];
                ir6[1] = (v2148_data + (v2140_data * v2146_data));
                float v2151_data = s0[26];
                float v2153_data = ir6[2];
                ir6[2] = (v2153_data + (v2140_data * v2151_data));
                float v2156_data = s0[37];
                float v2158_data = ir6[3];
                ir6[3] = (v2158_data + (v2140_data * v2156_data));
                float v2161_data = s0[48];
                float v2163_data = ir6[4];
                ir6[4] = (v2163_data + (v2140_data * v2161_data));
                float v2166_data = s0[60];
                float v2168_data = ir6[5];
                ir6[5] = (v2168_data + (v2140_data * v2166_data));
                float v2171_data = s0[79];
                float v2173_data = ir6[6];
                ir6[6] = (v2173_data + (v2140_data * v2171_data));
                float v2176_data = s0[82];
                float v2178_data = ir6[7];
                ir6[7] = (v2178_data + (v2140_data * v2176_data));
                float v2181_data = s0[101];
                float v2183_data = ir6[8];
                ir6[8] = (v2183_data + (v2140_data * v2181_data));
                float v2186_data = s0[105];
                float v2188_data = ir6[9];
                ir6[9] = (v2188_data + (v2140_data * v2186_data));
                float v2191_data = s0[124];
                float v2193_data = ir6[10];
                ir6[10] = (v2193_data + (v2140_data * v2191_data));
                float v2196_data = s0[143];
                float v2198_data = ir6[11];
                ir6[11] = (v2198_data + (v2140_data * v2196_data));
              }
              if (v15_lead < 12) {
                float v2204_data = r5[4];
                float v2205_data = s0[4];
                float v2207_data = ir6[0];
                ir6[0] = (v2207_data + (v2204_data * v2205_data));
                float v2210_data = s0[17];
                float v2212_data = ir6[1];
                ir6[1] = (v2212_data + (v2204_data * v2210_data));
                float v2215_data = s0[29];
                float v2217_data = ir6[2];
                ir6[2] = (v2217_data + (v2204_data * v2215_data));
                float v2220_data = s0[42];
                float v2222_data = ir6[3];
                ir6[3] = (v2222_data + (v2204_data * v2220_data));
                float v2225_data = s0[55];
                float v2227_data = ir6[4];
                ir6[4] = (v2227_data + (v2204_data * v2225_data));
                float v2230_data = s0[68];
                float v2232_data = ir6[5];
                ir6[5] = (v2232_data + (v2204_data * v2230_data));
                float v2235_data = s0[72];
                float v2237_data = ir6[6];
                ir6[6] = (v2237_data + (v2204_data * v2235_data));
                float v2240_data = s0[93];
                float v2242_data = ir6[7];
                ir6[7] = (v2242_data + (v2204_data * v2240_data));
                float v2245_data = s0[98];
                float v2247_data = ir6[8];
                ir6[8] = (v2247_data + (v2204_data * v2245_data));
                float v2250_data = s0[119];
                float v2252_data = ir6[9];
                ir6[9] = (v2252_data + (v2204_data * v2250_data));
                float v2255_data = s0[123];
                float v2257_data = ir6[10];
                ir6[10] = (v2257_data + (v2204_data * v2255_data));
                float v2260_data = s0[128];
                float v2262_data = ir6[11];
                ir6[11] = (v2262_data + (v2204_data * v2260_data));
              }
              if (v15_lead < 12) {
                float v2268_data = r5[5];
                float v2269_data = s0[5];
                float v2271_data = ir6[0];
                ir6[0] = (v2271_data + (v2268_data * v2269_data));
                float v2274_data = s0[16];
                float v2276_data = ir6[1];
                ir6[1] = (v2276_data + (v2268_data * v2274_data));
                float v2279_data = s0[28];
                float v2281_data = ir6[2];
                ir6[2] = (v2281_data + (v2268_data * v2279_data));
                float v2284_data = s0[43];
                float v2286_data = ir6[3];
                ir6[3] = (v2286_data + (v2268_data * v2284_data));
                float v2289_data = s0[54];
                float v2291_data = ir6[4];
                ir6[4] = (v2291_data + (v2268_data * v2289_data));
                float v2294_data = s0[69];
                float v2296_data = ir6[5];
                ir6[5] = (v2296_data + (v2268_data * v2294_data));
                float v2299_data = s0[73];
                float v2301_data = ir6[6];
                ir6[6] = (v2301_data + (v2268_data * v2299_data));
                float v2304_data = s0[92];
                float v2306_data = ir6[7];
                ir6[7] = (v2306_data + (v2268_data * v2304_data));
                float v2309_data = s0[99];
                float v2311_data = ir6[8];
                ir6[8] = (v2311_data + (v2268_data * v2309_data));
                float v2314_data = s0[118];
                float v2316_data = ir6[9];
                ir6[9] = (v2316_data + (v2268_data * v2314_data));
                float v2319_data = s0[122];
                float v2321_data = ir6[10];
                ir6[10] = (v2321_data + (v2268_data * v2319_data));
                float v2324_data = s0[129];
                float v2326_data = ir6[11];
                ir6[11] = (v2326_data + (v2268_data * v2324_data));
              }
              if (v15_lead < 12) {
                float v2332_data = r5[6];
                float v2333_data = s0[6];
                float v2335_data = ir6[0];
                ir6[0] = (v2335_data + (v2332_data * v2333_data));
                float v2338_data = s0[19];
                float v2340_data = ir6[1];
                ir6[1] = (v2340_data + (v2332_data * v2338_data));
                float v2343_data = s0[31];
                float v2345_data = ir6[2];
                ir6[2] = (v2345_data + (v2332_data * v2343_data));
                float v2348_data = s0[40];
                float v2350_data = ir6[3];
                ir6[3] = (v2350_data + (v2332_data * v2348_data));
                float v2353_data = s0[53];
                float v2355_data = ir6[4];
                ir6[4] = (v2355_data + (v2332_data * v2353_data));
                float v2358_data = s0[70];
                float v2360_data = ir6[5];
                ir6[5] = (v2360_data + (v2332_data * v2358_data));
                float v2363_data = s0[74];
                float v2365_data = ir6[6];
                ir6[6] = (v2365_data + (v2332_data * v2363_data));
                float v2368_data = s0[95];
                float v2370_data = ir6[7];
                ir6[7] = (v2370_data + (v2332_data * v2368_data));
                float v2373_data = s0[96];
                float v2375_data = ir6[8];
                ir6[8] = (v2375_data + (v2332_data * v2373_data));
                float v2378_data = s0[117];
                float v2380_data = ir6[9];
                ir6[9] = (v2380_data + (v2332_data * v2378_data));
                float v2383_data = s0[121];
                float v2385_data = ir6[10];
                ir6[10] = (v2385_data + (v2332_data * v2383_data));
                float v2388_data = s0[130];
                float v2390_data = ir6[11];
                ir6[11] = (v2390_data + (v2332_data * v2388_data));
              }
              if (v15_lead < 12) {
                float v2396_data = r5[7];
                float v2397_data = s0[7];
                float v2399_data = ir6[0];
                ir6[0] = (v2399_data + (v2396_data * v2397_data));
                float v2402_data = s0[18];
                float v2404_data = ir6[1];
                ir6[1] = (v2404_data + (v2396_data * v2402_data));
                float v2407_data = s0[30];
                float v2409_data = ir6[2];
                ir6[2] = (v2409_data + (v2396_data * v2407_data));
                float v2412_data = s0[41];
                float v2414_data = ir6[3];
                ir6[3] = (v2414_data + (v2396_data * v2412_data));
                float v2417_data = s0[52];
                float v2419_data = ir6[4];
                ir6[4] = (v2419_data + (v2396_data * v2417_data));
                float v2422_data = s0[71];
                float v2424_data = ir6[5];
                ir6[5] = (v2424_data + (v2396_data * v2422_data));
                float v2427_data = s0[75];
                float v2429_data = ir6[6];
                ir6[6] = (v2429_data + (v2396_data * v2427_data));
                float v2432_data = s0[94];
                float v2434_data = ir6[7];
                ir6[7] = (v2434_data + (v2396_data * v2432_data));
                float v2437_data = s0[97];
                float v2439_data = ir6[8];
                ir6[8] = (v2439_data + (v2396_data * v2437_data));
                float v2442_data = s0[116];
                float v2444_data = ir6[9];
                ir6[9] = (v2444_data + (v2396_data * v2442_data));
                float v2447_data = s0[120];
                float v2449_data = ir6[10];
                ir6[10] = (v2449_data + (v2396_data * v2447_data));
                float v2452_data = s0[131];
                float v2454_data = ir6[11];
                ir6[11] = (v2454_data + (v2396_data * v2452_data));
              }
              if (v15_lead < 12) {
                float v2460_data = r5[8];
                float v2461_data = s0[8];
                float v2463_data = ir6[0];
                ir6[0] = (v2463_data + (v2460_data * v2461_data));
                float v2466_data = s0[21];
                float v2468_data = ir6[1];
                ir6[1] = (v2468_data + (v2460_data * v2466_data));
                float v2471_data = s0[34];
                float v2473_data = ir6[2];
                ir6[2] = (v2473_data + (v2460_data * v2471_data));
                float v2476_data = s0[46];
                float v2478_data = ir6[3];
                ir6[3] = (v2478_data + (v2460_data * v2476_data));
                float v2481_data = s0[59];
                float v2483_data = ir6[4];
                ir6[4] = (v2483_data + (v2460_data * v2481_data));
                float v2486_data = s0[64];
                float v2488_data = ir6[5];
                ir6[5] = (v2488_data + (v2460_data * v2486_data));
                float v2491_data = s0[85];
                float v2493_data = ir6[6];
                ir6[6] = (v2493_data + (v2460_data * v2491_data));
                float v2496_data = s0[89];
                float v2498_data = ir6[7];
                ir6[7] = (v2498_data + (v2460_data * v2496_data));
                float v2501_data = s0[110];
                float v2503_data = ir6[8];
                ir6[8] = (v2503_data + (v2460_data * v2501_data));
                float v2506_data = s0[115];
                float v2508_data = ir6[9];
                ir6[9] = (v2508_data + (v2460_data * v2506_data));
                float v2511_data = s0[136];
                float v2513_data = ir6[10];
                ir6[10] = (v2513_data + (v2460_data * v2511_data));
                float v2516_data = s0[132];
                float v2518_data = ir6[11];
                ir6[11] = (v2518_data + (v2460_data * v2516_data));
              }
              if (v15_lead < 12) {
                float v2524_data = r5[9];
                float v2525_data = s0[9];
                float v2527_data = ir6[0];
                ir6[0] = (v2527_data + (v2524_data * v2525_data));
                float v2530_data = s0[20];
                float v2532_data = ir6[1];
                ir6[1] = (v2532_data + (v2524_data * v2530_data));
                float v2535_data = s0[35];
                float v2537_data = ir6[2];
                ir6[2] = (v2537_data + (v2524_data * v2535_data));
                float v2540_data = s0[47];
                float v2542_data = ir6[3];
                ir6[3] = (v2542_data + (v2524_data * v2540_data));
                float v2545_data = s0[58];
                float v2547_data = ir6[4];
                ir6[4] = (v2547_data + (v2524_data * v2545_data));
                float v2550_data = s0[65];
                float v2552_data = ir6[5];
                ir6[5] = (v2552_data + (v2524_data * v2550_data));
                float v2555_data = s0[84];
                float v2557_data = ir6[6];
                ir6[6] = (v2557_data + (v2524_data * v2555_data));
                float v2560_data = s0[88];
                float v2562_data = ir6[7];
                ir6[7] = (v2562_data + (v2524_data * v2560_data));
                float v2565_data = s0[111];
                float v2567_data = ir6[8];
                ir6[8] = (v2567_data + (v2524_data * v2565_data));
                float v2570_data = s0[114];
                float v2572_data = ir6[9];
                ir6[9] = (v2572_data + (v2524_data * v2570_data));
                float v2575_data = s0[137];
                float v2577_data = ir6[10];
                ir6[10] = (v2577_data + (v2524_data * v2575_data));
                float v2580_data = s0[133];
                float v2582_data = ir6[11];
                ir6[11] = (v2582_data + (v2524_data * v2580_data));
              }
              if (v15_lead < 12) {
                float v2588_data = r5[10];
                float v2589_data = s0[10];
                float v2591_data = ir6[0];
                ir6[0] = (v2591_data + (v2588_data * v2589_data));
                float v2594_data = s0[23];
                float v2596_data = ir6[1];
                ir6[1] = (v2596_data + (v2588_data * v2594_data));
                float v2599_data = s0[32];
                float v2601_data = ir6[2];
                ir6[2] = (v2601_data + (v2588_data * v2599_data));
                float v2604_data = s0[44];
                float v2606_data = ir6[3];
                ir6[3] = (v2606_data + (v2588_data * v2604_data));
                float v2609_data = s0[57];
                float v2611_data = ir6[4];
                ir6[4] = (v2611_data + (v2588_data * v2609_data));
                float v2614_data = s0[66];
                float v2616_data = ir6[5];
                ir6[5] = (v2616_data + (v2588_data * v2614_data));
                float v2619_data = s0[87];
                float v2621_data = ir6[6];
                ir6[6] = (v2621_data + (v2588_data * v2619_data));
                float v2624_data = s0[91];
                float v2626_data = ir6[7];
                ir6[7] = (v2626_data + (v2588_data * v2624_data));
                float v2629_data = s0[108];
                float v2631_data = ir6[8];
                ir6[8] = (v2631_data + (v2588_data * v2629_data));
                float v2634_data = s0[113];
                float v2636_data = ir6[9];
                ir6[9] = (v2636_data + (v2588_data * v2634_data));
                float v2639_data = s0[138];
                float v2641_data = ir6[10];
                ir6[10] = (v2641_data + (v2588_data * v2639_data));
                float v2644_data = s0[134];
                float v2646_data = ir6[11];
                ir6[11] = (v2646_data + (v2588_data * v2644_data));
              }
              if (v15_lead < 12) {
                float v2652_data = r5[11];
                float v2653_data = s0[11];
                float v2655_data = ir6[0];
                ir6[0] = (v2655_data + (v2652_data * v2653_data));
                float v2658_data = s0[22];
                float v2660_data = ir6[1];
                ir6[1] = (v2660_data + (v2652_data * v2658_data));
                float v2663_data = s0[33];
                float v2665_data = ir6[2];
                ir6[2] = (v2665_data + (v2652_data * v2663_data));
                float v2668_data = s0[45];
                float v2670_data = ir6[3];
                ir6[3] = (v2670_data + (v2652_data * v2668_data));
                float v2673_data = s0[56];
                float v2675_data = ir6[4];
                ir6[4] = (v2675_data + (v2652_data * v2673_data));
                float v2678_data = s0[67];
                float v2680_data = ir6[5];
                ir6[5] = (v2680_data + (v2652_data * v2678_data));
                float v2683_data = s0[86];
                float v2685_data = ir6[6];
                ir6[6] = (v2685_data + (v2652_data * v2683_data));
                float v2688_data = s0[90];
                float v2690_data = ir6[7];
                ir6[7] = (v2690_data + (v2652_data * v2688_data));
                float v2693_data = s0[109];
                float v2695_data = ir6[8];
                ir6[8] = (v2695_data + (v2652_data * v2693_data));
                float v2698_data = s0[112];
                float v2700_data = ir6[9];
                ir6[9] = (v2700_data + (v2652_data * v2698_data));
                float v2703_data = s0[139];
                float v2705_data = ir6[10];
                ir6[10] = (v2705_data + (v2652_data * v2703_data));
                float v2708_data = s0[135];
                float v2710_data = ir6[11];
                ir6[11] = (v2710_data + (v2652_data * v2708_data));
              }
              if (v15_lead < 12) {
                #pragma unroll
                for (int32_t v2716_n1 = 0; v2716_n1 < 12; ++v2716_n1) {
                  float v2718_data = ir6[v2716_n1];
                  r6[v2716_n1] = v2718_data;
                }
              }
              // glb_m3 = store{r>g}(r6);
              if (v15_lead < 12) {
                #pragma unroll
                for (int32_t v2724_i1 = 0; v2724_i1 < 12; ++v2724_i1) {
                  float v2726_data = r6[v2724_i1];
                  glb_m3[(v15_lead + (v2724_i1 * 12))] = v2726_data;
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


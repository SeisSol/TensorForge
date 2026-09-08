// === base name ===
kernel_4b748443ff

// === header ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4b748443ff(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4b748443ff(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8(8) {0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // OUT = +(TMP, dims=[1])
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
              float *const __restrict__ glb_m2 = &m2[batchId0 * 8 + 0 + m2_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 8) {
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 8; ++v10_i1) {
                  float v18_data = glb_m0[(v8_lead + (v10_i1 * 8))];
                  r0[v10_i1] = v18_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v8_lead < 8) {
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                  float v33_data = glb_m1[(v8_lead + (v25_i1 * 8))];
                  r1[v25_i1] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              if (v8_lead < 8) {
                float v40_data = r0[0];
                float v41_data = r1[0];
                float v44_data = r2[0];
                r2[0] = (v44_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 0))));
                float v47_data = r1[1];
                float v50_data = r2[1];
                r2[1] = (v50_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 0))));
                float v53_data = r1[2];
                float v56_data = r2[2];
                r2[2] = (v56_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 0))));
                float v59_data = r1[3];
                float v62_data = r2[3];
                r2[3] = (v62_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 0))));
                float v65_data = r1[4];
                float v68_data = r2[4];
                r2[4] = (v68_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 0))));
                float v71_data = r1[5];
                float v74_data = r2[5];
                r2[5] = (v74_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
                float v77_data = r1[6];
                float v80_data = r2[6];
                r2[6] = (v80_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 0))));
                float v83_data = r1[7];
                float v86_data = r2[7];
                r2[7] = (v86_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 0))));
              }
              if (v8_lead < 8) {
                float v92_data = r0[1];
                float v93_data = r1[0];
                float v96_data = r2[0];
                r2[0] = (v96_data + (v92_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 1))));
                float v99_data = r1[1];
                float v102_data = r2[1];
                r2[1] = (v102_data + (v92_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 1))));
                float v105_data = r1[2];
                float v108_data = r2[2];
                r2[2] = (v108_data + (v92_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 1))));
                float v111_data = r1[3];
                float v114_data = r2[3];
                r2[3] = (v114_data + (v92_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 1))));
                float v117_data = r1[4];
                float v120_data = r2[4];
                r2[4] = (v120_data + (v92_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 1))));
                float v123_data = r1[5];
                float v126_data = r2[5];
                r2[5] = (v126_data + (v92_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 1))));
                float v129_data = r1[6];
                float v132_data = r2[6];
                r2[6] = (v132_data + (v92_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 1))));
                float v135_data = r1[7];
                float v138_data = r2[7];
                r2[7] = (v138_data + (v92_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 1))));
              }
              if (v8_lead < 8) {
                float v144_data = r0[2];
                float v145_data = r1[0];
                float v148_data = r2[0];
                r2[0] = (v148_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 2))));
                float v151_data = r1[1];
                float v154_data = r2[1];
                r2[1] = (v154_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 2))));
                float v157_data = r1[2];
                float v160_data = r2[2];
                r2[2] = (v160_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 2))));
                float v163_data = r1[3];
                float v166_data = r2[3];
                r2[3] = (v166_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 2))));
                float v169_data = r1[4];
                float v172_data = r2[4];
                r2[4] = (v172_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 2))));
                float v175_data = r1[5];
                float v178_data = r2[5];
                r2[5] = (v178_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 2))));
                float v181_data = r1[6];
                float v184_data = r2[6];
                r2[6] = (v184_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 2))));
                float v187_data = r1[7];
                float v190_data = r2[7];
                r2[7] = (v190_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 2))));
              }
              if (v8_lead < 8) {
                float v196_data = r0[3];
                float v197_data = r1[0];
                float v200_data = r2[0];
                r2[0] = (v200_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 3))));
                float v203_data = r1[1];
                float v206_data = r2[1];
                r2[1] = (v206_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 3))));
                float v209_data = r1[2];
                float v212_data = r2[2];
                r2[2] = (v212_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v209_data, 3))));
                float v215_data = r1[3];
                float v218_data = r2[3];
                r2[3] = (v218_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v215_data, 3))));
                float v221_data = r1[4];
                float v224_data = r2[4];
                r2[4] = (v224_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v221_data, 3))));
                float v227_data = r1[5];
                float v230_data = r2[5];
                r2[5] = (v230_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v227_data, 3))));
                float v233_data = r1[6];
                float v236_data = r2[6];
                r2[6] = (v236_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v233_data, 3))));
                float v239_data = r1[7];
                float v242_data = r2[7];
                r2[7] = (v242_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 3))));
              }
              if (v8_lead < 8) {
                float v248_data = r0[4];
                float v249_data = r1[0];
                float v252_data = r2[0];
                r2[0] = (v252_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v249_data, 4))));
                float v255_data = r1[1];
                float v258_data = r2[1];
                r2[1] = (v258_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v255_data, 4))));
                float v261_data = r1[2];
                float v264_data = r2[2];
                r2[2] = (v264_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v261_data, 4))));
                float v267_data = r1[3];
                float v270_data = r2[3];
                r2[3] = (v270_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v267_data, 4))));
                float v273_data = r1[4];
                float v276_data = r2[4];
                r2[4] = (v276_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v273_data, 4))));
                float v279_data = r1[5];
                float v282_data = r2[5];
                r2[5] = (v282_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 4))));
                float v285_data = r1[6];
                float v288_data = r2[6];
                r2[6] = (v288_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v285_data, 4))));
                float v291_data = r1[7];
                float v294_data = r2[7];
                r2[7] = (v294_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 4))));
              }
              if (v8_lead < 8) {
                float v300_data = r0[5];
                float v301_data = r1[0];
                float v304_data = r2[0];
                r2[0] = (v304_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v301_data, 5))));
                float v307_data = r1[1];
                float v310_data = r2[1];
                r2[1] = (v310_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v307_data, 5))));
                float v313_data = r1[2];
                float v316_data = r2[2];
                r2[2] = (v316_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v313_data, 5))));
                float v319_data = r1[3];
                float v322_data = r2[3];
                r2[3] = (v322_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v319_data, 5))));
                float v325_data = r1[4];
                float v328_data = r2[4];
                r2[4] = (v328_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v325_data, 5))));
                float v331_data = r1[5];
                float v334_data = r2[5];
                r2[5] = (v334_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v331_data, 5))));
                float v337_data = r1[6];
                float v340_data = r2[6];
                r2[6] = (v340_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 5))));
                float v343_data = r1[7];
                float v346_data = r2[7];
                r2[7] = (v346_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 5))));
              }
              if (v8_lead < 8) {
                float v352_data = r0[6];
                float v353_data = r1[0];
                float v356_data = r2[0];
                r2[0] = (v356_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v353_data, 6))));
                float v359_data = r1[1];
                float v362_data = r2[1];
                r2[1] = (v362_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v359_data, 6))));
                float v365_data = r1[2];
                float v368_data = r2[2];
                r2[2] = (v368_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 6))));
                float v371_data = r1[3];
                float v374_data = r2[3];
                r2[3] = (v374_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v371_data, 6))));
                float v377_data = r1[4];
                float v380_data = r2[4];
                r2[4] = (v380_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v377_data, 6))));
                float v383_data = r1[5];
                float v386_data = r2[5];
                r2[5] = (v386_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v383_data, 6))));
                float v389_data = r1[6];
                float v392_data = r2[6];
                r2[6] = (v392_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 6))));
                float v395_data = r1[7];
                float v398_data = r2[7];
                r2[7] = (v398_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 6))));
              }
              if (v8_lead < 8) {
                float v404_data = r0[7];
                float v405_data = r1[0];
                float v408_data = r2[0];
                r2[0] = (v408_data + (v404_data * (sycl::group_broadcast(item.get_sub_group(), v405_data, 7))));
                float v411_data = r1[1];
                float v414_data = r2[1];
                r2[1] = (v414_data + (v404_data * (sycl::group_broadcast(item.get_sub_group(), v411_data, 7))));
                float v417_data = r1[2];
                float v420_data = r2[2];
                r2[2] = (v420_data + (v404_data * (sycl::group_broadcast(item.get_sub_group(), v417_data, 7))));
                float v423_data = r1[3];
                float v426_data = r2[3];
                r2[3] = (v426_data + (v404_data * (sycl::group_broadcast(item.get_sub_group(), v423_data, 7))));
                float v429_data = r1[4];
                float v432_data = r2[4];
                r2[4] = (v432_data + (v404_data * (sycl::group_broadcast(item.get_sub_group(), v429_data, 7))));
                float v435_data = r1[5];
                float v438_data = r2[5];
                r2[5] = (v438_data + (v404_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 7))));
                float v441_data = r1[6];
                float v444_data = r2[6];
                r2[6] = (v444_data + (v404_data * (sycl::group_broadcast(item.get_sub_group(), v441_data, 7))));
                float v447_data = r1[7];
                float v450_data = r2[7];
                r2[7] = (v450_data + (v404_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 7))));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = store{r>s}(localShrMem0, r2);
              if (v8_lead < 8) {
                #pragma unroll
                for (int32_t v457_i1 = 0; v457_i1 < 8; ++v457_i1) {
                  float v459_data = r2[v457_i1];
                  int32_t v466_a = v8_lead + (v457_i1 * 8);
                  s0[(v466_a ^ ((v466_a >> 5) & 31))] = v459_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m2 = +(s0, dims=[1])
              if (v8_lead < 8) {
                float v475_acc0 = 0.0f;
                #pragma unroll
                for (int32_t v474_r1 = 0; v474_r1 < 8; ++v474_r1) {
                  int32_t v482_a = v8_lead + (v474_r1 * 8);
                  float v486_data = s0[(v482_a ^ ((v482_a >> 5) & 31))];
                  v475_acc0 = (v475_acc0 + v486_data);
                }
                glb_m2[v8_lead] = v475_acc0;
              }
            }
          }
        }
      });
    }
  });
}


// === base name ===
kernel_12cc8348c5773f12

// === header ===
void launcher_kernel_12cc8348c5773f12(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_12cc8348c5773f12(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_12cc8348c5773f12(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_12cc8348c5773f12(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(M)
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v13_lead = item.get_local_id(0) % 16;
              if (v13_lead < 8) {
                #pragma unroll
                for (int32_t v15_i1 = 0; v15_i1 < 8; ++v15_i1) {
                  float v23_data = glb_m1[(v13_lead + (v15_i1 * 8))];
                  r0[v15_i1] = v23_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              if (v13_lead < 8) {
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
                  float v38_data = glb_m2[(v13_lead + (v30_i1 * 8))];
                  r1[v30_i1] = v38_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir2[8]{};
              if (v13_lead < 8) {
                float v46_data = r0[0];
                float v47_data = r1[0];
                float v50_data = ir2[0];
                ir2[0] = (v50_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 0))));
                float v53_data = r1[1];
                float v56_data = ir2[1];
                ir2[1] = (v56_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 0))));
                float v59_data = r1[2];
                float v62_data = ir2[2];
                ir2[2] = (v62_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 0))));
                float v65_data = r1[3];
                float v68_data = ir2[3];
                ir2[3] = (v68_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 0))));
                float v71_data = r1[4];
                float v74_data = ir2[4];
                ir2[4] = (v74_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
                float v77_data = r1[5];
                float v80_data = ir2[5];
                ir2[5] = (v80_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 0))));
                float v83_data = r1[6];
                float v86_data = ir2[6];
                ir2[6] = (v86_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 0))));
                float v89_data = r1[7];
                float v92_data = ir2[7];
                ir2[7] = (v92_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 0))));
              }
              if (v13_lead < 8) {
                float v98_data = r0[1];
                float v99_data = r1[0];
                float v102_data = ir2[0];
                ir2[0] = (v102_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 1))));
                float v105_data = r1[1];
                float v108_data = ir2[1];
                ir2[1] = (v108_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 1))));
                float v111_data = r1[2];
                float v114_data = ir2[2];
                ir2[2] = (v114_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 1))));
                float v117_data = r1[3];
                float v120_data = ir2[3];
                ir2[3] = (v120_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 1))));
                float v123_data = r1[4];
                float v126_data = ir2[4];
                ir2[4] = (v126_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 1))));
                float v129_data = r1[5];
                float v132_data = ir2[5];
                ir2[5] = (v132_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 1))));
                float v135_data = r1[6];
                float v138_data = ir2[6];
                ir2[6] = (v138_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 1))));
                float v141_data = r1[7];
                float v144_data = ir2[7];
                ir2[7] = (v144_data + (v98_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 1))));
              }
              if (v13_lead < 8) {
                float v150_data = r0[2];
                float v151_data = r1[0];
                float v154_data = ir2[0];
                ir2[0] = (v154_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 2))));
                float v157_data = r1[1];
                float v160_data = ir2[1];
                ir2[1] = (v160_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 2))));
                float v163_data = r1[2];
                float v166_data = ir2[2];
                ir2[2] = (v166_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 2))));
                float v169_data = r1[3];
                float v172_data = ir2[3];
                ir2[3] = (v172_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 2))));
                float v175_data = r1[4];
                float v178_data = ir2[4];
                ir2[4] = (v178_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 2))));
                float v181_data = r1[5];
                float v184_data = ir2[5];
                ir2[5] = (v184_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 2))));
                float v187_data = r1[6];
                float v190_data = ir2[6];
                ir2[6] = (v190_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 2))));
                float v193_data = r1[7];
                float v196_data = ir2[7];
                ir2[7] = (v196_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 2))));
              }
              if (v13_lead < 8) {
                float v202_data = r0[3];
                float v203_data = r1[0];
                float v206_data = ir2[0];
                ir2[0] = (v206_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 3))));
                float v209_data = r1[1];
                float v212_data = ir2[1];
                ir2[1] = (v212_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v209_data, 3))));
                float v215_data = r1[2];
                float v218_data = ir2[2];
                ir2[2] = (v218_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v215_data, 3))));
                float v221_data = r1[3];
                float v224_data = ir2[3];
                ir2[3] = (v224_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v221_data, 3))));
                float v227_data = r1[4];
                float v230_data = ir2[4];
                ir2[4] = (v230_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v227_data, 3))));
                float v233_data = r1[5];
                float v236_data = ir2[5];
                ir2[5] = (v236_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v233_data, 3))));
                float v239_data = r1[6];
                float v242_data = ir2[6];
                ir2[6] = (v242_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 3))));
                float v245_data = r1[7];
                float v248_data = ir2[7];
                ir2[7] = (v248_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 3))));
              }
              if (v13_lead < 8) {
                float v254_data = r0[4];
                float v255_data = r1[0];
                float v258_data = ir2[0];
                ir2[0] = (v258_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v255_data, 4))));
                float v261_data = r1[1];
                float v264_data = ir2[1];
                ir2[1] = (v264_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v261_data, 4))));
                float v267_data = r1[2];
                float v270_data = ir2[2];
                ir2[2] = (v270_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v267_data, 4))));
                float v273_data = r1[3];
                float v276_data = ir2[3];
                ir2[3] = (v276_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v273_data, 4))));
                float v279_data = r1[4];
                float v282_data = ir2[4];
                ir2[4] = (v282_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 4))));
                float v285_data = r1[5];
                float v288_data = ir2[5];
                ir2[5] = (v288_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v285_data, 4))));
                float v291_data = r1[6];
                float v294_data = ir2[6];
                ir2[6] = (v294_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 4))));
                float v297_data = r1[7];
                float v300_data = ir2[7];
                ir2[7] = (v300_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v297_data, 4))));
              }
              if (v13_lead < 8) {
                float v306_data = r0[5];
                float v307_data = r1[0];
                float v310_data = ir2[0];
                ir2[0] = (v310_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v307_data, 5))));
                float v313_data = r1[1];
                float v316_data = ir2[1];
                ir2[1] = (v316_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v313_data, 5))));
                float v319_data = r1[2];
                float v322_data = ir2[2];
                ir2[2] = (v322_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v319_data, 5))));
                float v325_data = r1[3];
                float v328_data = ir2[3];
                ir2[3] = (v328_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v325_data, 5))));
                float v331_data = r1[4];
                float v334_data = ir2[4];
                ir2[4] = (v334_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v331_data, 5))));
                float v337_data = r1[5];
                float v340_data = ir2[5];
                ir2[5] = (v340_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 5))));
                float v343_data = r1[6];
                float v346_data = ir2[6];
                ir2[6] = (v346_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 5))));
                float v349_data = r1[7];
                float v352_data = ir2[7];
                ir2[7] = (v352_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 5))));
              }
              if (v13_lead < 8) {
                float v358_data = r0[6];
                float v359_data = r1[0];
                float v362_data = ir2[0];
                ir2[0] = (v362_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v359_data, 6))));
                float v365_data = r1[1];
                float v368_data = ir2[1];
                ir2[1] = (v368_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 6))));
                float v371_data = r1[2];
                float v374_data = ir2[2];
                ir2[2] = (v374_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v371_data, 6))));
                float v377_data = r1[3];
                float v380_data = ir2[3];
                ir2[3] = (v380_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v377_data, 6))));
                float v383_data = r1[4];
                float v386_data = ir2[4];
                ir2[4] = (v386_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v383_data, 6))));
                float v389_data = r1[5];
                float v392_data = ir2[5];
                ir2[5] = (v392_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 6))));
                float v395_data = r1[6];
                float v398_data = ir2[6];
                ir2[6] = (v398_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 6))));
                float v401_data = r1[7];
                float v404_data = ir2[7];
                ir2[7] = (v404_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 6))));
              }
              if (v13_lead < 8) {
                float v410_data = r0[7];
                float v411_data = r1[0];
                float v414_data = ir2[0];
                ir2[0] = (v414_data + (v410_data * (sycl::group_broadcast(item.get_sub_group(), v411_data, 7))));
                float v417_data = r1[1];
                float v420_data = ir2[1];
                ir2[1] = (v420_data + (v410_data * (sycl::group_broadcast(item.get_sub_group(), v417_data, 7))));
                float v423_data = r1[2];
                float v426_data = ir2[2];
                ir2[2] = (v426_data + (v410_data * (sycl::group_broadcast(item.get_sub_group(), v423_data, 7))));
                float v429_data = r1[3];
                float v432_data = ir2[3];
                ir2[3] = (v432_data + (v410_data * (sycl::group_broadcast(item.get_sub_group(), v429_data, 7))));
                float v435_data = r1[4];
                float v438_data = ir2[4];
                ir2[4] = (v438_data + (v410_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 7))));
                float v441_data = r1[5];
                float v444_data = ir2[5];
                ir2[5] = (v444_data + (v410_data * (sycl::group_broadcast(item.get_sub_group(), v441_data, 7))));
                float v447_data = r1[6];
                float v450_data = ir2[6];
                ir2[6] = (v450_data + (v410_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 7))));
                float v453_data = r1[7];
                float v456_data = ir2[7];
                ir2[7] = (v456_data + (v410_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 7))));
              }
              if (v13_lead < 8) {
                #pragma unroll
                for (int32_t v462_n1 = 0; v462_n1 < 8; ++v462_n1) {
                  float v464_data = ir2[v462_n1];
                  r2[v462_n1] = v464_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v13_lead < 8) {
                #pragma unroll
                for (int32_t v470_i1 = 0; v470_i1 < 8; ++v470_i1) {
                  float v472_data = r2[v470_i1];
                  glb_m0[(v13_lead + (v470_i1 * 8))] = v472_data;
                }
              }
              // glb_m3 = abs(glb_m0)
              if (v13_lead < 8) {
                #pragma unroll
                for (int32_t v484_k1 = 0; v484_k1 < 8; ++v484_k1) {
                  int32_t v490_a = v484_k1 * 8;
                  float v492_data = glb_m0[(v13_lead + v490_a)];
                  glb_m3[(v13_lead + v490_a)] = (sycl::fabs(v492_data));
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


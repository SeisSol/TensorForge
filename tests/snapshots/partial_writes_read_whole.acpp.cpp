// === base name ===
kernel_3ea94ad3ce8c615b

// === header ===
void launcher_kernel_3ea94ad3ce8c615b(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3ea94ad3ce8c615b(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3ea94ad3ce8c615b(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3ea94ad3ce8c615b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (288, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×9(32×9) {0..32}×{0..9} pointer_based
        // m1 16×9(16×9) {0..16}×{0..9} pointer_based
        // m2 16×9(16×9) {0..16}×{0..9} pointer_based
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based
        // m4 9×9(9×9) {0..9}×{0..9} pointer_based
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[288 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[288];
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0][0 + m4_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v15_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
                int32_t v22_lead = v15_lead + (v16_i0 * 32);
                #pragma unroll
                for (int32_t v17_i1 = 0; v17_i1 < 9; ++v17_i1) {
                  float v25_data = glb_m0[(v22_lead + (v17_i1 * 32))];
                  r0[(v16_i0 + v17_i1)] = v25_data;
                }
              }
              float r2[9]{};
              // r2 = load{g>r}(glb_m1);
              if (v15_lead < 16) {
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 9; ++v32_i1) {
                  float v40_data = glb_m1[(v15_lead + (v32_i1 * 16))];
                  r2[v32_i1] = v40_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[9]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              float v46_data = r0[0];
              float v47_data = r1[0];
              r1[0] = (v47_data + v46_data);
              float v49_data = r0[1];
              float v50_data = r1[1];
              r1[1] = (v50_data + v49_data);
              float v52_data = r0[2];
              float v53_data = r1[2];
              r1[2] = (v53_data + v52_data);
              float v55_data = r0[3];
              float v56_data = r1[3];
              r1[3] = (v56_data + v55_data);
              float v58_data = r0[4];
              float v59_data = r1[4];
              r1[4] = (v59_data + v58_data);
              float v61_data = r0[5];
              float v62_data = r1[5];
              r1[5] = (v62_data + v61_data);
              float v64_data = r0[6];
              float v65_data = r1[6];
              r1[6] = (v65_data + v64_data);
              float v67_data = r0[7];
              float v68_data = r1[7];
              r1[7] = (v68_data + v67_data);
              float v70_data = r0[8];
              float v71_data = r1[8];
              r1[8] = (v71_data + v70_data);
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v76_i0 = 0; v76_i0 < 1; ++v76_i0) {
                int32_t v84_lead = v15_lead + (v76_i0 * 32);
                #pragma unroll
                for (int32_t v77_i1 = 0; v77_i1 < 9; ++v77_i1) {
                  float v79_data = r1[(v76_i0 + v77_i1)];
                  int32_t v86_a = v84_lead + (v77_i1 * 32);
                  s0[(v86_a ^ ((v86_a >> 5) & 31))] = v79_data;
                }
              }
              float r4[9]{};
              // r4 = load{g>r}(glb_m2);
              if (v15_lead < 16) {
                #pragma unroll
                for (int32_t v95_i1 = 0; v95_i1 < 9; ++v95_i1) {
                  float v103_data = glb_m2[(v15_lead + (v95_i1 * 16))];
                  r4[v95_i1] = v103_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m1););
              float r3[9]{};
              item.barrier();
              // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir3[9]{};
              if (v15_lead < 16) {
                float v111_data = r2[0];
                float v112_data = ir3[0];
                ir3[0] = (v112_data + v111_data);
                float v114_data = r2[1];
                float v115_data = ir3[1];
                ir3[1] = (v115_data + v114_data);
                float v117_data = r2[2];
                float v118_data = ir3[2];
                ir3[2] = (v118_data + v117_data);
                float v120_data = r2[3];
                float v121_data = ir3[3];
                ir3[3] = (v121_data + v120_data);
                float v123_data = r2[4];
                float v124_data = ir3[4];
                ir3[4] = (v124_data + v123_data);
                float v126_data = r2[5];
                float v127_data = ir3[5];
                ir3[5] = (v127_data + v126_data);
                float v129_data = r2[6];
                float v130_data = ir3[6];
                ir3[6] = (v130_data + v129_data);
                float v132_data = r2[7];
                float v133_data = ir3[7];
                ir3[7] = (v133_data + v132_data);
                float v135_data = r2[8];
                float v136_data = ir3[8];
                ir3[8] = (v136_data + v135_data);
              }
              if (v15_lead < 16) {
                #pragma unroll
                for (int32_t v142_n1 = 0; v142_n1 < 9; ++v142_n1) {
                  float v144_data = ir3[v142_n1];
                  int32_t v151_a = v15_lead + (v142_n1 * 32);
                  float v155_data = s0[(v151_a ^ ((v151_a >> 5) & 31))];
                  r3[v142_n1] = (v155_data + v144_data);
                }
              }
              item.barrier();
              // s0 = store{r>s}(localShrMem0, r3);
              if (v15_lead < 16) {
                #pragma unroll
                for (int32_t v162_i1 = 0; v162_i1 < 9; ++v162_i1) {
                  float v164_data = r3[v162_i1];
                  int32_t v171_a = v15_lead + (v162_i1 * 32);
                  s0[(v171_a ^ ((v171_a >> 5) & 31))] = v164_data;
                }
              }
              float r6[9]{};
              // r6 = load{g>r}(glb_m4);
              if (v15_lead < 9) {
                #pragma unroll
                for (int32_t v180_i1 = 0; v180_i1 < 9; ++v180_i1) {
                  float v188_data = glb_m4[(v15_lead + (v180_i1 * 9))];
                  r6[v180_i1] = v188_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m2););
              float r5[9]{};
              item.barrier();
              // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir5[9]{};
              if (v15_lead < 16) {
                float v196_data = r4[0];
                float v197_data = ir5[0];
                ir5[0] = (v197_data + v196_data);
                float v199_data = r4[1];
                float v200_data = ir5[1];
                ir5[1] = (v200_data + v199_data);
                float v202_data = r4[2];
                float v203_data = ir5[2];
                ir5[2] = (v203_data + v202_data);
                float v205_data = r4[3];
                float v206_data = ir5[3];
                ir5[3] = (v206_data + v205_data);
                float v208_data = r4[4];
                float v209_data = ir5[4];
                ir5[4] = (v209_data + v208_data);
                float v211_data = r4[5];
                float v212_data = ir5[5];
                ir5[5] = (v212_data + v211_data);
                float v214_data = r4[6];
                float v215_data = ir5[6];
                ir5[6] = (v215_data + v214_data);
                float v217_data = r4[7];
                float v218_data = ir5[7];
                ir5[7] = (v218_data + v217_data);
                float v220_data = r4[8];
                float v221_data = ir5[8];
                ir5[8] = (v221_data + v220_data);
              }
              if (v15_lead < 16) {
                #pragma unroll
                for (int32_t v227_n1 = 0; v227_n1 < 9; ++v227_n1) {
                  float v229_data = ir5[v227_n1];
                  int32_t v236_a = v15_lead + (v227_n1 * 32);
                  float v240_data = s0[(v236_a ^ ((v236_a >> 5) & 31))];
                  r5[v227_n1] = (v240_data + v229_data);
                }
              }
              item.barrier();
              // s0 = store{r>s}(localShrMem0, r5);
              if (v15_lead < 16) {
                #pragma unroll
                for (int32_t v247_i1 = 0; v247_i1 < 9; ++v247_i1) {
                  float v249_data = r5[v247_i1];
                  int32_t v256_a = v15_lead + (v247_i1 * 32);
                  s0[(v256_a ^ ((v256_a >> 5) & 31))] = v249_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m4););
              float r7[9]{};
              item.barrier();
              // r7 = +(s0 * r6) + None
              // [(0, 32), (0, 9)] [(0, 9)]
              float ir7[9]{};
              float v274_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
              float v275_data = r6[0];
              float v278_data = ir7[0];
              ir7[0] = (v278_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 0))));
              float v289_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
              float v290_data = r6[1];
              float v293_data = ir7[1];
              ir7[1] = (v293_data + (v289_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 0))));
              float v304_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
              float v305_data = r6[2];
              float v308_data = ir7[2];
              ir7[2] = (v308_data + (v304_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 0))));
              float v319_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
              float v320_data = r6[3];
              float v323_data = ir7[3];
              ir7[3] = (v323_data + (v319_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 0))));
              float v334_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
              float v335_data = r6[4];
              float v338_data = ir7[4];
              ir7[4] = (v338_data + (v334_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 0))));
              float v349_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
              float v350_data = r6[5];
              float v353_data = ir7[5];
              ir7[5] = (v353_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 0))));
              float v364_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
              float v365_data = r6[6];
              float v368_data = ir7[6];
              ir7[6] = (v368_data + (v364_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 0))));
              float v379_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
              float v380_data = r6[7];
              float v383_data = ir7[7];
              ir7[7] = (v383_data + (v379_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 0))));
              float v394_data = s0[(v15_lead ^ ((v15_lead >> 5) & 31))];
              float v395_data = r6[8];
              float v398_data = ir7[8];
              ir7[8] = (v398_data + (v394_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 0))));
              int32_t v408_a = v15_lead + 32;
              float v412_data = s0[(v408_a ^ ((v408_a >> 5) & 31))];
              float v416_data = ir7[0];
              ir7[0] = (v416_data + (v412_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 1))));
              int32_t v423_a = v15_lead + 32;
              float v427_data = s0[(v423_a ^ ((v423_a >> 5) & 31))];
              float v431_data = ir7[1];
              ir7[1] = (v431_data + (v427_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 1))));
              int32_t v438_a = v15_lead + 32;
              float v442_data = s0[(v438_a ^ ((v438_a >> 5) & 31))];
              float v446_data = ir7[2];
              ir7[2] = (v446_data + (v442_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 1))));
              int32_t v453_a = v15_lead + 32;
              float v457_data = s0[(v453_a ^ ((v453_a >> 5) & 31))];
              float v461_data = ir7[3];
              ir7[3] = (v461_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 1))));
              int32_t v468_a = v15_lead + 32;
              float v472_data = s0[(v468_a ^ ((v468_a >> 5) & 31))];
              float v476_data = ir7[4];
              ir7[4] = (v476_data + (v472_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 1))));
              int32_t v483_a = v15_lead + 32;
              float v487_data = s0[(v483_a ^ ((v483_a >> 5) & 31))];
              float v491_data = ir7[5];
              ir7[5] = (v491_data + (v487_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 1))));
              int32_t v498_a = v15_lead + 32;
              float v502_data = s0[(v498_a ^ ((v498_a >> 5) & 31))];
              float v506_data = ir7[6];
              ir7[6] = (v506_data + (v502_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 1))));
              int32_t v513_a = v15_lead + 32;
              float v517_data = s0[(v513_a ^ ((v513_a >> 5) & 31))];
              float v521_data = ir7[7];
              ir7[7] = (v521_data + (v517_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 1))));
              int32_t v528_a = v15_lead + 32;
              float v532_data = s0[(v528_a ^ ((v528_a >> 5) & 31))];
              float v536_data = ir7[8];
              ir7[8] = (v536_data + (v532_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 1))));
              int32_t v546_a = v15_lead + 64;
              float v550_data = s0[(v546_a ^ ((v546_a >> 5) & 31))];
              float v554_data = ir7[0];
              ir7[0] = (v554_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 2))));
              int32_t v561_a = v15_lead + 64;
              float v565_data = s0[(v561_a ^ ((v561_a >> 5) & 31))];
              float v569_data = ir7[1];
              ir7[1] = (v569_data + (v565_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 2))));
              int32_t v576_a = v15_lead + 64;
              float v580_data = s0[(v576_a ^ ((v576_a >> 5) & 31))];
              float v584_data = ir7[2];
              ir7[2] = (v584_data + (v580_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 2))));
              int32_t v591_a = v15_lead + 64;
              float v595_data = s0[(v591_a ^ ((v591_a >> 5) & 31))];
              float v599_data = ir7[3];
              ir7[3] = (v599_data + (v595_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 2))));
              int32_t v606_a = v15_lead + 64;
              float v610_data = s0[(v606_a ^ ((v606_a >> 5) & 31))];
              float v614_data = ir7[4];
              ir7[4] = (v614_data + (v610_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 2))));
              int32_t v621_a = v15_lead + 64;
              float v625_data = s0[(v621_a ^ ((v621_a >> 5) & 31))];
              float v629_data = ir7[5];
              ir7[5] = (v629_data + (v625_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 2))));
              int32_t v636_a = v15_lead + 64;
              float v640_data = s0[(v636_a ^ ((v636_a >> 5) & 31))];
              float v644_data = ir7[6];
              ir7[6] = (v644_data + (v640_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 2))));
              int32_t v651_a = v15_lead + 64;
              float v655_data = s0[(v651_a ^ ((v651_a >> 5) & 31))];
              float v659_data = ir7[7];
              ir7[7] = (v659_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 2))));
              int32_t v666_a = v15_lead + 64;
              float v670_data = s0[(v666_a ^ ((v666_a >> 5) & 31))];
              float v674_data = ir7[8];
              ir7[8] = (v674_data + (v670_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 2))));
              int32_t v684_a = v15_lead + 96;
              float v688_data = s0[(v684_a ^ ((v684_a >> 5) & 31))];
              float v692_data = ir7[0];
              ir7[0] = (v692_data + (v688_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 3))));
              int32_t v699_a = v15_lead + 96;
              float v703_data = s0[(v699_a ^ ((v699_a >> 5) & 31))];
              float v707_data = ir7[1];
              ir7[1] = (v707_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 3))));
              int32_t v714_a = v15_lead + 96;
              float v718_data = s0[(v714_a ^ ((v714_a >> 5) & 31))];
              float v722_data = ir7[2];
              ir7[2] = (v722_data + (v718_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 3))));
              int32_t v729_a = v15_lead + 96;
              float v733_data = s0[(v729_a ^ ((v729_a >> 5) & 31))];
              float v737_data = ir7[3];
              ir7[3] = (v737_data + (v733_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 3))));
              int32_t v744_a = v15_lead + 96;
              float v748_data = s0[(v744_a ^ ((v744_a >> 5) & 31))];
              float v752_data = ir7[4];
              ir7[4] = (v752_data + (v748_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 3))));
              int32_t v759_a = v15_lead + 96;
              float v763_data = s0[(v759_a ^ ((v759_a >> 5) & 31))];
              float v767_data = ir7[5];
              ir7[5] = (v767_data + (v763_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 3))));
              int32_t v774_a = v15_lead + 96;
              float v778_data = s0[(v774_a ^ ((v774_a >> 5) & 31))];
              float v782_data = ir7[6];
              ir7[6] = (v782_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 3))));
              int32_t v789_a = v15_lead + 96;
              float v793_data = s0[(v789_a ^ ((v789_a >> 5) & 31))];
              float v797_data = ir7[7];
              ir7[7] = (v797_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 3))));
              int32_t v804_a = v15_lead + 96;
              float v808_data = s0[(v804_a ^ ((v804_a >> 5) & 31))];
              float v812_data = ir7[8];
              ir7[8] = (v812_data + (v808_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 3))));
              int32_t v822_a = v15_lead + 128;
              float v826_data = s0[(v822_a ^ ((v822_a >> 5) & 31))];
              float v830_data = ir7[0];
              ir7[0] = (v830_data + (v826_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 4))));
              int32_t v837_a = v15_lead + 128;
              float v841_data = s0[(v837_a ^ ((v837_a >> 5) & 31))];
              float v845_data = ir7[1];
              ir7[1] = (v845_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 4))));
              int32_t v852_a = v15_lead + 128;
              float v856_data = s0[(v852_a ^ ((v852_a >> 5) & 31))];
              float v860_data = ir7[2];
              ir7[2] = (v860_data + (v856_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 4))));
              int32_t v867_a = v15_lead + 128;
              float v871_data = s0[(v867_a ^ ((v867_a >> 5) & 31))];
              float v875_data = ir7[3];
              ir7[3] = (v875_data + (v871_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 4))));
              int32_t v882_a = v15_lead + 128;
              float v886_data = s0[(v882_a ^ ((v882_a >> 5) & 31))];
              float v890_data = ir7[4];
              ir7[4] = (v890_data + (v886_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 4))));
              int32_t v897_a = v15_lead + 128;
              float v901_data = s0[(v897_a ^ ((v897_a >> 5) & 31))];
              float v905_data = ir7[5];
              ir7[5] = (v905_data + (v901_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 4))));
              int32_t v912_a = v15_lead + 128;
              float v916_data = s0[(v912_a ^ ((v912_a >> 5) & 31))];
              float v920_data = ir7[6];
              ir7[6] = (v920_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 4))));
              int32_t v927_a = v15_lead + 128;
              float v931_data = s0[(v927_a ^ ((v927_a >> 5) & 31))];
              float v935_data = ir7[7];
              ir7[7] = (v935_data + (v931_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 4))));
              int32_t v942_a = v15_lead + 128;
              float v946_data = s0[(v942_a ^ ((v942_a >> 5) & 31))];
              float v950_data = ir7[8];
              ir7[8] = (v950_data + (v946_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 4))));
              int32_t v960_a = v15_lead + 160;
              float v964_data = s0[(v960_a ^ ((v960_a >> 5) & 31))];
              float v968_data = ir7[0];
              ir7[0] = (v968_data + (v964_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 5))));
              int32_t v975_a = v15_lead + 160;
              float v979_data = s0[(v975_a ^ ((v975_a >> 5) & 31))];
              float v983_data = ir7[1];
              ir7[1] = (v983_data + (v979_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 5))));
              int32_t v990_a = v15_lead + 160;
              float v994_data = s0[(v990_a ^ ((v990_a >> 5) & 31))];
              float v998_data = ir7[2];
              ir7[2] = (v998_data + (v994_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 5))));
              int32_t v1005_a = v15_lead + 160;
              float v1009_data = s0[(v1005_a ^ ((v1005_a >> 5) & 31))];
              float v1013_data = ir7[3];
              ir7[3] = (v1013_data + (v1009_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 5))));
              int32_t v1020_a = v15_lead + 160;
              float v1024_data = s0[(v1020_a ^ ((v1020_a >> 5) & 31))];
              float v1028_data = ir7[4];
              ir7[4] = (v1028_data + (v1024_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 5))));
              int32_t v1035_a = v15_lead + 160;
              float v1039_data = s0[(v1035_a ^ ((v1035_a >> 5) & 31))];
              float v1043_data = ir7[5];
              ir7[5] = (v1043_data + (v1039_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 5))));
              int32_t v1050_a = v15_lead + 160;
              float v1054_data = s0[(v1050_a ^ ((v1050_a >> 5) & 31))];
              float v1058_data = ir7[6];
              ir7[6] = (v1058_data + (v1054_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 5))));
              int32_t v1065_a = v15_lead + 160;
              float v1069_data = s0[(v1065_a ^ ((v1065_a >> 5) & 31))];
              float v1073_data = ir7[7];
              ir7[7] = (v1073_data + (v1069_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 5))));
              int32_t v1080_a = v15_lead + 160;
              float v1084_data = s0[(v1080_a ^ ((v1080_a >> 5) & 31))];
              float v1088_data = ir7[8];
              ir7[8] = (v1088_data + (v1084_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 5))));
              int32_t v1098_a = v15_lead + 192;
              float v1102_data = s0[(v1098_a ^ ((v1098_a >> 5) & 31))];
              float v1106_data = ir7[0];
              ir7[0] = (v1106_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 6))));
              int32_t v1113_a = v15_lead + 192;
              float v1117_data = s0[(v1113_a ^ ((v1113_a >> 5) & 31))];
              float v1121_data = ir7[1];
              ir7[1] = (v1121_data + (v1117_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 6))));
              int32_t v1128_a = v15_lead + 192;
              float v1132_data = s0[(v1128_a ^ ((v1128_a >> 5) & 31))];
              float v1136_data = ir7[2];
              ir7[2] = (v1136_data + (v1132_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 6))));
              int32_t v1143_a = v15_lead + 192;
              float v1147_data = s0[(v1143_a ^ ((v1143_a >> 5) & 31))];
              float v1151_data = ir7[3];
              ir7[3] = (v1151_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 6))));
              int32_t v1158_a = v15_lead + 192;
              float v1162_data = s0[(v1158_a ^ ((v1158_a >> 5) & 31))];
              float v1166_data = ir7[4];
              ir7[4] = (v1166_data + (v1162_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 6))));
              int32_t v1173_a = v15_lead + 192;
              float v1177_data = s0[(v1173_a ^ ((v1173_a >> 5) & 31))];
              float v1181_data = ir7[5];
              ir7[5] = (v1181_data + (v1177_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 6))));
              int32_t v1188_a = v15_lead + 192;
              float v1192_data = s0[(v1188_a ^ ((v1188_a >> 5) & 31))];
              float v1196_data = ir7[6];
              ir7[6] = (v1196_data + (v1192_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 6))));
              int32_t v1203_a = v15_lead + 192;
              float v1207_data = s0[(v1203_a ^ ((v1203_a >> 5) & 31))];
              float v1211_data = ir7[7];
              ir7[7] = (v1211_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 6))));
              int32_t v1218_a = v15_lead + 192;
              float v1222_data = s0[(v1218_a ^ ((v1218_a >> 5) & 31))];
              float v1226_data = ir7[8];
              ir7[8] = (v1226_data + (v1222_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 6))));
              int32_t v1236_a = v15_lead + 224;
              float v1240_data = s0[(v1236_a ^ ((v1236_a >> 5) & 31))];
              float v1244_data = ir7[0];
              ir7[0] = (v1244_data + (v1240_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 7))));
              int32_t v1251_a = v15_lead + 224;
              float v1255_data = s0[(v1251_a ^ ((v1251_a >> 5) & 31))];
              float v1259_data = ir7[1];
              ir7[1] = (v1259_data + (v1255_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 7))));
              int32_t v1266_a = v15_lead + 224;
              float v1270_data = s0[(v1266_a ^ ((v1266_a >> 5) & 31))];
              float v1274_data = ir7[2];
              ir7[2] = (v1274_data + (v1270_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 7))));
              int32_t v1281_a = v15_lead + 224;
              float v1285_data = s0[(v1281_a ^ ((v1281_a >> 5) & 31))];
              float v1289_data = ir7[3];
              ir7[3] = (v1289_data + (v1285_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 7))));
              int32_t v1296_a = v15_lead + 224;
              float v1300_data = s0[(v1296_a ^ ((v1296_a >> 5) & 31))];
              float v1304_data = ir7[4];
              ir7[4] = (v1304_data + (v1300_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 7))));
              int32_t v1311_a = v15_lead + 224;
              float v1315_data = s0[(v1311_a ^ ((v1311_a >> 5) & 31))];
              float v1319_data = ir7[5];
              ir7[5] = (v1319_data + (v1315_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 7))));
              int32_t v1326_a = v15_lead + 224;
              float v1330_data = s0[(v1326_a ^ ((v1326_a >> 5) & 31))];
              float v1334_data = ir7[6];
              ir7[6] = (v1334_data + (v1330_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 7))));
              int32_t v1341_a = v15_lead + 224;
              float v1345_data = s0[(v1341_a ^ ((v1341_a >> 5) & 31))];
              float v1349_data = ir7[7];
              ir7[7] = (v1349_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 7))));
              int32_t v1356_a = v15_lead + 224;
              float v1360_data = s0[(v1356_a ^ ((v1356_a >> 5) & 31))];
              float v1364_data = ir7[8];
              ir7[8] = (v1364_data + (v1360_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 7))));
              int32_t v1374_a = v15_lead + 256;
              float v1378_data = s0[(v1374_a ^ ((v1374_a >> 5) & 31))];
              float v1382_data = ir7[0];
              ir7[0] = (v1382_data + (v1378_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 8))));
              int32_t v1389_a = v15_lead + 256;
              float v1393_data = s0[(v1389_a ^ ((v1389_a >> 5) & 31))];
              float v1397_data = ir7[1];
              ir7[1] = (v1397_data + (v1393_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 8))));
              int32_t v1404_a = v15_lead + 256;
              float v1408_data = s0[(v1404_a ^ ((v1404_a >> 5) & 31))];
              float v1412_data = ir7[2];
              ir7[2] = (v1412_data + (v1408_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 8))));
              int32_t v1419_a = v15_lead + 256;
              float v1423_data = s0[(v1419_a ^ ((v1419_a >> 5) & 31))];
              float v1427_data = ir7[3];
              ir7[3] = (v1427_data + (v1423_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 8))));
              int32_t v1434_a = v15_lead + 256;
              float v1438_data = s0[(v1434_a ^ ((v1434_a >> 5) & 31))];
              float v1442_data = ir7[4];
              ir7[4] = (v1442_data + (v1438_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 8))));
              int32_t v1449_a = v15_lead + 256;
              float v1453_data = s0[(v1449_a ^ ((v1449_a >> 5) & 31))];
              float v1457_data = ir7[5];
              ir7[5] = (v1457_data + (v1453_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 8))));
              int32_t v1464_a = v15_lead + 256;
              float v1468_data = s0[(v1464_a ^ ((v1464_a >> 5) & 31))];
              float v1472_data = ir7[6];
              ir7[6] = (v1472_data + (v1468_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 8))));
              int32_t v1479_a = v15_lead + 256;
              float v1483_data = s0[(v1479_a ^ ((v1479_a >> 5) & 31))];
              float v1487_data = ir7[7];
              ir7[7] = (v1487_data + (v1483_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 8))));
              int32_t v1494_a = v15_lead + 256;
              float v1498_data = s0[(v1494_a ^ ((v1494_a >> 5) & 31))];
              float v1502_data = ir7[8];
              ir7[8] = (v1502_data + (v1498_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 8))));
              #pragma unroll
              for (int32_t v1507_n0 = 0; v1507_n0 < 1; ++v1507_n0) {
                #pragma unroll
                for (int32_t v1508_n1 = 0; v1508_n1 < 9; ++v1508_n1) {
                  int32_t v1509_a = v1507_n0 + v1508_n1;
                  float v1510_data = ir7[v1509_a];
                  r7[v1509_a] = v1510_data;
                }
              }
              // glb_m3 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1515_i0 = 0; v1515_i0 < 1; ++v1515_i0) {
                int32_t v1523_lead = v15_lead + (v1515_i0 * 32);
                #pragma unroll
                for (int32_t v1516_i1 = 0; v1516_i1 < 9; ++v1516_i1) {
                  float v1518_data = r7[(v1515_i0 + v1516_i1)];
                  glb_m3[(v1523_lead + (v1516_i1 * 32))] = v1518_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}


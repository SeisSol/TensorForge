// === base name ===
kernel_f61651fe59

// === header ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f61651fe59(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f61651fe59(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(12×16) {4..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(12×16) {4..16}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              bool v9_g = v8_lead < 12;
              if (v9_g) {
                int32_t v17_a = (v8_lead + 4) - 4;
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 16; ++v10_i1) {
                  float v20_data = glb_m1[(v17_a + (v10_i1 * 12))];
                  r0[v10_i1] = v20_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
                int32_t v32_lead = v8_lead + (v26_i0 * 16);
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
                  float v35_data = glb_m2[(v32_lead + (v27_i1 * 16))];
                  r1[(v26_i0 + v27_i1)] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[8]{};
              if (v9_g) {
                float v43_data = r0[0];
                float v44_data = r1[0];
                float v47_data = ir2[0];
                ir2[0] = (v47_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v44_data, 0))));
                float v50_data = r1[1];
                float v53_data = ir2[1];
                ir2[1] = (v53_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 0))));
                float v56_data = r1[2];
                float v59_data = ir2[2];
                ir2[2] = (v59_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v56_data, 0))));
                float v62_data = r1[3];
                float v65_data = ir2[3];
                ir2[3] = (v65_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 0))));
                float v68_data = r1[4];
                float v71_data = ir2[4];
                ir2[4] = (v71_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 0))));
                float v74_data = r1[5];
                float v77_data = ir2[5];
                ir2[5] = (v77_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 0))));
                float v80_data = r1[6];
                float v83_data = ir2[6];
                ir2[6] = (v83_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 0))));
                float v86_data = r1[7];
                float v89_data = ir2[7];
                ir2[7] = (v89_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 0))));
              }
              if (v9_g) {
                float v95_data = r0[1];
                float v96_data = r1[0];
                float v99_data = ir2[0];
                ir2[0] = (v99_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 1))));
                float v102_data = r1[1];
                float v105_data = ir2[1];
                ir2[1] = (v105_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 1))));
                float v108_data = r1[2];
                float v111_data = ir2[2];
                ir2[2] = (v111_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 1))));
                float v114_data = r1[3];
                float v117_data = ir2[3];
                ir2[3] = (v117_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 1))));
                float v120_data = r1[4];
                float v123_data = ir2[4];
                ir2[4] = (v123_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 1))));
                float v126_data = r1[5];
                float v129_data = ir2[5];
                ir2[5] = (v129_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 1))));
                float v132_data = r1[6];
                float v135_data = ir2[6];
                ir2[6] = (v135_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 1))));
                float v138_data = r1[7];
                float v141_data = ir2[7];
                ir2[7] = (v141_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 1))));
              }
              if (v9_g) {
                float v147_data = r0[2];
                float v148_data = r1[0];
                float v151_data = ir2[0];
                ir2[0] = (v151_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 2))));
                float v154_data = r1[1];
                float v157_data = ir2[1];
                ir2[1] = (v157_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v154_data, 2))));
                float v160_data = r1[2];
                float v163_data = ir2[2];
                ir2[2] = (v163_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v160_data, 2))));
                float v166_data = r1[3];
                float v169_data = ir2[3];
                ir2[3] = (v169_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v166_data, 2))));
                float v172_data = r1[4];
                float v175_data = ir2[4];
                ir2[4] = (v175_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v172_data, 2))));
                float v178_data = r1[5];
                float v181_data = ir2[5];
                ir2[5] = (v181_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v178_data, 2))));
                float v184_data = r1[6];
                float v187_data = ir2[6];
                ir2[6] = (v187_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v184_data, 2))));
                float v190_data = r1[7];
                float v193_data = ir2[7];
                ir2[7] = (v193_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v190_data, 2))));
              }
              if (v9_g) {
                float v199_data = r0[3];
                float v200_data = r1[0];
                float v203_data = ir2[0];
                ir2[0] = (v203_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v200_data, 3))));
                float v206_data = r1[1];
                float v209_data = ir2[1];
                ir2[1] = (v209_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v206_data, 3))));
                float v212_data = r1[2];
                float v215_data = ir2[2];
                ir2[2] = (v215_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v212_data, 3))));
                float v218_data = r1[3];
                float v221_data = ir2[3];
                ir2[3] = (v221_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v218_data, 3))));
                float v224_data = r1[4];
                float v227_data = ir2[4];
                ir2[4] = (v227_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v224_data, 3))));
                float v230_data = r1[5];
                float v233_data = ir2[5];
                ir2[5] = (v233_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v230_data, 3))));
                float v236_data = r1[6];
                float v239_data = ir2[6];
                ir2[6] = (v239_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v236_data, 3))));
                float v242_data = r1[7];
                float v245_data = ir2[7];
                ir2[7] = (v245_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 3))));
              }
              if (v9_g) {
                float v251_data = r0[4];
                float v252_data = r1[0];
                float v255_data = ir2[0];
                ir2[0] = (v255_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v252_data, 4))));
                float v258_data = r1[1];
                float v261_data = ir2[1];
                ir2[1] = (v261_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v258_data, 4))));
                float v264_data = r1[2];
                float v267_data = ir2[2];
                ir2[2] = (v267_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v264_data, 4))));
                float v270_data = r1[3];
                float v273_data = ir2[3];
                ir2[3] = (v273_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v270_data, 4))));
                float v276_data = r1[4];
                float v279_data = ir2[4];
                ir2[4] = (v279_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v276_data, 4))));
                float v282_data = r1[5];
                float v285_data = ir2[5];
                ir2[5] = (v285_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v282_data, 4))));
                float v288_data = r1[6];
                float v291_data = ir2[6];
                ir2[6] = (v291_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v288_data, 4))));
                float v294_data = r1[7];
                float v297_data = ir2[7];
                ir2[7] = (v297_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 4))));
              }
              if (v9_g) {
                float v303_data = r0[5];
                float v304_data = r1[0];
                float v307_data = ir2[0];
                ir2[0] = (v307_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v304_data, 5))));
                float v310_data = r1[1];
                float v313_data = ir2[1];
                ir2[1] = (v313_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v310_data, 5))));
                float v316_data = r1[2];
                float v319_data = ir2[2];
                ir2[2] = (v319_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v316_data, 5))));
                float v322_data = r1[3];
                float v325_data = ir2[3];
                ir2[3] = (v325_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v322_data, 5))));
                float v328_data = r1[4];
                float v331_data = ir2[4];
                ir2[4] = (v331_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v328_data, 5))));
                float v334_data = r1[5];
                float v337_data = ir2[5];
                ir2[5] = (v337_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v334_data, 5))));
                float v340_data = r1[6];
                float v343_data = ir2[6];
                ir2[6] = (v343_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v340_data, 5))));
                float v346_data = r1[7];
                float v349_data = ir2[7];
                ir2[7] = (v349_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 5))));
              }
              if (v9_g) {
                float v355_data = r0[6];
                float v356_data = r1[0];
                float v359_data = ir2[0];
                ir2[0] = (v359_data + (v355_data * (sycl::group_broadcast(item.get_sub_group(), v356_data, 6))));
                float v362_data = r1[1];
                float v365_data = ir2[1];
                ir2[1] = (v365_data + (v355_data * (sycl::group_broadcast(item.get_sub_group(), v362_data, 6))));
                float v368_data = r1[2];
                float v371_data = ir2[2];
                ir2[2] = (v371_data + (v355_data * (sycl::group_broadcast(item.get_sub_group(), v368_data, 6))));
                float v374_data = r1[3];
                float v377_data = ir2[3];
                ir2[3] = (v377_data + (v355_data * (sycl::group_broadcast(item.get_sub_group(), v374_data, 6))));
                float v380_data = r1[4];
                float v383_data = ir2[4];
                ir2[4] = (v383_data + (v355_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 6))));
                float v386_data = r1[5];
                float v389_data = ir2[5];
                ir2[5] = (v389_data + (v355_data * (sycl::group_broadcast(item.get_sub_group(), v386_data, 6))));
                float v392_data = r1[6];
                float v395_data = ir2[6];
                ir2[6] = (v395_data + (v355_data * (sycl::group_broadcast(item.get_sub_group(), v392_data, 6))));
                float v398_data = r1[7];
                float v401_data = ir2[7];
                ir2[7] = (v401_data + (v355_data * (sycl::group_broadcast(item.get_sub_group(), v398_data, 6))));
              }
              if (v9_g) {
                float v407_data = r0[7];
                float v408_data = r1[0];
                float v411_data = ir2[0];
                ir2[0] = (v411_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v408_data, 7))));
                float v414_data = r1[1];
                float v417_data = ir2[1];
                ir2[1] = (v417_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v414_data, 7))));
                float v420_data = r1[2];
                float v423_data = ir2[2];
                ir2[2] = (v423_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v420_data, 7))));
                float v426_data = r1[3];
                float v429_data = ir2[3];
                ir2[3] = (v429_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v426_data, 7))));
                float v432_data = r1[4];
                float v435_data = ir2[4];
                ir2[4] = (v435_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v432_data, 7))));
                float v438_data = r1[5];
                float v441_data = ir2[5];
                ir2[5] = (v441_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v438_data, 7))));
                float v444_data = r1[6];
                float v447_data = ir2[6];
                ir2[6] = (v447_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v444_data, 7))));
                float v450_data = r1[7];
                float v453_data = ir2[7];
                ir2[7] = (v453_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 7))));
              }
              if (v9_g) {
                float v459_data = r0[8];
                float v460_data = r1[0];
                float v463_data = ir2[0];
                ir2[0] = (v463_data + (v459_data * (sycl::group_broadcast(item.get_sub_group(), v460_data, 8))));
                float v466_data = r1[1];
                float v469_data = ir2[1];
                ir2[1] = (v469_data + (v459_data * (sycl::group_broadcast(item.get_sub_group(), v466_data, 8))));
                float v472_data = r1[2];
                float v475_data = ir2[2];
                ir2[2] = (v475_data + (v459_data * (sycl::group_broadcast(item.get_sub_group(), v472_data, 8))));
                float v478_data = r1[3];
                float v481_data = ir2[3];
                ir2[3] = (v481_data + (v459_data * (sycl::group_broadcast(item.get_sub_group(), v478_data, 8))));
                float v484_data = r1[4];
                float v487_data = ir2[4];
                ir2[4] = (v487_data + (v459_data * (sycl::group_broadcast(item.get_sub_group(), v484_data, 8))));
                float v490_data = r1[5];
                float v493_data = ir2[5];
                ir2[5] = (v493_data + (v459_data * (sycl::group_broadcast(item.get_sub_group(), v490_data, 8))));
                float v496_data = r1[6];
                float v499_data = ir2[6];
                ir2[6] = (v499_data + (v459_data * (sycl::group_broadcast(item.get_sub_group(), v496_data, 8))));
                float v502_data = r1[7];
                float v505_data = ir2[7];
                ir2[7] = (v505_data + (v459_data * (sycl::group_broadcast(item.get_sub_group(), v502_data, 8))));
              }
              if (v9_g) {
                float v511_data = r0[9];
                float v512_data = r1[0];
                float v515_data = ir2[0];
                ir2[0] = (v515_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v512_data, 9))));
                float v518_data = r1[1];
                float v521_data = ir2[1];
                ir2[1] = (v521_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v518_data, 9))));
                float v524_data = r1[2];
                float v527_data = ir2[2];
                ir2[2] = (v527_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v524_data, 9))));
                float v530_data = r1[3];
                float v533_data = ir2[3];
                ir2[3] = (v533_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v530_data, 9))));
                float v536_data = r1[4];
                float v539_data = ir2[4];
                ir2[4] = (v539_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v536_data, 9))));
                float v542_data = r1[5];
                float v545_data = ir2[5];
                ir2[5] = (v545_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 9))));
                float v548_data = r1[6];
                float v551_data = ir2[6];
                ir2[6] = (v551_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 9))));
                float v554_data = r1[7];
                float v557_data = ir2[7];
                ir2[7] = (v557_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v554_data, 9))));
              }
              if (v9_g) {
                float v563_data = r0[10];
                float v564_data = r1[0];
                float v567_data = ir2[0];
                ir2[0] = (v567_data + (v563_data * (sycl::group_broadcast(item.get_sub_group(), v564_data, 10))));
                float v570_data = r1[1];
                float v573_data = ir2[1];
                ir2[1] = (v573_data + (v563_data * (sycl::group_broadcast(item.get_sub_group(), v570_data, 10))));
                float v576_data = r1[2];
                float v579_data = ir2[2];
                ir2[2] = (v579_data + (v563_data * (sycl::group_broadcast(item.get_sub_group(), v576_data, 10))));
                float v582_data = r1[3];
                float v585_data = ir2[3];
                ir2[3] = (v585_data + (v563_data * (sycl::group_broadcast(item.get_sub_group(), v582_data, 10))));
                float v588_data = r1[4];
                float v591_data = ir2[4];
                ir2[4] = (v591_data + (v563_data * (sycl::group_broadcast(item.get_sub_group(), v588_data, 10))));
                float v594_data = r1[5];
                float v597_data = ir2[5];
                ir2[5] = (v597_data + (v563_data * (sycl::group_broadcast(item.get_sub_group(), v594_data, 10))));
                float v600_data = r1[6];
                float v603_data = ir2[6];
                ir2[6] = (v603_data + (v563_data * (sycl::group_broadcast(item.get_sub_group(), v600_data, 10))));
                float v606_data = r1[7];
                float v609_data = ir2[7];
                ir2[7] = (v609_data + (v563_data * (sycl::group_broadcast(item.get_sub_group(), v606_data, 10))));
              }
              if (v9_g) {
                float v615_data = r0[11];
                float v616_data = r1[0];
                float v619_data = ir2[0];
                ir2[0] = (v619_data + (v615_data * (sycl::group_broadcast(item.get_sub_group(), v616_data, 11))));
                float v622_data = r1[1];
                float v625_data = ir2[1];
                ir2[1] = (v625_data + (v615_data * (sycl::group_broadcast(item.get_sub_group(), v622_data, 11))));
                float v628_data = r1[2];
                float v631_data = ir2[2];
                ir2[2] = (v631_data + (v615_data * (sycl::group_broadcast(item.get_sub_group(), v628_data, 11))));
                float v634_data = r1[3];
                float v637_data = ir2[3];
                ir2[3] = (v637_data + (v615_data * (sycl::group_broadcast(item.get_sub_group(), v634_data, 11))));
                float v640_data = r1[4];
                float v643_data = ir2[4];
                ir2[4] = (v643_data + (v615_data * (sycl::group_broadcast(item.get_sub_group(), v640_data, 11))));
                float v646_data = r1[5];
                float v649_data = ir2[5];
                ir2[5] = (v649_data + (v615_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 11))));
                float v652_data = r1[6];
                float v655_data = ir2[6];
                ir2[6] = (v655_data + (v615_data * (sycl::group_broadcast(item.get_sub_group(), v652_data, 11))));
                float v658_data = r1[7];
                float v661_data = ir2[7];
                ir2[7] = (v661_data + (v615_data * (sycl::group_broadcast(item.get_sub_group(), v658_data, 11))));
              }
              if (v9_g) {
                float v667_data = r0[12];
                float v668_data = r1[0];
                float v671_data = ir2[0];
                ir2[0] = (v671_data + (v667_data * (sycl::group_broadcast(item.get_sub_group(), v668_data, 12))));
                float v674_data = r1[1];
                float v677_data = ir2[1];
                ir2[1] = (v677_data + (v667_data * (sycl::group_broadcast(item.get_sub_group(), v674_data, 12))));
                float v680_data = r1[2];
                float v683_data = ir2[2];
                ir2[2] = (v683_data + (v667_data * (sycl::group_broadcast(item.get_sub_group(), v680_data, 12))));
                float v686_data = r1[3];
                float v689_data = ir2[3];
                ir2[3] = (v689_data + (v667_data * (sycl::group_broadcast(item.get_sub_group(), v686_data, 12))));
                float v692_data = r1[4];
                float v695_data = ir2[4];
                ir2[4] = (v695_data + (v667_data * (sycl::group_broadcast(item.get_sub_group(), v692_data, 12))));
                float v698_data = r1[5];
                float v701_data = ir2[5];
                ir2[5] = (v701_data + (v667_data * (sycl::group_broadcast(item.get_sub_group(), v698_data, 12))));
                float v704_data = r1[6];
                float v707_data = ir2[6];
                ir2[6] = (v707_data + (v667_data * (sycl::group_broadcast(item.get_sub_group(), v704_data, 12))));
                float v710_data = r1[7];
                float v713_data = ir2[7];
                ir2[7] = (v713_data + (v667_data * (sycl::group_broadcast(item.get_sub_group(), v710_data, 12))));
              }
              if (v9_g) {
                float v719_data = r0[13];
                float v720_data = r1[0];
                float v723_data = ir2[0];
                ir2[0] = (v723_data + (v719_data * (sycl::group_broadcast(item.get_sub_group(), v720_data, 13))));
                float v726_data = r1[1];
                float v729_data = ir2[1];
                ir2[1] = (v729_data + (v719_data * (sycl::group_broadcast(item.get_sub_group(), v726_data, 13))));
                float v732_data = r1[2];
                float v735_data = ir2[2];
                ir2[2] = (v735_data + (v719_data * (sycl::group_broadcast(item.get_sub_group(), v732_data, 13))));
                float v738_data = r1[3];
                float v741_data = ir2[3];
                ir2[3] = (v741_data + (v719_data * (sycl::group_broadcast(item.get_sub_group(), v738_data, 13))));
                float v744_data = r1[4];
                float v747_data = ir2[4];
                ir2[4] = (v747_data + (v719_data * (sycl::group_broadcast(item.get_sub_group(), v744_data, 13))));
                float v750_data = r1[5];
                float v753_data = ir2[5];
                ir2[5] = (v753_data + (v719_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 13))));
                float v756_data = r1[6];
                float v759_data = ir2[6];
                ir2[6] = (v759_data + (v719_data * (sycl::group_broadcast(item.get_sub_group(), v756_data, 13))));
                float v762_data = r1[7];
                float v765_data = ir2[7];
                ir2[7] = (v765_data + (v719_data * (sycl::group_broadcast(item.get_sub_group(), v762_data, 13))));
              }
              if (v9_g) {
                float v771_data = r0[14];
                float v772_data = r1[0];
                float v775_data = ir2[0];
                ir2[0] = (v775_data + (v771_data * (sycl::group_broadcast(item.get_sub_group(), v772_data, 14))));
                float v778_data = r1[1];
                float v781_data = ir2[1];
                ir2[1] = (v781_data + (v771_data * (sycl::group_broadcast(item.get_sub_group(), v778_data, 14))));
                float v784_data = r1[2];
                float v787_data = ir2[2];
                ir2[2] = (v787_data + (v771_data * (sycl::group_broadcast(item.get_sub_group(), v784_data, 14))));
                float v790_data = r1[3];
                float v793_data = ir2[3];
                ir2[3] = (v793_data + (v771_data * (sycl::group_broadcast(item.get_sub_group(), v790_data, 14))));
                float v796_data = r1[4];
                float v799_data = ir2[4];
                ir2[4] = (v799_data + (v771_data * (sycl::group_broadcast(item.get_sub_group(), v796_data, 14))));
                float v802_data = r1[5];
                float v805_data = ir2[5];
                ir2[5] = (v805_data + (v771_data * (sycl::group_broadcast(item.get_sub_group(), v802_data, 14))));
                float v808_data = r1[6];
                float v811_data = ir2[6];
                ir2[6] = (v811_data + (v771_data * (sycl::group_broadcast(item.get_sub_group(), v808_data, 14))));
                float v814_data = r1[7];
                float v817_data = ir2[7];
                ir2[7] = (v817_data + (v771_data * (sycl::group_broadcast(item.get_sub_group(), v814_data, 14))));
              }
              if (v9_g) {
                float v823_data = r0[15];
                float v824_data = r1[0];
                float v827_data = ir2[0];
                ir2[0] = (v827_data + (v823_data * (sycl::group_broadcast(item.get_sub_group(), v824_data, 15))));
                float v830_data = r1[1];
                float v833_data = ir2[1];
                ir2[1] = (v833_data + (v823_data * (sycl::group_broadcast(item.get_sub_group(), v830_data, 15))));
                float v836_data = r1[2];
                float v839_data = ir2[2];
                ir2[2] = (v839_data + (v823_data * (sycl::group_broadcast(item.get_sub_group(), v836_data, 15))));
                float v842_data = r1[3];
                float v845_data = ir2[3];
                ir2[3] = (v845_data + (v823_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 15))));
                float v848_data = r1[4];
                float v851_data = ir2[4];
                ir2[4] = (v851_data + (v823_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 15))));
                float v854_data = r1[5];
                float v857_data = ir2[5];
                ir2[5] = (v857_data + (v823_data * (sycl::group_broadcast(item.get_sub_group(), v854_data, 15))));
                float v860_data = r1[6];
                float v863_data = ir2[6];
                ir2[6] = (v863_data + (v823_data * (sycl::group_broadcast(item.get_sub_group(), v860_data, 15))));
                float v866_data = r1[7];
                float v869_data = ir2[7];
                ir2[7] = (v869_data + (v823_data * (sycl::group_broadcast(item.get_sub_group(), v866_data, 15))));
              }
              if (v9_g) {
                #pragma unroll
                for (int32_t v875_n1 = 0; v875_n1 < 8; ++v875_n1) {
                  float v877_data = ir2[v875_n1];
                  r2[v875_n1] = v877_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v9_g) {
                #pragma unroll
                for (int32_t v883_i1 = 0; v883_i1 < 8; ++v883_i1) {
                  float v885_data = r2[v883_i1];
                  glb_m0[(v8_lead + (v883_i1 * 12))] = v885_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


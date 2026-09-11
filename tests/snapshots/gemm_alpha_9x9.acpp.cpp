// === base name ===
kernel_c952f2d139605083

// === header ===
void launcher_kernel_c952f2d139605083(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_c952f2d139605083(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c952f2d139605083(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c952f2d139605083(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 81 + 0 + m2_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(0) % 16;
              if (v16_lead < 9) {
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 9; ++v18_i1) {
                  float v26_data = glb_m1[(v16_lead + (v18_i1 * 9))];
                  r0[v18_i1] = v26_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              if (v16_lead < 9) {
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
                  float v41_data = glb_m2[(v16_lead + (v33_i1 * 9))];
                  r1[v33_i1] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir2[9]{};
              if (v16_lead < 9) {
                float v49_data = r0[0];
                float v50_data = r1[0];
                float v53_data = ir2[0];
                ir2[0] = (v53_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 0))));
                float v56_data = r1[1];
                float v59_data = ir2[1];
                ir2[1] = (v59_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v56_data, 0))));
                float v62_data = r1[2];
                float v65_data = ir2[2];
                ir2[2] = (v65_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 0))));
                float v68_data = r1[3];
                float v71_data = ir2[3];
                ir2[3] = (v71_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 0))));
                float v74_data = r1[4];
                float v77_data = ir2[4];
                ir2[4] = (v77_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 0))));
                float v80_data = r1[5];
                float v83_data = ir2[5];
                ir2[5] = (v83_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 0))));
                float v86_data = r1[6];
                float v89_data = ir2[6];
                ir2[6] = (v89_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 0))));
                float v92_data = r1[7];
                float v95_data = ir2[7];
                ir2[7] = (v95_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 0))));
                float v98_data = r1[8];
                float v101_data = ir2[8];
                ir2[8] = (v101_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 0))));
              }
              if (v16_lead < 9) {
                float v107_data = r0[1];
                float v108_data = r1[0];
                float v111_data = ir2[0];
                ir2[0] = (v111_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 1))));
                float v114_data = r1[1];
                float v117_data = ir2[1];
                ir2[1] = (v117_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 1))));
                float v120_data = r1[2];
                float v123_data = ir2[2];
                ir2[2] = (v123_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 1))));
                float v126_data = r1[3];
                float v129_data = ir2[3];
                ir2[3] = (v129_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 1))));
                float v132_data = r1[4];
                float v135_data = ir2[4];
                ir2[4] = (v135_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 1))));
                float v138_data = r1[5];
                float v141_data = ir2[5];
                ir2[5] = (v141_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 1))));
                float v144_data = r1[6];
                float v147_data = ir2[6];
                ir2[6] = (v147_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 1))));
                float v150_data = r1[7];
                float v153_data = ir2[7];
                ir2[7] = (v153_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 1))));
                float v156_data = r1[8];
                float v159_data = ir2[8];
                ir2[8] = (v159_data + (v107_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 1))));
              }
              if (v16_lead < 9) {
                float v165_data = r0[2];
                float v166_data = r1[0];
                float v169_data = ir2[0];
                ir2[0] = (v169_data + (v165_data * (sycl::group_broadcast(item.get_sub_group(), v166_data, 2))));
                float v172_data = r1[1];
                float v175_data = ir2[1];
                ir2[1] = (v175_data + (v165_data * (sycl::group_broadcast(item.get_sub_group(), v172_data, 2))));
                float v178_data = r1[2];
                float v181_data = ir2[2];
                ir2[2] = (v181_data + (v165_data * (sycl::group_broadcast(item.get_sub_group(), v178_data, 2))));
                float v184_data = r1[3];
                float v187_data = ir2[3];
                ir2[3] = (v187_data + (v165_data * (sycl::group_broadcast(item.get_sub_group(), v184_data, 2))));
                float v190_data = r1[4];
                float v193_data = ir2[4];
                ir2[4] = (v193_data + (v165_data * (sycl::group_broadcast(item.get_sub_group(), v190_data, 2))));
                float v196_data = r1[5];
                float v199_data = ir2[5];
                ir2[5] = (v199_data + (v165_data * (sycl::group_broadcast(item.get_sub_group(), v196_data, 2))));
                float v202_data = r1[6];
                float v205_data = ir2[6];
                ir2[6] = (v205_data + (v165_data * (sycl::group_broadcast(item.get_sub_group(), v202_data, 2))));
                float v208_data = r1[7];
                float v211_data = ir2[7];
                ir2[7] = (v211_data + (v165_data * (sycl::group_broadcast(item.get_sub_group(), v208_data, 2))));
                float v214_data = r1[8];
                float v217_data = ir2[8];
                ir2[8] = (v217_data + (v165_data * (sycl::group_broadcast(item.get_sub_group(), v214_data, 2))));
              }
              if (v16_lead < 9) {
                float v223_data = r0[3];
                float v224_data = r1[0];
                float v227_data = ir2[0];
                ir2[0] = (v227_data + (v223_data * (sycl::group_broadcast(item.get_sub_group(), v224_data, 3))));
                float v230_data = r1[1];
                float v233_data = ir2[1];
                ir2[1] = (v233_data + (v223_data * (sycl::group_broadcast(item.get_sub_group(), v230_data, 3))));
                float v236_data = r1[2];
                float v239_data = ir2[2];
                ir2[2] = (v239_data + (v223_data * (sycl::group_broadcast(item.get_sub_group(), v236_data, 3))));
                float v242_data = r1[3];
                float v245_data = ir2[3];
                ir2[3] = (v245_data + (v223_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 3))));
                float v248_data = r1[4];
                float v251_data = ir2[4];
                ir2[4] = (v251_data + (v223_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 3))));
                float v254_data = r1[5];
                float v257_data = ir2[5];
                ir2[5] = (v257_data + (v223_data * (sycl::group_broadcast(item.get_sub_group(), v254_data, 3))));
                float v260_data = r1[6];
                float v263_data = ir2[6];
                ir2[6] = (v263_data + (v223_data * (sycl::group_broadcast(item.get_sub_group(), v260_data, 3))));
                float v266_data = r1[7];
                float v269_data = ir2[7];
                ir2[7] = (v269_data + (v223_data * (sycl::group_broadcast(item.get_sub_group(), v266_data, 3))));
                float v272_data = r1[8];
                float v275_data = ir2[8];
                ir2[8] = (v275_data + (v223_data * (sycl::group_broadcast(item.get_sub_group(), v272_data, 3))));
              }
              if (v16_lead < 9) {
                float v281_data = r0[4];
                float v282_data = r1[0];
                float v285_data = ir2[0];
                ir2[0] = (v285_data + (v281_data * (sycl::group_broadcast(item.get_sub_group(), v282_data, 4))));
                float v288_data = r1[1];
                float v291_data = ir2[1];
                ir2[1] = (v291_data + (v281_data * (sycl::group_broadcast(item.get_sub_group(), v288_data, 4))));
                float v294_data = r1[2];
                float v297_data = ir2[2];
                ir2[2] = (v297_data + (v281_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 4))));
                float v300_data = r1[3];
                float v303_data = ir2[3];
                ir2[3] = (v303_data + (v281_data * (sycl::group_broadcast(item.get_sub_group(), v300_data, 4))));
                float v306_data = r1[4];
                float v309_data = ir2[4];
                ir2[4] = (v309_data + (v281_data * (sycl::group_broadcast(item.get_sub_group(), v306_data, 4))));
                float v312_data = r1[5];
                float v315_data = ir2[5];
                ir2[5] = (v315_data + (v281_data * (sycl::group_broadcast(item.get_sub_group(), v312_data, 4))));
                float v318_data = r1[6];
                float v321_data = ir2[6];
                ir2[6] = (v321_data + (v281_data * (sycl::group_broadcast(item.get_sub_group(), v318_data, 4))));
                float v324_data = r1[7];
                float v327_data = ir2[7];
                ir2[7] = (v327_data + (v281_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 4))));
                float v330_data = r1[8];
                float v333_data = ir2[8];
                ir2[8] = (v333_data + (v281_data * (sycl::group_broadcast(item.get_sub_group(), v330_data, 4))));
              }
              if (v16_lead < 9) {
                float v339_data = r0[5];
                float v340_data = r1[0];
                float v343_data = ir2[0];
                ir2[0] = (v343_data + (v339_data * (sycl::group_broadcast(item.get_sub_group(), v340_data, 5))));
                float v346_data = r1[1];
                float v349_data = ir2[1];
                ir2[1] = (v349_data + (v339_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 5))));
                float v352_data = r1[2];
                float v355_data = ir2[2];
                ir2[2] = (v355_data + (v339_data * (sycl::group_broadcast(item.get_sub_group(), v352_data, 5))));
                float v358_data = r1[3];
                float v361_data = ir2[3];
                ir2[3] = (v361_data + (v339_data * (sycl::group_broadcast(item.get_sub_group(), v358_data, 5))));
                float v364_data = r1[4];
                float v367_data = ir2[4];
                ir2[4] = (v367_data + (v339_data * (sycl::group_broadcast(item.get_sub_group(), v364_data, 5))));
                float v370_data = r1[5];
                float v373_data = ir2[5];
                ir2[5] = (v373_data + (v339_data * (sycl::group_broadcast(item.get_sub_group(), v370_data, 5))));
                float v376_data = r1[6];
                float v379_data = ir2[6];
                ir2[6] = (v379_data + (v339_data * (sycl::group_broadcast(item.get_sub_group(), v376_data, 5))));
                float v382_data = r1[7];
                float v385_data = ir2[7];
                ir2[7] = (v385_data + (v339_data * (sycl::group_broadcast(item.get_sub_group(), v382_data, 5))));
                float v388_data = r1[8];
                float v391_data = ir2[8];
                ir2[8] = (v391_data + (v339_data * (sycl::group_broadcast(item.get_sub_group(), v388_data, 5))));
              }
              if (v16_lead < 9) {
                float v397_data = r0[6];
                float v398_data = r1[0];
                float v401_data = ir2[0];
                ir2[0] = (v401_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v398_data, 6))));
                float v404_data = r1[1];
                float v407_data = ir2[1];
                ir2[1] = (v407_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v404_data, 6))));
                float v410_data = r1[2];
                float v413_data = ir2[2];
                ir2[2] = (v413_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v410_data, 6))));
                float v416_data = r1[3];
                float v419_data = ir2[3];
                ir2[3] = (v419_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v416_data, 6))));
                float v422_data = r1[4];
                float v425_data = ir2[4];
                ir2[4] = (v425_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v422_data, 6))));
                float v428_data = r1[5];
                float v431_data = ir2[5];
                ir2[5] = (v431_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v428_data, 6))));
                float v434_data = r1[6];
                float v437_data = ir2[6];
                ir2[6] = (v437_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v434_data, 6))));
                float v440_data = r1[7];
                float v443_data = ir2[7];
                ir2[7] = (v443_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v440_data, 6))));
                float v446_data = r1[8];
                float v449_data = ir2[8];
                ir2[8] = (v449_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 6))));
              }
              if (v16_lead < 9) {
                float v455_data = r0[7];
                float v456_data = r1[0];
                float v459_data = ir2[0];
                ir2[0] = (v459_data + (v455_data * (sycl::group_broadcast(item.get_sub_group(), v456_data, 7))));
                float v462_data = r1[1];
                float v465_data = ir2[1];
                ir2[1] = (v465_data + (v455_data * (sycl::group_broadcast(item.get_sub_group(), v462_data, 7))));
                float v468_data = r1[2];
                float v471_data = ir2[2];
                ir2[2] = (v471_data + (v455_data * (sycl::group_broadcast(item.get_sub_group(), v468_data, 7))));
                float v474_data = r1[3];
                float v477_data = ir2[3];
                ir2[3] = (v477_data + (v455_data * (sycl::group_broadcast(item.get_sub_group(), v474_data, 7))));
                float v480_data = r1[4];
                float v483_data = ir2[4];
                ir2[4] = (v483_data + (v455_data * (sycl::group_broadcast(item.get_sub_group(), v480_data, 7))));
                float v486_data = r1[5];
                float v489_data = ir2[5];
                ir2[5] = (v489_data + (v455_data * (sycl::group_broadcast(item.get_sub_group(), v486_data, 7))));
                float v492_data = r1[6];
                float v495_data = ir2[6];
                ir2[6] = (v495_data + (v455_data * (sycl::group_broadcast(item.get_sub_group(), v492_data, 7))));
                float v498_data = r1[7];
                float v501_data = ir2[7];
                ir2[7] = (v501_data + (v455_data * (sycl::group_broadcast(item.get_sub_group(), v498_data, 7))));
                float v504_data = r1[8];
                float v507_data = ir2[8];
                ir2[8] = (v507_data + (v455_data * (sycl::group_broadcast(item.get_sub_group(), v504_data, 7))));
              }
              if (v16_lead < 9) {
                float v513_data = r0[8];
                float v514_data = r1[0];
                float v517_data = ir2[0];
                ir2[0] = (v517_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v514_data, 8))));
                float v520_data = r1[1];
                float v523_data = ir2[1];
                ir2[1] = (v523_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v520_data, 8))));
                float v526_data = r1[2];
                float v529_data = ir2[2];
                ir2[2] = (v529_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v526_data, 8))));
                float v532_data = r1[3];
                float v535_data = ir2[3];
                ir2[3] = (v535_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v532_data, 8))));
                float v538_data = r1[4];
                float v541_data = ir2[4];
                ir2[4] = (v541_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v538_data, 8))));
                float v544_data = r1[5];
                float v547_data = ir2[5];
                ir2[5] = (v547_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v544_data, 8))));
                float v550_data = r1[6];
                float v553_data = ir2[6];
                ir2[6] = (v553_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v550_data, 8))));
                float v556_data = r1[7];
                float v559_data = ir2[7];
                ir2[7] = (v559_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v556_data, 8))));
                float v562_data = r1[8];
                float v565_data = ir2[8];
                ir2[8] = (v565_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v562_data, 8))));
              }
              if (v16_lead < 9) {
                #pragma unroll
                for (int32_t v572_n1 = 0; v572_n1 < 9; ++v572_n1) {
                  float v574_data = ir2[v572_n1];
                  r2[v572_n1] = (v574_data * 13.0f);
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v16_lead < 9) {
                #pragma unroll
                for (int32_t v581_i1 = 0; v581_i1 < 9; ++v581_i1) {
                  float v583_data = r2[v581_i1];
                  glb_m0[(v16_lead + (v581_i1 * 9))] = v583_data;
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


// === base name ===
kernel_811013bfef05a889

// === header ===
void launcher_kernel_811013bfef05a889(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_811013bfef05a889(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_811013bfef05a889(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_811013bfef05a889(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 24×9(24×9) {0..24}×{0..9} strided
        // m1 24×24(24×24) {0..24}×{0..24} strided
        // m2 24×9(24×9) {0..24}×{0..9} strided
        // m0 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[0, 1] = m1 24×24(24×24) {0..24}×{0..24} strided({0..24}×{0..24})[0, -1]×m2 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v0_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v0_batchId0 < numElements0; v0_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v1_ahead1 = v0_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v0_batchId0 * 216 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v0_batchId0 * 576 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v0_batchId0 * 216 + 0 + m2_extraOffset];
              float r0[24]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v14_lead = item.get_local_id(0) % 32;
              if (v14_lead < 24) {
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 24; ++v16_i1) {
                  float v24_data = glb_m1[(v14_lead + (v16_i1 * 24))];
                  r0[v16_i1] = v24_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              if (v14_lead < 24) {
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 9; ++v31_i1) {
                  float v39_data = glb_m2[(v14_lead + (v31_i1 * 24))];
                  r1[v31_i1] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 24), (0, 9)] [(0, 24)]
              float ir2[9]{};
              if (v14_lead < 24) {
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
                float v96_data = r1[8];
                float v99_data = ir2[8];
                ir2[8] = (v99_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 0))));
              }
              if (v14_lead < 24) {
                float v105_data = r0[1];
                float v106_data = r1[0];
                float v109_data = ir2[0];
                ir2[0] = (v109_data + (v105_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 1))));
                float v112_data = r1[1];
                float v115_data = ir2[1];
                ir2[1] = (v115_data + (v105_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 1))));
                float v118_data = r1[2];
                float v121_data = ir2[2];
                ir2[2] = (v121_data + (v105_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 1))));
                float v124_data = r1[3];
                float v127_data = ir2[3];
                ir2[3] = (v127_data + (v105_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 1))));
                float v130_data = r1[4];
                float v133_data = ir2[4];
                ir2[4] = (v133_data + (v105_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 1))));
                float v136_data = r1[5];
                float v139_data = ir2[5];
                ir2[5] = (v139_data + (v105_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 1))));
                float v142_data = r1[6];
                float v145_data = ir2[6];
                ir2[6] = (v145_data + (v105_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 1))));
                float v148_data = r1[7];
                float v151_data = ir2[7];
                ir2[7] = (v151_data + (v105_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 1))));
                float v154_data = r1[8];
                float v157_data = ir2[8];
                ir2[8] = (v157_data + (v105_data * (sycl::group_broadcast(item.get_sub_group(), v154_data, 1))));
              }
              if (v14_lead < 24) {
                float v163_data = r0[2];
                float v164_data = r1[0];
                float v167_data = ir2[0];
                ir2[0] = (v167_data + (v163_data * (sycl::group_broadcast(item.get_sub_group(), v164_data, 2))));
                float v170_data = r1[1];
                float v173_data = ir2[1];
                ir2[1] = (v173_data + (v163_data * (sycl::group_broadcast(item.get_sub_group(), v170_data, 2))));
                float v176_data = r1[2];
                float v179_data = ir2[2];
                ir2[2] = (v179_data + (v163_data * (sycl::group_broadcast(item.get_sub_group(), v176_data, 2))));
                float v182_data = r1[3];
                float v185_data = ir2[3];
                ir2[3] = (v185_data + (v163_data * (sycl::group_broadcast(item.get_sub_group(), v182_data, 2))));
                float v188_data = r1[4];
                float v191_data = ir2[4];
                ir2[4] = (v191_data + (v163_data * (sycl::group_broadcast(item.get_sub_group(), v188_data, 2))));
                float v194_data = r1[5];
                float v197_data = ir2[5];
                ir2[5] = (v197_data + (v163_data * (sycl::group_broadcast(item.get_sub_group(), v194_data, 2))));
                float v200_data = r1[6];
                float v203_data = ir2[6];
                ir2[6] = (v203_data + (v163_data * (sycl::group_broadcast(item.get_sub_group(), v200_data, 2))));
                float v206_data = r1[7];
                float v209_data = ir2[7];
                ir2[7] = (v209_data + (v163_data * (sycl::group_broadcast(item.get_sub_group(), v206_data, 2))));
                float v212_data = r1[8];
                float v215_data = ir2[8];
                ir2[8] = (v215_data + (v163_data * (sycl::group_broadcast(item.get_sub_group(), v212_data, 2))));
              }
              if (v14_lead < 24) {
                float v221_data = r0[3];
                float v222_data = r1[0];
                float v225_data = ir2[0];
                ir2[0] = (v225_data + (v221_data * (sycl::group_broadcast(item.get_sub_group(), v222_data, 3))));
                float v228_data = r1[1];
                float v231_data = ir2[1];
                ir2[1] = (v231_data + (v221_data * (sycl::group_broadcast(item.get_sub_group(), v228_data, 3))));
                float v234_data = r1[2];
                float v237_data = ir2[2];
                ir2[2] = (v237_data + (v221_data * (sycl::group_broadcast(item.get_sub_group(), v234_data, 3))));
                float v240_data = r1[3];
                float v243_data = ir2[3];
                ir2[3] = (v243_data + (v221_data * (sycl::group_broadcast(item.get_sub_group(), v240_data, 3))));
                float v246_data = r1[4];
                float v249_data = ir2[4];
                ir2[4] = (v249_data + (v221_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 3))));
                float v252_data = r1[5];
                float v255_data = ir2[5];
                ir2[5] = (v255_data + (v221_data * (sycl::group_broadcast(item.get_sub_group(), v252_data, 3))));
                float v258_data = r1[6];
                float v261_data = ir2[6];
                ir2[6] = (v261_data + (v221_data * (sycl::group_broadcast(item.get_sub_group(), v258_data, 3))));
                float v264_data = r1[7];
                float v267_data = ir2[7];
                ir2[7] = (v267_data + (v221_data * (sycl::group_broadcast(item.get_sub_group(), v264_data, 3))));
                float v270_data = r1[8];
                float v273_data = ir2[8];
                ir2[8] = (v273_data + (v221_data * (sycl::group_broadcast(item.get_sub_group(), v270_data, 3))));
              }
              if (v14_lead < 24) {
                float v279_data = r0[4];
                float v280_data = r1[0];
                float v283_data = ir2[0];
                ir2[0] = (v283_data + (v279_data * (sycl::group_broadcast(item.get_sub_group(), v280_data, 4))));
                float v286_data = r1[1];
                float v289_data = ir2[1];
                ir2[1] = (v289_data + (v279_data * (sycl::group_broadcast(item.get_sub_group(), v286_data, 4))));
                float v292_data = r1[2];
                float v295_data = ir2[2];
                ir2[2] = (v295_data + (v279_data * (sycl::group_broadcast(item.get_sub_group(), v292_data, 4))));
                float v298_data = r1[3];
                float v301_data = ir2[3];
                ir2[3] = (v301_data + (v279_data * (sycl::group_broadcast(item.get_sub_group(), v298_data, 4))));
                float v304_data = r1[4];
                float v307_data = ir2[4];
                ir2[4] = (v307_data + (v279_data * (sycl::group_broadcast(item.get_sub_group(), v304_data, 4))));
                float v310_data = r1[5];
                float v313_data = ir2[5];
                ir2[5] = (v313_data + (v279_data * (sycl::group_broadcast(item.get_sub_group(), v310_data, 4))));
                float v316_data = r1[6];
                float v319_data = ir2[6];
                ir2[6] = (v319_data + (v279_data * (sycl::group_broadcast(item.get_sub_group(), v316_data, 4))));
                float v322_data = r1[7];
                float v325_data = ir2[7];
                ir2[7] = (v325_data + (v279_data * (sycl::group_broadcast(item.get_sub_group(), v322_data, 4))));
                float v328_data = r1[8];
                float v331_data = ir2[8];
                ir2[8] = (v331_data + (v279_data * (sycl::group_broadcast(item.get_sub_group(), v328_data, 4))));
              }
              if (v14_lead < 24) {
                float v337_data = r0[5];
                float v338_data = r1[0];
                float v341_data = ir2[0];
                ir2[0] = (v341_data + (v337_data * (sycl::group_broadcast(item.get_sub_group(), v338_data, 5))));
                float v344_data = r1[1];
                float v347_data = ir2[1];
                ir2[1] = (v347_data + (v337_data * (sycl::group_broadcast(item.get_sub_group(), v344_data, 5))));
                float v350_data = r1[2];
                float v353_data = ir2[2];
                ir2[2] = (v353_data + (v337_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 5))));
                float v356_data = r1[3];
                float v359_data = ir2[3];
                ir2[3] = (v359_data + (v337_data * (sycl::group_broadcast(item.get_sub_group(), v356_data, 5))));
                float v362_data = r1[4];
                float v365_data = ir2[4];
                ir2[4] = (v365_data + (v337_data * (sycl::group_broadcast(item.get_sub_group(), v362_data, 5))));
                float v368_data = r1[5];
                float v371_data = ir2[5];
                ir2[5] = (v371_data + (v337_data * (sycl::group_broadcast(item.get_sub_group(), v368_data, 5))));
                float v374_data = r1[6];
                float v377_data = ir2[6];
                ir2[6] = (v377_data + (v337_data * (sycl::group_broadcast(item.get_sub_group(), v374_data, 5))));
                float v380_data = r1[7];
                float v383_data = ir2[7];
                ir2[7] = (v383_data + (v337_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 5))));
                float v386_data = r1[8];
                float v389_data = ir2[8];
                ir2[8] = (v389_data + (v337_data * (sycl::group_broadcast(item.get_sub_group(), v386_data, 5))));
              }
              if (v14_lead < 24) {
                float v395_data = r0[6];
                float v396_data = r1[0];
                float v399_data = ir2[0];
                ir2[0] = (v399_data + (v395_data * (sycl::group_broadcast(item.get_sub_group(), v396_data, 6))));
                float v402_data = r1[1];
                float v405_data = ir2[1];
                ir2[1] = (v405_data + (v395_data * (sycl::group_broadcast(item.get_sub_group(), v402_data, 6))));
                float v408_data = r1[2];
                float v411_data = ir2[2];
                ir2[2] = (v411_data + (v395_data * (sycl::group_broadcast(item.get_sub_group(), v408_data, 6))));
                float v414_data = r1[3];
                float v417_data = ir2[3];
                ir2[3] = (v417_data + (v395_data * (sycl::group_broadcast(item.get_sub_group(), v414_data, 6))));
                float v420_data = r1[4];
                float v423_data = ir2[4];
                ir2[4] = (v423_data + (v395_data * (sycl::group_broadcast(item.get_sub_group(), v420_data, 6))));
                float v426_data = r1[5];
                float v429_data = ir2[5];
                ir2[5] = (v429_data + (v395_data * (sycl::group_broadcast(item.get_sub_group(), v426_data, 6))));
                float v432_data = r1[6];
                float v435_data = ir2[6];
                ir2[6] = (v435_data + (v395_data * (sycl::group_broadcast(item.get_sub_group(), v432_data, 6))));
                float v438_data = r1[7];
                float v441_data = ir2[7];
                ir2[7] = (v441_data + (v395_data * (sycl::group_broadcast(item.get_sub_group(), v438_data, 6))));
                float v444_data = r1[8];
                float v447_data = ir2[8];
                ir2[8] = (v447_data + (v395_data * (sycl::group_broadcast(item.get_sub_group(), v444_data, 6))));
              }
              if (v14_lead < 24) {
                float v453_data = r0[7];
                float v454_data = r1[0];
                float v457_data = ir2[0];
                ir2[0] = (v457_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v454_data, 7))));
                float v460_data = r1[1];
                float v463_data = ir2[1];
                ir2[1] = (v463_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v460_data, 7))));
                float v466_data = r1[2];
                float v469_data = ir2[2];
                ir2[2] = (v469_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v466_data, 7))));
                float v472_data = r1[3];
                float v475_data = ir2[3];
                ir2[3] = (v475_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v472_data, 7))));
                float v478_data = r1[4];
                float v481_data = ir2[4];
                ir2[4] = (v481_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v478_data, 7))));
                float v484_data = r1[5];
                float v487_data = ir2[5];
                ir2[5] = (v487_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v484_data, 7))));
                float v490_data = r1[6];
                float v493_data = ir2[6];
                ir2[6] = (v493_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v490_data, 7))));
                float v496_data = r1[7];
                float v499_data = ir2[7];
                ir2[7] = (v499_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v496_data, 7))));
                float v502_data = r1[8];
                float v505_data = ir2[8];
                ir2[8] = (v505_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v502_data, 7))));
              }
              if (v14_lead < 24) {
                float v511_data = r0[8];
                float v512_data = r1[0];
                float v515_data = ir2[0];
                ir2[0] = (v515_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v512_data, 8))));
                float v518_data = r1[1];
                float v521_data = ir2[1];
                ir2[1] = (v521_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v518_data, 8))));
                float v524_data = r1[2];
                float v527_data = ir2[2];
                ir2[2] = (v527_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v524_data, 8))));
                float v530_data = r1[3];
                float v533_data = ir2[3];
                ir2[3] = (v533_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v530_data, 8))));
                float v536_data = r1[4];
                float v539_data = ir2[4];
                ir2[4] = (v539_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v536_data, 8))));
                float v542_data = r1[5];
                float v545_data = ir2[5];
                ir2[5] = (v545_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 8))));
                float v548_data = r1[6];
                float v551_data = ir2[6];
                ir2[6] = (v551_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 8))));
                float v554_data = r1[7];
                float v557_data = ir2[7];
                ir2[7] = (v557_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v554_data, 8))));
                float v560_data = r1[8];
                float v563_data = ir2[8];
                ir2[8] = (v563_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v560_data, 8))));
              }
              if (v14_lead < 24) {
                float v569_data = r0[9];
                float v570_data = r1[0];
                float v573_data = ir2[0];
                ir2[0] = (v573_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v570_data, 9))));
                float v576_data = r1[1];
                float v579_data = ir2[1];
                ir2[1] = (v579_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v576_data, 9))));
                float v582_data = r1[2];
                float v585_data = ir2[2];
                ir2[2] = (v585_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v582_data, 9))));
                float v588_data = r1[3];
                float v591_data = ir2[3];
                ir2[3] = (v591_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v588_data, 9))));
                float v594_data = r1[4];
                float v597_data = ir2[4];
                ir2[4] = (v597_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v594_data, 9))));
                float v600_data = r1[5];
                float v603_data = ir2[5];
                ir2[5] = (v603_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v600_data, 9))));
                float v606_data = r1[6];
                float v609_data = ir2[6];
                ir2[6] = (v609_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v606_data, 9))));
                float v612_data = r1[7];
                float v615_data = ir2[7];
                ir2[7] = (v615_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v612_data, 9))));
                float v618_data = r1[8];
                float v621_data = ir2[8];
                ir2[8] = (v621_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v618_data, 9))));
              }
              if (v14_lead < 24) {
                float v627_data = r0[10];
                float v628_data = r1[0];
                float v631_data = ir2[0];
                ir2[0] = (v631_data + (v627_data * (sycl::group_broadcast(item.get_sub_group(), v628_data, 10))));
                float v634_data = r1[1];
                float v637_data = ir2[1];
                ir2[1] = (v637_data + (v627_data * (sycl::group_broadcast(item.get_sub_group(), v634_data, 10))));
                float v640_data = r1[2];
                float v643_data = ir2[2];
                ir2[2] = (v643_data + (v627_data * (sycl::group_broadcast(item.get_sub_group(), v640_data, 10))));
                float v646_data = r1[3];
                float v649_data = ir2[3];
                ir2[3] = (v649_data + (v627_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 10))));
                float v652_data = r1[4];
                float v655_data = ir2[4];
                ir2[4] = (v655_data + (v627_data * (sycl::group_broadcast(item.get_sub_group(), v652_data, 10))));
                float v658_data = r1[5];
                float v661_data = ir2[5];
                ir2[5] = (v661_data + (v627_data * (sycl::group_broadcast(item.get_sub_group(), v658_data, 10))));
                float v664_data = r1[6];
                float v667_data = ir2[6];
                ir2[6] = (v667_data + (v627_data * (sycl::group_broadcast(item.get_sub_group(), v664_data, 10))));
                float v670_data = r1[7];
                float v673_data = ir2[7];
                ir2[7] = (v673_data + (v627_data * (sycl::group_broadcast(item.get_sub_group(), v670_data, 10))));
                float v676_data = r1[8];
                float v679_data = ir2[8];
                ir2[8] = (v679_data + (v627_data * (sycl::group_broadcast(item.get_sub_group(), v676_data, 10))));
              }
              if (v14_lead < 24) {
                float v685_data = r0[11];
                float v686_data = r1[0];
                float v689_data = ir2[0];
                ir2[0] = (v689_data + (v685_data * (sycl::group_broadcast(item.get_sub_group(), v686_data, 11))));
                float v692_data = r1[1];
                float v695_data = ir2[1];
                ir2[1] = (v695_data + (v685_data * (sycl::group_broadcast(item.get_sub_group(), v692_data, 11))));
                float v698_data = r1[2];
                float v701_data = ir2[2];
                ir2[2] = (v701_data + (v685_data * (sycl::group_broadcast(item.get_sub_group(), v698_data, 11))));
                float v704_data = r1[3];
                float v707_data = ir2[3];
                ir2[3] = (v707_data + (v685_data * (sycl::group_broadcast(item.get_sub_group(), v704_data, 11))));
                float v710_data = r1[4];
                float v713_data = ir2[4];
                ir2[4] = (v713_data + (v685_data * (sycl::group_broadcast(item.get_sub_group(), v710_data, 11))));
                float v716_data = r1[5];
                float v719_data = ir2[5];
                ir2[5] = (v719_data + (v685_data * (sycl::group_broadcast(item.get_sub_group(), v716_data, 11))));
                float v722_data = r1[6];
                float v725_data = ir2[6];
                ir2[6] = (v725_data + (v685_data * (sycl::group_broadcast(item.get_sub_group(), v722_data, 11))));
                float v728_data = r1[7];
                float v731_data = ir2[7];
                ir2[7] = (v731_data + (v685_data * (sycl::group_broadcast(item.get_sub_group(), v728_data, 11))));
                float v734_data = r1[8];
                float v737_data = ir2[8];
                ir2[8] = (v737_data + (v685_data * (sycl::group_broadcast(item.get_sub_group(), v734_data, 11))));
              }
              if (v14_lead < 24) {
                float v743_data = r0[12];
                float v744_data = r1[0];
                float v747_data = ir2[0];
                ir2[0] = (v747_data + (v743_data * (sycl::group_broadcast(item.get_sub_group(), v744_data, 12))));
                float v750_data = r1[1];
                float v753_data = ir2[1];
                ir2[1] = (v753_data + (v743_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 12))));
                float v756_data = r1[2];
                float v759_data = ir2[2];
                ir2[2] = (v759_data + (v743_data * (sycl::group_broadcast(item.get_sub_group(), v756_data, 12))));
                float v762_data = r1[3];
                float v765_data = ir2[3];
                ir2[3] = (v765_data + (v743_data * (sycl::group_broadcast(item.get_sub_group(), v762_data, 12))));
                float v768_data = r1[4];
                float v771_data = ir2[4];
                ir2[4] = (v771_data + (v743_data * (sycl::group_broadcast(item.get_sub_group(), v768_data, 12))));
                float v774_data = r1[5];
                float v777_data = ir2[5];
                ir2[5] = (v777_data + (v743_data * (sycl::group_broadcast(item.get_sub_group(), v774_data, 12))));
                float v780_data = r1[6];
                float v783_data = ir2[6];
                ir2[6] = (v783_data + (v743_data * (sycl::group_broadcast(item.get_sub_group(), v780_data, 12))));
                float v786_data = r1[7];
                float v789_data = ir2[7];
                ir2[7] = (v789_data + (v743_data * (sycl::group_broadcast(item.get_sub_group(), v786_data, 12))));
                float v792_data = r1[8];
                float v795_data = ir2[8];
                ir2[8] = (v795_data + (v743_data * (sycl::group_broadcast(item.get_sub_group(), v792_data, 12))));
              }
              if (v14_lead < 24) {
                float v801_data = r0[13];
                float v802_data = r1[0];
                float v805_data = ir2[0];
                ir2[0] = (v805_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v802_data, 13))));
                float v808_data = r1[1];
                float v811_data = ir2[1];
                ir2[1] = (v811_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v808_data, 13))));
                float v814_data = r1[2];
                float v817_data = ir2[2];
                ir2[2] = (v817_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v814_data, 13))));
                float v820_data = r1[3];
                float v823_data = ir2[3];
                ir2[3] = (v823_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v820_data, 13))));
                float v826_data = r1[4];
                float v829_data = ir2[4];
                ir2[4] = (v829_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v826_data, 13))));
                float v832_data = r1[5];
                float v835_data = ir2[5];
                ir2[5] = (v835_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v832_data, 13))));
                float v838_data = r1[6];
                float v841_data = ir2[6];
                ir2[6] = (v841_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v838_data, 13))));
                float v844_data = r1[7];
                float v847_data = ir2[7];
                ir2[7] = (v847_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v844_data, 13))));
                float v850_data = r1[8];
                float v853_data = ir2[8];
                ir2[8] = (v853_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 13))));
              }
              if (v14_lead < 24) {
                float v859_data = r0[14];
                float v860_data = r1[0];
                float v863_data = ir2[0];
                ir2[0] = (v863_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v860_data, 14))));
                float v866_data = r1[1];
                float v869_data = ir2[1];
                ir2[1] = (v869_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v866_data, 14))));
                float v872_data = r1[2];
                float v875_data = ir2[2];
                ir2[2] = (v875_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v872_data, 14))));
                float v878_data = r1[3];
                float v881_data = ir2[3];
                ir2[3] = (v881_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v878_data, 14))));
                float v884_data = r1[4];
                float v887_data = ir2[4];
                ir2[4] = (v887_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v884_data, 14))));
                float v890_data = r1[5];
                float v893_data = ir2[5];
                ir2[5] = (v893_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v890_data, 14))));
                float v896_data = r1[6];
                float v899_data = ir2[6];
                ir2[6] = (v899_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v896_data, 14))));
                float v902_data = r1[7];
                float v905_data = ir2[7];
                ir2[7] = (v905_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v902_data, 14))));
                float v908_data = r1[8];
                float v911_data = ir2[8];
                ir2[8] = (v911_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v908_data, 14))));
              }
              if (v14_lead < 24) {
                float v917_data = r0[15];
                float v918_data = r1[0];
                float v921_data = ir2[0];
                ir2[0] = (v921_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v918_data, 15))));
                float v924_data = r1[1];
                float v927_data = ir2[1];
                ir2[1] = (v927_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v924_data, 15))));
                float v930_data = r1[2];
                float v933_data = ir2[2];
                ir2[2] = (v933_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v930_data, 15))));
                float v936_data = r1[3];
                float v939_data = ir2[3];
                ir2[3] = (v939_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v936_data, 15))));
                float v942_data = r1[4];
                float v945_data = ir2[4];
                ir2[4] = (v945_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 15))));
                float v948_data = r1[5];
                float v951_data = ir2[5];
                ir2[5] = (v951_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 15))));
                float v954_data = r1[6];
                float v957_data = ir2[6];
                ir2[6] = (v957_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v954_data, 15))));
                float v960_data = r1[7];
                float v963_data = ir2[7];
                ir2[7] = (v963_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v960_data, 15))));
                float v966_data = r1[8];
                float v969_data = ir2[8];
                ir2[8] = (v969_data + (v917_data * (sycl::group_broadcast(item.get_sub_group(), v966_data, 15))));
              }
              if (v14_lead < 24) {
                float v975_data = r0[16];
                float v976_data = r1[0];
                float v979_data = ir2[0];
                ir2[0] = (v979_data + (v975_data * (sycl::group_broadcast(item.get_sub_group(), v976_data, 16))));
                float v982_data = r1[1];
                float v985_data = ir2[1];
                ir2[1] = (v985_data + (v975_data * (sycl::group_broadcast(item.get_sub_group(), v982_data, 16))));
                float v988_data = r1[2];
                float v991_data = ir2[2];
                ir2[2] = (v991_data + (v975_data * (sycl::group_broadcast(item.get_sub_group(), v988_data, 16))));
                float v994_data = r1[3];
                float v997_data = ir2[3];
                ir2[3] = (v997_data + (v975_data * (sycl::group_broadcast(item.get_sub_group(), v994_data, 16))));
                float v1000_data = r1[4];
                float v1003_data = ir2[4];
                ir2[4] = (v1003_data + (v975_data * (sycl::group_broadcast(item.get_sub_group(), v1000_data, 16))));
                float v1006_data = r1[5];
                float v1009_data = ir2[5];
                ir2[5] = (v1009_data + (v975_data * (sycl::group_broadcast(item.get_sub_group(), v1006_data, 16))));
                float v1012_data = r1[6];
                float v1015_data = ir2[6];
                ir2[6] = (v1015_data + (v975_data * (sycl::group_broadcast(item.get_sub_group(), v1012_data, 16))));
                float v1018_data = r1[7];
                float v1021_data = ir2[7];
                ir2[7] = (v1021_data + (v975_data * (sycl::group_broadcast(item.get_sub_group(), v1018_data, 16))));
                float v1024_data = r1[8];
                float v1027_data = ir2[8];
                ir2[8] = (v1027_data + (v975_data * (sycl::group_broadcast(item.get_sub_group(), v1024_data, 16))));
              }
              if (v14_lead < 24) {
                float v1033_data = r0[17];
                float v1034_data = r1[0];
                float v1037_data = ir2[0];
                ir2[0] = (v1037_data + (v1033_data * (sycl::group_broadcast(item.get_sub_group(), v1034_data, 17))));
                float v1040_data = r1[1];
                float v1043_data = ir2[1];
                ir2[1] = (v1043_data + (v1033_data * (sycl::group_broadcast(item.get_sub_group(), v1040_data, 17))));
                float v1046_data = r1[2];
                float v1049_data = ir2[2];
                ir2[2] = (v1049_data + (v1033_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 17))));
                float v1052_data = r1[3];
                float v1055_data = ir2[3];
                ir2[3] = (v1055_data + (v1033_data * (sycl::group_broadcast(item.get_sub_group(), v1052_data, 17))));
                float v1058_data = r1[4];
                float v1061_data = ir2[4];
                ir2[4] = (v1061_data + (v1033_data * (sycl::group_broadcast(item.get_sub_group(), v1058_data, 17))));
                float v1064_data = r1[5];
                float v1067_data = ir2[5];
                ir2[5] = (v1067_data + (v1033_data * (sycl::group_broadcast(item.get_sub_group(), v1064_data, 17))));
                float v1070_data = r1[6];
                float v1073_data = ir2[6];
                ir2[6] = (v1073_data + (v1033_data * (sycl::group_broadcast(item.get_sub_group(), v1070_data, 17))));
                float v1076_data = r1[7];
                float v1079_data = ir2[7];
                ir2[7] = (v1079_data + (v1033_data * (sycl::group_broadcast(item.get_sub_group(), v1076_data, 17))));
                float v1082_data = r1[8];
                float v1085_data = ir2[8];
                ir2[8] = (v1085_data + (v1033_data * (sycl::group_broadcast(item.get_sub_group(), v1082_data, 17))));
              }
              if (v14_lead < 24) {
                float v1091_data = r0[18];
                float v1092_data = r1[0];
                float v1095_data = ir2[0];
                ir2[0] = (v1095_data + (v1091_data * (sycl::group_broadcast(item.get_sub_group(), v1092_data, 18))));
                float v1098_data = r1[1];
                float v1101_data = ir2[1];
                ir2[1] = (v1101_data + (v1091_data * (sycl::group_broadcast(item.get_sub_group(), v1098_data, 18))));
                float v1104_data = r1[2];
                float v1107_data = ir2[2];
                ir2[2] = (v1107_data + (v1091_data * (sycl::group_broadcast(item.get_sub_group(), v1104_data, 18))));
                float v1110_data = r1[3];
                float v1113_data = ir2[3];
                ir2[3] = (v1113_data + (v1091_data * (sycl::group_broadcast(item.get_sub_group(), v1110_data, 18))));
                float v1116_data = r1[4];
                float v1119_data = ir2[4];
                ir2[4] = (v1119_data + (v1091_data * (sycl::group_broadcast(item.get_sub_group(), v1116_data, 18))));
                float v1122_data = r1[5];
                float v1125_data = ir2[5];
                ir2[5] = (v1125_data + (v1091_data * (sycl::group_broadcast(item.get_sub_group(), v1122_data, 18))));
                float v1128_data = r1[6];
                float v1131_data = ir2[6];
                ir2[6] = (v1131_data + (v1091_data * (sycl::group_broadcast(item.get_sub_group(), v1128_data, 18))));
                float v1134_data = r1[7];
                float v1137_data = ir2[7];
                ir2[7] = (v1137_data + (v1091_data * (sycl::group_broadcast(item.get_sub_group(), v1134_data, 18))));
                float v1140_data = r1[8];
                float v1143_data = ir2[8];
                ir2[8] = (v1143_data + (v1091_data * (sycl::group_broadcast(item.get_sub_group(), v1140_data, 18))));
              }
              if (v14_lead < 24) {
                float v1149_data = r0[19];
                float v1150_data = r1[0];
                float v1153_data = ir2[0];
                ir2[0] = (v1153_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1150_data, 19))));
                float v1156_data = r1[1];
                float v1159_data = ir2[1];
                ir2[1] = (v1159_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1156_data, 19))));
                float v1162_data = r1[2];
                float v1165_data = ir2[2];
                ir2[2] = (v1165_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1162_data, 19))));
                float v1168_data = r1[3];
                float v1171_data = ir2[3];
                ir2[3] = (v1171_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1168_data, 19))));
                float v1174_data = r1[4];
                float v1177_data = ir2[4];
                ir2[4] = (v1177_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1174_data, 19))));
                float v1180_data = r1[5];
                float v1183_data = ir2[5];
                ir2[5] = (v1183_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1180_data, 19))));
                float v1186_data = r1[6];
                float v1189_data = ir2[6];
                ir2[6] = (v1189_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1186_data, 19))));
                float v1192_data = r1[7];
                float v1195_data = ir2[7];
                ir2[7] = (v1195_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1192_data, 19))));
                float v1198_data = r1[8];
                float v1201_data = ir2[8];
                ir2[8] = (v1201_data + (v1149_data * (sycl::group_broadcast(item.get_sub_group(), v1198_data, 19))));
              }
              if (v14_lead < 24) {
                float v1207_data = r0[20];
                float v1208_data = r1[0];
                float v1211_data = ir2[0];
                ir2[0] = (v1211_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v1208_data, 20))));
                float v1214_data = r1[1];
                float v1217_data = ir2[1];
                ir2[1] = (v1217_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v1214_data, 20))));
                float v1220_data = r1[2];
                float v1223_data = ir2[2];
                ir2[2] = (v1223_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v1220_data, 20))));
                float v1226_data = r1[3];
                float v1229_data = ir2[3];
                ir2[3] = (v1229_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v1226_data, 20))));
                float v1232_data = r1[4];
                float v1235_data = ir2[4];
                ir2[4] = (v1235_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v1232_data, 20))));
                float v1238_data = r1[5];
                float v1241_data = ir2[5];
                ir2[5] = (v1241_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v1238_data, 20))));
                float v1244_data = r1[6];
                float v1247_data = ir2[6];
                ir2[6] = (v1247_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v1244_data, 20))));
                float v1250_data = r1[7];
                float v1253_data = ir2[7];
                ir2[7] = (v1253_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v1250_data, 20))));
                float v1256_data = r1[8];
                float v1259_data = ir2[8];
                ir2[8] = (v1259_data + (v1207_data * (sycl::group_broadcast(item.get_sub_group(), v1256_data, 20))));
              }
              if (v14_lead < 24) {
                float v1265_data = r0[21];
                float v1266_data = r1[0];
                float v1269_data = ir2[0];
                ir2[0] = (v1269_data + (v1265_data * (sycl::group_broadcast(item.get_sub_group(), v1266_data, 21))));
                float v1272_data = r1[1];
                float v1275_data = ir2[1];
                ir2[1] = (v1275_data + (v1265_data * (sycl::group_broadcast(item.get_sub_group(), v1272_data, 21))));
                float v1278_data = r1[2];
                float v1281_data = ir2[2];
                ir2[2] = (v1281_data + (v1265_data * (sycl::group_broadcast(item.get_sub_group(), v1278_data, 21))));
                float v1284_data = r1[3];
                float v1287_data = ir2[3];
                ir2[3] = (v1287_data + (v1265_data * (sycl::group_broadcast(item.get_sub_group(), v1284_data, 21))));
                float v1290_data = r1[4];
                float v1293_data = ir2[4];
                ir2[4] = (v1293_data + (v1265_data * (sycl::group_broadcast(item.get_sub_group(), v1290_data, 21))));
                float v1296_data = r1[5];
                float v1299_data = ir2[5];
                ir2[5] = (v1299_data + (v1265_data * (sycl::group_broadcast(item.get_sub_group(), v1296_data, 21))));
                float v1302_data = r1[6];
                float v1305_data = ir2[6];
                ir2[6] = (v1305_data + (v1265_data * (sycl::group_broadcast(item.get_sub_group(), v1302_data, 21))));
                float v1308_data = r1[7];
                float v1311_data = ir2[7];
                ir2[7] = (v1311_data + (v1265_data * (sycl::group_broadcast(item.get_sub_group(), v1308_data, 21))));
                float v1314_data = r1[8];
                float v1317_data = ir2[8];
                ir2[8] = (v1317_data + (v1265_data * (sycl::group_broadcast(item.get_sub_group(), v1314_data, 21))));
              }
              if (v14_lead < 24) {
                float v1323_data = r0[22];
                float v1324_data = r1[0];
                float v1327_data = ir2[0];
                ir2[0] = (v1327_data + (v1323_data * (sycl::group_broadcast(item.get_sub_group(), v1324_data, 22))));
                float v1330_data = r1[1];
                float v1333_data = ir2[1];
                ir2[1] = (v1333_data + (v1323_data * (sycl::group_broadcast(item.get_sub_group(), v1330_data, 22))));
                float v1336_data = r1[2];
                float v1339_data = ir2[2];
                ir2[2] = (v1339_data + (v1323_data * (sycl::group_broadcast(item.get_sub_group(), v1336_data, 22))));
                float v1342_data = r1[3];
                float v1345_data = ir2[3];
                ir2[3] = (v1345_data + (v1323_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 22))));
                float v1348_data = r1[4];
                float v1351_data = ir2[4];
                ir2[4] = (v1351_data + (v1323_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 22))));
                float v1354_data = r1[5];
                float v1357_data = ir2[5];
                ir2[5] = (v1357_data + (v1323_data * (sycl::group_broadcast(item.get_sub_group(), v1354_data, 22))));
                float v1360_data = r1[6];
                float v1363_data = ir2[6];
                ir2[6] = (v1363_data + (v1323_data * (sycl::group_broadcast(item.get_sub_group(), v1360_data, 22))));
                float v1366_data = r1[7];
                float v1369_data = ir2[7];
                ir2[7] = (v1369_data + (v1323_data * (sycl::group_broadcast(item.get_sub_group(), v1366_data, 22))));
                float v1372_data = r1[8];
                float v1375_data = ir2[8];
                ir2[8] = (v1375_data + (v1323_data * (sycl::group_broadcast(item.get_sub_group(), v1372_data, 22))));
              }
              if (v14_lead < 24) {
                float v1381_data = r0[23];
                float v1382_data = r1[0];
                float v1385_data = ir2[0];
                ir2[0] = (v1385_data + (v1381_data * (sycl::group_broadcast(item.get_sub_group(), v1382_data, 23))));
                float v1388_data = r1[1];
                float v1391_data = ir2[1];
                ir2[1] = (v1391_data + (v1381_data * (sycl::group_broadcast(item.get_sub_group(), v1388_data, 23))));
                float v1394_data = r1[2];
                float v1397_data = ir2[2];
                ir2[2] = (v1397_data + (v1381_data * (sycl::group_broadcast(item.get_sub_group(), v1394_data, 23))));
                float v1400_data = r1[3];
                float v1403_data = ir2[3];
                ir2[3] = (v1403_data + (v1381_data * (sycl::group_broadcast(item.get_sub_group(), v1400_data, 23))));
                float v1406_data = r1[4];
                float v1409_data = ir2[4];
                ir2[4] = (v1409_data + (v1381_data * (sycl::group_broadcast(item.get_sub_group(), v1406_data, 23))));
                float v1412_data = r1[5];
                float v1415_data = ir2[5];
                ir2[5] = (v1415_data + (v1381_data * (sycl::group_broadcast(item.get_sub_group(), v1412_data, 23))));
                float v1418_data = r1[6];
                float v1421_data = ir2[6];
                ir2[6] = (v1421_data + (v1381_data * (sycl::group_broadcast(item.get_sub_group(), v1418_data, 23))));
                float v1424_data = r1[7];
                float v1427_data = ir2[7];
                ir2[7] = (v1427_data + (v1381_data * (sycl::group_broadcast(item.get_sub_group(), v1424_data, 23))));
                float v1430_data = r1[8];
                float v1433_data = ir2[8];
                ir2[8] = (v1433_data + (v1381_data * (sycl::group_broadcast(item.get_sub_group(), v1430_data, 23))));
              }
              if (v14_lead < 24) {
                #pragma unroll
                for (int32_t v1439_n1 = 0; v1439_n1 < 9; ++v1439_n1) {
                  float v1441_data = ir2[v1439_n1];
                  r2[v1439_n1] = v1441_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v14_lead < 24) {
                #pragma unroll
                for (int32_t v1447_i1 = 0; v1447_i1 < 9; ++v1447_i1) {
                  float v1449_data = r2[v1447_i1];
                  glb_m0[(v14_lead + (v1447_i1 * 24))] = v1449_data;
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


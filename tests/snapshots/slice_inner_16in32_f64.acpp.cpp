// === base name ===
kernel_61a53253b634aa0f

// === header ===
void launcher_kernel_61a53253b634aa0f(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_61a53253b634aa0f(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_61a53253b634aa0f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_61a53253b634aa0f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 32×32(32×32) {0..32}×{0..32} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v2_batchId0 * 128 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v2_batchId0 * 1024 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v2_batchId0 * 128 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v24_off = (v16_lead + (v17_i0 * 16)) + 8;
                #pragma unroll
                for (int32_t v18_i1 = 8; v18_i1 < 24; ++v18_i1) {
                  double v27_data = glb_m1[(v24_off + (v18_i1 * 32))];
                  r0[(v17_i0 + (v18_i1 - 8))] = v27_data;
                }
              }
              double r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
                int32_t v40_lead = v16_lead + (v34_i0 * 16);
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
                  double v43_data = glb_m2[(v40_lead + (v35_i1 * 16))];
                  r1[(v34_i0 + v35_i1)] = v43_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              double r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              double ir2[8]{};
              double v50_data = r0[0];
              double v51_data = r1[0];
              double v54_data = ir2[0];
              ir2[0] = (v54_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 0))));
              double v57_data = r1[1];
              double v60_data = ir2[1];
              ir2[1] = (v60_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 0))));
              double v63_data = r1[2];
              double v66_data = ir2[2];
              ir2[2] = (v66_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 0))));
              double v69_data = r1[3];
              double v72_data = ir2[3];
              ir2[3] = (v72_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 0))));
              double v75_data = r1[4];
              double v78_data = ir2[4];
              ir2[4] = (v78_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 0))));
              double v81_data = r1[5];
              double v84_data = ir2[5];
              ir2[5] = (v84_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 0))));
              double v87_data = r1[6];
              double v90_data = ir2[6];
              ir2[6] = (v90_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 0))));
              double v93_data = r1[7];
              double v96_data = ir2[7];
              ir2[7] = (v96_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 0))));
              double v101_data = r0[1];
              double v105_data = ir2[0];
              ir2[0] = (v105_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 1))));
              double v111_data = ir2[1];
              ir2[1] = (v111_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 1))));
              double v117_data = ir2[2];
              ir2[2] = (v117_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 1))));
              double v123_data = ir2[3];
              ir2[3] = (v123_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 1))));
              double v129_data = ir2[4];
              ir2[4] = (v129_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 1))));
              double v135_data = ir2[5];
              ir2[5] = (v135_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 1))));
              double v141_data = ir2[6];
              ir2[6] = (v141_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 1))));
              double v147_data = ir2[7];
              ir2[7] = (v147_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 1))));
              double v152_data = r0[2];
              double v156_data = ir2[0];
              ir2[0] = (v156_data + (v152_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 2))));
              double v162_data = ir2[1];
              ir2[1] = (v162_data + (v152_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 2))));
              double v168_data = ir2[2];
              ir2[2] = (v168_data + (v152_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 2))));
              double v174_data = ir2[3];
              ir2[3] = (v174_data + (v152_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 2))));
              double v180_data = ir2[4];
              ir2[4] = (v180_data + (v152_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 2))));
              double v186_data = ir2[5];
              ir2[5] = (v186_data + (v152_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 2))));
              double v192_data = ir2[6];
              ir2[6] = (v192_data + (v152_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 2))));
              double v198_data = ir2[7];
              ir2[7] = (v198_data + (v152_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 2))));
              double v203_data = r0[3];
              double v207_data = ir2[0];
              ir2[0] = (v207_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 3))));
              double v213_data = ir2[1];
              ir2[1] = (v213_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 3))));
              double v219_data = ir2[2];
              ir2[2] = (v219_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 3))));
              double v225_data = ir2[3];
              ir2[3] = (v225_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 3))));
              double v231_data = ir2[4];
              ir2[4] = (v231_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 3))));
              double v237_data = ir2[5];
              ir2[5] = (v237_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 3))));
              double v243_data = ir2[6];
              ir2[6] = (v243_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 3))));
              double v249_data = ir2[7];
              ir2[7] = (v249_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 3))));
              double v254_data = r0[4];
              double v258_data = ir2[0];
              ir2[0] = (v258_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 4))));
              double v264_data = ir2[1];
              ir2[1] = (v264_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 4))));
              double v270_data = ir2[2];
              ir2[2] = (v270_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 4))));
              double v276_data = ir2[3];
              ir2[3] = (v276_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 4))));
              double v282_data = ir2[4];
              ir2[4] = (v282_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 4))));
              double v288_data = ir2[5];
              ir2[5] = (v288_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 4))));
              double v294_data = ir2[6];
              ir2[6] = (v294_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 4))));
              double v300_data = ir2[7];
              ir2[7] = (v300_data + (v254_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 4))));
              double v305_data = r0[5];
              double v309_data = ir2[0];
              ir2[0] = (v309_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 5))));
              double v315_data = ir2[1];
              ir2[1] = (v315_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 5))));
              double v321_data = ir2[2];
              ir2[2] = (v321_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 5))));
              double v327_data = ir2[3];
              ir2[3] = (v327_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 5))));
              double v333_data = ir2[4];
              ir2[4] = (v333_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 5))));
              double v339_data = ir2[5];
              ir2[5] = (v339_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 5))));
              double v345_data = ir2[6];
              ir2[6] = (v345_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 5))));
              double v351_data = ir2[7];
              ir2[7] = (v351_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 5))));
              double v356_data = r0[6];
              double v360_data = ir2[0];
              ir2[0] = (v360_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 6))));
              double v366_data = ir2[1];
              ir2[1] = (v366_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 6))));
              double v372_data = ir2[2];
              ir2[2] = (v372_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 6))));
              double v378_data = ir2[3];
              ir2[3] = (v378_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 6))));
              double v384_data = ir2[4];
              ir2[4] = (v384_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 6))));
              double v390_data = ir2[5];
              ir2[5] = (v390_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 6))));
              double v396_data = ir2[6];
              ir2[6] = (v396_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 6))));
              double v402_data = ir2[7];
              ir2[7] = (v402_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 6))));
              double v407_data = r0[7];
              double v411_data = ir2[0];
              ir2[0] = (v411_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 7))));
              double v417_data = ir2[1];
              ir2[1] = (v417_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 7))));
              double v423_data = ir2[2];
              ir2[2] = (v423_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 7))));
              double v429_data = ir2[3];
              ir2[3] = (v429_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 7))));
              double v435_data = ir2[4];
              ir2[4] = (v435_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 7))));
              double v441_data = ir2[5];
              ir2[5] = (v441_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 7))));
              double v447_data = ir2[6];
              ir2[6] = (v447_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 7))));
              double v453_data = ir2[7];
              ir2[7] = (v453_data + (v407_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 7))));
              double v458_data = r0[8];
              double v462_data = ir2[0];
              ir2[0] = (v462_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 8))));
              double v468_data = ir2[1];
              ir2[1] = (v468_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 8))));
              double v474_data = ir2[2];
              ir2[2] = (v474_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 8))));
              double v480_data = ir2[3];
              ir2[3] = (v480_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 8))));
              double v486_data = ir2[4];
              ir2[4] = (v486_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 8))));
              double v492_data = ir2[5];
              ir2[5] = (v492_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 8))));
              double v498_data = ir2[6];
              ir2[6] = (v498_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 8))));
              double v504_data = ir2[7];
              ir2[7] = (v504_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 8))));
              double v509_data = r0[9];
              double v513_data = ir2[0];
              ir2[0] = (v513_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 9))));
              double v519_data = ir2[1];
              ir2[1] = (v519_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 9))));
              double v525_data = ir2[2];
              ir2[2] = (v525_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 9))));
              double v531_data = ir2[3];
              ir2[3] = (v531_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 9))));
              double v537_data = ir2[4];
              ir2[4] = (v537_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 9))));
              double v543_data = ir2[5];
              ir2[5] = (v543_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 9))));
              double v549_data = ir2[6];
              ir2[6] = (v549_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 9))));
              double v555_data = ir2[7];
              ir2[7] = (v555_data + (v509_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 9))));
              double v560_data = r0[10];
              double v564_data = ir2[0];
              ir2[0] = (v564_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 10))));
              double v570_data = ir2[1];
              ir2[1] = (v570_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 10))));
              double v576_data = ir2[2];
              ir2[2] = (v576_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 10))));
              double v582_data = ir2[3];
              ir2[3] = (v582_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 10))));
              double v588_data = ir2[4];
              ir2[4] = (v588_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 10))));
              double v594_data = ir2[5];
              ir2[5] = (v594_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 10))));
              double v600_data = ir2[6];
              ir2[6] = (v600_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 10))));
              double v606_data = ir2[7];
              ir2[7] = (v606_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 10))));
              double v611_data = r0[11];
              double v615_data = ir2[0];
              ir2[0] = (v615_data + (v611_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 11))));
              double v621_data = ir2[1];
              ir2[1] = (v621_data + (v611_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 11))));
              double v627_data = ir2[2];
              ir2[2] = (v627_data + (v611_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 11))));
              double v633_data = ir2[3];
              ir2[3] = (v633_data + (v611_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 11))));
              double v639_data = ir2[4];
              ir2[4] = (v639_data + (v611_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 11))));
              double v645_data = ir2[5];
              ir2[5] = (v645_data + (v611_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 11))));
              double v651_data = ir2[6];
              ir2[6] = (v651_data + (v611_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 11))));
              double v657_data = ir2[7];
              ir2[7] = (v657_data + (v611_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 11))));
              double v662_data = r0[12];
              double v666_data = ir2[0];
              ir2[0] = (v666_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 12))));
              double v672_data = ir2[1];
              ir2[1] = (v672_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 12))));
              double v678_data = ir2[2];
              ir2[2] = (v678_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 12))));
              double v684_data = ir2[3];
              ir2[3] = (v684_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 12))));
              double v690_data = ir2[4];
              ir2[4] = (v690_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 12))));
              double v696_data = ir2[5];
              ir2[5] = (v696_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 12))));
              double v702_data = ir2[6];
              ir2[6] = (v702_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 12))));
              double v708_data = ir2[7];
              ir2[7] = (v708_data + (v662_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 12))));
              double v713_data = r0[13];
              double v717_data = ir2[0];
              ir2[0] = (v717_data + (v713_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 13))));
              double v723_data = ir2[1];
              ir2[1] = (v723_data + (v713_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 13))));
              double v729_data = ir2[2];
              ir2[2] = (v729_data + (v713_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 13))));
              double v735_data = ir2[3];
              ir2[3] = (v735_data + (v713_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 13))));
              double v741_data = ir2[4];
              ir2[4] = (v741_data + (v713_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 13))));
              double v747_data = ir2[5];
              ir2[5] = (v747_data + (v713_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 13))));
              double v753_data = ir2[6];
              ir2[6] = (v753_data + (v713_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 13))));
              double v759_data = ir2[7];
              ir2[7] = (v759_data + (v713_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 13))));
              double v764_data = r0[14];
              double v768_data = ir2[0];
              ir2[0] = (v768_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 14))));
              double v774_data = ir2[1];
              ir2[1] = (v774_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 14))));
              double v780_data = ir2[2];
              ir2[2] = (v780_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 14))));
              double v786_data = ir2[3];
              ir2[3] = (v786_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 14))));
              double v792_data = ir2[4];
              ir2[4] = (v792_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 14))));
              double v798_data = ir2[5];
              ir2[5] = (v798_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 14))));
              double v804_data = ir2[6];
              ir2[6] = (v804_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 14))));
              double v810_data = ir2[7];
              ir2[7] = (v810_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 14))));
              double v815_data = r0[15];
              double v819_data = ir2[0];
              ir2[0] = (v819_data + (v815_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 15))));
              double v825_data = ir2[1];
              ir2[1] = (v825_data + (v815_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 15))));
              double v831_data = ir2[2];
              ir2[2] = (v831_data + (v815_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 15))));
              double v837_data = ir2[3];
              ir2[3] = (v837_data + (v815_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 15))));
              double v843_data = ir2[4];
              ir2[4] = (v843_data + (v815_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 15))));
              double v849_data = ir2[5];
              ir2[5] = (v849_data + (v815_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 15))));
              double v855_data = ir2[6];
              ir2[6] = (v855_data + (v815_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 15))));
              double v861_data = ir2[7];
              ir2[7] = (v861_data + (v815_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 15))));
              #pragma unroll
              for (int32_t v866_n0 = 0; v866_n0 < 1; ++v866_n0) {
                #pragma unroll
                for (int32_t v867_n1 = 0; v867_n1 < 8; ++v867_n1) {
                  int32_t v868_a = v866_n0 + v867_n1;
                  double v869_data = ir2[v868_a];
                  r2[v868_a] = v869_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v874_i0 = 0; v874_i0 < 1; ++v874_i0) {
                int32_t v882_lead = v16_lead + (v874_i0 * 16);
                #pragma unroll
                for (int32_t v875_i1 = 0; v875_i1 < 8; ++v875_i1) {
                  double v877_data = r2[(v874_i0 + v875_i1)];
                  glb_m0[(v882_lead + (v875_i1 * 16))] = v877_data;
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


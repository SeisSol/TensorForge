// === base name ===
kernel_968ec30ac7367816

// === header ===
void launcher_kernel_968ec30ac7367816(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_968ec30ac7367816(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_968ec30ac7367816(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_968ec30ac7367816(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v12_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
                int32_t v19_lead = v12_lead + (v13_i0 * 16);
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 16; ++v14_i1) {
                  float v22_data = glb_m1[(v19_lead + (v14_i1 * 16))];
                  r0[(v13_i0 + v14_i1)] = v22_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
                int32_t v34_lead = v12_lead + (v28_i0 * 16);
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
                  float v37_data = glb_m2[(v34_lead + (v29_i1 * 16))];
                  r1[(v28_i0 + v29_i1)] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v44_data = r0[0];
              float v45_data = r1[0];
              float v48_data = ir2[0];
              ir2[0] = (v48_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 0))));
              float v51_data = r1[1];
              float v54_data = ir2[1];
              ir2[1] = (v54_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 0))));
              float v57_data = r1[2];
              float v60_data = ir2[2];
              ir2[2] = (v60_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 0))));
              float v63_data = r1[3];
              float v66_data = ir2[3];
              ir2[3] = (v66_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 0))));
              float v69_data = r1[4];
              float v72_data = ir2[4];
              ir2[4] = (v72_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 0))));
              float v75_data = r1[5];
              float v78_data = ir2[5];
              ir2[5] = (v78_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 0))));
              float v81_data = r1[6];
              float v84_data = ir2[6];
              ir2[6] = (v84_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 0))));
              float v87_data = r1[7];
              float v90_data = ir2[7];
              ir2[7] = (v90_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 0))));
              float v95_data = r0[1];
              float v99_data = ir2[0];
              ir2[0] = (v99_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 1))));
              float v105_data = ir2[1];
              ir2[1] = (v105_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 1))));
              float v111_data = ir2[2];
              ir2[2] = (v111_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 1))));
              float v117_data = ir2[3];
              ir2[3] = (v117_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 1))));
              float v123_data = ir2[4];
              ir2[4] = (v123_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 1))));
              float v129_data = ir2[5];
              ir2[5] = (v129_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 1))));
              float v135_data = ir2[6];
              ir2[6] = (v135_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 1))));
              float v141_data = ir2[7];
              ir2[7] = (v141_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 1))));
              float v146_data = r0[2];
              float v150_data = ir2[0];
              ir2[0] = (v150_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 2))));
              float v156_data = ir2[1];
              ir2[1] = (v156_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 2))));
              float v162_data = ir2[2];
              ir2[2] = (v162_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 2))));
              float v168_data = ir2[3];
              ir2[3] = (v168_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 2))));
              float v174_data = ir2[4];
              ir2[4] = (v174_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 2))));
              float v180_data = ir2[5];
              ir2[5] = (v180_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 2))));
              float v186_data = ir2[6];
              ir2[6] = (v186_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 2))));
              float v192_data = ir2[7];
              ir2[7] = (v192_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 2))));
              float v197_data = r0[3];
              float v201_data = ir2[0];
              ir2[0] = (v201_data + (v197_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 3))));
              float v207_data = ir2[1];
              ir2[1] = (v207_data + (v197_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 3))));
              float v213_data = ir2[2];
              ir2[2] = (v213_data + (v197_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 3))));
              float v219_data = ir2[3];
              ir2[3] = (v219_data + (v197_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 3))));
              float v225_data = ir2[4];
              ir2[4] = (v225_data + (v197_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 3))));
              float v231_data = ir2[5];
              ir2[5] = (v231_data + (v197_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 3))));
              float v237_data = ir2[6];
              ir2[6] = (v237_data + (v197_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 3))));
              float v243_data = ir2[7];
              ir2[7] = (v243_data + (v197_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 3))));
              float v248_data = r0[4];
              float v252_data = ir2[0];
              ir2[0] = (v252_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 4))));
              float v258_data = ir2[1];
              ir2[1] = (v258_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 4))));
              float v264_data = ir2[2];
              ir2[2] = (v264_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 4))));
              float v270_data = ir2[3];
              ir2[3] = (v270_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 4))));
              float v276_data = ir2[4];
              ir2[4] = (v276_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 4))));
              float v282_data = ir2[5];
              ir2[5] = (v282_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 4))));
              float v288_data = ir2[6];
              ir2[6] = (v288_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 4))));
              float v294_data = ir2[7];
              ir2[7] = (v294_data + (v248_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 4))));
              float v299_data = r0[5];
              float v303_data = ir2[0];
              ir2[0] = (v303_data + (v299_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 5))));
              float v309_data = ir2[1];
              ir2[1] = (v309_data + (v299_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 5))));
              float v315_data = ir2[2];
              ir2[2] = (v315_data + (v299_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 5))));
              float v321_data = ir2[3];
              ir2[3] = (v321_data + (v299_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 5))));
              float v327_data = ir2[4];
              ir2[4] = (v327_data + (v299_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 5))));
              float v333_data = ir2[5];
              ir2[5] = (v333_data + (v299_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 5))));
              float v339_data = ir2[6];
              ir2[6] = (v339_data + (v299_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 5))));
              float v345_data = ir2[7];
              ir2[7] = (v345_data + (v299_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 5))));
              float v350_data = r0[6];
              float v354_data = ir2[0];
              ir2[0] = (v354_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 6))));
              float v360_data = ir2[1];
              ir2[1] = (v360_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 6))));
              float v366_data = ir2[2];
              ir2[2] = (v366_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 6))));
              float v372_data = ir2[3];
              ir2[3] = (v372_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 6))));
              float v378_data = ir2[4];
              ir2[4] = (v378_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 6))));
              float v384_data = ir2[5];
              ir2[5] = (v384_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 6))));
              float v390_data = ir2[6];
              ir2[6] = (v390_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 6))));
              float v396_data = ir2[7];
              ir2[7] = (v396_data + (v350_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 6))));
              float v401_data = r0[7];
              float v405_data = ir2[0];
              ir2[0] = (v405_data + (v401_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 7))));
              float v411_data = ir2[1];
              ir2[1] = (v411_data + (v401_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 7))));
              float v417_data = ir2[2];
              ir2[2] = (v417_data + (v401_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 7))));
              float v423_data = ir2[3];
              ir2[3] = (v423_data + (v401_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 7))));
              float v429_data = ir2[4];
              ir2[4] = (v429_data + (v401_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 7))));
              float v435_data = ir2[5];
              ir2[5] = (v435_data + (v401_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 7))));
              float v441_data = ir2[6];
              ir2[6] = (v441_data + (v401_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 7))));
              float v447_data = ir2[7];
              ir2[7] = (v447_data + (v401_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 7))));
              float v452_data = r0[8];
              float v456_data = ir2[0];
              ir2[0] = (v456_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 8))));
              float v462_data = ir2[1];
              ir2[1] = (v462_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 8))));
              float v468_data = ir2[2];
              ir2[2] = (v468_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 8))));
              float v474_data = ir2[3];
              ir2[3] = (v474_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 8))));
              float v480_data = ir2[4];
              ir2[4] = (v480_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 8))));
              float v486_data = ir2[5];
              ir2[5] = (v486_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 8))));
              float v492_data = ir2[6];
              ir2[6] = (v492_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 8))));
              float v498_data = ir2[7];
              ir2[7] = (v498_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 8))));
              float v503_data = r0[9];
              float v507_data = ir2[0];
              ir2[0] = (v507_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 9))));
              float v513_data = ir2[1];
              ir2[1] = (v513_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 9))));
              float v519_data = ir2[2];
              ir2[2] = (v519_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 9))));
              float v525_data = ir2[3];
              ir2[3] = (v525_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 9))));
              float v531_data = ir2[4];
              ir2[4] = (v531_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 9))));
              float v537_data = ir2[5];
              ir2[5] = (v537_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 9))));
              float v543_data = ir2[6];
              ir2[6] = (v543_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 9))));
              float v549_data = ir2[7];
              ir2[7] = (v549_data + (v503_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 9))));
              float v554_data = r0[10];
              float v558_data = ir2[0];
              ir2[0] = (v558_data + (v554_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 10))));
              float v564_data = ir2[1];
              ir2[1] = (v564_data + (v554_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 10))));
              float v570_data = ir2[2];
              ir2[2] = (v570_data + (v554_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 10))));
              float v576_data = ir2[3];
              ir2[3] = (v576_data + (v554_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 10))));
              float v582_data = ir2[4];
              ir2[4] = (v582_data + (v554_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 10))));
              float v588_data = ir2[5];
              ir2[5] = (v588_data + (v554_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 10))));
              float v594_data = ir2[6];
              ir2[6] = (v594_data + (v554_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 10))));
              float v600_data = ir2[7];
              ir2[7] = (v600_data + (v554_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 10))));
              float v605_data = r0[11];
              float v609_data = ir2[0];
              ir2[0] = (v609_data + (v605_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 11))));
              float v615_data = ir2[1];
              ir2[1] = (v615_data + (v605_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 11))));
              float v621_data = ir2[2];
              ir2[2] = (v621_data + (v605_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 11))));
              float v627_data = ir2[3];
              ir2[3] = (v627_data + (v605_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 11))));
              float v633_data = ir2[4];
              ir2[4] = (v633_data + (v605_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 11))));
              float v639_data = ir2[5];
              ir2[5] = (v639_data + (v605_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 11))));
              float v645_data = ir2[6];
              ir2[6] = (v645_data + (v605_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 11))));
              float v651_data = ir2[7];
              ir2[7] = (v651_data + (v605_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 11))));
              float v656_data = r0[12];
              float v660_data = ir2[0];
              ir2[0] = (v660_data + (v656_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 12))));
              float v666_data = ir2[1];
              ir2[1] = (v666_data + (v656_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 12))));
              float v672_data = ir2[2];
              ir2[2] = (v672_data + (v656_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 12))));
              float v678_data = ir2[3];
              ir2[3] = (v678_data + (v656_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 12))));
              float v684_data = ir2[4];
              ir2[4] = (v684_data + (v656_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 12))));
              float v690_data = ir2[5];
              ir2[5] = (v690_data + (v656_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 12))));
              float v696_data = ir2[6];
              ir2[6] = (v696_data + (v656_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 12))));
              float v702_data = ir2[7];
              ir2[7] = (v702_data + (v656_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 12))));
              float v707_data = r0[13];
              float v711_data = ir2[0];
              ir2[0] = (v711_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 13))));
              float v717_data = ir2[1];
              ir2[1] = (v717_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 13))));
              float v723_data = ir2[2];
              ir2[2] = (v723_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 13))));
              float v729_data = ir2[3];
              ir2[3] = (v729_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 13))));
              float v735_data = ir2[4];
              ir2[4] = (v735_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 13))));
              float v741_data = ir2[5];
              ir2[5] = (v741_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 13))));
              float v747_data = ir2[6];
              ir2[6] = (v747_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 13))));
              float v753_data = ir2[7];
              ir2[7] = (v753_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 13))));
              float v758_data = r0[14];
              float v762_data = ir2[0];
              ir2[0] = (v762_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 14))));
              float v768_data = ir2[1];
              ir2[1] = (v768_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 14))));
              float v774_data = ir2[2];
              ir2[2] = (v774_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 14))));
              float v780_data = ir2[3];
              ir2[3] = (v780_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 14))));
              float v786_data = ir2[4];
              ir2[4] = (v786_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 14))));
              float v792_data = ir2[5];
              ir2[5] = (v792_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 14))));
              float v798_data = ir2[6];
              ir2[6] = (v798_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 14))));
              float v804_data = ir2[7];
              ir2[7] = (v804_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 14))));
              float v809_data = r0[15];
              float v813_data = ir2[0];
              ir2[0] = (v813_data + (v809_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 15))));
              float v819_data = ir2[1];
              ir2[1] = (v819_data + (v809_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 15))));
              float v825_data = ir2[2];
              ir2[2] = (v825_data + (v809_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 15))));
              float v831_data = ir2[3];
              ir2[3] = (v831_data + (v809_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 15))));
              float v837_data = ir2[4];
              ir2[4] = (v837_data + (v809_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 15))));
              float v843_data = ir2[5];
              ir2[5] = (v843_data + (v809_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 15))));
              float v849_data = ir2[6];
              ir2[6] = (v849_data + (v809_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 15))));
              float v855_data = ir2[7];
              ir2[7] = (v855_data + (v809_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 15))));
              #pragma unroll
              for (int32_t v860_n0 = 0; v860_n0 < 1; ++v860_n0) {
                #pragma unroll
                for (int32_t v861_n1 = 0; v861_n1 < 8; ++v861_n1) {
                  int32_t v862_a = v860_n0 + v861_n1;
                  float v863_data = ir2[v862_a];
                  r2[v862_a] = v863_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v868_i0 = 0; v868_i0 < 1; ++v868_i0) {
                int32_t v876_lead = v12_lead + (v868_i0 * 16);
                #pragma unroll
                for (int32_t v869_i1 = 0; v869_i1 < 8; ++v869_i1) {
                  float v871_data = r2[(v868_i0 + v869_i1)];
                  glb_m0[(v876_lead + (v869_i1 * 16))] = v871_data;
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


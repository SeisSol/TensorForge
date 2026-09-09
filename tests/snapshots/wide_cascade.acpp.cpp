// === base name ===
kernel_100e2294729ed346

// === header ===
void launcher_kernel_100e2294729ed346(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_100e2294729ed346(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_100e2294729ed346(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_100e2294729ed346(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×11(16×11) {0..16}×{0..11} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×11(16×11) {0..16}×{0..11} strided
        // m0 16×11(16×11) {0..16}×{0..11} strided({0..16}×{0..11})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×11(16×11) {0..16}×{0..11} strided({0..16}×{0..11})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 176 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 176 + 0 + m2_extraOffset];
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
              float r1[11]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
                int32_t v34_lead = v12_lead + (v28_i0 * 16);
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 11; ++v29_i1) {
                  float v37_data = glb_m2[(v34_lead + (v29_i1 * 16))];
                  r1[(v28_i0 + v29_i1)] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[11]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 11)] [(0, 16)]
              float ir2[11]{};
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
              float v93_data = r1[8];
              float v96_data = ir2[8];
              ir2[8] = (v96_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 0))));
              float v99_data = r1[9];
              float v102_data = ir2[9];
              ir2[9] = (v102_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 0))));
              float v105_data = r1[10];
              float v108_data = ir2[10];
              ir2[10] = (v108_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 0))));
              float v113_data = r0[1];
              float v117_data = ir2[0];
              ir2[0] = (v117_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 1))));
              float v123_data = ir2[1];
              ir2[1] = (v123_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 1))));
              float v129_data = ir2[2];
              ir2[2] = (v129_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 1))));
              float v135_data = ir2[3];
              ir2[3] = (v135_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 1))));
              float v141_data = ir2[4];
              ir2[4] = (v141_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 1))));
              float v147_data = ir2[5];
              ir2[5] = (v147_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 1))));
              float v153_data = ir2[6];
              ir2[6] = (v153_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 1))));
              float v159_data = ir2[7];
              ir2[7] = (v159_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 1))));
              float v165_data = ir2[8];
              ir2[8] = (v165_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 1))));
              float v171_data = ir2[9];
              ir2[9] = (v171_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 1))));
              float v177_data = ir2[10];
              ir2[10] = (v177_data + (v113_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 1))));
              float v182_data = r0[2];
              float v186_data = ir2[0];
              ir2[0] = (v186_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 2))));
              float v192_data = ir2[1];
              ir2[1] = (v192_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 2))));
              float v198_data = ir2[2];
              ir2[2] = (v198_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 2))));
              float v204_data = ir2[3];
              ir2[3] = (v204_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 2))));
              float v210_data = ir2[4];
              ir2[4] = (v210_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 2))));
              float v216_data = ir2[5];
              ir2[5] = (v216_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 2))));
              float v222_data = ir2[6];
              ir2[6] = (v222_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 2))));
              float v228_data = ir2[7];
              ir2[7] = (v228_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 2))));
              float v234_data = ir2[8];
              ir2[8] = (v234_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 2))));
              float v240_data = ir2[9];
              ir2[9] = (v240_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 2))));
              float v246_data = ir2[10];
              ir2[10] = (v246_data + (v182_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 2))));
              float v251_data = r0[3];
              float v255_data = ir2[0];
              ir2[0] = (v255_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 3))));
              float v261_data = ir2[1];
              ir2[1] = (v261_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 3))));
              float v267_data = ir2[2];
              ir2[2] = (v267_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 3))));
              float v273_data = ir2[3];
              ir2[3] = (v273_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 3))));
              float v279_data = ir2[4];
              ir2[4] = (v279_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 3))));
              float v285_data = ir2[5];
              ir2[5] = (v285_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 3))));
              float v291_data = ir2[6];
              ir2[6] = (v291_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 3))));
              float v297_data = ir2[7];
              ir2[7] = (v297_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 3))));
              float v303_data = ir2[8];
              ir2[8] = (v303_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 3))));
              float v309_data = ir2[9];
              ir2[9] = (v309_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 3))));
              float v315_data = ir2[10];
              ir2[10] = (v315_data + (v251_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 3))));
              float v320_data = r0[4];
              float v324_data = ir2[0];
              ir2[0] = (v324_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 4))));
              float v330_data = ir2[1];
              ir2[1] = (v330_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 4))));
              float v336_data = ir2[2];
              ir2[2] = (v336_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 4))));
              float v342_data = ir2[3];
              ir2[3] = (v342_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 4))));
              float v348_data = ir2[4];
              ir2[4] = (v348_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 4))));
              float v354_data = ir2[5];
              ir2[5] = (v354_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 4))));
              float v360_data = ir2[6];
              ir2[6] = (v360_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 4))));
              float v366_data = ir2[7];
              ir2[7] = (v366_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 4))));
              float v372_data = ir2[8];
              ir2[8] = (v372_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 4))));
              float v378_data = ir2[9];
              ir2[9] = (v378_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 4))));
              float v384_data = ir2[10];
              ir2[10] = (v384_data + (v320_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 4))));
              float v389_data = r0[5];
              float v393_data = ir2[0];
              ir2[0] = (v393_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 5))));
              float v399_data = ir2[1];
              ir2[1] = (v399_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 5))));
              float v405_data = ir2[2];
              ir2[2] = (v405_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 5))));
              float v411_data = ir2[3];
              ir2[3] = (v411_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 5))));
              float v417_data = ir2[4];
              ir2[4] = (v417_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 5))));
              float v423_data = ir2[5];
              ir2[5] = (v423_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 5))));
              float v429_data = ir2[6];
              ir2[6] = (v429_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 5))));
              float v435_data = ir2[7];
              ir2[7] = (v435_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 5))));
              float v441_data = ir2[8];
              ir2[8] = (v441_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 5))));
              float v447_data = ir2[9];
              ir2[9] = (v447_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 5))));
              float v453_data = ir2[10];
              ir2[10] = (v453_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 5))));
              float v458_data = r0[6];
              float v462_data = ir2[0];
              ir2[0] = (v462_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 6))));
              float v468_data = ir2[1];
              ir2[1] = (v468_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 6))));
              float v474_data = ir2[2];
              ir2[2] = (v474_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 6))));
              float v480_data = ir2[3];
              ir2[3] = (v480_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 6))));
              float v486_data = ir2[4];
              ir2[4] = (v486_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 6))));
              float v492_data = ir2[5];
              ir2[5] = (v492_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 6))));
              float v498_data = ir2[6];
              ir2[6] = (v498_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 6))));
              float v504_data = ir2[7];
              ir2[7] = (v504_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 6))));
              float v510_data = ir2[8];
              ir2[8] = (v510_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 6))));
              float v516_data = ir2[9];
              ir2[9] = (v516_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 6))));
              float v522_data = ir2[10];
              ir2[10] = (v522_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 6))));
              float v527_data = r0[7];
              float v531_data = ir2[0];
              ir2[0] = (v531_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 7))));
              float v537_data = ir2[1];
              ir2[1] = (v537_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 7))));
              float v543_data = ir2[2];
              ir2[2] = (v543_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 7))));
              float v549_data = ir2[3];
              ir2[3] = (v549_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 7))));
              float v555_data = ir2[4];
              ir2[4] = (v555_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 7))));
              float v561_data = ir2[5];
              ir2[5] = (v561_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 7))));
              float v567_data = ir2[6];
              ir2[6] = (v567_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 7))));
              float v573_data = ir2[7];
              ir2[7] = (v573_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 7))));
              float v579_data = ir2[8];
              ir2[8] = (v579_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 7))));
              float v585_data = ir2[9];
              ir2[9] = (v585_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 7))));
              float v591_data = ir2[10];
              ir2[10] = (v591_data + (v527_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 7))));
              float v596_data = r0[8];
              float v600_data = ir2[0];
              ir2[0] = (v600_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 8))));
              float v606_data = ir2[1];
              ir2[1] = (v606_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 8))));
              float v612_data = ir2[2];
              ir2[2] = (v612_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 8))));
              float v618_data = ir2[3];
              ir2[3] = (v618_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 8))));
              float v624_data = ir2[4];
              ir2[4] = (v624_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 8))));
              float v630_data = ir2[5];
              ir2[5] = (v630_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 8))));
              float v636_data = ir2[6];
              ir2[6] = (v636_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 8))));
              float v642_data = ir2[7];
              ir2[7] = (v642_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 8))));
              float v648_data = ir2[8];
              ir2[8] = (v648_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 8))));
              float v654_data = ir2[9];
              ir2[9] = (v654_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 8))));
              float v660_data = ir2[10];
              ir2[10] = (v660_data + (v596_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 8))));
              float v665_data = r0[9];
              float v669_data = ir2[0];
              ir2[0] = (v669_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 9))));
              float v675_data = ir2[1];
              ir2[1] = (v675_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 9))));
              float v681_data = ir2[2];
              ir2[2] = (v681_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 9))));
              float v687_data = ir2[3];
              ir2[3] = (v687_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 9))));
              float v693_data = ir2[4];
              ir2[4] = (v693_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 9))));
              float v699_data = ir2[5];
              ir2[5] = (v699_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 9))));
              float v705_data = ir2[6];
              ir2[6] = (v705_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 9))));
              float v711_data = ir2[7];
              ir2[7] = (v711_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 9))));
              float v717_data = ir2[8];
              ir2[8] = (v717_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 9))));
              float v723_data = ir2[9];
              ir2[9] = (v723_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 9))));
              float v729_data = ir2[10];
              ir2[10] = (v729_data + (v665_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 9))));
              float v734_data = r0[10];
              float v738_data = ir2[0];
              ir2[0] = (v738_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 10))));
              float v744_data = ir2[1];
              ir2[1] = (v744_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 10))));
              float v750_data = ir2[2];
              ir2[2] = (v750_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 10))));
              float v756_data = ir2[3];
              ir2[3] = (v756_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 10))));
              float v762_data = ir2[4];
              ir2[4] = (v762_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 10))));
              float v768_data = ir2[5];
              ir2[5] = (v768_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 10))));
              float v774_data = ir2[6];
              ir2[6] = (v774_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 10))));
              float v780_data = ir2[7];
              ir2[7] = (v780_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 10))));
              float v786_data = ir2[8];
              ir2[8] = (v786_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 10))));
              float v792_data = ir2[9];
              ir2[9] = (v792_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 10))));
              float v798_data = ir2[10];
              ir2[10] = (v798_data + (v734_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 10))));
              float v803_data = r0[11];
              float v807_data = ir2[0];
              ir2[0] = (v807_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 11))));
              float v813_data = ir2[1];
              ir2[1] = (v813_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 11))));
              float v819_data = ir2[2];
              ir2[2] = (v819_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 11))));
              float v825_data = ir2[3];
              ir2[3] = (v825_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 11))));
              float v831_data = ir2[4];
              ir2[4] = (v831_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 11))));
              float v837_data = ir2[5];
              ir2[5] = (v837_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 11))));
              float v843_data = ir2[6];
              ir2[6] = (v843_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 11))));
              float v849_data = ir2[7];
              ir2[7] = (v849_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 11))));
              float v855_data = ir2[8];
              ir2[8] = (v855_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 11))));
              float v861_data = ir2[9];
              ir2[9] = (v861_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 11))));
              float v867_data = ir2[10];
              ir2[10] = (v867_data + (v803_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 11))));
              float v872_data = r0[12];
              float v876_data = ir2[0];
              ir2[0] = (v876_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 12))));
              float v882_data = ir2[1];
              ir2[1] = (v882_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 12))));
              float v888_data = ir2[2];
              ir2[2] = (v888_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 12))));
              float v894_data = ir2[3];
              ir2[3] = (v894_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 12))));
              float v900_data = ir2[4];
              ir2[4] = (v900_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 12))));
              float v906_data = ir2[5];
              ir2[5] = (v906_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 12))));
              float v912_data = ir2[6];
              ir2[6] = (v912_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 12))));
              float v918_data = ir2[7];
              ir2[7] = (v918_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 12))));
              float v924_data = ir2[8];
              ir2[8] = (v924_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 12))));
              float v930_data = ir2[9];
              ir2[9] = (v930_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 12))));
              float v936_data = ir2[10];
              ir2[10] = (v936_data + (v872_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 12))));
              float v941_data = r0[13];
              float v945_data = ir2[0];
              ir2[0] = (v945_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 13))));
              float v951_data = ir2[1];
              ir2[1] = (v951_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 13))));
              float v957_data = ir2[2];
              ir2[2] = (v957_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 13))));
              float v963_data = ir2[3];
              ir2[3] = (v963_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 13))));
              float v969_data = ir2[4];
              ir2[4] = (v969_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 13))));
              float v975_data = ir2[5];
              ir2[5] = (v975_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 13))));
              float v981_data = ir2[6];
              ir2[6] = (v981_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 13))));
              float v987_data = ir2[7];
              ir2[7] = (v987_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 13))));
              float v993_data = ir2[8];
              ir2[8] = (v993_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 13))));
              float v999_data = ir2[9];
              ir2[9] = (v999_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 13))));
              float v1005_data = ir2[10];
              ir2[10] = (v1005_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 13))));
              float v1010_data = r0[14];
              float v1014_data = ir2[0];
              ir2[0] = (v1014_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 14))));
              float v1020_data = ir2[1];
              ir2[1] = (v1020_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 14))));
              float v1026_data = ir2[2];
              ir2[2] = (v1026_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 14))));
              float v1032_data = ir2[3];
              ir2[3] = (v1032_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 14))));
              float v1038_data = ir2[4];
              ir2[4] = (v1038_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 14))));
              float v1044_data = ir2[5];
              ir2[5] = (v1044_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 14))));
              float v1050_data = ir2[6];
              ir2[6] = (v1050_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 14))));
              float v1056_data = ir2[7];
              ir2[7] = (v1056_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 14))));
              float v1062_data = ir2[8];
              ir2[8] = (v1062_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 14))));
              float v1068_data = ir2[9];
              ir2[9] = (v1068_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 14))));
              float v1074_data = ir2[10];
              ir2[10] = (v1074_data + (v1010_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 14))));
              float v1079_data = r0[15];
              float v1083_data = ir2[0];
              ir2[0] = (v1083_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 15))));
              float v1089_data = ir2[1];
              ir2[1] = (v1089_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 15))));
              float v1095_data = ir2[2];
              ir2[2] = (v1095_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 15))));
              float v1101_data = ir2[3];
              ir2[3] = (v1101_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 15))));
              float v1107_data = ir2[4];
              ir2[4] = (v1107_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 15))));
              float v1113_data = ir2[5];
              ir2[5] = (v1113_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 15))));
              float v1119_data = ir2[6];
              ir2[6] = (v1119_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 15))));
              float v1125_data = ir2[7];
              ir2[7] = (v1125_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 15))));
              float v1131_data = ir2[8];
              ir2[8] = (v1131_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 15))));
              float v1137_data = ir2[9];
              ir2[9] = (v1137_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 15))));
              float v1143_data = ir2[10];
              ir2[10] = (v1143_data + (v1079_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 15))));
              #pragma unroll
              for (int32_t v1148_n0 = 0; v1148_n0 < 1; ++v1148_n0) {
                #pragma unroll
                for (int32_t v1149_n1 = 0; v1149_n1 < 11; ++v1149_n1) {
                  int32_t v1150_a = v1148_n0 + v1149_n1;
                  float v1151_data = ir2[v1150_a];
                  r2[v1150_a] = v1151_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1156_i0 = 0; v1156_i0 < 1; ++v1156_i0) {
                int32_t v1164_lead = v12_lead + (v1156_i0 * 16);
                #pragma unroll
                for (int32_t v1157_i1 = 0; v1157_i1 < 11; ++v1157_i1) {
                  float v1159_data = r2[(v1156_i0 + v1157_i1)];
                  glb_m0[(v1164_lead + (v1157_i1 * 16))] = v1159_data;
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


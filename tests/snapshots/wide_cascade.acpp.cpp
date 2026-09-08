// === base name ===
kernel_5c7e4b9ce8

// === header ===
void launcher_kernel_5c7e4b9ce8(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5c7e4b9ce8(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5c7e4b9ce8(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5c7e4b9ce8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 176 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 176 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v15_lead = v8_lead + (v9_i0 * 16);
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 16; ++v10_i1) {
                  float v18_data = glb_m1[(v15_lead + (v10_i1 * 16))];
                  r0[(v9_i0 + v10_i1)] = v18_data;
                }
              }
              float r1[11]{};
              // r1 = load{g>r}(glb_m2);
              sycl::vec<float, 4> v21_lin = *(sycl::vec<float, 4>*)&glb_m2[0 + item.get_local_id(0) * 4];
              *(sycl::vec<float, 4>*)&r1[0] = v21_lin;
              sycl::vec<float, 4> v22_lin = *(sycl::vec<float, 4>*)&glb_m2[64 + item.get_local_id(0) * 4];
              *(sycl::vec<float, 4>*)&r1[4] = v22_lin;
              sycl::vec<float, 2> v23_lin = *(sycl::vec<float, 2>*)&glb_m2[128 + item.get_local_id(0) * 2];
              *(sycl::vec<float, 2>*)&r1[8] = v23_lin;
              float v24_lin = glb_m2[160 + item.get_local_id(0) * 1];
              r1[10] = v24_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[11]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 11)] [(0, 16)]
              float ir2[11]{};
              float v30_data = r0[0];
              float v31_data = r1[0];
              float v34_data = ir2[0];
              ir2[0] = (v34_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 0))));
              float v37_data = r1[1];
              float v40_data = ir2[1];
              ir2[1] = (v40_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 0))));
              float v43_data = r1[2];
              float v46_data = ir2[2];
              ir2[2] = (v46_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 0))));
              float v49_data = r1[3];
              float v52_data = ir2[3];
              ir2[3] = (v52_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 0))));
              float v55_data = r1[4];
              float v58_data = ir2[4];
              ir2[4] = (v58_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 0))));
              float v61_data = r1[5];
              float v64_data = ir2[5];
              ir2[5] = (v64_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
              float v67_data = r1[6];
              float v70_data = ir2[6];
              ir2[6] = (v70_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
              float v73_data = r1[7];
              float v76_data = ir2[7];
              ir2[7] = (v76_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
              float v79_data = r1[8];
              float v82_data = ir2[8];
              ir2[8] = (v82_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
              float v85_data = r1[9];
              float v88_data = ir2[9];
              ir2[9] = (v88_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
              float v91_data = r1[10];
              float v94_data = ir2[10];
              ir2[10] = (v94_data + (v30_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 0))));
              float v99_data = r0[1];
              float v103_data = ir2[0];
              ir2[0] = (v103_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 1))));
              float v109_data = ir2[1];
              ir2[1] = (v109_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 1))));
              float v115_data = ir2[2];
              ir2[2] = (v115_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 1))));
              float v121_data = ir2[3];
              ir2[3] = (v121_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 1))));
              float v127_data = ir2[4];
              ir2[4] = (v127_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 1))));
              float v133_data = ir2[5];
              ir2[5] = (v133_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 1))));
              float v139_data = ir2[6];
              ir2[6] = (v139_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 1))));
              float v145_data = ir2[7];
              ir2[7] = (v145_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 1))));
              float v151_data = ir2[8];
              ir2[8] = (v151_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 1))));
              float v157_data = ir2[9];
              ir2[9] = (v157_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 1))));
              float v163_data = ir2[10];
              ir2[10] = (v163_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 1))));
              float v168_data = r0[2];
              float v172_data = ir2[0];
              ir2[0] = (v172_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 2))));
              float v178_data = ir2[1];
              ir2[1] = (v178_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 2))));
              float v184_data = ir2[2];
              ir2[2] = (v184_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 2))));
              float v190_data = ir2[3];
              ir2[3] = (v190_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 2))));
              float v196_data = ir2[4];
              ir2[4] = (v196_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 2))));
              float v202_data = ir2[5];
              ir2[5] = (v202_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 2))));
              float v208_data = ir2[6];
              ir2[6] = (v208_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 2))));
              float v214_data = ir2[7];
              ir2[7] = (v214_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 2))));
              float v220_data = ir2[8];
              ir2[8] = (v220_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 2))));
              float v226_data = ir2[9];
              ir2[9] = (v226_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 2))));
              float v232_data = ir2[10];
              ir2[10] = (v232_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 2))));
              float v237_data = r0[3];
              float v241_data = ir2[0];
              ir2[0] = (v241_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 3))));
              float v247_data = ir2[1];
              ir2[1] = (v247_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 3))));
              float v253_data = ir2[2];
              ir2[2] = (v253_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 3))));
              float v259_data = ir2[3];
              ir2[3] = (v259_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 3))));
              float v265_data = ir2[4];
              ir2[4] = (v265_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 3))));
              float v271_data = ir2[5];
              ir2[5] = (v271_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 3))));
              float v277_data = ir2[6];
              ir2[6] = (v277_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 3))));
              float v283_data = ir2[7];
              ir2[7] = (v283_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 3))));
              float v289_data = ir2[8];
              ir2[8] = (v289_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 3))));
              float v295_data = ir2[9];
              ir2[9] = (v295_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 3))));
              float v301_data = ir2[10];
              ir2[10] = (v301_data + (v237_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 3))));
              float v306_data = r0[4];
              float v310_data = ir2[0];
              ir2[0] = (v310_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 4))));
              float v316_data = ir2[1];
              ir2[1] = (v316_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 4))));
              float v322_data = ir2[2];
              ir2[2] = (v322_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 4))));
              float v328_data = ir2[3];
              ir2[3] = (v328_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 4))));
              float v334_data = ir2[4];
              ir2[4] = (v334_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 4))));
              float v340_data = ir2[5];
              ir2[5] = (v340_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 4))));
              float v346_data = ir2[6];
              ir2[6] = (v346_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 4))));
              float v352_data = ir2[7];
              ir2[7] = (v352_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 4))));
              float v358_data = ir2[8];
              ir2[8] = (v358_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 4))));
              float v364_data = ir2[9];
              ir2[9] = (v364_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 4))));
              float v370_data = ir2[10];
              ir2[10] = (v370_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 4))));
              float v375_data = r0[5];
              float v379_data = ir2[0];
              ir2[0] = (v379_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 5))));
              float v385_data = ir2[1];
              ir2[1] = (v385_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 5))));
              float v391_data = ir2[2];
              ir2[2] = (v391_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 5))));
              float v397_data = ir2[3];
              ir2[3] = (v397_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 5))));
              float v403_data = ir2[4];
              ir2[4] = (v403_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 5))));
              float v409_data = ir2[5];
              ir2[5] = (v409_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 5))));
              float v415_data = ir2[6];
              ir2[6] = (v415_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 5))));
              float v421_data = ir2[7];
              ir2[7] = (v421_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 5))));
              float v427_data = ir2[8];
              ir2[8] = (v427_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 5))));
              float v433_data = ir2[9];
              ir2[9] = (v433_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 5))));
              float v439_data = ir2[10];
              ir2[10] = (v439_data + (v375_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 5))));
              float v444_data = r0[6];
              float v448_data = ir2[0];
              ir2[0] = (v448_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 6))));
              float v454_data = ir2[1];
              ir2[1] = (v454_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 6))));
              float v460_data = ir2[2];
              ir2[2] = (v460_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 6))));
              float v466_data = ir2[3];
              ir2[3] = (v466_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 6))));
              float v472_data = ir2[4];
              ir2[4] = (v472_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 6))));
              float v478_data = ir2[5];
              ir2[5] = (v478_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 6))));
              float v484_data = ir2[6];
              ir2[6] = (v484_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 6))));
              float v490_data = ir2[7];
              ir2[7] = (v490_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 6))));
              float v496_data = ir2[8];
              ir2[8] = (v496_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 6))));
              float v502_data = ir2[9];
              ir2[9] = (v502_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 6))));
              float v508_data = ir2[10];
              ir2[10] = (v508_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 6))));
              float v513_data = r0[7];
              float v517_data = ir2[0];
              ir2[0] = (v517_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 7))));
              float v523_data = ir2[1];
              ir2[1] = (v523_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 7))));
              float v529_data = ir2[2];
              ir2[2] = (v529_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 7))));
              float v535_data = ir2[3];
              ir2[3] = (v535_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 7))));
              float v541_data = ir2[4];
              ir2[4] = (v541_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 7))));
              float v547_data = ir2[5];
              ir2[5] = (v547_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 7))));
              float v553_data = ir2[6];
              ir2[6] = (v553_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 7))));
              float v559_data = ir2[7];
              ir2[7] = (v559_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 7))));
              float v565_data = ir2[8];
              ir2[8] = (v565_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 7))));
              float v571_data = ir2[9];
              ir2[9] = (v571_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 7))));
              float v577_data = ir2[10];
              ir2[10] = (v577_data + (v513_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 7))));
              float v582_data = r0[8];
              float v586_data = ir2[0];
              ir2[0] = (v586_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 8))));
              float v592_data = ir2[1];
              ir2[1] = (v592_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 8))));
              float v598_data = ir2[2];
              ir2[2] = (v598_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 8))));
              float v604_data = ir2[3];
              ir2[3] = (v604_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 8))));
              float v610_data = ir2[4];
              ir2[4] = (v610_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 8))));
              float v616_data = ir2[5];
              ir2[5] = (v616_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 8))));
              float v622_data = ir2[6];
              ir2[6] = (v622_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 8))));
              float v628_data = ir2[7];
              ir2[7] = (v628_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 8))));
              float v634_data = ir2[8];
              ir2[8] = (v634_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 8))));
              float v640_data = ir2[9];
              ir2[9] = (v640_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 8))));
              float v646_data = ir2[10];
              ir2[10] = (v646_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 8))));
              float v651_data = r0[9];
              float v655_data = ir2[0];
              ir2[0] = (v655_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 9))));
              float v661_data = ir2[1];
              ir2[1] = (v661_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 9))));
              float v667_data = ir2[2];
              ir2[2] = (v667_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 9))));
              float v673_data = ir2[3];
              ir2[3] = (v673_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 9))));
              float v679_data = ir2[4];
              ir2[4] = (v679_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 9))));
              float v685_data = ir2[5];
              ir2[5] = (v685_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 9))));
              float v691_data = ir2[6];
              ir2[6] = (v691_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 9))));
              float v697_data = ir2[7];
              ir2[7] = (v697_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 9))));
              float v703_data = ir2[8];
              ir2[8] = (v703_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 9))));
              float v709_data = ir2[9];
              ir2[9] = (v709_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 9))));
              float v715_data = ir2[10];
              ir2[10] = (v715_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 9))));
              float v720_data = r0[10];
              float v724_data = ir2[0];
              ir2[0] = (v724_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 10))));
              float v730_data = ir2[1];
              ir2[1] = (v730_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 10))));
              float v736_data = ir2[2];
              ir2[2] = (v736_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 10))));
              float v742_data = ir2[3];
              ir2[3] = (v742_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 10))));
              float v748_data = ir2[4];
              ir2[4] = (v748_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 10))));
              float v754_data = ir2[5];
              ir2[5] = (v754_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 10))));
              float v760_data = ir2[6];
              ir2[6] = (v760_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 10))));
              float v766_data = ir2[7];
              ir2[7] = (v766_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 10))));
              float v772_data = ir2[8];
              ir2[8] = (v772_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 10))));
              float v778_data = ir2[9];
              ir2[9] = (v778_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 10))));
              float v784_data = ir2[10];
              ir2[10] = (v784_data + (v720_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 10))));
              float v789_data = r0[11];
              float v793_data = ir2[0];
              ir2[0] = (v793_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 11))));
              float v799_data = ir2[1];
              ir2[1] = (v799_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 11))));
              float v805_data = ir2[2];
              ir2[2] = (v805_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 11))));
              float v811_data = ir2[3];
              ir2[3] = (v811_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 11))));
              float v817_data = ir2[4];
              ir2[4] = (v817_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 11))));
              float v823_data = ir2[5];
              ir2[5] = (v823_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 11))));
              float v829_data = ir2[6];
              ir2[6] = (v829_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 11))));
              float v835_data = ir2[7];
              ir2[7] = (v835_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 11))));
              float v841_data = ir2[8];
              ir2[8] = (v841_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 11))));
              float v847_data = ir2[9];
              ir2[9] = (v847_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 11))));
              float v853_data = ir2[10];
              ir2[10] = (v853_data + (v789_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 11))));
              float v858_data = r0[12];
              float v862_data = ir2[0];
              ir2[0] = (v862_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 12))));
              float v868_data = ir2[1];
              ir2[1] = (v868_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 12))));
              float v874_data = ir2[2];
              ir2[2] = (v874_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 12))));
              float v880_data = ir2[3];
              ir2[3] = (v880_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 12))));
              float v886_data = ir2[4];
              ir2[4] = (v886_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 12))));
              float v892_data = ir2[5];
              ir2[5] = (v892_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 12))));
              float v898_data = ir2[6];
              ir2[6] = (v898_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 12))));
              float v904_data = ir2[7];
              ir2[7] = (v904_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 12))));
              float v910_data = ir2[8];
              ir2[8] = (v910_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 12))));
              float v916_data = ir2[9];
              ir2[9] = (v916_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 12))));
              float v922_data = ir2[10];
              ir2[10] = (v922_data + (v858_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 12))));
              float v927_data = r0[13];
              float v931_data = ir2[0];
              ir2[0] = (v931_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 13))));
              float v937_data = ir2[1];
              ir2[1] = (v937_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 13))));
              float v943_data = ir2[2];
              ir2[2] = (v943_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 13))));
              float v949_data = ir2[3];
              ir2[3] = (v949_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 13))));
              float v955_data = ir2[4];
              ir2[4] = (v955_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 13))));
              float v961_data = ir2[5];
              ir2[5] = (v961_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 13))));
              float v967_data = ir2[6];
              ir2[6] = (v967_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 13))));
              float v973_data = ir2[7];
              ir2[7] = (v973_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 13))));
              float v979_data = ir2[8];
              ir2[8] = (v979_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 13))));
              float v985_data = ir2[9];
              ir2[9] = (v985_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 13))));
              float v991_data = ir2[10];
              ir2[10] = (v991_data + (v927_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 13))));
              float v996_data = r0[14];
              float v1000_data = ir2[0];
              ir2[0] = (v1000_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 14))));
              float v1006_data = ir2[1];
              ir2[1] = (v1006_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 14))));
              float v1012_data = ir2[2];
              ir2[2] = (v1012_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 14))));
              float v1018_data = ir2[3];
              ir2[3] = (v1018_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 14))));
              float v1024_data = ir2[4];
              ir2[4] = (v1024_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 14))));
              float v1030_data = ir2[5];
              ir2[5] = (v1030_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 14))));
              float v1036_data = ir2[6];
              ir2[6] = (v1036_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 14))));
              float v1042_data = ir2[7];
              ir2[7] = (v1042_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 14))));
              float v1048_data = ir2[8];
              ir2[8] = (v1048_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 14))));
              float v1054_data = ir2[9];
              ir2[9] = (v1054_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 14))));
              float v1060_data = ir2[10];
              ir2[10] = (v1060_data + (v996_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 14))));
              float v1065_data = r0[15];
              float v1069_data = ir2[0];
              ir2[0] = (v1069_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v31_data, 15))));
              float v1075_data = ir2[1];
              ir2[1] = (v1075_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 15))));
              float v1081_data = ir2[2];
              ir2[2] = (v1081_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 15))));
              float v1087_data = ir2[3];
              ir2[3] = (v1087_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 15))));
              float v1093_data = ir2[4];
              ir2[4] = (v1093_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 15))));
              float v1099_data = ir2[5];
              ir2[5] = (v1099_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 15))));
              float v1105_data = ir2[6];
              ir2[6] = (v1105_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 15))));
              float v1111_data = ir2[7];
              ir2[7] = (v1111_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 15))));
              float v1117_data = ir2[8];
              ir2[8] = (v1117_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 15))));
              float v1123_data = ir2[9];
              ir2[9] = (v1123_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 15))));
              float v1129_data = ir2[10];
              ir2[10] = (v1129_data + (v1065_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 15))));
              #pragma unroll
              for (int32_t v1134_n0 = 0; v1134_n0 < 1; ++v1134_n0) {
                #pragma unroll
                for (int32_t v1135_n1 = 0; v1135_n1 < 11; ++v1135_n1) {
                  int32_t v1136_a = v1134_n0 + v1135_n1;
                  float v1137_data = ir2[v1136_a];
                  r2[v1136_a] = v1137_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1142_i0 = 0; v1142_i0 < 1; ++v1142_i0) {
                int32_t v1150_lead = v8_lead + (v1142_i0 * 16);
                #pragma unroll
                for (int32_t v1143_i1 = 0; v1143_i1 < 11; ++v1143_i1) {
                  float v1145_data = r2[(v1142_i0 + v1143_i1)];
                  glb_m0[(v1150_lead + (v1143_i1 * 16))] = v1145_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


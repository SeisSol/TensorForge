// === base name ===
kernel_87f2838a59

// === header ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_87f2838a59(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_87f2838a59(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 32×32(32×32) {0..32}×{0..32} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v16_off = (v8_lead + (v9_i0 * 16)) + 8;
                #pragma unroll
                for (int32_t v10_i1 = 8; v10_i1 < 24; ++v10_i1) {
                  float v19_data = glb_m1[(v16_off + (v10_i1 * 32))];
                  r0[(v9_i0 + (v10_i1 - 8))] = v19_data;
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
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v42_data = r0[0];
              float v43_data = r1[0];
              float v46_data = ir2[0];
              ir2[0] = (v46_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 0))));
              float v49_data = r1[1];
              float v52_data = ir2[1];
              ir2[1] = (v52_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 0))));
              float v55_data = r1[2];
              float v58_data = ir2[2];
              ir2[2] = (v58_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 0))));
              float v61_data = r1[3];
              float v64_data = ir2[3];
              ir2[3] = (v64_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
              float v67_data = r1[4];
              float v70_data = ir2[4];
              ir2[4] = (v70_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
              float v73_data = r1[5];
              float v76_data = ir2[5];
              ir2[5] = (v76_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
              float v79_data = r1[6];
              float v82_data = ir2[6];
              ir2[6] = (v82_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
              float v85_data = r1[7];
              float v88_data = ir2[7];
              ir2[7] = (v88_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
              float v93_data = r0[1];
              float v97_data = ir2[0];
              ir2[0] = (v97_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 1))));
              float v103_data = ir2[1];
              ir2[1] = (v103_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 1))));
              float v109_data = ir2[2];
              ir2[2] = (v109_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 1))));
              float v115_data = ir2[3];
              ir2[3] = (v115_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 1))));
              float v121_data = ir2[4];
              ir2[4] = (v121_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 1))));
              float v127_data = ir2[5];
              ir2[5] = (v127_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 1))));
              float v133_data = ir2[6];
              ir2[6] = (v133_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 1))));
              float v139_data = ir2[7];
              ir2[7] = (v139_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 1))));
              float v144_data = r0[2];
              float v148_data = ir2[0];
              ir2[0] = (v148_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 2))));
              float v154_data = ir2[1];
              ir2[1] = (v154_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 2))));
              float v160_data = ir2[2];
              ir2[2] = (v160_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 2))));
              float v166_data = ir2[3];
              ir2[3] = (v166_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 2))));
              float v172_data = ir2[4];
              ir2[4] = (v172_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 2))));
              float v178_data = ir2[5];
              ir2[5] = (v178_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 2))));
              float v184_data = ir2[6];
              ir2[6] = (v184_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 2))));
              float v190_data = ir2[7];
              ir2[7] = (v190_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 2))));
              float v195_data = r0[3];
              float v199_data = ir2[0];
              ir2[0] = (v199_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 3))));
              float v205_data = ir2[1];
              ir2[1] = (v205_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 3))));
              float v211_data = ir2[2];
              ir2[2] = (v211_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 3))));
              float v217_data = ir2[3];
              ir2[3] = (v217_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 3))));
              float v223_data = ir2[4];
              ir2[4] = (v223_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 3))));
              float v229_data = ir2[5];
              ir2[5] = (v229_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 3))));
              float v235_data = ir2[6];
              ir2[6] = (v235_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 3))));
              float v241_data = ir2[7];
              ir2[7] = (v241_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 3))));
              float v246_data = r0[4];
              float v250_data = ir2[0];
              ir2[0] = (v250_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 4))));
              float v256_data = ir2[1];
              ir2[1] = (v256_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 4))));
              float v262_data = ir2[2];
              ir2[2] = (v262_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 4))));
              float v268_data = ir2[3];
              ir2[3] = (v268_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 4))));
              float v274_data = ir2[4];
              ir2[4] = (v274_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 4))));
              float v280_data = ir2[5];
              ir2[5] = (v280_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 4))));
              float v286_data = ir2[6];
              ir2[6] = (v286_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 4))));
              float v292_data = ir2[7];
              ir2[7] = (v292_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 4))));
              float v297_data = r0[5];
              float v301_data = ir2[0];
              ir2[0] = (v301_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 5))));
              float v307_data = ir2[1];
              ir2[1] = (v307_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 5))));
              float v313_data = ir2[2];
              ir2[2] = (v313_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 5))));
              float v319_data = ir2[3];
              ir2[3] = (v319_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 5))));
              float v325_data = ir2[4];
              ir2[4] = (v325_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 5))));
              float v331_data = ir2[5];
              ir2[5] = (v331_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 5))));
              float v337_data = ir2[6];
              ir2[6] = (v337_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 5))));
              float v343_data = ir2[7];
              ir2[7] = (v343_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 5))));
              float v348_data = r0[6];
              float v352_data = ir2[0];
              ir2[0] = (v352_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 6))));
              float v358_data = ir2[1];
              ir2[1] = (v358_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 6))));
              float v364_data = ir2[2];
              ir2[2] = (v364_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 6))));
              float v370_data = ir2[3];
              ir2[3] = (v370_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 6))));
              float v376_data = ir2[4];
              ir2[4] = (v376_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 6))));
              float v382_data = ir2[5];
              ir2[5] = (v382_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 6))));
              float v388_data = ir2[6];
              ir2[6] = (v388_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 6))));
              float v394_data = ir2[7];
              ir2[7] = (v394_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 6))));
              float v399_data = r0[7];
              float v403_data = ir2[0];
              ir2[0] = (v403_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 7))));
              float v409_data = ir2[1];
              ir2[1] = (v409_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 7))));
              float v415_data = ir2[2];
              ir2[2] = (v415_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 7))));
              float v421_data = ir2[3];
              ir2[3] = (v421_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 7))));
              float v427_data = ir2[4];
              ir2[4] = (v427_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 7))));
              float v433_data = ir2[5];
              ir2[5] = (v433_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 7))));
              float v439_data = ir2[6];
              ir2[6] = (v439_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 7))));
              float v445_data = ir2[7];
              ir2[7] = (v445_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 7))));
              float v450_data = r0[8];
              float v454_data = ir2[0];
              ir2[0] = (v454_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 8))));
              float v460_data = ir2[1];
              ir2[1] = (v460_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 8))));
              float v466_data = ir2[2];
              ir2[2] = (v466_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 8))));
              float v472_data = ir2[3];
              ir2[3] = (v472_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 8))));
              float v478_data = ir2[4];
              ir2[4] = (v478_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 8))));
              float v484_data = ir2[5];
              ir2[5] = (v484_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 8))));
              float v490_data = ir2[6];
              ir2[6] = (v490_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 8))));
              float v496_data = ir2[7];
              ir2[7] = (v496_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 8))));
              float v501_data = r0[9];
              float v505_data = ir2[0];
              ir2[0] = (v505_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 9))));
              float v511_data = ir2[1];
              ir2[1] = (v511_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 9))));
              float v517_data = ir2[2];
              ir2[2] = (v517_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 9))));
              float v523_data = ir2[3];
              ir2[3] = (v523_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 9))));
              float v529_data = ir2[4];
              ir2[4] = (v529_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 9))));
              float v535_data = ir2[5];
              ir2[5] = (v535_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 9))));
              float v541_data = ir2[6];
              ir2[6] = (v541_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 9))));
              float v547_data = ir2[7];
              ir2[7] = (v547_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 9))));
              float v552_data = r0[10];
              float v556_data = ir2[0];
              ir2[0] = (v556_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 10))));
              float v562_data = ir2[1];
              ir2[1] = (v562_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 10))));
              float v568_data = ir2[2];
              ir2[2] = (v568_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 10))));
              float v574_data = ir2[3];
              ir2[3] = (v574_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 10))));
              float v580_data = ir2[4];
              ir2[4] = (v580_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 10))));
              float v586_data = ir2[5];
              ir2[5] = (v586_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 10))));
              float v592_data = ir2[6];
              ir2[6] = (v592_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 10))));
              float v598_data = ir2[7];
              ir2[7] = (v598_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 10))));
              float v603_data = r0[11];
              float v607_data = ir2[0];
              ir2[0] = (v607_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 11))));
              float v613_data = ir2[1];
              ir2[1] = (v613_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 11))));
              float v619_data = ir2[2];
              ir2[2] = (v619_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 11))));
              float v625_data = ir2[3];
              ir2[3] = (v625_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 11))));
              float v631_data = ir2[4];
              ir2[4] = (v631_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 11))));
              float v637_data = ir2[5];
              ir2[5] = (v637_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 11))));
              float v643_data = ir2[6];
              ir2[6] = (v643_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 11))));
              float v649_data = ir2[7];
              ir2[7] = (v649_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 11))));
              float v654_data = r0[12];
              float v658_data = ir2[0];
              ir2[0] = (v658_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 12))));
              float v664_data = ir2[1];
              ir2[1] = (v664_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 12))));
              float v670_data = ir2[2];
              ir2[2] = (v670_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 12))));
              float v676_data = ir2[3];
              ir2[3] = (v676_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 12))));
              float v682_data = ir2[4];
              ir2[4] = (v682_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 12))));
              float v688_data = ir2[5];
              ir2[5] = (v688_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 12))));
              float v694_data = ir2[6];
              ir2[6] = (v694_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 12))));
              float v700_data = ir2[7];
              ir2[7] = (v700_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 12))));
              float v705_data = r0[13];
              float v709_data = ir2[0];
              ir2[0] = (v709_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 13))));
              float v715_data = ir2[1];
              ir2[1] = (v715_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 13))));
              float v721_data = ir2[2];
              ir2[2] = (v721_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 13))));
              float v727_data = ir2[3];
              ir2[3] = (v727_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 13))));
              float v733_data = ir2[4];
              ir2[4] = (v733_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 13))));
              float v739_data = ir2[5];
              ir2[5] = (v739_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 13))));
              float v745_data = ir2[6];
              ir2[6] = (v745_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 13))));
              float v751_data = ir2[7];
              ir2[7] = (v751_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 13))));
              float v756_data = r0[14];
              float v760_data = ir2[0];
              ir2[0] = (v760_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 14))));
              float v766_data = ir2[1];
              ir2[1] = (v766_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 14))));
              float v772_data = ir2[2];
              ir2[2] = (v772_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 14))));
              float v778_data = ir2[3];
              ir2[3] = (v778_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 14))));
              float v784_data = ir2[4];
              ir2[4] = (v784_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 14))));
              float v790_data = ir2[5];
              ir2[5] = (v790_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 14))));
              float v796_data = ir2[6];
              ir2[6] = (v796_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 14))));
              float v802_data = ir2[7];
              ir2[7] = (v802_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 14))));
              float v807_data = r0[15];
              float v811_data = ir2[0];
              ir2[0] = (v811_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 15))));
              float v817_data = ir2[1];
              ir2[1] = (v817_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 15))));
              float v823_data = ir2[2];
              ir2[2] = (v823_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 15))));
              float v829_data = ir2[3];
              ir2[3] = (v829_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 15))));
              float v835_data = ir2[4];
              ir2[4] = (v835_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 15))));
              float v841_data = ir2[5];
              ir2[5] = (v841_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 15))));
              float v847_data = ir2[6];
              ir2[6] = (v847_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 15))));
              float v853_data = ir2[7];
              ir2[7] = (v853_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 15))));
              #pragma unroll
              for (int32_t v858_n0 = 0; v858_n0 < 1; ++v858_n0) {
                #pragma unroll
                for (int32_t v859_n1 = 0; v859_n1 < 8; ++v859_n1) {
                  int32_t v860_a = v858_n0 + v859_n1;
                  float v861_data = ir2[v860_a];
                  r2[v860_a] = v861_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v866_i0 = 0; v866_i0 < 1; ++v866_i0) {
                int32_t v874_lead = v8_lead + (v866_i0 * 16);
                #pragma unroll
                for (int32_t v867_i1 = 0; v867_i1 < 8; ++v867_i1) {
                  float v869_data = r2[(v866_i0 + v867_i1)];
                  glb_m0[(v874_lead + (v867_i1 * 16))] = v869_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


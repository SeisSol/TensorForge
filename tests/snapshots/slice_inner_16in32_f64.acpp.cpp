// === base name ===
kernel_3d37ccf0b0

// === header ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3d37ccf0b0(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3d37ccf0b0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
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
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v16_off = (v8_lead + (v9_i0 * 16)) + 8;
                #pragma unroll
                for (int32_t v10_i1 = 8; v10_i1 < 24; ++v10_i1) {
                  double v19_data = glb_m1[(v16_off + (v10_i1 * 32))];
                  r0[(v9_i0 + (v10_i1 - 8))] = v19_data;
                }
              }
              double r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
                int32_t v32_lead = v8_lead + (v26_i0 * 16);
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
                  double v35_data = glb_m2[(v32_lead + (v27_i1 * 16))];
                  r1[(v26_i0 + v27_i1)] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              double r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              double ir2[8]{};
              double v42_data = r0[0];
              double v43_data = r1[0];
              double v46_data = ir2[0];
              ir2[0] = (v46_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 0))));
              double v49_data = r1[1];
              double v52_data = ir2[1];
              ir2[1] = (v52_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 0))));
              double v55_data = r1[2];
              double v58_data = ir2[2];
              ir2[2] = (v58_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 0))));
              double v61_data = r1[3];
              double v64_data = ir2[3];
              ir2[3] = (v64_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
              double v67_data = r1[4];
              double v70_data = ir2[4];
              ir2[4] = (v70_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
              double v73_data = r1[5];
              double v76_data = ir2[5];
              ir2[5] = (v76_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
              double v79_data = r1[6];
              double v82_data = ir2[6];
              ir2[6] = (v82_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
              double v85_data = r1[7];
              double v88_data = ir2[7];
              ir2[7] = (v88_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
              double v93_data = r0[1];
              double v97_data = ir2[0];
              ir2[0] = (v97_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 1))));
              double v103_data = ir2[1];
              ir2[1] = (v103_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 1))));
              double v109_data = ir2[2];
              ir2[2] = (v109_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 1))));
              double v115_data = ir2[3];
              ir2[3] = (v115_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 1))));
              double v121_data = ir2[4];
              ir2[4] = (v121_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 1))));
              double v127_data = ir2[5];
              ir2[5] = (v127_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 1))));
              double v133_data = ir2[6];
              ir2[6] = (v133_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 1))));
              double v139_data = ir2[7];
              ir2[7] = (v139_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 1))));
              double v144_data = r0[2];
              double v148_data = ir2[0];
              ir2[0] = (v148_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 2))));
              double v154_data = ir2[1];
              ir2[1] = (v154_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 2))));
              double v160_data = ir2[2];
              ir2[2] = (v160_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 2))));
              double v166_data = ir2[3];
              ir2[3] = (v166_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 2))));
              double v172_data = ir2[4];
              ir2[4] = (v172_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 2))));
              double v178_data = ir2[5];
              ir2[5] = (v178_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 2))));
              double v184_data = ir2[6];
              ir2[6] = (v184_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 2))));
              double v190_data = ir2[7];
              ir2[7] = (v190_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 2))));
              double v195_data = r0[3];
              double v199_data = ir2[0];
              ir2[0] = (v199_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 3))));
              double v205_data = ir2[1];
              ir2[1] = (v205_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 3))));
              double v211_data = ir2[2];
              ir2[2] = (v211_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 3))));
              double v217_data = ir2[3];
              ir2[3] = (v217_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 3))));
              double v223_data = ir2[4];
              ir2[4] = (v223_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 3))));
              double v229_data = ir2[5];
              ir2[5] = (v229_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 3))));
              double v235_data = ir2[6];
              ir2[6] = (v235_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 3))));
              double v241_data = ir2[7];
              ir2[7] = (v241_data + (v195_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 3))));
              double v246_data = r0[4];
              double v250_data = ir2[0];
              ir2[0] = (v250_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 4))));
              double v256_data = ir2[1];
              ir2[1] = (v256_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 4))));
              double v262_data = ir2[2];
              ir2[2] = (v262_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 4))));
              double v268_data = ir2[3];
              ir2[3] = (v268_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 4))));
              double v274_data = ir2[4];
              ir2[4] = (v274_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 4))));
              double v280_data = ir2[5];
              ir2[5] = (v280_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 4))));
              double v286_data = ir2[6];
              ir2[6] = (v286_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 4))));
              double v292_data = ir2[7];
              ir2[7] = (v292_data + (v246_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 4))));
              double v297_data = r0[5];
              double v301_data = ir2[0];
              ir2[0] = (v301_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 5))));
              double v307_data = ir2[1];
              ir2[1] = (v307_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 5))));
              double v313_data = ir2[2];
              ir2[2] = (v313_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 5))));
              double v319_data = ir2[3];
              ir2[3] = (v319_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 5))));
              double v325_data = ir2[4];
              ir2[4] = (v325_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 5))));
              double v331_data = ir2[5];
              ir2[5] = (v331_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 5))));
              double v337_data = ir2[6];
              ir2[6] = (v337_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 5))));
              double v343_data = ir2[7];
              ir2[7] = (v343_data + (v297_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 5))));
              double v348_data = r0[6];
              double v352_data = ir2[0];
              ir2[0] = (v352_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 6))));
              double v358_data = ir2[1];
              ir2[1] = (v358_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 6))));
              double v364_data = ir2[2];
              ir2[2] = (v364_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 6))));
              double v370_data = ir2[3];
              ir2[3] = (v370_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 6))));
              double v376_data = ir2[4];
              ir2[4] = (v376_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 6))));
              double v382_data = ir2[5];
              ir2[5] = (v382_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 6))));
              double v388_data = ir2[6];
              ir2[6] = (v388_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 6))));
              double v394_data = ir2[7];
              ir2[7] = (v394_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 6))));
              double v399_data = r0[7];
              double v403_data = ir2[0];
              ir2[0] = (v403_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 7))));
              double v409_data = ir2[1];
              ir2[1] = (v409_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 7))));
              double v415_data = ir2[2];
              ir2[2] = (v415_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 7))));
              double v421_data = ir2[3];
              ir2[3] = (v421_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 7))));
              double v427_data = ir2[4];
              ir2[4] = (v427_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 7))));
              double v433_data = ir2[5];
              ir2[5] = (v433_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 7))));
              double v439_data = ir2[6];
              ir2[6] = (v439_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 7))));
              double v445_data = ir2[7];
              ir2[7] = (v445_data + (v399_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 7))));
              double v450_data = r0[8];
              double v454_data = ir2[0];
              ir2[0] = (v454_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 8))));
              double v460_data = ir2[1];
              ir2[1] = (v460_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 8))));
              double v466_data = ir2[2];
              ir2[2] = (v466_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 8))));
              double v472_data = ir2[3];
              ir2[3] = (v472_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 8))));
              double v478_data = ir2[4];
              ir2[4] = (v478_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 8))));
              double v484_data = ir2[5];
              ir2[5] = (v484_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 8))));
              double v490_data = ir2[6];
              ir2[6] = (v490_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 8))));
              double v496_data = ir2[7];
              ir2[7] = (v496_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 8))));
              double v501_data = r0[9];
              double v505_data = ir2[0];
              ir2[0] = (v505_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 9))));
              double v511_data = ir2[1];
              ir2[1] = (v511_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 9))));
              double v517_data = ir2[2];
              ir2[2] = (v517_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 9))));
              double v523_data = ir2[3];
              ir2[3] = (v523_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 9))));
              double v529_data = ir2[4];
              ir2[4] = (v529_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 9))));
              double v535_data = ir2[5];
              ir2[5] = (v535_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 9))));
              double v541_data = ir2[6];
              ir2[6] = (v541_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 9))));
              double v547_data = ir2[7];
              ir2[7] = (v547_data + (v501_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 9))));
              double v552_data = r0[10];
              double v556_data = ir2[0];
              ir2[0] = (v556_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 10))));
              double v562_data = ir2[1];
              ir2[1] = (v562_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 10))));
              double v568_data = ir2[2];
              ir2[2] = (v568_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 10))));
              double v574_data = ir2[3];
              ir2[3] = (v574_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 10))));
              double v580_data = ir2[4];
              ir2[4] = (v580_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 10))));
              double v586_data = ir2[5];
              ir2[5] = (v586_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 10))));
              double v592_data = ir2[6];
              ir2[6] = (v592_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 10))));
              double v598_data = ir2[7];
              ir2[7] = (v598_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 10))));
              double v603_data = r0[11];
              double v607_data = ir2[0];
              ir2[0] = (v607_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 11))));
              double v613_data = ir2[1];
              ir2[1] = (v613_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 11))));
              double v619_data = ir2[2];
              ir2[2] = (v619_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 11))));
              double v625_data = ir2[3];
              ir2[3] = (v625_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 11))));
              double v631_data = ir2[4];
              ir2[4] = (v631_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 11))));
              double v637_data = ir2[5];
              ir2[5] = (v637_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 11))));
              double v643_data = ir2[6];
              ir2[6] = (v643_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 11))));
              double v649_data = ir2[7];
              ir2[7] = (v649_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 11))));
              double v654_data = r0[12];
              double v658_data = ir2[0];
              ir2[0] = (v658_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 12))));
              double v664_data = ir2[1];
              ir2[1] = (v664_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 12))));
              double v670_data = ir2[2];
              ir2[2] = (v670_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 12))));
              double v676_data = ir2[3];
              ir2[3] = (v676_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 12))));
              double v682_data = ir2[4];
              ir2[4] = (v682_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 12))));
              double v688_data = ir2[5];
              ir2[5] = (v688_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 12))));
              double v694_data = ir2[6];
              ir2[6] = (v694_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 12))));
              double v700_data = ir2[7];
              ir2[7] = (v700_data + (v654_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 12))));
              double v705_data = r0[13];
              double v709_data = ir2[0];
              ir2[0] = (v709_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 13))));
              double v715_data = ir2[1];
              ir2[1] = (v715_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 13))));
              double v721_data = ir2[2];
              ir2[2] = (v721_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 13))));
              double v727_data = ir2[3];
              ir2[3] = (v727_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 13))));
              double v733_data = ir2[4];
              ir2[4] = (v733_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 13))));
              double v739_data = ir2[5];
              ir2[5] = (v739_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 13))));
              double v745_data = ir2[6];
              ir2[6] = (v745_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 13))));
              double v751_data = ir2[7];
              ir2[7] = (v751_data + (v705_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 13))));
              double v756_data = r0[14];
              double v760_data = ir2[0];
              ir2[0] = (v760_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 14))));
              double v766_data = ir2[1];
              ir2[1] = (v766_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 14))));
              double v772_data = ir2[2];
              ir2[2] = (v772_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 14))));
              double v778_data = ir2[3];
              ir2[3] = (v778_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 14))));
              double v784_data = ir2[4];
              ir2[4] = (v784_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 14))));
              double v790_data = ir2[5];
              ir2[5] = (v790_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 14))));
              double v796_data = ir2[6];
              ir2[6] = (v796_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 14))));
              double v802_data = ir2[7];
              ir2[7] = (v802_data + (v756_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 14))));
              double v807_data = r0[15];
              double v811_data = ir2[0];
              ir2[0] = (v811_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 15))));
              double v817_data = ir2[1];
              ir2[1] = (v817_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 15))));
              double v823_data = ir2[2];
              ir2[2] = (v823_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 15))));
              double v829_data = ir2[3];
              ir2[3] = (v829_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 15))));
              double v835_data = ir2[4];
              ir2[4] = (v835_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 15))));
              double v841_data = ir2[5];
              ir2[5] = (v841_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 15))));
              double v847_data = ir2[6];
              ir2[6] = (v847_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 15))));
              double v853_data = ir2[7];
              ir2[7] = (v853_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 15))));
              #pragma unroll
              for (int32_t v858_n0 = 0; v858_n0 < 1; ++v858_n0) {
                #pragma unroll
                for (int32_t v859_n1 = 0; v859_n1 < 8; ++v859_n1) {
                  int32_t v860_a = v858_n0 + v859_n1;
                  double v861_data = ir2[v860_a];
                  r2[v860_a] = v861_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v866_i0 = 0; v866_i0 < 1; ++v866_i0) {
                int32_t v874_lead = v8_lead + (v866_i0 * 16);
                #pragma unroll
                for (int32_t v867_i1 = 0; v867_i1 < 8; ++v867_i1) {
                  double v869_data = r2[(v866_i0 + v867_i1)];
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


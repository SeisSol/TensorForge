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
              double v23_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v23_lin;
              double v24_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v24_lin;
              double v25_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v25_lin;
              double v26_lin = glb_m2[48 + item.get_local_id(0) * 1];
              r1[3] = v26_lin;
              double v27_lin = glb_m2[64 + item.get_local_id(0) * 1];
              r1[4] = v27_lin;
              double v28_lin = glb_m2[80 + item.get_local_id(0) * 1];
              r1[5] = v28_lin;
              double v29_lin = glb_m2[96 + item.get_local_id(0) * 1];
              r1[6] = v29_lin;
              double v30_lin = glb_m2[112 + item.get_local_id(0) * 1];
              r1[7] = v30_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              double r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              double ir2[8]{};
              double v36_data = r0[0];
              double v37_data = r1[0];
              double v40_data = ir2[0];
              ir2[0] = (v40_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 0))));
              double v43_data = r1[1];
              double v46_data = ir2[1];
              ir2[1] = (v46_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 0))));
              double v49_data = r1[2];
              double v52_data = ir2[2];
              ir2[2] = (v52_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 0))));
              double v55_data = r1[3];
              double v58_data = ir2[3];
              ir2[3] = (v58_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 0))));
              double v61_data = r1[4];
              double v64_data = ir2[4];
              ir2[4] = (v64_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
              double v67_data = r1[5];
              double v70_data = ir2[5];
              ir2[5] = (v70_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
              double v73_data = r1[6];
              double v76_data = ir2[6];
              ir2[6] = (v76_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
              double v79_data = r1[7];
              double v82_data = ir2[7];
              ir2[7] = (v82_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
              double v87_data = r0[1];
              double v91_data = ir2[0];
              ir2[0] = (v91_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 1))));
              double v97_data = ir2[1];
              ir2[1] = (v97_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 1))));
              double v103_data = ir2[2];
              ir2[2] = (v103_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 1))));
              double v109_data = ir2[3];
              ir2[3] = (v109_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 1))));
              double v115_data = ir2[4];
              ir2[4] = (v115_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 1))));
              double v121_data = ir2[5];
              ir2[5] = (v121_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 1))));
              double v127_data = ir2[6];
              ir2[6] = (v127_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 1))));
              double v133_data = ir2[7];
              ir2[7] = (v133_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 1))));
              double v138_data = r0[2];
              double v142_data = ir2[0];
              ir2[0] = (v142_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 2))));
              double v148_data = ir2[1];
              ir2[1] = (v148_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 2))));
              double v154_data = ir2[2];
              ir2[2] = (v154_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 2))));
              double v160_data = ir2[3];
              ir2[3] = (v160_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 2))));
              double v166_data = ir2[4];
              ir2[4] = (v166_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 2))));
              double v172_data = ir2[5];
              ir2[5] = (v172_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 2))));
              double v178_data = ir2[6];
              ir2[6] = (v178_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 2))));
              double v184_data = ir2[7];
              ir2[7] = (v184_data + (v138_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 2))));
              double v189_data = r0[3];
              double v193_data = ir2[0];
              ir2[0] = (v193_data + (v189_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 3))));
              double v199_data = ir2[1];
              ir2[1] = (v199_data + (v189_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 3))));
              double v205_data = ir2[2];
              ir2[2] = (v205_data + (v189_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 3))));
              double v211_data = ir2[3];
              ir2[3] = (v211_data + (v189_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 3))));
              double v217_data = ir2[4];
              ir2[4] = (v217_data + (v189_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 3))));
              double v223_data = ir2[5];
              ir2[5] = (v223_data + (v189_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 3))));
              double v229_data = ir2[6];
              ir2[6] = (v229_data + (v189_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 3))));
              double v235_data = ir2[7];
              ir2[7] = (v235_data + (v189_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 3))));
              double v240_data = r0[4];
              double v244_data = ir2[0];
              ir2[0] = (v244_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 4))));
              double v250_data = ir2[1];
              ir2[1] = (v250_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 4))));
              double v256_data = ir2[2];
              ir2[2] = (v256_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 4))));
              double v262_data = ir2[3];
              ir2[3] = (v262_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 4))));
              double v268_data = ir2[4];
              ir2[4] = (v268_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 4))));
              double v274_data = ir2[5];
              ir2[5] = (v274_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 4))));
              double v280_data = ir2[6];
              ir2[6] = (v280_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 4))));
              double v286_data = ir2[7];
              ir2[7] = (v286_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 4))));
              double v291_data = r0[5];
              double v295_data = ir2[0];
              ir2[0] = (v295_data + (v291_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 5))));
              double v301_data = ir2[1];
              ir2[1] = (v301_data + (v291_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 5))));
              double v307_data = ir2[2];
              ir2[2] = (v307_data + (v291_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 5))));
              double v313_data = ir2[3];
              ir2[3] = (v313_data + (v291_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 5))));
              double v319_data = ir2[4];
              ir2[4] = (v319_data + (v291_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 5))));
              double v325_data = ir2[5];
              ir2[5] = (v325_data + (v291_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 5))));
              double v331_data = ir2[6];
              ir2[6] = (v331_data + (v291_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 5))));
              double v337_data = ir2[7];
              ir2[7] = (v337_data + (v291_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 5))));
              double v342_data = r0[6];
              double v346_data = ir2[0];
              ir2[0] = (v346_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 6))));
              double v352_data = ir2[1];
              ir2[1] = (v352_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 6))));
              double v358_data = ir2[2];
              ir2[2] = (v358_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 6))));
              double v364_data = ir2[3];
              ir2[3] = (v364_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 6))));
              double v370_data = ir2[4];
              ir2[4] = (v370_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 6))));
              double v376_data = ir2[5];
              ir2[5] = (v376_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 6))));
              double v382_data = ir2[6];
              ir2[6] = (v382_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 6))));
              double v388_data = ir2[7];
              ir2[7] = (v388_data + (v342_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 6))));
              double v393_data = r0[7];
              double v397_data = ir2[0];
              ir2[0] = (v397_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 7))));
              double v403_data = ir2[1];
              ir2[1] = (v403_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 7))));
              double v409_data = ir2[2];
              ir2[2] = (v409_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 7))));
              double v415_data = ir2[3];
              ir2[3] = (v415_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 7))));
              double v421_data = ir2[4];
              ir2[4] = (v421_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 7))));
              double v427_data = ir2[5];
              ir2[5] = (v427_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 7))));
              double v433_data = ir2[6];
              ir2[6] = (v433_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 7))));
              double v439_data = ir2[7];
              ir2[7] = (v439_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 7))));
              double v444_data = r0[8];
              double v448_data = ir2[0];
              ir2[0] = (v448_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 8))));
              double v454_data = ir2[1];
              ir2[1] = (v454_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 8))));
              double v460_data = ir2[2];
              ir2[2] = (v460_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 8))));
              double v466_data = ir2[3];
              ir2[3] = (v466_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 8))));
              double v472_data = ir2[4];
              ir2[4] = (v472_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 8))));
              double v478_data = ir2[5];
              ir2[5] = (v478_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 8))));
              double v484_data = ir2[6];
              ir2[6] = (v484_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 8))));
              double v490_data = ir2[7];
              ir2[7] = (v490_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 8))));
              double v495_data = r0[9];
              double v499_data = ir2[0];
              ir2[0] = (v499_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 9))));
              double v505_data = ir2[1];
              ir2[1] = (v505_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 9))));
              double v511_data = ir2[2];
              ir2[2] = (v511_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 9))));
              double v517_data = ir2[3];
              ir2[3] = (v517_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 9))));
              double v523_data = ir2[4];
              ir2[4] = (v523_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 9))));
              double v529_data = ir2[5];
              ir2[5] = (v529_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 9))));
              double v535_data = ir2[6];
              ir2[6] = (v535_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 9))));
              double v541_data = ir2[7];
              ir2[7] = (v541_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 9))));
              double v546_data = r0[10];
              double v550_data = ir2[0];
              ir2[0] = (v550_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 10))));
              double v556_data = ir2[1];
              ir2[1] = (v556_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 10))));
              double v562_data = ir2[2];
              ir2[2] = (v562_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 10))));
              double v568_data = ir2[3];
              ir2[3] = (v568_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 10))));
              double v574_data = ir2[4];
              ir2[4] = (v574_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 10))));
              double v580_data = ir2[5];
              ir2[5] = (v580_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 10))));
              double v586_data = ir2[6];
              ir2[6] = (v586_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 10))));
              double v592_data = ir2[7];
              ir2[7] = (v592_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 10))));
              double v597_data = r0[11];
              double v601_data = ir2[0];
              ir2[0] = (v601_data + (v597_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 11))));
              double v607_data = ir2[1];
              ir2[1] = (v607_data + (v597_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 11))));
              double v613_data = ir2[2];
              ir2[2] = (v613_data + (v597_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 11))));
              double v619_data = ir2[3];
              ir2[3] = (v619_data + (v597_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 11))));
              double v625_data = ir2[4];
              ir2[4] = (v625_data + (v597_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 11))));
              double v631_data = ir2[5];
              ir2[5] = (v631_data + (v597_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 11))));
              double v637_data = ir2[6];
              ir2[6] = (v637_data + (v597_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 11))));
              double v643_data = ir2[7];
              ir2[7] = (v643_data + (v597_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 11))));
              double v648_data = r0[12];
              double v652_data = ir2[0];
              ir2[0] = (v652_data + (v648_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 12))));
              double v658_data = ir2[1];
              ir2[1] = (v658_data + (v648_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 12))));
              double v664_data = ir2[2];
              ir2[2] = (v664_data + (v648_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 12))));
              double v670_data = ir2[3];
              ir2[3] = (v670_data + (v648_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 12))));
              double v676_data = ir2[4];
              ir2[4] = (v676_data + (v648_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 12))));
              double v682_data = ir2[5];
              ir2[5] = (v682_data + (v648_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 12))));
              double v688_data = ir2[6];
              ir2[6] = (v688_data + (v648_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 12))));
              double v694_data = ir2[7];
              ir2[7] = (v694_data + (v648_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 12))));
              double v699_data = r0[13];
              double v703_data = ir2[0];
              ir2[0] = (v703_data + (v699_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 13))));
              double v709_data = ir2[1];
              ir2[1] = (v709_data + (v699_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 13))));
              double v715_data = ir2[2];
              ir2[2] = (v715_data + (v699_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 13))));
              double v721_data = ir2[3];
              ir2[3] = (v721_data + (v699_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 13))));
              double v727_data = ir2[4];
              ir2[4] = (v727_data + (v699_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 13))));
              double v733_data = ir2[5];
              ir2[5] = (v733_data + (v699_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 13))));
              double v739_data = ir2[6];
              ir2[6] = (v739_data + (v699_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 13))));
              double v745_data = ir2[7];
              ir2[7] = (v745_data + (v699_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 13))));
              double v750_data = r0[14];
              double v754_data = ir2[0];
              ir2[0] = (v754_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 14))));
              double v760_data = ir2[1];
              ir2[1] = (v760_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 14))));
              double v766_data = ir2[2];
              ir2[2] = (v766_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 14))));
              double v772_data = ir2[3];
              ir2[3] = (v772_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 14))));
              double v778_data = ir2[4];
              ir2[4] = (v778_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 14))));
              double v784_data = ir2[5];
              ir2[5] = (v784_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 14))));
              double v790_data = ir2[6];
              ir2[6] = (v790_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 14))));
              double v796_data = ir2[7];
              ir2[7] = (v796_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 14))));
              double v801_data = r0[15];
              double v805_data = ir2[0];
              ir2[0] = (v805_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 15))));
              double v811_data = ir2[1];
              ir2[1] = (v811_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 15))));
              double v817_data = ir2[2];
              ir2[2] = (v817_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 15))));
              double v823_data = ir2[3];
              ir2[3] = (v823_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 15))));
              double v829_data = ir2[4];
              ir2[4] = (v829_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 15))));
              double v835_data = ir2[5];
              ir2[5] = (v835_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 15))));
              double v841_data = ir2[6];
              ir2[6] = (v841_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 15))));
              double v847_data = ir2[7];
              ir2[7] = (v847_data + (v801_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 15))));
              #pragma unroll
              for (int32_t v852_n0 = 0; v852_n0 < 1; ++v852_n0) {
                #pragma unroll
                for (int32_t v853_n1 = 0; v853_n1 < 8; ++v853_n1) {
                  int32_t v854_a = v852_n0 + v853_n1;
                  double v855_data = ir2[v854_a];
                  r2[v854_a] = v855_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v860_i0 = 0; v860_i0 < 1; ++v860_i0) {
                int32_t v868_lead = v8_lead + (v860_i0 * 16);
                #pragma unroll
                for (int32_t v861_i1 = 0; v861_i1 < 8; ++v861_i1) {
                  double v863_data = r2[(v860_i0 + v861_i1)];
                  glb_m0[(v868_lead + (v861_i1 * 16))] = v863_data;
                }
              }
            }
          }
        }
      });
    }
  });
}


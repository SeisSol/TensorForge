// === base name ===
kernel_6476589cda2e1065

// === header ===
void launcher_kernel_6476589cda2e1065(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_6476589cda2e1065(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_6476589cda2e1065(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_6476589cda2e1065(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v23_lead = v16_lead + (v17_i0 * 16);
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
                  float v26_data = glb_m1[(v23_lead + (v18_i1 * 16))];
                  r0[(v17_i0 + v18_i1)] = v26_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v38_lead = v16_lead + (v32_i0 * 16);
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
                  float v41_data = glb_m2[(v38_lead + (v33_i1 * 16))];
                  r1[(v32_i0 + v33_i1)] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v48_data = r0[0];
              float v49_data = r1[0];
              float v52_data = ir2[0];
              ir2[0] = (v52_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 0))));
              float v55_data = r1[1];
              float v58_data = ir2[1];
              ir2[1] = (v58_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 0))));
              float v61_data = r1[2];
              float v64_data = ir2[2];
              ir2[2] = (v64_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
              float v67_data = r1[3];
              float v70_data = ir2[3];
              ir2[3] = (v70_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
              float v73_data = r1[4];
              float v76_data = ir2[4];
              ir2[4] = (v76_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
              float v79_data = r1[5];
              float v82_data = ir2[5];
              ir2[5] = (v82_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
              float v85_data = r1[6];
              float v88_data = ir2[6];
              ir2[6] = (v88_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
              float v91_data = r1[7];
              float v94_data = ir2[7];
              ir2[7] = (v94_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 0))));
              float v99_data = r0[1];
              float v103_data = ir2[0];
              ir2[0] = (v103_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 1))));
              float v109_data = ir2[1];
              ir2[1] = (v109_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 1))));
              float v115_data = ir2[2];
              ir2[2] = (v115_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 1))));
              float v121_data = ir2[3];
              ir2[3] = (v121_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 1))));
              float v127_data = ir2[4];
              ir2[4] = (v127_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 1))));
              float v133_data = ir2[5];
              ir2[5] = (v133_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 1))));
              float v139_data = ir2[6];
              ir2[6] = (v139_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 1))));
              float v145_data = ir2[7];
              ir2[7] = (v145_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 1))));
              float v150_data = r0[2];
              float v154_data = ir2[0];
              ir2[0] = (v154_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 2))));
              float v160_data = ir2[1];
              ir2[1] = (v160_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 2))));
              float v166_data = ir2[2];
              ir2[2] = (v166_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 2))));
              float v172_data = ir2[3];
              ir2[3] = (v172_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 2))));
              float v178_data = ir2[4];
              ir2[4] = (v178_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 2))));
              float v184_data = ir2[5];
              ir2[5] = (v184_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 2))));
              float v190_data = ir2[6];
              ir2[6] = (v190_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 2))));
              float v196_data = ir2[7];
              ir2[7] = (v196_data + (v150_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 2))));
              float v201_data = r0[3];
              float v205_data = ir2[0];
              ir2[0] = (v205_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 3))));
              float v211_data = ir2[1];
              ir2[1] = (v211_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 3))));
              float v217_data = ir2[2];
              ir2[2] = (v217_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 3))));
              float v223_data = ir2[3];
              ir2[3] = (v223_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 3))));
              float v229_data = ir2[4];
              ir2[4] = (v229_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 3))));
              float v235_data = ir2[5];
              ir2[5] = (v235_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 3))));
              float v241_data = ir2[6];
              ir2[6] = (v241_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 3))));
              float v247_data = ir2[7];
              ir2[7] = (v247_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 3))));
              float v252_data = r0[4];
              float v256_data = ir2[0];
              ir2[0] = (v256_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 4))));
              float v262_data = ir2[1];
              ir2[1] = (v262_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 4))));
              float v268_data = ir2[2];
              ir2[2] = (v268_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 4))));
              float v274_data = ir2[3];
              ir2[3] = (v274_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 4))));
              float v280_data = ir2[4];
              ir2[4] = (v280_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 4))));
              float v286_data = ir2[5];
              ir2[5] = (v286_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 4))));
              float v292_data = ir2[6];
              ir2[6] = (v292_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 4))));
              float v298_data = ir2[7];
              ir2[7] = (v298_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 4))));
              float v303_data = r0[5];
              float v307_data = ir2[0];
              ir2[0] = (v307_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 5))));
              float v313_data = ir2[1];
              ir2[1] = (v313_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 5))));
              float v319_data = ir2[2];
              ir2[2] = (v319_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 5))));
              float v325_data = ir2[3];
              ir2[3] = (v325_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 5))));
              float v331_data = ir2[4];
              ir2[4] = (v331_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 5))));
              float v337_data = ir2[5];
              ir2[5] = (v337_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 5))));
              float v343_data = ir2[6];
              ir2[6] = (v343_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 5))));
              float v349_data = ir2[7];
              ir2[7] = (v349_data + (v303_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 5))));
              float v354_data = r0[6];
              float v358_data = ir2[0];
              ir2[0] = (v358_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 6))));
              float v364_data = ir2[1];
              ir2[1] = (v364_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 6))));
              float v370_data = ir2[2];
              ir2[2] = (v370_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 6))));
              float v376_data = ir2[3];
              ir2[3] = (v376_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 6))));
              float v382_data = ir2[4];
              ir2[4] = (v382_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 6))));
              float v388_data = ir2[5];
              ir2[5] = (v388_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 6))));
              float v394_data = ir2[6];
              ir2[6] = (v394_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 6))));
              float v400_data = ir2[7];
              ir2[7] = (v400_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 6))));
              float v405_data = r0[7];
              float v409_data = ir2[0];
              ir2[0] = (v409_data + (v405_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 7))));
              float v415_data = ir2[1];
              ir2[1] = (v415_data + (v405_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 7))));
              float v421_data = ir2[2];
              ir2[2] = (v421_data + (v405_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 7))));
              float v427_data = ir2[3];
              ir2[3] = (v427_data + (v405_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 7))));
              float v433_data = ir2[4];
              ir2[4] = (v433_data + (v405_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 7))));
              float v439_data = ir2[5];
              ir2[5] = (v439_data + (v405_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 7))));
              float v445_data = ir2[6];
              ir2[6] = (v445_data + (v405_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 7))));
              float v451_data = ir2[7];
              ir2[7] = (v451_data + (v405_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 7))));
              float v456_data = r0[8];
              float v460_data = ir2[0];
              ir2[0] = (v460_data + (v456_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 8))));
              float v466_data = ir2[1];
              ir2[1] = (v466_data + (v456_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 8))));
              float v472_data = ir2[2];
              ir2[2] = (v472_data + (v456_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 8))));
              float v478_data = ir2[3];
              ir2[3] = (v478_data + (v456_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 8))));
              float v484_data = ir2[4];
              ir2[4] = (v484_data + (v456_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 8))));
              float v490_data = ir2[5];
              ir2[5] = (v490_data + (v456_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 8))));
              float v496_data = ir2[6];
              ir2[6] = (v496_data + (v456_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 8))));
              float v502_data = ir2[7];
              ir2[7] = (v502_data + (v456_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 8))));
              float v507_data = r0[9];
              float v511_data = ir2[0];
              ir2[0] = (v511_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 9))));
              float v517_data = ir2[1];
              ir2[1] = (v517_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 9))));
              float v523_data = ir2[2];
              ir2[2] = (v523_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 9))));
              float v529_data = ir2[3];
              ir2[3] = (v529_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 9))));
              float v535_data = ir2[4];
              ir2[4] = (v535_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 9))));
              float v541_data = ir2[5];
              ir2[5] = (v541_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 9))));
              float v547_data = ir2[6];
              ir2[6] = (v547_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 9))));
              float v553_data = ir2[7];
              ir2[7] = (v553_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 9))));
              float v558_data = r0[10];
              float v562_data = ir2[0];
              ir2[0] = (v562_data + (v558_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 10))));
              float v568_data = ir2[1];
              ir2[1] = (v568_data + (v558_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 10))));
              float v574_data = ir2[2];
              ir2[2] = (v574_data + (v558_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 10))));
              float v580_data = ir2[3];
              ir2[3] = (v580_data + (v558_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 10))));
              float v586_data = ir2[4];
              ir2[4] = (v586_data + (v558_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 10))));
              float v592_data = ir2[5];
              ir2[5] = (v592_data + (v558_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 10))));
              float v598_data = ir2[6];
              ir2[6] = (v598_data + (v558_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 10))));
              float v604_data = ir2[7];
              ir2[7] = (v604_data + (v558_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 10))));
              float v609_data = r0[11];
              float v613_data = ir2[0];
              ir2[0] = (v613_data + (v609_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 11))));
              float v619_data = ir2[1];
              ir2[1] = (v619_data + (v609_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 11))));
              float v625_data = ir2[2];
              ir2[2] = (v625_data + (v609_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 11))));
              float v631_data = ir2[3];
              ir2[3] = (v631_data + (v609_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 11))));
              float v637_data = ir2[4];
              ir2[4] = (v637_data + (v609_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 11))));
              float v643_data = ir2[5];
              ir2[5] = (v643_data + (v609_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 11))));
              float v649_data = ir2[6];
              ir2[6] = (v649_data + (v609_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 11))));
              float v655_data = ir2[7];
              ir2[7] = (v655_data + (v609_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 11))));
              float v660_data = r0[12];
              float v664_data = ir2[0];
              ir2[0] = (v664_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 12))));
              float v670_data = ir2[1];
              ir2[1] = (v670_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 12))));
              float v676_data = ir2[2];
              ir2[2] = (v676_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 12))));
              float v682_data = ir2[3];
              ir2[3] = (v682_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 12))));
              float v688_data = ir2[4];
              ir2[4] = (v688_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 12))));
              float v694_data = ir2[5];
              ir2[5] = (v694_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 12))));
              float v700_data = ir2[6];
              ir2[6] = (v700_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 12))));
              float v706_data = ir2[7];
              ir2[7] = (v706_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 12))));
              float v711_data = r0[13];
              float v715_data = ir2[0];
              ir2[0] = (v715_data + (v711_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 13))));
              float v721_data = ir2[1];
              ir2[1] = (v721_data + (v711_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 13))));
              float v727_data = ir2[2];
              ir2[2] = (v727_data + (v711_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 13))));
              float v733_data = ir2[3];
              ir2[3] = (v733_data + (v711_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 13))));
              float v739_data = ir2[4];
              ir2[4] = (v739_data + (v711_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 13))));
              float v745_data = ir2[5];
              ir2[5] = (v745_data + (v711_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 13))));
              float v751_data = ir2[6];
              ir2[6] = (v751_data + (v711_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 13))));
              float v757_data = ir2[7];
              ir2[7] = (v757_data + (v711_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 13))));
              float v762_data = r0[14];
              float v766_data = ir2[0];
              ir2[0] = (v766_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 14))));
              float v772_data = ir2[1];
              ir2[1] = (v772_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 14))));
              float v778_data = ir2[2];
              ir2[2] = (v778_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 14))));
              float v784_data = ir2[3];
              ir2[3] = (v784_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 14))));
              float v790_data = ir2[4];
              ir2[4] = (v790_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 14))));
              float v796_data = ir2[5];
              ir2[5] = (v796_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 14))));
              float v802_data = ir2[6];
              ir2[6] = (v802_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 14))));
              float v808_data = ir2[7];
              ir2[7] = (v808_data + (v762_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 14))));
              float v813_data = r0[15];
              float v817_data = ir2[0];
              ir2[0] = (v817_data + (v813_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 15))));
              float v823_data = ir2[1];
              ir2[1] = (v823_data + (v813_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 15))));
              float v829_data = ir2[2];
              ir2[2] = (v829_data + (v813_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 15))));
              float v835_data = ir2[3];
              ir2[3] = (v835_data + (v813_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 15))));
              float v841_data = ir2[4];
              ir2[4] = (v841_data + (v813_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 15))));
              float v847_data = ir2[5];
              ir2[5] = (v847_data + (v813_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 15))));
              float v853_data = ir2[6];
              ir2[6] = (v853_data + (v813_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 15))));
              float v859_data = ir2[7];
              ir2[7] = (v859_data + (v813_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 15))));
              #pragma unroll
              for (int32_t v864_n0 = 0; v864_n0 < 1; ++v864_n0) {
                #pragma unroll
                for (int32_t v865_n1 = 0; v865_n1 < 8; ++v865_n1) {
                  int32_t v866_a = v864_n0 + v865_n1;
                  float v867_data = ir2[v866_a];
                  r2[v866_a] = v867_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v872_i0 = 0; v872_i0 < 1; ++v872_i0) {
                int32_t v880_lead = v16_lead + (v872_i0 * 16);
                #pragma unroll
                for (int32_t v873_i1 = 0; v873_i1 < 8; ++v873_i1) {
                  float v875_data = r2[(v872_i0 + v873_i1)];
                  glb_m0[(v880_lead + (v873_i1 * 16))] = v875_data;
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


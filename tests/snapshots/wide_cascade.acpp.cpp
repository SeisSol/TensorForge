// === base name ===
kernel_1c0f327e992682f2

// === header ===
void launcher_kernel_1c0f327e992682f2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_1c0f327e992682f2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_1c0f327e992682f2(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1c0f327e992682f2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 176 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 176 + 0 + m2_extraOffset];
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
              float r1[11]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v38_lead = v16_lead + (v32_i0 * 16);
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 11; ++v33_i1) {
                  float v41_data = glb_m2[(v38_lead + (v33_i1 * 16))];
                  r1[(v32_i0 + v33_i1)] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[11]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 11)] [(0, 16)]
              float ir2[11]{};
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
              float v97_data = r1[8];
              float v100_data = ir2[8];
              ir2[8] = (v100_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 0))));
              float v103_data = r1[9];
              float v106_data = ir2[9];
              ir2[9] = (v106_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 0))));
              float v109_data = r1[10];
              float v112_data = ir2[10];
              ir2[10] = (v112_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 0))));
              float v117_data = r0[1];
              float v121_data = ir2[0];
              ir2[0] = (v121_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 1))));
              float v127_data = ir2[1];
              ir2[1] = (v127_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 1))));
              float v133_data = ir2[2];
              ir2[2] = (v133_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 1))));
              float v139_data = ir2[3];
              ir2[3] = (v139_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 1))));
              float v145_data = ir2[4];
              ir2[4] = (v145_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 1))));
              float v151_data = ir2[5];
              ir2[5] = (v151_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 1))));
              float v157_data = ir2[6];
              ir2[6] = (v157_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 1))));
              float v163_data = ir2[7];
              ir2[7] = (v163_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 1))));
              float v169_data = ir2[8];
              ir2[8] = (v169_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 1))));
              float v175_data = ir2[9];
              ir2[9] = (v175_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 1))));
              float v181_data = ir2[10];
              ir2[10] = (v181_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 1))));
              float v186_data = r0[2];
              float v190_data = ir2[0];
              ir2[0] = (v190_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 2))));
              float v196_data = ir2[1];
              ir2[1] = (v196_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 2))));
              float v202_data = ir2[2];
              ir2[2] = (v202_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 2))));
              float v208_data = ir2[3];
              ir2[3] = (v208_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 2))));
              float v214_data = ir2[4];
              ir2[4] = (v214_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 2))));
              float v220_data = ir2[5];
              ir2[5] = (v220_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 2))));
              float v226_data = ir2[6];
              ir2[6] = (v226_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 2))));
              float v232_data = ir2[7];
              ir2[7] = (v232_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 2))));
              float v238_data = ir2[8];
              ir2[8] = (v238_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 2))));
              float v244_data = ir2[9];
              ir2[9] = (v244_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 2))));
              float v250_data = ir2[10];
              ir2[10] = (v250_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 2))));
              float v255_data = r0[3];
              float v259_data = ir2[0];
              ir2[0] = (v259_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 3))));
              float v265_data = ir2[1];
              ir2[1] = (v265_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 3))));
              float v271_data = ir2[2];
              ir2[2] = (v271_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 3))));
              float v277_data = ir2[3];
              ir2[3] = (v277_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 3))));
              float v283_data = ir2[4];
              ir2[4] = (v283_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 3))));
              float v289_data = ir2[5];
              ir2[5] = (v289_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 3))));
              float v295_data = ir2[6];
              ir2[6] = (v295_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 3))));
              float v301_data = ir2[7];
              ir2[7] = (v301_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 3))));
              float v307_data = ir2[8];
              ir2[8] = (v307_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 3))));
              float v313_data = ir2[9];
              ir2[9] = (v313_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 3))));
              float v319_data = ir2[10];
              ir2[10] = (v319_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 3))));
              float v324_data = r0[4];
              float v328_data = ir2[0];
              ir2[0] = (v328_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 4))));
              float v334_data = ir2[1];
              ir2[1] = (v334_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 4))));
              float v340_data = ir2[2];
              ir2[2] = (v340_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 4))));
              float v346_data = ir2[3];
              ir2[3] = (v346_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 4))));
              float v352_data = ir2[4];
              ir2[4] = (v352_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 4))));
              float v358_data = ir2[5];
              ir2[5] = (v358_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 4))));
              float v364_data = ir2[6];
              ir2[6] = (v364_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 4))));
              float v370_data = ir2[7];
              ir2[7] = (v370_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 4))));
              float v376_data = ir2[8];
              ir2[8] = (v376_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 4))));
              float v382_data = ir2[9];
              ir2[9] = (v382_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 4))));
              float v388_data = ir2[10];
              ir2[10] = (v388_data + (v324_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 4))));
              float v393_data = r0[5];
              float v397_data = ir2[0];
              ir2[0] = (v397_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 5))));
              float v403_data = ir2[1];
              ir2[1] = (v403_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 5))));
              float v409_data = ir2[2];
              ir2[2] = (v409_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 5))));
              float v415_data = ir2[3];
              ir2[3] = (v415_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 5))));
              float v421_data = ir2[4];
              ir2[4] = (v421_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 5))));
              float v427_data = ir2[5];
              ir2[5] = (v427_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 5))));
              float v433_data = ir2[6];
              ir2[6] = (v433_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 5))));
              float v439_data = ir2[7];
              ir2[7] = (v439_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 5))));
              float v445_data = ir2[8];
              ir2[8] = (v445_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 5))));
              float v451_data = ir2[9];
              ir2[9] = (v451_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 5))));
              float v457_data = ir2[10];
              ir2[10] = (v457_data + (v393_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 5))));
              float v462_data = r0[6];
              float v466_data = ir2[0];
              ir2[0] = (v466_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 6))));
              float v472_data = ir2[1];
              ir2[1] = (v472_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 6))));
              float v478_data = ir2[2];
              ir2[2] = (v478_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 6))));
              float v484_data = ir2[3];
              ir2[3] = (v484_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 6))));
              float v490_data = ir2[4];
              ir2[4] = (v490_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 6))));
              float v496_data = ir2[5];
              ir2[5] = (v496_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 6))));
              float v502_data = ir2[6];
              ir2[6] = (v502_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 6))));
              float v508_data = ir2[7];
              ir2[7] = (v508_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 6))));
              float v514_data = ir2[8];
              ir2[8] = (v514_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 6))));
              float v520_data = ir2[9];
              ir2[9] = (v520_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 6))));
              float v526_data = ir2[10];
              ir2[10] = (v526_data + (v462_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 6))));
              float v531_data = r0[7];
              float v535_data = ir2[0];
              ir2[0] = (v535_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 7))));
              float v541_data = ir2[1];
              ir2[1] = (v541_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 7))));
              float v547_data = ir2[2];
              ir2[2] = (v547_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 7))));
              float v553_data = ir2[3];
              ir2[3] = (v553_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 7))));
              float v559_data = ir2[4];
              ir2[4] = (v559_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 7))));
              float v565_data = ir2[5];
              ir2[5] = (v565_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 7))));
              float v571_data = ir2[6];
              ir2[6] = (v571_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 7))));
              float v577_data = ir2[7];
              ir2[7] = (v577_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 7))));
              float v583_data = ir2[8];
              ir2[8] = (v583_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 7))));
              float v589_data = ir2[9];
              ir2[9] = (v589_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 7))));
              float v595_data = ir2[10];
              ir2[10] = (v595_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 7))));
              float v600_data = r0[8];
              float v604_data = ir2[0];
              ir2[0] = (v604_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 8))));
              float v610_data = ir2[1];
              ir2[1] = (v610_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 8))));
              float v616_data = ir2[2];
              ir2[2] = (v616_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 8))));
              float v622_data = ir2[3];
              ir2[3] = (v622_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 8))));
              float v628_data = ir2[4];
              ir2[4] = (v628_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 8))));
              float v634_data = ir2[5];
              ir2[5] = (v634_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 8))));
              float v640_data = ir2[6];
              ir2[6] = (v640_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 8))));
              float v646_data = ir2[7];
              ir2[7] = (v646_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 8))));
              float v652_data = ir2[8];
              ir2[8] = (v652_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 8))));
              float v658_data = ir2[9];
              ir2[9] = (v658_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 8))));
              float v664_data = ir2[10];
              ir2[10] = (v664_data + (v600_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 8))));
              float v669_data = r0[9];
              float v673_data = ir2[0];
              ir2[0] = (v673_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 9))));
              float v679_data = ir2[1];
              ir2[1] = (v679_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 9))));
              float v685_data = ir2[2];
              ir2[2] = (v685_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 9))));
              float v691_data = ir2[3];
              ir2[3] = (v691_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 9))));
              float v697_data = ir2[4];
              ir2[4] = (v697_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 9))));
              float v703_data = ir2[5];
              ir2[5] = (v703_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 9))));
              float v709_data = ir2[6];
              ir2[6] = (v709_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 9))));
              float v715_data = ir2[7];
              ir2[7] = (v715_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 9))));
              float v721_data = ir2[8];
              ir2[8] = (v721_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 9))));
              float v727_data = ir2[9];
              ir2[9] = (v727_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 9))));
              float v733_data = ir2[10];
              ir2[10] = (v733_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 9))));
              float v738_data = r0[10];
              float v742_data = ir2[0];
              ir2[0] = (v742_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 10))));
              float v748_data = ir2[1];
              ir2[1] = (v748_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 10))));
              float v754_data = ir2[2];
              ir2[2] = (v754_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 10))));
              float v760_data = ir2[3];
              ir2[3] = (v760_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 10))));
              float v766_data = ir2[4];
              ir2[4] = (v766_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 10))));
              float v772_data = ir2[5];
              ir2[5] = (v772_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 10))));
              float v778_data = ir2[6];
              ir2[6] = (v778_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 10))));
              float v784_data = ir2[7];
              ir2[7] = (v784_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 10))));
              float v790_data = ir2[8];
              ir2[8] = (v790_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 10))));
              float v796_data = ir2[9];
              ir2[9] = (v796_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 10))));
              float v802_data = ir2[10];
              ir2[10] = (v802_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 10))));
              float v807_data = r0[11];
              float v811_data = ir2[0];
              ir2[0] = (v811_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 11))));
              float v817_data = ir2[1];
              ir2[1] = (v817_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 11))));
              float v823_data = ir2[2];
              ir2[2] = (v823_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 11))));
              float v829_data = ir2[3];
              ir2[3] = (v829_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 11))));
              float v835_data = ir2[4];
              ir2[4] = (v835_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 11))));
              float v841_data = ir2[5];
              ir2[5] = (v841_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 11))));
              float v847_data = ir2[6];
              ir2[6] = (v847_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 11))));
              float v853_data = ir2[7];
              ir2[7] = (v853_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 11))));
              float v859_data = ir2[8];
              ir2[8] = (v859_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 11))));
              float v865_data = ir2[9];
              ir2[9] = (v865_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 11))));
              float v871_data = ir2[10];
              ir2[10] = (v871_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 11))));
              float v876_data = r0[12];
              float v880_data = ir2[0];
              ir2[0] = (v880_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 12))));
              float v886_data = ir2[1];
              ir2[1] = (v886_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 12))));
              float v892_data = ir2[2];
              ir2[2] = (v892_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 12))));
              float v898_data = ir2[3];
              ir2[3] = (v898_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 12))));
              float v904_data = ir2[4];
              ir2[4] = (v904_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 12))));
              float v910_data = ir2[5];
              ir2[5] = (v910_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 12))));
              float v916_data = ir2[6];
              ir2[6] = (v916_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 12))));
              float v922_data = ir2[7];
              ir2[7] = (v922_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 12))));
              float v928_data = ir2[8];
              ir2[8] = (v928_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 12))));
              float v934_data = ir2[9];
              ir2[9] = (v934_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 12))));
              float v940_data = ir2[10];
              ir2[10] = (v940_data + (v876_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 12))));
              float v945_data = r0[13];
              float v949_data = ir2[0];
              ir2[0] = (v949_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 13))));
              float v955_data = ir2[1];
              ir2[1] = (v955_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 13))));
              float v961_data = ir2[2];
              ir2[2] = (v961_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 13))));
              float v967_data = ir2[3];
              ir2[3] = (v967_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 13))));
              float v973_data = ir2[4];
              ir2[4] = (v973_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 13))));
              float v979_data = ir2[5];
              ir2[5] = (v979_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 13))));
              float v985_data = ir2[6];
              ir2[6] = (v985_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 13))));
              float v991_data = ir2[7];
              ir2[7] = (v991_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 13))));
              float v997_data = ir2[8];
              ir2[8] = (v997_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 13))));
              float v1003_data = ir2[9];
              ir2[9] = (v1003_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 13))));
              float v1009_data = ir2[10];
              ir2[10] = (v1009_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 13))));
              float v1014_data = r0[14];
              float v1018_data = ir2[0];
              ir2[0] = (v1018_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 14))));
              float v1024_data = ir2[1];
              ir2[1] = (v1024_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 14))));
              float v1030_data = ir2[2];
              ir2[2] = (v1030_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 14))));
              float v1036_data = ir2[3];
              ir2[3] = (v1036_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 14))));
              float v1042_data = ir2[4];
              ir2[4] = (v1042_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 14))));
              float v1048_data = ir2[5];
              ir2[5] = (v1048_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 14))));
              float v1054_data = ir2[6];
              ir2[6] = (v1054_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 14))));
              float v1060_data = ir2[7];
              ir2[7] = (v1060_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 14))));
              float v1066_data = ir2[8];
              ir2[8] = (v1066_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 14))));
              float v1072_data = ir2[9];
              ir2[9] = (v1072_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 14))));
              float v1078_data = ir2[10];
              ir2[10] = (v1078_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 14))));
              float v1083_data = r0[15];
              float v1087_data = ir2[0];
              ir2[0] = (v1087_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 15))));
              float v1093_data = ir2[1];
              ir2[1] = (v1093_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 15))));
              float v1099_data = ir2[2];
              ir2[2] = (v1099_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 15))));
              float v1105_data = ir2[3];
              ir2[3] = (v1105_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 15))));
              float v1111_data = ir2[4];
              ir2[4] = (v1111_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 15))));
              float v1117_data = ir2[5];
              ir2[5] = (v1117_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 15))));
              float v1123_data = ir2[6];
              ir2[6] = (v1123_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 15))));
              float v1129_data = ir2[7];
              ir2[7] = (v1129_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 15))));
              float v1135_data = ir2[8];
              ir2[8] = (v1135_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 15))));
              float v1141_data = ir2[9];
              ir2[9] = (v1141_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 15))));
              float v1147_data = ir2[10];
              ir2[10] = (v1147_data + (v1083_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 15))));
              #pragma unroll
              for (int32_t v1152_n0 = 0; v1152_n0 < 1; ++v1152_n0) {
                #pragma unroll
                for (int32_t v1153_n1 = 0; v1153_n1 < 11; ++v1153_n1) {
                  int32_t v1154_a = v1152_n0 + v1153_n1;
                  float v1155_data = ir2[v1154_a];
                  r2[v1154_a] = v1155_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1160_i0 = 0; v1160_i0 < 1; ++v1160_i0) {
                int32_t v1168_lead = v16_lead + (v1160_i0 * 16);
                #pragma unroll
                for (int32_t v1161_i1 = 0; v1161_i1 < 11; ++v1161_i1) {
                  float v1163_data = r2[(v1160_i0 + v1161_i1)];
                  glb_m0[(v1168_lead + (v1161_i1 * 16))] = v1163_data;
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


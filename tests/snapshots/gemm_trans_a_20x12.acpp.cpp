// === base name ===
kernel_7860e657865044f9

// === header ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7860e657865044f9 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7860e657865044f9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7860e657865044f9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
tensorforge::LaunchConfig launch_config_kernel_7860e657865044f9(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7860e657865044f9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7860e657865044f9(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7860e657865044f9(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7860e657865044f9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 12×16(12×16) {0..12}×{0..16} strided
        //   m1 20×12(20×12) {0..20}×{0..12} strided
        //   m2 20×16(20×16) {0..20}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[k,i] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[20,12]],"name":"m1","ordered":false,"parts":1,"shape":[20,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[20,16]],"name":"m2","ordered":false,"parts":1,"shape":[20,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,12]},{"addressing":"strided","bbox":[[0,0],[20,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,16]}],"permute":[[1,0],[0,1]],"target":[[-1,0],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 320 + 0 + m2_extraOffset];
              float r0[20]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v18_lead = item.get_local_id(2) % 16;
              bool v19_g = v18_lead < 12;
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 20; ++v15_i0) {
                if (v19_g) {
                  float v24_data = glb_m1[(v15_i0 + (v18_lead * 20))];
                  r0[v15_i0] = v24_data;
                }
              }
              float r1[32]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v29_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v30_i0 = 0; v30_i0 < 1; ++v30_i0) {
                int32_t v33_lead = v29_lead + (v30_i0 * 16);
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 16; ++v31_i1) {
                  float v36_data = glb_m2[(v33_lead + (v31_i1 * 20))];
                  r1[(v30_i0 + (v31_i1 * 2))] = v36_data;
                }
              }
              if (v29_lead < 4) {
                int32_t v42_lead = v29_lead + 16_i32;
                #pragma unroll
                for (int32_t v40_i1 = 0; v40_i1 < 16; ++v40_i1) {
                  float v45_data = glb_m2[(v42_lead + (v40_i1 * 20))];
                  r1[(1 + (v40_i1 * 2))] = v45_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir2[16]{};
              float v50_data = r0[0];
              float v51_data = r1[0];
              float v54_data = ir2[0];
              ir2[0] = (v54_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v57_data = r1[2];
              float v60_data = ir2[1];
              ir2[1] = (v60_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v63_data = r1[4];
              float v66_data = ir2[2];
              ir2[2] = (v66_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v69_data = r1[6];
              float v72_data = ir2[3];
              ir2[3] = (v72_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v75_data = r1[8];
              float v78_data = ir2[4];
              ir2[4] = (v78_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v81_data = r1[10];
              float v84_data = ir2[5];
              ir2[5] = (v84_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v87_data = r1[12];
              float v90_data = ir2[6];
              ir2[6] = (v90_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v93_data = r1[14];
              float v96_data = ir2[7];
              ir2[7] = (v96_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v99_data = r1[16];
              float v102_data = ir2[8];
              ir2[8] = (v102_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v105_data = r1[18];
              float v108_data = ir2[9];
              ir2[9] = (v108_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v111_data = r1[20];
              float v114_data = ir2[10];
              ir2[10] = (v114_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v117_data = r1[22];
              float v120_data = ir2[11];
              ir2[11] = (v120_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v123_data = r1[24];
              float v126_data = ir2[12];
              ir2[12] = (v126_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v129_data = r1[26];
              float v132_data = ir2[13];
              ir2[13] = (v132_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v135_data = r1[28];
              float v138_data = ir2[14];
              ir2[14] = (v138_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v141_data = r1[30];
              float v144_data = ir2[15];
              ir2[15] = (v144_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v146_data = r0[1];
              float v150_data = ir2[0];
              ir2[0] = (v150_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v156_data = ir2[1];
              ir2[1] = (v156_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v162_data = ir2[2];
              ir2[2] = (v162_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v168_data = ir2[3];
              ir2[3] = (v168_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v174_data = ir2[4];
              ir2[4] = (v174_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v180_data = ir2[5];
              ir2[5] = (v180_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v186_data = ir2[6];
              ir2[6] = (v186_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v192_data = ir2[7];
              ir2[7] = (v192_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v198_data = ir2[8];
              ir2[8] = (v198_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v204_data = ir2[9];
              ir2[9] = (v204_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v210_data = ir2[10];
              ir2[10] = (v210_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v216_data = ir2[11];
              ir2[11] = (v216_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v222_data = ir2[12];
              ir2[12] = (v222_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v228_data = ir2[13];
              ir2[13] = (v228_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v234_data = ir2[14];
              ir2[14] = (v234_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v240_data = ir2[15];
              ir2[15] = (v240_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v242_data = r0[2];
              float v246_data = ir2[0];
              ir2[0] = (v246_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v252_data = ir2[1];
              ir2[1] = (v252_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v258_data = ir2[2];
              ir2[2] = (v258_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v264_data = ir2[3];
              ir2[3] = (v264_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v270_data = ir2[4];
              ir2[4] = (v270_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v276_data = ir2[5];
              ir2[5] = (v276_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v282_data = ir2[6];
              ir2[6] = (v282_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v288_data = ir2[7];
              ir2[7] = (v288_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v294_data = ir2[8];
              ir2[8] = (v294_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v300_data = ir2[9];
              ir2[9] = (v300_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v306_data = ir2[10];
              ir2[10] = (v306_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v312_data = ir2[11];
              ir2[11] = (v312_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v318_data = ir2[12];
              ir2[12] = (v318_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v324_data = ir2[13];
              ir2[13] = (v324_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v330_data = ir2[14];
              ir2[14] = (v330_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v336_data = ir2[15];
              ir2[15] = (v336_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v338_data = r0[3];
              float v342_data = ir2[0];
              ir2[0] = (v342_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v348_data = ir2[1];
              ir2[1] = (v348_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v354_data = ir2[2];
              ir2[2] = (v354_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v360_data = ir2[3];
              ir2[3] = (v360_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v366_data = ir2[4];
              ir2[4] = (v366_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v372_data = ir2[5];
              ir2[5] = (v372_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v378_data = ir2[6];
              ir2[6] = (v378_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v384_data = ir2[7];
              ir2[7] = (v384_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v390_data = ir2[8];
              ir2[8] = (v390_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v396_data = ir2[9];
              ir2[9] = (v396_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v402_data = ir2[10];
              ir2[10] = (v402_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v408_data = ir2[11];
              ir2[11] = (v408_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v414_data = ir2[12];
              ir2[12] = (v414_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v420_data = ir2[13];
              ir2[13] = (v420_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v426_data = ir2[14];
              ir2[14] = (v426_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v432_data = ir2[15];
              ir2[15] = (v432_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v434_data = r0[4];
              float v438_data = ir2[0];
              ir2[0] = (v438_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v444_data = ir2[1];
              ir2[1] = (v444_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v450_data = ir2[2];
              ir2[2] = (v450_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v456_data = ir2[3];
              ir2[3] = (v456_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v462_data = ir2[4];
              ir2[4] = (v462_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v468_data = ir2[5];
              ir2[5] = (v468_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v474_data = ir2[6];
              ir2[6] = (v474_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v480_data = ir2[7];
              ir2[7] = (v480_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v486_data = ir2[8];
              ir2[8] = (v486_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v492_data = ir2[9];
              ir2[9] = (v492_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v498_data = ir2[10];
              ir2[10] = (v498_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v504_data = ir2[11];
              ir2[11] = (v504_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v510_data = ir2[12];
              ir2[12] = (v510_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v516_data = ir2[13];
              ir2[13] = (v516_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v522_data = ir2[14];
              ir2[14] = (v522_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v528_data = ir2[15];
              ir2[15] = (v528_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v530_data = r0[5];
              float v534_data = ir2[0];
              ir2[0] = (v534_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v540_data = ir2[1];
              ir2[1] = (v540_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v546_data = ir2[2];
              ir2[2] = (v546_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v552_data = ir2[3];
              ir2[3] = (v552_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v558_data = ir2[4];
              ir2[4] = (v558_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v564_data = ir2[5];
              ir2[5] = (v564_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v570_data = ir2[6];
              ir2[6] = (v570_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v576_data = ir2[7];
              ir2[7] = (v576_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v582_data = ir2[8];
              ir2[8] = (v582_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v588_data = ir2[9];
              ir2[9] = (v588_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v594_data = ir2[10];
              ir2[10] = (v594_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v600_data = ir2[11];
              ir2[11] = (v600_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v606_data = ir2[12];
              ir2[12] = (v606_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v612_data = ir2[13];
              ir2[13] = (v612_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v618_data = ir2[14];
              ir2[14] = (v618_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v624_data = ir2[15];
              ir2[15] = (v624_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v626_data = r0[6];
              float v630_data = ir2[0];
              ir2[0] = (v630_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v636_data = ir2[1];
              ir2[1] = (v636_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v642_data = ir2[2];
              ir2[2] = (v642_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v648_data = ir2[3];
              ir2[3] = (v648_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v654_data = ir2[4];
              ir2[4] = (v654_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v660_data = ir2[5];
              ir2[5] = (v660_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v666_data = ir2[6];
              ir2[6] = (v666_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v672_data = ir2[7];
              ir2[7] = (v672_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v678_data = ir2[8];
              ir2[8] = (v678_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v684_data = ir2[9];
              ir2[9] = (v684_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v690_data = ir2[10];
              ir2[10] = (v690_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v696_data = ir2[11];
              ir2[11] = (v696_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v702_data = ir2[12];
              ir2[12] = (v702_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v708_data = ir2[13];
              ir2[13] = (v708_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v714_data = ir2[14];
              ir2[14] = (v714_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v720_data = ir2[15];
              ir2[15] = (v720_data + (v626_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v722_data = r0[7];
              float v726_data = ir2[0];
              ir2[0] = (v726_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v732_data = ir2[1];
              ir2[1] = (v732_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v738_data = ir2[2];
              ir2[2] = (v738_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v744_data = ir2[3];
              ir2[3] = (v744_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v750_data = ir2[4];
              ir2[4] = (v750_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v756_data = ir2[5];
              ir2[5] = (v756_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v762_data = ir2[6];
              ir2[6] = (v762_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v768_data = ir2[7];
              ir2[7] = (v768_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v774_data = ir2[8];
              ir2[8] = (v774_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v780_data = ir2[9];
              ir2[9] = (v780_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v786_data = ir2[10];
              ir2[10] = (v786_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v792_data = ir2[11];
              ir2[11] = (v792_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v798_data = ir2[12];
              ir2[12] = (v798_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v804_data = ir2[13];
              ir2[13] = (v804_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v810_data = ir2[14];
              ir2[14] = (v810_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v816_data = ir2[15];
              ir2[15] = (v816_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v818_data = r0[8];
              float v822_data = ir2[0];
              ir2[0] = (v822_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v828_data = ir2[1];
              ir2[1] = (v828_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v834_data = ir2[2];
              ir2[2] = (v834_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v840_data = ir2[3];
              ir2[3] = (v840_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v846_data = ir2[4];
              ir2[4] = (v846_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v852_data = ir2[5];
              ir2[5] = (v852_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v858_data = ir2[6];
              ir2[6] = (v858_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v864_data = ir2[7];
              ir2[7] = (v864_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v870_data = ir2[8];
              ir2[8] = (v870_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v876_data = ir2[9];
              ir2[9] = (v876_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v882_data = ir2[10];
              ir2[10] = (v882_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v888_data = ir2[11];
              ir2[11] = (v888_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v894_data = ir2[12];
              ir2[12] = (v894_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v900_data = ir2[13];
              ir2[13] = (v900_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v906_data = ir2[14];
              ir2[14] = (v906_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v912_data = ir2[15];
              ir2[15] = (v912_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v914_data = r0[9];
              float v918_data = ir2[0];
              ir2[0] = (v918_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v924_data = ir2[1];
              ir2[1] = (v924_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v930_data = ir2[2];
              ir2[2] = (v930_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v936_data = ir2[3];
              ir2[3] = (v936_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v942_data = ir2[4];
              ir2[4] = (v942_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v948_data = ir2[5];
              ir2[5] = (v948_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v954_data = ir2[6];
              ir2[6] = (v954_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v960_data = ir2[7];
              ir2[7] = (v960_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v966_data = ir2[8];
              ir2[8] = (v966_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v972_data = ir2[9];
              ir2[9] = (v972_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v978_data = ir2[10];
              ir2[10] = (v978_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v984_data = ir2[11];
              ir2[11] = (v984_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v990_data = ir2[12];
              ir2[12] = (v990_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v996_data = ir2[13];
              ir2[13] = (v996_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1002_data = ir2[14];
              ir2[14] = (v1002_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1008_data = ir2[15];
              ir2[15] = (v1008_data + (v914_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1010_data = r0[10];
              float v1014_data = ir2[0];
              ir2[0] = (v1014_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1020_data = ir2[1];
              ir2[1] = (v1020_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1026_data = ir2[2];
              ir2[2] = (v1026_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1032_data = ir2[3];
              ir2[3] = (v1032_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1038_data = ir2[4];
              ir2[4] = (v1038_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1044_data = ir2[5];
              ir2[5] = (v1044_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1050_data = ir2[6];
              ir2[6] = (v1050_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1056_data = ir2[7];
              ir2[7] = (v1056_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1062_data = ir2[8];
              ir2[8] = (v1062_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1068_data = ir2[9];
              ir2[9] = (v1068_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1074_data = ir2[10];
              ir2[10] = (v1074_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1080_data = ir2[11];
              ir2[11] = (v1080_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1086_data = ir2[12];
              ir2[12] = (v1086_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1092_data = ir2[13];
              ir2[13] = (v1092_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1098_data = ir2[14];
              ir2[14] = (v1098_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1104_data = ir2[15];
              ir2[15] = (v1104_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1106_data = r0[11];
              float v1110_data = ir2[0];
              ir2[0] = (v1110_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1116_data = ir2[1];
              ir2[1] = (v1116_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1122_data = ir2[2];
              ir2[2] = (v1122_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1128_data = ir2[3];
              ir2[3] = (v1128_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1134_data = ir2[4];
              ir2[4] = (v1134_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1140_data = ir2[5];
              ir2[5] = (v1140_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1146_data = ir2[6];
              ir2[6] = (v1146_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1152_data = ir2[7];
              ir2[7] = (v1152_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1158_data = ir2[8];
              ir2[8] = (v1158_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1164_data = ir2[9];
              ir2[9] = (v1164_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1170_data = ir2[10];
              ir2[10] = (v1170_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1176_data = ir2[11];
              ir2[11] = (v1176_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1182_data = ir2[12];
              ir2[12] = (v1182_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1188_data = ir2[13];
              ir2[13] = (v1188_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1194_data = ir2[14];
              ir2[14] = (v1194_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1200_data = ir2[15];
              ir2[15] = (v1200_data + (v1106_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1202_data = r0[12];
              float v1206_data = ir2[0];
              ir2[0] = (v1206_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1212_data = ir2[1];
              ir2[1] = (v1212_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1218_data = ir2[2];
              ir2[2] = (v1218_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1224_data = ir2[3];
              ir2[3] = (v1224_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1230_data = ir2[4];
              ir2[4] = (v1230_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1236_data = ir2[5];
              ir2[5] = (v1236_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1242_data = ir2[6];
              ir2[6] = (v1242_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1248_data = ir2[7];
              ir2[7] = (v1248_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1254_data = ir2[8];
              ir2[8] = (v1254_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1260_data = ir2[9];
              ir2[9] = (v1260_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1266_data = ir2[10];
              ir2[10] = (v1266_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1272_data = ir2[11];
              ir2[11] = (v1272_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1278_data = ir2[12];
              ir2[12] = (v1278_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1284_data = ir2[13];
              ir2[13] = (v1284_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1290_data = ir2[14];
              ir2[14] = (v1290_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1296_data = ir2[15];
              ir2[15] = (v1296_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1298_data = r0[13];
              float v1302_data = ir2[0];
              ir2[0] = (v1302_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1308_data = ir2[1];
              ir2[1] = (v1308_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1314_data = ir2[2];
              ir2[2] = (v1314_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1320_data = ir2[3];
              ir2[3] = (v1320_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1326_data = ir2[4];
              ir2[4] = (v1326_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1332_data = ir2[5];
              ir2[5] = (v1332_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1338_data = ir2[6];
              ir2[6] = (v1338_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1344_data = ir2[7];
              ir2[7] = (v1344_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1350_data = ir2[8];
              ir2[8] = (v1350_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1356_data = ir2[9];
              ir2[9] = (v1356_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1362_data = ir2[10];
              ir2[10] = (v1362_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1368_data = ir2[11];
              ir2[11] = (v1368_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1374_data = ir2[12];
              ir2[12] = (v1374_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1380_data = ir2[13];
              ir2[13] = (v1380_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1386_data = ir2[14];
              ir2[14] = (v1386_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1392_data = ir2[15];
              ir2[15] = (v1392_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1394_data = r0[14];
              float v1398_data = ir2[0];
              ir2[0] = (v1398_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1404_data = ir2[1];
              ir2[1] = (v1404_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1410_data = ir2[2];
              ir2[2] = (v1410_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1416_data = ir2[3];
              ir2[3] = (v1416_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1422_data = ir2[4];
              ir2[4] = (v1422_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1428_data = ir2[5];
              ir2[5] = (v1428_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1434_data = ir2[6];
              ir2[6] = (v1434_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1440_data = ir2[7];
              ir2[7] = (v1440_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1446_data = ir2[8];
              ir2[8] = (v1446_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1452_data = ir2[9];
              ir2[9] = (v1452_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1458_data = ir2[10];
              ir2[10] = (v1458_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1464_data = ir2[11];
              ir2[11] = (v1464_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1470_data = ir2[12];
              ir2[12] = (v1470_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1476_data = ir2[13];
              ir2[13] = (v1476_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1482_data = ir2[14];
              ir2[14] = (v1482_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1488_data = ir2[15];
              ir2[15] = (v1488_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1490_data = r0[15];
              float v1494_data = ir2[0];
              ir2[0] = (v1494_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1500_data = ir2[1];
              ir2[1] = (v1500_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1506_data = ir2[2];
              ir2[2] = (v1506_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1512_data = ir2[3];
              ir2[3] = (v1512_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1518_data = ir2[4];
              ir2[4] = (v1518_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1524_data = ir2[5];
              ir2[5] = (v1524_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1530_data = ir2[6];
              ir2[6] = (v1530_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1536_data = ir2[7];
              ir2[7] = (v1536_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1542_data = ir2[8];
              ir2[8] = (v1542_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1548_data = ir2[9];
              ir2[9] = (v1548_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1554_data = ir2[10];
              ir2[10] = (v1554_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1560_data = ir2[11];
              ir2[11] = (v1560_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1566_data = ir2[12];
              ir2[12] = (v1566_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1572_data = ir2[13];
              ir2[13] = (v1572_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1578_data = ir2[14];
              ir2[14] = (v1578_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1584_data = ir2[15];
              ir2[15] = (v1584_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1586_data = r0[16];
              float v1587_data = r1[1];
              float v1590_data = ir2[0];
              ir2[0] = (v1590_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1587_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1593_data = r1[3];
              float v1596_data = ir2[1];
              ir2[1] = (v1596_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1593_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1599_data = r1[5];
              float v1602_data = ir2[2];
              ir2[2] = (v1602_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1599_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1605_data = r1[7];
              float v1608_data = ir2[3];
              ir2[3] = (v1608_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1605_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1611_data = r1[9];
              float v1614_data = ir2[4];
              ir2[4] = (v1614_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1611_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1617_data = r1[11];
              float v1620_data = ir2[5];
              ir2[5] = (v1620_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1623_data = r1[13];
              float v1626_data = ir2[6];
              ir2[6] = (v1626_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1623_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1629_data = r1[15];
              float v1632_data = ir2[7];
              ir2[7] = (v1632_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1629_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1635_data = r1[17];
              float v1638_data = ir2[8];
              ir2[8] = (v1638_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1635_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1641_data = r1[19];
              float v1644_data = ir2[9];
              ir2[9] = (v1644_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1641_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1647_data = r1[21];
              float v1650_data = ir2[10];
              ir2[10] = (v1650_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1647_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1653_data = r1[23];
              float v1656_data = ir2[11];
              ir2[11] = (v1656_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1653_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1659_data = r1[25];
              float v1662_data = ir2[12];
              ir2[12] = (v1662_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1659_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1665_data = r1[27];
              float v1668_data = ir2[13];
              ir2[13] = (v1668_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1665_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1671_data = r1[29];
              float v1674_data = ir2[14];
              ir2[14] = (v1674_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1677_data = r1[31];
              float v1680_data = ir2[15];
              ir2[15] = (v1680_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1682_data = r0[17];
              float v1686_data = ir2[0];
              ir2[0] = (v1686_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1587_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1692_data = ir2[1];
              ir2[1] = (v1692_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1593_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1698_data = ir2[2];
              ir2[2] = (v1698_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1599_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1704_data = ir2[3];
              ir2[3] = (v1704_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1605_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1710_data = ir2[4];
              ir2[4] = (v1710_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1611_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1716_data = ir2[5];
              ir2[5] = (v1716_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1722_data = ir2[6];
              ir2[6] = (v1722_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1623_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1728_data = ir2[7];
              ir2[7] = (v1728_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1629_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1734_data = ir2[8];
              ir2[8] = (v1734_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1635_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1740_data = ir2[9];
              ir2[9] = (v1740_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1641_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1746_data = ir2[10];
              ir2[10] = (v1746_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1647_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1752_data = ir2[11];
              ir2[11] = (v1752_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1653_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1758_data = ir2[12];
              ir2[12] = (v1758_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1659_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1764_data = ir2[13];
              ir2[13] = (v1764_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1665_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1770_data = ir2[14];
              ir2[14] = (v1770_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1776_data = ir2[15];
              ir2[15] = (v1776_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1778_data = r0[18];
              float v1782_data = ir2[0];
              ir2[0] = (v1782_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1587_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1788_data = ir2[1];
              ir2[1] = (v1788_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1593_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1794_data = ir2[2];
              ir2[2] = (v1794_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1599_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1800_data = ir2[3];
              ir2[3] = (v1800_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1605_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1806_data = ir2[4];
              ir2[4] = (v1806_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1611_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1812_data = ir2[5];
              ir2[5] = (v1812_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1818_data = ir2[6];
              ir2[6] = (v1818_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1623_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1824_data = ir2[7];
              ir2[7] = (v1824_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1629_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1830_data = ir2[8];
              ir2[8] = (v1830_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1635_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1836_data = ir2[9];
              ir2[9] = (v1836_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1641_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1842_data = ir2[10];
              ir2[10] = (v1842_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1647_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1848_data = ir2[11];
              ir2[11] = (v1848_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1653_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1854_data = ir2[12];
              ir2[12] = (v1854_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1659_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1860_data = ir2[13];
              ir2[13] = (v1860_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1665_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1866_data = ir2[14];
              ir2[14] = (v1866_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1872_data = ir2[15];
              ir2[15] = (v1872_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1874_data = r0[19];
              float v1878_data = ir2[0];
              ir2[0] = (v1878_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1587_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1884_data = ir2[1];
              ir2[1] = (v1884_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1593_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1890_data = ir2[2];
              ir2[2] = (v1890_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1599_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1896_data = ir2[3];
              ir2[3] = (v1896_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1605_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1902_data = ir2[4];
              ir2[4] = (v1902_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1611_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1908_data = ir2[5];
              ir2[5] = (v1908_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1914_data = ir2[6];
              ir2[6] = (v1914_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1623_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1920_data = ir2[7];
              ir2[7] = (v1920_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1629_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1926_data = ir2[8];
              ir2[8] = (v1926_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1635_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1932_data = ir2[9];
              ir2[9] = (v1932_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1641_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1938_data = ir2[10];
              ir2[10] = (v1938_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1647_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1944_data = ir2[11];
              ir2[11] = (v1944_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1653_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1950_data = ir2[12];
              ir2[12] = (v1950_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1659_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1956_data = ir2[13];
              ir2[13] = (v1956_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1665_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1962_data = ir2[14];
              ir2[14] = (v1962_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1671_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1968_data = ir2[15];
              ir2[15] = (v1968_data + (v1874_data * (sycl::select_from_group(item.get_sub_group(), v1677_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              bool v1970_g = v29_lead < 12;
              if (v1970_g) {
                #pragma unroll
                for (int32_t v1971_n1 = 0; v1971_n1 < 16; ++v1971_n1) {
                  float v1973_data = ir2[v1971_n1];
                  r2[v1971_n1] = v1973_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v1970_g) {
                #pragma unroll
                for (int32_t v1975_i1 = 0; v1975_i1 < 16; ++v1975_i1) {
                  float v1977_data = r2[v1975_i1];
                  glb_m0[(v29_lead + (v1975_i1 * 12))] = v1977_data;
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


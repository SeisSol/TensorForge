#! /usr/bin/bash
python benchmark_cuda.py
nvcc -g -G -O4 -dopt on benchmark_cuda_A_mul_Bt_band.cu -o A_Bt_band_opt
nvcc -g -G -O4 -dopt on benchmark_cuda_A_mul_Bt_band_compiler_time_value.cu -o A_Bt_band_opt_ctv
nvcc -g -G -O4 -dopt on benchmark_cuda_A_mul_Bt_single_row_b.cu -o A_Bt_single_row_opt
nvcc -g -G -O4 -dopt on benchmark_cuda_A_mul_Bt_single_row_b_compiler_time_value.cu -o A_Bt_single_row_opt_ctv
nvcc -g -G -O4 -dopt on benchmark_cuda_A_mul_Bt_chequered.cu -o A_Bt_chequered_opt
nvcc -g -G -O4 -dopt on benchmark_cuda_A_mul_Bt_chequered_compiler_time_value.cu -o A_Bt_chequered_opt_ctv
sudo /usr/lib/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu --export /home/primrose/Work/gemmforge/A_Bt_band_opt_rep       --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --filter-mode global --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --set full --import-source no --check-exit-code yes /home/primrose/Work/gemmforge/A_Bt_single_row_opt
sudo /usr/lib/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu --export /home/primrose/Work/gemmforge/A_Bt_band_opt_ctv_rep         --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --filter-mode global --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --set full --import-source no --check-exit-code yes /home/primrose/Work/gemmforge/A_Bt_single_row_opt_ctv
sudo /usr/lib/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu --export /home/primrose/Work/gemmforge/A_Bt_single_row_opt_rep --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --filter-mode global --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --set full --import-source no --check-exit-code yes /home/primrose/Work/gemmforge/A_Bt_band_opt
sudo /usr/lib/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu --export /home/primrose/Work/gemmforge/A_Bt_single_row_opt_ctv_rep   --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --filter-mode global --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --set full --import-source no --check-exit-code yes /home/primrose/Work/gemmforge/A_Bt_band_opt_ctv
sudo /usr/lib/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu --export /home/primrose/Work/gemmforge/A_Bt_chequered_opt_rep    --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --filter-mode global --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --set full --import-source no --check-exit-code yes /home/primrose/Work/gemmforge/A_Bt_chequered_opt
sudo /usr/lib/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu --export /home/primrose/Work/gemmforge/A_Bt_chequered_opt_ctv_rep    --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --filter-mode global --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --set full --import-source no --check-exit-code yes /home/primrose/Work/gemmforge/A_Bt_chequered_opt_ctv

#Copy to local:
#scp -r -o ProxyCommand="ssh -W %h:%p budanaz@lxhalle.in.tum.de" ge69xij@sccs-gpu-login.sccs.in.tum.de:/u/home/ge69xij/gemmfore_remote_retrieved  home/primrose/Work/gemmforge 

#Copy to remote:
#scp -r -o ProxyCommand="ssh -W %h:%p budanaz@lxhalle.in.tum.de" /home/primrose/Work/gemmforge ge69xij@sccs-gpu-login.sccs.in.tum.de:/u/home/ge69xij/gemmfore 

#Login:
#ssh -t budanaz@lxhalle.in.tum.de ssh -t ge69xij@sccs-gpu-login.sccs.in.tum.de

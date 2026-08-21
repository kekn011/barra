TM=$1; TN=$2; TK=$3; MM=$4; MN=$5; LX=$6; LY=$7
cd /mnt/c/Users/kevin/projects/pixel-cluster-base/src/experiments/roofline
sed -e "s/#define TM .*/#define TM $TM/" -e "s/#define TN .*/#define TN $TN/" -e "s/#define TK .*/#define TK $TK/" \
    -e "s/#define MM .*/#define MM $MM/" -e "s/#define MN .*/#define MN $MN/" \
    -e "s/layout(local_size_x = 16, local_size_y = 16)/layout(local_size_x = $LX, local_size_y = $LY)/" \
    -e "s/uint tid=ty\*16u+tx; const uint NT=256u;/uint tid=ty*${LX}u+tx; const uint NT=$((LX*LY))u;/" \
    -e "s/ar\[i\]=As\[ty\*MM+i\]\[kk\]/ar[i]=As[ty*MM+i][kk]/" \
    gemm_rb.comp > gemm_v.comp
NDK=$(ls -d ~/android-sdk/ndk/* | sort -V | tail -1); TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
SYS=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/34
glslangValidator -V --target-env vulkan1.1 gemm_v.comp -o gemm_v.spv >/dev/null 2>&1 || { echo "GLSL FAIL"; exit 1; }
xxd -i gemm_v.spv | sed "s/gemm_v_spv/g_spv/g" > gemm_spv.h
$TC/aarch64-linux-android34-clang -O2 gpugemm.c -o gpugemm -L$SYS -lvulkan 2>&1 | tail -1
$TC/llvm-strip gpugemm; echo "built TM=$TM TN=$TN TK=$TK MM=$MM MN=$MN wg=${LX}x${LY}"

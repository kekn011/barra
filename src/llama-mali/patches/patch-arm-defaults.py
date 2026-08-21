#!/usr/bin/env python3
# Mali-Voreinstellungen: KHR_cooperative_matrix auf ARM standardmaessig AUS (auf Mali-G715 langsamer als mul_mm:
# f16 2048^2 n=128 26,8 vs 65 GFLOPS). Wieder an mit GGML_VK_ARM_COOPMAT=1. Idempotent.
import os
p = os.path.expanduser('~/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp'); s = open(p).read()
if 'vk_arm_coopmat_off' in s:
    print('already patched'); raise SystemExit
helper = '''
// ARM_DEFAULTS: auf Mali ist KHR_coopmat langsamer als der mul_mm-Pfad -> standardmaessig aus (GGML_VK_ARM_COOPMAT=1 schaltet ein)
static bool vk_arm_coopmat_off(const vk::PhysicalDevice & pd) {
    return pd.getProperties().vendorID == VK_VENDOR_ID_ARM && !getenv("GGML_VK_ARM_COOPMAT");
}
'''
anchor = '#define VK_DEVICE_DESCRIPTOR_POOL_SIZE 256\n'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + helper, 1)
old1 = '''            } else if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = true;'''
new1 = '''            } else if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT") && !vk_arm_coopmat_off(device->physical_device)) {
                device->coopmat_support = true;'''
assert s.count(old1) == 1; s = s.replace(old1, new1)
old2 = '''       } else if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = true;'''
new2 = '''       } else if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT") && !vk_arm_coopmat_off(physical_device)) {
            coopmat_support = true;'''
assert s.count(old2) == 1; s = s.replace(old2, new2)
open(p, 'w').write(s); print('patched')

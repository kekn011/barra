// gpugemm - Mali-eigener register-geblockter FP16-GEMM, GFLOPS-Messung.
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "gemm_spv.h"   // g_spv[], g_spv_len
static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev;
static VkQueue queue; static uint32_t qfam; static VkCommandPool pool;
#define CK(c,m) do{ VkResult r=(c); if(r!=VK_SUCCESS){ fprintf(stderr,"%s -> %d\n",m,r); exit(2);} }while(0)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint32_t pick_mem(uint32_t want){ VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
  for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((mp.memoryTypes[i].propertyFlags&want)==want) return i; return UINT32_MAX; }
static void mkbuf(VkDeviceSize sz,uint32_t mt,VkBuffer*b,VkDeviceMemory*m,void**map){
  VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=sz,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
  CK(vkCreateBuffer(dev,&bci,0,b),"buf"); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,*b,&mr);
  VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=mt};
  CK(vkAllocateMemory(dev,&mai,0,m),"alloc"); CK(vkBindBufferMemory(dev,*b,*m,0),"bind");
  if(map) CK(vkMapMemory(dev,*m,0,VK_WHOLE_SIZE,0,map),"map"); }
static float h2f(uint16_t h){ uint32_t s=(uint32_t)(h&0x8000)<<16; int e=(h>>10)&0x1f; uint32_t m=h&0x3ff; uint32_t f;
  if(e==0){ f=s; } else { f=s|((uint32_t)(e-15+127)<<23)|(m<<13); } float r; memcpy(&r,&f,4); return r; }
static uint16_t f2h(float f){ // simpler: nur grobe f16-Kodierung fuer Init (Werte ~1)
  uint32_t x=*(uint32_t*)&f; uint32_t sign=(x>>16)&0x8000; int e=((x>>23)&0xff)-127+15; uint32_t man=(x>>13)&0x3ff;
  if(e<=0) return sign; if(e>=31) return sign|0x7c00; return sign|(e<<10)|man; }
int main(int argc,char**argv){
  uint32_t M=argc>1?atoi(argv[1]):2048, N=argc>2?atoi(argv[2]):2048, K=argc>3?atoi(argv[3]):2048;
  int ITERS=argc>4?atoi(argv[4]):20; uint32_t TILEM=argc>5?atoi(argv[5]):128; uint32_t TILEN=argc>6?atoi(argv[6]):128;
  VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
  CK(vkCreateInstance(&ici,0,&inst),"inst"); uint32_t n=1; vkEnumeratePhysicalDevices(inst,&n,&phys);
  VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys,&pp);
  uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,0);
  VkQueueFamilyProperties qf[16]; if(qn>16)qn=16; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,qf);
  qfam=UINT32_MAX; for(uint32_t i=0;i<qn;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfam=i;break;}
  float prio=1; VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qfam,.queueCount=1,.pQueuePriorities=&prio};
  VkPhysicalDeviceShaderFloat16Int8Features f16={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,.shaderFloat16=VK_TRUE};
  VkPhysicalDevice16BitStorageFeatures s16={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,.pNext=&f16,.storageBuffer16BitAccess=VK_TRUE};
  VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&s16,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
  CK(vkCreateDevice(phys,&dci,0,&dev),"dev"); vkGetDeviceQueue(dev,qfam,0,&queue);
  VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qfam};
  CK(vkCreateCommandPool(dev,&pci,0,&pool),"pool");
  uint32_t mt=pick_mem(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkBuffer A,B,C; VkDeviceMemory Am,Bm,Cm; void *Amap,*Bmap,*Cmap;
  mkbuf((VkDeviceSize)M*K*2,mt,&A,&Am,&Amap); mkbuf((VkDeviceSize)K*N*2,mt,&B,&Bm,&Bmap); mkbuf((VkDeviceSize)M*N*2,mt,&C,&Cm,&Cmap);
  uint16_t h1=f2h(0.01f); uint16_t*ap=Amap,*bp=Bmap;
  for(size_t i=0;i<(size_t)M*K;i++) ap[i]=h1; for(size_t i=0;i<(size_t)K*N;i++) bp[i]=h1;
  VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=g_spv_len,.pCode=(const uint32_t*)g_spv};
  VkShaderModule sm; CK(vkCreateShaderModule(dev,&smci,0,&sm),"shader");
  VkDescriptorSetLayoutBinding b[3]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},{2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
  VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=3,.pBindings=b};
  VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev,&dlci,0,&dsl),"dsl");
  VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,12};
  VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
  VkPipelineLayout pll; CK(vkCreatePipelineLayout(dev,&plci,0,&pll),"pll");
  VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},.layout=pll};
  VkPipeline pipe; CK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,0,&pipe),"pipe");
  VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};
  VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
  VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev,&dpci,0,&dp),"dp");
  VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl};
  VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev,&dsai,&ds),"ds");
  VkDescriptorBufferInfo bi0={A,0,VK_WHOLE_SIZE},bi1={B,0,VK_WHOLE_SIZE},bi2={C,0,VK_WHOLE_SIZE};
  VkWriteDescriptorSet w[3]={{.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi0},
    {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=1,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi1},
    {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=2,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi2}};
  vkUpdateDescriptorSets(dev,3,w,0,0);
  VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
  VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev,&cbai,&cb),"cb");
  VkCommandBufferBeginInfo bg={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  uint32_t pcv[3]={M,N,K}; uint32_t gx=N/TILEN, gy=M/TILEM;
  vkBeginCommandBuffer(cb,&bg); vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
  vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pll,0,1,&ds,0,0);
  vkCmdPushConstants(cb,pll,VK_SHADER_STAGE_COMPUTE_BIT,0,12,pcv);
  VkMemoryBarrier mb={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
  for(int it=0;it<ITERS;it++){ vkCmdDispatch(cb,gx,gy,1); vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0);}
  vkEndCommandBuffer(cb);
  VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fen; vkCreateFence(dev,&fci,0,&fen);
  VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
  CK(vkQueueSubmit(queue,1,&si,fen),"sw"); vkWaitForFences(dev,1,&fen,VK_TRUE,UINT64_MAX); vkResetFences(dev,1,&fen);
  double t0=now(); CK(vkQueueSubmit(queue,1,&si,fen),"s"); vkWaitForFences(dev,1,&fen,VK_TRUE,UINT64_MAX); double dt=(now()-t0)/ITERS;
  double flop=2.0*M*N*K;
  printf("# %s  GEMM f16 %ux%ux%u  grid=%ux%u\n",pp.deviceName,M,N,K,gx,gy);
  printf("MALI-GEMM: %.1f GFLOPS  (%.3f ms)\n",flop/dt/1e9,dt*1e3);
  uint16_t*cp=Cmap; printf("KORREKT: C[0]=%.5f C[mid]=%.5f C[last]=%.5f (erwartet ~0.2048)\n",h2f(cp[0]),h2f(cp[(size_t)M*N/2]),h2f(cp[(size_t)M*N-1]));
  return 0;
}

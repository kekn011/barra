/* gpu-zc — ZERO-COPY-Beweis auf der Mali-GPU (Vulkan).
 * Wir allozieren SELBST einen dmabuf (/dev/dma_heap/system), IMPORTIEREN ihn als
 * VkDeviceMemory (VK_KHR_external_memory_fd / VK_EXT_external_memory_dma_buf) und
 * binden ihn als Output-Buffer C des vadd-Shaders (c[i]=a[i]*2+b[i]). Die GPU
 * rechnet DIREKT in unseren dmabuf; wir lesen das Ergebnis aus DERSELBEN mmap
 * (unabhaengig von Vulkan) -> kein Copy. Bau: NDK clang gpu-zc.c -o gpu-zc -lvulkan */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)
#define CK(call,msg) do{ VkResult _r=(call); if(_r!=VK_SUCCESS){ P("%s -> VkResult %d\n",msg,_r); return 1; } }while(0)

struct dma_heap_allocation_data { __u64 len; __u32 fd; __u32 fd_flags; __u64 heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0, struct dma_heap_allocation_data)
#define N 64
#define NBYTES (N*4)
#define PAGE 4096

static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev;
static VkQueue queue; static uint32_t qfam; static VkCommandPool pool;
static uint32_t g_hostmem=UINT32_MAX;
static PFN_vkGetMemoryFdPropertiesKHR pGetMemFdProps;

static uint32_t pick_mem(uint32_t typeBits, uint32_t want){
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
  for(uint32_t i=0;i<mp.memoryTypeCount;i++)
    if((typeBits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&want)==want) return i;
  return UINT32_MAX;
}

/* normaler host-visible Buffer (fuer A,B) */
static int mkbuf(uint32_t sz, VkBuffer* b, VkDeviceMemory* m, void** map){
  VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=sz,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
  CK(vkCreateBuffer(dev,&bci,0,b),"CreateBuffer");
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,*b,&mr);
  VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=g_hostmem};
  CK(vkAllocateMemory(dev,&mai,0,m),"AllocMem");
  CK(vkBindBufferMemory(dev,*b,*m,0),"Bind");
  CK(vkMapMemory(dev,*m,0,VK_WHOLE_SIZE,0,map),"Map");
  return 0;
}

int main(void){
  setvbuf(stderr,0,_IONBF,0);
  /* ---- 1) dmabuf allozieren + Input NICHT (C ist Output) ---- */
  int heap=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC); if(heap<0){P("open heap fail\n");return 1;}
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=PAGE; d.fd_flags=O_RDWR|O_CLOEXEC;
  if(ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&d)<0){P("dma_heap alloc fail\n");return 1;}
  int dfd=(int)d.fd;
  int32_t* cmap=mmap(0,PAGE,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0);
  if(cmap==MAP_FAILED){P("mmap dmabuf fail\n");return 1;}
  memset(cmap,0,PAGE);
  P("dmabuf fd=%d (Output C) alloziert+gemappt\n",dfd);

  /* ---- 2) Vulkan mit External-Memory-Extensions ---- */
  VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
  CK(vkCreateInstance(&ici,0,&inst),"CreateInstance");
  uint32_t n=1; vkEnumeratePhysicalDevices(inst,&n,&phys);
  VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys,&pp); P("GPU: %s\n",pp.deviceName);
  uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,0);
  VkQueueFamilyProperties qf[16]; if(qn>16)qn=16; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,qf);
  qfam=UINT32_MAX; for(uint32_t i=0;i<qn;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfam=i;break;}
  const char* exts[]={ VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME };
  float prio=1.0f;
  VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qfam,.queueCount=1,.pQueuePriorities=&prio};
  VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=3,.ppEnabledExtensionNames=exts};
  CK(vkCreateDevice(phys,&dci,0,&dev),"CreateDevice (External-Mem-Ext unterstuetzt?)");
  vkGetDeviceQueue(dev,qfam,0,&queue);
  pGetMemFdProps=(PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(dev,"vkGetMemoryFdPropertiesKHR");
  if(!pGetMemFdProps){P("vkGetMemoryFdPropertiesKHR fehlt\n");return 1;}
  g_hostmem=pick_mem(~0u, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qfam};
  CK(vkCreateCommandPool(dev,&pci,0,&pool),"CmdPool");

  /* ---- 3) A,B normal; A[i]=i, B[i]=1000 ---- */
  VkBuffer bA,bB,bC; VkDeviceMemory mA,mB,mC=0; void *pA,*pB;
  if(mkbuf(NBYTES,&bA,&mA,&pA)) return 1;
  if(mkbuf(NBYTES,&bB,&mB,&pB)) return 1;
  for(int i=0;i<N;i++){ ((float*)pA)[i]=(float)i; ((float*)pB)[i]=1000.0f; }

  /* ---- 4) C = IMPORTIERTER dmabuf ---- */
  VkMemoryFdPropertiesKHR fp={.sType=VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
  int impfd=dup(dfd);   /* Vulkan uebernimmt den fd */
  CK(pGetMemFdProps(dev,VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,impfd,&fp),"GetMemoryFdProperties");
  uint32_t mt=pick_mem(fp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  P("importierbare memoryTypeBits=0x%x -> gewaehlt %u\n",fp.memoryTypeBits,mt);
  if(mt==UINT32_MAX){P("kein kompatibler host-visible Typ fuer den dmabuf\n");return 1;}
  VkExternalMemoryBufferCreateInfo ext={.sType=VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
  VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.pNext=&ext,.size=NBYTES,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
  CK(vkCreateBuffer(dev,&bci,0,&bC),"CreateBuffer(C,external)");
  VkImportMemoryFdInfoKHR imp={.sType=VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,.fd=impfd};
  VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.pNext=&imp,.allocationSize=PAGE,.memoryTypeIndex=mt};
  CK(vkAllocateMemory(dev,&mai,0,&mC),"AllocMem(import dmabuf)");
  CK(vkBindBufferMemory(dev,bC,mC,0),"Bind(C)");
  P(">>> dmabuf als VkDeviceMemory IMPORTIERT + an Buffer C gebunden <<<\n");

  /* ---- 5) vadd.spv laden + Pipeline ---- */
  FILE* sf=fopen("/data/local/tmp/vadd.spv","rb"); if(!sf){P("vadd.spv fehlt\n");return 1;}
  fseek(sf,0,SEEK_END); long slen=ftell(sf); fseek(sf,0,SEEK_SET);
  uint32_t* spirv=malloc(slen); if(fread(spirv,1,slen,sf)!=(size_t)slen){P("read spv\n");return 1;} fclose(sf);
  VkShaderModule sh; VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=slen,.pCode=spirv};
  CK(vkCreateShaderModule(dev,&smci,0,&sh),"ShaderModule");
  VkDescriptorSetLayoutBinding lb[3];
  for(int i=0;i<3;i++) lb[i]=(VkDescriptorSetLayoutBinding){.binding=i,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayout dsl; VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=3,.pBindings=lb};
  CK(vkCreateDescriptorSetLayout(dev,&dlci,0,&dsl),"DSL");
  VkPipelineLayout pll; VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl};
  CK(vkCreatePipelineLayout(dev,&plci,0,&pll),"PLL");
  VkPipeline pipe; VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sh,.pName="main"},.layout=pll};
  CK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,0,&pipe),"Pipeline");

  /* ---- 6) Descriptors + Dispatch ---- */
  VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=3};
  VkDescriptorPool dpool; VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
  CK(vkCreateDescriptorPool(dev,&dpci,0,&dpool),"DPool");
  VkDescriptorSet dset; VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dpool,.descriptorSetCount=1,.pSetLayouts=&dsl};
  CK(vkAllocateDescriptorSets(dev,&dsai,&dset),"DSet");
  VkBuffer bufs[3]={bA,bB,bC}; VkDescriptorBufferInfo dbi[3]; VkWriteDescriptorSet wr[3];
  for(int i=0;i<3;i++){ dbi[i]=(VkDescriptorBufferInfo){.buffer=bufs[i],.offset=0,.range=VK_WHOLE_SIZE};
    wr[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=dset,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbi[i]};}
  vkUpdateDescriptorSets(dev,3,wr,0,0);
  VkCommandBuffer cb; VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
  CK(vkAllocateCommandBuffers(dev,&cbai,&cb),"CmdBuf");
  VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cb,&bi);
  vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
  vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pll,0,1,&dset,0,0);
  vkCmdDispatch(cb,N/64,1,1);
  vkEndCommandBuffer(cb);
  VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
  CK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE),"Submit");
  vkQueueWaitIdle(queue);

  /* ---- 7) Ergebnis aus UNSERER dmabuf-mmap lesen (nicht ueber Vulkan) ---- */
  float* c=(float*)cmap;
  P("output (aus UNSERER dmabuf-mmap): c[0]=%.0f c[1]=%.0f c[10]=%.0f c[63]=%.0f\n",c[0],c[1],c[10],c[63]);
  P("erwartet c[i]=a[i]*2+b[i]=i*2+1000: [1000 1002 1020 1126]\n");
  if(c[0]==1000 && c[1]==1002 && c[10]==1020 && c[63]==1126)
    P(">>> ZERO-COPY OK: die GPU hat direkt in unseren dmabuf gerechnet, kein Copy. <<<\n");
  else P(">>> Ergebnis unerwartet <<<\n");
  return 0;
}

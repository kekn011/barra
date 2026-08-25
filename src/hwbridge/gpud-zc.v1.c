/* gpud-zc — GPU-Compute-Daemon mit ZERO-COPY-Transport (Mali-Vulkan).
 * Der Client schickt dmabuf-fds via SCM_RIGHTS + SPIR-V; KEINE Pufferdaten ueber
 * den Socket. gpud-zc importiert jeden fd als VkDeviceMemory (external_memory_fd),
 * bindet ihn als Storage-Buffer, dispatcht, schickt nur status zurueck. Der Client
 * liest Outputs aus seiner EIGENEN dmabuf-mmap. Bau: NDK clang -lvulkan.
 *
 * Protokoll GPZC (ein Request pro Verbindung):
 *   sendmsg: iov=[magic 0x47505A43, spirv_len, gx, gy, gz, nbuf] + SCM_RIGHTS: nbuf fds
 *   dann normal: nbuf * u32 size (dmabuf-Groesse je Puffer), dann spirv[spirv_len]
 *   Antwort: u32 status (0=ok) */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <errno.h>
#define MAGIC 0x47505A43u
#define MAXBUF 16
#define MAXSPIRV (16*1024*1024)
#define LOG(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev;
static VkQueue queue; static uint32_t qfam; static VkCommandPool pool;
static PFN_vkGetMemoryFdPropertiesKHR pGetMemFdProps;
#define CK(call,msg) do{ VkResult _r=(call); if(_r!=VK_SUCCESS){ LOG("%s -> %d\n",msg,_r); return -1; } }while(0)

static uint32_t pick_mem(uint32_t typeBits, uint32_t want){
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
  for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((typeBits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&want)==want) return i;
  return UINT32_MAX;
}
static int gpu_init(void){
  VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
  CK(vkCreateInstance(&ici,0,&inst),"CreateInstance");
  uint32_t n=1; vkEnumeratePhysicalDevices(inst,&n,&phys);
  VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys,&pp); LOG("[gpud-zc] GPU: %s\n",pp.deviceName);
  uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,0);
  VkQueueFamilyProperties qf[16]; if(qn>16)qn=16; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,qf);
  qfam=UINT32_MAX; for(uint32_t i=0;i<qn;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfam=i;break;}
  const char* exts[]={VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
  float prio=1.0f;
  VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qfam,.queueCount=1,.pQueuePriorities=&prio};
  VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=3,.ppEnabledExtensionNames=exts};
  CK(vkCreateDevice(phys,&dci,0,&dev),"CreateDevice");
  vkGetDeviceQueue(dev,qfam,0,&queue);
  pGetMemFdProps=(PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(dev,"vkGetMemoryFdPropertiesKHR");
  if(!pGetMemFdProps){ LOG("no GetMemoryFdPropertiesKHR\n"); return -1; }
  VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qfam};
  CK(vkCreateCommandPool(dev,&pci,0,&pool),"CmdPool");
  LOG("[gpud-zc] Vulkan bereit (external-memory)\n");
  return 0;
}
/* dmabuf-fd -> VkBuffer (importiert, kein Copy). fd wird von Vulkan uebernommen. */
static int import_buf(int fd, uint32_t size, VkBuffer* b, VkDeviceMemory* m){
  VkMemoryFdPropertiesKHR fp={.sType=VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
  if(pGetMemFdProps(dev,VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,fd,&fp)!=VK_SUCCESS) return -1;
  uint32_t mt=pick_mem(fp.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if(mt==UINT32_MAX) return -1;
  VkExternalMemoryBufferCreateInfo ext={.sType=VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
  VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.pNext=&ext,.size=size,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
  if(vkCreateBuffer(dev,&bci,0,b)!=VK_SUCCESS) return -1;
  VkImportMemoryFdInfoKHR imp={.sType=VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,.fd=fd};
  VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.pNext=&imp,.allocationSize=size,.memoryTypeIndex=mt};
  if(vkAllocateMemory(dev,&mai,0,m)!=VK_SUCCESS){ vkDestroyBuffer(dev,*b,0); return -1; }
  if(vkBindBufferMemory(dev,*b,*m,0)!=VK_SUCCESS){ vkFreeMemory(dev,*m,0); vkDestroyBuffer(dev,*b,0); return -1; }
  return 0;
}
static int rfull(int fd,void*b,long n){char*p=b;while(n>0){ssize_t r=read(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}

/* Pipeline-Cache: Shader-Compile + Pipeline-Erstellung dominieren die Latenz.
 * Cache nach SPIR-V-Inhalt (+nbuf); wiederholte gleiche Kernel kosten dann nur
 * noch dmabuf-Import + Descriptor + Dispatch. */
typedef struct { int used; uint64_t hash; uint32_t slen,nbuf;
  VkShaderModule shader; VkDescriptorSetLayout dsl; VkPipelineLayout pll; VkPipeline pipe; } PCacheEnt;
#define PCACHE 16
static PCacheEnt g_pc[PCACHE];
static uint32_t g_pc_next=0;
static long g_pc_hits=0, g_pc_miss=0;
static uint64_t fnv1a(const void* p, uint32_t n){ const unsigned char* b=p; uint64_t h=1469598103934665603ULL;
  for(uint32_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
static PCacheEnt* get_pipe(const uint32_t* spirv, uint32_t slen, uint32_t nbuf){
  uint64_t h=fnv1a(spirv,slen);
  for(int i=0;i<PCACHE;i++) if(g_pc[i].used&&g_pc[i].hash==h&&g_pc[i].slen==slen&&g_pc[i].nbuf==nbuf){ g_pc_hits++; return &g_pc[i]; }
  g_pc_miss++;
  int idx=-1; for(int i=0;i<PCACHE;i++) if(!g_pc[i].used){ idx=i; break; }
  if(idx<0){ idx=g_pc_next; g_pc_next=(g_pc_next+1)%PCACHE; PCacheEnt* e=&g_pc[idx];
    if(e->pipe)vkDestroyPipeline(dev,e->pipe,0); if(e->pll)vkDestroyPipelineLayout(dev,e->pll,0);
    if(e->dsl)vkDestroyDescriptorSetLayout(dev,e->dsl,0); if(e->shader)vkDestroyShaderModule(dev,e->shader,0);
    memset(e,0,sizeof *e); }
  PCacheEnt* e=&g_pc[idx];
  VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=slen,.pCode=spirv};
  if(vkCreateShaderModule(dev,&smci,0,&e->shader)!=VK_SUCCESS) return 0;
  VkDescriptorSetLayoutBinding lb[MAXBUF];
  for(uint32_t i=0;i<nbuf;i++) lb[i]=(VkDescriptorSetLayoutBinding){.binding=i,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=nbuf,.pBindings=lb};
  if(vkCreateDescriptorSetLayout(dev,&dlci,0,&e->dsl)!=VK_SUCCESS){ vkDestroyShaderModule(dev,e->shader,0); return 0; }
  VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&e->dsl};
  if(vkCreatePipelineLayout(dev,&plci,0,&e->pll)!=VK_SUCCESS){ vkDestroyDescriptorSetLayout(dev,e->dsl,0); vkDestroyShaderModule(dev,e->shader,0); return 0; }
  VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=e->shader,.pName="main"},.layout=e->pll};
  if(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,0,&e->pipe)!=VK_SUCCESS){ vkDestroyPipelineLayout(dev,e->pll,0); vkDestroyDescriptorSetLayout(dev,e->dsl,0); vkDestroyShaderModule(dev,e->shader,0); return 0; }
  e->used=1; e->hash=h; e->slen=slen; e->nbuf=nbuf;
  return e;
}

static void serve(int c){
  uint32_t status=1;
  uint32_t hdr[6]; int fds[MAXBUF]; int nfd=0;
  char cbuf[CMSG_SPACE(sizeof(int)*MAXBUF)];
  struct iovec iov={.iov_base=hdr,.iov_len=sizeof hdr};
  struct msghdr msg={.msg_iov=&iov,.msg_iovlen=1,.msg_control=cbuf,.msg_controllen=sizeof cbuf};
  if(recvmsg(c,&msg,MSG_WAITALL)!=(ssize_t)sizeof hdr){ return; }
  if(hdr[0]!=MAGIC){ return; }
  struct cmsghdr* cm=CMSG_FIRSTHDR(&msg);
  if(cm && cm->cmsg_type==SCM_RIGHTS){ nfd=(int)((cm->cmsg_len-CMSG_LEN(0))/sizeof(int));
    if(nfd>MAXBUF){ for(int i=0;i<nfd;i++) close(((int*)CMSG_DATA(cm))[i]); return; }   /* sonst Stack-Overflow von fds[MAXBUF] */
    memcpy(fds,CMSG_DATA(cm),nfd*sizeof(int)); }
  uint32_t spirv_len=hdr[1],gx=hdr[2],gy=hdr[3],gz=hdr[4],nbuf=hdr[5];
  if(nbuf==0||nbuf>MAXBUF||(int)nbuf!=nfd||spirv_len==0||spirv_len>MAXSPIRV||(spirv_len&3)){ write(c,&status,4); goto done_close; }
  uint32_t bsize[MAXBUF];
  if(rfull(c,bsize,nbuf*4)){ write(c,&status,4); goto done_close; }
  uint32_t* spirv=malloc(spirv_len);
  if(!spirv||rfull(c,spirv,spirv_len)){ free(spirv); write(c,&status,4); goto done_close; }

  VkBuffer buf[MAXBUF]={0}; VkDeviceMemory mem[MAXBUF]={0};
  int ok=1;
  for(uint32_t i=0;i<nbuf;i++) if(import_buf(fds[i],bsize[i],&buf[i],&mem[i])){ LOG("import buf %u fail\n",i); ok=0; break; }

  VkDescriptorSetLayout dsl=0; VkPipelineLayout pll=0; VkPipeline pipe=0;
  VkDescriptorPool dpool=0; VkDescriptorSet dset=0; VkCommandBuffer cb=0;
  if(ok){ PCacheEnt* pc=get_pipe(spirv,spirv_len,nbuf);   /* Shader+Pipeline aus dem Cache */
    if(!pc) ok=0; else { dsl=pc->dsl; pll=pc->pll; pipe=pc->pipe; }
    LOG("[gpud-zc] req nbuf=%u cache hits=%ld miss=%ld\n",nbuf,g_pc_hits,g_pc_miss);
  }
  if(ok){
    VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=nbuf};
    VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
    if(vkCreateDescriptorPool(dev,&dpci,0,&dpool)!=VK_SUCCESS) ok=0;
  }
  if(ok){
    VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dpool,.descriptorSetCount=1,.pSetLayouts=&dsl};
    if(vkAllocateDescriptorSets(dev,&dsai,&dset)!=VK_SUCCESS) ok=0;
  }
  if(ok){
    VkDescriptorBufferInfo dbi[MAXBUF]; VkWriteDescriptorSet wr[MAXBUF];
    for(uint32_t i=0;i<nbuf;i++){ dbi[i]=(VkDescriptorBufferInfo){.buffer=buf[i],.offset=0,.range=VK_WHOLE_SIZE};
      wr[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=dset,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbi[i]};}
    vkUpdateDescriptorSets(dev,nbuf,wr,0,0);
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    if(vkAllocateCommandBuffers(dev,&cbai,&cb)==VK_SUCCESS){
      VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      vkBeginCommandBuffer(cb,&bi);
      vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
      vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pll,0,1,&dset,0,0);
      vkCmdDispatch(cb,gx,gy,gz);
      vkEndCommandBuffer(cb);
      VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
      if(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE)==VK_SUCCESS){ vkQueueWaitIdle(queue); status=0; }
    }
  }
  write(c,&status,4);   /* Client liest Outputs aus SEINER dmabuf-mmap; keine Daten hier */

  if(cb) vkFreeCommandBuffers(dev,pool,1,&cb);
  if(dpool) vkDestroyDescriptorPool(dev,dpool,0);
  /* dsl/pll/pipe/shader gehoeren dem Pipeline-Cache und bleiben stehen */
  (void)dsl;(void)pll;(void)pipe;
  for(uint32_t i=0;i<nbuf;i++){ if(buf[i]) vkDestroyBuffer(dev,buf[i],0); if(mem[i]) vkFreeMemory(dev,mem[i],0); }
  free(spirv);
done_close:
  for(int i=0;i<nfd;i++) close(fds[i]);   /* unsere Kopien der fds schliessen */
}

int main(int argc,char**argv){
  const char* sock=argc>1?argv[1]:"/opt/hwbridge/gpuzc.sock";
  signal(SIGPIPE,SIG_IGN);
  if(gpu_init()) return 1;
  int srv=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof a.sun_path-1);
  unlink(sock);
  if(bind(srv,(struct sockaddr*)&a,sizeof a)<0){ LOG("bind %s: %s\n",sock,strerror(errno)); return 1; }
  chmod(sock,0666); listen(srv,8);
  LOG("[gpud-zc] lauscht auf %s (Zero-Copy, SCM_RIGHTS)\n",sock);
  for(;;){ int c=accept(srv,0,0); if(c<0)continue; serve(c); close(c); }
  return 0;
}

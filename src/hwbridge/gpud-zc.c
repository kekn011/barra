/* gpud-zc v3 — GPU-Compute-Daemon mit ZERO-COPY-Transport (Mali-Vulkan). v3 (GPZ3, s.u.) fuer ggml-gpud.
 * Der Client schickt dmabuf-fds via SCM_RIGHTS + SPIR-V; KEINE Pufferdaten ueber
 * den Socket. Importierte dmabufs sind VkDeviceMemory (external_memory_fd); die GPU
 * rechnet in-place, der Client liest aus seiner EIGENEN mmap. Bau: NDK clang -lvulkan.
 *
 * Zwei Protokolle auf demselben Socket, unterschieden am ersten u32 (Magic):
 *
 * (A) GPZC 0x47505A43 — Einzel-Request (v1, kompatibel):
 *   sendmsg iov=[magic, spirv_len, gx, gy, gz, nbuf] + SCM_RIGHTS(nbuf fds);
 *   dann nbuf*u32 size, dann spirv; Antwort u32 status. Import je Aufruf.
 *
 * (B) GPZ2 0x47505A32 — SESSION (v2): Verbindung bleibt offen, Puffer werden EINMAL
 *   importiert und ueber Handles angesprochen; K Stufen in EINEM Roundtrip.
 *   Jeder Befehl beginnt mit 6*u32 [magic, cmd, a, b, c, d]:
 *   cmd=1 IMPORT  : [.,1,nbuf,0,0,0] + SCM_RIGHTS(nbuf fds); dann nbuf*u32 size
 *                   -> u32 status, dann nbuf*u32 handle
 *   cmd=2 DISPATCH: [.,2,nstage,0,0,0]; je Stufe: u32 spirv_len,gx,gy,gz,nbuf,
 *                   nbuf*u32 handle, spirv[spirv_len]   -> u32 status
 *                   (ein Command-Buffer; Memory-Barrier zwischen den Stufen,
 *                    HOST_WRITE->SHADER am Anfang, SHADER_WRITE->HOST_READ am Ende; Fence)
 *   cmd=3 RELEASE : [.,3,nbuf,0,0,0]; dann nbuf*u32 handle -> u32 status
 *   Verbindungsende gibt alle Handles der Session frei.
 * fd-Eigentum: Vulkan uebernimmt den importierten fd (Spec) -> wir importieren ein
 * dup() und schliessen unsere Kopie selbst; KEIN Double-Close mehr (v1-Bug). */
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
#include <pthread.h>
#include <time.h>
#define MAGIC1 0x47505A43u
#define MAGIC2 0x47505A32u
#define MAXBUF 16          /* Puffer je Stufe / je IMPORT */
#define MAXH   64          /* Handles je Session */
#define MAXSTAGE 32
#define MAXSPIRV (16*1024*1024)
#define LOG(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev;
static VkQueue queue; static uint32_t qfam; static VkCommandPool pool;
static PFN_vkGetMemoryFdPropertiesKHR pGetMemFdProps;
static VkDeviceSize g_sbo_align=16;   /* minStorageBufferOffsetAlignment (v3 Binding-Offsets) */
/* Ein Thread je Verbindung (Sessions duerfen andere Clients nicht blockieren);
 * ALLE Vulkan-Aufrufe (Import/Pipeline-Cache/Dispatch/Free) laufen unter g_vk. */
static pthread_mutex_t g_vk=PTHREAD_MUTEX_INITIALIZER;
#define VK_LOCK()   pthread_mutex_lock(&g_vk)
#define VK_UNLOCK() pthread_mutex_unlock(&g_vk)
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
  /* Basis-Extensions + optional Integer-Dot-Product (fuer dotPacked4x8EXT im mmvq-Kernel, ggml-gpud M2). */
  const char* exts[8]; uint32_t nx=0;
  exts[nx++]=VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME;
  exts[nx++]=VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
  exts[nx++]=VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME;
  /* push_descriptor: auf Mali-G715 beim Submit Geraeteverlust (30.8.) -> Sets aus Session-Pool */
  int has_idp=0;
  { uint32_t ne=0; vkEnumerateDeviceExtensionProperties(phys,0,&ne,0); VkExtensionProperties* ep=calloc(ne?ne:1,sizeof *ep);
    if(ep){ vkEnumerateDeviceExtensionProperties(phys,0,&ne,ep);
      for(uint32_t i=0;i<ne;i++) if(!strcmp(ep[i].extensionName,VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) has_idp=1;
      free(ep); } }
  if(has_idp) exts[nx++]=VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
  VkPhysicalDeviceShaderIntegerDotProductFeatures idp={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,.shaderIntegerDotProduct=VK_TRUE};
  /* Subgroup-Eigenschaften protokollieren (Cluster-Reduktion im GEMV braucht CLUSTERED im COMPUTE-Stage). */
  VkPhysicalDeviceSubgroupProperties sgp={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 pp2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&sgp};
  vkGetPhysicalDeviceProperties2(phys,&pp2);
  LOG("[gpud-zc] subgroup: size %u, clustered=%d arithmetic=%d compute-stage=%d, integer-dot=%d\n",
      sgp.subgroupSize,(sgp.supportedOperations&VK_SUBGROUP_FEATURE_CLUSTERED_BIT)?1:0,
      (sgp.supportedOperations&VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)?1:0,
      (sgp.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)?1:0,has_idp);
  float prio=1.0f;
  VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qfam,.queueCount=1,.pQueuePriorities=&prio};
  VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=has_idp?&idp:NULL,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=nx,.ppEnabledExtensionNames=exts};
  CK(vkCreateDevice(phys,&dci,0,&dev),"CreateDevice");
  vkGetDeviceQueue(dev,qfam,0,&queue);
  pGetMemFdProps=(PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(dev,"vkGetMemoryFdPropertiesKHR");
  if(!pGetMemFdProps){ LOG("no GetMemoryFdPropertiesKHR\n"); return -1; }
  VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qfam,.flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT};
  CK(vkCreateCommandPool(dev,&pci,0,&pool),"CmdPool");
  g_sbo_align=pp.limits.minStorageBufferOffsetAlignment; if(!g_sbo_align) g_sbo_align=16;
  { uint32_t ne=0; vkEnumerateDeviceExtensionProperties(phys,0,&ne,0); VkExtensionProperties* ep=calloc(ne,sizeof *ep); vkEnumerateDeviceExtensionProperties(phys,0,&ne,ep); int pd=0,dt=0; for(uint32_t i=0;i<ne;i++){ if(!strcmp(ep[i].extensionName,"VK_KHR_push_descriptor")) pd=1; if(!strcmp(ep[i].extensionName,"VK_EXT_descriptor_buffer")) dt=1; } free(ep); LOG("[gpud-zc] Extensions: push_descriptor=%d descriptor_buffer=%d (von %u)\n",pd,dt,ne); }
  LOG("[gpud-zc] Vulkan bereit (external-memory, v2 sessions, v3 ggml: sbo-align %u, push %u B)\n",(unsigned)g_sbo_align,pp.limits.maxPushConstantsSize);
  return 0;
}
/* dmabuf-fd -> VkBuffer (importiert, kein Copy). Importiert wird ein dup(); der
 * Aufrufer behaelt/schliesst sein fd selbst. */
static int import_buf(int fd, uint32_t size, VkBuffer* b, VkDeviceMemory* m){
  VkMemoryFdPropertiesKHR fp={.sType=VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
  if(pGetMemFdProps(dev,VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,fd,&fp)!=VK_SUCCESS) return -1;
  uint32_t mt=pick_mem(fp.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if(mt==UINT32_MAX) mt=pick_mem(fp.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if(mt==UINT32_MAX) return -1;
  VkExternalMemoryBufferCreateInfo ext={.sType=VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
  VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.pNext=&ext,.size=size,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
  if(vkCreateBuffer(dev,&bci,0,b)!=VK_SUCCESS) return -1;
  int dfd=dup(fd); if(dfd<0){ vkDestroyBuffer(dev,*b,0); return -1; }
  VkImportMemoryFdInfoKHR imp={.sType=VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,.fd=dfd};
  VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.pNext=&imp,.allocationSize=size,.memoryTypeIndex=mt};
  if(vkAllocateMemory(dev,&mai,0,m)!=VK_SUCCESS){ close(dfd); vkDestroyBuffer(dev,*b,0); return -1; }
  /* ab hier gehoert dfd Vulkan */
  if(vkBindBufferMemory(dev,*b,*m,0)!=VK_SUCCESS){ vkFreeMemory(dev,*m,0); vkDestroyBuffer(dev,*b,0); return -1; }
  return 0;
}
static int rfull(int fd,void*b,long n){char*p=b;while(n>0){ssize_t r=read(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}
static int wfull(int fd,const void*b,long n){const char*p=b;while(n>0){ssize_t r=write(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}

/* Pipeline-Cache: Shader-Compile + Pipeline-Erstellung dominieren die Latenz.
 * Cache nach SPIR-V-Inhalt (+nbuf). */
typedef struct { int used; int pin; uint64_t hash; uint32_t slen,nbuf;
  VkShaderModule shader; VkDescriptorSetLayout dsl; VkPipelineLayout pll; VkPipeline pipe; } PCacheEnt;
#define PCACHE 16
static PCacheEnt g_pc[PCACHE];
static uint32_t g_pc_next=0;
static long g_pc_hits=0, g_pc_miss=0;
/* Alle in EINEM Dispatch referenzierten Eintraege werden gepinnt, damit die
 * Round-Robin-Eviction sie nicht zerstoert, waehrend spaetere Stufen ihre
 * PCacheEnt*-Zeiger noch halten (MAXSTAGE=32 > PCACHE=16). Nach run_stages
 * (noch unter g_vk) alle wieder entpinnen. */
static void unpin_all(void){ for(int i=0;i<PCACHE;i++) g_pc[i].pin=0; }
static uint64_t fnv1a(const void* p, uint32_t n){ const unsigned char* b=p; uint64_t h=1469598103934665603ULL;
  for(uint32_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
static PCacheEnt* get_pipe(const uint32_t* spirv, uint32_t slen, uint32_t nbuf){
  uint64_t h=fnv1a(spirv,slen);
  for(int i=0;i<PCACHE;i++) if(g_pc[i].used&&g_pc[i].hash==h&&g_pc[i].slen==slen&&g_pc[i].nbuf==nbuf){ g_pc_hits++; g_pc[i].pin=1; return &g_pc[i]; }
  g_pc_miss++;
  int idx=-1; for(int i=0;i<PCACHE;i++) if(!g_pc[i].used){ idx=i; break; }
  if(idx<0){
    /* freien Opfer-Slot suchen (Round-Robin), gepinnte Slots ueberspringen */
    for(uint32_t t=0;t<PCACHE;t++){ uint32_t cand=(g_pc_next+t)%PCACHE; if(!g_pc[cand].pin){ idx=(int)cand; break; } }
    if(idx<0){ LOG("[gpud-zc] pipeline-cache erschoepft: Dispatch referenziert mehr als %d distinkte Pipelines — abgelehnt\n",PCACHE); return 0; }
    g_pc_next=((uint32_t)idx+1)%PCACHE;
    PCacheEnt* e=&g_pc[idx];
    vkDeviceWaitIdle(dev);
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
  e->used=1; e->pin=1; e->hash=h; e->slen=slen; e->nbuf=nbuf;
  return e;
}

/* ---- gemeinsamer Ausfuehrungskern: K Stufen in EINEM Command-Buffer --------- */
typedef struct { PCacheEnt* pc; uint32_t gx,gy,gz,nbuf; VkBuffer buf[MAXBUF]; } Stage;
static int run_stages(Stage* st, uint32_t nstage){
  uint32_t total=0; for(uint32_t s=0;s<nstage;s++) total+=st[s].nbuf;
  VkDescriptorPool dpool=0; VkCommandBuffer cb=0; VkFence fence=0; int rc=-1;
  VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=total};
  VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=nstage,.poolSizeCount=1,.pPoolSizes=&ps};
  if(vkCreateDescriptorPool(dev,&dpci,0,&dpool)!=VK_SUCCESS) return -1;
  VkDescriptorSet dset[MAXSTAGE];
  for(uint32_t s=0;s<nstage;s++){
    VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dpool,.descriptorSetCount=1,.pSetLayouts=&st[s].pc->dsl};
    if(vkAllocateDescriptorSets(dev,&dsai,&dset[s])!=VK_SUCCESS) goto out;
    VkDescriptorBufferInfo dbi[MAXBUF]; VkWriteDescriptorSet wr[MAXBUF];
    for(uint32_t i=0;i<st[s].nbuf;i++){ dbi[i]=(VkDescriptorBufferInfo){.buffer=st[s].buf[i],.offset=0,.range=VK_WHOLE_SIZE};
      wr[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=dset[s],.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbi[i]};}
    vkUpdateDescriptorSets(dev,st[s].nbuf,wr,0,0);
  }
  VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
  if(vkAllocateCommandBuffers(dev,&cbai,&cb)!=VK_SUCCESS) goto out;
  VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cb,&bi);
  /* Host hat geschrieben -> Shader liest */
  VkMemoryBarrier h2s={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
  vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&h2s,0,0,0,0);
  VkMemoryBarrier s2s={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
  for(uint32_t s=0;s<nstage;s++){
    if(s) vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&s2s,0,0,0,0);
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,st[s].pc->pipe);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,st[s].pc->pll,0,1,&dset[s],0,0);
    vkCmdDispatch(cb,st[s].gx,st[s].gy,st[s].gz);
  }
  /* Shader hat geschrieben -> Host liest */
  VkMemoryBarrier s2h={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
  vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&s2h,0,0,0,0);
  vkEndCommandBuffer(cb);
  VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if(vkCreateFence(dev,&fci,0,&fence)!=VK_SUCCESS) goto out;
  VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
  if(vkQueueSubmit(queue,1,&si,fence)!=VK_SUCCESS) goto out;
  if(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX)!=VK_SUCCESS) goto out;
  rc=0;
out:
  if(fence) vkDestroyFence(dev,fence,0);
  if(cb) vkFreeCommandBuffers(dev,pool,1,&cb);
  if(dpool) vkDestroyDescriptorPool(dev,dpool,0);
  return rc;
}

/* ---- (A) v1 Einzel-Request --------------------------------------------------- */
static void serve_v1(int c, uint32_t* hdr, int* fds, int nfd){
  uint32_t status=1;
  uint32_t spirv_len=hdr[1],gx=hdr[2],gy=hdr[3],gz=hdr[4],nbuf=hdr[5];
  if(nbuf==0||nbuf>MAXBUF||(int)nbuf!=nfd||spirv_len==0||spirv_len>MAXSPIRV||(spirv_len&3)||gx>65535u||gy>65535u||gz>65535u){ wfull(c,&status,4); return; }
  uint32_t bsize[MAXBUF];
  if(rfull(c,bsize,nbuf*4)){ wfull(c,&status,4); return; }
  uint32_t* spirv=malloc(spirv_len);
  if(!spirv||rfull(c,spirv,spirv_len)){ free(spirv); wfull(c,&status,4); return; }
  Stage st; memset(&st,0,sizeof st); st.gx=gx; st.gy=gy; st.gz=gz; st.nbuf=nbuf;
  VkDeviceMemory mem[MAXBUF]={0}; int ok=1;
  VK_LOCK();
  for(uint32_t i=0;i<nbuf;i++) if(import_buf(fds[i],bsize[i],&st.buf[i],&mem[i])){ LOG("import buf %u fail\n",i); ok=0; break; }
  if(ok){ st.pc=get_pipe(spirv,spirv_len,nbuf); if(!st.pc) ok=0; }
  if(ok){ LOG("[gpud-zc] v1 req nbuf=%u cache hits=%ld miss=%ld\n",nbuf,g_pc_hits,g_pc_miss); if(run_stages(&st,1)==0) status=0; }
  unpin_all();
  for(uint32_t i=0;i<nbuf;i++){ if(st.buf[i]) vkDestroyBuffer(dev,st.buf[i],0); if(mem[i]) vkFreeMemory(dev,mem[i],0); }
  VK_UNLOCK();
  wfull(c,&status,4);
  free(spirv);
}

/* ---- (B) v2 Session ---------------------------------------------------------- */
typedef struct { int used; VkBuffer b; VkDeviceMemory m; uint32_t size; } HBuf;
static void serve_v2(int c, uint32_t* hdr, int* fds, int nfd){
  HBuf h[MAXH]; memset(h,0,sizeof h);
  long ndisp=0;
  for(;;){
    uint32_t cmd=hdr[1], status=1;
    if(cmd==1){                                   /* IMPORT */
      uint32_t nbuf=hdr[2]; uint32_t bsize[MAXBUF], hd[MAXBUF];
      if(nbuf==0||nbuf>MAXBUF||(int)nbuf!=nfd||rfull(c,bsize,nbuf*4)){ wfull(c,&status,4); goto next; }
      int ok=1;
      VK_LOCK();
      for(uint32_t i=0;i<nbuf;i++){
        int slot=-1; for(int k=0;k<MAXH;k++) if(!h[k].used){slot=k;break;}
        if(slot<0||import_buf(fds[i],bsize[i],&h[slot].b,&h[slot].m)){ ok=0; LOG("[gpud-zc] import fail (slot %d)\n",slot); break; }
        h[slot].used=1; h[slot].size=bsize[i]; hd[i]=(uint32_t)slot;
      }
      VK_UNLOCK();
      if(ok){ status=0; wfull(c,&status,4); wfull(c,hd,nbuf*4); } else wfull(c,&status,4);
    } else if(cmd==2){                            /* DISPATCH nstage */
      uint32_t nstage=hdr[2]; Stage st[MAXSTAGE]; uint32_t* spv[MAXSTAGE]; uint32_t slens[MAXSTAGE]; memset(spv,0,sizeof spv);
      int ok=(nstage>=1&&nstage<=MAXSTAGE);
      for(uint32_t s=0; ok&&s<nstage; s++){
        uint32_t sh[5]; if(rfull(c,sh,20)){ ok=0; break; }
        uint32_t slen=sh[0]; memset(&st[s],0,sizeof st[s]); st[s].gx=sh[1]; st[s].gy=sh[2]; st[s].gz=sh[3]; st[s].nbuf=sh[4];
        if(st[s].nbuf==0||st[s].nbuf>MAXBUF||slen==0||slen>MAXSPIRV||(slen&3)||sh[1]>65535u||sh[2]>65535u||sh[3]>65535u){ ok=0; break; }
        uint32_t hd[MAXBUF]; if(rfull(c,hd,st[s].nbuf*4)){ ok=0; break; }
        for(uint32_t i=0;i<st[s].nbuf;i++){ if(hd[i]>=MAXH||!h[hd[i]].used){ ok=0; break; } st[s].buf[i]=h[hd[i]].b; }
        if(!ok) break;
        spv[s]=malloc(slen); if(!spv[s]||rfull(c,spv[s],slen)){ ok=0; break; }
        slens[s]=slen;
      }
      if(ok){ VK_LOCK();      /* erst alles gelesen, dann GPU-Arbeit unter dem Mutex */
        for(uint32_t s=0;s<nstage;s++){ st[s].pc=get_pipe(spv[s],slens[s],st[s].nbuf); if(!st[s].pc){ ok=0; break; } }
        if(ok&&run_stages(st,nstage)==0) status=0;
        unpin_all();      /* in dieser Dispatch gepinnte Cache-Eintraege freigeben */
        VK_UNLOCK(); }
      ndisp++;
      if(ndisp<=3||ndisp%1000==0) LOG("[gpud-zc] v2 dispatch #%ld nstage=%u status=%u cache hits=%ld miss=%ld\n",ndisp,nstage,status,g_pc_hits,g_pc_miss);
      for(uint32_t s=0;s<MAXSTAGE;s++) free(spv[s]);
      wfull(c,&status,4);
    } else if(cmd==3){                            /* RELEASE */
      uint32_t nbuf=hdr[2]; uint32_t hd[MAXBUF];
      if(nbuf==0||nbuf>MAXBUF||rfull(c,hd,nbuf*4)){ wfull(c,&status,4); goto next; }
      VK_LOCK(); vkQueueWaitIdle(queue);
      for(uint32_t i=0;i<nbuf;i++) if(hd[i]<MAXH&&h[hd[i]].used){ vkDestroyBuffer(dev,h[hd[i]].b,0); vkFreeMemory(dev,h[hd[i]].m,0); memset(&h[hd[i]],0,sizeof h[0]); }
      VK_UNLOCK();
      status=0; wfull(c,&status,4);
    } else { wfull(c,&status,4); }
next:
    for(int i=0;i<nfd;i++) close(fds[i]); nfd=0;
    /* naechster Befehl (mit evtl. neuen fds) */
    { char cbuf[CMSG_SPACE(sizeof(int)*MAXBUF)];
      struct iovec iov={.iov_base=hdr,.iov_len=24};
      struct msghdr msg={.msg_iov=&iov,.msg_iovlen=1,.msg_control=cbuf,.msg_controllen=sizeof cbuf};
      if(recvmsg(c,&msg,MSG_WAITALL)!=24) break;
      struct cmsghdr* cm=CMSG_FIRSTHDR(&msg);
      if(cm&&cm->cmsg_type==SCM_RIGHTS){ nfd=(int)((cm->cmsg_len-CMSG_LEN(0))/sizeof(int)); if(nfd>MAXBUF){ for(int i=0;i<nfd;i++) close(((int*)CMSG_DATA(cm))[i]); nfd=0; break; } memcpy(fds,CMSG_DATA(cm),nfd*sizeof(int)); }
      if(hdr[0]!=MAGIC2) break; }
  }
  for(int i=0;i<nfd;i++) close(fds[i]);
  VK_LOCK(); vkQueueWaitIdle(queue);
  int live=0; for(int k=0;k<MAXH;k++) if(h[k].used){ live++; vkDestroyBuffer(dev,h[k].b,0); vkFreeMemory(dev,h[k].m,0); }
  VK_UNLOCK();
  LOG("[gpud-zc] session ende: %ld dispatches, %d handles freigegeben\n",ndisp,live);
}

/* ---- (C) GPZ3 0x47505A33 — SESSION v3 fuer ggml-gpud (30.8.2026) --------------------
 * Wie v2 (IMPORT=1, RELEASE=3), zusaetzlich:
 *   cmd=4 LOAD    : [.,4,slen,nbind,pcsize,0]; dann spirv[slen]
 *                   -> u32 status, u32 shader-handle (Pipeline mit nbind Storage-Bindings
 *                      und Push-Constant-Bereich pcsize Bytes; lebt bis Session-Ende)
 *   cmd=5 DISPATCH: [.,5,nstage,0,0,0]; je Stufe: u32 sh,gx,gy,gz,nbind,pcsize;
 *                   nbind*{u32 handle, u32 off_lo, u32 off_hi, u32 range (0=Rest)};
 *                   pcsize Bytes Push-Constants          -> u32 status
 *                   (EIN Command-Buffer, Barriere zwischen Stufen, ein Fence;
 *                    Descriptor-Sets mit Offset/Range je Binding)
 *   cmd=6 UNLOAD  : [.,6,sh,0,0,0] -> u32 status
 * Grenzen: MAXSH3 Shader je Session, MAXSTAGE3 Stufen je Dispatch, pcsize <= 128. */
#define MAGIC3 0x47505A33u
#define MAXSH3 512
#define MAXSTAGE3 4096
#define MAXPC3 128
typedef struct { int used; uint32_t nbind, pcsize; VkShaderModule shader; VkDescriptorSetLayout dsl; VkPipelineLayout pll; VkPipeline pipe; } Sh3;
typedef struct { uint32_t sh,gx,gy,gz,nbind,pcsize; uint32_t hd[MAXBUF]; VkDeviceSize off[MAXBUF], range[MAXBUF]; unsigned char pc[MAXPC3]; } Stage3;
static void sh3_free(Sh3* e){
  if(e->pipe)vkDestroyPipeline(dev,e->pipe,0); if(e->pll)vkDestroyPipelineLayout(dev,e->pll,0);
  if(e->dsl)vkDestroyDescriptorSetLayout(dev,e->dsl,0); if(e->shader)vkDestroyShaderModule(dev,e->shader,0);
  memset(e,0,sizeof *e);
}
static int sh3_load(Sh3* e, const uint32_t* spirv, uint32_t slen, uint32_t nbind, uint32_t pcsize){
  memset(e,0,sizeof *e);
  VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=slen,.pCode=spirv};
  if(vkCreateShaderModule(dev,&smci,0,&e->shader)!=VK_SUCCESS) return -1;
  VkDescriptorSetLayoutBinding lb[MAXBUF];
  for(uint32_t i=0;i<nbind;i++) lb[i]=(VkDescriptorSetLayoutBinding){.binding=i,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=nbind,.pBindings=lb};
  if(vkCreateDescriptorSetLayout(dev,&dlci,0,&e->dsl)!=VK_SUCCESS){ sh3_free(e); return -1; }
  VkPushConstantRange pcr={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=pcsize};
  VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&e->dsl,
    .pushConstantRangeCount=pcsize?1:0,.pPushConstantRanges=pcsize?&pcr:0};
  if(vkCreatePipelineLayout(dev,&plci,0,&e->pll)!=VK_SUCCESS){ sh3_free(e); return -1; }
  VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=e->shader,.pName="main"},.layout=e->pll};
  if(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,0,&e->pipe)!=VK_SUCCESS){ sh3_free(e); return -1; }
  e->used=1; e->nbind=nbind; e->pcsize=pcsize;
  return 0;
}
/* K Stufen in EINEM Command-Buffer, Descriptor-Sets mit Offset/Range, Push-Constants je Stufe */
static double t_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
static double g_t3[3];   /* ms: Record (inkl. Descriptor), Submit+Wait, gesamt (letzter Dispatch) */
/* Descriptor-Pool je Session: einmal gross angelegt, je Dispatch zurueckgesetzt; alle Sets in EINEM
 * vkAllocateDescriptorSets, alle Bindings in EINEM vkUpdateDescriptorSets. */
typedef struct { VkDescriptorPool pool; uint32_t maxsets, maxdesc; } DPool3;
static int dpool3_ensure(DPool3* dp, uint32_t nsets, uint32_t ndesc){
  if(dp->pool && nsets<=dp->maxsets && ndesc<=dp->maxdesc){ vkResetDescriptorPool(dev,dp->pool,0); return 0; }
  if(dp->pool){ vkDestroyDescriptorPool(dev,dp->pool,0); dp->pool=0; }
  uint32_t ms=nsets<256?256:nsets, md=ndesc<1024?1024:ndesc;
  VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=md};
  VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=ms,.poolSizeCount=1,.pPoolSizes=&ps};
  if(vkCreateDescriptorPool(dev,&dpci,0,&dp->pool)!=VK_SUCCESS){ dp->pool=0; return -1; }
  dp->maxsets=ms; dp->maxdesc=md; return 0;
}
/* K Stufen in EINEM Command-Buffer, Descriptor-Sets aus dem Session-Pool, Push-Constants je Stufe,
 * Barriere zwischen den Stufen (flags&1: keine Zwischen-Barrieren - NUR fuer Messungen), ein Fence. */
static int run_stages3(DPool3* dp, Sh3* sh, HBuf* h, Stage3* st, uint32_t nstage, uint32_t flags){
  double t0=t_now(),t1=0;
  VkCommandBuffer cb=0; VkFence fence=0; int rc=-1;
  uint32_t total=0; for(uint32_t s=0;s<nstage;s++) total+=st[s].nbind;
  VkDescriptorSet* dset=calloc(nstage,sizeof(VkDescriptorSet)); VkDescriptorSetLayout* lay=calloc(nstage,sizeof(VkDescriptorSetLayout));
  VkDescriptorBufferInfo* dbi=calloc(total?total:1,sizeof(VkDescriptorBufferInfo)); VkWriteDescriptorSet* wr=calloc(total?total:1,sizeof(VkWriteDescriptorSet));
  if(!dset||!lay||!dbi||!wr) goto out;
  if(dpool3_ensure(dp,nstage,total)) goto out;
  for(uint32_t s=0;s<nstage;s++) lay[s]=sh[st[s].sh].dsl;
  { VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp->pool,.descriptorSetCount=nstage,.pSetLayouts=lay};
    if(vkAllocateDescriptorSets(dev,&dsai,dset)!=VK_SUCCESS) goto out; }
  { uint32_t k=0; for(uint32_t s=0;s<nstage;s++) for(uint32_t i=0;i<st[s].nbind;i++,k++){
      dbi[k]=(VkDescriptorBufferInfo){.buffer=h[st[s].hd[i]].b,.offset=st[s].off[i],.range=st[s].range[i]?st[s].range[i]:VK_WHOLE_SIZE};
      wr[k]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=dset[s],.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbi[k]}; }
    if(k) vkUpdateDescriptorSets(dev,k,wr,0,0); }
  VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
  if(vkAllocateCommandBuffers(dev,&cbai,&cb)!=VK_SUCCESS) goto out;
  VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cb,&bi);
  VkMemoryBarrier h2s={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
  vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&h2s,0,0,0,0);
  VkMemoryBarrier s2s={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
  for(uint32_t s=0;s<nstage;s++){
    Sh3* e=&sh[st[s].sh];
    if(s&&!(flags&1)) vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&s2s,0,0,0,0);
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,e->pipe);
    if(st[s].nbind) vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,e->pll,0,1,&dset[s],0,0);
    if(st[s].pcsize) vkCmdPushConstants(cb,e->pll,VK_SHADER_STAGE_COMPUTE_BIT,0,st[s].pcsize,st[s].pc);
    vkCmdDispatch(cb,st[s].gx,st[s].gy,st[s].gz);
  }
  VkMemoryBarrier s2h={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
  vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&s2h,0,0,0,0);
  vkEndCommandBuffer(cb);
  t1=t_now();
  VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if(vkCreateFence(dev,&fci,0,&fence)!=VK_SUCCESS) goto out;
  VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
  if(vkQueueSubmit(queue,1,&si,fence)!=VK_SUCCESS) goto out;
  if(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX)!=VK_SUCCESS) goto out;
  rc=0;
out:
  { double t2=t_now(); g_t3[0]=t1?t1-t0:0; g_t3[1]=t1?t2-t1:0; g_t3[2]=t2-t0; }
  if(fence) vkDestroyFence(dev,fence,0);
  if(cb) vkFreeCommandBuffers(dev,pool,1,&cb);
  free(dset); free(lay); free(dbi); free(wr);
  return rc;
}
static void serve_v3(int c, uint32_t* hdr, int* fds, int nfd){
  HBuf h[MAXH]; memset(h,0,sizeof h);
  Sh3* sh=calloc(MAXSH3,sizeof(Sh3)); if(!sh) return;
  Stage3* st=calloc(MAXSTAGE3,sizeof(Stage3)); if(!st){ free(sh); return; }
  DPool3 dp; memset(&dp,0,sizeof dp); uint32_t sflags=0;
  long ndisp=0;
  for(;;){
    uint32_t cmd=hdr[1], status=1;
    if(cmd==1){                                   /* IMPORT (wie v2) */
      uint32_t nbuf=hdr[2]; uint32_t bsize[MAXBUF], hd[MAXBUF];
      if(nbuf==0||nbuf>MAXBUF||(int)nbuf!=nfd||rfull(c,bsize,nbuf*4)){ wfull(c,&status,4); goto next; }
      int ok=1;
      VK_LOCK();
      for(uint32_t i=0;i<nbuf;i++){
        int slot=-1; for(int k=0;k<MAXH;k++) if(!h[k].used){slot=k;break;}
        if(slot<0||import_buf(fds[i],bsize[i],&h[slot].b,&h[slot].m)){ ok=0; LOG("[gpud-zc] v3 import fail (slot %d, %u B)\n",slot,bsize[i]); break; }
        h[slot].used=1; h[slot].size=bsize[i]; hd[i]=(uint32_t)slot;
      }
      VK_UNLOCK();
      if(ok){ status=0; wfull(c,&status,4); wfull(c,hd,nbuf*4); } else wfull(c,&status,4);
    } else if(cmd==4){                            /* LOAD shader */
      uint32_t slen=hdr[2], nbind=hdr[3], pcsize=hdr[4]; uint32_t* spv=0; uint32_t shh=0;
      int ok=(slen&&slen<=MAXSPIRV&&!(slen&3)&&nbind<=MAXBUF&&pcsize<=MAXPC3&&!(pcsize&3));
      if(ok){ spv=malloc(slen); if(!spv||rfull(c,spv,slen)) ok=0; }
      if(!ok){ free(spv); wfull(c,&status,4); goto next; }
      int slot=-1; for(int k=0;k<MAXSH3;k++) if(!sh[k].used){slot=k;break;}
      if(slot<0){ LOG("[gpud-zc] v3: Shader-Tabelle voll (%d)\n",MAXSH3); free(spv); wfull(c,&status,4); goto next; }
      VK_LOCK(); int r=sh3_load(&sh[slot],spv,slen,nbind,pcsize); VK_UNLOCK();
      free(spv);
      if(r==0){ status=0; shh=(uint32_t)slot; wfull(c,&status,4); wfull(c,&shh,4); } else wfull(c,&status,4);
    } else if(cmd==5){                            /* DISPATCH nstage */
      uint32_t nstage=hdr[2]; int ok=(nstage>=1&&nstage<=MAXSTAGE3);
      for(uint32_t s=0; ok&&s<nstage; s++){
        uint32_t sx[6]; if(rfull(c,sx,24)){ ok=0; break; }
        Stage3* S=&st[s]; S->sh=sx[0]; S->gx=sx[1]; S->gy=sx[2]; S->gz=sx[3]; S->nbind=sx[4]; S->pcsize=sx[5];
        if(S->sh>=MAXSH3||!sh[S->sh].used||S->nbind>MAXBUF||S->nbind!=sh[S->sh].nbind||S->pcsize!=sh[S->sh].pcsize||sx[1]>65535u||sx[2]>65535u||sx[3]>65535u){ ok=0; break; }
        uint32_t bd[MAXBUF*4]; if(S->nbind&&rfull(c,bd,S->nbind*16)){ ok=0; break; }
        for(uint32_t i=0;i<S->nbind;i++){
          uint32_t hd=bd[i*4]; VkDeviceSize off=((VkDeviceSize)bd[i*4+2]<<32)|bd[i*4+1]; VkDeviceSize rg=bd[i*4+3];
          if(hd>=MAXH||!h[hd].used||off>h[hd].size||(off%g_sbo_align)||(rg&&off+rg>h[hd].size)){ ok=0; break; }
          S->hd[i]=hd; S->off[i]=off; S->range[i]=rg; }
        if(!ok) break;
        if(S->pcsize&&rfull(c,S->pc,S->pcsize)){ ok=0; break; }
      }
      if(ok){ VK_LOCK(); if(run_stages3(&dp,sh,h,st,nstage,sflags)==0) status=0; VK_UNLOCK(); }
      ndisp++;
      if(ndisp<=3||ndisp%1000==0||nstage>=256) LOG("[gpud-zc] v3 dispatch #%ld nstage=%u status=%u  record %.2f ms, gpu %.2f ms, gesamt %.2f ms\n",ndisp,nstage,status,g_t3[0],g_t3[1],g_t3[2]);
      wfull(c,&status,4);
    } else if(cmd==8){                            /* FLAGS (Messung): bit0 = keine Zwischen-Barrieren */
      sflags=hdr[2]; status=0; wfull(c,&status,4);
    } else if(cmd==7){                            /* EXIT: Daemon beendet sich, Supervisor respawnt (Dev-Loop ohne Reboot) */
      status=0; wfull(c,&status,4); LOG("[gpud-zc] EXIT auf Wunsch der Session (Dev-Loop)\n"); VK_LOCK(); vkDeviceWaitIdle(dev); _exit(0);
    } else if(cmd==6){                            /* UNLOAD shader */
      uint32_t shh=hdr[2];
      if(shh<MAXSH3&&sh[shh].used){ VK_LOCK(); vkQueueWaitIdle(queue); sh3_free(&sh[shh]); VK_UNLOCK(); status=0; }
      wfull(c,&status,4);
    } else if(cmd==3){                            /* RELEASE (wie v2) */
      uint32_t nbuf=hdr[2]; uint32_t hd[MAXBUF];
      if(nbuf==0||nbuf>MAXBUF||rfull(c,hd,nbuf*4)){ wfull(c,&status,4); goto next; }
      VK_LOCK(); vkQueueWaitIdle(queue);
      for(uint32_t i=0;i<nbuf;i++) if(hd[i]<MAXH&&h[hd[i]].used){ vkDestroyBuffer(dev,h[hd[i]].b,0); vkFreeMemory(dev,h[hd[i]].m,0); memset(&h[hd[i]],0,sizeof h[0]); }
      VK_UNLOCK();
      status=0; wfull(c,&status,4);
    } else { wfull(c,&status,4); }
next:
    for(int i=0;i<nfd;i++) close(fds[i]); nfd=0;
    { char cbuf[CMSG_SPACE(sizeof(int)*MAXBUF)];
      struct iovec iov={.iov_base=hdr,.iov_len=24};
      struct msghdr msg={.msg_iov=&iov,.msg_iovlen=1,.msg_control=cbuf,.msg_controllen=sizeof cbuf};
      if(recvmsg(c,&msg,MSG_WAITALL)!=24) break;
      struct cmsghdr* cm=CMSG_FIRSTHDR(&msg);
      if(cm&&cm->cmsg_type==SCM_RIGHTS){ nfd=(int)((cm->cmsg_len-CMSG_LEN(0))/sizeof(int)); if(nfd>MAXBUF){ for(int i=0;i<nfd;i++) close(((int*)CMSG_DATA(cm))[i]); break; } memcpy(fds,CMSG_DATA(cm),sizeof(int)*nfd); }
      if(hdr[0]!=MAGIC3) break; }
  }
  for(int i=0;i<nfd;i++) close(fds[i]);
  VK_LOCK(); vkQueueWaitIdle(queue);
  int live=0,lsh=0; for(int k=0;k<MAXH;k++) if(h[k].used){ live++; vkDestroyBuffer(dev,h[k].b,0); vkFreeMemory(dev,h[k].m,0); }
  for(int k=0;k<MAXSH3;k++) if(sh[k].used){ lsh++; sh3_free(&sh[k]); }
  if(dp.pool) vkDestroyDescriptorPool(dev,dp.pool,0);
  VK_UNLOCK();
  free(sh); free(st);
  LOG("[gpud-zc] v3 session ende: %ld dispatches, %d handles, %d shader freigegeben\n",ndisp,live,lsh);
}
static void serve(int c){
  uint32_t hdr[6]; int fds[MAXBUF]; int nfd=0;
  char cbuf[CMSG_SPACE(sizeof(int)*MAXBUF)];
  struct iovec iov={.iov_base=hdr,.iov_len=sizeof hdr};
  struct msghdr msg={.msg_iov=&iov,.msg_iovlen=1,.msg_control=cbuf,.msg_controllen=sizeof cbuf};
  if(recvmsg(c,&msg,MSG_WAITALL)!=(ssize_t)sizeof hdr) return;
  struct cmsghdr* cm=CMSG_FIRSTHDR(&msg);
  if(cm && cm->cmsg_type==SCM_RIGHTS){ nfd=(int)((cm->cmsg_len-CMSG_LEN(0))/sizeof(int)); if(nfd>MAXBUF){ for(int i=0;i<nfd;i++) close(((int*)CMSG_DATA(cm))[i]); return; } memcpy(fds,CMSG_DATA(cm),nfd*sizeof(int)); }
  if(hdr[0]==MAGIC1){ serve_v1(c,hdr,fds,nfd); for(int i=0;i<nfd;i++) close(fds[i]); }
  else if(hdr[0]==MAGIC2){ serve_v2(c,hdr,fds,nfd); }
  else if(hdr[0]==MAGIC3){ serve_v3(c,hdr,fds,nfd); }
  else { for(int i=0;i<nfd;i++) close(fds[i]); }
}

static void* conn_thread(void* a){ int c=*(int*)a; free(a); serve(c); close(c); return 0; }

int main(int argc,char**argv){
  const char* sock=argc>1?argv[1]:"/opt/hwbridge/gpuzc.sock";
  signal(SIGPIPE,SIG_IGN);
  if(gpu_init()) return 1;
  int srv=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof a.sun_path-1);
  unlink(sock);
  if(bind(srv,(struct sockaddr*)&a,sizeof a)<0){ LOG("bind %s: %s\n",sock,strerror(errno)); return 1; }
  chmod(sock,0666); listen(srv,8);
  LOG("[gpud-zc] lauscht auf %s (Zero-Copy v2: GPZC einzel + GPZ2 sessions)\n",sock);
  for(;;){ int c=accept(srv,0,0); if(c<0)continue;
    pthread_t th; int* pc=malloc(sizeof(int)); *pc=c;
    if(pthread_create(&th,0,conn_thread,pc)==0) pthread_detach(th); else { serve(c); close(c); free(pc); } }
  return 0;
}

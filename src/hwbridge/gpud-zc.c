/* gpud-zc v2 — GPU-Compute-Daemon mit ZERO-COPY-Transport (Mali-Vulkan).
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
  const char* exts[]={VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
  float prio=1.0f;
  VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qfam,.queueCount=1,.pQueuePriorities=&prio};
  VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=3,.ppEnabledExtensionNames=exts};
  CK(vkCreateDevice(phys,&dci,0,&dev),"CreateDevice");
  vkGetDeviceQueue(dev,qfam,0,&queue);
  pGetMemFdProps=(PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(dev,"vkGetMemoryFdPropertiesKHR");
  if(!pGetMemFdProps){ LOG("no GetMemoryFdPropertiesKHR\n"); return -1; }
  VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qfam,.flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT};
  CK(vkCreateCommandPool(dev,&pci,0,&pool),"CmdPool");
  LOG("[gpud-zc] Vulkan bereit (external-memory, v2 sessions)\n");
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
  if(nbuf==0||nbuf>MAXBUF||(int)nbuf!=nfd||spirv_len==0||spirv_len>MAXSPIRV||(spirv_len&3)){ wfull(c,&status,4); return; }
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
        if(st[s].nbuf==0||st[s].nbuf>MAXBUF||slen==0||slen>MAXSPIRV||(slen&3)){ ok=0; break; }
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

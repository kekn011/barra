/* gpud - generischer GPU-Compute-Daemon (Bionic, Vulkan) fuer die Mali-G715.
 * Lauscht auf Unix-Socket; Client bringt SPIR-V-Kernel + Buffer, gpud fuehrt aus und gibt Outputs zurueck.
 * glibc-Clients aus dem Ubuntu-Container sprechen ueber den geteilten Socket-Pfad. Bau: -lvulkan (NDK). */
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

#define MAGIC 0x47505531u   /* 'GPU1' - One-Shot (inline Buffer) */
#define MAGIC_CMD 0x47505532u /* 'GPU2' - Command-Modus (persistente Buffer) */
#define CMD_ALLOC 1u  /* u32 size            -> u32 status, u32 handle */
#define CMD_FREE 2u   /* u32 handle          -> u32 status */
#define CMD_UPLOAD 3u /* u32 handle,u32 size,[data] -> u32 status */
#define CMD_DOWNLOAD 4u /* u32 handle,u32 size -> u32 status,[data] */
#define CMD_RUN 5u    /* u32 slen,gx,gy,gz,nbind,handle[nbind],[spirv] -> u32 status */
#define MAXGBUF 256
#define MAXBUF 16
#define MAXSPIRV (16*1024*1024)
#define MAXDATA  (256*1024*1024)
#define F_INPUT  1u
#define F_OUTPUT 2u

static VkInstance g_inst;
static VkPhysicalDevice g_phys;
static VkDevice g_dev;
static VkQueue g_queue;
static uint32_t g_qfam;
static VkCommandPool g_pool;
static uint32_t g_memtype = UINT32_MAX;   /* host-visible+coherent */

#define VKCHK(call,msg) do{ VkResult _r=(call); if(_r!=VK_SUCCESS){ fprintf(stderr,"[gpud] %s -> VkResult %d\n",msg,_r); return -1; } }while(0)

static int gpu_init(void){
  VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="gpud",.apiVersion=VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
  VKCHK(vkCreateInstance(&ici,0,&g_inst),"vkCreateInstance");

  uint32_t n=0; VKCHK(vkEnumeratePhysicalDevices(g_inst,&n,0),"enumPhys count");
  if(n==0){ fprintf(stderr,"[gpud] keine Vulkan-GPU\n"); return -1; }
  VkPhysicalDevice devs[8]; if(n>8)n=8; vkEnumeratePhysicalDevices(g_inst,&n,devs);
  g_phys=devs[0];
  VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(g_phys,&pp);
  fprintf(stderr,"[gpud] GPU: %s (Vulkan %u.%u.%u)\n",pp.deviceName,
    VK_VERSION_MAJOR(pp.apiVersion),VK_VERSION_MINOR(pp.apiVersion),VK_VERSION_PATCH(pp.apiVersion));

  uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(g_phys,&qn,0);
  VkQueueFamilyProperties qf[16]; if(qn>16)qn=16; vkGetPhysicalDeviceQueueFamilyProperties(g_phys,&qn,qf);
  g_qfam=UINT32_MAX;
  for(uint32_t i=0;i<qn;i++) if(qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT){ g_qfam=i; break; }
  if(g_qfam==UINT32_MAX){ fprintf(stderr,"[gpud] keine Compute-Queue\n"); return -1; }

  float prio=1.0f;
  VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=g_qfam,.queueCount=1,.pQueuePriorities=&prio};
  VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
  VKCHK(vkCreateDevice(g_phys,&dci,0,&g_dev),"vkCreateDevice");
  vkGetDeviceQueue(g_dev,g_qfam,0,&g_queue);

  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_phys,&mp);
  uint32_t want=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((mp.memoryTypes[i].propertyFlags&want)==want){ g_memtype=i; break; }
  if(g_memtype==UINT32_MAX){ fprintf(stderr,"[gpud] kein host-visible+coherent Speicher\n"); return -1; }

  VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=g_qfam};
  VKCHK(vkCreateCommandPool(g_dev,&pci,0,&g_pool),"vkCreateCommandPool");
  fprintf(stderr,"[gpud] Vulkan bereit (queue-fam %u, memtype %u)\n",g_qfam,g_memtype);
  return 0;
}

static int read_full(int fd,void*b,long n){char*p=b;while(n>0){ssize_t r=read(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}
static int write_full(int fd,const void*b,long n){const char*p=b;while(n>0){ssize_t w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}
static int ru32(int fd,uint32_t*v){return read_full(fd,v,4);}

/* Pipeline-Cache: Shader-Compile + Pipeline-Erstellung dominieren die ~24ms/Request. Cache nach
 * SPIR-V-Inhalt (+nbuf); wiederholte gleiche Kernels kosten dann nur noch Buffer-Alloc + Dispatch. */
typedef struct { int used; uint64_t hash; uint32_t slen,nbuf;
  VkShaderModule shader; VkDescriptorSetLayout dsl; VkPipelineLayout pll; VkPipeline pipe; } PCacheEnt;
#define PCACHE 16
static PCacheEnt g_pc[PCACHE];
static uint32_t g_pc_next=0;
static long g_pc_hits=0, g_pc_miss=0;

static uint64_t fnv1a(const void* p, uint32_t n){
  const unsigned char* b=p; uint64_t h=1469598103934665603ULL;
  for(uint32_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h;
}
static PCacheEnt* get_pipe(const uint32_t* spirv, uint32_t slen, uint32_t nbuf){
  uint64_t h=fnv1a(spirv,slen);
  for(int i=0;i<PCACHE;i++) if(g_pc[i].used && g_pc[i].hash==h && g_pc[i].slen==slen && g_pc[i].nbuf==nbuf){ g_pc_hits++; return &g_pc[i]; }
  g_pc_miss++;
  int idx=-1; for(int i=0;i<PCACHE;i++) if(!g_pc[i].used){ idx=i; break; }
  if(idx<0){ idx=g_pc_next; g_pc_next=(g_pc_next+1)%PCACHE; PCacheEnt* e=&g_pc[idx];
    if(e->pipe)vkDestroyPipeline(g_dev,e->pipe,0); if(e->pll)vkDestroyPipelineLayout(g_dev,e->pll,0);
    if(e->dsl)vkDestroyDescriptorSetLayout(g_dev,e->dsl,0); if(e->shader)vkDestroyShaderModule(g_dev,e->shader,0);
    memset(e,0,sizeof *e); }
  PCacheEnt* e=&g_pc[idx];
  VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=slen,.pCode=spirv};
  if(vkCreateShaderModule(g_dev,&smci,0,&e->shader)!=VK_SUCCESS) return 0;
  VkDescriptorSetLayoutBinding lb[MAXBUF];
  for(uint32_t i=0;i<nbuf;i++) lb[i]=(VkDescriptorSetLayoutBinding){.binding=i,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=nbuf,.pBindings=lb};
  if(vkCreateDescriptorSetLayout(g_dev,&dlci,0,&e->dsl)!=VK_SUCCESS){ vkDestroyShaderModule(g_dev,e->shader,0); return 0; }
  VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&e->dsl};
  if(vkCreatePipelineLayout(g_dev,&plci,0,&e->pll)!=VK_SUCCESS){ vkDestroyDescriptorSetLayout(g_dev,e->dsl,0); vkDestroyShaderModule(g_dev,e->shader,0); return 0; }
  VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=e->shader,.pName="main"},.layout=e->pll};
  if(vkCreateComputePipelines(g_dev,VK_NULL_HANDLE,1,&cpci,0,&e->pipe)!=VK_SUCCESS){ vkDestroyPipelineLayout(g_dev,e->pll,0); vkDestroyDescriptorSetLayout(g_dev,e->dsl,0); vkDestroyShaderModule(g_dev,e->shader,0); return 0; }
  e->used=1; e->hash=h; e->slen=slen; e->nbuf=nbuf;
  return e;
}

/* eine Compute-Anfrage bedienen. gibt 0 bei sauber behandelt (auch fachlicher Fehler -> status!=0 gesendet). */
static void handle(int c){
  uint32_t magic=0,spirv_len=0,gx=0,gy=0,gz=0,nbuf=0;
  if(ru32(c,&magic)||magic!=MAGIC){ return; }
  if(ru32(c,&spirv_len)||ru32(c,&gx)||ru32(c,&gy)||ru32(c,&gz)||ru32(c,&nbuf)) return;
  if(spirv_len==0||spirv_len>MAXSPIRV||(spirv_len&3)||nbuf==0||nbuf>MAXBUF){ uint32_t s=1; write_full(c,&s,4); return; }
  uint32_t bsize[MAXBUF], bflags[MAXBUF]; long total=0;
  for(uint32_t i=0;i<nbuf;i++){ if(ru32(c,&bsize[i])||ru32(c,&bflags[i])){return;} total+=bsize[i]; if(bsize[i]==0||total>MAXDATA){uint32_t s=2;write_full(c,&s,4);return;} }
  uint32_t* spirv=malloc(spirv_len);
  if(!spirv||read_full(c,spirv,spirv_len)){ free(spirv); return; }

  /* Vulkan-Objekte fuer diese Anfrage (Pipeline kommt aus dem Cache) */
  VkDescriptorPool dpool=0; VkDescriptorSet dset=0; VkCommandBuffer cb=0;
  VkBuffer buf[MAXBUF]={0}; VkDeviceMemory mem[MAXBUF]={0}; void* map[MAXBUF]={0};
  uint32_t status=0;
  #define FAIL(code) do{ status=(code); goto cleanup; }while(0)

  /* Shader+Pipeline gecacht nach SPIR-V (teuerster Teil) */
  PCacheEnt* pc=get_pipe(spirv,spirv_len,nbuf);
  if(!pc) FAIL(10);
  VkDescriptorSetLayout dsl=pc->dsl; VkPipelineLayout pll=pc->pll; VkPipeline pipe=pc->pipe;

  /* Buffer + Speicher, Inputs hochladen */
  for(uint32_t i=0;i<nbuf;i++){
    VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bsize[i],.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    if(vkCreateBuffer(g_dev,&bci,0,&buf[i])!=VK_SUCCESS) FAIL(20);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev,buf[i],&mr);
    VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=g_memtype};
    if(vkAllocateMemory(g_dev,&mai,0,&mem[i])!=VK_SUCCESS) FAIL(21);
    if(vkBindBufferMemory(g_dev,buf[i],mem[i],0)!=VK_SUCCESS) FAIL(22);
    if(vkMapMemory(g_dev,mem[i],0,VK_WHOLE_SIZE,0,&map[i])!=VK_SUCCESS) FAIL(23);
    if(bflags[i]&F_INPUT){ if(read_full(c,map[i],bsize[i])) FAIL(24); }
    else memset(map[i],0,bsize[i]);
  }

  VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=nbuf};
  VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
  if(vkCreateDescriptorPool(g_dev,&dpci,0,&dpool)!=VK_SUCCESS) FAIL(30);
  VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dpool,.descriptorSetCount=1,.pSetLayouts=&dsl};
  if(vkAllocateDescriptorSets(g_dev,&dsai,&dset)!=VK_SUCCESS) FAIL(31);
  VkDescriptorBufferInfo dbi[MAXBUF]; VkWriteDescriptorSet wr[MAXBUF];
  for(uint32_t i=0;i<nbuf;i++){
    dbi[i]=(VkDescriptorBufferInfo){.buffer=buf[i],.offset=0,.range=VK_WHOLE_SIZE};
    wr[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=dset,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbi[i]};
  }
  vkUpdateDescriptorSets(g_dev,nbuf,wr,0,0);

  VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=g_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
  if(vkAllocateCommandBuffers(g_dev,&cbai,&cb)!=VK_SUCCESS) FAIL(40);
  VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cb,&bi);
  vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
  vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pll,0,1,&dset,0,0);
  vkCmdDispatch(cb,gx,gy,gz);
  vkEndCommandBuffer(cb);
  VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
  if(vkQueueSubmit(g_queue,1,&si,VK_NULL_HANDLE)!=VK_SUCCESS) FAIL(41);
  vkQueueWaitIdle(g_queue);

cleanup:
  /* Antwort: status, dann Output-Buffer in Reihenfolge (coherent -> direkt lesbar) */
  write_full(c,&status,4);
  if(status==0) for(uint32_t i=0;i<nbuf;i++) if(bflags[i]&F_OUTPUT) if(write_full(c,map[i],bsize[i])) break;

  for(uint32_t i=0;i<nbuf;i++){ if(map[i]) vkUnmapMemory(g_dev,mem[i]); if(buf[i]) vkDestroyBuffer(g_dev,buf[i],0); if(mem[i]) vkFreeMemory(g_dev,mem[i],0); }
  if(cb) vkFreeCommandBuffers(g_dev,g_pool,1,&cb);
  if(dpool) vkDestroyDescriptorPool(g_dev,dpool,0);
  /* dsl/pll/pipe/shader gehoeren dem Cache und bleiben stehen */
  (void)dsl; (void)pll; (void)pipe;
  free(spirv);
}

/* ================= Persistente GPU-Buffer (Command-Modus 'GPU2') ================= */
typedef struct { int used; VkBuffer buf; VkDeviceMemory mem; void* map; uint32_t size; } GBuf;
static GBuf g_gb[MAXGBUF];

static int gb_create(uint32_t size){
  int h=-1; for(int i=0;i<MAXGBUF;i++) if(!g_gb[i].used){ h=i; break; }
  if(h<0||size==0) return -1;
  VkBuffer buf=0; VkDeviceMemory mem=0; void* map=0;
  VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=size,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
  if(vkCreateBuffer(g_dev,&bci,0,&buf)!=VK_SUCCESS) return -1;
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev,buf,&mr);
  VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=g_memtype};
  if(vkAllocateMemory(g_dev,&mai,0,&mem)!=VK_SUCCESS){ vkDestroyBuffer(g_dev,buf,0); return -1; }
  if(vkBindBufferMemory(g_dev,buf,mem,0)!=VK_SUCCESS || vkMapMemory(g_dev,mem,0,VK_WHOLE_SIZE,0,&map)!=VK_SUCCESS){ vkFreeMemory(g_dev,mem,0); vkDestroyBuffer(g_dev,buf,0); return -1; }
  g_gb[h]=(GBuf){1,buf,mem,map,size};
  return h;
}
static void gb_destroy(uint32_t h){
  if(h>=MAXGBUF||!g_gb[h].used) return;
  if(g_gb[h].map) vkUnmapMemory(g_dev,g_gb[h].mem);
  vkDestroyBuffer(g_dev,g_gb[h].buf,0); vkFreeMemory(g_dev,g_gb[h].mem,0);
  memset(&g_gb[h],0,sizeof(GBuf));
}

/* CMD_RUN: Kernel auf PERSISTENTEN Buffern (per handle) - keinerlei Datenbewegung ueber den Socket. */
static uint32_t cmd_run(int c){
  uint32_t slen=0,gx=0,gy=0,gz=0,nbind=0;
  if(ru32(c,&slen)||ru32(c,&gx)||ru32(c,&gy)||ru32(c,&gz)||ru32(c,&nbind)) return 100;
  if(slen==0||slen>MAXSPIRV||(slen&3)||nbind==0||nbind>MAXBUF) return 101;
  uint32_t hnd[MAXBUF];
  for(uint32_t i=0;i<nbind;i++){ if(ru32(c,&hnd[i])) return 102; if(hnd[i]>=MAXGBUF||!g_gb[hnd[i]].used) return 103; }
  uint32_t* spirv=malloc(slen);
  if(!spirv||read_full(c,spirv,slen)){ free(spirv); return 104; }
  PCacheEnt* pc=get_pipe(spirv,slen,nbind); free(spirv);
  if(!pc) return 105;
  VkDescriptorPool dpool=0; VkDescriptorSet dset=0; VkCommandBuffer cb=0; uint32_t st=0;
  VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=nbind};
  VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
  if(vkCreateDescriptorPool(g_dev,&dpci,0,&dpool)!=VK_SUCCESS){ st=106; goto done; }
  VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dpool,.descriptorSetCount=1,.pSetLayouts=&pc->dsl};
  if(vkAllocateDescriptorSets(g_dev,&dsai,&dset)!=VK_SUCCESS){ st=107; goto done; }
  VkDescriptorBufferInfo dbi[MAXBUF]; VkWriteDescriptorSet wr[MAXBUF];
  for(uint32_t i=0;i<nbind;i++){
    dbi[i]=(VkDescriptorBufferInfo){.buffer=g_gb[hnd[i]].buf,.offset=0,.range=VK_WHOLE_SIZE};
    wr[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=dset,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbi[i]};
  }
  vkUpdateDescriptorSets(g_dev,nbind,wr,0,0);
  VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=g_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
  if(vkAllocateCommandBuffers(g_dev,&cbai,&cb)!=VK_SUCCESS){ st=108; goto done; }
  VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(cb,&bi);
  vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pc->pipe);
  vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pc->pll,0,1,&dset,0,0);
  vkCmdDispatch(cb,gx,gy,gz);
  vkEndCommandBuffer(cb);
  VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
  if(vkQueueSubmit(g_queue,1,&si,VK_NULL_HANDLE)!=VK_SUCCESS){ st=109; goto done; }
  vkQueueWaitIdle(g_queue);
done:
  if(cb) vkFreeCommandBuffers(g_dev,g_pool,1,&cb);
  if(dpool) vkDestroyDescriptorPool(g_dev,dpool,0);
  return st;
}

/* Command-Loop: persistente Buffer leben fuer die Dauer DIESER Verbindung (Daemon serialisiert
 * ohnehin eine Verbindung nach der anderen). ALLOC/UPLOAD einmal, RUN beliebig oft, DOWNLOAD am Ende. */
static void cmd_loop(int c){
  uint32_t magic; if(ru32(c,&magic)) return;   /* MAGIC_CMD (gepeekt) konsumieren */
  for(;;){
    uint32_t cmd; if(ru32(c,&cmd)) break;
    if(cmd==CMD_ALLOC){ uint32_t sz=0; if(ru32(c,&sz)) break; int h=gb_create(sz);
      uint32_t st=(h<0), hh=(h<0)?0:(uint32_t)h; if(write_full(c,&st,4)||write_full(c,&hh,4)) break; }
    else if(cmd==CMD_FREE){ uint32_t h=0; if(ru32(c,&h)) break; gb_destroy(h); uint32_t st=0; if(write_full(c,&st,4)) break; }
    else if(cmd==CMD_UPLOAD){ uint32_t h=0,sz=0; if(ru32(c,&h)||ru32(c,&sz)) break;
      if(h>=MAXGBUF||!g_gb[h].used||sz>g_gb[h].size){ uint32_t st=1; write_full(c,&st,4); break; }
      if(read_full(c,g_gb[h].map,sz)) break; uint32_t st=0; if(write_full(c,&st,4)) break; }
    else if(cmd==CMD_DOWNLOAD){ uint32_t h=0,sz=0; if(ru32(c,&h)||ru32(c,&sz)) break;
      uint32_t st=(h>=MAXGBUF||!g_gb[h].used||sz>g_gb[h].size); if(write_full(c,&st,4)) break;
      if(!st){ if(write_full(c,g_gb[h].map,sz)) break; } }
    else if(cmd==CMD_RUN){ uint32_t st=cmd_run(c); if(write_full(c,&st,4)) break; }
    else break;
  }
  for(int i=0;i<MAXGBUF;i++) gb_destroy(i);   /* Session-Ende: alle Buffer frei */
}

int main(int argc,char**argv){
  const char* sock=argc>1?argv[1]:"/data/local/tmp/gpu.sock";
  signal(SIGPIPE,SIG_IGN);
  if(gpu_init()!=0) return 1;

  int srv=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  unlink(sock);
  if(bind(srv,(struct sockaddr*)&a,sizeof a)<0){ fprintf(stderr,"[gpud] bind %s: %s\n",sock,strerror(errno)); return 1; }
  chmod(sock,0666); listen(srv,8);
  fprintf(stderr,"[gpud] bereit, lauscht auf %s\n",sock);

  fprintf(stderr,"[gpud]   Modi: 'GPU1'=One-Shot (inline), 'GPU2'=Command (persistente Buffer)\n");
  long nreq=0;
  for(;;){ int c=accept(srv,0,0); if(c<0)continue;
    uint32_t m=0; ssize_t pk=recv(c,&m,4,MSG_PEEK);
    if(pk==4 && m==MAGIC_CMD) cmd_loop(c);   /* persistente Buffer, viele Kommandos bis EOF */
    else handle(c);                          /* One-Shot (liest Magic selbst) */
    nreq++; if((nreq%1000)==1) fprintf(stderr,"[gpud] conn #%ld\n",nreq);
    close(c);
  }
  return 0;
}

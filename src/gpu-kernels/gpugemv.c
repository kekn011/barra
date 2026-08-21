// gpugemv — Mali-eigener Decode-GEMV-Bench: y = W x, Gewichts-Streaming GB/s gegen die 36,5-GB/s-Leselatte.
// Nutzung: gpugemv <shader.spv> <fmt 0=f16 1=q40 2=q4k> <M> <K> <ITERS> <RPW rows/workgroup> [xmode 0=keins 1=perm-f16 (Q40H) 2=perm-f16*1024 (Q40S)]
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
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
  if(e==0){ if(m){ float r=ldexpf((float)m,-24); return (h&0x8000)?-r:r; } f=s; }
  else if(e==31){ f=s|0x7f800000|(m<<13); } else { f=s|((uint32_t)(e-15+127)<<23)|(m<<13); }
  float r; memcpy(&r,&f,4); return r; }
static uint16_t f2h(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t sign=(x>>16)&0x8000; int e=((x>>23)&0xff)-127+15; uint32_t man=(x>>13)&0x3ff;
  if(e<=0) return sign; if(e>=31) return sign|0x7c00; return sign|(e<<10)|man; }
static uint32_t rng=12345u; static uint32_t rnd(){ rng=rng*1664525u+1013904223u; return rng; }
static float frnd(){ return (rnd()>>8)*(1.0f/16777216.0f)*2.0f-1.0f; }

int main(int argc,char**argv){
  if(argc<7){ fprintf(stderr,"usage: %s shader.spv fmt(0=f16,1=q40,2=q4k,3=q6k,4=q6kp) M K ITERS RPW\n",argv[0]); return 1; }
  const char*spvpath=argv[1]; int fmt=atoi(argv[2]); uint32_t M=atoi(argv[3]), K=atoi(argv[4]); int ITERS=atoi(argv[5]); uint32_t RPW=atoi(argv[6]); int xmode=argc>7?atoi(argv[7]):0;
  FILE*f=fopen(spvpath,"rb"); if(!f){perror("spv");return 1;} fseek(f,0,SEEK_END); long sl=ftell(f); fseek(f,0,SEEK_SET);
  uint32_t*spv=malloc(sl); if(fread(spv,1,sl,f)!=(size_t)sl){fprintf(stderr,"spv read\n");return 1;} fclose(f);

  VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
  CK(vkCreateInstance(&ici,0,&inst),"inst"); uint32_t n=1; vkEnumeratePhysicalDevices(inst,&n,&phys);
  VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys,&pp);
  VkPhysicalDeviceSubgroupProperties sgp={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 pp2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&sgp}; vkGetPhysicalDeviceProperties2(phys,&pp2);
  if(getenv("GEMV_EXT")){ uint32_t ne=0; vkEnumerateDeviceExtensionProperties(phys,0,&ne,0); VkExtensionProperties*ep=malloc(ne*sizeof*ep); vkEnumerateDeviceExtensionProperties(phys,0,&ne,ep);
    for(uint32_t i=0;i<ne;i++) if(strstr(ep[i].extensionName,"dot")||strstr(ep[i].extensionName,"int8")||strstr(ep[i].extensionName,"float16")||strstr(ep[i].extensionName,"subgroup")||strstr(ep[i].extensionName,"8bit")||strstr(ep[i].extensionName,"16bit")) printf("EXT %s\n",ep[i].extensionName);
    VkPhysicalDeviceShaderIntegerDotProductProperties dpp={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
    VkPhysicalDeviceProperties2 p3={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&dpp}; vkGetPhysicalDeviceProperties2(phys,&p3);
    printf("IDP accel: 8bitU=%u 8bitS=%u 8bitMixed=%u packedU=%u packedS=%u packedMixed=%u accSat8S=%u accSat8Mixed=%u accSatPacked8S=%u accSatPackedMixed=%u 4x8bitPackedAccSatMixed=%u 16bitS=%u\n",
      dpp.integerDotProduct8BitUnsignedAccelerated,dpp.integerDotProduct8BitSignedAccelerated,dpp.integerDotProduct8BitMixedSignednessAccelerated,
      dpp.integerDotProduct4x8BitPackedUnsignedAccelerated,dpp.integerDotProduct4x8BitPackedSignedAccelerated,dpp.integerDotProduct4x8BitPackedMixedSignednessAccelerated,
      dpp.integerDotProductAccumulatingSaturating8BitSignedAccelerated,dpp.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated,
      dpp.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated,dpp.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated,
      dpp.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated,dpp.integerDotProduct16BitSignedAccelerated);
    printf("EXT total %u, apiVersion %u.%u.%u\n",ne,VK_VERSION_MAJOR(pp.apiVersion),VK_VERSION_MINOR(pp.apiVersion),VK_VERSION_PATCH(pp.apiVersion)); }
  uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,0);
  VkQueueFamilyProperties qf[16]; if(qn>16)qn=16; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,qf);
  qfam=UINT32_MAX; for(uint32_t i=0;i<qn;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfam=i;break;}
  float prio=1; VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qfam,.queueCount=1,.pQueuePriorities=&prio};
  VkPhysicalDeviceShaderFloat16Int8Features f16={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,.shaderFloat16=VK_TRUE};
  VkPhysicalDevice16BitStorageFeatures s16={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,.pNext=&f16,.storageBuffer16BitAccess=VK_TRUE};
  VkPhysicalDeviceShaderIntegerDotProductFeatures idp={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,.shaderIntegerDotProduct=VK_TRUE};
  f16.pNext=&idp; const char*exts[]={"VK_KHR_shader_integer_dot_product"};
  VkPhysicalDeviceFeatures feats={.robustBufferAccess=(getenv("GEMV_NOROBUST")?VK_FALSE:VK_TRUE)};
  VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&s16,.enabledExtensionCount=1,.ppEnabledExtensionNames=exts,.pEnabledFeatures=&feats,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
  CK(vkCreateDevice(phys,&dci,0,&dev),"dev"); vkGetDeviceQueue(dev,qfam,0,&queue);
  VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qfam};
  CK(vkCreateCommandPool(dev,&pci,0,&pool),"pool");
  uint32_t mt=pick_mem(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Daten erzeugen; float-Referenz Wf[M][K] aus den quantisierten Werten rekonstruiert
  size_t wbytes=0, sbytes=16;
  if(fmt==0){ wbytes=(size_t)M*K*2; }
  else if(fmt==1){ wbytes=(size_t)M*K/32*16; sbytes=(size_t)M*K/32*2; }
  else if(fmt==2){ wbytes=(size_t)M*K/256*144; }
  else if(fmt==3){ wbytes=(size_t)M*K/256*210; }
  else { wbytes=(size_t)M*K/256*224; }
  size_t wpad = getenv("GEMV_PAD")?atoll(getenv("GEMV_PAD")):4096;   // Polster hinter den Gewichten (OOB-Schutz)
  VkBuffer W,X,Y,S,XH,SX; VkDeviceMemory Wm,Xm,Ym,Sm,XHm,SXm; void *Wmap,*Xmap,*Ymap,*Smap,*XHmap,*SXmap;
  mkbuf(wbytes+wpad,mt,&W,&Wm,&Wmap); mkbuf((size_t)K*4,mt,&X,&Xm,&Xmap); mkbuf((size_t)M*4,mt,&Y,&Ym,&Ymap); mkbuf(sbytes,mt,&S,&Sm,&Smap); mkbuf((size_t)K*2+16,mt,&XH,&XHm,&XHmap); mkbuf((size_t)K/32*8+16,mt,&SX,&SXm,&SXmap);
  float*xp=Xmap; for(uint32_t k=0;k<K;k++) xp[k]=frnd();
  { // xh: f16, fuer q4_0-Standardordnung paarweise permutiert (Wort j von uint i im Block: siehe Shader), optional *1024; sxb = 8*sum(x_block)
    uint16_t*xhp=XHmap; float*sxp=SXmap; float sc=(xmode==2)?1024.0f:1.0f;
    if(xmode==3){ // int8 pro 32er-Block: dx=max|x|/127, x8=round(x/dx); xh = 32 Bytes je Block; sxb[2b]=dx, sxb[2b+1]=8*sum(x8); Referenz-x = dequantisiert
      uint8_t*x8p=XHmap; for(uint32_t b=0;b<K/32;b++){ float*xb=xp+b*32; float mx=0; for(int e=0;e<32;e++) if(fabsf(xb[e])>mx) mx=fabsf(xb[e]);
        float dx=mx/127.0f; int sum=0; for(int e=0;e<32;e++){ int q=(int)lrintf(xb[e]/dx); if(q>127)q=127; if(q<-127)q=-127; x8p[b*32+e]=(uint8_t)(int8_t)q; sum+=q; xb[e]=q*dx; }
        sxp[2*b]=dx; sxp[2*b+1]=(float)sum; }
    } else
    if(xmode){ for(uint32_t k=0;k<K;k++) xp[k]=h2f(f2h(xp[k]*sc))/sc; }   // Referenz-x = f16-gerundet
    if(xmode!=3) for(uint32_t b=0;b<K/32;b++){ const float*xb=xp+b*32; double t=0; for(int e=0;e<32;e++) t+=xb[e]; sxp[b]=(float)(8.0*t);
      for(int i=0;i<4;i++){ int e0[4]={4*i,4*i+1,16+4*i,16+4*i+1}; int e1[4]={4*i+2,4*i+3,16+4*i+2,16+4*i+3};
        for(int j=0;j<4;j++){ uint32_t widx=b*16+i*4+j; if(xmode){ xhp[2*widx]=f2h(xb[e0[j]]*sc); xhp[2*widx+1]=f2h(xb[e1[j]]*sc);} else { xhp[2*widx]=f2h(xb[2*(i*4+j)]); xhp[2*widx+1]=f2h(xb[2*(i*4+j)+1]); } } } }
  }
  float*Wf=malloc((size_t)M*K*4);
  if(fmt==0){ uint16_t*wp=Wmap; for(size_t i=0;i<(size_t)M*K;i++){ float v=frnd()*0.1f; uint16_t h=f2h(v); wp[i]=h; Wf[i]=h2f(h);} }
  else if(fmt==1){ uint8_t*wp=Wmap; uint16_t*sp=Smap; size_t NB=K/32;
    for(uint32_t m=0;m<M;m++) for(size_t b=0;b<NB;b++){ size_t idx=m*NB+b; uint16_t dh=f2h(frnd()*0.01f+0.02f); sp[idx]=dh; float d=h2f(dh);
      uint8_t*q=wp+idx*16; for(int j=0;j<16;j++){ uint32_t lo=rnd()&0xF, hi=rnd()&0xF; q[j]=lo|(hi<<4);
        Wf[(size_t)m*K+b*32+j]=d*((float)lo-8.0f); Wf[(size_t)m*K+b*32+16+j]=d*((float)hi-8.0f);} } }
  else { uint8_t*wp=Wmap; size_t NSB=K/256;
    for(uint32_t m=0;m<M;m++) for(size_t b=0;b<NSB;b++){ uint8_t*blk=wp+(m*NSB+b)*144;
      uint16_t dh=f2h(frnd()*0.002f+0.004f), mh=f2h(frnd()*0.001f+0.002f); memcpy(blk,&dh,2); memcpy(blk+2,&mh,2);
      float d=h2f(dh), dmin=h2f(mh); uint8_t*sc=blk+4; for(int j=0;j<12;j++) sc[j]=rnd()&0xFF; uint8_t*qs=blk+16; for(int j=0;j<128;j++) qs[j]=rnd()&0xFF;
      for(int j=0;j<8;j++){ uint8_t s_,m_; if(j<4){ s_=sc[j]&63; m_=sc[j+4]&63; } else { s_=(sc[j+4]&0xF)|((sc[j-4]>>6)<<4); m_=(sc[j+4]>>4)|((sc[j]>>6)<<4); }
        float d1=d*s_, m1=dmin*m_; int l=j/2; const uint8_t*q=qs+l*32;
        for(int t=0;t<32;t++){ int nib=(j&1)?(q[t]>>4):(q[t]&0xF); Wf[(size_t)m*K+b*256+j*32+t]=d1*nib-m1; } } } }
  if(fmt==3||fmt==4){ uint8_t*wp=Wmap; size_t NSB=K/256; size_t BS=(fmt==3)?210:224;
    for(uint32_t m=0;m<M;m++) for(size_t b=0;b<NSB;b++){ uint8_t*blk=wp+(m*NSB+b)*BS; uint8_t*ql=blk; uint8_t*qh=blk+128; int8_t*sc=(int8_t*)(blk+192);
      for(int t=0;t<128;t++) ql[t]=rnd()&0xFF; for(int t=0;t<64;t++) qh[t]=rnd()&0xFF; for(int t=0;t<16;t++) sc[t]=(int8_t)((rnd()&0x7F)-64);
      uint16_t dh=f2h(frnd()*0.001f+0.002f); memcpy(blk+208,&dh,2); float d=h2f(dh);
      float*y=Wf+(size_t)m*K+b*256;
      for(int n=0;n<256;n+=128){ const uint8_t*qlp=ql+n/2; const uint8_t*qhp=qh+n/4; const int8_t*scp=sc+n/16;
        for(int l=0;l<32;l++){ int is=l/16;
          int q1=(int)((qlp[l]&0xF)|(((qhp[l]>>0)&3)<<4))-32, q2=(int)((qlp[l+32]&0xF)|(((qhp[l]>>2)&3)<<4))-32;
          int q3=(int)((qlp[l]>>4)|(((qhp[l]>>4)&3)<<4))-32, q4=(int)((qlp[l+32]>>4)|(((qhp[l]>>6)&3)<<4))-32;
          y[n+l]=d*scp[is]*q1; y[n+l+32]=d*scp[is+2]*q2; y[n+l+64]=d*scp[is+4]*q3; y[n+l+96]=d*scp[is+6]*q4; } } } }
  float*yref=malloc((size_t)M*4);
  for(uint32_t m=0;m<M;m++){ double a=0; const float*wr=Wf+(size_t)m*K; for(uint32_t k=0;k<K;k++) a+=(double)wr[k]*xp[k]; yref[m]=(float)a; }

  VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sl,.pCode=spv};
  VkShaderModule sm; CK(vkCreateShaderModule(dev,&smci,0,&sm),"shader");
  VkDescriptorSetLayoutBinding b[6]; for(int i=0;i<6;i++){ b[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}; }
  VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=6,.pBindings=b};
  VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev,&dlci,0,&dsl),"dsl");
  VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,8};
  VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
  VkPipelineLayout pll; CK(vkCreatePipelineLayout(dev,&plci,0,&pll),"pll");
  VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},.layout=pll};
  VkPipeline pipe; CK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,0,&pipe),"pipe");
  VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6};
  VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
  VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev,&dpci,0,&dp),"dp");
  VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl};
  VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev,&dsai,&ds),"ds");
  VkDescriptorBufferInfo bi[6]={{W,0,VK_WHOLE_SIZE},{X,0,VK_WHOLE_SIZE},{Y,0,VK_WHOLE_SIZE},{S,0,VK_WHOLE_SIZE},{XH,0,VK_WHOLE_SIZE},{SX,0,VK_WHOLE_SIZE}};
  VkWriteDescriptorSet wr[6]; for(int i=0;i<6;i++) wr[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
  vkUpdateDescriptorSets(dev,6,wr,0,0);
  VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
  VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev,&cbai,&cb),"cb");
  VkCommandBufferBeginInfo bg={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  uint32_t pcv[2]={M,K}; uint32_t gx=(M+RPW-1)/RPW;
  vkBeginCommandBuffer(cb,&bg); vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
  vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pll,0,1,&ds,0,0);
  vkCmdPushConstants(cb,pll,VK_SHADER_STAGE_COMPUTE_BIT,0,8,pcv);
  VkMemoryBarrier mb={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
  for(int it=0;it<ITERS;it++){ vkCmdDispatch(cb,gx,1,1); vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0);}
  vkEndCommandBuffer(cb);
  VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fen; vkCreateFence(dev,&fci,0,&fen);
  VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
  CK(vkQueueSubmit(queue,1,&si,fen),"sw"); vkWaitForFences(dev,1,&fen,VK_TRUE,UINT64_MAX); vkResetFences(dev,1,&fen);
  double best=1e9; for(int rep=0;rep<3;rep++){ double t0=now(); CK(vkQueueSubmit(queue,1,&si,fen),"s"); vkWaitForFences(dev,1,&fen,VK_TRUE,UINT64_MAX); vkResetFences(dev,1,&fen); double dt=(now()-t0)/ITERS; if(dt<best)best=dt; }
  double bytes=(double)wbytes+(fmt==1?(double)sbytes:0);
  float*yp=Ymap; double maxrel=0, maxabs=0; uint32_t bad=0;
  for(uint32_t m=0;m<M;m++){ double e=fabs((double)yp[m]-yref[m]); if(e>maxabs)maxabs=e; double rel=e/(fabs(yref[m])+1e-3); if(rel>maxrel)maxrel=rel; if(rel>2e-2) bad++; }
  const char*fn[]={"f16","q40","q4k","q6k","q6kp"};
  printf("# %s subgroup=%u  GEMV %s M=%u K=%u RPW=%u grid=%u  weights=%.1f MB\n",pp.deviceName,sgp.subgroupSize,fn[fmt],M,K,RPW,gx,bytes/1e6);
  printf("GEMV %s: %.3f ms  %.1f GB/s (Gewichte)   KORREKT: maxabs=%.2e maxrel=%.2e bad=%u/%u  y0=%.5f ref=%.5f\n",fn[fmt],best*1e3,bytes/best/1e9,maxabs,maxrel,bad,M,yp[0],yref[0]);
  return 0;
}

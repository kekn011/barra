/* colband - fuellt 6 waagerechte Baender mit ROHEN 32-bit-Werten (keine Umrechnung),
 * um das echte Panel-Pixelformat zu bestimmen. Haelt <sek> Sekunden, dann Panel aus. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
static void blw(const char* f,const char* v){ char p[256]; FILE* h; snprintf(p,sizeof p,"/sys/class/backlight/panel0-backlight/%s",f); if((h=fopen(p,"w"))){fputs(v,h);fclose(h);} }
int main(int argc,char**argv){
  int sek=argc>1?atoi(argv[1]):20;
  uint32_t vals[6]={0x00000000u,0xFF000000u,0x00FF0000u,0x0000FF00u,0x000000FFu,0xFFFFFFFFu};
  int DFD=open("/dev/dri/card0",O_RDWR|O_CLOEXEC); if(DFD<0){perror("card0");return 1;} ioctl(DFD,DRM_IOCTL_SET_MASTER,0);
  struct drm_mode_card_res res; memset(&res,0,sizeof res); ioctl(DFD,DRM_IOCTL_MODE_GETRESOURCES,&res);
  uint32_t cn[32],cr[32]; res.connector_id_ptr=(uint64_t)(uintptr_t)cn; res.count_connectors=res.count_connectors>32?32:res.count_connectors;
  res.crtc_id_ptr=(uint64_t)(uintptr_t)cr; res.count_crtcs=res.count_crtcs>32?32:res.count_crtcs; res.count_encoders=0; res.count_fbs=0;
  ioctl(DFD,DRM_IOCTL_MODE_GETRESOURCES,&res);
  struct drm_mode_modeinfo mode; uint32_t CONN=0,enc=0; int have=0;
  for(uint32_t i=0;i<res.count_connectors&&!have;i++){ struct drm_mode_get_connector c; memset(&c,0,sizeof c); c.connector_id=cn[i];
    if(ioctl(DFD,DRM_IOCTL_MODE_GETCONNECTOR,&c)<0)continue; if(!c.count_modes||c.connection!=1)continue;
    struct drm_mode_modeinfo ms[64]; c.modes_ptr=(uint64_t)(uintptr_t)ms; c.count_modes=c.count_modes>64?64:c.count_modes; c.count_encoders=0; c.count_props=0;
    if(ioctl(DFD,DRM_IOCTL_MODE_GETCONNECTOR,&c)<0)continue; if(!c.count_modes)continue; mode=ms[0]; CONN=c.connector_id; enc=c.encoder_id; have=1; }
  if(!have){fprintf(stderr,"kein Panel\n");return 1;}
  uint32_t crtc=0; if(enc){struct drm_mode_get_encoder e;memset(&e,0,sizeof e);e.encoder_id=enc;if(ioctl(DFD,DRM_IOCTL_MODE_GETENCODER,&e)==0&&e.crtc_id)crtc=e.crtc_id;}
  if(!crtc&&res.count_crtcs)crtc=cr[0];
  struct drm_mode_create_dumb cq; memset(&cq,0,sizeof cq); cq.width=mode.hdisplay; cq.height=mode.vdisplay; cq.bpp=32; ioctl(DFD,DRM_IOCTL_MODE_CREATE_DUMB,&cq);
  struct drm_mode_fb_cmd fb; memset(&fb,0,sizeof fb); fb.width=cq.width; fb.height=cq.height; fb.pitch=cq.pitch; fb.bpp=32; fb.depth=24; fb.handle=cq.handle; ioctl(DFD,DRM_IOCTL_MODE_ADDFB,&fb);
  struct drm_mode_map_dumb mq; memset(&mq,0,sizeof mq); mq.handle=cq.handle; ioctl(DFD,DRM_IOCTL_MODE_MAP_DUMB,&mq);
  uint8_t* FB=mmap(0,cq.size,PROT_READ|PROT_WRITE,MAP_SHARED,DFD,mq.offset); if(FB==MAP_FAILED)return 1;
  uint32_t Wd=cq.width,Hd=cq.height,P=cq.pitch;
  for(uint32_t y=0;y<Hd;y++){ uint32_t v=vals[(y*6)/Hd]; uint32_t* row=(uint32_t*)(FB+y*P); for(uint32_t x=0;x<Wd;x++)row[x]=v; }
  struct drm_mode_crtc CRT; memset(&CRT,0,sizeof CRT); CRT.crtc_id=crtc; CRT.fb_id=fb.fb_id; CRT.set_connectors_ptr=(uint64_t)(uintptr_t)&CONN; CRT.count_connectors=1; CRT.mode=mode; CRT.mode_valid=1;
  ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&CRT); blw("bl_power","0"); blw("brightness","3000");
  for(int i=0;i<sek;i++){ ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&CRT); sleep(1);} blw("brightness","0"); blw("bl_power","4"); return 0;
}

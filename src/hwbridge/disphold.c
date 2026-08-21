/* disphold - haelt das DSI-Panel an (DRM-Master) und schaltet nach N Sekunden
 * automatisch wieder ab. Platzhalter-Schirm (Dashboard folgt spaeter).
 * Bionic/NDK-Build, laeuft Android-seitig (Framework ist still -> DRM-Master frei).
 *   disphold <timeout_sekunden>   (0 = nie automatisch aus, laeuft bis SIGTERM)
 * Beim Beenden (Timeout ODER SIGTERM) wird das Backlight ausgeschaltet. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>

static const char* BL="/sys/class/backlight/panel0-backlight";
static void bl_write(const char* f,const char* v){ char p[256]; FILE* h; snprintf(p,sizeof p,"%s/%s",BL,f); if((h=fopen(p,"w"))){ fputs(v,h); fclose(h);} }
static void blank_and_exit(int sig){ (void)sig; bl_write("brightness","0"); bl_write("bl_power","4"); _exit(0); }

int main(int argc,char**argv){
  int secs=argc>1?atoi(argv[1]):60;
  signal(SIGTERM,blank_and_exit); signal(SIGINT,blank_and_exit);
  int fd=open("/dev/dri/card0",O_RDWR|O_CLOEXEC);
  if(fd<0){ perror("open card0"); return 1; }
  ioctl(fd,DRM_IOCTL_SET_MASTER,0);

  struct drm_mode_card_res res; memset(&res,0,sizeof res);
  if(ioctl(fd,DRM_IOCTL_MODE_GETRESOURCES,&res)<0){ perror("GETRES cnt"); return 1; }
  uint32_t conns[32],crtcs[32],encs[32],fbs[32];
  res.connector_id_ptr=(uint64_t)(uintptr_t)conns; res.count_connectors=res.count_connectors>32?32:res.count_connectors;
  res.crtc_id_ptr=(uint64_t)(uintptr_t)crtcs;       res.count_crtcs=res.count_crtcs>32?32:res.count_crtcs;
  res.encoder_id_ptr=(uint64_t)(uintptr_t)encs;     res.count_encoders=res.count_encoders>32?32:res.count_encoders;
  res.fb_id_ptr=(uint64_t)(uintptr_t)fbs;           res.count_fbs=res.count_fbs>32?32:res.count_fbs;
  if(ioctl(fd,DRM_IOCTL_MODE_GETRESOURCES,&res)<0){ perror("GETRES"); return 1; }

  struct drm_mode_modeinfo mode; int have=0; uint32_t conn_id=0,enc_id=0;
  for(uint32_t i=0;i<res.count_connectors && !have;i++){
    struct drm_mode_get_connector c; memset(&c,0,sizeof c); c.connector_id=conns[i];
    if(ioctl(fd,DRM_IOCTL_MODE_GETCONNECTOR,&c)<0) continue;
    if(c.count_modes==0 || c.connection!=1) continue;
    struct drm_mode_modeinfo modes[64]; uint32_t me[16],mp[64]; uint64_t mpv[64];
    c.modes_ptr=(uint64_t)(uintptr_t)modes; c.count_modes=c.count_modes>64?64:c.count_modes;
    c.encoders_ptr=(uint64_t)(uintptr_t)me; c.count_encoders=c.count_encoders>16?16:c.count_encoders;
    c.props_ptr=(uint64_t)(uintptr_t)mp; c.prop_values_ptr=(uint64_t)(uintptr_t)mpv; c.count_props=c.count_props>64?64:c.count_props;
    if(ioctl(fd,DRM_IOCTL_MODE_GETCONNECTOR,&c)<0) continue;
    if(c.count_modes==0) continue;
    mode=modes[0]; conn_id=c.connector_id; enc_id=c.encoder_id; have=1;
  }
  if(!have){ fprintf(stderr,"kein Panel\n"); return 1; }
  uint32_t crtc_id=0;
  if(enc_id){ struct drm_mode_get_encoder e; memset(&e,0,sizeof e); e.encoder_id=enc_id;
    if(ioctl(fd,DRM_IOCTL_MODE_GETENCODER,&e)==0 && e.crtc_id) crtc_id=e.crtc_id; }
  if(!crtc_id && res.count_crtcs) crtc_id=crtcs[0];

  struct drm_mode_create_dumb creq; memset(&creq,0,sizeof creq);
  creq.width=mode.hdisplay; creq.height=mode.vdisplay; creq.bpp=32;
  if(ioctl(fd,DRM_IOCTL_MODE_CREATE_DUMB,&creq)<0){ perror("CREATE_DUMB"); return 1; }
  struct drm_mode_fb_cmd fb; memset(&fb,0,sizeof fb);
  fb.width=creq.width; fb.height=creq.height; fb.pitch=creq.pitch; fb.bpp=32; fb.depth=24; fb.handle=creq.handle;
  if(ioctl(fd,DRM_IOCTL_MODE_ADDFB,&fb)<0){ perror("ADDFB"); return 1; }
  struct drm_mode_map_dumb mreq; memset(&mreq,0,sizeof mreq); mreq.handle=creq.handle;
  if(ioctl(fd,DRM_IOCTL_MODE_MAP_DUMB,&mreq)<0){ perror("MAP_DUMB"); return 1; }
  uint8_t* base=mmap(0,creq.size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,mreq.offset);
  if(base==MAP_FAILED){ perror("mmap"); return 1; }
  uint32_t W=creq.width,Hh=creq.height;

  /* Platzhalter: dunkler Grund + kraeftiger Kopf- und Fussbalken (klar 'an'). */
  for(uint32_t y=0;y<Hh;y++){ uint32_t* row=(uint32_t*)(base+y*creq.pitch);
    uint32_t c = (y<160||y>=Hh-160)? 0xFF1E88E5 /*blau*/ : 0xFF101418 /*fast schwarz*/;
    for(uint32_t x=0;x<W;x++) row[x]=c; }

  struct drm_mode_crtc set; memset(&set,0,sizeof set);
  set.crtc_id=crtc_id; set.fb_id=fb.fb_id;
  set.set_connectors_ptr=(uint64_t)(uintptr_t)&conn_id; set.count_connectors=1;
  set.mode=mode; set.mode_valid=1;
  if(ioctl(fd,DRM_IOCTL_MODE_SETCRTC,&set)<0){ perror("SETCRTC"); return 1; }
  bl_write("bl_power","0"); bl_write("brightness","3000");

  if(secs<=0){ for(;;) pause(); }
  else { sleep(secs); }
  blank_and_exit(0);
  return 0;
}

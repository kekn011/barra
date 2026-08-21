/* touchkms - der integrierte Beweis: EIN glibc-Prozess (Ubuntu-Container) wird DRM-Master,
 * treibt das Panel (dann scannt der Goodix-Touch) UND liest gleichzeitig event2.
 * Zeichnet an jeder Beruehrung ein Quadrat -> Finger wird live auf dem Panel verfolgt.
 * Aufruf: touchkms [sekunden]   (basepanel muss gestoppt sein) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <linux/input.h>

static uint8_t* base; static uint32_t W,Hh,pitch4; static int g_fd;
static struct drm_mode_crtc g_set;   /* fuer Re-Latch */
static void relatch(void){ ioctl(g_fd,DRM_IOCTL_MODE_SETCRTC,&g_set); }
static void fillsq(int cx,int cy,int r,uint32_t col){
  for(int y=cy-r;y<=cy+r;y++){ if(y<0||y>=(int)Hh) continue;
    uint32_t* row=(uint32_t*)(base+y*pitch4*4);
    for(int x=cx-r;x<=cx+r;x++){ if(x<0||x>=(int)W) continue; row[x]=col; } }
  relatch();   /* command-mode Panel: ganzen Frame neu uebertragen */
}

int main(int argc,char**argv){
  int secs=argc>1?atoi(argv[1]):20;
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
    printf("Panel %u: %ux%u@%uHz\n",conn_id,mode.hdisplay,mode.vdisplay,mode.vrefresh);
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
  base=mmap(0,creq.size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,mreq.offset);
  if(base==MAP_FAILED){ perror("mmap"); return 1; }
  W=creq.width; Hh=creq.height; pitch4=creq.pitch/4; g_fd=fd;

  /* sattes Gruen = 'Ubuntu besitzt jetzt den Schirm', oben ein weisser Balken */
  for(uint32_t y=0;y<Hh;y++){ uint32_t* row=(uint32_t*)(base+y*creq.pitch);
    for(uint32_t x=0;x<W;x++) row[x]= (y<140)?0xFFFFFFFF:0xFF0A6A2A; }

  struct drm_mode_crtc set; memset(&set,0,sizeof set);
  set.crtc_id=crtc_id; set.fb_id=fb.fb_id;
  set.set_connectors_ptr=(uint64_t)(uintptr_t)&conn_id; set.count_connectors=1;
  set.mode=mode; set.mode_valid=1;
  if(ioctl(fd,DRM_IOCTL_MODE_SETCRTC,&set)<0){ perror("SETCRTC"); return 1; }
  g_set=set;   /* fuer Re-Latch nach jedem Zeichnen merken */
  const char* BL="/sys/class/backlight/panel0-backlight"; char pp[256]; FILE* f;
  snprintf(pp,sizeof pp,"%s/bl_power",BL); if((f=fopen(pp,"w"))){ fputs("0",f); fclose(f); }
  snprintf(pp,sizeof pp,"%s/brightness",BL); if((f=fopen(pp,"w"))){ fputs("3000",f); fclose(f); }
  relatch();   /* command-mode Panel: Grund einmal explizit anstossen */

  /* Touch lesen, waehrend WIR den Master halten (Panel wird getrieben -> Goodix scannt) */
  int tfd=open("/dev/input/event2",O_RDONLY|O_NONBLOCK);
  if(tfd<0){ perror("open event2"); return 1; }
  int grab=1; ioctl(tfd,EVIOCGRAB,&grab);
  printf(">>> Panel steht. %d s: TIPPE - jede Beruehrung malt ein Quadrat <<<\n",secs); fflush(stdout);
  struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
  int x=0,y=0,taps=0; uint32_t palette[6]={0xFFFF4040,0xFF40FF40,0xFF4080FF,0xFFFFFF40,0xFFFF40FF,0xFF40FFFF};
  struct input_event ev;
  for(;;){
    struct timespec now; clock_gettime(CLOCK_MONOTONIC,&now);
    if(now.tv_sec-t0.tv_sec>=secs) break;
    ssize_t n=read(tfd,&ev,sizeof ev);
    if(n!=sizeof ev){ usleep(3000); continue; }
    if(ev.type==EV_ABS){ if(ev.code==0x35) x=ev.value; else if(ev.code==0x36) y=ev.value; }
    else if(ev.type==EV_KEY && ev.code==0x14a && ev.value==1){
      taps++; printf("  TOUCH #%d bei x=%d y=%d\n",taps,x,y); fflush(stdout);
      fillsq(x,y,55,0xFFFFFFFF);   /* weiss = format-unabhaengig eindeutig */
    }
  }
  printf("fertig: %d Beruehrungen.\n",taps);
  return 0;
}

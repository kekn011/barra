/* bootsplash - barra Boot-Splash (Android/Bionic, KMS aufs Command-Mode-DSI-Panel).
 * Zeigt "barra" + Fortschrittsbalken + Status-String, gelesen aus einer Datei, die
 * base-boot je Stufe schreibt:  <fortschritt 0-100>|<text>   z.B. "40|Ubuntu-Userland startet"
 * Laeuft NACH dem Composer-Stop (framework-aus) bis zur Dashboard-Uebergabe.
 * Beendet sich bei SIGTERM (blankt NICHT - dash2 uebernimmt nahtlos) oder wenn Statusfile "100|".
 *   bootsplash <statusfile> [maxsek]
 * Build (NDK): aarch64-linux-android31-clang -O2 bootsplash.c -o bootsplash -lm  (stb_truetype im Includepfad) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

static int DFD; static uint8_t* FB; static uint32_t W,H,PITCH; static struct drm_mode_crtc CRT; static uint32_t CONN;
static volatile int g_stop=0;
static void on_term(int s){ (void)s; g_stop=1; }
static void blw(const char* f,const char* v){ char p[256]; FILE* h; snprintf(p,sizeof p,"/sys/class/backlight/panel0-backlight/%s",f); if((h=fopen(p,"w"))){fputs(v,h);fclose(h);} }

/* Panel-Format [X,R,G,B] (empirisch, siehe dash2) */
static uint32_t col(int r,int g,int b){ return ((uint32_t)b<<24)|((uint32_t)g<<16)|((uint32_t)r<<8); }
static inline void pset(int x,int y,uint32_t c){ if(x<0||y<0||x>=(int)W||y>=(int)H)return; *(uint32_t*)(FB+y*PITCH+x*4)=c; }
static inline void blend(int x,int y,uint32_t c,int a){ if(a<=0||x<0||y<0||x>=(int)W||y>=(int)H)return; if(a>=255){pset(x,y,c);return;}
  uint32_t* d=(uint32_t*)(FB+y*PITCH+x*4); uint32_t o=*d,out=0;
  for(int s=8;s<32;s+=8){ int sc=(c>>s)&255,dc=(o>>s)&255; out|=(uint32_t)((sc*a+dc*(255-a))/255)<<s; } *d=out; }
static void rrect(int x,int y,int w,int h,int rad,uint32_t c){ if(rad>h/2)rad=h/2; if(rad>w/2)rad=w/2;
  for(int j=0;j<h;j++)for(int i=0;i<w;i++){ int dx=-1,dy=-1;
    if(i<rad&&j<rad){dx=rad-i;dy=rad-j;} else if(i>=w-rad&&j<rad){dx=i-(w-rad-1);dy=rad-j;}
    else if(i<rad&&j>=h-rad){dx=rad-i;dy=j-(h-rad-1);} else if(i>=w-rad&&j>=h-rad){dx=i-(w-rad-1);dy=j-(h-rad-1);}
    if(dx>=0){ float d=sqrtf((float)(dx*dx+dy*dy)); if(d>rad+0.5f)continue; if(d>rad-0.5f){ blend(x+i,y+j,c,(int)((rad+0.5f-d)*255)); continue;} }
    pset(x+i,y+j,c); } }
static void bar(int x,int y,int w,int h,float frac,uint32_t fg,uint32_t track){
  if(frac<0)frac=0; if(frac>1)frac=1; rrect(x,y,w,h,h/2,track);
  int fw=(int)(w*frac); if(fw<h&&frac>0.004f)fw=h; if(fw>w)fw=w; if(frac>0.004f) rrect(x,y,fw,h,h/2,fg); }

/* --- Font (stb_truetype, Roboto aus /system/fonts) --- */
static stbtt_fontinfo FONT; static uint8_t* FONTBUF;
static int font_load(void){ const char* c[]={"/system/fonts/Roboto-Regular.ttf","/system/fonts/RobotoStatic-Regular.ttf",0};
  for(int i=0;c[i];i++){ FILE* f=fopen(c[i],"rb"); if(!f)continue; fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    FONTBUF=malloc(n); if(fread(FONTBUF,1,n,f)!=(size_t)n){fclose(f);free(FONTBUF);FONTBUF=0;continue;} fclose(f);
    if(stbtt_InitFont(&FONT,FONTBUF,stbtt_GetFontOffsetForIndex(FONTBUF,0))) return 0; free(FONTBUF); FONTBUF=0; }
  return -1; }
static unsigned utf8(const char** s){ const unsigned char* p=(const unsigned char*)*s; unsigned c=*p;
  if(c<0x80){*s+=1;return c;} if((c>>5)==0x6&&p[1]){*s+=2;return ((c&0x1F)<<6)|(p[1]&0x3F);}
  if((c>>4)==0xE&&p[1]&&p[2]){*s+=3;return ((c&0x0F)<<12)|((p[1]&0x3F)<<6)|(p[2]&0x3F);} *s+=1; return c; }
static int textw(int px,const char* s){ float sc=stbtt_ScaleForPixelHeight(&FONT,px); float w=0;
  while(*s){ unsigned cp=utf8(&s); int adv,lsb; stbtt_GetCodepointHMetrics(&FONT,cp,&adv,&lsb); w+=adv*sc; } return (int)w; }
static void text(int x,int baseline,int px,uint32_t c,const char* s){ float sc=stbtt_ScaleForPixelHeight(&FONT,px); float pen=x;
  while(*s){ unsigned cp=utf8(&s); int adv,lsb,x0,y0,x1,y1; stbtt_GetCodepointHMetrics(&FONT,cp,&adv,&lsb);
    stbtt_GetCodepointBitmapBox(&FONT,cp,sc,sc,&x0,&y0,&x1,&y1); int bw=x1-x0,bh=y1-y0;
    if(bw>0&&bh>0){ uint8_t* bm=malloc(bw*bh); stbtt_MakeCodepointBitmap(&FONT,bm,bw,bh,bw,sc,sc,cp);
      for(int j=0;j<bh;j++)for(int i=0;i<bw;i++){ int a=bm[j*bw+i]; if(a) blend((int)pen+x0+i,baseline+y0+j,c,a); } free(bm); }
    pen+=adv*sc; } }
static void textc(int xc,int baseline,int px,uint32_t c,const char* s){ text(xc-textw(px,s)/2,baseline,px,c,s); }

/* --- Logo (vorskaliertes RGBA-PNG, Layout passend zur Bootanimation) --- */
static unsigned char* LOGO; static int LW,LH;
static void logo_load(void){ int n; LOGO=stbi_load("/data/adb/baseos/logo.png",&LW,&LH,&n,4); }
static void logo_draw(int cx,int cy){ if(!LOGO)return; int x0=cx-LW/2,y0=cy-LH/2;
  for(int j=0;j<LH;j++)for(int i=0;i<LW;i++){ const unsigned char* p=LOGO+((size_t)j*LW+i)*4;
    if(p[3]) blend(x0+i,y0+j,col(p[0],p[1],p[2]),p[3]); } }

int main(int argc,char**argv){
  const char* sf=argc>1?argv[1]:"/data/adb/baseos/run/splash.status"; int maxs=argc>2?atoi(argv[2]):240;
  signal(SIGTERM,on_term); signal(SIGINT,on_term);
  if(font_load()){ fprintf(stderr,"bootsplash: kein Roboto\n"); return 1; }
  logo_load();   /* optional: ohne /data/adb/baseos/logo.png bleibt das alte Text-Layout */
  DFD=open("/dev/dri/card0",O_RDWR|O_CLOEXEC); if(DFD<0){perror("card0");return 1;}
  { int t; for(t=0;t<25;t++){ if(ioctl(DFD,DRM_IOCTL_SET_MASTER,0)==0) break; usleep(200000);} fprintf(stderr,"bootsplash: SET_MASTER nach %d\n",t+1); }
  struct drm_mode_card_res res; memset(&res,0,sizeof res); ioctl(DFD,DRM_IOCTL_MODE_GETRESOURCES,&res);
  uint32_t cn[32],cr[32],en[32],fb_[32];
  res.connector_id_ptr=(uint64_t)(uintptr_t)cn; res.count_connectors=res.count_connectors>32?32:res.count_connectors;
  res.crtc_id_ptr=(uint64_t)(uintptr_t)cr; res.count_crtcs=res.count_crtcs>32?32:res.count_crtcs;
  res.encoder_id_ptr=(uint64_t)(uintptr_t)en; res.count_encoders=res.count_encoders>32?32:res.count_encoders;
  res.fb_id_ptr=(uint64_t)(uintptr_t)fb_; res.count_fbs=res.count_fbs>32?32:res.count_fbs;
  ioctl(DFD,DRM_IOCTL_MODE_GETRESOURCES,&res);
  struct drm_mode_modeinfo mode; int have=0; uint32_t enc=0;
  for(uint32_t i=0;i<res.count_connectors&&!have;i++){ struct drm_mode_get_connector c; memset(&c,0,sizeof c); c.connector_id=cn[i];
    if(ioctl(DFD,DRM_IOCTL_MODE_GETCONNECTOR,&c)<0)continue; if(!c.count_modes||c.connection!=1)continue;
    struct drm_mode_modeinfo ms[64]; uint32_t me[16],mp[64]; uint64_t mv[64];
    c.modes_ptr=(uint64_t)(uintptr_t)ms; c.count_modes=c.count_modes>64?64:c.count_modes;
    c.encoders_ptr=(uint64_t)(uintptr_t)me; c.count_encoders=c.count_encoders>16?16:c.count_encoders;
    c.props_ptr=(uint64_t)(uintptr_t)mp; c.prop_values_ptr=(uint64_t)(uintptr_t)mv; c.count_props=c.count_props>64?64:c.count_props;
    if(ioctl(DFD,DRM_IOCTL_MODE_GETCONNECTOR,&c)<0)continue; if(!c.count_modes)continue;
    mode=ms[0]; CONN=c.connector_id; enc=c.encoder_id; have=1; }
  if(!have){fprintf(stderr,"kein Panel\n");return 1;}
  uint32_t crtc=0; if(enc){struct drm_mode_get_encoder e;memset(&e,0,sizeof e);e.encoder_id=enc;if(ioctl(DFD,DRM_IOCTL_MODE_GETENCODER,&e)==0&&e.crtc_id)crtc=e.crtc_id;}
  if(!crtc&&res.count_crtcs)crtc=cr[0];
  struct drm_mode_create_dumb cq; memset(&cq,0,sizeof cq); cq.width=mode.hdisplay; cq.height=mode.vdisplay; cq.bpp=32; ioctl(DFD,DRM_IOCTL_MODE_CREATE_DUMB,&cq);
  struct drm_mode_fb_cmd fb; memset(&fb,0,sizeof fb); fb.width=cq.width; fb.height=cq.height; fb.pitch=cq.pitch; fb.bpp=32; fb.depth=24; fb.handle=cq.handle; ioctl(DFD,DRM_IOCTL_MODE_ADDFB,&fb);
  struct drm_mode_map_dumb mq; memset(&mq,0,sizeof mq); mq.handle=cq.handle; ioctl(DFD,DRM_IOCTL_MODE_MAP_DUMB,&mq);
  FB=mmap(0,cq.size,PROT_READ|PROT_WRITE,MAP_SHARED,DFD,mq.offset); if(FB==MAP_FAILED)return 1;
  W=cq.width; H=cq.height; PITCH=cq.pitch;
  memset(&CRT,0,sizeof CRT); CRT.crtc_id=crtc; CRT.fb_id=fb.fb_id; CRT.set_connectors_ptr=(uint64_t)(uintptr_t)&CONN; CRT.count_connectors=1; CRT.mode=mode; CRT.mode_valid=1;

  uint32_t BG=col(0,0,0), TXT=col(240,242,246), MUT=col(139,147,159), ACC=col(91,141,239), TRACK=col(38,42,51);
  int first=0; float shown=0; struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
  char last[160]="Startet"; int pct=0;
  for(;;){
    struct timespec now; clock_gettime(CLOCK_MONOTONIC,&now);
    if(g_stop || (maxs>0 && now.tv_sec-t0.tv_sec>=maxs)) break;
    /* Status lesen: "<pct>|<text>" */
    { FILE* f=fopen(sf,"r"); if(f){ char ln[200]; if(fgets(ln,sizeof ln,f)){ char* nl=strchr(ln,'\n'); if(nl)*nl=0;
        char* bar_=strchr(ln,'|'); if(bar_){ *bar_=0; int p=atoi(ln); if(p>=0&&p<=100)pct=p; if(bar_[1]) snprintf(last,sizeof last,"%s",bar_+1); } } fclose(f); } }
    if(pct>=100) break;
    if(shown<pct) shown+= (pct-shown)*0.25f+0.4f; if(shown>pct) shown=pct;   /* weich nachziehen */
    for(uint32_t yy=0;yy<H;yy++){uint32_t*row=(uint32_t*)(FB+yy*PITCH);for(uint32_t xx=0;xx<W;xx++)row[xx]=BG;}
    int bw=640,bh=20;
    if(LOGO){ /* Layout wie die Bootanimation: Logo-Mitte bei 3/8 H, Wortmarke darunter -> nahtloser Uebergang */
      logo_draw(W/2, (int)(H*3/8));
      textc(W/2, (int)(H*61/100), 110, TXT, "barra");
      bar((W-bw)/2, (int)(H*64/100), bw, bh, shown/100.0f, ACC, TRACK);
      textc(W/2, (int)(H*64/100)+84, 30, MUT, last);
      char pc[16]; snprintf(pc,sizeof pc,"%d %%",(int)shown); textc(W/2, (int)(H*64/100)+130, 24, col(96,103,114), pc);
    } else {
      textc(W/2, H/2-40, 150, TXT, "barra");
      bar((W-bw)/2, H/2+40, bw, bh, shown/100.0f, ACC, TRACK);
      textc(W/2, H/2+118, 30, MUT, last);
      char pc[16]; snprintf(pc,sizeof pc,"%d %%",(int)shown); textc(W/2, H/2+164, 24, col(96,103,114), pc);
    }
    if(first==0){ struct drm_mode_crtc offc; memset(&offc,0,sizeof offc); offc.crtc_id=crtc;
      ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&offc); usleep(150000); ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&CRT);
      blw("bl_power","0"); blw("brightness","3000"); usleep(30000); ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&CRT); }
    else ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&CRT);
    if(first<1000)first++;
    usleep(120000);
  }
  /* KEIN Blank: dash2 uebernimmt das Panel direkt (sonst kurz schwarz). Master freigeben. */
  ioctl(DFD,DRM_IOCTL_DROP_MASTER,0);
  return 0;
}

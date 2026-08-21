/* dash2 - modernes Status-Dashboard fuer das Pixel-Base-OS "barra" (headless Node).
 * glibc im Ubuntu-Container, FreeType (AA-Roboto, UTF-8) + rohe DRM/KMS aufs DSI-Panel.
 * Drei Sektionen: Akku & Konnektivitaet / Rechenleistung / Speicher. 1x/s, Auto-Aus nach <timeout> s.
 *   dash2 [timeout]
 * Build: gcc dash2.c -o dash2 $(pkg-config --cflags --libs freetype2) -lm -I.  (stb_image.h daneben) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <strings.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

/* ---------------- DRM/KMS ---------------- */
static int DFD; static uint8_t* FB; static uint32_t W,H,PITCH; static struct drm_mode_crtc CRT; static uint32_t CONN;
static const char* BLP="/sys/class/backlight/panel0-backlight";
static void blw(const char* f,const char* v){ char p[256]; FILE* h; snprintf(p,sizeof p,"%s/%s",BLP,f); if((h=fopen(p,"w"))){fputs(v,h);fclose(h);} }
static void die_blank(int s){ (void)s; blw("brightness","0"); blw("bl_power","4"); _exit(0); }
static void relatch(void){ ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&CRT); }

/* Panel-Pixelformat (empirisch per Balkentest): Speicher [X,R,G,B] low->high, LE=(B<<24)|(G<<16)|(R<<8) */
static uint32_t col(int r,int g,int b){ return ((uint32_t)b<<24)|((uint32_t)g<<16)|((uint32_t)r<<8); }
static inline void pset(int x,int y,uint32_t c){ if(x<0||y<0||x>=(int)W||y>=(int)H)return; *(uint32_t*)(FB+y*PITCH+x*4)=c; }
static inline void blend(int x,int y,uint32_t c,int a){ if(a<=0||x<0||y<0||x>=(int)W||y>=(int)H)return; if(a>=255){pset(x,y,c);return;}
  uint32_t* d=(uint32_t*)(FB+y*PITCH+x*4); uint32_t o=*d,out=0;
  for(int s=8;s<32;s+=8){ int sc=(c>>s)&255,dc=(o>>s)&255; int v=(sc*a+dc*(255-a))/255; out|=(uint32_t)v<<s; } *d=out; }
static void frect(int x,int y,int w,int h,uint32_t c){ for(int j=0;j<h;j++)for(int i=0;i<w;i++)pset(x+i,y+j,c); }
static void rrect(int x,int y,int w,int h,int rad,uint32_t c){ if(rad>h/2)rad=h/2; if(rad>w/2)rad=w/2;
  for(int j=0;j<h;j++)for(int i=0;i<w;i++){ int dx=-1,dy=-1;
    if(i<rad&&j<rad){dx=rad-i;dy=rad-j;} else if(i>=w-rad&&j<rad){dx=i-(w-rad-1);dy=rad-j;}
    else if(i<rad&&j>=h-rad){dx=rad-i;dy=j-(h-rad-1);} else if(i>=w-rad&&j>=h-rad){dx=i-(w-rad-1);dy=j-(h-rad-1);}
    if(dx>=0){ float d=sqrtf((float)(dx*dx+dy*dy)); if(d>rad+0.5f)continue; if(d>rad-0.5f){ blend(x+i,y+j,c,(int)((rad+0.5f-d)*255)); continue;} }
    pset(x+i,y+j,c); }
}
static void bar(int x,int y,int w,int h,float frac,uint32_t fg,uint32_t track){
  if(frac<0)frac=0; if(frac>1)frac=1; rrect(x,y,w,h,h/2,track);
  int fw=(int)(w*frac); if(fw<h&&frac>0.004f)fw=h; if(fw>w)fw=w; if(frac>0.004f) rrect(x,y,fw,h,h/2,fg);
}

/* Logo (vorskaliertes RGBA-PNG neben den anderen hwbridge-Assets; optional) */
static unsigned char* LOGO; static int LW,LH;
static void logo_load(void){ int n; LOGO=stbi_load("/opt/hwbridge/logo.png",&LW,&LH,&n,4); }
static void logo_draw(int x0,int y0){ if(!LOGO)return;
  for(int j=0;j<LH;j++)for(int i=0;i<LW;i++){ const unsigned char* p=LOGO+((size_t)j*LW+i)*4;
    if(p[3]) blend(x0+i,y0+j,col(p[0],p[1],p[2]),p[3]); } }

/* ---------------- FreeType + UTF-8 ---------------- */
static FT_Library FTL; static FT_Face FACE;
static unsigned utf8(const char** s){ const unsigned char* p=(const unsigned char*)*s; unsigned c=*p;
  if(c<0x80){*s+=1;return c;}
  if((c>>5)==0x6 && p[1]){*s+=2;return ((c&0x1F)<<6)|(p[1]&0x3F);}
  if((c>>4)==0xE && p[1]&&p[2]){*s+=3;return ((c&0x0F)<<12)|((p[1]&0x3F)<<6)|(p[2]&0x3F);}
  if((c>>3)==0x1E && p[1]&&p[2]&&p[3]){*s+=4;return ((c&0x07)<<18)|((p[1]&0x3F)<<12)|((p[2]&0x3F)<<6)|(p[3]&0x3F);}
  *s+=1; return c; }
static int text(int x,int baseline,int px,uint32_t c,const char* s){
  FT_Set_Pixel_Sizes(FACE,0,px); int pen=x;
  while(*s){ unsigned cp=utf8(&s); if(FT_Load_Char(FACE,cp,FT_LOAD_RENDER)) continue;
    FT_GlyphSlot g=FACE->glyph; FT_Bitmap* bm=&g->bitmap;
    for(unsigned r=0;r<bm->rows;r++)for(unsigned i=0;i<bm->width;i++){ int a=bm->buffer[r*bm->pitch+i];
      if(a) blend(pen+g->bitmap_left+i, baseline-g->bitmap_top+r, c, a); }
    pen += g->advance.x>>6; }
  return pen-x;
}
static int textw(int px,const char* s){ FT_Set_Pixel_Sizes(FACE,0,px); int wsum=0;
  while(*s){ unsigned cp=utf8(&s); if(FT_Load_Char(FACE,cp,FT_LOAD_DEFAULT))continue; wsum+=FACE->glyph->advance.x>>6; } return wsum; }
static void textr(int xr,int baseline,int px,uint32_t c,const char* s){ text(xr-textw(px,s),baseline,px,c,s); }

/* ---------------- Metriken ---------------- */
static long rdl(const char* p){ FILE* f=fopen(p,"r"); if(!f)return -1; long v=-1; if(fscanf(f,"%ld",&v)!=1)v=-1; fclose(f); return v; }
static void rds(const char* p,char* o,int n){ o[0]=0; FILE* f=fopen(p,"r"); if(!f)return; if(fgets(o,n,f)){char*nl=strchr(o,'\n');if(nl)*nl=0;} fclose(f); }
static long dfreq(const char* node,const char* f){ char p[192]; snprintf(p,sizeof p,"/sys/class/devfreq/%s/%s",node,f); return rdl(p); }
static float thermal(const char* type){ DIR* d=opendir("/sys/class/thermal"); if(!d)return -1; struct dirent* e; float out=-1;
  while((e=readdir(d))){ if(strncmp(e->d_name,"thermal_zone",12))continue; char p[128],t[48];
    snprintf(p,sizeof p,"/sys/class/thermal/%.32s/type",e->d_name); rds(p,t,sizeof t);
    if(!strcmp(t,type)){ snprintf(p,sizeof p,"/sys/class/thermal/%.32s/temp",e->d_name); long v=rdl(p); if(v>=0)out=v/1000.0f; break; } }
  closedir(d); return out; }
#define NCPU 9
static unsigned long long pc_busy[NCPU],pc_tot[NCPU]; static float cpu_load[NCPU];
static void cpu_sample(void){ FILE* f=fopen("/proc/stat","r"); if(!f)return; char ln[256];
  while(fgets(ln,sizeof ln,f)){ int id; unsigned long long u,ni,sy,idle,io,ir,so;
    if(sscanf(ln,"cpu%d %llu %llu %llu %llu %llu %llu %llu",&id,&u,&ni,&sy,&idle,&io,&ir,&so)==8 && id>=0&&id<NCPU){
      unsigned long long busy=u+ni+sy+ir+so, tot=busy+idle+io;
      unsigned long long db=busy-pc_busy[id], dt=tot-pc_tot[id];
      cpu_load[id]= dt? (float)db/(float)dt : 0; pc_busy[id]=busy; pc_tot[id]=tot; } }
  fclose(f); }
/* Beschleuniger-Last (Delta pro Frame, analog cpu_sample):
 *  TPU = aktive Zyklen / (Max-Takt * dt)  -> echte Busy-Fraktion (tpu_active_cycle_count).
 *  DSP = Job-Rate / Referenz  -> Aktivitaet; der GXP hat KEINEN Busy-Zeit-Zaehler,
 *        nur dsp_workload_count bewegt sich (gemessen 20.8.), daher Job-Rate als Proxy. */
#define TPU_CYC "/sys/class/edgetpu/edgetpu-soc/device/tpu_active_cycle_count"
#define DSP_WL  "/sys/devices/platform/20c00000.callisto/dsp_workload_count"
#define TPU_MAX_HZ 1.119e9      /* hoechste rio-Taktstufe (1119 MHz) */
#define DSP_REF_JPS 400.0       /* Job-Rate fuer vollen DSP-Balken (Saettigung ~550-660/s bei winzigen Kerneln) */
static unsigned long long acc_ptpu=0, acc_pdsp=0; static double acc_pt=0;
static float tpu_busy=0, dsp_busy=0;
static unsigned long long rdull(const char* p){ FILE* f=fopen(p,"r"); if(!f)return 0ULL; unsigned long long v=0; if(fscanf(f,"%llu",&v)!=1)v=0; fclose(f); return v; }
static void acc_sample(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); double tn=t.tv_sec+t.tv_nsec/1e9;
  double dt=acc_pt>0?tn-acc_pt:0; acc_pt=tn; if(dt<0.05) return;
  unsigned long long cyc=rdull(TPU_CYC), wl=rdull(DSP_WL);
  if(acc_ptpu && cyc>=acc_ptpu){ double fr=(double)(cyc-acc_ptpu)/(TPU_MAX_HZ*dt); tpu_busy=fr>1?1.f:(float)fr; } acc_ptpu=cyc;
  if(acc_pdsp && wl >=acc_pdsp){ double fr=((double)(wl-acc_pdsp)/dt)/DSP_REF_JPS; dsp_busy=fr>1?1.f:(float)fr; } acc_pdsp=wl;
}
static long meminfo(const char* key){ FILE* f=fopen("/proc/meminfo","r"); if(!f)return -1; char ln[128]; long v=-1;
  while(fgets(ln,sizeof ln,f)){ if(!strncmp(ln,key,strlen(key))){ sscanf(ln+strlen(key),"%ld",&v); break; } } fclose(f); return v; }
static int wifi_ip(char* o,int n){ o[0]=0; struct ifaddrs* ia; if(getifaddrs(&ia))return -1;
  for(struct ifaddrs* p=ia;p;p=p->ifa_next) if(p->ifa_addr&&p->ifa_addr->sa_family==AF_INET&&!strncmp(p->ifa_name,"wlan",4)){
    inet_ntop(AF_INET,&((struct sockaddr_in*)p->ifa_addr)->sin_addr,o,n); break; } freeifaddrs(ia); return o[0]?0:-1; }
static int sockok(const char* nm){ char p[192]; snprintf(p,sizeof p,"/opt/hwbridge/%s",nm); struct stat st; return stat(p,&st)==0; }

/* ---------------- Palette ---------------- */
#define BG      col(0,0,0)
#define TILEBG  col(22,24,29)
#define INNER   col(30,33,40)
#define BORDER  col(46,50,60)
#define TXT     col(237,240,244)
#define MUT     col(139,147,159)
#define DIM     col(96,103,114)
#define TRACK   col(38,42,51)
#define A_CPU   col(91,141,239)
#define A_RAM   col(34,195,195)
#define A_BAT   col(52,199,89)
#define A_GPU   col(48,209,88)
#define A_TPU   col(255,176,32)
#define A_DSP   col(178,104,255)
#define A_NET   col(76,154,255)
#define A_DISK  col(96,178,150)
#define RED     col(255,69,58)

static int tile(int x,int y,int w,int h,const char* title,uint32_t accent){
  rrect(x,y,w,h,22,TILEBG);
  rrect(x+26,y+30,10,10,5,accent);
  text(x+48,y+42,27,TXT,title);
  frect(x+26,y+62,w-52,1,BORDER);
  return y+90;
}
static void chip(int x,int y,int w,int h,const char* label,const char* val,uint32_t vcol){
  rrect(x,y,w,h,14,INNER); text(x+16,y+26,18,MUT,label); text(x+16,y+h-16,26,vcol,val);
}
static const char* tr_status(const char* s){
  if(!strcasecmp(s,"Charging"))return "Lädt";
  if(!strcasecmp(s,"Discharging"))return "Entlädt";
  if(!strcasecmp(s,"Not charging"))return "Lädt nicht";
  if(!strcasecmp(s,"Full"))return "Voll";
  if(!strcasecmp(s,"Unknown"))return "Unbekannt";
  return s[0]?s:"—";
}
/* Sektions-Ueberschrift: Versal-Label + feine Trennlinie. content-top zurueck. */
static int section(int x,int y,int w,const char* label){
  text(x+2,y+22,21,MUT,label); frect(x,y+34,w,2,BORDER); return y+50;
}
/* kompakte Beschleuniger-Kachel (Rechenleistung, 3er-Reihe). on=0 -> ausgegraut/deaktiviert. */
static void acc_tile(int x,int y,int w,int h,const char* name,const char* model,uint32_t acc,int on,
                     const char* primary,const char* sub,float frac,const char* bl,const char* bv){
  uint32_t bg=on?TILEBG:col(16,17,20), ac=on?acc:col(70,74,84), nm=on?TXT:MUT, pc=on?acc:DIM, bvc=on?TXT:DIM;
  rrect(x,y,w,h,20,bg);
  rrect(x+22,y+30,10,10,5,ac); text(x+42,y+41,24,nm,name);
  text(x+22,y+70,17,DIM,model);
  text(x+22,y+156,44,pc,primary); text(x+22,y+184,18,MUT,sub);
  bar(x+22,y+200,w-44,18,on?frac:0,ac,TRACK);
  int by=y+h-70; rrect(x+22,by,w-44,50,12,INNER);
  text(x+38,by+31,18,MUT,bl); textr(x+w-38,by+31,20,bvc,bv);
}

int main(int argc,char**argv){
  int secs=argc>1?atoi(argv[1]):60;
  signal(SIGTERM,die_blank); signal(SIGINT,die_blank);
  if(FT_Init_FreeType(&FTL)){fprintf(stderr,"FT init\n");return 1;}
  if(FT_New_Face(FTL,"/opt/hwbridge/Roboto-Regular.ttf",0,&FACE)){fprintf(stderr,"kein Roboto\n");return 1;}

  DFD=open("/dev/dri/card0",O_RDWR|O_CLOEXEC); if(DFD<0){perror("card0");return 1;}
  /* DRM-Master aktiv erwarten: die Android-Composer-HAL koennte den Master gerade
   * erst freigeben (dispctl/fw-quiet stoppt sie). Bis zu ~4s darauf warten. */
  { int t; for(t=0;t<20;t++){ if(ioctl(DFD,DRM_IOCTL_SET_MASTER,0)==0) break; usleep(200000); }
    fprintf(stderr,"dash2: SET_MASTER nach %d Versuchen\n",t+1); }
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

  logo_load();
  cpu_sample(); usleep(250000);
  int first=0; struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
  for(;;){
    struct timespec now; clock_gettime(CLOCK_MONOTONIC,&now);
    if(secs>0 && now.tv_sec-t0.tv_sec>=secs) break;
    cpu_sample(); acc_sample();
    for(uint32_t yy=0;yy<H;yy++){uint32_t*row=(uint32_t*)(FB+yy*PITCH);for(uint32_t xx=0;xx<W;xx++)row[xx]=BG;}
    char buf[160],hn[64]; int M=36, cw=W-2*M;
    rds("/opt/hwbridge/etc/hostname",hn,sizeof hn); if(!hn[0])rds("/etc/hostname",hn,sizeof hn); if(!hn[0])strcpy(hn,"pixel-node");
    int half=(cw-22)/2, xr=M+half+22;

    /* ===== Kopf ===== */
    { int tx=M; if(LOGO){ logo_draw(M,92-56); tx=M+LW+18; } text(tx,92,58,TXT,"barra"); }
    { int w=wifi_ip(buf,sizeof buf)==0; int pw=textw(20,w?"ONLINE":"OFFLINE")+52;
      rrect(M+cw-pw,58,pw,40,20, w?col(20,58,32):col(58,24,22));
      rrect(M+cw-pw+16,74,10,10,5, w?A_BAT:RED); text(M+cw-pw+34,86,20, w?A_BAT:RED, w?"ONLINE":"OFFLINE"); }
    { long us=0; FILE* f=fopen("/proc/uptime","r"); if(f){double d;if(fscanf(f,"%lf",&d)==1)us=(long)d;fclose(f);}
      float st=thermal("soc_therm");
      snprintf(buf,sizeof buf,"%s   ·   Laufzeit %ldh %02ldm   ·   SoC %.0f °C",hn,us/3600,(us%3600)/60,st<0?0:st); text(M,128,22,MUT,buf); }

    int y=158;
    /* ===== AKKU & KONNEKTIVITÄT ===== */
    y=section(M,y,cw,"AKKU & KONNEKTIVITÄT");
    { int th=410; int cy=tile(M,y,half,th,"Akku & Strom",A_BAT);
      long cap=rdl("/sys/class/power_supply/battery/capacity");
      char stt[24]; rds("/sys/class/power_supply/battery/status",stt,sizeof stt);
      long cc=rdl("/sys/class/power_supply/battery/charge_counter"), cf=rdl("/sys/class/power_supply/battery/charge_full");
      long tmp=rdl("/sys/class/power_supply/battery/temp"), cur=rdl("/sys/class/power_supply/battery/current_now"), vol=rdl("/sys/class/power_supply/battery/voltage_now");
      snprintf(buf,sizeof buf,"%ld%%",cap); text(M+26,cy+68,56,cap>20?A_BAT:RED,buf);
      text(M+26,cy+102,20,MUT,tr_status(stt));
      bar(M+26,cy+124,half-52,20,cap/100.0f,cap>20?A_BAT:RED,TRACK);
      double w=(cur>0&&vol>0)?(cur/1e6)*(vol/1e6):0; int ch3=(half-52-16)/3;
      char v1[40],v2[40],v3[40];
      snprintf(v1,sizeof v1,"%.2f W",w); snprintf(v2,sizeof v2,"%.2f Ah",cc/1e6); snprintf(v3,sizeof v3,"%.1f °C",tmp/10.0);
      chip(M+26,cy+168,ch3,92,"Leistung",v1,TXT);
      chip(M+26+ch3+8,cy+168,ch3,92,"Ladevolumen",v2,TXT);
      chip(M+26+2*(ch3+8),cy+168,ch3,92,"Temperatur",v3,TXT);
      snprintf(buf,sizeof buf,"Kapazität %.2f / %.2f Ah",cc/1e6,cf/1e6); text(M+26,cy+292,18,DIM,buf); }
    { int th=410; int cy=tile(xr,y,half,th,"Konnektivität",A_NET);
      char ip[64]; int w=wifi_ip(ip,sizeof ip)==0;
      rrect(xr+26,cy+12,14,14,7,w?A_BAT:RED); text(xr+50,cy+24,23,TXT,w?"WLAN verbunden":"WLAN aus");
      if(w) chip(xr+26,cy+56,half-52,92,"IPv4-Adresse",ip,TXT);
      int bt=0; { struct stat s; bt=stat("/sys/class/bluetooth/hci0",&s)==0; }
      rrect(xr+26,cy+204,14,14,7,bt?A_BAT:DIM); text(xr+50,cy+216,23,MUT,bt?"Bluetooth aktiv":"Bluetooth aus");
      if(!bt) text(xr+26,cy+256,18,DIM,"Controller stromlos (AOC-Firmware)"); }
    y+=410+36;

    /* ===== RECHENLEISTUNG ===== */
    y=section(M,y,cw,"RECHENLEISTUNG");
    { int th=740; int cy=tile(M,y,cw,th,"CPU  ·  Tensor G3  ·  9 Kerne",A_CPU);
      struct { const char* nm; int core; } rows[]={
        {"LITTLE",0},{"LITTLE",1},{"LITTLE",2},{"LITTLE",3},
        {"MID",4},{"MID",5},{"MID",6},{"MID",7},{"BIG",8} };
      float avg=0; for(int i=0;i<NCPU;i++)avg+=cpu_load[i]; avg/=NCPU;
      snprintf(buf,sizeof buf,"Ø %d%%",(int)(avg*100+.5f)); textr(M+cw-26,y+42,24,MUT,buf);
      int by=cy+10; const char* lastg="";
      for(int i=0;i<NCPU;i++){ int core=rows[i].core;
        if(strcmp(rows[i].nm,lastg)){ lastg=rows[i].nm; by+=12;
          int pw=textw(17,lastg)+28; rrect(M+26,by,pw,26,13,INNER); text(M+40,by+19,17,MUT,lastg); by+=38; }
        char fp[128]; snprintf(fp,sizeof fp,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",core); long fq=rdl(fp);
        float ld=cpu_load[core]; uint32_t bc = ld>0.85f?RED : ld>0.6f?A_TPU : A_CPU;
        snprintf(buf,sizeof buf,"Kern %d",core); text(M+40,by+25,21,TXT,buf);
        bar(M+180,by+9,cw-560,24,ld,bc,TRACK);
        snprintf(buf,sizeof buf,"%d%%",(int)(ld*100+.5f)); textr(M+cw-200,by+25,22,TXT,buf);
        snprintf(buf,sizeof buf,"%ld MHz",fq>0?fq/1000:0); textr(M+cw-40,by+25,21,MUT,buf);
        by+=52; }
      y+=th+16; }
    /* Beschleuniger-Reihe: immer 3 (GPU/TPU/DSP); ohne Bruecke -> ausgegraut */
    { int ah=400, gap=20, aw=(cw-2*gap)/3;
      { long gu=rdl("/sys/devices/platform/1f000000.mali/utilization"); if(gu<0)gu=0; if(gu>100)gu=100;
        char pv[32]; snprintf(pv,sizeof pv,"%ld %%",gu);
        float gt=thermal("G3D"); if(gt<0)gt=thermal("gpu"); char bv[32]; if(gt<0)snprintf(bv,sizeof bv,"—"); else snprintf(bv,sizeof bv,"%.0f °C",gt);
        acc_tile(M,y,aw,ah,"GPU","Mali-G715",A_GPU,1,pv,"Auslastung",gu/100.0f,"Temperatur",bv); }
      { int on=sockok("tpu.sock"); char pv[32],bv[32],sub[40];
        if(on){ long cf=dfreq("1a000000.rio","cur_freq");
          snprintf(pv,sizeof pv,"%d %%",(int)(tpu_busy*100+.5f));
          snprintf(sub,sizeof sub,"Auslastung · %ld MHz",cf>0?cf/1000000:0);
          float tt=thermal("TPU"); if(tt<0)snprintf(bv,sizeof bv,"—"); else snprintf(bv,sizeof bv,"%.0f °C",tt);
        } else { snprintf(pv,sizeof pv,"offline"); snprintf(sub,sizeof sub,"nicht aktiv"); snprintf(bv,sizeof bv,"—"); }
        acc_tile(M+aw+gap,y,aw,ah,"TPU","EdgeTPU G3",A_TPU,on,pv,sub,on?tpu_busy:0,"Temperatur",bv); }
      { int on=sockok("gxp.sock"); char pv[32],bv[40],sub[40];
        if(on){ long cf=dfreq("20c00000.callisto","cur_freq");
          snprintf(pv,sizeof pv,"%d %%",(int)(dsp_busy*100+.5f));
          snprintf(sub,sizeof sub,"Aktivitaet · %ld MHz",cf>0?cf/1000000:0);
          snprintf(bv,sizeof bv,"6 · Soft-Float");
        } else { snprintf(pv,sizeof pv,"offline"); snprintf(sub,sizeof sub,"nicht aktiv"); snprintf(bv,sizeof bv,"—"); }
        acc_tile(M+2*(aw+gap),y,aw,ah,"DSP","Callisto GXP",A_DSP,on,pv,sub,on?dsp_busy:0,"Kernel",bv); }
      y+=ah+36; }

    /* ===== SPEICHER ===== */
    y=section(M,y,cw,"SPEICHER");
    { int th=410; int cy=tile(M,y,half,th,"Arbeitsspeicher",A_RAM);
      long mt=meminfo("MemTotal:"), ma=meminfo("MemAvailable:"), st=meminfo("SwapTotal:"), sf=meminfo("SwapFree:");
      float used=mt>0?(float)(mt-ma)/mt:0; float zr=st>0?(float)(st-sf)/st:0;
      snprintf(buf,sizeof buf,"%.1f GB",(mt-ma)/1048576.0); text(M+26,cy+66,48,TXT,buf);
      snprintf(buf,sizeof buf,"von %.1f GB belegt (%d%%)",mt/1048576.0,(int)(used*100+.5f)); text(M+26,cy+102,20,MUT,buf);
      bar(M+26,cy+124,half-52,24,used,A_RAM,TRACK);
      text(M+26,cy+214,20,MUT,"zram-Swap"); snprintf(buf,sizeof buf,"%.1f / %.1f GB",(st-sf)/1048576.0,st/1048576.0); textr(M+half-26,cy+214,20,TXT,buf);
      bar(M+26,cy+230,half-52,20,zr,col(0,150,165),TRACK);
      snprintf(buf,sizeof buf,"%.1f GB verfügbar",ma/1048576.0); text(M+26,cy+300,18,DIM,buf); }
    { int th=410; int cy=tile(xr,y,half,th,"Datenspeicher",A_DISK);
      struct statvfs vs; double us=0,tot=0,av=0;
      if(statvfs("/",&vs)==0){ tot=(double)vs.f_blocks*vs.f_frsize; av=(double)vs.f_bavail*vs.f_frsize; us=tot-av; }
      float uf=tot>0?(float)(us/tot):0;
      snprintf(buf,sizeof buf,"%.1f GB",us/1073741824.0); text(xr+26,cy+66,48,TXT,buf);
      snprintf(buf,sizeof buf,"von %.1f GB belegt (%d%%)",tot/1073741824.0,(int)(uf*100+.5f)); text(xr+26,cy+102,20,MUT,buf);
      bar(xr+26,cy+124,half-52,24,uf,A_DISK,TRACK);
      snprintf(buf,sizeof buf,"%.1f GB frei",av/1073741824.0); text(xr+26,cy+214,20,MUT,buf);
      text(xr+26,cy+300,18,DIM,"userdata · geteilt mit Android"); }

    /* Erst-Uebernahme vom Android-HWComposer (headless-Boot: niemand hat das
     * Panel vorher per KMS uebernommen). Ein SETCRTC mit gleicher Mode wird als
     * Pageflip behandelt und ueberträgt auf dem Command-Mode-Panel NICHT -> echten
     * Modeset erzwingen (CRTC aus -> mit unserem FB an). Direkt nach dem Boot ist
     * das Panel manchmal noch transient und ein einzelner Modeset "greift" nicht;
     * daher auf mehreren fruehen Frames (0,2,4) wiederholen. Danach nur relatchen. */
    if(first==0){
      /* Ein sauberer voller Modeset: CRTC aus (Panel runter) -> mit unserem FB an.
       * Kein zweites Modeset (das war das sichtbare Blinken); base-boot wartet vor
       * der Uebergabe, damit das Panel hier schon settled ist. Relatch haelt es. */
      struct drm_mode_crtc offc; memset(&offc,0,sizeof offc); offc.crtc_id=crtc;
      ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&offc);
      usleep(150000);
      ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&CRT);
      blw("bl_power","0"); blw("brightness","3000");
      usleep(30000); ioctl(DFD,DRM_IOCTL_MODE_SETCRTC,&CRT);
    } else relatch();
    if(first<1000) first++;
    sleep(1);
  }
  die_blank(0); return 0;
}

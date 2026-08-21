/* genanim - erzeugt die Frames einer bootanimation (barra-Wortmarke + Ladebalken)
 * als PNG (zlib) + desc.txt. Laeuft im Container (glibc, FreeType, zlib).
 *   genanim <part-dir> <desc-datei>
 * Build: gcc genanim.c -o genanim $(pkg-config --cflags --libs freetype2) -lz */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <zlib.h>
#include <ft2build.h>
#include FT_FREETYPE_H

#define W 540
#define H 1200
#define NFR 24
#define FPS 12
static uint8_t IMG[H*W*3];

static void px(int x,int y,int r,int g,int b){ if(x<0||y<0||x>=W||y>=H)return; uint8_t*p=IMG+(y*W+x)*3; p[0]=r;p[1]=g;p[2]=b; }
static void blendpx(int x,int y,int r,int g,int b,int a){ if(a<=0||x<0||y<0||x>=W||y>=H)return; uint8_t*p=IMG+(y*W+x)*3;
  p[0]=(r*a+p[0]*(255-a))/255; p[1]=(g*a+p[1]*(255-a))/255; p[2]=(b*a+p[2]*(255-a))/255; }
static void rrect(int x,int y,int w,int h,int rad,int r,int g,int b){ if(rad>h/2)rad=h/2; if(rad>w/2)rad=w/2;
  for(int j=0;j<h;j++)for(int i=0;i<w;i++){ int dx=-1,dy=-1;
    if(i<rad&&j<rad){dx=rad-i;dy=rad-j;} else if(i>=w-rad&&j<rad){dx=i-(w-rad-1);dy=rad-j;}
    else if(i<rad&&j>=h-rad){dx=rad-i;dy=j-(h-rad-1);} else if(i>=w-rad&&j>=h-rad){dx=i-(w-rad-1);dy=j-(h-rad-1);}
    if(dx>=0){ float d=sqrtf((float)(dx*dx+dy*dy)); if(d>rad+0.5f)continue; if(d>rad-0.5f){ blendpx(x+i,y+j,r,g,b,(int)((rad+0.5f-d)*255)); continue;} }
    px(x+i,y+j,r,g,b); }
}
static void bar(int x,int y,int w,int h,float frac,int r,int g,int b){
  rrect(x,y,w,h,h/2,38,42,51);
  if(frac<0)frac=0; if(frac>1)frac=1; int fw=(int)(w*frac); if(fw<h&&frac>0.01f)fw=h; if(fw>w)fw=w;
  if(frac>0.01f) rrect(x,y,fw,h,h/2,r,g,b);
}
static FT_Face FACE;
static int tw(int px_,const char*s){ FT_Set_Pixel_Sizes(FACE,0,px_); int w=0; for(;*s;s++){ if(FT_Load_Char(FACE,(unsigned char)*s,FT_LOAD_DEFAULT))continue; w+=FACE->glyph->advance.x>>6; } return w; }
static void text(int x,int base,int px_,int r,int g,int b,const char*s){ FT_Set_Pixel_Sizes(FACE,0,px_); int pen=x;
  for(;*s;s++){ if(FT_Load_Char(FACE,(unsigned char)*s,FT_LOAD_RENDER))continue; FT_GlyphSlot gl=FACE->glyph; FT_Bitmap*bm=&gl->bitmap;
    for(unsigned yy=0;yy<bm->rows;yy++)for(unsigned xx=0;xx<bm->width;xx++){int a=bm->buffer[yy*bm->pitch+xx]; if(a) blendpx(pen+gl->bitmap_left+xx,base-gl->bitmap_top+yy,r,g,b,a);} pen+=gl->advance.x>>6; } }

static void chunk(FILE*f,const char*typ,const uint8_t*d,uint32_t n){ uint8_t l[4]={n>>24,n>>16,n>>8,n}; fwrite(l,1,4,f);
  fwrite(typ,1,4,f); if(n)fwrite(d,1,n,f); uLong c=crc32(0,(const Bytef*)typ,4); if(n)c=crc32(c,d,n);
  uint8_t cb[4]={c>>24,c>>16,c>>8,c}; fwrite(cb,1,4,f); }
static int writepng(const char*path){ FILE*f=fopen(path,"wb"); if(!f)return -1;
  static const uint8_t sig[8]={137,80,78,71,13,10,26,10}; fwrite(sig,1,8,f);
  uint8_t ih[13]={W>>24,W>>16,W>>8,W, H>>24,H>>16,H>>8,H, 8,2,0,0,0}; chunk(f,"IHDR",ih,13);
  uLong raw=(uLong)H*(1+W*3); uint8_t*rb=malloc(raw); uLong o=0;
  for(int y=0;y<H;y++){ rb[o++]=0; memcpy(rb+o,IMG+y*W*3,W*3); o+=W*3; }
  uLong cl=compressBound(raw); uint8_t*cb=malloc(cl); compress2(cb,&cl,rb,raw,6);
  chunk(f,"IDAT",cb,cl); chunk(f,"IEND",0,0); free(rb); free(cb); fclose(f); return 0; }

int main(int argc,char**argv){
  const char* dir=argc>1?argv[1]:"part0"; const char* desc=argc>2?argv[2]:"desc.txt";
  FT_Library ft; FT_Init_FreeType(&ft);
  if(FT_New_Face(ft,"/opt/hwbridge/Roboto-Regular.ttf",0,&FACE)){fprintf(stderr,"kein Roboto\n");return 1;}
  int bw=320, bh=12, bx=(W-bw)/2, by=H/2+30;
  for(int fr=0;fr<NFR;fr++){
    memset(IMG,0,sizeof IMG);                 /* schwarz */
    text((W-tw(66,"barra"))/2, H/2-20, 66, 240,242,246, "barra");
    float frac=(NFR>1)?(float)fr/(NFR-1):1;
    bar(bx,by,bw,bh,frac,91,141,239);
    char path[256]; snprintf(path,sizeof path,"%s/%05d.png",dir,fr); writepng(path);
  }
  FILE*d=fopen(desc,"w"); if(d){ fprintf(d,"%d %d %d\np 0 0 part0\n",W,H,FPS); fclose(d); }
  printf("OK %d Frames -> %s, desc -> %s\n",NFR,dir,desc);
  return 0;
}

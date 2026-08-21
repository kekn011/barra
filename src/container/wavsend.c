/* wavsend - schickt die PCM-Daten einer WAV-Datei an audiod-alsa (Ubuntu-Audio ueber Socket).
 * Reine Socket-Bytes, kein ALSA -> laeuft im glibc-Container ohne Haenger.
 * Erwartet 48kHz/stereo/S16LE (wie audiod). Aufruf: wavsend <sock> <datei.wav> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
static int wf(int fd,const void*b,long n){const char*p=b;while(n>0){long w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}

int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: wavsend <sock> <datei.wav>\n"); return 2; }
  FILE* f=fopen(argv[2],"rb"); if(!f){ perror("wav"); return 1; }
  char id[4]; uint32_t sz;
  if(fread(id,1,4,f)!=4 || memcmp(id,"RIFF",4)){ printf("kein RIFF\n"); return 1; }
  fread(&sz,4,1,f); fread(id,1,4,f); if(memcmp(id,"WAVE",4)){ printf("kein WAVE\n"); return 1; }
  uint32_t datasz=0; int found=0;
  while(fread(id,1,4,f)==4 && fread(&sz,4,1,f)==1){
    if(!memcmp(id,"data",4)){ datasz=sz; found=1; break; }
    fseek(f,(sz+1)&~1u,SEEK_CUR);   /* Chunk ueberspringen (auf gerade padden) */
  }
  if(!found){ printf("kein data-Chunk\n"); return 1; }

  int c=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX;
  strncpy(a.sun_path,argv[1],sizeof(a.sun_path)-1);
  if(connect(c,(struct sockaddr*)&a,sizeof a)<0){ printf("connect %s: %s\n",argv[1],strerror(errno)); return 1; }
  char buf[16384]; long total=datasz;
  while(datasz>0){ long r=fread(buf,1,datasz<sizeof buf?datasz:sizeof buf,f); if(r<=0) break; if(wf(c,buf,r)) { printf("send fail\n"); return 1; } datasz-=r; }
  close(c); fclose(f);
  printf("wavsend: %ld Bytes PCM gesendet\n",total-datasz);
  return 0;
}

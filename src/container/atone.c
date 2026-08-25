/* atone - schickt einen Sinuston (48kHz stereo S16LE) an audiod. Baut als Bionic UND glibc.
 * Aufruf: atone <sock> [freq=440] [sekunden=2] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#define RATE 48000
static int wf(int fd,const void*b,long n){const char*p=b;while(n>0){long w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}

int main(int argc,char**argv){
  if(argc<2){ fprintf(stderr,"usage: atone <sock> [freq=440] [sek=2]\n"); return 2; }
  const char* sock=argv[1]; double freq=argc>2?atof(argv[2]):440.0; double secs=argc>3?atof(argv[3]):2.0;
  if(!(freq>0.0&&freq<=20000.0)) freq=440.0;         /* unplausible Werte auf Default */
  if(!(secs>0.0&&secs<=3600.0)) secs=2.0;
  int c=socket(AF_UNIX,SOCK_STREAM,0); if(c<0){ perror("socket"); return 1; }
  if(strlen(sock)>=sizeof(((struct sockaddr_un*)0)->sun_path)){ fprintf(stderr,"Socket-Pfad zu lang\n"); return 2; }
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  if(connect(c,(struct sockaddr*)&a,sizeof a)<0){ printf("connect %s: %s\n",sock,strerror(errno)); return 1; }
  long total=(long)(RATE*secs);
  short chunk[2400*2];   /* 2400 Frames */
  long done=0;
  while(done<total){
    long f=total-done; if(f>2400) f=2400;
    for(long i=0;i<f;i++){
      double t=(double)(done+i)/RATE;
      double env=1.0;
      if(done+i<480) env=(done+i)/480.0;                 /* Fade-in 10ms */
      if(total-(done+i)<480) env=(total-(done+i))/480.0; /* Fade-out */
      short s=(short)(0.3*32767.0*env*sin(2.0*M_PI*freq*t));
      chunk[i*2]=s; chunk[i*2+1]=s;
    }
    if(wf(c,chunk,f*2*2)){ printf("send fail\n"); return 1; }
    done+=f;
  }
  close(c);
  printf("Ton gesendet: %.0f Hz, %.1f s\n",freq,secs);
  return 0;
}

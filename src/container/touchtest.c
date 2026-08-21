/* touchtest - liest den Goodix-Touchscreen (/dev/input/event2) aus Ubuntu (glibc, evdev).
 * Druckt Touch-Punkte (X/Y) + Down/Up. Beweist Touch-Zugriff aus dem Container.
 * Aufruf: touchtest [dev=/dev/input/event2] [sekunden=12] */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/input.h>
#include <time.h>

int main(int argc,char**argv){
  const char* dev=argc>1?argv[1]:"/dev/input/event2";
  int secs=argc>2?atoi(argv[2]):12;
  int fd=open(dev,O_RDONLY);
  if(fd<0){ perror("open"); return 1; }
  int grab=1; if(ioctl(fd,EVIOCGRAB,&grab)<0) printf("(EVIOCGRAB fehlgeschlagen - evtl. grabbt logind; lese trotzdem)\n");
  printf("lese %s, %d s -- TIPPE JETZT AUF DEN BILDSCHIRM\n",dev,secs); fflush(stdout);
  struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
  int x=0,y=0,down=0,taps=0;
  struct input_event ev;
  fcntl(fd,F_SETFL,O_NONBLOCK);
  for(;;){
    struct timespec now; clock_gettime(CLOCK_MONOTONIC,&now);
    if(now.tv_sec-t0.tv_sec>=secs) break;
    ssize_t n=read(fd,&ev,sizeof ev);
    if(n!=sizeof ev){ usleep(5000); continue; }
    if(ev.type==EV_ABS){ if(ev.code==0x35) x=ev.value; else if(ev.code==0x36) y=ev.value; }
    else if(ev.type==EV_KEY && ev.code==0x14a){ /* BTN_TOUCH */
      down=ev.value;
      if(down){ taps++; printf("  TOUCH DOWN #%d bei x=%d y=%d\n",taps,x,y); }
      else printf("  TOUCH UP    bei x=%d y=%d\n",x,y);
      fflush(stdout);
    }
  }
  printf("fertig: %d Beruehrungen erkannt.\n",taps);
  return 0;
}

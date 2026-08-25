/* btnd - Seitentasten-Daemon (Bionic). Liest /dev/input/event0 (gpio_keys: Power/Vol),
 * fuehrt pro Taste die in buttons.conf konfigurierte Aktion aus. headless (Framework aus ->
 * niemand sonst greift die Keys ab). pixel-config setzt die Belegung ueber den Config-Agenten.
 *   buttons.conf (KEY=AKTION je Zeile): POWER|VOLUMEUP|VOLUMEDOWN = shutdown|reboot|volup|voldown|log|none|exec:<pfad>
 * Lang-Druck (>800ms) nutzt optional AKTION_LONG (z.B. POWER_LONG=shutdown). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/input.h>
#include <time.h>

#define CONF "/data/adb/hwbridge/buttons.conf"
#define LOG  "/data/adb/hwbridge/btnd.log"
#define VOL  "/data/adb/hwbridge/vol.sh"
#define DISP "/data/adb/hwbridge/dispctl.sh"

static void logmsg(const char* m){ FILE* f=fopen(LOG,"a"); if(f){ fprintf(f,"[btnd] %s\n",m); fclose(f); } }

/* Aktion fuer einen Key-Namen aus der Config lesen (frisch je Druck -> Config-Aenderung sofort wirksam) */
static void get_action(const char* key, char* out, int n){
  out[0]=0;
  FILE* f=fopen(CONF,"r"); if(!f) return;
  char line[256];
  while(fgets(line,sizeof line,f)){
    char* eq=strchr(line,'='); if(!eq) continue; *eq=0;
    char* k=line; while(*k==' ')k++;
    char* v=eq+1; char* nl=strchr(v,'\n'); if(nl)*nl=0; while(*v==' ')v++;
    if(strcmp(k,key)==0){ snprintf(out,n,"%s",v); break; }
  }
  fclose(f);
}

static void run_action(const char* action){
  char m[256]; snprintf(m,sizeof m,"Aktion: %s",action); logmsg(m);
  if(!strcmp(action,"shutdown"))      system("setprop sys.powerctl shutdown");
  else if(!strcmp(action,"reboot"))   system("setprop sys.powerctl reboot");
  else if(!strcmp(action,"volup"))    system("sh " VOL " up 2>/dev/null");
  else if(!strcmp(action,"voldown"))  system("sh " VOL " down 2>/dev/null");
  else if(!strcmp(action,"display"))    system("sh " DISP " toggle 2>/dev/null &");
  else if(!strcmp(action,"displayon"))  system("sh " DISP " on 2>/dev/null &");
  else if(!strcmp(action,"displayoff")) system("sh " DISP " off 2>/dev/null &");
  else if(!strncmp(action,"exec:",5)) {
    /* Defense-in-depth: nur Skripte aus dem root-eigenen, container-unschreibbaren
     * Verzeichnis ausfuehren (cfgd validiert bereits, aber wir laufen als root). */
    const char* path=action+5;
    if(!strncmp(path,"/data/adb/hwbridge/actions/",sizeof("/data/adb/hwbridge/actions/")-1) && !strstr(path,"..")){
      char c[300]; snprintf(c,sizeof c,"sh %s &",path); system(c);
    }
  }
  /* log / none / leer -> nichts weiter */
}

int main(int argc,char**argv){
  const char* dev=argc>1?argv[1]:"/dev/input/event0";
  int fd=open(dev,O_RDONLY);
  if(fd<0){ logmsg("kann event-Device nicht oeffnen"); return 1; }
  logmsg("gestartet");
  struct input_event ev;
  struct timespec down_ts; int power_down=0;
  while(read(fd,&ev,sizeof ev)==sizeof ev){
    if(ev.type!=EV_KEY) continue;
    const char* key=0;
    if(ev.code==116) key="POWER"; else if(ev.code==115) key="VOLUMEUP"; else if(ev.code==114) key="VOLUMEDOWN";
    if(!key) continue;
    if(ev.value==1){   /* DOWN */
      if(ev.code==116){ clock_gettime(CLOCK_MONOTONIC,&down_ts); power_down=1; }
      else { char a[128]; get_action(key,a,sizeof a); if(a[0]) run_action(a); }  /* Vol: bei DOWN */
    } else if(ev.value==0 && ev.code==116 && power_down){   /* Power UP -> kurz/lang unterscheiden */
      power_down=0;
      struct timespec up; clock_gettime(CLOCK_MONOTONIC,&up);
      long ms=(up.tv_sec-down_ts.tv_sec)*1000 + (up.tv_nsec-down_ts.tv_nsec)/1000000;
      char a[128];
      if(ms>=800){ get_action("POWER_LONG",a,sizeof a); if(!a[0]) get_action("POWER",a,sizeof a); }
      else get_action("POWER",a,sizeof a);
      if(a[0]) run_action(a);
    }
  }
  return 0;
}

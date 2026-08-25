// tpuc1.c - fork-freier Ein-Schuss-Compile, damit frida-Hooks im Prozess bleiben.
// Ruft CompileTfliteFlatbuffer2 EINMAL direkt auf (Modell via fd, gueltige
// leere Options). frida hookt den Config-Lookup und registriert live.
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef int (*c2)(int, size_t, const void *, size_t, int, int *, size_t *, char **);

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *model = argc > 1 ? argv[1]
                        : getenv("MODEL") ? getenv("MODEL")
                        : "/data/local/tmp/mobilenet_v1_1.0_224_quant.tflite";
    const char *so = getenv("COMPILER_SO");
    if (!so) so = "/vendor/lib64/libedgetpu_tflite_compiler.so";
    void *H = dlopen(so, RTLD_LAZY | RTLD_GLOBAL);
    if (!H) { printf("dlopen: %s\n", dlerror()); return 1; }
    c2 compile2 = (c2)dlsym(H, "CompileTfliteFlatbuffer2");
    if (!compile2) { printf("dlsym FEHLT\n"); return 1; }
    printf("compiler @ %p (base-Hinweis)\n", (void *)compile2);
    // Pause, damit frida-Hooks sicher VOR dem Compile/der Registrierung sitzen.
    if (getenv("FRIDA_WAIT")) sleep(3);

    int mfd = open(model, O_RDONLY);
    struct stat st; fstat(mfd, &st);
    size_t msize = st.st_size;
    static unsigned char opts[512] = { 0xf8, 0x1f, 0x00 }; size_t opts_len = 3;
    if (getenv("OPTS_HEX")) {   /* alternatives CompilerOptions-Proto als Hex-String, z.B. "0801" = Feld 1 varint 1 */
        const char *h = getenv("OPTS_HEX"); opts_len = 0;
        for (; h[0] && h[1] && opts_len < sizeof opts; h += 2) { unsigned v; sscanf(h, "%2x", &v); opts[opts_len++] = (unsigned char) v; }
    }
    int out_fd = -1; size_t out_bytes = 0; char *status = NULL;
    int chip = getenv("CHIP") ? atoi(getenv("CHIP")) : 0;
    printf("=> compile %s (%zu) chip=%d\n", model, msize, chip);
    int rc = compile2(mfd, msize, opts, opts_len, chip, &out_fd, &out_bytes, &status);
    printf("rc=%d out_fd=%d out_bytes=%zu status=\"%s\"\n",
           rc, out_fd, out_bytes, status ? status : "(null)");
    if (out_fd >= 0 && out_bytes > 0) {
        void *pkg = malloc(out_bytes); lseek(out_fd, 0, SEEK_SET);
        ssize_t got = read(out_fd, pkg, out_bytes);
        int sf = open("/data/local/tmp/out.package", O_WRONLY|O_CREAT|O_TRUNC, 0644);
        if (sf >= 0) { write(sf, pkg, got); close(sf); }
        printf("PACKAGE %zd Bytes -> /data/local/tmp/out.package\n", got);
    }
    return 0;
}

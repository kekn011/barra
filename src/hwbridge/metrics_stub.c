// metrics_stub.c — drop-in stub for gxp_metrics_logger.so.
// The real one blocks in AServiceManager_waitForService(Stats). We return null/no-op
// so libgxp's GxpMetricsLogger proceeds without the Android Stats binder service.
#include <stdio.h>
void* GxpCapi_CreateMetricsLogger(void* a, void* b, void* c){ (void)a;(void)b;(void)c; return 0; }
void  GxpCapi_ReleaseMetricsLogger(void* a){ (void)a; }
int   GxpCapi_LogMcuCoreCountersMetrics(void* a,void* b,void* c,void* d){ return 0; }
int   GxpCapi_LogMcuCoreUsageMetrics(void* a,void* b,void* c,void* d){ return 0; }
int   GxpCapi_LogPowerMetrics(void* a,void* b,void* c,void* d){ return 0; }
int   GxpCapi_LogRuntimeUserFunctionMetrics(void* a,void* b,void* c,void* d){ return 0; }
int   GxpCapi_LogRuntimeVirtualDeviceMetrics(void* a,void* b,void* c,void* d){ return 0; }
int   GxpCapi_LogSystemErrorMetrics(void* a,void* b,void* c,void* d){ return 0; }

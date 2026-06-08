#ifndef SYSREFLECT_H
#define SYSREFLECT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SystemMonitor SystemMonitor;

// Initialize a new system monitor instance
SystemMonitor* sysreflect_new();

// Free a system monitor instance
void sysreflect_free(SystemMonitor* ptr);

// Refresh system data
void sysreflect_refresh(SystemMonitor* ptr);

// Get CPU usage percentage. 
// Returns >= 0.0 for success, -1.0 for Restricted, -2.0 for Unavailable.
float sysreflect_get_cpu_usage(SystemMonitor* ptr);

// Get total memory in bytes. Returns 0 if restricted.
uint64_t sysreflect_get_total_memory(SystemMonitor* ptr);

// Get used memory in bytes. Returns 0 if restricted.
uint64_t sysreflect_get_used_memory(SystemMonitor* ptr);

// Get number of CPU cores.
size_t sysreflect_get_core_count(SystemMonitor* ptr);

// Get host name. Returns a C string that MUST be freed with sysreflect_free_string.
char* sysreflect_get_host_name(SystemMonitor* ptr);

// Get OS version. Returns a C string that MUST be freed with sysreflect_free_string.
char* sysreflect_get_os_version(SystemMonitor* ptr);

// Get kernel version. Returns a C string that MUST be freed with sysreflect_free_string.
char* sysreflect_get_kernel_version(SystemMonitor* ptr);

// Get uptime in seconds.
uint64_t sysreflect_get_uptime(SystemMonitor* ptr);

// Get network RX in bytes.
uint64_t sysreflect_get_net_rx(SystemMonitor* ptr);

// Get network TX in bytes.
uint64_t sysreflect_get_net_tx(SystemMonitor* ptr);

// Free a string returned by SysReflect functions
void sysreflect_free_string(char* ptr);

// Inspect a model file or directory. Returns a JSON string that MUST be freed with sysreflect_free_string.
char* sysreflect_inspect_model(const char* path);

// Check if a model file or directory is safe to load based on system resources and safety margin
bool sysreflect_model_is_safe_to_load(const char* path, double safety_margin);

#ifdef __cplusplus
}
#endif

#endif // SYSREFLECT_H



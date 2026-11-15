#ifndef SYSREFLECT_H
#define SYSREFLECT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes returned by lmsafe functions
typedef enum {
    LMSAFE_OK = 0,
    LMSAFE_NULL_POINTER = 1,
    LMSAFE_INVALID_UTF8 = 2,
    LMSAFE_IO_ERROR = 3,
    LMSAFE_PARSE_ERROR = 4,
    LMSAFE_INSUFFICIENT_SPACE = 5,
    LMSAFE_MODEL_NOT_FOUND = 6,
    LMSAFE_UNKNOWN_FAMILY = 7,
    LMSAFE_STORAGE_ALREADY_INIT = 8,
    LMSAFE_TELEMETRY_RESTRICTED = 9,
} LmsafeError;

typedef struct SystemMonitor SystemMonitor;

// ──────────────────────────────────────────────────────────────────────────────
// Error Handling
// ──────────────────────────────────────────────────────────────────────────────

// Get the last error message. Returns null if no error.
// The caller MUST free the returned string with lmsafe_free_string.
char* lmsafe_last_error();

// Get the last error code. Returns LMSAFE_OK (0) if no error.
// Returns int instead of LmsafeError to simplify foreign-language interop.
int lmsafe_last_error_code();

// ──────────────────────────────────────────────────────────────────────────────
// System Monitor
// ──────────────────────────────────────────────────────────────────────────────

// Initialize a new system monitor instance
SystemMonitor* lmsafe_new();

// Free a system monitor instance
void lmsafe_free(SystemMonitor* ptr);

// Refresh system data
void lmsafe_refresh(SystemMonitor* ptr);

// Get CPU usage percentage. 
// Returns >= 0.0 for success, -1.0 for Restricted, -2.0 for Unavailable.
float lmsafe_get_cpu_usage(SystemMonitor* ptr);

// Get total memory in bytes. Returns 0 if restricted.
uint64_t lmsafe_get_total_memory(SystemMonitor* ptr);

// Get used memory in bytes. Returns 0 if restricted.
uint64_t lmsafe_get_used_memory(SystemMonitor* ptr);

// Get number of CPU cores.
size_t lmsafe_get_core_count(SystemMonitor* ptr);

// Get host name. Returns a C string that MUST be freed with lmsafe_free_string.
char* lmsafe_get_host_name(SystemMonitor* ptr);

// Get OS version. Returns a C string that MUST be freed with lmsafe_free_string.
char* lmsafe_get_os_version(SystemMonitor* ptr);

// Get kernel version. Returns a C string that MUST be freed with lmsafe_free_string.
char* lmsafe_get_kernel_version(SystemMonitor* ptr);

// Get uptime in seconds.
uint64_t lmsafe_get_uptime(SystemMonitor* ptr);

// Get network RX in bytes.
uint64_t lmsafe_get_net_rx(SystemMonitor* ptr);

// Get network TX in bytes.
uint64_t lmsafe_get_net_tx(SystemMonitor* ptr);

// Free a string returned by lmsafe functions
void lmsafe_free_string(char* ptr);

// ──────────────────────────────────────────────────────────────────────────────
// Model Inspection
// ──────────────────────────────────────────────────────────────────────────────

// Inspect a model file or directory. Returns a JSON string that MUST be freed with lmsafe_free_string.
// Returns null on error - call lmsafe_last_error() for details.
char* lmsafe_inspect_model(const char* path);

// Check if a model file or directory is safe to load based on system resources and safety margin
bool lmsafe_model_is_safe_to_load(const char* path, double safety_margin);

// ──────────────────────────────────────────────────────────────────────────────
// Safe Zone Storage
// ──────────────────────────────────────────────────────────────────────────────

// Initialize the base storage path for Safe Zone file management
bool lmsafe_init_storage(const char* base_path, bool sandboxed);

// Copy a model into the secure Safe Zone, segregated by model family.
// Returns a C string of the new destination path that MUST be freed.
// Returns null on error - call lmsafe_last_error() for details.
char* lmsafe_copy_to_safe_zone(const char* source_path, const char* family_str, double safety_margin);

// Clear out the Safe Zone storage for a particular model family
bool lmsafe_clean_safe_zone(const char* family_str);

// Retrieve the first available model in the Safe Zone for a given family.
// Returns a C string of the path that MUST be freed, or NULL if none exist.
char* lmsafe_get_first_stored_model(const char* family_str);

// Retrieve the Safe Zone directory for a given family.
// Returns a C string of the path that MUST be freed.
char* lmsafe_get_safe_zone_dir(const char* family_str);

// ──────────────────────────────────────────────────────────────────────────────
// Media Inspection & Storage
// ──────────────────────────────────────────────────────────────────────────────

// Copy a media file into the secure Safe Zone, segregated by media family.
// Returns a C string of the new destination path that MUST be freed.
// Returns null on error - call lmsafe_last_error() for details.
char* lmsafe_copy_media_to_safe_zone(const char* source_path_str, const char* media_family_str, double safety_margin);

// Clear out the Safe Zone storage for a particular media family
bool lmsafe_clean_media_safe_zone(const char* media_family_str);

// List all media files in the Safe Zone for a particular media family.
// Returns a JSON array of MediaInfo objects (file_name, size_bytes).
// Returns a C string that MUST be freed with lmsafe_free_string.
char* lmsafe_list_media_safe_zone(const char* media_family_str);

// Remove a specific media file from the Safe Zone by file name.
// Returns true on success, false on error.
bool lmsafe_remove_media_from_safe_zone(const char* media_family_str, const char* file_name);

// Inspect a media file. Returns a JSON string that MUST be freed with lmsafe_free_string.
// Returns null on error - call lmsafe_last_error() for details.
char* lmsafe_inspect_media_file(const char* path_str);

#ifdef __cplusplus
}
#endif

#endif // SYSREFLECT_H

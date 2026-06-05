/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // for dladdr / RTLD_NEXT
#endif
#include <backtrace.h>  // DWARF-aware symbolization: to get function name and file:line
#include <cxxabi.h>
#include <dlfcn.h>
#include <link.h>  // dl_iterate_phdr, struct dl_phdr_info
#include <new>
#include <pthread.h>  // mutex guarding the cached backtrace state
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <typeinfo>
#include <unistd.h>

// Resolve the real __cxa_throw at library load time
// to avoid calling dlsym() during an OOM situation
typedef void (*cxa_throw_t)(void*, std::type_info*, void (*)(void*));

static cxa_throw_t real_cxa_throw = nullptr;

// libbacktrace state. It reads the DWARF info of the executable AND every
// shared object loaded into the process, so it can turn a raw PC into
// function + file:line — something backtrace_symbols/dladdr can never do,
// because they only read the dynamic symbol table (.dynsym).
//
// IMPORTANT libbacktrace limitation: a backtrace_state snapshots the set of
// loaded filenames (via dl_iterate_phdr) exactly once, on its first use, and
// never refreshes it. Any shared object loaded *after* that first use — e.g.
// libcuvs.so / librmm.so, dlopened or lazily bound after some earlier throw
// already triggered initialization — is permanently invisible to that state,
// so its PCs resolve to "?? ??:0" and we fall back to dladdr's .dynsym names
// (which omit -fvisibility=hidden internals, hence the bare "??" frames).
//
// Fix: detect when the loaded-filename set has changed and build a fresh state.
// This happens only a handful of times (each dlopen/lazy-load during startup),
// NOT per throw, so the common path stays a cheap cached-pointer read and the
// per-recreation cost (libbacktrace never frees old states) is bounded.
static pthread_mutex_t bt_lock          = PTHREAD_MUTEX_INITIALIZER;
static struct backtrace_state* bt_state = nullptr;
static unsigned long bt_modsig          = 0;  // signature of the loaded-filename set

// dl_iterate_phdr callback: fold each loaded object into a cheap signature so
// we can tell when a library has been added/removed/relocated.
static int modsig_cb(struct dl_phdr_info* info, size_t /*sz*/, void* data)
{
  unsigned long* acc = static_cast<unsigned long*>(data);
  *acc               = (*acc * 1000003UL) ^ static_cast<unsigned long>(info->dlpi_addr);
  return 0;
}
static unsigned long current_modsig()
{
  unsigned long acc = 1469598103934665603UL;  // FNV-ish seed
  dl_iterate_phdr(modsig_cb, &acc);
  return acc;
}

// Return a backtrace_state whose filename list covers everything loaded *now*.
// libbacktrace offers no API to refresh a state, so when the filename set has
// changed we create a fresh one (and intentionally leak the old — libbacktrace
// states are never freed by design).
static struct backtrace_state* get_state()
{
  unsigned long sig = current_modsig();
  pthread_mutex_lock(&bt_lock);
  if (!bt_state || sig != bt_modsig) {
    // filename = nullptr -> uses /proc/self/exe, handles PIE/ASLR.
    // threaded = 1       -> internal state is guarded for use from any thread.
    bt_state  = backtrace_create_state(nullptr,
                                      /*threaded=*/1,
                                      /*error_cb=*/nullptr,
                                      /*data=*/nullptr);
    bt_modsig = sig;
  }
  struct backtrace_state* s = bt_state;
  pthread_mutex_unlock(&bt_lock);
  return s;
}

// __attribute__((constructor)) runs when the .so is loaded, before any throws
__attribute__((constructor)) static void init()
{
  // resolve the real __cxa_throw at library load time, so we never call
  // dlsym() during an OOM because dlsym() itself can throw a bad_alloc.
  real_cxa_throw = reinterpret_cast<cxa_throw_t>(dlsym(RTLD_NEXT, "__cxa_throw"));
}

// Emit one resolved frame, async-signal-ish: format into a stack buffer and
// write() directly to fd 2 (no stdio buffering). __cxa_demangle does allocate,
// which is acceptable here: the bad_alloc object is already constructed, so we
// are past the actual allocation failure point.
static void print_frame(
  int idx, uintptr_t pc, const char* module, const char* func, const char* file, int line) noexcept
{
  const char* name = func ? func : "??";

  // Demangle Itanium C++ names (_Z...) -> human readable. Falls back to the
  // raw name if it is not a mangled symbol or demangling fails.
  char* demangled = nullptr;
  if (func && func[0] == '_' && func[1] == 'Z') {
    int status = 0;
    demangled  = abi::__cxa_demangle(func, nullptr, nullptr, &status);
    if (status == 0 && demangled) { name = demangled; }
  }

  char buf[2048];
  int n;
  if (file) {
    n = snprintf(buf,
                 sizeof buf,
                 "  #%-2d 0x%012lx  %s\n        at %s:%d  (%s)\n",
                 idx,
                 static_cast<unsigned long>(pc),
                 name,
                 file,
                 line,
                 module ? module : "??");
  } else {
    // No DWARF line info for this module (e.g. stripped libc/libstdc++,
    // or the exe built without -g): show name + module + offset only.
    n = snprintf(buf,
                 sizeof buf,
                 "  #%-2d 0x%012lx  %s\n        in %s\n",
                 idx,
                 static_cast<unsigned long>(pc),
                 name,
                 module ? module : "??");
  }
  if (n > 0) {
    size_t len = static_cast<size_t>(n) < sizeof buf ? static_cast<size_t>(n) : sizeof buf - 1;
    write(STDERR_FILENO, buf, len);
  }
  free(demangled);
}

// Called once per frame (and once per inlined frame).
// function/file may be NULL and lineno 0 when DWARF is missing for that program counter (PC).
static int frame_callback(
  void* data, uintptr_t pc, const char* file, int lineno, const char* function) noexcept
{
  int* idx = static_cast<int*>(data);
  // Always get the filename path; also use dladdr's .dynsym name as a fallback
  // when libbacktrace found no function name (typical for libc/libstdc++
  // internals that have neither DWARF nor a static symtab on this system).
  const char* module = nullptr;
  Dl_info info;
  if (dladdr((void*)pc, &info)) {
    module = info.dli_fname;
    if (!function) {
      function = info.dli_sname;  // nearest exported dynamic symbol
    }
  }
  print_frame((*idx)++, pc, module, function, file, lineno);
  return 0;  // 0 = continue walking the stack
}

// Called on a real failure inside libbacktrace (rare). Report and move on.
static void error_callback(void* /*data*/, const char* msg, int /*errnum*/)
{
  const char header[] = "=== libbacktrace error ===\n";
  write(STDERR_FILENO, header, strlen(header));
  if (msg) {
    write(STDERR_FILENO, msg, strlen(msg));
  }
  write(STDERR_FILENO, "\n", 1);
}

extern "C" void __cxa_throw(void* obj, std::type_info* tinfo, void (*dest)(void*))
{
  const char header[] = "=== intercepted throw, backtrace ===\n";
  write(STDERR_FILENO, header, strlen(header));

  // Build/refresh the state against the filenames loaded *right now*, so PCs in
  // libraries loaded after an earlier throw (libcuvs.so, librmm.so, ...) still
  // resolve to function + file:line instead of dladdr's .dynsym-only "??".
  struct backtrace_state* st = get_state();
  if (st) {
    int idx = 0;
    // skip = 1 drops this __cxa_throw hook frame itself, so the trace
    // starts at the code that actually threw.
    backtrace_full(st, /*skip=*/1, frame_callback, error_callback, &idx);
  } else {
    const char msg[] = "(libbacktrace state unavailable)\n";
    write(STDERR_FILENO, msg, strlen(msg));
  }

  // call the real __cxa_throw()
  real_cxa_throw(obj, tinfo, dest);

  __builtin_unreachable();
}

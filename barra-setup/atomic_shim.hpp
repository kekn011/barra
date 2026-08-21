// mingw C++ shim: expose C11 <stdatomic.h> names as the C++ <atomic> ones.
#pragma once
#include <atomic>
using std::atomic_int;
using std::atomic_load;
using std::atomic_store;
using std::atomic_exchange;
using std::atomic_exchange_explicit;
using std::atomic_compare_exchange_strong;
using std::atomic_fetch_add;
using std::memory_order_relaxed;
using std::memory_order_seq_cst;

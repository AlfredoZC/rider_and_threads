#if defined(__SANITIZE_THREAD__)
#define DELIVERY_SIM_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define DELIVERY_SIM_TSAN 1
#endif
#endif

#ifdef DELIVERY_SIM_TSAN

extern "C" const char* __tsan_default_suppressions();

extern "C" const char* __tsan_default_suppressions() {
    return
        "deadlock:CPLAcquireMutex\n"
        "deadlock:CPLCreateOrAcquireMutexEx\n"
        "race:CPLCreateOrAcquireMutexEx\n";
}

#endif

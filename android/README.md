# DragonForge Android v0.1

Initial native Android companion target: **Samsung Galaxy S21 Ultra**.

The phone is a companion to the Linux DragonForge / Greenlight Recur host. Model
training stays on Linux; Android displays status and diagnostics over the
versioned `/api/v1` companion protocol.

## v0.1 scope

- native Kotlin + Jetpack Compose UI
- Android 11+ target compatibility for the S21 Ultra generation
- configurable Linux host address
- training state, step, loss, expert count and VRAM
- Training Doctor health
- pull/refresh action
- read-only operation

Remote training controls remain intentionally unavailable until secure pairing
and authentication are implemented.

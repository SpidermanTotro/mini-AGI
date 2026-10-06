# DragonForge Mobile

DragonForge is expanding beyond the Linux training host.

## iPhone — v0.1 in development

The native SwiftUI companion lives in `iphone/DragonForge/`.

Initial scope:
- connect to a local DragonForge/Greenlight Linux host
- training status, step, loss, expert count and VRAM
- Training Doctor health
- read-only companion API first
- secure pairing before any training-control endpoints

## Android — coming next

Android will use the same versioned companion API as iPhone rather than
introducing a second server protocol.

Planned initial scope:
- native Android dashboard
- Linux host pairing
- training status and telemetry
- Training Doctor findings
- A/B experiment results
- notifications for completed, failed or critical runs

Training remains on the Linux host. Mobile clients are companions; they do not
replace the CUDA training runtime.

## Linux — core platform

Linux remains the primary training, diagnostics, checkpoint, experiment and
local-model platform. Mobile work must not weaken Linux CI or change training
behavior merely to support a client.

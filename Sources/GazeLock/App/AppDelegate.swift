import AppKit
import AVFoundation
import CoreVideo
import Foundation
import SwiftUI
import SystemExtensions

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var menuBarController: MenuBarController!
    private var popoverController: NSPopover!
    private var controlStore: ControlStateStore!
    private var controlServer: ControlServer!
    private var cameraCapture: CameraCapture!
    private var pipeline: FramePipeline!

    // Main window (Phase 3c)
    private var mainWindow: NSWindow?
    private var calibrationStore: CalibrationStore!
    private var appProfileObserver: AppProfileObserver!
    private var overrides = AppProfileOverrides()

    // Observable preview state — SwiftUI views re-render on every update.
    private let previewState = PreviewState()

    // Calibration wizard (P3c.8)
    private var calibrationCapture: CalibrationCapture?
    private var calibrationHost: NSHostingController<CalibrationWizard>?
    private var calibrationDetector: LandmarkDetector?

    // Onboarding (P3c.11)
    private let onboardingDefaultsKey = "hasCompletedOnboarding"
    private var onboardingWindow: NSWindow?
    private var onboardingCoord: OnboardingCoordinator?
    private var systemExtensionDelegate: SystemExtensionDelegate?

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupPipeline()
        setupControlPlane()
        setupMainWindowDependencies()
        setupMenuBar()
        setupPopover()
        startCapture()

        if !UserDefaults.standard.bool(forKey: onboardingDefaultsKey) {
            showOnboarding()
        }
    }

    private func setupMainWindowDependencies() {
        calibrationStore = CalibrationStore()

        if let data = UserDefaults.standard.data(forKey: AppProfileOverridesDefaultsKey.key),
           let saved = try? JSONDecoder().decode(AppProfileOverrides.self, from: data) {
            overrides = saved
        }
        appProfileObserver = AppProfileObserver(
            overrides: overrides,
            defaultProfileId: controlStore.state.setupProfileId,
            store: controlStore
        )
        appProfileObserver.start()
    }

    private func setupPipeline() {
        do {
            pipeline = try FramePipeline()
        } catch {
            presentFatal("Could not initialise frame pipeline: \(error)")
            return
        }
    }

    private func setupControlPlane() {
        // Authoritative state store; changes push to the extension.
        controlStore = ControlStateStore(
            initial: .default,
            onPushNeeded: { [weak self] newState in
                self?.controlServer?.broadcast(newState)
            }
        )
        controlServer = ControlServer { [weak self] in
            self?.controlStore.state ?? .default
        }
        controlServer.start()
    }

    private func setupMenuBar() {
        menuBarController = MenuBarController()
        menuBarController.setState(.onIdle)
        menuBarController.onClick = { [weak self] in
            self?.togglePopover()
        }
    }

    private func setupPopover() {
        popoverController = NSPopover()
        popoverController.behavior = .transient
        popoverController.contentSize = NSSize(width: PopoverStyle.popoverWidth, height: 250)
        let root = PopoverView(
            store: controlStore,
            previewState: previewState,
            onOpenWindow: { [weak self] in
                self?.showMainWindow()
            }
        )
        popoverController.contentViewController = NSHostingController(rootView: root)
    }

    private func startCapture() {
        cameraCapture = CameraCapture()
        // Capture non-isolated references locally so the off-main sample
        // callback can use them without crossing actor isolation.
        // `FramePipeline` is a final class with no actor isolation; it
        // is safe to invoke from the capture queue.
        let pipelineRef = pipeline!
        cameraCapture.onFrame = { pb, ts in
            // Render BEFORE off-main (pure CoreImage → NSImage; safe).
            let before = PreviewFrameRenderer.nsImage(from: pb)
            // Hop to main only to read the latest state, then run
            // the pipeline off-main using the captured reference.
            Task { @MainActor [weak self] in
                guard let self else { return }
                // Feed calibration if active
                if let capture = self.calibrationCapture, capture.isRecording,
                   let detector = self.calibrationDetector,
                   let landmarks = try? detector.detect(in: pb, timestamp: ts) {
                    capture.feed(headPose: landmarks.headPoseRadians)
                }
                // Respect the correction toggle — if OFF, AFTER just mirrors
                // BEFORE (passthrough). Matches the extension-side gate.
                let after: NSImage?
                if self.controlStore.state.isEnabled {
                    let intensity = self.controlStore.state.intensity
                    let vAim = self.controlStore.state.verticalAimDeg
                    let hAim = self.controlStore.state.horizontalAimDeg
                    let sens = self.controlStore.state.sensitivity
                    after = await Self.processOffMain(
                        pipeline: pipelineRef,
                        pixelBuffer: pb,
                        timestamp: ts,
                        intensity: intensity,
                        verticalAimDeg: vAim,
                        horizontalAimDeg: hAim,
                        sensitivity: sens
                    )
                } else {
                    after = before
                }
                self.previewState.update(before: before, after: after)
            }
        }
        // Gate start on camera authorization. If .notDetermined, trigger the
        // system prompt first. If .denied/.restricted, surface an error linking
        // to System Settings (onboarding's camera step can be bypassed if the
        // `hasCompletedOnboarding` flag was set without the user actually
        // granting permission).
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            beginCapture()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self else { return }
                    if granted {
                        self.beginCapture()
                    } else {
                        self.menuBarController.setState(.error)
                        self.presentCameraDeniedAlert()
                    }
                }
            }
        case .denied, .restricted:
            menuBarController.setState(.error)
            presentCameraDeniedAlert()
        @unknown default:
            menuBarController.setState(.error)
            presentCameraDeniedAlert()
        }
    }

    private func beginCapture() {
        do {
            try cameraCapture.start()
        } catch {
            menuBarController.setState(.error)
            presentFatal("Could not start camera: \(error)")
        }
    }

    private func presentCameraDeniedAlert() {
        let alert = NSAlert()
        alert.messageText = "Camera access required"
        alert.informativeText = "GazeLock needs camera access to correct your eye gaze. "
            + "Open System Settings > Privacy & Security > Camera and enable GazeLock."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Open System Settings")
        alert.addButton(withTitle: "Later")
        if alert.runModal() == .alertFirstButtonReturn {
            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera") {
                NSWorkspace.shared.open(url)
            }
        }
    }

    /// Runs the pipeline and renders an NSImage off the main actor.
    /// `FramePipeline` carries no actor isolation, so this is valid.
    private static func processOffMain(
        pipeline: FramePipeline,
        pixelBuffer: CVPixelBuffer,
        timestamp: TimeInterval,
        intensity: Double,
        verticalAimDeg: Double,
        horizontalAimDeg: Double,
        sensitivity: Sensitivity
    ) async -> NSImage? {
        let out: CVPixelBuffer = (try? pipeline.process(
            pixelBuffer: pixelBuffer,
            timestamp: timestamp,
            intensity: intensity,
            verticalAimDeg: verticalAimDeg,
            horizontalAimDeg: horizontalAimDeg,
            sensitivity: sensitivity
        )) ?? pixelBuffer
        return PreviewFrameRenderer.nsImage(from: out)
    }

    private func togglePopover() {
        guard let button = menuBarController.button else { return }
        if popoverController.isShown {
            popoverController.performClose(nil)
        } else {
            popoverController.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        }
    }

    private func showMainWindow() {
        if mainWindow == nil {
            let overridesBinding = Binding<AppProfileOverrides>(
                get: { [weak self] in self?.overrides ?? AppProfileOverrides() },
                set: { [weak self] newVal in self?.setAppProfileOverrides(newVal) }
            )
            let root = MainWindow(
                store: controlStore,
                calibrationStore: calibrationStore,
                overrides: overridesBinding,
                previewState: previewState,
                onLaunchCalibration: { [weak self] in self?.launchCalibration() },
                onRunAutoDetect: { [weak self] in self?.runAutoDetect() }
            )
            let window = NSWindow(
                contentRect: NSRect(
                    x: 0, y: 0,
                    width: MainWindowStyle.windowWidth,
                    height: MainWindowStyle.windowHeight
                ),
                styleMask: [.titled, .closable, .miniaturizable, .resizable],
                backing: .buffered,
                defer: false
            )
            window.title = "GazeLock"
            window.center()
            window.contentViewController = NSHostingController(rootView: root)
            window.isReleasedWhenClosed = false
            window.delegate = self
            mainWindow = window
        }
        NSApp.setActivationPolicy(.regular)
        mainWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func setAppProfileOverrides(_ newVal: AppProfileOverrides) {
        overrides = newVal
        appProfileObserver?.setOverrides(newVal)
        if let data = try? JSONEncoder().encode(newVal) {
            UserDefaults.standard.set(data, forKey: AppProfileOverridesDefaultsKey.key)
        }
    }

    private func launchCalibration() {
        let capture = CalibrationCapture()
        calibrationCapture = capture
        calibrationDetector = LandmarkDetector()

        let secondaryNames = NSScreen.screens.dropFirst().enumerated().map { index, screen in
            let name = screen.localizedName
            return name.isEmpty ? "Screen \(index + 2)" : name
        }

        let wizard = CalibrationWizard(
            capture: capture,
            secondaryScreenNames: secondaryNames,
            baseProfileId: controlStore.state.setupProfileId,
            onSave: { [weak self] cal in
                self?.calibrationStore.save(cal)
                self?.dismissCalibration()
            },
            onCancel: { [weak self] in
                self?.dismissCalibration()
            }
        )
        let host = NSHostingController(rootView: wizard)
        calibrationHost = host

        guard let mainWindow else {
            presentInfo("Open the main window before calibrating.")
            dismissCalibration()
            return
        }
        mainWindow.contentViewController?.presentAsSheet(host)
    }

    private func dismissCalibration() {
        if let host = calibrationHost, let parent = host.parent {
            parent.dismiss(host)
        }
        calibrationHost = nil
        calibrationCapture = nil
        calibrationDetector = nil
    }

    private func runAutoDetect() {
        let proposed = AutoDetect.proposeProfileLive()
        let alert = NSAlert()
        alert.messageText = "Detected: \(proposed.name)"
        alert.informativeText = "Apply this setup?"
        alert.addButton(withTitle: "Use it")
        alert.addButton(withTitle: "Cancel")
        if alert.runModal() == .alertFirstButtonReturn {
            controlStore.setSetupProfileId(proposed.id)
        }
    }

    private func showOnboarding() {
        let coord = OnboardingCoordinator()
        onboardingCoord = coord

        let sheet = OnboardingSheet(
            coordinator: coord,
            onFinish: { [weak self] pickedProfileId in
                guard let self else { return }
                UserDefaults.standard.set(true, forKey: self.onboardingDefaultsKey)
                self.controlStore.setSetupProfileId(pickedProfileId)
                self.dismissOnboarding()
                self.showMainWindow()
            },
            onRequestExtensionInstall: { [weak self] in self?.requestExtensionInstall() },
            onRequestCameraPermission: { [weak self] in
                guard let self else { return false }
                return await self.requestCameraPermission()
            }
        )
        let host = NSHostingController(rootView: sheet)
        let window = NSWindow(contentViewController: host)
        window.title = "Welcome to GazeLock"
        window.styleMask = [.titled, .closable]
        window.center()
        window.isReleasedWhenClosed = false
        onboardingWindow = window
        NSApp.setActivationPolicy(.regular)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func dismissOnboarding() {
        onboardingWindow?.close()
        onboardingWindow = nil
        onboardingCoord = nil
    }

    private func requestExtensionInstall() {
        onboardingCoord?.extInstallState = .submitted

        let delegate = SystemExtensionDelegate(
            onNeedsApproval: { [weak self] in
                Task { @MainActor in
                    self?.onboardingCoord?.extInstallState = .waitingForUserApproval
                }
            },
            onResult: { [weak self] activated, message in
                Task { @MainActor in
                    guard let self else { return }
                    if activated {
                        self.onboardingCoord?.extInstallState = .activated
                        self.onboardingCoord?.extensionActivated = true
                    } else {
                        self.onboardingCoord?.extInstallState = .failed(
                            message ?? "Install failed. See Console.app for details."
                        )
                    }
                }
            }
        )
        systemExtensionDelegate = delegate

        let req = OSSystemExtensionRequest.activationRequest(
            forExtensionWithIdentifier: "com.gazelock.GazeLock.CameraExtension",
            queue: .main
        )
        req.delegate = delegate
        OSSystemExtensionManager.shared.submitRequest(req)
    }

    private func requestCameraPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    cont.resume(returning: granted)
                }
            }
        default:
            return false
        }
    }

    private func presentFatal(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "GazeLock"
        alert.informativeText = message
        alert.alertStyle = .critical
        alert.addButton(withTitle: "Quit")
        alert.runModal()
        NSApp.terminate(nil)
    }

    private func presentInfo(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "GazeLock"
        alert.informativeText = message
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}

extension AppDelegate: NSWindowDelegate {
    func windowWillClose(_ notification: Notification) {
        if (notification.object as? NSWindow) === mainWindow {
            NSApp.setActivationPolicy(.accessory)
        }
    }
}

/// Bridge for `OSSystemExtensionRequest` callbacks during onboarding.
/// Not main-actor isolated; hops back to the main actor inside callbacks.
final class SystemExtensionDelegate: NSObject, OSSystemExtensionRequestDelegate, @unchecked Sendable {
    private let onNeedsApproval: @Sendable () -> Void
    private let onResult: @Sendable (Bool, String?) -> Void

    init(
        onNeedsApproval: @escaping @Sendable () -> Void,
        onResult: @escaping @Sendable (Bool, String?) -> Void
    ) {
        self.onNeedsApproval = onNeedsApproval
        self.onResult = onResult
    }

    func request(
        _ request: OSSystemExtensionRequest,
        actionForReplacingExtension existing: OSSystemExtensionProperties,
        withExtension ext: OSSystemExtensionProperties
    ) -> OSSystemExtensionRequest.ReplacementAction {
        .replace
    }

    func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
        onNeedsApproval()
    }

    func request(
        _ request: OSSystemExtensionRequest,
        didFinishWithResult result: OSSystemExtensionRequest.Result
    ) {
        switch result {
        case .completed:
            onResult(true, nil)
        case .willCompleteAfterReboot:
            onResult(true, "Installed — reboot required to finish.")
        @unknown default:
            onResult(false, "Unknown install result.")
        }
    }

    func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
        let ns = error as NSError
        // Map OSSystemExtensionErrorCode → actionable hint. Values come from
        // <SystemExtensions/OSSystemExtensionError.h>.
        let hint: String
        switch ns.code {
        case 2:
            hint = " — missing entitlement; rebuild with DEVELOPMENT_TEAM set"
        case 3:
            hint = " — app must live in /Applications; run `make run` from repo root"
        case 4:
            hint = " — bundle cache may be stale; run `make run` to refresh Launch Services"
        case 8:
            hint = " — invalid signature; rebuild with a real DEVELOPMENT_TEAM"
        case 10:
            hint = " — blocked by system policy; try `sudo systemextensionsctl reset`"
        case 11, 12:
            hint = " — request was superseded; retrying may succeed"
        default:
            hint = ""
        }
        onResult(false, "\(ns.localizedDescription)\(hint)")
    }
}

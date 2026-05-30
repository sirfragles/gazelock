import CoreVideo
import Foundation

/// End-to-end per-frame orchestrator. Wires every pipeline stage.
///
/// Usage:
///     let pipeline = try FramePipeline()
///     let output = try pipeline.process(pixelBuffer: cvpb, timestamp: ts, intensity: 0.7)
///
/// If no face is detected in the frame, returns the input unchanged.
/// If no Core ML refiner is available, runs warp-only (pure Approach A).
public final class FramePipeline {
    public enum Error: Swift.Error {
        case metalInitFailed
        case pixelBufferAllocationFailed
    }

    private let detector: LandmarkDetector
    private let warp: MetalWarpPipeline
    private let refiner: CoreMLRefiner?
    private let lookAway: LookAwayDetector
    private let eyeROISize = (w: 96, h: 72)

    public init(bundle: Bundle = .main) throws {
        self.detector = LandmarkDetector()
        do {
            self.warp = try MetalWarpPipeline()
        } catch {
            throw Error.metalInitFailed
        }
        self.refiner = try? CoreMLRefiner(bundle: bundle)
        self.lookAway = LookAwayDetector(sensitivity: .normal)
    }

    /// Process a single frame. `intensity` is the user's base slider value;
    /// the LookAwayDetector scales it based on head pose. Aim offsets shift
    /// the correction target within the image plane.
    public func process(
        pixelBuffer: CVPixelBuffer,
        timestamp: TimeInterval,
        intensity: Double,
        verticalAimDeg: Double = -2.0,
        horizontalAimDeg: Double = 0.0,
        sensitivity: Sensitivity = .normal
    ) throws -> CVPixelBuffer {
        guard let landmarks = try detector.detect(in: pixelBuffer, timestamp: timestamp) else {
            return pixelBuffer  // no face → passthrough
        }

        lookAway.sensitivity = sensitivity
        let alpha = lookAway.compute(headPose: landmarks.headPoseRadians, timestampSeconds: timestamp)
        let effectiveIntensity = intensity * alpha

        if effectiveIntensity < 0.01 {
            return pixelBuffer
        }

        let output = try copyOf(pixelBuffer)

        for eye in [landmarks.leftEye, landmarks.rightEye] {
            try warpEye(
                eye: eye,
                headPose: landmarks.headPoseRadians,
                intensity: effectiveIntensity,
                verticalAimDeg: verticalAimDeg,
                horizontalAimDeg: horizontalAimDeg,
                source: pixelBuffer,
                destination: output
            )
        }

        return output
    }

    private func warpEye(
        eye: EyeLandmarks,
        headPose: HeadPose,
        intensity: Double,
        verticalAimDeg: Double,
        horizontalAimDeg: Double,
        source: CVPixelBuffer,
        destination: CVPixelBuffer
    ) throws {
        let targetIris = EyeGeometry.targetIrisPx(
            eye: eye,
            headPose: headPose,
            verticalAimDeg: verticalAimDeg,
            horizontalAimDeg: horizontalAimDeg
        )
        let displacement = Vec2(
            (targetIris.x - eye.pupilCenter.x) * intensity,
            (targetIris.y - eye.pupilCenter.y) * intensity
        )
        let effectiveTarget = eye.pupilCenter + displacement

        // ROI origin in full-image coordinates. Clamp so the ROI stays
        // within the frame even for landmarks near an edge.
        let imgW = CVPixelBufferGetWidth(source)
        let imgH = CVPixelBufferGetHeight(source)
        let roiW = eyeROISize.w
        let roiH = eyeROISize.h
        let roiX = max(0, min(imgW - roiW, Int(eye.pupilCenter.x) - roiW / 2))
        let roiY = max(0, min(imgH - roiH, Int(eye.pupilCenter.y) - roiH / 2))

        // TPS control points + queries MUST be in full-image coordinates
        // because IrisWarp.metal samples by dividing `srcPx / texSize` where
        // texSize is the full input texture's width × height. If TPS runs in
        // ROI-local coords the flow values collapse to the top-left 96×72 px
        // of the source texture — showing as tiny white boxes over the
        // irises.
        let source4 = [
            Vec2(Double(roiX), Double(roiY)),
            Vec2(Double(roiX + roiW - 1), Double(roiY)),
            Vec2(Double(roiX), Double(roiY + roiH - 1)),
            Vec2(Double(roiX + roiW - 1), Double(roiY + roiH - 1)),
            eye.pupilCenter,
        ]
        var target5 = source4
        target5[4] = effectiveTarget

        let tps = ThinPlateSpline.fit(source: target5, target: source4)  // inverse mapping for sampling

        // Evaluate the TPS at each output pixel INSIDE the ROI, using full-image
        // query coordinates. Result = full-image source pixel to sample from.
        var queries: [Vec2] = []
        queries.reserveCapacity(roiW * roiH)
        for y in 0..<roiH {
            for x in 0..<roiW {
                queries.append(Vec2(Double(roiX + x), Double(roiY + y)))
            }
        }
        let evaluated = tps.evaluate(at: queries)
        let flow = FlowField(width: roiW, height: roiH, data: evaluated)

        try warp.apply(
            source: source,
            destination: destination,
            flow: flow,
            roiOrigin: (roiX, roiY)
        )

        // Refiner pass: smooth TPS seams + restore natural skin/iris texture.
        // The refiner takes (warped ++ original) 6-channel and emits a clean
        // 3-channel patch. If the model isn't bundled or inference fails, we
        // keep the raw warp output (caller already sees it in `destination`).
        guard let refiner else { return }
        do {
            let warpedArr = try RefinerIOBridge.extractROI(
                from: destination,
                roiOrigin: (roiX, roiY)
            )
            let originalArr = try RefinerIOBridge.extractROI(
                from: source,
                roiOrigin: (roiX, roiY)
            )
            if let refined = try refiner.refine(warpedEye: warpedArr, originalEye: originalArr) {
                let bgra = RefinerIOBridge.multiArrayToBGRA(refined)
                Compositor.compositeROI(
                    full: destination,
                    eyePatch: bgra,
                    roiOrigin: (roiX, roiY),
                    roiSize: (roiW, roiH)
                )
            }
        } catch {
            // Refiner errors are non-fatal — raw warp is still in destination.
        }
    }

    private func copyOf(_ pb: CVPixelBuffer) throws -> CVPixelBuffer {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out)
        guard let output = out else { throw Error.pixelBufferAllocationFailed }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }
        if let src = CVPixelBufferGetBaseAddress(pb),
           let dst = CVPixelBufferGetBaseAddress(output) {
            memcpy(dst, src, CVPixelBufferGetBytesPerRow(pb) * h)
        }
        return output
    }
}

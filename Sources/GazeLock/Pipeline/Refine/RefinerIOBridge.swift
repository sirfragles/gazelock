import CoreML
import CoreVideo
import Foundation

/// Pixel ↔ MLMultiArray conversion for the refiner.
///
/// Convention (mirrors the ML training pipeline in
/// `ML/gazelock_ml/data/dataset.py`):
/// * HWC uint8 BGR → CHW float32 in [0, 1]
/// * Shape (1, 3, EYE_ROI_H, EYE_ROI_W) — batch 1, three channels, 72×96
enum RefinerIOBridge {
    static let roiWidth = 96
    static let roiHeight = 72

    /// Extract an ROI from a BGRA CVPixelBuffer and pack it into a
    /// `(1, 3, H, W)` float32 MLMultiArray in [0, 1] with BGR channel order.
    static func extractROI(
        from buffer: CVPixelBuffer,
        roiOrigin: (x: Int, y: Int)
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 3, NSNumber(value: roiHeight), NSNumber(value: roiWidth)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)

        let fullW = CVPixelBufferGetWidth(buffer)
        let fullH = CVPixelBufferGetHeight(buffer)
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            return array
        }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let scale: Float32 = 1.0 / 255.0

        let plane = roiWidth * roiHeight
        let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: 3 * plane)

        for y in 0..<roiHeight {
            let srcY = min(max(0, roiOrigin.y + y), fullH - 1)
            let rowPtr = base.advanced(by: srcY * stride)
            for x in 0..<roiWidth {
                let srcX = min(max(0, roiOrigin.x + x), fullW - 1)
                let pix = rowPtr.advanced(by: srcX * 4)
                // CVPixelBuffer BGRA = byte order B, G, R, A.
                let b = Float32(pix.load(fromByteOffset: 0, as: UInt8.self)) * scale
                let g = Float32(pix.load(fromByteOffset: 1, as: UInt8.self)) * scale
                let r = Float32(pix.load(fromByteOffset: 2, as: UInt8.self)) * scale
                let idx = y * roiWidth + x
                pointer[0 * plane + idx] = b
                pointer[1 * plane + idx] = g
                pointer[2 * plane + idx] = r
            }
        }
        return array
    }

    /// Pack a `(1, 3, H, W)` float32 MLMultiArray back into tightly-packed
    /// BGRA uint8 for `Compositor.compositeROI`.
    static func multiArrayToBGRA(_ array: MLMultiArray) -> [UInt8] {
        let plane = roiWidth * roiHeight
        let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: 3 * plane)
        var out = [UInt8](repeating: 255, count: plane * 4)
        for y in 0..<roiHeight {
            for x in 0..<roiWidth {
                let idx = y * roiWidth + x
                let b = pointer[0 * plane + idx]
                let g = pointer[1 * plane + idx]
                let r = pointer[2 * plane + idx]
                let outIdx = idx * 4
                out[outIdx + 0] = clampToU8(b * 255.0)
                out[outIdx + 1] = clampToU8(g * 255.0)
                out[outIdx + 2] = clampToU8(r * 255.0)
                out[outIdx + 3] = 255
            }
        }
        return out
    }

    private static func clampToU8(_ value: Float32) -> UInt8 {
        if value.isNaN { return 0 }
        return UInt8(min(255, max(0, value.rounded())))
    }
}

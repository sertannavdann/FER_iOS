import Foundation
import CoreGraphics

class GeometryUtils {
    // Bounding box smoothing (EMA on x, y, width, height)
    private var smoothedFaceRect: CGRect = .zero
    private let faceRectAlpha: CGFloat = 0.3
    private let faceRectMargin: CGFloat = 0.2 // 20% expansion
    
    func expandAndSmoothRect(_ rect: CGRect) -> CGRect {
        // Expand by margin
        let expandX = rect.width * faceRectMargin / 2
        let expandY = rect.height * faceRectMargin / 2
        let expanded = CGRect(
            x: rect.minX - expandX,
            y: rect.minY - expandY,
            width: rect.width + faceRectMargin * rect.width,
            height: rect.height + faceRectMargin * rect.height
        )
        let expandedClamped = clampToUnit(expanded)
        
        // EMA smoothing
        if smoothedFaceRect == .zero {
            smoothedFaceRect = expandedClamped
        } else {
            smoothedFaceRect = CGRect(
                x: faceRectAlpha * expandedClamped.minX + (1 - faceRectAlpha) * smoothedFaceRect.minX,
                y: faceRectAlpha * expandedClamped.minY + (1 - faceRectAlpha) * smoothedFaceRect.minY,
                width: faceRectAlpha * expandedClamped.width + (1 - faceRectAlpha) * smoothedFaceRect.width,
                height: faceRectAlpha * expandedClamped.height + (1 - faceRectAlpha) * smoothedFaceRect.height
            )
            smoothedFaceRect = clampToUnit(smoothedFaceRect)
        }
        
        return smoothedFaceRect
    }

    private func clampToUnit(_ rect: CGRect) -> CGRect {
        var x = rect.origin.x
        var y = rect.origin.y
        var w = rect.size.width
        var h = rect.size.height

        // Ensure origin is not less than 0
        x = max(0, x)
        y = max(0, y)

        // Ensure width/height non-negative
        w = max(0, w)
        h = max(0, h)

        // Clamp to stay within [0,1] bounds
        if x + w > 1 { w = 1 - x }
        if y + h > 1 { h = 1 - y }

        return CGRect(x: x, y: y, width: w, height: h)
    }
    
    func reset() {
        smoothedFaceRect = .zero
    }
}

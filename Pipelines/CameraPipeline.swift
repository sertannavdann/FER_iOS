import Foundation
import AVFoundation
import CoreVideo
import ImageIO
import simd
import ARKit

protocol CameraPipeline: AnyObject {
    var id: String { get }
    var pipelineDescription: String { get }
    /// Callback with pixel buffer, detected faces, and the orientation used for Vision detection.
    /// The orientation should be applied to VNImageRequestHandler to ensure regionOfInterest aligns correctly.
    var onFrameCapture: ((CVPixelBuffer, [DetectedFace], CGImagePropertyOrientation) -> Void)? { get set }
    var onDepthCapture: ((CVPixelBuffer) -> Void)? { get set }
    var onFacesDetected: (([DetectedFace]) -> Void)? { get set }
    var previewLayer: AVCaptureVideoPreviewLayer? { get }
    var arSession: ARSession? { get }
    
    func start()
    func stop(completion: (() -> Void)?)
}

struct DetectedFace {
    let boundingBox: CGRect
    let depthMeters: Float?
    let yaw: Float?
    let pitch: Float?
    let roll: Float?
    
    // MARK: - 3D Position Data (from ARKit)
    /// World position of the face center in meters (x, y, z)
    let worldPosition: SIMD3<Float>?
    /// Full 4x4 transform matrix from ARKit
    let transform: simd_float4x4?
    /// Face geometry blend shapes (for detailed expression tracking)
    let blendShapes: [String: Float]?
    
    init(
        boundingBox: CGRect,
        depthMeters: Float? = nil,
        yaw: Float? = nil,
        pitch: Float? = nil,
        roll: Float? = nil,
        worldPosition: SIMD3<Float>? = nil,
        transform: simd_float4x4? = nil,
        blendShapes: [String: Float]? = nil
    ) {
        self.boundingBox = boundingBox
        self.depthMeters = depthMeters
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.worldPosition = worldPosition
        self.transform = transform
        self.blendShapes = blendShapes
    }
}

// MARK: - Helper Extensions
extension simd_float4x4 {
    /// Extract translation (position) from transform matrix
    var position: SIMD3<Float> {
        return SIMD3<Float>(columns.3.x, columns.3.y, columns.3.z)
    }
    
    /// Extract Euler angles from rotation matrix (in radians)
    var eulerAngles: SIMD3<Float> {
        let pitch = asin(-columns.2.x)
        let yaw: Float
        let roll: Float
        
        if cos(pitch) > 0.0001 {
            yaw = atan2(columns.2.y, columns.2.z)
            roll = atan2(columns.1.x, columns.0.x)
        } else {
            yaw = 0
            roll = atan2(-columns.0.y, columns.1.y)
        }
        
        return SIMD3<Float>(pitch, yaw, roll)
    }
}

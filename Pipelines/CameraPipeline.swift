import Foundation
import AVFoundation
import CoreVideo
import ImageIO

protocol CameraPipeline: AnyObject {
    var id: String { get }
    var pipelineDescription: String { get }
    /// Callback with pixel buffer, detected faces, and the orientation used for Vision detection.
    /// The orientation should be applied to VNImageRequestHandler to ensure regionOfInterest aligns correctly.
    var onFrameCapture: ((CVPixelBuffer, [DetectedFace], CGImagePropertyOrientation) -> Void)? { get set }
    var onDepthCapture: ((CVPixelBuffer) -> Void)? { get set }
    var onFacesDetected: (([DetectedFace]) -> Void)? { get set }
    var previewLayer: AVCaptureVideoPreviewLayer? { get }
    
    func start()
    func stop(completion: (() -> Void)?)
}

struct DetectedFace {
    let boundingBox: CGRect
    let depthMeters: Float?        // Keep optional for future Vision Pro
    let yaw: Float?                // Head rotation left/right (degrees)
    let pitch: Float?              // Head tilt up/down (degrees)
    let roll: Float?               // Head rotation clockwise/counter (degrees)
}

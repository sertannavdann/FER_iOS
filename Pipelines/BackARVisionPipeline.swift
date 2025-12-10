import Foundation
import AVFoundation
import Vision
import ARKit
import UIKit

/// Back camera pipeline using ARKit with LiDAR for accurate 3D face positioning
class BackARVisionPipeline: NSObject, CameraPipeline {
    let id = "BackARVision"
    var pipelineDescription: String {
        return "ARKit + LiDAR (back camera)"
    }

    var onFrameCapture: ((CVPixelBuffer, [DetectedFace], CGImagePropertyOrientation) -> Void)?
    var onDepthCapture: ((CVPixelBuffer) -> Void)?
    var onFacesDetected: (([DetectedFace]) -> Void)?

    private(set) var arSession: ARSession?
    private let faceLandmarksRequest = VNDetectFaceLandmarksRequest()
    private let processingQueue = DispatchQueue(label: "camera.processing.back", qos: .userInteractive)

    // Use atomic flag instead of semaphore to avoid frame retention
    private var isProcessing = false
    private let processingLock = NSLock()

    private(set) var previewLayer: AVCaptureVideoPreviewLayer? = nil

    // MARK: - Device Capability Checks

    static var isLiDARAvailable: Bool {
        return ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
    }

    static var isARKitAvailable: Bool {
        return ARWorldTrackingConfiguration.isSupported
    }

    override init() {
        super.init()
        setupARSession()
    }

    private func setupARSession() {
        guard Self.isARKitAvailable else {
            Log.info("ARKit not available on this device")
            return
        }

        arSession = ARSession()
        arSession?.delegate = self
    }

    private func createARConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()

        // Enable LiDAR depth if available
        if Self.isLiDARAvailable {
            config.frameSemantics.insert(.sceneDepth)
            config.frameSemantics.insert(.smoothedSceneDepth)
            Log.info("LiDAR depth enabled")
        }

        // Use highest quality video format
        if let hiResFormat = ARWorldTrackingConfiguration.supportedVideoFormats
            .filter({ $0.captureDevicePosition == .back })
            .max(by: { $0.imageResolution.width < $1.imageResolution.width }) {
            config.videoFormat = hiResFormat
            Log.info("Using video format: \(hiResFormat.imageResolution)")
        }

        // Optimize for face detection use case
        config.isAutoFocusEnabled = true
        config.environmentTexturing = .none
        config.planeDetection = []

        return config
    }

    func start() {
        guard let session = arSession else { return }

        let config = createARConfiguration()
        Log.info("Starting ARKit session with LiDAR: \(Self.isLiDARAvailable)")
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop(completion: (() -> Void)? = nil) {
        Log.info("Stopping ARKit session")
        arSession?.pause()
        completion?()
    }

    // MARK: - Depth Sampling & 3D Position Calculation

    /// Sample depth at a normalized point and calculate 3D world position
    private func depthAndPosition(at normalizedPoint: CGPoint, in frame: ARFrame) -> (depth: Float, worldPos: SIMD3<Float>)? {
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else {
            return nil
        }

        let camera = frame.camera
        let depthMap = depthData.depthMap
        let imageResolution = camera.imageResolution

        // Convert normalized Vision coordinates to pixel coordinates in camera image space
        // Vision uses normalized coords (0,0) = bottom-left, (1,1) = top-right
        let imagePoint = CGPoint(
            x: normalizedPoint.x * imageResolution.width,
            y: (1.0 - normalizedPoint.y) * imageResolution.height
        )

        // Sample depth from depth map
        let depthWidth = CVPixelBufferGetWidth(depthMap)
        let depthHeight = CVPixelBufferGetHeight(depthMap)

        // Map image coordinates to depth map coordinates
        let depthX = Int((imagePoint.x / imageResolution.width) * CGFloat(depthWidth))
        let depthY = Int((imagePoint.y / imageResolution.height) * CGFloat(depthHeight))

        // Clamp to valid range
        let clampedX = max(0, min(depthWidth - 1, depthX))
        let clampedY = max(0, min(depthHeight - 1, depthY))

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else {
            return nil
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        let floatBuffer = baseAddress.assumingMemoryBound(to: Float32.self)
        let index = clampedY * (bytesPerRow / MemoryLayout<Float32>.size) + clampedX
        let depth = floatBuffer[index]

        // Validate depth
        guard depth.isFinite && depth > 0.1 && depth < 10.0 else {
            return nil
        }

        // Unproject 2D point + depth to 3D world position
        let intrinsics = camera.intrinsics
        let fx = intrinsics[0, 0]
        let fy = intrinsics[1, 1]
        let cx = intrinsics[2, 0]
        let cy = intrinsics[2, 1]

        // Calculate 3D point in camera space
        let x = (Float(imagePoint.x) - cx) * depth / fx
        let y = (Float(imagePoint.y) - cy) * depth / fy
        let z = -depth  // Negative because ARKit Z points backward

        // Transform from camera space to world space
        let cameraSpacePoint = SIMD4<Float>(x, y, z, 1.0)
        let worldSpacePoint = camera.transform * cameraSpacePoint

        return (
            depth: depth,
            worldPos: SIMD3<Float>(worldSpacePoint.x, worldSpacePoint.y, worldSpacePoint.z)
        )
    }
}

// MARK: - ARSessionDelegate

extension BackARVisionPipeline: ARSessionDelegate {

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Use lock-based check to avoid frame retention
        processingLock.lock()
        if isProcessing {
            processingLock.unlock()
            return
        }
        isProcessing = true
        processingLock.unlock()

        // CRITICAL: Process synchronously in autoreleasepool to prevent ARFrame retention
        // Do NOT pass ARFrame to async blocks
        autoreleasepool {
            let pixelBuffer = frame.capturedImage
            let depthMap = (frame.smoothedSceneDepth ?? frame.sceneDepth)?.depthMap
            
            // ARKit back camera is in landscape right orientation
            let orientation: CGImagePropertyOrientation = .right

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
            try? handler.perform([faceLandmarksRequest])

            let observations = faceLandmarksRequest.results ?? []
            
            // Process faces synchronously while we still have access to frame
            let faces: [DetectedFace] = observations.compactMap { observation -> DetectedFace? in
                let bbox = observation.boundingBox
                let faceCenter = CGPoint(x: bbox.midX, y: bbox.midY)

                let yawValue = observation.yaw?.floatValue
                let pitchValue = observation.pitch?.floatValue
                let rollValue = observation.roll?.floatValue

                // Sample depth and calculate 3D position while frame is still valid
                guard let result = depthAndPosition(at: faceCenter, in: frame) else {
                    return DetectedFace(
                        boundingBox: bbox,
                        depthMeters: nil,
                        yaw: yawValue,
                        pitch: pitchValue,
                        roll: rollValue,
                        worldPosition: nil,
                        transform: nil,
                        blendShapes: nil
                    )
                }

                var transform = matrix_identity_float4x4
                transform.columns.3 = SIMD4<Float>(result.worldPos.x, result.worldPos.y, result.worldPos.z, 1.0)

                if let yaw = yawValue, let pitch = pitchValue, let roll = rollValue {
                    let rotMat = eulerToRotationMatrix(
                        pitch: CGFloat(pitch),
                        yaw: CGFloat(yaw),
                        roll: CGFloat(roll)
                    )
                    transform = transform * rotMat
                }

                return DetectedFace(
                    boundingBox: bbox,
                    depthMeters: result.depth,
                    yaw: yawValue,
                    pitch: pitchValue,
                    roll: rollValue,
                    worldPosition: result.worldPos,
                    transform: transform,
                    blendShapes: nil
                )
            }

            // Release lock before dispatching
            processingLock.lock()
            isProcessing = false
            processingLock.unlock()

            // Dispatch only extracted data to main queue
            DispatchQueue.main.async { [weak self] in
                if let depthMap = depthMap {
                    self?.onDepthCapture?(depthMap)
                }
                self?.onFacesDetected?(faces)
                self?.onFrameCapture?(pixelBuffer, faces, orientation)
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        Log.error("ARSession failed: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        Log.info("ARSession interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        Log.info("ARSession interruption ended, restarting...")
        start()
    }
}

// MARK: - Helper Functions

private func eulerToRotationMatrix(pitch: CGFloat, yaw: CGFloat, roll: CGFloat) -> simd_float4x4 {
    let p = Float(pitch)
    let y = Float(yaw)
    let r = Float(roll)

    let cp = cos(p), sp = sin(p)
    let cy = cos(y), sy = sin(y)
    let cr = cos(r), sr = sin(r)

    var matrix = matrix_identity_float4x4

    // ZYX Euler rotation
    matrix.columns.0 = SIMD4<Float>(cy * cr, cy * sr, -sy, 0)
    matrix.columns.1 = SIMD4<Float>(sp * sy * cr - cp * sr, sp * sy * sr + cp * cr, sp * cy, 0)
    matrix.columns.2 = SIMD4<Float>(cp * sy * cr + sp * sr, cp * sy * sr - sp * cr, cp * cy, 0)
    matrix.columns.3 = SIMD4<Float>(0, 0, 0, 1)

    return matrix
}

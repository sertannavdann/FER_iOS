import Foundation
import AVFoundation
import Vision
import ARKit
import UIKit

/// Front camera pipeline using ARKit with TrueDepth for accurate 3D face tracking
class FrontVisionPipeline: NSObject, CameraPipeline {
    let id = "FrontVision"
    var pipelineDescription: String {
        return usesARKit ? "ARKit Face Tracking (TrueDepth)" : "Vision (fallback)"
    }

    var onFrameCapture: ((CVPixelBuffer, [DetectedFace], CGImagePropertyOrientation) -> Void)?
    var onDepthCapture: ((CVPixelBuffer) -> Void)?
    var onFacesDetected: (([DetectedFace]) -> Void)?

    // ARKit components
    private(set) var arSession: ARSession?
    private var usesARKit: Bool = false

    // Fallback AVFoundation components
    private var avSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private let visionRequest = VNDetectFaceLandmarksRequest()

    private let processingQueue = DispatchQueue(label: "camera.processing.front", qos: .userInteractive)

    // Use atomic flag instead of semaphore to avoid frame retention
    private var isProcessing = false
    private let processingLock = NSLock()

    private(set) var previewLayer: AVCaptureVideoPreviewLayer? = nil

    // MARK: - Device Capability Checks

    static var isFaceTrackingSupported: Bool {
        return ARFaceTrackingConfiguration.isSupported
    }

    override init() {
        super.init()

        if Self.isFaceTrackingSupported {
            usesARKit = true
            setupARSession()
            Log.info("Using ARKit face tracking with TrueDepth")
        } else {
            usesARKit = false
            setupAVSession()
            Log.info("ARKit face tracking not available, using Vision fallback")
        }
    }

    // MARK: - ARKit Setup

    private func setupARSession() {
        arSession = ARSession()
        arSession?.delegate = self
        arSession?.delegateQueue = processingQueue
    }

    private func createFaceTrackingConfiguration() -> ARFaceTrackingConfiguration {
        let config = ARFaceTrackingConfiguration()

        config.isWorldTrackingEnabled = false
        config.maximumNumberOfTrackedFaces = 1

        // Use highest quality video format
        if let hiResFormat = ARFaceTrackingConfiguration.supportedVideoFormats
            .max(by: { $0.imageResolution.width < $1.imageResolution.width }) {
            config.videoFormat = hiResFormat
            Log.info("Using face tracking format: \(hiResFormat.imageResolution)")
        }

        return config
    }

    // MARK: - AVFoundation Fallback Setup

    private func setupAVSession() {
        guard avSession == nil else { return }

        let session = AVCaptureSession()
        session.beginConfiguration()
        session.sessionPreset = .high

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            Log.error("Failed to access front camera")
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: processingQueue)
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        if let connection = output.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
            } else {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
            connection.isVideoMirrored = true
        }

        session.commitConfiguration()

        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.videoGravity = .resizeAspectFill
        self.previewLayer = preview

        self.avSession = session
        self.videoOutput = output
    }

    // MARK: - CameraPipeline Protocol

    func start() {
        if usesARKit {
            startARSession()
        } else {
            startAVSession()
        }
    }

    func stop(completion: (() -> Void)? = nil) {
        if usesARKit {
            Log.info("Stopping ARKit face tracking")
            arSession?.pause()
            completion?()
        } else {
            processingQueue.async { [weak self] in
                if let session = self?.avSession, session.isRunning {
                    Log.info("Stopping AVCapture session")
                    session.stopRunning()
                }
                completion?()
            }
        }
    }

    private func startARSession() {
        guard let session = arSession else { return }

        let config = createFaceTrackingConfiguration()
        Log.info("Starting ARKit face tracking")
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    private func startAVSession() {
        processingQueue.async { [weak self] in
            guard let self = self, let session = self.avSession else { return }

            if !session.isRunning {
                Log.info("Starting AVCapture session")
                session.startRunning()

                if let connection = self.previewLayer?.connection {
                    if #available(iOS 17.0, *) {
                        if connection.isVideoRotationAngleSupported(90) {
                            connection.videoRotationAngle = 90
                        }
                    } else {
                        if connection.isVideoOrientationSupported {
                            connection.videoOrientation = .portrait
                        }
                    }
                }
            }
        }
    }
}

// MARK: - ARSessionDelegate

extension FrontVisionPipeline: ARSessionDelegate {

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Use lock-based check to avoid frame retention
        processingLock.lock()
        if isProcessing {
            processingLock.unlock()
            return
        }
        isProcessing = true
        processingLock.unlock()

        // CRITICAL: Extract all needed data from ARFrame IMMEDIATELY and synchronously
        // Copy pixel buffer (CVPixelBuffer is reference-counted separately)
        let pixelBuffer = frame.capturedImage
        
        // Extract face anchors immediately
        let faceAnchors = frame.anchors.compactMap { $0 as? ARFaceAnchor }
        
        // Process face data synchronously
        let faces: [DetectedFace] = faceAnchors.map { anchor in
                let transform = anchor.transform
                
                let position = SIMD3<Float>(
                    transform.columns.3.x,
                    transform.columns.3.y,
                    transform.columns.3.z
                )
                
                let depth = abs(position.z)
                
                let euler = extractEulerAngles(from: transform)
                let pitch = euler.x * 180.0 / Float.pi
                let yaw = euler.y * 180.0 / Float.pi
                let roll = euler.z * 180.0 / Float.pi

                // Calculate bounding box from face depth (closer faces = larger bbox)
                // Estimate bbox size dynamically instead of using hardcoded centered rect
                let depthScale = CGFloat(max(0.3, min(0.9, 1.0 / max(0.5, abs(position.z)))))
                let bboxSize = depthScale * 0.5  // Scale from 15% to 45% of frame

                // Center the square bbox
                let boundingBox = CGRect(
                    x: 0.5 - bboxSize / 2,
                    y: 0.5 - bboxSize / 2,
                    width: bboxSize,
                    height: bboxSize
                )
                
                var shapes: [String: Float] = [:]
                for (key, value) in anchor.blendShapes {
                    shapes[key.rawValue] = value.floatValue
                }
                
                return DetectedFace(
                    boundingBox: boundingBox,
                    depthMeters: depth,
                    yaw: yaw,
                    pitch: pitch,
                    roll: roll,
                    worldPosition: position,
                    transform: transform,
                    blendShapes: shapes
                )
            }
            
        // Release processing lock
        processingLock.lock()
        isProcessing = false
        processingLock.unlock()
        
        // Dispatch with extracted data (no frame references)
        DispatchQueue.main.async { [weak self] in
            self?.onFacesDetected?(faces)
            self?.onFrameCapture?(pixelBuffer, faces, .up)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        Log.error("ARSession failed: \(error.localizedDescription)")

        if usesARKit {
            Log.info("Falling back to AVFoundation")
            usesARKit = false
            setupAVSession()
            startAVSession()
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        Log.info("ARSession interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        Log.info("ARSession interruption ended, restarting...")
        startARSession()
    }

    // MARK: - Helper Methods

    private func extractEulerAngles(from matrix: simd_float4x4) -> SIMD3<Float> {
        // Extract Euler angles (XYZ order) from rotation matrix
        let sy = sqrt(matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0])

        let singular = sy < 1e-6

        let x: Float  // pitch
        let y: Float  // yaw
        let z: Float  // roll

        if !singular {
            x = atan2(matrix[2][1], matrix[2][2])
            y = atan2(-matrix[2][0], sy)
            z = atan2(matrix[1][0], matrix[0][0])
        } else {
            x = atan2(-matrix[1][2], matrix[1][1])
            y = atan2(-matrix[2][0], sy)
            z = 0
        }

        return SIMD3<Float>(x, y, z)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate (Fallback)

extension FrontVisionPipeline: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !usesARKit else { return }

        processingLock.lock()
        if isProcessing {
            processingLock.unlock()
            return
        }
        isProcessing = true
        processingLock.unlock()

        defer {
            processingLock.lock()
            isProcessing = false
            processingLock.unlock()
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([visionRequest])

        let observations = visionRequest.results ?? []
        let faces = observations.map { observation -> DetectedFace in
            DetectedFace(
                boundingBox: observation.boundingBox,
                depthMeters: nil,
                yaw: observation.yaw?.floatValue,
                pitch: observation.pitch?.floatValue,
                roll: observation.roll?.floatValue,
                worldPosition: nil,
                transform: nil,
                blendShapes: nil
            )
        }

        DispatchQueue.main.async { [weak self] in
            self?.onFacesDetected?(faces)
            self?.onFrameCapture?(pixelBuffer, faces, .up)
        }
    }
}

import Foundation
import AVFoundation
import Vision
import UIKit

class BackARVisionPipeline: NSObject, CameraPipeline {
    let id = "BackARVision"
    var pipelineDescription: String {
        return "Vision (back camera)"
    }
    
    var onFrameCapture: ((CVPixelBuffer, [DetectedFace], CGImagePropertyOrientation) -> Void)?
    var onDepthCapture: ((CVPixelBuffer) -> Void)?
    var onFacesDetected: (([DetectedFace]) -> Void)?
    
    private var fallbackSession: AVCaptureSession?
    private var fallbackOutput: AVCaptureVideoDataOutput?
    private let faceLandmarksRequest = VNDetectFaceLandmarksRequest()
    private let processingQueue = DispatchQueue(label: "camera.processing.back")
    private let processingSemaphore = DispatchSemaphore(value: 1)
    
    private(set) var previewLayer: AVCaptureVideoPreviewLayer?
    
    override init() {
        super.init()
        setupFallbackSession()
    }

    private func selectBackCamera() -> AVCaptureDevice? {
        let deviceTypes: [AVCaptureDevice.DeviceType] = [
            .builtInDualWideCamera,
            .builtInDualCamera,
            .builtInWideAngleCamera,
            .builtInTripleCamera
        ]
        let discovery = AVCaptureDevice.DiscoverySession(deviceTypes: deviceTypes, mediaType: .video, position: .back)
        return discovery.devices.first
    }
    
    private func setupFallbackSession() {
        guard fallbackSession == nil else { return }
        let session = AVCaptureSession()
        session.beginConfiguration()
        session.sessionPreset = .high
        
        guard let camera = selectBackCamera(),
              camera.supportsSessionPreset(session.sessionPreset),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("Failed to access back camera")
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
        
        // Keep video/depth buffers in sensor-native orientation (landscape)
        // Vision handles rotation via orientation parameter
        if let connection = output.connection(with: .video) {
            connection.isVideoMirrored = false
        }
        
        session.commitConfiguration()
        
        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.videoGravity = .resizeAspectFill
        // Set preview layer orientation for correct display
        if let previewConnection = preview.connection {
            if #available(iOS 17.0, *) {
                if previewConnection.isVideoRotationAngleSupported(90) {
                    previewConnection.videoRotationAngle = 90
                }
            } else {
                if previewConnection.isVideoOrientationSupported {
                    previewConnection.videoOrientation = .portrait
                }
            }
        }
        self.previewLayer = preview
        
        self.fallbackSession = session
        self.fallbackOutput = output
    }
    
    func start() {
        startFallbackSession()
    }
    
    func stop(completion: (() -> Void)? = nil) {
        processingQueue.async {
            if let session = self.fallbackSession, session.isRunning {
                Log.info("Stopping fallback AVCapture session")
                session.stopRunning()
            }
            // Release camera resources to avoid contention when switching pipelines.
            self.fallbackOutput = nil
            self.fallbackSession = nil
            self.previewLayer = nil
            completion?()
        }
    }

    private func startFallbackSession() {
        setupFallbackSession()
        processingQueue.async {
            if let session = self.fallbackSession, !session.isRunning {
                Log.info("Starting fallback AVCapture session")
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
extension BackARVisionPipeline: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Drop frame if processing is busy
        guard processingSemaphore.wait(timeout: .now() + 0.05) == .success else { return }
        defer { processingSemaphore.signal() }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // Sensor output is landscape; rotate right to present portrait to Vision
        let orientation: CGImagePropertyOrientation = .right
        
        autoreleasepool {
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
            try? handler.perform([faceLandmarksRequest])

            let observations = faceLandmarksRequest.results as? [VNFaceObservation] ?? []
            let faces = observations.map { observation -> DetectedFace in
                let rect = paddedRect(from: observation.boundingBox)

                // Extract orientation angles (convert radians to degrees)
                let yaw = observation.yaw.map { Float($0.floatValue) * 180.0 / .pi }
                let pitch = observation.pitch.map { Float($0.floatValue) * 180.0 / .pi }
                let roll = observation.roll.map { Float($0.floatValue) * 180.0 / .pi }

                return DetectedFace(
                    boundingBox: rect,
                    depthMeters: nil,  // Remove active depth estimation
                    yaw: yaw,
                    pitch: pitch,
                    roll: roll
                )
            }
            onFacesDetected?(faces)
            // Pass .right orientation to match the orientation used for Vision face detection
            onFrameCapture?(pixelBuffer, faces, orientation)
        }
    }
}

private extension BackARVisionPipeline {
    func paddedRect(from bbox: CGRect, padding: CGFloat = 0.08) -> CGRect {
        let expandX = bbox.width * padding / 2
        let expandY = bbox.height * padding / 2
        let expanded = CGRect(x: bbox.minX - expandX,
                              y: bbox.minY - expandY,
                              width: bbox.width + bbox.width * padding,
                              height: bbox.height + bbox.height * padding)
        return clampToUnit(expanded)
    }

    func clampToUnit(_ rect: CGRect) -> CGRect {
        var x = rect.origin.x
        var y = rect.origin.y
        var w = rect.size.width
        var h = rect.size.height

        x = max(0, x)
        y = max(0, y)
        w = max(0, w)
        h = max(0, h)
        if x + w > 1 { w = 1 - x }
        if y + h > 1 { h = 1 - y }
        return CGRect(x: x, y: y, width: w, height: h)
    }
}



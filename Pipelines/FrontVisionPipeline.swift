import Foundation
import AVFoundation
import Vision
import UIKit

class FrontVisionPipeline: NSObject, CameraPipeline {
    let id = "FrontVision"
    let pipelineDescription = "Vision"
    
    var onFrameCapture: ((CVPixelBuffer, [DetectedFace], CGImagePropertyOrientation) -> Void)?
    var onDepthCapture: ((CVPixelBuffer) -> Void)?
    var onFacesDetected: (([DetectedFace]) -> Void)?
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let faceDetectionRequest = VNDetectFaceLandmarksRequest()
    private let processingQueue = DispatchQueue(label: "camera.processing.front")
    private let geometryUtils = GeometryUtils()
    
    private(set) var previewLayer: AVCaptureVideoPreviewLayer?
    
    override init() {
        super.init()
        setupSession()
    }
    
    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .high
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("Failed to access front camera")
            session.commitConfiguration()
            return
        }
        
        if session.canAddInput(input) {
            session.addInput(input)
        }
        
        if !session.outputs.contains(videoOutput) {
            videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            }
        }
        
        if let connection = videoOutput.connection(with: .video) {
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
    }
    
    func start() {
        processingQueue.async {
            if !self.session.isRunning {
                Log.info("Starting front AVCapture session")
                self.session.startRunning()
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
    
    func stop(completion: (() -> Void)? = nil) {
        processingQueue.async {
            if self.session.isRunning {
                Log.info("Stopping front AVCapture session")
                self.session.stopRunning()
            }
            completion?()
        }
    }
}

extension FrontVisionPipeline: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([faceDetectionRequest])

        let faces = (faceDetectionRequest.results as? [VNFaceObservation] ?? []).map { observation in
            // Extract orientation angles (convert radians to degrees)
            let yaw = observation.yaw.map { Float($0.floatValue) * 180.0 / .pi }
            let pitch = observation.pitch.map { Float($0.floatValue) * 180.0 / .pi }
            let roll = observation.roll.map { Float($0.floatValue) * 180.0 / .pi }

            return DetectedFace(
                boundingBox: geometryUtils.expandAndSmoothRect(observation.boundingBox),
                depthMeters: nil,
                yaw: yaw,
                pitch: pitch,
                roll: roll
            )
        }
        onFacesDetected?(faces)
        // Buffer is already rotated to portrait via videoRotationAngle, so use .up
        onFrameCapture?(pixelBuffer, faces, .up)
    }
}

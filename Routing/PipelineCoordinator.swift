import Foundation
import AVFoundation
import SwiftUI
import Combine
import ImageIO

class PipelineCoordinator: ObservableObject {
    @Published var previewLayer: AVCaptureVideoPreviewLayer?
    @Published var faces: [DetectedFace] = []
    @Published var isBackCamera: Bool = false
    @Published var isARKitActive: Bool = false
    @Published var pipelineInfo: String = "Initializing..."
    
    private var frontPipeline: FrontVisionPipeline?
    private var backPipeline: BackARVisionPipeline?
    private var activePipeline: CameraPipeline?
    
    // Frame callback for inference - includes orientation for Vision alignment
    var onFrameCapture: ((CVPixelBuffer, [DetectedFace], CGImagePropertyOrientation) -> Void)?
    var onDepthCapture: ((CVPixelBuffer) -> Void)?
    
    init() {
        // Initialize pipelines lazily or upfront? 
        // Let's initialize front first as it's default
        setupFrontPipeline()
    }
    
    private func setupFrontPipeline() {
        if frontPipeline == nil {
            frontPipeline = FrontVisionPipeline()
        }
        Log.info("Switching to front pipeline")
        activatePipeline(frontPipeline!)
        isBackCamera = false
    }
    
    private func setupBackPipeline() {
        if backPipeline == nil {
            backPipeline = BackARVisionPipeline()
        }
        Log.info("Switching to back pipeline")
        activatePipeline(backPipeline!)
        isBackCamera = true
    }
    
    private func activatePipeline(_ pipeline: CameraPipeline) {
        let startNewPipeline = { [weak self] in
            DispatchQueue.main.async {
                guard let self = self else { return }
                
                self.activePipeline = pipeline
                Log.debug("Starting pipeline: \(pipeline.id)")
                
                // Wire up callbacks
                pipeline.onFrameCapture = { [weak self] buffer, faces, orientation in
                    self?.onFrameCapture?(buffer, faces, orientation)
                }
                
                pipeline.onDepthCapture = { [weak self] buffer in
                    self?.onDepthCapture?(buffer)
                }
                
                pipeline.onFacesDetected = { [weak self] faces in
                    DispatchQueue.main.async {
                        self?.faces = faces
                    }
                }
                
                pipeline.start()
                
                self.previewLayer = pipeline.previewLayer
                self.pipelineInfo = pipeline.pipelineDescription
                self.isARKitActive = false
            }
        }

        if let active = activePipeline {
            Log.debug("Stopping active pipeline: \(active.id)")
            active.onFrameCapture = nil
            active.onFacesDetected = nil
            active.onDepthCapture = nil
            active.stop {
                Log.debug("Stopped pipeline: \(active.id)")
                // Start new pipeline after stop completes to avoid overlapping sessions.
                DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.3) {
                    startNewPipeline()
                }
            }
        } else {
            startNewPipeline()
        }
    }
    
    func switchCamera() {
        if isBackCamera {
            setupFrontPipeline()
        } else {
            setupBackPipeline()
        }
    }
    
    func start() {
        activePipeline?.start()
    }
    
    func stop() {
        activePipeline?.stop(completion: nil)
    }
}

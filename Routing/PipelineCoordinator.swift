import Foundation
import Foundation
import AVFoundation
import SwiftUI
import Combine
import ImageIO
import ARKit

@MainActor
class PipelineCoordinator: ObservableObject, PipelineCoordinatorProtocol {
    @Published var previewLayer: AVCaptureVideoPreviewLayer?
    @Published var arSession: ARSession?
    @Published var faces: [DetectedFace] = []
    @Published var isBackCamera: Bool = false
    @Published var isARKitActive: Bool = false
    @Published var pipelineInfo: String = "Initializing..."
    @Published var arGraphManager: ARGraphSurfaceManager?

    private var frontPipeline: FrontVisionPipeline?
    private var backPipeline: BackARVisionPipeline?
    private var activePipeline: CameraPipeline?
    private var isPaused: Bool = false

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
        arGraphManager?.remove()
        arGraphManager = nil
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
                self.arSession = pipeline.arSession
                self.pipelineInfo = pipeline.pipelineDescription
                self.isARKitActive = (pipeline.arSession != nil)

                // Create AR graph manager when AR session is available (back camera)
                if let session = pipeline.arSession {
                    self.arGraphManager = ARGraphSurfaceManager(arSession: session)
                } else {
                    self.arGraphManager?.remove()
                    self.arGraphManager = nil
                }
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
        arGraphManager?.cleanup()
    }

    // MARK: - PipelineCoordinatorProtocol

    func pause() {
        guard !isPaused else { return }
        isPaused = true
        Log.info("[PipelineCoordinator] Pausing active pipeline")
        activePipeline?.stop(completion: nil)
    }

    func resume() {
        guard isPaused else { return }
        isPaused = false
        Log.info("[PipelineCoordinator] Resuming active pipeline")
        activePipeline?.start()
    }
}

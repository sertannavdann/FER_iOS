import Foundation
import ARKit
import RealityKit
import Metal
import simd

/// Manages the 3D AR graph surface rendering
/// Coordinates texture generation, entity updates, and positioning
@MainActor
class ARGraphSurfaceManager {

    // MARK: - Properties

    private let arSession: ARSession
    private var anchorEntity: AnchorEntity?
    private var graphEntity: ProbabilityGraphEntity?
    private var textureGenerator: GraphTextureGenerator?

    private var isSetup: Bool = false
    private var lastUpdateTime: Date = .distantPast
    private let targetFPS: Double = 15.0  // Match UI update rate

    // MARK: - Initialization

    init(arSession: ARSession) {
        self.arSession = arSession
        Log.info("[ARGraphSurface] Manager initialized")
    }

    // MARK: - Setup

    /// Setup the AR graph surface in the given ARView
    /// Must be called before update()
    func setup(in arView: ARView) {
        guard !isSetup else {
            Log.warn("[ARGraphSurface] Already setup, skipping")
            return
        }

        // Create anchor entity
        let anchor = AnchorEntity(world: .zero)
        arView.scene.addAnchor(anchor)
        anchorEntity = anchor

        // Create graph entity
        let entity = ProbabilityGraphEntity()
        anchor.addChild(entity)
        graphEntity = entity

        // Create texture generator
        textureGenerator = GraphTextureGenerator(size: CGSize(width: 512, height: 512))
        guard textureGenerator != nil else {
            Log.error("[ARGraphSurface] Failed to create texture generator")
            return
        }

        isSetup = true
        Log.info("[ARGraphSurface] Setup complete")
    }

    // MARK: - Update

    /// Update the AR graph surface with new data
    /// - Parameters:
    ///   - history: Probability history array
    ///   - faceTransform: Face 4x4 transformation matrix (nil if no face detected)
    ///   - settings: Current inference settings
    func update(history: [[Float]], faceTransform: simd_float4x4?, settings: InferenceSettings) {
        guard isSetup else {
            Log.warn("[ARGraphSurface] Not setup, call setup() first")
            return
        }

        guard let graphEntity = graphEntity else { return }

        // Check if AR graph is enabled
        guard settings.enableARGraph else {
            graphEntity.hide()
            return
        }

        // Handle face detection state immediately (no throttle) to honor hysteresis
        guard let transform = faceTransform else {
            graphEntity.handleFaceLost()
            return
        }

        // Throttle heavy work to target FPS
        let now = Date()
        let minInterval = 1.0 / targetFPS
        guard now.timeIntervalSince(lastUpdateTime) >= minInterval else {
            return
        }
        lastUpdateTime = now

        // Get current camera transform from AR session
        guard let currentFrame = arSession.currentFrame else {
            return
        }
        let cameraTransform = currentFrame.camera.transform

        // Generate texture from probability history
          if let textureGenerator = textureGenerator,
              let texture = textureGenerator.render(history: history) {
            graphEntity.updateTexture(texture)
        }

        // Update entity configuration from settings
        let entityConfig = ProbabilityGraphEntity.Configuration(
            baseWidth: settings.graphSurfaceBaseWidth,
            baseHeight: settings.graphSurfaceBaseHeight,
            offsetMeters: settings.graphSurfaceOffsetMeters,
            scalingFactor: settings.graphSurfaceScalingFactor,
            minSize: settings.graphSurfaceMinSize,
            maxSize: settings.graphSurfaceMaxSize,
            smoothingFactor: 0.15,
            side: settings.graphSurfaceSide
        )
        graphEntity.updateConfiguration(entityConfig)

        // Update position and scale based on face transform
        graphEntity.updatePosition(faceTransform: transform, cameraTransform: cameraTransform)
    }

    // MARK: - Cleanup

    /// Cleanup resources and hide the graph
    func cleanup() {
        graphEntity?.hide()
        textureGenerator?.clearCache()
        Log.info("[ARGraphSurface] Cleaned up")
    }

    /// Remove all entities from the scene
    func remove() {
        anchorEntity?.removeFromParent()
        anchorEntity = nil
        graphEntity = nil
        textureGenerator = nil
        isSetup = false
        Log.info("[ARGraphSurface] Removed from scene")
    }
}

import Foundation
import RealityKit
import Metal
import simd
import CoreImage

/// RealityKit entity that displays the probability graph as a 3D surface
/// Positioned perpendicular to the detected face with distance-based scaling
class ProbabilityGraphEntity: Entity {

    // MARK: - Configuration

    struct Configuration {
        var baseWidth: Float = 0.3         // Base width in meters (30cm)
        var baseHeight: Float = 0.4        // Base height in meters (40cm)
        var offsetMeters: Float = 0.5      // Distance from face center (50cm)
        var scalingFactor: Float = 1.0     // Scaling multiplier
        var minSize: Float = 0.2           // Minimum scale (20cm)
        var maxSize: Float = 0.8           // Maximum scale (80cm)
        var smoothingFactor: Float = 0.15  // EMA smoothing for position/scale
        var side: GraphSide = .right       // Left or right of face (from InferenceSettings)
    }

    var config: Configuration

    // MARK: - Private Properties

    private var planeEntity: ModelEntity?
    private var smoothedPosition: SIMD3<Float>?
    private var smoothedScale: Float?
    private var lastAppliedPosition: SIMD3<Float>?
    private var lastAppliedScale: Float?
    private var faceLostTime: Date?
    private let faceLostHysteresis: TimeInterval = 0.3  // 300ms before hiding

    private var currentMaterial: UnlitMaterial?

    // MARK: - Initialization

    init(config: Configuration = Configuration()) {
        self.config = config
        super.init()
        setupPlane()
    }

    required init() {
        self.config = Configuration()
        super.init()
        setupPlane()
    }

    // MARK: - Setup

    private func setupPlane() {
        // Create plane mesh
        let mesh = MeshResource.generatePlane(
            width: config.baseWidth,
            height: config.baseHeight
        )

        // Create default material (will be updated with texture later)
        var material = UnlitMaterial()
        material.color = .init(tint: .white)
        currentMaterial = material

        // Create model entity
        let entity = ModelEntity(mesh: mesh, materials: [material])
        entity.isEnabled = false  // Start hidden
        planeEntity = entity

        addChild(entity)
        Log.debug("[GraphEntity] Plane created with size: \(config.baseWidth)x\(config.baseHeight)m")
    }

    // MARK: - Texture Update

    /// Update the graph texture on the surface
    func updateTexture(_ texture: MTLTexture) {
        guard var material = currentMaterial else { return }

        // Convert Metal texture to CGImage then to TextureResource
        guard let cgImage = makeCGImage(from: texture) else {
            Log.error("[GraphEntity] Failed to create CGImage from texture")
            return
        }

        do {
            let textureResource = try TextureResource.generate(from: cgImage, options: .init(semantic: .color))
            material.color = .init(texture: .init(textureResource))
            currentMaterial = material

            planeEntity?.model?.materials = [material]
        } catch {
            Log.error("[GraphEntity] Failed to create texture resource: \(error)")
        }
    }

    // MARK: - Position Update

    /// Update surface position and scale based on face transform
    /// - Parameters:
    ///   - faceTransform: 4x4 transformation matrix of the face
    ///   - cameraTransform: 4x4 transformation matrix of the camera
    func updatePosition(faceTransform: simd_float4x4, cameraTransform: simd_float4x4) {
        // Reset face lost timer
        faceLostTime = nil

        // Extract face position and rotation
        let facePos = simd_make_float3(faceTransform.columns.3)
        let faceRotation = simd_quatf(faceTransform)

        // Calculate perpendicular offset position
        let faceRight = faceRotation.act(SIMD3<Float>(1, 0, 0))
        let offsetDirection = config.side == .left ? -faceRight : faceRight
        let targetPosition = facePos + offsetDirection * config.offsetMeters

        // Calculate distance from camera for scaling
        let cameraPos = simd_make_float3(cameraTransform.columns.3)
        let distance = simd_distance(cameraPos, facePos)

        // Calculate scale (linear scaling with distance)
        let referenceDistance: Float = 1.0  // Reference: 1 meter
        let targetScale = (distance / referenceDistance) * config.scalingFactor
        let clampedScale = max(config.minSize, min(config.maxSize, targetScale))

        // Apply EMA smoothing to position
        if let smoothed = smoothedPosition {
            smoothedPosition = mix(smoothed, targetPosition, t: config.smoothingFactor)
        } else {
            smoothedPosition = targetPosition
        }

        // Apply EMA smoothing to scale
        if let smoothed = smoothedScale {
            smoothedScale = mix(smoothed, clampedScale, t: config.smoothingFactor)
        } else {
            smoothedScale = clampedScale
        }

        // Apply to entity with delta thresholds to avoid micro updates
        guard let finalPos = smoothedPosition, let finalScale = smoothedScale else { return }

        let positionDelta = lastAppliedPosition.map { simd_distance($0, finalPos) } ?? .infinity
        let scaleDelta = lastAppliedScale.map { abs($0 - finalScale) / max($0, 0.0001) } ?? .infinity

        // Skip tiny updates (<1cm position and <5% scale)
        guard positionDelta >= 0.01 || scaleDelta >= 0.05 else { return }

        // Update position
        planeEntity?.position = finalPos
        lastAppliedPosition = finalPos

        // Update scale (uniform scaling)
        planeEntity?.scale = SIMD3<Float>(repeating: finalScale)
        lastAppliedScale = finalScale

        // Billboard mode: always face the camera
        let toCamera = simd_normalize(cameraPos - finalPos)
        let up = SIMD3<Float>(0, 1, 0)
        let right = simd_normalize(simd_cross(up, toCamera))
        let actualUp = simd_cross(toCamera, right)
        
        // Construct look-at rotation matrix
        var lookMatrix = matrix_identity_float4x4
        lookMatrix.columns.0 = SIMD4<Float>(right.x, right.y, right.z, 0)
        lookMatrix.columns.1 = SIMD4<Float>(actualUp.x, actualUp.y, actualUp.z, 0)
        lookMatrix.columns.2 = SIMD4<Float>(toCamera.x, toCamera.y, toCamera.z, 0)
        lookMatrix.columns.3 = SIMD4<Float>(0, 0, 0, 1)
        
        planeEntity?.orientation = simd_quatf(lookMatrix)
        
        Log.debug("[GraphEntity] Updated position: \(finalPos), scale: \(finalScale), distance: \(simd_distance(cameraPos, facePos))m")

        // Show entity if hidden
        if planeEntity?.isEnabled == false {
            show()
        }
    }

    /// Handle face lost scenario with hysteresis
    func handleFaceLost() {
        let now = Date()

        if faceLostTime == nil {
            // First frame without face - start hysteresis timer
            faceLostTime = now
            return
        }

        // Check if hysteresis period has passed
        guard let lostTime = faceLostTime,
              now.timeIntervalSince(lostTime) >= faceLostHysteresis else {
            return  // Still within hysteresis - keep last known position
        }

        // Hysteresis passed - face is truly lost
        hide()
    }

    // MARK: - Visibility

    /// Show the surface with fade-in animation
    func show() {
        planeEntity?.isEnabled = true
        Log.debug("[GraphEntity] Surface shown")
    }

    /// Hide the surface with fade-out animation
    func hide() {
        planeEntity?.isEnabled = false
        smoothedPosition = nil
        smoothedScale = nil
        faceLostTime = nil
        lastAppliedPosition = nil
        lastAppliedScale = nil
        Log.debug("[GraphEntity] Surface hidden")
    }

    // MARK: - Configuration Updates

    /// Update configuration (will recreate plane if size changed)
    func updateConfiguration(_ newConfig: Configuration) {
        let needsRecreate = newConfig.baseWidth != config.baseWidth ||
                           newConfig.baseHeight != config.baseHeight

        config = newConfig

        if needsRecreate {
            planeEntity?.removeFromParent()
            setupPlane()
        }
    }
}

// MARK: - Helper Functions

private let sharedCIContext = CIContext()

private func makeCGImage(from texture: MTLTexture) -> CGImage? {
    guard let ciImage = CIImage(mtlTexture: texture, options: nil) else {
        return nil
    }
    return sharedCIContext.createCGImage(ciImage, from: ciImage.extent)
}

/// Linear interpolation between two float values
private func mix(_ a: Float, _ b: Float, t: Float) -> Float {
    return a * (1 - t) + b * t
}

/// Linear interpolation between two float3 vectors
private func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, t: Float) -> SIMD3<Float> {
    return a * (1 - t) + b * t
}

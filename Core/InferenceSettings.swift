import SwiftUI

// MARK: - ML Model Selection
enum MLModelType: String, Codable, CaseIterable, Identifiable {
    case ferModelFP32 = "FER_Model_FP32"
    case ferMobileNetV2 = "FER_MobileNetV2"
    case ferMobileNetV2FP32 = "FER_MobileNetV2_FP32"
    case ferModel = "FER_Model"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .ferModelFP32: return "FER Model (FP32)"
        case .ferMobileNetV2: return "MobileNetV2"
        case .ferMobileNetV2FP32: return "MobileNetV2 (FP32)"
        case .ferModel: return "FER Model"
        }
    }
}

// MARK: - AR Graph Side Selection
enum GraphSide: String, Codable, CaseIterable, Identifiable {
    case left = "Left"
    case right = "Right"

    var id: String { rawValue }
}

// MARK: - Inference Settings (persisted)
struct InferenceSettings: Codable, RawRepresentable, Equatable {
    // FER Model Settings
    let ringBufferSize: Int
    let emaAlpha: Double
    let neutralBoost: Double
    let framesForAverage: Int
    let faceExpansionRatio: Double
    let showFaceRect: Bool
    let selectedModel: MLModelType

    // AR Graph Surface Settings
    let enableARGraph: Bool
    let graphSurfaceOffsetMeters: Float
    let graphSurfaceSide: GraphSide
    let graphSurfaceBaseWidth: Float
    let graphSurfaceBaseHeight: Float
    let graphSurfaceScalingFactor: Float
    let graphSurfaceMinSize: Float
    let graphSurfaceMaxSize: Float
    let enableScreenRecording: Bool  // Allow screen recording in AR mode
    let enableLiDAR: Bool  // Enable LiDAR depth estimation (back camera only)

    init(
        ringBufferSize: Int = 60,
        emaAlpha: Double = 0.1,
        neutralBoost: Double = 2.0,
        framesForAverage: Int = 30,
        faceExpansionRatio: Double = 0.2,
        showFaceRect: Bool = true,
        selectedModel: MLModelType = .ferModelFP32,
        enableARGraph: Bool = false,  // Disabled by default to reduce overhead
        graphSurfaceOffsetMeters: Float = 0.5,
        graphSurfaceSide: GraphSide = .right,
        graphSurfaceBaseWidth: Float = 0.3,
        graphSurfaceBaseHeight: Float = 0.4,
        graphSurfaceScalingFactor: Float = 1.0,
        graphSurfaceMinSize: Float = 0.2,
        graphSurfaceMaxSize: Float = 0.8,
        enableScreenRecording: Bool = false,
        enableLiDAR: Bool = true
    ) {
        self.ringBufferSize = ringBufferSize
        self.emaAlpha = emaAlpha
        self.neutralBoost = neutralBoost
        self.framesForAverage = framesForAverage
        self.faceExpansionRatio = faceExpansionRatio
        self.showFaceRect = showFaceRect
        self.selectedModel = selectedModel
        self.enableARGraph = enableARGraph
        self.graphSurfaceOffsetMeters = graphSurfaceOffsetMeters
        self.graphSurfaceSide = graphSurfaceSide
        self.graphSurfaceBaseWidth = graphSurfaceBaseWidth
        self.graphSurfaceBaseHeight = graphSurfaceBaseHeight
        self.graphSurfaceScalingFactor = graphSurfaceScalingFactor
        self.graphSurfaceMinSize = graphSurfaceMinSize
        self.graphSurfaceMaxSize = graphSurfaceMaxSize
        self.enableScreenRecording = enableScreenRecording
        self.enableLiDAR = enableLiDAR
    }

    init?(rawValue: Data) {
        guard let decoded = try? JSONDecoder().decode(InferenceSettings.self, from: rawValue) else { return nil }
        self = decoded
    }

    var rawValue: Data {
        (try? JSONEncoder().encode(self)) ?? Data()
    }

    static func == (lhs: InferenceSettings, rhs: InferenceSettings) -> Bool {
        lhs.ringBufferSize == rhs.ringBufferSize &&
        lhs.emaAlpha == rhs.emaAlpha &&
        lhs.neutralBoost == rhs.neutralBoost &&
        lhs.framesForAverage == rhs.framesForAverage &&
        lhs.faceExpansionRatio == rhs.faceExpansionRatio &&
        lhs.showFaceRect == rhs.showFaceRect &&
        lhs.selectedModel == rhs.selectedModel &&
        lhs.enableARGraph == rhs.enableARGraph &&
        lhs.graphSurfaceOffsetMeters == rhs.graphSurfaceOffsetMeters &&
        lhs.graphSurfaceSide == rhs.graphSurfaceSide &&
        lhs.graphSurfaceBaseWidth == rhs.graphSurfaceBaseWidth &&
        lhs.graphSurfaceBaseHeight == rhs.graphSurfaceBaseHeight &&
        lhs.graphSurfaceScalingFactor == rhs.graphSurfaceScalingFactor &&
        lhs.graphSurfaceMinSize == rhs.graphSurfaceMinSize &&
        lhs.graphSurfaceMaxSize == rhs.graphSurfaceMaxSize &&
        lhs.enableScreenRecording == rhs.enableScreenRecording &&
        lhs.enableLiDAR == rhs.enableLiDAR
    }

    // Convenience copy-with helper to avoid reconstructing manually
    func updating(
        ringBufferSize: Int? = nil,
        emaAlpha: Double? = nil,
        neutralBoost: Double? = nil,
        framesForAverage: Int? = nil,
        faceExpansionRatio: Double? = nil,
        showFaceRect: Bool? = nil,
        selectedModel: MLModelType? = nil,
        enableARGraph: Bool? = nil,
        graphSurfaceOffsetMeters: Float? = nil,
        graphSurfaceSide: GraphSide? = nil,
        graphSurfaceBaseWidth: Float? = nil,
        graphSurfaceBaseHeight: Float? = nil,
        graphSurfaceScalingFactor: Float? = nil,
        graphSurfaceMinSize: Float? = nil,
        graphSurfaceMaxSize: Float? = nil,
        enableScreenRecording: Bool? = nil,
        enableLiDAR: Bool? = nil
    ) -> InferenceSettings {
        InferenceSettings(
            ringBufferSize: ringBufferSize ?? self.ringBufferSize,
            emaAlpha: emaAlpha ?? self.emaAlpha,
            neutralBoost: neutralBoost ?? self.neutralBoost,
            framesForAverage: framesForAverage ?? self.framesForAverage,
            faceExpansionRatio: faceExpansionRatio ?? self.faceExpansionRatio,
            showFaceRect: showFaceRect ?? self.showFaceRect,
            selectedModel: selectedModel ?? self.selectedModel,
            enableARGraph: enableARGraph ?? self.enableARGraph,
            graphSurfaceOffsetMeters: graphSurfaceOffsetMeters ?? self.graphSurfaceOffsetMeters,
            graphSurfaceSide: graphSurfaceSide ?? self.graphSurfaceSide,
            graphSurfaceBaseWidth: graphSurfaceBaseWidth ?? self.graphSurfaceBaseWidth,
            graphSurfaceBaseHeight: graphSurfaceBaseHeight ?? self.graphSurfaceBaseHeight,
            graphSurfaceScalingFactor: graphSurfaceScalingFactor ?? self.graphSurfaceScalingFactor,
            graphSurfaceMinSize: graphSurfaceMinSize ?? self.graphSurfaceMinSize,
            graphSurfaceMaxSize: graphSurfaceMaxSize ?? self.graphSurfaceMaxSize,
            enableScreenRecording: enableScreenRecording ?? self.enableScreenRecording,
            enableLiDAR: enableLiDAR ?? self.enableLiDAR
        )
    }
}


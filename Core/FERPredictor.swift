import SwiftUI
import Vision
import CoreML
import ImageIO
import simd
import CoreImage

// MARK: - FER Predictor using Core ML
class FERPredictor: ObservableObject {
    private var model: VNCoreMLModel?
    private var request: VNCoreMLRequest?
    @Published var faceOutputs: [FacePrediction] = []
    @Published var probabilityHistory: [[Float]] = []
    
    private var smoothers: [Int: TemporalSmoother] = [:]
    private var currentSettings: InferenceSettings
    private let historyLimit = 75  // 5 seconds at 15fps
    
    init(settings: InferenceSettings) {
        self.currentSettings = settings
        loadModel()
        update(settings: settings)
    }

    func update(settings: InferenceSettings) {
        currentSettings = settings
        // Reset smoothers with new settings for each active face
        smoothers = smoothers.mapValues { _ in TemporalSmoother(settings: settings) }
    }
    
    private func loadModel() {
        // Try to load the model from the app bundle
        guard let modelURL = Bundle.main.url(forResource: "FER_Model", withExtension: "mlmodelc") ??
                             Bundle.main.url(forResource: "FER_Model", withExtension: "mlpackage") else {
            print("Model not found in bundle")
            return
        }
        
        do {
            let compiledURL: URL
            if modelURL.pathExtension == "mlpackage" || modelURL.pathExtension == "mlmodel" {
                compiledURL = try MLModel.compileModel(at: modelURL)
            } else {
                compiledURL = modelURL
            }
            
            let mlModel = try MLModel(contentsOf: compiledURL)
            model = try VNCoreMLModel(for: mlModel)
            
            let coreRequest = VNCoreMLRequest(model: model!)
            coreRequest.imageCropAndScaleOption = .scaleFill
            request = coreRequest
            
            print("Model loaded successfully")
        } catch {
            print("Error loading model: \(error)")
        }
    }
    
    func predict(pixelBuffer: CVPixelBuffer, faces: [DetectedFace], orientation: CGImagePropertyOrientation = .up) {
        guard let request = request else { return }
        guard let targetFace = faces.max(by: { $0.boundingBox.area < $1.boundingBox.area }) else {
            DispatchQueue.main.async {
                self.faceOutputs = []
            }
            return
        }
        
        ensureSmoothers(count: 1)
        
        // Clamp bounding box to [0, 1] to avoid Vision errors
        let bbox = targetFace.boundingBox
        let clampedRect = CGRect(
            x: max(0, bbox.minX),
            y: max(0, bbox.minY),
            width: min(1.0 - max(0, bbox.minX), bbox.width),
            height: min(1.0 - max(0, bbox.minY), bbox.height)
        )
        request.regionOfInterest = clampedRect

        // Explicitly convert to grayscale to match C++ implementation (RGB -> Gray -> RGB)
        // This ensures the model receives the expected feature distribution
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let grayscale = ciImage.applyingFilter("CIPhotoEffectMono")
        let handler = VNImageRequestHandler(ciImage: grayscale, orientation: orientation, options: [:])

        do {
            try handler.perform([request])
        } catch {
            print("Prediction error: \(error)")
            return
        }

        guard let raw = extractProbabilities(from: request) else { return }
        let smoothed = smoothers[0]?.smooth(raw) ?? raw

        guard let maxIdx = smoothed.indices.max(by: { smoothed[$0] < smoothed[$1] }) else { return }
        
        let prediction = FacePrediction(
            boundingBox: targetFace.boundingBox,
            depthMeters: targetFace.depthMeters,
            probabilities: smoothed,
            dominantEmotion: emotionClasses[maxIdx],
            dominantEmoji: emotionEmojis[maxIdx],
            confidence: smoothed[maxIdx],
            yaw: targetFace.yaw,
            pitch: targetFace.pitch,
            roll: targetFace.roll,
            worldPosition: targetFace.worldPosition,
            transform: targetFace.transform,
            blendShapes: targetFace.blendShapes
        )

        var updatedHistory = probabilityHistory
        updatedHistory.append(smoothed)
        if updatedHistory.count > historyLimit {
            updatedHistory.removeFirst()
        }

        DispatchQueue.main.async {
            self.probabilityHistory = updatedHistory
            self.faceOutputs = [prediction]
        }
    }
    
    private func extractProbabilities(from request: VNCoreMLRequest) -> [Float]? {
        if let results = request.results as? [VNCoreMLFeatureValueObservation],
           let first = results.first,
           let multiArray = first.featureValue.multiArrayValue {
            // Assume MultiArray output is logits (raw scores) -> apply softmax
            let logits = extractProbabilities(from: multiArray)
            guard logits.count == 7 else { return nil }
            return softmax(logits)
        } else if let results = request.results as? [VNClassificationObservation] {
            // Classification observations are already probabilities (confidence)
            return classificationToProbabilities(results)
        }
        return nil
    }
    
    private func extractProbabilities(from multiArray: MLMultiArray) -> [Float] {
        var result = [Float]()
        for i in 0..<multiArray.count {
            result.append(Float(truncating: multiArray[i]))
        }
        return result
    }
    
    private func classificationToProbabilities(_ observations: [VNClassificationObservation]) -> [Float] {
        var probs = [Float](repeating: 0, count: 7)
        for obs in observations {
            if let idx = emotionClasses.firstIndex(of: obs.identifier.lowercased()) {
                probs[idx] = obs.confidence
            }
        }
        return probs
    }
    
    private func softmax(_ logits: [Float]) -> [Float] {
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }
    
    func reset() {
        smoothers = [:]
    }

    private func ensureSmoothers(count: Int) {
        if smoothers.count != count {
            var new: [Int: TemporalSmoother] = [:]
            for idx in 0..<count {
                new[idx] = TemporalSmoother(settings: currentSettings)
            }
            smoothers = new
        }
    }
}

private extension CGRect {
    var area: CGFloat { width * height }
}

struct FacePrediction: Identifiable {
    let id = UUID()
    let boundingBox: CGRect
    let depthMeters: Float?
    let probabilities: [Float]
    let dominantEmotion: String
    let dominantEmoji: String
    let confidence: Float
    let yaw: Float?
    let pitch: Float?
    let roll: Float?
    
    // MARK: - 3D Position Data (from ARKit)
    /// World position of the face center in meters (x, y, z)
    let worldPosition: SIMD3<Float>?
    /// Full 4x4 transform matrix from ARKit
    let transform: simd_float4x4?
    /// Face geometry blend shapes (for detailed expression tracking)
    let blendShapes: [String: Float]?
    
    init(
        boundingBox: CGRect,
        depthMeters: Float?,
        probabilities: [Float],
        dominantEmotion: String,
        dominantEmoji: String,
        confidence: Float,
        yaw: Float?,
        pitch: Float?,
        roll: Float?,
        worldPosition: SIMD3<Float>? = nil,
        transform: simd_float4x4? = nil,
        blendShapes: [String: Float]? = nil
    ) {
        self.boundingBox = boundingBox
        self.depthMeters = depthMeters
        self.probabilities = probabilities
        self.dominantEmotion = dominantEmotion
        self.dominantEmoji = dominantEmoji
        self.confidence = confidence
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.worldPosition = worldPosition
        self.transform = transform
        self.blendShapes = blendShapes
    }
}

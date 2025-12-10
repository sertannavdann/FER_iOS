import SwiftUI

// MARK: - Inference Settings (persisted)
struct InferenceSettings: Codable, RawRepresentable, Equatable {
    let ringBufferSize: Int
    let emaAlpha: Double
    let neutralBoost: Double
    let framesForAverage: Int
    let show3DModel: Bool

    init(ringBufferSize: Int = 3, emaAlpha: Double = 0.15, neutralBoost: Double = 2.0, framesForAverage: Int = 3, show3DModel: Bool = true) {
        self.ringBufferSize = ringBufferSize
        self.emaAlpha = emaAlpha
        self.neutralBoost = neutralBoost
        self.framesForAverage = framesForAverage
        self.show3DModel = show3DModel
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
        lhs.show3DModel == rhs.show3DModel
    }
}

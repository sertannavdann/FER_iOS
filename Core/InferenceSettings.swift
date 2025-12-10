import SwiftUI

// MARK: - Inference Settings (persisted)
struct InferenceSettings: Codable, RawRepresentable, Equatable {
    let ringBufferSize: Int
    let emaAlpha: Double
    let neutralBoost: Double
    let framesForAverage: Int

    init(ringBufferSize: Int = 60, emaAlpha: Double = 0.1, neutralBoost: Double = 2.0, framesForAverage: Int = 30) {
        self.ringBufferSize = ringBufferSize
        self.emaAlpha = emaAlpha
        self.neutralBoost = neutralBoost
        self.framesForAverage = framesForAverage
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
        lhs.framesForAverage == rhs.framesForAverage
    }
}

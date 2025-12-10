import Foundation

// MARK: - Temporal Smoothing
class TemporalSmoother {
    private var emaState: [Float] = []
    private var history: [[Float]] = []
    private var settings: InferenceSettings
    private let neutralIndex = 4 // neutral is index 4 in the correct class order
    
    init(settings: InferenceSettings) {
        self.settings = settings
    }
    
    func update(settings: InferenceSettings) {
        self.settings = settings
    }
    
    func smooth(_ probabilities: [Float]) -> [Float] {
        // Boost neutral and renormalize
        var adjusted = probabilities
        if adjusted.count > neutralIndex {
            adjusted[neutralIndex] *= Float(settings.neutralBoost)
        }
        let sum = adjusted.reduce(0, +)
        if sum > 0 {
            adjusted = adjusted.map { $0 / sum }
        }
        
        // EMA smoothing
        if emaState.isEmpty {
            emaState = adjusted
        } else {
            for i in 0..<adjusted.count {
                emaState[i] = Float(settings.emaAlpha) * adjusted[i] + (1 - Float(settings.emaAlpha)) * emaState[i]
            }
        }
        
        // Add to history with configurable ring buffer size
        history.append(emaState)
        if history.count > settings.ringBufferSize {
            history.removeFirst()
        }
        
        return meanFromRecentFrames()
    }
    
    private func meanFromRecentFrames() -> [Float] {
        guard !history.isEmpty, let first = history.first else { return emaState }
        let window = max(1, min(settings.framesForAverage, history.count))
        var sums = [Float](repeating: 0, count: first.count)
        let start = history.count - window
        for i in start..<history.count {
            let frame = history[i]
            for c in 0..<frame.count {
                sums[c] += frame[c]
            }
        }
        let denom = Float(window)
        return sums.map { $0 / denom }
    }
    
    func reset() {
        emaState = []
        history = []
    }
}

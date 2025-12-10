import Foundation

// MARK: - Temporal Smoothing
class TemporalSmoother {
    private var emaState: [Float] = []
    private var history: [[Float]] = []
    private var settings: InferenceSettings
    private let neutralIndex = 3 // neutral is index 3 in the correct class order
    
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
        
        return medianFromRecentFrames()
    }
    
    private func medianFromRecentFrames() -> [Float] {
        guard !history.isEmpty, let first = history.first else { return emaState }
        let window = max(1, min(settings.framesForAverage, history.count))
        let start = history.count - window
        
        var medians = [Float](repeating: 0, count: first.count)
        
        for c in 0..<first.count {
            var values: [Float] = []
            for i in start..<history.count {
                if c < history[i].count {
                    values.append(history[i][c])
                }
            }
            values.sort()
            if !values.isEmpty {
                medians[c] = values[values.count / 2]
            }
        }
        return medians
    }
    
    func reset() {
        emaState = []
        history = []
    }
}

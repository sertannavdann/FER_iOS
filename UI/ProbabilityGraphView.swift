import SwiftUI

/// Draws a simple probability history graph.
/// `history[frameIndex][emotionIndex]` in range [0, 1].
struct ProbabilityGraphView: View {
    let history: [[Float]]

    var body: some View {
        Canvas { context, size in
            drawGraph(context: &context, size: size)
        }
        .frame(minHeight: 120)
    }

    // MARK: - Drawing

    private func drawGraph(context: inout GraphicsContext, size: CGSize) {
        guard !history.isEmpty else { return }

        let frameCount = history.count
        let emotionCount = history.first?.count ?? 0
        guard emotionCount > 0 else { return }

        let sectionWidth = size.width / CGFloat(max(emotionCount, 1))

        // Limit history to last N frames to keep drawing cheap.
        let maxFramesToDraw = min(frameCount, 120)
        let startIndex = frameCount - maxFramesToDraw
        let stepDenominator = max(1, maxFramesToDraw - 1)

        for emotionIndex in 0..<emotionCount {
            let xOffset = CGFloat(emotionIndex) * sectionWidth
            var path = Path()

            for (i, frameIndex) in (startIndex..<frameCount).enumerated() {
                let probs = history[frameIndex]
                guard emotionIndex < probs.count else { continue }

                let p = max(0, min(1, CGFloat(probs[emotionIndex])))
                let localX = CGFloat(i) / CGFloat(stepDenominator) * sectionWidth
                let y = size.height * (1 - p)
                let point = CGPoint(x: xOffset + localX, y: y)

                if i == 0 {
                    path.move(to: point)
                } else {
                    path.addLine(to: point)
                }
            }

            let color = colorForEmotion(at: emotionIndex)
            context.stroke(path, with: .color(color), lineWidth: 2)
        }
    }

    // MARK: - Helpers

    private func colorForEmotion(at index: Int) -> Color {
        // Simple fixed palette; index > palette.count falls back to gray.
        let palette: [Color] = [
            .red, .orange, .yellow, .green,
            .blue, .purple, .pink
        ]
        return index < palette.count ? palette[index] : .gray
    }
}

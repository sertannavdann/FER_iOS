import SwiftUI

/// Timeline graph with 7 horizontal lanes showing emotion probability history
/// Each lane represents one emotion, with newest values on the right
struct ProbabilityTimelineGraph: View {
    let history: [[Float]]           // Array of frames, each frame has 7 probabilities
    let maxFramesToShow: Int = 75    // Show last 75 frames

    private let laneHeight: CGFloat = 30
    private let laneSpacing: CGFloat = 4
    private let margin: CGFloat = 8

    var body: some View {
        Canvas { context, size in
            guard !history.isEmpty else { return }

            let numLanes = min(emotionClasses.count, emotionColors.count)
            let totalHeight = CGFloat(numLanes) * (laneHeight + laneSpacing) - laneSpacing
            let startY = (size.height - totalHeight) / 2

            // Determine which frames to display (last N frames)
            let framesToShow = Array(history.suffix(maxFramesToShow))
            let numFrames = framesToShow.count
            guard numFrames > 0 else { return }

            // Calculate x-axis scaling
            let graphWidth = size.width - 2 * margin
            let frameWidth = graphWidth / CGFloat(max(numFrames - 1, 1))

            // Find dominant emotion for each frame (for glow highlighting)
            let dominantIndices = framesToShow.map { frame -> Int? in
                frame.indices.max(by: { frame[$0] < frame[$1] })
            }

            // Draw each emotion lane
            for laneIdx in 0..<numLanes {
                let laneY = startY + CGFloat(laneIdx) * (laneHeight + laneSpacing)
                let laneRect = CGRect(x: margin, y: laneY, width: graphWidth, height: laneHeight)

                // Background for lane
                context.fill(
                    Path(roundedRect: laneRect, cornerRadius: 4),
                    with: .color(Color.gray.opacity(0.1))
                )

                // Draw probability curve for this emotion
                var path = Path()
                var glowPath = Path()

                for (frameIdx, frame) in framesToShow.enumerated() {
                    guard laneIdx < frame.count else { continue }

                    let probability = CGFloat(frame[laneIdx]).clamped(to: 0...1)
                    let x = margin + CGFloat(frameIdx) * frameWidth

                    // Y position: top of lane = 1.0 probability, bottom = 0.0
                    let y = laneY + laneHeight * (1.0 - probability)

                    let point = CGPoint(x: x, y: y)
                    if frameIdx == 0 {
                        path.move(to: point)
                        glowPath.move(to: point)
                    } else {
                        path.addLine(to: point)
                        glowPath.addLine(to: point)
                    }
                }

                // Highlight if this emotion is dominant in recent frames
                let isRecentlyDominant = dominantIndices.suffix(5).contains(laneIdx)

                if isRecentlyDominant {
                    // Glow effect for dominant emotion
                    context.stroke(
                        glowPath,
                        with: .color(emotionColors[laneIdx].opacity(0.3)),
                        style: StrokeStyle(lineWidth: 8, lineCap: .round, lineJoin: .round)
                    )
                }

                // Main probability curve
                context.stroke(
                    path,
                    with: .color(emotionColors[laneIdx]),
                    style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round)
                )

                // Emotion label
                let label = Text(emotionClasses[laneIdx].capitalized)
                    .font(.caption2)
                    .foregroundColor(.primary)
                context.draw(
                    label,
                    at: CGPoint(x: margin + 4, y: laneY + 4),
                    anchor: .topLeading
                )

                // Current probability value on the right
                if let lastFrame = framesToShow.last, laneIdx < lastFrame.count {
                    let valueText = Text(String(format: "%.0f%%", lastFrame[laneIdx] * 100))
                        .font(.caption2)
                        .foregroundColor(emotionColors[laneIdx])
                    context.draw(
                        valueText,
                        at: CGPoint(x: size.width - margin - 4, y: laneY + laneHeight - 4),
                        anchor: .bottomTrailing
                    )
                }
            }
        }
        .frame(height: CGFloat(emotionClasses.count) * (laneHeight + laneSpacing) + 2 * margin)
        .drawingGroup()  // Rasterize for better performance
    }
}

private extension CGFloat {
    func clamped(to range: ClosedRange<CGFloat>) -> CGFloat {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}

#Preview {
    ProbabilityTimelineGraph(
        history: Array(repeating: [0.2, 0.4, 0.8, 0.1, 0.6, 0.3, 0.5], count: 50)
    )
    .padding()
    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
}

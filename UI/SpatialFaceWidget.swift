import SwiftUI

/// Unified spatial widget that displays above detected face
/// Combines 3D face model and probability timeline graph
struct SpatialFaceWidget: View {
    let prediction: FacePrediction?
    let history: [[Float]]
    let geometry: GeometryProxy
    let settings: InferenceSettings
    let isARGraphActive: Bool

    @State private var smoothedPosition: CGPoint = .zero
    private let positionAlpha: CGFloat = 0.2

    var body: some View {
        Group {
            if isARGraphActive {
                EmptyView()
            } else if let prediction = prediction {
                let widgetFrame = calculateWidgetFrame(for: prediction, in: geometry.size)

                ZStack {
                    if settings.showFaceRect {
                        let faceRect = convertedRect(prediction.boundingBox, in: geometry.size)
                        let roiRect = convertedRect(prediction.inferenceROI, in: geometry.size)

                        Rectangle()
                            .stroke(Color.green, lineWidth: 2)
                            .frame(width: faceRect.width, height: faceRect.height)
                            .position(x: faceRect.midX, y: faceRect.midY)

                        Rectangle()
                            .stroke(Color.yellow.opacity(0.8), style: StrokeStyle(lineWidth: 1, dash: [5]))
                            .frame(width: roiRect.width, height: roiRect.height)
                            .position(x: roiRect.midX, y: roiRect.midY)
                    }

                    VStack(spacing: 12) {
                        ProbabilityTimelineGraph(history: history)
                            .frame(width: widgetFrame.width)
                            .padding(8)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                            .shadow(color: .black.opacity(0.2), radius: 4, x: 0, y: 2)
                    }
                    .position(smoothedPosition)
                    .transition(.opacity.combined(with: .scale(scale: 0.8)))
                }
                .onAppear {
                    smoothedPosition = CGPoint(x: widgetFrame.midX, y: widgetFrame.midY)
                }
                .onChange(of: prediction.boundingBox) { _, _ in
                    updateSmoothedPosition(to: CGPoint(x: widgetFrame.midX, y: widgetFrame.midY))
                }
            } else {
                EmptyView()
            }
        }
    }

    /// Calculate widget frame based on face bounding box
    private func calculateWidgetFrame(for prediction: FacePrediction, in size: CGSize) -> CGRect {
        let faceRect = convertedRect(prediction.boundingBox, in: size)

        // Size: scales with face height * 1.5, clamped to min/max
        let baseWidth = faceRect.height * 1.5
        let width = min(max(baseWidth, 180), 300)

        // Height depends on whether 3D model is shown
        let graphHeight: CGFloat = 240  // Fixed graph height
        let height = graphHeight + 8

        // Position: top-center of face, offset upward by half widget height
        let x = faceRect.midX  // Center horizontally on face
        let y = faceRect.minY - height / 2 - 20  // Offset above face with 20pt margin

        // Clamp to screen bounds
        let clampedX = max(width / 2, min(x, size.width - width / 2))
        let clampedY = max(height / 2, min(y, size.height - height / 2))

        return CGRect(x: clampedX, y: clampedY, width: width, height: height)
    }

    /// Convert Vision normalized rect (origin bottom-left) to UIKit coordinates (origin top-left)
    private func convertedRect(_ normalizedRect: CGRect, in size: CGSize) -> CGRect {
        let x = normalizedRect.minX * size.width
        let y = (1 - normalizedRect.maxY) * size.height
        let width = normalizedRect.width * size.width
        let height = normalizedRect.height * size.height
        return CGRect(x: x, y: y, width: width, height: height)
    }

    /// EMA smoothing for widget position (alpha=0.2)
    private func updateSmoothedPosition(to newPosition: CGPoint) {
        // Only update if position changed by >2pt (avoid micro-jitter)
        let delta = hypot(newPosition.x - smoothedPosition.x, newPosition.y - smoothedPosition.y)
        guard delta > 2.0 || smoothedPosition == .zero else { return }

        withAnimation(.easeInOut(duration: 0.15)) {
            if smoothedPosition == .zero {
                smoothedPosition = newPosition
            } else {
                smoothedPosition = CGPoint(
                    x: positionAlpha * newPosition.x + (1 - positionAlpha) * smoothedPosition.x,
                    y: positionAlpha * newPosition.y + (1 - positionAlpha) * smoothedPosition.y
                )
            }
        }
    }
}

#Preview {
    GeometryReader { geometry in
        ZStack {
            Color.black.ignoresSafeArea()

            SpatialFaceWidget(
                prediction: FacePrediction(
                    boundingBox: CGRect(x: 0.3, y: 0.4, width: 0.4, height: 0.3),
                    inferenceROI: CGRect(x: 0.2, y: 0.3, width: 0.6, height: 0.5),
                    depthMeters: nil,
                    probabilities: [0.1, 0.2, 0.15, 0.7, 0.05, 0.1, 0.2],
                    dominantEmotion: "happy",
                    dominantEmoji: "😊",
                    confidence: 0.7,
                    yaw: 15,
                    pitch: -10,
                    roll: 5
                ),
                history: Array(repeating: [0.2, 0.4, 0.8, 0.1, 0.6, 0.3, 0.5], count: 50),
                geometry: geometry,
                settings: InferenceSettings(),
                isARGraphActive: false
            )
        }
    }
}

import SwiftUI
import AVFoundation
import ARKit
import RealityKit

// MARK: - AR Session View (RealityKit)
struct ARSessionView: UIViewRepresentable {
    let session: ARSession
    let graphManager: ARGraphSurfaceManager?

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        view.automaticallyConfigureSession = false  // Session is managed by pipeline
        view.environment.sceneUnderstanding.options = []
        view.renderOptions.insert(.disableMotionBlur)
        view.renderOptions.insert(.disableDepthOfField)
        view.session = session
        
        // Enable video recording if requested
        view.cameraMode = .ar

        // Attach AR graph surface once the ARView is ready
        graphManager?.setup(in: view)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        if uiView.session !== session {
            uiView.session = session
            graphManager?.setup(in: uiView)
        }
    }
}

// MARK: - Camera Preview View
struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer?
    
    class Coordinator {
        var currentLayer: AVCaptureVideoPreviewLayer?
    }
    
    func makeCoordinator() -> Coordinator { Coordinator() }
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            let newLayer = previewLayer
            let current = context.coordinator.currentLayer
            
            // Remove previous preview layer if it differs
            if let existing = current, existing !== newLayer {
                existing.removeFromSuperlayer()
                context.coordinator.currentLayer = nil
            }
            
            guard let layer = newLayer else { return }
            layer.frame = uiView.bounds
            if layer.superlayer !== uiView.layer {
                uiView.layer.insertSublayer(layer, at: 0)
            }
            context.coordinator.currentLayer = layer
        }
    }
}

extension Collection {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

// MARK: - Settings Sheet
struct SettingsSheet: View {
    @Binding var settings: InferenceSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            Form {
                Section("Model") {
                    Picker("ML Model", selection: Binding(
                        get: { settings.selectedModel },
                        set: { settings = settings.updating(selectedModel: $0) }
                    )) {
                        ForEach(MLModelType.allCases) { model in
                            Text(model.displayName).tag(model)
                        }
                    }
                }

                Section("Smoothing") {
                    Stepper("Ring Buffer: \(settings.ringBufferSize)", value: Binding(
                        get: { settings.ringBufferSize },
                        set: { settings = settings.updating(ringBufferSize: $0) }
                    ), in: 1...10)

                    Slider(value: Binding(
                        get: { settings.emaAlpha },
                        set: { settings = settings.updating(emaAlpha: $0) }
                    ), in: 0.05...0.5) {
                        Text("EMA Alpha")
                    }
                    Text(String(format: "EMA α: %.2f", settings.emaAlpha))

                    Slider(value: Binding(
                        get: { settings.neutralBoost },
                        set: { settings = settings.updating(neutralBoost: $0) }
                    ), in: 1.0...3.0) {
                        Text("Neutral Boost")
                    }
                    Text(String(format: "Neutral boost: %.2f×", settings.neutralBoost))

                    Stepper("Frames for Avg: \(settings.framesForAverage)", value: Binding(
                        get: { settings.framesForAverage },
                        set: { settings = settings.updating(framesForAverage: $0) }
                    ), in: 1...10)

                    Slider(value: Binding(
                        get: { settings.faceExpansionRatio },
                        set: { settings = settings.updating(faceExpansionRatio: $0) }
                    ), in: 0.0...0.5) {
                        Text("Face Expansion")
                    }
                    Text(String(format: "Face Expansion: %.0f%%", settings.faceExpansionRatio * 100))

                    Toggle("Show Face Rect", isOn: Binding(
                        get: { settings.showFaceRect },
                        set: { settings = settings.updating(showFaceRect: $0) }
                    ))
                }

                Section("AR Graph") {
                    Toggle("Enable AR Graph", isOn: Binding(
                        get: { settings.enableARGraph },
                        set: { settings = settings.updating(enableARGraph: $0) }
                    ))

                    Picker("Graph Side", selection: Binding(
                        get: { settings.graphSurfaceSide },
                        set: { settings = settings.updating(graphSurfaceSide: $0) }
                    )) {
                        ForEach(GraphSide.allCases) { side in
                            Text(side.rawValue).tag(side)
                        }
                    }

                    Slider(value: Binding(
                        get: { Double(settings.graphSurfaceOffsetMeters) },
                        set: { settings = settings.updating(graphSurfaceOffsetMeters: Float($0)) }
                    ), in: 0.2...1.5) {
                        Text("Offset from Face (m)")
                    }
                    Text(String(format: "Offset: %.2fm", settings.graphSurfaceOffsetMeters))

                    Slider(value: Binding(
                        get: { Double(settings.graphSurfaceBaseWidth) },
                        set: { settings = settings.updating(graphSurfaceBaseWidth: Float($0)) }
                    ), in: 0.2...0.6) {
                        Text("Base Width (m)")
                    }
                    Text(String(format: "Width: %.2fm", settings.graphSurfaceBaseWidth))

                    Slider(value: Binding(
                        get: { Double(settings.graphSurfaceBaseHeight) },
                        set: { settings = settings.updating(graphSurfaceBaseHeight: Float($0)) }
                    ), in: 0.2...0.8) {
                        Text("Base Height (m)")
                    }
                    Text(String(format: "Height: %.2fm", settings.graphSurfaceBaseHeight))

                    Slider(value: Binding(
                        get: { Double(settings.graphSurfaceScalingFactor) },
                        set: { settings = settings.updating(graphSurfaceScalingFactor: Float($0)) }
                    ), in: 0.5...2.0) {
                        Text("Scaling Factor")
                    }
                    Text(String(format: "Scale: %.2fx", settings.graphSurfaceScalingFactor))

                    Slider(value: Binding(
                        get: { Double(settings.graphSurfaceMinSize) },
                        set: { settings = settings.updating(graphSurfaceMinSize: Float($0)) }
                    ), in: 0.1...0.4) {
                        Text("Min Size (m)")
                    }
                    Text(String(format: "Min: %.2fm", settings.graphSurfaceMinSize))

                    Slider(value: Binding(
                        get: { Double(settings.graphSurfaceMaxSize) },
                        set: { settings = settings.updating(graphSurfaceMaxSize: Float($0)) }
                    ), in: 0.4...1.2) {
                        Text("Max Size (m)")
                    }
                    Text(String(format: "Max: %.2fm", settings.graphSurfaceMaxSize))
                }

                Section("LiDAR Depth (Back Camera)") {
                    Toggle("Enable LiDAR", isOn: Binding(
                        get: { settings.enableLiDAR },
                        set: { settings = settings.updating(enableLiDAR: $0) }
                    ))

                    Text("Disables depth estimation and 3D positioning to improve performance.")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    if !settings.enableLiDAR && settings.enableARGraph {
                        HStack {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.orange)
                            Text("AR Graph requires LiDAR")
                                .font(.caption)
                        }
                    }
                }

                Section("AR Debug") {
                    Toggle("Enable Screen Recording", isOn: Binding(
                        get: { settings.enableScreenRecording },
                        set: { settings = settings.updating(enableScreenRecording: $0) }
                    ))
                    Text("Allows video recording of AR view (may impact performance)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Inference Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

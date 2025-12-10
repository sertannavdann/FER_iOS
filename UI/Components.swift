import SwiftUI
import AVFoundation
import ARKit

// MARK: - AR Session View
struct ARSessionView: UIViewRepresentable {
    let session: ARSession
    
    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.automaticallyUpdatesLighting = true
        return view
    }
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {
        if uiView.session != session {
            uiView.session = session
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
                Section("Smoothing") {
                    Stepper("Ring Buffer: \(settings.ringBufferSize)", value: Binding(
                        get: { settings.ringBufferSize },
                        set: { settings = InferenceSettings(ringBufferSize: $0, emaAlpha: settings.emaAlpha, neutralBoost: settings.neutralBoost, framesForAverage: settings.framesForAverage, show3DModel: settings.show3DModel) }
                    ), in: 1...10)

                    Slider(value: Binding(
                        get: { settings.emaAlpha },
                        set: { settings = InferenceSettings(ringBufferSize: settings.ringBufferSize, emaAlpha: $0, neutralBoost: settings.neutralBoost, framesForAverage: settings.framesForAverage, show3DModel: settings.show3DModel) }
                    ), in: 0.05...0.5) {
                        Text("EMA Alpha")
                    }
                    Text(String(format: "EMA α: %.2f", settings.emaAlpha))

                    Slider(value: Binding(
                        get: { settings.neutralBoost },
                        set: { settings = InferenceSettings(ringBufferSize: settings.ringBufferSize, emaAlpha: settings.emaAlpha, neutralBoost: $0, framesForAverage: settings.framesForAverage, show3DModel: settings.show3DModel) }
                    ), in: 1.0...3.0) {
                        Text("Neutral Boost")
                    }
                    Text(String(format: "Neutral boost: %.2f×", settings.neutralBoost))

                    Stepper("Frames for Avg: \(settings.framesForAverage)", value: Binding(
                        get: { settings.framesForAverage },
                        set: { settings = InferenceSettings(ringBufferSize: settings.ringBufferSize, emaAlpha: settings.emaAlpha, neutralBoost: settings.neutralBoost, framesForAverage: $0, show3DModel: settings.show3DModel) }
                    ), in: 1...10)
                }

                Section("Visualization") {
                    Toggle("Show 3D Face Model", isOn: Binding(
                        get: { settings.show3DModel },
                        set: { settings = InferenceSettings(ringBufferSize: settings.ringBufferSize, emaAlpha: settings.emaAlpha, neutralBoost: settings.neutralBoost, framesForAverage: settings.framesForAverage, show3DModel: $0) }
                    ))
                    Text("Display a 3D ellipsoid that rotates with face orientation")
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

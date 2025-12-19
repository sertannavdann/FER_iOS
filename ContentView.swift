//
//  ContentView.swift
//  FacialExpressionDetection_iOS
//
//  Created by Sertan Avdan on 2025-12-08.
//

import SwiftUI
import AVFoundation
import Vision
import CoreML
import Combine
#if os(iOS)
import UIKit
import ARKit
#endif

// MARK: - Main Content View
struct ContentView: View {
    @AppStorage("inferenceSettings") private var settingsData: Data = InferenceSettings().rawValue
    private var settings: InferenceSettings {
        get { InferenceSettings(rawValue: settingsData) ?? InferenceSettings() }
        set { settingsData = newValue.rawValue }
    }

    @StateObject private var lifecycle = AppLifecycleManager.shared
    @StateObject private var coordinator = PipelineCoordinator()
    @StateObject private var predictor = FERPredictor(settings: InferenceSettings())
    @State private var showSettings = false
    @State private var predictions: [FacePrediction] = []
    @State private var lastUIUpdate: Date = .distantPast
    private let uiUpdateInterval: TimeInterval = 1.0 / 15.0  // 15fps
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Camera preview
                if let session = coordinator.arSession {
                    ARSessionView(session: session, graphManager: coordinator.arGraphManager)
                        .ignoresSafeArea()
                } else {
                    CameraPreviewView(previewLayer: coordinator.previewLayer)
                        .ignoresSafeArea()
                }

                // Spatial Face Widget (replaces old overlays)
                SpatialFaceWidget(
                    prediction: predictions.first,
                    history: predictor.probabilityHistory,
                    geometry: geometry,
                    settings: settings,
                    isARGraphActive: coordinator.isBackCamera && settings.enableARGraph && coordinator.isARKitActive
                )
                
                // UI Overlay
                VStack {
                    // Top bar with emotion display
                    HStack {
                        if let first = predictions.first {
                            HStack(spacing: 12) {
                                Text(first.dominantEmoji)
                                    .font(.system(size: 50))
                                
                                VStack(alignment: .leading) {
                                    Text(first.dominantEmotion.capitalized)
                                        .font(.title2)
                                        .fontWeight(.bold)
                                    
                                    Text(String(format: "%.1f%% confidence", first.confidence * 100))
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .padding()
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                        } else {
                            HStack {
                                Image(systemName: "face.dashed")
                                    .font(.title)
                                Text("No face detected")
                                    .font(.headline)
                            }
                            .padding()
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                        }
                        
                        // Pipeline info badge
                        if coordinator.isBackCamera {
                            Text(coordinator.pipelineInfo)
                                .font(.caption2)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(coordinator.isARKitActive ? Color.green.opacity(0.8) : Color.orange.opacity(0.8), in: Capsule())
                                .foregroundColor(.white)
                        }
                        
                        Spacer()
                        
                        HStack(spacing: 12) {
                            Button {
                                coordinator.switchCamera()
                            } label: {
                                Image(systemName: "camera.rotate")
                                    .font(.title2)
                                    .padding()
                                    .background(.ultraThinMaterial, in: Circle())
                            }

                            Button {
                                showSettings = true
                            } label: {
                                Image(systemName: "slider.horizontal.3")
                                    .font(.title2)
                                    .padding()
                                    .background(.ultraThinMaterial, in: Circle())
                            }
                        }
                    }
                    .padding()
                    
                    Spacer()
                }
            }
        }
        .onChange(of: settingsData) { _, _ in
            predictor.update(settings: settings)
        }
        .onChange(of: showSettings) { _, isOpen in
            // Sync settings sheet state to lifecycle manager
            lifecycle.isSettingsOpen = isOpen
        }
        .onDisappear {
            coordinator.stop()
        }
        .sheet(isPresented: $showSettings) {
            SettingsSheet(settings: Binding(
                get: { settings },
                set: { settingsData = $0.rawValue }
            ))
        }
        .onAppear {
            // Register components with lifecycle manager
            lifecycle.register(coordinator: coordinator)
            lifecycle.register(predictor: predictor)

            predictor.update(settings: settings)
            coordinator.onFrameCapture = { pixelBuffer, faces, orientation in
                predictor.predict(pixelBuffer: pixelBuffer, faces: faces, orientation: orientation)
            }
            coordinator.start()
        }
        .onReceive(predictor.$faceOutputs.receive(on: DispatchQueue.main)) { outputs in
            let now = Date()
            guard now.timeIntervalSince(lastUIUpdate) >= uiUpdateInterval else { return }
            lastUIUpdate = now
            predictions = outputs
        }
        .onReceive(predictor.$probabilityHistory.receive(on: DispatchQueue.main)) { history in
            // Update AR graph surface using latest history and face transform
            guard coordinator.isBackCamera,
                  settings.enableARGraph,
                  let manager = coordinator.arGraphManager else { return }

            let faceTransform = coordinator.faces.first(where: { $0.transform != nil })?.transform
            manager.update(history: history, faceTransform: faceTransform, settings: settings)
        }
        .onReceive(coordinator.$faces.receive(on: DispatchQueue.main)) { faces in
            // Keep AR surface aligned even when history hasn't changed
            guard coordinator.isBackCamera,
                  settings.enableARGraph,
                  let manager = coordinator.arGraphManager else { return }

            let faceTransform = faces.first(where: { $0.transform != nil })?.transform
            manager.update(history: predictor.probabilityHistory, faceTransform: faceTransform, settings: settings)
        }
    }
}

#Preview {
    ContentView()
}


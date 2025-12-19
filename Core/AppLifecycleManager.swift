import Foundation
import SwiftUI
import Combine
import UIKit

/// Global state manager for app lifecycle coordination
/// Manages pause/resume of camera pipelines and predictions based on app state
@MainActor
class AppLifecycleManager: ObservableObject {

    // MARK: - Singleton

    static let shared = AppLifecycleManager()

    // MARK: - Published State

    /// Whether the app is currently active in the foreground
    @Published var isActive: Bool = true

    /// Whether the settings sheet is currently presented
    @Published var isSettingsOpen: Bool = false

    /// Whether all pipelines should be paused (settings open or app backgrounded)
    @Published var shouldPausePipelines: Bool = false

    // MARK: - Private Properties

    private var cancellables = Set<AnyCancellable>()

    // References to components that need pausing (weak to avoid retain cycles)
    private weak var coordinator: PipelineCoordinatorProtocol?
    private weak var predictor: FERPredictorProtocol?

    // MARK: - Initialization

    private init() {
        setupNotifications()
        setupStateObservers()
        Log.info("[AppLifecycle] Manager initialized")
    }

    // MARK: - Setup

    private func setupNotifications() {
        // Monitor app lifecycle notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appWillResignActive),
            name: UIApplication.willResignActiveNotification,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appWillEnterForeground),
            name: UIApplication.willEnterForegroundNotification,
            object: nil
        )
    }

    private func setupStateObservers() {
        // Combine isActive and isSettingsOpen to determine if pipelines should pause
        Publishers.CombineLatest($isActive, $isSettingsOpen)
            .map { isActive, isSettingsOpen in
                // Pause if app is inactive OR settings are open
                !isActive || isSettingsOpen
            }
            .removeDuplicates()
            .sink { [weak self] shouldPause in
                Task { @MainActor in
                    self?.shouldPausePipelines = shouldPause
                    if shouldPause {
                        self?.pauseAll()
                    } else {
                        self?.resumeAll()
                    }
                }
            }
            .store(in: &cancellables)
    }

    // MARK: - Registration

    /// Register the pipeline coordinator for lifecycle management
    func register(coordinator: PipelineCoordinatorProtocol) {
        self.coordinator = coordinator
        Log.debug("[AppLifecycle] Coordinator registered")
    }

    /// Register the FER predictor for lifecycle management
    func register(predictor: FERPredictorProtocol) {
        self.predictor = predictor
        Log.debug("[AppLifecycle] Predictor registered")
    }

    // MARK: - Public Control

    /// Manually pause all pipelines and predictions
    func pauseAll() {
        Log.info("[AppLifecycle] Pausing all pipelines and predictions")
        coordinator?.pause()
        predictor?.pause()
    }

    /// Manually resume all pipelines and predictions
    func resumeAll() {
        // Only resume if we're not supposed to be paused
        guard !shouldPausePipelines else {
            Log.debug("[AppLifecycle] Skipping resume - should remain paused")
            return
        }

        Log.info("[AppLifecycle] Resuming all pipelines and predictions")
        coordinator?.resume()
        predictor?.resume()
    }

    // MARK: - Notification Handlers

    @objc private func appWillResignActive(_ notification: Notification) {
        Task { @MainActor in
            isActive = false
            Log.info("[AppLifecycle] App will resign active")
        }
    }

    @objc private func appDidBecomeActive(_ notification: Notification) {
        Task { @MainActor in
            isActive = true
            Log.info("[AppLifecycle] App did become active")
        }
    }

    @objc private func appDidEnterBackground(_ notification: Notification) {
        Task { @MainActor in
            isActive = false
            Log.info("[AppLifecycle] App entered background")
        }
    }

    @objc private func appWillEnterForeground(_ notification: Notification) {
        Task { @MainActor in
            Log.info("[AppLifecycle] App will enter foreground")
            // isActive will be set in didBecomeActive
        }
    }

    // MARK: - Cleanup

    deinit {
        NotificationCenter.default.removeObserver(self)
        cancellables.removeAll()
    }
}

// MARK: - Protocols

/// Protocol for components that can be paused/resumed
protocol PipelineCoordinatorProtocol: AnyObject {
    func pause()
    func resume()
}

protocol FERPredictorProtocol: AnyObject {
    func pause()
    func resume()
}

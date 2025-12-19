import Foundation
import Metal
import SwiftUI
import CoreImage
import UIKit

/// Converts SwiftUI ProbabilityTimelineGraph to Metal texture for 3D rendering
@MainActor
class GraphTextureGenerator {

    // MARK: - Properties

    private let device: MTLDevice
    private let textureCache: CVMetalTextureCache
    private let textureSize: CGSize
    private let targetFPS: Double = 15.0  // Match UI update rate

    private var lastRenderTime: Date = .distantPast
    private var cachedTexture: MTLTexture?
    private var renderer: ImageRenderer<ProbabilityTimelineGraph>?
    private var reusableContext: CGContext?
    private var reusableTexture: MTLTexture?

    // MARK: - Initialization

    init?(size: CGSize = CGSize(width: 512, height: 512)) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Log.error("[GraphTexture] Failed to create Metal device")
            return nil
        }

        var textureCache: CVMetalTextureCache?
        let result = CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        guard result == kCVReturnSuccess, let cache = textureCache else {
            Log.error("[GraphTexture] Failed to create texture cache")
            return nil
        }

        self.device = device
        self.textureCache = cache
        self.textureSize = size

        Log.info("[GraphTexture] Initialized with size: \(size)")
    }

    // MARK: - Texture Generation

    /// Render probability history to Metal texture
    /// - Parameter history: Probability history array [[Float]]
    /// - Returns: Metal texture or nil if rendering fails
    func render(history: [[Float]]) -> MTLTexture? {
        // Throttle to target FPS
        let now = Date()
        let minInterval = 1.0 / targetFPS
        guard now.timeIntervalSince(lastRenderTime) >= minInterval else {
            // Return cached texture if within throttle interval
            return cachedTexture
        }
        lastRenderTime = now

        // Create or update renderer
        let graph = ProbabilityTimelineGraph(history: history)
        if renderer == nil {
            renderer = ImageRenderer(content: graph)
            renderer?.proposedSize = ProposedViewSize(textureSize)
        } else {
            renderer?.content = graph
            renderer?.proposedSize = ProposedViewSize(textureSize)
        }

        guard let renderer = renderer else {
            Log.error("[GraphTexture] Renderer not available")
            return cachedTexture
        }

        // Render to UIImage
        guard let uiImage = renderer.uiImage else {
            Log.error("[GraphTexture] Failed to render to UIImage")
            return cachedTexture
        }

        // Convert UIImage to Metal texture
        guard let texture = createMetalTexture(from: uiImage) else {
            Log.error("[GraphTexture] Failed to create Metal texture")
            return cachedTexture
        }

        // Cache for reuse
        cachedTexture = texture
        return texture
    }

    // MARK: - Image to Texture Conversion

    private func createMetalTexture(from image: UIImage) -> MTLTexture? {
        // Convert UIImage to CGImage
        guard let cgImage = image.cgImage else {
            return nil
        }

        let width = cgImage.width
        let height = cgImage.height

        guard let texture = makeReusableTexture(width: width, height: height) else {
            Log.error("[GraphTexture] Failed to obtain reusable texture")
            return nil
        }

        guard let context = makeReusableContext(width: width, height: height) else {
            return nil
        }

        // Draw image into bitmap context
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Copy bitmap data to texture
        guard let data = context.data else {
            return nil
        }

        let bytesPerRow = context.bytesPerRow
        let region = MTLRegionMake2D(0, 0, width, height)

        texture.replace(
            region: region,
            mipmapLevel: 0,
            withBytes: data,
            bytesPerRow: bytesPerRow
        )

        return texture
    }

    private func makeReusableContext(width: Int, height: Int) -> CGContext? {
        // Reuse bitmap context to avoid frequent allocations
        if let context = reusableContext,
           context.width == width,
           context.height == height {
            return context
        }

        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )

        reusableContext = context
        return context
    }

    private func makeReusableTexture(width: Int, height: Int) -> MTLTexture? {
        if let texture = reusableTexture,
           texture.width == width,
           texture.height == height {
            return texture
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .renderTarget]
        descriptor.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            return nil
        }
        reusableTexture = texture
        return texture
    }

    // MARK: - Cleanup

    /// Clear cached texture to free memory
    func clearCache() {
        cachedTexture = nil
        renderer = nil
        Log.debug("[GraphTexture] Cache cleared")
    }
}

// MARK: - ImageRenderer Extension

extension ImageRenderer {
    /// Render to UIImage (convenience property)
    @MainActor
    var uiImage: UIImage? {
        return self.uiImage(scale: 1.0)
    }

    /// Render to UIImage with specified scale
    @MainActor
    func uiImage(scale: CGFloat) -> UIImage? {
        let targetSize = CGSize(
            width: self.proposedSize.width ?? 0,
            height: self.proposedSize.height ?? 0
        )

        return UIGraphicsImageRenderer(size: targetSize).image { context in
            self.render { _, renderer in
                renderer(context.cgContext)
            }
        }
    }
}

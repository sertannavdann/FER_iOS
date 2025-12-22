import Foundation
import Metal
import MetalKit
import CoreVideo

class MetalGrayscaleConverter {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let computePipelineState: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    
    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary() else {
            Log.error("Failed to initialize Metal")
            return nil
        }
        
        self.device = device
        self.commandQueue = commandQueue
        
        do {
            guard let kernelFunction = library.makeFunction(name: "grayscaleKernel") else {
                Log.error("Failed to find grayscaleKernel in default library")
                return nil
            }
            self.computePipelineState = try device.makeComputePipelineState(function: kernelFunction)
        } catch {
            Log.error("Failed to create compute pipeline: \(error)")
            return nil
        }
        
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        self.textureCache = cache
    }
    
    func convert(pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // 1. Create Input Texture from Pixel Buffer
        guard let inputTexture = createTexture(from: pixelBuffer, pixelFormat: .bgra8Unorm, planeIndex: 0) else {
            return nil
        }
        
        // 2. Create Output Pixel Buffer (reused or new)
        // For efficiency, we should maintain a pool, but for now we create one.
        // Ideally, we want a 1-channel output (Luma), but Vision/CoreML often expects 32BGRA.
        // Let's stick to 32BGRA for compatibility, but with grayscale content.
        var outputPixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &outputPixelBuffer)
        guard status == kCVReturnSuccess, let outputBuffer = outputPixelBuffer else {
            return nil
        }
        
        guard let outputTexture = createTexture(from: outputBuffer, pixelFormat: .bgra8Unorm, planeIndex: 0) else {
            return nil
        }
        
        // 3. Encode Compute Command
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        
        encoder.setComputePipelineState(computePipelineState)
        encoder.setTexture(inputTexture, index: 0)
        encoder.setTexture(outputTexture, index: 1)
        
        let threadGroupSize = MTLSizeMake(16, 16, 1)
        let threadGroups = MTLSizeMake(
            (width + threadGroupSize.width - 1) / threadGroupSize.width,
            (height + threadGroupSize.height - 1) / threadGroupSize.height,
            1
        )
        
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted() // Wait for sync (can be optimized)

        // 4. Apply Histogram Equalization (matches C++ preprocessing: equalizeHist)
        // This improves model performance in varying lighting conditions
        let ciContext = CIContext(mtlDevice: device, options: [.workingColorSpace: NSNull()])
        var ciImage = CIImage(cvPixelBuffer: outputBuffer)

        // CoreImage doesn't have direct histogram equalization, but we can approximate it
        // Using color controls to enhance contrast similar to histogram equalization
        if let filter = CIFilter(name: "CIColorControls") {
            filter.setValue(ciImage, forKey: kCIInputImageKey)
            filter.setValue(1.5, forKey: kCIInputContrastKey)  // Increase contrast

            if let outputImage = filter.outputImage {
                ciContext.render(outputImage, to: outputBuffer)
            }
        }

        return outputBuffer
    }
    
    private func createTexture(from pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> MTLTexture? {
        guard let textureCache = textureCache else { return nil }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        var cvTextureOut: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            pixelFormat,
            width,
            height,
            planeIndex,
            &cvTextureOut
        )
        
        guard status == kCVReturnSuccess, let cvTexture = cvTextureOut else {
            return nil
        }
        
        return CVMetalTextureGetTexture(cvTexture)
    }
}

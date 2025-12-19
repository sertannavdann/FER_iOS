#include <metal_stdlib>
using namespace metal;

// Standard Rec. 601 luma coefficients
constant half3 kRec601Luma = half3(0.299, 0.587, 0.114);

kernel void grayscaleKernel(texture2d<half, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }
    
    half4 color = inTexture.read(gid);
    half gray = dot(color.rgb, kRec601Luma);
    
    // Write grayscale value to all channels (or just R if using R8Unorm)
    // Here we assume BGRA/RGBA output for compatibility, so we replicate the gray value
    outTexture.write(half4(gray, gray, gray, 1.0), gid);
}

import SwiftUI
import SceneKit

/// 3D face model view using SceneKit - displays an elongated ellipsoid that rotates with face orientation
struct FaceModelView: UIViewRepresentable {
    let yaw: Float?
    let pitch: Float?
    let roll: Float?

    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.backgroundColor = .clear
        scnView.autoenablesDefaultLighting = true
        scnView.allowsCameraControl = false
        scnView.antialiasingMode = .multisampling2X  // Balance quality and performance
        scnView.preferredFramesPerSecond = 15  // Limit to 15fps

        // Create scene
        let scene = SCNScene()
        scnView.scene = scene

        // Create elongated ellipsoid (sphere scaled 1:1.5:1)
        let sphere = SCNSphere(radius: 0.5)
        let faceNode = SCNNode(geometry: sphere)
        faceNode.scale = SCNVector3(1.0, 1.5, 1.0)  // Elongate vertically
        faceNode.name = "faceModel"

        // Neutral skin-tone material
        let material = SCNMaterial()
        material.diffuse.contents = UIColor(red: 0.95, green: 0.85, blue: 0.75, alpha: 1.0)
        material.lightingModel = .lambert
        material.isDoubleSided = false
        sphere.materials = [material]

        scene.rootNode.addChildNode(faceNode)

        // Camera setup (orthographic for consistent perspective)
        let camera = SCNCamera()
        camera.usesOrthographicProjection = true
        camera.orthographicScale = 1.2
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0, 3)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)

        // Store context for updates
        context.coordinator.sceneView = scnView

        return scnView
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        guard let faceNode = uiView.scene?.rootNode.childNode(withName: "faceModel", recursively: false) else {
            return
        }

        // Apply rotation from detected face orientation
        // SceneKit uses radians, our values are in degrees
        let yawRad = (yaw ?? 0) * .pi / 180.0
        let pitchRad = (pitch ?? 0) * .pi / 180.0
        let rollRad = (roll ?? 0) * .pi / 180.0

        // Create rotation: yaw (Y), pitch (X), roll (Z)
        // Order matters: apply in Y-X-Z sequence to match Vision's coordinate system
        let yawRotation = SCNMatrix4MakeRotation(yawRad, 0, 1, 0)      // Y-axis (left/right)
        let pitchRotation = SCNMatrix4MakeRotation(-pitchRad, 1, 0, 0) // X-axis (up/down, inverted)
        let rollRotation = SCNMatrix4MakeRotation(rollRad, 0, 0, 1)    // Z-axis (tilt)

        // Combine rotations
        var transform = SCNMatrix4Identity
        transform = SCNMatrix4Mult(transform, yawRotation)
        transform = SCNMatrix4Mult(transform, pitchRotation)
        transform = SCNMatrix4Mult(transform, rollRotation)

        // Apply scale (ellipsoid elongation)
        let scale = SCNMatrix4MakeScale(1.0, 1.5, 1.0)
        transform = SCNMatrix4Mult(scale, transform)

        // Animate rotation smoothly
        SCNTransaction.begin()
        SCNTransaction.animationDuration = 0.1  // Smooth 100ms transition
        faceNode.transform = transform
        SCNTransaction.commit()
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator {
        var sceneView: SCNView?
    }
}

#Preview {
    FaceModelView(yaw: 15, pitch: -10, roll: 5)
        .frame(width: 200, height: 200)
        .background(Color.gray.opacity(0.2))
}

import Foundation

/// Log severity levels
enum LogLevel: String {
    case debug = "DEBUG"
    case info = "INFO"
    case warn = "WARN"
    case error = "ERROR"
}

/// Core logging engine with file/function/line tracking and runtime control
struct LogEngine {
    /// Default enabled in DEBUG, off in Release; can be toggled at runtime.
    static var isEnabled: Bool = {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }()
    
    /// Enable or disable console logging at runtime
    static func enable(_ flag: Bool) {
        isEnabled = flag
    }
    
    /// Log a message with level, file, function, and line information
    static func log(level: LogLevel,
                    _ message: @autoclosure () -> String,
                    file: String,
                    function: String,
                    line: Int) {
        guard isEnabled else { return }
        let fileName = (file as NSString).lastPathComponent
        print("[\(level.rawValue)] \(fileName):\(line) \(function) - \(message())")
    }
}

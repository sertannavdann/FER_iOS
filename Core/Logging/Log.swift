import Foundation
import os.log

/// Unified logging interface
/// - Uses Apple's `os.Logger` for system-level logging (visible in Console.app)
/// - Uses `LogEngine` for console output with file/function/line info
struct Log {
    private static let subsystem = Bundle.main.bundleIdentifier ?? "com.app.fer"
    private static let logger = Logger(subsystem: subsystem, category: "FER")
    
    /// Enable or disable console logging at runtime
    static func enable(_ flag: Bool) {
        LogEngine.enable(flag)
    }
    
    /// Whether console logging is currently enabled
    static var isEnabled: Bool {
        get { LogEngine.isEnabled }
        set { LogEngine.isEnabled = newValue }
    }
    
    // MARK: - Debug
    
    static func debug(_ message: @autoclosure () -> String,
                      file: String = #file,
                      function: String = #function,
                      line: Int = #line) {
        #if DEBUG
        let msg = message()
        logger.debug("\(msg)")
        LogEngine.log(level: .debug, msg, file: file, function: function, line: line)
        #endif
    }
    
    // MARK: - Info
    
    static func info(_ message: @autoclosure () -> String,
                     file: String = #file,
                     function: String = #function,
                     line: Int = #line) {
        let msg = message()
        logger.info("\(msg)")
        LogEngine.log(level: .info, msg, file: file, function: function, line: line)
    }
    
    // MARK: - Warning
    
    static func warn(_ message: @autoclosure () -> String,
                     file: String = #file,
                     function: String = #function,
                     line: Int = #line) {
        let msg = message()
        logger.warning("\(msg)")
        LogEngine.log(level: .warn, msg, file: file, function: function, line: line)
    }
    
    // MARK: - Error
    
    static func error(_ message: @autoclosure () -> String,
                      file: String = #file,
                      function: String = #function,
                      line: Int = #line) {
        let msg = message()
        logger.error("\(msg)")
        LogEngine.log(level: .error, msg, file: file, function: function, line: line)
    }
}

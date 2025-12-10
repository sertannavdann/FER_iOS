import Foundation

enum LogLevel: String {
    case debug = "DEBUG"
    case info = "INFO"
    case warn = "WARN"
    case error = "ERROR"
}

struct Log {
    // Default enabled in DEBUG, off in Release; can be toggled at runtime.
    static var isEnabled: Bool = {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }()
    
    static func enable(_ flag: Bool) {
        isEnabled = flag
    }
    
    static func debug(_ message: @autoclosure () -> String,
                      file: String = #file,
                      function: String = #function,
                      line: Int = #line) {
        log(level: .debug, message(), file: file, function: function, line: line)
    }
    
    static func info(_ message: @autoclosure () -> String,
                     file: String = #file,
                     function: String = #function,
                     line: Int = #line) {
        log(level: .info, message(), file: file, function: function, line: line)
    }
    
    static func warn(_ message: @autoclosure () -> String,
                     file: String = #file,
                     function: String = #function,
                     line: Int = #line) {
        log(level: .warn, message(), file: file, function: function, line: line)
    }
    
    static func error(_ message: @autoclosure () -> String,
                      file: String = #file,
                      function: String = #function,
                      line: Int = #line) {
        log(level: .error, message(), file: file, function: function, line: line)
    }
    
    private static func log(level: LogLevel,
                            _ message: @autoclosure () -> String,
                            file: String,
                            function: String,
                            line: Int) {
        guard isEnabled else { return }
        let fileName = (file as NSString).lastPathComponent
        print("[\(level.rawValue)] \(fileName):\(line) \(function) - \(message())")
    }
}

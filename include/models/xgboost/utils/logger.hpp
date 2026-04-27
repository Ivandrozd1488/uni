#pragma once

/**
 * @file logger.hpp
 * @brief Application-controlled logging hooks for XGBoost-style components.
 */

#include <chrono>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

namespace xgb {

/** @brief Callback target used to consume formatted text log records. */
using LogSink = std::function<void(const std::string&)>;

/** @brief Severity threshold for SDK log records. */
enum class LogLevel { DEBUG = 0, INFO, WARNING, ERROR };

/** @brief One structured key/value field attached to a log record. */
struct LogField {
  std::string key;
  std::string value;
};

/** @brief Structured record emitted by the logging system. */
struct LogRecord {
  LogLevel level{LogLevel::INFO};
  std::string message;
  std::vector<LogField> fields;
  const char* file{nullptr};
  int line{0};
  std::string timestamp;
};

/** @brief Callback target used to consume structured log records. */
using StructuredLogSink = std::function<void(const LogRecord&)>;

/**
 * @brief Central logger for XGBoost-style components.
 * @note Defaults to silent/no-op in library mode. Applications can attach
 * a structured sink or opt into a human-readable console formatter.
 */
class Logger {
public:
  /** @brief Access the process-local logger instance. */
  static Logger& instance();

  /** @brief Set the minimum emitted severity level. */
  void set_level(LogLevel level) noexcept { min_level_ = level; }
  /** @brief Silence all log output without changing configured sinks. */
  void set_silent(bool silent) noexcept { silent_ = silent; }
  /** @brief Install an application-owned sink for formatted text messages. */
  void set_sink(LogSink sink) { sink_ = std::move(sink); }
  /** @brief Install an application-owned structured sink. */
  void set_structured_sink(StructuredLogSink sink) { structured_sink_ = std::move(sink); }
  /** @brief Enable/disable built-in human-readable console formatting. */
  void enable_console_formatter(bool enabled = true) noexcept { console_formatter_enabled_ = enabled; }

  /**
   * @brief Emit a log record.
   * @param level Record severity.
   * @param msg Message text.
   * @param fields Optional structured key/value fields.
   * @param file Optional source file metadata.
   * @param line Optional source line metadata.
   */
  void log(LogLevel level, const std::string& msg,
     std::initializer_list<LogField> fields = {},
     const char* file = nullptr, int line = 0);

  /** @brief Emit a debug-level record. */
  void debug(const std::string& msg) { log(LogLevel::DEBUG, msg, {}); }
  /** @brief Emit an info-level record. */
  void info(const std::string& msg) { log(LogLevel::INFO,  msg, {}); }
  /** @brief Emit a warning-level record. */
  void warning(const std::string& msg) { log(LogLevel::WARNING, msg, {}); }
  /** @brief Emit an error-level record. */
  void error(const std::string& msg) { log(LogLevel::ERROR, msg, {}); }

private:
  Logger() = default;
  LogLevel min_level_{LogLevel::WARNING};
  bool silent_{true};
  bool console_formatter_enabled_{false};
  LogSink sink_{};
  StructuredLogSink structured_sink_{};

  static const char* level_str(LogLevel l);
  static std::string timestamp();
  static std::string format_record(const LogRecord& rec);
};

//        
//  Convenience macros
//        
#define XGB_LOG_DEBUG(msg)             \
  ::xgb::Logger::instance().log(         \
    ::xgb::LogLevel::DEBUG, (msg), {}, __FILE__, __LINE__)

#define XGB_LOG_INFO(msg)            \
  ::xgb::Logger::instance().log(         \
    ::xgb::LogLevel::INFO, (msg), {}, __FILE__, __LINE__)

#define XGB_LOG_WARN(msg)            \
  ::xgb::Logger::instance().log(         \
    ::xgb::LogLevel::WARNING, (msg), {}, __FILE__, __LINE__)

#define XGB_LOG_ERROR(msg)             \
  ::xgb::Logger::instance().log(         \
    ::xgb::LogLevel::ERROR, (msg), {}, __FILE__, __LINE__)

// Stream-style helper      
struct LogStream {
  std::ostringstream oss;
  LogLevel level;
  explicit LogStream(LogLevel l) : level(l) {}
  ~LogStream() { Logger::instance().log(level, oss.str(), {}); }
  template<typename T>
  LogStream& operator<<(const T& v) { oss << v; return *this; }
};

#define XGB_INFO  ::xgb::LogStream(::xgb::LogLevel::INFO)
#define XGB_DEBUG ::xgb::LogStream(::xgb::LogLevel::DEBUG)

} // namespace xgb

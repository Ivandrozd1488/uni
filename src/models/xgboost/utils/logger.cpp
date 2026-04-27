#include "models/xgboost/utils/logger.hpp"
#include <ctime>
#include <iostream>
#include <mutex>

namespace xgb {

static std::mutex g_log_mutex;

Logger& Logger::instance() {
  static Logger inst;
  return inst;
}

void Logger::log(LogLevel level, const std::string& msg,
       std::initializer_list<LogField> fields,
       const char* file, int line)
{
  if (silent_ || level < min_level_) return;
  std::lock_guard<std::mutex> lock(g_log_mutex);
  LogRecord rec;
  rec.level = level;
  rec.message = msg;
  rec.fields.assign(fields.begin(), fields.end());
  rec.file = file;
  rec.line = line;
  rec.timestamp = timestamp();

  if (structured_sink_) {
    structured_sink_(rec);
    return;
  }
  if (sink_) {
    sink_(format_record(rec));
    return;
  }
  if (console_formatter_enabled_) {
    std::clog << format_record(rec) << '\n';
  }
}

const char* Logger::level_str(LogLevel l) {
  switch (l) {
    case LogLevel::DEBUG: return "DEBUG  ";
    case LogLevel::INFO:  return "INFO ";
    case LogLevel::WARNING: return "WARNING";
    case LogLevel::ERROR: return "ERROR  ";
  }
  return "???";
}

std::string Logger::timestamp() {
  auto now = std::chrono::system_clock::now();
  auto t   = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
#ifdef _WIN32
  localtime_s(&tm_buf, &t);
#else
  localtime_r(&t, &tm_buf);
#endif
  char buf[20];
  std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm_buf);
  return buf;
}

std::string Logger::format_record(const LogRecord& rec) {
  std::ostringstream stream;
  stream << "[" << rec.timestamp << "] " << level_str(rec.level) << " " << rec.message;
  for (const auto& f : rec.fields) {
    stream << " " << f.key << "=" << f.value;
  }
  return stream.str();
}

} // namespace xgb

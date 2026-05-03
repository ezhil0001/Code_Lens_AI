/**
 * Development Environment Configuration
 * 
 * Used during development with ng serve
 */

export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000',
  wsUrl: 'ws://localhost:8000',
  
  // Feature flags
  cacheEnabled: true,
  streamingEnabled: true,
  healthCheckInterval: 30000, // 30 seconds
  
  // API endpoints
  endpoints: {
    chat: '/api/v1/chat',
    chatStream: '/api/v1/chat/stream',
    health: '/api/v1/health',
    healthDetailed: '/api/v1/health/detailed',
    cacheStatus: '/api/v1/chat/cache/status',
    cacheClear: '/api/v1/chat/cache/clear',
  },
  
  // Timeouts
  timeouts: {
    streaming: 60000,    // 60 seconds for streaming
    nonStreaming: 30000, // 30 seconds for regular requests
    health: 5000,        // 5 seconds for health checks
  },
  
  // Cache settings
  cache: {
    localStoragePrefix: 'chat_',
    ttl: 86400000, // 24 hours in ms
  },
};

/**
 * Production Environment Configuration
 * 
 * Used when deployed to production
 */

export const environment = {
  production: true,
  apiUrl: 'https://api.yourdomain.com',
  wsUrl: 'wss://api.yourdomain.com',
  
  // Feature flags
  cacheEnabled: true,
  streamingEnabled: true,
  healthCheckInterval: 60000, // 60 seconds (less frequent in prod)
  
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

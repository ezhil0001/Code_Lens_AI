/**
 * Development Environment Configuration
 * 
 * Used during development with ng serve
 */

export const environment = {
  production: false,
  apiUrl: 'http://localhost:8001',
  wsUrl: 'ws://localhost:8001',
  
  // Feature flags
  cacheEnabled: true,
  streamingEnabled: true,
  healthCheckInterval: 30000, // 30 seconds
  
  // API endpoints
  endpoints: {
    health: '/api/v1/health',
    healthDetailed: '/api/v1/health/detailed',

    // v2 — LangGraph multi-agent streaming + chat utilities
    chatStreamV2: '/api/v2/chat/stream',
    cacheStatus: '/api/v2/chat/cache/status',
    cacheClear: '/api/v2/chat/cache/clear',
    sessions: '/api/v2/sessions',   // base for /{id}/checkpoints, /resume, /branch, /replay
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


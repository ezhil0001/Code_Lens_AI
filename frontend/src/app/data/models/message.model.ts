/**
 * Message Model
 * Represents a single message in the chat
 */
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  
  // RAG-specific metadata
  citations?: Citation[];
  model?: string;
  tokenCount?: number;
  processingTimeMs?: number;
  
  // UI state
  isStreaming?: boolean;
  error?: string | null;
}

/**
 * Citation Model
 * Reference to source code used in RAG response
 */
export interface Citation {
  sourceFile: string;
  repository: string;
  lineStart: number;
  lineEnd: number;
  codeSnippet: string;
  relevanceScore?: number;
  language?: string;
}

/**
 * Chat Session Model
 * Represents a conversation thread
 */
export interface ChatSession {
  id: string;
  title: string;
  repository?: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  settings?: ChatSettings;
}

/**
 * Chat Settings Model
 * User preferences for chat behavior
 */
export interface ChatSettings {
  model: 'gpt-4' | 'gpt-3.5-turbo' | 'groq' | 'local';
  temperature: number;
  maxTokens: number;
  useHybridSearch: boolean;
  citations: boolean;
}

/**
 * Search Result Model
 * Result from hybrid search
 */
export interface SearchResult {
  id: string;
  score: number;
  content: string;
  metadata: {
    file: string;
    language: string;
    startLine: number;
    endLine: number;
  };
  source: 'vector' | 'bm25' | 'hybrid';
}

/**
 * Stream Token Model
 * Individual token from streaming response
 */
export interface StreamToken {
  token: string;
  type: 'content' | 'metadata' | 'citations' | 'done';
  metadata?: any;
}

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

/**
 * APIService
 * Handles all HTTP requests to FastAPI backend
 * Includes authentication, error handling, request/response interceptors
 */
@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'http://localhost:8001/api/v1';

  constructor(private http: HttpClient) { }

  /**
   * Get health status of backend
   */
  getHealth(): Observable<any> {
    return this.http.get(`${this.apiUrl}/health`);
  }

  /**
   * Get list of available repositories
   */
  getRepositories(): Observable<any> {
    return this.http.get(`${this.apiUrl}/repositories`);
  }

  /**
   * Get repository details and file tree
   */
  getRepository(repoId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/repositories/${repoId}`);
  }

  /**
   * Search within repository (hybrid: BM25 + vector)
   */
  searchRepository(query: string, repoId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/search`, {
      query,
      repo_id: repoId,
      use_hybrid_search: true
    });
  }

  /**
   * Get code file contents
   */
  getCodeFile(repoId: string, filePath: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/repositories/${repoId}/files`, {
      params: { path: filePath }
    });
  }

  /**
   * List files in directory
   */
  listDirectory(repoId: string, dirPath: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/repositories/${repoId}/directory`, {
      params: { path: dirPath }
    });
  }

  /**
   * Save conversation history
   */
  saveConversation(data: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/conversations`, data);
  }

  /**
   * Get conversation history
   */
  getConversationHistory(conversationId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/conversations/${conversationId}`);
  }
}

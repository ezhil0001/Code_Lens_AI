import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

/**
 * IngestService
 * Handles document upload and ingestion into ChromaDB
 */
@Injectable({
  providedIn: 'root'
})
export class IngestService {
  private apiUrl = `${environment.apiUrl}/api/v1`;

  constructor(private http: HttpClient) { }

  /** Ingest endpoints are authenticated; attach the stored bearer token. */
  private authHeaders(): Record<string, string> {
    const token = localStorage.getItem('auth_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  /**
   * Upload and ingest document files
   * @param files - Files to upload
   * @returns Observable with ingestion status
   */
  uploadDocuments(files: File[]): Observable<any> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    return this.http.post(`${this.apiUrl}/ingest/documents`, formData, {
      headers: this.authHeaders(),
      reportProgress: true,
      responseType: 'json'
    });
  }

  /**
   * Ingest content from URL
   * @param url - URL to ingest
   * @returns Observable with ingestion status
   */
  ingestFromUrl(url: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/ingest/url`, { url }, {
      headers: this.authHeaders(),
    });
  }

  /**
   * Get ingestion status
   * @returns Observable with status information
   */
  getIngestionStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/ingest/status`, {
      headers: this.authHeaders(),
    });
  }

  /**
   * Clear all ingested documents
   * @returns Observable with clear status
   */
  clearDocuments(): Observable<any> {
    return this.http.delete(`${this.apiUrl}/ingest/clear`, {
      headers: this.authHeaders(),
    });
  }
}

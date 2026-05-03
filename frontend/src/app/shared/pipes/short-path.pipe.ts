import { Pipe, PipeTransform } from '@angular/core';

/**
 * ShortPathPipe
 * Shortens file paths for display
 * Example: /long/path/to/file.ts -> file.ts
 */
@Pipe({
  name: 'shortPath',
  standalone: true
})
export class ShortPathPipe implements PipeTransform {
  transform(value: string): string {
    if (!value) return '';
    
    // Get just the filename
    const parts = value.split('/');
    return parts[parts.length - 1] || value;
  }
}

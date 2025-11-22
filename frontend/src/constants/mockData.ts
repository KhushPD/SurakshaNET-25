/**
 * Mock Data & Constants
 * ======================
 * Placeholder data for development.
 * Replace with real API calls in production.
 */

import type { RadarDataPoint, AlertLog } from '../types';

// Mock radar chart data
// TODO: Replace with API call to GET /api/dashboard/radar-stats
export const MOCK_RADAR_DATA: RadarDataPoint[] = [
    { subject: 'DDoS', A: 120, fullMark: 150 },
    { subject: 'Phishing', A: 98, fullMark: 150 },
    { subject: 'Malware', A: 86, fullMark: 150 },
    { subject: 'SQL Inj', A: 99, fullMark: 150 },
    { subject: 'XSS', A: 85, fullMark: 150 },
    { subject: 'Brute Force', A: 65, fullMark: 150 },
];

// Mock recent alerts
// TODO: Replace with API call to GET /api/alerts/recent
export const MOCK_ALERTS: AlertLog[] = [
    { id: '#FL-2093', severity: 'High', type: 'DDoS Attempt', status: 'Blocked', timestamp: '2023-10-27 14:30' },
    { id: '#FL-2094', severity: 'Medium', type: 'SQL Injection', status: 'Flagged', timestamp: '2023-10-27 14:28' },
    { id: '#FL-2095', severity: 'Low', type: 'Port Scan', status: 'Monitored', timestamp: '2023-10-27 14:15' },
    { id: '#FL-2096', severity: 'High', type: 'Malware Payload', status: 'Blocked', timestamp: '2023-10-27 13:50' },
    { id: '#FL-2097', severity: 'Medium', type: 'Brute Force', status: 'Flagged', timestamp: '2023-10-27 13:45' },
];

// API endpoints
export const API_BASE_URL = 'http://localhost:8000';

export const API_ENDPOINTS = {
    LOGIN: `${API_BASE_URL}/auth/login`,
    HEALTH: `${API_BASE_URL}/health`,
    PREDICT: `${API_BASE_URL}/predict`,
    MODELS: `${API_BASE_URL}/models`,
    STATS: `${API_BASE_URL}/stats`,
};

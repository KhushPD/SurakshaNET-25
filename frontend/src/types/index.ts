/**
 * Type Definitions
 * ================
 * Shared TypeScript interfaces used across the application.
 * These match the backend API response structures.
 */

// Radar chart data point
export interface RadarDataPoint {
  subject: string;
  A: number;
  fullMark: number;
}

// Dashboard KPI statistics
export interface DashboardStats {
  totalRecords: string;
  maliciousFlows: string;
  recallScore: string;
  falseNegatives: number;
}

// Alert/threat log entry
export interface AlertLog {
  id: string;
  severity: 'High' | 'Medium' | 'Low';
  type: string;
  status: string;
  timestamp: string;
}

// Component props for StatCard
export interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: string;
  subtext?: string;
  alert?: boolean;
  warning?: boolean;
}

// Navigation item props
export interface NavItemProps {
  id: string;
  label: string;
  icon: React.ReactNode;
  activeTab: string;
  onClick: (id: string) => void;
}

// User authentication state
export interface User {
  username: string;
  token?: string;
}

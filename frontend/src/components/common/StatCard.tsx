/**
 * StatCard Component
 * ==================
 * Reusable card component for displaying KPI statistics.
 * Shows title, value, icon, and optional trend/subtext.
 * Enhanced with glassmorphism design.
 */

import type { StatCardProps } from '../../types';

const StatCard: React.FC<StatCardProps> = ({
    title,
    value,
    icon,
    trend,
    subtext,
    alert,
    warning
}) => (
    <div
        className={`p-5 rounded-xl shadow-lg border backdrop-blur-xl transition-all duration-300 hover:scale-105 hover:shadow-xl ${
            alert 
                ? 'bg-red-50/80 dark:bg-red-900/30 border-red-200/50 dark:border-red-700/50' 
                : warning 
                ? 'bg-orange-50/80 dark:bg-orange-900/30 border-orange-200/50 dark:border-orange-700/50' 
                : 'bg-white/80 dark:bg-gray-800/40 border-gray-200/50 dark:border-gray-700/50'
        }`}
    >
        <div className="flex justify-between items-start">
            <div>
                <p className={`text-sm font-medium mb-1 ${
                    alert ? 'text-red-600 dark:text-red-400' 
                    : warning ? 'text-orange-600 dark:text-orange-400' 
                    : 'text-gray-600 dark:text-gray-400'
                }`}>
                    {title}
                </p>
                <h3 className={`text-2xl font-bold ${
                    alert ? 'text-red-700 dark:text-red-300' 
                    : warning ? 'text-orange-700 dark:text-orange-300' 
                    : 'text-gray-900 dark:text-white'
                }`}>
                    {value}
                </h3>
            </div>
            <div
                className={`p-2 rounded-lg backdrop-blur-sm ${
                    alert ? 'bg-red-100/80 dark:bg-red-800/30' 
                    : warning ? 'bg-orange-100/80 dark:bg-orange-800/30' 
                    : 'bg-gray-100/80 dark:bg-gray-700/30'
                }`}
            >
                {icon}
            </div>
        </div>
        <div className="mt-4 flex items-center text-xs">
            {trend && (
                <span className="text-green-600 dark:text-green-400 font-medium bg-green-50/80 dark:bg-green-900/30 px-2 py-0.5 rounded mr-2 backdrop-blur-sm">
                    {trend}
                </span>
            )}
            {subtext && (
                <span
                    className={`${
                        alert ? 'text-red-600 dark:text-red-400' 
                        : warning ? 'text-orange-600 dark:text-orange-400' 
                        : 'text-gray-500 dark:text-gray-400'
                    }`}
                >
                    {subtext}
                </span>
            )}
        </div>
    </div>
);

export default StatCard;

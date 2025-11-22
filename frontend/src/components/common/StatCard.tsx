/**
 * StatCard Component
 * ==================
 * Reusable card component for displaying KPI statistics.
 * Shows title, value, icon, and optional trend/subtext.
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
        className={`bg-white p-5 rounded-xl shadow-sm border ${alert ? 'border-red-100' : warning ? 'border-orange-100' : 'border-gray-100'
            } hover:shadow-md transition-shadow`}
    >
        <div className="flex justify-between items-start">
            <div>
                <p className="text-sm font-medium text-gray-500 mb-1">{title}</p>
                <h3 className="text-2xl font-bold text-gray-800">{value}</h3>
            </div>
            <div
                className={`p-2 rounded-lg ${alert ? 'bg-red-50' : warning ? 'bg-orange-50' : 'bg-gray-50'
                    }`}
            >
                {icon}
            </div>
        </div>
        <div className="mt-4 flex items-center text-xs">
            {trend && (
                <span className="text-green-600 font-medium bg-green-50 px-2 py-0.5 rounded mr-2">
                    {trend}
                </span>
            )}
            {subtext && (
                <span
                    className={`${alert ? 'text-red-500' : warning ? 'text-orange-500' : 'text-gray-400'
                        }`}
                >
                    {subtext}
                </span>
            )}
        </div>
    </div>
);

export default StatCard;

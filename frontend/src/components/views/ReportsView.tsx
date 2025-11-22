/**
 * Reports View
 * ============
 * View for generating compliance and analysis reports.
 * Placeholder - requires backend PDF generation capabilities.
 */

import { FileText } from 'lucide-react';

interface ReportsViewProps {
    isDarkMode: boolean;
}

const ReportsView: React.FC<ReportsViewProps> = ({ isDarkMode }) => (
    <div className={`animate-in fade-in duration-300 p-8 rounded-xl shadow-sm min-h-[400px] flex flex-col items-center justify-center text-center transition-colors ${isDarkMode
            ? 'bg-gray-900/80 border border-gray-800'
            : 'bg-white border border-gray-200'
        }`}>
        <div className={`p-4 rounded-full mb-4 ${isDarkMode
                ? 'bg-green-500/10 border border-green-500/20'
                : 'bg-green-100'
            }`}>
            <FileText className={`w-12 h-12 ${isDarkMode ? 'text-green-400' : 'text-green-700'
                }`} />
        </div>
        <h2 className={`text-2xl font-bold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'
            }`}>Compliance & Reporting</h2>
        <p className={`max-w-md ${isDarkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
            Generate ISO 27001 compliant security reports. Backend requires PDF generation capabilities.
        </p>
        <button className={`mt-6 px-6 py-2 border-2 font-medium rounded-lg transition-colors ${isDarkMode
                ? 'border-green-500 text-green-400 hover:bg-green-950/30'
                : 'border-green-700 text-green-700 hover:bg-green-50'
            }`}>
            Generate PDF Report
        </button>
    </div>
);

export default ReportsView;

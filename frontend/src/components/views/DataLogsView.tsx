/**
 * Data Logs View
 * ===============
 * View for accessing historical traffic data logs.
 * Placeholder - requires backend integration for paginated logs.
 */

import { Database } from 'lucide-react';

interface DataLogsViewProps {
    isDarkMode: boolean;
}

const DataLogsView: React.FC<DataLogsViewProps> = ({ isDarkMode }) => (
    <div className={`animate-in fade-in duration-300 p-8 rounded-xl shadow-sm min-h-[400px] flex flex-col items-center justify-center text-center transition-colors ${isDarkMode
            ? 'bg-gray-900/80 border border-gray-800'
            : 'bg-white border border-gray-200'
        }`}>
        <div className={`p-4 rounded-full mb-4 ${isDarkMode
                ? 'bg-green-500/10 border border-green-500/20'
                : 'bg-green-100'
            }`}>
            <Database className={`w-12 h-12 ${isDarkMode ? 'text-green-400' : 'text-green-700'
                }`} />
        </div>
        <h2 className={`text-2xl font-bold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'
            }`}>Data Logs Repository</h2>
        <p className={`max-w-md ${isDarkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
            Access historical network traffic data. Backend integration required to fetch paginated logs.
        </p>
        <button className={`mt-6 px-6 py-2 rounded-lg transition-colors ${isDarkMode
                ? 'bg-green-600 hover:bg-green-500 text-white'
                : 'bg-green-700 hover:bg-green-600 text-white'
            }`}>
            Initiate Query
        </button>
    </div>
);

export default DataLogsView;

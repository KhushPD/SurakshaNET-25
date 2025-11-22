/**
 * Threat Intelligence View
 * =========================
 * View for global threat intelligence feeds.
 * Placeholder - requires backend integration for real-time threat data.
 */

import { Radar as RadarIcon } from 'lucide-react';

interface ThreatIntelViewProps {
    isDarkMode: boolean;
}

const ThreatIntelView: React.FC<ThreatIntelViewProps> = ({ isDarkMode }) => (
    <div className={`animate-in fade-in duration-300 p-8 rounded-xl shadow-sm min-h-[400px] flex flex-col items-center justify-center text-center transition-colors ${isDarkMode
            ? 'bg-gray-900/80 border border-gray-800'
            : 'bg-white border border-gray-200'
        }`}>
        <div className={`p-4 rounded-full mb-4 ${isDarkMode
                ? 'bg-green-500/10 border border-green-500/20'
                : 'bg-green-100'
            }`}>
            <RadarIcon className={`w-12 h-12 ${isDarkMode ? 'text-green-400' : 'text-green-700'
                }`} />
        </div>
        <h2 className={`text-2xl font-bold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'
            }`}>Global Threat Intelligence</h2>
        <p className={`max-w-md ${isDarkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
            Real-time threat feeds from global security partners and ML model insights.
        </p>
    </div>
);

export default ThreatIntelView;

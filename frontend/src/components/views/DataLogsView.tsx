/**
 * Data Logs View
 * ===============
 * View for accessing historical traffic data logs.
 */

import { Database, Play, RefreshCw, Trash2, Activity, Sparkles } from 'lucide-react';
import { useState, useEffect } from 'react';

interface DataLogsViewProps {
    isDarkMode: boolean;
    onAskAI?: (log: NetworkLog) => void;
}

interface NetworkLog {
    id: string;
    timestamp: string;
    source_ip: string;
    destination_ip: string;
    status: string;
    intrusion_type?: string;
    confidence?: number;
    request_type?: string;
    response_code?: number;
    duration?: number;
}

interface NetworkStats {
    total_logs: number;
    total_intrusions: number;
    total_benign: number;
    intrusion_rate: number;
}

const DataLogsView: React.FC<DataLogsViewProps> = ({ isDarkMode, onAskAI }) => {
    const [logs, setLogs] = useState<NetworkLog[]>([]);
    const [stats, setStats] = useState<NetworkStats | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isTesting, setIsTesting] = useState(false);

    const API_BASE = 'http://localhost:8000';

    const fetchLogs = async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE}/network/logs?limit=100`);
            const data = await response.json();
            setLogs(data.logs || []);
            localStorage.setItem('networkLogs', JSON.stringify(data.logs || []));
        } catch (error) {
            console.error('Error fetching logs:', error);
            // Load from localStorage if API fails
            const cached = localStorage.getItem('networkLogs');
            if (cached) setLogs(JSON.parse(cached));
        }
        setIsLoading(false);
    };

    const fetchStats = async () => {
        try {
            const response = await fetch(`${API_BASE}/network/stats`);
            const data = await response.json();
            setStats(data.statistics);
        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    };

    const runTest = async (testType: string) => {
        setIsTesting(true);
        try {
            const response = await fetch(`${API_BASE}/network/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    traffic_type: testType,
                    requests_count: testType === 'mixed' ? 50 : 30,
                    attack_ratio: 0.3
                })
            });
            const data = await response.json();
            alert(data.message);
            await fetchLogs();
            await fetchStats();
        } catch (error) {
            console.error('Error running test:', error);
            alert('Test failed. Make sure backend is running.');
        }
        setIsTesting(false);
    };

    const clearLogs = async () => {
        if (!confirm('Clear all logs?')) return;
        try {
            await fetch(`${API_BASE}/network/logs`, { method: 'DELETE' });
            setLogs([]);
            localStorage.removeItem('networkLogs');
            await fetchStats();
        } catch (error) {
            console.error('Error clearing logs:', error);
        }
    };

    useEffect(() => {
        fetchLogs();
        fetchStats();
    }, []);

    return (
        <div className="space-y-6 p-6">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                    <div className="text-sm opacity-70">Total Logs</div>
                    <div className="text-2xl font-bold">{stats?.total_logs || 0}</div>
                </div>
                <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-red-900/30' : 'bg-red-50'}`}>
                    <div className="text-sm opacity-70">Intrusions</div>
                    <div className="text-2xl font-bold text-red-500">{stats?.total_intrusions || 0}</div>
                </div>
                <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-green-900/30' : 'bg-green-50'}`}>
                    <div className="text-sm opacity-70">Benign</div>
                    <div className="text-2xl font-bold text-green-500">{stats?.total_benign || 0}</div>
                </div>
                <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-blue-900/30' : 'bg-blue-50'}`}>
                    <div className="text-sm opacity-70">Attack Rate</div>
                    <div className="text-2xl font-bold text-blue-500">
                        {((stats?.intrusion_rate || 0) * 100).toFixed(1)}%
                    </div>
                </div>
            </div>

            {/* Test Buttons */}
            <div className={`p-6 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    Network Traffic Tests
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
                    <button
                        onClick={() => runTest('normal')}
                        disabled={isTesting}
                        className={`px-4 py-2 rounded-lg transition-colors ${isDarkMode
                            ? 'bg-green-600 hover:bg-green-500'
                            : 'bg-green-700 hover:bg-green-600'
                        } text-white disabled:opacity-50`}
                    >
                        Normal
                    </button>
                    <button
                        onClick={() => runTest('http_flood')}
                        disabled={isTesting}
                        className={`px-4 py-2 rounded-lg transition-colors ${isDarkMode
                            ? 'bg-orange-600 hover:bg-orange-500'
                            : 'bg-orange-700 hover:bg-orange-600'
                        } text-white disabled:opacity-50`}
                    >
                        HTTP Flood
                    </button>
                    <button
                        onClick={() => runTest('port_scan')}
                        disabled={isTesting}
                        className={`px-4 py-2 rounded-lg transition-colors ${isDarkMode
                            ? 'bg-red-600 hover:bg-red-500'
                            : 'bg-red-700 hover:bg-red-600'
                        } text-white disabled:opacity-50`}
                    >
                        Port Scan
                    </button>
                    <button
                        onClick={() => runTest('sql_injection')}
                        disabled={isTesting}
                        className={`px-4 py-2 rounded-lg transition-colors ${isDarkMode
                            ? 'bg-purple-600 hover:bg-purple-500'
                            : 'bg-purple-700 hover:bg-purple-600'
                        } text-white disabled:opacity-50`}
                    >
                        SQL Injection
                    </button>
                    <button
                        onClick={() => runTest('xss_attack')}
                        disabled={isTesting}
                        className={`px-4 py-2 rounded-lg transition-colors ${isDarkMode
                            ? 'bg-pink-600 hover:bg-pink-500'
                            : 'bg-pink-700 hover:bg-pink-600'
                        } text-white disabled:opacity-50`}
                    >
                        XSS Attack
                    </button>
                    <button
                        onClick={() => runTest('mixed')}
                        disabled={isTesting}
                        className={`px-4 py-2 rounded-lg transition-colors ${isDarkMode
                            ? 'bg-blue-600 hover:bg-blue-500'
                            : 'bg-blue-700 hover:bg-blue-600'
                        } text-white disabled:opacity-50`}
                    >
                        Mixed Traffic
                    </button>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3">
                <button
                    onClick={fetchLogs}
                    disabled={isLoading}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${isDarkMode
                        ? 'bg-blue-600 hover:bg-blue-500'
                        : 'bg-blue-700 hover:bg-blue-600'
                    } text-white disabled:opacity-50`}
                >
                    <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
                <button
                    onClick={clearLogs}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${isDarkMode
                        ? 'bg-red-600 hover:bg-red-500'
                        : 'bg-red-700 hover:bg-red-600'
                    } text-white`}
                >
                    <Trash2 className="w-4 h-4" />
                    Clear Logs
                </button>
            </div>

            {/* Logs Table */}
            <div className={`rounded-lg overflow-hidden ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className={isDarkMode ? 'bg-gray-900' : 'bg-gray-100'}>
                            <tr>
                                <th className="px-4 py-3 text-left text-sm font-semibold">Time</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold">Source IP</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold">Dest IP</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold">Status</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold">Type</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold">Details</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {logs.length === 0 ? (
                                <tr>
                                    <td colSpan={7} className="px-4 py-8 text-center opacity-50">
                                        No logs available. Run a test to generate traffic.
                                    </td>
                                </tr>
                            ) : (
                                logs.map((log) => (
                                    <tr
                                        key={log.id}
                                        className={`border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'
                                        }`}
                                    >
                                        <td className="px-4 py-3 text-sm">
                                            {new Date(log.timestamp).toLocaleTimeString()}
                                        </td>
                                        <td className="px-4 py-3 text-sm font-mono">{log.source_ip}</td>
                                        <td className="px-4 py-3 text-sm font-mono">{log.destination_ip}</td>
                                        <td className="px-4 py-3">
                                            <span
                                                className={`px-2 py-1 rounded text-xs font-semibold ${log.status === 'detected'
                                                    ? 'bg-red-500/20 text-red-500'
                                                    : 'bg-green-500/20 text-green-500'
                                                }`}
                                            >
                                                {log.status}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-sm">
                                            {log.intrusion_type || log.request_type || '-'}
                                        </td>
                                        <td className="px-4 py-3 text-sm">
                                            {log.confidence
                                                ? `Confidence: ${(log.confidence * 100).toFixed(0)}%`
                                                : log.response_code
                                                ? `HTTP ${log.response_code}`
                                                : '-'}
                                        </td>
                                        <td className="px-4 py-3">
                                            <button
                                                onClick={() => onAskAI?.(log)}
                                                className={`flex items-center gap-1 px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                                                    isDarkMode
                                                        ? 'bg-green-600 hover:bg-green-500 text-white'
                                                        : 'bg-green-700 hover:bg-green-600 text-white'
                                                }`}
                                                title="Analyze with AI"
                                            >
                                                <Sparkles className="w-3 h-3" />
                                                Ask AI
                                            </button>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default DataLogsView;

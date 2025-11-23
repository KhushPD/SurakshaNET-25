/**
 * DashboardHome - Real-Time Monitoring Dashboard
 * ==============================================
 * Displays real-time network traffic analysis and visualizations.
 */

import { useState, useEffect } from 'react';
import StatCard from '../common/StatCard';
import { Activity, Shield, AlertTriangle, TrendingUp } from 'lucide-react';

interface DashboardHomeProps {
    isDarkMode: boolean;
}

interface RealtimeMetrics {
    total_processed: number;
    total_attacks: number;
    total_normal: number;
    attack_rate_percent: number;
    last_update: string;
}

interface RealtimePlots {
    spider_plot: string;
    pie_chart: string;
    attack_type_distribution: string;
    binary_classification: string;
    timeline: string;
    prediction_confidence: string;
}

interface AttackTypeData {
    id: number;
    type: string;
    count: number;
    percentage: number;
}

const DashboardHome: React.FC<DashboardHomeProps> = ({ isDarkMode }) => {
    const [metrics, setMetrics] = useState<RealtimeMetrics | null>(null);
    const [plots, setPlots] = useState<RealtimePlots | null>(null);
    const [attackTypes, setAttackTypes] = useState<AttackTypeData[] | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const API_BASE_URL = 'http://localhost:8000';

    // Fetch real-time data
    const fetchRealtimeData = async () => {
        try {
            // Fetch visualizations and metrics
            const vizResponse = await fetch(`${API_BASE_URL}/realtime/visualizations`);

            if (!vizResponse.ok) {
                throw new Error('Failed to fetch real-time data');
            }

            const vizData = await vizResponse.json();

            if (vizData.status === 'success') {
                setMetrics(vizData.metrics_summary);
                setPlots(vizData.plots);
                setError(null);
            }

            // Fetch attack type breakdown
            try {
                const attackTypesResponse = await fetch(`${API_BASE_URL}/realtime/attack-types`);
                if (attackTypesResponse.ok) {
                    const attackTypesData = await attackTypesResponse.json();
                    if (attackTypesData.status === 'success') {
                        setAttackTypes(attackTypesData.attack_types);
                    }
                }
            } catch (attackErr) {
                console.error('Error fetching attack types:', attackErr);
                // Don't set error state, just log it
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
            console.error('Error fetching real-time data:', err);
        } finally {
            setLoading(false);
        }
    };

    // Initial fetch and auto-refresh every 5 seconds
    useEffect(() => {
        fetchRealtimeData();
        const interval = setInterval(fetchRealtimeData, 5000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
                    <p className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>
                        Loading real-time data...
                    </p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className={`p-6 rounded-lg ${isDarkMode ? 'bg-red-900/20 text-red-300' : 'bg-red-100 text-red-700'}`}>
                    <AlertTriangle className="w-12 h-12 mx-auto mb-4" />
                    <p className="text-center font-semibold">Error: {error}</p>
                    <p className="text-center text-sm mt-2">Make sure the backend server is running on port 8000</p>
                </div>
            </div>
        );
    }

    return (
        <div className={`min-h-screen p-6 ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-bold mb-2">Real-Time Network Monitoring</h1>
                <p className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>
                    Live threat detection and analysis
                </p>
            </div>

            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard
                    title="Total Processed"
                    value={(metrics?.total_processed ?? 0).toLocaleString()}
                    icon={<Activity className="w-6 h-6" />}
                    subtext="Network packets"
                />
                <StatCard
                    title="Normal Traffic"
                    value={(metrics?.total_normal ?? 0).toLocaleString()}
                    icon={<Shield className="w-6 h-6" />}
                    subtext="Safe connections"
                />
                <StatCard
                    title="Threats Detected"
                    value={(metrics?.total_attacks ?? 0).toLocaleString()}
                    icon={<AlertTriangle className="w-6 h-6" />}
                    alert={true}
                    subtext="Malicious activity"
                />
                <StatCard
                    title="Attack Rate"
                    value={`${(metrics?.attack_rate_percent ?? 0).toFixed(2)}%`}
                    icon={<TrendingUp className="w-6 h-6" />}
                    warning={metrics ? metrics.attack_rate_percent > 10 : false}
                    subtext="Threat percentage"
                />
            </div>

            {/* Visualization Plots */}
            {plots && Object.keys(plots).length > 0 ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Spider/Radar Plot */}
                    {plots.spider_plot && (
                        <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                            <h2 className="text-xl font-semibold mb-4">Threat Vector Analysis</h2>
                            <img
                                src={plots.spider_plot}
                                alt="Threat Vector Analysis"
                                className="w-full h-auto"
                            />
                        </div>
                    )}

                    {/* Pie Chart */}
                    {plots.pie_chart && (
                        <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                            <h2 className="text-xl font-semibold mb-4">Traffic Distribution</h2>
                            <img
                                src={plots.pie_chart}
                                alt="Traffic Distribution"
                                className="w-full h-auto"
                            />
                        </div>
                    )}

                    {/* Attack Type Distribution */}
                    {plots.attack_type_distribution && (
                        <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                            <h2 className="text-xl font-semibold mb-4">Attack Type Distribution</h2>
                            <img
                                src={plots.attack_type_distribution}
                                alt="Attack Type Distribution"
                                className="w-full h-auto"
                            />
                        </div>
                    )}

                    {/* Binary Classification */}
                    {plots.binary_classification && (
                        <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                            <h2 className="text-xl font-semibold mb-4">Binary Classification Metrics</h2>
                            <img
                                src={plots.binary_classification}
                                alt="Binary Classification Metrics"
                                className="w-full h-auto"
                            />
                        </div>
                    )}

                    {/* Timeline */}
                    {plots.timeline && (
                        <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                            <h2 className="text-xl font-semibold mb-4">Detection Timeline</h2>
                            <img
                                src={plots.timeline}
                                alt="Detection Timeline"
                                className="w-full h-auto"
                            />
                        </div>
                    )}

                    {/* Prediction Confidence */}
                    {plots.prediction_confidence && (
                        <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                            <h2 className="text-xl font-semibold mb-4">Prediction Confidence</h2>
                            <img
                                src={plots.prediction_confidence}
                                alt="Prediction Confidence"
                                className="w-full h-auto"
                            />
                        </div>
                    )}
                </div>
            ) : (
                <div className={`text-center py-12 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    <p className="text-lg mb-2">No visualization data available yet</p>
                    <p className="text-sm">Start real-time monitoring to see live plots</p>
                </div>
            )}

            {/* Attack Type Breakdown */}
            {attackTypes && attackTypes.length > 0 && (
                <div className="mt-8">
                    <h2 className="text-2xl font-bold mb-4">Attack Type Breakdown</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                        {attackTypes.map((attackType) => {
                            // Define colors based on attack type
                            const getTypeColor = (type: string) => {
                                switch (type) {
                                    case 'Normal':
                                        return isDarkMode
                                            ? 'bg-green-900/30 border-green-700'
                                            : 'bg-green-50 border-green-300';
                                    case 'DoS':
                                        return isDarkMode
                                            ? 'bg-red-900/30 border-red-700'
                                            : 'bg-red-50 border-red-300';
                                    case 'Probe':
                                        return isDarkMode
                                            ? 'bg-orange-900/30 border-orange-700'
                                            : 'bg-orange-50 border-orange-300';
                                    case 'R2L':
                                        return isDarkMode
                                            ? 'bg-purple-900/30 border-purple-700'
                                            : 'bg-purple-50 border-purple-300';
                                    case 'U2R':
                                        return isDarkMode
                                            ? 'bg-yellow-900/30 border-yellow-700'
                                            : 'bg-yellow-50 border-yellow-300';
                                    default:
                                        return isDarkMode
                                            ? 'bg-gray-800 border-gray-700'
                                            : 'bg-gray-100 border-gray-300';
                                }
                            };

                            const getTextColor = (type: string) => {
                                switch (type) {
                                    case 'Normal':
                                        return isDarkMode ? 'text-green-300' : 'text-green-700';
                                    case 'DoS':
                                        return isDarkMode ? 'text-red-300' : 'text-red-700';
                                    case 'Probe':
                                        return isDarkMode ? 'text-orange-300' : 'text-orange-700';
                                    case 'R2L':
                                        return isDarkMode ? 'text-purple-300' : 'text-purple-700';
                                    case 'U2R':
                                        return isDarkMode ? 'text-yellow-300' : 'text-yellow-700';
                                    default:
                                        return isDarkMode ? 'text-gray-300' : 'text-gray-700';
                                }
                            };

                            return (
                                <div
                                    key={attackType.id}
                                    className={`rounded-lg border-2 p-4 ${getTypeColor(attackType.type)}`}
                                >
                                    <h3 className={`text-lg font-semibold mb-2 ${getTextColor(attackType.type)}`}>
                                        {attackType.type}
                                    </h3>
                                    <div className={`text-3xl font-bold mb-1 ${getTextColor(attackType.type)}`}>
                                        {attackType.count.toLocaleString()}
                                    </div>
                                    <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                        {attackType.percentage.toFixed(2)}% of total
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Last Update Info */}
            {metrics?.last_update && (
                <div className={`mt-6 text-center text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                    Last updated: {new Date(metrics.last_update).toLocaleString()}
                </div>
            )}
        </div>
    );
};

export default DashboardHome;

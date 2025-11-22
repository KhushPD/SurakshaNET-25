/**
 * Reports View - Security Analysis & Reports
 * ===========================================
 * Complete security analysis interface with CSV upload and ML-powered threat detection.
 * Displays visualizations, statistics, and detailed predictions.
 * (Exact copy of DashboardHomeNew functionality)
 */

import { useState, useEffect } from 'react';
import {
    Upload,
    Activity,
    Shield,
    AlertTriangle,
    TrendingUp,
    Target,
    Radio,
    PieChart as PieChartIcon,
    BarChart3,
    Clock,
    Brain,
    CheckCircle,
    Save,
    Trash2,
    History
} from 'lucide-react';

/**
 * Custom hook for count-up animation
 * Animates a number from 0 to target value
 */
const useCountUp = (end: number, duration: number = 2000, decimals: number = 0) => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        let startTime: number | null = null;
        let animationFrame: number;

        const animate = (currentTime: number) => {
            if (!startTime) startTime = currentTime;
            const progress = Math.min((currentTime - startTime) / duration, 1);

            // Easing function for smooth animation (easeOutQuart)
            const easeOut = 1 - Math.pow(1 - progress, 4);
            const currentCount = easeOut * end;

            setCount(currentCount);

            if (progress < 1) {
                animationFrame = requestAnimationFrame(animate);
            } else {
                setCount(end); // Ensure we end exactly at the target
            }
        };

        animationFrame = requestAnimationFrame(animate);

        return () => {
            if (animationFrame) {
                cancelAnimationFrame(animationFrame);
            }
        };
    }, [end, duration]);

    return decimals > 0 ? count.toFixed(decimals) : Math.floor(count).toLocaleString();
};

interface PredictionData {
    summary: {
        total_samples: number;
        attack_percentage: number;
        normal_percentage: number;
        binary_distribution: Array<{ label: string; count: number; percentage: number }>;
        multiclass_distribution: Array<{ label: string; count: number; percentage: number }>;
    };
    plots: {
        spider_plot: string;
        pie_chart: string;
        timeline: string;
        attack_type_distribution: string;
        prediction_confidence: string;
        binary_classification: string;
    };
}

interface SavedReport {
    id: string;
    filename: string;
    timestamp: string;
    data: PredictionData;
}

const STORAGE_KEY = 'surakshaNET_reports';
const MAX_REPORTS = 5; // Limit to prevent localStorage overflow

interface ReportsViewProps {
    isDarkMode: boolean;
}

const ReportsView: React.FC<ReportsViewProps> = ({ isDarkMode }) => {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<PredictionData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [savedReports, setSavedReports] = useState<SavedReport[]>([]);
    const [showHistory, setShowHistory] = useState(false);

    // Load saved reports from localStorage on mount
    useEffect(() => {
        loadSavedReports();
        // Try to load last report
        const lastReport = getLastReport();
        if (lastReport) {
            setData(lastReport.data);
            console.log('📂 Loaded last report from localStorage:', lastReport.filename);
        }
    }, []);

    // Save to localStorage whenever data changes
    useEffect(() => {
        if (data && file) {
            saveReport(file.name, data);
        }
    }, [data, file]);

    // LocalStorage functions
    const loadSavedReports = () => {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                const reports = JSON.parse(stored) as SavedReport[];
                setSavedReports(reports);
                console.log(`📚 Loaded ${reports.length} saved reports from localStorage`);
            }
        } catch (err) {
            console.error('Failed to load saved reports:', err);
        }
    };

    const saveReport = (filename: string, reportData: PredictionData) => {
        try {
            const newReport: SavedReport = {
                id: Date.now().toString(),
                filename,
                timestamp: new Date().toISOString(),
                data: reportData
            };

            const stored = localStorage.getItem(STORAGE_KEY);
            let reports = stored ? JSON.parse(stored) as SavedReport[] : [];

            // Check if report with same filename already exists
            reports = reports.filter(r => r.filename !== filename);

            // Add new report at the beginning
            reports.unshift(newReport);

            // Keep only MAX_REPORTS
            if (reports.length > MAX_REPORTS) {
                reports = reports.slice(0, MAX_REPORTS);
            }

            localStorage.setItem(STORAGE_KEY, JSON.stringify(reports));
            setSavedReports(reports);
            console.log('💾 Report saved to localStorage:', filename);
        } catch (err) {
            console.error('Failed to save report:', err);
            // If storage is full, try to clear old reports
            if (err instanceof Error && err.name === 'QuotaExceededError') {
                clearOldReports();
            }
        }
    };

    const getLastReport = (): SavedReport | null => {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                const reports = JSON.parse(stored) as SavedReport[];
                return reports.length > 0 ? reports[0] : null;
            }
        } catch (err) {
            console.error('Failed to get last report:', err);
        }
        return null;
    };

    const loadReport = (report: SavedReport) => {
        setData(report.data);
        setShowHistory(false);
        console.log('📂 Loaded report:', report.filename);
    };

    const deleteReport = (reportId: string) => {
        try {
            const reports = savedReports.filter(r => r.id !== reportId);
            localStorage.setItem(STORAGE_KEY, JSON.stringify(reports));
            setSavedReports(reports);
            console.log('🗑️ Deleted report:', reportId);
        } catch (err) {
            console.error('Failed to delete report:', err);
        }
    };

    const clearAllReports = () => {
        try {
            localStorage.removeItem(STORAGE_KEY);
            setSavedReports([]);
            console.log('🗑️ Cleared all saved reports');
        } catch (err) {
            console.error('Failed to clear reports:', err);
        }
    };

    const clearOldReports = () => {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                const reports = JSON.parse(stored) as SavedReport[];
                const reduced = reports.slice(0, 2); // Keep only 2 most recent
                localStorage.setItem(STORAGE_KEY, JSON.stringify(reduced));
                setSavedReports(reduced);
                console.log('🧹 Cleared old reports, kept 2 most recent');
            }
        } catch (err) {
            console.error('Failed to clear old reports:', err);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError(null);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError('Please select a CSV file');
            return;
        }

        setLoading(true);
        setError(null);
        console.log('🚀 Starting upload:', file.name);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData,
            });

            console.log('📡 Response:', response.status);

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();
            console.log('✅ Success! Plots:', Object.keys(result.plots || {}));

            setData(result);
            // Report will be auto-saved by useEffect
        } catch (err) {
            console.error('❌ Error:', err);
            setError(err instanceof Error ? err.message : 'Upload failed');
        } finally {
            setLoading(false);
        }
    };

    const handleClearData = () => {
        setData(null);
        setFile(null);
        setError(null);
    };

    // StatCard component with count-up animation
    const StatCard: React.FC<{
        icon: any;
        label: string;
        value: number;
        suffix?: string;
        decimals?: number;
    }> = ({ icon: Icon, label, value, suffix = '', decimals = 0 }) => {
        const animatedValue = useCountUp(value, 2000, decimals);

        return (
            <div className={`rounded-lg p-5 hover:shadow-lg transition-all duration-200 ${isDarkMode
                ? 'bg-gray-900/80 border border-gray-800 hover:border-green-900/40'
                : 'bg-white border border-gray-200 hover:border-green-300'
                }`}>
                <div className="flex items-center justify-between">
                    <div>
                        <p className={`text-xs uppercase tracking-wider mb-1.5 ${isDarkMode ? 'text-gray-500' : 'text-gray-600'
                            }`}>{label}</p>
                        <p className={`text-2xl font-bold ${isDarkMode ? 'text-green-400' : 'text-green-600'
                            }`}>
                            {animatedValue}{suffix}
                        </p>
                    </div>
                    <div className={`p-2.5 rounded-lg ${isDarkMode
                        ? 'bg-green-500/10 border border-green-500/20'
                        : 'bg-green-100 border border-green-200'
                        }`}>
                        <Icon className={`w-5 h-5 ${isDarkMode ? 'text-green-400' : 'text-green-600'
                            }`} />
                    </div>
                </div>
            </div>
        );
    };

    const renderStats = () => {
        return (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <StatCard
                    icon={Activity}
                    label="Total Samples"
                    value={data?.summary.total_samples || 0}
                />
                <StatCard
                    icon={Shield}
                    label="Normal Traffic"
                    value={data?.summary.normal_percentage || 0}
                    suffix="%"
                    decimals={1}
                />
                <StatCard
                    icon={AlertTriangle}
                    label="Attack Traffic"
                    value={data?.summary.attack_percentage || 0}
                    suffix="%"
                    decimals={1}
                />
                <StatCard
                    icon={TrendingUp}
                    label="Detection Rate"
                    value={data ? 100 : 0}
                    suffix="%"
                    decimals={0}
                />
            </div>
        );
    };

    const plots = [
        { key: 'spider_plot', title: 'Spider Plot', subtitle: 'Threat Vector Analysis', icon: Radio },
        { key: 'pie_chart', title: 'Pie Chart', subtitle: 'Binary Classification', icon: PieChartIcon },
        { key: 'timeline', title: 'Timeline', subtitle: 'Attack Detection Over Time', icon: Clock },
        { key: 'attack_type_distribution', title: 'Attack Types', subtitle: 'Multi-class Distribution', icon: BarChart3 },
        { key: 'prediction_confidence', title: 'Confidence', subtitle: 'Prediction Scores', icon: Brain },
        { key: 'binary_classification', title: 'Binary Metrics', subtitle: 'Detailed Classification', icon: CheckCircle }
    ];

    return (
        <div className="min-h-screen p-6">
            {/* Header */}
            <div className="mb-10">
                <h1 className={`text-3xl font-bold mb-1 ${isDarkMode ? 'text-green-400' : 'text-green-600'
                    }`}>Network Intrusion Detection</h1>
                <p className={`text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-600'
                    }`}>Upload CSV data to analyze threats with ML models</p>
            </div>

            {/* Upload Section */}
            <div className={`rounded-xl p-6 mb-6 shadow-lg transition-colors ${isDarkMode
                ? 'bg-gray-900/90 border border-gray-800'
                : 'bg-white border border-gray-200'
                }`}>
                <div className="flex items-center justify-between mb-5">
                    <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg ${isDarkMode
                            ? 'bg-green-500/10 border border-green-500/20'
                            : 'bg-green-100 border border-green-200'
                            }`}>
                            <Upload className={`w-5 h-5 ${isDarkMode ? 'text-green-400' : 'text-green-600'
                                }`} />
                        </div>
                        <div>
                            <h2 className={`text-lg font-semibold ${isDarkMode ? 'text-gray-200' : 'text-gray-800'
                                }`}>Upload Dataset</h2>
                            <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-600'
                                }`}>Select CSV file with network traffic data</p>
                        </div>
                    </div>

                    {/* History Button */}
                    {savedReports.length > 0 && (
                        <button
                            onClick={() => setShowHistory(!showHistory)}
                            className={`flex items-center gap-2 px-3 py-1.5 border rounded-lg transition text-xs ${isDarkMode
                                ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600'
                                : 'bg-gray-100 border-gray-300 hover:bg-gray-200'
                                }`}
                        >
                            <History className={`w-4 h-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'
                                }`} />
                            <span className={`font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-700'
                                }`}>
                                History ({savedReports.length})
                            </span>
                        </button>
                    )}
                </div>

                {/* History Panel */}
                {showHistory && savedReports.length > 0 && (
                    <div className={`mb-5 p-3 rounded-lg border transition-colors ${isDarkMode
                        ? 'bg-gray-800/50 border-gray-800'
                        : 'bg-gray-50 border-gray-200'
                        }`}>
                        <div className="flex items-center justify-between mb-2">
                            <h3 className={`text-sm font-semibold ${isDarkMode ? 'text-gray-300' : 'text-gray-700'
                                }`}>Saved Reports</h3>
                            <button
                                onClick={clearAllReports}
                                className="text-xs text-red-400 hover:text-red-300 flex items-center gap-1"
                            >
                                <Trash2 className="w-3 h-3" />
                                Clear All
                            </button>
                        </div>
                        <div className="space-y-1.5">
                            {savedReports.map((report) => (
                                <div
                                    key={report.id}
                                    className={`flex items-center justify-between p-2.5 border rounded-md transition ${isDarkMode
                                        ? 'bg-gray-900/50 border-gray-800 hover:bg-gray-800 hover:border-gray-700'
                                        : 'bg-white border-gray-200 hover:bg-gray-50 hover:border-gray-300'
                                        }`}
                                >
                                    <div className="flex-1 min-w-0">
                                        <p className={`text-xs font-medium truncate ${isDarkMode ? 'text-gray-300' : 'text-gray-700'
                                            }`}>{report.filename}</p>
                                        <p className={`text-xs ${isDarkMode ? 'text-gray-600' : 'text-gray-500'
                                            }`}>
                                            {new Date(report.timestamp).toLocaleString()}
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-2 ml-3">
                                        <button
                                            onClick={() => loadReport(report)}
                                            className={`px-2.5 py-1 text-xs text-white rounded transition ${isDarkMode ? 'bg-green-600 hover:bg-green-500' : 'bg-green-700 hover:bg-green-600'
                                                }`}
                                        >
                                            Load
                                        </button>
                                        <button
                                            onClick={() => deleteReport(report.id)}
                                            className="p-1 text-red-400 hover:bg-red-950/30 rounded transition"
                                        >
                                            <Trash2 className="w-3.5 h-3.5" />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <div className="flex flex-col sm:flex-row gap-3">
                    <label className={`flex-1 flex items-center justify-center h-28 border-2 border-dashed rounded-lg cursor-pointer transition ${isDarkMode
                        ? 'border-gray-800 hover:border-green-900/50 hover:bg-gray-800/30'
                        : 'border-gray-300 hover:border-green-500 hover:bg-green-50'
                        }`}>
                        <div className="text-center">
                            <Upload className={`w-6 h-6 mx-auto mb-1.5 ${isDarkMode ? 'text-green-500/70' : 'text-green-600'
                                }`} />
                            <span className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'
                                }`}>{file ? file.name : 'Choose file or drag here'}</span>
                            <span className={`block text-xs mt-0.5 ${isDarkMode ? 'text-gray-600' : 'text-gray-500'
                                }`}>CSV files only • Auto-saved to browser</span>
                        </div>
                        <input type="file" className="hidden" accept=".csv" onChange={handleFileChange} />
                    </label>

                    <div className="flex flex-col gap-2 sm:w-40">
                        <button
                            onClick={handleUpload}
                            disabled={!file || loading}
                            className={`px-4 py-2.5 rounded-lg text-sm font-semibold text-white transition ${!file || loading
                                ? 'bg-gray-700 cursor-not-allowed'
                                : isDarkMode
                                    ? 'bg-green-600 hover:bg-green-500 hover:shadow-md'
                                    : 'bg-green-700 hover:bg-green-600 hover:shadow-md'
                                }`}
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                    Analyzing
                                </span>
                            ) : (
                                <span className="flex items-center justify-center gap-2">
                                    <Activity className="w-4 h-4" />
                                    Analyze
                                </span>
                            )}
                        </button>

                        {data && (
                            <button
                                onClick={handleClearData}
                                className={`px-4 py-2 border text-sm rounded-lg font-medium transition ${isDarkMode
                                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600 text-gray-300'
                                    : 'bg-white border-gray-300 hover:bg-gray-50 text-gray-700'
                                    }`}
                            >
                                Clear
                            </button>
                        )}
                    </div>
                </div>

                {error && (
                    <div className={`mt-3 p-3 border rounded-lg ${isDarkMode
                        ? 'bg-red-950/30 border-red-900/40'
                        : 'bg-red-50 border-red-200'
                        }`}>
                        <p className={`text-sm ${isDarkMode ? 'text-red-400' : 'text-red-700'
                            }`}>❌ {error}</p>
                    </div>
                )}
            </div>

            {/* Stats */}
            {renderStats()}

            {/* Visualization Grid */}
            {data && data.plots && (
                <div>
                    <div className="flex items-center justify-between mb-4">
                        <h2 className={`text-xl font-semibold flex items-center gap-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'
                            }`}>
                            <Target className={`w-5 h-5 ${isDarkMode ? 'text-green-500' : 'text-green-600'
                                }`} />
                            Visualization Dashboard
                        </h2>
                        <div className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border ${isDarkMode
                            ? 'text-gray-500 bg-gray-900/50 border-gray-800'
                            : 'text-gray-600 bg-gray-100 border-gray-200'
                            }`}>
                            <Save className={`w-3.5 h-3.5 ${isDarkMode ? 'text-green-500/70' : 'text-green-600'
                                }`} />
                            <span>Saved</span>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {plots.map((plot, index) => {
                            const Icon = plot.icon;
                            const imgSrc = data.plots[plot.key as keyof typeof data.plots];

                            return (
                                <div
                                    key={plot.key}
                                    className={`border rounded-lg overflow-hidden hover:shadow-lg transition-all duration-200 animate-fade-in-up ${isDarkMode
                                        ? 'bg-gray-900/80 border-gray-800 hover:border-gray-700'
                                        : 'bg-white border-gray-200 hover:border-gray-300'
                                        }`}
                                    style={{ animationDelay: `${index * 0.1}s` }}
                                >
                                    <div className={`border-l-4 px-4 py-3 ${isDarkMode
                                        ? 'border-green-500/70 bg-gray-850/50'
                                        : 'border-green-500 bg-green-50/50'
                                        }`}>
                                        <div className="flex items-center gap-2.5">
                                            <Icon className={`w-4 h-4 ${isDarkMode ? 'text-green-400' : 'text-green-600'
                                                }`} />
                                            <div>
                                                <h3 className={`text-sm font-semibold ${isDarkMode ? 'text-gray-200' : 'text-gray-800'
                                                    }`}>{plot.title}</h3>
                                                <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-600'
                                                    }`}>{plot.subtitle}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className={`p-4 ${isDarkMode ? 'bg-gray-900/40' : 'bg-gray-50/30'
                                        }`}>
                                        {imgSrc ? (
                                            <img
                                                src={imgSrc}
                                                alt={plot.title}
                                                className={`w-full h-auto rounded-md border animate-scale-in ${isDarkMode ? 'border-gray-800/50' : 'border-gray-200'
                                                    }`}
                                                style={{ animationDelay: `${index * 0.1 + 0.2}s` }}
                                                onLoad={() => console.log(`✅ ${plot.title} loaded`)}
                                                onError={() => console.error(`❌ Failed: ${plot.title}`)}
                                            />
                                        ) : (
                                            <div className={`flex items-center justify-center h-56 rounded-md border ${isDarkMode
                                                ? 'bg-gray-900/50 border-gray-800'
                                                : 'bg-gray-100 border-gray-200'
                                                }`}>
                                                <p className={`text-sm ${isDarkMode ? 'text-gray-600' : 'text-gray-500'
                                                    }`}>No data</p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Empty State */}
            {!data && !loading && (
                <div className="text-center py-16">
                    <div className={`inline-block p-5 rounded-full mb-4 border ${isDarkMode
                        ? 'bg-green-500/10 border-green-500/20'
                        : 'bg-green-100 border-green-200'
                        }`}>
                        <Activity className={`w-12 h-12 ${isDarkMode ? 'text-green-400' : 'text-green-600'
                            }`} />
                    </div>
                    <h3 className={`text-lg font-semibold mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'
                        }`}>Ready to Analyze</h3>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-600'
                        }`}>Upload a CSV file to start detection</p>
                </div>
            )}

            {/* Animation Styles */}
            <style>{`
                @keyframes fade-in-up {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }

                @keyframes scale-in {
                    from {
                        opacity: 0;
                        transform: scale(0.95);
                    }
                    to {
                        opacity: 1;
                        transform: scale(1);
                    }
                }

                .animate-fade-in-up {
                    animation: fade-in-up 0.6s ease-out forwards;
                    opacity: 0;
                }

                .animate-scale-in {
                    animation: scale-in 0.5s ease-out forwards;
                    opacity: 0;
                }
            `}</style>
        </div>
    );
};

export default ReportsView;

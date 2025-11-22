/**
 * DashboardHome - Complete Rebuild
 * =================================
 * Displays 6 visualization plots from uploaded CSV data:
 * 1. Spider Plot - Threat vector analysis
 * 2. Pie Chart - Binary classification
 * 3. Timeline - Attack detection over time
 * 4. Attack Type Distribution - Multi-class breakdown
 * 5. Prediction Confidence - Confidence histogram
 * 6. Binary Classification - Detailed metrics
 */

import { useState } from 'react';
import {
    Upload,
    Activity,
    Shield,
    AlertTriangle,
    TrendingUp,
    Target,
    Radio,
    PieChart,
    BarChart3,
    Clock,
    Brain,
    CheckCircle
} from 'lucide-react';

// Types for API response
interface PredictionResponse {
    sample_id: number;
    binary_prediction: string;
    binary_confidence: number;
    multiclass_prediction: string;
    multiclass_confidence: number;
}

interface ClassDistribution {
    label: string;
    count: number;
    percentage: number;
}

interface PredictionSummary {
    total_samples: number;
    binary_distribution: ClassDistribution[];
    multiclass_distribution: ClassDistribution[];
    attack_percentage: number;
    normal_percentage: number;
}

interface ReportResponse {
    summary: PredictionSummary;
    predictions: PredictionResponse[];
    plots: {
        binary_distribution?: string;
        multiclass_distribution?: string;
        confidence_distribution?: string;
        attack_timeline?: string;
    };
    message: string;
}

const DashboardHome: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [reportData, setReportData] = useState<ReportResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    /**
     * Handle file selection
     */
    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            if (!file.name.endsWith('.csv')) {
                setError('Please upload a CSV file');
                return;
            }
            setSelectedFile(file);
            setError(null);
        }
    };

    /**
     * Upload file to backend for prediction
     */
    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a file first');
            return;
        }

        setIsUploading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch(API_ENDPOINTS.PREDICT, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const data: ReportResponse = await response.json();
            setReportData(data);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to process file');
            console.error('Upload error:', err);
        } finally {
            setIsUploading(false);
        }
    };

    /**
     * Clear results and reset
     */
    const handleClear = () => {
        setSelectedFile(null);
        setReportData(null);
        setError(null);
    };

    return (
        <div className="animate-in fade-in duration-300">
            {/* Header Section */}
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-2xl font-bold text-gray-800">Security Overview</h1>
                    <p className="text-sm text-gray-500 mt-1">Upload network traffic CSV for ML-powered threat detection.</p>
                </div>
                <div className="text-right hidden sm:block">
                    <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Last Updated</span>
                    <p className="text-sm font-mono text-green-900 font-semibold">
                        {new Date().toLocaleString('en-US', {
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit'
                        })}
                    </p>
                </div>
            </div>

            {/* File Upload Section */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 mb-8">
                <div className="flex items-center gap-3 mb-4">
                    <Upload className="text-green-600" size={24} />
                    <h3 className="text-lg font-semibold text-gray-800">Upload Network Traffic Data</h3>
                </div>

                <div className="flex flex-col md:flex-row gap-4 items-start md:items-center">
                    <div className="flex-1">
                        <label className="block">
                            <input
                                type="file"
                                accept=".csv"
                                onChange={handleFileSelect}
                                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100 cursor-pointer"
                            />
                        </label>
                        {selectedFile && (
                            <p className="text-sm text-gray-600 mt-2 flex items-center gap-2">
                                <FileText size={16} />
                                {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
                            </p>
                        )}
                    </div>

                    <div className="flex gap-2">
                        <button
                            onClick={handleUpload}
                            disabled={!selectedFile || isUploading}
                            className="px-6 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                        >
                            {isUploading ? (
                                <>
                                    <Loader2 className="animate-spin" size={18} />
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <BarChart3 size={18} />
                                    Analyze
                                </>
                            )}
                        </button>

                        {reportData && (
                            <button
                                onClick={handleClear}
                                className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300 transition-colors"
                            >
                                Clear
                            </button>
                        )}
                    </div>
                </div>

                {error && (
                    <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                        <p className="text-red-700 text-sm font-medium">{error}</p>
                    </div>
                )}

                {reportData && (
                    <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                        <p className="text-green-700 text-sm font-medium">{reportData.message}</p>
                    </div>
                )}
            </div>

            {/* Results Section - Shows after file analysis */}
            {reportData && (
                <div className="mb-8 space-y-6">
                    {/* KPI Stats from Analysis */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <StatCard
                            title="Total Samples Analyzed"
                            value={reportData.summary.total_samples.toLocaleString()}
                            icon={<Database className="text-blue-600" />}
                            trend="From uploaded CSV"
                        />
                        <StatCard
                            title="Attack Traffic"
                            value={`${reportData.summary.attack_percentage.toFixed(1)}%`}
                            icon={<AlertTriangle className="text-red-500" />}
                            subtext={`${reportData.summary.binary_distribution.find(d => d.label === 'Attack')?.count || 0} threats detected`}
                            alert={reportData.summary.attack_percentage > 5}
                        />
                        <StatCard
                            title="Normal Traffic"
                            value={`${reportData.summary.normal_percentage.toFixed(1)}%`}
                            icon={<CheckCircle className="text-green-600" />}
                            subtext={`${reportData.summary.binary_distribution.find(d => d.label === 'Normal')?.count || 0} normal flows`}
                        />
                        <StatCard
                            title="Attack Types Detected"
                            value={reportData.summary.multiclass_distribution.filter(d => d.label !== 'Normal' && d.count > 0).length}
                            icon={<Activity className="text-orange-500" />}
                            subtext="Unique threat categories"
                            warning={reportData.summary.multiclass_distribution.filter(d => d.label !== 'Normal' && d.count > 0).length > 0}
                        />
                    </div>

                    {/* Visualization Plots from Backend */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {reportData.plots.binary_distribution && (
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                                <h3 className="text-lg font-semibold text-gray-800 mb-4">Binary Classification</h3>
                                <img
                                    src={`data:image/png;base64,${reportData.plots.binary_distribution}`}
                                    alt="Binary Distribution"
                                    className="w-full h-auto"
                                />
                            </div>
                        )}

                        {reportData.plots.multiclass_distribution && (
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                                <h3 className="text-lg font-semibold text-gray-800 mb-4">Attack Type Distribution</h3>
                                <img
                                    src={`data:image/png;base64,${reportData.plots.multiclass_distribution}`}
                                    alt="Multiclass Distribution"
                                    className="w-full h-auto"
                                />
                            </div>
                        )}

                        {reportData.plots.confidence_distribution && (
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                                <h3 className="text-lg font-semibold text-gray-800 mb-4">Prediction Confidence</h3>
                                <img
                                    src={`data:image/png;base64,${reportData.plots.confidence_distribution}`}
                                    alt="Confidence Distribution"
                                    className="w-full h-auto"
                                />
                            </div>
                        )}

                        {reportData.plots.attack_timeline && (
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                                <h3 className="text-lg font-semibold text-gray-800 mb-4">Attack Timeline</h3>
                                <img
                                    src={`data:image/png;base64,${reportData.plots.attack_timeline}`}
                                    alt="Attack Timeline"
                                    className="w-full h-auto"
                                />
                            </div>
                        )}
                    </div>

                    {/* Predictions Table */}
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                        <h3 className="text-lg font-semibold text-gray-800 mb-4">
                            Sample Predictions (Showing {Math.min(reportData.predictions.length, 50)} of {reportData.summary.total_samples})
                        </h3>
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sample ID</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Binary Class</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Attack Type</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {reportData.predictions.slice(0, 50).map((pred) => (
                                        <tr key={pred.sample_id} className="hover:bg-gray-50 transition-colors">
                                            <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                                                #{pred.sample_id}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-sm">
                                                <span className={`px-2 py-1 rounded-full text-xs font-semibold ${pred.binary_prediction === 'Attack'
                                                    ? 'bg-red-100 text-red-800'
                                                    : 'bg-green-100 text-green-800'
                                                    }`}>
                                                    {pred.binary_prediction}
                                                </span>
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                                                {(pred.binary_confidence * 100).toFixed(1)}%
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                                                {pred.multiclass_prediction}
                                            </td>
                                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                                                {(pred.multiclass_confidence * 100).toFixed(1)}%
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Class Distribution Details */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">Binary Distribution</h3>
                            <div className="space-y-3">
                                {reportData.summary.binary_distribution.map((dist) => (
                                    <div key={dist.label} className="flex items-center justify-between">
                                        <span className="text-sm font-medium text-gray-700">{dist.label}</span>
                                        <div className="flex items-center gap-3">
                                            <span className="text-sm text-gray-600">{dist.count} samples</span>
                                            <span className="text-sm font-semibold text-gray-900">{dist.percentage.toFixed(1)}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">Attack Type Distribution</h3>
                            <div className="space-y-3">
                                {reportData.summary.multiclass_distribution.map((dist) => (
                                    <div key={dist.label} className="flex items-center justify-between">
                                        <span className="text-sm font-medium text-gray-700">{dist.label}</span>
                                        <div className="flex items-center gap-3">
                                            <span className="text-sm text-gray-600">{dist.count} samples</span>
                                            <span className="text-sm font-semibold text-gray-900">{dist.percentage.toFixed(1)}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Default Dashboard (shown when no results) */}
            {!reportData && (
                <>
                    {/* KPI Stats Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                        <StatCard
                            title="Total Records Analyzed"
                            value="1,240,592"
                            icon={<Database className="text-blue-600" />}
                            trend="+12% today"
                        />
                        <StatCard
                            title="Malicious Flows"
                            value="2.4%"
                            icon={<AlertTriangle className="text-red-500" />}
                            subtext="32 active threats"
                            alert
                        />
                        <StatCard
                            title="Model Recall Score"
                            value="0.94"
                            icon={<CheckCircle className="text-green-600" />}
                            subtext="High Confidence"
                        />
                        <StatCard
                            title="False Negatives"
                            value="12"
                            icon={<Activity className="text-orange-500" />}
                            subtext="Requires Review"
                            warning
                        />
                    </div>

                    {/* Charts Section */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                        {/* Radar Chart */}
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 lg:col-span-1 flex flex-col h-96">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">Threat Vector Analysis</h3>
                            <ResponsiveContainer width="100%" height="100%">
                                <RadarChart data={MOCK_RADAR_DATA}>
                                    <PolarGrid stroke="#d1d5db" />
                                    <PolarAngleAxis
                                        dataKey="subject"
                                        tick={{ fill: '#6b7280', fontSize: 12 }}
                                    />
                                    <PolarRadiusAxis
                                        angle={90}
                                        domain={[0, 150]}
                                        tick={{ fill: '#6b7280', fontSize: 10 }}
                                    />
                                    <Radar
                                        name="Threat Level"
                                        dataKey="A"
                                        stroke="#15803d"
                                        fill="#22c55e"
                                        fillOpacity={0.6}
                                    />
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Alerts Table */}
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 lg:col-span-2 overflow-hidden h-96 flex flex-col">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent Security Alerts</h3>
                            <div className="overflow-y-auto flex-1">
                                <table className="min-w-full divide-y divide-gray-200">
                                    <thead className="bg-gray-50 sticky top-0">
                                        <tr>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Severity</th>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                        </tr>
                                    </thead>
                                    <tbody className="bg-white divide-y divide-gray-200">
                                        {MOCK_ALERTS.map((alert) => (
                                            <tr key={alert.id} className="hover:bg-gray-50 transition-colors">
                                                <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">{alert.id}</td>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm">
                                                    <span
                                                        className={`px-2 py-1 rounded-full text-xs font-semibold ${alert.severity === 'High' ? 'bg-red-100 text-red-800' :
                                                            alert.severity === 'Medium' ? 'bg-orange-100 text-orange-800' :
                                                                'bg-yellow-100 text-yellow-800'
                                                            }`}
                                                    >
                                                        {alert.severity}
                                                    </span>
                                                </td>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">{alert.type}</td>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">{alert.status}</td>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">{alert.timestamp}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

export default DashboardHome;

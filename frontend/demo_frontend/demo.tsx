import React, { useState, useEffect } from 'react';
import {
    Shield,
    Lock,
    Activity,
    AlertTriangle,
    CheckCircle,
    Search,
    Bell,
    User,
    LogOut,
    FileText,
    Database,
    Radar as RadarIcon,
    Menu,
    X,
    LucideIcon
} from 'lucide-react';
// Removed unused Recharts imports - using backend-generated plots instead

// ============================================================================
// PHASE 1: TYPE DEFINITIONS & API CONTRACTS
// ============================================================================
// BACKEND NOTE: Ensure your API responses match these interfaces.

// Expected JSON from GET /api/dashboard/kpi-stats
interface DashboardStats {
    totalRecords: string;
    maliciousFlows: string;
    recallScore: string;
    falseNegatives: number;
}

// Expected JSON from GET /api/alerts/recent
interface AlertLog {
    id: string;
    severity: 'High' | 'Medium' | 'Low';
    type: string;
    status: string;
    timestamp: string;
}

// UI Component Props (Internal Use)
interface StatCardProps {
    title: string;
    value: string | number;
    icon: React.ReactNode;
    trend?: string;
    subtext?: string;
    alert?: boolean;
    warning?: boolean;
}

interface NavItemProps {
    id: string;
    label: string;
    icon: React.ReactNode;
    activeTab: string;
    onClick: (id: string) => void;
}

// ============================================================================
// PHASE 2: MOCK DATA & BACKEND PLACEHOLDERS
// ============================================================================

// BACKEND NOTE: Replace with fetch to GET /api/alerts/recent
const RECENT_ALERTS: AlertLog[] = [
    { id: '#FL-2093', severity: 'High', type: 'DDoS Attempt', status: 'Blocked', timestamp: '2023-10-27 14:30' },
    { id: '#FL-2094', severity: 'Medium', type: 'SQL Injection', status: 'Flagged', timestamp: '2023-10-27 14:28' },
    { id: '#FL-2095', severity: 'Low', type: 'Port Scan', status: 'Monitored', timestamp: '2023-10-27 14:15' },
    { id: '#FL-2096', severity: 'High', type: 'Malware Payload', status: 'Blocked', timestamp: '2023-10-27 13:50' },
    { id: '#FL-2097', severity: 'Medium', type: 'Brute Force', status: 'Flagged', timestamp: '2023-10-27 13:45' },
];

// ============================================================================
// PHASE 3: AUTHENTICATION VIEW
// ============================================================================

const LoginView = ({ onLogin }: { onLogin: (u: string, p: string) => void }) => {
    const [user, setUser] = useState('');
    const [pass, setPass] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);

        // BACKEND INTEGRATION POINT:
        // try {
        //   const response = await fetch('/api/auth/login', { method: 'POST', body: ... });
        //   const { token } = await response.json();
        //   localStorage.setItem('token', token);
        //   onLogin(user, pass);
        // } catch (err) { ... }

        // Simulating API delay
        setTimeout(() => {
            setIsLoading(false);
            onLogin(user, pass);
        }, 800);
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100 p-4 font-sans">
            <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-md border-t-4 border-green-800">
                <div className="flex flex-col items-center mb-8">
                    <div className="bg-green-50 p-3 rounded-full mb-4">
                        <Shield className="w-8 h-8 text-green-900" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-800">SecuGuard Access</h2>
                    <p className="text-gray-500 text-sm">Authorized Personnel Only</p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                        <div className="relative">
                            <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                            <input
                                type="text"
                                required
                                value={user}
                                onChange={(e) => setUser(e.target.value)}
                                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-800 focus:border-green-800 outline-none transition-colors"
                                placeholder="Enter ID"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                        <div className="relative">
                            <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                            <input
                                type="password"
                                required
                                value={pass}
                                onChange={(e) => setPass(e.target.value)}
                                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-800 focus:border-green-800 outline-none transition-colors"
                                placeholder="••••••••"
                            />
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full bg-green-900 hover:bg-green-800 text-white font-semibold py-3 rounded-lg transition duration-200 shadow-md hover:shadow-lg disabled:opacity-70 flex justify-center items-center"
                    >
                        {isLoading ? (
                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        ) : (
                            "Secure Login"
                        )}
                    </button>
                </form>

                <div className="mt-6 text-center">
                    <span className="text-xs text-gray-400">System v2.4.0 | Encrypted Connection</span>
                </div>
            </div>
        </div>
    );
};

// ============================================================================
// PHASE 4: SHARED UI COMPONENTS
// ============================================================================

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, trend, subtext, alert, warning }) => (
    <div className={`bg-white p-5 rounded-xl shadow-sm border ${alert ? 'border-red-100' : warning ? 'border-orange-100' : 'border-gray-100'} hover:shadow-md transition-shadow`}>
        <div className="flex justify-between items-start">
            <div>
                <p className="text-sm font-medium text-gray-500 mb-1">{title}</p>
                <h3 className="text-2xl font-bold text-gray-800">{value}</h3>
            </div>
            <div className={`p-2 rounded-lg ${alert ? 'bg-red-50' : warning ? 'bg-orange-50' : 'bg-gray-50'}`}>
                {icon}
            </div>
        </div>
        <div className="mt-4 flex items-center text-xs">
            {trend && <span className="text-green-600 font-medium bg-green-50 px-2 py-0.5 rounded mr-2">{trend}</span>}
            {subtext && <span className={`${alert ? 'text-red-500' : warning ? 'text-orange-500' : 'text-gray-400'}`}>{subtext}</span>}
        </div>
    </div>
);

const Navbar = ({
    activeTab,
    setActiveTab,
    username,
    onLogout,
    mobileMenuOpen,
    setMobileMenuOpen
}: any) => {
    const NavItem: React.FC<NavItemProps> = ({ id, label, icon, activeTab, onClick }) => {
        const isActive = activeTab === id;
        return (
            <button
                onClick={() => onClick(id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors w-full md:w-auto
          ${isActive
                        ? 'bg-green-800 text-white shadow-sm'
                        : 'text-green-200 hover:bg-green-800 hover:text-white'
                    }`}
            >
                {icon}
                {label}
            </button>
        );
    };

    return (
        <nav className="bg-green-900 text-white shadow-md z-20 sticky top-0">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">

                    {/* Brand */}
                    <div className="flex items-center flex-shrink-0">
                        <Shield className="w-8 h-8 text-green-300" />
                        <span className="ml-2 text-xl font-bold tracking-wider">NETSEC</span>
                    </div>

                    {/* Desktop Menu */}
                    <div className="hidden md:block">
                        <div className="ml-10 flex items-baseline space-x-4">
                            <NavItem id="dashboard" label="Dashboard" icon={<Activity className="w-4 h-4" />} activeTab={activeTab} onClick={setActiveTab} />
                            <NavItem id="datalogs" label="Data Logs" icon={<Database className="w-4 h-4" />} activeTab={activeTab} onClick={setActiveTab} />
                            <NavItem id="threatintel" label="Threat Intel" icon={<RadarIcon className="w-4 h-4" />} activeTab={activeTab} onClick={setActiveTab} />
                            <NavItem id="reports" label="Reports" icon={<FileText className="w-4 h-4" />} activeTab={activeTab} onClick={setActiveTab} />
                        </div>
                    </div>

                    {/* Right Side Icons & Profile */}
                    <div className="hidden md:flex items-center gap-4">
                        <div className="bg-green-800 rounded-full p-1.5 px-3 flex items-center">
                            <Search className="w-4 h-4 text-green-300" />
                            <input type="text" placeholder="Search..." className="bg-transparent border-none outline-none text-sm ml-2 w-24 text-white placeholder-green-400/70 focus:w-40 transition-all" />
                        </div>
                        <button className="p-1 rounded-full hover:bg-green-800 relative">
                            <Bell className="w-5 h-5 text-green-200" />
                            <span className="absolute top-0 right-0 block h-2 w-2 rounded-full ring-2 ring-green-900 bg-red-500"></span>
                        </button>
                        <div className="flex items-center gap-3 pl-4 border-l border-green-800">
                            <div className="flex flex-col items-end">
                                <span className="text-sm font-medium leading-none">{username}</span>
                                <span className="text-xs text-green-300">Admin</span>
                            </div>
                            <div className="h-8 w-8 rounded-full bg-green-800 flex items-center justify-center text-sm font-bold border border-green-600">
                                {username.charAt(0).toUpperCase()}
                            </div>
                            <button onClick={onLogout} className="ml-2 text-green-300 hover:text-white">
                                <LogOut className="w-5 h-5" />
                            </button>
                        </div>
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="-mr-2 flex md:hidden">
                        <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="bg-green-800 p-2 rounded-md text-green-200 hover:text-white">
                            {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Menu Dropdown */}
            {mobileMenuOpen && (
                <div className="md:hidden bg-green-800 pb-3 pt-2 px-2 space-y-1 shadow-inner">
                    <NavItem id="dashboard" label="Dashboard" icon={<Activity className="w-4 h-4" />} activeTab={activeTab} onClick={(id) => { setActiveTab(id); setMobileMenuOpen(false); }} />
                    <NavItem id="datalogs" label="Data Logs" icon={<Database className="w-4 h-4" />} activeTab={activeTab} onClick={(id) => { setActiveTab(id); setMobileMenuOpen(false); }} />
                    <NavItem id="threatintel" label="Threat Intel" icon={<RadarIcon className="w-4 h-4" />} activeTab={activeTab} onClick={(id) => { setActiveTab(id); setMobileMenuOpen(false); }} />
                    <NavItem id="reports" label="Reports" icon={<FileText className="w-4 h-4" />} activeTab={activeTab} onClick={(id) => { setActiveTab(id); setMobileMenuOpen(false); }} />
                    <div className="border-t border-green-700 mt-4 pt-4 pb-2 flex items-center justify-between px-3">
                        <span className="text-green-100">Signed in as {username}</span>
                        <button onClick={onLogout} className="text-sm bg-green-900 px-3 py-1 rounded">Logout</button>
                    </div>
                </div>
            )}
        </nav>
    );
};

// ============================================================================
// PHASE 5: FEATURE VIEWS (DASHBOARD & MODULES)
// ============================================================================

// Added state management for uploaded data and visualizations
const DashboardHome = () => {
    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [plotData, setPlotData] = useState<any>(null);
    const [summary, setSummary] = useState<any>(null);
    const [error, setError] = useState<string>('');

    // Function to handle file upload and fetch visualizations from backend
    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploadedFile(file);
        setIsUploading(true);
        setError('');

        try {
            // Create form data for file upload
            const formData = new FormData();
            formData.append('file', file);

            console.log('🚀 Starting upload:', file.name);

            // Send file to backend API
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData,
            });

            console.log('📡 Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('❌ Response error:', errorText);
                throw new Error(`Upload failed: ${response.status} - ${errorText}`);
            }

            // Parse response with plots and summary data
            const data = await response.json();
            console.log('✅ Backend response received');
            console.log('📊 Plots data:', data.plots);
            console.log('📈 Summary data:', data.summary);

            // Verify plots exist and are valid base64 images
            if (data.plots) {
                const plotKeys = Object.keys(data.plots);
                console.log(`📦 Found ${plotKeys.length} plots:`, plotKeys);

                plotKeys.forEach(key => {
                    const plotData = data.plots[key];
                    if (plotData && plotData.startsWith('data:image')) {
                        console.log(`✓ Plot ${key}: ${plotData.length} bytes`);
                    } else {
                        console.error(`✗ Plot ${key}: Invalid format`);
                    }
                });
            } else {
                console.error('❌ No plots in response!');
            }

            setPlotData(data.plots);
            setSummary(data.summary);
            console.log('🎉 State updated with plots and summary');

            // Test if images can render by checking format
            if (data.plots && data.plots.binary_distribution) {
                const testImg = new Image();
                testImg.onload = () => console.log('✅ Test: Image format is valid and can be rendered');
                testImg.onerror = () => console.error('❌ Test: Image format CANNOT be rendered by browser');
                testImg.src = data.plots.binary_distribution.substring(0, 1000); // Test with first 1000 chars
            }

        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Failed to upload file';
            setError(errorMsg);
            console.error('❌ Upload error:', err);
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="animate-in fade-in duration-300">
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-2xl font-bold text-gray-800">Security Overview</h1>
                    <p className="text-sm text-gray-500 mt-1">Real-time threat monitoring and model performance.</p>
                </div>
                <div className="text-right hidden sm:block">
                    <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Last Updated Log</span>
                    <p className="text-sm font-mono text-green-900 font-semibold">{new Date().toLocaleString()}</p>
                </div>
            </div>

            {/* File Upload Section - Enhanced Design */}
            <div className="bg-gradient-to-r from-green-50 to-blue-50 p-8 rounded-2xl shadow-lg border-2 border-green-200 mb-8">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-3 bg-green-800 rounded-xl">
                        <Database className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-gray-800">Upload Network Traffic Data</h3>
                        <p className="text-sm text-gray-600">Analyze CSV files with ML-powered threat detection</p>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <label className="flex-1 cursor-pointer group">
                        <div className="border-3 border-dashed border-green-300 rounded-xl p-8 hover:border-green-600 hover:bg-white transition-all duration-300 text-center group-hover:shadow-lg">
                            <div className="flex flex-col items-center gap-3">
                                <div className="p-4 bg-green-100 rounded-full group-hover:bg-green-200 transition-colors">
                                    <Database className="w-10 h-10 text-green-700" />
                                </div>
                                <div>
                                    <p className="text-base font-semibold text-gray-700 mb-1">
                                        {uploadedFile ? (
                                            <span className="text-green-800">📄 {uploadedFile.name}</span>
                                        ) : (
                                            'Click or drag to upload CSV file'
                                        )}
                                    </p>
                                    <p className="text-xs text-gray-500">
                                        Supported format: .csv | Max size: 50MB
                                    </p>
                                </div>
                            </div>
                        </div>
                        <input
                            type="file"
                            accept=".csv"
                            onChange={handleFileUpload}
                            className="hidden"
                        />
                    </label>
                    {isUploading && (
                        <div className="flex flex-col items-center gap-2 text-green-800 bg-white px-6 py-4 rounded-xl shadow-md">
                            <div className="w-8 h-8 border-4 border-green-800 border-t-transparent rounded-full animate-spin"></div>
                            <span className="text-sm font-semibold">Analyzing...</span>
                        </div>
                    )}
                </div>
                {error && (
                    <div className="mt-4 p-4 bg-red-50 border-2 border-red-200 rounded-xl flex items-center gap-3">
                        <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0" />
                        <span className="text-sm text-red-700 font-medium">{error}</span>
                    </div>
                )}
                {/* Status indicator showing loaded data */}
                {plotData && (
                    <div className="mt-4 p-3 bg-green-50 border-2 border-green-200 rounded-xl flex items-center gap-3">
                        <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                        <span className="text-sm text-green-700 font-medium">
                            ✓ Analysis complete! Loaded {Object.keys(plotData).length} visualizations
                        </span>
                    </div>
                )}
            </div>

            {/* KPI Stats Grid - Display data from uploaded file */}
            {summary && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <StatCard
                        title="Total Records Analyzed"
                        value={summary.total_samples?.toLocaleString() || '0'}
                        icon={<Database className="text-blue-600" />}
                    />
                    <StatCard
                        title="Malicious Flows"
                        value={`${summary.attack_percentage?.toFixed(1) || 0}%`}
                        icon={<AlertTriangle className="text-red-500" />}
                        subtext={`${summary.binary_distribution?.find((d: any) => d.label === 'Attack')?.count || 0} attacks detected`}
                        alert
                    />
                    <StatCard
                        title="Normal Traffic"
                        value={summary.binary_distribution?.find((d: any) => d.label === 'Normal')?.count?.toLocaleString() || '0'}
                        icon={<CheckCircle className="text-green-600" />}
                        subtext="Benign flows"
                    />
                    <StatCard
                        title="Normal Traffic %"
                        value={`${summary.normal_percentage?.toFixed(1) || 0}%`}
                        icon={<Activity className="text-green-500" />}
                        subtext="Safe network activity"
                    />
                </div>
            )}

            {/* Visualization Plots - Display actual plots from backend */}
            {plotData && Object.keys(plotData).length > 0 ? (
                <div>
                    <div className="mb-6">
                        <h2 className="text-xl font-bold text-gray-800 mb-2">Threat Analysis Visualizations</h2>
                        <p className="text-sm text-gray-600">Real-time visual insights from uploaded network traffic data</p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                        {/* Binary Distribution Plot - PIE CHART */}
                        {plotData.binary_distribution && (
                            <div className="bg-gradient-to-br from-white to-gray-50 p-6 rounded-2xl shadow-lg border-2 border-gray-100 hover:shadow-xl transition-all duration-300">
                                <div className="flex items-center gap-3 mb-4">
                                    <div className="p-2 bg-blue-100 rounded-lg">
                                        <Shield className="w-5 h-5 text-blue-600" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold text-gray-800">Binary Classification</h3>
                                        <p className="text-xs text-gray-500">Normal vs Attack Distribution</p>
                                    </div>
                                </div>
                                {plotData.binary_distribution.startsWith('data:image') ? (
                                    <div className="bg-white rounded-xl p-4 shadow-inner">
                                        <img
                                            src={plotData.binary_distribution}
                                            alt="Binary Classification Pie Chart"
                                            className="w-full h-auto rounded-lg"
                                            onLoad={(e) => {
                                                console.log('✅ Binary distribution image loaded successfully');
                                            }}
                                            onError={(e) => {
                                                console.error('❌ Failed to load binary distribution image');
                                                console.error('Image data length:', plotData.binary_distribution.length);
                                                console.error('First 100 chars:', plotData.binary_distribution.substring(0, 100));
                                                e.currentTarget.src = '';
                                                e.currentTarget.alt = '❌ Failed to load image';
                                            }}
                                        />
                                    </div>
                                ) : (
                                    <div className="text-red-500 text-sm bg-red-50 p-4 rounded-lg">
                                        ⚠ Invalid image format. Data starts with: {plotData.binary_distribution.substring(0, 50)}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Attack Type Distribution Plot - BAR CHART */}
                        {plotData.multiclass_distribution && (
                            <div className="bg-gradient-to-br from-white to-gray-50 p-6 rounded-2xl shadow-lg border-2 border-gray-100 hover:shadow-xl transition-all duration-300">
                                <div className="flex items-center gap-3 mb-4">
                                    <div className="p-2 bg-red-100 rounded-lg">
                                        <AlertTriangle className="w-5 h-5 text-red-600" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold text-gray-800">Attack Type Distribution</h3>
                                        <p className="text-xs text-gray-500">Multi-Class Attack Categories</p>
                                    </div>
                                </div>
                                {plotData.multiclass_distribution.startsWith('data:image') ? (
                                    <div className="bg-white rounded-xl p-4 shadow-inner">
                                        <img
                                            src={plotData.multiclass_distribution}
                                            alt="Attack Type Distribution Bar Chart"
                                            className="w-full h-auto rounded-lg"
                                            onLoad={() => console.log('✅ Multiclass distribution image loaded')}
                                            onError={(e) => {
                                                console.error('❌ Failed to load multiclass distribution image');
                                                e.currentTarget.src = '';
                                                e.currentTarget.alt = '❌ Failed to load image';
                                            }}
                                        />
                                    </div>
                                ) : (
                                    <div className="text-red-500 text-sm bg-red-50 p-4 rounded-lg">⚠ Invalid image format</div>
                                )}
                            </div>
                        )}

                        {/* Prediction Confidence Plot - HISTOGRAM */}
                        {plotData.confidence_distribution && (
                            <div className="bg-gradient-to-br from-white to-gray-50 p-6 rounded-2xl shadow-lg border-2 border-gray-100 hover:shadow-xl transition-all duration-300">
                                <div className="flex items-center gap-3 mb-4">
                                    <div className="p-2 bg-purple-100 rounded-lg">
                                        <Activity className="w-5 h-5 text-purple-600" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold text-gray-800">Prediction Confidence</h3>
                                        <p className="text-xs text-gray-500">Model Confidence Score Distribution</p>
                                    </div>
                                </div>
                                {plotData.confidence_distribution.startsWith('data:image') ? (
                                    <div className="bg-white rounded-xl p-4 shadow-inner">
                                        <img
                                            src={plotData.confidence_distribution}
                                            alt="Prediction Confidence Histogram"
                                            className="w-full h-auto rounded-lg"
                                            onLoad={() => console.log('✅ Confidence distribution image loaded')}
                                            onError={(e) => {
                                                console.error('❌ Failed to load confidence distribution image');
                                                e.currentTarget.src = '';
                                                e.currentTarget.alt = '❌ Failed to load image';
                                            }}
                                        />
                                    </div>
                                ) : (
                                    <div className="text-red-500 text-sm bg-red-50 p-4 rounded-lg">⚠ Invalid image format</div>
                                )}
                            </div>
                        )}

                        {/* Attack Timeline Plot - LINE CHART */}
                        {plotData.attack_timeline && (
                            <div className="bg-gradient-to-br from-white to-gray-50 p-6 rounded-2xl shadow-lg border-2 border-gray-100 hover:shadow-xl transition-all duration-300">
                                <div className="flex items-center gap-3 mb-4">
                                    <div className="p-2 bg-orange-100 rounded-lg">
                                        <Activity className="w-5 h-5 text-orange-600" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold text-gray-800">Attack Timeline</h3>
                                        <p className="text-xs text-gray-500">Temporal Attack Detection Pattern</p>
                                    </div>
                                </div>
                                {plotData.attack_timeline.startsWith('data:image') ? (
                                    <div className="bg-white rounded-xl p-4 shadow-inner">
                                        <img
                                            src={plotData.attack_timeline}
                                            alt="Attack Detection Timeline"
                                            className="w-full h-auto rounded-lg"
                                            onLoad={() => console.log('✅ Attack timeline image loaded')}
                                            onError={(e) => {
                                                console.error('❌ Failed to load attack timeline image');
                                                e.currentTarget.src = '';
                                                e.currentTarget.alt = '❌ Failed to load image';
                                            }}
                                        />
                                    </div>
                                ) : (
                                    <div className="text-red-500 text-sm bg-red-50 p-4 rounded-lg">⚠ Invalid image format</div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            ) : plotData && Object.keys(plotData).length === 0 ? (
                <div className="bg-yellow-50 p-6 rounded-xl shadow-sm border border-yellow-200 text-center">
                    <AlertTriangle className="w-12 h-12 text-yellow-500 mx-auto mb-3" />
                    <h3 className="text-lg font-semibold text-gray-700 mb-2">No Plots Generated</h3>
                    <p className="text-gray-600">The backend returned data but no visualization plots were generated.</p>
                </div>
            ) : null}

            {/* Show enhanced message when no data uploaded */}
            {!plotData && !isUploading && (
                <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-12 rounded-2xl shadow-lg border-2 border-gray-200 text-center">
                    <div className="flex flex-col items-center gap-6 max-w-2xl mx-auto">
                        <div className="relative">
                            <div className="absolute inset-0 bg-green-200 rounded-full blur-2xl opacity-30"></div>
                            <div className="relative p-6 bg-white rounded-full shadow-lg">
                                <Database className="w-20 h-20 text-green-700" />
                            </div>
                        </div>
                        <div>
                            <h3 className="text-2xl font-bold text-gray-800 mb-3">Ready to Analyze Network Traffic</h3>
                            <p className="text-gray-600 mb-6">Upload a CSV file to get started with ML-powered threat detection and visualization</p>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
                                <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                    <div className="flex items-center gap-2 mb-2">
                                        <CheckCircle className="w-5 h-5 text-green-600" />
                                        <span className="font-semibold text-gray-800">Binary Classification</span>
                                    </div>
                                    <p className="text-xs text-gray-600">Normal vs Attack pie chart</p>
                                </div>
                                <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                    <div className="flex items-center gap-2 mb-2">
                                        <CheckCircle className="w-5 h-5 text-green-600" />
                                        <span className="font-semibold text-gray-800">Attack Types</span>
                                    </div>
                                    <p className="text-xs text-gray-600">Multi-class distribution bar chart</p>
                                </div>
                                <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                    <div className="flex items-center gap-2 mb-2">
                                        <CheckCircle className="w-5 h-5 text-green-600" />
                                        <span className="font-semibold text-gray-800">Confidence & Timeline</span>
                                    </div>
                                    <p className="text-xs text-gray-600">Model confidence and temporal analysis</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Placeholder Views for Other Tabs
const DataLogsView = () => (
    <div className="animate-in fade-in duration-300 p-8 bg-white rounded-xl shadow-sm border border-gray-200 min-h-[400px] flex flex-col items-center justify-center text-center">
        <div className="bg-green-50 p-4 rounded-full mb-4"><Database className="w-12 h-12 text-green-900" /></div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Data Logs Repository</h2>
        <p className="text-gray-500 max-w-md">Access historical traffic data. Backend integration required to fetch paginated logs.</p>
        <button className="mt-6 px-6 py-2 bg-green-900 text-white rounded-lg hover:bg-green-800 transition-colors">Initiate Query</button>
    </div>
);

const ThreatIntelView = () => (
    <div className="animate-in fade-in duration-300 p-8 bg-white rounded-xl shadow-sm border border-gray-200 min-h-[400px] flex flex-col items-center justify-center text-center">
        <div className="bg-green-50 p-4 rounded-full mb-4"><RadarIcon className="w-12 h-12 text-green-900" /></div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Global Threat Intelligence</h2>
        <p className="text-gray-500 max-w-md">Real-time feeds from global security partners.</p>
    </div>
);

const ReportsView = () => {
    const [isGenerating, setIsGenerating] = useState(false);
    const [reports, setReports] = useState<any[]>([]);
    const [error, setError] = useState('');
    const [latestReport, setLatestReport] = useState<string | null>(null);

    // Fetch available reports on mount
    useEffect(() => {
        fetchReports();
    }, []);

    const fetchReports = async () => {
        try {
            const response = await fetch('http://localhost:8000/reports');
            const data = await response.json();
            setReports(data.reports || []);
        } catch (err) {
            console.error('Failed to fetch reports:', err);
        }
    };

    const generateReport = async () => {
        setIsGenerating(true);
        setError('');

        try {
            console.log('🚀 Generating ML Model Evaluation Report...');

            const response = await fetch('http://localhost:8000/generate-report', {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`Report generation failed: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('✅ Report generated:', data);

            setLatestReport(data.report_name);

            // Refresh reports list
            await fetchReports();

        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Failed to generate report';
            setError(errorMsg);
            console.error('❌ Report generation error:', err);
        } finally {
            setIsGenerating(false);
        }
    };

    const openReport = (reportName: string) => {
        // Open report in new window - points to local reports folder
        const reportUrl = `http://localhost:8000/reports/${reportName}`;
        window.open(reportUrl, '_blank');
    };

    return (
        <div className="animate-in fade-in duration-300">
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-gray-800 mb-2">ML Model Reports</h1>
                <p className="text-sm text-gray-600">Generate and view comprehensive evaluation reports</p>
            </div>

            {/* Generate Report Section */}
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-8 rounded-2xl shadow-lg border-2 border-purple-200 mb-8">
                <div className="flex items-center justify-between gap-6">
                    <div className="flex items-center gap-4">
                        <div className="p-4 bg-purple-600 rounded-xl shadow-lg">
                            <FileText className="w-8 h-8 text-white" />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold text-gray-800 mb-1">Generate New Report</h3>
                            <p className="text-sm text-gray-600">
                                Comprehensive ML model evaluation with metrics, confusion matrices, and ROC curves
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={generateReport}
                        disabled={isGenerating}
                        className={`px-8 py-4 rounded-xl font-semibold shadow-lg transition-all duration-300 flex items-center gap-3 ${isGenerating
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-purple-600 hover:bg-purple-700 text-white hover:shadow-xl'
                            }`}
                    >
                        {isGenerating ? (
                            <>
                                <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin"></div>
                                <span>Generating...</span>
                            </>
                        ) : (
                            <>
                                <FileText className="w-5 h-5" />
                                <span>Generate Report</span>
                            </>
                        )}
                    </button>
                </div>

                {error && (
                    <div className="mt-4 p-4 bg-red-50 border-2 border-red-200 rounded-xl flex items-center gap-3">
                        <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0" />
                        <span className="text-sm text-red-700 font-medium">{error}</span>
                    </div>
                )}

                {latestReport && (
                    <div className="mt-4 p-4 bg-green-50 border-2 border-green-200 rounded-xl flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                            <span className="text-sm text-green-700 font-medium">
                                ✓ Report generated: {latestReport}
                            </span>
                        </div>
                        <button
                            onClick={() => openReport(latestReport)}
                            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
                        >
                            View Report
                        </button>
                    </div>
                )}
            </div>

            {/* Report Features */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="bg-white p-6 rounded-xl shadow-sm border-2 border-gray-200 hover:shadow-lg transition-shadow">
                    <div className="p-3 bg-blue-100 rounded-lg w-fit mb-3">
                        <Activity className="w-6 h-6 text-blue-600" />
                    </div>
                    <h4 className="font-bold text-gray-800 mb-2">Performance Metrics</h4>
                    <p className="text-sm text-gray-600">Accuracy, Precision, Recall, F1-Score for all models</p>
                </div>
                <div className="bg-white p-6 rounded-xl shadow-sm border-2 border-gray-200 hover:shadow-lg transition-shadow">
                    <div className="p-3 bg-purple-100 rounded-lg w-fit mb-3">
                        <Database className="w-6 h-6 text-purple-600" />
                    </div>
                    <h4 className="font-bold text-gray-800 mb-2">Confusion Matrices</h4>
                    <p className="text-sm text-gray-600">Detailed prediction breakdowns with heatmaps</p>
                </div>
                <div className="bg-white p-6 rounded-xl shadow-sm border-2 border-gray-200 hover:shadow-lg transition-shadow">
                    <div className="p-3 bg-green-100 rounded-lg w-fit mb-3">
                        <Activity className="w-6 h-6 text-green-600" />
                    </div>
                    <h4 className="font-bold text-gray-800 mb-2">ROC Curves</h4>
                    <p className="text-sm text-gray-600">AUC scores and classification thresholds</p>
                </div>
                <div className="bg-white p-6 rounded-xl shadow-sm border-2 border-gray-200 hover:shadow-lg transition-shadow">
                    <div className="p-3 bg-orange-100 rounded-lg w-fit mb-3">
                        <FileText className="w-6 h-6 text-orange-600" />
                    </div>
                    <h4 className="font-bold text-gray-800 mb-2">Feature Importance</h4>
                    <p className="text-sm text-gray-600">Top features contributing to predictions</p>
                </div>
            </div>

            {/* Available Reports */}
            <div className="bg-white p-6 rounded-2xl shadow-lg border-2 border-gray-200">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-3">
                    <FileText className="w-6 h-6 text-purple-600" />
                    Available Reports
                </h3>

                {reports.length > 0 ? (
                    <div className="space-y-3">
                        {reports.map((report, index) => (
                            <div
                                key={index}
                                className="flex items-center justify-between p-4 bg-gray-50 rounded-xl border border-gray-200 hover:bg-gray-100 transition-colors"
                            >
                                <div className="flex items-center gap-4">
                                    <div className="p-2 bg-purple-100 rounded-lg">
                                        <FileText className="w-5 h-5 text-purple-600" />
                                    </div>
                                    <div>
                                        <h4 className="font-semibold text-gray-800 text-sm">{report.name}</h4>
                                        <p className="text-xs text-gray-500">Generated: {report.date} • Size: {report.size}</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => openReport(report.name)}
                                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium flex items-center gap-2"
                                >
                                    <FileText className="w-4 h-4" />
                                    Open
                                </button>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-12">
                        <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                        <p className="text-gray-500">No reports available. Generate your first report above!</p>
                    </div>
                )}
            </div>
        </div>
    );
};

// ============================================================================
// PHASE 6: MAIN APP CONTROLLER
// ============================================================================

const CyberDashboard: React.FC = () => {
    const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
    const [username, setUsername] = useState<string>('');
    const [activeTab, setActiveTab] = useState<string>('dashboard');
    const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

    // Check for existing session on mount (e.g., JWT in localStorage)
    useEffect(() => {
        // const token = localStorage.getItem('token');
        // if (token) setIsLoggedIn(true);
    }, []);

    const handleLogin = (u: string, p: string) => {
        if (u && p) {
            setUsername(u);
            setIsLoggedIn(true);
        }
    };

    const handleLogout = () => {
        // localStorage.removeItem('token');
        setIsLoggedIn(false);
        setUsername('');
    };

    // View Router
    const renderContent = () => {
        switch (activeTab) {
            case 'dashboard': return <DashboardHome />;
            case 'datalogs': return <DataLogsView />;
            case 'threatintel': return <ThreatIntelView />;
            case 'reports': return <ReportsView />;
            default: return <DashboardHome />;
        }
    };

    if (!isLoggedIn) {
        return <LoginView onLogin={handleLogin} />;
    }

    return (
        <div className="flex flex-col h-screen bg-gray-50 text-gray-800 font-sans overflow-hidden">
            <Navbar
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                username={username}
                onLogout={handleLogout}
                mobileMenuOpen={mobileMenuOpen}
                setMobileMenuOpen={setMobileMenuOpen}
            />
            <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
                <div className="max-w-7xl mx-auto">
                    {renderContent()}
                </div>
            </main>
        </div>
    );
};

export default CyberDashboard;
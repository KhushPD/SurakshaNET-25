/**
 * Navbar Component
 * ================
 * Main navigation bar with logo, navigation items, user info, and mobile menu.
 * Handles navigation between different views and user logout.
 */

import {
    Activity,
    Database,
    Radar as RadarIcon,
    FileText,
    User,
    Menu,
    X,
    Moon,
    Sun
} from 'lucide-react';
import type { NavItemProps } from '../../types';

interface NavbarProps {
    activeTab: string;
    setActiveTab: (tab: string) => void;
    username: string;
    onLogout: () => void;
    mobileMenuOpen: boolean;
    setMobileMenuOpen: (open: boolean) => void;
    isDarkMode: boolean;
    toggleTheme: () => void;
}

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

const Navbar: React.FC<NavbarProps> = ({
    activeTab,
    setActiveTab,
    username,
    onLogout,
    mobileMenuOpen,
    setMobileMenuOpen,
    isDarkMode,
    toggleTheme
}) => {
    return (
        <nav className={`shadow-md z-20 sticky top-0 transition-colors duration-300 ${isDarkMode ? 'bg-green-900 text-white' : 'bg-green-700 text-white'
            }`}>
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">

                    {/* Logo Section */}
                    <div className="flex items-center flex-shrink-0">
                        <img src="/logo.png" alt="SurakshaNET Logo" className="w-8 h-8" />
                        <span className="ml-2 text-xl font-bold">SurakshaNET</span>
                    </div>

                    {/* Desktop Navigation */}
                    <div className="hidden md:block">
                        <div className="ml-10 flex items-baseline space-x-2">
                            <NavItem id="dashboard" label="Dashboard" icon={<Activity className="w-4 h-4" />} activeTab={activeTab} onClick={setActiveTab} />
                            <NavItem id="datalogs" label="Data Logs" icon={<Database className="w-4 h-4" />} activeTab={activeTab} onClick={setActiveTab} />
                            <NavItem id="threatintel" label="Threat Intel" icon={<RadarIcon className="w-4 h-4" />} activeTab={activeTab} onClick={setActiveTab} />
                            <NavItem id="reports" label="Reports" icon={<FileText className="w-4 h-4" />} activeTab={activeTab} onClick={setActiveTab} />
                        </div>
                    </div>

                    {/* Desktop User Menu */}
                    <div className="hidden md:flex items-center gap-4">
                        <button
                            onClick={toggleTheme}
                            className="p-2 bg-green-800 hover:bg-green-700 rounded-md transition-colors"
                            title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                        >
                            {isDarkMode ? <Sun className="w-4 h-4 text-yellow-300" /> : <Moon className="w-4 h-4 text-blue-300" />}
                        </button>
                        <div className="flex items-center gap-2 bg-green-800 px-3 py-1.5 rounded-md">
                            <User className="w-4 h-4 text-green-300" />
                            <span className="text-sm text-green-100">{username}</span>
                        </div>
                        <button
                            onClick={onLogout}
                            className="text-sm bg-green-800 hover:bg-green-700 px-4 py-1.5 rounded-md transition-colors"
                        >
                            Logout
                        </button>
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="-mr-2 flex md:hidden">
                        <button
                            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                            className="inline-flex items-center justify-center p-2 rounded-md text-green-200 hover:text-white hover:bg-green-800 transition-colors"
                        >
                            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Menu Dropdown */}
            {mobileMenuOpen && (
                <div className="md:hidden bg-green-800 pb-3 pt-2 px-2 space-y-1 shadow-inner">
                    <NavItem
                        id="dashboard"
                        label="Dashboard"
                        icon={<Activity className="w-4 h-4" />}
                        activeTab={activeTab}
                        onClick={(id) => { setActiveTab(id); setMobileMenuOpen(false); }}
                    />
                    <NavItem
                        id="datalogs"
                        label="Data Logs"
                        icon={<Database className="w-4 h-4" />}
                        activeTab={activeTab}
                        onClick={(id) => { setActiveTab(id); setMobileMenuOpen(false); }}
                    />
                    <NavItem
                        id="threatintel"
                        label="Threat Intel"
                        icon={<RadarIcon className="w-4 h-4" />}
                        activeTab={activeTab}
                        onClick={(id) => { setActiveTab(id); setMobileMenuOpen(false); }}
                    />
                    <NavItem
                        id="reports"
                        label="Reports"
                        icon={<FileText className="w-4 h-4" />}
                        activeTab={activeTab}
                        onClick={(id) => { setActiveTab(id); setMobileMenuOpen(false); }}
                    />
                    <div className="border-t border-green-700 mt-4 pt-4 pb-2 px-3 space-y-3">
                        <button
                            onClick={toggleTheme}
                            className="w-full flex items-center justify-between bg-green-900 px-3 py-2 rounded hover:bg-green-800 transition-colors"
                        >
                            <span className="text-sm text-green-100">
                                {isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                            </span>
                            {isDarkMode ? <Sun className="w-4 h-4 text-yellow-300" /> : <Moon className="w-4 h-4 text-blue-300" />}
                        </button>
                        <div className="flex items-center justify-between">
                            <span className="text-green-100">Signed in as {username}</span>
                            <button
                                onClick={onLogout}
                                className="text-sm bg-green-900 px-3 py-1 rounded hover:bg-green-800 transition-colors"
                            >
                                Logout
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </nav>
    );
};

export default Navbar;

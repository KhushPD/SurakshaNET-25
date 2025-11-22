/**
 * Main Application Component
 * ==========================
 * Root component that manages application state and routing.
 * Handles authentication, navigation, and view rendering.
 */

import './App.css';
import { useState, useEffect } from 'react';
import LandingPage from './components/views/LandingPage';
import LoginView from './components/auth/LoginView';
import Navbar from './components/layout/Navbar';
import DashboardHome from './components/views/DashboardHomeNew';
import DataLogsView from './components/views/DataLogsView';
import ThreatIntelView from './components/views/ThreatIntelView';
import ReportsView from './components/views/ReportsView';

function App() {
  // Landing page state
  const [showLanding, setShowLanding] = useState<boolean>(true);

  // Authentication state
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const [username, setUsername] = useState<string>('');

  // Navigation state
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  // Theme state
  const [isDarkMode, setIsDarkMode] = useState<boolean>(() => {
    const saved = localStorage.getItem('theme');
    return saved ? saved === 'dark' : true; // Default to dark
  });

  // Check for existing session on mount
  useEffect(() => {
    // TODO: Check localStorage for token and validate with backend
    // const token = localStorage.getItem('token');
    // if (token) {
    //   validateToken(token).then(isValid => {
    //     if (isValid) setIsLoggedIn(true);
    //   });
    // }
  }, []);

  // Save theme preference
  useEffect(() => {
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  // Toggle theme
  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Handle "Get Started" button click
  const handleGetStarted = () => {
    setShowLanding(false);
  };

  // Handle user login
  const handleLogin = (user: string, pass: string) => {
    if (user && pass) {
      setUsername(user);
      setIsLoggedIn(true);
      // TODO: Save token to localStorage
      // localStorage.setItem('token', token);
    }
  };

  // Handle user logout
  const handleLogout = () => {
    // TODO: Clear token from localStorage
    // localStorage.removeItem('token');
    setIsLoggedIn(false);
    setUsername('');
    setActiveTab('dashboard');
    setShowLanding(true); // Return to landing page
  };

  // Render appropriate view based on active tab
  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardHome isDarkMode={isDarkMode} />;
      case 'datalogs':
        return <DataLogsView isDarkMode={isDarkMode} />;
      case 'threatintel':
        return <ThreatIntelView isDarkMode={isDarkMode} />;
      case 'reports':
        return <ReportsView isDarkMode={isDarkMode} />;
      default:
        return <DashboardHome isDarkMode={isDarkMode} />;
    }
  };

  // Show landing page first
  if (showLanding && !isLoggedIn) {
    return <LandingPage onGetStarted={handleGetStarted} isDarkMode={isDarkMode} toggleTheme={toggleTheme} />;
  }

  // Show login page if not authenticated
  if (!isLoggedIn) {
    return <LoginView onLogin={handleLogin} isDarkMode={isDarkMode} toggleTheme={toggleTheme} />;
  }

  // Show main dashboard if authenticated
  return (
    <div className={`flex flex-col h-screen font-sans overflow-hidden transition-colors duration-300 ${isDarkMode
      ? 'bg-gradient-to-br from-black via-gray-900 to-green-950 text-gray-200'
      : 'bg-gradient-to-br from-gray-50 via-white to-green-50 text-gray-800'
      }`}>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        username={username}
        onLogout={handleLogout}
        mobileMenuOpen={mobileMenuOpen}
        setMobileMenuOpen={setMobileMenuOpen}
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
      />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

export default App;
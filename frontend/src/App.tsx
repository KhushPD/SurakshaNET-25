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

  // Chatbot context state
  const [chatbotContext, setChatbotContext] = useState<any>(null);
  const [chatbotMessage, setChatbotMessage] = useState<string>('');

  // IP Block notification state
  const [blockNotifications, setBlockNotifications] = useState<any[]>([]);
  const [lastCheckTime, setLastCheckTime] = useState<string>(new Date().toISOString());

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

  // Poll for new IP blocks
  useEffect(() => {
    if (!isLoggedIn) return;

    const checkForBlocks = async () => {
      try {
        const response = await fetch(`http://localhost:8000/recent-blocks?since=${lastCheckTime}`);
        const data = await response.json();
        
        if (data.status === 'success' && data.blocks.length > 0) {
          setBlockNotifications(prev => [...prev, ...data.blocks]);
          setLastCheckTime(new Date().toISOString());
          
          // Auto-dismiss after 8 seconds
          setTimeout(() => {
            setBlockNotifications(prev => prev.filter(n => !data.blocks.some(b => b.id === n.id)));
          }, 8000);
        }
      } catch (error) {
        console.error('Error fetching blocks:', error);
      }
    };

    const interval = setInterval(checkForBlocks, 3000); // Check every 3 seconds
    return () => clearInterval(interval);
  }, [isLoggedIn, lastCheckTime]);

  const dismissNotification = (id: string) => {
    setBlockNotifications(prev => prev.filter(n => n.id !== id));
  };

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

  // Handle Ask AI button from DataLogs
  const handleAskAI = (log: any) => {
    setChatbotMessage('Analyze this log entry');
    setChatbotContext({ log_data: log });
    setActiveTab('threatintel');
  };

  // Clear context when switching away from threatintel
  useEffect(() => {
    if (activeTab !== 'threatintel') {
      setChatbotMessage('');
      setChatbotContext(null);
    }
  }, [activeTab]);

  // Render appropriate view based on active tab
  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardHome isDarkMode={isDarkMode} />;
      case 'datalogs':
        return <DataLogsView isDarkMode={isDarkMode} onAskAI={handleAskAI} />;
      case 'threatintel':
        return <ThreatIntelView 
          isDarkMode={isDarkMode} 
          initialMessage={chatbotMessage}
          initialContext={chatbotContext}
        />;
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

      {/* IP Block Notifications */}
      <div className="fixed bottom-4 right-4 z-50 space-y-2 max-w-sm">
        {blockNotifications.map((block) => (
          <div
            key={block.id}
            className={`p-4 rounded-lg shadow-2xl border backdrop-blur-md animate-slide-in ${
              isDarkMode
                ? 'bg-red-900/90 border-red-700 text-red-100'
                : 'bg-red-50/90 border-red-300 text-red-900'
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-lg">🚫</span>
                  <h4 className="font-semibold text-sm">IP Blocked</h4>
                </div>
                <p className="text-xs font-mono mb-1">{block.ip}</p>
                <p className="text-xs opacity-90">{block.reason}</p>
                <p className="text-xs opacity-75 mt-1">Confidence: {block.confidence}</p>
              </div>
              <button
                onClick={() => dismissNotification(block.id)}
                className={`text-lg leading-none hover:opacity-70 transition-opacity ${
                  isDarkMode ? 'text-red-300' : 'text-red-700'
                }`}
              >
                ×
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
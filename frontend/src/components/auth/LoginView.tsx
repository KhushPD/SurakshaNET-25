/**
 * Login View Component
 * ====================
 * Authentication page with username/password form.
 * Styled to match LandingPage with FloatingLines background and sequential animations.
 */

import { useState, useMemo } from 'react';
import { Shield, Loader2, Moon, Sun } from 'lucide-react';
import FloatingLines from '../common/FloatingLines';

interface LoginViewProps {
    onLogin: (username: string, password: string) => void;
    isDarkMode: boolean;
    toggleTheme: () => void;
}

const LoginView: React.FC<LoginViewProps> = ({ onLogin, isDarkMode, toggleTheme }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);

        // Simulating API delay
        setTimeout(() => {
            setIsLoading(false);
            onLogin(username, password);
        }, 800);
    };

    // Memoize ALL FloatingLines props to prevent re-renders when typing
    const floatingLinesGradient = useMemo(
        () => isDarkMode
            ? ['#10b981', '#059669', '#047857', '#065f46']
            : ['#34d399', '#10b981', '#059669', '#047857'],
        [isDarkMode]
    );

    const enabledWaves = useMemo<Array<'top' | 'middle' | 'bottom'>>(() => ['middle', 'bottom'], []);
    const lineCount = useMemo(() => [8, 6], []);
    const lineDistance = useMemo(() => [3, 5], []);

    return (
        <div className={`min-h-screen relative overflow-hidden transition-colors duration-300 ${isDarkMode
            ? 'bg-black'
            : 'bg-white'
            }`}>
            {/* Animated Background */}
            <div className="absolute inset-0 z-0">
                <FloatingLines
                    linesGradient={floatingLinesGradient}
                    enabledWaves={enabledWaves}
                    lineCount={lineCount}
                    lineDistance={lineDistance}
                    animationSpeed={0.8}
                    interactive={true}
                    bendRadius={4.0}
                    bendStrength={-0.3}
                    mouseDamping={0.08}
                    parallax={true}
                    parallaxStrength={0.15}
                    mixBlendMode="screen"
                />
            </div>

            {/* Theme Toggle */}
            <div className="absolute top-6 right-6 z-20">
                <button
                    onClick={toggleTheme}
                    className={`p-3 rounded-full transition-all duration-300 backdrop-blur-md ${isDarkMode
                        ? 'bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700'
                        : 'bg-white/50 hover:bg-gray-100/50 border border-gray-200'
                        }`}
                    title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                >
                    {isDarkMode ? (
                        <Sun className="w-5 h-5 text-yellow-400" />
                    ) : (
                        <Moon className="w-5 h-5 text-gray-700" />
                    )}
                </button>
            </div>

            {/* Main Content */}
            <div className="relative z-10 min-h-screen flex items-center justify-center px-4">
                <div className="w-full max-w-md">
                    {/* Logo/Icon */}
                    <div className="flex justify-center mb-8 animate-fade-in">
                        <div className={`p-6 rounded-full backdrop-blur-xl ${isDarkMode
                            ? 'bg-green-500/10 border-2 border-green-500/30 shadow-lg shadow-green-500/20'
                            : 'bg-green-100/80 border-2 border-green-300/50 shadow-lg shadow-green-200/50'
                            }`}>
                            <Shield className={`w-16 h-16 ${isDarkMode ? 'text-green-400' : 'text-green-700'
                                }`} />
                        </div>
                    </div>

                    {/* Title */}
                    <h2 className={`text-4xl md:text-5xl font-bold mb-4 text-center animate-slide-up ${isDarkMode
                        ? 'text-transparent bg-clip-text bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400'
                        : 'text-transparent bg-clip-text bg-gradient-to-r from-green-700 via-emerald-700 to-teal-700'
                        }`}>
                        SurakshaNET Access
                    </h2>

                    {/* Subtitle */}
                    <p className={`text-center text-lg mb-8 animate-slide-up animation-delay-100 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'
                        }`}>
                        Authorized Personnel Only
                    </p>

                    {/* Login Form */}
                    <div className={`p-8 rounded-xl backdrop-blur-xl shadow-2xl animate-slide-up animation-delay-200 ${isDarkMode
                        ? 'bg-gray-900/80 border border-gray-800/50'
                        : 'bg-white/80 border border-gray-200/50'
                        }`}>
                        <form onSubmit={handleSubmit} className="space-y-6">
                            <div>
                                <label htmlFor="username" className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'
                                    }`}>
                                    Username
                                </label>
                                <input
                                    id="username"
                                    type="text"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    className={`w-full px-4 py-3 rounded-lg outline-none transition backdrop-blur-sm ${isDarkMode
                                        ? 'bg-gray-800/50 border border-gray-700 text-gray-200 focus:ring-2 focus:ring-green-500 focus:border-green-500'
                                        : 'bg-white/50 border border-gray-300 text-gray-900 focus:ring-2 focus:ring-green-600 focus:border-green-600'
                                        }`}
                                    placeholder="Enter your username"
                                    required
                                />
                            </div>

                            <div>
                                <label htmlFor="password" className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'
                                    }`}>
                                    Password
                                </label>
                                <input
                                    id="password"
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    className={`w-full px-4 py-3 rounded-lg outline-none transition backdrop-blur-sm ${isDarkMode
                                        ? 'bg-gray-800/50 border border-gray-700 text-gray-200 focus:ring-2 focus:ring-green-500 focus:border-green-500'
                                        : 'bg-white/50 border border-gray-300 text-gray-900 focus:ring-2 focus:ring-green-600 focus:border-green-600'
                                        }`}
                                    placeholder="Enter your password"
                                    required
                                />
                            </div>

                            <button
                                type="submit"
                                disabled={isLoading}
                                className={`w-full font-semibold py-3 rounded-lg transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl disabled:opacity-70 disabled:hover:scale-100 flex justify-center items-center ${isDarkMode
                                    ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white shadow-green-500/50 hover:shadow-green-500/60'
                                    : 'bg-gradient-to-r from-green-700 to-emerald-700 hover:from-green-600 hover:to-emerald-600 text-white shadow-green-300/50 hover:shadow-green-400/60'
                                    }`}
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                                        Authenticating...
                                    </>
                                ) : (
                                    'Sign In'
                                )}
                            </button>
                        </form>

                        <div className="mt-6 text-center">
                            <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'
                                }`}>System v1.0.0 | Encrypted Connection</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Custom Animations - Same as LandingPage */}
            <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes slide-up {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fade-in {
          animation: fade-in 1s ease-out forwards;
        }

        .animate-slide-up {
          animation: slide-up 0.8s ease-out forwards;
        }

        .animation-delay-100 {
          animation-delay: 0.1s;
          opacity: 0;
        }

        .animation-delay-200 {
          animation-delay: 0.2s;
          opacity: 0;
        }
      `}</style>
        </div>
    );
};

export default LoginView;

/**
 * Landing Page Component
 * ======================
 * Welcome page with animated background and call-to-action.
 * Features FloatingLines Three.js animation and hero section.
 */

import { ArrowRight, Moon, Sun } from 'lucide-react';
import FloatingLines from '../common/FloatingLines';

interface LandingPageProps {
    onGetStarted: () => void;
    isDarkMode: boolean;
    toggleTheme: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted, isDarkMode, toggleTheme }) => {
    return (
        <div className={`min-h-screen relative overflow-hidden transition-colors duration-300 ${isDarkMode
            ? 'bg-black'
            : 'bg-white'
            }`}>
            {/* Animated Background */}
            <div className="absolute inset-0 z-0">
                <FloatingLines
                    linesGradient={
                        isDarkMode
                            ? ['#10b981', '#059669', '#047857', '#065f46']
                            : ['#34d399', '#10b981', '#059669', '#047857']
                    }
                    enabledWaves={['middle', 'bottom']}
                    lineCount={[8, 6]}
                    lineDistance={[3, 5]}
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
                <div className="max-w-4xl mx-auto text-center">
                    {/* Logo/Icon */}
                    <div className="flex justify-center mb-8 animate-fade-in">
                        <div className={`p-6 rounded-full backdrop-blur-xl ${isDarkMode
                            ? 'bg-green-500/10 border-2 border-green-500/30 shadow-lg shadow-green-500/20'
                            : 'bg-green-100/80 border-2 border-green-300/50 shadow-lg shadow-green-200/50'
                            }`}>
                            <img src="/logo.png" alt="SurakshaNET Logo" className="w-16 h-16" />
                        </div>
                    </div>

                    {/* Title */}
                    <h1 className={`text-6xl md:text-7xl font-bold mb-6 animate-slide-up ${isDarkMode
                        ? 'text-transparent bg-clip-text bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400'
                        : 'text-transparent bg-clip-text bg-gradient-to-r from-green-700 via-emerald-700 to-teal-700'
                        }`}>
                        SurakshaNET
                    </h1>

                    {/* Subtitle */}
                    <p className={`text-xl md:text-2xl mb-4 animate-slide-up animation-delay-100 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'
                        }`}>
                        Advanced Network Security & Threat Intelligence
                    </p>

                    {/* Description */}
                    <p className={`text-base md:text-lg mb-12 max-w-2xl mx-auto animate-slide-up animation-delay-200 ${isDarkMode ? 'text-white' : 'text-white'
                        }`}>
                        Real-time monitoring, intelligent threat detection, and comprehensive security analytics
                        to protect your digital infrastructure.
                    </p>

                    {/* CTA Button */}
                    <div className="animate-slide-up animation-delay-300">
                        <button
                            onClick={onGetStarted}
                            className={`group relative px-8 py-4 text-lg font-semibold rounded-full transition-all duration-300 transform hover:scale-105 active:scale-95 ${isDarkMode
                                ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white shadow-lg shadow-green-500/50 hover:shadow-xl hover:shadow-green-500/60'
                                : 'bg-gradient-to-r from-green-700 to-emerald-700 hover:from-green-600 hover:to-emerald-600 text-white shadow-lg shadow-green-300/50 hover:shadow-xl hover:shadow-green-400/60'
                                }`}
                        >
                            <span className="flex items-center gap-2">
                                Get Started
                                <ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
                            </span>
                        </button>
                    </div>

                    {/* Features Grid */}
                    <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in animation-delay-400">
                        {[
                            { title: 'Real-Time Monitoring', desc: 'Track network activity 24/7' },
                            { title: 'Threat Intelligence', desc: 'AI-powered threat detection' },
                            { title: 'Detailed Reports', desc: 'Comprehensive security analytics' }
                        ].map((feature, idx) => (
                            <div
                                key={idx}
                                className={`p-6 rounded-xl backdrop-blur-md transition-all duration-300 hover:scale-105 ${isDarkMode
                                    ? 'bg-gray-900/40 border border-gray-800/50 hover:bg-gray-900/60 hover:border-green-500/30'
                                    : 'bg-white/40 border border-gray-200/50 hover:bg-white/60 hover:border-green-300/50'
                                    }`}
                            >
                                <h3 className={`text-lg font-semibold mb-2 ${isDarkMode ? 'text-green-400' : 'text-green-700'
                                    }`}>
                                    {feature.title}
                                </h3>
                                <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'
                                    }`}>
                                    {feature.desc}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div className="absolute bottom-6 left-0 right-0 z-10 text-center">
                <p className={`text-sm ${isDarkMode ? 'text-gray-600' : 'text-gray-500'
                    }`}>
                    © 2025 SurakshaNET | Secure. Monitor. Protect.
                </p>
            </div>

            {/* Custom Animations */}
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

        .animation-delay-300 {
          animation-delay: 0.3s;
          opacity: 0;
        }

        .animation-delay-400 {
          animation-delay: 0.4s;
          opacity: 0;
        }
      `}</style>
        </div>
    );
};

export default LandingPage;

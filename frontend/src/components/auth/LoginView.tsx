/**
 * Login View Component
 * ====================
 * Authentication page with username/password form.
 * TODO: Integrate with backend authentication API.
 */

import { useState } from 'react';
import { Shield, Loader2, Moon, Sun } from 'lucide-react';

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

        // TODO: Replace with real API call
        // try {
        //   const response = await fetch('/api/auth/login', { 
        //     method: 'POST', 
        //     body: JSON.stringify({ username, password })
        //   });
        //   const { token } = await response.json();
        //   localStorage.setItem('token', token);
        //   onLogin(username, password);
        // } catch (err) { ... }

        // Simulating API delay
        setTimeout(() => {
            setIsLoading(false);
            onLogin(username, password);
        }, 800);
    };

    return (
        <div className={`min-h-screen flex items-center justify-center p-4 font-sans transition-colors duration-300 ${isDarkMode
                ? 'bg-gradient-to-br from-black via-gray-900 to-green-950'
                : 'bg-gradient-to-br from-gray-50 via-white to-green-50'
            }`}>
            <div className={`p-8 rounded-xl shadow-2xl w-full max-w-md transition-colors duration-300 ${isDarkMode
                    ? 'bg-gray-900/90 border border-gray-800'
                    : 'bg-white border border-gray-200'
                }`}>
                {/* Theme Toggle Button */}
                <div className="flex justify-end mb-4">
                    <button
                        onClick={toggleTheme}
                        className={`p-2 rounded-md transition-colors ${isDarkMode
                                ? 'bg-gray-800 hover:bg-gray-700'
                                : 'bg-gray-100 hover:bg-gray-200'
                            }`}
                        title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                    >
                        {isDarkMode ? <Sun className="w-4 h-4 text-yellow-400" /> : <Moon className="w-4 h-4 text-gray-700" />}
                    </button>
                </div>

                <div className="flex flex-col items-center mb-8">
                    <div className={`p-3 rounded-full mb-4 ${isDarkMode
                            ? 'bg-green-500/10 border border-green-500/20'
                            : 'bg-green-100 border border-green-200'
                        }`}>
                        <Shield className={`w-8 h-8 ${isDarkMode ? 'text-green-400' : 'text-green-700'
                            }`} />
                    </div>
                    <h2 className={`text-2xl font-bold ${isDarkMode ? 'text-green-400' : 'text-green-700'
                        }`}>SurakshaNET Access</h2>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-500' : 'text-gray-600'
                        }`}>Authorized Personnel Only</p>
                </div>

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
                            className={`w-full px-4 py-2 rounded-lg outline-none transition ${isDarkMode
                                    ? 'bg-gray-800 border border-gray-700 text-gray-200 focus:ring-2 focus:ring-green-500 focus:border-green-500'
                                    : 'bg-white border border-gray-300 text-gray-900 focus:ring-2 focus:ring-green-600 focus:border-green-600'
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
                            className={`w-full px-4 py-2 rounded-lg outline-none transition ${isDarkMode
                                    ? 'bg-gray-800 border border-gray-700 text-gray-200 focus:ring-2 focus:ring-green-500 focus:border-green-500'
                                    : 'bg-white border border-gray-300 text-gray-900 focus:ring-2 focus:ring-green-600 focus:border-green-600'
                                }`}
                            placeholder="Enter your password"
                            required
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={isLoading}
                        className={`w-full font-semibold py-3 rounded-lg transition duration-200 shadow-lg hover:shadow-xl disabled:opacity-70 flex justify-center items-center ${isDarkMode
                                ? 'bg-green-600 hover:bg-green-500 text-white'
                                : 'bg-green-700 hover:bg-green-600 text-white'
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
                    <span className={`text-xs ${isDarkMode ? 'text-gray-600' : 'text-gray-500'
                        }`}>System v1.0.0 | Encrypted Connection</span>
                </div>
            </div>
        </div>
    );
};

export default LoginView;

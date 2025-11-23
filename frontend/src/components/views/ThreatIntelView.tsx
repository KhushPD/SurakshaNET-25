/**
 * Threat Intelligence View
 * =========================
 * View for global threat intelligence feeds with AI chatbot.
 */

import { Send, Sparkles, Loader2 } from 'lucide-react';
import { useState, useEffect, useRef, useCallback } from 'react';

interface ThreatIntelViewProps {
    isDarkMode: boolean;
    initialMessage?: string;
    initialContext?: Record<string, unknown>;
}

interface Message {
    id: string;
    type: 'user' | 'bot';
    content: string;
    timestamp: Date;
}

const ThreatIntelView: React.FC<ThreatIntelViewProps> = ({ isDarkMode, initialMessage, initialContext }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [hasProcessedInitial, setHasProcessedInitial] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const API_BASE = 'http://localhost:8000';

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = useCallback(async (messageText?: string, context?: Record<string, unknown>) => {
        const textToSend = messageText || input;
        if (!textToSend.trim()) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            type: 'user',
            content: textToSend,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const requestBody: Record<string, unknown> = { message: textToSend };
            if (context) {
                requestBody.context = context;
            }

            const response = await fetch(`${API_BASE}/chatbot/message`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();

            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                type: 'bot',
                content: data.response || 'Sorry, I could not process that request.',
                timestamp: new Date()
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Chat error:', error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                type: 'bot',
                content: '⚠️ Connection error. Please ensure the backend is running.',
                timestamp: new Date()
            };
            setMessages(prev => [...prev, errorMessage]);
        }

        setIsLoading(false);
    }, [input]);

    useEffect(() => {
        // Welcome message only once
        if (messages.length === 0) {
            setMessages([{
                id: Date.now().toString(),
                type: 'bot',
                content: `Hello! I'm your **Threat Intelligence Assistant** 🛡️

I can help you with:
- 🔍 Analyzing network logs and threats
- 🎯 Explaining attack patterns
- 💡 Providing security recommendations
- 🤖 Answering cybersecurity questions

Try asking me about DoS attacks, SQL injection, or click "Ask AI" on any log entry!`,
                timestamp: new Date()
            }]);
        }
    }, []);

    useEffect(() => {
        // Handle initial message from DataLogs when context is provided
        if (initialMessage && initialContext && !hasProcessedInitial && messages.length > 0) {
            setHasProcessedInitial(true);
            // Small delay to ensure welcome message is rendered
            setTimeout(() => {
                handleSend(initialMessage, initialContext);
            }, 300);
        }
    }, [initialMessage, initialContext, hasProcessedInitial, messages.length, handleSend]);

    const formatMessage = (content: string) => {
        // Simple markdown-like formatting
        return content
            .split('\n')
            .map((line, i) => {
                if (line.startsWith('## ')) {
                    return <h3 key={i} className="text-xl font-bold mt-4 mb-2">{line.slice(3)}</h3>;
                } else if (line.startsWith('### ')) {
                    return <h4 key={i} className="text-lg font-semibold mt-3 mb-2">{line.slice(4)}</h4>;
                } else if (line.startsWith('**') && line.endsWith('**')) {
                    return <p key={i} className="font-bold mt-2">{line.slice(2, -2)}</p>;
                } else if (line.startsWith('- ')) {
                    return <li key={i} className="ml-4 mb-1">{line.slice(2)}</li>;
                } else if (line.trim() === '') {
                    return <br key={i} />;
                } else {
                    // Handle inline bold
                    const parts = line.split(/(\*\*.*?\*\*)/g);
                    return (
                        <p key={i} className="mb-1">
                            {parts.map((part, j) => 
                                part.startsWith('**') && part.endsWith('**') 
                                    ? <strong key={j}>{part.slice(2, -2)}</strong>
                                    : part
                            )}
                        </p>
                    );
                }
            });
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-6">
            <div className="w-full max-w-4xl">
                {/* Hero Section */}
                <div className="text-center mb-8 space-y-4">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full mb-4"
                         style={{
                             background: isDarkMode 
                                 ? 'radial-gradient(circle at center, rgba(34, 197, 94, 0.2), transparent)'
                                 : 'radial-gradient(circle at center, rgba(34, 197, 94, 0.15), transparent)',
                             backdropFilter: 'blur(20px)'
                         }}>
                        <Sparkles className={`w-8 h-8 ${isDarkMode ? 'text-green-400' : 'text-green-600'}`} />
                    </div>
                    <h1 className={`text-4xl md:text-5xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                        Smarter security insights with
                        <br />
                        <span className="bg-gradient-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent">
                            AI-powered analysis
                        </span>
                    </h1>
                    <p className={`text-lg ${isDarkMode ? 'text-gray-400' : 'text-gray-600'} max-w-2xl mx-auto`}>
                        Analyze threats, explain attacks, and get expert security recommendations
                        powered by advanced AI.
                    </p>
                </div>

                {/* Chat Container with Glow Effect */}
                <div className="relative">
                    {/* Glow Effect */}
                    <div className="absolute -inset-1 bg-gradient-to-r from-green-500/20 via-emerald-500/20 to-green-500/20 rounded-2xl blur-xl opacity-75"></div>
                    
                    {/* Chat Box */}
                    <div className={`relative rounded-2xl overflow-hidden ${
                        isDarkMode 
                            ? 'bg-gray-900/90 border border-gray-800' 
                            : 'bg-white/90 border border-gray-200'
                    }`} style={{ backdropFilter: 'blur(20px)' }}>
                        
                        {/* Messages Area */}
                        <div className="h-96 overflow-y-auto p-6 space-y-4">
                            {messages.map((message) => (
                                <div
                                    key={message.id}
                                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    <div
                                        className={`max-w-2xl rounded-2xl px-4 py-3 ${
                                            message.type === 'user'
                                                ? isDarkMode
                                                    ? 'bg-gradient-to-r from-green-600 to-emerald-600 text-white'
                                                    : 'bg-gradient-to-r from-green-700 to-emerald-700 text-white'
                                                : isDarkMode
                                                ? 'bg-gray-800/80 text-gray-200 border border-gray-700/50'
                                                : 'bg-gray-50 text-gray-800 border border-gray-200'
                                        }`}
                                    >
                                        {message.type === 'bot' ? (
                                            <div className="space-y-1 text-sm leading-relaxed">
                                                {formatMessage(message.content)}
                                            </div>
                                        ) : (
                                            <div className="text-sm">{message.content}</div>
                                        )}
                                    </div>
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className={`rounded-2xl px-4 py-3 flex items-center gap-2 ${
                                        isDarkMode ? 'bg-gray-800/80 text-gray-200' : 'bg-gray-50 text-gray-800'
                                    }`}>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        <span className="text-sm">Analyzing...</span>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input Area with Glassmorphism */}
                        <div className={`border-t p-4 ${
                            isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50/50 border-gray-200'
                        }`}>
                            <div className="flex gap-3 items-center">
                                {/* Input Field */}
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSend()}
                                    placeholder="Ask about threats, attacks, or security best practices..."
                                    disabled={isLoading}
                                    className={`flex-1 px-4 py-3 rounded-xl outline-none transition-all ${
                                        isDarkMode
                                            ? 'bg-gray-900/50 border border-gray-700 text-gray-200 placeholder-gray-500 focus:border-green-500 focus:ring-2 focus:ring-green-500/20'
                                            : 'bg-white border border-gray-300 text-gray-800 placeholder-gray-400 focus:border-green-600 focus:ring-2 focus:ring-green-500/20'
                                    } disabled:opacity-50`}
                                />

                                {/* Send Button with Gradient */}
                                <button
                                    onClick={() => handleSend()}
                                    disabled={isLoading || !input.trim()}
                                    className={`p-3 rounded-xl transition-all flex items-center justify-center ${
                                        isDarkMode
                                            ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white'
                                            : 'bg-gradient-to-r from-green-700 to-emerald-700 hover:from-green-600 hover:to-emerald-600 text-white'
                                    } disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-green-500/20`}
                                >
                                    <Send className="w-5 h-5" />
                                </button>
                            </div>
                            
                            {/* AI Notice */}
                            <p className={`text-xs mt-3 text-center ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                AI-powered threat intelligence • Real-time analysis • Enterprise security
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ThreatIntelView;

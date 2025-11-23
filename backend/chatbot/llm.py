"""
Threat Intelligence Chatbot with Groq LLM
==========================================
Chatbot for threat intelligence using Groq API

Setup:
1. Get free API key from: https://console.groq.com
2. Add to .env file: GROQ_API_KEY=your_key_here
3. Install: pip install groq python-dotenv
"""
import os
from typing import Optional, Dict, List
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not installed. Install with: pip install groq")

class ThreatIntelChatbot:
    """Chatbot for threat intelligence analysis"""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.system_prompt = """You are a cybersecurity threat intelligence assistant specializing in network security and intrusion detection. 

Your role:
- Analyze network logs and identify threats
- Explain attack patterns (DoS, Probe, R2L, U2R, SQL Injection, XSS) in simple words with examples
- Provide actionable security recommendations in step by step manner
- Help with threat mitigation strategies

Be concise, technical, and security-focused. Format responses with clear sections using markdown."""
        
        # Initialize Groq client
        self.groq_client = None
        self.use_groq = False
        
        api_key = os.getenv('GROQ_API_KEY')
        if api_key and GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key=api_key)
                self.use_groq = True
                logger.info("✓ Groq AI enabled for chatbot")
            except Exception as e:
                logger.error(f"Groq initialization failed: {e}")
                self.use_groq = False
        else:
            if not api_key:
                logger.warning("⚠ GROQ_API_KEY not found in .env file. Using fallback responses.")
            if not GROQ_AVAILABLE:
                logger.warning("⚠ Install Groq: pip install groq")
    
    def analyze_log(self, log_data: Dict) -> str:
        """Analyze a specific log entry"""
        response = f"""
## Log Analysis

**Source IP:** {log_data.get('source_ip', 'Unknown')}
**Destination IP:** {log_data.get('destination_ip', 'Unknown')}
**Status:** {log_data.get('status', 'Unknown')}
**Type:** {log_data.get('intrusion_type') or log_data.get('request_type', 'Unknown')}

"""
        
        if log_data.get('status') == 'detected':
            confidence = log_data.get('confidence', 0) * 100
            intrusion_type = log_data.get('intrusion_type', 'Unknown')
            
            response += f"""
### 🚨 Threat Detected

**Attack Type:** {intrusion_type}
**Confidence:** {confidence:.1f}%

**Recommended Actions:**
"""
            
            if intrusion_type == 'DoS':
                response += """
- Block source IP immediately
- Check for similar patterns from other IPs
- Verify server capacity and enable rate limiting
- Review firewall rules
"""
            elif intrusion_type == 'Probe':
                response += """
- Monitor source IP for further activity
- Check for port scanning patterns
- Review exposed services
- Enable intrusion prevention system (IPS)
"""
            elif intrusion_type in ['R2L', 'U2R']:
                response += """
- **CRITICAL**: Potential unauthorized access attempt
- Block source IP and review authentication logs
- Check for compromised credentials
- Audit system access controls
"""
            else:
                response += """
- Monitor the source IP
- Review recent activity patterns
- Update security rules if needed
"""
                
            if confidence > 90:
                response += "\n**High confidence detection - immediate action recommended**"
            elif confidence > 70:
                response += "\n**Medium-high confidence - monitor closely**"
            else:
                response += "\n**Lower confidence - verify with additional analysis**"
                
        else:
            response += """
### ✅ Normal Traffic

This appears to be legitimate traffic. No immediate action required.
Continue monitoring for any anomalies.
"""
        
        return response
    
    def chat(self, message: str, context: Optional[Dict] = None) -> str:
        """Process chat message with optional context - AI first, fallback only on failure"""
        
        # If context provided (log data), analyze it with AI
        if context and context.get('log_data'):
            return self._analyze_log_with_ai(context['log_data'], message)
        
        # ALWAYS try Groq AI first if available
        if self.use_groq and self.groq_client:
            ai_response = self._chat_with_groq(message)
            # Only fallback if AI explicitly failed (returns None or error message)
            if ai_response and not ai_response.startswith("⚠️ AI Error"):
                return ai_response
        
        # Fallback to pattern-based responses only if AI failed
        return self._fallback_response(message)
    
    def _chat_with_groq(self, message: str) -> str:
        """Chat using Groq AI - returns AI response or error indicator"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": message
            })
            
            # Keep only last 10 messages to manage context
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Create messages with system prompt
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history
            
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model="qwen/qwen3-32b",  # Fast and capable model
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                top_p=0.9,
                stream=False
            )
            
            # Extract content and remove thinking tags if present
            assistant_message = response.choices[0].message.content
            
            # Remove <think>...</think> tags and their content
            import re
            assistant_message = re.sub(r'<think>.*?</think>', '', assistant_message, flags=re.DOTALL)
            assistant_message = assistant_message.strip()
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            logger.info("✓ Groq AI response generated successfully")
            return assistant_message
            
        except Exception as e:
            logger.error(f"❌ Groq API error: {e}")
            # Return error indicator so fallback can be used
            return f"⚠️ AI Error: {str(e)}"
    
    def _analyze_log_with_ai(self, log_data: Dict, user_message: str) -> str:
        """Analyze log with AI - prioritize AI response, fallback only on failure"""
        
        # Add XAI explanation if available
        xai_section = ""
        try:
            from reporting.blockchain_service import xai_explainer
            if xai_explainer and 'features' in log_data:
                features = log_data.get('features', [])
                if features:
                    xai_text = xai_explainer.get_explanation_text(features)
                    xai_section = f"\n\n{xai_text}\n"
        except Exception as e:
            logger.debug(f"XAI explanation not available: {e}")
        
        # Add blockchain verification
        blockchain_section = ""
        try:
            from reporting.blockchain_service import threat_blockchain
            block = threat_blockchain.get_block_by_ip(log_data.get('source_ip'))
            if block:
                blockchain_section = f"""

## 🔗 Blockchain Verification

**Block Hash:** `{block['current_hash'][:32]}...`
**Block Index:** #{block['index']}
**Timestamp:** {block['timestamp']}
**Previous Hash:** `{block['previous_hash'][:16]}...`

✅ This threat record is stored in our immutable blockchain audit trail.
"""
        except Exception as e:
            logger.debug(f"Blockchain info not available: {e}")
        
        # Build comprehensive log analysis prompt
        log_info = f"""Analyze this network log entry in detail:

**Log Details:**
- Source IP: {log_data.get('source_ip', 'Unknown')}
- Destination IP: {log_data.get('destination_ip', 'Unknown')}
- Status: {log_data.get('status', 'Unknown')}
- Attack Type: {log_data.get('intrusion_type') or log_data.get('request_type', 'Unknown')}
- Detection Confidence: {log_data.get('confidence', 0) * 100:.1f}%
- Timestamp: {log_data.get('timestamp', 'Unknown')}

**Provide a comprehensive analysis with:**
1. **Threat Assessment** - What type of attack is this and how serious is it?
2. **Technical Analysis** - Explain the attack pattern and what the attacker is trying to do
3. **Recommended Actions** - Specific steps to take (blocking, monitoring, investigation)
4. **Risk Level** - Critical/High/Medium/Low and why
5. **Prevention** - How to prevent similar attacks in the future

Format your response with clear markdown sections (##) and bullet points."""

        # ALWAYS try AI first
        if self.use_groq and self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": log_info}
                    ],
                    temperature=0.5,
                    max_tokens=1500,
                    stream=False
                )
                ai_response = response.choices[0].message.content
                
                # Remove <think>...</think> tags and their content
                import re
                ai_response = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL)
                ai_response = ai_response.strip()
                
                # Append XAI and blockchain sections
                full_response = ai_response + xai_section + blockchain_section
                
                logger.info("✓ Groq AI log analysis completed")
                return full_response
            except Exception as e:
                logger.error(f"❌ Groq API error during log analysis: {e}")
        
        # Only fallback if AI failed - still include XAI and blockchain
        logger.warning("⚠️ Using fallback log analysis (AI unavailable)")
        fallback = self.analyze_log(log_data)
        return fallback + xai_section + blockchain_section
    
    def _fallback_response(self, message: str) -> str:
        """Pattern-based fallback responses"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return """Hello! I'm your Threat Intelligence Assistant. 

I can help you with:
- Analyzing network logs and threats
- Explaining attack patterns
- Providing security recommendations
- Answering cybersecurity questions

How can I assist you today?"""
        
        elif any(word in message_lower for word in ['dos', 'ddos', 'denial of service']):
            return """## DoS/DDoS Attacks

**Description:** Attempts to overwhelm a system with traffic, making it unavailable.

**Detection Indicators:**
- High volume of requests from single/multiple IPs
- Abnormal traffic patterns
- Server performance degradation

**Mitigation:**
- Rate limiting and traffic filtering
- IP blocking for malicious sources
- CDN and load balancing
- DDoS protection services

**Prevention:**
- Configure proper firewall rules
- Monitor traffic patterns
- Use intrusion prevention systems"""
        
        elif any(word in message_lower for word in ['probe', 'scan', 'reconnaissance']):
            return """## Probe/Scanning Attacks

**Description:** Attackers scan your network to discover vulnerabilities.

**Detection Indicators:**
- Sequential port scanning
- Multiple connection attempts
- Service enumeration attempts

**Mitigation:**
- Block scanning IPs
- Reduce service exposure
- Use port knocking
- Enable stealth mode

**Prevention:**
- Minimize exposed services
- Use intrusion detection systems
- Regular security audits"""
        
        elif any(word in message_lower for word in ['r2l', 'remote to local']):
            return """## R2L (Remote to Local) Attacks

**Description:** Unauthorized access attempts from remote machines.

**Detection Indicators:**
- Failed login attempts
- Brute force patterns
- Credential stuffing

**Mitigation:**
- **IMMEDIATE:** Block source IPs
- Force password resets
- Enable MFA
- Review access logs

**Prevention:**
- Strong authentication
- Multi-factor authentication
- Account lockout policies
- Monitor failed logins"""
        
        elif any(word in message_lower for word in ['u2r', 'user to root', 'privilege']):
            return """## U2R (User to Root) Attacks

**Description:** Privilege escalation attempts.

**Detection Indicators:**
- Suspicious privilege changes
- Unauthorized access attempts
- Buffer overflow attempts

**Mitigation:**
- **CRITICAL:** Isolate affected systems
- Audit privilege changes
- Review sudo/admin access
- Check for rootkits

**Prevention:**
- Principle of least privilege
- Regular security patches
- File integrity monitoring
- Security hardening"""
        
        elif any(word in message_lower for word in ['sql', 'injection']):
            return """## SQL Injection

**Description:** Malicious SQL code injection into queries.

**Detection Indicators:**
- SQL syntax in URL parameters
- Database error messages
- Unusual query patterns

**Mitigation:**
- Use parameterized queries
- Input validation
- Web Application Firewall (WAF)
- Database activity monitoring

**Prevention:**
- Prepared statements
- ORM frameworks
- Input sanitization
- Regular code reviews"""
        
        elif any(word in message_lower for word in ['xss', 'cross-site']):
            return """## Cross-Site Scripting (XSS)

**Description:** Injection of malicious scripts into web pages.

**Detection Indicators:**
- Script tags in input
- Suspicious JavaScript execution
- Cookie theft attempts

**Mitigation:**
- Content Security Policy (CSP)
- Output encoding
- Input validation
- WAF rules

**Prevention:**
- Sanitize all user inputs
- Encode outputs
- Use security headers
- Regular security testing"""
        
        elif any(word in message_lower for word in ['block', 'ip', 'firewall']):
            return """## IP Blocking & Firewall

**Current System:**
- Automatic blocking for high-confidence threats (>85%)
- Default block duration: 30 minutes
- Can manually block/unblock IPs

**Best Practices:**
- Block malicious IPs immediately
- Maintain blocklist updates
- Use geo-blocking when appropriate
- Whitelist trusted sources
- Regular firewall rule reviews

**Commands:**
- View blocked IPs: `/blocked-ips` endpoint
- Block IP: `/block-ip` endpoint
- Unblock IP: `/unblock-ip` endpoint"""
        
        elif any(word in message_lower for word in ['help', 'what can you do']):
            return """## Threat Intelligence Assistant

**I can help with:**

1. **Log Analysis** - Click "Ask AI" on any log to get detailed analysis
2. **Threat Explanation** - Ask about DoS, Probe, R2L, U2R attacks
3. **Security Recommendations** - Get mitigation strategies
4. **Attack Patterns** - Learn about different attack types
5. **Best Practices** - Security configuration advice

**Try asking:**
- "What is a DoS attack?"
- "How to prevent SQL injection?"
- "Explain probe attacks"
- "What are best practices for IP blocking?"

Or click the "Ask AI" button on any log entry!"""
        
        else:
            return f"""I'm your Threat Intelligence Assistant. 

You asked: "{message}"

I can help with network security topics like:
- Attack types (DoS, Probe, R2L, U2R, SQL Injection, XSS)
- Log analysis and threat detection
- Security recommendations
- IP blocking and firewall rules

Try asking about specific attack types or click "Ask AI" on a log entry for detailed analysis!"""
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []

# Global chatbot instance
threat_intel_bot = ThreatIntelChatbot()

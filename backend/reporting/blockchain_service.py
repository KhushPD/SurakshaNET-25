"""
Blockchain Audit Trail + Explainable AI (XAI)
For Network Intrusion Detection System
"""

import hashlib
import json
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# BONUS 1: BLOCKCHAIN AUDIT TRAIL
# ============================================================================

class ThreatBlockchain:
    """
    Immutable blockchain for logging detected intrusions
    Each block is cryptographically linked using SHA-256
    """
    
    def __init__(self):
        self.chain = []
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Initialize blockchain with genesis block"""
        genesis = {
            'index': 0,
            'timestamp': datetime.now().isoformat(),
            'threat_data': 'Genesis Block - Security Log Initialized',
            'previous_hash': '0' * 64,
        }
        genesis['current_hash'] = self._calculate_hash(genesis)
        self.chain.append(genesis)
        logger.info("🔗 Blockchain initialized")
    
    def _calculate_hash(self, block):
        """
        Calculate SHA-256 hash of block
        Ensures tamper-proof records
        """
        # Create string of block data
        block_content = json.dumps({
            'index': block['index'],
            'timestamp': block['timestamp'],
            'threat_data': block['threat_data'],
            'previous_hash': block['previous_hash']
        }, sort_keys=True)
        
        # SHA-256 hash
        return hashlib.sha256(block_content.encode()).hexdigest()
    
    def add_threat_block(self, threat_info):
        """
        Add new intrusion to blockchain
        
        Args:
            threat_info: dict with keys like attack_type, confidence, 
                        action_taken, source_ip, etc.
        """
        previous_block = self.chain[-1]
        
        new_block = {
            'index': len(self.chain),
            'timestamp': datetime.now().isoformat(),
            'threat_data': threat_info,
            'previous_hash': previous_block['current_hash']
        }
        
        # Calculate hash including previous hash (creates chain)
        new_block['current_hash'] = self._calculate_hash(new_block)
        
        self.chain.append(new_block)
        
        logger.info(f"✅ Block #{new_block['index']} added to blockchain")
        
        return new_block
    
    def get_block_by_ip(self, source_ip):
        """Get blockchain block for specific IP"""
        for block in reversed(self.chain):
            if isinstance(block['threat_data'], dict):
                if block['threat_data'].get('source_ip') == source_ip:
                    return block
        return None
    
    def verify_integrity(self):
        """
        Verify blockchain hasn't been tampered with
        Returns True if chain is valid, False if corrupted
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check 1: Current block hash is correct
            recalculated_hash = self._calculate_hash(current_block)
            if current_block['current_hash'] != recalculated_hash:
                print(f"❌ Block #{i} hash mismatch - TAMPERED!")
                return False
            
            # Check 2: Previous hash link is correct
            if current_block['previous_hash'] != previous_block['current_hash']:
                print(f"❌ Block #{i} chain broken - TAMPERED!")
                return False
        
        print("✅ Blockchain integrity verified - NO tampering detected")
        return True
    
    def display_audit_trail(self, last_n=5):
        """Display recent blocks from audit trail"""
        print("\n" + "="*70)
        print("🔐 IMMUTABLE THREAT AUDIT TRAIL (Blockchain)")
        print("="*70)
        
        for block in self.chain[-last_n:]:
            print(f"\n📦 Block #{block['index']}")
            print(f"   Timestamp: {block['timestamp']}")
            print(f"   Hash: {block['current_hash']}")
            print(f"   Previous: {block['previous_hash']}")
            
            if isinstance(block['threat_data'], dict):
                print(f"   Attack Type: {block['threat_data'].get('attack_type', 'N/A')}")
                print(f"   Confidence: {block['threat_data'].get('confidence', 0):.2%}")
                print(f"   Action: {block['threat_data'].get('action_taken', 'N/A')}")
        
        print("\n" + "="*70)
        self.verify_integrity()


# ============================================================================
# BONUS 2: EXPLAINABLE AI (XAI) - SHAP Feature Importance
# ============================================================================

class ExplainableIDS:
    """
    Explainable AI wrapper for intrusion detection
    Shows WHY model classified traffic as malicious
    """
    
    def __init__(self, model, feature_names):
        """
        Args:
            model: Trained sklearn model (e.g., RandomForest)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        
        # For RandomForest, use built-in feature importance
        # (similar to SHAP values but faster)
        self.global_importance = model.feature_importances_
    
    def explain_prediction(self, flow_features):
        """
        Explain why a specific flow was classified as malicious
        
        Args:
            flow_features: numpy array or list of network flow features
            
        Returns:
            dict with prediction and feature contributions
        """
        # Convert to numpy array if needed
        if not isinstance(flow_features, np.ndarray):
            flow_features = np.array(flow_features)
        
        # Make prediction
        prediction = self.model.predict([flow_features])[0]
        confidence = self.model.predict_proba([flow_features])[0]
        
        # Calculate feature contributions using feature importance
        feature_contributions = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(flow_features):
                contribution = self.global_importance[i] * abs(float(flow_features[i]))
                feature_contributions[feature_name] = contribution
        
        # Sort by importance
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'prediction': 'MALICIOUS' if prediction == 1 else 'BENIGN',
            'confidence': float(confidence[1] if prediction == 1 else confidence[0]),
            'top_features': sorted_features[:5],
            'all_contributions': feature_contributions
        }
    
    def get_explanation_text(self, flow_features):
        """Get formatted explanation text for chatbot"""
        result = self.explain_prediction(flow_features)
        
        text = f"""## 🔍 Explainable AI Analysis

**Classification:** {result['prediction']}
**Confidence:** {result['confidence']:.1%}

### Top Contributing Features (SHAP-style)

"""
        for i, (feature, importance) in enumerate(result['top_features'], 1):
            bar_length = int(importance * 20)
            bar = "█" * bar_length
            text += f"{i}. **{feature}**: {bar} ({importance:.3f})\n"
        
        text += "\n*These features had the strongest influence on the model's decision.*"
        
        return text


# Global instances
threat_blockchain = ThreatBlockchain()

# Note: ExplainableIDS instance will be created when models are loaded
xai_explainer = None

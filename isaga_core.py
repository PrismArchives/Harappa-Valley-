"""
ISAGA 2.0: Indus Script Administrative Grammar Analyzer
-------------------------------------------------------
A computational framework for validating, visualizing, and predicting
Indus Valley sign sequences using Bayesian inference and Graph Theory.

Copyright (c) 2025 IndusLogic
License: MIT
"""

import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

# ============================================
# MODULE 1: REAL SIGN DATABASE (Mahadevan/Parpola)
# ============================================

class IndusSignDatabase:
    """Centralized database of Indus signs with Mahadevan/Parpola IDs"""
    
    # Core administrative signs based on Mahadevan's corpus
    SIGN_CATALOG = {
        # TERMINAL SIGNS (End of transaction)
        342: {"name": "JAR", "role": "TERMINAL", "frequency": 1200, "description": "Storage jar - marks completion"},
        343: {"name": "MARKED_JAR", "role": "TERMINAL", "frequency": 450, "description": "Jar with internal mark"},
        344: {"name": "DOUBLE_JAR", "role": "TERMINAL", "frequency": 120, "description": "Paired jars"},
        
        # COMMODITY SIGNS (Root concepts)
        59: {"name": "FISH", "role": "COMMODITY", "frequency": 850, "description": "Fish commodity (star/deity?)"},
        60: {"name": "FISH2", "role": "COMMODITY", "frequency": 320, "description": "Fish variant"},
        211: {"name": "WHEEL", "role": "COMMODITY", "frequency": 560, "description": "Wheel/Sun symbol"},
        123: {"name": "MAN", "role": "COMMODITY", "frequency": 410, "description": "Anthropomorphic figure"},
        
        # OPERATOR SIGNS (Grammatical markers)
        99: {"name": "ARROW", "role": "OPERATOR", "frequency": 380, "description": "Directional/transformative marker"},
        456: {"name": "UNICORN", "role": "OPERATOR", "frequency": 290, "description": "Seal icon - possibly header"},
        789: {"name": "STROKE", "role": "QUANTITY", "frequency": 670, "description": "Counting stroke"},
        
        # COMPOUND SIGNS
        65: {"name": "FISH+ROOF", "role": "MODIFIED_COMMODITY", "frequency": 180, "description": "Fish with roof modifier"},
        212: {"name": "WHEEL+ARROW", "role": "MODIFIED_COMMODITY", "frequency": 95, "description": "Wheel with arrow"},
    }
    
    # Grammar constraints based on Wells (2015) and Parpola (1994)
    GRAMMAR_RULES = {
        "TERMINAL_MUST_END": True,
        "MIN_SEQUENCE_LENGTH": 2,
        "MAX_SEQUENCE_LENGTH": 8,
        "FORBIDDEN_TRANSITIONS": [
            ("TERMINAL", "COMMODITY"),  # Can't go from terminal to commodity
            ("TERMINAL", "OPERATOR"),   # Can't go from terminal to operator
            ("QUANTITY", "OPERATOR"),   # Can't go from quantity to operator
        ]
    }
    
    @classmethod
    def get_sign_info(cls, sign_id: int) -> Dict:
        return cls.SIGN_CATALOG.get(sign_id, {
            "name": f"SIGN_{sign_id}", "role": "UNKNOWN", "frequency": 1, "description": "Unidentified sign"
        })
    
    @classmethod
    def get_role(cls, sign_id: int) -> str:
        return cls.get_sign_info(sign_id).get("role", "UNKNOWN")
    
    @classmethod
    def get_name(cls, sign_id: int) -> str:
        return cls.get_sign_info(sign_id).get("name", f"SIGN_{sign_id}")
    
    @classmethod
    def can_follow(cls, sign1_id: int, sign2_id: int) -> bool:
        """Check if sign2 can follow sign1 based on grammar rules"""
        role1 = cls.get_role(sign1_id)
        role2 = cls.get_role(sign2_id)
        
        if (role1, role2) in cls.GRAMMAR_RULES["FORBIDDEN_TRANSITIONS"]:
            return False
        if role1 == "TERMINAL": # Terminals are sinks
            return False
            
        return True

# ============================================
# MODULE 2: CORE ANALYZER (Syntax Validation)
# ============================================

class IndusInscription:
    """Represents a single inscription with validation logic"""
    
    def __init__(self, sign_ids: List[int], provenance: str = "Unknown"):
        self.signs = sign_ids
        self.provenance = provenance
        self.db = IndusSignDatabase()
        
    def validate_syntax(self) -> Dict:
        """Validate inscription against grammar rules"""
        
        if len(self.signs) < self.db.GRAMMAR_RULES["MIN_SEQUENCE_LENGTH"]:
            return {"valid": False, "error": "Sequence too short (Frag?)"}
            
        # Check terminal position
        last_sign = self.signs[-1]
        if self.db.get_role(last_sign) != "TERMINAL":
            # Some quantity sequences are exceptions, but we flag for now
            if self.db.get_role(last_sign) != "QUANTITY":
                return {"valid": False, "error": "Protocol Violation: Missing Terminal Seal (Jar)."}
        
        # Check forbidden transitions
        for i in range(len(self.signs) - 1):
            if not self.db.can_follow(self.signs[i], self.signs[i+1]):
                role1 = self.db.get_role(self.signs[i])
                role2 = self.db.get_role(self.signs[i+1])
                return {"valid": False, "error": f"Logic Error: {role1} cannot flow into {role2}"}
        
        return {"valid": True, "message": "Valid Administrative Protocol"}
    
    def to_readable_string(self) -> str:
        return " → ".join([self.db.get_name(s) for s in self.signs])

# ============================================
# MODULE 3: CORPUS ANALYZER (Bigram Statistics)
# ============================================

class CorpusAnalyzer:
    """Analyzes corpus patterns and builds transition matrix"""
    
    def __init__(self):
        self.db = IndusSignDatabase()
        self.bigram_counts = defaultdict(int)
        
    def add_inscription(self, inscription: IndusInscription):
        for i in range(len(inscription.signs) - 1):
            source = self.db.get_name(inscription.signs[i])
            target = self.db.get_name(inscription.signs[i + 1])
            self.bigram_counts[(source, target)] += 1
            
    def get_transition_probability(self, source_name: str, target_name: str) -> float:
        total_source = sum(count for (src, _), count in self.bigram_counts.items() if src == source_name)
        if total_source == 0: return 0.0
        return self.bigram_counts.get((source_name, target_name), 0) / total_source

# ============================================
# MODULE 4: PREDICTIVE REPAIR ENGINE (Bayesian)
# ============================================

class PredictiveRepairEngine:
    """Bayesian reconstruction engine for broken inscriptions"""
    
    def __init__(self, corpus_analyzer: CorpusAnalyzer):
        self.analyzer = corpus_analyzer
        self.db = IndusSignDatabase()
        self.transition_matrix = self._build_transition_matrix()
        
    def _build_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        matrix = defaultdict(dict)
        source_totals = defaultdict(int)
        for (source, target), count in self.analyzer.bigram_counts.items():
            source_totals[source] += count
        for (source, target), count in self.analyzer.bigram_counts.items():
            matrix[source][target] = count / source_totals[source]
        return matrix
    
    def predict_missing_sign(self, sequence: List[int], gap_index: int) -> List[Tuple[int, float, str]]:
        """Predict missing sign at gap_index. Returns list of (ID, Conf, Logic)"""
        
        # Context
        pre_sign = sequence[gap_index - 1] if gap_index > 0 else None
        post_sign = sequence[gap_index + 1] if gap_index < len(sequence) - 1 else None
        
        pre_name = self.db.get_name(pre_sign) if pre_sign else "START"
        post_name = self.db.get_name(post_sign) if post_sign else "END"
        
        candidates = []
        
        for sign_id, info in self.db.SIGN_CATALOG.items():
            sign_name = info["name"]
            
            # P(pre -> candidate)
            p_forward = 0.0
            if pre_name in self.transition_matrix:
                p_forward = self.transition_matrix[pre_name].get(sign_name, 0.0)
            
            # P(candidate -> post)
            p_backward = 0.0
            if sign_name in self.transition_matrix:
                p_backward = self.transition_matrix[sign_name].get(post_name, 0.0)
            
            # Joint Probability
            joint_prob = p_forward * p_backward
            
            if joint_prob > 0:
                # Grammar Penalty
                if pre_sign and not self.db.can_follow(pre_sign, sign_id): joint_prob *= 0.1
                if post_sign and not self.db.can_follow(sign_id, post_sign): joint_prob *= 0.1
                    
                explanation = f"P({pre_name}→{sign_name})={p_forward:.2f} * P({sign_name}→{post_name})={p_backward:.2f}"
                candidates.append((sign_id, joint_prob, explanation))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:3]

# ============================================
# MODULE 5: NETWORK VISUALIZER
# ============================================

class IndusNetworkVisualizer:
    """Network analysis of sign transitions"""
    
    def __init__(self, corpus_analyzer: CorpusAnalyzer):
        self.analyzer = corpus_analyzer
        self.graph = nx.DiGraph()
        self._build_graph()
        
    def _build_graph(self):
        for (source, target), weight in self.analyzer.bigram_counts.items():
            self.graph.add_edge(source, target, weight=weight)
    
    def analyze_network_properties(self) -> Dict:
        """Calculate network metrics (Centrality, Sinks)"""
        in_degree = dict(self.graph.in_degree(weight='weight'))
        out_degree = dict(self.graph.out_degree(weight='weight'))
        
        # Identify Terminals (High In, Low Out)
        terminals = []
        for node in self.graph.nodes():
            i_d = in_degree.get(node, 0)
            o_d = out_degree.get(node, 0)
            if i_d > o_d: terminals.append((node, i_d, o_d))
        
        terminals.sort(key=lambda x: x[1], reverse=True)
        return {"terminals": terminals[:5], "density": nx.density(self.graph)}

# ============================================
# MODULE 6: WEB HELPER
# ============================================

def prepare_streamlit_app():
    """Exports data for the Streamlit UI"""
    db = IndusSignDatabase()
    options = []
    for sid, info in db.SIGN_CATALOG.items():
        options.append({
            "id": sid,
            "name": info["name"],
            "role": info["role"],
            "description": info["description"],
            "frequency": info["frequency"]
        })
    options.sort(key=lambda x: x["frequency"], reverse=True)
    return {"sign_catalog": options, "rules": db.GRAMMAR_RULES}

# ============================================
# EXECUTION (If run directly)
# ============================================

if __name__ == "__main__":
    print("ISAGA 2.0 | Engine Online")
    
    # 1. Train
    analyzer = CorpusAnalyzer()
    training_data = [
        [59, 99, 342],    # FISH -> ARROW -> JAR
        [211, 99, 342],   # WHEEL -> ARROW -> JAR
        [123, 456, 342],  # MAN -> UNICORN -> JAR
        [59, 789, 342],   # FISH -> STROKE -> JAR
    ]
    for seq in training_data:
        analyzer.add_inscription(IndusInscription(seq))
    
    # 2. Predict Broken Seal
    print("\n[TEST] Predicting Broken Seal: [WHEEL] -> [?] -> [JAR]")
    engine = PredictiveRepairEngine(analyzer)
    broken_seq = [211, None, 342]
    predictions = engine.predict_missing_sign(broken_seq, 1)
    
    for rank, (sid, conf, log) in enumerate(predictions, 1):
        name = IndusSignDatabase.get_name(sid)
        print(f"{rank}. {name} (Conf: {conf:.4f}) | Logic: {log}")

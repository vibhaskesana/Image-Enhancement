import copy
from pathlib import Path
import sys

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.recommendation.engine import RuleBasedEngine

class CandidateGenerator:
    """
    Generates multiple candidate pipelines (Standard, Conservative, Aggressive, Minimalist)
    so the system can mathematically evaluate which produces the best result.
    """
    def __init__(self):
        self.base_engine = RuleBasedEngine(activation_threshold=0.45) # Lower threshold to catch borderline issues
        
    def generate_candidates(self, severity_profile):
        candidates = {}
        
        # 1. The Standard Pipeline (Direct output of our AI)
        standard_pipeline = self.base_engine.generate_pipeline(severity_profile)
        if not standard_pipeline:
            return {"Standard": []} # No enhancements needed
            
        candidates["Standard"] = standard_pipeline
        
        # 2. The Conservative Pipeline (15% weaker math)
        conservative = copy.deepcopy(standard_pipeline)
        for step in conservative:
            step['severity'] = max(0.1, step['severity'] * 0.85)
            # Recalculate kwargs based on lower severity
            step['kwargs'] = self._recalculate_kwargs(step['tool'], step['severity'])
        candidates["Conservative"] = conservative
        
        # 3. The Aggressive Pipeline (15% stronger math)
        aggressive = copy.deepcopy(standard_pipeline)
        for step in aggressive:
            step['severity'] = min(1.0, step['severity'] * 1.15)
            step['kwargs'] = self._recalculate_kwargs(step['tool'], step['severity'])
        candidates["Aggressive"] = aggressive
        
        # 4. The Minimalist Pipeline (Drop the least severe degradation)
        if len(standard_pipeline) > 1:
            minimalist = copy.deepcopy(standard_pipeline)
            # Find the step with the lowest severity and remove it
            least_severe_step = min(minimalist, key=lambda x: x['severity'])
            minimalist.remove(least_severe_step)
            candidates["Minimalist"] = minimalist
            
        return candidates

    def _recalculate_kwargs(self, tool, severity):
        """Helper to recalculate mathematical parameters based on modified severity"""
        if tool == 'denoise':
            d = int(5 + (severity - 0.5) * 12)
            return {'d': d, 'sigma_color': 75, 'sigma_space': 75}
        elif tool == 'gamma_correction':
            gamma = max(0.2, 0.8 - (severity - 0.5) * 1.0)
            return {'gamma': gamma}
        elif tool == 'clahe':
            clip = 2.0 + (severity - 0.5) * 4.0
            return {'clip_limit': clip}
        elif tool == 'sharpen':
            amount = 1.0 + (severity - 0.5) * 4.0
            return {'amount': amount}
        elif tool == 'exposure_correction':
            suppress = max(0.3, 0.8 - (severity - 0.5) * 0.8)
            return {'highlight_suppression': suppress}
        return {}

if __name__ == "__main__":
    generator = CandidateGenerator()
    
    # Fake profile: High noise, borderline dark
    profile = {
        "low_brightness": 0.55,
        "low_contrast": 0.10,
        "noise": 0.80,
        "blur": 0.10,
        "overexposure": 0.00
    }
    
    candidates = generator.generate_candidates(profile)
    
    print("=== AI CANDIDATE PIPELINES GENERATED ===")
    for name, pipeline in candidates.items():
        tools = [step['tool'] for step in pipeline]
        print(f"{name:15}: {' -> '.join(tools) if tools else 'None'}")
        
    print("\nDeep dive into 'Gamma' parameters for this image:")
    print(f"Conservative Gamma : {candidates['Conservative'][1]['kwargs']['gamma']:.3f} (Lighter edit)")
    print(f"Standard Gamma     : {candidates['Standard'][1]['kwargs']['gamma']:.3f} (Standard edit)")
    print(f"Aggressive Gamma   : {candidates['Aggressive'][1]['kwargs']['gamma']:.3f} (Harsh edit)")

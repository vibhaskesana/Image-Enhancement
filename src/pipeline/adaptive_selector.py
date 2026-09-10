import cv2
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.recommendation.engine import RuleBasedEngine
from src.recommendation.candidate_generator import CandidateGenerator
from src.evaluation.quality import evaluate_no_reference_quality

class AdaptiveSelector:
    """
    The ultimate "Judge" of the system.
    It executes all candidate pipelines and picks the one with the highest quality score.
    """
    def __init__(self):
        self.generator = CandidateGenerator()
        self.engine = RuleBasedEngine(activation_threshold=0.45)

    def process_and_select(self, image_path, severity_profile):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Image not found.")

        # 1. Generate the candidate strategies based on the AI's diagnosis
        candidates = self.generator.generate_candidates(severity_profile)

        # Add the "Original" image as a baseline candidate
        # (If all enhancements make the picture worse, just return the original)
        candidates["Original (Baseline)"] = []

        results = {}
        best_score = -999
        best_candidate = None
        best_image = None

        # 2. Execute and Grade every single candidate
        for name, pipeline in candidates.items():
            processed_img, pipeline_str = self.engine.apply_pipeline(image, pipeline)
            ebs, sharp, contrast, artifacts = evaluate_no_reference_quality(processed_img)

            results[name] = {
                "pipeline": pipeline_str,
                "ebs_score": ebs,
                "sharpness": sharp,
                "contrast": contrast,
                "artifacts": artifacts
            }

            # 3. Pick the winner
            if ebs > best_score:
                best_score = ebs
                best_candidate = name
                best_image = processed_img

        return best_image, best_candidate, results


if __name__ == "__main__":
    selector = AdaptiveSelector()

    test_img = BASE_DIR / "data" / "processed" / "multi_label_dataset" / "images" / "sample_0_deg_2.jpg"

    fake_profile = {
        "low_brightness": 0.85,
        "low_contrast": 0.30,
        "noise": 0.75,
        "blur": 0.10,
        "overexposure": 0.00
    }

    if test_img.exists():
        best_img, best_name, results = selector.process_and_select(test_img, fake_profile)
        print(f"WINNER: {best_name}")
        for name, info in results.items():
            print(f"  [{name}] Score: {info['ebs_score']} | {info['pipeline']}")
